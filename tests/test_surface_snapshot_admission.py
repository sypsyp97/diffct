"""Snapshot placement follows admitted devices and common collective order."""

from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
import torch.distributed as dist

from diffct import Projector, operators
from tests.test_detector_surfaces import _arc_offsets, _case, _close, _data, _matrix, _numpy, _projector
from tests.test_trajectory_surface_integration import _run_two, _start_group


class _CudaValue(torch.Tensor):
    """CPU storage with explicit CUDA placement/copy metadata for admission."""

    @staticmethod
    def wrap(value, device, copies):
        result = value.as_subclass(_CudaValue)
        result._device, result._copies = torch.device(device), copies
        return result

    @property
    def device(self):
        return getattr(self, "_device", torch.device("cpu"))

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    def detach(self):
        return self.wrap(self.as_subclass(torch.Tensor).detach(), self.device, self._copies)

    def clone(self, *args, **kwargs):
        value = self.as_subclass(torch.Tensor).clone(*args, **kwargs)
        self._copies.append(("clone", self.device, self.device, value.nbytes))
        return self.wrap(value, self.device, self._copies)

    def to(self, *args, **kwargs):
        destination = torch.device(kwargs.pop("device", self.device))
        if args and isinstance(args[0], (str, torch.device)):
            destination, args = torch.device(args[0]), args[1:]
        value = self.as_subclass(torch.Tensor).to(*args, device="cpu", **kwargs)
        self._copies.append(("to", self.device, destination, value.nbytes))
        return value if destination.type == "cpu" else self.wrap(value, destination, self._copies)


@contextmanager
def _metadata_inventory(budgets):
    unlimited = 1 << 40

    def budget(device):
        return budgets.get(torch.device(device), unlimited)

    with ExitStack() as patches:
        patches.enter_context(mock.patch.object(torch.cuda, "is_available", return_value=True))
        patches.enter_context(mock.patch.object(torch.cuda, "current_device", return_value=0))
        patches.enter_context(mock.patch.object(torch.cuda, "get_allocator_backend", return_value="cudaMallocAsync"))
        patches.enter_context(mock.patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("metadata admission initialized CUDA")))
        patches.enter_context(mock.patch.object(operators, "_memory_lower_bounds", side_effect=lambda device: (budget(device), unlimited, unlimited)))
        patches.enter_context(mock.patch.object(operators, "_memory_budget", side_effect=lambda device: (budget(device), unlimited)))
        patches.enter_context(mock.patch.object(operators, "_reservation_budget", return_value=unlimited))
        yield


@pytest.mark.parametrize("is_project", (True, False), ids=("project", "backproject"))
def test_distributed_surface_prepare_has_identical_collective_order_for_cpu_and_cuda_input(is_project):
    sequences = []
    for device in ("cpu", "cuda:0"):
        case = _case("cone", views=3)
        operator = _projector(case, lambda u, v: torch.stack((u, v, .03 * u.square()), -1), backend="torch")
        events = []

        def preflight(error=None, metadata=None, **kwargs):
            if error is not None:
                raise error
            events.append(kwargs.get("cache_key"))

        # Run both actual preparation branches and their actual planner and
        # snapshot methods. Only process-group transport/device inventory vary.
        operator.backend, operator._distributed, operator.world_size = "numba", True, 2
        operator._devices = (torch.device("cuda:0"),)
        operator._partition = SimpleNamespace(preflight=preflight)
        data = torch.zeros(case.shape if is_project else case.sino_shape)
        if device != "cpu":
            data = _CudaValue.wrap(data, device, [])
        with _metadata_inventory({}), mock.patch.object(dist, "get_backend", return_value="gloo"), \
                mock.patch.object(dist, "all_reduce", side_effect=lambda *args, **kwargs: events.append("budget reduction")):
            operator._prepare_execution(data, is_project)
        assert set(events) >= {"geometry layout", "surface snapshot", "native chunks", "budget reduction"}
        sequences.append(events)
    assert sequences[0] == sequences[1], (
        f"mixed CPU/CUDA ranks enter different collectives: CPU={sequences[0]}, CUDA={sequences[1]}"
    )


class _SaveForeignTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, offsets, parameter):
        ctx.save_for_backward(parameter)
        return offsets + parameter.as_subclass(torch.Tensor).square()

    @staticmethod
    def backward(ctx, cotangent):
        parameter, = ctx.saved_tensors
        return cotangent, 2 * parameter * cotangent


@pytest.mark.parametrize("placement", ("foreign-input", "unselected-offset", "unselected-frame", "unselected-saved-tensor"))
def test_full_surface_snapshot_does_not_clone_on_unbudgeted_cuda_device(placement):
    case = _case("cone", views=3)
    case.shape, case.detector, case.pitch = (3, 4, 5), (4, 47), (.52, .08)
    offsets = torch.from_numpy(_arc_offsets(case, np.array([3.3, 3.7, 4.1]), .037))
    operator = _projector(case, lambda u, v: offsets, backend="torch")
    operator.backend = "numba"
    compute = torch.device("cuda:1" if placement == "foreign-input" else "cuda:0")
    foreign = torch.device("cuda:0" if placement == "foreign-input" else "cuda:2")
    operator._devices = (compute,)
    copies = []
    source = _CudaValue.wrap(torch.ones(case.shape), "cuda:0", copies)
    expected_index = None
    if placement in ("foreign-input", "unselected-offset"):
        original = _CudaValue.wrap(offsets, foreign, copies)
        operator.detector_surface = lambda u, v: original
        expected_index = len(case.trajectory)
    elif placement == "unselected-frame":
        original = _CudaValue.wrap(operator._trajectory[1], foreign, copies)
        operator._trajectory = (operator._trajectory[0], original, *operator._trajectory[2:])
        expected_index = 1
    else:
        original = _CudaValue.wrap(torch.full_like(offsets, .01).requires_grad_(), foreign, copies)
        operator.detector_surface = lambda u, v: _SaveForeignTensor.apply(offsets, original)
    # 6000 bytes admits the 4992-byte resident data/result allowance on a
    # foreign input device, but cannot admit this 13536-byte offset snapshot.
    assert offsets.nbytes == 13536 and 8 * (np.prod(case.shape) + np.prod(case.sino_shape)) < 6000
    with _metadata_inventory({foreign: 6000}):
        plan, spec, snapshots = operator._prepare_execution(source, True)
    assert spec.kind == "offsets" and snapshots
    unbudgeted = [record for record in copies if record[2] == foreign]
    assert not unbudgeted, (
        f"surface preparation cloned on {foreign} outside admitted snapshot storage: {unbudgeted}; plan={plan}"
    )
    if expected_index is not None:
        snapshot = snapshots[expected_index]
        assert snapshot.device.type == "cpu"
        before = original.as_subclass(torch.Tensor).clone()
        assert snapshot.untyped_storage()._cdata != original.untyped_storage()._cdata
        with torch.no_grad():
            original.add_(.125)
        torch.testing.assert_close(snapshot, before, rtol=0, atol=0)


def _mixed_device_worker(rank, rendezvous, directory, operation):
    _start_group(rank, rendezvous, "gloo", timeout=12)
    record = {"error": None}
    try:
        case = _case("cone", views=4)
        case.shape, case.detector = (2, 3, 4), (3, 3)
        offsets = torch.from_numpy(_arc_offsets(case, 3.7, .037))
        enabled = [True]

        def surface(u, v):
            return offsets if enabled[0] else torch.full_like(offsets, float("nan"))

        operator = _projector(case, surface, backend="numba", devices=[0], distributed=True)
        image, rays = _data(case, device="cpu")
        matrix = _matrix(case, _arc_offsets(case, 3.7, .037))
        projecting = operation.startswith("project")
        owned = operator.view_slice
        source = (image if projecting else rays[owned]).to("cpu" if rank == 0 else "cuda:0")
        expected = ((matrix @ _numpy(image).ravel()).reshape(case.sino_shape)[owned] if projecting
                    else (matrix.T @ _numpy(rays).ravel()).reshape(case.shape))
        output = torch.full(operator.projection_shape if projecting else operator.volume_shape,
                            73., dtype=torch.float32, device=source.device)
        try:
            with torch.no_grad():
                result = (getattr(operator, operation)(source, output) if operation.endswith("_into")
                          else getattr(operator, operation)(source))
            _close(result, expected)
            assert result.device == source.device
            if operation.endswith("_into"):
                assert result is output
                enabled[0] = False
                output.fill_(73.)
                with pytest.raises(ValueError):
                    getattr(operator, operation)(source, output)
                torch.testing.assert_close(output, torch.full_like(output, 73.), rtol=0, atol=0)
            record["numerical_match"] = True
        except (ValueError, RuntimeError, AssertionError) as error:
            record["error"] = f"{type(error).__name__}: {error}"
        Path(directory, f"rank-{rank}.json").write_text(json.dumps(record), encoding="utf-8")
    finally:
        dist.destroy_process_group()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None
                    or not dist.is_available() or not dist.is_gloo_available(),
                    reason="one NVIDIA GPU and Gloo are required")
@pytest.mark.parametrize("operation", ("project", "backproject", "project_into", "backproject_into"))
def test_gloo_legacy_surface_accepts_mixed_cpu_cuda_rank_inputs(tmp_path, operation):
    _run_two(_mixed_device_worker, tmp_path, operation, seconds=55)
    records = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert all(record["error"] is None and record.get("numerical_match") for record in records), records
