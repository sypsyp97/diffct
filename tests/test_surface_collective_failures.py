"""Distributed surface requests fail together before ray kernels or reductions."""

import json
import time
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffct import ParameterizedSurface, Projector, operators
from diffct.chunking import _ChunkPlan
from tests.test_detector_surfaces import (
    _arc_offsets,
    _case,
    _finite_difference,
    _matrix,
    _numpy,
)
from tests.test_distributed_training_transforms import (
    _check as _check_distributed_data_transform,
)

GROUP_TIMEOUT = 5
GLOO = pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(),
                         reason="collective surface validation requires Gloo")


def _run_two(worker, tmp_path, *args):
    rendezvous = (tmp_path / "rendezvous").resolve().as_uri()
    context = mp.spawn(worker, args=(rendezvous, str(tmp_path), *args), nprocs=2, join=False)
    deadline = time.monotonic() + 50
    try:
        while not context.join(timeout=1):
            assert time.monotonic() < deadline, "surface workers exceeded 50 seconds"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    return [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]


def _record(directory, rank, records):
    root = Path(directory)
    (root / f"rank-{rank}.json").write_text(json.dumps(records), encoding="utf-8")
    # Do not let teardown replace a peer's validation error with disconnection.
    deadline = time.monotonic() + 2 * GROUP_TIMEOUT + 5
    while not all((root / f"rank-{peer}.json").exists() for peer in range(2)):
        assert time.monotonic() < deadline, "peer did not finish its bounded surface request"
        time.sleep(.01)


def _init(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=GROUP_TIMEOUT))


def _attempt(function, calls):
    started = time.monotonic()
    try:
        function()
    except (TypeError, ValueError, RuntimeError) as error:
        return {"error": type(error).__name__, "message": str(error), "calls": list(calls),
                "elapsed": time.monotonic() - started}
    return {"error": None, "message": None, "calls": list(calls),
            "elapsed": time.monotonic() - started}


def _assert_collective_rejection(records):
    assert len(records[0]) == len(records[1])
    for left, right in zip(*records):
        assert left["error"] == right["error"] == "ValueError", (left, right)
        assert left["message"] == right["message"], (left, right)
        for record in (left, right):
            assert not record["calls"], ("invalid surface reached numerical execution", record)
            assert record["elapsed"] < GROUP_TIMEOUT, ("validation waited for transport timeout", record)


class _CpuStream:
    def wait_event(self, event):
        pass

    def wait_stream(self, stream):
        pass

    def synchronize(self):
        pass


class _CpuEvent:
    def synchronize(self):
        pass


class _CpuGpus:
    def __getitem__(self, index):
        return nullcontext()


class _CpuNativeTensor(torch.Tensor):
    @property
    def is_cuda(self):
        return True


class _CpuTransfer:
    """Replace transport only; real native geometry staging and dispatch run."""

    def __init__(self, devices, elements, stats, slots=1):
        self.device = devices[0]
        self.stats = stats
        self.capacity = 0
        self.host = [None]
        self.compute = self.upload = self.download = _CpuStream()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def for_device(self, device):
        return self

    def reclaim_output(self, *args, **kwargs):
        pass

    def empty(self, shape):
        return torch.empty(shape, dtype=torch.float32)

    def event(self, stream):
        return _CpuEvent()

    def upload_tensor(self, source, target):
        target.copy_(source)
        return _CpuEvent()

    def upload_store(self, source, selected, target):
        target.copy_(source.read(selected))
        return _CpuEvent()


def _cpu_surface_geometry_into(operator, geometry, spec, views, pixels, centre,
                               targets, pipe, groups, offsets=None):
    """Replace fused device writes with real CPU geometry for collective tests."""
    frame_count = 4 if operator.beam == "cone" else 3
    if offsets is None:
        offsets = operators._sample_offsets(operator.beam, operator.detector_shape,
                                             operator.detector_spacing, geometry, spec, views, pixels)
    local = (*(value[views] for value in geometry[:frame_count]), offsets)
    source, points = operators._sample_geometry(
        operator.beam, operator.detector_shape, operator.detector_spacing, local,
        operators._GeometrySpec("offsets"), views, pixels, operator.voxel_spacing,
        local_geometry=True, native_legacy=spec.kind == "offsets",
    )
    for index, value in enumerate((source, points)):
        if index == 1 or operator.beam != "parallel":
            value = value - value.new_tensor(centre)
        targets[index].copy_(value.reshape_as(targets[index]))
    return _CpuEvent()


@contextmanager
def _native_cpu():
    """Keep admission and ray dispatch real; replace CUDA transport and writes."""
    import numba

    with mock.patch.object(operators, "select_backend", return_value="numba"), \
            mock.patch.object(operators, "_normalize_devices", return_value=(torch.device("cpu"),)), \
            mock.patch.object(torch.cuda, "device", side_effect=lambda *args: nullcontext()), \
            mock.patch.object(torch.cuda, "stream", side_effect=lambda *args: nullcontext()), \
            mock.patch.object(torch.cuda, "current_stream", return_value=_CpuStream()), \
            mock.patch.object(torch.cuda, "get_allocator_backend", return_value="native"), \
            mock.patch.object(operators, "_memory_lower_bounds", return_value=(1 << 30, 1 << 30, 1 << 30)), \
            mock.patch.object(operators, "_memory_budget", return_value=(1 << 30, 1 << 30)), \
            mock.patch.object(operators, "_reservation_budget", return_value=1 << 30), \
            mock.patch.object(torch.Tensor, "record_stream", return_value=None), \
            mock.patch.object(numba, "cuda", SimpleNamespace(gpus=_CpuGpus()), create=True), \
            mock.patch.object(operators._pipeline, "_TransferPipeline", _CpuTransfer), \
            mock.patch.object(Projector, "_surface_geometry_into", _cpu_surface_geometry_into), \
            mock.patch.object(operators, "_prepare_volume", side_effect=lambda beam, value: value):
        yield


@contextmanager
def _numerical_calls(operator, calls):
    original_reduce = dist.all_reduce

    def reduce(value, *args, **kwargs):
        if torch.is_floating_point(value):
            calls.append("floating all_reduce")
            raise RuntimeError("numerical reduction started before validation")
        return original_reduce(value, *args, **kwargs)

    def kernel(*args, **kwargs):
        calls.append("ray kernel")
        raise RuntimeError("ray kernel started before validation")

    with mock.patch.object(dist, "all_reduce", side_effect=reduce), \
            mock.patch.object(operator, "_project_raw", side_effect=kernel), \
            mock.patch.object(operator, "_backproject_raw", side_effect=kernel), \
            mock.patch.object(operators, "_project_into", side_effect=kernel), \
            mock.patch.object(operators, "_backproject_into", side_effect=kernel):
        if operator._reference is None:
            yield
        else:
            with mock.patch.object(operator._reference, "_intervals", side_effect=kernel):
                yield


def _simple_problem(views=2):
    source = torch.tensor([[-9., 0.]], dtype=torch.float64).repeat(views, 1).requires_grad_()
    center = torch.tensor([[6., 0.]], dtype=torch.float64).repeat(views, 1).requires_grad_()
    axis = torch.tensor([[0., 1.]], dtype=torch.float64).repeat(views, 1).requires_grad_()
    return source, center, axis


def _operator(trajectory, surface, backend, **kwargs):
    return Projector(trajectory, (2, 3), (3,), beam="fan", detector_surface=surface,
                     backend=backend, precision="fast" if backend == "numba" else "float64",
                     distributed=True, schedule="spatial", **kwargs)


def _invoke(operator, operation, mode):
    is_project = operation.startswith("project")
    shape = operator.volume_shape if is_project else operator.projection_shape
    output_shape = operator.projection_shape if is_project else operator.volume_shape
    value = torch.ones(shape, dtype=torch.float32)
    if mode == "full":
        value = value.as_subclass(_CpuNativeTensor)
    plan = (None if mode == "full" else
            _ChunkPlan((1, 2), 1, (torch.device("cpu"),), (1,), gpu_budget_bytes=1 << 30))
    with mock.patch.object(operator, "_execution_plan", return_value=plan):
        if operation.endswith("_into"):
            return getattr(operator, operation)(value, torch.full(output_shape, 37., dtype=operator.dtype))
        return getattr(operator, operation)(value)


def _assert_valid_request_reaches_kernel(operator, operation, mode):
    calls = []
    with _numerical_calls(operator, calls):
        record = _attempt(partial(_invoke, operator, operation, mode), calls)
    assert record["error"] == "RuntimeError" and calls == ["ray kernel"], record


def _layout_worker(rank, rendezvous, directory, backend, mismatch, phase):
    _init(rank, rendezvous)
    records = []
    try:
        state = {"changed": phase == "constructor"}
        parameter = torch.tensor(.037, dtype=torch.float64, requires_grad=True)

        def surface(u, v):
            offsets = torch.stack((u + parameter * u.square(), v, torch.zeros_like(u)), dim=-1)
            if rank == 1 and state["changed"]:
                if mismatch == "shape":
                    offsets = offsets.unsqueeze(0).expand(2, -1, -1)
                else:
                    offsets = offsets.float()
            return offsets

        with _native_cpu() if backend == "numba" else nullcontext():
            if phase == "constructor":
                records.append(_attempt(lambda: _operator(_simple_problem(), surface, backend), []))
            else:
                operator = _operator(_simple_problem(), surface, backend)
                modes = ("full", "streamed") if backend == "numba" else ("portable",)
                for mode in modes:
                    operations = ("project", "backproject", "project_into", "backproject_into")
                    for operation in operations:
                        if mode == "full" and operation.endswith("_into"):
                            continue
                        state["changed"] = False
                        _assert_valid_request_reaches_kernel(operator, operation, mode)
                        state["changed"] = True
                        calls = []
                        with _numerical_calls(operator, calls):
                            records.append(_attempt(partial(_invoke, operator, operation, mode), calls))
        _record(directory, rank, records)
    finally:
        dist.destroy_process_group()


@GLOO
@pytest.mark.parametrize("backend", ["torch", "numba"])
@pytest.mark.parametrize("mismatch", ["shape", "dtype"])
@pytest.mark.parametrize("phase", ["constructor", "invocation"])
def test_legacy_surface_actual_sample_layout_must_agree_on_every_rank(
    tmp_path, backend, mismatch, phase,
):
    records = _run_two(_layout_worker, tmp_path, backend, mismatch, phase)
    _assert_collective_rejection(records)


def _mutation_worker(rank, rendezvous, directory, kind, failure):
    _init(rank, rendezvous)
    records = []
    try:
        for mode in ("full", "streamed"):
            for operation in ("project", "backproject", "project_into", "backproject_into"):
                if mode == "full" and operation.endswith("_into"):
                    continue
                trajectory = _simple_problem()
                displacement = torch.zeros(2, dtype=torch.float64, requires_grad=True)
                state = {"changed": False}

                def offsets(u, v, ids, parameter, state=state):
                    value = parameter[ids, None] + torch.zeros_like(u)
                    result = torch.stack((u.expand_as(value), torch.zeros_like(value), value), dim=-1)
                    if state["changed"] and 1 in ids.tolist():
                        if failure == "shape":
                            return result[..., :2]
                        if failure == "nonfinite":
                            result = result * float("nan")
                        if failure == "middle":
                            result = result.clone()
                            result[..., 1] = 1.
                    return result

                if kind == "legacy":
                    def surface(u, v, offsets=offsets, displacement=displacement):
                        return offsets(u, v, torch.arange(2), displacement)
                else:
                    surface = ParameterizedSurface(offsets, (displacement,))
                with _native_cpu():
                    operator = _operator(trajectory, surface, "numba")
                    _assert_valid_request_reaches_kernel(operator, operation, mode)
                    with torch.no_grad():
                        if failure == "trajectory":
                            trajectory[0][1, 0] = 1_000_000.01
                            trajectory[1][1, 0] = 1_000_002.01
                        elif failure == "coincident":
                            displacement[1] = -15.
                    state["changed"] = True
                    calls = []
                    with _numerical_calls(operator, calls):
                        records.append(_attempt(partial(_invoke, operator, operation, mode), calls))
        _record(directory, rank, records)
    finally:
        dist.destroy_process_group()


@GLOO
@pytest.mark.parametrize("kind,failure", [
    ("legacy", "trajectory"), ("parameterized", "trajectory"),
    ("legacy", "coincident"), ("parameterized", "coincident"),
    ("parameterized", "shape"), ("parameterized", "nonfinite"),
    ("parameterized", "middle"),
])
def test_native_live_surface_validation_reaches_every_rank_before_full_or_streamed_numerics(
    tmp_path, kind, failure,
):
    records = _run_two(_mutation_worker, tmp_path, kind, failure)
    _assert_collective_rejection(records)


def _valid_worker(rank, rendezvous, directory, partition):
    _init(rank, rendezvous)
    try:
        for views in (1, 3):
            for kind in ("legacy", "parameterized"):
                for layout in ("shared", "per-view"):
                    case = _case("fan", views=views)
                    if partition == "space":
                        case.shape = (1, case.shape[1])
                    parameter = torch.full(() if layout == "shared" else (views,), .037,
                                           dtype=torch.float64, requires_grad=True)
                    state = {"expand": False}

                    def sampler(u, v, ids, value):
                        coefficient = value if value.ndim == 0 else value[ids, None]
                        x = 3.7 * torch.sin(u / 3.7) + coefficient * u.square()
                        return torch.stack((x, v.expand_as(x),
                                            (3.7 * (1 - torch.cos(u / 3.7))).expand_as(x)), dim=-1)

                    if kind == "legacy":
                        def surface(u, v, views=views, parameter=parameter, state=state):
                            offsets = sampler(u, v, torch.arange(views), parameter)
                            return offsets.expand(views, -1, -1) if state["expand"] else offsets
                    else:
                        surface = ParameterizedSurface(sampler, (parameter,))
                    operator = Projector(case.trajectory, case.shape, case.detector, beam="fan",
                                         detector_spacing=case.pitch[0], voxel_spacing=case.spacing,
                                         backend="torch", precision="float64", detector_surface=surface,
                                         distributed=True, partition=partition, view_chunk_size=1,
                                         volume_chunk_shape=(1, 3), detector_chunk_shape=(2,))
                    generator = torch.Generator().manual_seed(716)
                    image = torch.randn(case.shape, dtype=torch.float64, generator=generator)
                    rays = torch.randn(case.sino_shape, dtype=torch.float64, generator=generator)
                    coefficient = _numpy(parameter)
                    reference_radius = 3.7 if layout == "shared" else np.full(views, 3.7)
                    matrix = _matrix(case, _arc_offsets(case, reference_radius, coefficient))

                    def objective(value, case=case, rays=rays, image=image, radius=reference_radius):
                        return float(_numpy(rays).ravel() @
                                     _matrix(case, _arc_offsets(case, radius, value)) @ _numpy(image).ravel())

                    if layout == "shared":
                        expected_gradient = np.asarray(_finite_difference(objective, float(coefficient)))
                    else:
                        expected_gradient = np.asarray([
                            _finite_difference(objective, coefficient, direction)
                            for direction in np.eye(views)
                        ])
                    assert np.max(np.abs(expected_gradient)) > 1e-6
                    for operation in ("project", "backproject"):
                        # A legal callback layout change is allowed when all ranks agree.
                        state["expand"] = kind == "legacy" and layout == "shared" and operation == "project"
                        parameter.grad = None
                        if operation == "project":
                            value = image[operator.volume_slice].clone().requires_grad_()
                            weight = rays[operator.view_slice]
                            expected = (matrix @ _numpy(image).ravel()).reshape(case.sino_shape)[operator.view_slice]
                            expected_data = (matrix.T @ _numpy(rays).ravel()).reshape(case.shape)[operator.volume_slice]
                            replicas = operator.space_world_size
                        else:
                            value = rays[operator.view_slice].clone().requires_grad_()
                            weight = image[operator.volume_slice]
                            expected = (matrix.T @ _numpy(rays).ravel()).reshape(case.shape)[operator.volume_slice]
                            expected_data = (matrix @ _numpy(image).ravel()).reshape(case.sino_shape)[operator.view_slice]
                            replicas = operator.view_world_size
                        actual = getattr(operator, operation)(value)
                        torch.testing.assert_close(actual, torch.from_numpy(expected), rtol=3e-11, atol=3e-12)
                        ((actual * weight).sum() / replicas).backward()
                        torch.testing.assert_close(value.grad, torch.from_numpy(expected_data), rtol=3e-11, atol=3e-12)
                        torch.testing.assert_close(parameter.grad, torch.from_numpy(expected_gradient), rtol=4e-5, atol=4e-7)
        _record(directory, rank, [{"valid": True}])
    finally:
        dist.destroy_process_group()


@GLOO
@pytest.mark.parametrize("partition", ["views", "space"])
def test_valid_shared_and_global_surface_gradients_keep_view_and_space_empty_ranks(partition, tmp_path):
    records = _run_two(_valid_worker, tmp_path, partition)
    assert all(record == [{"valid": True}] for record in records)


def _surface_data_transform(operator, transform, value):
    function = operator.project

    def linear_loss(data):
        return function(data).sum() / operator.space_world_size

    def quadratic_loss(data):
        return function(data).square().sum() / (2 * operator.space_world_size)

    if transform == "grad":
        return torch.func.grad(linear_loss)(value)
    if transform == "jvp":
        return torch.func.jvp(function, (value,), (torch.ones_like(value),))
    if transform == "vmap":
        return torch.func.vmap(function)(torch.stack((value, value)))
    assert transform == "hvp"
    return torch.func.jvp(torch.func.grad(quadratic_loss), (value,), (torch.ones_like(value),))


def _surface_transform_worker(rank, rendezvous, directory):
    _init(rank, rendezvous)
    records = []
    try:
        case = _case("fan", views=3)
        case.shape, case.detector = (2, 3), (3,)
        contract_case = SimpleNamespace(shape=case.shape, projection_shape=case.sino_shape,
                                        detector=case.detector)
        matrix = torch.from_numpy(_matrix(case, _arc_offsets(case, 3.7, .037)))

        def curved_offsets(u, v):
            return torch.stack((3.7 * torch.sin(u / 3.7) + .037 * u.square(), v,
                                3.7 * (1 - torch.cos(u / 3.7))), dim=-1)

        def make_operator(surface):
            return Projector(case.trajectory, case.shape, case.detector, beam="fan",
                             detector_spacing=case.pitch[0], voxel_spacing=case.spacing,
                             detector_surface=surface, backend="torch", precision="float64",
                             distributed=True, view_chunk_size=1, volume_chunk_shape=(1, 2),
                             detector_chunk_shape=(2,))

        for kind in ("legacy", "parameterized"):
            surface = (curved_offsets if kind == "legacy" else
                       ParameterizedSurface(lambda u, v, ids: curved_offsets(u, v)))
            operator = make_operator(surface)
            for operation in ("project", "backproject"):
                for transform in ("grad", "jvp", "vmap", "hvp"):
                    _check_distributed_data_transform(operator, contract_case, matrix, operation,
                                                      transform, "cpu")

        for mismatch in ("shape", "dtype"):
            state = {"changed": False}

            def surface(u, v, state=state, mismatch=mismatch):
                offsets = curved_offsets(u, v)
                if state["changed"] and rank == 1:
                    return offsets.expand(3, -1, -1) if mismatch == "shape" else offsets.float()
                return offsets

            operator = make_operator(surface)
            state["changed"] = True
            value = torch.ones(operator.volume_shape, dtype=operator.dtype)
            for transform in ("grad", "jvp", "vmap", "hvp"):
                calls = []
                with _numerical_calls(operator, calls):
                    records.append(_attempt(partial(_surface_data_transform, operator, transform, value), calls))
        _record(directory, rank, records)
    finally:
        dist.destroy_process_group()


@GLOO
def test_surface_controls_preserve_distributed_data_transforms_and_collective_layout_rejection(tmp_path):
    records = _run_two(_surface_transform_worker, tmp_path)
    _assert_collective_rejection(records)
