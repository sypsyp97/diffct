"""Per-call detector frames preserve surface, collective, and staging contracts.

The numerical reference intersects each voxel independently with each ray.
It does not call a DiffCT geometry helper or a production projector.
"""

import json
import math
import time
from collections import Counter
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
import torch.distributed as dist
from tests._multiprocessing import spawn
from torch.utils._python_dispatch import TorchDispatchMode

from diffct import ParameterizedSurface, Projector, operators
from diffct.backends import ReferenceProjector

_OPERATIONS = ("project", "backproject", "__call__", "project_into", "backproject_into")
_GROUP_TIMEOUT = 6
_GLOO = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is required",
)
_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA is required",
)
_NCCL = pytest.mark.skipif(
    not dist.is_available() or not dist.is_nccl_available() or torch.cuda.device_count() < 2,
    reason="two NVIDIA CUDA devices and NCCL are required",
)


def _frames(views=5, *, three_dimensional=False):
    angles = .287 + torch.arange(views, dtype=torch.float64) * .617
    radial = torch.stack((angles.cos(), angles.sin()), -1)
    tangent = torch.stack((-angles.sin(), angles.cos()), -1)
    center = torch.tensor([.193, -.157], dtype=torch.float64).expand(views, -1)
    if not three_dimensional:
        return -8.3 * radial + center, 5.7 * radial + center, tangent
    radial = torch.cat((radial, torch.zeros((views, 1), dtype=torch.float64)), -1)
    tangent = torch.cat((tangent, torch.zeros((views, 1), dtype=torch.float64)), -1)
    center = torch.cat((center, torch.linspace(-.27, .19, views)[:, None]), -1)
    vertical = torch.tensor([0., 0., 1.], dtype=torch.float64).expand(views, -1)
    return -8.3 * radial + center, 5.7 * radial + center, tangent, vertical


def _override(frames):
    result = tuple(value.clone() for value in frames)
    result[1][:, 0].add_(.083)
    result[1][:, 1].sub_(.041)
    return result


def _offsets(u, v, coefficient):
    return torch.stack((3.9 * torch.sin(u / 3.9) + coefficient * u.square(),
                        v, 3.9 * (1 - torch.cos(u / 3.9))), -1)


def _ray_matrix(frames, coefficient, shape=(4, 5), detector=7, spacing=.71, pitch=.83):
    """Intersect each 2D voxel box with each source-to-pixel segment."""
    source, center, axis = (value.detach().cpu().double().numpy() for value in frames)
    rows = []
    for view in range(len(source)):
        for pixel in range(detector):
            u = (pixel + .5 - detector / 2) * pitch
            tangent = 3.9 * np.sin(u / 3.9) + coefficient * u * u
            depth = 3.9 * (1 - np.cos(u / 3.9))
            point = center[view] + tangent * axis[view] + depth * np.array(
                [axis[view, 1], -axis[view, 0]],
            )
            direction = point - source[view]
            lengths = []
            for y, x in np.ndindex(shape):
                lower = (np.array([x, y]) - np.array(shape[::-1]) / 2) * spacing
                upper = lower + spacing
                enter, leave = 0., 1.
                for dimension in range(2):
                    if direction[dimension] == 0:
                        if not lower[dimension] <= source[view, dimension] < upper[dimension]:
                            leave = -1.
                    else:
                        first = (lower[dimension] - source[view, dimension]) / direction[dimension]
                        last = (upper[dimension] - source[view, dimension]) / direction[dimension]
                        enter = max(enter, min(first, last))
                        leave = min(leave, max(first, last))
                lengths.append(max(leave - enter, 0.) * np.linalg.norm(direction))
            rows.append(lengths)
    return torch.tensor(np.asarray(rows), dtype=torch.float64)


def _data(shape, phase=0.):
    index = torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape)
    return (.31 + torch.sin(index * .73 + phase) + .29 * torch.cos(index * .37 - phase))


def _operator(frames, surface, *, backend="torch", distributed=False, **kwargs):
    return Projector(
        frames, (4, 5), (7,), beam="fan", detector_spacing=.83, voxel_spacing=.71,
        detector_surface=surface, backend=backend,
        precision="fast" if backend == "numba" else "float64",
        distributed=distributed, **kwargs,
    )


def _finite_difference(frames, image, rays, value=.037):
    def objective(coefficient):
        return float(rays.flatten() @ _ray_matrix(frames, coefficient) @ image.flatten())

    estimates = [(objective(value + step) - objective(value - step)) / (2 * step)
                 for step in (2e-6, 1e-6)]
    assert estimates[0] == pytest.approx(estimates[1], rel=2e-5, abs=2e-7), \
        "finite-difference fixture crosses a voxel edge"
    assert abs(estimates[1]) > 1e-3, "surface gradient fixture is uninformative"
    return estimates[1]


def test_independent_segment_oracle_has_hand_computed_unit_and_diagonal_chords():
    axis = torch.tensor([[0., 1.]], dtype=torch.float64)
    horizontal = (torch.tensor([[-2., .13]], dtype=torch.float64),
                  torch.tensor([[2., .13]], dtype=torch.float64), axis)
    diagonal = (torch.tensor([[-2., -2.]], dtype=torch.float64),
                torch.tensor([[2., 2.]], dtype=torch.float64), axis)
    assert _ray_matrix(horizontal, .037, shape=(1, 1), detector=1, spacing=1.).item() == pytest.approx(1.)
    assert _ray_matrix(diagonal, .037, shape=(1, 1), detector=1, spacing=1.).item() == pytest.approx(math.sqrt(2.))


@pytest.mark.parametrize("operation", _OPERATIONS)
def test_callable_surface_override_entry_points_match_independent_ray_matrix(operation):
    frames = _frames()
    changed = _override(frames)
    operator = _operator(frames, lambda u, v: _offsets(u, v, .037), view_chunk_size=2,
                         detector_chunk_shape=(3,), volume_chunk_shape=(2, 3))
    matrix = _ray_matrix(changed, .037)
    image, rays = _data((4, 5)), _data((5, 7), .41)
    projecting = operation in ("project", "__call__", "project_into")
    source = image if projecting else rays
    expected = (matrix @ image.flatten()).reshape(5, 7) if projecting else (
        matrix.T @ rays.flatten()).reshape(4, 5)
    if operation.endswith("_into"):
        output = torch.full_like(expected, 73.)
        assert getattr(operator, operation)(source, output, trajectory=changed) is output
        actual = output
    else:
        actual = getattr(operator, operation)(source, trajectory=changed)
    torch.testing.assert_close(actual, expected, rtol=3e-11, atol=3e-12)
    original = _ray_matrix(frames, .037)
    assert (matrix - original).abs().max() > 1e-3
    torch.testing.assert_close(operator(image).flatten(), original @ image.flatten(),
                               rtol=3e-11, atol=3e-12)


@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_captured_callable_parameter_override_gradient_matches_independent_finite_difference(operation):
    coefficient = torch.tensor(.037, dtype=torch.float64, requires_grad=True)
    frames, image, rays = _override(_frames()), _data((4, 5)), _data((5, 7), .41)
    operator = _operator(_frames(), lambda u, v: _offsets(u, v, coefficient), view_chunk_size=2)
    source, weight = (image, rays) if operation == "project" else (rays, image)
    result = getattr(operator, operation)(source, trajectory=frames)
    gradient, = torch.autograd.grad((result * weight).sum(), coefficient)
    assert torch.isfinite(gradient)
    assert float(gradient) == pytest.approx(_finite_difference(frames, image, rays), rel=4e-5, abs=5e-7)


@contextmanager
def _forbid_numerics(calls):
    def kernel(*args, **kwargs):
        calls.append("ray kernel")
        raise AssertionError("invalid override reached a ray kernel")

    original_reduce = dist.all_reduce

    def reduce(value, *args, **kwargs):
        if torch.is_floating_point(value):
            calls.append("floating all_reduce")
            raise AssertionError("invalid override reached a numerical reduction")
        return original_reduce(value, *args, **kwargs)

    with mock.patch.object(Projector, "_project_raw", kernel), \
            mock.patch.object(Projector, "_backproject_raw", kernel), \
            mock.patch.object(ReferenceProjector, "_intervals", kernel), \
            mock.patch.object(operators, "_project_into", kernel), \
            mock.patch.object(operators, "_backproject_into", kernel), \
            mock.patch.object(dist, "all_reduce", reduce):
        yield


def _start_group(rank, rendezvous, transport, timeout=_GROUP_TIMEOUT):
    torch.set_num_threads(1)
    if transport == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(transport, init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=timeout))


def _rejection_worker(rank, rendezvous, directory, transport, operation):
    _start_group(rank, rendezvous, transport)
    try:
        backend = "numba" if transport == "nccl" else "torch"
        kwargs = {"devices": [rank]} if backend == "numba" else {}
        operator = _operator(_frames(), lambda u, v: _offsets(u, v, .037),
                             backend=backend, distributed=True, **kwargs)
        frames = list(_override(_frames()))
        if rank == 1:
            frames[1][0, 0] = float("nan")
        projecting = operation in ("project", "__call__", "project_into")
        shape = operator.volume_shape if projecting else operator.projection_shape
        output_shape = operator.projection_shape if projecting else operator.volume_shape
        device = torch.device("cuda", rank) if transport == "nccl" else torch.device("cpu")
        value = torch.ones(shape, dtype=operator.dtype, device=device)
        output = torch.full(output_shape, 73., dtype=operator.dtype, device=device)
        calls, error = [], None
        started = time.monotonic()
        try:
            with _forbid_numerics(calls):
                if operation.endswith("_into"):
                    getattr(operator, operation)(value, output, trajectory=frames)
                else:
                    getattr(operator, operation)(value, trajectory=frames)
        except (TypeError, ValueError, RuntimeError, AssertionError) as failure:
            error = type(failure).__name__
        record = {"error": error, "calls": calls, "elapsed": time.monotonic() - started,
                  "unchanged": bool(torch.all(output == 73.))}
        root = Path(directory)
        (root / f"rank-{rank}.json").write_text(json.dumps(record), encoding="utf-8")
        deadline = time.monotonic() + 2
        while not all((root / f"rank-{peer}.json").exists() for peer in range(2)):
            assert time.monotonic() < deadline, "peer did not reject its override"
            time.sleep(.01)
    finally:
        dist.destroy_process_group()


def _run_two(worker, tmp_path, *args, seconds=35):
    rendezvous = (tmp_path / "rendezvous").resolve().as_uri()
    context = spawn(worker, args=(rendezvous, str(tmp_path), *args), nprocs=2, join=False)
    deadline = time.monotonic() + seconds
    try:
        while not context.join(timeout=1):
            assert time.monotonic() < deadline, f"distributed override workers exceeded {seconds} seconds"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=3)


def _check_rejection(tmp_path, transport, operation):
    _run_two(_rejection_worker, tmp_path, transport, operation)
    records = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    for record in records:
        assert record["error"] == "ValueError", records
        assert not record["calls"], records
        assert record["unchanged"], records
        assert record["elapsed"] < _GROUP_TIMEOUT, "rejection waited for the process-group timeout"


@_GLOO
@pytest.mark.parametrize("operation", _OPERATIONS)
def test_gloo_curved_override_one_invalid_rank_rejects_before_numerical_work(tmp_path, operation):
    _check_rejection(tmp_path, "gloo", operation)


@pytest.mark.cuda
@_NCCL
@pytest.mark.parametrize("operation", _OPERATIONS)
def test_nccl_native_curved_override_one_invalid_rank_rejects_before_numerical_work(tmp_path, operation):
    _check_rejection(tmp_path, "nccl", operation)


def _gradient_worker(rank, rendezvous, directory, transport):
    _start_group(rank, rendezvous, transport, timeout=30)
    try:
        backend = "numba" if transport == "nccl" else "torch"
        device = torch.device("cuda", rank) if transport == "nccl" else torch.device("cpu")
        coefficient = torch.tensor(.037, dtype=torch.float64, requires_grad=True)
        operator = _operator(
            _frames(), lambda u, v: _offsets(u, v, coefficient), backend=backend,
            distributed=True, view_chunk_size=2,
            **({"devices": [rank]} if backend == "numba" else {}),
        )
        frames, image, rays = _override(_frames()), _data((4, 5)), _data((5, 7), .41)
        expected = _finite_difference(frames, image, rays)
        local_rays = torch.zeros_like(rays)
        local_rays[operator.view_slice] = rays[operator.view_slice]
        local = _finite_difference(frames, image, local_rays)
        assert abs(local - expected) > 1e-3, "fixture cannot distinguish local and SUM gradients"
        values = []
        for operation in ("project", "backproject"):
            source, weight = ((image, rays[operator.view_slice]) if operation == "project"
                              else (rays[operator.view_slice], image))
            with _WorldPointTrace((7,)) as trace:
                result = getattr(operator, operation)(source.to(device), trajectory=frames)
                replicas = operator.space_world_size if operation == "project" else operator.view_world_size
                gradient, = torch.autograd.grad((result.double() * weight.to(device)).sum() / replicas, coefficient)
            if backend == "numba":
                assert trace.records, "native override did not compose detector world points"
                maximum = min(2, operator.view_slice.stop - operator.view_slice.start)
                assert all(actual_device == device and shape[0] <= maximum
                           for shape, actual_device in trace.records), trace.records
            assert torch.isfinite(gradient)
            tolerance = 3e-3 if backend == "numba" else 4e-5
            assert float(gradient) == pytest.approx(expected, rel=tolerance, abs=3e-4 if backend == "numba" else 5e-7)
            values.append(float(gradient))
        (Path(directory) / f"rank-{rank}.json").write_text(json.dumps(values), encoding="utf-8")
    finally:
        dist.destroy_process_group()


@_GLOO
def test_gloo_captured_callable_parameter_override_gradients_are_global_sums(tmp_path):
    _run_two(_gradient_worker, tmp_path, "gloo")


@pytest.mark.cuda
@_NCCL
def test_nccl_native_captured_callable_parameter_override_gradients_are_global_sums(tmp_path):
    _run_two(_gradient_worker, tmp_path, "nccl", seconds=90)


@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_flat_override_admission_ray_count_does_not_grow_with_detector(beam, monkeypatch):
    frames = _frames(3, three_dimensional=beam == "cone")
    frames = (100 * frames[0], 100 * frames[1], *frames[2:])
    detectors = ((7, 5), (43, 29)) if beam == "cone" else ((7,), (509,))
    original = ReferenceProjector._rays
    counts = []

    def rays(reference, views, pixels, device, *args, **kwargs):
        if kwargs.get("validate", args[0] if args else False):
            counts.append((views.stop - views.start) * math.prod(pixel.stop - pixel.start for pixel in pixels))
        return original(reference, views, pixels, device, *args, **kwargs)

    def execute(owner, value, *args):
        return args[0] if args else value

    monkeypatch.setattr(Projector, "_project", execute)
    monkeypatch.setattr(Projector, "_backproject", execute)
    monkeypatch.setattr(Projector, "_execute_into", execute)
    for detector in detectors:
        operator = Projector(frames, (3, 4, 5) if beam == "cone" else (4, 5), detector,
                             beam=beam, backend="torch", precision="float64")
        with mock.patch.object(ReferenceProjector, "_rays", rays):
            for operation in _OPERATIONS:
                counts.clear()
                projecting = operation in ("project", "__call__", "project_into")
                source = torch.zeros(operator.volume_shape if projecting else operator.projection_shape,
                                     dtype=torch.float64)
                if operation.endswith("_into"):
                    output = torch.empty(operator.projection_shape if projecting else operator.volume_shape,
                                         dtype=torch.float64)
                    getattr(operator, operation)(source, output, trajectory=_override(frames))
                else:
                    getattr(operator, operation)(source, trajectory=_override(frames))
                assert sum(counts) <= 8 * len(frames[0]), (beam, detector, operation, counts)


@pytest.mark.parametrize("backend", ["torch", "numba"])
@pytest.mark.parametrize("source_x,pitch", [
    (-1., 2e15),
    (-2e6, 4e6),
], ids=["pixels-beyond-1e15", "distant-source-and-wide-pixels"])
def test_flat_override_range_limits_apply_to_frames_and_allow_distant_pixels(
    backend, source_x, pitch, monkeypatch,
):
    frames = (
        torch.tensor([[source_x, 0.]], dtype=torch.float64),
        torch.tensor([[0., 0.]], dtype=torch.float64),
        torch.tensor([[0., 1.]], dtype=torch.float64),
    )
    coordinates = torch.tensor([-pitch, 0., pitch], dtype=torch.float64)
    points = frames[1][:, None] + coordinates[None, :, None] * frames[2][:, None]
    assert torch.isfinite(points.float()).all()
    assert torch.isfinite(points.float() - frames[0].float()[:, None]).all()
    assert torch.all(torch.any(points.float() != frames[0].float()[:, None], dim=-1))
    assert torch.linalg.vector_norm(points[:, (0, 2)], dim=-1).min() > (
        1e15 if pitch == 2e15 else 1e6)
    calls = []

    def execute(owner, value, *args):
        calls.append(owner.backend)
        return args[0] if args else value

    monkeypatch.setattr(Projector, "_project", execute)
    monkeypatch.setattr(Projector, "_backproject", execute)
    monkeypatch.setattr(Projector, "_execute_into", execute)
    # Replace device discovery and execution only. Native admission retains
    # its real FP32 frame and pixel arithmetic on the CPU transport device.
    with mock.patch.object(operators, "select_backend", return_value=backend), \
            mock.patch.object(operators, "_normalize_devices", return_value=(torch.device("cpu"),)), \
            mock.patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("admission initialized CUDA")):
        operator = Projector(frames, (2, 3), (3,), beam="fan", detector_spacing=pitch,
                             backend=backend, precision="fast")
        assert operator.backend == backend
        for operation in _OPERATIONS:
            projecting = operation in ("project", "__call__", "project_into")
            source = torch.ones(operator.volume_shape if projecting else operator.projection_shape)
            output = torch.empty(operator.projection_shape if projecting else operator.volume_shape)
            for trajectory in (None, frames):
                if operation.endswith("_into"):
                    assert getattr(operator, operation)(source, output, trajectory=trajectory) is output
                else:
                    assert getattr(operator, operation)(source, trajectory=trajectory) is source
    assert calls == [backend] * (2 * len(_OPERATIONS))


class _FrameStagingTrace(TorchDispatchMode):
    """Follow snapshots and count frame copies with an explicit target device."""

    def __init__(self, frames):
        super().__init__()
        self.origins = {value.untyped_storage()._cdata: index for index, value in enumerate(frames)}
        self.keepalive = list(frames)
        self.counts = Counter()

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        if (str(function) in ("aten.clone.default", "aten._to_copy.default")
                and args and isinstance(args[0], torch.Tensor) and isinstance(result, torch.Tensor)):
            source = args[0]
            storage = source.untyped_storage()._cdata
            if storage in self.origins:
                origin = self.origins[storage]
                self.origins[result.untyped_storage()._cdata] = origin
                self.keepalive.append(result)
                if (kwargs is not None and "device" in kwargs
                        and source.dtype == torch.float64 and result.dtype == torch.float32):
                    self.counts[(origin, result.device)] += 1
        return result


@pytest.mark.parametrize("beam,ambiguous", [("fan", False), ("cone", False), ("cone", True)],
                         ids=["fan-corners", "cone-corners", "cone-exact-fallback"])
def test_native_flat_override_stages_each_frame_once_for_all_admission_corners(
    beam, ambiguous, monkeypatch,
):
    frames = _frames(5, three_dimensional=beam == "cone")
    detector = (7, 5) if beam == "cone" else (7,)
    if ambiguous:
        diagonal = math.sqrt(.5)
        frames = tuple(torch.tensor([value], dtype=torch.float64) for value in (
            [0., 0., -5.], [0., 0., 5.], [diagonal, 0., diagonal], [0., 1., 0.],
        ))
        detector = (41, 3)
        corners = torch.stack([frames[1][0] + u * frames[2][0] + v * frames[3][0]
                               for u in (-20., 20.) for v in (-1., 1.)])
        assert torch.all(frames[0][0] >= corners.amin(0))
        assert torch.all(frames[0][0] <= corners.amax(0))
        normal = torch.linalg.cross(frames[2], frames[3], dim=-1)
        assert abs(float(((frames[1] - frames[0]) * normal).sum())) > 1.

    def execute(owner, value, *args):
        return args[0] if args else value

    monkeypatch.setattr(Projector, "_project", execute)
    monkeypatch.setattr(Projector, "_backproject", execute)
    monkeypatch.setattr(Projector, "_execute_into", execute)
    # CPU transport preserves native FP32 staging and every admission check.
    with mock.patch.object(operators, "select_backend", return_value="numba"), \
            mock.patch.object(operators, "_normalize_devices", return_value=(torch.device("cpu"),)), \
            mock.patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("admission initialized CUDA")):
        operator = Projector(frames, (3, 4, 5) if beam == "cone" else (4, 5), detector,
                             beam=beam, backend="numba", precision="fast")
        for operation in _OPERATIONS:
            projecting = operation in ("project", "__call__", "project_into")
            source = torch.ones(operator.volume_shape if projecting else operator.projection_shape)
            output = torch.empty(operator.projection_shape if projecting else operator.volume_shape)
            with _FrameStagingTrace(frames) as trace:
                if operation.endswith("_into"):
                    getattr(operator, operation)(source, output, trajectory=frames)
                else:
                    getattr(operator, operation)(source, trajectory=frames)
            expected = Counter({(index, torch.device("cpu")): 1 for index in range(len(frames))})
            assert trace.counts == expected, (beam, ambiguous, operation, trace.counts)


class _WorldPointTrace(TorchDispatchMode):
    """Observe real frame-plus-offset additions without replacing arithmetic."""

    def __init__(self, detector):
        super().__init__()
        self.detector, self.records = detector, []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        if (str(function) == "aten.add.Tensor" and isinstance(result, torch.Tensor)
                and result.ndim == len(self.detector) + 2
                and result.shape[-1] == len(self.detector) + 1
                and all(0 < size <= full for size, full in zip(result.shape[1:-1], self.detector))):
            self.records.append((tuple(result.shape), result.device))
        return result


@pytest.mark.cuda
@_CUDA
@pytest.mark.parametrize("devices", [(0,), (0, 1)])
@pytest.mark.parametrize("policy", ["full-legacy", "bounded-legacy", "bounded-parameterized"])
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_native_override_world_points_use_compute_devices_and_owned_batches(devices, policy, operation):
    if len(devices) > torch.cuda.device_count():
        pytest.skip("two CUDA devices are required for this ownership case")
    frames = _frames(5, three_dimensional=True)
    detector = (7, 5)
    samples = []

    def sampler(u, v, ids):
        samples.append(tuple(ids.tolist()))
        return _offsets(u, v, .037)

    surface = ParameterizedSurface(sampler) if policy.endswith("parameterized") else (
        lambda u, v: _offsets(u, v, .037))
    bounded = policy.startswith("bounded")
    operator = Projector(
        frames, (3, 4, 5), detector, beam="cone", detector_spacing=(.83, .67),
        voxel_spacing=.71, detector_surface=surface, backend="numba", devices=list(devices),
        **({"view_chunk_size": 2, "detector_chunk_shape": (3, 2),
            "volume_chunk_shape": (2, 3, 3)} if bounded else {}),
    )
    samples.clear()
    source = torch.ones(operator.volume_shape if operation == "project" else operator.projection_shape,
                        dtype=torch.float32,
                        device="cpu" if bounded else f"cuda:{devices[0]}")
    if policy == "full-legacy":
        assert operator._execution_plan(source, operation == "project") is None
    with _WorldPointTrace(detector) as trace:
        result = getattr(operator, operation)(source, trajectory=_override(frames))
    assert torch.isfinite(result).all()
    assert trace.records, "native override did not expose world-point composition"
    allowed = {torch.device("cuda", index) for index in devices}
    ownership = {torch.device("cuda", index): (5 // len(devices) + int(offset < 5 % len(devices)))
                 for offset, index in enumerate(devices)}
    for shape, device in trace.records:
        assert device in allowed, ("native override expanded world points away from compute GPUs", shape, device)
        assert shape[0] <= min(ownership[device], 2 if bounded else 5), (shape, device, ownership)
        if bounded:
            assert shape[1] <= 3 and shape[2] <= 2, ("validation ignored the admitted pixel batch", shape)
    if samples:
        shards, start = [], 0
        for count in ownership.values():
            shards.append(set(range(start, start + count)))
            start += count
        assert all(any(set(ids) <= shard for shard in shards) for ids in samples), samples


@pytest.mark.cuda
@_CUDA
@pytest.mark.parametrize("bounded", [False, True], ids=["full", "bounded"])
def test_native_override_rejects_actual_interior_surface_pixel_on_compute_gpu(bounded):
    frames = (
        torch.tensor([[-9., 0., 0.]], dtype=torch.float64),
        torch.tensor([[6., 0., 0.]], dtype=torch.float64),
        torch.tensor([[0., 1., 0.]], dtype=torch.float64),
        torch.tensor([[0., 0., 1.]], dtype=torch.float64),
    )

    def surface(u, v):
        depth = torch.where((u == 0) & (v == 0), torch.full_like(u, -16.), .03 * u.square())
        return torch.stack((u, v, depth), -1)

    operator = Projector(
        frames, (2, 3, 4), (5, 5), beam="cone", detector_surface=surface,
        backend="numba", devices=[0],
        **({"view_chunk_size": 1, "detector_chunk_shape": (2, 2)} if bounded else {}),
    )
    changed = tuple(value.clone() for value in frames)
    changed[1][0, 0] = 7.
    value = torch.ones(operator.volume_shape, dtype=torch.float32, device="cuda:0")
    if not bounded:
        assert operator._execution_plan(value, True) is None
    calls = []
    with _WorldPointTrace((5, 5)) as trace, _forbid_numerics(calls), \
            pytest.raises(ValueError, match="differ|coincid"):
        operator.project(value, trajectory=changed)
    assert not calls
    assert trace.records, "actual interior pixel was not composed for validation"
    assert all(device == torch.device("cuda:0") for _, device in trace.records), trace.records


@pytest.mark.cuda
@_CUDA
@pytest.mark.parametrize("surface_kind", ["legacy", "parameterized"])
def test_native_override_admission_stages_each_original_cuda_frame_once(surface_kind, monkeypatch):
    frames = tuple(value.cuda() for value in _frames(3, three_dimensional=True))
    coefficient = torch.tensor(.037, dtype=torch.float64, device="cuda:0")
    surface = (ParameterizedSurface(lambda u, v, ids, p: _offsets(u, v, p), (coefficient,))
               if surface_kind == "parameterized" else lambda u, v: _offsets(u, v, .037))
    operator = Projector(frames, (3, 4, 5), (7, 5), beam="cone", detector_surface=surface,
                         backend="numba", devices=[0], view_chunk_size=1, detector_chunk_shape=(2, 2))
    counts = Counter()
    original_to, original_cpu = torch.Tensor.to, torch.Tensor.cpu

    def to(value, *args, **kwargs):
        result = original_to(value, *args, **kwargs)
        if value.device.type == "cuda" and result.device.type == "cpu":
            counts["d2h"] += 1
        return result

    def cpu(value, *args, **kwargs):
        result = original_cpu(value, *args, **kwargs)
        counts["d2h"] += int(value.device.type == "cuda")
        return result

    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    monkeypatch.setattr(Projector, "_project", lambda owner, value: value)
    operator.project(torch.zeros(operator.volume_shape), trajectory=frames)
    if surface_kind == "legacy":
        assert counts["d2h"] == 0, ("legacy admission staged CUDA frames on CPU", counts)
    else:
        # Explicit sampler parameters retain their public CPU sampling contract.
        components = len(frames) + 1
        assert 0 < counts["d2h"] <= components, (surface_kind, counts, components)
