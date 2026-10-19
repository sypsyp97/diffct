"""Surface expansion uses its arithmetic precision and only owned view batches.

Numerical references reuse the original NumPy voxel-box oracle. Allocation
observers record real tensor operations and do not replace geometry code.
"""

import time
from contextlib import ExitStack
from datetime import timedelta
from unittest import mock

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from diffct import ParameterizedSurface
import diffct.operators as operators
from tests.test_detector_surfaces import (
    _arc_offsets, _case, _close, _data, _finite_difference, _matrix, _numpy,
    _projector,
)


CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="native surface execution requires NVIDIA CUDA",
)


def _unlimited_native_validation_memory():
    patches = ExitStack()
    unlimited = (1 << 63) - 1
    patches.enter_context(mock.patch.object(operators, "_memory_lower_bounds", return_value=(unlimited,) * 3))
    patches.enter_context(mock.patch.object(operators, "_memory_budget", return_value=(unlimited,) * 2))
    patches.enter_context(mock.patch.object(operators, "_reservation_budget", return_value=unlimited))
    patches.enter_context(mock.patch.object(torch.cuda, "get_allocator_backend", return_value="cudaMallocAsync"))
    return patches


def _arc(u, v, radius):
    return torch.stack((radius * torch.sin(u / radius) + .037 * u.square(),
                        v, radius * (1 - torch.cos(u / radius))), dim=-1)


def _surface(kind, radius, records=None, offset_device="cpu"):
    if kind == "legacy":
        def callback(u, v):
            return _arc(u, v, radius).to(device=offset_device)
        return callback

    def sampler(u, v, view_indices, value):
        if records is not None:
            records.append(tuple(view_indices.tolist()))
        return _arc(u, v, value).to(device=offset_device)

    return ParameterizedSurface(sampler, (radius,))


class _WorldPointAdds(TorchDispatchMode):
    """Observe broadcast frame additions, excluding shared local offsets."""

    def __init__(self, case):
        super().__init__()
        self.case = case
        self.records = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        if (str(function) == "aten.add.Tensor" and isinstance(result, torch.Tensor)
                and result.ndim == len(self.case.detector) + 2
                and result.shape[-1] == len(self.case.shape)
                and all(0 < size <= full for size, full in
                        zip(result.shape[1:-1], self.case.detector))):
            self.records.append((tuple(result.shape), result.device, result.dtype))
        return result


def _check_expansion(records, *, dtype, devices, maximum_views, pixels=None):
    assert records, "surface execution did not expose a world-point addition"
    for shape, device, actual_dtype in records:
        assert actual_dtype == dtype, ("world-point arithmetic used the wrong precision", shape, actual_dtype)
        assert device in devices, ("world points were expanded away from the compute device", shape, device)
        assert shape[0] <= maximum_views, ("world points included unowned views", shape, maximum_views)
        if pixels is not None:
            assert all(size <= limit for size, limit in zip(shape[1:-1], pixels)), \
                ("world points exceeded the selected detector rectangle", shape, pixels)


@pytest.mark.parametrize("beam", ["parallel", "fan", "cone"])
@pytest.mark.parametrize("kind", ["legacy", "parameterized"])
@pytest.mark.parametrize("precision", ["fast", "float32", "float64"])
def test_surface_compute_precision_values_adjoint_and_gradients_match_original_oracle(
    beam, kind, precision,
):
    case = _case(beam, learnable=True)
    radius = torch.tensor(3.7, dtype=torch.float64, requires_grad=True)
    operator = _projector(case, _surface(kind, radius), backend="torch",
                          precision=precision, view_chunk_size=2)
    image, rays = _data(case, device="cpu")
    matrix = _matrix(case, _arc_offsets(case, 3.7, .037))
    image_np, rays_np = _numpy(image).ravel(), _numpy(rays).ravel()
    geometry = tuple(_numpy(component) for component in case.trajectory)
    expected_dtype = torch.float64 if precision == "float64" else torch.float32
    tolerance = (2e-11, 2e-11) if precision == "float64" else (4e-4, 6e-5)
    expected_radius = _finite_difference(
        lambda value: float(rays_np @ _matrix(case, _arc_offsets(case, value, .037)) @ image_np),
        3.7,
    )
    generator = np.random.default_rng(922)
    frame_checks = []
    for component, value in enumerate(geometry):
        direction = generator.normal(size=value.shape)
        if beam == "parallel" and component == 0:
            direction -= (direction * value).sum(-1, keepdims=True) * value
        direction /= np.linalg.norm(direction)

        def objective(changed):
            frames = list(geometry)
            frames[component] = changed
            return float(rays_np @ _matrix(case, _arc_offsets(case, 3.7, .037), frames) @ image_np)

        frame_checks.append((direction, _finite_difference(objective, value, direction)))
    outputs = []
    for operation, source, weight, linear in (
        ("project", image, rays, matrix),
        ("backproject", rays, image, matrix.T),
    ):
        data = source.double().requires_grad_()
        with _WorldPointAdds(case) as trace:
            result = getattr(operator, operation)(data)
        assert result.dtype == expected_dtype and result.device == data.device
        _check_expansion(trace.records, dtype=expected_dtype, devices={data.device}, maximum_views=2)
        _close(result, (linear @ _numpy(source).ravel()).reshape(result.shape),
               rtol=tolerance[0], atol=tolerance[1])
        loss = (result.double() * weight.double()).sum()
        gradients = torch.autograd.grad(loss, (data, radius, *case.trajectory))
        _close(gradients[0], (linear.T @ _numpy(weight).ravel()).reshape(data.shape),
               rtol=tolerance[0], atol=tolerance[1])
        assert abs(expected_radius) > 1e-3
        assert gradients[1].item() == pytest.approx(expected_radius, rel=5e-3, abs=3e-4)
        for gradient, (direction, expected) in zip(gradients[2:], frame_checks):
            assert torch.isfinite(gradient).all()
            assert abs(expected) > 1e-3
            actual = float((_numpy(gradient) * direction).sum())
            assert actual == pytest.approx(expected, rel=5e-3, abs=3e-4)
        outputs.append(result.detach())
    lhs = (outputs[0].double() * rays.double()).sum()
    rhs = (image.double() * outputs[1].double()).sum()
    torch.testing.assert_close(lhs, rhs, rtol=tolerance[0], atol=tolerance[1])


@pytest.mark.cuda
@CUDA_REQUIRED
@pytest.mark.parametrize("kind", ["legacy", "parameterized"])
@pytest.mark.parametrize("offset_device", ["cpu", "cuda:0"])
def test_native_streamed_offsets_expand_only_compute_device_view_pixel_batches(kind, offset_device):
    case = _case("cone", views=5, learnable=True)
    radius = torch.tensor(3.7, dtype=torch.float64, requires_grad=True)
    samples = []
    operator = _projector(
        case, _surface(kind, radius, samples, offset_device), backend="numba",
        devices=[0], volume_chunk_shape=(3, 4, 5), view_chunk_size=2,
        detector_chunk_shape=(4, 3), schedule="spatial",
    )
    samples.clear()
    image, rays = _data(case, device="cpu")
    matrix = _matrix(case, _arc_offsets(case, 3.7, .037))
    expected_radius = _finite_difference(
        lambda value: float(_numpy(rays).ravel() @
                            _matrix(case, _arc_offsets(case, value, .037)) @ _numpy(image).ravel()),
        3.7,
    )
    for operation, source, weight, linear in (
        ("project", image, rays, matrix), ("backproject", rays, image, matrix.T),
    ):
        with _WorldPointAdds(case) as trace:
            result = getattr(operator, operation)(source)
            gradient, = torch.autograd.grad((result.double() * weight.double()).sum(), radius)
        _check_expansion(trace.records, dtype=torch.float64, devices={torch.device("cuda:0")},
                         maximum_views=2, pixels=(4, 3))
        assert result.device.type == "cpu" and result.dtype == torch.float32
        _close(result, (linear @ _numpy(source).ravel()).reshape(result.shape))
        assert gradient.item() == pytest.approx(expected_radius, rel=5e-3, abs=4e-4)
    if kind == "parameterized":
        assert samples and all(0 < len(ids) <= 2 for ids in samples)
        assert set().union(*map(set, samples)) == set(range(case.views))


@pytest.mark.cuda
@CUDA_REQUIRED
def test_per_view_legacy_surface_vjp_keeps_cuda_gradients_batch_local():
    case = _case("cone", views=5)
    radii = (3.2 + .23 * torch.arange(case.views, dtype=torch.float64)).requires_grad_()

    def callback(u, v):
        radius = radii.reshape(case.views, 1, 1)
        x = radius * torch.sin(u / radius) + .037 * u.square()
        return torch.stack((x, v.expand_as(x), radius * (1 - torch.cos(u / radius))), dim=-1)

    class NoGlobalCudaOffsets(_WorldPointAdds):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            result = super().__torch_dispatch__(function, types, args, kwargs)
            for value in tree_flatten(result)[0]:
                if isinstance(value, torch.Tensor) and value.is_cuda:
                    assert tuple(value.shape) != (*case.sino_shape, 3), \
                        ("VJP created a global CUDA offset or gradient tensor", str(function))
            return result

    devices = list(range(min(2, torch.cuda.device_count())))
    operator = _projector(
        case, callback, backend="numba", devices=devices, view_chunk_size=2,
        detector_chunk_shape=(4, 3), volume_chunk_shape=(3, 4, 5), schedule="spatial",
    )
    image, rays = _data(case, device="cpu")
    image_np, rays_np, radii_np = _numpy(image).ravel(), _numpy(rays).ravel(), _numpy(radii)
    matrix = _matrix(case, _arc_offsets(case, radii_np, .037))
    expected_gradient = []
    for index in range(case.views):
        direction = np.eye(case.views)[index]
        expected_gradient.append(_finite_difference(
            lambda values: float(rays_np @ _matrix(case, _arc_offsets(case, values, .037)) @ image_np),
            radii_np, direction,
        ))
    for operation, source, weight, linear in (
        ("project", image, rays, matrix), ("backproject", rays, image, matrix.T),
    ):
        with NoGlobalCudaOffsets(case) as trace:
            result = getattr(operator, operation)(source)
            gradient, = torch.autograd.grad((result.double() * weight.double()).sum(), radii)
        _check_expansion(trace.records, dtype=torch.float64,
                         devices={torch.device("cuda", index) for index in devices},
                         maximum_views=2, pixels=(4, 3))
        assert gradient.device == radii.device and gradient.dtype == radii.dtype
        _close(result, (linear @ _numpy(source).ravel()).reshape(result.shape))
        _close(gradient, np.asarray(expected_gradient), rtol=5e-3, atol=4e-4)


@pytest.mark.cuda
@CUDA_REQUIRED
@pytest.mark.parametrize("kind", ["legacy", "parameterized"])
def test_two_native_devices_expand_each_owned_view_shard(kind):
    if torch.cuda.device_count() < 2:
        pytest.skip("two NVIDIA CUDA devices are required")
    case = _case("cone", views=5)
    radius = torch.tensor(3.7, dtype=torch.float64)
    samples = []
    operator = _projector(case, _surface(kind, radius, samples), backend="numba", devices=[0, 1])
    image, rays = _data(case)
    matrix = _matrix(case, _arc_offsets(case, 3.7, .037))
    for operation, source, linear in (
        ("project", image, matrix), ("backproject", rays, matrix.T),
    ):
        samples.clear()
        with _WorldPointAdds(case) as trace:
            result = getattr(operator, operation)(source)
        _check_expansion(trace.records, dtype=torch.float64,
                         devices={torch.device("cuda:0"), torch.device("cuda:1")}, maximum_views=3)
        assert {device for _, device, _ in trace.records} == {
            torch.device("cuda:0"), torch.device("cuda:1"),
        }
        assert all(shape[0] <= (3 if device.index == 0 else 2)
                   for shape, device, _ in trace.records)
        if kind == "parameterized":
            assert samples and all(set(ids) <= {0, 1, 2} or set(ids) <= {3, 4} for ids in samples)
            assert set().union(*map(set, samples)) == set(range(case.views))
        _close(result, (linear @ _numpy(source).ravel()).reshape(result.shape))


def test_native_constructor_validates_each_device_view_shard():
    case = _case("cone", views=5)
    radius = torch.tensor(3.7, dtype=torch.float64)
    # Device labels stand for two compute workers. Tensor arithmetic stays real
    # on CPU so constructor view ownership is testable without CUDA hardware.
    with mock.patch.object(operators, "select_backend", return_value="numba"), \
            mock.patch.object(operators, "_normalize_devices", return_value=(torch.device("cpu"),) * 2), \
            _unlimited_native_validation_memory(), \
            _WorldPointAdds(case) as trace:
        _projector(case, _surface("legacy", radius), devices=[0, 1])
    assert trace.records
    assert all(shape[0] <= 3 for shape, _, _ in trace.records), \
        "constructor expanded views assigned to another compute device"


class _CudaMetadataTensor(torch.Tensor):
    """Supply CUDA metadata to admission without allocating CUDA storage."""

    @property
    def device(self):
        return torch.device("cuda:0")

    @property
    def is_cuda(self):
        return True


@pytest.mark.parametrize("geometry_dtype, budget, full_fits", [
    (torch.float32, 250 * 1024, True),
    (torch.float64, 250 * 1024, False),
    (torch.float64, 400 * 1024, True),
])
def test_full_surface_admission_counts_actual_composition_dtype_only_for_owned_views(
    geometry_dtype, budget, full_fits,
):
    case = _case("cone", views=8)
    case.shape, case.detector = (2, 3, 4), (24, 20)
    case.trajectory = tuple(value.to(geometry_dtype) for value in case.trajectory)
    radius = torch.tensor(3.7, dtype=torch.float64)

    def surface(u, v):
        return _arc(u, v, radius).to(geometry_dtype)

    with mock.patch.object(operators, "select_backend", return_value="numba"), \
            mock.patch.object(operators, "_normalize_devices", return_value=(torch.device("cpu"),) * 2), \
            _unlimited_native_validation_memory():
        operator = _projector(case, surface, backend="numba", precision="fast")
    # This planner reads metadata only. Real CUDA integration above observes
    # actual points; this case keeps its admission boundary testable on CPU.
    operator._devices = (torch.device("cuda:0"), torch.device("cuda:1"))
    data = torch.empty(case.shape).as_subclass(_CudaMetadataTensor)
    unlimited = (1 << 63) - 1
    with mock.patch.object(torch.cuda, "is_available", return_value=True), \
            mock.patch.object(torch.cuda, "get_allocator_backend", return_value="native"), \
            mock.patch.object(operators, "_memory_lower_bounds", return_value=(budget, unlimited, unlimited)), \
            mock.patch.object(operators, "_memory_budget", return_value=(budget, unlimited)), \
            mock.patch.object(operators, "_reservation_budget", return_value=unlimited):
        geometry_spec, _ = operator._effective_geometry(None, torch.device("cpu"))
        plan = operator._execution_plan(data, True, geometry_spec=geometry_spec)
    assert (plan is None) == full_fits, \
        "full admission must count the actual composition dtype for four owned views and native FP32 conversion"


def _gloo_local_surface_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        for views in (1, 5):
            for kind in ("legacy", "parameterized"):
                case = _case("fan", views=views)
                radius = torch.tensor(3.7, dtype=torch.float64)
                samples = []
                operator = _projector(
                    case, _surface(kind, radius, samples), backend="torch", precision="float32",
                    distributed=True, view_chunk_size=2,
                )
                samples.clear()
                image, rays = _data(case, device="cpu")
                matrix = _matrix(case, _arc_offsets(case, 3.7, .037))
                owned = operator.view_slice
                expected_ids = set(range(owned.start, owned.stop))
                for operation, source, expected in (
                    ("project", image, (matrix @ _numpy(image).ravel()).reshape(case.sino_shape)[owned]),
                    ("backproject", rays[owned], (matrix.T @ _numpy(rays).ravel()).reshape(case.shape)),
                ):
                    samples.clear()
                    with _WorldPointAdds(case) as trace:
                        result = getattr(operator, operation)(source)
                    if expected_ids:
                        _check_expansion(trace.records, dtype=torch.float32,
                                         devices={torch.device("cpu")}, maximum_views=min(2, len(expected_ids)))
                    else:
                        assert not trace.records, "empty rank expanded world points"
                    if kind == "parameterized":
                        assert all(set(ids) <= expected_ids and len(ids) <= 2 for ids in samples)
                        assert set().union(*map(set, samples)) == expected_ids
                    _close(result, expected)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(),
                    reason="rank-local sampling requires Gloo")
def test_gloo_surface_sampling_and_world_points_are_rank_local_including_empty_rank(tmp_path):
    rendezvous = (tmp_path / "surface-local-views").resolve().as_uri()
    context = mp.spawn(_gloo_local_surface_worker, args=(rendezvous,), nprocs=2, join=False)
    deadline = time.monotonic() + 90
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise AssertionError("rank-local surface test exceeded 90 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def _gloo_invalid_native_surface_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=15))
    try:
        for kind in ("legacy", "parameterized"):
            case = _case("fan", views=2)
            case.shape, case.detector, case.spacing = (1, 1), (1,), 1.
            case.trajectory = (
                torch.tensor([[-9., 0.], [1_000_000.01, 0.]], dtype=torch.float64),
                torch.tensor([[6., .02], [1_000_000.01, .02]], dtype=torch.float64),
                torch.tensor([[0., 1.], [0., 1.]], dtype=torch.float64),
            )
            samples = []

            def callback(u, v):
                return torch.stack((torch.zeros_like(u), torch.zeros_like(v), torch.zeros_like(u)), dim=-1)

            def sampler(u, v, ids):
                samples.append(tuple(ids.tolist()))
                return callback(u, v)

            surface = callback if kind == "legacy" else ParameterizedSurface(sampler)
            with mock.patch.object(operators, "select_backend", return_value="numba"), \
                    mock.patch.object(operators, "_normalize_devices", return_value=(torch.device("cpu"),)), \
                    _unlimited_native_validation_memory():
                with pytest.raises(ValueError, match="1e6"):
                    _projector(case, surface, backend="numba", distributed=True)
            if kind == "parameterized":
                assert samples and all(set(ids) <= {rank} for ids in samples), \
                    "native constructor sampled another rank's views"
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(),
                    reason="collective surface admission requires Gloo")
def test_native_surface_constructor_rejects_one_invalid_rank_collectively(tmp_path):
    rendezvous = (tmp_path / "surface-invalid-rank").resolve().as_uri()
    context = mp.spawn(_gloo_invalid_native_surface_worker, args=(rendezvous,), nprocs=2, join=False)
    deadline = time.monotonic() + 40
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise AssertionError("rank-local surface admission did not fail collectively")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


@pytest.mark.parametrize("kind", ["legacy", "parameterized"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_surface_constructor_still_rejects_nonfinite_callback_offsets(kind, bad):
    def callback(u, v):
        offsets = torch.stack((u, v, torch.zeros_like(u)), dim=-1)
        offsets.reshape(-1, 3)[-1, -1] = bad
        return offsets

    surface = callback if kind == "legacy" else ParameterizedSurface(
        lambda u, v, ids: callback(u, v),
    )
    with pytest.raises(ValueError, match="finite"):
        _projector(_case("cone"), surface, backend="torch", precision="float32")


@pytest.mark.parametrize("kind", ["legacy", "parameterized"])
def test_native_surface_constructor_preserves_raw_distance_limit_before_float32_rounding(kind):
    case = _case("fan", views=1)
    case.shape, case.detector, case.spacing = (1, 1), (1,), 1.
    case.trajectory = (
        torch.tensor([[1_000_000.01, 0.]], dtype=torch.float64),
        torch.tensor([[1_000_000.01, .02]], dtype=torch.float64),
        torch.tensor([[0., 1.]], dtype=torch.float64),
    )

    def callback(u, v):
        return torch.stack((torch.zeros_like(u), torch.zeros_like(v), torch.zeros_like(u)), dim=-1)

    surface = callback if kind == "legacy" else ParameterizedSurface(
        lambda u, v, ids: callback(u, v),
    )
    assert all(torch.linalg.vector_norm(value).item() > 1e6 for value in case.trajectory[:2])
    assert all(value.float()[0, 0].item() == 1e6 for value in case.trajectory[:2])
    # Constructor admission must preserve raw bounds. Replacing only backend
    # selection and its device inventory keeps this check runnable without CUDA.
    with mock.patch.object(operators, "select_backend", return_value="numba"), \
            mock.patch.object(operators, "_normalize_devices", return_value=(torch.device("cpu"),)), \
            _unlimited_native_validation_memory():
        with pytest.raises(ValueError, match="1e6"):
            _projector(case, surface, backend="numba", precision="fast")


@pytest.mark.parametrize("kind", ["legacy", "parameterized"])
def test_native_constructor_uses_exact_norm_of_float32_surface_coordinates(kind):
    case = _case("fan", views=1)
    case.shape, case.detector, case.spacing = (1, 1), (1,), 1.
    case.trajectory = (
        torch.tensor([[1_000_000., 1.]], dtype=torch.float32),
        torch.tensor([[1_000_000., 2.]], dtype=torch.float32),
        torch.tensor([[0., 1.]], dtype=torch.float32),
    )

    def callback(u, v):
        return torch.zeros((*u.shape, 3), dtype=torch.float32)

    surface = callback if kind == "legacy" else ParameterizedSurface(
        lambda u, v, ids: callback(u, v),
    )
    assert all(torch.linalg.vector_norm(value.double()).item() > 1e6 for value in case.trajectory[:2])
    assert all(torch.linalg.vector_norm(value).item() == 1e6 for value in case.trajectory[:2])
    with mock.patch.object(operators, "select_backend", return_value="numba"), \
            mock.patch.object(operators, "_normalize_devices", return_value=(torch.device("cpu"),)), \
            _unlimited_native_validation_memory():
        with pytest.raises(ValueError, match="1e6"):
            _projector(case, surface, backend="numba", precision="fast")
