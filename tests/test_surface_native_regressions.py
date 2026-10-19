"""Native surface composition, snapshots, and constructor memory regressions."""

from contextlib import ExitStack
from unittest import mock

import numpy as np
import pytest
import torch
from torch.utils._pytree import tree_flatten

from diffct import ParameterizedSurface, Projector, operators
from tests.test_detector_surfaces import (
    _arc_offsets,
    _case,
    _close,
    _data,
    _finite_difference,
    _matrix,
    _numpy,
    _projector,
)
from tests.test_surface_compute_shards import _arc, _WorldPointAdds


CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="native surface regressions require NVIDIA CUDA",
)
UNLIMITED = (1 << 63) - 1


def _native_metadata(*, available=UNLIMITED, allocation=UNLIMITED,
                     reservation=UNLIMITED, devices=1):
    """Use CPU arithmetic with explicit native memory metadata."""
    patches = ExitStack()
    patches.enter_context(mock.patch.object(operators, "select_backend", return_value="numba"))
    patches.enter_context(mock.patch.object(
        operators, "_normalize_devices", return_value=(torch.device("cpu"),) * devices,
    ))
    patches.enter_context(mock.patch.object(
        operators, "_memory_lower_bounds", return_value=(available, allocation, reservation),
    ))
    patches.enter_context(mock.patch.object(
        operators, "_memory_budget", return_value=(available, allocation),
    ))
    patches.enter_context(mock.patch.object(operators, "_reservation_budget", return_value=reservation))
    # Exact byte reservations make all three independent limits testable on CPU.
    patches.enter_context(mock.patch.object(torch.cuda, "get_allocator_backend", return_value="cudaMallocAsync"))
    return patches


def _cancellation_case(beam, kind, *, frame_dtype=torch.float64, offset_dtype=torch.float64,
                       endpoint=1., learnable=False):
    dimensions = 3 if beam == "cone" else 2
    source = torch.zeros((1, dimensions), dtype=frame_dtype)
    center, axis_u = torch.zeros_like(source), torch.zeros_like(source)
    center[0, 0] = 1e8 if offset_dtype == torch.float64 else 1e8 + endpoint
    axis_u[0, 1] = 1.
    frames = (source, center, axis_u)
    if beam == "cone":
        axis_v = torch.zeros_like(source)
        axis_v[0, 2] = 1.
        frames += (axis_v,)
    frames = tuple(value.requires_grad_(learnable) for value in frames)
    normal = torch.tensor(-1e8 + endpoint if offset_dtype == torch.float64 else -1e8,
                          dtype=offset_dtype, requires_grad=learnable)
    tangent = .13 if endpoint < .5 else 0.
    vertical = -.07 if beam == "cone" and endpoint < .5 else 0.

    def offsets(u, v, value):
        return torch.stack((torch.full_like(u, tangent), torch.full_like(v, vertical),
                            torch.ones_like(u) * value), dim=-1).to(offset_dtype)

    if kind == "legacy":
        def surface(u, v):
            return offsets(u, v, normal)
    else:
        def sampler(u, v, ids, value):
            return offsets(u, v, value)
        surface = ParameterizedSurface(sampler, (normal,))
    return frames, surface, normal, tangent, vertical


@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("kind", ["legacy", "parameterized"])
@pytest.mark.parametrize("frame_dtype", [torch.float32, torch.float64])
def test_native_constructor_accepts_promoted_cancellation_before_final_float32_cast(
    beam, kind, frame_dtype,
):
    frames, surface, _, _, _ = _cancellation_case(beam, kind, frame_dtype=frame_dtype)
    dimensions = 3 if beam == "cone" else 2
    case = _case(beam, views=1)
    case.shape, case.detector = (1,) * dimensions, (1,) * (dimensions - 1)
    case.trajectory = frames
    # The source is zero. The completed detector point is x=1, so admission
    # must accept it even though casting its two x components first gives zero.
    with _native_metadata(), _WorldPointAdds(case) as trace:
        Projector(frames, case.shape, case.detector, beam=beam,
                  detector_surface=surface, backend="numba", devices=[0])
    assert trace.records
    assert all(dtype == torch.float64 for _, _, dtype in trace.records), \
        "native admission recomposed promoted geometry after an early FP32 cast"


class _ConstructorAllocationLimit(_WorldPointAdds):
    def __init__(self, case, limit):
        super().__init__(case)
        self.limit = limit

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        count = len(self.records)
        result = super().__torch_dispatch__(function, types, args, kwargs)
        if len(self.records) != count:
            required = result.numel() * result.element_size()
            assert required <= self.limit, (
                "constructor expanded a surface batch before memory admission",
                tuple(result.shape), required, self.limit,
            )
        return result


@pytest.mark.parametrize("kind", ["legacy", "parameterized"])
@pytest.mark.parametrize("limit_kind", ["available", "allocation", "reservation"])
def test_native_constructor_shrinks_automatic_surface_validation_to_live_memory(kind, limit_kind):
    case = _case("cone", views=9)
    case.detector = (15, 13)
    radius = torch.tensor(3.7, dtype=torch.float64)
    covered = set()

    def callback(u, v):
        return _arc(u, v, radius)

    def sampler(u, v, ids, value):
        i = (u / case.pitch[0] + case.detector[0] / 2 - .5).round().long()
        j = (v / case.pitch[1] + case.detector[1] / 2 - .5).round().long()
        covered.update((view, row, column) for view in ids.tolist()
                       for row, column in zip(i.flatten().tolist(), j.flatten().tolist()))
        return _arc(u, v, value)

    surface = callback if kind == "legacy" else ParameterizedSurface(sampler, (radius,))
    limit = 4096
    with _native_metadata(**{limit_kind: limit}), _ConstructorAllocationLimit(case, limit) as trace:
        operator = _projector(case, surface, backend="numba", devices=[0])
    assert operator.projection_shape == case.sino_shape
    assert trace.records and all(dtype == torch.float64 for _, _, dtype in trace.records)
    if kind == "parameterized":
        assert covered == set(np.ndindex(case.sino_shape)), "memory sizing skipped detector pixels or views"


@pytest.mark.parametrize("limit_kind", ["available", "allocation", "reservation"])
def test_native_constructor_rejects_unfit_minimum_before_expanding_world_points(limit_kind):
    case = _case("cone", views=1)
    surface = ParameterizedSurface(lambda u, v, ids: _arc(u, v, torch.tensor(3.7)))
    with _native_metadata(**{limit_kind: 1}), _WorldPointAdds(case) as trace, \
            pytest.raises(RuntimeError, match="memory|budget|minimum|fit|limit"):
        _projector(case, surface, backend="numba", devices=[0])
    assert not trace.records, "constructor expanded points before rejecting an impossible memory budget"


@pytest.mark.cuda
@CUDA_REQUIRED
@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("kind", ["legacy", "parameterized"])
@pytest.mark.parametrize("layout", ["double-offsets", "double-frames"])
def test_native_promoted_cancellation_values_adjoint_data_and_geometry_gradients(beam, kind, layout):
    frame_dtype = torch.float32 if layout == "double-offsets" else torch.float64
    offset_dtype = torch.float64 if layout == "double-offsets" else torch.float32
    frames, surface, normal, tangent, vertical = _cancellation_case(
        beam, kind, frame_dtype=frame_dtype, offset_dtype=offset_dtype, endpoint=.25, learnable=True,
    )
    dimensions = 3 if beam == "cone" else 2
    shape, detector = (1,) * dimensions, (1,) * (dimensions - 1)
    operator = Projector(frames, shape, detector, beam=beam, detector_surface=surface,
                         backend="numba", devices=[0], view_chunk_size=1, schedule="spatial")
    # Both endpoints lie inside the unit voxel. The complete segment therefore
    # contributes its Euclidean length, with no voxel boundary approximation.
    point = np.array([.25, np.float32(tangent) if offset_dtype == torch.float32 else tangent]
                     + ([np.float32(vertical) if offset_dtype == torch.float32 else vertical] if beam == "cone" else []))
    length = np.linalg.norm(point)
    direction = point / length
    du, dv, n = _numpy(frames[2])[0], (_numpy(frames[3])[0] if beam == "cone" else None), float(normal.detach())
    frame_derivatives = [-direction, direction]
    if beam == "cone":
        frame_derivatives += [tangent * direction + n * np.cross(dv, direction),
                              vertical * direction + n * np.cross(direction, du)]
    else:
        frame_derivatives.append(np.array([tangent * direction[0] - n * direction[1],
                                           tangent * direction[1] + n * direction[0]]))
    image = torch.full(shape, 1.25, device="cuda", requires_grad=True)
    rays = torch.full(operator.projection_shape, 1.5, device="cuda", requires_grad=True)
    projected, backprojected = operator.project(image), operator.backproject(rays)
    _close(projected, np.full(operator.projection_shape, 1.25 * length), rtol=2e-6, atol=2e-7)
    _close(backprojected, np.full(shape, 1.5 * length), rtol=2e-6, atol=2e-7)
    torch.testing.assert_close((projected * rays).sum(), (image * backprojected).sum(), rtol=2e-6, atol=2e-7)
    for output, data, weight in ((projected, image, rays), (backprojected, rays, image)):
        gradients = torch.autograd.grad((output * weight.detach()).sum(), (data, *frames, normal))
        _close(gradients[0], np.full(data.shape, float(weight.detach().item()) * length), rtol=2e-6, atol=2e-7)
        factor = 1.25 * 1.5
        for actual, expected in zip(gradients[1:-1], frame_derivatives):
            _close(actual, (factor * expected)[None], rtol=3e-5, atol=2e-5)
        assert gradients[-1].item() == pytest.approx(factor * direction[0], rel=3e-5, abs=2e-5)
        assert all(gradient.dtype == original.dtype and gradient.device == original.device
                   for gradient, original in zip(gradients, (data, *frames, normal)))


class _NoDuplicateCudaOffsets(_WorldPointAdds):
    def __init__(self, case, offsets):
        super().__init__(case)
        self.offset_shape = tuple(offsets.shape)
        self.original_storage = offsets.untyped_storage()._cdata

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = super().__torch_dispatch__(function, types, args, kwargs)
        for value in tree_flatten(result)[0]:
            if (isinstance(value, torch.Tensor) and value.is_cuda and torch.is_floating_point(value)
                    and tuple(value.shape) == self.offset_shape):
                assert value.untyped_storage()._cdata == self.original_storage, (
                    "native forward duplicated complete CUDA offsets outside the tile budget", str(function),
                )
        return result


@pytest.mark.cuda
@CUDA_REQUIRED
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_cuda_per_view_offsets_use_host_snapshot_and_survive_post_forward_mutation(operation):
    case = _case("cone", views=5, learnable=True)
    offsets_np = _arc_offsets(case, np.linspace(3.2, 4.1, case.views), .037)
    offsets = torch.tensor(offsets_np, device="cuda", dtype=torch.float64, requires_grad=True)
    operator = _projector(
        case, lambda u, v: offsets, backend="numba", devices=[0], view_chunk_size=2,
        detector_chunk_shape=(4, 3), volume_chunk_shape=(3, 4, 5), schedule="spatial",
    )
    image, rays = _data(case, device="cpu")
    source, weight = (image, rays) if operation == "project" else (rays, image)
    source = source.requires_grad_()
    frames_np = tuple(_numpy(value) for value in case.trajectory)
    matrix = _matrix(case, offsets_np, frames_np)
    linear = matrix if operation == "project" else matrix.T
    snapshot_geometry = operator._effective_geometry

    def host_snapshot(*args, **kwargs):
        spec, values = snapshot_geometry(*args, **kwargs)
        assert all(value.device.type == "cpu" for value in values), \
            "native surface snapshots must use host storage"
        assert values[-1].requires_grad and values[-1].grad_fn is not None, \
            "offset snapshot lost the caller's autograd connection"
        return spec, values

    with _NoDuplicateCudaOffsets(case, offsets) as trace, \
            mock.patch.object(operator, "_effective_geometry", side_effect=host_snapshot):
        output = getattr(operator, operation)(source)
    assert trace.records and all(device == torch.device("cuda:0") and dtype == torch.float64
                                 and shape[0] <= 2 and shape[1] <= 4 and shape[2] <= 3
                                 for shape, device, dtype in trace.records)
    _close(output, (linear @ _numpy(source).ravel()).reshape(weight.shape))
    with torch.no_grad():
        offsets.add_(.19)
        case.trajectory[1].add_(.071)
    gradients = torch.autograd.grad((output.double() * weight.double()).sum(), (source, offsets, *case.trajectory))
    _close(gradients[0], (linear.T @ _numpy(weight).ravel()).reshape(source.shape))
    image_np, rays_np = _numpy(image).ravel(), _numpy(rays).ravel()
    generator = np.random.default_rng(1019)
    for index, (value, gradient) in enumerate(zip((offsets_np, *frames_np), gradients[1:])):
        direction = generator.normal(size=value.shape)
        direction /= np.linalg.norm(direction)

        def objective(changed, index=index):
            geometry, local = list(frames_np), offsets_np
            if index == 0:
                local = changed
            else:
                geometry[index - 1] = changed
            return float(rays_np @ _matrix(case, local, geometry) @ image_np)

        expected = _finite_difference(objective, value, direction)
        assert abs(expected) > 1e-3
        actual = float((_numpy(gradient) * direction).sum())
        assert actual == pytest.approx(expected, rel=5e-3, abs=4e-4)
    assert all(gradient.dtype == original.dtype and gradient.device == original.device
               for gradient, original in zip(gradients, (source, offsets, *case.trajectory)))
