"""Public detector-surface contract; independent float64 cell-length oracle.

The oracle intersects each voxel's box with each ray in NumPy. It neither
calls DiffCT nor derives a reference from another production projector.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import math

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffct import Projector


BEAMS = ("parallel", "fan", "cone")


@dataclass
class _Case:
    beam: str
    trajectory: tuple[torch.Tensor, ...]
    shape: tuple[int, ...]
    detector: tuple[int, ...]
    pitch: tuple[float, ...]
    spacing: float = 0.73

    @property
    def views(self):
        return self.trajectory[0].shape[0]

    @property
    def sino_shape(self):
        return (self.views, *self.detector)


def _case(beam, views=3, *, learnable=False):
    angles = 0.231 + torch.arange(views, dtype=torch.float64) * 0.713
    radial = torch.stack((angles.cos(), angles.sin()), dim=-1)
    tangent = torch.stack((-angles.sin(), angles.cos()), dim=-1)
    offset = torch.tensor([0.173, -0.239], dtype=torch.float64)
    if beam == "parallel":
        geometry = (radial, offset.expand_as(radial).clone(), tangent)
    elif beam == "fan":
        geometry = (-9 * radial + offset, 6 * radial + offset, tangent)
    else:
        tilt = 0.197
        radial = torch.cat((radial * math.cos(tilt),
                            torch.full((views, 1), math.sin(tilt))), dim=-1)
        tangent = torch.cat((tangent, torch.zeros(views, 1)), dim=-1)
        vertical = torch.linalg.cross(radial, tangent, dim=-1)
        offset = torch.cat((offset.expand(views, 2),
                            torch.linspace(-0.317, 0.413, views)[:, None]), dim=-1)
        geometry = (-9 * radial + offset, 6 * radial + offset,
                    tangent, vertical)
    # cross(det_u, det_v), or (det_u_y, -det_u_x), points away from
    # the source here. Positive cylindrical sag increases the ray length.
    geometry = tuple(g.clone().requires_grad_(learnable) for g in geometry)
    return _Case(beam, geometry, (5, 7, 8) if beam == "cone" else (7, 9),
                 (6, 5) if beam == "cone" else (8,),
                 (0.91, 0.67) if beam == "cone" else (0.91,))


def _axis_case(beam):
    case = _case(beam)
    dimensions = 3 if beam == "cone" else 2
    source = torch.zeros(case.views, dimensions, dtype=torch.float64)
    center, u = source.clone(), source.clone()
    source[:, 0], center[:, 0], u[:, 1] = -9.0, 6.0, 1.0
    if beam == "parallel":
        source[:, 0] = 1.0
        geometry = (source, center, u)
    elif beam == "fan":
        geometry = (source, center, u)
    else:
        v = torch.zeros_like(u)
        v[:, 2] = 1.0
        geometry = (source, center, u, v)
    case.trajectory = geometry
    return case


def _projector(case, surface=None, **kwargs):
    return Projector(
        case.trajectory, case.shape, case.detector, beam=case.beam,
        detector_spacing=case.pitch if case.beam == "cone" else case.pitch[0],
        voxel_spacing=case.spacing, detector_surface=surface, **kwargs,
    )


def _flat_surface(u, v):
    return torch.stack((u, v, torch.zeros_like(u)), dim=-1)


def _arc_surface(radius, coefficient=0.0):
    def surface(u, v):
        u, v = u.double(), v.double()
        r = torch.as_tensor(radius, device=u.device, dtype=u.dtype)
        c = torch.as_tensor(coefficient, device=u.device, dtype=u.dtype)
        if r.ndim:
            r = r.reshape(-1, *([1] * u.ndim))
        if c.ndim:
            c = c.reshape(-1, *([1] * u.ndim))
        x = r * torch.sin(u / r) + c * u.square()
        return torch.stack((x, v + torch.zeros_like(x),
                            r * (1 - torch.cos(u / r))), dim=-1)
    return surface


def _grids(case):
    u = (np.arange(case.detector[0], dtype=np.float64) + 0.5
         - case.detector[0] / 2) * case.pitch[0]
    if case.beam == "cone":
        v = (np.arange(case.detector[1], dtype=np.float64) + 0.5
             - case.detector[1] / 2) * case.pitch[1]
        return np.meshgrid(u, v, indexing="ij")
    return u, np.zeros_like(u)


def _arc_offsets(case, radius=3.7, coefficient=0.0):
    u, v = _grids(case)
    r, c = np.asarray(radius, dtype=np.float64), np.asarray(coefficient,
                                                          dtype=np.float64)
    if r.ndim:
        r = r.reshape(-1, *([1] * u.ndim))
    if c.ndim:
        c = c.reshape(-1, *([1] * u.ndim))
    x = r * np.sin(u / r) + c * u ** 2
    return np.stack((x, np.broadcast_to(v, x.shape),
                     r * (1 - np.cos(u / r))), axis=-1)


def _numpy(tensor):
    return tensor.detach().cpu().double().numpy().copy()


def _cell_lengths(shape, spacing, origin, direction, *, segment):
    """Exact Siddon cell lengths via independent voxel slab intersections."""
    # Arrays are [z,] y, x; physical geometry is x, y[, z].
    indices = np.indices(shape).reshape(len(shape), -1)[::-1].T
    sizes = np.asarray(shape[::-1], dtype=np.float64)
    lower = (indices - sizes / 2) * spacing
    upper = lower + spacing
    enter = np.full(len(indices), 0.0 if segment else -np.inf)
    leave = np.full(len(indices), 1.0 if segment else np.inf)
    for axis, delta in enumerate(direction):
        if delta == 0:
            inside = ((lower[:, axis] <= origin[axis])
                      & (origin[axis] < upper[:, axis]))
            leave[~inside] = -np.inf
        else:
            a = (lower[:, axis] - origin[axis]) / delta
            b = (upper[:, axis] - origin[axis]) / delta
            enter = np.maximum(enter, np.minimum(a, b))
            leave = np.minimum(leave, np.maximum(a, b))
    return np.maximum(leave - enter, 0) * np.linalg.norm(direction)


def _matrix(case, offsets, geometry=None):
    geometry = (tuple(_numpy(g) for g in case.trajectory) if geometry is None
                else tuple(np.asarray(g, dtype=np.float64) for g in geometry))
    offsets = np.broadcast_to(offsets, (*case.sino_shape, 3))
    matrix = np.empty((math.prod(case.sino_shape), math.prod(case.shape)),
                      dtype=np.float64)
    for row, index in enumerate(np.ndindex(case.sino_shape)):
        view = index[0]
        du = geometry[2][view]
        normal = (np.cross(du, geometry[3][view]) if case.beam == "cone"
                  else np.array([du[1], -du[0]]))
        local = offsets[index]
        point = geometry[1][view] + local[0] * du + local[2] * normal
        if case.beam == "cone":
            point = point + local[1] * geometry[3][view]
        origin = point if case.beam == "parallel" else geometry[0][view]
        direction = (geometry[0][view] if case.beam == "parallel"
                     else point - origin)
        matrix[row] = _cell_lengths(case.shape, case.spacing, origin, direction,
                                    segment=case.beam != "parallel")
    return matrix


def _data(case, device="cuda:0"):
    generator = np.random.default_rng(416)
    # Signed, asymmetric, cell-constant data makes axis/centering errors visible.
    image = torch.tensor(generator.normal(size=case.shape), dtype=torch.float32,
                         device=device)
    sino = torch.tensor(generator.normal(size=case.sino_shape),
                        dtype=torch.float32, device=device)
    return image, sino


def _close(actual, expected, *, rtol=4e-4, atol=6e-5):
    torch.testing.assert_close(actual.detach().cpu().double(),
                               torch.as_tensor(expected, dtype=torch.float64),
                               rtol=rtol, atol=atol)


def _finite_difference(function, value, direction=1.0):
    # Two step sizes verify that these rays stay in the same smooth cell regime.
    h = 2e-5
    full = (function(value + h * direction) - function(value - h * direction)) / (2 * h)
    half = (function(value + h / 2 * direction)
            - function(value - h / 2 * direction)) / h
    np.testing.assert_allclose(full, half, rtol=3e-6, atol=3e-7,
                               err_msg="finite differences cross a voxel edge")
    return half


def _objective(case, image, sino, radius=3.7, coefficient=0.037, geometry=None):
    matrix = _matrix(case, _arc_offsets(case, radius, coefficient), geometry)
    return float(sino.reshape(-1) @ matrix @ image.reshape(-1))


def test_cell_length_oracle_hand_computed_segment_and_line():
    image = np.array([[1., 2., 4.], [8., 16., 32.]])
    origin, direction = np.array([-2., 0.21]), np.array([1.75, 0.])
    lengths = _cell_lengths(image.shape, 0.7, origin, direction, segment=True)
    # Segment ends at x=-0.25: 0.7 of the first cell and 0.1 of the second.
    assert lengths @ image.ravel() == pytest.approx(0.7 * 8 + 0.1 * 16)
    lengths = _cell_lengths(image.shape, 0.7, origin, np.array([1., 0.]),
                            segment=False)
    assert lengths @ image.ravel() == pytest.approx(0.7 * (8 + 16 + 32))
    miss = _cell_lengths(image.shape, 0.7, np.array([0., 1.]),
                         np.array([1., 0.]), segment=False)
    assert not miss.any()


def test_cell_length_oracle_xyz_to_zyx_and_inside_source():
    image = np.arange(24, dtype=np.float64).reshape(2, 3, 4) + 1
    lengths = _cell_lengths(image.shape, 0.65, np.array([-2., 0.13, -0.21]),
                            np.array([1., 0., 0.]), segment=False)
    assert lengths @ image.ravel() == pytest.approx(image[0, 1].sum() * 0.65)
    lengths = _cell_lengths(image.shape, 0.65, np.array([0.13, 0.13, -0.21]),
                            np.array([2., 0., 0.]), segment=True)
    assert lengths @ image.ravel() == pytest.approx(0.52 * image[0, 1, 2]
                                                  + 0.65 * image[0, 1, 3])


@pytest.mark.parametrize("surface", [7, {}, torch.zeros(1)],
                         ids=["number", "mapping", "tensor"])
def test_surface_must_be_callable(surface):
    with pytest.raises(TypeError, match="callable|surface"):
        _projector(_case("fan"), surface)


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("kind", ["not-tensor", "integer", "complex", "nan", "inf"])
def test_cpu_rejects_surface_type_dtype_and_nonfinite_values(beam, kind):
    def surface(u, v):
        offsets = _flat_surface(u, v)
        if kind == "not-tensor":
            return offsets.tolist()
        if kind == "integer":
            return offsets.to(torch.int64)
        if kind == "complex":
            return offsets.to(torch.complex64)
        offsets = offsets.clone()
        offsets.reshape(-1, 3)[-1, 2] = float(kind)
        return offsets

    exception = ValueError if kind in ("nan", "inf") else TypeError
    with pytest.raises(exception, match="tensor|float|dtype|finite|surface"):
        _projector(_case(beam), surface)


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("kind", ["components", "missing-axis", "singleton-view", "wrong-views"])
def test_cpu_rejects_inexact_surface_shapes(beam, kind):
    def surface(u, v):
        offsets = _flat_surface(u, v)
        if kind == "components":
            return offsets[..., :2]
        if kind == "missing-axis":
            return offsets[0]
        return offsets.unsqueeze(0).expand(1 if kind == "singleton-view" else 4,
                                          *offsets.shape)

    with pytest.raises(ValueError, match="shape"):
        _projector(_case(beam), surface)


@pytest.mark.parametrize("beam", ["parallel", "fan"])
@pytest.mark.parametrize("middle", [0.125, 1e-12])
def test_cpu_rejects_any_nonzero_middle_offset_in_2d(beam, middle):
    def surface(u, v):
        offsets = _flat_surface(u, v)
        offsets[-1, 1] = middle
        return offsets

    with pytest.raises(ValueError, match="2D|middle|zero|v.*component"):
        _projector(_case(beam), surface)


def _bad_endpoint_surface(case, kind):
    def surface(u, v):
        offsets = _flat_surface(u, v).double().unsqueeze(0).expand(
            case.views, *u.shape, 3).clone()
        pixel = (1, *([0] * len(case.detector)))
        if kind == "coincident":
            # Axis-aligned frame makes this an exactly coincident endpoint.
            offsets[pixel] = torch.tensor([0., 0., -15.], device=u.device)
        else:
            offsets[pixel][2] = (1.5e6 if kind == "soft" else 2e15) * case.spacing
        return offsets
    return surface


@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_cpu_rejects_one_coincident_source_and_surface_pixel(beam):
    case = _axis_case(beam)
    with pytest.raises(ValueError, match="coincid|source.*(pixel|detector)|differ"):
        _projector(case, _bad_endpoint_surface(case, "coincident"))


@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("kind", ["soft", "hard"])
def test_cpu_distance_bounds_use_actual_surface_pixels(beam, kind):
    case = _axis_case(beam)
    if kind == "soft":
        # The center is near, but this source and one actual pixel are both far.
        case.trajectory[0][1, 0] = -1.5e6 * case.spacing
    with pytest.raises(ValueError, match="1e6" if kind == "soft" else "1e15"):
        _projector(case, _bad_endpoint_surface(case, kind))


@pytest.mark.parametrize("beam", BEAMS)
def test_cpu_accepts_shared_and_exact_per_view_floating_surface_shapes(beam):
    case = _case(beam)
    for radius in (3.7, torch.tensor([3.2, 4.1, 5.3], dtype=torch.float64)):
        projector = _projector(case, _arc_surface(radius))
        assert projector.projection_shape == case.sino_shape


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flat_surface_preserves_default_project_and_backproject(beam, dtype):
    case = _case(beam)
    image, sino = _data(case)
    default = _projector(case)
    flat = _projector(case, lambda u, v: _flat_surface(u, v).to(dtype))
    for operation, data in (("project", image), ("backproject", sino)):
        actual = getattr(flat, operation)(data.double())
        expected = getattr(default, operation)(data.double())
        assert actual.device == data.device and actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
    # Control: independently check the original flat convention before curved RED.
    u, v = _grids(case)
    matrix = _matrix(case, np.stack((u, v, np.zeros_like(u)), axis=-1))
    _close(default.project(image), (matrix @ _numpy(image).ravel()).reshape(case.sino_shape))


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_centered_physical_ij_grids_and_fresh_sampling(beam, operation):
    case = _case(beam)
    samples = []

    def surface(u, v):
        assert isinstance(u, torch.Tensor) and torch.is_floating_point(u)
        assert isinstance(v, torch.Tensor) and torch.is_floating_point(v)
        samples.append((_numpy(u), _numpy(v)))
        return _flat_surface(u, v)

    projector = _projector(case, surface)
    image, sino = _data(case)
    data = image if operation == "project" else sino
    for _ in range(2):
        before = len(samples)
        getattr(projector, operation)(data)
        assert len(samples) > before, "detector_surface was ignored or cached"
        for actual_u, actual_v in samples[before:]:
            expected_u, expected_v = _grids(case)
            np.testing.assert_allclose(actual_u, expected_u, rtol=2e-7, atol=2e-7)
            np.testing.assert_allclose(actual_v, expected_v, rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("per_view", [False, True], ids=["shared", "per-view"])
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_curved_projection_and_adjoint_match_independent_cell_oracle(beam, per_view):
    case = _case(beam)
    radius = np.array([2.9, 4.2, 5.1]) if per_view else 3.7
    coefficient = np.array([0.029, -0.043, 0.017]) if per_view else 0.037
    projector = _projector(case, _arc_surface(radius, coefficient))
    image, sino = _data(case)
    matrix = _matrix(case, _arc_offsets(case, radius, coefficient))
    projected, backprojected = projector.project(image), projector.backproject(sino)
    assert tuple(projected.shape) == case.sino_shape
    assert tuple(backprojected.shape) == case.shape
    for output in (projected, backprojected):
        assert output.is_cuda and output.dtype == torch.float32
    _close(projected, (matrix @ _numpy(image).ravel()).reshape(case.sino_shape))
    _close(backprojected, (matrix.T @ _numpy(sino).ravel()).reshape(case.shape))


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_curved_adjoint_inner_product(beam):
    case = _case(beam)
    projector = _projector(case, _arc_surface(3.7, 0.037))
    image, sino = _data(case)
    lhs = (projector.project(image).double() * sino.double()).sum()
    rhs = (image.double() * projector.backproject(sino).double()).sum()
    torch.testing.assert_close(lhs, rhs, rtol=3e-5, atol=5e-5)


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_curved_data_gradient_and_second_derivative_use_true_endpoints(beam, operation):
    case = _case(beam)
    # Data Hessians remain supported with trainable surface geometry present.
    radius = torch.tensor(3.7, dtype=torch.float64, requires_grad=True)
    projector = _projector(case, _arc_surface(radius, 0.037))
    image, sino = _data(case)
    data = (image if operation == "project" else sino).double().requires_grad_()
    direction = torch.linspace(-0.4, 0.7, data.numel(), device=data.device,
                               dtype=data.dtype).reshape_as(data)
    output = getattr(projector, operation)(data)
    gradient, = torch.autograd.grad(output.double().square().sum() / 2, data,
                                    create_graph=True)
    hessian_vector, = torch.autograd.grad((gradient * direction).sum(), data)
    assert gradient.dtype == data.dtype and gradient.device == data.device
    matrix = _matrix(case, _arc_offsets(case, 3.7, 0.037))
    operator = matrix if operation == "project" else matrix.T
    normal = operator.T @ operator
    _close(gradient, (normal @ _numpy(data).ravel()).reshape(data.shape))
    _close(hessian_vector, (normal @ _numpy(direction).ravel()).reshape(data.shape))


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_curve_and_each_trajectory_component_gradients_against_independent_fd(beam, operation):
    case = _case(beam, learnable=True)
    radius = torch.tensor(3.7, dtype=torch.float64, requires_grad=True)
    coefficient = torch.tensor(0.037, dtype=torch.float64, requires_grad=True)
    projector = _projector(case, _arc_surface(radius, coefficient))
    image, sino = _data(case)
    data, weight = (image, sino) if operation == "project" else (sino, image)
    data = data.clone().requires_grad_()
    loss = (getattr(projector, operation)(data).double() * weight.double()).sum()
    parameters = (radius, coefficient, *case.trajectory)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    for name, gradient in zip(("radius", "coefficient", "source/direction", "center",
                               "detector_u", "detector_v"), gradients):
        assert gradient is not None, f"detector surface lost the {name} gradient"
        assert torch.isfinite(gradient).all()
    image_np, sino_np = _numpy(image), _numpy(sino)
    expected_r = _finite_difference(
        lambda r: _objective(case, image_np, sino_np, radius=r), 3.7)
    expected_c = _finite_difference(
        lambda c: _objective(case, image_np, sino_np, coefficient=c), 0.037)
    for actual, expected in ((gradients[0].item(), expected_r),
                              (gradients[1].item(), expected_c)):
        assert abs(expected) > 1e-3, "uninformative finite-difference fixture"
        assert actual == pytest.approx(expected, rel=5e-3, abs=3e-4)
    geometry = tuple(_numpy(g) for g in case.trajectory)
    generator = np.random.default_rng(922)
    for component, gradient in enumerate(gradients[2:]):
        direction = generator.normal(size=geometry[component].shape)
        if case.beam == "parallel" and component == 0:
            # Unit ray directions have tangent derivatives; avoid imposing an
            # unspecified nonunit parallel-ray length convention on the API.
            direction -= (direction * geometry[component]).sum(
                axis=-1, keepdims=True) * geometry[component]
        direction /= np.linalg.norm(direction)

        def value(perturbed):
            changed = list(geometry)
            changed[component] = perturbed
            return _objective(case, image_np, sino_np, geometry=changed)

        expected = _finite_difference(value, geometry[component], direction)
        actual = float((_numpy(gradient) * direction).sum())
        assert abs(expected) > 1e-3, "uninformative trajectory gradient direction"
        assert actual == pytest.approx(expected, rel=5e-3, abs=3e-4)


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_repeated_backward_reads_updated_radius_and_learnable_trajectory(beam, operation):
    case = _case(beam)
    case.trajectory[1].requires_grad_()
    radius = torch.tensor(3.7, dtype=torch.float64, requires_grad=True)
    projector = _projector(case, _arc_surface(radius, 0.037))
    image, sino = _data(case)
    data, weight = (image, sino) if operation == "project" else (sino, image)
    previous = None
    for current_radius in (3.7, 4.3, 3.1):
        with torch.no_grad():
            radius.fill_(current_radius)
            case.trajectory[1][:, 0].add_(0.083)
        radius.grad = case.trajectory[1].grad = None
        output = getattr(projector, operation)(data.clone().requires_grad_())
        (output.double() * weight.double()).sum().backward()
        assert radius.grad is not None, "captured radius disconnected from backward"
        assert case.trajectory[1].grad is not None
        assert torch.isfinite(case.trajectory[1].grad).all()
        expected_gradient = _finite_difference(
            lambda r: _objective(case, _numpy(image), _numpy(sino), radius=r),
            current_radius)
        assert radius.grad.item() == pytest.approx(expected_gradient, rel=5e-3, abs=3e-4)
        matrix = _matrix(case, _arc_offsets(case, current_radius, 0.037))
        expected = ((matrix @ _numpy(image).ravel()).reshape(case.sino_shape)
                    if operation == "project" else
                    (matrix.T @ _numpy(sino).ravel()).reshape(case.shape))
        _close(output, expected)
        if previous is not None:
            assert not torch.allclose(output, previous), "updated geometry was cached"
        previous = output.detach().clone()


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fixed_original_trajectory_is_frozen_for_curved_surface(beam):
    case = _case(beam)
    projector = _projector(case, _arc_surface(3.7, 0.037))
    matrix = _matrix(case, _arc_offsets(case, 3.7, 0.037))
    image, sino = _data(case)
    before_project = projector.project(image)
    before_backproject = projector.backproject(sino)
    # Mutate the original tensors after staging once; fixed geometry stays frozen.
    case.trajectory[1][:, 0].add_(0.61)
    if beam != "parallel":
        case.trajectory[0][:, 1].sub_(0.43)
    after_project, after_backproject = projector.project(image), projector.backproject(sino)
    torch.testing.assert_close(after_project, before_project, rtol=0, atol=0)
    torch.testing.assert_close(after_backproject, before_backproject, rtol=3e-5, atol=3e-5)
    _close(after_project, (matrix @ _numpy(image).ravel()).reshape(case.sino_shape))
    _close(after_backproject, (matrix.T @ _numpy(sino).ravel()).reshape(case.shape))


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("parameter", ["radius", "trajectory"])
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_geometry_second_derivative_remains_explicitly_unsupported(beam, operation, parameter):
    case = _case(beam, learnable=True)
    radius = torch.tensor(3.7, dtype=torch.float64, requires_grad=True)
    projector = _projector(case, _arc_surface(radius, 0.037))
    image, sino = _data(case)
    data, weight = (image, sino) if operation == "project" else (sino, image)
    chosen = radius if parameter == "radius" else case.trajectory[1]
    output = getattr(projector, operation)(data)
    first, = torch.autograd.grad((output.double() * weight.double()).sum(), chosen,
                                 create_graph=True, allow_unused=True)
    assert first is not None, "surface geometry was disconnected"
    with pytest.raises(RuntimeError, match="second derivatives.*geometry|geometry.*second derivatives"):
        torch.autograd.grad(first.sum(), chosen)


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cpu_and_mixed_device_geometry_and_cpu_surface_outputs(beam):
    case = _case(beam)
    case.trajectory = tuple(g.to(device="cuda:0" if i % 2 else "cpu",
                                 dtype=torch.float32 if i % 2 else torch.float64)
                            for i, g in enumerate(case.trajectory))

    def cpu_surface(u, v):
        return _arc_surface(3.7, 0.037)(u.cpu(), v.cpu())

    projector = _projector(case, cpu_surface)
    image, sino = _data(case)
    matrix = _matrix(case, _arc_offsets(case, 3.7, 0.037))
    _close(projector.project(image), (matrix @ _numpy(image).ravel()).reshape(case.sino_shape))
    _close(projector.backproject(sino), (matrix.T @ _numpy(sino).ravel()).reshape(case.shape))


@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("kind", ["not-tensor", "integer", "shape", "nan", "middle",
                                  "coincident", "hard"])
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_callback_mutation_is_validated_on_each_later_call(operation, kind):
    case = _axis_case("fan")
    state = {"bad": False}

    def surface(u, v):
        offsets = _flat_surface(u, v)
        if not state["bad"]:
            return offsets
        if kind == "not-tensor":
            return offsets.tolist()
        if kind == "integer":
            return offsets.long()
        if kind == "shape":
            return offsets[..., :2]
        if kind in ("coincident", "hard"):
            return _bad_endpoint_surface(case, kind)(u, v)
        offsets = offsets.clone()
        offsets[-1, 1 if kind == "middle" else 2] = 0.125 if kind == "middle" else math.nan
        return offsets

    projector = _projector(case, surface)
    image, sino = _data(case)
    data = image if operation == "project" else sino
    getattr(projector, operation)(data)
    state["bad"] = True
    exception = TypeError if kind in ("not-tensor", "integer") else ValueError
    with pytest.raises(exception):
        getattr(projector, operation)(data)


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("views", [1, 3], ids=["empty-shard", "uneven-shards"])
@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA GPUs are required")
def test_two_gpu_per_view_surfaces_and_sum_gradients(beam, views):
    case = _case(beam, views)
    radii = 3.2 + np.arange(views, dtype=np.float64) * 0.63
    radius = torch.tensor(radii, requires_grad=True)
    projector = _projector(case, _arc_surface(radius, 0.037), devices=[0, 1])
    matrix = _matrix(case, _arc_offsets(case, radii, 0.037))
    image, sino = _data(case)
    image.requires_grad_()
    projection = projector.project(image)
    _close(projection, (matrix @ _numpy(image).ravel()).reshape(case.sino_shape))
    (projection.double() * sino.double()).sum().backward()
    _close(image.grad, (matrix.T @ _numpy(sino).ravel()).reshape(case.shape))
    assert radius.grad is not None
    direction = np.linspace(0.23, -0.31, views)
    expected = _finite_difference(
        lambda r: _objective(case, _numpy(image), _numpy(sino), radius=r), radii, direction)
    assert float(radius.grad.numpy() @ direction) == pytest.approx(expected, rel=5e-3, abs=3e-4)
    sino.requires_grad_()
    backprojection = projector.backproject(sino)
    _close(backprojection, (matrix.T @ _numpy(sino).ravel()).reshape(case.shape))
    (backprojection.double() * image.detach().double()).sum().backward()
    _close(sino.grad, (matrix @ _numpy(image).ravel()).reshape(case.sino_shape))


def _distributed_surface_worker(rank, beam, views, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        case = _case(beam, views)
        radii = 3.2 + np.arange(views, dtype=np.float64) * 0.63
        radius = torch.tensor(radii, requires_grad=True)
        projector = _projector(case, _arc_surface(radius, 0.037), distributed=True)
        matrix = _matrix(case, _arc_offsets(case, radii, 0.037))
        image, sino = _data(case, device=f"cuda:{rank}")
        start = rank * (views // 2) + min(rank, views % 2)
        stop = start + views // 2 + int(rank < views % 2)
        assert projector.view_slice == slice(start, stop)
        local_matrix = matrix[start * math.prod(case.detector):stop * math.prod(case.detector)]
        local_sino = sino[start:stop].clone()
        image.requires_grad_()
        projection = projector.project(image)
        _close(projection, (local_matrix @ _numpy(image).ravel()).reshape(
            stop - start, *case.detector))
        (projection.double() * local_sino.double()).sum().backward()
        # Replicated image and surface gradients must SUM rank-local view losses.
        _close(image.grad, (matrix.T @ _numpy(sino).ravel()).reshape(case.shape))
        assert radius.grad is not None
        direction = np.linspace(0.23, -0.31, views)
        expected = _finite_difference(
            lambda r: _objective(case, _numpy(image), _numpy(sino), radius=r), radii, direction)
        assert float(radius.grad.numpy() @ direction) == pytest.approx(expected, rel=5e-3, abs=3e-4)
        radius.grad = None
        local_sino.requires_grad_()
        backprojection = projector.backproject(local_sino)
        _close(backprojection, (matrix.T @ _numpy(sino).ravel()).reshape(case.shape))
        # Different cotangents on both ranks expose a missing SUM or extra average.
        weight = image.detach() * (rank + 1) / 2
        (backprojection.double() * weight.double()).sum().backward()
        _close(local_sino.grad, (local_matrix @ (_numpy(image) * 1.5).ravel()).reshape(
            stop - start, *case.detector))
        assert radius.grad is not None
        assert float(radius.grad.numpy() @ direction) == pytest.approx(
            expected * 1.5, rel=5e-3, abs=3e-4)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("views", [1, 3], ids=["empty-rank", "uneven-ranks"])
@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2 or not dist.is_nccl_available(),
                    reason="Two CUDA GPUs and real NCCL are required")
def test_two_gpu_nccl_surface_sharding_and_sum_semantics(beam, views, tmp_path):
    rendezvous = (tmp_path / "surface-nccl").resolve().as_uri()
    mp.spawn(_distributed_surface_worker, args=(beam, views, rendezvous),
             nprocs=2, join=True)
