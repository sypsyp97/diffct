"""Native finite rays retain voxel weights when both endpoints are distant."""

from dataclasses import replace
import math

from numba import config
import pytest
import torch

from diffct import Projector
from tests.test_reference_backend import _Case, _matrix


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.version.hip is not None or config.ENABLE_CUDASIM,
        reason="endpoint precision requires real NVIDIA CUDA",
    ),
]

# Match the existing float32 cell-matrix contract. With at most 15 intersected
# cells, 4e-5 allows traversal/accumulation rounding without accepting the
# approximately 0.03 voxel-weight error caused by distant slab parameters.
_TOLERANCE = 4e-5


def _case(beam, *, radius=9e5, reverse=False):
    shape = (3, 5, 7) if beam == "cone" else (5, 7)
    detector = (1, 1) if beam == "cone" else (1,)
    direction = torch.tensor([1.0, .37, .19] if beam == "cone" else [1.0, .37])
    direction = direction / torch.linalg.vector_norm(direction)
    u = torch.tensor([[-direction[1], direction[0], 0.0]] if beam == "cone"
                     else [[-direction[1], direction[0]]])
    u = u / torch.linalg.vector_norm(u)
    source, center = (-radius * direction)[None], (radius * direction)[None]
    trajectory = (center, source, u) if reverse else (source, center, u)
    if beam == "cone":
        trajectory += (torch.linalg.cross(direction[None], u, dim=-1),)
    return _Case(beam, trajectory, shape, detector, (1.0,) * len(detector), 1.0)


def _operator(case, *, tiled=False):
    options = {"volume_chunk_shape": (2,) * len(case.shape), "schedule": "spatial"} if tiled else {}
    operator = Projector(
        case.trajectory, case.shape, case.detector, beam=case.beam,
        backend="numba", distributed=False, **options,
    )
    assert operator.backend == "numba" and operator.dtype == torch.float32
    return operator


def _image(case):
    return (torch.arange(math.prod(case.shape), dtype=torch.float32).sin() + .4).reshape(case.shape)


def _close(actual, expected):
    torch.testing.assert_close(
        actual.detach().cpu().double(), expected.detach().cpu().double(),
        rtol=_TOLERANCE, atol=_TOLERANCE,
    )


@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_axis_far_endpoints_and_finite_segment_have_exact_chord_and_voxel_weights(beam):
    case = _case(beam)
    dimensions = len(case.shape)
    source = torch.zeros((3, dimensions), dtype=torch.float32)
    center, u = torch.zeros_like(source), torch.zeros_like(source)
    # Two complete rays with opposite directions, then an interior segment.
    # The latter must stop at its endpoints rather than become an infinite line.
    source[:, 0] = torch.tensor([-9e5, 9e5, -.5])
    center[:, 0] = torch.tensor([9e5, -9e5, .75])
    u[:, 1] = 1.0
    trajectory = (source, center, u)
    if beam == "cone":
        v = torch.zeros_like(source)
        v[:, 2] = 1.0
        trajectory += (v,)
    case = replace(case, trajectory=trajectory)
    operator = _operator(case)

    # Pure physical-grid oracles: no projector or slab routine computes these.
    chords = torch.tensor([7.0, 7.0, 1.25]).reshape(case.projection_shape)
    _close(operator.project(torch.ones(case.shape, device="cuda")), chords)
    weights = torch.zeros(case.shape)
    center_line = tuple(size // 2 for size in case.shape[:-1]) + (slice(None),)
    weights[center_line] = torch.tensor([2.0, 2.0, 2.0, 3.0, 2.25, 2.0, 2.0])
    _close(operator.backproject(torch.ones(case.projection_shape, device="cuda")), weights)


@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("layout", ["near", "far", "far-reversed-tiled"])
@pytest.mark.parametrize("operation", ["project", "backproject", "project_into", "backproject_into"])
def test_oblique_endpoint_weights_match_independent_cells(beam, layout, operation):
    tiled = layout == "far-reversed-tiled"
    case = _case(beam, radius=100.0 if layout == "near" else 9e5, reverse=tiled)
    matrix = _matrix(case)
    assert torch.count_nonzero(matrix) > case.shape[-1]
    # These odd volume dimensions and two-cell tiles translate endpoints by
    # multiples of 0.5, exactly representable at all selected FP32 coordinates.
    # Tiling therefore needs no relaxed tolerance for geometry quantization.
    operator = _operator(case, tiled=tiled)
    image = _image(case)
    ray_weight = torch.full(case.projection_shape, 1.5)
    if operation.startswith("project"):
        data = image.cuda()
        expected = (matrix @ image.double().flatten()).reshape(case.projection_shape)
    else:
        data = ray_weight.cuda()
        expected = (matrix.T @ ray_weight.double().flatten()).reshape(case.shape)
    if operation.endswith("_into"):
        output = torch.full(expected.shape, torch.nan, dtype=torch.float32, device="cuda")
        actual = getattr(operator, operation)(data, output)
        assert actual is output
    else:
        actual = getattr(operator, operation)(data)
    assert actual.dtype == torch.float32 and actual.device == data.device
    _close(actual, expected)


@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_far_endpoint_data_gradients_match_independent_adjoint(beam, operation):
    case = _case(beam)
    matrix = _matrix(case)
    operator = _operator(case)
    image = _image(case).cuda().requires_grad_()
    ray_weight = torch.full(case.projection_shape, 1.5, device="cuda", requires_grad=True)
    forward, adjoint = operator.project(image), operator.backproject(ray_weight)
    # The adjoint identity alone would allow matching but inaccurate operators.
    _close((forward.double() * ray_weight.double()).sum(),
           (image.double() * adjoint.double()).sum())
    if operation == "project":
        gradient, = torch.autograd.grad((forward * ray_weight.detach()).sum(), image)
        expected = (matrix.T @ ray_weight.detach().cpu().double().flatten()).reshape(case.shape)
    else:
        gradient, = torch.autograd.grad((adjoint * image.detach()).sum(), ray_weight)
        expected = (matrix @ image.detach().cpu().double().flatten()).reshape(case.projection_shape)
    _close(gradient, expected)


@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_far_endpoint_geometry_vjp_matches_independent_cell_finite_difference(beam, operation):
    case = _case(beam)
    trajectory = tuple(value.clone().requires_grad_() for value in case.trajectory)
    case = replace(case, trajectory=trajectory)
    operator = _operator(case)
    image = _image(case)
    ray_weight = torch.full(case.projection_shape, 1.5)
    if operation == "project":
        loss = (operator.project(image.cuda()) * ray_weight.cuda()).sum()
    else:
        loss = (operator.backproject(ray_weight.cuda()) * image.cuda()).sum()
    gradients = torch.autograd.grad(loss, trajectory)

    # Perturb the exact represented FP32 endpoints in float64. Perturbing FP32
    # tensors themselves would round away these steps at radius 9e5.
    matrix = _matrix(case)
    expected = torch.zeros((2, len(case.shape)), dtype=torch.float64)
    for group in range(2):
        for axis in range(len(case.shape)):
            estimates = []
            for step in (.01, .02):
                losses = []
                for sign in (-1, 1):
                    shifted = [value.detach().double().clone() for value in trajectory]
                    shifted[group][0, axis] += sign * step
                    shifted_matrix = _matrix(replace(case, trajectory=tuple(shifted)))
                    # Every perturbation stays in the same voxel topology.
                    assert torch.equal(shifted_matrix > 0, matrix > 0)
                    losses.append((shifted_matrix @ image.double().flatten()
                                   @ ray_weight.double().flatten()).item())
                estimates.append((losses[1] - losses[0]) / (2 * step))
            # Step agreement guards cancellation and a nonsmooth FD oracle.
            assert math.isclose(*estimates, rel_tol=2e-5, abs_tol=2e-6)
            expected[group, axis] = estimates[0]
    _close(torch.cat(gradients[:2]), expected)
    # A single central detector pixel has no dependence on frame orientations.
    for gradient in gradients[2:]:
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=0)
