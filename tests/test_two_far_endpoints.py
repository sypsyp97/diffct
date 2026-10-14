"""Accepted fan/cone rays need accurate weights when both endpoints are far."""

import math

import pytest
import torch

from diffct import Projector
from tests.test_reference_backend import _Case, _matrix


@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("layout", ["forward-whole", "reverse-anisotropic-tiled"])
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_both_far_endpoints_match_exact_axis_chord_and_unit_voxel_weights(
    beam, precision, layout, operation,
):
    shape = (3, 5, 7) if beam == "cone" else (5, 7)
    dimensions = len(shape)
    dtype = torch.float64 if precision == "float64" else torch.float32
    tiled = layout == "reverse-anisotropic-tiled"
    spacing = ((2.0, 1.5, .75) if beam == "cone" else (1.5, .75)) if tiled else 1.0
    x_pitch = spacing[-1] if isinstance(spacing, tuple) else spacing
    source = torch.zeros((1, dimensions), dtype=torch.float64)
    detector, u = torch.zeros_like(source), torch.zeros_like(source)
    # Equality is accepted by the 1e6-voxel near-endpoint guard.
    source[0, 0] = (1 if tiled else -1) * 1e6 * x_pitch
    detector[0, 0], u[0, 1] = -source[0, 0], 1.0
    trajectory = (source, detector, u)
    if beam == "cone":
        v = torch.zeros_like(source)
        v[0, 2] = 1.0
        trajectory += (v,)
    options = {"volume_chunk_shape": (2,) * dimensions} if tiled else {}
    operator = Projector(trajectory, shape, (1, 1) if beam == "cone" else (1,),
                         beam=beam, voxel_spacing=spacing, backend="torch",
                         precision=precision, distributed=False, **options)
    if operation == "project":
        actual = operator.project(torch.ones(shape, dtype=dtype))
        expected = torch.full_like(actual, shape[-1] * x_pitch)
    else:
        actual = operator.backproject(torch.ones(operator.projection_shape, dtype=dtype))
        expected = torch.zeros_like(actual)
        center_line = tuple(size // 2 for size in shape[:-1]) + (slice(None),)
        expected[center_line] = x_pitch
    # These axis oracles use physical grid dimensions only; no traversal or
    # second projector computes either the total chord or any voxel weight.
    tolerance = 2e-10 if precision == "float64" else 4e-5
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_oblique_two_far_endpoint_ray_matches_independent_cell_intersections(beam, precision):
    dtype = torch.float64 if precision == "float64" else torch.float32
    shape = (3, 5, 7) if beam == "cone" else (5, 7)
    direction = torch.tensor([1.0, .37, .19] if beam == "cone" else [1.0, .37], dtype=dtype)
    direction = direction / torch.linalg.vector_norm(direction)
    source, detector = (-9e5 * direction)[None], (9e5 * direction)[None]
    u = torch.tensor([[-direction[1], direction[0], 0.0]] if beam == "cone"
                     else [[-direction[1], direction[0]]], dtype=dtype)
    u = u / torch.linalg.vector_norm(u)
    trajectory = (source, detector, u)
    if beam == "cone":
        trajectory += (torch.linalg.cross(direction[None], u, dim=-1),)
    detector_shape = (1, 1) if beam == "cone" else (1,)
    case = _Case(beam, trajectory, shape, detector_shape, (1.0,) * (len(shape) - 1), 1.0)
    # Each matrix entry independently clips one voxel box in Python float64.
    # It neither walks the production ray-plane intervals nor invokes DiffCT.
    matrix = _matrix(case)
    assert matrix.max() > 0 and torch.count_nonzero(matrix) > shape[-1]
    operator = Projector(trajectory, shape, detector_shape, beam=beam, backend="torch",
                         precision=precision, distributed=False,
                         volume_chunk_shape=(2,) * len(shape))
    image = (torch.arange(math.prod(shape), dtype=dtype).sin() + .4).reshape(shape)
    ray_weight = torch.full(case.projection_shape, 1.5, dtype=dtype)
    expected_forward = (matrix @ image.double().flatten()).reshape(case.projection_shape)
    expected_adjoint = (matrix.T @ ray_weight.double().flatten()).reshape(shape)
    tolerance = 2e-9 if precision == "float64" else 4e-5
    torch.testing.assert_close(operator.project(image).double(), expected_forward,
                               rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(operator.backproject(ray_weight).double(), expected_adjoint,
                               rtol=tolerance, atol=tolerance)
