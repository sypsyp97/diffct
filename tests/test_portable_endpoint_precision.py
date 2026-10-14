"""Analytic chords for accepted fan/cone segments with a distant endpoint."""

import pytest
import torch

from diffct import Projector


@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("spacing_kind", ["scalar", "per-axis"])
@pytest.mark.parametrize(
    ("source_x", "detector_x"),
    [
        pytest.param(-100.0, 100.0, id="near-control"),
        pytest.param(-1e7, 100.0, id="far-source"),
        pytest.param(-100.0, 1e7, id="far-detector"),
        pytest.param(-1e15, 100.0, id="hard-limit-source"),
        pytest.param(-100.0, 1e15, id="hard-limit-detector"),
    ],
)
def test_portable_endpoint_chord_and_adjoint(
    beam, precision, spacing_kind, source_x, detector_x
):
    shape = (3, 5, 7) if beam == "cone" else (5, 7)
    spacing = 1.0
    if spacing_kind == "per-axis":
        spacing = (2.0, 1.5, .75) if beam == "cone" else (1.5, .75)
    x_pitch = spacing[-1] if isinstance(spacing, tuple) else spacing
    dtype = torch.float64 if precision == "float64" else torch.float32

    # Coordinates are in x-pitch units. x_pitch is the smallest pitch, so the
    # far endpoint stays within 1e15 voxels under every per-axis scaling.
    dimensions = len(shape)
    source = torch.zeros((1, dimensions), dtype=torch.float64)
    detector = torch.zeros_like(source)
    source[0, 0] = source_x * x_pitch
    detector[0, 0] = detector_x * x_pitch
    u = torch.zeros_like(source)
    u[0, 1] = 1.0
    trajectory = (source, detector, u)
    if beam == "cone":
        v = torch.zeros_like(source)
        v[0, 2] = 1.0
        trajectory += (v,)

    operator = Projector(
        trajectory, shape, (1, 1) if beam == "cone" else (1,),
        beam=beam, voxel_spacing=spacing, backend="torch", precision=precision,
        devices=None, distributed=False,
    )
    density, cotangent = 1.25, 1.5
    volume = torch.full(shape, density, dtype=dtype)
    weights = torch.full(operator.projection_shape, cotangent, dtype=dtype)
    projection = operator.project(volume)
    adjoint = operator.backproject(weights)

    # One centred detector pixel gives the x-axis segment through the central
    # voxel row/line. Both endpoints are outside the x faces; the exact chord
    # is seven x pitches. No traversal or geometry helper supplies the oracle.
    expected_projection = torch.full_like(projection, density * shape[-1] * x_pitch)
    expected_adjoint = torch.zeros_like(volume)
    central_line = tuple(size // 2 for size in shape[:-1]) + (slice(None),)
    expected_adjoint[central_line] = cotangent * x_pitch
    tolerance = 2e-11 if precision == "float64" else 4e-5
    assert projection.device.type == adjoint.device.type == "cpu"
    torch.testing.assert_close(projection, expected_projection, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(adjoint, expected_adjoint, rtol=tolerance, atol=tolerance)
