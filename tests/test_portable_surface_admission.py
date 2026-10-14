"""Surface admission must validate the arithmetic used by portable execution."""

import math

import pytest
import torch

from diffct import Projector


def _cancellation_case(beam):
    dimensions = 3 if beam == "cone" else 2
    source = torch.zeros((1, dimensions), dtype=torch.float64)
    center = torch.zeros_like(source)
    center[0, 0] = 1e8
    axis_u = torch.zeros_like(source)
    axis_u[0, 1] = 1.0
    trajectory = (source, center, axis_u)
    if beam == "cone":
        axis_v = torch.zeros_like(source)
        axis_v[0, 2] = 1.0
        trajectory += (axis_v,)
    calls = []

    def surface(u, v):
        calls.append(1)
        return torch.stack((torch.zeros_like(u), torch.zeros_like(v),
                            torch.full_like(u, -1e8 + 1)), dim=-1)

    shape = (1,) * dimensions
    detector_shape = (1,) * (dimensions - 1)
    return trajectory, shape, detector_shape, surface, calls


@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_float32_rejects_surface_point_coincident_after_component_staging(beam):
    trajectory, shape, detector_shape, surface, calls = _cancellation_case(beam)
    # In float64 the pixel is (1, 0[, 0]). Casting its two x components before
    # addition instead gives 1e8 + (-1e8) = 0, exactly the source position.
    with pytest.raises(ValueError, match="differ|coincid"):
        Projector(trajectory, shape, detector_shape, beam=beam,
                  detector_surface=surface, backend="torch", precision="float32")
    assert len(calls) == 1


@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_float64_surface_cancellation_preserves_analytic_chord_and_adjoint(beam):
    trajectory, shape, detector_shape, surface, calls = _cancellation_case(beam)
    operator = Projector(trajectory, shape, detector_shape, beam=beam,
                         detector_surface=surface, backend="torch", precision="float64")
    assert len(calls) == 1
    density, cotangent = 1.25, 1.5
    image = torch.full(shape, density, dtype=torch.float64)
    weight = torch.full(operator.projection_shape, cotangent, dtype=torch.float64)
    projection = operator.project(image)
    assert len(calls) == 2
    adjoint = operator.backproject(weight)
    assert len(calls) == 3

    # The segment [0, 1] on x intersects the unit voxel [-.5, .5] for exactly
    # .5 physical units. Its one-entry projection matrix and transpose are .5.
    torch.testing.assert_close(projection, torch.full_like(projection, density * .5),
                               rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(adjoint, torch.full_like(adjoint, cotangent * .5),
                               rtol=2e-11, atol=2e-11)


def test_float32_rejects_surface_point_with_nonfinite_intermediate_arithmetic():
    diagonal = math.sqrt(.5)
    direction = torch.tensor([[-diagonal, diagonal]], dtype=torch.float64)
    center = torch.tensor([[3e38, -1e38]], dtype=torch.float64)
    axis_u = torch.tensor([[diagonal, diagonal]], dtype=torch.float64)

    def surface(u, v):
        return torch.stack((torch.full_like(u, 3e38), torch.zeros_like(v),
                            torch.full_like(u, -3e38)), dim=-1)

    # The completed float64 point (3e38, (6*sqrt(.5) - 1)*1e38) fits float32.
    # Actual float32 center + local_u * axis_u overflows in x before the normal
    # displacement can cancel it. Admission must reject that execution point.
    completed_point = torch.tensor([3e38, (6 * diagonal - 1) * 1e38], dtype=torch.float64)
    assert torch.isfinite(completed_point.float()).all().item()
    with pytest.raises(ValueError, match="finite"):
        Projector((direction, center, axis_u), (1, 1), (1,), beam="parallel",
                  detector_surface=surface, backend="torch", precision="float32")
