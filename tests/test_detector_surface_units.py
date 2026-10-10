"""Surface validation preserves valid changes of physical length units."""

import pytest
import torch

from diffct import Projector


def _arguments(beam, scale):
    dimensions = 3 if beam == "cone" else 2
    source = torch.zeros(1, dimensions, dtype=torch.float64)
    center, det_u = source.clone(), source.clone()
    source[0, 0], center[0, 0], det_u[0, 1] = -scale, scale, 1.0
    trajectory = (source, center, det_u)
    if beam == "cone":
        det_v = torch.zeros_like(source)
        det_v[0, 2] = 1.0
        trajectory += (det_v,)
    return trajectory, (1,) * dimensions, (1, 1) if beam == "cone" else (1,)


def _flat_surface(u, v):
    return torch.stack((u, v, torch.zeros_like(u)), dim=-1)


@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("scale", [1.0, 1e-24], ids=["unit", "tiny"])
def test_distinct_surface_endpoints_preserve_physical_unit_rescaling(beam, scale):
    arguments = _arguments(beam, scale)
    source, center = arguments[0][:2]
    assert not torch.equal(source.float(), center.float())
    kwargs = dict(beam=beam, detector_spacing=scale, voxel_spacing=scale)
    legacy = Projector(*arguments, **kwargs)
    surface = Projector(*arguments, detector_surface=_flat_surface, **kwargs)
    assert surface.projection_shape == legacy.projection_shape

    if torch.cuda.is_available():
        image = torch.ones(arguments[1], device="cuda", dtype=torch.float32)
        sinogram = torch.ones(surface.projection_shape, device=image.device)
        for operation, data in (("project", image), ("backproject", sinogram)):
            actual = getattr(surface, operation)(data)
            flat = getattr(legacy, operation)(data)
            assert actual.device == image.device and actual.dtype == torch.float32
            # The central ray traverses one full unit-box cell: chord = scale.
            # Normalize before comparing; a zero result must never pass.
            actual_units, flat_units = actual.double() / scale, flat.double() / scale
            ones = torch.ones_like(actual_units)
            torch.testing.assert_close(actual_units, flat_units, rtol=3e-5, atol=0)
            torch.testing.assert_close(actual_units, ones, rtol=3e-5, atol=0)
            torch.testing.assert_close(flat_units, ones, rtol=3e-5, atol=0)


@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_distinct_float64_endpoints_that_collapse_in_float32_still_rejected(beam):
    arguments = _arguments(beam, 1.0)
    source, center = arguments[0][:2]
    source[0, 0], center[0, 0] = 1.0, 1.0 + 1e-9
    assert not torch.equal(source, center)
    assert torch.equal(source.float(), center.float())
    with pytest.raises(ValueError, match="source and detector surface pixels must differ"):
        Projector(*arguments, beam=beam, detector_spacing=1.0, voxel_spacing=1.0,
                  detector_surface=_flat_surface)
