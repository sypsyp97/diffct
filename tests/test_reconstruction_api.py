"""Composition contracts for FBP/FDK and slice-windowed native helical FDK.

Initial RED is against the declared API scaffold, not a pre-existing regression.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from diffct import (
    angular_integration_weights,
    cone_weighted_backproject,
    fan_weighted_backproject,
    fbp,
    fdk,
    helical_fdk,
    helical_view_weights,
    helical_weighted_backproject,
    parallel_weighted_backproject,
    parker_weights,
    ramp_filter_1d,
    spiral_trajectory_3d,
)


BEAMS = ("parallel", "fan", "cone")
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _random(shape, seed=200, *, requires_grad=False):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(shape, generator=generator, device="cuda", dtype=torch.float64,
                       requires_grad=requires_grad)


def _acquisition(beam, *, short_scan=False):
    angles = (torch.linspace(0, math.radians(230), 9) if short_scan
              else torch.arange(7) * (2 * math.pi / 7))
    u = torch.stack((angles.cos(), angles.sin()), dim=1)
    normal = torch.stack((angles.sin(), -angles.cos()), dim=1)
    center = torch.tensor([0.15, -0.2])
    shape, sino_shape, pitch = (3, 5), (len(angles), 11), (0.7,)
    if beam == "parallel":
        trajectory = (-normal, center + u * 0.2, u)
    else:
        if beam == "cone":
            u = F.pad(u, (0, 1))
            normal = F.pad(normal, (0, 1))
            v = torch.tensor([0.0, 0.0, 1.0]).expand(len(angles), -1)
            center = torch.tensor([0.15, -0.2, 0.1])
            shape, sino_shape, pitch = (2, 3, 5), (len(angles), 11, 7), (0.7, 0.9)
        source = center - normal * 6.0
        detector = center + normal * 6.0 + u * 0.3
        if beam == "cone":
            detector = detector + v * -0.2
        trajectory = (source, detector, u, v) if beam == "cone" else (source, detector, u)
    return {
        "trajectory": trajectory, "angles": angles, "shape": shape,
        "sino_shape": sino_shape, "pitch": pitch, "voxel_spacing": 0.5,
        "isocenter": center,
    }


def _cosine(beam, sinogram, acquisition):
    if beam == "parallel":
        return torch.ones_like(sinogram)
    source, detector, u = (t.to(sinogram) for t in acquisition["trajectory"][:3])
    n_u = sinogram.shape[1]
    coordinates = (torch.arange(n_u, device=sinogram.device, dtype=sinogram.dtype)
                   - (n_u - 1) / 2) * acquisition["pitch"][0]
    if beam == "fan":
        normal = torch.stack((-u[:, 1], u[:, 0]), dim=1)
        rays = detector[:, None, :] - source[:, None, :] + coordinates[None, :, None] * u[:, None, :]
    else:
        v = acquisition["trajectory"][3].to(sinogram)
        normal = torch.linalg.cross(u, v)
        n_v = sinogram.shape[2]
        vertical = (torch.arange(n_v, device=sinogram.device, dtype=sinogram.dtype)
                    - (n_v - 1) / 2) * acquisition["pitch"][1]
        rays = (detector[:, None, None, :] - source[:, None, None, :]
                + coordinates[None, :, None, None] * u[:, None, None, :]
                + vertical[None, None, :, None] * v[:, None, None, :])
    normal = normal / torch.linalg.vector_norm(normal, dim=1, keepdim=True)
    distance = ((detector - source) * normal).sum(dim=1).abs()
    broadcast = (len(source),) + (1,) * (sinogram.ndim - 1)
    return distance.reshape(broadcast) / torch.linalg.vector_norm(rays, dim=-1)


def _filter(sinogram, pitch, *, response=None, pad_factor=2, window="ram-lak"):
    if response is None:
        return ramp_filter_1d(sinogram, dim=1, sample_spacing=pitch,
                              pad_factor=pad_factor, window=window)
    # Explicit response is the complete multiplier, already in physical units.
    n_u = sinogram.shape[1]
    padded_length = n_u * pad_factor
    left = (padded_length - n_u) // 2
    right = padded_length - n_u - left
    padded = F.pad(sinogram, (0, 0, left, right) if sinogram.ndim == 3 else (left, right))
    broadcast = (1, len(response)) + (1,) * (sinogram.ndim - 2)
    filtered = torch.fft.irfft(torch.fft.rfft(padded, dim=1) * response.reshape(broadcast),
                               n=padded_length, dim=1)
    return filtered.narrow(1, left, n_u)


def _gather(beam, sinogram, acquisition):
    trajectory = acquisition["trajectory"]
    shape, pitch, spacing = acquisition["shape"], acquisition["pitch"], acquisition["voxel_spacing"]
    if beam == "parallel":
        return parallel_weighted_backproject(sinogram, *trajectory, *pitch, *shape, spacing)
    if beam == "fan":
        return fan_weighted_backproject(sinogram, *trajectory, *pitch, *shape, spacing,
                                       isocenter=acquisition["isocenter"])
    return cone_weighted_backproject(sinogram, *trajectory, *shape, *pitch, spacing,
                                    isocenter=acquisition["isocenter"])


def _manual(beam, sinogram, acquisition, *, projection_weights=None, angular_weights=None,
            filter_response=None, short_scan=False, window="ram-lak"):
    projection = (_cosine(beam, sinogram, acquisition) if projection_weights is None
                  else projection_weights)
    weighted = sinogram * projection
    if short_scan:
        offset = 0.3  # Tangential detector displacement in _acquisition.
        parker = parker_weights(acquisition["angles"].to(sinogram.device), sinogram.shape[1],
                                 acquisition["pitch"][0], 12.0, detector_offset=offset)
        weighted = weighted * (parker[..., None] if beam == "cone" else parker)
    filtered = _filter(weighted, acquisition["pitch"][0], response=filter_response, window=window)
    angular = (angular_integration_weights(acquisition["angles"].to(sinogram.device))
               if angular_weights is None else angular_weights)
    broadcast = (len(angular),) + (1,) * (sinogram.ndim - 1)
    return _gather(beam, filtered * angular.reshape(broadcast), acquisition)


def _reconstruct(beam, sinogram, acquisition, **overrides):
    kwargs = {
        "detector_spacing": acquisition["pitch"] if beam == "cone" else acquisition["pitch"][0],
        "voxel_spacing": acquisition["voxel_spacing"], "isocenter": acquisition["isocenter"],
        **overrides,
    }
    if beam == "cone":
        return fdk(sinogram, acquisition["trajectory"], acquisition["shape"],
                   acquisition["angles"], **kwargs)
    return fbp(sinogram, acquisition["trajectory"], acquisition["shape"],
               acquisition["angles"], beam=beam, **kwargs)


@pytest.mark.cuda
@CUDA
@pytest.mark.parametrize("beam", BEAMS)
def test_default_reconstruction_matches_native_pipeline_with_detector_offsets(beam):
    acquisition = _acquisition(beam)
    sinogram = _random(acquisition["sino_shape"])
    expected = _manual(beam, sinogram, acquisition, window="hamming")
    actual = _reconstruct(beam, sinogram, acquisition, window="hamming")
    assert actual.shape == acquisition["shape"] and actual.dtype == torch.float32
    assert actual.device == sinogram.device
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.cuda
@CUDA
@pytest.mark.parametrize("beam", BEAMS)
def test_explicit_weight_and_filter_overrides_replace_defaults_and_keep_gradients(beam):
    acquisition = _acquisition(beam)
    n_views, n_u = acquisition["sino_shape"][:2]
    sinogram = _random(acquisition["sino_shape"], 201, requires_grad=True)
    projection = (_random(sinogram.shape, 202) * 0.1 + 0.8).requires_grad_()
    angular = (_random((n_views,), 203) * 0.03 + 0.2).requires_grad_()
    response = (_random((n_u + 1,), 204) * 0.1 + 0.4).requires_grad_()
    overrides = {"projection_weights": projection, "angular_weights": angular,
                 "filter_response": response, "window": "ignored-by-explicit-response"}
    actual = _reconstruct(beam, sinogram, acquisition, **overrides)
    expected = _manual(beam, sinogram, acquisition, **overrides)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    cotangent = _random(acquisition["shape"], 205).float()
    inputs = (sinogram, projection, angular, response)
    expected_gradients = torch.autograd.grad(expected, inputs, cotangent)
    actual_gradients = torch.autograd.grad(actual, inputs, cotangent)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(actual_gradient).all() and actual_gradient.abs().max() > 0
        assert actual_gradient.dtype == torch.float64
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=3e-5, atol=3e-6)


@pytest.mark.cuda
@CUDA
@pytest.mark.parametrize("beam", BEAMS)
def test_reconstruction_returns_signed_values_without_clamping(beam):
    acquisition = _acquisition(beam)
    sinogram = torch.ones(acquisition["sino_shape"], device="cuda")
    overrides = {
        "projection_weights": torch.ones_like(sinogram),
        "angular_weights": torch.ones(len(acquisition["angles"]), device="cuda"),
        "filter_response": -torch.ones(sinogram.shape[1] + 1, device="cuda"),
    }
    expected = _manual(beam, sinogram, acquisition, **overrides)
    actual = _reconstruct(beam, sinogram, acquisition, **overrides)
    assert expected.min() < -0.1 and actual.min() < -0.1
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.cuda
@CUDA
@pytest.mark.parametrize("beam", BEAMS)
def test_physical_length_rescaling_preserves_reconstruction_amplitude(beam):
    acquisition = _acquisition(beam)
    sinogram = _random(acquisition["sino_shape"], 206)
    scale = 2.5
    scaled = {**acquisition,
              "pitch": tuple(p * scale for p in acquisition["pitch"]),
              "voxel_spacing": acquisition["voxel_spacing"] * scale,
              "isocenter": acquisition["isocenter"] * scale}
    first, second, *axes = acquisition["trajectory"]
    scaled["trajectory"] = ((first, second * scale, *axes) if beam == "parallel"
                            else (first * scale, second * scale, *axes))
    expected = _reconstruct(beam, sinogram, acquisition)
    actual = _reconstruct(beam, sinogram * scale, scaled)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


@pytest.mark.cuda
@CUDA
@pytest.mark.parametrize("beam", ("fan", "cone"))
def test_short_scan_composes_parker_before_filtering(beam):
    acquisition = _acquisition(beam, short_scan=True)
    sinogram = _random(acquisition["sino_shape"], 207)
    expected = _manual(beam, sinogram, acquisition, short_scan=True)
    actual = _reconstruct(beam, sinogram, acquisition, short_scan=True)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


@pytest.mark.cuda
@CUDA
def test_parallel_short_scan_is_rejected():
    acquisition = _acquisition("parallel", short_scan=True)
    with pytest.raises(ValueError, match=r"(?i)(parallel|short.scan)"):
        _reconstruct("parallel", _random(acquisition["sino_shape"]), acquisition, short_scan=True)


def _overlap_reference(angles, source_z, slice_z):
    """Independent cell-overlap rule; acquisition endpoints close the end cells."""
    order = angles.argsort()
    angle = angles[order].double().cpu()
    heights = source_z[order].double().cpu()
    z_order = heights.argsort()
    z, angle_at_z = heights[z_order], angle[z_order]
    slices = slice_z.double().cpu()
    upper = torch.searchsorted(z, slices).clamp(1, len(z) - 1)
    fraction = (slices - z[upper - 1]) / (z[upper] - z[upper - 1])
    center = angle_at_z[upper - 1] + fraction * (angle_at_z[upper] - angle_at_z[upper - 1])
    edges = torch.cat((angle[:1], (angle[:-1] + angle[1:]) / 2, angle[-1:]))
    overlap = (torch.minimum(edges[1:][None, :], center[:, None] + math.pi)
               - torch.maximum(edges[:-1][None, :], center[:, None] - math.pi)).clamp(min=0) / 2
    result = torch.empty_like(overlap)
    result[:, order.cpu()] = overlap
    return result


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("reversed_scan", (False, True))
@pytest.mark.parametrize("nonlinear_source_z", (False, True))
def test_helical_weights_are_exact_nonuniform_cell_overlaps(device, reversed_scan, nonlinear_source_z):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    index = torch.arange(49, device=device, dtype=torch.float64)
    angles = torch.linspace(-3 * math.pi, 3 * math.pi, 49, device=device, dtype=torch.float64)
    angles = angles + 0.025 * index.sin()
    source_z = angles * 0.6
    if nonlinear_source_z:
        source_z = source_z + 0.008 * angles.pow(3)
    slice_z = torch.tensor([-1.3, 0.2, 1.1], device=device, dtype=torch.float64)
    if reversed_scan:
        angles, source_z = angles.flip(0), source_z.flip(0)
    expected = _overlap_reference(angles, source_z, slice_z)
    actual = helical_view_weights(angles, source_z, slice_z)
    assert actual.shape == (3, 49) and actual.device == angles.device
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(actual.sum(dim=1).double(),
                               torch.full((3,), math.pi, device=device, dtype=torch.float64),
                               rtol=2e-6, atol=2e-7)
    assert (actual == 0).any() and (actual > 0).any()


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_helical_full_turn_windows_may_touch_acquisition_boundaries(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    angles = torch.linspace(-3 * math.pi, 3 * math.pi, 49, device=device, dtype=torch.float64)
    source_z = angles * 0.6
    # Centers at -2*pi and 2*pi give windows whose outer endpoints are acquired.
    slices = torch.tensor([-2 * math.pi, 2 * math.pi], device=device, dtype=torch.float64) * 0.6
    expected = _overlap_reference(angles, source_z, slices)
    actual = helical_view_weights(angles, source_z, slices)
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(actual.sum(dim=1).double(),
                               torch.full((2,), math.pi, device=device, dtype=torch.float64),
                               rtol=2e-6, atol=2e-7)
    assert actual[0, 0] > 0 and actual[1, -1] > 0


@pytest.mark.parametrize("invalid", (
    "uncovered", "zero_pitch", "angle_order", "duplicate_z", "nonmonotone_z", "shape", "nonfinite",
))
def test_invalid_helical_window_inputs_are_rejected(invalid):
    angles = torch.linspace(-3 * math.pi, 3 * math.pi, 49)
    source_z = torch.linspace(-6.0, 6.0, 49)
    slices = torch.tensor([-0.5, 0.5])
    if invalid == "uncovered":
        slices = torch.tensor([5.9])
    elif invalid == "zero_pitch":
        source_z = torch.zeros_like(source_z)
    elif invalid == "angle_order":
        angles[20] = angles[19] - 0.1
    elif invalid == "duplicate_z":
        source_z[20] = source_z[19]
    elif invalid == "nonmonotone_z":
        source_z[20] = source_z[19] - 0.1
    elif invalid == "shape":
        source_z = source_z[:-1]
    else:
        angles[20] = float("nan")
    with pytest.raises(ValueError):
        helical_view_weights(angles, source_z, slices)


def _helix(n_views=49, *, large_detector=False):
    trajectory = spiral_trajectory_3d(n_views, 20.0, 40.0, z_range=12.0, n_turns=3.0,
                                     device="cpu")
    angles = torch.arange(n_views) * (6 * math.pi / n_views)
    return {
        "trajectory": trajectory, "angles": angles, "shape": (3, 5, 5),
        "sino_shape": (n_views, 129, 65) if large_detector else (n_views, 9, 17),
        "pitch": (0.125, 0.25) if large_detector else (1.0, 1.0),
        "voxel_spacing": 0.5, "isocenter": torch.zeros(3),
    }


def _slice_z(acquisition):
    depth = acquisition["shape"][0]
    return (torch.arange(depth) + 0.5 - depth / 2) * acquisition["voxel_spacing"]


def _manual_helical_gather(sinogram, acquisition, view_weights):
    slices = []
    for index, z in enumerate(_slice_z(acquisition)):
        shift = torch.tensor([0.0, 0.0, z])
        source, detector, u, v = acquisition["trajectory"]
        shifted = {**acquisition, "shape": (1, *acquisition["shape"][1:]),
                   "trajectory": (source - shift, detector - shift, u, v),
                   "isocenter": acquisition["isocenter"] - shift}
        weighted = sinogram * view_weights[index].to(sinogram.device)[:, None, None]
        slices.append(_gather("cone", weighted, shifted)[0])
    return torch.stack(slices)


@pytest.mark.cuda
@CUDA
def test_helical_gather_matches_shifted_single_slice_composition_and_gradients():
    acquisition = _helix()
    sinogram = _random(acquisition["sino_shape"], 208, requires_grad=True)
    weights = (_random((3, 49), 209) * 0.02 + 0.1).requires_grad_()
    expected = _manual_helical_gather(sinogram, acquisition, weights)
    actual = helical_weighted_backproject(
        sinogram, acquisition["trajectory"], acquisition["shape"], weights,
        isocenter=acquisition["isocenter"], detector_spacing=acquisition["pitch"],
        voxel_spacing=acquisition["voxel_spacing"],
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    cotangent = _random(acquisition["shape"], 210).float()
    expected_gradients = torch.autograd.grad(expected, (sinogram, weights), cotangent)
    actual_gradients = torch.autograd.grad(actual, (sinogram, weights), cotangent)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(actual_gradient).all() and actual_gradient.abs().max() > 0
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=3e-5, atol=3e-6)


@pytest.mark.cuda
@CUDA
def test_helical_fdk_matches_manual_default_pipeline_and_reversed_scan():
    acquisition = _helix()
    sinogram = _random(acquisition["sino_shape"], 211)
    weights = _overlap_reference(acquisition["angles"], acquisition["trajectory"][0][:, 2],
                                  _slice_z(acquisition))
    filtered = _filter(sinogram * _cosine("cone", sinogram, acquisition), acquisition["pitch"][0])
    expected = _manual_helical_gather(filtered, acquisition, weights)
    kwargs = {"isocenter": acquisition["isocenter"], "detector_spacing": acquisition["pitch"],
              "voxel_spacing": acquisition["voxel_spacing"]}
    actual = helical_fdk(sinogram, acquisition["trajectory"], acquisition["shape"],
                         acquisition["angles"], **kwargs)
    reversed_result = helical_fdk(
        sinogram.flip(0), tuple(t.flip(0) for t in acquisition["trajectory"]),
        acquisition["shape"], acquisition["angles"].flip(0), **kwargs,
    )
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(reversed_result, expected, rtol=3e-5, atol=3e-6)


@pytest.mark.cuda
@CUDA
def test_helical_fdk_learnable_overrides_replace_defaults():
    acquisition = _helix()
    sinogram = _random(acquisition["sino_shape"], 212, requires_grad=True)
    projection = (_random(sinogram.shape, 213) * 0.05 + 0.8).requires_grad_()
    weights = (_random((3, 49), 214) * 0.01 + 0.03).requires_grad_()
    response = (_random((sinogram.shape[1] + 1,), 215) * 0.03 + 0.2).requires_grad_()
    filtered = _filter(sinogram * projection, acquisition["pitch"][0], response=response)
    expected = _manual_helical_gather(filtered, acquisition, weights)
    actual = helical_fdk(
        sinogram, acquisition["trajectory"], acquisition["shape"], acquisition["angles"],
        isocenter=acquisition["isocenter"], detector_spacing=acquisition["pitch"],
        voxel_spacing=acquisition["voxel_spacing"], projection_weights=projection,
        view_weights=weights, filter_response=response, window="ignored-by-explicit-response",
    )
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    cotangent = _random(acquisition["shape"], 216).float()
    inputs = (sinogram, projection, weights, response)
    expected_gradients = torch.autograd.grad(expected, inputs, cotangent)
    actual_gradients = torch.autograd.grad(actual, inputs, cotangent)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(actual_gradient).all() and actual_gradient.abs().max() > 0
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=3e-5, atol=3e-6)


@pytest.mark.cuda
@CUDA
def test_three_turn_helical_cylinder_center_amplitude_is_one():
    acquisition = _helix(240, large_detector=True)
    _, n_u, n_v = acquisition["sino_shape"]
    u = (torch.arange(n_u, device="cuda") - (n_u - 1) / 2) * acquisition["pitch"][0]
    v = (torch.arange(n_v, device="cuda") - (n_v - 1) / 2) * acquisition["pitch"][1]
    horizontal_length = torch.sqrt(40.0**2 + u.square())
    distance = 20.0 * u / horizontal_length
    chord = 2 * torch.sqrt((2.0**2 - distance.square()).clamp(min=0))
    # Exact line integrals of an infinite uniform cylinder of radius two.
    projection = chord[:, None] * torch.sqrt(horizontal_length[:, None].square() + v[None, :].square())
    projection = projection / horizontal_length[:, None]
    sinogram = projection.expand(acquisition["sino_shape"][0], -1, -1).contiguous()
    actual = helical_fdk(
        sinogram, acquisition["trajectory"], acquisition["shape"], acquisition["angles"],
        isocenter=acquisition["isocenter"], detector_spacing=acquisition["pitch"],
        voxel_spacing=acquisition["voxel_spacing"],
    )
    torch.testing.assert_close(actual[:, 2, 2], torch.ones(3, device="cuda"), rtol=0, atol=0.03)


@pytest.mark.cuda
@CUDA
@pytest.mark.parametrize("invalid", ("varying_radius", "tilted_detector", "zero_pitch", "weight_shape"))
def test_helical_gather_rejects_unsupported_geometry_and_weight_shapes(invalid):
    acquisition = _helix()
    source, detector, u, v = (t.clone() for t in acquisition["trajectory"])
    weights = torch.ones(3, 49, device="cuda")
    if invalid == "varying_radius":
        source[10, :2] *= 1.1
    elif invalid == "tilted_detector":
        v = math.cos(0.1) * v + math.sin(0.1) * torch.linalg.cross(u, v)
    elif invalid == "zero_pitch":
        source[:, 2] = detector[:, 2] = 0
    else:
        weights = torch.ones(3, 50, device="cuda")
    with pytest.raises(ValueError):
        helical_weighted_backproject(
            _random(acquisition["sino_shape"]), (source, detector, u, v), acquisition["shape"],
            weights, isocenter=acquisition["isocenter"], detector_spacing=acquisition["pitch"],
            voxel_spacing=acquisition["voxel_spacing"],
        )


@pytest.mark.cuda
@CUDA
@pytest.mark.parametrize(("beam", "trajectory_kind"), (
    ("cone", "helix"), ("fan", "varying_radius"), ("cone", "varying_radius"),
))
def test_parker_rejects_noncircular_geometry_with_constant_detector_distance(beam, trajectory_kind):
    acquisition = _helix() if trajectory_kind == "helix" else _acquisition(beam, short_scan=True)
    if trajectory_kind == "varying_radius":
        source, detector, *axes = acquisition["trajectory"]
        radial = F.normalize(source - acquisition["isocenter"], dim=1)
        shift = radial * torch.linspace(0.0, 0.6, len(source))[:, None]
        # Translate each source/detector pair together: SDD and offsets stay fixed.
        acquisition["trajectory"] = (source + shift, detector + shift, *axes)
        torch.testing.assert_close(acquisition["trajectory"][1] - acquisition["trajectory"][0],
                                   detector - source, rtol=1e-6, atol=2e-6)
    with pytest.raises(ValueError, match=r"(?i)(circular|orbit)"):
        _reconstruct(beam, _random(acquisition["sino_shape"], 217), acquisition, short_scan=True)


@pytest.mark.cuda
@CUDA
@pytest.mark.parametrize("beam", ("fan", "cone"))
@pytest.mark.parametrize("reversed_scan", (False, True))
def test_parker_preserves_mirrored_detector_representation(beam, reversed_scan):
    acquisition = _acquisition(beam, short_scan=True)
    sinogram = _random(acquisition["sino_shape"], 218)
    expected = _reconstruct(beam, sinogram, acquisition, short_scan=True)
    if reversed_scan:
        acquisition = {**acquisition, "angles": acquisition["angles"].flip(0),
                       "trajectory": tuple(t.flip(0) for t in acquisition["trajectory"])}
        sinogram = sinogram.flip(0)
    # Acquisition reversal alone preserves this same physical data.
    unmirrored = _reconstruct(beam, sinogram, acquisition, short_scan=True)
    torch.testing.assert_close(unmirrored, expected, rtol=3e-5, atol=3e-6)
    source, detector, u, *remaining = acquisition["trajectory"]
    mirrored = {**acquisition, "trajectory": (source, detector, -u, *remaining)}
    actual = _reconstruct(beam, sinogram.flip(1), mirrored, short_scan=True)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_helical_large_absolute_angles_match_overlap_reference_and_sum_pi(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    # Use the actual quantized float32 samples as the independent reference input.
    angles = torch.linspace(0, 6 * math.pi, 49, device=device, dtype=torch.float32) + 1_000_000
    source_z = torch.linspace(-6.0, 6.0, 49, device=device)
    slices = torch.tensor([0.0], device=device)
    expected = _overlap_reference(angles, source_z, slices)
    actual = helical_view_weights(angles, source_z, slices)
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(actual.sum(dim=1).double(),
                               torch.full((1,), math.pi, device=device, dtype=torch.float64),
                               rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_helical_large_absolute_angles_reject_uncovered_window(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    angles = torch.linspace(0, 6 * math.pi, 49, device=device, dtype=torch.float32) + 1_000_000
    source_z = torch.linspace(-6.0, 6.0, 49, device=device)
    slices = torch.tensor([-4.3], device=device)
    # This turn starts about half a radian before the first acquired angle.
    clipped = _overlap_reference(angles, source_z, slices)
    assert clipped.sum() < math.pi - 0.1
    with pytest.raises(ValueError, match=r"(?i)(cover|window|fully)"):
        helical_view_weights(angles, source_z, slices)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_helical_mixed_precision_heights_preserve_overlap_and_weight_dtype(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    angles = torch.linspace(-3 * math.pi, 3 * math.pi, 49, device=device, dtype=torch.float32)
    source_z = torch.linspace(-6.0, 6.0, 49, device=device, dtype=torch.float64) + 100_000_000
    slices = torch.tensor([100_000_000.0], device=device, dtype=torch.float64)
    assert torch.all(source_z[1:] > source_z[:-1])
    expected = _overlap_reference(angles, source_z, slices)
    actual = helical_view_weights(angles, source_z, slices)
    assert actual.dtype == angles.dtype and actual.device == angles.device
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(actual.sum(dim=1).double(),
                               torch.full((1,), math.pi, device=device, dtype=torch.float64),
                               rtol=2e-6, atol=2e-7)


def _assert_low_precision_reconstruction_matches_float32(actual, expected, inputs, reference_inputs):
    assert actual.dtype == torch.float32 and actual.device == inputs[0].device
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    cotangent = _random(actual.shape, 225).float()
    actual_gradients = torch.autograd.grad(actual, inputs, cotangent)
    reference_gradients = torch.autograd.grad(expected, reference_inputs, cotangent)
    for leaf, actual_gradient, reference_gradient in zip(inputs, actual_gradients, reference_gradients):
        assert actual_gradient.dtype == leaf.dtype
        assert torch.isfinite(actual_gradient).all() and actual_gradient.abs().max() > 0
        # Float32 atomic accumulation may straddle a leaf cast's rounding boundary.
        # Allow one leaf-dtype relative ULP; retain the float32/float64 bounds.
        torch.testing.assert_close(actual_gradient, reference_gradient.to(leaf.dtype),
                                   rtol=max(3e-5, torch.finfo(leaf.dtype).eps), atol=3e-6)


@pytest.mark.cuda
@CUDA
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("custom_response", (False, True))
@pytest.mark.parametrize(("beam", "short_scan"), (
    ("parallel", False), ("fan", False), ("cone", False), ("fan", True),
))
def test_low_precision_circular_reconstruction_matches_float32_data_and_leaf_gradients(
        dtype, custom_response, beam, short_scan):
    acquisition = _acquisition(beam, short_scan=short_scan)
    assert acquisition["sino_shape"][1] == 11
    assert all(component.dtype == torch.float32 for component in acquisition["trajectory"])
    sinogram = _random(acquisition["sino_shape"], 220).to(dtype).requires_grad_()
    reference = sinogram.detach().float().requires_grad_()
    overrides = {}
    if custom_response:
        overrides = {
            "projection_weights": (_random(sinogram.shape, 221) * 0.05 + 0.8).to(dtype).requires_grad_(),
            "angular_weights": (_random((len(acquisition["angles"]),), 222) * 0.01 + 0.2).requires_grad_(),
            "filter_response": (_random((12,), 223) * 0.03 + 0.2).to(dtype).requires_grad_(),
        }
    reference_overrides = {name: value.detach().float().requires_grad_()
                           for name, value in overrides.items()}
    expected = _reconstruct(beam, reference, acquisition, short_scan=short_scan, **reference_overrides)
    actual = _reconstruct(beam, sinogram, acquisition, short_scan=short_scan, **overrides)
    _assert_low_precision_reconstruction_matches_float32(
        actual, expected, (sinogram, *overrides.values()), (reference, *reference_overrides.values()),
    )


@pytest.mark.cuda
@CUDA
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("mode", ("default", "custom", "weighted"))
def test_low_precision_helical_reconstruction_matches_float32_data_and_leaf_gradients(dtype, mode):
    acquisition = {**_helix(), "sino_shape": (49, 11, 17)}
    assert all(component.dtype == torch.float32 for component in acquisition["trajectory"])
    sinogram = _random(acquisition["sino_shape"], 226).to(dtype).requires_grad_()
    reference = sinogram.detach().float().requires_grad_()
    kwargs = {"isocenter": acquisition["isocenter"], "detector_spacing": acquisition["pitch"],
              "voxel_spacing": acquisition["voxel_spacing"]}
    overrides = {}
    if mode != "default":
        overrides["view_weights"] = (_random((3, 49), 227) * 0.01 + 0.1).to(dtype).requires_grad_()
    if mode == "custom":
        overrides.update({
            "projection_weights": (_random(sinogram.shape, 228) * 0.05 + 0.8).to(dtype).requires_grad_(),
            "filter_response": (_random((12,), 229) * 0.03 + 0.2).to(dtype).requires_grad_(),
        })
    reference_overrides = {name: value.detach().float().requires_grad_()
                           for name, value in overrides.items()}
    arguments = (acquisition["trajectory"], acquisition["shape"])
    if mode == "weighted":
        expected = helical_weighted_backproject(reference, *arguments, **kwargs, **reference_overrides)
        actual = helical_weighted_backproject(sinogram, *arguments, **kwargs, **overrides)
    else:
        expected = helical_fdk(reference, *arguments, acquisition["angles"], **kwargs, **reference_overrides)
        actual = helical_fdk(sinogram, *arguments, acquisition["angles"], **kwargs, **overrides)
    _assert_low_precision_reconstruction_matches_float32(
        actual, expected, (sinogram, *overrides.values()), (reference, *reference_overrides.values()),
    )


@pytest.mark.parametrize(("dtype", "uncovered_height"), (
    (torch.float16, -4.01), (torch.bfloat16, -4.3),
))
def test_low_precision_helical_angles_reject_incomplete_turn_without_relaxing_coverage(dtype, uncovered_height):
    angles = torch.linspace(-3 * math.pi, 3 * math.pi, 49, dtype=dtype)
    source_z = torch.linspace(-6.0, 6.0, 49)
    covered = torch.tensor([0.0])
    expected = _overlap_reference(angles, source_z, covered).to(dtype)
    actual = helical_view_weights(angles, source_z, covered)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    uncovered = torch.tensor([uncovered_height])
    assert _overlap_reference(angles, source_z, uncovered).sum() < math.pi - 1e-3
    # Identical quantized angles already fail the float32 coverage contract.
    with pytest.raises(ValueError, match=r"(?i)(cover|window|fully)"):
        helical_view_weights(angles.float(), source_z, uncovered)
    with pytest.raises(ValueError, match=r"(?i)(cover|window|fully)"):
        helical_view_weights(angles, source_z, uncovered)
