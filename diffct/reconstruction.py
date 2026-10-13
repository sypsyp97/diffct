"""Composable analytical reconstruction APIs for fixed, flat-detector geometry."""

import math
import numbers

import torch
import torch.nn.functional as F

from .analytical import (
    _analytical_geometry, angular_integration_weights, cone_weighted_backproject,
    fan_weighted_backproject, parallel_weighted_backproject,
    parker_weights, ramp_filter_1d,
)
from .operators import _detector_spacing, _positive_float, _shape, _validate_trajectory


def _fixed(value, name):
    if isinstance(value, torch.Tensor) and value.requires_grad:
        raise ValueError(f"{name} gradients are not supported by analytical reconstruction")


def _prepare(sinogram, trajectory, volume_shape, beam, detector_spacing, voxel_spacing, isocenter):
    if not isinstance(sinogram, torch.Tensor) or not sinogram.is_cuda:
        raise ValueError("Analytical reconstruction requires a CUDA sinogram tensor")
    if not torch.is_floating_point(sinogram):
        raise ValueError("sinogram must have a real floating-point dtype")
    # FFT and geometric checks need at least float32 precision. The native
    # cast keeps gradients connected to low-precision input/parameter leaves.
    if sinogram.dtype not in (torch.float32, torch.float64):
        sinogram = sinogram.float()
    rank = 3 if beam == "cone" else 2
    if sinogram.ndim != rank or min(sinogram.shape[1:]) < 2:
        raise ValueError("sinogram rank must match the beam and detector axes need at least two bins")
    for component in trajectory:
        _fixed(component, "trajectory")
    _fixed(isocenter, "isocenter")
    geometry, views = _validate_trajectory(trajectory, beam)
    if sinogram.shape[0] != views:
        raise ValueError("sinogram and trajectory must have the same number of views")
    shape = _shape(volume_shape, rank, "volume_shape")
    pitch = _detector_spacing(detector_spacing, beam)
    spacing = _positive_float(voxel_spacing, "voxel_spacing")
    geometry = tuple(t.to(device=sinogram.device, dtype=sinogram.dtype) for t in geometry)
    return sinogram, geometry, shape, pitch, spacing


def _angles(angles, views, device):
    _fixed(angles, "angles")
    value = torch.as_tensor(angles, device=device)
    if not torch.is_floating_point(value) or value.shape != (views,) or not torch.isfinite(value).all():
        raise ValueError("angles must be a finite real vector with one unwrapped radian angle per view")
    return value


def _weights(value, sinogram, name, shape=None):
    value = torch.as_tensor(value, device=sinogram.device, dtype=sinogram.dtype)
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite values")
    if shape is not None:
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    else:
        try:
            if torch.broadcast_shapes(value.shape, sinogram.shape) != sinogram.shape:
                raise ValueError
        except (RuntimeError, ValueError) as error:
            raise ValueError(f"{name} must broadcast to the sinogram shape") from error
    return value


def _cosine(sinogram, geometry, pitch):
    """Detector-normal cosine from actual per-view rays, including offsets."""
    source, detector, axis_u = geometry[:3]
    vector = detector - source
    if sinogram.ndim == 2:
        normal = torch.stack((-axis_u[:, 1], axis_u[:, 0]), dim=1)
    else:
        normal = torch.linalg.cross(axis_u, geometry[3], dim=1)
    normal = F.normalize(normal, dim=1)
    distance = (vector * normal).sum(1).abs()
    u = (torch.arange(sinogram.shape[1], device=sinogram.device, dtype=sinogram.dtype)
         - (sinogram.shape[1] - 1) / 2) * pitch[0]
    offset_u = (vector * axis_u).sum(1)
    u = u[None, :] + offset_u[:, None]
    if sinogram.ndim == 2:
        return distance[:, None] / torch.sqrt(distance[:, None].square() + u.square())
    v = (torch.arange(sinogram.shape[2], device=sinogram.device, dtype=sinogram.dtype)
         - (sinogram.shape[2] - 1) / 2) * pitch[1]
    offset_v = (vector * geometry[3]).sum(1)
    v = v[None, :] + offset_v[:, None]
    return distance[:, None, None] / torch.sqrt(
        distance[:, None, None].square() + u[:, :, None].square() + v[:, None, :].square())


def _filter(sinogram, pitch, window, pad_factor, response):
    if isinstance(pad_factor, bool) or not isinstance(pad_factor, numbers.Integral) or pad_factor < 2:
        raise ValueError("pad_factor must be an integer >=2 for analytical reconstruction")
    if response is None:
        return ramp_filter_1d(sinogram, dim=1, sample_spacing=pitch,
                              pad_factor=pad_factor, window=window)
    n_u = sinogram.shape[1]
    fft_size = pad_factor * n_u
    response = _weights(response, sinogram, "filter_response", (fft_size // 2 + 1,))
    left = (fft_size - n_u) // 2
    right = fft_size - n_u - left
    padded = F.pad(sinogram, (0, 0, left, right) if sinogram.ndim == 3 else (left, right))
    broadcast = (1, len(response)) + (1,) * (sinogram.ndim - 2)
    filtered = torch.fft.irfft(torch.fft.rfft(padded, dim=1) * response.reshape(broadcast),
                               n=fft_size, dim=1)
    return filtered.narrow(1, left, n_u)


def _parker(angles, sinogram, geometry, pitch, isocenter, geometry_eps):
    source, detector, u = geometry[:3]
    vector = detector - source
    normal = (torch.stack((-u[:, 1], u[:, 0]), dim=1) if sinogram.ndim == 2
              else torch.linalg.cross(u, geometry[3], dim=1))
    normal = F.normalize(normal, dim=1)
    normal = torch.where(((vector * normal).sum(1) < 0)[:, None], -normal, normal)
    distance = (vector * normal).sum(1).abs()
    offset = (vector * u).sum(1)
    # Preserve each input's quantization after _prepare promotes the geometry.
    # Eight eps covers coordinate rounding, subtraction, projection and a
    # comparison between two views; its physical scale also covers translation.
    coordinate_scale = float(torch.linalg.vector_norm(
        source.double().abs() + detector.double().abs(), dim=1).max())
    arithmetic_eps = torch.finfo(source.dtype).eps
    # V participates in the cone normal/distance, but never the U-offset dot.
    distance_precision = max((arithmetic_eps, *geometry_eps))
    offset_precision = max((arithmetic_eps, *geometry_eps[:3]))
    distance_tolerance = max(1e-5, 8 * distance_precision * coordinate_scale)
    offset_tolerance = max(1e-5, 8 * offset_precision * coordinate_scale)
    if not (torch.allclose(distance, distance[:1].expand_as(distance), rtol=1e-4, atol=distance_tolerance)
            and torch.allclose(offset, offset[:1].expand_as(offset), rtol=1e-4, atol=offset_tolerance)):
        raise ValueError("Parker short-scan weighting requires constant detector distance and offset")
    sdd, offset_u = float(distance[0]), float(offset[0])
    gamma_max = math.atan(((sinogram.shape[1] - 1) * pitch[0] / 2 + abs(offset_u)) / sdd)
    span = float(angles.max() - angles.min())
    if span < math.pi + 2 * gamma_max - 1e-5:
        raise ValueError("short scan must cover at least pi + twice the maximum fan angle")
    if not (torch.all(angles[1:] > angles[:-1]) or torch.all(angles[1:] < angles[:-1])):
        raise ValueError("Parker short-scan angles must be strictly monotone")

    center, _, _ = _analytical_geometry(source, detector, normal, isocenter)
    radial = (source - center).double()
    radius = torch.linalg.vector_norm(radial, dim=1)
    if (not torch.allclose(radius, radius[:1].expand_as(radius), rtol=1e-4, atol=1e-5)
            or not torch.allclose(normal.double(), -F.normalize(radial, dim=1), rtol=1e-4, atol=1e-5)):
        raise ValueError("Parker weighting requires a circular orbit with radial detector normals")
    delta = angles.double() - angles[0].double()
    cosine, sine = delta.cos(), delta.sin()
    index = int(sine.abs().argmax())
    if abs(float(sine[index])) < 1e-4:
        raise ValueError("Parker weighting needs circular views that define the orbit orientation")
    tangent = (radial[index] - radial[0] * cosine[index]) / sine[index]
    predicted = radial[0] * cosine[:, None] + tangent * sine[:, None]
    if (not torch.allclose(predicted, radial, rtol=1e-4, atol=1e-5)
            or not torch.allclose(tangent.norm(), radius[0], rtol=1e-4, atol=1e-5)):
        raise ValueError("Parker weighting requires angles matching a circular orbit")
    # The detector u axis points opposite the source tangent for the scalar
    # Parker helper's convention. Derive its sign from the actual orbit, so a
    # detector-axis mirror plus a data-bin mirror leaves reconstruction intact.
    expected_u = F.normalize(radial[0] * sine[:, None] - tangent * cosine[:, None], dim=1)
    handedness = (expected_u * u.double()).sum(1)
    if not torch.allclose(handedness.abs(), torch.ones_like(handedness), rtol=1e-4, atol=1e-5):
        raise ValueError("Parker weighting requires detector u tangent to the circular orbit")
    if not (torch.all(handedness > 0) or torch.all(handedness < 0)):
        raise ValueError("Parker weighting requires consistent detector-u orientation")
    direction = 1 if angles[-1] > angles[0] else -1
    fan_sign = direction * (1 if handedness[0] > 0 else -1)
    weights = parker_weights(delta * direction, sinogram.shape[1], pitch[0], sdd,
                              detector_offset=fan_sign * offset_u)
    if fan_sign < 0:
        weights = weights.flip(1)
    return weights[..., None] if sinogram.ndim == 3 else weights


def _pipeline(sinogram, geometry, angles, beam, pitch, window, pad_factor,
              projection_weights, angular_weights, filter_response, short_scan, isocenter,
              geometry_eps):
    if short_scan and beam == "parallel":
        raise ValueError("Parker short-scan weights apply to fan or cone beams, not parallel")
    if projection_weights is None:
        weighted = sinogram if beam == "parallel" else sinogram * _cosine(sinogram, geometry, pitch)
    else:
        weighted = sinogram * _weights(projection_weights, sinogram, "projection_weights")
    if short_scan:
        weighted = weighted * _parker(angles, sinogram, geometry, pitch, isocenter, geometry_eps)
    filtered = _filter(weighted, pitch[0], window, pad_factor, filter_response)
    relative_angles = angles.double() - angles[0].double()
    angular = (angular_integration_weights(relative_angles) if angular_weights is None
               else _weights(angular_weights, sinogram, "angular_weights", (len(angles),)))
    return filtered * angular.reshape((len(angles),) + (1,) * (sinogram.ndim - 1))


def fbp(sinogram, trajectory, volume_shape, angles, *, beam="parallel",
        detector_spacing=1.0, voxel_spacing=1.0, window="ram-lak", pad_factor=2,
        projection_weights=None, angular_weights=None, filter_response=None,
        short_scan=False, isocenter=None):
    """Differentiable parallel/fan FBP for fixed flat-detector geometry.

    ``sinogram`` is a CUDA tensor ``(views, bins)``; ``volume_shape`` is ``(H,W)``.
    ``angles`` are explicit unwrapped radians. Detector/voxel spacings share the
    trajectory's physical unit. ``projection_weights`` and ``angular_weights``
    replace the default cosine and angular quadrature, respectively.
    ``filter_response`` replaces the complete real rFFT multiplier, including
    its physical units; length is ``pad_factor*bins//2+1``. Its use ignores
    ``window``. These tensors retain autograd. ``short_scan=True`` adds Parker
    weighting before filtering for a circular fan scan with sufficient coverage.
    Low-precision inputs are promoted to float32 for reconstruction, retaining
    gradients to their original dtype. Output is signed float32; geometry
    gradients are rejected.
    """
    if beam not in ("parallel", "fan"):
        raise ValueError("fbp beam must be 'parallel' or 'fan'; use fdk for cone data")
    sinogram, geometry, shape, pitch, spacing = _prepare(
        sinogram, trajectory, volume_shape, beam, detector_spacing, voxel_spacing, isocenter)
    angles = _angles(angles, len(geometry[0]), sinogram.device)
    filtered = _pipeline(sinogram, geometry, angles, beam, pitch, window, pad_factor,
                         projection_weights, angular_weights, filter_response, short_scan, isocenter,
                         tuple(torch.finfo(component.dtype).eps for component in trajectory))
    if beam == "parallel":
        return parallel_weighted_backproject(filtered, *geometry, pitch[0], *shape, spacing)
    return fan_weighted_backproject(filtered, *geometry, pitch[0], *shape, spacing, isocenter=isocenter)


def fdk(sinogram, trajectory, volume_shape, angles, *, detector_spacing=1.0,
        voxel_spacing=1.0, window="ram-lak", pad_factor=2,
        projection_weights=None, angular_weights=None, filter_response=None,
        short_scan=False, isocenter=None):
    """Differentiable flat-detector FDK; fixed geometry, approximate inversion.

    ``sinogram`` is ``(views,U,V)`` on CUDA; ``volume_shape`` is ``(D,H,W)``.
    Filtering runs along U. ``detector_spacing`` is a scalar or ``(du,dv)``.
    The explicit weight/filter overrides and ``short_scan`` follow :func:`fbp`.
    Returns signed float32 without clamping. The backward pass is the precise
    discrete interpolation transpose; it does not make FDK an exact inverse.
    Use :func:`helical_fdk` for slice-centered integration of an axial helix.
    """
    sinogram, geometry, shape, pitch, spacing = _prepare(
        sinogram, trajectory, volume_shape, "cone", detector_spacing, voxel_spacing, isocenter)
    angles = _angles(angles, len(geometry[0]), sinogram.device)
    filtered = _pipeline(sinogram, geometry, angles, "cone", pitch, window, pad_factor,
                         projection_weights, angular_weights, filter_response, short_scan, isocenter,
                         tuple(torch.finfo(component.dtype).eps for component in trajectory))
    return cone_weighted_backproject(filtered, *geometry, *shape, *pitch, spacing, isocenter=isocenter)


def helical_view_weights(angles, source_z, slice_z):
    """Return ``(slices,views)`` weights for complete slice-centered turns.

    ``angles`` are strictly monotone, unwrapped radians; ``source_z`` is the
    corresponding strictly monotone source height, with at least two samples.
    ``slice_z`` holds physical slice centers. Reversed acquisitions are valid.
    Interpolate the angle at each height, then integrate angular sample cells
    over the centered ``2*pi`` interval. Weights include the half redundancy
    factor and sum to pi. Missing coverage raises instead of renormalizing.
    These acquisition coordinates are fixed; learn replacement weights instead.
    """
    for value, name in ((angles, "angles"), (source_z, "source_z"), (slice_z, "slice_z")):
        _fixed(value, name)
    angles = torch.as_tensor(angles)
    if not torch.is_floating_point(angles) or angles.ndim != 1 or len(angles) < 2:
        raise ValueError("angles must be a real vector with at least two views")
    output_dtype = angles.dtype
    source_z = torch.as_tensor(source_z, device=angles.device, dtype=torch.float64)
    slice_z = torch.as_tensor(slice_z, device=angles.device, dtype=torch.float64)
    if (source_z.shape != angles.shape or slice_z.ndim != 1 or len(slice_z) == 0
            or not all(torch.isfinite(t).all() for t in (angles, source_z, slice_z))):
        raise ValueError("angles/source_z must match and slice_z must be a finite vector")
    for value, name in ((angles, "angles"), (source_z, "source_z")):
        differences = value[1:] - value[:-1]
        if not (torch.all(differences > 0) or torch.all(differences < 0)):
            raise ValueError(f"{name} must be strictly monotone; use circular fdk for zero pitch")
    # Compute interval arithmetic relative to the first angle in float64. A
    # large absolute angle must neither erase pi nor widen the coverage guard.
    angles = angles.double() - angles[0].double()
    source_z, slice_z = source_z.double(), slice_z.double()
    ordered, order = angles.sort()
    heights = source_z[order]
    heights, z_order = heights.sort()
    angle_at_z = ordered[z_order]
    if torch.any(slice_z < heights[0]) or torch.any(slice_z > heights[-1]):
        raise ValueError("Source heights do not cover the requested slices")
    upper = torch.searchsorted(heights, slice_z).clamp(1, len(heights) - 1)
    fraction = (slice_z - heights[upper - 1]) / (heights[upper] - heights[upper - 1])
    center = angle_at_z[upper - 1] + fraction * (angle_at_z[upper] - angle_at_z[upper - 1])
    lower, upper_angle = center - math.pi, center + math.pi
    # Output rounding must not relax the acquired-window coverage contract.
    precision = min(torch.finfo(output_dtype).eps, torch.finfo(torch.float32).eps)
    tolerance = 8 * precision * max(1.0, float(ordered[-1] - ordered[0]))
    if torch.any(lower < ordered[0] - tolerance) or torch.any(upper_angle > ordered[-1] + tolerance):
        raise ValueError("Every reconstructed slice needs a fully acquired 2*pi angular window")
    # Clip Voronoi integration cells, including fractional boundary cells.
    # Their union is the acquired interval. Missing views are never normalized.
    edges = torch.cat((ordered[:1], (ordered[:-1] + ordered[1:]) / 2, ordered[-1:]))
    overlap = (torch.minimum(edges[1:][None, :], upper_angle[:, None])
               - torch.maximum(edges[:-1][None, :], lower[:, None])).clamp_min(0) * 0.5
    weights = torch.empty_like(overlap)
    weights[:, order] = overlap
    return weights.to(output_dtype)


def _axial_helix(geometry, isocenter):
    _fixed(isocenter, "isocenter")
    source, detector, u, v = geometry
    center = torch.as_tensor(isocenter, device=source.device, dtype=source.dtype)
    if center.shape != (3,) or not torch.isfinite(center).all():
        raise ValueError("isocenter must be a finite physical (x,y,z) vector")
    radial = source[:, :2] - center[:2]
    radius = torch.linalg.vector_norm(radial, dim=1)
    if torch.any(radius <= 0) or not torch.allclose(radius, radius[:1].expand_as(radius), rtol=1e-4, atol=1e-5):
        raise ValueError("helical FDK requires a constant source radius about isocenter")
    if (not torch.allclose(u[:, 2], torch.zeros_like(u[:, 2]), atol=1e-5)
            or not torch.allclose(v[:, :2], torch.zeros_like(v[:, :2]), atol=1e-5)
            or not torch.allclose(source[:, 2], detector[:, 2], rtol=1e-4, atol=1e-5)):
        raise ValueError("helical FDK requires an axial helix and an untilted vertical detector")
    normal = F.normalize(torch.linalg.cross(u, v), dim=1)
    align = ((detector - source) * normal).sum(1)
    normal = torch.where(align[:, None] < 0, -normal, normal)
    if not torch.allclose(normal[:, :2], -radial / radius[:, None], rtol=1e-4, atol=1e-5):
        raise ValueError("helical FDK requires radial detector normals")
    dz = source[1:, 2] - source[:-1, 2]
    if len(dz) == 0 or not (torch.all(dz > 0) or torch.all(dz < 0)):
        raise ValueError("helical source heights must be strictly monotone; use circular fdk for zero pitch")
    return center


def helical_weighted_backproject(sinogram, trajectory, volume_shape, view_weights,
                                *, isocenter, detector_spacing=1.0,
                                voxel_spacing=1.0):
    """Gather filtered cone data using replaceable ``(D,views)`` weights.

    ``sinogram`` is a filtered CUDA ``(views,U,V)`` tensor. ``view_weights``
    includes angular quadrature and redundancy and is applied AFTER filtering.
    Gradients flow to both tensors through the discrete cone-gather transpose.
    Fixed axial geometry, constant source radius and radial flat detectors are
    required. Caller-supplied weights own their angular coverage and truncation.
    One slice is processed per launch, avoiding a views-by-volume weight array;
    a fused slice-weight kernel can replace this if launch overhead dominates.
    """
    sinogram, geometry, shape, pitch, spacing = _prepare(
        sinogram, trajectory, volume_shape, "cone", detector_spacing, voxel_spacing, isocenter)
    center = _axial_helix(geometry, isocenter)
    weights = _weights(view_weights, sinogram, "view_weights", (shape[0], len(geometry[0])))
    slices = []
    # One launch per slice; fuse slice weights if measured launch overhead
    # dominates. Preserve ordinary PyTorch weight gradients here.
    for index in range(shape[0]):
        z = (index + 0.5 - shape[0] / 2) * spacing
        shift = sinogram.new_tensor((0, 0, z))
        source, detector, u, v = geometry
        weighted = sinogram * weights[index, :, None, None]
        plane = cone_weighted_backproject(
            weighted, source - shift, detector - shift, u, v,
            1, shape[1], shape[2], *pitch, spacing, isocenter=center - shift)
        slices.append(plane[0])
    return torch.stack(slices)


def helical_fdk(sinogram, trajectory, volume_shape, angles, *, isocenter,
                detector_spacing=1.0, voxel_spacing=1.0, window="ram-lak",
                pad_factor=2, projection_weights=None, view_weights=None,
                filter_response=None):
    """Approximate axial helical FDK with one complete angular turn per slice.

    Filters detector U once, then applies :func:`helical_view_weights` inside
    the analytical gather. ``isocenter`` is required in physical coordinates.
    Source heights and unwrapped ``angles`` must agree with the actual helix;
    finite scans must cover a full turn around every requested slice.
    ``view_weights`` replaces the complete ``(D,views)`` window/quadrature stage.
    Projection/filter replacements follow :func:`fbp`. Custom tensors retain
    autograd and are not normalized or clamped. Detector truncation is not
    corrected. This native-cone approximation does not implement cone-parallel
    rebinning, 3D conjugate weights or exact Katsevich inversion.
    """
    sinogram, geometry, shape, pitch, spacing = _prepare(
        sinogram, trajectory, volume_shape, "cone", detector_spacing, voxel_spacing, isocenter)
    center = _axial_helix(geometry, isocenter)
    angles = _angles(angles, len(geometry[0]), sinogram.device)
    relative = geometry[0][:, :2] - center[:2]
    observed = torch.atan2(-relative[:, 0], relative[:, 1])
    phase = observed.double() - angles.double()
    if not torch.allclose(torch.cos(phase - phase[0]), torch.ones_like(phase), rtol=0, atol=1e-5):
        raise ValueError("angles must describe the source's actual unwrapped rotation")
    if view_weights is None:
        z = (torch.arange(shape[0], device=sinogram.device, dtype=torch.float64)
             + 0.5 - shape[0] / 2) * spacing
        view_weights = helical_view_weights(angles, geometry[0][:, 2], z)
    else:
        # Still validate the acquired order when a caller replaces the windows.
        difference = angles[1:] - angles[:-1]
        if not (torch.all(difference > 0) or torch.all(difference < 0)):
            raise ValueError("helical angles must be strictly monotone")
    weighted = (sinogram * _cosine(sinogram, geometry, pitch) if projection_weights is None
                else sinogram * _weights(projection_weights, sinogram, "projection_weights"))
    filtered = _filter(weighted, pitch[0], window, pad_factor, filter_response)
    return helical_weighted_backproject(filtered, geometry, shape, view_weights,
                                        isocenter=center, detector_spacing=pitch, voxel_spacing=spacing)
