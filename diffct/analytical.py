"""Analytical reconstruction helpers (ramp filter, cosine weights, Parker,
angular weights, voxel-driven FBP/FDK backprojection wrappers).

Ported from the main branch's ``diffct.differentiable`` module and rewritten
for the dev branch's arbitrary-trajectory kernel API. Every helper here is
trajectory-agnostic or takes the same ``(src_pos, det_center, det_u_vec,
det_v_vec)`` arrays as the ``ParallelProjectorFunction`` /
``FanProjectorFunction`` / ``ConeProjectorFunction`` classes, so they
compose cleanly with both circular and non-circular trajectories.

Fixes the long-standing FDK / FBP amplitude bugs on the dev branch:
    * Analytical backprojection now goes through dedicated voxel-driven
      gather kernels with the classical ``(sid/U)^2`` weight instead of
      the Siddon adjoint.
    * The analytical Fourier-convention constant ``1/(2*pi)`` is applied
      automatically inside the ``*_weighted_backproject`` wrappers, so
      a unit-density disk reconstructs to amplitude 1 without any
      extra manual scaling in user code.
    * ``ramp_filter_1d`` is scaled by ``1 / sample_spacing`` so the
      reconstruction amplitude is stable across different detector
      pitches.
"""

from __future__ import annotations

import math

import torch

from .constants import _DTYPE
from .utils import (
    DeviceManager,
    TorchCUDABridge,
    _get_numba_external_stream_for,
    _grid_2d,
    _grid_3d,
)
from .kernels import (
    _parallel_2d_fbp_backproject_kernel,
    _fan_2d_fbp_backproject_kernel,
    _cone_3d_fdk_backproject_kernel,
)


# ============================================================================
# Detector coordinate helper
# ============================================================================


def detector_coordinates_1d(num_detectors, detector_spacing, detector_offset=0.0,
                            device=None, dtype=torch.float32):
    """Return detector cell centre coordinates along a 1D axis.

    The convention matches the CUDA kernels: the detector array is centred,
    ``u[k] = (k - (num_detectors - 1) * 0.5) * detector_spacing + detector_offset``.
    """
    k = torch.arange(num_detectors, device=device, dtype=dtype)
    return (k - 0.5 * (num_detectors - 1)) * detector_spacing + detector_offset


# ============================================================================
# Angular integration weights
# ============================================================================


def angular_integration_weights(angles, redundant_full_scan=True):
    """Trapezoidal per-view integration weights for the analytical FBP/FDK sum.

    For a full ``2*pi`` scan, each redundant ray is sampled twice so the
    classical FBP/FDK formula carries a ``1/2`` redundancy factor. When
    ``redundant_full_scan=True`` (the default) this factor is baked into
    the returned weights: they sum to ``pi`` over a uniform full scan,
    which is exactly what a voxel-driven gather backprojector with the
    ``(sid/U)^2`` weight needs to produce a unit-amplitude reconstruction.

    For short scans (Parker-weighted) use ``redundant_full_scan=False``:
    the weights then sum to the actual angular range and the Parker
    window handles redundancy at the sinogram-filtering stage.
    """
    angles = angles.to(dtype=torch.float32).contiguous()
    n = angles.shape[0]
    if n == 0:
        return torch.zeros_like(angles)
    if n == 1:
        return torch.ones_like(angles) * (math.pi if redundant_full_scan else 1.0)

    device = angles.device
    angles_sorted, sort_idx = torch.sort(angles)
    diffs = angles_sorted[1:] - angles_sorted[:-1]

    # Endpoint-excluded uniform circular scans are periodic quadrature
    # problems, not open-interval trapezoids. Treat both parallel-beam
    # half scans over pi and fan/cone full scans over 2*pi as periodic
    # when the missing closure gap matches the interior angular step.
    median_step = torch.median(diffs)
    span = angles_sorted[-1] - angles_sorted[0]
    period = None
    for candidate in (math.pi, 2.0 * math.pi):
        closure = angles.new_tensor(candidate) - span
        tol = max(1e-4, 0.05 * float(abs(median_step).item()))
        # Full circular acquisitions can be endpoint-excluded
        # (closure ~= median_step), endpoint-included (closure ~= 0),
        # or downsampled from a finer full scan (closure smaller than
        # the kept-view step, as in the walnut fixture). Treat these as
        # periodic when the missing closure is no larger than one normal
        # kept-view interval.
        if closure >= -tol and float(closure.item()) <= 1.5 * float(abs(median_step).item()) + tol:
            period = candidate
            break

    w = torch.zeros(n, device=device, dtype=angles.dtype)
    if period is not None:
        closure = torch.clamp(angles.new_tensor(period) - span, min=0.0)
        w[0] = 0.5 * (closure + diffs[0])
        w[-1] = 0.5 * (diffs[-1] + closure)
        if n > 2:
            w[1:-1] = 0.5 * (diffs[:-1] + diffs[1:])
    else:
        # Open-interval trapezoidal rule: each interior sample gets
        # (prev + next) / 2, boundaries get half the adjacent step.
        w[0] = 0.5 * diffs[0]
        w[-1] = 0.5 * diffs[-1]
        if n > 2:
            w[1:-1] = 0.5 * (diffs[:-1] + diffs[1:])

    full_scan_tol = max(1e-4, 0.05 * float(abs(median_step).item()))
    is_periodic_full_scan = period is not None and abs(period - 2.0 * math.pi) <= full_scan_tol
    is_open_full_scan = period is None and abs(float((span - 2.0 * math.pi).item())) <= full_scan_tol
    if redundant_full_scan and (is_periodic_full_scan or is_open_full_scan):
        w = w * 0.5  # absorb the 1/2 FBP/FDK redundancy factor

    # Return weights in the input order so the caller can just multiply
    # ``sino * d_beta`` without worrying about angle ordering.
    out = torch.empty_like(w)
    out[sort_idx] = w
    return out


# ============================================================================
# Cosine pre-weights
# ============================================================================


def fan_cosine_weights(num_detectors, detector_spacing, sdd, detector_offset=0.0,
                       device=None, dtype=torch.float32):
    """Per-detector-cell ``cos(gamma)`` weight for fan-beam FBP.

    ``cos(gamma) = sdd / sqrt(sdd^2 + u^2)`` where ``u`` is the in-plane
    detector coordinate. Multiplying the sinogram by this weight before
    the ramp filter compensates for the extra path length that off-
    principal-ray fans traverse.
    """
    u = detector_coordinates_1d(
        num_detectors, detector_spacing, detector_offset, device=device, dtype=dtype
    )
    return sdd / torch.sqrt(sdd * sdd + u * u)


def cone_cosine_weights(det_u, det_v, du, dv, sdd,
                        detector_offset_u=0.0, detector_offset_v=0.0,
                        device=None, dtype=torch.float32):
    """Per-detector-cell ``cos(gamma_cone)`` weight for cone-beam FDK.

    ``cos(gamma_cone) = sdd / sqrt(sdd^2 + u^2 + v^2)``. Returns a
    ``(det_u, det_v)`` tensor ready to broadcast across the view axis
    of a cone sinogram ``(n_views, det_u, det_v)``.
    """
    u = detector_coordinates_1d(det_u, du, detector_offset_u, device=device, dtype=dtype)
    v = detector_coordinates_1d(det_v, dv, detector_offset_v, device=device, dtype=dtype)
    uu = u.view(-1, 1)
    vv = v.view(1, -1)
    return sdd / torch.sqrt(sdd * sdd + uu * uu + vv * vv)


# ============================================================================
# Parker short-scan weights (circular fan / cone only)
# ============================================================================


def parker_weights(angles, num_detectors, detector_spacing, sdd, detector_offset=0.0):
    """Parker redundancy weights for a circular fan-beam short scan.

    The formula follows Parker (1982): each ray at view ``beta`` and
    fan angle ``gamma`` is weighted by a smooth window that guarantees
    every ray contributes exactly once when integrated over the scan.
    For a full ``2*pi`` scan this returns a constant ``1`` tensor (the
    usual redundancy ``1/2`` factor lives in
    ``angular_integration_weights`` instead).

    Only meaningful for circular (fan-shaped) trajectories — when using
    non-circular trajectories you typically want either a constant weight
    (full-coverage) or a custom redundancy scheme.
    """
    angles = angles.to(dtype=torch.float32).contiguous()
    n_ang = angles.shape[0]
    device = angles.device

    u = detector_coordinates_1d(
        num_detectors, detector_spacing, detector_offset,
        device=device, dtype=angles.dtype,
    )
    gamma = torch.atan(u / sdd)  # (n_det,)
    gamma_max = float(gamma.abs().max().item())

    angles_sorted = torch.sort(angles).values
    scan_range = float((angles_sorted[-1] - angles_sorted[0]).item())
    step = float(torch.median(angles_sorted[1:] - angles_sorted[:-1]).item())
    # Full-scan detection depends on acquired angles, not detector fan width.
    # Allow one missing endpoint interval for endpoint-excluded full scans.
    if scan_range >= 2.0 * math.pi - 1.5 * step - max(1e-6, 0.05 * step):
        # Already a full scan — no Parker needed.
        return torch.ones((n_ang, num_detectors), device=device, dtype=angles.dtype)

    # Generalized Parker taper for pi + 2*gamma_max <= coverage < 2*pi.
    # The minimum also accommodates an endpoint-excluded minimal short scan.
    delta = max(gamma_max, 0.5 * (scan_range - math.pi))
    beta = angles - angles.min()
    beta = beta.view(-1, 1)
    gamma_b = gamma.view(1, -1)

    w = torch.ones_like(beta.expand(n_ang, num_detectors))

    # Region 1: 0 <= beta < 2*(delta - gamma)
    r1 = beta < 2.0 * (delta - gamma_b)
    arg1 = (math.pi * 0.25) * beta / (delta - gamma_b + 1e-12)
    w = torch.where(r1, torch.sin(arg1) ** 2, w)

    # Region 3: pi - 2*gamma <= beta <= pi + 2*delta
    r3 = beta > math.pi - 2.0 * gamma_b
    arg3 = (math.pi * 0.25) * (math.pi + 2.0 * delta - beta) / (delta + gamma_b + 1e-12)
    w = torch.where(r3, torch.sin(arg3) ** 2, w)

    # Outside scan range: zero (shouldn't happen for well-formed inputs).
    w = torch.where(beta < 0, torch.zeros_like(w), w)
    w = torch.where(beta > math.pi + 2.0 * delta, torch.zeros_like(w), w)
    return w


# ============================================================================
# Ramp filter
# ============================================================================

_RAMP_WINDOWS = ("ram-lak", "hann", "hanning", "hamming", "cosine", "shepp-logan")


def _ramp_window(name, freqs):
    """Frequency-domain apodization window for the ramp filter.

    ``freqs`` are in cycles per sample (matching ``torch.fft.fftfreq``).
    The normalized coordinate ``nf = 2 * |freqs|`` maps DC to 0 and the
    Nyquist bin to 1.
    """
    if name is None:
        return torch.ones_like(freqs)
    nf = torch.abs(freqs) * 2.0
    nf = torch.clamp(nf, max=1.0)
    key = name.lower().replace("_", "-")
    if key in ("hann", "hanning"):
        return 0.5 * (1.0 + torch.cos(torch.pi * nf))
    if key == "hamming":
        return 0.54 + 0.46 * torch.cos(torch.pi * nf)
    if key == "cosine":
        return torch.cos(torch.pi * nf * 0.5)
    if key == "shepp-logan":
        arg = torch.pi * nf * 0.5
        return torch.where(
            nf > 1e-8, torch.sin(arg) / torch.clamp(arg, min=1e-8), torch.ones_like(nf)
        )
    if key in ("ram-lak", "ramlak", "none"):
        return torch.ones_like(freqs)
    raise ValueError(
        f"Unknown ramp window {name!r}. Expected one of {_RAMP_WINDOWS} or None."
    )


def ramp_filter_1d(sinogram_tensor, dim=-1, sample_spacing=1.0, pad_factor=1,
                   window=None, use_rfft=True):
    """Apply a finite discrete Ram-Lak filter along ``dim`` using FFT.

    The angular-frequency impulse response is ``h[0] = pi/2``,
    ``h[k] = -2/(pi*k**2)`` for odd ``k``, and zero for nonzero even
    ``k`` (Kak and Slaney, chapter 3, equation 61, scaled by ``2*pi``).
    Transforming this finite kernel preserves its small positive DC
    response and avoids the low-frequency bias of directly sampling
    ``abs(omega)``. ``pad_factor >= 2`` gives linear convolution over the
    retained samples for an unwindowed filter.

    Parameters
    ----------
    sinogram_tensor : torch.Tensor
        Real-valued sinogram tensor.
    dim : int, optional
        Axis along which to filter. Default ``-1``.
    sample_spacing : float, optional
        Physical spacing between detector samples along ``dim``. The continuous
        ramp filter has units of ``1/length``, so the FFT output is rescaled by
        ``1 / sample_spacing`` to respect physical units.
    pad_factor : int, optional
        Zero-pad the signal to ``pad_factor * N`` along ``dim`` before the FFT
        to suppress circular-convolution wrap-around artifacts.
    window : str or None, optional
        Frequency-domain apodization: ``None`` / ``"ram-lak"`` (unwindowed),
        ``"hann"``, ``"hamming"``, ``"cosine"``, or ``"shepp-logan"``.
    use_rfft : bool, optional
        Use the real-valued FFT path when ``True`` (default).
    """
    dim_pos = dim if dim >= 0 else sinogram_tensor.ndim + dim
    n = sinogram_tensor.shape[dim_pos]

    if pad_factor is None or pad_factor < 1:
        pad_factor = 1
    pad_factor = int(pad_factor)
    n_pad = n * pad_factor
    if n_pad > n:
        pad_total = n_pad - n
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        pad_spec = [0, 0] * sinogram_tensor.ndim
        rev_dim = sinogram_tensor.ndim - 1 - dim_pos
        pad_spec[2 * rev_dim] = pad_left
        pad_spec[2 * rev_dim + 1] = pad_right
        x = torch.nn.functional.pad(sinogram_tensor, pad_spec, mode="constant", value=0.0)
    else:
        pad_left = 0
        x = sinogram_tensor

    scale = 1.0 / float(sample_spacing)
    # Wrapped signed lags give a symmetric kernel for even and odd FFT sizes.
    lags = torch.arange(n_pad, device=x.device)
    lags = torch.minimum(lags, n_pad - lags)
    impulse = torch.zeros(n_pad, device=x.device, dtype=x.real.dtype)
    impulse[0] = math.pi / 2.0
    odd = lags % 2 == 1
    impulse[odd] = -2.0 / (math.pi * lags[odd].to(impulse.dtype).square())

    if use_rfft and torch.is_floating_point(x):
        freqs = torch.fft.rfftfreq(n_pad, device=x.device, dtype=x.dtype)
        ramp = torch.fft.rfft(impulse).real * _ramp_window(window, freqs) * scale
        shape = [1] * x.ndim
        shape[dim_pos] = ramp.shape[0]
        ramp = ramp.reshape(shape)
        x_fft = torch.fft.rfft(x, dim=dim_pos)
        x_filtered = torch.fft.irfft(x_fft * ramp, n=n_pad, dim=dim_pos)
    else:
        freqs = torch.fft.fftfreq(n_pad, device=x.device, dtype=x.real.dtype)
        ramp = torch.fft.fft(impulse).real * _ramp_window(window, freqs) * scale
        shape = [1] * x.ndim
        shape[dim_pos] = ramp.shape[0]
        ramp = ramp.reshape(shape)
        x_fft = torch.fft.fft(x, dim=dim_pos)
        x_filtered = torch.real(torch.fft.ifft(x_fft * ramp, dim=dim_pos))

    if n_pad > n:
        x_filtered = x_filtered.narrow(dim_pos, pad_left, n).contiguous()
    return x_filtered


# ============================================================================
# Analytical FBP/FDK backprojection wrappers (voxel-driven gather)
# ============================================================================


def _as_contig_f32(t, device):
    return DeviceManager.ensure_device(t, device).to(dtype=torch.float32).contiguous()


def parallel_weighted_backproject(sinogram, ray_dir, det_origin, det_u_vec,
                                  detector_spacing, H, W, voxel_spacing=1.0):
    """Voxel-driven parallel-beam FBP backprojection with analytical constant.

    Runs the dedicated parallel-beam FBP gather kernel (one thread per
    output pixel, linear interpolation of the filtered sinogram, no
    distance weighting because parallel beam has no source) and
    multiplies by ``1 / (2 * pi)`` so a unit-density disk reconstructs
    to amplitude 1.

    Assumes the input sinogram has already been ramp-filtered with
    ``ramp_filter_1d`` and weighted by ``angular_integration_weights``.
    """
    if sinogram.shape[1] < 2:
        raise ValueError("Analytical backprojection requires at least two detector bins")
    device = DeviceManager.get_device(sinogram)
    sinogram = _as_contig_f32(sinogram, device)
    ray_dir = _as_contig_f32(ray_dir, device)
    det_origin = _as_contig_f32(det_origin, device)
    det_u_vec = _as_contig_f32(det_u_vec, device)

    n_views, n_det = sinogram.shape
    Ny, Nx = H, W
    reco = torch.zeros((Ny, Nx), dtype=torch.float32, device=device)

    d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
    d_reco = TorchCUDABridge.tensor_to_cuda_array(reco)
    d_ray_dir = TorchCUDABridge.tensor_to_cuda_array(ray_dir)
    d_det_origin = TorchCUDABridge.tensor_to_cuda_array(det_origin)
    d_det_u_vec = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)

    cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)
    grid, tpb = _grid_2d(Nx, Ny)

    pt_stream = torch.cuda.current_stream()
    numba_stream = _get_numba_external_stream_for(pt_stream)

    _parallel_2d_fbp_backproject_kernel[grid, tpb, numba_stream](
        d_sino, n_views, n_det, d_reco, Nx, Ny,
        _DTYPE(detector_spacing), d_ray_dir, d_det_origin, d_det_u_vec,
        cx, cy, _DTYPE(voxel_spacing),
    )

    return reco * (1.0 / (2.0 * math.pi))


def _analytical_geometry(src_pos, det_center, n, isocenter):
    """Resolve an orbit's physical isocenter and detector-normal distances."""
    n = n / torch.linalg.vector_norm(n, dim=1, keepdim=True).clamp(min=1e-9)
    cs = det_center - src_pos
    align = (cs * n).sum(dim=1, keepdim=True)
    n = torch.where(align < 0, -n, n)

    if isocenter is None:
        # Intersect source lines along detector normals in least squares.
        # Centering first makes the fit translation invariant even for a short
        # arc. CPU float64 supports rank-revealing solves on every CUDA device.
        source = src_pos.detach().to(device="cpu", dtype=torch.float64)
        normal = n.detach().to(device="cpu", dtype=torch.float64)
        normal = normal / torch.linalg.vector_norm(normal, dim=1, keepdim=True)
        reference = source.mean(dim=0)
        perpendicular = torch.eye(source.shape[1], dtype=source.dtype) - (
            normal[:, :, None] * normal[:, None, :]
        )
        rhs = (perpendicular @ (source - reference)[:, :, None]).sum(dim=0)
        # Rank tolerance must reflect the input geometry's precision, not the
        # solve dtype; float32 roundoff must not create a spurious intersection.
        fit = torch.linalg.lstsq(perpendicular.sum(dim=0), rhs,
                                  rcond=torch.finfo(src_pos.dtype).eps * source.shape[1])
        if int(fit.rank) < source.shape[1]:
            raise ValueError("Cannot infer isocenter from these views; supply isocenter explicitly")
        isocenter = reference + fit.solution[:, 0]
    isocenter = torch.as_tensor(isocenter, device=src_pos.device, dtype=src_pos.dtype)
    if isocenter.shape != (src_pos.shape[1],) or not torch.isfinite(isocenter).all():
        raise ValueError("isocenter must be a finite coordinate vector matching the trajectory")
    sid_n = ((isocenter - src_pos) * n).sum(dim=1)
    sdd_n = (cs * n).sum(dim=1)
    if not (torch.isfinite(sid_n).all() and torch.isfinite(sdd_n).all()
            and (sid_n > 1e-6).all() and (sdd_n > 1e-6).all()):
        raise ValueError("Source-to-isocenter and source-to-detector distances must be positive")
    return isocenter, float(sid_n.mean().item()), float(sdd_n.mean().item())


def fan_weighted_backproject(sinogram, src_pos, det_center, det_u_vec,
                             detector_spacing, H, W, voxel_spacing=1.0, *, isocenter=None):
    """Voxel-driven fan-beam FBP backprojection with analytical constant.

    Runs the dedicated fan-beam FBP gather kernel (one thread per output
    pixel, ``(sid_n/U_n)^2`` distance weighting, linear interpolation of
    the filtered sinogram) and multiplies by the analytical constant
    ``sdd_mean / (2 * pi * sid_mean)`` so a unit-density disk reconstructs
    to amplitude 1. Both means are measured in the direction of the
    detector normal per view, so the helper reduces to the classical
    ``sdd/(2*pi*sid)`` on a circular orbit. ``isocenter`` is a physical
    coordinate pair; by default it is inferred from the source lines along
    detector normals. Supply it explicitly for an ambiguous or non-circular
    acquisition. Analytical FBP remains acquisition-specific.
    """
    if sinogram.shape[1] < 2:
        raise ValueError("Analytical backprojection requires at least two detector bins")
    device = DeviceManager.get_device(sinogram)
    sinogram = _as_contig_f32(sinogram, device)
    src_pos = _as_contig_f32(src_pos, device)
    det_center = _as_contig_f32(det_center, device)
    det_u_vec = _as_contig_f32(det_u_vec, device)
    normal = torch.stack((-det_u_vec[:, 1], det_u_vec[:, 0]), dim=1)
    isocenter, sid_mean, sdd_mean = _analytical_geometry(
        src_pos, det_center, normal, isocenter,
    )

    n_views, n_det = sinogram.shape
    Ny, Nx = H, W
    reco = torch.zeros((Ny, Nx), dtype=torch.float32, device=device)

    d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
    d_reco = TorchCUDABridge.tensor_to_cuda_array(reco)
    d_src_pos = TorchCUDABridge.tensor_to_cuda_array(src_pos)
    d_det_center = TorchCUDABridge.tensor_to_cuda_array(det_center)
    d_det_u_vec = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)

    cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)
    grid, tpb = _grid_2d(Nx, Ny)

    pt_stream = torch.cuda.current_stream()
    numba_stream = _get_numba_external_stream_for(pt_stream)

    _fan_2d_fbp_backproject_kernel[grid, tpb, numba_stream](
        d_sino, n_views, n_det, d_reco, Nx, Ny,
        _DTYPE(detector_spacing), d_src_pos, d_det_center, d_det_u_vec,
        cx, cy, _DTYPE(voxel_spacing),
        _DTYPE(isocenter[0].item() / voxel_spacing),
        _DTYPE(isocenter[1].item() / voxel_spacing),
    )

    scale = sdd_mean / (2.0 * math.pi * sid_mean)
    return reco * scale


def cone_weighted_backproject(sinogram, src_pos, det_center, det_u_vec, det_v_vec,
                              D, H, W, du, dv, voxel_spacing=1.0, *, isocenter=None):
    """Voxel-driven cone-beam FDK backprojection with analytical constant.

    Runs the dedicated cone-beam FDK gather kernel (one thread per output
    voxel, ``(sid_n/U_n)^2`` distance weighting, bilinear interpolation of
    the filtered sinogram) and multiplies by ``sdd_mean/(2*pi*sid_mean)``
    so a unit-density sphere reconstructs to amplitude 1. Both means are
    taken in the per-view detector-normal direction, so the helper
    reduces to the classical ``sdd/(2*pi*sid)`` on a circular orbit.
    ``isocenter`` is a physical coordinate triple, inferred from source
    lines along detector normals by default. Supply it explicitly for an
    ambiguous or non-circular acquisition. FDK remains an approximation
    outside its circular-acquisition assumptions.
    """
    if min(sinogram.shape[1:]) < 2:
        raise ValueError("Analytical backprojection requires at least two detector bins on each axis")
    device = DeviceManager.get_device(sinogram)
    sinogram = _as_contig_f32(sinogram, device)
    src_pos = _as_contig_f32(src_pos, device)
    det_center = _as_contig_f32(det_center, device)
    det_u_vec = _as_contig_f32(det_u_vec, device)
    det_v_vec = _as_contig_f32(det_v_vec, device)
    normal = torch.cross(det_u_vec, det_v_vec, dim=1)
    isocenter, sid_mean, sdd_mean = _analytical_geometry(
        src_pos, det_center, normal, isocenter,
    )

    n_views, n_u, n_v = sinogram.shape
    Nx, Ny, Nz = W, H, D
    # Allocate reco in WHD layout to match the FDK gather kernel's
    # coalescing (iz warp-adjacent, stride-1 along the D axis).
    reco_perm = torch.zeros((Nx, Ny, Nz), dtype=torch.float32, device=device)

    d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
    d_reco = TorchCUDABridge.tensor_to_cuda_array(reco_perm)
    d_src_pos = TorchCUDABridge.tensor_to_cuda_array(src_pos)
    d_det_center = TorchCUDABridge.tensor_to_cuda_array(det_center)
    d_det_u_vec = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)
    d_det_v_vec = TorchCUDABridge.tensor_to_cuda_array(det_v_vec)

    cx = _DTYPE(Nx * 0.5)
    cy = _DTYPE(Ny * 0.5)
    cz = _DTYPE(Nz * 0.5)
    grid, tpb = _grid_3d(Nz, Ny, Nx)

    pt_stream = torch.cuda.current_stream()
    numba_stream = _get_numba_external_stream_for(pt_stream)

    _cone_3d_fdk_backproject_kernel[grid, tpb, numba_stream](
        d_sino, n_views, n_u, n_v, d_reco, Nx, Ny, Nz,
        _DTYPE(du), _DTYPE(dv),
        d_src_pos, d_det_center, d_det_u_vec, d_det_v_vec,
        cx, cy, cz, _DTYPE(voxel_spacing),
        _DTYPE(isocenter[0].item() / voxel_spacing),
        _DTYPE(isocenter[1].item() / voxel_spacing),
        _DTYPE(isocenter[2].item() / voxel_spacing),
    )

    scale = sdd_mean / (2.0 * math.pi * sid_mean)

    # Convert back to (D, H, W) layout for the user.
    reco = reco_perm.permute(2, 1, 0).contiguous() * scale
    return reco
