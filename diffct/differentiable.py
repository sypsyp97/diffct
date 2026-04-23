import math
import numpy as np
import torch
from numba import cuda

# ---------------------------------------------------------------------------
# Global settings & helpers
# ---------------------------------------------------------------------------

_DTYPE              = np.float32
# CUDA thread block configurations optimized for different dimensionalities
# 2D blocks: 16x16 = 256 threads per block, optimal for 2D ray-tracing kernels
# Balances occupancy with shared memory usage for parallel/fan beam projections
_TPB_2D             = (16, 16)
# 3D blocks: 8x8x8 = 512 threads per block, optimal for 3D cone beam kernels
# Smaller per-dimension size accommodates higher register usage in 3D algorithms
_TPB_3D             = (8,  8,  8)
# Smaller TPB for the 3-D SF kernels. SF-TT in particular spills past the
# 128-register-per-thread budget that (8, 8, 8) allows, so we drop to
# (4, 4, 4) = 64 threads/block, giving each thread ~1024 registers.
_TPB_SF_3D          = (4,  4,  4)
# CUDA fastmath optimization: enables aggressive floating-point optimizations
# Trades numerical precision for performance in ray-tracing calculations
# Safe for CT reconstruction where slight precision loss is acceptable for speed gains
_FASTMATH_DECORATOR = cuda.jit(cache=True, fastmath=True)
# The analytical FDK gather kernel gets its own decorator with fastmath disabled.
# FDK is a one-shot reconstruction so the few-percent speedup of fastmath is not
# worth the loss of IEEE-correct rounding in the (sid/U)^2 distance weighting and
# the bilinear detector interpolation near detector edges.
_FDK_ACCURACY_DECORATOR = cuda.jit(cache=True, fastmath=False)

_INF                = _DTYPE(np.inf)
_NEG_INF            = _DTYPE(-np.inf)
_ZERO               = _DTYPE(0.0)
_ONE                = _DTYPE(1.0)
_HALF               = _DTYPE(0.5)
_QUARTER            = _DTYPE(0.25)
_EPSILON            = _DTYPE(1e-6)
# === Device Management Utilities ===
class DeviceManager:
    @staticmethod
    def get_device(tensor):
        """Get the device of a PyTorch tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor whose device to determine.

        Returns
        -------
        torch.device
            Device of the tensor or CPU if unavailable.

        Examples
        --------
        >>> DeviceManager.get_device(torch.tensor([1, 2, 3]))
        device(type='cpu')
        """
        return tensor.device if hasattr(tensor, "device") else torch.device("cpu")

    @staticmethod
    def ensure_device(tensor, device):
        """Ensure a tensor resides on a given device.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to move.
        device : torch.device
            Desired device.

        Returns
        -------
        torch.Tensor
            Tensor on the specified device. Unchanged if already on it.

        Examples
        --------
        >>> DeviceManager.ensure_device(
        ...     torch.tensor([1, 2, 3]),
        ...     torch.device('cuda')
        ... )
        tensor([1, 2, 3], device='cuda:0')
        """
        if hasattr(tensor, "to") and tensor.device != device:
            return tensor.to(device)
        return tensor

# === PyTorch-CUDA Bridge ===
class TorchCUDABridge:
    @staticmethod
    def tensor_to_cuda_array(tensor):
        """Convert a PyTorch CUDA tensor to a Numba CUDA DeviceNDArray.

        Provides a zero-copy view of a detached PyTorch tensor as a Numba CUDA array,
        avoiding CPU data transfers. The returned array shares memory with the
        original tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            PyTorch tensor on a CUDA device.

        Returns
        -------
        numba.cuda.cudadrv.devicearray.DeviceNDArray
            Numba CUDA array view sharing memory with `tensor`.

        Raises
        ------
        ValueError
            If `tensor` is not on a CUDA device.

        Examples
        --------
        >>> t = torch.randn(10, device='cuda')
        >>> arr = TorchCUDABridge.tensor_to_cuda_array(t)
        """
        if not tensor.is_cuda:
            raise ValueError("Tensor must be on CUDA device")
        return cuda.as_cuda_array(tensor.detach())

# ---------------------------------------------------------------------------
# Stream helper (cached external Numba stream)
# ---------------------------------------------------------------------------
_cached_stream_ptr = None
_cached_numba_stream = None

def _get_numba_external_stream_for(pt_stream=None):
    """
    Return a cached numba.cuda.external_stream for the current PyTorch CUDA stream.
    Caches by the underlying CUDA stream pointer to avoid repeated construction.
    """
    global _cached_stream_ptr, _cached_numba_stream
    if pt_stream is None:
        pt_stream = torch.cuda.current_stream()
    # Torch exposes an underlying CUDA stream handle via .cuda_stream
    ptr = int(pt_stream.cuda_stream)
    if _cached_stream_ptr == ptr and _cached_numba_stream is not None:
        return _cached_numba_stream
    numba_stream = cuda.external_stream(pt_stream.cuda_stream)
    _cached_stream_ptr = ptr
    _cached_numba_stream = numba_stream
    return numba_stream

# === GPU-aware Trigonometric Table Generation ===
# Caching removed: torch.Tensor is unhashable for lru_cache
def _trig_tables(angles, dtype=_DTYPE, device=None):
    """Compute cosine and sine tables for input angles.

    Precompute cosine and sine values and return as torch tensors on the
    same device as `angles`.

    Parameters
    ----------
    angles : array-like or torch.Tensor
        Projection angles in radians. Can be a NumPy array or a PyTorch tensor on CPU or CUDA.
    dtype : numpy.dtype or torch.dtype, optional
        Desired data type for output tables. Default is `_DTYPE`.

    Returns
    -------
    cos : torch.Tensor
        Cosine values of `angles` on the same device.
    sin : torch.Tensor
        Sine values of `angles` on the same device.

    Examples
    --------
    >>> angles = torch.linspace(0, torch.pi, 180, device='cuda')
    >>> cos, sin = _trig_tables(angles)
    >>> cos.device
    device(type='cuda', index=0)
    """
    if isinstance(angles, torch.Tensor):
        device = angles.device if device is None else device
        # Compute both cos and sin in one call to avoid redundant kernel launches
        angles_device = angles.to(dtype=dtype, device=device)
        cos = torch.cos(angles_device)
        sin = torch.sin(angles_device)
        return cos, sin
    else:
        # fallback for non-tensor inputs: compute via PyTorch on CPU for consistency
        # Determine desired torch dtype
        if isinstance(dtype, torch.dtype):
            torch_dtype = dtype
        else:
            _NP_TO_TORCH = {
                np.float32: torch.float32,
                np.float64: torch.float64,
            }
            torch_dtype = _NP_TO_TORCH.get(dtype, torch.float32)
        # Convert input angles to a CPU torch tensor and compute both simultaneously
        angles_cpu = torch.tensor(angles, dtype=torch_dtype)
        cos_cpu = torch.cos(angles_cpu)
        sin_cpu = torch.sin(angles_cpu)
        if device is not None:
            return cos_cpu.to(device), sin_cpu.to(device)
        else:
            return cos_cpu, sin_cpu


# ############################################################################
# MEMORY LAYOUT VALIDATION
# ############################################################################

def _validate_3d_memory_layout(tensor, expected_order='DHW'):
    """Validate 3D tensor memory layout to prevent coordinate system inconsistencies.

    Parameters
    ----------
    tensor : torch.Tensor
        3D tensor to validate
    expected_order : str, optional
        Expected memory order ('DHW', 'VHW', etc.). Default is 'DHW'.

    Raises
    ------
    ValueError
        If tensor has unexpected memory layout or is non-contiguous
    """
    shape = tensor.shape
    if len(shape) != 3:
        raise ValueError(f"Expected 3D tensor, got {len(shape)}D")

    # Early return for common case - contiguous tensor with expected ordering
    if tensor.is_contiguous() and expected_order in ('DHW', 'VHW'):
        # For DHW and VHW, the expected order matches memory layout when contiguous
        return
    
    # Only check memory order for DHW and VHW, not for internal WHD layout
    if expected_order in ('DHW', 'VHW'):
        if not tensor.is_contiguous():
            raise ValueError(
                "Input tensor must be contiguous. Call .contiguous() before passing to "
                "cone beam functions to avoid memory duplication and ensure correct results."
            )

        strides = tensor.stride()
        order_mapping = {
            'DHW': (0, 1, 2),  # Depth, Height, Width
            'VHW': (0, 1, 2),  # Views, Height, Width (for sinograms)
        }
        if expected_order not in order_mapping:
            raise ValueError(f"Unsupported expected_order: {expected_order}")

        expected_stride_order = order_mapping[expected_order]
        # Check if actual strides match expected order
        sorted_strides = sorted(enumerate(strides), key=lambda x: x[1], reverse=True)
        actual_order = tuple(idx for idx, _ in sorted_strides)

        if actual_order != expected_stride_order:
            # Create appropriate error message based on context
            if expected_order == 'VHW':
                actual_str = f"({shape[0]}, {shape[1]}, {shape[2]})"
                expected_str = "(Views, Height, Width)"
                fix_str = "ensure your sinogram has shape (num_views, det_v, det_u)"
            elif expected_order == 'DHW':
                actual_str = f"({shape[0]}, {shape[1]}, {shape[2]})"
                expected_str = "(Depth, Height, Width)"
                fix_str = "ensure your volume has shape (D, H, W)"
            else:
                actual_str = str(tuple(shape))
                expected_str = expected_order
                fix_str = "check tensor dimensions"

            raise ValueError(
                f"Memory layout mismatch: expected {expected_str} order, "
                f"but tensor has shape {actual_str}. Please {fix_str} and ensure "
                f"the tensor is contiguous (.contiguous()) before passing to the function."
            )
    # For 'WHD' (internal layout), skip stride check entirely


def _grid_2d(n1, n2, tpb=_TPB_2D):
    """Compute 2D CUDA grid and block dimensions.

    Determine optimal grid and block sizes for 2D CUDA ray-tracing kernels.

    Parameters
    ----------
    n1 : int
        Number of elements along the first dimension (e.g., projection angles).
    n2 : int
        Number of elements along the second dimension (e.g., detector elements).
    tpb : tuple of int, optional
        Threads per block (default is `_TPB_2D`) to balance occupancy and memory.

    Returns
    -------
    grid : tuple of int
        Blocks count per axis.
    tpb : tuple of int
        Threads per block per axis.

    Examples
    --------
    >>> grid, tpb = _grid_2d(180, 256)
    >>> grid
    (12, 16)
    >>> tpb
    (16, 16)
    """
    return (math.ceil(n1 / tpb[0]), math.ceil(n2 / tpb[1])), tpb


def _grid_3d(n1, n2, n3, tpb=_TPB_3D):
    """Compute 3D CUDA grid and block dimensions.

    Determine optimal grid and block sizes for 3D CUDA cone-beam kernels.

    Parameters
    ----------
    n1 : int
        Number of elements along the first dimension (e.g., projection views).
    n2 : int
        Number of elements along the second dimension (e.g., detector u-axis).
    n3 : int
        Number of elements along the third dimension (e.g., detector v-axis).
    tpb : tuple of int, optional
        Threads per block (default is `_TPB_3D`) to balance occupancy and registers.

    Returns
    -------
    grid : tuple of int
        Blocks count per axis.
    tpb : tuple of int
        Threads per block per axis.

    Examples
    --------
    >>> grid, tpb = _grid_3d(360, 256, 256)
    >>> grid
    (45, 32, 32)
    >>> tpb
    (8, 8, 8)
    """
    return (
        math.ceil(n1 / tpb[0]),
        math.ceil(n2 / tpb[1]),
        math.ceil(n3 / tpb[2]),
    ), tpb


def detector_coordinates_1d(num_detectors, detector_spacing, detector_offset=0.0, device=None, dtype=torch.float32):
    """Return centered detector coordinates in physical units.

    Coordinates follow the convention ``(i - (N-1)/2) * spacing + offset``.
    This avoids a half-pixel center bias for even detector counts.
    """
    idx = torch.arange(num_detectors, device=device, dtype=dtype)
    return (idx - (num_detectors - 1) * 0.5) * detector_spacing + detector_offset


def angular_integration_weights(angles, redundant_full_scan=True):
    """Compute per-view integration weights from the provided angle samples.

    Parameters
    ----------
    angles : torch.Tensor
        1D projection angles in radians.
    redundant_full_scan : bool, optional
        If ``True``, applies a 0.5 factor for near-``2*pi`` scans to account for
        view redundancy in reconstruction formulas using full circular data.
    """
    if not isinstance(angles, torch.Tensor):
        angles = torch.tensor(angles, dtype=torch.float32)
    a = angles.to(dtype=torch.float32)
    if a.ndim != 1 or a.numel() < 2:
        raise ValueError("angles must be a 1D tensor with at least 2 elements")

    a_sorted, sort_idx = torch.sort(a)
    d = a_sorted[1:] - a_sorted[:-1]
    median_step = torch.median(d)
    span = a_sorted[-1] - a_sorted[0]

    period = None
    for candidate in (math.pi, 2.0 * math.pi):
        closure = a.new_tensor(candidate) - span
        tol = max(1e-4, 0.05 * float(abs(median_step).item()))
        if closure >= -tol and float(closure.item()) <= 1.5 * float(abs(median_step).item()) + tol:
            period = candidate
            break

    if period is not None:
        closure = torch.clamp(a.new_tensor(period) - span, min=0.0)
        w_sorted = torch.empty_like(a_sorted)
        w_sorted[0] = 0.5 * (closure + d[0])
        w_sorted[-1] = 0.5 * (d[-1] + closure)
        if a.numel() > 2:
            w_sorted[1:-1] = 0.5 * (d[:-1] + d[1:])
        if redundant_full_scan and abs(period - 2.0 * math.pi) <= max(1e-4, 0.05 * float(abs(median_step).item())):
            w_sorted = 0.5 * w_sorted
    else:
        # Non-periodic trapezoidal weights for partial scans.
        w_sorted = torch.empty_like(a_sorted)
        w_sorted[1:-1] = 0.5 * (a_sorted[2:] - a_sorted[:-2])
        w_sorted[0] = 0.5 * (a_sorted[1] - a_sorted[0])
        w_sorted[-1] = 0.5 * (a_sorted[-1] - a_sorted[-2])
        w_sorted = torch.abs(w_sorted)

    w = torch.empty_like(w_sorted)
    w[sort_idx] = w_sorted
    if (
        redundant_full_scan
        and period is None
        and abs(float((span - 2.0 * math.pi).item())) <= max(1e-4, 0.05 * float(abs(median_step).item()))
    ):
        w = 0.5 * w
    return w


def fan_cosine_weights(num_detectors, detector_spacing, sdd, detector_offset=0.0, device=None, dtype=torch.float32):
    """Return fan-beam cosine pre-weights ``cos(gamma)`` for each detector bin."""
    u = detector_coordinates_1d(num_detectors, detector_spacing, detector_offset, device=device, dtype=dtype)
    gamma = torch.atan(u / sdd)
    return torch.cos(gamma)


def cone_cosine_weights(det_u, det_v, du, dv, sdd, detector_offset_u=0.0, detector_offset_v=0.0, device=None, dtype=torch.float32):
    """Return FDK cosine pre-weights ``D/sqrt(D^2 + u^2 + v^2)`` on a 2D detector."""
    u = detector_coordinates_1d(det_u, du, detector_offset_u, device=device, dtype=dtype).view(det_u, 1)
    v = detector_coordinates_1d(det_v, dv, detector_offset_v, device=device, dtype=dtype).view(1, det_v)
    return sdd / torch.sqrt(sdd * sdd + u * u + v * v)


def parker_weights(angles, num_detectors, detector_spacing, sdd, detector_offset=0.0):
    """Return Parker redundancy weights for fan/cone short-scan geometries.

    For full scans (near ``2*pi``), this returns all ones.
    """
    if not isinstance(angles, torch.Tensor):
        angles = torch.tensor(angles, dtype=torch.float32)
    a = angles.to(dtype=torch.float32)
    if a.ndim != 1 or a.numel() < 2:
        raise ValueError("angles must be a 1D tensor with at least 2 elements")

    # Approximate covered range including the final sample interval.
    coverage = torch.abs(a[-1] - a[0]) + torch.abs(a[1] - a[0])
    if coverage >= (2.0 * torch.pi - 1e-3):
        return torch.ones((a.numel(), num_detectors), dtype=a.dtype, device=a.device)

    u = detector_coordinates_1d(
        num_detectors,
        detector_spacing,
        detector_offset=detector_offset,
        device=a.device,
        dtype=a.dtype,
    )
    gamma = torch.atan(u / sdd).view(1, num_detectors)
    gamma_max = torch.max(torch.abs(gamma))
    min_short_scan = torch.pi + 2.0 * gamma_max
    if coverage < (min_short_scan - 1e-3):
        raise ValueError(
            "Insufficient angular coverage for Parker weighting. "
            f"Need at least {float(min_short_scan):.6f} rad, got {float(coverage):.6f} rad."
        )

    beta = (a - a[0]).view(-1, 1)
    eps = 1e-6

    # Exact Parker form for minimal short scan (pi + 2*gamma_max).
    if coverage <= (min_short_scan + 1e-3):
        t1 = 2.0 * (gamma_max - gamma)
        t2 = torch.pi - 2.0 * gamma
        t3 = torch.pi + 2.0 * gamma_max
        t4 = 2.0 * (gamma_max + gamma)

        w1 = 0.5 * (1.0 - torch.cos(torch.pi * beta / torch.clamp(t1, min=eps)))
        w3 = 0.5 * (1.0 - torch.cos(torch.pi * (t3 - beta) / torch.clamp(t4, min=eps)))

        cond1 = (beta >= 0.0) & (beta < t1)
        cond2 = (beta >= t1) & (beta <= t2)
        cond3 = (beta > t2) & (beta <= t3)
        ones = torch.ones_like(w1)
        zeros = torch.zeros_like(w1)
        return torch.where(cond1, w1, torch.where(cond2, ones, torch.where(cond3, w3, zeros)))

    # Fallback for over-scan (<2*pi): cosine taper at both ends.
    ramp = 2.0 * gamma_max
    wb = torch.ones_like(a)
    if ramp > eps:
        b = beta[:, 0]
        lead = 0.5 * (1.0 - torch.cos(torch.pi * torch.clamp(b / ramp, min=0.0, max=1.0)))
        trail = 0.5 * (1.0 - torch.cos(torch.pi * torch.clamp((coverage - b) / ramp, min=0.0, max=1.0)))
        wb = torch.minimum(torch.maximum(lead, trail), torch.ones_like(lead))
    return wb.view(-1, 1).expand(-1, num_detectors)


def _ramp_window(name, freqs):
    """Build a frequency-domain apodization window for the ramp filter.

    ``freqs`` are in cycles per sample (same convention as ``torch.fft.fftfreq``
    and ``torch.fft.rfftfreq``). The normalized coordinate ``nf = 2 * |freqs|``
    maps DC to 0 and the Nyquist bin to 1.
    """
    if name is None:
        return torch.ones_like(freqs)
    nf = torch.abs(freqs) * 2.0  # [0, 1] at Nyquist
    nf = torch.clamp(nf, max=1.0)
    key = name.lower().replace("_", "-")
    if key in ("hann", "hanning"):
        return 0.5 * (1.0 + torch.cos(torch.pi * nf))
    if key in ("hamming",):
        return 0.54 + 0.46 * torch.cos(torch.pi * nf)
    if key in ("cosine",):
        return torch.cos(torch.pi * nf * 0.5)
    if key in ("shepp-logan",):
        # sinc(nf/2); sinc(0)=1 by limit.
        arg = torch.pi * nf * 0.5
        return torch.where(
            nf > 1e-8, torch.sin(arg) / torch.clamp(arg, min=1e-8), torch.ones_like(nf)
        )
    if key in ("ram-lak", "ramlak", "none"):
        return torch.ones_like(freqs)
    raise ValueError(f"Unknown ramp window '{name}'")


def ramp_filter_1d(
    sinogram_tensor,
    dim=-1,
    sample_spacing=1.0,
    pad_factor=1,
    window=None,
    use_rfft=True,
):
    """Apply a 1D ramp filter along ``dim`` using FFT.

    Parameters
    ----------
    sinogram_tensor : torch.Tensor
        Real-valued sinogram tensor.
    dim : int, optional
        Axis along which to filter. Default ``-1``.
    sample_spacing : float, optional
        Physical spacing between detector samples along ``dim``. The continuous
        ramp filter has units of ``1/length``, so the FFT output is rescaled by
        ``1/sample_spacing`` to respect physical units. Default ``1.0`` (pure
        sample-unit behavior, matching the historical call signature).
    pad_factor : int, optional
        Zero-pad the signal to ``pad_factor * N`` along ``dim`` before the FFT
        to suppress circular-convolution wrap-around artifacts. Default ``1``
        (no padding).
    window : str or None, optional
        Frequency-domain apodization: ``None``/``"ram-lak"`` (unwindowed),
        ``"hann"``, ``"hamming"``, ``"cosine"``, or ``"shepp-logan"``.
        Default ``None``.
    use_rfft : bool, optional
        Use the real-valued FFT path when ``True``. Default ``True``.

    Notes
    -----
    Pre-existing callers that pass only ``(sinogram_tensor, dim)`` keep the same
    filter shape because the defaults preserve the original behavior up to the
    rfft path (which is numerically equivalent for real inputs) and the
    ``1/sample_spacing`` scale (which is exactly 1 when ``sample_spacing=1.0``).
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

    if use_rfft and torch.is_floating_point(x):
        freqs = torch.fft.rfftfreq(n_pad, device=x.device, dtype=x.dtype)
        ramp = torch.abs(2.0 * torch.pi * freqs) * _ramp_window(window, freqs) * scale
        shape = [1] * x.ndim
        shape[dim_pos] = ramp.shape[0]
        ramp = ramp.reshape(shape)
        x_fft = torch.fft.rfft(x, dim=dim_pos)
        x_filtered = torch.fft.irfft(x_fft * ramp, n=n_pad, dim=dim_pos)
    else:
        freqs = torch.fft.fftfreq(n_pad, device=x.device, dtype=torch.float32)
        freqs = freqs.to(dtype=x.real.dtype if x.is_complex() else x.dtype)
        ramp = torch.abs(2.0 * torch.pi * freqs) * _ramp_window(window, freqs) * scale
        shape = [1] * x.ndim
        shape[dim_pos] = ramp.shape[0]
        ramp = ramp.reshape(shape)
        x_fft = torch.fft.fft(x, dim=dim_pos)
        x_filtered = torch.real(torch.fft.ifft(x_fft * ramp, dim=dim_pos))

    if n_pad > n:
        x_filtered = x_filtered.narrow(dim_pos, pad_left, n).contiguous()
    return x_filtered


# ############################################################################
# SHARED CUDA KERNELS
# ############################################################################

# ------------------------------------------------------------------
# 2-D PARALLEL BEAM KERNELS
# ------------------------------------------------------------------

@_FASTMATH_DECORATOR
def _parallel_2d_forward_kernel(
    d_image, Nx, Ny,
    d_sino, n_ang, n_det,
    det_spacing, d_cos, d_sin, cx, cy, voxel_spacing,
    det_offset, center_offset_x, center_offset_y
):
    """Compute the 2D parallel beam forward projection.

    This CUDA kernel implements cell-constant Siddon ray tracing for
    2D parallel beam forward projection.

    Parameters
    ----------
    d_image : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Input 2D image array on CUDA.
    Nx : int
        Number of voxels along the x-axis.
    Ny : int
        Number of voxels along the y-axis.
    d_sino : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Output sinogram array on CUDA.
    n_ang : int
        Number of projection angles.
    n_det : int
        Number of detector elements.
    det_spacing : float
        Physical spacing between detector elements.
    d_cos : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Precomputed cosine values of projection angles.
    d_sin : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Precomputed sine values of projection angles.
    cx : float
        Half of image width in voxels.
    cy : float
        Half of image height in voxels.
    voxel_spacing : float
        Physical size of one voxel (in same units as det_spacing, sid, sdd).

    Notes
    -----
    The cell-constant Siddon method provides accurate ray-volume intersection by:
      - Calculating ray-volume boundary intersections to define traversal limits.
      - Iterating through voxels along the ray path via parametric equations.
      - Aggregating each traversed pixel value based on ray segment lengths.
    """
    # CUDA THREAD ORGANIZATION: 2D grid maps directly to ray geometry
    # Each thread processes one ray defined by (projection_angle, detector_element) pair
    # Thread indexing: iang = projection angle index, idet = detector element index
    iang, idet = cuda.grid(2)
    if iang >= n_ang or idet >= n_det:
        return

    # === RAY GEOMETRY SETUP ===
    # Extract projection angle and compute detector position
    cos_a = d_cos[iang]  # Precomputed cosine of projection angle
    sin_a = d_sin[iang]  # Precomputed sine of projection angle
    # Normalize all physical distances to voxel units
    u = (np.float32(idet) - (np.float32(n_det) - _ONE) * _HALF) * det_spacing / voxel_spacing + det_offset

    # Define ray direction and starting point for parallel beam geometry
    # Ray direction is perpendicular to detector array (cos_a, sin_a)
    # Ray starting point is offset along detector by distance u in voxel units
    dir_x, dir_y = cos_a, sin_a
    pnt_x = u * -sin_a + center_offset_x
    pnt_y = u * cos_a + center_offset_y

    # === RAY-VOLUME INTERSECTION CALCULATION ===
    # Compute parametric intersection points with volume boundaries using ray equation r(t) = pnt + t*dir
    # Volume extends from [-cx, cx] x [-cy, cy] in voxel coordinate system
    # Mathematical basis: For ray r(t) = origin + t*direction, solve r(t) = boundary for parameter t
    t_min, t_max = _NEG_INF, _INF  # Initialize ray parameter range to unbounded
    
    # X-direction boundary intersections
    # Handle non-parallel rays: compute intersection parameters with left (-cx) and right (+cx) boundaries
    if abs(dir_x) > _EPSILON:  # Ray not parallel to x-axis (avoid division by zero)
        tx1, tx2 = (-cx - pnt_x) / dir_x, (cx - pnt_x) / dir_x  # Left and right boundary intersections
        # Update valid parameter range: intersection of current range with x-boundary constraints
        # min/max operations ensure we get the entry/exit points correctly regardless of ray direction
        t_min, t_max = max(t_min, min(tx1, tx2)), min(t_max, max(tx1, tx2))  # Update valid parameter range
    elif pnt_x < -cx or pnt_x > cx:  # Ray parallel to x-axis but outside volume bounds
        # Edge case: ray never intersects volume if parallel and outside boundaries
        d_sino[iang, idet] = _ZERO; return

    # Y-direction boundary intersections (identical logic to x-direction)
    # Handle non-parallel rays: compute intersection parameters with bottom (-cy) and top (+cy) boundaries
    if abs(dir_y) > _EPSILON:  # Ray not parallel to y-axis (avoid division by zero)
        ty1, ty2 = (-cy - pnt_y) / dir_y, (cy - pnt_y) / dir_y  # Bottom and top boundary intersections
        # Intersect y-boundary constraints with existing parameter range from x-boundaries
        t_min, t_max = max(t_min, min(ty1, ty2)), min(t_max, max(ty1, ty2))  # Intersect with x-range
    elif pnt_y < -cy or pnt_y > cy:  # Ray parallel to y-axis but outside volume bounds
        # Edge case: ray never intersects volume if parallel and outside boundaries
        d_sino[iang, idet] = _ZERO; return

    # Boundary intersection validation: check if ray actually intersects the volume
    # If t_min >= t_max, the ray misses the volume entirely (no valid intersection interval)
    if t_min >= t_max:
        d_sino[iang, idet] = _ZERO; return

    # === SIDDON METHOD VOXEL TRAVERSAL INITIALIZATION ===
    accum = _ZERO  # Accumulated projection value along ray
    t = t_min    # Current ray parameter (distance from ray start)
    
    # Convert ray entry point to voxel indices (image coordinate system)
    ix = int(math.floor(pnt_x + t * dir_x + cx))  # Current voxel x-index
    iy = int(math.floor(pnt_y + t * dir_y + cy))  # Current voxel y-index

    # Determine traversal direction and step sizes for each axis
    step_x, step_y = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1)  # Voxel stepping direction
    # Hoist inverse directions to reduce divisions and branches
    inv_dir_x = (_ONE / dir_x) if abs(dir_x) > _EPSILON else _ZERO
    inv_dir_y = (_ONE / dir_y) if abs(dir_y) > _EPSILON else _ZERO
    dt_x = abs(inv_dir_x) if abs(dir_x) > _EPSILON else _INF
    dt_y = abs(inv_dir_y) if abs(dir_y) > _EPSILON else _INF

    # Calculate parameter values for next voxel boundary crossings using inv_dir_*
    next_ix = ix + (1 if step_x > 0 else 0)
    next_iy = iy + (1 if step_y > 0 else 0)
    tx = (np.float32(next_ix) - cx - pnt_x) * inv_dir_x if abs(dir_x) > _EPSILON else _INF
    ty = (np.float32(next_iy) - cy - pnt_y) * inv_dir_y if abs(dir_y) > _EPSILON else _INF

    # === MAIN RAY TRAVERSAL LOOP ===
    # Step through voxels along ray path, accumulating cell-constant contributions.
    while t < t_max:
        if 0 <= ix < Nx and 0 <= iy < Ny:
            # Determine next voxel boundary crossing (minimum of x, y boundaries or ray exit)
            t_next = min(tx, ty, t_max)
            seg_len = t_next - t  # Length of ray segment within current voxel region
            if seg_len > _EPSILON:  # Only process segments with meaningful length (avoid numerical noise)
                accum += d_image[iy, ix] * seg_len
        
        # === VOXEL BOUNDARY CROSSING LOGIC ===
        # Advance to next voxel based on which boundary is crossed first
        if tx <= ty:  # X-boundary crossed first
            t = tx
            ix += step_x  # Move to next voxel in x-direction
            tx += dt_x    # Update next x-boundary crossing parameter
        else:         # Y-boundary crossed first
            t = ty
            iy += step_y  # Move to next voxel in y-direction
            ty += dt_y    # Update next y-boundary crossing parameter
    
    d_sino[iang, idet] = accum

@_FASTMATH_DECORATOR
def _parallel_2d_backward_kernel(
    d_sino, n_ang, n_det,
    d_image, Nx, Ny,
    det_spacing, d_cos, d_sin, cx, cy, voxel_spacing,
    det_offset, center_offset_x, center_offset_y
):
    """Compute the 2D parallel beam backprojection.

    This CUDA kernel implements the adjoint of cell-constant Siddon ray tracing for
    2D parallel beam backprojection.

    Parameters
    ----------
    d_sino : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Input sinogram array on CUDA.
    n_ang : int
        Number of projection angles.
    n_det : int
        Number of detector elements.
    d_image : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Output image gradient array on CUDA.
    Nx : int
        Number of voxels along the x-axis.
    Ny : int
        Number of voxels along the y-axis.
    det_spacing : float
        Physical spacing between detector elements.
    d_cos : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Precomputed cosine values of projection angles.
    d_sin : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Precomputed sine values of projection angles.
    cx : float
        Half of image width in voxels.
    cy : float
        Half of image height in voxels.
    voxel_spacing : float
        Physical size of one voxel (in same units as det_spacing, sid, sdd).

    Notes
    -----
    This operation is the adjoint of the forward projection. Sinogram values
    are distributed back into the volume along identical ray paths using
    atomic operations to ensure thread-safe accumulation.
    """
    iang, idet = cuda.grid(2)
    if iang >= n_ang or idet >= n_det:
        return

    # === RAY GEOMETRY SETUP (identical to forward projection) ===
    val   = d_sino[iang, idet]  # Sinogram value to backproject
    cos_a = d_cos[iang]         # Precomputed cosine of projection angle
    sin_a = d_sin[iang]         # Precomputed sine of projection angle
    # Normalize all physical distances to voxel units
    u = (np.float32(idet) - (np.float32(n_det) - _ONE) * _HALF) * det_spacing / voxel_spacing + det_offset

    # Define ray direction and starting point for parallel beam geometry
    dir_x, dir_y = cos_a, sin_a
    pnt_x = u * -sin_a + center_offset_x
    pnt_y = u * cos_a + center_offset_y

    # === RAY-VOLUME INTERSECTION CALCULATION (identical to forward) ===
    t_min, t_max = _NEG_INF, _INF
    if abs(dir_x) > _EPSILON:
        tx1, tx2 = (-cx - pnt_x) / dir_x, (cx - pnt_x) / dir_x
        t_min, t_max = max(t_min, min(tx1, tx2)), min(t_max, max(tx1, tx2))
    elif pnt_x < -cx or pnt_x > cx: return

    if abs(dir_y) > _EPSILON:
        ty1, ty2 = (-cy - pnt_y) / dir_y, (cy - pnt_y) / dir_y
        t_min, t_max = max(t_min, min(ty1, ty2)), min(t_max, max(ty1, ty2))
    elif pnt_y < -cy or pnt_y > cy: return

    if t_min >= t_max: return

    # === SIDDON METHOD TRAVERSAL INITIALIZATION ===
    t = t_min
    ix = int(math.floor(pnt_x + t * dir_x + cx))
    iy = int(math.floor(pnt_y + t * dir_y + cy))

    step_x, step_y = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1)
    inv_dir_x = (_ONE / dir_x) if abs(dir_x) > _EPSILON else _ZERO
    inv_dir_y = (_ONE / dir_y) if abs(dir_y) > _EPSILON else _ZERO
    dt_x = abs(inv_dir_x) if abs(dir_x) > _EPSILON else _INF
    dt_y = abs(inv_dir_y) if abs(dir_y) > _EPSILON else _INF
    next_ix = ix + (1 if step_x > 0 else 0)
    next_iy = iy + (1 if step_y > 0 else 0)
    tx = (np.float32(next_ix) - cx - pnt_x) * inv_dir_x if abs(dir_x) > _EPSILON else _INF
    ty = (np.float32(next_iy) - cy - pnt_y) * inv_dir_y if abs(dir_y) > _EPSILON else _INF

    # === BACKPROJECTION TRAVERSAL LOOP ===
    # Adjoint of the cell-constant Siddon forward projection.
    while t < t_max:
        if 0 <= ix < Nx and 0 <= iy < Ny:
            t_next = min(tx, ty, t_max)
            seg_len = t_next - t
            if seg_len > _EPSILON:
                cuda.atomic.add(d_image, (iy, ix), val * seg_len)

        # Advance to next voxel (identical logic to forward projection)
        if tx <= ty:
            t = tx
            ix += step_x
            tx += dt_x
        else:
            t = ty
            iy += step_y
            ty += dt_y

# ------------------------------------------------------------------
# 2-D FAN BEAM KERNELS
# ------------------------------------------------------------------

@_FASTMATH_DECORATOR
def _fan_2d_forward_kernel(
    d_image, Nx, Ny,
    d_sino, n_ang, n_det,
    det_spacing, d_cos, d_sin,
    sdd, sid, cx, cy, voxel_spacing,
    det_offset, center_offset_x, center_offset_y
):
    """Compute the 2D fan beam forward projection.

    This CUDA kernel implements cell-constant Siddon ray tracing for
    2D fan beam forward projection.

    Parameters
    ----------
    d_image : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Input 2D image array on CUDA.
    Nx : int
        Number of voxels along the x-axis.
    Ny : int
        Number of voxels along the y-axis.
    d_sino : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Output fan beam sinogram array on CUDA.
    n_ang : int
        Number of projection angles.
    n_det : int
        Number of detector elements.
    det_spacing : float
        Physical spacing between detector elements.
    d_cos : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Precomputed cosine values of projection angles.
    d_sin : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Precomputed sine values of projection angles.
    sdd : float
        Source-to-Detector Distance (SDD), total distance from source to detector.
    sid : float
        Source-to-Isocenter Distance (SID), distance from source to isocenter.
    cx : float
        Half of image width in voxels.
    cy : float
        Half of image height in voxels.
    voxel_spacing : float
        Physical size of one voxel (in same units as det_spacing, sid, sdd).

    Notes
    -----
    Fan beam geometry diverges from parallel beam in that its rays originate
    from a single point source to a linear detector array. Rays connect the
    rotated source position around the isocenter to each detector pixel.
    """
    iang, idet = cuda.grid(2)
    if iang >= n_ang or idet >= n_det:
        return

    # === FAN BEAM GEOMETRY SETUP ===
    cos_a = d_cos[iang]  # Precomputed cosine of projection angle
    sin_a = d_sin[iang]  # Precomputed sine of projection angle
    # Normalize all physical distances to voxel units
    u = (np.float32(idet) - (np.float32(n_det) - _ONE) * _HALF) * det_spacing / voxel_spacing + det_offset
    sid_v = sid / voxel_spacing  # Source-to-isocenter distance in voxel units
    sdd_v = sdd / voxel_spacing  # Source-to-detector distance in voxel units

    # Calculate source and detector positions for current projection angle
    # Source position: rotated by angle around isocenter at distance sid (SID)
    src_x = -sid_v * sin_a + center_offset_x  # Source x-coordinate in voxel units
    src_y = sid_v * cos_a + center_offset_y  # Source y-coordinate in voxel units
    
    # Detector element position: IDD = SDD - SID (Isocenter-to-Detector Distance)
    idd = sdd_v - sid_v
    det_x = idd * sin_a + u * cos_a + center_offset_x  # Detector x-coordinate in voxel units
    det_y = -idd * cos_a + u * sin_a + center_offset_y  # Detector y-coordinate in voxel units

    # === RAY DIRECTION CALCULATION ===
    # Ray direction vector from source to detector element
    dir_x, dir_y = det_x - src_x, det_y - src_y
    length = math.sqrt(dir_x * dir_x + dir_y * dir_y)  # Ray length
    if length < _EPSILON:  # Degenerate ray case
        d_sino[iang, idet] = _ZERO; return
    
    # Normalize ray direction vector for parametric traversal
    inv_len = _ONE / length
    dir_x, dir_y = dir_x * inv_len, dir_y * inv_len

    # === RAY-VOLUME INTERSECTION CALCULATION ===
    # Compute intersection with volume boundaries using source position as ray origin
    t_min, t_max = _NEG_INF, _INF
    if abs(dir_x) > _EPSILON:
        tx1, tx2 = (-cx - src_x) / dir_x, (cx - src_x) / dir_x  # Volume boundary intersections
        t_min, t_max = max(t_min, min(tx1, tx2)), min(t_max, max(tx1, tx2))
    elif src_x < -cx or src_x > cx:  # Source outside volume bounds
        d_sino[iang, idet] = _ZERO; return

    if abs(dir_y) > _EPSILON:
        ty1, ty2 = (-cy - src_y) / dir_y, (cy - src_y) / dir_y
        t_min, t_max = max(t_min, min(ty1, ty2)), min(t_max, max(ty1, ty2))
    elif src_y < -cy or src_y > cy:
        d_sino[iang, idet] = _ZERO; return

    if t_min >= t_max:  # No valid intersection
        d_sino[iang, idet] = _ZERO; return

    # === SIDDON METHOD TRAVERSAL (same algorithm as parallel beam) ===
    accum = _ZERO  # Accumulated projection value
    t = t_min    # Current ray parameter
    
    # Convert ray entry point to voxel indices (using source as ray origin)
    ix = int(math.floor(src_x + t * dir_x + cx))
    iy = int(math.floor(src_y + t * dir_y + cy))

    # Traversal parameters (identical to parallel beam implementation)
    step_x, step_y = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1)
    inv_dir_x = (_ONE / dir_x) if abs(dir_x) > _EPSILON else _ZERO
    inv_dir_y = (_ONE / dir_y) if abs(dir_y) > _EPSILON else _ZERO
    dt_x = abs(inv_dir_x) if abs(dir_x) > _EPSILON else _INF
    dt_y = abs(inv_dir_y) if abs(dir_y) > _EPSILON else _INF
    next_ix = ix + (1 if step_x > 0 else 0)
    next_iy = iy + (1 if step_y > 0 else 0)
    tx = (np.float32(next_ix) - cx - src_x) * inv_dir_x if abs(dir_x) > _EPSILON else _INF
    ty = (np.float32(next_iy) - cy - src_y) * inv_dir_y if abs(dir_y) > _EPSILON else _INF

    # Main traversal loop with cell-constant Siddon integration.
    while t < t_max:
        if 0 <= ix < Nx and 0 <= iy < Ny:
            t_next = min(tx, ty, t_max)
            seg_len = t_next - t
            if seg_len > _EPSILON:
                accum += d_image[iy, ix] * seg_len
        
        # Voxel boundary crossing logic (identical to parallel beam)
        if tx <= ty:
            t = tx
            ix += step_x
            tx += dt_x
        else:
            t = ty
            iy += step_y
            ty += dt_y
    
    d_sino[iang, idet] = accum

@_FASTMATH_DECORATOR
def _fan_2d_backward_kernel(
    d_sino, n_ang, n_det,
    d_image, Nx, Ny,
    det_spacing, d_cos, d_sin,
    sdd, sid, cx, cy, voxel_spacing,
    det_offset, center_offset_x, center_offset_y,
):
    """Pure adjoint ``P^T`` of the fan-beam forward projector.

    Cell-constant Siddon ray-driven scatter with no
    distance weighting. Used by ``FanProjectorFunction.backward`` and
    ``FanBackprojectorFunction.forward``. The analytical FBP path lives
    in ``_fan_2d_fbp_backproject_kernel`` / ``fan_weighted_backproject``
    and does *not* go through this kernel.
    """
    iang, idet = cuda.grid(2)
    if iang >= n_ang or idet >= n_det:
        return

    # === BACKPROJECTION VALUE AND GEOMETRY SETUP ===
    val   = d_sino[iang, idet]  # Sinogram value to backproject along this ray
    cos_a = d_cos[iang]         # Precomputed cosine of projection angle
    sin_a = d_sin[iang]         # Precomputed sine of projection angle
    # Normalize all physical distances to voxel units
    u = (np.float32(idet) - (np.float32(n_det) - _ONE) * _HALF) * det_spacing / voxel_spacing + det_offset
    sid_v = sid / voxel_spacing  # Source-to-isocenter distance in voxel units
    sdd_v = sdd / voxel_spacing  # Source-to-detector distance in voxel units

    # Calculate source and detector positions for current projection angle
    # Source position: rotated by angle around isocenter at distance sid (SID)
    src_x = -sid_v * sin_a + center_offset_x  # Source x-coordinate in voxel units
    src_y = sid_v * cos_a + center_offset_y  # Source y-coordinate in voxel units
    
    # Detector element position: IDD = SDD - SID (Isocenter-to-Detector Distance)
    idd = sdd_v - sid_v
    det_x = idd * sin_a + u * cos_a + center_offset_x  # Detector x-coordinate in voxel units
    det_y = -idd * cos_a + u * sin_a + center_offset_y  # Detector y-coordinate in voxel units

    # === RAY DIRECTION CALCULATION ===
    # Ray direction vector from source to detector element
    dir_x, dir_y = det_x - src_x, det_y - src_y
    length = math.sqrt(dir_x * dir_x + dir_y * dir_y)  # Ray length
    if length < _EPSILON: return  # Skip degenerate rays
    inv_len = _ONE / length        # Normalization factor for ray direction
    dir_x, dir_y = dir_x * inv_len, dir_y * inv_len  # Normalized ray direction vector

    # === RAY-VOLUME INTERSECTION CALCULATION ===
    # Compute intersection with volume boundaries using source position as ray origin
    t_min, t_max = _NEG_INF, _INF
    if abs(dir_x) > _EPSILON:
        tx1, tx2 = (-cx - src_x) / dir_x, (cx - src_x) / dir_x
        t_min, t_max = max(t_min, min(tx1, tx2)), min(t_max, max(tx1, tx2))
    elif src_x < -cx or src_x > cx: return

    if abs(dir_y) > _EPSILON:
        ty1, ty2 = (-cy - src_y) / dir_y, (cy - src_y) / dir_y
        t_min, t_max = max(t_min, min(ty1, ty2)), min(t_max, max(ty1, ty2))
    elif src_y < -cy or src_y > cy: return

    if t_min >= t_max: return

    # === SIDDON METHOD TRAVERSAL INITIALIZATION ===
    t = t_min
    ix = int(math.floor(src_x + t * dir_x + cx))
    iy = int(math.floor(src_y + t * dir_y + cy))

    step_x, step_y = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1)
    inv_dir_x = (_ONE / dir_x) if abs(dir_x) > _EPSILON else _ZERO
    inv_dir_y = (_ONE / dir_y) if abs(dir_y) > _EPSILON else _ZERO
    dt_x = abs(inv_dir_x) if abs(dir_x) > _EPSILON else _INF
    dt_y = abs(inv_dir_y) if abs(dir_y) > _EPSILON else _INF
    next_ix = ix + (1 if step_x > 0 else 0)
    next_iy = iy + (1 if step_y > 0 else 0)
    tx = (np.float32(next_ix) - cx - src_x) * inv_dir_x if abs(dir_x) > _EPSILON else _INF
    ty = (np.float32(next_iy) - cy - src_y) * inv_dir_y if abs(dir_y) > _EPSILON else _INF

    # === FAN BEAM BACKPROJECTION TRAVERSAL LOOP ===
    # Adjoint of the cell-constant Siddon forward projection.
    while t < t_max:
        if 0 <= ix < Nx and 0 <= iy < Ny:
            t_next = min(tx, ty, t_max)
            seg_len = t_next - t
            if seg_len > _EPSILON:
                cuda.atomic.add(d_image, (iy, ix), val * seg_len)

        # === VOXEL BOUNDARY CROSSING LOGIC ===
        # Advance to next voxel based on which boundary is crossed first
        if tx <= ty:
            t = tx
            ix += step_x
            tx += dt_x
        else:
            t = ty
            iy += step_y
            ty += dt_y


@_FDK_ACCURACY_DECORATOR
def _fan_2d_sf_forward_kernel(
    d_image, Nx, Ny,
    d_sino, n_ang, n_det,
    det_spacing, d_cos, d_sin,
    sdd, sid, cx, cy, voxel_spacing,
    det_offset, center_offset_x, center_offset_y
):
    """Separable-footprint (SF-TR) forward projector for 2D fan beam.

    Voxel-driven: one thread per (view, voxel). Projects the four corners of
    the unit-square voxel onto the flat detector to obtain a trapezoidal
    footprint in the detector coordinate u, then closed-form-integrates that
    trapezoid over each overlapping detector bin and atomically accumulates
    ``value * chord * bin_fraction`` into the sinogram. Mass is conserved:
    summed across all bins, a single voxel contributes exactly ``v * chord``,
    the same total as the thin-ray Siddon kernel, but spread across the true
    finite-width footprint instead of concentrated at one bin.
    """
    iang, iy, ix = cuda.grid(3)
    if iang >= n_ang or iy >= Ny or ix >= Nx:
        return

    val = d_image[iy, ix]
    if val == _ZERO:
        return

    cos_a = d_cos[iang]
    sin_a = d_sin[iang]
    sid_v = sid / voxel_spacing
    sdd_v = sdd / voxel_spacing
    det_spacing_v = det_spacing / voxel_spacing

    # Voxel center at grid-point world position. Matches the Siddon / FDK
    # convention where d_vol[ix, iy] is a sample AT world (ix - cx, iy - cy)
    # and bilinear interpolation linearly connects neighbouring samples. The
    # SF "box" around each sample has unit side length, extending +-0.5.
    x_c = (np.float32(ix) - cx) - center_offset_x
    y_c = (np.float32(iy) - cy) - center_offset_y

    # Project the four corners to the flat detector.
    # u(x, y) = sdd_v * (x*cos + y*sin) / (sid_v + x*sin - y*cos)
    x_m = x_c - _HALF
    x_p = x_c + _HALF
    y_m = y_c - _HALF
    y_p = y_c + _HALF

    d1 = sid_v + x_m * sin_a - y_m * cos_a
    d2 = sid_v + x_p * sin_a - y_m * cos_a
    d3 = sid_v + x_p * sin_a - y_p * cos_a
    d4 = sid_v + x_m * sin_a - y_p * cos_a
    if d1 <= _EPSILON or d2 <= _EPSILON or d3 <= _EPSILON or d4 <= _EPSILON:
        return  # voxel straddles the source plane — SF invalid

    u1 = sdd_v * (x_m * cos_a + y_m * sin_a) / d1
    u2 = sdd_v * (x_p * cos_a + y_m * sin_a) / d2
    u3 = sdd_v * (x_p * cos_a + y_p * sin_a) / d3
    u4 = sdd_v * (x_m * cos_a + y_p * sin_a) / d4

    # 4-element sorting network → (u_min, u_lo, u_hi, u_max)
    a = u1
    b = u2
    c = u3
    d = u4
    if a > b:
        a, b = b, a
    if c > d:
        c, d = d, c
    if a > c:
        a, c = c, a
    if b > d:
        b, d = d, b
    if b > c:
        b, c = c, b
    u_min = a
    u_lo = b
    u_hi = c
    u_max = d

    span = u_max - u_min
    if span < _EPSILON:
        return

    # Central ray through voxel center → chord length through unit square.
    # For ray direction (rx, ry) with rx^2 + ry^2 = 1, chord = 1/max(|rx|,|ry|).
    rx = x_c - (-sid_v * sin_a)
    ry = y_c - (sid_v * cos_a)
    r_len = math.sqrt(rx * rx + ry * ry)
    if r_len < _EPSILON:
        return
    rx /= r_len
    ry /= r_len
    m = abs(rx)
    ar = abs(ry)
    if ar > m:
        m = ar
    if m < _EPSILON:
        return
    chord = _ONE / m

    # Footprint is a trapezoid in detector u with peak = chord (= line integral
    # along the central ray). Each sinogram bin stores the cell-averaged line
    # integral (thin-ray convention, matching Siddon): contribution from this
    # voxel to bin k is (val * chord / det_spacing_v) * bin_integral_of_unit_trapezoid.
    plateau = u_hi - u_lo
    weight = val * chord / det_spacing_v

    # Detector bin range overlapping [u_min, u_max].
    # idet = (u - det_offset) / det_spacing_v + 0.5*(n_det - 1)
    half = (np.float32(n_det) - _ONE) * _HALF
    k_lo_f = (u_min - det_offset) / det_spacing_v + half - _HALF
    k_hi_f = (u_max - det_offset) / det_spacing_v + half + _HALF
    k_lo = int(math.floor(k_lo_f))
    k_hi = int(math.ceil(k_hi_f))
    if k_lo < 0:
        k_lo = 0
    if k_hi > n_det - 1:
        k_hi = n_det - 1
    if k_hi < k_lo:
        return

    rise_w = u_lo - u_min
    fall_w = u_max - u_hi

    for k in range(k_lo, k_hi + 1):
        u_k = (np.float32(k) - half) * det_spacing_v + det_offset
        u_L = u_k - _HALF * det_spacing_v
        u_R = u_k + _HALF * det_spacing_v

        aL = u_L if u_L > u_min else u_min
        aR = u_R if u_R < u_max else u_max
        if aL >= aR:
            continue

        raw = _ZERO  # unnormalized trapezoid integral over [aL, aR] with peak = 1

        # Rising segment [u_min, u_lo], height (u - u_min)/rise_w.
        if rise_w > _EPSILON:
            r_lo = aL if aL > u_min else u_min
            r_hi = aR if aR < u_lo else u_lo
            if r_hi > r_lo:
                raw += _HALF * ((r_hi - u_min) * (r_hi - u_min) -
                              (r_lo - u_min) * (r_lo - u_min)) / rise_w

        # Plateau [u_lo, u_hi], height = 1.
        if plateau > _EPSILON:
            p_lo = aL if aL > u_lo else u_lo
            p_hi = aR if aR < u_hi else u_hi
            if p_hi > p_lo:
                raw += p_hi - p_lo

        # Falling segment [u_hi, u_max], height (u_max - u)/fall_w.
        if fall_w > _EPSILON:
            f_lo = aL if aL > u_hi else u_hi
            f_hi = aR if aR < u_max else u_max
            if f_hi > f_lo:
                raw += _HALF * ((u_max - f_lo) * (u_max - f_lo) -
                              (u_max - f_hi) * (u_max - f_hi)) / fall_w

        if raw > _ZERO:
            cuda.atomic.add(d_sino, (iang, k), weight * raw)


@_FDK_ACCURACY_DECORATOR
def _fan_2d_sf_backward_kernel(
    d_grad_sino, n_ang, n_det,
    d_grad_img, Nx, Ny,
    det_spacing, d_cos, d_sin,
    sdd, sid, cx, cy, voxel_spacing,
    det_offset, center_offset_x, center_offset_y
):
    """Pure adjoint (transpose) of _fan_2d_sf_forward_kernel.

    Voxel-driven gather. One thread per output pixel, loops over views and
    over the detector bins inside each view's footprint, rebuilding the same
    trapezoidal SF coefficients as the forward kernel and accumulating
    ``weight * raw * grad_sino[iang, k]`` into ``grad_img[iy, ix]``. No atomic
    adds needed because each thread owns a unique output pixel. The SF
    coefficients (``chord``, ``raw``, ``1/det_spacing_v``) are byte-for-byte
    identical to the forward kernel, so the inner-product identity
    ``<A x, y> = <x, A^T y>`` holds exactly up to float32 accumulation order.
    """
    ix, iy = cuda.grid(2)
    if iy >= Ny or ix >= Nx:
        return

    sid_v = sid / voxel_spacing
    sdd_v = sdd / voxel_spacing
    det_spacing_v = det_spacing / voxel_spacing
    half = (np.float32(n_det) - _ONE) * _HALF

    # Grid-point convention (see _fan_2d_sf_forward_kernel for details).
    x_c = (np.float32(ix) - cx) - center_offset_x
    y_c = (np.float32(iy) - cy) - center_offset_y
    x_m = x_c - _HALF
    x_p = x_c + _HALF
    y_m = y_c - _HALF
    y_p = y_c + _HALF

    grad_val = _ZERO

    for iang in range(n_ang):
        cos_a = d_cos[iang]
        sin_a = d_sin[iang]

        d1 = sid_v + x_m * sin_a - y_m * cos_a
        d2 = sid_v + x_p * sin_a - y_m * cos_a
        d3 = sid_v + x_p * sin_a - y_p * cos_a
        d4 = sid_v + x_m * sin_a - y_p * cos_a
        if d1 <= _EPSILON or d2 <= _EPSILON or d3 <= _EPSILON or d4 <= _EPSILON:
            continue

        u1 = sdd_v * (x_m * cos_a + y_m * sin_a) / d1
        u2 = sdd_v * (x_p * cos_a + y_m * sin_a) / d2
        u3 = sdd_v * (x_p * cos_a + y_p * sin_a) / d3
        u4 = sdd_v * (x_m * cos_a + y_p * sin_a) / d4

        a = u1
        b = u2
        c = u3
        d = u4
        if a > b:
            a, b = b, a
        if c > d:
            c, d = d, c
        if a > c:
            a, c = c, a
        if b > d:
            b, d = d, b
        if b > c:
            b, c = c, b
        u_min = a
        u_lo = b
        u_hi = c
        u_max = d

        span = u_max - u_min
        if span < _EPSILON:
            continue

        rx = x_c - (-sid_v * sin_a)
        ry = y_c - (sid_v * cos_a)
        r_len = math.sqrt(rx * rx + ry * ry)
        if r_len < _EPSILON:
            continue
        rx /= r_len
        ry /= r_len
        m = abs(rx)
        ar = abs(ry)
        if ar > m:
            m = ar
        if m < _EPSILON:
            continue
        chord = _ONE / m

        plateau = u_hi - u_lo
        weight = chord / det_spacing_v

        k_lo_f = (u_min - det_offset) / det_spacing_v + half - _HALF
        k_hi_f = (u_max - det_offset) / det_spacing_v + half + _HALF
        k_lo = int(math.floor(k_lo_f))
        k_hi = int(math.ceil(k_hi_f))
        if k_lo < 0:
            k_lo = 0
        if k_hi > n_det - 1:
            k_hi = n_det - 1
        if k_hi < k_lo:
            continue

        rise_w = u_lo - u_min
        fall_w = u_max - u_hi

        for k in range(k_lo, k_hi + 1):
            u_k = (np.float32(k) - half) * det_spacing_v + det_offset
            u_L = u_k - _HALF * det_spacing_v
            u_R = u_k + _HALF * det_spacing_v

            aL = u_L if u_L > u_min else u_min
            aR = u_R if u_R < u_max else u_max
            if aL >= aR:
                continue

            raw = _ZERO
            if rise_w > _EPSILON:
                r_lo_ = aL if aL > u_min else u_min
                r_hi_ = aR if aR < u_lo else u_lo
                if r_hi_ > r_lo_:
                    raw += _HALF * ((r_hi_ - u_min) * (r_hi_ - u_min) -
                                  (r_lo_ - u_min) * (r_lo_ - u_min)) / rise_w

            if plateau > _EPSILON:
                p_lo_ = aL if aL > u_lo else u_lo
                p_hi_ = aR if aR < u_hi else u_hi
                if p_hi_ > p_lo_:
                    raw += p_hi_ - p_lo_

            if fall_w > _EPSILON:
                f_lo_ = aL if aL > u_hi else u_hi
                f_hi_ = aR if aR < u_max else u_max
                if f_hi_ > f_lo_:
                    raw += _HALF * ((u_max - f_lo_) * (u_max - f_lo_) -
                                  (u_max - f_hi_) * (u_max - f_hi_)) / fall_w

            if raw > _ZERO:
                grad_val += weight * raw * d_grad_sino[iang, k]

    d_grad_img[iy, ix] = grad_val


# ------------------------------------------------------------------
# 3-D CONE BEAM KERNELS
# ------------------------------------------------------------------

@_FASTMATH_DECORATOR
def _cone_3d_forward_kernel(
    d_vol, Nx, Ny, Nz,
    d_sino, n_views, n_u, n_v,
    du, dv, d_cos, d_sin,
    sdd, sid, cx, cy, cz, voxel_spacing,
    det_offset_u, det_offset_v,
    center_offset_x, center_offset_y, center_offset_z
):
    """Compute the 3D cone-beam forward projection.

    This CUDA kernel implements cell-constant Siddon ray tracing for
    3D cone-beam forward projection.

    Parameters
    ----------
    d_vol : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Input 3D volume array on CUDA.
    Nx : int
        Number of voxels along the x-axis.
    Ny : int
        Number of voxels along the y-axis.
    Nz : int
        Number of voxels along the z-axis.
    d_sino : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Output cone-beam sinogram array on CUDA.
    n_views : int
        Number of projection views.
    n_u : int
        Number of detector elements along the u-axis.
    n_v : int
        Number of detector elements along the v-axis.
    du : float
        Physical spacing between detector elements along the u-axis.
    dv : float
        Physical spacing between detector elements along the v-axis.
    d_cos : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Precomputed cosine values of projection angles.
    d_sin : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Precomputed sine values of projection angles.
    sdd : float
        Source-to-Detector Distance (SDD), total distance from source to detector.
    sid : float
        Source-to-Isocenter Distance (SID), distance from source to isocenter.
    cx : float
        Half of volume width along x-axis (in voxels).
    cy : float
        Half of volume height along y-axis (in voxels).
    cz : float
        Half of volume depth along z-axis (in voxels).
    voxel_spacing : float
        Physical size of one voxel (in same units as du, dv, sid, sdd).

    Notes
    -----
    Cone-beam geometry extends the fan-beam configuration to 3D by employing
    a 2D detector array and integrating each ray through piecewise-constant
    voxel cells.
    """
    # Put detector-v on threadIdx.x; d_sino[view, u, v] is row-major with v
    # as the stride-1 axis, and adjacent v rays also tend to walk nearby z cells.
    iv, iu, iview = cuda.grid(3)
    if iview >= n_views or iu >= n_u or iv >= n_v:
        return

    # === 3D CONE BEAM GEOMETRY SETUP ===
    cos_a, sin_a = d_cos[iview], d_sin[iview]  # Projection angle trigonometry
    # Normalize all physical distances to voxel units
    u = (np.float32(iu) - (np.float32(n_u) - _ONE) * _HALF) * du / voxel_spacing + det_offset_u
    v = (np.float32(iv) - (np.float32(n_v) - _ONE) * _HALF) * dv / voxel_spacing + det_offset_v
    sid_v = sid / voxel_spacing  # Source-to-isocenter distance in voxel units
    sdd_v = sdd / voxel_spacing  # Source-to-detector distance in voxel units

    # Calculate 3D source and detector positions
    # Source rotates in xy-plane around isocenter, z-coordinate is zero
    src_x = -sid_v * sin_a + center_offset_x
    src_y = sid_v * cos_a + center_offset_y
    src_z = center_offset_z
    
    # Detector element position: IDD = SDD - SID (Isocenter-to-Detector Distance)
    # u-coordinate is in-plane offset, v-coordinate is vertical (z-direction)
    idd = sdd_v - sid_v
    det_x = idd * sin_a + u * cos_a + center_offset_x  # In-plane x-coordinate in voxel units
    det_y = -idd * cos_a + u * sin_a + center_offset_y  # In-plane y-coordinate in voxel units
    det_z = v + center_offset_z  # Vertical z-coordinate in voxel units

    # === 3D RAY DIRECTION CALCULATION ===
    # Ray direction vector from source to detector element in 3D space
    dir_x, dir_y, dir_z = det_x - src_x, det_y - src_y, det_z - src_z
    length = math.sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z)  # 3D ray length
    if length < _EPSILON:  # Degenerate ray case
        d_sino[iview, iu, iv] = _ZERO; return
    
    # Normalize 3D ray direction vector for parametric traversal
    inv_len = _ONE / length
    dir_x, dir_y, dir_z = dir_x*inv_len, dir_y*inv_len, dir_z*inv_len

    # === 3D RAY-VOLUME INTERSECTION CALCULATION ===
    # Compute intersection with 3D volume boundaries using source position as ray origin
    t_min, t_max = _NEG_INF, _INF
    
    # X-direction boundary intersections
    if abs(dir_x) > _EPSILON:
        tx1, tx2 = (-cx - src_x) / dir_x, (cx - src_x) / dir_x
        t_min, t_max = max(t_min, min(tx1, tx2)), min(t_max, max(tx1, tx2))
    elif src_x < -cx or src_x > cx:  # Source outside x-bounds
        d_sino[iview, iu, iv] = _ZERO; return
    
    # Y-direction boundary intersections
    if abs(dir_y) > _EPSILON:
        ty1, ty2 = (-cy - src_y) / dir_y, (cy - src_y) / dir_y
        t_min, t_max = max(t_min, min(ty1, ty2)), min(t_max, max(ty1, ty2))
    elif src_y < -cy or src_y > cy:  # Source outside y-bounds
        d_sino[iview, iu, iv] = _ZERO; return
    
    # Z-direction boundary intersections (extends 2D algorithm to 3D)
    if abs(dir_z) > _EPSILON:
        tz1, tz2 = (-cz - src_z) / dir_z, (cz - src_z) / dir_z
        t_min, t_max = max(t_min, min(tz1, tz2)), min(t_max, max(tz1, tz2))
    elif src_z < -cz or src_z > cz:  # Source outside z-bounds
        d_sino[iview, iu, iv] = _ZERO; return

    if t_min >= t_max:  # No valid 3D intersection
        d_sino[iview, iu, iv] = _ZERO; return

    # === 3D SIDDON METHOD TRAVERSAL INITIALIZATION ===
    accum = _ZERO  # Accumulated projection value
    t = t_min    # Current ray parameter
    
    # Convert 3D ray entry point to voxel indices
    ix = int(math.floor(src_x + t * dir_x + cx))  # Current voxel x-index
    iy = int(math.floor(src_y + t * dir_y + cy))  # Current voxel y-index
    iz = int(math.floor(src_z + t * dir_z + cz))  # Current voxel z-index

    # 3D traversal parameters (extends 2D algorithm)
    step_x, step_y, step_z = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1), (1 if dir_z >= 0 else -1)
    inv_dir_x = (_ONE / dir_x) if abs(dir_x) > _EPSILON else _ZERO
    inv_dir_y = (_ONE / dir_y) if abs(dir_y) > _EPSILON else _ZERO
    inv_dir_z = (_ONE / dir_z) if abs(dir_z) > _EPSILON else _ZERO
    dt_x = abs(inv_dir_x) if abs(dir_x) > _EPSILON else _INF  # Parameter increment per x-voxel
    dt_y = abs(inv_dir_y) if abs(dir_y) > _EPSILON else _INF  # Parameter increment per y-voxel
    dt_z = abs(inv_dir_z) if abs(dir_z) > _EPSILON else _INF  # Parameter increment per z-voxel

    # Calculate parameter values for next 3D voxel boundary crossings
    next_ix = ix + (1 if step_x > 0 else 0)
    next_iy = iy + (1 if step_y > 0 else 0)
    next_iz = iz + (1 if step_z > 0 else 0)
    tx = (np.float32(next_ix) - cx - src_x) * inv_dir_x if abs(dir_x) > _EPSILON else _INF
    ty = (np.float32(next_iy) - cy - src_y) * inv_dir_y if abs(dir_y) > _EPSILON else _INF
    tz = (np.float32(next_iz) - cz - src_z) * inv_dir_z if abs(dir_z) > _EPSILON else _INF

    # === 3D TRAVERSAL LOOP WITH CELL-CONSTANT SIDDON INTEGRATION ===
    while t < t_max:
        if 0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz:
            # Determine next 3D voxel boundary crossing (minimum of x, y, z boundaries or ray exit)
            t_next = min(tx, ty, tz, t_max)
            seg_len = t_next - t
            if seg_len > _EPSILON:
                accum += d_vol[ix, iy, iz] * seg_len

        # === 3D VOXEL BOUNDARY CROSSING LOGIC ===
        # Advance to next voxel based on which boundary is crossed first in 3D
        if tx <= ty and tx <= tz:      # X-boundary crossed first
            t = tx
            ix += step_x
            tx += dt_x
        elif ty <= tx and ty <= tz:    # Y-boundary crossed first
            t = ty
            iy += step_y
            ty += dt_y
        else:                          # Z-boundary crossed first
            t = tz
            iz += step_z
            tz += dt_z
    
    d_sino[iview, iu, iv] = accum

@_FASTMATH_DECORATOR
def _cone_3d_backward_kernel(
    d_sino, n_views, n_u, n_v,
    d_vol, Nx, Ny, Nz,
    du, dv, d_cos, d_sin,
    sdd, sid, cx, cy, cz, voxel_spacing,
    det_offset_u, det_offset_v,
    center_offset_x, center_offset_y, center_offset_z,
):
    """Pure adjoint ``P^T`` of the cone-beam forward projector.

    Cell-constant Siddon ray-driven scatter with no
    distance weighting. Used by ``ConeProjectorFunction.backward`` and
    ``ConeBackprojectorFunction.forward``. The analytical FDK path lives
    in ``_cone_3d_fdk_backproject_kernel`` / ``cone_weighted_backproject``
    and does *not* go through this kernel.
    """
    # Match the forward kernel's launch order: detector-v is warp-adjacent.
    iv, iu, iview = cuda.grid(3)
    if iview >= n_views or iu >= n_u or iv >= n_v:
        return

    # === 3D BACKPROJECTION VALUE AND GEOMETRY SETUP ===
    g = d_sino[iview, iu, iv]  # Sinogram value to backproject along this ray
    cos_a, sin_a = d_cos[iview], d_sin[iview]  # Projection angle trigonometry
    # Normalize all physical distances to voxel units
    u = (np.float32(iu) - (np.float32(n_u) - _ONE) * _HALF) * du / voxel_spacing + det_offset_u
    v = (np.float32(iv) - (np.float32(n_v) - _ONE) * _HALF) * dv / voxel_spacing + det_offset_v
    sid_v = sid / voxel_spacing  # Source-to-isocenter distance in voxel units
    sdd_v = sdd / voxel_spacing  # Source-to-detector distance in voxel units

    # Calculate 3D source and detector positions
    # Source rotates in xy-plane around isocenter, z-coordinate is zero
    src_x = -sid_v * sin_a + center_offset_x
    src_y = sid_v * cos_a + center_offset_y
    src_z = center_offset_z
    
    # Detector element position: IDD = SDD - SID (Isocenter-to-Detector Distance)
    # u-coordinate is in-plane offset, v-coordinate is vertical (z-direction)
    idd = sdd_v - sid_v
    det_x = idd * sin_a + u * cos_a + center_offset_x  # In-plane x-coordinate in voxel units
    det_y = -idd * cos_a + u * sin_a + center_offset_y  # In-plane y-coordinate in voxel units
    det_z = v + center_offset_z  # Vertical z-coordinate in voxel units

    # === 3D RAY DIRECTION CALCULATION ===
    # Ray direction vector from source to detector element in 3D space
    dir_x, dir_y, dir_z = det_x - src_x, det_y - src_y, det_z - src_z
    length = math.sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z)  # 3D ray length
    if length < _EPSILON: return  # Skip degenerate rays
    inv_len = _ONE / length        # Normalization factor for ray direction
    dir_x, dir_y, dir_z = dir_x*inv_len, dir_y*inv_len, dir_z*inv_len  # Normalized 3D ray direction vector

    # === 3D RAY-VOLUME INTERSECTION CALCULATION ===
    # Compute intersection with 3D volume boundaries using source position as ray origin
    t_min, t_max = _NEG_INF, _INF
    
    # X-direction boundary intersections
    if abs(dir_x) > _EPSILON:
        tx1, tx2 = (-cx - src_x) / dir_x, (cx - src_x) / dir_x
        t_min, t_max = max(t_min, min(tx1, tx2)), min(t_max, max(tx1, tx2))
    elif src_x < -cx or src_x > cx: return
    
    # Y-direction boundary intersections
    if abs(dir_y) > _EPSILON:
        ty1, ty2 = (-cy - src_y) / dir_y, (cy - src_y) / dir_y
        t_min, t_max = max(t_min, min(ty1, ty2)), min(t_max, max(ty1, ty2))
    elif src_y < -cy or src_y > cy: return
    
    # Z-direction boundary intersections (extends 2D algorithm to 3D)
    if abs(dir_z) > _EPSILON:
        tz1, tz2 = (-cz - src_z) / dir_z, (cz - src_z) / dir_z
        t_min, t_max = max(t_min, min(tz1, tz2)), min(t_max, max(tz1, tz2))
    elif src_z < -cz or src_z > cz: return

    if t_min >= t_max: return

    # === 3D SIDDON METHOD TRAVERSAL INITIALIZATION ===
    t = t_min
    ix = int(math.floor(src_x + t * dir_x + cx))  # Current voxel x-index
    iy = int(math.floor(src_y + t * dir_y + cy))  # Current voxel y-index
    iz = int(math.floor(src_z + t * dir_z + cz))  # Current voxel z-index

    # 3D traversal parameters (extends 2D algorithm)
    step_x, step_y, step_z = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1), (1 if dir_z >= 0 else -1)
    inv_dir_x = (_ONE / dir_x) if abs(dir_x) > _EPSILON else _ZERO
    inv_dir_y = (_ONE / dir_y) if abs(dir_y) > _EPSILON else _ZERO
    inv_dir_z = (_ONE / dir_z) if abs(dir_z) > _EPSILON else _ZERO
    dt_x = abs(inv_dir_x) if abs(dir_x) > _EPSILON else _INF  # Parameter increment per x-voxel
    dt_y = abs(inv_dir_y) if abs(dir_y) > _EPSILON else _INF  # Parameter increment per y-voxel
    dt_z = abs(inv_dir_z) if abs(dir_z) > _EPSILON else _INF  # Parameter increment per z-voxel

    # Calculate parameter values for next 3D voxel boundary crossings
    next_ix = ix + (1 if step_x > 0 else 0)
    next_iy = iy + (1 if step_y > 0 else 0)
    next_iz = iz + (1 if step_z > 0 else 0)
    tx = (np.float32(next_ix) - cx - src_x) * inv_dir_x if abs(dir_x) > _EPSILON else _INF
    ty = (np.float32(next_iy) - cy - src_y) * inv_dir_y if abs(dir_y) > _EPSILON else _INF
    tz = (np.float32(next_iz) - cz - src_z) * inv_dir_z if abs(dir_z) > _EPSILON else _INF

    # === 3D CONE BEAM BACKPROJECTION TRAVERSAL LOOP ===
    # Adjoint of the cell-constant Siddon forward projection.
    while t < t_max:
        if 0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz:
            # Determine next 3D voxel boundary crossing (minimum of x, y, z boundaries or ray exit)
            t_next = min(tx, ty, tz, t_max)
            seg_len = t_next - t
            if seg_len > _EPSILON:
                cuda.atomic.add(d_vol, (ix, iy, iz), g * seg_len)

        # === 3D VOXEL BOUNDARY CROSSING LOGIC ===
        # Advance to next voxel based on which boundary is crossed first in 3D
        if tx <= ty and tx <= tz:      # X-boundary crossed first
            t = tx
            ix += step_x
            tx += dt_x
        elif ty <= tx and ty <= tz:    # Y-boundary crossed first
            t = ty
            iy += step_y
            ty += dt_y
        else:                          # Z-boundary crossed first
            t = tz
            iz += step_z
            tz += dt_z


# ------------------------------------------------------------------
# 3-D cone-beam separable-footprint kernels (SF-TR and SF-TT)
# ------------------------------------------------------------------
# SF-TR uses a trapezoidal footprint in the transaxial (u) direction and a
# rectangular footprint in the axial (v) direction, evaluated with the
# magnification at the voxel centre. SF-TT uses a trapezoidal footprint in
# BOTH directions, with the axial trapezoid taking its rise/fall from the
# difference between U_near and U_far across the voxel, which matters at
# large z offsets where the axial magnification varies measurably across a
# single voxel. Both variants are voxel-driven and compute a closed-form
# integral of the separable footprint over each detector bin. The adjoint
# kernels gather instead of scatter, rebuild the same coefficients, and
# preserve the exact <Ax, y> = <x, A^T y> identity up to float32 accumulation.


@_FDK_ACCURACY_DECORATOR
def _cone_3d_sf_tr_forward_kernel(
    d_vol, Nx, Ny, Nz,
    d_sino, n_views, n_u, n_v,
    du, dv, d_cos, d_sin,
    sdd, sid, cx, cy, cz, voxel_spacing,
    det_offset_u, det_offset_v,
    center_offset_x, center_offset_y, center_offset_z
):
    """Cone-beam SF-TR forward projector.

    Voxel-driven scatter: one thread per voxel (iterating views inside),
    trapezoidal transaxial footprint built from the 4 ``(x, y)`` voxel
    corners and a rectangular axial footprint using ``U`` at the voxel
    centre. Contributions are atomic-added into the sinogram.
    """
    iz, iy, ix = cuda.grid(3)
    if ix >= Nx or iy >= Ny or iz >= Nz:
        return

    val = d_vol[ix, iy, iz]
    if val == _ZERO:
        return

    sid_v = sid / voxel_spacing
    sdd_v = sdd / voxel_spacing
    du_v = du / voxel_spacing
    dv_v = dv / voxel_spacing
    u_half = (np.float32(n_u) - _ONE) * _HALF
    v_half = (np.float32(n_v) - _ONE) * _HALF

    # Grid-point convention: voxel[ix,iy,iz] sits at world (ix-cx, iy-cy, iz-cz).
    x_c = (np.float32(ix) - cx) - center_offset_x
    y_c = (np.float32(iy) - cy) - center_offset_y
    z_c = (np.float32(iz) - cz) - center_offset_z
    x_m = x_c - _HALF
    x_p = x_c + _HALF
    y_m = y_c - _HALF
    y_p = y_c + _HALF
    z_m = z_c - _HALF
    z_p = z_c + _HALF

    for iview in range(n_views):
        cos_a = d_cos[iview]
        sin_a = d_sin[iview]

        # Transaxial corner projections (z independent of u for a flat detector).
        d1 = sid_v + x_m * sin_a - y_m * cos_a
        d2 = sid_v + x_p * sin_a - y_m * cos_a
        d3 = sid_v + x_p * sin_a - y_p * cos_a
        d4 = sid_v + x_m * sin_a - y_p * cos_a
        if d1 <= _EPSILON or d2 <= _EPSILON or d3 <= _EPSILON or d4 <= _EPSILON:
            continue

        u1 = sdd_v * (x_m * cos_a + y_m * sin_a) / d1
        u2 = sdd_v * (x_p * cos_a + y_m * sin_a) / d2
        u3 = sdd_v * (x_p * cos_a + y_p * sin_a) / d3
        u4 = sdd_v * (x_m * cos_a + y_p * sin_a) / d4

        a = u1
        b = u2
        c = u3
        d = u4
        if a > b:
            a, b = b, a
        if c > d:
            c, d = d, c
        if a > c:
            a, c = c, a
        if b > d:
            b, d = d, b
        if b > c:
            b, c = c, b
        u_min = a
        u_lo = b
        u_hi = c
        u_max = d
        if (u_max - u_min) < _EPSILON:
            continue

        # Axial rectangle using U at voxel centre — SF-TR approximation.
        U_c = sid_v + x_c * sin_a - y_c * cos_a
        if U_c <= _EPSILON:
            continue
        mag_c = sdd_v / U_c
        v_bot = z_m * mag_c
        v_top = z_p * mag_c
        if v_bot > v_top:
            v_bot, v_top = v_top, v_bot
        v_span = v_top - v_bot
        if v_span < _EPSILON:
            continue

        # Central-ray chord length through the unit cube.
        rx = x_c - (-sid_v * sin_a)
        ry = y_c - (sid_v * cos_a)
        rz = z_c
        r_len = math.sqrt(rx * rx + ry * ry + rz * rz)
        if r_len < _EPSILON:
            continue
        rx /= r_len
        ry /= r_len
        rz /= r_len
        m = abs(rx)
        ar = abs(ry)
        if ar > m:
            m = ar
        az = abs(rz)
        if az > m:
            m = az
        if m < _EPSILON:
            continue
        chord = _ONE / m

        plateau = u_hi - u_lo
        rise_w = u_lo - u_min
        fall_w = u_max - u_hi
        weight = val * chord / (du_v * dv_v)

        k_u_lo = int(math.floor((u_min - det_offset_u) / du_v + u_half - _HALF))
        k_u_hi = int(math.ceil((u_max - det_offset_u) / du_v + u_half + _HALF))
        if k_u_lo < 0:
            k_u_lo = 0
        if k_u_hi > n_u - 1:
            k_u_hi = n_u - 1
        if k_u_hi < k_u_lo:
            continue

        k_v_lo = int(math.floor((v_bot - det_offset_v) / dv_v + v_half - _HALF))
        k_v_hi = int(math.ceil((v_top - det_offset_v) / dv_v + v_half + _HALF))
        if k_v_lo < 0:
            k_v_lo = 0
        if k_v_hi > n_v - 1:
            k_v_hi = n_v - 1
        if k_v_hi < k_v_lo:
            continue

        for ku in range(k_u_lo, k_u_hi + 1):
            u_k = (np.float32(ku) - u_half) * du_v + det_offset_u
            u_L = u_k - _HALF * du_v
            u_R = u_k + _HALF * du_v

            aL = u_L if u_L > u_min else u_min
            aR = u_R if u_R < u_max else u_max
            if aL >= aR:
                continue

            raw_u = _ZERO
            if rise_w > _EPSILON:
                r_lo_ = aL if aL > u_min else u_min
                r_hi_ = aR if aR < u_lo else u_lo
                if r_hi_ > r_lo_:
                    raw_u += _HALF * ((r_hi_ - u_min) * (r_hi_ - u_min) -
                                    (r_lo_ - u_min) * (r_lo_ - u_min)) / rise_w
            if plateau > _EPSILON:
                p_lo_ = aL if aL > u_lo else u_lo
                p_hi_ = aR if aR < u_hi else u_hi
                if p_hi_ > p_lo_:
                    raw_u += p_hi_ - p_lo_
            if fall_w > _EPSILON:
                f_lo_ = aL if aL > u_hi else u_hi
                f_hi_ = aR if aR < u_max else u_max
                if f_hi_ > f_lo_:
                    raw_u += _HALF * ((u_max - f_lo_) * (u_max - f_lo_) -
                                    (u_max - f_hi_) * (u_max - f_hi_)) / fall_w
            if raw_u <= _ZERO:
                continue

            for kv in range(k_v_lo, k_v_hi + 1):
                v_k = (np.float32(kv) - v_half) * dv_v + det_offset_v
                v_L = v_k - _HALF * dv_v
                v_R = v_k + _HALF * dv_v

                aLv = v_L if v_L > v_bot else v_bot
                aRv = v_R if v_R < v_top else v_top
                if aLv >= aRv:
                    continue
                raw_v = aRv - aLv  # rectangle, height = 1

                cuda.atomic.add(d_sino, (iview, ku, kv), weight * raw_u * raw_v)


@_FDK_ACCURACY_DECORATOR
def _cone_3d_sf_tr_backward_kernel(
    d_grad_sino, n_views, n_u, n_v,
    d_grad_vol, Nx, Ny, Nz,
    du, dv, d_cos, d_sin,
    sdd, sid, cx, cy, cz, voxel_spacing,
    det_offset_u, det_offset_v,
    center_offset_x, center_offset_y, center_offset_z
):
    """Pure adjoint (transpose) of _cone_3d_sf_tr_forward_kernel.

    Voxel-driven gather. No atomic adds — each thread owns a unique output
    voxel and accumulates ``weight * raw_u * raw_v * grad_sino`` into a
    local value, writing at the end.
    """
    iz, iy, ix = cuda.grid(3)
    if ix >= Nx or iy >= Ny or iz >= Nz:
        return

    sid_v = sid / voxel_spacing
    sdd_v = sdd / voxel_spacing
    du_v = du / voxel_spacing
    dv_v = dv / voxel_spacing
    u_half = (np.float32(n_u) - _ONE) * _HALF
    v_half = (np.float32(n_v) - _ONE) * _HALF

    x_c = (np.float32(ix) - cx) - center_offset_x
    y_c = (np.float32(iy) - cy) - center_offset_y
    z_c = (np.float32(iz) - cz) - center_offset_z
    x_m = x_c - _HALF
    x_p = x_c + _HALF
    y_m = y_c - _HALF
    y_p = y_c + _HALF
    z_m = z_c - _HALF
    z_p = z_c + _HALF

    grad_val = _ZERO

    for iview in range(n_views):
        cos_a = d_cos[iview]
        sin_a = d_sin[iview]

        d1 = sid_v + x_m * sin_a - y_m * cos_a
        d2 = sid_v + x_p * sin_a - y_m * cos_a
        d3 = sid_v + x_p * sin_a - y_p * cos_a
        d4 = sid_v + x_m * sin_a - y_p * cos_a
        if d1 <= _EPSILON or d2 <= _EPSILON or d3 <= _EPSILON or d4 <= _EPSILON:
            continue

        u1 = sdd_v * (x_m * cos_a + y_m * sin_a) / d1
        u2 = sdd_v * (x_p * cos_a + y_m * sin_a) / d2
        u3 = sdd_v * (x_p * cos_a + y_p * sin_a) / d3
        u4 = sdd_v * (x_m * cos_a + y_p * sin_a) / d4

        a = u1
        b = u2
        c = u3
        d = u4
        if a > b:
            a, b = b, a
        if c > d:
            c, d = d, c
        if a > c:
            a, c = c, a
        if b > d:
            b, d = d, b
        if b > c:
            b, c = c, b
        u_min = a
        u_lo = b
        u_hi = c
        u_max = d
        if (u_max - u_min) < _EPSILON:
            continue

        U_c = sid_v + x_c * sin_a - y_c * cos_a
        if U_c <= _EPSILON:
            continue
        mag_c = sdd_v / U_c
        v_bot = z_m * mag_c
        v_top = z_p * mag_c
        if v_bot > v_top:
            v_bot, v_top = v_top, v_bot
        v_span = v_top - v_bot
        if v_span < _EPSILON:
            continue

        rx = x_c - (-sid_v * sin_a)
        ry = y_c - (sid_v * cos_a)
        rz = z_c
        r_len = math.sqrt(rx * rx + ry * ry + rz * rz)
        if r_len < _EPSILON:
            continue
        rx /= r_len
        ry /= r_len
        rz /= r_len
        m = abs(rx)
        ar = abs(ry)
        if ar > m:
            m = ar
        az = abs(rz)
        if az > m:
            m = az
        if m < _EPSILON:
            continue
        chord = _ONE / m

        plateau = u_hi - u_lo
        rise_w = u_lo - u_min
        fall_w = u_max - u_hi
        weight = chord / (du_v * dv_v)

        k_u_lo = int(math.floor((u_min - det_offset_u) / du_v + u_half - _HALF))
        k_u_hi = int(math.ceil((u_max - det_offset_u) / du_v + u_half + _HALF))
        if k_u_lo < 0:
            k_u_lo = 0
        if k_u_hi > n_u - 1:
            k_u_hi = n_u - 1
        if k_u_hi < k_u_lo:
            continue

        k_v_lo = int(math.floor((v_bot - det_offset_v) / dv_v + v_half - _HALF))
        k_v_hi = int(math.ceil((v_top - det_offset_v) / dv_v + v_half + _HALF))
        if k_v_lo < 0:
            k_v_lo = 0
        if k_v_hi > n_v - 1:
            k_v_hi = n_v - 1
        if k_v_hi < k_v_lo:
            continue

        for ku in range(k_u_lo, k_u_hi + 1):
            u_k = (np.float32(ku) - u_half) * du_v + det_offset_u
            u_L = u_k - _HALF * du_v
            u_R = u_k + _HALF * du_v

            aL = u_L if u_L > u_min else u_min
            aR = u_R if u_R < u_max else u_max
            if aL >= aR:
                continue

            raw_u = _ZERO
            if rise_w > _EPSILON:
                r_lo_ = aL if aL > u_min else u_min
                r_hi_ = aR if aR < u_lo else u_lo
                if r_hi_ > r_lo_:
                    raw_u += _HALF * ((r_hi_ - u_min) * (r_hi_ - u_min) -
                                    (r_lo_ - u_min) * (r_lo_ - u_min)) / rise_w
            if plateau > _EPSILON:
                p_lo_ = aL if aL > u_lo else u_lo
                p_hi_ = aR if aR < u_hi else u_hi
                if p_hi_ > p_lo_:
                    raw_u += p_hi_ - p_lo_
            if fall_w > _EPSILON:
                f_lo_ = aL if aL > u_hi else u_hi
                f_hi_ = aR if aR < u_max else u_max
                if f_hi_ > f_lo_:
                    raw_u += _HALF * ((u_max - f_lo_) * (u_max - f_lo_) -
                                    (u_max - f_hi_) * (u_max - f_hi_)) / fall_w
            if raw_u <= _ZERO:
                continue

            for kv in range(k_v_lo, k_v_hi + 1):
                v_k = (np.float32(kv) - v_half) * dv_v + det_offset_v
                v_L = v_k - _HALF * dv_v
                v_R = v_k + _HALF * dv_v

                aLv = v_L if v_L > v_bot else v_bot
                aRv = v_R if v_R < v_top else v_top
                if aLv >= aRv:
                    continue
                raw_v = aRv - aLv

                grad_val += weight * raw_u * raw_v * d_grad_sino[iview, ku, kv]

    d_grad_vol[ix, iy, iz] = grad_val


@_FDK_ACCURACY_DECORATOR
def _cone_3d_sf_tt_forward_kernel(
    d_vol, Nx, Ny, Nz,
    d_sino, n_views, n_u, n_v,
    du, dv, d_cos, d_sin,
    sdd, sid, cx, cy, cz, voxel_spacing,
    det_offset_u, det_offset_v,
    center_offset_x, center_offset_y, center_offset_z
):
    """Cone-beam SF-TT forward projector.

    Same transaxial trapezoid as SF-TR, but the axial footprint is also a
    trapezoid built from four z-corner projections using U_near and U_far
    across the voxel. This captures the variation of axial magnification
    inside a single voxel at large cone angles, at the cost of more inner
    work per voxel-view. For axial-centred voxels the axial trapezoid
    collapses towards the SF-TR rectangle.
    """
    iz, iy, ix = cuda.grid(3)
    if ix >= Nx or iy >= Ny or iz >= Nz:
        return

    val = d_vol[ix, iy, iz]
    if val == _ZERO:
        return

    sid_v = sid / voxel_spacing
    sdd_v = sdd / voxel_spacing
    du_v = du / voxel_spacing
    dv_v = dv / voxel_spacing
    u_half = (np.float32(n_u) - _ONE) * _HALF
    v_half = (np.float32(n_v) - _ONE) * _HALF

    x_c = (np.float32(ix) - cx) - center_offset_x
    y_c = (np.float32(iy) - cy) - center_offset_y
    z_c = (np.float32(iz) - cz) - center_offset_z
    x_m = x_c - _HALF
    x_p = x_c + _HALF
    y_m = y_c - _HALF
    y_p = y_c + _HALF
    z_m = z_c - _HALF
    z_p = z_c + _HALF

    for iview in range(n_views):
        cos_a = d_cos[iview]
        sin_a = d_sin[iview]

        d1 = sid_v + x_m * sin_a - y_m * cos_a
        d2 = sid_v + x_p * sin_a - y_m * cos_a
        d3 = sid_v + x_p * sin_a - y_p * cos_a
        d4 = sid_v + x_m * sin_a - y_p * cos_a
        if d1 <= _EPSILON or d2 <= _EPSILON or d3 <= _EPSILON or d4 <= _EPSILON:
            continue

        u1 = sdd_v * (x_m * cos_a + y_m * sin_a) / d1
        u2 = sdd_v * (x_p * cos_a + y_m * sin_a) / d2
        u3 = sdd_v * (x_p * cos_a + y_p * sin_a) / d3
        u4 = sdd_v * (x_m * cos_a + y_p * sin_a) / d4

        a = u1
        b = u2
        c = u3
        d = u4
        if a > b:
            a, b = b, a
        if c > d:
            c, d = d, c
        if a > c:
            a, c = c, a
        if b > d:
            b, d = d, b
        if b > c:
            b, c = c, b
        u_min = a
        u_lo = b
        u_hi = c
        u_max = d
        if (u_max - u_min) < _EPSILON:
            continue

        # U_near / U_far across the voxel → axial trapezoid from 4 z-projections.
        U_near = d1
        if d2 < U_near:
            U_near = d2
        if d3 < U_near:
            U_near = d3
        if d4 < U_near:
            U_near = d4
        U_far = d1
        if d2 > U_far:
            U_far = d2
        if d3 > U_far:
            U_far = d3
        if d4 > U_far:
            U_far = d4
        mag_near = sdd_v / U_near
        mag_far = sdd_v / U_far

        v_bot_near = z_m * mag_near
        v_bot_far = z_m * mag_far
        v_top_near = z_p * mag_near
        v_top_far = z_p * mag_far

        # Sort the 4 v values to find v_min <= v_lo <= v_hi <= v_max.
        av = v_bot_near
        bv = v_bot_far
        cv = v_top_near
        dv_ = v_top_far
        if av > bv:
            av, bv = bv, av
        if cv > dv_:
            cv, dv_ = dv_, cv
        if av > cv:
            av, cv = cv, av
        if bv > dv_:
            bv, dv_ = dv_, bv
        if bv > cv:
            bv, cv = cv, bv
        v_min = av
        v_lo = bv
        v_hi = cv
        v_max = dv_

        v_span = v_max - v_min
        if v_span < _EPSILON:
            continue

        # Central-ray chord.
        rx = x_c - (-sid_v * sin_a)
        ry = y_c - (sid_v * cos_a)
        rz = z_c
        r_len = math.sqrt(rx * rx + ry * ry + rz * rz)
        if r_len < _EPSILON:
            continue
        rx /= r_len
        ry /= r_len
        rz /= r_len
        m = abs(rx)
        ar = abs(ry)
        if ar > m:
            m = ar
        az = abs(rz)
        if az > m:
            m = az
        if m < _EPSILON:
            continue
        chord = _ONE / m

        plateau_u = u_hi - u_lo
        rise_u = u_lo - u_min
        fall_u = u_max - u_hi

        plateau_v = v_hi - v_lo
        rise_v = v_lo - v_min
        fall_v = v_max - v_hi

        weight = val * chord / (du_v * dv_v)

        k_u_lo = int(math.floor((u_min - det_offset_u) / du_v + u_half - _HALF))
        k_u_hi = int(math.ceil((u_max - det_offset_u) / du_v + u_half + _HALF))
        if k_u_lo < 0:
            k_u_lo = 0
        if k_u_hi > n_u - 1:
            k_u_hi = n_u - 1
        if k_u_hi < k_u_lo:
            continue

        k_v_lo = int(math.floor((v_min - det_offset_v) / dv_v + v_half - _HALF))
        k_v_hi = int(math.ceil((v_max - det_offset_v) / dv_v + v_half + _HALF))
        if k_v_lo < 0:
            k_v_lo = 0
        if k_v_hi > n_v - 1:
            k_v_hi = n_v - 1
        if k_v_hi < k_v_lo:
            continue

        for ku in range(k_u_lo, k_u_hi + 1):
            u_k = (np.float32(ku) - u_half) * du_v + det_offset_u
            u_L = u_k - _HALF * du_v
            u_R = u_k + _HALF * du_v

            aLu = u_L if u_L > u_min else u_min
            aRu = u_R if u_R < u_max else u_max
            if aLu >= aRu:
                continue

            raw_u = _ZERO
            if rise_u > _EPSILON:
                r_lo_ = aLu if aLu > u_min else u_min
                r_hi_ = aRu if aRu < u_lo else u_lo
                if r_hi_ > r_lo_:
                    raw_u += _HALF * ((r_hi_ - u_min) * (r_hi_ - u_min) -
                                    (r_lo_ - u_min) * (r_lo_ - u_min)) / rise_u
            if plateau_u > _EPSILON:
                p_lo_ = aLu if aLu > u_lo else u_lo
                p_hi_ = aRu if aRu < u_hi else u_hi
                if p_hi_ > p_lo_:
                    raw_u += p_hi_ - p_lo_
            if fall_u > _EPSILON:
                f_lo_ = aLu if aLu > u_hi else u_hi
                f_hi_ = aRu if aRu < u_max else u_max
                if f_hi_ > f_lo_:
                    raw_u += _HALF * ((u_max - f_lo_) * (u_max - f_lo_) -
                                    (u_max - f_hi_) * (u_max - f_hi_)) / fall_u
            if raw_u <= _ZERO:
                continue

            for kv in range(k_v_lo, k_v_hi + 1):
                v_k = (np.float32(kv) - v_half) * dv_v + det_offset_v
                v_L = v_k - _HALF * dv_v
                v_R = v_k + _HALF * dv_v

                aLv = v_L if v_L > v_min else v_min
                aRv = v_R if v_R < v_max else v_max
                if aLv >= aRv:
                    continue

                raw_v = _ZERO
                if rise_v > _EPSILON:
                    r_lo_v = aLv if aLv > v_min else v_min
                    r_hi_v = aRv if aRv < v_lo else v_lo
                    if r_hi_v > r_lo_v:
                        raw_v += _HALF * ((r_hi_v - v_min) * (r_hi_v - v_min) -
                                        (r_lo_v - v_min) * (r_lo_v - v_min)) / rise_v
                if plateau_v > _EPSILON:
                    p_lo_v = aLv if aLv > v_lo else v_lo
                    p_hi_v = aRv if aRv < v_hi else v_hi
                    if p_hi_v > p_lo_v:
                        raw_v += p_hi_v - p_lo_v
                if fall_v > _EPSILON:
                    f_lo_v = aLv if aLv > v_hi else v_hi
                    f_hi_v = aRv if aRv < v_max else v_max
                    if f_hi_v > f_lo_v:
                        raw_v += _HALF * ((v_max - f_lo_v) * (v_max - f_lo_v) -
                                        (v_max - f_hi_v) * (v_max - f_hi_v)) / fall_v
                if raw_v <= _ZERO:
                    continue

                cuda.atomic.add(d_sino, (iview, ku, kv), weight * raw_u * raw_v)


@_FDK_ACCURACY_DECORATOR
def _cone_3d_sf_tt_backward_kernel(
    d_grad_sino, n_views, n_u, n_v,
    d_grad_vol, Nx, Ny, Nz,
    du, dv, d_cos, d_sin,
    sdd, sid, cx, cy, cz, voxel_spacing,
    det_offset_u, det_offset_v,
    center_offset_x, center_offset_y, center_offset_z
):
    """Pure adjoint (transpose) of _cone_3d_sf_tt_forward_kernel.

    Voxel-driven gather mirror of the SF-TT scatter. Same separable
    trapezoid-trapezoid coefficients, no atomics.
    """
    iz, iy, ix = cuda.grid(3)
    if ix >= Nx or iy >= Ny or iz >= Nz:
        return

    sid_v = sid / voxel_spacing
    sdd_v = sdd / voxel_spacing
    du_v = du / voxel_spacing
    dv_v = dv / voxel_spacing
    u_half = (np.float32(n_u) - _ONE) * _HALF
    v_half = (np.float32(n_v) - _ONE) * _HALF

    x_c = (np.float32(ix) - cx) - center_offset_x
    y_c = (np.float32(iy) - cy) - center_offset_y
    z_c = (np.float32(iz) - cz) - center_offset_z
    x_m = x_c - _HALF
    x_p = x_c + _HALF
    y_m = y_c - _HALF
    y_p = y_c + _HALF
    z_m = z_c - _HALF
    z_p = z_c + _HALF

    grad_val = _ZERO

    for iview in range(n_views):
        cos_a = d_cos[iview]
        sin_a = d_sin[iview]

        d1 = sid_v + x_m * sin_a - y_m * cos_a
        d2 = sid_v + x_p * sin_a - y_m * cos_a
        d3 = sid_v + x_p * sin_a - y_p * cos_a
        d4 = sid_v + x_m * sin_a - y_p * cos_a
        if d1 <= _EPSILON or d2 <= _EPSILON or d3 <= _EPSILON or d4 <= _EPSILON:
            continue

        u1 = sdd_v * (x_m * cos_a + y_m * sin_a) / d1
        u2 = sdd_v * (x_p * cos_a + y_m * sin_a) / d2
        u3 = sdd_v * (x_p * cos_a + y_p * sin_a) / d3
        u4 = sdd_v * (x_m * cos_a + y_p * sin_a) / d4

        a = u1
        b = u2
        c = u3
        d = u4
        if a > b:
            a, b = b, a
        if c > d:
            c, d = d, c
        if a > c:
            a, c = c, a
        if b > d:
            b, d = d, b
        if b > c:
            b, c = c, b
        u_min = a
        u_lo = b
        u_hi = c
        u_max = d
        if (u_max - u_min) < _EPSILON:
            continue

        U_near = d1
        if d2 < U_near:
            U_near = d2
        if d3 < U_near:
            U_near = d3
        if d4 < U_near:
            U_near = d4
        U_far = d1
        if d2 > U_far:
            U_far = d2
        if d3 > U_far:
            U_far = d3
        if d4 > U_far:
            U_far = d4
        mag_near = sdd_v / U_near
        mag_far = sdd_v / U_far

        v_bot_near = z_m * mag_near
        v_bot_far = z_m * mag_far
        v_top_near = z_p * mag_near
        v_top_far = z_p * mag_far

        av = v_bot_near
        bv = v_bot_far
        cv = v_top_near
        dv_ = v_top_far
        if av > bv:
            av, bv = bv, av
        if cv > dv_:
            cv, dv_ = dv_, cv
        if av > cv:
            av, cv = cv, av
        if bv > dv_:
            bv, dv_ = dv_, bv
        if bv > cv:
            bv, cv = cv, bv
        v_min = av
        v_lo = bv
        v_hi = cv
        v_max = dv_
        if (v_max - v_min) < _EPSILON:
            continue

        rx = x_c - (-sid_v * sin_a)
        ry = y_c - (sid_v * cos_a)
        rz = z_c
        r_len = math.sqrt(rx * rx + ry * ry + rz * rz)
        if r_len < _EPSILON:
            continue
        rx /= r_len
        ry /= r_len
        rz /= r_len
        m = abs(rx)
        ar = abs(ry)
        if ar > m:
            m = ar
        az = abs(rz)
        if az > m:
            m = az
        if m < _EPSILON:
            continue
        chord = _ONE / m

        plateau_u = u_hi - u_lo
        rise_u = u_lo - u_min
        fall_u = u_max - u_hi
        plateau_v = v_hi - v_lo
        rise_v = v_lo - v_min
        fall_v = v_max - v_hi
        weight = chord / (du_v * dv_v)

        k_u_lo = int(math.floor((u_min - det_offset_u) / du_v + u_half - _HALF))
        k_u_hi = int(math.ceil((u_max - det_offset_u) / du_v + u_half + _HALF))
        if k_u_lo < 0:
            k_u_lo = 0
        if k_u_hi > n_u - 1:
            k_u_hi = n_u - 1
        if k_u_hi < k_u_lo:
            continue

        k_v_lo = int(math.floor((v_min - det_offset_v) / dv_v + v_half - _HALF))
        k_v_hi = int(math.ceil((v_max - det_offset_v) / dv_v + v_half + _HALF))
        if k_v_lo < 0:
            k_v_lo = 0
        if k_v_hi > n_v - 1:
            k_v_hi = n_v - 1
        if k_v_hi < k_v_lo:
            continue

        for ku in range(k_u_lo, k_u_hi + 1):
            u_k = (np.float32(ku) - u_half) * du_v + det_offset_u
            u_L = u_k - _HALF * du_v
            u_R = u_k + _HALF * du_v

            aLu = u_L if u_L > u_min else u_min
            aRu = u_R if u_R < u_max else u_max
            if aLu >= aRu:
                continue

            raw_u = _ZERO
            if rise_u > _EPSILON:
                r_lo_ = aLu if aLu > u_min else u_min
                r_hi_ = aRu if aRu < u_lo else u_lo
                if r_hi_ > r_lo_:
                    raw_u += _HALF * ((r_hi_ - u_min) * (r_hi_ - u_min) -
                                    (r_lo_ - u_min) * (r_lo_ - u_min)) / rise_u
            if plateau_u > _EPSILON:
                p_lo_ = aLu if aLu > u_lo else u_lo
                p_hi_ = aRu if aRu < u_hi else u_hi
                if p_hi_ > p_lo_:
                    raw_u += p_hi_ - p_lo_
            if fall_u > _EPSILON:
                f_lo_ = aLu if aLu > u_hi else u_hi
                f_hi_ = aRu if aRu < u_max else u_max
                if f_hi_ > f_lo_:
                    raw_u += _HALF * ((u_max - f_lo_) * (u_max - f_lo_) -
                                    (u_max - f_hi_) * (u_max - f_hi_)) / fall_u
            if raw_u <= _ZERO:
                continue

            for kv in range(k_v_lo, k_v_hi + 1):
                v_k = (np.float32(kv) - v_half) * dv_v + det_offset_v
                v_L = v_k - _HALF * dv_v
                v_R = v_k + _HALF * dv_v

                aLv = v_L if v_L > v_min else v_min
                aRv = v_R if v_R < v_max else v_max
                if aLv >= aRv:
                    continue

                raw_v = _ZERO
                if rise_v > _EPSILON:
                    r_lo_v = aLv if aLv > v_min else v_min
                    r_hi_v = aRv if aRv < v_lo else v_lo
                    if r_hi_v > r_lo_v:
                        raw_v += _HALF * ((r_hi_v - v_min) * (r_hi_v - v_min) -
                                        (r_lo_v - v_min) * (r_lo_v - v_min)) / rise_v
                if plateau_v > _EPSILON:
                    p_lo_v = aLv if aLv > v_lo else v_lo
                    p_hi_v = aRv if aRv < v_hi else v_hi
                    if p_hi_v > p_lo_v:
                        raw_v += p_hi_v - p_lo_v
                if fall_v > _EPSILON:
                    f_lo_v = aLv if aLv > v_hi else v_hi
                    f_hi_v = aRv if aRv < v_max else v_max
                    if f_hi_v > f_lo_v:
                        raw_v += _HALF * ((v_max - f_lo_v) * (v_max - f_lo_v) -
                                        (v_max - f_hi_v) * (v_max - f_hi_v)) / fall_v
                if raw_v <= _ZERO:
                    continue

                grad_val += weight * raw_u * raw_v * d_grad_sino[iview, ku, kv]

    d_grad_vol[ix, iy, iz] = grad_val


# ------------------------------------------------------------------
# Analytical FDK backprojection kernel (voxel-driven gather)
# ------------------------------------------------------------------
# The Siddon-based ``_cone_3d_backward_kernel`` above is the exact adjoint of
# ``_cone_3d_forward_kernel`` and stays unchanged so that the cone autograd
# path in ConeProjectorFunction / ConeBackprojectorFunction / the iterative
# example is bit-for-bit untouched. FDK reconstruction is a different
# operation: it wants the classical voxel-driven gather form, where each
# voxel reads a single bilinearly-interpolated sample from the filtered
# sinogram at the (u, v) it projects to, weighted by ``(SID / U)^2``. This
# dedicated kernel implements that gather and is only called by
# ``cone_weighted_backproject``.

@_FDK_ACCURACY_DECORATOR
def _cone_3d_fdk_backproject_kernel(
    d_sino, n_views, n_u, n_v,
    d_vol, Nx, Ny, Nz,
    du, dv, d_cos, d_sin,
    sdd, sid, cx, cy, cz, voxel_spacing,
    det_offset_u, det_offset_v,
    center_offset_x, center_offset_y, center_offset_z,
):
    """Voxel-driven FDK backprojection gather.

    For each voxel in the output volume, for each view, compute the source-
    to-voxel distance ``U`` along the central ray direction, project the
    voxel onto the detector, bilinearly sample the filtered sinogram, and
    accumulate ``(SID/U)^2 * sample``. The cosine pre-weight, ramp filter and
    angular integration weights are expected to already have been baked into
    ``d_sino`` by the Python wrapper.

    Geometry conventions match the existing cone forward/backward kernels:
    the source rotates in the xy-plane around the isocenter at radius SID,
    the detector is a plane at distance SDD from the source perpendicular to
    the central ray, and all ``*_offset_*`` values are already in voxel
    units at call time. Indices follow the existing permuted layout where
    ``ix`` marches along the W axis, ``iy`` along H and ``iz`` along D.

    Memory-access layout note
    -------------------------
    Numba ``cuda.grid(3)`` returns ``(threadIdx.x+..., threadIdx.y+...,
    threadIdx.z+...)`` in that order, so the FIRST element is the
    warp-adjacent axis (lanes 0..31 span the first index). We deliberately
    unpack as ``(iz, iy, ix)`` - not ``(ix, iy, iz)`` - so that ``iz``
    becomes warp-adjacent. The output buffer ``d_vol`` has WHD layout
    ``(W, H, D)``, row-major, which means its innermost stride-1 axis is
    ``D`` (== ``iz``). With ``iz`` warp-adjacent, the 32 threads of a warp
    write to ``d_vol[ix, iy, iz..iz+31]`` - 32 consecutive float32 cells,
    one coalesced transaction per warp. The sinogram reads
    ``d_sino[iview, iu, iv]`` are also coalesced because ``iv`` varies
    linearly with ``iz`` (``v_det = z_v * sdd/U``) and ``d_sino`` has its
    innermost stride-1 axis on ``iv``, so a warp's ``fv`` values span a
    handful of adjacent ``iv`` bins. Matching this ordering, the Python
    wrapper launches the grid as ``_grid_3d(D, H, W)``.

    Expected performance impact of this layout is *small*: the kernel
    does ~4 sinogram loads per view per output voxel but only ONE write
    at the end, so total memory traffic is overwhelmingly read-bound and
    the L2 easily absorbs any stray write transactions. An A/B
    benchmark of the old ``ix``-first layout versus this ``iz``-first
    layout measured ~1.0x to 1.14x speedup on 64^3..160^3 volumes - well
    within run-to-run noise. We keep the coalesced layout anyway because
    it is the right default and future kernel edits that shift work
    toward the write path will benefit immediately without re-auditing.
    """
    iz, iy, ix = cuda.grid(3)
    if ix >= Nx or iy >= Ny or iz >= Nz:
        return

    # Voxel position in the "voxel-units, isocenter-centered" frame used by
    # the shared kernels. Matches the inverse of ``mid = src + t*dir + cx``.
    x_v = (np.float32(ix) - cx) - center_offset_x
    y_v = (np.float32(iy) - cy) - center_offset_y
    z_v = (np.float32(iz) - cz) - center_offset_z

    sid_v = sid / voxel_spacing
    sdd_v = sdd / voxel_spacing
    du_v = du / voxel_spacing
    dv_v = dv / voxel_spacing

    u_half = (np.float32(n_u) - _ONE) * _HALF
    v_half = (np.float32(n_v) - _ONE) * _HALF

    accum = _ZERO
    for iview in range(n_views):
        cos_a = d_cos[iview]
        sin_a = d_sin[iview]

        # Distance from source to voxel along the central ray direction.
        # Source: (-sid_v*sin_a, sid_v*cos_a, 0). Central ray: (sin_a, -cos_a, 0).
        U = sid_v + x_v * sin_a - y_v * cos_a
        if U <= _EPSILON:
            continue

        mag = sdd_v / U
        # Perpendicular in-plane detector coordinate at the detector plane.
        u_det = (x_v * cos_a + y_v * sin_a) * mag
        v_det = z_v * mag

        # Detector bin indices. Inverse of the forward kernel's u = (iu - (n_u-1)/2)*du + offset.
        fu = (u_det - det_offset_u) / du_v + u_half
        fv = (v_det - det_offset_v) / dv_v + v_half

        if fu < _ZERO or fu > (np.float32(n_u) - _ONE) or fv < _ZERO or fv > (np.float32(n_v) - _ONE):
            continue

        iu0 = int(math.floor(fu))
        iv0 = int(math.floor(fv))
        if iu0 >= n_u - 1:
            iu0 = n_u - 2
        if iv0 >= n_v - 1:
            iv0 = n_v - 2
        if iu0 < 0:
            iu0 = 0
        if iv0 < 0:
            iv0 = 0
        tu = fu - np.float32(iu0)
        tv = fv - np.float32(iv0)
        if tu < _ZERO:
            tu = _ZERO
        elif tu > _ONE:
            tu = _ONE
        if tv < _ZERO:
            tv = _ZERO
        elif tv > _ONE:
            tv = _ONE

        s00 = d_sino[iview, iu0,     iv0    ]
        s10 = d_sino[iview, iu0 + 1, iv0    ]
        s01 = d_sino[iview, iu0,     iv0 + 1]
        s11 = d_sino[iview, iu0 + 1, iv0 + 1]

        omtu = _ONE - tu
        omtv = _ONE - tv
        sample = (
            s00 * omtu * omtv
            + s10 * tu   * omtv
            + s01 * omtu * tv
            + s11 * tu   * tv
        )

        w = sid_v / U
        accum += (w * w) * sample

    d_vol[ix, iy, iz] = accum


# ------------------------------------------------------------------
# Analytical parallel-beam FBP backprojection kernel (voxel-driven gather)
# ------------------------------------------------------------------
# Parallel beam has no source and therefore no ``(sid/U)^2`` distance
# weight - the gather is just a linear detector interpolation summed
# over views. Unlike the Siddon-scatter adjoint, this kernel has no
# per-angle ``seg_len`` bias, so it delivers the same one-percent
# calibration accuracy as the fan and cone gather kernels. Only
# ``parallel_weighted_backproject`` calls it; autograd keeps using
# ``_parallel_2d_backward_kernel`` for the pure adjoint path.

@_FDK_ACCURACY_DECORATOR
def _parallel_2d_fbp_backproject_kernel(
    d_sino, n_ang, n_det,
    d_image, Nx, Ny,
    det_spacing, d_cos, d_sin,
    cx, cy, voxel_spacing,
    det_offset, center_offset_x, center_offset_y,
):
    """Voxel-driven parallel-beam FBP backprojection gather.

    For each pixel and each view, compute the detector-u coordinate the
    pixel projects to (``u = -x*sin_a + y*cos_a`` in voxel units,
    matching the parallel forward kernel's ``pnt = (-u*sin_a, u*cos_a)``
    and ``dir = (cos_a, sin_a)`` convention), linearly sample the
    filtered sinogram at that position, and accumulate. Cosine
    pre-weighting, ramp filtering and per-view integration weights are
    expected to already be baked into ``d_sino`` by the Python wrapper.

    Memory-access layout note
    -------------------------
    Numba ``cuda.grid(2)`` returns ``(threadIdx.x+..., threadIdx.y+...)``
    in that order, so the FIRST element is the warp-adjacent axis. We
    unpack as ``(ix, iy)``, which puts ``ix`` on ``threadIdx.x``. The
    output buffer has shape ``(Ny, Nx)`` row-major, whose innermost
    stride-1 axis is ``Nx`` (== ``ix``). Writing ``d_image[iy, ix]`` is
    therefore coalesced: the 32 threads of a warp write 32 consecutive
    float32 cells in one transaction. The grid is launched as
    ``_grid_2d(Nx, Ny)`` to match this ordering.
    """
    ix, iy = cuda.grid(2)
    if ix >= Nx or iy >= Ny:
        return

    x_v = (np.float32(ix) - cx) - center_offset_x
    y_v = (np.float32(iy) - cy) - center_offset_y

    det_v = det_spacing / voxel_spacing
    u_half = (np.float32(n_det) - _ONE) * _HALF

    accum = _ZERO
    for iang in range(n_ang):
        cos_a = d_cos[iang]
        sin_a = d_sin[iang]

        # Detector-u coordinate of the pixel (parallel convention).
        u_pix = -x_v * sin_a + y_v * cos_a

        # Inverse of the forward kernel's u index formula.
        fu = (u_pix - det_offset) / det_v + u_half
        if fu < _ZERO or fu > (np.float32(n_det) - _ONE):
            continue

        idet0 = int(math.floor(fu))
        if idet0 >= n_det - 1:
            idet0 = n_det - 2
        if idet0 < 0:
            idet0 = 0
        tu = fu - np.float32(idet0)
        if tu < _ZERO:
            tu = _ZERO
        elif tu > _ONE:
            tu = _ONE

        s0 = d_sino[iang, idet0]
        s1 = d_sino[iang, idet0 + 1]
        accum += s0 * (_ONE - tu) + s1 * tu

    d_image[iy, ix] = accum


# ------------------------------------------------------------------
# Analytical fan-beam FBP backprojection kernel (voxel-driven gather)
# ------------------------------------------------------------------
# Same design as the cone FDK gather above but in 2D. The Siddon-based
# ``_fan_2d_backward_kernel`` stays the pure adjoint of the fan forward
# projector and drives autograd; this dedicated FBP kernel is the
# analytical path and is only invoked by ``fan_weighted_backproject``.

@_FDK_ACCURACY_DECORATOR
def _fan_2d_fbp_backproject_kernel(
    d_sino, n_ang, n_det,
    d_image, Nx, Ny,
    det_spacing, d_cos, d_sin,
    sdd, sid, cx, cy, voxel_spacing,
    det_offset, center_offset_x, center_offset_y,
):
    """Voxel-driven fan-beam FBP backprojection gather.

    For each image pixel and each view, compute the source-to-pixel
    distance ``U`` along the central ray, project the pixel onto the
    detector, linearly sample the filtered sinogram, and accumulate
    ``(SID/U)^2 * sample``. Cosine pre-weighting, ramp filtering and
    per-view integration weights are expected to already be baked into
    ``d_sino`` by the Python wrapper. Matches the fan forward kernel's
    geometry: source at ``(-sid_v*sin_a, sid_v*cos_a)`` relative to the
    isocenter, detector at distance SDD from the source perpendicular
    to the central ray, ``u`` on the detector is in-plane.

    Memory-access layout note
    -------------------------
    Same coalescing pattern as the parallel FBP gather kernel above:
    ``ix`` is on ``threadIdx.x`` (warp-adjacent), and the output buffer
    ``(Ny, Nx)`` has ``Nx`` as its innermost stride-1 axis, so
    ``d_image[iy, ix]`` writes 32 consecutive float32 cells per warp.
    Sinogram reads ``d_sino[iang, idet0..idet0+1]`` also sweep through
    a small span of the stride-1 detector axis as ``ix`` varies across
    a warp, so they are approximately coalesced as well. Launch grid is
    ``_grid_2d(Nx, Ny)``.
    """
    ix, iy = cuda.grid(2)
    if ix >= Nx or iy >= Ny:
        return

    # Pixel position in the "voxel-units, isocenter-centered" frame used
    # by the Siddon forward kernel. Matches the inverse of
    # ``mid = src + t*dir + cx`` in the forward kernel.
    x_v = (np.float32(ix) - cx) - center_offset_x
    y_v = (np.float32(iy) - cy) - center_offset_y

    sid_v = sid / voxel_spacing
    sdd_v = sdd / voxel_spacing
    det_v = det_spacing / voxel_spacing
    u_half = (np.float32(n_det) - _ONE) * _HALF

    accum = _ZERO
    for iang in range(n_ang):
        cos_a = d_cos[iang]
        sin_a = d_sin[iang]

        # Distance from source to pixel along the central ray direction.
        U = sid_v + x_v * sin_a - y_v * cos_a
        if U <= _EPSILON:
            continue

        mag = sdd_v / U
        u_det = (x_v * cos_a + y_v * sin_a) * mag

        # Inverse of the forward kernel's u index formula.
        fu = (u_det - det_offset) / det_v + u_half
        if fu < _ZERO or fu > (np.float32(n_det) - _ONE):
            continue

        idet0 = int(math.floor(fu))
        if idet0 >= n_det - 1:
            idet0 = n_det - 2
        if idet0 < 0:
            idet0 = 0
        tu = fu - np.float32(idet0)
        if tu < _ZERO:
            tu = _ZERO
        elif tu > _ONE:
            tu = _ONE

        s0 = d_sino[iang, idet0]
        s1 = d_sino[iang, idet0 + 1]
        sample = s0 * (_ONE - tu) + s1 * tu

        w = sid_v / U
        accum += (w * w) * sample

    d_image[iy, ix] = accum


# ------------------------------------------------------------------
# Analytical SF-based FBP / FDK backprojection kernels
# ------------------------------------------------------------------
# These gather kernels replace the voxel-driven bilinear gather in
# ``fan_weighted_backproject`` / ``cone_weighted_backproject`` when the
# caller selects a separable-footprint backend. The math of each kernel
# is the matched-adjoint SF backward (four-corner trapezoidal footprint
# integration of the filtered sinogram) PLUS the classical
# ``(SID / U)^2`` FBP / FDK weight that the VD gather kernels already
# apply, where ``U`` is the source-to-voxel-center distance along the
# central ray direction. The effect compared to VD gather is:
#
#   - at the nominal voxel size ``du * sid / sdd``: near-identical MTF
#     (LEAP's SF_vs_VD analysis confirms this).
#   - at sub-nominal voxel sizes (voxels whose projected footprint
#     covers less than one detector bin): SF gives meaningfully higher
#     spatial resolution because VD always averages the filtered
#     sinogram over four detector bins regardless of voxel size.
#   - at supra-nominal voxel sizes: SF gives higher SNR because its
#     footprint spans more detector bins and averages more noise.
#
# The existing SF adjoint kernels (``_fan_2d_sf_backward_kernel``,
# ``_cone_3d_sf_tr_backward_kernel``, ``_cone_3d_sf_tt_backward_kernel``)
# stay unchanged because they are the byte-accurate matched adjoints of
# the SF forward kernels used by the autograd ``*ProjectorFunction`` /
# ``*BackprojectorFunction`` paths. The new ``*_fbp_*`` / ``*_fdk_*``
# kernels below are ONLY used by the analytical pipeline wrappers and
# are not the autograd adjoint of anything.


@_FDK_ACCURACY_DECORATOR
def _fan_2d_sf_fbp_backproject_kernel(
    d_sino, n_ang, n_det,
    d_image, Nx, Ny,
    det_spacing, d_cos, d_sin,
    sdd, sid, cx, cy, voxel_spacing,
    det_offset, center_offset_x, center_offset_y,
):
    """Voxel-driven fan-beam FBP SF backprojection gather, chord-weighted form.

    Matched-adjoint formulation inspired by LEAP's ``fanBeamBackprojectorKernel_SF``
    (``projectors_SF.cu``): each voxel's trapezoidal u-footprint is still
    built from the four projected corners and closed-form integrated per
    detector bin, but the per-view weight is the chord through the unit
    voxel times the fan ``sid/U`` first-power weight, with NO division
    by the footprint area. Compared to the classical ``(SID/U)^2``-
    per-voxel FBP form, this matches LEAP's SF: ``bpWeight * chord``
    replaces ``(sid/U)^2 / area``. The two differ by a constant factor
    of ``sdd / sid`` (the magnification) at the isocenter and by a
    small position-dependent correction elsewhere, which the Python
    wrapper absorbs in the FBP scale.

    Analytical ramp filtering, cosine pre-weighting and Parker short-
    scan weights are expected to already be baked into ``d_sino`` before
    this kernel runs.
    """
    ix, iy = cuda.grid(2)
    if iy >= Ny or ix >= Nx:
        return

    sid_v = sid / voxel_spacing
    sdd_v = sdd / voxel_spacing
    det_spacing_v = det_spacing / voxel_spacing
    half = (np.float32(n_det) - _ONE) * _HALF

    x_c = (np.float32(ix) - cx) - center_offset_x
    y_c = (np.float32(iy) - cy) - center_offset_y
    x_m = x_c - _HALF
    x_p = x_c + _HALF
    y_m = y_c - _HALF
    y_p = y_c + _HALF

    accum = _ZERO

    for iang in range(n_ang):
        cos_a = d_cos[iang]
        sin_a = d_sin[iang]

        # Fan-beam ``sid/U`` first-power weight (LEAP's ``bpWeight``).
        U_c = sid_v + x_c * sin_a - y_c * cos_a
        if U_c <= _EPSILON:
            continue
        bp_weight = sid_v / U_c

        # Chord through the unit voxel along the source-to-voxel ray
        # direction. This is the physical line integral of the voxel
        # at this view and matches the SF forward kernel convention.
        rx = x_c + sid_v * sin_a
        ry = y_c - sid_v * cos_a
        r_len = math.sqrt(rx * rx + ry * ry)
        if r_len < _EPSILON:
            continue
        rx /= r_len
        ry /= r_len
        max_dir = abs(rx)
        ar_dir = abs(ry)
        if ar_dir > max_dir:
            max_dir = ar_dir
        if max_dir < _EPSILON:
            continue
        chord = _ONE / max_dir

        d1 = sid_v + x_m * sin_a - y_m * cos_a
        d2 = sid_v + x_p * sin_a - y_m * cos_a
        d3 = sid_v + x_p * sin_a - y_p * cos_a
        d4 = sid_v + x_m * sin_a - y_p * cos_a
        if d1 <= _EPSILON or d2 <= _EPSILON or d3 <= _EPSILON or d4 <= _EPSILON:
            continue

        u1 = sdd_v * (x_m * cos_a + y_m * sin_a) / d1
        u2 = sdd_v * (x_p * cos_a + y_m * sin_a) / d2
        u3 = sdd_v * (x_p * cos_a + y_p * sin_a) / d3
        u4 = sdd_v * (x_m * cos_a + y_p * sin_a) / d4

        a = u1
        b = u2
        c = u3
        d = u4
        if a > b:
            a, b = b, a
        if c > d:
            c, d = d, c
        if a > c:
            a, c = c, a
        if b > d:
            b, d = d, b
        if b > c:
            b, c = c, b
        u_min = a
        u_lo = b
        u_hi = c
        u_max = d

        span = u_max - u_min
        if span < _EPSILON:
            continue

        plateau = u_hi - u_lo
        # Chord-weighted LEAP form: ``bp_weight * chord / det_spacing_v``.
        # No footprint-area division - the raw trapezoid integral
        # ``sum_k raw[k]`` stays intact. The Python wrapper scales the
        # final image by ``1 / (2*pi)`` instead of the classical
        # ``sdd / (2*pi*sid)`` to absorb the resulting magnification.
        weight = bp_weight * chord / det_spacing_v

        k_lo_f = (u_min - det_offset) / det_spacing_v + half - _HALF
        k_hi_f = (u_max - det_offset) / det_spacing_v + half + _HALF
        k_lo = int(math.floor(k_lo_f))
        k_hi = int(math.ceil(k_hi_f))
        if k_lo < 0:
            k_lo = 0
        if k_hi > n_det - 1:
            k_hi = n_det - 1
        if k_hi < k_lo:
            continue

        rise_w = u_lo - u_min
        fall_w = u_max - u_hi

        for k in range(k_lo, k_hi + 1):
            u_k = (np.float32(k) - half) * det_spacing_v + det_offset
            u_L = u_k - _HALF * det_spacing_v
            u_R = u_k + _HALF * det_spacing_v

            aL = u_L if u_L > u_min else u_min
            aR = u_R if u_R < u_max else u_max
            if aL >= aR:
                continue

            raw = _ZERO
            if rise_w > _EPSILON:
                r_lo_ = aL if aL > u_min else u_min
                r_hi_ = aR if aR < u_lo else u_lo
                if r_hi_ > r_lo_:
                    raw += _HALF * ((r_hi_ - u_min) * (r_hi_ - u_min) -
                                  (r_lo_ - u_min) * (r_lo_ - u_min)) / rise_w

            if plateau > _EPSILON:
                p_lo_ = aL if aL > u_lo else u_lo
                p_hi_ = aR if aR < u_hi else u_hi
                if p_hi_ > p_lo_:
                    raw += p_hi_ - p_lo_

            if fall_w > _EPSILON:
                f_lo_ = aL if aL > u_hi else u_hi
                f_hi_ = aR if aR < u_max else u_max
                if f_hi_ > f_lo_:
                    raw += _HALF * ((u_max - f_lo_) * (u_max - f_lo_) -
                                  (u_max - f_hi_) * (u_max - f_hi_)) / fall_w

            if raw > _ZERO:
                accum += weight * raw * d_sino[iang, k]

    d_image[iy, ix] = accum


@_FDK_ACCURACY_DECORATOR
def _cone_3d_sf_tr_fdk_backproject_kernel(
    d_sino, n_views, n_u, n_v,
    d_vol, Nx, Ny, Nz,
    du, dv, d_cos, d_sin,
    sdd, sid, cx, cy, cz, voxel_spacing,
    det_offset_u, det_offset_v,
    center_offset_x, center_offset_y, center_offset_z,
):
    """Voxel-driven cone-beam FDK SF-TR backprojection gather, chord-weighted.

    Matched-adjoint formulation inspired by LEAP's tilt==0 branch of
    ``coneBeamBackprojectorKernel_SF`` (``projectors_SF.cu``). Transaxial
    trapezoid + axial rectangle footprint math as before, but the
    per-view weight is the in-plane chord through the unit voxel times
    ``sqrt(1 + (v_proj/sdd)^2)`` (the LEAP v-chord correction), with NO
    division by the footprint area and NO ``(sid/U)^2`` FDK weight.
    LEAP's cone SF backprojection is a pure matched adjoint; the
    classical FDK magnification is absorbed by the Python-wrapper
    scale constant (``sid / (2*pi*sdd)`` instead of ``sdd / (2*pi*sid)``).
    """
    iz, iy, ix = cuda.grid(3)
    if ix >= Nx or iy >= Ny or iz >= Nz:
        return

    sid_v = sid / voxel_spacing
    sdd_v = sdd / voxel_spacing
    du_v = du / voxel_spacing
    dv_v = dv / voxel_spacing
    u_half = (np.float32(n_u) - _ONE) * _HALF
    v_half = (np.float32(n_v) - _ONE) * _HALF

    x_c = (np.float32(ix) - cx) - center_offset_x
    y_c = (np.float32(iy) - cy) - center_offset_y
    z_c = (np.float32(iz) - cz) - center_offset_z
    x_m = x_c - _HALF
    x_p = x_c + _HALF
    y_m = y_c - _HALF
    y_p = y_c + _HALF
    z_m = z_c - _HALF
    z_p = z_c + _HALF

    accum = _ZERO

    for iview in range(n_views):
        cos_a = d_cos[iview]
        sin_a = d_sin[iview]

        U_c = sid_v + x_c * sin_a - y_c * cos_a
        if U_c <= _EPSILON:
            continue

        # In-plane chord through the unit voxel along the source-to-
        # voxel ray direction (LEAP's ``l_phi`` uses the central-ray
        # approximation ``1/max(|cos|,|sin|) * sqrt(1+u^2/sdd^2)``;
        # we compute the exact chord from the per-voxel ray direction
        # to stay consistent with ``_cone_3d_sf_tr_forward_kernel``).
        rx = x_c + sid_v * sin_a
        ry = y_c - sid_v * cos_a
        r_len2 = rx * rx + ry * ry
        if r_len2 < _EPSILON:
            continue
        r_len = math.sqrt(r_len2)
        rx /= r_len
        ry /= r_len
        max_dir = abs(rx)
        ar_dir = abs(ry)
        if ar_dir > max_dir:
            max_dir = ar_dir
        if max_dir < _EPSILON:
            continue
        chord_u = _ONE / max_dir

        d1 = sid_v + x_m * sin_a - y_m * cos_a
        d2 = sid_v + x_p * sin_a - y_m * cos_a
        d3 = sid_v + x_p * sin_a - y_p * cos_a
        d4 = sid_v + x_m * sin_a - y_p * cos_a
        if d1 <= _EPSILON or d2 <= _EPSILON or d3 <= _EPSILON or d4 <= _EPSILON:
            continue

        u1 = sdd_v * (x_m * cos_a + y_m * sin_a) / d1
        u2 = sdd_v * (x_p * cos_a + y_m * sin_a) / d2
        u3 = sdd_v * (x_p * cos_a + y_p * sin_a) / d3
        u4 = sdd_v * (x_m * cos_a + y_p * sin_a) / d4

        a = u1
        b = u2
        c = u3
        d = u4
        if a > b:
            a, b = b, a
        if c > d:
            c, d = d, c
        if a > c:
            a, c = c, a
        if b > d:
            b, d = d, b
        if b > c:
            b, c = c, b
        u_min = a
        u_lo = b
        u_hi = c
        u_max = d
        if (u_max - u_min) < _EPSILON:
            continue

        mag_c = sdd_v / U_c
        v_proj_c = z_c * mag_c
        v_arg = v_proj_c / sdd_v
        v_chord = math.sqrt(_ONE + v_arg * v_arg)
        v_bot = z_m * mag_c
        v_top = z_p * mag_c
        if v_bot > v_top:
            v_bot, v_top = v_top, v_bot
        v_span = v_top - v_bot
        if v_span < _EPSILON:
            continue

        plateau = u_hi - u_lo
        rise_w = u_lo - u_min
        fall_w = u_max - u_hi
        # Chord-weighted LEAP form (no footprint-area division, no
        # ``(sid/U)^2`` per-voxel FDK weight). The Python wrapper
        # rescales the final volume by ``sid / (2*pi*sdd)`` instead
        # of ``sdd / (2*pi*sid)`` to absorb the ``mag^2`` difference.
        weight = chord_u * v_chord / (du_v * dv_v)

        k_u_lo = int(math.floor((u_min - det_offset_u) / du_v + u_half - _HALF))
        k_u_hi = int(math.ceil((u_max - det_offset_u) / du_v + u_half + _HALF))
        if k_u_lo < 0:
            k_u_lo = 0
        if k_u_hi > n_u - 1:
            k_u_hi = n_u - 1
        if k_u_hi < k_u_lo:
            continue

        k_v_lo = int(math.floor((v_bot - det_offset_v) / dv_v + v_half - _HALF))
        k_v_hi = int(math.ceil((v_top - det_offset_v) / dv_v + v_half + _HALF))
        if k_v_lo < 0:
            k_v_lo = 0
        if k_v_hi > n_v - 1:
            k_v_hi = n_v - 1
        if k_v_hi < k_v_lo:
            continue

        for ku in range(k_u_lo, k_u_hi + 1):
            u_k = (np.float32(ku) - u_half) * du_v + det_offset_u
            u_L = u_k - _HALF * du_v
            u_R = u_k + _HALF * du_v

            aL = u_L if u_L > u_min else u_min
            aR = u_R if u_R < u_max else u_max
            if aL >= aR:
                continue

            raw_u = _ZERO
            if rise_w > _EPSILON:
                r_lo_ = aL if aL > u_min else u_min
                r_hi_ = aR if aR < u_lo else u_lo
                if r_hi_ > r_lo_:
                    raw_u += _HALF * ((r_hi_ - u_min) * (r_hi_ - u_min) -
                                    (r_lo_ - u_min) * (r_lo_ - u_min)) / rise_w
            if plateau > _EPSILON:
                p_lo_ = aL if aL > u_lo else u_lo
                p_hi_ = aR if aR < u_hi else u_hi
                if p_hi_ > p_lo_:
                    raw_u += p_hi_ - p_lo_
            if fall_w > _EPSILON:
                f_lo_ = aL if aL > u_hi else u_hi
                f_hi_ = aR if aR < u_max else u_max
                if f_hi_ > f_lo_:
                    raw_u += _HALF * ((u_max - f_lo_) * (u_max - f_lo_) -
                                    (u_max - f_hi_) * (u_max - f_hi_)) / fall_w
            if raw_u <= _ZERO:
                continue

            for kv in range(k_v_lo, k_v_hi + 1):
                v_k = (np.float32(kv) - v_half) * dv_v + det_offset_v
                v_L = v_k - _HALF * dv_v
                v_R = v_k + _HALF * dv_v

                aLv = v_L if v_L > v_bot else v_bot
                aRv = v_R if v_R < v_top else v_top
                if aLv >= aRv:
                    continue
                raw_v = aRv - aLv

                accum += weight * raw_u * raw_v * d_sino[iview, ku, kv]

    d_vol[ix, iy, iz] = accum


@_FDK_ACCURACY_DECORATOR
def _cone_3d_sf_tt_fdk_backproject_kernel(
    d_sino, n_views, n_u, n_v,
    d_vol, Nx, Ny, Nz,
    du, dv, d_cos, d_sin,
    sdd, sid, cx, cy, cz, voxel_spacing,
    det_offset_u, det_offset_v,
    center_offset_x, center_offset_y, center_offset_z,
):
    """Voxel-driven cone-beam FDK SF-TT backprojection gather, chord-weighted.

    Matched-adjoint formulation inspired by LEAP's cone SF. Four-corner
    transaxial trapezoid and four-corner axial trapezoid footprint math
    as before, but the per-view weight is the in-plane chord through the
    unit voxel (from the per-voxel ray direction, matching the SF
    forward kernel) times ``sqrt(1+(v_proj/sdd)^2)``. No footprint-area
    division and no ``(sid/U)^2`` FDK weight; the Python-wrapper scale
    constant absorbs the magnification difference.
    """
    iz, iy, ix = cuda.grid(3)
    if ix >= Nx or iy >= Ny or iz >= Nz:
        return

    sid_v = sid / voxel_spacing
    sdd_v = sdd / voxel_spacing
    du_v = du / voxel_spacing
    dv_v = dv / voxel_spacing
    u_half = (np.float32(n_u) - _ONE) * _HALF
    v_half = (np.float32(n_v) - _ONE) * _HALF

    x_c = (np.float32(ix) - cx) - center_offset_x
    y_c = (np.float32(iy) - cy) - center_offset_y
    z_c = (np.float32(iz) - cz) - center_offset_z
    x_m = x_c - _HALF
    x_p = x_c + _HALF
    y_m = y_c - _HALF
    y_p = y_c + _HALF
    z_m = z_c - _HALF
    z_p = z_c + _HALF

    accum = _ZERO

    for iview in range(n_views):
        cos_a = d_cos[iview]
        sin_a = d_sin[iview]

        U_c = sid_v + x_c * sin_a - y_c * cos_a
        if U_c <= _EPSILON:
            continue

        rx = x_c + sid_v * sin_a
        ry = y_c - sid_v * cos_a
        r_len2 = rx * rx + ry * ry
        if r_len2 < _EPSILON:
            continue
        r_len = math.sqrt(r_len2)
        rx /= r_len
        ry /= r_len
        max_dir = abs(rx)
        ar_dir = abs(ry)
        if ar_dir > max_dir:
            max_dir = ar_dir
        if max_dir < _EPSILON:
            continue
        chord_u = _ONE / max_dir

        d1 = sid_v + x_m * sin_a - y_m * cos_a
        d2 = sid_v + x_p * sin_a - y_m * cos_a
        d3 = sid_v + x_p * sin_a - y_p * cos_a
        d4 = sid_v + x_m * sin_a - y_p * cos_a
        if d1 <= _EPSILON or d2 <= _EPSILON or d3 <= _EPSILON or d4 <= _EPSILON:
            continue

        u1 = sdd_v * (x_m * cos_a + y_m * sin_a) / d1
        u2 = sdd_v * (x_p * cos_a + y_m * sin_a) / d2
        u3 = sdd_v * (x_p * cos_a + y_p * sin_a) / d3
        u4 = sdd_v * (x_m * cos_a + y_p * sin_a) / d4

        a = u1
        b = u2
        c = u3
        d = u4
        if a > b:
            a, b = b, a
        if c > d:
            c, d = d, c
        if a > c:
            a, c = c, a
        if b > d:
            b, d = d, b
        if b > c:
            b, c = c, b
        u_min = a
        u_lo = b
        u_hi = c
        u_max = d
        if (u_max - u_min) < _EPSILON:
            continue

        U_near = d1
        if d2 < U_near:
            U_near = d2
        if d3 < U_near:
            U_near = d3
        if d4 < U_near:
            U_near = d4
        U_far = d1
        if d2 > U_far:
            U_far = d2
        if d3 > U_far:
            U_far = d3
        if d4 > U_far:
            U_far = d4
        mag_near = sdd_v / U_near
        mag_far = sdd_v / U_far

        v_bot_near = z_m * mag_near
        v_bot_far = z_m * mag_far
        v_top_near = z_p * mag_near
        v_top_far = z_p * mag_far

        av = v_bot_near
        bv = v_bot_far
        cv = v_top_near
        dv_sv = v_top_far
        if av > bv:
            av, bv = bv, av
        if cv > dv_sv:
            cv, dv_sv = dv_sv, cv
        if av > cv:
            av, cv = cv, av
        if bv > dv_sv:
            bv, dv_sv = dv_sv, bv
        if bv > cv:
            bv, cv = cv, bv
        v_min = av
        v_lo = bv
        v_hi = cv
        v_max = dv_sv
        if (v_max - v_min) < _EPSILON:
            continue

        plateau_u = u_hi - u_lo
        rise_u = u_lo - u_min
        fall_u = u_max - u_hi
        plateau_v = v_hi - v_lo
        rise_v = v_lo - v_min
        fall_v = v_max - v_hi
        # Chord-weighted LEAP form. ``v_chord = sqrt(1+(v/sdd)^2)`` is
        # evaluated at the voxel's projected v centre (mean of the 4
        # z-corner projections) to match LEAP's convention.
        v_proj_c = _QUARTER * (v_min + v_lo + v_hi + v_max)
        v_arg_tt = v_proj_c / sdd_v
        v_chord = math.sqrt(_ONE + v_arg_tt * v_arg_tt)
        u_span = u_max - u_min
        v_span = v_max - v_min
        weight = chord_u * v_chord / (du_v * dv_v)

        k_u_lo = int(math.floor((u_min - det_offset_u) / du_v + u_half - _HALF))
        k_u_hi = int(math.ceil((u_max - det_offset_u) / du_v + u_half + _HALF))
        if k_u_lo < 0:
            k_u_lo = 0
        if k_u_hi > n_u - 1:
            k_u_hi = n_u - 1
        if k_u_hi < k_u_lo:
            continue

        k_v_lo = int(math.floor((v_min - det_offset_v) / dv_v + v_half - _HALF))
        k_v_hi = int(math.ceil((v_max - det_offset_v) / dv_v + v_half + _HALF))
        if k_v_lo < 0:
            k_v_lo = 0
        if k_v_hi > n_v - 1:
            k_v_hi = n_v - 1
        if k_v_hi < k_v_lo:
            continue

        for ku in range(k_u_lo, k_u_hi + 1):
            u_k = (np.float32(ku) - u_half) * du_v + det_offset_u
            u_L = u_k - _HALF * du_v
            u_R = u_k + _HALF * du_v

            aLu = u_L if u_L > u_min else u_min
            aRu = u_R if u_R < u_max else u_max
            if aLu >= aRu:
                continue

            raw_u = _ZERO
            if rise_u > _EPSILON:
                r_lo_ = aLu if aLu > u_min else u_min
                r_hi_ = aRu if aRu < u_lo else u_lo
                if r_hi_ > r_lo_:
                    raw_u += _HALF * ((r_hi_ - u_min) * (r_hi_ - u_min) -
                                    (r_lo_ - u_min) * (r_lo_ - u_min)) / rise_u
            if plateau_u > _EPSILON:
                p_lo_ = aLu if aLu > u_lo else u_lo
                p_hi_ = aRu if aRu < u_hi else u_hi
                if p_hi_ > p_lo_:
                    raw_u += p_hi_ - p_lo_
            if fall_u > _EPSILON:
                f_lo_ = aLu if aLu > u_hi else u_hi
                f_hi_ = aRu if aRu < u_max else u_max
                if f_hi_ > f_lo_:
                    raw_u += _HALF * ((u_max - f_lo_) * (u_max - f_lo_) -
                                    (u_max - f_hi_) * (u_max - f_hi_)) / fall_u
            if raw_u <= _ZERO:
                continue

            for kv in range(k_v_lo, k_v_hi + 1):
                v_k = (np.float32(kv) - v_half) * dv_v + det_offset_v
                v_L = v_k - _HALF * dv_v
                v_R = v_k + _HALF * dv_v

                aLv = v_L if v_L > v_min else v_min
                aRv = v_R if v_R < v_max else v_max
                if aLv >= aRv:
                    continue

                raw_v = _ZERO
                if rise_v > _EPSILON:
                    r_lo_v = aLv if aLv > v_min else v_min
                    r_hi_v = aRv if aRv < v_lo else v_lo
                    if r_hi_v > r_lo_v:
                        raw_v += _HALF * ((r_hi_v - v_min) * (r_hi_v - v_min) -
                                        (r_lo_v - v_min) * (r_lo_v - v_min)) / rise_v
                if plateau_v > _EPSILON:
                    p_lo_v = aLv if aLv > v_lo else v_lo
                    p_hi_v = aRv if aRv < v_hi else v_hi
                    if p_hi_v > p_lo_v:
                        raw_v += p_hi_v - p_lo_v
                if fall_v > _EPSILON:
                    f_lo_v = aLv if aLv > v_hi else v_hi
                    f_hi_v = aRv if aRv < v_max else v_max
                    if f_hi_v > f_lo_v:
                        raw_v += _HALF * ((v_max - f_lo_v) * (v_max - f_lo_v) -
                                        (v_max - f_hi_v) * (v_max - f_hi_v)) / fall_v
                if raw_v <= _ZERO:
                    continue

                accum += weight * raw_u * raw_v * d_sino[iview, ku, kv]

    d_vol[ix, iy, iz] = accum


# ############################################################################
# DIFFERENTIABLE TORCH FUNCTIONS
# ############################################################################

class ParallelProjectorFunction(torch.autograd.Function):
    """
    Summary
    -------
    PyTorch autograd function for differentiable 2D parallel beam forward projection.

    Notes
    -----
    Provides a differentiable interface to the CUDA-accelerated Siddon ray-tracing
    method with a cell-constant image basis for parallel beam CT geometry. The forward pass computes
    the sinogram from a 2D image using parallel beam geometry. The backward pass
    computes gradients using the adjoint backprojection operation. Requires
    CUDA-capable hardware and a properly configured CUDA environment; all input
    tensors must reside on the same CUDA device.

    Examples
    --------
    >>> import torch
    >>> from diffct.differentiable import ParallelProjectorFunction
    >>>
    >>> # Create a 2D image with gradient tracking
    >>> image = torch.randn(128, 128, device='cuda', requires_grad=True)
    >>> # Define projection parameters
    >>> angles = torch.linspace(0, torch.pi, 180, device='cuda')
    >>> num_detectors = 128
    >>> detector_spacing = 1.0
    >>> # Compute forward projection
    >>> projector = ParallelProjectorFunction.apply
    >>> sinogram = projector(image, angles, num_detectors, detector_spacing)
    >>> # Compute loss and gradients
    >>> loss = sinogram.sum()
    >>> loss.backward()
    >>> print(f"Gradient shape: {image.grad.shape}")  # (128, 128)
    """
    @staticmethod
    def forward(
        ctx,
        image,
        angles,
        num_detectors,
        detector_spacing=1.0,
        voxel_spacing=1.0,
        detector_offset=0.0,
        center_offset_x=0.0,
        center_offset_y=0.0,
    ):
        """Compute the 2D parallel beam forward projection (Radon transform) of
        an image using CUDA acceleration.

        Parameters
        ----------
        image : torch.Tensor
            2D input image tensor of shape (H, W), must be on a CUDA device and of type float32.
        angles : torch.Tensor
            1D tensor of projection angles in radians, shape (num_angles,), must be on the same CUDA device as `image`.
        num_detectors : int
            Number of detector elements in the sinogram (columns).
        detector_spacing : float, optional
            Physical spacing between detector elements (default: 1.0).
        voxel_spacing : float, optional
            Physical size of one voxel (in same units as detector_spacing, default: 1.0).

        Returns
        -------
        sinogram : torch.Tensor
            2D tensor of shape (num_angles, num_detectors) containing the forward projection (sinogram) on the same device as `image`.

        Notes
        -----
        - All input tensors must be on the same CUDA device.
        - The operation is fully differentiable and supports autograd.
        - Uses cell-constant Siddon ray tracing.

        Examples
        --------
        >>> image = torch.randn(128, 128, device='cuda', requires_grad=True)
        >>> angles = torch.linspace(0, torch.pi, 180, device='cuda')
        >>> sinogram = ParallelProjectorFunction.apply(
        ...     image, angles, 128, 1.0
        ... )
        """
        device = DeviceManager.get_device(image)
        image = DeviceManager.ensure_device(image, device)
        angles = DeviceManager.ensure_device(angles, device)

        # Ensure input is float32 for kernel compatibility
        image = image.to(dtype=torch.float32).contiguous()
        angles = angles.to(dtype=torch.float32).contiguous()

        Ny, Nx = image.shape
        n_angles = angles.shape[0]

        # Allocate output tensor on the same device
        sinogram = torch.zeros((n_angles, num_detectors), dtype=image.dtype, device=device)

        # Prepare trigonometric tables on the correct device
        d_cos, d_sin = _trig_tables(angles, dtype=image.dtype, device=device)

        # Get Numba CUDA array views for kernel
        d_image = TorchCUDABridge.tensor_to_cuda_array(image)
        d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        grid, tpb = _grid_2d(n_angles, num_detectors)
        cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)

        pt_stream = torch.cuda.current_stream()
        numba_stream = _get_numba_external_stream_for(pt_stream)
        det_offset_v = _DTYPE(detector_offset / voxel_spacing)
        center_offset_x_v = _DTYPE(center_offset_x / voxel_spacing)
        center_offset_y_v = _DTYPE(center_offset_y / voxel_spacing)

        _parallel_2d_forward_kernel[grid, tpb, numba_stream](
            d_image, Nx, Ny, d_sino, n_angles, num_detectors,
            _DTYPE(detector_spacing), d_cos_arr, d_sin_arr, cx, cy, _DTYPE(voxel_spacing),
            det_offset_v, center_offset_x_v, center_offset_y_v
        )

        ctx.save_for_backward(angles)
        ctx.intermediate = (
            num_detectors,
            detector_spacing,
            Ny,
            Nx,
            voxel_spacing,
            detector_offset,
            center_offset_x,
            center_offset_y,
        )
        return sinogram
    
    @staticmethod
    def backward(ctx, grad_sinogram):
        angles, = ctx.saved_tensors
        (
            num_detectors,
            detector_spacing,
            Ny,
            Nx,
            voxel_spacing,
            detector_offset,
            center_offset_x,
            center_offset_y,
        ) = ctx.intermediate
        device = DeviceManager.get_device(grad_sinogram)
        grad_sinogram = DeviceManager.ensure_device(grad_sinogram, device)
        angles = DeviceManager.ensure_device(angles, device)

        grad_sinogram = grad_sinogram.to(dtype=torch.float32).contiguous()
        angles = angles.to(dtype=torch.float32).contiguous()

        n_angles = angles.shape[0]
        grad_image = torch.zeros((Ny, Nx), dtype=grad_sinogram.dtype, device=device)
        d_cos, d_sin = _trig_tables(angles, dtype=grad_sinogram.dtype, device=device)

        d_grad_sino = TorchCUDABridge.tensor_to_cuda_array(grad_sinogram)
        d_img_grad = TorchCUDABridge.tensor_to_cuda_array(grad_image)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        grid, tpb = _grid_2d(n_angles, num_detectors)
        cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)

        pt_stream = torch.cuda.current_stream()
        numba_stream = _get_numba_external_stream_for(pt_stream)
        det_offset_v = _DTYPE(detector_offset / voxel_spacing)
        center_offset_x_v = _DTYPE(center_offset_x / voxel_spacing)
        center_offset_y_v = _DTYPE(center_offset_y / voxel_spacing)

        _parallel_2d_backward_kernel[grid, tpb, numba_stream](
            d_grad_sino, n_angles, num_detectors,
            d_img_grad, Nx, Ny,
            _DTYPE(detector_spacing), d_cos_arr, d_sin_arr, cx, cy, _DTYPE(voxel_spacing),
            det_offset_v, center_offset_x_v, center_offset_y_v
        )

        return grad_image, None, None, None, None, None, None, None


class ParallelBackprojectorFunction(torch.autograd.Function):
    """
    Summary
    -------
    PyTorch autograd function for differentiable 2D parallel beam backprojection.
    
    Notes
    -----
    Provides a differentiable interface to the CUDA-accelerated Siddon ray-tracing
    method with a cell-constant image basis for parallel beam backprojection. The forward pass computes a 2D
    reconstruction from sinogram data using parallel beam backprojection, and the
    backward pass computes gradients via forward projection as the adjoint operation.
    Requires CUDA-capable hardware and consistent device placements.
    
    
    Examples
    --------
    >>> import torch
    >>> from diffct.differentiable import ParallelBackprojectorFunction
    >>>
    >>> sinogram = torch.randn(180, 128, device='cuda', requires_grad=True)
    >>> angles = torch.linspace(0, torch.pi, 180, device='cuda')
    >>> recon = ParallelBackprojectorFunction.apply(sinogram, angles, 1.0, 128, 128)
    >>> loss = recon.sum()
    >>> loss.backward()
    >>> print(sinogram.grad.shape)  # (180, 128)
    """
    @staticmethod
    def forward(
        ctx,
        sinogram,
        angles,
        detector_spacing=1.0,
        H=128,
        W=128,
        voxel_spacing=1.0,
        detector_offset=0.0,
        center_offset_x=0.0,
        center_offset_y=0.0,
    ):
        """Compute the 2D parallel beam backprojection (adjoint Radon
        transform) of a sinogram using CUDA acceleration.

        Parameters
        ----------
        sinogram : torch.Tensor
            2D input sinogram tensor of shape (num_angles, num_detectors), must be on a CUDA device and of type float32.
        angles : torch.Tensor
            1D tensor of projection angles in radians, shape (num_angles,), must be on the same CUDA device as `sinogram`.
        detector_spacing : float, optional
            Physical spacing between detector elements (default: 1.0).
        H : int, optional
            Height of the output reconstruction image (default: 128).
        W : int, optional
            Width of the output reconstruction image (default: 128).
        voxel_spacing : float, optional
            Physical size of one voxel (in same units as detector_spacing, default: 1.0).

        Returns
        -------
        reco : torch.Tensor
            2D tensor of shape (H, W) containing the reconstructed image on the same device as `sinogram`.

        Notes
        -----
        - All input tensors must be on the same CUDA device.
        - The operation is fully differentiable and supports autograd.
        - Uses the adjoint of cell-constant Siddon ray tracing.

        Examples
        --------
        >>> sinogram = torch.randn(180, 128, device='cuda', requires_grad=True)
        >>> angles = torch.linspace(0, torch.pi, 180, device='cuda')
        >>> reco = ParallelBackprojectorFunction.apply(
        ...     sinogram, angles, 1.0, 128, 128
        ... )
        """
        device = DeviceManager.get_device(sinogram)
        sinogram = DeviceManager.ensure_device(sinogram, device)
        angles = DeviceManager.ensure_device(angles, device)

        # Ensure input is float32 for kernel compatibility
        sinogram = sinogram.to(dtype=torch.float32).contiguous()
        angles = angles.to(dtype=torch.float32).contiguous()

        n_ang, n_det = sinogram.shape
        Ny, Nx = H, W
    
        # Allocate output tensor on the same device
        reco = torch.zeros((Ny, Nx), dtype=sinogram.dtype, device=device)

        # Prepare trigonometric tables on the correct device
        d_cos, d_sin = _trig_tables(angles, dtype=sinogram.dtype, device=device)

        # Get Numba CUDA array views for kernel
        d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
        d_reco = TorchCUDABridge.tensor_to_cuda_array(reco)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        grid, tpb = _grid_2d(n_ang, n_det)
        cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)

        pt_stream = torch.cuda.current_stream()
        numba_stream = _get_numba_external_stream_for(pt_stream)
        det_offset_v = _DTYPE(detector_offset / voxel_spacing)
        center_offset_x_v = _DTYPE(center_offset_x / voxel_spacing)
        center_offset_y_v = _DTYPE(center_offset_y / voxel_spacing)

        _parallel_2d_backward_kernel[grid, tpb, numba_stream](
            d_sino, n_ang, n_det, d_reco, Nx, Ny,
            _DTYPE(detector_spacing), d_cos_arr, d_sin_arr, cx, cy, _DTYPE(voxel_spacing),
            det_offset_v, center_offset_x_v, center_offset_y_v
        )

        ctx.save_for_backward(angles)
        ctx.intermediate = (
            H,
            W,
            detector_spacing,
            sinogram.shape[0],
            sinogram.shape[1],
            voxel_spacing,
            detector_offset,
            center_offset_x,
            center_offset_y,
        )
        return reco

    @staticmethod
    def backward(ctx, grad_output):
        angles, = ctx.saved_tensors
        (
            H,
            W,
            detector_spacing,
            n_ang,
            n_det,
            voxel_spacing,
            detector_offset,
            center_offset_x,
            center_offset_y,
        ) = ctx.intermediate
        device = DeviceManager.get_device(grad_output)
        grad_output = DeviceManager.ensure_device(grad_output, device)
        angles = DeviceManager.ensure_device(angles, device)

        grad_output = grad_output.to(dtype=torch.float32).contiguous()
        angles = angles.to(dtype=torch.float32).contiguous()

        Ny, Nx = grad_output.shape

        # Allocate output tensor on the same device
        grad_sino = torch.zeros((n_ang, n_det), dtype=grad_output.dtype, device=device)

        # Prepare trigonometric tables on the correct device
        d_cos, d_sin = _trig_tables(angles, dtype=grad_output.dtype, device=device)

        # Get Numba CUDA array views for kernel
        d_grad_out = TorchCUDABridge.tensor_to_cuda_array(grad_output)
        d_sino_grad = TorchCUDABridge.tensor_to_cuda_array(grad_sino)
        d_cos = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        grid, tpb = _grid_2d(n_ang, n_det)
        cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)

        pt_stream = torch.cuda.current_stream()
        numba_stream = _get_numba_external_stream_for(pt_stream)
        det_offset_v = _DTYPE(detector_offset / voxel_spacing)
        center_offset_x_v = _DTYPE(center_offset_x / voxel_spacing)
        center_offset_y_v = _DTYPE(center_offset_y / voxel_spacing)

        _parallel_2d_forward_kernel[grid, tpb, numba_stream](
            d_grad_out, Nx, Ny, d_sino_grad, n_ang, n_det,
            _DTYPE(detector_spacing), d_cos, d_sin, cx, cy, _DTYPE(voxel_spacing),
            det_offset_v, center_offset_x_v, center_offset_y_v
        )

        return grad_sino, None, None, None, None, None, None, None, None


class FanProjectorFunction(torch.autograd.Function):
    """
    Summary
    -------
    PyTorch autograd function for differentiable 2D fan beam forward projection.
    
    Notes
    -----
    Provides a differentiable interface to the CUDA-accelerated Siddon ray-tracing
    method with a cell-constant image basis for fan beam geometry, where rays diverge from a point
    X-ray source to a linear detector array. The forward pass computes sinograms
    using divergent beam geometry. The backward pass returns the **pure adjoint**
    ``P^T`` (Siddon scatter, no distance weighting) so that it is the correct
    gradient of ``y = P(x)`` with respect to ``x``. Analytical fan-beam FBP
    distance weighting is handled in ``fan_weighted_backproject``, not here.
    
    
    Examples
    --------
    >>> import torch
    >>> from diffct.differentiable import FanProjectorFunction
    >>>
    >>> image = torch.randn(256, 256, device='cuda', requires_grad=True)
    >>> angles = torch.linspace(0, 2 * torch.pi, 360, device='cuda')
    >>> sinogram = FanProjectorFunction.apply(image, angles, 512, 1.0, 1500.0, 1000.0)
    >>> loss = sinogram.sum()
    >>> loss.backward()
    >>> print(image.grad.shape)  # (256, 256)
    """
    @staticmethod
    def forward(
        ctx,
        image,
        angles,
        num_detectors,
        detector_spacing,
        sdd,
        sid,
        voxel_spacing=1.0,
        detector_offset=0.0,
        center_offset_x=0.0,
        center_offset_y=0.0,
        backend="siddon",
    ):
        """Compute the 2D fan beam forward projection of an image using CUDA
        acceleration.

        Parameters
        ----------
        backend : str, optional
            Forward projector backend. ``"siddon"`` (default) is the existing
            ray-driven cell-constant Siddon kernel. ``"sf"`` is a prototype separable-footprint (SF-TR)
            voxel-driven projector that projects each voxel's footprint as a
            trapezoid on the detector and integrates it closed-form over each
            detector cell; it is mass-conserving and closer to the physical
            finite-width-cell integral, with a matched voxel-driven gather
            adjoint for autograd.

        Parameters
        ----------
        image : torch.Tensor
            2D input image tensor of shape (H, W), must be on a CUDA device and of type float32.
        angles : torch.Tensor
            1D tensor of projection angles in radians, shape (num_angles,), must be on the same CUDA device as `image`.
        num_detectors : int
            Number of detector elements in the sinogram (columns).
        detector_spacing : float
            Physical spacing between detector elements.
        sdd : float
            Source-to-Detector Distance (SDD). The total distance from the X-ray
            source to the detector, passing through the isocenter.
        sid : float
            Source-to-Isocenter Distance (SID). The distance from the X-ray
            source to the center of rotation (isocenter).
        voxel_spacing : float, optional
            Physical size of one voxel (in same units as detector_spacing, sdd, sid, default: 1.0).

        Returns
        -------
        sinogram : torch.Tensor
            2D tensor of shape (num_angles, num_detectors) containing the fan beam sinogram on the same device as `image`.

        Notes
        -----
        - All input tensors must be on the same CUDA device.
        - The operation is fully differentiable and supports autograd.
        - Fan beam geometry uses divergent rays from a point source to the detector.
        - Uses cell-constant Siddon ray tracing.

        Examples
        --------
        >>> image = torch.randn(256, 256, device='cuda', requires_grad=True)
        >>> angles = torch.linspace(0, 2 * torch.pi, 360, device='cuda')
        >>> sinogram = FanProjectorFunction.apply(
        ...     image, angles, 512, 1.0, 1500.0, 1000.0
        ... )
        """
        device = DeviceManager.get_device(image)
        image = DeviceManager.ensure_device(image, device)
        angles = DeviceManager.ensure_device(angles, device)

        image = image.to(dtype=torch.float32).contiguous()
        angles = angles.to(dtype=torch.float32).contiguous()

        Ny, Nx = image.shape
        n_ang = angles.shape[0]

        sinogram = torch.zeros((n_ang, num_detectors), dtype=image.dtype, device=device)
        d_cos, d_sin = _trig_tables(angles, dtype=image.dtype, device=device)

        d_image = TorchCUDABridge.tensor_to_cuda_array(image)
        d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)

        pt_stream = torch.cuda.current_stream()
        numba_stream = _get_numba_external_stream_for(pt_stream)
        det_offset_v = _DTYPE(detector_offset / voxel_spacing)
        center_offset_x_v = _DTYPE(center_offset_x / voxel_spacing)
        center_offset_y_v = _DTYPE(center_offset_y / voxel_spacing)

        if backend == "siddon":
            grid, tpb = _grid_2d(n_ang, num_detectors)
            _fan_2d_forward_kernel[grid, tpb, numba_stream](
                d_image, Nx, Ny, d_sino, n_ang, num_detectors,
                _DTYPE(detector_spacing), d_cos_arr, d_sin_arr,
                _DTYPE(sdd), _DTYPE(sid), cx, cy, _DTYPE(voxel_spacing),
                det_offset_v, center_offset_x_v, center_offset_y_v
            )
        elif backend == "sf":
            grid, tpb = _grid_3d(n_ang, Ny, Nx)
            _fan_2d_sf_forward_kernel[grid, tpb, numba_stream](
                d_image, Nx, Ny, d_sino, n_ang, num_detectors,
                _DTYPE(detector_spacing), d_cos_arr, d_sin_arr,
                _DTYPE(sdd), _DTYPE(sid), cx, cy, _DTYPE(voxel_spacing),
                det_offset_v, center_offset_x_v, center_offset_y_v
            )
        else:
            raise ValueError(
                f"FanProjectorFunction: unknown backend {backend!r}; "
                "expected 'siddon' or 'sf'."
            )

        ctx.save_for_backward(angles)
        ctx.intermediate = (
            num_detectors,
            detector_spacing,
            Ny,
            Nx,
            sdd,
            sid,
            voxel_spacing,
            detector_offset,
            center_offset_x,
            center_offset_y,
            backend,
        )
        return sinogram

    @staticmethod
    def backward(ctx, grad_sinogram):
        angles, = ctx.saved_tensors
        (
            n_det,
            det_spacing,
            Ny,
            Nx,
            sdd,
            sid,
            voxel_spacing,
            detector_offset,
            center_offset_x,
            center_offset_y,
            backend,
        ) = ctx.intermediate
        device = DeviceManager.get_device(grad_sinogram)
        grad_sinogram = DeviceManager.ensure_device(grad_sinogram, device)
        angles = DeviceManager.ensure_device(angles, device)

        grad_sinogram = grad_sinogram.to(dtype=torch.float32).contiguous()
        angles = angles.to(dtype=torch.float32).contiguous()

        n_ang = angles.shape[0]
        grad_img = torch.zeros((Ny, Nx), dtype=grad_sinogram.dtype, device=device)
        d_cos, d_sin = _trig_tables(angles, dtype=grad_sinogram.dtype, device=device)

        d_grad_sino = TorchCUDABridge.tensor_to_cuda_array(grad_sinogram)
        d_img_grad = TorchCUDABridge.tensor_to_cuda_array(grad_img)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)

        pt_stream = torch.cuda.current_stream()
        numba_stream = _get_numba_external_stream_for(pt_stream)
        det_offset_v = _DTYPE(detector_offset / voxel_spacing)
        center_offset_x_v = _DTYPE(center_offset_x / voxel_spacing)
        center_offset_y_v = _DTYPE(center_offset_y / voxel_spacing)

        if backend == "siddon":
            # Pure adjoint P^T of the Siddon forward projector. FBP distance
            # weighting lives in fan_weighted_backproject, not the gradient path.
            grid, tpb = _grid_2d(n_ang, n_det)
            _fan_2d_backward_kernel[grid, tpb, numba_stream](
                d_grad_sino, n_ang, n_det, d_img_grad, Nx, Ny,
                _DTYPE(det_spacing), d_cos_arr, d_sin_arr,
                _DTYPE(sdd), _DTYPE(sid), cx, cy, _DTYPE(voxel_spacing),
                det_offset_v, center_offset_x_v, center_offset_y_v,
            )
        elif backend == "sf":
            # Pure adjoint of the SF-TR forward: voxel-driven gather that
            # rebuilds the same trapezoidal coefficients and accumulates
            # weight * raw * grad_sino into each image pixel.
            grid, tpb = _grid_2d(Nx, Ny)
            _fan_2d_sf_backward_kernel[grid, tpb, numba_stream](
                d_grad_sino, n_ang, n_det, d_img_grad, Nx, Ny,
                _DTYPE(det_spacing), d_cos_arr, d_sin_arr,
                _DTYPE(sdd), _DTYPE(sid), cx, cy, _DTYPE(voxel_spacing),
                det_offset_v, center_offset_x_v, center_offset_y_v,
            )
        else:
            raise ValueError(
                f"FanProjectorFunction.backward: unknown backend {backend!r}"
            )

        return grad_img, None, None, None, None, None, None, None, None, None, None


class FanBackprojectorFunction(torch.autograd.Function):
    """
    Summary
    -------
    PyTorch autograd function for differentiable 2D fan beam backprojection.
    
    Notes
    -----
    Provides a differentiable interface to the CUDA-accelerated Siddon ray-tracing
    method with a cell-constant image basis for fan beam backprojection. The forward pass runs
    the **pure adjoint** ``P^T`` of the fan forward projector (Siddon scatter
    with no distance weighting), and the backward pass
    computes gradients via forward projection. The forward/backward pair is
    therefore a self-consistent autograd adjoint pair. Analytical FBP distance
    weighting lives in ``fan_weighted_backproject`` and is *not* applied here.
    
    
    Examples
    --------
    >>> import torch
    >>> from diffct.differentiable import FanBackprojectorFunction
    >>>
    >>> sinogram = torch.randn(360, 512, device='cuda', requires_grad=True)
    >>> angles = torch.linspace(0, 2 * torch.pi, 360, device='cuda')
    >>> recon = FanBackprojectorFunction.apply(sinogram, angles, 1.0, 256, 256, 1500.0, 1000.0)
    >>> loss = recon.sum()
    >>> loss.backward()
    >>> print(sinogram.grad.shape)  # (360, 512)
    """
    @staticmethod
    def forward(
        ctx,
        sinogram,
        angles,
        detector_spacing,
        H,
        W,
        sdd,
        sid,
        voxel_spacing=1.0,
        detector_offset=0.0,
        center_offset_x=0.0,
        center_offset_y=0.0,
        backend="siddon",
    ):
        """Compute the 2D fan beam backprojection of a sinogram using CUDA
        acceleration.

        Parameters
        ----------
        backend : str, optional
            Adjoint backend, matches ``FanProjectorFunction``. ``"siddon"``
            (default) uses the ray-driven Siddon scatter kernel, ``"sf"`` uses
            the voxel-driven gather adjoint of the SF-TR projector.

        Parameters
        ----------
        sinogram : torch.Tensor
            2D input fan beam sinogram tensor of shape (num_angles, num_detectors), must be on a CUDA device and of type float32.
        angles : torch.Tensor
            1D tensor of projection angles in radians, shape (num_angles,), must be on the same CUDA device as `sinogram`.
        detector_spacing : float
            Physical spacing between detector elements.
        H : int
            Height of the output reconstruction image.
        W : int
            Width of the output reconstruction image.
        sdd : float
            Source-to-Detector Distance (SDD). The total distance from the X-ray
            source to the detector, passing through the isocenter.
        sid : float
            Source-to-Isocenter Distance (SID). The distance from the X-ray
            source to the center of rotation (isocenter).
        voxel_spacing : float, optional
            Physical size of one voxel (in same units as detector_spacing, sdd, sid, default: 1.0).

        Returns
        -------
        reco : torch.Tensor
            2D tensor of shape (H, W) containing the reconstructed image on the same device as `sinogram`.

        Notes
        -----
        - All input tensors must be on the same CUDA device.
        - The operation is fully differentiable and supports autograd.
        - Fan beam geometry uses divergent rays from a point source to the detector.
        - Uses the adjoint of cell-constant Siddon ray tracing.

        Examples
        --------
        >>> sinogram = torch.randn(360, 512, device='cuda', requires_grad=True)
        >>> angles = torch.linspace(0, 2*torch.pi, 360, device='cuda')
        >>> reco = FanBackprojectorFunction.apply(
        ...     sinogram, angles, 1.0, 256, 256, 1000.0, 500.0
        ... )
        """
        device = DeviceManager.get_device(sinogram)
        sinogram = DeviceManager.ensure_device(sinogram, device)
        angles = DeviceManager.ensure_device(angles, device)

        sinogram = sinogram.to(dtype=torch.float32).contiguous()
        angles = angles.to(dtype=torch.float32).contiguous()

        n_ang, n_det = sinogram.shape
        Ny, Nx = H, W
    
        reco = torch.zeros((Ny, Nx), dtype=sinogram.dtype, device=device)
        d_cos, d_sin = _trig_tables(angles, dtype=sinogram.dtype, device=device)

        d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
        d_reco = TorchCUDABridge.tensor_to_cuda_array(reco)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)

        pt_stream = torch.cuda.current_stream()
        numba_stream = _get_numba_external_stream_for(pt_stream)
        det_offset_v = _DTYPE(detector_offset / voxel_spacing)
        center_offset_x_v = _DTYPE(center_offset_x / voxel_spacing)
        center_offset_y_v = _DTYPE(center_offset_y / voxel_spacing)

        if backend == "siddon":
            # Pure adjoint. See the class docstring - this Function is
            # deliberately *not* the weighted FBP path.
            grid, tpb = _grid_2d(n_ang, n_det)
            _fan_2d_backward_kernel[grid, tpb, numba_stream](
                d_sino, n_ang, n_det, d_reco, Nx, Ny,
                _DTYPE(detector_spacing), d_cos_arr, d_sin_arr,
                _DTYPE(sdd), _DTYPE(sid), cx, cy, _DTYPE(voxel_spacing),
                det_offset_v, center_offset_x_v, center_offset_y_v,
            )
        elif backend == "sf":
            grid, tpb = _grid_2d(Nx, Ny)
            _fan_2d_sf_backward_kernel[grid, tpb, numba_stream](
                d_sino, n_ang, n_det, d_reco, Nx, Ny,
                _DTYPE(detector_spacing), d_cos_arr, d_sin_arr,
                _DTYPE(sdd), _DTYPE(sid), cx, cy, _DTYPE(voxel_spacing),
                det_offset_v, center_offset_x_v, center_offset_y_v,
            )
        else:
            raise ValueError(
                f"FanBackprojectorFunction: unknown backend {backend!r}; "
                "expected 'siddon' or 'sf'."
            )

        ctx.save_for_backward(angles)
        ctx.intermediate = (
            H,
            W,
            detector_spacing,
            n_ang,
            n_det,
            sdd,
            sid,
            voxel_spacing,
            detector_offset,
            center_offset_x,
            center_offset_y,
            backend,
        )
        return reco

    @staticmethod
    def backward(ctx, grad_output):
        angles, = ctx.saved_tensors
        (
            H,
            W,
            det_spacing,
            n_ang,
            n_det,
            sdd,
            sid,
            voxel_spacing,
            detector_offset,
            center_offset_x,
            center_offset_y,
            backend,
        ) = ctx.intermediate
        device = DeviceManager.get_device(grad_output)
        grad_output = DeviceManager.ensure_device(grad_output, device)
        angles = DeviceManager.ensure_device(angles, device)

        grad_output = grad_output.to(dtype=torch.float32).contiguous()
        angles = angles.to(dtype=torch.float32).contiguous()

        Ny, Nx = grad_output.shape

        grad_sino = torch.zeros((n_ang, n_det), dtype=grad_output.dtype, device=device)
        d_cos, d_sin = _trig_tables(angles, dtype=grad_output.dtype, device=device)

        d_grad_out = TorchCUDABridge.tensor_to_cuda_array(grad_output)
        d_sino_grad = TorchCUDABridge.tensor_to_cuda_array(grad_sino)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)

        pt_stream = torch.cuda.current_stream()
        numba_stream = _get_numba_external_stream_for(pt_stream)
        det_offset_v = _DTYPE(detector_offset / voxel_spacing)
        center_offset_x_v = _DTYPE(center_offset_x / voxel_spacing)
        center_offset_y_v = _DTYPE(center_offset_y / voxel_spacing)

        if backend == "siddon":
            grid, tpb = _grid_2d(n_ang, n_det)
            _fan_2d_forward_kernel[grid, tpb, numba_stream](
                d_grad_out, Nx, Ny, d_sino_grad, n_ang, n_det,
                _DTYPE(det_spacing), d_cos_arr, d_sin_arr,
                _DTYPE(sdd), _DTYPE(sid), cx, cy, _DTYPE(voxel_spacing),
                det_offset_v, center_offset_x_v, center_offset_y_v
            )
        elif backend == "sf":
            grid, tpb = _grid_3d(n_ang, Ny, Nx)
            _fan_2d_sf_forward_kernel[grid, tpb, numba_stream](
                d_grad_out, Nx, Ny, d_sino_grad, n_ang, n_det,
                _DTYPE(det_spacing), d_cos_arr, d_sin_arr,
                _DTYPE(sdd), _DTYPE(sid), cx, cy, _DTYPE(voxel_spacing),
                det_offset_v, center_offset_x_v, center_offset_y_v
            )
        else:
            raise ValueError(
                f"FanBackprojectorFunction.backward: unknown backend {backend!r}"
            )

        return grad_sino, None, None, None, None, None, None, None, None, None, None, None


class ConeProjectorFunction(torch.autograd.Function):
    """
    Summary
    -------
    PyTorch autograd function for differentiable 3D cone beam forward projection.
    
    Notes
    -----
    Provides a differentiable interface to the CUDA-accelerated Siddon ray-tracing
    method with a cell-constant voxel basis for 3D cone beam geometry. Rays emanate from a point
    X-ray source to a 2D detector array capturing volumetric projection data.
    The forward pass computes 3D projections. The backward pass returns the
    **pure adjoint** ``P^T`` of the forward projector - a Siddon ray-driven
    scatter with no distance weighting -
    so that ``P^T`` is the mathematically correct gradient of ``y = P(x)``
    with respect to ``x``. Analytical FDK distance weighting is handled
    separately in ``cone_weighted_backproject``, not here.
    
    
    Examples
    --------
    >>> import torch
    >>> from diffct.differentiable import ConeProjectorFunction
    >>>
    >>> volume = torch.randn(128, 128, 128, device='cuda', requires_grad=True)
    >>> angles = torch.linspace(0, 2 * torch.pi, 360, device='cuda')
    >>> projections = ConeProjectorFunction.apply(volume, angles, 256, 256, 1.0, 1.0, 1500.0, 1000.0)
    >>> loss = projections.sum()
    >>> loss.backward()
    >>> print(volume.grad.shape)  # (128, 128, 128)
    """
    @staticmethod
    def forward(
        ctx,
        volume,
        angles,
        det_u,
        det_v,
        du,
        dv,
        sdd,
        sid,
        voxel_spacing=1.0,
        detector_offset_u=0.0,
        detector_offset_v=0.0,
        center_offset_x=0.0,
        center_offset_y=0.0,
        center_offset_z=0.0,
        backend="siddon",
    ):
        """Compute the 3D cone beam forward projection of a volume using CUDA
        acceleration.

        Parameters
        ----------
        backend : str, optional
            Forward projector backend. ``"siddon"`` (default) is the existing
            ray-driven cell-constant Siddon kernel. ``"sf_tr"`` and ``"sf_tt"``
            are separable-footprint voxel-driven projectors (Long et al.,
            IEEE TMI 2010). SF-TR uses a trapezoidal transaxial footprint
            and a rectangular axial footprint evaluated at voxel-centre
            magnification; SF-TT uses trapezoids in BOTH directions, with
            the axial trapezoid built from ``U_near`` and ``U_far`` across
            the voxel to capture axial magnification variation at large
            cone angles. Both SF backends have matched voxel-driven gather
            adjoint kernels and support autograd.

        Parameters
        ----------
        volume : torch.Tensor
            3D input volume tensor of shape (D, H, W), must be on a CUDA device and of type float32.
        angles : torch.Tensor
            1D tensor of projection angles in radians, shape (num_views,), must be on the same CUDA device as `volume`.
        det_u : int
            Number of detector elements along the u-axis (width).
        det_v : int
            Number of detector elements along the v-axis (height).
        du : float
            Physical spacing between detector elements along the u-axis.
        dv : float
            Physical spacing between detector elements along the v-axis.
        sdd : float
            Source-to-Detector Distance (SDD). The total distance from the X-ray
            source to the detector, passing through the isocenter.
        sid : float
            Source-to-Isocenter Distance (SID). The distance from the X-ray
            source to the center of rotation (isocenter).
        voxel_spacing : float, optional
            Physical size of one voxel (in same units as du, dv, sdd, sid, default: 1.0).

        Returns
        -------
        sino : torch.Tensor
            3D tensor of shape (num_views, det_u, det_v) containing the cone beam projections on the same device as `volume`.

        Notes
        -----
        - All input tensors must be on the same CUDA device.
        - The operation is fully differentiable and supports autograd.
        - Cone beam geometry uses a point source and a 2D detector array.
        - Uses cell-constant Siddon ray tracing.

        Examples
        --------
        >>> volume = torch.randn(128, 128, 128, device='cuda', requires_grad=True)
        >>> angles = torch.linspace(0, 2*torch.pi, 360, device='cuda')
        >>> sino = ConeProjectorFunction.apply(
        ...     volume, angles, 256, 256, 1.0, 1.0, 1500.0, 1000.0
        ... )
        """
        device = DeviceManager.get_device(volume)
        volume = DeviceManager.ensure_device(volume, device)
        angles = DeviceManager.ensure_device(angles, device)

        volume = volume.to(dtype=torch.float32).contiguous()
        angles = angles.to(dtype=torch.float32).contiguous()

        D, H, W = volume.shape
        n_views = angles.shape[0]
        
        # Validate memory layout to prevent coordinate system inconsistencies
        _validate_3d_memory_layout(volume, expected_order='DHW')

        sino = torch.zeros((n_views, det_u, det_v), dtype=volume.dtype, device=device)
        d_cos, d_sin = _trig_tables(angles, dtype=volume.dtype, device=device)

        volume_perm = volume.permute(2, 1, 0).contiguous()
        d_vol = TorchCUDABridge.tensor_to_cuda_array(volume_perm)
        d_sino = TorchCUDABridge.tensor_to_cuda_array(sino)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        cx, cy, cz = _DTYPE(W * 0.5), _DTYPE(H * 0.5), _DTYPE(D * 0.5)

        pt_stream = torch.cuda.current_stream()
        numba_stream = _get_numba_external_stream_for(pt_stream)
        det_offset_u_v = _DTYPE(detector_offset_u / voxel_spacing)
        det_offset_v_v = _DTYPE(detector_offset_v / voxel_spacing)
        center_offset_x_v = _DTYPE(center_offset_x / voxel_spacing)
        center_offset_y_v = _DTYPE(center_offset_y / voxel_spacing)
        center_offset_z_v = _DTYPE(center_offset_z / voxel_spacing)

        if backend == "siddon":
            grid, tpb = _grid_3d(det_v, det_u, n_views)
            _cone_3d_forward_kernel[grid, tpb, numba_stream](
                d_vol, W, H, D, d_sino, n_views, det_u, det_v,
                _DTYPE(du), _DTYPE(dv), d_cos_arr, d_sin_arr,
                _DTYPE(sdd), _DTYPE(sid),
                cx, cy, cz, _DTYPE(voxel_spacing),
                det_offset_u_v, det_offset_v_v,
                center_offset_x_v, center_offset_y_v, center_offset_z_v
            )
        elif backend == "sf_tr":
            grid, tpb = _grid_3d(D, H, W, tpb=_TPB_SF_3D)
            _cone_3d_sf_tr_forward_kernel[grid, tpb, numba_stream](
                d_vol, W, H, D, d_sino, n_views, det_u, det_v,
                _DTYPE(du), _DTYPE(dv), d_cos_arr, d_sin_arr,
                _DTYPE(sdd), _DTYPE(sid),
                cx, cy, cz, _DTYPE(voxel_spacing),
                det_offset_u_v, det_offset_v_v,
                center_offset_x_v, center_offset_y_v, center_offset_z_v
            )
        elif backend == "sf_tt":
            grid, tpb = _grid_3d(D, H, W, tpb=_TPB_SF_3D)
            _cone_3d_sf_tt_forward_kernel[grid, tpb, numba_stream](
                d_vol, W, H, D, d_sino, n_views, det_u, det_v,
                _DTYPE(du), _DTYPE(dv), d_cos_arr, d_sin_arr,
                _DTYPE(sdd), _DTYPE(sid),
                cx, cy, cz, _DTYPE(voxel_spacing),
                det_offset_u_v, det_offset_v_v,
                center_offset_x_v, center_offset_y_v, center_offset_z_v
            )
        else:
            raise ValueError(
                f"ConeProjectorFunction: unknown backend {backend!r}; "
                "expected 'siddon', 'sf_tr', or 'sf_tt'."
            )

        ctx.save_for_backward(angles)
        ctx.intermediate = (
            D,
            H,
            W,
            det_u,
            det_v,
            du,
            dv,
            sdd,
            sid,
            voxel_spacing,
            detector_offset_u,
            detector_offset_v,
            center_offset_x,
            center_offset_y,
            center_offset_z,
            backend,
        )
        return sino

    @staticmethod
    def backward(ctx, grad_sinogram):
        angles, = ctx.saved_tensors
        (
            D,
            H,
            W,
            det_u,
            det_v,
            du,
            dv,
            sdd,
            sid,
            voxel_spacing,
            detector_offset_u,
            detector_offset_v,
            center_offset_x,
            center_offset_y,
            center_offset_z,
            backend,
        ) = ctx.intermediate
        device = DeviceManager.get_device(grad_sinogram)
        grad_sinogram = DeviceManager.ensure_device(grad_sinogram, device)
        angles = DeviceManager.ensure_device(angles, device)

        grad_sinogram = grad_sinogram.to(dtype=torch.float32).contiguous()
        angles = angles.to(dtype=torch.float32).contiguous()

        n_views = angles.shape[0]

        grad_vol_perm = torch.zeros((W, H, D), dtype=grad_sinogram.dtype, device=device)
        d_cos, d_sin = _trig_tables(angles, dtype=grad_sinogram.dtype, device=device)

        d_grad_sino = TorchCUDABridge.tensor_to_cuda_array(grad_sinogram)
        d_vol_grad = TorchCUDABridge.tensor_to_cuda_array(grad_vol_perm)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        cx, cy, cz = _DTYPE(W * 0.5), _DTYPE(H * 0.5), _DTYPE(D * 0.5)

        pt_stream = torch.cuda.current_stream()
        numba_stream = _get_numba_external_stream_for(pt_stream)
        det_offset_u_v = _DTYPE(detector_offset_u / voxel_spacing)
        det_offset_v_v = _DTYPE(detector_offset_v / voxel_spacing)
        center_offset_x_v = _DTYPE(center_offset_x / voxel_spacing)
        center_offset_y_v = _DTYPE(center_offset_y / voxel_spacing)
        center_offset_z_v = _DTYPE(center_offset_z / voxel_spacing)

        if backend == "siddon":
            # Pure adjoint P^T of the Siddon forward projector.
            grid, tpb = _grid_3d(det_v, det_u, n_views)
            _cone_3d_backward_kernel[grid, tpb, numba_stream](
                d_grad_sino, n_views, det_u, det_v, d_vol_grad, W, H, D,
                _DTYPE(du), _DTYPE(dv), d_cos_arr, d_sin_arr,
                _DTYPE(sdd), _DTYPE(sid), cx, cy, cz, _DTYPE(voxel_spacing),
                det_offset_u_v, det_offset_v_v,
                center_offset_x_v, center_offset_y_v, center_offset_z_v,
            )
        elif backend == "sf_tr":
            grid, tpb = _grid_3d(D, H, W, tpb=_TPB_SF_3D)
            _cone_3d_sf_tr_backward_kernel[grid, tpb, numba_stream](
                d_grad_sino, n_views, det_u, det_v, d_vol_grad, W, H, D,
                _DTYPE(du), _DTYPE(dv), d_cos_arr, d_sin_arr,
                _DTYPE(sdd), _DTYPE(sid), cx, cy, cz, _DTYPE(voxel_spacing),
                det_offset_u_v, det_offset_v_v,
                center_offset_x_v, center_offset_y_v, center_offset_z_v,
            )
        elif backend == "sf_tt":
            grid, tpb = _grid_3d(D, H, W, tpb=_TPB_SF_3D)
            _cone_3d_sf_tt_backward_kernel[grid, tpb, numba_stream](
                d_grad_sino, n_views, det_u, det_v, d_vol_grad, W, H, D,
                _DTYPE(du), _DTYPE(dv), d_cos_arr, d_sin_arr,
                _DTYPE(sdd), _DTYPE(sid), cx, cy, cz, _DTYPE(voxel_spacing),
                det_offset_u_v, det_offset_v_v,
                center_offset_x_v, center_offset_y_v, center_offset_z_v,
            )
        else:
            raise ValueError(
                f"ConeProjectorFunction.backward: unknown backend {backend!r}"
            )

        grad_vol = grad_vol_perm.permute(2, 1, 0).contiguous()
        return grad_vol, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class ConeBackprojectorFunction(torch.autograd.Function):
    """
    Summary
    -------
    PyTorch autograd function for differentiable 3D cone beam backprojection.

    Notes
    -----
    Provides a differentiable interface to the CUDA-accelerated Siddon ray-tracing
    method with a cell-constant voxel basis for 3D cone beam backprojection. The forward pass
    runs the **pure adjoint** ``P^T`` of the cone forward projector: a Siddon
    ray-driven scatter with no distance
    weighting. The backward pass computes gradients via 3D cone beam forward
    projection, which is exactly the adjoint of this pure ``P^T`` - so the
    forward/backward pair is self-consistent for autograd. Analytical FDK
    distance weighting belongs in ``cone_weighted_backproject`` and is *not*
    applied here. Requires CUDA-capable hardware and consistent device
    placements.
    
    This operation may be memory- and computationally-intensive due to 3D geometry.
    Consider using gradient checkpointing, smaller volumes, or distributed computing
    for large-scale applications, and ensure sufficient GPU memory is available.


    Examples
    --------
    >>> import torch
    >>> from diffct.differentiable import ConeBackprojectorFunction
    >>>
    >>> projections = torch.randn(360, 256, 256, device='cuda', requires_grad=True)
    >>> angles = torch.linspace(0, 2 * torch.pi, 360, device='cuda')
    >>> D, H, W = 128, 128, 128
    >>> du, dv = 1.0, 1.0
    >>> sdd, sid = 1500.0, 1000.0
    >>> backprojector = ConeBackprojectorFunction.apply
    >>> volume = backprojector(projections, angles, D, H, W, du, dv, sdd, sid)
    >>> loss = volume.sum()
    >>> loss.backward()
    >>> print(f"Projection gradient shape: {projections.grad.shape}")  # (360, 256, 256)
    """
    @staticmethod
    def forward(
        ctx,
        sinogram,
        angles,
        D,
        H,
        W,
        du,
        dv,
        sdd,
        sid,
        voxel_spacing=1.0,
        detector_offset_u=0.0,
        detector_offset_v=0.0,
        center_offset_x=0.0,
        center_offset_y=0.0,
        center_offset_z=0.0,
        backend="siddon",
    ):
        """Compute the 3D cone beam backprojection of a projection sinogram
        using CUDA acceleration.

        Parameters
        ----------
        backend : str, optional
            Adjoint backend, matches ``ConeProjectorFunction``. ``"siddon"``
            (default), ``"sf_tr"``, or ``"sf_tt"``. See
            ``ConeProjectorFunction`` for details.

        Parameters
        ----------
        sinogram : torch.Tensor
            3D input cone beam projection tensor of shape (num_views, det_u, det_v), must be on a CUDA device and of type float32.
        angles : torch.Tensor
            1D tensor of projection angles in radians, shape (num_views,), must be on the same CUDA device as `sinogram`.
        D : int
            Depth (z-dimension) of the output reconstruction volume.
        H : int
            Height (y-dimension) of the output reconstruction volume.
        W : int
            Width (x-dimension) of the output reconstruction volume.
        du : float
            Physical spacing between detector elements along the u-axis.
        dv : float
            Physical spacing between detector elements along the v-axis.
        sdd : float
            Source-to-Detector Distance (SDD). The total distance from the X-ray
            source to the detector, passing through the isocenter.
        sid : float
            Source-to-Isocenter Distance (SID). The distance from the X-ray
            source to the center of rotation (isocenter).
        voxel_spacing : float, optional
            Physical size of one voxel (in same units as du, dv, sdd, sid, default: 1.0).

        Returns
        -------
        vol : torch.Tensor
            3D tensor of shape (D, H, W) containing the reconstructed volume on the same device as `sinogram`.

        Notes
        -----
        - All input tensors must be on the same CUDA device.
        - The operation is fully differentiable and supports autograd.
        - Cone beam geometry uses a point source and a 2D detector array.
        - Uses the adjoint of cell-constant Siddon ray tracing.

        Examples
        --------
        >>> projections = torch.randn(360, 256, 256, device='cuda', requires_grad=True)
        >>> angles = torch.linspace(0, 2*torch.pi, 360, device='cuda')
        >>> vol = ConeBackprojectorFunction.apply(
        ...     projections, angles, 128, 128, 128, 1.0, 1.0, 1500.0, 1000.0
        ... )
        """
        device = DeviceManager.get_device(sinogram)
        sinogram = DeviceManager.ensure_device(sinogram, device)
        angles = DeviceManager.ensure_device(angles, device)

        sinogram = sinogram.to(dtype=torch.float32).contiguous()
        angles = angles.to(dtype=torch.float32).contiguous()

        n_views, n_u, n_v = sinogram.shape
        
        # Validate memory layout to prevent coordinate system inconsistencies
        _validate_3d_memory_layout(sinogram, expected_order='VHW')

        vol_perm = torch.zeros((W, H, D), dtype=sinogram.dtype, device=device)
        d_cos, d_sin = _trig_tables(angles, dtype=sinogram.dtype, device=device)

        d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
        d_reco = TorchCUDABridge.tensor_to_cuda_array(vol_perm)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        cx, cy, cz = _DTYPE(W * 0.5), _DTYPE(H * 0.5), _DTYPE(D * 0.5)

        pt_stream = torch.cuda.current_stream()
        numba_stream = _get_numba_external_stream_for(pt_stream)
        det_offset_u_v = _DTYPE(detector_offset_u / voxel_spacing)
        det_offset_v_v = _DTYPE(detector_offset_v / voxel_spacing)
        center_offset_x_v = _DTYPE(center_offset_x / voxel_spacing)
        center_offset_y_v = _DTYPE(center_offset_y / voxel_spacing)
        center_offset_z_v = _DTYPE(center_offset_z / voxel_spacing)

        if backend == "siddon":
            # Pure adjoint of the Siddon forward projector. See the class
            # docstring for why this Function deliberately does *not*
            # apply FDK distance weighting.
            grid, tpb = _grid_3d(n_v, n_u, n_views)
            _cone_3d_backward_kernel[grid, tpb, numba_stream](
                d_sino, n_views, n_u, n_v, d_reco, W, H, D,
                _DTYPE(du), _DTYPE(dv), d_cos_arr, d_sin_arr,
                _DTYPE(sdd), _DTYPE(sid), cx, cy, cz, _DTYPE(voxel_spacing),
                det_offset_u_v, det_offset_v_v,
                center_offset_x_v, center_offset_y_v, center_offset_z_v,
            )
        elif backend == "sf_tr":
            grid, tpb = _grid_3d(D, H, W, tpb=_TPB_SF_3D)
            _cone_3d_sf_tr_backward_kernel[grid, tpb, numba_stream](
                d_sino, n_views, n_u, n_v, d_reco, W, H, D,
                _DTYPE(du), _DTYPE(dv), d_cos_arr, d_sin_arr,
                _DTYPE(sdd), _DTYPE(sid), cx, cy, cz, _DTYPE(voxel_spacing),
                det_offset_u_v, det_offset_v_v,
                center_offset_x_v, center_offset_y_v, center_offset_z_v,
            )
        elif backend == "sf_tt":
            grid, tpb = _grid_3d(D, H, W, tpb=_TPB_SF_3D)
            _cone_3d_sf_tt_backward_kernel[grid, tpb, numba_stream](
                d_sino, n_views, n_u, n_v, d_reco, W, H, D,
                _DTYPE(du), _DTYPE(dv), d_cos_arr, d_sin_arr,
                _DTYPE(sdd), _DTYPE(sid), cx, cy, cz, _DTYPE(voxel_spacing),
                det_offset_u_v, det_offset_v_v,
                center_offset_x_v, center_offset_y_v, center_offset_z_v,
            )
        else:
            raise ValueError(
                f"ConeBackprojectorFunction: unknown backend {backend!r}; "
                "expected 'siddon', 'sf_tr', or 'sf_tt'."
            )

        ctx.save_for_backward(angles)
        ctx.intermediate = (
            D,
            H,
            W,
            n_u,
            n_v,
            du,
            dv,
            sdd,
            sid,
            voxel_spacing,
            detector_offset_u,
            detector_offset_v,
            center_offset_x,
            center_offset_y,
            center_offset_z,
            backend,
        )
        vol = vol_perm.permute(2, 1, 0).contiguous()
        return vol

    @staticmethod
    def backward(ctx, grad_output):
        angles, = ctx.saved_tensors
        (
            D,
            H,
            W,
            n_u,
            n_v,
            du,
            dv,
            sdd,
            sid,
            voxel_spacing,
            detector_offset_u,
            detector_offset_v,
            center_offset_x,
            center_offset_y,
            center_offset_z,
            backend,
        ) = ctx.intermediate
        device = DeviceManager.get_device(grad_output)
        grad_output = DeviceManager.ensure_device(grad_output, device)
        angles = DeviceManager.ensure_device(angles, device)

        grad_output = grad_output.to(dtype=torch.float32).contiguous()
        angles = angles.to(dtype=torch.float32).contiguous()

        n_views = angles.shape[0]

        grad_sino = torch.zeros((n_views, n_u, n_v), dtype=grad_output.dtype, device=device)
        d_cos, d_sin = _trig_tables(angles, dtype=grad_output.dtype, device=device)

        grad_output_perm = grad_output.permute(2, 1, 0).contiguous()
        d_grad_out = TorchCUDABridge.tensor_to_cuda_array(grad_output_perm)
        d_sino_grad = TorchCUDABridge.tensor_to_cuda_array(grad_sino)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        cx, cy, cz = _DTYPE(W * 0.5), _DTYPE(H * 0.5), _DTYPE(D * 0.5)

        pt_stream = torch.cuda.current_stream()
        numba_stream = _get_numba_external_stream_for(pt_stream)
        det_offset_u_v = _DTYPE(detector_offset_u / voxel_spacing)
        det_offset_v_v = _DTYPE(detector_offset_v / voxel_spacing)
        center_offset_x_v = _DTYPE(center_offset_x / voxel_spacing)
        center_offset_y_v = _DTYPE(center_offset_y / voxel_spacing)
        center_offset_z_v = _DTYPE(center_offset_z / voxel_spacing)

        if backend == "siddon":
            grid, tpb = _grid_3d(n_v, n_u, n_views)
            _cone_3d_forward_kernel[grid, tpb, numba_stream](
                d_grad_out, W, H, D, d_sino_grad, n_views, n_u, n_v,
                _DTYPE(du), _DTYPE(dv), d_cos_arr, d_sin_arr,
                _DTYPE(sdd), _DTYPE(sid), cx, cy, cz, _DTYPE(voxel_spacing),
                det_offset_u_v, det_offset_v_v,
                center_offset_x_v, center_offset_y_v, center_offset_z_v
            )
        elif backend == "sf_tr":
            grid, tpb = _grid_3d(D, H, W, tpb=_TPB_SF_3D)
            _cone_3d_sf_tr_forward_kernel[grid, tpb, numba_stream](
                d_grad_out, W, H, D, d_sino_grad, n_views, n_u, n_v,
                _DTYPE(du), _DTYPE(dv), d_cos_arr, d_sin_arr,
                _DTYPE(sdd), _DTYPE(sid), cx, cy, cz, _DTYPE(voxel_spacing),
                det_offset_u_v, det_offset_v_v,
                center_offset_x_v, center_offset_y_v, center_offset_z_v
            )
        elif backend == "sf_tt":
            grid, tpb = _grid_3d(D, H, W, tpb=_TPB_SF_3D)
            _cone_3d_sf_tt_forward_kernel[grid, tpb, numba_stream](
                d_grad_out, W, H, D, d_sino_grad, n_views, n_u, n_v,
                _DTYPE(du), _DTYPE(dv), d_cos_arr, d_sin_arr,
                _DTYPE(sdd), _DTYPE(sid), cx, cy, cz, _DTYPE(voxel_spacing),
                det_offset_u_v, det_offset_v_v,
                center_offset_x_v, center_offset_y_v, center_offset_z_v
            )
        else:
            raise ValueError(
                f"ConeBackprojectorFunction.backward: unknown backend {backend!r}"
            )

        return grad_sino, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def parallel_weighted_backproject(
    sinogram,
    angles,
    detector_spacing,
    H,
    W,
    voxel_spacing=1.0,
    detector_offset=0.0,
    center_offset_x=0.0,
    center_offset_y=0.0,
):
    """Parallel-beam backprojection for analytical FBP pipelines.

    Dispatches to the dedicated parallel-beam FBP voxel-driven gather
    kernel (``_parallel_2d_fbp_backproject_kernel``). The shared
    Siddon-based parallel backward kernel continues to serve the
    autograd adjoint path used by ``ParallelProjectorFunction`` /
    ``ParallelBackprojectorFunction``; this function is only the
    analytical FBP path. There is no distance weighting in parallel
    geometry (no source, no magnification), so the kernel simply
    gathers one linearly-interpolated detector sample per view per
    pixel and sums. The analytical ``1/(2*pi)`` FBP constant is
    applied inside the wrapper so the returned image is already
    amplitude-calibrated - a unit-density disk reconstructs to
    amplitude 1.
    """
    if not sinogram.is_cuda:
        raise ValueError("sinogram must be on CUDA device")
    device = sinogram.device
    sinogram = sinogram.to(dtype=torch.float32).contiguous()
    angles = angles.to(dtype=torch.float32, device=device).contiguous()

    n_ang, n_det = sinogram.shape
    Ny, Nx = H, W
    reco = torch.zeros((Ny, Nx), dtype=sinogram.dtype, device=device)
    d_cos, d_sin = _trig_tables(angles, dtype=sinogram.dtype, device=device)

    d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
    d_reco = TorchCUDABridge.tensor_to_cuda_array(reco)
    d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
    d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

    grid, tpb = _grid_2d(Nx, Ny)
    cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)
    det_offset_v = _DTYPE(detector_offset / voxel_spacing)
    center_offset_x_v = _DTYPE(center_offset_x / voxel_spacing)
    center_offset_y_v = _DTYPE(center_offset_y / voxel_spacing)

    pt_stream = torch.cuda.current_stream()
    numba_stream = _get_numba_external_stream_for(pt_stream)
    _parallel_2d_fbp_backproject_kernel[grid, tpb, numba_stream](
        d_sino, n_ang, n_det, d_reco, Nx, Ny,
        _DTYPE(detector_spacing), d_cos_arr, d_sin_arr,
        cx, cy, _DTYPE(voxel_spacing),
        det_offset_v, center_offset_x_v, center_offset_y_v,
    )
    # Parallel-beam analytical FBP scale: only the ``1/(2*pi)``
    # Fourier convention factor is needed (no SDD/SID magnification
    # because parallel beam has no source).
    scale = 1.0 / (2.0 * math.pi)
    reco.mul_(scale)
    return reco


def fan_weighted_backproject(
    sinogram,
    angles,
    detector_spacing,
    H,
    W,
    sdd,
    sid,
    voxel_spacing=1.0,
    detector_offset=0.0,
    center_offset_x=0.0,
    center_offset_y=0.0,
    backend="siddon",
):
    """Fan-beam weighted backprojection for analytical FBP pipelines.

    Dispatches to one of two gather kernels based on ``backend``:

    - ``"siddon"`` (default): the bilinear voxel-driven fan FBP gather
      ``_fan_2d_fbp_backproject_kernel`` — fastest, bilinearly samples
      the filtered sinogram at each pixel's projected u-coordinate.
    - ``"sf"``: the separable-footprint fan FBP gather
      ``_fan_2d_sf_fbp_backproject_kernel`` — integrates the filtered
      sinogram over each pixel's trapezoidal footprint. At nominal
      voxel size (``detector_spacing * sid / sdd``) this gives nearly
      the same MTF as the bilinear gather; at sub-nominal voxel sizes
      it gives measurably higher spatial resolution, and at supra-
      nominal it gives higher SNR (see LEAP's SF vs VD analysis).

    Both paths apply the ``(SID / U)^2`` FBP weight per view inside
    the kernel and the ``sdd / (2 * pi * sid)`` analytical FBP scale
    in Python, so amplitude calibration is preserved across backends.
    The Siddon-based fan autograd adjoint
    (``_fan_2d_backward_kernel``) and the SF-matched adjoint
    (``_fan_2d_sf_backward_kernel``) stay unchanged — they continue
    to serve ``FanProjectorFunction`` / ``FanBackprojectorFunction``
    and are not touched by this analytical wrapper.
    """
    if backend not in ("siddon", "sf"):
        raise ValueError(
            f"backend must be 'siddon' or 'sf', got {backend!r}"
        )
    if not sinogram.is_cuda:
        raise ValueError("sinogram must be on CUDA device")
    device = sinogram.device
    sinogram = sinogram.to(dtype=torch.float32).contiguous()
    angles = angles.to(dtype=torch.float32, device=device).contiguous()

    n_ang, n_det = sinogram.shape
    Ny, Nx = H, W
    reco = torch.zeros((Ny, Nx), dtype=sinogram.dtype, device=device)
    d_cos, d_sin = _trig_tables(angles, dtype=sinogram.dtype, device=device)

    d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
    d_reco = TorchCUDABridge.tensor_to_cuda_array(reco)
    d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
    d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

    grid, tpb = _grid_2d(Nx, Ny)
    cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)
    det_offset_v = _DTYPE(detector_offset / voxel_spacing)
    center_offset_x_v = _DTYPE(center_offset_x / voxel_spacing)
    center_offset_y_v = _DTYPE(center_offset_y / voxel_spacing)

    pt_stream = torch.cuda.current_stream()
    numba_stream = _get_numba_external_stream_for(pt_stream)
    if backend == "siddon":
        _fan_2d_fbp_backproject_kernel[grid, tpb, numba_stream](
            d_sino, n_ang, n_det, d_reco, Nx, Ny,
            _DTYPE(detector_spacing), d_cos_arr, d_sin_arr,
            _DTYPE(sdd), _DTYPE(sid), cx, cy, _DTYPE(voxel_spacing),
            det_offset_v, center_offset_x_v, center_offset_y_v,
        )
        # Classical FBP constant ``sdd / (2*pi*sid)`` for the VD
        # gather (which applies ``(sid/U)^2`` per voxel per view).
        scale = float(sdd) / (2.0 * math.pi * float(sid))
    else:  # backend == "sf"
        _fan_2d_sf_fbp_backproject_kernel[grid, tpb, numba_stream](
            d_sino, n_ang, n_det, d_reco, Nx, Ny,
            _DTYPE(detector_spacing), d_cos_arr, d_sin_arr,
            _DTYPE(sdd), _DTYPE(sid), cx, cy, _DTYPE(voxel_spacing),
            det_offset_v, center_offset_x_v, center_offset_y_v,
        )
        # LEAP chord-weighted form: the kernel produces
        # ``bp_weight * chord * raw_in_voxel_len`` which carries an
        # implicit ``1/det_spacing_v`` factor (since ``raw`` is a
        # closed-form trapezoid integral in voxel-index length). At
        # nominal voxel size (``det_spacing_v = 1``) the classical
        # factor ``sdd/(2*pi*sid)`` divided by the magnification
        # ``sdd/sid`` reduces to ``1/(2*pi)``; for general voxel /
        # detector ratios the scale is
        # ``det_spacing_v / (2*pi) = (det_spacing/voxel_spacing)/(2*pi)``
        # so sub- and supra-nominal voxel grids still reconstruct to
        # unit density without manual re-scaling.
        scale = float(detector_spacing) / (float(voxel_spacing) * 2.0 * math.pi)
    reco.mul_(scale)
    return reco


def cone_weighted_backproject(
    sinogram,
    angles,
    D,
    H,
    W,
    du,
    dv,
    sdd,
    sid,
    voxel_spacing=1.0,
    detector_offset_u=0.0,
    detector_offset_v=0.0,
    center_offset_x=0.0,
    center_offset_y=0.0,
    center_offset_z=0.0,
    backend="siddon",
):
    """Cone-beam weighted backprojection for analytical FDK pipelines.

    Dispatches to one of three gather kernels based on ``backend``:

    - ``"siddon"`` (default): the bilinear voxel-driven FDK gather
      ``_cone_3d_fdk_backproject_kernel`` — fastest, bilinearly samples
      the filtered sinogram at each voxel's projected ``(u, v)``.
    - ``"sf_tr"``: the separable-footprint FDK gather
      ``_cone_3d_sf_tr_fdk_backproject_kernel`` — integrates the filtered
      sinogram over each voxel's transaxial trapezoidal footprint and
      axial rectangular footprint at the voxel-centre magnification.
    - ``"sf_tt"``: the separable-footprint FDK gather
      ``_cone_3d_sf_tt_fdk_backproject_kernel`` — same transaxial
      trapezoid as ``"sf_tr"`` but the axial footprint is also a
      trapezoid built from four z-corner projections, which more
      faithfully captures the axial magnification variation inside
      one voxel at large cone angles.

    At nominal voxel size (``du * sid / sdd``) the three backends give
    nearly identical MTFs; at sub-nominal voxels the SF variants give
    measurably higher spatial resolution, and at supra-nominal voxels
    they give higher SNR (see LEAP's SF vs VD analysis). All three
    VD applies the classical ``(SID / U)^2`` FDK weight per view inside
    the kernel and the ``sdd / (2 * pi * sid)`` analytical FDK scale in
    Python. The SF paths use the LEAP chord-weighted form inside the
    kernels plus the matching SF scale below, so amplitude calibration is
    preserved across backends.
    The Siddon-based cone autograd adjoint
    (``_cone_3d_backward_kernel``) and the SF-matched adjoints
    (``_cone_3d_sf_tr_backward_kernel`` /
    ``_cone_3d_sf_tt_backward_kernel``) stay unchanged — they
    continue to serve ``ConeProjectorFunction`` /
    ``ConeBackprojectorFunction`` and are not touched by this wrapper.
    """
    if backend not in ("siddon", "sf_tr", "sf_tt"):
        raise ValueError(
            f"backend must be 'siddon', 'sf_tr', or 'sf_tt', got {backend!r}"
        )
    if not sinogram.is_cuda:
        raise ValueError("sinogram must be on CUDA device")
    device = sinogram.device
    sinogram = sinogram.to(dtype=torch.float32).contiguous()
    angles = angles.to(dtype=torch.float32, device=device).contiguous()

    n_views, n_u, n_v = sinogram.shape
    _validate_3d_memory_layout(sinogram, expected_order='VHW')

    # Keep the same WHD permuted buffer layout as the shared kernels use, so
    # the final ``permute(2, 1, 0)`` returns a (D, H, W) contiguous result.
    vol_perm = torch.zeros((W, H, D), dtype=sinogram.dtype, device=device)
    d_cos, d_sin = _trig_tables(angles, dtype=sinogram.dtype, device=device)

    d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
    d_reco = TorchCUDABridge.tensor_to_cuda_array(vol_perm)
    d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
    d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

    # Launch grid is (D, H, W) to match the kernel's ``iz, iy, ix = cuda.grid(3)``
    # unpack order. Numba puts the first grid dimension on ``threadIdx.x``
    # (the warp-adjacent axis), so by putting D first we make ``iz``
    # warp-adjacent, which matches the innermost stride-1 axis of the WHD
    # output buffer - coalesced writes. See the memory-access note in the
    # kernel docstring. The SF variants are register-heavier so they
    # use the smaller ``_TPB_SF_3D`` block shape.
    if backend == "siddon":
        grid, tpb = _grid_3d(D, H, W)
    else:
        grid, tpb = _grid_3d(D, H, W, tpb=_TPB_SF_3D)
    cx, cy, cz = _DTYPE(W * 0.5), _DTYPE(H * 0.5), _DTYPE(D * 0.5)
    det_offset_u_v = _DTYPE(detector_offset_u / voxel_spacing)
    det_offset_v_v = _DTYPE(detector_offset_v / voxel_spacing)
    center_offset_x_v = _DTYPE(center_offset_x / voxel_spacing)
    center_offset_y_v = _DTYPE(center_offset_y / voxel_spacing)
    center_offset_z_v = _DTYPE(center_offset_z / voxel_spacing)

    pt_stream = torch.cuda.current_stream()
    numba_stream = _get_numba_external_stream_for(pt_stream)
    if backend == "siddon":
        _cone_3d_fdk_backproject_kernel[grid, tpb, numba_stream](
            d_sino, n_views, n_u, n_v, d_reco, W, H, D,
            _DTYPE(du), _DTYPE(dv), d_cos_arr, d_sin_arr,
            _DTYPE(sdd), _DTYPE(sid), cx, cy, cz, _DTYPE(voxel_spacing),
            det_offset_u_v, det_offset_v_v,
            center_offset_x_v, center_offset_y_v, center_offset_z_v,
        )
    elif backend == "sf_tr":
        _cone_3d_sf_tr_fdk_backproject_kernel[grid, tpb, numba_stream](
            d_sino, n_views, n_u, n_v, d_reco, W, H, D,
            _DTYPE(du), _DTYPE(dv), d_cos_arr, d_sin_arr,
            _DTYPE(sdd), _DTYPE(sid), cx, cy, cz, _DTYPE(voxel_spacing),
            det_offset_u_v, det_offset_v_v,
            center_offset_x_v, center_offset_y_v, center_offset_z_v,
        )
    else:  # backend == "sf_tt"
        _cone_3d_sf_tt_fdk_backproject_kernel[grid, tpb, numba_stream](
            d_sino, n_views, n_u, n_v, d_reco, W, H, D,
            _DTYPE(du), _DTYPE(dv), d_cos_arr, d_sin_arr,
            _DTYPE(sdd), _DTYPE(sid), cx, cy, cz, _DTYPE(voxel_spacing),
            det_offset_u_v, det_offset_v_v,
            center_offset_x_v, center_offset_y_v, center_offset_z_v,
        )
    # Final analytical FDK scale factor.
    # - VD (siddon) path: classical ``sdd / (2*pi*sid)``. The kernel
    #   applies ``(sid/U)^2`` per voxel and 1/(2*pi) comes from the
    #   ramp-filter radian-frequency convention.
    # - SF (sf_tr / sf_tt) path: the LEAP chord-weighted form replaces
    #   ``(sid/U)^2`` with ``chord_u * sqrt(1+(v/sdd)^2)``. The kernel
    #   output carries an implicit ``1/(du_v*dv_v)`` factor from the
    #   voxel-index-length ``raw`` integrals, and at the isocenter the
    #   chord formula differs from classical FDK by an ``mag^2``
    #   factor. Combining everything:
    #       scale = (du*dv*sid) / (2*pi*sdd*voxel^2)
    #   which reduces to ``sid/(2*pi*sdd)`` at nominal (du=dv=voxel)
    #   and automatically renormalises sub- and supra-nominal voxel
    #   grids so they reconstruct to unit density.
    if backend == "siddon":
        scale = float(sdd) / (2.0 * math.pi * float(sid))
    else:
        v_sq = float(voxel_spacing) * float(voxel_spacing)
        scale = float(du) * float(dv) * float(sid) / (
            2.0 * math.pi * float(sdd) * v_sq
        )
    vol = vol_perm.permute(2, 1, 0).contiguous()
    vol.mul_(scale)
    return vol
