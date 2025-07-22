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
# CUDA fastmath optimization: enables aggressive floating-point optimizations
# Trades numerical precision for performance in ray-tracing calculations
# Safe for CT reconstruction where slight precision loss is acceptable for speed gains
_FASTMATH_DECORATOR = cuda.jit(cache=True, fastmath=True)
_INF                = _DTYPE(np.inf)
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
        if hasattr(tensor, "to"):
            return tensor if tensor.device == device else tensor.to(device)
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

    @staticmethod
    def cuda_array_to_tensor(cuda_array, tensor_template):
        """Convert a Numba CUDA array to a PyTorch tensor.

        Wrap a Numba CUDA DeviceNDArray as a PyTorch tensor with matching device
        and dtype from a template tensor, sharing underlying memory.

        Parameters
        ----------
        cuda_array : numba.cuda.cudadrv.devicearray.DeviceNDArray
            Numba CUDA array to wrap.
        tensor_template : torch.Tensor
            Template tensor specifying device and dtype.

        Returns
        -------
        torch.Tensor
            PyTorch tensor sharing data with the CUDA array on the template's
            device and dtype.

        Examples
        --------
        >>> arr = cuda.device_array((10,), dtype=np.float32)
        >>> t = torch.zeros(10, device='cuda')
        >>> new_t = TorchCUDABridge.cuda_array_to_tensor(arr, t)
        """
        return torch.as_tensor(cuda_array, device=tensor_template.device, dtype=tensor_template.dtype)

# === GPU-aware Trigonometric Table Generation ===
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
        cos = torch.cos(angles).to(dtype=dtype)
        sin = torch.sin(angles).to(dtype=dtype)
        return cos.to(device), sin.to(device)
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
        # Convert input angles to a CPU torch tensor
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
    if len(tensor.shape) != 3:
        raise ValueError(f"Expected 3D tensor, got {len(tensor.shape)}D")
    
    # Check if tensor is contiguous to avoid memory duplication
    if not tensor.is_contiguous():
        raise ValueError(
            "Input tensor must be contiguous. Call .contiguous() before passing to "
            "cone beam functions to avoid memory duplication and ensure correct results."
        )
    
    # Validate expected memory order based on stride patterns
    strides = tensor.stride()
    
    # Map expected orders to stride patterns
    order_mapping = {
        'DHW': (0, 1, 2),  # Depth, Height, Width
        'VHW': (0, 1, 2),  # Views, Height, Width (for sinograms)
        'WHD': (2, 1, 0),  # Width, Height, Depth (internal WHD format)
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
            actual_str = f"({tensor.shape[0]}, {tensor.shape[1]}, {tensor.shape[2]})"
            expected_str = "(Views, Height, Width)"
            fix_str = "ensure your sinogram has shape (num_views, det_v, det_u)"
        elif expected_order == 'DHW':
            actual_str = f"({tensor.shape[0]}, {tensor.shape[1]}, {tensor.shape[2]})"
            expected_str = "(Depth, Height, Width)"
            fix_str = "ensure your volume has shape (D, H, W)"
        else:
            actual_str = str(tuple(tensor.shape))
            expected_str = expected_order
            fix_str = "check tensor dimensions"
            
        raise ValueError(
            f"Memory layout mismatch: expected {expected_str} order, "
            f"but tensor has shape {actual_str}. Please {fix_str} and ensure "
            f"the tensor is contiguous (.contiguous()) before passing to the function."
        )


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
    det_spacing, d_cos, d_sin, cx, cy, voxel_spacing
):
    """Compute the 2D parallel beam forward projection.

    This CUDA kernel implements the Siddon-Joseph ray-tracing algorithm for
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
    The Siddon-Joseph algorithm provides accurate ray-volume intersection by:
      - Calculating ray-volume boundary intersections to define traversal limits.
      - Iterating through voxels along the ray path via parametric equations.
      - Determining bilinear interpolation weights for sub-voxel sampling.
      - Aggregating weighted voxel values based on ray segment lengths.
    """
    # CUDA THREAD ORGANIZATION: 2D grid maps directly to ray geometry
    # Each thread processes one ray defined by (projection_angle, detector_element) pair
    # Thread indexing: iang = projection angle index, idet = detector element index
    # Memory access pattern: Threads in same warp access consecutive detector elements (coalesced)
    iang, idet = cuda.grid(2)
    if iang >= n_ang or idet >= n_det:
        return

    # === RAY GEOMETRY SETUP ===
    # Extract projection angle and compute detector position
    cos_a = d_cos[iang]  # Precomputed cosine of projection angle
    sin_a = d_sin[iang]  # Precomputed sine of projection angle
    # Normalize all physical distances to voxel units
    u     = (idet - n_det * 0.5) * det_spacing / voxel_spacing  # Detector coordinate in voxel units

    # Define ray direction and starting point for parallel beam geometry
    # Ray direction is perpendicular to detector array (cos_a, sin_a)
    # Ray starting point is offset along detector by distance u in voxel units
    dir_x, dir_y = cos_a, sin_a
    pnt_x, pnt_y = u * -sin_a, u * cos_a

    # === RAY-VOLUME INTERSECTION CALCULATION ===
    # Compute parametric intersection points with volume boundaries using ray equation r(t) = pnt + t*dir
    # Volume extends from [-cx, cx] x [-cy, cy] in voxel coordinate system
    # Mathematical basis: For ray r(t) = origin + t*direction, solve r(t) = boundary for parameter t
    t_min, t_max = -_INF, _INF  # Initialize ray parameter range to unbounded
    
    # X-direction boundary intersections
    # Handle non-parallel rays: compute intersection parameters with left (-cx) and right (+cx) boundaries
    if abs(dir_x) > _EPSILON:  # Ray not parallel to x-axis (avoid division by zero)
        tx1, tx2 = (-cx - pnt_x) / dir_x, (cx - pnt_x) / dir_x  # Left and right boundary intersections
        # Update valid parameter range: intersection of current range with x-boundary constraints
        # min/max operations ensure we get the entry/exit points correctly regardless of ray direction
        t_min, t_max = max(t_min, min(tx1, tx2)), min(t_max, max(tx1, tx2))  # Update valid parameter range
    elif pnt_x < -cx or pnt_x > cx:  # Ray parallel to x-axis but outside volume bounds
        # Edge case: ray never intersects volume if parallel and outside boundaries
        d_sino[iang, idet] = 0.0; return

    # Y-direction boundary intersections (identical logic to x-direction)
    # Handle non-parallel rays: compute intersection parameters with bottom (-cy) and top (+cy) boundaries
    if abs(dir_y) > _EPSILON:  # Ray not parallel to y-axis (avoid division by zero)
        ty1, ty2 = (-cy - pnt_y) / dir_y, (cy - pnt_y) / dir_y  # Bottom and top boundary intersections
        # Intersect y-boundary constraints with existing parameter range from x-boundaries
        t_min, t_max = max(t_min, min(ty1, ty2)), min(t_max, max(ty1, ty2))  # Intersect with x-range
    elif pnt_y < -cy or pnt_y > cy:  # Ray parallel to y-axis but outside volume bounds
        # Edge case: ray never intersects volume if parallel and outside boundaries
        d_sino[iang, idet] = 0.0; return

    # Boundary intersection validation: check if ray actually intersects the volume
    # If t_min >= t_max, the ray misses the volume entirely (no valid intersection interval)
    if t_min >= t_max:
        d_sino[iang, idet] = 0.0; return

    # === SIDDON-JOSEPH VOXEL TRAVERSAL INITIALIZATION ===
    accum = 0.0  # Accumulated projection value along ray
    t = t_min    # Current ray parameter (distance from ray start)
    
    # Convert ray entry point to voxel indices (image coordinate system)
    ix = int(math.floor(pnt_x + t * dir_x + cx))  # Current voxel x-index
    iy = int(math.floor(pnt_y + t * dir_y + cy))  # Current voxel y-index

    # Determine traversal direction and step sizes for each axis
    step_x, step_y = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1)  # Voxel stepping direction
    dt_x = abs(1.0 / dir_x) if abs(dir_x) > _EPSILON else _INF  # Parameter increment to cross one voxel in x
    dt_y = abs(1.0 / dir_y) if abs(dir_y) > _EPSILON else _INF  # Parameter increment to cross one voxel in y
    
    # Calculate parameter values for next voxel boundary crossings
    tx = ((ix + (step_x > 0)) - cx - pnt_x) / dir_x if abs(dir_x) > _EPSILON else _INF  # Next x-boundary crossing
    ty = ((iy + (step_y > 0)) - cy - pnt_y) / dir_y if abs(dir_y) > _EPSILON else _INF  # Next y-boundary crossing

    # === MAIN RAY TRAVERSAL LOOP ===
    # Step through voxels along ray path, accumulating weighted contributions
    while t < t_max:
        # Check if current voxel indices are within valid interpolation bounds
        if 0 <= ix < Nx and 0 <= iy < Ny:
            # Determine next voxel boundary crossing (minimum of x, y boundaries or ray exit)
            t_next = min(tx, ty, t_max)
            seg_len = t_next - t  # Length of ray segment within current voxel region
            
            if seg_len > _EPSILON:  # Only process segments with meaningful length (avoid numerical noise)
                # === BILINEAR INTERPOLATION SAMPLING ===
                # Sample volume at ray segment midpoint for accurate integration
                # Mathematical basis: Midpoint rule for numerical integration along ray segments
                mid_x = pnt_x + (t + seg_len * 0.5) * dir_x + cx  # Midpoint x-coordinate in image space
                mid_y = pnt_y + (t + seg_len * 0.5) * dir_y + cy  # Midpoint y-coordinate in image space
                
                # Convert continuous coordinates to discrete voxel indices and fractional weights
                # Floor operation gives base voxel index, fractional part gives interpolation weights
                ix0, iy0 = int(math.floor(mid_x)), int(math.floor(mid_y))  # Base voxel indices (bottom-left corner)
                dx, dy = mid_x - ix0, mid_y - iy0  # Fractional parts: distance from base voxel center [0,1]
                
                # Clamp indices to stay in-bounds during interpolation
                ix0 = max(0, min(ix0, Nx - 2))
                iy0 = max(0, min(iy0, Ny - 2))
                
                # === BILINEAR INTERPOLATION WEIGHT CALCULATION ===
                # Mathematical basis: Bilinear interpolation formula f(x,y) = Σ f(xi,yi) * wi(x,y)
                # where wi(x,y) are the bilinear basis functions for each corner voxel
                # Weights are products of 1D linear interpolation weights: (1-dx) or dx, (1-dy) or dy
                val = (
                    d_image[iy0,     ix0]     * (1 - dx) * (1 - dy) +
                    d_image[iy0,     ix0 + 1] * dx       * (1 - dy) +
                    d_image[iy0 + 1, ix0]     * (1 - dx) * dy       +
                    d_image[iy0 + 1, ix0 + 1] * dx       * dy
                )
                # Accumulate contribution weighted by ray segment length (discrete line integral approximation)
                # This implements the Radon transform: integral of f(x,y) along the ray path
                accum += val * seg_len
        
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
    det_spacing, d_cos, d_sin, cx, cy, voxel_spacing
):
    """Compute the 2D parallel beam backprojection.

    This CUDA kernel implements the Siddon-Joseph algorithm for 2D parallel
    beam backprojection.

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
    u     = (idet - n_det * 0.5) * det_spacing / voxel_spacing  # Detector coordinate in voxel units

    # Define ray direction and starting point for parallel beam geometry
    dir_x, dir_y = cos_a, sin_a
    pnt_x, pnt_y = u * -sin_a, u * cos_a

    # === RAY-VOLUME INTERSECTION CALCULATION (identical to forward) ===
    t_min, t_max = -_INF, _INF
    if abs(dir_x) > _EPSILON:
        tx1, tx2 = (-cx - pnt_x) / dir_x, (cx - pnt_x) / dir_x
        t_min, t_max = max(t_min, min(tx1, tx2)), min(t_max, max(tx1, tx2))
    elif pnt_x < -cx or pnt_x > cx: return

    if abs(dir_y) > _EPSILON:
        ty1, ty2 = (-cy - pnt_y) / dir_y, (cy - pnt_y) / dir_y
        t_min, t_max = max(t_min, min(ty1, ty2)), min(t_max, max(ty1, ty2))
    elif pnt_y < -cy or pnt_y > cy: return

    if t_min >= t_max: return

    # === SIDDON-JOSEPH TRAVERSAL INITIALIZATION ===
    t = t_min
    ix = int(math.floor(pnt_x + t * dir_x + cx))
    iy = int(math.floor(pnt_y + t * dir_y + cy))

    step_x, step_y = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1)
    dt_x = abs(1.0 / dir_x) if abs(dir_x) > _EPSILON else _INF
    dt_y = abs(1.0 / dir_y) if abs(dir_y) > _EPSILON else _INF
    tx = ((ix + (step_x > 0)) - cx - pnt_x) / dir_x if abs(dir_x) > _EPSILON else _INF
    ty = ((iy + (step_y > 0)) - cy - pnt_y) / dir_y if abs(dir_y) > _EPSILON else _INF

    # === BACKPROJECTION TRAVERSAL LOOP ===
    # Distribute sinogram value along ray path using bilinear interpolation
    while t < t_max:
        if 0 <= ix < Nx and 0 <= iy < Ny:
            t_next = min(tx, ty, t_max)
            seg_len = t_next - t
            if seg_len > _EPSILON:
                # Sample at ray segment midpoint (same as forward projection)
                mid_x = pnt_x + (t + seg_len * 0.5) * dir_x + cx
                mid_y = pnt_y + (t + seg_len * 0.5) * dir_y + cy
                ix0, iy0 = int(math.floor(mid_x)), int(math.floor(mid_y))
                dx, dy = mid_x - ix0, mid_y - iy0
                
                # Clamp indices to stay in-bounds during interpolation
                ix0 = max(0, min(ix0, Nx - 2))
                iy0 = max(0, min(iy0, Ny - 2))
                
                # === ATOMIC BACKPROJECTION WITH BILINEAR WEIGHTS ===
                # Distribute contribution weighted by segment length and interpolation weights
                # CUDA ATOMIC OPERATIONS: Essential for thread safety in backprojection
                # Multiple threads (rays) can write to the same voxel simultaneously, causing race conditions
                # Atomic add operations serialize these writes, ensuring correct accumulation of contributions
                # Performance impact: Atomic operations are slower than regular writes but necessary for correctness
                # Memory access pattern: Global memory atomics with potential bank conflicts, but unavoidable
                cval = val * seg_len  # Contribution value for this ray segment
                cuda.atomic.add(d_image, (iy0,     ix0),     cval * (1 - dx) * (1 - dy))
                cuda.atomic.add(d_image, (iy0,     ix0 + 1), cval * dx       * (1 - dy))
                cuda.atomic.add(d_image, (iy0 + 1, ix0),     cval * (1 - dx) * dy)
                cuda.atomic.add(d_image, (iy0 + 1, ix0 + 1), cval * dx       * dy)

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
    sdd, sid, cx, cy, voxel_spacing
):
    """Compute the 2D fan beam forward projection.

    This CUDA kernel implements the Siddon-Joseph algorithm for 2D fan beam
    forward projection.

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
    u     = (idet - n_det * 0.5) * det_spacing / voxel_spacing  # Detector coordinate in voxel units
    sid_v = sid / voxel_spacing  # Source-to-isocenter distance in voxel units
    sdd_v = sdd / voxel_spacing  # Source-to-detector distance in voxel units

    # Calculate source and detector positions for current projection angle
    # Source position: rotated by angle around isocenter at distance sid (SID)
    src_x = -sid_v * sin_a  # Source x-coordinate in voxel units
    src_y =  sid_v * cos_a  # Source y-coordinate in voxel units
    
    # Detector element position: IDD = SDD - SID (Isocenter-to-Detector Distance)
    idd = sdd_v - sid_v
    det_x = idd * sin_a + u * cos_a   # Detector x-coordinate in voxel units
    det_y = -idd * cos_a + u * sin_a  # Detector y-coordinate in voxel units

    # === RAY DIRECTION CALCULATION ===
    # Ray direction vector from source to detector element
    dir_x, dir_y = det_x - src_x, det_y - src_y
    length = math.sqrt(dir_x * dir_x + dir_y * dir_y)  # Ray length
    if length < _EPSILON:  # Degenerate ray case
        d_sino[iang, idet] = 0.0; return
    
    # Normalize ray direction vector for parametric traversal
    inv_len = 1.0 / length
    dir_x, dir_y = dir_x * inv_len, dir_y * inv_len

    # === RAY-VOLUME INTERSECTION CALCULATION ===
    # Compute intersection with volume boundaries using source position as ray origin
    t_min, t_max = -_INF, _INF
    if abs(dir_x) > _EPSILON:
        tx1, tx2 = (-cx - src_x) / dir_x, (cx - src_x) / dir_x  # Volume boundary intersections
        t_min, t_max = max(t_min, min(tx1, tx2)), min(t_max, max(tx1, tx2))
    elif src_x < -cx or src_x > cx:  # Source outside volume bounds
        d_sino[iang, idet] = 0.0; return

    if abs(dir_y) > _EPSILON:
        ty1, ty2 = (-cy - src_y) / dir_y, (cy - src_y) / dir_y
        t_min, t_max = max(t_min, min(ty1, ty2)), min(t_max, max(ty1, ty2))
    elif src_y < -cy or src_y > cy:
        d_sino[iang, idet] = 0.0; return

    if t_min >= t_max:  # No valid intersection
        d_sino[iang, idet] = 0.0; return

    # === SIDDON-JOSEPH TRAVERSAL (same algorithm as parallel beam) ===
    accum = 0.0  # Accumulated projection value
    t = t_min    # Current ray parameter
    
    # Convert ray entry point to voxel indices (using source as ray origin)
    ix = int(math.floor(src_x + t * dir_x + cx))
    iy = int(math.floor(src_y + t * dir_y + cy))

    # Traversal parameters (identical to parallel beam implementation)
    step_x, step_y = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1)
    dt_x = abs(1.0 / dir_x) if abs(dir_x) > _EPSILON else _INF
    dt_y = abs(1.0 / dir_y) if abs(dir_y) > _EPSILON else _INF
    tx = ((ix + (step_x > 0)) - cx - src_x) / dir_x if abs(dir_x) > _EPSILON else _INF
    ty = ((iy + (step_y > 0)) - cy - src_y) / dir_y if abs(dir_y) > _EPSILON else _INF

    # Main traversal loop with bilinear interpolation (identical to parallel beam)
    while t < t_max:
        if 0 <= ix < Nx and 0 <= iy < Ny:
            t_next = min(tx, ty, t_max)
            seg_len = t_next - t
            if seg_len > _EPSILON:
                # Sample at midpoint using source as ray origin
                mid_x = src_x + (t + seg_len * 0.5) * dir_x + cx
                mid_y = src_y + (t + seg_len * 0.5) * dir_y + cy
                ix0, iy0 = int(math.floor(mid_x)), int(math.floor(mid_y))
                dx, dy = mid_x - ix0, mid_y - iy0
                
                # Clamp indices to stay in-bounds during interpolation
                ix0 = max(0, min(ix0, Nx - 2))
                iy0 = max(0, min(iy0, Ny - 2))
                
                # Bilinear interpolation (identical to parallel beam)
                val = (
                    d_image[iy0,     ix0]     * (1 - dx) * (1 - dy) +
                    d_image[iy0,     ix0 + 1] * dx       * (1 - dy) +
                    d_image[iy0 + 1, ix0]     * (1 - dx) * dy       +
                    d_image[iy0 + 1, ix0 + 1] * dx       * dy
                )
                accum += val * seg_len
        
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
    sdd, sid, cx, cy, voxel_spacing
):
    """Compute the 2D fan beam backprojection.

    This CUDA kernel implements the Siddon-Joseph algorithm for 2D fan beam
    backprojection.

    Parameters
    ----------
    d_sino : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Input fan beam sinogram array on CUDA.
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
    As the adjoint to the fan beam forward projection, this operation
    distributes sinogram values back into the volume along divergent ray
    paths using atomic operations for thread-safe accumulation.
    """
    iang, idet = cuda.grid(2)
    if iang >= n_ang or idet >= n_det:
        return

    # === BACKPROJECTION VALUE AND GEOMETRY SETUP ===
    val   = d_sino[iang, idet]  # Sinogram value to backproject along this ray
    cos_a = d_cos[iang]         # Precomputed cosine of projection angle
    sin_a = d_sin[iang]         # Precomputed sine of projection angle
    # Normalize all physical distances to voxel units
    u     = (idet - n_det * 0.5) * det_spacing / voxel_spacing  # Detector coordinate in voxel units
    sid_v = sid / voxel_spacing  # Source-to-isocenter distance in voxel units
    sdd_v = sdd / voxel_spacing  # Source-to-detector distance in voxel units

    # Calculate source and detector positions for current projection angle
    # Source position: rotated by angle around isocenter at distance sid (SID)
    src_x = -sid_v * sin_a  # Source x-coordinate in voxel units
    src_y =  sid_v * cos_a  # Source y-coordinate in voxel units
    
    # Detector element position: IDD = SDD - SID (Isocenter-to-Detector Distance)
    idd = sdd_v - sid_v
    det_x = idd * sin_a + u * cos_a   # Detector x-coordinate in voxel units
    det_y = -idd * cos_a + u * sin_a  # Detector y-coordinate in voxel units

    # === RAY DIRECTION CALCULATION ===
    # Ray direction vector from source to detector element
    dir_x, dir_y = det_x - src_x, det_y - src_y
    length = math.sqrt(dir_x * dir_x + dir_y * dir_y)  # Ray length
    if length < _EPSILON: return  # Skip degenerate rays
    inv_len = 1.0 / length        # Normalization factor for ray direction
    dir_x, dir_y = dir_x * inv_len, dir_y * inv_len  # Normalized ray direction vector

    # === RAY-VOLUME INTERSECTION CALCULATION ===
    # Compute intersection with volume boundaries using source position as ray origin
    t_min, t_max = -_INF, _INF
    if abs(dir_x) > _EPSILON:
        tx1, tx2 = (-cx - src_x) / dir_x, (cx - src_x) / dir_x
        t_min, t_max = max(t_min, min(tx1, tx2)), min(t_max, max(tx1, tx2))
    elif src_x < -cx or src_x > cx: return

    if abs(dir_y) > _EPSILON:
        ty1, ty2 = (-cy - src_y) / dir_y, (cy - src_y) / dir_y
        t_min, t_max = max(t_min, min(ty1, ty2)), min(t_max, max(ty1, ty2))
    elif src_y < -cy or src_y > cy: return

    if t_min >= t_max: return

    # === SIDDON-JOSEPH TRAVERSAL INITIALIZATION ===
    t = t_min
    ix = int(math.floor(src_x + t * dir_x + cx))
    iy = int(math.floor(src_y + t * dir_y + cy))

    step_x, step_y = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1)
    dt_x = abs(1.0 / dir_x) if abs(dir_x) > _EPSILON else _INF
    dt_y = abs(1.0 / dir_y) if abs(dir_y) > _EPSILON else _INF
    tx = ((ix + (step_x > 0)) - cx - src_x) / dir_x if abs(dir_x) > _EPSILON else _INF
    ty = ((iy + (step_y > 0)) - cy - src_y) / dir_y if abs(dir_y) > _EPSILON else _INF

    # === FAN BEAM BACKPROJECTION TRAVERSAL LOOP ===
    # Distribute sinogram value along divergent ray path using bilinear interpolation
    while t < t_max:
        if 0 <= ix < Nx and 0 <= iy < Ny:
            t_next = min(tx, ty, t_max)
            seg_len = t_next - t
            if seg_len > _EPSILON:
                # Sample at ray segment midpoint using source as ray origin
                mid_x = src_x + (t + seg_len * 0.5) * dir_x + cx
                mid_y = src_y + (t + seg_len * 0.5) * dir_y + cy
                ix0, iy0 = int(math.floor(mid_x)), int(math.floor(mid_y))
                dx, dy = mid_x - ix0, mid_y - iy0
                
                # Clamp indices to stay in-bounds during interpolation
                ix0 = max(0, min(ix0, Nx - 2))
                iy0 = max(0, min(iy0, Ny - 2))
                
                # === ATOMIC BACKPROJECTION WITH BILINEAR WEIGHTS ===
                # Distribute contribution weighted by segment length and interpolation weights
                # CUDA ATOMIC OPERATIONS: Critical for fan beam backprojection thread safety
                # Fan beam rays converge at source, creating higher probability of voxel write conflicts
                # Atomic operations prevent race conditions when multiple divergent rays write to same voxel
                # Performance consideration: Fan beam geometry may have more atomic contention than parallel beam
                cval = val * seg_len  # Contribution value for this ray segment
                cuda.atomic.add(d_image, (iy0,     ix0),     cval * (1 - dx) * (1 - dy))
                cuda.atomic.add(d_image, (iy0,     ix0 + 1), cval * dx       * (1 - dy))
                cuda.atomic.add(d_image, (iy0 + 1, ix0),     cval * (1 - dx) * dy)
                cuda.atomic.add(d_image, (iy0 + 1, ix0 + 1), cval * dx       * dy)

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

# ------------------------------------------------------------------
# 3-D CONE BEAM KERNELS
# ------------------------------------------------------------------

@_FASTMATH_DECORATOR
def _cone_3d_forward_kernel(
    d_vol, Nx, Ny, Nz,
    d_sino, n_views, n_u, n_v,
    du, dv, d_cos, d_sin,
    sdd, sid, cx, cy, cz, voxel_spacing
):
    """Compute the 3D cone-beam forward projection.

    This CUDA kernel implements the Siddon-Joseph algorithm for 3D cone-beam
    forward projection.

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
    a 2D detector array and trilinear interpolation for accurate volumetric
    sampling.
    """
    iview, iu, iv = cuda.grid(3)
    if iview >= n_views or iu >= n_u or iv >= n_v:
        return

    # === 3D CONE BEAM GEOMETRY SETUP ===
    cos_a, sin_a = d_cos[iview], d_sin[iview]  # Projection angle trigonometry
    # Normalize all physical distances to voxel units
    u     = (iu - n_u * 0.5) * du / voxel_spacing  # Detector u-coordinate in voxel units
    v     = (iv - n_v * 0.5) * dv / voxel_spacing  # Detector v-coordinate in voxel units
    sid_v = sid / voxel_spacing  # Source-to-isocenter distance in voxel units
    sdd_v = sdd / voxel_spacing  # Source-to-detector distance in voxel units

    # Calculate 3D source and detector positions
    # Source rotates in xy-plane around isocenter, z-coordinate is zero
    src_x, src_y, src_z = -sid_v * sin_a, sid_v * cos_a, 0.0
    
    # Detector element position: IDD = SDD - SID (Isocenter-to-Detector Distance)
    # u-coordinate is in-plane offset, v-coordinate is vertical (z-direction)
    idd = sdd_v - sid_v
    det_x = idd * sin_a + u * cos_a   # In-plane x-coordinate in voxel units
    det_y = -idd * cos_a + u * sin_a  # In-plane y-coordinate in voxel units
    det_z = v                                           # Vertical z-coordinate in voxel units

    # === 3D RAY DIRECTION CALCULATION ===
    # Ray direction vector from source to detector element in 3D space
    dir_x, dir_y, dir_z = det_x - src_x, det_y - src_y, det_z - src_z
    length = math.sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z)  # 3D ray length
    if length < _EPSILON:  # Degenerate ray case
        d_sino[iview, iu, iv] = 0.0; return
    
    # Normalize 3D ray direction vector for parametric traversal
    inv_len = 1.0 / length
    dir_x, dir_y, dir_z = dir_x*inv_len, dir_y*inv_len, dir_z*inv_len

    # === 3D RAY-VOLUME INTERSECTION CALCULATION ===
    # Compute intersection with 3D volume boundaries using source position as ray origin
    t_min, t_max = -_INF, _INF
    
    # X-direction boundary intersections
    if abs(dir_x) > _EPSILON:
        tx1, tx2 = (-cx - src_x) / dir_x, (cx - src_x) / dir_x
        t_min, t_max = max(t_min, min(tx1, tx2)), min(t_max, max(tx1, tx2))
    elif src_x < -cx or src_x > cx:  # Source outside x-bounds
        d_sino[iview, iu, iv] = 0.0; return
    
    # Y-direction boundary intersections
    if abs(dir_y) > _EPSILON:
        ty1, ty2 = (-cy - src_y) / dir_y, (cy - src_y) / dir_y
        t_min, t_max = max(t_min, min(ty1, ty2)), min(t_max, max(ty1, ty2))
    elif src_y < -cy or src_y > cy:  # Source outside y-bounds
        d_sino[iview, iu, iv] = 0.0; return
    
    # Z-direction boundary intersections (extends 2D algorithm to 3D)
    if abs(dir_z) > _EPSILON:
        tz1, tz2 = (-cz - src_z) / dir_z, (cz - src_z) / dir_z
        t_min, t_max = max(t_min, min(tz1, tz2)), min(t_max, max(tz1, tz2))
    elif src_z < -cz or src_z > cz:  # Source outside z-bounds
        d_sino[iview, iu, iv] = 0.0; return

    if t_min >= t_max:  # No valid 3D intersection
        d_sino[iview, iu, iv] = 0.0; return

    # === 3D SIDDON-JOSEPH TRAVERSAL INITIALIZATION ===
    accum = 0.0  # Accumulated projection value
    t = t_min    # Current ray parameter
    
    # Convert 3D ray entry point to voxel indices
    ix = int(math.floor(src_x + t * dir_x + cx))  # Current voxel x-index
    iy = int(math.floor(src_y + t * dir_y + cy))  # Current voxel y-index
    iz = int(math.floor(src_z + t * dir_z + cz))  # Current voxel z-index

    # 3D traversal parameters (extends 2D algorithm)
    step_x, step_y, step_z = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1), (1 if dir_z >= 0 else -1)
    dt_x = abs(1.0 / dir_x) if abs(dir_x) > _EPSILON else _INF  # Parameter increment per x-voxel
    dt_y = abs(1.0 / dir_y) if abs(dir_y) > _EPSILON else _INF  # Parameter increment per y-voxel
    dt_z = abs(1.0 / dir_z) if abs(dir_z) > _EPSILON else _INF  # Parameter increment per z-voxel
    
    # Calculate parameter values for next 3D voxel boundary crossings
    tx = ((ix + (step_x > 0)) - cx - src_x) / dir_x if abs(dir_x) > _EPSILON else _INF
    ty = ((iy + (step_y > 0)) - cy - src_y) / dir_y if abs(dir_y) > _EPSILON else _INF
    tz = ((iz + (step_z > 0)) - cz - src_z) / dir_z if abs(dir_z) > _EPSILON else _INF

    # === 3D TRAVERSAL LOOP WITH TRILINEAR INTERPOLATION ===
    while t < t_max:
        # Check if current 3D voxel indices are within valid interpolation bounds
        if 0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz:
            # Determine next 3D voxel boundary crossing (minimum of x, y, z boundaries or ray exit)
            t_next = min(tx, ty, tz, t_max)
            seg_len = t_next - t
            if seg_len > _EPSILON:
                # === TRILINEAR INTERPOLATION SAMPLING ===
                # Sample 3D volume at ray segment midpoint for accurate integration
                # Mathematical basis: Midpoint rule for numerical integration along 3D ray segments
                mid_x = src_x + (t + seg_len * 0.5) * dir_x + cx  # Midpoint x-coordinate in volume space
                mid_y = src_y + (t + seg_len * 0.5) * dir_y + cy  # Midpoint y-coordinate in volume space
                mid_z = src_z + (t + seg_len * 0.5) * dir_z + cz  # Midpoint z-coordinate in volume space
                
                # Convert continuous 3D coordinates to discrete voxel indices and fractional weights
                # Floor operation gives base voxel index, fractional part gives interpolation weights
                ix0, iy0, iz0 = int(math.floor(mid_x)), int(math.floor(mid_y)), int(math.floor(mid_z))  # Base voxel indices (corner 0,0,0)
                dx, dy, dz = mid_x - ix0, mid_y - iy0, mid_z - iz0  # Fractional parts: distance from base voxel center [0,1]
                
                # Clamp indices to stay in-bounds during interpolation
                ix0 = max(0, min(ix0, Nx - 2))
                iy0 = max(0, min(iy0, Ny - 2))
                iz0 = max(0, min(iz0, Nz - 2))
                
                # === TRILINEAR INTERPOLATION WEIGHT CALCULATION ===
                # Mathematical basis: Trilinear interpolation formula f(x,y,z) = Σ f(xi,yi,zi) * wi(x,y,z)
                # where wi(x,y,z) are the trilinear basis functions for each corner voxel of the 3D cube
                # Weights are products of 1D linear interpolation weights: (1-dx) or dx, (1-dy) or dy, (1-dz) or dz
                # Each of the 8 cube corners gets a weight proportional to its distance from the sample point
                val = (
                    d_vol[ix0,     iy0,     iz0]     * (1-dx)*(1-dy)*(1-dz) +  # Corner (0,0,0): weight = product of distances from opposite faces
                    d_vol[ix0 + 1, iy0,     iz0]     * dx*(1-dy)*(1-dz) +     # Corner (1,0,0): weight = dx * (1-dy) * (1-dz)
                    d_vol[ix0,     iy0 + 1, iz0]     * (1-dx)*dy*(1-dz) +     # Corner (0,1,0): weight = (1-dx) * dy * (1-dz)
                    d_vol[ix0,     iy0,     iz0 + 1] * (1-dx)*(1-dy)*dz +     # Corner (0,0,1): weight = (1-dx) * (1-dy) * dz
                    d_vol[ix0 + 1, iy0 + 1, iz0]     * dx*dy*(1-dz) +         # Corner (1,1,0): weight = dx * dy * (1-dz)
                    d_vol[ix0 + 1, iy0,     iz0 + 1] * dx*(1-dy)*dz +         # Corner (1,0,1): weight = dx * (1-dy) * dz
                    d_vol[ix0,     iy0 + 1, iz0 + 1] * (1-dx)*dy*dz +         # Corner (0,1,1): weight = (1-dx) * dy * dz
                    d_vol[ix0 + 1, iy0 + 1, iz0 + 1] * dx*dy*dz               # Corner (1,1,1): weight = dx * dy * dz
                )
                # Accumulate contribution weighted by 3D ray segment length (discrete line integral approximation)
                # This implements the 3D Radon transform: integral of f(x,y,z) along the ray path
                accum += val * seg_len

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
    sdd, sid, cx, cy, cz, voxel_spacing
):
    """Compute the 3D cone-beam backprojection.

    This CUDA kernel implements the Siddon-Joseph algorithm for 3D cone-beam
    backprojection.

    Parameters
    ----------
    d_sino : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Input cone-beam sinogram array on CUDA.
    n_views : int
        Number of projection views.
    n_u : int
        Number of detector elements along the u-axis.
    n_v : int
        Number of detector elements along the v-axis.
    d_vol : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Output 3D volume gradient array on CUDA.
    Nx : int
        Number of voxels along the x-axis.
    Ny : int
        Number of voxels along the y-axis.
    Nz : int
        Number of voxels along the z-axis.
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
    As the adjoint to the cone-beam forward projection, this operation
    distributes sinogram values back into the 3D volume along ray paths using
    atomic operations for thread-safe accumulation.
    """
    iview, iu, iv = cuda.grid(3)
    if iview >= n_views or iu >= n_u or iv >= n_v:
        return

    # === 3D BACKPROJECTION VALUE AND GEOMETRY SETUP ===
    g = d_sino[iview, iu, iv]  # Sinogram value to backproject along this ray
    cos_a, sin_a = d_cos[iview], d_sin[iview]  # Projection angle trigonometry
    # Normalize all physical distances to voxel units
    u     = (iu - n_u * 0.5) * du / voxel_spacing  # Detector u-coordinate in voxel units
    v     = (iv - n_v * 0.5) * dv / voxel_spacing  # Detector v-coordinate in voxel units
    sid_v = sid / voxel_spacing  # Source-to-isocenter distance in voxel units
    sdd_v = sdd / voxel_spacing  # Source-to-detector distance in voxel units

    # Calculate 3D source and detector positions
    # Source rotates in xy-plane around isocenter, z-coordinate is zero
    src_x, src_y, src_z = -sid_v * sin_a, sid_v * cos_a, 0.0
    
    # Detector element position: IDD = SDD - SID (Isocenter-to-Detector Distance)
    # u-coordinate is in-plane offset, v-coordinate is vertical (z-direction)
    idd = sdd_v - sid_v
    det_x = idd * sin_a + u * cos_a   # In-plane x-coordinate in voxel units
    det_y = -idd * cos_a + u * sin_a  # In-plane y-coordinate in voxel units
    det_z = v                                           # Vertical z-coordinate in voxel units

    # === 3D RAY DIRECTION CALCULATION ===
    # Ray direction vector from source to detector element in 3D space
    dir_x, dir_y, dir_z = det_x - src_x, det_y - src_y, det_z - src_z
    length = math.sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z)  # 3D ray length
    if length < _EPSILON: return  # Skip degenerate rays
    inv_len = 1.0 / length        # Normalization factor for ray direction
    dir_x, dir_y, dir_z = dir_x*inv_len, dir_y*inv_len, dir_z*inv_len  # Normalized 3D ray direction vector

    # === 3D RAY-VOLUME INTERSECTION CALCULATION ===
    # Compute intersection with 3D volume boundaries using source position as ray origin
    t_min, t_max = -_INF, _INF
    
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

    # === 3D SIDDON-JOSEPH TRAVERSAL INITIALIZATION ===
    t = t_min
    ix = int(math.floor(src_x + t * dir_x + cx))  # Current voxel x-index
    iy = int(math.floor(src_y + t * dir_y + cy))  # Current voxel y-index
    iz = int(math.floor(src_z + t * dir_z + cz))  # Current voxel z-index

    # 3D traversal parameters (extends 2D algorithm)
    step_x, step_y, step_z = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1), (1 if dir_z >= 0 else -1)
    dt_x = abs(1.0 / dir_x) if abs(dir_x) > _EPSILON else _INF  # Parameter increment per x-voxel
    dt_y = abs(1.0 / dir_y) if abs(dir_y) > _EPSILON else _INF  # Parameter increment per y-voxel
    dt_z = abs(1.0 / dir_z) if abs(dir_z) > _EPSILON else _INF  # Parameter increment per z-voxel
    
    # Calculate parameter values for next 3D voxel boundary crossings
    tx = ((ix + (step_x > 0)) - cx - src_x) / dir_x if abs(dir_x) > _EPSILON else _INF
    ty = ((iy + (step_y > 0)) - cy - src_y) / dir_y if abs(dir_y) > _EPSILON else _INF
    tz = ((iz + (step_z > 0)) - cz - src_z) / dir_z if abs(dir_z) > _EPSILON else _INF

    # === 3D CONE BEAM BACKPROJECTION TRAVERSAL LOOP ===
    # Distribute sinogram value along divergent 3D ray path using trilinear interpolation
    while t < t_max:
        # Check if current 3D voxel indices are within valid interpolation bounds
        if 0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz:
            # Determine next 3D voxel boundary crossing (minimum of x, y, z boundaries or ray exit)
            t_next = min(tx, ty, tz, t_max)
            seg_len = t_next - t
            if seg_len > _EPSILON:
                # === TRILINEAR INTERPOLATION SAMPLING ===
                # Sample 3D volume at ray segment midpoint using source as ray origin
                mid_x = src_x + (t + seg_len * 0.5) * dir_x + cx  # Midpoint x-coordinate
                mid_y = src_y + (t + seg_len * 0.5) * dir_y + cy  # Midpoint y-coordinate
                mid_z = src_z + (t + seg_len * 0.5) * dir_z + cz  # Midpoint z-coordinate
                
                # Convert continuous 3D coordinates to voxel indices and interpolation weights
                ix0, iy0, iz0 = int(math.floor(mid_x)), int(math.floor(mid_y)), int(math.floor(mid_z))
                dx, dy, dz = mid_x - ix0, mid_y - iy0, mid_z - iz0  # Fractional parts for 3D weights
                
                # Clamp indices to stay in-bounds during interpolation
                ix0 = max(0, min(ix0, Nx - 2))
                iy0 = max(0, min(iy0, Ny - 2))
                iz0 = max(0, min(iz0, Nz - 2))
                
                # === ATOMIC BACKPROJECTION WITH TRILINEAR WEIGHTS ===
                # Distribute contribution weighted by segment length and interpolation weights
                # CUDA 3D ATOMIC OPERATIONS: Most complex atomic pattern in cone beam backprojection
                # 8 atomic writes per ray segment (one per cube corner) increases memory contention significantly
                # Cone beam geometry creates maximum ray convergence, highest probability of write conflicts
                # Performance impact: 3D atomics are most expensive due to volume of concurrent writes
                # Memory bandwidth: 8 atomic operations per interpolation point can saturate memory subsystem
                cval = g * seg_len  # Contribution value for this ray segment
                cuda.atomic.add(d_vol, (ix0,     iy0,     iz0),     cval * (1-dx)*(1-dy)*(1-dz))  # Corner (0,0,0) - atomic write
                cuda.atomic.add(d_vol, (ix0 + 1, iy0,     iz0),     cval * dx*(1-dy)*(1-dz))      # Corner (1,0,0) - atomic write
                cuda.atomic.add(d_vol, (ix0,     iy0 + 1, iz0),     cval * (1-dx)*dy*(1-dz))      # Corner (0,1,0) - atomic write
                cuda.atomic.add(d_vol, (ix0,     iy0,     iz0 + 1), cval * (1-dx)*(1-dy)*dz)      # Corner (0,0,1) - atomic write
                cuda.atomic.add(d_vol, (ix0 + 1, iy0 + 1, iz0),     cval * dx*dy*(1-dz))          # Corner (1,1,0) - atomic write
                cuda.atomic.add(d_vol, (ix0 + 1, iy0,     iz0 + 1), cval * dx*(1-dy)*dz)          # Corner (1,0,1) - atomic write
                cuda.atomic.add(d_vol, (ix0,     iy0 + 1, iz0 + 1), cval * (1-dx)*dy*dz)          # Corner (0,1,1) - atomic write
                cuda.atomic.add(d_vol, (ix0 + 1, iy0 + 1, iz0 + 1), cval * dx*dy*dz)              # Corner (1,1,1) - atomic write

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
    Provides a differentiable interface to the CUDA-accelerated Siddon-Joseph
    ray-tracing algorithm for parallel beam CT geometry. The forward pass computes
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
    def forward(ctx, image, angles, num_detectors, detector_spacing=1.0, voxel_spacing=1.0):
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
        - Uses the Siddon-Joseph algorithm for accurate ray tracing and bilinear interpolation.

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
        image = image.to(dtype=torch.float32)
        angles = angles.to(dtype=torch.float32)

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

        _parallel_2d_forward_kernel[grid, tpb](
            d_image, Nx, Ny, d_sino, n_angles, num_detectors,
            _DTYPE(detector_spacing), d_cos_arr, d_sin_arr, cx, cy, _DTYPE(voxel_spacing)
        )

        ctx.save_for_backward(angles)
        ctx.intermediate = (num_detectors, detector_spacing, Ny, Nx, voxel_spacing)
        return sinogram
    
    @staticmethod
    def backward(ctx, grad_sinogram):
        angles, = ctx.saved_tensors
        num_detectors, detector_spacing, Ny, Nx, voxel_spacing = ctx.intermediate
        device = DeviceManager.get_device(grad_sinogram)
        grad_sinogram = DeviceManager.ensure_device(grad_sinogram, device)
        angles = DeviceManager.ensure_device(angles, device)

        grad_sinogram = grad_sinogram.to(dtype=torch.float32)
        angles = angles.to(dtype=torch.float32)

        n_angles = angles.shape[0]
        grad_image = torch.zeros((Ny, Nx), dtype=grad_sinogram.dtype, device=device)
        d_cos, d_sin = _trig_tables(angles, dtype=grad_sinogram.dtype, device=device)

        d_grad_sino = TorchCUDABridge.tensor_to_cuda_array(grad_sinogram)
        d_img_grad = TorchCUDABridge.tensor_to_cuda_array(grad_image)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        grid, tpb = _grid_2d(n_angles, num_detectors)
        cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)

        _parallel_2d_backward_kernel[grid, tpb](
            d_grad_sino, n_angles, num_detectors,
            d_img_grad, Nx, Ny,
            _DTYPE(detector_spacing), d_cos_arr, d_sin_arr, cx, cy, _DTYPE(voxel_spacing)
        )

        return grad_image, None, None, None, None


class ParallelBackprojectorFunction(torch.autograd.Function):
    """
    Summary
    -------
    PyTorch autograd function for differentiable 2D parallel beam backprojection.
    
    Notes
    -----
    Provides a differentiable interface to the CUDA-accelerated Siddon-Joseph ray-tracing
    algorithm for parallel beam backprojection. The forward pass computes a 2D
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
    def forward(ctx, sinogram, angles, detector_spacing=1.0, H=128, W=128, voxel_spacing=1.0):
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
        - Uses the Siddon-Joseph algorithm for accurate ray tracing and bilinear interpolation.

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
        sinogram = sinogram.to(dtype=torch.float32)
        angles = angles.to(dtype=torch.float32)

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

        _parallel_2d_backward_kernel[grid, tpb](
            d_sino, n_ang, n_det, d_reco, Nx, Ny,
            _DTYPE(detector_spacing), d_cos_arr, d_sin_arr, cx, cy, _DTYPE(voxel_spacing)
        )

        ctx.save_for_backward(angles)
        ctx.intermediate = (H, W, detector_spacing, sinogram.shape[0], sinogram.shape[1], voxel_spacing)
        return reco

    @staticmethod
    def backward(ctx, grad_output):
        angles, = ctx.saved_tensors
        H, W, detector_spacing, n_ang, n_det, voxel_spacing = ctx.intermediate
        device = DeviceManager.get_device(grad_output)
        grad_output = DeviceManager.ensure_device(grad_output, device)
        angles = DeviceManager.ensure_device(angles, device)

        grad_output = grad_output.to(dtype=torch.float32)
        angles = angles.to(dtype=torch.float32)

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

        _parallel_2d_forward_kernel[grid, tpb](
            d_grad_out, Nx, Ny, d_sino_grad, n_ang, n_det,
            _DTYPE(detector_spacing), d_cos, d_sin, cx, cy, _DTYPE(voxel_spacing)
        )

        return grad_sino, None, None, None, None, None


class FanProjectorFunction(torch.autograd.Function):
    """
    Summary
    -------
    PyTorch autograd function for differentiable 2D fan beam forward projection.
    
    Notes
    -----
    Provides a differentiable interface to the CUDA-accelerated Siddon-Joseph
    ray-tracing algorithm for fan beam geometry, where rays diverge from a point
    X-ray source to a linear detector array. The forward pass computes sinograms
    using divergent beam geometry, and the backward pass computes gradients via
    adjoint backprojection.
    
    
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
    def forward(ctx, image, angles, num_detectors, detector_spacing, sdd, sid, voxel_spacing=1.0):
        """Compute the 2D fan beam forward projection of an image using CUDA
        acceleration.

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
        - Uses the Siddon-Joseph algorithm for accurate ray tracing and bilinear interpolation.

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

        image = image.to(dtype=torch.float32)
        angles = angles.to(dtype=torch.float32)

        Ny, Nx = image.shape
        n_ang = angles.shape[0]

        sinogram = torch.zeros((n_ang, num_detectors), dtype=image.dtype, device=device)
        d_cos, d_sin = _trig_tables(angles, dtype=image.dtype, device=device)

        d_image = TorchCUDABridge.tensor_to_cuda_array(image)
        d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        grid, tpb = _grid_2d(n_ang, num_detectors)
        cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)

        _fan_2d_forward_kernel[grid, tpb](
            d_image, Nx, Ny, d_sino, n_ang, num_detectors,
            _DTYPE(detector_spacing), d_cos_arr, d_sin_arr,
            _DTYPE(sdd), _DTYPE(sid), cx, cy, _DTYPE(voxel_spacing)
        )

        ctx.save_for_backward(angles)
        ctx.intermediate = (num_detectors, detector_spacing, Ny, Nx,
                            sdd, sid, voxel_spacing)
        return sinogram

    @staticmethod
    def backward(ctx, grad_sinogram):
        angles, = ctx.saved_tensors
        (n_det, det_spacing, Ny, Nx, sdd, sid, voxel_spacing) = ctx.intermediate
        device = DeviceManager.get_device(grad_sinogram)
        grad_sinogram = DeviceManager.ensure_device(grad_sinogram, device)
        angles = DeviceManager.ensure_device(angles, device)

        grad_sinogram = grad_sinogram.to(dtype=torch.float32)
        angles = angles.to(dtype=torch.float32)

        n_ang = angles.shape[0]
        grad_img = torch.zeros((Ny, Nx), dtype=grad_sinogram.dtype, device=device)
        d_cos, d_sin = _trig_tables(angles, dtype=grad_sinogram.dtype, device=device)

        d_grad_sino = TorchCUDABridge.tensor_to_cuda_array(grad_sinogram)
        d_img_grad = TorchCUDABridge.tensor_to_cuda_array(grad_img)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        grid, tpb = _grid_2d(n_ang, n_det)
        cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)

        _fan_2d_backward_kernel[grid, tpb](
            d_grad_sino, n_ang, n_det, d_img_grad, Nx, Ny,
            _DTYPE(det_spacing), d_cos_arr, d_sin_arr,
            _DTYPE(sdd), _DTYPE(sid), cx, cy, _DTYPE(voxel_spacing)
        )

        return grad_img, None, None, None, None, None, None


class FanBackprojectorFunction(torch.autograd.Function):
    """
    Summary
    -------
    PyTorch autograd function for differentiable 2D fan beam backprojection.
    
    Notes
    -----
    Provides a differentiable interface to the CUDA-accelerated Siddon-Joseph
    ray-tracing algorithm for fan beam backprojection. Implements the adjoint
    of the fan beam projection operator, distributing sinogram values back into
    the reconstruction volume along divergent ray paths. The forward pass
    computes reconstruction from sinogram data, and the backward pass computes
    gradients via forward projection.
    
    
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
    def forward(ctx, sinogram, angles, detector_spacing, H, W, sdd, sid, voxel_spacing=1.0):
        """Compute the 2D fan beam backprojection of a sinogram using CUDA
        acceleration.

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
        - Uses the Siddon-Joseph algorithm for accurate ray tracing and bilinear interpolation.

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

        sinogram = sinogram.to(dtype=torch.float32)
        angles = angles.to(dtype=torch.float32)

        n_ang, n_det = sinogram.shape
        Ny, Nx = H, W
    
        reco = torch.zeros((Ny, Nx), dtype=sinogram.dtype, device=device)
        d_cos, d_sin = _trig_tables(angles, dtype=sinogram.dtype, device=device)

        d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
        d_reco = TorchCUDABridge.tensor_to_cuda_array(reco)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        grid, tpb = _grid_2d(n_ang, n_det)
        cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)

        _fan_2d_backward_kernel[grid, tpb](
            d_sino, n_ang, n_det, d_reco, Nx, Ny,
            _DTYPE(detector_spacing), d_cos_arr, d_sin_arr,
            _DTYPE(sdd), _DTYPE(sid), cx, cy, _DTYPE(voxel_spacing)
        )

        ctx.save_for_backward(angles)
        ctx.intermediate = (H, W, detector_spacing, n_ang, n_det, sdd, sid, voxel_spacing)
        return reco

    @staticmethod
    def backward(ctx, grad_output):
        angles, = ctx.saved_tensors
        (H, W, det_spacing, n_ang, n_det, sdd, sid, voxel_spacing) = ctx.intermediate
        device = DeviceManager.get_device(grad_output)
        grad_output = DeviceManager.ensure_device(grad_output, device)
        angles = DeviceManager.ensure_device(angles, device)

        grad_output = grad_output.to(dtype=torch.float32)
        angles = angles.to(dtype=torch.float32)

        Ny, Nx = grad_output.shape

        grad_sino = torch.zeros((n_ang, n_det), dtype=grad_output.dtype, device=device)
        d_cos, d_sin = _trig_tables(angles, dtype=grad_output.dtype, device=device)

        d_grad_out = TorchCUDABridge.tensor_to_cuda_array(grad_output)
        d_sino_grad = TorchCUDABridge.tensor_to_cuda_array(grad_sino)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        grid, tpb = _grid_2d(n_ang, n_det)
        cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)

        _fan_2d_forward_kernel[grid, tpb](
            d_grad_out, Nx, Ny, d_sino_grad, n_ang, n_det,
            _DTYPE(det_spacing), d_cos_arr, d_sin_arr,
            _DTYPE(sdd), _DTYPE(sid), cx, cy, _DTYPE(voxel_spacing)
        )

        return grad_sino, None, None, None, None, None, None, None


class ConeProjectorFunction(torch.autograd.Function):
    """
    Summary
    -------
    PyTorch autograd function for differentiable 3D cone beam forward projection.
    
    Notes
    -----
    Provides a differentiable interface to the CUDA-accelerated Siddon-Joseph
    ray-tracing algorithm for 3D cone beam geometry. Rays emanate from a point
    X-ray source to a 2D detector array capturing volumetric projection data.
    The forward pass computes 3D projections, and the backward pass computes
    gradients via adjoint 3D backprojection. Requires significant GPU memory.
    
    
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
    def forward(ctx, volume, angles, det_u, det_v, du, dv, sdd, sid, voxel_spacing=1.0):
        """Compute the 3D cone beam forward projection of a volume using CUDA
        acceleration.

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
        - Uses the Siddon-Joseph algorithm for accurate 3D ray tracing and trilinear interpolation.

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

        volume = volume.to(dtype=torch.float32)
        angles = angles.to(dtype=torch.float32)

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

        grid, tpb = _grid_3d(n_views, det_u, det_v)
        cx, cy, cz = _DTYPE(W * 0.5), _DTYPE(H * 0.5), _DTYPE(D * 0.5)

        _cone_3d_forward_kernel[grid, tpb](
            d_vol, W, H, D, d_sino, n_views, det_u, det_v,
            _DTYPE(du), _DTYPE(dv), d_cos_arr, d_sin_arr,
            _DTYPE(sdd), _DTYPE(sid),
            cx, cy, cz, _DTYPE(voxel_spacing)
        )

        ctx.save_for_backward(angles)
        ctx.intermediate = (D, H, W, det_u, det_v, du, dv,
                            sdd, sid, voxel_spacing)
        return sino

    @staticmethod
    def backward(ctx, grad_sinogram):
        angles, = ctx.saved_tensors
        (D, H, W, det_u, det_v, du, dv,
         sdd, sid, voxel_spacing) = ctx.intermediate
        device = DeviceManager.get_device(grad_sinogram)
        grad_sinogram = DeviceManager.ensure_device(grad_sinogram, device)
        angles = DeviceManager.ensure_device(angles, device)

        grad_sinogram = grad_sinogram.to(dtype=torch.float32)
        angles = angles.to(dtype=torch.float32)

        n_views = angles.shape[0]

        grad_vol_perm = torch.zeros((W, H, D), dtype=grad_sinogram.dtype, device=device)
        d_cos, d_sin = _trig_tables(angles, dtype=grad_sinogram.dtype, device=device)

        d_grad_sino = TorchCUDABridge.tensor_to_cuda_array(grad_sinogram)
        d_vol_grad = TorchCUDABridge.tensor_to_cuda_array(grad_vol_perm)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        grid, tpb = _grid_3d(n_views, det_u, det_v)
        cx, cy, cz = _DTYPE(W * 0.5), _DTYPE(H * 0.5), _DTYPE(D * 0.5)

        _cone_3d_backward_kernel[grid, tpb](
            d_grad_sino, n_views, det_u, det_v, d_vol_grad, W, H, D,
            _DTYPE(du), _DTYPE(dv), d_cos_arr, d_sin_arr,
            _DTYPE(sdd), _DTYPE(sid), cx, cy, cz, _DTYPE(voxel_spacing)
        )

        grad_vol = grad_vol_perm.permute(2, 1, 0).contiguous()
        return grad_vol, None, None, None, None, None, None, None, None


class ConeBackprojectorFunction(torch.autograd.Function):
    """
    Summary
    -------
    PyTorch autograd function for differentiable 3D cone beam backprojection.

    Notes
    -----
    Provides a differentiable interface to the CUDA-accelerated Siddon-Joseph
    ray-tracing algorithm for 3D cone beam backprojection. The forward pass
    computes a 3D reconstruction from cone beam projection data using
    backprojection as the adjoint operation. The backward pass computes gradients
    via 3D cone beam forward projection. Requires CUDA-capable hardware and
    consistent device placements.
    
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
    def forward(ctx, sinogram, angles, D, H, W, du, dv, sdd, sid, voxel_spacing=1.0):
        """Compute the 3D cone beam backprojection of a projection sinogram
        using CUDA acceleration.

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
        - Uses the Siddon-Joseph algorithm for accurate 3D ray tracing and trilinear interpolation.

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

        sinogram = sinogram.to(dtype=torch.float32)
        angles = angles.to(dtype=torch.float32)

        n_views, n_u, n_v = sinogram.shape
        
        # Validate memory layout to prevent coordinate system inconsistencies
        _validate_3d_memory_layout(sinogram, expected_order='VHW')

        vol_perm = torch.zeros((W, H, D), dtype=sinogram.dtype, device=device)
        d_cos, d_sin = _trig_tables(angles, dtype=sinogram.dtype, device=device)

        d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
        d_reco = TorchCUDABridge.tensor_to_cuda_array(vol_perm)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        grid, tpb = _grid_3d(n_views, n_u, n_v)
        cx, cy, cz = _DTYPE(W * 0.5), _DTYPE(H * 0.5), _DTYPE(D * 0.5)

        _cone_3d_backward_kernel[grid, tpb](
            d_sino, n_views, n_u, n_v, d_reco, W, H, D,
            _DTYPE(du), _DTYPE(dv), d_cos_arr, d_sin_arr,
            _DTYPE(sdd), _DTYPE(sid), cx, cy, cz, _DTYPE(voxel_spacing)
        )

        ctx.save_for_backward(angles)
        ctx.intermediate = (D, H, W, n_u, n_v, du, dv,
                            sdd, sid, voxel_spacing)
        vol = vol_perm.permute(2, 1, 0).contiguous()
        return vol

    @staticmethod
    def backward(ctx, grad_output):
        angles, = ctx.saved_tensors
        (D, H, W, n_u, n_v, du, dv,
         sdd, sid, voxel_spacing) = ctx.intermediate
        device = DeviceManager.get_device(grad_output)
        grad_output = DeviceManager.ensure_device(grad_output, device)
        angles = DeviceManager.ensure_device(angles, device)

        grad_output = grad_output.to(dtype=torch.float32)
        angles = angles.to(dtype=torch.float32)

        n_views = angles.shape[0]

        grad_sino = torch.zeros((n_views, n_u, n_v), dtype=grad_output.dtype, device=device)
        d_cos, d_sin = _trig_tables(angles, dtype=grad_output.dtype, device=device)

        grad_output_perm = grad_output.permute(2, 1, 0).contiguous()
        d_grad_out = TorchCUDABridge.tensor_to_cuda_array(grad_output_perm)
        d_sino_grad = TorchCUDABridge.tensor_to_cuda_array(grad_sino)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        grid, tpb = _grid_3d(n_views, n_u, n_v)
        cx, cy, cz = _DTYPE(W * 0.5), _DTYPE(H * 0.5), _DTYPE(D * 0.5)

        _cone_3d_forward_kernel[grid, tpb](
            d_grad_out, W, H, D, d_sino_grad, n_views, n_u, n_v,
            _DTYPE(du), _DTYPE(dv), d_cos_arr, d_sin_arr,
            _DTYPE(sdd), _DTYPE(sid), cx, cy, cz, _DTYPE(voxel_spacing)
        )

        return grad_sino, None, None, None, None, None, None, None, None, None