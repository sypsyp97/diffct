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
_FASTMATH_DECORATOR = cuda.jit(fastmath=True)
_INF                = _DTYPE(np.inf)
_EPSILON            = _DTYPE(1e-6)
# === Device Management Utilities ===
class DeviceManager:
    @staticmethod
    def get_device(tensor):
        """Return the device of a PyTorch tensor."""
        return tensor.device if hasattr(tensor, "device") else torch.device("cpu")

    @staticmethod
    def ensure_device(tensor, device):
        """Move tensor to the specified device if not already there."""
        if hasattr(tensor, "to"):
            return tensor if tensor.device == device else tensor.to(device)
        return tensor

# === PyTorch-CUDA Bridge ===
class TorchCUDABridge:
    @staticmethod
    def tensor_to_cuda_array(tensor):
        """Convert PyTorch CUDA tensor to Numba CUDA array without CPU transfer.
        Always detaches the tensor to avoid autograd Variable issues."""
        if not tensor.is_cuda:
            raise ValueError("Tensor must be on CUDA device")
        return cuda.as_cuda_array(tensor.detach())

    @staticmethod
    def cuda_array_to_tensor(cuda_array, tensor_template):
        """Convert CUDA array to PyTorch tensor on the same device as tensor_template."""
        return torch.as_tensor(cuda_array, device=tensor_template.device, dtype=tensor_template.dtype)

# === GPU-aware Trigonometric Table Generation ===
def _trig_tables(angles, dtype=_DTYPE):
    """
    Precompute trigonometric lookup tables for projection angles and transfer to GPU memory.
    Accepts either numpy arrays or torch tensors (on CPU or CUDA).
    Returns (cos, sin) as torch tensors on the same device as input.
    """
    if isinstance(angles, torch.Tensor):
        device = angles.device
        cos = torch.cos(angles).to(dtype=dtype)
        sin = torch.sin(angles).to(dtype=dtype)
        return cos.to(device), sin.to(device)
    else:
        # fallback for numpy arrays
        cos_host = np.cos(angles).astype(dtype)
        sin_host = np.sin(angles).astype(dtype)
        return torch.from_numpy(cos_host), torch.from_numpy(sin_host)




def _grid_2d(n1, n2, tpb=_TPB_2D):
    """
    CUDA 2D grid configuration for optimal thread organization in ray-tracing kernels.
    
    Thread organization strategy:
    - Each thread processes one ray (projection angle, detector element pair)
    - Grid dimensions calculated to cover all rays with minimal thread divergence
    - Block size (16x16) chosen to maximize occupancy while fitting in shared memory
    
    Memory access optimization:
    - Threads in same warp access nearby detector elements (coalesced reads from sinogram)
    - Ray geometry calculations benefit from spatial locality in trigonometric tables
    """
    return (math.ceil(n1 / tpb[0]), math.ceil(n2 / tpb[1])), tpb


def _grid_3d(n1, n2, n3, tpb=_TPB_3D):
    """
    CUDA 3D grid configuration for optimal thread organization in cone beam kernels.
    
    Thread organization strategy:
    - Each thread processes one ray (view, detector_u, detector_v triplet)
    - 3D grid maps directly to 3D detector array for intuitive thread-to-ray mapping
    - Block size (8x8x8) balances occupancy with register pressure from 3D calculations
    
    Performance considerations:
    - Smaller block size accommodates higher register usage in 3D ray-tracing
    - 3D thread indexing enables efficient detector array traversal patterns
    - Memory coalescing optimized for 3D sinogram access patterns
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
    det_spacing, d_cos, d_sin, cx, cy
):
    """
    CUDA kernel implementing the Siddon-Joseph ray-tracing algorithm for 2D parallel beam forward projection.
    
    The Siddon-Joseph algorithm performs accurate ray-volume intersection by:
    1. Computing ray-volume boundary intersections to determine traversal limits
    2. Stepping through voxels along the ray path using parametric ray equations
    3. Computing bilinear interpolation weights for sub-voxel sampling
    4. Accumulating weighted voxel values proportional to ray segment lengths
    
    Mathematical basis: For ray r(t) = start + t*direction, the algorithm finds
    intersections with voxel boundaries and integrates the volume function along the ray.
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
    u     = (idet - (n_det - 1) * 0.5) * det_spacing  # Detector coordinate (centered)

    # Define ray direction and starting point for parallel beam geometry
    # Ray direction is perpendicular to detector array (cos_a, sin_a)
    # Ray starting point is offset along detector by distance u
    dir_x, dir_y = cos_a, sin_a
    pnt_x, pnt_y = u * -sin_a, u * cos_a

    # === RAY-VOLUME INTERSECTION CALCULATION ===
    # Compute parametric intersection points with volume boundaries using ray equation r(t) = pnt + t*dir
    # Volume extends from [-cx, cx] x [-cy, cy] in image coordinate system
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
        if 0 <= ix < Nx - 1 and 0 <= iy < Ny - 1:
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
                
                # === BILINEAR INTERPOLATION WEIGHT CALCULATION ===
                # Mathematical basis: Bilinear interpolation formula f(x,y) = Σ f(xi,yi) * wi(x,y)
                # where wi(x,y) are the bilinear basis functions for each corner voxel
                # Weights are products of 1D linear interpolation weights: (1-dx) or dx, (1-dy) or dy
                val = (
                    d_image[ix0,     iy0]     * (1 - dx) * (1 - dy) +  # Bottom-left:  weight = distance from opposite corner
                    d_image[ix0 + 1, iy0]     * dx       * (1 - dy) +  # Bottom-right: weight = dx * (1-dy)
                    d_image[ix0,     iy0 + 1] * (1 - dx) * dy       +  # Top-left:     weight = (1-dx) * dy  
                    d_image[ix0 + 1, iy0 + 1] * dx       * dy          # Top-right:    weight = dx * dy
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
    det_spacing, d_cos, d_sin, cx, cy
):
    """
    CUDA kernel implementing the Siddon-Joseph algorithm for 2D parallel beam backprojection.
    
    This is the adjoint operation to forward projection, distributing sinogram values back
    into the volume along the same ray paths. Uses identical ray-tracing logic but with
    atomic operations to handle concurrent writes from multiple threads.
    
    Mathematical basis: Implements the transpose of the forward projection matrix,
    distributing detector measurements back along ray paths with bilinear interpolation.
    """
    iang, idet = cuda.grid(2)
    if iang >= n_ang or idet >= n_det:
        return

    # === RAY GEOMETRY SETUP (identical to forward projection) ===
    val   = d_sino[iang, idet]  # Sinogram value to backproject
    cos_a = d_cos[iang]         # Precomputed cosine of projection angle
    sin_a = d_sin[iang]         # Precomputed sine of projection angle
    u     = (idet - (n_det - 1) * 0.5) * det_spacing  # Detector coordinate (centered)

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
        if 0 <= ix < Nx - 1 and 0 <= iy < Ny - 1:
            t_next = min(tx, ty, t_max)
            seg_len = t_next - t
            if seg_len > _EPSILON:
                # Sample at ray segment midpoint (same as forward projection)
                mid_x = pnt_x + (t + seg_len * 0.5) * dir_x + cx
                mid_y = pnt_y + (t + seg_len * 0.5) * dir_y + cy
                ix0, iy0 = int(math.floor(mid_x)), int(math.floor(mid_y))
                dx, dy = mid_x - ix0, mid_y - iy0
                
                # === ATOMIC BACKPROJECTION WITH BILINEAR WEIGHTS ===
                # Distribute contribution weighted by segment length and interpolation weights
                # CUDA ATOMIC OPERATIONS: Essential for thread safety in backprojection
                # Multiple threads (rays) can write to the same voxel simultaneously, causing race conditions
                # Atomic add operations serialize these writes, ensuring correct accumulation of contributions
                # Performance impact: Atomic operations are slower than regular writes but necessary for correctness
                # Memory access pattern: Global memory atomics with potential bank conflicts, but unavoidable
                cval = val * seg_len  # Contribution value for this ray segment
                cuda.atomic.add(d_image, (ix0,     iy0),     cval * (1 - dx) * (1 - dy))  # Bottom-left corner atomic write
                cuda.atomic.add(d_image, (ix0 + 1, iy0),     cval * dx       * (1 - dy))  # Bottom-right corner atomic write
                cuda.atomic.add(d_image, (ix0,     iy0 + 1), cval * (1 - dx) * dy)        # Top-left corner atomic write
                cuda.atomic.add(d_image, (ix0 + 1, iy0 + 1), cval * dx       * dy)        # Top-right corner atomic write

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
    src_dist, iso_dist, cx, cy
):
    """
    CUDA kernel implementing the Siddon-Joseph algorithm for 2D fan beam forward projection.
    
    Fan beam geometry differs from parallel beam by having rays emanate from a point source
    rather than being parallel. Each ray connects the X-ray source to a detector element,
    creating a divergent beam pattern. The same Siddon-Joseph traversal algorithm is used
    but with different ray geometry calculations.
    
    Coordinate system: Source rotates around isocenter, detector array is positioned
    at fixed distance from source, rays connect source to individual detector pixels.
    """
    iang, idet = cuda.grid(2)
    if iang >= n_ang or idet >= n_det:
        return

    # === FAN BEAM GEOMETRY SETUP ===
    cos_a = d_cos[iang]  # Precomputed cosine of projection angle
    sin_a = d_sin[iang]  # Precomputed sine of projection angle
    u     = (idet - (n_det - 1) * 0.5) * det_spacing  # Detector coordinate (centered)

    # Calculate source and detector positions for current projection angle
    # Source position: rotated by angle around isocenter at distance iso_dist
    src_x = -iso_dist * sin_a  # Source x-coordinate in world space
    src_y =  iso_dist * cos_a  # Source y-coordinate in world space
    
    # Detector element position: on detector array at distance src_dist from isocenter
    # Detector array is perpendicular to source-isocenter line, offset by u
    det_x = (src_dist - iso_dist) * sin_a + u * cos_a   # Detector x-coordinate
    det_y = -(src_dist - iso_dist) * cos_a + u * sin_a  # Detector y-coordinate

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
        if 0 <= ix < Nx - 1 and 0 <= iy < Ny - 1:
            t_next = min(tx, ty, t_max)
            seg_len = t_next - t
            if seg_len > _EPSILON:
                # Sample at midpoint using source as ray origin
                mid_x = src_x + (t + seg_len * 0.5) * dir_x + cx
                mid_y = src_y + (t + seg_len * 0.5) * dir_y + cy
                ix0, iy0 = int(math.floor(mid_x)), int(math.floor(mid_y))
                dx, dy = mid_x - ix0, mid_y - iy0
                
                # Bilinear interpolation (identical to parallel beam)
                val = (
                    d_image[ix0,     iy0]     * (1 - dx) * (1 - dy) +
                    d_image[ix0 + 1, iy0]     * dx       * (1 - dy) +
                    d_image[ix0,     iy0 + 1] * (1 - dx) * dy       +
                    d_image[ix0 + 1, iy0 + 1] * dx       * dy
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
    src_dist, iso_dist, cx, cy
):
    """
    CUDA kernel implementing the Siddon-Joseph algorithm for 2D fan beam backprojection.
    
    This is the adjoint operation to fan beam forward projection, distributing sinogram values
    back into the volume along the same ray paths. Uses identical ray-tracing logic but with
    atomic operations to handle concurrent writes from multiple threads.
    
    Mathematical basis: Implements the transpose of the fan beam forward projection matrix,
    distributing detector measurements back along divergent ray paths with bilinear interpolation.
    """
    iang, idet = cuda.grid(2)
    if iang >= n_ang or idet >= n_det:
        return

    # === BACKPROJECTION VALUE AND GEOMETRY SETUP ===
    val   = d_sino[iang, idet]  # Sinogram value to backproject along this ray
    cos_a = d_cos[iang]         # Precomputed cosine of projection angle
    sin_a = d_sin[iang]         # Precomputed sine of projection angle
    u     = (idet - (n_det - 1) * 0.5) * det_spacing  # Detector coordinate (centered)

    # Calculate source and detector positions for current projection angle
    # Source position: rotated by angle around isocenter at distance iso_dist
    src_x = -iso_dist * sin_a  # Source x-coordinate in world space
    src_y =  iso_dist * cos_a  # Source y-coordinate in world space
    
    # Detector element position: on detector array at distance src_dist from isocenter
    det_x = (src_dist - iso_dist) * sin_a + u * cos_a   # Detector x-coordinate
    det_y = -(src_dist - iso_dist) * cos_a + u * sin_a  # Detector y-coordinate

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
        if 0 <= ix < Nx - 1 and 0 <= iy < Ny - 1:
            t_next = min(tx, ty, t_max)
            seg_len = t_next - t
            if seg_len > _EPSILON:
                # Sample at ray segment midpoint using source as ray origin
                mid_x = src_x + (t + seg_len * 0.5) * dir_x + cx
                mid_y = src_y + (t + seg_len * 0.5) * dir_y + cy
                ix0, iy0 = int(math.floor(mid_x)), int(math.floor(mid_y))
                dx, dy = mid_x - ix0, mid_y - iy0
                
                # === ATOMIC BACKPROJECTION WITH BILINEAR WEIGHTS ===
                # Distribute contribution weighted by segment length and interpolation weights
                # CUDA ATOMIC OPERATIONS: Critical for fan beam backprojection thread safety
                # Fan beam rays converge at source, creating higher probability of voxel write conflicts
                # Atomic operations prevent race conditions when multiple divergent rays write to same voxel
                # Performance consideration: Fan beam geometry may have more atomic contention than parallel beam
                cval = val * seg_len  # Contribution value for this ray segment
                cuda.atomic.add(d_image, (ix0,     iy0),     cval * (1 - dx) * (1 - dy))  # Bottom-left corner atomic write
                cuda.atomic.add(d_image, (ix0 + 1, iy0),     cval * dx       * (1 - dy))  # Bottom-right corner atomic write
                cuda.atomic.add(d_image, (ix0,     iy0 + 1), cval * (1 - dx) * dy)        # Top-left corner atomic write
                cuda.atomic.add(d_image, (ix0 + 1, iy0 + 1), cval * dx       * dy)        # Top-right corner atomic write

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
    src_dist, iso_dist, cx, cy, cz
):
    """
    CUDA kernel implementing the Siddon-Joseph algorithm for 3D cone beam forward projection.
    
    Cone beam geometry extends fan beam to 3D with a 2D detector array. Rays emanate from
    a point source to each detector pixel, creating a cone-shaped beam. The Siddon-Joseph
    algorithm is extended to 3D with trilinear interpolation and 3D voxel traversal.
    
    Coordinate system: Source rotates around isocenter in xy-plane, 2D detector array
    positioned at fixed distance from source with (u,v) coordinates, z-axis is vertical.
    """
    iview, iu, iv = cuda.grid(3)
    if iview >= n_views or iu >= n_u or iv >= n_v:
        return

    # === 3D CONE BEAM GEOMETRY SETUP ===
    cos_a, sin_a = d_cos[iview], d_sin[iview]  # Projection angle trigonometry
    u, v = (iu - (n_u - 1) * 0.5) * du, (iv - (n_v - 1) * 0.5) * dv  # Detector coordinates (centered)

    # Calculate 3D source and detector positions
    # Source rotates in xy-plane around isocenter, z-coordinate is zero
    src_x, src_y, src_z = -iso_dist * sin_a, iso_dist * cos_a, 0.0
    
    # Detector element position: 2D array perpendicular to source-isocenter line
    # u-coordinate is in-plane offset, v-coordinate is vertical (z-direction)
    det_x = (src_dist - iso_dist) * sin_a + u * cos_a   # In-plane x-coordinate
    det_y = -(src_dist - iso_dist) * cos_a + u * sin_a  # In-plane y-coordinate  
    det_z = v                                           # Vertical z-coordinate

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
        if 0 <= ix < Nx - 1 and 0 <= iy < Ny - 1 and 0 <= iz < Nz - 1:
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
    src_dist, iso_dist, cx, cy, cz
):
    """
    CUDA kernel implementing the Siddon-Joseph algorithm for 3D cone beam backprojection.
    
    This is the adjoint operation to cone beam forward projection, distributing sinogram values
    back into the 3D volume along the same ray paths. Uses identical ray-tracing logic but with
    atomic operations to handle concurrent writes from multiple threads.
    
    Mathematical basis: Implements the transpose of the cone beam forward projection matrix,
    distributing detector measurements back along divergent 3D ray paths with trilinear interpolation.
    """
    iview, iu, iv = cuda.grid(3)
    if iview >= n_views or iu >= n_u or iv >= n_v:
        return

    # === 3D BACKPROJECTION VALUE AND GEOMETRY SETUP ===
    g = d_sino[iview, iu, iv]  # Sinogram value to backproject along this ray
    cos_a, sin_a = d_cos[iview], d_sin[iview]  # Projection angle trigonometry
    u, v = (iu - (n_u - 1) * 0.5) * du, (iv - (n_v - 1) * 0.5) * dv  # Detector coordinates (centered)

    # Calculate 3D source and detector positions
    # Source rotates in xy-plane around isocenter, z-coordinate is zero
    src_x, src_y, src_z = -iso_dist * sin_a, iso_dist * cos_a, 0.0
    
    # Detector element position: 2D array perpendicular to source-isocenter line
    # u-coordinate is in-plane offset, v-coordinate is vertical (z-direction)
    det_x = (src_dist - iso_dist) * sin_a + u * cos_a   # In-plane x-coordinate
    det_y = -(src_dist - iso_dist) * cos_a + u * sin_a  # In-plane y-coordinate
    det_z = v                                           # Vertical z-coordinate

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
        if 0 <= ix < Nx - 1 and 0 <= iy < Ny - 1 and 0 <= iz < Nz - 1:
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
    PyTorch autograd function for differentiable 2D parallel beam forward projection.
    
    This class provides a differentiable interface to the CUDA-accelerated Siddon-Joseph
    ray-tracing algorithm for parallel beam geometry. It implements the Radon transform
    for parallel beam CT geometry with full gradient support for optimization-based
    reconstruction algorithms.
    
    The forward pass computes the sinogram (projection data) from a 2D image using
    parallel beam geometry, where all rays are parallel and perpendicular to the
    detector array. The backward pass computes gradients using the adjoint backprojection
    operation, enabling end-to-end differentiable CT reconstruction.
    
    Key Features:
        - CUDA-accelerated Siddon-Joseph ray tracing for accurate volume sampling
        - Bilinear interpolation for smooth gradients and sub-pixel accuracy
        - Automatic GPU memory management and device consistency
        - Full PyTorch autograd integration for gradient-based optimization
        - Optimized thread organization for maximum GPU utilization
    
    Mathematical Background:
        The parallel beam Radon transform is defined as:
        
        .. math::
            R[f](s,\\theta) = \\int_{-\\infty}^{\\infty} f(s\\cos\\theta - t\\sin\\theta, s\\sin\\theta + t\\cos\\theta) dt
        
        where f(x,y) is the input image, s is the detector coordinate, and θ is the
        projection angle. This implementation uses the Siddon-Joseph algorithm for
        discrete approximation of the line integral.
    
    Example:
        >>> import torch
        >>> from diffct.differentiable import ParallelProjectorFunction
        >>> 
        >>> # Create a 2D image with gradient tracking
        >>> image = torch.randn(128, 128, device='cuda', requires_grad=True)
        >>> 
        >>> # Define projection parameters
        >>> angles = torch.linspace(0, torch.pi, 180, device='cuda')
        >>> num_detectors = 128
        >>> detector_spacing = 1.0
        >>> 
        >>> # Compute forward projection
        >>> projector = ParallelProjectorFunction.apply
        >>> sinogram = projector(image, angles, num_detectors, detector_spacing)
        >>> 
        >>> # Compute loss and gradients
        >>> loss = sinogram.sum()
        >>> loss.backward()
        >>> print(f"Gradient shape: {image.grad.shape}")  # (128, 128)
    
    Note:
        This function requires CUDA-capable hardware and properly configured CUDA
        environment. All input tensors must be on the same CUDA device.
    """
    @staticmethod
    def forward(ctx, image, angles, num_detectors, detector_spacing=1.0):
        """
        Optimized: Compute parallel beam forward projection (Radon transform) with minimal CPU-GPU transfers.
        """
        device = DeviceManager.get_device(image)
        image = DeviceManager.ensure_device(image, device)
        angles = DeviceManager.ensure_device(angles, device)

        # Ensure input is float32 for kernel compatibility
        image = image.to(dtype=torch.float32)
        angles = angles.to(dtype=torch.float32)

        Nx, Ny = image.shape
        n_angles = angles.shape[0]

        # Allocate output tensor on the same device
        sinogram = torch.zeros((n_angles, num_detectors), dtype=image.dtype, device=device)

        # Prepare trigonometric tables on the correct device
        d_cos, d_sin = _trig_tables(angles, dtype=image.dtype)

        # Get Numba CUDA array views for kernel
        d_image = TorchCUDABridge.tensor_to_cuda_array(image)
        d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        grid, tpb = _grid_2d(n_angles, num_detectors)
        cx, cy = _DTYPE((Nx - 1) * 0.5), _DTYPE((Ny - 1) * 0.5)

        _parallel_2d_forward_kernel[grid, tpb](
            d_image, Nx, Ny, d_sino, n_angles, num_detectors,
            _DTYPE(detector_spacing), d_cos_arr, d_sin_arr, cx, cy
        )

        ctx.save_for_backward(angles)
        ctx.intermediate = (num_detectors, detector_spacing, image.shape[0], image.shape[1])
        return sinogram

    def backward(ctx, grad_sinogram):
        angles, = ctx.saved_tensors
        num_detectors, detector_spacing, H, W = ctx.intermediate
        device = DeviceManager.get_device(grad_sinogram)
        grad_sinogram = DeviceManager.ensure_device(grad_sinogram, device)
        angles = DeviceManager.ensure_device(angles, device)

        grad_sinogram = grad_sinogram.to(dtype=torch.float32)
        angles = angles.to(dtype=torch.float32)

        n_angles = angles.shape[0]
        Nx, Ny = W, H

        grad_image = torch.zeros((Nx, Ny), dtype=grad_sinogram.dtype, device=device)
        d_cos, d_sin = _trig_tables(angles, dtype=grad_sinogram.dtype)

        d_grad_sino = TorchCUDABridge.tensor_to_cuda_array(grad_sinogram)
        d_img_grad = TorchCUDABridge.tensor_to_cuda_array(grad_image)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        grid, tpb = _grid_2d(n_angles, num_detectors)
        cx, cy = _DTYPE((Nx - 1) * 0.5), _DTYPE((Ny - 1) * 0.5)

        _parallel_2d_backward_kernel[grid, tpb](
            d_grad_sino, n_angles, num_detectors,
            d_img_grad, Nx, Ny,
            _DTYPE(detector_spacing), d_cos_arr, d_sin_arr, cx, cy
        )

        return grad_image, None, None, None


class ParallelBackprojectorFunction(torch.autograd.Function):
    """
    PyTorch autograd function for differentiable 2D parallel beam backprojection.
    
    This class provides a differentiable interface to the CUDA-accelerated Siddon-Joseph
    ray-tracing algorithm for parallel beam backprojection. It implements the adjoint
    (transpose) of the Radon transform, distributing sinogram values back into the
    reconstruction volume along the same ray paths used in forward projection.
    
    The forward pass computes a 2D reconstruction from sinogram data using parallel
    beam backprojection. The backward pass computes gradients using forward projection,
    enabling end-to-end differentiable CT reconstruction pipelines.
    
    Key Features:
        - CUDA-accelerated Siddon-Joseph ray tracing with atomic operations
        - Bilinear interpolation for smooth gradient flow and sub-pixel accuracy
        - Thread-safe voxel accumulation using CUDA atomic operations
        - Full PyTorch autograd integration for gradient-based optimization
        - Automatic memory management and device consistency
    
    Mathematical Background:
        The parallel beam backprojection is the adjoint of the Radon transform:
        
        .. math::
            R^*[p](x,y) = \\int_0^\\pi p(x\\cos\\theta + y\\sin\\theta, \\theta) d\\theta
        
        where p(s,θ) is the sinogram data, and R* is the backprojection operator.
        This operation distributes each projection ray's value back along its path
        through the reconstruction volume.
    
    Example:
        >>> import torch
        >>> from diffct.differentiable import ParallelBackprojectorFunction
        >>> 
        >>> # Create sinogram data with gradient tracking
        >>> sinogram = torch.randn(180, 128, device='cuda', requires_grad=True)
        >>> 
        >>> # Define reconstruction parameters
        >>> angles = torch.linspace(0, torch.pi, 180, device='cuda')
        >>> detector_spacing = 1.0
        >>> H, W = 128, 128  # Reconstruction size
        >>> 
        >>> # Compute backprojection
        >>> backprojector = ParallelBackprojectorFunction.apply
        >>> reconstruction = backprojector(sinogram, angles, detector_spacing, H, W)
        >>> 
        >>> # Compute loss and gradients
        >>> loss = reconstruction.sum()
        >>> loss.backward()
        >>> print(f"Sinogram gradient shape: {sinogram.grad.shape}")  # (180, 128)
    
    Note:
        This function requires CUDA-capable hardware and properly configured CUDA
        environment. All input tensors must be on the same CUDA device.
    """
    @staticmethod
    def forward(ctx, sinogram, angles, detector_spacing=1.0, H=128, W=128):
        """
        Optimized: Compute parallel beam backprojection (adjoint Radon transform) with minimal CPU-GPU transfers.
        """
        device = DeviceManager.get_device(sinogram)
        sinogram = DeviceManager.ensure_device(sinogram, device)
        angles = DeviceManager.ensure_device(angles, device)

        # Ensure input is float32 for kernel compatibility
        sinogram = sinogram.to(dtype=torch.float32)
        angles = angles.to(dtype=torch.float32)

        n_ang, n_det = sinogram.shape
        Nx, Ny = W, H

        # Allocate output tensor on the same device
        reco = torch.zeros((Nx, Ny), dtype=sinogram.dtype, device=device)

        # Prepare trigonometric tables on the correct device
        d_cos, d_sin = _trig_tables(angles, dtype=sinogram.dtype)

        # Get Numba CUDA array views for kernel
        d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
        d_reco = TorchCUDABridge.tensor_to_cuda_array(reco)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        grid, tpb = _grid_2d(n_ang, n_det)
        cx, cy = _DTYPE((Nx - 1) * 0.5), _DTYPE((Ny - 1) * 0.5)

        _parallel_2d_backward_kernel[grid, tpb](
            d_sino, n_ang, n_det, d_reco, Nx, Ny,
            _DTYPE(detector_spacing), d_cos_arr, d_sin_arr, cx, cy
        )

        ctx.save_for_backward(angles)
        ctx.intermediate = (H, W, detector_spacing, sinogram.shape[0], sinogram.shape[1])
        return reco

    @staticmethod
    def backward(ctx, grad_output):
        angles, = ctx.saved_tensors
        H, W, detector_spacing, n_ang, n_det = ctx.intermediate
        device = DeviceManager.get_device(grad_output)
        grad_output = DeviceManager.ensure_device(grad_output, device)
        angles = DeviceManager.ensure_device(angles, device)

        grad_output = grad_output.to(dtype=torch.float32)
        angles = angles.to(dtype=torch.float32)

        Nx, Ny = grad_output.shape

        # Allocate output tensor on the same device
        grad_sino = torch.zeros((n_ang, n_det), dtype=grad_output.dtype, device=device)

        # Prepare trigonometric tables on the correct device
        d_cos, d_sin = _trig_tables(angles, dtype=grad_output.dtype)

        # Get Numba CUDA array views for kernel
        d_grad_out = TorchCUDABridge.tensor_to_cuda_array(grad_output)
        d_sino_grad = TorchCUDABridge.tensor_to_cuda_array(grad_sino)
        d_cos = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        grid, tpb = _grid_2d(n_ang, n_det)
        cx, cy = _DTYPE((Nx - 1) * 0.5), _DTYPE((Ny - 1) * 0.5)

        _parallel_2d_forward_kernel[grid, tpb](
            d_grad_out, Nx, Ny, d_sino_grad, n_ang, n_det,
            _DTYPE(detector_spacing), d_cos, d_sin, cx, cy
        )

        return grad_sino, None, None, None, None


class FanProjectorFunction(torch.autograd.Function):
    """
    PyTorch autograd function for differentiable 2D fan beam forward projection.
    
    This class provides a differentiable interface to the CUDA-accelerated Siddon-Joseph
    ray-tracing algorithm for fan beam geometry. Fan beam geometry uses divergent rays
    emanating from a point X-ray source to a linear detector array, which is common
    in medical CT scanners and provides faster data acquisition than parallel beam.
    
    The forward pass computes the sinogram from a 2D image using fan beam geometry,
    where rays diverge from a point source to individual detector elements. The backward
    pass computes gradients using the adjoint backprojection operation with the same
    divergent ray geometry.
    
    Key Features:
        - CUDA-accelerated Siddon-Joseph ray tracing for divergent beam geometry
        - Point source to detector ray path calculations with geometric corrections
        - Bilinear interpolation for smooth gradients and sub-pixel accuracy
        - Full PyTorch autograd integration for gradient-based optimization
        - Configurable source and detector distances for flexible geometry setup
    
    Mathematical Background:
        Fan beam projection involves rays from a point source at distance `source_distance`
        from the isocenter to detector elements at distance `detector_distance` from the source.
        The magnification factor M = detector_distance / source_distance determines the
        geometric scaling between object and detector coordinates.
        
        Each ray connects the source position (rotated around isocenter) to a specific
        detector element, creating a divergent beam pattern that covers the field of view.
    
    Example:
        >>> import torch
        >>> from diffct.differentiable import FanProjectorFunction
        >>> 
        >>> # Create a 2D image with gradient tracking
        >>> image = torch.randn(256, 256, device='cuda', requires_grad=True)
        >>> 
        >>> # Define fan beam geometry parameters
        >>> angles = torch.linspace(0, 2*torch.pi, 360, device='cuda')
        >>> num_detectors = 512
        >>> detector_spacing = 1.0
        >>> source_distance = 1000.0    # Distance from source to isocenter
        >>> detector_distance = 1500.0  # Distance from source to detector
        >>> 
        >>> # Compute forward projection
        >>> projector = FanProjectorFunction.apply
        >>> sinogram = projector(image, angles, num_detectors, detector_spacing,
        ...                     source_distance, detector_distance)
        >>> 
        >>> # Compute loss and gradients
        >>> loss = sinogram.sum()
        >>> loss.backward()
        >>> print(f"Gradient shape: {image.grad.shape}")  # (256, 256)
    
    Note:
        Fan beam geometry requires 360° angular sampling (2π radians) for complete
        reconstruction, unlike parallel beam which only needs 180°. The source distance
        should be much larger than the object size to minimize geometric distortions.
    """
    @staticmethod
    def forward(ctx, image, angles, num_detectors, detector_spacing, source_distance, isocenter_distance):
        """
        Optimized: Compute fan beam forward projection with minimal CPU-GPU transfers.
        """
        device = DeviceManager.get_device(image)
        image = DeviceManager.ensure_device(image, device)
        angles = DeviceManager.ensure_device(angles, device)

        image = image.to(dtype=torch.float32)
        angles = angles.to(dtype=torch.float32)

        Nx, Ny = image.shape
        n_ang = angles.shape[0]

        sinogram = torch.zeros((n_ang, num_detectors), dtype=image.dtype, device=device)
        d_cos, d_sin = _trig_tables(angles, dtype=image.dtype)

        d_image = TorchCUDABridge.tensor_to_cuda_array(image)
        d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        grid, tpb = _grid_2d(n_ang, num_detectors)
        cx, cy = _DTYPE((Nx - 1) * 0.5), _DTYPE((Ny - 1) * 0.5)

        _fan_2d_forward_kernel[grid, tpb](
            d_image, Nx, Ny, d_sino, n_ang, num_detectors,
            _DTYPE(detector_spacing), d_cos_arr, d_sin_arr,
            _DTYPE(source_distance), _DTYPE(isocenter_distance), cx, cy
        )

        ctx.save_for_backward(angles)
        ctx.intermediate = (num_detectors, detector_spacing, image.shape[0], image.shape[1],
                            source_distance, isocenter_distance)
        return sinogram

    @staticmethod
    def backward(ctx, grad_sinogram):
        angles, = ctx.saved_tensors
        (n_det, det_spacing, H, W, src_dist, iso_dist) = ctx.intermediate
        device = DeviceManager.get_device(grad_sinogram)
        grad_sinogram = DeviceManager.ensure_device(grad_sinogram, device)
        angles = DeviceManager.ensure_device(angles, device)

        grad_sinogram = grad_sinogram.to(dtype=torch.float32)
        angles = angles.to(dtype=torch.float32)

        n_ang = angles.shape[0]
        Nx, Ny = W, H

        grad_img = torch.zeros((Nx, Ny), dtype=grad_sinogram.dtype, device=device)
        d_cos, d_sin = _trig_tables(angles, dtype=grad_sinogram.dtype)

        d_grad_sino = TorchCUDABridge.tensor_to_cuda_array(grad_sinogram)
        d_img_grad = TorchCUDABridge.tensor_to_cuda_array(grad_img)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        grid, tpb = _grid_2d(n_ang, n_det)
        cx, cy = _DTYPE((Nx - 1) * 0.5), _DTYPE((Ny - 1) * 0.5)

        _fan_2d_backward_kernel[grid, tpb](
            d_grad_sino, n_ang, n_det, d_img_grad, Nx, Ny,
            _DTYPE(det_spacing), d_cos_arr, d_sin_arr,
            _DTYPE(src_dist), _DTYPE(iso_dist), cx, cy
        )

        return grad_img, None, None, None, None, None


class FanBackprojectorFunction(torch.autograd.Function):
    """
    PyTorch autograd function for differentiable 2D fan beam backprojection.
    
    This class provides a differentiable interface to the CUDA-accelerated Siddon-Joseph
    ray-tracing algorithm for fan beam backprojection. It implements the adjoint of the
    fan beam projection operator, distributing sinogram values back into the reconstruction
    volume along divergent ray paths from point source to detector elements.
    
    The forward pass computes a 2D reconstruction from fan beam sinogram data using
    backprojection with divergent ray geometry. The backward pass computes gradients
    using fan beam forward projection, enabling end-to-end differentiable CT reconstruction.
    
    Key Features:
        - CUDA-accelerated Siddon-Joseph ray tracing with divergent beam geometry
        - Point source to detector ray path calculations with geometric corrections
        - Thread-safe voxel accumulation using CUDA atomic operations
        - Full PyTorch autograd integration for gradient-based optimization
        - Configurable source and detector distances for flexible geometry setup
    
    Mathematical Background:
        Fan beam backprojection distributes each detector measurement back along its
        corresponding divergent ray path. The ray geometry is determined by the source
        position (rotating around isocenter) and the detector element position, creating
        a divergent pattern that reconstructs the object with geometric magnification.
    
    Example:
        >>> import torch
        >>> from diffct.differentiable import FanBackprojectorFunction
        >>> 
        >>> # Create fan beam sinogram data with gradient tracking
        >>> sinogram = torch.randn(360, 512, device='cuda', requires_grad=True)
        >>> 
        >>> # Define reconstruction parameters
        >>> angles = torch.linspace(0, 2*torch.pi, 360, device='cuda')
        >>> detector_spacing = 1.0
        >>> H, W = 256, 256  # Reconstruction size
        >>> source_distance = 1000.0
        >>> isocenter_distance = 500.0
        >>> 
        >>> # Compute backprojection
        >>> backprojector = FanBackprojectorFunction.apply
        >>> reconstruction = backprojector(sinogram, angles, detector_spacing, H, W,
        ...                               source_distance, isocenter_distance)
        >>> 
        >>> # Compute loss and gradients
        >>> loss = reconstruction.sum()
        >>> loss.backward()
        >>> print(f"Sinogram gradient shape: {sinogram.grad.shape}")  # (360, 512)
    
    Note:
        Fan beam backprojection requires careful handling of ray convergence at the source,
        which can lead to higher atomic contention in CUDA kernels compared to parallel beam.
        The geometry parameters must match those used in forward projection for consistency.
    """
    @staticmethod
    def forward(ctx, sinogram, angles, detector_spacing, H, W, source_distance, isocenter_distance):
        """
        Optimized: Compute fan beam backprojection with minimal CPU-GPU transfers.
        """
        device = DeviceManager.get_device(sinogram)
        sinogram = DeviceManager.ensure_device(sinogram, device)
        angles = DeviceManager.ensure_device(angles, device)

        sinogram = sinogram.to(dtype=torch.float32)
        angles = angles.to(dtype=torch.float32)

        n_ang, n_det = sinogram.shape
        Nx, Ny = W, H

        reco = torch.zeros((Nx, Ny), dtype=sinogram.dtype, device=device)
        d_cos, d_sin = _trig_tables(angles, dtype=sinogram.dtype)

        d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
        d_reco = TorchCUDABridge.tensor_to_cuda_array(reco)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        grid, tpb = _grid_2d(n_ang, n_det)
        cx, cy = _DTYPE((Nx - 1) * 0.5), _DTYPE((Ny - 1) * 0.5)

        _fan_2d_backward_kernel[grid, tpb](
            d_sino, n_ang, n_det, d_reco, Nx, Ny,
            _DTYPE(detector_spacing), d_cos_arr, d_sin_arr,
            _DTYPE(source_distance), _DTYPE(isocenter_distance), cx, cy
        )

        ctx.save_for_backward(angles)
        ctx.intermediate = (H, W, detector_spacing, n_ang, n_det, source_distance, isocenter_distance)
        return reco

    @staticmethod
    def backward(ctx, grad_output):
        angles, = ctx.saved_tensors
        (H, W, det_spacing, n_ang, n_det, src_dist, iso_dist) = ctx.intermediate
        device = DeviceManager.get_device(grad_output)
        grad_output = DeviceManager.ensure_device(grad_output, device)
        angles = DeviceManager.ensure_device(angles, device)

        grad_output = grad_output.to(dtype=torch.float32)
        angles = angles.to(dtype=torch.float32)

        Nx, Ny = grad_output.shape

        grad_sino = torch.zeros((n_ang, n_det), dtype=grad_output.dtype, device=device)
        d_cos, d_sin = _trig_tables(angles, dtype=grad_output.dtype)

        d_grad_out = TorchCUDABridge.tensor_to_cuda_array(grad_output)
        d_sino_grad = TorchCUDABridge.tensor_to_cuda_array(grad_sino)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        grid, tpb = _grid_2d(n_ang, n_det)
        cx, cy = _DTYPE((Nx - 1) * 0.5), _DTYPE((Ny - 1) * 0.5)

        _fan_2d_forward_kernel[grid, tpb](
            d_grad_out, Nx, Ny, d_sino_grad, n_ang, n_det,
            _DTYPE(det_spacing), d_cos_arr, d_sin_arr,
            _DTYPE(src_dist), _DTYPE(iso_dist), cx, cy
        )

        return grad_sino, None, None, None, None, None, None


class ConeProjectorFunction(torch.autograd.Function):
    """
    PyTorch autograd function for differentiable 3D cone beam forward projection.
    
    This class provides a differentiable interface to the CUDA-accelerated Siddon-Joseph
    ray-tracing algorithm for 3D cone beam geometry. Cone beam geometry uses a point X-ray
    source and 2D detector array to capture volumetric projection data in a single circular
    scan, enabling full 3D CT reconstruction from one acquisition.
    
    The forward pass computes 3D projection data from a volume using cone beam geometry,
    where rays emanate from a point source to each pixel of a 2D detector array. The
    backward pass computes gradients using 3D adjoint backprojection, enabling end-to-end
    differentiable volumetric CT reconstruction.
    
    Key Features:
        - CUDA-accelerated 3D Siddon-Joseph ray tracing with trilinear interpolation
        - Point source to 2D detector array ray path calculations
        - Optimized 3D thread organization for maximum GPU utilization
        - Full PyTorch autograd integration for gradient-based optimization
        - Configurable detector array size and spacing for flexible geometry
    
    Mathematical Background:
        3D cone beam projection involves rays from a point source to each detector pixel
        (u,v) on a 2D detector array. The source rotates around the isocenter in the xy-plane
        while the detector maintains fixed distance from the source. Each ray samples the
        3D volume using trilinear interpolation for accurate gradient computation.
        
        The cone angle should be kept small (< 30°) to minimize cone beam artifacts in
        reconstruction. For larger volumes, helical scanning may be required.
    
    Example:
        >>> import torch
        >>> from diffct.differentiable import ConeProjectorFunction
        >>> 
        >>> # Create a 3D volume with gradient tracking
        >>> volume = torch.randn(128, 128, 128, device='cuda', requires_grad=True)
        >>> 
        >>> # Define cone beam geometry parameters
        >>> angles = torch.linspace(0, 2*torch.pi, 360, device='cuda')
        >>> det_u, det_v = 256, 256  # 2D detector array size
        >>> du, dv = 1.0, 1.0       # Detector pixel spacing
        >>> source_distance = 1000.0    # Source to isocenter distance
        >>> isocenter_distance = 500.0  # Isocenter to detector distance
        >>> 
        >>> # Compute 3D forward projection
        >>> projector = ConeProjectorFunction.apply
        >>> projections = projector(volume, angles, det_u, det_v, du, dv,
        ...                        source_distance, isocenter_distance)
        >>> 
        >>> # Compute loss and gradients
        >>> loss = projections.sum()
        >>> loss.backward()
        >>> print(f"Volume gradient shape: {volume.grad.shape}")  # (128, 128, 128)
    
    Note:
        3D cone beam projection is computationally intensive and requires significant GPU
        memory. Consider using smaller volumes or gradient checkpointing for large-scale
        applications. The cone angle should be minimized to reduce reconstruction artifacts.
    """
    @staticmethod
    def forward(ctx, volume, angles, det_u, det_v, du, dv, source_distance, isocenter_distance):
        """
        Optimized: Compute 3D cone beam forward projection with minimal CPU-GPU transfers.
        """
        device = DeviceManager.get_device(volume)
        volume = DeviceManager.ensure_device(volume, device)
        angles = DeviceManager.ensure_device(angles, device)

        volume = volume.to(dtype=torch.float32)
        angles = angles.to(dtype=torch.float32)

        D, H, W = volume.shape
        n_views = angles.shape[0]

        sino = torch.zeros((n_views, det_u, det_v), dtype=volume.dtype, device=device)
        d_cos, d_sin = _trig_tables(angles, dtype=volume.dtype)

        d_vol = TorchCUDABridge.tensor_to_cuda_array(volume)
        d_sino = TorchCUDABridge.tensor_to_cuda_array(sino)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        grid, tpb = _grid_3d(n_views, det_u, det_v)
        cx, cy, cz = _DTYPE((D-1)*0.5), _DTYPE((H-1)*0.5), _DTYPE((W-1)*0.5)

        _cone_3d_forward_kernel[grid, tpb](
            d_vol, D, H, W, d_sino, n_views, det_u, det_v,
            _DTYPE(du), _DTYPE(dv), d_cos_arr, d_sin_arr,
            _DTYPE(source_distance), _DTYPE(isocenter_distance),
            cx, cy, cz
        )

        ctx.save_for_backward(angles)
        ctx.intermediate = (D, H, W, det_u, det_v, du, dv,
                            source_distance, isocenter_distance)
        return sino

    @staticmethod
    def backward(ctx, grad_sinogram):
        angles, = ctx.saved_tensors
        (D, H, W, det_u, det_v, du, dv,
         src_dist, iso_dist) = ctx.intermediate
        device = DeviceManager.get_device(grad_sinogram)
        grad_sinogram = DeviceManager.ensure_device(grad_sinogram, device)
        angles = DeviceManager.ensure_device(angles, device)

        grad_sinogram = grad_sinogram.to(dtype=torch.float32)
        angles = angles.to(dtype=torch.float32)

        n_views = angles.shape[0]

        grad_vol = torch.zeros((D, H, W), dtype=grad_sinogram.dtype, device=device)
        d_cos, d_sin = _trig_tables(angles, dtype=grad_sinogram.dtype)

        d_grad_sino = TorchCUDABridge.tensor_to_cuda_array(grad_sinogram)
        d_vol_grad = TorchCUDABridge.tensor_to_cuda_array(grad_vol)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        grid, tpb = _grid_3d(n_views, det_u, det_v)
        cx, cy, cz = _DTYPE((D-1)*0.5), _DTYPE((H-1)*0.5), _DTYPE((W-1)*0.5)

        _cone_3d_backward_kernel[grid, tpb](
            d_grad_sino, n_views, det_u, det_v, d_vol_grad, D, H, W,
            _DTYPE(du), _DTYPE(dv), d_cos_arr, d_sin_arr,
            _DTYPE(src_dist), _DTYPE(iso_dist), cx, cy, cz
        )

        return grad_vol, None, None, None, None, None, None, None


class ConeBackprojectorFunction(torch.autograd.Function):
    """
    PyTorch autograd function for differentiable 3D cone beam backprojection.
    
    This class provides a differentiable interface to the CUDA-accelerated Siddon-Joseph
    ray-tracing algorithm for 3D cone beam backprojection. It implements the adjoint of the
    3D cone beam projection operator, distributing projection values from a 2D detector array
    back into a 3D reconstruction volume along divergent ray paths.
    
    The forward pass computes a 3D reconstruction from cone beam projection data using
    backprojection with 3D divergent ray geometry. The backward pass computes gradients
    using 3D cone beam forward projection, enabling end-to-end differentiable volumetric
    CT reconstruction pipelines.
    
    Key Features:
        - CUDA-accelerated 3D Siddon-Joseph ray tracing with trilinear interpolation
        - Point source to 2D detector array ray path calculations in 3D space
        - Thread-safe 3D voxel accumulation using CUDA atomic operations
        - Full PyTorch autograd integration for gradient-based optimization
        - Optimized 3D memory access patterns and thread organization
    
    Mathematical Background:
        3D cone beam backprojection distributes each detector pixel measurement back along
        its corresponding 3D ray path from source to detector. The reconstruction process
        involves trilinear interpolation for sub-voxel accuracy and atomic operations for
        thread-safe accumulation when multiple rays contribute to the same voxel.
        
        Due to the 3D nature and ray convergence, this operation has the highest computational
        complexity and memory requirements among all projection operators in DiffCT.
    
    Example:
        >>> import torch
        >>> from diffct.differentiable import ConeBackprojectorFunction
        >>> 
        >>> # Create 3D cone beam projection data with gradient tracking
        >>> projections = torch.randn(360, 256, 256, device='cuda', requires_grad=True)
        >>> 
        >>> # Define reconstruction parameters
        >>> angles = torch.linspace(0, 2*torch.pi, 360, device='cuda')
        >>> D, H, W = 128, 128, 128  # 3D reconstruction size
        >>> du, dv = 1.0, 1.0       # Detector pixel spacing
        >>> source_distance = 1000.0
        >>> isocenter_distance = 500.0
        >>> 
        >>> # Compute 3D backprojection
        >>> backprojector = ConeBackprojectorFunction.apply
        >>> volume = backprojector(projections, angles, D, H, W, du, dv,
        ...                       source_distance, isocenter_distance)
        >>> 
        >>> # Compute loss and gradients
        >>> loss = volume.sum()
        >>> loss.backward()
        >>> print(f"Projection gradient shape: {projections.grad.shape}")  # (360, 256, 256)
    
    Note:
        3D cone beam backprojection is the most memory and computationally intensive operation
        in DiffCT. Consider using gradient checkpointing, smaller volumes, or distributed
        computing for large-scale applications. Ensure sufficient GPU memory is available.
    """
    @staticmethod
    def forward(ctx, sinogram, angles, D, H, W, du, dv, source_distance, isocenter_distance):
        """
        Optimized: Compute 3D cone beam backprojection with minimal CPU-GPU transfers.
        """
        device = DeviceManager.get_device(sinogram)
        sinogram = DeviceManager.ensure_device(sinogram, device)
        angles = DeviceManager.ensure_device(angles, device)

        sinogram = sinogram.to(dtype=torch.float32)
        angles = angles.to(dtype=torch.float32)

        n_views, n_u, n_v = sinogram.shape

        vol = torch.zeros((D, H, W), dtype=sinogram.dtype, device=device)
        d_cos, d_sin = _trig_tables(angles, dtype=sinogram.dtype)

        d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
        d_reco = TorchCUDABridge.tensor_to_cuda_array(vol)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        grid, tpb = _grid_3d(n_views, n_u, n_v)
        cx, cy, cz = _DTYPE((D-1)*0.5), _DTYPE((H-1)*0.5), _DTYPE((W-1)*0.5)

        _cone_3d_backward_kernel[grid, tpb](
            d_sino, n_views, n_u, n_v, d_reco, D, H, W,
            _DTYPE(du), _DTYPE(dv), d_cos_arr, d_sin_arr,
            _DTYPE(source_distance), _DTYPE(isocenter_distance), cx, cy, cz
        )

        ctx.save_for_backward(angles)
        ctx.intermediate = (D, H, W, n_u, n_v, du, dv,
                            source_distance, isocenter_distance)
        return vol

    @staticmethod
    def backward(ctx, grad_output):
        angles, = ctx.saved_tensors
        (D, H, W, n_u, n_v, du, dv,
         src_dist, iso_dist) = ctx.intermediate
        device = DeviceManager.get_device(grad_output)
        grad_output = DeviceManager.ensure_device(grad_output, device)
        angles = DeviceManager.ensure_device(angles, device)

        grad_output = grad_output.to(dtype=torch.float32)
        angles = angles.to(dtype=torch.float32)

        n_views = angles.shape[0]

        grad_sino = torch.zeros((n_views, n_u, n_v), dtype=grad_output.dtype, device=device)
        d_cos, d_sin = _trig_tables(angles, dtype=grad_output.dtype)

        d_grad_out = TorchCUDABridge.tensor_to_cuda_array(grad_output)
        d_sino_grad = TorchCUDABridge.tensor_to_cuda_array(grad_sino)
        d_cos_arr = TorchCUDABridge.tensor_to_cuda_array(d_cos)
        d_sin_arr = TorchCUDABridge.tensor_to_cuda_array(d_sin)

        grid, tpb = _grid_3d(n_views, n_u, n_v)
        cx, cy, cz = _DTYPE((D-1)*0.5), _DTYPE((H-1)*0.5), _DTYPE((W-1)*0.5)

        _cone_3d_forward_kernel[grid, tpb](
            d_grad_out, D, H, W, d_sino_grad, n_views, n_u, n_v,
            _DTYPE(du), _DTYPE(dv), d_cos_arr, d_sin_arr,
            _DTYPE(src_dist), _DTYPE(iso_dist), cx, cy, cz
        )

        return grad_sino, None, None, None, None, None, None, None, None