import math
import numpy as np
from numba import cuda

# ------------------------------------------------------------------
# GLOBAL SETTINGS
# ------------------------------------------------------------------

_DTYPE = np.float32
# CUDA thread block configurations optimized for different dimensionalities
# 2D blocks: 16x16 = 256 threads per block, optimal for 2D ray-tracing kernels
# Balances occupancy with shared memory usage for parallel/fan beam projections
_TPB_2D = (16, 16)
# 3D blocks: 8x8x8 = 512 threads per block, optimal for 3D cone beam kernels  
# Smaller per-dimension size accommodates higher register usage in 3D algorithms
_TPB_3D = (8, 8, 8)
# CUDA fastmath optimization: enables aggressive floating-point optimizations
# Trades numerical precision for performance in ray-tracing calculations
# Safe for CT reconstruction where slight precision loss is acceptable for speed gains
_FASTMATH_DECORATOR = cuda.jit(fastmath=True)
_INF = _DTYPE(np.inf)
_EPSILON = _DTYPE(1e-6)

# ------------------------------------------------------------------
# SMALL HOST HELPERS
# ------------------------------------------------------------------


def _trig_tables(angles: np.ndarray, dtype=_DTYPE):
    """
    Precompute trigonometric lookup tables for projection angles and transfer to GPU memory.
    
    This helper function optimizes ray geometry calculations by precomputing cosine and sine
    values for all projection angles, avoiding repeated trigonometric calculations in CUDA kernels.
    The precomputed values are transferred to GPU device memory for efficient access during
    ray-tracing operations.
    
    Coordinate system context:
    - Angles define the rotation of the X-ray source/detector system around the reconstruction volume
    - For parallel beam: angles determine ray direction vectors (cos_a, sin_a)
    - For fan/cone beam: angles determine source position relative to isocenter
    
    Performance optimization:
    - Eliminates expensive trigonometric function calls in GPU kernels
    - Enables coalesced memory access to angle-dependent geometry parameters
    - Reduces register pressure in ray-tracing kernels by using precomputed values
    
    Args:
        angles: Array of projection angles in radians
        dtype: Data type for GPU arrays (typically float32 for performance)
        
    Returns:
        Tuple of (d_cos, d_sin): GPU device arrays containing precomputed trigonometric values
    """
    cos_host = np.cos(angles).astype(dtype)
    sin_host = np.sin(angles).astype(dtype)
    return cuda.to_device(cos_host), cuda.to_device(sin_host)


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


# ------------------------------------------------------------------
# 2-D PARALLEL GEOMETRY (SIDDON-JOSEPH ALGORITHM)
# ------------------------------------------------------------------

@_FASTMATH_DECORATOR
def forward_parallel_2d_kernel(
    d_image,
    Nx, Ny,
    d_sino,
    n_views, n_det,
    det_spacing,
    d_cos, d_sin,
    cx, cy
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
    if iang >= n_views or idet >= n_det:
        return

    # === RAY GEOMETRY SETUP ===
    # Extract projection angle and compute detector position
    cos_a = d_cos[iang]  # Precomputed cosine of projection angle
    sin_a = d_sin[iang]  # Precomputed sine of projection angle
    u = (idet - (n_det - 1) * 0.5) * det_spacing  # Detector coordinate (centered)

    # Define ray direction and starting point for parallel beam geometry
    # Ray direction is perpendicular to detector array (cos_a, sin_a)
    # Ray starting point is offset along detector by distance u
    dir_x, dir_y = cos_a, sin_a
    pnt_x, pnt_y = u * (-sin_a), u * (cos_a)

    # === RAY-VOLUME INTERSECTION CALCULATION ===
    # Compute parametric intersection points with volume boundaries using ray equation r(t) = start + t*dir
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
    t = t_min      # Current ray parameter (distance from ray start)
    accum = 0.0    # Accumulated projection value along ray

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
            # Determine next voxel boundary crossing
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
def back_parallel_2d_kernel(
    d_sino,
    Nx, Ny,
    d_reco,
    n_views, n_det, det_spacing,
    d_cos, d_sin,
    cx, cy
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
    if iang >= n_views or idet >= n_det:
        return

    # === RAY GEOMETRY SETUP (identical to forward projection) ===
    val   = d_sino[iang, idet]  # Sinogram value to backproject
    cos_a = d_cos[iang]         # Precomputed cosine of projection angle
    sin_a = d_sin[iang]         # Precomputed sine of projection angle
    u     = (idet - (n_det - 1) * 0.5) * det_spacing  # Detector coordinate (centered)

    # Define ray direction and starting point for parallel beam geometry
    dir_x, dir_y = cos_a, sin_a
    pnt_x, pnt_y = u * -sin_a, u * cos_a

    # === RAY-VOLUME INTERSECTION CALCULATION ===
    # Compute parametric intersection points with volume boundaries using ray equation r(t) = start + t*dir
    # Volume extends from [-cx, cx] x [-cy, cy] in image coordinate system
    t_min, t_max = -_INF, _INF  # Ray parameter range for valid volume intersection
    
    # X-direction boundary intersections
    if abs(dir_x) > _EPSILON:
        tx1, tx2 = (-cx - pnt_x) / dir_x, (cx - pnt_x) / dir_x
        t_min, t_max = max(t_min, min(tx1, tx2)), min(t_max, max(tx1, tx2))
    elif pnt_x < -cx or pnt_x > cx: return

    # Y-direction boundary intersections
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
                
                # === ATOMIC BACKPROJECTION WITH BILINEAR WEIGHTS ===
                # Distribute contribution weighted by segment length and interpolation weights
                # CUDA ATOMIC OPERATIONS: Essential for thread safety in backprojection
                # Multiple threads (rays) can write to the same voxel simultaneously, causing race conditions
                # Atomic add operations serialize these writes, ensuring correct accumulation of contributions
                # Performance impact: Atomic operations are slower than regular writes but necessary for correctness
                # Memory access pattern: Global memory atomics with potential bank conflicts, but unavoidable
                cval = val * seg_len  # Contribution value for this ray segment
                cuda.atomic.add(d_reco, (ix0,     iy0),     cval * (1 - dx) * (1 - dy))  # Bottom-left corner atomic write
                cuda.atomic.add(d_reco, (ix0 + 1, iy0),     cval * dx       * (1 - dy))  # Bottom-right corner atomic write
                cuda.atomic.add(d_reco, (ix0,     iy0 + 1), cval * (1 - dx) * dy)        # Top-left corner atomic write
                cuda.atomic.add(d_reco, (ix0 + 1, iy0 + 1), cval * dx       * dy)        # Top-right corner atomic write

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


def forward_parallel_2d(
    image: np.ndarray,
    n_views: int,
    n_det: int,
    det_spacing: float,
    angles: np.ndarray,
):
    """
    Host function for 2D parallel beam forward projection using Siddon-Joseph ray-tracing.
    
    This function orchestrates the complete forward projection pipeline:
    1. Coordinate system transformation: converts input image to GPU-compatible format
    2. Memory management: transfers data to GPU device memory for CUDA processing
    3. Geometry setup: precomputes trigonometric tables and grid configurations
    4. Kernel execution: launches CUDA ray-tracing kernel with optimal thread organization
    5. Result retrieval: copies computed sinogram back to host memory
    
    Coordinate system transformations:
    - Input image: standard numpy array with (height, width) indexing
    - GPU processing: transposed to (width, height) for optimal memory access patterns
    - Volume centering: image coordinates shifted to center volume at origin
    - Ray geometry: parallel rays perpendicular to detector array at each projection angle
    
    Algorithm integration:
    - Utilizes _trig_tables() helper for efficient angle preprocessing
    - Employs _grid_2d() helper for optimal CUDA thread organization
    - Executes forward_parallel_2d_kernel() for actual ray-tracing computation
    - Handles all GPU memory allocation and data transfer operations
    
    Args:
        image: 2D reconstruction volume as numpy array (height, width)
        n_views: Number of projection angles in the sinogram
        n_det: Number of detector elements per projection
        det_spacing: Physical spacing between detector elements
        angles: Array of projection angles in radians
        
    Returns:
        numpy.ndarray: Computed sinogram with shape (n_views, n_det)
    """
    image_np = image.astype(_DTYPE, copy=False).T
    kernel_Nx, kernel_Ny = image_np.shape
    d_image = cuda.to_device(image_np)
    d_cos, d_sin = _trig_tables(angles, _DTYPE)
    d_sino = cuda.device_array((n_views, n_det), dtype=_DTYPE)

    grid, tpb = _grid_2d(n_views, n_det)
    cx = _DTYPE((kernel_Nx - 1) * 0.5)
    cy = _DTYPE((kernel_Ny - 1) * 0.5)

    forward_parallel_2d_kernel[grid, tpb](
        d_image, kernel_Nx, kernel_Ny, d_sino,
        n_views, n_det, _DTYPE(det_spacing),
        d_cos, d_sin, cx, cy
    )
    return d_sino.copy_to_host()


def back_parallel_2d(
    sinogram: np.ndarray,
    reco_H: int, reco_W: int,
    det_spacing: float,
    angles: np.ndarray,
):
    """
    Host function for 2D parallel beam backprojection using Siddon-Joseph ray-tracing.
    
    This function orchestrates the complete backprojection pipeline:
    1. Data preparation: converts input sinogram to GPU-compatible format
    2. Memory management: transfers sinogram to GPU and allocates reconstruction volume
    3. Geometry setup: precomputes trigonometric tables and grid configurations
    4. Kernel execution: launches CUDA backprojection kernel with optimal thread organization
    5. Result retrieval: copies reconstructed volume back to host memory
    
    Coordinate system transformations:
    - Input sinogram: standard numpy array with (n_views, n_det) indexing
    - GPU processing: direct transfer without transposition for optimal memory access
    - Volume initialization: zero-filled reconstruction volume allocated on GPU
    - Ray geometry: identical to forward projection but with atomic accumulation
    - Output volume: transposed back to standard (height, width) format
    
    Algorithm integration:
    - Utilizes _trig_tables() helper for efficient angle preprocessing
    - Employs _grid_2d() helper for optimal CUDA thread organization
    - Executes back_parallel_2d_kernel() for actual ray-tracing computation
    - Handles all GPU memory allocation and data transfer operations
    - Uses atomic operations for thread-safe voxel accumulation
    
    Args:
        sinogram: Input projection data as numpy array (n_views, n_det)
        reco_H: Height of reconstruction volume in pixels
        reco_W: Width of reconstruction volume in pixels
        det_spacing: Physical spacing between detector elements
        angles: Array of projection angles in radians
        
    Returns:
        numpy.ndarray: Reconstructed volume with shape (reco_H, reco_W)
    """
    sinogram = sinogram.astype(_DTYPE, copy=False)
    n_views, n_det = sinogram.shape
    kernel_Nx, kernel_Ny = int(reco_W), int(reco_H)

    d_sino = cuda.to_device(sinogram)
    d_cos, d_sin = _trig_tables(angles, _DTYPE)
    d_reco = cuda.to_device(np.zeros((kernel_Nx, kernel_Ny), dtype=_DTYPE))

    grid, tpb = _grid_2d(n_views, n_det)
    cx = _DTYPE((kernel_Nx - 1) * 0.5)
    cy = _DTYPE((kernel_Ny - 1) * 0.5)

    back_parallel_2d_kernel[grid, tpb](
        d_sino, kernel_Nx, kernel_Ny, d_reco,
        n_views, n_det, _DTYPE(det_spacing),
        d_cos, d_sin, cx, cy
    )
    return d_reco.copy_to_host().T


# ------------------------------------------------------------------
# 2-D FAN GEOMETRY (SIDDON-JOSEPH ALGORITHM)
# ------------------------------------------------------------------

@_FASTMATH_DECORATOR
def forward_fan_2d_kernel(
    d_image,
    Nx, Ny,
    d_sino,
    n_views, n_det,
    det_spacing,
    d_cos, d_sin,
    src_dist, iso_dist,
    cx, cy
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
    if iang >= n_views or idet >= n_det:
        return

    # === FAN BEAM GEOMETRY SETUP ===
    cos_a = d_cos[iang]  # Precomputed cosine of projection angle
    sin_a = d_sin[iang]  # Precomputed sine of projection angle
    u = (idet - (n_det - 1) * 0.5) * det_spacing  # Detector coordinate (centered)

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
    t_min, t_max = -_INF, _INF  # Ray parameter range for valid volume intersection
    
    # X-direction boundary intersections
    if abs(dir_x) > _EPSILON:
        tx1, tx2 = (-cx - src_x) / dir_x, (cx - src_x) / dir_x  # Volume boundary intersections
        t_min, t_max = max(t_min, min(tx1, tx2)), min(t_max, max(tx1, tx2))  # Update valid parameter range
    elif src_x < -cx or src_x > cx:  # Source outside volume bounds
        d_sino[iang, idet] = 0.0; return

    # Y-direction boundary intersections
    if abs(dir_y) > _EPSILON:
        ty1, ty2 = (-cy - src_y) / dir_y, (cy - src_y) / dir_y
        t_min, t_max = max(t_min, min(ty1, ty2)), min(t_max, max(ty1, ty2))
    elif src_y < -cy or src_y > cy:
        d_sino[iang, idet] = 0.0; return

    if t_min >= t_max:  # No valid intersection
        d_sino[iang, idet] = 0.0; return

    # === SIDDON-JOSEPH TRAVERSAL (same algorithm as parallel beam) ===
    t = t_min    # Current ray parameter
    accum = 0.0  # Accumulated projection value
    
    # Convert ray entry point to voxel indices (using source as ray origin)
    ix = int(math.floor(src_x + t * dir_x + cx))  # Current voxel x-index
    iy = int(math.floor(src_y + t * dir_y + cy))  # Current voxel y-index

    # Traversal parameters (identical to parallel beam implementation)
    step_x, step_y = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1)  # Voxel stepping direction
    dt_x = abs(1.0 / dir_x) if abs(dir_x) > _EPSILON else _INF  # Parameter increment to cross one voxel in x
    dt_y = abs(1.0 / dir_y) if abs(dir_y) > _EPSILON else _INF  # Parameter increment to cross one voxel in y
    
    # Calculate parameter values for next voxel boundary crossings
    tx = ((ix + (step_x > 0)) - cx - src_x) / dir_x if abs(dir_x) > _EPSILON else _INF  # Next x-boundary crossing
    ty = ((iy + (step_y > 0)) - cy - src_y) / dir_y if abs(dir_y) > _EPSILON else _INF  # Next y-boundary crossing

    # === MAIN TRAVERSAL LOOP WITH BILINEAR INTERPOLATION ===
    # Step through voxels along ray path, accumulating weighted contributions
    while t < t_max:
        # Check if current voxel indices are within valid interpolation bounds
        if 0 <= ix < Nx - 1 and 0 <= iy < Ny - 1:
            # Determine next voxel boundary crossing (minimum of x, y boundaries or ray exit)
            t_next = min(tx, ty, t_max)
            seg_len = t_next - t  # Length of ray segment within current voxel region
            
            if seg_len > _EPSILON:  # Only process segments with meaningful length (avoid numerical noise)
                # === BILINEAR INTERPOLATION SAMPLING ===
                # Sample at midpoint using source as ray origin for accurate integration
                # Mathematical basis: Midpoint rule for numerical integration along ray segments
                mid_x = src_x + (t + seg_len * 0.5) * dir_x + cx  # Midpoint x-coordinate in image space
                mid_y = src_y + (t + seg_len * 0.5) * dir_y + cy  # Midpoint y-coordinate in image space
                
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
def back_fan_2d_kernel(
    d_sino,
    n_views, n_det,
    Nx, Ny,
    d_reco,
    det_spacing,
    d_cos, d_sin,
    src_dist, iso_dist,
    cx, cy
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
    if iang >= n_views or idet >= n_det:
        return

    # === BACKPROJECTION VALUE AND GEOMETRY SETUP ===
    val   = d_sino[iang, idet]  # Sinogram value to backproject along this ray
    cos_a = d_cos[iang]         # Precomputed cosine of projection angle
    sin_a = d_sin[iang]         # Precomputed sine of projection angle
    u = (idet - (n_det - 1) * 0.5) * det_spacing  # Detector coordinate (centered)

    # Calculate source and detector positions for current projection angle
    # Source position: rotated by angle around isocenter at distance iso_dist
    src_x, src_y = -iso_dist * sin_a, iso_dist * cos_a  # Source coordinates in world space
    
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
    t_min, t_max = -_INF, _INF  # Ray parameter range for valid volume intersection
    
    # X-direction boundary intersections
    if abs(dir_x) > _EPSILON:
        tx1, tx2 = (-cx - src_x) / dir_x, (cx - src_x) / dir_x  # Volume boundary intersections
        t_min, t_max = max(t_min, min(tx1, tx2)), min(t_max, max(tx1, tx2))  # Update valid parameter range
    elif src_x < -cx or src_x > cx: return  # Source outside volume bounds

    # Y-direction boundary intersections
    if abs(dir_y) > _EPSILON:
        ty1, ty2 = (-cy - src_y) / dir_y, (cy - src_y) / dir_y  # Volume boundary intersections
        t_min, t_max = max(t_min, min(ty1, ty2)), min(t_max, max(ty1, ty2))  # Intersect with x-range
    elif src_y < -cy or src_y > cy: return  # Source outside volume bounds

    if t_min >= t_max: return  # No valid intersection

    # === SIDDON-JOSEPH TRAVERSAL INITIALIZATION ===
    t = t_min    # Current ray parameter
    ix = int(math.floor(src_x + t * dir_x + cx))  # Current voxel x-index
    iy = int(math.floor(src_y + t * dir_y + cy))  # Current voxel y-index

    # Determine traversal direction and step sizes for each axis
    step_x, step_y = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1)  # Voxel stepping direction
    dt_x = abs(1.0 / dir_x) if abs(dir_x) > _EPSILON else _INF  # Parameter increment to cross one voxel in x
    dt_y = abs(1.0 / dir_y) if abs(dir_y) > _EPSILON else _INF  # Parameter increment to cross one voxel in y
    
    # Calculate parameter values for next voxel boundary crossings
    tx = ((ix + (step_x > 0)) - cx - src_x) / dir_x if abs(dir_x) > _EPSILON else _INF  # Next x-boundary crossing
    ty = ((iy + (step_y > 0)) - cy - src_y) / dir_y if abs(dir_y) > _EPSILON else _INF  # Next y-boundary crossing

    # === FAN BEAM BACKPROJECTION TRAVERSAL LOOP ===
    # Distribute sinogram value along divergent ray path using bilinear interpolation
    while t < t_max:
        # Check if current voxel indices are within valid interpolation bounds
        if 0 <= ix < Nx - 1 and 0 <= iy < Ny - 1:
            # Determine next voxel boundary crossing (minimum of x, y boundaries or ray exit)
            t_next = min(tx, ty, t_max)
            seg_len = t_next - t  # Length of ray segment within current voxel region
            
            if seg_len > _EPSILON:  # Only process segments with meaningful length (avoid numerical noise)
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
                # Sample at ray segment midpoint using source as ray origin (same as forward projection)
                # Mathematical basis: Midpoint rule for numerical integration along ray segments
                mid_x = src_x + (t + seg_len * 0.5) * dir_x + cx  # Midpoint x-coordinate in image space
                mid_y = src_y + (t + seg_len * 0.5) * dir_y + cy  # Midpoint y-coordinate in image space
                
                # Convert continuous coordinates to discrete voxel indices and fractional weights
                # Floor operation gives base voxel index, fractional part gives interpolation weights
                ix0, iy0 = int(math.floor(mid_x)), int(math.floor(mid_y))  # Base voxel indices (bottom-left corner)
                dx, dy = mid_x - ix0, mid_y - iy0  # Fractional parts: distance from base voxel center [0,1]

                # === BILINEAR WEIGHT DISTRIBUTION FOR BACKPROJECTION ===
                # Mathematical basis: Transpose of bilinear interpolation - distribute value to 4 neighboring voxels
                # CUDA ATOMIC OPERATIONS: Critical for fan beam backprojection thread safety
                # Fan beam rays converge at source, creating higher probability of voxel write conflicts
                # Atomic operations prevent race conditions when multiple divergent rays write to same voxel
                # Performance consideration: Fan beam geometry may have more atomic contention than parallel beam
                cuda.atomic.add(d_reco, (ix0,     iy0),     cval * (1 - dx) * (1 - dy))  # Bottom-left corner atomic write
                cuda.atomic.add(d_reco, (ix0 + 1, iy0),     cval * dx       * (1 - dy))  # Bottom-right corner atomic write


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


def forward_fan_2d(
    image: np.ndarray,
    n_views: int,
    n_det: int,
    det_spacing: float,
    angles: np.ndarray,
    src_dist: float,
    iso_dist: float,
):
    """
    Host function for 2D fan beam forward projection using Siddon-Joseph ray-tracing.
    
    This function orchestrates the complete fan beam forward projection pipeline:
    1. Coordinate system transformation: converts input image to GPU-compatible format
    2. Memory management: transfers data to GPU device memory for CUDA processing
    3. Geometry setup: precomputes trigonometric tables and configures fan beam parameters
    4. Kernel execution: launches CUDA ray-tracing kernel with divergent ray geometry
    5. Result retrieval: copies computed sinogram back to host memory
    
    Fan beam coordinate system transformations:
    - Source position: rotates around isocenter at distance iso_dist from origin
    - Detector array: positioned at distance src_dist from isocenter, perpendicular to source-isocenter line
    - Ray geometry: divergent rays from point source to individual detector elements
    - Volume centering: image coordinates shifted to center volume at isocenter
    
    Geometric parameters:
    - src_dist: total distance from source to detector array
    - iso_dist: distance from isocenter to source (< src_dist)
    - Detector distance from isocenter: (src_dist - iso_dist)
    - Ray divergence angle determined by detector element position and source location
    
    Algorithm integration:
    - Utilizes _trig_tables() helper for efficient angle preprocessing
    - Employs _grid_2d() helper for optimal CUDA thread organization
    - Executes forward_fan_2d_kernel() for actual divergent ray-tracing computation
    - Handles all GPU memory allocation and data transfer operations
    
    Args:
        image: 2D reconstruction volume as numpy array (height, width)
        n_views: Number of projection angles in the sinogram
        n_det: Number of detector elements per projection
        det_spacing: Physical spacing between detector elements
        angles: Array of projection angles in radians
        src_dist: Distance from X-ray source to detector array
        iso_dist: Distance from isocenter to X-ray source
        
    Returns:
        numpy.ndarray: Computed sinogram with shape (n_views, n_det)
    """
    image_np = image.astype(_DTYPE, copy=False).T
    kernel_Nx, kernel_Ny = image_np.shape
    d_image = cuda.to_device(image_np)
    d_cos, d_sin = _trig_tables(angles, _DTYPE)
    d_sino = cuda.device_array((n_views, n_det), dtype=_DTYPE)

    grid, tpb = _grid_2d(n_views, n_det)
    cx = _DTYPE((kernel_Nx - 1) * 0.5)
    cy = _DTYPE((kernel_Ny - 1) * 0.5)

    forward_fan_2d_kernel[grid, tpb](
        d_image, kernel_Nx, kernel_Ny, d_sino,
        n_views, n_det, _DTYPE(det_spacing),
        d_cos, d_sin,
        _DTYPE(src_dist), _DTYPE(iso_dist),
        cx, cy
    )
    return d_sino.copy_to_host()


def back_fan_2d(
    sinogram: np.ndarray,
    reco_H: int, reco_W: int,
    det_spacing: float,
    angles: np.ndarray,
    src_dist: float,
    iso_dist: float,
):
    """
    Host function for 2D fan beam backprojection using Siddon-Joseph ray-tracing.
    
    This function orchestrates the complete fan beam backprojection pipeline:
    1. Data preparation: converts input sinogram to GPU-compatible format
    2. Memory management: transfers sinogram to GPU and allocates reconstruction volume
    3. Geometry setup: precomputes trigonometric tables and configures fan beam parameters
    4. Kernel execution: launches CUDA backprojection kernel with divergent ray geometry
    5. Result retrieval: copies reconstructed volume back to host memory
    
    Fan beam coordinate system transformations:
    - Source position: rotates around isocenter at distance iso_dist from origin
    - Detector array: positioned at distance src_dist from isocenter, perpendicular to source-isocenter line
    - Ray geometry: divergent rays from point source to individual detector elements
    - Volume initialization: zero-filled reconstruction volume allocated on GPU
    - Output volume: transposed back to standard (height, width) format
    
    Geometric parameters:
    - src_dist: total distance from source to detector array
    - iso_dist: distance from isocenter to source (< src_dist)
    - Detector distance from isocenter: (src_dist - iso_dist)
    - Ray divergence angle determined by detector element position and source location
    
    Algorithm integration:
    - Utilizes _trig_tables() helper for efficient angle preprocessing
    - Employs _grid_2d() helper for optimal CUDA thread organization
    - Executes back_fan_2d_kernel() for actual divergent ray-tracing computation
    - Handles all GPU memory allocation and data transfer operations
    - Uses atomic operations for thread-safe voxel accumulation with higher contention than parallel beam
    
    Args:
        sinogram: Input projection data as numpy array (n_views, n_det)
        reco_H: Height of reconstruction volume in pixels
        reco_W: Width of reconstruction volume in pixels
        det_spacing: Physical spacing between detector elements
        angles: Array of projection angles in radians
        src_dist: Distance from X-ray source to detector array
        iso_dist: Distance from isocenter to X-ray source
        
    Returns:
        numpy.ndarray: Reconstructed volume with shape (reco_H, reco_W)
    """
    sinogram = sinogram.astype(_DTYPE, copy=False)
    n_views, n_det = sinogram.shape
    kernel_Nx, kernel_Ny = int(reco_W), int(reco_H)

    d_sino = cuda.to_device(sinogram)
    d_cos, d_sin = _trig_tables(angles, _DTYPE)
    d_reco = cuda.to_device(np.zeros((kernel_Nx, kernel_Ny), dtype=_DTYPE))

    grid, tpb = _grid_2d(n_views, n_det)
    cx = _DTYPE((kernel_Nx - 1) * 0.5)
    cy = _DTYPE((kernel_Ny - 1) * 0.5)

    back_fan_2d_kernel[grid, tpb](
        d_sino,
        n_views, n_det,
        kernel_Nx, kernel_Ny,
        d_reco,
        _DTYPE(det_spacing),
        d_cos, d_sin,
        _DTYPE(src_dist), _DTYPE(iso_dist),
        cx, cy
    )
    return d_reco.copy_to_host().T


# ------------------------------------------------------------------
# 3-D CONE GEOMETRY (SIDDON-JOSEPH ALGORITHM)
# ------------------------------------------------------------------

@_FASTMATH_DECORATOR
def forward_cone_3d_kernel(
    d_vol,
    Nx, Ny, Nz,
    d_sino,
    n_views, n_u, n_v,
    du, dv,
    d_cos, d_sin,
    src_dist, iso_dist,
    cx, cy, cz
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
    cos_a = d_cos[iview]  # Precomputed cosine of projection angle
    sin_a = d_sin[iview]  # Precomputed sine of projection angle
    u = (iu - (n_u - 1) * 0.5) * du  # Detector u-coordinate (centered)
    v = (iv - (n_v - 1) * 0.5) * dv  # Detector v-coordinate (centered)

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
        d_sino[iview, iu, iv] = 0.0
        return
    
    # Normalize 3D ray direction vector for parametric traversal
    inv_len = 1.0 / length
    dir_x, dir_y, dir_z = dir_x*inv_len, dir_y*inv_len, dir_z*inv_len

    # === 3D RAY-VOLUME INTERSECTION CALCULATION ===
    # Compute intersection with 3D volume boundaries using source position as ray origin
    t_min, t_max = -_INF, _INF  # Ray parameter range for valid volume intersection
    
    # X-direction boundary intersections
    if abs(dir_x) > _EPSILON:
        tx1, tx2 = (-cx - src_x) / dir_x, (cx - src_x) / dir_x  # Volume boundary intersections
        t_min, t_max = max(t_min, min(tx1, tx2)), min(t_max, max(tx1, tx2))  # Update valid parameter range
    elif src_x < -cx or src_x > cx:  # Source outside x-bounds
        d_sino[iview, iu, iv] = 0.0; return
    
    # Y-direction boundary intersections
    if abs(dir_y) > _EPSILON:
        ty1, ty2 = (-cy - src_y) / dir_y, (cy - src_y) / dir_y  # Volume boundary intersections
        t_min, t_max = max(t_min, min(ty1, ty2)), min(t_max, max(ty1, ty2))  # Intersect with x-range
    elif src_y < -cy or src_y > cy:  # Source outside y-bounds
        d_sino[iview, iu, iv] = 0.0; return
    
    # Z-direction boundary intersections (extends 2D algorithm to 3D)
    if abs(dir_z) > _EPSILON:
        tz1, tz2 = (-cz - src_z) / dir_z, (cz - src_z) / dir_z  # Volume boundary intersections
        t_min, t_max = max(t_min, min(tz1, tz2)), min(t_max, max(tz1, tz2))  # Intersect with xy-range
    elif src_z < -cz or src_z > cz:  # Source outside z-bounds
        d_sino[iview, iu, iv] = 0.0; return

    if t_min >= t_max:  # No valid 3D intersection
        d_sino[iview, iu, iv] = 0.0; return

    # === 3D SIDDON-JOSEPH TRAVERSAL INITIALIZATION ===
    t = t_min    # Current ray parameter
    accum = 0.0  # Accumulated projection value
    
    # Convert 3D ray entry point to voxel indices
    ix = int(math.floor(src_x + t * dir_x + cx))  # Current voxel x-index
    iy = int(math.floor(src_y + t * dir_y + cy))  # Current voxel y-index
    iz = int(math.floor(src_z + t * dir_z + cz))  # Current voxel z-index

    # 3D traversal parameters (extends 2D algorithm)
    step_x, step_y, step_z = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1), (1 if dir_z >= 0 else -1)  # Voxel stepping direction
    dt_x = abs(1.0 / dir_x) if abs(dir_x) > _EPSILON else _INF  # Parameter increment per x-voxel
    dt_y = abs(1.0 / dir_y) if abs(dir_y) > _EPSILON else _INF  # Parameter increment per y-voxel
    dt_z = abs(1.0 / dir_z) if abs(dir_z) > _EPSILON else _INF  # Parameter increment per z-voxel
    
    # Calculate parameter values for next 3D voxel boundary crossings
    tx = ((ix + (step_x > 0)) - cx - src_x) / dir_x if abs(dir_x) > _EPSILON else _INF  # Next x-boundary crossing
    ty = ((iy + (step_y > 0)) - cy - src_y) / dir_y if abs(dir_y) > _EPSILON else _INF  # Next y-boundary crossing
    tz = ((iz + (step_z > 0)) - cz - src_z) / dir_z if abs(dir_z) > _EPSILON else _INF  # Next z-boundary crossing

    # === 3D TRAVERSAL LOOP WITH TRILINEAR INTERPOLATION ===
    # Step through voxels along ray path, accumulating weighted contributions
    while t < t_max:
        # Check if current 3D voxel indices are within valid interpolation bounds
        if 0 <= ix < Nx - 1 and 0 <= iy < Ny - 1 and 0 <= iz < Nz - 1:
            # Determine next 3D voxel boundary crossing (minimum of x, y, z boundaries or ray exit)
            t_next = min(tx, ty, tz, t_max)
            seg_len = t_next - t  # Length of ray segment within current voxel region
            
            if seg_len > _EPSILON:  # Only process segments with meaningful length (avoid numerical noise)
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
            ix += step_x  # Move to next voxel in x-direction
            tx += dt_x    # Update next x-boundary crossing parameter
        elif ty <= tx and ty <= tz:    # Y-boundary crossed first
            t = ty
            iy += step_y  # Move to next voxel in y-direction
            ty += dt_y    # Update next y-boundary crossing parameter
        else:                          # Z-boundary crossed first
            t = tz
            iz += step_z  # Move to next voxel in z-direction
            tz += dt_z    # Update next z-boundary crossing parameter

    d_sino[iview, iu, iv] = accum


@_FASTMATH_DECORATOR
def back_cone_3d_kernel(
    d_sino,
    n_views, n_u, n_v,
    Nx, Ny, Nz,
    d_reco,
    du, dv,
    d_cos, d_sin,
    src_dist, iso_dist,
    cx, cy, cz
):
    """
    CUDA kernel implementing the Siddon-Joseph algorithm for 3D cone beam backprojection.
    
    This is the adjoint operation to cone beam forward projection, distributing sinogram values
    back into the volume along the same ray paths. Uses identical ray-tracing logic but with
    atomic operations to handle concurrent writes from multiple threads.
    
    Mathematical basis: Implements the transpose of the cone beam forward projection matrix,
    distributing detector measurements back along divergent 3D ray paths with trilinear interpolation.
    """
    iview, iu, iv = cuda.grid(3)
    if iview >= n_views or iu >= n_u or iv >= n_v:
        return

    # === 3D BACKPROJECTION VALUE AND GEOMETRY SETUP ===
    val = d_sino[iview, iu, iv]  # Sinogram value to backproject along this ray
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
    t_min, t_max = -_INF, _INF  # Ray parameter range for valid volume intersection
    
    # X-direction boundary intersections
    if abs(dir_x) > _EPSILON:
        tx1, tx2 = (-cx - src_x) / dir_x, (cx - src_x) / dir_x  # Volume boundary intersections
        t_min, t_max = max(t_min, min(tx1, tx2)), min(t_max, max(tx1, tx2))  # Update valid parameter range
    elif src_x < -cx or src_x > cx: return  # Source outside x-bounds
    
    # Y-direction boundary intersections
    if abs(dir_y) > _EPSILON:
        ty1, ty2 = (-cy - src_y) / dir_y, (cy - src_y) / dir_y  # Volume boundary intersections
        t_min, t_max = max(t_min, min(ty1, ty2)), min(t_max, max(ty1, ty2))  # Intersect with x-range
    elif src_y < -cy or src_y > cy: return  # Source outside y-bounds
    
    # Z-direction boundary intersections
    if abs(dir_z) > _EPSILON:
        tz1, tz2 = (-cz - src_z) / dir_z, (cz - src_z) / dir_z  # Volume boundary intersections
        t_min, t_max = max(t_min, min(tz1, tz2)), min(t_max, max(tz1, tz2))  # Intersect with xy-range
    elif src_z < -cz or src_z > cz: return  # Source outside z-bounds

    if t_min >= t_max: return  # No valid 3D intersection

    # === 3D SIDDON-JOSEPH TRAVERSAL INITIALIZATION ===
    t = t_min    # Current ray parameter
    ix = int(math.floor(src_x + t * dir_x + cx))  # Current voxel x-index
    iy = int(math.floor(src_y + t * dir_y + cy))  # Current voxel y-index
    iz = int(math.floor(src_z + t * dir_z + cz))  # Current voxel z-index

    # 3D traversal parameters
    step_x, step_y, step_z = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1), (1 if dir_z >= 0 else -1)  # Voxel stepping direction
    dt_x = abs(1.0 / dir_x) if abs(dir_x) > _EPSILON else _INF  # Parameter increment per x-voxel
    dt_y = abs(1.0 / dir_y) if abs(dir_y) > _EPSILON else _INF  # Parameter increment per y-voxel
    dt_z = abs(1.0 / dir_z) if abs(dir_z) > _EPSILON else _INF  # Parameter increment per z-voxel
    
    # Calculate parameter values for next 3D voxel boundary crossings
    tx = ((ix + (step_x > 0)) - cx - src_x) / dir_x if abs(dir_x) > _EPSILON else _INF  # Next x-boundary crossing
    ty = ((iy + (step_y > 0)) - cy - src_y) / dir_y if abs(dir_y) > _EPSILON else _INF  # Next y-boundary crossing
    tz = ((iz + (step_z > 0)) - cz - src_z) / dir_z if abs(dir_z) > _EPSILON else _INF  # Next z-boundary crossing

    # === 3D CONE BEAM BACKPROJECTION TRAVERSAL LOOP ===
    # Distribute sinogram value along divergent 3D ray path using trilinear interpolation
    while t < t_max:
        # Check if current 3D voxel indices are within valid interpolation bounds
        if 0 <= ix < Nx - 1 and 0 <= iy < Ny - 1 and 0 <= iz < Nz - 1:
            # Determine next 3D voxel boundary crossing (minimum of x, y, z boundaries or ray exit)
            t_next = min(tx, ty, tz, t_max)
            seg_len = t_next - t  # Length of ray segment within current voxel region
            
            if seg_len > _EPSILON:  # Only process segments with meaningful length (avoid numerical noise)
                # Sample at ray segment midpoint using source as ray origin
                mid_x = src_x + (t + seg_len * 0.5) * dir_x + cx
                mid_y = src_y + (t + seg_len * 0.5) * dir_y + cy
                mid_z = src_z + (t + seg_len * 0.5) * dir_z + cz
                ix0, iy0, iz0 = int(math.floor(mid_x)), int(math.floor(mid_y)), int(math.floor(mid_z))
                dx, dy, dz = mid_x - ix0, mid_y - iy0, mid_z - iz0
                
                # === ATOMIC BACKPROJECTION WITH TRILINEAR WEIGHTS ===
                # Distribute contribution weighted by segment length and interpolation weights
                # CUDA ATOMIC OPERATIONS: Critical for cone beam backprojection thread safety
                # Cone beam rays converge at source, creating highest probability of voxel write conflicts
                # Atomic operations prevent race conditions when multiple 3D rays write to same voxel
                # Performance consideration: 3D cone beam geometry has highest atomic contention
                cval = val * seg_len  # Contribution value for this ray segment
                # Sample at ray segment midpoint using source as ray origin
                mid_x = src_x + (t + seg_len * 0.5) * dir_x + cx  # Midpoint x-coordinate
                mid_y = src_y + (t + seg_len * 0.5) * dir_y + cy  # Midpoint y-coordinate
                mid_z = src_z + (t + seg_len * 0.5) * dir_z + cz  # Midpoint z-coordinate
                
                # Convert continuous 3D coordinates to voxel indices and interpolation weights
                ix0, iy0, iz0 = int(math.floor(mid_x)), int(math.floor(mid_y)), int(math.floor(mid_z))  # Base voxel indices
                dx, dy, dz = mid_x - ix0, mid_y - iy0, mid_z - iz0  # Fractional parts for 3D weights

                # CUDA 3D ATOMIC OPERATIONS: Most complex atomic pattern in cone beam backprojection
                # 8 atomic writes per ray segment (one per cube corner) increases memory contention significantly
                # Cone beam geometry creates maximum ray convergence, highest probability of write conflicts
                # Performance impact: 3D atomics are most expensive due to volume of concurrent writes
                # Memory bandwidth: 8 atomic operations per interpolation point can saturate memory subsystem
                cuda.atomic.add(d_reco, (ix0,     iy0,     iz0),     cval * (1-dx)*(1-dy)*(1-dz))  # Corner (0,0,0) - atomic write
                cuda.atomic.add(d_reco, (ix0 + 1, iy0,     iz0),     cval * dx*(1-dy)*(1-dz))      # Corner (1,0,0) - atomic write
                cuda.atomic.add(d_reco, (ix0,     iy0 + 1, iz0),     cval * (1-dx)*dy*(1-dz))      # Corner (0,1,0) - atomic write
                cuda.atomic.add(d_reco, (ix0,     iy0,     iz0 + 1), cval * (1-dx)*(1-dy)*dz)      # Corner (0,0,1) - atomic write
                cuda.atomic.add(d_reco, (ix0 + 1, iy0 + 1, iz0),     cval * dx*dy*(1-dz))          # Corner (1,1,0) - atomic write
                cuda.atomic.add(d_reco, (ix0 + 1, iy0,     iz0 + 1), cval * dx*(1-dy)*dz)          # Corner (1,0,1) - atomic write
                cuda.atomic.add(d_reco, (ix0,     iy0 + 1, iz0 + 1), cval * (1-dx)*dy*dz)          # Corner (0,1,1) - atomic write
                cuda.atomic.add(d_reco, (ix0 + 1, iy0 + 1, iz0 + 1), cval * dx*dy*dz)              # Corner (1,1,1) - atomic write

        # === 3D VOXEL BOUNDARY CROSSING LOGIC ===
        # Advance to next voxel based on which boundary is crossed first in 3D
        if tx <= ty and tx <= tz:      # X-boundary crossed first
            t = tx
            ix += step_x  # Move to next voxel in x-direction
            tx += dt_x    # Update next x-boundary crossing parameter
        elif ty <= tx and ty <= tz:    # Y-boundary crossed first
            t = ty
            iy += step_y  # Move to next voxel in y-direction
            ty += dt_y    # Update next y-boundary crossing parameter
        else:                          # Z-boundary crossed first
            t = tz
            iz += step_z  # Move to next voxel in z-direction
            tz += dt_z    # Update next z-boundary crossing parameter


def forward_cone_3d(
    volume: np.ndarray,
    n_views: int,
    n_u: int, n_v: int,
    du: float, dv: float,
    angles: np.ndarray,
    src_dist: float,
    iso_dist: float,
):
    """
    Host function for 3D cone beam forward projection using Siddon-Joseph ray-tracing.
    
    This function orchestrates the complete 3D cone beam forward projection pipeline:
    1. Coordinate system transformation: converts 3D volume to GPU-compatible format with axis transposition
    2. Memory management: transfers volume data to GPU device memory for CUDA processing
    3. Geometry setup: precomputes trigonometric tables and configures 3D cone beam parameters
    4. Kernel execution: launches CUDA ray-tracing kernel with 3D divergent ray geometry
    5. Result retrieval: copies computed 3D sinogram back to host memory
    
    3D cone beam coordinate system transformations:
    - Input volume: standard numpy array with (depth, height, width) indexing
    - GPU processing: transposed to (width, height, depth) for optimal memory access patterns
    - Source position: rotates around isocenter in xy-plane at distance iso_dist from origin
    - 2D detector array: positioned at distance src_dist from isocenter, with (u,v) coordinates
    - Ray geometry: cone of divergent rays from point source to 2D detector array
    - Volume centering: 3D coordinates shifted to center volume at isocenter
    
    Geometric parameters:
    - src_dist: total distance from source to detector array
    - iso_dist: distance from isocenter to source (< src_dist)
    - du, dv: physical spacing between detector elements in u and v directions
    - n_u, n_v: number of detector elements in horizontal and vertical directions
    - Ray divergence determined by detector element position and source location in 3D
    
    Algorithm integration:
    - Utilizes _trig_tables() helper for efficient angle preprocessing
    - Employs _grid_3d() helper for optimal 3D CUDA thread organization
    - Executes forward_cone_3d_kernel() for actual 3D divergent ray-tracing computation
    - Handles all GPU memory allocation and data transfer operations
    - Uses trilinear interpolation for accurate 3D volume sampling
    
    Args:
        volume: 3D reconstruction volume as numpy array (depth, height, width)
        n_views: Number of projection angles in the sinogram
        n_u: Number of detector elements in horizontal direction
        n_v: Number of detector elements in vertical direction
        du: Physical spacing between detector elements in u-direction
        dv: Physical spacing between detector elements in v-direction
        angles: Array of projection angles in radians
        src_dist: Distance from X-ray source to detector array
        iso_dist: Distance from isocenter to X-ray source
        
    Returns:
        numpy.ndarray: Computed 3D sinogram with shape (n_views, n_u, n_v)
    """
    volume_np = volume.astype(_DTYPE, copy=False).transpose((2, 1, 0))
    kernel_Nx, kernel_Ny, kernel_Nz = volume_np.shape
    d_vol = cuda.to_device(volume_np)
    d_cos, d_sin = _trig_tables(angles, _DTYPE)
    d_sino = cuda.device_array((n_views, n_u, n_v), dtype=_DTYPE)

    grid, tpb = _grid_3d(n_views, n_u, n_v)
    cx = _DTYPE((kernel_Nx - 1) * 0.5)
    cy = _DTYPE((kernel_Ny - 1) * 0.5)
    cz = _DTYPE((kernel_Nz - 1) * 0.5)

    forward_cone_3d_kernel[grid, tpb](
        d_vol, kernel_Nx, kernel_Ny, kernel_Nz,
        d_sino,
        n_views, n_u, n_v,
        _DTYPE(du), _DTYPE(dv),
        d_cos, d_sin,
        _DTYPE(src_dist), _DTYPE(iso_dist),
        cx, cy, cz
    )
    return d_sino.copy_to_host()


def back_cone_3d(
    sinogram: np.ndarray,
    reco_D: int, reco_H: int, reco_W: int,
    du: float, dv: float,
    angles: np.ndarray,
    src_dist: float,
    iso_dist: float,
):
    """
    Host function for 3D cone beam backprojection using Siddon-Joseph ray-tracing.
    
    This function orchestrates the complete 3D cone beam backprojection pipeline:
    1. Data preparation: converts input 3D sinogram to GPU-compatible format
    2. Memory management: transfers sinogram to GPU and allocates 3D reconstruction volume
    3. Geometry setup: precomputes trigonometric tables and configures 3D cone beam parameters
    4. Kernel execution: launches CUDA backprojection kernel with 3D divergent ray geometry
    5. Result retrieval: copies reconstructed 3D volume back to host memory
    
    3D cone beam coordinate system transformations:
    - Input sinogram: standard numpy array with (n_views, n_u, n_v) indexing
    - GPU processing: direct transfer without transposition for optimal memory access
    - Volume initialization: zero-filled 3D reconstruction volume allocated on GPU
    - Source position: rotates around isocenter in xy-plane at distance iso_dist from origin
    - 2D detector array: positioned at distance src_dist from isocenter, with (u,v) coordinates
    - Ray geometry: cone of divergent rays from point source to 2D detector array
    - Output volume: transposed back to standard (depth, height, width) format
    
    Geometric parameters:
    - src_dist: total distance from source to detector array
    - iso_dist: distance from isocenter to source (< src_dist)
    - du, dv: physical spacing between detector elements in u and v directions
    - reco_D, reco_H, reco_W: dimensions of reconstruction volume (depth, height, width)
    - Ray divergence determined by detector element position and source location in 3D
    
    Algorithm integration:
    - Utilizes _trig_tables() helper for efficient angle preprocessing
    - Employs _grid_3d() helper for optimal 3D CUDA thread organization
    - Executes back_cone_3d_kernel() for actual 3D divergent ray-tracing computation
    - Handles all GPU memory allocation and data transfer operations
    - Uses atomic operations for thread-safe voxel accumulation with highest contention
    - Employs trilinear interpolation for accurate 3D volume reconstruction
    
    Args:
        sinogram: Input 3D projection data as numpy array (n_views, n_u, n_v)
        reco_D: Depth of reconstruction volume in pixels
        reco_H: Height of reconstruction volume in pixels
        reco_W: Width of reconstruction volume in pixels
        du: Physical spacing between detector elements in u-direction
        dv: Physical spacing between detector elements in v-direction
        angles: Array of projection angles in radians
        src_dist: Distance from X-ray source to detector array
        iso_dist: Distance from isocenter to X-ray source
        
    Returns:
        numpy.ndarray: Reconstructed 3D volume with shape (reco_D, reco_H, reco_W)
    """
    sinogram = sinogram.astype(_DTYPE, copy=False)
    n_views, n_u, n_v = sinogram.shape
    kernel_Nx, kernel_Ny, kernel_Nz = int(reco_W), int(reco_H), int(reco_D)

    d_sino = cuda.to_device(sinogram)
    d_cos, d_sin = _trig_tables(angles, _DTYPE)
    d_reco = cuda.to_device(np.zeros((kernel_Nx, kernel_Ny, kernel_Nz), dtype=_DTYPE))

    grid, tpb = _grid_3d(n_views, n_u, n_v)
    cx = _DTYPE((kernel_Nx - 1) * 0.5)
    cy = _DTYPE((kernel_Ny - 1) * 0.5)
    cz = _DTYPE((kernel_Nz - 1) * 0.5)

    back_cone_3d_kernel[grid, tpb](
        d_sino,
        n_views, n_u, n_v,
        kernel_Nx, kernel_Ny, kernel_Nz,
        d_reco,
        _DTYPE(du), _DTYPE(dv),
        d_cos, d_sin,
        _DTYPE(src_dist), _DTYPE(iso_dist),
        cx, cy, cz
    )
    return d_reco.copy_to_host().transpose((2, 1, 0))