"""CUDA kernels for 2D parallel beam projections.

This module contains CUDA kernels implementing the Siddon ray-tracing method
for 2D parallel beam forward projection and backprojection.
"""

import math
from numba import cuda

from ..constants import _FASTMATH_DECORATOR, _INF, _EPSILON


# ============================================================================
# 2D Parallel Beam Forward Projection Kernel
# ============================================================================

@_FASTMATH_DECORATOR
def _parallel_2d_forward_kernel(
    d_image, Nx, Ny,
    d_sino, n_ang, n_det,
    det_spacing, d_ray_dir, d_det_origin, d_det_u_vec, cx, cy, voxel_spacing
):
    """Compute the 2D parallel beam forward projection with arbitrary ray trajectories.

    This CUDA kernel implements the Siddon ray-tracing method with interpolation for
    2D parallel beam forward projection. Supports arbitrary ray directions and detector
    positions for each view, enabling non-circular trajectories.

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
    d_ray_dir : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Ray direction unit vectors for each view, shape (n_ang, 2).
    d_det_origin : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Detector origin positions for each view, shape (n_ang, 2), in physical units.
    d_det_u_vec : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Detector u-direction unit vectors for each view, shape (n_ang, 2).
    cx : float
        Half of image width in voxels.
    cy : float
        Half of image height in voxels.
    voxel_spacing : float
        Physical size of one voxel (in same units as det_spacing).

    Notes
    -----
    Supports arbitrary parallel beam geometries by specifying ray direction,
    detector origin, and detector orientation vectors for each view.
    Uses bilinear interpolation for accurate volumetric sampling.
    """
    iang, idet = cuda.grid(2)
    if iang >= n_ang or idet >= n_det:
        return

    # === 2D PARALLEL BEAM GEOMETRY SETUP (ARBITRARY TRAJECTORY) ===
    # Read ray direction (parallel for all rays in this view)
    dir_x = d_ray_dir[iang, 0]
    dir_y = d_ray_dir[iang, 1]

    # Read detector origin and orientation vector
    det_ox = d_det_origin[iang, 0] / voxel_spacing
    det_oy = d_det_origin[iang, 1] / voxel_spacing

    u_vec_x = d_det_u_vec[iang, 0]
    u_vec_y = d_det_u_vec[iang, 1]

    # Calculate detector element offset from origin
    u_offset = (idet - n_det * 0.5) * det_spacing / voxel_spacing

    # Ray starting point: detector origin + offset along u-direction
    pnt_x = det_ox + u_offset * u_vec_x
    pnt_y = det_oy + u_offset * u_vec_y

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

    # === SIDDON METHOD VOXEL TRAVERSAL INITIALIZATION ===
    accum = 0.0  # Accumulated projection value along ray
    t = t_min    # Current ray parameter (distance from ray start)
    
    # Convert ray entry point to voxel indices (image coordinate system)
    ix = int(math.floor(pnt_x + t * dir_x + cx))  # Current voxel x-index
    iy = int(math.floor(pnt_y + t * dir_y + cy))  # Current voxel y-index

    # Determine traversal direction and step sizes for each axis
    step_x, step_y = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1)  # Voxel stepping direction
    # Hoist inverse directions to reduce divisions and branches
    inv_dir_x = (1.0 / dir_x) if abs(dir_x) > _EPSILON else 0.0
    inv_dir_y = (1.0 / dir_y) if abs(dir_y) > _EPSILON else 0.0
    dt_x = abs(inv_dir_x) if abs(dir_x) > _EPSILON else _INF
    dt_y = abs(inv_dir_y) if abs(dir_y) > _EPSILON else _INF

    # Calculate parameter values for next voxel boundary crossings using inv_dir_*
    tx = ((ix + (step_x > 0)) - cx - pnt_x) * inv_dir_x if abs(dir_x) > _EPSILON else _INF
    ty = ((iy + (step_y > 0)) - cy - pnt_y) * inv_dir_y if abs(dir_y) > _EPSILON else _INF

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
                t_mid = t + seg_len * 0.5
                mid_x = pnt_x + t_mid * dir_x + cx  # Midpoint x-coordinate in image space
                mid_y = pnt_y + t_mid * dir_y + cy  # Midpoint y-coordinate in image space

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
                one_minus_dx = 1.0 - dx
                one_minus_dy = 1.0 - dy
                v00 = d_image[iy0, ix0]
                v10 = d_image[iy0, ix0 + 1]
                v01 = d_image[iy0 + 1, ix0]
                v11 = d_image[iy0 + 1, ix0 + 1]
                row0 = (v00 * one_minus_dx + v10 * dx) * one_minus_dy
                row1 = (v01 * one_minus_dx + v11 * dx) * dy
                val = row0 + row1
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


# ============================================================================
# 2D Parallel Beam Backprojection Kernel
# ============================================================================

@_FASTMATH_DECORATOR
def _parallel_2d_backward_kernel(
    d_sino, n_ang, n_det,
    d_image, Nx, Ny,
    det_spacing, d_ray_dir, d_det_origin, d_det_u_vec, cx, cy, voxel_spacing
):
    """Compute the 2D parallel beam backprojection with arbitrary ray trajectories.

    This CUDA kernel implements the Siddon ray-tracing method with interpolation for
    2D parallel beam backprojection. Supports arbitrary ray directions and detector
    positions for each view, enabling non-circular trajectories.

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
    d_ray_dir : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Ray direction unit vectors for each view, shape (n_ang, 2).
    d_det_origin : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Detector origin positions for each view, shape (n_ang, 2), in physical units.
    d_det_u_vec : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Detector u-direction unit vectors for each view, shape (n_ang, 2).
    cx : float
        Half of image width in voxels.
    cy : float
        Half of image height in voxels.
    voxel_spacing : float
        Physical size of one voxel (in same units as det_spacing).

    Notes
    -----
    This operation is the adjoint of the forward projection. Sinogram values
    are distributed back into the volume along identical ray paths using
    atomic operations to ensure thread-safe accumulation.
    Supports arbitrary parallel beam geometries.
    """
    iang, idet = cuda.grid(2)
    if iang >= n_ang or idet >= n_det:
        return

    # === 2D BACKPROJECTION VALUE AND GEOMETRY SETUP (ARBITRARY TRAJECTORY) ===
    val = d_sino[iang, idet]  # Sinogram value to backproject

    # Read ray direction (parallel for all rays in this view)
    dir_x = d_ray_dir[iang, 0]
    dir_y = d_ray_dir[iang, 1]

    # Read detector origin and orientation vector
    det_ox = d_det_origin[iang, 0] / voxel_spacing
    det_oy = d_det_origin[iang, 1] / voxel_spacing

    u_vec_x = d_det_u_vec[iang, 0]
    u_vec_y = d_det_u_vec[iang, 1]

    # Calculate detector element offset from origin
    u_offset = (idet - n_det * 0.5) * det_spacing / voxel_spacing

    # Ray starting point: detector origin + offset along u-direction
    pnt_x = det_ox + u_offset * u_vec_x
    pnt_y = det_oy + u_offset * u_vec_y

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

    # === SIDDON METHOD TRAVERSAL INITIALIZATION ===
    t = t_min
    ix = int(math.floor(pnt_x + t * dir_x + cx))
    iy = int(math.floor(pnt_y + t * dir_y + cy))

    step_x, step_y = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1)
    inv_dir_x = (1.0 / dir_x) if abs(dir_x) > _EPSILON else 0.0
    inv_dir_y = (1.0 / dir_y) if abs(dir_y) > _EPSILON else 0.0
    dt_x = abs(inv_dir_x) if abs(dir_x) > _EPSILON else _INF
    dt_y = abs(inv_dir_y) if abs(dir_y) > _EPSILON else _INF
    tx = ((ix + (step_x > 0)) - cx - pnt_x) * inv_dir_x if abs(dir_x) > _EPSILON else _INF
    ty = ((iy + (step_y > 0)) - cy - pnt_y) * inv_dir_y if abs(dir_y) > _EPSILON else _INF

    # === BACKPROJECTION TRAVERSAL LOOP ===
    # Distribute sinogram value along ray path using bilinear interpolation
    while t < t_max:
        if 0 <= ix < Nx and 0 <= iy < Ny:
            t_next = min(tx, ty, t_max)
            seg_len = t_next - t
            if seg_len > _EPSILON:
                # Sample at ray segment midpoint (same as forward projection)
                t_mid = t + seg_len * 0.5
                mid_x = pnt_x + t_mid * dir_x + cx
                mid_y = pnt_y + t_mid * dir_y + cy
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
                one_minus_dx = 1.0 - dx
                one_minus_dy = 1.0 - dy
                cuda.atomic.add(d_image, (iy0,     ix0),     cval * one_minus_dx * one_minus_dy)
                cuda.atomic.add(d_image, (iy0,     ix0 + 1), cval * dx          * one_minus_dy)
                cuda.atomic.add(d_image, (iy0 + 1, ix0),     cval * one_minus_dx * dy)
                cuda.atomic.add(d_image, (iy0 + 1, ix0 + 1), cval * dx          * dy)

        # Advance to next voxel (identical logic to forward projection)
        if tx <= ty:
            t = tx
            ix += step_x
            tx += dt_x
        else:
            t = ty
            iy += step_y
            ty += dt_y

