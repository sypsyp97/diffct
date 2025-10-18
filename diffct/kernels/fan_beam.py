"""CUDA kernels for 2D fan beam projections.

This module contains CUDA kernels implementing the Siddon ray-tracing method
for 2D fan beam forward projection and backprojection.
"""

import math
from numba import cuda

from ..constants import _FASTMATH_DECORATOR, _INF, _EPSILON


# ============================================================================
# 2D Fan Beam Forward Projection Kernel
# ============================================================================

@_FASTMATH_DECORATOR
def _fan_2d_forward_kernel(
    d_image, Nx, Ny,
    d_sino, n_ang, n_det,
    det_spacing, d_src_pos, d_det_center, d_det_u_vec,
    cx, cy, voxel_spacing
):
    """Compute the 2D fan beam forward projection with arbitrary source-detector trajectories.

    This CUDA kernel implements the Siddon ray-tracing method with interpolation for
    2D fan beam forward projection. Supports arbitrary source and detector positions
    for each view, enabling non-circular trajectories.

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
    d_src_pos : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Source positions for each view, shape (n_ang, 2), in physical units.
    d_det_center : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Detector center positions for each view, shape (n_ang, 2), in physical units.
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
    Supports arbitrary fan beam geometries by specifying source position,
    detector center, and detector orientation vectors for each view.
    Uses bilinear interpolation for accurate volumetric sampling.
    """
    iang, idet = cuda.grid(2)
    if iang >= n_ang or idet >= n_det:
        return

    # === 2D FAN BEAM GEOMETRY SETUP (ARBITRARY TRAJECTORY) ===
    # Read source position from position matrix (in physical units)
    src_x = d_src_pos[iang, 0] / voxel_spacing
    src_y = d_src_pos[iang, 1] / voxel_spacing

    # Read detector center and orientation vector
    det_cx = d_det_center[iang, 0] / voxel_spacing
    det_cy = d_det_center[iang, 1] / voxel_spacing

    u_vec_x = d_det_u_vec[iang, 0]
    u_vec_y = d_det_u_vec[iang, 1]

    # Calculate detector element offset from center
    u_offset = (idet - n_det * 0.5) * det_spacing / voxel_spacing

    # Calculate 2D detector element position using center + u*u_vec
    det_x = det_cx + u_offset * u_vec_x
    det_y = det_cy + u_offset * u_vec_y

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

    # === SIDDON METHOD TRAVERSAL (same algorithm as parallel beam) ===
    accum = 0.0  # Accumulated projection value
    t = t_min    # Current ray parameter
    
    # Convert ray entry point to voxel indices (using source as ray origin)
    ix = int(math.floor(src_x + t * dir_x + cx))
    iy = int(math.floor(src_y + t * dir_y + cy))

    # Traversal parameters (identical to parallel beam implementation)
    step_x, step_y = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1)
    inv_dir_x = (1.0 / dir_x) if abs(dir_x) > _EPSILON else 0.0
    inv_dir_y = (1.0 / dir_y) if abs(dir_y) > _EPSILON else 0.0
    dt_x = abs(inv_dir_x) if abs(dir_x) > _EPSILON else _INF
    dt_y = abs(inv_dir_y) if abs(dir_y) > _EPSILON else _INF
    tx = ((ix + (step_x > 0)) - cx - src_x) * inv_dir_x if abs(dir_x) > _EPSILON else _INF
    ty = ((iy + (step_y > 0)) - cy - src_y) * inv_dir_y if abs(dir_y) > _EPSILON else _INF

    # Main traversal loop with bilinear interpolation (identical to parallel beam)
    while t < t_max:
        if 0 <= ix < Nx and 0 <= iy < Ny:
            t_next = min(tx, ty, t_max)
            seg_len = t_next - t
            if seg_len > _EPSILON:
                # Sample at midpoint using source as ray origin
                t_mid = t + seg_len * 0.5
                mid_x = src_x + t_mid * dir_x + cx
                mid_y = src_y + t_mid * dir_y + cy
                ix0, iy0 = int(math.floor(mid_x)), int(math.floor(mid_y))
                dx, dy = mid_x - ix0, mid_y - iy0
                
                # Clamp indices to stay in-bounds during interpolation
                ix0 = max(0, min(ix0, Nx - 2))
                iy0 = max(0, min(iy0, Ny - 2))
                
                # Bilinear interpolation (identical to parallel beam)
                one_minus_dx = 1.0 - dx
                one_minus_dy = 1.0 - dy
                v00 = d_image[iy0, ix0]
                v10 = d_image[iy0, ix0 + 1]
                v01 = d_image[iy0 + 1, ix0]
                v11 = d_image[iy0 + 1, ix0 + 1]
                row0 = (v00 * one_minus_dx + v10 * dx) * one_minus_dy
                row1 = (v01 * one_minus_dx + v11 * dx) * dy
                val = row0 + row1
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

# ============================================================================
# 2D Fan Beam Backprojection Kernel
# ============================================================================

def _fan_2d_backward_kernel(
    d_sino, n_ang, n_det,
    d_image, Nx, Ny,
    det_spacing, d_src_pos, d_det_center, d_det_u_vec,
    cx, cy, voxel_spacing
):
    """Compute the 2D fan beam backprojection with arbitrary source-detector trajectories.

    This CUDA kernel implements the Siddon ray-tracing method with interpolation for
    2D fan beam backprojection. Supports arbitrary source and detector positions
    for each view, enabling non-circular trajectories.

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
    d_src_pos : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Source positions for each view, shape (n_ang, 2), in physical units.
    d_det_center : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Detector center positions for each view, shape (n_ang, 2), in physical units.
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
    As the adjoint to the fan beam forward projection, this operation
    distributes sinogram values back into the volume along divergent ray
    paths using atomic operations for thread-safe accumulation.
    Supports arbitrary fan beam geometries.
    """
    iang, idet = cuda.grid(2)
    if iang >= n_ang or idet >= n_det:
        return

    # === 2D BACKPROJECTION VALUE AND GEOMETRY SETUP (ARBITRARY TRAJECTORY) ===
    val = d_sino[iang, idet]  # Sinogram value to backproject along this ray

    # Read source position from position matrix (in physical units)
    src_x = d_src_pos[iang, 0] / voxel_spacing
    src_y = d_src_pos[iang, 1] / voxel_spacing

    # Read detector center and orientation vector
    det_cx = d_det_center[iang, 0] / voxel_spacing
    det_cy = d_det_center[iang, 1] / voxel_spacing

    u_vec_x = d_det_u_vec[iang, 0]
    u_vec_y = d_det_u_vec[iang, 1]

    # Calculate detector element offset from center
    u_offset = (idet - n_det * 0.5) * det_spacing / voxel_spacing

    # Calculate 2D detector element position using center + u*u_vec
    det_x = det_cx + u_offset * u_vec_x
    det_y = det_cy + u_offset * u_vec_y

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

    # === SIDDON METHOD TRAVERSAL INITIALIZATION ===
    t = t_min
    ix = int(math.floor(src_x + t * dir_x + cx))
    iy = int(math.floor(src_y + t * dir_y + cy))

    step_x, step_y = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1)
    inv_dir_x = (1.0 / dir_x) if abs(dir_x) > _EPSILON else 0.0
    inv_dir_y = (1.0 / dir_y) if abs(dir_y) > _EPSILON else 0.0
    dt_x = abs(inv_dir_x) if abs(dir_x) > _EPSILON else _INF
    dt_y = abs(inv_dir_y) if abs(dir_y) > _EPSILON else _INF
    tx = ((ix + (step_x > 0)) - cx - src_x) * inv_dir_x if abs(dir_x) > _EPSILON else _INF
    ty = ((iy + (step_y > 0)) - cy - src_y) * inv_dir_y if abs(dir_y) > _EPSILON else _INF

    # === FAN BEAM BACKPROJECTION TRAVERSAL LOOP ===
    # Distribute sinogram value along divergent ray path using bilinear interpolation
    while t < t_max:
        if 0 <= ix < Nx and 0 <= iy < Ny:
            t_next = min(tx, ty, t_max)
            seg_len = t_next - t
            if seg_len > _EPSILON:
                # Sample at ray segment midpoint using source as ray origin
                t_mid = t + seg_len * 0.5
                mid_x = src_x + t_mid * dir_x + cx
                mid_y = src_y + t_mid * dir_y + cy
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
                one_minus_dx = 1.0 - dx
                one_minus_dy = 1.0 - dy
                cuda.atomic.add(d_image, (iy0,     ix0),     cval * one_minus_dx * one_minus_dy)
                cuda.atomic.add(d_image, (iy0,     ix0 + 1), cval * dx          * one_minus_dy)
                cuda.atomic.add(d_image, (iy0 + 1, ix0),     cval * one_minus_dx * dy)
                cuda.atomic.add(d_image, (iy0 + 1, ix0 + 1), cval * dx          * dy)

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

