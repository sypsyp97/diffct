"""CUDA kernels for 3D cone beam projections.

This module contains CUDA kernels implementing the Siddon ray-tracing method
for 3D cone beam forward projection and backprojection.
"""

import math
from numba import cuda

from ..constants import _FASTMATH_DECORATOR, _INF, _EPSILON


# ============================================================================
# 3D Cone Beam Forward Projection Kernel
# ============================================================================

@_FASTMATH_DECORATOR
def _cone_3d_forward_kernel(
    d_vol, Nx, Ny, Nz,
    d_sino, n_views, n_u, n_v,
    du, dv, d_src_pos, d_det_center, d_det_u_vec, d_det_v_vec,
    cx, cy, cz, voxel_spacing
):
    """Compute the 3D cone-beam forward projection with arbitrary source-detector trajectories.

    This CUDA kernel implements the Siddon ray-tracing method with interpolation for
    3D cone-beam forward projection. Supports arbitrary source and detector positions
    for each view, enabling non-circular trajectories.

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
    d_src_pos : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Source positions for each view, shape (n_views, 3), in physical units.
    d_det_center : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Detector center positions for each view, shape (n_views, 3), in physical units.
    d_det_u_vec : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Detector u-direction unit vectors for each view, shape (n_views, 3).
    d_det_v_vec : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Detector v-direction unit vectors for each view, shape (n_views, 3).
    cx : float
        Half of volume width along x-axis (in voxels).
    cy : float
        Half of volume height along y-axis (in voxels).
    cz : float
        Half of volume depth along z-axis (in voxels).
    voxel_spacing : float
        Physical size of one voxel (in same units as source/detector positions).

    Notes
    -----
    Supports arbitrary cone-beam geometries by specifying source position,
    detector center, and detector orientation vectors for each view.
    Uses trilinear interpolation for accurate volumetric sampling.
    """
    iview, iu, iv = cuda.grid(3)
    if iview >= n_views or iu >= n_u or iv >= n_v:
        return

    # === 3D CONE BEAM GEOMETRY SETUP (ARBITRARY TRAJECTORY) ===
    # Read source position from position matrix (in physical units)
    src_x = d_src_pos[iview, 0] / voxel_spacing
    src_y = d_src_pos[iview, 1] / voxel_spacing
    src_z = d_src_pos[iview, 2] / voxel_spacing

    # Read detector center and orientation vectors
    det_cx = d_det_center[iview, 0] / voxel_spacing
    det_cy = d_det_center[iview, 1] / voxel_spacing
    det_cz = d_det_center[iview, 2] / voxel_spacing

    u_vec_x = d_det_u_vec[iview, 0]
    u_vec_y = d_det_u_vec[iview, 1]
    u_vec_z = d_det_u_vec[iview, 2]

    v_vec_x = d_det_v_vec[iview, 0]
    v_vec_y = d_det_v_vec[iview, 1]
    v_vec_z = d_det_v_vec[iview, 2]

    # Calculate detector element offset from center
    u_offset = (iu - n_u * 0.5) * du / voxel_spacing
    v_offset = (iv - n_v * 0.5) * dv / voxel_spacing

    # Calculate 3D detector element position using center + u*u_vec + v*v_vec
    det_x = det_cx + u_offset * u_vec_x + v_offset * v_vec_x
    det_y = det_cy + u_offset * u_vec_y + v_offset * v_vec_y
    det_z = det_cz + u_offset * u_vec_z + v_offset * v_vec_z

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

    # === 3D SIDDON METHOD TRAVERSAL INITIALIZATION ===
    accum = 0.0  # Accumulated projection value
    t = t_min    # Current ray parameter
    
    # Convert 3D ray entry point to voxel indices
    ix = int(math.floor(src_x + t * dir_x + cx))  # Current voxel x-index
    iy = int(math.floor(src_y + t * dir_y + cy))  # Current voxel y-index
    iz = int(math.floor(src_z + t * dir_z + cz))  # Current voxel z-index

    # 3D traversal parameters (extends 2D algorithm)
    step_x, step_y, step_z = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1), (1 if dir_z >= 0 else -1)
    inv_dir_x = (1.0 / dir_x) if abs(dir_x) > _EPSILON else 0.0
    inv_dir_y = (1.0 / dir_y) if abs(dir_y) > _EPSILON else 0.0
    inv_dir_z = (1.0 / dir_z) if abs(dir_z) > _EPSILON else 0.0
    dt_x = abs(inv_dir_x) if abs(dir_x) > _EPSILON else _INF  # Parameter increment per x-voxel
    dt_y = abs(inv_dir_y) if abs(dir_y) > _EPSILON else _INF  # Parameter increment per y-voxel
    dt_z = abs(inv_dir_z) if abs(dir_z) > _EPSILON else _INF  # Parameter increment per z-voxel

    # Calculate parameter values for next 3D voxel boundary crossings
    tx = ((ix + (step_x > 0)) - cx - src_x) * inv_dir_x if abs(dir_x) > _EPSILON else _INF
    ty = ((iy + (step_y > 0)) - cy - src_y) * inv_dir_y if abs(dir_y) > _EPSILON else _INF
    tz = ((iz + (step_z > 0)) - cz - src_z) * inv_dir_z if abs(dir_z) > _EPSILON else _INF

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
                t_mid = t + seg_len * 0.5
                mid_x = src_x + t_mid * dir_x + cx  # Midpoint x-coordinate in volume space
                mid_y = src_y + t_mid * dir_y + cy  # Midpoint y-coordinate in volume space
                mid_z = src_z + t_mid * dir_z + cz  # Midpoint z-coordinate in volume space

                # Convert continuous 3D coordinates to discrete voxel indices and fractional weights
                ix0, iy0, iz0 = int(math.floor(mid_x)), int(math.floor(mid_y)), int(math.floor(mid_z))
                dx, dy, dz = mid_x - ix0, mid_y - iy0, mid_z - iz0

                # Clamp indices to stay in-bounds during interpolation
                ix0 = max(0, min(ix0, Nx - 2))
                iy0 = max(0, min(iy0, Ny - 2))
                iz0 = max(0, min(iz0, Nz - 2))

                # Precompute complements
                omdx = 1.0 - dx
                omdy = 1.0 - dy
                omdz = 1.0 - dz

                # === TRILINEAR INTERPOLATION WEIGHT CALCULATION ===
                val = (
                    d_vol[ix0,     iy0,     iz0]     * omdx*omdy*omdz +
                    d_vol[ix0 + 1, iy0,     iz0]     * dx  *omdy*omdz +
                    d_vol[ix0,     iy0 + 1, iz0]     * omdx*dy  *omdz +
                    d_vol[ix0,     iy0,     iz0 + 1] * omdx*omdy*dz   +
                    d_vol[ix0 + 1, iy0 + 1, iz0]     * dx  *dy  *omdz +
                    d_vol[ix0 + 1, iy0,     iz0 + 1] * dx  *omdy*dz   +
                    d_vol[ix0,     iy0 + 1, iz0 + 1] * omdx*dy  *dz   +
                    d_vol[ix0 + 1, iy0 + 1, iz0 + 1] * dx  *dy  *dz
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

# ============================================================================
# 3D Cone Beam Backprojection Kernel
# ============================================================================

def _cone_3d_backward_kernel(
    d_sino, n_views, n_u, n_v,
    d_vol, Nx, Ny, Nz,
    du, dv, d_src_pos, d_det_center, d_det_u_vec, d_det_v_vec,
    cx, cy, cz, voxel_spacing
):
    """Compute the 3D cone-beam backprojection with arbitrary source-detector trajectories.

    This CUDA kernel implements the Siddon ray-tracing method with interpolation for
    3D cone-beam backprojection. Supports arbitrary source and detector positions
    for each view, enabling non-circular trajectories.

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
    d_src_pos : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Source positions for each view, shape (n_views, 3), in physical units.
    d_det_center : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Detector center positions for each view, shape (n_views, 3), in physical units.
    d_det_u_vec : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Detector u-direction unit vectors for each view, shape (n_views, 3).
    d_det_v_vec : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Detector v-direction unit vectors for each view, shape (n_views, 3).
    cx : float
        Half of volume width along x-axis (in voxels).
    cy : float
        Half of volume height along y-axis (in voxels).
    cz : float
        Half of volume depth along z-axis (in voxels).
    voxel_spacing : float
        Physical size of one voxel (in same units as source/detector positions).

    Notes
    -----
    As the adjoint to the cone-beam forward projection, this operation
    distributes sinogram values back into the 3D volume along ray paths using
    atomic operations for thread-safe accumulation.
    Supports arbitrary cone-beam geometries.
    """
    iview, iu, iv = cuda.grid(3)
    if iview >= n_views or iu >= n_u or iv >= n_v:
        return

    # === 3D BACKPROJECTION VALUE AND GEOMETRY SETUP (ARBITRARY TRAJECTORY) ===
    g = d_sino[iview, iu, iv]  # Sinogram value to backproject along this ray

    # Read source position from position matrix (in physical units)
    src_x = d_src_pos[iview, 0] / voxel_spacing
    src_y = d_src_pos[iview, 1] / voxel_spacing
    src_z = d_src_pos[iview, 2] / voxel_spacing

    # Read detector center and orientation vectors
    det_cx = d_det_center[iview, 0] / voxel_spacing
    det_cy = d_det_center[iview, 1] / voxel_spacing
    det_cz = d_det_center[iview, 2] / voxel_spacing

    u_vec_x = d_det_u_vec[iview, 0]
    u_vec_y = d_det_u_vec[iview, 1]
    u_vec_z = d_det_u_vec[iview, 2]

    v_vec_x = d_det_v_vec[iview, 0]
    v_vec_y = d_det_v_vec[iview, 1]
    v_vec_z = d_det_v_vec[iview, 2]

    # Calculate detector element offset from center
    u_offset = (iu - n_u * 0.5) * du / voxel_spacing
    v_offset = (iv - n_v * 0.5) * dv / voxel_spacing

    # Calculate 3D detector element position using center + u*u_vec + v*v_vec
    det_x = det_cx + u_offset * u_vec_x + v_offset * v_vec_x
    det_y = det_cy + u_offset * u_vec_y + v_offset * v_vec_y
    det_z = det_cz + u_offset * u_vec_z + v_offset * v_vec_z

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

    # === 3D SIDDON METHOD TRAVERSAL INITIALIZATION ===
    t = t_min
    ix = int(math.floor(src_x + t * dir_x + cx))  # Current voxel x-index
    iy = int(math.floor(src_y + t * dir_y + cy))  # Current voxel y-index
    iz = int(math.floor(src_z + t * dir_z + cz))  # Current voxel z-index

    # 3D traversal parameters (extends 2D algorithm)
    step_x, step_y, step_z = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1), (1 if dir_z >= 0 else -1)
    inv_dir_x = (1.0 / dir_x) if abs(dir_x) > _EPSILON else 0.0
    inv_dir_y = (1.0 / dir_y) if abs(dir_y) > _EPSILON else 0.0
    inv_dir_z = (1.0 / dir_z) if abs(dir_z) > _EPSILON else 0.0
    dt_x = abs(inv_dir_x) if abs(dir_x) > _EPSILON else _INF  # Parameter increment per x-voxel
    dt_y = abs(inv_dir_y) if abs(dir_y) > _EPSILON else _INF  # Parameter increment per y-voxel
    dt_z = abs(inv_dir_z) if abs(dir_z) > _EPSILON else _INF  # Parameter increment per z-voxel

    # Calculate parameter values for next 3D voxel boundary crossings
    tx = ((ix + (step_x > 0)) - cx - src_x) * inv_dir_x if abs(dir_x) > _EPSILON else _INF
    ty = ((iy + (step_y > 0)) - cy - src_y) * inv_dir_y if abs(dir_y) > _EPSILON else _INF
    tz = ((iz + (step_z > 0)) - cz - src_z) * inv_dir_z if abs(dir_z) > _EPSILON else _INF

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
                t_mid = t + seg_len * 0.5
                mid_x = src_x + t_mid * dir_x + cx
                mid_y = src_y + t_mid * dir_y + cy
                mid_z = src_z + t_mid * dir_z + cz

                # Convert continuous 3D coordinates to voxel indices and interpolation weights
                ix0, iy0, iz0 = int(math.floor(mid_x)), int(math.floor(mid_y)), int(math.floor(mid_z))
                dx, dy, dz = mid_x - ix0, mid_y - iy0, mid_z - iz0

                # Clamp indices to stay in-bounds during interpolation
                ix0 = max(0, min(ix0, Nx - 2))
                iy0 = max(0, min(iy0, Ny - 2))
                iz0 = max(0, min(iz0, Nz - 2))

                # Precompute complements and contribution
                omdx = 1.0 - dx
                omdy = 1.0 - dy
                omdz = 1.0 - dz
                cval = g * seg_len

                # === ATOMIC BACKPROJECTION WITH TRILINEAR WEIGHTS ===
                cuda.atomic.add(d_vol, (ix0,     iy0,     iz0),     cval * omdx*omdy*omdz)
                cuda.atomic.add(d_vol, (ix0 + 1, iy0,     iz0),     cval * dx  *omdy*omdz)
                cuda.atomic.add(d_vol, (ix0,     iy0 + 1, iz0),     cval * omdx*dy  *omdz)
                cuda.atomic.add(d_vol, (ix0,     iy0,     iz0 + 1), cval * omdx*omdy*dz)
                cuda.atomic.add(d_vol, (ix0 + 1, iy0 + 1, iz0),     cval * dx  *dy  *omdz)
                cuda.atomic.add(d_vol, (ix0 + 1, iy0,     iz0 + 1), cval * dx  *omdy*dz)
                cuda.atomic.add(d_vol, (ix0,     iy0 + 1, iz0 + 1), cval * omdx*dy  *dz)
                cuda.atomic.add(d_vol, (ix0 + 1, iy0 + 1, iz0 + 1), cval * dx  *dy  *dz)

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

