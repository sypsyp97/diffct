"""CUDA kernels for 3D cone beam projections.

This module contains CUDA kernels implementing the Siddon ray-tracing method
for 3D cone beam forward projection and backprojection, plus a dedicated
voxel-driven FDK gather kernel for analytical reconstruction pipelines.
"""

import math
import numpy as np
from numba import cuda

from ..constants import (
    _FASTMATH_DECORATOR,
    _FDK_ACCURACY_DECORATOR,
    _BIG,
    _TINY,
    _ZERO,
    _ONE,
    _HALF,
    _EPSILON,
)


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

    This CUDA kernel implements cell-constant Siddon ray tracing for
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
    Integrates each ray through piecewise-constant voxel cells.
    """
    iv, iu, iview = cuda.grid(3)
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
    u_offset = (np.float32(iu) + _HALF - np.float32(n_u) * _HALF) * du / voxel_spacing
    v_offset = (np.float32(iv) + _HALF - np.float32(n_v) * _HALF) * dv / voxel_spacing

    # Calculate 3D detector element position using center + u*u_vec + v*v_vec
    det_x = det_cx + u_offset * u_vec_x + v_offset * v_vec_x
    det_y = det_cy + u_offset * u_vec_y + v_offset * v_vec_y
    det_z = det_cz + u_offset * u_vec_z + v_offset * v_vec_z

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
    # Measure the ray parameter from the endpoint nearer the volume centre, so a
    # distant source does not make the float32 parameters coarse. Only the
    # source-to-detector segment is integrated.
    if src_x * src_x + src_y * src_y + src_z * src_z <= det_x * det_x + det_y * det_y + det_z * det_z:
        org_x = src_x
        org_y = src_y
        org_z = src_z
        t_min, t_max = _ZERO, length
    else:
        org_x = det_x
        org_y = det_y
        org_z = det_z
        t_min, t_max = -length, _ZERO
    
    # X-direction boundary intersections
    if abs(dir_x) > _TINY:
        tx1, tx2 = (-cx - org_x) / dir_x, (cx - org_x) / dir_x
        t_min, t_max = max(t_min, min(tx1, tx2)), min(t_max, max(tx1, tx2))
    elif org_x < -cx or org_x > cx:  # Ray origin outside x-bounds
        d_sino[iview, iu, iv] = _ZERO; return
    
    # Y-direction boundary intersections
    if abs(dir_y) > _TINY:
        ty1, ty2 = (-cy - org_y) / dir_y, (cy - org_y) / dir_y
        t_min, t_max = max(t_min, min(ty1, ty2)), min(t_max, max(ty1, ty2))
    elif org_y < -cy or org_y > cy:  # Ray origin outside y-bounds
        d_sino[iview, iu, iv] = _ZERO; return
    
    # Z-direction boundary intersections (extends 2D algorithm to 3D)
    if abs(dir_z) > _TINY:
        tz1, tz2 = (-cz - org_z) / dir_z, (cz - org_z) / dir_z
        t_min, t_max = max(t_min, min(tz1, tz2)), min(t_max, max(tz1, tz2))
    elif org_z < -cz or org_z > cz:  # Ray origin outside z-bounds
        d_sino[iview, iu, iv] = _ZERO; return

    if t_min >= t_max:  # No valid 3D intersection
        d_sino[iview, iu, iv] = _ZERO; return

    # The traversal restarts at the entry point, so its parameters stay
    # within the volume diameter.
    ent_x = org_x + t_min * dir_x
    ent_y = org_y + t_min * dir_y
    ent_z = org_z + t_min * dir_z
    ray_x = dir_x
    ray_y = dir_y
    ray_z = dir_z
    t_end = t_max - t_min

    # === 3D SIDDON METHOD TRAVERSAL INITIALIZATION ===
    accum = _ZERO  # Accumulated projection value
    t = _ZERO
    
    # Convert 3D ray entry point to voxel indices
    ix = int(math.floor(ent_x + t * ray_x + cx))  # Current voxel x-index
    iy = int(math.floor(ent_y + t * ray_y + cy))  # Current voxel y-index
    iz = int(math.floor(ent_z + t * ray_z + cz))  # Current voxel z-index

    # 3D traversal parameters (extends 2D algorithm)
    step_x, step_y, step_z = (1 if ray_x >= 0 else -1), (1 if ray_y >= 0 else -1), (1 if ray_z >= 0 else -1)
    inv_dir_x = (_ONE / ray_x) if abs(ray_x) > _TINY else _ZERO
    inv_dir_y = (_ONE / ray_y) if abs(ray_y) > _TINY else _ZERO
    inv_dir_z = (_ONE / ray_z) if abs(ray_z) > _TINY else _ZERO
    dt_x = abs(inv_dir_x) if abs(ray_x) > _TINY else _BIG  # Parameter increment per x-voxel
    dt_y = abs(inv_dir_y) if abs(ray_y) > _TINY else _BIG  # Parameter increment per y-voxel
    dt_z = abs(inv_dir_z) if abs(ray_z) > _TINY else _BIG  # Parameter increment per z-voxel

    # Calculate parameter values for next 3D voxel boundary crossings
    next_ix = ix + (1 if step_x > 0 else 0)
    next_iy = iy + (1 if step_y > 0 else 0)
    next_iz = iz + (1 if step_z > 0 else 0)
    tx = (np.float32(next_ix) - cx - ent_x) * inv_dir_x if abs(ray_x) > _TINY else _BIG
    ty = (np.float32(next_iy) - cy - ent_y) * inv_dir_y if abs(ray_y) > _TINY else _BIG
    tz = (np.float32(next_iz) - cz - ent_z) * inv_dir_z if abs(ray_z) > _TINY else _BIG

    # === 3D TRAVERSAL LOOP WITH CELL-CONSTANT SIDDON INTEGRATION ===
    while t < t_end:
        if 0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz:
            # Determine next 3D voxel boundary crossing (minimum of x, y, z boundaries or ray exit)
            t_next = min(tx, ty, tz, t_end)
            seg_len = t_next - t
            if seg_len > _ZERO:
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
    
    # Siddon segment lengths are in voxel units; voxel_spacing makes them physical lengths.
    d_sino[iview, iu, iv] = accum * voxel_spacing

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

    This CUDA kernel implements the adjoint of cell-constant Siddon ray tracing for
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
    iv, iu, iview = cuda.grid(3)
    if iview >= n_views or iu >= n_u or iv >= n_v:
        return

    # === 3D BACKPROJECTION VALUE AND GEOMETRY SETUP (ARBITRARY TRAJECTORY) ===
    # Siddon segment lengths are in voxel units; voxel_spacing makes them physical lengths.
    g = d_sino[iview, iu, iv] * voxel_spacing

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
    u_offset = (np.float32(iu) + _HALF - np.float32(n_u) * _HALF) * du / voxel_spacing
    v_offset = (np.float32(iv) + _HALF - np.float32(n_v) * _HALF) * dv / voxel_spacing

    # Calculate 3D detector element position using center + u*u_vec + v*v_vec
    det_x = det_cx + u_offset * u_vec_x + v_offset * v_vec_x
    det_y = det_cy + u_offset * u_vec_y + v_offset * v_vec_y
    det_z = det_cz + u_offset * u_vec_z + v_offset * v_vec_z

    # === 3D RAY DIRECTION CALCULATION ===
    # Ray direction vector from source to detector element in 3D space
    dir_x, dir_y, dir_z = det_x - src_x, det_y - src_y, det_z - src_z
    length = math.sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z)  # 3D ray length
    if length < _EPSILON: return  # Skip degenerate rays
    inv_len = _ONE / length        # Normalization factor for ray direction
    dir_x, dir_y, dir_z = dir_x*inv_len, dir_y*inv_len, dir_z*inv_len  # Normalized 3D ray direction vector

    # === 3D RAY-VOLUME INTERSECTION CALCULATION ===
    # Measure the ray parameter from the endpoint nearer the volume centre, so a
    # distant source does not make the float32 parameters coarse. Only the
    # source-to-detector segment is integrated.
    if src_x * src_x + src_y * src_y + src_z * src_z <= det_x * det_x + det_y * det_y + det_z * det_z:
        org_x = src_x
        org_y = src_y
        org_z = src_z
        t_min, t_max = _ZERO, length
    else:
        org_x = det_x
        org_y = det_y
        org_z = det_z
        t_min, t_max = -length, _ZERO
    
    # X-direction boundary intersections
    if abs(dir_x) > _TINY:
        tx1, tx2 = (-cx - org_x) / dir_x, (cx - org_x) / dir_x
        t_min, t_max = max(t_min, min(tx1, tx2)), min(t_max, max(tx1, tx2))
    elif org_x < -cx or org_x > cx: return
    
    # Y-direction boundary intersections
    if abs(dir_y) > _TINY:
        ty1, ty2 = (-cy - org_y) / dir_y, (cy - org_y) / dir_y
        t_min, t_max = max(t_min, min(ty1, ty2)), min(t_max, max(ty1, ty2))
    elif org_y < -cy or org_y > cy: return
    
    # Z-direction boundary intersections (extends 2D algorithm to 3D)
    if abs(dir_z) > _TINY:
        tz1, tz2 = (-cz - org_z) / dir_z, (cz - org_z) / dir_z
        t_min, t_max = max(t_min, min(tz1, tz2)), min(t_max, max(tz1, tz2))
    elif org_z < -cz or org_z > cz: return

    if t_min >= t_max: return

    # The traversal restarts at the entry point, so its parameters stay
    # within the volume diameter.
    ent_x = org_x + t_min * dir_x
    ent_y = org_y + t_min * dir_y
    ent_z = org_z + t_min * dir_z
    ray_x = dir_x
    ray_y = dir_y
    ray_z = dir_z
    t_end = t_max - t_min

    # === 3D SIDDON METHOD TRAVERSAL INITIALIZATION ===
    t = _ZERO
    ix = int(math.floor(ent_x + t * ray_x + cx))  # Current voxel x-index
    iy = int(math.floor(ent_y + t * ray_y + cy))  # Current voxel y-index
    iz = int(math.floor(ent_z + t * ray_z + cz))  # Current voxel z-index

    # 3D traversal parameters (extends 2D algorithm)
    step_x, step_y, step_z = (1 if ray_x >= 0 else -1), (1 if ray_y >= 0 else -1), (1 if ray_z >= 0 else -1)
    inv_dir_x = (_ONE / ray_x) if abs(ray_x) > _TINY else _ZERO
    inv_dir_y = (_ONE / ray_y) if abs(ray_y) > _TINY else _ZERO
    inv_dir_z = (_ONE / ray_z) if abs(ray_z) > _TINY else _ZERO
    dt_x = abs(inv_dir_x) if abs(ray_x) > _TINY else _BIG  # Parameter increment per x-voxel
    dt_y = abs(inv_dir_y) if abs(ray_y) > _TINY else _BIG  # Parameter increment per y-voxel
    dt_z = abs(inv_dir_z) if abs(ray_z) > _TINY else _BIG  # Parameter increment per z-voxel

    # Calculate parameter values for next 3D voxel boundary crossings
    next_ix = ix + (1 if step_x > 0 else 0)
    next_iy = iy + (1 if step_y > 0 else 0)
    next_iz = iz + (1 if step_z > 0 else 0)
    tx = (np.float32(next_ix) - cx - ent_x) * inv_dir_x if abs(ray_x) > _TINY else _BIG
    ty = (np.float32(next_iy) - cy - ent_y) * inv_dir_y if abs(ray_y) > _TINY else _BIG
    tz = (np.float32(next_iz) - cz - ent_z) * inv_dir_z if abs(ray_z) > _TINY else _BIG

    # === 3D CONE BEAM BACKPROJECTION TRAVERSAL LOOP ===
    # Adjoint of the cell-constant Siddon forward projection.
    while t < t_end:
        if 0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz:
            # Determine next 3D voxel boundary crossing (minimum of x, y, z boundaries or ray exit)
            t_next = min(tx, ty, tz, t_end)
            seg_len = t_next - t
            if seg_len > _ZERO:
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


# ============================================================================
# 3D Cone Beam Analytical FDK Backprojection Gather (voxel-driven)
# ============================================================================
#
# Distinct from ``_cone_3d_backward_kernel`` above, which is the pure
# Siddon adjoint ``P^T`` used by autograd. This kernel is the classical
# voxel-driven FDK gather: for each output voxel it projects the voxel
# centre onto the detector for each view, bilinearly samples the
# already-filtered sinogram, and accumulates ``(sid/U_n)^2 * sample``.
# Both forward and adjoint autograd paths deliberately stay on the
# Siddon pair; only ``cone_weighted_backproject`` dispatches here.


@_FDK_ACCURACY_DECORATOR
def _cone_3d_fdk_backproject_kernel(
    d_sino, n_views, n_u, n_v,
    d_vol, Nx, Ny, Nz,
    du, dv, d_src_pos, d_det_center, d_det_u_vec, d_det_v_vec,
    cx, cy, cz, voxel_spacing, iso_x, iso_y, iso_z
):
    """Voxel-driven FDK backprojection gather (arbitrary trajectory).

    For each volume voxel, loops over all views and accumulates a
    bilinearly-interpolated sinogram sample weighted by ``(sid_v/U_n)^2``,
    where:

      * ``sid_v = (isocenter - S_v) . n_v`` is the source-to-isocenter
        distance along the detector normal,
      * ``U_n = (P - S_v) . n_v`` is the signed distance from the source
        to the voxel along the detector normal ``n_v = u_v x v_v``.

    For a canonical circular orbit this reduces to the classical
    ``(sid/U)^2`` FDK weight with ``U = sid + x*sin(b) - y*cos(b)``.

    Coalescing layout: ``d_vol`` has WHD layout ``(Nx, Ny, Nz)`` so its
    innermost stride-1 axis is ``Nz``. We therefore unpack
    ``(iz, iy, ix) = cuda.grid(3)`` and have the Python wrapper launch
    the grid as ``_grid_3d(Nz, Ny, Nx)`` so ``iz`` is warp-adjacent and
    the per-voxel writes are coalesced.
    """
    iz, iy, ix = cuda.grid(3)
    if ix >= Nx or iy >= Ny or iz >= Nz:
        return
    if n_u < 2 or n_v < 2:
        d_vol[ix, iy, iz] = _ZERO
        return

    # Voxel position in voxel-unit, origin-centred coordinates.
    # Sample at voxel centres, matching the cell-constant Siddon projector.
    x_v = np.float32(ix) + _HALF - cx
    y_v = np.float32(iy) + _HALF - cy
    z_v = np.float32(iz) + _HALF - cz

    du_v = du / voxel_spacing
    dv_v = dv / voxel_spacing

    half_u = (np.float32(n_u) - _ONE) * _HALF
    half_v = (np.float32(n_v) - _ONE) * _HALF

    accum = _ZERO
    for iview in range(n_views):
        # Source position in voxel units.
        sx = d_src_pos[iview, 0] / voxel_spacing
        sy = d_src_pos[iview, 1] / voxel_spacing
        sz = d_src_pos[iview, 2] / voxel_spacing

        # Detector centre in voxel units.
        dcx = d_det_center[iview, 0] / voxel_spacing
        dcy = d_det_center[iview, 1] / voxel_spacing
        dcz = d_det_center[iview, 2] / voxel_spacing

        # Detector orientation (unit vectors, ratio-free).
        ux = d_det_u_vec[iview, 0]
        uy = d_det_u_vec[iview, 1]
        uz = d_det_u_vec[iview, 2]
        vx = d_det_v_vec[iview, 0]
        vy = d_det_v_vec[iview, 1]
        vz = d_det_v_vec[iview, 2]

        # Detector normal n = u x v. Normalized defensively in case the
        # caller supplied non-orthonormal u/v axes.
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        n_len = math.sqrt(nx * nx + ny * ny + nz * nz)
        if n_len < _EPSILON:
            continue
        nx /= n_len
        ny /= n_len
        nz /= n_len

        # Orient the normal so it points from the source toward the
        # detector. Without this flip, user-supplied (u_vec, v_vec) pairs
        # whose cross product happens to point the other way would make
        # every voxel's U_n negative and get silently skipped, producing
        # a zero reconstruction. The circular_trajectory_3d helper already
        # produces the right-handed convention so this is a no-op there.
        align = (dcx - sx) * nx + (dcy - sy) * ny + (dcz - sz) * nz
        if align < _ZERO:
            nx = -nx
            ny = -ny
            nz = -nz

        # Signed distance from source to voxel along the detector normal
        # (this is the U in the (sid/U)^2 FDK weight).
        px = x_v - sx
        py = y_v - sy
        pz = z_v - sz
        U_n = px * nx + py * ny + pz * nz
        if U_n <= _EPSILON:
            continue

        # Source-to-detector distance along the normal.
        sdd_n = (dcx - sx) * nx + (dcy - sy) * ny + (dcz - sz) * nz
        if sdd_n <= _EPSILON:
            continue

        # Source-to-isocenter distance, independent of coordinate origin.
        sid_n = (iso_x - sx) * nx + (iso_y - sy) * ny + (iso_z - sz) * nz
        if sid_n <= _EPSILON:
            continue

        mag = sdd_n / U_n

        # Hit point on the detector plane.
        hx = sx + mag * px
        hy = sy + mag * py
        hz = sz + mag * pz

        # Detector coordinates relative to detector centre.
        rx = hx - dcx
        ry = hy - dcy
        rz = hz - dcz

        u_det = rx * ux + ry * uy + rz * uz
        v_det = rx * vx + ry * vy + rz * vz

        # Convert (u, v) to fractional bin index.
        fu = u_det / du_v + half_u
        fv = v_det / dv_v + half_v

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
            s00 * omtu * omtv +
            s10 * tu   * omtv +
            s01 * omtu * tv   +
            s11 * tu   * tv
        )

        # FDK weight (sid_n / U_n)^2.
        w_ratio = sid_n / U_n
        w = w_ratio * w_ratio

        accum += w * sample

    d_vol[ix, iy, iz] = accum
