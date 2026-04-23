"""CUDA kernels for 2D parallel beam projections.

This module contains CUDA kernels implementing the Siddon ray-tracing method
for 2D parallel beam forward projection and backprojection, plus a
dedicated voxel-driven FBP gather kernel for analytical reconstruction
pipelines.
"""

import math
import numpy as np
from numba import cuda

from ..constants import (
    _FASTMATH_DECORATOR,
    _FDK_ACCURACY_DECORATOR,
    _INF,
    _NEG_INF,
    _ZERO,
    _ONE,
    _HALF,
    _EPSILON,
)


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

    This CUDA kernel implements cell-constant Siddon ray tracing for
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
    Integrates each ray through piecewise-constant pixel cells.
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
    u_offset = (np.float32(idet) - np.float32(n_det) * _HALF) * det_spacing / voxel_spacing

    # Ray starting point: detector origin + offset along u-direction
    pnt_x = det_ox + u_offset * u_vec_x
    pnt_y = det_oy + u_offset * u_vec_y

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

    This CUDA kernel implements the adjoint of cell-constant Siddon ray tracing for
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
    u_offset = (np.float32(idet) - np.float32(n_det) * _HALF) * det_spacing / voxel_spacing

    # Ray starting point: detector origin + offset along u-direction
    pnt_x = det_ox + u_offset * u_vec_x
    pnt_y = det_oy + u_offset * u_vec_y

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



# ============================================================================
# 2D Parallel Beam Analytical FBP Backprojection Gather (voxel-driven)
# ============================================================================
#
# Distinct from ``_parallel_2d_backward_kernel`` above, which is the
# pure Siddon adjoint ``P^T`` used by autograd. This kernel is the
# classical voxel-driven FBP gather: for each output pixel it projects
# the pixel centre onto the detector for each view, linearly samples
# the filtered sinogram, and accumulates. Parallel beam has no source,
# so there is no ``(sid/U)^2`` distance weighting.


@_FDK_ACCURACY_DECORATOR
def _parallel_2d_fbp_backproject_kernel(
    d_sino, n_views, n_det,
    d_image, Nx, Ny,
    det_spacing, d_ray_dir, d_det_origin, d_det_u_vec,
    cx, cy, voxel_spacing
):
    """Voxel-driven parallel-beam FBP backprojection gather (arbitrary trajectory).

    For each output pixel ``(ix, iy)`` loops over views and accumulates
    a linearly interpolated filtered-sinogram sample. The pixel's
    detector coordinate is ``u = (P - det_origin) . det_u_vec`` (no
    magnification because the rays are parallel).
    """
    ix, iy = cuda.grid(2)
    if ix >= Nx or iy >= Ny:
        return

    x_v = np.float32(ix) - cx
    y_v = np.float32(iy) - cy

    det_spacing_v = det_spacing / voxel_spacing
    half_u = np.float32(n_det) * _HALF

    accum = _ZERO
    for iview in range(n_views):
        dox = d_det_origin[iview, 0] / voxel_spacing
        doy = d_det_origin[iview, 1] / voxel_spacing

        ux = d_det_u_vec[iview, 0]
        uy = d_det_u_vec[iview, 1]

        rx = x_v - dox
        ry = y_v - doy

        u_det = rx * ux + ry * uy

        fu = u_det / det_spacing_v + half_u
        if fu < _ZERO or fu > (np.float32(n_det) - _ONE):
            continue

        iu0 = int(math.floor(fu))
        if iu0 >= n_det - 1:
            iu0 = n_det - 2
        if iu0 < 0:
            iu0 = 0
        tu = fu - np.float32(iu0)
        if tu < _ZERO:
            tu = _ZERO
        elif tu > _ONE:
            tu = _ONE

        s0 = d_sino[iview, iu0]
        s1 = d_sino[iview, iu0 + 1]
        sample = s0 * (_ONE - tu) + s1 * tu

        accum += sample

    d_image[iy, ix] = accum
