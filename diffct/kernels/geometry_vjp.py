"""CUDA kernels for gradients of the Siddon projector with respect to geometry.

For a cell-constant image the Siddon projection of one ray is

    p = s * |D| * sum_k f_k (tau_{k+1} - tau_k),   D = B - A,

where A is the source (or, for parallel beams, the detector pixel), B the
detector pixel, ``s`` the voxel spacing and ``tau_k`` the fractions along
``D`` at which the ray enters and leaves cell ``k``. Collecting terms gives
``sum_j c_j tau_j`` with ``c_j = f_before - f_after`` at each crossed cell
face (``-f_first`` at an entry face, ``+f_last`` at an exit face). A face
``x_a = b`` gives ``tau = (b - A_a) / D_a``, so

    d tau / d A_a = (tau - 1) / D_a,    d tau / d B_a = -tau / D_a.

Segment ends at the source or the detector do not move a face crossing and
contribute nothing. The derivative is exact for the discrete model wherever
the ray does not pass through a cell edge or corner. On an edge, where two
crossings coincide, the kernels return the derivative of one adjacent side.

Each thread traces one ray with the same setup and traversal as the
projector kernels and adds ``grad_sino * dp/dgeometry`` to per-view
gradients with atomic additions.
"""

import math
import numpy as np
from numba import cuda

from ..constants import (
    _FASTMATH_DECORATOR,
    _BIG,
    _TINY,
    _ZERO,
    _ONE,
    _HALF,
    _EPSILON,
)


@cuda.jit(device=True)
def _face_terms(c, t_org, t_a0, t_b0, length, d):
    """Return (d/dA_a, d/dB_a) of ``c * tau`` for a face crossed at ``t_org``.

    ``t_org`` is the ray parameter of the crossing; ``t_a0`` and ``t_b0`` are
    the parameters of A and B on the same unit-speed ray, ``d`` is the unit
    direction component normal to the face.
    """
    tau_minus_one = (t_org - t_b0) / length
    tau = (t_org - t_a0) / length
    return c * tau_minus_one / d, -c * tau / d


@_FASTMATH_DECORATOR
def _fan_2d_geometry_vjp_kernel(
    d_image, Nx, Ny,
    d_grad_sino, n_ang, n_det,
    det_spacing, d_src_pos, d_det_center, d_det_u_vec,
    cx, cy, voxel_spacing,
    d_grad_src, d_grad_det_center, d_grad_det_u,
):
    """Accumulate fan-beam geometry gradients of ``<grad_sino, A x>`` per view."""
    iang, idet = cuda.grid(2)
    if iang >= n_ang or idet >= n_det:
        return
    g = d_grad_sino[iang, idet]
    if g == _ZERO:
        return

    # Ray setup: identical to _fan_2d_forward_kernel, plus the face that sets
    # the entry and the exit (-1 = segment end at the source or detector).
    src_x = d_src_pos[iang, 0] / voxel_spacing
    src_y = d_src_pos[iang, 1] / voxel_spacing
    det_cx = d_det_center[iang, 0] / voxel_spacing
    det_cy = d_det_center[iang, 1] / voxel_spacing
    u_vec_x = d_det_u_vec[iang, 0]
    u_vec_y = d_det_u_vec[iang, 1]
    u_phys = (np.float32(idet) + _HALF - np.float32(n_det) * _HALF) * det_spacing
    u_offset = u_phys / voxel_spacing
    det_x = det_cx + u_offset * u_vec_x
    det_y = det_cy + u_offset * u_vec_y

    dir_x, dir_y = det_x - src_x, det_y - src_y
    length = math.sqrt(dir_x * dir_x + dir_y * dir_y)
    if length < _EPSILON:
        return
    inv_len = _ONE / length
    dir_x, dir_y = dir_x * inv_len, dir_y * inv_len

    if src_x * src_x + src_y * src_y <= det_x * det_x + det_y * det_y:
        org_x = src_x
        org_y = src_y
        t_min, t_max = _ZERO, length
        t_a0, t_b0 = _ZERO, length
    else:
        org_x = det_x
        org_y = det_y
        t_min, t_max = -length, _ZERO
        t_a0, t_b0 = -length, _ZERO
    ent_axis = -1
    ex_axis = -1
    if abs(dir_x) > _TINY:
        tx1, tx2 = (-cx - org_x) / dir_x, (cx - org_x) / dir_x
        lo, hi = min(tx1, tx2), max(tx1, tx2)
        if lo > t_min:
            t_min = lo
            ent_axis = 0
        if hi < t_max:
            t_max = hi
            ex_axis = 0
    elif org_x < -cx or org_x > cx:
        return
    if abs(dir_y) > _TINY:
        ty1, ty2 = (-cy - org_y) / dir_y, (cy - org_y) / dir_y
        lo, hi = min(ty1, ty2), max(ty1, ty2)
        if lo > t_min:
            t_min = lo
            ent_axis = 1
        if hi < t_max:
            t_max = hi
            ex_axis = 1
    elif org_y < -cy or org_y > cy:
        return
    if t_min >= t_max:
        return

    ent_x = org_x + t_min * dir_x
    ent_y = org_y + t_min * dir_y
    ray_x = dir_x
    ray_y = dir_y
    t_end = t_max - t_min

    accum = _ZERO
    t = _ZERO
    ix = int(math.floor(ent_x + cx))
    iy = int(math.floor(ent_y + cy))
    step_x, step_y = (1 if ray_x >= 0 else -1), (1 if ray_y >= 0 else -1)
    inv_dir_x = (_ONE / ray_x) if abs(ray_x) > _TINY else _ZERO
    inv_dir_y = (_ONE / ray_y) if abs(ray_y) > _TINY else _ZERO
    dt_x = abs(inv_dir_x) if abs(ray_x) > _TINY else _BIG
    dt_y = abs(inv_dir_y) if abs(ray_y) > _TINY else _BIG
    next_ix = ix + (1 if step_x > 0 else 0)
    next_iy = iy + (1 if step_y > 0 else 0)
    tx = (np.float32(next_ix) - cx - ent_x) * inv_dir_x if abs(ray_x) > _TINY else _BIG
    ty = (np.float32(next_iy) - cy - ent_y) * inv_dir_y if abs(ray_y) > _TINY else _BIG

    ga_x = _ZERO
    ga_y = _ZERO
    gb_x = _ZERO
    gb_y = _ZERO
    f_prev = _ZERO
    pend_axis = ent_axis
    pend_t = _ZERO
    first = True
    while t < t_end:
        f = _ZERO
        if 0 <= ix < Nx and 0 <= iy < Ny:
            f = d_image[iy, ix]
        t_next = min(tx, ty, t_end)
        seg_len = t_next - t
        # The face crossed just before this cell: entry face or interior face.
        if pend_axis >= 0:
            c = (-f) if first else (f_prev - f)
            if pend_axis == 0:
                da, db = _face_terms(c, pend_t + t_min, t_a0, t_b0, length, ray_x)
                ga_x += da
                gb_x += db
            else:
                da, db = _face_terms(c, pend_t + t_min, t_a0, t_b0, length, ray_y)
                ga_y += da
                gb_y += db
        first = False
        if seg_len > _ZERO:
            accum += f * seg_len
        f_prev = f
        if tx <= ty:
            t = tx
            pend_t = tx
            pend_axis = 0
            ix += step_x
            tx += dt_x
        else:
            t = ty
            pend_t = ty
            pend_axis = 1
            iy += step_y
            ty += dt_y
    if ex_axis == 0:
        da, db = _face_terms(f_prev, t_max, t_a0, t_b0, length, ray_x)
        ga_x += da
        gb_x += db
    elif ex_axis == 1:
        da, db = _face_terms(f_prev, t_max, t_a0, t_b0, length, ray_y)
        ga_y += da
        gb_y += db

    # Length term: p = s * |D| * S with S = accum / |D| (voxel units).
    S = accum / length
    ga_x -= S * ray_x
    ga_y -= S * ray_y
    gb_x += S * ray_x
    gb_y += S * ray_y

    cuda.atomic.add(d_grad_src, (iang, 0), g * ga_x)
    cuda.atomic.add(d_grad_src, (iang, 1), g * ga_y)
    cuda.atomic.add(d_grad_det_center, (iang, 0), g * gb_x)
    cuda.atomic.add(d_grad_det_center, (iang, 1), g * gb_y)
    cuda.atomic.add(d_grad_det_u, (iang, 0), g * u_phys * gb_x)
    cuda.atomic.add(d_grad_det_u, (iang, 1), g * u_phys * gb_y)


@_FASTMATH_DECORATOR
def _cone_3d_geometry_vjp_kernel(
    d_vol, Nx, Ny, Nz,
    d_grad_sino, n_views, n_u, n_v,
    du, dv, d_src_pos, d_det_center, d_det_u_vec, d_det_v_vec,
    cx, cy, cz, voxel_spacing,
    d_grad_src, d_grad_det_center, d_grad_det_u, d_grad_det_v,
):
    """Accumulate cone-beam geometry gradients of ``<grad_sino, A x>`` per view."""
    iv, iu, iview = cuda.grid(3)
    if iview >= n_views or iu >= n_u or iv >= n_v:
        return
    g = d_grad_sino[iview, iu, iv]
    if g == _ZERO:
        return

    src_x = d_src_pos[iview, 0] / voxel_spacing
    src_y = d_src_pos[iview, 1] / voxel_spacing
    src_z = d_src_pos[iview, 2] / voxel_spacing
    det_cx = d_det_center[iview, 0] / voxel_spacing
    det_cy = d_det_center[iview, 1] / voxel_spacing
    det_cz = d_det_center[iview, 2] / voxel_spacing
    u_vec_x = d_det_u_vec[iview, 0]
    u_vec_y = d_det_u_vec[iview, 1]
    u_vec_z = d_det_u_vec[iview, 2]
    v_vec_x = d_det_v_vec[iview, 0]
    v_vec_y = d_det_v_vec[iview, 1]
    v_vec_z = d_det_v_vec[iview, 2]
    u_phys = (np.float32(iu) + _HALF - np.float32(n_u) * _HALF) * du
    v_phys = (np.float32(iv) + _HALF - np.float32(n_v) * _HALF) * dv
    u_offset = u_phys / voxel_spacing
    v_offset = v_phys / voxel_spacing
    det_x = det_cx + u_offset * u_vec_x + v_offset * v_vec_x
    det_y = det_cy + u_offset * u_vec_y + v_offset * v_vec_y
    det_z = det_cz + u_offset * u_vec_z + v_offset * v_vec_z

    dir_x, dir_y, dir_z = det_x - src_x, det_y - src_y, det_z - src_z
    length = math.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)
    if length < _EPSILON:
        return
    inv_len = _ONE / length
    dir_x, dir_y, dir_z = dir_x * inv_len, dir_y * inv_len, dir_z * inv_len

    if (src_x * src_x + src_y * src_y + src_z * src_z
            <= det_x * det_x + det_y * det_y + det_z * det_z):
        org_x = src_x
        org_y = src_y
        org_z = src_z
        t_min, t_max = _ZERO, length
        t_a0, t_b0 = _ZERO, length
    else:
        org_x = det_x
        org_y = det_y
        org_z = det_z
        t_min, t_max = -length, _ZERO
        t_a0, t_b0 = -length, _ZERO
    ent_axis = -1
    ex_axis = -1
    if abs(dir_x) > _TINY:
        tx1, tx2 = (-cx - org_x) / dir_x, (cx - org_x) / dir_x
        lo, hi = min(tx1, tx2), max(tx1, tx2)
        if lo > t_min:
            t_min = lo
            ent_axis = 0
        if hi < t_max:
            t_max = hi
            ex_axis = 0
    elif org_x < -cx or org_x > cx:
        return
    if abs(dir_y) > _TINY:
        ty1, ty2 = (-cy - org_y) / dir_y, (cy - org_y) / dir_y
        lo, hi = min(ty1, ty2), max(ty1, ty2)
        if lo > t_min:
            t_min = lo
            ent_axis = 1
        if hi < t_max:
            t_max = hi
            ex_axis = 1
    elif org_y < -cy or org_y > cy:
        return
    if abs(dir_z) > _TINY:
        tz1, tz2 = (-cz - org_z) / dir_z, (cz - org_z) / dir_z
        lo, hi = min(tz1, tz2), max(tz1, tz2)
        if lo > t_min:
            t_min = lo
            ent_axis = 2
        if hi < t_max:
            t_max = hi
            ex_axis = 2
    elif org_z < -cz or org_z > cz:
        return
    if t_min >= t_max:
        return

    ent_x = org_x + t_min * dir_x
    ent_y = org_y + t_min * dir_y
    ent_z = org_z + t_min * dir_z
    ray_x = dir_x
    ray_y = dir_y
    ray_z = dir_z
    t_end = t_max - t_min

    accum = _ZERO
    t = _ZERO
    ix = int(math.floor(ent_x + cx))
    iy = int(math.floor(ent_y + cy))
    iz = int(math.floor(ent_z + cz))
    step_x, step_y, step_z = (1 if ray_x >= 0 else -1), (1 if ray_y >= 0 else -1), (1 if ray_z >= 0 else -1)
    inv_dir_x = (_ONE / ray_x) if abs(ray_x) > _TINY else _ZERO
    inv_dir_y = (_ONE / ray_y) if abs(ray_y) > _TINY else _ZERO
    inv_dir_z = (_ONE / ray_z) if abs(ray_z) > _TINY else _ZERO
    dt_x = abs(inv_dir_x) if abs(ray_x) > _TINY else _BIG
    dt_y = abs(inv_dir_y) if abs(ray_y) > _TINY else _BIG
    dt_z = abs(inv_dir_z) if abs(ray_z) > _TINY else _BIG
    next_ix = ix + (1 if step_x > 0 else 0)
    next_iy = iy + (1 if step_y > 0 else 0)
    next_iz = iz + (1 if step_z > 0 else 0)
    tx = (np.float32(next_ix) - cx - ent_x) * inv_dir_x if abs(ray_x) > _TINY else _BIG
    ty = (np.float32(next_iy) - cy - ent_y) * inv_dir_y if abs(ray_y) > _TINY else _BIG
    tz = (np.float32(next_iz) - cz - ent_z) * inv_dir_z if abs(ray_z) > _TINY else _BIG

    ga_x = _ZERO
    ga_y = _ZERO
    ga_z = _ZERO
    gb_x = _ZERO
    gb_y = _ZERO
    gb_z = _ZERO
    f_prev = _ZERO
    pend_axis = ent_axis
    pend_t = _ZERO
    first = True
    while t < t_end:
        f = _ZERO
        if 0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz:
            f = d_vol[ix, iy, iz]
        t_next = min(tx, ty, tz, t_end)
        seg_len = t_next - t
        if pend_axis >= 0:
            c = (-f) if first else (f_prev - f)
            if pend_axis == 0:
                da, db = _face_terms(c, pend_t + t_min, t_a0, t_b0, length, ray_x)
                ga_x += da
                gb_x += db
            elif pend_axis == 1:
                da, db = _face_terms(c, pend_t + t_min, t_a0, t_b0, length, ray_y)
                ga_y += da
                gb_y += db
            else:
                da, db = _face_terms(c, pend_t + t_min, t_a0, t_b0, length, ray_z)
                ga_z += da
                gb_z += db
        first = False
        if seg_len > _ZERO:
            accum += f * seg_len
        f_prev = f
        if tx <= ty and tx <= tz:
            t = tx
            pend_t = tx
            pend_axis = 0
            ix += step_x
            tx += dt_x
        elif ty <= tx and ty <= tz:
            t = ty
            pend_t = ty
            pend_axis = 1
            iy += step_y
            ty += dt_y
        else:
            t = tz
            pend_t = tz
            pend_axis = 2
            iz += step_z
            tz += dt_z
    if ex_axis == 0:
        da, db = _face_terms(f_prev, t_max, t_a0, t_b0, length, ray_x)
        ga_x += da
        gb_x += db
    elif ex_axis == 1:
        da, db = _face_terms(f_prev, t_max, t_a0, t_b0, length, ray_y)
        ga_y += da
        gb_y += db
    elif ex_axis == 2:
        da, db = _face_terms(f_prev, t_max, t_a0, t_b0, length, ray_z)
        ga_z += da
        gb_z += db

    S = accum / length
    ga_x -= S * ray_x
    ga_y -= S * ray_y
    ga_z -= S * ray_z
    gb_x += S * ray_x
    gb_y += S * ray_y
    gb_z += S * ray_z

    cuda.atomic.add(d_grad_src, (iview, 0), g * ga_x)
    cuda.atomic.add(d_grad_src, (iview, 1), g * ga_y)
    cuda.atomic.add(d_grad_src, (iview, 2), g * ga_z)
    cuda.atomic.add(d_grad_det_center, (iview, 0), g * gb_x)
    cuda.atomic.add(d_grad_det_center, (iview, 1), g * gb_y)
    cuda.atomic.add(d_grad_det_center, (iview, 2), g * gb_z)
    cuda.atomic.add(d_grad_det_u, (iview, 0), g * u_phys * gb_x)
    cuda.atomic.add(d_grad_det_u, (iview, 1), g * u_phys * gb_y)
    cuda.atomic.add(d_grad_det_u, (iview, 2), g * u_phys * gb_z)
    cuda.atomic.add(d_grad_det_v, (iview, 0), g * v_phys * gb_x)
    cuda.atomic.add(d_grad_det_v, (iview, 1), g * v_phys * gb_y)
    cuda.atomic.add(d_grad_det_v, (iview, 2), g * v_phys * gb_z)


@_FASTMATH_DECORATOR
def _parallel_2d_geometry_vjp_kernel(
    d_image, Nx, Ny,
    d_grad_sino, n_ang, n_det,
    det_spacing, d_ray_dir, d_det_origin, d_det_u_vec,
    cx, cy, voxel_spacing,
    d_grad_ray_dir, d_grad_det_origin, d_grad_det_u,
):
    """Accumulate parallel-beam geometry gradients of ``<grad_sino, A x>`` per view.

    The ray is ``P + t r`` over the whole line, with P the detector pixel and
    r the ray direction as given. A face crossing ``t = (b - P_a) / r_a`` gives
    ``dt/dP_a = -1 / r_a`` and ``dt/dr_a = -t / r_a``.
    """
    iang, idet = cuda.grid(2)
    if iang >= n_ang or idet >= n_det:
        return
    g = d_grad_sino[iang, idet]
    if g == _ZERO:
        return

    # Ray setup: identical to _parallel_2d_forward_kernel (float64), plus the
    # entry and exit faces.
    dir_x = np.float64(d_ray_dir[iang, 0])
    dir_y = np.float64(d_ray_dir[iang, 1])
    det_ox = np.float64(d_det_origin[iang, 0]) / voxel_spacing
    det_oy = np.float64(d_det_origin[iang, 1]) / voxel_spacing
    u_vec_x = np.float64(d_det_u_vec[iang, 0])
    u_vec_y = np.float64(d_det_u_vec[iang, 1])
    u_phys = (np.float32(idet) + _HALF - np.float32(n_det) * _HALF) * det_spacing
    u_offset = (np.float64(idet) + _HALF - np.float64(n_det) * _HALF) * det_spacing / voxel_spacing
    pnt_x = det_ox + u_offset * u_vec_x
    pnt_y = det_oy + u_offset * u_vec_y

    t_min, t_max = -_BIG, _BIG
    ent_axis = -1
    ex_axis = -1
    if dir_x != 0.0:
        tx1, tx2 = (-cx - pnt_x) / dir_x, (cx - pnt_x) / dir_x
        lo, hi = min(tx1, tx2), max(tx1, tx2)
        if lo > t_min:
            t_min = lo
            ent_axis = 0
        if hi < t_max:
            t_max = hi
            ex_axis = 0
    elif pnt_x < -cx or pnt_x > cx:
        return
    if dir_y != 0.0:
        ty1, ty2 = (-cy - pnt_y) / dir_y, (cy - pnt_y) / dir_y
        lo, hi = min(ty1, ty2), max(ty1, ty2)
        if lo > t_min:
            t_min = lo
            ent_axis = 1
        if hi < t_max:
            t_max = hi
            ex_axis = 1
    elif pnt_y < -cy or pnt_y > cy:
        return
    if t_min >= t_max:
        return

    ent_x = np.float32(pnt_x + t_min * dir_x)
    ent_y = np.float32(pnt_y + t_min * dir_y)
    ray_x = np.float32(dir_x)
    ray_y = np.float32(dir_y)
    t_end = np.float32(t_max - t_min)

    t = _ZERO
    ix = int(math.floor(ent_x + cx))
    iy = int(math.floor(ent_y + cy))
    step_x, step_y = (1 if ray_x >= 0 else -1), (1 if ray_y >= 0 else -1)
    inv_dir_x = (_ONE / ray_x) if abs(ray_x) > _TINY else _ZERO
    inv_dir_y = (_ONE / ray_y) if abs(ray_y) > _TINY else _ZERO
    dt_x = abs(inv_dir_x) if abs(ray_x) > _TINY else _BIG
    dt_y = abs(inv_dir_y) if abs(ray_y) > _TINY else _BIG
    next_ix = ix + (1 if step_x > 0 else 0)
    next_iy = iy + (1 if step_y > 0 else 0)
    tx = (np.float32(next_ix) - cx - ent_x) * inv_dir_x if abs(ray_x) > _TINY else _BIG
    ty = (np.float32(next_iy) - cy - ent_y) * inv_dir_y if abs(ray_y) > _TINY else _BIG

    # Per axis: C = sum of face coefficients, T = sum of coefficient * t,
    # with t measured from the entry point (t_min is added afterwards).
    c_x = _ZERO
    c_y = _ZERO
    ct_x = _ZERO
    ct_y = _ZERO
    f_prev = _ZERO
    pend_axis = ent_axis
    pend_t = _ZERO
    first = True
    while t < t_end:
        f = _ZERO
        if 0 <= ix < Nx and 0 <= iy < Ny:
            f = d_image[iy, ix]
        if pend_axis >= 0:
            c = (-f) if first else (f_prev - f)
            if pend_axis == 0:
                c_x += c
                ct_x += c * pend_t
            else:
                c_y += c
                ct_y += c * pend_t
        first = False
        f_prev = f
        if tx <= ty:
            t = tx
            pend_t = tx
            pend_axis = 0
            ix += step_x
            tx += dt_x
        else:
            t = ty
            pend_t = ty
            pend_axis = 1
            iy += step_y
            ty += dt_y
    if ex_axis == 0:
        c_x += f_prev
        ct_x += f_prev * t_end
    elif ex_axis == 1:
        c_y += f_prev
        ct_y += f_prev * t_end

    # p = s * sum_j c_j t_j; dt/dP_a = -1/r_a (P in voxels) and dt/dr_a = -t/r_a.
    # t_min stays float64: a distant detector origin makes it large.
    gp_x = _ZERO
    gp_y = _ZERO
    gr_x = _ZERO
    gr_y = _ZERO
    if abs(ray_x) > _TINY:
        gp_x = -c_x / ray_x
        gr_x = np.float32(-voxel_spacing * (t_min * c_x + ct_x) / ray_x)
    if abs(ray_y) > _TINY:
        gp_y = -c_y / ray_y
        gr_y = np.float32(-voxel_spacing * (t_min * c_y + ct_y) / ray_y)

    cuda.atomic.add(d_grad_ray_dir, (iang, 0), g * gr_x)
    cuda.atomic.add(d_grad_ray_dir, (iang, 1), g * gr_y)
    cuda.atomic.add(d_grad_det_origin, (iang, 0), g * gp_x)
    cuda.atomic.add(d_grad_det_origin, (iang, 1), g * gp_y)
    cuda.atomic.add(d_grad_det_u, (iang, 0), g * u_phys * gp_x)
    cuda.atomic.add(d_grad_det_u, (iang, 1), g * u_phys * gp_y)
