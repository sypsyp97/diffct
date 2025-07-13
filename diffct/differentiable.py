import math
import numpy as np
import torch
from numba import cuda

# ---------------------------------------------------------------------------
# Global settings & helpers
# ---------------------------------------------------------------------------

_DTYPE              = np.float32            # Change to np.float64 if desired
_TPB_2D             = (16, 16)
_TPB_3D             = (8,  8,  8)
_FASTMATH_DECORATOR = cuda.jit(fastmath=True)
_INF                = _DTYPE(1e10)          # A large number to represent infinity
_EPSILON            = _DTYPE(1e-9)          # A small number for safe division


def _trig_tables(angles: np.ndarray, dtype=_DTYPE):
    """Return device arrays (d_cos, d_sin) for all projection angles."""
    cos_host = np.cos(angles).astype(dtype)
    sin_host = np.sin(angles).astype(dtype)
    return cuda.to_device(cos_host), cuda.to_device(sin_host)


def _grid_2d(n1, n2, tpb=_TPB_2D):
    return (math.ceil(n1 / tpb[0]), math.ceil(n2 / tpb[1])), tpb


def _grid_3d(n1, n2, n3, tpb=_TPB_3D):
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
    iang, idet = cuda.grid(2)
    if iang >= n_ang or idet >= n_det:
        return

    cos_a = d_cos[iang]
    sin_a = d_sin[iang]
    u     = (idet - (n_det - 1) * 0.5) * det_spacing

    dir_x, dir_y = cos_a, sin_a
    pnt_x, pnt_y = u * -sin_a, u * cos_a

    t_min, t_max = -_INF, _INF
    if abs(dir_x) > _EPSILON:
        tx1, tx2 = (-cx - pnt_x) / dir_x, (cx - pnt_x) / dir_x
        t_min, t_max = max(t_min, min(tx1, tx2)), min(t_max, max(tx1, tx2))
    elif pnt_x < -cx or pnt_x > cx:
        d_sino[iang, idet] = 0.0; return

    if abs(dir_y) > _EPSILON:
        ty1, ty2 = (-cy - pnt_y) / dir_y, (cy - pnt_y) / dir_y
        t_min, t_max = max(t_min, min(ty1, ty2)), min(t_max, max(ty1, ty2))
    elif pnt_y < -cy or pnt_y > cy:
        d_sino[iang, idet] = 0.0; return

    if t_min >= t_max:
        d_sino[iang, idet] = 0.0; return

    accum = 0.0
    t = t_min
    ix = int(math.floor(pnt_x + t * dir_x + cx))
    iy = int(math.floor(pnt_y + t * dir_y + cy))

    step_x, step_y = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1)
    dt_x = abs(1.0 / dir_x) if abs(dir_x) > _EPSILON else _INF
    dt_y = abs(1.0 / dir_y) if abs(dir_y) > _EPSILON else _INF
    tx = ((ix + (step_x > 0)) - cx - pnt_x) / dir_x if abs(dir_x) > _EPSILON else _INF
    ty = ((iy + (step_y > 0)) - cy - pnt_y) / dir_y if abs(dir_y) > _EPSILON else _INF

    while t < t_max:
        if 0 <= ix < Nx - 1 and 0 <= iy < Ny - 1:
            t_next = min(tx, ty, t_max)
            seg_len = t_next - t
            if seg_len > _EPSILON:
                mid_x = pnt_x + (t + seg_len * 0.5) * dir_x + cx
                mid_y = pnt_y + (t + seg_len * 0.5) * dir_y + cy
                ix0, iy0 = int(math.floor(mid_x)), int(math.floor(mid_y))
                dx, dy = mid_x - ix0, mid_y - iy0
                val = (
                    d_image[ix0,     iy0]     * (1 - dx) * (1 - dy) +
                    d_image[ix0 + 1, iy0]     * dx       * (1 - dy) +
                    d_image[ix0,     iy0 + 1] * (1 - dx) * dy       +
                    d_image[ix0 + 1, iy0 + 1] * dx       * dy
                )
                accum += val * seg_len
        
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
def _parallel_2d_backward_kernel(
    d_sino, n_ang, n_det,
    d_image, Nx, Ny,
    det_spacing, d_cos, d_sin, cx, cy
):
    iang, idet = cuda.grid(2)
    if iang >= n_ang or idet >= n_det:
        return

    val   = d_sino[iang, idet]
    cos_a = d_cos[iang]
    sin_a = d_sin[iang]
    u     = (idet - (n_det - 1) * 0.5) * det_spacing

    dir_x, dir_y = cos_a, sin_a
    pnt_x, pnt_y = u * -sin_a, u * cos_a

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

    t = t_min
    ix = int(math.floor(pnt_x + t * dir_x + cx))
    iy = int(math.floor(pnt_y + t * dir_y + cy))

    step_x, step_y = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1)
    dt_x = abs(1.0 / dir_x) if abs(dir_x) > _EPSILON else _INF
    dt_y = abs(1.0 / dir_y) if abs(dir_y) > _EPSILON else _INF
    tx = ((ix + (step_x > 0)) - cx - pnt_x) / dir_x if abs(dir_x) > _EPSILON else _INF
    ty = ((iy + (step_y > 0)) - cy - pnt_y) / dir_y if abs(dir_y) > _EPSILON else _INF

    while t < t_max:
        if 0 <= ix < Nx - 1 and 0 <= iy < Ny - 1:
            t_next = min(tx, ty, t_max)
            seg_len = t_next - t
            if seg_len > _EPSILON:
                mid_x = pnt_x + (t + seg_len * 0.5) * dir_x + cx
                mid_y = pnt_y + (t + seg_len * 0.5) * dir_y + cy
                ix0, iy0 = int(math.floor(mid_x)), int(math.floor(mid_y))
                dx, dy = mid_x - ix0, mid_y - iy0
                
                cval = val * seg_len
                cuda.atomic.add(d_image, (ix0,     iy0),     cval * (1 - dx) * (1 - dy))
                cuda.atomic.add(d_image, (ix0 + 1, iy0),     cval * dx       * (1 - dy))
                cuda.atomic.add(d_image, (ix0,     iy0 + 1), cval * (1 - dx) * dy)
                cuda.atomic.add(d_image, (ix0 + 1, iy0 + 1), cval * dx       * dy)

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
    iang, idet = cuda.grid(2)
    if iang >= n_ang or idet >= n_det:
        return

    cos_a = d_cos[iang]
    sin_a = d_sin[iang]
    u     = (idet - (n_det - 1) * 0.5) * det_spacing

    src_x = -iso_dist * sin_a
    src_y =  iso_dist * cos_a
    det_x = (src_dist - iso_dist) * sin_a + u * cos_a
    det_y = -(src_dist - iso_dist) * cos_a + u * sin_a

    dir_x, dir_y = det_x - src_x, det_y - src_y
    length = math.sqrt(dir_x * dir_x + dir_y * dir_y)
    if length < _EPSILON:
        d_sino[iang, idet] = 0.0; return
    inv_len = 1.0 / length
    dir_x, dir_y = dir_x * inv_len, dir_y * inv_len

    t_min, t_max = -_INF, _INF
    if abs(dir_x) > _EPSILON:
        tx1, tx2 = (-cx - src_x) / dir_x, (cx - src_x) / dir_x
        t_min, t_max = max(t_min, min(tx1, tx2)), min(t_max, max(tx1, tx2))
    elif src_x < -cx or src_x > cx:
        d_sino[iang, idet] = 0.0; return

    if abs(dir_y) > _EPSILON:
        ty1, ty2 = (-cy - src_y) / dir_y, (cy - src_y) / dir_y
        t_min, t_max = max(t_min, min(ty1, ty2)), min(t_max, max(ty1, ty2))
    elif src_y < -cy or src_y > cy:
        d_sino[iang, idet] = 0.0; return

    if t_min >= t_max:
        d_sino[iang, idet] = 0.0; return

    accum = 0.0
    t = t_min
    ix = int(math.floor(src_x + t * dir_x + cx))
    iy = int(math.floor(src_y + t * dir_y + cy))

    step_x, step_y = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1)
    dt_x = abs(1.0 / dir_x) if abs(dir_x) > _EPSILON else _INF
    dt_y = abs(1.0 / dir_y) if abs(dir_y) > _EPSILON else _INF
    tx = ((ix + (step_x > 0)) - cx - src_x) / dir_x if abs(dir_x) > _EPSILON else _INF
    ty = ((iy + (step_y > 0)) - cy - src_y) / dir_y if abs(dir_y) > _EPSILON else _INF

    while t < t_max:
        if 0 <= ix < Nx - 1 and 0 <= iy < Ny - 1:
            t_next = min(tx, ty, t_max)
            seg_len = t_next - t
            if seg_len > _EPSILON:
                mid_x = src_x + (t + seg_len * 0.5) * dir_x + cx
                mid_y = src_y + (t + seg_len * 0.5) * dir_y + cy
                ix0, iy0 = int(math.floor(mid_x)), int(math.floor(mid_y))
                dx, dy = mid_x - ix0, mid_y - iy0
                val = (
                    d_image[ix0,     iy0]     * (1 - dx) * (1 - dy) +
                    d_image[ix0 + 1, iy0]     * dx       * (1 - dy) +
                    d_image[ix0,     iy0 + 1] * (1 - dx) * dy       +
                    d_image[ix0 + 1, iy0 + 1] * dx       * dy
                )
                accum += val * seg_len
        
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
    iang, idet = cuda.grid(2)
    if iang >= n_ang or idet >= n_det:
        return

    val   = d_sino[iang, idet]
    cos_a = d_cos[iang]
    sin_a = d_sin[iang]
    u     = (idet - (n_det - 1) * 0.5) * det_spacing

    src_x = -iso_dist * sin_a
    src_y =  iso_dist * cos_a
    det_x = (src_dist - iso_dist) * sin_a + u * cos_a
    det_y = -(src_dist - iso_dist) * cos_a + u * sin_a

    dir_x, dir_y = det_x - src_x, det_y - src_y
    length = math.sqrt(dir_x * dir_x + dir_y * dir_y)
    if length < _EPSILON: return
    inv_len = 1.0 / length
    dir_x, dir_y = dir_x * inv_len, dir_y * inv_len

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

    t = t_min
    ix = int(math.floor(src_x + t * dir_x + cx))
    iy = int(math.floor(src_y + t * dir_y + cy))

    step_x, step_y = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1)
    dt_x = abs(1.0 / dir_x) if abs(dir_x) > _EPSILON else _INF
    dt_y = abs(1.0 / dir_y) if abs(dir_y) > _EPSILON else _INF
    tx = ((ix + (step_x > 0)) - cx - src_x) / dir_x if abs(dir_x) > _EPSILON else _INF
    ty = ((iy + (step_y > 0)) - cy - src_y) / dir_y if abs(dir_y) > _EPSILON else _INF

    while t < t_max:
        if 0 <= ix < Nx - 1 and 0 <= iy < Ny - 1:
            t_next = min(tx, ty, t_max)
            seg_len = t_next - t
            if seg_len > _EPSILON:
                mid_x = src_x + (t + seg_len * 0.5) * dir_x + cx
                mid_y = src_y + (t + seg_len * 0.5) * dir_y + cy
                ix0, iy0 = int(math.floor(mid_x)), int(math.floor(mid_y))
                dx, dy = mid_x - ix0, mid_y - iy0
                cval = val * seg_len
                cuda.atomic.add(d_image, (ix0,     iy0),     cval * (1 - dx) * (1 - dy))
                cuda.atomic.add(d_image, (ix0 + 1, iy0),     cval * dx       * (1 - dy))
                cuda.atomic.add(d_image, (ix0,     iy0 + 1), cval * (1 - dx) * dy)
                cuda.atomic.add(d_image, (ix0 + 1, iy0 + 1), cval * dx       * dy)

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
    iview, iu, iv = cuda.grid(3)
    if iview >= n_views or iu >= n_u or iv >= n_v:
        return

    cos_a, sin_a = d_cos[iview], d_sin[iview]
    u, v = (iu - (n_u - 1) * 0.5) * du, (iv - (n_v - 1) * 0.5) * dv

    src_x, src_y, src_z = -iso_dist * sin_a, iso_dist * cos_a, 0.0
    det_x = (src_dist - iso_dist) * sin_a + u * cos_a
    det_y = -(src_dist - iso_dist) * cos_a + u * sin_a
    det_z = v

    dir_x, dir_y, dir_z = det_x - src_x, det_y - src_y, det_z - src_z
    length = math.sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z)
    if length < _EPSILON:
        d_sino[iview, iu, iv] = 0.0; return
    inv_len = 1.0 / length
    dir_x, dir_y, dir_z = dir_x*inv_len, dir_y*inv_len, dir_z*inv_len

    t_min, t_max = -_INF, _INF
    if abs(dir_x) > _EPSILON:
        tx1, tx2 = (-cx - src_x) / dir_x, (cx - src_x) / dir_x
        t_min, t_max = max(t_min, min(tx1, tx2)), min(t_max, max(tx1, tx2))
    elif src_x < -cx or src_x > cx:
        d_sino[iview, iu, iv] = 0.0; return
    if abs(dir_y) > _EPSILON:
        ty1, ty2 = (-cy - src_y) / dir_y, (cy - src_y) / dir_y
        t_min, t_max = max(t_min, min(ty1, ty2)), min(t_max, max(ty1, ty2))
    elif src_y < -cy or src_y > cy:
        d_sino[iview, iu, iv] = 0.0; return
    if abs(dir_z) > _EPSILON:
        tz1, tz2 = (-cz - src_z) / dir_z, (cz - src_z) / dir_z
        t_min, t_max = max(t_min, min(tz1, tz2)), min(t_max, max(tz1, tz2))
    elif src_z < -cz or src_z > cz:
        d_sino[iview, iu, iv] = 0.0; return

    if t_min >= t_max:
        d_sino[iview, iu, iv] = 0.0; return

    accum = 0.0
    t = t_min
    ix = int(math.floor(src_x + t * dir_x + cx))
    iy = int(math.floor(src_y + t * dir_y + cy))
    iz = int(math.floor(src_z + t * dir_z + cz))

    step_x, step_y, step_z = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1), (1 if dir_z >= 0 else -1)
    dt_x = abs(1.0 / dir_x) if abs(dir_x) > _EPSILON else _INF
    dt_y = abs(1.0 / dir_y) if abs(dir_y) > _EPSILON else _INF
    dt_z = abs(1.0 / dir_z) if abs(dir_z) > _EPSILON else _INF
    tx = ((ix + (step_x > 0)) - cx - src_x) / dir_x if abs(dir_x) > _EPSILON else _INF
    ty = ((iy + (step_y > 0)) - cy - src_y) / dir_y if abs(dir_y) > _EPSILON else _INF
    tz = ((iz + (step_z > 0)) - cz - src_z) / dir_z if abs(dir_z) > _EPSILON else _INF

    while t < t_max:
        if 0 <= ix < Nx - 1 and 0 <= iy < Ny - 1 and 0 <= iz < Nz - 1:
            t_next = min(tx, ty, tz, t_max)
            seg_len = t_next - t
            if seg_len > _EPSILON:
                mid_x = src_x + (t + seg_len * 0.5) * dir_x + cx
                mid_y = src_y + (t + seg_len * 0.5) * dir_y + cy
                mid_z = src_z + (t + seg_len * 0.5) * dir_z + cz
                ix0, iy0, iz0 = int(math.floor(mid_x)), int(math.floor(mid_y)), int(math.floor(mid_z))
                dx, dy, dz = mid_x - ix0, mid_y - iy0, mid_z - iz0
                val = (
                    d_vol[ix0,     iy0,     iz0]     * (1-dx)*(1-dy)*(1-dz) +
                    d_vol[ix0 + 1, iy0,     iz0]     * dx*(1-dy)*(1-dz) +
                    d_vol[ix0,     iy0 + 1, iz0]     * (1-dx)*dy*(1-dz) +
                    d_vol[ix0,     iy0,     iz0 + 1] * (1-dx)*(1-dy)*dz +
                    d_vol[ix0 + 1, iy0 + 1, iz0]     * dx*dy*(1-dz) +
                    d_vol[ix0 + 1, iy0,     iz0 + 1] * dx*(1-dy)*dz +
                    d_vol[ix0,     iy0 + 1, iz0 + 1] * (1-dx)*dy*dz +
                    d_vol[ix0 + 1, iy0 + 1, iz0 + 1] * dx*dy*dz
                )
                accum += val * seg_len

        if tx <= ty and tx <= tz:
            t = tx
            ix += step_x
            tx += dt_x
        elif ty <= tx and ty <= tz:
            t = ty
            iy += step_y
            ty += dt_y
        else:
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
    iview, iu, iv = cuda.grid(3)
    if iview >= n_views or iu >= n_u or iv >= n_v:
        return

    g = d_sino[iview, iu, iv]
    cos_a, sin_a = d_cos[iview], d_sin[iview]
    u, v = (iu - (n_u - 1) * 0.5) * du, (iv - (n_v - 1) * 0.5) * dv

    src_x, src_y, src_z = -iso_dist * sin_a, iso_dist * cos_a, 0.0
    det_x = (src_dist - iso_dist) * sin_a + u * cos_a
    det_y = -(src_dist - iso_dist) * cos_a + u * sin_a
    det_z = v

    dir_x, dir_y, dir_z = det_x - src_x, det_y - src_y, det_z - src_z
    length = math.sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z)
    if length < _EPSILON: return
    inv_len = 1.0 / length
    dir_x, dir_y, dir_z = dir_x*inv_len, dir_y*inv_len, dir_z*inv_len

    t_min, t_max = -_INF, _INF
    if abs(dir_x) > _EPSILON:
        tx1, tx2 = (-cx - src_x) / dir_x, (cx - src_x) / dir_x
        t_min, t_max = max(t_min, min(tx1, tx2)), min(t_max, max(tx1, tx2))
    elif src_x < -cx or src_x > cx: return
    if abs(dir_y) > _EPSILON:
        ty1, ty2 = (-cy - src_y) / dir_y, (cy - src_y) / dir_y
        t_min, t_max = max(t_min, min(ty1, ty2)), min(t_max, max(ty1, ty2))
    elif src_y < -cy or src_y > cy: return
    if abs(dir_z) > _EPSILON:
        tz1, tz2 = (-cz - src_z) / dir_z, (cz - src_z) / dir_z
        t_min, t_max = max(t_min, min(tz1, tz2)), min(t_max, max(tz1, tz2))
    elif src_z < -cz or src_z > cz: return

    if t_min >= t_max: return

    t = t_min
    ix = int(math.floor(src_x + t * dir_x + cx))
    iy = int(math.floor(src_y + t * dir_y + cy))
    iz = int(math.floor(src_z + t * dir_z + cz))

    step_x, step_y, step_z = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1), (1 if dir_z >= 0 else -1)
    dt_x = abs(1.0 / dir_x) if abs(dir_x) > _EPSILON else _INF
    dt_y = abs(1.0 / dir_y) if abs(dir_y) > _EPSILON else _INF
    dt_z = abs(1.0 / dir_z) if abs(dir_z) > _EPSILON else _INF
    tx = ((ix + (step_x > 0)) - cx - src_x) / dir_x if abs(dir_x) > _EPSILON else _INF
    ty = ((iy + (step_y > 0)) - cy - src_y) / dir_y if abs(dir_y) > _EPSILON else _INF
    tz = ((iz + (step_z > 0)) - cz - src_z) / dir_z if abs(dir_z) > _EPSILON else _INF

    while t < t_max:
        if 0 <= ix < Nx - 1 and 0 <= iy < Ny - 1 and 0 <= iz < Nz - 1:
            t_next = min(tx, ty, tz, t_max)
            seg_len = t_next - t
            if seg_len > _EPSILON:
                mid_x = src_x + (t + seg_len * 0.5) * dir_x + cx
                mid_y = src_y + (t + seg_len * 0.5) * dir_y + cy
                mid_z = src_z + (t + seg_len * 0.5) * dir_z + cz
                ix0, iy0, iz0 = int(math.floor(mid_x)), int(math.floor(mid_y)), int(math.floor(mid_z))
                dx, dy, dz = mid_x - ix0, mid_y - iy0, mid_z - iz0
                cval = g * seg_len
                cuda.atomic.add(d_vol, (ix0,     iy0,     iz0),     cval * (1-dx)*(1-dy)*(1-dz))
                cuda.atomic.add(d_vol, (ix0 + 1, iy0,     iz0),     cval * dx*(1-dy)*(1-dz))
                cuda.atomic.add(d_vol, (ix0,     iy0 + 1, iz0),     cval * (1-dx)*dy*(1-dz))
                cuda.atomic.add(d_vol, (ix0,     iy0,     iz0 + 1), cval * (1-dx)*(1-dy)*dz)
                cuda.atomic.add(d_vol, (ix0 + 1, iy0 + 1, iz0),     cval * dx*dy*(1-dz))
                cuda.atomic.add(d_vol, (ix0 + 1, iy0,     iz0 + 1), cval * dx*(1-dy)*dz)
                cuda.atomic.add(d_vol, (ix0,     iy0 + 1, iz0 + 1), cval * (1-dx)*dy*dz)
                cuda.atomic.add(d_vol, (ix0 + 1, iy0 + 1, iz0 + 1), cval * dx*dy*dz)

        if tx <= ty and tx <= tz:
            t = tx
            ix += step_x
            tx += dt_x
        elif ty <= tx and ty <= tz:
            t = ty
            iy += step_y
            ty += dt_y
        else:
            t = tz
            iz += step_z
            tz += dt_z


# ############################################################################
# DIFFERENTIABLE TORCH FUNCTIONS
# ############################################################################

class ParallelProjectorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, angles, num_detectors, detector_spacing=1.0):
        device = image.device
        # Kernel expects (W, H), input is (H, W), so transpose
        image_np = image.detach().cpu().numpy().astype(_DTYPE, copy=False).T
        angles_np = angles.detach().cpu().numpy().astype(_DTYPE, copy=False)

        Nx, Ny = image_np.shape
        n_angles = angles_np.shape[0]

        d_image = cuda.to_device(image_np)
        d_cos, d_sin = _trig_tables(angles_np, _DTYPE)
        d_sino = cuda.device_array((n_angles, num_detectors), dtype=_DTYPE)

        grid, tpb = _grid_2d(n_angles, num_detectors)
        cx, cy = _DTYPE((Nx - 1) * 0.5), _DTYPE((Ny - 1) * 0.5)

        _parallel_2d_forward_kernel[grid, tpb](
            d_image, Nx, Ny, d_sino, n_angles, num_detectors,
            _DTYPE(detector_spacing), d_cos, d_sin, cx, cy
        )

        sino_np = d_sino.copy_to_host()
        sinogram = torch.as_tensor(sino_np, device=device)
        
        ctx.save_for_backward(angles)
        ctx.intermediate = (num_detectors, detector_spacing, image.shape[0], image.shape[1])
        return sinogram

    @staticmethod
    def backward(ctx, grad_sinogram):
        angles, = ctx.saved_tensors
        num_detectors, detector_spacing, H, W = ctx.intermediate
        device = grad_sinogram.device

        grad_np = grad_sinogram.detach().cpu().numpy().astype(_DTYPE, copy=False)
        ang_np = angles.detach().cpu().numpy().astype(_DTYPE, copy=False)
        n_angles = ang_np.shape[0]
        
        Nx, Ny = W, H

        d_grad_sino = cuda.to_device(grad_np)
        d_cos, d_sin = _trig_tables(ang_np, _DTYPE)
        d_img_grad = cuda.to_device(np.zeros((Nx, Ny), dtype=_DTYPE))

        grid, tpb = _grid_2d(n_angles, num_detectors)
        cx, cy = _DTYPE((Nx - 1) * 0.5), _DTYPE((Ny - 1) * 0.5)

        # The backward pass of forward projection is back-projection
        _parallel_2d_backward_kernel[grid, tpb](
            d_grad_sino, n_angles, num_detectors,
            d_img_grad, Nx, Ny,
            _DTYPE(detector_spacing), d_cos, d_sin, cx, cy
        )

        grad_image_np = d_img_grad.copy_to_host()
        # Kernel produced (W, H), transpose to (H, W) for PyTorch
        grad_image = torch.as_tensor(grad_image_np.T, device=device)
        return grad_image, None, None, None


class ParallelBackprojectorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sinogram, angles, detector_spacing=1.0, H=128, W=128):
        device = sinogram.device
        sino_np = sinogram.detach().cpu().numpy().astype(_DTYPE, copy=False)
        angles_np = angles.detach().cpu().numpy().astype(_DTYPE, copy=False)

        n_ang, n_det = sino_np.shape
        Nx, Ny = W, H # Kernel expects (W, H)

        d_sino = cuda.to_device(sino_np)
        d_cos, d_sin = _trig_tables(angles_np, _DTYPE)
        d_reco = cuda.to_device(np.zeros((Nx, Ny), dtype=_DTYPE))

        grid, tpb = _grid_2d(n_ang, n_det)
        cx, cy = _DTYPE((Nx - 1) * 0.5), _DTYPE((Ny - 1) * 0.5)

        _parallel_2d_backward_kernel[grid, tpb](
            d_sino, n_ang, n_det, d_reco, Nx, Ny,
            _DTYPE(detector_spacing), d_cos, d_sin, cx, cy
        )

        reco_np = d_reco.copy_to_host()
        # Kernel produced (W, H), transpose to (H, W) for PyTorch
        reco = torch.as_tensor(reco_np.T, device=device)
        
        ctx.save_for_backward(angles)
        ctx.intermediate = (H, W, detector_spacing, sinogram.shape[0], sinogram.shape[1])
        return reco

    @staticmethod
    def backward(ctx, grad_output):
        angles, = ctx.saved_tensors
        H, W, detector_spacing, n_ang, n_det = ctx.intermediate
        device = grad_output.device

        # Kernel expects (W, H), input is (H, W), so transpose
        grad_np = grad_output.detach().cpu().numpy().astype(_DTYPE, copy=False).T
        angles_np = angles.detach().cpu().numpy().astype(_DTYPE, copy=False)
        Nx, Ny = grad_np.shape

        d_grad_out = cuda.to_device(grad_np)
        d_cos, d_sin = _trig_tables(angles_np, _DTYPE)
        d_sino_grad = cuda.device_array((n_ang, n_det), dtype=_DTYPE)

        grid, tpb = _grid_2d(n_ang, n_det)
        cx, cy = _DTYPE((Nx - 1) * 0.5), _DTYPE((Ny - 1) * 0.5)

        # The backward pass of back-projection is forward projection
        _parallel_2d_forward_kernel[grid, tpb](
            d_grad_out, Nx, Ny, d_sino_grad, n_ang, n_det,
            _DTYPE(detector_spacing), d_cos, d_sin, cx, cy
        )

        grad_sino_np = d_sino_grad.copy_to_host()
        grad_sino = torch.as_tensor(grad_sino_np, device=device)
        return grad_sino, None, None, None, None


class FanProjectorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, angles, num_detectors, detector_spacing, source_distance, isocenter_distance):
        device = image.device
        img_np = image.detach().cpu().numpy().astype(_DTYPE, copy=False).T
        ang_np = angles.detach().cpu().numpy().astype(_DTYPE, copy=False)

        Nx, Ny = img_np.shape
        n_ang = ang_np.shape[0]

        d_image = cuda.to_device(img_np)
        d_cos, d_sin = _trig_tables(ang_np, _DTYPE)
        d_sino = cuda.device_array((n_ang, num_detectors), dtype=_DTYPE)

        grid, tpb = _grid_2d(n_ang, num_detectors)
        cx, cy = _DTYPE((Nx - 1) * 0.5), _DTYPE((Ny - 1) * 0.5)

        _fan_2d_forward_kernel[grid, tpb](
            d_image, Nx, Ny, d_sino, n_ang, num_detectors,
            _DTYPE(detector_spacing), d_cos, d_sin,
            _DTYPE(source_distance), _DTYPE(isocenter_distance), cx, cy
        )

        sino_np = d_sino.copy_to_host()
        sino = torch.as_tensor(sino_np, device=device)
        
        ctx.save_for_backward(angles)
        ctx.intermediate = (num_detectors, detector_spacing, image.shape[0], image.shape[1],
                            source_distance, isocenter_distance)
        return sino

    @staticmethod
    def backward(ctx, grad_sinogram):
        angles, = ctx.saved_tensors
        (n_det, det_spacing, H, W, src_dist, iso_dist) = ctx.intermediate
        device = grad_sinogram.device

        grad_np = grad_sinogram.detach().cpu().numpy().astype(_DTYPE, copy=False)
        ang_np = angles.detach().cpu().numpy().astype(_DTYPE, copy=False)
        n_ang = ang_np.shape[0]
        Nx, Ny = W, H

        d_grad_sino = cuda.to_device(grad_np)
        d_cos, d_sin = _trig_tables(ang_np, _DTYPE)
        d_img_grad = cuda.to_device(np.zeros((Nx, Ny), dtype=_DTYPE))

        grid, tpb = _grid_2d(n_ang, n_det)
        cx, cy = _DTYPE((Nx - 1) * 0.5), _DTYPE((Ny - 1) * 0.5)

        _fan_2d_backward_kernel[grid, tpb](
            d_grad_sino, n_ang, n_det, d_img_grad, Nx, Ny,
            _DTYPE(det_spacing), d_cos, d_sin,
            _DTYPE(src_dist), _DTYPE(iso_dist), cx, cy
        )
        
        grad_img_np = d_img_grad.copy_to_host()
        grad_img = torch.as_tensor(grad_img_np.T, device=device)
        return grad_img, None, None, None, None, None


class FanBackprojectorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sinogram, angles, detector_spacing, H, W, source_distance, isocenter_distance):
        device = sinogram.device
        sino_np = sinogram.detach().cpu().numpy().astype(_DTYPE, copy=False)
        ang_np = angles.detach().cpu().numpy().astype(_DTYPE, copy=False)
        
        n_ang, n_det = sino_np.shape
        Nx, Ny = W, H

        d_sino = cuda.to_device(sino_np)
        d_cos, d_sin = _trig_tables(ang_np, _DTYPE)
        d_reco = cuda.to_device(np.zeros((Nx, Ny), dtype=_DTYPE))

        grid, tpb = _grid_2d(n_ang, n_det)
        cx, cy = _DTYPE((Nx - 1) * 0.5), _DTYPE((Ny - 1) * 0.5)

        _fan_2d_backward_kernel[grid, tpb](
            d_sino, n_ang, n_det, d_reco, Nx, Ny,
            _DTYPE(detector_spacing), d_cos, d_sin,
            _DTYPE(source_distance), _DTYPE(isocenter_distance), cx, cy
        )

        reco_np = d_reco.copy_to_host()
        image = torch.as_tensor(reco_np.T, device=device)
        
        ctx.save_for_backward(angles)
        ctx.intermediate = (H, W, detector_spacing, n_ang, n_det, source_distance, isocenter_distance)
        return image

    @staticmethod
    def backward(ctx, grad_output):
        angles, = ctx.saved_tensors
        (H, W, det_spacing, n_ang, n_det, src_dist, iso_dist) = ctx.intermediate
        device = grad_output.device

        grad_np = grad_output.detach().cpu().numpy().astype(_DTYPE, copy=False).T
        ang_np = angles.detach().cpu().numpy().astype(_DTYPE, copy=False)
        Nx, Ny = grad_np.shape

        d_grad_out = cuda.to_device(grad_np)
        d_cos, d_sin = _trig_tables(ang_np, _DTYPE)
        d_sino_grad = cuda.device_array((n_ang, n_det), dtype=_DTYPE)

        grid, tpb = _grid_2d(n_ang, n_det)
        cx, cy = _DTYPE((Nx - 1) * 0.5), _DTYPE((Ny - 1) * 0.5)

        _fan_2d_forward_kernel[grid, tpb](
            d_grad_out, Nx, Ny, d_sino_grad, n_ang, n_det,
            _DTYPE(det_spacing), d_cos, d_sin,
            _DTYPE(src_dist), _DTYPE(iso_dist), cx, cy
        )
        
        grad_sino_np = d_sino_grad.copy_to_host()
        grad_sino = torch.as_tensor(grad_sino_np, device=device)
        return grad_sino, None, None, None, None, None, None


class ConeProjectorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, angles, det_u, det_v, du, dv, source_distance, isocenter_distance):
        device = volume.device
        vol_np = volume.detach().cpu().numpy().astype(_DTYPE, copy=False).transpose((2, 1, 0))
        ang_np = angles.detach().cpu().numpy().astype(_DTYPE, copy=False)

        Nx, Ny, Nz = vol_np.shape
        n_views = ang_np.shape[0]

        d_vol = cuda.to_device(vol_np)
        d_cos, d_sin = _trig_tables(ang_np, _DTYPE)
        d_sino = cuda.device_array((n_views, det_u, det_v), dtype=_DTYPE)

        grid, tpb = _grid_3d(n_views, det_u, det_v)
        cx, cy, cz = _DTYPE((Nx-1)*0.5), _DTYPE((Ny-1)*0.5), _DTYPE((Nz-1)*0.5)

        _cone_3d_forward_kernel[grid, tpb](
            d_vol, Nx, Ny, Nz, d_sino, n_views, det_u, det_v,
            _DTYPE(du), _DTYPE(dv), d_cos, d_sin,
            _DTYPE(source_distance), _DTYPE(isocenter_distance),
            cx, cy, cz
        )

        sino_np = d_sino.copy_to_host()
        sino = torch.as_tensor(sino_np, device=device)
        
        ctx.save_for_backward(angles)
        ctx.intermediate = (Nx, Ny, Nz, det_u, det_v, du, dv,
                            source_distance, isocenter_distance)
        return sino

    @staticmethod
    def backward(ctx, grad_sinogram):
        angles, = ctx.saved_tensors
        (Nx, Ny, Nz, det_u, det_v, du, dv,
         src_dist, iso_dist) = ctx.intermediate
        device = grad_sinogram.device

        grad_np = grad_sinogram.detach().cpu().numpy().astype(_DTYPE, copy=False)
        ang_np = angles.detach().cpu().numpy().astype(_DTYPE, copy=False)
        n_views = ang_np.shape[0]

        d_grad_sino = cuda.to_device(grad_np)
        d_cos, d_sin = _trig_tables(ang_np, _DTYPE)
        d_vol_grad = cuda.to_device(np.zeros((Nx, Ny, Nz), dtype=_DTYPE))

        grid, tpb = _grid_3d(n_views, det_u, det_v)
        cx, cy, cz = _DTYPE((Nx-1)*0.5), _DTYPE((Ny-1)*0.5), _DTYPE((Nz-1)*0.5)

        _cone_3d_backward_kernel[grid, tpb](
            d_grad_sino, n_views, det_u, det_v, d_vol_grad, Nx, Ny, Nz,
            _DTYPE(du), _DTYPE(dv), d_cos, d_sin,
            _DTYPE(src_dist), _DTYPE(iso_dist), cx, cy, cz
        )

        grad_vol_np = d_vol_grad.copy_to_host()
        grad_vol = torch.as_tensor(grad_vol_np.transpose((2, 1, 0)), device=device)
        return grad_vol, None, None, None, None, None, None, None


class ConeBackprojectorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sinogram, angles, D, H, W, du, dv, source_distance, isocenter_distance):
        device = sinogram.device
        sino_np = sinogram.detach().cpu().numpy().astype(_DTYPE, copy=False)
        ang_np = angles.detach().cpu().numpy().astype(_DTYPE, copy=False)
        
        n_views, n_u, n_v = sino_np.shape
        Nx, Ny, Nz = W, H, D

        d_sino = cuda.to_device(sino_np)
        d_cos, d_sin = _trig_tables(ang_np, _DTYPE)
        d_reco = cuda.to_device(np.zeros((Nx, Ny, Nz), dtype=_DTYPE))

        grid, tpb = _grid_3d(n_views, n_u, n_v)
        cx, cy, cz = _DTYPE((Nx-1)*0.5), _DTYPE((Ny-1)*0.5), _DTYPE((Nz-1)*0.5)

        _cone_3d_backward_kernel[grid, tpb](
            d_sino, n_views, n_u, n_v, d_reco, Nx, Ny, Nz,
            _DTYPE(du), _DTYPE(dv), d_cos, d_sin,
            _DTYPE(source_distance), _DTYPE(isocenter_distance), cx, cy, cz
        )

        vol_np = d_reco.copy_to_host()
        vol = torch.as_tensor(vol_np.transpose((2, 1, 0)), device=device)
        
        ctx.save_for_backward(angles)
        ctx.intermediate = (D, H, W, n_u, n_v, du, dv,
                            source_distance, isocenter_distance)
        return vol

    @staticmethod
    def backward(ctx, grad_output):
        angles, = ctx.saved_tensors
        (D, H, W, n_u, n_v, du, dv,
         src_dist, iso_dist) = ctx.intermediate
        device = grad_output.device

        grad_np = grad_output.detach().cpu().numpy().astype(_DTYPE, copy=False).transpose((2, 1, 0))
        ang_np = angles.detach().cpu().numpy().astype(_DTYPE, copy=False)
        n_views = ang_np.shape[0]
        Nx, Ny, Nz = grad_np.shape

        d_grad_out = cuda.to_device(grad_np)
        d_cos, d_sin = _trig_tables(ang_np, _DTYPE)
        d_sino_grad = cuda.device_array((n_views, n_u, n_v), dtype=_DTYPE)

        grid, tpb = _grid_3d(n_views, n_u, n_v)
        cx, cy, cz = _DTYPE((Nx-1)*0.5), _DTYPE((Ny-1)*0.5), _DTYPE((Nz-1)*0.5)

        _cone_3d_forward_kernel[grid, tpb](
            d_grad_out, Nx, Ny, Nz, d_sino_grad, n_views, n_u, n_v,
            _DTYPE(du), _DTYPE(dv), d_cos, d_sin,
            _DTYPE(src_dist), _DTYPE(iso_dist), cx, cy, cz
        )
        
        grad_sino_np = d_sino_grad.copy_to_host()
        grad_sino = torch.as_tensor(grad_sino_np, device=device)
        return grad_sino, None, None, None, None, None, None, None, None