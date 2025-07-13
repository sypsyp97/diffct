import math
import numpy as np
from numba import cuda

# ------------------------------------------------------------------
# GLOBAL SETTINGS
# ------------------------------------------------------------------

_DTYPE = np.float32
_TPB_2D = (16, 16)
_TPB_3D = (8, 8, 8)
_FASTMATH_DECORATOR = cuda.jit(fastmath=True)
_INF = _DTYPE(np.inf)
_EPSILON = _DTYPE(1e-9)

# ------------------------------------------------------------------
# SMALL HOST HELPERS
# ------------------------------------------------------------------


def _trig_tables(angles: np.ndarray, dtype=_DTYPE):
    """Return (d_cos, d_sin) device vectors for all projection angles."""
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
    iview, idet = cuda.grid(2)
    if iview >= n_views or idet >= n_det:
        return

    cos_a = d_cos[iview]
    sin_a = d_sin[iview]
    u = (idet - (n_det - 1) * 0.5) * det_spacing

    dir_x, dir_y = cos_a, sin_a
    start_x, start_y = u * (-sin_a), u * (cos_a)

    t_min, t_max = -_INF, _INF
    if abs(dir_x) > _EPSILON:
        tx1 = (-cx - start_x) / dir_x
        tx2 = (cx - start_x) / dir_x
        t_min = max(t_min, min(tx1, tx2))
        t_max = min(t_max, max(tx1, tx2))
    elif start_x < -cx or start_x > cx:
        d_sino[iview, idet] = 0.0
        return

    if abs(dir_y) > _EPSILON:
        ty1 = (-cy - start_y) / dir_y
        ty2 = (cy - start_y) / dir_y
        t_min = max(t_min, min(ty1, ty2))
        t_max = min(t_max, max(ty1, ty2))
    elif start_y < -cy or start_y > cy:
        d_sino[iview, idet] = 0.0
        return

    if t_min >= t_max:
        d_sino[iview, idet] = 0.0
        return

    t = t_min
    accum = 0.0

    ix = int(math.floor(start_x + t * dir_x + cx))
    iy = int(math.floor(start_y + t * dir_y + cy))

    step_x = 1 if dir_x >= 0 else -1
    step_y = 1 if dir_y >= 0 else -1
    dt_x = abs(1.0 / dir_x) if abs(dir_x) > _EPSILON else _INF
    dt_y = abs(1.0 / dir_y) if abs(dir_y) > _EPSILON else _INF

    tx = ((ix + (step_x > 0)) - cx - start_x) / dir_x if abs(dir_x) > _EPSILON else _INF
    ty = ((iy + (step_y > 0)) - cy - start_y) / dir_y if abs(dir_y) > _EPSILON else _INF

    while t < t_max:
        if 0 <= ix < Nx - 1 and 0 <= iy < Ny - 1:
            t_next = min(tx, ty, t_max)
            seg_len = t_next - t

            if seg_len > 0:
                mid_x = start_x + (t + seg_len * 0.5) * dir_x + cx
                mid_y = start_y + (t + seg_len * 0.5) * dir_y + cy
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

    d_sino[iview, idet] = accum


@_FASTMATH_DECORATOR
def back_parallel_2d_kernel(
    d_sino,
    Nx, Ny,
    d_reco,
    n_views, n_det, det_spacing,
    d_cos, d_sin,
    cx, cy
):
    iview, idet = cuda.grid(2)
    if iview >= n_views or idet >= n_det:
        return

    val = d_sino[iview, idet]
    cos_a = d_cos[iview]
    sin_a = d_sin[iview]
    u = (idet - (n_det - 1) * 0.5) * det_spacing

    dir_x, dir_y = cos_a, sin_a
    start_x, start_y = u * (-sin_a), u * (cos_a)

    t_min, t_max = -_INF, _INF
    if abs(dir_x) > _EPSILON:
        tx1, tx2 = (-cx - start_x) / dir_x, (cx - start_x) / dir_x
        t_min, t_max = max(t_min, min(tx1, tx2)), min(t_max, max(tx1, tx2))
    elif start_x < -cx or start_x > cx: return

    if abs(dir_y) > _EPSILON:
        ty1, ty2 = (-cy - start_y) / dir_y, (cy - start_y) / dir_y
        t_min, t_max = max(t_min, min(ty1, ty2)), min(t_max, max(ty1, ty2))
    elif start_y < -cy or start_y > cy: return

    if t_min >= t_max: return

    t = t_min
    ix = int(math.floor(start_x + t * dir_x + cx))
    iy = int(math.floor(start_y + t * dir_y + cy))

    step_x, step_y = (1 if dir_x >= 0 else -1), (1 if dir_y >= 0 else -1)
    dt_x = abs(1.0 / dir_x) if abs(dir_x) > _EPSILON else _INF
    dt_y = abs(1.0 / dir_y) if abs(dir_y) > _EPSILON else _INF
    tx = ((ix + (step_x > 0)) - cx - start_x) / dir_x if abs(dir_x) > _EPSILON else _INF
    ty = ((iy + (step_y > 0)) - cy - start_y) / dir_y if abs(dir_y) > _EPSILON else _INF

    while t < t_max:
        if 0 <= ix < Nx - 1 and 0 <= iy < Ny - 1:
            t_next = min(tx, ty, t_max)
            seg_len = t_next - t
            
            if seg_len > 0:
                cval = val * seg_len
                mid_x = start_x + (t + seg_len * 0.5) * dir_x + cx
                mid_y = start_y + (t + seg_len * 0.5) * dir_y + cy
                ix0, iy0 = int(math.floor(mid_x)), int(math.floor(mid_y))
                dx, dy = mid_x - ix0, mid_y - iy0

                cuda.atomic.add(d_reco, (ix0,     iy0),     cval * (1 - dx) * (1 - dy))
                cuda.atomic.add(d_reco, (ix0 + 1, iy0),     cval * dx       * (1 - dy))
                cuda.atomic.add(d_reco, (ix0,     iy0 + 1), cval * (1 - dx) * dy)
                cuda.atomic.add(d_reco, (ix0 + 1, iy0 + 1), cval * dx       * dy)

        if tx <= ty:
            t = tx
            ix += step_x
            tx += dt_x
        else:
            t = ty
            iy += step_y
            ty += dt_y


def forward_parallel_2d(
    image: np.ndarray,
    n_views: int,
    n_det: int,
    det_spacing: float,
    angles: np.ndarray,
):
    """Computes the 2D parallel-beam forward projection."""
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
    """Computes the 2D parallel-beam back-projection."""
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
    iview, idet = cuda.grid(2)
    if iview >= n_views or idet >= n_det:
        return

    cos_a = d_cos[iview]
    sin_a = d_sin[iview]
    u = (idet - (n_det - 1) * 0.5) * det_spacing

    src_x = -iso_dist * sin_a
    src_y =  iso_dist * cos_a
    det_x = (src_dist - iso_dist) * sin_a + u * cos_a
    det_y = -(src_dist - iso_dist) * cos_a + u * sin_a

    dir_x, dir_y = det_x - src_x, det_y - src_y
    length = math.sqrt(dir_x * dir_x + dir_y * dir_y)
    if length < _EPSILON:
        d_sino[iview, idet] = 0.0
        return
    inv_len = 1.0 / length
    dir_x, dir_y = dir_x * inv_len, dir_y * inv_len

    t_min, t_max = -_INF, _INF
    if abs(dir_x) > _EPSILON:
        tx1, tx2 = (-cx - src_x) / dir_x, (cx - src_x) / dir_x
        t_min, t_max = max(t_min, min(tx1, tx2)), min(t_max, max(tx1, tx2))
    elif src_x < -cx or src_x > cx:
        d_sino[iview, idet] = 0.0
        return

    if abs(dir_y) > _EPSILON:
        ty1, ty2 = (-cy - src_y) / dir_y, (cy - src_y) / dir_y
        t_min, t_max = max(t_min, min(ty1, ty2)), min(t_max, max(ty1, ty2))
    elif src_y < -cy or src_y > cy:
        d_sino[iview, idet] = 0.0
        return

    if t_min >= t_max:
        d_sino[iview, idet] = 0.0
        return

    t = t_min
    accum = 0.0
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

            if seg_len > 0:
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

    d_sino[iview, idet] = accum


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
    iview, idet = cuda.grid(2)
    if iview >= n_views or idet >= n_det:
        return

    val = d_sino[iview, idet]
    cos_a = d_cos[iview]
    sin_a = d_sin[iview]
    u = (idet - (n_det - 1) * 0.5) * det_spacing

    src_x, src_y = -iso_dist * sin_a, iso_dist * cos_a
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
            
            if seg_len > 0:
                cval = val * seg_len
                mid_x = src_x + (t + seg_len * 0.5) * dir_x + cx
                mid_y = src_y + (t + seg_len * 0.5) * dir_y + cy
                ix0, iy0 = int(math.floor(mid_x)), int(math.floor(mid_y))
                dx, dy = mid_x - ix0, mid_y - iy0

                cuda.atomic.add(d_reco, (ix0,     iy0),     cval * (1 - dx) * (1 - dy))
                cuda.atomic.add(d_reco, (ix0 + 1, iy0),     cval * dx       * (1 - dy))
                cuda.atomic.add(d_reco, (ix0,     iy0 + 1), cval * (1 - dx) * dy)
                cuda.atomic.add(d_reco, (ix0 + 1, iy0 + 1), cval * dx       * dy)

        if tx <= ty:
            t = tx
            ix += step_x
            tx += dt_x
        else:
            t = ty
            iy += step_y
            ty += dt_y


def forward_fan_2d(
    image: np.ndarray,
    n_views: int,
    n_det: int,
    det_spacing: float,
    angles: np.ndarray,
    src_dist: float,
    iso_dist: float,
):
    """Computes the 2D fan-beam forward projection."""
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
    """Computes the 2D fan-beam back-projection."""
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
    iview, iu, iv = cuda.grid(3)
    if iview >= n_views or iu >= n_u or iv >= n_v:
        return

    cos_a = d_cos[iview]
    sin_a = d_sin[iview]
    u = (iu - (n_u - 1) * 0.5) * du
    v = (iv - (n_v - 1) * 0.5) * dv

    src_x, src_y, src_z = -iso_dist * sin_a, iso_dist * cos_a, 0.0
    det_x = (src_dist - iso_dist) * sin_a + u * cos_a
    det_y = -(src_dist - iso_dist) * cos_a + u * sin_a
    det_z = v

    dir_x, dir_y, dir_z = det_x - src_x, det_y - src_y, det_z - src_z
    length = math.sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z)
    if length < _EPSILON:
        d_sino[iview, iu, iv] = 0.0
        return
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

    t = t_min
    accum = 0.0
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

            if seg_len > 0:
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
    iview, iu, iv = cuda.grid(3)
    if iview >= n_views or iu >= n_u or iv >= n_v:
        return

    val = d_sino[iview, iu, iv]
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
            
            if seg_len > 0:
                cval = val * seg_len
                mid_x = src_x + (t + seg_len * 0.5) * dir_x + cx
                mid_y = src_y + (t + seg_len * 0.5) * dir_y + cy
                mid_z = src_z + (t + seg_len * 0.5) * dir_z + cz
                ix0, iy0, iz0 = int(math.floor(mid_x)), int(math.floor(mid_y)), int(math.floor(mid_z))
                dx, dy, dz = mid_x - ix0, mid_y - iy0, mid_z - iz0

                cuda.atomic.add(d_reco, (ix0,     iy0,     iz0),     cval * (1-dx)*(1-dy)*(1-dz))
                cuda.atomic.add(d_reco, (ix0 + 1, iy0,     iz0),     cval * dx*(1-dy)*(1-dz))
                cuda.atomic.add(d_reco, (ix0,     iy0 + 1, iz0),     cval * (1-dx)*dy*(1-dz))
                cuda.atomic.add(d_reco, (ix0,     iy0,     iz0 + 1), cval * (1-dx)*(1-dy)*dz)
                cuda.atomic.add(d_reco, (ix0 + 1, iy0 + 1, iz0),     cval * dx*dy*(1-dz))
                cuda.atomic.add(d_reco, (ix0 + 1, iy0,     iz0 + 1), cval * dx*(1-dy)*dz)
                cuda.atomic.add(d_reco, (ix0,     iy0 + 1, iz0 + 1), cval * (1-dx)*dy*dz)
                cuda.atomic.add(d_reco, (ix0 + 1, iy0 + 1, iz0 + 1), cval * dx*dy*dz)

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


def forward_cone_3d(
    volume: np.ndarray,
    n_views: int,
    n_u: int, n_v: int,
    du: float, dv: float,
    angles: np.ndarray,
    src_dist: float,
    iso_dist: float,
):
    """Computes the 3D cone-beam forward projection."""
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
    """Computes the 3D cone-beam back-projection."""
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