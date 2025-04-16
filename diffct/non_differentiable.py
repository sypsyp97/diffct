import math
import numpy as np
from numba import cuda, float32

# ------------------------------------------------------------------
# GLOBAL SETTINGS
# ------------------------------------------------------------------

_DTYPE             = np.float32          # switch to float64 if needed
_TPB_2D            = (16, 16)            # threads per block for 2‑D kernels
_TPB_3D            = (8,  8,  8)         # threads per block for 3‑D kernels
_FASTMATH_DECORATOR = cuda.jit(fastmath=True)

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
# 2‑D PARALLEL GEOMETRY
# ------------------------------------------------------------------

@_FASTMATH_DECORATOR
def forward_parallel_2d_kernel(
    d_image,
    Nx, Ny,                         # image size
    diag,                           # diagonal length of the image
    d_sino,
    n_views, n_det,
    det_spacing,
    d_cos, d_sin,                   # trig lookup tables
    step, cx, cy                    # ray‑marching parameters
):
    iview, idet = cuda.grid(2)
    if iview >= n_views or idet >= n_det:
        return

    cos_a = d_cos[iview]
    sin_a = d_sin[iview]
    u     = (idet - (n_det - 1) * 0.5) * det_spacing

    # integral limits (covers the whole image square)
    t_min, t_max = -diag, diag
    t            = t_min
    accum        = 0.0

    while t < t_max:
        x  = u * (-sin_a) + t * cos_a
        y  = u * ( cos_a) + t * sin_a
        ix = x + cx
        iy = y + cy
        ix0 = int(math.floor(ix))
        iy0 = int(math.floor(iy))

        if 0 <= ix0 < Nx - 1 and 0 <= iy0 < Ny - 1:
            dx = ix - ix0
            dy = iy - iy0
            accum += (
                d_image[ix0,     iy0]     * (1 - dx) * (1 - dy) +
                d_image[ix0 + 1, iy0]     * dx       * (1 - dy) +
                d_image[ix0,     iy0 + 1] * (1 - dx) * dy       +
                d_image[ix0 + 1, iy0 + 1] * dx       * dy
            ) * step
        t += step

    d_sino[iview, idet] = accum


@_FASTMATH_DECORATOR
def back_parallel_2d_kernel(
    d_sino,
    Nx, Ny, diag,
    d_reco,
    n_views, n_det, det_spacing,
    d_cos, d_sin,
    step, cx, cy
):
    iview, idet = cuda.grid(2)
    if iview >= n_views or idet >= n_det:
        return

    val   = d_sino[iview, idet]
    cos_a = d_cos[iview]
    sin_a = d_sin[iview]
    u     = (idet - (n_det - 1) * 0.5) * det_spacing

    t_min, t_max = -diag, diag
    t = t_min
    while t < t_max:
        x  = u * (-sin_a) + t * cos_a
        y  = u * ( cos_a) + t * sin_a
        ix = x + cx
        iy = y + cy
        ix0 = int(math.floor(ix))
        iy0 = int(math.floor(iy))

        if 0 <= ix0 < Nx - 1 and 0 <= iy0 < Ny - 1:
            dx = ix - ix0
            dy = iy - iy0
            cval = val * step
            cuda.atomic.add(d_reco, (ix0,     iy0),     cval * (1 - dx) * (1 - dy))
            cuda.atomic.add(d_reco, (ix0 + 1, iy0),     cval * dx       * (1 - dy))
            cuda.atomic.add(d_reco, (ix0,     iy0 + 1), cval * (1 - dx) * dy)
            cuda.atomic.add(d_reco, (ix0 + 1, iy0 + 1), cval * dx       * dy)
        t += step


def forward_parallel_2d(
    image: np.ndarray,
    n_views: int,
    n_det: int,
    det_spacing: float,
    angles: np.ndarray,
    step_size: float = 0.5,
):
    # host‑side preparations --------------------------------------------------
    image = image.astype(_DTYPE, copy=False)
    Nx, Ny = image.shape
    diag   = _DTYPE(math.sqrt(Nx * Nx + Ny * Ny))

    d_image = cuda.to_device(image)
    d_cos, d_sin = _trig_tables(angles, _DTYPE)
    d_sino  = cuda.device_array((n_views, n_det), dtype=_DTYPE)

    (grid, tpb) = _grid_2d(n_views, n_det)

    cx = _DTYPE((Nx - 1) * 0.5)
    cy = _DTYPE((Ny - 1) * 0.5)

    forward_parallel_2d_kernel[grid, tpb](
        d_image, Nx, Ny, diag, d_sino,
        n_views, n_det, _DTYPE(det_spacing),
        d_cos, d_sin,
        _DTYPE(step_size), cx, cy
    )
    return d_sino.copy_to_host()


def back_parallel_2d(
    sinogram: np.ndarray,
    Nx: int, Ny: int,
    det_spacing: float,
    angles: np.ndarray,
    step_size: float = 0.5,
):
    sinogram = sinogram.astype(_DTYPE, copy=False)
    n_views, n_det = sinogram.shape
    diag = _DTYPE(math.sqrt(Nx * Nx + Ny * Ny))

    d_sino = cuda.to_device(sinogram)
    d_cos, d_sin = _trig_tables(angles, _DTYPE)
    d_reco = cuda.to_device(np.zeros((Nx, Ny), dtype=_DTYPE))

    (grid, tpb) = _grid_2d(n_views, n_det)
    cx = _DTYPE((Nx - 1) * 0.5)
    cy = _DTYPE((Ny - 1) * 0.5)

    back_parallel_2d_kernel[grid, tpb](
        d_sino, Nx, Ny, diag, d_reco,
        n_views, n_det, _DTYPE(det_spacing),
        d_cos, d_sin,
        _DTYPE(step_size), cx, cy
    )
    return d_reco.copy_to_host()


# ------------------------------------------------------------------
# 2‑D FAN GEOMETRY
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
    step, cx, cy
):
    iview, idet = cuda.grid(2)
    if iview >= n_views or idet >= n_det:
        return

    cos_a = d_cos[iview]
    sin_a = d_sin[iview]
    u     = (idet - (n_det - 1) * 0.5) * det_spacing

    # ray end points ----------------------------------------------------------
    src_x = -iso_dist * sin_a
    src_y =  iso_dist * cos_a

    det_x = (src_dist - iso_dist) * sin_a + u * cos_a
    det_y = -(src_dist - iso_dist) * cos_a + u * sin_a

    dir_x = det_x - src_x
    dir_y = det_y - src_y
    length = math.sqrt(dir_x * dir_x + dir_y * dir_y)
    inv_len = 1.0 / length
    dir_x *= inv_len
    dir_y *= inv_len

    accum = 0.0
    t     = 0.0
    while t < length:
        x  = src_x + t * dir_x
        y  = src_y + t * dir_y
        ix = x + cx
        iy = y + cy
        ix0 = int(math.floor(ix))
        iy0 = int(math.floor(iy))
        if 0 <= ix0 < Nx - 1 and 0 <= iy0 < Ny - 1:
            dx = ix - ix0
            dy = iy - iy0
            accum += (
                d_image[ix0,     iy0]     * (1 - dx) * (1 - dy) +
                d_image[ix0 + 1, iy0]     * dx       * (1 - dy) +
                d_image[ix0,     iy0 + 1] * (1 - dx) * dy       +
                d_image[ix0 + 1, iy0 + 1] * dx       * dy
            ) * step
        t += step

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
    step, cx, cy
):
    iview, idet = cuda.grid(2)
    if iview >= n_views or idet >= n_det:
        return

    val   = d_sino[iview, idet]
    cos_a = d_cos[iview]
    sin_a = d_sin[iview]
    u     = (idet - (n_det - 1) * 0.5) * det_spacing

    src_x = -iso_dist * sin_a
    src_y =  iso_dist * cos_a
    det_x = (src_dist - iso_dist) * sin_a + u * cos_a
    det_y = -(src_dist - iso_dist) * cos_a + u * sin_a

    dir_x = det_x - src_x
    dir_y = det_y - src_y
    length = math.sqrt(dir_x * dir_x + dir_y * dir_y)
    inv_len = 1.0 / length
    dir_x *= inv_len
    dir_y *= inv_len

    t = 0.0
    while t < length:
        x  = src_x + t * dir_x
        y  = src_y + t * dir_y
        ix = x + cx
        iy = y + cy
        ix0 = int(math.floor(ix))
        iy0 = int(math.floor(iy))
        if 0 <= ix0 < Nx - 1 and 0 <= iy0 < Ny - 1:
            dx = ix - ix0
            dy = iy - iy0
            cval = val * step
            cuda.atomic.add(d_reco, (ix0,     iy0),     cval * (1 - dx) * (1 - dy))
            cuda.atomic.add(d_reco, (ix0 + 1, iy0),     cval * dx       * (1 - dy))
            cuda.atomic.add(d_reco, (ix0,     iy0 + 1), cval * (1 - dx) * dy)
            cuda.atomic.add(d_reco, (ix0 + 1, iy0 + 1), cval * dx       * dy)
        t += step


def forward_fan_2d(
    image: np.ndarray,
    n_views: int,
    n_det: int,
    det_spacing: float,
    angles: np.ndarray,
    src_dist: float,
    iso_dist: float,
    step_size: float = 0.5,
):
    image = image.astype(_DTYPE, copy=False)
    Nx, Ny = image.shape
    d_image = cuda.to_device(image)
    d_cos, d_sin = _trig_tables(angles, _DTYPE)
    d_sino = cuda.device_array((n_views, n_det), dtype=_DTYPE)

    (grid, tpb) = _grid_2d(n_views, n_det)
    cx = _DTYPE((Nx - 1) * 0.5)
    cy = _DTYPE((Ny - 1) * 0.5)

    forward_fan_2d_kernel[grid, tpb](
        d_image, Nx, Ny, d_sino,
        n_views, n_det, _DTYPE(det_spacing),
        d_cos, d_sin,
        _DTYPE(src_dist), _DTYPE(iso_dist),
        _DTYPE(step_size), cx, cy
    )
    return d_sino.copy_to_host()


def back_fan_2d(
    sinogram: np.ndarray,
    Nx, Ny,
    det_spacing: float,
    angles: np.ndarray,
    src_dist: float,
    iso_dist: float,
    step_size: float = 0.5,
):
    sinogram = sinogram.astype(_DTYPE, copy=False)
    n_views, n_det = sinogram.shape

    d_sino = cuda.to_device(sinogram)
    d_cos, d_sin = _trig_tables(angles, _DTYPE)
    d_reco = cuda.to_device(np.zeros((Nx, Ny), dtype=_DTYPE))

    (grid, tpb) = _grid_2d(n_views, n_det)
    cx = _DTYPE((Nx - 1) * 0.5)
    cy = _DTYPE((Ny - 1) * 0.5)

    back_fan_2d_kernel[grid, tpb](
        d_sino,
        n_views, n_det,
        Nx, Ny,
        d_reco,
        _DTYPE(det_spacing),
        d_cos, d_sin,
        _DTYPE(src_dist), _DTYPE(iso_dist),
        _DTYPE(step_size), cx, cy
    )
    return d_reco.copy_to_host()


# ------------------------------------------------------------------
# 3‑D CONE GEOMETRY
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
    step, cx, cy, cz
):
    iview, iu, iv = cuda.grid(3)
    if iview >= n_views or iu >= n_u or iv >= n_v:
        return

    cos_a = d_cos[iview]
    sin_a = d_sin[iview]
    u     = (iu - (n_u - 1) * 0.5) * du
    v     = (iv - (n_v - 1) * 0.5) * dv

    src_x = -iso_dist * sin_a
    src_y =  iso_dist * cos_a
    src_z = 0.0

    det_x = (src_dist - iso_dist) * sin_a + u * cos_a
    det_y = -(src_dist - iso_dist) * cos_a + u * sin_a
    det_z = v

    dir_x = det_x - src_x
    dir_y = det_y - src_y
    dir_z = det_z - src_z
    length = math.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)
    inv_len = 1.0 / length
    dir_x *= inv_len
    dir_y *= inv_len
    dir_z *= inv_len

    accum = 0.0
    t = 0.0
    while t < length:
        x  = src_x + t * dir_x
        y  = src_y + t * dir_y
        z  = src_z + t * dir_z
        ix = x + cx
        iy = y + cy
        iz = z + cz
        ix0 = int(math.floor(ix))
        iy0 = int(math.floor(iy))
        iz0 = int(math.floor(iz))

        if 0 <= ix0 < Nx - 1 and 0 <= iy0 < Ny - 1 and 0 <= iz0 < Nz - 1:
            dx = ix - ix0
            dy = iy - iy0
            dz = iz - iz0

            accum += (
                d_vol[ix0,     iy0,     iz0]     * (1 - dx) * (1 - dy) * (1 - dz) +
                d_vol[ix0 + 1, iy0,     iz0]     * dx       * (1 - dy) * (1 - dz) +
                d_vol[ix0,     iy0 + 1, iz0]     * (1 - dx) * dy       * (1 - dz) +
                d_vol[ix0,     iy0,     iz0 + 1] * (1 - dx) * (1 - dy) * dz       +
                d_vol[ix0 + 1, iy0 + 1, iz0]     * dx       * dy       * (1 - dz) +
                d_vol[ix0 + 1, iy0,     iz0 + 1] * dx       * (1 - dy) * dz       +
                d_vol[ix0,     iy0 + 1, iz0 + 1] * (1 - dx) * dy       * dz       +
                d_vol[ix0 + 1, iy0 + 1, iz0 + 1] * dx       * dy       * dz
            ) * step
        t += step

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
    step, cx, cy, cz
):
    iview, iu, iv = cuda.grid(3)
    if iview >= n_views or iu >= n_u or iv >= n_v:
        return

    val   = d_sino[iview, iu, iv]
    cos_a = d_cos[iview]
    sin_a = d_sin[iview]
    u     = (iu - (n_u - 1) * 0.5) * du
    v     = (iv - (n_v - 1) * 0.5) * dv

    src_x = -iso_dist * sin_a
    src_y =  iso_dist * cos_a
    src_z = 0.0

    det_x = (src_dist - iso_dist) * sin_a + u * cos_a
    det_y = -(src_dist - iso_dist) * cos_a + u * sin_a
    det_z = v

    dir_x = det_x - src_x
    dir_y = det_y - src_y
    dir_z = det_z - src_z
    length = math.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)
    inv_len = 1.0 / length
    dir_x *= inv_len
    dir_y *= inv_len
    dir_z *= inv_len

    t = 0.0
    while t < length:
        x  = src_x + t * dir_x
        y  = src_y + t * dir_y
        z  = src_z + t * dir_z
        ix = x + cx
        iy = y + cy
        iz = z + cz
        ix0 = int(math.floor(ix))
        iy0 = int(math.floor(iy))
        iz0 = int(math.floor(iz))
        if 0 <= ix0 < Nx - 1 and 0 <= iy0 < Ny - 1 and 0 <= iz0 < Nz - 1:
            dx = ix - ix0
            dy = iy - iy0
            dz = iz - iz0
            cval = val * step
            cuda.atomic.add(d_reco, (ix0,     iy0,     iz0),     cval * (1 - dx) * (1 - dy) * (1 - dz))
            cuda.atomic.add(d_reco, (ix0 + 1, iy0,     iz0),     cval * dx       * (1 - dy) * (1 - dz))
            cuda.atomic.add(d_reco, (ix0,     iy0 + 1, iz0),     cval * (1 - dx) * dy       * (1 - dz))
            cuda.atomic.add(d_reco, (ix0,     iy0,     iz0 + 1), cval * (1 - dx) * (1 - dy) * dz)
            cuda.atomic.add(d_reco, (ix0 + 1, iy0 + 1, iz0),     cval * dx       * dy       * (1 - dz))
            cuda.atomic.add(d_reco, (ix0 + 1, iy0,     iz0 + 1), cval * dx       * (1 - dy) * dz)
            cuda.atomic.add(d_reco, (ix0,     iy0 + 1, iz0 + 1), cval * (1 - dx) * dy       * dz)
            cuda.atomic.add(d_reco, (ix0 + 1, iy0 + 1, iz0 + 1), cval * dx       * dy       * dz)
        t += step


def forward_cone_3d(
    volume: np.ndarray,
    n_views: int,
    n_u: int, n_v: int,
    du: float, dv: float,
    angles: np.ndarray,
    src_dist: float,
    iso_dist: float,
    step_size: float = 0.5,
):
    volume = volume.astype(_DTYPE, copy=False)
    Nx, Ny, Nz = volume.shape
    d_vol = cuda.to_device(volume)
    d_cos, d_sin = _trig_tables(angles, _DTYPE)
    d_sino = cuda.device_array((n_views, n_u, n_v), dtype=_DTYPE)

    (grid, tpb) = _grid_3d(n_views, n_u, n_v)
    cx = _DTYPE((Nx - 1) * 0.5)
    cy = _DTYPE((Ny - 1) * 0.5)
    cz = _DTYPE((Nz - 1) * 0.5)

    forward_cone_3d_kernel[grid, tpb](
        d_vol, Nx, Ny, Nz,
        d_sino,
        n_views, n_u, n_v,
        _DTYPE(du), _DTYPE(dv),
        d_cos, d_sin,
        _DTYPE(src_dist), _DTYPE(iso_dist),
        _DTYPE(step_size), cx, cy, cz
    )
    return d_sino.copy_to_host()


def back_cone_3d(
    sinogram: np.ndarray,
    Nx: int, Ny: int, Nz: int,
    du: float, dv: float,
    angles: np.ndarray,
    src_dist: float,
    iso_dist: float,
    step_size: float = 0.5,
):
    sinogram = sinogram.astype(_DTYPE, copy=False)
    n_views, n_u, n_v = sinogram.shape
    d_sino = cuda.to_device(sinogram)
    d_cos, d_sin = _trig_tables(angles, _DTYPE)
    d_reco = cuda.to_device(np.zeros((Nx, Ny, Nz), dtype=_DTYPE))

    (grid, tpb) = _grid_3d(n_views, n_u, n_v)
    cx = _DTYPE((Nx - 1) * 0.5)
    cy = _DTYPE((Ny - 1) * 0.5)
    cz = _DTYPE((Nz - 1) * 0.5)

    back_cone_3d_kernel[grid, tpb](
        d_sino,
        n_views, n_u, n_v,
        Nx, Ny, Nz,
        d_reco,
        _DTYPE(du), _DTYPE(dv),
        d_cos, d_sin,
        _DTYPE(src_dist), _DTYPE(iso_dist),
        _DTYPE(step_size), cx, cy, cz
    )
    return d_reco.copy_to_host()
