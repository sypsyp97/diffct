import math
import numpy as np
from numba import cuda

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

    num_steps = int(math.ceil((t_max - t_min) / step))
    start_t = t_min + step * 0.5 # Start at the center of the first step

    for n in range(num_steps):
        t = start_t + n * step
        if t >= t_max: # Ensure t doesn't exceed t_max
            break
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
    num_steps = int(math.ceil((t_max - t_min) / step))
    start_t = t_min + step * 0.5 # Start at the center of the first step

    for n in range(num_steps):
        t = start_t + n * step
        if t >= t_max: # Ensure t doesn't exceed t_max
            break
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


def forward_parallel_2d(
    image: np.ndarray,
    n_views: int,
    n_det: int,
    det_spacing: float,
    angles: np.ndarray,
    step_size: float = 0.5,
):
    # host‑side preparations --------------------------------------------------
    # Host layout (y, x), kernel expects (x, y) -> transpose
    image_np = image.astype(_DTYPE, copy=False).T
    Nx, Ny = image_np.shape # Now Nx=W, Ny=H
    diag   = _DTYPE(math.sqrt(Nx * Nx + Ny * Ny)) * 0.5 # Half diagonal

    d_image = cuda.to_device(image_np) # Send transposed data
    d_cos, d_sin = _trig_tables(angles, _DTYPE)
    d_sino  = cuda.device_array((n_views, n_det), dtype=_DTYPE)

    (grid, tpb) = _grid_2d(n_views, n_det)

    # Center calculation uses transposed dims (Nx=W, Ny=H)
    cx = _DTYPE((Nx - 1) * 0.5)
    cy = _DTYPE((Ny - 1) * 0.5)

    forward_parallel_2d_kernel[grid, tpb](
        d_image, Nx, Ny, diag, d_sino, # Use transposed Nx, Ny, half-diag
        n_views, n_det, _DTYPE(det_spacing),
        d_cos, d_sin,
        _DTYPE(step_size), cx, cy # Use centers based on transposed dims
    )
    # Sinogram layout is correct, no transpose needed
    return d_sino.copy_to_host()


def back_parallel_2d(
    sinogram: np.ndarray,
    Nx: int, Ny: int,
    det_spacing: float,
    angles: np.ndarray,
    step_size: float = 0.5,
):
    # Input Nx, Ny are target reco dimensions (H, W)
    sinogram = sinogram.astype(_DTYPE, copy=False)
    n_views, n_det = sinogram.shape
    # Use transposed dimensions (W, H) for kernel consistency
    kernel_Nx, kernel_Ny = Ny, Nx
    diag = _DTYPE(math.sqrt(kernel_Nx * kernel_Nx + kernel_Ny * kernel_Ny)) * 0.5 # Half diagonal

    d_sino = cuda.to_device(sinogram)
    d_cos, d_sin = _trig_tables(angles, _DTYPE)
    # Create reco buffer with kernel layout (x, y) / (W, H)
    d_reco = cuda.to_device(np.zeros((kernel_Nx, kernel_Ny), dtype=_DTYPE))

    (grid, tpb) = _grid_2d(n_views, n_det)
    # Center calculation uses kernel dims (W, H)
    cx = _DTYPE((kernel_Nx - 1) * 0.5)
    cy = _DTYPE((kernel_Ny - 1) * 0.5)

    back_parallel_2d_kernel[grid, tpb](
        d_sino, kernel_Nx, kernel_Ny, diag, d_reco, # Use kernel dims, half-diag
        n_views, n_det, _DTYPE(det_spacing),
        d_cos, d_sin,
        _DTYPE(step_size), cx, cy # Use centers based on kernel dims
    )
    # Kernel output d_reco is (x, y), need (y, x) for host -> transpose back
    return d_reco.copy_to_host().T


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
    num_steps = int(math.ceil(length / step))
    start_t = step * 0.5 # Start at the center of the first step (t=0)

    for n in range(num_steps):
        t = start_t + n * step
        if t >= length: # Ensure t doesn't exceed length
            break
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

    num_steps = int(math.ceil(length / step))
    start_t = step * 0.5 # Start at the center of the first step (t=0)

    for n in range(num_steps):
        t = start_t + n * step
        if t >= length: # Ensure t doesn't exceed length
            break
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
    # Host layout (y, x), kernel expects (x, y) -> transpose
    image_np = image.astype(_DTYPE, copy=False).T
    Nx, Ny = image_np.shape # Now Nx=W, Ny=H
    d_image = cuda.to_device(image_np) # Send transposed data
    d_cos, d_sin = _trig_tables(angles, _DTYPE)
    d_sino = cuda.device_array((n_views, n_det), dtype=_DTYPE)

    (grid, tpb) = _grid_2d(n_views, n_det)
    # Center calculation uses transposed dims (Nx=W, Ny=H)
    cx = _DTYPE((Nx - 1) * 0.5)
    cy = _DTYPE((Ny - 1) * 0.5)

    forward_fan_2d_kernel[grid, tpb](
        d_image, Nx, Ny, d_sino, # Use transposed Nx, Ny
        n_views, n_det, _DTYPE(det_spacing),
        d_cos, d_sin,
        _DTYPE(src_dist), _DTYPE(iso_dist),
        _DTYPE(step_size), cx, cy # Use centers based on transposed dims
    )
    # Sinogram layout is correct, no transpose needed
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
    # Input Nx, Ny are target reco dimensions (H, W)
    sinogram = sinogram.astype(_DTYPE, copy=False)
    n_views, n_det = sinogram.shape
    # Use transposed dimensions (W, H) for kernel consistency
    kernel_Nx, kernel_Ny = Ny, Nx

    d_sino = cuda.to_device(sinogram)
    d_cos, d_sin = _trig_tables(angles, _DTYPE)
    # Create reco buffer with kernel layout (x, y) / (W, H)
    d_reco = cuda.to_device(np.zeros((kernel_Nx, kernel_Ny), dtype=_DTYPE))

    (grid, tpb) = _grid_2d(n_views, n_det)
    # Center calculation uses kernel dims (W, H)
    cx = _DTYPE((kernel_Nx - 1) * 0.5)
    cy = _DTYPE((kernel_Ny - 1) * 0.5)

    back_fan_2d_kernel[grid, tpb](
        d_sino,
        n_views, n_det,
        kernel_Nx, kernel_Ny, # Use kernel dims
        d_reco, # Reco buffer (W, H)
        _DTYPE(det_spacing),
        d_cos, d_sin,
        _DTYPE(src_dist), _DTYPE(iso_dist),
        _DTYPE(step_size), cx, cy # Use centers based on kernel dims
    )
    # Kernel output d_reco is (x, y), need (y, x) for host -> transpose back
    return d_reco.copy_to_host().T


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
    num_steps = int(math.ceil(length / step))
    start_t = step * 0.5 # Start at the center of the first step (t=0)

    for n in range(num_steps):
        t = start_t + n * step
        if t >= length: # Ensure t doesn't exceed length
            break
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

    num_steps = int(math.ceil(length / step))
    start_t = step * 0.5 # Start at the center of the first step (t=0)

    for n in range(num_steps):
        t = start_t + n * step
        if t >= length: # Ensure t doesn't exceed length
            break
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
    # Host layout (z, y, x), kernel expects (x, y, z) -> transpose
    volume_np = volume.astype(_DTYPE, copy=False).transpose((2, 1, 0))
    Nx, Ny, Nz = volume_np.shape # Now Nx=W, Ny=H, Nz=D
    d_vol = cuda.to_device(volume_np) # Send transposed data
    d_cos, d_sin = _trig_tables(angles, _DTYPE)
    d_sino = cuda.device_array((n_views, n_u, n_v), dtype=_DTYPE)

    (grid, tpb) = _grid_3d(n_views, n_u, n_v)
    # Center calculation uses transposed dims (Nx=W, Ny=H, Nz=D)
    cx = _DTYPE((Nx - 1) * 0.5)
    cy = _DTYPE((Ny - 1) * 0.5)
    cz = _DTYPE((Nz - 1) * 0.5)

    forward_cone_3d_kernel[grid, tpb](
        d_vol, Nx, Ny, Nz, # Use transposed Nx, Ny, Nz
        d_sino,
        n_views, n_u, n_v,
        _DTYPE(du), _DTYPE(dv),
        d_cos, d_sin,
        _DTYPE(src_dist), _DTYPE(iso_dist),
        _DTYPE(step_size), cx, cy, cz # Use centers based on transposed dims
    )
    # Sinogram layout is correct, no transpose needed
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
    # Input Nx, Ny, Nz are target reco dimensions (D, H, W)
    sinogram = sinogram.astype(_DTYPE, copy=False)
    n_views, n_u, n_v = sinogram.shape
    # Use transposed dimensions (W, H, D) for kernel consistency
    kernel_Nx, kernel_Ny, kernel_Nz = Nz, Ny, Nx

    d_sino = cuda.to_device(sinogram)
    d_cos, d_sin = _trig_tables(angles, _DTYPE)
    # Create reco buffer with kernel layout (x, y, z) / (W, H, D)
    d_reco = cuda.to_device(np.zeros((kernel_Nx, kernel_Ny, kernel_Nz), dtype=_DTYPE))

    (grid, tpb) = _grid_3d(n_views, n_u, n_v)
    # Center calculation uses kernel dims (W, H, D)
    cx = _DTYPE((kernel_Nx - 1) * 0.5)
    cy = _DTYPE((kernel_Ny - 1) * 0.5)
    cz = _DTYPE((kernel_Nz - 1) * 0.5)

    back_cone_3d_kernel[grid, tpb](
        d_sino,
        n_views, n_u, n_v,
        kernel_Nx, kernel_Ny, kernel_Nz, # Use kernel dims
        d_reco, # Reco buffer (W, H, D)
        _DTYPE(du), _DTYPE(dv),
        d_cos, d_sin,
        _DTYPE(src_dist), _DTYPE(iso_dist),
        _DTYPE(step_size), cx, cy, cz # Use centers based on kernel dims
    )
    # Kernel output d_reco is (x, y, z), need (z, y, x) for host -> transpose back
    return d_reco.copy_to_host().transpose((2, 1, 0))
