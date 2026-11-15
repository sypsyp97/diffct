"""CUDA kernels for 3D parallel-beam projections along infinite lines.

Each detector pixel defines one line through its world point with the
direction ``ray_dir`` of its view. The line is clipped to the volume box and
traversed with cell-constant Siddon ray tracing. Forward projection,
backprojection and the geometry VJP share one ray setup and one traversal,
so they visit the same cells with the same segment lengths.

Coordinates are voxel units: a physical coordinate ``X_a`` maps to
``X_a / s_a`` with the xyz pitches ``s_a``. The voxel-space direction
``rho = ray_dir / s`` is normalised to ``d = rho / |rho|``, so ray parameters
are voxel-space distances. A voxel-space distance ``t`` has the physical
length ``q * t`` with ``q = |diag(s) d|``.
"""

import math
import numpy as np
from numba import cuda

from .spacing import _axis_pitch, _ray_scale

from ..constants import (
    _FASTMATH_DECORATOR,
    _BIG,
    _TINY,
    _ZERO,
    _ONE,
)


@cuda.jit(device=True, inline="always")
def _parallel_3d_ray(iview, iu, iv, n_u, n_v, du, dv, d_ray_dir, d_det_center,
                     d_det_u_vec, d_det_v_vec, voxel_spacing, d_detector_positions):
    """Return one pixel line in float64 voxel units.

    Returns the pixel point, the unit voxel-space direction, the length of
    ``ray_dir / s`` before normalisation, and the physical detector offsets
    ``u_phys`` and ``v_phys`` of the pixel centre.
    """
    pitch_x = np.float64(_axis_pitch(voxel_spacing, 0))
    pitch_y = np.float64(_axis_pitch(voxel_spacing, 1))
    pitch_z = np.float64(_axis_pitch(voxel_spacing, 2))

    dir_x = np.float64(d_ray_dir[iview, 0]) / pitch_x
    dir_y = np.float64(d_ray_dir[iview, 1]) / pitch_y
    dir_z = np.float64(d_ray_dir[iview, 2]) / pitch_z
    length = math.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)
    dir_x = dir_x / length
    dir_y = dir_y / length
    dir_z = dir_z / length

    u_phys = (np.float64(iu) + 0.5 - np.float64(n_u) * 0.5) * np.float64(du)
    v_phys = (np.float64(iv) + 0.5 - np.float64(n_v) * 0.5) * np.float64(dv)
    if d_detector_positions.size == 0:
        pnt_x = (np.float64(d_det_center[iview, 0]) + u_phys * np.float64(d_det_u_vec[iview, 0])
                 + v_phys * np.float64(d_det_v_vec[iview, 0])) / pitch_x
        pnt_y = (np.float64(d_det_center[iview, 1]) + u_phys * np.float64(d_det_u_vec[iview, 1])
                 + v_phys * np.float64(d_det_v_vec[iview, 1])) / pitch_y
        pnt_z = (np.float64(d_det_center[iview, 2]) + u_phys * np.float64(d_det_u_vec[iview, 2])
                 + v_phys * np.float64(d_det_v_vec[iview, 2])) / pitch_z
    else:
        pnt_x = np.float64(d_detector_positions[iview, iu, iv, 0]) / pitch_x
        pnt_y = np.float64(d_detector_positions[iview, iu, iv, 1]) / pitch_y
        pnt_z = np.float64(d_detector_positions[iview, iu, iv, 2]) / pitch_z
    return pnt_x, pnt_y, pnt_z, dir_x, dir_y, dir_z, length, u_phys, v_phys


@cuda.jit(device=True, inline="always")
def _clip_axis(point, direction, half, axis, t_min, t_max, ent_axis, ex_axis):
    """Clip ``[t_min, t_max]`` to the slab ``-half <= point + t * direction <= half``.

    The axis that sets a new bound is recorded as the entry or exit axis. A
    line parallel to the slab and outside it returns an empty interval.
    """
    if direction != 0.0:
        first = (-half - point) / direction
        second = (half - point) / direction
        lower = min(first, second)
        upper = max(first, second)
        if lower > t_min:
            t_min = lower
            ent_axis = axis
        if upper < t_max:
            t_max = upper
            ex_axis = axis
    elif point < -half or point > half:
        t_max = t_min
    return t_min, t_max, ent_axis, ex_axis


@cuda.jit(device=True, inline="always")
def _parallel_3d_clip(pnt_x, pnt_y, pnt_z, dir_x, dir_y, dir_z, cx, cy, cz):
    """Clip the infinite line to the volume box in float64.

    Returns ``t_min``, ``t_max`` and the entry and exit axes. The line misses
    the box when ``t_min >= t_max``.
    """
    t_min = -np.float64(_BIG)
    t_max = np.float64(_BIG)
    ent_axis = -1
    ex_axis = -1
    t_min, t_max, ent_axis, ex_axis = _clip_axis(pnt_x, dir_x, cx, 0, t_min, t_max, ent_axis, ex_axis)
    t_min, t_max, ent_axis, ex_axis = _clip_axis(pnt_y, dir_y, cy, 1, t_min, t_max, ent_axis, ex_axis)
    t_min, t_max, ent_axis, ex_axis = _clip_axis(pnt_z, dir_z, cz, 2, t_min, t_max, ent_axis, ex_axis)
    return t_min, t_max, ent_axis, ex_axis


@cuda.jit(device=True, inline="always")
def _parallel_3d_trace(d_vol, Nx, Ny, Nz, ent_x, ent_y, ent_z, ray_x, ray_y, ray_z, t_end,
                       cx, cy, cz, ent_axis, g, scatter):
    """Traverse one clipped line in float32 from its entry point.

    With ``scatter`` False, read the volume and return the voxel-unit line
    integral ``S = sum f * segment`` together with the face sums of the
    geometry VJP. With ``scatter`` True, add ``g * segment`` into each cell
    with atomics; the returned sums are then zero.

    Each crossed face ``j`` on axis ``a`` gets the coefficient ``c_j``:
    ``-f_first`` at the entry face, ``f_before - f_after`` inside, and
    ``+f_last`` at the exit face, which the caller adds. ``C_a`` sums
    ``c_j``; ``CT_a`` sums ``c_j`` times the local face parameter.
    """
    accum = _ZERO
    c_x = _ZERO
    c_y = _ZERO
    c_z = _ZERO
    ct_x = _ZERO
    ct_y = _ZERO
    ct_z = _ZERO
    f_prev = _ZERO
    pend_axis = ent_axis
    pend_t = _ZERO
    first = True
    t = _ZERO

    ix = int(math.floor(ent_x + cx))
    iy = int(math.floor(ent_y + cy))
    iz = int(math.floor(ent_z + cz))
    step_x = 1 if ray_x >= 0 else -1
    step_y = 1 if ray_y >= 0 else -1
    step_z = 1 if ray_z >= 0 else -1
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

    while t < t_end:
        inside = 0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz
        f = _ZERO
        if inside and not scatter:
            f = d_vol[ix, iy, iz]
        segment = min(tx, ty, tz, t_end) - t
        if inside and segment > _ZERO:
            if scatter:
                cuda.atomic.add(d_vol, (ix, iy, iz), g * segment)
            else:
                accum += f * segment
        if pend_axis >= 0:
            c = (-f) if first else (f_prev - f)
            if pend_axis == 0:
                c_x += c
                ct_x += c * pend_t
            elif pend_axis == 1:
                c_y += c
                ct_y += c * pend_t
            else:
                c_z += c
                ct_z += c * pend_t
        first = False
        f_prev = f

        if tx <= ty and tx <= tz:
            t = tx
            pend_axis = 0
            ix += step_x
            tx += dt_x
        elif ty <= tx and ty <= tz:
            t = ty
            pend_axis = 1
            iy += step_y
            ty += dt_y
        else:
            t = tz
            pend_axis = 2
            iz += step_z
            tz += dt_z
        pend_t = t
    return accum, c_x, c_y, c_z, ct_x, ct_y, ct_z, f_prev


@_FASTMATH_DECORATOR
def _parallel_3d_forward_kernel(
    d_vol, Nx, Ny, Nz,
    d_sino, n_views, n_u, n_v,
    du, dv, d_ray_dir, d_det_center, d_det_u_vec, d_det_v_vec,
    cx, cy, cz, voxel_spacing, d_detector_positions,
):
    """Project a native ``(W, H, D)`` volume along parallel3d pixel lines.

    ``d_detector_positions`` is empty for a flat detector; otherwise it holds
    ``(views, U, V, 3)`` world points and replaces the flat composition. Every
    pixel is written, including lines that miss the volume.
    """
    iv, iu, iview = cuda.grid(3)
    if iview >= n_views or iu >= n_u or iv >= n_v:
        return
    pnt_x, pnt_y, pnt_z, dir_x, dir_y, dir_z, _length, _u_phys, _v_phys = _parallel_3d_ray(
        iview, iu, iv, n_u, n_v, du, dv, d_ray_dir, d_det_center, d_det_u_vec, d_det_v_vec,
        voxel_spacing, d_detector_positions,
    )
    t_min, t_max, ent_axis, _ex_axis = _parallel_3d_clip(pnt_x, pnt_y, pnt_z, dir_x, dir_y, dir_z, cx, cy, cz)
    if t_min >= t_max:
        d_sino[iview, iu, iv] = _ZERO
        return
    accum = _parallel_3d_trace(
        d_vol, Nx, Ny, Nz,
        np.float32(pnt_x + t_min * dir_x), np.float32(pnt_y + t_min * dir_y), np.float32(pnt_z + t_min * dir_z),
        np.float32(dir_x), np.float32(dir_y), np.float32(dir_z), np.float32(t_max - t_min),
        cx, cy, cz, ent_axis, _ZERO, False,
    )[0]
    d_sino[iview, iu, iv] = accum * _ray_scale(voxel_spacing, dir_x, dir_y, dir_z)


@_FASTMATH_DECORATOR
def _parallel_3d_backward_kernel(
    d_sino, n_views, n_u, n_v,
    d_vol, Nx, Ny, Nz,
    du, dv, d_ray_dir, d_det_center, d_det_u_vec, d_det_v_vec,
    cx, cy, cz, voxel_spacing, d_detector_positions,
):
    """Add the matched adjoint of one parallel3d sinogram into a native volume."""
    iv, iu, iview = cuda.grid(3)
    if iview >= n_views or iu >= n_u or iv >= n_v:
        return
    pnt_x, pnt_y, pnt_z, dir_x, dir_y, dir_z, _length, _u_phys, _v_phys = _parallel_3d_ray(
        iview, iu, iv, n_u, n_v, du, dv, d_ray_dir, d_det_center, d_det_u_vec, d_det_v_vec,
        voxel_spacing, d_detector_positions,
    )
    g = d_sino[iview, iu, iv] * _ray_scale(voxel_spacing, dir_x, dir_y, dir_z)
    if g == _ZERO:
        return
    t_min, t_max, ent_axis, _ex_axis = _parallel_3d_clip(pnt_x, pnt_y, pnt_z, dir_x, dir_y, dir_z, cx, cy, cz)
    if t_min >= t_max:
        return
    _parallel_3d_trace(
        d_vol, Nx, Ny, Nz,
        np.float32(pnt_x + t_min * dir_x), np.float32(pnt_y + t_min * dir_y), np.float32(pnt_z + t_min * dir_z),
        np.float32(dir_x), np.float32(dir_y), np.float32(dir_z), np.float32(t_max - t_min),
        cx, cy, cz, ent_axis, g, True,
    )


@cuda.jit(device=True, inline="always")
def _parallel_3d_axis_terms(c, ct, t_min, ray, direction, pitch, scale, integral):
    """Return the point term and the unprojected direction term of one axis.

    With ``p = q * S`` and face parameters ``t = (b - P_a) / d_a``:
    ``dp/dP_a = -q * C_a / (d_a * s_a)`` for the physical point, and
    ``dp/dd_a = -q * T_a / d_a + S * s_a^2 * d_a / q`` for the unit
    voxel-space direction, where ``T_a = t_min * C_a + CT_a``.
    """
    point = np.float64(0.0)
    unit = np.float64(0.0)
    if abs(ray) > _TINY:
        point = -scale * c / (np.float64(ray) * pitch)
        unit = -scale * (t_min * c + ct) / np.float64(ray)
    unit += integral * pitch * pitch * direction / scale
    return point, unit


@_FASTMATH_DECORATOR
def _parallel_3d_geometry_vjp_kernel(
    d_vol, Nx, Ny, Nz,
    d_grad_sino, n_views, n_u, n_v,
    du, dv, d_ray_dir, d_det_center, d_det_u_vec, d_det_v_vec,
    cx, cy, cz, voxel_spacing,
    d_grad_ray_dir, d_grad_det_center, d_grad_det_u, d_grad_det_v,
    d_detector_positions, d_grad_detector_positions,
):
    """Accumulate parallel3d geometry gradients of ``<grad_sino, A x>``.

    The direction gradient is the derivative through ``d = rho / |rho|``, so it
    is orthogonal to ``ray_dir``: the line integral does not depend on the
    length of ``ray_dir``. The point gradient goes to the flat detector frame
    (center, ``u_phys`` times ``det_u``, ``v_phys`` times ``det_v``) or, with
    detector positions, to each pixel point.
    """
    iv, iu, iview = cuda.grid(3)
    if iview >= n_views or iu >= n_u or iv >= n_v:
        return
    g = d_grad_sino[iview, iu, iv]
    if g == _ZERO:
        return
    pnt_x, pnt_y, pnt_z, dir_x, dir_y, dir_z, length, u_phys, v_phys = _parallel_3d_ray(
        iview, iu, iv, n_u, n_v, du, dv, d_ray_dir, d_det_center, d_det_u_vec, d_det_v_vec,
        voxel_spacing, d_detector_positions,
    )
    t_min, t_max, ent_axis, ex_axis = _parallel_3d_clip(pnt_x, pnt_y, pnt_z, dir_x, dir_y, dir_z, cx, cy, cz)
    if t_min >= t_max:
        return
    ray_x = np.float32(dir_x)
    ray_y = np.float32(dir_y)
    ray_z = np.float32(dir_z)
    t_end = np.float32(t_max - t_min)
    accum, c_x, c_y, c_z, ct_x, ct_y, ct_z, f_last = _parallel_3d_trace(
        d_vol, Nx, Ny, Nz,
        np.float32(pnt_x + t_min * dir_x), np.float32(pnt_y + t_min * dir_y), np.float32(pnt_z + t_min * dir_z),
        ray_x, ray_y, ray_z, t_end, cx, cy, cz, ent_axis, _ZERO, False,
    )
    if ex_axis == 0:
        c_x += f_last
        ct_x += f_last * t_end
    elif ex_axis == 1:
        c_y += f_last
        ct_y += f_last * t_end
    elif ex_axis == 2:
        c_z += f_last
        ct_z += f_last * t_end

    scale = np.float64(_ray_scale(voxel_spacing, dir_x, dir_y, dir_z))
    pitch_x = np.float64(_axis_pitch(voxel_spacing, 0))
    pitch_y = np.float64(_axis_pitch(voxel_spacing, 1))
    pitch_z = np.float64(_axis_pitch(voxel_spacing, 2))
    integral = np.float64(accum)
    gp_x, gd_x = _parallel_3d_axis_terms(c_x, ct_x, t_min, ray_x, dir_x, pitch_x, scale, integral)
    gp_y, gd_y = _parallel_3d_axis_terms(c_y, ct_y, t_min, ray_y, dir_y, pitch_y, scale, integral)
    gp_z, gd_z = _parallel_3d_axis_terms(c_z, ct_z, t_min, ray_z, dir_z, pitch_z, scale, integral)
    radial = gd_x * dir_x + gd_y * dir_y + gd_z * dir_z
    gr_x = (gd_x - radial * dir_x) / (length * pitch_x)
    gr_y = (gd_y - radial * dir_y) / (length * pitch_y)
    gr_z = (gd_z - radial * dir_z) / (length * pitch_z)

    cuda.atomic.add(d_grad_ray_dir, (iview, 0), np.float32(g * gr_x))
    cuda.atomic.add(d_grad_ray_dir, (iview, 1), np.float32(g * gr_y))
    cuda.atomic.add(d_grad_ray_dir, (iview, 2), np.float32(g * gr_z))
    if d_detector_positions.size == 0:
        cuda.atomic.add(d_grad_det_center, (iview, 0), np.float32(g * gp_x))
        cuda.atomic.add(d_grad_det_center, (iview, 1), np.float32(g * gp_y))
        cuda.atomic.add(d_grad_det_center, (iview, 2), np.float32(g * gp_z))
        cuda.atomic.add(d_grad_det_u, (iview, 0), np.float32(g * u_phys * gp_x))
        cuda.atomic.add(d_grad_det_u, (iview, 1), np.float32(g * u_phys * gp_y))
        cuda.atomic.add(d_grad_det_u, (iview, 2), np.float32(g * u_phys * gp_z))
        cuda.atomic.add(d_grad_det_v, (iview, 0), np.float32(g * v_phys * gp_x))
        cuda.atomic.add(d_grad_det_v, (iview, 1), np.float32(g * v_phys * gp_y))
        cuda.atomic.add(d_grad_det_v, (iview, 2), np.float32(g * v_phys * gp_z))
    else:
        d_grad_detector_positions[iview, iu, iv, 0] = np.float32(g * gp_x)
        d_grad_detector_positions[iview, iu, iv, 1] = np.float32(g * gp_y)
        d_grad_detector_positions[iview, iu, iv, 2] = np.float32(g * gp_z)
