"""Discrete transposes of the analytical FBP/FDK gather kernels.

Geometry arithmetic and interpolation bounds mirror the gather kernels;
only the final sample operation changes to an atomic detector-bin scatter.
"""

import math

import numpy as np
from numba import cuda

from .spacing import _axis_pitch, _is_per_axis

from ..constants import _EPSILON, _FDK_ACCURACY_DECORATOR, _HALF, _ONE, _ZERO


@_FDK_ACCURACY_DECORATOR
def _parallel_2d_fbp_transpose_kernel(
    d_sino, n_views, n_det,
    d_image, Nx, Ny,
    det_spacing, d_ray_dir, d_det_origin, d_det_u_vec,
    cx, cy, voxel_spacing
):
    """Scatter image cotangents through the parallel analytical gather stencil."""
    ix, iy = cuda.grid(2)
    if ix >= Nx or iy >= Ny:
        return
    if n_det < 2:
        return

    # Sample at voxel centres, matching the cell-constant Siddon projector.
    x_v = np.float32(ix) + _HALF - cx
    y_v = np.float32(iy) + _HALF - cy

    if _is_per_axis(voxel_spacing):
        x_v *= _axis_pitch(voxel_spacing, 0)
        y_v *= _axis_pitch(voxel_spacing, 1)
        coordinate_scale = _ONE
    else:
        coordinate_scale = _axis_pitch(voxel_spacing, 0)

    det_spacing_v = det_spacing / coordinate_scale
    half_u = (np.float32(n_det) - _ONE) * _HALF

    for iview in range(n_views):
        dox = d_det_origin[iview, 0] / coordinate_scale
        doy = d_det_origin[iview, 1] / coordinate_scale

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

        value = d_image[iy, ix]
        cuda.atomic.add(d_sino, (iview, iu0), value * (_ONE - tu))
        cuda.atomic.add(d_sino, (iview, iu0 + 1), value * tu)


@_FDK_ACCURACY_DECORATOR
def _fan_2d_fbp_transpose_kernel(
    d_sino, n_views, n_det,
    d_image, Nx, Ny,
    det_spacing, d_src_pos, d_det_center, d_det_u_vec,
    cx, cy, voxel_spacing, iso_x, iso_y
):
    """Scatter image cotangents through the fan analytical gather stencil."""
    ix, iy = cuda.grid(2)
    if ix >= Nx or iy >= Ny:
        return
    if n_det < 2:
        return

    # Sample at voxel centres, matching the cell-constant Siddon projector.
    x_v = np.float32(ix) + _HALF - cx
    y_v = np.float32(iy) + _HALF - cy

    if _is_per_axis(voxel_spacing):
        x_v *= _axis_pitch(voxel_spacing, 0)
        y_v *= _axis_pitch(voxel_spacing, 1)
        coordinate_scale = _ONE
    else:
        coordinate_scale = _axis_pitch(voxel_spacing, 0)

    det_spacing_v = det_spacing / coordinate_scale
    half_u = (np.float32(n_det) - _ONE) * _HALF

    for iview in range(n_views):
        sx = d_src_pos[iview, 0] / coordinate_scale
        sy = d_src_pos[iview, 1] / coordinate_scale

        dcx = d_det_center[iview, 0] / coordinate_scale
        dcy = d_det_center[iview, 1] / coordinate_scale

        ux = d_det_u_vec[iview, 0]
        uy = d_det_u_vec[iview, 1]

        # Detector normal in 2D: rotate u_vec 90 degrees. Choose the
        # orientation that points from the source toward the detector,
        # so that ``U_n`` is positive for voxels on the detector side.
        nx = -uy
        ny = ux
        # Align normal so that (det_center - src) . n > 0.
        align = (dcx - sx) * nx + (dcy - sy) * ny
        if align < _ZERO:
            nx = -nx
            ny = -ny

        # Perpendicular distance from source to voxel along normal.
        px = x_v - sx
        py = y_v - sy
        U_n = px * nx + py * ny
        if U_n <= _EPSILON:
            continue

        sdd_n = (dcx - sx) * nx + (dcy - sy) * ny
        if sdd_n <= _EPSILON:
            continue
        mag = sdd_n / U_n

        # Source-to-isocenter distance, independent of coordinate origin.
        sid_n = (iso_x - sx) * nx + (iso_y - sy) * ny
        if sid_n <= _EPSILON:
            continue

        hx = sx + mag * px
        hy = sy + mag * py

        rx = hx - dcx
        ry = hy - dcy
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

        w_ratio = sid_n / U_n
        w = w_ratio * w_ratio
        value = d_image[iy, ix] * w
        cuda.atomic.add(d_sino, (iview, iu0), value * (_ONE - tu))
        cuda.atomic.add(d_sino, (iview, iu0 + 1), value * tu)


@_FDK_ACCURACY_DECORATOR
def _cone_3d_fdk_transpose_kernel(
    d_sino, n_views, n_u, n_v,
    d_vol, Nx, Ny, Nz,
    du, dv, d_src_pos, d_det_center, d_det_u_vec, d_det_v_vec,
    cx, cy, cz, voxel_spacing, iso_x, iso_y, iso_z
):
    """Scatter image cotangents through the cone analytical gather stencil."""
    iz, iy, ix = cuda.grid(3)
    if ix >= Nx or iy >= Ny or iz >= Nz:
        return
    if n_u < 2 or n_v < 2:
        return

    # Voxel position in voxel-unit, origin-centred coordinates.
    # Sample at voxel centres, matching the cell-constant Siddon projector.
    x_v = np.float32(ix) + _HALF - cx
    y_v = np.float32(iy) + _HALF - cy
    z_v = np.float32(iz) + _HALF - cz

    if _is_per_axis(voxel_spacing):
        x_v *= _axis_pitch(voxel_spacing, 0)
        y_v *= _axis_pitch(voxel_spacing, 1)
        z_v *= _axis_pitch(voxel_spacing, 2)
        coordinate_scale = _ONE
    else:
        coordinate_scale = _axis_pitch(voxel_spacing, 0)

    du_v = du / coordinate_scale
    dv_v = dv / coordinate_scale

    half_u = (np.float32(n_u) - _ONE) * _HALF
    half_v = (np.float32(n_v) - _ONE) * _HALF

    for iview in range(n_views):
        # Source position in voxel units.
        sx = d_src_pos[iview, 0] / coordinate_scale
        sy = d_src_pos[iview, 1] / coordinate_scale
        sz = d_src_pos[iview, 2] / coordinate_scale

        # Detector centre in voxel units.
        dcx = d_det_center[iview, 0] / coordinate_scale
        dcy = d_det_center[iview, 1] / coordinate_scale
        dcz = d_det_center[iview, 2] / coordinate_scale

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

        omtu = _ONE - tu
        omtv = _ONE - tv
        w_ratio = sid_n / U_n
        w = w_ratio * w_ratio
        value = d_vol[ix, iy, iz] * w
        cuda.atomic.add(d_sino, (iview, iu0, iv0), value * omtu * omtv)
        cuda.atomic.add(d_sino, (iview, iu0 + 1, iv0), value * tu * omtv)
        cuda.atomic.add(d_sino, (iview, iu0, iv0 + 1), value * omtu * tv)
        cuda.atomic.add(d_sino, (iview, iu0 + 1, iv0 + 1), value * tu * tv)
