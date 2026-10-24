"""Compose bounded detector-offset fragments into native geometry buffers."""

import math

import numpy as np
from numba import cuda
from numba.core import config
from numba.cuda import libdevice

from .spacing import _axis_pitch, _is_per_axis

if config.ENABLE_CUDASIM:
    # The simulator does not implement the libdevice arithmetic stubs.
    @cuda.jit(device=True, inline=True)
    def _surface_multiply(left, right, single):
        if single:
            return np.float64(np.float32(left) * np.float32(right))
        return np.float64(left) * np.float64(right)

    @cuda.jit(device=True, inline=True)
    def _surface_add(left, right, single):
        if single:
            return np.float64(np.float32(left) + np.float32(right))
        return np.float64(left) + np.float64(right)

else:
    @cuda.jit(device=True, inline=True)
    def _surface_multiply(left, right, single):
        if single:
            return np.float64(libdevice.fmul_rn(np.float32(left), np.float32(right)))
        return libdevice.dmul_rn(np.float64(left), np.float64(right))

    @cuda.jit(device=True, inline=True)
    def _surface_add(left, right, single):
        if single:
            return np.float64(libdevice.fadd_rn(np.float32(left), np.float32(right)))
        return libdevice.dadd_rn(np.float64(left), np.float64(right))


@cuda.jit(cache=True, fastmath=False)
def compose_surface(frames, offsets, source, points, view_index, pixel_start,
                    count, capacity, rank, parallel, frame32, offset32,
                    centre, voxel_spacing):
    """Write one view fragment while preserving the original tensor arithmetic.

    ``frames`` contains one packed source, center, u axis, and optional v axis
    and normal. The caller computes the normal with the framework's cross product.
    ``offsets`` stores component-major u, optional v, and depth columns.
    Each column has ``capacity`` entries. Its final element is a persistent
    validation status: 0 valid, 1 nonfinite, 2 coincident, 3 near limit,
    or 4 far limit. The caller initializes this element before its first launch.

    ``frame32`` gives the arithmetic dtype of each packed frame component.
    ``offset32`` gives the offset arithmetic dtype. Values are stored exactly
    in float64 staging buffers. Results use caller-owned float32 buffers.

    A launch may compose several views: grid row ``y``
    writes view ``view_index + y`` from the ``y``-th of ``gridDim.y`` equal
    frame packets in ``frames``. A shared offset buffer holds ``rank * capacity``
    values plus status; a larger buffer holds one such packet per output view.
    """
    pixel = cuda.grid(1)
    view_index += cuda.blockIdx.y
    base = cuda.blockIdx.y * (frames.size // cuda.gridDim.y)
    if pixel == 0:
        for axis in range(rank):
            value = frames[base + axis]
            if not parallel:
                value = _surface_add(value, -centre[axis], frame32[0])
            source[view_index, axis] = np.float32(value)
    if pixel >= count:
        return

    offset_base = view_index * rank * capacity if offsets.size > rank * capacity + 1 else 0
    tangent = offsets[offset_base + pixel]
    depth = offsets[offset_base + (rank - 1) * capacity + pixel]
    tangent32 = offset32 and frame32[2]
    point32 = frame32[1] and tangent32
    normal32 = frame32[2]
    vertical = np.float64(0.0)
    vertical32 = False
    if rank == 3:
        vertical = offsets[offset_base + capacity + pixel]
        vertical32 = offset32 and frame32[3]
        point32 = point32 and vertical32
        normal32 = normal32 and frame32[3]
    depth32 = offset32 and normal32
    final32 = point32 and depth32

    finite = math.isfinite(tangent) and math.isfinite(depth)
    if rank == 3:
        finite = finite and math.isfinite(vertical)
    coincident = True
    source_squared = np.float64(0.0)
    point_squared = np.float64(0.0)
    for axis in range(rank):
        point = _surface_add(
            frames[base + rank + axis],
            _surface_multiply(tangent, frames[base + 2 * rank + axis], tangent32),
            frame32[1] and tangent32,
        )
        if rank == 3:
            point = _surface_add(
                point, _surface_multiply(vertical, frames[base + 3 * rank + axis], vertical32),
                point32,
            )
            normal = frames[base + 4 * rank + axis]
        else:
            normal = frames[base + 2 * rank + 1] if axis == 0 else -frames[base + 2 * rank]
        point = _surface_add(point, _surface_multiply(depth, normal, depth32), final32)
        source_value = frames[base + axis]
        source_native = np.float32(source_value)
        point_native = np.float32(point)
        finite = (finite and math.isfinite(source_value) and math.isfinite(point)
                  and math.isfinite(source_native) and math.isfinite(point_native))
        coincident = coincident and source_native == point_native
        range_source, range_point = source_value, point
        if _is_per_axis(voxel_spacing):
            pitch = _axis_pitch(voxel_spacing, axis)
            range_source /= pitch
            range_point /= pitch
        source_squared = _surface_add(
            source_squared, _surface_multiply(range_source, range_source, False), False,
        )
        point_squared = _surface_add(
            point_squared, _surface_multiply(range_point, range_point, False), False,
        )
        points[view_index, pixel_start + pixel, axis] = np.float32(
            _surface_add(point, -centre[axis], final32),
        )

    error = np.float64(0.0)
    if not finite:
        error = np.float64(1.0)
    elif not parallel:
        if coincident:
            error = np.float64(2.0)
        if _is_per_axis(voxel_spacing):
            source_distance = math.sqrt(source_squared)
            point_distance = math.sqrt(point_squared)
        else:
            source_distance = math.sqrt(source_squared) / _axis_pitch(voxel_spacing, 0)
            point_distance = math.sqrt(point_squared) / _axis_pitch(voxel_spacing, 0)
        if min(source_distance, point_distance) > 1e6:
            error = np.float64(3.0)
        if source_distance > 1e15 or point_distance > 1e15:
            error = np.float64(4.0)
    if error != 0.0:
        cuda.atomic.max(offsets, offsets.size - 1, error)
