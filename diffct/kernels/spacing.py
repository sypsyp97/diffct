"""Compile-time scalar and per-axis native voxel metrics."""

import math

import numpy as np
from numba import types
from numba.extending import overload


def _is_per_axis(spacing):
    """Select the spacing representation before CUDA device compilation."""
    return isinstance(spacing, tuple)


@overload(_is_per_axis, target="cuda")
def _is_per_axis_overload(spacing):
    per_axis = isinstance(spacing, types.BaseTuple)

    def impl(spacing):
        return per_axis

    return impl


def _axis_pitch(spacing, axis):
    """Map tensor-order pitches to xyz without changing scalar arithmetic."""
    if isinstance(spacing, tuple):
        return spacing[len(spacing) - 1 - axis]
    return spacing


@overload(_axis_pitch, target="cuda")
def _axis_pitch_overload(spacing, axis):
    if isinstance(spacing, types.BaseTuple):
        rank = len(spacing)

        def impl(spacing, axis):
            return spacing[rank - 1 - axis]
    else:
        def impl(spacing, axis):
            return spacing

    return impl


def _ray_scale_2d(spacing, dx, dy, dz):
    x = np.float64(spacing[-1]) * dx
    y = np.float64(spacing[-2]) * dy
    squared = x * x + y * y
    return np.float32(math.sqrt(squared))


def _ray_scale_3d(spacing, dx, dy, dz):
    x = np.float64(spacing[-1]) * dx
    y = np.float64(spacing[-2]) * dy
    squared = x * x + y * y
    z = np.float64(spacing[0]) * dz
    squared += z * z
    return np.float32(math.sqrt(squared))


def _ray_scale(spacing, dx, dy, dz):
    """Physical length per unit of normalized voxel-space ray distance."""
    if isinstance(spacing, tuple):
        if len(spacing) == 3:
            return _ray_scale_3d(spacing, dx, dy, dz)
        return _ray_scale_2d(spacing, dx, dy, dz)
    return spacing


@overload(_ray_scale, target="cuda")
def _ray_scale_overload(spacing, dx, dy, dz):
    if isinstance(spacing, types.BaseTuple):
        return _ray_scale_3d if len(spacing) == 3 else _ray_scale_2d

    def impl(spacing, dx, dy, dz):
        return spacing

    return impl
