"""Compile-time scalar and per-axis native voxel metrics."""

import math

import numpy as np
from numba import cuda


@cuda.jit(device=True, inline=True)
def _axis_pitch(spacing, axis):
    """Map tensor-order pitches to xyz without changing scalar arithmetic."""
    if isinstance(spacing, tuple):
        return spacing[len(spacing) - 1 - axis]
    return spacing


@cuda.jit(device=True, inline=True)
def _ray_scale(spacing, dx, dy, dz):
    """Physical length per unit of normalized voxel-space ray distance."""
    if isinstance(spacing, tuple):
        x = np.float64(spacing[-1]) * dx
        y = np.float64(spacing[-2]) * dy
        squared = x * x + y * y
        if len(spacing) == 3:
            z = np.float64(spacing[0]) * dz
            squared += z * z
        return np.float32(math.sqrt(squared))
    return spacing
