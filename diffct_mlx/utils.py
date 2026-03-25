"""Utility functions for DiffCT-MLX package.

This module provides helper functions for grid computation
and memory layout validation for the MLX-based CT reconstruction.
"""

import math
from .constants import _TG_2D, _TG_3D


def _grid_2d(n1, n2):
    """Compute Metal grid and threadgroup sizes for 2D kernels.

    Parameters
    ----------
    n1 : int
        First dimension size (e.g., number of views/angles).
    n2 : int
        Second dimension size (e.g., number of detectors).

    Returns
    -------
    grid : tuple of int
        Metal grid dimensions (total threads in each dimension).
    threadgroup : tuple of int
        Metal threadgroup dimensions.
    """
    tg = _TG_2D
    grid = (
        math.ceil(n1 / tg[0]) * tg[0],
        math.ceil(n2 / tg[1]) * tg[1],
        1,
    )
    threadgroup = (tg[0], tg[1], 1)
    return grid, threadgroup


def _grid_3d(n1, n2, n3):
    """Compute Metal grid and threadgroup sizes for 3D kernels.

    Parameters
    ----------
    n1 : int
        First dimension size (e.g., number of views).
    n2 : int
        Second dimension size (e.g., number of u-detectors).
    n3 : int
        Third dimension size (e.g., number of v-detectors).

    Returns
    -------
    grid : tuple of int
        Metal grid dimensions (total threads in each dimension).
    threadgroup : tuple of int
        Metal threadgroup dimensions.
    """
    tg = _TG_3D
    grid = (
        math.ceil(n1 / tg[0]) * tg[0],
        math.ceil(n2 / tg[1]) * tg[1],
        math.ceil(n3 / tg[2]) * tg[2],
    )
    threadgroup = (tg[0], tg[1], tg[2])
    return grid, threadgroup
