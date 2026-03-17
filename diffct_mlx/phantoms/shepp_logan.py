"""Shepp-Logan phantom generators for 2D and 3D examples."""

from __future__ import annotations

import numpy as np


def _shape_2d(shape_or_nx, ny: int | None = None) -> tuple[int, int]:
    """Accept `(ny, nx)` or `(nx, ny)` style input used in the examples."""
    if ny is None:
        if not isinstance(shape_or_nx, (tuple, list)) or len(shape_or_nx) != 2:
            raise ValueError("Expected shape `(ny, nx)` or arguments `(nx, ny)`.")
        ny, nx = shape_or_nx
    else:
        nx = shape_or_nx
    ny = int(ny)
    nx = int(nx)
    if ny <= 0 or nx <= 0:
        raise ValueError(f"Shape must be positive, got {(ny, nx)!r}.")
    return ny, nx


def _shape_3d(shape_or_nz, ny: int | None = None, nx: int | None = None) -> tuple[int, int, int]:
    """Accept `(nz, ny, nx)` or separate `nz, ny, nx` arguments."""
    if ny is None or nx is None:
        if not isinstance(shape_or_nz, (tuple, list)) or len(shape_or_nz) != 3:
            raise ValueError("Expected shape `(nz, ny, nx)` or arguments `(nz, ny, nx)`.")
        nz, ny, nx = shape_or_nz
    else:
        nz = shape_or_nz
    nz = int(nz)
    ny = int(ny)
    nx = int(nx)
    if nz <= 0 or ny <= 0 or nx <= 0:
        raise ValueError(f"Shape must be positive, got {(nz, ny, nx)!r}.")
    return nz, ny, nx


def shepp_logan_2d(shape_or_nx, ny: int | None = None) -> np.ndarray:
    """Generate a 2D Shepp-Logan phantom as a NumPy array."""
    ny, nx = _shape_2d(shape_or_nx, ny)
    yy, xx = np.mgrid[:ny, :nx]
    xx = (xx - (nx - 1) / 2) / ((nx - 1) / 2)
    yy = (yy - (ny - 1) / 2) / ((ny - 1) / 2)

    ellipses = np.array(
        [
            [0, 0, 0.69, 0.92, 0, 1],
            [0, -0.0184, 0.6624, 0.874, 0, -0.8],
            [0.22, 0, 0.11, 0.31, -np.pi / 10, -0.2],
            [-0.22, 0, 0.16, 0.41, np.pi / 10, -0.2],
            [0, 0.35, 0.21, 0.25, 0, 0.1],
            [0, 0.1, 0.046, 0.046, 0, 0.1],
            [0, -0.1, 0.046, 0.046, 0, 0.1],
            [-0.08, -0.605, 0.046, 0.023, 0, 0.1],
            [0, -0.605, 0.023, 0.023, 0, 0.1],
            [0.06, -0.605, 0.023, 0.046, 0, 0.1],
        ],
        dtype=np.float32,
    )

    x0 = ellipses[:, 0][:, None, None]
    y0 = ellipses[:, 1][:, None, None]
    a = ellipses[:, 2][:, None, None]
    b = ellipses[:, 3][:, None, None]
    phi = ellipses[:, 4][:, None, None]
    val = ellipses[:, 5][:, None, None]

    c, s = np.cos(phi), np.sin(phi)
    xc, yc = xx[None] - x0, yy[None] - y0
    xp = c * xc - s * yc
    yp = s * xc + c * yc
    mask = (xp**2 / a**2 + yp**2 / b**2) <= 1.0
    return np.clip(np.sum(mask * val, axis=0), 0.0, 1.0).astype(np.float32)


def shepp_logan_3d(shape_or_nz, ny: int | None = None, nx: int | None = None) -> np.ndarray:
    """Generate a 3D Shepp-Logan phantom as a NumPy array."""
    nz, ny, nx = _shape_3d(shape_or_nz, ny, nx)
    zz, yy, xx = np.mgrid[:nz, :ny, :nx]
    xx = (xx - (nx - 1) / 2) / ((nx - 1) / 2)
    yy = (yy - (ny - 1) / 2) / ((ny - 1) / 2)
    zz = (zz - (nz - 1) / 2) / ((nz - 1) / 2)

    ellipsoids = np.array(
        [
            [0, 0, 0, 0.69, 0.92, 0.81, 0, 1],
            [0, -0.0184, 0, 0.6624, 0.874, 0.78, 0, -0.8],
            [0.22, 0, 0, 0.11, 0.31, 0.22, -np.pi / 10, -0.2],
            [-0.22, 0, 0, 0.16, 0.41, 0.28, np.pi / 10, -0.2],
            [0, 0.35, -0.15, 0.21, 0.25, 0.41, 0, 0.1],
            [0, 0.1, 0.25, 0.046, 0.046, 0.05, 0, 0.1],
            [0, -0.1, 0.25, 0.046, 0.046, 0.05, 0, 0.1],
            [-0.08, -0.605, 0, 0.046, 0.023, 0.05, 0, 0.1],
            [0, -0.605, 0, 0.023, 0.023, 0.02, 0, 0.1],
            [0.06, -0.605, 0, 0.023, 0.046, 0.02, 0, 0.1],
        ],
        dtype=np.float32,
    )

    x0 = ellipsoids[:, 0][:, None, None, None]
    y0 = ellipsoids[:, 1][:, None, None, None]
    z0 = ellipsoids[:, 2][:, None, None, None]
    a = ellipsoids[:, 3][:, None, None, None]
    b = ellipsoids[:, 4][:, None, None, None]
    c = ellipsoids[:, 5][:, None, None, None]
    phi = ellipsoids[:, 6][:, None, None, None]
    val = ellipsoids[:, 7][:, None, None, None]

    cos_p, sin_p = np.cos(phi), np.sin(phi)
    xc = xx[None] - x0
    yc = yy[None] - y0
    zc = zz[None] - z0
    xp = cos_p * xc - sin_p * yc
    yp = sin_p * xc + cos_p * yc

    mask = (xp**2 / a**2 + yp**2 / b**2 + zc**2 / c**2) <= 1.0
    return np.clip(np.sum(mask * val, axis=0), 0.0, 1.0).astype(np.float32)
