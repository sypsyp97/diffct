"""Shared phantom builders for accuracy tests."""

import math

import numpy as np


def shepp_logan_2d(N):
    """2D Shepp-Logan clipped to [0, 1] on an NxN grid using the kernel's
    grid-point convention ``cx = N * 0.5``."""
    phantom = np.zeros((N, N), dtype=np.float32)
    ellipses = [
        (0.0, 0.0, 0.69, 0.92, 0.0, 1.0),
        (0.0, -0.0184, 0.6624, 0.8740, 0.0, -0.8),
        (0.22, 0.0, 0.11, 0.31, -18.0, -0.8),
        (-0.22, 0.0, 0.16, 0.41, 18.0, -0.8),
        (0.0, 0.35, 0.21, 0.25, 0.0, 0.7),
    ]
    cx = N * 0.5
    ys, xs = np.mgrid[0:N, 0:N].astype(np.float32)
    xn = (xs - cx) / (N / 2)
    yn = (ys - cx) / (N / 2)
    for (x0, y0, a, b, angdeg, ampl) in ellipses:
        th = math.radians(angdeg)
        xp = (xn - x0) * math.cos(th) + (yn - y0) * math.sin(th)
        yp = -(xn - x0) * math.sin(th) + (yn - y0) * math.cos(th)
        mask = (xp * xp) / (a * a) + (yp * yp) / (b * b) <= 1.0
        phantom[mask] += ampl
    return np.clip(phantom, 0.0, 1.0)


def shepp_logan_3d(N):
    """3D Shepp-Logan clipped to [0, 1] on an NxNxN grid using the kernel's
    grid-point convention ``cx = cy = cz = N * 0.5``."""
    el_params = np.array(
        [
            [0, 0, 0, 0.69, 0.92, 0.81, 0, 0, 0, 1.0],
            [0, -0.0184, 0, 0.6624, 0.874, 0.78, 0, 0, 0, -0.8],
            [0.22, 0, 0, 0.11, 0.31, 0.22, -np.pi / 10.0, 0, 0, -0.2],
            [-0.22, 0, 0, 0.16, 0.41, 0.28, np.pi / 10.0, 0, 0, -0.2],
            [0, 0.35, -0.15, 0.21, 0.25, 0.41, 0, 0, 0, 0.1],
            [0, 0.10, 0.25, 0.046, 0.046, 0.05, 0, 0, 0, 0.1],
            [0, -0.10, 0.25, 0.046, 0.046, 0.05, 0, 0, 0, 0.1],
            [-0.08, -0.605, 0, 0.046, 0.023, 0.05, 0, 0, 0, 0.1],
            [0, -0.605, 0, 0.023, 0.023, 0.02, 0, 0, 0, 0.1],
            [0.06, -0.605, 0, 0.023, 0.046, 0.02, 0, 0, 0, 0.1],
        ],
        dtype=np.float32,
    )
    zz, yy, xx = np.mgrid[:N, :N, :N].astype(np.float32)
    xx = (xx - N * 0.5) / (N * 0.5)
    yy = (yy - N * 0.5) / (N * 0.5)
    zz = (zz - N * 0.5) / (N * 0.5)
    x_pos = el_params[:, 0][:, None, None, None]
    y_pos = el_params[:, 1][:, None, None, None]
    z_pos = el_params[:, 2][:, None, None, None]
    a_axis = el_params[:, 3][:, None, None, None]
    b_axis = el_params[:, 4][:, None, None, None]
    c_axis = el_params[:, 5][:, None, None, None]
    phi = el_params[:, 6][:, None, None, None]
    val = el_params[:, 9][:, None, None, None]
    xc = xx[None, ...] - x_pos
    yc = yy[None, ...] - y_pos
    zc = zz[None, ...] - z_pos
    c = np.cos(phi)
    s = np.sin(phi)
    xp = c * xc - s * yc
    yp = s * xc + c * yc
    zp = zc
    mask = (
        (xp ** 2) / (a_axis ** 2)
        + (yp ** 2) / (b_axis ** 2)
        + (zp ** 2) / (c_axis ** 2)
        <= 1.0
    )
    phantom = np.sum(mask * val, axis=0)
    return np.clip(phantom, 0.0, 1.0).astype(np.float32)
