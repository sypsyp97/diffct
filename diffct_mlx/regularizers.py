"""Regularizers and POCS regularization steps for iterative CT reconstruction."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .tv_gradients import awtv_gradient
from .tv_gradients import _awtv_objective as awtv_objective
from .tv_gradients import _tv_objective as tv_objective
from .tv_gradients import tv_gradient


def _as_array(volume):
    """Convert arbitrary array-like inputs to MLX arrays."""
    return mx.array(volume)


def _scalar_norm(value) -> float:
    """Return an MLX norm as a Python float."""
    return float(np.asarray(mx.linalg.norm(value)))


def l2_regularizer(volume):
    """Mean squared voxel magnitude."""
    volume = _as_array(volume)
    return mx.mean(mx.square(volume))


def normalize_reconstruction_volume(volume, eps=1e-6):
    """Project a nonnegative reconstruction volume to a stable max-one scale."""
    volume = mx.maximum(_as_array(volume), 0.0)
    scale = mx.maximum(mx.max(volume), float(eps))
    return volume / scale


def tv_regularizer(volume, eps=1e-6):
    """Isotropic total variation over the shared interior of an N-D volume."""
    return tv_objective(volume, eta=eps)


def tv_regularizer_3d(volume, eps=1e-6):
    """Backward-compatible alias for isotropic TV regularization."""
    return tv_regularizer(volume, eps=eps)


def awtv_regularizer(volume, delta, eps=1e-6):
    """Adaptive-weighted TV regularizer over the shared interior of an N-D volume."""
    return awtv_objective(volume, delta=delta, eta=eps)


def tv_pocs(volume, backup_volume, reg_iteration_count, alpha, eta=1e-6):
    """Regularization step of TV-POCS."""
    volume = _as_array(volume)
    backup_volume = _as_array(backup_volume)
    dp = _scalar_norm(volume - backup_volume)

    for _ in range(int(reg_iteration_count)):
        gradient_volume = tv_gradient(volume, eta=eta)
        norm = _scalar_norm(gradient_volume)
        if norm > 0.0:
            gradient_volume = gradient_volume / norm
        volume = volume - float(alpha) * dp * gradient_volume

    backup_volume = volume
    return volume, backup_volume


def asd_pocs(
    volume,
    backup_volume,
    iteration_index,
    reg_iteration_count,
    dd,
    dtvg,
    alpha,
    alpha_red,
    r_max,
    epsilon,
    eta=1e-6,
):
    """Regularization step of ASD-POCS."""
    volume = _as_array(volume)
    backup_volume = _as_array(backup_volume)

    dp = _scalar_norm(volume - backup_volume)
    if int(iteration_index) == 0:
        dtvg = float(alpha) * dp

    backup_volume = volume

    for _ in range(int(reg_iteration_count)):
        gradient_volume = tv_gradient(volume, eta=eta)
        norm = _scalar_norm(gradient_volume)
        if norm > 0.0:
            gradient_volume = gradient_volume / norm
        volume = volume - float(dtvg) * gradient_volume

    dg = _scalar_norm(volume - backup_volume)
    if dg > float(r_max) * dp and float(dd) > float(epsilon):
        dtvg = float(dtvg) * float(alpha_red)

    backup_volume = volume
    return volume, backup_volume, float(dtvg)


def awtv_pocs(volume, backup_volume, reg_iteration_count, alpha, delta, eta=1e-6):
    """Regularization step of AwTV-POCS."""
    volume = _as_array(volume)
    backup_volume = _as_array(backup_volume)
    dp = _scalar_norm(volume - backup_volume)

    for _ in range(int(reg_iteration_count)):
        gradient_volume = awtv_gradient(volume, delta=delta, eta=eta)
        norm = _scalar_norm(gradient_volume)
        if norm > 0.0:
            gradient_volume = gradient_volume / norm
        volume = volume - dp * float(alpha) * gradient_volume

    backup_volume = volume
    alpha = float(alpha) * 0.995
    return volume, backup_volume, alpha
