"""Regularizers for differentiable CT reconstruction."""

import mlx.core as mx


def l2_regularizer(volume):
    """Mean squared voxel magnitude."""
    volume = mx.array(volume)
    return mx.mean(mx.square(volume))


def normalize_reconstruction_volume(volume, eps=1e-6):
    """Project a nonnegative reconstruction volume to a stable max-one scale."""
    volume = mx.maximum(mx.array(volume), 0.0)
    scale = mx.maximum(mx.max(volume), float(eps))
    return volume / scale


def tv_regularizer_3d(volume, eps=1e-6):
    """Isotropic 3D total variation over forward differences."""
    volume = mx.array(volume)
    dx = volume[1:, :-1, :-1] - volume[:-1, :-1, :-1]
    dy = volume[:-1, 1:, :-1] - volume[:-1, :-1, :-1]
    dz = volume[:-1, :-1, 1:] - volume[:-1, :-1, :-1]
    return mx.mean(mx.sqrt(dx * dx + dy * dy + dz * dz + float(eps)))
