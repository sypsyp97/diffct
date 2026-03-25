"""TV and AwTV gradient helpers for iterative CT reconstruction."""

from __future__ import annotations

import mlx.core as mx


def _as_array(volume):
    """Convert arbitrary array-like inputs to MLX arrays."""
    return mx.array(volume)


def _tv_objective(volume, eta=1e-6):
    """Isotropic TV objective over the shared interior of an N-D array."""
    volume = _as_array(volume)
    if any(int(dim) < 2 for dim in volume.shape):
        return mx.array(0.0, dtype=volume.dtype)

    anchor = [slice(0, -1)] * volume.ndim
    diffs = []
    for axis in range(volume.ndim):
        shifted = list(anchor)
        shifted[axis] = slice(1, None)
        diffs.append(volume[tuple(shifted)] - volume[tuple(anchor)])

    sq_mag = diffs[0] * diffs[0]
    for diff in diffs[1:]:
        sq_mag = sq_mag + diff * diff
    return mx.mean(mx.sqrt(sq_mag + float(eta)))


def weight_d_volume(d_volume, delta):
    """Compute the AwTV diffusion weighting for a finite-difference field."""
    if delta <= 0.0:
        raise ValueError("delta must be positive.")
    d_volume = _as_array(d_volume)
    scaled = d_volume / float(delta)
    return mx.exp(-(scaled * scaled)) * d_volume


def _awtv_objective(volume, delta, eta=1e-6):
    """Adaptive-weighted TV objective over the shared interior of an N-D array."""
    if delta <= 0.0:
        raise ValueError("delta must be positive.")

    volume = _as_array(volume)
    if any(int(dim) < 2 for dim in volume.shape):
        return mx.array(0.0, dtype=volume.dtype)

    anchor = [slice(0, -1)] * volume.ndim
    diffs = []
    for axis in range(volume.ndim):
        shifted = list(anchor)
        shifted[axis] = slice(1, None)
        diffs.append(volume[tuple(shifted)] - volume[tuple(anchor)])

    weighted_diffs = [weight_d_volume(diff, delta) for diff in diffs]
    sq_mag = diffs[0] * weighted_diffs[0]
    for diff, weighted_diff in zip(diffs[1:], weighted_diffs[1:]):
        sq_mag = sq_mag + diff * weighted_diff

    return mx.mean(mx.sqrt(sq_mag + float(eta)))


def tv_gradient(volume, eta=1e-6):
    """Gradient of the isotropic TV objective."""
    volume = _as_array(volume)
    return mx.grad(lambda x: _tv_objective(x, eta=eta))(volume)


def awtv_gradient(volume, delta, eta=1e-6):
    """Gradient of the adaptive-weighted TV objective."""
    volume = _as_array(volume)
    return mx.grad(lambda x: _awtv_objective(x, delta=delta, eta=eta))(volume)
