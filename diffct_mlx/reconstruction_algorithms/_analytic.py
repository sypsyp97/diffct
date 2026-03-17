"""Shared helpers for analytical reconstruction algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import math
import mlx.core as mx
import numpy as np


AnalyticalWeighting = Callable[[mx.array], mx.array]
AnalyticalFilter = Callable[[mx.array], mx.array]
AnalyticalBackProjector = Callable[[mx.array], mx.array]


@dataclass
class AnalyticalReconstructionParameters:
    """Common parameters for analytical reconstruction algorithms."""

    detector_spacing: float = 1.0
    voxel_spacing: float = 1.0
    enforce_positivity: bool = True
    normalization_scale: float | None = None
    dtype: Any = mx.float32


def _as_array(value, dtype=None):
    """Convert arbitrary array-like inputs to an MLX array."""
    if dtype is None:
        return mx.array(value)
    return mx.array(value, dtype=dtype)


def _positive_and_scale(reconstruction, *, enforce_positivity, normalization_scale):
    """Apply optional positivity and multiplicative normalization."""
    reconstruction = mx.array(reconstruction)
    if enforce_positivity:
        reconstruction = mx.maximum(reconstruction, 0.0)
    if normalization_scale is not None:
        reconstruction = reconstruction * float(normalization_scale)
    return reconstruction


def default_angular_normalization(num_views: int) -> float:
    """Standard circular-scan angular normalization used in the examples."""
    if int(num_views) <= 0:
        raise ValueError("num_views must be positive.")
    return math.pi / float(num_views)


def ramp_filter_2d(sinogram):
    """Apply a ramp filter along the detector axis of a 2D sinogram."""
    sino_np = np.asarray(sinogram, dtype=np.float32)
    if sino_np.ndim != 2:
        raise ValueError(f"Expected a 2D sinogram, got shape {sino_np.shape!r}.")

    num_detectors = sino_np.shape[1]
    freqs = np.fft.fftfreq(num_detectors)
    # Use the standard discrete ramp for FFT frequencies in cycles / sample.
    ramp = (2.0 * np.abs(freqs)).astype(np.float32)
    sino_fft = np.fft.fft(sino_np, axis=1)
    filtered = np.real(np.fft.ifft(sino_fft * ramp[None, :], axis=1)).astype(np.float32)
    return mx.array(filtered)


def ramp_filter_3d(sinogram):
    """Apply a ramp filter along the detector-u axis of a 3D cone sinogram."""
    sino_np = np.asarray(sinogram, dtype=np.float32)
    if sino_np.ndim != 3:
        raise ValueError(f"Expected a 3D sinogram, got shape {sino_np.shape!r}.")

    num_detectors_u = sino_np.shape[1]
    freqs = np.fft.fftfreq(num_detectors_u)
    ramp = (2.0 * np.abs(freqs)).astype(np.float32).reshape(1, num_detectors_u, 1)
    sino_fft = np.fft.fft(sino_np, axis=1)
    filtered = np.real(np.fft.ifft(sino_fft * ramp, axis=1)).astype(np.float32)
    return mx.array(filtered)


def ramp_filter(sinogram, axis=1):
    """Apply a ramp filter along an arbitrary detector axis."""
    sino_np = np.asarray(sinogram, dtype=np.float32)
    if sino_np.ndim < 2:
        raise ValueError(f"Expected sinogram with at least 2 dimensions, got shape {sino_np.shape!r}.")

    axis = int(axis)
    if axis < 0:
        axis += sino_np.ndim
    if axis < 0 or axis >= sino_np.ndim:
        raise ValueError(f"Invalid filter axis {axis} for sinogram with ndim={sino_np.ndim}.")

    detector_count = sino_np.shape[axis]
    freqs = np.fft.fftfreq(detector_count)
    ramp = (2.0 * np.abs(freqs)).astype(np.float32)
    shape = [1] * sino_np.ndim
    shape[axis] = detector_count
    sino_fft = np.fft.fft(sino_np, axis=axis)
    filtered = np.real(np.fft.ifft(sino_fft * ramp.reshape(shape), axis=axis)).astype(np.float32)
    return mx.array(filtered)
