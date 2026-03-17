"""Core helpers for geometry-agnostic iterative reconstruction algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import mlx.core as mx
import numpy as np


ArrayLike = Any
ForwardProjector = Callable[[mx.array, int], mx.array]
BackProjector = Callable[[mx.array, int], mx.array]
SARTDebugCallback = Callable[[dict[str, Any]], None]


@dataclass
class ReconstructionParameters:
    """Parameters shared by SART and the ART-TV family."""

    volume_shape: tuple[int, ...]
    iteration_count: int
    sart_iteration_count: int = 1
    pixel_extreme_values: tuple[float, float] = (0.0, float("inf"))
    voxel_extreme_values: tuple[float, float] = (-float("inf"), float("inf"))
    initial_volume: ArrayLike | None = None
    enforce_positivity: bool = True
    positivity_mode: str = "per_iteration"
    projection_order: Sequence[int] | None = None
    shuffle_projection_order: bool = True
    projection_order_seed: int = 0
    sart_debug_callback: SARTDebugCallback | None = None
    raylength_thresholding: bool = True
    raylength_quantile: float = 1e-3
    raylength_epsilon: float = 5e-4
    preserve_unmasked_computed_projection: bool = False
    detector_border_u: int = 0
    detector_border_v: int = 0
    backprojection_scale: float = 1.0
    dtype: Any = mx.float32


@dataclass
class RegularizationParameters:
    """Parameters shared by TV-based regularization steps."""

    reg_iteration_count: int = 20
    alpha: float = 0.2
    tv_eps: float = 1e-6


def as_mx_array(value: ArrayLike, dtype: Any = mx.float32) -> mx.array:
    """Convert a NumPy/MLX input to an MLX array."""
    return mx.array(value, dtype=dtype)


def normalize_measured_projections(
    measured_projections: Sequence[ArrayLike],
    dtype: Any,
) -> list[mx.array]:
    """Convert measured projections once up front."""
    return [as_mx_array(projection, dtype=dtype) for projection in measured_projections]


def validate_reconstruction_inputs(
    params: ReconstructionParameters,
    measured_projections: Sequence[ArrayLike],
) -> None:
    """Validate the generic reconstruction inputs."""
    if not measured_projections:
        raise ValueError("At least one measured projection is required.")
    if not params.volume_shape:
        raise ValueError("volume_shape must not be empty.")
    if any(int(dim) <= 0 for dim in params.volume_shape):
        raise ValueError(f"volume_shape must be positive in every axis, got {params.volume_shape!r}.")
    if params.iteration_count <= 0:
        raise ValueError("iteration_count must be positive.")
    if params.sart_iteration_count <= 0:
        raise ValueError("sart_iteration_count must be positive.")
    if params.positivity_mode not in {"per_iteration", "final", "none"}:
        raise ValueError("positivity_mode must be one of: 'per_iteration', 'final', 'none'.")
    if params.raylength_quantile < 0.0 or params.raylength_quantile >= 1.0:
        raise ValueError("raylength_quantile must be in [0, 1).")
    if params.raylength_epsilon <= 0.0:
        raise ValueError("raylength_epsilon must be positive.")
    if params.detector_border_u < 0 or params.detector_border_v < 0:
        raise ValueError("detector border widths must be non-negative.")
    if params.initial_volume is not None and tuple(np.shape(params.initial_volume)) != tuple(params.volume_shape):
        raise ValueError(
            "initial_volume shape does not match volume_shape: "
            f"{tuple(np.shape(params.initial_volume))!r} vs {tuple(params.volume_shape)!r}."
        )


def projection_order(params: ReconstructionParameters, num_projections: int) -> tuple[int, ...]:
    """Resolve and validate the per-iteration projection traversal order."""
    if params.projection_order is None:
        if not params.shuffle_projection_order:
            return tuple(range(num_projections))
        rng = np.random.default_rng(int(params.projection_order_seed))
        order = np.arange(num_projections, dtype=np.int32)
        rng.shuffle(order)
        return tuple(int(index) for index in order)

    order = tuple(int(index) for index in params.projection_order)
    if len(order) != num_projections:
        raise ValueError(
            "projection_order must contain one entry per measured projection: "
            f"expected {num_projections}, got {len(order)}."
        )
    invalid = [index for index in order if index < 0 or index >= num_projections]
    if invalid:
        raise ValueError(f"projection_order contains invalid indices: {invalid!r}.")
    return order


def initialize_volume(params: ReconstructionParameters) -> tuple[mx.array, mx.array, mx.array]:
    """Create the reconstruction, ones, and zero reference volumes."""
    if params.initial_volume is None:
        volume = mx.zeros(params.volume_shape, dtype=params.dtype)
    else:
        volume = as_mx_array(params.initial_volume, dtype=params.dtype)
    ones_volume = mx.ones(params.volume_shape, dtype=params.dtype)
    zero_volume = mx.zeros(params.volume_shape, dtype=params.dtype)
    return volume, ones_volume, zero_volume


def clamp_volume(
    volume: mx.array,
    voxel_extreme_values: tuple[float, float],
    *,
    enforce_positivity: bool,
) -> mx.array:
    """Apply configured bounds to a reconstruction volume."""
    lower, upper = voxel_extreme_values
    if enforce_positivity:
        lower = max(lower, 0.0)
    if np.isfinite(lower):
        volume = mx.maximum(volume, float(lower))
    if np.isfinite(upper):
        volume = mx.minimum(volume, float(upper))
    return volume


def should_enforce_positivity(
    params: ReconstructionParameters,
    *,
    stage: str,
) -> bool:
    """Resolve whether positivity should be enforced at the given stage."""
    if not params.enforce_positivity:
        return False
    if params.positivity_mode == "none":
        return False
    if params.positivity_mode == "final":
        return stage == "final"
    return stage in {"iteration", "regularization", "final"}


def clamp_reconstruction_volume(
    volume: mx.array,
    params: ReconstructionParameters,
    *,
    stage: str,
) -> mx.array:
    """Clamp a reconstruction volume according to the configured positivity policy."""
    return clamp_volume(
        volume,
        params.voxel_extreme_values,
        enforce_positivity=should_enforce_positivity(params, stage=stage),
    )


def print_progress(iteration: int, iteration_count: int) -> None:
    """Print a small textual progress bar."""
    iteration_number = iteration + 1
    finished = int(10.0 * iteration_number / iteration_count)
    remaining = max(0, 10 - finished)
    progress = ("█" * finished) + ("-" * remaining)
    print(
        f"\rProgress: |{progress}| Finished Iterations: {iteration_number} out of {iteration_count}",
        end="",
        flush=True,
    )
    if iteration_number == iteration_count:
        print()


def scalar_norm(value: mx.array) -> float:
    """Return an MLX norm as a Python float."""
    return float(np.asarray(mx.linalg.norm(value)))


def projection_residual_norm(
    volume: mx.array,
    measured_projection: mx.array,
    forward_project: ForwardProjector,
    projection_index: int,
) -> float:
    """Compute the residual norm for one measured projection."""
    computed_projection = forward_project(volume, projection_index)
    return scalar_norm(measured_projection - computed_projection)


def threshold_small_raylengths(
    raylength_projection: mx.array,
    quantile: float,
) -> mx.array:
    """Drop the smallest positive raylength values to improve SART stability."""
    if quantile <= 0.0:
        return raylength_projection
    raylength_np = np.asarray(raylength_projection)
    positive_values = raylength_np[raylength_np > 0.0]
    if positive_values.size == 0:
        return raylength_projection
    threshold = float(np.quantile(positive_values, quantile))
    mask = (raylength_projection > 0.0) & (raylength_projection <= threshold)
    return mx.where(mask, 0.0, raylength_projection)


def prepare_raylength_projection(
    raylength_projection: mx.array,
    params: ReconstructionParameters,
) -> mx.array:
    """Apply the configured raylength preprocessing once per projection."""
    if not params.raylength_thresholding:
        return raylength_projection
    return threshold_small_raylengths(raylength_projection, params.raylength_quantile)


def compute_sart_correction(
    measured_projection: mx.array,
    computed_projection: mx.array,
    raylength_projection: mx.array,
    params: ReconstructionParameters,
) -> mx.array:
    """Compute the SART correction image for one measured projection."""
    clipped_projection = prepare_measured_projection(measured_projection, computed_projection, params)
    clipped_projection = apply_detector_border_mask(clipped_projection, params, fill_value=0.0)
    epsilon = float(params.raylength_epsilon)
    mask = raylength_projection > epsilon
    numerator = clipped_projection - computed_projection
    denominator = mx.maximum(raylength_projection, epsilon)
    normalized_update = numerator / denominator
    if params.preserve_unmasked_computed_projection:
        positive_but_small = (raylength_projection > 0.0) & ~mask
        safe_background = mx.zeros_like(computed_projection)
        return mx.where(
            mask,
            normalized_update,
            mx.where(positive_but_small, computed_projection, safe_background),
        )
    normalized_update = mx.where(mask, normalized_update, 0.0)
    return apply_detector_border_mask(normalized_update, params, fill_value=0.0)


def prepare_measured_projection(
    measured_projection: mx.array,
    computed_projection: mx.array,
    params: ReconstructionParameters,
) -> mx.array:
    """Apply measurement clipping before the SART normalization step."""
    lower, upper = params.pixel_extreme_values
    clipped_projection = measured_projection
    if np.isfinite(upper):
        clipped_projection = mx.where(clipped_projection > float(upper), computed_projection, clipped_projection)
    if np.isfinite(lower):
        clipped_projection = mx.where(clipped_projection < float(lower), 0.0, clipped_projection)
    return clipped_projection


def apply_detector_border_mask(
    value: mx.array,
    params: ReconstructionParameters,
    *,
    fill_value: float = 0.0,
) -> mx.array:
    """Suppress detector-border pixels in 1D or 2D projection arrays."""
    border_u = int(params.detector_border_u)
    border_v = int(params.detector_border_v)
    if border_u <= 0 and border_v <= 0:
        return value

    value_np = np.array(np.asarray(value), copy=True)
    if value_np.ndim == 1:
        if border_u > 0:
            trim_u = min(border_u, value_np.shape[0])
            value_np[:trim_u] = fill_value
            value_np[-trim_u:] = fill_value
        return mx.array(value_np, dtype=value.dtype)

    if value_np.ndim != 2:
        return value

    if border_u > 0:
        trim_u = min(border_u, value_np.shape[0])
        value_np[:trim_u, :] = fill_value
        value_np[-trim_u:, :] = fill_value
    if border_v > 0:
        trim_v = min(border_v, value_np.shape[1])
        value_np[:, :trim_v] = fill_value
        value_np[:, -trim_v:] = fill_value
    return mx.array(value_np, dtype=value.dtype)


def _array_stats(name: str, value: mx.array) -> dict[str, float]:
    """Return compact numeric stats for one MLX array."""
    value_np = np.asarray(value)
    return {
        f"{name}_min": float(np.min(value_np)),
        f"{name}_max": float(np.max(value_np)),
        f"{name}_mean": float(np.mean(value_np)),
        f"{name}_norm": float(np.linalg.norm(value_np)),
    }


def _raylength_stats(raylength_projection: mx.array) -> dict[str, float]:
    """Return SART-relevant stats for the raylength projection."""
    raylength_np = np.asarray(raylength_projection)
    positive_values = raylength_np[raylength_np > 0.0]
    if positive_values.size == 0:
        return {
            "raylength_nonzero_count": 0.0,
            "raylength_positive_min": 0.0,
            "raylength_positive_p001": 0.0,
            "raylength_positive_mean": 0.0,
            "raylength_positive_max": 0.0,
        }
    return {
        "raylength_nonzero_count": float(positive_values.size),
        "raylength_positive_min": float(np.min(positive_values)),
        "raylength_positive_p001": float(np.quantile(positive_values, 1e-3)),
        "raylength_positive_mean": float(np.mean(positive_values)),
        "raylength_positive_max": float(np.max(positive_values)),
    }


def run_sart_sweeps(
    volume: mx.array,
    measured_projections: Sequence[mx.array],
    ones_volume: mx.array,
    forward_project: ForwardProjector,
    back_project: BackProjector,
    params: ReconstructionParameters,
    *,
    beta: float = 1.0,
    outer_iteration_index: int = 0,
) -> mx.array:
    """Run the configured number of SART sweeps over all provided projections."""
    order = projection_order(params, len(measured_projections))
    for sweep_index in range(params.sart_iteration_count):
        for projection_index in order:
            measured_projection = measured_projections[projection_index]
            computed_projection = forward_project(volume, projection_index)
            raylength_projection = prepare_raylength_projection(
                forward_project(ones_volume, projection_index),
                params,
            )
            correction_image = compute_sart_correction(
                measured_projection=measured_projection,
                computed_projection=computed_projection,
                raylength_projection=raylength_projection,
                params=params,
            )
            backprojection_volume = back_project(correction_image, projection_index)
            volume = volume + (float(beta) * float(params.backprojection_scale) * backprojection_volume)
            volume = clamp_reconstruction_volume(volume, params, stage="sart_update")
            if params.sart_debug_callback is not None:
                debug_stats = {
                    "outer_iteration": float(outer_iteration_index),
                    "sart_sweep": float(sweep_index),
                    "projection_index": float(projection_index),
                    "beta": float(beta),
                    "backprojection_scale": float(params.backprojection_scale),
                    **_array_stats("measured", measured_projection),
                    **_array_stats("computed", computed_projection),
                    **_raylength_stats(raylength_projection),
                    **_array_stats("correction", correction_image),
                    **_array_stats("backprojection", backprojection_volume),
                    **_array_stats("volume", volume),
                }
                params.sart_debug_callback(debug_stats)
    return volume
