"""Discrete algebraic reconstruction technique (DART) for 2D and 3D problems."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from itertools import product
from typing import Sequence

import mlx.core as mx
import numpy as np

from ._core import (
    ArrayLike,
    BackProjector,
    ForwardProjector,
    ReconstructionParameters,
    clamp_reconstruction_volume,
    forward_project_views,
    initialize_volume,
    normalize_measured_projections,
    print_progress,
    run_iterative_sweeps,
    validate_reconstruction_inputs,
)


@dataclass
class DARTParameters(ReconstructionParameters):
    """Configuration for discrete-valued DART reconstruction."""

    gray_levels: Sequence[float] = (0.0, 1.0)
    free_pixel_probability: float = 0.15
    initial_reconstruction_sweeps: int | None = None
    apply_smoothing: bool = True
    smoothing_beta: float = 0.2
    convergence_epsilon: float | None = None
    boundary_connectivity: str = "axial"
    segmentation_threshold_method: str = "midpoint"
    segmentation_threshold_value: float | None = None
    otsu_foreground_only: bool = False
    otsu_percentile_window: tuple[float, float] | None = None
    binary_fill_holes: bool = False
    return_segmented_volume: bool = False
    random_seed: int = 0


def _validate_dart_inputs(
    params: DARTParameters,
    measured_projections: Sequence[ArrayLike],
) -> np.ndarray:
    """Validate DART-specific inputs and return the gray levels as NumPy."""
    validate_reconstruction_inputs(params, measured_projections)
    if len(params.volume_shape) not in {2, 3}:
        raise ValueError(
            "DART currently supports only 2D and 3D reconstruction volumes, "
            f"got volume_shape={params.volume_shape!r}."
        )
    gray_levels = np.asarray(params.gray_levels, dtype=np.float32)
    if gray_levels.ndim != 1 or gray_levels.size < 2:
        raise ValueError("gray_levels must contain at least two discrete values.")
    if not np.all(np.isfinite(gray_levels)):
        raise ValueError("gray_levels must be finite.")
    if np.any(np.diff(gray_levels) < 0.0):
        raise ValueError("gray_levels must be sorted in ascending order.")
    if not 0.0 <= float(params.free_pixel_probability) <= 1.0:
        raise ValueError("free_pixel_probability must lie in [0, 1].")
    if params.initial_reconstruction_sweeps is not None and params.initial_reconstruction_sweeps <= 0:
        raise ValueError("initial_reconstruction_sweeps must be positive when specified.")
    if params.iterative_update_method != "sart":
        raise ValueError("DART currently requires iterative_update_method='sart'.")
    if not 0.0 <= float(params.smoothing_beta) <= 1.0:
        raise ValueError("smoothing_beta must lie in [0, 1].")
    if params.convergence_epsilon is not None and not 0.0 <= float(params.convergence_epsilon) <= 1.0:
        raise ValueError("convergence_epsilon must lie in [0, 1] when specified.")
    if params.boundary_connectivity not in {"axial", "full"}:
        raise ValueError("boundary_connectivity must be either 'axial' or 'full'.")
    if params.segmentation_threshold_method not in {"midpoint", "otsu"}:
        raise ValueError("segmentation_threshold_method must be 'midpoint' or 'otsu'.")
    if params.segmentation_threshold_value is not None and gray_levels.size != 2:
        raise ValueError("A manual segmentation threshold currently supports binary DART only.")
    if params.segmentation_threshold_method == "otsu" and gray_levels.size != 2:
        raise ValueError("Otsu thresholding currently supports binary DART only.")
    if params.binary_fill_holes and gray_levels.size != 2:
        raise ValueError("binary_fill_holes currently supports binary DART only.")
    if params.otsu_percentile_window is not None:
        lower_percentile, upper_percentile = params.otsu_percentile_window
        if lower_percentile < 0.0 or upper_percentile > 100.0 or lower_percentile >= upper_percentile:
            raise ValueError("otsu_percentile_window must satisfy 0 <= low < high <= 100.")
    return gray_levels


def _otsu_threshold(
    volume: np.ndarray,
    *,
    foreground_only: bool = False,
    percentile_window: tuple[float, float] | None = None,
) -> float:
    """Compute an Otsu threshold from a NumPy array."""
    values = np.asarray(volume, dtype=np.float32)
    values = values[np.isfinite(values)]
    if foreground_only:
        positive_values = values[values > 0.0]
        if positive_values.size > 0:
            values = positive_values
    if percentile_window is not None and values.size > 0:
        lower_percentile, upper_percentile = percentile_window
        lower_value, upper_value = np.percentile(values, [lower_percentile, upper_percentile])
        values = values[(values >= lower_value) & (values <= upper_value)]
    if values.size == 0:
        return 0.0
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    if value_max <= value_min:
        return value_min
    histogram, bin_edges = np.histogram(values, bins=256, range=(value_min, value_max))
    histogram = histogram.astype(np.float64)
    total = float(histogram.sum())
    if total <= 0.0:
        return 0.5 * (value_min + value_max)
    probabilities = histogram / total
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    cumulative_probability = np.cumsum(probabilities)
    cumulative_mean = np.cumsum(probabilities * bin_centers)
    global_mean = float(cumulative_mean[-1])
    denominator = cumulative_probability * (1.0 - cumulative_probability)
    numerator = (global_mean * cumulative_probability - cumulative_mean) ** 2
    between_class_variance = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    return float(bin_centers[int(np.argmax(between_class_variance))])


def _segment_volume(
    volume: np.ndarray,
    gray_levels: np.ndarray,
    *,
    threshold_method: str = "midpoint",
    threshold_value: float | None = None,
    otsu_foreground_only: bool = False,
    otsu_percentile_window: tuple[float, float] | None = None,
) -> np.ndarray:
    """Project a continuous-valued image onto the configured gray levels."""
    if threshold_value is not None:
        thresholds = np.array([float(threshold_value)], dtype=np.float32)
    elif threshold_method == "otsu":
        threshold = _otsu_threshold(
            volume,
            foreground_only=otsu_foreground_only,
            percentile_window=otsu_percentile_window,
        )
        thresholds = np.array([threshold], dtype=np.float32)
    else:
        thresholds = 0.5 * (gray_levels[:-1] + gray_levels[1:])
    segment_indices = np.digitize(volume, thresholds, right=False)
    return gray_levels[segment_indices]


def _neighbor_offsets(
    ndim: int,
    *,
    connectivity: str,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate neighbor offsets for the requested DART connectivity."""
    if connectivity == "axial":
        offsets: list[tuple[int, ...]] = []
        for axis in range(ndim):
            negative = [0] * ndim
            positive = [0] * ndim
            negative[axis] = -1
            positive[axis] = 1
            offsets.append(tuple(negative))
            offsets.append(tuple(positive))
        return tuple(offsets)
    if connectivity == "full":
        return tuple(
            offset
            for offset in product((-1, 0, 1), repeat=ndim)
            if any(component != 0 for component in offset)
        )
    raise ValueError(f"Unsupported connectivity {connectivity!r}.")


def _boundary_pixels(
    segmented: np.ndarray,
    *,
    connectivity: str = "axial",
) -> np.ndarray:
    """Return a boolean mask marking cells that touch another gray level."""
    padded = np.pad(segmented, 1, mode="edge")
    boundary = np.zeros(segmented.shape, dtype=bool)
    spatial_shape = segmented.shape
    for offset in _neighbor_offsets(segmented.ndim, connectivity=connectivity):
        slices = tuple(
            slice(1 + offset[axis], 1 + offset[axis] + spatial_shape[axis])
            for axis in range(segmented.ndim)
        )
        neighbor_view = padded[slices]
        boundary |= neighbor_view != segmented
    return boundary


def _sample_free_pixels(
    shape: tuple[int, ...],
    free_pixel_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly sample additional free pixels."""
    return rng.random(shape) < float(free_pixel_probability)


def _smooth_boundary_voxels(
    image: np.ndarray,
    boundary_mask: np.ndarray,
    *,
    beta: float,
    connectivity: str = "axial",
) -> np.ndarray:
    """Blend boundary voxels toward the mean of their local neighborhood."""
    image = np.asarray(image, dtype=np.float32)
    boundary_mask = np.asarray(boundary_mask, dtype=bool)
    if beta <= 0.0 or not np.any(boundary_mask):
        return np.array(image, copy=True)

    offsets = _neighbor_offsets(image.ndim, connectivity=connectivity)
    padded = np.pad(image, 1, mode="edge")
    neighbor_sum = np.zeros_like(image, dtype=np.float32)
    spatial_shape = image.shape
    for offset in offsets:
        slices = tuple(
            slice(1 + offset[axis], 1 + offset[axis] + spatial_shape[axis])
            for axis in range(image.ndim)
        )
        neighbor_sum += padded[slices]
    neighbor_mean = neighbor_sum / float(len(offsets))

    smoothed = np.array(image, copy=True)
    smoothed[boundary_mask] = (
        (1.0 - float(beta)) * image[boundary_mask]
        + float(beta) * neighbor_mean[boundary_mask]
    )
    return smoothed


def _fill_binary_holes(segmented: np.ndarray, gray_levels: np.ndarray) -> np.ndarray:
    """Fill enclosed air pockets inside a binary object mask."""
    background_mask = segmented == float(gray_levels[0])
    if not np.any(background_mask):
        return segmented

    visited = np.zeros_like(background_mask, dtype=bool)
    queue: deque[tuple[int, ...]] = deque()

    for axis in range(background_mask.ndim):
        for boundary_index in (0, background_mask.shape[axis] - 1):
            boundary_slice = [slice(None)] * background_mask.ndim
            boundary_slice[axis] = boundary_index
            boundary_coords = np.argwhere(background_mask[tuple(boundary_slice)])
            for coord in boundary_coords:
                full_coord = list(coord)
                full_coord.insert(axis, boundary_index)
                full_coord_tuple = tuple(full_coord)
                if not visited[full_coord_tuple]:
                    visited[full_coord_tuple] = True
                    queue.append(full_coord_tuple)

    neighbor_offsets = []
    for axis in range(background_mask.ndim):
        negative = [0] * background_mask.ndim
        positive = [0] * background_mask.ndim
        negative[axis] = -1
        positive[axis] = 1
        neighbor_offsets.append(tuple(negative))
        neighbor_offsets.append(tuple(positive))

    while queue:
        current = queue.popleft()
        for offset in neighbor_offsets:
            neighbor = tuple(current[axis] + offset[axis] for axis in range(background_mask.ndim))
            if any(index < 0 or index >= background_mask.shape[axis] for axis, index in enumerate(neighbor)):
                continue
            if visited[neighbor] or not background_mask[neighbor]:
                continue
            visited[neighbor] = True
            queue.append(neighbor)

    hole_mask = background_mask & ~visited
    if not np.any(hole_mask):
        return segmented
    filled = np.array(segmented, copy=True)
    filled[hole_mask] = float(gray_levels[-1])
    return filled


def _segment_dart_volume(
    volume: np.ndarray,
    gray_levels: np.ndarray,
    params: DARTParameters,
) -> np.ndarray:
    """Segment a DART iterate and apply optional binary hole filling immediately."""
    segmented = _segment_volume(
        volume,
        gray_levels,
        threshold_method=params.segmentation_threshold_method,
        threshold_value=params.segmentation_threshold_value,
        otsu_foreground_only=params.otsu_foreground_only,
        otsu_percentile_window=params.otsu_percentile_window,
    )
    if params.binary_fill_holes:
        segmented = _fill_binary_holes(segmented, gray_levels)
    return segmented


def reconstruct_dart(
    measured_projections: Sequence[ArrayLike],
    forward_project: ForwardProjector,
    back_project: BackProjector,
    params: DARTParameters,
    *,
    show_progress: bool = True,
) -> mx.array:
    """Reconstruct a 2D or 3D discrete-valued image/volume with DART."""
    gray_levels = _validate_dart_inputs(params, measured_projections)
    measured = normalize_measured_projections(measured_projections, params.dtype)
    volume, ones_volume, _ = initialize_volume(params)

    initial_sweeps = params.initial_reconstruction_sweeps
    if initial_sweeps is None:
        initial_sweeps = params.sart_iteration_count
    initial_params = replace(params, sart_iteration_count=initial_sweeps)
    volume = run_iterative_sweeps(
        volume=volume,
        measured_projections=measured,
        ones_volume=ones_volume,
        forward_project=forward_project,
        back_project=back_project,
        params=initial_params,
        outer_iteration_index=-1,
    )
    volume = clamp_reconstruction_volume(volume, params, stage="iteration")

    rng = np.random.default_rng(int(params.random_seed))
    for iteration in range(params.iteration_count):
        volume_np = np.asarray(volume)
        segmented = _segment_dart_volume(volume_np, gray_levels, params)
        boundary_mask = _boundary_pixels(
            segmented,
            connectivity=params.boundary_connectivity,
        )
        free_pixel_mask = boundary_mask | _sample_free_pixels(
            params.volume_shape,
            params.free_pixel_probability,
            rng,
        )

        # DART restarts each masked subproblem from the discrete segmentation.
        updated_volume_np = np.array(segmented, copy=True)
        updated_volume = mx.array(updated_volume_np, dtype=params.dtype)

        fixed_volume_np = np.array(updated_volume_np, copy=True)
        fixed_volume_np[free_pixel_mask] = 0.0
        fixed_volume = mx.array(fixed_volume_np, dtype=params.dtype)
        free_mask_volume = mx.array(free_pixel_mask.astype(np.float32), dtype=params.dtype)

        fixed_projections = forward_project_views(
            fixed_volume,
            forward_project,
            len(measured),
            params,
        )
        free_measurements = [
            measured_projection - fixed_projections[projection_index]
            for projection_index, measured_projection in enumerate(measured)
        ]

        def masked_forward_project(volume: mx.array, projection_index: int) -> mx.array:
            masked_volume = volume * free_mask_volume
            return forward_project(masked_volume, projection_index)

        def masked_back_project(projection: mx.array, projection_index: int) -> mx.array:
            backprojection = back_project(projection, projection_index)
            return backprojection * free_mask_volume

        volume = run_iterative_sweeps(
            volume=updated_volume,
            measured_projections=free_measurements,
            ones_volume=ones_volume,
            forward_project=masked_forward_project,
            back_project=masked_back_project,
            params=params,
            outer_iteration_index=iteration,
        )
        if params.apply_smoothing and iteration < params.iteration_count - 1:
            smoothed_volume = _smooth_boundary_voxels(
                np.asarray(volume),
                boundary_mask,
                beta=float(params.smoothing_beta),
                connectivity=params.boundary_connectivity,
            )
            volume = mx.array(smoothed_volume, dtype=params.dtype)
        volume = clamp_reconstruction_volume(volume, params, stage="iteration")
        converged = False
        if params.convergence_epsilon is not None:
            updated_segmented = _segment_dart_volume(np.asarray(volume), gray_levels, params)
            changed_ratio = float(np.count_nonzero(segmented != updated_segmented)) / float(segmented.size)
            converged = changed_ratio <= float(params.convergence_epsilon)
        if show_progress:
            print_progress(iteration, params.iteration_count)
            if converged and iteration < params.iteration_count - 1:
                print()
        if converged:
            break

    volume = clamp_reconstruction_volume(volume, params, stage="final")
    if params.return_segmented_volume:
        segmented = _segment_dart_volume(np.asarray(volume), gray_levels, params)
        return mx.array(segmented, dtype=params.dtype)
    return volume


run_dart = reconstruct_dart
