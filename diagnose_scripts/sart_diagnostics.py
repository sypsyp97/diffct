"""Utilities for step-by-step SART debugging and projection inspection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

try:
    from ..diffct_mlx.reconstruction_algorithms._core import (
        ArrayLike,
        BackProjector,
        ForwardProjector,
        ReconstructionParameters,
        _array_stats,
        _raylength_stats,
        apply_detector_border_mask,
        clamp_reconstruction_volume,
        compute_sart_correction,
        initialize_volume,
        normalize_measured_projections,
        prepare_measured_projection,
        prepare_raylength_projection,
        print_progress,
        projection_order,
        validate_reconstruction_inputs,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from diffct_mlx.reconstruction_algorithms._core import (
        ArrayLike,
        BackProjector,
        ForwardProjector,
        ReconstructionParameters,
        _array_stats,
        _raylength_stats,
        apply_detector_border_mask,
        clamp_reconstruction_volume,
        compute_sart_correction,
        initialize_volume,
        normalize_measured_projections,
        prepare_measured_projection,
        prepare_raylength_projection,
        print_progress,
        projection_order,
        validate_reconstruction_inputs,
    )


@dataclass
class SARTProjectionSnapshot:
    """Saved projection-space state for one SART view update."""

    outer_iteration: int
    sart_sweep: int
    projection_index: int
    prepared_projection: np.ndarray
    computed_projection: np.ndarray
    residual_projection: np.ndarray
    raylength_projection: np.ndarray
    correction_image: np.ndarray
    backprojection_norm: float
    volume_max_after_update: float


@dataclass
class SARTDiagnosticsResult:
    """Outputs collected while replaying SART step by step."""

    final_volume: mx.array
    snapshots: list[SARTProjectionSnapshot]
    scalar_stats: list[dict[str, float]]


def _border_mean(value: mx.array | np.ndarray, border_u: int, border_v: int) -> float:
    """Compute the mean over detector-border pixels."""
    value_np = np.asarray(value)
    if value_np.ndim == 1:
        trim_u = max(1, min(max(border_u, 1), value_np.shape[0] // 2 if value_np.shape[0] > 1 else 1))
        border = np.concatenate([value_np[:trim_u], value_np[-trim_u:]], axis=0)
        return float(np.mean(border))

    if value_np.ndim != 2:
        return float(np.mean(value_np))

    trim_u = max(1, min(max(border_u, 1), value_np.shape[0] // 2 if value_np.shape[0] > 1 else 1))
    trim_v = max(1, min(max(border_v, 1), value_np.shape[1] // 2 if value_np.shape[1] > 1 else 1))
    mask = np.zeros_like(value_np, dtype=bool)
    mask[:trim_u, :] = True
    mask[-trim_u:, :] = True
    mask[:, :trim_v] = True
    mask[:, -trim_v:] = True
    return float(np.mean(value_np[mask]))


def _to_numpy_copy(value: mx.array) -> np.ndarray:
    """Materialize an MLX array as a standalone NumPy copy."""
    return np.array(np.asarray(value), copy=True)


def default_projection_indices(num_projections: int) -> tuple[int, ...]:
    """Choose a few representative projection indices."""
    if num_projections <= 0:
        return ()
    candidates = [0, num_projections // 2, num_projections - 1]
    ordered_unique: list[int] = []
    for index in candidates:
        if index not in ordered_unique:
            ordered_unique.append(int(index))
    return tuple(ordered_unique)


def diagnose_sart(
    measured_projections: Sequence[ArrayLike],
    forward_project: ForwardProjector,
    back_project: BackProjector,
    params: ReconstructionParameters,
    *,
    tracked_projection_indices: Sequence[int] | None = None,
    show_progress: bool = False,
) -> SARTDiagnosticsResult:
    """Replay SART while capturing per-view scalar stats and projection snapshots."""
    validate_reconstruction_inputs(params, measured_projections)
    measured = normalize_measured_projections(measured_projections, params.dtype)
    volume, ones_volume, _ = initialize_volume(params)
    order = projection_order(params, len(measured))

    if tracked_projection_indices is None:
        tracked = set(default_projection_indices(len(measured)))
    else:
        tracked = {int(index) for index in tracked_projection_indices}

    snapshots: list[SARTProjectionSnapshot] = []
    scalar_stats: list[dict[str, float]] = []

    for iteration in range(params.iteration_count):
        skip_first_sart = iteration == 0 and params.initial_volume is not None
        if not skip_first_sart:
            for sweep_index in range(params.sart_iteration_count):
                for projection_index in order:
                    measured_projection = measured[projection_index]
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
                    prepared_projection = prepare_measured_projection(
                        measured_projection,
                        computed_projection,
                        params,
                    )
                    residual_projection = prepared_projection - computed_projection
                    prepared_border_mean = _border_mean(
                        prepared_projection,
                        params.detector_border_u,
                        params.detector_border_v,
                    )
                    prepared_projection = apply_detector_border_mask(prepared_projection, params, fill_value=0.0)
                    backprojection_volume = back_project(correction_image, projection_index)
                    volume = volume + (
                        float(params.backprojection_scale) * backprojection_volume
                    )
                    volume = clamp_reconstruction_volume(volume, params, stage="sart_update")

                    debug_stats = {
                        "outer_iteration": float(iteration),
                        "sart_sweep": float(sweep_index),
                        "projection_index": float(projection_index),
                        "backprojection_scale": float(params.backprojection_scale),
                        **_array_stats("measured", measured_projection),
                        **_array_stats("computed", computed_projection),
                        **_raylength_stats(raylength_projection),
                        **_array_stats("correction", correction_image),
                        **_array_stats("backprojection", backprojection_volume),
                        **_array_stats("volume", volume),
                        "prepared_border_mean": prepared_border_mean,
                        "computed_border_mean": _border_mean(
                            computed_projection,
                            params.detector_border_u,
                            params.detector_border_v,
                        ),
                        "residual_border_mean": _border_mean(
                            residual_projection,
                            params.detector_border_u,
                            params.detector_border_v,
                        ),
                        "correction_border_mean": _border_mean(
                            correction_image,
                            params.detector_border_u,
                            params.detector_border_v,
                        ),
                    }
                    scalar_stats.append(debug_stats)

                    if projection_index in tracked:
                        snapshots.append(
                            SARTProjectionSnapshot(
                                outer_iteration=iteration,
                                sart_sweep=sweep_index,
                                projection_index=projection_index,
                                prepared_projection=_to_numpy_copy(prepared_projection),
                                computed_projection=_to_numpy_copy(computed_projection),
                                residual_projection=_to_numpy_copy(residual_projection),
                                raylength_projection=_to_numpy_copy(raylength_projection),
                                correction_image=_to_numpy_copy(correction_image),
                                backprojection_norm=float(debug_stats["backprojection_norm"]),
                                volume_max_after_update=float(debug_stats["volume_max"]),
                            )
                        )

        volume = clamp_reconstruction_volume(volume, params, stage="iteration")

        if show_progress:
            print_progress(iteration, params.iteration_count)

    return SARTDiagnosticsResult(
        final_volume=clamp_reconstruction_volume(volume, params, stage="final"),
        snapshots=snapshots,
        scalar_stats=scalar_stats,
    )


def plot_sart_projection_diagnostics(
    result: SARTDiagnosticsResult,
    *,
    output_path: str | Path | None = None,
    figure_title: str | None = None,
    projection_index: int | None = None,
) -> None:
    """Plot the evolution of one tracked projection across SART steps."""
    if not result.snapshots:
        raise ValueError("No SART projection snapshots were collected.")

    if projection_index is None:
        projection_index = int(result.snapshots[0].projection_index)
    snapshots = [snapshot for snapshot in result.snapshots if snapshot.projection_index == projection_index]
    if not snapshots:
        raise ValueError(f"No snapshots were collected for projection_index={projection_index}.")

    ncols = len(snapshots)
    fig, axes = plt.subplots(5, ncols, figsize=(4.2 * ncols, 15.0), squeeze=False)

    for column, snapshot in enumerate(snapshots):
        display_min = float(np.percentile(snapshot.prepared_projection, 1.0))
        display_max = float(np.percentile(snapshot.prepared_projection, 99.0))
        if not np.isfinite(display_max) or display_max <= display_min:
            display_min = float(np.min(snapshot.prepared_projection))
            display_max = float(np.max(snapshot.prepared_projection))
            if display_max <= display_min:
                display_max = display_min + 1.0

        correction_abs_max = float(np.percentile(np.abs(snapshot.correction_image), 99.0))
        if correction_abs_max <= 0.0:
            correction_abs_max = 1.0

        residual_abs_max = float(np.percentile(np.abs(snapshot.residual_projection), 99.0))
        if residual_abs_max <= 0.0:
            residual_abs_max = 1.0

        title = (
            f"iter={snapshot.outer_iteration} "
            f"sweep={snapshot.sart_sweep} "
            f"view={snapshot.projection_index}"
        )

        axes[0, column].imshow(
            snapshot.prepared_projection,
            cmap="gray",
            origin="lower",
            vmin=display_min,
            vmax=display_max,
        )
        axes[0, column].set_title(f"Prepared Projection\n{title}")
        axes[0, column].axis("off")

        axes[1, column].imshow(
            snapshot.computed_projection,
            cmap="gray",
            origin="lower",
            vmin=display_min,
            vmax=display_max,
        )
        axes[1, column].set_title("Computed Projection")
        axes[1, column].axis("off")

        axes[2, column].imshow(
            snapshot.residual_projection,
            cmap="bwr",
            origin="lower",
            vmin=-residual_abs_max,
            vmax=residual_abs_max,
        )
        axes[2, column].set_title("Prepared - Computed")
        axes[2, column].axis("off")

        axes[3, column].imshow(snapshot.raylength_projection, cmap="viridis", origin="lower")
        axes[3, column].set_title("Raylength")
        axes[3, column].axis("off")

        axes[4, column].imshow(
            snapshot.correction_image,
            cmap="bwr",
            origin="lower",
            vmin=-correction_abs_max,
            vmax=correction_abs_max,
        )
        axes[4, column].set_title(
            "Correction\n"
            f"bp_norm={snapshot.backprojection_norm:.3f}\n"
            f"vol_max={snapshot.volume_max_after_update:.3f}"
        )
        axes[4, column].axis("off")

    if figure_title is not None:
        fig.suptitle(figure_title)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")

    plt.show()
