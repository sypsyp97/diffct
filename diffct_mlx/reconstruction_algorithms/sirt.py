"""Geometry-agnostic SIRT reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mlx.core as mx

from ._core import (
    ArrayLike,
    BackProjector,
    ForwardProjector,
    ReconstructionParameters,
    clamp_reconstruction_volume,
    initialize_volume,
    normalize_measured_projections,
    print_progress,
    run_sirt_sweeps,
    validate_reconstruction_inputs,
)


@dataclass
class SIRTParameters(ReconstructionParameters):
    """SIRT-specific configuration."""

    iterative_update_method: str = "sirt"


def reconstruct_sirt(
    measured_projections: Sequence[ArrayLike],
    forward_project: ForwardProjector,
    back_project: BackProjector,
    params: SIRTParameters,
    *,
    show_progress: bool = True,
) -> mx.array:
    """Reconstruct with SIRT using user-provided per-view operators."""
    validate_reconstruction_inputs(params, measured_projections)
    measured = normalize_measured_projections(measured_projections, params.dtype)
    volume, ones_volume, _ = initialize_volume(params)

    for iteration in range(params.iteration_count):
        skip_first_sirt = iteration == 0 and params.initial_volume is not None
        if not skip_first_sirt:
            volume = run_sirt_sweeps(
                volume=volume,
                measured_projections=measured,
                ones_volume=ones_volume,
                forward_project=forward_project,
                back_project=back_project,
                params=params,
                outer_iteration_index=iteration,
            )
        volume = clamp_reconstruction_volume(volume, params, stage="iteration")
        if show_progress:
            print_progress(iteration, params.iteration_count)

    volume = clamp_reconstruction_volume(volume, params, stage="final")
    return volume


run_sirt = reconstruct_sirt
