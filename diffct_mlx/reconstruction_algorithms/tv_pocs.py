"""Geometry-agnostic TV-POCS reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mlx.core as mx

from ..regularizers import tv_pocs as tv_pocs_step
from ._core import (
    ArrayLike,
    BackProjector,
    ForwardProjector,
    ReconstructionParameters,
    RegularizationParameters,
    clamp_reconstruction_volume,
    initialize_volume,
    normalize_measured_projections,
    print_progress,
    run_sart_sweeps,
    validate_reconstruction_inputs,
)


@dataclass
class TVPOCSParameters(RegularizationParameters):
    """TV-POCS regularization parameters."""


TV_POCS_Parameter = TVPOCSParameters


def reconstruct_tv_pocs(
    measured_projections: Sequence[ArrayLike],
    forward_project: ForwardProjector,
    back_project: BackProjector,
    reco_params: ReconstructionParameters,
    reg_params: TVPOCSParameters,
    *,
    show_progress: bool = False,
) -> mx.array:
    """Reconstruct with TV-POCS using user-provided per-view operators."""
    validate_reconstruction_inputs(reco_params, measured_projections)
    measured = normalize_measured_projections(measured_projections, reco_params.dtype)
    volume, ones_volume, zero_volume = initialize_volume(reco_params)

    for iteration in range(reco_params.iteration_count):
        skip_first_sart = iteration == 0 and reco_params.initial_volume is not None
        iteration_reference = zero_volume if skip_first_sart else volume

        if not skip_first_sart:
            volume = run_sart_sweeps(
                volume=volume,
                measured_projections=measured,
                ones_volume=ones_volume,
                forward_project=forward_project,
                back_project=back_project,
                params=reco_params,
                outer_iteration_index=iteration,
            )

        volume = clamp_reconstruction_volume(volume, reco_params, stage="iteration")

        if reg_params.reg_iteration_count > 0:
            volume, _ = tv_pocs_step(
                volume,
                iteration_reference,
                reg_params.reg_iteration_count,
                reg_params.alpha,
                eta=reg_params.tv_eps,
            )
            volume = clamp_reconstruction_volume(volume, reco_params, stage="regularization")

        if show_progress:
            print_progress(iteration, reco_params.iteration_count)

    volume = clamp_reconstruction_volume(volume, reco_params, stage="final")
    return volume


run_tv_pocs = reconstruct_tv_pocs
