"""Geometry-agnostic ASD-POCS reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mlx.core as mx

from ..regularizers import asd_pocs as asd_pocs_step
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
    projection_order,
    projection_residual_norm,
    run_iterative_sweeps,
    scalar_norm,
    validate_reconstruction_inputs,
)


@dataclass
class ASDPOCSParameters(RegularizationParameters):
    """ASD-POCS regularization parameters."""

    epsilon: float = 5.0
    r_max: float = 0.95
    alpha_red: float = 0.95
    beta: float = 1.0
    beta_red: float = 0.995


ASD_POCS_Parameter = ASDPOCSParameters


def reconstruct_asd_pocs(
    measured_projections: Sequence[ArrayLike],
    forward_project: ForwardProjector,
    back_project: BackProjector,
    reco_params: ReconstructionParameters,
    reg_params: ASDPOCSParameters,
    *,
    show_progress: bool = False,
) -> mx.array:
    """Reconstruct with ASD-POCS using user-provided per-view operators."""
    validate_reconstruction_inputs(reco_params, measured_projections)
    measured = normalize_measured_projections(measured_projections, reco_params.dtype)
    order = projection_order(reco_params, len(measured))
    volume, ones_volume, zero_volume = initialize_volume(reco_params)

    beta = float(reg_params.beta)
    dtvg = 0.0
    first_projection_index = order[0]

    for iteration in range(reco_params.iteration_count):
        skip_first_sart = iteration == 0 and reco_params.initial_volume is not None
        iteration_reference = zero_volume if skip_first_sart else volume

        if not skip_first_sart:
            volume = run_iterative_sweeps(
                volume=volume,
                measured_projections=measured,
                ones_volume=ones_volume,
                forward_project=forward_project,
                back_project=back_project,
                params=reco_params,
                beta=beta,
                outer_iteration_index=iteration,
            )

        volume = clamp_reconstruction_volume(volume, reco_params, stage="iteration")

        if reg_params.reg_iteration_count > 0 and iteration < (reco_params.iteration_count - 1):
            dd = projection_residual_norm(
                volume,
                measured[first_projection_index],
                forward_project,
                first_projection_index,
            )
            volume, _, dtvg = asd_pocs_step(
                volume,
                iteration_reference,
                iteration,
                reg_params.reg_iteration_count,
                dd,
                dtvg,
                reg_params.alpha,
                reg_params.alpha_red,
                reg_params.r_max,
                reg_params.epsilon,
                eta=reg_params.tv_eps,
            )
            volume = clamp_reconstruction_volume(volume, reco_params, stage="regularization")

            beta *= float(reg_params.beta_red)

        if show_progress:
            print_progress(iteration, reco_params.iteration_count)

    volume = clamp_reconstruction_volume(volume, reco_params, stage="final")
    return volume


run_asd_pocs = reconstruct_asd_pocs
