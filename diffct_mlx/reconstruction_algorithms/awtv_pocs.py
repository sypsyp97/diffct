"""Geometry-agnostic AwTV-POCS reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mlx.core as mx
import numpy as np

from ..regularizers import awtv_pocs as awtv_pocs_step
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
    run_iterative_sweeps,
    validate_reconstruction_inputs,
)


@dataclass
class AwTVPOCSParameters(RegularizationParameters):
    """AwTV-POCS regularization parameters."""

    epsilon: float = 1.0
    delta: float = 0.6e-2
    beta: float = 1.0
    beta_red: float = 0.995


AwTV_POCS_Parameter = AwTVPOCSParameters


def reconstruct_awtv_pocs(
    measured_projections: Sequence[ArrayLike],
    forward_project: ForwardProjector,
    back_project: BackProjector,
    reco_params: ReconstructionParameters,
    reg_params: AwTVPOCSParameters,
    *,
    show_progress: bool = False,
) -> mx.array:
    """Reconstruct with AwTV-POCS using user-provided per-view operators."""
    validate_reconstruction_inputs(reco_params, measured_projections)
    measured = normalize_measured_projections(measured_projections, reco_params.dtype)
    order = projection_order(reco_params, len(measured))
    volume, ones_volume, zero_volume = initialize_volume(reco_params)

    beta = float(reg_params.beta)
    previous_pocs_projection = None
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

        if reg_params.reg_iteration_count > 0:
            current_pocs_projection = forward_project(volume, first_projection_index)
            if previous_pocs_projection is not None:
                ds = float(np.asarray(mx.linalg.norm(previous_pocs_projection - current_pocs_projection)))
                if ds < float(reg_params.epsilon):
                    beta *= float(reg_params.beta_red)

            volume, _, _ = awtv_pocs_step(
                volume,
                iteration_reference,
                reg_params.reg_iteration_count,
                reg_params.alpha,
                reg_params.delta,
                eta=reg_params.tv_eps,
            )
            volume = clamp_reconstruction_volume(volume, reco_params, stage="regularization")
            previous_pocs_projection = current_pocs_projection

        if show_progress:
            print_progress(iteration, reco_params.iteration_count)

    volume = clamp_reconstruction_volume(volume, reco_params, stage="final")
    return volume


run_awtv_pocs = reconstruct_awtv_pocs
