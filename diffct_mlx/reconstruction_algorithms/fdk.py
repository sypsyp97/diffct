"""Projector-agnostic Feldkamp-Davis-Kress-style reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

from ._analytic import (
    AnalyticalBackProjector,
    AnalyticalFilter,
    AnalyticalReconstructionParameters,
    AnalyticalWeighting,
    _as_array,
    _positive_and_scale,
    default_angular_normalization,
    ramp_filter,
)


@dataclass
class FDKParameters(AnalyticalReconstructionParameters):
    """Parameters for projector-agnostic FDK reconstruction."""

    filter_axis: int = 1


def reconstruct_fdk(
    sinogram,
    back_project: AnalyticalBackProjector,
    params: FDKParameters,
    *,
    weight_projections: AnalyticalWeighting | None = None,
    filter_projections: AnalyticalFilter | None = None,
):
    """Reconstruct with FDK using user-provided weighting/filtering/backprojection."""
    sinogram = _as_array(sinogram, dtype=params.dtype)
    if sinogram.ndim < 3:
        raise ValueError(f"Expected cone-beam sinogram with at least 3 dimensions, got shape {sinogram.shape!r}.")

    weighted_sinogram = sinogram if weight_projections is None else _as_array(weight_projections(sinogram), dtype=params.dtype)
    if weighted_sinogram.shape != sinogram.shape:
        raise ValueError(
            "weight_projections must preserve sinogram shape: "
            f"expected {sinogram.shape!r}, got {weighted_sinogram.shape!r}."
        )

    if filter_projections is None:
        filtered_sinogram = ramp_filter(weighted_sinogram, axis=params.filter_axis)
    else:
        filtered_sinogram = _as_array(filter_projections(weighted_sinogram), dtype=params.dtype)
    if filtered_sinogram.shape != sinogram.shape:
        raise ValueError(
            "filter_projections must preserve sinogram shape: "
            f"expected {sinogram.shape!r}, got {filtered_sinogram.shape!r}."
        )

    reconstruction = _as_array(back_project(filtered_sinogram), dtype=params.dtype)
    normalization = params.normalization_scale
    if normalization is None:
        normalization = default_angular_normalization(sinogram.shape[0])
    return _positive_and_scale(
        reconstruction,
        enforce_positivity=params.enforce_positivity,
        normalization_scale=normalization,
    )


run_fdk = reconstruct_fdk
