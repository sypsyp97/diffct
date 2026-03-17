"""Standalone SART diagnostics for simulated and measured cone-beam data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import diffct_mlx


def _default_measured_cone_config() -> diffct_mlx.MeasuredConeDataConfig:
    """Return the sample-data configuration used by the cone iterative example."""
    sample_root = Path(__file__).resolve().parent / "non_circular_trajectory" / "sample_data"
    real_data_dir = sample_root / "sim_obj_1_tif"
    return diffct_mlx.MeasuredConeDataConfig(
        data_dir=real_data_dir,
        trajectory_json_path=real_data_dir / "sim_obj_1_geometry_diffct.json",
        reference_volume_path=sample_root / "reko" / "sim_obj_1_diffct.npy",
        reference_meta_path=sample_root / "reko" / "sim_obj_1_diffct.json",
    )


def _print_scalar_summary(result: diffct_mlx.SARTDiagnosticsResult, *, limit: int = 12) -> None:
    """Print the first few collected SART scalar statistics."""
    print("\nFirst diagnostic steps")
    for stats in result.scalar_stats[:limit]:
        print(
            f"iter={int(stats['outer_iteration'])} "
            f"sweep={int(stats['sart_sweep'])} "
            f"view={int(stats['projection_index']):03d} "
            f"meas_max={stats['measured_max']:.4f} "
            f"comp_max={stats['computed_max']:.4f} "
            f"ray_p001={stats['raylength_positive_p001']:.6f} "
            f"corr_norm={stats['correction_norm']:.4f} "
            f"prep_b={stats['prepared_border_mean']:.4f} "
            f"comp_b={stats['computed_border_mean']:.4f} "
            f"res_b={stats['residual_border_mean']:.4f} "
            f"corr_b={stats['correction_border_mean']:.4f} "
            f"bp_norm={stats['backprojection_norm']:.4f} "
            f"vol_max={stats['volume_max']:.4f}"
        )


def diagnose_3d_cone(
    *,
    data_source: str = "measured",
    tracked_projection_index: int = 0,
    iteration_count: int = 3,
    sart_iteration_count: int = 2,
    log_transform: bool | None = None,
    revert: bool | None = None,
    preserve_unmasked_computed_projection: bool = False,
    positivity_mode: str = "final",
    detector_border_u: int = 8,
    detector_border_v: int = 8,
    backprojection_scale: float = 0.05,
) -> diffct_mlx.SARTDiagnosticsResult:
    """Follow one cone-beam projection across multiple SART steps."""
    if data_source == "measured":
        measured_config = _default_measured_cone_config()
        if log_transform is not None:
            measured_config.log_transform = bool(log_transform)
        if revert is not None:
            measured_config.revert = bool(revert)
        case = diffct_mlx.build_measured_cone_3d_case(measured_config)
        figure_title = f"SART Diagnostics: Measured Cone 3D, view {tracked_projection_index}"
        output_name = f"diagnose_sart_measured_view_{tracked_projection_index:03d}.png"
    elif data_source == "shepp_logan":
        case = diffct_mlx.build_cone_3d_case()
        figure_title = f"SART Diagnostics: Shepp-Logan Cone 3D, view {tracked_projection_index}"
        output_name = f"diagnose_sart_shepp_logan_view_{tracked_projection_index:03d}.png"
    else:
        raise ValueError("data_source must be either 'measured' or 'shepp_logan'.")

    params = diffct_mlx.ReconstructionParameters(
        volume_shape=case.volume_shape,
        iteration_count=iteration_count,
        sart_iteration_count=sart_iteration_count,
        enforce_positivity=True,
        positivity_mode=positivity_mode,
        shuffle_projection_order=False,
        preserve_unmasked_computed_projection=preserve_unmasked_computed_projection,
        detector_border_u=detector_border_u,
        detector_border_v=detector_border_v,
        backprojection_scale=backprojection_scale,
    )
    measured_projections = [case.sinogram[index] for index in range(case.sinogram.shape[0])]

    result = diffct_mlx.diagnose_sart(
        measured_projections,
        case.forward_single,
        case.back_single,
        params,
        tracked_projection_indices=(tracked_projection_index,),
        show_progress=True,
    )

    _print_scalar_summary(result)

    final_volume_np = np.asarray(result.final_volume)
    print(
        "\nFinal volume after diagnostic pass "
        f"shape={final_volume_np.shape} "
        f"range=[{final_volume_np.min():.4f}, {final_volume_np.max():.4f}]"
    )

    diffct_mlx.plot_sart_projection_diagnostics(
        result,
        output_path=Path(output_name),
        figure_title=figure_title,
        projection_index=tracked_projection_index,
    )
    return result


if __name__ == "__main__":
    diagnose_3d_cone(
    data_source="measured",
    tracked_projection_index=0,
    iteration_count=3,
        sart_iteration_count=2,
        log_transform=True,
        revert=False,
        preserve_unmasked_computed_projection=True,
        positivity_mode="final",
        detector_border_u=16,
        detector_border_v=16,
        backprojection_scale=0.05,
)
