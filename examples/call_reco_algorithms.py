"""Compare reconstruction algorithms across 2D and 3D CT geometries."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import diffct_mlx


def _default_measured_cone_config() -> diffct_mlx.MeasuredConeDataConfig:
    """Return measured-data loading and preprocessing settings for this example.

    This helper only configures how the TIFF projections and geometry are read:
    sample-data paths, log-domain conversion, per-view air normalization, and
    air-baseline subtraction. It does not change any reconstruction algorithm
    parameters.
    """
    sample_root = Path(__file__).resolve().parent / "non_circular_trajectory" / "sample_data"
    real_data_dir = sample_root / "sim_obj_1_tif"
    return diffct_mlx.MeasuredConeDataConfig(
        data_dir=real_data_dir,
        trajectory_json_path=real_data_dir / "sim_obj_1_geometry_diffct.json",
        reference_volume_path=sample_root / "reko" / "sim_obj_1_diffct.npy",
        reference_meta_path=sample_root / "reko" / "sim_obj_1_diffct.json",
        log_transform=True,
        revert=False,
        viewwise_i0=True,
        air_border_px=16,
        subtract_air_baseline=True,
        air_baseline_percentile=50.0,
    )


def _apply_measured_iterative_example_settings(case: diffct_mlx.ReconstructionCase) -> diffct_mlx.ReconstructionCase:
    """Override reconstruction parameters for the measured-data comparison run.

    This helper only adjusts iterative reconstruction settings on an already
    built case, for example SART step size, positivity handling, detector-border
    masking, and TV-family regularization strengths. It does not modify how the
    measured projections were loaded or preprocessed.
    """
    return replace(
        case,
        iterative_iteration_count=10,
        iterative_sart_iteration_count=2,
        iterative_backprojection_scale=0.05,
        iterative_positivity_mode="final",
        iterative_detector_border_u=16,
        iterative_detector_border_v=16,
        iterative_preserve_unmasked_computed_projection=True,
        tv_reg_iteration_count=20,
        tv_alpha=0.15,
        asd_reg_iteration_count=15,
        asd_alpha=0.12,
        asd_epsilon=0.2,
        awtv_reg_iteration_count=20,
        awtv_alpha=0.15,
        awtv_epsilon=0.08,
        awtv_delta=0.6e-2,
    )


def _compute_metrics(reconstruction, reference):
    """Compute image-quality metrics when a reference is available."""
    if reference is None:
        return None
    reconstruction_np = np.asarray(reconstruction)
    reference_np = np.asarray(reference)
    mse = float(np.mean((reconstruction_np - reference_np) ** 2))
    dynamic_range = float(reference_np.max() - reference_np.min())
    if dynamic_range <= 0.0:
        dynamic_range = 1.0
    psnr = float("inf") if mse <= 0.0 else 10.0 * np.log10((dynamic_range**2) / mse)
    return {"mse": mse, "psnr": psnr}


def _print_result(case_name: str, algorithm_name: str, reconstruction, reference) -> None:
    """Print a compact summary for one reconstruction result."""
    reconstruction_np = np.asarray(reconstruction)
    metrics = _compute_metrics(reconstruction, reference)
    if metrics is None:
        print(
            f"{case_name:<18} {algorithm_name:<10} "
            f"shape={tuple(reconstruction_np.shape)!s:<16} "
            f"range=[{reconstruction_np.min():.4f}, {reconstruction_np.max():.4f}]"
        )
        return
    print(
        f"{case_name:<18} {algorithm_name:<10} "
        f"shape={tuple(reconstruction_np.shape)!s:<16} "
        f"range=[{reconstruction_np.min():.4f}, {reconstruction_np.max():.4f}] "
        f"mse={metrics['mse']:.6f} "
        f"psnr={metrics['psnr']:.2f} dB"
    )


def _iterative_results(case: diffct_mlx.ReconstructionCase) -> dict[str, object]:
    """Run the iterative algorithms for one geometry case."""
    shared_reco_params = diffct_mlx.ReconstructionParameters(
        volume_shape=case.volume_shape,
        iteration_count=case.iterative_iteration_count,
        sart_iteration_count=case.iterative_sart_iteration_count,
        enforce_positivity=True,
        positivity_mode=case.iterative_positivity_mode,
        preserve_unmasked_computed_projection=case.iterative_preserve_unmasked_computed_projection,
        detector_border_u=case.iterative_detector_border_u,
        detector_border_v=case.iterative_detector_border_v,
        backprojection_scale=case.iterative_backprojection_scale,
    )
    sart_params = diffct_mlx.SARTParameters(
        volume_shape=case.volume_shape,
        iteration_count=case.iterative_iteration_count,
        sart_iteration_count=case.iterative_sart_iteration_count,
        enforce_positivity=True,
        positivity_mode=case.iterative_positivity_mode,
        preserve_unmasked_computed_projection=case.iterative_preserve_unmasked_computed_projection,
        detector_border_u=case.iterative_detector_border_u,
        detector_border_v=case.iterative_detector_border_v,
        backprojection_scale=case.iterative_backprojection_scale,
    )
    measured_projections = [case.sinogram[i] for i in range(case.sinogram.shape[0])]

    return {
        "SART": diffct_mlx.run_sart(
            measured_projections,
            case.forward_single,
            case.back_single,
            sart_params,
            show_progress=True,
        ),
        "TV-POCS": diffct_mlx.run_tv_pocs(
            measured_projections,
            case.forward_single,
            case.back_single,
            shared_reco_params,
            diffct_mlx.TVPOCSParameters(
                reg_iteration_count=case.tv_reg_iteration_count,
                alpha=case.tv_alpha,
            ),
            show_progress=True,
        ),
        "ASD-POCS": diffct_mlx.run_asd_pocs(
            measured_projections,
            case.forward_single,
            case.back_single,
            shared_reco_params,
            diffct_mlx.ASDPOCSParameters(
                reg_iteration_count=case.asd_reg_iteration_count,
                alpha=case.asd_alpha,
                epsilon=case.asd_epsilon,
                beta=1.0,
            ),
            show_progress=True,
        ),
        "AwTV-POCS": diffct_mlx.run_awtv_pocs(
            measured_projections,
            case.forward_single,
            case.back_single,
            shared_reco_params,
            diffct_mlx.AwTVPOCSParameters(
                reg_iteration_count=case.awtv_reg_iteration_count,
                alpha=case.awtv_alpha,
                epsilon=case.awtv_epsilon,
                delta=case.awtv_delta,
                beta=1.0,
            ),
            show_progress=True,
        ),
    }


def _fbp_result(case: diffct_mlx.ReconstructionCase):
    """Run generic FBP for one case."""
    if not case.supports_fbp or case.fbp_normalization_scale is None:
        raise ValueError(f"FBP is not configured for case {case.name!r}.")
    return diffct_mlx.run_fbp(
        case.sinogram,
        back_project=case.back_project_all,
        params=diffct_mlx.FBPParameters(
            normalization_scale=case.fbp_normalization_scale,
            filter_axis=1,
        ),
        weight_projections=case.fbp_weight,
    )


def _fdk_result(case: diffct_mlx.ReconstructionCase):
    """Run generic FDK for one case."""
    if not case.supports_fdk or case.fdk_normalization_scale is None:
        raise ValueError(f"FDK is not configured for case {case.name!r}.")
    return diffct_mlx.run_fdk(
        case.sinogram,
        back_project=case.back_project_all,
        params=diffct_mlx.FDKParameters(
            normalization_scale=case.fdk_normalization_scale,
            filter_axis=1,
        ),
        weight_projections=case.fdk_weight,
    )


def _plot_comparison_results(
    case: diffct_mlx.ReconstructionCase,
    results: dict[str, object],
    output_name: str,
    names: list[str],
) -> None:
    """Plot comparisons in a layout similar to the iterative cone example."""
    n = len(names)
    reference_np = None if case.reference is None else np.asarray(case.reference)
    reference_slice = None
    if reference_np is not None:
        reference_slice = reference_np[reference_np.shape[0] // 2] if reference_np.ndim == 3 else reference_np

    measured_panel = None
    if reference_slice is None:
        sinogram_np = np.asarray(case.sinogram)
        measured_panel = sinogram_np[sinogram_np.shape[0] // 2] if sinogram_np.ndim == 3 else sinogram_np

    fig = plt.figure(figsize=(4.5 * n, 10))

    for index, name in enumerate(names):
        reconstruction_np = np.asarray(results[name])
        reconstruction_slice = (
            reconstruction_np[reconstruction_np.shape[0] // 2] if reconstruction_np.ndim == 3 else reconstruction_np
        )
        display_max = float(np.max(reconstruction_slice))
        if display_max <= 0.0:
            display_max = 1.0

        plt.subplot(3, n, index + 1)
        metrics = _compute_metrics(results[name], case.reference)
        if reference_slice is not None and metrics is not None:
            error_map = np.abs(reconstruction_slice - reference_slice)
            plt.imshow(error_map, cmap="magma")
            plt.title(f"{name} Error\nMSE: {metrics['mse']:.4e}\nPSNR: {metrics['psnr']:.2f} dB")
        else:
            plt.axis("off")
            plt.title(f"{name}\nNo reference")
        plt.axis("off")

        plt.subplot(3, n, n + index + 1)
        if reference_slice is not None:
            plt.imshow(reference_slice, cmap="gray", vmin=0, vmax=1)
            plt.title(case.reference_title or "Reference")
        elif measured_panel is not None:
            plt.imshow(measured_panel, cmap="gray")
            plt.title("Measured view")
        else:
            plt.axis("off")
            plt.title("No reference")
        plt.axis("off")

        plt.subplot(3, n, 2 * n + index + 1)
        plt.imshow(reconstruction_slice, cmap="gray", vmin=0, vmax=display_max)
        plt.title(name)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_name, dpi=200, bbox_inches="tight")
    plt.show()


def compare_2d_parallel() -> dict[str, object]:
    """Compare FBP and iterative methods on 2D parallel-beam Shepp-Logan data."""
    case = diffct_mlx.build_parallel_2d_case()
    ordered_names = ["FBP", "SART", "TV-POCS", "ASD-POCS", "AwTV-POCS"]
    results = {"FBP": _fbp_result(case)}
    results.update(_iterative_results(case))

    print("\n2D Parallel")
    for name in ordered_names:
        _print_result(case.name, name, results[name], case.reference)

    _plot_comparison_results(case, results, "compare_2d_parallel.png", ordered_names)
    return results


def compare_2d_fan() -> dict[str, object]:
    """Compare FBP and iterative methods on 2D fan-beam Shepp-Logan data."""
    case = diffct_mlx.build_fan_2d_case()
    ordered_names = ["FBP", "SART", "TV-POCS", "ASD-POCS", "AwTV-POCS"]
    results = {"FBP": _fbp_result(case)}
    results.update(_iterative_results(case))

    print("\n2D Fan")
    for name in ordered_names:
        _print_result(case.name, name, results[name], case.reference)

    _plot_comparison_results(case, results, "compare_2d_fan.png", ordered_names)
    return results


def compare_3d_cone(
    *, 
    data_source: str = "shepp_logan",
    measured_config: diffct_mlx.MeasuredConeDataConfig | None = None) -> dict[str, object]:
    """Compare cone-beam reconstruction algorithms for synthetic or measured data."""
    if data_source == "shepp_logan":
        case = diffct_mlx.build_cone_3d_case()
        output_name = "compare_3d_cone.png"
    elif data_source == "measured":
        case = diffct_mlx.build_measured_cone_3d_case(measured_config or _default_measured_cone_config())
        case = _apply_measured_iterative_example_settings(case)
        output_name = "compare_3d_cone_measured.png"
    else:
        raise ValueError("data_source must be either 'shepp_logan' or 'measured'.")

    results: dict[str, object] = {}
    ordered_names: list[str] = []
    if case.supports_fdk:
        results["FDK"] = _fdk_result(case)
        ordered_names.append("FDK")
    if case.supports_fbp:
        results["FBP"] = _fbp_result(case)
        ordered_names.append("FBP")

    if data_source == "measured" and not case.supports_fbp and not case.supports_fdk:
        print("\n3D Cone (Measured)")
        print("Skipping FBP/FDK: the loaded measured trajectory is arbitrary rather than circular.")
    else:
        print("\n3D Cone")

    iterative = _iterative_results(case)
    results.update(iterative)
    ordered_names.extend(["SART", "TV-POCS", "ASD-POCS", "AwTV-POCS"])

    for name in ordered_names:
        _print_result(case.name, name, results[name], case.reference)

    _plot_comparison_results(case, results, output_name, ordered_names)
    return results


def main() -> None:
    # compare_2d_parallel()
    # compare_2d_fan()
    compare_3d_cone(data_source="shepp_logan")
    # compare_3d_cone(data_source="measured", measured_config=_default_measured_cone_config())


if __name__ == "__main__":
    main()
