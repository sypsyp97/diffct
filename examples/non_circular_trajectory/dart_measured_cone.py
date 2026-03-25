"""Run DART on measured non-circular cone-beam projections."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import diffct_mlx


def default_measured_cone_config() -> diffct_mlx.MeasuredConeDataConfig:
    """Return the measured-data configuration used by this example."""
    sample_root = Path(__file__).resolve().parent / "sample_data"
    data_dir = sample_root / "sim_obj_1_tif"
    return diffct_mlx.MeasuredConeDataConfig(
        data_dir=data_dir,
        trajectory_json_path=data_dir / "sim_obj_1_geometry_diffct.json",
        reference_volume_path=sample_root / "reko" / "sim_obj_1_diffct.npy",
        reference_meta_path=sample_root / "reko" / "sim_obj_1_diffct.json",
        log_transform=True,
        revert=False,
        viewwise_i0=True,
        air_border_px=16,
        subtract_air_baseline=True,
        air_baseline_percentile=50.0,
    )


def compute_metrics(reference: np.ndarray | None, reconstruction: np.ndarray) -> tuple[float, float] | None:
    """Return MSE and PSNR when a reference volume is available."""
    if reference is None:
        return None
    mse = float(np.mean((reconstruction - reference) ** 2))
    dynamic_range = float(reference.max() - reference.min())
    if dynamic_range <= 0.0:
        dynamic_range = 1.0
    psnr = float("inf") if mse <= 0.0 else 10.0 * np.log10((dynamic_range**2) / mse)
    return mse, psnr


def print_summary(name: str, reconstruction: np.ndarray, reference: np.ndarray | None) -> None:
    """Print a compact reconstruction summary."""
    metrics = compute_metrics(reference, reconstruction)
    if metrics is None:
        print(
            f"{name:<10} "
            f"shape={tuple(reconstruction.shape)!s:<18} "
            f"range=[{reconstruction.min():.4f}, {reconstruction.max():.4f}]"
        )
        return
    mse, psnr = metrics
    print(
        f"{name:<10} "
        f"shape={tuple(reconstruction.shape)!s:<18} "
        f"range=[{reconstruction.min():.4f}, {reconstruction.max():.4f}] "
        f"mse={mse:.6f} "
        f"psnr={psnr:.2f} dB"
    )


def main() -> None:
    case = diffct_mlx.build_measured_cone_3d_case(default_measured_cone_config())
    measured_projections = [case.sinogram[index] for index in range(int(case.sinogram.shape[0]))]

    sart_params = diffct_mlx.SARTParameters(
        volume_shape=case.volume_shape,  # Reconstruction grid as (z, y, x); normally match the target/reference volume exactly.
        iteration_count=10,  # Number of outer reconstruction sweeps; ~5-20 is a reasonable range before noise/artifacts often dominate.
        sart_iteration_count=2,  # Number of SART updates per outer sweep; 1-4 is typical, higher values trade speed for stronger per-sweep correction.
        enforce_positivity=True,  # Clamp negative voxels after updates; usually keep True for attenuation data, False only if negatives are physically meaningful.
        positivity_mode="final",  # When positivity is applied; "final" is a safe default, stronger modes may stabilize noisy cases but bias values more.
        preserve_unmasked_computed_projection=True,  # Reuse forward projections outside the active detector mask; usually True unless debugging masking behavior.
        detector_border_u=16,  # Ignore this many detector columns at each horizontal edge; try 0-32 depending on how unreliable detector borders are.
        detector_border_v=16,  # Ignore this many detector rows at each vertical edge; try 0-32 when top/bottom rows contain air or truncation artifacts.
        projection_chunk_size=16,  # Batch geometry-invariant forward passes over small view chunks to reduce Python/kernel-launch overhead without changing SART semantics.
        voxel_extreme_values=(0.0, 1.0),  # Expected min/max voxel values used for clipping/normalization; set to your material range, often (0, 1) or known attenuation bounds.
        backprojection_scale=0.05,  # Relaxation step size for backprojection updates; ~0.01-0.2 is a common tuning range, smaller is safer, larger converges faster but can overshoot.
        shuffle_projection_order=False,  # Randomize view order each sweep; False is more reproducible, True can reduce directional bias in some cases.
    )
    dart_params = diffct_mlx.DARTParameters(
        volume_shape=case.volume_shape,  # Reconstruction grid as (z, y, x); keep aligned with the measured geometry and any reference volume.
        iteration_count=10,  # Number of DART outer iterations; ~5-15 is a good starting range for binary or few-material objects.
        sart_iteration_count=2,  # SART updates used inside each DART iteration; 1-4 is typical, larger values refine more but increase runtime.
        initial_reconstruction_sweeps=8,  # A slightly longer warmup usually gives a cleaner initial segmentation and fewer frozen voids in measured data.
        gray_levels=(0.0, 1.0),  # Discrete material values assumed by DART; use 2-5 known attenuation levels, e.g. air/object = (0, 1).
        free_pixel_probability=0.01,  # Keep the random free-pixel fraction low on measured binary data so DART does not carve noise-driven holes through solid regions.
        enforce_positivity=True,  # Clamp negative voxels after updates; usually True for attenuation tomography.
        positivity_mode="final",  # Positivity timing/mode; "final" is conservative, stronger modes can help unstable data at the cost of bias.
        preserve_unmasked_computed_projection=True,  # Reuse forward projections outside the active detector mask; typically leave True for performance.
        segmentation_threshold_method="otsu",  # Thresholding rule for converting warm starts into labels; "otsu" is a robust default for bimodal or sparse-material data.
        otsu_foreground_only=True,  # Run Otsu on the foreground-intensity subset only; often helps when air/background dominates the histogram.
        otsu_percentile_window=(1.0, 99.5),  # Ignore extreme histogram tails before Otsu; ~0.5-2.0 and ~98-99.9 percentiles are reasonable robust settings.
        binary_fill_holes=True,  # Fill enclosed holes during DART segmentation and in the final output; useful for solid objects, disable for porous or hollow structures.
        smoothing_beta=0.35,  # Slightly stronger boundary smoothing helps collapse small cavity artifacts before re-segmentation.
        boundary_connectivity="full",  # Treat diagonal contacts as part of the boundary in 3D so small voids are less likely to become frozen interior background.
        convergence_epsilon=1e-3,  # Stop once the discrete labels stabilize enough, which helps avoid late-iteration overfitting of measurement noise.
        detector_border_u=16,  # Ignore this many detector columns at each horizontal edge; 0-32 is a practical range for measured systems.
        detector_border_v=16,  # Ignore this many detector rows at each vertical edge; 0-32 is common when detector edges are noisy or truncated.
        projection_chunk_size=16,  # Use chunked multi-view forwards for fixed-part and raylength passes; this helps throughput on large measured datasets.
        voxel_extreme_values=(0.0, 1.0),  # Expected min/max voxel values for clipping/normalization; match the discrete gray-level range when possible.
        backprojection_scale=0.05,  # Relaxation step size for update strength; ~0.01-0.2 is a useful tuning range depending on stability and noise.
        shuffle_projection_order=True,  # Randomize projection order between sweeps; often helpful in DART to reduce view-order bias.
        return_segmented_volume=True,  # Return the discretized DART result instead of the intermediate continuous image; usually True for segmentation-focused use.
        random_seed=7,  # Seed for stochastic free-pixel selection and shuffling; any fixed integer is fine when you want reproducible runs.
    )

    print("Running SART baseline...")
    sart_reconstruction = diffct_mlx.run_sart(
        measured_projections,
        case.forward_single,
        case.back_single,
        sart_params,
        show_progress=True,
    )

    print("Running DART...")
    dart_reconstruction = diffct_mlx.run_dart(
        measured_projections,
        case.forward_single,
        case.back_single,
        dart_params,
        show_progress=True,
    )

    reference_np = None if case.reference is None else np.asarray(case.reference)
    sart_np = np.asarray(sart_reconstruction)
    dart_np = np.asarray(dart_reconstruction)

    print("\nResults")
    print_summary("SART", sart_np, reference_np)
    print_summary("DART", dart_np, reference_np)

    sinogram_np = np.asarray(case.sinogram)
    mid_view = sinogram_np.shape[0] // 2
    mid_slice = case.volume_shape[0] // 2

    panels: list[tuple[str, np.ndarray, str]] = [
        ("Measured mid-view", sinogram_np[mid_view].T, "gray"),
        ("SART mid-slice", sart_np[mid_slice], "gray"),
        ("DART mid-slice", dart_np[mid_slice], "gray"),
    ]
    if reference_np is not None:
        panels.extend(
            [
                ("Reference mid-slice", reference_np[mid_slice], "gray"),
                ("|SART - Ref|", np.abs(sart_np[mid_slice] - reference_np[mid_slice]), "magma"),
                ("|DART - Ref|", np.abs(dart_np[mid_slice] - reference_np[mid_slice]), "magma"),
            ]
        )

    figure = plt.figure(figsize=(12, 8 if len(panels) > 3 else 4))
    rows = 2 if len(panels) > 3 else 1
    cols = 3
    for index, (title, image, cmap) in enumerate(panels, start=1):
        axis = figure.add_subplot(rows, cols, index)
        if title == "Measured mid-view":
            axis.imshow(image, cmap=cmap, origin="lower")
        elif cmap == "gray":
            axis.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
        else:
            axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")

    figure.tight_layout()
    output_path = Path(__file__).with_name("dart_measured_cone.png")
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
