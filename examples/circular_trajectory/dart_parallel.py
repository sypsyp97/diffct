"""Run DART on a discrete 2D parallel-beam phantom."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

import diffct_mlx


def quantize_image(image: np.ndarray, gray_levels: np.ndarray) -> np.ndarray:
    """Quantize a continuous phantom onto a discrete set of gray levels."""
    thresholds = 0.5 * (gray_levels[:-1] + gray_levels[1:])
    indices = np.digitize(image, thresholds, right=False)
    return gray_levels[indices]


def compute_metrics(reference: np.ndarray, reconstruction: np.ndarray) -> tuple[float, float]:
    """Return MSE and PSNR against the discrete reference image."""
    mse = float(np.mean((reconstruction - reference) ** 2))
    dynamic_range = float(reference.max() - reference.min())
    if dynamic_range <= 0.0:
        dynamic_range = 1.0
    psnr = float("inf") if mse <= 0.0 else 10.0 * np.log10((dynamic_range**2) / mse)
    return mse, psnr


def main() -> None:
    image_shape = (96, 96)
    num_views = 120
    num_detectors = 144
    detector_spacing = 1.0
    voxel_spacing = 1.0
    gray_levels = np.array([0.0, 0.35, 0.7, 1.0], dtype=np.float32)

    continuous_reference = np.asarray(diffct_mlx.shepp_logan_2d(image_shape), dtype=np.float32)
    reference = quantize_image(continuous_reference, gray_levels)
    reference_mx = mx.array(reference, dtype=mx.float32)

    ray_dir, det_origin, det_u_vec = diffct_mlx.circular_trajectory_2d_parallel(num_views)
    sinogram = diffct_mlx.parallel_forward(
        reference_mx,
        ray_dir,
        det_origin,
        det_u_vec,
        num_detectors=num_detectors,
        detector_spacing=detector_spacing,
        voxel_spacing=voxel_spacing,
    )
    mx.eval(sinogram)
    measured_projections = [sinogram[index] for index in range(int(sinogram.shape[0]))]
    forward_single, back_single, _ = diffct_mlx.make_parallel_2d_operators(
        ray_dir,
        det_origin,
        det_u_vec,
        image_shape=image_shape,
        num_detectors=num_detectors,
        detector_spacing=detector_spacing,
        voxel_spacing=voxel_spacing,
    )

    print("Running SART baseline...")
    sart_reconstruction = diffct_mlx.run_sart(
        measured_projections,
        forward_single,
        back_single,
        diffct_mlx.SARTParameters(
            volume_shape=image_shape,
            iteration_count=12,
            sart_iteration_count=1,
            voxel_extreme_values=(0.0, 1.0),
            backprojection_scale=0.18,
            shuffle_projection_order=False,
        ),
        show_progress=True,
    )

    print("Running DART...")
    dart_reconstruction = diffct_mlx.run_dart(
        measured_projections,
        forward_single,
        back_single,
        diffct_mlx.DARTParameters(
            volume_shape=image_shape,
            iteration_count=12,
            sart_iteration_count=4,
            initial_reconstruction_sweeps=8,
            gray_levels=tuple(float(level) for level in gray_levels),
            free_pixel_probability=0.12,
            voxel_extreme_values=(0.0, 1.0),
            backprojection_scale=0.18,
            shuffle_projection_order=False,
            random_seed=7,
        ),
        show_progress=True,
    )

    reference_np = np.asarray(reference_mx)
    sart_np = np.asarray(sart_reconstruction)
    dart_np = np.asarray(dart_reconstruction)
    sart_mse, sart_psnr = compute_metrics(reference_np, sart_np)
    dart_mse, dart_psnr = compute_metrics(reference_np, dart_np)

    print("\nResults")
    print(f"SART  mse={sart_mse:.6f}  psnr={sart_psnr:.2f} dB")
    print(f"DART  mse={dart_mse:.6f}  psnr={dart_psnr:.2f} dB")

    figure = plt.figure(figsize=(12, 4))
    panels = [
        ("Reference", reference_np),
        (f"SART\nPSNR: {sart_psnr:.2f} dB", sart_np),
        (f"DART\nPSNR: {dart_psnr:.2f} dB", dart_np),
    ]
    for index, (title, image) in enumerate(panels, start=1):
        axis = figure.add_subplot(1, 3, index)
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")

    figure.tight_layout()
    output_path = Path(__file__).with_name("dart_parallel.png")
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
