"""Run DART on a discrete 3D cone-beam phantom."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

import diffct_mlx


def quantize_volume(volume: np.ndarray, gray_levels: np.ndarray) -> np.ndarray:
    """Quantize a continuous phantom onto a discrete set of gray levels."""
    thresholds = 0.5 * (gray_levels[:-1] + gray_levels[1:])
    indices = np.digitize(volume, thresholds, right=False)
    return gray_levels[indices]


def compute_metrics(reference: np.ndarray, reconstruction: np.ndarray) -> tuple[float, float]:
    """Return MSE and PSNR against the discrete reference volume."""
    mse = float(np.mean((reconstruction - reference) ** 2))
    dynamic_range = float(reference.max() - reference.min())
    if dynamic_range <= 0.0:
        dynamic_range = 1.0
    psnr = float("inf") if mse <= 0.0 else 10.0 * np.log10((dynamic_range**2) / mse)
    return mse, psnr


def main() -> None:
    volume_shape = (48, 48, 48)
    num_views = 72
    detector_shape = (72, 72)
    du = 1.0
    dv = 1.0
    voxel_spacing = 1.0
    sid = 600.0
    sdd = 900.0
    gray_levels = np.array([0.0, 0.35, 0.7, 1.0], dtype=np.float32)

    continuous_reference = np.asarray(diffct_mlx.shepp_logan_3d(volume_shape), dtype=np.float32)
    reference = quantize_volume(continuous_reference, gray_levels)
    reference_mx = mx.array(reference, dtype=mx.float32)

    det_u_count, det_v_count = detector_shape
    src_pos, det_center, det_u_vec, det_v_vec = diffct_mlx.circular_trajectory_3d(num_views, sid, sdd)
    sinogram = diffct_mlx.cone_forward(
        reference_mx,
        src_pos,
        det_center,
        det_u_vec,
        det_v_vec,
        det_u=det_u_count,
        det_v=det_v_count,
        du=du,
        dv=dv,
        voxel_spacing=voxel_spacing,
    )
    mx.eval(sinogram)
    measured_projections = [sinogram[index] for index in range(int(sinogram.shape[0]))]
    forward_single, back_single, _ = diffct_mlx.make_cone_3d_operators(
        src_pos,
        det_center,
        det_u_vec,
        det_v_vec,
        volume_shape=volume_shape,
        detector_shape=detector_shape,
        du=du,
        dv=dv,
        voxel_spacing=voxel_spacing,
    )

    print("Running SART baseline...")
    sart_reconstruction = diffct_mlx.run_sart(
        measured_projections,
        forward_single,
        back_single,
        diffct_mlx.SARTParameters(
            volume_shape=volume_shape,
            iteration_count=8,
            sart_iteration_count=1,
            voxel_extreme_values=(0.0, 1.0),
            backprojection_scale=0.1,
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
            volume_shape=volume_shape,
            iteration_count=12,
            sart_iteration_count=4,
            initial_reconstruction_sweeps=8,
            gray_levels=tuple(float(level) for level in gray_levels),
            free_pixel_probability=0.08,
            voxel_extreme_values=(0.0, 1.0),
            backprojection_scale=0.1,
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

    mid_slice = volume_shape[0] // 2
    mid_view = num_views // 2
    figure = plt.figure(figsize=(12, 8))
    panels = [
        ("Reference mid-slice", reference_np[mid_slice]),
        (f"SART mid-slice\nPSNR: {sart_psnr:.2f} dB", sart_np[mid_slice]),
        (f"DART mid-slice\nPSNR: {dart_psnr:.2f} dB", dart_np[mid_slice]),
        ("Measured mid-view", np.asarray(sinogram[mid_view]).T),
        ("|SART - Ref|", np.abs(sart_np[mid_slice] - reference_np[mid_slice])),
        ("|DART - Ref|", np.abs(dart_np[mid_slice] - reference_np[mid_slice])),
    ]
    for index, (title, image) in enumerate(panels, start=1):
        axis = figure.add_subplot(2, 3, index)
        if index == 4:
            axis.imshow(image, cmap="gray", origin="lower")
        elif index >= 5:
            axis.imshow(image, cmap="magma")
        else:
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")

    figure.tight_layout()
    output_path = Path(__file__).with_name("dart_cone.png")
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
