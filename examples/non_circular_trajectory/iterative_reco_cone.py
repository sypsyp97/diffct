"""Iterative Reconstruction — Non-Circular Cone Beam (MLX / Apple Silicon).

Demonstrates gradient-based iterative reconstruction using non-standard
3D cone beam trajectories (spiral, sinusoidal, saddle, custom figure-8).
Optimises a non-negative volume with Adam to match the measured sinogram
via MSE loss.
"""

import json
import numpy as np
import mlx.core as mx
import mlx.optimizers as optim
import matplotlib.pyplot as plt
import diffct_mlx
from pathlib import Path
from diffct_mlx.real_measured_data_helper import (
    auto_voxel_spacing_from_detector,
    load_tiff_projections,
    normalize_volume,
    resize_volume_to_shape,
    shift_detector_center,
    transform_detector_offsets,
)

try:
    _load_arbitrary_cone_geometry_from_json = diffct_mlx.load_arbitrary_cone_geometry_from_json
except AttributeError:
    from diffct_mlx.geometry import load_arbitrary_cone_geometry_from_json as _load_arbitrary_cone_geometry_from_json




# ── 3D Phantom ───────────────────────────────────────────────────────────────

def shepp_logan_3d(shape):
    """Generate a 3D Shepp-Logan phantom (numpy)."""
    zz, yy, xx = np.mgrid[:shape[0], :shape[1], :shape[2]]
    xx = (xx - (shape[2] - 1) / 2) / ((shape[2] - 1) / 2)
    yy = (yy - (shape[1] - 1) / 2) / ((shape[1] - 1) / 2)
    zz = (zz - (shape[0] - 1) / 2) / ((shape[0] - 1) / 2)

    el_params = np.array([
        [0, 0, 0, 0.69, 0.92, 0.81, 0, 1],
        [0, -0.0184, 0, 0.6624, 0.874, 0.78, 0, -0.8],
        [0.22, 0, 0, 0.11, 0.31, 0.22, -np.pi / 10, -0.2],
        [-0.22, 0, 0, 0.16, 0.41, 0.28, np.pi / 10, -0.2],
        [0, 0.35, -0.15, 0.21, 0.25, 0.41, 0, 0.1],
        [0, 0.1, 0.25, 0.046, 0.046, 0.05, 0, 0.1],
        [0, -0.1, 0.25, 0.046, 0.046, 0.05, 0, 0.1],
        [-0.08, -0.605, 0, 0.046, 0.023, 0.05, 0, 0.1],
        [0, -0.605, 0, 0.023, 0.023, 0.02, 0, 0.1],
        [0.06, -0.605, 0, 0.023, 0.046, 0.02, 0, 0.1],
    ], dtype=np.float32)

    x0 = el_params[:, 0][:, None, None, None]
    y0 = el_params[:, 1][:, None, None, None]
    z0 = el_params[:, 2][:, None, None, None]
    a = el_params[:, 3][:, None, None, None]
    b = el_params[:, 4][:, None, None, None]
    c = el_params[:, 5][:, None, None, None]
    phi = el_params[:, 6][:, None, None, None]
    val = el_params[:, 7][:, None, None, None]

    cos_p, sin_p = np.cos(phi), np.sin(phi)
    xc = xx[None] - x0
    yc = yy[None] - y0
    zc = zz[None] - z0
    xp = cos_p * xc - sin_p * yc
    yp = sin_p * xc + cos_p * yc

    mask = (xp ** 2 / a ** 2 + yp ** 2 / b ** 2 + zc ** 2 / c ** 2) <= 1.0
    vol = np.clip(np.sum(mask * val, axis=0), 0, 1).astype(np.float32)
    return vol


# ── Custom trajectory ────────────────────────────────────────────────────────

def custom_figure8_trajectory(angles, sid):
    """Figure-8-like source path that stays away from the isocenter."""
    src_x = -sid * mx.sin(angles)
    src_y = sid * mx.cos(angles) + 0.15 * sid * mx.sin(2 * angles)
    src_z = 50.0 * mx.sin(2 * angles)
    return mx.stack([src_x, src_y, src_z], axis=1)


# ── Iterative reconstruction ────────────────────────────────────────────────

def run_reconstruction(trajectory_name,
                       src_pos, det_center, det_u_vec, det_v_vec,
                       phantom, det_u, det_v, du, dv,
                       voxel_spacing, epochs=100, lr=1e-1,
                       target_sino=None, reference_volume=None):
    """Run gradient-based iterative reconstruction for a cone beam trajectory."""
    print(f"\n{'=' * 60}")
    print(f"Processing {trajectory_name} Trajectory")
    print(f"{'=' * 60}")

    Nz, Ny, Nx = phantom.shape

    if target_sino is None:
        print("Generating sinogram...")
        target_sino = diffct_mlx.cone_forward(
            phantom, src_pos, det_center, det_u_vec, det_v_vec,
            det_u, det_v, du, dv, voxel_spacing,
        )
    else:
        print("Using loaded measured sinogram...")
        target_sino = mx.array(target_sino, dtype=mx.float32)
    mx.eval(target_sino)

    # Zero init is sufficient here because the cone projector is linear.
    params = {"reco": mx.zeros((Nz, Ny, Nx), dtype=mx.float32)}
    optimizer = optim.Adam(learning_rate=lr, bias_correction=True)

    def loss_fn(reco_val):
        current_sino = diffct_mlx.cone_forward(
            reco_val,
            src_pos, det_center, det_u_vec, det_v_vec,
            det_u, det_v, du, dv, voxel_spacing,
        )
        return mx.mean((current_sino - target_sino) ** 2)

    loss_and_grad = mx.value_and_grad(loss_fn)
    loss_values = []
    reference_np = None if reference_volume is None else np.asarray(reference_volume)

    print("Starting iterative reconstruction...")
    for epoch in range(epochs):
        loss_val, grad = loss_and_grad(params["reco"])
        mx.eval(loss_val, grad)

        params = optimizer.apply_gradients({"reco": grad}, params)
        params["reco"] = mx.maximum(params["reco"], 0.0)
        mx.eval(params["reco"])

        loss_values.append(float(loss_val))
        if (epoch % 10 == 0 or epoch == epochs - 1) and reference_np is not None:
            reco_np = np.array(params["reco"])
            mse = float(np.mean((reco_np - reference_np) ** 2))
            psnr = 10 * np.log10(1.0 / mse)
            print(f"  Epoch {epoch:4d}  Loss: {loss_val.item():.6f}  PSNR: {psnr:.2f} dB")
        elif epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:4d}  Loss: {loss_val.item():.6f}")

    reco_np = np.array(params["reco"])
    metrics = None
    if reference_np is not None:
        mse = float(np.mean((reco_np - reference_np) ** 2))
        dynamic_range = float(np.max(reference_np) - np.min(reference_np))
        if dynamic_range <= 0.0:
            dynamic_range = 1.0
        psnr = float("inf") if mse <= 0.0 else 10 * np.log10((dynamic_range ** 2) / mse)
        print(f"\n  {trajectory_name} — MSE: {mse:.6f}  PSNR: {psnr:.2f} dB")
        metrics = {"mse": mse, "psnr": psnr}
    else:
        print(f"\n  {trajectory_name} — final loss: {loss_values[-1]:.6f}")

    return loss_values, reco_np, metrics


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    Nx, Ny, Nz = 64, 64, 64
    phantom_np = shepp_logan_3d((Nz, Ny, Nx))
    phantom = mx.array(phantom_np)

    num_views = 400
    det_u, det_v = 128, 128
    du, dv = 1.0, 1.0
    voxel_spacing = 1.0
    sdd = 600.0
    sid = 400.0
    epochs = 100
    measured_fov_margin_mm = 8.0
    mid = Nz // 2

    results = {}

    # # 1. Spiral trajectory
    # print("\nGenerating Spiral Trajectory...")
    # s, dc, du_v, dv_v = diffct_mlx.spiral_trajectory_3d(
    #     num_views, sid, sdd, z_range=80.0, n_turns=2.0,
    # )
    # lv, reco, metrics = run_reconstruction(
    #     "Spiral", s, dc, du_v, dv_v,
    #     phantom, det_u, det_v, du, dv, voxel_spacing, epochs,
    #     reference_volume=phantom_np,
    # )
    # results["Spiral"] = {
    #     "loss": lv,
    #     "reco": reco,
    #     "reference": phantom_np[mid],
    #     "reference_title": "Original",
    #     "metrics": metrics,
    # }

    # # 2. Sinusoidal trajectory
    # print("\nGenerating Sinusoidal Trajectory...")
    # s, dc, du_v, dv_v = diffct_mlx.sinusoidal_trajectory_3d(
    #     num_views, sid, sdd, amplitude=50.0, frequency=3.0,
    # )
    # lv, reco, metrics = run_reconstruction(
    #     "Sinusoidal", s, dc, du_v, dv_v,
    #     phantom, det_u, det_v, du, dv, voxel_spacing, epochs,
    #     reference_volume=phantom_np,
    # )
    # results["Sinusoidal"] = {
    #     "loss": lv,
    #     "reco": reco,
    #     "reference": phantom_np[mid],
    #     "reference_title": "Original",
    #     "metrics": metrics,
    # }

    # # 3. Saddle trajectory
    # print("\nGenerating Saddle Trajectory...")
    # s, dc, du_v, dv_v = diffct_mlx.saddle_trajectory_3d(
    #     num_views, sid, sdd, z_amplitude=60.0, radial_amplitude=40.0,
    # )
    # lv, reco, metrics = run_reconstruction(
    #     "Saddle", s, dc, du_v, dv_v,
    #     phantom, det_u, det_v, du, dv, voxel_spacing, epochs,
    #     reference_volume=phantom_np,
    # )
    # results["Saddle"] = {
    #     "loss": lv,
    #     "reco": reco,
    #     "reference": phantom_np[mid],
    #     "reference_title": "Original",
    #     "metrics": metrics,
    # }

    # # 4. Custom figure-8 trajectory
    # print("\nGenerating Custom (Figure-8) Trajectory...")
    # s, dc, du_v, dv_v = diffct_mlx.custom_trajectory_3d(
    #     num_views, sid, sdd, source_path_fn=custom_figure8_trajectory,
    # )
    # lv, reco, metrics = run_reconstruction(
    #     "Figure-8", s, dc, du_v, dv_v,
    #     phantom, det_u, det_v, du, dv, voxel_spacing, epochs,
    #     reference_volume=phantom_np,
    # )
    # results["Figure-8"] = {
    #     "loss": lv,
    #     "reco": reco,
    #     "reference": phantom_np[mid],
    #     "reference_title": "Original",
    #     "metrics": metrics,
    # }

    # 5. Real measured arbitrary trajectory
    print("\nLoading measured TIFF data and arbitrary geometry...")
    real_data_dir = Path(__file__).parent / "sample_data" / "sim_obj_1_tif"
    trajectory_json_path = real_data_dir / "sim_obj_1_geometry_diffct.json"
    reference_volume_path = Path(__file__).parent / "sample_data" / "reko" / "sim_obj_1_diffct.npy"
    reference_meta_path = Path(__file__).parent / "sample_data" / "reko" / "sim_obj_1_diffct.json"
    reference_meta = None
    resized_reference_voxel_spacing = None
    if reference_meta_path.exists():
        reference_meta = json.loads(reference_meta_path.read_text())
        source_shape = tuple(int(x) for x in reference_meta["shape_zyx"])
        target_shape = (Nz, Ny, Nx)
        resize_factors = tuple(src / dst for src, dst in zip(source_shape, target_shape))
        if max(resize_factors) - min(resize_factors) > 1e-6:
            raise ValueError(
                f"Reference volume resize must stay isotropic, got factors {resize_factors}"
            )
        resized_reference_voxel_spacing = float(reference_meta["voxel_size_mm"]) * resize_factors[0]
        reference_volume_np = resize_volume_to_shape(np.load(reference_volume_path), (Nz, Ny, Nx))
        reference_volume_np = normalize_volume(reference_volume_np, new_min=0.0, new_max=1.0)
    else:
        reference_volume_np = None
     
    geometry_convention_config = {
    "transpose_uv": True,
    "flip_u": False,
    "flip_v": False,
    }
    with trajectory_json_path.open("r", encoding="utf-8") as f:
        geometry_payload = json.load(f)
    s, dc, du_v, dv_v = _load_arbitrary_cone_geometry_from_json(
        trajectory_json_path,
        flip_det_u=False,
        flip_det_v=False,
        recenter_to_isocenter=True,
    )
    view_stride = max(1, int(np.ceil(s.shape[0] / num_views)))
    det_v_binning = max(1, int(np.ceil(geometry_payload["detector"]["num_pixels"]["v"] / det_v)))
    det_u_binning = max(1, int(np.ceil(geometry_payload["detector"]["num_pixels"]["u"] / det_u)))
    measured_sino_np = load_tiff_projections(
        real_data_dir,
        log_transform=True,
        view_stride=view_stride,
        detector_binning_u=det_u_binning,
        detector_binning_v=det_v_binning,
        debug_visualization=False,
        debug_output_path=real_data_dir / "projection_log_transform_debug.png",
    )
    s = s[::view_stride]
    dc = dc[::view_stride]
    du_v = du_v[::view_stride]
    dv_v = dv_v[::view_stride]
    measured_du = float(geometry_payload["detector"]["pixel_size_mm"]["u"]) * det_u_binning
    measured_dv = float(geometry_payload["detector"]["pixel_size_mm"]["v"]) * det_v_binning
    header_offset_u_px = float(geometry_payload["detector"]["offset_px"].get("horizontal", 0.0) or 0.0) / det_u_binning
    header_offset_v_px = float(geometry_payload["detector"]["offset_px"].get("vertical", 0.0) or 0.0) / det_v_binning
    measured_det_v = measured_sino_np.shape[1]
    measured_det_u = measured_sino_np.shape[2]
    if resized_reference_voxel_spacing is not None:
        measured_voxel_spacing = float(resized_reference_voxel_spacing)
    else:
        measured_voxel_spacing = auto_voxel_spacing_from_detector(
            (Nz, Ny, Nx),
            (measured_det_u, measured_det_v),
            measured_du,
            measured_dv,
            magnification=float(geometry_payload["source"]["magnification"]),
            fov_margin_mm=measured_fov_margin_mm,
        )
    # The TIFF projections are read with a flipped detector convention, so both
    # the forward and backward projector must use the transposed detector geometry.
    du_v, dv_v, measured_du, measured_dv, measured_det_u, measured_det_v = diffct_mlx.apply_detector_geometry_convention(
        du_v,
        dv_v,
        du=measured_du,
        dv=measured_dv,
        det_u=measured_det_u,
        det_v=measured_det_v,
        flip_u=geometry_convention_config["flip_u"],
        flip_v=geometry_convention_config["flip_v"],
        transpose_uv=geometry_convention_config["transpose_uv"],
    )
    header_offset_u_px, header_offset_v_px = transform_detector_offsets(
        header_offset_u_px,
        header_offset_v_px,
        geometry_convention_config,
    )
    dc = shift_detector_center(
        dc,
        du_v,
        dv_v,
        measured_du,
        measured_dv,
        offset_u_px=header_offset_u_px,
        offset_v_px=header_offset_v_px,
    )
    if resized_reference_voxel_spacing is not None:
        measured_voxel_spacing = float(resized_reference_voxel_spacing)
    else:
        measured_voxel_spacing = auto_voxel_spacing_from_detector(
            (Nz, Ny, Nx),
            (measured_det_u, measured_det_v),
            measured_du,
            measured_dv,
            magnification=float(geometry_payload["source"]["magnification"]),
            fov_margin_mm=measured_fov_margin_mm,
        )

    lv, reco, metrics = run_reconstruction(
        "Measured Arbitrary", s, dc, du_v, dv_v,
        phantom, measured_det_u, measured_det_v, measured_du, measured_dv,
        measured_voxel_spacing, epochs,
        target_sino=measured_sino_np,
        reference_volume=reference_volume_np,
    )

    if reference_volume_np is not None:
        results["Measured Arbitrary"] = {
            "loss": lv,
            "reco": reco,
            "reference": reference_volume_np[reference_volume_np.shape[0] // 2],
            "reference_title": "Reference Volume",
            "metrics": metrics,
        }
    else:
        results["Measured Arbitrary"] = {
            "loss": lv,
            "reco": reco,
            "reference": None,
            "reference_title": None,
            "metrics": metrics,
        }

    # Optional: inspect whether the loaded detector basis and central rays are
    # internally consistent after the isocenter recentering step.
    # diag = diffct_mlx.diagnose_cone_geometry(s, dc, du_v, dv_v)
    # print(
    #     "Loaded geometry stats: "
    #     f"SID {diag['sid_min_mm']:.1f}..{diag['sid_max_mm']:.1f} mm, "
    #     f"SDD {diag['sdd_min_mm']:.1f}..{diag['sdd_max_mm']:.1f} mm, "
    #     f"max|u·v|={diag['det_u_dot_det_v_max_abs']:.4f}, "
    #     f"max|v·ray|={diag['det_v_dot_ray_max_abs']:.4f}, "
    #     f"isocenter=({diag['estimated_isocenter_x_mm']:.1f}, "
    #     f"{diag['estimated_isocenter_y_mm']:.1f}, "
    #     f"{diag['estimated_isocenter_z_mm']:.1f}) mm"
    # )


    # ── Plot ─────────────────────────────────────────────────────────────────
    names = list(results.keys())
    n = len(names)
    fig = plt.figure(figsize=(5 * n, 12))

    for idx, name in enumerate(names):
        result = results[name]
        lv = result["loss"]
        reco = result["reco"]
        # Loss
        plt.subplot(3, n, idx + 1)
        plt.plot(lv)
        plt.title(f"{name} — Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.yscale("log")
        plt.grid(True)
        # Original
        if result["reference"] is not None:
            plt.subplot(3, n, n + idx + 1)
            plt.imshow(result["reference"], cmap="gray")
            plt.title(result["reference_title"])
            plt.axis("off")
        # Reconstruction
        plt.subplot(3, n, 2 * n + idx + 1)
        plt.imshow(reco[mid], cmap="gray", vmin=0, vmax=1)
        metric_label = ""
        if result["metrics"] is not None:
            metric_label = f"\nPSNR: {result['metrics']['psnr']:.2f} dB"
        plt.title(f"{name}{metric_label}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("iterative_cone.png", dpi=200, bbox_inches="tight")
    plt.show()

    # Summary
    print(f"\n{'=' * 60}")
    print(f"{'Trajectory':<20} {'MSE':<12} {'PSNR (dB)':<12}")
    print("-" * 60)
    for name in names:
        metrics = results[name]["metrics"]
        if metrics is None:
            print(f"{name:<20} {'n/a':<12} {'n/a':<12}")
            continue
        print(f"{name:<20} {metrics['mse']:<12.6f} {metrics['psnr']:<12.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
