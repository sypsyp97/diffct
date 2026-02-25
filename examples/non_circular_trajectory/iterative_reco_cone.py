"""Iterative Reconstruction — Non-Circular Cone Beam (MLX / Apple Silicon).

Demonstrates gradient-based iterative reconstruction using non-standard
3D cone beam trajectories (spiral, sinusoidal, saddle, custom figure-8).
Optimises a learnable volume to match the measured sinogram via MSE loss.
"""

import math
import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt
import diffct_mlx


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
    """Figure-8 source path in 3D."""
    src_x = -sid * mx.sin(angles)
    src_y = sid * mx.cos(angles) * mx.sin(angles)       # figure-8 in y
    src_z = 50.0 * mx.sin(2 * angles)                   # z oscillation
    return mx.stack([src_x, src_y, src_z], axis=1)


# ── Iterative reconstruction ────────────────────────────────────────────────

def run_reconstruction(trajectory_name,
                       src_pos, det_center, det_u_vec, det_v_vec,
                       phantom, det_u, det_v, du, dv,
                       voxel_spacing, epochs=500):
    """Run gradient-based iterative reconstruction for a cone beam trajectory."""
    print(f"\n{'=' * 60}")
    print(f"Processing {trajectory_name} Trajectory")
    print(f"{'=' * 60}")

    Nz, Ny, Nx = phantom.shape

    # Ground-truth sinogram
    print("Generating sinogram...")
    target_sino = diffct_mlx.cone_forward(
        phantom, src_pos, det_center, det_u_vec, det_v_vec,
        det_u, det_v, du, dv, voxel_spacing,
    )
    mx.eval(target_sino)

    # Learnable volume
    reco = mx.zeros((Nz, Ny, Nx), dtype=mx.float32)

    def loss_fn(reco_val):
        current_sino = diffct_mlx.cone_forward(
            mx.maximum(reco_val, 0.0),
            src_pos, det_center, det_u_vec, det_v_vec,
            det_u, det_v, du, dv, voxel_spacing,
        )
        return mx.mean((current_sino - target_sino) ** 2)

    loss_and_grad = mx.value_and_grad(loss_fn)
    lr = 1e-1
    loss_values = []

    print("Starting iterative reconstruction...")
    for epoch in range(epochs):
        loss_val, grad = loss_and_grad(reco)
        mx.eval(loss_val, grad)

        reco = reco - lr * grad
        mx.eval(reco)

        loss_values.append(float(loss_val))
        if epoch % 50 == 0:
            print(f"  Epoch {epoch:4d}  Loss: {loss_val.item():.6f}")

    reco_np = np.array(mx.maximum(reco, 0.0))
    phantom_np = np.array(phantom)
    mse = float(np.mean((reco_np - phantom_np) ** 2))
    psnr = 10 * np.log10(1.0 / mse)
    print(f"\n  {trajectory_name} — MSE: {mse:.6f}  PSNR: {psnr:.2f} dB")

    return loss_values, reco_np


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    Nx, Ny, Nz = 64, 64, 64
    phantom_np = shepp_logan_3d((Nz, Ny, Nx))
    phantom = mx.array(phantom_np)

    num_views = 180
    det_u, det_v = 128, 128
    du, dv = 1.0, 1.0
    voxel_spacing = 1.0
    sdd = 600.0
    sid = 400.0
    epochs = 500

    results = {}

    # 1. Spiral trajectory
    print("\nGenerating Spiral Trajectory...")
    s, dc, du_v, dv_v = diffct_mlx.spiral_trajectory_3d(
        num_views, sid, sdd, z_range=80.0, n_turns=2.0,
    )
    lv, reco = run_reconstruction(
        "Spiral", s, dc, du_v, dv_v,
        phantom, det_u, det_v, du, dv, voxel_spacing, epochs,
    )
    results["Spiral"] = (lv, reco)

    # 2. Sinusoidal trajectory
    print("\nGenerating Sinusoidal Trajectory...")
    s, dc, du_v, dv_v = diffct_mlx.sinusoidal_trajectory_3d(
        num_views, sid, sdd, amplitude=50.0, frequency=3.0,
    )
    lv, reco = run_reconstruction(
        "Sinusoidal", s, dc, du_v, dv_v,
        phantom, det_u, det_v, du, dv, voxel_spacing, epochs,
    )
    results["Sinusoidal"] = (lv, reco)

    # 3. Saddle trajectory
    print("\nGenerating Saddle Trajectory...")
    s, dc, du_v, dv_v = diffct_mlx.saddle_trajectory_3d(
        num_views, sid, sdd, z_amplitude=60.0, radial_amplitude=40.0,
    )
    lv, reco = run_reconstruction(
        "Saddle", s, dc, du_v, dv_v,
        phantom, det_u, det_v, du, dv, voxel_spacing, epochs,
    )
    results["Saddle"] = (lv, reco)

    # 4. Custom figure-8 trajectory
    print("\nGenerating Custom (Figure-8) Trajectory...")
    s, dc, du_v, dv_v = diffct_mlx.custom_trajectory_3d(
        num_views, sid, sdd, source_path_fn=custom_figure8_trajectory,
    )
    lv, reco = run_reconstruction(
        "Figure-8", s, dc, du_v, dv_v,
        phantom, det_u, det_v, du, dv, voxel_spacing, epochs,
    )
    results["Figure-8"] = (lv, reco)

    # ── Plot ─────────────────────────────────────────────────────────────────
    mid = Nz // 2
    names = list(results.keys())
    n = len(names)
    fig = plt.figure(figsize=(5 * n, 12))

    for idx, name in enumerate(names):
        lv, reco = results[name]
        # Loss
        plt.subplot(3, n, idx + 1)
        plt.plot(lv)
        plt.title(f"{name} — Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.yscale("log")
        plt.grid(True)
        # Original
        plt.subplot(3, n, n + idx + 1)
        plt.imshow(phantom_np[mid], cmap="gray", vmin=0, vmax=1)
        if idx == 0:
            plt.title("Original")
        plt.axis("off")
        # Reconstruction
        plt.subplot(3, n, 2 * n + idx + 1)
        mse = float(np.mean((reco - phantom_np) ** 2))
        psnr = 10 * np.log10(1.0 / mse)
        plt.imshow(reco[mid], cmap="gray", vmin=0, vmax=1)
        plt.title(f"{name}\nPSNR: {psnr:.2f} dB")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("iterative_cone.png", dpi=200, bbox_inches="tight")
    plt.show()

    # Summary
    print(f"\n{'=' * 60}")
    print(f"{'Trajectory':<20} {'MSE':<12} {'PSNR (dB)':<12}")
    print("-" * 60)
    for name in names:
        _, reco = results[name]
        mse = float(np.mean((reco - phantom_np) ** 2))
        psnr = 10 * np.log10(1.0 / mse)
        print(f"{name:<20} {mse:<12.6f} {psnr:<12.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
