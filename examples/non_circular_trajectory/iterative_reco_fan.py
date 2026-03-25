"""Iterative Reconstruction — Non-Circular Fan Beam (MLX / Apple Silicon).

Demonstrates gradient-based iterative reconstruction using non-standard
fan beam trajectories (sinusoidal + custom elliptical).  Optimises a
learnable image to match the measured sinogram via MSE loss.
"""

import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt
import diffct_mlx


# ── Custom trajectory ────────────────────────────────────────────────────────

def custom_ellipse_trajectory(angles, sid):
    """Elliptical source path — wider in x than y."""
    src_x = -sid * 1.5 * mx.sin(angles)
    src_y = sid * mx.cos(angles)
    return mx.stack([src_x, src_y], axis=1)


# ── Iterative reconstruction ────────────────────────────────────────────────

def run_reconstruction(trajectory_name, src_pos, det_center, det_u_vec,
                       phantom, n_det, det_spacing, voxel_spacing, epochs=500):
    """Run gradient-based iterative reconstruction."""
    print(f"\n{'=' * 60}")
    print(f"Processing {trajectory_name} Trajectory")
    print(f"{'=' * 60}")

    Ny, Nx = phantom.shape

    # Generate ground-truth sinogram
    print("Generating sinogram...")
    target_sino = diffct_mlx.fan_forward(
        phantom, src_pos, det_center, det_u_vec,
        n_det, det_spacing, voxel_spacing,
    )
    mx.eval(target_sino)

    # Learnable reconstruction — small random init so gradients flow
    reco = 0.01 * mx.random.normal((Ny, Nx)).astype(mx.float32)

    def loss_fn(reco_val):
        current_sino = diffct_mlx.fan_forward(
            reco_val,
            src_pos, det_center, det_u_vec,
            n_det, det_spacing, voxel_spacing,
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
        if epoch % 100 == 0:
            print(f"  Epoch {epoch:4d}  Loss: {loss_val.item():.6f}")

    reco_np = np.array(mx.maximum(reco, 0.0))
    phantom_np = np.array(phantom)
    mse = float(np.mean((reco_np - phantom_np) ** 2))
    psnr = 10 * np.log10(1.0 / mse)
    print(f"\n  {trajectory_name} — MSE: {mse:.6f}  PSNR: {psnr:.2f} dB")

    return loss_values, reco_np


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    Nx, Ny = 128, 128
    phantom_np = diffct_mlx.shepp_logan_2d((Ny, Nx))
    phantom = mx.array(phantom_np)

    num_views = 360
    n_det = 256
    det_spacing = 1.0
    voxel_spacing = 1.0
    sdd = 600.0
    sid = 400.0
    epochs = 1000

    results = {}

    # 1. Sinusoidal trajectory
    print("\nGenerating Sinusoidal Trajectory...")
    src_pos, det_center, det_u_vec = diffct_mlx.sinusoidal_trajectory_2d_fan(
        num_views, sid, sdd, amplitude=50.0, frequency=3.0,
    )
    lv, reco = run_reconstruction(
        "Sinusoidal", src_pos, det_center, det_u_vec,
        phantom, n_det, det_spacing, voxel_spacing, epochs,
    )
    results["Sinusoidal"] = (lv, reco)

    # 2. Elliptical (custom) trajectory
    print("\nGenerating Elliptical (Custom) Trajectory...")
    src_pos, det_center, det_u_vec = diffct_mlx.custom_trajectory_2d_fan(
        num_views, sid, sdd, source_path_fn=custom_ellipse_trajectory,
    )
    lv, reco = run_reconstruction(
        "Elliptical", src_pos, det_center, det_u_vec,
        phantom, n_det, det_spacing, voxel_spacing, epochs,
    )
    results["Elliptical"] = (lv, reco)

    # ── Plot ─────────────────────────────────────────────────────────────────
    names = list(results.keys())
    fig = plt.figure(figsize=(15, 10))

    for idx, name in enumerate(names):
        lv, reco = results[name]
        plt.subplot(3, 2, idx + 1)
        plt.plot(lv)
        plt.title(f"{name} — Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.yscale("log")
        plt.grid(True)

        plt.subplot(3, 2, idx + 3)
        plt.imshow(phantom_np, cmap="gray", vmin=0, vmax=1)
        plt.title("Original" if idx == 0 else "")
        plt.axis("off")

        plt.subplot(3, 2, idx + 5)
        mse = float(np.mean((reco - phantom_np) ** 2))
        psnr = 10 * np.log10(1.0 / mse)
        plt.imshow(reco, cmap="gray", vmin=0, vmax=1)
        plt.title(f"{name}\nPSNR: {psnr:.2f} dB")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("iterative_fan.png", dpi=200, bbox_inches="tight")
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
