"""Iterative Reconstruction — Non-Circular Parallel Beam (MLX / Apple Silicon).

Demonstrates gradient-based iterative reconstruction using non-standard
parallel beam trajectories (sinusoidal + custom wobble).  Optimises a
learnable image to match the measured sinogram via MSE loss and AdamW.
"""

import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt
import diffct_mlx


# ── Phantom ──────────────────────────────────────────────────────────────────

def shepp_logan_2d(shape):
    """Generate a 2D Shepp-Logan phantom (numpy)."""
    Ny, Nx = shape
    yy, xx = np.mgrid[:Ny, :Nx]
    xx = (xx - (Nx - 1) / 2) / ((Nx - 1) / 2)
    yy = (yy - (Ny - 1) / 2) / ((Ny - 1) / 2)

    el_params = np.array([
        [0, 0, 0.69, 0.92, 0, 1],
        [0, -0.0184, 0.6624, 0.874, 0, -0.8],
        [0.22, 0, 0.11, 0.31, -np.pi / 10, -0.2],
        [-0.22, 0, 0.16, 0.41, np.pi / 10, -0.2],
        [0, 0.35, 0.21, 0.25, 0, 0.1],
        [0, 0.1, 0.046, 0.046, 0, 0.1],
        [0, -0.1, 0.046, 0.046, 0, 0.1],
        [-0.08, -0.605, 0.046, 0.023, 0, 0.1],
        [0, -0.605, 0.023, 0.023, 0, 0.1],
        [0.06, -0.605, 0.023, 0.046, 0, 0.1],
    ], dtype=np.float32)

    x0 = el_params[:, 0][:, None, None]
    y0 = el_params[:, 1][:, None, None]
    a = el_params[:, 2][:, None, None]
    b = el_params[:, 3][:, None, None]
    phi = el_params[:, 4][:, None, None]
    val = el_params[:, 5][:, None, None]

    c, s = np.cos(phi), np.sin(phi)
    xc, yc = xx[None] - x0, yy[None] - y0
    xp = c * xc - s * yc
    yp = s * xc + c * yc
    mask = (xp ** 2 / a ** 2 + yp ** 2 / b ** 2) <= 1.0
    return np.clip(np.sum(mask * val, axis=0), 0, 1).astype(np.float32)


# ── Custom trajectory functions ──────────────────────────────────────────────

def custom_wobble_rays(angles):
    """Wobbling ray directions."""
    wobble = 0.1 * mx.sin(3 * angles)
    adj = angles + wobble
    return mx.stack([mx.cos(adj), mx.sin(adj)], axis=1)


def custom_wobble_origins(angles):
    """Wobbling detector origins."""
    offset = 30.0 * mx.sin(2 * angles)
    return mx.stack([-offset * mx.sin(angles), offset * mx.cos(angles)], axis=1)


# ── Iterative reconstruction ────────────────────────────────────────────────

def run_reconstruction(trajectory_name, ray_dir, det_origin, det_u_vec,
                       phantom, n_det, det_spacing, voxel_spacing, epochs=500):
    """Run gradient-based iterative reconstruction."""
    print(f"\n{'=' * 60}")
    print(f"Processing {trajectory_name} Trajectory")
    print(f"{'=' * 60}")

    Ny, Nx = phantom.shape

    # Generate ground-truth sinogram
    print("Generating sinogram...")
    target_sino = diffct_mlx.parallel_forward(
        phantom, ray_dir, det_origin, det_u_vec,
        n_det, det_spacing, voxel_spacing,
    )
    mx.eval(target_sino)

    # Learnable reconstruction — small random init so gradients flow
    reco = 0.01 * mx.random.normal((Ny, Nx)).astype(mx.float32)
    lr = 1e-1

    def loss_fn(reco_val):
        current_sino = diffct_mlx.parallel_forward(
            reco_val,
            ray_dir, det_origin, det_u_vec,
            n_det, det_spacing, voxel_spacing,
        )
        return mx.mean((current_sino - target_sino) ** 2)

    loss_and_grad = mx.value_and_grad(loss_fn)
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
    phantom_np = shepp_logan_2d((Ny, Nx))
    phantom = mx.array(phantom_np)

    num_views = 180
    n_det = 256
    det_spacing = 1.0
    voxel_spacing = 1.0
    epochs = 1000

    results = {}

    # 1. Sinusoidal trajectory
    print("\nGenerating Sinusoidal Trajectory...")
    ray_dir, det_origin, det_u_vec = diffct_mlx.sinusoidal_trajectory_2d_parallel(
        num_views, amplitude=50.0, frequency=3.0,
    )
    lv, reco = run_reconstruction(
        "Sinusoidal", ray_dir, det_origin, det_u_vec,
        phantom, n_det, det_spacing, voxel_spacing, epochs,
    )
    results["Sinusoidal"] = (lv, reco)

    # 2. Custom wobble trajectory
    print("\nGenerating Wobble (Custom) Trajectory...")
    ray_dir, det_origin, det_u_vec = diffct_mlx.custom_trajectory_2d_parallel(
        num_views, ray_dir_fn=custom_wobble_rays,
        det_origin_fn=custom_wobble_origins,
    )
    lv, reco = run_reconstruction(
        "Wobble", ray_dir, det_origin, det_u_vec,
        phantom, n_det, det_spacing, voxel_spacing, epochs,
    )
    results["Wobble"] = (lv, reco)

    # ── Plot ─────────────────────────────────────────────────────────────────
    names = list(results.keys())
    fig = plt.figure(figsize=(15, 10))

    for idx, name in enumerate(names):
        lv, reco = results[name]
        # Loss curve
        plt.subplot(3, 2, idx + 1)
        plt.plot(lv)
        plt.title(f"{name} — Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.yscale("log")
        plt.grid(True)
        # Original
        plt.subplot(3, 2, idx + 3)
        plt.imshow(phantom_np, cmap="gray", vmin=0, vmax=1)
        plt.title("Original" if idx == 0 else "")
        plt.axis("off")
        # Reconstruction
        plt.subplot(3, 2, idx + 5)
        mse = float(np.mean((reco - phantom_np) ** 2))
        psnr = 10 * np.log10(1.0 / mse)
        plt.imshow(reco, cmap="gray", vmin=0, vmax=1)
        plt.title(f"{name}\nPSNR: {psnr:.2f} dB")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("iterative_parallel.png", dpi=200, bbox_inches="tight")
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
