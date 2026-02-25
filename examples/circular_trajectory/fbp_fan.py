"""FBP Fan Beam Reconstruction Example (MLX / Apple Silicon).

Demonstrates:
  - 2D Shepp-Logan phantom generation
  - Differentiable forward projection with circular fan beam geometry
  - Cosine-weighted, ramp-filtered backprojection (FBP)
  - Gradient computation through the full pipeline
"""

import math
import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt
import diffct_mlx


# ── Phantom ──────────────────────────────────────────────────────────────────

def shepp_logan_2d(Nx, Ny):
    """Generate a 2D Shepp-Logan phantom (numpy)."""
    phantom = np.zeros((Ny, Nx), dtype=np.float32)
    ellipses = [
        (0.0, 0.0, 0.69, 0.92, 0, 1.0),
        (0.0, -0.0184, 0.6624, 0.8740, 0, -0.8),
        (0.22, 0.0, 0.11, 0.31, -18.0, -0.8),
        (-0.22, 0.0, 0.16, 0.41, 18.0, -0.8),
        (0.0, 0.35, 0.21, 0.25, 0, 0.7),
    ]
    cx, cy = (Nx - 1) / 2, (Ny - 1) / 2
    for ix in range(Nx):
        for iy in range(Ny):
            xn = (ix - cx) / (Nx / 2)
            yn = (iy - cy) / (Ny / 2)
            val = 0.0
            for x0, y0, a, b, angdeg, ampl in ellipses:
                th = np.deg2rad(angdeg)
                xp = (xn - x0) * np.cos(th) + (yn - y0) * np.sin(th)
                yp = -(xn - x0) * np.sin(th) + (yn - y0) * np.cos(th)
                if xp * xp / (a * a) + yp * yp / (b * b) <= 1.0:
                    val += ampl
            phantom[iy, ix] = val
    return np.clip(phantom, 0.0, 1.0)


# ── Ramp filter ──────────────────────────────────────────────────────────────

def ramp_filter(sinogram):
    """Apply ramp filter along detector axis (axis-1) using numpy FFT."""
    sino_np = np.array(sinogram)
    n_det = sino_np.shape[1]
    freqs = np.fft.fftfreq(n_det)
    ramp = np.abs(2.0 * np.pi * freqs).astype(np.float32)
    sino_fft = np.fft.fft(sino_np, axis=1)
    filtered = np.real(np.fft.ifft(sino_fft * ramp[None, :], axis=1)).astype(np.float32)
    return mx.array(filtered)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Parameters
    Nx, Ny = 256, 256
    num_angles = 360
    num_detectors = 600
    detector_spacing = 1.0
    voxel_spacing = 1.0
    sdd = 800.0
    sid = 500.0

    # Phantom
    phantom_np = shepp_logan_2d(Nx, Ny)
    image = mx.array(phantom_np)

    # Geometry
    src_pos, det_center, det_u_vec = diffct_mlx.circular_trajectory_2d_fan(
        num_angles, sid, sdd,
    )

    # Forward projection
    sinogram = diffct_mlx.fan_forward(
        image, src_pos, det_center, det_u_vec,
        num_detectors, detector_spacing, voxel_spacing,
    )
    mx.eval(sinogram)

    # ── Fan-beam FBP: cosine weighting → ramp filter → backprojection ────────
    # Weight = cos(gamma) where gamma is the fan angle for each detector element
    u = (mx.arange(num_detectors) - (num_detectors - 1) / 2) * detector_spacing
    gamma = mx.arctan(u / sdd)
    weights = mx.cos(gamma).reshape(1, -1)          # (1, n_det)

    sino_weighted = sinogram * weights
    sinogram_filt = ramp_filter(sino_weighted)

    reco = diffct_mlx.fan_backward(
        sinogram_filt, src_pos, det_center, det_u_vec,
        detector_spacing=detector_spacing, H=Ny, W=Nx, voxel_spacing=voxel_spacing,
    )
    # Non-negativity + FBP normalisation
    reco = mx.maximum(reco, 0.0) * (math.pi / num_angles)
    mx.eval(reco)

    # ── Gradient demo ────────────────────────────────────────────────────────
    def loss_fn(img):
        s = diffct_mlx.fan_forward(
            img, src_pos, det_center, det_u_vec,
            num_detectors, detector_spacing, voxel_spacing,
        )
        sw = s * weights
        sf = ramp_filter(sw)
        r = diffct_mlx.fan_backward(
            sf, src_pos, det_center, det_u_vec,
            detector_spacing=detector_spacing, H=Ny, W=Nx,
            voxel_spacing=voxel_spacing,
        )
        r = mx.maximum(r, 0.0) * (math.pi / num_angles)
        return mx.mean((r - img) ** 2)

    loss_val = loss_fn(image)
    grad_fn = mx.grad(loss_fn)
    gradient = grad_fn(image)
    mx.eval(loss_val, gradient)

    print(f"Loss: {float(loss_val):.6f}")
    print(f"Centre pixel gradient: {float(gradient[Ny // 2, Nx // 2]):.6e}")

    # ── Visualisation ────────────────────────────────────────────────────────
    sino_np = np.array(sinogram)
    reco_np = np.array(reco)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(phantom_np, cmap="gray")
    plt.title("Phantom")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(sino_np, aspect="auto", cmap="gray")
    plt.title("Fan Sinogram")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(reco_np, cmap="gray")
    plt.title("Fan FBP Reconstruction")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("fbp_fan.png", dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Phantom range: [{phantom_np.min():.3f}, {phantom_np.max():.3f}]")
    print(f"Reco range:    [{reco_np.min():.3f}, {reco_np.max():.3f}]")


if __name__ == "__main__":
    main()
