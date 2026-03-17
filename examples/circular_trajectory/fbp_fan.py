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


# ── Ramp filter ──────────────────────────────────────────────────────────────

def ramp_filter(sinogram):
    """Apply ramp filter along detector axis (axis-1) using numpy FFT."""
    sino_np = np.array(sinogram)
    n_det = sino_np.shape[1]
    freqs = np.fft.fftfreq(n_det)
    ramp = (2.0 * np.abs(freqs)).astype(np.float32)
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
    phantom_np = diffct_mlx.shepp_logan_2d(Nx, Ny)
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
    # Non-negativity + empirical normalisation matching the current adjoint
    # backprojector implementation used in diffct_mlx.
    reco = mx.maximum(reco, 0.0) * (math.pi / (2.0 * num_angles))
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
        r = mx.maximum(r, 0.0) * (math.pi / (2.0 * num_angles))
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
