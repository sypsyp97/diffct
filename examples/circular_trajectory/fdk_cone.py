"""FDK Cone Beam Reconstruction Example (MLX / Apple Silicon).

Demonstrates:
  - 3D Shepp-Logan phantom generation
  - Differentiable forward projection with circular cone beam geometry
  - FDK-weighted, ramp-filtered backprojection
  - Gradient computation through the full pipeline
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


# ── Ramp filter (3-D sino: views × u × v) ───────────────────────────────────

def ramp_filter_3d(sinogram):
    """Apply ramp filter along the u-axis (axis-1) using numpy FFT."""
    sino_np = np.array(sinogram)
    n_u = sino_np.shape[1]
    freqs = np.fft.fftfreq(n_u)
    ramp = np.abs(2.0 * np.pi * freqs).astype(np.float32).reshape(1, n_u, 1)
    sino_fft = np.fft.fft(sino_np, axis=1)
    filtered = np.real(np.fft.ifft(sino_fft * ramp, axis=1)).astype(np.float32)
    return mx.array(filtered)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Parameters
    Nx, Ny, Nz = 128, 128, 128
    num_views = 360
    det_u_count, det_v_count = 256, 256
    du, dv = 1.0, 1.0
    voxel_spacing = 1.0
    sdd = 900.0
    sid = 600.0

    # Phantom
    phantom_np = shepp_logan_3d((Nz, Ny, Nx))
    volume = mx.array(phantom_np)

    # Geometry
    src_pos, det_center, det_u_vec, det_v_vec = diffct_mlx.circular_trajectory_3d(
        num_views, sid, sdd,
    )

    # Forward projection
    sinogram = diffct_mlx.cone_forward(
        volume, src_pos, det_center, det_u_vec, det_v_vec,
        det_u_count, det_v_count, du, dv, voxel_spacing,
    )
    mx.eval(sinogram)

    # ── FDK: weight → ramp filter → backprojection ──────────────────────────
    # Weight = D / sqrt(D² + u² + v²)
    u_coords = (mx.arange(det_u_count) - (det_u_count - 1) / 2) * du
    v_coords = (mx.arange(det_v_count) - (det_v_count - 1) / 2) * dv
    u_coords = u_coords.reshape(1, det_u_count, 1)
    v_coords = v_coords.reshape(1, 1, det_v_count)
    weights = sdd / mx.sqrt(sdd ** 2 + u_coords ** 2 + v_coords ** 2)

    sino_weighted = sinogram * weights
    sinogram_filt = ramp_filter_3d(sino_weighted)

    reco = diffct_mlx.cone_backward(
        sinogram_filt, src_pos, det_center, det_u_vec, det_v_vec,
        D=Nz, H=Ny, W=Nx, du=du, dv=dv, voxel_spacing=voxel_spacing,
    )
    # Non-negativity + FDK normalisation
    reco = mx.maximum(reco, 0.0) * (math.pi / num_views)
    mx.eval(reco)

    # ── Gradient demo ────────────────────────────────────────────────────────
    def loss_fn(vol):
        s = diffct_mlx.cone_forward(
            vol, src_pos, det_center, det_u_vec, det_v_vec,
            det_u_count, det_v_count, du, dv, voxel_spacing,
        )
        sw = s * weights
        sf = ramp_filter_3d(sw)
        r = diffct_mlx.cone_backward(
            sf, src_pos, det_center, det_u_vec, det_v_vec,
            D=Nz, H=Ny, W=Nx, du=du, dv=dv, voxel_spacing=voxel_spacing,
        )
        r = mx.maximum(r, 0.0) * (math.pi / num_views)
        return mx.mean((r - vol) ** 2)

    loss_val = loss_fn(volume)
    grad_fn = mx.grad(loss_fn)
    gradient = grad_fn(volume)
    mx.eval(loss_val, gradient)

    print(f"Loss: {float(loss_val):.6f}")
    print(f"Centre voxel gradient: {float(gradient[Nz // 2, Ny // 2, Nx // 2]):.6e}")
    print(f"Reconstruction shape: {reco.shape}")

    # ── Visualisation ────────────────────────────────────────────────────────
    reco_np = np.array(reco)
    sino_np = np.array(sinogram)
    mid = Nz // 2

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(phantom_np[mid], cmap="gray")
    plt.title("Phantom mid-slice")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(sino_np[num_views // 2].T, cmap="gray", origin="lower")
    plt.title("Sinogram mid-view")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(reco_np[mid], cmap="gray")
    plt.title("FDK Reconstruction")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("fdk_cone.png", dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Phantom range: [{phantom_np.min():.3f}, {phantom_np.max():.3f}]")
    print(f"Reco range:    [{reco_np.min():.3f}, {reco_np.max():.3f}]")


if __name__ == "__main__":
    main()
