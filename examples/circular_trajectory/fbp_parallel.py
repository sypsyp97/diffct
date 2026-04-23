"""Circular-orbit parallel-beam FBP reconstruction (arbitrary-trajectory API).

Uses the analytical helpers ported from the main branch:

    ParallelProjectorFunction.apply   # forward projection
    ramp_filter_1d                    # ramp filter along detector axis
    angular_integration_weights       # per-view integration weights
    parallel_weighted_backproject     # voxel-driven FBP gather with 1/(2*pi)

Parallel beam has no source, so there is no ``(sid/U)^2`` distance
weight - the gather kernel simply projects each pixel onto the detector
along the ray direction and samples the filtered sinogram. The
analytical ``1/(2*pi)`` Fourier-convention constant is applied inside
``parallel_weighted_backproject``.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from diffct import (
    ParallelProjectorFunction,
    circular_trajectory_2d_parallel,
    ramp_filter_1d,
    angular_integration_weights,
    parallel_weighted_backproject,
)


def shepp_logan_2d(Nx, Ny):
    phantom = np.zeros((Ny, Nx), dtype=np.float32)
    ellipses = [
        (0.0,    0.0,    0.69,   0.92,    0.0,  1.0),
        (0.0,   -0.0184, 0.6624, 0.8740,  0.0, -0.8),
        (0.22,   0.0,    0.11,   0.31,  -18.0, -0.8),
        (-0.22,  0.0,    0.16,   0.41,   18.0, -0.8),
        (0.0,    0.35,   0.21,   0.25,    0.0,  0.7),
    ]
    cx = Nx * 0.5
    cy = Ny * 0.5
    ys, xs = np.mgrid[0:Ny, 0:Nx].astype(np.float32)
    xn = (xs - cx) / (Nx / 2)
    yn = (ys - cy) / (Ny / 2)
    for (x0, y0, a, b, angdeg, ampl) in ellipses:
        th = math.radians(angdeg)
        xp = (xn - x0) * math.cos(th) + (yn - y0) * math.sin(th)
        yp = -(xn - x0) * math.sin(th) + (yn - y0) * math.cos(th)
        mask = (xp * xp) / (a * a) + (yp * yp) / (b * b) <= 1.0
        phantom[mask] += ampl
    return np.clip(phantom, 0.0, 1.0)


def main():
    Nx, Ny = 256, 256
    phantom = shepp_logan_2d(Nx, Ny)

    # Parallel FBP integrates over half a rotation (0 to pi) because each
    # ray and its 180-degree twin are the same measurement.
    num_angles = 360
    num_detectors = 512
    detector_spacing = 1.0
    voxel_spacing = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_torch = torch.tensor(phantom, device=device, dtype=torch.float32)

    ray_dir, det_origin, det_u_vec = circular_trajectory_2d_parallel(
        num_angles, device=device
    )
    angles_torch = torch.linspace(
        0.0, math.pi, num_angles + 1, device=device
    )[:-1]

    # ---- 1. forward projection ----
    sinogram = ParallelProjectorFunction.apply(
        image_torch, ray_dir, det_origin, det_u_vec,
        num_detectors, detector_spacing, voxel_spacing,
    )

    # ---- 2. ramp filter (no cosine pre-weight for parallel) ----
    sinogram_filt = ramp_filter_1d(
        sinogram,
        dim=1,
        sample_spacing=detector_spacing,
        pad_factor=2,
        window="hann",
    ).contiguous()

    # ---- 3. angular integration weights ----
    # Parallel beam covers [0, pi] so there is no redundancy factor.
    d_beta = angular_integration_weights(
        angles_torch, redundant_full_scan=False
    ).view(-1, 1)
    sinogram_filt = sinogram_filt * d_beta

    # ---- 4. voxel-driven FBP gather + 1/(2*pi) constant ----
    reconstruction_raw = parallel_weighted_backproject(
        sinogram_filt, ray_dir, det_origin, det_u_vec,
        detector_spacing, Ny, Nx, voxel_spacing=voxel_spacing,
    )
    reconstruction = F.relu(reconstruction_raw)

    raw_mse = torch.mean((reconstruction_raw - image_torch) ** 2).item()
    clamped_mse = torch.mean((reconstruction - image_torch) ** 2).item()
    print("Parallel Beam FBP example (arbitrary-trajectory API)")
    print(f"  Raw MSE:             {raw_mse:.6f}")
    print(f"  Clamped MSE:         {clamped_mse:.6f}")
    print(f"  Raw reco range:      [{reconstruction_raw.min().item():.4f}, "
          f"{reconstruction_raw.max().item():.4f}]")
    print(f"  Phantom range:       [{float(phantom.min()):.4f}, "
          f"{float(phantom.max()):.4f}]")

    sinogram_cpu = sinogram.detach().cpu().numpy()
    reco_cpu = reconstruction.detach().cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(phantom, cmap="gray")
    plt.title("Phantom")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(sinogram_cpu, cmap="gray", aspect="auto")
    plt.title("Parallel Sinogram")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(reco_cpu, cmap="gray")
    plt.title("Parallel Reconstruction")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
