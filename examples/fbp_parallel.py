"""Parallel-beam FBP reconstruction example.

Pipeline (matches the fan-beam and cone-beam analytical examples):

    ParallelProjectorFunction.apply  # forward projection (sinogram)
    ramp_filter_1d                   # ramp filter along detector axis
    angular_integration_weights      # per-view integration weights
    parallel_weighted_backproject    # voxel-driven FBP gather

Parallel beam has no source, so there is no cosine pre-weighting and
no distance weighting during backprojection - the FBP pipeline only
needs ramp filtering, angular weights, and the ``1/(2*pi)`` analytical
constant that ``parallel_weighted_backproject`` applies internally.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from diffct.differentiable import (
    ParallelProjectorFunction,
    angular_integration_weights,
    parallel_weighted_backproject,
    ramp_filter_1d,
)


def shepp_logan_2d(Nx, Ny):
    """2D Shepp-Logan phantom clipped to ``[0, 1]``."""
    Nx = int(Nx)
    Ny = int(Ny)
    phantom = np.zeros((Ny, Nx), dtype=np.float32)
    # (x0, y0, a, b, angle_deg, amplitude)
    ellipses = [
        (0.0,    0.0,    0.69,   0.92,    0.0,  1.0),
        (0.0,   -0.0184, 0.6624, 0.8740,  0.0, -0.8),
        (0.22,   0.0,    0.11,   0.31,  -18.0, -0.8),
        (-0.22,  0.0,    0.16,   0.41,   18.0, -0.8),
        (0.0,    0.35,   0.21,   0.25,    0.0,  0.7),
    ]
    cx = (Nx - 1) * 0.5
    cy = (Ny - 1) * 0.5
    for ix in range(Nx):
        for iy in range(Ny):
            xnorm = (ix - cx) / (Nx / 2)
            ynorm = (iy - cy) / (Ny / 2)
            val = 0.0
            for (x0, y0, a, b, angdeg, ampl) in ellipses:
                th = np.deg2rad(angdeg)
                xprime = (xnorm - x0) * np.cos(th) + (ynorm - y0) * np.sin(th)
                yprime = -(xnorm - x0) * np.sin(th) + (ynorm - y0) * np.cos(th)
                if xprime * xprime / (a * a) + yprime * yprime / (b * b) <= 1.0:
                    val += ampl
            phantom[iy, ix] = val
    return np.clip(phantom, 0.0, 1.0)


def main():
    # ------------------------------------------------------------------
    # 1. Image geometry
    # ------------------------------------------------------------------
    # ``Nx`` / ``Ny`` are the reconstruction grid size in pixels. The
    # phantom tensor has shape ``(Ny, Nx)`` (rows, cols) which matches
    # the ``(H, W)`` layout every 2D routine in diffct expects.
    Nx, Ny = 256, 256
    phantom = shepp_logan_2d(Nx, Ny)

    # ``voxel_spacing`` is the physical pixel pitch in the same units as
    # ``detector_spacing``. Only ratios matter internally.
    voxel_spacing = 1.0

    # ------------------------------------------------------------------
    # 2. Source trajectory (parallel beam on a circular orbit)
    # ------------------------------------------------------------------
    # For parallel beam the Radon inversion is periodic with period pi:
    # views at beta and beta+pi carry the same information up to a
    # flip of t. Sampling a full 2*pi range double-counts each ray and
    # we correct for it with ``redundant_full_scan=True`` in the
    # angular weights. You can equivalently use
    # ``np.linspace(0, np.pi, num_angles, endpoint=False)`` together
    # with ``redundant_full_scan=False``.
    num_angles = 360
    angles_np = np.linspace(
        0.0, 2.0 * np.pi, num_angles, endpoint=False
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # 3. Detector geometry
    # ------------------------------------------------------------------
    # ``num_detectors`` is the number of detector cells. ``detector_spacing``
    # is their physical pitch. Parallel beam needs enough detector cells
    # to cover the diagonal of the field of view (approximately
    # ``sqrt(Nx^2 + Ny^2) * voxel_spacing / detector_spacing``) so that
    # no ray through the FOV ever projects outside the detector.
    num_detectors = 512
    detector_spacing = 1.0
    detector_offset = 0.0

    # ------------------------------------------------------------------
    # 4. Move everything to CUDA
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_torch = torch.tensor(
        phantom, device=device, dtype=torch.float32, requires_grad=True
    )
    angles_torch = torch.tensor(angles_np, device=device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # 5. Forward projection: image -> parallel sinogram
    # ------------------------------------------------------------------
    # ``ParallelProjectorFunction`` is the differentiable Siddon-based
    # parallel-beam forward projector. It returns a
    # (num_angles, num_detectors) sinogram. The call is autograd-aware
    # so the same function can be used inside an iterative reconstruction
    # loop (see ``iterative_reco_parallel.py``).
    sinogram = ParallelProjectorFunction.apply(
        image_torch,
        angles_torch,
        num_detectors,
        detector_spacing,
        voxel_spacing,
    )

    # ==================================================================
    # 6. FBP analytical reconstruction
    # ==================================================================

    # --- 6.1  1D ramp filter along the detector axis -----------------
    # Parallel beam does not need a cosine pre-weight (there is no fan
    # angle). The only filtering step is the 1D ramp along the
    # detector-u axis. See the fdk_cone.py example for the full list
    # of ``ramp_filter_1d`` options.
    sinogram_filt = ramp_filter_1d(
        sinogram,
        dim=1,
        sample_spacing=detector_spacing,
        pad_factor=2,
        window="hann",
    ).contiguous()

    # --- 6.2  Per-view angular integration weights -------------------
    # For a full 2*pi scan uniformly sampled, each view contributes
    # ``pi / num_angles`` to the FBP integral (the ``1/2`` redundancy
    # factor is absorbed by ``redundant_full_scan=True``).
    d_beta = angular_integration_weights(
        angles_torch, redundant_full_scan=True
    ).view(-1, 1)
    sinogram_filt = sinogram_filt * d_beta

    # --- 6.3  Voxel-driven FBP backprojection ------------------------
    # ``parallel_weighted_backproject`` dispatches to the dedicated
    # parallel-beam FBP gather kernel. For each pixel and each view it
    # computes the detector-u coordinate the pixel projects to,
    # linearly samples the filtered sinogram and accumulates. The
    # ``1/(2*pi)`` Fourier-convention constant is applied inside the
    # wrapper so the returned image is already amplitude-calibrated.
    reconstruction_raw = parallel_weighted_backproject(
        sinogram_filt,
        angles_torch,
        detector_spacing,
        Ny,
        Nx,
        voxel_spacing=voxel_spacing,
        detector_offset=detector_offset,
    )
    reconstruction = F.relu(reconstruction_raw)

    # ------------------------------------------------------------------
    # 7. Quantitative summary + a gradient sanity check
    # ------------------------------------------------------------------
    raw_loss = torch.mean((reconstruction_raw - image_torch) ** 2)
    clamped_loss = torch.mean((reconstruction - image_torch) ** 2)
    # Trigger autograd to make sure the differentiable forward still
    # flows a gradient back to ``image_torch`` after the pipeline.
    clamped_loss.backward()

    print("Parallel Beam FBP example (circular full scan):")
    print(f"  Raw MSE:              {raw_loss.item():.6f}")
    print(f"  Clamped MSE:          {clamped_loss.item():.6f}")
    print(f"  Reconstruction shape: {tuple(reconstruction.shape)}")
    print(
        "  Raw reco data range:  "
        f"[{reconstruction_raw.min().item():.4f}, {reconstruction_raw.max().item():.4f}]"
    )
    print(
        "  Clamped reco range:   "
        f"[{reconstruction.min().item():.4f}, {reconstruction.max().item():.4f}]"
    )
    print(
        "  Phantom data range:   "
        f"[{float(phantom.min()):.4f}, {float(phantom.max()):.4f}]"
    )
    print(
        "  |phantom.grad|.mean:  "
        f"{image_torch.grad.abs().mean().item():.6e}"
    )

    # ------------------------------------------------------------------
    # 8. Visualization
    # ------------------------------------------------------------------
    sinogram_cpu = sinogram.detach().cpu().numpy()
    reco_cpu = reconstruction.detach().cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(phantom, cmap="gray")
    plt.title("Phantom")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(sinogram_cpu, aspect="auto", cmap="gray")
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
