"""Circular-orbit fan-beam reconstruction from (simulated) real CT data.

Real-data analogue of ``fbp_fan.py``. Instead of treating the forward
projection output as "the sinogram" and feeding it straight into FBP,
this script runs the line integrals through the full X-ray detection
chain:

    line integrals   ->  intensity = I0 * exp(-sum(mu * dl))   # Beer-Lambert
                    ->  I_noisy    ~ Poisson(intensity)        # photon counts
                    ->  sino_hat   = -log(I_noisy / I0)         # preprocess

The resulting ``sino_hat`` is the kind of sinogram a real fan-beam
detector pipeline produces after flat-field / air-scan normalisation,
and it is fed into the usual analytical FBP pipeline (Parker ->
cosine -> ramp -> angular weights -> gather) unchanged.

**For your own real data:** replace step 5 (synthetic forward
projection) with an ``intensity`` tensor of shape
``(num_angles, num_detectors)`` loaded from disk, then drop steps
6.1 and 6.4 (the ``mu_scale`` scale-and-unscale) - real data already
lives in physical units, so just do
``sinogram = -log(clamp(intensity, eps) / I0)`` and send it straight
into step 7. The reconstruction will come out in your own physical
attenuation units (typically 1/mm).

Note: ``torch.poisson`` is not differentiable, so the pipeline below
does not flow gradients through the noise model. For gradient-based
training / LDCT simulation, swap the Poisson sampler for a
reparametrisable Gaussian with matched variance.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from diffct.differentiable import (
    FanProjectorFunction,
    angular_integration_weights,
    fan_cosine_weights,
    fan_weighted_backproject,
    parker_weights,
    ramp_filter_1d,
)


def shepp_logan_2d(Nx, Ny):
    """2D Shepp-Logan phantom clipped to ``[0, 1]``."""
    Nx = int(Nx)
    Ny = int(Ny)
    phantom = np.zeros((Ny, Nx), dtype=np.float32)
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
    Nx, Ny = 256, 256
    phantom = shepp_logan_2d(Nx, Ny)

    # ``voxel_spacing`` is now in *millimetres* so the photon / linear
    # attenuation units used in step 6 line up dimensionally.
    voxel_spacing = 1.0

    # ------------------------------------------------------------------
    # 2. Detector geometry
    # ------------------------------------------------------------------
    num_detectors = 600
    detector_spacing = 1.0
    detector_offset = 0.0
    sdd = 800.0
    sid = 500.0

    # ------------------------------------------------------------------
    # 3. Source trajectory (circular orbit, full or Parker short scan)
    # ------------------------------------------------------------------
    apply_parker = False

    if apply_parker:
        u_max = ((num_detectors - 1) * 0.5) * detector_spacing + abs(detector_offset)
        gamma_max = math.atan(u_max / sdd)
        scan_range = math.pi + 2.0 * gamma_max
    else:
        scan_range = 2.0 * math.pi

    num_angles = 360
    angles_np = np.linspace(
        0.0, scan_range, num_angles, endpoint=False
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # 4. Move everything to CUDA
    # ------------------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA.")
    device = torch.device("cuda")
    image_torch = torch.tensor(phantom, device=device, dtype=torch.float32)
    angles_torch = torch.tensor(angles_np, device=device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # 4.5  Pick a forward projector backend
    # ------------------------------------------------------------------
    # ``FanProjectorFunction`` accepts ``backend="siddon"`` (default) or
    # ``backend="sf"`` for the separable-footprint projector. See
    # ``fbp_fan.py`` for the full trade-off discussion. We keep the
    # default at ``"sf"`` here so the example exercises the matched
    # separable-footprint path end-to-end; switching to ``"siddon"``
    # gives a visually equivalent FBP reconstruction at lower runtime.
    projector_backend = "sf"

    # ------------------------------------------------------------------
    # 5. Forward projection -> ground-truth line integrals
    # ------------------------------------------------------------------
    # The chosen forward projector produces ``sum(phantom * dl)`` line
    # integrals, in the same units as ``-log(I/I0)`` would give on real
    # data. In a real pipeline you would *skip this step* and load your
    # own intensity tensor directly in step 6.3.
    sinogram_clean = FanProjectorFunction.apply(
        image_torch,
        angles_torch,
        num_detectors,
        detector_spacing,
        sdd,
        sid,
        voxel_spacing,
        detector_offset,
        0.0,                # center_offset_x
        0.0,                # center_offset_y
        projector_backend,
    )

    # ==================================================================
    # 6. Simulate real-data acquisition
    # ==================================================================

    # --- Acquisition parameters --------------------------------------
    mu_scale = 0.02           # 1/mm, water-like at ~70 keV
    I0 = 1.0e4
    rng_seed = 0
    eps_intensity = 0.5

    # --- 6.1  Attach a physical attenuation scale --------------------
    sinogram_phys = sinogram_clean * mu_scale * voxel_spacing

    # --- 6.2  Beer-Lambert: line integrals -> photon counts ----------
    intensity_clean = I0 * torch.exp(-sinogram_phys)

    # --- 6.3  Poisson photon-count noise -----------------------------
    gen = torch.Generator(device=device).manual_seed(rng_seed)
    intensity_noisy = torch.poisson(intensity_clean, generator=gen)

    # --- 6.4  ``-log`` preprocessing: photon counts -> line integrals
    intensity_clamped = torch.clamp(intensity_noisy, min=eps_intensity)
    sinogram_phys_noisy = -torch.log(intensity_clamped / I0)
    # Undo the physical-unit scaling so the sinogram matches the
    # "phantom_value * voxel" units that ``fan_weighted_backproject``
    # expects. For real data you would skip this line and reconstruct
    # directly into physical mu units instead.
    sinogram_noisy = sinogram_phys_noisy / (mu_scale * voxel_spacing)

    # ==================================================================
    # 7. FBP analytical reconstruction
    # ==================================================================

    # --- 7.1  Optional Parker redundancy weighting -------------------
    if apply_parker:
        parker = parker_weights(
            angles_torch, num_detectors, detector_spacing, sdd, detector_offset
        )
        sinogram_noisy = sinogram_noisy * parker

    # --- 7.2  Fan-beam cosine pre-weighting --------------------------
    weights = fan_cosine_weights(
        num_detectors,
        detector_spacing,
        sdd,
        detector_offset=detector_offset,
        device=device,
        dtype=image_torch.dtype,
    ).unsqueeze(0)
    sino_weighted = sinogram_noisy * weights

    # --- 7.3  1D ramp filter along the detector axis -----------------
    sinogram_filt = ramp_filter_1d(
        sino_weighted,
        dim=1,
        sample_spacing=detector_spacing,
        pad_factor=2,
        window="hann",
    ).contiguous()

    # --- 7.4  Per-view angular integration weights -------------------
    d_beta = angular_integration_weights(
        angles_torch, redundant_full_scan=(not apply_parker)
    ).view(-1, 1)
    sinogram_filt = sinogram_filt * d_beta

    # --- 7.5  Voxel-driven FBP backprojection ------------------------
    reconstruction_raw = fan_weighted_backproject(
        sinogram_filt,
        angles_torch,
        detector_spacing,
        Ny,
        Nx,
        sdd,
        sid,
        voxel_spacing=voxel_spacing,
        detector_offset=detector_offset,
    )
    reconstruction = F.relu(reconstruction_raw)

    # ------------------------------------------------------------------
    # 8. Quantitative summary
    # ------------------------------------------------------------------
    noisy_mse = torch.mean((reconstruction - image_torch) ** 2)
    min_photons = intensity_noisy.min().item()
    max_sino_phys = sinogram_phys.max().item()

    scan_label = "Parker short scan" if apply_parker else "full 2*pi scan"
    print(f"Fan Beam FBP example (simulated real-data workflow, {scan_label}):")
    print(f"  Backend:                {projector_backend}")
    print(f"  I0 (photons/bin/view):  {I0:.1e}")
    print(f"  mu_scale (1/mm):        {mu_scale}")
    print(f"  Max line integral:      {max_sino_phys:.3f}")
    print(f"  Min noisy photons:      {min_photons:.1f}")
    print(f"  Noisy reco MSE:         {noisy_mse.item():.6f}")
    print(f"  Reconstruction shape:   {tuple(reconstruction.shape)}")
    print(
        "  Reco data range:        "
        f"[{reconstruction.min().item():.4f}, {reconstruction.max().item():.4f}]"
    )
    print(
        "  Phantom data range:     "
        f"[{float(phantom.min()):.4f}, {float(phantom.max()):.4f}]"
    )
    print("  Tip: raise I0 to 1e5 / 1e6 to see the reconstruction de-noise.")

    # ------------------------------------------------------------------
    # 9. Visualization
    # ------------------------------------------------------------------
    sino_clean_cpu = sinogram_clean.detach().cpu().numpy()
    sino_noisy_cpu = sinogram_noisy.detach().cpu().numpy()
    intensity_noisy_cpu = intensity_noisy.detach().cpu().numpy()
    reco_cpu = reconstruction.detach().cpu().numpy()

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(phantom, cmap="gray")
    plt.title("Phantom")
    plt.axis("off")
    plt.subplot(2, 3, 2)
    plt.imshow(sino_clean_cpu, aspect="auto", cmap="gray")
    plt.title("Clean sinogram")
    plt.axis("off")
    plt.subplot(2, 3, 3)
    plt.imshow(intensity_noisy_cpu, aspect="auto", cmap="gray")
    plt.title(f"Noisy intensity (I0={I0:.0e})")
    plt.axis("off")
    plt.subplot(2, 3, 4)
    plt.imshow(sino_noisy_cpu, aspect="auto", cmap="gray")
    plt.title("Recovered (-log) sinogram")
    plt.axis("off")
    plt.subplot(2, 3, 5)
    plt.imshow(reco_cpu, cmap="gray")
    plt.title("Noisy reconstruction")
    plt.axis("off")
    plt.subplot(2, 3, 6)
    plt.imshow(reco_cpu - phantom, cmap="gray")
    plt.title("Reco - phantom")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
