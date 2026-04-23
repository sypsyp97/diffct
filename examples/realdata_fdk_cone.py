"""Circular-orbit cone-beam reconstruction from (simulated) real CT data.

Real-data analogue of ``fdk_cone.py``. Instead of treating the forward
projection output as "the sinogram" and feeding it straight into FDK,
this script runs the line integrals through the full X-ray detection
chain:

    line integrals   ->  intensity = I0 * exp(-sum(mu * dl))   # Beer-Lambert
                    ->  I_noisy    ~ Poisson(intensity)        # photon counts
                    ->  sino_hat   = -log(I_noisy / I0)         # preprocess

The resulting ``sino_hat`` is the kind of sinogram a real cone-beam
detector pipeline produces after flat-field / air-scan normalisation,
and it is fed into the usual analytical FDK pipeline (Parker ->
cosine -> ramp -> angular weights -> gather) unchanged.

**For your own real data:** replace step 5 (synthetic forward
projection) with an ``intensity`` tensor of shape
``(num_views, det_u, det_v)`` loaded from disk, then drop steps
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
    ConeProjectorFunction,
    angular_integration_weights,
    cone_cosine_weights,
    cone_weighted_backproject,
    parker_weights,
    ramp_filter_1d,
)


def shepp_logan_3d(shape):
    """Build a 3D Shepp-Logan phantom with shape ``(Nz, Ny, Nx)``."""
    zz, yy, xx = np.mgrid[: shape[0], : shape[1], : shape[2]]
    xx = (xx - (shape[2] - 1) / 2) / ((shape[2] - 1) / 2)
    yy = (yy - (shape[1] - 1) / 2) / ((shape[1] - 1) / 2)
    zz = (zz - (shape[0] - 1) / 2) / ((shape[0] - 1) / 2)

    el_params = np.array(
        [
            [0,     0,       0,     0.69,  0.92,  0.81, 0,             0, 0,  1.0],
            [0,    -0.0184,  0,     0.6624, 0.874, 0.78, 0,            0, 0, -0.8],
            [0.22,  0,       0,     0.11,  0.31,  0.22, -np.pi / 10.0, 0, 0, -0.2],
            [-0.22, 0,       0,     0.16,  0.41,  0.28,  np.pi / 10.0, 0, 0, -0.2],
            [0,     0.35,   -0.15,  0.21,  0.25,  0.41, 0,             0, 0,  0.1],
            [0,     0.10,    0.25,  0.046, 0.046, 0.05, 0,             0, 0,  0.1],
            [0,    -0.10,    0.25,  0.046, 0.046, 0.05, 0,             0, 0,  0.1],
            [-0.08,-0.605,   0,     0.046, 0.023, 0.05, 0,             0, 0,  0.1],
            [0,    -0.605,   0,     0.023, 0.023, 0.02, 0,             0, 0,  0.1],
            [0.06, -0.605,   0,     0.023, 0.046, 0.02, 0,             0, 0,  0.1],
        ],
        dtype=np.float32,
    )

    x_pos = el_params[:, 0][:, None, None, None]
    y_pos = el_params[:, 1][:, None, None, None]
    z_pos = el_params[:, 2][:, None, None, None]
    a_axis = el_params[:, 3][:, None, None, None]
    b_axis = el_params[:, 4][:, None, None, None]
    c_axis = el_params[:, 5][:, None, None, None]
    phi = el_params[:, 6][:, None, None, None]
    val = el_params[:, 9][:, None, None, None]

    xc = xx[None, ...] - x_pos
    yc = yy[None, ...] - y_pos
    zc = zz[None, ...] - z_pos

    c = np.cos(phi)
    s = np.sin(phi)
    xp = c * xc - s * yc
    yp = s * xc + c * yc
    zp = zc

    mask = (
        (xp ** 2) / (a_axis ** 2)
        + (yp ** 2) / (b_axis ** 2)
        + (zp ** 2) / (c_axis ** 2)
        <= 1.0
    )

    phantom = np.sum(mask * val, axis=0)
    return np.clip(phantom, 0.0, 1.0)


def main():
    # ------------------------------------------------------------------
    # 1. Volume geometry
    # ------------------------------------------------------------------
    Nx, Ny, Nz = 128, 128, 128
    phantom_cpu = shepp_logan_3d((Nz, Ny, Nx))

    # ``voxel_spacing`` is now in *millimetres* so the photon / linear
    # attenuation units used in step 6 line up dimensionally.
    voxel_spacing = 1.0

    # ------------------------------------------------------------------
    # 2. Detector geometry
    # ------------------------------------------------------------------
    det_u, det_v = 256, 256
    du, dv = 1.0, 1.0
    detector_offset_u = 0.0
    detector_offset_v = 0.0
    sdd = 900.0
    sid = 600.0

    # ------------------------------------------------------------------
    # 3. Source trajectory (circular orbit, full or Parker short scan)
    # ------------------------------------------------------------------
    apply_parker = False

    if apply_parker:
        u_max = ((det_u - 1) * 0.5) * du + abs(detector_offset_u)
        gamma_max = math.atan(u_max / sdd)
        scan_range = math.pi + 2.0 * gamma_max
    else:
        scan_range = 2.0 * math.pi

    num_views = 360
    angles_np = np.linspace(
        0.0, scan_range, num_views, endpoint=False
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # 4. Move everything to CUDA
    # ------------------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA.")
    device = torch.device("cuda")
    phantom_torch = torch.tensor(
        phantom_cpu, device=device, dtype=torch.float32
    ).contiguous()
    angles_torch = torch.tensor(angles_np, device=device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # 4.5  Pick a forward projector backend
    # ------------------------------------------------------------------
    # ``ConeProjectorFunction`` accepts ``backend="siddon"`` (default),
    # ``"sf_tr"``, or ``"sf_tt"``. See ``fdk_cone.py`` for the full
    # trade-off discussion. We keep the default at ``"sf_tt"`` here so
    # the example exercises the fully separable-footprint cone path
    # end-to-end; switching to ``"siddon"`` gives a visually equivalent
    # FDK reconstruction at lower runtime.
    projector_backend = "sf_tt"

    # ------------------------------------------------------------------
    # 5. Forward projection -> ground-truth line integrals
    # ------------------------------------------------------------------
    # The chosen forward projector produces ``sum(phantom * dl)`` line
    # integrals, in the same units as ``-log(I/I0)`` would give on real
    # data. In a real pipeline you would *skip this step* and load your
    # own intensity tensor directly in step 6.3.
    sinogram_clean = ConeProjectorFunction.apply(
        phantom_torch,
        angles_torch,
        det_u,
        det_v,
        du,
        dv,
        sdd,
        sid,
        voxel_spacing,
        detector_offset_u,
        detector_offset_v,
        0.0,                # center_offset_x
        0.0,                # center_offset_y
        0.0,                # center_offset_z
        projector_backend,
    )

    # ==================================================================
    # 6. Simulate real-data acquisition
    # ==================================================================

    # --- Acquisition parameters --------------------------------------
    # See ``realdata_fbp_parallel.py`` for a line-by-line explanation
    # of what each knob means physically.
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
    # "phantom_value * voxel" units that ``cone_weighted_backproject``
    # expects. For real data you would skip this line and reconstruct
    # directly into physical mu units instead.
    sinogram_noisy = sinogram_phys_noisy / (mu_scale * voxel_spacing)

    # ==================================================================
    # 7. FDK analytical reconstruction
    # ==================================================================

    # --- 7.1  Optional Parker redundancy weighting -------------------
    if apply_parker:
        parker = parker_weights(
            angles_torch,
            det_u,
            du,
            sdd,
            detector_offset=detector_offset_u,
        )
        sinogram_noisy = sinogram_noisy * parker.unsqueeze(-1)

    # --- 7.2  Cone-beam cosine pre-weighting -------------------------
    weights = cone_cosine_weights(
        det_u,
        det_v,
        du,
        dv,
        sdd,
        detector_offset_u=detector_offset_u,
        detector_offset_v=detector_offset_v,
        device=device,
        dtype=phantom_torch.dtype,
    ).unsqueeze(0)
    sino_weighted = sinogram_noisy * weights

    # --- 7.3  1D ramp filter along the detector-u direction ----------
    sinogram_filt = ramp_filter_1d(
        sino_weighted,
        dim=1,
        sample_spacing=du,
        pad_factor=2,
        window="hann",
    ).contiguous()

    # --- 7.4  Per-view angular integration weights -------------------
    d_beta = angular_integration_weights(
        angles_torch, redundant_full_scan=(not apply_parker)
    ).view(-1, 1, 1)
    sinogram_filt = sinogram_filt * d_beta

    # --- 7.5  Voxel-driven FDK backprojection ------------------------
    reconstruction_raw = cone_weighted_backproject(
        sinogram_filt,
        angles_torch,
        Nz,
        Ny,
        Nx,
        du,
        dv,
        sdd,
        sid,
        voxel_spacing=voxel_spacing,
        detector_offset_u=detector_offset_u,
        detector_offset_v=detector_offset_v,
    )
    reconstruction = F.relu(reconstruction_raw)

    # ------------------------------------------------------------------
    # 8. Quantitative summary
    # ------------------------------------------------------------------
    noisy_mse = torch.mean((reconstruction - phantom_torch) ** 2)
    min_photons = intensity_noisy.min().item()
    max_sino_phys = sinogram_phys.max().item()

    scan_label = "Parker short scan" if apply_parker else "full 2*pi scan"
    print(f"Cone Beam FDK example (simulated real-data workflow, {scan_label}):")
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
        f"[{float(phantom_cpu.min()):.4f}, {float(phantom_cpu.max()):.4f}]"
    )
    print("  Tip: raise I0 to 1e5 / 1e6 to see the reconstruction de-noise.")

    # ------------------------------------------------------------------
    # 9. Visualization
    # ------------------------------------------------------------------
    sino_clean_cpu = sinogram_clean.detach().cpu().numpy()
    sino_noisy_cpu = sinogram_noisy.detach().cpu().numpy()
    intensity_noisy_cpu = intensity_noisy.detach().cpu().numpy()
    reco_cpu = reconstruction.detach().cpu().numpy()
    mid_view = num_views // 2
    mid_slice = Nz // 2

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(phantom_cpu[mid_slice, :, :], cmap="gray")
    plt.title("Phantom mid-slice")
    plt.axis("off")
    plt.subplot(2, 3, 2)
    plt.imshow(sino_clean_cpu[mid_view].T, cmap="gray", origin="lower")
    plt.title("Clean sinogram mid-view")
    plt.axis("off")
    plt.subplot(2, 3, 3)
    plt.imshow(intensity_noisy_cpu[mid_view].T, cmap="gray", origin="lower")
    plt.title(f"Noisy intensity (I0={I0:.0e})")
    plt.axis("off")
    plt.subplot(2, 3, 4)
    plt.imshow(sino_noisy_cpu[mid_view].T, cmap="gray", origin="lower")
    plt.title("Recovered (-log) sinogram")
    plt.axis("off")
    plt.subplot(2, 3, 5)
    plt.imshow(reco_cpu[mid_slice, :, :], cmap="gray")
    plt.title("Noisy reco mid-slice")
    plt.axis("off")
    plt.subplot(2, 3, 6)
    plt.imshow(reco_cpu[mid_slice, :, :] - phantom_cpu[mid_slice, :, :], cmap="gray")
    plt.title("Reco - phantom (mid-slice)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
