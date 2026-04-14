"""Circular-orbit fan-beam FBP reconstruction example.

Pipeline (matches the cone-beam FDK example, one dimension lower):

    FanProjectorFunction.apply      # forward projection (sinogram)
    parker_weights (optional)       # short-scan redundancy weighting
    fan_cosine_weights              # cos(gamma) pre-weighting
    ramp_filter_1d                  # ramp filter along detector axis
    angular_integration_weights     # per-view integration weights
    fan_weighted_backproject        # voxel-driven FBP gather

Every geometry and reconstruction parameter below is annotated with
its meaning, units, and available options so the file can be used as
a reference for the other analytical 2D entry points. This is the
dev-branch arbitrary-trajectory version: geometry is expressed as
per-view ``(src_pos, det_center, det_u_vec)`` tensors produced by the
``circular_trajectory_2d_fan`` helper, and every subsequent helper
accepts the same arrays.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from diffct import (
    FanProjectorFunction,
    angular_integration_weights,
    circular_trajectory_2d_fan,
    fan_cosine_weights,
    fan_weighted_backproject,
    parker_weights,
    ramp_filter_1d,
)


def shepp_logan_2d(Nx, Ny):
    """2D Shepp-Logan phantom clipped to ``[0, 1]``."""
    phantom = np.zeros((Ny, Nx), dtype=np.float32)
    ellipses = [
        (0.0,    0.0,    0.69,   0.92,    0.0,  1.0),
        (0.0,   -0.0184, 0.6624, 0.8740,  0.0, -0.8),
        (0.22,   0.0,    0.11,   0.31,  -18.0, -0.8),
        (-0.22,  0.0,    0.16,   0.41,   18.0, -0.8),
        (0.0,    0.35,   0.21,   0.25,    0.0,  0.7),
    ]
    cx = Nx * 0.5  # kernel convention
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
    # ------------------------------------------------------------------
    # 1. Image geometry
    # ------------------------------------------------------------------
    # ``Nx`` / ``Ny`` are the reconstruction grid size in pixels. The
    # phantom tensor has shape ``(Ny, Nx)`` which matches the ``(H, W)``
    # layout every 2D routine in diffct expects.
    Nx, Ny = 256, 256
    phantom = shepp_logan_2d(Nx, Ny)

    # ``voxel_spacing`` is the physical size of one pixel in the same
    # length unit used by ``detector_spacing``, ``sdd`` and ``sid``
    # below. Internally all physical quantities are divided by
    # ``voxel_spacing``, so only their *ratios* matter.
    voxel_spacing = 1.0

    # ------------------------------------------------------------------
    # 2. Detector geometry
    # ------------------------------------------------------------------
    # (Listed before the trajectory so the short-scan coverage below can
    # use the detector fan angle to compute ``pi + 2*gamma_max``.)
    #
    # ``num_detectors`` is the number of detector cells along the
    # detector axis. ``detector_spacing`` is their physical pitch. Make
    # sure the detector is wide enough that no ray through the FOV ever
    # projects outside it, otherwise truncation artifacts appear at the
    # image edges.
    num_detectors = 600
    detector_spacing = 1.0

    # ``sdd`` = source-to-detector distance, ``sid`` = source-to-
    # isocenter distance, both in physical units. The magnification at
    # the detector is ``sdd / sid`` (here = 1.6). Typical clinical fan
    # beam systems have magnifications around 1.3 - 2.0.
    sdd = 800.0
    sid = 500.0

    # ------------------------------------------------------------------
    # 3. Source trajectory (circular orbit)
    # ------------------------------------------------------------------
    # ``apply_parker`` selects between two supported trajectories:
    #
    #   False -> full 2*pi scan, 1/2 FBP redundancy factor absorbed
    #            inside ``redundant_full_scan=True``.
    #
    #   True  -> minimal short scan ``pi + 2*gamma_max`` with Parker
    #            redundancy weighting. ``gamma_max`` is computed from
    #            the detector half-width and ``sdd``.
    #
    # Both branches reuse the same ``num_angles`` so reconstruction
    # runtime stays identical - only the angular *range* changes.
    apply_parker = False

    if apply_parker:
        # Use dev's detector convention (u[k] = (k - N/2)*ds) for the
        # maximum |u| on the detector.
        u_max = (num_detectors * 0.5) * detector_spacing
        gamma_max = math.atan(u_max / sdd)
        scan_range = math.pi + 2.0 * gamma_max
    else:
        scan_range = 2.0 * math.pi

    num_angles = 360

    # ------------------------------------------------------------------
    # 4. Move everything to CUDA and build the trajectory arrays
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_torch = torch.tensor(phantom, device=device, dtype=torch.float32)

    src_pos, det_center, det_u_vec = circular_trajectory_2d_fan(
        num_angles,
        sid=sid,
        sdd=sdd,
        start_angle=0.0,
        end_angle=scan_range,
        device=device,
    )
    angles_torch = torch.linspace(
        0.0, scan_range, num_angles + 1, device=device, dtype=torch.float32
    )[:-1]

    # ------------------------------------------------------------------
    # 5. Forward projection: image -> fan sinogram
    # ------------------------------------------------------------------
    # ``FanProjectorFunction`` is the differentiable Siddon-based fan-
    # beam forward projector. It returns a ``(num_angles, num_detectors)``
    # sinogram. The call is autograd-aware so the same function can be
    # used inside an iterative reconstruction loop (see
    # ``iterative_reco_fan.py``).
    sinogram = FanProjectorFunction.apply(
        image_torch,
        src_pos,
        det_center,
        det_u_vec,
        num_detectors,
        detector_spacing,
        voxel_spacing,
    )

    # ==================================================================
    # 6. FBP analytical reconstruction
    # ==================================================================

    # --- 6.1  Optional Parker redundancy weighting -------------------
    # For short-scan trajectories, ``parker_weights`` returns a
    # ``(num_angles, num_detectors)`` weight that smoothly tapers views
    # near the two ends of the angular range so each ray contributes
    # exactly once. For a full 2*pi scan this helper returns all-ones
    # and is a no-op.
    if apply_parker:
        parker = parker_weights(
            angles_torch, num_detectors, detector_spacing, sdd
        )
        sinogram = sinogram * parker

    # --- 6.2  Fan-beam cosine pre-weighting --------------------------
    # Multiplies each detector cell by ``cos(gamma) = sdd / sqrt(sdd^2
    # + u^2)``, the cosine of the fan angle. Compensates for the extra
    # path length that off-center rays traverse relative to the
    # principal ray.
    weights = fan_cosine_weights(
        num_detectors,
        detector_spacing,
        sdd,
        device=device,
        dtype=image_torch.dtype,
    ).unsqueeze(0)
    sino_weighted = sinogram * weights

    # --- 6.3  1D ramp filter along the detector axis -----------------
    # See the fdk_cone.py example for the full option list. Recommended
    # fan FBP call: ``sample_spacing=detector_spacing``, ``pad_factor=2``,
    # ``window="hann"``.
    sinogram_filt = ramp_filter_1d(
        sino_weighted,
        dim=1,
        sample_spacing=detector_spacing,
        pad_factor=2,
        window="hann",
    ).contiguous()

    # --- 6.4  Per-view angular integration weights -------------------
    # For a full 2*pi scan uniformly sampled, each view contributes
    # ``pi / num_angles`` to the FBP integral; the ``1/2`` redundancy
    # factor of the full-scan formula is absorbed inside
    # ``redundant_full_scan=True``. For a short scan with Parker
    # weights we already handle redundancy there, so pass
    # ``redundant_full_scan=False``.
    d_beta = angular_integration_weights(
        angles_torch, redundant_full_scan=(not apply_parker)
    ).view(-1, 1)
    sinogram_filt = sinogram_filt * d_beta

    # --- 6.5  Voxel-driven FBP backprojection ------------------------
    # ``fan_weighted_backproject`` dispatches to the dedicated fan FBP
    # gather kernel. For each pixel it projects onto the detector,
    # linearly samples the filtered sinogram, multiplies by the
    # per-view ``(sid_n/U_n)^2`` weight (generalised to the detector
    # normal for arbitrary trajectories) and accumulates over views.
    # The analytical ``sdd/(2*pi*sid)`` FBP constant is applied inside
    # the wrapper so the returned image is already amplitude-calibrated.
    reconstruction_raw = fan_weighted_backproject(
        sinogram_filt,
        src_pos,
        det_center,
        det_u_vec,
        detector_spacing,
        Ny,
        Nx,
        voxel_spacing=voxel_spacing,
    )
    reconstruction = F.relu(reconstruction_raw)

    # ------------------------------------------------------------------
    # 7. Quantitative summary
    # ------------------------------------------------------------------
    raw_loss = torch.mean((reconstruction_raw - image_torch) ** 2)
    clamped_loss = torch.mean((reconstruction - image_torch) ** 2)

    scan_label = "Parker short scan" if apply_parker else "full 2*pi scan"
    print(f"Fan Beam FBP example ({scan_label}):")
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
    plt.imshow(sinogram_cpu, cmap="gray", aspect="auto")
    plt.title("Fan Sinogram")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(reco_cpu, cmap="gray")
    plt.title("Fan Reconstruction")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
