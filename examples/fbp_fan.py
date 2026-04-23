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
a reference for the other analytical 2D entry points.
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

    # ``voxel_spacing`` is the physical size of one pixel in the same
    # length unit used by ``detector_spacing``, ``sdd`` and ``sid``
    # (commonly millimeters). Internally, all physical quantities are
    # divided by ``voxel_spacing``, so only their *ratios* matter.
    voxel_spacing = 1.0

    # ------------------------------------------------------------------
    # 2. Detector geometry
    # ------------------------------------------------------------------
    # (Listed before the trajectory so the short-scan coverage below can
    # use the detector fan angle to compute ``pi + 2*gamma_max``.)
    #
    # ``num_detectors`` is the number of detector cells along the
    # detector axis. ``detector_spacing`` is their physical pitch.
    # Make sure the detector is wide enough that no ray that intersects
    # the reconstructed field of view ever projects outside it - rays
    # that miss the detector are zero-filled and introduce truncation
    # artifacts at the image edges.
    num_detectors = 600
    detector_spacing = 1.0

    # Principal-ray offset (shifts the whole detector sideways).
    # Non-zero values model a detector that is not perfectly centered
    # on the source-isocenter line.
    detector_offset = 0.0

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
    # ``FanProjectorFunction`` / ``FanBackprojectorFunction`` both accept a
    # ``backend`` keyword that selects the underlying CUDA kernel family.
    # The choice applies to both forward and adjoint, and each option ships
    # with a matched scatter/gather pair so autograd and the standalone
    # Backprojector Function work unchanged.
    #
    #   "siddon" (default) - Ray-driven cell-constant Siddon traversal:
    #                        each traversed pixel contributes its value
    #                        weighted by the exact chord length, no sub-
    #                        pixel interpolation. One thread per (view,
    #                        detector bin). Fastest forward. Pick this
    #                        when you only need a forward projection and
    #                        not a matched cell-integrated model.
    #
    #   "sf"               - Voxel-driven separable-footprint projector
    #                        (SF-TR of Long-Fessler-Balter, IEEE TMI 2010).
    #                        Each voxel's projection footprint is a
    #                        trapezoid built from the four projected
    #                        corners and closed-form integrated over each
    #                        detector cell, so **mass is conserved per
    #                        voxel**. About 3x slower forward than
    #                        "siddon". On analytical FBP reconstructions
    #                        with the matched SF gather backprojector
    #                        (``fan_weighted_backproject(backend="sf")``),
    #                        SF and VD produce visually identical edge
    #                        profiles on Shepp-Logan at typical CBCT
    #                        magnifications - the "SF is sharper"
    #                        advantage that shows up in the SF / LEAP
    #                        literature only manifests at extreme
    #                        sub-nominal voxel sizes that are not hit
    #                        in standard examples. The real reason to
    #                        pick "sf" is the **forward** side: if you
    #                        plan to use this projector inside an
    #                        iterative reco, a learned prior, or any
    #                        loss that compares sinograms directly,
    #                        a cell-integrated mass-conserving forward
    #                        is the right model.
    #
    # The default is kept at "sf" so the reader can see the SF path run
    # end-to-end; switching it to "siddon" gives a visually equivalent
    # reconstruction at this geometry.
    projector_backend = "sf"

    # ------------------------------------------------------------------
    # 5. Forward projection: image -> fan sinogram
    # ------------------------------------------------------------------
    # ``FanProjectorFunction`` is the differentiable fan-beam forward
    # projector. It returns a (num_angles, num_detectors) sinogram. The
    # call is autograd-aware so the same function can be used inside an
    # iterative reconstruction loop (see ``iterative_reco_fan.py``).
    # ``backend`` selects the CUDA kernel family used for both the forward
    # and its adjoint - see step 4.5 above for the trade-offs.
    sinogram = FanProjectorFunction.apply(
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
    # 6. FBP analytical reconstruction
    # ==================================================================

    # --- 6.1  Optional Parker redundancy weighting -------------------
    # For short-scan trajectories, ``parker_weights`` returns a
    # ``(num_angles, num_detectors)`` weight that smoothly tapers views
    # near the two ends of the angular range so each ray contributes
    # exactly once. For a full 2*pi scan this helper returns all-ones
    # and is a no-op, so the ``if`` is really only for short scans.
    if apply_parker:
        parker = parker_weights(
            angles_torch, num_detectors, detector_spacing, sdd, detector_offset
        )
        sinogram = sinogram * parker

    # --- 6.2  Fan-beam cosine pre-weighting --------------------------
    # Multiplies each detector cell by ``cos(gamma) = sdd / sqrt(sdd^2
    # + u^2)``, i.e. the cosine of the fan angle. This compensates for
    # the extra path length that off-center rays traverse relative to
    # the principal ray.
    weights = fan_cosine_weights(
        num_detectors,
        detector_spacing,
        sdd,
        detector_offset=detector_offset,
        device=device,
        dtype=image_torch.dtype,
    ).unsqueeze(0)
    sino_weighted = sinogram * weights

    # --- 6.3  1D ramp filter along the detector axis -----------------
    # See the fdk_cone.py example for the full list of options - the
    # same ramp filter is used here. For fan FBP the recommended call
    # is ``sample_spacing=detector_spacing``, ``pad_factor=2``,
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
    # ``pi / num_angles`` to the FBP integral (the ``1/2`` redundancy
    # factor of the full-scan formula is absorbed inside
    # ``redundant_full_scan=True``). For a short scan with Parker
    # weights we already handle redundancy there, so pass
    # ``redundant_full_scan=False``.
    d_beta = angular_integration_weights(
        angles_torch, redundant_full_scan=(not apply_parker)
    ).view(-1, 1)
    sinogram_filt = sinogram_filt * d_beta

    # --- 6.5  Voxel-driven FBP backprojection ------------------------
    # ``fan_weighted_backproject`` dispatches to one of two fan FBP
    # gather kernels based on ``backend``:
    #
    #   "siddon" (default) - bilinear voxel-driven gather: linearly
    #                        sample the filtered sinogram at each
    #                        pixel's projected u-coordinate, multiply
    #                        by ``(sid/U)^2`` and accumulate.
    #   "sf"               - LEAP-style chord-weighted separable-
    #                        footprint gather: integrate the filtered
    #                        sinogram over each pixel's trapezoidal
    #                        footprint and weight by the in-plane
    #                        chord through the voxel plus the fan
    #                        ``sid/U`` first-power weight (matches the
    #                        matched-adjoint form in LEAP's
    #                        ``projectors_SF.cu``). Amplitude-calibrated
    #                        against the Siddon path on Shepp-Logan;
    #                        edge profiles at typical CBCT geometries
    #                        are visually indistinguishable from the
    #                        Siddon path. Pick this when you want a
    #                        cell-integrated forward / backward model
    #                        matched to the SF forward projector.
    #
    # Here we pass the same ``projector_backend`` we picked at step
    # 4.5 so forward and backward stay consistent. Amplitude is
    # calibrated by the wrapper so the returned image is ready to
    # compare against ``image_torch`` directly.
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
        backend=projector_backend,
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
