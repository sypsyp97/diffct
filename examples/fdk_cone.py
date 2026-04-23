"""Circular-orbit cone-beam FDK reconstruction example.

This script demonstrates the full analytical FDK pipeline on a 3D
Shepp-Logan phantom using ``diffct``. The call chain is:

    ConeProjectorFunction.apply    # forward projection (sinogram)
    cone_cosine_weights            # 1/r^2 cosine pre-weight
    ramp_filter_1d                 # row-wise ramp filter along u
    angular_integration_weights    # per-view integration weights
    cone_weighted_backproject      # voxel-driven FDK gather

Every geometry and reconstruction parameter below is annotated with
its meaning, units, and available options so the script can be used
as a reference for configuring the other cone-beam entry points in
the library.
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
    """Build a 3D Shepp-Logan phantom.

    Parameters
    ----------
    shape : tuple of int
        Volume shape as ``(Nz, Ny, Nx)`` in the same order the rest of the
        library uses. Values are clipped to ``[0, 1]``.
    """
    # Voxel grid normalized to [-1, 1] on each axis so the standard
    # Shepp-Logan ellipsoid parameters can be reused directly.
    zz, yy, xx = np.mgrid[: shape[0], : shape[1], : shape[2]]
    xx = (xx - (shape[2] - 1) / 2) / ((shape[2] - 1) / 2)
    yy = (yy - (shape[1] - 1) / 2) / ((shape[1] - 1) / 2)
    zz = (zz - (shape[0] - 1) / 2) / ((shape[0] - 1) / 2)

    # Columns: x0, y0, z0, a, b, c, phi, _, _, amplitude.
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
    # Each ellipsoid only rotates around the z-axis here.
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
    # ``Nx / Ny / Nz`` are the number of voxels along each axis. The
    # phantom tensor has shape ``(Nz, Ny, Nx)`` which matches the
    # ``(D, H, W)`` layout that every cone-beam routine in diffct
    # expects. Making the volume larger increases the reconstruction
    # grid but also increases memory and runtime roughly ``O(N^3)``.
    Nx, Ny, Nz = 128, 128, 128
    phantom_cpu = shepp_logan_3d((Nz, Ny, Nx))

    # ``voxel_spacing`` is the physical size of one voxel in the same
    # length unit used by ``du``, ``dv``, ``sdd`` and ``sid`` below
    # (commonly millimeters). All geometry math inside the CUDA kernels
    # is done in voxel units, and physical spacings are divided by
    # ``voxel_spacing`` internally, so only the *ratios* matter.
    voxel_spacing = 1.0

    # ------------------------------------------------------------------
    # 2. Detector geometry
    # ------------------------------------------------------------------
    # (Listed before the trajectory so the short-scan coverage below can
    # use the detector fan angle to compute ``pi + 2*gamma_max``.)
    #
    # ``det_u`` / ``det_v`` are the number of detector cells along the
    # u (in-plane / horizontal) and v (axial / vertical) directions.
    # ``du`` / ``dv`` are their physical spacings. Together they define
    # the detector size in physical units:
    #     detector_width_u  = det_u * du
    #     detector_height_v = det_v * dv
    # Make sure the detector is large enough that no ray that intersects
    # the reconstructed field-of-view ever projects outside it; rays
    # that miss the detector are zero-filled and create truncation
    # artifacts.
    det_u, det_v = 256, 256
    du, dv = 1.0, 1.0

    # Principal-ray offsets. A nonzero ``detector_offset_u`` models a
    # detector that is shifted sideways relative to the ideal principal
    # ray (useful for half-fan / offset-detector acquisitions). These
    # values are in the same physical unit as ``du`` / ``dv``.
    detector_offset_u = 0.0
    detector_offset_v = 0.0

    # ``sdd`` = source-to-detector distance, ``sid`` = source-to-
    # isocenter distance, both in physical units. Their ratio
    # ``sdd / sid`` is the geometric magnification at the detector.
    # Typical clinical cone-beam systems have a magnification around
    # 1.3 - 2.0; here we pick 1.5.
    sdd = 900.0
    sid = 600.0

    # ------------------------------------------------------------------
    # 3. Source trajectory (circular orbit)
    # ------------------------------------------------------------------
    # ``apply_parker`` selects between two supported trajectories:
    #
    #   False -> full 2*pi circular scan. Each ray is sampled twice
    #            (once going one way, once the opposite), so the FDK
    #            formula carries a 1/2 redundancy factor which is
    #            absorbed by ``redundant_full_scan=True`` inside the
    #            angular integration weights.
    #
    #   True  -> minimal short scan of length ``pi + 2*gamma_max``
    #            where ``gamma_max = atan(u_max / sdd)`` is the maximum
    #            fan angle. Every ray is sampled *at least once*, some
    #            rays twice; the standard Parker window smoothly tapers
    #            the duplicate regions so each ray's total contribution
    #            integrates to pi (the same effective weight as the
    #            full-scan case after the 1/2 redundancy factor), and
    #            the angular weights use plain trapezoidal rule
    #            (``redundant_full_scan=False``).
    #
    # Short scans are common on clinical C-arm and cone-beam systems
    # where a full 2*pi rotation is mechanically impossible.
    apply_parker = False

    if apply_parker:
        u_max = ((det_u - 1) * 0.5) * du + abs(detector_offset_u)
        gamma_max = math.atan(u_max / sdd)
        scan_range = math.pi + 2.0 * gamma_max
    else:
        scan_range = 2.0 * math.pi

    # ``num_views`` is the number of projection angles sampled on the
    # orbit. More views generally mean smaller angular aliasing and a
    # cleaner FDK reconstruction, at the cost of memory and runtime
    # linear in ``num_views``.
    num_views = 360

    # Angles in radians. ``endpoint=False`` avoids duplicating the start
    # angle at the end of the sweep.
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
    # ``ConeProjectorFunction`` / ``ConeBackprojectorFunction`` both accept
    # a ``backend`` keyword that selects the underlying CUDA kernel family.
    # The choice applies to both forward and adjoint, and each option ships
    # with a matched scatter/gather pair so autograd and the standalone
    # Backprojector Function work unchanged.
    #
    #   "siddon"           - 3D ray-driven Siddon traversal with
    #                        trilinear interpolation. One thread per
    #                        (view, det_u, det_v). Fastest forward.
    #                        Pick this when you only need a forward
    #                        projection and don't need a matched cell-
    #                        integrated model.
    #
    #   "sf_tr"            - 3D voxel-driven separable-footprint with a
    #                        trapezoidal transaxial (u) footprint and a
    #                        rectangular axial (v) footprint evaluated
    #                        at the voxel-centre magnification. **Mass-
    #                        conserving per voxel**, closed-form
    #                        integrated over each detector cell in
    #                        both u and v. About 2x slower than siddon
    #                        in our benchmarks. On analytical FDK with
    #                        the matched SF gather backprojector
    #                        (``cone_weighted_backproject(backend=
    #                        "sf_tr")``) SF and Siddon VD produce
    #                        visually equivalent edge profiles on
    #                        Shepp-Logan and the walnut example at
    #                        typical CBCT geometries; the "SF is sharper
    #                        at sub-nominal" claim you see in the SF /
    #                        LEAP literature only shows up in an extreme
    #                        sub-nominal regime that isn't hit by the
    #                        default geometries shipped here. The real
    #                        reason to pick "sf_tr" is that you want a
    #                        **cell-integrated forward model** for
    #                        iterative reco / learned priors / sinogram
    #                        losses.
    #
    #   "sf_tt"            - Same transaxial trapezoid as SF-TR but the
    #                        axial footprint is ALSO a trapezoid, built
    #                        from four ``(U_near, U_far) x (z_bot, z_top)``
    #                        corner projections. Captures the variation
    #                        of axial magnification inside a single voxel
    #                        at large cone angles. In practice this
    #                        refines the SF-TR result marginally at the
    #                        cost of ~40% more runtime.
    #
    # All three backends are byte-accurate matched adjoint pairs on the
    # autograd path (verified by ``tests/test_adjoint_inner_product.py``).
    # The default is kept at "sf_tr" so the reader sees the SF path run
    # end-to-end; switching to "siddon" gives a visually equivalent FDK
    # reconstruction at this geometry.
    projector_backend = "sf_tr"

    # ------------------------------------------------------------------
    # 5. Forward projection: volume -> sinogram
    # ------------------------------------------------------------------
    # ``ConeProjectorFunction`` is the differentiable cone-beam forward
    # projector. It returns a (num_views, det_u, det_v) sinogram. This
    # call is autograd-aware, so using the same function inside an
    # iterative reconstruction loop is supported (see
    # ``iterative_reco_cone.py``). ``backend`` selects the CUDA kernel
    # family used for both the forward and its adjoint - see step 4.5
    # above for the trade-offs.
    sinogram = ConeProjectorFunction.apply(
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
    # 6. FDK analytical reconstruction
    # ==================================================================

    # --- 6.0  Optional Parker redundancy weighting -------------------
    # ``parker_weights`` returns a ``(num_views, det_u)`` tensor that
    # tapers rays in the redundantly-sampled regions of a short scan.
    # For a full 2*pi scan it detects the coverage and returns all
    # ones (no-op), so this branch is safe to run unconditionally; we
    # gate it on ``apply_parker`` just for clarity.
    #
    # The returned weight depends on the in-plane fan angle only, so
    # we broadcast it across the v (axial) direction via ``unsqueeze(-1)``.
    if apply_parker:
        parker = parker_weights(
            angles_torch,
            det_u,
            du,
            sdd,
            detector_offset=detector_offset_u,
        )
        sinogram = sinogram * parker.unsqueeze(-1)

    # --- 6.1  Cosine pre-weight --------------------------------------
    # Multiplies each detector pixel by ``sdd / sqrt(sdd^2 + u^2 + v^2)``,
    # i.e. the cosine of the cone angle. This compensates for the
    # extra path length that off-center rays traverse relative to the
    # principal ray. The ``unsqueeze(0)`` broadcasts the 2D weight over
    # every view.
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
    sino_weighted = sinogram * weights

    # --- 6.2  1D ramp filter along the detector-u direction ----------
    # ``ramp_filter_1d`` is a generic building block. For high-quality
    # FDK reconstruction the recommended arguments are:
    #
    #   dim : int
    #       The detector-u axis. For sinograms of shape
    #       ``(views, u, v)`` that axis is ``1``.
    #
    #   sample_spacing : float
    #       Physical detector-u spacing ``du``. The ramp filter is
    #       rescaled by ``1 / sample_spacing`` so that the output is
    #       in physical units, and the amplitude of the reconstruction
    #       stays calibrated across different detector pitches.
    #
    #   pad_factor : int
    #       Zero-pad the signal to ``pad_factor * N`` along ``dim``
    #       before the FFT. Options:
    #           1 (default)  - no padding, fastest, but circular
    #                          convolution wrap-around can contaminate
    #                          the edges of the reconstruction.
    #           2 (typical)  - good balance for cone-beam FDK.
    #           4 (strict)   - used when the object is very close to
    #                          the detector edges.
    #
    #   window : str or None
    #       Frequency-domain apodization applied on top of the bare
    #       ramp. Available options are:
    #           None / "ram-lak" - sharp Ram-Lak ramp, no smoothing.
    #           "hann"           - Hann window, strongly suppresses
    #                              high-frequency noise. Used below.
    #           "hamming"        - slightly milder than Hann.
    #           "cosine"         - equivalent to a half-cosine rolloff.
    #           "shepp-logan"    - sinc rolloff, classical choice.
    #
    #   use_rfft : bool
    #       Use ``torch.fft.rfft`` instead of ``fft`` when the input is
    #       real. Default ``True``; set to ``False`` only if you need
    #       the complex path (e.g. to reuse an already-complex buffer).
    sinogram_filt = ramp_filter_1d(
        sino_weighted,
        dim=1,
        sample_spacing=du,
        pad_factor=2,
        window="hann",
    ).contiguous()

    # --- 6.3  Per-view angular integration weights -------------------
    # For a full 2*pi scan each view contributes ``pi / num_views`` to
    # the FDK integral: the ``1/2`` redundancy factor of the FDK
    # formula is absorbed inside ``redundant_full_scan=True``. For a
    # Parker short scan the redundancy has already been handled by the
    # Parker window above, so we fall back to a plain trapezoidal rule
    # with ``redundant_full_scan=False`` (the 1/2 factor would then be
    # incorrect).
    d_beta = angular_integration_weights(
        angles_torch, redundant_full_scan=(not apply_parker)
    ).view(-1, 1, 1)
    sinogram_filt = sinogram_filt * d_beta

    # --- 6.4  Voxel-driven FDK backprojection -------------------------
    # ``cone_weighted_backproject`` dispatches to one of three voxel-
    # driven gather kernels based on ``backend``:
    #
    #   "siddon" (default) - bilinear gather: for each voxel compute
    #                        its projected detector ``(u, v)``,
    #                        bilinearly sample the filtered sinogram
    #                        and accumulate ``(sid/U)^2 * sample``.
    #   "sf_tr"            - LEAP-style chord-weighted separable-
    #                        footprint gather: integrate the filtered
    #                        sinogram over each voxel's transaxial
    #                        trapezoidal footprint and axial
    #                        rectangular footprint, weighted by the
    #                        in-plane chord through the voxel and the
    #                        ``sqrt(1+(v/sdd)^2)`` axial correction
    #                        (matches the tilt==0 branch of LEAP's
    #                        ``coneBeamBackprojectorKernel_SF`` in
    #                        ``projectors_SF.cu``). Matches the SF-TR
    #                        forward projector.
    #   "sf_tt"            - same as "sf_tr" but the axial footprint
    #                        is also a trapezoid built from four
    #                        z-corner projections. Matches SF-TT.
    #
    # On Shepp-Logan and the walnut example all three backends give
    # visually indistinguishable edge profiles and amplitude matches
    # within ~1%; the SF backends are worth the ~2-3x forward cost
    # only if you also want a cell-integrated *forward* model
    # (iterative reco, learned priors, sinogram losses). Here we pass
    # the same ``projector_backend`` we picked at step 4.5 so forward
    # and backward stay consistent. Amplitude is calibrated inside
    # the wrapper so the returned volume is ready to compare against
    # ``phantom_torch`` directly.
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
        backend=projector_backend,
    )

    # Optional non-negativity clamp. FDK can produce small negative
    # values near sharp edges because the ramp filter has negative
    # lobes in the spatial domain; for visualization and quantitative
    # comparison against the non-negative phantom we clamp with a ReLU.
    reconstruction = F.relu(reconstruction_raw)

    # ------------------------------------------------------------------
    # 7. Quantitative summary
    # ------------------------------------------------------------------
    raw_loss = torch.mean((reconstruction_raw - phantom_torch) ** 2)
    clamped_loss = torch.mean((reconstruction - phantom_torch) ** 2)

    scan_label = "Parker short scan" if apply_parker else "full 2*pi scan"
    print(f"Cone Beam FDK example ({scan_label}):")
    print(f"  Raw MSE:             {raw_loss.item():.6f}")
    print(f"  Clamped MSE:         {clamped_loss.item():.6f}")
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
        f"[{float(phantom_cpu.min()):.4f}, {float(phantom_cpu.max()):.4f}]"
    )

    # ------------------------------------------------------------------
    # 8. Visualization
    # ------------------------------------------------------------------
    reconstruction_cpu = reconstruction.detach().cpu().numpy()
    sinogram_cpu = sinogram.detach().cpu().numpy()
    mid_slice = Nz // 2

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(phantom_cpu[mid_slice, :, :], cmap="gray")
    plt.title("Phantom mid-slice")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    # Transpose so the v axis is vertical in the figure.
    plt.imshow(sinogram_cpu[num_views // 2].T, cmap="gray", origin="lower")
    plt.title("Sinogram mid-view")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(reconstruction_cpu[mid_slice, :, :], cmap="gray")
    plt.title("Recon mid-slice")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
