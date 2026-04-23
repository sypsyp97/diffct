"""Circular-orbit cone-beam FDK reconstruction example.

Pipeline:

    ConeProjectorFunction.apply    # forward projection (sinogram)
    parker_weights (optional)      # short-scan redundancy weighting
    cone_cosine_weights            # 1/r^2 cosine pre-weight
    ramp_filter_1d                 # row-wise ramp filter along u
    angular_integration_weights    # per-view integration weights
    cone_weighted_backproject      # voxel-driven FDK gather

Every geometry and reconstruction parameter below is annotated with
its meaning, units, and available options. This is the dev-branch
arbitrary-trajectory version: geometry comes from
``circular_trajectory_3d`` as per-view ``(src_pos, det_center,
det_u_vec, det_v_vec)`` arrays.
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from diffct import (
    ConeProjectorFunction,
    angular_integration_weights,
    circular_trajectory_3d,
    cone_cosine_weights,
    cone_weighted_backproject,
    parker_weights,
    ramp_filter_1d,
)


def shepp_logan_3d(shape):
    """Build a 3D Shepp-Logan phantom clipped to ``[0, 1]``."""
    Nz, Ny, Nx = shape
    zz, yy, xx = np.mgrid[:Nz, :Ny, :Nx].astype(np.float32)
    xx = (xx - Nx * 0.5) / (Nx * 0.5)
    yy = (yy - Ny * 0.5) / (Ny * 0.5)
    zz = (zz - Nz * 0.5) / (Nz * 0.5)

    el_params = np.array(
        [
            [0, 0, 0, 0.69, 0.92, 0.81, 0, 0, 0, 1.0],
            [0, -0.0184, 0, 0.6624, 0.874, 0.78, 0, 0, 0, -0.8],
            [0.22, 0, 0, 0.11, 0.31, 0.22, -np.pi / 10.0, 0, 0, -0.2],
            [-0.22, 0, 0, 0.16, 0.41, 0.28, np.pi / 10.0, 0, 0, -0.2],
            [0, 0.35, -0.15, 0.21, 0.25, 0.41, 0, 0, 0, 0.1],
            [0, 0.10, 0.25, 0.046, 0.046, 0.05, 0, 0, 0, 0.1],
            [0, -0.10, 0.25, 0.046, 0.046, 0.05, 0, 0, 0, 0.1],
            [-0.08, -0.605, 0, 0.046, 0.023, 0.05, 0, 0, 0, 0.1],
            [0, -0.605, 0, 0.023, 0.023, 0.02, 0, 0, 0, 0.1],
            [0.06, -0.605, 0, 0.023, 0.046, 0.02, 0, 0, 0, 0.1],
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
    return np.clip(phantom, 0.0, 1.0).astype(np.float32)


def main():
    # ------------------------------------------------------------------
    # 1. Volume geometry
    # ------------------------------------------------------------------
    # ``Nx / Ny / Nz`` are the number of voxels along each axis. The
    # phantom tensor has shape ``(Nz, Ny, Nx) = (D, H, W)``. Making the
    # volume larger increases reconstruction quality but also memory
    # and runtime which scale roughly ``O(N^3)``.
    Nx, Ny, Nz = 128, 128, 128
    phantom_cpu = shepp_logan_3d((Nz, Ny, Nx))

    # ``voxel_spacing`` is the physical size of one voxel in the same
    # length unit used by ``du``, ``dv``, ``sdd`` and ``sid`` below.
    # All geometry math inside the CUDA kernels is done in voxel units,
    # and physical spacings are divided by ``voxel_spacing`` internally,
    # so only the *ratios* matter.
    voxel_spacing = 1.0

    # ------------------------------------------------------------------
    # 2. Detector geometry
    # ------------------------------------------------------------------
    # (Listed before the trajectory so the short-scan coverage below can
    # use the detector fan angle to compute ``pi + 2*gamma_max``.)
    #
    # ``det_u`` / ``det_v`` are the number of detector cells along the
    # u (in-plane / horizontal) and v (axial / vertical) directions.
    # ``du`` / ``dv`` are their physical spacings.
    det_u, det_v = 256, 256
    du, dv = 1.0, 1.0

    # ``sdd`` = source-to-detector distance, ``sid`` = source-to-
    # isocenter distance. Their ratio is the geometric magnification
    # at the detector. Typical clinical cone-beam systems have
    # magnifications around 1.3 - 2.0; here we pick 1.5.
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
    #            where ``gamma_max`` is the maximum fan angle. Every
    #            ray is sampled at least once, some rays twice; the
    #            Parker window smoothly tapers the duplicate regions
    #            so each ray's total contribution integrates to pi.
    #
    # Short scans are common on clinical C-arm and cone-beam systems
    # where a full 2*pi rotation is mechanically impossible.
    apply_parker = False

    if apply_parker:
        u_max = (det_u * 0.5) * du  # dev convention
        gamma_max = math.atan(u_max / sdd)
        scan_range = math.pi + 2.0 * gamma_max
    else:
        scan_range = 2.0 * math.pi

    num_views = 360

    # ------------------------------------------------------------------
    # 4. Move everything to CUDA and build the trajectory arrays
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phantom_torch = torch.tensor(
        phantom_cpu, device=device, dtype=torch.float32
    ).contiguous()

    src_pos, det_center, det_u_vec, det_v_vec = circular_trajectory_3d(
        num_views,
        sid=sid,
        sdd=sdd,
        start_angle=0.0,
        end_angle=scan_range,
        device=device,
    )
    angles_torch = torch.linspace(
        0.0, scan_range, num_views + 1, device=device, dtype=torch.float32
    )[:-1]

    # ------------------------------------------------------------------
    # 5. Forward projection: volume -> sinogram
    # ------------------------------------------------------------------
    # ``ConeProjectorFunction`` is the differentiable Siddon-based cone-
    # beam forward projector. It returns a ``(num_views, det_u, det_v)``
    # sinogram. The call is autograd-aware so the same function can be
    # used inside an iterative reconstruction loop (see
    # ``iterative_reco_cone.py``).
    sinogram = ConeProjectorFunction.apply(
        phantom_torch,
        src_pos,
        det_center,
        det_u_vec,
        det_v_vec,
        det_u,
        det_v,
        du,
        dv,
        voxel_spacing,
    )

    # ==================================================================
    # 6. FDK analytical reconstruction
    # ==================================================================

    # --- 6.1  Optional Parker redundancy weighting -------------------
    # ``parker_weights`` returns a ``(num_views, det_u)`` tensor that
    # tapers rays in the redundantly-sampled regions of a short scan.
    # For a full 2*pi scan it returns all ones (no-op).
    if apply_parker:
        parker = parker_weights(
            angles_torch, det_u, du, sdd
        )
        sinogram = sinogram * parker.unsqueeze(-1)

    # --- 6.2  Cosine pre-weight --------------------------------------
    # Multiplies each detector pixel by ``sdd / sqrt(sdd^2 + u^2 + v^2)``,
    # i.e. the cosine of the cone angle. Compensates for the extra path
    # length that off-center rays traverse relative to the principal
    # ray. The ``unsqueeze(0)`` broadcasts over the view axis.
    weights = cone_cosine_weights(
        det_u, det_v, du, dv, sdd,
        device=device, dtype=phantom_torch.dtype,
    ).unsqueeze(0)
    sino_weighted = sinogram * weights

    # --- 6.3  1D ramp filter along the detector-u direction ----------
    # ``ramp_filter_1d`` is a generic building block. For high-quality
    # FDK reconstruction the recommended arguments are:
    #
    #   dim=1                  u-axis of the ``(views, u, v)`` sinogram
    #   sample_spacing=du      physical detector-u spacing (amplitude-
    #                          calibrated output across detector pitches)
    #   pad_factor=2           zero-pad the signal to suppress
    #                          circular-convolution wrap-around
    #   window="hann"          smoothly suppresses high-frequency noise
    #
    # Other ``window`` options: ``None`` / ``"ram-lak"`` (sharp Ram-Lak,
    # no smoothing), ``"hamming"``, ``"cosine"``, ``"shepp-logan"``.
    sinogram_filt = ramp_filter_1d(
        sino_weighted,
        dim=1,
        sample_spacing=du,
        pad_factor=2,
        window="hann",
    ).contiguous()

    # --- 6.4  Per-view angular integration weights -------------------
    # For a full 2*pi scan each view contributes ``pi / num_views`` to
    # the FDK integral (with the ``1/2`` redundancy factor absorbed by
    # ``redundant_full_scan=True``). For a Parker short scan the
    # redundancy is already baked into the Parker window above, so we
    # use a plain trapezoidal rule with ``redundant_full_scan=False``.
    d_beta = angular_integration_weights(
        angles_torch, redundant_full_scan=(not apply_parker)
    ).view(-1, 1, 1)
    sinogram_filt = sinogram_filt * d_beta

    # --- 6.5  Voxel-driven FDK backprojection -------------------------
    # ``cone_weighted_backproject`` dispatches to a dedicated voxel-
    # driven gather kernel (separate from the Siddon-based autograd
    # cone backprojector). For every voxel it computes the detector
    # ``(u, v)`` the voxel projects to, bilinearly samples the filtered
    # sinogram, and accumulates ``(sid_n / U_n)^2 * sample`` over all
    # views, where ``U_n`` is the per-view source-to-voxel distance
    # along the detector normal and ``sid_n`` is its iso analogue. For
    # a circular orbit both reduce to the classical scalars. The final
    # ``sdd / (2*pi*sid)`` analytical constant is applied inside the
    # wrapper.
    reconstruction_raw = cone_weighted_backproject(
        sinogram_filt,
        src_pos,
        det_center,
        det_u_vec,
        det_v_vec,
        Nz,
        Ny,
        Nx,
        du,
        dv,
        voxel_spacing=voxel_spacing,
    )

    # Optional non-negativity clamp. FDK can produce small negative
    # values near sharp edges because the ramp filter has negative
    # lobes in the spatial domain; for visualization we clamp with ReLU.
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
