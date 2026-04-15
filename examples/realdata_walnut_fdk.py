"""Cone-beam FDK reconstruction of a real walnut scan.

This example uses the Helsinki walnut cone-beam CT dataset by
Alexander Meaney (2022), Zenodo record 6986012, distributed under
CC-BY 4.0. A downsampled, flat-field-normalised and log-converted
subset lives at ``examples/data/walnut_cone.npz`` (~20 MB) together
with its physical scan geometry; see ``examples/data/NOTICE`` for
the full attribution.

Pipeline (identical to ``fdk_cone.py``):

    load sinogram      # from .npz (already preprocessed into line integrals)
    cone_cosine_weights            # 1/r^2 cosine pre-weight
    ramp_filter_1d                 # row-wise ramp filter along u
    angular_integration_weights    # per-view integration weights
    cone_weighted_backproject      # voxel-driven FDK gather

The raw dataset was acquired on the University of Helsinki industrial
CBCT scanner (Oxford Instruments XTF5011 tube, Hamamatsu C7942CA-22
detector) at ``SDD = 553.74 mm`` / ``SID = 210.66 mm``, 721
projections over a full ``2*pi`` rotation, 0.050 mm native detector
pitch. For this example we bin the detector 8x to 256x256 pixels,
take every 3rd projection (241 / 721), and store as float16 so the
file fits under ~25 MB while keeping walnut-scale detail.

The example reconstructs at **half the nominal voxel size** on a
512x512x512 grid with the **separable-footprint** backprojector
(``backend="sf_tr"``, LEAP's chord-weighted matched-adjoint form
from ``projectors_SF.cu``) and a **Hamming** ramp window. The
choice of backend is pedagogical - switching to
``backend="siddon"`` gives a visually equivalent walnut
reconstruction at this geometry; both backends match in amplitude
and produce essentially indistinguishable edge profiles on this
dataset. The real reason to prefer SF here is that its forward
model is cell-integrated and mass-conserving (see the Core
Algorithm section in the README for when that matters). Runtime
is ~50 s on an RTX-class GPU; for a faster 256^3 nominal-voxel
preview flip the settings in step 2 (they are left inline so you
can swap them).

The on-disk preprocessing is a per-projection flat field estimated
from the 99.5th percentile of each raw image - adequate for a demo
and visually tight, but a publication-quality pipeline would use real
flat / dark field scans. See ``examples/data/preprocess_walnut.py``
for the full procedure.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from diffct.differentiable import (
    angular_integration_weights,
    cone_cosine_weights,
    cone_weighted_backproject,
    ramp_filter_1d,
)


DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "walnut_cone.npz"
)


def main():
    # ------------------------------------------------------------------
    # 1. Load the preprocessed walnut sinogram
    # ------------------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Walnut sinogram not found at {DATA_PATH}. "
            "The file should be shipped in the repo at "
            "examples/data/walnut_cone.npz. If missing, regenerate it "
            "from the Zenodo source (https://doi.org/10.5281/zenodo.6986012) "
            "using the tooling described in examples/data/NOTICE."
        )

    data = np.load(DATA_PATH)
    # Stored as float16 to save space; promote to float32 for the
    # CUDA kernels.
    sinogram_np = data["sinogram"].astype(np.float32)
    angles_np = data["angles"].astype(np.float32)
    sdd = float(data["sdd"])
    sid = float(data["sid"])
    du = float(data["du"])
    dv = float(data["dv"])

    # The preprocessing stored ``(views, H, W)`` matching the raw TIFF
    # axis order (H = vertical ~= axial v, W = horizontal ~= in-plane u).
    # diffct's cone API expects ``(views, u, v)``, so transpose here
    # once rather than inside every kernel call.
    sinogram_np = sinogram_np.transpose(0, 2, 1).copy()
    det_u = sinogram_np.shape[1]
    det_v = sinogram_np.shape[2]

    print(
        f"Walnut sinogram: {sinogram_np.shape} "
        f"views={len(angles_np)}  det=({det_u}x{det_v})"
    )
    print(
        f"  sdd={sdd:.2f} mm, sid={sid:.2f} mm, mag={sdd / sid:.2f}x"
    )
    print(f"  binned detector pitch: du={du} mm, dv={dv} mm")

    # ------------------------------------------------------------------
    # 2. Volume geometry
    # ------------------------------------------------------------------
    # The "nominal" voxel size for this scan is the detector pitch
    # divided by the magnification, ``du * sid / sdd = 0.152 mm``. At
    # that voxel size one voxel at isocenter maps to exactly one
    # detector cell, which is what the VD gather backprojector is
    # tuned for.
    #
    # We reconstruct at HALF the nominal voxel size on a 512^3 grid
    # so the volume extent stays at ~39 mm (still enclosing the whole
    # walnut) while the voxel grid is fine enough to resolve the
    # walnut-shell microstructure. Both ``backend="siddon"`` and
    # ``backend="sf_tr"`` give visually equivalent reconstructions
    # at this geometry - the walnut detail you see comes from the
    # fine voxel grid plus the Hamming ramp window, not from the
    # choice of SF vs VD gather. See the "Core Algorithm" section
    # of ``README.md`` for the honest SF-vs-VD discussion.
    voxel_nominal = du * sid / sdd
    voxel_spacing = 0.5 * voxel_nominal          # 0.076 mm, sub-nominal
    Nx = Ny = Nz = 2 * det_u                     # 512

    # ------------------------------------------------------------------
    # 3. Move everything to CUDA
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sinogram = torch.from_numpy(sinogram_np).to(device=device)
    angles = torch.from_numpy(angles_np).to(device=device)

    # ------------------------------------------------------------------
    # 3.5  Pick a backprojector backend
    # ------------------------------------------------------------------
    # ``cone_weighted_backproject`` accepts:
    #
    #   "siddon"           - voxel-driven bilinear gather. Fastest.
    #                        Bilinearly samples the filtered sinogram
    #                        at each voxel's projected ``(u,v)`` and
    #                        accumulates ``(sid/U)^2 * sample``.
    #
    #   "sf_tr"            - LEAP chord-weighted separable-footprint
    #                        gather (matched-adjoint form from
    #                        ``projectors_SF.cu``). Integrates the
    #                        filtered sinogram over each voxel's
    #                        transaxial trapezoidal + axial
    #                        rectangular footprint, weighted by the
    #                        in-plane chord through the unit voxel
    #                        and the ``sqrt(1+(v/sdd)^2)`` axial
    #                        correction. ~5x slower than "siddon".
    #                        Amplitude-matched to Siddon VD on unit-
    #                        density phantoms.
    #
    #   "sf_tt"            - same as "sf_tr" but uses a trapezoidal
    #                        axial footprint built from four z-corner
    #                        projections. Slightly higher axial
    #                        accuracy at extreme cone angles, at
    #                        ~40 % more runtime than "sf_tr".
    #
    # We default to "sf_tr" here so the reader sees the SF path run
    # end-to-end on real data. Switching to "siddon" gives a
    # visually equivalent reconstruction at a fraction of the
    # runtime; pick SF when you want a matched cell-integrated
    # forward model (iterative reco / learned priors).
    projector_backend = "sf_tr"

    # ==================================================================
    # 4. FDK analytical reconstruction
    # ==================================================================

    # --- 4.1  Cosine pre-weight --------------------------------------
    weights = cone_cosine_weights(
        det_u,
        det_v,
        du,
        dv,
        sdd,
        device=device,
        dtype=sinogram.dtype,
    ).unsqueeze(0)
    sino_weighted = sinogram * weights

    # --- 4.2  1D ramp filter along the detector-u direction ----------
    # Using a "hamming" window here instead of "hann": it has a
    # sharper high-frequency rolloff so sub-nominal voxels get to keep
    # more of the high-frequency detail in the reconstruction, at the
    # cost of slightly more ring-like high-frequency noise. Hamming is
    # a good pairing with the SF backprojector; for the VD path,
    # "hann" is usually more forgiving.
    sinogram_filt = ramp_filter_1d(
        sino_weighted,
        dim=1,
        sample_spacing=du,
        pad_factor=2,
        window="hamming",
    ).contiguous()

    # --- 4.3  Per-view angular integration weights -------------------
    # Full 2*pi scan -> use ``redundant_full_scan=True`` to absorb the
    # FDK 1/2 redundancy factor.
    d_beta = angular_integration_weights(
        angles, redundant_full_scan=True
    ).view(-1, 1, 1)
    sinogram_filt = sinogram_filt * d_beta

    # --- 4.4  Voxel-driven FDK backprojection ------------------------
    reconstruction_raw = cone_weighted_backproject(
        sinogram_filt,
        angles,
        Nz,
        Ny,
        Nx,
        du,
        dv,
        sdd,
        sid,
        voxel_spacing=voxel_spacing,
        backend=projector_backend,
    )
    reconstruction = F.relu(reconstruction_raw)

    # ------------------------------------------------------------------
    # 5. Quantitative summary
    # ------------------------------------------------------------------
    print("Walnut FDK reconstruction:")
    print(f"  Backend:            {projector_backend}")
    print(f"  Volume shape:       {tuple(reconstruction.shape)}")
    print(
        f"  Voxel size:         {voxel_spacing:.4f} mm  "
        f"(nominal = {voxel_nominal:.4f} mm)"
    )
    print(f"  Volume extent:      {Nx * voxel_spacing:.1f} mm on a side")
    print(
        "  Reco data range:    "
        f"[{reconstruction.min().item():.4f}, {reconstruction.max().item():.4f}]"
    )

    # ------------------------------------------------------------------
    # 6. Visualization
    # ------------------------------------------------------------------
    reco_cpu = reconstruction.detach().cpu().numpy()      # (Nz, Ny, Nx)
    sino_cpu = sinogram_np                                # (views, u, v)
    num_views = sino_cpu.shape[0]
    mid_view = num_views // 2

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    # Top row: sinogram views at 0 / 90 / 270 degrees.
    for ax, (deg, idx) in zip(
        axes[0],
        [(0, 0), (90, num_views // 4), (270, 3 * num_views // 4)],
    ):
        ax.imshow(sino_cpu[idx].T, cmap="gray", origin="lower")
        ax.set_title(f"Projection ~{deg} deg")
        ax.axis("off")
    # Bottom row: three orthogonal reconstruction slices.
    axes[1, 0].imshow(reco_cpu[Nz // 2, :, :], cmap="gray")
    axes[1, 0].set_title("Axial slice (z = mid)")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(reco_cpu[:, Ny // 2, :], cmap="gray")
    axes[1, 1].set_title("Coronal slice (y = mid)")
    axes[1, 1].axis("off")
    axes[1, 2].imshow(reco_cpu[:, :, Nx // 2], cmap="gray")
    axes[1, 2].set_title("Sagittal slice (x = mid)")
    axes[1, 2].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
