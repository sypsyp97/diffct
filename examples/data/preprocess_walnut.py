"""Preprocess the Helsinki walnut cone-beam dataset into a small .npz.

Zenodo record: https://doi.org/10.5281/zenodo.6986012
Author: Alexander Meaney. License: CC-BY 4.0.

Given the eight ``20201111_walnut_projections_*.zip`` archives from
the Zenodo record, produce a single ``walnut_cone.npz`` that contains

  - a (num_views, det_u, det_v) sinogram of *line integrals* (i.e.
    already flat-field normalised and ``-log``'d),
  - per-view angles in radians,
  - the scan geometry (sdd, sid, detector pixel pitch at the binning
    level chosen below),
  - a short "source" string for attribution.

Usage::

    python preprocess_walnut.py walnut_cone.npz /path/to/scratch/with/zips

Peak memory ~1 GB; final file ~20 MB (at the default 8x bin / 256
crop / every-3rd-view setting).  See ``examples/data/NOTICE`` for the
full attribution.
"""

import glob
import os
import sys
import zipfile

import numpy as np
import tifffile


# ----- Scan geometry (from 20201111_walnut_.txt in the Zenodo record) -----
SDD_FULL = 553.74           # mm, source-to-detector distance
SID_FULL = 210.66           # mm, source-to-origin (isocenter) distance
DET_PIXEL_FULL = 0.050      # mm, raw detector pixel pitch
NUM_VIEWS_FULL = 721        # projections from 0 to 360 deg in 0.5 deg steps

# ----- Preprocessing knobs -----
# Binning 8x on a 2368x2240 detector gives 296x280; center-cropping
# that to 256x256 preserves the walnut-filling portion of the detector
# while dropping the empty-air borders.  Dropping to every 3rd view
# (241 of 721) is plenty for a non-aliasing FDK reconstruction of a
# ~40 mm object, and float16 halves the storage.
BIN = 8
CROP = 256
VIEW_STRIDE = 3
FLAT_PERCENTILE = 99.5      # per-projection "air" estimate, in pct
EPS_PHOTONS = 1.0           # clamp floor before -log


def _iter_zip_projections(zip_path):
    """Yield (0-based global index, raw uint16 image) for each TIFF."""
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith('.tif'))
        for n in names:
            stem = os.path.splitext(os.path.basename(n))[0]
            idx = int(stem.split('_')[-1]) - 1              # 1-based -> 0-based
            with z.open(n) as f:
                yield idx, tifffile.imread(f)


def _bin_block_mean(img, b):
    """Mean-pool ``img`` by a factor of ``b`` on each axis."""
    H, W = img.shape
    Hc = (H // b) * b
    Wc = (W // b) * b
    view = img[:Hc, :Wc].astype(np.float32)
    return view.reshape(Hc // b, b, Wc // b, b).mean(axis=(1, 3))


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'walnut_cone.npz'
    zip_dir = sys.argv[2] if len(sys.argv) > 2 else '/tmp/walnut_dl'
    zips = sorted(glob.glob(os.path.join(zip_dir, 'projections_*.zip')))
    print(f'Found {len(zips)} ZIPs in {zip_dir}')

    num_kept = (NUM_VIEWS_FULL + VIEW_STRIDE - 1) // VIEW_STRIDE
    binH = 2368 // BIN
    binW = 2240 // BIN
    h0 = (binH - CROP) // 2
    w0 = (binW - CROP) // 2
    assert h0 >= 0 and w0 >= 0, (
        f'Binned shape {binH}x{binW} smaller than crop {CROP}'
    )

    # Two arrays built up during the stream pass:
    #   ``cropped``  : binned + center-cropped intensity (not yet -log'd),
    #                  stored as float32 for accuracy during normalisation.
    #   ``i0_per``   : per-projection bright reference (air estimate).
    cropped = np.zeros((num_kept, CROP, CROP), dtype=np.float32)
    i0_per = np.zeros((num_kept,), dtype=np.float32)
    mask = np.zeros((num_kept,), dtype=bool)

    for zi, zp in enumerate(zips):
        print(f'[{zi + 1}/{len(zips)}] {os.path.basename(zp)}')
        for idx, raw in _iter_zip_projections(zp):
            if idx % VIEW_STRIDE != 0:
                continue
            kept_i = idx // VIEW_STRIDE
            if kept_i >= num_kept:
                continue
            # Per-projection flat field: use a high percentile of the
            # *raw* detector values, which approximates the unattenuated
            # air reading for this exact projection.  Using a percentile
            # (not the max) makes it robust to hot pixels; using it per-
            # projection absorbs any slow tube-intensity drift between
            # views.  We still assume a zero dark field because the
            # dataset does not ship one.
            i0_per[kept_i] = np.percentile(raw.astype(np.float32),
                                           FLAT_PERCENTILE)
            binned = _bin_block_mean(raw, BIN)
            cropped[kept_i] = binned[h0:h0 + CROP, w0:w0 + CROP]
            mask[kept_i] = True

    if not mask.all():
        missing = np.where(~mask)[0]
        print(f'WARNING: missing views: {missing[:10]}... ({len(missing)})')

    print(f'Binned+cropped stack: {cropped.shape} {cropped.dtype}')
    print(
        f'Per-projection I0: min={i0_per.min():.1f} '
        f'max={i0_per.max():.1f} mean={i0_per.mean():.1f}'
    )

    # Flat-field + Beer-Lambert inversion.  The binned ``cropped`` values
    # are mean counts per 8x8 block, so they live in the same units as
    # ``i0_per`` (also a mean of raw counts) and the ratio is unitless.
    i0 = i0_per[:, None, None]
    ratio = np.clip(cropped / i0, EPS_PHOTONS / i0, 1.0)
    sinogram = -np.log(ratio)

    print(f'Sinogram: {sinogram.shape} {sinogram.dtype}')
    print(
        f'Range: [{sinogram.min():.3f}, {sinogram.max():.3f}]  '
        f'mean={sinogram.mean():.3f}'
    )

    # Physical angles in radians.
    angles_all = np.linspace(
        0.0, 2.0 * np.pi, NUM_VIEWS_FULL, endpoint=False, dtype=np.float32
    )
    angles = angles_all[::VIEW_STRIDE][:num_kept]

    # Binned pixel pitch
    du = dv = DET_PIXEL_FULL * BIN

    print(f'Binned pixel: {du} mm  det=({CROP} x {CROP})')
    print(f'Angles: {angles.shape}  [{angles[0]:.3f}, {angles[-1]:.3f}] rad')

    # Store the sinogram as float16: the -log values live in
    # [0, ~2] so float16 precision (~1e-4) is more than enough for FDK
    # reconstruction, and it halves the file size.
    np.savez_compressed(
        out_path,
        sinogram=sinogram.astype(np.float16),
        angles=angles,
        sdd=np.float32(SDD_FULL),
        sid=np.float32(SID_FULL),
        du=np.float32(du),
        dv=np.float32(dv),
        detector_u=np.int32(CROP),
        detector_v=np.int32(CROP),
        i0_mean=np.float32(i0_per.mean()),
        source='Meaney 2022, Zenodo 6986012, CC-BY 4.0',
    )
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f'Wrote {out_path} ({size_mb:.2f} MB)')


if __name__ == '__main__':
    main()
