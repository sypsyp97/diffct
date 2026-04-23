"""Preprocess the Helsinki walnut cone-beam dataset into a small .npz.

Zenodo record: https://doi.org/10.5281/zenodo.6986012
Author: Alexander Meaney. License: CC-BY 4.0.

Given the eight ``20201111_walnut_projections_*.zip`` archives from
the Zenodo record, produce a single ``walnut_cone.npz`` that contains

  - a (num_views, det_u, det_v) sinogram of *line integrals* (i.e.
    already flat-field normalised and ``-log``'d),
  - per-view angles in radians,
  - the scan geometry (sdd, sid, detector pixel pitch at the binning
    level chosen below). Angles are stored in diffct's cone-geometry
    handedness, which is the negative of the scanner's reported
    positive-angle convention for this dataset,
  - a short "source" string for attribution.

Usage::

    python preprocess_walnut.py walnut_cone.npz /path/to/scratch/with/zips

Peak memory ~1 GB; final file ~20 MB (at the default 8x bin / 256
crop / every-3rd-view setting).  See ``examples/data/NOTICE`` for the
full attribution.
"""

import argparse
import glob
import os
import zipfile

import numpy as np
import tifffile


# ----- Scan geometry (from 20201111_walnut_.txt in the Zenodo record) -----
SDD_FULL = 553.74           # mm, source-to-detector distance
SID_FULL = 210.66           # mm, source-to-origin (isocenter) distance
DET_PIXEL_FULL = 0.050      # mm, raw detector pixel pitch
NUM_VIEWS_FULL = 721        # projections from 0 to 360 deg in 0.5 deg steps
ANGLE_FIRST_DEG = 0.0
ANGLE_INTERVAL_DEG = 0.5
# The scanner reports increasing angles in the opposite handedness to
# diffct's cone geometry convention. Store angles in diffct coordinates.
ANGLE_DIRECTION = -1.0

# ----- Default preprocessing knobs -----
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


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess Helsinki walnut cone-beam projections."
    )
    parser.add_argument(
        "out_path",
        nargs="?",
        default="walnut_cone.npz",
        help="Output .npz path.",
    )
    parser.add_argument(
        "zip_dir",
        nargs="?",
        default="/tmp/walnut_dl",
        help="Directory containing the Zenodo projection ZIP archives.",
    )
    parser.add_argument(
        "--bin",
        type=int,
        default=BIN,
        help="Detector mean-pooling factor. Use 1 for native detector pitch.",
    )
    parser.add_argument(
        "--crop",
        type=int,
        default=CROP,
        help="Square center crop after binning. Use 0 to keep the full binned detector.",
    )
    parser.add_argument(
        "--view-stride",
        type=int,
        default=VIEW_STRIDE,
        help="Keep every Nth projection. Use 1 for all 721 views.",
    )
    parser.add_argument(
        "--angle-direction",
        choices=("-1", "1"),
        default=str(int(ANGLE_DIRECTION)),
        help="Scanner-to-diffct angular handedness. The Helsinki walnut scan uses -1.",
    )
    parser.add_argument(
        "--flat-percentile",
        type=float,
        default=FLAT_PERCENTILE,
        help="Per-projection bright-field percentile.",
    )
    parser.add_argument(
        "--storage-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Sinogram dtype stored in the .npz.",
    )
    parser.add_argument(
        "--uncompressed",
        action="store_true",
        help="Use np.savez instead of np.savez_compressed.",
    )
    return parser.parse_args()


def _projection_zips(zip_dir):
    patterns = (
        "20201111_walnut_projections_*.zip",
        "projections_*.zip",
    )
    paths = []
    seen = set()
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(zip_dir, pattern))):
            norm = os.path.abspath(path)
            if norm not in seen:
                seen.add(norm)
                paths.append(path)
    return paths


def main():
    args = _parse_args()
    if args.bin < 1:
        raise ValueError('--bin must be >= 1')
    if args.crop < 0:
        raise ValueError('--crop must be >= 0')
    if args.view_stride < 1:
        raise ValueError('--view-stride must be >= 1')

    zips = _projection_zips(args.zip_dir)
    print(f'Found {len(zips)} ZIPs in {args.zip_dir}')

    num_kept = (NUM_VIEWS_FULL + args.view_stride - 1) // args.view_stride
    binH = 2368 // args.bin
    binW = 2240 // args.bin
    cropH = binH if args.crop == 0 else args.crop
    cropW = binW if args.crop == 0 else args.crop
    h0 = (binH - cropH) // 2
    w0 = (binW - cropW) // 2
    assert h0 >= 0 and w0 >= 0, (
        f'Binned shape {binH}x{binW} smaller than crop {cropH}x{cropW}'
    )

    # Two arrays built up during the stream pass:
    #   ``cropped``  : binned + center-cropped intensity (not yet -log'd),
    #                  stored as float32 for accuracy during normalisation.
    #   ``i0_per``   : per-projection bright reference (air estimate).
    cropped = np.zeros((num_kept, cropH, cropW), dtype=np.float32)
    i0_per = np.zeros((num_kept,), dtype=np.float32)
    mask = np.zeros((num_kept,), dtype=bool)

    for zi, zp in enumerate(zips):
        print(f'[{zi + 1}/{len(zips)}] {os.path.basename(zp)}')
        for idx, raw in _iter_zip_projections(zp):
            if idx % args.view_stride != 0:
                continue
            kept_i = idx // args.view_stride
            if kept_i >= num_kept:
                continue
            # Per-projection flat field: use a high percentile of the
            # *raw* detector values, which approximates the unattenuated
            # air reading for this exact projection.  Using a percentile
            # (not the max) makes it robust to hot pixels; using it per-
            # projection absorbs any slow tube-intensity drift between
            # views.  We still assume a zero dark field because the
            # dataset does not ship one.
            i0_per[kept_i] = np.percentile(
                raw.astype(np.float32),
                args.flat_percentile,
            )
            binned = _bin_block_mean(raw, args.bin)
            cropped[kept_i] = binned[h0:h0 + cropH, w0:w0 + cropW]
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

    # Physical angles in radians. The source metadata says 721 images from
    # 0 to 360 deg in 0.5 deg increments, i.e. the endpoint is included.
    view_indices = np.arange(NUM_VIEWS_FULL, dtype=np.int32)
    angle_direction = float(args.angle_direction)
    angles_all = np.deg2rad(
        angle_direction * (ANGLE_FIRST_DEG + ANGLE_INTERVAL_DEG * view_indices)
    ).astype(np.float32)
    angles = angles_all[::args.view_stride][:num_kept]
    kept_indices = view_indices[::args.view_stride][:num_kept]

    # Binned pixel pitch
    du = dv = DET_PIXEL_FULL * args.bin

    print(f'Binned pixel: {du} mm  det=({cropW} x {cropH})')
    print(f'Angles: {angles.shape}  [{angles[0]:.3f}, {angles[-1]:.3f}] rad')

    # Store the sinogram as float16: the -log values live in
    # [0, ~2] so float16 precision (~1e-4) is more than enough for FDK
    # reconstruction, and it halves the file size.
    save = np.savez if args.uncompressed else np.savez_compressed
    save(
        args.out_path,
        sinogram=sinogram.astype(args.storage_dtype),
        angles=angles,
        view_indices=kept_indices,
        sdd=np.float32(SDD_FULL),
        sid=np.float32(SID_FULL),
        du=np.float32(du),
        dv=np.float32(dv),
        detector_u=np.int32(cropW),
        detector_v=np.int32(cropH),
        view_stride=np.int32(args.view_stride),
        detector_bin=np.int32(args.bin),
        crop_h=np.int32(cropH),
        crop_w=np.int32(cropW),
        storage_dtype=args.storage_dtype,
        i0_mean=np.float32(i0_per.mean()),
        source='Meaney 2022, Zenodo 6986012, CC-BY 4.0',
    )
    size_mb = os.path.getsize(args.out_path) / (1024 * 1024)
    mode = 'uncompressed' if args.uncompressed else 'compressed'
    print(f'Wrote {args.out_path} ({size_mb:.2f} MB, {mode}, {args.storage_dtype})')


if __name__ == '__main__':
    main()
