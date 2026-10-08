"""Inputs for docs/video/diffct_intro.py.

data2d.npz: central slice of the 3D Shepp-Logan phantom and its parallel-beam
sinogram over 180 degrees (scipy rotate + sum, for display only).
recon_slices.npz, recon_psnr.json: central slices and PSNR of real diffct
reconstructions (128^3 helical scan, FDK / SIRT 200 / CGLS 30 / TV 200); run on a
CUDA GPU from the repository root with ``python docs/video/make_inputs.py --gpu``.
calib_history.json: loss history of a geometry calibration run (64^3, 150 steps).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent / "examples"))


def phantom_and_sinogram():
    from scipy.ndimage import rotate
    from _common import shepp_logan_3d
    phantom = shepp_logan_3d(128)[64]
    angles = np.linspace(0, 180, 180, endpoint=False)
    sinogram = np.stack([rotate(phantom, -a, reshape=False, order=1).sum(axis=0) for a in angles])
    np.savez_compressed(HERE / "data2d.npz", phantom=phantom, sinogram=sinogram.astype(np.float32), angles=angles)


def reconstructions():
    import torch
    from diffct import Projector
    from _common import Scan, cgls, fdk, psnr, shepp_logan_3d, sirt, tv_reconstruction
    scan = Scan(128, 360)
    trajectory = scan.trajectory("helical")
    operator = Projector(trajectory, (128,) * 3, scan.detector, detector_spacing=scan.pitch)
    truth = torch.from_numpy(shepp_logan_3d(128)).cuda()
    y = operator.project(truth)
    results = {"fdk": fdk(scan, y, trajectory), "sirt": sirt(operator, y, 200),
               "cgls": cgls(operator, y, 30), "tv": tv_reconstruction(operator, y, 200)}
    np.savez_compressed(HERE / "recon_slices.npz", phantom=truth[64].cpu().numpy(),
                        **{k: v[64].cpu().numpy() for k, v in results.items()})
    (HERE / "recon_psnr.json").write_text(json.dumps({k: psnr(v, truth) for k, v in results.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", action="store_true", help="also recompute the reconstructions")
    args = parser.parse_args()
    phantom_and_sinogram()
    if args.gpu:
        reconstructions()
