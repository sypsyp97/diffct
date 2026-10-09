"""Inputs for docs/video/diffct_intro.py.

Run on a CUDA GPU from the repository root with ``python docs/video/make_inputs.py --gpu``:

walnut_measured.npz: axial and coronal central slices of the measured walnut
(examples/data/walnut_cone.npz, all 240 views) reconstructed by FDK, SIRT,
CGLS and TV, plus a few measured projections.
recon_slices.npz, recon_psnr.json: the FDK walnut volume used as a phantom for a
simulated helical scan with 1% noise, and its FDK / SIRT / CGLS / TV
reconstructions with PSNR.

Without ``--gpu`` the script only rebuilds data2d.npz: the walnut axial slice
from walnut_measured.npz and its parallel-beam sinogram over 180 degrees
(scipy rotate + sum, for display only). The script does not write
calib_history.json: that file is the loss history of a geometry calibration
run (64^3, 150 steps) and is kept as it is.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent / "examples"))

SIZE = 256
HELICAL_VIEWS = 720
NOISE = 0.01


def sections(volume):
    from _common import central_section
    return {"axial": central_section(volume, "axial").cpu().numpy(),
            "coronal": central_section(volume, "coronal").cpu().numpy()}


def measured():
    import torch
    from diffct import Projector
    from _common import cgls, sirt, tv_reconstruction
    from walnut_reconstruction import DATA, circular_geometry, fdk, load_scan, shell_scale
    sinogram, angles, sid, sdd, du, dv, offset_u, offset_v = load_scan(DATA)
    voxel = 256 * du * sid / sdd / SIZE
    y = torch.from_numpy(sinogram).cuda()
    trajectory = circular_geometry(angles, sid, sdd, offset_u, offset_v)
    full = fdk(y, angles, trajectory, du, dv, offset_u, offset_v, sdd, SIZE, voxel, "hann")
    scale = shell_scale(full)
    operator = Projector(trajectory, (SIZE,) * 3, sinogram.shape[1:], detector_spacing=(du, dv), voxel_spacing=voxel)
    target = y * scale
    results = {"fdk": full * scale, "sirt": sirt(operator, target, 200), "cgls": cgls(operator, target, 20),
               "tv": tv_reconstruction(operator, target, 300, weight=0.3)}
    arrays = {f"{k}_{s}": v for k, volume in results.items() for s, v in sections(volume).items()}
    picks = np.linspace(0, len(angles), 4, endpoint=False).astype(int)
    np.savez_compressed(HERE / "walnut_measured.npz", projections=sinogram[picks], **arrays)
    return results["fdk"].clamp(0, 1)


def simulated(truth):
    import torch
    from diffct import Projector
    from _common import Scan, cgls, fdk, psnr, sirt, tv_reconstruction
    scan = Scan(SIZE, HELICAL_VIEWS)
    trajectory = scan.trajectory("helical")
    operator = Projector(trajectory, (SIZE,) * 3, scan.detector, detector_spacing=scan.pitch)
    y = operator.project(truth)
    generator = torch.Generator(device="cuda").manual_seed(1234)
    y = y + NOISE * y.max() * torch.randn(y.shape, generator=generator, device="cuda")
    results = {"fdk": fdk(scan, y, trajectory), "sirt": sirt(operator, y, 200),
               "cgls": cgls(operator, y, 30), "tv": tv_reconstruction(operator, y, 200, weight=1.0)}
    arrays = {f"{k}_{s}": v for k, volume in {"phantom": truth, **results}.items() for s, v in sections(volume).items()}
    np.savez_compressed(HERE / "recon_slices.npz", **arrays)
    (HERE / "recon_psnr.json").write_text(json.dumps({k: psnr(v, truth) for k, v in results.items()}))


def phantom_and_sinogram():
    from scipy.ndimage import rotate
    image = np.load(HERE / "walnut_measured.npz")["fdk_axial"]
    angles = np.linspace(0, 180, 180, endpoint=False)
    sinogram = np.stack([rotate(image, -a, reshape=False, order=1).sum(axis=0) for a in angles])
    np.savez_compressed(HERE / "data2d.npz", phantom=image, sinogram=sinogram.astype(np.float32), angles=angles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", action="store_true", help="also recompute the reconstructions")
    args = parser.parse_args()
    if args.gpu:
        simulated(measured())
    phantom_and_sinogram()
