"""Reconstruct and assemble the measured helical README comparison.

Prepare the case with ``ldct_prep.py`` first. Reconstruct one method per run:

    python docs/assets/gallery/clinical_methods.py reconstruct adam --case DIR/L213 --out OUT
    python docs/assets/gallery/clinical_methods.py reconstruct sirt --case DIR/L213 --out OUT
    python docs/assets/gallery/clinical_methods.py reconstruct cgls --case DIR/L213 --out OUT
    python docs/assets/gallery/clinical_methods.py reconstruct tv --case DIR/L213 --out OUT

Then assemble the display arrays:

    python docs/assets/gallery/clinical_methods.py assemble --case DIR/L213 --out OUT

The Adam least-squares reconstruction is the unregularized limit of
``tv_reconstruction``. FDK is not included: its slice-windowed helical
approximation gives strong z-band artifacts for this measured scan.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from diffct import ClinicalScan, Projector, cgls, clinical_geometry, sirt, tv_reconstruction

CENTER_COLUMN = 442.2
Z_VOXEL = 3.0
XY_VOXEL = 0.652344
XY_SIZE = 768


def half(x):
    return x.detach().float().cpu().numpy().astype(np.float16)


def load_case(case):
    m = dict(np.load(case / "meta.npz"))
    proj = np.load(case / "proj.npy", mmap_mode="r")
    scan = ClinicalScan(
        projections=proj, angle=m["phi"], z=m["z"], radius=m["rho"], sdd=m["sdd"],
        focal_angle_offset=m["dphi"], focal_z_offset=m["dz"], focal_radius_offset=m["drho"],
        instance=m["instance"], detector_spacing=(float(m["du"][0]), float(m["dv"][0])),
        central_element=(float(m["ce_col"][0]) - 1, float(m["ce_row"][0]) - 1),
        detector_shape=str(m["detector_shape"]), ffs_mode=str(m["ffs"]),
        water_attenuation=float(m["water_attenuation"]), manufacturer=str(m["manufacturer"]))
    return scan, proj, m


def build(case):
    scan, proj, m = load_case(case)
    z = m["z"]
    n_rows, dv, rho, sdd = proj.shape[2], float(m["dv"][0]), float(m["rho"][0]), float(m["sdd"][0])
    margin = math.ceil(n_rows / 2 * dv * (rho + 250) / sdd + np.abs(m["dz"]).max())
    mid = 0.5 * (z.min() + z.max())
    depth = int(math.ceil((z.max() - z.min() + 2 * margin) / Z_VOXEL))
    geometry = clinical_geometry(scan, z_center=mid, central_column=CENTER_COLUMN)
    operator = Projector(geometry.trajectory, (depth, XY_SIZE, XY_SIZE), geometry.detector_shape,
                         detector_spacing=geometry.detector_spacing,
                         voxel_spacing=(Z_VOXEL, XY_VOXEL, XY_VOXEL),
                         detector_surface=geometry.detector_surface)
    return operator, proj, scan, mid, depth


def reconstruct(args):
    case, out = Path(args.case), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    operator, proj, scan, mid, depth = build(case)
    measurements = torch.from_numpy(np.ascontiguousarray(proj)).cuda().div_(scan.water_attenuation)
    settings = {
        "adam": (tv_reconstruction, 150, {"weight": 0.0, "lr": 0.1}),
        "sirt": (sirt, 100, {}),
        "cgls": (cgls, 15, {}),
        "tv": (tv_reconstruction, 150, {"weight": 3.0, "lr": 0.1}),
    }
    solver, iterations, options = settings[args.method]
    start = time.time()
    result = solver(operator, measurements, iterations, **options)
    torch.cuda.synchronize()
    volume = result.volume
    axial_index = int(round((-364.0 - mid) / Z_VOXEL + depth / 2 - 0.5))
    arrays = {
        "axial": half(volume[axial_index]),
        "coronal": half(volume[:, volume.shape[1] // 2, :]),
        "sagittal": half(volume[:, :, volume.shape[2] // 2]),
        "residual": np.asarray(result.residual_history, dtype=np.float64),
        "iterations": np.asarray(result.iterations), "seconds": np.asarray(time.time() - start),
        "voxel_spacing": np.asarray([Z_VOXEL, XY_VOXEL, XY_VOXEL]),
    }
    target = out / f"clinical_{args.method}.npz"
    temporary = out / f"clinical_{args.method}.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.rename(target)
    print("saved", target, flush=True)


def orient_axial_recon(array):
    return array[::-1, ::-1]


def assemble(args):
    case, out = Path(args.case), Path(args.out)
    ref = np.load(case / "ref.npz")
    hu = np.clip(1 + ref["hu"] / 1000, 0, None).astype(np.float32)
    axial_index = int(np.argmin(np.abs(ref["z"] + 363)))
    zref, zrec, xrec = slice(18, 64), slice(35, 81), slice(128, 640)

    def reference_plane(array):
        return array[zref][::-1]

    def recon_plane(array):
        return array[:, xrec][zrec][:, ::-1][::-1]

    arrays = {
        "reference_axial": hu[axial_index],
        "reference_coronal": reference_plane(hu[:, 256, :]),
        "reference_sagittal": reference_plane(hu[:, :, 256]),
    }
    for method in ("adam", "sirt", "cgls", "tv"):
        data = np.load(out / f"clinical_{method}.npz")
        arrays[f"{method}_axial"] = orient_axial_recon(data["axial"][128:640, 128:640])
        arrays[f"{method}_coronal"] = recon_plane(data["coronal"])
        arrays[f"{method}_sagittal"] = recon_plane(data["sagittal"])
    arrays["voxel_spacing"] = np.asarray([Z_VOXEL, XY_VOXEL, XY_VOXEL])
    np.savez_compressed(out / "helical_methods.npz", **{k: v.astype(np.float16) for k, v in arrays.items()})
    print("saved", out / "helical_methods.npz", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("reconstruct", "assemble"):
        p = sub.add_parser(name)
        p.add_argument("--case", required=True)
        p.add_argument("--out", required=True)
        if name == "reconstruct":
            p.add_argument("method", choices=("adam", "sirt", "cgls", "tv"))
    args = parser.parse_args()
    {"reconstruct": reconstruct, "assemble": assemble}[args.command](args)


if __name__ == "__main__":
    main()
