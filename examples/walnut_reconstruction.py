"""Reconstruct a real walnut from measured cone-beam projections.

The data are 240 measured projections of a walnut on a full circular orbit
(Meaney 2022, Zenodo 6986012, CC-BY 4.0; see examples/data/NOTICE). The scan
geometry is passed to Projector as explicit per-view tensors, the same way as
a calibrated trajectory. FDK, SIRT, CGLS and TV-regularized least squares all
use every measured view:

    python examples/walnut_reconstruction.py --figure walnut.png
    python examples/walnut_reconstruction.py --size 512 --devices 0 1 2 3
    python examples/walnut_reconstruction.py --size 128 --algorithms --save-volume walnut128.npy

``--save-volume`` stores the FDK volume scaled to [0, 1]; pass it to
``iterative_reconstruction.py --phantom`` to simulate other trajectories.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from diffct import (Projector, angular_integration_weights, cone_cosine_weights,
                    cone_weighted_backproject, ramp_filter_1d)
from _common import cgls, save_slices, sirt, tv_reconstruction

DATA = Path(__file__).parent / "data" / "walnut_cone.npz"
ALGORITHMS = {"sirt": sirt, "cgls": cgls, "tv": tv_reconstruction}
DEFAULT_ITERATIONS = {"sirt": 200, "cgls": 20, "tv": 300}


def load_scan(path):
    """Measured line integrals as ``(views, U, V)`` plus the circular scan geometry in mm."""
    data = np.load(path)
    # Stored as (views, rows, columns): columns run along u, rows along v.
    sinogram = data["sinogram"].astype(np.float32).transpose(0, 2, 1)
    angles = data["angles"].astype(np.float64)
    if math.isclose(abs(angles[-1] - angles[0]), 2 * math.pi, abs_tol=1e-4):
        sinogram, angles = sinogram[:-1], angles[:-1]  # the last view repeats the first
    return sinogram, angles, float(data["sid"]), float(data["sdd"]), float(data["du"]), float(data["dv"])


def circular_geometry(angles, sid, sdd):
    """Source, detector centre and detector axes for each measured angle (rotation about z)."""
    a = torch.as_tensor(angles, dtype=torch.float64)
    cos, sin, zero = torch.cos(a), torch.sin(a), torch.zeros_like(a)
    source = torch.stack((-sid * sin, sid * cos, zero), dim=1)
    detector = torch.stack(((sdd - sid) * sin, -(sdd - sid) * cos, zero), dim=1)
    u_axis = torch.stack((cos, sin, zero), dim=1)
    v_axis = torch.stack((zero, zero, torch.ones_like(a)), dim=1)
    return tuple(t.float() for t in (source, detector, u_axis, v_axis))


def fdk(sinogram, angles, trajectory, du, dv, sdd, size, voxel, window):
    """Circular-orbit FDK: cosine weights, ramp filter along u, angular weights, weighted backprojection."""
    device = sinogram.device
    n_u, n_v = sinogram.shape[1:]
    weights = cone_cosine_weights(n_u, n_v, du, dv, sdd, device=device).unsqueeze(0)
    filtered = ramp_filter_1d(sinogram * weights, dim=1, sample_spacing=du, pad_factor=2, window=window)
    angle_weights = angular_integration_weights(torch.as_tensor(angles, dtype=torch.float32, device=device))
    filtered = (filtered * angle_weights.view(-1, 1, 1)).contiguous()
    return cone_weighted_backproject(filtered, *[t.to(device) for t in trajectory], size, size, size,
                                     du, dv, voxel_spacing=voxel).clamp_min(0)


def shell_scale(volume):
    """Factor that maps the 99.95th percentile of ``volume`` (the walnut shell) to 1."""
    flat = volume.flatten()
    # torch.quantile accepts at most 2**24 elements; subsample larger volumes.
    stride = max(7, flat.numel() // 2**24 + 1)
    return 1.0 / torch.quantile(flat[::stride], 0.9995).item()


def timed(function):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--size", type=int, default=256, help="volume edge length; the field of view stays 39 mm")
    parser.add_argument("--window", default="hann", choices=["ram-lak", "shepp-logan", "cosine", "hamming", "hann"])
    parser.add_argument("--algorithms", nargs="*", choices=list(ALGORITHMS), default=list(ALGORITHMS),
                        help="iterative solvers after FDK; give none to run FDK only")
    parser.add_argument("--iterations", type=int, help="override the per-algorithm default " + str(DEFAULT_ITERATIONS))
    parser.add_argument("--tv-weight", type=float, default=0.3)
    parser.add_argument("--devices", type=int, nargs="+", help="GPUs for the iterative solvers in one process")
    parser.add_argument("--figure", type=Path, help="save axial and coronal central slices to this PNG")
    parser.add_argument("--save-volume", type=Path, help="save the FDK volume, scaled to [0, 1], as .npy")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("diffct needs a CUDA GPU")

    sinogram, angles, sid, sdd, du, dv = load_scan(args.data)
    detector = sinogram.shape[1:]
    # Nominal voxel: one detector cell projected to the isocentre; the default 256^3 grid spans 39 mm.
    voxel = 256 * du * sid / sdd / args.size
    measured = torch.from_numpy(sinogram).cuda()
    trajectory = circular_geometry(angles, sid, sdd)
    print(f"walnut: {len(angles)} measured views, detector {tuple(detector)}, sid {sid:.2f} mm, "
          f"sdd {sdd:.2f} mm; volume {args.size}^3 at {voxel:.4f} mm")

    def run_fdk():
        return fdk(measured, angles, trajectory, du, dv, sdd, args.size, voxel, args.window)

    run_fdk()  # compile kernels and warm up
    reconstruction, seconds = timed(run_fdk)
    print(f"  fdk  {1e3 * seconds:9.1f} ms ({args.window} window)")
    # Scale attenuation so the walnut shell is about 1; the iterative solvers' defaults assume that range.
    scale = shell_scale(reconstruction)
    panels = [(f"FDK\n{args.window} window", reconstruction * scale)]
    if args.save_volume:
        np.save(args.save_volume, panels[0][1].clamp(0, 1).cpu().numpy())
        print("saved", args.save_volume)

    if args.algorithms:
        operator = Projector(trajectory, (args.size,) * 3, detector, detector_spacing=(du, dv),
                             voxel_spacing=voxel, devices=args.devices)
        target = measured * scale
        for name in args.algorithms:
            iterations = args.iterations or DEFAULT_ITERATIONS[name]
            options = {"weight": args.tv_weight} if name == "tv" else {}
            ALGORITHMS[name](operator, target, 1, **options)  # compile kernels and warm up
            reconstruction, seconds = timed(lambda: ALGORITHMS[name](operator, target, iterations, **options))
            print(f"  {name:4s} {1e3 * seconds:9.1f} ms ({1e3 * seconds / iterations:7.2f} ms/it, {iterations} it)")
            label = f"TV + Adam\n{iterations} it, weight {args.tv_weight:g}" if name == "tv" else f"{name.upper()}\n{iterations} it"
            panels.append((label, reconstruction))

    if args.figure:
        save_slices(args.figure, panels, f"Measured walnut: {len(angles)} views, circular cone beam, {args.size}³",
                    vmax=1.0, sections=("axial", "coronal"))


if __name__ == "__main__":
    main()
