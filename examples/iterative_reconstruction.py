"""Iterative cone-beam reconstruction of a 3D phantom on any trajectory.

Runs CGLS, SIRT and TV-regularized nonnegative least squares (Adam through
autograd), and FDK as a baseline when the full sinogram is on one GPU. The same
script runs on one GPU, on several GPUs in one process, or on several processes
and nodes:

    python examples/iterative_reconstruction.py --trajectory helical
    python examples/iterative_reconstruction.py --devices 0 1 2 3
    torchrun --nproc-per-node=4 examples/iterative_reconstruction.py
    (multiple nodes: see examples/slurm/multi_node.sbatch)

The phantom is the 3D Shepp-Logan by default. ``--phantom`` loads a ``(size,)*3``
volume with values in [0, 1] instead, for example the walnut volume written by
``walnut_reconstruction.py --save-volume``.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from diffct import Projector
from _common import (TRAJECTORIES, Scan, cgls, fdk, init_process, psnr, save_slices,
                     shepp_logan_3d, sirt, synchronize, tv_reconstruction)

ALGORITHMS = {"cgls": cgls, "sirt": sirt, "tv": tv_reconstruction}
# CGLS converges fastest; SIRT and Adam need more, cheaper-to-tune steps.
DEFAULT_ITERATIONS = {"cgls": 30, "sirt": 200, "tv": 200}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=128, help="volume edge length in voxels")
    parser.add_argument("--views", type=int, default=360)
    parser.add_argument("--trajectory", choices=TRAJECTORIES, default="helical")
    parser.add_argument("--algorithms", nargs="+", choices=list(ALGORITHMS), default=list(ALGORITHMS))
    parser.add_argument("--iterations", type=int, help="override the per-algorithm default "
                        + str(DEFAULT_ITERATIONS))
    parser.add_argument("--devices", type=int, nargs="+", help="GPUs for one process (not with torchrun)")
    parser.add_argument("--tv-weight", type=float, default=1.0, help="TV penalty weight for the tv algorithm")
    parser.add_argument("--noise", type=float, default=0.0, help="Gaussian noise, relative to the largest projection")
    parser.add_argument("--figure", type=Path, help="save central slices to this PNG")
    parser.add_argument("--phantom", type=Path, help="(size,)*3 .npy volume in [0, 1] instead of Shepp-Logan")
    args = parser.parse_args()

    distributed, device = init_process()
    if distributed and args.devices:
        parser.error("use either torchrun or --devices, not both")
    scan = Scan(args.size, args.views)
    trajectory = scan.trajectory(args.trajectory)
    # One Projector serves every mode: devices=... splits views over local GPUs,
    # distributed=True splits them over torchrun ranks.
    operator = Projector(trajectory, (args.size,) * 3, scan.detector, detector_spacing=scan.pitch,
                         devices=args.devices, distributed=distributed)
    rank0 = operator.rank == 0
    if args.phantom:
        volume = np.load(args.phantom).astype(np.float32)
        if volume.shape != (args.size,) * 3 or not (np.isfinite(volume).all() and 0 <= volume.min() and volume.max() <= 1):
            raise ValueError(f"{args.phantom} must hold a {args.size}^3 volume with values in [0, 1]")
        truth = torch.from_numpy(volume).to(device)
    else:
        truth = torch.from_numpy(shepp_logan_3d(args.size)).to(device)
    phantom_name = args.phantom.stem if args.phantom else "Shepp-Logan"

    measurements = operator.project(truth)  # this rank's views
    if args.noise > 0:
        # Same noise for every launch mode: global peak, noise drawn for all views.
        peak = measurements.max()
        if distributed:
            torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX, group=operator.process_group)
        generator = torch.Generator(device=device).manual_seed(1234)
        noise = torch.randn((args.views, *scan.detector), generator=generator, device=device)
        measurements = measurements + args.noise * peak * noise[operator.view_slice]

    if rank0:
        print(f"{args.trajectory} trajectory, {args.size}^3 volume, {args.views} views, "
              f"{operator.world_size} process(es), devices per process: {args.devices or [device.index]}")
    panels = [("phantom\nground truth", truth)]
    if not distributed:
        fdk(scan, measurements, trajectory)  # compile kernels and warm up
        synchronize(distributed)
        start = time.perf_counter()
        reconstruction = fdk(scan, measurements, trajectory)
        synchronize(distributed)
        print(f"  fdk   {1e3 * (time.perf_counter() - start):8.1f} ms            PSNR {psnr(reconstruction, truth):6.2f} dB")
        panels.append((f"FDK\n{psnr(reconstruction, truth):.1f} dB", reconstruction))

    for name in args.algorithms:
        iterations = args.iterations or DEFAULT_ITERATIONS[name]
        options = {"weight": args.tv_weight} if name == "tv" else {}
        ALGORITHMS[name](operator, measurements, 1, **options)  # compile kernels and warm up
        synchronize(distributed)
        start = time.perf_counter()
        reconstruction = ALGORITHMS[name](operator, measurements, iterations, **options)
        synchronize(distributed)
        elapsed = time.perf_counter() - start
        if rank0:
            print(f"  {name:5s} {1e3 * elapsed:8.1f} ms ({1e3 * elapsed / iterations:6.2f} ms/it, {iterations} it)"
                  f"  PSNR {psnr(reconstruction, truth):6.2f} dB")
        panels.append((f"{name.upper()}, {iterations} it\n{psnr(reconstruction, truth):.1f} dB", reconstruction))

    if rank0 and args.figure:
        noise = f", {args.noise:.0%} noise" if args.noise > 0 else ""
        save_slices(args.figure, panels, f"{args.trajectory} cone-beam scan, {args.size}³ {phantom_name}{noise}",
                    vmax=1.0 if args.phantom else 0.4, sections=("axial", "coronal"))
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
