"""CGLS with disk-backed state and optional fixed spatial ownership.

    python examples/disk_reconstruction.py --output disk-run
    torchrun --standalone --nproc-per-node=2 examples/disk_reconstruction.py \
        --partition space --output disk-space-run

The default generates explicitly synthetic measurements blockwise on disk.
Every rank writes its own directory. Existing files are never overwritten.
Use --measurements with a float32/float64 .npy file for the same scan; in views
mode supply a rank-local sinogram (the path may contain {rank}).
"""

import argparse
import itertools
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from diffct import NpyStore, ParameterizedSurface, Projector, circular_trajectory_3d


def _blocks(shape, limit):
    widths = list(shape)
    while math.prod(widths) > limit:
        axis = max(range(len(widths)), key=widths.__getitem__)
        widths[axis] = (widths[axis] + 1) // 2
    for start in itertools.product(*(range(0, size, max(1, width))
                                     for size, width in zip(shape, widths))):
        yield tuple(slice(first, min(first + width, size))
                    for first, width, size in zip(start, widths, shape))


def _phantom(store, owned, n, block_elements):
    for region in _blocks(store.shape, block_elements):
        axes = [(torch.arange(s.start + owner.start, s.stop + owner.start) + .5 - n / 2) / (n / 2)
                for s, owner in zip(region, owned)]
        z, y, x = axes[0][:, None, None], axes[1][None, :, None], axes[2][None, None, :]
        outer = ((x / .69).square() + (y / .92).square() + (z / .81).square() <= 1).float()
        inner = ((x / .55).square() + ((y + .02) / .75).square() + (z / .65).square() <= 1).float()
        store.write(region, outer - .65 * inner)
    store.flush()


def _surface(kind, views, radius):
    if kind == "flat":
        return None

    def sample(u, v, view_indices, r):
        if kind == "cylinder":
            return torch.stack((r * torch.sin(u / r), v, r * (torch.cos(u / r) - 1)), -1)
        factor = 1 + .1 * torch.sin(2 * math.pi * view_indices / views)
        normal = factor[:, None, None] * (u.square() - v.square()) / (2 * r)
        return torch.stack((u.expand_as(normal), v.expand_as(normal), normal), -1)

    return ParameterizedSurface(sample, parameters=(torch.tensor(radius, dtype=torch.float64),))


def _relative_residual(operator, prediction, measurements, block_elements, device):
    totals = torch.zeros(2, dtype=torch.float64)
    # Rays are replicated in space mode; count them once for a global norm.
    if operator.partition == "views" or operator.rank == 0:
        for region in _blocks(prediction.shape, block_elements):
            value = measurements.read(region).double()
            difference = prediction.read(region).double().sub_(value)
            totals[0] += difference.square().sum()
            totals[1] += value.square().sum()
    if operator.world_size > 1:
        if dist.get_backend(operator.process_group) == "nccl":
            totals = totals.to(device)
        dist.all_reduce(totals, group=operator.process_group)
        totals = totals.cpu()
    return math.sqrt(float(totals[0] / totals[1])) if totals[1] else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--views", type=int, default=48)
    parser.add_argument("--iterations", type=int, default=5,
                        help="target total completed iterations, including a resumed checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    parser.add_argument("--measurements", help="read-only .npy; optional {rank} in path")
    parser.add_argument("--resume", help="completed checkpoint rank directory; optional {rank}")
    parser.add_argument("--partition", choices=("views", "space"), default="views")
    parser.add_argument("--surface", choices=("flat", "cylinder", "saddle"), default="flat")
    parser.add_argument("--block-elements", type=int, default=1048576)
    parser.add_argument("--devices", nargs="+", type=int)
    parser.add_argument("--chunk-shape", nargs=3, type=int)
    parser.add_argument("--view-chunk-size", type=int)
    parser.add_argument("--detector-chunk-shape", nargs=2, type=int)
    parser.add_argument("--backend", choices=("nccl", "gloo"),
                        default="nccl" if dist.is_nccl_available() else "gloo")
    args = parser.parse_args()
    if min(args.size, args.views, args.iterations, args.block_elements) < 1:
        parser.error("size, views, iterations and block elements must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for the native projection kernels")

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed and args.devices is not None:
        parser.error("torchrun selects one GPU per rank; omit --devices")
    devices = args.devices or [int(os.environ.get("LOCAL_RANK", "0"))]
    torch.cuda.set_device(devices[0])
    device = torch.device("cuda", devices[0])
    if distributed:
        dist.init_process_group(args.backend)
    try:
        from _block_cgls import cgls_into

        n = args.size
        trajectory = circular_trajectory_3d(args.views, 2.5 * n, 4 * n, device="cpu")
        operator = Projector(trajectory, (n, n, n), (2 * n, 2 * n),
                             detector_spacing=.8, devices=None if distributed else devices,
                             distributed=distributed, partition=args.partition,
                             detector_surface=_surface(args.surface, args.views, 4 * n),
                             volume_chunk_shape=args.chunk_shape,
                             view_chunk_size=args.view_chunk_size,
                             detector_chunk_shape=args.detector_chunk_shape)
        directory = args.output / f"rank-{operator.rank:04d}"
        directory.mkdir(parents=True, exist_ok=False)
        if args.measurements:
            measurements = NpyStore(args.measurements.format(rank=operator.rank), mode="r")
        else:
            truth = NpyStore.create(directory / "synthetic-truth.npy", operator.volume_shape)
            _phantom(truth, operator.volume_slice, n, args.block_elements)
            measurements = NpyStore.create(directory / "synthetic-rays.npy", operator.projection_shape)
            operator.project_into(truth, measurements)
            measurements.flush()

        output = NpyStore.create(directory / "reconstruction.npy", operator.volume_shape)
        for card in devices:
            torch.cuda.synchronize(card)
            torch.cuda.reset_peak_memory_stats(card)
        start = time.perf_counter()
        cgls_into(operator, measurements, output, args.iterations,
                  workspace=directory / "state", block_elements=args.block_elements,
                  checkpoint_directory=directory / "checkpoints",
                  resume=args.resume.format(rank=operator.rank) if args.resume else None)
        output.flush()
        prediction = NpyStore.create(directory / "final-prediction.npy", operator.projection_shape)
        operator.project_into(output, prediction)
        relative = _relative_residual(operator, prediction, measurements, args.block_elements, device)
        report = {"rank": operator.rank, "world_size": operator.world_size,
                  "partition": operator.partition, "surface": args.surface,
                  "global_volume_shape": operator.global_volume_shape,
                  "local_volume_shape": operator.volume_shape,
                  "projection_shape": operator.projection_shape,
                  "synthetic_measurements": args.measurements is None,
                  "relative_residual": relative, "seconds": time.perf_counter() - start,
                  "x_s_p_bytes": 3 * 4 * math.prod(operator.volume_shape),
                  "r_q_bytes": 2 * 4 * math.prod(operator.projection_shape),
                  "peak_cuda_tensor_bytes": {str(card): torch.cuda.max_memory_allocated(card) for card in devices},
                  "solver": getattr(operator, "last_solver_stats", None),
                  "last_execution": operator.last_execution_stats}
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report), flush=True)
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
