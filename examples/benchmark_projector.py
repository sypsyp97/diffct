"""Check physical correctness, numeric parity and end-to-end GPU speedup.

Local: python examples/benchmark_projector.py --devices 0 1
NCCL: torchrun --standalone --nproc-per-node=2 examples/benchmark_projector.py
Synthetic uniform boxes have independently calculable ray intersection lengths.
Timings include tensor transfers, gathering and gradient communication.
"""

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import socket
import statistics
import time

import numpy as np
import torch
import torch.distributed as dist

from diffct import (Projector, circular_trajectory_2d_fan,
                    circular_trajectory_2d_parallel, spiral_trajectory_3d)


def box_lengths(trajectory, shape, detector, spacing, beam, indices):
    """CPU float64 slab intersections, independent of the CUDA projectors."""
    geometry = [value.detach().cpu().numpy().astype(np.float64) for value in trajectory]
    pixels = int(np.prod(detector))
    views, pixel = np.divmod(indices, pixels)
    if beam == "cone":
        u, v = np.divmod(pixel, detector[1])
        positions = (geometry[1][views]
                     + ((u - (detector[0] - 1) / 2) * spacing[0])[:, None] * geometry[2][views]
                     + ((v - (detector[1] - 1) / 2) * spacing[1])[:, None] * geometry[3][views])
    else:
        positions = (geometry[1][views]
                     + ((pixel - (detector[0] - 1) / 2) * spacing[0])[:, None] * geometry[2][views])
    if beam == "parallel":
        origins, directions = positions, geometry[0][views]
    else:
        origins = geometry[0][views]
        directions = positions - origins
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    half_extent = np.asarray(shape[::-1], dtype=np.float64) / 2
    moving = np.abs(directions) > 1e-14
    with np.errstate(divide="ignore", invalid="ignore"):
        first = (-half_extent - origins) / directions
        second = (half_extent - origins) / directions
    near = np.where(moving, np.minimum(first, second), -np.inf).max(axis=1)
    far = np.where(moving, np.maximum(first, second), np.inf).min(axis=1)
    outside = ((~moving) & (np.abs(origins) > half_extent)).any(axis=1)
    if beam != "parallel":
        near = np.maximum(near, 0)
    return np.where(outside, 0, np.maximum(far - near, 0))


def physical_check(trajectory, shape, detector, spacing, beam, projection):
    count = projection.numel()
    indices = np.unique(np.linspace(0, count - 1, min(count, 257), dtype=np.int64))
    expected = torch.from_numpy(box_lengths(trajectory, shape, detector, spacing, beam, indices))
    actual = projection.detach().flatten()[torch.as_tensor(indices, device=projection.device)].cpu().double()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-4)
    return {"reference": "CPU float64 analytic uniform-box ray intersections",
            "rays": len(indices), "max_abs": float((actual - expected).abs().max()),
            "rtol": 1e-4, "atol": 2e-4, "passed": True}


def parity(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-5)
    return {"max_abs": float((actual - expected).abs().max()) if actual.numel() else 0.0,
            "reference_scale": float(expected.abs().max()) if expected.numel() else 0.0,
            "rtol": 5e-4, "atol": 5e-5, "passed": True}


def synchronize(devices):
    for device in devices:
        torch.cuda.synchronize(device)


def measure(operation, devices, warmup, repeats, distributed=False):
    for _ in range(warmup):
        operation()
    synchronize(devices)
    samples = []
    for _ in range(repeats):
        if distributed:
            dist.barrier()
        synchronize(devices)
        start = time.perf_counter()
        operation()
        synchronize(devices)
        seconds = time.perf_counter() - start
        if distributed:
            elapsed = torch.tensor(seconds, device=devices[0], dtype=torch.float64)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            seconds = float(elapsed)
        samples.append(seconds)
    return {"median_seconds": statistics.median(samples), "samples_seconds": samples}


def gradient_iteration(operator, volume):
    image = volume.detach().clone().requires_grad_()
    prediction = operator.project(image)
    loss = 0.5 * prediction.square().sum()
    return torch.autograd.grad(loss, image)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, nargs="+")
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--views", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-speedup", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.sizes) < 2 or args.views < 2 or args.warmup < 1 or args.repeats < 2:
        parser.error("sizes/views must be >= 2, warmup >= 1 and repeats >= 2")
    if not math.isfinite(args.min_speedup) or args.min_speedup < 0:
        parser.error("--min-speedup must be finite and nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed and args.devices is not None:
        parser.error("under torchrun, use one GPU per process without --devices")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if distributed:
        dist.init_process_group("nccl", timeout=timedelta(seconds=300))
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    device = torch.device("cuda", torch.cuda.current_device())
    devices = [device] if distributed else [torch.device("cuda", index) for index in
                                            (args.devices or range(torch.cuda.device_count()))]
    result = {"data": "synthetic uniform boxes, arbitrary helical cone trajectory",
              "mode": "nccl" if distributed else "local_devices",
              "world_size": world_size, "candidate_gpu_count": world_size if distributed else len(devices),
              "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "timing": "wall time including per-call transfers, gathering, autograd and reductions; warmed kernels",
              "single_gpu_reference_rank": 0, "cases": [], "physics": []}
    try:
        if result["candidate_gpu_count"] < 2 and args.min_speedup > 0:
            raise RuntimeError("Acceleration verification requires at least two GPUs; "
                               "--min-speedup=0 permits a single-GPU utility check")
        # An independent physical check for each beam precedes performance work.
        for beam in ("parallel", "fan", "cone"):
            shape = (15, 19, 21) if beam == "cone" else (19, 21)
            detector = (17, 15) if beam == "cone" else (17,)
            spacing = (1.0, 0.8) if beam == "cone" else (1.0,)
            if beam == "parallel":
                trajectory = circular_trajectory_2d_parallel(7, device="cpu")
            elif beam == "fan":
                trajectory = circular_trajectory_2d_fan(7, 45.0, 70.0, device="cpu")
            else:
                trajectory = spiral_trajectory_3d(7, 45.0, 70.0, z_range=4.0, n_turns=1.25, device="cpu")
            single = Projector(trajectory, shape, detector, beam=beam,
                               detector_spacing=spacing if beam == "cone" else spacing[0])
            projection = single.project(torch.ones(shape, device=device))
            result["physics"].append({"beam": beam, **physical_check(
                trajectory, shape, detector, spacing, beam, projection)})

        for size in args.sizes:
            shape = (size, size, size)
            detector = (size * 3 // 2, size)
            trajectory = spiral_trajectory_3d(args.views, size * 2.5, size * 4.0,
                                              z_range=size * 0.4, n_turns=1.25, device="cpu")
            single = Projector(trajectory, shape, detector)
            candidate = Projector(trajectory, shape, detector,
                                  devices=None if distributed else devices, distributed=distributed)
            volume = torch.ones(shape, device=device)
            single_projection = single.project(volume)
            full_sinogram = torch.full_like(single_projection, 0.001)
            single_back = single.backproject(full_sinogram)
            single_gradient = gradient_iteration(single, volume)
            reference = physical_check(trajectory, shape, detector, (1.0, 1.0), "cone", single_projection)
            local_sinogram = full_sinogram[candidate.view_slice].contiguous()
            candidate_projection = candidate.project(volume)
            candidate_back = candidate.backproject(local_sinogram)
            candidate_gradient = gradient_iteration(candidate, volume)
            case = {"volume_shape": list(shape), "detector_shape": list(detector),
                    "global_views": args.views, "physical_reference": reference,
                    "parity": {"project": parity(candidate_projection, single_projection[candidate.view_slice]),
                               "backproject": parity(candidate_back, single_back),
                               "iteration_gradient": parity(candidate_gradient, single_gradient)},
                    "timings": {}}
            baseline_operations = {"project": lambda: single.project(volume),
                                   "backproject": lambda: single.backproject(full_sinogram),
                                   "gradient_iteration": lambda: gradient_iteration(single, volume)}
            candidate_operations = {"project": lambda: candidate.project(volume),
                                    "backproject": lambda: candidate.backproject(local_sinogram),
                                    "gradient_iteration": lambda: gradient_iteration(candidate, volume)}
            for name in baseline_operations:
                if distributed:
                    dist.barrier()
                baseline = (measure(baseline_operations[name], [device], args.warmup, args.repeats)
                            if rank == 0 else None)
                if distributed:
                    baseline_packet = [baseline]
                    dist.broadcast_object_list(baseline_packet, src=0)
                    baseline = baseline_packet[0]
                multi = measure(candidate_operations[name], devices, args.warmup, args.repeats, distributed)
                case["timings"][name] = {"single_gpu": baseline, "candidate": multi,
                                         "speedup": baseline["median_seconds"] / multi["median_seconds"]}
            case["acceleration_passed"] = (
                case["timings"]["gradient_iteration"]["speedup"] >= args.min_speedup)
            result["cases"].append(case)
        worker = {"rank": rank, "host": socket.gethostname(),
                  "gpus": [{"index": item.index, "name": torch.cuda.get_device_name(item)} for item in devices],
                  "physics": result["physics"],
                  "case_checks": [{"volume_shape": case["volume_shape"],
                                   "parity": case["parity"],
                                   "physical_reference": case["physical_reference"]}
                                  for case in result["cases"]]}
        workers = [worker]
        if distributed:
            workers = [None] * world_size
            dist.all_gather_object(workers, worker)
        result["workers"] = workers
        result["distinct_node_count"] = len({item["host"] for item in workers})
        result["passed"] = all(case["acceleration_passed"] for case in result["cases"])
        result["min_speedup"] = args.min_speedup
        if rank == 0:
            payload = json.dumps(result, indent=2)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(payload + "\n", encoding="utf-8")
            print(payload)
        return 0 if result["passed"] else 1
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
