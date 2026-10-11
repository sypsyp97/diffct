"""Measure the README workload on one GPU or one torchrun rank per GPU.

Run each rank count in a fresh process. Examples:
  python examples/benchmark_scaling.py --output baseline.json
  torchrun --standalone --nproc-per-node=4 examples/benchmark_scaling.py \
      --expected-nodes 1 --output four.json

The JSON contains raw timings, metadata and complete-array validation results.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import time
from datetime import timedelta
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

import diffct
from diffct import Projector, circular_trajectory_3d

WARMUP = 3
REPEATS = 7
ITERATIONS = 10
TOLERANCES = {"project": 5e-4, "backproject": 5e-4, "cgls": 5e-3}


def squared_norm(tensor, distributed=False):
    """Sum squared FP32 state in FP64; reduce only sharded ray arrays."""
    value = tensor.double().square().sum()
    if distributed:
        dist.all_reduce(value)
    return value


@torch.no_grad()
def cgls(operator, measurements, iterations=ITERATIONS):
    """Run fixed updates, including zero initialization and the first adjoint."""
    x = torch.zeros(operator.volume_shape, dtype=torch.float32, device=measurements.device)
    r = measurements.clone()
    s = operator.backproject(r)
    p = s.clone()
    gamma = squared_norm(s)
    tiny = torch.finfo(torch.float64).tiny
    for _ in range(iterations):
        q = operator.project(p)
        alpha = gamma / squared_norm(q, operator.world_size > 1).clamp_min(tiny)
        x.add_(p * alpha.to(torch.float32))
        r.sub_(q * alpha.to(torch.float32))
        s = operator.backproject(r)
        gamma_new = squared_norm(s)
        p.mul_((gamma_new / gamma.clamp_min(tiny)).to(torch.float32)).add_(s)
        gamma = gamma_new
    return x


def measure(operation, distributed, denominator=1, warmup=WARMUP, repeats=REPEATS):
    for _ in range(warmup):
        operation()
    samples = []
    for _ in range(repeats):
        if distributed:
            dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = operation()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        del value
        if distributed:
            slowest = torch.tensor(elapsed, dtype=torch.float64, device="cuda")
            dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
            elapsed = slowest.item()
        samples.append(elapsed)
    return {"samples_seconds": samples, "median_seconds": statistics.median(samples),
            "denominator": denominator,
            "median_seconds_per_unit": statistics.median(samples) / denominator}


def parity(actual, expected, tolerance):
    """Compare complete arrays with a predeclared relative L2 tolerance."""
    difference = actual.double() - expected.double()
    relative = (squared_norm(difference) / squared_norm(expected).clamp_min(1e-30)).sqrt().item()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    return {"finite": finite, "relative_l2": relative if math.isfinite(relative) else None,
            "max_absolute_error": difference.abs().max().item() if finite else None,
            "relative_l2_tolerance": tolerance, "passed": finite and relative <= tolerance}


def digest(*arrays):
    value = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        value.update(str((array.shape, array.dtype.str)).encode())
        value.update(array.tobytes())
    return value.hexdigest()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[512])
    parser.add_argument("--expected-nodes", type=int, choices=(1, 2))
    args = parser.parse_args()
    if min(args.sizes) < 2:
        parser.error("sizes must be at least 2")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in (1, 4, 8):
        parser.error("the README protocol requires 1, 4, or 8 ranks")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    rank = dist.get_rank() if distributed else 0
    try:
        properties = torch.cuda.get_device_properties(device)
        worker = {"rank": rank, "local_rank": local_rank, "host": socket.gethostname(),
                  "gpu": properties.name, "gpu_memory_bytes": properties.total_memory,
                  "compute_capability": [properties.major, properties.minor],
                  "multiprocessors": properties.multi_processor_count}
        workers = [worker]
        if distributed:
            workers = [None] * world_size
            dist.all_gather_object(workers, worker)
        hosts = list(dict.fromkeys(item["host"] for item in workers))
        if len({(item["host"], item["local_rank"]) for item in workers}) != world_size:
            raise RuntimeError("each rank must own a different local GPU")
        if args.expected_nodes is not None and len(hosts) != args.expected_nodes:
            raise RuntimeError("observed node count differs from --expected-nodes")
        if args.expected_nodes is not None and any(
                sum(item["host"] == host for item in workers) != world_size // args.expected_nodes
                for host in hosts):
            raise RuntimeError("ranks must be balanced across the expected nodes")
        for item in workers:
            item["node_index"] = hosts.index(item.pop("host"))
        versions = {}
        for package in ("diffct", "torch", "numpy", "numba", "numba-cuda"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        try:
            driver = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True, timeout=10).splitlines()
        except (OSError, subprocess.SubprocessError):
            driver = []
        report = {"schema_version": 1, "runtime_diffct_version": diffct.__version__,
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "software": {"packages": versions, "python": platform.python_version(),
                               "cuda": torch.version.cuda, "nccl": torch.cuda.nccl.version(),
                               "nvidia_driver": sorted(set(driver)),
                               "os": platform.system(), "architecture": platform.machine()},
                  "world_size": world_size, "node_count": len(hosts), "workers": workers,
                  "protocol": {"warmup": WARMUP, "repeats": REPEATS,
                               "statistic": "median of per-sample maximum rank elapsed time",
                               "timer": "perf_counter, CUDA synchronized at both boundaries",
                               "rank_max_reduction": "outside timed region",
                               "cgls_iterations": ITERATIONS,
                               "cgls_warmup_iterations": 2, "cgls_repeats": 3,
                               "cgls_denominator": ITERATIONS,
                               "cgls_initialization_included": True,
                               "cgls_norm_dtype": "float64", "state_dtype": "float32",
                               "partition": "views", "state_device": "cuda",
                               "allocator_cap_imposed": False,
                               "speedup_threshold": None}, "cases": []}
        for n in args.sizes:
            shape = (n, n, n)
            # Sample ellipsoids at voxel centres in normalized coordinates.
            axis = (np.arange(n, dtype=np.float32) + 0.5 - n / 2) / (n / 2)
            zz, yy, xx = np.meshgrid(axis, axis, axis, indexing="ij", sparse=True)
            phantom = np.zeros(shape, dtype=np.float32)
            ellipsoids = np.array([
                [0, 0, 0, .69, .92, .81, 0, 1],
                [0, -.0184, 0, .6624, .874, .78, 0, -.8],
                [.22, 0, 0, .11, .31, .22, -math.pi / 10, -.2],
                [-.22, 0, 0, .16, .41, .28, math.pi / 10, -.2],
                [0, .35, -.15, .21, .25, .41, 0, .1],
                [0, .10, .25, .046, .046, .05, 0, .1],
                [0, -.10, .25, .046, .046, .05, 0, .1],
                [-.08, -.605, 0, .046, .023, .05, 0, .1],
                [0, -.605, 0, .023, .023, .02, 0, .1],
                [.06, -.605, 0, .023, .046, .02, 0, .1]], dtype=np.float32)
            for x0, y0, z0, a, b, c, phi, value in ellipsoids:
                xp = math.cos(phi) * (xx - x0) - math.sin(phi) * (yy - y0)
                yp = math.sin(phi) * (xx - x0) + math.cos(phi) * (yy - y0)
                phantom += value * ((xp / a) ** 2 + (yp / b) ** 2 + ((zz - z0) / c) ** 2 <= 1)
            phantom = np.clip(phantom, 0, 1)
            del zz, yy, xx, xp, yp
            trajectory = circular_trajectory_3d(360, 2.5 * n, 4.0 * n, device="cpu")
            input_digest = digest(phantom, *(component.numpy() for component in trajectory))
            single = Projector(trajectory, shape, (2 * n, 3 * n // 2), detector_spacing=1.25)
            candidate = Projector(trajectory, shape, (2 * n, 3 * n // 2),
                                  detector_spacing=1.25, distributed=distributed)
            volume = torch.from_numpy(phantom).to(device)
            full_projection = single.project(volume)
            measurements = full_projection.clone()
            full_back = single.backproject(measurements)
            full_reconstruction = cgls(single, measurements)
            local_measurements = measurements[candidate.view_slice].clone()
            projection = candidate.project(volume)
            back = candidate.backproject(local_measurements)
            reconstruction = cgls(candidate, local_measurements)
            checks = {"project": parity(projection, full_projection[candidate.view_slice], TOLERANCES["project"]),
                      "backproject": parity(back, full_back, TOLERANCES["backproject"]),
                      "cgls": parity(reconstruction, full_reconstruction, TOLERANCES["cgls"])}
            lhs = (projection.double() * local_measurements.double()).sum()
            if distributed:
                dist.all_reduce(lhs)
            rhs = (volume.double() * back.double()).sum()
            adjoint_error = ((lhs - rhs).abs() / torch.maximum(lhs.abs(), rhs.abs()).clamp_min(1e-30)).item()
            residual = candidate.project(reconstruction) - local_measurements
            relative_residual = (squared_norm(residual, distributed)
                                 / squared_norm(local_measurements, distributed)).sqrt().item()
            mse = (reconstruction - volume).double().square().mean().item()
            quality = {"rank": rank, "single_gpu_parity": checks,
                       "adjoint_relative_error": adjoint_error if math.isfinite(adjoint_error) else None,
                       "adjoint_tolerance": 5e-4,
                       "relative_residual": relative_residual if math.isfinite(relative_residual) else None,
                       "mse": mse if math.isfinite(mse) else None,
                       "psnr_db": -10 * math.log10(mse) if 0 < mse < math.inf else None,
                       "passed": all(check["passed"] for check in checks.values())
                       and math.isfinite(adjoint_error) and adjoint_error <= 5e-4
                       and math.isfinite(relative_residual) and relative_residual < 1}
            quality_all = [quality]
            if distributed:
                quality_all = [None] * world_size
                dist.all_gather_object(quality_all, quality)
            case = {"size": n, "volume_shape": list(shape), "detector_shape": [2 * n, 3 * n // 2],
                    "views": 360, "sid": 2.5 * n, "sdd": 4.0 * n, "pitch": 1.25,
                    "voxel_spacing": 1.0, "trajectory": "circular", "phantom": "Shepp-Logan 3D",
                    "input_sha256": input_digest,
                    "measurements_sha256": digest(measurements.cpu().numpy()),
                    "quality": quality_all, "timings": {}}
            del full_projection, full_back, full_reconstruction, projection, back, reconstruction, residual
            del single, measurements
            torch.cuda.empty_cache()
            operations = {"project": partial(candidate.project, volume),
                          "backproject": partial(candidate.backproject, local_measurements),
                          "cgls": partial(cgls, candidate, local_measurements)}
            for name, operation in operations.items():
                if name == "cgls":
                    cgls(candidate, local_measurements, iterations=2)
                    case["timings"][name] = measure(operation, distributed, ITERATIONS, warmup=0, repeats=3)
                else:
                    case["timings"][name] = measure(operation, distributed)
            report["cases"].append(case)
        report["passed"] = all(item["passed"] for case in report["cases"] for item in case["quality"])
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"version": report["runtime_diffct_version"], "ranks": world_size,
                              "passed": report["passed"]}))
        return 0 if report["passed"] else 1
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
