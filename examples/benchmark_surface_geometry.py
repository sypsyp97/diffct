"""Synthetic curved-detector GPU benchmark; no dataset or measured scanner header.

Defaults: 1338 helical views, 736x64 detector, 96x512x512 volume, unit pitches,
SID=600, SDD=1000, z range=96, one turn. Central pixel 369.125 is zero-based.
The source-centered cylinder uses (SDD*sin(a), v, SDD*(cos(a)-1)) offsets.
Run with PYTHONPATH pointing to the checkout under comparison. Streamed plans
are reported and skipped unless --allow-streamed is explicitly supplied.
"""

import argparse
import json
import math
import platform
import statistics
import time
from unittest.mock import patch

import numba
import torch

import diffct
from diffct import ParameterizedSurface, Projector, spiral_trajectory_3d


class StreamedPlan(RuntimeError):
    def __init__(self, plan):
        super().__init__("Actual per-call execution plan is streamed; use --allow-streamed to time it.")
        self.plan = plan


def plan_metadata(plan):
    return {key: getattr(plan, key) for key in (
        "chunk_shape", "view_chunk_size", "detector_chunk_shape", "gpu_budget_bytes",
        "estimated_gpu_bytes", "allocation_limit", "reservation_budget_bytes",
    )}


def measure(operation, args, device):
    original = Projector._execution_plan
    streamed_plan = None

    def guarded_plan(owner, *positional, **keywords):
        nonlocal streamed_plan
        plan = original(owner, *positional, **keywords)
        if plan is not None:
            streamed_plan = plan
            if not args.allow_streamed:
                raise StreamedPlan(plan)
        return plan

    torch.cuda.synchronize(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    wall, cuda = [], []
    try:
        with patch.object(Projector, "_execution_plan", guarded_plan):
            for _ in range(args.warmup):
                operation()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for _ in range(args.samples):
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                begin.record()
                result = operation()
                end.record()
                torch.cuda.synchronize(device)
                wall.append(time.perf_counter() - started)
                cuda.append(begin.elapsed_time(end) / 1000)
                del result
        measurements = {"status": "measured", "wall_seconds": wall, "cuda_event_seconds": cuda,
                        "median_wall_seconds": statistics.median(wall),
                        "median_cuda_event_seconds": statistics.median(cuda),
                        "full_admission": streamed_plan is None}
        if streamed_plan is not None:
            measurements["actual_plan"] = plan_metadata(streamed_plan)
    except StreamedPlan as error:
        measurements = {"status": "skipped", "reason": str(error), "full_admission": False,
                        "actual_plan": plan_metadata(error.plan)}
    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    return {**measurements, "memory_scope": "Warmup and samples; PyTorch CUDA allocator",
            "baseline_allocated_bytes": baseline_allocated, "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
            "incremental_peak_allocated_bytes": peak_allocated - baseline_allocated,
            "incremental_peak_reserved_bytes": peak_reserved - baseline_reserved}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--surface", choices=("legacy", "parameterized", "both"), default="both")
    parser.add_argument("--views", type=int, default=1338)
    parser.add_argument("--detector", type=int, nargs=2, default=(736, 64))
    parser.add_argument("--volume", type=int, nargs=3, default=(96, 512, 512))
    parser.add_argument("--sid", type=float, default=600.0)
    parser.add_argument("--sdd", type=float, default=1000.0)
    parser.add_argument("--z-range", type=float, default=96.0)
    parser.add_argument("--turns", type=float, default=1.0)
    parser.add_argument("--central-pixel", type=float, default=369.125)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--allow-streamed", action="store_true")
    args = parser.parse_args()
    if min(args.views, *args.detector, *args.volume, args.samples) < 1 or args.warmup < 0:
        parser.error("dimensions/samples must be positive; warmup must be nonnegative")
    if (not all(math.isfinite(x) for x in
                (args.sid, args.sdd, args.z_range, args.turns, args.central_pixel))
            or not 0 < args.sid < args.sdd or args.z_range < 0 or args.turns <= 0):
        parser.error("require finite geometry, 0 < SID < SDD, z-range >= 0 and turns > 0")
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise SystemExit("This benchmark requires an NVIDIA CUDA GPU; CPU execution is unsupported.")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    generator = torch.Generator(device=device).manual_seed(0)
    image = torch.rand(tuple(args.volume), device=device, dtype=torch.float32, generator=generator)
    rays = torch.ones((args.views, *args.detector), device=device, dtype=torch.float32)
    base = spiral_trajectory_3d(args.views, args.sid, args.sdd, z_range=args.z_range,
                               n_turns=args.turns, start_angle=0.0, device=device, dtype=torch.float32)
    free, total = torch.cuda.mem_get_info(device)
    print(json.dumps({"configuration": vars(args), "synthetic": True,
                      "volume_axis_order": "zyx", "detector_axis_order": "uv",
                      "voxel_spacing": 1.0, "detector_spacing": [1.0, 1.0],
                      "image": "CUDA uniform random [0,1), seed=0", "rays": "CUDA ones",
                      "loss": "0.5 * mean((project(image) - rays)^2)",
                      "central_pixel_indexing": "zero-based", "cylinder_radius": args.sdd,
                      "start_angle": 0.0, "backend": "numba", "precision": "fast",
                      "frame_dtype": "float32", "parameter_dtype": "float64",
                      "offset_dtype": "float32", "geometry_device": str(device),
                      "python": platform.python_version(), "platform": platform.platform(),
                      "diffct": diffct.__version__, "diffct_source": diffct.__file__,
                      "torch": torch.__version__, "numba": numba.__version__,
                      "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
                      "free_bytes_after_inputs": free, "total_gpu_bytes": total,
                      "timing": "Synchronized wall and current-stream CUDA events, including host gaps; no profiler."}),
          flush=True)
    kinds = ("legacy", "parameterized") if args.surface == "both" else (args.surface,)
    for kind in kinds:
        for mode in ("no_geometry", "surface_gradient", "trajectory_gradient"):
            frames = tuple(value.detach().clone().requires_grad_(mode == "trajectory_gradient") for value in base)
            central = torch.tensor(args.central_pixel, device=device, dtype=torch.float64,
                                   requires_grad=mode == "surface_gradient")

            def offsets(u, v, value):
                angle = (u.to(device) + (args.detector[0] - 1) / 2 - value.to(device)) / args.sdd
                return torch.stack((args.sdd * angle.sin(), v.to(device),
                                    args.sdd * (angle.cos() - 1)), dim=-1).float()

            surface = (lambda u, v: offsets(u, v, central)) if kind == "legacy" else ParameterizedSurface(
                lambda u, v, ids, value: offsets(u, v, value), (central,),
            )
            operator = Projector(frames, tuple(args.volume), tuple(args.detector), beam="cone",
                                 detector_surface=surface, voxel_spacing=1.0, detector_spacing=(1.0, 1.0),
                                 backend="numba", precision="fast", devices=[device], distributed=False)
            operations = ("project", "backproject") if mode == "no_geometry" else ("loss_gradient",)
            for operation in operations:
                is_project = operation != "backproject"
                with torch.no_grad():
                    spec, geometry = operator._effective_geometry(None, device)
                    plan = operator._execution_plan(image if is_project else rays, is_project, geometry_spec=spec)
                    del geometry
                record = {"surface": kind, "mode": mode, "operation": operation,
                          "full_admission": plan is None}
                if plan is not None:
                    record["plan"] = plan_metadata(plan)
                print(json.dumps({"admission": record}), flush=True)
                if plan is not None and not args.allow_streamed:
                    print(json.dumps({**record, "status": "skipped",
                                      "reason": "Current execution plan is streamed; use --allow-streamed to time it."}),
                          flush=True)
                    continue

                def run(current=operator):
                    if mode == "no_geometry":
                        with torch.no_grad():
                            return current.project(image) if is_project else current.backproject(rays)
                    loss = 0.5 * (current.project(image) - rays).square().mean()
                    return torch.autograd.grad(loss, (central,) if mode == "surface_gradient" else frames)

                print(json.dumps({**record, **measure(run, args, device)}), flush=True)
                del run
            del operator


if __name__ == "__main__":
    main()
