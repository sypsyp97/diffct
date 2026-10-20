"""Measure public per-call trajectory admission, with numerical execution bypassed.

Use --backend torch --frame-device cpu for CPU-only admission measurements.
Use PYTHONPATH pointing to the checkout being measured. Raw admission timings
run separately from the instrumented counting pass, never projector throughput.
"""

import argparse
from contextlib import ExitStack
import json
import math
import platform
import time
from unittest.mock import patch

import torch

from diffct import Projector, circular_trajectory_3d
from diffct.backends import ReferenceProjector


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numba", "torch"), default="numba")
    parser.add_argument("--frame-device", choices=("cpu", "cuda", "both"), default="both")
    parser.add_argument("--volume", type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument("--views", type=int, default=360)
    parser.add_argument("--detector", type=int, nargs=2, default=(384, 256))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    if min(*args.volume, args.views, *args.detector, args.samples) < 1 or args.warmup < 0:
        parser.error("dimensions/samples must be positive; warmup must be nonnegative")
    return args


def _admit(operator, frames, image, *, counting=False):
    counts = {"validation_ray_batches": 0, "validation_ray_inputs": 0,
              "accelerator_to_cpu_to_calls": 0, "accelerator_to_cpu_cpu_calls": 0}
    rays, tensor_to, tensor_cpu = ReferenceProjector._rays, torch.Tensor.to, torch.Tensor.cpu
    sentinel = torch.empty(0)

    def traced_rays(reference, views, pixels, device, *args, **kwargs):
        if kwargs.get("validate", args[0] if args else False):
            counts["validation_ray_batches"] += 1
            counts["validation_ray_inputs"] += (views.stop - views.start) * math.prod(
                pixel.stop - pixel.start for pixel in pixels
            )
        return rays(reference, views, pixels, device, *args, **kwargs)

    def traced_to(value, *args, **kwargs):
        result = tensor_to(value, *args, **kwargs)
        counts["accelerator_to_cpu_to_calls"] += int(value.device.type != "cpu" and result.device.type == "cpu")
        return result

    def traced_cpu(value, *args, **kwargs):
        result = tensor_cpu(value, *args, **kwargs)
        counts["accelerator_to_cpu_cpu_calls"] += int(value.device.type != "cpu")
        return result

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    with ExitStack() as stack:
        stack.enter_context(patch.object(Projector, "_project", lambda owner, value: sentinel))
        if counting:
            stack.enter_context(patch.object(ReferenceProjector, "_rays", traced_rays))
            stack.enter_context(patch.object(torch.Tensor, "to", traced_to))
            stack.enter_context(patch.object(torch.Tensor, "cpu", traced_cpu))
        started = time.perf_counter() if not counting else None
        result = operator.project(image, trajectory=frames)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started if not counting else None
    assert result is sentinel, "numerical execution was not bypassed"
    if not counting:
        return {"measurement": "raw-admission-timing", "seconds": elapsed}
    counts["accelerator_to_cpu_tensor_calls"] = (counts["accelerator_to_cpu_to_calls"]
                                                 + counts["accelerator_to_cpu_cpu_calls"])
    return {"measurement": "instrumented-counting-pass; not timed", **counts}


def main():
    args = _arguments()
    cuda = torch.cuda.is_available()
    if args.backend == "numba" and (not cuda or torch.version.hip is not None):
        raise SystemExit("native admission measurement requires NVIDIA CUDA")
    devices = ("cpu", "cuda") if args.frame_device == "both" and cuda else (args.frame_device,)
    if devices == ("both",):
        devices = ("cpu",)
    if "cuda" in devices and not cuda:
        raise SystemExit("CUDA-resident frames requested, but CUDA is unavailable")
    print(json.dumps({"measurement": "admission-only; public project(..., trajectory=...), _project bypassed",
                      "counter_scope": "Python Tensor.to/Tensor.cpu accelerator-to-CPU tensor requests; excludes scalar item/synchronization",
                      "platform": platform.platform(), "python": platform.python_version(),
                      "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                      "hardware": torch.cuda.get_device_name(0) if cuda else platform.processor(),
                      "backend": args.backend, "precision": "fast", "frame_dtype": "float32",
                      "sid": 1000.0, "sdd": 1500.0,
                      "volume": args.volume, "views": args.views, "detector": args.detector,
                      "chunks": "default", "warmup": args.warmup, "samples": args.samples}), flush=True)
    image = torch.empty(tuple(args.volume), dtype=torch.float32)
    for device in devices:
        frames = circular_trajectory_3d(args.views, 1000.0, 1500.0, device=device, dtype=torch.float32)
        operator = Projector(frames, tuple(args.volume), tuple(args.detector), beam="cone", backend=args.backend)
        print(json.dumps({"frame_device": device, **_admit(operator, frames, image, counting=True)}), flush=True)
        for _ in range(args.warmup):
            _admit(operator, frames, image)
        for sample in range(args.samples):
            print(json.dumps({"frame_device": device, "sample": sample + 1,
                              **_admit(operator, frames, image)}), flush=True)


if __name__ == "__main__":
    main()
