"""Measure synthetic launch/transfer overheads with synchronized CUDA timing.

Run the same script with PYTHONPATH pointing to each checkout being compared.
No datasets, compiler flags, global allocator settings, or accuracy modes change.
"""

import argparse
import importlib.metadata
import json
import statistics
import time

import torch

import diffct
from diffct import ParameterizedSurface, Projector, circular_trajectory_3d, cgls, sirt, tv_reconstruction, fdk


def measure(operation, warmup, samples):
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    elapsed = []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = operation()
        torch.cuda.synchronize()
        elapsed.append((time.perf_counter() - start) * 1000)
        del result
    return {"wall_ms": elapsed, "median_ms": statistics.median(elapsed),
            "incremental_peak_bytes": torch.cuda.max_memory_allocated() - allocated}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--detector", type=int, nargs=2, default=(96, 16))
    parser.add_argument("--volume", type=int, nargs=3, default=(8, 32, 32))
    parser.add_argument("--case", choices=("all", "reuse", "live", "legacy", "per-view", "sampled",
                                          "cgls", "sirt", "tv", "fdk"),
                        default="all")
    args = parser.parse_args()
    if min(args.samples, args.views, *args.detector, *args.volume) < 1 or args.warmup < 0:
        parser.error("dimensions and samples must be positive; warmup must be nonnegative")
    if not torch.cuda.is_available() or torch.version.hip is not None:
        parser.error("NVIDIA CUDA is required")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(0)
    image = torch.rand(tuple(args.volume), device=device)
    frames = circular_trajectory_3d(args.views, 60., 100., device=device, dtype=torch.float32)
    detector = tuple(args.detector)
    axes = [(torch.arange(size, device=device) + .5 - size / 2) * .5 for size in detector]
    u, v = torch.meshgrid(*axes, indexing="ij")
    offsets = torch.stack((100 * (u / 100).sin(), v, 100 * (1 - (u / 100).cos())), -1)
    per_view = offsets.unsqueeze(0).expand(args.views, *offsets.shape).clone()
    coefficient = torch.tensor(.0005, dtype=torch.float64)

    def sampled(u, v, ids, curvature):
        return torch.stack((u, v, curvature * u.square()), -1)

    print(json.dumps({"configuration": vars(args), "synthetic": True,
                      "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                      "cuda": torch.version.cuda, "numba_cuda": importlib.metadata.version("numba-cuda"),
                      "source": diffct.__file__, "timing": "synchronized wall time, warm JIT"}), flush=True)
    for name in ("live", "legacy", "per-view", "sampled", "cgls", "sirt", "tv", "fdk"):
        if args.case not in ("all", name) and not (args.case == "reuse" and name in ("sirt", "tv", "fdk")):
            continue
        geometry = tuple(value.clone().requires_grad_(name == "live") for value in frames)
        surface = ({"legacy": lambda u, v: offsets, "per-view": lambda u, v: per_view,
                    "sampled": ParameterizedSurface(sampled, (coefficient,))}.get(name))
        chunks = (dict(view_chunk_size=args.views, detector_chunk_shape=detector, schedule="spatial")
                  if name in ("per-view", "sampled") else {})
        operator = Projector(geometry, tuple(args.volume), detector, beam="cone",
                             detector_spacing=.5, detector_surface=surface,
                             backend="numba", distributed=False, **chunks)
        with torch.no_grad():
            rays = operator.project(image)
            if name in ("cgls", "sirt", "tv"):
                solve = {"cgls": cgls, "sirt": sirt, "tv": tv_reconstruction}[name]
                operations = {name + "_5": lambda: solve(operator, rays, iterations=5, rtol=0, atol=0)}
            elif name == "fdk":
                angles = torch.arange(args.views, device=device) * (2 * torch.pi / args.views)
                operations = {"fdk_tiled": lambda: fdk(
                    rays, geometry, tuple(args.volume), angles, detector_spacing=.5,
                    backend="numba", volume_chunk_shape=tuple(max(1, size // 2) for size in args.volume),
                    isocenter=(0., 0., 0.),
                )}
            else:
                operations = {"project": lambda: operator.project(image),
                              "backproject": lambda: operator.backproject(rays)}
            for operation_name, operation in operations.items():
                result = measure(operation, args.warmup, args.samples)
                print(json.dumps({"case": name, "operation": operation_name, **result}), flush=True)


if __name__ == "__main__":
    main()
