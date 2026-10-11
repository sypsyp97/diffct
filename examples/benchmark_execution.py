"""Compare real CUDA execution orders, copy payloads and slab/block shapes.

Run from the repository root. Inputs/results use CPU memory. This deliberately
small benchmark reports its actual workload; timing ratios do not extrapolate
to a larger scan or to a different GPU. Compilation/pilots are warmed first.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from diffct import Projector, circular_trajectory_3d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--views", type=int, default=160)
    parser.add_argument("--detector", nargs=2, type=int, default=(128, 96), metavar=("U", "V"))
    parser.add_argument("--view-chunk-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--devices", nargs="+", type=int)
    parser.add_argument("--schedules", nargs="+", choices=("auto", "spatial", "views", "window"),
                        default=("auto", "spatial", "views", "window"))
    parser.add_argument("--shapes", nargs="+", choices=("auto", "slab", "block"),
                        default=("auto", "slab", "block"))
    parser.add_argument("--output", type=Path, help="optional measured JSON report")
    args = parser.parse_args()
    if args.size < 8 or min(args.views, args.view_chunk_size, args.repeats, *args.detector) < 1:
        parser.error("size must be >=8; view, batch, detector and repeat counts must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    devices = args.devices or [torch.cuda.current_device()]
    torch.cuda.set_device(devices[0])
    torch.manual_seed(632)
    n = args.size
    volume = torch.rand((n, n, n))
    sinogram = torch.rand((args.views, *args.detector))
    trajectory = circular_trajectory_3d(args.views, 2.5 * n, 4. * n, device="cpu")
    chunks = {"auto": None, "slab": (max(1, n // 8), n, n),
              "block": (max(1, n // 2),) * 3}
    report = {"torch": torch.__version__, "gpus": [torch.cuda.get_device_name(d) for d in devices],
              "seed": 632, "volume_shape": list(volume.shape),
              "projection_shape": list(sinogram.shape), "runs": []}
    references = {}
    for shape in args.shapes:
        for schedule in args.schedules:
            operator = Projector(trajectory, volume.shape, args.detector, devices=devices,
                                 volume_chunk_shape=chunks[shape],
                                 view_chunk_size=args.view_chunk_size, schedule=schedule)
            for operation, value in (("project", volume), ("backproject", sinogram)):
                run = getattr(operator, operation)
                run(value)
                selection = operator.last_execution_stats
                for device in devices:
                    torch.cuda.synchronize(device)
                durations = []
                peaks = []
                for _ in range(args.repeats):
                    for device in devices:
                        torch.cuda.reset_peak_memory_stats(device)
                    start = time.perf_counter()
                    result = run(value)
                    for device in devices:
                        torch.cuda.synchronize(device)
                    durations.append(time.perf_counter() - start)
                    peaks.append([torch.cuda.max_memory_allocated(device) for device in devices])
                if operation in references:
                    torch.testing.assert_close(result, references[operation], rtol=8e-4, atol=3e-3)
                else:
                    references[operation] = result
                measured = {"operation": operation, "requested_schedule": schedule,
                            "requested_shape": shape, "seconds": durations,
                            "median_seconds": statistics.median(durations),
                            "peak_gpu_bytes": [max(p[i] for p in peaks) for i in range(len(devices))],
                            "selection": selection,
                            "execution": operator.last_execution_stats}
                report["runs"].append(measured)
                print(json.dumps(measured), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
