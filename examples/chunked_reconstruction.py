"""CPU-backed CGLS using CUDA tiles; automatic memory sizing by default.

Run from the repository root. Full arrays and CGLS state still need host RAM.
This example supports one process with one or several GPUs.
"""

import argparse
import time

import torch

from diffct import Projector
from _common import Scan, cgls, shepp_logan_3d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--views", type=int, default=60)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--devices", nargs="+", type=int)
    parser.add_argument("--chunk-shape", nargs=3, type=int, metavar=("D", "H", "W"),
                        help="optional spatial limit; default sizes automatically")
    parser.add_argument("--view-chunk-size", type=int,
                        help="optional view-batch limit; default sizes automatically")
    args = parser.parse_args()
    if args.size < 8 or args.views < 1 or args.iterations < 1:
        parser.error("size must be >=8, views and iterations must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required; CPU arrays are computed on CUDA tiles")

    devices = args.devices if args.devices is not None else [torch.cuda.current_device()]
    torch.cuda.set_device(devices[0])
    scan = Scan(args.size, args.views)
    operator = Projector(
        scan.trajectory("circular"), (args.size,) * 3, scan.detector,
        detector_spacing=scan.pitch, devices=args.devices,
        volume_chunk_shape=args.chunk_shape, view_chunk_size=args.view_chunk_size,
    )
    truth = torch.from_numpy(shepp_logan_3d(args.size))

    # Warm the real kernels with small CPU arrays before measuring allocations.
    warm = Projector(scan.trajectory("circular"), (8, 8, 8), (8, 8),
                     devices=args.devices, volume_chunk_shape=(4, 4, 4),
                     view_chunk_size=1)
    warm.backproject(warm.project(torch.ones(warm.volume_shape)))
    for device in devices:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    measurements = operator.project(truth)
    reconstruction = cgls(operator, measurements, args.iterations)
    residual = torch.linalg.vector_norm(operator.project(reconstruction) - measurements)
    relative_residual = residual / torch.linalg.vector_norm(measurements)
    for device in devices:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    assert measurements.device.type == reconstruction.device.type == "cpu"
    assert torch.isfinite(reconstruction).all() and torch.isfinite(relative_residual)
    print(f"volume {tuple(truth.shape)}, sinogram {tuple(measurements.shape)}; arrays on CPU")
    print(f"CGLS {args.iterations} iterations: relative residual {relative_residual.item():.6g}")
    print(f"projection, reconstruction and residual check: {elapsed:.3f} s")
    for device in devices:
        peak = torch.cuda.max_memory_allocated(device) / 2 ** 20
        print(f"cuda:{device}: peak PyTorch tensor allocation {peak:.3f} MiB")


if __name__ == "__main__":
    main()
