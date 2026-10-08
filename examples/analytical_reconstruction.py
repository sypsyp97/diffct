"""Analytical reconstruction: parallel-beam FBP, fan-beam FBP and cone-beam FDK.

Each pipeline is the textbook one: optional cosine weighting, ramp filter with
a window, angular weights, voxel-driven weighted backprojection. The window
trades sharpness against noise: "ram-lak" keeps all frequencies, "shepp-logan"
and "cosine" damp the highest, "hamming" and "hann" damp the most.

    python examples/analytical_reconstruction.py --window shepp-logan --figure fbp.png
"""

import argparse
import math
from pathlib import Path

import torch

from diffct import (
    Projector,
    angular_integration_weights,
    circular_trajectory_2d_fan,
    circular_trajectory_2d_parallel,
    fan_cosine_weights,
    fan_weighted_backproject,
    parallel_weighted_backproject,
    ramp_filter_1d,
)
from _common import Scan, fdk, psnr, save_slices, shepp_logan_3d


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--window", default="shepp-logan",
                        choices=["ram-lak", "shepp-logan", "cosine", "hamming", "hann"])
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("diffct needs a CUDA GPU")
    n = args.size
    phantom = torch.from_numpy(shepp_logan_3d(n)).cuda()
    image = phantom[n // 2]
    views = 360
    full_turn = torch.linspace(0.0, 2 * math.pi, views + 1, device="cuda")[:-1]
    half_turn = torch.linspace(0.0, math.pi, views + 1, device="cuda")[:-1]

    # Parallel beam over 180 degrees; detector pitch 0.5 voxel.
    n_det, pitch = 3 * n, 0.5
    trajectory = circular_trajectory_2d_parallel(views, end_angle=math.pi, device="cpu")
    sinogram = Projector(trajectory, (n, n), n_det, beam="parallel", detector_spacing=pitch).project(image)
    filtered = ramp_filter_1d(sinogram, dim=1, sample_spacing=pitch, pad_factor=2, window=args.window)
    filtered = filtered * angular_integration_weights(half_turn, redundant_full_scan=False).view(-1, 1)
    parallel = parallel_weighted_backproject(filtered.contiguous(), *[t.cuda() for t in trajectory],
                                             pitch, n, n).clamp_min(0)

    # Fan beam over a full turn; magnification 1.6, detector pitch 0.8 (0.5 voxel at the isocentre).
    sid, sdd, n_det, pitch = 2.5 * n, 4.0 * n, 3 * n, 0.8
    trajectory = circular_trajectory_2d_fan(views, sid, sdd, device="cpu")
    sinogram = Projector(trajectory, (n, n), n_det, beam="fan", detector_spacing=pitch).project(image)
    weighted = sinogram * fan_cosine_weights(n_det, pitch, sdd, device="cuda").unsqueeze(0)
    filtered = ramp_filter_1d(weighted, dim=1, sample_spacing=pitch, pad_factor=2, window=args.window)
    filtered = filtered * angular_integration_weights(full_turn, redundant_full_scan=True).view(-1, 1)
    fan = fan_weighted_backproject(filtered.contiguous(), *[t.cuda() for t in trajectory],
                                   pitch, n, n).clamp_min(0)

    # Cone beam FDK on a circular orbit.
    scan = Scan(n, views)
    trajectory = scan.trajectory("circular")
    sinogram = Projector(trajectory, (n, n, n), scan.detector, detector_spacing=scan.pitch).project(phantom)
    cone = fdk(scan, sinogram, trajectory, window=args.window)

    print(f"window {args.window}: parallel FBP {psnr(parallel, image):.2f} dB, "
          f"fan FBP {psnr(fan, image):.2f} dB, cone FDK {psnr(cone, phantom):.2f} dB")
    if args.figure:
        save_slices(args.figure, [("phantom", phantom),
                                  (f"parallel FBP, {psnr(parallel, image):.1f} dB", parallel),
                                  (f"fan FBP, {psnr(fan, image):.1f} dB", fan),
                                  (f"cone FDK, {psnr(cone, phantom):.1f} dB", cone)],
                    f"{n}² / {n}³ Shepp-Logan, {args.window} window")


if __name__ == "__main__":
    main()
