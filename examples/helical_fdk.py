"""Approximate helical FDK: cosine, row filter, slice windows and gather.

The synthetic axial helix covers a full turn around every reconstructed slice.
This native-cone approximation is not an exact helical inverse.
    python examples/helical_fdk.py --size 32 --views 720 --figure helical.png
"""

import argparse
import math
from pathlib import Path

import torch

from diffct import (
    Projector, cone_cosine_weights, helical_fdk, helical_view_weights,
    helical_weighted_backproject, ramp_filter_1d, spiral_trajectory_3d,
)
from _common import psnr, save_slices, shepp_logan_3d


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--views", type=int, default=720)
    parser.add_argument("--turns", type=float, default=4.0)
    parser.add_argument("--window", default="shepp-logan",
                        choices=("ram-lak", "shepp-logan", "cosine", "hamming", "hann"))
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    if args.size < 8 or args.views < 16 or args.turns < 3:
        parser.error("size >=8, views >=16 and turns >=3 are required for this example")
    if not torch.cuda.is_available():
        raise RuntimeError("diffct needs a CUDA GPU")
    n, views, turns = args.size, args.views, args.turns
    sid, sdd, pitch = 4.0 * n, 6.4 * n, 0.8
    trajectory = spiral_trajectory_3d(views, sid, sdd, z_range=2.0 * n,
                                       n_turns=turns, device="cpu")
    angles = torch.arange(views) * (2 * math.pi * turns / views)  # unwrapped radians
    shape, detector = (n,) * 3, (3 * n, 3 * n)
    truth = torch.from_numpy(shepp_logan_3d(n)).cuda()
    operator = Projector(trajectory, shape, detector, detector_spacing=pitch)
    measurements = operator.project(truth)

    # 1. Ordinary detector cosine weighting and ramp along u, never along v.
    cosine = cone_cosine_weights(*detector, pitch, pitch, sdd, device="cuda")
    filtered = ramp_filter_1d(measurements * cosine, dim=1, sample_spacing=pitch,
                              pad_factor=2, window=args.window)
    # 2. One complete angular turn per physical slice. The weights include dβ/2.
    slice_z = torch.arange(n) + 0.5 - n / 2
    weights = helical_view_weights(angles, trajectory[0][:, 2], slice_z).cuda()
    # 3. Weights enter AFTER filtering. They are explicit, replaceable tensors.
    learned_weights = weights.detach().clone().requires_grad_()
    manual = helical_weighted_backproject(filtered, trajectory, shape, learned_weights,
                                          isocenter=(0, 0, 0), detector_spacing=pitch)
    loss = (manual - truth).square().mean()
    loss.backward()
    with torch.no_grad():
        packaged = helical_fdk(measurements, trajectory, shape, angles,
                               isocenter=(0, 0, 0), detector_spacing=pitch, window=args.window)
    print(f"{turns:g}-turn helix, {n}^3: approximate FDK {psnr(manual, truth):.2f} dB")
    print(f"manual/API maximum difference: {(manual - packaged).abs().max().item():.3g}; "
          f"weight gradient norm: {learned_weights.grad.norm().item():.6g}")
    print(f"per-slice integration weights sum: {weights.sum(1).min().item():.6f} "
          f"to {weights.sum(1).max().item():.6f} (pi)")
    if args.figure:
        save_slices(args.figure, [("synthetic truth", truth), ("slice-windowed FDK", manual.detach())],
                    f"{turns:g}-turn axial helix, approximate FDK", sections=("axial", "coronal"))


if __name__ == "__main__":
    main()
