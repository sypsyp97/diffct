"""Calibrate a cone-beam trajectory from projections with geometry gradients.

The measured scan is a circular orbit whose views have random angle errors
and whose detector is shifted. Starting from the nominal orbit, Adam fits the
per-view angles and the detector shift so that projections of the known
phantom match the measurements. The trajectory is built from these parameters
with ordinary torch operations, so gradients flow from the projector through
the trajectory tensors into the parameters.

    python examples/geometry_calibration.py
    python examples/geometry_calibration.py --devices 0 1 2 3
    torchrun --nproc-per-node=4 examples/geometry_calibration.py
"""

import argparse
import math
import time

import torch
import torch.distributed as dist

from diffct import Projector
from _common import Scan, init_process, shepp_logan_3d, synchronize


def circular_orbit(scan, angle_error, detector_shift):
    """Differentiable circular trajectory: nominal angles plus errors, shifted detector."""
    angles = (torch.arange(scan.views, dtype=torch.float64, device=angle_error.device)
              * (2 * math.pi / scan.views) + angle_error)
    c, s = torch.cos(angles), torch.sin(angles)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    source = torch.stack([-scan.sid * s, scan.sid * c, zero], dim=1)
    det_u = torch.stack([c, s, zero], dim=1)
    det_v = torch.stack([zero, zero, one], dim=1)
    idd = scan.sdd - scan.sid
    det_center = (torch.stack([idd * s, -idd * c, zero], dim=1)
                  + detector_shift[0] * det_u + detector_shift[1] * det_v)
    return source, det_center, det_u, det_v


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--views", type=int, default=360)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--angle-error-deg", type=float, default=0.5, help="RMS of the true per-view angle error")
    # A larger angle step (2e-3 rad) sends a few views into a neighbouring local minimum.
    parser.add_argument("--lr-angle", type=float, default=1e-3, help="Adam step size for the angles (rad)")
    parser.add_argument("--lr-shift", type=float, default=5e-2, help="Adam step size for the shift (voxels)")
    parser.add_argument("--devices", type=int, nargs="+", help="GPUs for one process (not with torchrun)")
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be at least 1")
    distributed, device = init_process()
    if distributed and args.devices:
        parser.error("use either torchrun or --devices, not both")
    scan = Scan(args.size, args.views)
    shape = (args.size,) * 3
    phantom = torch.from_numpy(shepp_logan_3d(args.size)).to(device)

    def make_operator(angle_error, detector_shift):
        return Projector(circular_orbit(scan, angle_error, detector_shift), shape, scan.detector,
                         detector_spacing=scan.pitch, devices=args.devices, distributed=distributed)

    # Ground truth, identical on every rank (fixed seed).
    generator = torch.Generator().manual_seed(7)
    true_angle_error = (torch.randn(args.views, generator=generator, dtype=torch.float64)
                        * math.radians(args.angle_error_deg)).to(device)
    true_shift = torch.tensor([1.5, -1.0], dtype=torch.float64, device=device)
    with torch.no_grad():
        measured = make_operator(true_angle_error, true_shift).project(phantom)

    angle_error = torch.zeros(args.views, dtype=torch.float64, device=device, requires_grad=True)
    detector_shift = torch.zeros(2, dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([{"params": [angle_error], "lr": args.lr_angle},
                                  {"params": [detector_shift], "lr": args.lr_shift}])
    rank0 = not distributed or dist.get_rank() == 0
    synchronize(distributed)
    start = time.perf_counter()
    for step in range(args.steps + 1):
        optimizer.zero_grad()
        # A new trajectory per step: the Projector keeps references to tensors that
        # require gradients and reads their current values at every call.
        operator = make_operator(angle_error, detector_shift)
        loss = (operator.project(phantom) - measured).square().sum()
        if step % 50 == 0:  # report the parameters this loss was computed with
            total = loss.detach().clone()
            if distributed:
                dist.all_reduce(total)
            angle_rms = math.degrees(torch.sqrt(torch.mean((angle_error - true_angle_error) ** 2)).item())
            shift_error = torch.linalg.vector_norm(detector_shift - true_shift).item()
            angle_max = math.degrees((angle_error - true_angle_error).abs().max().item())
            if rank0:
                print(f"step {step:4d}  loss {total.item():.3e}  angle error RMS {angle_rms:.4f} deg"
                      f" (max {angle_max:.3f})  detector shift error {shift_error:.4f} voxels")
        if step < args.steps:
            loss.backward()  # distributed mode sums geometry gradients over ranks
            optimizer.step()
    synchronize(distributed)
    if rank0:
        print(f"{1e3 * (time.perf_counter() - start) / args.steps:.1f} ms per step, "
              f"recovered shift {[round(v, 4) for v in detector_shift.tolist()]}, true {true_shift.tolist()}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
