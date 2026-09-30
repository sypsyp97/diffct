"""Small synthetic helical reconstruction, on one GPU or under torchrun."""

import argparse
import json
import os
from pathlib import Path
import socket

import torch
import torch.distributed as dist

from diffct import Projector, spiral_trajectory_3d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This reconstruction requires a CUDA GPU")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl")
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        trajectory = spiral_trajectory_3d(
            17, sid=40.0, sdd=65.0, z_range=6.0, n_turns=1.0, device="cpu"
        )
        operator = Projector(trajectory, (16, 16, 16), (24, 20),
                             distributed=distributed)
        phantom = torch.zeros(operator.volume_shape, device=device)
        phantom[5:11, 5:11, 5:11] = 1.0
        measurements = operator.project(phantom)
        # A is nonnegative; the maximum row sum bounds the spectral radius of A^T A.
        step = 0.9 / operator.backproject(operator.project(torch.ones_like(phantom))).max()
        volume = torch.zeros_like(phantom)
        residuals = []
        for _ in range(args.iterations):
            volume = volume.detach().requires_grad_()
            loss = 0.5 * (operator.project(volume) - measurements).square().sum()
            loss.backward()
            total_loss = loss.detach().clone()
            if distributed:
                dist.all_reduce(total_loss)
            residuals.append(float(total_loss))
            volume = (volume - step * volume.grad).clamp_min(0).detach()
        final_loss = 0.5 * (operator.project(volume) - measurements).square().sum()
        if distributed:
            dist.all_reduce(final_loss)
        residuals.append(float(final_loss))
        worker = {"rank": operator.rank, "host": socket.gethostname(),
                  "gpu": torch.cuda.get_device_name(device)}
        workers = [worker]
        if distributed:
            workers = [None] * operator.world_size
            dist.all_gather_object(workers, worker)
        if operator.rank == 0:
            result = {"data": "synthetic cube phantom on a helical trajectory",
                      "world_size": operator.world_size, "workers": workers,
                      "torch": torch.__version__, "cuda": torch.version.cuda,
                      "residuals": residuals,
                      "mse": float((volume - phantom).square().mean())}
            output = json.dumps(result, indent=2)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(output + "\n", encoding="utf-8")
            print(output)
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
