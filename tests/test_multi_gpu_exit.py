"""Interpreter shutdown after CUDA kernels run on two devices."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import torch


@pytest.mark.cuda
def test_multi_gpu_exit():
    if torch.cuda.device_count() < 2:
        pytest.skip("At least two CUDA devices are required")

    workload = textwrap.dedent("""\
        import torch
        import diffct

        for index in (0, 1):
            device = torch.device(f"cuda:{index}")
            with torch.cuda.device(device):
                parallel = diffct.circular_trajectory_2d_parallel(4, device=device)
                fan = diffct.circular_trajectory_2d_fan(
                    4, sid=20.0, sdd=30.0, device=device
                )
                for geometry, projector, backprojector, weighted in (
                    (parallel, diffct.ParallelProjectorFunction,
                     diffct.ParallelBackprojectorFunction,
                     diffct.parallel_weighted_backproject),
                    (fan, diffct.FanProjectorFunction,
                     diffct.FanBackprojectorFunction,
                     diffct.fan_weighted_backproject),
                ):
                    image = torch.ones(
                        (8, 8), device=device, dtype=torch.float32,
                        requires_grad=True,
                    )
                    projected = projector.apply(image, *geometry, 12, 1.0)
                    projected.sum().backward()
                    sinogram = projected.detach().requires_grad_(True)
                    backprojector.apply(sinogram, *geometry, 1.0, 8, 8).sum().backward()
                    weighted(sinogram.detach(), *geometry, 1.0, 8, 8)

                cone = diffct.circular_trajectory_3d(
                    4, sid=20.0, sdd=30.0, device=device
                )
                volume = torch.ones(
                    (4, 4, 4), device=device, dtype=torch.float32,
                    requires_grad=True,
                )
                projected = diffct.ConeProjectorFunction.apply(
                    volume, *cone, 8, 8, 1.0, 1.0,
                )
                projected.sum().backward()
                sinogram = projected.detach().requires_grad_(True)
                diffct.ConeBackprojectorFunction.apply(
                    sinogram, *cone, 4, 4, 4, 1.0, 1.0,
                ).sum().backward()
                diffct.cone_weighted_backproject(
                    sinogram.detach(), *cone, 4, 4, 4, 1.0, 1.0,
                )
                torch.cuda.synchronize(device)
        """)
    env = os.environ.copy()
    repository = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join([
        str(repository),
        *(str(Path(entry or os.getcwd()).resolve()) for entry in sys.path),
    ])
    child = subprocess.run(
        [sys.executable, "-c", workload],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert child.returncode == 0, (
        f"CUDA subprocess exited with return code {child.returncode}\n"
        f"Child stderr tail:\n{child.stderr[-8000:]}"
    )
    for marker in ("Exception ignored", "Traceback", "ValueError", "OverflowError"):
        assert marker not in child.stderr, (
            f"CUDA subprocess reported {marker!r} during cleanup\n"
            f"Child stderr tail:\n{child.stderr[-8000:]}"
        )
