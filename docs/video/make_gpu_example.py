"""Compute the GPU chapter's 2D data-flow illustration with diffct on the CPU.

Run from the repository root with PyTorch installed:
    python docs/video/make_gpu_example.py

These are simulated projections of the archived measured walnut slice, not
new acquisition data or GPU timing measurements. Rendering only needs NumPy.
"""
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from diffct import Projector


def build(phantom):
    image = np.asarray(phantom, dtype=np.float64).reshape(64, 4, 64, 4).mean((1, 3))
    angles = torch.arange(64, dtype=torch.float64) * (torch.pi / 64)
    direction = torch.stack((angles.cos(), angles.sin()), -1)
    detector_u = torch.stack((-angles.sin(), angles.cos()), -1)
    operator = Projector(
        (direction, torch.zeros_like(direction), detector_u), (64, 64), (96,),
        beam="parallel", backend="torch", precision="float64",
        volume_chunk_shape=(16, 64), view_chunk_size=16,
    )
    value = torch.from_numpy(image)
    projection = operator.project(value)
    row_bounds = np.arange(0, 65, 16)
    partials = []
    for start, stop in zip(row_bounds[:-1], row_bounds[1:]):
        tile = torch.zeros_like(value)
        tile[start:stop] = value[start:stop]
        partials.append(operator.project(tile).numpy())
    adjoints = []
    for start in (0, 32):
        views = torch.zeros_like(projection)
        views[start:start + 32] = projection[start:start + 32]
        adjoints.append(operator.backproject(views).numpy())
    return dict(image=image, angles=angles.numpy(), projection=projection.numpy(),
                tile_projection=np.stack(partials), row_bounds=row_bounds,
                backprojection=operator.backproject(projection).numpy(),
                view_backprojection=np.stack(adjoints))


if __name__ == "__main__":
    torch.set_num_threads(1)
    inputs = np.load(HERE / "data2d.npz")
    arrays = build(inputs["phantom"])
    np.savez_compressed(HERE / "gpu_example.npz", **arrays)
    print(f"Computed {arrays['projection'].shape} projections and their tile/view contributions")
