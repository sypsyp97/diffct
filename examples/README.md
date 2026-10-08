# Examples

Runnable scripts for the `Projector` API. Install diffct first (see the [root README](../README.md)), then run every command from the repository root.
All scripts need a CUDA GPU, except `plot_trajectory.py`.

## Index

| File | What it does | Launch modes |
| --- | --- | --- |
| `quickstart.py` | Projector basics for parallel, fan and cone beams: projection, backprojection, adjoint check, image and geometry gradients. | 1 GPU |
| `analytical_reconstruction.py` | Parallel-beam FBP, fan-beam FBP and cone-beam FDK. `--window` selects the ramp-filter window. | 1 GPU |
| `iterative_reconstruction.py` | Any trajectory (`--trajectory circular/helical/saddle/sinusoidal`). CGLS, SIRT and TV-regularized nonnegative least squares (Adam through autograd). FDK baseline. `--noise` adds Gaussian noise. | 1 GPU, `--devices`, torchrun, Slurm multi-node |
| `geometry_calibration.py` | Recovers per-view angle errors and a detector shift from projections, using geometry gradients. | 1 GPU, `--devices`, torchrun, Slurm multi-node |
| `benchmark_projector.py` | Checks correctness and measures speed of one GPU against several GPUs. | `--devices` in one process, torchrun |
| `plot_trajectory.py` | Plots a trajectory generator. | CPU |
| `slurm/multi_node.sbatch` | Template: one torchrun launcher per node. | Slurm |

Shared helpers are in `_common.py`: the Shepp-Logan phantom, the scan geometry, FDK, CGLS, SIRT and TV.

## Launch modes

Each mode uses the same script. Use the command that matches your allocation.

**1. One GPU**

```bash
python examples/iterative_reconstruction.py
```

**2. One process, several GPUs**

```bash
python examples/iterative_reconstruction.py --devices 0 1 2 3
```

The views are split over the listed GPUs. No process group is needed.

**3. One node, torchrun (one process per GPU)**

```bash
torchrun --nproc-per-node=4 examples/iterative_reconstruction.py
```

Do not pass `--devices` with torchrun. The script stops with an error when both are given.

**4. Several nodes, Slurm**

```bash
sbatch examples/slurm/multi_node.sbatch examples/iterative_reconstruction.py --trajectory helical
```

The template starts one torchrun launcher per node and uses NCCL between all GPUs. Before you submit:

1. Edit `#SBATCH --account` to your account.
2. Edit `#SBATCH --partition` to your GPU partition.
3. Edit `#SBATCH --nodes` and `#SBATCH --gres=gpu:4` to the node count and GPU count for your job.
4. Set `GPUS_PER_NODE` to the number of GPUs on each node. The default is 4. For example, `GPUS_PER_NODE=8 sbatch ...`.

`benchmark_projector.py` checks correctness and speed. It uses these two launch forms (also listed in its docstring):

```bash
python examples/benchmark_projector.py --devices 0 1
torchrun --standalone --nproc-per-node=2 examples/benchmark_projector.py
```

## Rules for distributed losses

These rules apply when you write your own loss with `Projector(..., distributed=True)`. The examples follow them.

- Projection losses: each rank computes its term on its own view shard. The total loss is the sum of these terms over ranks. Use `dist.all_reduce` on a scalar before you log it or compare it.
- Losses on the output of `backproject()`: divide the loss by `world_size` before `backward()`. Backprojection backward sums the output gradients over ranks.
- Losses on the replicated volume only (for example a TV term on `x`): add the full term on every rank. Do not divide it by the number of ranks. Each rank computes the same gradient, and the projector already sums the data gradient over ranks.
- Every rank must make the same calls in the same order, including `project()`, `backproject()` and `backward()`. This applies to ranks with zero views.
- Do not add a DDP reduction on top. The image-gradient collectives already sum over ranks.

`iterative_reconstruction.py` and `geometry_calibration.py` show these rules in use.

## Results

All numbers are measured on Leonardo Booster (A100 64 GB, PyTorch 2.10 cu126, numba-cuda 0.30.4) on 2026-10-09.

### Iterative reconstruction

128³ Shepp-Logan, 360 views, one GPU, noise-free. Values are PSNR (dB).

| trajectory | FDK | CGLS 30 it | SIRT 200 it | TV 200 it |
| --- | ---: | ---: | ---: | ---: |
| circular | 34.10 | 38.58 | 32.73 | 48.37 |
| helical | 32.13 | 38.49 | 32.80 | 48.37 |
| saddle | 29.83 | 38.82 | 32.63 | 48.27 |
| sinusoidal | 33.84 | 38.49 | 32.76 | 48.50 |

FDK is designed for the circular orbit and is approximate away from its central plane. For the other orbits the approximation is coarser.

With 1% Gaussian noise (helical): FDK 31.19, CGLS 35.83, SIRT 32.68, TV 47.46.

Time per iteration, same helical run (CGLS / SIRT / TV):

| launch mode | CGLS | SIRT | TV |
| --- | ---: | ---: | ---: |
| 1 GPU | 69.2 ms | 66.2 ms | 67.5 ms |
| one process, 4 GPUs | 22.7 ms | 20.3 ms | 21.7 ms |
| torchrun, 4 GPUs | 19.3 ms | 17.5 ms | 18.5 ms |
| torchrun, 8 GPUs on 2 nodes | 12.0 ms | 10.6 ms | 11.5 ms |

PSNR is identical in every launch mode.

### Projector scaling

Circular scan, 360 views, detector (2n, 1.5n) with pitch 1.25, 128³ volume. Times are for 1 GPU, then 4 GPUs (1 node), then 8 GPUs (2 nodes). Speedup is relative to 1 GPU.

| operation | 1 GPU | 4 GPUs | 8 GPUs (2 nodes) |
| --- | ---: | ---: | ---: |
| forward | 8.46 ms | 2.67 ms (3.2×) | 1.69 ms (5.0×) |
| adjoint | 20.69 ms | 5.91 ms (3.5×) | 4.53 ms (4.6×) |
| CGLS iteration | 30.07 ms | 8.09 ms (3.7×) | 5.54 ms (5.4×) |

64³ CGLS iteration: 4.75 ms on 1 GPU, 1.57 ms on 4 GPUs (3.0×), 1.56 ms on 8 GPUs (3.0×). Small problems are dominated by transfers and kernel launches.

### Analytical reconstruction

128² and 128³ Shepp-Logan. PSNR (dB):

| window | parallel FBP | fan FBP | cone FDK |
| --- | ---: | ---: | ---: |
| shepp-logan | 38.10 | 34.92 | 34.10 |
| hann | 27.66 | 28.12 | 29.22 |

The window and the detector sampling set the sharpness.

### Geometry calibration

64³ volume, 360 views. The true per-view angle error is 0.54° RMS. The true detector shift is (1.5, -1.0) voxels. Adam runs 300 steps.

- Angle error after calibration: 0.0001° to 0.0025° RMS.
- Detector shift: recovered to less than 0.001 voxel in every launch mode.
- Time per step: 19.4 ms on 1 GPU, 8.2 ms on torchrun with 4 GPUs, 6.7 ms on 8 GPUs (2 nodes).

### FDK check against ASTRA

FDK from diffct matches ASTRA 2.5.0 `FDK_CUDA` on the same projections. The PSNR difference is within 0.3 dB. The maximum pixel difference is below 1% of the phantom maximum.

## Example scan

The examples use this scan for an n³ volume, unit voxels:

- Source 2.5 n from the isocentre.
- Detector 4 n from the source (magnification 1.6).
- Detector array of (3n, 2n) cells.
- Pitch 0.8 (0.5 voxel at the isocentre).
- 360 views.
