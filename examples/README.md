# Examples

Runnable scripts for the `Projector` API. Install diffct from a source checkout first
(see the [root README](https://github.com/sypsyp97/diffct/blob/main/README.md)), then run every command from the repository
root. Projection/reconstruction scripts need CUDA; `plot_trajectory.py` runs on
CPU. Plotting and the reconstruction scripts' `--figure` option also need
`matplotlib`, which is not a core diffct dependency.

## Start here

For the API, shapes, adjoint checks and autograd, run:

```bash
python examples/quickstart.py
```

For native curved-detector projection, matched backprojection, an adjoint check
and volume/surface gradients on a small circular cone scan, run:

```bash
python examples/curved_detector.py
```

This example uses the curved operators directly, without resampling or FDK.
The analytical reconstruction examples continue to use flat detectors. The
[surface API](../docs/REFERENCE.md#parameterized-detector-surfaces) describes
the callback coordinates, output shapes and first-order geometry gradients.

For a smaller iterative demonstration on one GPU:

```bash
python examples/iterative_reconstruction.py --size 32 --views 32 \
    --trajectory helical --algorithms cgls --iterations 5
```

This still computes an FDK baseline. Five CGLS steps show how to use the script.
The default iterative run uses a 128³ volume, 360 views, and CGLS/SIRT/TV with 30/200/200
iterations respectively. Its TV weight is 1.0.
The measured walnut script uses SIRT/CGLS/TV with 200/20/300 iterations and TV weight 0.3.
These weights are example settings. Choose the TV weight for your geometry,
voxel spacing, resolution and data scale.

To inspect a helical trajectory without a GPU (PyTorch and matplotlib required):

```bash
python examples/plot_trajectory.py --trajectory spiral3d --device cpu \
    --output plots/spiral3d.png
```

The reconstruction CLI calls this trajectory `helical`; the plotting CLI calls
it `spiral3d`, and both use `spiral_trajectory_3d`. Use `--help` on the configurable
scripts to see their own options.

## Index

| File | What it does | Launch modes |
| --- | --- | --- |
| `quickstart.py` | Projector basics for parallel, fan and cone beams: projection, backprojection, adjoint check, image and geometry gradients. | 1 GPU |
| `curved_detector.py` | Native cylindrical cone detector: projection, matched backprojection, adjoint check and volume/surface gradients. | 1 GPU |
| `chunked_reconstruction.py` | CPU-backed CGLS with automatically sized CUDA tiles; optional `--chunk-shape` and `--view-chunk-size`; reports residual and peak CUDA tensor allocation. | 1 GPU, `--devices` in one process |
| `analytical_reconstruction.py` | Parallel-beam FBP, fan-beam FBP and cone-beam FDK. `--window` selects the ramp-filter window. | 1 GPU |
| `iterative_reconstruction.py` | Cone-beam reconstruction with `--trajectory circular`, `helical`, `saddle` or `sinusoidal`. CGLS, SIRT and TV-regularized nonnegative least squares (Adam through autograd). FDK baseline outside distributed mode. `--noise` adds Gaussian noise. | 1 GPU, `--devices`, torchrun, Slurm multi-node |
| `geometry_calibration.py` | Recovers per-view angle errors and a detector shift from projections, using geometry gradients. | 1 GPU, `--devices`, torchrun, Slurm multi-node |
| `benchmark_projector.py` | Checks correctness and measures speed of one GPU against several GPUs. | `--devices` in one process, torchrun |
| `plot_trajectory.py` | Plots a trajectory generator. | CPU |
| `slurm/multi_node.sbatch` | Template: one torchrun launcher per node. | Slurm |

Shared helpers are in `_common.py`: the Shepp-Logan phantom, the scan geometry, FDK, CGLS, SIRT and TV.

## Launch modes

The reconstruction and geometry-calibration scripts support all four modes
below. Other scripts support only the modes in the index. Use the command that
matches your allocation. These existing reconstruction/calibration scripts use
CUDA-resident volumes. For CPU-backed large volumes use `chunked_reconstruction.py`;
its automatic spatial tiles and view batches bound GPU working buffers.

**1. One GPU**

```bash
python examples/iterative_reconstruction.py
```

**2. One process, several GPUs**

```bash
python examples/iterative_reconstruction.py --devices 0 1 2 3
```

The views are split over the listed GPUs and the full sinogram is returned on
the input device. No process group is needed.

**3. One node, torchrun (one process per GPU)**

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 \
    examples/iterative_reconstruction.py
```

With multiple ranks, each process selects its GPU from `LOCAL_RANK`, creates an
NCCL process group, and stores its own sinogram shard. The volume and optimizer
state are replicated. Do not pass `--devices` for these multi-process runs.
The iterative script skips the FDK baseline in distributed mode because it does
not gather the full sinogram. When using its `--noise` option, keep `--views`
at least as large as the number of processes: the example's noise setup requires
a nonempty local shard, even though `Projector` itself supports empty shards.

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
python -m torch.distributed.run --standalone --nproc-per-node=2 \
    examples/benchmark_projector.py
```

By default, the benchmark requires at least two GPUs and exits nonzero if a
requested workload fails its numerical checks or minimum speedup (default:
1.0× for the complete gradient iteration). `--min-speedup=0` also permits a
single-GPU utility check. A completed run can still fail its acceleration check.
Small workloads are not guaranteed to speed up. See the
[distributed guide](https://github.com/sypsyp97/diffct/blob/main/docs/DISTRIBUTED.md) for full standalone examples,
cross-node checks and memory constraints.

## Rules for distributed losses

These rules apply when you write your own loss with `Projector(..., distributed=True)`. The examples follow them.

- Projection losses: each rank computes a sum on its own view shard. For a global mean, divide each local sum by the global ray count; a local `.mean()` is wrong for unequal shards and undefined for empty shards. The projector sums image and learnable-geometry gradients across ranks.
- For reporting a global data loss, `dist.all_reduce` a detached clone of the local scalar. Do not all-reduce the loss used for `backward()` yourself.
- Losses on the output of `backproject()`: divide each rank's identical replicated-output loss by `operator.world_size` before `backward()`. Backprojection backward sums the output gradients over ranks.
- Losses on the replicated volume only (for example a TV term on `x`): add the full term on every rank. Do not divide it by the number of ranks. Each rank computes the same gradient, and the projector already sums the data gradient over ranks.
- Every rank must make the same calls in the same order, including `project()`, `backproject()` and `backward()`. This applies to ranks with zero views.
- Do not add a DDP reduction on top. The image-gradient collectives already sum over ranks.

`iterative_reconstruction.py` and `geometry_calibration.py` show these rules in use.

## Example scan

The examples use this scan for an n³ volume, unit voxels:

- Source 2.5 n from the isocentre.
- Detector 4 n from the source (magnification 1.6).
- Detector array of (3n, 2n) cells.
- Pitch 0.8 (0.5 voxel at the isocentre).
- 360 views.
