# Multiple GPUs and nodes

`Projector` partitions views across GPUs/ranks. Spatial tiles and view batches
are selected automatically when data is CPU-backed or the estimated CUDA
working set does not fit. Each GPU then holds a tile instead of a complete
volume. Distributed mode stores each rank's sinogram shard; single-process
multi-GPU mode gathers the full sinogram onto the input device.

For a volume larger than one GPU, pass CPU arrays to `project()` and
`backproject()`. Full volumes and iterative state remain replicated in host RAM
across ranks. `volume_chunk_shape` and `view_chunk_size` optionally override the
automatic limits; all ranks must use the same shapes and explicit settings.
Automatic ranks agree on a common memory budget and collective schedule.
NCCL sums CPU-backed outputs/gradients through bounded CUDA staging buffers.
View ordering, empty ranks and the loss-scaling rules below still apply.

In the single-process CPU path, the cards compute their view batches
concurrently. Each card receives every volume tile, so host transfers limit the
multi-GPU speedup.

Ranks do not add memory capacity: every rank holds the complete volume and
iterative state in host RAM. For a large volume, run one process per node with
`devices=[...]` for its GPUs instead of one process per GPU.

NCCL cannot reduce CPU tensors in your own solver or reporting code. Stage
scalar reductions on the rank's CUDA device, then copy the scalar back to CPU.
The CPU CGLS example supports one process with one or several GPUs.

Install diffct as described in the [root README](https://github.com/sypsyp97/diffct/blob/main/README.md) and run
commands from the repository root. CUDA is required for projection and
backprojection. The multi-process examples below also require PyTorch with
NCCL support.

## One process, multiple GPUs

This complete example needs two visible CUDA GPUs:

```python
import torch
from diffct import Projector, spiral_trajectory_3d

trajectory = spiral_trajectory_3d(
    31, sid=80.0, sdd=128.0, z_range=8.0, n_turns=1.0, device="cpu"
)
operator = Projector(
    trajectory, (32, 32, 32), (48, 40), devices=["cuda:0", "cuda:1"]
)
volume = torch.ones((32, 32, 32), device="cuda:0", requires_grad=True)
sinogram = operator.project(volume)                 # (31, 48, 40), cuda:0
adjoint = operator.backproject(sinogram.detach())    # (32, 32, 32), cuda:0
sinogram.square().sum().backward()                   # volume.grad on cuda:0
```

No process group is required. Views are split into contiguous shards in device
list order. Outputs return to the input tensor's device as float32;
input gradients use the input's device and dtype. Uneven view counts are
supported, and devices with no views do not launch an empty CUDA grid.
Use the ordinary full-sinogram loss in this mode; no world-size scaling applies.

## One GPU per process

Save the following as `distributed_demo.py` in the repository root. It creates
the same synthetic volume on each rank, generates only the local measurements,
and differentiates one global mean-squared data term. No input files are needed.

```python
import os

import torch
import torch.distributed as dist
from diffct import Projector, spiral_trajectory_3d

# Select the process's GPU before creating the NCCL process group.
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group("nccl")
try:
    n_views = 31
    detector_shape = (48, 40)  # (U, V)
    trajectory = spiral_trajectory_3d(
        n_views, sid=80.0, sdd=128.0, z_range=8.0, n_turns=1.0, device="cpu"
    )
    operator = Projector(
        trajectory, (32, 32, 32), detector_shape, distributed=True
    )
    truth = torch.ones(operator.volume_shape, device=device)
    local_measurements = operator.project(truth)
    volume = torch.zeros_like(truth, requires_grad=True)
    local_prediction = operator.project(volume)

    # Divide each local SUM by the GLOBAL ray count, not by its shard size.
    total_rays = n_views * detector_shape[0] * detector_shape[1]
    local_loss = 0.5 * (local_prediction - local_measurements).square().sum() / total_rays
    local_loss.backward()  # every rank receives the full, summed volume gradient

    # Reduce a detached copy for reporting, not the tensor used for backward.
    global_loss = local_loss.detach().clone()
    dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
    if operator.rank == 0:
        print(f"global half-MSE: {global_loss.item():.6g}")
finally:
    dist.destroy_process_group()
```

Launch it on one node with two GPUs:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 distributed_demo.py
```

### Shard shapes and replicated state

All ranks must use identical full trajectories, shapes and spacings, and start
with the same replicated volume. The operator does not broadcast these inputs.
If geometry is learnable, keep its values and optimizer state synchronized too;
all ranks must agree on whether any trajectory tensor requires gradients.

- `operator.view_slice` identifies this rank's contiguous range of global views.
  To use existing full measurements, select
  `local_measurements = full_measurements[operator.view_slice].to(device)`.
  Alternatively, load just that range without storing the full sinogram per rank.
- `operator.projection_shape` is `(local_views, detectors)` for parallel/fan beams
  or `(local_views, U, V)` for cone beams. It is the output shape of `project()`
  and the required input shape of `backproject()`.
- `backproject(local_measurements)` sums contributions from all ranks and returns
  a full replicated volume on each rank's input device. It is the matched
  adjoint, not an inverse reconstruction.

Every rank must participate in the same collective calls and backward sequence,
including ranks with zero views. Do not put a `backproject()` or a participating
`backward()` inside a rank-0-only branch. An optional `process_group` restricts
the operator to that initialized group; `rank`, `world_size` and `view_slice`
then refer to that group. The caller owns group initialization and cleanup.

### Loss scaling

These rules apply to `distributed=True`, using the operator's process group:

- **Projection data terms:** form the loss from local views and call `backward()`
  on every rank. The operator SUM-reduces volume and learnable-geometry gradients.
  Use a local `.sum()` for global least squares; divide that sum by the global
  number of rays for a mean. A local `.mean()` weights unequal shards incorrectly
  and is undefined for an empty shard.
- **Losses on replicated backprojection output:** divide each rank's copy of the
  same output loss by `operator.world_size` before `backward()`. Backprojection
  backward sums those output cotangents across ranks.
- **Regularizers directly on the replicated volume:** add the full regularizer
  on each rank, without dividing by world size. Its gradient is computed locally;
  the projector has already summed the data-term gradient. The TV example in
  [`_common.py`](https://github.com/sypsyp97/diffct/blob/main/examples/_common.py) follows this convention.
- **Logging:** SUM-reduce a detached copy of a local data term. A regularizer that
  is already replicated should be logged once, not summed again.
- **DDP:** do not add another DDP reduction for the image or geometry gradients
  already reduced by `Projector`.

For an end-to-end reconstruction with synchronized optimizer steps, run:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
    examples/iterative_reconstruction.py
```

## Multiple nodes under Slurm

On a cluster that permits multi-node allocations, start one torchrun launcher
per node. This is the generic two-node form with one GPU per node:

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --gpus-per-task=1 \
  python -m torch.distributed.run --nnodes=2 --nproc-per-node=1 \
  --rdzv-backend=c10d --rdzv-id="$SLURM_JOB_ID" \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  examples/iterative_reconstruction.py
```

For several nodes, submit the repository template instead. It runs the same
launch for every node in the allocation:

```bash
sbatch examples/slurm/multi_node.sbatch examples/iterative_reconstruction.py --size 128
```

Set the account, partition and GPU request in `examples/slurm/multi_node.sbatch`
for your cluster, and set `GPUS_PER_NODE` to the number of GPUs on each node.

Use the Python environment and GPU resource flags appropriate to the allocation.
Launch all ranks within one cluster allocation.

## Two nodes without a Slurm launcher

Use the same checkout and CUDA Python environment on both allocated nodes.
On each node, set the first node's reachable hostname, the local GPU count,
and the node rank (`0` on the first node, `1` on the second). Both nodes must
run the following command, using the same port:

```bash
export MASTER_ADDR=first-node-hostname
export GPUS_PER_NODE=2
export NODE_RANK=0  # use 1 on the second node
python -m torch.distributed.run --nnodes=2 --nproc-per-node="$GPUS_PER_NODE" \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port=29500 \
  tests/distributed_projector_check.py --require-cuda --require-cross-node \
  --expected-world-size="$((2 * GPUS_PER_NODE))" --output nccl-crossnode.json
```

After the numeric check passes, run this on both nodes to compare complete
gradient iterations against a single GPU on the first node:

```bash
python -m torch.distributed.run --nnodes=2 --nproc-per-node="$GPUS_PER_NODE" \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port=29500 \
  examples/benchmark_projector.py --repeats 9 --output crossnode-benchmark.json
```

The two-node example defaults to two GPUs per node; set `GPUS_PER_NODE` to the
number allocated on each node. The reports are written by rank 0.

