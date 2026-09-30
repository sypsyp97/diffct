# Multiple GPUs and nodes

`Projector` partitions views, not voxels. Every GPU holds a full volume. This
reduces per-device projection work and sinogram storage; it does not pool GPU
memory for a volume that cannot fit on one device.

## One process, multiple GPUs

```python
operator = Projector(trajectory, (32, 32, 32), (48, 40),
                     devices=["cuda:0", "cuda:1"])
sinogram = operator.project(volume)
adjoint = operator.backproject(sinogram)
```

No process group is required. Outputs and gradients belong to the input tensor's
device. Uneven view counts are supported and devices with no views contribute
zero without launching an empty CUDA grid.

## One GPU per process

Set the CUDA device before initializing NCCL:

```python
import os
import torch
import torch.distributed as dist
from diffct import Projector

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl")
operator = Projector(trajectory, (32, 32, 32), (48, 40), distributed=True)
local_measurements = full_measurements[operator.view_slice]
local_prediction = operator.project(volume)
loss = (local_prediction - local_measurements).square().sum()
loss.backward()  # each rank receives the summed image gradient
dist.destroy_process_group()
```

All ranks must use identical trajectory and shape settings and start with the
same replicated volume. `project()` returns only the rank's contiguous shard;
`projection_shape` describes its shape. `backproject()` takes that shard and
returns the sum over all ranks, replicated on each input device. Both operations
support ordinary autograd. Backprojection backward sums output cotangents across
ranks, so divide a loss on its replicated output by `world_size`. Do not combine
these image-gradient collectives with an extra DDP reduction.

Every rank must participate in the same collective calls and backward sequence,
including ranks with zero views. An optional `process_group` restricts the
operator to that initialized group. The caller owns group initialization and
cleanup; the library does not silently open network connections.

Run the included small iterative reconstruction:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
    examples/distributed_reconstruction.py
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
  examples/distributed_reconstruction.py
```

Alex requires explicit multi-node authorization and a matching `a100multi` or
`a40multi` QoS. Its multi-node allocations reserve all eight GPUs on each node;
use `--gres=gpu:a100:8 --qos=a100multi` when allocating, and
`--nproc-per-node=8` with one launcher per node to use them. The ordinary Alex
and tinygpu partitions have `MaxNodes=1`. Alex's FAU allocation does not permit
multi-node jobs; a separate NHR project is required. See the
[official Alex multi-node instructions](https://doc.nhr.fau.de/clusters/alex/#multi-node-job-available-on-demand-for-nhr-projects).
Actual cross-node GPU validation remains pending on a cluster with the required
allocation permission.

Use the Python environment and GPU resource flags appropriate to the allocation.
Launch all ranks within one cluster allocation. Running between two
independent clusters requires working inter-cluster NCCL networking and is not
implied by validating two nodes within a cluster.

## Two-node checks without a Slurm launcher

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

## Numeric checks

```bash
python -m pytest tests/test_projector_api.py -q
python -m torch.distributed.run --standalone --nproc-per-node=2 \
    tests/distributed_projector_check.py --require-cuda \
    --expected-world-size=2 --output result.json
```

Use the same two-node launch above with the check script and
`--require-cross-node` to verify cross-node execution. This flag requires at
least two distinct hostnames. The JSON records every rank's actual host, GPU,
environment and numeric errors. These checks execute the production operator
with NCCL and CUDA kernels and fail when the required hardware is unavailable.

## Correctness and acceleration

The result-based benchmark first checks single-GPU ray lengths against an
independent CPU float64 calculation for uniform boxes. It then compares
single- and multiple-GPU projections, backprojections and image gradients.
The cone trajectory is helical. Timings include per-call transfers, gathering,
autograd and distributed reductions, after kernel warmup:

```bash
python examples/benchmark_projector.py --devices 0 1 --output local-benchmark.json
python -m torch.distributed.run --standalone --nproc-per-node=2 \
    examples/benchmark_projector.py --output nccl-benchmark.json
```

The default size sweep uses 64 and 128 cubed volumes and 512 views. Each report
contains repeated timings and the measured speedup over one GPU. The benchmark
requires the complete gradient iteration to run faster for each requested
workload; it exits nonzero if numeric checks or the acceleration threshold fail.
Use `--sizes`, `--views`, and `--repeats` for your actual workload. Tiny problems
can be dominated by transfer and launch overhead, so test the acquisition size
you intend to run rather than assuming a fixed speedup from GPU count.

See [measured results](VALIDATION.md): two A100s in NCCL mode accelerated both
tested sizes; the single-process `devices` mode accelerated 128 cubed but was
slower for 64 cubed. The default benchmark preserves that failure.
