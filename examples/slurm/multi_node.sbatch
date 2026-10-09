#!/bin/bash
# Run a diffct example on several nodes: one torchrun launcher per node,
# one process per GPU, NCCL between all GPUs.
#
#   sbatch examples/slurm/multi_node.sbatch examples/iterative_reconstruction.py --size 128
#
# Set account, partition and GPU request for your cluster. The script
# assumes GPUS_PER_NODE GPUs on every node and runs from the repository root.
#SBATCH --job-name=diffct-multi-node
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00

set -euo pipefail
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# A port per job, so that two jobs sharing a first node do not collide.
MASTER_PORT=${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 1000))}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

srun --export=ALL --nodes="$SLURM_NNODES" --ntasks-per-node=1 \
  python -m torch.distributed.run \
    --nnodes="$SLURM_NNODES" --nproc-per-node="$GPUS_PER_NODE" \
    --rdzv-backend=c10d --rdzv-id="$SLURM_JOB_ID" \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    "$@"
