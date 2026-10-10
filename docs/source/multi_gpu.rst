Multiple GPUs and Nodes
=======================

By default, ``Projector`` splits views across GPUs and automatically streams spatial tiles
when arrays are CPU-backed or the estimated CUDA working set does not fit.
For volumes larger than one GPU, use CPU arrays or numerical block stores;
see :doc:`chunking`.
There are two modes:

- **One process, several GPUs:** pass ``devices`` to ``Projector``.
- **One process per GPU:** launch with ``torchrun`` and pass ``distributed=True``.
  Default ``partition="views"`` holds a local view shard and a replicated volume.
  ``partition="space"`` holds an owned volume slab and replicated ray batches.
  GPU working buffers hold bounded tiles, pixels and view batches.

Spatial ownership
-----------------

``partition="space"`` splits the first tensor axis into balanced half-open
slabs. ``global_volume_shape`` is the constructor shape; ``volume_slice`` gives
the rank's global ownership. ``volume_shape`` and ``local_volume_shape`` give
the local input/output shape. Global physical coordinates are preserved, and
empty slabs are valid. Projection SUMs slab contributions; backprojection
generates only the owned volume. Collective order does not depend on local
tile counts. Shared parameter gradients are reduced after the local pixel VJP.

In space mode, divide identical replicated projection losses by ``world_size``
before backward; sum losses on local backprojection slabs. CGLS SUMs owned
volume norms but counts replicated ray norms once. Volume regularizers with
neighbors across slabs require application-level halo exchange. The existing
TV/SIRT examples continue to use views mode. The complete ownership and loss
table is in ``docs/DISTRIBUTED.md`` in the repository.

For replicated CPU state use one process per node controlling multiple GPUs
to avoid extra host copies. For true spatial ownership use one process per
GPU, with streaming inside an owned slab when needed. Disk-backed CGLS and
rank-local checkpoints are available in ``examples/disk_reconstruction.py``;
see :doc:`chunking`. The solver capacity includes ``x,s,p`` together plus ray,
geometry and pipeline state. No view-by-space process grid is introduced.

One process, several GPUs
-------------------------

No process group is needed. Views are split into contiguous shards in the order
of the device list. Outputs return to the input tensor's device.

.. code-block:: python

   import torch
   from diffct import Projector, spiral_trajectory_3d

   trajectory = spiral_trajectory_3d(
       31, sid=80.0, sdd=128.0, z_range=8.0, n_turns=1.0, device="cpu")
   operator = Projector(trajectory, (32, 32, 32), (48, 40),
                        devices=["cuda:0", "cuda:1"])
   volume = torch.ones((32, 32, 32), device="cuda:0", requires_grad=True)
   sinogram = operator.project(volume)                # (31, 48, 40), cuda:0
   adjoint = operator.backproject(sinogram.detach())  # (32, 32, 32), cuda:0
   sinogram.square().sum().backward()                 # volume.grad on cuda:0

Use the ordinary full-sinogram loss in this mode. No world-size scaling applies.

One process per GPU
-------------------

Save the following as ``distributed_demo.py``. Each rank creates the same
volume, computes its local measurements, and differentiates one global data term.

.. code-block:: python

   import os

   import torch
   import torch.distributed as dist
   from diffct import Projector, spiral_trajectory_3d

   # Select this process's GPU before creating the NCCL process group.
   device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
   torch.cuda.set_device(device)
   dist.init_process_group("nccl")
   try:
       n_views = 31
       detector_shape = (48, 40)  # (U, V)
       trajectory = spiral_trajectory_3d(
           n_views, sid=80.0, sdd=128.0, z_range=8.0, n_turns=1.0, device="cpu")
       operator = Projector(trajectory, (32, 32, 32), detector_shape,
                            distributed=True)

       truth = torch.ones(operator.volume_shape, device=device)
       local_measurements = operator.project(truth)  # this rank's views only
       volume = torch.zeros_like(truth, requires_grad=True)
       local_prediction = operator.project(volume)

       # Divide the local sum by the global ray count, not by the shard size.
       total_rays = n_views * detector_shape[0] * detector_shape[1]
       local_loss = (0.5 * (local_prediction - local_measurements).square().sum()
                     / total_rays)
       local_loss.backward()  # every rank receives the full volume gradient

       # Reduce a detached copy for reporting only.
       report = local_loss.detach().clone()
       dist.all_reduce(report, op=dist.ReduceOp.SUM)
       if operator.rank == 0:
           print(f"global half-MSE: {report.item():.6g}")
   finally:
       dist.destroy_process_group()

Launch on one node with two GPUs:

.. code-block:: bash

   torchrun --standalone --nproc-per-node=2 distributed_demo.py

Launch on two nodes with two GPUs each. Run the same command on both nodes,
changing only ``--node-rank``. Use the hostname of node 0 as ``--master-addr``:

.. code-block:: bash

   # On node 0:
   torchrun --nnodes=2 --node-rank=0 --nproc-per-node=2 \
       --master-addr=<node0-hostname> --master-port=29500 distributed_demo.py

   # On node 1:
   torchrun --nnodes=2 --node-rank=1 --nproc-per-node=2 \
       --master-addr=<node0-hostname> --master-port=29500 distributed_demo.py

Both nodes need the same checkout and Python environment. Every rank must
join the same collective calls and backward passes, including ranks with zero
views.

Views mode: data on each rank
-----------------------------

- ``operator.view_slice``: the contiguous range of global views on this rank.
  To use full measurements, select ``full_measurements[operator.view_slice]``.
- ``operator.projection_shape``: the shape of ``project()`` output on this rank,
  and the required input shape of ``backproject()``.
- The full volume. All ranks must start with the same volume and trajectory.
  The operator does not broadcast them.
- ``backproject()`` sums the contributions of all ranks and returns the full
  volume on each rank. It is the matched adjoint, not an inverse.

Views mode: loss scaling
------------------------

The projector sums the volume and learnable-geometry gradients across ranks. Scale
the loss as follows:

- **Projection data terms:** form the loss from local views and call ``backward()``
  on every rank. Use a local ``.sum()`` for global least squares. Divide that sum by
  the global number of rays for a mean. A local ``.mean()`` weights unequal shards
  incorrectly and is undefined for an empty shard.
- **Losses on replicated backprojection output:** divide each rank's copy of the
  same output loss by ``operator.world_size`` before ``backward()``.
- **DDP:** do not add another DDP reduction for image or geometry gradients that
  ``Projector`` already reduces.

For regularizers, logging and the full rule set, see
`docs/DISTRIBUTED.md <https://github.com/sypsyp97/diffct/blob/main/docs/DISTRIBUTED.md>`_.

Iterative reconstruction with several GPUs
------------------------------------------

The iterative example runs on one process with several GPUs or with torchrun:

.. code-block:: bash

   python examples/iterative_reconstruction.py --devices 0 1 2 3
   torchrun --nproc-per-node=4 examples/iterative_reconstruction.py

For multi-node Slurm jobs, see ``examples/slurm/multi_node.sbatch`` in the
repository. Set the account, partition and GPU request for your cluster.

For the full multi-GPU and multi-node reference, see
`docs/DISTRIBUTED.md <https://github.com/sypsyp97/diffct/blob/main/docs/DISTRIBUTED.md>`_.
