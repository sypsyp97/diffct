Large Volumes and Chunked Execution
===================================

Pass CPU volumes and sinograms to ``Projector`` when complete arrays exceed GPU
memory. Its projection and matched adjoint execute on CUDA; outputs and data
gradients return to the input device. Full arrays and iterative state still
need host RAM.

.. code-block:: python

   import torch
   from diffct import Projector, circular_trajectory_3d

   trajectory = circular_trajectory_3d(60, sid=160, sdd=256, device="cpu")
   A = Projector(trajectory, (64, 64, 64), (192, 128), detector_spacing=0.8)
   x = torch.ones(A.volume_shape, device="cpu", requires_grad=True)
   y = A.project(x)                  # CPU tensor, real CUDA computation
   y.square().mean().backward()      # x.grad stays on CPU
   adjoint = A.backproject(y.detach())

Automatic selection and manual limits
-------------------------------------

``volume_chunk_shape=None`` is the automatic default. CPU data always streams.
For CUDA data, the original full-volume path is retained when its estimated
additional working set fits each selected card and the output device. Otherwise
the operation streams spatial tiles and view batches.

The arithmetic estimate accounts for float32 staging, cone layout/output
copies, view data, frame or sampled pixel geometry, VJP buffers and collective
scratch. It reserves 25% of available memory for runtime overhead, respects
PyTorch's per-process allocator ceiling and sizes against the smallest
participating budget. Concurrent allocations can still cause allocation errors.
On older PyTorch versions without the public memory-fraction getter, the
estimate can observe driver/allocator memory only; use explicit limits when a
process ceiling is set there.

To override the automatic limits:

.. code-block:: python

   A = Projector(trajectory, (64, 64, 64), (192, 128), detector_spacing=0.8,
                 volume_chunk_shape=(16, 32, 32), view_chunk_size=8)

Spatial limits follow tensor order: ``(H, W)`` for parallel/fan and ``(D, H, W)``
for cone beams. Limits are positive integral dimensions, clipped at the volume
edge. ``view_chunk_size`` may also be supplied alone to force view streaming.
Its automatic value is at most 32 and can shrink when geometry/data buffers
require it. Explicit limits are not silently reduced.

Tiles retain the global origin-centred voxel lattice. Projection adds disjoint
tile integrals; the adjoint accumulates each tile over view batches. Odd sizes,
partial tails and internal boundaries retain the same cell ownership. Chunking
preserves the cell-constant Siddon model and matched adjoint, with float32
rounding differences from a different summation order. Smaller tiles may cost
more transfers and kernel launches.

Autograd, curved detectors and several GPUs
-------------------------------------------

Data gradients and data Hessians, plus first-order frame/surface gradients, use
the streamed path too. Saved CPU geometry keeps backward consistent with its
forward call if parameters change. Geometry second derivatives remain
unsupported.

Streamed ``detector_surface`` callbacks must return CPU offsets. Complete
sampled world geometry is not expanded on CUDA. CUDA callbacks remain supported
when the automatic CUDA full-volume path fits. See :ref:`detector-surfaces`.

``devices=[0, 1]`` still partitions views in list order. Each card streams tiles;
the full output returns to the input device. Distributed ranks retain local
view shards and replicated host volumes, agree on a common execution plan, and
stage NCCL sums in bounded CUDA buffers. All ranks need the same explicit
limits and must participate in the same calls. The existing loss-scaling rules
in :doc:`multi_gpu` still apply. A solver's own scalar NCCL reductions must use
CUDA tensors.

CPU streaming currently stages devices serially. This supports bounded-memory
execution across cards; it does not promise a multi-GPU throughput improvement.

Caller-owned CUDA inputs and complete CUDA outputs still occupy their full
size. Keep arrays, trajectory tensors and learnable surface parameters on CPU
for bounded GPU residency. CUDA geometry snapshots or CUDA intermediates made
inside a callback also consume GPU memory outside the streamed tile buffers.
Analytical FBP/FDK helpers
retain their existing execution path; this feature belongs to the native
projection/adjoint pair.

Runnable iterative reconstruction
---------------------------------

The example reuses the existing CGLS solver with CPU arrays. It supports one
process with one or several GPUs and prints measured residuals and peak
PyTorch CUDA tensor allocation. The peak excludes driver/context memory.

.. code-block:: bash

   python examples/chunked_reconstruction.py
   python examples/chunked_reconstruction.py --chunk-shape 16 32 32 --view-chunk-size 8
   python examples/chunked_reconstruction.py --devices 0 1

.. literalinclude:: ../../examples/chunked_reconstruction.py
   :language: python
   :linenos:
