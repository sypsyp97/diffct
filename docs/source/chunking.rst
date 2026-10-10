Large Volumes and Chunked Execution
===================================

Pass CPU volumes and sinograms to ``Projector`` when complete arrays exceed GPU
memory. Projection and the matched adjoint execute on CUDA; outputs and data
gradients return to the input device. Tensor methods still hold complete arrays
on that device. For data exceeding host RAM, use the caller-owned block-store
methods below and the disk-backed CGLS example.

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
                 volume_chunk_shape=(16, 32, 32), view_chunk_size=8,
                 detector_chunk_shape=(64, 64))

Spatial limits follow tensor order: ``(H, W)`` for parallel/fan and ``(D, H, W)``
for cone beams. Limits are positive integral dimensions, clipped at the volume
edge. ``view_chunk_size`` may also be supplied alone to force view streaming.
Its automatic value is at most 32 and can shrink when geometry/data buffers
require it. ``detector_chunk_shape`` follows ``(U,)`` or ``(U,V)`` order.
Automatic execution can split a single detector image when it does not fit;
reducing the volume alone cannot solve that case. Pixel rectangles keep the
global detector origin. Explicit limits must fit the planned working set.

Tiles retain the global origin-centred voxel lattice. Projection adds disjoint
tile integrals; the adjoint accumulates each tile over view batches. Odd sizes,
partial tails and internal boundaries retain the same cell ownership. Chunking
preserves the cell-constant Siddon model and matched adjoint, with float32
rounding differences from a different summation order. Smaller tiles may cost
more transfers and kernel launches.

Resident layouts and accumulation
---------------------------------

For backprojection, each active GPU keeps one contiguous native-layout
accumulator for a spatial tile. Every view batch adds to that accumulator;
layout conversion and return to the caller happen after all views contribute.
The accumulator is cleared once per tile. Projection prepares the native volume
layout once per tile/device and reuses its ray output buffer across view
batches. Geometry VJPs also prepare each tile once across their view batches. Cone kernels
retain contiguous ``(W, H, D)`` access; tensor volumes remain ``(D, H, W)``.
Forward kernels overwrite each active ray, including missed rays and short
final batches, so they do not need unconditional output clearing.

For example, a ``1024**3`` float32 volume is 4 GiB. With 720 views in batches of
32, returning a complete partial volume after each of the 23 batches would
transfer 92 GiB. Returning the completed accumulated volume transfers 4 GiB
on a single GPU, excluding projections, geometry and other solver data. This
is a copy-payload estimate; it does not imply a 23-fold reconstruction speedup.
Float32 summation order can change slightly, while the underlying Siddon cell
model and matched adjoint stay the same.

Scheduling and transfers
------------------------

``schedule="auto"`` compares a few spatial-first, view-first and bounded-window
orders, including contiguous z slabs and three-dimensional blocks. Selection
uses bounded real CUDA pilot times scaled by the full native launch count,
with estimated full-workload copy payloads as a tie-breaker. The pilots use
small residency/order prefixes, so selection is a heuristic; reduced copy
payloads alone do not establish a runtime improvement. The benchmark example
preserves the initial selection report separately from its warmed timings.
``schedule="spatial"``, ``"views"`` or ``"window"`` enables a reproducible
comparison. Backprojection always completes each retained volume accumulator
before returning it. View-first backprojection is eligible only when all its
accumulators fit together.

The transfer pipeline uses at most two reusable pinned slots per device,
separate upload, compute and download streams, and CUDA events. Events protect
input consumption, host output reads and reuse of GPU buffers. Each composite
host slot is capped at 8 MiB, with a separate 8 MiB pageable read/cast scratch
ceiling. These are memory policy ceilings. Larger resident GPU tiles use
bounded host fragments; fragmentation changes copy call counts, while the
completed volume payload is transferred once. Planned local surface geometry
also has an independent host working-set ceiling. Arbitrary allocations inside
a user sampler and OS file-cache residency remain outside these ceilings.
All additional GPU slots, layouts, geometry and collective staging contribute
to the GPU budget. Very small budgets can use a bounded serial fallback.

``A.last_execution_stats`` reports the last instrumented operation's schedule,
limits, layout/kernel counts and actual upload/download/peer-copy payloads.
Pilot timing and payloads are reported separately; candidate full-workload
costs are estimates. These counts are not physical bus measurements. The value
is ``None`` before execution or for an uninstrumented full-CUDA path. Run
``python examples/benchmark_execution.py`` to compare schedules and slab/block
shapes on your hardware.

Caller-owned blocks and disk files
------------------------------------------------------------

``TensorStore`` provides detached numerical block access to an existing tensor.
``NpyStore`` maps a float32/float64 ``.npy`` file. Creating a store reserves a new
path exclusively; existing files are not overwritten. Block reads/casts and
writes do not materialize a complete mapped array.

.. code-block:: python

   from diffct import NpyStore

   # The file has shape A.volume_shape, in tensor axis order.
   source = NpyStore("volume.npy", mode="r")
   rays = NpyStore.create("projections.npy", A.projection_shape)
   A.project_into(source, rays)
   rays.flush()
   result = NpyStore.create("adjoint.npy", A.volume_shape)
   A.backproject_into(rays, result)
   result.flush()

``project_into`` and ``backproject_into`` return the supplied output. They
accept tensors, these stores, or structural backends with ``shape``, ``dtype``,
``read(slices)``, ``write(slices,value,accumulate=False)`` and ``flush()``. Outputs must be
float32 and writable. Each completed output region overwrites its prior
contents; no full result is allocated internally. These numerical methods do
not build an autograd graph. Use tensor ``project``/``backproject`` for
differentiation.

Input and output must have distinct backing data. Identical objects,
detectable overlapping tensors and the same canonical mapped file path are
rejected before execution. Custom backends must guarantee that hidden backing storage is
distinct; an opaque object's path attribute alone is not an alias guarantee.
Calling a store's ``read`` with a complete array slice can still request a
complete array; keep application-side reads and vector updates blockwise too.

Autograd, curved detectors and several GPUs
-------------------------------------------

Data gradients and data Hessians, plus first-order frame/surface gradients, use
the streamed path too. Saved CPU geometry keeps backward consistent with its
forward call if parameters change. Geometry second derivatives remain
unsupported.

For bounded geometry, use ``ParameterizedSurface(sampler, parameters=...)``.
Its sampler receives global physical pixel centres, global view IDs and
explicit parameter snapshots, and generates only the executing view/pixel
batch. Pixel cotangents are reduced through the local sampler/frame graph to
parameter gradients before communication. The captured sampler and one set of
frame/parameter snapshots preserve backward behavior after parameter mutation.
See :ref:`detector-surfaces`.

Legacy two-argument callbacks still sample their complete detector geometry
on the host and retain that footprint. Streamed legacy callbacks must return
CPU offsets. CUDA callbacks remain supported when the automatic CUDA
full-volume path fits. Callback values are not cached across operations.

``devices=[0, 1]`` still partitions views in list order. Each card streams tiles;
the full output returns to the input device. Distributed ``partition="views"``
ranks retain local view shards and replicated volumes. Local GPU backprojection
sums and NCCL reductions occur while the tile is resident, before its final
host transfer. Gloo uses bounded CPU staging. All ranks need matching settings
and must participate in the same calls, including empty ranks. The
ownership-specific loss rules in :doc:`multi_gpu` apply. Scalar NCCL reductions
also use CUDA tensors.

With CPU streaming, the cards compute their view batches concurrently. Each
card receives every volume tile, so host transfers limit the multi-GPU speedup.

With ``partition="views"``, every rank holds the complete tensor volume and
iterative state. One process per node controlling ``devices=[...]`` avoids
extra host replicas. ``partition="space"`` instead assigns balanced contiguous
first-axis slabs to ranks. ``global_volume_shape`` retains the constructor
shape; ``volume_slice`` gives global ownership and ``volume_shape``/
``local_volume_shape`` give the required local tensor/store shape. Projection
SUMs the slab contributions into replicated ray batches; backprojection
returns only the owned slab. Physical coordinates remain on the global voxel
lattice. Empty slabs are valid. This is one spatial process dimension, without
a combined view-by-space process grid.

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

The tensor CGLS helper computes FP64 norms in bounded pieces and reuses vector
update buffers. Its complete tensor states still need memory. Disk-backed
CGLS keeps ``x,s,p`` plus ``r,q`` in stores, performs blockwise vector updates
and writes completed-iteration checkpoints without gathering the volume.
``q`` remains available until the global norm determines ``alpha`` and the
``x``/residual updates finish. Account for all three volume states together;
fitting one volume does not establish that the complete solver fits.
The example helpers set ``operator.last_solver_stats`` with state and scratch
capacity metadata. This report belongs to the examples; the core package
continues to provide projection and backprojection. The disk entrypoint saves
it alongside the actual residual, elapsed time and CUDA tensor peak.

.. code-block:: bash

   python examples/disk_reconstruction.py --output disk-run
   python examples/disk_reconstruction.py --surface saddle --output saddle-run
   torchrun --standalone --nproc-per-node=2 examples/disk_reconstruction.py \
       --partition space --output spatial-run

The default disk example generates explicitly synthetic measurements
blockwise. Each rank owns its output, workspace and checkpoint files. Resume
requires matching partition, ownership, shapes, scan and measurements. Files
back the arrays; the OS may cache their pages in RAM. Local tests and bounded
models provide evidence separately from physical multi-GPU or cross-node runs.
The iteration count is the target total: resuming iteration 2 with
``--iterations 5`` performs three further iterations in a new run directory.

.. literalinclude:: ../../examples/chunked_reconstruction.py
   :language: python
   :linenos:
