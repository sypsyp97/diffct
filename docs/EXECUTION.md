# Precision, training and reconstruction

`Projector`, `fbp` and `fdk` expose `backend` and `precision` explicitly.
The native NVIDIA path remains the default when compatible. The PyTorch
backend uses the same cell-constant Siddon model and physical coordinates.

| Setting | Computation | Devices |
|---|---|---|
| `backend="auto", precision="fast"` | Native FP32 for scalar or per-axis spacing and views/space partitions; otherwise PyTorch FP32 | NVIDIA CUDA, or the input's PyTorch device |
| `backend="numba"` | Native FP32 fast-math traversal, with FP64 ray clipping and entry setup | NVIDIA CUDA; `precision="fast"`, views/space |
| `backend="torch", precision="float32"` | PyTorch FP32 data and local traversal, with FP64 ray clipping and entry setup | CPU, NVIDIA CUDA, PyTorch ROCm |
| `precision="float64"` | PyTorch FP64 throughout the data operations, including analytical FFT/filter/gather | CPU, NVIDIA CUDA, PyTorch ROCm |

`backend="auto"` selects PyTorch for strict precision modes, hybrid
ownership, or `parallel3d` beams. An incompatible explicit Numba
request raises an error. `Projector.backend`, `.precision` and `.dtype` describe the selected
execution. Supplying a float64 input to `precision="fast"` still yields FP32.
Create trajectories and data in float64 for a double-precision baseline;
casting an already rounded trajectory cannot recover its original precision.
Matched interpolation/traversal coefficients do not promise bitwise
determinism of floating-point scatter accumulation.

FP32 fan/cone modes retain the acquisition distance guards: each view (or
sampled surface ray) needs one endpoint within `1e6` voxels of the volume
center, and both endpoints within `1e15`. With per-axis pitches, distance is
`norm((x/dx, y/dy[, z/dz]))`. FP64 validates finite, distinct endpoints without
these FP32 cutoffs. Segment setup starts from the endpoint nearer the volume
center, then clips and locates each tile's entry using bounded per-ray FP64
vectors. Traversal parameters are measured locally from that entry in the
selected precision, preserving voxel widths when both endpoints are distant.
Native surface preprocessing can preserve a higher-precision geometry dtype
before staging completed rays to FP32; its data transforms retain that behavior.
Explicit PyTorch FP32 validates component arithmetic in FP32. Use FP64 when
that arithmetic rounds distinct source/pixel coordinates together.

Install the appropriate PyTorch build first. `pip install diffct` suffices for
the PyTorch backend. Importing the library and using this backend do not load
Numba CUDA or require a CUDA runtime; native execution initializes them on
first use. The CUDA compiler extras are needed for native Numba execution.
ROCm uses PyTorch's `"cuda"` device namespace, as described in the
[PyTorch HIP notes](https://docs.pytorch.org/docs/stable/notes/hip.html).
Hardware support depends on the installed PyTorch build and process-group
backend. Existing published native performance numbers do not measure the
portable precision backend or hybrid execution.

## Call-time trajectory validation

An ordinary flat detector's trajectory override is checked at its corners,
with work proportional to the view count. If these bounds cannot exclude a
source/pixel coincidence in the execution dtype, the affected views fall back
to bounded pixel checks. Curved detectors still check sampled pixels. Native
admission snapshots surface frames and parameters on CPU, then validates owned
batches on compute devices. Validation batches shrink to fit memory and staging
limits; they are separate from execution chunks. Portable admission stages
frames once per validation device.

Retained constructor frames with gradients use the same admission on every
call, so optimizer updates cannot bypass geometry checks. Flat CUDA frames
stay on an accelerator for range and pixel checks; only scalar decisions
need host reads. Fixed constructor trajectories remain snapshots.
Updated parameterized surfaces are checked on portable calls and native
data transforms. Native tensor-returning calls check each batch before use;
`*_into` calls preflight all batches before changing caller output.

Native CUDA calls with either surface interface use full execution when the
device memory, allocation and reservation limits admit the workload. Full
geometry backward retains the surface composition graph on the compute device
and pulls back all owned pixels together. This avoids per-row host autograd
replay. Explicit chunk limits and workloads that cannot fit still use bounded
streaming.

Run `python examples/benchmark_surface_geometry.py` for a synthetic helical
scan with 1338 views, a 736x64 curved detector and a 96x512x512 volume. It reports
separate synchronized forward, adjoint and geometry-gradient timings for both
surface interfaces. By default it reports and skips streamed plans; use
`--allow-streamed` to time that potentially slower fallback.

Run `python examples/benchmark_trajectory_admission.py --warmup 1 --samples 3`
to measure this validation with CPU and CUDA trajectories. The script bypasses
numerical projection and reports admission time, plus a separate untimed pass
counting validation rays and tensor transfers. These are not projection
throughput measurements. For CPU-only admission measurements, add
`--backend torch --frame-device cpu`.

Run `python examples/benchmark_gpu_overheads.py --warmup 3 --samples 9` to
measure launch and transfer overheads on a small synthetic cone-beam case. It
reports synchronized wall times and incremental peak allocation for trainable
trajectories, legacy, per-view and sampled surfaces, CGLS, SIRT, TV and tiled
FDK. Run it once for each checkout with `PYTHONPATH` set to that checkout.
Small cases expose dispatch costs; do not extrapolate their ratios to large,
traversal-bound reconstructions.

## Batch dimensions and axis spacing

Any leading batch/channel dimensions precede the configured volume or ray
shape. Projection and adjoint preserve them, including empty dimensions.
`voxel_spacing=(dy, dx)` or `(dz, dy, dx)` follows tensor axis order; trajectory
coordinates remain `(x, y[, z])`. Spacings are fixed physical settings.
Both backends support per-axis spacing. Native traversal scales each world
coordinate by its axis pitch and integrates in physical length units.
Each native per-axis pitch must lie in the closed practical range `[1e-6, 1e6]`.
The largest pitch divided by the smallest must not exceed `1e4`.
These bounds are checked before FP32 conversion. FP32 representability alone
does not prevent extreme pitches from erasing ray segments or overflowing
geometry gradients. The conservative range limits these scaling effects;
existing FP32 geometry range checks still apply.
`backend="numba"` raises outside these bounds. `backend="auto"` selects PyTorch;
on NVIDIA CUDA, its fallback warning names the violated bound.
Scalar spacing retains its existing admission checks.

Per-axis finite rays are degenerate only when their endpoints coincide.
Detector offsets multiply their axis components before division by the pitch.
Endpoint metric derivatives use FP64 intermediates before the final FP32 cast.
Scalar spacing retains its existing arithmetic and degeneracy threshold.
Helical reconstruction accepts `(dz, dy, dx)` and uses `dz` for slice positions.

For 2D parallel rays, scalar spacing preserves the original direction-gradient
convention. Per-axis spacing differentiates physical ray length. The gradients
can differ in their radial component. Compare direction perturbations tangent
to the required unit vector, or parameterize directions with angles.

```python
import torch
from diffct import Projector, circular_trajectory_2d_parallel

trajectory = circular_trajectory_2d_parallel(12, device="cpu", dtype=torch.float64)
A = Projector(trajectory, (8, 10), 16, beam="parallel",
              backend="torch", precision="float64", voxel_spacing=(0.7, 0.5),
              volume_chunk_shape=(4, 5), view_chunk_size=3)
x = torch.ones(2, 3, 8, 10, dtype=torch.float64, requires_grad=True)
y = A.project(x)                           # (2, 3, 12, 16)
tangent = torch.ones_like(x)
value, jvp = torch.func.jvp(A.project, (x,), (tangent,))
batched = torch.vmap(A.project)(x)          # same leading dimensions
```

Data gradients, JVP, `vmap`, `jacrev`, `jacfwd` and second data derivatives use
the portable PyTorch traversal when a function transform is active.
Ordinary native calls retain Numba execution. Native batches dispatch their
samples separately. Geometry and surface derivatives are first-order.
Forward-mode derivatives (`torch.func.jvp`, `jacfwd`) with respect to trajectory
tensors raise an error. Other function transforms of geometry and
`torch.compile` are not promised.

Distributed data `grad`, JVP, `vmap` and Hessian-vector products require every
rank to call the same transforms in the same order, with matching leading batch
sizes. Inputs and tangent slices describe one coherent global tensor; scalar
losses follow the replica divisors below. Dense per-rank Jacobians have no
global ownership convention in this API; use a single-process projector for
`jacrev`/`jacfwd` matrices.

## Streaming analytical reconstruction

`fbp`/`fdk` accept `view_chunk_size`, `volume_chunk_shape`, `devices`,
`distributed`, `process_group`, `partition` and `partition_grid`. Global angles,
quadrature and the isocenter are resolved once, before processing view batches.
Fan/cone normalization uses each view's SID and SDD. Filtering needs complete
detector-U rows; the view limit bounds those resident FFT rows. Spatial gathers
use global voxel centers even for partial tiles and owned slabs.

`fbp_into`/`fdk_into` read and write `TensorStore`, `NpyStore` or compatible block
stores. They return the supplied output object. Shapes, writable state, aliases,
internal output overlap, dtype and geometry/options are checked before data I/O.
Output dtype must match the precision mode. These methods are numerical and
reject autograd inputs/outputs/weights. Tensor-returning APIs retain gradients
for data, explicit weights and filter responses. Analytical geometry remains
fixed and flat; FDK remains an approximate inverse.

Explicit chunks bound working tiles and view batches; autograd still needs its
complete returned arrays. Portable autograd does not record the ray traversal.
This also applies inside `torch.vmap` and `torch.func.jvp` when an input or
tangent requires gradients. It keeps the geometry snapshot, and the input only
when geometry gradients are needed. Backward applies the matched adjoint for
data gradients. For geometry gradients, it recomputes one ray batch and tile at
a time, so that memory does not grow with the number of batches. Each geometry
batch keeps a small autograd graph, so automatic geometry batches use a smaller
budget (see below) and large leading batches are split, as in the forward call.
Geometry gradients cost one extra traversal. Returned arrays and gradients
still require their own memory.

Projector's portable path sizes its automatically chunked intermediate arrays
from the compute device. Caller inputs, the returned array and allocations
inside surface callbacks or samplers are excluded.

- CUDA compute devices (NVIDIA CUDA or PyTorch ROCm, through `devices=[...]` or
  a CUDA input): one call uses at most a quarter of the memory that the PyTorch
  allocator can still use when the call starts, for its work arrays and
  distributed reduction buffers. A per-ray and per-voxel byte model sizes the
  work arrays. It counts the crossing tables of the volume tiles that a ray
  batch visits, the previous tile's tables, volume tile copies, ray setup and
  validation. Outputs accumulate in place, so the call makes no temporary copy
  of the full output. With `devices=[...]`, the smallest device sets the size.
  Distributed ranks use the smallest value of all ranks, because their
  collectives follow the same tile and batch order. Distributed backward
  through a backprojection sums one cotangent tile at a time. There is no
  floor: with very little memory the plan shrinks to one voxel and one ray per
  step. One work array stays below 2^31 elements.
- CPU inputs, outputs or `*_into` stores with CUDA compute devices add a host
  staging limit. It is the smaller of a quarter of the free host memory and the
  CUDA limit, but never less than 8 MiB, the fixed CPU ceiling. Where the
  platform does not report free host memory, the limit is 8 MiB.
- CPU and other devices keep a fixed 8 MiB ceiling and their previous plans.

Geometry backward recomputes each ray batch and tile with the CUDA limit
divided by two. One recomputed pair, including its autograd graph and
`torch.autograd.grad`, measured at most 1.3 times the work-array model of its
plan, so a pair stays within the same quarter. Geometry backward may keep one
extra buffer as large as the incoming cotangent, outside that quarter.
This buffer makes a strided cotangent contiguous or holds its distributed sum.
Known limit: distributed geometry backward with replicated outputs may hold up
to three cotangent-sized buffers during the cotangent reduction. These buffers
are outside the quarter-memory bound. The geometry gradients need their own
memory. Large leading batches are split into pieces that every rank chooses
alike, and each piece of a forward call is copied into the returned array, so
at most one piece is extra.

Explicit chunk hints take priority over these limits. For large mapped batches,
also use `torch.vmap(..., chunk_size=...)` to control the transform's workspace.

With `backend="auto"`, a `Projector` that falls back to the portable backend on
an NVIDIA CUDA machine emits one `UserWarning` at construction. The warning
names the reasons: per-axis `voxel_spacing` outside the native practical range, a `precision` other than
`"fast"`, `partition="hybrid"` or `beam="parallel3d"`. Pass `backend="torch"`
to select the portable backend explicitly and silence the warning.
Analytical tile order can reread and refilter the source for another volume
tile. The native path admits tiles against available CUDA memory. The separate
`helical_fdk` approximation and low-level CUDA helpers retain their original
execution contracts.

## Hybrid ownership and losses

`partition="hybrid", partition_grid=(view_ranks, space_ranks)` uses
`rank = view_rank * space_ranks + space_rank`; the grid product must equal the
process-group size. With `partition_grid=None`, the operator chooses a near-square
factorization, with at least as many view ranks as space ranks.
Each space rank owns a balanced first-axis slab; each view rank owns a balanced
view shard. Both retain global physical coordinates.
Forward ray contributions sum across space ranks; adjoint/analytical volume
tiles sum across view ranks. Empty and uneven shards participate normally.
Distributed PyTorch-backend projectors require the same `volume_chunk_shape`,
`view_chunk_size` and `detector_chunk_shape` arguments on every rank.
Neither reduction gathers a whole volume.

| Output/loss | views | space | hybrid |
|---|---|---|---|
| Projection loss replica divisor | 1 | `world_size` | `space_world_size` |
| Adjoint/analytical image loss replica divisor | `world_size` | 1 | `view_world_size` |

Use local **sums** divided by the global physical element count for a global
mean; local `.mean()` fails for uneven or empty ownership. All participating
ranks run backward. Geometry/surface gradients already sum globally; avoid a
second DDP reduction. `total_variation` returns a global scalar with the correct
derivative for its owned volume input. If that input is a replicated output of
another distributed operator, apply that operator's loss replica divisor.

## Numerical solver API

`cgls`, `sirt` and `tv_reconstruction` return
`SolverResult(volume, iterations, residual_history, converged, reason)`.
History contains the initial residual and one value per completed iteration.
Convergence means `||y - A x|| <= atol + rtol * ||y||`, including the initial
iterate. `callback(iteration, residual, volume)` runs after each completed step;
return `False` to stop. Inputs and optional `x0` are preserved. Solves detach
their numerical state from autograd.
Without a callback, SIRT and TV reuse the residual of the previous history
entry for the next update, which saves one projection per iteration. With a
callback, and after a resume, they recompute the residual because the callback
can change the volume or the operator. CGLS reads its two scalar norms of an
iteration together.

```python
from diffct import cgls, sirt, tv_reconstruction

result = cgls(A, y[0, 0], iterations=30, rtol=1e-5,
              checkpoint_directory="checkpoints")
print(result.iterations, result.residual_history[-1], result.reason)
# SIRT controls: relaxation=1.0, nonnegative=True.
# TV controls: weight=1.0, lr=0.1; Adam minimizes half the global mean squared error
# plus physical mean TV and clamps the volume after every step.
```

`total_variation` uses forward differences divided by each axis pitch, zero
differences at the global outer boundary, and `sqrt(sum(diff**2) + eps)` at
every voxel. `reduction="mean"` divides by the global voxel count; `"sum"`
retains the sum. Spatial/hybrid execution exchanges neighboring faces and
their backward contributions, including uneven and empty slabs.

`cgls_into(A, measurements, output, iterations, workspace=..., block_elements=...)`
keeps state in mapped files and returns `SolverResult.volume is output`.
The workspace must be new; initialization, vector operations, norms,
fingerprints and checkpoint copies operate on bounded blocks. Measurements
are checked for finite values in bounded blocks before creating the workspace
or writing the output; every rank agrees on admission before execution.

Checkpoint paths are `step-000002/rank-00000/metadata.json`, for example.
Resume from the corresponding rank directory using `resume=...`; `iterations`
is the total target, including saved steps. Complete CGLS/SIRT/Adam state and
residual history are restored. Fingerprints bind the acquisition, measurements,
precision, ownership and algorithm settings. Ranks validate consistency before
numerical execution, flush state before publishing completed metadata, and
reject mismatched/incomplete snapshots together. Every rank must agree on
whether `x0`, `resume` and `checkpoint_directory` are supplied; checkpoint and
resume paths may be rank-local. A different rank layout needs
a new solve rather than an implicit redistribution of the saved state.

Checkpoints fingerprint the actual sampled detector rays. Legacy surface
callbacks need complete detector grids; their possible per-view offset output
must fit the fingerprint block budget or the solve raises before execution.
Use `ParameterizedSurface` to sample and fingerprint larger surfaces in blocks.
