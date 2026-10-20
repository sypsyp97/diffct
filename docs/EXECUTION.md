# Precision, training and reconstruction

`Projector`, `fbp` and `fdk` expose `backend` and `precision` explicitly.
The native NVIDIA path remains the default when compatible. The PyTorch
backend uses the same cell-constant Siddon model and physical coordinates.

| Setting | Computation | Devices |
|---|---|---|
| `backend="auto", precision="fast"` | Native FP32 for scalar spacing and views/space partitions; otherwise PyTorch FP32 | NVIDIA CUDA, or the input's PyTorch device |
| `backend="numba"` | Native FP32 fast-math traversal, with FP64 ray clipping and entry setup | NVIDIA CUDA; scalar voxel spacing, `precision="fast"`, views/space |
| `backend="torch", precision="float32"` | PyTorch FP32 data and local traversal, with FP64 ray clipping and entry setup | CPU, NVIDIA CUDA, PyTorch ROCm |
| `precision="float64"` | PyTorch FP64 throughout the data operations, including analytical FFT/filter/gather | CPU, NVIDIA CUDA, PyTorch ROCm |

`backend="auto"` selects PyTorch for either strict precision mode, per-axis
spacing, hybrid ownership and `parallel3d` beams. An incompatible explicit Numba
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

## Batch dimensions and axis spacing

Any leading batch/channel dimensions precede the configured volume or ray
shape. Projection and adjoint preserve them, including empty dimensions.
`voxel_spacing=(dy, dx)` or `(dz, dy, dx)` follows tensor axis order; trajectory
coordinates remain `(x, y[, z])`. Spacings are fixed physical settings.

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
the differentiable PyTorch traversal when a function transform is active.
Ordinary native calls retain Numba execution. Native batches dispatch their
samples separately. Geometry and surface derivatives are first-order;
function transforms of geometry and `torch.compile` are not promised.

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
complete returned arrays and saved graph. Projector's portable path uses an
8 MiB ceiling for automatically sized intermediate arrays in ordinary tensor calls, excluding caller
data, returned arrays and allocations inside legacy surface callbacks. Explicit
chunk hints take priority. The portable reference path can be slower, so start
with explicit spatial/view/pixel limits for large scans. For large mapped
batches, also use `torch.vmap(..., chunk_size=...)` to control the transform's
workspace. Returned arrays and the complete saved autograd graph still require
their own memory.
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
