# diffct reference

## Capabilities and limits

| Area | Supported here | Important limit |
|---|---|---|
| Acquisition | 2D parallel/fan and 3D cone beams; per-view geometry and parameterized detector surfaces | Unit direction axes; no arbitrary-trajectory `sf`, `sf_tr` or `sf_tt` backend |
| Autograd | Volume/sinogram gradients and second derivatives; first-order trajectory and surface gradients | Geometry second derivatives raise an error; surface samples are checked on every call |
| Execution | Native NVIDIA CUDA or portable PyTorch; explicit FP32/FP64, leading batch dimensions, per-axis spacing, devices and distributed ownership | Native fast mode remains FP32; strict/hybrid/per-axis execution uses PyTorch; see [precision and execution](EXECUTION.md) |
| Reconstruction | Library CGLS/SIRT/TV; streamed differentiable `fbp`/`fdk`, numerical store outputs, approximate `helical_fdk` and composable helpers | Analytical geometry is fixed and flat; helical/low-level helpers retain their original CUDA contract; `backproject()` is not an inverse |
| Memory | Automatic spatial/view/pixel blocks, bounded transfers, numerical tensor/file stores and rank-owned volume slabs | Tensor autograd still holds complete inputs/outputs/gradients; disk solvers must keep their vector operations blockwise |

Analytical helpers use flat detectors. Analytical fan/cone backprojection infers the physical isocenter from source
lines along detector normals. For non-circular or ambiguous geometry (including
a single view), supply `isocenter=(x, y)` or `(x, y, z)`. Analytical interpolation
requires at least two detector bins per axis. The finite discrete Ram-Lak filter
uses `pad_factor>=2` for linear convolution when unwindowed.

Analytical gathers have their own discrete transpose kernels, with sinogram
gradients and higher derivatives. Filters and explicit weights use PyTorch
autograd. Geometry gradients are rejected by analytical APIs; use `Projector`
for calibration. See the [analytical learning guide](source/analytical_learning.rst)
for custom stages, circular short scans and the slice-windowed helical approximation.

## Custom or calibrated trajectories

Use a generator for a standard scan, or pass your calibrated tensors directly.
The tuple describes every source position, detector centre and detector axis;
no circular-orbit fit is required. For example, two illustrative cone views:

```python
calibrated = tuple(torch.tensor(rows, dtype=torch.float32) for rows in (
    [[-320.0, 0.0, -20.0], [0.0, -300.0, 25.0]],  # source (x, y, z)
    [[192.0, 0.0, -20.0], [0.0, 212.0, 25.0]],    # detector centre (x, y, z)
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],         # unit detector u axes
    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],         # unit detector v axes
))
A_custom = Projector(calibrated, (64, 96, 128), (192, 128),
                     beam="cone", voxel_spacing=1.0, detector_spacing=(0.8, 1.0))
y_custom = A_custom.project(torch.ones(A_custom.volume_shape, device="cuda"))
# y_custom.shape == (2, 192, 128): (views, detector_u, detector_v)
```

Replace the two rows in each component with all views from your calibration,
in measurement order. Coordinates are world `(x, y, z)`; volume tensors are
`(z, y, x)` / `(D, H, W)`. Use one length unit for positions and spacings;
axes are unit directions, not pixel-sized vectors. See
[Geometry and units](#geometry-and-units) for 2D tuples and centring conventions.
The `custom_trajectory_3d` helper instead derives a detector pose from a source
path looking toward the origin; use explicit tuples when detector poses are
independently calibrated.

## Parameterized detector surfaces

Pass `detector_surface=surface` to `Projector` to determine each pixel's world
position from a local parameterized surface. The same Siddon kernels compute
`project()` and its matched `backproject()` using these positions.
Omitting this argument keeps the existing flat detector.

For example, a source-centred cylindrical detector curves along its horizontal axis:

```python
import torch
from diffct import Projector, spiral_trajectory_3d

trajectory = spiral_trajectory_3d(
    90, sid=80.0, sdd=128.0, z_range=10.0, n_turns=1.0, device="cpu")
radius = torch.tensor(128.0, requires_grad=True)

def cylinder(u, v):
    u, v = u.to(radius.device), v.to(radius.device)
    angle = u / radius
    return torch.stack((radius * angle.sin(), v,
                        radius * (angle.cos() - 1)), dim=-1)

A = Projector(trajectory, (32, 32, 32), (96, 64),
              detector_spacing=(0.8, 1.0), detector_surface=cylinder)
volume = torch.ones(A.volume_shape, device="cuda")
sinogram = A.project(volume)                 # (90, 96, 64)
sinogram.square().sum().backward()           # radius.grad
adjoint = A.backproject(sinogram.detach())
```

For a runnable circular cone example with native projection, matched
backprojection and gradient checks, run `python examples/curved_detector.py`
from the repository root.

For bounded surface geometry, make parameters explicit:

```python
from diffct import ParameterizedSurface

def sample(u, v, view_indices, r):
    # u,v are a GLOBAL physical pixel rectangle; view_indices are global IDs.
    return torch.stack((r * torch.sin(u / r), v,
                        r * (torch.cos(u / r) - 1)), -1)

surface = ParameterizedSurface(sample, parameters=(radius,))
A = Projector(trajectory, (32, 32, 32), (96, 64),
              detector_spacing=(0.8, 1.0), detector_surface=surface,
              view_chunk_size=8, detector_chunk_shape=(32, 32))
sinogram = A.project(torch.ones(A.volume_shape, device="cpu"))
sinogram.square().sum().backward()  # radius.grad, using bounded pixel VJPs
```

The sampler returns `(*pixel_shape,3)` or
`(len(view_indices),*pixel_shape,3)` local offsets. Grids, IDs and explicit
parameter snapshots are on CPU, preserving parameter dtypes; gradients return
to the original parameter device/dtype. All mutable geometry tensors must be
explicit parameters. Other captured configuration must be immutable and the
sampler must be pure. Per-view coupling uses global IDs or explicit parameters,
including parameters for views outside the executing batch. Each forward
captures the sampler and one set of frame/parameter snapshots; backward uses
those values after mutation or replacement. World points and pixel VJPs are
generated for the executing device's view and pixel batch; shared parameter
gradients are reduced before communication. Samplers can return CPU or CUDA offsets.
The local frame and 2D constraints below apply to both surface interfaces.
Allocations inside user samplers remain the caller's responsibility.

The legacy two-argument callback receives complete centred physical float64 grids on CPU:
`u[i] = (i + 0.5 - U / 2) * du` and
`v[j] = (j + 0.5 - V / 2) * dv`, with `ij` indexing.
It returns a finite floating-point PyTorch tensor of local `(u, v, n)` offsets,
with shape `(U, V, 3)` shared by all views, or `(views, U, V, 3)` for a separate
surface in each view. Here `views` is the total trajectory view count, including
in distributed mode; the operator handles view sharding. Both full-volume and
streamed calls accept CPU or CUDA offsets. Move the grids to your parameters'
device inside the callback when needed.
The world position is
`center + offset_u * det_u + offset_v * det_v + offset_n * cross(det_u, det_v)`.
The normal's sign follows the supplied frame. In the generated frame above it
points away from the source, so the example uses negative normal offsets and
an initial radius equal to the source-to-detector distance.
`torch.stack((u, v, torch.zeros_like(u)), -1)` gives a flat surface.

For fan and parallel beams the inputs have shape `(U,)`, `v` is zero, and the
return shape is `(U, 3)` or `(views, U, 3)`. The middle offset must be zero.
The local normal is `(det_u_y, -det_u_x)`. Fan rays connect each source to its
pixel; parallel rays pass through each pixel along the view's `ray_dir`.

Legacy callbacks receive the complete detector grids and are evaluated
at construction and afresh on each operation,
so captured PyTorch parameters can change and receive first-order gradients.
The operator transfers selected offsets to each compute device before expanding
world points. Each device expands only its owned views, or the current streamed
view and pixel batch. Native execution preserves the dtype promotion of offsets
and frames during composition, then converts complete coordinates to float32.
The PyTorch backend uses the selected precision. Memory planning includes the
composition dtype and local view counts; samplers with an unknown output dtype
use a conservative float64 allowance.
Offset snapshots stay on CPU, with their gradient connections intact. Validation
uses bounded batches and reports local failures to all distributed ranks before
numerical work. Ranks must agree on the actual callback offset shape and dtype
at construction and on each call.
Allocations made by the callback itself still depend on its implementation;
use `ParameterizedSurface` to sample offsets for selected global view IDs.
The existing trajectory rule still applies: if no component requires gradients,
the entire trajectory is cloned. Otherwise all components are read live, with
gradients for the differentiable components. Second derivatives through the
geometry are unsupported. Surfaces must return the documented shape and finite
coordinates on every call; a source must not coincide with any pixel.

Explicit positions cost two coordinates per 2D pixel or three per cone pixel,
in the composition dtype, plus gradients when needed. Sources and ray directions remain
per-view tensors. Views are sharded with the same multi-GPU and distributed
rules as flat detectors. Each pixel samples one ray; these kernels do not
integrate over a finite detector-pixel area. FBP/FDK and analytical weighting
helpers retain their flat-detector assumptions and do not accept the surface
callback. Native curved projection and matched backprojection need no
resampling. To reuse a flat-detector FDK pipeline, first resample the measured
line integrals onto a virtual flat detector along matching rays; this adds
interpolation error and requires ray coverage. That conversion is outside
the core operator.

## Chunked execution

Backprojection keeps each native GPU tile accumulator across all contributing
view/pixel batches, then downloads the completed tile. Projection and geometry
VJPs reuse contiguous native layouts. Automatic scheduling compares spatial,
view and bounded-window orders, slabs and blocks, using copy estimates and
short real pilots. `last_execution_stats` separates actual copy payloads from
pilot work and estimated candidate costs. Copy counters exclude distributed
layout metadata and planning control messages. Manual `schedule` choices support
reproducible comparisons. Pinned slots, host fragments and local geometry have
memory ceilings independent of GPU capacity.

When complete arrays exceed RAM, use `TensorStore`, `NpyStore` or a structural
block backend with `project_into`/`backproject_into`. These methods overwrite
caller-provided output blocks matching `Projector.dtype`, return the output object and do not build
autograd graphs. Known read-only and aliased outputs are rejected before
execution; hidden custom-store aliases are the caller's responsibility.
The disk CGLS example updates `x,s,p,r,q` blockwise and saves rank-local
completed-iteration checkpoints. It does not materialize complete residuals
or volumes. See [the full memory guide](https://sypsyp97.github.io/diffct/chunking.html).

`Projector` accepts floating CPU data by default. The native backend streams
through NVIDIA CUDA while full arrays and returned gradients stay on the input
device. The PyTorch backend also executes directly on CPU or supported
accelerators. See [precision and execution](EXECUTION.md) for backend selection.

For the native backend, `volume_chunk_shape=None` chooses an execution plan per call. CPU
inputs always stream. CUDA inputs use the original full-volume path when a
conservative additional working-set estimate fits each selected card and the
output device; otherwise they stream. Memory is sized against the smallest
participating budget, with 25% headroom for runtime allocations. Cone layout
copies, projection batches, geometry/VJP buffers and collective staging count
toward the estimate. Concurrent allocations can still cause an allocation
error; available memory is not a reservation.
For PyTorch's native allocator, streamed admission also checks segment capacity
separately from tensor payload bytes. Partially occupied cached segments are
not treated as wholly available for new buffers. Small and large allocation
pools have different segment sizes, so a fitting sum of tensor sizes alone
does not establish that a plan can be allocated.
The segment estimate assumes standard native allocator settings and fresh
fixed-buffer inventories; it does not guarantee allocation under arbitrary
fragmentation or allocator overrides.
The tensor CGLS example budgets state segments separately from subsequent
pipeline and norm segments. If these phases cannot fit under the same rules
used after state allocation, ranks agree to keep solver state on CPU before
allocating it.
The state bound includes compatible idle cached segments that state allocation
may reuse and pin on the caller's stream.
PyTorch versions without the public memory-fraction getter expose only
driver/allocator memory to this estimate; use explicit limits when setting a
process ceiling on those versions.

Manual overrides are `volume_chunk_shape=(H, W)` for 2D or `(D, H, W)` for cone
beams, `view_chunk_size=<positive integer>` and `detector_chunk_shape=(U,)`
or `(U,V)`. Spatial or detector limits force streaming;
a view limit alone also forces streamed batches. The automatic view batch is
capped at 32 and can shrink to fit its geometry/projection buffers. Oversized
spatial limits are clipped at the volume edge; nondivisible tails retain their
global voxel positions. Explicit limits are not automatically made smaller.

Streamed compute uses the caller's CUDA stream. Internal buffers share that
allocation owner across calls on the same stream; upload and download use
separate streams with explicit reuse dependencies. A newly allocated volume buffer may wait for
earlier caller work before upload, while existing ray/geometry slots retain
their bounded prefetch overlap.
Transfers hold source and destination references in the bounded completion-event
queue until they finish. Completed internal copies release these references
directly, avoiding stale transfer-stream reservations across calls.

Legacy detector-surface callbacks can return CPU or CUDA offsets.
Explicit `ParameterizedSurface` samples local view/pixel batches.
Both interfaces expand world points on the compute device after view sharding.
Geometry snapshots
keep data and first-order geometry backward consistent with the forward call,
including later parameter changes. Geometry second derivatives remain
unsupported. No cache or autograd state retains all streamed CUDA tiles.
For bounded GPU residency, keep data, trajectory tensors and learnable surface
parameters on CPU. CUDA geometry snapshots and CUDA intermediates created by a
callback consume additional GPU memory outside the tile buffers.

Each tile integrates a disjoint part of the original voxel grid. Forward
contributions add; adjoint blocks accumulate across view batches. The
cell-constant Siddon model and matched adjoint are preserved, with float32
rounding differences from a changed summation order. Very small tiles cost more
transfers and kernel launches. This API covers native projection and matched
backprojection; analytical FBP/FDK helpers retain their existing memory path.

See [the CPU-backed CGLS example](../examples/chunked_reconstruction.py), the
[chunking guide](source/chunking.rst) and [distributed rules](DISTRIBUTED.md).

## Geometry gradients

Trajectory tensors with `requires_grad=True` get gradients for the source,
detector centre and detector axes. The gradient is the exact derivative of the
cell-constant model. Set `requires_grad=True` before constructing `Projector`;
otherwise it snapshots the geometry. Rebuild it after changing fixed geometry.

```python
source, det_center, det_u, det_v = (t.clone() for t in trajectory)
source.requires_grad_()
A = Projector((source, det_center, det_u, det_v), (128, 128, 128), (384, 256), detector_spacing=0.8)
(A.project(volume) - sinogram).square().sum().backward()   # source.grad has shape (360, 3)
```

Second derivatives with respect to the geometry raise an error. Edge cases are
listed in [Geometry and units](#geometry-and-units).

## Several GPUs and nodes

Choose the execution mode by how you want to hold the projections:

| Mode | Configuration | Projection ownership |
|---|---|---|
| One GPU | Default `Projector(...)` | Full sinogram on the input device |
| One process, several GPUs | `devices=[0, 1, ...]` | Views computed on several GPUs, then full sinogram gathered on the input device |
| Distributed views | `distributed=True, partition="views"` after initialization | Rank-local sinogram with shape `A.projection_shape`, indexed by `A.view_slice`; replicated volume |
| Distributed space | `distributed=True, partition="space"` after initialization | Replicated ray batches; rank-owned volume with local shape `A.volume_shape` and global ownership `A.volume_slice` |

CPU-backed arrays use bounded spatial tiles and view batches on each GPU.
CUDA inputs/outputs occupy their complete size on the caller's device, and a
fitting full-volume path may replicate the volume across GPUs. GPU memory is
not pooled. Views mode keeps projections sharded, while backprojection sums
and replicates the volume on the input device. Space mode holds local slabs,
SUMs their forward ray contributions and returns local backprojection only.

**One process, several GPUs:**

```python
A = Projector(trajectory, (128, 128, 128), (384, 256), detector_spacing=0.8, devices=[0, 1, 2, 3])
```

**One process per GPU, one or more nodes:**

```bash
torchrun --nproc-per-node=4 examples/iterative_reconstruction.py --trajectory helical      # one node
sbatch examples/slurm/multi_node.sbatch examples/iterative_reconstruction.py --trajectory helical   # several nodes
```

With `distributed=True`, every rank must make the same `project`, `backproject`
and backward calls, even on ranks with zero views. Use each rank's local
projection loss with SUM semantics in views mode; the operator sums image and geometry
gradients across ranks. Divide a loss on replicated backprojection output by
`world_size`. In space mode, divide replicated projection losses by `world_size`
and sum owned backprojection losses. Do not add a DDP gradient reduction on
top of the operator.
Initialization, loss examples, Slurm details and the cross-node
check are in [docs/DISTRIBUTED.md](DISTRIBUTED.md).

## Geometry and units

| Beam | Trajectory tuple, one row per view | Volume | Sinogram |
|---|---|---|---|
| `parallel` | `(ray_dir, det_origin, det_u)`, each `(views, 2)` | `(H, W)` | `(views, U)` |
| `fan` | `(src_pos, det_center, det_u)`, each `(views, 2)` | `(H, W)` | `(views, U)` |
| `cone` | `(src_pos, det_center, det_u, det_v)`, each `(views, 3)` | `(D, H, W)` | `(views, U, V)` |

- Use the helpers in `diffct.geometry` (`circular_*`, `spiral_*`,
  `sinusoidal_*`, `saddle_*`, `random_*`, `custom_*`), or pass calibrated
  tensors.
- Geometry rows use world `(x, y)` or `(x, y, z)` coordinates. Volume axes
  run in `(y, x)` or `(z, y, x)` order. Leading batch/channel dimensions are preserved.
- Direction vectors have unit length. `ray_dir` is orthogonal to `det_u`, and
  `det_u` is orthogonal to `det_v`. Detector pitch is a separate argument:
  a scalar in 2D, a scalar or `(du, dv)` pair for cone beams.
  `detector_shape=(U, V)` and cone sinograms always follow `(views, U, V)`.
- The volume is centred on the origin. Voxel `i` of an axis with `N` voxels has
  its centre at `(i + 0.5 - N / 2) * axis_spacing`. Voxel spacing is a scalar
  or a tuple `(dy, dx)` / `(dz, dy, dx)` in tensor axis order.
- The default flat detector array is centred. Pixel `k` of `N_det` pixels lies at
  `(k - (N_det - 1) / 2) * pitch` from `det_center` along `det_u`
  (`det_origin` for parallel beams), the same convention as `main`. For cone
  beams, add the analogous offset along `det_v` using its own pitch.
- Projections are line integrals in the length unit of the geometry. For flat
  detectors, `Projector` rejects views where the source equals the detector
  centre, or the detector is edge-on to the source. Surface detectors check the
  actual pixel positions instead.
- The native kernels work in float32. In native fan and cone beams, the source or the
  detector centre must be within 1e6 voxels of the volume centre in each view;
  both must be within 1e15 voxels. For surfaces these bounds apply to every
  source/pixel pair. Geometry and floating-point volume/sinogram inputs may be
  on CPU or CUDA. Native CPU inputs use streamed CUDA computation.
  Ray positions are accurate to about 6e-8 times that nearer distance.

**Gradient edge cases:**

- On an exact voxel edge or corner, the kernels return the derivative of one
  adjacent side. Central finite differences average both sides, so they can
  differ. This occurs in symmetric setups, for example a principal ray parallel
  to a volume axis, a source on a voxel face plane, and equal detector pitches.
  For finite-difference checks, rotate the trajectory slightly, for example with
  `start_angle=0.1`.
- Trajectory frame checks run at construction; surface positions are also
  checked on every call. Optimize angles and offsets, not raw axis vectors, so
  the geometry stays valid.

