# diffct reference

## Capabilities and limits

| Area | Supported here | Important limit |
|---|---|---|
| Acquisition | 2D parallel/fan and 3D cone beams; per-view geometry and parameterized detector surfaces | Unit direction axes; no arbitrary-trajectory `sf`, `sf_tr` or `sf_tt` backend |
| Autograd | Volume/sinogram gradients and second derivatives; first-order trajectory and surface gradients | Geometry second derivatives raise an error; surface samples are checked on every call |
| Execution | CUDA, one-process multi-GPU, or distributed ranks across nodes | Kernels and outputs are float32; CPU geometry is allowed, CPU projection is not |
| Reconstruction | Matched adjoint plus separate FBP/FDK helpers | `backproject()` is not an inverse; FDK remains approximate and does not become exact for arbitrary scans |
| Memory | Views split across GPUs/ranks | Every participating GPU needs a full volume; local multi-GPU also gathers the full sinogram on the input device |

Analytical helpers use flat detectors. Analytical fan/cone backprojection infers the physical isocenter from source
lines along detector normals. For non-circular or ambiguous geometry (including
a single view), supply `isocenter=(x, y)` or `(x, y, z)`. Analytical interpolation
requires at least two detector bins per axis. The finite discrete Ram-Lak filter
uses `pad_factor>=2` for linear convolution when unwindowed.

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

The callback receives centred physical float64 parameter grids on CPU:
`u[i] = (i + 0.5 - U / 2) * du` and
`v[j] = (j + 0.5 - V / 2) * dv`, with `ij` indexing.
It returns a finite floating-point PyTorch tensor of local `(u, v, n)` offsets,
with shape `(U, V, 3)` shared by all views, or `(views, U, V, 3)` for a separate
surface in each view. Here `views` is the total trajectory view count, including
in distributed mode; the operator handles view sharding. It may return offsets
on CPU or CUDA; move the grids to your parameters' device inside the callback
when needed. The world position is
`center + offset_u * det_u + offset_v * det_v + offset_n * cross(det_u, det_v)`.
The normal's sign follows the supplied frame. In the generated frame above it
points away from the source, so the example uses negative normal offsets and
an initial radius equal to the source-to-detector distance.
`torch.stack((u, v, torch.zeros_like(u)), -1)` gives a flat surface.

For fan and parallel beams the inputs have shape `(U,)`, `v` is zero, and the
return shape is `(U, 3)` or `(views, U, 3)`. The middle offset must be zero.
The local normal is `(det_u_y, -det_u_x)`. Fan rays connect each source to its
pixel; parallel rays pass through each pixel along the view's `ray_dir`.

Surface callbacks are evaluated afresh on each call, so captured PyTorch
parameters can change and receive first-order gradients. The existing rule for
trajectory tensors still applies: fixed tensors are cloned, while tensors that
require gradients are read live. Second derivatives through the geometry are
unsupported. Surfaces must return the documented shape and finite coordinates
on every call; a source must not coincide with any pixel.

Explicit positions cost two float32 coordinates per 2D pixel or three per
cone pixel, plus gradients when needed. Sources and ray directions remain
per-view tensors. Views are sharded with the same multi-GPU and distributed
rules as flat detectors. Each pixel samples one ray; these kernels do not
integrate over a finite detector-pixel area. FBP/FDK and analytical weighting
helpers retain their flat-detector assumptions and do not accept the surface
callback. Native curved projection and matched backprojection need no
resampling. To reuse a flat-detector FDK pipeline, first resample the measured
line integrals onto a virtual flat detector along matching rays; this adds
interpolation error and requires ray coverage. That conversion is outside
the core operator.

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
| One GPU | Default `Projector(...)` | Full sinogram on the input CUDA device |
| One process, several GPUs | `devices=[0, 1, ...]` | Views computed on several GPUs, then full sinogram gathered on the input CUDA device |
| One GPU per process, one or more nodes | `distributed=True` after process-group initialization | Rank-local sinogram with shape `A.projection_shape`, indexed by `A.view_slice` |

All modes require the full volume on each participating GPU. Local multi-GPU
execution also needs room for the full input/output sinogram on the caller's
device, plus temporary copies; it does not pool memory. Distributed mode keeps
projections sharded, while backprojection sums and replicates the full volume.

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
projection loss with SUM semantics; the operator sums image and geometry
gradients across ranks. Divide a loss on replicated backprojection output by
`world_size`. Do not add a DDP gradient reduction on top of the operator.
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
  run in `(y, x)` or `(z, y, x)` order. There are no batch/channel dimensions.
- Direction vectors have unit length. `ray_dir` is orthogonal to `det_u`, and
  `det_u` is orthogonal to `det_v`. Detector pitch is a separate argument:
  a scalar in 2D, a scalar or `(du, dv)` pair for cone beams.
  `detector_shape=(U, V)` and cone sinograms always follow `(views, U, V)`.
- The volume is centred on the origin. Voxel `i` of an axis with `N` voxels has
  its centre at `(i + 0.5 - N / 2) * voxel_spacing`. Voxel spacing is one
  isotropic value.
- The default flat detector array is centred. Pixel `k` of `N_det` pixels lies at
  `(k - (N_det - 1) / 2) * pitch` from `det_center` along `det_u`
  (`det_origin` for parallel beams), the same convention as `main`. For cone
  beams, add the analogous offset along `det_v` using its own pitch.
- Projections are line integrals in the length unit of the geometry. For flat
  detectors, `Projector` rejects views where the source equals the detector
  centre, or the detector is edge-on to the source. Surface detectors check the
  actual pixel positions instead.
- The kernels work in float32. In fan and cone beams, the source or the
  detector centre must be within 1e6 voxels of the volume centre in each view;
  both must be within 1e15 voxels. For surfaces these bounds apply to every
  source/pixel pair. Geometry may be on CPU or CUDA, but volumes
  and sinograms passed to the operator must be floating-point CUDA tensors.
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

