# diffct reference

## Capabilities and limits

| Area | Supported here | Important limit |
|---|---|---|
| Acquisition | 2D parallel/fan and 3D cone beams; per-view source/detector geometry | Flat detectors with unit direction axes; no arbitrary-trajectory `sf`, `sf_tr` or `sf_tt` backend |
| Autograd | Volume/sinogram gradients and second derivatives; first-order geometry gradients | Geometry second derivatives raise an error; validity checks run at construction |
| Execution | CUDA, one-process multi-GPU, or distributed ranks across nodes | Kernels and outputs are float32; CPU geometry is allowed, CPU projection is not |
| Reconstruction | Matched adjoint plus separate FBP/FDK helpers | `backproject()` is not an inverse; FDK remains approximate and does not become exact for arbitrary scans |
| Memory | Views split across GPUs/ranks | Every participating GPU needs a full volume; local multi-GPU also gathers the full sinogram on the input device |

Analytical fan/cone backprojection infers the physical isocenter from source
lines along detector normals. For non-circular or ambiguous geometry (including
a single view), supply `isocenter=(x, y)` or `(x, y, z)`. Analytical interpolation
requires at least two detector bins per axis. The finite discrete Ram-Lak filter
uses `pad_factor>=2` for linear convolution when unwindowed.

## Validation summary

- **Validated.** FDK matches ASTRA 2.5.0 `FDK_CUDA` on the same projections:
  PSNR within 0.3 dB, and a maximum pixel difference below 1% of the phantom
  maximum. Geometry gradients match an independent float64 reference to about
  1e-6. The [recorded A100 validation](VALIDATION.md) reports 229 passing
  pytest tests; these are measured results, not guarantees for every scan.

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
- The detector array is centred. Pixel `k` of `N_det` pixels lies at
  `(k - (N_det - 1) / 2) * pitch` from `det_center` along `det_u`
  (`det_origin` for parallel beams), the same convention as `main`. For cone
  beams, add the analogous offset along `det_v` using its own pitch.
- Projections are line integrals in the length unit of the geometry. `Projector`
  rejects views where the source equals the detector centre, or the detector is
  edge-on to the source.
- The kernels work in float32. In fan and cone beams, the source or the
  detector centre must be within 1e6 voxels of the volume centre in each view;
  both must be within 1e15 voxels. Geometry may be on CPU or CUDA, but volumes
  and sinograms passed to the operator must be floating-point CUDA tensors.
  Ray positions are accurate to about 6e-8 times that nearer distance.

**Gradient edge cases:**

- On an exact voxel edge or corner, the kernels return the derivative of one
  adjacent side. Central finite differences average both sides, so they can
  differ. This occurs in symmetric setups, for example a principal ray parallel
  to a volume axis, a source on a voxel face plane, and equal detector pitches.
  For finite-difference checks, rotate the trajectory slightly, for example with
  `start_angle=0.1`.
- Geometry checks run only at construction. Optimize angles and offsets, not raw
  axis vectors, so the geometry stays valid.

## Example scan and calibration results

Launch modes, distributed-loss rules and all measured results are in [examples/README.md](../examples/README.md). Shared helpers are in `examples/_common.py`. The example scan uses n³ unit
voxels, a source 2.5 n from the isocentre, a detector 4 n from the source
(magnification 1.6), (3n, 2n) detector cells, pitch 0.8 and 360 views.

Geometry calibration (64³, 360 views, per-view angle error 0.54° RMS, detector
shift (1.5, -1.0) voxels, Adam 300 steps) recovers the angle error to
0.0001-0.0025° RMS and the detector shift to below 0.001 voxel in every launch
mode.
