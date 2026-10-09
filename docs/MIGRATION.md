# Migrating from diffct 1.x to 2.0

diffct 2.0 replaces the circular-orbit API of 1.x with per-view trajectories.
This page lists what changes for code written against 1.x.

The recommended entry point is `diffct.Projector`. Its geometry is a tuple of
per-view tensors, independent of whether the scan is circular, helical, tilted,
or calibrated. Existing `dev` geometry helpers and low-level Function classes
remain available.

## API and shape reference

Select the beam explicitly for 2D data; the default is `beam="cone"`.
A trajectory describes one geometry per view and is not restricted to a named
orbit. Use a helper or pass your calibrated tensors directly.

| Beam | Trajectory tuple | Shape of each component | Volume shape | Detector shape | Sinogram shape |
| --- | --- | --- | --- | --- | --- |
| `parallel` | `(ray_dir, det_origin, det_u)` | `(views, 2)` | `(H, W)` | `U` or `(U,)` | `(views, U)` |
| `fan` | `(src_pos, det_center, det_u)` | `(views, 2)` | `(H, W)` | `U` or `(U,)` | `(views, U)` |
| `cone` | `(src_pos, det_center, det_u, det_v)` | `(views, 3)` | `(D, H, W)` | `(U, V)` | `(views, U, V)` |

Geometry coordinates are `(x, y)` or `(x, y, z)`, while volume indices are
`(y, x)` or `(z, y, x)`. Cone sinograms use `(views, U, V)`, not the common
image-display convention `(views, V, U)`; transpose imported measurements if
necessary. In distributed mode, only the leading view dimension is sharded.

Trajectory tensors may be on CPU or CUDA. Projection and backprojection inputs
must be floating-point CUDA tensors of exactly the documented shape, without
batch or channel dimensions. Computation and outputs are float32 even if the
input is float64; autograd returns input gradients in the input's dtype.
`voxel_spacing` is one positive scalar (isotropic voxels). `detector_spacing`
is one positive scalar for 2D, and a scalar or `(u_pitch, v_pitch)` for cone.

## Minimal migration example

This is a complete single-GPU fan-beam example using the current API:

```python
import torch
from diffct import Projector, circular_trajectory_2d_fan

trajectory = circular_trajectory_2d_fan(
    180, sid=160.0, sdd=256.0, device="cpu"
)
operator = Projector(
    trajectory,
    volume_shape=(64, 64),
    detector_shape=192,
    beam="fan",
    detector_spacing=0.8,
    voxel_spacing=1.0,
)
image = torch.ones((64, 64), device="cuda", requires_grad=True)
sinogram = operator.project(image)                 # (180, 192)
adjoint = operator.backproject(sinogram.detach())  # (64, 64)
sinogram.square().mean().backward()                # image.grad is populated
```

See [`examples/quickstart.py`](https://github.com/sypsyp97/diffct/blob/main/examples/quickstart.py) for parallel, fan and
helical cone examples, including adjoint and geometry-gradient checks. Use
[the distributed guide](https://github.com/sypsyp97/diffct/blob/main/docs/DISTRIBUTED.md) before adapting a loss to multiple ranks;
its normalization differs from a local per-rank mean.

## Moving from circular-only main

- Replace scalar `angles / sid / sdd` arguments with a trajectory helper or
  explicit source, detector-center, and detector-axis tensors. Parallel scans
  use ray direction, detector origin, and detector u direction.
- Configure `Projector(trajectory, volume_shape, detector_shape, ...)` once.
  Use `project(volume)` and `backproject(sinogram)` in the reconstruction loop.
- Volume axes are `(height, width)` or `(depth, height, width)`. Cone projections
  are `(views, detector_u, detector_v)`, matching `dev`'s existing kernels.
  Cone detector shapes and pitches follow `(u, v)` in that order.
- Detector cell `index` lies at `(index - (size - 1) / 2) * pitch` from the
  detector centre, the same convention as circular-only `main`. Earlier `dev`
  versions used `(index - size / 2) * pitch`; geometry calibrated for them is
  half a detector bin off.
- Voxel `index` along an axis of size `N` has its centre at
  `(index + 0.5 - N / 2) * voxel_spacing`; the volume is centred on the
  origin. The projector, the backprojector and the FBP/FDK helpers use this
  convention.
- Projections are line integrals in the length unit of the geometry. Scale
  measured `-log(I / I0)` data and the attenuation image in the same unit.
- The `main` separable-footprint backends (`sf`, `sf_tr`, `sf_tt`) are not
  available in 2.0. All projectors use the matched cell-constant Siddon model.
  Code that depends on SF can stay on diffct 1.3.4.

`backproject()` computes the matched adjoint, not an inverse reconstruction.
Use it in an iterative solver, or use the existing analytical helpers when
their FBP/FDK assumptions fit the acquisition. A spiral trajectory does not
become an exact FDK acquisition merely because its rays can be projected.

## Fixed versus learnable geometry

For fixed geometry, construct `Projector` once and reuse it. If none of the
trajectory tensors requires gradients at construction, the operator stores a
clone: editing your original tensors later does not update that operator.

If any trajectory tensor has `requires_grad=True` at construction, the operator
keeps references to the supplied tensors and reads their current values on each
call. When a trajectory is computed from learnable parameters such as angles or
shifts, rebuild that trajectory and the `Projector` inside each optimization
step to create the new autograd graph. See
[`examples/geometry_calibration.py`](https://github.com/sypsyp97/diffct/blob/main/examples/geometry_calibration.py).

Keep every geometry valid as it changes: components must have the same nonzero
view count and finite values; direction axes must be unit vectors; parallel
`ray_dir`/`det_u` and cone `det_u`/`det_v` must be orthogonal. Fan/cone sources
must differ from their detector centers, and detectors must not be edge-on to
the source. These checks run at construction, not after each parameter update.

Geometry derivatives are first-order only and are piecewise derivatives of the
cell-constant Siddon model. At voxel-boundary crossings they need not agree with
a centered finite difference. Second derivatives with respect to geometry raise
an error; volume and sinogram second derivatives are supported.
