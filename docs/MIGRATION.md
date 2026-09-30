# Preparing the arbitrary-trajectory branch for main

This candidate branch is based on `dev` at `cb516cf`. It does not change the
remote default branch, create a release, or update PyPI.

The recommended entry point is `diffct.Projector`. Its geometry is a tuple of
per-view tensors, independent of whether the scan is circular, helical, tilted,
or calibrated. Existing `dev` geometry helpers and low-level Function classes
remain available.

## Moving from circular-only main

- Replace scalar `angles / sid / sdd` arguments with a trajectory helper or
  explicit source, detector-center, and detector-axis tensors. Parallel scans
  use ray direction, detector origin, and detector u direction.
- Configure `Projector(trajectory, volume_shape, detector_shape, ...)` once.
  Use `project(volume)` and `backproject(sinogram)` in the reconstruction loop.
- Volume axes are `(height, width)` or `(depth, height, width)`. Cone projections
  are `(views, detector_u, detector_v)`, matching `dev`'s existing kernels.
  Cone detector shapes and pitches follow `(u, v)` in that order.
- The `dev` detector coordinate convention is `(index - size / 2) * pitch`.
  Circular-only `main` used `(index - (size - 1) / 2) * pitch`. Check detector
  calibration when migrating; these conventions differ by half a detector bin.
- The `main` separable-footprint backends (`sf`, `sf_tr`, `sf_tt`) are not
  implemented for arbitrary trajectories in this candidate. The matched
  cell-constant Siddon projector is used. Code depending on SF needs a separate
  migration decision before a remote promotion.

`backproject()` computes the matched adjoint, not an inverse reconstruction.
Use it in an iterative solver, or use the existing analytical helpers when
their FBP/FDK assumptions fit the acquisition. A spiral trajectory does not
become an exact FDK acquisition merely because its rays can be projected.

## Verification and publication

Run `python -m pytest tests/ -q` on a CUDA host, then the distributed check in
`docs/DISTRIBUTED.md`. CPU-only orchestration checks do not establish GPU
correctness. Compare numeric results and gradients to the single-GPU operator;
do not treat a successfully launched job as a completed validation.

Keep the existing remote `main`, `dev`, and published artifacts unchanged while
validating this candidate branch. Remote promotion and release are separate actions.
