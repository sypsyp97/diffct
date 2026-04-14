# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Parker short-scan demos** in `examples/fdk_cone.py` and `examples/fbp_fan.py`.
  Both examples now expose an `apply_parker` switch that not only multiplies
  by `parker_weights` but also samples angles on the minimal short-scan range
  `pi + 2*gamma_max`, where `gamma_max` is computed from the detector half-width
  and `sdd`. (Before this change the `apply_parker` flag in `fbp_fan.py` was
  silently a no-op because the angular range stayed at `2*pi`.)
- **`tests/test_gradcheck.py`** — `torch.autograd.gradcheck` coverage for
  `ParallelProjectorFunction`, `FanProjectorFunction`, and `ConeProjectorFunction`.
  Compares the analytical backward Jacobian to a finite-difference numerical
  Jacobian over small inputs with float32-calibrated tolerances. This is the
  strongest possible autograd-correctness guard beyond the existing inner-product
  adjoint tests.
- **`tests/test_ramp_filter_windows.py`** — 27 parametrised tests covering every
  `ramp_filter_1d` window option (`None`/`"ram-lak"`, `"hann"`, `"hanning"`,
  `"hamming"`, `"cosine"`, `"shepp-logan"`) at three layers: direct `_ramp_window`
  helper (DC gain, Nyquist value, monotonicity), full `ramp_filter_1d` end-to-end
  (shape, DC annihilation, pad correctness, rfft vs complex-fft parity,
  `sample_spacing` scaling), and a sanity check that apodisation really does
  reduce peak amplitude on a step input.
- **`CHANGELOG.md`** — this file.

## [1.2.10] - 2026-04-14

This release is a correctness overhaul of every analytical reconstruction path
(parallel / fan / cone) **and** of the autograd adjoint path shared by the
iterative examples. It also ships a unified architecture where each geometry
has its own dedicated voxel-driven FBP/FDK gather kernel, separate from the
pure Siddon adjoint used by autograd.

### Fixed

#### Analytical FBP / FDK amplitude bugs

Prior to this release, all three analytical reconstruction examples produced
results that were off by large constant factors:

| Example | Pre-fix reco range | Post-fix reco range | MSE improvement |
|---|---|---|---|
| `fbp_parallel.py` | [0, 6.52] | [−0.02, 1.01] | ~580× |
| `fbp_fan.py` | [0, 6.33] | [−0.08, 1.01] | ~900× |
| `fdk_cone.py` | [0, 10.08] | [−0.08, 1.00] | ~600× |

Root causes:
- Cone and fan used `(sdd/U)^2` with `U = sdd + x·sin − y·cos` instead of
  the correct FDK weight `(sid/U)^2` with `U = sid + x·sin − y·cos`.
- All three geometries were missing the `1/(2π)` Fourier-convention
  constant from the FBP/FDK reconstruction formula.
- Cone and fan used a Siddon ray-driven scatter for the analytical path,
  which is neither a true adjoint nor a classical FBP/FDK gather.

#### Autograd adjoint bug (cone + fan)

`ConeProjectorFunction.backward`, `ConeBackprojectorFunction.forward`,
`FanProjectorFunction.backward` and `FanBackprojectorFunction.forward`
were all passing `distance_weight=1.0` to the shared Siddon backward
kernel. This meant the autograd backward was **not the true adjoint** `P^T`
of the forward projector — it had a `(sdd/U)^2` per-voxel factor baked in,
biasing the gradient by ~2–3× depending on voxel position. Any iterative
reconstruction that relied on autograd (including `iterative_reco_cone.py`
and `iterative_reco_fan.py`) was running on a biased gradient flow.

Parallel beam was unaffected (its backward kernel had no `distance_weight`
parameter to begin with).

This release removes the `distance_weight` parameter from both
`_cone_3d_backward_kernel` and `_fan_2d_backward_kernel` entirely, so the
dead code path cannot be accidentally reintroduced, and fixes the four
autograd call sites to use the pure adjoint.

### Added

#### Voxel-driven gather kernels for every geometry

Three new CUDA kernels, all under a dedicated `fastmath=False` decorator
for FDK-grade accuracy:

- `_parallel_2d_fbp_backproject_kernel` — no distance weighting (no source).
- `_fan_2d_fbp_backproject_kernel` — `(sid/U)^2` + linear detector interp.
- `_cone_3d_fdk_backproject_kernel` — `(sid/U)^2` + bilinear detector interp.

Each kernel is voxel-driven: one thread per output pixel/voxel, loops over
views, computes the projected detector coordinate, interpolates the filtered
sinogram, weights and accumulates.

#### `parallel_weighted_backproject` (new public helper)

Mirrors `fan_weighted_backproject` and `cone_weighted_backproject`. Applies
the `1/(2π)` Fourier-convention constant so a unit-density disk reconstructs
to amplitude 1. Exported from `diffct`.

#### `ramp_filter_1d` — new backward-compatible kwargs

- `sample_spacing` (default `1.0`): physical detector cell pitch. Output is
  rescaled by `1/sample_spacing` for physical-unit correctness.
- `pad_factor` (default `1`): zero-pad to `pad_factor * N` before the FFT to
  suppress circular-convolution wrap-around. `2` is recommended for FBP/FDK.
- `window` (default `None`): frequency-domain apodization. Options: `None` /
  `"ram-lak"`, `"hann"`, `"hamming"`, `"cosine"`, `"shepp-logan"`.
- `use_rfft` (default `True`): faster real-FFT path for real-valued inputs.

Existing `ramp_filter_1d(sino, dim=1)` calls keep working unchanged.

#### Analytical FBP / FDK scale factors

Each analytical helper now applies the correct analytical constant
automatically so reconstructions are already amplitude-calibrated:

- `parallel_weighted_backproject`: `1 / (2π)`.
- `fan_weighted_backproject`: `sdd / (2π · sid)`.
- `cone_weighted_backproject`: `sdd / (2π · sid)`.

### Tests

Test suite expanded from 6 to **27 tests**:

- `tests/test_adjoint_inner_product.py` — the definitive check. For random `x`
  and `y`, asserts `⟨A x, y⟩ = ⟨x, A^T y⟩` for parallel, fan and cone autograd
  pairs. Permanently guards against any regression that would reintroduce the
  `distance_weight=1.0` bug.
- `tests/test_fdk_cone_accuracy.py` — 128³ Shepp-Logan RMSE / amplitude bounds.
- `tests/test_fdk_cone_offsets.py` — detector, center and combined offsets with
  amplitude assertions.
- `tests/test_cone_projector_autograd.py` — gradient finiteness and non-zero
  sanity.
- `tests/test_fbp_fan_accuracy.py` — 256×256 Shepp-Logan RMSE.
- `tests/test_fbp_fan_offsets.py` — three offset configurations.
- `tests/test_fbp_parallel_accuracy.py` — parallel FBP RMSE + offset.

### Examples

`examples/fdk_cone.py`, `examples/fbp_fan.py` and `examples/fbp_parallel.py`
have been rewritten with a consistent 8-step layout and detailed inline
comments documenting every geometry variable (what it is, units, typical
values, available options). Each example prints raw MSE, clamped MSE, and the
reconstruction / phantom data ranges.

The iterative reconstruction examples (`iterative_reco_parallel.py`,
`iterative_reco_fan.py`, `iterative_reco_cone.py`) are unchanged — they
silently benefit from the fixed autograd adjoint.

### Documentation

- `docs/source/api.rst`: adds autofunction entries for
  `parallel_weighted_backproject`, documents the new `ramp_filter_1d`
  options, and adds an "Analytical FBP / FDK architecture" section
  explaining the scale-factor derivation.
- `docs/source/fdk_cone_example.rst`: formula updated to reflect the
  voxel-gather kernel and `(sid/U)^2` weight.
- `docs/source/fbp_fan_example.rst` and `fbp_parallel_example.rst`: completely
  rewritten to reference the new analytical helpers and fix a sign-convention
  inconsistency in the math section.

### Compatibility

This release is source-compatible with 1.2.9 for every public entry point. The
`distance_weight` parameter was internal and is removed from the private
backward kernels; no public API touched that argument.

## [1.2.9] - 2026-02-14

Restored `main` to the state of 1.2.9 by reverting an accidental merge.
See [GitHub release v1.2.9](https://github.com/sypsyp97/diffct/releases/tag/v1.2.9).

## [1.2.8] - 2026-02-14

See [GitHub release v1.2.8](https://github.com/sypsyp97/diffct/releases/tag/v1.2.8)
and the commit history for details.

## [1.2.7] - 2025-08-05

See [GitHub release v1.2.7](https://github.com/sypsyp97/diffct/releases/tag/v1.2.7).

## Earlier versions

Releases 1.2.0 through 1.2.6 and the 1.1.x / 1.0.x lines are tracked on
[GitHub releases](https://github.com/sypsyp97/diffct/releases). This CHANGELOG
was introduced in 1.2.10 and does not back-fill detailed notes for earlier
versions beyond pointers to the GitHub release pages.

[Unreleased]: https://github.com/sypsyp97/diffct/compare/v1.2.10...HEAD
[1.2.10]: https://github.com/sypsyp97/diffct/releases/tag/v1.2.10
[1.2.9]: https://github.com/sypsyp97/diffct/releases/tag/v1.2.9
[1.2.8]: https://github.com/sypsyp97/diffct/releases/tag/v1.2.8
[1.2.7]: https://github.com/sypsyp97/diffct/releases/tag/v1.2.7
