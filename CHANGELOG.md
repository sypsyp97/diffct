# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.2] - 2026-10-09

### Fixed

- The exit-time cleanup from 2.0.1 set numba-cuda's pending-deallocation
  ratio to infinity. numba-cuda converts ``capacity * ratio`` to ``int``, so
  each deallocation queued during shutdown raised ``ValueError`` or
  ``OverflowError``. The exit code stayed 0, but stderr showed the error. The
  exit hook now disables numba-cuda's pending-deallocation queues instead of
  changing the limits, so no queue is flushed at exit and no error is raised.
  The hook no longer imports ``numba.cuda.core.config``, which older
  numba-cuda releases do not have.

### Documentation

- The install instructions state that ``pip install diffct`` without a CUDA
  extra needs a system CUDA Toolkit. Use ``diffct[cu12]`` or ``diffct[cu13]``.
- README images and links use absolute URLs, so they also work on PyPI.

## [2.0.1] - 2026-10-09

### Fixed

- The low-level projector and backprojector Functions and the analytical
  weighted backprojections run in the CUDA context of their input. Before,
  inputs on another GPU than the current device could use the wrong context
  and stream.
- A process that ran diffct kernels on several GPUs no longer crashes at
  interpreter exit: numba-cuda's flush of pending CUDA module unloads during
  shutdown is skipped, and process exit releases the modules.
- ``custom_trajectory_3d`` and ``custom_trajectory_2d_fan`` raise
  ``ValueError`` for a source at the isocenter or a non-finite source,
  instead of returning NaN geometry. The figure-8 example in the
  ``custom_trajectory_3d`` docstring no longer puts a source at the origin.
- The class docstring examples of the six projector Functions use the
  per-view trajectory API.
- ``_trig_tables`` accepts NumPy dtypes for tensor input.
- Walnut example: ``preprocess_walnut.py`` raises when projection archives or
  views are missing or an air level is not positive, instead of writing NaN
  data. A non-default crop keeps the detector centre offset, and
  ``walnut_reconstruction.py`` applies it to the geometry and the FDK
  weights. The default crop has no offset.
- Examples: ``psnr`` returns infinity for identical images; distributed
  reductions in CGLS, TV and the noise peak use the operator's process
  group.

### Added

- ``Projector.process_group``: the process group of a distributed operator.

## [2.0.0] - 2026-10-09

Version 2.0.0 replaces the circular-orbit API of 1.x with arbitrary
per-view trajectories. It breaks the 1.x interface; see
[docs/MIGRATION.md](docs/MIGRATION.md).

### Breaking

- ``diffct.Projector`` is the main entry point. Geometry is a tuple of
  per-view tensors (source, detector centre, detector axes) instead of
  scalar ``angles``, ``sid`` and ``sdd``.
- The low-level ``*ProjectorFunction`` / ``*BackprojectorFunction`` classes
  and the ``*_weighted_backproject`` helpers take per-view geometry tensors.
- The separable-footprint backends (``sf``, ``sf_tr``, ``sf_tt``) of 1.3.x
  are removed. All projectors use the matched cell-constant Siddon model.

### Added

- Multi-GPU execution in one process (``devices=[...]``) and one process per
  GPU with torchrun and NCCL, on one or several nodes.
- Install extras ``diffct[cu12]`` and ``diffct[cu13]`` pull in the matching
  numba-cuda CUDA bindings, so one ``pip install`` gives a working import.
- Measured walnut example (``examples/walnut_reconstruction.py``), intro video
  and README figures.
- Geometry gradients. Trajectory tensors may require gradients; ``Projector``
  and the low-level Function classes return gradients for source, detector
  centre and detector axes (parallel: ray direction, detector origin and
  axis). CUDA kernels compute the exact derivative of the cell-constant
  Siddon model from the face crossings of each ray. Distributed mode sums
  the gradients over ranks.
- Second derivatives for volumes and sinograms: backward passes are autograd
  Functions themselves. Second derivatives with respect to the geometry
  raise ``RuntimeError``.

### Changed

- Detector arrays are centred: cell ``k`` lies at
  ``(k - (N - 1) / 2) * pitch`` from ``det_center`` (or ``det_origin``), as in
  ``main``. Earlier dev versions used ``(k - N / 2) * pitch``. The projector,
  the FBP/FDK gathers and ``detector_coordinates_1d`` all changed.
- Siddon projections are line integrals in physical length. Earlier
  versions returned voxel-length integrals, which differ by the factor
  ``voxel_spacing``. Backprojection scales the same way, so the pair stays
  matched. Results with ``voxel_spacing == 1`` do not change.
- FBP and FDK gather kernels sample at voxel centres,
  ``(index + 0.5 - N / 2) * voxel_spacing``, as the cell-constant Siddon
  projector does. Earlier reconstructions were half a voxel off.
- Siddon projector and backprojector kernels no longer skip ray pieces of
  length ``<= 1e-6``. They skip only pieces of length exactly 0 or less.
  Forward projection is now a continuous function of the geometry: each piece
  changes a projection value by at most cell value x ``1e-6``.

### Fixed

- Fan and cone rays integrate only from the source to the detector pixel.
  A source or detector inside the volume box gave the full-line integral.
- Fan and cone ray setup measures the ray parameter from the endpoint
  nearer the volume centre, and the traversal restarts at the volume entry
  point. A distant source gave relative errors up to 5e-3 (source 1e6
  voxels away) and could stop the traversal. Parallel-beam setup runs in
  float64 for a distant detector origin. Projection speed does not change.
- Ray setup treats a direction component as axis-parallel only below
  1e-20, so a distant source with a small lateral component is no longer
  rejected.
- Siddon kernels use a finite sentinel instead of infinity, because
  ``fastmath`` lets LLVM assume that infinity never occurs.
- ``Projector`` rejects fan and cone views whose source coincides with the
  detector centre or whose detector is edge-on to the source, and views where
  both lie farther than 1e6 voxels from the volume centre.
- The traversal treats a direction component as axis-parallel only below
  1e-20. With the earlier 1e-6 limit, a ray that grazed a grid line was
  assigned to the neighbouring row.
- ``tests/distributed_projector_check.py`` no longer issues a MAX
  all-reduce after a local failure while other ranks wait in a SUM
  all-reduce.

## [1.3.0.dev0] - 2026-04-14 (dev branch, not released on PyPI)

First sync of the dev (arbitrary-trajectory) branch against the main
branch's 1.2.10 / 1.2.11 analytical reconstruction overhaul. Brings
dev up to functional parity with main 1.2.11 except for the 1.3.0
separable-footprint (SF) projector backends, which rely on closed-form
circular-orbit geometry and are not yet generalised to arbitrary
trajectories.

### Added

#### Analytical reconstruction helpers (``diffct.analytical``)

A new module exposes the following helpers, all trajectory-agnostic so
they work with both the circular trajectories and the arbitrary
``(src_pos, det_center, det_u_vec[, det_v_vec])`` trajectory arrays
that dev-branch kernels already accept:

- ``detector_coordinates_1d`` — detector cell centre coordinates.
- ``angular_integration_weights`` — trapezoidal per-view weights for
  the analytical FBP/FDK sum. Optional ``redundant_full_scan`` flag
  absorbs the ``1/2`` redundancy factor for full scans.
- ``fan_cosine_weights`` / ``cone_cosine_weights`` — per-detector-cell
  ``cos(gamma)`` pre-weight for fan / cone FBP pipelines.
- ``parker_weights`` — Parker short-scan redundancy weights for
  circular fan geometries.
- ``ramp_filter_1d`` — generic 1D ramp filter with ``sample_spacing``,
  ``pad_factor``, ``window`` (``"ram-lak"``, ``"hann"``, ``"hamming"``,
  ``"cosine"``, ``"shepp-logan"``), and ``use_rfft`` options, rescaled
  by ``1 / sample_spacing`` so the output is in physical units.
- ``parallel_weighted_backproject`` / ``fan_weighted_backproject`` /
  ``cone_weighted_backproject`` — voxel-driven FBP / FDK backprojection
  wrappers that dispatch to new dedicated gather kernels and apply the
  analytical Fourier-convention constant (``1/(2*pi)`` for parallel,
  ``sdd_mean/(2*pi*sid_mean)`` for fan and cone). These are the
  **recommended path for analytical reconstruction** and replace the
  previous pattern of passing a filtered sinogram through
  ``*BackprojectorFunction.apply``.

#### Voxel-driven FBP / FDK gather kernels

Three new CUDA kernels, each under the dedicated ``fastmath=False``
``_FDK_ACCURACY_DECORATOR`` added to ``diffct.constants``:

- ``_parallel_2d_fbp_backproject_kernel`` — no distance weighting.
- ``_fan_2d_fbp_backproject_kernel`` — ``(|S|/U_n)^2`` weighted,
  where ``U_n`` is the signed distance from the per-view source to
  the voxel along the detector normal.
- ``_cone_3d_fdk_backproject_kernel`` — same pattern in 3D.

Every kernel is voxel-driven: one thread per output pixel/voxel, loops
over views inside, projects the voxel onto the detector using the
per-view ``(src_pos, det_center, det_u_vec[, det_v_vec])`` arrays,
bilinearly samples the filtered sinogram, applies the per-view weight
and accumulates. They are completely separate from the pure Siddon
adjoint kernels that back the autograd path; autograd is untouched.

#### Tests (``tests/``)

The dev branch had no test directory at all prior to this change.
``pytest.ini`` and ``tests/__init__.py`` are new, plus 58 tests across
11 files that mirror the main-branch test layout:

- ``tests/test_adjoint_inner_product.py`` — ``<A x, y> = <x, A^T y>``
  identity for parallel / fan / cone autograd pairs, plus an extra
  ``test_cone_autograd_backward_matches_backprojector_forward`` that
  protects the autograd ``ConeProjectorFunction.backward`` from
  drifting away from the standalone ``ConeBackprojectorFunction``.
- ``tests/test_gradcheck.py`` — ``torch.autograd.gradcheck`` for every
  projector Function with float32-calibrated tolerances.
- ``tests/test_weights.py`` — unit tests for ``detector_coordinates_1d``,
  ``angular_integration_weights``, ``fan_cosine_weights``,
  ``cone_cosine_weights`` and ``parker_weights``.
- ``tests/test_cuda_smoke.py`` — end-to-end smoke tests for every
  Projector / Backprojector Function pair + the analytical
  ``*_weighted_backproject`` wrappers.
- ``tests/test_cone_projector_autograd.py`` — gradient finiteness and
  sparsity guard for the cone projector Function (circular + spiral
  trajectories).
- ``tests/test_fbp_parallel_accuracy.py`` /
  ``tests/test_fbp_fan_accuracy.py`` /
  ``tests/test_fdk_cone_accuracy.py`` — quantitative RMSE and
  amplitude bounds for a full Shepp-Logan FBP / FDK pipeline per
  geometry. These would have tripped on the old dev reconstruction
  path, which silently produced wrong-amplitude volumes.
- ``tests/test_fbp_fan_offsets.py`` /
  ``tests/test_fdk_cone_offsets.py`` — adapted from main's offset
  tests: dev's arbitrary-trajectory kernels have no scalar
  ``detector_offset`` / ``center_offset_*`` parameters, so offsets
  are applied by shifting the trajectory arrays directly. Verifies
  that the FBP / FDK gather kernels handle non-centred trajectories.
- ``tests/test_ramp_filter_windows.py`` — 29 parametrised tests covering
  every ``_ramp_window`` option (DC gain, Nyquist value, non-negativity)
  and the full ``ramp_filter_1d`` end-to-end (shape, DC annihilation,
  rfft vs complex-fft parity, ``sample_spacing`` scaling, high-frequency
  pass-through).

#### Benchmark suite (``tests/benchmarks/``)

Opt-in ``pytest-benchmark`` suite covering every CUDA kernel in the
library (forward projector, pure-adjoint backprojector, and the full
analytical FBP / FDK pipeline) across three sizes for each of the
three geometries. 27 benchmarks total, excluded from the default
``pytest tests/`` run via ``--ignore=tests/benchmarks`` in
``pytest.ini``. Run explicitly with
``pytest tests/benchmarks/ --benchmark-only``.

#### Parker short-scan demos in examples

``examples/circular_trajectory/fbp_fan.py`` and
``examples/circular_trajectory/fdk_cone.py`` now expose an
``apply_parker`` switch. When enabled, the example switches the
trajectory to a minimal ``pi + 2*gamma_max`` short scan and applies
``parker_weights`` to the sinogram before the ramp filter; when
disabled, the pipeline runs a full ``2*pi`` scan with the ``1/2``
redundancy factor absorbed by ``angular_integration_weights``. Both
branches produce correctly amplitude-calibrated reconstructions.

#### Documentation (``docs/source/api.rst``)

New "Analytical Reconstruction Helpers", "Ramp Filter Options", and
"Analytical FBP / FDK architecture" sections describe the new
``diffct.analytical`` module, enumerate every ``ramp_filter_1d``
option, and explain the analytical scale factors used in each
``*_weighted_backproject`` wrapper.

### Fixed

#### FBP / FDK amplitude bugs in the ``circular_trajectory/*.py`` examples

Prior to this release the circular-trajectory FBP / FDK examples
(``examples/circular_trajectory/fbp_parallel.py``,
``examples/circular_trajectory/fbp_fan.py``,
``examples/circular_trajectory/fdk_cone.py``) produced amplitude-wrong
reconstructions because they:

- used ``*BackprojectorFunction.apply`` (the pure Siddon adjoint) as if
  it were an FBP gather, so they missed the ``(sid/U)^2`` / ``(|S|/U_n)^2``
  distance weight that the classical FBP / FDK formula requires;
- hand-rolled a ramp filter missing the ``1/sample_spacing`` scale;
- multiplied by a standalone ``pi/num_views`` normalisation that only
  absorbs the angular step, not the ``1/(2*pi)`` Fourier-convention
  constant.

All three examples now go through the new
``parallel_weighted_backproject`` / ``fan_weighted_backproject`` /
``cone_weighted_backproject`` wrappers and end-to-end produce raw MSE
matching the main branch's 1.2.11 release to within rounding:

| Example                                 | Raw MSE   | Range             |
|-----------------------------------------|-----------|-------------------|
| ``circular_trajectory/fbp_parallel.py`` | ~0.00366  | ``[-0.02, 1.00]`` |
| ``circular_trajectory/fbp_fan.py``      | ~0.00220  | ``[-0.10, 1.01]`` |
| ``circular_trajectory/fdk_cone.py``     | ~0.00333  | ``[-0.07, 1.00]`` |

### Notes on the sync from ``main``

The ``main`` branch's 1.2.10 / 1.2.11 / 1.3.0 releases introduced several
changes which this update brings to the dev branch, adapted where
necessary for the arbitrary-trajectory kernel API:

- ``1.2.10`` FDK / FBP voxel-driven gather, ``ramp_filter_1d`` options,
  analytical constants, and the ``(sid/U)^2`` correctness fix: **ported**.
  Because dev's kernels take per-view ``(src_pos, det_center, u_vec,
  v_vec)`` arrays instead of closed-form ``sin(beta)/cos(beta)`` math,
  the new gather kernels use ``U_n = (P - S) . n`` (where
  ``n = u_vec x v_vec``) as the generalisation of the classical ``U``.
  For a canonical circular orbit this reduces to the textbook
  ``sid + x*sin(beta) - y*cos(beta)``.
- ``1.2.10`` cone / fan autograd ``distance_weight=1.0`` bug fix:
  **not applicable**. The dev cone / fan backward kernels were
  already correct (they never had that parameter), so the autograd
  adjoint identity holds on dev and is now protected by
  ``tests/test_adjoint_inner_product.py``.
- ``1.2.11`` gradcheck / ramp window tests / CHANGELOG: **ported**.
- ``1.2.11`` benchmark suite: **not ported yet** (dev's benchmarks
  directory stays empty).
- ``1.3.0`` SF-TR / SF-TT projector backends: **not ported**. SF relies
  on closed-form ``sin(beta)/cos(beta)`` math for the trapezoidal
  transaxial / rectangular axial footprints. Generalising SF to
  arbitrary per-view ``(S, C, u_vec, v_vec)`` trajectories is a
  non-trivial research effort (the "separable" part is not separable
  when the detector is not plane-aligned to the voxel axes) and will
  be tackled in a dedicated follow-up.

[Unreleased]: https://github.com/sypsyp97/diffct/tree/dev

## [1.3.4] - 2026-04-23

### Changed
- **Siddon projectors switched to cell-constant integration.** The 2D
  parallel, 2D fan, and 3D cone Siddon forward/backward kernels in
  `diffct/differentiable.py` now integrate each ray by accumulating
  `d_image[iy, ix] * seg_len` on the traversed pixel/voxel instead of
  sampling the bi-/tri-linear interpolant at the segment midpoint.
  This simplifies the inner loop, removes the interpolation weight
  bookkeeping, and keeps the forward / adjoint pair matched by
  construction (still verified by
  `tests/test_adjoint_inner_product.py`). Kernel docstrings and the
  `"siddon"` backend blurbs in `examples/iterative_reco_*.py` have
  been updated to say *cell-constant Siddon*.
- **Iterative examples clamp the reconstruction in-place** instead of
  wrapping the forward pass in `nn.ReLU`. `examples/iterative_reco_parallel.py`,
  `iterative_reco_fan.py`, and `iterative_reco_cone.py` now call
  `self.model.reco.clamp_(min=0.0)` under `torch.no_grad()` after each
  optimizer step, so gradients flow through the unclamped volume while
  the stored state stays non-negative.
- **`examples/iterative_reco_fan.py` default backend** is now
  `"siddon"` (was `"sf"`), matching the parallel and cone iterative
  examples.

### Fixed
- **FDK cone accuracy test tolerance.** `tests/test_fdk_cone_accuracy.py`
  raises the maximum central z-profile error bound from `0.6` to `0.65`
  to reflect the slightly sharper one-sample transition error at
  ellipsoid boundaries under cell-constant forward projection; global
  FDK RMSE is unchanged.

## [1.3.3] - 2026-04-23

### Fixed
- **Kernel numerical consistency and speed.** The CUDA kernels in
  `diffct/differentiable.py` are now consistently `float32` end to end —
  literal constants (`0.0`, `0.5`, `1.0`, `-inf`, ...) and integer-to-float
  casts inside the Siddon and SF kernels are routed through typed `_DTYPE`
  helpers so numba no longer implicitly promotes intermediates to `float64`.
  This removes a silent precision mismatch that was slowing down every
  projector/backprojector kernel; the library runs substantially faster
  with no change in public-API behavior.
- **Thread indexing and grid configuration** in the 3D cone Siddon
  forward / backward kernels and the 2D fan SF backward kernel. The
  `cuda.grid(...)` unpack order and the matching `_grid_2d` / `_grid_3d`
  launch arguments are aligned so the warp-adjacent axis matches the
  stride-1 axis of the output buffer: `(iv, iu, iview)` for 3D cone
  (`d_sino[view, u, v]` / `d_vol[ix, iy, iz]`) and `(ix, iy)` for 2D fan
  SF. Fixes mismatched launches between `ConeProjectorFunction`,
  `ConeBackprojectorFunction`, `FanProjectorFunction`, and
  `FanBackprojectorFunction` and their kernels.
- **Walnut rotation-axis correction** in the shipped preprocessing and
  sample data. `examples/data/preprocess_walnut.py` now applies the
  Zenodo record's recommended 5-raw-pixel left shift before binning and
  cropping, records the correction in the `.npz` metadata, and the
  shipped `examples/data/walnut_cone.npz` / `walnut_reco.png` are
  regenerated accordingly. `examples/realdata_walnut_fdk.py` and
  `examples/data/NOTICE` are updated to surface the correction and the
  richer walnut metadata.

### Changed
- **Examples now require CUDA explicitly.** `examples/fbp_parallel.py`,
  `fbp_fan.py`, `fdk_cone.py`, `iterative_reco_*.py`, `realdata_fbp_*.py`,
  `realdata_fdk_cone.py`, and `realdata_walnut_fdk.py` assert
  `torch.cuda.is_available()` and initialize the CUDA device up front
  instead of silently falling back to CPU (which the numba-cuda kernels
  never supported anyway).

## [1.3.2] - 2026-04-23

### Added
- **Expanded angular-weight regression tests** in
  `tests/test_weights.py`, covering periodic `pi` half scans,
  downsampled full scans, endpoint-included 721-view full scans, and
  open short scans.

### Changed
- **`angular_integration_weights` is now coverage-aware and order-stable.**
  The helper sorts angles internally, detects `pi`- and `2*pi`-periodic
  sampling more robustly, handles endpoint-included trajectories, and
  maps the computed weights back to the caller's original angle order.
- **Walnut tooling is now configurable from the command line.**
  `examples/data/preprocess_walnut.py` gained CLI switches for binning,
  cropping, view stride, angle handedness, storage dtype, and archive
  discovery; `examples/realdata_walnut_fdk.py` gained CLI controls for
  backend, output volume size, voxel scale, filter window, output file
  writing, and display suppression.
- **README polish for bilingual navigation and repo layout.**
  `README.md` now exposes the Chinese README link near the top and
  clarifies the file tree entries for `README.md` / `README.zh.md`.

### Fixed
- **Angular quadrature artifacts** in analytical reconstructions when
  full scans were endpoint-included or angularly downsampled. These
  trajectories now receive periodic integration weights instead of being
  misclassified as open scans.
- **Walnut example angle metadata and scan handedness.**
  The walnut preprocessing and FDK example now use the dataset's
  scanner-to-diffct angle convention explicitly and preserve the full
  721-view `0 .. 360 deg` acquisition description in the stored
  metadata.

## [1.3.1] - 2026-04-15

### Added
- **Real-data walnut FDK example** at `examples/realdata_walnut_fdk.py`,
  running the full analytical FDK pipeline on a preprocessed subset of
  the Helsinki walnut cone-beam CT dataset ([Meaney 2022, Zenodo
  10.5281/zenodo.6986012](https://doi.org/10.5281/zenodo.6986012),
  CC-BY 4.0). The 241-view, 256×256-per-view sinogram is shipped at
  `examples/data/walnut_cone.npz` (~25 MB, float16, 8× binned, 256²
  center crop) together with a `preprocess_walnut.py` regenerator
  script, an attribution `NOTICE`, and a `walnut_reco.png` sample
  montage. Reconstructs at half the nominal voxel size on a 512³ grid
  using `backend="sf_tr"` + Hamming ramp window.
- **Synthetic "real-data" pipeline examples** for all three geometries
  (`examples/realdata_fbp_parallel.py`, `realdata_fbp_fan.py`,
  `realdata_fdk_cone.py`) that run a Beer-Lambert + Poisson + `-log`
  forward model on Shepp-Logan and reconstruct with the analytical
  pipeline, showing the proper place for `-log` preprocessing.
- **Chinese README** at `README.zh.md`, mirroring `README.md` with
  bidirectional language links in both files.
- **Core Algorithm section** in the READMEs (EN + ZH) explaining the
  Siddon + bilinear/trilinear interpolation design choice, the
  trade-off (slight blur at nominal voxel size), and when to pick
  SF vs VD honestly (see "Changed" below).

### Changed
- **SF FBP / FDK backprojection kernels rewritten in LEAP's
  chord-weighted matched-adjoint form** (from `projectors_SF.cu`).
  The three analytical SF gather kernels
  (`_fan_2d_sf_fbp_backproject_kernel`,
  `_cone_3d_sf_tr_fdk_backproject_kernel`,
  `_cone_3d_sf_tt_fdk_backproject_kernel`) now use the per-voxel
  chord through the unit voxel instead of a `(sid/U)²`-plus-footprint-
  area form, with a matching `sqrt(1+(v/sdd)²)` axial chord correction
  on the cone side. Amplitude is calibrated against Siddon VD at
  every voxel / detector-pitch combination tested (nominal, 0.64×
  sub-nominal, 0.40× sub-nominal, supra-nominal) via the new
  `det_spacing/(voxel·2π)` fan scale and
  `(du·dv·sid)/(2π·sdd·voxel²)` cone scale constants.
- **Honest README Core Algorithm section on SF vs VD.** The earlier
  "SF lowers MSE by ~17 %" and "SF is measurably sharper at sub-
  nominal" claims in 1.3.0 are walked back: on Shepp-Logan and the
  walnut dataset, LEAP-form SF and Siddon VD produce visually
  indistinguishable edge profiles at typical CBCT magnifications,
  and MSE differences are fractions of a percent. SF's real value
  is its **mass-conserving cell-integrated forward model**, which
  matters for iterative reconstruction, learned priors, and
  sinogram-level losses — not as a drop-in sharpness knob on
  analytical FBP / FDK. The shipped `fbp_fan.py`, `fdk_cone.py`,
  and `realdata_walnut_fdk.py` example comments are updated to
  match.

### Fixed
- **SF FBP / FDK weight normalization** when the voxel's trapezoidal
  footprint straddles the default 4-corner sort. The 1.3.0 SF
  backprojection kernels used `fdk_w / (span · v_span)` (bounding-box
  area) which under-amplified oblique views by up to 2× per axis and
  produced a dim, angularly-modulated reconstruction. Diagnosed by
  inspecting a Shepp-Logan SF reconstruction, corrected first to
  `fdk_w / ((span+plateau)/2 · v_span)` (true trapezoid area) and
  then fully replaced by the LEAP chord-weighted form above.

### Acknowledgements
- [LEAP](https://github.com/LLNL/LEAP) (LLNL / Hyojin Kim et al.)
  added to `README.md` / `README.zh.md` Acknowledgements.
  `projectors_SF.cu` is the reference implementation the new
  analytical SF backprojection kernels are ported from.

## [1.3.0] - 2026-04-14

### Added
- **Separable-footprint (SF) projector backends** for 2D fan and 3D cone
  beam geometries, following Long-Fessler-Balter (IEEE TMI 2010).
  `FanProjectorFunction` / `FanBackprojectorFunction` now accept
  `backend="siddon"` (default, unchanged) or `backend="sf"`.
  `ConeProjectorFunction` / `ConeBackprojectorFunction` accept
  `backend="siddon"` (default), `"sf_tr"` (trapezoid in transaxial,
  rectangle in axial) or `"sf_tt"` (trapezoid in both directions with
  axial rise/fall from `U_near`/`U_far` across the voxel). Each SF
  backend is implemented as a matched scatter/gather pair of CUDA
  kernels so both forward and adjoint are available — autograd works
  end-to-end for every backend, and the standalone Backprojector
  Function can also be driven in SF mode for iterative reconstruction.
- **Adjoint inner-product tests** for every new SF backend in
  `tests/test_adjoint_inner_product.py`, plus `torch.autograd.gradcheck`
  coverage in `tests/test_gradcheck.py`. Test count grew from 57 to 66.

### Changed
- **Internal grid-point convention** is now uniformly `x_v = ix - cx`
  (matching `_cone_3d_fdk_backproject_kernel`). The SF kernels use the
  same convention so voxel centers align with where the Siddon/FDK
  paths interpret `d_vol[ix, iy, iz]`.
- **`_TPB_SF_3D = (4, 4, 4)`** introduced for the 3D SF kernels. SF-TT
  in particular exceeds the 128-register-per-thread budget allowed by
  the default `(8, 8, 8)` launch; the smaller block gives each thread
  ~1024 registers and avoids `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`.

### Notes
- **Raw forward accuracy vs analytical Radon**: on both 2D fan
  (512² Shepp-Logan) and 3D cone (96³ and 128³ 3D Shepp-Logan) phantoms,
  measured against the analytical ellipse / ellipsoid Radon transform,
  the existing Siddon projector with bilinear / trilinear interpolation
  is slightly more accurate (~3% better RMSE, ~15% better max error)
  than SF-TR / SF-TT at the thin-ray sampling level. The classic SF
  advantage reported in Long et al. is over the piecewise-constant DD
  projector; diffct's Siddon uses smooth interpolation, which covers
  most of the gap SF was designed to close.
- **Full analytical reconstruction (forward + FBP/FDK)**: despite the
  raw-forward result above, in a complete analytical pipeline
  (`FanProjectorFunction` / `ConeProjectorFunction` -> `ramp_filter_1d`
  -> `fan_weighted_backproject` / `cone_weighted_backproject`) SF
  **measurably improves reconstruction quality**. On the 256²
  Shepp-Logan `fbp_fan.py` example the raw MSE drops from
  `0.00216` (Siddon) to `0.00178` (SF-TR), a ~17% improvement.
  On the 128³ `fdk_cone.py` example it drops from `0.00325` (Siddon)
  to `0.00271` (SF-TR), again ~17%. The mechanism is that SF's
  mass-conserving cell-integrated forward pairs more naturally with
  the voxel-driven FBP/FDK gather backprojector than Siddon's
  thin-ray sampling, which feeds extra high-frequency ringing into
  the ramp filter.
- **Performance**: SF-TR is ~2× and SF-TT ~2.5–4× slower than Siddon for
  cone beam, and ~3× slower than Siddon for fan beam. The extra cost
  comes from voxel-driven scatter with atomic adds (forward) versus
  ray-driven parallelism in Siddon.
- **Recommendation**: `backend="siddon"` remains the default for
  backward compatibility and raw speed. For **analytical FBP/FDK
  reconstruction** where reconstruction MSE matters more than forward
  runtime, `"sf"` / `"sf_tr"` is a quantifiably better choice — at
  2–3× forward cost you get ~17% lower reconstruction MSE on standard
  phantoms. For **iterative reconstruction with autograd**, either
  backend is valid; SF has the theoretical advantage of exact voxel-
  scale mass conservation on both forward and adjoint, but in practice
  Siddon's smoothing usually helps optimizer convergence. SF-TT's
  axial-trapezoid refinement over SF-TR improves RMSE only by ~0.001
  and is not worth the extra cost for typical cone angles; use SF-TR
  by default and reserve SF-TT for extreme cone-angle research.

## [1.2.11] - 2026-04-14

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
  Jacobian over small inputs with float32-calibrated tolerances. Strongest
  autograd-correctness guard beyond the existing inner-product adjoint tests.
- **`tests/test_ramp_filter_windows.py`** — 27 parametrised tests covering every
  `ramp_filter_1d` window option (`None`/`"ram-lak"`, `"hann"`, `"hanning"`,
  `"hamming"`, `"cosine"`, `"shepp-logan"`) at three layers: direct `_ramp_window`
  helper (DC gain, Nyquist value, monotonicity), full `ramp_filter_1d` end-to-end
  (shape, DC annihilation, pad correctness, rfft vs complex-fft parity,
  `sample_spacing` scaling), and a high-frequency attenuation sanity check on a
  step input.
- **`CHANGELOG.md`** — new Keep-a-Changelog file at the repo root.
- **`tests/benchmarks/`** — opt-in `pytest-benchmark` suite covering every CUDA
  kernel in the library (forward projector, pure-adjoint backprojector, and
  analytical FBP/FDK pipeline) across three sizes for each of the three
  geometries. 27 benchmarks total. Excluded from the default `pytest tests/`
  run via `--ignore=tests/benchmarks` in `pytest.ini` so the normal test suite
  stays fast; run explicitly with `pytest tests/benchmarks/ --benchmark-only`.
  Enables before/after perf comparisons via `--benchmark-save=` /
  `--benchmark-compare=`.

### Changed
- **`_cone_3d_fdk_backproject_kernel`** memory-access layout. The kernel now
  unpacks `cuda.grid(3)` as `(iz, iy, ix)` instead of `(ix, iy, iz)`, and
  `cone_weighted_backproject` launches the grid as `_grid_3d(D, H, W)` instead
  of `_grid_3d(W, H, D)`. This makes `iz` warp-adjacent, which matches the
  innermost stride-1 axis of the WHD output buffer, so the final `d_vol` write
  becomes coalesced. Measured speedup on a 64^3–160^3 sweep: 1.0x–1.14x,
  within run-to-run noise (the kernel is read-bound on the sinogram bilinear
  samples, not write-bound, so L2 absorbed the old uncoalesced writes). The
  fix is kept as the right default and future-proofs the kernel against
  edits that shift work toward the write path. Matching coalescing comments
  were added to the 2D `_parallel_2d_fbp_backproject_kernel` and
  `_fan_2d_fbp_backproject_kernel` kernels, which were already coalesced
  (their first grid index is also warp-adjacent and matches the `Nx` innermost
  axis of their `(Ny, Nx)` output buffers).

### Notes
- No public-API changes. Pure-adjoint kernels and autograd Function classes
  are untouched, so the autograd and iterative-reconstruction paths are
  byte-for-byte equivalent to 1.2.10.

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

[1.3.3]: https://github.com/sypsyp97/diffct/releases/tag/v1.3.3
[1.3.2]: https://github.com/sypsyp97/diffct/releases/tag/v1.3.2
[1.3.1]: https://github.com/sypsyp97/diffct/releases/tag/v1.3.1
[1.3.0]: https://github.com/sypsyp97/diffct/releases/tag/v1.3.0
[1.2.11]: https://github.com/sypsyp97/diffct/releases/tag/v1.2.11
[1.2.10]: https://github.com/sypsyp97/diffct/releases/tag/v1.2.10
[1.2.9]: https://github.com/sypsyp97/diffct/releases/tag/v1.2.9
[1.2.8]: https://github.com/sypsyp97/diffct/releases/tag/v1.2.8
[1.2.7]: https://github.com/sypsyp97/diffct/releases/tag/v1.2.7
