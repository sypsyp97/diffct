# Changelog

All notable changes to the ``diffct`` dev branch are documented in this file.
The dev branch is the arbitrary-trajectory evolution of the library and is
maintained in parallel with the ``main`` branch's circular-orbit lineage.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.3.0.dev0] - 2026-04-14

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
