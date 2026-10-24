# CPU verification checks

Run `python -m pytest formal -q` from the repository root.
The workflow runs on a GitHub CPU runner; it does not require a GPU.

The labels describe the evidence each check supplies:

- **PROOF**: Z3 finds no counterexample over the stated real-number domain, or SymPy proves an exact identity.
- **BOUNDED**: the check enumerates a specified finite domain. Floating-point comparisons use stated tolerances.
- **PROPERTY TEST**: Hypothesis samples inputs. A passing run is not an exhaustive result.

Proofs apply to small mathematical models, not compiled Numba CUDA kernels.
Source guards check selected expressions to expose model drift; they do not prove that the whole implementation matches a model.
The slab model treats only exactly zero directions as parallel. It excludes `_TINY` approximations and finite `_BIG` sentinels.
The partition proof assumes ordered, complete crossings. The quadrature proof assumes at least two samples and a nonnegative closure gap.
Coordinate models use positive scalar or tensor-order per-axis pitches in 2D/3D.
The ray metric model excludes zero-length rays and uses unit directions in voxel coordinates.
It proves a squared-length identity; the physical length uses the nonnegative square root.
Bounded metric checks execute the Python helpers and selected CUDA-overload bodies on the CPU.
They do not compile CUDA code.

| Property | Method | Code covered | Bug class caught | Exclusions |
| --- | --- | --- | --- | --- |
| Slab clipping keeps a nonempty segment inside the box and its endpoint interval | PROOF: Z3 reals | Fan/cone forward and backward ray setup in `diffct/kernels/fan_beam.py` and `cone_beam.py` | Integrating behind an internal source or beyond an internal detector | Rounding, near-zero direction approximation, CUDA compilation |
| Ordered piece lengths are nonnegative and telescope to the physical chord length; forward and adjoint use the same factor | PROOF: induction/algebra | Siddon `seg_len = t_next - t` accumulation with one positive physical-length factor per ray | Missing/double-counting a piece or using different forward/adjoint factors in the model | Assumes complete crossings; does not prove traversal or termination |
| Voxel and detector coordinate maps have exact inverses and centred reflection; flat detector composition agrees with world-point scaling | PROOF: SymPy, scalar/per-axis pitches in 2D/3D | Voxel-centre gather coordinates; Siddon detector offsets; `detector_coordinates_1d` | Half-cell shifts and wrong inverse offsets | Floating-point index conversion and interpolation bounds |
| Normalized voxel-space rays recover physical squared length; uniform per-axis pitches reduce to scalar pitch | PROOF: SymPy, scalar/per-axis pitches in 2D/3D | Mathematical metric used by `spacing.py:_axis_pitch/_ray_scale` | Wrong physical-length units or metric normalization in the model | Nonzero rays, positive pitches, exact arithmetic; excludes casts and CUDA compilation |
| Trapezoidal quadrature totals its open span or periodic interval | PROOF: algebra | `angular_integration_weights` | Incorrect endpoint/closure contributions in the model | Scan classification and float32 summation |
| Python helpers and selected CUDA-overload bodies preserve axis pitches and physical partition lengths | BOUNDED: 2D/3D, scalar/uniform/anisotropic pitches, nonzero displacements in `{-1,0,1}` per axis and four-piece partitions | `_is_per_axis`, `_axis_pitch`, `_ray_scale`, their overload selectors and selected implementations in `spacing.py` | Wrong dispatch, tensor-to-world axis order, metric specialization or scalar return | CPU execution only; excludes CUDA compilation and scalar bitwise compatibility |
| Selected model expressions remain present in the source; selected metric mutations fail the guards | BOUNDED: AST guards and 34 single-function in-memory mutations, each with an unmodified-source control | Scalar/per-axis metric helpers and CUDA overloads; all three beam pairs, coordinate transforms and analytical gather branches in `test_proofs.py` | A proof silently outliving a changed formula, pitch order, normalization, length factor or CUDA target | Not semantic equivalence of the complete function or evidence of CUDA compilation |
| Ideal Parker taper pairs sum to one with the source's plus-gamma convention | PROOF: SymPy | `parker_weights` taper arguments | Wrong complementary taper argument | Excludes the runtime `1e-12` denominators, scan classification and arbitrary-trajectory ray pairing |
| Actual angular weights have the expected total and stay nonnegative; permutations preserve unique-angle weights and duplicate-angle aggregate mass | BOUNDED / PROPERTY TEST | CPU `angular_integration_weights` and scan classification | Half/full-scan normalization and reordered-view errors | Individual weights at duplicate angles, malformed angles and GPU arithmetic |
| Actual Parker taper pairs sum to one within tolerance; full scans pass through; 240/330/350-degree weighted mass is pi | BOUNDED | CPU `parker_weights`, shared scan classification | Applying both Parker taper and full-scan redundancy | Does not prove general redundant-ray coverage |
| Ramp windows are even and have unit DC; filtering agrees across FFT paths and physical scaling | BOUNDED / PROPERTY TEST | CPU `_ramp_window` and `ramp_filter_1d` | Asymmetric windows, FFT-path drift, missing pitch factor | Reconstruction accuracy or CUDA gather kernels |
| Cosine weights and trajectory frames satisfy their geometric invariants; custom parallel paths normalize directions and reject one wrong shape | PROPERTY TEST / BOUNDED | CPU cosine helpers and `diffct/geometry.py` generators | Off-centre detector weights, nonunit or nonorthogonal frames, incorrect shape acceptance | All custom paths, geometry VJP kernels, GPU wrappers |
| Every tiny-grid image/sinogram basis gives matching forward/adjoint matrices and independent cell intersections | BOUNDED: CUDASIM, scalar/uniform-tuple/anisotropic pitches | Actual parallel/fan/cone forward/backward kernels | Wrong traversal, axis ordering, physical scale, unmatched adjoint | All real-valued inputs, hardware atomics and autograd dispatch |
| Uniform boxes give analytic chords for internal endpoints, distant endpoints, misses and grazing rays | BOUNDED: CUDASIM | Same production Siddon kernels | The source-inside, distant-source and grazing bugs listed in `CHANGELOG.md` 2.0.0 | Larger grids, all trajectories, CUDA fastmath |
| Local frame maps recover 2D offsets and have the stated 3D cross normal; arcs have unit-speed circular sections and a flat limit | PROOF: SymPy | Mathematical surface coordinate contract in `docs/REFERENCE.md` | Wrong normal convention, offset order or arc parameter units | Complete source equivalence and floating-point sampling |
| World-point and arc-radius derivatives obey the chain rule, including a moving detector frame | PROOF: SymPy | Mathematical chain rule used by surface autograd | Missing normal/frame contributions in the model | PyTorch graph dispatch and derivatives at voxel edges |
| Every tiny-grid basis gives the independent curved forward/adjoint matrix; explicit flat points reproduce the default branch | BOUNDED: CUDASIM | Native pixel-position branches of the parallel/fan/cone Siddon kernels | Ignored endpoints, wrong ray/pixel indices, unmatched adjoints | Callback sampling, hardware arithmetic and all trajectories |
| Every pixel-coordinate VJP agrees with stable independent finite differences | BOUNDED: CUDASIM | Native pixel-position branches of geometry VJP kernels | Wrong endpoint-gradient units or routing | Source/ray-direction VJPs, frame/parameter chain rule, compiled CUDA and autograd |
| Positive integer chunk limits give each cell one half-open owner; tile-centred coordinates recover the global cell and detector ray | PROOF: Z3 integers / SymPy | Mathematical partition and translation models used by streamed execution | Overlapping/missing cells and half-cell tile-origin errors in the model | Source equivalence, scheduler and floating-point translations |
| Disjoint block forward maps have the assembled adjoint, including replicated rank SUM and an empty rank | PROOF: finite symbolic model | Four cells, three rays, two blocks and two nonempty/one empty view shards | Missing block embeddings or duplicated replicated losses in the model | Arbitrary distributed programs and real NCCL |
| Cartesian tiles partition every cell and recover exact rational cell centres | BOUNDED: exhaustive rational arithmetic, scalar/per-axis pitches | Model extents 1..3, chunk limits 1..4, 2D/3D grids | Tail clipping and axis-pitch errors in the model | Production slice generator equivalence and all integer sizes |
| Every tiny-grid block basis gives the independent full forward/adjoint matrix; boundaries and source-inside rays agree | BOUNDED: CUDASIM | Production Siddon kernels on shifted flat/curved blocks and batches of at most two views | Artificial tile boundary ownership, wrong translation/axis order and unmatched block adjoint | PyTorch scheduling, compiled CUDA, allocation bounds and larger domains |
| Block entry/exit VJP terms sum to independent endpoint/source derivatives | BOUNDED: CUDASIM | Native pixel-position geometry VJP kernels over disjoint blocks | Missing cancellation of artificial boundaries | Frame/callback chain rule, all real-valued geometry and hardware arithmetic |
| Tagged host/GPU slots preserve completed values across every modeled event interleaving; removing each lifetime dependency exposes a counterexample | BOUNDED: finite state exploration | Idealized upload/compute/download/reuse protocol in `test_execution_protocol.py` | Reading unfinished copies, premature host overwrite or GPU buffer reuse in the model | Production event wiring, actual overlap, allocator ownership and hardware |
| Independent host ceilings and all retained GPU accumulators satisfy their admitted capacity | PROOF: Z3 integers | Explicit slot, scratch, geometry, snapshot and accumulator assumptions | Omitting a resident buffer from the mathematical memory budget | Actual allocator rounding, arbitrary sampler allocations and implementation estimates |
| Resident local/rank sums followed by a completed-tile download preserve every contribution, including an empty rank | PROOF: finite symbolic model | Two tiles, two contributing ranks and one empty rank in `test_execution_protocol.py` | Missing contributions or downloading intermediate partials in the model | Compiled kernels, collective ordering and real NCCL |
| Balanced slabs cover the global volume without overlap, and preserve global cell coordinates | PROOF: Z3 integers / SymPy; BOUNDED: exhaustive intervals | Idealized first-axis ownership, including more ranks than cells | Missing cells, overlapping owners and recentered local slabs in the model | Production metadata agreement and GPU scheduling |
| Spatial and view partitions have the intended block adjoints and norm ownership | PROOF: finite symbolic model | Three cells/rays, two contributing ranks and one empty rank | Duplicate replicated norms or cotangents, missing spatial contributions | Floating-point reductions and autograd participation |
| Every rank follows common collective rounds despite differing local work and trainability | BOUNDED: finite protocol model | Zero/uneven tile counts, differing GPU counts and all three-rank trainability combinations | Collectives conditional on local tiles or differentiable inputs in the model | Actual process-group order and error recovery |
| CGLS keeps q through global qq, alpha and the residual update | BOUNDED: exact rational state exploration | Scalar instance A=3/2, y=4/3; early reuse has a wrong-result counterexample | Premature q overwrite or reuse in the model | General convergence, floating-point arithmetic and source equivalence |
| CGLS capacity includes x/s/p, r/q and transient dot/pipeline scratch | PROOF: Z3 integers | Independent live regions across projection, norm and update phases; omitting each region has a capacity counterexample | Admitting only a raw volume or omitting solver state/scratch | Allocator rounding, implementation estimates and arbitrary callback memory |
| Union reservation bounds nonoverlapping surface composition/replay despite pipeline pins; phase maxima do not | BOUNDED: exhaustive reachable allocation/free orders for 512 three-request inventories at eight pool/rounding boundaries and 64 six-request inventories at two small-pool sizes; concrete 24 vs 22 MiB counterexample; four source mutation guards | `test_surface_phase_reservation.py`; `_surface_work_phases`, payload/largest-allocation inventory, union inventory and streamed admission in `_execution_plan`, `_candidate_plans`, `_plan_buffers` | Reusing freed phase bytes in a reservation estimate when other allocations pin their segments | Assumes empty cache, one owner, each request allocated/freed once, composition fully freed before replay, pipeline events anywhere, standard native best-fit pools with splitting/coalescing and birth-order ties; excludes preexisting holes, repeated epochs, events, concurrent owners, allocator overrides, sampler internals, actual CUDA and full source equivalence |

## Simulator domain

`test_cudasim.py` launches the production kernels directly on NumPy float32 arrays.
It bypasses the PyTorch CUDA-tensor bridge. The independent reference intersects each ray with each cell using float64 slab arithmetic.
It does not reproduce the Siddon traversal.

The matrix check enumerates parallel, fan and cone beams, two rectangular grids, and every image and sinogram basis vector.
Each beam/grid pair uses scalar pitches 1.0 and 1.7, a uniform tuple at 1.7,
and tensor-order anisotropic pitches `(0.7, 1.9)` or `(0.7, 1.3, 1.9)`.
The independent reference uses physical cell bounds; it does not call the native metric helpers.
The chord check enumerates named ray cases on those beam/grid/spacing combinations.
Comparisons use `rtol=3e-5`, `atol=3e-6`; the grazing case also uses a row-dependent image to catch assignment to the wrong voxel row.
All finite domains and numerical tolerances are in the check files.

`test_detector_surfaces.py` adds exhaustive curved matrix cases, explicit
flat-point comparisons and endpoint VJP cases. The curved domain includes
shared and per-view arcs, as well as cone surfaces whose three local offsets
depend jointly on both detector parameters. The VJP checks perturb every pixel
coordinate and require unchanged intersected cells and agreement between two
finite-difference step sizes. Exact symbolic checks cover the associated
surface and chain-rule models. Shapes, spacings and tolerances are documented
in the test module.

`test_chunked_projector.py` adds exact model checks and bounded checks.
The Cartesian partition check enumerates 2D/3D extents 1..3,
limits 1..4 and every cell using scalar spacing 13/10 and per-axis rational
pitches `(7/10, 13/10)` or `(7/10, 13/10, 19/10)`. Production block matrices
enumerate every image/sinogram basis for grids `(H,W)=(3,4)` and
`(D,H,W)=(3,2,4)`, three views, three or `(2,2)` detector pixels, spacing 1.3,
flat asymmetric tiles and per-view curved/coupled one-cell tiles. Extra
boundary cases cover internal faces/edges/corners, source-inside rays and
misses. VJP cases sum block source/direction and pixel derivatives;
float64 finite differences use two step sizes and unchanged intersected cells.
Matrix comparisons use `rtol=4e-5, atol=5e-6`; VJPs use
`rtol=4e-4, atol=4e-5`.

These chunk checks verify mathematical models and actual low-level kernel
bodies with independently assembled blocks. They do not prove the Python
scheduler matches the model or establish GPU-memory bounds. The real CUDA
tests in `tests/test_chunked_projector.py` separately check dispatch, autograd,
saved geometry, automatic sizing and peak/staging bounds. Hardware-gated
two-GPU/NCCL cases require a suitable host.

`test_execution_protocol.py` separately explores every reachable interleaving
of one to three tagged jobs using one or two composite staging slots. Queued
copies sample their source when they complete; the model allows safe early
host reuse after upload and safe GPU reuse after the corresponding consumer.
Each missing dependency has a concrete failing execution. Z3 admission checks
assume the stated component sizes, and SymPy checks exact resident sums over
the finite rank/tile domain. These checks do not establish that the Python
pipeline, allocator or distributed runtime implements the model. The real
CUDA schedule and transfer tests independently observe native launches,
copies, events, storage lifetimes and peak allocations.

Additional bounded models enumerate CPU-fallback requests for one through
four ranks and check that a shared decision preserves collective order.
They also distinguish post-initialization CUDA producer readiness from
storage retention for one through three reads. A separate admission check
requires the prior native allocation to retire before a one-epoch budget
admits its successor. Runtime tests observe these conditions independently.

The same protocol module models balanced spatial ownership, exact block
adjoints and norm counting, collective participation, and CGLS state lifetime.
Its slab enumeration covers sizes zero through nine and one through five
ranks. Symbolic checks separately cover integer slab boundaries and exact
coordinate translations. The CGLS lifetime model uses one exact rational
iteration and explores early q reuse; it does not prove convergence for a
general system. Memory proofs use distinct named state regions and explicit
phase lifetimes. Real two-process Gloo tests independently exercise empty and
uneven ownership, autograd, bounded stores and sharded checkpoints. Physical
multi-GPU/NCCL tests remain hardware gated.

Numba's [simulator documentation](https://nvidia.github.io/numba-cuda/user/simulator.html) describes its execution and limitations.
The simulator executes Python kernel bodies—its arithmetic and scheduling do not establish the behavior of compiled GPU code.
This suite does not verify CUDA type inference, LLVM fastmath, streams,
multi-GPU execution or PyTorch autograd wrappers. Its bounded pixel VJP checks
do not establish the full trajectory/surface gradient path.
The GPU test suite in `tests/` covers those paths on a CUDA machine.

## Installation

On Linux, install CPU torch first:

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r formal/requirements.txt
python -m pip check
NUMBA_ENABLE_CUDASIM=1 FORMAL_REQUIRE_CUDASIM=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  python -m pytest formal -q -ra --junitxml=formal/results.xml
```

The pinned requirements install `numba-cuda[cu12]` on Linux.
On macOS arm64, that target has no compatible wheel. The local command uses Numba's bundled simulator:

```bash
NUMBA_ENABLE_CUDASIM=1 FORMAL_REQUIRE_CUDASIM=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  uv run --no-project --python 3.12 --with-requirements formal/requirements.txt \
  python -m pytest formal -q -ra --junitxml=formal/results.xml
```

`conftest.py` enables CUDASIM before Numba imports. Missing simulator dependencies may skip local kernel checks.
`FORMAL_REQUIRE_CUDASIM=1` or `CI=true` turns simulator unavailability into a failure.
Actual package-import and numerical failures remain errors.
CPU analytical checks execute the source functions without importing the CUDA-dependent package initializer.
No CUDA implementation is replaced by a mock.

Known failures, if found, retain their assertion with a specific strict `xfail` reason.
An unexpected pass then fails CI, so a repaired defect cannot remain hidden as an expected failure.
