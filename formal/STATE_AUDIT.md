# Implicit-state audit — DiffCT

Baseline: `f78a772` (`main`, 2026-11-11). The original developer checkout is unchanged. Work uses isolated TDD worktrees.

## Scope and contracts

Discovery covers the installed Python package, distributed execution, checkpoint persistence, autograd ownership, native transfer lifetimes, lazy kernel loading, seeded random generation and documentation search. Examples and documentation build processes are inspected as call-local entrypoints; this is not a claim of exhaustive numerical correctness or arbitrary concurrent-call support.

The repair cohort is runtime store failure propagation, analytical weight/output alias admission, search request cancellation and read-only CGLS callbacks (explicitly selected by the user during the audit). Public signatures and numerical formulas remain unchanged. Store operations can leave partial output after an exception; transactional rollback is not promised. Process-group transport failure itself is outside the recovery contract. Search keeps its existing shared newest-query policy.

## Inventory and traceability

| State owner | Retained facts and transitions | Relevant existing verification / audit conclusion |
|---|---|---|
| `pipeline._HostSlot` | Pinned buffer, pending event and deferred drain: reserve → copy → wait → consume → reuse | `test_transfer_pipeline`, `test_transfer_retirement_lifetimes`, `test_completed_transfer_reuse` |
| `pipeline._DevicePipeline` | Captured producer/upload/compute/download streams; retained transfer tensors and completion events; close drains and releases | `test_cuda_sink_lifetimes`, `test_output_pipeline_overlap`, `test_execution_buffers`; single-card runtime checked |
| `pipeline._TransferPipeline` | Lazy device-pipe ownership; context closes every created pipe even on error | Transfer suites; no new defect reproduced |
| `chunking` | Immutable plan; live allocator fraction/cache/stream observations determine admission | `test_allocator_capacity`, `test_plan_memory_bounds`; finite admission models |
| `storage.TensorStore` | Caller tensor survives detached reads and overwrite/accumulate writes | `test_block_storage`; numerical exceptions require parent agreement |
| `storage.NpyStore` | Exclusive create → mapped access → explicit flush; mapping persists as store owner | `test_block_storage`, checkpoint tests; no explicit close API promised |
| `streaming` admission | Delegated backing traversal; writable/type/shape/autograd/internal and cross-store overlap checks | `test_block_storage`; analytical weight aliases missing from admission |
| `distribution.Partition` | Parent/subgroup ownership, balanced slices, current-metadata digest cache | `test_partition_lifecycle`, `test_spatial_partition`, `test_portable_distributed_geometry` |
| Hybrid group cache | Weak parent identity → unanimous cache reuse or ordered subgroup construction | Existing concurrency model assumes serialized calls; arbitrary simultaneous constructors remain unsupported |
| `ReferenceProjector` | Per-call snapshot/plan, bounded batch/tile accumulation, successful statistics publication | `test_reference_backend`, portable execution suites; reproduced one-rank I/O failure strands peer |
| Portable autograd | Saved call owner/mode/input geometry; adjoint and bounded VJP replay; collective participation | `test_training_transforms`, `test_portable_bounded_autograd`, distributed transform suites |
| `Projector` constructor/call owner | Detached static frames or live learnable references; override and parameterized-surface owner copies isolate outstanding graphs | `test_call_trajectory`, `test_live_trajectory_validation`, `test_surface_parameter_updates` |
| Fixed geometry cache | Device/view-slice key; stage and record readiness; hit waits and records consumer stream | `test_projector_geometry_devices`, native GPU suites; fixed constructor geometry is cloned |
| Schedule cache | Workload/budget/layout identity → pilot → selected metadata; collective selection bypasses local reuse | `test_execution_schedules`, `test_reconstruction_work_reuse` |
| Execution statistics | Reset before call; publish after success; error must leave no successful report | Runtime store failure regression cohort |
| Native autograd Functions | Saved tensor version/lifetime, dimensions and spacings; matched data adjoints; explicit geometry second-derivative rejection | `test_gradcheck`, `test_geometry_autograd`, native geometry/surface suites |
| `ParameterizedSurface` | Sampler and parameter references; invocation captures Python wrapper; execution snapshots tensors for replay | Surface snapshot, saved-budget and replay tests; nonparameter sampler configuration must remain immutable |
| Low-level analytical autograd | Private cloned geometry/isocenter; gather/transpose contexts alternate for data derivatives | `test_analytical_autograd`; not a proof of GPU floating-point behavior |
| External-stream bridge | One cached device/pointer and wrapper; public numerical callers enter matching contexts | No supported runtime failure reproduced; split global publication is a concurrency limitation, not a claimed fixed bug |
| Lazy kernel exports | Unloaded → import dispatcher → module-global publication; observers resolve real current exports | `test_lazy_native_observers` |
| CUDA exit handler | Finalizer registration order → process-exit deallocation suppression | `test_cuda_exit_deallocations`, `test_multi_gpu_exit`; physical multi-GPU execution unavailable here |
| Seeded geometry generation | Per-call private device RNG when seeded; default global RNG only when unseeded | `test_geometry_input_validation`; no newly reproduced defect |
| CGLS recurrence | `x,s,p,r,q,gamma`, norm/history/count; checkpoint and callback after iteration update | Solver and checkpoint suites; user selected read-only callback iterates; new tests reject changes before the next numerical call |
| SIRT recurrence | Iterate, cached row/column normalizers and residual; callbacks/resume force residual recomputation | `test_reconstruction_work_reuse`, `test_solver_api` |
| TV optimizer | Leaf iterate, Adam moments/step, scoped gradient graph, clamp and history; resume restores state | Solver/TV/checkpoint suites |
| Mapped CGLS workspace | Fresh directory admission → mapped states → bounded recurrence → flush/checkpoint/callback | `test_block_cgls`, `test_cgls_state_admission`, `test_solver_collective_failures` |
| Checkpoint persistence | Signature/fingerprints; exclusive rank directories; all arrays flushed before atomic metadata publication and rank agreement | `test_solver_checkpoint_edges`, `test_checkpoint_metadata_validation` |
| Solver callback control | Exceptions and literal-False stop agree before next numerical work | `test_solver_collective_failures`; CGLS mutation rejection takes precedence over a callback returning `False`; SIRT/TV retain their existing mutation behavior |
| TV halo context | Captured peer ownership/device/face shape; forward exchange, backward opposite face gradient | `test_spatial_partition`, TV tests; NCCL-specific paths remain hardware gated |
| Analytical execution | Constructor snapshots geometry and derives angles; retained weight tensors reread per tile; bounded filter/gather then group sum | Reproduced output/projection-weight alias corrupts later tiles; numerical I/O errors also need collective agreement |
| Analytical participation | Bundled input-gradient node fixes collective order, including empty ranks | `test_analytical_distribution`, `test_analytical_autograd` |
| Search engine | Cached index promise; model off/idle/loading/ready/error; worker fallback; one pending request ID and persistent preferences | Existing browser ordering/cancellation/error suites; cross-widget cancellation reproduced |
| Search widget | Query ticket, debounce timer, keyword/semantic results, pending flag, expansion and keyboard focus | Current canceled request must settle; obsolete request must not clear newer pending state |
| Search worker | Extractor, latest queued query, busy flag, download counters and pinned/config-caching fetch closure | Existing single-load, cancellation and leak browser tests; only latest queued query retained by design |
| Search core/build | Index vector views/BM25/page maps; build-only extractor and quantization work buffers | Pure ranking functions and build-time state; no new defect reproduced |

## Source-backed questions kept visible

1. The original CGLS live-iterate callback could invalidate retained residual and direction: a reported residual of `0.11559865798979033` differed from the actual `2.2658331394195557`. The user chose a read-only callback contract. Detection concerns the supplied iterate; arbitrary side effects on closed-over operator objects or globals are not sandboxed.
2. Checkpoint signatures are captured before iteration. External changes to operator geometry can make later state and the original signature disagree. Changing operator geometry during a solve is outside the read-only CGLS callback contract; this audit does not claim to detect arbitrary Python side effects.
3. Shared mutable statistics/group caches assume serialized calls. Finite concurrency counterexamples are not evidence that all public APIs support arbitrary concurrent use.
4. Native CUDA distributed runtime abort/recovery remains distinct from portable numerical store I/O. Existing formal bindings explicitly describe an external abort policy; the new failure-recovery tests concern the portable projector and analytical store paths. A single physical GPU cannot validate multi-device/NCCL lifecycle guarantees.

## Evidence

- Baseline CPU: `python -m pytest tests -q --tb=short`, WSL CPU PyTorch 2.13.0, threads limited to two: **2013 passed, 1700 skipped**, 257.45 seconds.
- Baseline physical CUDA transfer/lifetime cohort: **40 passed, 13 skipped**, 87.28 seconds, RTX 4070 SUPER. Two-device cases skipped.
- Combined CPU suite with branch instrumentation and real subprocess coverage: **2189 passed, 1701 skipped**, 786.87 seconds. This checkpoint precedes the final nonfloating-tensor callback diagnostic guard; the subsequent complete CGLS slice is **53 passed**, including the physical native CUDA case, 18.99 seconds.
- Combined initial repair regressions: **121 passed** (69 alias cases and 52 store-failure/recovery cases); alias admission coverage was subsequently strengthened to **78 passing cases**.
- Native analytical host-store execution plus CGLS native callback: **7 passed** with source coverage instrumentation, 7.35 seconds. Native compilation is runtime-tested; Python coverage does not measure compiled CUDA kernel instructions.
- Browser fixture: **6 passed**. Actual `search-ui.js` and `search-core.js` execute through real DOM entrypoints with a controlled Worker and finite test index. Built Sphinx integration, model inference and arbitrary worker failure liveness are outside this fixture.
- Search source mutation: Stryker 9.6.1 generated **24** in-scope mutants; **20 assertion kills, 4 survivors**, no errors/timeouts/no-coverage, raw score **83.33%**. IDs 19–22 only change `console.error` selection; DOM/results/pending behavior is unchanged under the frozen UI contract. Raw score is retained without an adjusted percentage.
- Search coverage: `runSemantic` has **17/17 converter lines and 12/12 V8 block branches**; whole `search-ui.js` has **418/509 lines and 108/145 V8 blocks**. Converted lines include non-executable syntax; V8 blocks do not establish an exhaustive AST/condition denominator. The sixth late-model-error test is runtime evidence; the cancellation model does not add a global error state.
- The maintained TLC runner completed **77/77 expected outcomes**: the original 45 checks plus five new positive configurations, fourteen negative controls and thirteen reachability witnesses. Positive model assumptions and sensitivity/reachability verdicts remain distinct; a deliberately violated witness is not a software mutant kill.

| Positive finite model | Bounds | Generated / distinct states | Search depth |
|---|---|---:|---:|
| `StoreFailure.uneven` | Four parent ranks, two subgroups, two rounds, one failed request then a healthy retry | 14331 / 5552 | 78 |
| `StoreFailure.empty` | Same, with an empty subgroup | 7560 / 3320 | 72 |
| `AnalyticalAlias.small` | Five semantic input kinds, three byte-footprint classes, two tiles | 93 / 78 | 6 |
| `SearchCancel.positive` | Two hosts, two nonreused tickets each, four requests | 251545 / 64613 | 19 |
| `CglsCallback.positive` | Two ranks, two iterations, absent/observer/mutator/raising callbacks | 566 / 429 | 9 |

The final CGLS metadata-replacement repair adds four integer/complex replacement cases and two one-rank observer cases. Each replacement reproduced the old wrong exception before the repair; the observer tests reject a source mutation that initializes the stop flag incorrectly. Final CGLS source and all 53 tests passed independent review. Content checks do not sandbox closed-over objects, post-return writes or mutate-then-restore operations.

Python source coverage and mutation receipts are being finalized against the frozen integration checkpoint. Tool gaps and unclosed outcomes remain explicit. The current mutmut inventory contains 398 I/O-cohort mutants; its decorated-function filter generates none for `_store_phase`, so explicit source-level controls are required in addition to the framework run.

Reproduce the committed finite checks from the repository root:

```text
python formal/tla/check.py --only models --skip-apalache --out <local-results-directory>
python -m pytest tests/test_runtime_store_failures.py tests/test_analytical_input_alias.py tests/test_cgls_readonly_callback.py -q
node docs/search/node_modules/@playwright/test/cli.js test -c docs/search/tests/runtime-cancel.config.mjs
```

No inventory, finite TLC completion or passing sample establishes that every real-world state machine is defect-free. Uncovered branches, surviving mutants and unexecuted hardware boundaries must be reported explicitly.
