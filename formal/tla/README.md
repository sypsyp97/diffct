# Portable execution protocols

Run from the repository root:

```bash
python formal/tla/collect_trace.py --out .validation/formal-tla/traces --hybrid
python formal/tla/check.py --large \
  --cache .cache/formal-tla --out .validation/formal-tla/checks \
  --trace-dir .validation/formal-tla/traces
```

The collector requires `formal/requirements.txt`. The runner uses Python's standard library and Java.
CI supplies Java 21 through `actions/setup-java`. `--java /path/to/java` selects another executable.
Use `--only models` for model checks or `--only traces --trace-dir PATH` for observed executions.
The default runs TLC and Apalache. `--skip-apalache` explicitly selects TLC only.

## Tool pins

[tools.json](tools.json) pins release URLs and SHA-256 hashes for
[TLA+ tools 1.7.4](https://github.com/tlaplus/tlaplus/releases/tag/v1.7.4) and
[Apalache 0.62.4](https://github.com/apalache-mc/apalache/releases/tag/v0.62.4).
The runner verifies downloads and cached files. Missing tools, changed files, timeouts and parser errors fail the run.
The TLC pin uses the stable release. Upstream replaces its 1.8.0 prerelease download during development;
the [upstream README](https://github.com/tlaplus/tlaplus) describes that policy.

## Finite domains

| Module | Domain | Safety | Progress |
| --- | --- | --- | --- |
| `BudgetAgreement` | Two ranks; allocator bounds 12/96 and CUDA-data bounds 6/48; host capacity 32/64; CPU/CUDA, staging and explicit-hint flags; leading limits 1/2 | Smallest device, host and leading limits; no device floor | Agreement finishes |
| `NativeAdmission` | Two ranks; scalar and seven per-axis boundary categories; auto/numba/torch, precision, partition, beam, NVIDIA availability and trajectory dtypes | Native eligibility; agreed selection, spacing and dtype; auto fallback warning | Selection and layout check finish within the modeled boundary |
| `Distributed` | Two or three ranks; two rounds; different budgets, tile counts, trainability and CPU fallback | Budget equality; compatible collective-history prefixes | Every rank terminates; errors reach every rank |
| `MemoryEnvelope` | Device limit 8, host limit 2, request sizes 1..4; two calls; up to three exempt cotangent buffers | Work/host bounds; one reduced tile; separate cotangent exception; empty returned inventory | Cleanup and return |
| `PipelineLifecycle` | Two jobs, two independently selected host/device slots, two calls | Completed reads; safe reuse; empty pools after close; fixed pool capacity | Calls close after success or exceptions |
| `AutogradRouting` | 448 grad/transform/geometry/world/batch forms | Recording uses bounded Function; single-process pure JVP avoids it; geometry JVP rejects | Route selection finishes |
| `RecomputeOrder` | Three ranks, two tiles, three ray batches, two leading pieces, three parameter blocks; both directions and reduction flags | Output SUMs precede pieces; input SUMs precede pairs; parameter SUMs finish replay | Every rank finishes replay |
| `ThreadSafety` | Two threads, shared statistics, one group-cache entry | Statistics isolation and single cache creation under serialization | Both calls finish |

TLC checks deadlocks. Terminal states have explicit self-loops.
Temporal checks require weak fairness of `Next` and successful eventual completion of modeled asynchronous copies.
Finite constants bound exploration. No extra state constraint removes legal executions.
The runner checks rank symmetry separately for safety. Temporal checks do not use symmetry.

`Distributed.Abort` models a required out-of-band failure signal. Current numerical paths do not implement that action.
Removing it must produce a deadlock. `BudgetAgreement` follows the merged negated-MAX agreement, including CUDA data devices and CPU fallback.
It does not subtract a retained forward graph: portable traversal runs without recording one.
`MemoryEnvelope` assumes exact costs for named work allocations—the tables, tile/piece copies, one recomputed pair's graph and reduced tiles.
Caller data, returned arrays, geometry gradients and callback allocations are excluded.
Up to three cotangent-sized buffers in replicated-output geometry backward are modeled separately, outside the work ceiling.
This inventory is an abstract admission policy. It does not prove the byte estimates, minimum-size plan admission or allocator peaks of Python execution.
Apalache checks that inventory through eight transitions, with request sizes up to 1,048,576 and a separate host ceiling.
This is a bounded symbolic check, not an inductive proof of the Python implementation.

[BINDINGS.md](BINDINGS.md) maps every model, configuration, safety mutation and trace observation to the release source.
Native admission uses finite boundary categories; actual traces check `[1e-6, 1e6]` and the inclusive ratio limit `1e4` before conversion.
The layout model starts after local constructor validation. It makes no all-rank progress claim for errors before `Partition` exists.

## Mutation checks

Every listed safety property has a required counterexample.
Mutations remove budget reductions, participation conditions, memory charges, lifetime conditions, cleanup or routing conditions.
Other mutations swap replay order or remove thread serialization.
The runner requires the named invariant or specified deadlock. Unrelated errors fail the check.
Observation mutations exercise every invariant in `ObservedProtocol` and `TraceValidation`.
Reordered real collective entries and geometry pairs must also fail.
Mutation traces test the verifier. They are not evidence of library defects.

## Code observations

`collect_trace.py --source PATH` imports another source tree without editing it.
Each worker records its source path, every library Python file's SHA-256, the exported revision and PyTorch version.
The validator rejects source mismatches, incomplete workers and missing required observation kinds.
The collector wraps real collectives, group creation, ray pairs, Function dispatch and pipeline calls.
NDJSON records distinguish entry, completion and failure. Failed collectives receive no invented completion event.

CPU runs cover forward/backward, trajectory and surface gradients, empty leading batches, JVP and vmap forms.
Two ranks exercise view and space partitions. `--hybrid` adds four-rank hybrid, empty-view and empty-space cases.
Another Projector construction checks process-group reuse.
The admission case injects an invalid local shape. The numerical case injects an OOM after admission and retains the failed participant.
This separates protocol behavior from peer-exit detection. The process-group timeout is three seconds on CPU and ten seconds on CUDA.
Workers have a separate watchdog.
Rendezvous files have unique names. The harness cleans up only workers it created.

The collector also runs the real scalar planner with specified allocator observations and real process-group agreement.
Ranks have unequal free memory, mixed CPU/CUDA data devices and CPU compute fallback.
Allocator observations are injected. This is not a CUDA allocation or performance measurement.
The memory observations execute planned ray tables and require zero retained traversal tensors on CPU.
They record array payloads and exclude caller inputs and outputs. These are allocation-size witnesses, not physical CUDA peaks.

`TraceValidation` constrains `ObservedProtocol.ProtocolStep` to the observed sequence.
It checks subgroup signatures, call completion, dispatch, outcomes, minimum budgets, selection, rejection, warnings and resource observations.
`RecomputeTrace` constrains the standalone `RecomputeOrder.Step` to observed reductions and rectangle IDs.
Its oracle enumerates rectangles and leading pieces from the plan and ownership. It does not take expected order from the trace.
Replay checks explicit chunks and automatically split `(3, 2)` leading batches in both directions, with and without input SUMs.
The harness caps leading limits to exercise splitting; these injected limits are not hardware memory measurements.

Expected implementation failures are strict: the statistics race and numerical failure isolation.
The former budget-floor and retained-forward-graph failures are no longer expected on this release.
The numerical case also checks mixed success/error outcomes for one call. A missing expected counterexample fails the runner.
Review a repair before updating its expectation. Do not use one Projector concurrently from several Python threads.
Distributed callers must preserve collective ordering across owners that share a process group.

## CUDA observations

On a host with four visible CUDA devices:

```bash
python formal/tla/collect_trace.py --source . --device cuda --backend nccl \
  --hybrid --out .validation/formal-tla/cuda
python formal/tla/collect_trace.py --source . --device cuda --backend nccl \
  --native --cases serial --out .validation/formal-tla/cuda-native
python formal/tla/check.py --only traces \
  --trace-dir .validation/formal-tla/cuda --out .validation/formal-tla/cuda-checks
python formal/tla/check.py --only traces \
  --trace-dir .validation/formal-tla/cuda-native --out .validation/formal-tla/cuda-native-checks
```

`--native` selects native serial calls and host-store staging, with scalar and per-axis pitches and an injected store exception.
Distributed cases use the portable NCCL path.
`--cases serial` avoids repeating the portable distributed observations in the native pass.
The collector enables CUDA allocation history, records per-call peaks, and saves snapshots beside the traces.
It records whether the installed PyTorch supports pinned-host history.
It saves distributed flight-recorder dumps through `_dump_fr_trace`, or `_dump_nccl_trace` on older NCCL builds.
It sets `TORCH_NCCL_TRACE_BUFFER_SIZE`, `TORCH_FR_BUFFER_SIZE` and `TORCH_DISTRIBUTED_DEBUG=DETAIL`.
If the NCCL watchdog aborts a worker, the parent records the observed timeout exit separately.

CPU checks do not establish CUDA event wiring, physical peaks, allocator rounding, fragmentation, NCCL behavior or compiled-kernel correctness.
CUDA trace validation checks the observed completion points and collective signatures; it does not prove every event dependency or numerical result.
Event-synchronization failures and arbitrary callback allocations are outside the resource model.
The inventory model does not prove C1 for the real allocator. Physical budget claims need classified snapshots and measured peaks.
Finite models do not prove unbounded rank counts, arbitrary nested transforms, second geometry derivatives or thread-safe Projector sharing.
