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
| `BudgetAgreement` | Two ranks; free/host capacity 32 or 64 units; graph cost 0 or 8; tiles 0..2; CPU/CUDA and host-data flags | Identical device and host limits | Agreement finishes |
| `Distributed` | Two or three ranks; two rounds; different budgets, tile counts, trainability and CPU fallback | Budget equality; compatible collective-history prefixes | Every rank terminates; errors reach every rank |
| `MemoryEnvelope` | Device limit 8, host limit 2, request sizes 1..4; two calls | Device/host bounds; one reduced tile; empty returned inventory | Cleanup and return |
| `PipelineLifecycle` | Two jobs, two independently selected host/device slots, two calls | Completed reads; safe reuse; empty pools after close; fixed pool capacity | Calls close after success or exceptions |
| `AutogradRouting` | 448 grad/transform/geometry/world/batch forms | Recording uses bounded Function; single-process pure JVP avoids it; geometry JVP rejects | Route selection finishes |
| `RecomputeOrder` | Three ranks, two tiles, three ray batches, two leading pieces; both directions | Piece and pair order | Every rank visits every pair |
| `ThreadSafety` | Two threads, shared statistics, one group-cache entry | Statistics isolation and single cache creation under serialization | Both calls finish |

TLC checks deadlocks. Terminal states have explicit self-loops.
Temporal checks require weak fairness of `Next` and successful eventual completion of modeled asynchronous copies.
Finite constants bound exploration. No extra state constraint removes legal executions.
The runner checks rank symmetry separately for safety. Temporal checks do not use symmetry.

`Distributed.Abort` models a required out-of-band failure signal. Current numerical paths do not implement that action.
Removing it must produce a deadlock. `BudgetAgreement` abstracts an admitted graph cost; it does not reproduce `_graph_room`.
`MemoryEnvelope` assumes exact costs for named allocations—the work arrays, copies, graph and reduced tiles.
Apalache checks that inventory through eight transitions, with request sizes up to 1,048,576 and a separate host ceiling.
This is a bounded symbolic check, not an inductive proof of the Python implementation.

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
Each worker records its source path, backend-file SHA-256 and PyTorch version.
The collector wraps real collectives, group creation, ray pairs, Function dispatch and pipeline calls.
NDJSON records distinguish entry, completion and failure. Failed collectives receive no invented completion event.

CPU runs cover forward/backward, trajectory and surface gradients, empty leading batches, JVP and vmap forms.
Two ranks exercise view and space partitions. `--hybrid` adds four ranks and another Projector construction to check group reuse.
The admission case injects an invalid local shape. The numerical case injects an OOM after admission and retains the failed participant.
This separates protocol behavior from peer-exit detection. The process-group timeout is three seconds; workers have a separate watchdog.
Rendezvous files have unique names. The harness cleans up only workers it created.

Budget-enabled trees also run the real scalar planner with specified allocator observations and real Gloo agreement.
Ranks have unequal free memory, mixed CPU/CUDA data devices and CPU compute fallback.
Allocator observations are injected. This is not a CUDA allocation or performance measurement.
The memory witnesses also execute planned ray tables and a retained data-gradient graph on CPU.
They record array payloads and exclude caller inputs and outputs. These are allocation-size witnesses, not physical CUDA peaks.

`TraceValidation` constrains `ObservedProtocol.ProtocolStep` to the observed sequence.
It checks subgroup signatures, dispatch, outcomes, budgets and resource observations.
`RecomputeTrace` constrains the standalone `RecomputeOrder.Step` to observed rectangle IDs.
Its oracle enumerates rectangles from the plan and ownership. It does not take expected pair order from the trace.
Replay checks explicit chunks and one leading piece. The standalone model checks two pieces across three ranks.

Expected implementation failures are strict: the statistics race, numerical failure isolation and the budget floor on budget-enabled trees.
The numerical case also checks mixed success/error outcomes for one call. A missing expected counterexample fails the runner.
Review a repair before updating its expectation. Do not use one Projector concurrently from several Python threads.
Distributed callers must preserve collective ordering across owners that share a process group.

## CUDA observations

On a host with four visible CUDA devices:

```bash
python formal/tla/collect_trace.py --source . --device cuda --backend nccl \
  --native --hybrid --out .validation/formal-tla/cuda
python formal/tla/check.py --only traces \
  --trace-dir .validation/formal-tla/cuda --out .validation/formal-tla/cuda-checks
```

`--native` selects native serial calls and host-store staging. Distributed cases use the portable NCCL path.
Omit `--native` for portable CUDA routing and geometry observations too.
The collector enables CUDA and pinned-host allocation history, records per-call peaks, and saves snapshots beside the traces.
It saves distributed flight-recorder dumps with PyTorch 2.14's unified `_dump_fr_trace` API.
It sets `TORCH_NCCL_TRACE_BUFFER_SIZE`, `TORCH_FR_BUFFER_SIZE` and `TORCH_DISTRIBUTED_DEBUG=DETAIL`.
If the NCCL watchdog aborts a worker, the parent records the observed timeout exit separately.

CPU checks do not establish CUDA event wiring, physical peaks, allocator rounding, fragmentation, NCCL behavior or compiled-kernel correctness.
Event-synchronization failures and arbitrary callback allocations are outside the resource model.
The CUDA collector and its watchdog-abort branch require hardware acceptance. This work submits no GPU jobs.
The inventory model does not prove C1 for the real allocator. Physical budget claims need classified snapshots and measured peaks.
