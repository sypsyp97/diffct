# Protocol source bindings

These bindings target main `f3447e2` (release 2.4.4). Trace validation compares the SHA-256 of every library Python file with the checked tree.
Line references below refer to that immutable revision. The models are handwritten protocol abstractions, not generated Python semantics.

## Models and configurations

| Model and configuration | Implementation binding | Safety property and required mutation |
| --- | --- | --- |
| `BudgetAgreement.small.cfg` | `diffct/backends.py:306` computes allocator limits over compute and CUDA data devices; `:331` applies the host ceiling; `:340` agrees negated limits through `diffct/distribution.py:58`; `diffct/backends.py:423` agrees leading limits | `DeviceAgreement`: `device`; `HostAgreement`: `host`; `DeviceMinimum`: `deviceMax` restores the removed floor; `HostMinimum`: `hostMax`; `LeadingAgreement`: `localLeading` |
| `NativeAdmission.small.cfg` | `diffct/constants.py:18` checks per-axis bounds before FP32 conversion; `diffct/backends.py:95` selects the backend; `diffct/operators.py:435` handles parallel3d; `:485` checks explicit native spacing; `:610` agrees selected backend, spacing and trajectory dtypes; `:556` warns on auto fallback | `NativeRange`: `range`; `SelectionAgreement`: `selection`; `SpacingAgreement`: `spacing`; `TrajectoryAgreement`: `dtype`; `FallbackWarning`: `warning` |
| `Distributed.small.cfg`, `Distributed.large.cfg`, generated `Distributed.symmetry.cfg` | `diffct/distribution.py:83` preflights local errors and metadata; `diffct/backends.py:849` issues subgroup SUMs; `:890` preflights inputs; `:1016` and `:189` force distributed autograd/geometry participation | `BudgetAgreement`: `localBudget`; `CollectiveOrder`: generated `order`; empty/untrained participation: `skipEmpty`/`skipUntrained` deadlocks; absent numerical abort: `localError` deadlock |
| `MemoryEnvelope.small.cfg`, `MemoryEnvelope.symbolic.cfg` | `diffct/backends.py:360` and `:387` estimate work/staging; `:875` disables traversal recording; `:923` keeps one output piece; `:958` and `:1007` reduce one tile at a time; `:1049` halves geometry work limits; `:1051` retains the exempt cotangent | `DeviceBound`: `work`, `copy`, `graph`, `reduction`; `HostBound`: `host`; `OneReducedTile`: `twoTiles`; `CotangentException`: `cotangent`; `NoLeaks`: `leak` |
| `PipelineLifecycle.small.cfg` | `diffct/pipeline.py:47` waits before host drains; `:96` finishes a slot before reuse; `:113` retains upload ownership; `:152` waits before downloads; `:205` closes after success/errors; `:248` clears the per-call pipeline | `NoUnfinishedRead`: `unfinishedUpload`/`unfinishedDownload`; `NoEarlyReuse`: `reuseHost`/`releaseEarly`; `NoLeaks`: `leak`; `PoolBound`: `growPool` |
| `AutogradRouting.small.cfg` | `diffct/backends.py:56` inspects wrapped values/tangents; `:1016` selects `_PortableApply`; `:194` handles data JVP; `diffct/operators.py:77` rejects trajectory tangents before dispatch | `BoundedRecording`: `shallow`/`empty`; `PureForwardJvp`: `forceFunction`; `GeometryForwardRejected`: `geometryJvp` |
| `RecomputeOrder.small.cfg` | `diffct/backends.py:1059` reduces output replicas; `:1083` chooses a common leading limit; `:1101` visits leading pieces; `:1105` visits projection tiles then ray batches; `:1122` visits adjoint ray batches then tiles; `:1143` reduces parameter-gradient blocks | `ForwardOrder`: `rankOrder`; it constrains the complete reduction/pair sequence, not ordinary forward traversal order |
| `ThreadSafety.small.cfg` | `diffct/backends.py:878` clears shared statistics; `:928` reads recursively published statistics; `diffct/distribution.py:18` checks/creates/stores cached groups | `StatsIsolation`: generated `stats`; `GroupReuse`: generated `groups`; both remove the model's assumed external serialization |

All small configurations check termination under weak fairness. Distributed large also checks error termination.
The generated symmetry configuration checks safety only. It does not use symmetry to claim liveness.
Apalache checks eight transitions of the same inventory with larger integer costs; its graph mutation must violate `DeviceBound`.
The memory inventory assumes sound admission and exact costs. It does not derive those assumptions from `_work_model`.
Its three-buffer cotangent exception is a separate permitted inventory, not evidence of a measured physical peak.
The distributed abort action and external thread serialization describe required policies that the library does not supply.

## Trace bindings

| Observation | Real call or state | Checked constraint |
| --- | --- | --- |
| `collective_enter` / `collective_exit` | Wrapped `torch.distributed.all_reduce`, `all_gather`, `broadcast`; includes control and subgroup data calls | `CollectiveAgreement`, `CompletionCounts`, `CollectivesFinished`; a changed real signature must fail |
| `group_create` | Wrapped `new_group` from `diffct/distribution.py:18`; a second Projector uses the same live parent | `GroupReuse`; repeated creation mutation must fail |
| `plan_agreement`, `budget_result`, `budget_observation` | Real `_work_budget`, `_agreed_limit`, `_work_chunks`, `_intervals` and traversal under saved-tensor hooks | `PlanAgreement`, `BudgetMinimum`, `BudgetBound`; local limits and hardware observations are explicitly injected |
| `route` | Actual `_PortableApply.forward` entries around public project/backproject, backward, no-grad, JVP and vmap calls | `BoundedRoute`, `PureForwardRoute`, `GeometryRejected`; one malformed observation per property |
| `selection_observation`, `fallback_warning`, `request_rejection` | Actual auto constructors at pitch boundaries; explicit native rejection; rank layout mismatch; a warmed per-call trajectory dtype mismatch | `NativeSelection`, `FallbackWarning`, `AdmissionRejected`; wrong selection, missing warning and accepted rejection mutations |
| `call_begin`, `call_end`, `call_error`, `runtime_timeout` | Public calls and direct geometry replays; numerical OOM injection keeps the failed rank alive | `CallsFinished`, `NoMixedOutcomes`, `NoRuntimeTimeout`, `NoUnexpectedCrash`; one malformed observation per property |
| `host_read`, `host_reuse`, `pool_close` | `_HostSlot.finish`, `_DevicePipeline.slot`, `_DevicePipeline.close`; scalar/per-axis native stores and a store-write exception | `NoUnfinishedRead`, `NoLeaks`; unfinished read and retained-pool mutations |
| `geometry_begin`, `chunk_plan`, `leading_plan`, `pair_begin`, `geometry_reduce` | Real `_geometry_grad`, `_work_chunks`, `_leading_pieces`, `_intervals` and `_reduce` | `RecomputeTrace` constrains the same `RecomputeOrder.Step`; rectangles/pieces are independently enumerated; swap and truncation mutations must fail |

Every invariant in `ObservedProtocol` and `TraceValidation` has a named malformed execution in `guard_mutations`.
Successful traces must also contain the observation kinds required for their scenario; an empty manifest cannot pass.
Required fields must be present. Defaults apply only to fields that the event does not use.
Every geometry call is parsed. A prior TLC result is reused only for identical domain constants and the complete mapped event sequence.
`RecomputeTrace.ObservedLength` has a truncation mutation. Pair/reduction reordering blocks `TraceStep` and must produce a TLC deadlock.
The collector preserves failed collective entries, actual worker exit codes and parent-observed watchdog termination.

## Boundaries

The numerical-failure and shared-statistics counterexamples remain expected implementation failures.
The budget-floor and retained-forward-graph failures from the pre-merge branches are removed from the expected failures.
Local constructor exceptions before `Partition` exists have no modeled all-rank termination guarantee.
Accepted metadata checks do not promise that arbitrary rank-local numerical exceptions propagate without timeout.
Hardware snapshots and flight-recorder files support inspection of the collected execution; TLC does not classify every allocation or validate derivative values.
