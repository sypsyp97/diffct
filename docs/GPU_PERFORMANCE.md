# GPU execution audit

This audit compares the changes in this branch with v2.4.2 (`62c97c7`).
The goal is to remove avoidable host/device transfers and repeated work while
preserving eager geometry errors, independent autograd snapshots, bounded
streaming, numerical recurrence, and distributed ownership.

## Changes

| Path | Change | Preserved boundary |
| --- | --- | --- |
| Trainable native trajectory admission | Reduce validity flags on the device, then observe their combined status; compose flat detector corners together | Invalid geometry still fails before numerical execution or output writes; error details are read only on failure |
| Streamed sampled and per-view detector surfaces | Stage complete admitted view/pixel batches and compose them with one batched kernel | Account for packed frames, offsets, replay workspace and allocation limits; retain the small bounded path when a batch does not fit |
| Full native legacy surfaces | Keep admitted independent snapshots on the input GPU | CPU sampler interfaces, foreign-device fallback and distributed collective ordering remain explicit |
| Surface staging | Reuse shared legacy offsets within a call and pixel tile | Do not cache mutable geometry across calls or replace independent backward snapshots |
| CGLS | Read next gradient norm and residual norm together; retain device scalar reductions and device-local vector updates | FP64 sequential accumulation, denominator checks, convergence, callbacks and checkpoint state retain their semantics |
| SIRT and TV reconstruction | Reuse the preceding residual when no callback can mutate the volume or operator; reuse the initial measurement norm for a zero initial volume | Callback-enabled calls recompute, and resume starts with a fresh residual |
| Numerical analytical reconstruction | Filter each view batch once and reuse it across volume tiles; avoid concatenating a single full-volume contribution | Active autograd and distributed paths retain their original lifetime and reduction behavior |

## Scope and retained transfers

The audit covers projector dispatch, geometry generation/admission, native and
portable backends, surface composition and its VJP, chunk planning, stream and
buffer lifetimes, block stores, distributed collectives, iterative solvers,
analytical filtering/backprojection, and the CUDA array interface bridge.

Some CPU work remains intentional:

- `ParameterizedSurface` samplers receive CPU grids, view IDs and parameters.
  Changing that contract would break existing callbacks. Only their admitted
  outputs are staged to the compute device.
- Mapped stores, checkpoint serialization and acquisition fingerprints need
  bounded host data. Their transfers are not part of an ordinary in-memory
  iteration's vector arithmetic.
- Streaming geometry gradients retain bounded host accumulation when a full
  graph does not fit. Allocating the entire replay graph would defeat the
  memory limit.
- Portable transforms and higher-order autograd retain PyTorch composition
  and validation. CPU metadata snapshots in analytical execution also support
  distributed acquisition agreement.
- Eager validation, Python callbacks and convergence decisions still require
  host observations. Removing them would change when errors and early stops
  become visible.

The earlier report's fixed 21-pixel staging observation is reproducible on
admitted sampled/per-view jobs. Its 14,208-launch example is a loop-count
calculation, not a measured public call: the conservative host budget can
split or reject that proposed batch before those launches occur. Shared legacy
offsets already had a batched path in v2.4.2. The changes extend batching to
the other paths rather than attributing all curved detectors the same cost.

## CUDA/library choices

The implementation retains the existing Torch allocator, pinned staging,
explicit stream/event dependencies and zero-copy Numba array views. It does
not introduce a second GPU allocator or globally change precision settings.

The installed Torch tensor CUDA array interface is version 2 and does not
advertise a producer stream. Numba's default `as_cuda_array(sync=True)` therefore
does not itself prove a host synchronization in this bridge. Disabling it
without a measured benefit would weaken compatibility with other producers.
See the [Numba CUDA array interface documentation](https://nvidia.github.io/numba-cuda/user/cuda_array_interface.html).

CUDA Graph capture requires stable execution and excludes CPU-synchronizing
operations such as `.item()`. Dynamic samplers, admission failures and Python
solver stopping logic do not currently meet that contract. Graph capture is
therefore not enabled implicitly; see [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs).
New matrix-multiplication/low-precision features are not substitutes for the
existing ray traversal and atomic backprojection kernels.

The isolated Windows compatibility check uses PyTorch 2.14.1+cu132,
Numba 0.68.0, numba-cuda 0.30.4 and the CUDA 13.2 toolkit packages. The new
61-test GPU/solver/reconstruction slice passes with that combination; the
337 formal/property/simulator checks also pass with PyTorch 2.14.1 and
Numba 0.68.0.

Installing the newest toolkit independently exposed a real version-mixing
failure: with Torch's cu132 wheel already imported, CUDA 13.4 NVVM used the
nvJitLink DLL preloaded from `torch/lib` and failed with
`ERROR_OUTDATED_LIBRARY`. Aligning the compiler toolkit with the Torch wheel
resolved it. For this tested cu132 environment, the additional constraints are:

```bash
python -m pip install "cuda-toolkit[cccl,cudart,nvrtc,nvvm]==13.2.*" \
    "nvidia-nvjitlink==13.2.*"
```

This is an environment-specific installation example, not a new global
dependency pin. The existing environment was preserved. CUDA component
versions and driver compatibility are documented in the
[NVIDIA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/).

## Reproducing measurements

`examples/benchmark_gpu_overheads.py` uses synthetic data, warms JIT compilation,
records synchronized wall times and incremental peak allocated memory, and
prints the actual source path and library/GPU versions as JSON lines. Run the
same script in separate processes with `PYTHONPATH` pointing to each checkout.
Alternate baseline/candidate order, and do not run tests concurrently with
timings. These measurements include Python admission overhead, not just kernel
event duration.

```bash
python examples/benchmark_gpu_overheads.py --warmup 3 --samples 9
python examples/benchmark_surface_geometry.py --views 384 --detector 736 64 \
    --volume 96 512 512 --warmup 2 --samples 7
```

Small synthetic cases are useful for exposing dispatch and staging overhead;
their speedups must not be extrapolated to large traversal-bound reconstructions.

## Measured results

The final A/B/B/A comparison ran in one Alex job on an A100-SXM4-40GB
`3g.20gb` MIG device, using PyTorch 2.13.0+cu130, Numba 0.67.0 and
numba-cuda 0.30.4. Each block used three warmups and nine timed samples.
The table pools the 18 samples for each version; these are synchronized wall
times, including Python overhead. The small case has 8 views, a 96 x 16
detector, and an 8 x 32 x 32 volume. Solver rows use five iterations.
The [raw samples and candidate source hashes](benchmarks/gpu-performance-20261022.json)
are included for review.

| Case | v2.4.2 (ms) | Candidate (ms) | Ratio |
| --- | ---: | ---: | ---: |
| Live trajectory, project | 7.022 | 4.724 | 1.49x |
| Live trajectory, backproject | 7.079 | 4.742 | 1.49x |
| Per-view streamed surface, project | 275.739 | 7.299 | 37.78x |
| Per-view streamed surface, backproject | 275.978 | 7.435 | 37.12x |
| Sampled streamed surface, project | 285.337 | 5.569 | 51.24x |
| Sampled streamed surface, backproject | 284.816 | 5.523 | 51.57x |
| Full legacy surface, project | 2.855 | 2.530 | 1.13x |
| Full legacy surface, backproject | 2.849 | 2.533 | 1.12x |
| CGLS | 19.051 | 17.948 | 1.06x |
| SIRT | 23.240 | 17.036 | 1.36x |
| TV reconstruction | 32.960 | 26.213 | 1.26x |
| FDK, eight volume tiles | 33.557 | 20.767 | 1.62x |

CGLS timing does not establish a stable speedup: its two baseline block
medians were 20.378 and 17.973 ms, versus 17.972 and 17.867 ms for the
candidate. The reduction from three to two iteration readbacks is separately
checked by instrumentation. MIG shares its physical GPU with other partitions,
so small timing differences should be treated cautiously.

The larger surface benchmark uses 384 views, a 736 x 64 detector, and a
96 x 512 x 512 volume, with two warmups and seven samples per block. All
listed cases admitted full execution. Traversal dominates here: legacy
project/backproject were 163.773/206.732 ms before and 162.818/206.046 ms
after; parameterized project/backproject were 163.425/206.731 ms before and
163.205/206.474 ms after. Parameterized geometry-gradient time was essentially
unchanged (454.796 versus 454.374 ms). These are not large-kernel throughput
speedups.

The extra GPU workspace is deliberate and bounded. In the small sampled
projection, incremental peak allocation grew from 312,832 to 608,768 bytes
while eliminating fragmented composition. The large legacy surface-gradient
case grew from 1,176,320,512 to 1,611,053,568 bytes: the full path reserves and
lazily allocates a conservative saved-tensor arena, in addition to independent
geometry snapshots. Its median time changed from 455.517 to 451.873 ms.
The arena can be larger than the values actually saved by a callback; admission
accounts for its full capacity. Overflow, foreign-device and unsupported-layout
saves retain independent CPU fallback. Callback-owned allocations remain outside
DiffCT's workspace budget.
