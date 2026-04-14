# diffct benchmarks

Performance benchmarks for every CUDA kernel in `diffct`, built on
`pytest-benchmark`. These are **opt-in** — they are excluded from the
default `pytest tests/` run by `--ignore=tests/benchmarks` in
`pytest.ini`, so the regular test suite stays fast.

## Running

Run everything:

```bash
/c/Users/sun/miniconda3/envs/cuda12/python.exe -m pytest tests/benchmarks/ \
    --benchmark-only --benchmark-columns=min,mean,median,stddev,rounds
```

Run a single geometry:

```bash
pytest tests/benchmarks/test_bench_cone.py --benchmark-only
```

Compare against a saved baseline (useful for before/after perf work):

```bash
# first, save a baseline
pytest tests/benchmarks/ --benchmark-only --benchmark-save=baseline

# ... make your changes ...

# then compare
pytest tests/benchmarks/ --benchmark-only --benchmark-compare=baseline
```

## What gets benchmarked

Each geometry has a matching benchmark file that covers three kernels
at three sizes (small / medium / large). All benchmarks run on CUDA
and `pytest.skip` automatically when CUDA is unavailable.

| File | Kernel under test |
|---|---|
| `test_bench_parallel.py` | `ParallelProjectorFunction`, `ParallelBackprojectorFunction`, `parallel_weighted_backproject` |
| `test_bench_fan.py` | `FanProjectorFunction`, `FanBackprojectorFunction`, `fan_weighted_backproject` |
| `test_bench_cone.py` | `ConeProjectorFunction`, `ConeBackprojectorFunction`, `cone_weighted_backproject` |

For the weighted-backproject helpers the benchmark times the full
analytical FBP / FDK pipeline (cosine pre-weight + ramp filter +
angular weights + voxel-gather kernel), because that is what a user
actually pays for when reconstructing.

## How it works

`pytest-benchmark` re-runs the target callable many times and reports
the statistics (min / mean / median / stddev / rounds). CUDA streams
are synchronised inside every call so we measure real GPU time rather
than async kernel launch latency. The `pytest.ini` defaults
(`benchmark_disable_gc`, `benchmark_warmup`,
`benchmark_min_rounds=3`, `benchmark_max_time=2.0`) keep each
benchmark under a couple of seconds while still averaging out noise.
