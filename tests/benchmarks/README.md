# diffct benchmarks (dev branch)

Performance benchmarks for every CUDA kernel in the dev branch's
arbitrary-trajectory `diffct`, built on `pytest-benchmark`. These
are **opt-in** — they are excluded from the default `pytest tests/`
run by `--ignore=tests/benchmarks` in `pytest.ini`, so the regular
test suite stays fast.

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

Compare against a saved baseline:

```bash
pytest tests/benchmarks/ --benchmark-only --benchmark-save=baseline
# ... make your changes ...
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

## Trajectory API

The dev branch kernels take per-view `(src_pos, det_center, det_u_vec
[, det_v_vec])` arrays instead of closed-form `sdd / sid / angles`
scalars. The benchmarks use `circular_trajectory_*` helpers from
`diffct.geometry` to build a canonical circular orbit for each run,
so the numbers reported here are directly comparable to a classical
circular-orbit FBP / FDK implementation.
