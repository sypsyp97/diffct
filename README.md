# diffct: Differentiable CT Operators for Arbitrary Trajectories

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.14999333-blue.svg?style=flat-square)](https://doi.org/10.5281/zenodo.14999333)
[![PyPI version](https://img.shields.io/pypi/v/diffct.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/diffct/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat-square)](https://sypsyp97.github.io/diffct/)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/sypsyp97/diffct/docs.yml?branch=main&label=CI&style=flat-square)](https://github.com/sypsyp97/diffct/actions)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sypsyp97/diffct)

diffct gives CUDA forward projectors and matched backprojectors for parallel,
fan and cone beam CT. Every view has its own source, detector position and
detector axes, so circular, helical and calibrated trajectories use the same
code. The operators are PyTorch autograd functions and run on one GPU, several
GPUs or several nodes.

> **Branch status.** This README describes the candidate branch
> `codex/arbitrary-trajectory-multigpu`. The GitHub default branch and the PyPI
> release do not contain the `Projector` API yet. Read the
> [migration notes](docs/MIGRATION.md) before you move from the circular-only API.
>
> The Apple/MLX port is maintained by
> [Linda-Sophie Schneider](https://github.com/Linda-SophieSchneider) at
> [DiffCT-MLX](https://github.com/Linda-SophieSchneider/DiffCT-MLX).

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Geometry and units](#geometry-and-units)
- [Gradients](#gradients)
- [Multiple GPUs and nodes](#multiple-gpus-and-nodes)
- [Analytical FBP and FDK](#analytical-fbp-and-fdk)
- [Validation](#validation)
- [Repository layout](#repository-layout)
- [Citation](#citation)

## Installation

You need a CUDA GPU, Python 3.10 or later, PyTorch, NumPy and Numba CUDA.

```bash
git clone https://github.com/sypsyp97/diffct.git
cd diffct
git checkout codex/arbitrary-trajectory-multigpu

conda create -n diffct python=3.12
conda activate diffct
# Install PyTorch for your CUDA version: https://pytorch.org/get-started/locally/
pip install "numpy<2.5" "numba-cuda[cu12]"   # use [cu13] for CUDA 13
pip install -e .
```

<details>
<summary>Notes on CUDA versions</summary>

- Numba CUDA imports `numpy.row_stack`, which NumPy 2.5 removes. Keep
  `numpy<2.5`.
- Keep NVVM and NVJitLink compatible with the CUDA libraries that PyTorch
  loads. A newer NVVM with an older NVJitLink fails when the kernels compile.
- A tested CUDA 13 set:
  `pip install "numpy<2.5" "numba-cuda[cu13]" "cuda-toolkit[cccl,cudart,nvrtc,nvvm]==13.0.2" "nvidia-nvjitlink<13.1"`.
- A tested CUDA 12 set: PyTorch 2.10 (cu126) with `numba-cuda[cu12]` 0.30.4 on
  NVIDIA driver 535.

</details>

## Quick start

Configure the acquisition once. Then call `project()` and `backproject()`.

```python
import torch
from diffct import Projector, spiral_trajectory_3d

trajectory = spiral_trajectory_3d(
    60, sid=100.0, sdd=160.0, z_range=12.0, n_turns=1.0, device="cpu"
)
operator = Projector(trajectory, volume_shape=(32, 32, 32),
                     detector_shape=(48, 40), detector_spacing=(1.0, 1.0))

volume = torch.ones(operator.volume_shape, device="cuda", requires_grad=True)
sinogram = operator.project(volume)        # (views, detector_u, detector_v)
adjoint = operator.backproject(sinogram)   # matched adjoint, not an FDK image
sinogram.square().sum().backward()         # gradient with respect to volume
```

For 2D, set `beam="fan"` or `beam="parallel"`, use a `(height, width)` volume
and an integer detector size. Iterative reconstruction examples are in
[`examples/non_circular_trajectory/`](examples/non_circular_trajectory/).

## Geometry and units

| Beam | Trajectory tuple, one row per view | Volume | Sinogram |
|---|---|---|---|
| `parallel` | `(ray_dir, det_origin, det_u)`, each `(views, 2)` | `(H, W)` | `(views, U)` |
| `fan` | `(src_pos, det_center, det_u)`, each `(views, 2)` | `(H, W)` | `(views, U)` |
| `cone` | `(src_pos, det_center, det_u, det_v)`, each `(views, 3)` | `(D, H, W)` | `(views, U, V)` |

- Use the helpers in `diffct.geometry` (`circular_*`, `spiral_*`,
  `sinusoidal_*`, `saddle_*`, `random_*`, `custom_*`), or supply calibrated
  tensors.
- Direction vectors have unit length. `ray_dir` is orthogonal to `det_u`, and
  `det_u` is orthogonal to `det_v`. Detector pitch is a separate argument.
- The volume is centred on the origin. Voxel `i` of an axis with `N` voxels has
  its centre at `(i + 0.5 - N / 2) * voxel_spacing`. Voxel spacing is one
  isotropic value.
- The detector array is centred: pixel `k` of `N_det` pixels lies at
  `(k - (N_det - 1) / 2) * pitch` from `det_center` along `det_u`, as in `main`.
- Projections are line integrals in the length unit of the geometry. Fan and
  cone rays run from the source to the detector pixel.
- `Projector` rejects views where the source equals the detector centre or the
  detector is edge-on to the source.
- The kernels work in float32. In fan and cone beams, the source or the
  detector centre must be within 1e6 voxels of the volume centre in each view.
  Ray positions are accurate to about 6e-8 times that nearer distance.

## Gradients

- `project()` and `backproject()` are differentiable with respect to the volume
  and the sinogram. Second derivatives, for example Hessian-vector products,
  work for these inputs.
- Set `requires_grad=True` on trajectory tensors to get geometry gradients, for
  example for calibration or trajectory optimization. The projector then keeps
  references to these tensors and reads their current values at every call.
  The low-level Function classes also return geometry gradients.
- The geometry gradient is the exact derivative of the cell-constant model. It
  is not defined where a ray passes exactly through a voxel edge or corner.
- Second derivatives with respect to the geometry raise an error.
- The geometry checks run only at construction. Keep optimized geometry valid,
  for example by optimizing angles and offsets instead of raw axis vectors.

## Multiple GPUs and nodes

`Projector` splits views between GPUs. Each GPU holds the full volume, so more
GPUs make the operator faster but do not let a larger volume fit.

**One process, several GPUs.** Add `devices=["cuda:0", "cuda:1"]`. Projections
come back in acquisition order. Backprojections are summed. Results return to
the device of the input tensor.

**One process per GPU, one or more nodes.** Initialize an NCCL process group and
add `distributed=True`.

- `project()` returns only the views of the local rank.
  `operator.view_slice` selects the same views from a full measurement tensor.
- `backproject()`, the image gradient and the geometry gradient are summed over
  all ranks.
- Sum projection-domain losses over ranks. Divide a loss on the replicated
  backprojection by `operator.world_size`.
- Do not add a DDP gradient reduction on top of the operator.
- Every rank must make the same `project`, `backproject` and backward calls,
  also a rank that has no views. Otherwise the other ranks block.

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
    examples/distributed_reconstruction.py
```

Slurm launch commands and the numeric cross-node check are in
[docs/DISTRIBUTED.md](docs/DISTRIBUTED.md).

## Analytical FBP and FDK

`diffct.analytical` gives the parts of an FBP or FDK pipeline: `ramp_filter_1d`,
`fan_cosine_weights`, `cone_cosine_weights`, `parker_weights`,
`angular_integration_weights`, and the voxel-driven gathers
`parallel_weighted_backproject`, `fan_weighted_backproject` and
`cone_weighted_backproject`. They accept the same trajectory tensors. FBP and
FDK are exact only for the scan types that they assume. For other trajectories,
use iterative reconstruction with `Projector`. Examples are in
[`examples/circular_trajectory/`](examples/circular_trajectory/).

## Validation

Run the test suite on a CUDA host:

```bash
python -m pytest tests/ -q
pytest tests/benchmarks/ --benchmark-only    # optional performance suite
```

The tests check the adjoint identity `<Ax, y> = <x, A^T y>`, autograd
gradients, ray lengths against a float64 CPU reference, geometry gradients
against a float64 reference model, second derivatives, FBP/FDK accuracy,
geometry validation, and multi-GPU parity with one GPU.

Measured speed on Leonardo Booster (A100-SXM-64GB, NCCL, one process per GPU):
helical cone beam, 1024 views, one gradient iteration (projection and image
gradient), median of 9 runs. Multi-GPU results match one GPU within
`rtol=5e-4`.

| GPUs | 128³ volume | 256³ volume |
|---|---:|---:|
| 1 | 68.1 ms | 526.7 ms |
| 4, one node | 18.2 ms (3.74×) | 136.7 ms (3.85×) |
| 8, two nodes | 10.9 ms (6.25×) | 81.0 ms (6.51×) |

Small workloads accelerate less, because transfers and launches take a larger
share. Measure your own volume, detector and view count with
`examples/benchmark_projector.py`.

Details, commands and the limits of each check are in
[docs/VALIDATION.md](docs/VALIDATION.md).

## Repository layout

```text
diffct/
  operators.py     Projector: geometry checks, device and process orchestration
  projectors.py    autograd Function classes for each beam
  kernels/         Numba CUDA kernels: Siddon projector/backprojector, FBP/FDK gathers
  geometry.py      trajectory generators
  analytical.py    ramp filter, weights, FBP/FDK wrappers
examples/          circular, non-circular and distributed examples
tests/             pytest suite, distributed check, optional benchmarks
docs/              Sphinx sources, distributed, validation and migration notes
```

`diffct.differentiable` is a deprecated alias for the Function classes.

## Citation

```bibtex
@software{diffct2025,
  author    = {Yipeng Sun},
  title     = {diffct: Differentiable Computed Tomography Reconstruction with CUDA},
  year      = 2025,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.14999333},
  url       = {https://doi.org/10.5281/zenodo.14999333}
}
```

## License and acknowledgements

Apache 2.0, see [LICENSE](LICENSE). The project draws on
[PYRO-NN](https://github.com/csyben/PYRO-NN) and
[geometry_gradients_CT](https://github.com/mareikethies/geometry_gradients_CT).
Issues and pull requests are welcome.
