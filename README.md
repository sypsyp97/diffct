# diffct: Differentiable Computed Tomography Operators

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.14999333-blue.svg?style=flat-square)](https://doi.org/10.5281/zenodo.14999333)
[![PyPI version](https://img.shields.io/pypi/v/diffct.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/diffct/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat-square)](https://sypsyp97.github.io/diffct/)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/sypsyp97/diffct/docs.yml?branch=main&label=CI&style=flat-square)](https://github.com/sypsyp97/diffct/actions)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sypsyp97/diffct)

A high-performance, CUDA-accelerated library for CT reconstruction with
end-to-end differentiable operators, supporting both **canonical circular
orbits** and **arbitrary per-view trajectories** (spiral, saddle, random,
custom). Built for optimization and deep-learning integration.

⭐ **Please star this project if you find it useful!**

**Apple/MLX maintenance:** The former `apple` branch is maintained by
[Linda-Sophie Schneider](https://github.com/Linda-SophieSchneider) at
[Linda-SophieSchneider/DiffCT-MLX](https://github.com/Linda-SophieSchneider/DiffCT-MLX).

## Candidate branch for `main`

This checkout builds on `dev`: arbitrary per-view trajectories are the
default geometry model. It is a validation candidate on
`codex/arbitrary-trajectory-multigpu`; GitHub's default
branch and published PyPI releases have not changed. See
[the migration notes](docs/MIGRATION.md) before moving from the circular-only API.

## Quick start

Configure the acquisition once, then use `project()` and `backproject()`:

```python
import torch
from diffct import Projector, spiral_trajectory_3d

trajectory = spiral_trajectory_3d(
    60, sid=100.0, sdd=160.0, z_range=12.0, n_turns=1.0, device="cpu"
)
operator = Projector(trajectory, volume_shape=(32, 32, 32),
                     detector_shape=(48, 40), detector_spacing=(1.0, 1.0))
volume = torch.ones(operator.volume_shape, device="cuda", requires_grad=True)
sinogram = operator.project(volume)       # (views, detector_u, detector_v)
adjoint = operator.backproject(sinogram)  # matched adjoint, not FDK
sinogram.square().sum().backward()
```

For 2D use `beam="fan"` or `beam="parallel"`, a `(height, width)` volume
shape, and an integer detector size. Supply the tuple returned by the geometry
helpers, or your own calibrated per-view tensors. Detector axes are unit
vectors; detector pitch is supplied separately. Volume spacing is isotropic.
Geometry is fixed; gradients propagate through volumes and sinograms.

For multiple GPUs in one process, add `devices=["cuda:0", "cuda:1"]`.
Views are divided between devices; projections are concatenated in acquisition
order and backprojections are summed. Outputs return to the input device.
Each GPU stores the full volume and its own view shard.

For multiple processes or nodes, initialize `torch.distributed` and add
`distributed=True`. Each rank returns only its local views; `operator.view_slice`
selects those views from a full measurement tensor. Backprojection and image
gradients are summed across ranks. All ranks must execute matching calls,
including backward calls. Use sums for rank-local projection losses; divide
losses on replicated backprojection outputs by `operator.world_size`.
Do not add DDP gradient reduction on top of this operator.

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
    examples/distributed_reconstruction.py
```

For Slurm and multiple nodes, see [distributed execution](docs/DISTRIBUTED.md).
The [local validation report](docs/VALIDATION.md) records numeric checks and
measured two-A100 speedups, including the small workload that did not accelerate
in the single-process multi-GPU mode. Multi-node GPU validation is pending
allocation permission.
Arbitrary trajectory support applies to the forward/adjoint model and iterative
reconstruction. Analytical FBP/FDK still has acquisition-specific assumptions.

## ✨ Features

- **Fast:** CUDA-accelerated forward and backward projectors (Numba
  CUDA kernels), coalesced memory access for the FDK gather.
- **Differentiable:** End-to-end gradient propagation via
  ``torch.autograd``; projector / backprojector pairs have numerical
  adjoint checks in ``tests/test_adjoint_inner_product.py``
  and ``tests/test_gradcheck.py``.
- **Arbitrary trajectories:** Kernels consume per-view source /
  detector position arrays, so circular, spiral, saddle, sinusoidal
  or any user-supplied orbit works from the same code path. See
  ``diffct.geometry`` for built-in trajectory generators.
- **Analytical reconstruction:** Amplitude-calibrated FBP / FDK
  pipelines via ``ramp_filter_1d``, ``fan_cosine_weights`` /
  ``cone_cosine_weights``, ``parker_weights``,
  ``angular_integration_weights``, and
  ``parallel_weighted_backproject`` / ``fan_weighted_backproject`` /
  ``cone_weighted_backproject``. Each wrapper dispatches to a
  dedicated voxel-driven gather kernel with the correct
  ``(sid_n / U_n)^2`` weighting and Fourier-convention constant.
- **Modular:** Library split into ``diffct.projectors``,
  ``diffct.geometry``, ``diffct.analytical``, ``diffct.kernels``,
  ``diffct.utils``, ``diffct.constants``. ``diffct.differentiable``
  is retained as a deprecated backward-compatibility shim.
- **Tested:** 62 pytest tests covering adjoint identity, gradcheck,
  smoke, accuracy, offset handling, and 29 ramp-filter window cases.
  Opt-in 27-case ``pytest-benchmark`` perf suite under
  ``tests/benchmarks/``.

## 📐 Supported Geometries

- **Parallel Beam:** 2D parallel-beam geometry
- **Fan Beam:** 2D fan-beam geometry
- **Cone Beam:** 3D cone-beam geometry

Every geometry supports both canonical circular orbits (via the
``circular_trajectory_*`` helpers) and arbitrary trajectories (any
user-supplied ``(n_views, 2 or 3)`` tensors).

## 🧩 Code Structure

```bash
diffct/
├── diffct/
│   ├── __init__.py            # public API re-exports
│   ├── constants.py           # dtype, TPB, JIT decorators
│   ├── utils.py               # DeviceManager, TorchCUDABridge, grid helpers
│   ├── geometry.py            # trajectory generators (circular, spiral, ...)
│   ├── operators.py           # Projector API, device and process orchestration
│   ├── projectors.py          # autograd Function classes
│   ├── analytical.py          # ramp filter, cosine weights, Parker, FBP/FDK wrappers
│   ├── kernels/
│   │   ├── parallel_beam.py   # Siddon forward/adjoint + FBP gather
│   │   ├── fan_beam.py        # Siddon forward/adjoint + FBP gather
│   │   └── cone_beam.py       # Siddon forward/adjoint + FDK gather
│   └── differentiable.py      # deprecated compat shim
├── examples/
│   ├── circular_trajectory/   # canonical circular-orbit examples (fbp/fdk + iterative)
│   ├── non_circular_trajectory/  # spiral / custom trajectory examples
│   ├── distributed_reconstruction.py  # helical reconstruction with torchrun
│   └── plot_trajectory.py     # visualise a trajectory generator
├── tests/
│   ├── test_*.py              # adjoint / gradcheck / accuracy / weights / ramp-filter
│   └── benchmarks/            # opt-in pytest-benchmark perf suite
├── docs/                      # Sphinx documentation sources
├── pyproject.toml
├── pytest.ini
├── CHANGELOG.md               # dev-branch change log
├── README.md
└── LICENSE
```

## 🚀 Quick Start

### Prerequisites

- CUDA-capable GPU
- Python 3.10+
- [PyTorch](https://pytorch.org/get-started/locally/), [NumPy](https://numpy.org/), [Numba](https://numba.readthedocs.io/en/stable/user/installing.html), [CUDA](https://developer.nvidia.com/cuda-toolkit)

### Installation

Install this candidate branch from its checkout with `pip install -e .`.
The current PyPI package does not include this candidate's high-level API.

**CUDA 12:**
```bash
# Clone the repository and check out the candidate branch
git clone https://github.com/sypsyp97/diffct.git
cd diffct
git checkout codex/arbitrary-trajectory-multigpu

# Create and activate conda environment
conda create -n diffct python=3.12
conda activate diffct

# Install CUDA (here 12.8.1 as example) and PyTorch, and Numba
conda install nvidia/label/cuda-12.8.1::cuda-toolkit

# Install PyTorch, follow: https://pytorch.org/get-started/locally/

# Install Numba with CUDA 12
pip install "numpy<2.5" "numba-cuda[cu12]"

# Install diffct (editable)
pip install -e .
```

<details>
<summary>CUDA 13 installation</summary>

```bash
git clone https://github.com/sypsyp97/diffct.git
cd diffct
git checkout codex/arbitrary-trajectory-multigpu
conda create -n diffct python=3.12
conda activate diffct
# Install PyTorch from https://pytorch.org/get-started/locally/
pip install "numpy<2.5" "numba-cuda[cu13]" \
    "cuda-toolkit[cccl,cudart,nvrtc,nvvm]==13.0.2" "nvidia-nvjitlink<13.1"
pip install -e .
```

</details>

Numba CUDA currently imports `numpy.row_stack`, which NumPy 2.5 removed.
The NumPy upper bound keeps the CUDA compiler import working. Keep NVVM and
NVJitLink compatible with the CUDA libraries loaded by PyTorch; the CUDA 13.0
recipe above was used for this candidate's checks. A newer NVVM with an older
NVJitLink can fail at kernel compilation before any projection runs.

### Running the tests

```bash
python -m pytest tests/ -q
pytest tests/benchmarks/ --benchmark-only    # opt-in perf suite
```

## 📝 Citation

If you use this library in your research, please cite:

```bibtex
@software{diffct2025,
  author       = {Yipeng Sun},
  title        = {diffct: Differentiable Computed Tomography 
                 Reconstruction with CUDA},
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.14999333},
  url          = {https://doi.org/10.5281/zenodo.14999333}
}
```

## 📄 License

This project is licensed under the Apache 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

This project was highly inspired by:

- [PYRO-NN](https://github.com/csyben/PYRO-NN)
- [geometry_gradients_CT](https://github.com/mareikethies/geometry_gradients_CT)

Issues and contributions are welcome!
