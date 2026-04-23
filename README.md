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

## 🔀 Branches

### `main` Branch (Stable, PyPI)
The stable release branch supporting **circular-orbit** CT reconstruction.
Every versioned release on [PyPI](https://pypi.org/project/diffct/) comes
from `main`. Use this if you only need conventional circular fan / cone
beam scans and want a pinned, tested release.

### `dev` Branch (You are here — arbitrary trajectories)
The `dev` branch is the arbitrary-trajectory evolution of the library.
Kernels take per-view ``(src_pos, det_center, det_u_vec[, det_v_vec])``
arrays instead of closed-form ``sdd / sid / beta`` scalars, so you can
reconstruct along **spiral, saddle, sinusoidal, or any user-supplied
trajectory** without touching the CUDA kernels. All of the analytical
FBP / FDK helpers, adjoint guarantees, and gradcheck / benchmark
coverage from `main` are kept in sync — see [CHANGELOG.md](CHANGELOG.md)
for the detailed parity list. The only feature currently deferred from
`main` is the 1.3.0 separable-footprint (SF) projector backends, which
rely on closed-form circular geometry.

⚠️ **Note:** `dev` is under active development and is not published to
PyPI. If you find any bugs please
[raise an issue](https://github.com/sypsyp97/diffct/issues).

## ✨ Features

- **Fast:** CUDA-accelerated forward and backward projectors (Numba
  CUDA kernels), coalesced memory access for the FDK gather.
- **Differentiable:** End-to-end gradient propagation via
  ``torch.autograd``; every projector / backprojector pair is
  byte-accurate adjoints verified by ``tests/test_adjoint_inner_product.py``
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

`dev` is not on PyPI — install it from source by cloning the
repository and using an editable install.

**CUDA 12 (recommended):**
```bash
# Clone the repository and check out the dev branch
git clone https://github.com/sypsyp97/diffct.git
cd diffct
git checkout dev

# Create and activate conda environment
conda create -n diffct python=3.12
conda activate diffct

# Install CUDA (here 12.8.1 as example) and PyTorch, and Numba
conda install nvidia/label/cuda-12.8.1::cuda-toolkit

# Install PyTorch, follow: https://pytorch.org/get-started/locally/

# Install Numba with CUDA 12
pip install numba-cuda[cu12]

# Install diffct (editable)
pip install -e .
```

<details>
<summary>CUDA 13 installation</summary>

```bash
git clone https://github.com/sypsyp97/diffct.git
cd diffct
git checkout dev
conda create -n diffct python=3.12
conda activate diffct
conda install nvidia/label/cuda-13.0.2::cuda-toolkit
# Install PyTorch from https://pytorch.org/get-started/locally/
pip install numba-cuda[cu13]
pip install -e .
```

</details>

<details>
<summary>CUDA 11 installation</summary>

```bash
git clone https://github.com/sypsyp97/diffct.git
cd diffct
git checkout dev
conda create -n diffct python=3.12
conda activate diffct
conda install nvidia/label/cuda-11.8.0::cuda-toolkit
# Install PyTorch from https://pytorch.org/get-started/locally/
pip install numba-cuda[cu11]
pip install -e .
```

</details>

### Running the tests

```bash
pytest tests/ -q                             # 58 tests, ~5 s
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
