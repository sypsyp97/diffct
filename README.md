# diffct: Differentiable Computed Tomography Operators

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.14999333-blue.svg?style=flat-square)](https://doi.org/10.5281/zenodo.14999333)
[![PyPI version](https://img.shields.io/pypi/v/diffct.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/diffct/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat-square)](https://sypsyp97.github.io/diffct/)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/sypsyp97/diffct/docs.yml?branch=main&label=CI&style=flat-square)](https://github.com/sypsyp97/diffct/actions)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sypsyp97/diffct)

A high-performance, CUDA-accelerated library for circular orbits CT
reconstruction with end-to-end differentiable operators, amplitude-
calibrated analytical FBP / FDK, and a separable-footprint projector
family for cell-integrated forward models. Built for optimization and
deep-learning integration.

⭐ **Please star this project if you find it is useful!**

## 🔀 Branches

### Main Branch (Stable, PyPI)
This is the **stable release** branch supporting circular-orbit CT
reconstruction. Every versioned release on
[PyPI](https://pypi.org/project/diffct/) comes from `main`. See
[CHANGELOG.md](CHANGELOG.md) for the full release history.

### Dev Branch (arbitrary trajectories)
The `dev` branch is the arbitrary-trajectory evolution of the library.
Kernels take per-view ``(src_pos, det_center, det_u_vec[, det_v_vec])``
arrays instead of closed-form `sdd / sid / beta` scalars, so you can
reconstruct along **spiral, saddle, sinusoidal, or any user-supplied
trajectory**. It is kept in sync with `main`'s 1.2.11 analytical
reconstruction overhaul (ramp filter, weighted backproject, tests,
benchmark suite); the only thing currently deferred from `main` is the
1.3.0 separable-footprint (SF) backends, because generalising the
trapezoidal footprint math to arbitrary trajectories is a separate
research effort.

⚠️ **Note:** The dev branch is under active development and is not
published to PyPI. If you find any bugs please
[raise an issue](https://github.com/sypsyp97/diffct/issues).

## ✨ Features

- **Fast:** CUDA-accelerated forward and backward projectors with
  Numba CUDA kernels, plus dedicated voxel-driven FBP / FDK gather
  kernels with coalesced memory writes.
- **Differentiable:** End-to-end gradient propagation via
  ``torch.autograd``. Every projector / backprojector pair is a
  byte-accurate adjoint, verified by
  ``tests/test_adjoint_inner_product.py`` and
  ``torch.autograd.gradcheck`` in ``tests/test_gradcheck.py``.
- **Analytical reconstruction:** Amplitude-calibrated FBP / FDK
  pipelines via ``ramp_filter_1d`` (Ram-Lak / Hann / Hamming /
  cosine / Shepp-Logan windows, configurable padding and physical
  ``sample_spacing``), ``fan_cosine_weights`` /
  ``cone_cosine_weights``, ``parker_weights``,
  ``angular_integration_weights``, and
  ``parallel_weighted_backproject`` / ``fan_weighted_backproject`` /
  ``cone_weighted_backproject``. A unit-density phantom reconstructs
  back to amplitude 1 without any manual scaling.
- **Separable-footprint projectors:** Optional ``backend="sf"``
  (fan) and ``backend="sf_tr"`` / ``"sf_tt"`` (cone) selectors on
  every ``FanProjectorFunction`` / ``ConeProjectorFunction`` call
  expose voxel-driven SF projectors (Long-Fessler-Balter, IEEE TMI
  2010). In the full analytical FBP / FDK pipeline these measurably
  lower reconstruction MSE by ~17 % on standard Shepp-Logan phantoms
  versus the default Siddon backend, at ~2-3x forward cost.
- **Tested:** 66 pytest tests covering adjoint identity, gradcheck,
  smoke, FBP / FDK accuracy per geometry, detector / center offsets,
  and 29 ramp-filter window cases. Opt-in 27-case
  ``pytest-benchmark`` perf suite under ``tests/benchmarks/`` for
  before/after regression tracking.

## 📐 Supported Geometries

- **Parallel Beam:** 2D parallel-beam geometry
- **Fan Beam:** 2D fan-beam geometry
- **Cone Beam:** 3D cone-beam geometry

## 🧩 Code Structure

```bash
diffct/
├── diffct/
│   ├── __init__.py            # public API re-exports
│   └── differentiable.py      # CUDA kernels, autograd Functions,
│                              # analytical helpers, SF backends
├── examples/                  # circular-orbit example scripts
│   ├── fbp_parallel.py
│   ├── fbp_fan.py             # with Parker short-scan switch
│   ├── fdk_cone.py            # with Parker short-scan switch
│   ├── iterative_reco_parallel.py
│   ├── iterative_reco_fan.py
│   └── iterative_reco_cone.py
├── tests/
│   ├── test_*.py              # adjoint / gradcheck / accuracy /
│   │                          # offsets / weights / ramp-filter
│   └── benchmarks/            # opt-in pytest-benchmark perf suite
├── docs/                      # Sphinx documentation sources
├── pyproject.toml             # project metadata
├── pytest.ini
├── CHANGELOG.md               # Keep-a-Changelog release notes
├── README.md
└── LICENSE
```

## 🚀 Quick Start

### Prerequisites

- CUDA-capable GPU
- Python 3.10+
- [PyTorch](https://pytorch.org/get-started/locally/), [NumPy](https://numpy.org/), [Numba](https://numba.readthedocs.io/en/stable/user/installing.html), [CUDA](https://developer.nvidia.com/cuda-toolkit)

### Installation

**CUDA 12 (Recommended):**
```bash
# Create and activate conda environment
conda create -n diffct python=3.12
conda activate diffct

# Install CUDA (here 12.8.1 as example) PyTorch, and Numba
conda install nvidia/label/cuda-12.8.1::cuda-toolkit

# Install Pytorch, you can find the commend here: https://pytorch.org/get-started/locally/

# Install Numba with CUDA 12
pip install numba-cuda[cu12]

# Install diffct
pip install diffct
```

<details>
<summary>CUDA 13 Installation</summary>

```bash
# Create and activate conda environment
conda create -n diffct python=3.12
conda activate diffct

# Install CUDA (here 13.0.2 as example) PyTorch, and Numba
conda install nvidia/label/cuda-13.0.2::cuda-toolkit

# Install Pytorch, you can find the commend here: https://pytorch.org/get-started/locally/

# Install Numba with CUDA 13
pip install numba-cuda[cu13]

# Install diffct
pip install diffct
```

</details>

<details>
<summary>CUDA 11 Installation</summary>

```bash
# Create and activate conda environment
conda create -n diffct python=3.12
conda activate diffct

# Install CUDA (here 11.8.0 as example) PyTorch, and Numba
conda install nvidia/label/cuda-11.8.0::cuda-toolkit

# Install Pytorch, you can find the commend here: https://pytorch.org/get-started/locally/

# Install Numba with CUDA 11
pip install numba-cuda[cu11]

# Install diffct
pip install diffct
```

</details>

### Running the tests

```bash
pytest tests/ -q                             # 66 tests, ~15 s
pytest tests/benchmarks/ --benchmark-only    # opt-in perf suite, requires pytest-benchmark
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
