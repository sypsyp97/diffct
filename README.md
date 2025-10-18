# diffct: Differentiable Computed Tomography Operators

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.14999333-blue.svg?style=flat-square)](https://doi.org/10.5281/zenodo.14999333)
[![PyPI version](https://img.shields.io/pypi/v/diffct.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/diffct/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat-square)](https://sypsyp97.github.io/diffct/)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/sypsyp97/diffct/docs.yml?branch=main&label=CI&style=flat-square)](https://github.com/sypsyp97/diffct/actions)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sypsyp97/diffct)

A high-performance, CUDA-accelerated library for CT reconstruction with end-to-end differentiable operators, enabling advanced optimization and deep learning integration.

⭐ **Please star this project if you find it is useful!**

## ✨ Features

- **Fast:** CUDA-accelerated projection and backprojection operations
- **Differentiable:** End-to-end gradient propagation for deep learning workflows

## 📐 Supported Geometries

- **Parallel Beam:** 2D parallel-beam geometry
- **Fan Beam:** 2D fan-beam geometry
- **Cone Beam:** 3D cone-beam geometry

## 🚀 Quick Start

### Prerequisites

- CUDA-capable GPU
- Python 3.10+
- [PyTorch](https://pytorch.org/get-started/locally/), [NumPy](https://numpy.org/), [Numba](https://numba.readthedocs.io/en/stable/user/installing.html), [CUDA](https://developer.nvidia.com/cuda-toolkit)

### Installation

**CUDA 12:**
```bash
# Clone the repository
git clone https://github.com/sypsyp97/diffct.git
cd diffct

# Create and activate conda environment
conda create -n diffct python=3.12
conda activate diffct

# Install CUDA (here 12.8.1 as example) PyTorch, and Numba
conda install nvidia/label/cuda-12.8.1::cuda-toolkit

# Install Pytorch, you can find the commend here: https://pytorch.org/get-started/locally/

# Install Numba with CUDA 12
pip install numba-cuda[cu12]

# Install diffct
pip install -e .
```

**CUDA 11:**
```bash
# Clone the repository
git clone https://github.com/sypsyp97/diffct.git
cd diffct

# Create and activate conda environment
conda create -n diffct python=3.12
conda activate diffct

# Install CUDA (here 11.8.0 as example) PyTorch, and Numba
conda install nvidia/label/cuda-11.8.0::cuda-toolkit

# Install Pytorch, you can find the commend here: https://pytorch.org/get-started/locally/

# Install Numba with CUDA 11
pip install numba-cuda[cu11]

# Install diffct
pip install -e .
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
