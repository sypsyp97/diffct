# DiffCT: Differentiable Computed Tomography Operators

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.14999333-blue.svg?style=flat-square)](https://doi.org/10.5281/zenodo.14999333)
[![PyPI version](https://img.shields.io/pypi/v/diffct.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/diffct/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sypsyp97/diffct)

A high-performance, CUDA-accelerated library for circular orbits CT reconstruction with end-to-end differentiable operators, enabling advanced optimization and deep learning integration.

⭐ **Please star this project if you find it useful!**

## ✨ Features

- **Fast:** CUDA-accelerated projection and backprojection operations
- **Differentiable:** End-to-end gradient propagation for deep learning workflows

## 📐 Supported Geometries

- **Parallel Beam:** 2D parallel-beam geometry
- **Fan Beam:** 2D fan-beam geometry
- **Cone Beam:** 3D cone-beam geometry

## 🧩 Code Structure

```bash
diffct/
├── diffct/
│   ├── __init__.py            # Package initialization
│   ├── differentiable.py      # Differentiable CT operators
├── examples/                  # Example usages
│   ├── fbp_parallel.py
│   ├── fbp_fan.py
│   ├── fdk_cone.py
│   ├── iterative_reco_cone.py
│   ├── iterative_reco_fan.py
├── pyproject.toml             # Project metadata
├── README.md                  # README
├── LICENSE                    # License
├── requirements.txt           # Dependencies
```

## 🚀 Quick Start

### Prerequisites

- CUDA-capable GPU
- Python 3.10+
- PyTorch, NumPy, Numba with CUDA support

### Installation

```bash
# Create and activate environment
conda create -n diffct python=3.10
conda activate diffct

# Install CUDA support
conda install cudatoolkit

git clone https://github.com/sypsyp97/diffct.git
cd diffct

pip install -r requirements.txt
pip install diffct
```

## 📝 Citation

If you use this library in your research, please cite:

```bibtex
@software{DiffCT2025,
  author       = {Yipeng Sun},
  title        = {DiffCT: Differentiable Computed Tomography 
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
