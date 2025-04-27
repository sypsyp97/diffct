# DiffCT: Differentiable Computed Tomography Operators

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.14999333-blue.svg?style=flat-square)](https://doi.org/10.5281/zenodo.14999333)
[![PyPI version](https://img.shields.io/pypi/v/diffct.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/diffct/)
[![Documentation](https://img.shields.io/badge/Documentation-DeepWiki-blue.svg?style=flat-square)](https://deepwiki.com/sypsyp97/diffct)

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
│   ├── non_differentiable.py  # CUDA implementation
│   ├── differentiable.py      # Differentiable implementation
├── examples/                  # Example usages
│   ├── non_differentiable     # Non-differentiable examples
│   │   ├── parallel.py        
│   │   ├── fan.py             
│   │   ├── cone.py            
│   ├── differentiable         # Differentiable examples
│   │   ├── parallel.py        
│   │   ├── fan.py             
│   │   ├── cone.py            
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

## 📚 Usage Examples

### Non-Differentiable CUDA Implementation

```python
import torch
import numpy as np
from diffct.non_differentiable import forward_parallel_2d, back_parallel_2d

# Create phantom
phantom = shepp_logan_2d(256, 256)

# Configure geometry
angles = np.linspace(0, 2*np.pi, 360, endpoint=False)

# Forward projection
sinogram = forward_parallel_2d(
    phantom, 
    num_views=360,
    num_detectors=512, 
    detector_spacing=1.0, 
    angles=angles, 
    step=0.5
)

# Reconstruction
sinogram_filtered = ramp_filter(torch.from_numpy(sinogram)).numpy()
reconstruction = back_parallel_2d(
    sinogram_filtered, 
    Nx=256, Ny=256,
    detector_spacing=1.0, 
    angles=angles, 
    step=0.5
) / 360  # Normalize by number of angles
```

### Differentiable Implementation

```python
import torch
from diffct.differentiable import ParallelProjectorFunction, ParallelBackprojectorFunction

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create phantom tensor with gradient tracking
phantom = torch.tensor(shepp_logan_2d(256, 256), device=device, requires_grad=True)
angles = torch.linspace(0, 2*np.pi, 360, device=device)

# Forward projection
sinogram = ParallelProjectorFunction.apply(
    phantom, 
    angles, 
    num_detectors=512, 
    detector_spacing=1.0, 
    step_size=0.5
)

# Filtered backprojection
sinogram_filtered = ramp_filter(sinogram).requires_grad_(True)
reconstruction = ParallelBackprojectorFunction.apply(
    sinogram_filtered,
    angles, 
    detector_spacing=1.0, 
    step_size=0.5, 
    Nx=256, Ny=256
) / 360  # Normalize

# Compute loss and gradients
loss = torch.mean((reconstruction - phantom)**2)
loss.backward()  # Gradients flow through the entire pipeline
```

## 📊 HU Calibration

1. **With Original Image:**
    - Normalize image before forward projection
    - Apply inverse transformation to restore HU range
    - Consider histogram matching if needed

2. **With Sinogram Only:**
    - Reconstruct to get raw values
    - Calibrate using reference points (air ≈ -1000 HU, water ≈ 0 HU)
    - Or use calibration markers with known attenuation coefficients

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
