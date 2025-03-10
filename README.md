# Differentiable Computed Tomography Reconstruction with CUDA

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
[![DOI](https://zenodo.org/badge/945931443.svg)](https://doi.org/10.5281/zenodo.14999333)

An CUDA-based library for computed tomography (CT) projection and reconstruction with differentiable operators.

Please star this project if you use this repository in your research. Thank you!

## Overview

This library provides GPU-accelerated implementations of CT geometry with circular trajectories. The following geometries are supported:

- Parallel beam (2D)
- Fan beam (2D)
- Cone beam (3D)

Each geometry includes:

- Forward projection operators (ray-tracing)
- Backprojection operators
- Differentiable versions of both operators for gradient-based optimization
- GPU acceleration for high-performance computation

## Code Structure

```bash
├── parallel_cuda.py           # CUDA implementation of parallel beam CT
├── parallel_differentiable.py # Differentiable parallel beam CT and example
├── parallel_example.py        # Usage example for parallel beam CT
├── fan_cuda.py                # CUDA implementation of fan beam CT
├── fan_differentiable.py      # Differentiable fan beam CT and example
├── fan_example.py             # Usage example for fan beam CT
├── cone_cuda.py               # CUDA implementation of cone beam CT
├── cone_differentiable.py     # Differentiable cone beam CT and example
├── cone_example.py            # Usage example for cone beam CT
```

## Requirements

- CUDA-capable GPU
- Python 3.10+
- NumPy
- PyTorch
- Numba with CUDA support
- Matplotlib (for examples)

## Usage Examples

### Installation

```bash
pip install -r requirements.txt
```

### Standard CUDA Implementation

```python
import numpy as np
from parallel_cuda import forward_parallel_2d, back_parallel_2d

# Create phantom
phantom = shepp_logan_2d(256, 256)

# Forward projection
sinogram = forward_parallel_2d(
    phantom, num_views=180, num_detectors=512, 
    detector_spacing=1.0, angles=angles, step_size=0.5
)

# Filter sinogram
sinogram_filtered = ramp_filter(sinogram)

# Backprojection (reconstruction)
reconstruction = back_parallel_2d(
    sinogram_filtered, Nx=256, Ny=256, 
    detector_spacing=1.0, angles=angles, step_size=0.5
)
```

### Differentiable Implementation

```python
import torch
from parallel_differentiable import ParallelProjectorFunction, ParallelBackprojectorFunction

# Convert phantom to PyTorch tensor
phantom_tensor = torch.tensor(phantom, device='cuda', requires_grad=True)
angles_tensor = torch.tensor(angles, device='cuda')

# Forward projection
sinogram = ParallelProjectorFunction.apply(
    phantom_tensor, angles_tensor, num_detectors=512, 
    detector_spacing=1.0, step_size=0.5
)

# Filter sinogram on GPU
sinogram_filtered = ramp_filter(sinogram)

# Backprojection
reconstruction = ParallelBackprojectorFunction.apply(
    sinogram_filtered, angles_tensor, detector_spacing=1.0, 
    step_size=0.5, Nx=256, Ny=256
)

# Gradient computation
loss = torch.mean((reconstruction - phantom_tensor)**2)
loss.backward()
```

## License

This project is licensed under the Apache 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project was highly inspired by the [PYRO-NN](https://github.com/csyben/PYRO-NN) and [geometry_gradients_CT](https://github.com/mareikethies/geometry_gradients_CT) repositories.
