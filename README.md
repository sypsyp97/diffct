# diffct-mlx: Differentiable CT for Apple Silicon

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.14999333-blue.svg?style=flat-square)](https://doi.org/10.5281/zenodo.14999333)

A high-performance, differentiable computed tomography (CT) reconstruction library built with [MLX](https://github.com/ml-explore/mlx) and custom Metal kernels, optimized for Apple Silicon (M-series) chips.

This is the Apple Silicon port of [diffct](https://github.com/sypsyp97/diffct), replacing CUDA/PyTorch with MLX/Metal for native M-series GPU acceleration.

## Features

- **Apple Silicon Native:** Custom Metal kernels via `mx.fast.metal_kernel` — no CUDA required
- **Differentiable:** End-to-end gradient propagation using `mx.custom_function` with custom VJPs
- **Siddon Ray-Tracing:** Bilinear (2D) and trilinear (3D) interpolation for accurate projection
- **Atomic Backprojection:** Thread-safe gradient accumulation using Metal atomic operations

## Supported Geometries

| Geometry | Forward | Backward | Differentiable |
|----------|---------|----------|----------------|
| 2D Parallel Beam | ✅ | ✅ | ✅ |
| 2D Fan Beam | ✅ | ✅ | ✅ |
| 3D Cone Beam | ✅ | ✅ | ✅ |

## Trajectory Generators

- **Circular** — standard single-rotation scan
- **Spiral / Helical** — helical CT with z-axis translation (3D)
- **Sinusoidal** — variable source-to-isocenter distance
- **Saddle** — combined z-oscillation and radial variation (3D)
- **Random** — perturbed circular with configurable noise (3D)
- **Custom** — user-defined source path functions

## Quick Start

### Prerequisites

- Apple Silicon Mac (M1/M2/M3/M4 series)
- Python 3.10+
- macOS 13.5+

### Installation

```bash
# Clone the repository
git clone https://github.com/sypsyp97/diffct.git
cd diffct
git checkout feature/mlx-apple-silicon

# Create and activate conda environment
conda create -n diffct-mlx python=3.11
conda activate diffct-mlx

# Install MLX and dependencies
pip install mlx numpy matplotlib

# Install diffct-mlx
pip install -e .
```

### Basic Usage

```python
import mlx.core as mx
import diffct_mlx

# Create a 64x64 test image
image = mx.ones((64, 64), dtype=mx.float32)

# Generate parallel beam geometry (90 views)
ray_dir, det_origin, det_u_vec = diffct_mlx.circular_trajectory_2d_parallel(90)

# Forward projection → sinogram
sino = diffct_mlx.parallel_forward(
    image, ray_dir, det_origin, det_u_vec,
    num_detectors=92, detector_spacing=1.0, voxel_spacing=1.0
)

# Backprojection → reconstruction
reco = diffct_mlx.parallel_backward(
    sino, ray_dir, det_origin, det_u_vec,
    H=64, W=64, detector_spacing=1.0, voxel_spacing=1.0
)
```

### Gradient Computation

Since all projectors are differentiable, you can compute gradients directly:

```python
import mlx.core as mx
import diffct_mlx

def loss_fn(image):
    ray_dir, det_origin, det_u_vec = diffct_mlx.circular_trajectory_2d_parallel(90)
    sino = diffct_mlx.parallel_forward(image, ray_dir, det_origin, det_u_vec, 92)
    return mx.sum(sino ** 2)

image = mx.ones((64, 64), dtype=mx.float32)
grad_fn = mx.grad(loss_fn)
gradient = grad_fn(image)
```

### 3D Cone Beam Example

```python
import mlx.core as mx
import diffct_mlx

# Create a 32x64x64 volume (D, H, W)
volume = mx.ones((32, 64, 64), dtype=mx.float32)

# Generate cone beam geometry
src, det_c, det_u, det_v = diffct_mlx.circular_trajectory_3d(
    n_views=60, sid=500.0, sdd=1000.0
)

# Forward projection
sino = diffct_mlx.cone_forward(
    volume, src, det_c, det_u, det_v,
    det_u=64, det_v=32, du=1.0, dv=1.0, voxel_spacing=1.0
)

# Backprojection
reco = diffct_mlx.cone_backward(
    sino, src, det_c, det_u, det_v,
    D=32, H=64, W=64, du=1.0, dv=1.0, voxel_spacing=1.0
)
```

## API Reference

### Projectors

| Function | Description |
|----------|-------------|
| `parallel_forward(image, ray_dir, det_origin, det_u_vec, ...)` | 2D parallel beam forward projection |
| `parallel_backward(sinogram, ray_dir, det_origin, det_u_vec, ...)` | 2D parallel beam backprojection |
| `fan_forward(image, src_pos, det_center, det_u_vec, ...)` | 2D fan beam forward projection |
| `fan_backward(sinogram, src_pos, det_center, det_u_vec, ...)` | 2D fan beam backprojection |
| `cone_forward(volume, src_pos, det_center, det_u_vec, det_v_vec, ...)` | 3D cone beam forward projection |
| `cone_backward(sinogram, src_pos, det_center, det_u_vec, det_v_vec, ...)` | 3D cone beam backprojection |

### Trajectory Generators

| Function | Geometry |
|----------|----------|
| `circular_trajectory_2d_parallel(n_views, ...)` | 2D parallel |
| `sinusoidal_trajectory_2d_parallel(n_views, ...)` | 2D parallel |
| `custom_trajectory_2d_parallel(n_views, ...)` | 2D parallel |
| `circular_trajectory_2d_fan(n_views, sid, sdd, ...)` | 2D fan |
| `sinusoidal_trajectory_2d_fan(n_views, sid, sdd, ...)` | 2D fan |
| `custom_trajectory_2d_fan(n_views, sid, sdd, ...)` | 2D fan |
| `circular_trajectory_3d(n_views, sid, sdd, ...)` | 3D cone |
| `spiral_trajectory_3d(n_views, sid, sdd, ...)` | 3D cone |
| `sinusoidal_trajectory_3d(n_views, sid, sdd, ...)` | 3D cone |
| `saddle_trajectory_3d(n_views, sid, sdd, ...)` | 3D cone |
| `random_trajectory_3d(n_views, sid_mean, sdd_mean, ...)` | 3D cone |
| `custom_trajectory_3d(n_views, sid, sdd, ...)` | 3D cone |

## Examples

Ready-to-run scripts are provided in the `examples/` directory:

### Circular Trajectory (Analytical Reconstruction)

| Script | Description |
|--------|-------------|
| `examples/circular_trajectory/fbp_parallel.py` | FBP with ramp filter — 2D parallel beam |
| `examples/circular_trajectory/fbp_fan.py` | FBP with cosine weighting + ramp filter — 2D fan beam |
| `examples/circular_trajectory/fdk_cone.py` | FDK with distance weighting + ramp filter — 3D cone beam |

### Non-Circular Trajectory (Iterative Reconstruction)

| Script | Description |
|--------|-------------|
| `examples/non_circular_trajectory/iterative_reco_parallel.py` | Gradient-based iterative reco — sinusoidal & custom wobble trajectories |
| `examples/non_circular_trajectory/iterative_reco_fan.py` | Gradient-based iterative reco — sinusoidal & custom elliptical trajectories |
| `examples/non_circular_trajectory/iterative_reco_cone.py` | Gradient-based iterative reco — spiral, sinusoidal, saddle & figure-8 trajectories |

Run any example with:

```bash
conda activate diffct-mlx
python examples/circular_trajectory/fbp_parallel.py
```

## Package Structure

```
diffct_mlx/
├── __init__.py          # Public API exports
├── constants.py         # MLX-specific constants and dtypes
├── utils.py             # Grid computation utilities
├── geometry.py          # Trajectory generation functions
├── projectors.py        # Differentiable projector functions with VJPs
└── kernels/
    ├── __init__.py
    ├── parallel_beam.py # Metal kernels for 2D parallel beam
    ├── fan_beam.py      # Metal kernels for 2D fan beam
    └── cone_beam.py     # Metal kernels for 3D cone beam
```

## Citation

If you use this library in your research, please cite:

```bibtex
@software{diffct2025,
  author       = {Yipeng Sun, Linda-Sophie Schneider},
  title        = {diffct: Differentiable Computed Tomography
                 Reconstruction with CUDA},
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.14999333},
  url          = {https://doi.org/10.5281/zenodo.14999333}
}
```

## License

This project is licensed under the Apache 2.0 License — see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project was highly inspired by:

- [PYRO-NN](https://github.com/csyben/PYRO-NN)
- [geometry_gradients_CT](https://github.com/mareikethies/geometry_gradients_CT)
- [MLX](https://github.com/ml-explore/mlx) by Apple

Issues and contributions are welcome!
