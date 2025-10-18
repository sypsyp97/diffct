# diffct/__init__.py
"""DiffCT - Differentiable CT Reconstruction Package.

A GPU-accelerated, differentiable computed tomography (CT) forward and backward
projection library built with PyTorch and Numba CUDA.
"""

from .projectors import (
    ParallelProjectorFunction,
    ParallelBackprojectorFunction,
    FanProjectorFunction,
    FanBackprojectorFunction,
    ConeProjectorFunction,
    ConeBackprojectorFunction,
)

from .geometry import (
    circular_trajectory_3d,
    random_trajectory_3d,
    spiral_trajectory_3d,
    sinusoidal_trajectory_3d,
    saddle_trajectory_3d,
    custom_trajectory_3d,
)

# For backwards compatibility, also import from differentiable module if it still exists
try:
    from . import differentiable
except ImportError:
    pass

__version__ = '1.2.7'

__all__ = [
    'ParallelProjectorFunction',
    'ParallelBackprojectorFunction',
    'FanProjectorFunction',
    'FanBackprojectorFunction',
    'ConeProjectorFunction',
    'ConeBackprojectorFunction',
    'circular_trajectory_3d',
    'random_trajectory_3d',
    'spiral_trajectory_3d',
    'sinusoidal_trajectory_3d',
    'saddle_trajectory_3d',
    'custom_trajectory_3d',
]