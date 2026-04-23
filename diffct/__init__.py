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
    circular_trajectory_2d_fan,
    sinusoidal_trajectory_2d_fan,
    custom_trajectory_2d_fan,
    circular_trajectory_2d_parallel,
    sinusoidal_trajectory_2d_parallel,
    custom_trajectory_2d_parallel,
)

from .analytical import (
    detector_coordinates_1d,
    angular_integration_weights,
    fan_cosine_weights,
    cone_cosine_weights,
    parker_weights,
    ramp_filter_1d,
    parallel_weighted_backproject,
    fan_weighted_backproject,
    cone_weighted_backproject,
)

# For backwards compatibility, also import from differentiable module if it still exists
try:
    from . import differentiable
except ImportError:
    pass

__version__ = '1.3.3.dev0'

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
    'circular_trajectory_2d_fan',
    'sinusoidal_trajectory_2d_fan',
    'custom_trajectory_2d_fan',
    'circular_trajectory_2d_parallel',
    'sinusoidal_trajectory_2d_parallel',
    'custom_trajectory_2d_parallel',
    'detector_coordinates_1d',
    'angular_integration_weights',
    'fan_cosine_weights',
    'cone_cosine_weights',
    'parker_weights',
    'ramp_filter_1d',
    'parallel_weighted_backproject',
    'fan_weighted_backproject',
    'cone_weighted_backproject',
]
