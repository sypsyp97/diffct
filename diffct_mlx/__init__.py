# diffct_mlx/__init__.py
"""DiffCT-MLX — Differentiable CT Reconstruction for Apple Silicon.

A GPU-accelerated, differentiable computed tomography (CT) forward and backward
projection library built with MLX and custom Metal kernels, optimized for
Apple Silicon (M-series) chips.
"""

from .projectors import (
    parallel_forward,
    parallel_backward,
    fan_forward,
    fan_backward,
    cone_forward,
    cone_backward,
)

from .real_measured_data_helper import (
    apply_detector_array_convention,
    apply_detector_geometry_convention,
    apply_upper_left_detector_transform,
    auto_voxel_spacing_from_detector,
    build_upper_left_detector_transform,
    diagnose_cone_geometry,
    estimate_cone_isocenter,
    normalize_volume,
    resize_volume_to_shape,
    shift_detector_center,
    transform_detector_offsets,
)

from .geometry import (
    circular_trajectory_3d,
    random_trajectory_3d,
    spiral_trajectory_3d,
    sinusoidal_trajectory_3d,
    saddle_trajectory_3d,
    custom_trajectory_3d,
    load_arbitrary_cone_geometry_from_json,
    circular_trajectory_2d_fan,
    sinusoidal_trajectory_2d_fan,
    custom_trajectory_2d_fan,
    circular_trajectory_2d_parallel,
    sinusoidal_trajectory_2d_parallel,
    custom_trajectory_2d_parallel,
)

__version__ = '1.0.0.dev0'

__all__ = [
    # Projector functions
    'parallel_forward',
    'parallel_backward',
    'fan_forward',
    'fan_backward',
    'cone_forward',
    'cone_backward',
    # 3D trajectory generators
    'circular_trajectory_3d',
    'random_trajectory_3d',
    'spiral_trajectory_3d',
    'sinusoidal_trajectory_3d',
    'saddle_trajectory_3d',
    'custom_trajectory_3d',
    'apply_detector_array_convention',
    'apply_detector_geometry_convention',
    'apply_upper_left_detector_transform',
    'auto_voxel_spacing_from_detector',
    'build_upper_left_detector_transform',
    'diagnose_cone_geometry',
    'estimate_cone_isocenter',
    'normalize_volume',
    'resize_volume_to_shape',
    'shift_detector_center',
    'transform_detector_offsets',
    'load_arbitrary_cone_geometry_from_json',
    # 2D fan beam trajectory generators
    'circular_trajectory_2d_fan',
    'sinusoidal_trajectory_2d_fan',
    'custom_trajectory_2d_fan',
    # 2D parallel beam trajectory generators
    'circular_trajectory_2d_parallel',
    'sinusoidal_trajectory_2d_parallel',
    'custom_trajectory_2d_parallel',
]
