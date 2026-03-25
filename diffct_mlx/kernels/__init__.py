"""Metal kernels for DiffCT-MLX.

This package contains Metal shader source strings and compiled kernel
objects for CT forward projection and backprojection on Apple Silicon.
"""

from .parallel_beam import (
    parallel_2d_forward_kernel,
    parallel_2d_backward_kernel,
)
from .fan_beam import (
    fan_2d_forward_kernel,
    fan_2d_backward_kernel,
)
from .cone_beam import (
    cone_3d_forward_kernel,
    cone_3d_backward_kernel,
)

__all__ = [
    'parallel_2d_forward_kernel',
    'parallel_2d_backward_kernel',
    'fan_2d_forward_kernel',
    'fan_2d_backward_kernel',
    'cone_3d_forward_kernel',
    'cone_3d_backward_kernel',
]
