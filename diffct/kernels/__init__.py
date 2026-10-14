"""CUDA kernels for CT projections.

This subpackage contains CUDA kernels for 2D and 3D CT forward projection
and backprojection operations.
"""

from importlib import import_module

_KERNEL_MODULES = {
    '_parallel_2d_forward_kernel': '.parallel_beam',
    '_parallel_2d_backward_kernel': '.parallel_beam',
    '_parallel_2d_fbp_backproject_kernel': '.parallel_beam',
    '_fan_2d_forward_kernel': '.fan_beam',
    '_fan_2d_backward_kernel': '.fan_beam',
    '_fan_2d_fbp_backproject_kernel': '.fan_beam',
    '_cone_3d_forward_kernel': '.cone_beam',
    '_cone_3d_backward_kernel': '.cone_beam',
    '_cone_3d_fdk_backproject_kernel': '.cone_beam',
    '_parallel_2d_geometry_vjp_kernel': '.geometry_vjp',
    '_fan_2d_geometry_vjp_kernel': '.geometry_vjp',
    '_cone_3d_geometry_vjp_kernel': '.geometry_vjp',
    '_parallel_2d_fbp_transpose_kernel': '.analytical_transpose',
    '_fan_2d_fbp_transpose_kernel': '.analytical_transpose',
    '_cone_3d_fdk_transpose_kernel': '.analytical_transpose',
}


def __getattr__(name):
    """Load the real CUDA dispatcher only when its export is requested."""
    if name not in _KERNEL_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    kernel = getattr(import_module(_KERNEL_MODULES[name], __name__), name)
    globals()[name] = kernel
    return kernel

__all__ = [
    '_parallel_2d_forward_kernel',
    '_parallel_2d_backward_kernel',
    '_parallel_2d_fbp_backproject_kernel',
    '_fan_2d_forward_kernel',
    '_fan_2d_backward_kernel',
    '_fan_2d_fbp_backproject_kernel',
    '_cone_3d_forward_kernel',
    '_cone_3d_backward_kernel',
    '_cone_3d_fdk_backproject_kernel',
    '_parallel_2d_geometry_vjp_kernel',
    '_fan_2d_geometry_vjp_kernel',
    '_cone_3d_geometry_vjp_kernel',
    '_parallel_2d_fbp_transpose_kernel',
    '_fan_2d_fbp_transpose_kernel',
    '_cone_3d_fdk_transpose_kernel',
]
