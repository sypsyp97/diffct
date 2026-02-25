"""MLX differentiable projector functions for CT reconstruction.

This module provides differentiable forward projection and backprojection
functions using custom Metal kernels on Apple Silicon, with VJP (vector-Jacobian
product) support for gradient computation via ``mx.custom_function``.
"""

import numpy as np
import mlx.core as mx

from .constants import _MX_DTYPE
from .utils import _grid_2d, _grid_3d
from .kernels import (
    parallel_2d_forward_kernel,
    parallel_2d_backward_kernel,
    fan_2d_forward_kernel,
    fan_2d_backward_kernel,
    cone_3d_forward_kernel,
    cone_3d_backward_kernel,
)


# ============================================================================
# 2D Parallel Beam
# ============================================================================

def _parallel_forward_impl(image, ray_dir, det_origin, det_u_vec,
                           num_detectors, detector_spacing=1.0, voxel_spacing=1.0):
    """Raw parallel beam forward projection (no VJP)."""
    image = image.astype(_MX_DTYPE)
    ray_dir = ray_dir.astype(_MX_DTYPE)
    det_origin = det_origin.astype(_MX_DTYPE)
    det_u_vec = det_u_vec.astype(_MX_DTYPE)

    Ny, Nx = image.shape
    n_ang = ray_dir.shape[0]
    n_det = int(num_detectors)

    cx = float(Nx * 0.5)
    cy = float(Ny * 0.5)

    params = mx.array([n_ang, n_det, Nx, Ny], dtype=mx.int32)
    fparams = mx.array([detector_spacing, cx, cy, voxel_spacing], dtype=mx.float32)

    grid, tg = _grid_2d(n_ang, n_det)

    outputs = parallel_2d_forward_kernel(
        inputs=[image, ray_dir, det_origin, det_u_vec, params, fparams],
        output_shapes=[(n_ang, n_det)],
        output_dtypes=[_MX_DTYPE],
        grid=grid,
        threadgroup=tg,
    )
    return outputs[0]


def _parallel_backward_impl(sinogram, ray_dir, det_origin, det_u_vec,
                            detector_spacing, H, W, voxel_spacing=1.0):
    """Raw parallel beam backprojection (no VJP)."""
    sinogram = sinogram.astype(_MX_DTYPE)
    ray_dir = ray_dir.astype(_MX_DTYPE)
    det_origin = det_origin.astype(_MX_DTYPE)
    det_u_vec = det_u_vec.astype(_MX_DTYPE)

    n_ang, n_det = sinogram.shape
    Ny, Nx = int(H), int(W)

    cx = float(Nx * 0.5)
    cy = float(Ny * 0.5)

    params = mx.array([n_ang, n_det, Nx, Ny], dtype=mx.int32)
    fparams = mx.array([detector_spacing, cx, cy, voxel_spacing], dtype=mx.float32)

    grid, tg = _grid_2d(n_ang, n_det)

    outputs = parallel_2d_backward_kernel(
        inputs=[sinogram, ray_dir, det_origin, det_u_vec, params, fparams],
        output_shapes=[(Ny, Nx)],
        output_dtypes=[_MX_DTYPE],
        grid=grid,
        threadgroup=tg,
        init_value=0,
    )
    return outputs[0]


@mx.custom_function
def parallel_forward(image, ray_dir, det_origin, det_u_vec,
                     num_detectors=128, detector_spacing=1.0, voxel_spacing=1.0):
    """Differentiable 2D parallel beam forward projection.

    Parameters
    ----------
    image : mx.array
        2D input image, shape ``(H, W)``.
    ray_dir : mx.array
        Ray direction unit vectors, shape ``(n_views, 2)``.
    det_origin : mx.array
        Detector origin positions, shape ``(n_views, 2)``.
    det_u_vec : mx.array
        Detector u-direction unit vectors, shape ``(n_views, 2)``.
    num_detectors : int
        Number of detector elements.
    detector_spacing : float
        Physical spacing between detector elements.
    voxel_spacing : float
        Physical voxel size.

    Returns
    -------
    sinogram : mx.array
        Shape ``(n_views, num_detectors)``.
    """
    return _parallel_forward_impl(image, ray_dir, det_origin, det_u_vec,
                                  num_detectors, detector_spacing, voxel_spacing)


@parallel_forward.vjp
def parallel_forward_vjp(primals, cotangent, _):
    image, ray_dir, det_origin, det_u_vec = primals[:4]
    num_detectors = primals[4] if len(primals) > 4 else 128
    detector_spacing = primals[5] if len(primals) > 5 else 1.0
    voxel_spacing = primals[6] if len(primals) > 6 else 1.0

    Ny, Nx = image.shape
    grad_image = _parallel_backward_impl(
        cotangent, ray_dir, det_origin, det_u_vec,
        detector_spacing, Ny, Nx, voxel_spacing
    )
    return (grad_image, None, None, None, None, None, None)


@mx.custom_function
def parallel_backward(sinogram, ray_dir, det_origin, det_u_vec,
                      detector_spacing=1.0, H=128, W=128, voxel_spacing=1.0):
    """Differentiable 2D parallel beam backprojection.

    Parameters
    ----------
    sinogram : mx.array
        2D sinogram, shape ``(n_views, num_detectors)``.
    ray_dir : mx.array
        Ray direction unit vectors, shape ``(n_views, 2)``.
    det_origin : mx.array
        Detector origin positions, shape ``(n_views, 2)``.
    det_u_vec : mx.array
        Detector u-direction unit vectors, shape ``(n_views, 2)``.
    detector_spacing : float
        Physical spacing between detector elements.
    H, W : int
        Output image dimensions.
    voxel_spacing : float
        Physical voxel size.

    Returns
    -------
    reco : mx.array
        Shape ``(H, W)``.
    """
    return _parallel_backward_impl(sinogram, ray_dir, det_origin, det_u_vec,
                                   detector_spacing, H, W, voxel_spacing)


@parallel_backward.vjp
def parallel_backward_vjp(primals, cotangent, _):
    sinogram, ray_dir, det_origin, det_u_vec = primals[:4]
    detector_spacing = primals[4] if len(primals) > 4 else 1.0
    n_ang, n_det = sinogram.shape

    grad_sinogram = _parallel_forward_impl(
        cotangent, ray_dir, det_origin, det_u_vec,
        n_det, detector_spacing, 1.0
    )
    return (grad_sinogram, None, None, None, None, None, None, None)


# ============================================================================
# 2D Fan Beam
# ============================================================================

def _fan_forward_impl(image, src_pos, det_center, det_u_vec,
                      num_detectors, detector_spacing=1.0, voxel_spacing=1.0):
    """Raw fan beam forward projection."""
    image = image.astype(_MX_DTYPE)
    src_pos = src_pos.astype(_MX_DTYPE)
    det_center = det_center.astype(_MX_DTYPE)
    det_u_vec = det_u_vec.astype(_MX_DTYPE)

    Ny, Nx = image.shape
    n_ang = src_pos.shape[0]
    n_det = int(num_detectors)

    cx = float(Nx * 0.5)
    cy = float(Ny * 0.5)

    params = mx.array([n_ang, n_det, Nx, Ny], dtype=mx.int32)
    fparams = mx.array([detector_spacing, cx, cy, voxel_spacing], dtype=mx.float32)

    grid, tg = _grid_2d(n_ang, n_det)

    outputs = fan_2d_forward_kernel(
        inputs=[image, src_pos, det_center, det_u_vec, params, fparams],
        output_shapes=[(n_ang, n_det)],
        output_dtypes=[_MX_DTYPE],
        grid=grid,
        threadgroup=tg,
    )
    return outputs[0]


def _fan_backward_impl(sinogram, src_pos, det_center, det_u_vec,
                       detector_spacing, H, W, voxel_spacing=1.0):
    """Raw fan beam backprojection."""
    sinogram = sinogram.astype(_MX_DTYPE)
    src_pos = src_pos.astype(_MX_DTYPE)
    det_center = det_center.astype(_MX_DTYPE)
    det_u_vec = det_u_vec.astype(_MX_DTYPE)

    n_ang, n_det = sinogram.shape
    Ny, Nx = int(H), int(W)

    cx = float(Nx * 0.5)
    cy = float(Ny * 0.5)

    params = mx.array([n_ang, n_det, Nx, Ny], dtype=mx.int32)
    fparams = mx.array([detector_spacing, cx, cy, voxel_spacing], dtype=mx.float32)

    grid, tg = _grid_2d(n_ang, n_det)

    outputs = fan_2d_backward_kernel(
        inputs=[sinogram, src_pos, det_center, det_u_vec, params, fparams],
        output_shapes=[(Ny, Nx)],
        output_dtypes=[_MX_DTYPE],
        grid=grid,
        threadgroup=tg,
        init_value=0,
    )
    return outputs[0]


@mx.custom_function
def fan_forward(image, src_pos, det_center, det_u_vec,
                num_detectors=128, detector_spacing=1.0, voxel_spacing=1.0):
    """Differentiable 2D fan beam forward projection.

    Parameters
    ----------
    image : mx.array
        2D input image, shape ``(H, W)``.
    src_pos : mx.array
        Source positions, shape ``(n_views, 2)``.
    det_center : mx.array
        Detector center positions, shape ``(n_views, 2)``.
    det_u_vec : mx.array
        Detector u-direction unit vectors, shape ``(n_views, 2)``.
    num_detectors : int
        Number of detector elements.
    detector_spacing : float
        Physical spacing between detector elements.
    voxel_spacing : float
        Physical voxel size.

    Returns
    -------
    sinogram : mx.array
        Shape ``(n_views, num_detectors)``.
    """
    return _fan_forward_impl(image, src_pos, det_center, det_u_vec,
                             num_detectors, detector_spacing, voxel_spacing)


@fan_forward.vjp
def fan_forward_vjp(primals, cotangent, _):
    image, src_pos, det_center, det_u_vec = primals[:4]
    detector_spacing = primals[5] if len(primals) > 5 else 1.0
    voxel_spacing = primals[6] if len(primals) > 6 else 1.0

    Ny, Nx = image.shape
    grad_image = _fan_backward_impl(
        cotangent, src_pos, det_center, det_u_vec,
        detector_spacing, Ny, Nx, voxel_spacing
    )
    return (grad_image, None, None, None, None, None, None)


@mx.custom_function
def fan_backward(sinogram, src_pos, det_center, det_u_vec,
                 detector_spacing=1.0, H=128, W=128, voxel_spacing=1.0):
    """Differentiable 2D fan beam backprojection.

    Parameters
    ----------
    sinogram : mx.array
        2D fan beam sinogram, shape ``(n_views, num_detectors)``.
    src_pos : mx.array
        Source positions, shape ``(n_views, 2)``.
    det_center : mx.array
        Detector center positions, shape ``(n_views, 2)``.
    det_u_vec : mx.array
        Detector u-direction unit vectors, shape ``(n_views, 2)``.
    detector_spacing : float
        Physical spacing between detector elements.
    H, W : int
        Output image dimensions.
    voxel_spacing : float
        Physical voxel size.

    Returns
    -------
    reco : mx.array
        Shape ``(H, W)``.
    """
    return _fan_backward_impl(sinogram, src_pos, det_center, det_u_vec,
                              detector_spacing, H, W, voxel_spacing)


@fan_backward.vjp
def fan_backward_vjp(primals, cotangent, _):
    sinogram, src_pos, det_center, det_u_vec = primals[:4]
    detector_spacing = primals[4] if len(primals) > 4 else 1.0
    n_ang, n_det = sinogram.shape

    grad_sinogram = _fan_forward_impl(
        cotangent, src_pos, det_center, det_u_vec,
        n_det, detector_spacing, 1.0
    )
    return (grad_sinogram, None, None, None, None, None, None, None)


# ============================================================================
# 3D Cone Beam
# ============================================================================

def _cone_forward_impl(volume, src_pos, det_center, det_u_vec, det_v_vec,
                       det_u, det_v, du, dv, voxel_spacing=1.0):
    """Raw cone beam forward projection."""
    volume = volume.astype(_MX_DTYPE)
    src_pos = src_pos.astype(_MX_DTYPE)
    det_center = det_center.astype(_MX_DTYPE)
    det_u_vec = det_u_vec.astype(_MX_DTYPE)
    det_v_vec = det_v_vec.astype(_MX_DTYPE)

    D, H, W = volume.shape
    n_views = src_pos.shape[0]
    n_u, n_v = int(det_u), int(det_v)

    # Permute DHW → WHD for kernel (matches CUDA convention)
    volume_perm = mx.transpose(volume, axes=(2, 1, 0))
    # Make contiguous
    volume_perm = mx.array(volume_perm)

    Nx, Ny, Nz = W, H, D
    cx = float(Nx * 0.5)
    cy = float(Ny * 0.5)
    cz = float(Nz * 0.5)

    params = mx.array([n_views, n_u, n_v, Nx, Ny, Nz], dtype=mx.int32)
    fparams = mx.array([du, dv, cx, cy, cz, voxel_spacing], dtype=mx.float32)

    grid, tg = _grid_3d(n_views, n_u, n_v)

    outputs = cone_3d_forward_kernel(
        inputs=[volume_perm, src_pos, det_center, det_u_vec, det_v_vec, params, fparams],
        output_shapes=[(n_views, n_u, n_v)],
        output_dtypes=[_MX_DTYPE],
        grid=grid,
        threadgroup=tg,
    )
    return outputs[0]


def _cone_backward_impl(sinogram, src_pos, det_center, det_u_vec, det_v_vec,
                        D, H, W, du, dv, voxel_spacing=1.0):
    """Raw cone beam backprojection."""
    sinogram = sinogram.astype(_MX_DTYPE)
    src_pos = src_pos.astype(_MX_DTYPE)
    det_center = det_center.astype(_MX_DTYPE)
    det_u_vec = det_u_vec.astype(_MX_DTYPE)
    det_v_vec = det_v_vec.astype(_MX_DTYPE)

    n_views, n_u, n_v = sinogram.shape
    Nx, Ny, Nz = int(W), int(H), int(D)

    cx = float(Nx * 0.5)
    cy = float(Ny * 0.5)
    cz = float(Nz * 0.5)

    params = mx.array([n_views, n_u, n_v, Nx, Ny, Nz], dtype=mx.int32)
    fparams = mx.array([du, dv, cx, cy, cz, voxel_spacing], dtype=mx.float32)

    grid, tg = _grid_3d(n_views, n_u, n_v)

    outputs = cone_3d_backward_kernel(
        inputs=[sinogram, src_pos, det_center, det_u_vec, det_v_vec, params, fparams],
        output_shapes=[(Nx, Ny, Nz)],  # WHD layout
        output_dtypes=[_MX_DTYPE],
        grid=grid,
        threadgroup=tg,
        init_value=0,
    )

    # Permute WHD → DHW
    vol = mx.transpose(outputs[0], axes=(2, 1, 0))
    return mx.array(vol)


@mx.custom_function
def cone_forward(volume, src_pos, det_center, det_u_vec, det_v_vec,
                 det_u=128, det_v=128, du=1.0, dv=1.0, voxel_spacing=1.0):
    """Differentiable 3D cone beam forward projection.

    Parameters
    ----------
    volume : mx.array
        3D input volume, shape ``(D, H, W)``.
    src_pos : mx.array
        Source positions, shape ``(n_views, 3)``.
    det_center : mx.array
        Detector center positions, shape ``(n_views, 3)``.
    det_u_vec : mx.array
        Detector u-direction unit vectors, shape ``(n_views, 3)``.
    det_v_vec : mx.array
        Detector v-direction unit vectors, shape ``(n_views, 3)``.
    det_u, det_v : int
        Number of detector elements along u and v axes.
    du, dv : float
        Physical spacing between detector elements along u and v.
    voxel_spacing : float
        Physical voxel size.

    Returns
    -------
    sino : mx.array
        Shape ``(n_views, det_u, det_v)``.
    """
    return _cone_forward_impl(volume, src_pos, det_center, det_u_vec, det_v_vec,
                              det_u, det_v, du, dv, voxel_spacing)


@cone_forward.vjp
def cone_forward_vjp(primals, cotangent, _):
    volume, src_pos, det_center, det_u_vec, det_v_vec = primals[:5]
    du = primals[7] if len(primals) > 7 else 1.0
    dv = primals[8] if len(primals) > 8 else 1.0
    voxel_spacing = primals[9] if len(primals) > 9 else 1.0

    D, H, W = volume.shape
    grad_volume = _cone_backward_impl(
        cotangent, src_pos, det_center, det_u_vec, det_v_vec,
        D, H, W, du, dv, voxel_spacing
    )
    return (grad_volume, None, None, None, None, None, None, None, None, None)


@mx.custom_function
def cone_backward(sinogram, src_pos, det_center, det_u_vec, det_v_vec,
                  D=128, H=128, W=128, du=1.0, dv=1.0, voxel_spacing=1.0):
    """Differentiable 3D cone beam backprojection.

    Parameters
    ----------
    sinogram : mx.array
        3D projection data, shape ``(n_views, det_u, det_v)``.
    src_pos : mx.array
        Source positions, shape ``(n_views, 3)``.
    det_center : mx.array
        Detector center positions, shape ``(n_views, 3)``.
    det_u_vec : mx.array
        Detector u-direction unit vectors, shape ``(n_views, 3)``.
    det_v_vec : mx.array
        Detector v-direction unit vectors, shape ``(n_views, 3)``.
    D, H, W : int
        Output volume dimensions (depth, height, width).
    du, dv : float
        Physical spacing between detector elements.
    voxel_spacing : float
        Physical voxel size.

    Returns
    -------
    vol : mx.array
        Shape ``(D, H, W)``.
    """
    return _cone_backward_impl(sinogram, src_pos, det_center, det_u_vec, det_v_vec,
                               D, H, W, du, dv, voxel_spacing)


@cone_backward.vjp
def cone_backward_vjp(primals, cotangent, _):
    sinogram, src_pos, det_center, det_u_vec, det_v_vec = primals[:5]
    du = primals[8] if len(primals) > 8 else 1.0
    dv = primals[9] if len(primals) > 9 else 1.0
    voxel_spacing = primals[10] if len(primals) > 10 else 1.0

    n_views, n_u, n_v = sinogram.shape

    grad_sinogram = _cone_forward_impl(
        cotangent, src_pos, det_center, det_u_vec, det_v_vec,
        n_u, n_v, du, dv, voxel_spacing
    )
    return (grad_sinogram, None, None, None, None, None, None, None, None, None, None)
