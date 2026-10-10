"""PyTorch autograd functions for CT projections.

This module contains PyTorch autograd Function classes that wrap CUDA kernels
for differentiable CT forward projection and backprojection operations.
"""

import torch
import numpy as np

from .constants import _DTYPE
from .kernels import (
    _parallel_2d_forward_kernel,
    _parallel_2d_backward_kernel,
    _fan_2d_forward_kernel,
    _fan_2d_backward_kernel,
    _cone_3d_forward_kernel,
    _cone_3d_backward_kernel,
    _parallel_2d_geometry_vjp_kernel,
    _fan_2d_geometry_vjp_kernel,
    _cone_3d_geometry_vjp_kernel,
)
from .utils import (
    DeviceManager,
    TorchCUDABridge,
    _get_numba_external_stream_for,
    _on_device_of,
    _trig_tables,
    _cuda_context,
    _validate_3d_memory_layout,
    _grid_2d,
    _grid_3d,
)


def _empty_detector_positions(device, beam):
    """Use a rank-correct empty array because CUDA cannot marshal None."""
    shape = (0, 0, 0, 3) if beam == "cone" else (0, 0, 2)
    tensor = torch.empty(shape, dtype=torch.float32, device=device)
    return tensor, TorchCUDABridge.tensor_to_cuda_array(tensor)


def _prepare_volume(beam, volume):
    """Prepare one contiguous native HW/WHD tile for repeated ray launches."""
    volume = volume.detach().to(dtype=torch.float32).contiguous()
    if beam == "cone":
        _validate_3d_memory_layout(volume, expected_order="DHW")
        return volume.permute(2, 1, 0).contiguous()
    return volume


def _launch_ray_kernel(beam, native_volume, sinogram, geometry, detector_spacing,
                       voxel_spacing, *, backproject, after_launch=None):
    """Dispatch the existing matched kernels into caller-owned native buffers."""
    device = native_volume.device
    with _cuda_context(device):
        geom = [component.detach().to(device=device, dtype=torch.float32).contiguous()
                for component in geometry]
        d_geom = [TorchCUDABridge.tensor_to_cuda_array(component) for component in geom]
        if len(geom) == 2:
            d_positions = d_geom[1]
            d_geom = [d_geom[0]] * (4 if beam == "cone" else 3)
        else:
            empty_positions, d_positions = _empty_detector_positions(device, beam)
        d_volume = TorchCUDABridge.tensor_to_cuda_array(native_volume)
        d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
        stream = _get_numba_external_stream_for(torch.cuda.current_stream(device))
        if beam == "cone":
            W, H, D = native_volume.shape
            n_views, n_u, n_v = sinogram.shape
            du, dv = detector_spacing
            grid, tpb = _grid_3d(n_v, n_u, n_views)
            arguments = (
                (d_sino, n_views, n_u, n_v, d_volume, W, H, D)
                if backproject else
                (d_volume, W, H, D, d_sino, n_views, n_u, n_v)
            )
            kernel = _cone_3d_backward_kernel if backproject else _cone_3d_forward_kernel
            kernel[grid, tpb, stream](
                *arguments, _DTYPE(du), _DTYPE(dv), *d_geom,
                _DTYPE(W * 0.5), _DTYPE(H * 0.5), _DTYPE(D * 0.5),
                _DTYPE(voxel_spacing), d_positions,
            )
        else:
            H, W = native_volume.shape
            n_views, n_det = sinogram.shape
            spacing = detector_spacing[0] if isinstance(detector_spacing, (tuple, list)) else detector_spacing
            grid, tpb = _grid_2d(n_views, n_det)
            arguments = (
                (d_sino, n_views, n_det, d_volume, W, H)
                if backproject else
                (d_volume, W, H, d_sino, n_views, n_det)
            )
            if beam == "parallel":
                kernel = _parallel_2d_backward_kernel if backproject else _parallel_2d_forward_kernel
            else:
                kernel = _fan_2d_backward_kernel if backproject else _fan_2d_forward_kernel
            kernel[grid, tpb, stream](
                *arguments, _DTYPE(spacing), *d_geom,
                _DTYPE(W * 0.5), _DTYPE(H * 0.5), _DTYPE(voxel_spacing), d_positions,
            )
        if after_launch is not None:
            after_launch()


def _project_into(beam, prepared_volume, geometry, detector_spacing, voxel_spacing, out, *, after_launch=None):
    """Overwrite every active ray, including misses, in a reusable output view."""
    _launch_ray_kernel(beam, prepared_volume, out, geometry, detector_spacing,
                       voxel_spacing, backproject=False, after_launch=after_launch)
    return out


def _backproject_into(beam, sinogram, geometry, detector_spacing, voxel_spacing, accumulator, *, after_launch=None):
    """Add one ray batch into an initialized contiguous native accumulator."""
    sinogram = sinogram.detach().to(device=accumulator.device, dtype=torch.float32).contiguous()
    _launch_ray_kernel(beam, accumulator, sinogram, geometry, detector_spacing,
                       voxel_spacing, backproject=True, after_launch=after_launch)
    return accumulator


def _surface_project(beam, volume, geometry, detector_shape, detector_spacing, voxel_spacing):
    """Run a native ray projector with one detector endpoint per pixel."""
    with _cuda_context(volume.device):
        prepared = _prepare_volume(beam, volume)
        sino = torch.empty((geometry[0].shape[0], *detector_shape),
                           dtype=torch.float32, device=volume.device)
        return _project_into(beam, prepared, geometry, detector_spacing, voxel_spacing, sino)


def _surface_backproject(beam, sinogram, geometry, volume_shape, detector_spacing, voxel_spacing):
    """Run the matched native adjoint with one detector endpoint per pixel."""
    with _cuda_context(sinogram.device):
        native_shape = volume_shape[::-1] if beam == "cone" else volume_shape
        volume = torch.zeros(native_shape, dtype=torch.float32, device=sinogram.device)
        _backproject_into(beam, sinogram, geometry, detector_spacing, voxel_spacing, volume)
        return volume.permute(2, 1, 0).contiguous() if beam == "cone" else volume


# ============================================================================
# Geometry gradients
# ============================================================================

def _geometry_vjp(beam, volume, grad_sino, geometry, detector_spacing, voxel_spacing,
                  *, prepared_volume=None):
    """Return d<grad_sino, A(geometry) volume>/d(geometry), one tensor per component.

    ``detector_spacing`` is a scalar for 2D beams and ``(du, dv)`` for cone
    beams. Results take the dtype and device of each geometry component.
    """
    vol = _prepare_volume(beam, volume) if prepared_volume is None else prepared_volume
    device = vol.device
    cot = grad_sino.detach().to(device=device, dtype=torch.float32).contiguous()
    geom = [g.detach().to(device=device, dtype=torch.float32).contiguous() for g in geometry]
    grads = [torch.zeros_like(g) for g in geom]
    d_geom = [TorchCUDABridge.tensor_to_cuda_array(g) for g in geom]
    d_grads = [TorchCUDABridge.tensor_to_cuda_array(g) for g in grads]
    if len(geom) == 2:
        d_positions, d_grad_positions = d_geom[1], d_grads[1]
        # Native endpoints bypass the flat center/axes and their gradients.
        n_components = 4 if beam == "cone" else 3
        d_geom = [d_geom[0]] * n_components
        d_grads = [d_grads[0]] * n_components
    else:
        _empty_positions, d_positions = _empty_detector_positions(device, beam)
        d_grad_positions = d_positions
    numba_stream = _get_numba_external_stream_for(torch.cuda.current_stream(device))
    if beam == "cone":
        W, H, D = vol.shape
        n_views, n_u, n_v = cot.shape
        du, dv = detector_spacing
        d_vol = TorchCUDABridge.tensor_to_cuda_array(vol)
        grid, tpb = _grid_3d(n_v, n_u, n_views)
        _cone_3d_geometry_vjp_kernel[grid, tpb, numba_stream](
            d_vol, W, H, D, TorchCUDABridge.tensor_to_cuda_array(cot), n_views, n_u, n_v,
            _DTYPE(du), _DTYPE(dv), *d_geom,
            _DTYPE(W * 0.5), _DTYPE(H * 0.5), _DTYPE(D * 0.5), _DTYPE(voxel_spacing),
            *d_grads, d_positions, d_grad_positions,
        )
    else:
        Ny, Nx = vol.shape
        n_views, n_det = cot.shape
        kernel = _parallel_2d_geometry_vjp_kernel if beam == "parallel" else _fan_2d_geometry_vjp_kernel
        grid, tpb = _grid_2d(n_views, n_det)
        kernel[grid, tpb, numba_stream](
            TorchCUDABridge.tensor_to_cuda_array(vol), Nx, Ny,
            TorchCUDABridge.tensor_to_cuda_array(cot), n_views, n_det,
            _DTYPE(detector_spacing), *d_geom,
            _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5), _DTYPE(voxel_spacing),
            *d_grads, d_positions, d_grad_positions,
        )
    return tuple(grad.to(dtype=g.dtype, device=g.device) for grad, g in zip(grads, geometry))


class _GeometryVJPFunction(torch.autograd.Function):
    """First-order geometry gradient; differentiating it again raises."""

    @staticmethod
    @_on_device_of("volume")
    def forward(ctx, beam, volume, grad_sino, detector_spacing, voxel_spacing, *geometry):
        return _geometry_vjp(beam, volume, grad_sino, geometry, detector_spacing, voxel_spacing)

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise RuntimeError(
            "diffct does not support second derivatives with respect to the geometry"
        )


def _geometry_grads(needs_grad, beam, volume, grad_sino, geometry, detector_spacing, voxel_spacing):
    """Geometry gradients for the components in ``needs_grad``; None for the rest."""
    if not any(needs_grad):
        return (None,) * len(geometry)
    grads = _GeometryVJPFunction.apply(
        beam, volume, grad_sino, detector_spacing, voxel_spacing, *geometry
    )
    return tuple(grad if needed else None for grad, needed in zip(grads, needs_grad))


# ============================================================================
# PyTorch Autograd Functions
# ============================================================================

class ParallelProjectorFunction(torch.autograd.Function):
    """
    Summary
    -------
    PyTorch autograd function for differentiable 2D parallel beam forward projection.

    Notes
    -----
    Provides a differentiable interface to the CUDA-accelerated Siddon ray-tracing
    method with a cell-constant image basis for parallel beam CT geometry. The forward pass computes
    the sinogram from a 2D image using parallel beam geometry. The backward pass
    computes gradients using the adjoint backprojection operation. Requires
    CUDA-capable hardware and a properly configured CUDA environment; all input
    tensors must reside on the same CUDA device.

    Examples
    --------
    >>> import torch
    >>> from diffct import ParallelProjectorFunction, circular_trajectory_2d_parallel
    >>>
    >>> image = torch.randn(128, 128, device='cuda', requires_grad=True)
    >>> ray_dir, det_origin, det_u_vec = circular_trajectory_2d_parallel(180, device='cuda')
    >>> sinogram = ParallelProjectorFunction.apply(image, ray_dir, det_origin, det_u_vec, 192, 1.0)
    >>> sinogram.sum().backward()
    >>> image.grad.shape
    torch.Size([128, 128])
    """
    @staticmethod
    @_on_device_of("image")
    def forward(ctx, image, ray_dir, det_origin, det_u_vec, num_detectors, detector_spacing=1.0, voxel_spacing=1.0):
        """Compute the 2D parallel beam forward projection with arbitrary trajectories using CUDA acceleration.

        Parameters
        ----------
        image : torch.Tensor
            2D input image tensor of shape (H, W), must be on a CUDA device and of type float32.
        ray_dir : torch.Tensor
            Ray direction unit vectors for each view, shape (n_views, 2).
        det_origin : torch.Tensor
            Detector origin positions for each view, shape (n_views, 2), in physical units.
        det_u_vec : torch.Tensor
            Detector u-direction unit vectors for each view, shape (n_views, 2).
        num_detectors : int
            Number of detector elements in the sinogram (columns).
        detector_spacing : float, optional
            Physical spacing between detector elements (default: 1.0).
        voxel_spacing : float, optional
            Physical size of one voxel (in same units as detector_spacing, default: 1.0).

        Returns
        -------
        sinogram : torch.Tensor
            2D tensor of shape (n_views, num_detectors) containing the forward projection (sinogram) on the same device as `image`.

        Notes
        -----
        - All input tensors must be on the same CUDA device.
        - The operation is fully differentiable and supports autograd.
        - Supports arbitrary parallel beam geometries.
        - Uses cell-constant Siddon ray tracing.

        Examples
        --------
        >>> image = torch.randn(128, 128, device='cuda', requires_grad=True)
        >>> ray_dir, det_origin, det_u_vec = circular_trajectory_2d_parallel(180, device='cuda')
        >>> sinogram = ParallelProjectorFunction.apply(
        ...     image, ray_dir, det_origin, det_u_vec, 128, 1.0
        ... )
        """
        # Original inputs: backward builds a differentiable graph through them.
        ctx.save_for_backward(
            image if any(ctx.needs_input_grad[1:4]) else None,
            ray_dir, det_origin, det_u_vec,
        )
        device = DeviceManager.get_device(image)
        image = DeviceManager.ensure_device(image, device)
        ray_dir = DeviceManager.ensure_device(ray_dir, device)
        det_origin = DeviceManager.ensure_device(det_origin, device)
        det_u_vec = DeviceManager.ensure_device(det_u_vec, device)

        # Ensure input is float32 for kernel compatibility
        image = image.to(dtype=torch.float32).contiguous()
        ray_dir = ray_dir.to(dtype=torch.float32).contiguous()
        det_origin = det_origin.to(dtype=torch.float32).contiguous()
        det_u_vec = det_u_vec.to(dtype=torch.float32).contiguous()

        Ny, Nx = image.shape
        n_views = ray_dir.shape[0]

        sinogram = torch.empty((n_views, num_detectors), dtype=image.dtype, device=device)
        _project_into(
            "parallel", _prepare_volume("parallel", image),
            (ray_dir, det_origin, det_u_vec), detector_spacing, voxel_spacing, sinogram,
        )

        ctx.intermediate = (num_detectors, detector_spacing, Ny, Nx, voxel_spacing)
        return sinogram
    
    @staticmethod
    def backward(ctx, grad_sinogram):
        image, ray_dir, det_origin, det_u_vec = ctx.saved_tensors
        num_detectors, detector_spacing, Ny, Nx, voxel_spacing = ctx.intermediate
        grad_image = None
        if ctx.needs_input_grad[0]:
            # The adjoint is an autograd Function too, so second derivatives work.
            grad_image = ParallelBackprojectorFunction.apply(
                grad_sinogram, ray_dir, det_origin, det_u_vec, detector_spacing, Ny, Nx, voxel_spacing
            )
        geometry_grads = _geometry_grads(
            ctx.needs_input_grad[1:4], "parallel", image, grad_sinogram,
            (ray_dir, det_origin, det_u_vec), detector_spacing, voxel_spacing,
        )
        return (grad_image, *geometry_grads, None, None, None)


class ParallelBackprojectorFunction(torch.autograd.Function):
    """
    Summary
    -------
    PyTorch autograd function for differentiable 2D parallel beam backprojection.
    
    Notes
    -----
    Provides a differentiable interface to the CUDA-accelerated Siddon ray-tracing
    method with a cell-constant image basis for parallel beam backprojection. The forward pass computes a 2D
    reconstruction from sinogram data using parallel beam backprojection, and the
    backward pass computes gradients via forward projection as the adjoint operation.
    Requires CUDA-capable hardware and consistent device placements.
    
    
    Examples
    --------
    >>> import torch
    >>> from diffct import ParallelBackprojectorFunction, circular_trajectory_2d_parallel
    >>>
    >>> sinogram = torch.randn(180, 192, device='cuda', requires_grad=True)
    >>> ray_dir, det_origin, det_u_vec = circular_trajectory_2d_parallel(180, device='cuda')
    >>> image = ParallelBackprojectorFunction.apply(sinogram, ray_dir, det_origin, det_u_vec, 1.0, 128, 128)
    >>> image.sum().backward()
    >>> sinogram.grad.shape
    torch.Size([180, 192])
    """
    @staticmethod
    @_on_device_of("sinogram")
    def forward(ctx, sinogram, ray_dir, det_origin, det_u_vec, detector_spacing=1.0, H=128, W=128, voxel_spacing=1.0):
        """Compute the 2D parallel beam backprojection with arbitrary trajectories using CUDA acceleration.

        Parameters
        ----------
        sinogram : torch.Tensor
            2D input sinogram tensor of shape (n_views, num_detectors), must be on a CUDA device and of type float32.
        ray_dir : torch.Tensor
            Ray direction unit vectors for each view, shape (n_views, 2).
        det_origin : torch.Tensor
            Detector origin positions for each view, shape (n_views, 2), in physical units.
        det_u_vec : torch.Tensor
            Detector u-direction unit vectors for each view, shape (n_views, 2).
        detector_spacing : float, optional
            Physical spacing between detector elements (default: 1.0).
        H : int, optional
            Height of the output reconstruction image (default: 128).
        W : int, optional
            Width of the output reconstruction image (default: 128).
        voxel_spacing : float, optional
            Physical size of one voxel (in same units as detector_spacing, default: 1.0).

        Returns
        -------
        reco : torch.Tensor
            2D tensor of shape (H, W) containing the reconstructed image on the same device as `sinogram`.

        Notes
        -----
        - All input tensors must be on the same CUDA device.
        - The operation is fully differentiable and supports autograd.
        - Supports arbitrary parallel beam geometries.
        - Uses the adjoint of cell-constant Siddon ray tracing.

        Examples
        --------
        >>> sinogram = torch.randn(180, 128, device='cuda', requires_grad=True)
        >>> ray_dir, det_origin, det_u_vec = circular_trajectory_2d_parallel(180, device='cuda')
        >>> reco = ParallelBackprojectorFunction.apply(
        ...     sinogram, ray_dir, det_origin, det_u_vec, 1.0, 128, 128
        ... )
        """
        # Original inputs: backward builds a differentiable graph through them.
        ctx.save_for_backward(
            sinogram if any(ctx.needs_input_grad[1:4]) else None,
            ray_dir, det_origin, det_u_vec,
        )
        device = DeviceManager.get_device(sinogram)
        sinogram = DeviceManager.ensure_device(sinogram, device)
        ray_dir = DeviceManager.ensure_device(ray_dir, device)
        det_origin = DeviceManager.ensure_device(det_origin, device)
        det_u_vec = DeviceManager.ensure_device(det_u_vec, device)

        # Ensure input is float32 for kernel compatibility
        sinogram = sinogram.to(dtype=torch.float32).contiguous()
        ray_dir = ray_dir.to(dtype=torch.float32).contiguous()
        det_origin = det_origin.to(dtype=torch.float32).contiguous()
        det_u_vec = det_u_vec.to(dtype=torch.float32).contiguous()

        n_views, n_det = sinogram.shape
        Ny, Nx = H, W

        reco = torch.zeros((Ny, Nx), dtype=sinogram.dtype, device=device)
        _backproject_into(
            "parallel", sinogram, (ray_dir, det_origin, det_u_vec),
            detector_spacing, voxel_spacing, reco,
        )

        ctx.intermediate = (H, W, detector_spacing, sinogram.shape[0], sinogram.shape[1], voxel_spacing)
        return reco

    @staticmethod
    def backward(ctx, grad_output):
        sinogram, ray_dir, det_origin, det_u_vec = ctx.saved_tensors
        H, W, detector_spacing, n_views, n_det, voxel_spacing = ctx.intermediate
        grad_sino = None
        if ctx.needs_input_grad[0]:
            grad_sino = ParallelProjectorFunction.apply(
                grad_output, ray_dir, det_origin, det_u_vec, n_det, detector_spacing, voxel_spacing
            )
        # <grad_output, A^T y> = <A grad_output, y>, so the projector's VJP applies.
        geometry_grads = _geometry_grads(
            ctx.needs_input_grad[1:4], "parallel", grad_output, sinogram,
            (ray_dir, det_origin, det_u_vec), detector_spacing, voxel_spacing,
        )
        return (grad_sino, *geometry_grads, None, None, None, None)


class FanProjectorFunction(torch.autograd.Function):
    """
    Summary
    -------
    PyTorch autograd function for differentiable 2D fan beam forward projection.
    
    Notes
    -----
    Provides a differentiable interface to the CUDA-accelerated Siddon ray-tracing
    method with a cell-constant image basis for fan beam geometry, where rays diverge from a point
    X-ray source to a linear detector array. The forward pass computes sinograms
    using divergent beam geometry, and the backward pass computes gradients via
    adjoint backprojection.
    
    
    Examples
    --------
    >>> import torch
    >>> from diffct import FanProjectorFunction, circular_trajectory_2d_fan
    >>>
    >>> image = torch.randn(256, 256, device='cuda', requires_grad=True)
    >>> src_pos, det_center, det_u_vec = circular_trajectory_2d_fan(360, sid=1000.0, sdd=1500.0, device='cuda')
    >>> sinogram = FanProjectorFunction.apply(image, src_pos, det_center, det_u_vec, 512, 1.0)
    >>> sinogram.sum().backward()
    >>> image.grad.shape
    torch.Size([256, 256])
    """
    @staticmethod
    @_on_device_of("image")
    def forward(ctx, image, src_pos, det_center, det_u_vec, num_detectors, detector_spacing, voxel_spacing=1.0):
        """Compute the 2D fan beam forward projection with arbitrary trajectories using CUDA acceleration.

        Parameters
        ----------
        image : torch.Tensor
            2D input image tensor of shape (H, W), must be on a CUDA device and of type float32.
        src_pos : torch.Tensor
            Source positions for each view, shape (n_views, 2), in physical units.
        det_center : torch.Tensor
            Detector center positions for each view, shape (n_views, 2), in physical units.
        det_u_vec : torch.Tensor
            Detector u-direction unit vectors for each view, shape (n_views, 2).
        num_detectors : int
            Number of detector elements in the sinogram (columns).
        detector_spacing : float
            Physical spacing between detector elements.
        voxel_spacing : float, optional
            Physical size of one voxel (in same units as detector_spacing, default: 1.0).

        Returns
        -------
        sinogram : torch.Tensor
            2D tensor of shape (n_views, num_detectors) containing the fan beam sinogram on the same device as `image`.

        Notes
        -----
        - All input tensors must be on the same CUDA device.
        - The operation is fully differentiable and supports autograd.
        - Supports arbitrary fan beam geometries.
        - Uses cell-constant Siddon ray tracing.

        Examples
        --------
        >>> image = torch.randn(256, 256, device='cuda', requires_grad=True)
        >>> src_pos, det_center, det_u_vec = circular_trajectory_2d_fan(360, 1000.0, 1500.0, device='cuda')
        >>> sinogram = FanProjectorFunction.apply(
        ...     image, src_pos, det_center, det_u_vec, 512, 1.0
        ... )
        """
        # Original inputs: backward builds a differentiable graph through them.
        ctx.save_for_backward(
            image if any(ctx.needs_input_grad[1:4]) else None,
            src_pos, det_center, det_u_vec,
        )
        device = DeviceManager.get_device(image)
        image = DeviceManager.ensure_device(image, device)
        src_pos = DeviceManager.ensure_device(src_pos, device)
        det_center = DeviceManager.ensure_device(det_center, device)
        det_u_vec = DeviceManager.ensure_device(det_u_vec, device)

        image = image.to(dtype=torch.float32).contiguous()
        src_pos = src_pos.to(dtype=torch.float32).contiguous()
        det_center = det_center.to(dtype=torch.float32).contiguous()
        det_u_vec = det_u_vec.to(dtype=torch.float32).contiguous()

        Ny, Nx = image.shape
        n_views = src_pos.shape[0]

        sinogram = torch.empty((n_views, num_detectors), dtype=image.dtype, device=device)
        _project_into(
            "fan", _prepare_volume("fan", image), (src_pos, det_center, det_u_vec),
            detector_spacing, voxel_spacing, sinogram,
        )

        ctx.intermediate = (num_detectors, detector_spacing, Ny, Nx, voxel_spacing)
        return sinogram

    @staticmethod
    def backward(ctx, grad_sinogram):
        image, src_pos, det_center, det_u_vec = ctx.saved_tensors
        num_detectors, detector_spacing, Ny, Nx, voxel_spacing = ctx.intermediate
        grad_image = None
        if ctx.needs_input_grad[0]:
            # The adjoint is an autograd Function too, so second derivatives work.
            grad_image = FanBackprojectorFunction.apply(
                grad_sinogram, src_pos, det_center, det_u_vec, detector_spacing, Ny, Nx, voxel_spacing
            )
        geometry_grads = _geometry_grads(
            ctx.needs_input_grad[1:4], "fan", image, grad_sinogram,
            (src_pos, det_center, det_u_vec), detector_spacing, voxel_spacing,
        )
        return (grad_image, *geometry_grads, None, None, None)


class FanBackprojectorFunction(torch.autograd.Function):
    """
    Summary
    -------
    PyTorch autograd function for differentiable 2D fan beam backprojection.
    
    Notes
    -----
    Provides a differentiable interface to the CUDA-accelerated Siddon ray-tracing
    method with a cell-constant image basis for fan beam backprojection. Implements the adjoint
    of the fan beam projection operator, distributing sinogram values back into
    the reconstruction volume along divergent ray paths. The forward pass
    computes reconstruction from sinogram data, and the backward pass computes
    gradients via forward projection.
    
    
    Examples
    --------
    >>> import torch
    >>> from diffct import FanBackprojectorFunction, circular_trajectory_2d_fan
    >>>
    >>> sinogram = torch.randn(360, 512, device='cuda', requires_grad=True)
    >>> src_pos, det_center, det_u_vec = circular_trajectory_2d_fan(360, sid=1000.0, sdd=1500.0, device='cuda')
    >>> image = FanBackprojectorFunction.apply(sinogram, src_pos, det_center, det_u_vec, 1.0, 256, 256)
    >>> image.sum().backward()
    >>> sinogram.grad.shape
    torch.Size([360, 512])
    """
    @staticmethod
    @_on_device_of("sinogram")
    def forward(ctx, sinogram, src_pos, det_center, det_u_vec, detector_spacing, H, W, voxel_spacing=1.0):
        """Compute the 2D fan beam backprojection with arbitrary trajectories using CUDA acceleration.

        Parameters
        ----------
        sinogram : torch.Tensor
            2D input fan beam sinogram tensor of shape (n_views, num_detectors), must be on a CUDA device and of type float32.
        src_pos : torch.Tensor
            Source positions for each view, shape (n_views, 2), in physical units.
        det_center : torch.Tensor
            Detector center positions for each view, shape (n_views, 2), in physical units.
        det_u_vec : torch.Tensor
            Detector u-direction unit vectors for each view, shape (n_views, 2).
        detector_spacing : float
            Physical spacing between detector elements.
        H : int
            Height of the output reconstruction image.
        W : int
            Width of the output reconstruction image.
        voxel_spacing : float, optional
            Physical size of one voxel (in same units as detector_spacing, default: 1.0).

        Returns
        -------
        reco : torch.Tensor
            2D tensor of shape (H, W) containing the reconstructed image on the same device as `sinogram`.

        Notes
        -----
        - All input tensors must be on the same CUDA device.
        - The operation is fully differentiable and supports autograd.
        - Supports arbitrary fan beam geometries.
        - Uses the adjoint of cell-constant Siddon ray tracing.

        Examples
        --------
        >>> sinogram = torch.randn(360, 512, device='cuda', requires_grad=True)
        >>> src_pos, det_center, det_u_vec = circular_trajectory_2d_fan(360, 1000.0, 1500.0, device='cuda')
        >>> reco = FanBackprojectorFunction.apply(
        ...     sinogram, src_pos, det_center, det_u_vec, 1.0, 256, 256
        ... )
        """
        # Original inputs: backward builds a differentiable graph through them.
        ctx.save_for_backward(
            sinogram if any(ctx.needs_input_grad[1:4]) else None,
            src_pos, det_center, det_u_vec,
        )
        device = DeviceManager.get_device(sinogram)
        sinogram = DeviceManager.ensure_device(sinogram, device)
        src_pos = DeviceManager.ensure_device(src_pos, device)
        det_center = DeviceManager.ensure_device(det_center, device)
        det_u_vec = DeviceManager.ensure_device(det_u_vec, device)

        sinogram = sinogram.to(dtype=torch.float32).contiguous()
        src_pos = src_pos.to(dtype=torch.float32).contiguous()
        det_center = det_center.to(dtype=torch.float32).contiguous()
        det_u_vec = det_u_vec.to(dtype=torch.float32).contiguous()

        n_views, n_det = sinogram.shape
        Ny, Nx = H, W

        reco = torch.zeros((Ny, Nx), dtype=sinogram.dtype, device=device)
        _backproject_into(
            "fan", sinogram, (src_pos, det_center, det_u_vec),
            detector_spacing, voxel_spacing, reco,
        )

        ctx.intermediate = (H, W, detector_spacing, n_views, n_det, voxel_spacing)
        return reco

    @staticmethod
    def backward(ctx, grad_output):
        sinogram, src_pos, det_center, det_u_vec = ctx.saved_tensors
        H, W, detector_spacing, n_views, n_det, voxel_spacing = ctx.intermediate
        grad_sino = None
        if ctx.needs_input_grad[0]:
            grad_sino = FanProjectorFunction.apply(
                grad_output, src_pos, det_center, det_u_vec, n_det, detector_spacing, voxel_spacing
            )
        # <grad_output, A^T y> = <A grad_output, y>, so the projector's VJP applies.
        geometry_grads = _geometry_grads(
            ctx.needs_input_grad[1:4], "fan", grad_output, sinogram,
            (src_pos, det_center, det_u_vec), detector_spacing, voxel_spacing,
        )
        return (grad_sino, *geometry_grads, None, None, None, None)


class ConeProjectorFunction(torch.autograd.Function):
    """
    Summary
    -------
    PyTorch autograd function for differentiable 3D cone beam forward projection.
    
    Notes
    -----
    Provides a differentiable interface to the CUDA-accelerated Siddon ray-tracing
    method with a cell-constant voxel basis for 3D cone beam geometry. Rays emanate from a point
    X-ray source to a 2D detector array capturing volumetric projection data.
    The forward pass computes 3D projections, and the backward pass computes
    gradients via adjoint 3D backprojection. Requires significant GPU memory.
    
    
    Examples
    --------
    >>> import torch
    >>> from diffct import ConeProjectorFunction, circular_trajectory_3d
    >>>
    >>> volume = torch.randn(128, 128, 128, device='cuda', requires_grad=True)
    >>> src_pos, det_center, det_u_vec, det_v_vec = circular_trajectory_3d(360, sid=1000.0, sdd=1500.0, device='cuda')
    >>> projections = ConeProjectorFunction.apply(
    ...     volume, src_pos, det_center, det_u_vec, det_v_vec, 256, 256, 1.0, 1.0
    ... )
    >>> projections.sum().backward()
    >>> volume.grad.shape
    torch.Size([128, 128, 128])
    """
    @staticmethod
    @_on_device_of("volume")
    def forward(ctx, volume, src_pos, det_center, det_u_vec, det_v_vec, det_u, det_v, du, dv, voxel_spacing=1.0):
        """Compute the 3D cone beam forward projection with arbitrary trajectories using CUDA acceleration.

        Parameters
        ----------
        volume : torch.Tensor
            3D input volume tensor of shape (D, H, W), must be on a CUDA device and of type float32.
        src_pos : torch.Tensor
            Source positions for each view, shape (n_views, 3), in physical units.
        det_center : torch.Tensor
            Detector center positions for each view, shape (n_views, 3), in physical units.
        det_u_vec : torch.Tensor
            Detector u-direction unit vectors for each view, shape (n_views, 3).
        det_v_vec : torch.Tensor
            Detector v-direction unit vectors for each view, shape (n_views, 3).
        det_u : int
            Number of detector elements along the u-axis (width).
        det_v : int
            Number of detector elements along the v-axis (height).
        du : float
            Physical spacing between detector elements along the u-axis.
        dv : float
            Physical spacing between detector elements along the v-axis.
        voxel_spacing : float, optional
            Physical size of one voxel (in same units as positions, default: 1.0).

        Returns
        -------
        sino : torch.Tensor
            3D tensor of shape (n_views, det_u, det_v) containing the cone beam projections on the same device as `volume`.

        Notes
        -----
        - All input tensors must be on the same CUDA device.
        - The operation is fully differentiable and supports autograd.
        - Supports arbitrary source and detector trajectories, not limited to circular orbits.
        - Uses cell-constant Siddon ray tracing.

        Examples
        --------
        >>> volume = torch.randn(128, 128, 128, device='cuda', requires_grad=True)
        >>> # Create circular trajectory
        >>> src_pos, det_center, det_u_vec, det_v_vec = circular_trajectory_3d(360, 1000.0, 1500.0, device='cuda')
        >>> sino = ConeProjectorFunction.apply(
        ...     volume, src_pos, det_center, det_u_vec, det_v_vec, 256, 256, 1.0, 1.0
        ... )
        """
        # Original inputs: backward builds a differentiable graph through them.
        ctx.save_for_backward(
            volume if any(ctx.needs_input_grad[1:5]) else None,
            src_pos, det_center, det_u_vec, det_v_vec,
        )
        device = DeviceManager.get_device(volume)
        volume = DeviceManager.ensure_device(volume, device)
        src_pos = DeviceManager.ensure_device(src_pos, device)
        det_center = DeviceManager.ensure_device(det_center, device)
        det_u_vec = DeviceManager.ensure_device(det_u_vec, device)
        det_v_vec = DeviceManager.ensure_device(det_v_vec, device)

        volume = volume.to(dtype=torch.float32).contiguous()
        src_pos = src_pos.to(dtype=torch.float32).contiguous()
        det_center = det_center.to(dtype=torch.float32).contiguous()
        det_u_vec = det_u_vec.to(dtype=torch.float32).contiguous()
        det_v_vec = det_v_vec.to(dtype=torch.float32).contiguous()

        D, H, W = volume.shape
        n_views = src_pos.shape[0]

        sino = torch.empty((n_views, det_u, det_v), dtype=volume.dtype, device=device)
        _project_into(
            "cone", _prepare_volume("cone", volume),
            (src_pos, det_center, det_u_vec, det_v_vec), (du, dv), voxel_spacing, sino,
        )

        ctx.intermediate = (D, H, W, det_u, det_v, du, dv, voxel_spacing)
        return sino

    @staticmethod
    def backward(ctx, grad_sinogram):
        volume, src_pos, det_center, det_u_vec, det_v_vec = ctx.saved_tensors
        (D, H, W, det_u, det_v, du, dv, voxel_spacing) = ctx.intermediate
        grad_vol = None
        if ctx.needs_input_grad[0]:
            # The adjoint is an autograd Function too, so second derivatives work.
            grad_vol = ConeBackprojectorFunction.apply(
                grad_sinogram, src_pos, det_center, det_u_vec, det_v_vec, D, H, W, du, dv, voxel_spacing
            )
        geometry_grads = _geometry_grads(
            ctx.needs_input_grad[1:5], "cone", volume, grad_sinogram,
            (src_pos, det_center, det_u_vec, det_v_vec), (du, dv), voxel_spacing,
        )
        return (grad_vol, *geometry_grads, None, None, None, None, None)


class ConeBackprojectorFunction(torch.autograd.Function):
    """
    Summary
    -------
    PyTorch autograd function for differentiable 3D cone beam backprojection.

    Notes
    -----
    Provides a differentiable interface to the CUDA-accelerated Siddon ray-tracing
    method with a cell-constant voxel basis for 3D cone beam backprojection. The forward pass
    computes a 3D reconstruction from cone beam projection data using
    backprojection as the adjoint operation. The backward pass computes gradients
    via 3D cone beam forward projection. Requires CUDA-capable hardware and
    consistent device placements.
    
    This operation may be memory- and computationally-intensive due to 3D geometry.
    Consider using gradient checkpointing, smaller volumes, or distributed computing
    for large-scale applications, and ensure sufficient GPU memory is available.


    Examples
    --------
    >>> import torch
    >>> from diffct import ConeBackprojectorFunction, circular_trajectory_3d
    >>>
    >>> projections = torch.randn(360, 256, 256, device='cuda', requires_grad=True)
    >>> src_pos, det_center, det_u_vec, det_v_vec = circular_trajectory_3d(360, sid=1000.0, sdd=1500.0, device='cuda')
    >>> volume = ConeBackprojectorFunction.apply(
    ...     projections, src_pos, det_center, det_u_vec, det_v_vec, 128, 128, 128, 1.0, 1.0
    ... )
    >>> volume.sum().backward()
    >>> projections.grad.shape
    torch.Size([360, 256, 256])
    """
    @staticmethod
    @_on_device_of("sinogram")
    def forward(ctx, sinogram, src_pos, det_center, det_u_vec, det_v_vec, D, H, W, du, dv, voxel_spacing=1.0):
        """Compute the 3D cone beam backprojection with arbitrary trajectories using CUDA acceleration.

        Parameters
        ----------
        sinogram : torch.Tensor
            3D input cone beam projection tensor of shape (n_views, det_u, det_v), must be on a CUDA device and of type float32.
        src_pos : torch.Tensor
            Source positions for each view, shape (n_views, 3), in physical units.
        det_center : torch.Tensor
            Detector center positions for each view, shape (n_views, 3), in physical units.
        det_u_vec : torch.Tensor
            Detector u-direction unit vectors for each view, shape (n_views, 3).
        det_v_vec : torch.Tensor
            Detector v-direction unit vectors for each view, shape (n_views, 3).
        D : int
            Depth (z-dimension) of the output reconstruction volume.
        H : int
            Height (y-dimension) of the output reconstruction volume.
        W : int
            Width (x-dimension) of the output reconstruction volume.
        du : float
            Physical spacing between detector elements along the u-axis.
        dv : float
            Physical spacing between detector elements along the v-axis.
        voxel_spacing : float, optional
            Physical size of one voxel (in same units as positions, default: 1.0).

        Returns
        -------
        vol : torch.Tensor
            3D tensor of shape (D, H, W) containing the reconstructed volume on the same device as `sinogram`.

        Notes
        -----
        - All input tensors must be on the same CUDA device.
        - The operation is fully differentiable and supports autograd.
        - Supports arbitrary source and detector trajectories.
        - Uses the adjoint of cell-constant Siddon ray tracing.

        Examples
        --------
        >>> projections = torch.randn(360, 256, 256, device='cuda', requires_grad=True)
        >>> src_pos, det_center, det_u_vec, det_v_vec = circular_trajectory_3d(360, 1000.0, 1500.0, device='cuda')
        >>> vol = ConeBackprojectorFunction.apply(
        ...     projections, src_pos, det_center, det_u_vec, det_v_vec, 128, 128, 128, 1.0, 1.0
        ... )
        """
        # Original inputs: backward builds a differentiable graph through them.
        ctx.save_for_backward(
            sinogram if any(ctx.needs_input_grad[1:5]) else None,
            src_pos, det_center, det_u_vec, det_v_vec,
        )
        device = DeviceManager.get_device(sinogram)
        sinogram = DeviceManager.ensure_device(sinogram, device)
        src_pos = DeviceManager.ensure_device(src_pos, device)
        det_center = DeviceManager.ensure_device(det_center, device)
        det_u_vec = DeviceManager.ensure_device(det_u_vec, device)
        det_v_vec = DeviceManager.ensure_device(det_v_vec, device)

        sinogram = sinogram.to(dtype=torch.float32).contiguous()
        src_pos = src_pos.to(dtype=torch.float32).contiguous()
        det_center = det_center.to(dtype=torch.float32).contiguous()
        det_u_vec = det_u_vec.to(dtype=torch.float32).contiguous()
        det_v_vec = det_v_vec.to(dtype=torch.float32).contiguous()

        n_views, n_u, n_v = sinogram.shape

        # Validate memory layout to prevent coordinate system inconsistencies
        _validate_3d_memory_layout(sinogram, expected_order='VHW')

        vol_perm = torch.zeros((W, H, D), dtype=sinogram.dtype, device=device)
        _backproject_into(
            "cone", sinogram, (src_pos, det_center, det_u_vec, det_v_vec),
            (du, dv), voxel_spacing, vol_perm,
        )

        ctx.intermediate = (D, H, W, n_u, n_v, du, dv, voxel_spacing)
        vol = vol_perm.permute(2, 1, 0).contiguous()
        return vol

    @staticmethod
    def backward(ctx, grad_output):
        sinogram, src_pos, det_center, det_u_vec, det_v_vec = ctx.saved_tensors
        (D, H, W, n_u, n_v, du, dv, voxel_spacing) = ctx.intermediate
        grad_sino = None
        if ctx.needs_input_grad[0]:
            grad_sino = ConeProjectorFunction.apply(
                grad_output, src_pos, det_center, det_u_vec, det_v_vec, n_u, n_v, du, dv, voxel_spacing
            )
        # <grad_output, A^T y> = <A grad_output, y>, so the projector's VJP applies.
        geometry_grads = _geometry_grads(
            ctx.needs_input_grad[1:5], "cone", grad_output, sinogram,
            (src_pos, det_center, det_u_vec, det_v_vec), (du, dv), voxel_spacing,
        )
        return (grad_sino, *geometry_grads, None, None, None, None, None, None)
