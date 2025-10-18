"""PyTorch autograd functions for CT projections.

This module contains PyTorch autograd Function classes that wrap CUDA kernels
for differentiable CT forward projection and backprojection operations.
"""

import torch
import numpy as np

from .constants import _DTYPE
from .utils import (
    DeviceManager,
    TorchCUDABridge,
    _get_numba_external_stream_for,
    _trig_tables,
    _validate_3d_memory_layout,
    _grid_2d,
    _grid_3d,
)
from .kernels import (
    _parallel_2d_forward_kernel,
    _parallel_2d_backward_kernel,
    _fan_2d_forward_kernel,
    _fan_2d_backward_kernel,
    _cone_3d_forward_kernel,
    _cone_3d_backward_kernel,
)


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
    method with interpolation for parallel beam CT geometry. The forward pass computes
    the sinogram from a 2D image using parallel beam geometry. The backward pass
    computes gradients using the adjoint backprojection operation. Requires
    CUDA-capable hardware and a properly configured CUDA environment; all input
    tensors must reside on the same CUDA device.

    Examples
    --------
    >>> import torch
    >>> from diffct.differentiable import ParallelProjectorFunction
    >>>
    >>> # Create a 2D image with gradient tracking
    >>> image = torch.randn(128, 128, device='cuda', requires_grad=True)
    >>> # Define projection parameters
    >>> angles = torch.linspace(0, torch.pi, 180, device='cuda')
    >>> num_detectors = 128
    >>> detector_spacing = 1.0
    >>> # Compute forward projection
    >>> projector = ParallelProjectorFunction.apply
    >>> sinogram = projector(image, angles, num_detectors, detector_spacing)
    >>> # Compute loss and gradients
    >>> loss = sinogram.sum()
    >>> loss.backward()
    >>> print(f"Gradient shape: {image.grad.shape}")  # (128, 128)
    """
    @staticmethod
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
        - Uses the Siddon method with interpolation for accurate ray tracing and bilinear interpolation.

        Examples
        --------
        >>> image = torch.randn(128, 128, device='cuda', requires_grad=True)
        >>> ray_dir, det_origin, det_u_vec = circular_trajectory_2d_parallel(180, device='cuda')
        >>> sinogram = ParallelProjectorFunction.apply(
        ...     image, ray_dir, det_origin, det_u_vec, 128, 1.0
        ... )
        """
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

        # Allocate output tensor on the same device
        sinogram = torch.zeros((n_views, num_detectors), dtype=image.dtype, device=device)

        # Get Numba CUDA array views for kernel
        d_image = TorchCUDABridge.tensor_to_cuda_array(image)
        d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
        d_ray_dir_arr = TorchCUDABridge.tensor_to_cuda_array(ray_dir)
        d_det_origin_arr = TorchCUDABridge.tensor_to_cuda_array(det_origin)
        d_det_u_vec_arr = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)

        grid, tpb = _grid_2d(n_views, num_detectors)
        cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)

        pt_stream = torch.cuda.current_stream()
        numba_stream = _get_numba_external_stream_for(pt_stream)
        _parallel_2d_forward_kernel[grid, tpb, numba_stream](
            d_image, Nx, Ny, d_sino, n_views, num_detectors,
            _DTYPE(detector_spacing), d_ray_dir_arr, d_det_origin_arr, d_det_u_vec_arr, cx, cy, _DTYPE(voxel_spacing)
        )

        ctx.save_for_backward(ray_dir, det_origin, det_u_vec)
        ctx.intermediate = (num_detectors, detector_spacing, Ny, Nx, voxel_spacing)
        return sinogram
    
    @staticmethod
    def backward(ctx, grad_sinogram):
        ray_dir, det_origin, det_u_vec = ctx.saved_tensors
        num_detectors, detector_spacing, Ny, Nx, voxel_spacing = ctx.intermediate
        device = DeviceManager.get_device(grad_sinogram)
        grad_sinogram = DeviceManager.ensure_device(grad_sinogram, device)
        ray_dir = DeviceManager.ensure_device(ray_dir, device)
        det_origin = DeviceManager.ensure_device(det_origin, device)
        det_u_vec = DeviceManager.ensure_device(det_u_vec, device)

        grad_sinogram = grad_sinogram.to(dtype=torch.float32).contiguous()
        ray_dir = ray_dir.to(dtype=torch.float32).contiguous()
        det_origin = det_origin.to(dtype=torch.float32).contiguous()
        det_u_vec = det_u_vec.to(dtype=torch.float32).contiguous()

        n_views = ray_dir.shape[0]
        grad_image = torch.zeros((Ny, Nx), dtype=grad_sinogram.dtype, device=device)

        d_grad_sino = TorchCUDABridge.tensor_to_cuda_array(grad_sinogram)
        d_img_grad = TorchCUDABridge.tensor_to_cuda_array(grad_image)
        d_ray_dir_arr = TorchCUDABridge.tensor_to_cuda_array(ray_dir)
        d_det_origin_arr = TorchCUDABridge.tensor_to_cuda_array(det_origin)
        d_det_u_vec_arr = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)

        grid, tpb = _grid_2d(n_views, num_detectors)
        cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)

        pt_stream = torch.cuda.current_stream()
        numba_stream = _get_numba_external_stream_for(pt_stream)
        _parallel_2d_backward_kernel[grid, tpb, numba_stream](
            d_grad_sino, n_views, num_detectors,
            d_img_grad, Nx, Ny,
            _DTYPE(detector_spacing), d_ray_dir_arr, d_det_origin_arr, d_det_u_vec_arr, cx, cy, _DTYPE(voxel_spacing)
        )

        return grad_image, None, None, None, None, None, None


class ParallelBackprojectorFunction(torch.autograd.Function):
    """
    Summary
    -------
    PyTorch autograd function for differentiable 2D parallel beam backprojection.
    
    Notes
    -----
    Provides a differentiable interface to the CUDA-accelerated Siddon ray-tracing
    method with interpolation for parallel beam backprojection. The forward pass computes a 2D
    reconstruction from sinogram data using parallel beam backprojection, and the
    backward pass computes gradients via forward projection as the adjoint operation.
    Requires CUDA-capable hardware and consistent device placements.
    
    
    Examples
    --------
    >>> import torch
    >>> from diffct.differentiable import ParallelBackprojectorFunction
    >>>
    >>> sinogram = torch.randn(180, 128, device='cuda', requires_grad=True)
    >>> angles = torch.linspace(0, torch.pi, 180, device='cuda')
    >>> recon = ParallelBackprojectorFunction.apply(sinogram, angles, 1.0, 128, 128)
    >>> loss = recon.sum()
    >>> loss.backward()
    >>> print(sinogram.grad.shape)  # (180, 128)
    """
    @staticmethod
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
        - Uses the Siddon method with interpolation for accurate ray tracing and bilinear interpolation.

        Examples
        --------
        >>> sinogram = torch.randn(180, 128, device='cuda', requires_grad=True)
        >>> ray_dir, det_origin, det_u_vec = circular_trajectory_2d_parallel(180, device='cuda')
        >>> reco = ParallelBackprojectorFunction.apply(
        ...     sinogram, ray_dir, det_origin, det_u_vec, 1.0, 128, 128
        ... )
        """
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

        # Allocate output tensor on the same device
        reco = torch.zeros((Ny, Nx), dtype=sinogram.dtype, device=device)

        # Get Numba CUDA array views for kernel
        d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
        d_reco = TorchCUDABridge.tensor_to_cuda_array(reco)
        d_ray_dir_arr = TorchCUDABridge.tensor_to_cuda_array(ray_dir)
        d_det_origin_arr = TorchCUDABridge.tensor_to_cuda_array(det_origin)
        d_det_u_vec_arr = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)

        grid, tpb = _grid_2d(n_views, n_det)
        cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)

        pt_stream = torch.cuda.current_stream()
        numba_stream = _get_numba_external_stream_for(pt_stream)
        _parallel_2d_backward_kernel[grid, tpb, numba_stream](
            d_sino, n_views, n_det, d_reco, Nx, Ny,
            _DTYPE(detector_spacing), d_ray_dir_arr, d_det_origin_arr, d_det_u_vec_arr, cx, cy, _DTYPE(voxel_spacing)
        )

        ctx.save_for_backward(ray_dir, det_origin, det_u_vec)
        ctx.intermediate = (H, W, detector_spacing, sinogram.shape[0], sinogram.shape[1], voxel_spacing)
        return reco

    @staticmethod
    def backward(ctx, grad_output):
        ray_dir, det_origin, det_u_vec = ctx.saved_tensors
        H, W, detector_spacing, n_views, n_det, voxel_spacing = ctx.intermediate
        device = DeviceManager.get_device(grad_output)
        grad_output = DeviceManager.ensure_device(grad_output, device)
        ray_dir = DeviceManager.ensure_device(ray_dir, device)
        det_origin = DeviceManager.ensure_device(det_origin, device)
        det_u_vec = DeviceManager.ensure_device(det_u_vec, device)

        grad_output = grad_output.to(dtype=torch.float32).contiguous()
        ray_dir = ray_dir.to(dtype=torch.float32).contiguous()
        det_origin = det_origin.to(dtype=torch.float32).contiguous()
        det_u_vec = det_u_vec.to(dtype=torch.float32).contiguous()

        Ny, Nx = grad_output.shape

        # Allocate output tensor on the same device
        grad_sino = torch.zeros((n_views, n_det), dtype=grad_output.dtype, device=device)

        # Get Numba CUDA array views for kernel
        d_grad_out = TorchCUDABridge.tensor_to_cuda_array(grad_output)
        d_sino_grad = TorchCUDABridge.tensor_to_cuda_array(grad_sino)
        d_ray_dir_arr = TorchCUDABridge.tensor_to_cuda_array(ray_dir)
        d_det_origin_arr = TorchCUDABridge.tensor_to_cuda_array(det_origin)
        d_det_u_vec_arr = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)

        grid, tpb = _grid_2d(n_views, n_det)
        cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)

        pt_stream = torch.cuda.current_stream()
        numba_stream = _get_numba_external_stream_for(pt_stream)
        _parallel_2d_forward_kernel[grid, tpb, numba_stream](
            d_grad_out, Nx, Ny, d_sino_grad, n_views, n_det,
            _DTYPE(detector_spacing), d_ray_dir_arr, d_det_origin_arr, d_det_u_vec_arr, cx, cy, _DTYPE(voxel_spacing)
        )

        return grad_sino, None, None, None, None, None, None, None


class FanProjectorFunction(torch.autograd.Function):
    """
    Summary
    -------
    PyTorch autograd function for differentiable 2D fan beam forward projection.
    
    Notes
    -----
    Provides a differentiable interface to the CUDA-accelerated Siddon ray-tracing
    method with interpolation for fan beam geometry, where rays diverge from a point
    X-ray source to a linear detector array. The forward pass computes sinograms
    using divergent beam geometry, and the backward pass computes gradients via
    adjoint backprojection.
    
    
    Examples
    --------
    >>> import torch
    >>> from diffct.differentiable import FanProjectorFunction
    >>>
    >>> image = torch.randn(256, 256, device='cuda', requires_grad=True)
    >>> angles = torch.linspace(0, 2 * torch.pi, 360, device='cuda')
    >>> sinogram = FanProjectorFunction.apply(image, angles, 512, 1.0, 1500.0, 1000.0)
    >>> loss = sinogram.sum()
    >>> loss.backward()
    >>> print(image.grad.shape)  # (256, 256)
    """
    @staticmethod
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
        - Uses the Siddon method with interpolation for accurate ray tracing and bilinear interpolation.

        Examples
        --------
        >>> image = torch.randn(256, 256, device='cuda', requires_grad=True)
        >>> src_pos, det_center, det_u_vec = circular_trajectory_2d_fan(360, 1000.0, 1500.0, device='cuda')
        >>> sinogram = FanProjectorFunction.apply(
        ...     image, src_pos, det_center, det_u_vec, 512, 1.0
        ... )
        """
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

        sinogram = torch.zeros((n_views, num_detectors), dtype=image.dtype, device=device)

        d_image = TorchCUDABridge.tensor_to_cuda_array(image)
        d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
        d_src_pos_arr = TorchCUDABridge.tensor_to_cuda_array(src_pos)
        d_det_center_arr = TorchCUDABridge.tensor_to_cuda_array(det_center)
        d_det_u_vec_arr = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)

        grid, tpb = _grid_2d(n_views, num_detectors)
        cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)

        pt_stream = torch.cuda.current_stream()
        numba_stream = _get_numba_external_stream_for(pt_stream)
        _fan_2d_forward_kernel[grid, tpb, numba_stream](
            d_image, Nx, Ny, d_sino, n_views, num_detectors,
            _DTYPE(detector_spacing), d_src_pos_arr, d_det_center_arr, d_det_u_vec_arr,
            cx, cy, _DTYPE(voxel_spacing)
        )

        ctx.save_for_backward(src_pos, det_center, det_u_vec)
        ctx.intermediate = (num_detectors, detector_spacing, Ny, Nx, voxel_spacing)
        return sinogram

    @staticmethod
    def backward(ctx, grad_sinogram):
        src_pos, det_center, det_u_vec = ctx.saved_tensors
        (n_det, det_spacing, Ny, Nx, voxel_spacing) = ctx.intermediate
        device = DeviceManager.get_device(grad_sinogram)
        grad_sinogram = DeviceManager.ensure_device(grad_sinogram, device)
        src_pos = DeviceManager.ensure_device(src_pos, device)
        det_center = DeviceManager.ensure_device(det_center, device)
        det_u_vec = DeviceManager.ensure_device(det_u_vec, device)

        grad_sinogram = grad_sinogram.to(dtype=torch.float32).contiguous()
        src_pos = src_pos.to(dtype=torch.float32).contiguous()
        det_center = det_center.to(dtype=torch.float32).contiguous()
        det_u_vec = det_u_vec.to(dtype=torch.float32).contiguous()

        n_views = src_pos.shape[0]
        grad_img = torch.zeros((Ny, Nx), dtype=grad_sinogram.dtype, device=device)

        d_grad_sino = TorchCUDABridge.tensor_to_cuda_array(grad_sinogram)
        d_img_grad = TorchCUDABridge.tensor_to_cuda_array(grad_img)
        d_src_pos_arr = TorchCUDABridge.tensor_to_cuda_array(src_pos)
        d_det_center_arr = TorchCUDABridge.tensor_to_cuda_array(det_center)
        d_det_u_vec_arr = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)

        grid, tpb = _grid_2d(n_views, n_det)
        cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)

        pt_stream = torch.cuda.current_stream()
        numba_stream = _get_numba_external_stream_for(pt_stream)
        _fan_2d_backward_kernel[grid, tpb, numba_stream](
            d_grad_sino, n_views, n_det, d_img_grad, Nx, Ny,
            _DTYPE(det_spacing), d_src_pos_arr, d_det_center_arr, d_det_u_vec_arr,
            cx, cy, _DTYPE(voxel_spacing)
        )

        return grad_img, None, None, None, None, None, None


class FanBackprojectorFunction(torch.autograd.Function):
    """
    Summary
    -------
    PyTorch autograd function for differentiable 2D fan beam backprojection.
    
    Notes
    -----
    Provides a differentiable interface to the CUDA-accelerated Siddon ray-tracing
    method with interpolation for fan beam backprojection. Implements the adjoint
    of the fan beam projection operator, distributing sinogram values back into
    the reconstruction volume along divergent ray paths. The forward pass
    computes reconstruction from sinogram data, and the backward pass computes
    gradients via forward projection.
    
    
    Examples
    --------
    >>> import torch
    >>> from diffct.differentiable import FanBackprojectorFunction
    >>>
    >>> sinogram = torch.randn(360, 512, device='cuda', requires_grad=True)
    >>> angles = torch.linspace(0, 2 * torch.pi, 360, device='cuda')
    >>> recon = FanBackprojectorFunction.apply(sinogram, angles, 1.0, 256, 256, 1500.0, 1000.0)
    >>> loss = recon.sum()
    >>> loss.backward()
    >>> print(sinogram.grad.shape)  # (360, 512)
    """
    @staticmethod
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
        - Uses the Siddon method with interpolation for accurate ray tracing and bilinear interpolation.

        Examples
        --------
        >>> sinogram = torch.randn(360, 512, device='cuda', requires_grad=True)
        >>> src_pos, det_center, det_u_vec = circular_trajectory_2d_fan(360, 1000.0, 1500.0, device='cuda')
        >>> reco = FanBackprojectorFunction.apply(
        ...     sinogram, src_pos, det_center, det_u_vec, 1.0, 256, 256
        ... )
        """
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

        d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
        d_reco = TorchCUDABridge.tensor_to_cuda_array(reco)
        d_src_pos_arr = TorchCUDABridge.tensor_to_cuda_array(src_pos)
        d_det_center_arr = TorchCUDABridge.tensor_to_cuda_array(det_center)
        d_det_u_vec_arr = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)

        grid, tpb = _grid_2d(n_views, n_det)
        cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)

        pt_stream = torch.cuda.current_stream()
        numba_stream = _get_numba_external_stream_for(pt_stream)
        _fan_2d_backward_kernel[grid, tpb, numba_stream](
            d_sino, n_views, n_det, d_reco, Nx, Ny,
            _DTYPE(detector_spacing), d_src_pos_arr, d_det_center_arr, d_det_u_vec_arr,
            cx, cy, _DTYPE(voxel_spacing)
        )

        ctx.save_for_backward(src_pos, det_center, det_u_vec)
        ctx.intermediate = (H, W, detector_spacing, n_views, n_det, voxel_spacing)
        return reco

    @staticmethod
    def backward(ctx, grad_output):
        src_pos, det_center, det_u_vec = ctx.saved_tensors
        (H, W, det_spacing, n_views, n_det, voxel_spacing) = ctx.intermediate
        device = DeviceManager.get_device(grad_output)
        grad_output = DeviceManager.ensure_device(grad_output, device)
        src_pos = DeviceManager.ensure_device(src_pos, device)
        det_center = DeviceManager.ensure_device(det_center, device)
        det_u_vec = DeviceManager.ensure_device(det_u_vec, device)

        grad_output = grad_output.to(dtype=torch.float32).contiguous()
        src_pos = src_pos.to(dtype=torch.float32).contiguous()
        det_center = det_center.to(dtype=torch.float32).contiguous()
        det_u_vec = det_u_vec.to(dtype=torch.float32).contiguous()

        Ny, Nx = grad_output.shape

        grad_sino = torch.zeros((n_views, n_det), dtype=grad_output.dtype, device=device)

        d_grad_out = TorchCUDABridge.tensor_to_cuda_array(grad_output)
        d_sino_grad = TorchCUDABridge.tensor_to_cuda_array(grad_sino)
        d_src_pos_arr = TorchCUDABridge.tensor_to_cuda_array(src_pos)
        d_det_center_arr = TorchCUDABridge.tensor_to_cuda_array(det_center)
        d_det_u_vec_arr = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)

        grid, tpb = _grid_2d(n_views, n_det)
        cx, cy = _DTYPE(Nx * 0.5), _DTYPE(Ny * 0.5)

        pt_stream = torch.cuda.current_stream()
        numba_stream = _get_numba_external_stream_for(pt_stream)
        _fan_2d_forward_kernel[grid, tpb, numba_stream](
            d_grad_out, Nx, Ny, d_sino_grad, n_views, n_det,
            _DTYPE(det_spacing), d_src_pos_arr, d_det_center_arr, d_det_u_vec_arr,
            cx, cy, _DTYPE(voxel_spacing)
        )

        return grad_sino, None, None, None, None, None, None, None


class ConeProjectorFunction(torch.autograd.Function):
    """
    Summary
    -------
    PyTorch autograd function for differentiable 3D cone beam forward projection.
    
    Notes
    -----
    Provides a differentiable interface to the CUDA-accelerated Siddon ray-tracing
    method with interpolation for 3D cone beam geometry. Rays emanate from a point
    X-ray source to a 2D detector array capturing volumetric projection data.
    The forward pass computes 3D projections, and the backward pass computes
    gradients via adjoint 3D backprojection. Requires significant GPU memory.
    
    
    Examples
    --------
    >>> import torch
    >>> from diffct.differentiable import ConeProjectorFunction
    >>>
    >>> volume = torch.randn(128, 128, 128, device='cuda', requires_grad=True)
    >>> angles = torch.linspace(0, 2 * torch.pi, 360, device='cuda')
    >>> projections = ConeProjectorFunction.apply(volume, angles, 256, 256, 1.0, 1.0, 1500.0, 1000.0)
    >>> loss = projections.sum()
    >>> loss.backward()
    >>> print(volume.grad.shape)  # (128, 128, 128)
    """
    @staticmethod
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
        - Uses the Siddon method with trilinear interpolation for accurate 3D ray tracing.

        Examples
        --------
        >>> volume = torch.randn(128, 128, 128, device='cuda', requires_grad=True)
        >>> # Create circular trajectory
        >>> src_pos, det_center, det_u_vec, det_v_vec = circular_trajectory_3d(360, 1000.0, 1500.0, device='cuda')
        >>> sino = ConeProjectorFunction.apply(
        ...     volume, src_pos, det_center, det_u_vec, det_v_vec, 256, 256, 1.0, 1.0
        ... )
        """
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

        # Validate memory layout to prevent coordinate system inconsistencies
        _validate_3d_memory_layout(volume, expected_order='DHW')

        sino = torch.zeros((n_views, det_u, det_v), dtype=volume.dtype, device=device)

        volume_perm = volume.permute(2, 1, 0).contiguous()
        d_vol = TorchCUDABridge.tensor_to_cuda_array(volume_perm)
        d_sino = TorchCUDABridge.tensor_to_cuda_array(sino)
        d_src_pos_arr = TorchCUDABridge.tensor_to_cuda_array(src_pos)
        d_det_center_arr = TorchCUDABridge.tensor_to_cuda_array(det_center)
        d_det_u_vec_arr = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)
        d_det_v_vec_arr = TorchCUDABridge.tensor_to_cuda_array(det_v_vec)

        grid, tpb = _grid_3d(n_views, det_u, det_v)
        cx, cy, cz = _DTYPE(W * 0.5), _DTYPE(H * 0.5), _DTYPE(D * 0.5)

        pt_stream = torch.cuda.current_stream()
        numba_stream = _get_numba_external_stream_for(pt_stream)
        _cone_3d_forward_kernel[grid, tpb, numba_stream](
            d_vol, W, H, D, d_sino, n_views, det_u, det_v,
            _DTYPE(du), _DTYPE(dv), d_src_pos_arr, d_det_center_arr, d_det_u_vec_arr, d_det_v_vec_arr,
            cx, cy, cz, _DTYPE(voxel_spacing)
        )

        ctx.save_for_backward(src_pos, det_center, det_u_vec, det_v_vec)
        ctx.intermediate = (D, H, W, det_u, det_v, du, dv, voxel_spacing)
        return sino

    @staticmethod
    def backward(ctx, grad_sinogram):
        src_pos, det_center, det_u_vec, det_v_vec = ctx.saved_tensors
        (D, H, W, det_u, det_v, du, dv, voxel_spacing) = ctx.intermediate
        device = DeviceManager.get_device(grad_sinogram)
        grad_sinogram = DeviceManager.ensure_device(grad_sinogram, device)
        src_pos = DeviceManager.ensure_device(src_pos, device)
        det_center = DeviceManager.ensure_device(det_center, device)
        det_u_vec = DeviceManager.ensure_device(det_u_vec, device)
        det_v_vec = DeviceManager.ensure_device(det_v_vec, device)

        grad_sinogram = grad_sinogram.to(dtype=torch.float32).contiguous()
        src_pos = src_pos.to(dtype=torch.float32).contiguous()
        det_center = det_center.to(dtype=torch.float32).contiguous()
        det_u_vec = det_u_vec.to(dtype=torch.float32).contiguous()
        det_v_vec = det_v_vec.to(dtype=torch.float32).contiguous()

        n_views = src_pos.shape[0]

        grad_vol_perm = torch.zeros((W, H, D), dtype=grad_sinogram.dtype, device=device)

        d_grad_sino = TorchCUDABridge.tensor_to_cuda_array(grad_sinogram)
        d_vol_grad = TorchCUDABridge.tensor_to_cuda_array(grad_vol_perm)
        d_src_pos_arr = TorchCUDABridge.tensor_to_cuda_array(src_pos)
        d_det_center_arr = TorchCUDABridge.tensor_to_cuda_array(det_center)
        d_det_u_vec_arr = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)
        d_det_v_vec_arr = TorchCUDABridge.tensor_to_cuda_array(det_v_vec)

        grid, tpb = _grid_3d(n_views, det_u, det_v)
        cx, cy, cz = _DTYPE(W * 0.5), _DTYPE(H * 0.5), _DTYPE(D * 0.5)

        pt_stream = torch.cuda.current_stream()
        numba_stream = _get_numba_external_stream_for(pt_stream)
        _cone_3d_backward_kernel[grid, tpb, numba_stream](
            d_grad_sino, n_views, det_u, det_v, d_vol_grad, W, H, D,
            _DTYPE(du), _DTYPE(dv), d_src_pos_arr, d_det_center_arr, d_det_u_vec_arr, d_det_v_vec_arr,
            cx, cy, cz, _DTYPE(voxel_spacing)
        )

        grad_vol = grad_vol_perm.permute(2, 1, 0).contiguous()
        return grad_vol, None, None, None, None, None, None, None, None, None


class ConeBackprojectorFunction(torch.autograd.Function):
    """
    Summary
    -------
    PyTorch autograd function for differentiable 3D cone beam backprojection.

    Notes
    -----
    Provides a differentiable interface to the CUDA-accelerated Siddon ray-tracing
    method with interpolation for 3D cone beam backprojection. The forward pass
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
    >>> from diffct.differentiable import ConeBackprojectorFunction
    >>>
    >>> projections = torch.randn(360, 256, 256, device='cuda', requires_grad=True)
    >>> angles = torch.linspace(0, 2 * torch.pi, 360, device='cuda')
    >>> D, H, W = 128, 128, 128
    >>> du, dv = 1.0, 1.0
    >>> sdd, sid = 1500.0, 1000.0
    >>> backprojector = ConeBackprojectorFunction.apply
    >>> volume = backprojector(projections, angles, D, H, W, du, dv, sdd, sid)
    >>> loss = volume.sum()
    >>> loss.backward()
    >>> print(f"Projection gradient shape: {projections.grad.shape}")  # (360, 256, 256)
    """
    @staticmethod
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
        - Uses the Siddon method with trilinear interpolation for accurate 3D ray tracing.

        Examples
        --------
        >>> projections = torch.randn(360, 256, 256, device='cuda', requires_grad=True)
        >>> src_pos, det_center, det_u_vec, det_v_vec = circular_trajectory_3d(360, 1000.0, 1500.0, device='cuda')
        >>> vol = ConeBackprojectorFunction.apply(
        ...     projections, src_pos, det_center, det_u_vec, det_v_vec, 128, 128, 128, 1.0, 1.0
        ... )
        """
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

        d_sino = TorchCUDABridge.tensor_to_cuda_array(sinogram)
        d_reco = TorchCUDABridge.tensor_to_cuda_array(vol_perm)
        d_src_pos_arr = TorchCUDABridge.tensor_to_cuda_array(src_pos)
        d_det_center_arr = TorchCUDABridge.tensor_to_cuda_array(det_center)
        d_det_u_vec_arr = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)
        d_det_v_vec_arr = TorchCUDABridge.tensor_to_cuda_array(det_v_vec)

        grid, tpb = _grid_3d(n_views, n_u, n_v)
        cx, cy, cz = _DTYPE(W * 0.5), _DTYPE(H * 0.5), _DTYPE(D * 0.5)

        pt_stream = torch.cuda.current_stream()
        numba_stream = _get_numba_external_stream_for(pt_stream)
        _cone_3d_backward_kernel[grid, tpb, numba_stream](
            d_sino, n_views, n_u, n_v, d_reco, W, H, D,
            _DTYPE(du), _DTYPE(dv), d_src_pos_arr, d_det_center_arr, d_det_u_vec_arr, d_det_v_vec_arr,
            cx, cy, cz, _DTYPE(voxel_spacing)
        )

        ctx.save_for_backward(src_pos, det_center, det_u_vec, det_v_vec)
        ctx.intermediate = (D, H, W, n_u, n_v, du, dv, voxel_spacing)
        vol = vol_perm.permute(2, 1, 0).contiguous()
        return vol

    @staticmethod
    def backward(ctx, grad_output):
        src_pos, det_center, det_u_vec, det_v_vec = ctx.saved_tensors
        (D, H, W, n_u, n_v, du, dv, voxel_spacing) = ctx.intermediate
        device = DeviceManager.get_device(grad_output)
        grad_output = DeviceManager.ensure_device(grad_output, device)
        src_pos = DeviceManager.ensure_device(src_pos, device)
        det_center = DeviceManager.ensure_device(det_center, device)
        det_u_vec = DeviceManager.ensure_device(det_u_vec, device)
        det_v_vec = DeviceManager.ensure_device(det_v_vec, device)

        grad_output = grad_output.to(dtype=torch.float32).contiguous()
        src_pos = src_pos.to(dtype=torch.float32).contiguous()
        det_center = det_center.to(dtype=torch.float32).contiguous()
        det_u_vec = det_u_vec.to(dtype=torch.float32).contiguous()
        det_v_vec = det_v_vec.to(dtype=torch.float32).contiguous()

        n_views = src_pos.shape[0]

        grad_sino = torch.zeros((n_views, n_u, n_v), dtype=grad_output.dtype, device=device)

        grad_output_perm = grad_output.permute(2, 1, 0).contiguous()
        d_grad_out = TorchCUDABridge.tensor_to_cuda_array(grad_output_perm)
        d_sino_grad = TorchCUDABridge.tensor_to_cuda_array(grad_sino)
        d_src_pos_arr = TorchCUDABridge.tensor_to_cuda_array(src_pos)
        d_det_center_arr = TorchCUDABridge.tensor_to_cuda_array(det_center)
        d_det_u_vec_arr = TorchCUDABridge.tensor_to_cuda_array(det_u_vec)
        d_det_v_vec_arr = TorchCUDABridge.tensor_to_cuda_array(det_v_vec)

        grid, tpb = _grid_3d(n_views, n_u, n_v)
        cx, cy, cz = _DTYPE(W * 0.5), _DTYPE(H * 0.5), _DTYPE(D * 0.5)

        pt_stream = torch.cuda.current_stream()
        numba_stream = _get_numba_external_stream_for(pt_stream)
        _cone_3d_forward_kernel[grid, tpb, numba_stream](
            d_grad_out, W, H, D, d_sino_grad, n_views, n_u, n_v,
            _DTYPE(du), _DTYPE(dv), d_src_pos_arr, d_det_center_arr, d_det_u_vec_arr, d_det_v_vec_arr,
            cx, cy, cz, _DTYPE(voxel_spacing)
        )

        return grad_sino, None, None, None, None, None, None, None, None, None, None


# ------------------------------------------------------------------
# HELPER FUNCTIONS FOR ARBITRARY TRAJECTORIES
# ------------------------------------------------------------------
