"""Utility classes and helper functions for DiffCT package.

This module provides utility classes and functions for device management,
PyTorch-CUDA bridging, stream caching, trigonometric table generation,
memory layout validation, and CUDA grid computation.
"""

import math
import numpy as np
import torch
from numba import cuda

from .constants import _DTYPE, _TPB_2D, _TPB_3D, _INF, _EPSILON


# ============================================================================
# Device Management Utilities
# ============================================================================

class DeviceManager:
    """Utilities for managing PyTorch tensor devices."""

    @staticmethod
    def get_device(tensor):
        """Get the device of a PyTorch tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor whose device to determine.

        Returns
        -------
        torch.device
            Device of the tensor or CPU if unavailable.

        Examples
        --------
        >>> DeviceManager.get_device(torch.tensor([1, 2, 3]))
        device(type='cpu')
        """
        return tensor.device if hasattr(tensor, "device") else torch.device("cpu")

    @staticmethod
    def ensure_device(tensor, device):
        """Ensure a tensor resides on a given device.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to move.
        device : torch.device
            Desired device.

        Returns
        -------
        torch.Tensor
            Tensor on the specified device. Unchanged if already on it.

        Examples
        --------
        >>> DeviceManager.ensure_device(
        ...     torch.tensor([1, 2, 3]),
        ...     torch.device('cuda')
        ... )
        tensor([1, 2, 3], device='cuda:0')
        """
        if hasattr(tensor, "to") and tensor.device != device:
            return tensor.to(device)
        return tensor


# ============================================================================
# PyTorch-CUDA Bridge
# ============================================================================

class TorchCUDABridge:
    """Bridge between PyTorch tensors and Numba CUDA arrays."""

    @staticmethod
    def tensor_to_cuda_array(tensor):
        """Convert a PyTorch CUDA tensor to a Numba CUDA DeviceNDArray.

        Provides a zero-copy view of a detached PyTorch tensor as a Numba CUDA array,
        avoiding CPU data transfers. The returned array shares memory with the
        original tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            PyTorch tensor on a CUDA device.

        Returns
        -------
        numba.cuda.cudadrv.devicearray.DeviceNDArray
            Numba CUDA array view sharing memory with `tensor`.

        Raises
        ------
        ValueError
            If `tensor` is not on a CUDA device.

        Examples
        --------
        >>> t = torch.randn(10, device='cuda')
        >>> arr = TorchCUDABridge.tensor_to_cuda_array(t)
        """
        if not tensor.is_cuda:
            raise ValueError("Tensor must be on CUDA device")
        return cuda.as_cuda_array(tensor.detach())


# ============================================================================
# Stream Management (cached external Numba stream)
# ============================================================================

_cached_stream_ptr = None
_cached_numba_stream = None


def _get_numba_external_stream_for(pt_stream=None):
    """Return a cached numba.cuda.external_stream for the current PyTorch CUDA stream.

    Caches by the underlying CUDA stream pointer to avoid repeated construction.

    Parameters
    ----------
    pt_stream : torch.cuda.Stream, optional
        PyTorch CUDA stream. If None, uses current stream.

    Returns
    -------
    numba.cuda.cudadrv.driver.Stream
        Numba external stream wrapper around PyTorch CUDA stream.
    """
    global _cached_stream_ptr, _cached_numba_stream
    if pt_stream is None:
        pt_stream = torch.cuda.current_stream()
    # Torch exposes an underlying CUDA stream handle via .cuda_stream
    ptr = int(pt_stream.cuda_stream)
    if _cached_stream_ptr == ptr and _cached_numba_stream is not None:
        return _cached_numba_stream
    numba_stream = cuda.external_stream(pt_stream.cuda_stream)
    _cached_stream_ptr = ptr
    _cached_numba_stream = numba_stream
    return numba_stream


# ============================================================================
# GPU-aware Trigonometric Table Generation
# ============================================================================

def _trig_tables(angles, dtype=_DTYPE, device=None):
    """Compute cosine and sine tables for input angles.

    Precompute cosine and sine values and return as torch tensors on the
    same device as `angles`.

    Parameters
    ----------
    angles : array-like or torch.Tensor
        Projection angles in radians. Can be a NumPy array or a PyTorch tensor on CPU or CUDA.
    dtype : numpy.dtype or torch.dtype, optional
        Desired data type for output tables. Default is `_DTYPE`.
    device : torch.device, optional
        Target device for output tensors. If None, uses the device of `angles`.

    Returns
    -------
    cos : torch.Tensor
        Cosine values of `angles` on the same device.
    sin : torch.Tensor
        Sine values of `angles` on the same device.

    Examples
    --------
    >>> angles = torch.linspace(0, torch.pi, 180, device='cuda')
    >>> cos, sin = _trig_tables(angles)
    >>> cos.device
    device(type='cuda', index=0)
    """
    if isinstance(angles, torch.Tensor):
        device = angles.device if device is None else device
        # Compute both cos and sin in one call to avoid redundant kernel launches
        angles_device = angles.to(dtype=dtype, device=device)
        cos = torch.cos(angles_device)
        sin = torch.sin(angles_device)
        return cos, sin
    else:
        # fallback for non-tensor inputs: compute via PyTorch on CPU for consistency
        # Determine desired torch dtype
        if isinstance(dtype, torch.dtype):
            torch_dtype = dtype
        else:
            _NP_TO_TORCH = {
                np.float32: torch.float32,
                np.float64: torch.float64,
            }
            torch_dtype = _NP_TO_TORCH.get(dtype, torch.float32)
        # Convert input angles to a CPU torch tensor and compute both simultaneously
        angles_cpu = torch.tensor(angles, dtype=torch_dtype)
        cos_cpu = torch.cos(angles_cpu)
        sin_cpu = torch.sin(angles_cpu)
        if device is not None:
            return cos_cpu.to(device), sin_cpu.to(device)
        else:
            return cos_cpu, sin_cpu


# ============================================================================
# Memory Layout Validation
# ============================================================================

def _validate_3d_memory_layout(tensor, expected_order='DHW'):
    """Validate 3D tensor memory layout to prevent coordinate system inconsistencies.

    Parameters
    ----------
    tensor : torch.Tensor
        3D tensor to validate
    expected_order : str, optional
        Expected memory order ('DHW', 'VHW', etc.). Default is 'DHW'.

    Raises
    ------
    ValueError
        If tensor has unexpected memory layout or is non-contiguous
    """
    shape = tensor.shape
    if len(shape) != 3:
        raise ValueError(f"Expected 3D tensor, got {len(shape)}D")

    # Early return for common case - contiguous tensor with expected ordering
    if tensor.is_contiguous() and expected_order in ('DHW', 'VHW'):
        # For DHW and VHW, the expected order matches memory layout when contiguous
        return

    # Only check memory order for DHW and VHW, not for internal WHD layout
    if expected_order in ('DHW', 'VHW'):
        if not tensor.is_contiguous():
            raise ValueError(
                "Input tensor must be contiguous. Call .contiguous() before passing to "
                "cone beam functions to avoid memory duplication and ensure correct results."
            )

        strides = tensor.stride()
        order_mapping = {
            'DHW': (0, 1, 2),  # Depth, Height, Width
            'VHW': (0, 1, 2),  # Views, Height, Width (for sinograms)
        }
        if expected_order not in order_mapping:
            raise ValueError(f"Unsupported expected_order: {expected_order}")

        expected_stride_order = order_mapping[expected_order]
        # Check if actual strides match expected order
        sorted_strides = sorted(enumerate(strides), key=lambda x: x[1], reverse=True)
        actual_order = tuple(idx for idx, _ in sorted_strides)

        if actual_order != expected_stride_order:
            # Create appropriate error message based on context
            if expected_order == 'VHW':
                actual_str = f"({shape[0]}, {shape[1]}, {shape[2]})"
                expected_str = "(Views, Height, Width)"
                fix_str = "ensure your sinogram has shape (num_views, det_v, det_u)"
            elif expected_order == 'DHW':
                actual_str = f"({shape[0]}, {shape[1]}, {shape[2]})"
                expected_str = "(Depth, Height, Width)"
                fix_str = "ensure your volume has shape (D, H, W)"
            else:
                actual_str = str(tuple(shape))
                expected_str = expected_order
                fix_str = "check tensor dimensions"

            raise ValueError(
                f"Memory layout mismatch: expected {expected_str} order, "
                f"but tensor has shape {actual_str}. Please {fix_str} and ensure "
                f"the tensor is contiguous (.contiguous()) before passing to the function."
            )
    # For 'WHD' (internal layout), skip stride check entirely


# ============================================================================
# CUDA Grid Computation
# ============================================================================

def _grid_2d(n1, n2, tpb=_TPB_2D):
    """Compute 2D CUDA grid and block dimensions.

    Determine optimal grid and block sizes for 2D CUDA ray-tracing kernels.

    Parameters
    ----------
    n1 : int
        Number of elements along the first dimension (e.g., projection angles).
    n2 : int
        Number of elements along the second dimension (e.g., detector elements).
    tpb : tuple of int, optional
        Threads per block (default is `_TPB_2D`) to balance occupancy and memory.

    Returns
    -------
    grid : tuple of int
        Blocks count per axis.
    tpb : tuple of int
        Threads per block per axis.

    Examples
    --------
    >>> grid, tpb = _grid_2d(180, 256)
    >>> grid
    (12, 16)
    >>> tpb
    (16, 16)
    """
    return (math.ceil(n1 / tpb[0]), math.ceil(n2 / tpb[1])), tpb


def _grid_3d(n1, n2, n3, tpb=_TPB_3D):
    """Compute 3D CUDA grid and block dimensions.

    Determine optimal grid and block sizes for 3D CUDA cone-beam kernels.

    Parameters
    ----------
    n1 : int
        Number of elements along the first dimension (e.g., projection views).
    n2 : int
        Number of elements along the second dimension (e.g., detector u-axis).
    n3 : int
        Number of elements along the third dimension (e.g., detector v-axis).
    tpb : tuple of int, optional
        Threads per block (default is `_TPB_3D`) to balance occupancy and registers.

    Returns
    -------
    grid : tuple of int
        Blocks count per axis.
    tpb : tuple of int
        Threads per block per axis.

    Examples
    --------
    >>> grid, tpb = _grid_3d(360, 256, 256)
    >>> grid
    (45, 32, 32)
    >>> tpb
    (8, 8, 8)
    """
    return (
        math.ceil(n1 / tpb[0]),
        math.ceil(n2 / tpb[1]),
        math.ceil(n3 / tpb[2]),
    ), tpb
