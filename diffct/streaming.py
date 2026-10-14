"""Store validation and the bounded staging seam shared by tensor execution."""

import numpy as np
import torch

from .storage import TensorStore, NpyStore


def _as_store(value):
    if isinstance(value, torch.Tensor):
        return TensorStore(value)
    if not all(hasattr(value, name) for name in ("shape", "dtype", "read", "write", "flush")):
        raise TypeError("input/output must be a tensor or block store")
    return value


def _tensor_span(tensor):
    if tensor.numel() == 0:
        return None
    start = tensor.data_ptr()
    stop = start + (1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))) * tensor.element_size()
    return start, stop


def _tensor_metadata(tensor, base=None, offset=0):
    if tensor.device.type == "cpu":
        return tensor.detach().numpy()
    # These read-only virtual views are only passed to shares_memory, which
    # examines addresses/strides without dereferencing data. One byte models
    # CUDA address differences; no tensor data is copied or volume allocated.
    if base is None:
        base = np.empty(1, dtype=np.uint8)
    head = np.lib.stride_tricks.as_strided(
        base, shape=(offset + 1,), strides=(1,), writeable=False,
    )[offset:]
    return np.lib.stride_tricks.as_strided(
        head, shape=(*tensor.shape, tensor.element_size()),
        strides=(*(stride * tensor.element_size() for stride in tensor.stride()), 1),
        writeable=False,
    )


def _shares_memory(first, second):
    try:
        # Bound the exact integer-stride solver; reject uncertain exotic
        # layouts rather than risk aliasing or an unbounded metadata search.
        return np.shares_memory(first, second, max_work=1000)
    except getattr(np, "exceptions", np).TooHardError as error:
        raise ValueError("Cannot prove tensor stores do not overlap") from error


def _tensor_internal_overlap(tensor):
    if tensor.numel() == 0:
        return False
    extent = 1
    for stride, size in sorted((stride, size) for size, stride in zip(tensor.shape, tensor.stride())
                               if size > 1):
        if stride == 0:
            return True
        if stride < extent:
            break
        extent += (size - 1) * stride
    else:
        return False

    array = _tensor_metadata(tensor)
    for axis, size in enumerate(tensor.shape):
        if size > 1:
            first, rest = [slice(None)] * array.ndim, [slice(None)] * array.ndim
            first[axis], rest[axis] = slice(0, 1), slice(1, None)
            # Any equal-address pair can be shifted along this axis until
            # one index is zero, preserving the other coordinates and bounds.
            if _shares_memory(array[tuple(first)], array[tuple(rest)]):
                return True
    return False


def _tensor_overlap(first, second):
    if first.device != second.device:
        return False
    a, b = _tensor_span(first), _tensor_span(second)
    if a is None or b is None or max(a[0], b[0]) >= min(a[1], b[1]):
        return False
    base = np.empty(1, dtype=np.uint8)
    origin = min(a[0], b[0])
    return _shares_memory(_tensor_metadata(first, base, a[0] - origin),
                          _tensor_metadata(second, base, b[0] - origin))


def _validate_stores(projector, source, output, is_project):
    if source is output:
        raise ValueError("input and output stores must have distinct backing data")
    if torch.is_grad_enabled():
        for value in (source, output):
            if isinstance(value, torch.Tensor) and value.requires_grad:
                raise ValueError("into methods are numerical; detach autograd inputs and outputs")
    source, output = _as_store(source), _as_store(output)
    expected_input = projector.volume_shape if is_project else projector.projection_shape
    expected_output = projector.projection_shape if is_project else projector.volume_shape
    if tuple(source.shape) != expected_input or tuple(output.shape) != expected_output:
        raise ValueError("input/output store shape does not match the projector")
    if source.dtype not in (torch.float32, torch.float64) or output.dtype != torch.float32:
        raise TypeError("input must be floating point and output dtype must be float32")
    if not getattr(output, "writable", True):
        raise PermissionError("output store is read-only")
    if isinstance(output, TensorStore) and _tensor_internal_overlap(output.tensor):
        raise ValueError("output tensor store has internal overlap")
    if isinstance(source, TensorStore) and isinstance(output, TensorStore):
        if _tensor_overlap(source.tensor, output.tensor):
            raise ValueError("input and output tensor stores overlap")
    if isinstance(source, NpyStore) and isinstance(output, NpyStore) and source.path.samefile(output.path):
        raise ValueError("input and output NpyStore paths alias")
    return source, output
