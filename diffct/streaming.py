"""Store validation and the bounded staging seam shared by tensor execution."""

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
    if isinstance(source, TensorStore) and isinstance(output, TensorStore):
        first, second = source.tensor, output.tensor
        if first.device == second.device:
            a, b = _tensor_span(first), _tensor_span(second)
            if a is not None and b is not None and max(a[0], b[0]) < min(a[1], b[1]):
                raise ValueError("input and output tensor stores overlap")
    if isinstance(source, NpyStore) and isinstance(output, NpyStore) and source.path == output.path:
        raise ValueError("input and output NpyStore paths alias")
    return source, output
