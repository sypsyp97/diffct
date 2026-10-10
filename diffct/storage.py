"""Numerical block access to caller tensors and mapped NumPy files."""

import numbers
from pathlib import Path

import numpy as np
import torch


_NUMPY_DTYPES = {torch.float32: np.dtype("float32"), torch.float64: np.dtype("float64")}


def _block_index(slices, rank):
    slices = (slices,) if isinstance(slices, slice) else slices
    if not isinstance(slices, (tuple, list)) or not all(isinstance(item, slice) for item in slices):
        raise TypeError("block indices must be slices")
    if len(slices) > rank:
        raise ValueError("block indices exceed the store rank")
    return tuple(slices)


def _write_value(value, shape):
    if not isinstance(value, torch.Tensor):
        raise TypeError("block values must be tensors")
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"block value shape must be {tuple(shape)}")
    return value.detach()


class TensorStore:
    def __init__(self, tensor):
        if not isinstance(tensor, torch.Tensor) or not torch.is_floating_point(tensor):
            raise TypeError("TensorStore requires a floating-point tensor")
        self.tensor = tensor
        self.writable = True

    @property
    def shape(self):
        return tuple(self.tensor.shape)

    @property
    def dtype(self):
        return self.tensor.dtype

    @property
    def device(self):
        return self.tensor.device

    def read(self, slices):
        return self.tensor[_block_index(slices, len(self.shape))].detach()

    def write(self, slices, value, *, accumulate=False):
        with torch.no_grad():
            block = self.tensor[_block_index(slices, len(self.shape))]
            value = _write_value(value, block.shape)
            if accumulate:
                block.add_(value.to(device=self.device, dtype=self.dtype))
            else:
                block.copy_(value)

    def flush(self):
        pass


class NpyStore:
    def __init__(self, path, *, mode="r+"):
        if mode not in ("r", "r+"):
            raise ValueError("NpyStore mode must be 'r' or 'r+'")
        self.path = Path(path).resolve()
        self._array = np.load(self.path, mmap_mode=mode, allow_pickle=False)
        if not isinstance(self._array, np.memmap) or self._array.dtype not in _NUMPY_DTYPES.values():
            raise TypeError("NpyStore requires a mapped float32 or float64 .npy file")

    @property
    def shape(self):
        return tuple(self._array.shape)

    @property
    def dtype(self):
        return next(dtype for dtype, numpy_dtype in _NUMPY_DTYPES.items()
                    if self._array.dtype == numpy_dtype)

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def writable(self):
        return self._array.flags.writeable

    def read(self, slices):
        block = self._array[_block_index(slices, len(self.shape))]
        return torch.from_numpy(block.copy())

    def write(self, slices, value, *, accumulate=False):
        if not self.writable:
            raise PermissionError("cannot write a read-only NpyStore")
        block = self._array[_block_index(slices, len(self.shape))]
        value = _write_value(value, block.shape).to(device="cpu", dtype=self.dtype).numpy()
        if accumulate:
            block += value
        else:
            block[...] = value

    def flush(self):
        self._array.flush()

    @classmethod
    def create(cls, path, shape, *, dtype=torch.float32):
        if dtype not in _NUMPY_DTYPES:
            raise TypeError("NpyStore dtype must be float32 or float64")
        shape = tuple(shape)
        if any(isinstance(size, bool) or not isinstance(size, numbers.Integral) or size < 0
               for size in shape):
            raise ValueError("NpyStore shape must contain nonnegative integer dimensions")
        path = Path(path).resolve()
        # Reserve the new path exclusively before NumPy writes its header.
        with path.open("xb"):
            pass
        store = cls.__new__(cls)
        store.path = path
        store._array = np.lib.format.open_memmap(
            path, mode="w+", dtype=_NUMPY_DTYPES[dtype], shape=shape,
        )
        return store
