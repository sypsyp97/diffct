"""Numerical block stores: real files, guarded block I/O and native kernels."""

from contextlib import ExitStack, contextmanager
from datetime import timedelta
from functools import lru_cache
import math
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

import diffct
from tests.test_chunked_projector import _Launches, _offsets
from tests.test_detector_surfaces import _case, _close, _data, _matrix, _numpy, _projector


BEAMS = ("parallel", "fan", "cone")
cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _api(name):
    value = getattr(diffct, name, None)
    assert value is not None, f"required declared export diffct.{name} is missing"
    return value


def _slices(index, shape):
    index = (index,) if isinstance(index, slice) else tuple(index)
    assert len(index) <= len(shape) and all(isinstance(item, slice) for item in index)
    index += (slice(None),) * (len(shape) - len(index))
    return index


class _GuardedStore:
    """Structural store backed by real caller-owned tensor data."""
    def __init__(self, tensor, block_shape, *, writable=True):
        self.tensor = tensor
        self.shape, self.dtype = tuple(tensor.shape), tensor.dtype
        self.block_shape, self.writable = block_shape, writable
        self.reads, self.writes = [], []
        self.overwrites = torch.zeros(self.shape, dtype=torch.int32)
        # Deliberately equal opaque metadata. Only known NpyStore paths are aliases.
        self.path = "opaque-backend-identifier"

    def _selection(self, index):
        index = _slices(index, self.shape)
        selected = self.tensor[index]
        assert all(actual <= limit for actual, limit in zip(selected.shape, self.block_shape)), \
            ("unbounded structural-store I/O", selected.shape, self.block_shape)
        return index, selected

    def read(self, index):
        index, selected = self._selection(index)
        self.reads.append(index)
        return selected.detach().clone()

    def write(self, index, value, *, accumulate=False):
        assert self.writable, "read-only structural store was written"
        index, selected = self._selection(index)
        assert tuple(value.shape) == tuple(selected.shape)
        self.writes.append((index, accumulate))
        if not accumulate:
            self.overwrites[index] += 1
        with torch.no_grad():
            selected.add_(value) if accumulate else selected.copy_(value)

    def flush(self):
        pass


class _BlockAllocations(TorchDispatchMode):
    """Reject fresh complete input/output work arrays; permit caller-owned views."""
    def __init__(self, shapes, maximum):
        super().__init__()
        self.shapes, self.maximum = set(map(tuple, shapes)), maximum

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        inputs = {(value.device, value.untyped_storage()._cdata)
                  for value in tree_flatten(args)[0] if isinstance(value, torch.Tensor)}
        result = function(*args, **(kwargs or {}))
        for value in tree_flatten(result)[0]:
            if not isinstance(value, torch.Tensor):
                continue
            fresh = (value.device, value.untyped_storage()._cdata) not in inputs
            if fresh and tuple(value.shape) in self.shapes:
                assert value.numel() <= self.maximum, ("complete tensor work array", str(function), value.shape)
        return result


class _ObservedMemmap(np.memmap):
    def __array_finalize__(self, original):
        super().__array_finalize__(original)
        self.block_limit = getattr(original, "block_limit", None)

    def _bounded(self):
        if self.block_limit is not None:
            assert self.size <= self.block_limit, ("whole mmap cast/copy/accumulate", self.shape)

    def astype(self, *args, **kwargs):
        self._bounded()
        return super().astype(*args, **kwargs)

    def copy(self, *args, **kwargs):
        self._bounded()
        return super().copy(*args, **kwargs)

    def __iadd__(self, value):
        self._bounded()
        return super().__iadd__(value)


@contextmanager
def _numpy_blocks(maximum):
    """Observe actual NumPy allocations/mappings; always forward their operations."""
    with ExitStack() as patches:
        for name in ("empty", "zeros", "ones", "full", "array", "asarray", "ascontiguousarray", "copy", "fromfile"):
            original = getattr(np, name)

            def bounded(*args, _original=original, _name=name, **kwargs):
                if _name in ("empty", "zeros", "ones", "full"):
                    shape = args[0] if args else kwargs["shape"]
                    elements = math.prod(shape) if isinstance(shape, (tuple, list)) else int(shape)
                    assert elements <= maximum, ("whole NumPy allocation", _name, shape)
                result = _original(*args, **kwargs)
                if isinstance(result, np.ndarray) and result.size > maximum and not isinstance(result, np.memmap):
                    source = args[0] if args else None
                    assert isinstance(source, np.ndarray) and np.shares_memory(result, source), \
                        ("whole NumPy copy/cast", _name, result.shape)
                return result
            patches.enter_context(mock.patch.object(np, name, bounded))
        for module, name in ((np, "load"), (np.lib.format, "open_memmap")):
            original = getattr(module, name)

            def mapping(*args, _original=original, **kwargs):
                result = _original(*args, **kwargs)
                assert isinstance(result, np.memmap), "existing .npy was loaded into a complete RAM array"
                result = result.view(_ObservedMemmap)
                result.block_limit = maximum
                return result
            patches.enter_context(mock.patch.object(module, name, mapping))
        yield


@contextmanager
def _rejection(pattern):
    with pytest.raises((TypeError, ValueError, RuntimeError, PermissionError), match=pattern) as error:
        yield
    assert not isinstance(error.value, NotImplementedError), "declared capability is still unsupported"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tensor_store_rectangles_detached_reads_numerical_writes_and_original_backing(dtype):
    original = torch.arange(7 * 9, dtype=dtype).reshape(7, 9).requires_grad_()
    store = _api("TensorStore")(original)
    assert tuple(store.shape) == (7, 9) and store.dtype == dtype
    index = (slice(1, 4), slice(2, 6))
    expected = original.detach()[index].clone()
    actual = store.read(index)
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad and actual.device == original.device and actual.dtype == dtype
    value = torch.full((3, 4), -.7, dtype=dtype, requires_grad=True)
    store.write(index, value)
    store.write(index, value, accumulate=True)
    store.flush()
    torch.testing.assert_close(original.detach()[index], value.detach() * 2)
    assert original.grad is None and value.grad is None


@pytest.mark.cuda
@cuda_required
def test_tensor_store_preserves_cuda_device_and_uses_only_requested_rectangle():
    original = torch.arange(7 * 9, device="cuda:0", dtype=torch.float64).reshape(7, 9)
    with _BlockAllocations((original.shape,), 12):
        store = _api("TensorStore")(original)
        actual = store.read((slice(1, 4), slice(2, 6)))
        store.write((slice(1, 4), slice(2, 6)), torch.ones((3, 4), device=original.device))
    assert actual.device == original.device and actual.dtype == original.dtype
    assert torch.all(original[1:4, 2:6] == 1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_npy_store_create_read_cast_accumulate_flush_and_reopen_are_block_bounded(tmp_path, dtype):
    shape, block = (17, 19, 23), (3, 4, 5)
    path = tmp_path / "volume.npy"
    with _numpy_blocks(math.prod(block)), _BlockAllocations((shape,), math.prod(block)):
        store = _api("NpyStore").create(path, shape, dtype=dtype)
        assert tuple(store.shape) == shape and store.dtype == dtype
        index = (slice(2, 5), slice(3, 7), slice(4, 9))
        values = torch.linspace(-.4, .9, math.prod(block), dtype=torch.float64).reshape(block)
        store.write(index, values)
        store.write(index, values / 2, accumulate=True)
        actual = store.read(index)
        assert actual.device.type == "cpu" and actual.dtype == dtype and not actual.requires_grad
        torch.testing.assert_close(actual.double(), values * 1.5, rtol=2e-6, atol=2e-7)
        store.flush()
        reopened = _api("NpyStore")(path, mode="r")
        torch.testing.assert_close(reopened.read(index), actual)
    mmap = np.load(path, mmap_mode="r")
    assert isinstance(mmap, np.memmap)
    np.testing.assert_allclose(mmap[index], (values * 1.5).numpy(), rtol=2e-6, atol=2e-7)


def test_npy_store_existing_path_rejected_without_overwrite_and_readonly_write_fails(tmp_path):
    path = tmp_path / "existing.npy"
    values = np.arange(7 * 9, dtype=np.float32).reshape(7, 9)
    np.save(path, values)
    before = path.read_bytes()
    with pytest.raises((FileExistsError, ValueError)):
        _api("NpyStore").create(path, values.shape)
    assert path.read_bytes() == before
    readonly = _api("NpyStore")(path, mode="r")
    with _rejection("read.only|writ|mode"):
        readonly.write((slice(1, 3), slice(2, 5)), torch.ones(2, 3))
    assert path.read_bytes() == before


def test_npy_store_create_and_reopen_empty_rank_shard(tmp_path):
    path, shape = tmp_path / "empty-rays.npy", (0, 3, 4)
    with _numpy_blocks(1), _BlockAllocations((shape,), 1):
        store = _api("NpyStore").create(path, shape)
        assert tuple(store.shape) == shape and store.dtype == torch.float32
        block = store.read((slice(0, 0), slice(None), slice(None)))
        assert tuple(block.shape) == shape and block.numel() == 0
        store.write((slice(0, 0), slice(None), slice(None)), block)
        store.flush()
        reopened = _api("NpyStore")(path, mode="r")
        assert tuple(reopened.read((slice(0, 0), slice(None), slice(None))).shape) == shape
    assert tuple(np.load(path, mmap_mode="r").shape) == shape


def _execution_case(beam):
    case = _case(beam, views=5)
    case.shape = (5, 7, 9) if beam == "cone" else (7, 9)
    chunk = (3, 4, 5) if beam == "cone" else (4, 5)
    pixels = (4, 3) if beam == "cone" else (3,)
    return case, chunk, pixels


@lru_cache(maxsize=1)
def _warm_native_point_kernels():
    """Compile legacy point signatures before measuring bounded data memory.

    Numba's first import constructs fixed Unicode tables with NumPy. Those
    compiler tables are outside the projector's runtime block allocation.
    Fresh tiny fixtures keep warmup separate from each tested store/output.
    """
    for beam in BEAMS:
        case = _case(beam, views=1)
        case.shape = (2, 3, 4) if beam == "cone" else (3, 4)
        case.detector = (2, 3) if beam == "cone" else (3,)

        def legacy(u, v):
            return torch.stack((u, v, torch.zeros_like(u)), dim=-1)

        projector = _projector(case, legacy, volume_chunk_shape=case.shape, view_chunk_size=1)
        image, sino = _data(case, device="cpu")
        projector.project(image)
        projector.backproject(sino)


def _backend(kind, tensor, block_shape, path, *, output=False):
    if kind == "tensor":
        return tensor
    if kind == "tensor-store":
        return _api("TensorStore")(tensor)
    if kind == "guarded":
        return _GuardedStore(tensor, block_shape)
    np.save(path, _numpy(tensor).astype(np.float32 if output else np.float64))
    return _api("NpyStore")(path, mode="r+" if output else "r")


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("backend", ["tensor", "tensor-store", "npy", "guarded"])
def test_into_writes_only_blocks_returns_supplied_output_and_matches_cell_oracle(beam, operation, backend, tmp_path):
    _warm_native_point_kernels()
    case, chunk, pixels = _execution_case(beam)
    projector = _projector(case, volume_chunk_shape=chunk, view_chunk_size=2, detector_chunk_shape=pixels)
    image, sino = _data(case, device="cpu")
    input_tensor = (image if operation == "project" else sino).double()
    output_shape = case.sino_shape if operation == "project" else case.shape
    output_tensor = torch.full(output_shape, torch.nan)
    input_limit, output_limit = (chunk, (2, *pixels)) if operation == "project" else ((2, *pixels), chunk)
    source = _backend(backend, input_tensor, input_limit, tmp_path / "input.npy")
    output = _backend(backend, output_tensor, output_limit, tmp_path / "output.npy", output=True)
    maximum = max(math.prod(chunk), 2 * math.prod(pixels) * len(chunk))
    with _BlockAllocations((case.shape, case.sino_shape), maximum), _numpy_blocks(maximum), _Launches() as launches:
        returned = getattr(projector, f"{operation}_into")(source, output)
    assert returned is output
    matrix = _matrix(case, _offsets(case, flat=True))
    operator = matrix if operation == "project" else matrix.T
    expected = (operator @ _numpy(input_tensor).ravel()).reshape(output_shape)
    actual = (torch.tensor(np.load(tmp_path / "output.npy", mmap_mode="r")) if backend == "npy" else
              output.tensor if backend == "guarded" else output_tensor)
    _close(actual, expected)
    assert actual.dtype == torch.float32 and actual.device.type == "cpu"
    assert launches.records and {record[1] for record in launches.records} == {beam}
    assert {record[2] for record in launches.records} == {"forward" if operation == "project" else "backward"}
    for record in launches.records:
        assert 0 < record[-1][0] <= 2 and all(a <= b for a, b in zip(record[-1][1:], pixels))
    if backend == "guarded":
        assert source.reads and output.writes
        assert torch.all(output.overwrites == 1), "each output cell must overwrite once before partial sums"


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_into_dense_cuda_inputs_and_outputs_preserve_caller_device(operation):
    case, chunk, pixels = _execution_case("cone")
    projector = _projector(case, volume_chunk_shape=chunk, view_chunk_size=2, detector_chunk_shape=pixels)
    image, sino = _data(case)
    source = image if operation == "project" else sino
    output = torch.full(case.sino_shape if operation == "project" else case.shape, torch.nan, device=source.device)
    with _Launches() as launches:
        returned = getattr(projector, f"{operation}_into")(source, output)
    matrix = _matrix(case, _offsets(case, flat=True))
    operator = matrix if operation == "project" else matrix.T
    _close(output, (operator @ _numpy(source).ravel()).reshape(output.shape))
    assert returned is output and returned.device == source.device
    assert launches.records


@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("kind", ["shape", "dtype", "input-grad", "output-grad", "readonly-opaque"])
def test_into_rejects_invalid_output_or_autograd_before_reads_writes_or_kernels(operation, kind):
    case, chunk, pixels = _execution_case("fan")
    projector = _projector(case, volume_chunk_shape=chunk, view_chunk_size=2)
    image, sino = _data(case, device="cpu")
    source = image if operation == "project" else sino
    output = torch.zeros(case.sino_shape if operation == "project" else case.shape)
    pattern = "shape" if kind == "shape" else "float32|dtype" if kind == "dtype" else "read.only|writ" if kind == "readonly-opaque" else "grad|numerical|detach"
    if kind == "shape":
        output = output[:-1]
    elif kind == "dtype":
        output = output.double()
    elif kind == "input-grad":
        source.requires_grad_()
    elif kind == "output-grad":
        output.requires_grad_()
    else:
        output = _GuardedStore(output, chunk if operation == "backproject" else (2, *pixels), writable=False)
    with _Launches() as launches, _rejection(pattern):
        getattr(projector, f"{operation}_into")(source, output)
    assert not launches.records
    if isinstance(output, _GuardedStore):
        assert not output.reads and not output.writes


@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("kind", ["same-object", "overlapping-tensor-stores", "same-npy-path"])
def test_into_rejects_known_aliases_before_execution(operation, kind, tmp_path):
    case = _case("parallel", views=5)
    case.shape, case.detector = (5, 7), (7,)
    projector = _projector(case, volume_chunk_shape=(3, 4), view_chunk_size=2)
    tensor = torch.linspace(-.4, .7, 35).reshape(5, 7)
    if kind == "same-object":
        source = output = _GuardedStore(tensor, (3, 4))
    elif kind == "overlapping-tensor-stores":
        base = torch.arange(70, dtype=torch.float32)
        source = _api("TensorStore")(base[:35].reshape(5, 7))
        output = _api("TensorStore")(base[17:52].reshape(5, 7))
    else:
        path = tmp_path / "same.npy"
        np.save(path, tensor.numpy())
        (tmp_path / "sub").mkdir()
        source = _api("NpyStore")(path, mode="r+")
        output = _api("NpyStore")(tmp_path / "sub" / ".." / "same.npy", mode="r+")
    with _Launches() as launches, _rejection("alias|overlap|same|distinct|backing"):
        getattr(projector, f"{operation}_into")(source, output)
    assert not launches.records


@pytest.mark.cuda
@cuda_required
def test_nonoverlapping_tensor_store_views_of_same_allocation_are_valid():
    case = _case("parallel", views=5)
    case.shape, case.detector = (5, 7), (7,)
    projector = _projector(case, volume_chunk_shape=(3, 4), view_chunk_size=2)
    base = torch.zeros(70)
    base[:35] = torch.linspace(-.4, .7, 35)
    source_tensor, output_tensor = base[:35].reshape(5, 7), base[35:].reshape(5, 7)
    original = source_tensor.clone()
    source, output = _api("TensorStore")(source_tensor), _api("TensorStore")(output_tensor)
    assert projector.project_into(source, output) is output
    matrix = _matrix(case, _offsets(case, flat=True))
    _close(output_tensor, (matrix @ original.numpy().ravel()).reshape(5, 7))
    torch.testing.assert_close(source_tensor, original)


def test_npy_readonly_output_rejected_before_input_store_reads(tmp_path):
    case, chunk, pixels = _execution_case("cone")
    projector = _projector(case, volume_chunk_shape=chunk, view_chunk_size=2)
    image, sino = _data(case, device="cpu")
    path = tmp_path / "readonly.npy"
    np.save(path, sino.numpy())
    source = _GuardedStore(image, chunk)
    output = _api("NpyStore")(path, mode="r")
    with _Launches() as launches, _rejection("read.only|writ|mode"):
        projector.project_into(source, output)
    assert not source.reads and not launches.records


def _empty_rank_worker(rank, beam, rendezvous, directory):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        case = _case(beam, views=1)
        chunk = (3, 4, 5) if beam == "cone" else (4, 5)
        pixels = (4, 3) if beam == "cone" else (3,)
        projector = _projector(case, distributed=True, volume_chunk_shape=chunk,
                               view_chunk_size=2, detector_chunk_shape=pixels)
        image, sino = _data(case, device="cpu")
        selected = projector.view_slice
        local_sino = sino[selected]
        output = _api("NpyStore").create(Path(directory) / f"rays-{beam}-{rank}.npy", tuple(local_sino.shape))
        projector.project_into(_api("TensorStore")(image), output)
        output.flush()
        matrix = _matrix(case, _offsets(case, flat=True))
        expected = (matrix @ image.numpy().ravel()).reshape(case.sino_shape)[selected]
        _close(output.read((slice(None),) * len(local_sino.shape)), expected)
        back = _api("NpyStore").create(Path(directory) / f"volume-{beam}-{rank}.npy", case.shape)
        projector.backproject_into(_api("TensorStore")(local_sino), back)
        back.flush()
        _close(back.read((slice(None),) * len(case.shape)),
               (matrix.T @ sino.numpy().ravel()).reshape(case.shape))
    finally:
        dist.destroy_process_group()


@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2 or not dist.is_nccl_available(),
                    reason="Two CUDA GPUs and real NCCL are required")
@pytest.mark.parametrize("beam", BEAMS)
def test_empty_rank_block_outputs_and_bounded_backprojection_sum(beam, tmp_path):
    mp.spawn(_empty_rank_worker, args=(beam, (tmp_path / "block-nccl").resolve().as_uri(), str(tmp_path)),
             nprocs=2, join=True)


def _independently_wrapped_cpu_alias_case(operation, overlap):
    case = _case("parallel", views=3)
    case.shape, case.detector = (3, 5), (7,)
    projector = _projector(case, volume_chunk_shape=(2, 3), view_chunk_size=2)
    input_shape, output_shape = ((case.shape, case.sino_shape) if operation == "project" else
                                 (case.sino_shape, case.shape))
    input_count, output_count = math.prod(input_shape), math.prod(output_shape)
    start = 3 if overlap else input_count
    backing = np.arange(input_count + output_count, dtype=np.float32)
    source = torch.from_numpy(backing[:input_count]).reshape(input_shape)
    output = torch.from_numpy(backing[start:start + output_count]).reshape(output_shape)
    # Each from_numpy call creates its own storage wrapper around real shared bytes.
    assert source.untyped_storage()._cdata != output.untyped_storage()._cdata
    spans = [(tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size())
             for tensor in (source, output)]
    assert (max(span[0] for span in spans) < min(span[1] for span in spans)) == overlap
    if not overlap:
        assert spans[0][1] == spans[1][0], "adjacent backing intervals are disjoint"
    return projector, backing, source, output


@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("backend", ["tensor", "tensor-store"])
@pytest.mark.parametrize("overlap", [True, False], ids=["overlap", "disjoint"])
def test_into_independently_wrapped_cpu_bytes_validate_before_backend(operation, backend, overlap):
    projector, backing, source, output = _independently_wrapped_cpu_alias_case(operation, overlap)
    original = backing.copy()
    store_type = _api("TensorStore")
    if backend == "tensor-store":
        source, output = store_type(source), store_type(output)
    with mock.patch.object(torch.cuda, "is_available", return_value=False) as backend_probe, \
            mock.patch.object(store_type, "read", autospec=True, side_effect=store_type.read) as reads, \
            mock.patch.object(store_type, "write", autospec=True, side_effect=store_type.write) as writes, \
            _Launches() as launches:
        try:
            if overlap:
                with _rejection("alias|overlap|distinct|backing"):
                    getattr(projector, f"{operation}_into")(source, output)
                backend_probe.assert_not_called()
            else:
                # CPU-only admission control: valid disjoint data reaches the real
                # backend requirement without executing any projection or copies.
                with pytest.raises(RuntimeError, match="CUDA is required"):
                    getattr(projector, f"{operation}_into")(source, output)
                backend_probe.assert_called_once_with()
        finally:
            np.testing.assert_array_equal(backing, original)
            reads.assert_not_called()
            writes.assert_not_called()
            assert not launches.records
