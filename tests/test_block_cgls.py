"""Example-side CGLS: independent numerics, real block I/O and bounded state.

The CPU diagonal operator is an actual linear map for solver-only checks, not
a replacement GPU projector. Native cases use the production Projector and
the independent voxel-cell matrix. Examples are deliberately absent from the
standalone sdist; source checkout and the wheel CI's copied examples run these
tests. No complete-array conversion is permitted for numerical disk state.
"""

from contextlib import ExitStack
import builtins
from datetime import timedelta
from functools import lru_cache
import gc
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
from unittest import mock

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

import diffct
from tests.test_block_storage import (
    _GuardedStore, _numpy_blocks, _rejection, _warm_native_point_kernels,
)
from tests.test_chunked_projector import _memory_inventory, _offsets
from tests.test_detector_surfaces import _close, _data, _matrix, _numpy, _projector
from tests.test_execution_schedules import _CopyTrace
from tests.test_spatial_partition import _fixture, _owned_columns, _run_processes, _slab, _space_projector


ROOT = Path(__file__).resolve().parents[1]
BLOCK = 31
cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="real native CUDA is required")


@lru_cache(maxsize=None)
def _example(name):
    path = ROOT / "examples" / f"{name}.py"
    if not path.is_file():
        assert not (ROOT / ".git").exists(), f"source checkout missing required example {path}"
        pytest.skip("example-side tests require checkout/copied examples; examples are excluded from standalone sdist")
    spec = importlib.util.spec_from_file_location(f"_diffct_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _entry():
    module = _example("_block_cgls")
    function = getattr(module, "cgls_into", None)
    assert callable(function), "INTERFACE_PENDING: examples/_block_cgls.py::cgls_into declaration"
    signature = inspect.signature(function)
    assert tuple(signature.parameters) == ("operator", "measurements", "output", "iterations", "workspace",
                                           "block_elements", "checkpoint_directory", "resume")
    assert signature.parameters["block_elements"].default == 1048576
    return function


def _reference(matrix, measurements, iterations):
    """Independent float64 CGLS recurrence; tiny oracle arrays only."""
    matrix, measurements = np.asarray(matrix, dtype=np.float64), np.asarray(measurements, dtype=np.float64)
    x = np.zeros(matrix.shape[1], dtype=np.float64)
    r = measurements.ravel().copy()
    s = matrix.T @ r
    p, gamma = s.copy(), float(s @ s)
    states = []
    for iteration in range(iterations):
        q = matrix @ p
        qq = float(q @ q)
        if gamma == 0 or qq == 0:
            break
        alpha = gamma / qq
        x += alpha * p
        r -= alpha * q
        s = matrix.T @ r
        new_gamma = float(s @ s)
        p = s + (new_gamma / gamma) * p
        gamma = new_gamma
        states.append({"iteration": iteration + 1, "x": x.copy(), "s": s.copy(), "p": p.copy(),
                       "r": r.copy(), "q": q.copy(), "gamma": gamma})
    return x, states


class _SolverAllocations(TorchDispatchMode):
    def __init__(self, maximum, *, forbid_full=(), native=None):
        super().__init__()
        self.maximum, self.full_shapes = maximum, set(map(tuple, forbid_full))
        self.native = native or [False]
        self.fp64_max = 0

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        inputs = {(item.device, item.untyped_storage()._cdata) for item in tree_flatten(args)[0]
                  if isinstance(item, torch.Tensor)}
        result = function(*args, **(kwargs or {}))
        if not self.native[0]:
            for tensor in tree_flatten(result)[0]:
                if not isinstance(tensor, torch.Tensor):
                    continue
                fresh = (tensor.device, tensor.untyped_storage()._cdata) not in inputs
                if fresh and tensor.dtype == torch.float64:
                    self.fp64_max = max(self.fp64_max, tensor.numel())
                    assert tensor.numel() <= self.maximum, "whole solver vector converted/operated in FP64"
                if fresh and tuple(tensor.shape) in self.full_shapes:
                    assert tensor.numel() <= self.maximum, "whole solver state allocated/copied/updated"
        return result


class _Diagonal:
    """Actual CPU diagonal map with both functional and output-reuse APIs."""
    world_size, rank, process_group, partition = 1, 0, None, "views"

    def __init__(self, diagonal):
        self.diagonal = diagonal
        self.volume_shape = self.projection_shape = tuple(diagonal.shape)
        self.native, self.calls = [False], []

    def _apply(self, value, output=None, *, kind):
        assert not torch.is_grad_enabled(), "numerical CGLS retained an autograd graph"
        value = value.tensor if isinstance(value, diffct.TensorStore) else value
        output = output.tensor if isinstance(output, diffct.TensorStore) else output
        self.calls.append((kind, value.data_ptr(), None if output is None else output.data_ptr()))
        self.native[0] = True
        try:
            result = value * self.diagonal.to(value.dtype) if output is None else torch.mul(value, self.diagonal, out=output)
        finally:
            self.native[0] = False
        return result

    def project(self, value):
        return self._apply(value, kind="project")

    def backproject(self, value):
        return self._apply(value, kind="backproject")

    def project_into(self, value, output):
        self._apply(value, output, kind="project_into")
        return output

    def backproject_into(self, value, output):
        self._apply(value, output, kind="backproject_into")
        return output


@pytest.mark.parametrize("size", [0, 1048593], ids=["empty", "noncontiguous-over-default-block"])
def test_common_cgls_bounded_fp64_noncontiguous_empty_and_reusable_direction(size):
    function = _example("_common").cgls
    diagonal = (1 + torch.arange(size, dtype=torch.float32).remainder(7) * .13)
    measurements = torch.linspace(.1, 1.3, 2 * size, dtype=torch.float32)[::2].requires_grad_()
    operator = _Diagonal(diagonal)
    with torch.no_grad(), _SolverAllocations(1048576, native=operator.native) as allocations:
        result = function(operator, measurements, 3)
    assert not result.requires_grad and result.grad_fn is None and measurements.grad is None
    x = np.zeros(size)
    y, d = _numpy(measurements), _numpy(diagonal)
    r, s = y.copy(), d * y
    p, gamma = s.copy(), float(s @ s)
    for _ in range(3):
        q = d * p
        qq = float(q @ q)
        if gamma == 0 or qq == 0:
            break
        alpha = gamma / qq
        x += alpha * p
        r -= alpha * q
        s = d * r
        next_gamma = float(s @ s)
        p = s + (next_gamma / gamma) * p
        gamma = next_gamma
    _close(result, x, rtol=2e-4, atol=4e-5)
    pointers = [pointer for kind, pointer, _ in operator.calls if kind.startswith("project")]
    assert len(set(pointers)) <= 1, "CGLS replaced the full p update buffer each iteration"
    if size:
        assert allocations.fp64_max <= 1048576
        assert any(kind.endswith("_into") for kind, _, _ in operator.calls), "compatible numerical output buffers were not reused"


def test_common_cgls_is_numerical_even_when_called_with_trainable_measurements():
    diagonal = torch.linspace(.7, 1.9, 19)
    measurements = torch.linspace(-.2, .8, 19).requires_grad_()
    operator = _Diagonal(diagonal)
    output = _example("_common").cgls(operator, measurements, 2)
    assert output.grad_fn is None and not output.requires_grad and measurements.grad is None


class _SolverStore(_GuardedStore):
    def __init__(self, tensor, *, maximum=BLOCK, writable=True):
        super().__init__(tensor, tuple(tensor.shape), writable=writable)
        self.maximum = maximum
        self.device = tensor.device

    def _selection(self, index):
        index, block = super()._selection(index)
        assert block.numel() <= self.maximum, "solver read/write exceeded block_elements"
        return index, block

    def __array__(self, *args, **kwargs):
        raise AssertionError("solver implicitly converted an entire store to an array")


class _ObservedOperator:
    def __init__(self, operator, *, into_only=True):
        self.operator, self.into_only = operator, into_only
        self.native, self.calls = [False], []
        self.q_identity, self.q_locked = None, False

    def __getattr__(self, name):
        if name in ("project", "backproject", "project_into", "backproject_into"):
            original = getattr(self.operator, name)

            def observed(*args, **kwargs):
                assert not torch.is_grad_enabled(), "numerical CGLS invoked projector under autograd"
                assert not self.into_only or name.endswith("_into"), "bounded solver allocated a complete projector result"
                identities = []
                for value in args:
                    if isinstance(value, torch.Tensor):
                        identities.append(("tensor", value.untyped_storage()._cdata))
                    elif isinstance(value, diffct.TensorStore):
                        identities.append(("tensor", value.tensor.untyped_storage()._cdata))
                    elif isinstance(value, diffct.NpyStore):
                        identities.append(("npy", str(value.path)))
                    else:
                        identities.append(("opaque", id(value)))
                self.calls.append((name, identities))
                if name == "backproject_into":
                    # The residual is now consumed by A^T: all preceding x/r
                    # updates must have finished before q can be reused.
                    self.q_locked = False
                self.native[0] = True
                try:
                    result = original(*args, **kwargs)
                finally:
                    self.native[0] = False
                if name == "project_into":
                    self.q_identity, self.q_locked = identities[-1], True
                return result
            return observed
        return getattr(self.operator, name)


class _CudaObservedOperator(_ObservedOperator):
    def __getattr__(self, name):
        original = super().__getattr__(name)
        if name not in ("project", "backproject", "project_into", "backproject_into"):
            return original

        def observed(*args, **kwargs):
            for value in args:
                tensor = value.tensor if isinstance(value, diffct.TensorStore) else value
                assert isinstance(tensor, torch.Tensor) and tensor.is_cuda and tensor.dtype == torch.float32, \
                    "admitted common CGLS state left its CUDA device"
            return original(*args, **kwargs)
        return observed


class _CudaSolverTrace(_CopyTrace):
    def __init__(self, case, native):
        super().__init__(case)
        self.native, self.scalar_reads = native, 0
        self.dot_storages, self.dot_cuda_peak = {}, 0

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = super().__torch_dispatch__(function, types, args, kwargs)
        if not self.native[0]:
            if str(function) == "aten._local_scalar_dense.default" and args[0].is_cuda:
                assert args[0].numel() == 1, "non-scalar CUDA state converted into a Python value"
                self.scalar_reads += 1
            for tensor in tree_flatten(result)[0]:
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda and tensor.dtype == torch.float64:
                    storage = tensor.untyped_storage()
                    key = storage._cdata
                    previous = self.dot_storages.get(key)
                    if previous is None or torch.UntypedStorage._expired(previous[0]):
                        if previous is not None:
                            torch.UntypedStorage._free_weak_ref(previous[0])
                        # Native allocation rounding, separately owned typed dot
                        # storage. A shared scratch arena is counted only once.
                        self.dot_storages[key] = (storage._weak_ref(), (storage.nbytes() + 511) // 512 * 512)
            live = sum(size for weak, size in self.dot_storages.values() if not torch.UntypedStorage._expired(weak))
            self.dot_cuda_peak = max(self.dot_cuda_peak, live)
        return result

    def __exit__(self, *exception):
        try:
            return super().__exit__(*exception)
        finally:
            for weak, _ in self.dot_storages.values():
                torch.UntypedStorage._free_weak_ref(weak)


def _npy_io(records, maximum=BLOCK, *, operator=None):
    patches = ExitStack()
    for method in ("read", "write", "flush"):
        original = getattr(diffct.NpyStore, method)

        def observed(store, *args, _method=method, _original=original, **kwargs):
            if _method in ("read", "write"):
                block = store._array[args[0]]
                assert block.size <= maximum, "working/checkpoint mmap I/O exceeded block_elements"
                if _method == "write":
                    assert args[1].numel() <= maximum
                    if operator is not None and operator.q_locked and not operator.native[0]:
                        assert ("npy", str(store.path)) != operator.q_identity, \
                            "q overwritten before globally fixed alpha and completed x/r updates"
            records.append((_method, str(store.path)))
            return _original(store, *args, **kwargs)
        patches.enter_context(mock.patch.object(diffct.NpyStore, method, observed))
    real_flush = np.memmap.flush

    def flush(mapping):
        backing = mapping
        if mapping.filename is None and mapping.size == 0:
            # NumPy shares_memory(empty_view, base) is false. Its memmap
            # finalizer therefore drops filename/_mmap on the observed view.
            # Flush the genuine mapped base, preserving publication evidence.
            seen = set()
            while backing is not None and id(backing) not in seen:
                seen.add(id(backing))
                if isinstance(backing, np.memmap) and backing.filename is not None:
                    break
                backing = getattr(backing, "base", None)
            assert isinstance(backing, np.memmap) and backing.filename is not None, \
                "empty observed mapping has no genuine mapped file backing"
        result = real_flush(backing)
        records.append(("flush", str(Path(backing.filename).resolve())))
        return result
    patches.enter_context(mock.patch.object(np.memmap, "flush", flush))
    return patches


def _checkpoint_publish(directory, records):
    """Observe real final JSON publication after every mapped state flush."""
    directory = Path(directory).resolve()

    def before_publish(path):
        path = Path(path).resolve()
        if path.suffix == ".json" and path.is_relative_to(directory):
            for name in ("x", "s", "p", "r", "q"):
                state = path.parent / f"{name}.npy"
                assert state.is_file(), "metadata published before complete checkpoint states"
                assert ("flush", str(state)) in records, "metadata published before actual state flush"

    real_open, real_path_open, real_replace, real_rename = builtins.open, Path.open, os.replace, Path.rename

    def open_file(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and any(flag in mode for flag in "wax"):
            before_publish(file)
        return real_open(file, mode, *args, **kwargs)

    def replace(source, target, *args, **kwargs):
        before_publish(target)
        return real_replace(source, target, *args, **kwargs)

    def path_open(path, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax"):
            before_publish(path)
        return real_path_open(path, mode, *args, **kwargs)

    def rename(source, target):
        before_publish(target)
        return real_rename(source, target)

    patches = ExitStack()
    patches.enter_context(mock.patch.object(builtins, "open", open_file))
    patches.enter_context(mock.patch.object(Path, "open", path_open))
    patches.enter_context(mock.patch.object(os, "replace", replace))
    patches.enter_context(mock.patch.object(Path, "rename", rename))
    return patches


def _solver_fixture(beam):
    case, _ = _fixture(beam)
    image, measurements = _data(case, device="cpu")
    chunk = (2, 3, 4) if beam == "cone" else (2, 4)
    pixels = (3, 2) if beam == "cone" else (3,)
    operator = _projector(case, volume_chunk_shape=chunk, detector_chunk_shape=pixels, view_chunk_size=2)
    matrix = _matrix(case, _offsets(case, flat=True))
    return case, image, measurements, operator, matrix


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", ["parallel", "fan", "cone"])
@pytest.mark.parametrize("backend", ["tensor", "opaque", "npy"])
def test_cgls_into_all_beams_matches_independent_recurrence_with_only_bounded_state_io(beam, backend, tmp_path):
    function = _entry()
    _warm_native_point_kernels()
    case, _, measurements, operator, matrix = _solver_fixture(beam)
    expected, _ = _reference(matrix, _numpy(measurements), 4)
    if backend == "tensor":
        source = measurements.transpose(-1, -2).contiguous().transpose(-1, -2)
        output = torch.full(case.shape[::-1], float("nan")).permute(*reversed(range(len(case.shape))))
        assert not output.is_contiguous()
    elif backend == "opaque":
        source = _SolverStore(measurements, writable=False)
        output = _SolverStore(torch.full(case.shape, float("nan")))
    else:
        np.save(tmp_path / "measurements.npy", _numpy(measurements).astype(np.float32))
        source = diffct.NpyStore(tmp_path / "measurements.npy", mode="r")
        output = diffct.NpyStore.create(tmp_path / "result.npy", case.shape)
        output._array.fill(np.nan)
    observed, io = _ObservedOperator(operator), []
    with _SolverAllocations(BLOCK, forbid_full=(case.shape, case.sino_shape), native=observed.native), \
            _numpy_blocks(BLOCK), _npy_io(io, operator=observed):
        actual = function(observed, source, output, 4, workspace=tmp_path / "work", block_elements=BLOCK)
    assert actual is output
    values = output if isinstance(output, torch.Tensor) else output.tensor if isinstance(output, _SolverStore) else torch.from_numpy(output._array.copy())
    _close(values, expected.reshape(case.shape), rtol=8e-4, atol=3e-4)
    assert values.grad_fn is None
    assert {name for name, _ in observed.calls} == {"project_into", "backproject_into"}
    for name in ("project_into", "backproject_into"):
        sinks = [identities[-1] for method, identities in observed.calls if method == name]
        assert len(set(sinks)) == 1, f"{name} did not reuse q/s backing state"
    assert io and all(Path(path).resolve().is_relative_to(tmp_path.resolve()) for _, path in io)
    if isinstance(output, _SolverStore):
        assert output.writes and torch.all(output.overwrites > 0)


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("iterations", [0, 2])
def test_zero_measurements_and_zero_iterations_overwrite_nan_output_without_graph(iterations, tmp_path):
    case, _, measurements, operator, _ = _solver_fixture("parallel")
    output = torch.full(case.shape, float("nan"), requires_grad=True)
    measurements = torch.zeros_like(measurements, requires_grad=True)
    result = _entry()(operator, measurements, output, iterations, workspace=tmp_path / "work", block_elements=BLOCK)
    assert result is output and torch.count_nonzero(output) == 0
    assert output.grad_fn is None and output.grad is None and measurements.grad is None


@pytest.mark.parametrize("argument,value", [("iterations", -1), ("iterations", True), ("iterations", 1.5),
                                           ("block_elements", 0), ("block_elements", True), ("block_elements", 1.5)])
def test_cgls_into_rejects_invalid_counts_before_store_reads(argument, value, tmp_path):
    case, _ = _fixture("parallel")
    operator = _projector(case)
    source = _SolverStore(torch.ones(case.sino_shape), writable=False)
    output = _SolverStore(torch.full(case.shape, float("nan")))
    kwargs = {"workspace": tmp_path / "work", "block_elements": BLOCK}
    iterations = 1
    if argument == "iterations":
        iterations = value
    else:
        kwargs[argument] = value
    with _rejection("iteration|block|integer|positive|nonnegative"):
        _entry()(operator, source, output, iterations, **kwargs)
    assert not source.reads and not output.writes


def test_cgls_into_existing_workspace_or_readonly_output_preserves_unrelated_files(tmp_path):
    case, _ = _fixture("parallel")
    operator = _projector(case)
    source = _SolverStore(torch.ones(case.sino_shape), writable=False)
    output = _SolverStore(torch.zeros(case.shape))
    work = tmp_path / "occupied"
    work.mkdir()
    sentinel = work / "unrelated.txt"
    sentinel.write_text("owned elsewhere", encoding="utf-8")
    with _rejection("exist|workspace|empty|owned|directory"):
        _entry()(operator, source, output, 1, workspace=work, block_elements=BLOCK)
    assert sentinel.read_text() == "owned elsewhere"
    output.writable = False
    with _rejection("read.?only|writable|write"):
        _entry()(operator, source, output, 1, workspace=tmp_path / "new", block_elements=BLOCK)
    assert not source.reads


@pytest.mark.cuda
@cuda_required
def test_common_dense_cuda_cgls_admits_all_five_states_or_falls_back_to_host(tmp_path):
    _warm_native_point_kernels()
    case, _ = _fixture("parallel", first=19)
    case.shape, case.detector = (19, 29), (9,)
    _, measurements = _data(case, device="cuda")
    operator = _ObservedOperator(_projector(case), into_only=False)
    matrix = _matrix(case, _offsets(case, flat=True))
    expected, _ = _reference(matrix, _numpy(measurements), 2)
    torch.cuda.synchronize()
    initial = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), _memory_inventory({torch.cuda.current_device(): 8192}):
        result = _example("_common").cgls(operator, measurements, 2)
    torch.cuda.synchronize()
    # Three volume states require 6612 logical bytes before r/q and pipeline;
    # even packing cannot fit the 6144-byte budget. One serial raw native tile
    # still fits (2560 volume + 512 rays + 3*512 frames = 4608 native bytes).
    assert result.device.type == "cpu", "dense GPU solver omitted x/s/p/r/q from admission"
    assert torch.cuda.max_memory_allocated() - initial <= 6144
    _close(result, expected.reshape(case.shape), rtol=8e-4, atol=3e-4)


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", ["parallel", "fan", "cone"])
def test_common_admitted_cuda_cgls_keeps_state_and_dot_blocks_on_device(beam):
    _warm_native_point_kernels()
    case, _ = _fixture(beam)
    case.trajectory = tuple(value.float() for value in case.trajectory)
    image, measurements_cpu = _data(case, device="cpu")
    measurements = measurements_cpu.cuda().requires_grad_()
    original_measurements = measurements_cpu.clone()
    operator = _CudaObservedOperator(_projector(case, schedule="spatial", volume_chunk_shape=case.shape,
                                               view_chunk_size=2))
    matrix = _matrix(case, _offsets(case, flat=True))
    expected, states = _reference(matrix, _numpy(measurements_cpu), 3)
    assert len(states) == 3, "oracle terminated before repeated q/s reuse and norm decisions"
    operator.operator.project(image)
    operator.operator.backproject(measurements_cpu)
    gc.collect()
    torch.cuda.synchronize()
    initial = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with _SolverAllocations(1048576, native=operator.native) as allocations, \
            _CudaSolverTrace(case, operator.native) as trace:
        result = _example("_common").cgls(operator, measurements, 3)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - initial
    # Establish native numerical behavior and all five resident states before
    # classifying any transport regression. Fixture/result oracle downloads are
    # outside the observer and do not contribute to the solver's counters.
    _close(result, expected.reshape(case.shape), rtol=8e-4, atol=3e-4)
    torch.testing.assert_close(measurements.detach().cpu(), original_measurements)
    assert result.is_cuda and not result.requires_grad and result.grad_fn is None and measurements.grad is None
    assert {name for name, _ in operator.calls} == {"project_into", "backproject_into"}
    for kind in ("project_into", "backproject_into"):
        calls = [identities for name, identities in operator.calls if name == kind]
        assert len({identities[0] for identities in calls}) == len({identities[-1] for identities in calls}) == 1, \
            "common CGLS replaced its p/r or q/s working buffer"
    assert trace.launches and all(record["volume"]["device"].type == "cuda" for record in trace.launches)
    report = operator.last_solver_stats
    volume, rays = math.prod(case.shape), math.prod(case.sino_shape)
    def rounded(elements):
        return (4 * elements + 511) // 512 * 512
    assert report["state_backend"] == "cuda"
    assert report["requested_volume_state_bytes"] == 3 * volume * 4
    assert report["requested_ray_state_bytes"] == 2 * rays * 4
    assert report["estimated_cuda_state_bytes"] >= 3 * rounded(volume) + 2 * rounded(rays)
    assert report["estimated_gpu_bytes"] >= report["estimated_cuda_state_bytes"] + report["estimated_pipeline_and_geometry_bytes"] + trace.dot_cuda_peak
    assert 0 < peak <= report["estimated_gpu_bytes"] <= report["gpu_budget_bytes"]
    assert allocations.fp64_max <= 1048576
    # Copies execute normally. Only a reduced scalar may cross to the host for
    # alpha/beta; CUDA residual and FP64 dot blocks stay on their device.
    downloads = [record for record in trace.copies if record["direction"] == "d2h"]
    assert all(math.prod(record["target"]["shape"]) <= 1 for record in downloads), \
        ("admitted CUDA CGLS downloaded measurement/state/dot blocks to CPU", downloads)
    scalar_downloads = len(downloads) + trace.scalar_reads
    assert scalar_downloads >= 2 * len(states) + 1, "actual globally owned norm decisions were not observed"


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", ["parallel", "cone"])
def test_checkpoints_save_completed_bounded_states_and_resume_to_total_target(beam, tmp_path):
    function = _entry()
    _warm_native_point_kernels()
    case, _, measurements, operator, matrix = _solver_fixture(beam)
    _, reference = _reference(matrix, _numpy(measurements), 5)
    source = _SolverStore(measurements, writable=False)
    output = _SolverStore(torch.full(case.shape, float("nan")))
    checkpoints, io = tmp_path / "checkpoints", []
    observed = _ObservedOperator(operator)
    with _SolverAllocations(BLOCK, forbid_full=(case.shape, case.sino_shape), native=observed.native), \
            _numpy_blocks(BLOCK), _npy_io(io, operator=observed), _checkpoint_publish(checkpoints, io):
        function(observed, source, output, 2, workspace=tmp_path / "first", block_elements=BLOCK,
                 checkpoint_directory=checkpoints)
    metadata_files = list(checkpoints.rglob("*.json"))
    assert metadata_files, "completed iteration checkpoint metadata missing"
    candidates = [(path, json.loads(path.read_text())) for path in metadata_files]
    checkpoint, metadata = next((path.parent, metadata) for path, metadata in candidates if metadata["iteration"] == 2)
    assert metadata["gamma"] == pytest.approx(reference[1]["gamma"], rel=2e-3)
    assert tuple(metadata["global_volume_shape"]) == tuple(metadata["local_volume_shape"]) == case.shape
    assert metadata["partition"] == "views" and metadata["rank"] == 0 and metadata["world_size"] == 1
    assert metadata["schema_version"] == 1 and metadata["dtype"] == "float32"
    assert metadata["volume_slice"] == [[0, size] for size in case.shape]
    assert tuple(metadata["detector_shape"]) == case.detector
    assert tuple(metadata["projection_shape"]) == case.sino_shape
    assert metadata["beam"] == case.beam and metadata["voxel_spacing"] == case.spacing
    assert tuple(metadata["detector_spacing"]) == case.pitch
    for name, shape in (("x", case.shape), ("s", case.shape), ("p", case.shape),
                        ("r", case.sino_shape), ("q", case.sino_shape)):
        path = checkpoint / f"{name}.npy"
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        assert values.shape == shape and values.dtype == np.float32
        np.testing.assert_allclose(values.ravel(), reference[1][name], rtol=2e-3, atol=3e-4)
        assert ("flush", str(path.resolve())) in io, "checkpoint state not flushed"
    resumed = _SolverStore(torch.full(case.shape, float("nan")))
    with _SolverAllocations(BLOCK, forbid_full=(case.shape, case.sino_shape), native=observed.native), \
            _numpy_blocks(BLOCK), _npy_io(io, operator=observed):
        function(observed, source, resumed, 5, workspace=tmp_path / "resumed", block_elements=BLOCK, resume=checkpoint)
    _close(resumed.tensor, reference[-1]["x"].reshape(case.shape), rtol=2e-3, atol=5e-4)
    assert len([name for name, _ in observed.calls if name == "project_into"]) == 5
    with _rejection("iteration|target|resume|completed"):
        function(operator, source, _SolverStore(torch.zeros(case.shape)), 1,
                 workspace=tmp_path / "too-early", block_elements=BLOCK, resume=checkpoint)
    # This metadata change is an actual incompatible checkpoint, not mocked I/O.
    metadata_path = next(path for path, data in candidates if data["iteration"] == 2)
    original_metadata = metadata_path.read_text()
    alternatives = {"schema_version": 2, "partition": "space", "dtype": "float64", "rank": 1, "world_size": 2,
                    "beam": "fan" if beam == "parallel" else "parallel", "voxel_spacing": case.spacing * 1.2,
                    "detector_spacing": [value * 1.2 for value in case.pitch],
                    "global_volume_shape": [case.shape[0] + 1, *case.shape[1:]],
                    "local_volume_shape": [case.shape[0] + 1, *case.shape[1:]],
                    "volume_slice": [[0, case.shape[0] + 1], *[[0, size] for size in case.shape[1:]]],
                    "detector_shape": [case.detector[0] + 1, *case.detector[1:]],
                    "projection_shape": [case.views + 1, *case.detector]}
    for field, value in alternatives.items():
        damaged = json.loads(original_metadata)
        damaged[field] = value
        metadata_path.write_text(json.dumps(damaged), encoding="utf-8")
        invalid_output = _SolverStore(torch.full(case.shape, float("nan")))
        before = len(io)
        with _npy_io(io), _rejection("checkpoint|resume|partition|metadata|shape|ownership|dtype|version|spacing|rank|beam|scan"):
            function(operator, source, invalid_output, 5,
                     workspace=tmp_path / f"incompatible-{field}", block_elements=BLOCK, resume=checkpoint)
        assert not invalid_output.writes
        assert not any(method == "write" for method, _ in io[before:]), "incompatible resume wrote state before metadata admission"
    metadata_path.write_text(original_metadata, encoding="utf-8")


def _distributed_solver_worker(rank, partition, first, views, backend, rendezvous, directory):
    torch.cuda.set_device(rank if backend == "nccl" else 0)
    _warm_native_point_kernels()  # Spawned processes have independent cold Numba import registries.
    dist.init_process_group(backend, init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        case, _ = _fixture("cone", first, views=views)
        chunk, pixels = (2, 3, 4), (3, 2)
        operator = _space_projector(case, partition=partition, distributed=True,
                                    volume_chunk_shape=chunk, detector_chunk_shape=pixels, view_chunk_size=2,
                                    devices=[torch.cuda.current_device()])
        _, full_measurements = _data(case, device="cpu")
        matrix = _matrix(case, _offsets(case, flat=True))
        expected, reference = _reference(matrix, _numpy(full_measurements), 3)
        measurements = (full_measurements if partition == "space" else full_measurements[operator.view_slice]).clone()
        source = _SolverStore(measurements, writable=False)
        rank_directory = Path(directory) / f"rank-{rank}"
        rank_directory.mkdir()
        output = diffct.NpyStore.create(rank_directory / "output.npy", operator.local_volume_shape)
        output._array.fill(np.nan)
        observed, io, scalars, q_decisions = _ObservedOperator(operator), [], [], []
        original_reduce = dist.all_reduce

        def scalar_reduce(tensor, *args, **kwargs):
            is_scalar = not observed.native[0] and tensor.is_floating_point() and tensor.numel() == 1
            if is_scalar:
                if backend == "nccl":
                    assert tensor.is_cuda, "NCCL CGLS scalar used CPU staging"
                scalars.append(float(tensor.detach().cpu().item()))
            result = original_reduce(tensor, *args, **kwargs)
            if is_scalar and observed.q_locked:
                # q remains the real completed ray state through the global
                # scalar decision; records are metadata and bounded reads only.
                q_decisions.append(observed.q_identity)
            return result

        checkpoint_root = rank_directory / "checkpoints"
        with mock.patch.object(dist, "all_reduce", scalar_reduce), \
                _SolverAllocations(BLOCK, forbid_full=(operator.volume_shape, operator.projection_shape), native=observed.native), \
                _numpy_blocks(BLOCK), _npy_io(io, operator=observed), _checkpoint_publish(checkpoint_root, io):
            result = _entry()(observed, source, output, 3, workspace=rank_directory / "work", block_elements=BLOCK,
                              checkpoint_directory=checkpoint_root)
        assert result is output and scalars and q_decisions
        columns = (_owned_columns(case.shape, _slab(case.shape, rank)) if partition == "space"
                   else np.arange(math.prod(case.shape)))
        _close(torch.from_numpy(output._array.copy()), expected[columns].reshape(operator.volume_shape), rtol=2e-3, atol=5e-4)
        metadata = [(path, json.loads(path.read_text())) for path in checkpoint_root.rglob("*.json")]
        path, saved = next((path, data) for path, data in metadata if data["iteration"] == 3)
        assert saved["gamma"] == pytest.approx(reference[-1]["gamma"], rel=3e-3)
        assert saved["rank"] == rank and saved["world_size"] == 2 and saved["partition"] == partition
        assert saved["volume_slice"] == [[part.start, part.stop] for part in operator.volume_slice]
        assert tuple(saved["local_volume_shape"]) == operator.volume_shape
        assert np.load(path.parent / "x.npy", mmap_mode="r").shape == operator.volume_shape
        (rank_directory / "observations.json").write_text(json.dumps({"scalars": scalars,
            "q_decisions": q_decisions, "owned_shape": operator.volume_shape}), encoding="utf-8")
    finally:
        dist.destroy_process_group()


@pytest.mark.cuda
@cuda_required
@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="real two-process Gloo is required")
@pytest.mark.parametrize("partition,first,views", [("space", 5, 5), ("space", 1, 5), ("views", 5, 5), ("views", 5, 1)],
                         ids=["space-uneven", "space-empty", "views-uneven", "views-empty"])
def test_real_gloo_disk_cgls_global_norms_q_lifetime_empty_ownership_and_local_checkpoints(partition, first, views, tmp_path):
    _run_processes(_distributed_solver_worker, (partition, first, views, "gloo",
                                               (tmp_path / "solver").resolve().as_uri(), str(tmp_path)), timeout=120)
    records = [json.loads((tmp_path / f"rank-{rank}" / "observations.json").read_text()) for rank in range(2)]
    assert len(records[0]["scalars"]) == len(records[1]["scalars"])
    assert len(records[0]["q_decisions"]) == len(records[1]["q_decisions"])


@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2 or not dist.is_nccl_available(),
                    reason="two actual CUDA GPUs and real NCCL are required")
@pytest.mark.parametrize("partition", ["views", "space"])
def test_real_nccl_disk_cgls_norm_decisions_use_cuda_scalars_and_owned_state(partition, tmp_path):
    _run_processes(_distributed_solver_worker, (partition, 1, 1, "nccl",
                                               (tmp_path / "solver-nccl").resolve().as_uri(), str(tmp_path)), timeout=120)


class _FallbackCopyTrace(_CopyTrace):
    def __init__(self, case):
        super().__init__(case)
        self.cuda_fp32_casts = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = super().__torch_dispatch__(function, types, args, kwargs)
        if str(function) == "aten._to_copy.default" and args[0].is_cuda and args[0].dtype == torch.float64 \
                and isinstance(result, torch.Tensor) and result.is_cuda and result.dtype == torch.float32:
            self.cuda_fp32_casts.append({"phase": self.phase, "elements": result.numel(),
                                         "bytes": result.numel() * result.element_size()})
        return result


def _fp64_fallback_copy_trial():
    # Same tiny real matrix fixture as the existing dense admission regression.
    # Available-room metadata constrains admission; allocations/copies/kernels
    # remain real and no CUDA OOM is fabricated.
    _warm_native_point_kernels()
    case, _ = _fixture("parallel", first=19)
    case.shape, case.detector = (19, 29), (9,)
    case.trajectory = tuple(value.float() for value in case.trajectory)
    image, cpu_measurements = _data(case, device="cpu")
    saved = cpu_measurements.double()
    measurements = saved.cuda().requires_grad_()
    operator = _ObservedOperator(_projector(case, schedule="spatial", volume_chunk_shape=case.shape,
                                           view_chunk_size=2))
    matrix = _matrix(case, _offsets(case, flat=True))
    expected, _ = _reference(matrix, _numpy(saved), 2)
    operator.operator.project(image)
    operator.operator.backproject(cpu_measurements)
    try:
        blocks = importlib.import_module("_block_cgls")
    except ModuleNotFoundError as error:
        if error.name != "_block_cgls":
            raise
        blocks = importlib.import_module("examples._block_cgls")
    copy_blocks, initial = blocks._copy_blocks, {}
    trace = _FallbackCopyTrace(case)

    def initial_copy(source, target, elements):
        assert source.device.type == "cuda" and source.dtype == torch.float64 and target.device.type == "cpu"
        trace.phase = "initial-copy"
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = copy_blocks(source, target, elements)
        torch.cuda.synchronize()
        initial["extra_cuda_peak_bytes"] = torch.cuda.max_memory_allocated() - baseline
        trace.phase = "solver"
        return result

    with _memory_inventory({torch.cuda.current_device(): 8192}), \
            _SolverAllocations(1048576, native=operator.native), trace, \
            mock.patch.object(blocks, "_copy_blocks", side_effect=initial_copy):
        result = _example("_common").cgls(operator, measurements, 2)
    _close(result, expected.reshape(case.shape), rtol=8e-4, atol=3e-4)
    torch.testing.assert_close(measurements.detach().cpu(), saved)
    assert result.device.type == "cpu" and not result.requires_grad and result.grad_fn is None and measurements.grad is None
    assert {name for name, _ in operator.calls} == {"project_into", "backproject_into"}
    for kind in ("project_into", "backproject_into"):
        calls = [identities for name, identities in operator.calls if name == kind]
        assert len({value[0] for value in calls}) == len({value[-1] for value in calls}) == 1
    assert trace.launches
    report = operator.last_solver_stats
    assert report["state_backend"] == "cpu" and report["gpu_budget_bytes"] == 6144
    assert report["estimated_cuda_state_bytes"] == report["estimated_cuda_dot_scratch_bytes"] == 0
    initial["casts"] = [record for record in trace.cuda_fp32_casts if record["phase"] == "initial-copy"]
    initial["downloads"] = [record for record in trace.copies if record["phase"] == "initial-copy" and record["direction"] == "d2h"]
    initial["expected_fp64_bytes"] = measurements.numel() * 8

    # The block-copy stress is independent of detector/native size: one full
    # default FP64 block plus a seven-element tail, two real bounded writes.
    size = 1048576 + 7
    original = torch.arange(size, dtype=torch.float64).remainder(17).sub_(8).div_(9)
    source = diffct.TensorStore(original.cuda().requires_grad_())
    target = _SolverStore(torch.full((size,), torch.nan), maximum=1048576)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), _SolverAllocations(1048576), _FallbackCopyTrace(case) as direct_trace:
        copy_blocks(source, target, 1048576)
    torch.cuda.synchronize()
    direct_peak = torch.cuda.max_memory_allocated() - baseline
    torch.testing.assert_close(target.tensor, original.float(), rtol=0, atol=0)
    torch.testing.assert_close(source.tensor.detach().cpu(), original)
    assert source.tensor.grad is None and not target.tensor.requires_grad and target.tensor.grad_fn is None
    assert torch.all(target.overwrites == 1) and len(target.writes) == 2
    assert [index[0].stop - (index[0].start or 0) for index, _ in target.writes] == [1048576, 7]
    direct = dict(extra_cuda_peak_bytes=direct_peak, casts=direct_trace.cuda_fp32_casts,
                  downloads=[record for record in direct_trace.copies if record["direction"] == "d2h"],
                  expected_fp64_bytes=size * 8, write_elements=[1048576, 7])
    return {"report": report, "initial": initial, "direct": direct,
            "matrix_native_numerics": "PASS", "block_copy_numerics": "PASS",
            "immutability_no_grad_reused_state": "PASS"}


@pytest.mark.cuda
@cuda_required
def test_common_cpu_state_fp64_cuda_input_copies_before_cast_without_unbudgeted_gpu_scratch():
    trial = _fp64_fallback_copy_trial()
    report = trial["report"]
    for phase in ("initial", "direct"):
        record = trial[phase]
        assert record["extra_cuda_peak_bytes"] <= report["estimated_gpu_bytes"] <= report["gpu_budget_bytes"], \
            ("CPU-target FP64 copy cast allocated unbudgeted CUDA scratch", phase, trial)
        assert record["casts"] == [], ("CPU-target FP64 input was converted to FP32 on CUDA before its CPU copy", phase, trial)
        assert record["downloads"] and all(copy["equal_dtype"] for copy in record["downloads"])
        assert sum(copy["bytes"] for copy in record["downloads"]) == record["expected_fp64_bytes"], \
            "FP64 blocks must cross to CPU before the bounded FP32 conversion"
