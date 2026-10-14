"""Batch boundaries retain gradients and bound real portable allocations."""

import math
import weakref

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from diffct import Projector
from tests.test_portable_execution_bounds import WORK_BYTES, bounded_cpu_threads
from tests.test_reference_backend import CUDA_REQUIRED, _Case, _case, _data, _matrix


BACKENDS = ["torch", pytest.param("numba", marks=[pytest.mark.cuda, CUDA_REQUIRED])]
DEVICES = ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda, CUDA_REQUIRED])]


def _empty_operator(backend):
    case = _case("parallel", scalar=True)
    return Projector(
        case.trajectory, (2, 3), case.detector, beam=case.beam,
        detector_spacing=case.pitch[0], voxel_spacing=case.spacing,
        backend=backend, precision="fast" if backend == "numba" else "float64",
    )


def _empty_contract(value, leading, shape, dtype, device):
    assert value.shape == (*leading, *shape)
    assert value.dtype == dtype and value.device == device
    assert value.numel() == 0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("leading", [(0,), (2, 0)])
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_empty_batches_support_ordinary_backward(backend, device, leading, operation):
    operator = _empty_operator(backend)
    input_shape, output_shape = operator.volume_shape, operator.projection_shape
    if operation == "backproject":
        input_shape, output_shape = output_shape, input_shape
    value = torch.empty((*leading, *input_shape), dtype=torch.float64,
                        device=device, requires_grad=True)
    result = getattr(operator, operation)(value)
    _empty_contract(result, leading, output_shape, operator.dtype, value.device)
    assert result.requires_grad, "empty batch output detached from its differentiable input"
    result.sum().backward()
    assert value.grad is not None
    _empty_contract(value.grad, leading, input_shape, value.dtype, value.device)
    torch.testing.assert_close(value.grad, torch.zeros_like(value))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("leading", [(0,), (2, 0)])
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_empty_batches_support_func_grad_and_jvp(backend, device, leading, operation):
    operator = _empty_operator(backend)
    input_shape, output_shape = operator.volume_shape, operator.projection_shape
    if operation == "backproject":
        input_shape, output_shape = output_shape, input_shape
    value = torch.empty((*leading, *input_shape), dtype=torch.float64, device=device)
    function = getattr(operator, operation)
    gradient = torch.func.grad(lambda x: function(x).sum())(value)
    _empty_contract(gradient, leading, input_shape, value.dtype, value.device)
    torch.testing.assert_close(gradient, torch.zeros_like(value))
    primal, tangent = torch.func.jvp(function, (value,), (torch.zeros_like(value),))
    for result in (primal, tangent):
        _empty_contract(result, leading, output_shape, operator.dtype, value.device)


class _WorkingAllocations(TorchDispatchMode):
    """Record real storage, excluding caller arrays and the returned result.

    Weak references distinguish live result aliases from freed intermediates
    whose allocator addresses may later be reused. Expanded index views count
    their backing storage, rather than their larger virtual tensor payload.
    """

    def __init__(self, *caller_arrays):
        super().__init__()
        self.caller_storage = {value.untyped_storage().data_ptr() for value in caller_arrays}
        self.records = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        pending = [result]
        while pending:
            value = pending.pop()
            if isinstance(value, torch.Tensor):
                storage = value.untyped_storage()
                if storage.data_ptr() not in self.caller_storage:
                    self.records.append((storage.nbytes(), str(func), tuple(value.shape), weakref.ref(value)))
            elif isinstance(value, (tuple, list)):
                pending.extend(value)
        return result

    def largest_intermediate(self, output):
        result_storage = output.untyped_storage().data_ptr()
        retained = []
        for payload, operation, shape, reference in self.records:
            value = reference()
            if value is None or value.untyped_storage().data_ptr() != result_storage:
                retained.append((payload, operation, shape))
        return max(retained, default=(0, "", ()), key=lambda record: record[0])


def _batch_case():
    # Translated and reversed axis-aligned rays cross full voxel-box chords.
    directions = torch.tensor([[1., 0.], [-1., 0.], [1., 0.], [-1., 0.]], dtype=torch.float64)
    origins = torch.tensor([[.17, -.23], [-.31, .193], [.27, .41], [-.13, -.37]], dtype=torch.float64)
    axes = torch.tensor([[0., 1.]] * 4, dtype=torch.float64)
    return _Case("parallel", (directions, origins, axes), (19, 31), (17,), (.31,), (.7, .31))


def _batched_apply(operation, leading, chunks):
    case = _batch_case()
    matrix = _matrix(case)  # Independent per-voxel box intersections, not Siddon traversal.
    input_shape, output_shape = case.shape, case.projection_shape
    if operation == "backproject":
        matrix = matrix.T
        input_shape, output_shape = output_shape, input_shape
    amplitudes = torch.linspace(.3, 1.7, math.prod(leading), dtype=torch.float64).reshape(leading)
    value = amplitudes[(..., None, None)] * _data(input_shape, phase=.37)
    expected = (value.reshape(-1, math.prod(input_shape)) @ matrix.T).reshape(*leading, *output_shape)
    # Caller and result arrays fit the ceiling; the large allocation comes
    # specifically from ray intervals replicated across batch/channel axes.
    assert value.untyped_storage().nbytes() < WORK_BYTES
    assert expected.untyped_storage().nbytes() < WORK_BYTES
    operator = Projector(
        case.trajectory, case.shape, case.detector, beam=case.beam,
        detector_spacing=case.pitch[0], voxel_spacing=case.spacing,
        backend="torch", precision="float64", **chunks,
    )
    with _WorkingAllocations(value) as allocations:
        result = getattr(operator, operation)(value)
    assert result.shape == (*leading, *output_shape)
    assert result.dtype == torch.float64 and result.device == value.device
    torch.testing.assert_close(result, expected, rtol=2e-11, atol=2e-11)
    return operator, allocations.largest_intermediate(result)


@pytest.mark.parametrize("leading", [(1000,), (20, 50)])
@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.usefixtures("bounded_cpu_threads")
def test_portable_automatic_cap_includes_all_leading_axes(operation, leading):
    _, (payload, allocation, shape) = _batched_apply(operation, leading, {})
    assert payload <= WORK_BYTES, (
        f"{allocation} allocated {shape} ({payload} storage bytes); "
        f"automatic batched intermediates must fit {WORK_BYTES} bytes"
    )


@pytest.mark.parametrize("leading", [(1000,), (20, 50)])
@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("chunks, bounded", [
    ({"volume_chunk_shape": (7, 11), "view_chunk_size": 2, "detector_chunk_shape": (5,)}, True),
    ({"volume_chunk_shape": (19, 31), "view_chunk_size": 4, "detector_chunk_shape": (17,)}, False),
], ids=["tight-explicit-limits", "full-explicit-override"])
@pytest.mark.usefixtures("bounded_cpu_threads")
def test_portable_explicit_hints_keep_priority_for_batches(operation, leading, chunks, bounded):
    operator, (payload, _, _) = _batched_apply(operation, leading, chunks)
    stats = operator.last_execution_stats
    assert stats["chunk_shape"] == chunks["volume_chunk_shape"]
    assert stats["view_chunk_size"] == chunks["view_chunk_size"]
    assert stats["detector_chunk_shape"] == chunks["detector_chunk_shape"]
    # Full explicit hints may exceed the automatic working-table ceiling.
    if bounded:
        assert payload <= WORK_BYTES


@pytest.mark.parametrize("leading", [(300000,), (600, 500)])
@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.usefixtures("bounded_cpu_threads")
def test_portable_automatic_cap_at_minimum_ray_and_voxel_chunks(operation, leading):
    trajectory = tuple(torch.tensor([frame], dtype=torch.float64) for frame in (
        [1., 0.], [.137, .031], [0., 1.],
    ))
    case = _Case("parallel", trajectory, (1, 1), (3,), (.2,), (1., 1.))
    matrix = _matrix(case)  # Three independent box chords through the single voxel.
    input_shape, output_shape = case.shape, case.projection_shape
    if operation == "backproject":
        matrix = matrix.T
        input_shape, output_shape = output_shape, input_shape
    amplitudes = torch.linspace(.3, 1.7, math.prod(leading), dtype=torch.float64).reshape(leading)
    value = amplitudes[(..., None, None)] * _data(input_shape, phase=.37)
    expected = (value.reshape(-1, math.prod(input_shape)) @ matrix.T).reshape(*leading, *output_shape)
    assert value.untyped_storage().nbytes() < WORK_BYTES
    assert expected.untyped_storage().nbytes() < WORK_BYTES
    operator = Projector(
        case.trajectory, case.shape, case.detector, beam=case.beam,
        detector_spacing=case.pitch[0], voxel_spacing=case.spacing,
        backend="torch", precision="float64",
    )
    with _WorkingAllocations(value) as allocations:
        result = getattr(operator, operation)(value)
    assert result.shape == (*leading, *output_shape)
    assert result.dtype == torch.float64 and result.device == value.device
    torch.testing.assert_close(result, expected, rtol=2e-11, atol=2e-11)
    payload, allocation, shape = allocations.largest_intermediate(result)
    assert payload <= WORK_BYTES, (
        f"{allocation} allocated {shape} ({payload} storage bytes); "
        f"automatic intermediates must fit {WORK_BYTES} bytes even at minimum ray/voxel chunks"
    )
