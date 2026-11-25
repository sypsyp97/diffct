"""Real solver recurrences keep CUDA reductions and block updates local."""

import json
import math
from unittest import mock

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import diffct
from diffct import _solver_state
from tests.test_cgls_guard_exact_local import (
    BITS, DEFAULT_BLOCK, FLOATS, LAYOUTS, READ_ONLY, WIDTH, _FreshTensors, _bits, _flip, _guard, _iterate, _rejects,
    _signed,
)
from tests.test_solver_api import _DiskAllocations, _MatrixOperator, _SolverStore, _cgls_reference, _checkpoint


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(
    not torch.cuda.is_available(), reason="device-local solver contracts require CUDA",
)]


class _CudaReads(TorchDispatchMode):
    """Count actual scalar extraction or tensor readback, forwarding all math."""

    def __init__(self):
        super().__init__()
        self.events, self.iterations, self.updates = [], [], []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        name, source = str(function), None
        if name == "aten._local_scalar_dense.default":
            source = args[0]
        elif name == "aten._to_copy.default" and result.device.type == "cpu":
            source = args[0]
        elif name == "aten.copy_.default" and args[0].device.type == "cpu":
            source = args[1]
        elif name in ("aten.equal.default", "aten.is_nonzero.default") and args[0].is_cuda:
            # One host boolean computed on the device: a synchronizing readback.
            source = torch.empty((), dtype=torch.bool, device=args[0].device)
        if isinstance(source, torch.Tensor) and source.is_cuda:
            event = (name, tuple(source.shape), source.numel() * source.element_size())
            self.events.append(event)
            if self.iterations:
                self.iterations[-1].append(event)
        if (name in ("aten.add_.Tensor", "aten.sub_.Tensor") and args[0].is_cuda
                and tuple(args[0].shape) in (_MatrixOperator.volume_shape, _MatrixOperator.projection_shape)):
            self.updates.append((name, tuple(args[0].shape)))
        return result


def _problem(dtype):
    operator = _MatrixOperator(dtype=dtype)
    operator.dtype, operator.backend, operator.precision = dtype, "torch", str(dtype)
    truth = torch.tensor([[-.3, .7, 1.1], [.2, -.15, .9]], dtype=dtype, device="cuda")
    measurements = operator.project(truth)
    operator.calls.clear()
    return operator, measurements


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("stop", ("limit", "callback", "observer", "tolerance"))
def test_resident_cgls_reads_two_reduction_results_plus_one_guard_result_per_iteration(dtype, stop):
    operator, measurements = _problem(dtype)
    original = measurements.clone()
    callbacks = []

    def callback(iteration, residual, volume):
        callbacks.append((iteration, residual, volume.clone()))
        return iteration < 2 if stop == "callback" else None

    options = dict(rtol=.04 if stop == "tolerance" else 0., atol=0.)
    if stop in ("callback", "observer"):
        options["callback"] = callback
    project = operator.project
    backproject = operator.backproject

    def observed(volume):
        # Fresh solves start with A^T r. The first A p marks the loop boundary,
        # excluding request validation and the three initialization norms.
        trace.iterations.append([])
        assert volume.is_cuda, "resident CGLS staged its projection input on CPU"
        return project(volume)

    def observed_adjoint(rays):
        assert rays.is_cuda, "resident CGLS staged its backprojection input on CPU"
        return backproject(rays)

    with _CudaReads() as trace, mock.patch.object(operator, "project", observed), \
            mock.patch.object(operator, "backproject", observed_adjoint):
        result = diffct.cgls(operator, measurements, 20 if stop == "tolerance" else 4, **options)
    assert result.iterations > 0 and len(trace.iterations) == result.iterations
    expected, history, snapshots = _cgls_reference(
        operator.matrix.numpy(), measurements.cpu().numpy(), result.iterations,
    )
    tolerance = dict(rtol=3e-5, atol=3e-6) if dtype == torch.float32 else dict(rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(result.volume.cpu().numpy().ravel(), expected, **tolerance)
    np.testing.assert_allclose(result.residual_history, history, **tolerance)
    torch.testing.assert_close(measurements, original, rtol=0, atol=0)
    assert result.volume.device == measurements.device and result.volume.dtype == dtype
    if stop in ("callback", "observer"):
        assert result.iterations == (2 if stop == "callback" else 4) and not result.converged
        assert [value[0] for value in callbacks] == list(range(1, result.iterations + 1))
        for (_, residual, volume), expected_volume, expected_residual in zip(callbacks, snapshots, history[1:]):
            assert isinstance(residual, float)
            np.testing.assert_allclose(volume.cpu().numpy().ravel(), expected_volume, **tolerance)
            assert residual == pytest.approx(expected_residual, **dict(rel=tolerance["rtol"], abs=tolerance["atol"]))
    elif stop == "tolerance":
        threshold = .04 * np.linalg.norm(measurements.cpu().double().numpy())
        assert result.converged and result.iterations < 20
        assert result.residual_history[-1] <= threshold < result.residual_history[-2]
    else:
        assert result.iterations == 4 and not result.converged
    counts = [len(reads) for reads in trace.iterations]
    assert all(math.prod(shape) <= 2 for _, shape, _ in trace.events), (
        f"resident CGLS copied more than a scalar pair to CPU: {trace.events}"
    )
    # The read-only callback guard may add one readback of at most two values.
    limit = 3 if "callback" in options else 2
    assert all(count <= limit for count in counts), (
        f"resident CGLS made {counts} CUDA readbacks per iteration; maximum is {limit}: {trace.iterations}"
    )


@pytest.mark.parametrize("layout", ("contiguous", "transposed"))
@pytest.mark.parametrize("dtype", FLOATS, ids=str)
@pytest.mark.parametrize("block", (2, 5, DEFAULT_BLOCK))
@pytest.mark.parametrize("mutated", (False, True), ids=("observer", "mutation"))
def test_cuda_guard_reads_back_at_most_one_value_pair(dtype, layout, block, mutated):
    volume = _iterate(dtype, layout, device="cuda")
    before = _bits(volume)

    def callback(iteration, residual, value):
        if mutated:
            _flip(value, (2, 3), 1)

    with _CudaReads() as trace:
        if mutated:
            with pytest.raises(ValueError, match=READ_ONLY):
                _guard(volume, callback, block)
        else:
            assert _guard(volume, callback, block) is False
    assert not mutated or not torch.equal(_bits(volume), before)
    assert len(trace.events) <= 1 and all(math.prod(shape) <= 2 for _, shape, _ in trace.events), (
        f"CUDA guard copied the iterate to CPU: {trace.events}"
    )


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", FLOATS, ids=str)
def test_cuda_guard_rejects_every_single_bit_change_at_every_position(dtype, layout):
    width = WIDTH[BITS[dtype]]
    missed = []
    for flat in range(12):
        index = divmod(flat, 4)
        for bit in range(width):
            block = (2, 4, 6, DEFAULT_BLOCK)[(flat + bit) % 4]
            mask = _signed(1 << bit, width)
            if not _rejects(_iterate(dtype, layout, device="cuda"), lambda value: _flip(value, index, mask), block):
                missed.append((index, bit, block))
    assert not missed, f"{len(missed)} single-bit CUDA iterate changes escaped the guard: {missed[:8]}"


@pytest.mark.parametrize("dtype", FLOATS, ids=str)
def test_cuda_guard_rejects_swaps_signed_zero_and_nan_payload_changes(dtype):
    width = WIDTH[BITS[dtype]]
    integer = BITS[dtype]

    def swap(value):
        bits = value.view(integer)
        left, right = bits[0, 0].clone(), bits[2, 3].clone()
        bits[0, 0], bits[2, 3] = right, left

    def bit_pattern(index, xor):
        return lambda value: value.view(integer)[index].bitwise_xor_(_signed(xor, width))

    changes = {"swap": swap, "+0 to -0": bit_pattern((0, 0), 1 << (width - 1)),
               "-0 to +0": bit_pattern((0, 1), 1 << (width - 1)), "NaN payload": bit_pattern((0, 2), 1)}
    for name, change in changes.items():
        volume = _iterate(dtype, device="cuda")
        if name == "NaN payload":
            assert torch.isnan(volume[0, 2]).item()
        assert _rejects(volume, change, 5), f"{name} escaped the CUDA guard"


@pytest.mark.parametrize("dtype", FLOATS, ids=str)
@pytest.mark.parametrize("means", ("inplace", "data", "dlpack", "cuda-array-interface", "set-storage"))
def test_cuda_guard_rejects_writes_by_any_means(dtype, means):
    volume = _iterate(dtype, "transposed", device="cuda")
    before = _bits(volume)
    integer = BITS[dtype]

    def mutate(value):
        if means == "inplace":
            value.view(integer)[1, 2].bitwise_xor_(1)
        elif means == "data":
            value.data.view(integer)[1, 2].bitwise_xor_(1)
        elif means == "dlpack":
            torch.from_dlpack(value.detach().view(integer))[1, 2].bitwise_xor_(1)
        elif means == "cuda-array-interface":
            from numba import cuda

            alias = cuda.as_cuda_array(value.detach().view(integer))
            alias[1, 2] = int(alias[1, 2]) ^ 1
        else:
            changed = value.detach().clone()
            changed.view(integer)[1, 2].bitwise_xor_(1)
            value.set_(changed)

    if means == "cuda-array-interface":
        pytest.importorskip("numba.cuda")
    assert _rejects(volume, mutate, 5), f"{means} write escaped the CUDA guard"
    assert not torch.equal(_bits(volume), before)


@pytest.mark.parametrize("dtype", FLOATS, ids=str)
@pytest.mark.parametrize("block", (2, 4, 16))
@pytest.mark.parametrize("mutated", (False, True), ids=("observer", "mutation"))
def test_cuda_guard_allocates_at_most_block_elements_per_tensor(dtype, block, mutated):
    volume = _iterate(dtype, "transposed", shape=(8, 8), device="cuda")

    def callback(iteration, residual, value):
        trace.active = False
        if mutated:
            _flip(value, (7, 7), 1)
        trace.active = True

    with _FreshTensors() as trace:
        if mutated:
            with pytest.raises(ValueError, match=READ_ONLY):
                _guard(volume, callback, block)
        else:
            _guard(volume, callback, block)
    oversized = [entry for entry in trace.sizes if entry[-1] > block]
    assert not oversized, f"CUDA guard allocated more than block_elements={block} elements: {oversized}"


@pytest.mark.parametrize("dtype", FLOATS, ids=str)
@pytest.mark.parametrize("mode", ("observer", "bit", "sign", "dlpack", "move-to-cpu", "dtype", "shape"))
def test_resident_cgls_guard_is_exact_before_the_next_projection(dtype, mode):
    operator, measurements = _problem(dtype)
    integer = BITS[dtype]
    boundary = []

    def callback(iteration, residual, volume):
        assert volume.is_cuda
        if iteration == 2 and mode != "observer":
            boundary.append(len(operator.calls))
            if mode == "bit":
                volume.view(integer)[1, 2].bitwise_xor_(1)
            elif mode == "sign":
                volume.view(integer)[0, 0].bitwise_xor_(_signed(1 << (WIDTH[integer] - 1), WIDTH[integer]))
            elif mode == "dlpack":
                torch.from_dlpack(volume.detach().view(integer))[0, 1].bitwise_xor_(1)
            elif mode == "move-to-cpu":
                volume.data = volume.cpu()
            elif mode == "dtype":
                volume.data = volume.view({torch.float16: torch.bfloat16, torch.bfloat16: torch.float16}.get(
                    dtype, integer))
            else:
                volume.data = volume.reshape(3, 2)

    baseline = diffct.cgls(operator, measurements, 4, rtol=0., atol=0.)
    operator.calls.clear()
    if mode == "observer":
        observed = diffct.cgls(operator, measurements, 4, rtol=0., atol=0., callback=callback)
        assert torch.equal(_bits(observed.volume), _bits(baseline.volume))
        assert observed.residual_history == baseline.residual_history
        return
    with pytest.raises(ValueError, match=READ_ONLY):
        diffct.cgls(operator, measurements, 4, rtol=0., atol=0., callback=callback)
    assert boundary == [5] and len(operator.calls) == 5, "iterate change crossed the next projection"


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_cgls_into_cuda_output_rejects_device_change_before_next_projection(dtype, tmp_path):
    class CudaMatrixOperator(_MatrixOperator):
        def _into(self, value, output, *, forward):
            self.calls.append("project_into" if forward else "backproject_into")
            source = diffct.TensorStore(value) if isinstance(value, torch.Tensor) else value
            sink = diffct.TensorStore(output) if isinstance(output, torch.Tensor) else output
            matrix = (self.matrix if forward else self.matrix.T).to(device="cuda", dtype=sink.dtype)
            for row, output_index in enumerate(np.ndindex(sink.shape)):
                result = torch.zeros((), dtype=sink.dtype, device="cuda")
                for column, source_index in enumerate(np.ndindex(source.shape)):
                    selected = tuple(slice(index, index + 1) for index in source_index)
                    result += matrix[row, column] * source.read(selected).reshape(()).to(matrix)
                selected = tuple(slice(index, index + 1) for index in output_index)
                sink.write(selected, result.reshape((1,) * len(sink.shape)))
            return output

    operator = CudaMatrixOperator(dtype=dtype)
    operator.dtype = dtype
    truth = torch.tensor([[-.3, .7, 1.1], [.2, -.15, .9]], dtype=dtype, device="cuda")
    measurements = operator.project(truth)
    operator.calls.clear()
    output = torch.empty(operator.volume_shape, dtype=dtype, device="cuda")
    boundary = []

    def callback(iteration, residual, volume):
        assert volume is output and volume.is_cuda
        if iteration == 2:
            boundary.append(len(operator.calls))
            volume.data = volume.cpu()
            assert volume.device.type == "cpu"

    with pytest.raises(ValueError, match=READ_ONLY):
        diffct.cgls_into(operator, measurements, output, 4, workspace=tmp_path / "workspace",
                         block_elements=3, rtol=0., atol=0., callback=callback)
    assert boundary == [5] and len(operator.calls) == 5, "device change crossed the next numerical call"


@pytest.mark.parametrize("kind", ("denominator-zero", "initial-zero-gamma", "terminal-zero-gamma"))
def test_cgls_breakdown_preserves_state_before_invalid_update(kind):
    operator, _ = _problem(torch.float32)
    operator.matrix.zero_()
    if kind != "initial-zero-gamma":
        operator.matrix[:6].copy_(torch.eye(6) * (1e-30 if kind == "denominator-zero" else 1.))
    measurements = torch.ones(operator.projection_shape, device="cuda")
    callbacks = []
    with _CudaReads() as trace:
        result = diffct.cgls(operator, measurements, 4, rtol=0., atol=0.,
                            callback=lambda iteration, residual, volume: callbacks.append(iteration))
    completed = int(kind == "terminal-zero-gamma")
    assert result.iterations == completed and result.converged == (kind != "denominator-zero")
    assert result.reason == ("normal equation breakdown" if kind == "denominator-zero"
                             else "normal equations satisfied")
    assert callbacks == ([1] if completed else [])
    expected_calls = (["backproject", "project", "backproject"] if completed else
                      ["backproject", "project"] if kind == "denominator-zero" else ["backproject"])
    assert operator.calls == expected_calls
    torch.testing.assert_close(result.volume, torch.full_like(result.volume, float(completed)), rtol=0, atol=0)
    torch.testing.assert_close(measurements, torch.ones_like(measurements), rtol=0, atol=0)
    assert result.residual_history == ([math.sqrt(15), 3.] if completed else [math.sqrt(15)])
    if not completed:
        assert not trace.updates, "breakdown mutated x or residual before validating the denominator/gamma"


def test_cuda_cgls_checkpoint_gamma_and_resume_are_exact(tmp_path):
    operator, measurements = _problem(torch.float64)
    options = dict(rtol=0., atol=0.)
    full = diffct.cgls(operator, measurements, 4, **options)
    partial = diffct.cgls(operator, measurements, 2, checkpoint_directory=tmp_path / "checkpoint", **options)
    saved = _checkpoint(tmp_path / "checkpoint", 2)
    metadata = json.loads((saved / "metadata.json").read_text())
    adjoint = torch.from_numpy(np.load(saved / "s.npy")).cuda()
    assert metadata["gamma"] == adjoint.double().square().sum().item()
    assert metadata["residual_history"] == partial.residual_history
    resumed = diffct.cgls(operator, measurements, 4, resume=saved, **options)
    torch.testing.assert_close(resumed.volume, full.volume, rtol=0, atol=0)
    assert resumed.residual_history == full.residual_history and resumed.iterations == full.iterations


def test_cuda_squared_norm_retains_sequential_fp64_block_reduction_order():
    # FP64 ULP at 2**54 is four. Each subsequent two-element block sums to two
    # and rounds away; combining all blocks first would retain the eight ones.
    values = torch.tensor([2.**27, 0., *([1.] * 8)], device="cuda", dtype=torch.float64)
    original = values.clone()
    reads = []
    read = diffct.TensorStore.read

    def bounded(store, selected):
        result = read(store, selected)
        reads.append(result.numel())
        assert result.numel() <= 2
        return result

    with mock.patch.object(diffct.TensorStore, "read", bounded):
        actual = _solver_state.squared_norm(_MatrixOperator(), values, "volume", elements=2)
    assert actual == 2.**54 and reads == [2] * 5
    torch.testing.assert_close(values, original, rtol=0, atol=0)


def test_device_result_squared_norm_keeps_fp64_order_without_host_read():
    values = torch.tensor([2.**27, 0., *([1.] * 8)], device="cuda", dtype=torch.float64)
    original = values.clone()
    reads = []
    read = diffct.TensorStore.read

    def bounded(store, selected):
        result = read(store, selected)
        reads.append(result.numel())
        assert result.numel() <= 2
        return result

    with _CudaReads() as trace, mock.patch.object(diffct.TensorStore, "read", bounded):
        actual = _solver_state.squared_norm(
            _MatrixOperator(), values, "volume", elements=2, return_tensor=True,
        )
    assert isinstance(actual, torch.Tensor), "device-result norm returned a host scalar"
    assert actual.device == values.device and actual.dtype == torch.float64 and actual.ndim == 0
    assert actual.item() == 2.**54 and reads == [2] * 5
    torch.testing.assert_close(values, original, rtol=0, atol=0)
    assert not trace.events, f"device-result norm read values on the host: {trace.events}"


def test_device_result_scalar_sum_keeps_cuda_value_without_host_read():
    value = torch.tensor(.37, dtype=torch.float64, device="cuda")
    with _CudaReads() as trace:
        actual = _solver_state.scalar_sum(_MatrixOperator(), value, "volume", return_tensor=True)
    assert isinstance(actual, torch.Tensor), "device-result scalar sum returned a host scalar"
    assert actual.device == value.device and actual.dtype == torch.float64 and actual.ndim == 0
    torch.testing.assert_close(actual, value, rtol=0, atol=0)
    assert not trace.events, f"device-result scalar sum read values on the host: {trace.events}"


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("reduction", ("mean", "sum"))
def test_single_rank_cuda_total_variation_has_no_host_read(dtype, reduction):
    operator = _MatrixOperator(dtype=dtype)
    volume = torch.tensor([[-.2, .7, 1.1], [.4, -.1, .9]], device="cuda", dtype=dtype, requires_grad=True)
    options = dict(voxel_spacing=(.6, .9), reduction=reduction)
    expected = diffct.total_variation(volume, **options)
    expected_gradient, = torch.autograd.grad(expected, volume)
    with _CudaReads() as trace:
        actual = diffct.total_variation(volume, operator=operator, **options)
        gradient, = torch.autograd.grad(actual, volume)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
    assert actual.device == volume.device and actual.dtype == volume.dtype
    assert not trace.events, f"single-rank CUDA TV read values on the host: {trace.events}"


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_cuda_update_blocks_keeps_target_local_with_bounded_mapped_source(dtype, tmp_path):
    target = torch.linspace(-.9, 1.7, 35, dtype=dtype, device="cuda").reshape(7, 5).T
    values = np.linspace(.2, 2.1, target.numel()).reshape(target.shape)
    np.save(tmp_path / "source.npy", values)
    source = _SolverStore(diffct.NpyStore(tmp_path / "source.npy", mode="r"), 4)
    sink = diffct.TensorStore(target)
    expected = target.clone().mul_(.37).add_(torch.from_numpy(values).to(target), alpha=-.21)
    writes = []
    write = diffct.TensorStore.write

    def bounded(store, selected, value, **kwargs):
        assert value.numel() <= 4
        writes.append((value.device, value.dtype, value.numel()))
        return write(store, selected, value, **kwargs)

    with _CudaReads() as trace, _DiskAllocations(4), mock.patch.object(diffct.TensorStore, "write", bounded):
        _solver_state.update_blocks(sink, source, -.21, elements=4, scale=.37)
    torch.testing.assert_close(target, expected, rtol=2e-7 if dtype == torch.float32 else 2e-15,
                               atol=3e-8 if dtype == torch.float32 else 2e-16)
    np.testing.assert_array_equal(np.load(tmp_path / "source.npy"), values)
    assert writes and sum(size for _, _, size in writes) == target.numel()
    assert source.reads and len(source.reads) == len(writes)
    assert not trace.events, f"CUDA target blocks were staged on CPU: {trace.events}"
    assert all(device == target.device and actual_dtype == dtype for device, actual_dtype, _ in writes), (
        f"CUDA target blocks were updated on another device: {writes}"
    )
