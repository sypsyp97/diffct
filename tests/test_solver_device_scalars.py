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
@pytest.mark.parametrize("stop", ("limit", "callback", "tolerance"))
def test_resident_cgls_reads_at_most_two_reduction_results_per_iteration(dtype, stop):
    operator, measurements = _problem(dtype)
    original = measurements.clone()
    callbacks = []

    def callback(iteration, residual, volume):
        callbacks.append((iteration, residual, volume.clone()))
        return iteration < 2

    options = dict(rtol=.04 if stop == "tolerance" else 0., atol=0.)
    if stop == "callback":
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
    if stop == "callback":
        assert result.iterations == 2 and not result.converged
        assert [value[0] for value in callbacks] == [1, 2]
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
    assert all(count <= 2 for count in counts), (
        f"resident CGLS made {counts} CUDA reduction readbacks per iteration; maximum is 2"
    )


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
    assert result.iterations == completed and not result.converged
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
