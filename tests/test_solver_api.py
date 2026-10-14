"""Public numerical solvers: independent matrix recurrences and genuine disk state."""

from contextlib import ExitStack
from datetime import timedelta
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import time
from unittest import mock

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch.utils._pytree import tree_flatten

import diffct
from tests.test_detector_surfaces import _case, _grids, _matrix
from tests.test_block_cgls import _SolverAllocations
from tests.test_block_storage import _numpy_blocks
from tests.test_execution_features import (
    GLOO, _api, _hybrid, _no_whole_volume_collectives, _random, _run_four, _tv_oracle,
)


ALGORITHMS = ("cgls", "sirt", "tv_reconstruction")


class _MatrixOperator:
    """A real positive linear map; no solver or projector mathematics mocked."""
    world_size, rank, process_group, partition = 1, 0, None, "views"
    beam, voxel_spacing, detector_spacing = "parallel", .73, (.91,)
    volume_shape, global_volume_shape, local_volume_shape = (2, 3), (2, 3), (2, 3)
    projection_shape, detector_shape, view_slice = (5, 3), (3,), slice(0, 5)
    volume_slice = (slice(0, 2), slice(0, 3))
    partition_grid = None

    def __init__(self, *, dtype=torch.float32):
        matrix = torch.arange(90, dtype=dtype).reshape(15, 6).remainder(11) * .013 + .025
        matrix[:6] += torch.diag(torch.tensor([.8, 1.1, .9, 1.4, 1.2, 1.5], dtype=dtype))
        self.matrix = matrix
        self._trajectory = tuple(value.to(dtype) for value in _case("parallel", views=5).trajectory)
        self.calls = []

    def project(self, volume):
        self.calls.append("project")
        return (self.matrix.to(volume) @ volume.reshape(-1)).reshape(self.projection_shape)

    def backproject(self, sinogram):
        self.calls.append("backproject")
        return (self.matrix.to(sinogram).T @ sinogram.reshape(-1)).reshape(self.volume_shape)

    def _into(self, value, output, *, forward):
        self.calls.append("project_into" if forward else "backproject_into")
        source = diffct.TensorStore(value) if isinstance(value, torch.Tensor) else value
        sink = diffct.TensorStore(output) if isinstance(output, torch.Tensor) else output
        matrix = self.matrix if forward else self.matrix.T
        # Element blocks keep the oracle compatible with any block_elements.
        for row, output_index in enumerate(np.ndindex(sink.shape)):
            result = torch.zeros((), dtype=sink.dtype)
            for column, source_index in enumerate(np.ndindex(source.shape)):
                selected = tuple(slice(index, index + 1) for index in source_index)
                result += matrix[row, column].to(sink.dtype) * source.read(selected).reshape(()).to(sink.dtype)
            selected = tuple(slice(index, index + 1) for index in output_index)
            sink.write(selected, result.reshape((1,) * len(sink.shape)))
        return output

    def project_into(self, value, output):
        return self._into(value, output, forward=True)

    def backproject_into(self, value, output):
        return self._into(value, output, forward=False)


def _fixture():
    operator = _MatrixOperator()
    truth = torch.tensor([[-.3, .7, 1.1], [.2, -.15, .9]], dtype=torch.float32)
    measurements = operator.project(truth).detach()
    operator.calls.clear()
    return operator, measurements, truth


def _cgls_reference(matrix, data, iterations, x0=None):
    matrix, data = np.asarray(matrix, dtype=np.float64), np.asarray(data, dtype=np.float64).ravel()
    x = np.zeros(matrix.shape[1]) if x0 is None else np.asarray(x0, dtype=np.float64).ravel().copy()
    residual = data - matrix @ x
    direction = matrix.T @ residual
    gamma = direction @ direction
    history, snapshots = [np.linalg.norm(residual)], []
    for _ in range(iterations):
        projected = matrix @ direction
        denominator = projected @ projected
        if gamma == 0 or denominator == 0:
            break
        alpha = gamma / denominator
        x += alpha * direction
        residual -= alpha * projected
        adjoint = matrix.T @ residual
        next_gamma = adjoint @ adjoint
        direction = adjoint + (next_gamma / gamma) * direction
        gamma = next_gamma
        history.append(np.linalg.norm(residual))
        snapshots.append(x.copy())
    return x, history, snapshots


def _sirt_reference(matrix, data, iterations, x0=None, relaxation=1., nonnegative=True):
    matrix, data = np.asarray(matrix, dtype=np.float64), np.asarray(data, dtype=np.float64).ravel()
    x = np.zeros(matrix.shape[1]) if x0 is None else np.asarray(x0, dtype=np.float64).ravel().copy()
    row, column = matrix.sum(1), matrix.sum(0)
    row_inverse = np.divide(1., row, out=np.zeros_like(row), where=row > 0)
    column_inverse = np.divide(1., column, out=np.zeros_like(column), where=column > 0)
    history, snapshots = [np.linalg.norm(data - matrix @ x)], []
    for _ in range(iterations):
        x += relaxation * column_inverse * (matrix.T @ (row_inverse * (data - matrix @ x)))
        if nonnegative:
            x = np.maximum(x, 0)
        history.append(np.linalg.norm(data - matrix @ x))
        snapshots.append(x.copy())
    return x, history, snapshots


def _result(result, *, iterations=None, output=None):
    result_type = getattr(diffct, "SolverResult", None)
    assert result_type is not None, "INTERFACE_PENDING: diffct.SolverResult"
    assert isinstance(result, result_type)
    assert isinstance(result.iterations, int) and result.iterations >= 0
    if iterations is not None:
        assert result.iterations == iterations
    if output is not None:
        assert result.volume is output
    assert isinstance(result.converged, bool)
    assert isinstance(result.reason, str) and result.reason.strip()
    assert len(result.residual_history) == result.iterations + 1
    assert all(math.isfinite(float(value)) and float(value) >= 0 for value in result.residual_history)
    return result.volume


@pytest.mark.parametrize("name,default", [("cgls", 30), ("sirt", 100), ("tv_reconstruction", 100)])
def test_public_solver_defaults_and_controls(name, default):
    signature = inspect.signature(_api(name))
    assert signature.parameters["iterations"].default == default
    for key, value in dict(x0=None, rtol=1e-6, atol=0., callback=None,
                           checkpoint_directory=None, resume=None).items():
        assert key in signature.parameters, f"INTERFACE_PENDING: {name}.{key}"
        assert signature.parameters[key].default == value


@pytest.mark.parametrize("name", ["cgls", "sirt"])
def test_matrix_solver_matches_independent_recurrence_and_records_initial_plus_completed_residuals(name):
    operator, measurements, _ = _fixture()
    x0 = torch.tensor([[.2, -.1, .3], [.1, -.2, .05]])
    original, initial = measurements.clone(), x0.clone()
    expected, history, _ = (_cgls_reference if name == "cgls" else _sirt_reference)(
        operator.matrix.numpy(), measurements.numpy(), 4, x0.numpy())
    result = _api(name)(operator, measurements.requires_grad_(), 4, x0=x0, rtol=0., atol=0.)
    actual = _result(result, iterations=4)
    np.testing.assert_allclose(actual.detach().numpy().ravel(), expected, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(result.residual_history, history, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(measurements.detach(), original, rtol=0, atol=0)
    torch.testing.assert_close(x0, initial, rtol=0, atol=0)
    assert measurements.grad is None and not actual.requires_grad and actual.grad_fn is None


@pytest.mark.parametrize("name", ALGORITHMS)
def test_solver_callback_receives_completed_iteration_current_residual_and_volume_and_false_stops(name):
    operator, measurements, _ = _fixture()
    calls = []
    def callback(iteration, residual, volume):
        observed = torch.linalg.vector_norm(measurements - operator.project(volume)).item()
        assert float(residual) == pytest.approx(observed, rel=3e-5, abs=3e-6)
        calls.append((iteration, float(residual), volume.detach().clone()))
        return iteration < 2
    result = _api(name)(operator, measurements, 8, callback=callback, rtol=0., atol=0.)
    actual = _result(result, iterations=2)
    assert [item[0] for item in calls] == [1, 2]
    torch.testing.assert_close(actual, calls[-1][2], rtol=0, atol=0)
    np.testing.assert_allclose(result.residual_history[1:], [item[1] for item in calls], rtol=3e-5, atol=3e-6)
    assert not result.converged


@pytest.mark.parametrize("name", ALGORITHMS)
@pytest.mark.parametrize("mode", ["zero", "relative", "absolute", "warm-start"])
def test_solver_stops_at_absolute_plus_relative_measurement_norm_including_initial_residual(name, mode):
    operator, measurements, truth = _fixture()
    options = {}
    if mode == "zero":
        measurements = torch.zeros_like(measurements)
    elif mode == "relative":
        options["rtol"] = 1.01
    elif mode == "absolute":
        options.update(rtol=0., atol=torch.linalg.vector_norm(measurements).item() + .01)
    else:
        options["x0"] = truth
    result = _api(name)(operator, measurements, 7, **options)
    actual = _result(result, iterations=0)
    assert result.converged
    torch.testing.assert_close(actual, truth if mode == "warm-start" else torch.zeros_like(truth),
                               rtol=1e-5, atol=1e-6)


def test_cgls_convergence_during_iteration_is_residual_based():
    operator, measurements, _ = _fixture()
    threshold = .04 * torch.linalg.vector_norm(measurements).item()
    result = _api("cgls")(operator, measurements, 20, rtol=.04)
    actual = _result(result)
    assert 0 < result.iterations < 20 and result.converged
    actual_residual = torch.linalg.vector_norm(measurements - operator.project(actual)).item()
    assert actual_residual <= threshold
    assert result.residual_history[-1] == pytest.approx(actual_residual, rel=3e-5, abs=3e-6)
    assert result.residual_history[-2] > threshold


@pytest.mark.parametrize("name", ALGORITHMS)
@pytest.mark.parametrize("option", [dict(iterations=-1), dict(iterations=True), dict(iterations=1.5),
                                  dict(rtol=-1), dict(rtol=float("nan")), dict(atol=-1),
                                  dict(atol=float("inf")), dict(callback=7), dict(x0=torch.zeros(1))])
def test_solver_validates_controls_before_operator_execution_or_input_mutation(name, option):
    function = _api(name)
    operator, measurements, _ = _fixture()
    original = measurements.clone()
    with pytest.raises((ValueError, TypeError)):
        function(operator, measurements, **option)
    assert operator.calls == []
    torch.testing.assert_close(measurements, original, rtol=0, atol=0)


@pytest.mark.parametrize("relaxation,nonnegative", [(.35, True), (.7, False)])
def test_sirt_row_column_normalization_relaxation_and_optional_nonnegativity(relaxation, nonnegative):
    operator, measurements, truth = _fixture()
    x0 = truth * .8
    expected, history, _ = _sirt_reference(operator.matrix.numpy(), measurements.numpy(), 3,
                                          x0.numpy(), relaxation, nonnegative)
    result = _api("sirt")(operator, measurements, 3, x0=x0, relaxation=relaxation,
                          nonnegative=nonnegative, rtol=0., atol=0.)
    actual = _result(result, iterations=3)
    np.testing.assert_allclose(actual.numpy().ravel(), expected, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(result.residual_history, history, rtol=3e-5, atol=3e-6)
    assert bool((actual < 0).any()) is (not nonnegative)


@pytest.mark.parametrize("name,option", [("sirt", dict(relaxation=0)), ("sirt", dict(relaxation=float("nan"))),
                                       ("sirt", dict(nonnegative="yes")),
                                       ("tv_reconstruction", dict(weight=-1)),
                                       ("tv_reconstruction", dict(weight=float("nan"))),
                                       ("tv_reconstruction", dict(lr=0))])
def test_algorithm_specific_controls_validate_before_computing(name, option):
    function = _api(name)
    operator, measurements, _ = _fixture()
    with pytest.raises((ValueError, TypeError)):
        function(operator, measurements, **option)
    assert operator.calls == []


def test_tv_reconstruction_uses_adam_global_mean_data_term_and_spacing_aware_tv():
    operator, measurements, _ = _fixture()
    x0 = torch.full(operator.volume_shape, .25)
    expected = x0.clone().requires_grad_()
    optimizer = torch.optim.Adam([expected], lr=.037)
    history = [torch.linalg.vector_norm(operator.matrix @ expected.detach().flatten() - measurements.flatten()).item()]
    for _ in range(3):
        optimizer.zero_grad()
        residual = operator.matrix @ expected.flatten() - measurements.flatten()
        loss = .5 * residual.square().mean() + .11 * _tv_oracle(expected, operator.voxel_spacing)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            expected.clamp_(min=0)
        history.append(torch.linalg.vector_norm(operator.matrix @ expected.detach().flatten() - measurements.flatten()).item())
    result = _api("tv_reconstruction")(operator, measurements, 3, weight=.11, lr=.037,
                                      x0=x0, rtol=0., atol=0.)
    actual = _result(result, iterations=3)
    torch.testing.assert_close(actual, expected.detach(), rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(result.residual_history, history, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(x0, torch.full_like(x0, .25), rtol=0, atol=0)


class _SolverStore:
    def __init__(self, store, limit):
        self.store, self.limit = store, limit
        self.reads, self.writes = [], []

    def __getattr__(self, name):
        return getattr(self.store, name)

    def _selection(self, index):
        index = (index,) if isinstance(index, slice) else tuple(index)
        index += (slice(None),) * (len(self.shape) - len(index))
        count = math.prod(len(range(*item.indices(size))) for item, size in zip(index, self.shape))
        assert count <= self.limit, "disk solver read/update/copy exceeded block_elements"
        return index

    def read(self, index):
        index = self._selection(index)
        self.reads.append(index)
        return self.store.read(index)

    def write(self, index, value, *, accumulate=False):
        index = self._selection(index)
        self.writes.append(index)
        return self.store.write(index, value, accumulate=accumulate)

    def flush(self):
        return self.store.flush()


class _DiskAllocations(_SolverAllocations):
    """Bound fresh working tensors in every floating dtype, including fingerprints."""
    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        inputs = {(value.device, value.untyped_storage()._cdata)
                  for value in tree_flatten((args, kwargs or {}))[0] if isinstance(value, torch.Tensor)}
        result = super().__torch_dispatch__(function, types, args, kwargs)
        for value in tree_flatten(result)[0]:
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                fresh = (value.device, value.untyped_storage()._cdata) not in inputs
                assert not fresh or value.numel() <= self.maximum, \
                    ("unbounded fresh disk-solver tensor", str(function), tuple(value.shape), str(value.dtype))
        return result


@pytest.mark.parametrize("kind", ["tensor", "npy"])
def test_cgls_into_returns_supplied_output_and_solves_with_bounded_real_disk_state(kind, tmp_path, monkeypatch):
    function = _api("cgls_into")
    operator, measurements, _ = _fixture()
    np.save(tmp_path / "measurements.npy", measurements.numpy())
    source = _SolverStore(diffct.NpyStore(tmp_path / "measurements.npy", mode="r"), 5)
    raw_output = (diffct.NpyStore.create(tmp_path / "output.npy", operator.volume_shape)
                  if kind == "npy" else torch.full(operator.volume_shape, torch.nan))
    output = (_SolverStore(raw_output, 5) if kind == "npy"
              else _SolverStore(diffct.TensorStore(raw_output), 5))
    original_read, original_write = diffct.NpyStore.read, diffct.NpyStore.write
    def read(store, index):
        value = original_read(store, index)
        assert value.numel() <= 5, "complete disk workspace state read"
        return value
    def write(store, index, value, **kwargs):
        assert value.numel() <= 5, "complete disk workspace state updated"
        return original_write(store, index, value, **kwargs)
    monkeypatch.setattr(diffct.NpyStore, "read", read)
    monkeypatch.setattr(diffct.NpyStore, "write", write)
    with _DiskAllocations(5, forbid_full=(operator.volume_shape, operator.projection_shape, (6,), (15,))), \
            _numpy_blocks(5):
        result = function(operator, source, output, 4, workspace=tmp_path / "workspace",
                          block_elements=5, rtol=0., atol=0.)
    _result(result, iterations=4, output=output)
    expected, history, _ = _cgls_reference(operator.matrix.numpy(), measurements.numpy(), 4)
    actual = torch.from_numpy(np.load(tmp_path / "output.npy").copy()) if kind == "npy" else raw_output
    np.testing.assert_allclose(actual.numpy().ravel(), expected, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(result.residual_history, history, rtol=3e-5, atol=3e-6)
    assert source.reads and output.writes
    assert list((tmp_path / "workspace").glob("*.npy")), "solver did not create genuine mapped state"
    np.testing.assert_array_equal(np.load(tmp_path / "measurements.npy"), measurements.numpy())


@pytest.mark.parametrize("problem", ["iterations", "block", "tolerance", "shape", "readonly", "workspace"])
def test_cgls_into_prevalidates_before_mutating_output_or_creating_workspace(problem, tmp_path):
    function = _api("cgls_into")
    operator, measurements, _ = _fixture()
    raw = torch.full(operator.volume_shape, 97.)
    output = _SolverStore(diffct.TensorStore(raw), 5)
    workspace = tmp_path / "workspace"
    options = dict(iterations=2, block_elements=5, workspace=workspace)
    if problem == "iterations":
        options["iterations"] = -1
    elif problem == "block":
        options["block_elements"] = 0
    elif problem == "tolerance":
        options["rtol"] = float("nan")
    elif problem == "shape":
        output = _SolverStore(diffct.TensorStore(torch.full((1, 1), 97.)), 5)
    elif problem == "readonly":
        np.save(tmp_path / "readonly.npy", raw.numpy())
        output = _SolverStore(diffct.NpyStore(tmp_path / "readonly.npy", mode="r"), 5)
    else:
        workspace.mkdir()
        (workspace / "user-data.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises((ValueError, TypeError, PermissionError)):
        function(operator, measurements, output, **options)
    assert operator.calls == [] and output.reads == [] and output.writes == []
    torch.testing.assert_close(raw, torch.full_like(raw, 97.), rtol=0, atol=0)
    if problem == "workspace":
        assert [item.name for item in workspace.iterdir()] == ["user-data.txt"]
    else:
        assert not workspace.exists()


@pytest.mark.parametrize("mode", ["callback", "relative", "absolute"])
def test_disk_cgls_controls_stop_at_completed_iteration_and_keep_original_output(mode, tmp_path):
    function = _api("cgls_into")
    operator, measurements, _ = _fixture()
    output = diffct.NpyStore.create(tmp_path / "output.npy", operator.volume_shape)
    calls = []
    options = dict(rtol=0., atol=0.)
    if mode == "callback":
        def callback(iteration, residual, volume):
            assert volume is output
            expected, history, _ = _cgls_reference(operator.matrix.numpy(), measurements.numpy(), iteration)
            np.testing.assert_allclose(np.load(output.path).ravel(), expected, rtol=3e-5, atol=3e-6)
            assert float(residual) == pytest.approx(history[-1], rel=3e-5, abs=3e-6)
            calls.append(iteration)
            return iteration < 2
        options["callback"] = callback
    elif mode == "relative":
        options["rtol"] = .04
    else:
        options["atol"] = .04 * torch.linalg.vector_norm(measurements).item()
    result = function(operator, measurements, output, 20, workspace=tmp_path / "work",
                      block_elements=5, **options)
    _result(result, output=output)
    if mode == "callback":
        assert calls == [1, 2] and result.iterations == 2 and not result.converged
    else:
        assert 0 < result.iterations < 20 and result.converged
        assert result.residual_history[-1] <= .04 * torch.linalg.vector_norm(measurements).item()
        assert result.residual_history[-2] > .04 * torch.linalg.vector_norm(measurements).item()


def _checkpoint(directory, iteration, rank=0):
    return directory / f"step-{iteration:06d}" / f"rank-{rank:05d}"


@pytest.mark.parametrize("name", ALGORITHMS)
def test_solver_checkpoint_resume_matches_uninterrupted_solution_and_complete_residual_history(name, tmp_path):
    operator, measurements, _ = _fixture()
    function = _api(name)
    options = dict(rtol=0., atol=0.)
    if name == "tv_reconstruction":
        options.update(weight=.11, lr=.037)
    expected = function(operator, measurements, 5, **options)
    partial = function(operator, measurements, 2, checkpoint_directory=tmp_path / "checkpoint", **options)
    _result(partial, iterations=2)
    saved = _checkpoint(tmp_path / "checkpoint", 2)
    assert (saved / "metadata.json").is_file()
    resumed = function(operator, measurements, 5, resume=saved, **options)
    actual = _result(resumed, iterations=5)
    torch.testing.assert_close(actual, _result(expected, iterations=5), rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(resumed.residual_history, expected.residual_history, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("change", ["measurements", "geometry", "spacing", "dtype", "algorithm", "target"])
def test_checkpoint_fingerprint_rejects_same_shape_changes_before_operator_computation(change, tmp_path):
    operator, measurements, _ = _fixture()
    function = _api("cgls")
    function(operator, measurements, 2, rtol=0., atol=0., checkpoint_directory=tmp_path / "checkpoint")
    saved = _checkpoint(tmp_path / "checkpoint", 2)
    target = 4
    if change == "measurements":
        measurements = measurements.clone()
        measurements[0, 0] += .03
    elif change == "geometry":
        operator._trajectory[1][0, 0] += .03
    elif change == "spacing":
        operator.voxel_spacing *= 1.1
    elif change == "dtype":
        measurements = measurements.double()
    elif change == "algorithm":
        function = _api("sirt")
    else:
        target = 1
    operator.calls.clear()
    with pytest.raises((ValueError, TypeError)):
        function(operator, measurements, target, rtol=0., atol=0., resume=saved)
    assert operator.calls == [], "resume validation occurred after executing changed scan"


@pytest.mark.parametrize("name,changed", [("sirt", dict(relaxation=.6)), ("sirt", dict(nonnegative=False)),
                                        ("tv_reconstruction", dict(weight=.2)),
                                        ("tv_reconstruction", dict(lr=.037))])
def test_checkpoint_rejects_changed_algorithm_controls_before_computation(name, changed, tmp_path):
    operator, measurements, _ = _fixture()
    function = _api(name)
    function(operator, measurements, 2, rtol=0., atol=0., checkpoint_directory=tmp_path / "checkpoint")
    saved = _checkpoint(tmp_path / "checkpoint", 2)
    operator.calls.clear()
    with pytest.raises((ValueError, TypeError)):
        function(operator, measurements, 4, rtol=0., atol=0., resume=saved, **changed)
    assert operator.calls == [], "algorithm controls were rejected after computing a resumed iteration"


def test_disk_cgls_checkpoint_resume_matches_uninterrupted_result_and_preserves_output_identity(tmp_path):
    function = _api("cgls_into")
    operator, measurements, _ = _fixture()
    np.save(tmp_path / "measurements.npy", measurements.numpy())
    source = diffct.NpyStore(tmp_path / "measurements.npy", mode="r")
    uninterrupted = diffct.NpyStore.create(tmp_path / "full.npy", operator.volume_shape)
    function(operator, source, uninterrupted, 5, workspace=tmp_path / "full-work", block_elements=5,
             rtol=0., atol=0.)
    partial = diffct.NpyStore.create(tmp_path / "partial.npy", operator.volume_shape)
    function(operator, source, partial, 2, workspace=tmp_path / "partial-work", block_elements=5,
             rtol=0., atol=0., checkpoint_directory=tmp_path / "checkpoint")
    resumed_output = diffct.NpyStore.create(tmp_path / "resumed.npy", operator.volume_shape)
    result = function(operator, source, resumed_output, 5, workspace=tmp_path / "resume-work", block_elements=5,
                      rtol=0., atol=0., resume=_checkpoint(tmp_path / "checkpoint", 2))
    _result(result, iterations=5, output=resumed_output)
    np.testing.assert_allclose(np.load(resumed_output.path), np.load(uninterrupted.path), rtol=3e-5, atol=3e-6)


def test_disk_checkpoint_data_fingerprint_is_block_bounded_and_rejects_before_output_mutation(tmp_path):
    function = _api("cgls_into")
    operator, measurements, _ = _fixture()
    np.save(tmp_path / "measurements.npy", measurements.numpy())
    source = _SolverStore(diffct.NpyStore(tmp_path / "measurements.npy"), 5)
    first = diffct.NpyStore.create(tmp_path / "first.npy", operator.volume_shape)
    function(operator, source, first, 2, workspace=tmp_path / "first-work", block_elements=5,
             rtol=0., atol=0., checkpoint_directory=tmp_path / "checkpoint")
    source.store.write((slice(0, 1), slice(0, 1)), measurements[:1, :1] + .1)
    output = torch.full(operator.volume_shape, 83.)
    operator.calls.clear()
    with pytest.raises((ValueError, TypeError)):
        function(operator, source, output, 4, workspace=tmp_path / "resume-work", block_elements=5,
                 rtol=0., atol=0., resume=_checkpoint(tmp_path / "checkpoint", 2))
    assert operator.calls == [] and not (tmp_path / "resume-work").exists()
    torch.testing.assert_close(output, torch.full_like(output, 83.), rtol=0, atol=0)


def test_checkpoint_state_write_failure_leaves_no_committed_metadata(tmp_path, monkeypatch):
    function = _api("cgls_into")
    operator, measurements, _ = _fixture()
    output = diffct.NpyStore.create(tmp_path / "output.npy", operator.volume_shape)
    original = diffct.NpyStore.write
    def fail(store, selected, value, **kwargs):
        if store.path.parent.name == "rank-00000" and store.path.name == "q.npy":
            raise OSError("test checkpoint state write interrupted")
        return original(store, selected, value, **kwargs)
    monkeypatch.setattr(diffct.NpyStore, "write", fail)
    with pytest.raises(OSError):
        function(operator, measurements, output, 2, workspace=tmp_path / "work", block_elements=5,
                 rtol=0., atol=0., checkpoint_directory=tmp_path / "checkpoint")
    assert list((tmp_path / "checkpoint").rglob("metadata.json")) == []


class _CheckpointWrites:
    """Real per-file write coverage and flush records visible to all test ranks."""
    def __init__(self, root, case, rank):
        self.root, self.case, self.rank = root.resolve(), case, rank
        self.states, self.commits = {}, set()
        self.original_os_replace = os.replace
        self.original_path_replace = Path.replace

    def _state(self, path):
        path = Path(path).resolve()
        checkpoint = self.root / "checkpoint"
        return (path.is_relative_to(checkpoint) and path.parent.name.startswith("rank-")
                and path.parent.parent.name.startswith("step-") and path.suffix == ".npy")

    def _marker(self, path):
        return (self.root / "checkpoint-observations" / path.parent.parent.name
                / path.parent.name / (path.name + ".json"))

    def _record(self, path, shape):
        if path not in self.states:
            self.states[path] = dict(coverage=np.zeros(shape, dtype=bool), flushed=False, regions=0)
        return self.states[path]

    def _save(self, path):
        state = self.states[path]
        marker = self._marker(path)
        marker.parent.mkdir(parents=True, exist_ok=True)
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(shape=list(state["coverage"].shape),
                                             written=int(state["coverage"].sum()),
                                             regions=state["regions"], flushed=state["flushed"])), encoding="utf-8")
        self.original_os_replace(temporary, marker)

    def _before_replace(self, source, target):
        target = Path(target).resolve()
        if target.name != "metadata.json" or not target.is_relative_to(self.root / "checkpoint"):
            return
        document = json.loads(Path(source).read_text(encoding="utf-8"))
        assert document["iteration"] == int(target.parent.parent.name.removeprefix("step-")), \
            "metadata publication iteration does not match completed checkpoint step"
        step = target.parent.parent
        for peer in range(4):
            view_rank, space_rank = divmod(peer, 2)
            first, extra = divmod(self.case.shape[0], 2)
            volume_shape = (first + int(space_rank < extra), *self.case.shape[1:])
            views, extra_views = divmod(self.case.views, 2)
            ray_shape = (views + int(view_rank < extra_views), *self.case.detector)
            for name in ("x", "s", "p", "r", "q"):
                state_path = step / f"rank-{peer:05d}" / f"{name}.npy"
                marker = self._marker(state_path)
                assert state_path.is_file() and marker.is_file(), \
                    "metadata committed before every rank completed actual checkpoint state writes"
                observed = json.loads(marker.read_text(encoding="utf-8"))
                expected_shape = volume_shape if name in ("x", "s", "p") else ray_shape
                assert observed["shape"] == list(expected_shape)
                assert observed["written"] == math.prod(expected_shape) and observed["flushed"], \
                    ("metadata committed before all state regions were written and flushed", peer, name, observed)
        self.commits.add(str(target))

    def _after_replace(self, source, target):
        source, target = Path(source).resolve(), Path(target).resolve()
        if self._state(target) and source in self.states:
            self.states[target] = self.states[source]
            self._save(target)

    def __enter__(self):
        self.stack = ExitStack()
        original_write, original_flush = diffct.NpyStore.write, diffct.NpyStore.flush
        def write(store, selected, value, **kwargs):
            path = store.path.resolve()
            tracked = self._state(path)
            if tracked and self.rank == 3 and path not in self.states:
                # A deterministic lag exposes publication without cross-rank
                # completion agreement; only this test's checkpoint writes wait.
                time.sleep(.03)
            result = original_write(store, selected, value, **kwargs)
            if tracked:
                state = self._record(path, store.shape)
                state["coverage"][selected] = True
                state["regions"] += 1
                state["flushed"] = False
                self._save(path)
            return result
        def flush(store):
            result = original_flush(store)
            path = store.path.resolve()
            if self._state(path):
                self._record(path, store.shape)["flushed"] = True
                self._save(path)
            return result
        def replace_path(path, target):
            self._before_replace(path, target)
            result = self.original_path_replace(path, target)
            self._after_replace(path, target)
            return result
        def replace_os(source, target, *args, **kwargs):
            self._before_replace(source, target)
            result = self.original_os_replace(source, target, *args, **kwargs)
            self._after_replace(source, target)
            return result
        self.stack.enter_context(mock.patch.object(diffct.NpyStore, "write", write))
        self.stack.enter_context(mock.patch.object(diffct.NpyStore, "flush", flush))
        self.stack.enter_context(mock.patch.object(Path, "replace", replace_path))
        self.stack.enter_context(mock.patch.object(os, "replace", replace_os))
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)


def _checkpoint_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=45))
    try:
        root = Path(directory)
        case = _case("parallel", views=5)
        case.shape, case.detector = (3, 4), (7,)
        u, v = _grids(case)
        matrix = torch.from_numpy(_matrix(case, np.stack((u, v, np.zeros_like(u)), -1)))
        truth = _random(case.shape, 430)
        data = (matrix @ truth.flatten()).reshape(case.sino_shape)
        operator = _hybrid(case)
        local = data[operator.view_slice].clone()
        function = _api("cgls")
        expected, history, _ = _cgls_reference(matrix.numpy(), data.numpy(), 4)
        with _no_whole_volume_collectives(math.prod(case.shape[1:])), \
                _CheckpointWrites(root, case, rank) as writes:
            partial = function(operator, local, 2, rtol=0., atol=0., checkpoint_directory=root / "checkpoint")
            _result(partial, iterations=2)
            assert writes.commits, "atomic metadata publication seam was never observed"
            saved = _checkpoint(root / "checkpoint", 2, rank)
            resumed = function(operator, local, 4, rtol=0., atol=0., resume=saved)
            actual = _result(resumed, iterations=4)
            full_expected = torch.from_numpy(expected.reshape(case.shape))[operator.volume_slice]
            torch.testing.assert_close(actual.double(), full_expected, rtol=4e-5, atol=4e-6)
            np.testing.assert_allclose(resumed.residual_history, history, rtol=4e-5, atol=4e-6)
            for failure in ("data", "ownership", "iteration"):
                invalid_data, invalid_saved = local, saved
                if failure == "data" and rank == 3:
                    invalid_data = local.clone()
                    invalid_data[0, 0] += .1
                elif failure == "ownership" and rank == 3:
                    invalid_saved = _checkpoint(root / "checkpoint", 2, 2)
                elif failure == "iteration" and rank == 3:
                    invalid_saved = root / "wrong-iteration"
                    shutil.copytree(saved, invalid_saved)
                    document = json.loads((invalid_saved / "metadata.json").read_text(encoding="utf-8"))
                    document["iteration"] = 1
                    (invalid_saved / "metadata.json").write_text(json.dumps(document), encoding="utf-8")
                def forbidden(*args, **kwargs):
                    raise AssertionError("inconsistent distributed resume performed a projection before collective rejection")
                with ExitStack() as guards:
                    for method in ("project", "backproject", "project_into", "backproject_into"):
                        guards.enter_context(mock.patch.object(operator, method, forbidden))
                    with pytest.raises((ValueError, TypeError)):
                        function(operator, invalid_data, 4, rtol=0., atol=0., resume=invalid_saved)
                reached = torch.tensor(1)
                dist.all_reduce(reached)
                assert reached.item() == 4
        (root / f"checkpoint-rank-{rank}.json").write_text(
            json.dumps(dict(iterations=resumed.iterations, residual_history=list(resumed.residual_history),
                            rejected=["data", "ownership", "iteration"], commits=len(writes.commits))), encoding="utf-8")
    finally:
        dist.destroy_process_group()


@GLOO
def test_real_four_rank_hybrid_checkpoint_commits_complete_state_and_rejects_one_rank_mismatches_collectively(tmp_path):
    _api("cgls")
    assert "partition_grid" in inspect.signature(diffct.Projector).parameters, \
        "INTERFACE_PENDING: Projector.partition_grid"
    _run_four(_checkpoint_worker, ((tmp_path / "checkpoint-rendezvous").resolve().as_uri(), str(tmp_path)), timeout=150)
    records = [json.loads((tmp_path / f"checkpoint-rank-{rank}.json").read_text()) for rank in range(4)]
    assert all(record["iterations"] == 4 and len(record["residual_history"]) == 5
               and record["commits"] > 0 for record in records)
