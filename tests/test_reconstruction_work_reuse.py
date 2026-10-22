"""Reuse numerical residuals and filters without changing mutable/autograd paths."""

from contextlib import contextmanager, nullcontext
import math
import weakref
from unittest import mock

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

import diffct
from diffct import analytical_execution as analytical
from tests.test_execution_features import (
    _analytical_oracle, _overrides, _random, _reconstruct, _scan, _tv_oracle,
)
from tests.test_solver_api import _MatrixOperator, _checkpoint


def _problem():
    operator = _MatrixOperator(dtype=torch.float64)
    operator.dtype = torch.float64
    truth = torch.tensor([[-.3, .7, 1.1], [.2, -.15, .9]], dtype=torch.float64)
    data = operator.project(truth)
    operator.calls.clear()
    return operator, data


def _settings(name):
    return (dict(relaxation=.7, nonnegative=False) if name == "sirt"
            else dict(weight=.11, lr=.037))


def _mutate(kind, volume, matrix):
    with torch.no_grad():
        if kind == "volume":
            volume.add_(.17)
        elif kind == "geometry":
            matrix.mul_(1.13)


def _reference(name, matrix, data, iterations, x0=None, mutation=None):
    """Independent matrix recurrence; mutable callbacks run after history."""
    matrix, data = matrix.clone(), data.flatten()
    x = torch.zeros((2, 3), dtype=data.dtype) if x0 is None else x0.clone()
    if name == "sirt":
        rows, columns = matrix.sum(1), matrix.sum(0)
        row_inverse = torch.where(rows > 0, rows.reciprocal(), 0.)
        column_inverse = torch.where(columns > 0, columns.reciprocal(), 0.)
    else:
        x.requires_grad_()
        optimizer = torch.optim.Adam([x], lr=.037)
    history = [torch.linalg.vector_norm(matrix @ x.detach().flatten() - data).item()]
    snapshots = []
    for iteration in range(1, iterations + 1):
        if name == "sirt":
            residual = data - matrix @ x.flatten()
            x.add_((.7 * column_inverse * (matrix.T @ (row_inverse * residual))).reshape_as(x))
        else:
            optimizer.zero_grad()
            residual = matrix @ x.flatten() - data
            loss = .5 * residual.square().mean() + .11 * _tv_oracle(x, .73)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                x.clamp_(min=0)
        history.append(torch.linalg.vector_norm(matrix @ x.detach().flatten() - data).item())
        snapshots.append(x.detach().clone())
        if iteration == 1:
            _mutate(mutation, x, matrix)
    return x.detach(), history, snapshots


@pytest.mark.parametrize("name", ("sirt", "tv_reconstruction"))
@pytest.mark.parametrize("warm_start", (False, True))
def test_numerical_solver_reuses_completed_residual(name, warm_start):
    operator, data = _problem()
    x0 = torch.full(operator.volume_shape, .25, dtype=data.dtype) if warm_start else None
    expected, history, _ = _reference(name, operator.matrix, data, 3, x0)
    result = getattr(diffct, name)(operator, data, 3, x0=x0, rtol=0., atol=0., **_settings(name))
    assert result.iterations == 3 and not result.converged
    torch.testing.assert_close(result.volume, expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(result.residual_history, history, rtol=1e-11, atol=1e-12)
    maximum = 3 + int(warm_start) + int(name == "sirt")
    actual = operator.calls.count("project")
    assert actual <= maximum, f"{name} used {actual} projections; completed residual reuse permits {maximum}"


@pytest.mark.parametrize("name", ("sirt", "tv_reconstruction"))
@pytest.mark.parametrize("mutation", ("volume", "geometry"))
def test_solver_recomputes_after_callback_mutates_volume_or_operator(name, mutation):
    operator, data = _problem()
    x0 = torch.full(operator.volume_shape, .25, dtype=data.dtype)
    expected, history, snapshots = _reference(name, operator.matrix, data, 3, x0, mutation)
    observed = []

    def callback(iteration, residual, volume):
        observed.append((iteration, residual, volume.clone()))
        if iteration == 1:
            _mutate(mutation, volume, operator.matrix)

    result = getattr(diffct, name)(operator, data, 3, x0=x0, rtol=0., atol=0.,
                                  callback=callback, **_settings(name))
    torch.testing.assert_close(result.volume, expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(result.residual_history, history, rtol=1e-11, atol=1e-12)
    assert [item[0] for item in observed] == [1, 2, 3]
    for (_, residual, volume), expected_residual, expected_volume in zip(observed, history[1:], snapshots):
        assert residual == pytest.approx(expected_residual, rel=1e-11, abs=1e-12)
        torch.testing.assert_close(volume, expected_volume, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("name", ("sirt", "tv_reconstruction"))
def test_residual_reuse_recomputes_resumed_state_and_preserves_history(name, tmp_path):
    operator, data = _problem()
    options = dict(rtol=0., atol=0., **_settings(name))
    full = getattr(diffct, name)(operator, data, 4, **options)
    partial = getattr(diffct, name)(operator, data, 2, checkpoint_directory=tmp_path / "checkpoint", **options)
    resumed = getattr(diffct, name)(operator, data, 4, resume=_checkpoint(tmp_path / "checkpoint", 2), **options)
    assert partial.iterations == 2 and resumed.iterations == 4
    torch.testing.assert_close(resumed.volume, full.volume, rtol=0, atol=0)
    assert resumed.residual_history == full.residual_history


def _scan_four(beam):
    case = _scan(beam)
    case["trajectory"] = tuple(value[:4] for value in case["trajectory"])
    case["angles"] = case["angles"][:4]
    case["sino_shape"] = (4, *case["sino_shape"][1:])
    case["shape"] = (2, 4, 6) if beam == "cone" else (4, 6)
    chunks = (2, 2, 3) if beam == "cone" else (2, 3)
    return case, chunks


class _VolumeTemporaries(TorchDispatchMode):
    def __init__(self, shape):
        super().__init__()
        self.shape, self.complete = shape, []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        inputs = {value.untyped_storage()._cdata for value in tree_flatten((args, kwargs or {}))[0]
                  if isinstance(value, torch.Tensor)}
        result = function(*args, **(kwargs or {}))
        for value in tree_flatten(result)[0]:
            if (isinstance(value, torch.Tensor) and tuple(value.shape) == self.shape
                    and value.untyped_storage()._cdata not in inputs):
                self.complete.append(str(function))
        return result


@contextmanager
def _filter_work(chunks):
    filter_calls, gathers, filtered_refs = [], [], []
    original_filter = analytical._filter
    original_filtered = analytical.AnalyticalExecution._filtered
    original_gather = analytical.AnalyticalExecution._portable_gather

    def filtering(data, *args, **kwargs):
        filter_calls.append(tuple(data.shape))
        assert data.shape[0] <= 2
        return original_filter(data, *args, **kwargs)

    def filtered(owner, views, device):
        result = original_filtered(owner, views, device)
        filtered_refs.append(weakref.ref(result[0]))
        return result

    def gather(owner, data, geometry, views, spatial):
        assert all(ref() is None or ref() is data for ref in filtered_refs), \
            "analytical reconstruction retained a previous filtered batch"
        shape = tuple(part.stop - part.start for part in spatial)
        assert all(size <= limit for size, limit in zip(shape, chunks))
        gathers.append(((views.start, views.stop), tuple((part.start, part.stop) for part in spatial)))
        return original_gather(owner, data, geometry, views, spatial)

    with mock.patch.object(analytical, "_filter", filtering), \
            mock.patch.object(analytical.AnalyticalExecution, "_filtered", filtered), \
            mock.patch.object(analytical.AnalyticalExecution, "_portable_gather", gather):
        yield filter_calls, gathers


@pytest.mark.parametrize("beam", ("parallel", "fan", "cone"))
@pytest.mark.parametrize("disabled_grad", (False, True), ids=("numerical-inputs", "no-grad-context"))
def test_numerical_analytical_filters_one_view_batch_across_volume_tiles(beam, disabled_grad):
    case, chunks = _scan_four(beam)
    data = _random(case["sino_shape"], 950, grad=disabled_grad)
    weights = _overrides(case, grad=disabled_grad)
    options = dict(backend="torch", precision="float64", isocenter=case["center"],
                   view_chunk_size=2, volume_chunk_shape=chunks)
    # Active data autograd selects the existing tile-first path, preserving its
    # per-tile addition order as the exact numerical control.
    control = _reconstruct(case, data.detach().clone().requires_grad_(), **options,
                           **{name: value.detach() for name, value in weights.items()}).detach()
    oracle = _analytical_oracle(case, data.detach(), **{name: value.detach() for name, value in weights.items()})
    with (torch.no_grad() if disabled_grad else nullcontext()), _filter_work(chunks) as trace, \
            _VolumeTemporaries(case["shape"]) as allocations:
        actual = _reconstruct(case, data, **options, **weights)
    torch.testing.assert_close(actual, control, rtol=0, atol=0)
    torch.testing.assert_close(actual, oracle, rtol=2e-11, atol=2e-12)
    assert not actual.requires_grad
    assert len(allocations.complete) <= 2, (
        f"analytical execution allocated extra whole-volume temporaries: {allocations.complete}"
    )
    filters, gathers = trace
    assert len(gathers) == 8
    assert len(filters) == 2, f"four volume tiles and two view batches filtered {len(filters)} times; expected 2"
    assert [views for views, _ in gathers] == [(0, 2)] * 4 + [(2, 4)] * 4
    assert len({spatial for _, spatial in gathers}) == math.prod(
        math.ceil(size / chunk) for size, chunk in zip(case["shape"], chunks)
    ) == 4


@pytest.mark.parametrize("beam", ("parallel", "cone"))
@pytest.mark.parametrize("trainable", ("data", "weights"))
def test_active_analytical_autograd_keeps_original_filter_path_and_gradients(beam, trainable):
    case, chunks = _scan_four(beam)
    data = _random(case["sino_shape"], 952, grad=trainable == "data")
    weights = _overrides(case, grad=trainable == "weights")
    expected = _analytical_oracle(case, data, **weights)
    with mock.patch.object(analytical, "_filter", wraps=analytical._filter) as filtering:
        actual = _reconstruct(case, data, backend="torch", precision="float64", isocenter=case["center"],
                              view_chunk_size=2, volume_chunk_shape=chunks, **weights)
    leaves = (data,) if trainable == "data" else tuple(weights.values())
    cotangent = _random(case["shape"], 953)
    expected_gradients = torch.autograd.grad(expected, leaves, cotangent)
    actual_gradients = torch.autograd.grad(actual, leaves, cotangent)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    for observed, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(observed, reference, rtol=3e-11, atol=3e-12)
    assert filtering.call_count == 8, "active autograd bypassed the original tile-first path"
