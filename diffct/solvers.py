"""Numerical iterative reconstruction with residual controls and restart state."""

from dataclasses import dataclass
import math
from pathlib import Path

import torch

from .pipeline import _fragments
from .storage import NpyStore, TensorStore
from .streaming import _validate_stores
from ._solver_state import (
    agreement, callback_stop, callbacks_enabled, copy_blocks, count, fill_blocks, finite,
    load_checkpoint, save_checkpoint, scalar_sum, signature, squared_norm, update_blocks,
)
from ._solver_tv import total_variation


@dataclass
class SolverResult:
    volume: object
    iterations: int
    residual_history: list[float]
    converged: bool
    reason: str


def _controls(iterations, rtol, atol, callback, checkpoint_directory, resume):
    iterations = count(iterations, "iterations")
    rtol, atol = finite(rtol, "rtol"), finite(atol, "atol")
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable")
    for directory in (checkpoint_directory, resume):
        if directory is not None:
            Path(directory)
    return iterations, rtol, atol


def _request(operator, measurements, iterations, x0, rtol, atol, callback,
             checkpoint_directory, resume, settings):
    error, controls = None, None
    try:
        controls = _controls(iterations, rtol, atol, callback, checkpoint_directory, resume)
        if not isinstance(measurements, torch.Tensor) or not torch.is_floating_point(measurements):
            raise TypeError("measurements must be a real floating-point tensor")
        if tuple(measurements.shape) != tuple(operator.projection_shape) or not torch.isfinite(measurements).all():
            raise ValueError("measurements must be finite and match projection_shape")
        if x0 is not None and (not isinstance(x0, torch.Tensor) or not torch.is_floating_point(x0)
                               or tuple(x0.shape) != tuple(operator.volume_shape) or not torch.isfinite(x0).all()):
            raise ValueError("x0 must be a finite tensor matching volume_shape")
        for name, value in settings.items():
            if name == "nonnegative":
                if not isinstance(value, bool):
                    raise TypeError("nonnegative must be a boolean")
            else:
                finite(value, name, positive=name in ("relaxation", "lr"))
    except (ValueError, TypeError, OSError) as caught:
        error = caught
    agreement(operator, error, (controls, settings, x0 is not None,
                                resume is not None, checkpoint_directory is not None),
              cache_key="solver request")
    return (*controls, callbacks_enabled(operator, callback))


def _initial(operator, data, x0):
    dtype = getattr(operator, "dtype", data.dtype)
    return (torch.zeros(operator.volume_shape, device=data.device, dtype=dtype)
            if x0 is None else x0.detach().to(device=data.device, dtype=dtype).clone())


def _read_tensor(store, device):
    return store.read(tuple(slice(0, size) for size in store.shape)).to(device=device)


def _status(residual, threshold, stopped=False):
    if stopped:
        return False, "callback stopped"
    if residual <= threshold:
        return True, "residual tolerance reached"
    return False, "iteration limit reached"


@torch.no_grad()
def cgls(operator, measurements, iterations=30, *, x0=None, rtol=1e-6, atol=0.,
         callback=None, checkpoint_directory=None, resume=None):
    """CGLS with a total iteration target, including resumed iterations."""
    iterations, rtol, atol, callbacks = _request(operator, measurements, iterations, x0, rtol, atol,
                                     callback, checkpoint_directory, resume, {})
    data = measurements.detach().to(dtype=getattr(operator, "dtype", measurements.dtype))
    metadata = signature(operator, measurements, "cgls", {}, 1048576, compute_dtype=data.dtype) if checkpoint_directory is not None or resume is not None else None
    if resume is not None:
        saved, stores = load_checkpoint(operator, resume, metadata, iterations, 1048576, ("x", "s", "p", "r", "q"))
        states = {name: _read_tensor(store, data.device) for name, store in stores.items()}
        completed, history, norm, gamma = saved["iteration"], list(saved["residual_history"]), saved["measurement_norm"], saved["gamma"]
    else:
        x = _initial(operator, data, x0)
        residual = data.clone() if x0 is None else data - operator.project(x)
        norm = math.sqrt(squared_norm(operator, data, "rays"))
        history = [math.sqrt(squared_norm(operator, residual, "rays"))]
        completed = 0
        adjoint = (operator.backproject(residual) if iterations and history[0] > atol + rtol * norm
                   else torch.zeros_like(x))
        states = dict(x=x, s=adjoint, p=adjoint.clone(), r=residual, q=torch.zeros_like(data))
        gamma = squared_norm(operator, adjoint, "volume")
    threshold = atol + rtol * norm
    stopped, breakdown = False, False
    while completed < iterations and history[-1] > threshold:
        if gamma == 0:
            breakdown = True
            break
        states["q"] = operator.project(states["p"])
        denominator = squared_norm(operator, states["q"], "rays")
        if denominator == 0:
            breakdown = True
            break
        alpha = gamma / denominator
        states["x"].add_(states["p"], alpha=alpha)
        states["r"].sub_(states["q"], alpha=alpha)
        states["s"] = operator.backproject(states["r"])
        next_gamma = squared_norm(operator, states["s"], "volume")
        states["p"].mul_(next_gamma / gamma).add_(states["s"])
        gamma = next_gamma
        completed += 1
        history.append(math.sqrt(squared_norm(operator, states["r"], "rays")))
        if checkpoint_directory is not None:
            save_checkpoint(operator, checkpoint_directory, states, metadata, completed, history,
                            norm, 1048576, gamma=gamma)
        if callbacks:
            stopped = callback_stop(operator, callback, completed, history[-1], states["x"])
        if stopped:
            break
    converged, reason = _status(history[-1], threshold, stopped)
    if breakdown and not converged:
        reason = "normal equation breakdown"
    return SolverResult(states["x"].detach(), completed, history, converged, reason)


@torch.no_grad()
def sirt(operator, measurements, iterations=100, *, x0=None, rtol=1e-6, atol=0.,
         callback=None, checkpoint_directory=None, resume=None, relaxation=1., nonnegative=True):
    """Row/column-normalized SIRT, with optional nonnegative iterates."""
    settings = dict(relaxation=relaxation, nonnegative=nonnegative)
    iterations, rtol, atol, callbacks = _request(operator, measurements, iterations, x0, rtol, atol,
                                     callback, checkpoint_directory, resume, settings)
    data = measurements.detach().to(dtype=getattr(operator, "dtype", measurements.dtype))
    metadata = signature(operator, measurements, "sirt", settings, 1048576, compute_dtype=data.dtype) if checkpoint_directory is not None or resume is not None else None
    if resume is not None:
        saved, stores = load_checkpoint(operator, resume, metadata, iterations, 1048576, ("x", "row", "column"))
        states = {name: _read_tensor(store, data.device) for name, store in stores.items()}
        completed, history, norm = saved["iteration"], list(saved["residual_history"]), saved["measurement_norm"]
    else:
        states = dict(x=_initial(operator, data, x0))
        completed, norm = 0, math.sqrt(squared_norm(operator, data, "rays"))
        initial = data if x0 is None else data - operator.project(states["x"])
        history = [math.sqrt(squared_norm(operator, initial, "rays"))]
        if iterations and history[0] > atol + rtol * norm:
            row = operator.project(torch.ones_like(states["x"]))
            column = operator.backproject(torch.ones_like(data))
            states["row"] = torch.where(row > 0, row.reciprocal(), 0.)
            states["column"] = torch.where(column > 0, column.reciprocal(), 0.)
    threshold, stopped = atol + rtol * norm, False
    while completed < iterations and history[-1] > threshold:
        residual = data - operator.project(states["x"])
        states["x"].add_(states["column"] * operator.backproject(states["row"] * residual), alpha=relaxation)
        if nonnegative:
            states["x"].clamp_(min=0)
        completed += 1
        history.append(math.sqrt(squared_norm(operator, data - operator.project(states["x"]), "rays")))
        if checkpoint_directory is not None:
            save_checkpoint(operator, checkpoint_directory, states, metadata, completed, history, norm, 1048576)
        if callbacks:
            stopped = callback_stop(operator, callback, completed, history[-1], states["x"])
        if stopped:
            break
    converged, reason = _status(history[-1], threshold, stopped)
    return SolverResult(states["x"].detach(), completed, history, converged, reason)


@torch.no_grad()
def tv_reconstruction(operator, measurements, iterations=100, weight=1., lr=.1, *,
                      x0=None, rtol=1e-6, atol=0., callback=None,
                      checkpoint_directory=None, resume=None):
    """Adam on half the global mean squared data error plus physical mean TV."""
    settings = dict(weight=weight, lr=lr)
    iterations, rtol, atol, callbacks = _request(operator, measurements, iterations, x0, rtol, atol,
                                     callback, checkpoint_directory, resume, settings)
    data = measurements.detach().to(dtype=getattr(operator, "dtype", measurements.dtype))
    metadata = signature(operator, measurements, "tv_reconstruction", settings, 1048576, compute_dtype=data.dtype) if checkpoint_directory is not None or resume is not None else None
    saved, stores = (load_checkpoint(operator, resume, metadata, iterations, 1048576,
                                    ("x", "exp_avg", "exp_avg_sq", "step")) if resume is not None else (None, None))
    x = (_read_tensor(stores["x"], data.device) if stores is not None else _initial(operator, data, x0)).requires_grad_()
    optimizer = torch.optim.Adam([x], lr=lr)
    if saved is not None:
        optimizer.state[x] = {name: _read_tensor(stores[name], "cpu" if name == "step" else x.device)
                              for name in ("exp_avg", "exp_avg_sq", "step")}
        optimizer.state[x]["step"] = optimizer.state[x]["step"].reshape(())
        completed, history, norm = saved["iteration"], list(saved["residual_history"]), saved["measurement_norm"]
    else:
        completed, norm = 0, math.sqrt(squared_norm(operator, data, "rays"))
        initial = data if x0 is None else data - operator.project(x)
        history = [math.sqrt(squared_norm(operator, initial, "rays"))]
    threshold, stopped = atol + rtol * norm, False
    ray_count = scalar_sum(operator, data.numel(), "rays")
    while completed < iterations and history[-1] > threshold:
        residual = operator.project(x) - data
        gradient = operator.backproject(residual) / max(ray_count, 1)
        if weight:
            with torch.enable_grad():
                tv = total_variation(x, operator=operator, voxel_spacing=operator.voxel_spacing)
                tv_gradient, = torch.autograd.grad(tv, x)
            gradient = gradient + weight * tv_gradient
        optimizer.zero_grad()
        x.grad = gradient.to(x)
        optimizer.step()
        x.clamp_(min=0)
        completed += 1
        history.append(math.sqrt(squared_norm(operator, data - operator.project(x), "rays")))
        if checkpoint_directory is not None:
            states = dict(x=x, **optimizer.state[x])
            states["step"] = states["step"].reshape(1)
            save_checkpoint(operator, checkpoint_directory, states, metadata, completed, history, norm, 1048576)
        if callbacks:
            stopped = callback_stop(operator, callback, completed, history[-1], x.detach())
        if stopped:
            break
    converged, reason = _status(history[-1], threshold, stopped)
    return SolverResult(x.detach(), completed, history, converged, reason)


@torch.no_grad()
def cgls_into(operator, measurements, output, iterations, *, workspace,
              block_elements=1048576, rtol=1e-6, atol=0., callback=None,
              checkpoint_directory=None, resume=None):
    """CGLS into caller storage with mapped state and bounded block operations.

    ``workspace`` must be a new directory. Every state update, norm,
    fingerprint and checkpoint copy contains at most ``block_elements`` values.
    The result retains the exact output object supplied by the caller.
    """
    error, controls = None, None
    try:
        controls = _controls(iterations, rtol, atol, callback, checkpoint_directory, resume)
        block_elements = count(block_elements, "block_elements", positive=True)
        numerical = measurements.detach() if isinstance(measurements, torch.Tensor) else measurements
        source, x = _validate_stores(operator, numerical, output, False)
        for selected in _fragments(tuple(source.shape), block_elements):
            if not torch.isfinite(source.read(selected)).all():
                raise ValueError("measurements must be finite")
        directory = Path(workspace).resolve()
        if directory.exists():
            raise ValueError("workspace must be a new directory")
    except Exception as caught:
        error = caught
    agreement(operator, error, (controls, block_elements,
                                resume is not None, checkpoint_directory is not None),
              cache_key="solver request")
    iterations, rtol, atol = controls
    callbacks = callbacks_enabled(operator, callback)
    metadata = signature(operator, source, "cgls_into", {}, block_elements, compute_dtype=x.dtype) if checkpoint_directory is not None or resume is not None else None
    previous = (load_checkpoint(operator, resume, metadata, iterations, block_elements, ("x", "s", "p", "r", "q"))
                if resume is not None else None)
    error = None
    try:
        directory.mkdir(parents=True)
        states = dict(x=x)
        for name in ("s", "p", "r", "q"):
            shape = operator.volume_shape if name in ("s", "p") else operator.projection_shape
            states[name] = NpyStore.create(directory / f"{name}.npy", shape, dtype=x.dtype)
    except Exception as caught:
        error = caught
    # Workspace admission must succeed everywhere before any rank can zero
    # caller output, copy a resumed state, or enter a numerical collective.
    agreement(operator, error)
    if previous is not None:
        saved, stores = previous
        for name, store in states.items():
            copy_blocks(stores[name], store, block_elements)
        completed, history, norm, gamma = saved["iteration"], list(saved["residual_history"]), saved["measurement_norm"], saved["gamma"]
    else:
        completed = 0
        fill_blocks(x, block_elements)
        fill_blocks(states["q"], block_elements)
        copy_blocks(source, states["r"], block_elements)
        norm = math.sqrt(squared_norm(operator, source, "rays", block_elements))
        history = [norm]
        if iterations and norm > atol + rtol * norm:
            operator.backproject_into(states["r"], states["s"])
            copy_blocks(states["s"], states["p"], block_elements)
        else:
            fill_blocks(states["s"], block_elements)
            fill_blocks(states["p"], block_elements)
        gamma = squared_norm(operator, states["s"], "volume", block_elements)
    threshold, stopped, breakdown = atol + rtol * norm, False, False
    while completed < iterations and history[-1] > threshold:
        if gamma == 0:
            breakdown = True
            break
        operator.project_into(states["p"], states["q"])
        denominator = squared_norm(operator, states["q"], "rays", block_elements)
        if denominator == 0:
            breakdown = True
            break
        alpha = gamma / denominator
        update_blocks(states["x"], states["p"], alpha, block_elements)
        update_blocks(states["r"], states["q"], -alpha, block_elements)
        operator.backproject_into(states["r"], states["s"])
        next_gamma = squared_norm(operator, states["s"], "volume", block_elements)
        update_blocks(states["p"], states["s"], 1., block_elements, next_gamma / gamma)
        gamma = next_gamma
        completed += 1
        history.append(math.sqrt(squared_norm(operator, states["r"], "rays", block_elements)))
        states["x"].flush()
        if checkpoint_directory is not None:
            save_checkpoint(operator, checkpoint_directory, states, metadata, completed, history,
                            norm, block_elements, gamma=gamma)
        if callbacks:
            stopped = callback_stop(operator, callback, completed, history[-1], output)
        if stopped:
            break
    for store in states.values():
        store.flush()
    converged, reason = _status(history[-1], threshold, stopped)
    if breakdown and not converged:
        reason = "normal equation breakdown"
    return SolverResult(output, completed, history, converged, reason)
