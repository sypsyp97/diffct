"""Numerical CGLS with mapped working state and rectangular block updates."""

import json
import math
import numbers
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from diffct import NpyStore, TensorStore
from diffct.chunking import _allocation_bytes
from diffct.pipeline import _fragments
from diffct.streaming import _validate_stores


def _count(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < int(positive):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} integer")
    return int(value)


def _copy_blocks(source, target, elements):
    for selected in _fragments(tuple(target.shape), elements):
        value = source.read(selected)
        if torch.device(getattr(target, "device", "cpu")).type == "cpu":
            value = value.to(device="cpu")
        value = value.to(dtype=torch.float32)
        target.write(selected, value)


def _fill_blocks(target, elements):
    for selected in _fragments(tuple(target.shape), elements):
        shape = tuple(item.stop - item.start for item in selected)
        target.write(selected, torch.zeros(shape, dtype=torch.float32, device="cpu"))


def _update_blocks(target, source, factor, elements, *, scale=1.0):
    for selected in _fragments(tuple(target.shape), elements):
        value = target.read(selected).to(device="cpu", dtype=torch.float32)
        value.mul_(scale).add_(source.read(selected).to(device="cpu", dtype=torch.float32), alpha=factor)
        target.write(selected, value)


def _norm(operator, store, kind, elements):
    """FP64 blocks and one globally owned scalar; never convert a whole state."""
    partition = getattr(operator, "partition", "views")
    replicated = (kind == "volume") == (partition == "views")
    device = torch.device(getattr(store, "device", "cpu"))
    value = torch.zeros((), dtype=torch.float64, device=device)
    if not replicated or operator.rank == 0:
        for selected in _fragments(tuple(store.shape), elements):
            original = store.read(selected)
            block = original.to(device=device, dtype=torch.float64)
            if block is original:
                block = block.clone()
            value.add_(block.square_().sum())
            del block, original
    if operator.world_size > 1:
        group = operator.process_group
        device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend(group) == "nccl" else "cpu"
        value = value.to(device=device)
        dist.all_reduce(value, op=dist.ReduceOp.SUM, group=group)
    return value.item()


def _solver_capacity(operator, measurements, block_elements, output_device, *, state_device=None):
    volume = math.prod(operator.volume_shape)
    rays = math.prod(operator.projection_shape)
    states = 3 * _allocation_bytes(volume) + 2 * _allocation_bytes(rays) if torch.cuda.is_available() else 4 * (3 * volume + 2 * rays)
    state_device = torch.device(output_device if state_device is None else state_device)
    dot_elements = min(block_elements, max(volume, rays))
    nccl = operator.world_size > 1 and dist.get_backend(operator.process_group) == "nccl"
    # One block, its persistent accumulator and its reduction scalar. NCCL
    # may additionally stage that scalar onto the process's collective device.
    dot_cuda = (_allocation_bytes(2 * dot_elements) + (2 + int(nccl)) * _allocation_bytes(2)
                if state_device.type == "cuda" else _allocation_bytes(2) if nccl else 0)
    cuda_states = states if state_device.type == "cuda" else 0
    plan = None
    if hasattr(operator, "_execution_plan"):
        plan = operator._execution_plan(measurements, False, SimpleNamespace(device=output_device))
    pipeline = plan.estimated_gpu_bytes if plan is not None else 0
    if plan is not None:
        # Admission covers every fitting order which the bounded pilot may pick.
        from diffct import ParameterizedSurface
        from diffct.surfaces import _GeometrySpec
        surface = operator.detector_surface
        spec = _GeometrySpec("sampled", surface.sampler) if isinstance(surface, ParameterizedSurface) else _GeometrySpec("points") if surface is not None else None
        candidates = operator._candidate_plans(plan, False, spec)
        pipeline = max((candidate.estimated_gpu_bytes for candidate in candidates), default=pipeline)
    return dict(
        requested_volume_state_bytes=12 * volume,
        requested_ray_state_bytes=8 * rays,
        required_cuda_state_bytes=states,
        estimated_cuda_state_bytes=cuda_states,
        fp64_dot_scratch_bytes=8 * dot_elements + 16 + 8 * int(nccl),
        fp64_dot_scratch_device=str(state_device),
        estimated_cuda_dot_scratch_bytes=dot_cuda,
        estimated_pipeline_and_geometry_bytes=pipeline,
        gpu_budget_bytes=plan.gpu_budget_bytes if plan is not None else None,
        estimated_gpu_bytes=cuda_states + pipeline + dot_cuda,
    )


def _checkpoint_metadata(operator):
    return dict(
        schema_version=1, global_volume_shape=list(operator.global_volume_shape),
        local_volume_shape=list(operator.volume_shape),
        volume_slice=[[part.start, part.stop] for part in operator.volume_slice],
        partition=operator.partition, rank=operator.rank, world_size=operator.world_size,
        beam=operator.beam, voxel_spacing=operator.voxel_spacing,
        detector_spacing=list(operator.detector_spacing), detector_shape=list(operator.detector_shape),
        projection_shape=list(operator.projection_shape), dtype="float32",
    )


def _resume_states(directory, metadata, iterations):
    directory = Path(directory).resolve()
    with (directory / "metadata.json").open(encoding="utf-8") as file:
        saved = json.load(file)
    if any(saved.get(name) != value for name, value in metadata.items()):
        raise ValueError("checkpoint resume metadata does not match partition, ownership, shape, dtype or scan")
    completed = _count(saved.get("iteration"), "checkpoint iteration")
    gamma = saved.get("gamma")
    if not isinstance(gamma, (float, int)) or not math.isfinite(gamma) or gamma < 0:
        raise ValueError("checkpoint gamma must be finite and nonnegative")
    if completed > iterations:
        raise ValueError("target iterations are below the completed resume iteration")
    states = {name: NpyStore(directory / f"{name}.npy", mode="r") for name in ("x", "s", "p", "r", "q")}
    for name, store in states.items():
        shape = metadata["local_volume_shape"] if name in ("x", "s", "p") else metadata["projection_shape"]
        if tuple(store.shape) != tuple(shape) or store.dtype != torch.float32:
            raise ValueError("checkpoint state shape or dtype does not match resume metadata")
    return completed, float(gamma), states


def _checkpoint(directory, states, metadata, iteration, gamma, elements):
    step = Path(directory).resolve() / f"step-{iteration:06d}"
    step.mkdir(parents=True, exist_ok=True)
    rank = step / f"rank-{metadata['rank']:05d}"
    rank.mkdir()
    for name, source in states.items():
        target = NpyStore.create(rank / f"{name}.npy", source.shape)
        _copy_blocks(source, target, elements)
        target.flush()
    document = dict(metadata, iteration=iteration, gamma=gamma)
    temporary = rank / "metadata.tmp"
    with temporary.open("x", encoding="utf-8") as file:
        json.dump(document, file)
    temporary.replace(rank / "metadata.json")


@torch.no_grad()
def cgls_into(operator, measurements, output, iterations, *, workspace,
              block_elements=1048576, checkpoint_directory=None, resume=None):
    """Reconstruct into supplied storage, using a new directory for s/p/r/q.

    Reads, updates, FP64 norms and checkpoint copies contain at most
    ``block_elements`` values. Checkpoints store each rank's owned volume and
    rays. ``iterations`` is the total target, including resumed iterations;
    resume requires the same geometry and measurements supplied by the caller.
    The example assigns a tensor-free capacity report to ``last_solver_stats``.
    """
    iterations = _count(iterations, "iterations")
    block_elements = _count(block_elements, "block_elements", positive=True)
    source, x = _validate_stores(operator, measurements, output, False)
    metadata = _checkpoint_metadata(operator)
    saved = _resume_states(resume, metadata, iterations) if resume is not None else None
    directory = Path(workspace).resolve()
    if directory.exists():
        raise ValueError("workspace must be a new owned directory")
    report = _solver_capacity(operator, source, block_elements, torch.device(getattr(x, "device", "cpu")), state_device="cpu")
    report["state_backend"] = "disk" if isinstance(x, NpyStore) else "disk+" + str(torch.device(getattr(x, "device", "cpu"))) + "-output"
    operator.last_solver_stats = report
    directory.mkdir(parents=True)
    states = {"x": x}
    for name in ("s", "p", "r", "q"):
        shape = operator.volume_shape if name in ("s", "p") else operator.projection_shape
        states[name] = NpyStore.create(directory / f"{name}.npy", shape)
    if saved is not None:
        completed, gamma, previous = saved
        for name, target in states.items():
            _copy_blocks(previous[name], target, block_elements)
    else:
        completed = 0
        _fill_blocks(x, block_elements)
        _fill_blocks(states["q"], block_elements)
        if iterations == 0:
            for name in ("s", "p", "r"):
                _fill_blocks(states[name], block_elements)
            gamma = 0.0
        else:
            _copy_blocks(source, states["r"], block_elements)
            operator.backproject_into(states["r"], states["s"])
            _copy_blocks(states["s"], states["p"], block_elements)
            gamma = _norm(operator, states["s"], "volume", block_elements)
    for iteration in range(completed, iterations):
        if gamma == 0:
            break
        operator.project_into(states["p"], states["q"])
        qq = _norm(operator, states["q"], "rays", block_elements)
        if qq == 0:
            break
        alpha = gamma / qq
        _update_blocks(x, states["p"], alpha, block_elements)
        _update_blocks(states["r"], states["q"], -alpha, block_elements)
        operator.backproject_into(states["r"], states["s"])
        next_gamma = _norm(operator, states["s"], "volume", block_elements)
        _update_blocks(states["p"], states["s"], 1.0, block_elements, scale=next_gamma / gamma)
        gamma = next_gamma
        if checkpoint_directory is not None:
            _checkpoint(checkpoint_directory, states, metadata, iteration + 1, gamma, block_elements)
    for store in states.values():
        store.flush()
    return output
