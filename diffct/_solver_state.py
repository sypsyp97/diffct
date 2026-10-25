"""Bounded solver state, owned reductions and complete checkpoint publication."""

import hashlib
import json
import math
import numbers
from pathlib import Path

import torch
import torch.distributed as dist

from .utils import _work_dtype

from .pipeline import _fragments
from .storage import NpyStore, TensorStore
from .distribution import control_status, preflight


def count(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < int(positive):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} integer")
    return int(value)


def finite(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def agreement(operator, error=None, token=None, *, cache_key=None):
    """Every rank rejects a bad request before numerical execution starts."""
    world = getattr(operator, "world_size", 1)
    if world == 1:
        if error is not None:
            raise error
        return
    error = None if error is None else ValueError("distributed solver agreement failed: " + str(error))
    mismatch = "distributed solver agreement failed: inconsistent checkpoint iteration or controls"
    partition = getattr(operator, "_partition", None)
    if partition is not None:
        partition.preflight(error, token, cache_key=cache_key, mismatch=mismatch)
    else:
        preflight(error, token, process_group=operator.process_group, world_size=world,
                  mismatch=mismatch)


def owns_values(operator, kind):
    """Select one copy of each physical ray or voxel for scalar reductions."""
    partition = getattr(operator, "partition", "views")
    rank = getattr(operator, "rank", 0)
    return (getattr(operator, "view_rank", 0) == 0 if kind == "volume"
            else getattr(operator, "space_rank", 0) == 0) if partition == "hybrid" else (
        rank == 0 if (kind == "volume") == (partition == "views") else True
    )


def scalar_sum(operator, value, kind, *, return_tensor=False):
    """Count physical rays/voxels once, optionally retaining the reduction tensor."""
    world = getattr(operator, "world_size", 1)
    result_device = value.device if isinstance(value, torch.Tensor) else torch.device("cpu")
    value = value if owns_values(operator, kind) else 0.
    if world > 1:
        group = operator.process_group
        device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend(group) == "nccl" else torch.device("cpu")
        accumulator = torch.as_tensor(value, dtype=torch.float64, device=device).clone()
        dist.all_reduce(accumulator, group=group)
        return accumulator if return_tensor else accumulator.item()
    if return_tensor:
        return torch.as_tensor(value, dtype=_work_dtype(result_device), device=result_device)
    return float(value)


def squared_norm(operator, value, kind, elements=1048576, *, return_tensor=False):
    store = TensorStore(value) if isinstance(value, torch.Tensor) else value
    # Scale every block by its own peak before squaring and track the
    # running maximum on the device: unscaled FP32 squares overflow for
    # large measurements and underflow for tiny geometries.
    device = getattr(store, "device", torch.device("cpu"))
    if _work_dtype(device) == torch.float64:
        total = torch.zeros((), dtype=torch.float64, device=device)
        if owns_values(operator, kind):
            for selected in _fragments(tuple(store.shape), elements):
                # Clone even an FP64 view: norms must never alter caller measurements.
                block = store.read(selected).to(dtype=torch.float64).clone()
                # Preserve the block reductions and their sequential FP64 order.
                total.add_(block.square_().sum().to(device=total.device))
        return scalar_sum(operator, total, kind, return_tensor=return_tensor)
    # FP32 devices (MPS): squares of large measurements overflow and tiny
    # geometries underflow, so every block is rescaled by its peak and the
    # running sum follows the largest peak seen. All block math stays on the
    # device; the two closing scalar reads fit the per-iteration reduction
    # budget, and their product leaves FP32 before it can overflow.
    peak = torch.zeros((), device=device, dtype=torch.float32)
    total = torch.zeros((), device=device, dtype=torch.float32)
    if owns_values(operator, kind):
        for selected in _fragments(tuple(store.shape), elements):
            block = store.read(selected)
            if not block.numel():
                continue
            block_peak = block.detach().abs().amax()
            safe = torch.where(block_peak > 0, block_peak, torch.ones_like(block_peak))
            combined = torch.maximum(peak, block_peak)
            guard = torch.where(combined > 0, combined, torch.ones_like(combined))
            total = (total * (peak / guard).square()
                     + (block.detach() / safe).square().sum() * (block_peak / guard).square())
            peak = combined
    return scalar_sum(operator, peak.item() ** 2 * total.item(), kind,
                      return_tensor=return_tensor)


def copy_blocks(source, target, elements):
    for selected in _fragments(tuple(target.shape), elements):
        target.write(selected, source.read(selected).to(device=getattr(target, "device", "cpu"), dtype=target.dtype))


def fill_blocks(target, elements):
    for selected in _fragments(tuple(target.shape), elements):
        shape = tuple(part.stop - part.start for part in selected)
        target.write(selected, torch.zeros(shape, dtype=target.dtype))


def update_blocks(target, source, factor, elements, scale=1.):
    device = getattr(target, "device", "cpu")
    for selected in _fragments(tuple(target.shape), elements):
        block = target.read(selected).to(device=device, dtype=target.dtype).clone()
        block.mul_(scale).add_(source.read(selected).to(device=device, dtype=target.dtype), alpha=factor)
        target.write(selected, block)


def fingerprint(value, elements):
    store = TensorStore(value.detach()) if isinstance(value, torch.Tensor) else value
    digest = hashlib.sha256(json.dumps((tuple(store.shape), str(store.dtype))).encode())
    for selected in _fragments(tuple(store.shape), elements):
        block = store.read(selected).detach().to(device="cpu").contiguous()
        # Hash storage bytes without requiring NumPy support for the original
        # floating dtype (for example bfloat16). This view allocates no payload.
        digest.update(block.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _effective_surface(operator, frames, elements):
    """Hash the actual physical pixel rays, without materializing a volume."""
    surface = getattr(operator, "detector_surface", None)
    if surface is None:
        return None
    from .backends import ReferenceProjector
    from .surfaces import ParameterizedSurface, _detector_grids

    dimensions = len(operator.global_volume_shape)
    if elements < max(3, dimensions):
        raise ValueError("block_elements cannot hold one physical detector pixel")
    if isinstance(surface, ParameterizedSurface):
        if any(value.device.type != "cpu" and value.numel() > elements for value in surface.parameters):
            raise ValueError("surface parameters cannot be staged within block_elements")
        geometry = (*frames, *surface.parameters)
    else:
        # Legacy callbacks require complete grids. Their documented output can
        # contain every view, so admit that snapshot only when it fits the
        # bound; a smaller arbitrary callback invocation may change its meaning.
        if operator._n_views * math.prod(operator.detector_shape) * 3 > elements:
            raise ValueError("legacy surface fingerprint exceeds block_elements; use ParameterizedSurface")
        pixels = tuple(slice(0, size) for size in operator.detector_shape)
        u, v = _detector_grids(operator.detector_shape, operator.detector_spacing, pixels)
        offsets = surface(u, v)
        if not isinstance(offsets, torch.Tensor) or not torch.is_floating_point(offsets):
            raise TypeError("detector_surface must return floating-point offsets")
        if tuple(offsets.shape) not in ((*operator.detector_shape, 3),
                                       (operator._n_views, *operator.detector_shape, 3)):
            raise ValueError("legacy detector surface must match the complete pixel grids")
        geometry = (*frames, offsets)
    reference = getattr(operator, "_reference", None)
    if reference is None:
        reference = ReferenceProjector(operator, validate=False)
    description = json.dumps((operator._n_views, operator.detector_shape, str(operator.dtype))).encode()
    origins_digest, directions_digest = hashlib.sha256(description), hashlib.sha256(description)
    for view in range(operator._n_views):
        for pixels in _fragments(operator.detector_shape, elements // max(3, dimensions)):
            origins, directions, _ = reference._rays(slice(view, view + 1), pixels,
                                                      torch.device("cpu"), validate=True, geometry=geometry)
            # Separate streams preserve row-major ray ordering across different
            # block sizes; changing a memory ceiling does not change the hash.
            origins_digest.update(origins.detach().contiguous().numpy().tobytes())
            directions_digest.update(directions.detach().contiguous().numpy().tobytes())
    return [origins_digest.hexdigest(), directions_digest.hexdigest()]


def _signature_local(operator, source, algorithm, settings, elements, compute_dtype):
    frames = getattr(operator, "_learnable", ()) or getattr(operator, "_trajectory", ())
    geometry = tuple(fingerprint(value, elements) for value in frames)
    surface = getattr(operator, "detector_surface", None)
    surface_values = (() if surface is None else getattr(surface, "parameters", ()))
    if isinstance(surface, torch.Tensor):
        surface_values = (surface,)
    return dict(
        schema_version=1, algorithm=algorithm, settings=settings,
        rank=getattr(operator, "rank", 0), world_size=getattr(operator, "world_size", 1),
        partition=getattr(operator, "partition", "views"),
        partition_grid=getattr(operator, "partition_grid", None),
        global_volume_shape=list(operator.global_volume_shape), volume_shape=list(operator.volume_shape),
        volume_slice=[[part.start, part.stop] for part in operator.volume_slice],
        projection_shape=list(operator.projection_shape),
        view_slice=[operator.view_slice.start, operator.view_slice.stop],
        beam=operator.beam, voxel_spacing=operator.voxel_spacing,
        detector_spacing=list(operator.detector_spacing), detector_shape=list(operator.detector_shape),
        input_dtype=str(source.dtype), geometry=list(geometry),
        surface=[fingerprint(value, elements) for value in surface_values],
        effective_surface=_effective_surface(operator, frames, elements),
        compute_dtype=str(compute_dtype), precision=getattr(operator, "precision", None),
        backend=getattr(operator, "backend", None),
        measurements=fingerprint(source, elements),
    )


def signature(operator, source, algorithm, settings, elements, *, compute_dtype=None):
    error, metadata = None, None
    try:
        dtype = getattr(operator, "dtype", source.dtype) if compute_dtype is None else compute_dtype
        metadata = _signature_local(operator, source, algorithm, settings, elements, dtype)
    except Exception as caught:
        error = caught
    agreement(operator, error)
    if getattr(operator, "world_size", 1) > 1:
        records = [None] * operator.world_size
        dist.all_gather_object(records, metadata, group=operator.process_group)
        local_fields = {"rank", "volume_shape", "volume_slice", "projection_shape", "view_slice", "measurements"}
        scans = [{key: value for key, value in record.items() if key not in local_fields} for record in records]
        if any(scan != scans[0] for scan in scans):
            raise ValueError("distributed checkpoint scan and compute configuration must agree")
        replicas = {}
        for record in records:
            views = tuple(record["view_slice"])
            digest = record["measurements"]
            if views in replicas and replicas[views] != digest:
                raise ValueError("distributed checkpoint measurement replicas must agree")
            replicas[views] = digest
    return metadata


def load_checkpoint(operator, directory, expected, iterations, elements, names):
    error, saved, states = None, None, None
    try:
        directory = Path(directory).resolve()
        saved = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if not isinstance(saved, dict):
            raise ValueError("checkpoint metadata must be an object")
        # JSON canonicalization also normalizes tuple-valued physical spacing.
        canonical = json.loads(json.dumps(expected))
        if any(saved.get(key) != value for key, value in canonical.items()):
            raise ValueError("checkpoint scan, data, ownership, dtype or algorithm settings do not match")
        completed = count(saved.get("iteration"), "checkpoint iteration")
        history = saved.get("residual_history")
        if completed > iterations or not isinstance(history, list) or len(history) != completed + 1:
            raise ValueError("checkpoint iteration or residual history is invalid")
        for value in history:
            finite(value, "checkpoint residual")
        finite(saved.get("measurement_norm"), "checkpoint measurement norm")
        if expected["algorithm"] in ("cgls", "cgls_into"):
            finite(saved.get("gamma"), "checkpoint gamma")
        manifest = saved.get("states", {})
        if not isinstance(manifest, dict):
            raise ValueError("checkpoint states must be an object")
        if set(manifest) != set(names):
            raise ValueError("checkpoint does not contain the complete solver state")
        states = {}
        for name in names:
            specification = manifest[name]
            if not isinstance(specification, dict):
                raise ValueError("checkpoint state specification must be an object")
            store = NpyStore(directory / f"{name}.npy", mode="r")
            shape = operator.projection_shape if name in ("r", "q", "row") else operator.volume_shape
            if name == "step":
                shape = (1,)
            if tuple(store.shape) != tuple(shape) or list(store.shape) != specification.get("shape") or str(store.dtype) != specification.get("dtype"):
                raise ValueError("checkpoint solver state shape or dtype does not match")
            if fingerprint(store, elements) != specification.get("fingerprint"):
                raise ValueError("checkpoint solver state fingerprint does not match")
            states[name] = store
    except (ValueError, TypeError, OSError, EOFError, KeyError) as caught:
        error = caught
    token = None if error is not None else (saved["iteration"], saved["residual_history"], saved.get("gamma"), saved["measurement_norm"])
    agreement(operator, error, token)
    return saved, states


def save_checkpoint(operator, directory, states, metadata, iteration, history, measurement_norm, elements, **scalars):
    error, manifest, rank = None, {}, None
    try:
        step = Path(directory).resolve() / f"step-{iteration:06d}"
        step.mkdir(parents=True, exist_ok=True)
        rank = step / f"rank-{getattr(operator, 'rank', 0):05d}"
        rank.mkdir()
        for name, source in states.items():
            source = TensorStore(source.detach()) if isinstance(source, torch.Tensor) else source
            target = NpyStore.create(rank / f"{name}.npy", source.shape, dtype=source.dtype)
            copy_blocks(source, target, elements)
            target.flush()
            manifest[name] = dict(shape=list(source.shape), dtype=str(source.dtype), fingerprint=fingerprint(target, elements))
    except (OSError, ValueError, TypeError) as caught:
        error = caught
    # All owned states on every rank are written and flushed before any rank
    # atomically publishes the metadata marking a completed iteration.
    agreement(operator, error, (iteration, history))
    error = None
    try:
        document = dict(metadata, iteration=iteration, residual_history=history,
                        measurement_norm=measurement_norm, states=manifest, **scalars)
        temporary = rank / "metadata.tmp"
        with temporary.open("x", encoding="utf-8") as file:
            json.dump(document, file)
            file.flush()
        temporary.replace(rank / "metadata.json")
    except Exception as caught:
        error = caught
    # A failed metadata write/rename must stop every rank before another
    # projection or callback can enter a numerical collective.
    agreement(operator, error, iteration)


def callbacks_enabled(operator, callback):
    """Check callback presence once, including callbacks on only some ranks."""
    return bool(control_status((int(callback is not None),),
                               process_group=getattr(operator, "process_group", None),
                               world_size=getattr(operator, "world_size", 1))[0])


def callback_stop(operator, callback, iteration, residual, volume):
    error, stop = None, False
    try:
        stop = callback is not None and callback(iteration, residual, volume) is False
    except Exception as caught:
        error = caught
    failed, stop = control_status((int(error is not None), int(stop)),
                                  process_group=getattr(operator, "process_group", None),
                                  world_size=getattr(operator, "world_size", 1))
    if failed:
        agreement(operator, error)
    return bool(stop)
