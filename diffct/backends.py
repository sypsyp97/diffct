"""Portable, tiled Siddon traversal and backend selection."""

import itertools
import math
import numbers
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.autograd import forward_ad

from .chunking import _memory_lower_bounds
from .distribution import control_status
from .surfaces import ParameterizedSurface


# Maximum automatic ray/plane work array payload for host memory; caller
# data/results and legacy callback allocations are separate. Explicit chunk
# hints take priority.
_REFERENCE_WORK_BYTES = 8 * 1024 * 1024
# Live bytes per ray outside the crossing tables: ray setup, validation and
# FP64 clipping vectors (see `ReferenceProjector._work_model`).
_RAY_WORK_BYTES = 1024
# Largest element count of one automatic CUDA work array. Many PyTorch kernels
# use 32-bit indexing per tensor, so larger arrays only add risk, not speed.
_WORK_ELEMENTS = 2 ** 31 - 1


def _host_free_bytes():
    """Free host memory where the platform reports it, else the fixed 8 MiB ceiling."""
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return _REFERENCE_WORK_BYTES


# A geometry-gradient pair keeps its autograd graph and backward temporaries.
# Measured peak of one recomputed pair (forward, graph and autograd.grad): at
# most 1.27x ReferenceProjector._work_model of its plan and about 1.45x a plain
# forward pair. Geometry backward plans with 1/this of the CUDA budget (or, on
# the fixed 8 MiB plan, this many times the leading batch), leaving at least
# 1.5x slack for the previous pair's tables.
_GEOMETRY_WORK_FACTOR = 2


def _forward_tangent(value):
    """Forward-mode tangent of ``value`` at the current level, below vmap batching."""
    while torch._C._functorch.is_batchedtensor(value):
        value = torch._C._functorch.get_unwrapped(value)
    return forward_ad.unpack_dual(value).tangent


def _records_autograd(value):
    """Whether reverse-mode autograd can record ops on ``value``.

    Function transforms wrap tensors (vmap batches, jvp duals) whose own
    ``requires_grad`` is False even when a wrapped value or tangent requires it.
    """
    while isinstance(value, torch.Tensor):
        if value.requires_grad:
            return True
        tangent = _forward_tangent(value)
        if tangent is not None and _records_autograd(tangent):
            return True
        if not torch._C._functorch.is_functorch_wrapped_tensor(value):
            return False
        value = torch._C._functorch.get_unwrapped(value)
    return False


def _zero_dependency(value):
    """Connect a zero scalar without reading values or copying strided data."""
    return value.unsqueeze(0)[:0].sum()


def voxel_spacing_tuple(spacing, rank):
    """Return positive physical pitches in tensor axis order."""
    values = spacing if isinstance(spacing, (tuple, list)) else (spacing,) * rank
    if len(values) != rank:
        raise ValueError(f"voxel_spacing must be a scalar or have {rank} dimensions")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError("voxel_spacing must contain positive finite scalars")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("voxel_spacing must contain positive finite scalars")
        result.append(value)
    return tuple(result)


def select_backend(backend, precision, spacing, partition="views"):
    """Select native NVIDIA traversal for fast views/space execution."""
    if not isinstance(backend, str) or backend not in ("auto", "numba", "torch"):
        raise ValueError("backend must be 'auto', 'numba', or 'torch'")
    if not isinstance(precision, str) or precision not in ("fast", "float32", "float64"):
        raise ValueError("precision must be 'fast', 'float32', or 'float64'")
    compatible = precision == "fast" and partition != "hybrid"
    available = torch.cuda.is_available() and torch.version.hip is None
    if backend == "numba":
        if not compatible:
            raise ValueError("numba requires precision='fast' and views/space partition")
        if not available:
            raise RuntimeError("numba requires NVIDIA CUDA")
    return ("numba" if compatible and available else "torch") if backend == "auto" else backend


def _rectangles(shape, chunks):
    chunks = shape if chunks is None else chunks
    for starts in itertools.product(*(range(0, size, step) for size, step in zip(shape, chunks))):
        yield tuple(slice(start, min(start + step, size))
                    for start, step, size in zip(starts, chunks, shape))


def _flat_admission_batches(owner, geometry, detector_chunk_shape, *, batched_corners=False):
    """Bound flat admission with corners and exact ambiguous-view fallback."""
    views = owner.view_slice
    if views.start == views.stop:
        return
    # Flat point component arithmetic is monotone in each detector
    # coordinate; finite corner points and source differences bound the grid.
    endpoints = tuple((slice(0, 1),) if size == 1 else (slice(0, 1), slice(size - 1, size))
                      for size in owner.detector_shape)
    if batched_corners:
        # Native validation can compose the Cartesian corner set in one batch.
        # Stepped pixel slices retain the exact global endpoint coordinates.
        yield views, tuple(slice(0, size, max(1, size - 1)) for size in owner.detector_shape)
    else:
        for pixels in itertools.product(*endpoints):
            yield views, pixels
    if owner.beam in ("parallel", "parallel3d"):
        return
    source, lower = (value[views].to(dtype=owner.dtype) for value in geometry[:2])
    upper = lower
    for size, pitch, axis in zip(owner.detector_shape, owner.detector_spacing, geometry[2:]):
        coordinates = (torch.tensor([.5 - size / 2, size - .5 - size / 2], dtype=torch.float64)
                       * pitch).to(device=lower.device, dtype=owner.dtype)
        products = coordinates[None, :, None] * axis[views, None, :].to(dtype=owner.dtype)
        lower = lower + products.amin(dim=1)
        upper = upper + products.amax(dim=1)
    # Corners cannot exclude an interior pixel rounded onto the source.
    # A staged source outside this AABB cannot coincide; inconclusive views
    # retain the exact bounded pixel check rather than changing admission.
    ambiguous = torch.all((source >= lower) & (source <= upper), dim=-1)
    for index in torch.nonzero(ambiguous, as_tuple=False).flatten().tolist():
        selected = slice(views.start + index, views.start + index + 1)
        for pixels in _rectangles(owner.detector_shape, detector_chunk_shape):
            yield selected, pixels


_GEOMETRY_SECOND_DERIVATIVE = "diffct does not support second derivatives with respect to the geometry"
_GEOMETRY_FORWARD_DERIVATIVE = "diffct does not support forward-mode derivatives with respect to the geometry"


class _PortableApply(torch.autograd.Function):
    """Portable linear map whose backward is the matched adjoint.

    Autograd keeps the input (only for geometry gradients) and the geometry
    snapshot, never per-batch ray tables. Replicated distributed cotangents
    are SUMmed by the adjoint.
    """

    @staticmethod
    def forward(tensor, reference, is_project, reduce_cotangent, admit, participation, *geometry):
        return reference._tensor_apply(tensor, is_project, reduce_cotangent, geometry=geometry, admit=admit)

    @staticmethod
    def setup_context(ctx, inputs, output):
        tensor, reference, is_project, reduce_cotangent, admit, participation, *geometry = inputs
        ctx.reference, ctx.is_project = reference, is_project
        ctx.reduce_cotangent = reduce_cotangent
        ctx.dtype = tensor.dtype
        ctx.geometry_grad = _PortableApply._geometry_grad_needed(reference.owner, ctx.needs_input_grad[6:])
        ctx.save_for_backward(tensor if ctx.geometry_grad else None, *geometry)
        ctx.save_for_forward(tensor, *geometry)
        ctx.set_materialize_grads(False)

    @staticmethod
    def _geometry_grad_needed(owner, needs_input_grad):
        # Distributed ranks run the geometry collectives in lockstep.
        return owner._geometry_trainable if owner.world_size > 1 else any(needs_input_grad)

    @staticmethod
    def jvp(ctx, tangent, reference, is_project, reduce_cotangent, admit, participation, *geometry_tangents):
        if any(value is not None for value in geometry_tangents):
            raise RuntimeError(_GEOMETRY_FORWARD_DERIVATIVE)
        tensor, *geometry = ctx.saved_tensors
        tangent = torch.zeros_like(tensor) if tangent is None else tangent
        return _PortableApply.apply(tangent, ctx.reference, ctx.is_project, ctx.reduce_cotangent, False,
                                    ctx.reference.owner._participation(), *geometry)

    @staticmethod
    def vmap(info, in_dims, tensor, reference, is_project, reduce_cotangent, admit, participation, *geometry):
        if any(dimension is not None for dimension in in_dims[5:]):
            raise RuntimeError("portable vmap supports data batching with fixed geometry")
        dimension = in_dims[0]
        if dimension is not None:
            tensor = tensor.movedim(dimension, 0)
        result = _PortableApply.apply(tensor, reference, is_project, reduce_cotangent, admit,
                                      participation, *geometry)
        return result, 0 if dimension is not None else None

    @staticmethod
    def backward(ctx, cotangent):
        tensor, *geometry = ctx.saved_tensors
        owner = ctx.reference.owner
        gradient = None
        # The adjoint is this Function again, so data second derivatives work.
        # Distributed ranks always run it to keep its collectives in lockstep.
        if ctx.needs_input_grad[0] or owner.world_size > 1:
            gradient = _PortableApply.apply(cotangent, ctx.reference, not ctx.is_project, True, False,
                                            owner._participation(), *geometry).to(dtype=ctx.dtype)
        geometry_gradients = (None,) * len(geometry)
        if ctx.geometry_grad:
            gradients = _PortableGeometry.apply(ctx.reference, tensor, cotangent,
                                                ctx.is_project, ctx.reduce_cotangent, *geometry)
            geometry_gradients = tuple(value if needed else None
                                       for value, needed in zip(gradients, ctx.needs_input_grad[6:]))
        return gradient, None, None, None, None, None, *geometry_gradients


class _PortableGeometry(torch.autograd.Function):
    """First-order geometry VJP; data adjoints remain differentiable separately."""

    @staticmethod
    def forward(reference, tensor, cotangent, is_project, reduce_cotangent, *geometry):
        return reference._geometry_grad(tensor, cotangent, is_project, reduce_cotangent, geometry)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def vmap(info, in_dims, reference, tensor, cotangent, is_project, reduce_cotangent, *geometry):
        # Reverse-mode Jacobians batch cotangents; evaluate one VJP per entry.
        if any(dimension is not None for dimension in in_dims[5:]):
            raise RuntimeError("portable vmap supports data batching with fixed geometry")
        tensor_dimension, cotangent_dimension = in_dims[1:3]
        results = []
        for index in range(info.batch_size):
            results.append(_PortableGeometry.apply(
                reference,
                tensor if tensor_dimension is None else tensor.select(tensor_dimension, index),
                cotangent if cotangent_dimension is None else cotangent.select(cotangent_dimension, index),
                is_project, reduce_cotangent, *geometry,
            ))
        return tuple(torch.stack(values) for values in zip(*results)), (0,) * len(geometry)

    @staticmethod
    def backward(ctx, *gradients):
        raise RuntimeError(_GEOMETRY_SECOND_DERIVATIVE)

    @staticmethod
    def jvp(ctx, *tangents):
        raise RuntimeError(_GEOMETRY_SECOND_DERIVATIVE)


class ReferenceProjector:
    """Apply sparse ray intervals with gather and its scatter-add adjoint.

    Each ray batch visits bounded spatial tiles in the global physical frame.
    Storage scales with the planes crossed by those rays, never rays times
    every voxel. Autograd uses the matched adjoint for data and recomputes one
    ray batch and tile at a time for geometry gradients.
    """

    def __init__(self, owner, *, validate=True, synchronize=True):
        self.owner = owner
        self.spacing = voxel_spacing_tuple(owner.voxel_spacing, len(owner.global_volume_shape))
        self.chunk_shape, self.view_chunk_size, self.detector_chunk_shape = self._work_chunks()
        if not validate:
            # Call-time owners validate frames separately and sample surfaces
            # from the invocation's single geometry snapshot below.
            return
        with torch.enable_grad():
            geometry = self._snapshot_geometry(synchronize=synchronize)
        if (owner.detector_surface is not None
                and not isinstance(owner.detector_surface, ParameterizedSurface)):
            trainable = any(value.requires_grad for value in geometry)
            if synchronize and owner.world_size > 1:
                device = (torch.device("cuda", torch.cuda.current_device())
                          if dist.get_backend(owner.process_group) == "nccl" else torch.device("cpu"))
                flag = torch.tensor(int(trainable), device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=owner.process_group)
                trainable = bool(flag.item())
            owner._geometry_trainable = owner._geometry_trainable or trainable
        self._validate_geometry(geometry, geometry[0].device, admission=True,
                                synchronize=synchronize)

    def _admission_batches(self, geometry, plan):
        if self.owner.detector_surface is not None:
            yield from self._batches(*plan[1:])
        else:
            yield from _flat_admission_batches(self.owner, geometry, plan[2])

    def _work_budget(self, devices, data_devices=(), *, agree=False):
        """Automatic work limits as ``(cuda_bytes, host_bytes)``, or None for the 8 MiB plan.

        When every compute device is a CUDA device, ``cuda_bytes`` is a quarter
        of the smallest memory that the PyTorch allocator can still use on the
        compute devices and CUDA data devices when the call starts. Host data
        devices (CPU inputs, outputs or stores) add ``host_bytes`` for host
        staging copies: the smaller of a quarter of the free host memory and
        ``cuda_bytes``, but at least 8 MiB. Distributed ranks share the smallest
        limits, because their collectives follow the tile and batch order.
        Numerical traversal never records an autograd graph: gradients use
        `_PortableApply`, and geometry backward recomputes one pair at a time
        within ``1 / _GEOMETRY_WORK_FACTOR`` of these limits.
        """
        owner = self.owner
        if all(hint is not None for hint in (owner.volume_chunk_shape, owner.view_chunk_size,
                                             owner.detector_chunk_shape)):
            return None
        devices = tuple(dict.fromkeys(torch.device(device) for device in devices))
        data_devices = tuple(torch.device(device) for device in data_devices)
        budget = None
        if all(device.type == "cuda" for device in devices):
            cuda = tuple(dict.fromkeys((*devices, *(device for device in data_devices if device.type == "cuda"))))
            # _memory_lower_bounds keeps three quarters of the available memory.
            usable = min(_memory_lower_bounds(device)[0] // 3 for device in cuda) * 4
            host = (max(_REFERENCE_WORK_BYTES, min(_host_free_bytes() // 4, usable // 4))
                    if any(device.type != "cuda" for device in data_devices) else None)
            # No floor: with very little memory the plan shrinks to one voxel
            # and one ray per step, which still makes progress.
            budget = (usable // 4, host)
        if agree and owner.world_size > 1:
            # MAX of negated values gives the smallest limits. 1 marks the fixed
            # 8 MiB plan; -2**62 marks a rank without host staging.
            absent = -(1 << 62)
            shared, staged = control_status(
                (1 if budget is None else -budget[0],
                 absent if budget is None or budget[1] is None else -budget[1]),
                process_group=owner.process_group, world_size=owner.world_size,
            )
            budget = None if shared > 0 else (-shared, None if staged == absent else -staged)
        return budget

    @staticmethod
    def _halve(chunks):
        axis = max(range(len(chunks)), key=chunks.__getitem__)
        chunks[axis] = max(1, (chunks[axis] + 1) // 2)

    def _tile_fits(self, chunks, batch_elements, budget):
        """Whether one tile and one ray use at most half of each work budget."""
        cuda_bytes, host = budget
        return (sum(self._work_model(chunks, batch_elements)) <= cuda_bytes // 2
                and math.prod(chunks) * batch_elements <= _WORK_ELEMENTS
                and (host is None or sum(self._host_model(chunks, batch_elements)) <= host // 2))

    def _work_model(self, chunks, batch_elements):
        """Upper bound of live CUDA work bytes as (per volume tile, per ray).

        One ray batch with R rays and a tile with P = sum(chunks) + D + 2
        crossing planes needs at most tile + R * ray bytes. With b bytes per
        element, D dimensions and L leading batch elements:

        - tile: 3 * L * b + 32 per voxel. Volume tile copies or adjoint blocks
          (current, previous and zero-filled), a summed tile in distributed
          reductions, and int64 voxel indices of `_geometry_grad` and stores.
        - ray: P * ((4 + D + 2 * L) * b + 40) + 2 * L * b + 1024. The crossing
          table of `_intervals` keeps crossings, sorted times, lengths, midpoints
          and int64 voxel indices with two int64 temporaries. The previous
          tile's indices, lengths and gathered values are still alive, and
          the gather and its product need 2 * L * b. The constant covers ray
          setup, validation and the FP64 clipping vectors.

        Outputs are accumulated in place, so no full-output copy is made.
        """
        element_bytes = torch.empty((), dtype=self.owner.dtype).element_size()
        dimensions = len(chunks)
        planes = sum(chunks) + dimensions + 2
        tile = math.prod(chunks) * (3 * batch_elements * element_bytes + 32)
        ray = (planes * ((4 + dimensions + 2 * batch_elements) * element_bytes + 40)
               + 2 * batch_elements * element_bytes + _RAY_WORK_BYTES)
        return tile, ray

    def _host_model(self, chunks, batch_elements):
        """Upper bound of host staging bytes as (per volume tile, per ray).

        With host inputs or outputs, a volume tile is staged as a contiguous
        (and possibly cast) copy, or an adjoint block returns with its int64
        voxel indices: 2 * L * b + 32 per voxel. Each ray stages input or
        output values, int64 ray indices and FP64 detector coordinates:
        2 * L * b + 64.
        """
        element_bytes = torch.empty((), dtype=self.owner.dtype).element_size()
        return (math.prod(chunks) * (2 * batch_elements * element_bytes + 32),
                2 * batch_elements * element_bytes + 64)

    def _leading_limit(self, plan, budget, batch_elements, piece_bytes=0, on_host=False):
        """Largest leading batch whose work fits the CUDA and host limits of ``plan``.

        Both models are linear in the leading elements: fixed + L * each. A
        split stages one output piece of ``piece_bytes`` per leading element
        on the input's side before copying it into the returned array.
        """
        rays = plan[1] * math.prod(plan[2])
        result = None
        for model, limit, staged in ((self._work_model, budget[0], not on_host),
                                     (self._host_model, budget[1], on_host)):
            if limit is None:
                continue
            fixed = model(plan[0], 0)
            fixed = fixed[0] + rays * fixed[1]
            each = model(plan[0], 1)
            each = each[0] + rays * each[1] - fixed
            if staged and batch_elements * each + fixed > limit:
                each += piece_bytes
            allowed = max(1, (limit - fixed) // each)
            result = allowed if result is None else min(result, allowed)
        return result

    def _agreed_limit(self, limit):
        """The smallest ``limit`` of all ranks, so every rank splits leading dimensions alike."""
        owner = self.owner
        if owner.world_size == 1:
            return limit
        shared, = control_status((-limit,), process_group=owner.process_group, world_size=owner.world_size)
        return -shared

    @staticmethod
    def _leading_pieces(leading, batch_limit):
        """Leading-dimension slices with at most ``batch_limit`` elements, in row-major order.

        The largest axis splits first; a piece that is still too large splits
        again along its own largest axis.
        """
        def split(piece):
            sizes = [part.stop - part.start for part in piece]
            count = math.prod(sizes)
            if batch_limit is None or count <= batch_limit:
                yield piece
                return
            axis = max(range(len(sizes)), key=sizes.__getitem__)
            step = max(1, batch_limit // (count // sizes[axis]))
            for first in range(piece[axis].start, piece[axis].stop, step):
                yield from split((*piece[:axis], slice(first, min(first + step, piece[axis].stop)),
                                  *piece[axis + 1:]))

        yield from split(tuple(slice(0, size) for size in leading))

    def _work_chunks(self, batch_elements=1, budget=None):
        owner = self.owner
        chunks = list(owner.volume_chunk_shape or owner.global_volume_shape)
        element_bytes = torch.empty((), dtype=owner.dtype).element_size()
        batch_elements = max(1, batch_elements)
        dimensions = len(chunks)
        if budget is None:
            if owner.volume_chunk_shape is None:
                while (math.prod(chunks) * element_bytes * batch_elements > _REFERENCE_WORK_BYTES
                       or (sum(chunks) + dimensions + 2) * max(4, dimensions) * element_bytes
                       * batch_elements > _REFERENCE_WORK_BYTES) and max(chunks) > 1:
                    axis = max(range(dimensions), key=chunks.__getitem__)
                    chunks[axis] = max(1, (chunks[axis] + 1) // 2)
            plane_elements = (sum(chunks) + dimensions + 2) * max(4, dimensions)
            ray_limit = max(1, _REFERENCE_WORK_BYTES // (plane_elements * element_bytes * batch_elements))
        else:
            if owner.volume_chunk_shape is None:
                # A volume tile and one ray use at most half of each budget,
                # so ray batches keep at least the other half.
                while not self._tile_fits(chunks, batch_elements, budget) and max(chunks) > 1:
                    self._halve(chunks)
            cuda_bytes, host = budget
            tile, ray = self._work_model(chunks, batch_elements)
            # The largest per-ray tables are the gathered values (L per crossing)
            # and the crossing midpoints (D per crossing).
            elements = (sum(chunks) + dimensions + 2) * max(dimensions, batch_elements)
            ray_limit = min((cuda_bytes - tile) // ray, _WORK_ELEMENTS // elements)
            if host is not None:
                host_tile, host_ray = self._host_model(chunks, batch_elements)
                ray_limit = min(ray_limit, (host - host_tile) // host_ray)
            ray_limit = max(1, ray_limit)
        view_chunk = owner.view_chunk_size or min(32, owner._n_views)
        detector = list(owner.detector_chunk_shape or owner.detector_shape)
        if owner.detector_chunk_shape is None:
            while math.prod(detector) > max(1, ray_limit // view_chunk):
                axis = max(range(len(detector)), key=detector.__getitem__)
                detector[axis] = max(1, (detector[axis] + 1) // 2)
        if owner.view_chunk_size is None:
            view_chunk = min(view_chunk, max(1, ray_limit // math.prod(detector)))
        return tuple(chunks), view_chunk, tuple(detector)

    def _batches(self, view_chunk_size=None, detector_chunk_shape=None):
        owner = self.owner
        step = self.view_chunk_size if view_chunk_size is None else view_chunk_size
        for start in range(owner.view_slice.start, owner.view_slice.stop, step):
            views = slice(start, min(start + step, owner.view_slice.stop))
            for pixels in _rectangles(owner.detector_shape, detector_chunk_shape or self.detector_chunk_shape):
                yield views, pixels

    def _snapshot_geometry(self, *, synchronize=True):
        owner = self.owner
        frames = owner._learnable or owner._trajectory
        surface = owner.detector_surface
        if surface is None:
            return tuple(value.clone() for value in frames)
        error, metadata, snapshots = None, None, ()
        try:
            if isinstance(surface, ParameterizedSurface):
                # Parameters are sampled on CPU. Preserve their autograd state
                # when a sampler differentiates a potential during admission.
                with torch.enable_grad():
                    parameters = tuple(value.to(device="cpu", copy=True) for value in surface.parameters)
                snapshots = (*tuple(value.clone() for value in frames), *parameters)
            else:
                axes = [(torch.arange(size, device="cpu", dtype=torch.float64) + .5 - size / 2) * pitch
                        for size, pitch in zip(owner.detector_shape, owner.detector_spacing)]
                u, v = (torch.meshgrid(*axes, indexing="ij") if len(axes) == 2
                        else (axes[0], torch.zeros_like(axes[0])))

                def pack(value):
                    return value.detach().to(device="cpu").clone(), value.device

                def unpack(saved):
                    value, device = saved
                    return value.to(device=device)

                context = (nullcontext() if torch._C._are_functorch_transforms_active()
                           else torch.autograd.graph.saved_tensors_hooks(pack, unpack))
                with context:
                    offsets = surface(u, v)
                if not isinstance(offsets, torch.Tensor) or not torch.is_floating_point(offsets):
                    raise TypeError("detector_surface must return a floating-point tensor")
                if tuple(offsets.shape) not in ((*owner.detector_shape, 3),
                                               (owner._n_views, *owner.detector_shape, 3)):
                    raise ValueError("detector_surface output shape must match the complete detector grids")
                metadata = (tuple(offsets.shape), str(offsets.dtype))
                snapshots = (*tuple(value.clone() for value in frames), offsets.to(device="cpu", copy=True))
        except (TypeError, ValueError, RuntimeError) as failure:
            error = failure
        if synchronize:
            owner._partition.preflight(error, metadata, cache_key="portable surface snapshot",
                                       revision=owner._agreement_revision())
        elif error is not None:
            raise error
        return snapshots

    def _validate_geometry(self, geometry, fallback, *, admission=False, synchronize=True):
        """Finish every local geometry check before any rank starts numerics."""
        owner = self.owner
        if not admission and owner.detector_surface is None:
            return
        devices = tuple(torch.device(device) for device in (owner._devices or (fallback,)))
        error = None
        try:
            # Validation has no collectives, so each rank sizes its own batches.
            plan = self._work_chunks(budget=self._work_budget(devices))
            frame_count = 4 if owner.beam in ("cone", "parallel3d") else 3
            parameters = geometry[frame_count:]
            if isinstance(owner.detector_surface, ParameterizedSurface):
                with torch.enable_grad():
                    parameters = tuple(value.to(device="cpu") for value in parameters)
            with torch.no_grad():
                # Reuse frame transfers across detector chunks. Offset batches
                # remain bounded and move only when a ray batch needs them.
                staged = {device: (*tuple(value.to(device=device) for value in geometry[:frame_count]),
                                    *parameters) for device in set(devices)}
                batches = (self._admission_batches(staged[devices[0]], plan) if admission
                           else self._batches(*plan[1:]))
                for batch, (views, pixels) in enumerate(batches):
                    device = devices[batch % len(devices)]
                    origins, directions, _ = self._rays(
                        views, pixels, device, validate=True, geometry=staged[device],
                    )
                    self._check_rays(origins, directions)
        except (TypeError, ValueError, RuntimeError) as failure:
            error = failure
        if synchronize:
            owner._partition.preflight(error)
        elif error is not None:
            raise error

    def _check_rays(self, origins, directions):
        owner = self.owner
        if (not torch.isfinite(origins.to(dtype=owner.dtype)).all().item()
                or not torch.isfinite(directions.to(dtype=owner.dtype)).all().item()):
            raise ValueError("detector surface world coordinates must be finite")
        if (owner.beam not in ("parallel", "parallel3d")
                and torch.any(torch.all(directions == 0, dim=-1)).item()):
            raise ValueError("source and detector surface pixels must differ")

    def _rays(self, views, pixels, device, validate=False, stats=None, geometry=None, *, admit=False):
        """Rays of one view/pixel batch.

        ``admit`` applies the admission checks of ``_validate_geometry`` to the
        same sampled offsets before composing the numerical rays, so a
        parameterized sampler runs once for both.
        """
        owner = self.owner
        axes = [(torch.arange(pixel.start, pixel.stop, dtype=torch.float64, device="cpu")
                 + .5 - size / 2) * pitch
                for pixel, size, pitch in zip(pixels, owner.detector_shape, owner.detector_spacing)]
        u, v = (torch.meshgrid(*axes, indexing="ij") if len(axes) == 2
                else (axes[0], torch.zeros_like(axes[0])))
        ids = torch.arange(views.start, views.stop, device="cpu")
        surface = owner.detector_surface
        frame_count = 4 if owner.beam in ("cone", "parallel3d") else 3
        if isinstance(surface, ParameterizedSurface):
            parameters = tuple(self._transfer(value, torch.device("cpu"), stats)
                               for value in (surface.parameters if geometry is None else geometry[frame_count:]))
            offsets = surface.sampler(u, v, ids, *parameters)
        elif surface is not None:
            values = self._snapshot_geometry() if geometry is None else geometry
            offsets = values[frame_count]
            offsets = offsets[(slice(None), *pixels)] if offsets.ndim == u.ndim + 2 else offsets[pixels]
        else:
            offsets = torch.stack((u, v, torch.zeros_like(u)), dim=-1)
        if not isinstance(offsets, torch.Tensor) or not torch.is_floating_point(offsets):
            raise TypeError("detector_surface must return a floating-point tensor")
        shared_shape = (*u.shape, 3)
        if tuple(offsets.shape) == (owner._n_views, *shared_shape):
            offsets = offsets[views]
        elif tuple(offsets.shape) not in (shared_shape, (len(ids), *shared_shape)):
            raise ValueError("detector_surface output shape must match the selected view/pixel batch")
        if validate or admit:
            if not torch.isfinite(offsets).all().item():
                raise ValueError("detector_surface offsets must be finite")
            if owner.beam not in ("cone", "parallel3d") and torch.any(offsets[..., 1] != 0).item():
                raise ValueError("2D detector_surface middle offsets must be zero")
        def compose(validate, stats):
            check_range = (validate and owner.beam not in ("parallel", "parallel3d")
                           and owner.dtype == torch.float32 and surface is not None)
            dtype = torch.float64 if check_range else owner.dtype
            frames = (owner._learnable or owner._trajectory) if geometry is None else geometry[:frame_count]
            native_surface = owner.backend == "numba" and surface is not None
            native_legacy = native_surface and not isinstance(surface, ParameterizedSurface)
            if native_surface and not native_legacy:
                dtype = torch.float32
                for component in (*frames, offsets):
                    dtype = torch.promote_types(dtype, component.dtype)
            local_dtype = offsets.dtype if native_legacy else dtype
            local = self._transfer(offsets.to(dtype=local_dtype), device, stats)
            if local.ndim == u.ndim + 1:
                local = local.unsqueeze(0)
            dimensions = len(owner.global_volume_shape)
            selected = tuple(self._transfer(value[views].to(
                dtype=torch.promote_types(torch.float32, value.dtype) if native_legacy else dtype,
            ), device, stats) for value in frames)

            def frame(value):
                return value.reshape(len(ids), *([1] * u.ndim), dimensions)

            def world_points(offsets, components):
                axis_u = frame(components[2])
                points = frame(components[1]) + offsets[..., 0, None] * axis_u
                if owner.beam in ("cone", "parallel3d"):
                    axis_v = frame(components[3])
                    points = points + offsets[..., 1, None] * axis_v
                    normal_dtype = torch.promote_types(axis_u.dtype, axis_v.dtype)
                    normal = torch.linalg.cross(axis_u.to(dtype=normal_dtype),
                                                axis_v.to(dtype=normal_dtype), dim=-1)
                else:
                    normal = torch.stack((axis_u[..., 1], -axis_u[..., 0]), dim=-1)
                return points + offsets[..., 2, None] * normal

            if check_range and not native_surface:
                # Certify ordinary batches without expanding raw FP64 world points.
                # L1 norms bound Euclidean distances without squaring tiny bases.
                pitches = local.new_tensor(self.spacing[::-1])
                maximum = local.abs().amax(dim=tuple(range(1, local.ndim - 1)))
                axis_u = selected[2]
                if owner.beam == "cone":
                    basis = (axis_u, selected[3], torch.linalg.cross(axis_u, selected[3], dim=-1))
                    components = (0, 1, 2)
                else:
                    basis = (axis_u, torch.stack((axis_u[:, 1], -axis_u[:, 0]), dim=-1))
                    components = (0, 2)
                source_bound = torch.linalg.vector_norm(selected[0] / pitches, ord=1, dim=-1)
                pixel_bound = torch.linalg.vector_norm(selected[1] / pitches, ord=1, dim=-1)
                for component, axis in zip(components, basis):
                    pixel_bound = pixel_bound + maximum[:, component] * torch.linalg.vector_norm(
                        axis / pitches, ord=1, dim=-1,
                    )
                # Cover rounding in the few frame/offset operations; nonfinite or
                # boundary-adjacent bounds remain inconclusive and use exact points.
                slack = 64 * torch.finfo(torch.float64).eps
                source_bound = source_bound + slack * source_bound.clamp_min(1)
                pixel_bound = pixel_bound + slack * pixel_bound.clamp_min(1)
                certified = (torch.isfinite(source_bound) & torch.isfinite(pixel_bound)
                             & (torch.minimum(source_bound, pixel_bound) <= 1e6)
                             & (source_bound <= 1e15) & (pixel_bound <= 1e15))
                if certified.all().item():
                    check_range = False
                    dtype = owner.dtype
                    selected = tuple(value.to(dtype=dtype) for value in selected)
                    local = local.to(dtype=dtype)
            points = world_points(local, selected)
            if check_range:
                # Raw FP64 coordinates preserve the exact admission boundaries.
                sources = frame(selected[0]).expand_as(points)
                pitches = points.new_tensor(self.spacing[::-1])
                source_distance = torch.linalg.vector_norm(sources / pitches, dim=-1)
                pixel_distance = torch.linalg.vector_norm(points / pitches, dim=-1)
                if torch.any(torch.minimum(source_distance, pixel_distance) > 1e6).item():
                    raise ValueError(
                        "the source or a detector pixel must lie within 1e6 voxels "
                        "of the volume center in every view"
                    )
                if torch.any((source_distance > 1e15) | (pixel_distance > 1e15)).item():
                    raise ValueError(
                        "the source and detector surface pixels must lie within 1e15 voxels "
                        "of the volume center"
                    )
            if native_surface:
                # Native surface preprocessing composes before staging complete
                # rays to FP32. Data transforms must preserve that operator.
                selected = tuple(value.to(dtype=owner.dtype) for value in selected)
                points = points.to(dtype=owner.dtype)
            elif validate and dtype != owner.dtype:
                # Casting the completed point misses cancellation/overflow during
                # FP32 component arithmetic. Reuse the same sampled offsets.
                selected = tuple(value.to(dtype=owner.dtype) for value in selected)
                points = world_points(local.to(dtype=owner.dtype), selected)
            if owner.beam in ("parallel", "parallel3d"):
                origins, directions = points, frame(selected[0]).expand_as(points)
            else:
                sources = frame(selected[0]).expand_as(points)
                pitches = points.new_tensor(self.spacing[::-1])
                source_distance = torch.linalg.vector_norm(sources / pitches, dim=-1)
                pixel_distance = torch.linalg.vector_norm(points / pitches, dim=-1)
                if validate:
                    if torch.any(torch.all(
                        points.to(dtype=owner.dtype) == sources.to(dtype=owner.dtype), dim=-1,
                    )).item():
                        raise ValueError("source and detector surface pixels must differ")
                # Starting near the volume avoids subtracting its voxel planes
                # from a distant endpoint; reversing a segment preserves [0, 1].
                source_nearer = (source_distance <= pixel_distance).unsqueeze(-1)
                origins = torch.where(source_nearer, sources, points)
                difference = points - sources
                directions = torch.where(source_nearer, difference, -difference)

            pixel_ids = torch.meshgrid(*(torch.arange(pixel.start, pixel.stop, device=device)
                                         for pixel in pixels), indexing="ij")
            detector_ids = torch.zeros_like(pixel_ids[0])
            for coordinate, size in zip(pixel_ids, owner.detector_shape):
                detector_ids = detector_ids * size + coordinate
            ray_ids = ((self._transfer(ids, device, stats) - owner.view_slice.start).reshape(
                len(ids), *([1] * u.ndim),
            ) * math.prod(owner.detector_shape) + detector_ids).reshape(-1)
            return origins.reshape(-1, dimensions), directions.reshape(-1, dimensions), ray_ids

        if admit:
            with torch.no_grad():
                self._check_rays(*compose(True, None)[:2])
        return compose(validate, stats)

    def _intervals(self, origins, directions, tile):
        owner = self.owner
        world_tile = tile[::-1]
        shape = owner.global_volume_shape[::-1]
        spacing = self.spacing[::-1]
        # Clip and locate the tile entry using bounded per-ray FP64 vectors.
        # Global FP32 segment parameters lose voxel widths when both endpoints
        # are distant. Plane tables below use local physical distances and the
        # chosen precision, including for native data-transform dispatch.
        origins = origins.to(dtype=torch.float64)
        directions = directions.to(dtype=torch.float64)
        ray_length = torch.linalg.vector_norm(directions, dim=-1)
        directions = directions / ray_length[:, None]
        lower = origins.new_tensor([(part.start - size / 2) * pitch
                                    for part, size, pitch in zip(world_tile, shape, spacing)])
        upper = origins.new_tensor([(part.stop - size / 2) * pitch
                                    for part, size, pitch in zip(world_tile, shape, spacing)])
        constant = directions == 0
        safe_direction = torch.where(constant, torch.ones_like(directions), directions)
        first, last = (lower - origins) / safe_direction, (upper - origins) / safe_direction
        enter = torch.where(constant, -torch.inf, torch.minimum(first, last)).amax(dim=-1)
        leave = torch.where(constant, torch.inf, torch.maximum(first, last)).amin(dim=-1)
        if owner.beam not in ("parallel", "parallel3d"):
            enter, leave = enter.clamp_min(0), torch.minimum(leave, ray_length)
        inside = (~constant | ((origins >= lower) & (origins < upper))).all(dim=-1)
        valid = inside & (leave > enter)
        enter, leave = torch.where(valid, enter, 0), torch.where(valid, leave, 0)
        origins = (origins + directions * enter[:, None]).to(dtype=owner.dtype)
        directions = directions.to(dtype=owner.dtype)
        leave = (leave - enter).to(dtype=owner.dtype)
        enter = torch.zeros_like(leave)
        constant = directions == 0
        safe_direction = torch.where(constant, torch.ones_like(directions), directions)
        crossings = [enter[:, None], leave[:, None]]
        for axis, (part, size, pitch) in enumerate(zip(world_tile, shape, spacing)):
            planes = (torch.arange(part.start, part.stop + 1, device=origins.device,
                                   dtype=origins.dtype) - size / 2) * pitch
            times = (planes[None, :] - origins[:, axis, None]) / safe_direction[:, axis, None]
            times = torch.where(constant[:, axis, None], enter[:, None], times)
            crossings.append(torch.maximum(enter[:, None], torch.minimum(leave[:, None], times)))
        times = torch.cat(crossings, dim=-1).sort(dim=-1).values
        lengths = times.diff(dim=-1) * torch.linalg.vector_norm(directions, dim=-1)[:, None]
        midpoints = origins[:, None, :] + directions[:, None, :] * (
            (times[:, :-1] + times[:, 1:]) / 2
        )[..., None]
        indices = torch.zeros_like(lengths, dtype=torch.int64)
        for axis in range(len(shape) - 1, -1, -1):
            coordinate = torch.floor(midpoints[..., axis] / spacing[axis] + shape[axis] / 2)
            part = world_tile[axis]
            coordinate = coordinate.to(torch.int64).clamp(part.start, part.stop - 1) - part.start
            indices = indices * (part.stop - part.start) + coordinate
        return indices, lengths

    def _tiles(self, chunk_shape=None):
        owner = self.owner
        for local in _rectangles(owner.volume_shape, chunk_shape or self.chunk_shape):
            world = tuple(slice(part.start + owned.start, part.stop + owned.start)
                          for part, owned in zip(local, owner.volume_slice))
            yield local, world

    def _stats(self, is_project, plan=None):
        owner = self.owner
        chunks, views, pixels = plan or (self.chunk_shape, self.view_chunk_size, self.detector_chunk_shape)
        return dict(operation="project" if is_project else "backproject", backend="torch",
                    precision=owner.precision, dtype=str(owner.dtype),
                    schedule="spatial" if is_project else "views",
                    chunk_shape=chunks, view_chunk_size=views, detector_chunk_shape=pixels,
                    devices=[], ray_tile_pairs=0, store_reads=0, store_writes=0,
                    read_bytes=0, write_bytes=0, h2d_bytes=0, d2h_bytes=0, p2p_bytes=0,
                    collective_calls=0, collective_bytes=0)

    def _to(self, value, device, stats):
        value = value.to(dtype=self.owner.dtype)
        return self._transfer(value, device, stats)

    @staticmethod
    def _transfer(value, device, stats=None):
        source = value.device
        if source != device and stats is not None:
            key = ("h2d_bytes" if source.type == "cpu" else
                   "d2h_bytes" if device.type == "cpu" else "p2p_bytes")
            stats[key] += value.numel() * value.element_size()
        return value.to(device=device)

    def _device(self, batch, fallback, stats):
        devices = self.owner._devices or (fallback,)
        device = devices[batch % len(devices)]
        if str(device) not in stats["devices"]:
            stats["devices"].append(str(device))
        return device

    def _reduce(self, value, group, count, stats):
        if count == 1 or value.numel() == 0:
            return value
        device = (torch.device("cuda", torch.cuda.current_device()) if dist.get_backend(group) == "nccl"
                  else torch.device("cpu"))
        result = self._to(value, device, stats).clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
        stats["collective_calls"] += 1
        stats["collective_bytes"] += result.numel() * result.element_size()
        return self._to(result, value.device, stats)

    def _pixel_slices(self, views, pixels):
        return (slice(views.start - self.owner.view_slice.start, views.stop - self.owner.view_slice.start), *pixels)

    @staticmethod
    def _voxel_ids(tile, shape, device):
        coordinates = torch.meshgrid(*(torch.arange(part.start, part.stop, device=device)
                                       for part in tile), indexing="ij")
        indices = torch.zeros_like(coordinates[0])
        for coordinate, size in zip(coordinates, shape):
            indices = indices * size + coordinate
        return indices.flatten()

    # Numerical traversal never records autograd: `_PortableApply` provides
    # gradients, and in-place accumulation must not build CopySlices graphs.
    @torch.no_grad()
    def _tensor_apply(self, tensor, is_project, reduce_cotangent=False, *, geometry=None,
                      reduce_output=True, _plan=None, admit=False):
        owner = self.owner
        owner.last_execution_stats = None
        start = time.perf_counter()
        stats = self._stats(is_project)
        shape = owner.volume_shape if is_project else owner.projection_shape
        output_shape = owner.projection_shape if is_project else owner.volume_shape
        error, leading = None, None
        if not isinstance(tensor, torch.Tensor) or not torch.is_floating_point(tensor):
            error = TypeError("input must be a floating-point torch.Tensor")
        elif tensor.ndim < len(shape) or tuple(tensor.shape[-len(shape):]) != shape:
            error = ValueError(f"input trailing shape must be {shape}")
        else:
            leading = tuple(tensor.shape[:-len(shape)])
        owner._partition.preflight(error, (is_project, reduce_cotangent, leading),
                                   cache_key="portable input",
                                   revision=owner._agreement_revision())
        batch_elements = math.prod(leading)
        if _plan is None:
            budget = self._work_budget(owner._devices or (tensor.device,), (tensor.device,), agree=True)
            plan = self._work_chunks(batch_elements, budget)
        else:
            plan, budget = _plan
        stats = self._stats(is_project, plan)
        # Ray/voxel chunks cannot shrink below one. Split leading dimensions
        # as views when even that minimum would exceed the automatic payload.
        # Full explicit hints keep their priority over this automatic limit.
        # Pieces already fit, so a piece call (``_plan`` given) never splits.
        if _plan is None and any(hint is None for hint in (owner.volume_chunk_shape, owner.view_chunk_size,
                                                           owner.detector_chunk_shape)):
            if budget is None:
                planes = (sum(plan[0]) + len(shape) + 2) * max(4, len(shape))
                per_batch = max(math.prod(plan[0]), planes * plan[1] * math.prod(plan[2]))
                batch_limit = max(1, _REFERENCE_WORK_BYTES // (per_batch * torch.empty(
                    (), dtype=owner.dtype,
                ).element_size()))
            else:
                piece = math.prod(output_shape) * torch.empty((), dtype=owner.dtype).element_size()
                # The local output piece differs between ranks; all ranks use the smallest limit.
                batch_limit = self._agreed_limit(self._leading_limit(plan, budget, batch_elements, piece,
                                                                     on_host=tensor.device.type != "cuda"))
            if batch_elements > batch_limit:
                counters = ("ray_tile_pairs", "store_reads", "store_writes", "read_bytes", "write_bytes",
                            "h2d_bytes", "d2h_bytes", "p2p_bytes", "collective_calls", "collective_bytes")
                # Each piece is copied into the returned array, so only one piece is extra.
                result = _zero_dependency(tensor).to(dtype=owner.dtype).expand(
                    *leading, *output_shape).clone(memory_format=torch.contiguous_format)
                for piece in self._leading_pieces(leading, batch_limit):
                    result[piece] = self._tensor_apply(
                        tensor[piece], is_project, reduce_cotangent, geometry=geometry,
                        reduce_output=reduce_output, _plan=(plan, budget), admit=admit,
                    )
                    part_stats = owner.last_execution_stats
                    stats["schedule"] = part_stats["schedule"]
                    for key in counters:
                        stats[key] += part_stats[key]
                    for device in part_stats["devices"]:
                        if device not in stats["devices"]:
                            stats["devices"].append(device)
                stats["elapsed_seconds"] = time.perf_counter() - start
                owner.last_execution_stats = stats
                return result
        # The returned array is the only full-size output: it starts as zeros
        # (a differentiable seam for empty ownership and empty leading
        # dimensions) and every contribution accumulates in place.
        output = _zero_dependency(tensor).to(dtype=owner.dtype).expand(
            *leading, math.prod(output_shape)).clone(memory_format=torch.contiguous_format)
        if is_project:
            tiles = list(self._tiles(plan[0]))

            def accumulate(values, tile, world, origins, directions, rays, device):
                indices, lengths = self._intervals(origins, directions, world)
                local = self._to(values, device, stats).reshape(*leading, math.prod(values.shape[-len(tile):]))
                selected = torch.index_select(local, -1, indices.flatten()).reshape(*leading, *indices.shape)
                contributions = self._to((selected * lengths).sum(-1), tensor.device, stats)
                output.scatter_add_(-1, rays.expand(*leading, -1), contributions)
                stats["ray_tile_pairs"] += 1

            if reduce_cotangent:
                # A replicated cotangent is summed one tile at a time, so only
                # one summed tile is alive; each tile sets up the rays again.
                stats["schedule"] = "spatial"
                for tile_index, (tile, world) in enumerate(tiles):
                    values = self._reduce(tensor[(..., *tile)], owner.view_group, owner.view_world_size, stats)
                    for batch, (views, pixels) in enumerate(self._batches(*plan[1:])):
                        device = self._device(batch, tensor.device, stats)
                        origins, directions, rays = self._rays(views, pixels, device, stats=stats,
                                                               geometry=geometry,
                                                               admit=admit and tile_index == 0)
                        rays = self._transfer(rays, tensor.device, stats)
                        accumulate(values, tile, world, origins, directions, rays, device)
                    del values
            else:
                # Set up each ray batch once for all volume tiles. Every ray
                # still receives its tile contributions in tile order.
                stats["schedule"] = "views"
                for batch, (views, pixels) in enumerate(self._batches(*plan[1:])):
                    device = self._device(batch, tensor.device, stats)
                    origins, directions, rays = self._rays(views, pixels, device, stats=stats, geometry=geometry,
                                                           admit=admit)
                    rays = self._transfer(rays, tensor.device, stats)
                    for tile, world in tiles:
                        accumulate(tensor[(..., *tile)], tile, world, origins, directions, rays, device)
            if reduce_output and owner.space_world_size > 1:
                # Each ray belongs to one batch: replace its partial sum in place.
                for views, pixels in self._batches(*plan[1:]):
                    _, _, rays = self._rays(views, pixels, tensor.device, stats=stats, geometry=geometry)
                    values = torch.index_select(output, -1, rays)
                    values = self._reduce(values, owner.space_group, owner.space_world_size, stats)
                    output.scatter_(-1, rays.expand(*leading, -1), values)
        else:
            volume = output.view(*leading, *output_shape)
            for batch, (views, pixels) in enumerate(self._batches(*plan[1:])):
                device = self._device(batch, tensor.device, stats)
                values = tensor[(..., *self._pixel_slices(views, pixels))]
                if reduce_cotangent:
                    values = self._reduce(values, owner.space_group, owner.space_world_size, stats)
                values = self._to(values, device, stats).reshape(*leading, math.prod(values.shape[-len(shape):]))
                origins, directions, _ = self._rays(views, pixels, device, stats=stats, geometry=geometry,
                                                    admit=admit)
                for tile, world in self._tiles(plan[0]):
                    indices, lengths = self._intervals(origins, directions, world)
                    tile_shape = tuple(part.stop - part.start for part in tile)
                    contributions = (values[..., None] * lengths).reshape(*leading, indices.numel())
                    block = values.new_zeros((*leading, math.prod(tile_shape))).scatter_add(
                        -1, indices.flatten().expand(*leading, -1), contributions,
                    )
                    # Add the tile block into its strided output view: no voxel
                    # index tensor is built or kept by autograd.
                    volume[(..., *tile)].add_(self._to(block, tensor.device, stats).reshape(*leading, *tile_shape))
                    stats["ray_tile_pairs"] += 1
            if reduce_output and owner.view_world_size > 1:
                # Each voxel belongs to one tile: replace its partial sum in place.
                for tile, _ in self._tiles(plan[0]):
                    view = volume[(..., *tile)]
                    view.copy_(self._reduce(view, owner.view_group, owner.view_world_size, stats))
        stats["elapsed_seconds"] = time.perf_counter() - start
        owner.last_execution_stats = stats
        return output.reshape(*leading, *output_shape)

    def _apply(self, tensor, is_project):
        geometry = self._snapshot_geometry()
        owner = self.owner
        # A single process returns no output before every batch passes, so a
        # parameterized surface is admitted in the numerical sampling pass and
        # its sampler runs once per batch. Distributed calls finish all checks
        # before any rank starts numerics.
        admit = owner.world_size == 1 and isinstance(owner.detector_surface, ParameterizedSurface)
        if not admit:
            self._validate_geometry(geometry, tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu"))
        # Recording the traversal would keep ray tables for every tile and ray
        # batch. The Function keeps the geometry snapshot instead.
        if torch.is_grad_enabled() and (owner.world_size > 1 or any(
                _records_autograd(value) for value in (tensor, *geometry))):
            return _PortableApply.apply(tensor, self, is_project, False, admit, owner._participation(), *geometry)
        return self._tensor_apply(tensor, is_project, geometry=geometry, admit=admit)

    def _geometry_grad(self, tensor, cotangent, is_project, reduce_cotangent, geometry):
        owner = self.owner
        stats = self._stats(is_project)
        # Output replicas contribute one cotangent SUM before local ray math.
        output_shape = owner.projection_shape if is_project else owner.volume_shape
        input_shape = owner.volume_shape if is_project else owner.projection_shape
        leading = tuple(cotangent.shape[:-len(output_shape)])
        batch_elements = math.prod(leading)
        # One pair's geometry graph plus its backward temporaries is bounded by
        # _GEOMETRY_WORK_FACTOR times the forward work arrays of the same plan.
        replicas = owner.space_world_size if is_project else owner.view_world_size
        budget = self._work_budget(owner._devices or (cotangent.device,), (cotangent.device, tensor.device),
                                   agree=True)
        if budget is None:
            plan = self._work_chunks(max(1, batch_elements) * _GEOMETRY_WORK_FACTOR)
        else:
            budget = tuple(None if limit is None else limit // _GEOMETRY_WORK_FACTOR for limit in budget)
            plan = self._work_chunks(batch_elements, budget)
        # One cotangent-sized buffer is outside the work budget. Clone before
        # reshaping so a strided cotangent cannot create a second full copy.
        # Replicated sums replace disjoint batches or tiles in that same buffer.
        reduced = cotangent.detach()
        if replicas > 1:
            reduced = reduced.clone(memory_format=torch.contiguous_format)
        reduced = reduced.reshape(*leading, math.prod(output_shape))
        if replicas > 1 and is_project:
            for views, pixels in self._batches(*plan[1:]):
                _, _, ids = self._rays(views, pixels, cotangent.device, geometry=geometry)
                values = self._reduce(torch.index_select(reduced, -1, ids), owner.space_group,
                                      owner.space_world_size, stats)
                reduced.scatter_(-1, ids.expand(*leading, -1), values)
        elif replicas > 1:
            for tile, _ in self._tiles(plan[0]):
                ids = self._voxel_ids(tile, owner.volume_shape, cotangent.device)
                values = self._reduce(torch.index_select(reduced, -1, ids), owner.view_group,
                                      owner.view_world_size, stats)
                reduced.scatter_(-1, ids.expand(*leading, -1), values)
        # Split leading dimensions, as _tensor_apply does, when even the smallest
        # chunks exceed the automatic payload. Every rank uses the same pieces
        # in the same order.
        batch_limit = None
        if any(hint is None for hint in (owner.volume_chunk_shape, owner.view_chunk_size,
                                         owner.detector_chunk_shape)):
            if budget is None:
                planes = (sum(plan[0]) + len(input_shape) + 2) * max(4, len(input_shape))
                per_batch = max(math.prod(plan[0]), planes * plan[1] * math.prod(plan[2]))
                batch_limit = max(1, _REFERENCE_WORK_BYTES // (per_batch * _GEOMETRY_WORK_FACTOR * torch.empty(
                    (), dtype=owner.dtype,
                ).element_size()))
            else:
                batch_limit = self._agreed_limit(self._leading_limit(plan, budget, batch_elements,
                                                                     on_host=tensor.device.type != "cuda"))
        pieces = self._leading_pieces(leading, batch_limit)
        # Recompute one ray batch and tile at a time in the forward traversal
        # order, so input SUMs keep their collective order and only one pair's
        # graph is alive. The loss is <reduced cotangent, A_local(geometry) input>.
        leaves = tuple(value.detach().requires_grad_() for value in geometry)
        gradients = [torch.zeros_like(value) for value in leaves]

        def accumulate(loss, inputs, totals):
            # The zero dependency keeps unused inputs connected. With several
            # devices, a pair's loss and the geometry leaves can be on different
            # CUDA devices, so each zero moves to the loss's device.
            loss = loss + sum(_zero_dependency(value).to(loss.device) for value in inputs)
            for total, part in zip(totals, torch.autograd.grad(loss, inputs)):
                total.add_(part)

        with torch.enable_grad():
            for piece in pieces:
                data, weights_flat = tensor.detach()[piece], reduced[piece]
                part = tuple(data.shape[:len(leading)])
                if is_project:
                    for tile, world in self._tiles(plan[0]):
                        values = data[(..., *tile)]
                        if reduce_cotangent:
                            values = self._reduce(values, owner.view_group, owner.view_world_size, stats)
                        count = math.prod(values.shape[-len(tile):])
                        for batch, (views, pixels) in enumerate(self._batches(*plan[1:])):
                            device = self._device(batch, data.device, stats)
                            origins, directions, rays = self._rays(views, pixels, device, stats=stats,
                                                                   geometry=leaves)
                            indices, lengths = self._intervals(origins, directions, world)
                            local = self._to(values, device, stats).reshape(*part, count)
                            selected = torch.index_select(local, -1, indices.flatten()).reshape(*part, *indices.shape)
                            weights = self._to(torch.index_select(
                                weights_flat, -1, self._transfer(rays, weights_flat.device),
                            ), device, stats)
                            accumulate(((selected * lengths).sum(-1) * weights).sum(), leaves, gradients)
                else:
                    for batch, (views, pixels) in enumerate(self._batches(*plan[1:])):
                        device = self._device(batch, data.device, stats)
                        values = data[(..., *self._pixel_slices(views, pixels))]
                        if reduce_cotangent:
                            values = self._reduce(values, owner.space_group, owner.space_world_size, stats)
                        values = self._to(values, device, stats).reshape(
                            *part, math.prod(values.shape[-len(input_shape):]),
                        )
                        # Tiles pull back to this batch's rays; the rays pull back once.
                        origins, directions, _ = self._rays(views, pixels, device, stats=stats, geometry=leaves)
                        rays = (origins.detach().requires_grad_(), directions.detach().requires_grad_())
                        ray_gradients = [torch.zeros_like(value) for value in rays]
                        for tile, world in self._tiles(plan[0]):
                            indices, lengths = self._intervals(*rays, world)
                            voxels = self._voxel_ids(tile, owner.volume_shape, weights_flat.device)
                            weights = self._to(torch.index_select(weights_flat, -1, voxels), device, stats)
                            weights = torch.index_select(weights, -1, indices.flatten()).reshape(*part, *indices.shape)
                            accumulate((values[..., None] * lengths * weights).sum(), rays, ray_gradients)
                        accumulate((origins * ray_gradients[0]).sum() + (directions * ray_gradients[1]).sum(),
                                   leaves, gradients)
        result = []
        for value in gradients:
            flat_value = value.reshape(-1)
            pieces = []
            # Reuse the host staging payload ceiling for parameter-gradient SUMs.
            block_elements = max(1, _REFERENCE_WORK_BYTES // value.element_size())
            for start in range(0, value.numel(), block_elements):
                block = self._reduce(flat_value[start:start + block_elements], owner.process_group,
                                     owner.world_size, stats)
                pieces.append(block.to(dtype=value.dtype))
            result.append(torch.cat(pieces).reshape_as(value) if pieces else torch.zeros_like(value))
        return tuple(result)

    def into(self, source, sink, is_project):
        """Bounded numerical store I/O with the same sparse ray coefficients."""
        owner = self.owner
        owner.last_execution_stats = None
        start = time.perf_counter()
        fallback = torch.device(getattr(source, "device", "cpu"))
        output_device = torch.device(getattr(sink, "device", "cpu"))
        geometry = self._snapshot_geometry()
        self._validate_geometry(geometry, fallback)
        # CUDA compute devices use the CUDA budget; host stores bound host staging.
        plan = self._work_chunks(budget=self._work_budget(
            owner._devices or (fallback,), (fallback, output_device), agree=True,
        ))
        stats = self._stats(is_project, plan)
        stats["schedule"] = "views" if is_project else "spatial"

        def read(slices, device):
            value = source.read(slices)
            stats["store_reads"] += 1
            stats["read_bytes"] += value.numel() * value.element_size()
            return self._to(value, device, stats)

        def write(slices, value):
            value = self._to(value, output_device, stats)
            sink.write(slices, value)
            stats["store_writes"] += 1
            stats["write_bytes"] += value.numel() * value.element_size()

        if is_project:
            for batch, (views, pixels) in enumerate(self._batches(*plan[1:])):
                device = self._device(batch, fallback, stats)
                origins, directions, _ = self._rays(views, pixels, device, stats=stats, geometry=geometry)
                values = torch.zeros(len(origins), device=device, dtype=owner.dtype)
                for tile, world in self._tiles(plan[0]):
                    block = read(tile, device).reshape(-1)
                    indices, lengths = self._intervals(origins, directions, world)
                    values = values + (block[indices] * lengths).sum(-1)
                    stats["ray_tile_pairs"] += 1
                values = self._reduce(values, owner.space_group, owner.space_world_size, stats)
                shape = (views.stop - views.start, *(part.stop - part.start for part in pixels))
                write(self._pixel_slices(views, pixels), values.reshape(shape))
        else:
            for tile, world in self._tiles(plan[0]):
                tile_shape = tuple(part.stop - part.start for part in tile)
                values = torch.zeros(math.prod(tile_shape), device=output_device, dtype=owner.dtype)
                for batch, (views, pixels) in enumerate(self._batches(*plan[1:])):
                    device = self._device(batch, fallback, stats)
                    block = read(self._pixel_slices(views, pixels), device).reshape(-1)
                    origins, directions, _ = self._rays(views, pixels, device, stats=stats, geometry=geometry)
                    indices, lengths = self._intervals(origins, directions, world)
                    contribution = block.new_zeros(math.prod(tile_shape)).scatter_add(
                        0, indices.flatten(), (block[:, None] * lengths).flatten(),
                    )
                    values = values + self._to(contribution, output_device, stats)
                    stats["ray_tile_pairs"] += 1
                values = self._reduce(values, owner.view_group, owner.view_world_size, stats)
                write(tile, values.reshape(tile_shape))
        sink.flush()
        stats["elapsed_seconds"] = time.perf_counter() - start
        owner.last_execution_stats = stats

    def project(self, tensor):
        return self._apply(tensor, True)

    def backproject(self, tensor):
        return self._apply(tensor, False)
