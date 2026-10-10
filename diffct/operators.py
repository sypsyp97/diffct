"""High-level operators for arbitrary CT trajectories."""

import math
import numbers

import torch
from numba import cuda

from .chunking import _ChunkPlan, _largest_buffer_bytes, _memory_budget, _tiles, _working_set_bytes
from .projectors import (
    _geometry_vjp,
    _surface_backproject,
    _surface_project,
    ConeBackprojectorFunction,
    ConeProjectorFunction,
    FanBackprojectorFunction,
    FanProjectorFunction,
    ParallelBackprojectorFunction,
    ParallelProjectorFunction,
)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} dimensions must be positive integers")
    if value <= 0:
        raise ValueError(f"{name} dimensions must be positive integers")
    return int(value)


def _shape(value, rank, name):
    if not isinstance(value, (tuple, list)) or len(value) != rank:
        raise TypeError(f"{name} must have {rank} dimensions")
    return tuple(_positive_int(dim, name) for dim in value)


def _positive_float(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a positive finite scalar")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return value


def _detector_spacing(value, beam):
    if beam != "cone":
        if isinstance(value, (tuple, list)):
            raise TypeError("2D detector_spacing must be a scalar")
        return (_positive_float(value, "detector_spacing"),)
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise TypeError("cone detector_spacing must be a scalar or pair")
        return tuple(_positive_float(item, "detector_spacing") for item in value)
    pitch = _positive_float(value, "detector_spacing")
    return (pitch, pitch)


def _validate_trajectory(trajectory, beam, *, detector_surface=False):
    dimensions = 3 if beam == "cone" else 2
    count = 4 if beam == "cone" else 3
    if not isinstance(trajectory, (tuple, list)) or len(trajectory) != count:
        raise TypeError(f"{beam} trajectory must contain {count} tensors")

    geometry = []
    n_views = None
    for component in trajectory:
        if not isinstance(component, torch.Tensor):
            raise TypeError("trajectory components must be tensors")
        if component.ndim != 2 or component.shape[1] != dimensions:
            raise ValueError(f"trajectory tensors must have shape (views, {dimensions})")
        if not torch.is_floating_point(component):
            raise TypeError("trajectory tensors must have floating-point dtype")
        if component.shape[0] == 0:
            raise ValueError("trajectory must contain at least one view")
        if n_views is not None and component.shape[0] != n_views:
            raise ValueError("trajectory tensors must have the same number of views")
        n_views = component.shape[0]
        if not torch.isfinite(component).all().item():
            raise ValueError("trajectory tensors must contain only finite values")
        geometry.append(component.detach().clone())

    axes = (geometry[0], geometry[2]) if beam == "parallel" else (
        (geometry[2], geometry[3]) if beam == "cone" else (geometry[2],)
    )
    for axis in axes:
        norms = torch.linalg.vector_norm(axis, dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4):
            raise ValueError("trajectory direction axes must be unit vectors")
    if beam in ("parallel", "cone"):
        first, second = (geometry[0], geometry[2]) if beam == "parallel" else axes
        dots = torch.sum(first * second.to(first.device), dim=1)
        if torch.any(torch.abs(dots) > 1e-4).item():
            raise ValueError("trajectory direction axes must be orthogonal")
    if beam in ("fan", "cone") and not detector_surface:
        # float64 avoids overflow of half-precision input and dtype mismatches in cross().
        source = geometry[0].double()
        principal = geometry[1].to(source.device, torch.float64) - source
        lengths = torch.linalg.vector_norm(principal, dim=1)
        if torch.any(lengths == 0).item():
            raise ValueError("source and detector center must differ in every view")
        principal = principal / lengths[:, None]
        det_u = geometry[2].to(source.device, torch.float64)
        if beam == "fan":
            facing = principal[:, 0] * det_u[:, 1] - principal[:, 1] * det_u[:, 0]
        else:
            normal = torch.linalg.cross(det_u, geometry[3].to(source.device, torch.float64), dim=1)
            facing = torch.sum(principal * normal, dim=1)
        if torch.any(torch.abs(facing) < 1e-4).item():
            raise ValueError("the detector must not be edge-on to the source in any view")

    return tuple(geometry), n_views


def _normalize_devices(devices):
    if devices is None:
        return None
    if not isinstance(devices, (tuple, list)) or not devices:
        raise ValueError("devices must be a nonempty sequence of CUDA devices")

    normalized = []
    for value in devices:
        if isinstance(value, bool):
            raise TypeError("devices must contain CUDA devices")
        if isinstance(value, int):
            device = torch.device("cuda", value)
        else:
            try:
                device = torch.device(value)
            except (TypeError, RuntimeError) as error:
                raise TypeError("devices must contain CUDA devices") from error
        if device.type != "cuda":
            raise TypeError("devices must contain CUDA devices")
        if not torch.cuda.is_available():
            raise ValueError("CUDA devices are unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is not available")
        normalized.append(torch.device("cuda", index))
    if len(set(normalized)) != len(normalized):
        raise ValueError("devices must be unique")
    return tuple(normalized)


def _balanced_slice(n_views, rank, world_size):
    base, extra = divmod(n_views, world_size)
    start = rank * base + min(rank, extra)
    return slice(start, start + base + int(rank < extra))


# Operator modes and their adjoints. "project_reduced" sums its input over
# ranks before projecting; it is the adjoint of the replicated backprojection.
_ADJOINT_MODE = {
    "project": "backproject",
    "backproject": "project_reduced",
    "project_reduced": "backproject",
}


def _all_reduce_copy(projector, tensor, plan=None):
    """Return a SUM over ranks of ``tensor`` (the tensor itself without ranks)."""
    if not (projector._distributed and projector.world_size > 1):
        return tensor
    reduced = tensor.detach().clone(memory_format=torch.contiguous_format)
    if plan is not None:
        projector._reduce_streamed_(reduced, plan)
        return reduced
    with torch.cuda.device(reduced.device):
        torch.distributed.all_reduce(
            reduced, op=torch.distributed.ReduceOp.SUM, group=projector._process_group
        )
    return reduced


class _ProjectorAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, projector, mode, plan, *geometry):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("input must be a torch.Tensor")
        ctx.projector = projector
        ctx.mode = mode
        ctx.plan = plan
        ctx.input_device = tensor.device
        ctx.input_dtype = tensor.dtype
        needs_geometry = any(component.requires_grad for component in geometry) or (
            projector._distributed and projector.world_size > 1
            and projector._geometry_trainable
        )
        ctx.save_for_backward(tensor if needs_geometry else None, *geometry)
        return projector._run(
            tensor,
            is_project=mode != "backproject",
            reduce_cotangent=mode == "project_reduced",
            geometry=geometry if geometry else None,
            plan=plan,
        )

    @staticmethod
    def backward(ctx, grad_output):
        tensor, *geometry = ctx.saved_tensors
        projector = ctx.projector
        # Both branches below contain collectives. In distributed mode they run
        # on every rank, so ranks whose inputs differ in requires_grad still
        # issue the same collective sequence.
        collective = projector._distributed and projector.world_size > 1
        grad_input = None
        if ctx.needs_input_grad[0] or collective:
            # The adjoint is this Function again, so second derivatives with
            # respect to volumes and sinograms work.
            grad_input = _ProjectorAutograd.apply(
                grad_output, projector, _ADJOINT_MODE[ctx.mode], ctx.plan, *geometry
            ).to(device=ctx.input_device, dtype=ctx.input_dtype)
            if not ctx.needs_input_grad[0]:
                grad_input = None
        geometry_grads = (None,) * len(geometry)
        if any(ctx.needs_input_grad[4:]) or (
            collective and projector._geometry_trainable
        ):
            # Every mode is <cotangent, A_local(geometry) volume> for some pair.
            if ctx.mode == "project":
                volume, cotangent = tensor, grad_output
            elif ctx.mode == "project_reduced":
                volume, cotangent = _all_reduce_copy(projector, tensor, ctx.plan), grad_output
            else:
                volume, cotangent = _all_reduce_copy(projector, grad_output, ctx.plan), tensor
            grads = _GeometryGradAutograd.apply(projector, volume, cotangent, ctx.plan, *geometry)
            geometry_grads = tuple(
                grad if needed else None
                for grad, needed in zip(grads, ctx.needs_input_grad[4:])
            )
        return (grad_input, None, None, None, *geometry_grads)


class _GeometryGradAutograd(torch.autograd.Function):
    """First-order geometry gradient; differentiating it again raises."""

    @staticmethod
    def forward(ctx, projector, volume, cotangent, plan, *geometry):
        return projector._geometry_grad(volume, cotangent, geometry, plan)

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise RuntimeError(
            "diffct does not support second derivatives with respect to the geometry"
        )


class Projector:
    """Project CPU or CUDA volumes along parallel, fan, or cone trajectories.

    A parallel trajectory is ``(ray_dir, det_origin, det_u)`` and a fan
    trajectory is ``(src_pos, det_center, det_u)``; each component has shape
    ``(views, 2)``. A cone trajectory is
    ``(src_pos, det_center, det_u, det_v)`` with components shaped
    ``(views, 3)``. Direction axes are unit vectors; parallel ``ray_dir`` and
    ``det_u`` and cone ``det_u`` and ``det_v`` are orthogonal. If no trajectory
    tensor requires gradients, the projector clones the geometry. Otherwise it
    keeps references, reads the current values at every call, and returns
    gradients for those tensors; the checks above run only at construction.
    Second derivatives are available for volumes and sinograms, not for the
    geometry.

    ``detector_surface(u, v)`` optionally supplies local ``(u, v, n)`` offsets
    for every pixel. It receives centred physical float64 grids on CPU and
    returns a floating-point tensor shaped ``(*detector_shape, 3)`` or
    ``(views, *detector_shape, 3)``. Cone normals are ``cross(det_u, det_v)``;
    2D normals are ``(det_u_y, -det_u_x)`` with zero middle offsets. Offsets
    must be on CPU for streamed execution. CUDA offsets remain supported on a
    fitting full-volume CUDA path. Constructor world validation runs on CPU.
    The callback is sampled and validated on every call,
    with first-order gradients for captured PyTorch parameters. Omitting it
    preserves the flat detector.

    Volumes use ``(H, W)`` for parallel and fan beams and ``(D, H, W)`` for
    cone beams. Sinograms use ``(views, detectors)`` or ``(views, U, V)``.
    Results are float32 and returned to the input tensor's device. CPU inputs
    stream spatial tiles and view batches through CUDA. CUDA inputs retain the
    full-volume path when its additional working set fits every device;
    otherwise they stream too. ``volume_chunk_shape`` sets explicit spatial
    limits in tensor order; ``view_chunk_size`` sets an explicit view limit.
    Either forces streaming. With no overrides, sizing uses live free memory,
    reusable allocator cache and the observable per-process fraction ceiling,
    retaining 25% headroom. It halves the largest spatial axis until buffers
    fit; automatic view batches are at most 32 and shrink if needed. Older
    PyTorch without a public fraction getter cannot account for that ceiling.
    Concurrent external allocations can still cause an allocation error.

    Full host volumes/sinograms stay on CPU; streamed native CUDA buffers are
    bounded by the chosen tile and view batch. Caller-owned CUDA data/results
    and CUDA work inside a supplied callback are outside this memory bound.
    CPU multi-device transfers currently serialize device batches, so no
    speedup is guaranteed. Local
    devices split views in list order; distributed mode returns rank-local
    projections and replicated backprojections. Distributed projection losses
    use SUM across ranks; divide replicated backprojection losses by
    ``world_size`` and do not add a second DDP reduction for image or geometry
    gradients.
    """

    def __init__(self, trajectory, volume_shape, detector_shape, *, beam="cone",
                 detector_spacing=1.0, voxel_spacing=1.0, devices=None,
                 distributed=False, process_group=None, detector_surface=None,
                 volume_chunk_shape=None, view_chunk_size=None):
        if beam not in ("parallel", "fan", "cone"):
            raise ValueError("beam must be 'parallel', 'fan', or 'cone'")
        self.beam = beam
        rank = 3 if beam == "cone" else 2
        self.volume_shape = _shape(volume_shape, rank, "volume_shape")
        self.volume_chunk_shape = (
            None if volume_chunk_shape is None
            else _shape(volume_chunk_shape, rank, "volume_chunk_shape")
        )
        self.view_chunk_size = (
            None if view_chunk_size is None
            else _positive_int(view_chunk_size, "view_chunk_size")
        )
        if beam == "cone":
            self.detector_shape = _shape(detector_shape, 2, "detector_shape")
        elif isinstance(detector_shape, (tuple, list)):
            self.detector_shape = _shape(detector_shape, 1, "detector_shape")
        else:
            self.detector_shape = (_positive_int(detector_shape, "detector_shape"),)

        self.detector_spacing = _detector_spacing(detector_spacing, beam)
        self.voxel_spacing = _positive_float(voxel_spacing, "voxel_spacing")
        if detector_surface is not None and not callable(detector_surface):
            raise TypeError("detector_surface must be callable or None")
        self.detector_surface = detector_surface
        self._trajectory, n_views = _validate_trajectory(
            trajectory, beam, detector_surface=detector_surface is not None
        )
        self._n_views = n_views
        self._learnable = (
            tuple(trajectory) if any(c.requires_grad for c in trajectory) else ()
        )
        surface_trainable = False
        if detector_surface is not None:
            with torch.enable_grad():
                _, surface_offsets = self._sample_surface(
                    cpu_only=self.volume_chunk_shape is not None or self.view_chunk_size is not None,
                    validate_on_cpu=True,
                )
                surface_trainable = surface_offsets.requires_grad
        self._geometry_trainable = bool(self._learnable) or surface_trainable
        if beam in ("fan", "cone") and detector_surface is None:
            # The kernels set up each ray in float32 from the endpoint nearer the
            # volume centre, which is the origin.
            distances = torch.stack([
                torch.linalg.vector_norm(component.double(), dim=1).cpu()
                for component in self._trajectory[:2]
            ]) / self.voxel_spacing
            if torch.any(distances.min(dim=0).values > 1e6).item():
                raise ValueError(
                    "the source or the detector center must lie within 1e6 voxels "
                    "of the volume center in every view"
                )
            if torch.any(distances > 1e15).item():
                raise ValueError(
                    "the source and the detector center must lie within 1e15 voxels "
                    "of the volume center"
                )
        self._geometry_cache = {}
        self._devices = _normalize_devices(devices)
        if not isinstance(distributed, bool):
            raise TypeError("distributed must be a bool")
        self._distributed = distributed
        if process_group is not None and not self._distributed:
            raise ValueError("process_group requires distributed=True")
        if self._distributed:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("distributed=True requires an initialized process group")
            try:
                self.rank = torch.distributed.get_rank(process_group)
                self.world_size = torch.distributed.get_world_size(process_group)
            except (RuntimeError, ValueError) as error:
                raise ValueError("process_group must include the calling rank") from error
            if self.rank < 0 or self.world_size <= 0 or self.rank >= self.world_size:
                raise ValueError("process_group must include the calling rank")
        else:
            self.rank = 0
            self.world_size = 1
        self._process_group = process_group
        if self._distributed and self.world_size > 1:
            # Backward issues an extra all_reduce for geometry gradients, so
            # every rank must agree on whether the trajectory requires them.
            backend = torch.distributed.get_backend(process_group)
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if backend == "nccl" else torch.device("cpu")
            )
            flag = 1 if self._geometry_trainable else 0
            bounds = torch.tensor([flag, -flag], device=device)
            torch.distributed.all_reduce(
                bounds, op=torch.distributed.ReduceOp.MAX, group=process_group
            )
            if bounds[0].item() != -bounds[1].item():
                raise ValueError(
                    "all ranks must agree on whether the geometry requires gradients"
                )
        self.view_slice = _balanced_slice(n_views, self.rank, self.world_size)
        self.projection_shape = (
            self.view_slice.stop - self.view_slice.start,
            *self.detector_shape,
        )

    def _geometry_for(self, device, global_slice, geometry=None):
        if geometry is not None:
            return tuple(
                component.detach()[global_slice]
                .to(device=device, dtype=torch.float32)
                .contiguous()
                for component in geometry
            )
        if self._learnable:
            # Learnable geometry changes between calls, so it is staged every time.
            return tuple(
                component.detach()[global_slice]
                .to(device=device, dtype=torch.float32)
                .contiguous()
                for component in self._learnable
            )
        key = (device.index, global_slice.start, global_slice.stop)
        stream = torch.cuda.current_stream(device)
        cached = self._geometry_cache.get(key)
        if cached is None:
            geometry = tuple(
                component[global_slice]
                .to(device=device, dtype=torch.float32, non_blocking=True)
                .contiguous()
                for component in self._trajectory
            )
            ready = torch.cuda.Event()
            ready.record(stream)
            self._geometry_cache[key] = (geometry, ready)
        else:
            geometry, ready = cached
            stream.wait_event(ready)
        for component in geometry:
            component.record_stream(stream)
        return geometry

    def _execution_plan(self, tensor, is_project):
        self._validate_input(tensor, is_project)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to execute the projector")
        devices = self._devices or (
            tensor.device if tensor.is_cuda else torch.device("cuda", torch.cuda.current_device()),
        )
        local_views = self.projection_shape[0]
        active_devices = devices[:min(len(devices), local_views)]
        collective_devices = ()
        if self._distributed and self.world_size > 1 and torch.distributed.get_backend(self._process_group) == "nccl":
            collective_devices = (devices[0], torch.device("cuda", torch.cuda.current_device()))
        inventory = tuple(dict.fromkeys((
            *active_devices, *((tensor.device,) if tensor.is_cuda else ()), *collective_devices,
        )))
        observed = {device: _memory_budget(device) for device in inventory}
        budgets = {device: limits[0] for device, limits in observed.items()}
        allocation_limit = min((limits[1] for limits in observed.values()), default=(1 << 63) - 1)
        surface = self.detector_surface is not None
        # Sampling precedes view sharding. Conservatively allow the complete
        # float64 world-point expression/snapshot on each participating device,
        # without calling the callback again just to discover its device.
        world_bytes = 4 * 8 * self._n_views * math.prod(self.detector_shape) * len(self.volume_shape) if surface else 0
        volume_elements = math.prod(self.volume_shape)
        projection_elements = math.prod(self.projection_shape)
        can_use_full = tensor.is_cuda and self.volume_chunk_shape is None and self.view_chunk_size is None
        for device_rank, device in enumerate(devices):
            shard = _balanced_slice(local_views, device_rank, len(devices))
            views = shard.stop - shard.start
            if views == 0:
                continue
            needed = _working_set_bytes(self.beam, self.volume_shape, views, self.detector_shape, surface)
            needed += world_bytes
            if tensor.is_cuda and device == tensor.device:
                needed += 8 * (volume_elements + projection_elements)
            can_use_full = can_use_full and needed <= budgets[device]
            can_use_full = can_use_full and _largest_buffer_bytes(
                self.beam, self.volume_shape, views, self.detector_shape, surface
            ) <= allocation_limit
        if tensor.is_cuda and tensor.device not in active_devices:
            can_use_full = can_use_full and 8 * (volume_elements + projection_elements) + world_bytes <= budgets[tensor.device]
        can_use_full = can_use_full and all(world_bytes <= amount for amount in budgets.values())
        if tensor.is_cuda:
            can_use_full = can_use_full and 4 * max(volume_elements, projection_elements) <= allocation_limit

        # CUDA results stay on the caller's device. CPU execution never reserves
        # a full volume/sinogram there. Include transfer scratch on an origin
        # device even when it is not one of the selected compute devices.
        streamed_budgets = dict(budgets)
        if tensor.is_cuda:
            streamed_budgets[tensor.device] -= 4 * max(volume_elements, projection_elements)
        budget = min(streamed_budgets.values(), default=(1 << 63) - 1)
        if tensor.is_cuda and 4 * max(volume_elements, projection_elements) > allocation_limit:
            budget = -1
        if self._distributed and self.world_size > 1:
            control_device = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.distributed.get_backend(self._process_group) == "nccl" else torch.device("cpu")
            )
            settings = torch.tensor(
                (*self.volume_shape, *(self.volume_chunk_shape or (0,) * len(self.volume_shape)),
                 self.view_chunk_size or 0), dtype=torch.int64, device=control_device,
            )
            lower, upper = settings.clone(), settings.clone()
            torch.distributed.all_reduce(lower, op=torch.distributed.ReduceOp.MIN, group=self._process_group)
            torch.distributed.all_reduce(upper, op=torch.distributed.ReduceOp.MAX, group=self._process_group)
            if not torch.equal(lower, upper):
                raise ValueError("distributed ranks must agree on volume shape and explicit chunk limits")
            agreement = torch.tensor([int(can_use_full), budget, allocation_limit], dtype=torch.int64, device=control_device)
            torch.distributed.all_reduce(agreement, op=torch.distributed.ReduceOp.MIN, group=self._process_group)
            can_use_full, budget, allocation_limit = agreement.cpu().tolist()
        if can_use_full:
            return None

        chunk = tuple(min(limit, size) for limit, size in zip(
            self.volume_chunk_shape or self.volume_shape, self.volume_shape
        ))
        batch = self.view_chunk_size or min(32, self._n_views)
        minimum_shape = (1,) * len(chunk) if self.volume_chunk_shape is None else chunk

        def fits(shape):
            return (
                _working_set_bytes(self.beam, shape, batch, self.detector_shape, surface) <= budget
                and _largest_buffer_bytes(self.beam, shape, batch, self.detector_shape, surface) <= allocation_limit
            )

        while not fits(minimum_shape):
            if self.view_chunk_size is not None or batch == 1:
                raise RuntimeError("CUDA memory budget cannot fit the minimum tile/view batch; reduce explicit limits")
            batch = max(1, batch // 2)
        while not fits(chunk):
            if self.volume_chunk_shape is not None or max(chunk) == 1:
                raise RuntimeError("CUDA memory budget cannot fit the requested tile/view batch")
            axis = max(range(len(chunk)), key=chunk.__getitem__)
            smaller = list(chunk)
            smaller[axis] = max(1, smaller[axis] // 2)
            chunk = tuple(smaller)
        return _ChunkPlan(chunk, batch, devices)

    def _validate_input(self, tensor, is_project):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("input must be a torch.Tensor")
        expected_shape = self.volume_shape if is_project else self.projection_shape
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"input shape must be {expected_shape}")
        if not torch.is_floating_point(tensor):
            raise TypeError("input tensor must have floating-point dtype")
        if tensor.device.type not in ("cpu", "cuda"):
            raise TypeError("input tensor must be on CPU or CUDA")

    def _run(self, tensor, is_project, reduce_cotangent=False, geometry=None, plan=None):
        self._validate_input(tensor, is_project)
        if plan is not None:
            return self._run_streamed(tensor, is_project, reduce_cotangent, geometry, plan)
        if not tensor.is_cuda:
            raise TypeError("input tensor must be on CUDA")
        input_device = tensor.device
        if reduce_cotangent and self._distributed and self.world_size > 1:
            tensor = tensor.detach().clone(memory_format=torch.contiguous_format)
            with torch.cuda.device(input_device):
                torch.distributed.all_reduce(
                    tensor,
                    op=torch.distributed.ReduceOp.SUM,
                    group=self._process_group,
                )

        devices = self._devices or (input_device,)
        local_views = self.projection_shape[0]

        # Stage every input and geometry shard before launching any raw kernel.
        staged = []
        for device_rank, device in enumerate(devices):
            local_slice = _balanced_slice(local_views, device_rank, len(devices))
            if local_slice.start == local_slice.stop:
                continue
            global_slice = slice(
                self.view_slice.start + local_slice.start,
                self.view_slice.start + local_slice.stop,
            )
            source = tensor if is_project else tensor[local_slice]
            with torch.cuda.device(device):
                staged_input = source.to(
                    device=device, dtype=torch.float32, non_blocking=True
                ).contiguous()
                staged_geometry = self._geometry_for(device, global_slice, geometry)
            staged.append((device, staged_input, staged_geometry))

        # Keep all outputs alive until every device has launched its shard.
        outputs = []
        for device, staged_input, geometry in staged:
            with torch.cuda.device(device), cuda.gpus[device.index]:
                output = (
                    self._project_raw(staged_input, geometry)
                    if is_project
                    else self._backproject_raw(staged_input, geometry)
                )
            outputs.append((device, output))

        with torch.cuda.device(input_device):
            if is_project:
                pieces = [
                    output.to(device=input_device, non_blocking=True)
                    for _, output in outputs
                ]
                if pieces:
                    return torch.cat(pieces, dim=0)
                return torch.empty(
                    self.projection_shape, dtype=torch.float32, device=input_device
                )

            result = torch.zeros(
                self.volume_shape, dtype=torch.float32, device=input_device
            )
            for _, output in outputs:
                result.add_(output.to(device=input_device, non_blocking=True))
            if self._distributed and self.world_size > 1:
                torch.distributed.all_reduce(
                    result,
                    op=torch.distributed.ReduceOp.SUM,
                    group=self._process_group,
                )
            return result

    def _view_batches(self, plan):
        local_views = self.projection_shape[0]
        for device_rank, device in enumerate(plan.devices):
            shard = _balanced_slice(local_views, device_rank, len(plan.devices))
            for start in range(shard.start, shard.stop, plan.view_chunk_size):
                stop = min(start + plan.view_chunk_size, shard.stop)
                yield device, slice(start, stop), slice(
                    self.view_slice.start + start, self.view_slice.start + stop
                )

    def _tile_geometry(self, geometry, global_slice, centre, device):
        components = geometry if geometry is not None else self._trajectory
        translated = []
        for index, component in enumerate(components):
            value = component.detach()[global_slice].to(
                device="cpu", dtype=torch.promote_types(torch.float32, component.dtype)
            )
            position = index == 1 or (index == 0 and self.beam != "parallel")
            if position:
                value = value - value.new_tensor(centre)
            translated.append(value.to(device=device, dtype=torch.float32).contiguous())
        return tuple(translated)

    def _run_streamed(self, tensor, is_project, reduce_cotangent, geometry, plan):
        if reduce_cotangent:
            tensor = _all_reduce_copy(self, tensor, plan)
        shape = self.projection_shape if is_project else self.volume_shape
        result = torch.zeros(shape, dtype=torch.float32, device=tensor.device)
        for spatial_slice, tile_shape, centre in _tiles(
            self.volume_shape, plan.chunk_shape, self.voxel_spacing
        ):
            for device, local_slice, global_slice in self._view_batches(plan):
                source = tensor[spatial_slice] if is_project else tensor[local_slice]
                with torch.cuda.device(device), cuda.gpus[device.index]:
                    staged = source.detach().to(device=device, dtype=torch.float32).contiguous()
                    tile_geometry = self._tile_geometry(geometry, global_slice, centre, device)
                    output = (
                        self._project_raw(staged, tile_geometry) if is_project
                        else self._backproject_raw(staged, tile_geometry, tile_shape)
                    )
                    destination = local_slice if is_project else spatial_slice
                    result[destination].add_(output.to(device=tensor.device))
                    del staged, tile_geometry, output
        if not is_project and self._distributed and self.world_size > 1:
            self._reduce_streamed_(result, plan)
        return result

    def _reduce_streamed_(self, tensor, plan):
        """SUM a host result with bounded NCCL staging, also on empty view ranks."""
        if torch.distributed.get_backend(self._process_group) != "nccl":
            torch.distributed.all_reduce(
                tensor, op=torch.distributed.ReduceOp.SUM, group=self._process_group
            )
            return
        if tuple(tensor.shape) == self.volume_shape:
            slices = (item[0] for item in _tiles(self.volume_shape, plan.chunk_shape, self.voxel_spacing))
        else:
            slices = (slice(start, min(start + plan.view_chunk_size, tensor.shape[0]))
                      for start in range(0, tensor.shape[0], plan.view_chunk_size))
        device = plan.devices[0]
        with torch.cuda.device(device):
            for selected in slices:
                staged = tensor[selected].to(device=device, dtype=torch.float32).contiguous().clone()
                torch.distributed.all_reduce(
                    staged, op=torch.distributed.ReduceOp.SUM, group=self._process_group
                )
                tensor[selected].copy_(staged.to(device=tensor.device, dtype=tensor.dtype))
                del staged

    def _geometry_grad(self, volume, cotangent, geometry, plan=None):
        """Return d<cotangent, A(geometry) volume>/d(geometry) over this rank's views.

        ``cotangent`` holds the rank's view shard. The result covers the full
        trajectory; distributed mode sums it over ranks.
        """
        if plan is not None:
            return self._geometry_grad_streamed(volume, cotangent, geometry, plan)
        devices = self._devices or (volume.device,)
        local_views = self.projection_shape[0]
        spacing = self.detector_spacing if self.beam == "cone" else self.detector_spacing[0]
        grads = [
            torch.zeros(component.shape, dtype=torch.float32, device=volume.device)
            for component in geometry
        ]
        for device_rank, device in enumerate(devices):
            local_slice = _balanced_slice(local_views, device_rank, len(devices))
            if local_slice.start == local_slice.stop:
                continue
            global_slice = slice(
                self.view_slice.start + local_slice.start,
                self.view_slice.start + local_slice.stop,
            )
            with torch.cuda.device(device), cuda.gpus[device.index]:
                local_geometry = tuple(
                    component.detach()[global_slice].to(device=device, dtype=torch.float32)
                    for component in geometry
                )
                parts = _geometry_vjp(
                    self.beam,
                    volume.detach().to(device=device),
                    cotangent.detach()[local_slice].to(device=device),
                    local_geometry,
                    spacing,
                    self.voxel_spacing,
                )
            for grad, part in zip(grads, parts):
                grad[global_slice] += part.to(device=volume.device)
        if self._distributed and self.world_size > 1:
            flat = torch.cat([grad.flatten() for grad in grads])
            with torch.cuda.device(volume.device):
                torch.distributed.all_reduce(
                    flat, op=torch.distributed.ReduceOp.SUM, group=self._process_group
                )
            grads = list(flat.split([grad.numel() for grad in grads]))
            grads = [part.view(component.shape) for part, component in zip(grads, geometry)]
        return tuple(
            grad.to(dtype=component.dtype, device=component.device)
            for grad, component in zip(grads, geometry)
        )

    def _geometry_grad_streamed(self, volume, cotangent, geometry, plan):
        spacing = self.detector_spacing if self.beam == "cone" else self.detector_spacing[0]
        grads = [torch.zeros(component.shape, dtype=torch.float32, device="cpu")
                 for component in geometry]
        for spatial_slice, _, centre in _tiles(self.volume_shape, plan.chunk_shape, self.voxel_spacing):
            for device, local_slice, global_slice in self._view_batches(plan):
                with torch.cuda.device(device), cuda.gpus[device.index]:
                    staged_volume = volume.detach()[spatial_slice].to(device=device, dtype=torch.float32).contiguous()
                    staged_cotangent = cotangent.detach()[local_slice].to(device=device, dtype=torch.float32).contiguous()
                    tile_geometry = self._tile_geometry(geometry, global_slice, centre, device)
                    parts = _geometry_vjp(
                        self.beam, staged_volume, staged_cotangent, tile_geometry,
                        spacing, self.voxel_spacing,
                    )
                    for grad, part in zip(grads, parts):
                        grad[global_slice].add_(part.to(device="cpu"))
                    del staged_volume, staged_cotangent, tile_geometry, parts, part
        if self._distributed and self.world_size > 1:
            for grad in grads:
                self._reduce_streamed_(grad, plan)
        return tuple(grad.to(device=component.device, dtype=component.dtype)
                     for grad, component in zip(grads, geometry))

    def _project_raw(self, volume, geometry):
        if self.detector_surface is not None:
            return _surface_project(
                self.beam, volume, geometry, self.detector_shape,
                self.detector_spacing, self.voxel_spacing,
            )
        if self.beam == "parallel":
            return ParallelProjectorFunction.apply(
                volume, *geometry, self.detector_shape[0],
                self.detector_spacing[0], self.voxel_spacing,
            )
        if self.beam == "fan":
            return FanProjectorFunction.apply(
                volume, *geometry, self.detector_shape[0],
                self.detector_spacing[0], self.voxel_spacing,
            )
        return ConeProjectorFunction.apply(
            volume, *geometry, *self.detector_shape,
            *self.detector_spacing, self.voxel_spacing,
        )

    def _backproject_raw(self, sinogram, geometry, volume_shape=None):
        volume_shape = self.volume_shape if volume_shape is None else volume_shape
        if self.detector_surface is not None:
            return _surface_backproject(
                self.beam, sinogram, geometry, volume_shape,
                self.detector_spacing, self.voxel_spacing,
            )
        if self.beam == "parallel":
            return ParallelBackprojectorFunction.apply(
                sinogram, *geometry, self.detector_spacing[0],
                *volume_shape, self.voxel_spacing,
            )
        if self.beam == "fan":
            return FanBackprojectorFunction.apply(
                sinogram, *geometry, self.detector_spacing[0],
                *volume_shape, self.voxel_spacing,
            )
        return ConeBackprojectorFunction.apply(
            sinogram, *geometry, *volume_shape,
            *self.detector_spacing, self.voxel_spacing,
        )

    @property
    def process_group(self):
        """Process group of a distributed operator, or None for the default group.

        Pass it as ``group=`` to collectives that combine per-rank results of this
        operator, so they use the same ranks as the operator.
        """
        return self._process_group

    def project(self, volume):
        """Project a floating ``(H, W)`` or ``(D, H, W)`` CPU/CUDA tensor.

        The output shape is ``projection_shape``; in distributed mode it holds
        this rank's contiguous view slice. The float32 result stays on
        ``volume.device``. CPU data streams through CUDA automatically; manual
        chunk limits override automatic sizing. Execution requires CUDA.
        """
        plan = self._execution_plan(volume, True)
        geometry = self._effective_geometry(plan)
        return _ProjectorAutograd.apply(volume, self, "project", plan, *geometry)

    def backproject(self, sinogram):
        """Backproject a floating-point CPU/CUDA sinogram with the matched adjoint.

        The input must have shape ``projection_shape`` and uses ``(views, U,
        V)`` order for cone beams. The float32 volume is returned to
        ``sinogram.device``. CPU data streams through CUDA automatically;
        manual chunk limits override automatic sizing. Execution requires CUDA.
        Distributed mode SUM-reduces it to every rank, so
        divide a replicated-output loss by ``world_size``.
        """
        plan = self._execution_plan(sinogram, False)
        geometry = self._effective_geometry(plan)
        return _ProjectorAutograd.apply(sinogram, self, "backproject", plan, *geometry)

    def __call__(self, volume):
        """Alias for :meth:`project`."""
        return self.project(volume)

    def _effective_geometry(self, plan=None):
        if self.detector_surface is None:
            return tuple(component.to(device="cpu" if plan is not None else component.device).clone()
                         for component in self._learnable)
        # Callback graphs can save mutable captured parameters. Snapshot their
        # values on the host, restoring their original device only when the
        # callback's own backward needs them. Native tile graphs are not saved.
        def pack(tensor):
            return tensor.detach().to(device="cpu").clone(), tensor.device

        def unpack(saved):
            tensor, device = saved
            return tensor.to(device=device)

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            geometry = self._sample_surface(cpu_only=plan is not None)[0]
            return tuple(component.clone() for component in geometry)

    def _sample_surface(self, cpu_only=False, validate_on_cpu=False):
        """Return effective per-view source/direction and sampled pixel points."""
        if self.beam == "cone":
            n_u, n_v = self.detector_shape
            du, dv = self.detector_spacing
            u_axis = (torch.arange(n_u, dtype=torch.float64, device="cpu") + 0.5 - n_u / 2) * du
            v_axis = (torch.arange(n_v, dtype=torch.float64, device="cpu") + 0.5 - n_v / 2) * dv
            u, v = torch.meshgrid(u_axis, v_axis, indexing="ij")
        else:
            (n_u,) = self.detector_shape
            (du,) = self.detector_spacing
            u = (torch.arange(n_u, dtype=torch.float64, device="cpu") + 0.5 - n_u / 2) * du
            v = torch.zeros_like(u)

        offsets = self.detector_surface(u, v)
        if not isinstance(offsets, torch.Tensor):
            raise TypeError("detector_surface must return a floating-point tensor")
        if not torch.is_floating_point(offsets):
            raise TypeError("detector_surface must return a floating-point tensor")
        if cpu_only and offsets.device.type != "cpu":
            raise TypeError("streamed detector_surface offsets must be on CPU")
        if validate_on_cpu:
            offsets = offsets.to(device="cpu")
        detector_rank = len(self.detector_shape)
        shared_shape = (*self.detector_shape, 3)
        per_view_shape = (self._n_views, *shared_shape)
        if tuple(offsets.shape) not in (shared_shape, per_view_shape):
            raise ValueError(
                f"detector_surface output must have shape {shared_shape} or {per_view_shape}"
            )
        if not torch.isfinite(offsets).all().item():
            raise ValueError("detector_surface offsets must be finite")
        if self.beam != "cone" and torch.any(offsets[..., 1] != 0).item():
            raise ValueError("2D detector_surface middle offsets must be zero")

        per_view = offsets.ndim == detector_rank + 2
        local = offsets if per_view else offsets.unsqueeze(0)
        trajectory = self._learnable if self._learnable else self._trajectory
        if self.beam == "parallel":
            direction, center, det_u = trajectory
        else:
            source, center = trajectory[:2]
            det_u = trajectory[2]
            det_v = trajectory[3] if self.beam == "cone" else None

        def frame(component):
            return component.to(
                device=offsets.device,
                dtype=torch.promote_types(torch.float32, component.dtype),
            ).reshape(
                (component.shape[0],) + (1,) * detector_rank + (component.shape[1],)
            )

        center = frame(center)
        det_u = frame(det_u)
        local_u = local[..., 0].unsqueeze(-1)
        local_n = local[..., 2].unsqueeze(-1)
        points = center + local_u * det_u
        if self.beam == "cone":
            det_v = frame(det_v)
            normal_dtype = torch.promote_types(det_u.dtype, det_v.dtype)
            normal = torch.linalg.cross(
                det_u.to(dtype=normal_dtype),
                det_v.to(dtype=normal_dtype),
                dim=-1,
            )
            points = points + local[..., 1].unsqueeze(-1) * det_v + local_n * normal
        else:
            normal = torch.stack((det_u[..., 1], -det_u[..., 0]), dim=-1)
            points = points + local_n * normal

        if not torch.isfinite(points).all().item() or not torch.isfinite(
            points.detach().to(dtype=torch.float32)
        ).all().item():
            raise ValueError("detector_surface world points must be finite in float32")

        if self.beam == "parallel":
            direction = direction.to(device=points.device)
            if not torch.isfinite(direction).all().item() or not torch.isfinite(
                direction.detach().to(dtype=torch.float32)
            ).all().item():
                raise ValueError("trajectory ray directions must be finite in float32")
            effective = direction, points
        else:
            source = frame(source).reshape(source.shape[0], -1, source.shape[1])[:, 0]
            effective = source, points
            if not torch.isfinite(source).all().item() or not torch.isfinite(
                source.detach().to(dtype=torch.float32)
            ).all().item():
                raise ValueError("detector surface sources must be finite in float32")
            source_points = source.to(dtype=torch.float64).unsqueeze(1)
            endpoint_points = points.reshape(points.shape[0], -1, points.shape[-1]).double()
            source_distance = torch.linalg.vector_norm(source_points, dim=-1)
            endpoint_distance = torch.linalg.vector_norm(endpoint_points, dim=-1)
            distances = torch.cat((source_distance, endpoint_distance), dim=1) / self.voxel_spacing
            nearest = torch.minimum(source_distance, endpoint_distance) / self.voxel_spacing
            staged_coincident = torch.all(
                endpoint_points.to(dtype=torch.float32)
                == source_points.to(dtype=torch.float32), dim=-1
            )
            if torch.any(staged_coincident).item():
                raise ValueError("source and detector surface pixels must differ")
            if torch.any(nearest > 1e6).item():
                raise ValueError(
                    "the source or a detector pixel must lie within 1e6 voxels "
                    "of the volume center in every view"
                )
            if torch.any(distances > 1e15).item():
                raise ValueError(
                    "the source and detector surface pixels must lie within 1e15 voxels "
                    "of the volume center"
                )
        return effective, offsets
