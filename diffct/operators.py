"""High-level operators for arbitrary CT trajectories."""

import math
import numbers
from dataclasses import replace
import time

import torch
from numba import cuda

from .chunking import (_ChunkPlan, _host_geometry_bytes, _largest_buffer_bytes, _memory_budget,
                       _pipeline_bytes, _pixel_tiles, _tiles, _working_set_bytes)
from . import pipeline as _pipeline
from .storage import TensorStore
from .streaming import _validate_stores
from .surfaces import ParameterizedSurface, _GeometrySpec, _sample_geometry
from .projectors import (
    _backproject_into,
    _geometry_vjp,
    _prepare_volume,
    _project_into,
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


class _ReducedInput(TensorStore):
    """A requested SUM, performed once per canonical resident tile, never eagerly."""
    reduced = True


def _all_reduce_copy(projector, tensor, plan=None):
    """Return a SUM over ranks of ``tensor`` (the tensor itself without ranks)."""
    if not (projector._distributed and projector.world_size > 1):
        return tensor
    if plan is not None:
        return _ReducedInput(tensor)
    reduced = tensor.detach().clone(memory_format=torch.contiguous_format)
    with torch.cuda.device(reduced.device):
        torch.distributed.all_reduce(
            reduced, op=torch.distributed.ReduceOp.SUM, group=projector._process_group
        )
    return reduced


class _ProjectorAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, projector, mode, plan, geometry_spec, participation, *geometry):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("input must be a torch.Tensor")
        ctx.projector = projector
        ctx.mode = mode
        ctx.plan = plan
        ctx.geometry_spec = geometry_spec
        ctx.input_device = tensor.device
        ctx.input_dtype = tensor.dtype
        needs_geometry = any(component.requires_grad for component in geometry) or (
            projector._distributed and projector.world_size > 1
            and projector._geometry_trainable
        )
        ctx.save_for_backward(tensor if needs_geometry else None, *geometry)
        return projector._run(
            tensor,
            is_project=mode in ("project", "project_reduced"),
            reduce_cotangent=mode in ("project_reduced", "backproject_reduced"),
            geometry=geometry if geometry else None,
            plan=plan,
            geometry_spec=geometry_spec,
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
            adjoint = ({"project": "backproject_reduced", "backproject": "project",
                        "backproject_reduced": "project"}[ctx.mode]
                       if projector.partition == "space" else _ADJOINT_MODE[ctx.mode])
            grad_input = _ProjectorAutograd.apply(
                grad_output, projector, adjoint, ctx.plan, ctx.geometry_spec, projector._participation(), *geometry
            ).to(device=ctx.input_device, dtype=ctx.input_dtype)
            if not ctx.needs_input_grad[0]:
                grad_input = None
        geometry_grads = (None,) * len(geometry)
        if any(ctx.needs_input_grad[6:]) or (
            collective and projector._geometry_trainable
        ):
            # Every mode is <cotangent, A_local(geometry) volume> for some pair.
            if projector.partition == "space":
                if ctx.mode == "project":
                    volume, cotangent = tensor, _all_reduce_copy(projector, grad_output, ctx.plan)
                else:
                    volume, cotangent = grad_output, tensor
                    if ctx.mode == "backproject_reduced":
                        cotangent = _all_reduce_copy(projector, cotangent, ctx.plan)
            elif ctx.mode == "project":
                volume, cotangent = tensor, grad_output
            elif ctx.mode == "project_reduced":
                volume, cotangent = _all_reduce_copy(projector, tensor, ctx.plan), grad_output
            else:
                volume, cotangent = _all_reduce_copy(projector, grad_output, ctx.plan), tensor
            grads = _GeometryGradAutograd.apply(projector, volume, cotangent, ctx.plan, ctx.geometry_spec, *geometry)
            geometry_grads = tuple(
                grad if needed else None
                for grad, needed in zip(grads, ctx.needs_input_grad[6:])
            )
        return (grad_input, None, None, None, None, None, *geometry_grads)


class _GeometryGradAutograd(torch.autograd.Function):
    """First-order geometry gradient; differentiating it again raises."""

    @staticmethod
    def forward(ctx, projector, volume, cotangent, plan, geometry_spec, *geometry):
        return projector._geometry_grad(volume, cotangent, geometry, plan, geometry_spec)

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

    ``ParameterizedSurface(sampler, parameters)`` instead samples only the
    selected global view IDs and detector rectangle. Its explicit parameters
    and trajectory values are snapshotted once per call, with batch-local
    first-order parameter gradients. ``detector_chunk_shape`` bounds pixel
    rectangles; their physical origins remain those of the full detector.

    Volumes use ``(H, W)`` for parallel and fan beams and ``(D, H, W)`` for
    cone beams. Sinograms use ``(views, detectors)`` or ``(views, U, V)``.
    Results are float32 and returned to the input tensor's device. CPU inputs
    stream spatial tiles and view batches through CUDA. CUDA inputs retain the
    full-volume path when its additional working set fits every device;
    otherwise they stream too. ``volume_chunk_shape`` sets explicit spatial
    limits in tensor order; ``view_chunk_size`` sets an explicit view limit.
    Either forces streaming. With no overrides, sizing uses live free memory,
    reusable allocator cache and the observable per-process fraction ceiling,
    retaining 25% headroom. Automatic sizing compares fitting first-axis slabs
    and Cartesian blocks; view batches are at most 32 and shrink if needed. Older
    PyTorch without a public fraction getter cannot account for that ceiling.
    Concurrent external allocations can still cause an allocation error.

    ``schedule="auto"`` selects spatial, view or bounded two-tile window order
    using estimated full-workload time from short native pilots, with estimated
    copy bytes breaking ties. The same orders can be requested with
    ``"spatial"``, ``"views"`` or ``"window"``;
    capacity may require a serial spatial fallback. Streaming uses distinct
    upload, compute and download streams and two reusable host slots capped
    at 8 MiB each. Store reads and completed output drains are fragmented
    independently of the resident GPU tile size. ``last_execution_stats``
    reports actual copy payloads, native launches/layouts, planned capacity
    and pilot metadata; it is None before execution, after errors, and on the
    uninstrumented full-CUDA path. Reports and selection caches retain metadata.

    Full host volumes/sinograms stay on CPU; streamed native CUDA buffers are
    bounded by the chosen tile and view batch. Caller-owned CUDA data/results
    and CUDA work inside a supplied callback are outside this memory bound.
    Local devices launch their view batches concurrently, but each device
    receives every owned volume tile, so host transfers limit the speedup.
    Local devices split views in list order. ``partition="views"`` returns
    rank-local projections and replicated backprojections. ``partition="space"``
    assigns balanced first-axis slabs and returns replicated projections and
    owned backprojections. ``global_volume_shape`` is the constructor shape;
    ``volume_slice`` identifies the owned global slab, and ``volume_shape``
    and ``local_volume_shape`` give its required tensor shape. Empty slabs are
    valid. Physical coordinates retain the global volume frame.
    For replicated outputs, count the loss once globally or divide each
    rank's loss by ``world_size``. Geometry gradients already SUM over ranks;
    do not apply an additional DDP reduction to them. Space ranks participate
    in backward even when their local input and geometry require no gradients.
    """

    def __init__(self, trajectory, volume_shape, detector_shape, *, beam="cone",
                 detector_spacing=1.0, voxel_spacing=1.0, devices=None,
                 distributed=False, process_group=None, detector_surface=None,
                 volume_chunk_shape=None, view_chunk_size=None, detector_chunk_shape=None,
                 schedule="auto", partition="views"):
        self.partition = partition
        if not isinstance(partition, str) or partition not in ("views", "space"):
            raise ValueError("partition must be 'views' or 'space'")
        self.schedule = schedule
        self.last_execution_stats = None
        if not isinstance(schedule, str) or schedule not in ("auto", "spatial", "views", "window"):
            raise ValueError("schedule must be 'auto', 'spatial', 'views', or 'window'")
        self._schedule_cache = {}
        self._snapshot_d2h_bytes = 0
        if beam not in ("parallel", "fan", "cone"):
            raise ValueError("beam must be 'parallel', 'fan', or 'cone'")
        self.beam = beam
        rank = 3 if beam == "cone" else 2
        self.volume_shape = _shape(volume_shape, rank, "volume_shape")
        self.global_volume_shape = self.volume_shape
        self.local_volume_shape = self.volume_shape
        self.volume_slice = tuple(slice(0, size) for size in self.volume_shape)
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
        self.detector_chunk_shape = (
            None if detector_chunk_shape is None
            else _shape(detector_chunk_shape, len(self.detector_shape), "detector_chunk_shape")
        )

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
        if isinstance(detector_surface, ParameterizedSurface):
            frames = tuple(component.detach().to(device="cpu") for component in self._trajectory)
            parameters = tuple(parameter.detach().to(device="cpu") for parameter in detector_surface.parameters)
            spec = _GeometrySpec("sampled", detector_surface.sampler)
            _sample_geometry(
                self.beam, self.detector_shape, self.detector_spacing, (*frames, *parameters), spec,
                slice(0, 1), tuple(slice(0, 1) for _ in self.detector_shape), self.voxel_spacing,
            )
            surface_trainable = any(parameter.requires_grad for parameter in detector_surface.parameters)
        elif detector_surface is not None:
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
        self._agree_geometry_layout()
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
            if self.partition == "views" and bounds[0].item() != -bounds[1].item():
                raise ValueError(
                    "all ranks must agree on whether the geometry requires gradients"
                )
            self._geometry_trainable = bool(bounds[0].item())
            if self.partition == "space" and self._geometry_trainable:
                self._learnable = tuple(trajectory)
        if self.partition == "space":
            first = _balanced_slice(self.global_volume_shape[0], self.rank, self.world_size)
            self.volume_slice = (first, *(slice(0, size) for size in self.global_volume_shape[1:]))
            self.volume_shape = (first.stop - first.start, *self.global_volume_shape[1:])
        self.local_volume_shape = self.volume_shape
        self._planning_volume_shape = ((math.ceil(self.global_volume_shape[0] / self.world_size),
                                       *self.global_volume_shape[1:]) if self.partition == "space"
                                      else self.global_volume_shape)
        self.view_slice = (slice(0, n_views) if self.partition == "space"
                           else _balanced_slice(n_views, self.rank, self.world_size))
        self.projection_shape = (
            self.view_slice.stop - self.view_slice.start,
            *self.detector_shape,
        )

    def _agree_geometry_layout(self):
        if not (self._distributed and self.world_size > 1):
            return
        surface = self.detector_surface
        if isinstance(surface, ParameterizedSurface):
            kind = "sampled"
            parameters = tuple((tuple(value.shape), str(value.dtype)) for value in surface.parameters)
        else:
            kind, parameters = ("points" if surface is not None else "flat"), ()
        metadata = (self.partition, self.schedule, self.global_volume_shape, self.beam, self.voxel_spacing,
                    self.detector_spacing, self._n_views, self.detector_shape, kind, parameters)
        layouts = [None] * self.world_size
        torch.distributed.all_gather_object(layouts, metadata, group=self._process_group)
        if any(layout != metadata for layout in layouts):
            raise ValueError("distributed ranks must agree on geometry and surface parameter count, shapes and dtypes")

    def _participation(self):
        if self.partition == "space" and self._distributed and self.world_size > 1 and torch.is_grad_enabled():
            return torch.zeros((), dtype=torch.float32, device="cpu", requires_grad=True)
        return None

    def _volume_tiles(self, chunk_shape):
        offset = tuple((owned.start + local / 2 - global_size / 2) * self.voxel_spacing
                       for owned, local, global_size in zip(self.volume_slice, self.volume_shape, self.global_volume_shape))[::-1]
        for spatial, shape, centre in _tiles(self.volume_shape, chunk_shape, self.voxel_spacing):
            yield spatial, shape, tuple(value + shift for value, shift in zip(centre, offset))

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

    def _execution_plan(self, tensor, is_project, output_store=None):
        self._validate_input(tensor, is_project)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to execute the projector")
        input_device = torch.device(getattr(tensor, "device", "cpu"))
        is_cuda = input_device.type == "cuda"
        output_device = torch.device(getattr(output_store, "device", input_device if is_cuda else "cpu"))
        devices = self._devices or (
            input_device if is_cuda else torch.device("cuda", torch.cuda.current_device()),
        )
        local_views = self.projection_shape[0]
        active_devices = devices[:min(len(devices), local_views)]
        foreign_output = output_device.type == "cuda" and any(device != output_device for device in active_devices or devices[:1])
        collective_devices = ()
        if self._distributed and self.world_size > 1:
            collective_devices = (devices[0],)
            if torch.distributed.get_backend(self._process_group) == "nccl":
                collective_devices += (torch.device("cuda", torch.cuda.current_device()),)
        inventory = tuple(dict.fromkeys((
            *active_devices, *((input_device,) if is_cuda else ()),
            *((output_device,) if output_device.type == "cuda" else ()), *collective_devices,
        )))
        observed = {device: _memory_budget(device) for device in inventory}
        budgets = {device: limits[0] for device, limits in observed.items()}
        allocation_limit = min((limits[1] for limits in observed.values()), default=(1 << 63) - 1)
        parameterized = isinstance(self.detector_surface, ParameterizedSurface)
        surface = self.detector_surface is not None or self.detector_chunk_shape is not None
        # Sampling precedes view sharding. Conservatively allow the complete
        # float64 world-point expression/snapshot on each participating device,
        # without calling the callback again just to discover its device.
        world_bytes = (4 * 8 * self._n_views * math.prod(self.detector_shape) * len(self.volume_shape)
                       if self.detector_surface is not None and not parameterized else 0)
        volume_elements = math.prod(self.volume_shape)
        projection_elements = math.prod(self.projection_shape)
        can_use_full = (isinstance(tensor, torch.Tensor) and is_cuda and output_store is None
                        and self.volume_chunk_shape is None and self.view_chunk_size is None
                        and self.detector_chunk_shape is None and not parameterized and self.schedule == "auto"
                        and self.partition == "views")
        for device_rank, device in enumerate(devices):
            shard = _balanced_slice(local_views, device_rank, len(devices))
            views = shard.stop - shard.start
            if views == 0:
                continue
            needed = _working_set_bytes(self.beam, self.volume_shape, views, self.detector_shape, surface)
            needed += world_bytes
            if is_cuda and device == input_device:
                needed += 8 * (volume_elements + projection_elements)
            can_use_full = can_use_full and needed <= budgets[device]
            can_use_full = can_use_full and _largest_buffer_bytes(
                self.beam, self.volume_shape, views, self.detector_shape, surface
            ) <= allocation_limit
        if is_cuda and input_device not in active_devices:
            can_use_full = can_use_full and 8 * (volume_elements + projection_elements) + world_bytes <= budgets[input_device]
        can_use_full = can_use_full and all(world_bytes <= amount for amount in budgets.values())
        if is_cuda:
            can_use_full = can_use_full and 4 * max(volume_elements, projection_elements) <= allocation_limit

        # CUDA results stay on the caller's device. CPU execution never reserves
        # a full volume/sinogram there. Include transfer scratch on an origin
        # device even when it is not one of the selected compute devices.
        streamed_budgets = dict(budgets)
        if is_cuda and output_store is None:
            streamed_budgets[input_device] -= 4 * max(volume_elements, projection_elements)
        budget = min(streamed_budgets.values(), default=(1 << 63) - 1)
        if is_cuda and output_store is None and 4 * max(volume_elements, projection_elements) > allocation_limit:
            budget = -1
        if self._distributed and self.world_size > 1:
            control_device = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.distributed.get_backend(self._process_group) == "nccl" else torch.device("cpu")
            )
            settings = torch.tensor(
                (*self.global_volume_shape, *(self.volume_chunk_shape or (0,) * len(self.volume_shape)),
                 self.view_chunk_size or 0, *(self.detector_chunk_shape or (0,) * len(self.detector_shape))),
                dtype=torch.int64, device=control_device,
            )
            lower, upper = settings.clone(), settings.clone()
            torch.distributed.all_reduce(lower, op=torch.distributed.ReduceOp.MIN, group=self._process_group)
            torch.distributed.all_reduce(upper, op=torch.distributed.ReduceOp.MAX, group=self._process_group)
            if not torch.equal(lower, upper):
                raise ValueError("distributed ranks must agree on volume shape and explicit chunk limits")
            agreement = torch.tensor([int(can_use_full), budget, allocation_limit, int(not foreign_output)], dtype=torch.int64, device=control_device)
            torch.distributed.all_reduce(agreement, op=torch.distributed.ReduceOp.MIN, group=self._process_group)
            can_use_full, budget, allocation_limit, same_output = agreement.cpu().tolist()
            foreign_output = not same_output
        if can_use_full:
            return None

        chunk = tuple(min(limit, size) for limit, size in zip(
            self.volume_chunk_shape or self._planning_volume_shape, self._planning_volume_shape
        ))
        batch = self.view_chunk_size or min(32, self._n_views)
        pixels = tuple(min(limit, size) for limit, size in zip(
            self.detector_chunk_shape or self.detector_shape, self.detector_shape,
        ))
        minimum_shape = (1,) * len(chunk) if self.volume_chunk_shape is None else chunk
        peers = len(active_devices) > 1 or (self._distributed and self.world_size > 1)
        extra_rays = (2 * int(self.partition == "space") + int(is_project and foreign_output)
                      + int(is_project and self.partition == "space" and peers))

        def fits(shape):
            points = surface or pixels != self.detector_shape
            return (
                _pipeline_bytes(self.beam, shape, batch, pixels, points,
                                geometry_grad=self._geometry_trainable,
                                ray_accumulators=extra_rays, peers=peers or foreign_output) <= budget
                and _largest_buffer_bytes(self.beam, shape, batch, pixels, points) <= allocation_limit
                and _host_geometry_bytes(self.beam, batch, pixels, points) <= _pipeline._GEOMETRY_WORK_BYTES
            )

        while not fits(minimum_shape):
            if self.view_chunk_size is None and batch > 1:
                batch = max(1, batch // 2)
            elif self.detector_chunk_shape is None and max(pixels) > 1:
                axis = max(range(len(pixels)), key=pixels.__getitem__)
                smaller = list(pixels)
                smaller[axis] = max(1, smaller[axis] // 2)
                pixels = tuple(smaller)
            else:
                raise RuntimeError("CUDA or host geometry memory budget cannot fit the minimum tile/view/pixel batch; reduce explicit limits")
        while not fits(chunk):
            if self.volume_chunk_shape is not None or max(chunk) == 1:
                raise RuntimeError("CUDA memory budget cannot fit the requested tile/view batch")
            axis = max(range(len(chunk)), key=chunk.__getitem__)
            smaller = list(chunk)
            smaller[axis] = max(1, smaller[axis] // 2)
            chunk = tuple(smaller)
        estimate = _pipeline_bytes(self.beam, chunk, batch, pixels,
                                   surface or pixels != self.detector_shape,
                                   geometry_grad=self._geometry_trainable,
                                   ray_accumulators=extra_rays, peers=peers or foreign_output)
        return _ChunkPlan(chunk, batch, devices, pixels, budget, allocation_limit,
                          estimated_gpu_bytes=estimate, foreign_output=foreign_output)

    def _validate_input(self, tensor, is_project):
        if not isinstance(tensor, torch.Tensor):
            if not all(hasattr(tensor, name) for name in ("shape", "dtype", "read", "write", "flush")):
                raise TypeError("input must be a torch.Tensor or block store")
            expected_shape = self.volume_shape if is_project else self.projection_shape
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(f"input shape must be {expected_shape}")
            if tensor.dtype not in (torch.float32, torch.float64):
                raise TypeError("input tensor/store must have floating-point dtype")
            return
        expected_shape = self.volume_shape if is_project else self.projection_shape
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"input shape must be {expected_shape}")
        if not torch.is_floating_point(tensor):
            raise TypeError("input tensor must have floating-point dtype")
        if tensor.device.type not in ("cpu", "cuda"):
            raise TypeError("input tensor must be on CPU or CUDA")

    def _run(self, tensor, is_project, reduce_cotangent=False, geometry=None, plan=None, geometry_spec=None):
        self._validate_input(tensor, is_project)
        if plan is not None:
            return self._run_streamed(tensor, is_project, reduce_cotangent, geometry, plan, geometry_spec)
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
                if len(pieces) == 1:
                    return pieces[0]
                if pieces:
                    return torch.cat(pieces, dim=0)
                return torch.empty(
                    self.projection_shape, dtype=torch.float32, device=input_device
                )

            if len(outputs) == 1:
                result = outputs[0][1].to(device=input_device, non_blocking=True)
            else:
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

    def _view_rounds(self, plan):
        """Yield rounds of view batches with at most one batch per device.

        A round launches on every device before any result is collected, so
        the devices compute concurrently while each holds one batch at a time.
        """
        local_views = self.projection_shape[0]
        shards = [
            _balanced_slice(local_views, device_rank, len(plan.devices))
            for device_rank in range(len(plan.devices))
        ]
        longest = max(shard.stop - shard.start for shard in shards)
        for offset in range(0, longest, plan.view_chunk_size):
            batches = []
            for device, shard in zip(plan.devices, shards):
                start = shard.start + offset
                if start >= shard.stop:
                    continue
                stop = min(start + plan.view_chunk_size, shard.stop)
                batches.append((device, slice(start, stop), slice(
                    self.view_slice.start + start, self.view_slice.start + stop
                )))
            yield batches

    def _tile_geometry(self, geometry, global_slice, centre, device, pixels=None, geometry_spec=None):
        components = geometry if geometry is not None else self._trajectory
        points = (geometry_spec is not None and geometry_spec.kind == "sampled") or (
            pixels is not None and (self.detector_chunk_shape is not None or
                                    tuple(item.stop - item.start for item in pixels) != self.detector_shape)
        )
        if geometry_spec is not None and geometry_spec.kind == "points":
            components = (components[0][global_slice], components[1][(global_slice, *pixels)])
        elif points:
            components = _sample_geometry(
                self.beam, self.detector_shape, self.detector_spacing, components,
                geometry_spec, global_slice, pixels, self.voxel_spacing,
            )
        else:
            components = tuple(component[global_slice] for component in components)
        translated = []
        for index, component in enumerate(components):
            value = component.detach().to(
                device="cpu", dtype=torch.promote_types(torch.float32, component.dtype)
            )
            position = index == 1 or (index == 0 and self.beam != "parallel")
            if position:
                value = value - value.new_tensor(centre)
            translated.append(value if device is None else value.to(device=device, dtype=torch.float32).contiguous())
        return tuple(translated)

    def _run_streamed(self, tensor, is_project, reduce_cotangent, geometry, plan, geometry_spec=None):
        output_device = tensor.device
        shape = self.projection_shape if is_project else self.volume_shape
        result = torch.empty(shape, dtype=torch.float32, device=output_device)
        source = (_ReducedInput(tensor) if reduce_cotangent and self._distributed and self.world_size > 1
                  else TensorStore(tensor))
        self._run_store_streamed(source, TensorStore(result), is_project,
                                 geometry, plan, geometry_spec)
        return result

    def _run_store_streamed(self, source, sink, is_project, geometry, plan, geometry_spec):
        self.last_execution_stats = None
        try:
            chosen, pilots, cached = self._choose_schedule(source, sink, is_project, geometry, plan, geometry_spec)
            stats = self._execution_stats(chosen, is_project)
            stats["d2h_bytes"] = self._snapshot_d2h_bytes
            self._snapshot_d2h_bytes = 0
            stats["pilot"], stats["selection_cached"] = pilots, cached
            if self.partition == "space" and is_project:
                self._space_project(source, sink, geometry, chosen, geometry_spec, stats)
            elif self.partition == "space" and getattr(source, "reduced", False):
                self._space_back_reduced(source, sink, geometry, chosen, geometry_spec, stats)
            else:
                self._execute_schedule(source, sink, is_project, geometry, chosen, geometry_spec, stats)
            self.last_execution_stats = stats
        except BaseException:
            self.last_execution_stats = None
            raise

    def _execution_stats(self, plan, is_project):
        return dict(operation="project" if is_project else "backproject", schedule=plan.schedule,
                    chunk_shape=tuple(plan.chunk_shape), view_chunk_size=plan.view_chunk_size,
                    detector_chunk_shape=tuple(plan.detector_chunk_shape or self.detector_shape),
                    window_size=plan.window_size, gpu_budget_bytes=plan.gpu_budget_bytes,
                    estimated_gpu_bytes=plan.estimated_gpu_bytes, pinned_bytes_peak=0,
                    h2d_bytes=0, d2h_bytes=0, p2p_bytes=0, kernel_launches=0,
                    layout_preparations=0, collective_calls=0, collective_bytes=0, pilot=[],
                    **({"fallback_reason": plan.fallback_reason} if plan.fallback_reason else {}))

    def _candidate_plans(self, plan, is_project, spec):
        pixels = plan.detector_chunk_shape or self.detector_shape
        volume_shape = self._planning_volume_shape
        points = spec is not None or self.detector_chunk_shape is not None or pixels != self.detector_shape
        peers = len(plan.devices) > 1 or (self._distributed and self.world_size > 1) or plan.foreign_output

        def size(shape, slots=1, retained=1, ray_accumulators=0):
            return _pipeline_bytes(self.beam, shape, plan.view_chunk_size, pixels, points,
                                   slots=slots, retained=retained, geometry_grad=self._geometry_trainable,
                                   peers=peers, ray_accumulators=ray_accumulators)

        shapes = [plan.chunk_shape]
        if self.volume_chunk_shape is None and len(volume_shape) == 3:
            slab = list(volume_shape)
            def fits(candidate):
                return (size(tuple(candidate)) <= plan.gpu_budget_bytes
                        and _largest_buffer_bytes(self.beam, candidate, plan.view_chunk_size, pixels, points)
                        <= plan.allocation_limit)
            while not fits(slab) and slab[0] > 1:
                slab[0] = max(1, slab[0] // 2)
            slab_fits = fits(slab)
            block = list(plan.chunk_shape)
            while sum(a < b for a, b in zip(block, volume_shape)) < 2 and max(block) > 1:
                axis = max(range(len(block)), key=block.__getitem__)
                block[axis] = max(1, block[axis] // 2)
            shapes = [tuple(slab), tuple(block)] if slab_fits else [plan.chunk_shape, tuple(block)]
        shapes = list(dict.fromkeys(shapes))[:2]
        candidates = []
        for shape in shapes:
            count = math.prod(math.ceil(size / part) for size, part in zip(volume_shape, shape))
            common_views = self._n_views if self.partition == "space" else math.ceil(self._n_views / self.world_size)
            batches = math.ceil(max(1, common_views) / plan.view_chunk_size)
            singleton = count == batches == 1 and pixels == self.detector_shape
            for schedule in (("views",) if self.partition == "space" and is_project else
                             ("spatial",) if singleton and self.schedule == "auto" else
                             ("spatial", "views", "window")):
                retained = count if schedule == "views" and not is_project else min(2, count) if schedule == "window" else 1
                slots = 2
                extra_rays = 2 if self.partition == "space" or (is_project and schedule == "views" and count > 1) else 0
                extra_rays += int(is_project and plan.foreign_output)
                extra_rays += int(is_project and self.partition == "space" and peers)
                estimate = size(shape, slots, retained, extra_rays)
                host_fits = 2 * _host_geometry_bytes(self.beam, plan.view_chunk_size, pixels, points) <= _pipeline._GEOMETRY_WORK_BYTES
                if estimate > plan.gpu_budget_bytes or not host_fits:
                    if schedule == "window":
                        continue
                    slots, estimate = 1, size(shape, 1, retained, extra_rays)
                if estimate > plan.gpu_budget_bytes:
                    continue
                candidates.append(replace(plan, chunk_shape=shape, schedule=schedule,
                                          window_size=retained, slots=slots, estimated_gpu_bytes=estimate,
                                          fallback_reason="two-slot pipeline exceeds live CUDA allocation budget" if slots == 1 else ""))
        if self.partition == "space" and is_project:
            if not candidates:
                raise RuntimeError("CUDA memory budget cannot fit a resident spatial-partition ray batch")
            return [replace(candidates[0], fallback_reason="space projection retains complete ray batches before SUM")]
        if self.schedule != "auto":
            selected = [candidate for candidate in candidates if candidate.schedule == self.schedule]
            if selected:
                return selected[:1]
            spatial = [candidate for candidate in candidates if candidate.schedule == "spatial"]
            if spatial:
                return [replace(spatial[0], fallback_reason="requested schedule accumulators/window exceed CUDA budget")]
            raise RuntimeError("CUDA memory budget cannot fit the requested schedule accumulators")
        return candidates

    def _choose_schedule(self, source, sink, is_project, geometry, plan, spec):
        candidates = self._candidate_plans(plan, is_project, spec)
        if not candidates:
            raise RuntimeError("CUDA memory budget cannot fit a streamed schedule")
        key = (is_project, plan.chunk_shape, plan.view_chunk_size, plan.detector_chunk_shape,
               tuple(device.index for device in plan.devices), str(source.dtype), plan.gpu_budget_bytes,
               plan.allocation_limit, str(getattr(source, "device", "cpu")), str(getattr(sink, "device", "cpu")),
               spec.kind if spec is not None else "flat", self.schedule)
        collective = self._distributed and self.world_size > 1
        if not collective and key in self._schedule_cache:
            return self._schedule_cache[key], [], True
        pilots = []
        index = 0
        if self.schedule == "auto" and len(candidates) > 1 and (not collective or self.rank == 0):
            for candidate in candidates:
                pilots.append(self._pilot_candidate(source, sink, is_project, geometry, candidate, spec))
            # Scale measured prefixes to the full workload; bytes break ties.
            def cost(item):
                pilot = pilots[item[0]]
                return (pilot["estimated_elapsed_ms"],
                        pilot["estimated_h2d_bytes"] + pilot["estimated_d2h_bytes"])
            index = min(enumerate(candidates), key=cost)[0]
        if collective:
            selection = [index]
            root = torch.distributed.get_global_rank(self._process_group, 0) if self._process_group is not None else 0
            torch.distributed.broadcast_object_list(selection, src=root, group=self._process_group)
            index = selection[0]
        chosen = candidates[index]
        if not collective:
            self._schedule_cache[key] = chosen
        return chosen, pilots, False

    def _pool_elements(self, plan, spec):
        pixels = plan.detector_chunk_shape or self.detector_shape
        rank, rays = len(self.volume_shape), plan.view_chunk_size * math.prod(pixels)
        points = spec is not None or self.detector_chunk_shape is not None or pixels != self.detector_shape
        geometry = rays * rank if points else plan.view_chunk_size * rank
        batch = (plan.view_chunk_size * rank + geometry if points else
                 geometry * (4 if self.beam == "cone" else 3))
        return max(math.prod(plan.chunk_shape), rays + batch)

    def _native_tile(self, source, spatial, shape, pipe, is_project):
        pipe.reclaim_output()
        with torch.cuda.device(pipe.device), torch.cuda.stream(pipe.upload):
            staged = torch.empty(shape if is_project or self.beam != "cone" else shape[::-1],
                                 dtype=torch.float32, device=pipe.device)
        if is_project:
            ready = pipe.upload_store(source, spatial, staged)
            with torch.cuda.stream(pipe.compute):
                pipe.compute.wait_event(ready)
                staged.record_stream(pipe.compute)
                native = _prepare_volume(self.beam, staged)
                pipe.stats["layout_preparations"] += 1
        else:
            native = staged
            with torch.cuda.stream(pipe.compute):
                pipe.compute.wait_event(pipe.event(pipe.upload))
                native.record_stream(pipe.compute)
                native.zero_()
        return native

    def _ray_job(self, source, native, geometry, centre, local, global_views, pixels,
                 spec, pipe, groups, index, is_project, before_launch=None,
                 geometry_vjp=False, cpu_geometry=None, cached_rays=None, after_launch=None):
        shape = (local.stop - local.start, *(value.stop - value.start for value in pixels))
        prefetch = groups.get(index, {}).pop("prefetch", None)
        cpu_geometry = (prefetch["geometry"] if prefetch is not None else
                        self._tile_geometry(geometry, global_views, centre, None, pixels, spec)
                        if cpu_geometry is None else cpu_geometry)
        buffers = self._ray_buffers(groups, index, pipe, shape, tuple(value.numel() for value in cpu_geometry))
        if buffers["done"] is not None:
            pipe.upload.wait_event(buffers["done"])
        ray_owner = groups[prefetch["ray_index"]] if prefetch is not None else buffers
        rays = ray_owner["rays"][:math.prod(shape)].view(shape) if cached_rays is None else cached_rays
        staged_geometry = []
        ready = prefetch["ready"] if prefetch is not None else None
        for component, (original, target) in enumerate(zip(cpu_geometry, buffers["geometry"])):
            value = target[:original.numel()].view(original.shape)
            if prefetch is None:
                ready = pipe.upload_tensor(original, value)
            staged_geometry.append(value)
        if not is_project and cached_rays is None and not (prefetch is not None and prefetch["rays"]):
            ready = pipe.upload_store(source, (local, *pixels), rays)
        if before_launch is not None:
            callback = before_launch(rays)
            if callback is not None:
                after_launch = callback
        with torch.cuda.device(pipe.device), torch.cuda.stream(pipe.compute), cuda.gpus[pipe.device.index]:
            pipe.compute.wait_event(ready)
            if buffers["done"] is not None:
                pipe.compute.wait_event(buffers["done"])
            for value in (native, rays, *staged_geometry):
                value.record_stream(pipe.compute)
            if geometry_vjp:
                spacing = self.detector_spacing if self.beam == "cone" else self.detector_spacing[0]
                parts = _geometry_vjp(self.beam, None, rays, tuple(staged_geometry), spacing,
                                      self.voxel_spacing, prepared_volume=native)
            elif is_project:
                _project_into(self.beam, native, tuple(staged_geometry), self.detector_spacing, self.voxel_spacing,
                              rays, after_launch=after_launch)
            else:
                _backproject_into(self.beam, rays, tuple(staged_geometry), self.detector_spacing, self.voxel_spacing,
                                  native, after_launch=after_launch)
            pipe.stats["kernel_launches"] += 1
            done = pipe.event(pipe.compute)
            buffers["done"] = done
            ray_owner["done"] = done
        return parts if geometry_vjp else rays, done, buffers, ray_owner

    def _ray_buffers(self, groups, index, pipe, shape, sizes):
        if index not in groups:
            with torch.cuda.device(pipe.device), torch.cuda.stream(pipe.upload):
                groups[index] = dict(rays=torch.empty(groups.get("_max_rays", math.prod(shape)), dtype=torch.float32, device=pipe.device),
                                     geometry=[torch.empty(size, dtype=torch.float32, device=pipe.device)
                                               for size in groups.get("_max_geometry", sizes)], done=None)
        return groups[index]

    def _groups_for(self, plan, spec):
        rank, pixels = len(self.volume_shape), plan.detector_chunk_shape or self.detector_shape
        rays = plan.view_chunk_size * math.prod(pixels)
        points = spec is not None or self.detector_chunk_shape is not None or pixels != self.detector_shape
        sizes = ((plan.view_chunk_size * rank, rays * rank) if points
                 else (plan.view_chunk_size * rank,) * (4 if self.beam == "cone" else 3))
        return {"_max_rays": rays, "_max_geometry": sizes}

    def _prefetch_job(self, job, source, geometry, spec, transfer, groups, index, is_project,
                      native, reduced, active_rays, next_epoch=False):
        device, local, global_views, pixels, tile_index, spatial, tile_shape, centre = job
        pipe = transfer.for_device(device)
        cpu_geometry = self._tile_geometry(geometry, global_views, centre, None, pixels, spec)
        shape = (local.stop - local.start, *(item.stop - item.start for item in pixels))
        buffers = self._ray_buffers(groups[device], index, pipe, shape, tuple(value.numel() for value in cpu_geometry))
        targets = [target[:value.numel()].view(value.shape)
                   for value, target in zip(cpu_geometry, buffers["geometry"])]
        inputs = list(cpu_geometry)
        read_rays = not is_project and tile_index == 0
        ray_owner, ray_index = buffers, index
        if read_rays:
            if device == active_rays.device and buffers["rays"].untyped_storage()._cdata == active_rays.untyped_storage()._cdata:
                ray_index = 1 - index
                ray_owner = self._ray_buffers(groups[device], ray_index, pipe, shape,
                                               tuple(value.numel() for value in cpu_geometry))
            targets.append(ray_owner["rays"][:math.prod(shape)].view(shape))
        total = sum(value.numel() for value in cpu_geometry) + (math.prod(shape) if read_rays else 0)
        itemsize = torch.empty((), dtype=source.dtype, device="cpu").element_size()
        packed = total <= pipe.capacity and (not read_rays or math.prod(shape) * itemsize <= _pipeline._HOST_SCRATCH_BYTES)
        if packed and read_rays:
            inputs.append(source.read((local, *pixels)).detach())
        packed = packed and all(not value.is_cuda for value in inputs)
        slot, copies = None, []
        if packed:
            slot = pipe.slot()
            offset = 0
            for value, target in zip(inputs, targets):
                pinned = slot.tensor[offset:offset + value.numel()].view(value.shape)
                pinned.copy_(value)
                copies.append((pinned, target))
                offset += value.numel()

        def upload():
            with torch.cuda.device(device), torch.cuda.stream(pipe.upload):
                if buffers["done"] is not None:
                    pipe.upload.wait_event(buffers["done"])
                if read_rays and ray_owner is not buffers and ray_owner["done"] is not None:
                    pipe.upload.wait_event(ray_owner["done"])
                if packed:
                    for pinned, target in copies:
                        _pipeline._copy(target, pinned, pipe.stats)
                    ready = slot.event = pipe.event(pipe.upload)
                else:
                    for value, target in zip(cpu_geometry, targets):
                        ready = pipe.upload_tensor(value, target)
                    if read_rays:
                        ready = pipe.upload_store(source, (local, *pixels), targets[-1])
                buffers["prefetch"] = dict(geometry=cpu_geometry, ready=ready, rays=read_rays, ray_index=ray_index)
            if is_project and device not in native[tile_index] and not reduced:
                if next_epoch:
                    # Current and upcoming native windows already use both
                    # producer epochs; retire any older output epoch first.
                    pipe.reclaim_output(keep=0)
                native[tile_index][device] = self._native_tile(source, spatial, tile_shape, pipe, True)
        return upload

    def _execute_schedule(self, source, sink, is_project, geometry, plan, spec, stats):
        elements = self._pool_elements(plan, spec)
        with _pipeline._TransferPipeline(plan.devices, elements, stats, slots=plan.slots) as transfer:
            groups = {device: self._groups_for(plan, spec)
                      for device in plan.devices}
            sequences = {device: 0 for device in plan.devices}
            tiles = self._volume_tiles(plan.chunk_shape)
            if plan.schedule == "views" and is_project:
                self._project_views(source, sink, geometry, plan, spec, stats, transfer, groups)
                return
            window = plan.window_size
            iterator = iter(tiles)
            first_window = True
            def take_window():
                result = []
                for _ in range(window):
                    tile = next(iterator, None)
                    if tile is None:
                        break
                    result.append(tile)
                return result
            current = take_window()
            native = [{} for _ in current]
            while current:
                upcoming = take_window()
                upcoming_native = [{} for _ in upcoming]
                if getattr(source, "reduced", False):
                    for values, (spatial, shape, _) in zip(native, current):
                        primary = plan.devices[0]
                        pipe = transfer.for_device(primary)
                        values[primary] = self._native_tile(source, spatial, shape, pipe, True)
                        self._reduce_native(values[primary], pipe)
                    del values
                pending = {}
                def jobs():
                    for batches in self._view_rounds(plan):
                        for pixels, _ in _pixel_tiles(self.detector_shape, plan.detector_chunk_shape or self.detector_shape):
                            for device, local, global_views in batches:
                                for tile_index, (spatial, shape, centre) in enumerate(current):
                                    yield device, local, global_views, pixels, tile_index, spatial, shape, centre
                iterator_jobs = iter(jobs())
                job = next(iterator_jobs, None)
                next_window_job = None
                if upcoming and plan.slots > 1:
                    first_batches = next(self._view_rounds(plan), ())
                    if first_batches:
                        next_device, next_local, next_global = first_batches[0]
                        next_pixels, _ = next(_pixel_tiles(self.detector_shape, plan.detector_chunk_shape or self.detector_shape))
                        next_spatial, next_shape, next_centre = upcoming[0]
                        next_window_job = (next_device, next_local, next_global, next_pixels,
                                           0, next_spatial, next_shape, next_centre)
                ray_cache, partial_rays = {}, {}
                while job is not None:
                    following = next(iterator_jobs, None)
                    device, local, global_views, pixels, tile_index, spatial, shape, centre = job
                    pipe = transfer.for_device(device)
                    if device not in native[tile_index]:
                        if getattr(source, "reduced", False):
                            native[tile_index][device] = self._replicate_native(
                                native[tile_index][plan.devices[0]], transfer.for_device(plan.devices[0]), pipe,
                            )
                        else:
                            native[tile_index][device] = self._native_tile(source, spatial, shape, pipe, is_project)
                    index = tile_index if is_project and len(current) > 1 else sequences[device] % plan.slots
                    sequences[device] += 1
                    def before_launch(active_rays):
                        if is_project and tile_index == 0 and device in pending:
                            selected, previous, event, previous_buffers = pending.pop(device)
                            previous_buffers["done"] = pipe.drain_tensor(
                                previous, event, sink, selected, accumulate=not first_window,
                            )
                        next_job = following if following is not None else next_window_job
                        if next_job is not None and plan.slots > 1:
                            next_device, _, _, _, next_tile, _, _, _ = next_job
                            next_tiles = current if following is not None else upcoming
                            next_index = (next_tile if is_project and len(next_tiles) > 1
                                          else sequences[next_device] % plan.slots)
                            if next_device != device or next_index != index:
                                return self._prefetch_job(next_job, source, geometry, spec, transfer, groups,
                                                          next_index, is_project, native if following is not None else upcoming_native,
                                                          getattr(source, "reduced", False), active_rays,
                                                          next_epoch=following is None)
                        return None
                    rays, done, buffers, ray_owner = self._ray_job(
                        source, native[tile_index][device], geometry, centre, local, global_views,
                        pixels, spec, pipe, groups[device], index, is_project, before_launch,
                        cached_rays=ray_cache[device][0] if not is_project and tile_index > 0 else None,
                    )
                    if not is_project:
                        if tile_index == 0:
                            ray_cache[device] = rays, ray_owner
                        ray_cache[device][1]["done"] = done
                    elif len(current) > 1:
                        if tile_index == 0:
                            partial_rays[device] = rays, buffers
                        else:
                            with torch.cuda.stream(pipe.compute):
                                partial_rays[device][0].add_(rays)
                                done = pipe.event(pipe.compute)
                            rays, buffers = partial_rays[device]
                    if is_project and tile_index == len(current) - 1:
                        buffers["done"] = done
                        pending[device] = ((local, *pixels), rays, done, buffers)
                    job = following
                for device, (selected, rays, done, buffers) in pending.items():
                    pipe = transfer.for_device(device)
                    buffers["done"] = pipe.drain_tensor(rays, done, sink, selected, accumulate=not first_window)
                if not is_project:
                    for values, (spatial, shape, _) in zip(native, current):
                        self._finish_native(values, spatial, shape, sink, plan, stats, transfer)
                    del values
                del native, pending
                current, native = upcoming, upcoming_native
                first_window = False

    def _project_views(self, source, sink, geometry, plan, spec, stats, transfer, groups):
        if getattr(source, "reduced", False):
            # A canonical spatial pass issues every SUM, including empty ranks.
            spatial_plan = replace(plan, schedule="spatial", window_size=1)
            self._execute_schedule(source, sink, True, geometry, spatial_plan, spec, stats)
            stats["schedule"], stats["window_size"] = "spatial", 1
            stats["fallback_reason"] = "reduced volume cotangents use canonical spatial tile collectives"
            return
        for batches in self._view_rounds(plan):
            for pixels, _ in _pixel_tiles(self.detector_shape, plan.detector_chunk_shape or self.detector_shape):
                for device, local, global_views in batches:
                    pipe = transfer.for_device(device)
                    accumulator = None
                    count = math.prod(math.ceil(size / part) for size, part in zip(self.volume_shape, plan.chunk_shape))
                    for spatial, shape, centre in self._volume_tiles(plan.chunk_shape):
                        native = self._native_tile(source, spatial, shape, pipe, True)
                        rays, done, buffers, _ = self._ray_job(source, native, geometry, centre, local, global_views,
                                                           pixels, spec, pipe, groups[device], 0, True)
                        with torch.cuda.stream(pipe.compute):
                            if accumulator is None:
                                accumulator = rays.clone() if count > 1 else rays
                            else:
                                accumulator.add_(rays)
                            done = pipe.event(pipe.compute)
                            buffers["done"] = done
                        del native
                    buffers["done"] = pipe.drain_tensor(accumulator, done, sink, (local, *pixels))
                    del accumulator

    def _global_ray_batches(self, plan):
        """Canonical collectives, independent of local device/tile counts."""
        for start in range(0, self._n_views, plan.view_chunk_size):
            views = slice(start, min(start + plan.view_chunk_size, self._n_views))
            for pixels, _ in _pixel_tiles(self.detector_shape, plan.detector_chunk_shape or self.detector_shape):
                yield views, pixels

    def _ray_devices(self, views, plan):
        count = views.stop - views.start
        for index, device in enumerate(plan.devices):
            part = _balanced_slice(count, index, len(plan.devices))
            if part.start < part.stop:
                yield device, part, slice(views.start + part.start, views.start + part.stop)

    def _drain_shared_rays(self, tensor, ready, sink, selected, pipe):
        if not (self._distributed and self.world_size > 1):
            return pipe.drain_tensor(tensor, ready, sink, selected)
        if torch.distributed.get_backend(self._process_group) == "nccl":
            with torch.cuda.device(pipe.device), torch.cuda.stream(pipe.compute):
                torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=self._process_group)
                pipe.stats["collective_calls"] += 1
                pipe.stats["collective_bytes"] += tensor.numel() * tensor.element_size()
                ready = pipe.event(pipe.compute)
            return pipe.drain_tensor(tensor, ready, sink, selected)
        owner = self
        class GlooRays:
            device = torch.device("cpu")
            synchronous = True
            def write(self, selected, value, *, accumulate=False):
                torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM, group=owner._process_group)
                pipe.stats["collective_calls"] += 1
                pipe.stats["collective_bytes"] += value.numel() * value.element_size()
                sink.write(selected, value, accumulate=accumulate)
                if torch.device(getattr(sink, "device", "cpu")).type == "cuda":
                    pipe.stats["h2d_bytes"] += value.numel() * value.element_size()
        return pipe.drain_tensor(tensor, ready, GlooRays(), selected)

    def _space_project(self, source, sink, geometry, plan, spec, stats):
        with _pipeline._TransferPipeline(plan.devices, self._pool_elements(plan, spec), stats,
                                        slots=plan.slots) as transfer:
            primary = transfer.for_device(plan.devices[0])
            groups = {device: self._groups_for(plan, spec) for device in plan.devices}
            for views, pixels in self._global_ray_batches(plan):
                shape = (views.stop - views.start, *(item.stop - item.start for item in pixels))
                primary.reclaim_output()
                with torch.cuda.device(primary.device), torch.cuda.stream(primary.compute):
                    total = torch.zeros(shape, dtype=torch.float32, device=primary.device)
                for device, part, global_views in self._ray_devices(views, plan):
                    pipe = transfer.for_device(device)
                    for spatial, tile_shape, centre in self._volume_tiles(plan.chunk_shape):
                        native = self._native_tile(source, spatial, tile_shape, pipe, True)
                        rays, ready, buffers, _ = self._ray_job(source, native, geometry, centre, global_views,
                                                              global_views, pixels, spec, pipe, groups[device], 0, True)
                        with torch.cuda.device(primary.device), torch.cuda.stream(primary.compute):
                            primary.compute.wait_event(ready)
                            value = rays.to(device=primary.device, non_blocking=True)
                            if device != primary.device:
                                stats["p2p_bytes"] += value.numel() * value.element_size()
                            total[part].add_(value)
                            value.record_stream(primary.compute)
                            ready = primary.event(primary.compute)
                            if device != primary.device:
                                primary.retain(ready, rays, None)
                            buffers["done"] = ready
                        del native, value, rays
                ready = primary.event(primary.compute)
                self._drain_shared_rays(total, ready, sink, (views, *pixels), primary)
                del total

    def _reduced_ray_batch(self, source, views, pixels, pipe, groups):
        shape = (views.stop - views.start, *(item.stop - item.start for item in pixels))
        buffers = self._ray_buffers(groups, 0, pipe, shape, ())
        if buffers["done"] is not None:
            pipe.upload.wait_event(buffers["done"])
        rays = buffers["rays"][:math.prod(shape)].view(shape)
        ready = pipe.upload_store(source, (views, *pixels), rays)
        with torch.cuda.stream(pipe.compute):
            pipe.compute.wait_event(ready)
            rays.record_stream(pipe.compute)
        self._reduce_native(rays, pipe)
        buffers["done"] = pipe.event(pipe.compute)
        return rays, buffers["done"]

    def _space_back_reduced(self, source, sink, geometry, plan, spec, stats, *, volume_source=None, consume=None):
        """Idle ranks repeat every ray SUM for each agreed local tile window."""
        count = math.prod(math.ceil(size / part) for size, part in zip(self._planning_volume_shape, plan.chunk_shape))
        iterator = iter(self._volume_tiles(plan.chunk_shape))
        with _pipeline._TransferPipeline(plan.devices, self._pool_elements(plan, spec), stats,
                                        slots=plan.slots) as transfer:
            primary = transfer.for_device(plan.devices[0])
            # Reduction rays and native job rays need distinct producer epochs.
            reduction = {"_max_rays": plan.view_chunk_size * math.prod(plan.detector_chunk_shape or self.detector_shape),
                         "_max_geometry": ()}
            groups = {device: self._groups_for(plan, spec) for device in plan.devices}
            for _ in range(math.ceil(count / plan.window_size)):
                current = [tile for _ in range(plan.window_size) if (tile := next(iterator, None)) is not None]
                native = [{} for _ in current]
                for views, pixels in self._global_ray_batches(plan):
                    reduced, ready = self._reduced_ray_batch(source, views, pixels, primary, reduction)
                    if not current:
                        continue
                    for device, part, global_views in self._ray_devices(views, plan):
                        pipe = transfer.for_device(device)
                        shape = (part.stop - part.start, *(item.stop - item.start for item in pixels))
                        buffers = self._ray_buffers(groups[device], 0, pipe, shape, ())
                        if buffers["done"] is not None:
                            pipe.upload.wait_event(buffers["done"])
                        ray_input = buffers["rays"][:math.prod(shape)].view(shape)
                        pipe.upload.wait_event(ready)
                        copied = pipe.upload_tensor(reduced[part], ray_input)
                        primary.upload.wait_event(copied)
                        for values, (spatial, tile_shape, centre) in zip(native, current):
                            if device not in values:
                                values[device] = self._native_tile(volume_source, spatial, tile_shape, pipe,
                                                                  volume_source is not None)
                            if consume is not None:
                                consume(pipe, groups[device], values[device], global_views, global_views, pixels,
                                        centre, ray_input)
                            else:
                                self._ray_job(source, values[device], geometry, centre, global_views, global_views,
                                              pixels, spec, pipe, groups[device], 0, False, cached_rays=ray_input)
                        primary.upload.wait_event(pipe.event(pipe.compute))
                if consume is None:
                    for values, (spatial, tile_shape, _) in zip(native, current):
                        self._finish_native(values, spatial, tile_shape, sink, plan, stats, transfer)
                if current:
                    del values
                del native

    def _replicate_native(self, native, origin, target):
        with torch.cuda.device(target.device), torch.cuda.stream(target.upload):
            target.upload.wait_event(origin.event(origin.compute))
            copied = native.to(device=target.device, non_blocking=True)
            target.stats["p2p_bytes"] += copied.numel() * copied.element_size()
            target.retain(target.event(target.upload), native, copied)
        target.compute.wait_event(target.event(target.upload))
        copied.record_stream(target.compute)
        return copied

    def _reduce_native(self, native, pipe):
        stats = pipe.stats
        if torch.distributed.get_backend(self._process_group) == "nccl":
            with torch.cuda.device(pipe.device), torch.cuda.stream(pipe.compute):
                torch.distributed.all_reduce(native, op=torch.distributed.ReduceOp.SUM, group=self._process_group)
                stats["collective_calls"] += 1
                stats["collective_bytes"] += native.numel() * native.element_size()
            return
        owner = self
        ready = pipe.event(pipe.compute)

        class GlooTile:
            device = torch.device("cpu")
            synchronous = True
            def write(self, selected, value, *, accumulate=False):
                torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM, group=owner._process_group)
                stats["collective_calls"] += 1
                stats["collective_bytes"] += value.numel() * value.element_size()
                pipe.upload.wait_event(ready)
                pipe.upload_tensor(value, native[selected])

        pipe.drain_tensor(native, ready, GlooTile(), tuple(slice(0, size) for size in native.shape))
        pipe.compute.wait_event(pipe.event(pipe.upload))

    def _finish_native(self, values, spatial, shape, sink, plan, stats, transfer):
        device = plan.devices[0]
        pipe = transfer.for_device(device)
        pipe.reclaim_output()
        if device not in values:
            values[device] = self._native_tile(None, spatial, shape, pipe, False)
        accumulator = values[device]
        with torch.cuda.device(device), torch.cuda.stream(pipe.compute):
            for peer, partial in values.items():
                if peer == device:
                    continue
                peer_pipe = transfer.for_device(peer)
                pipe.compute.wait_event(peer_pipe.event(peer_pipe.compute))
                copied = partial.to(device=device, non_blocking=True)
                stats["p2p_bytes"] += copied.numel() * copied.element_size()
                accumulator.add_(copied)
                copied.record_stream(pipe.compute)
                pipe.retain(pipe.event(pipe.compute), partial, None)
                del copied
            if self.partition == "views" and self._distributed and self.world_size > 1 and torch.distributed.get_backend(self._process_group) == "nccl":
                torch.distributed.all_reduce(accumulator, op=torch.distributed.ReduceOp.SUM, group=self._process_group)
                stats["collective_calls"] += 1
                stats["collective_bytes"] += accumulator.numel() * accumulator.element_size()
            output = accumulator.permute(2, 1, 0).contiguous() if self.beam == "cone" else accumulator
            ready = pipe.event(pipe.compute)
        if self.partition == "views" and self._distributed and self.world_size > 1 and torch.distributed.get_backend(self._process_group) != "nccl":
            owner = self

            class GlooOutput:
                device = torch.device("cpu")
                synchronous = True
                def write(self, selected, value, *, accumulate=False):
                    torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM, group=owner._process_group)
                    stats["collective_calls"] += 1
                    stats["collective_bytes"] += value.numel() * value.element_size()
                    sink.write(selected, value, accumulate=accumulate)

            pipe.drain_tensor(output, ready, GlooOutput(), spatial)
            for slot in pipe.host:
                slot.finish()
        else:
            pipe.drain_tensor(output, ready, sink, spatial)

    def _pilot_candidate(self, source, sink, is_project, geometry, plan, spec):
        shape = list(plan.chunk_shape)
        while math.prod(shape) > 4096:
            axis = max(range(len(shape)), key=shape.__getitem__)
            shape[axis] = max(1, shape[axis] // 2)
        if plan.schedule != "spatial" and tuple(shape) == self.volume_shape and max(shape) > 1:
            axis = max(range(len(shape)), key=shape.__getitem__)
            shape[axis] = max(1, shape[axis] // 2)
        pixels = list(plan.detector_chunk_shape or self.detector_shape)
        views = min(4, plan.view_chunk_size, self.projection_shape[0])
        while views * math.prod(pixels) > 4096:
            axis = max(range(len(pixels)), key=pixels.__getitem__)
            pixels[axis] = max(1, pixels[axis] // 2)
        trial = replace(plan, chunk_shape=tuple(shape), detector_chunk_shape=tuple(pixels), view_chunk_size=views)
        stats = self._execution_stats(trial, is_project)
        tile_iterator = iter(self._volume_tiles(trial.chunk_shape))
        first_tile = next(tile_iterator)
        tiles = [first_tile, next(tile_iterator, first_tile)]
        first_view = slice(0, views)
        second_view = slice(views, min(2 * views, self.projection_shape[0])) if self.projection_shape[0] > views else first_view
        pixel_slices = tuple(slice(0, size) for size in pixels)
        elapsed = 0
        measured = ([(0, first_view), (0, second_view)] if plan.schedule == "spatial" else
                    [(0, first_view), (1, first_view)] if plan.schedule == "views" else
                    [(0, first_view), (1, first_view), (0, second_view)])
        for sequence in ([(0, first_view)], measured):
            start = time.perf_counter()
            with _pipeline._TransferPipeline((plan.devices[0],), self._pool_elements(trial, spec), stats,
                                            slots=plan.slots) as transfer:
                pipe = transfer.for_device(plan.devices[0])
                native_cache, ray_cache = {}, {}
                native = None
                groups = self._groups_for(trial, spec)
                for index, (tile_index, local) in enumerate(sequence):
                    spatial, tile_shape, centre = tiles[tile_index]
                    if plan.schedule == "views" and is_project and index:
                        native_cache.clear()
                        native = None
                    if tile_index not in native_cache or (plan.schedule == "views" and is_project):
                        native_cache[tile_index] = self._native_tile(source, spatial, tile_shape, pipe, is_project)
                    native = native_cache[tile_index]
                    global_views = slice(self.view_slice.start + local.start, self.view_slice.start + local.stop)
                    cached = ray_cache.get(local.start) if not is_project and plan.schedule != "spatial" else None
                    rays, ready, buffers, ray_owner = self._ray_job(source, native, geometry, centre, local, global_views,
                                                       pixel_slices, spec, pipe, groups, index % plan.slots,
                                                       is_project, cached_rays=cached[0] if cached else None)
                    if not is_project:
                        if cached is None:
                            ray_cache[local.start] = rays, ray_owner
                        ray_cache[local.start][1]["done"] = ready
                    with torch.cuda.stream(pipe.compute):
                        output = rays if is_project else native.permute(2, 1, 0).contiguous() if self.beam == "cone" else native
                        ready = pipe.event(pipe.compute)
                    result = torch.empty(output.shape, dtype=torch.float32)
                    selected = tuple(slice(0, size) for size in output.shape)
                    buffers["done"] = pipe.drain_tensor(output, ready, TensorStore(result), selected)
            elapsed = (time.perf_counter() - start) * 1000
            del native, rays, output, result, native_cache, ray_cache, groups, buffers, ray_owner, cached, pipe, transfer
        tiles = math.prod(math.ceil(size / part) for size, part in zip(self.volume_shape, plan.chunk_shape))
        batches = sum(math.ceil((_balanced_slice(self.projection_shape[0], index, len(plan.devices)).stop
                                - _balanced_slice(self.projection_shape[0], index, len(plan.devices)).start) / plan.view_chunk_size)
                      for index in range(len(plan.devices)))
        rectangles = math.prod(math.ceil(size / part) for size, part in zip(self.detector_shape, plan.detector_chunk_shape))
        volume_bytes, ray_bytes = 4 * math.prod(self.volume_shape), 4 * math.prod(self.projection_shape)
        rank = len(self.volume_shape)
        points = spec is not None or self.detector_chunk_shape is not None or plan.detector_chunk_shape != self.detector_shape
        geometry_bytes = 4 * rank * self.projection_shape[0] * (
            1 + math.prod(self.detector_shape) if points else (4 if self.beam == "cone" else 3) * rectangles)
        uploads = (volume_bytes * (batches * rectangles if plan.schedule == "views" else min(len(plan.devices), self.projection_shape[0]))
                   if is_project else ray_bytes * math.ceil(tiles / plan.window_size))
        if torch.device(getattr(source, "device", "cpu")).type == "cuda":
            uploads = 0
        stats.update(schedule=plan.schedule, chunk_shape=plan.chunk_shape, view_chunk_size=plan.view_chunk_size,
                     detector_chunk_shape=plan.detector_chunk_shape, window_size=plan.window_size, elapsed_ms=elapsed,
                     measured_kernel_launches=len(measured), representative_device=plan.devices[0].index,
                     timing_scope="one local representative device; residency/order prefix after one warmup launch",
                      selection_policy="measured prefix time scaled by full native launch count, then estimated copy bytes",
                     estimated_kernel_launches=tiles * batches * rectangles,
                     estimated_elapsed_ms=elapsed / len(measured) * tiles * batches * rectangles,
                     estimated_h2d_bytes=uploads + tiles * geometry_bytes,
                     estimated_d2h_bytes=(ray_bytes * (1 if plan.schedule == "views" else math.ceil(tiles / plan.window_size))
                        if is_project else volume_bytes) if torch.device(getattr(sink, "device", "cpu")).type == "cpu" else 0)
        return stats

    def _reduce_streamed_(self, tensor, plan, stats, spec):
        """SUM gradient fragments without cloning a complete host gradient."""
        block = max(1, min(tensor.numel(), _pipeline._HOST_SCRATCH_BYTES // tensor.element_size(),
                           max(self._groups_for(plan, spec)["_max_geometry"])))
        slices = _pipeline._fragments(tuple(tensor.shape), block)
        if torch.distributed.get_backend(self._process_group) != "nccl":
            staged = torch.empty(block, dtype=tensor.dtype)
            for selected in slices:
                value = tensor[selected]
                fragment = staged[:value.numel()].view(value.shape)
                fragment.copy_(value)
                torch.distributed.all_reduce(fragment, op=torch.distributed.ReduceOp.SUM, group=self._process_group)
                stats["collective_calls"] += 1
                stats["collective_bytes"] += fragment.numel() * fragment.element_size()
                value.copy_(fragment)
            return
        store = TensorStore(tensor)
        with _pipeline._TransferPipeline((plan.devices[0],), block, stats, slots=1) as transfer:
            pipe = transfer.for_device(plan.devices[0])
            with torch.cuda.device(pipe.device), torch.cuda.stream(pipe.upload):
                staged = torch.empty(block, dtype=torch.float32, device=pipe.device)
            for selected in slices:
                shape = tuple(item.stop - item.start for item in selected)
                fragment = staged[:math.prod(shape)].view(shape)
                ready = pipe.upload_store(store, selected, fragment)
                with torch.cuda.stream(pipe.compute):
                    pipe.compute.wait_event(ready)
                    fragment.record_stream(pipe.compute)
                    torch.distributed.all_reduce(fragment, op=torch.distributed.ReduceOp.SUM, group=self._process_group)
                    stats["collective_calls"] += 1
                    stats["collective_bytes"] += fragment.numel() * fragment.element_size()
                    ready = pipe.event(pipe.compute)
                consumed = pipe.drain_tensor(fragment, ready, store, selected)
                pipe.upload.wait_event(consumed)

    def _geometry_grad(self, volume, cotangent, geometry, plan=None, geometry_spec=None):
        """Return d<cotangent, A(geometry) volume>/d(geometry) over this rank's views.

        ``cotangent`` holds the rank's view shard. The result covers the full
        trajectory; distributed mode sums it over ranks.
        """
        if plan is not None:
            return self._geometry_grad_streamed(volume, cotangent, geometry, plan, geometry_spec)
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

    def _geometry_grad_streamed(self, volume, cotangent, geometry, plan, geometry_spec=None):
        self.last_execution_stats = None
        grads = [torch.zeros(component.shape, dtype=torch.float32, device="cpu")
                 for component in geometry]
        detector_chunk = plan.detector_chunk_shape or self.detector_shape
        sampled = (geometry_spec is not None and geometry_spec.kind == "sampled") or (
            (geometry_spec is None or geometry_spec.kind != "points")
            and (self.detector_chunk_shape is not None or detector_chunk != self.detector_shape)
        )
        source = volume if isinstance(volume, _ReducedInput) else TensorStore(volume)
        cot_store = cotangent if isinstance(cotangent, _ReducedInput) else TensorStore(cotangent)
        stats = self._execution_stats(plan, True)
        stats["operation"] = "geometry_vjp"
        def consume(pipe, local_groups, native, local_slice, global_slice, pixels, centre, cached_rays=None):
            if sampled:
                with torch.enable_grad():
                    inputs = tuple(component.detach().requires_grad_() for component in geometry)
                    local_geometry = _sample_geometry(self.beam, self.detector_shape, self.detector_spacing,
                                                      inputs, geometry_spec, global_slice, pixels, self.voxel_spacing)
                cpu_geometry = tuple(value.detach() - value.new_tensor(centre)
                                     if index == 1 or self.beam != "parallel" else value.detach()
                                     for index, value in enumerate(local_geometry))
            else:
                cpu_geometry = self._tile_geometry(geometry, global_slice, centre, None, pixels, geometry_spec)
            parts, ready, _, _ = self._ray_job(cot_store, native, geometry, centre, local_slice, global_slice,
                                               pixels, geometry_spec, pipe, local_groups, 0, False,
                                               geometry_vjp=True, cpu_geometry=cpu_geometry, cached_rays=cached_rays)
            weights = []
            for part in parts:
                weight = torch.empty(part.shape, dtype=torch.float32)
                pipe.drain_tensor(part, ready, TensorStore(weight),
                                  tuple(slice(0, size) for size in weight.shape))
                weights.append(weight)
            for slot in pipe.host:
                slot.finish()
            if sampled:
                weights = tuple(weight.to(dtype=value.dtype) for weight, value in zip(weights, local_geometry))
                with torch.enable_grad():
                    reduced = torch.autograd.grad(local_geometry, inputs, weights, allow_unused=True)
                for grad, part in zip(grads, reduced):
                    if part is not None:
                        grad.add_(part)
            else:
                for index, (grad, weight) in enumerate(zip(grads, weights)):
                    selected = ((global_slice, *pixels) if geometry_spec is not None
                                and geometry_spec.kind == "points" and index == 1 else global_slice)
                    grad[selected].add_(weight)
        if self.partition == "space" and getattr(cot_store, "reduced", False):
            self._space_back_reduced(cot_store, None, geometry, plan, geometry_spec, stats,
                                     volume_source=source, consume=consume)
            if self._distributed and self.world_size > 1:
                for grad in grads:
                    self._reduce_streamed_(grad, plan, stats, geometry_spec)
            self.last_execution_stats = stats
            return tuple(grad.to(device=component.device, dtype=component.dtype)
                         for grad, component in zip(grads, geometry))
        groups = {device: {} for device in plan.devices}
        with _pipeline._TransferPipeline(plan.devices, self._pool_elements(plan, geometry_spec), stats,
                                        slots=plan.slots) as transfer:
            for spatial_slice, shape, centre in self._volume_tiles(plan.chunk_shape):
                volume_tiles = {}
                if getattr(source, "reduced", False):
                    primary = plan.devices[0]
                    pipe = transfer.for_device(primary)
                    volume_tiles[primary] = self._native_tile(source, spatial_slice, shape, pipe, True)
                    self._reduce_native(volume_tiles[primary], pipe)
                for batches in self._view_rounds(plan):
                    for pixels, _ in _pixel_tiles(self.detector_shape, detector_chunk):
                        for device, local_slice, global_slice in batches:
                            pipe = transfer.for_device(device)
                            if device not in volume_tiles:
                                volume_tiles[device] = (self._replicate_native(
                                    volume_tiles[plan.devices[0]], transfer.for_device(plan.devices[0]), pipe,
                                ) if getattr(source, "reduced", False) else
                                    self._native_tile(source, spatial_slice, shape, pipe, True))
                            consume(pipe, groups[device], volume_tiles[device], local_slice, global_slice, pixels, centre)
                del volume_tiles
        del groups
        if self._distributed and self.world_size > 1:
            for grad in grads:
                self._reduce_streamed_(grad, plan, stats, geometry_spec)
        self.last_execution_stats = stats
        return tuple(grad.to(device=component.device, dtype=component.dtype)
                     for grad, component in zip(grads, geometry))

    def _project_raw(self, volume, geometry):
        if len(geometry) == 2:
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
        if len(geometry) == 2:
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

        The output shape is ``projection_shape``; views partitioning uses this
        rank's contiguous view slice, while space partitioning replicates all
        views. The input shape is the owned ``volume_shape``. The result stays on
        ``volume.device``. CPU data streams through CUDA automatically; manual
        chunk limits override automatic sizing. Execution requires CUDA.
        """
        self.last_execution_stats = None
        self._snapshot_d2h_bytes = 0
        plan = self._execution_plan(volume, True)
        geometry_spec, geometry = self._effective_geometry(plan, volume.device)
        return _ProjectorAutograd.apply(volume, self, "project", plan, geometry_spec, self._participation(), *geometry)

    def backproject(self, sinogram):
        """Backproject a floating-point CPU/CUDA sinogram with the matched adjoint.

        The input must have shape ``projection_shape`` and uses ``(views, U,
        V)`` order for cone beams. The float32 volume is returned to
        ``sinogram.device``. CPU data streams through CUDA automatically;
        manual chunk limits override automatic sizing. Execution requires CUDA.
        Views partitioning SUM-reduces the volume to every rank; space
        partitioning returns only ``volume_slice`` in ``volume_shape`` order.
        Divide replicated-output losses by ``world_size``.
        """
        self.last_execution_stats = None
        self._snapshot_d2h_bytes = 0
        plan = self._execution_plan(sinogram, False)
        geometry_spec, geometry = self._effective_geometry(plan, sinogram.device)
        return _ProjectorAutograd.apply(sinogram, self, "backproject", plan, geometry_spec, self._participation(), *geometry)

    def project_into(self, volume, output):
        """Numerically project tensor/block-store input into supplied float32 output.

        Returns ``output``; no complete result is allocated. Direct tensors
        requiring autograd must be detached or used under ``torch.no_grad()``.
        Known overlapping/read-only outputs are rejected before execution.
        """
        return self._execute_into(volume, output, True)

    def backproject_into(self, sinogram, output):
        """Write the matched numerical adjoint into supplied float32 block output.

        Accepts tensors or stores with read/write/shape/dtype/flush metadata.
        Returns ``output`` after bounded view/pixel contributions complete.
        """
        return self._execute_into(sinogram, output, False)

    def _execute_into(self, value, output, is_project):
        self.last_execution_stats = None
        self._snapshot_d2h_bytes = 0
        source, sink = _validate_stores(self, value, output, is_project)
        with torch.no_grad():
            plan = self._execution_plan(source, is_project, sink)
            geometry_spec, geometry = self._effective_geometry(plan, torch.device(getattr(source, "device", "cpu")))
            self._run_store_streamed(source, sink, is_project, geometry if geometry else None, plan, geometry_spec)
        return output

    def __call__(self, volume):
        """Alias for :meth:`project`."""
        return self.project(volume)

    def _effective_geometry(self, plan, input_device):
        self._agree_geometry_layout()
        if isinstance(self.detector_surface, ParameterizedSurface):
            surface = self.detector_surface
            frames = self._learnable if self._learnable else self._trajectory
            values = (*frames, *surface.parameters)
            snapshots = tuple(value.clone() if value.device.type == "cpu" else value.to(device="cpu", copy=True)
                              for value in values)
            self._snapshot_d2h_bytes += sum(value.numel() * value.element_size() for value in values if value.is_cuda)
            return _GeometrySpec("sampled", surface.sampler), snapshots
        if self.detector_surface is None:
            if plan is not None:
                self._snapshot_d2h_bytes += sum(value.numel() * value.element_size() for value in self._learnable if value.is_cuda)
            return None, tuple(component.to(device="cpu" if plan is not None else component.device).clone()
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
            geometry = self._sample_surface(
                cpu_only=plan is not None, input_device=None if plan is not None else input_device
            )[0]
            return _GeometrySpec("points"), tuple(component.clone() for component in geometry)

    def _sample_surface(self, cpu_only=False, validate_on_cpu=False, input_device=None):
        """Return effective per-view source/direction and sampled pixel points.

        ``input_device`` is the CUDA input device of a full-volume call. The
        execution plan budgets world points only on the compute and input
        devices, so offsets on another GPU are expanded on the input device.
        """
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
        if (input_device is not None and offsets.is_cuda
                and offsets.device not in (*(self._devices or ()), input_device)):
            offsets = offsets.to(device=input_device)
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
