"""High-level operators for arbitrary CT trajectories."""

import math
import numbers

import torch
from numba import cuda

from .projectors import (
    _geometry_vjp,
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


def _validate_trajectory(trajectory, beam):
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
    if beam in ("fan", "cone"):
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


def _all_reduce_copy(projector, tensor):
    """Return a SUM over ranks of ``tensor`` (the tensor itself without ranks)."""
    if not (projector._distributed and projector.world_size > 1):
        return tensor
    reduced = tensor.detach().clone(memory_format=torch.contiguous_format)
    with torch.cuda.device(reduced.device):
        torch.distributed.all_reduce(
            reduced, op=torch.distributed.ReduceOp.SUM, group=projector._process_group
        )
    return reduced


class _ProjectorAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, projector, mode, *geometry):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("input must be a torch.Tensor")
        ctx.projector = projector
        ctx.mode = mode
        ctx.input_device = tensor.device
        ctx.input_dtype = tensor.dtype
        ctx.save_for_backward(tensor if geometry else None, *geometry)
        return projector._run(
            tensor,
            is_project=mode != "backproject",
            reduce_cotangent=mode == "project_reduced",
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
                grad_output, projector, _ADJOINT_MODE[ctx.mode], *geometry
            ).to(device=ctx.input_device, dtype=ctx.input_dtype)
            if not ctx.needs_input_grad[0]:
                grad_input = None
        geometry_grads = (None,) * len(geometry)
        if any(ctx.needs_input_grad[3:]) or (collective and geometry):
            # Every mode is <cotangent, A_local(geometry) volume> for some pair.
            if ctx.mode == "project":
                volume, cotangent = tensor, grad_output
            elif ctx.mode == "project_reduced":
                volume, cotangent = _all_reduce_copy(projector, tensor), grad_output
            else:
                volume, cotangent = _all_reduce_copy(projector, grad_output), tensor
            grads = _GeometryGradAutograd.apply(projector, volume, cotangent, *geometry)
            geometry_grads = tuple(
                grad if needed else None
                for grad, needed in zip(grads, ctx.needs_input_grad[3:])
            )
        return (grad_input, None, None, *geometry_grads)


class _GeometryGradAutograd(torch.autograd.Function):
    """First-order geometry gradient; differentiating it again raises."""

    @staticmethod
    def forward(ctx, projector, volume, cotangent, *geometry):
        return projector._geometry_grad(volume, cotangent, geometry)

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise RuntimeError(
            "diffct does not support second derivatives with respect to the geometry"
        )


class Projector:
    """Project CUDA volumes along fixed parallel, fan, or cone trajectories.

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

    Volumes use ``(H, W)`` for parallel and fan beams and ``(D, H, W)`` for
    cone beams. Sinograms use ``(views, detectors)`` or ``(views, U, V)``.
    Results are float32 and returned to the input tensor's CUDA device. Local
    devices split views in list order; distributed mode returns rank-local
    projections and replicated backprojections. Distributed projection losses
    use SUM across ranks; divide replicated backprojection losses by
    ``world_size`` and do not add a second DDP reduction for image or geometry
    gradients.
    """

    def __init__(self, trajectory, volume_shape, detector_shape, *, beam="cone",
                 detector_spacing=1.0, voxel_spacing=1.0, devices=None,
                 distributed=False, process_group=None):
        if beam not in ("parallel", "fan", "cone"):
            raise ValueError("beam must be 'parallel', 'fan', or 'cone'")
        self.beam = beam
        rank = 3 if beam == "cone" else 2
        self.volume_shape = _shape(volume_shape, rank, "volume_shape")
        if beam == "cone":
            self.detector_shape = _shape(detector_shape, 2, "detector_shape")
        elif isinstance(detector_shape, (tuple, list)):
            self.detector_shape = _shape(detector_shape, 1, "detector_shape")
        else:
            self.detector_shape = (_positive_int(detector_shape, "detector_shape"),)

        self.detector_spacing = _detector_spacing(detector_spacing, beam)
        self.voxel_spacing = _positive_float(voxel_spacing, "voxel_spacing")
        self._trajectory, n_views = _validate_trajectory(trajectory, beam)
        self._learnable = (
            tuple(trajectory) if any(c.requires_grad for c in trajectory) else ()
        )
        if beam in ("fan", "cone"):
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
            flag = 1 if self._learnable else 0
            bounds = torch.tensor([flag, -flag], device=device)
            torch.distributed.all_reduce(
                bounds, op=torch.distributed.ReduceOp.MAX, group=process_group
            )
            if bounds[0].item() != -bounds[1].item():
                raise ValueError(
                    "all ranks must agree on whether the trajectory requires gradients"
                )
        self.view_slice = _balanced_slice(n_views, self.rank, self.world_size)
        self.projection_shape = (
            self.view_slice.stop - self.view_slice.start,
            *self.detector_shape,
        )

    def _geometry_for(self, device, global_slice):
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

    def _run(self, tensor, is_project, reduce_cotangent=False):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("input must be a torch.Tensor")
        expected_shape = self.volume_shape if is_project else self.projection_shape
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"input shape must be {expected_shape}")
        if not tensor.is_cuda:
            raise TypeError("input tensor must be on CUDA")
        if not torch.is_floating_point(tensor):
            raise TypeError("input tensor must have floating-point dtype")
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
                staged_geometry = self._geometry_for(device, global_slice)
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

    def _geometry_grad(self, volume, cotangent, geometry):
        """Return d<cotangent, A(geometry) volume>/d(geometry) over this rank's views.

        ``cotangent`` holds the rank's view shard. The result covers the full
        trajectory; distributed mode sums it over ranks.
        """
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
                parts = _geometry_vjp(
                    self.beam,
                    volume.detach().to(device=device),
                    cotangent.detach()[local_slice].to(device=device),
                    tuple(
                        component.detach()[global_slice].to(device=device, dtype=torch.float32)
                        for component in geometry
                    ),
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

    def _project_raw(self, volume, geometry):
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

    def _backproject_raw(self, sinogram, geometry):
        if self.beam == "parallel":
            return ParallelBackprojectorFunction.apply(
                sinogram, *geometry, self.detector_spacing[0],
                *self.volume_shape, self.voxel_spacing,
            )
        if self.beam == "fan":
            return FanBackprojectorFunction.apply(
                sinogram, *geometry, self.detector_spacing[0],
                *self.volume_shape, self.voxel_spacing,
            )
        return ConeBackprojectorFunction.apply(
            sinogram, *geometry, *self.volume_shape,
            *self.detector_spacing, self.voxel_spacing,
        )

    def project(self, volume):
        """Project a ``(H, W)`` or ``(D, H, W)`` CUDA tensor to float32.

        The output shape is ``projection_shape``; in distributed mode it holds
        this rank's contiguous view slice. The result stays on ``volume.device``.
        """
        return _ProjectorAutograd.apply(volume, self, "project", *self._learnable)

    def backproject(self, sinogram):
        """Backproject a floating-point CUDA sinogram.

        The input must have shape ``projection_shape`` and uses ``(views, U,
        V)`` order for cone beams. The float32 volume is returned to
        ``sinogram.device``; distributed mode SUM-reduces it to every rank, so
        divide a replicated-output loss by ``world_size``.
        """
        return _ProjectorAutograd.apply(sinogram, self, "backproject", *self._learnable)

    def __call__(self, volume):
        """Alias for :meth:`project`."""
        return self.project(volume)
