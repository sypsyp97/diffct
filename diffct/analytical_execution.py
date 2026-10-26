"""Bounded analytical filtering and voxel gathers in the global physical frame."""

import hashlib
import math
import numbers
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn.functional as distributed_functional

from .analytical import (
    _analytical_geometry, _angular_scan_period, _ramp_window,
    angular_integration_weights, cone_weighted_backproject,
    fan_weighted_backproject, parallel_weighted_backproject,
)
from .backends import _zero_dependency, select_backend, voxel_spacing_tuple
from .constants import _native_spacing
from .chunking import _memory_budget, _pixel_tiles
from .distribution import Partition, preflight
from .operators import _detector_spacing, _normalize_devices, _positive_int, _shape, _validate_trajectory
from .reconstruction import _angles, _cosine, _filter, _fixed, _parker_setup
from .streaming import _as_store, _validate_stores


def _weight(value, shape, name, *, exact=False):
    if value is None:
        return None
    value = torch.as_tensor(value)
    if value.is_complex() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite real floating-point values")
    try:
        valid = value.shape == shape if exact else torch.broadcast_shapes(value.shape, shape) == shape
    except RuntimeError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must {'have' if exact else 'broadcast to'} shape {shape}")
    return value


def _sum(value, group, count):
    if count == 1 or value.numel() == 0:
        return value
    device = (torch.device("cuda", torch.cuda.current_device()) if dist.get_backend(group) == "nccl"
              else torch.device("cpu"))
    staged = value.to(device)
    if torch.is_grad_enabled() and staged.requires_grad:
        reduced = distributed_functional.all_reduce(staged, op=dist.ReduceOp.SUM, group=group)
    else:
        reduced = staged.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=group)
    return reduced.to(value.device)


def _sum_blocks(value, group, count, block_elements):
    if count == 1:
        return value
    flat = value.reshape(-1)
    blocks = [_sum(flat.narrow(0, start, min(block_elements, flat.numel() - start)), group, count)
              for start in range(0, flat.numel(), block_elements)]
    return torch.cat(blocks).reshape_as(value) if blocks else value.clone()


class _AnalyticalInputs(torch.autograd.Function):
    """Transpose all replicated inputs in one fixed collective order.

    A single multi-output node waits for both data and weight cotangents.
    Empty ranks therefore issue the same data/parameter reductions as peers.
    """

    @staticmethod
    def forward(ctx, participation, partition, block_elements, data, *parameters):
        ctx.partition, ctx.block_elements = partition, block_elements
        return tuple(value.view_as(value) for value in (data, *parameters))

    @staticmethod
    def backward(ctx, data, *parameters):
        partition = ctx.partition
        gradients = [_sum_blocks(data, partition.space_group, partition.space_world_size, ctx.block_elements)]
        gradients.extend(_sum_blocks(value, partition.process_group, partition.world_size, ctx.block_elements)
                         for value in parameters)
        return (None, None, None,
                *(value if needed else None for value, needed in zip(gradients, ctx.needs_input_grad[3:])))


class AnalyticalExecution:
    def __init__(self, source, trajectory, volume_shape, angles, **kwargs):
        distributed, group = kwargs.get("distributed", False), kwargs.get("process_group")
        if not isinstance(distributed, bool):
            raise TypeError("distributed must be a bool")
        if distributed and (not dist.is_available() or not dist.is_initialized()):
            raise RuntimeError("distributed=True requires an initialized process group")
        world = dist.get_world_size(group) if distributed else 1
        error, metadata = None, None
        try:
            beam = kwargs.get("beam")
            if beam not in ("parallel", "fan", "cone"):
                raise ValueError("beam must be 'parallel', 'fan', or 'cone'")
            shape = _shape(volume_shape, 3 if beam == "cone" else 2, "volume_shape")
            for component in trajectory:
                _fixed(component, "trajectory")
            _, views = _validate_trajectory(trajectory, beam)
            _angles(angles, views, torch.device("cpu"))
            metadata = (shape, views, kwargs.get("partition", "views"), kwargs.get("partition_grid"))
        except (ValueError, TypeError, RuntimeError, OSError) as caught:
            error = caught
        preflight(error, metadata, process_group=group, world_size=world)
        self.partition = Partition(shape, views, distributed=distributed, process_group=group,
                                   partition=kwargs.get("partition", "views"),
                                   partition_grid=kwargs.get("partition_grid"))
        error = None
        try:
            self._initialize(source, trajectory, volume_shape, angles, **kwargs)
            weights = (self.projection_weights, self.angular_weights, self.filter_response)
            schema = tuple((tuple(value.shape), str(value.dtype), value.requires_grad)
                           if value is not None else None for value in weights)
            acquisition = tuple((tuple(value.shape), str(value.dtype), hashlib.sha256(
                value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
                for value in (*self.geometry, self.angles))
            metadata = (self.beam, str(self.dtype), self.backend, self.spacing, self.pitch,
                        self.detector_shape, self.view_chunk_size, self.chunk_shape,
                        self.window if self.filter_response is None else None, self.pad_factor,
                        bool(self.short_scan), schema, acquisition,
                        None if self.isocenter is None else tuple(self.isocenter.tolist()))
        except (ValueError, TypeError, RuntimeError, OSError) as caught:
            error = caught
        self.partition.preflight(error, metadata)
        self._admit_native(kwargs.get("view_chunk_size"), kwargs.get("volume_chunk_shape"))
        self.participation = None
        self.input_dependency = None

    def _initialize(self, source, trajectory, volume_shape, angles, *, beam,
                 detector_spacing=1.0, voxel_spacing=1.0, window="ram-lak", pad_factor=2,
                 projection_weights=None, angular_weights=None, filter_response=None,
                 short_scan=False, isocenter=None, backend="auto", precision="fast", devices=None,
                 view_chunk_size=None, volume_chunk_shape=None, distributed=False, process_group=None,
                 partition="views", partition_grid=None):
        if beam not in ("parallel", "fan", "cone"):
            raise ValueError("beam must be 'parallel', 'fan', or 'cone'")
        self.beam = beam
        self.backend = select_backend(backend, precision, voxel_spacing, partition)
        self.dtype = torch.float64 if precision == "float64" else torch.float32
        self.rank = 3 if beam == "cone" else 2
        self.global_volume_shape = _shape(volume_shape, self.rank, "volume_shape")
        self.spacing = voxel_spacing_tuple(voxel_spacing, self.rank)
        self.voxel_spacing = self.spacing if isinstance(voxel_spacing, (tuple, list)) else self.spacing[0]
        if self.backend == "numba" and isinstance(self.voxel_spacing, tuple):
            _native_spacing(self.voxel_spacing)
        self.pitch = _detector_spacing(detector_spacing, beam)
        self.devices = _normalize_devices(devices)
        if not isinstance(source, torch.Tensor):
            source = _as_store(source)
        if len(source.shape) != self.rank or min(source.shape[1:]) < 2:
            raise ValueError("sinogram rank must match the beam and detector axes need at least two bins")
        if isinstance(source, torch.Tensor):
            if not torch.is_floating_point(source):
                raise ValueError("sinogram must have a real floating-point dtype")
        elif source.dtype not in (torch.float32, torch.float64):
            raise TypeError("sinogram store must have dtype float32 or float64")
        self.source = source
        self.output_device = torch.device(getattr(source, "device", "cpu"))
        if self.dtype == torch.float64 and self.output_device.type == "mps":
            raise ValueError("precision must be 'fast' or 'float32' on MPS; MPS provides no float64")
        self.detector_shape = tuple(source.shape[1:])
        for component in trajectory:
            _fixed(component, "trajectory")
        _fixed(isocenter, "isocenter")
        geometry, self.n_views = _validate_trajectory(trajectory, beam)
        self.geometry_finfo = tuple(torch.finfo(component.dtype) for component in trajectory)
        self.geometry = tuple(component.to(device="cpu", dtype=self.dtype) for component in geometry)
        self.angles = _angles(angles, self.n_views, torch.device("cpu"))
        self.volume_shape = self.partition.local_volume_shape
        self.projection_shape = (self.partition.view_slice.stop - self.partition.view_slice.start,
                                 *self.detector_shape)
        if tuple(source.shape) != self.projection_shape:
            raise ValueError("sinogram shape must match the owned trajectory views")
        self.view_chunk_size = min(32, self.n_views) if view_chunk_size is None else _positive_int(
            view_chunk_size, "view_chunk_size")
        self.chunk_shape = (self.global_volume_shape if volume_chunk_shape is None else
                            _shape(volume_chunk_shape, self.rank, "volume_chunk_shape"))
        if isinstance(pad_factor, bool) or not isinstance(pad_factor, numbers.Integral) or pad_factor < 2:
            raise ValueError("pad_factor must be an integer >=2 for analytical reconstruction")
        self.pad_factor, self.window = int(pad_factor), window
        full_shape = (self.n_views, *self.detector_shape)
        self.projection_weights = _weight(projection_weights, full_shape, "projection_weights")
        self.angular_weights = _weight(angular_weights, (self.n_views,), "angular_weights", exact=True)
        self.filter_response = _weight(filter_response,
                                      (self.pad_factor * self.detector_shape[0] // 2 + 1,),
                                      "filter_response", exact=True)
        if filter_response is None:
            _ramp_window(window, torch.tensor([0.0], dtype=self.dtype))
        if self.angular_weights is None:
            relative = self.angles.double() - self.angles[0].double()
            self.angular_weights = angular_integration_weights(relative)
        if not isinstance(short_scan, bool):
            raise TypeError("short_scan must be a bool")
        if short_scan and beam == "parallel":
            raise ValueError("Parker short-scan weights apply to fan or cone beams, not parallel")
        self.short_scan = None
        self.isocenter, self.normal, self.sid, self.sdd = None, None, None, None
        if beam != "parallel":
            source_position, detector, u = self.geometry[:3]
            normal = (torch.stack((-u[:, 1], u[:, 0]), 1) if beam == "fan"
                      else torch.linalg.cross(u, self.geometry[3], dim=1))
            normal = F.normalize(normal, dim=1)
            self.normal = torch.where(((detector - source_position) * normal).sum(1)[:, None] < 0,
                                      -normal, normal)
            self.isocenter, self.sid, self.sdd = _analytical_geometry(
                source_position, detector, self.normal, isocenter)
        if short_scan:
            metadata = SimpleNamespace(ndim=self.rank, shape=full_shape)
            beta, distance, offset, sign = _parker_setup(
                self.angles, metadata, self.geometry, self.pitch, self.isocenter, self.geometry_finfo)
            ordered = beta.sort().values
            span, step = ordered[-1] - ordered[0], torch.median(ordered[1:] - ordered[:-1])
            self.short_scan = (beta, distance, offset, sign, float(span),
                               _angular_scan_period(span, step))
    def _admit_native(self, view_chunk_size, volume_chunk_shape):
        if self.backend == "numba":
            workers = self.devices or ((self.output_device,) if self.output_device.type == "cuda"
                                      else (torch.device("cuda", torch.cuda.current_device()),))
            budget = min(_memory_budget(device)[0] for device in workers)
            if self.partition.world_size > 1:
                device = (torch.device("cuda", torch.cuda.current_device())
                          if dist.get_backend(self.partition.process_group) == "nccl" else torch.device("cpu"))
                shared_budget = torch.tensor(budget, dtype=torch.int64, device=device)
                dist.all_reduce(shared_budget, op=dist.ReduceOp.MIN, group=self.partition.process_group)
                budget = int(shared_budget)
            ray_bytes = self.view_chunk_size * math.prod(self.detector_shape) * 4 * (self.pad_factor + 3)
            if volume_chunk_shape is None:
                parts = list(self.chunk_shape)
                while 3 * math.prod(parts) * 4 + ray_bytes > budget and max(parts) > 1:
                    axis = max(range(self.rank), key=parts.__getitem__)
                    parts[axis] = max(1, (parts[axis] + 1) // 2)
                self.chunk_shape = tuple(parts)
            if view_chunk_size is None:
                while 3 * math.prod(self.chunk_shape) * 4 + ray_bytes > budget and self.view_chunk_size > 1:
                    self.view_chunk_size = max(1, self.view_chunk_size // 2)
                    ray_bytes = self.view_chunk_size * math.prod(self.detector_shape) * 4 * (self.pad_factor + 3)
            if 3 * math.prod(self.chunk_shape) * 4 + ray_bytes > budget:
                raise MemoryError("analytical chunks exceed available CUDA working memory")

    def _workers(self):
        if self.devices is not None:
            return self.devices
        if self.backend == "numba" and self.output_device.type != "cuda":
            return (torch.device("cuda", torch.cuda.current_device()),)
        return (self.output_device,)

    def _view_batches(self):
        owned = self.partition.view_slice
        workers = self._workers()
        for index, start in enumerate(range(owned.start, owned.stop, self.view_chunk_size)):
            yield slice(start, min(start + self.view_chunk_size, owned.stop)), workers[index % len(workers)]

    def _parker_batch(self, views, device):
        beta, distance, offset, sign, span, period = self.short_scan
        shape = (views.stop - views.start, self.detector_shape[0])
        if period == 2 * math.pi:
            result = torch.ones(shape, device=device, dtype=self.dtype)
        else:
            u = (torch.arange(self.detector_shape[0], device=device, dtype=self.dtype)
                 - (self.detector_shape[0] - 1) / 2) * self.pitch[0] + offset
            gamma = torch.atan(u / distance)[None, :]
            delta = max(float(gamma.abs().max()), (span - math.pi) / 2)
            b = beta[views].to(device=device, dtype=self.dtype)[:, None]
            result = torch.ones(shape, device=device, dtype=self.dtype)
            result = torch.where(b < 2 * (delta - gamma),
                                 torch.sin(math.pi * .25 * b / (delta - gamma + 1e-12)).square(), result)
            result = torch.where(b > math.pi - 2 * gamma,
                                 torch.sin(math.pi * .25 * (math.pi + 2 * delta - b)
                                           / (delta + gamma + 1e-12)).square(), result)
            result = torch.where((b < 0) | (b > math.pi + 2 * delta), 0, result)
        if sign < 0:
            result = result.flip(1)
        return result[..., None] if self.beam == "cone" else result

    def _filtered(self, views, device):
        local = slice(views.start - self.partition.view_slice.start,
                      views.stop - self.partition.view_slice.start)
        selected = (local, *(slice(None) for _ in self.detector_shape))
        data = (self.source[selected] if isinstance(self.source, torch.Tensor) else self.source.read(selected))
        data = data.to(device=device, dtype=self.dtype)
        geometry = tuple(component[views].to(device=device) for component in self.geometry)
        if self.projection_weights is None:
            weighted = data if self.beam == "parallel" else data * _cosine(data, geometry, self.pitch)
        else:
            weights = self.projection_weights.broadcast_to((self.n_views, *self.detector_shape))[views]
            weighted = data * weights.to(device=device, dtype=self.dtype)
        if self.short_scan is not None:
            weighted = weighted * self._parker_batch(views, device)
        response = None if self.filter_response is None else self.filter_response.to(device=device, dtype=self.dtype)
        filtered = _filter(weighted, self.pitch[0], self.window, self.pad_factor, response)
        angular = self.angular_weights[views].to(device=device, dtype=self.dtype)
        return filtered * angular.reshape((-1,) + (1,) * (self.rank - 1)), geometry

    def _portable_gather(self, data, geometry, views, spatial):
        axes = [(torch.arange(part.start + owned.start, part.stop + owned.start,
                              device=data.device, dtype=self.dtype) + .5 - size / 2) * pitch
                for part, owned, size, pitch in zip(spatial, self.partition.volume_slice,
                                                    self.global_volume_shape, self.spacing)]
        positions = torch.stack(torch.meshgrid(*axes, indexing="ij")[::-1], -1).reshape(-1, self.rank)
        result = torch.zeros(positions.shape[0], device=data.device, dtype=self.dtype)
        if self.beam != "parallel":
            normal = self.normal[views].to(data.device)
            sid, sdd = self.sid[views].to(data.device), self.sdd[views].to(data.device)
        for view in range(len(data)):
            if self.beam == "parallel":
                hit = positions - geometry[1][view]
                factor = 1 / (2 * math.pi)
                valid = torch.ones_like(result, dtype=torch.bool)
            else:
                vector = positions - geometry[0][view]
                distance = (vector * normal[view]).sum(1)
                valid = distance > 1e-6
                safe_distance = torch.where(valid, distance, 1)
                hit = geometry[0][view] + (sdd[view] / safe_distance)[:, None] * vector - geometry[1][view]
                factor = (sid[view] / safe_distance).square() * sdd[view] / (2 * math.pi * sid[view])
            fu = (hit * geometry[2][view]).sum(1) / self.pitch[0] + (data.shape[1] - 1) / 2
            valid = valid & (fu >= 0) & (fu <= data.shape[1] - 1)
            iu = fu.floor().long().clamp(0, data.shape[1] - 2)
            tu = (fu - iu).clamp(0, 1)
            if self.beam == "cone":
                fv = (hit * geometry[3][view]).sum(1) / self.pitch[1] + (data.shape[2] - 1) / 2
                valid = valid & (fv >= 0) & (fv <= data.shape[2] - 1)
                iv = fv.floor().long().clamp(0, data.shape[2] - 2)
                tv = (fv - iv).clamp(0, 1)
                sample = ((1 - tu) * ((1 - tv) * data[view, iu, iv] + tv * data[view, iu, iv + 1])
                          + tu * ((1 - tv) * data[view, iu + 1, iv] + tv * data[view, iu + 1, iv + 1]))
            else:
                sample = (1 - tu) * data[view, iu] + tu * data[view, iu + 1]
            result = result + torch.where(valid, sample * factor, 0)
        return result.reshape(tuple(part.stop - part.start for part in spatial))

    def _native_gather(self, data, geometry, spatial):
        shape = tuple(part.stop - part.start for part in spatial)
        centre = torch.tensor([(part.start + owned.start + size / 2 - full / 2) * pitch
                               for part, owned, size, full, pitch in zip(spatial, self.partition.volume_slice,
                                                                         shape, self.global_volume_shape,
                                                                         self.spacing)][::-1],
                              device=data.device, dtype=self.dtype)
        if self.beam == "parallel":
            geometry = (geometry[0], geometry[1] - centre, geometry[2])
            return parallel_weighted_backproject(data, *geometry, self.pitch[0], *shape, self.voxel_spacing)
        geometry = (geometry[0] - centre, geometry[1] - centre, *geometry[2:])
        isocenter = self.isocenter.to(data.device) - centre
        if self.beam == "fan":
            return fan_weighted_backproject(data, *geometry, self.pitch[0], *shape,
                                            self.voxel_spacing, isocenter=isocenter)
        return cone_weighted_backproject(data, *geometry, *shape, *self.pitch,
                                         self.voxel_spacing, isocenter=isocenter)

    def _tile(self, spatial):
        shape = tuple(part.stop - part.start for part in spatial)
        value = torch.zeros(shape, device=self.output_device, dtype=self.dtype)
        if self.input_dependency is not None:
            value = value + self.input_dependency
        for views, device in self._view_batches():
            filtered, geometry = self._filtered(views, device)
            partial = (self._native_gather(filtered, geometry, spatial) if self.backend == "numba"
                       else self._portable_gather(filtered, geometry, views, spatial))
            value = value + partial.to(self.output_device)
        return _sum(value, self.partition.view_group, self.partition.view_world_size)

    def _prepare_gradients(self):
        partition = self.partition
        if partition.world_size == 1 or not torch.is_grad_enabled():
            return
        options = (self.projection_weights, self.angular_weights, self.filter_response)
        schema = tuple((tuple(value.shape), str(value.dtype), value.requires_grad)
                       if isinstance(value, torch.Tensor) else None for value in options)
        schemas = [None] * partition.world_size
        dist.all_gather_object(schemas, schema, group=partition.process_group)
        if any(item != schema for item in schemas):
            raise ValueError("distributed ranks must agree on analytical weight shapes, dtypes and gradient flags")
        device = (torch.device("cuda", torch.cuda.current_device())
                  if dist.get_backend(partition.process_group) == "nccl" else torch.device("cpu"))
        flag = int(self.source.requires_grad or any(value is not None and value.requires_grad for value in options))
        needs_grad = torch.tensor(flag, dtype=torch.int64, device=device)
        dist.all_reduce(needs_grad, op=dist.ReduceOp.MAX, group=partition.process_group)
        if not int(needs_grad):
            return
        self.participation = torch.zeros((), device=self.output_device, dtype=self.dtype, requires_grad=True)
        limit = self.view_chunk_size * math.prod(self.detector_shape)
        names = [name for name in ("projection_weights", "angular_weights", "filter_response")
                 if getattr(self, name) is not None and getattr(self, name).requires_grad]
        self.source, *parameters = _AnalyticalInputs.apply(
            self.participation, partition, limit, self.source, *(getattr(self, name) for name in names))
        for name, value in zip(names, parameters):
            setattr(self, name, value)

    def tensor_result(self):
        self._prepare_gradients()
        result = torch.zeros(self.volume_shape, device=self.output_device, dtype=self.dtype)
        inputs = (self.source, self.projection_weights, self.angular_weights, self.filter_response)
        active_gradients = torch.is_grad_enabled() and any(value is not None and value.requires_grad
                                                         for value in inputs)
        if self.partition.world_size == 1 and not active_gradients:
            # Each tile still accumulates view batches in their original order.
            # The numerical result itself holds the partial sums between batches.
            for views, device in self._view_batches():
                filtered, geometry = self._filtered(views, device)
                for spatial, _ in _pixel_tiles(self.volume_shape, self.chunk_shape):
                    partial = (self._native_gather(filtered, geometry, spatial) if self.backend == "numba"
                               else self._portable_gather(filtered, geometry, views, spatial))
                    result[spatial].add_(partial.to(self.output_device))
                    del partial
                del filtered, geometry
            return result
        # Every owned tile's backward collective precedes the bundled input
        # transpose, even when this rank owns no data views. Empty space groups
        # have no tile collectives but still contribute zero input cotangents.
        self.input_dependency = _zero_dependency(self.source).to(device=self.output_device, dtype=self.dtype)
        for option in (self.projection_weights, self.angular_weights, self.filter_response):
            if option is not None and option.requires_grad:
                self.input_dependency = self.input_dependency + _zero_dependency(option).to(
                    device=self.output_device, dtype=self.dtype)
        result = result + self.input_dependency
        for spatial, _ in _pixel_tiles(self.volume_shape, self.chunk_shape):
            result[spatial] = self._tile(spatial)
        return result

    def into(self, output):
        original = output
        error, source, sink = None, None, None
        try:
            source, sink = _validate_stores(self, self.source, output, False)
            for option in (self.projection_weights, self.angular_weights, self.filter_response):
                if isinstance(option, torch.Tensor) and option.requires_grad:
                    raise ValueError("into methods are numerical; detach autograd weights")
        except (ValueError, TypeError, RuntimeError, OSError) as caught:
            error = caught
        self.partition.preflight(error)
        self.source = source
        with torch.no_grad():
            for spatial, _ in _pixel_tiles(self.volume_shape, self.chunk_shape):
                sink.write(spatial, self._tile(spatial))
            sink.flush()
        return original


def reconstruct(source, trajectory, volume_shape, angles, **kwargs):
    if not isinstance(source, torch.Tensor):
        raise TypeError("fbp/fdk input must be a tensor; use fbp_into/fdk_into for stores")
    return AnalyticalExecution(source, trajectory, volume_shape, angles, **kwargs).tensor_result()


def reconstruct_into(source, output, trajectory, volume_shape, angles, **kwargs):
    return AnalyticalExecution(source, trajectory, volume_shape, angles, **kwargs).into(output)
