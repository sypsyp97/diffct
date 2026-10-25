"""Explicit detector samplers and bounded, global-pixel geometry preparation."""

from dataclasses import dataclass

import torch

from .utils import _raise_geometry_errors, _work_dtype

@dataclass(frozen=True)
class _GeometrySpec:
    kind: str
    sampler: object = None
    dtype: object = None
    # One (U, V, 3) host offset snapshot shared by all views.
    shared: bool = False


class ParameterizedSurface:
    """Sample local detector offsets from global CPU pixel/view coordinates.

    ``sampler(u, v, view_indices, *parameters)`` returns shared or per-view
    ``(..., 3)`` offsets on CPU or CUDA. Parameters are explicit finite floating tensors;
    other geometry configuration must be immutable. Projector snapshots the
    sampler reference and parameter values per invocation. Snapshot storage
    scales with parameter sizes, and the sampler's own allocations remain
    caller-controlled.
    """

    def __init__(self, sampler, parameters=()):
        if not callable(sampler):
            raise TypeError("surface sampler must be callable")
        self.sampler = sampler
        self.parameters = tuple(parameters)
        for parameter in self.parameters:
            if not isinstance(parameter, torch.Tensor) or not torch.is_floating_point(parameter):
                raise TypeError("surface parameters must be floating-point tensors")
            if not torch.isfinite(parameter).all().item():
                raise ValueError("surface parameters must be finite")

    def __call__(self, u, v, view_indices):
        return self.sampler(u, v, view_indices, *self.parameters)


def _detector_grids(detector_shape, spacing, pixels):
    axes = [(torch.arange(selected.start, selected.stop, selected.step or 1,
                          dtype=torch.float64, device="cpu") + .5 - size / 2) * pitch
            for selected, size, pitch in zip(pixels, detector_shape, spacing)]
    if len(axes) == 2:
        return torch.meshgrid(*axes, indexing="ij")
    return axes[0], torch.zeros_like(axes[0])


def _sample_offsets(beam, detector_shape, spacing, geometry, spec, views, pixels,
                    *, validate=True, local_geometry=False):
    """Sample one offset batch without expanding world coordinates."""
    frame_count = 4 if beam in ("cone", "parallel3d") else 3
    frames = geometry[:frame_count]
    dimensions = len(detector_shape)
    view_count = (views.stop - views.start if local_geometry
                  else len(range(*views.indices(frames[0].shape[0]))))
    pixel_shape = tuple(len(range(*selected.indices(size)))
                        for selected, size in zip(pixels, detector_shape))
    if spec is not None and spec.kind == "offsets":
        complete_offsets = geometry[frame_count]
        selection = pixels if complete_offsets.ndim == dimensions + 1 else (views, *pixels)
        offsets = complete_offsets if local_geometry else complete_offsets[selection]
    else:
        u, v = _detector_grids(detector_shape, spacing, pixels)
        ids = torch.arange(views.start, views.stop, dtype=torch.int64, device="cpu")
        offsets = (spec.sampler(u, v, ids, *geometry[frame_count:])
                   if spec is not None and spec.kind == "sampled"
                   else torch.stack((u, v, torch.zeros_like(u)), dim=-1))
    if not isinstance(offsets, torch.Tensor) or not torch.is_floating_point(offsets):
        raise TypeError("detector surface must return a floating-point tensor")
    shared_shape = (*pixel_shape, 3)
    if tuple(offsets.shape) not in (shared_shape, (view_count, *shared_shape)):
        raise ValueError("detector surface shape must match the selected view/pixel batch")
    if validate and beam not in ("cone", "parallel3d") and torch.any(offsets[..., 1] != 0).item():
        raise ValueError("2D detector surface middle coordinates must be zero")
    return offsets


def _sample_geometry(beam, detector_shape, spacing, geometry, spec, views, pixels, voxel_spacing,
                     *, device=None, dtype=None, validate=True, local_geometry=False,
                     native_legacy=False):
    """Build one selected world-point batch on the requested compute device."""
    frame_count = 4 if beam in ("cone", "parallel3d") else 3
    frames = geometry[:frame_count]
    device = torch.device("cpu") if device is None else torch.device(device)
    dimensions = len(detector_shape)
    view_count = (views.stop - views.start if local_geometry
                  else len(range(*views.indices(frames[0].shape[0]))))
    offsets = _sample_offsets(beam, detector_shape, spacing, geometry, spec, views, pixels,
                              validate=validate, local_geometry=local_geometry)
    if dtype is None:
        dtype = torch.float32
        for component in (*frames, offsets):
            dtype = torch.promote_types(dtype, component.dtype)
    if dtype.itemsize > _work_dtype(device).itemsize:
        # CPU-side FP64 geometry must not follow FP64-less compute devices.
        dtype = _work_dtype(device)
    elif dtype not in (torch.float32, torch.float64):
        raise ValueError("surface geometry dtype must be float32 or float64")
    offset_dtype = offsets.dtype if native_legacy else dtype
    if beam in ("cone", "parallel3d"):
        offsets = offsets.to(device=device, dtype=offset_dtype)
        local = offsets if offsets.ndim == dimensions + 2 else offsets.unsqueeze(0)
        tangent, vertical, depth = local.unbind(-1)
    else:
        tangent = offsets[..., 0].to(device=device, dtype=offset_dtype)
        depth = offsets[..., 2].to(device=device, dtype=offset_dtype)
        if offsets.ndim == dimensions + 1:
            tangent, depth = tangent.unsqueeze(0), depth.unsqueeze(0)
    checks = [(~torch.isfinite(offsets).all(), "detector surface coordinates must be finite")] if validate else []

    def frame(component):
        selected = component if local_geometry else component[views]
        frame_dtype = torch.promote_types(torch.float32, component.dtype) if native_legacy else dtype
        return selected.to(device=device, dtype=frame_dtype).reshape(
            view_count, *([1] * dimensions), component.shape[-1],
        )

    center, axis_u = frame(frames[1]), frame(frames[2])
    points = center + tangent[..., None] * axis_u
    if beam in ("cone", "parallel3d"):
        axis_v = frame(frames[3])
        normal_dtype = torch.promote_types(axis_u.dtype, axis_v.dtype)
        normal = torch.linalg.cross(axis_u.to(dtype=normal_dtype),
                                    axis_v.to(dtype=normal_dtype), dim=-1)
        points = points + vertical[..., None] * axis_v + depth[..., None] * normal
    else:
        normal = torch.stack((axis_u[..., 1], -axis_u[..., 0]), dim=-1)
        points = points + depth[..., None] * normal
    source_dtype = torch.promote_types(torch.float32, frames[0].dtype) if native_legacy else dtype
    source = (frames[0] if local_geometry else frames[0][views]).to(device=device, dtype=source_dtype)
    if validate:
        for component in (source, points):
            checks.append((~torch.isfinite(component).all() | ~torch.isfinite(component.float()).all(),
                           "surface world coordinates must be finite in float32"))
        if beam not in ("parallel", "parallel3d") and view_count:
            sources = source.unsqueeze(1)
            endpoints = points.reshape(view_count, -1, source.shape[-1])
            checks.append((torch.any(torch.all(sources.float() == endpoints.float(), dim=-1)),
                           "source and detector surface pixels must differ"))
            work = _work_dtype(sources.device)
            if isinstance(voxel_spacing, (tuple, list)):
                pitches = sources.new_tensor(voxel_spacing[::-1], dtype=work)
                source_distance = torch.linalg.vector_norm(sources.to(dtype=work) / pitches, dim=-1)
                pixel_distance = torch.linalg.vector_norm(endpoints.to(dtype=work) / pitches, dim=-1)
            else:
                source_distance = torch.linalg.vector_norm(sources.to(dtype=work), dim=-1) / voxel_spacing
                pixel_distance = torch.linalg.vector_norm(endpoints.to(dtype=work), dim=-1) / voxel_spacing
            checks.append((torch.any(torch.minimum(source_distance, pixel_distance) > 1e6),
                           "surface source or pixel must lie within 1e6 voxels"))
            checks.append((torch.any(source_distance > 1e15) | torch.any(pixel_distance > 1e15),
                           "surface coordinates must lie within 1e15 voxels"))
        _raise_geometry_errors(checks)
    return source, points
