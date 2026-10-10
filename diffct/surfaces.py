"""Explicit detector samplers and bounded, global-pixel geometry preparation."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class _GeometrySpec:
    kind: str
    sampler: object = None


class ParameterizedSurface:
    """Sample local detector offsets from global CPU pixel/view coordinates.

    ``sampler(u, v, view_indices, *parameters)`` returns shared or per-view
    ``(..., 3)`` offsets. Parameters are explicit finite floating tensors;
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
    axes = [(torch.arange(selected.start, selected.stop, dtype=torch.float64, device="cpu") + .5 - size / 2) * pitch
            for selected, size, pitch in zip(pixels, detector_shape, spacing)]
    if len(axes) == 2:
        return torch.meshgrid(*axes, indexing="ij")
    return axes[0], torch.zeros_like(axes[0])


def _sample_geometry(beam, detector_shape, spacing, geometry, spec, views, pixels, voxel_spacing):
    """Build one CPU world-point batch; preserve its small sampler/frame graph."""
    frame_count = 4 if beam == "cone" else 3
    frames = geometry[:frame_count]
    u, v = _detector_grids(detector_shape, spacing, pixels)
    ids = torch.arange(views.start, views.stop, dtype=torch.int64, device="cpu")
    offsets = (spec.sampler(u, v, ids, *geometry[frame_count:])
               if spec is not None and spec.kind == "sampled"
               else torch.stack((u, v, torch.zeros_like(u)), dim=-1))
    if not isinstance(offsets, torch.Tensor) or not torch.is_floating_point(offsets):
        raise TypeError("detector surface must return a floating-point tensor")
    if offsets.device.type != "cpu":
        raise TypeError("streamed detector surface coordinates must be on CPU")
    shared_shape = (*u.shape, 3)
    if tuple(offsets.shape) not in (shared_shape, (len(ids), *shared_shape)):
        raise ValueError("detector surface shape must match the selected view/pixel batch")
    if not torch.isfinite(offsets).all().item():
        raise ValueError("detector surface coordinates must be finite")
    if beam != "cone" and torch.any(offsets[..., 1] != 0).item():
        raise ValueError("2D detector surface middle coordinates must be zero")
    local = offsets if offsets.ndim == u.ndim + 2 else offsets.unsqueeze(0)

    def frame(component):
        return component[views].to(device="cpu", dtype=torch.promote_types(component.dtype, offsets.dtype)).reshape(
            len(ids), *([1] * u.ndim), component.shape[-1],
        )

    center, axis_u = frame(frames[1]), frame(frames[2])
    points = center + local[..., 0, None] * axis_u
    if beam == "cone":
        axis_v = frame(frames[3])
        normal = torch.linalg.cross(axis_u, axis_v, dim=-1)
        points = points + local[..., 1, None] * axis_v + local[..., 2, None] * normal
    else:
        normal = torch.stack((axis_u[..., 1], -axis_u[..., 0]), dim=-1)
        points = points + local[..., 2, None] * normal
    source = frames[0][views].to(device="cpu")
    for component in (source, points):
        if not torch.isfinite(component).all().item() or not torch.isfinite(component.float()).all().item():
            raise ValueError("surface world coordinates must be finite in float32")
    if beam != "parallel":
        sources = source.double().unsqueeze(1)
        endpoints = points.reshape(len(ids), -1, source.shape[-1]).double()
        if torch.any(torch.all(sources.float() == endpoints.float(), dim=-1)).item():
            raise ValueError("surface source and detector coordinates must differ")
        source_distance = torch.linalg.vector_norm(sources, dim=-1) / voxel_spacing
        pixel_distance = torch.linalg.vector_norm(endpoints, dim=-1) / voxel_spacing
        if torch.any(torch.minimum(source_distance, pixel_distance) > 1e6).item():
            raise ValueError("surface source or pixel must lie within 1e6 voxels")
        if torch.any(source_distance > 1e15).item() or torch.any(pixel_distance > 1e15).item():
            raise ValueError("surface coordinates must lie within 1e15 voxels")
    return source, points
