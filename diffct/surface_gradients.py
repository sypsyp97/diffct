"""Bounded replay of surface composition for native geometry cotangents."""

import operator

import torch

from .surfaces import _GeometrySpec, _sample_geometry


def surface_batch_vjp(beam, detector_shape, detector_spacing, inputs, offsets,
                      views, pixels, voxel_spacing, device, weights, capacity,
                      *, native_legacy):
    """Pull back native cotangents through one sampled surface batch.

    ``inputs`` contains local CPU frame leaves and sampler parameter leaves.
    ``offsets`` retains the sampler graph. Replay uses one view and one detector
    row segment at a time; ``capacity`` limits each segment's pixel count.
    The returned gradients use CPU storage and each input's original dtype.
    """
    if isinstance(capacity, bool):
        raise ValueError("surface replay capacity must be a positive integer")
    try:
        capacity = operator.index(capacity)
    except TypeError as error:
        raise ValueError("surface replay capacity must be a positive integer") from error
    if capacity < 1:
        raise ValueError("surface replay capacity must be a positive integer")

    frame_count = 4 if beam in ("cone", "parallel3d") else 3
    frames = inputs[:frame_count]
    dimensions = len(detector_shape)
    pixel_shape = tuple(len(range(*selected.indices(size)))
                        for selected, size in zip(pixels, detector_shape))
    per_view = offsets.ndim == dimensions + 2
    common_dtype = torch.float32
    for component in (*frames, offsets):
        common_dtype = torch.promote_types(common_dtype, component.dtype)
    frame_dtypes = tuple(torch.promote_types(torch.float32, value.dtype)
                         if native_legacy else common_dtype for value in frames)
    offset_dtype = (torch.promote_types(torch.float32, offsets.dtype)
                    if native_legacy else common_dtype)
    frame_grads = [torch.zeros(value.shape, dtype=dtype, device="cpu")
                   for value, dtype in zip(frames, frame_dtypes)]
    offset_cotangent = torch.zeros(offsets.shape, dtype=offset_dtype, device="cpu")
    offset_spec = _GeometrySpec("offsets")

    # Each rectangle has one row. A wide row is split into bounded segments.
    rows = range(pixel_shape[0]) if dimensions == 2 else range(1)
    width = pixel_shape[-1]
    with torch.enable_grad():
        for view in range(frames[0].shape[0]):
            first_fragment = True
            for row in rows:
                for start in range(0, width, capacity):
                    stop = min(start + capacity, width)
                    local_pixels = ((slice(row, row + 1), slice(start, stop))
                                    if dimensions == 2 else (slice(start, stop),))
                    global_pixels = tuple(slice(selected.start + local.start,
                                                selected.start + local.stop)
                                          for selected, local in zip(pixels, local_pixels))
                    offset_selection = ((slice(view, view + 1), *local_pixels)
                                        if per_view else local_pixels)
                    replay_frames = tuple(value[view:view + 1].detach().to(dtype=dtype).requires_grad_()
                                          for value, dtype in zip(frames, frame_dtypes))
                    replay_offsets = offsets[offset_selection].detach().to(
                        device="cpu", dtype=offset_dtype,
                    ).requires_grad_()
                    replay_inputs = (*replay_frames, replay_offsets)
                    with torch.autograd.graph.save_on_cpu():
                        source, points = _sample_geometry(
                            beam, detector_shape, detector_spacing, replay_inputs,
                            offset_spec, slice(views.start + view, views.start + view + 1),
                            global_pixels, voxel_spacing, device=device,
                            dtype=common_dtype, validate=False, local_geometry=True,
                            native_legacy=native_legacy,
                        )
                    point_weight = weights[1][(slice(view, view + 1), *local_pixels)].detach().to(
                        device=points.device, dtype=points.dtype,
                    )
                    if first_fragment:
                        source_weight = weights[0][view:view + 1].detach().to(
                            device=source.device, dtype=source.dtype,
                        )
                        parts = torch.autograd.grad(
                            (source, points), replay_inputs, (source_weight, point_weight),
                            allow_unused=True,
                        )
                        del source_weight
                    else:
                        parts = torch.autograd.grad(points, replay_inputs, point_weight,
                                                    allow_unused=True)
                    for gradient, part in zip(frame_grads, parts[:frame_count]):
                        if part is not None:
                            gradient[view:view + 1].add_(part.detach().to(device="cpu"))
                    if parts[-1] is not None:
                        offset_cotangent[offset_selection].add_(parts[-1].detach().to(device="cpu"))
                    first_fragment = False
                    # Do not keep a fragment's device graph across the next replay.
                    del source, points, point_weight, parts, replay_inputs
                    del replay_frames, replay_offsets

        sampler_grads = (torch.autograd.grad(
            offsets, inputs,
            offset_cotangent.to(device=offsets.device, dtype=offsets.dtype),
            allow_unused=True,
        ) if offsets.requires_grad else (None,) * len(inputs))

    result = []
    for index, (value, part) in enumerate(zip(inputs, sampler_grads)):
        if index < frame_count:
            gradient = frame_grads[index]
            if part is not None:
                gradient.add_(part.detach().to(device="cpu", dtype=gradient.dtype))
            result.append(gradient.to(dtype=value.dtype))
        else:
            result.append(None if part is None else part.detach().to(device="cpu", dtype=value.dtype))
    return tuple(result)
