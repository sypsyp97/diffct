"""CPU evidence for native geometry staging, snapshots, and pilot ownership."""

from collections import Counter
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from diffct import ParameterizedSurface, operators
from diffct.chunking import _ChunkPlan
from diffct.surfaces import _GeometrySpec, _sample_geometry
from tests.test_detector_surfaces import _case, _numpy, _projector


def _world(frames, offsets, beam):
    """Compose the caller's quantized values, retaining each arithmetic dtype."""
    dimensions = offsets.ndim - 2
    shape = (frames[0].shape[0], *([1] * dimensions), frames[0].shape[-1])
    source, center, axis = (value.astype(np.promote_types(np.float32, value.dtype)).reshape(shape)
                            for value in frames[:3])
    result = center + offsets[..., 0, None] * axis
    if beam == "cone":
        vertical = frames[3].astype(np.promote_types(np.float32, frames[3].dtype)).reshape(shape)
        result = result + offsets[..., 1, None] * vertical + offsets[..., 2, None] * np.cross(axis, vertical)
    else:
        normal = np.stack((axis[..., 1], -axis[..., 0]), axis=-1)
        result = result + offsets[..., 2, None] * normal
    return result


class _ReachedUpload(Exception):
    pass


@pytest.mark.parametrize("beam", ["parallel", "fan", "cone"])
@pytest.mark.parametrize("kind", ["flat-frames", "flat-pixels", "explicit-points"])
def test_flat_and_explicit_geometry_stays_cpu_until_reusable_pipeline_upload(beam, kind):
    case = _case(beam, views=5)
    pixel_shape = (3, 2) if beam == "cone" else (3,)
    partial = kind != "flat-frames"
    projector = _projector(case, backend="torch", detector_chunk_shape=pixel_shape if partial else None)
    selected = tuple(slice(1, 1 + size) for size in pixel_shape) if partial else tuple(
        slice(0, size) for size in case.detector)
    view = slice(1, 3)
    center = (.19, -.23, .31) if beam == "cone" else (.19, -.23)
    original = tuple(value.detach().numpy().copy() for value in case.trajectory)
    geometry, spec = case.trajectory, None
    if kind == "explicit-points":
        axes = [(np.arange(size) + .5 - size / 2) * pitch for size, pitch in zip(case.detector, case.pitch)]
        u, v = np.meshgrid(*axes, indexing="ij") if beam == "cone" else (axes[0], np.zeros_like(axes[0]))
        offsets = np.broadcast_to(np.stack((u, v, .07 * u ** 2), -1), (*case.sino_shape, 3))
        world = _world(original, offsets, beam)
        geometry = (case.trajectory[0], torch.from_numpy(world))
        spec = _GeometrySpec("points")
        expected = [original[0][view], world[(view, *selected)]]
    elif partial:
        axes = [(np.arange(item.start, item.stop) + .5 - size / 2) * pitch
                for size, pitch, item in zip(case.detector, case.pitch, selected)]
        u, v = np.meshgrid(*axes, indexing="ij") if beam == "cone" else (axes[0], np.zeros_like(axes[0]))
        offsets = np.broadcast_to(np.stack((u, v, np.zeros_like(u)), -1), (2, *pixel_shape, 3))
        expected = [original[0][view], _world(tuple(value[view] for value in original), offsets, beam)]
    else:
        expected = [value[view] for value in original]
    expected = [value - np.asarray(center) if index == 1 or (index == 0 and beam != "parallel") else value
                for index, value in enumerate(expected)]
    plan = _ChunkPlan(case.shape, 2, (torch.device("cuda:0"),),
                      detector_chunk_shape=tuple(item.stop - item.start for item in selected))
    groups = projector._groups_for(plan, spec)
    records = []

    def upload(source, target):
        assert source.device.type == "cpu", "geometry bypassed reusable pipeline upload"
        assert target.dtype == torch.float32
        records.append((source.detach().numpy().copy(), target.untyped_storage()._cdata))
        if len(records) % len(expected) == 0:
            raise _ReachedUpload

    pipe = SimpleNamespace(device=torch.device("cuda:0"), empty=torch.empty, upload_tensor=upload)
    with mock.patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("geometry initialized CUDA before upload")):
        for _ in range(2):
            with pytest.raises(_ReachedUpload):
                projector._ray_job(None, None, geometry, center, slice(0, 2), view, selected,
                                   spec, pipe, groups, 0, True)
    for actual, reference in zip(records, expected * 2):
        np.testing.assert_allclose(actual[0], reference, rtol=0, atol=0)
    assert [record[1] for record in records[:len(expected)]] == [
        record[1] for record in records[len(expected):]], "geometry targets were allocated again"


class _SnapshotCopies(TorchDispatchMode):
    def __init__(self, originals):
        super().__init__()
        self.origins = {value.untyped_storage()._cdata: index for index, value in enumerate(originals)}
        self.counts = Counter()

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        if args and isinstance(args[0], torch.Tensor) and str(function) in (
                "aten.clone.default", "aten._to_copy.default"):
            index = self.origins.get(args[0].untyped_storage()._cdata)
            if index is not None:
                assert result.device.type == "cpu", "snapshot retained accelerator storage"
                self.counts[index] += 1
                self.origins[result.untyped_storage()._cdata] = index
        return result


@pytest.mark.parametrize("kind", ["flat", "legacy", "parameterized"])
@pytest.mark.parametrize("origin", ["cpu", "cuda"])
def test_geometry_snapshots_copy_each_original_once_per_call_and_keep_gradients(kind, origin):
    if origin == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA originals require CUDA storage")
    case = _case("cone", views=5, learnable=True)
    case.trajectory = tuple(value.detach().to(origin).requires_grad_() for value in case.trajectory)
    parameter = torch.tensor(.17, dtype=torch.float64, device=origin, requires_grad=True)
    offsets = torch.arange(np.prod(case.detector) * 3, dtype=torch.float64, device=origin).reshape(
        *case.detector, 3).requires_grad_()
    surface = None
    extra = ()
    if kind == "legacy":
        surface, extra = (lambda u, v: offsets), (offsets,)
    elif kind == "parameterized":
        surface = ParameterizedSurface(lambda u, v, ids, gain: torch.stack((u, v, gain * u), -1), (parameter,))
        extra = (parameter,)
    projector = _projector(case, surface, backend="torch")
    originals = (*case.trajectory, *extra)
    plan = _ChunkPlan(case.shape, 2, (torch.device("cuda:0"),), detector_chunk_shape=(3, 2))
    for invocation in range(2):
        expected = tuple(_numpy(value) for value in originals)
        with _SnapshotCopies(originals) as trace:
            _, snapshots = projector._effective_geometry(plan, torch.device("cpu"))
        assert trace.counts == Counter({index: 1 for index in range(len(originals))}), \
            "snapshot count depends on device, batch count, or detector pixels"
        assert len(snapshots) == len(originals)
        for value, original, reference in zip(snapshots, originals, expected):
            assert value.device.type == "cpu" and value.dtype == original.dtype
            assert value.untyped_storage()._cdata != original.untyped_storage()._cdata
            np.testing.assert_array_equal(value.detach().numpy(), reference)
        with torch.no_grad():
            for original in originals:
                original.add_(.031 * (invocation + 1))
        gradients = torch.autograd.grad(sum(value.sum() for value in snapshots), originals)
        for value, gradient, reference in zip(snapshots, gradients, expected):
            np.testing.assert_array_equal(value.detach().numpy(), reference)
            torch.testing.assert_close(gradient, torch.ones_like(gradient), rtol=0, atol=0)


@pytest.mark.parametrize("schedule", ["spatial", "views", "window"])
@pytest.mark.parametrize("is_project", [True, False])
def test_schedule_pilot_samples_only_representative_devices_owned_view_shard(schedule, is_project):
    case = _case("cone", views=12)
    samples = []

    def sampler(u, v, ids):
        samples.append(tuple(ids.tolist()))
        assert set(ids.tolist()) <= {5, 6, 7}, "pilot sampled views owned by another compute device"
        return torch.stack((u, v, .09 * u.square()), -1)

    projector = _projector(case, backend="torch")
    projector.view_slice = slice(5, 12)
    projector.projection_shape = (7, *case.detector)
    spec = _GeometrySpec("sampled", sampler)
    plan = _ChunkPlan((2, 3, 4), 4, tuple(torch.device("cuda", index) for index in range(3)),
                      detector_chunk_shape=(3, 2), schedule=schedule, window_size=2)
    pipe = SimpleNamespace(device=plan.devices[0], compute=None, event=lambda stream: None,
                           drain_tensor=lambda *args, **kwargs: None)
    transfer = mock.MagicMock()
    transfer.__enter__.return_value.for_device.return_value = pipe
    calls = []

    def ray_job(source, native, geometry, center, local, global_views, pixels, spec, pipe, groups,
                index, is_project, **kwargs):
        calls.append((local, global_views))
        assert 0 <= local.start < local.stop <= 3
        source, points = projector._tile_geometry(geometry, global_views, center, torch.device("cpu"), pixels, spec)
        buffers = {"done": None}
        return points.sum(-1).float(), None, buffers, buffers

    with mock.patch.object(operators._pipeline, "_TransferPipeline", return_value=transfer), \
            mock.patch.object(projector, "_native_tile", side_effect=lambda source, selected, shape, pipe, project: torch.ones(shape)), \
            mock.patch.object(projector, "_ray_job", side_effect=ray_job), \
            mock.patch.object(torch.cuda, "stream", side_effect=lambda stream: nullcontext()), \
            mock.patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("pilot fixture initialized CUDA")):
        report = projector._pilot_candidate(SimpleNamespace(device=torch.device("cpu")),
                                            SimpleNamespace(device=torch.device("cpu")), is_project,
                                            case.trajectory, plan, spec)
    assert calls and samples and set().union(*map(set, samples)) == {5, 6, 7}
    assert all(global_views.start == 5 + local.start and global_views.stop == 5 + local.stop
               for local, global_views in calls)
    assert report["representative_device"] == 0


class _LocalStorage(TorchDispatchMode):
    def __init__(self, inputs, maximum):
        super().__init__()
        self.existing = {value.untyped_storage()._cdata for value in inputs}
        self.maximum, self.peak = maximum, 0

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        for value in tree_flatten(result)[0]:
            if isinstance(value, torch.Tensor) and value.untyped_storage()._cdata not in self.existing:
                size = value.untyped_storage().nbytes()
                self.peak = max(self.peak, size)
                assert size <= self.maximum, ("world composition allocated outside its local batch", str(function), size)
        return result


@pytest.mark.parametrize("detector", [(7, 9), (173, 179)])
@pytest.mark.parametrize("layout", ["double-offsets", "double-center", "double-source", "double-vertical"])
def test_native_surface_composition_preserves_promoted_order_and_local_gradient_storage(detector, layout):
    case = _case("cone", views=5)
    case.detector = detector
    frames = [torch.zeros((5, 3), dtype=torch.float32) for _ in range(4)]
    frames[0][:, 0] = -2.
    frames[1][:, 0], frames[1][:, 1] = 2 ** 24, 2 ** 24
    frames[2][:, 1], frames[3][:, 2] = 1., 1.
    index = {"double-center": 1, "double-source": 0, "double-vertical": 3}.get(layout)
    if index is not None:
        frames[index] = frames[index].double()
    offset_dtype = torch.float64 if layout == "double-offsets" else torch.float32
    offsets = torch.zeros((5, *detector, 3), dtype=offset_dtype)
    offsets[..., 0], offsets[..., 1], offsets[..., 2] = 1., .25, -(2 ** 24)
    if layout == "double-offsets":
        offsets[..., 2] += .25
    elif layout == "double-center":
        frames[1][:, 0] += .25
    geometry = tuple(value.requires_grad_() for value in (*frames, offsets))
    projector = _projector(case, backend="torch")
    views, pixels = slice(1, 3), (slice(2, 4), slice(3, 6))
    spec = _GeometrySpec("offsets")
    inputs, _ = projector._surface_batch_inputs(geometry, spec, views, pixels)
    values = [value.detach().numpy().copy() for value in inputs]
    expected = _world(values[:4], values[4], "cone")
    weight = np.arange(1, expected.size + 1, dtype=np.float64).reshape(expected.shape)
    maximum = 2 * 2 * 3 * 3 * 8
    with _LocalStorage(inputs, maximum) as trace:
        source, points = _sample_geometry("cone", detector, case.pitch, inputs, spec, views, pixels,
                                          case.spacing, local_geometry=True, native_legacy=True, validate=False)
        gradients = torch.autograd.grad((points * torch.from_numpy(weight).to(points.dtype)).sum() + source.sum(), inputs)
    assert points.dtype == torch.from_numpy(expected).dtype
    np.testing.assert_array_equal(points.detach().numpy(), expected)
    np.testing.assert_array_equal(source.detach().numpy(), values[0])
    assert trace.peak <= maximum
    u, v = values[2][:, None, None], values[3][:, None, None]
    offset = values[4].astype(np.float64)
    expected_gradients = [np.ones_like(values[0]), weight.sum((1, 2)),
                          (offset[..., 0, None] * weight + offset[..., 2, None] * np.cross(v, weight)).sum((1, 2)),
                          (offset[..., 1, None] * weight + offset[..., 2, None] * np.cross(weight, u)).sum((1, 2)),
                          np.stack(((weight * u).sum(-1), (weight * v).sum(-1),
                                    (weight * np.cross(u, v)).sum(-1)), -1)]
    for gradient, original, reference in zip(gradients, inputs, expected_gradients):
        assert gradient.shape == original.shape and gradient.dtype == original.dtype
        np.testing.assert_allclose(gradient.detach().numpy(), reference.astype(gradient.detach().numpy().dtype), rtol=1e-6, atol=0)
