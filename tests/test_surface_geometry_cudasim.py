"""Run fused geometry staging and its real kernel with CPU transport storage."""

import importlib
import math
from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from numba.core import config
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from diffct import operators
from diffct.chunking import _ChunkPlan
from diffct.surfaces import _GeometrySpec
from tests.test_detector_surfaces import _case, _projector
from tests.test_surface_pipeline_contracts import _world


pytestmark = pytest.mark.skipif(not config.ENABLE_CUDASIM,
                                reason="CPU kernel integration requires NUMBA_ENABLE_CUDASIM=1")


class _CudaOffsetMetadata(torch.Tensor):
    """Exercise the accelerator-input branch while retaining CPU test storage."""

    @property
    def device(self):
        return torch.device("cuda:0")

    @property
    def is_cuda(self):
        return True


class _CpuGeometryPipe:
    def __init__(self):
        self.device = torch.device("cuda:0")
        self.waits, self.synchronizations, self.allocations = [], [], []
        self.upload = SimpleNamespace(wait_stream=lambda stream: self.waits.append(stream))
        self.stats = dict(h2d_bytes=0, d2h_bytes=0, p2p_bytes=0)

    def empty(self, shape, *, dtype=torch.float32):
        value = torch.empty(shape, dtype=dtype)
        self.allocations.append(value)
        return value

    def event(self, stream):
        return SimpleNamespace(synchronize=lambda: self.synchronizations.append(stream))

    def retain(self, event, source, target):
        pass


class _NoWorldAllocation(TorchDispatchMode):
    def __init__(self, inputs, state, world_shape):
        super().__init__()
        self.existing = {value.untyped_storage()._cdata for value in inputs}
        self.state, self.world_shape = state, world_shape

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        for value in tree_flatten(result)[0]:
            if not isinstance(value, torch.Tensor):
                continue
            storage = value.untyped_storage()._cdata
            if self.state["sampling"]:
                self.existing.add(storage)
            elif storage not in self.existing and value.dtype == torch.float64:
                assert tuple(value.shape) != self.world_shape, \
                    ("fused staging materialized a complete FP64 world batch", str(function))
                assert value.numel() < math.prod(self.world_shape), \
                    ("fused staging allocated a complete FP64 world buffer", str(function), value.shape)
        return result


class _KernelEntries:
    """Observe each production dispatcher launch, then run it unchanged."""

    def __init__(self, kernel):
        self.kernel, self.records = kernel, []

    def __getitem__(self, configuration):
        launch = self.kernel[configuration]

        def run(*args):
            frames, offsets, source, points, view, start, count, capacity = args[:8]
            self.records.append(dict(frames=frames.copy(), source=source, points=points,
                                     view=view, start=start, count=count, capacity=capacity))
            return launch(*args)

        return run


def _cpu_geometry_transport(entries):
    module = importlib.import_module("diffct.kernels.surface_geometry")
    patches = ExitStack()
    patches.enter_context(mock.patch.object(torch.cuda, "device", side_effect=lambda *args: nullcontext()))
    patches.enter_context(mock.patch.object(torch.cuda, "stream", side_effect=lambda *args: nullcontext()))
    patches.enter_context(mock.patch.object(torch.cuda, "current_stream", return_value=object()))
    patches.enter_context(mock.patch.object(torch.cuda, "get_allocator_backend", return_value="native"))
    patches.enter_context(mock.patch.object(operators, "_get_numba_external_stream_for", return_value=0))
    patches.enter_context(mock.patch.object(operators.TorchCUDABridge, "tensor_to_cuda_array",
                                           side_effect=lambda value: value.detach().numpy()))
    patches.enter_context(mock.patch.object(module, "compose_surface", entries))
    return patches


def _offset_values(detector, pitch, views, pixels, per_view, dtype):
    axes = [(np.arange(selected.start, selected.stop) + .5 - size / 2) * spacing
            for size, spacing, selected in zip(detector, pitch, pixels)]
    u, v = np.meshgrid(*axes, indexing="ij") if len(axes) == 2 else (axes[0], np.zeros_like(axes[0]))
    gain = np.arange(views.start, views.stop).reshape(-1, *([1] * u.ndim)) * .125 if per_view else 0.
    tangent = u + 1. + gain
    vertical = np.broadcast_to(v, tangent.shape)
    depth = np.broadcast_to(-(2 ** 24) + .25 + .03125 * u, tangent.shape)
    return np.stack((tangent, vertical, depth), -1).astype(dtype)


@pytest.mark.parametrize("beam,kind,per_view,offset_origin,layout", [
    ("cone", "offsets", False, "cpu", "double-offsets"),
    ("cone", "offsets", True, "cpu", "double-source"),
    ("cone", "offsets", True, "cuda-metadata", "double-vertical"),
    ("cone", "sampled", False, "cuda-metadata", "double-center"),
    ("fan", "offsets", True, "cpu", "double-offsets"),
    ("fan", "sampled", True, "cuda-metadata", "double-source"),
    ("parallel", "offsets", False, "cpu", "double-center"),
    ("parallel", "sampled", False, "cuda-metadata", "double-source"),
])
def test_fused_helper_runs_real_kernel_for_global_batches_into_reused_bounded_targets(
    beam, kind, per_view, offset_origin, layout,
):
    case = _case(beam, views=7)
    rank = len(case.shape)
    case.detector = (11, 13) if beam == "cone" else (29,)
    frames = [torch.zeros((7, rank), dtype=torch.float32) for _ in case.trajectory]
    frames[0][:, 0] = 1. if beam == "parallel" else -2.
    frames[1][:, 0], frames[1][:, 1], frames[2][:, 1] = 2 ** 24, 2 ** 24, 1.
    if beam == "cone":
        frames[3][:, 2] = 1.
    double_index = {"double-center": 1, "double-source": 0, "double-vertical": 3}.get(layout)
    if double_index is not None:
        frames[double_index] = frames[double_index].double()
    if layout == "double-center":
        frames[1][:, 0] += .25
    case.trajectory = tuple(frames)
    projector = _projector(case, backend="torch")
    offset_dtype = np.float64 if layout == "double-offsets" else np.float32
    state, samples = {"sampling": False}, []

    def sampler(u, v, ids):
        state["sampling"] = True
        try:
            assert u.device.type == v.device.type == ids.device.type == "cpu"
            samples.append((u.numpy().copy(), v.numpy().copy(), tuple(ids.tolist())))
            gain = ids.reshape(-1, *([1] * u.ndim)) * .125 if per_view else 0.
            tangent = u + 1. + gain
            result = torch.stack((tangent, v.expand_as(tangent),
                                  (-(2 ** 24) + .25 + .03125 * u).expand_as(tangent)), -1)
            result = result.to(torch.float64 if offset_dtype == np.float64 else torch.float32)
            return result.as_subclass(_CudaOffsetMetadata) if offset_origin == "cuda-metadata" else result
        finally:
            state["sampling"] = False

    spec = _GeometrySpec(kind, sampler if kind == "sampled" else None)
    geometry = case.trajectory
    if kind == "offsets":
        full = tuple(slice(0, size) for size in case.detector)
        values = _offset_values(case.detector, case.pitch, slice(0, 7), full, per_view, offset_dtype)
        offsets = torch.from_numpy(values)
        if offset_origin == "cuda-metadata":
            offsets = offsets.as_subclass(_CudaOffsetMetadata)
        geometry += (offsets,)
    first_pixels = (slice(2, 7), slice(3, 8)) if beam == "cone" else (slice(2, 27),)
    tail_pixels = (slice(8, 11), slice(8, 13)) if beam == "cone" else (slice(26, 29),)
    batches = [(slice(2, 4), first_pixels), (slice(6, 7), tail_pixels)]
    centre = (.1875, -.3125, .4375)[:rank]
    source_buffer, point_buffer = torch.empty(2 * rank), torch.empty(2 * 25 * rank)
    original_storage = (source_buffer.data_ptr(), point_buffer.data_ptr())
    groups, pipe = {}, _CpuGeometryPipe()
    kernel = importlib.import_module("diffct.kernels.surface_geometry").compose_surface
    entries = _KernelEntries(kernel)
    with _cpu_geometry_transport(entries):
        for views, pixels in batches:
            count = views.stop - views.start
            shape = (count, *(selected.stop - selected.start for selected in pixels), rank)
            source_buffer.fill_(77.)
            point_buffer.fill_(77.)
            targets = (source_buffer[:count * rank], point_buffer[:math.prod(shape)])
            before = len(entries.records)
            with _NoWorldAllocation((*geometry, source_buffer, point_buffer), state, (2, 5, 5, rank) if rank == 3 else (2, 25, rank)):
                ready = projector._surface_geometry_into(geometry, spec, views, pixels, centre,
                                                        targets, pipe, groups)
            assert ready is not None and pipe.synchronizations
            local_offsets = _offset_values(case.detector, case.pitch, views, pixels, per_view, offset_dtype)
            if not per_view:
                local_offsets = np.broadcast_to(local_offsets, (count, *local_offsets.shape))
            values = [value[views].numpy().copy() for value in case.trajectory]
            if kind == "sampled":
                common = np.result_type(np.float32, *(value.dtype for value in values), local_offsets.dtype)
                values, local_offsets = [value.astype(common) for value in values], local_offsets.astype(common)
            expected = _world(values, local_offsets, beam)
            expected = (expected - np.asarray(centre, dtype=expected.dtype)).astype(np.float32)
            expected_source = values[0].astype(np.promote_types(np.float32, values[0].dtype))
            if beam != "parallel":
                expected_source -= np.asarray(centre, dtype=expected_source.dtype)
            np.testing.assert_array_equal(targets[0].view(count, rank).numpy(), expected_source.astype(np.float32))
            np.testing.assert_array_equal(targets[1].view(shape).numpy(), expected)
            assert torch.all(source_buffer[count * rank:] == 77.)
            assert torch.all(point_buffer[math.prod(shape):] == 77.)
            assert (source_buffer.data_ptr(), point_buffer.data_ptr()) == original_storage
            observed = entries.records[before:]
            assert observed and {record["view"] for record in observed} == set(range(count))
            for record in observed:
                assert np.shares_memory(record["source"], source_buffer.numpy())
                assert np.shares_memory(record["points"], point_buffer.numpy())
                assert record["count"] <= record["capacity"]
                assert record["start"] + record["count"] <= math.prod(shape[1:-1])
                frame_values = [value[views.start + record["view"]].numpy() for value in case.trajectory]
                original_prefix = np.concatenate([value.astype(np.float64) for value in frame_values])
                np.testing.assert_array_equal(record["frames"][:original_prefix.size], original_prefix)
                assert record["frames"].nbytes <= 512, "packed frame arena exceeded one native allocation unit"
                if rank == 3:
                    normal_dtype = np.result_type(np.float32, frame_values[2].dtype, frame_values[3].dtype)
                    if kind == "sampled":
                        normal_dtype = np.result_type(normal_dtype, *(value.dtype for value in frame_values), local_offsets.dtype)
                    normal = np.cross(frame_values[2].astype(normal_dtype), frame_values[3].astype(normal_dtype))
                    np.testing.assert_array_equal(record["frames"][12:15], normal.astype(np.float64))
    assert len(pipe.allocations) == 2, "packed workspaces were allocated again for the tail batch"
    workspace_bytes = sum(math.ceil(value.numel() * value.element_size() / 512) * 512 for value in pipe.allocations)
    target_bytes = sum(math.ceil(value.numel() * value.element_size() / 512) * 512 for value in (source_buffer, point_buffer))
    assert workspace_bytes <= target_bytes, "packed staging exceeded the native geometry buffer bytes"
    assert any(record["count"] < record["capacity"] for record in entries.records)
    assert len(samples) == (2 if kind == "sampled" else 0), "surface sampler ran more than once per native batch"
    if kind == "sampled":
        assert [sample[2] for sample in samples] == [(2, 3), (6,)]
        for sample, (_, pixels) in zip(samples, batches):
            axes = [(np.arange(selected.start, selected.stop) + .5 - size / 2) * pitch
                    for size, pitch, selected in zip(case.detector, case.pitch, pixels)]
            expected_u, expected_v = np.meshgrid(*axes, indexing="ij") if rank == 3 else (axes[0], np.zeros_like(axes[0]))
            np.testing.assert_array_equal(sample[0], expected_u)
            np.testing.assert_array_equal(sample[1], expected_v)
    assert bool(pipe.waits) == (offset_origin == "cuda-metadata")


@pytest.mark.parametrize("failure,message", [
    ("nonfinite", "finite"), ("coincident", "must differ"),
    ("near", "1e6"), ("far", "1e15"),
])
def test_fused_kernel_status_is_synchronized_before_ray_job_can_launch_or_continue(failure, message):
    case = _case("fan", views=3)
    projector = _projector(case, backend="torch")
    frames = (torch.tensor([[-2., 0.]], dtype=torch.float64).repeat(3, 1),
              torch.tensor([[3., 0.]], dtype=torch.float64).repeat(3, 1),
              torch.tensor([[0., 1.]], dtype=torch.float32).repeat(3, 1))
    offsets = torch.zeros((2, 3, 3), dtype=torch.float64)
    if failure == "nonfinite":
        offsets[1, -1, 0] = float("nan")
    elif failure == "coincident":
        frames[0][2].zero_()
        frames[1][2].zero_()
    elif failure == "near":
        frames[0][2, 0], frames[1][2, 0] = 1_000_000.01, 1_000_002.01
    else:
        frames[0][2, 0], frames[1][2, 0] = 1e15 + .25, 1.
    sampled = []

    def sampler(u, v, ids):
        sampled.append(tuple(ids.tolist()))
        return offsets

    spec = _GeometrySpec("sampled", sampler)
    plan = _ChunkPlan(case.shape, 2, (torch.device("cuda:0"),), detector_chunk_shape=(3,))
    groups, pipe = projector._groups_for(plan, spec), _CpuGeometryPipe()
    entries = _KernelEntries(importlib.import_module("diffct.kernels.surface_geometry").compose_surface)
    continuations = []
    with _cpu_geometry_transport(entries), \
            mock.patch.object(operators, "_project_into", wraps=operators._project_into) as ray_kernel:
        with pytest.raises(ValueError, match=message):
            projector._ray_job(None, torch.ones(case.shape), frames, (.1875, -.3125),
                               slice(0, 2), slice(1, 3), (slice(2, 5),), spec, pipe, groups, 0, True,
                               before_launch=lambda rays: continuations.append("continued"))
        assert entries.records and sampled == [(1, 2)]
        assert pipe.synchronizations, "host consumed validation status before its upload event completed"
        assert not continuations and ray_kernel.call_count == 0
        workspaces = tuple(value.data_ptr() for value in pipe.allocations)
        frames[0][:, 0], frames[1][:, 0] = -2., 3.
        offsets.zero_()
        targets = groups[0]["geometry"]
        projector._surface_geometry_into(frames, spec, slice(1, 3), (slice(2, 5),),
                                         (.1875, -.3125), targets, pipe, groups)
        assert tuple(value.data_ptr() for value in pipe.allocations) == workspaces


def test_cuda_offsets_capture_producer_before_upload_and_retain_each_copy_until_its_event():
    case = _case("fan", views=3)
    projector = _projector(case, backend="torch")
    producer = object()
    active = {"stream": producer}
    queries, copies, retained, events = [], [], [], []
    pipe = _CpuGeometryPipe()

    def wait_stream(stream):
        assert active["stream"] is pipe.upload
        assert stream is producer, "offset upload waited on itself instead of its producer"
        pipe.waits.append(stream)

    pipe.upload.wait_stream = wait_stream

    @contextmanager
    def stream_context(stream):
        previous = active["stream"]
        active["stream"] = stream
        try:
            yield
        finally:
            active["stream"] = previous

    def current_stream(device):
        queries.append(active["stream"])
        return active["stream"]

    def event(stream):
        result = SimpleNamespace(stream=stream, complete=False)

        def synchronize():
            assert copies, "validation completion preceded every offset copy"
            for copy in copies:
                owners = [entry for entry in retained if entry[0] == copy[0]]
                assert owners, "offset storage was not retained through its copy event"
                assert all(not entry[1].complete for entry in owners)
                assert all(entry[2].untyped_storage()._cdata == copy[1] for entry in owners)
            for previous in events:
                previous.complete = True
            pipe.synchronizations.append(stream)

        result.synchronize = synchronize
        events.append(result)
        return result

    def retain(completion, source, target):
        assert completion.stream is pipe.upload and not completion.complete
        if source.is_cuda:
            assert copies and copies[-1][1] == source.untyped_storage()._cdata
            retained.append((copies[-1][0], completion, source, target))

    pipe.event, pipe.retain = event, retain
    original_copy = operators._pipeline._copy

    def copy(target, source, stats):
        if source.is_cuda:
            assert active["stream"] is pipe.upload
            assert pipe.waits and all(stream is producer for stream in pipe.waits), \
                "offset copy started before waiting for its original producer"
            copies.append((len(copies), source.untyped_storage()._cdata))
        return original_copy(target, source, stats)

    def sampler(u, v, ids):
        assert active["stream"] is producer
        return torch.stack((u, v, .125 * u.square()), -1).as_subclass(_CudaOffsetMetadata)

    targets = (torch.empty((2, 2)), torch.empty((2, 4, 2)))
    entries = _KernelEntries(importlib.import_module("diffct.kernels.surface_geometry").compose_surface)
    centre = (.1875, -.3125)
    with _cpu_geometry_transport(entries), \
            mock.patch.object(torch.cuda, "stream", side_effect=stream_context), \
            mock.patch.object(torch.cuda, "current_stream", side_effect=current_stream), \
            mock.patch.object(operators._pipeline, "_copy", side_effect=copy):
        projector._surface_geometry_into(case.trajectory, _GeometrySpec("sampled", sampler),
                                         slice(1, 3), (slice(2, 6),), centre, targets, pipe, {})
    assert queries == [producer], "producer stream was queried after switching to the upload stream"
    assert copies and len(retained) == len(copies)
    assert all(entry[1].complete for entry in retained)
    assert entries.records and active["stream"] is producer
    u = (np.arange(2, 6) + .5 - case.detector[0] / 2) * case.pitch[0]
    offsets = np.broadcast_to(np.stack((u, np.zeros_like(u), .125 * u ** 2), -1), (2, 4, 3))
    frames = tuple(value[1:3].numpy().copy() for value in case.trajectory)
    expected = (_world(frames, offsets, "fan") - np.asarray(centre)).astype(np.float32)
    np.testing.assert_array_equal(targets[0].numpy(), (frames[0] - np.asarray(centre)).astype(np.float32))
    np.testing.assert_array_equal(targets[1].numpy(), expected)


@pytest.mark.parametrize("offset_origin", ["cpu", "cuda-metadata"])
@pytest.mark.parametrize("per_view", [False, True], ids=["shared", "per-view"])
def test_noncontiguous_sampler_offsets_reach_kernel_without_complete_offset_copy(offset_origin, per_view):
    case = _case("cone", views=4)
    case.detector = (13, 17)
    projector = _projector(case, backend="torch")
    views, pixels = slice(1, 3), (slice(3, 10), slice(4, 15))
    axes = [(np.arange(selected.start, selected.stop) + .5 - size / 2) * pitch
            for size, pitch, selected in zip(case.detector, case.pitch, pixels)]
    u, v = np.meshgrid(*axes, indexing="ij")
    variation = np.arange(1, 3).reshape(2, 1, 1) * .125 if per_view else 0.
    tangent = u + variation
    values = np.stack((tangent, np.broadcast_to(v, tangent.shape),
                       np.broadcast_to(.0625 * u ** 2, tangent.shape)), -1)
    offsets = torch.from_numpy(values).transpose(-3, -2).contiguous().transpose(-3, -2)
    if offset_origin == "cuda-metadata":
        offsets = offsets.as_subclass(_CudaOffsetMetadata)
    assert not offsets.is_contiguous()
    samples = []

    def sampler(actual_u, actual_v, ids):
        samples.append(tuple(ids.tolist()))
        np.testing.assert_array_equal(actual_u.numpy(), u)
        np.testing.assert_array_equal(actual_v.numpy(), v)
        return offsets

    targets = (torch.empty((2, 3)), torch.empty((2, 7, 11, 3)))
    original_storages = {value.untyped_storage()._cdata for value in (*case.trajectory, offsets, *targets)}
    complete_view_bytes = 7 * 11 * 3 * offsets.element_size()
    allocations = []

    class NoCompleteOffsetCopy(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            result = function(*args, **(kwargs or {}))
            for value in tree_flatten(result)[0]:
                if not isinstance(value, torch.Tensor) or value.untyped_storage()._cdata in original_storages:
                    continue
                size = value.untyped_storage().nbytes()
                allocations.append(size)
                assert size < complete_view_bytes, (
                    "noncontiguous offsets created a complete view copy outside caller-owned targets",
                    str(function), tuple(value.shape), size, complete_view_bytes,
                )
            return result

    entries = _KernelEntries(importlib.import_module("diffct.kernels.surface_geometry").compose_surface)
    centre, pipe = (.1875, -.3125, .4375), _CpuGeometryPipe()
    with _cpu_geometry_transport(entries), NoCompleteOffsetCopy():
        projector._surface_geometry_into(case.trajectory, _GeometrySpec("sampled", sampler),
                                         views, pixels, centre, targets, pipe, {})
    assert samples == [(1, 2)] and entries.records and allocations
    assert max(allocations) < complete_view_bytes
    local = values if per_view else np.broadcast_to(values, (2, *values.shape))
    frames = tuple(value[views].numpy().copy() for value in case.trajectory)
    expected = (_world(frames, local, "cone") - np.asarray(centre)).astype(np.float32)
    np.testing.assert_array_equal(targets[0].numpy(), (frames[0] - np.asarray(centre)).astype(np.float32))
    np.testing.assert_array_equal(targets[1].numpy(), expected)
    assert bool(pipe.waits) == (offset_origin == "cuda-metadata")


def test_fused_helper_uses_framework_fp32_cross_before_deep_fp64_cancellation():
    case = _case("cone", views=1)
    case.detector = (1, 1)
    projector = _projector(case, backend="torch")
    axis_u = torch.tensor([[-.7559716702, .3963489830, -.5209743977]], dtype=torch.float32)
    axis_v = torch.tensor([[-.1834404022, -.8922420740, -.4126181304]], dtype=torch.float32)
    normal = torch.linalg.cross(axis_u, axis_v)
    depth = 1e7
    target = np.asarray([[.125, -.25, .375]], dtype=np.float64)
    center = torch.from_numpy(-depth * normal.double().numpy() + target)
    source = torch.tensor([[-1., -2., -3.]], dtype=torch.float32)
    frames = (source, center, axis_u, axis_v)
    offsets = torch.tensor([[[.03125, -.0625, depth]]], dtype=torch.float64)
    targets = (torch.empty((1, 3)), torch.empty((1, 1, 1, 3)))
    entries = _KernelEntries(importlib.import_module("diffct.kernels.surface_geometry").compose_surface)
    centre, pipe = (.0625, -.125, .25), _CpuGeometryPipe()
    with _cpu_geometry_transport(entries):
        projector._surface_geometry_into((*frames, offsets), _GeometrySpec("offsets"),
                                         slice(0, 1), (slice(0, 1), slice(0, 1)), centre,
                                         targets, pipe, {})
    # Large depth exposes normal rounding before the FP64 cancellation.
    expected = center.numpy() + .03125 * axis_u.double().numpy()
    expected = expected - .0625 * axis_v.double().numpy() + depth * normal.double().numpy()
    expected = (expected - np.asarray(centre)).astype(np.float32)
    np.testing.assert_array_equal(targets[1].numpy().reshape(1, 3), expected)
    np.testing.assert_array_equal(targets[0].numpy(),
                                  (source.numpy() - np.asarray(centre, dtype=np.float32)).astype(np.float32))
    assert entries.records
    np.testing.assert_array_equal(entries.records[0]["frames"][12:15], normal.double().numpy()[0])
    assert all(value.numel() * value.element_size() <= 512 for value in pipe.allocations)
