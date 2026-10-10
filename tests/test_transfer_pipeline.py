"""Real stream/event lifetimes and independent host/GPU capacity limits."""

from contextlib import ExitStack
from datetime import timedelta
import gc
import math
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils._pytree import tree_flatten

import diffct.pipeline as pipeline
from diffct.utils import TorchCUDABridge
from tests.test_block_storage import _GuardedStore, _rejection, _slices
from tests.test_chunked_projector import _memory_inventory, _offsets
from tests.test_detector_surfaces import _close, _data, _matrix, _numpy
from tests.test_execution_schedules import (
    _CopyTrace, _check_report, _schedule_case, _scheduled_projector, _stream, cuda_required,
)


def _ceilings(*, slot=512, scratch=512, geometry=8192):
    patches = ExitStack()
    for name, value in (("_HOST_SLOT_BYTES", slot), ("_HOST_SCRATCH_BYTES", scratch),
                        ("_GEOMETRY_WORK_BYTES", geometry)):
        assert hasattr(pipeline, name), f"INTERFACE_PENDING: diffct.pipeline.{name}"
        patches.enter_context(mock.patch.object(pipeline, name, value))
    return patches


class _FragmentStore(_GuardedStore):
    def __init__(self, tensor, maximum_bytes, *, writable=True):
        super().__init__(tensor, tuple(tensor.shape), writable=writable)
        self.maximum_bytes = maximum_bytes

    def read(self, index):
        value = self.tensor[_slices(index, self.shape)]
        assert value.numel() * value.element_size() <= self.maximum_bytes, \
            "pageable read fragment exceeded its independent source-dtype ceiling"
        return super().read(index)

    def write(self, index, value, *, accumulate=False):
        assert value.numel() * value.element_size() <= self.maximum_bytes, \
            "host drain fragment exceeded its pinned slot ceiling"
        return super().write(index, value, accumulate=accumulate)


class _PipelineTrace(_CopyTrace):
    """Observe actual dependencies; inserted probe events never unblock work."""
    def __init__(self, case, *, caller_tensors=(), delay=False, host_fragments=False):
        super().__init__(case)
        self.caller_cpu = {tensor.untyped_storage()._cdata for tensor in caller_tensors if not tensor.is_cuda}
        self.storage_refs, self.pinned_peak = {}, 0
        self.event_records, self.waits, self.event_syncs, self.global_syncs = {}, [], [], []
        self.uploads, self.downloads, self.gpu_reads = {}, {}, {}
        self.producers = {}
        self.kernel_done, self.probe_events, self.recorded_owners = [], [], set()
        self.h2d_while_kernel_pending = 0
        self.delay = delay
        self.host_fragments = host_fragments
        self._original_record = torch.cuda.Event.record

    @staticmethod
    def _key(tensor):
        return tensor.device, tensor.untyped_storage()._cdata

    def _track(self, tensor):
        key = self._key(tensor)
        previous = self.storage_refs.get(key)
        if previous is None or torch.UntypedStorage._expired(previous["weak"]):
            if previous is not None:
                torch.UntypedStorage._free_weak_ref(previous["weak"])
            storage = tensor.untyped_storage()
            self.storage_refs[key] = {"weak": storage._weak_ref(), "bytes": storage.nbytes(),
                                      "pinned": tensor.device.type == "cpu" and tensor.is_pinned(),
                                      "allocation_stream": _stream() if tensor.is_cuda else None}
        if tensor.device.type == "cpu" and key[1] not in self.caller_cpu:
            limit = max(pipeline._HOST_SLOT_BYTES, pipeline._HOST_SCRATCH_BYTES, pipeline._GEOMETRY_WORK_BYTES)
            assert tensor.untyped_storage().nbytes() <= limit, "large fresh host allocation escaped independent ceilings"
            if self.host_fragments and tuple(tensor.shape) in (self.case.shape, self.case.shape[::-1], self.case.sino_shape):
                assert tensor.untyped_storage().nbytes() <= max(pipeline._HOST_SLOT_BYTES, pipeline._HOST_SCRATCH_BYTES), \
                    "complete host input/output/layout temporary escaped fragmented transfers"
        live = sum(record["bytes"] for record in self.storage_refs.values()
                   if record["pinned"] and not torch.UntypedStorage._expired(record["weak"]))
        self.pinned_peak = max(self.pinned_peak, live)
        assert live <= 2 * pipeline._HOST_SLOT_BYTES, "more than two composite pinned slots are live"

    def _probe_event(self):
        event = torch.cuda.Event()
        self._original_record(event, torch.cuda.current_stream())
        self.probe_events.append(event)
        return event

    def _depends(self, target_stream, target_sequence, source_stream, source_sequence):
        pending, seen = [(target_stream, target_sequence)], set()
        while pending:
            stream, before = pending.pop()
            if (stream, before) in seen:
                continue
            seen.add((stream, before))
            if stream == source_stream and before > source_sequence:
                return True
            for wait in self.waits:
                if wait["stream"] == stream and wait["sequence"] < before:
                    recorded = wait["record"]
                    pending.append((recorded["stream"], recorded["sequence"]))
        return False

    def _gpu_dependency(self, key, sequence):
        for record in (*self.uploads.get(key, ()), *self.downloads.get(key, ()), *self.producers.get(key, ())):
            if not record["event"].query():
                assert self._depends(_stream(), sequence, record["stream"], record["sequence"]), \
                    "GPU buffer read/overwrite lacks the producer/copy event dependency"

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        for key, records in self.gpu_reads.items():
            lifetime = self.storage_refs.get(key)
            if lifetime is not None and torch.UntypedStorage._expired(lifetime["weak"]):
                for record in records:
                    if not record["event"].query() and record["stream"] != lifetime["allocation_stream"]:
                        assert (key, record["stream"]) in self.recorded_owners, \
                            "cross-stream native input freed without allocator ownership or final-event retention"
        name = str(function)
        inputs = [value for value in tree_flatten(args)[0] if isinstance(value, torch.Tensor)]
        mutates = name in {"aten.copy_.default", "aten.zero_.default", "aten.fill_.Scalar", "aten.add_.Tensor"}
        if mutates and inputs:
            target = inputs[0]
            if target.device.type == "cpu":
                for record in self.uploads.get(self._key(target), ()):
                    assert record["event"].query(), "pinned host input overwritten before H2D consumed it"
            else:
                self._gpu_dependency(self._key(target), self.sequence + 1)
        source = None
        if name == "aten.copy_.default" and inputs[0].device.type == "cpu" and inputs[1].is_cuda:
            source = inputs[1]
        elif name == "aten._to_copy.default" and inputs[0].is_cuda and \
                torch.device((kwargs or {}).get("device", inputs[0].device)).type == "cpu":
            source = inputs[0]
        if source is not None:
            self._gpu_dependency(self._key(source), self.sequence + 1)
        views = {"aten.detach.default", "aten.slice.Tensor", "aten.select.int", "aten.view.default",
                 "aten._unsafe_view.default", "aten.as_strided.default", "aten.permute.default",
                 "aten.transpose.int", "aten.alias.default"}
        if name not in views:
            # Shape/view operations may happen before completion; arithmetic or
            # a host copy must observe completed D2H data.
            for tensor in inputs[1:] if mutates else inputs:
                if tensor.device.type == "cpu":
                    for record in self.downloads.get(self._key(tensor), ()):
                        assert record["event"].query(), "CPU output read preceded D2H completion"
        before = len(self.copies)
        result = super().__torch_dispatch__(function, types, args, kwargs)
        for tensor in (*inputs, *tree_flatten(result)[0]):
            if isinstance(tensor, torch.Tensor):
                self._track(tensor)
        if len(self.copies) > before:
            record = self.copies[-1]
            event = self._probe_event()
            tracked = {"sequence": record["sequence"], "stream": record["stream"], "event": event}
            source_key = record["source"]["device"], record["source"]["storage"]
            target_key = record["target"]["device"], record["target"]["storage"]
            if record["direction"] == "h2d":
                self.uploads.setdefault(source_key, []).append(tracked)
                self.uploads.setdefault(target_key, []).append(tracked)
                self.h2d_while_kernel_pending += any(not event.query() for event in self.kernel_done)
            elif record["direction"] == "d2h":
                self.downloads.setdefault(source_key, []).append(tracked)
                self.downloads.setdefault(target_key, []).append(tracked)
        return result

    def __enter__(self):
        super().__enter__()
        tracer = self
        original_record, original_wait = torch.cuda.Event.record, torch.cuda.Event.wait
        original_sync, original_global = torch.cuda.Event.synchronize, torch.cuda.synchronize
        original_owner, original_bridge = torch.Tensor.record_stream, TorchCUDABridge.tensor_to_cuda_array

        def record(event, stream=None):
            stream = stream or torch.cuda.current_stream()
            result = original_record(event, stream)
            tracer.event_records[event.cuda_event] = {"sequence": tracer._step(),
                                                       "stream": (stream.device.index, int(stream.cuda_stream))}
            return result

        def wait(event, stream=None):
            stream = stream or torch.cuda.current_stream()
            result = original_wait(event, stream)
            if event.cuda_event in tracer.event_records:
                tracer.waits.append({"sequence": tracer._step(),
                                     "stream": (stream.device.index, int(stream.cuda_stream)),
                                     "record": dict(tracer.event_records[event.cuda_event])})
            return result

        def synchronize(event):
            result = original_sync(event)
            tracer.event_syncs.append(tracer._step())
            return result

        def global_synchronize(device=None):
            result = original_global(device)
            tracer.global_syncs.append(tracer._step())
            return result

        def owner(tensor, stream):
            tracer.recorded_owners.add((tracer._key(tensor), (stream.device.index, int(stream.cuda_stream))))
            return original_owner(tensor, stream)

        def bridge(tensor):
            tracer._track(tensor)
            return original_bridge(tensor)

        for target, attribute, replacement in (
                (torch.cuda.Event, "record", record), (torch.cuda.Event, "wait", wait),
                (torch.cuda.Event, "synchronize", synchronize), (torch.cuda, "synchronize", global_synchronize),
                (torch.Tensor, "record_stream", owner), (TorchCUDABridge, "tensor_to_cuda_array", bridge)):
            self.patches.enter_context(mock.patch.object(target, attribute, replacement))
        return self

    def _proxy(self, kernel, beam, kind):
        parent = super()._proxy(kernel, beam, kind)
        tracer = self

        class Proxy:
            def __getitem__(self, configuration):
                launch = parent[configuration]

                def observed(*args, **kwargs):
                    owners = set()
                    for array in args:
                        if hasattr(array, "__cuda_array_interface__"):
                            pointer = array.__cuda_array_interface__["data"][0]
                            key = torch.device("cuda", torch.cuda.current_device()), pointer
                            metadata = tracer.bridged.get(key)
                            if metadata is not None:
                                owner = metadata["device"], metadata["storage"]
                                owners.add(owner)
                                tracer._gpu_dependency(owner, tracer.sequence + 1)
                    if tracer.delay:
                        torch.cuda._sleep(50000000)
                    result = launch(*args, **kwargs)
                    event = tracer._probe_event()
                    tracer.kernel_done.append(event)
                    produced = tracer.launches[-1]["rays" if kind == "forward" else "volume"]
                    producer_key = produced["device"], produced["storage"]
                    tracer.producers.setdefault(producer_key, []).append({"event": event, "stream": _stream(),
                                                                          "sequence": tracer.launches[-1]["sequence"]})
                    for owner in owners:
                        tracer.gpu_reads.setdefault(owner, []).append({"event": event, "stream": _stream()})
                    return result
                return observed
        return Proxy()

    def __exit__(self, *exception):
        try:
            return super().__exit__(*exception)
        finally:
            for record in self.storage_refs.values():
                torch.UntypedStorage._free_weak_ref(record["weak"])


def test_private_host_ceilings_are_independent_positive_memory_policies():
    assert pipeline._HOST_SLOT_BYTES == pipeline._HOST_SCRATCH_BYTES == pipeline._GEOMETRY_WORK_BYTES == 8 * 1024 ** 2


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", ("parallel", "fan", "cone"))
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_large_fitting_native_tile_uses_bounded_host_fragments_and_two_pinned_slots(beam, operation):
    case = _schedule_case(beam, views=5)
    image, sino = _data(case, device="cpu")
    data = (image if operation == "project" else sino).double()
    shape = case.sino_shape if operation == "project" else case.shape
    source = _FragmentStore(data, 128)
    output = _FragmentStore(torch.full(shape, torch.nan), 256)
    caller = (source.tensor, output.tensor, output.overwrites, *case.trajectory)
    with _ceilings(slot=256, scratch=128), _memory_inventory({torch.cuda.current_device(): 80 * 1024 ** 3}):
        projector = _scheduled_projector(case, "spatial", volume_chunk_shape=case.shape, view_chunk_size=2)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with _PipelineTrace(case, caller_tensors=caller, host_fragments=True) as trace:
            assert getattr(projector, f"{operation}_into")(source, output) is output
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline
        report = _check_report(projector, trace, operation, schedule="spatial")
        assert tuple(report["chunk_shape"]) == case.shape, "host cap unnecessarily shrank a fitting resident GPU tile"
        assert peak <= report["estimated_gpu_bytes"] <= report["gpu_budget_bytes"]
        assert 0 < trace.pinned_peak == report["pinned_bytes_peak"] <= 512
        assert all(record["bytes"] <= 256 for record in trace.copies), "unfragmented transfer exceeded the host slot"
        assert all(record["volume"]["contiguous"] for record in trace.launches)
        assert len({record["volume"]["allocation"] for record in trace.launches}) == 1, \
            "one fitting native tile was reallocated across host fragments or view batches"
    matrix = _matrix(case, _offsets(case, flat=True))
    operator = matrix if operation == "project" else matrix.T
    _close(output.tensor, (operator @ _numpy(data).ravel()).reshape(shape))
    assert torch.all(output.overwrites == 1)
    assert len(source.reads) > 1 and len(output.writes) > 1
    assert report["d2h_bytes"] == math.prod(shape) * 4, "completed output payload was downloaded more than once"


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_distinct_transfer_compute_streams_real_event_dependencies_and_pending_work_overlap(operation):
    case = _schedule_case("cone", views=7)
    image, sino = _data(case, device="cpu")
    data = image if operation == "project" else sino
    projector = _scheduled_projector(case, "window", volume_chunk_shape=(3, 4, 5), view_chunk_size=2)
    projector.project(image)  # Compile real native kernels outside the observed invocation.
    torch.cuda.synchronize()
    with _PipelineTrace(case, caller_tensors=(image, sino, *case.trajectory), delay=True) as trace:
        result = getattr(projector, operation)(data)
    report = _check_report(projector, trace, operation, schedule="window")
    streams = {direction: {item["stream"] for item in trace.copies if item["direction"] == direction}
               for direction in ("h2d", "d2h")}
    compute = {record["stream"] for record in trace.launches}
    assert streams["h2d"] and streams["d2h"] and compute
    assert not (streams["h2d"] & compute or streams["d2h"] & compute or streams["h2d"] & streams["d2h"])
    assert trace.waits, "no actual inter-stream event waits were observed"
    assert len(trace.global_syncs) <= 1, "global synchronization serialized each batch"
    assert trace.h2d_while_kernel_pending > 0, "next upload was never queued while real native work remained pending"
    assert all(item["non_blocking"] for item in trace.copies if item["source"]["pinned"] or item["target"]["pinned"])
    assert report["window_size"] == 2
    matrix = _matrix(case, _offsets(case, flat=True))
    operator = matrix if operation == "project" else matrix.T
    _close(result, (operator @ _numpy(data).ravel()).reshape(result.shape))


@pytest.mark.cuda
@cuda_required
def test_reused_slots_short_tails_and_allocator_churn_preserve_all_completed_values():
    case = _schedule_case("cone", views=7)
    projector = _scheduled_projector(case, "window", volume_chunk_shape=(3, 4, 5), view_chunk_size=2)
    image, sino = _data(case, device="cpu")
    matrix = _matrix(case, _offsets(case, flat=True))
    for trial in range(3):
        current = image * (trial + 1) + .017 * trial
        churn = [torch.empty(47 + index, device="cuda").fill_(.3) for index in range(6)]
        del churn
        gc.collect()
        with _PipelineTrace(case, caller_tensors=(current, sino, *case.trajectory), delay=True) as trace:
            result = projector.project(current)
        _close(result, (matrix @ _numpy(current).ravel()).reshape(case.sino_shape))
        assert {record["views"] for record in trace.launches} == {1, 2}
        assert trace.pinned_peak > 0


@pytest.mark.cuda
@cuda_required
def test_store_error_drains_inflight_work_and_next_operation_can_reuse_pipeline():
    case = _schedule_case("cone", views=7)
    image, _ = _data(case, device="cpu")

    class FailingOutput(_GuardedStore):
        def write(self, index, value, *, accumulate=False):
            raise RuntimeError("test output backend failed")

    output = FailingOutput(torch.empty(case.sino_shape), (2, *case.detector))
    source = _GuardedStore(image, (3, 4, 5))
    projector = _scheduled_projector(case, "window", volume_chunk_shape=(3, 4, 5), view_chunk_size=2)
    with _PipelineTrace(case, caller_tensors=(image, output.tensor, output.overwrites, *case.trajectory), delay=True) as trace:
        with pytest.raises(RuntimeError, match="test output backend failed"):
            projector.project_into(source, output)
        assert all(event.query() for event in trace.probe_events), "error escaped with pending transfers/native work"
        assert projector.last_execution_stats is None
    result = projector.project(image)
    matrix = _matrix(case, _offsets(case, flat=True))
    _close(result, (matrix @ _numpy(image).ravel()).reshape(case.sino_shape))


@pytest.mark.cuda
@cuda_required
def test_tiny_gpu_budget_reports_serial_pipeline_fallback_without_partial_volume_downloads():
    case = _schedule_case("parallel", views=5)
    image, sino = _data(case, device="cpu")
    with _memory_inventory({torch.cuda.current_device(): 4096}):
        projector = _scheduled_projector(case, volume_chunk_shape=(1, 2), view_chunk_size=1)
        with _PipelineTrace(case, caller_tensors=(image, sino, *case.trajectory)) as trace:
            result = projector.backproject(sino)
    report = _check_report(projector, trace, "backproject")
    assert report["window_size"] == 1 and report.get("fallback_reason")
    assert report["d2h_bytes"] == 4 * math.prod(case.shape)
    matrix = _matrix(case, _offsets(case, flat=True))
    _close(result, (matrix.T @ _numpy(sino).ravel()).reshape(case.shape))


@pytest.mark.cuda
@cuda_required
def test_explicit_geometry_batch_that_exceeds_independent_host_ceiling_fails_before_native_launch():
    from tests.test_batched_surfaces import _Sampler, _fixture
    case, parameters = _fixture("cone")
    sampler = _Sampler(case)
    surface = __import__("diffct").ParameterizedSurface(sampler, parameters=parameters)
    image, _ = _data(case, device="cpu")
    with _ceilings(geometry=64), _CopyTrace(case) as trace, _rejection("geometry|host|memory|budget|fit|limit"):
        projector = _scheduled_projector(case, surface=surface, volume_chunk_shape=(3, 4, 5),
                                         view_chunk_size=2, detector_chunk_shape=(4, 3))
        projector.project(image)
    assert not trace.launches


@pytest.mark.cuda
@cuda_required
def test_automatic_geometry_batches_obey_host_ceiling_even_with_large_gpu_inventory_and_vjp():
    import diffct
    from tests.test_detector_surfaces import _grids
    case = _schedule_case("cone", views=5)
    coefficient = torch.tensor(.021, dtype=torch.float64, requires_grad=True)
    samples = []

    def sampler(u, v, ids, parameter):
        assert u.device.type == v.device.type == ids.device.type == parameter.device.type == "cpu"
        # Shared float64 offsets plus one float64 world point per selected view
        # are necessary live geometry, even before cotangent/graph scratch.
        assert 8 * u.numel() * 3 * (1 + len(ids)) <= 2048
        samples.append((tuple(u.shape), tuple(ids.tolist())))
        return torch.stack((u, v, parameter * (u.square() + .09 * u * v)), dim=-1)

    surface = diffct.ParameterizedSurface(sampler, parameters=(coefficient,))
    image, sino = _data(case, device="cpu")
    image.requires_grad_()
    with _ceilings(geometry=2048), _memory_inventory({torch.cuda.current_device(): 80 * 1024 ** 3}):
        projector = _scheduled_projector(case, surface=surface, volume_chunk_shape=case.shape)
        with _PipelineTrace(case, caller_tensors=(image, sino, coefficient, *case.trajectory)) as trace:
            output = projector.project(image)
            gradients = torch.autograd.grad((output * sino).sum(), (image, coefficient))
    u, v = _grids(case)
    offsets = __import__("numpy").stack((u, v, .021 * (u ** 2 + .09 * u * v)), axis=-1)
    matrix = _matrix(case, offsets)
    _close(output, (matrix @ _numpy(image).ravel()).reshape(case.sino_shape))
    _close(gradients[0], (matrix.T @ _numpy(sino).ravel()).reshape(case.shape))
    assert torch.isfinite(gradients[1]) and abs(float(gradients[1])) > 1e-3
    assert any(record["kind"] == "geometry_vjp" for record in trace.launches)
    assert samples and max(len(ids) * math.prod(shape) for shape, ids in samples) < math.prod(case.sino_shape)
    assert {index for _, ids in samples for index in ids} == set(range(case.views))


@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two actual CUDA GPUs are required")
@pytest.mark.parametrize("beam", ("parallel", "fan", "cone"))
def test_two_local_gpu_accumulators_combine_resident_before_one_completed_host_payload(beam):
    case = _schedule_case(beam, views=7)
    projector = _scheduled_projector(case, "spatial", devices=[0, 1], volume_chunk_shape=(3, 4, 5) if beam == "cone" else (4, 5),
                                     view_chunk_size=2)
    _, sino = _data(case, device="cpu")
    with _CopyTrace(case) as trace:
        output = projector.backproject(sino)
    report = _check_report(projector, trace, "backproject", schedule="spatial")
    assert {record["device"].index for record in trace.launches} == {0, 1}
    assert report["p2p_bytes"] > 0
    assert report["d2h_bytes"] == 4 * math.prod(case.shape)
    matrix = _matrix(case, _offsets(case, flat=True))
    _close(output, (matrix.T @ _numpy(sino).ravel()).reshape(case.shape))


def _resident_nccl_worker(rank, beam, views, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=120))
    try:
        case = _schedule_case(beam, views=views)
        chunk = (3, 4, 5) if beam == "cone" else (4, 5)
        projector = _scheduled_projector(case, "spatial", devices=[rank], distributed=True,
                                         volume_chunk_shape=chunk, view_chunk_size=1)
        _, global_sino = _data(case, device="cpu")
        local = global_sino[projector.view_slice].clone()
        collectives = []
        original = dist.all_reduce
        with _CopyTrace(case) as trace:
            def reduce(tensor, *args, **kwargs):
                operation = kwargs.get("op", args[0] if args else dist.ReduceOp.SUM)
                if operation == dist.ReduceOp.SUM and tensor.dtype == torch.float32:
                    assert tensor.is_cuda and tensor.numel() <= math.prod(chunk)
                    collectives.append({"sequence": trace._step(), "storage": tensor.untyped_storage()._cdata,
                                        "bytes": tensor.numel() * tensor.element_size()})
                return original(tensor, *args, **kwargs)
            with mock.patch.object(dist, "all_reduce", reduce):
                output = projector.backproject(local)
        report = _check_report(projector, trace, "backproject", schedule="spatial")
        tiles = math.prod(math.ceil(size / part) for size, part in zip(case.shape, chunk))
        assert len(collectives) == tiles
        assert report["collective_calls"] == tiles
        assert report["collective_bytes"] == sum(record["bytes"] for record in collectives)
        assert report["d2h_bytes"] == 4 * math.prod(case.shape)
        # Same resident storage participates in SUM; no CPU partial plus reupload.
        for launch in trace.launches:
            assert any(item["storage"] == launch["volume"]["storage"] and item["sequence"] > launch["sequence"]
                       for item in collectives)
        for copied in [item for item in trace.copies if item["direction"] == "d2h"]:
            assert any(item["sequence"] < copied["sequence"] for item in collectives)
        matrix = _matrix(case, _offsets(case, flat=True))
        _close(output, (matrix.T @ _numpy(global_sino).ravel()).reshape(case.shape))
        if projector.projection_shape[0] == 0:
            assert not trace.launches and collectives, "empty rank skipped tile collectives"
    finally:
        dist.destroy_process_group()


@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2 or not dist.is_nccl_available(),
                    reason="Two actual CUDA GPUs and real NCCL are required")
@pytest.mark.parametrize("beam", ("parallel", "fan", "cone"))
@pytest.mark.parametrize("views", [1, 3], ids=["empty-rank", "uneven-views"])
def test_nccl_tile_sum_is_resident_bounded_before_download_including_empty_rank(beam, views, tmp_path):
    mp.spawn(_resident_nccl_worker, args=(beam, views, (tmp_path / "resident-tile-sum").resolve().as_uri()),
             nprocs=2, join=True)
