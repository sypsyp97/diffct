"""Completed tile downloads may stay pending while the next native work queues."""

import gc
import math

import pytest
import torch
from torch.utils._pytree import tree_flatten

from tests.test_block_storage import _GuardedStore
from tests.test_chunked_projector import _offsets
from tests.test_detector_surfaces import _close, _data, _matrix, _numpy
from tests.test_execution_schedules import (
    BEAMS, _check_report, _schedule_case, _scheduled_projector, _stream, cuda_required,
)
from tests.test_transfer_pipeline import _ceilings, _PipelineTrace


class _DownloadOverlap(_PipelineTrace):
    """Delay real output-stream work; every copy and native launch still executes."""
    def __init__(self, case, *, caller_tensors=(), serialize=False):
        super().__init__(case, caller_tensors=caller_tensors)
        self.serialize = serialize
        self.output_events, self.native_observations = [], []
        self.download_pending_after_copy = 0

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        inputs = [value for value in tree_flatten(args)[0] if isinstance(value, torch.Tensor)]
        name = str(function)
        download = (name == "aten.copy_.default" and inputs[0].device.type == "cpu" and inputs[1].is_cuda
                    or name == "aten._to_copy.default" and inputs[0].is_cuda
                    and torch.device((kwargs or {}).get("device", inputs[0].device)).type == "cpu")
        if download:
            # Queue actual GPU work on the active D2H stream. At 50M cycles the
            # probe remains pending across Python dispatch/metadata overhead.
            # This changes stress duration, never numerical copy/kernel results.
            torch.cuda._sleep(50000000)
        before = len(self.copies)
        result = super().__torch_dispatch__(function, types, args, kwargs)
        if download:
            assert len(self.copies) == before + 1 and self.copies[-1]["direction"] == "d2h"
            event = self.probe_events[-1]
            self.output_events.append(event)
            self.download_pending_after_copy += int(not event.query())
        return result

    def _proxy(self, kernel, beam, kind):
        parent = super()._proxy(kernel, beam, kind)
        tracer = self

        class Proxy:
            def __getitem__(self, configuration):
                launch = parent[configuration]

                def observed(*args, **kwargs):
                    if tracer.serialize:
                        # A real event barrier is the negative control. It does
                        # not fabricate a completed flag or substitute a kernel.
                        for event in tracer.output_events:
                            event.synchronize()
                    pending = sum(not event.query() for event in tracer.output_events)
                    tracer.native_observations.append({"stream": _stream(), "pending_downloads": pending,
                                                       "previous_downloads": len(tracer.output_events)})
                    return launch(*args, **kwargs)
                return observed
        return Proxy()


def _output_overlap_trial(beam, *, serialize=False):
    case = _schedule_case(beam, views=5)
    case.shape = (5, 7, 9) if beam == "cone" else (5, 7)
    case.detector = (4, 3) if beam == "cone" else (7,)
    chunk, batch = (3, 4, 5) if beam == "cone" else (3, 4), 2
    operator = _scheduled_projector(case, "window", volume_chunk_shape=chunk, view_chunk_size=batch)
    image, sino = _data(case, device="cpu")
    source = _GuardedStore(sino, (batch, *case.detector))
    sink = _GuardedStore(torch.full(case.shape, torch.nan), chunk)
    # Cold Numba imports and all caller/oracle allocations precede measurement.
    operator.project(image)
    operator.backproject(sino)
    matrix = _matrix(case, _offsets(case, flat=True))
    expected = (matrix.T @ _numpy(sino).ravel()).reshape(case.shape)
    del matrix
    gc.collect()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with _ceilings(), _DownloadOverlap(case, serialize=serialize,
            caller_tensors=(image, sino, sink.tensor, source.overwrites, sink.overwrites, *case.trajectory)) as trace:
        assert operator.backproject_into(source, sink) is sink
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    report = _check_report(operator, trace, "backproject", schedule="window")
    # Every cell and completed write passes before the scheduling assertion.
    _close(sink.tensor, expected)
    assert torch.all(sink.overwrites == 1)
    tiles = math.prod(math.ceil(size / part) for size, part in zip(case.shape, chunk))
    assert len(sink.writes) == tiles and report["kernel_launches"] == tiles * math.ceil(case.views / batch)
    assert report["d2h_bytes"] == math.prod(case.shape) * 4 and report["p2p_bytes"] == 0
    assert not report["pilot"] and report["window_size"] == 2
    assert tuple(report["chunk_shape"]) == chunk and report["view_chunk_size"] == batch
    assert 0 < peak <= report["estimated_gpu_bytes"] <= report["gpu_budget_bytes"]
    assert 0 < trace.pinned_peak == report["pinned_bytes_peak"] <= 2 * 512
    assert trace.waits and not trace.global_syncs
    assert torch.cuda.memory_allocated() == baseline
    downloads = [record for record in trace.copies if record["direction"] == "d2h"]
    compute = {record["stream"] for record in trace.launches}
    assert downloads and not ({record["stream"] for record in downloads} & compute)
    assert all(record["non_blocking"] and record["target"]["pinned"] for record in downloads)
    assert trace.download_pending_after_copy == len(downloads) > 1, "stress work completed before the copy probes"
    assert any(record["previous_downloads"] for record in trace.native_observations), "fixture lacks a native launch after a completed-tile drain"
    return {"beam": beam, "serialize": serialize, "report": report, "peak_cuda_bytes": peak,
            "pinned_peak_bytes": trace.pinned_peak, "actual_downloads": len(downloads),
            "pending_after_copy": trace.download_pending_after_copy,
            "native_with_download_pending": sum(record["pending_downloads"] > 0 for record in trace.native_observations),
            "native_observations": trace.native_observations,
            "numeric_oracle": "PASS", "completed_tile_writes": "PASS", "actual_copy_report": "PASS"}


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("serialize", [False, True], ids=["pipeline", "event-serialized-control"])
def test_previous_completed_tile_download_overlaps_next_real_native_enqueue(beam, serialize):
    trial = _output_overlap_trial(beam, serialize=serialize)
    if serialize:
        assert trial["native_with_download_pending"] == 0, trial
    else:
        assert trial["native_with_download_pending"] > 0, \
            ("completed tile download was always synchronized before the next native launch", trial)
