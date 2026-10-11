"""Caller CUDA sinks must not retain every completed ray batch or volume tile."""

import gc
import math

import pytest
import torch

from diffct import TensorStore
from tests.test_block_storage import _GuardedStore, _slices
from tests.test_chunked_projector import _memory_inventory, _offsets
from tests.test_detector_surfaces import _close, _data, _matrix, _numpy
from tests.test_execution_schedules import (
    BEAMS, _check_report, _schedule_case, _scheduled_projector, cuda_required,
)
from tests.test_transfer_pipeline import _PipelineTrace


class _CallerSink(TensorStore):
    """Forward real same-GPU writes; ownership counts stay on the test's CPU."""
    def __init__(self, tensor, maximum):
        super().__init__(tensor)
        self.maximum, self.writes = maximum, []
        self.overwrites = torch.zeros(tuple(tensor.shape), dtype=torch.int32)

    def write(self, index, value, *, accumulate=False):
        selected = _slices(index, self.shape)
        assert value.is_cuda and value.device == self.device
        assert all(size <= limit for size, limit in zip(value.shape, self.maximum)), \
            "completed output escaped its tile or ray-batch limits"
        self.writes.append((selected, accumulate))
        if not accumulate:
            self.overwrites[selected] += 1
        return super().write(selected, value, accumulate=accumulate)


def _sink_shapes(beam):
    return ((5, 7, 9), (13, 17, 19)) if beam == "cone" else ((5, 7), (19, 23))


def _sink_trial(operation, shape, views, *, beam="cone", delay=False):
    case = _schedule_case(beam, views=views)
    case.shape, case.detector = shape, (4, 3) if beam == "cone" else (7,)
    chunk, batch = (3, 4, 5) if beam == "cone" else (3, 4), 2
    # View-first projection emits distinct completed ray accumulators; spatial
    # adjoint emits distinct DHW output tiles. Both lifetimes require bounding.
    schedule = "views" if operation == "project" else "spatial"
    projector = _scheduled_projector(case, schedule, volume_chunk_shape=chunk, view_chunk_size=batch)
    image, sino = _data(case, device="cpu")
    data = image if operation == "project" else sino
    source = _GuardedStore(data, chunk if operation == "project" else (batch, *case.detector))
    output_shape = case.sino_shape if operation == "project" else case.shape
    output = torch.full(output_shape, torch.nan, device=torch.cuda.current_device())
    sink = _CallerSink(output, (batch, *case.detector) if operation == "project" else chunk)
    pointer, storage = output.data_ptr(), output.untyped_storage()._cdata
    # Compile both actual kernels into a separate CPU sink; exclude fixture,
    # caller-output and cold compilation allocations from the extra peak.
    projector.project(image)
    projector.backproject(sino)
    matrix = _matrix(case, _offsets(case, flat=True))
    expected = ((matrix if operation == "project" else matrix.T) @ _numpy(data).ravel()).reshape(output_shape)
    del matrix
    gc.collect()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    # Synthetic available-room metadata constrains the plan; copies, kernels,
    # streams, numerical output and additional PyTorch CUDA peak remain real.
    with _memory_inventory({torch.cuda.current_device(): 32 * 1024}), \
            _PipelineTrace(case, caller_tensors=(image, sino, source.overwrites, sink.overwrites,
                                                output, *case.trajectory), delay=delay) as trace:
        assert getattr(projector, f"{operation}_into")(source, sink) is sink
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    remaining = torch.cuda.memory_allocated() - baseline
    report = _check_report(projector, trace, operation, schedule=schedule)
    # Verify every actual cell before the resource assertions, including tails
    # and the NaN-initialized output. The CUDA allocation remains caller-owned.
    _close(output, expected)
    assert sink.tensor is output and output.data_ptr() == pointer and output.untyped_storage()._cdata == storage
    assert torch.all(sink.overwrites == 1)
    tiles = math.prod(math.ceil(size / part) for size, part in zip(shape, chunk))
    batches = math.ceil(views / batch)
    assert tuple(report["chunk_shape"]) == chunk and report["view_chunk_size"] == batch
    assert report["kernel_launches"] == tiles * batches
    assert {record["views"] for record in trace.launches} == {1, 2}
    assert len(sink.writes) == (batches if operation == "project" else tiles)
    assert report["h2d_bytes"] > 0 and report["d2h_bytes"] == report["p2p_bytes"] == 0
    assert trace.waits and report["pinned_bytes_peak"] == trace.pinned_peak > 0
    assert not report["pilot"]
    assert remaining == 0, "completed execution kept additional CUDA tensor storage"
    return peak, report, tiles, batches, trace.h2d_while_kernel_pending


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_same_gpu_caller_sink_peak_is_bounded_with_tile_and_view_growth(beam, operation):
    small_shape, larger_shape = _sink_shapes(beam)
    small = _sink_trial(operation, small_shape, 5, beam=beam)
    many_tiles = _sink_trial(operation, larger_shape, 5, beam=beam)
    many_views = _sink_trial(operation, small_shape, 37, beam=beam)
    assert many_tiles[2] >= 10 * small[2] and many_views[3] >= 6 * small[3]
    for trial in (small, many_tiles, many_views):
        assert 0 < trial[0] <= trial[1]["estimated_gpu_bytes"] <= trial[1]["gpu_budget_bytes"], \
            ("caller CUDA sink retained completed work beyond its plan", beam, operation, trial)
    # Identical explicit tile/ray limits admit identical resident buffers.
    # A full-input or per-completed-output estimate cannot mask growing peaks.
    estimate = small[1]["estimated_gpu_bytes"]
    assert estimate == many_tiles[1]["estimated_gpu_bytes"] == many_views[1]["estimated_gpu_bytes"]
    assert many_tiles[0] <= small[0] + estimate
    assert many_views[0] <= small[0] + estimate


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_same_gpu_caller_sink_accounts_for_delayed_native_and_unfinished_output_lifetimes(beam, operation):
    shape, _ = _sink_shapes(beam)
    trial = _sink_trial(operation, shape, 5, beam=beam, delay=True)
    assert trial[4] > 0, "delay did not preserve actual pending native work during the observed copies"
    assert 0 < trial[0] <= trial[1]["estimated_gpu_bytes"] <= trial[1]["gpu_budget_bytes"], \
        ("caller CUDA sink did not account for unfinished output lifetimes", beam, operation, trial)
