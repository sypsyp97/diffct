"""Event-owned references survive pending transfers and bounded error cleanup."""

import gc
from types import SimpleNamespace
from unittest import mock
import weakref

import numpy as np
import pytest
import torch
from tests._multiprocessing import spawn

from diffct import chunking
import diffct.pipeline as pipeline
from tests.test_chunked_projector import _offsets
from tests.test_detector_surfaces import _case, _data, _matrix, _projector


MIB = 1024 ** 2
cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


class _Payload:
    pass


class _Completion:
    """CPU-only lifetime probe, not a substitute CUDA event or numerical proof."""
    def __init__(self, references=()):
        self.references, self.synchronized = references, False

    def query(self):
        return False

    def synchronize(self):
        gc.collect()
        assert all(reference() is not None for reference in self.references), \
            "oldest transfer lost source/target references before its completion barrier"
        self.synchronized = True


def _entry():
    source, target = _Payload(), _Payload()
    references = (weakref.ref(source), weakref.ref(target))
    return (_Completion(references), source, target), references


@pytest.mark.parametrize("slots", [1, 2])
def test_oldest_retained_payload_references_live_at_completion_barrier_entry(slots):
    entries = [_entry() for _ in range(slots)]
    oldest, references = entries[0][0][0], entries[0][1]
    owner = SimpleNamespace(host=[None] * slots, pending=[entry[0] for entry in entries])
    del entries
    following, source, target = _Completion(), _Payload(), _Payload()
    pipeline._DevicePipeline.retain(owner, following, source, target)
    assert oldest.synchronized and len(owner.pending) == slots
    gc.collect()
    assert all(reference() is None for reference in references), "completed oldest payload remained retained"


class _Store:
    def __init__(self, tensor, *, decoded=False, fail_read=0, fail_write=False):
        self.tensor, self.shape, self.dtype, self.device = tensor, tuple(tensor.shape), tensor.dtype, tensor.device
        self.writable, self.decoded = True, decoded
        self.fail_read, self.fail_write, self.read_count = fail_read, fail_write, 0
        self.events = []

    def read(self, selected):
        self.read_count += 1
        if self.read_count == self.fail_read:
            raise RuntimeError("transfer test read aborted")
        value = self.tensor[selected].clone()
        if self.decoded:
            torch.cuda._sleep(50000000)
            value.mul_(1.7).add_(.23)
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
            self.events.append(ready)
        return value

    def write(self, selected, value, *, accumulate=False):
        if self.fail_write:
            raise RuntimeError("transfer test write aborted")
        self.tensor[selected].add_(value) if accumulate else self.tensor[selected].copy_(value)

    def flush(self):
        pass


@torch.no_grad()
def _lifetime_worker(rank, mode):
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    ceiling = 12 * MIB
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(ceiling / total, 0)
    original_copy = pipeline._copy
    probes, delayed = [], False

    def copy(target, source, stats):
        nonlocal delayed
        direction = "h2d" if target.is_cuda and not source.is_cuda else "d2h" if source.is_cuda and not target.is_cuda else "other"
        wanted = "h2d" if mode == "h2d-abort" else "d2h"
        if direction == wanted and not delayed:
            delayed = True
            torch.cuda._sleep(200000000)
            pending = torch.cuda.Event()
            pending.record(torch.cuda.current_stream())
            assert not pending.query(), "delay completed before the pending-transfer observation"
            chunking._reservation_budget(torch.device("cuda:0"))
            assert not pending.query(), "capacity observation waited for unrelated pending transfer work"
            probes.append(pending)
        return original_copy(target, source, stats)

    with mock.patch.object(pipeline, "_copy", side_effect=copy):
        if mode == "h2d-abort":
            stats = dict(h2d_bytes=0, d2h_bytes=0, p2p_bytes=0, pinned_bytes_peak=0)
            pipe = pipeline._DevicePipeline(torch.device("cuda:0"), 17, stats)
            target = pipe.empty((37,))
            target.fill_(-19.)
            pipe.upload.wait_stream(pipe.caller)
            source = _Store(torch.linspace(-.4, .8, 37), fail_read=2)
            caught = False
            try:
                pipe.upload_store(source, (slice(0, 37),), target)
            except RuntimeError as error:
                assert "transfer test read aborted" in str(error)
                error.__traceback__ = None  # The exception must not accidentally own the target.
                caught = True
            assert caught and source.read_count == 2
            del target
            churn = torch.full((37,), 73., device="cuda:0")
            pipe.close()
            torch.testing.assert_close(churn.cpu(), torch.full((37,), 73.))
        else:
            case = _case("cone", views=3)
            image, _ = _data(case, device="cpu")
            source = _Store(image.cuda(), decoded=True)
            operator = _projector(case, volume_chunk_shape=(3, 4, 5), view_chunk_size=2,
                                  detector_chunk_shape=(4, 3), schedule="window")
            expected = (_matrix(case, _offsets(case, flat=True)) @
                        (image.double().numpy().ravel() * 1.7 + .23)).reshape(case.sino_shape)
            output = _Store(torch.empty(case.sino_shape), fail_write=mode == "d2h-abort")
            if output.fail_write:
                with pytest.raises(RuntimeError, match="transfer test write aborted"):
                    operator.project_into(source, output)
                assert operator.last_execution_stats is None
                assert all(event.query() for event in probes + source.events)
                output.fail_write = False
            for _ in range(2):
                operator.project_into(source, output)
                torch.testing.assert_close(output.tensor.double(), torch.from_numpy(expected), rtol=4e-4, atol=6e-5)
                assert all(event.query() for event in probes + source.events)
            torch.testing.assert_close(source.tensor.cpu(), image)
    assert delayed and probes and all(event.query() for event in probes)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_reserved() <= ceiling


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("mode", ["decoded-delay", "h2d-abort", "d2h-abort"])
def test_pending_transfer_values_and_error_cleanup_survive_capacity_observation(mode):
    if torch.cuda.get_allocator_backend() != "native":
        pytest.skip("native retirement controls require the native CUDA allocator")
    spawn(_lifetime_worker, args=(mode,), nprocs=1, join=True)
