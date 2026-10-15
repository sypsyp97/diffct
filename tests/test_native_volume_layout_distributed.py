"""Two-rank NCCL native-layout contracts; each rank owns one CUDA device."""

from datetime import timedelta
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.test_native_volume_layout import (
    DETECTOR, SHAPE, VIEWS, _assert_atomic_close, _assert_native,
    _geometry, _native_view, _projector, _values,
)


def _worker(rank, rendezvous, mismatch):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        geometry = _geometry()
        options = dict(geometry=geometry, distributed=True, partition="views",
                       devices=[rank])
        if mismatch:
            # Both ranks must reject at construction, before any projection.
            with pytest.raises(ValueError, match=(
                "distributed ranks must agree on .*native volume layout"
            )):
                _projector(bool(rank), **options)
            return

        ordinary = _projector(False, **options)
        native = _projector(True, **options)
        assert ordinary.rank == native.rank == rank
        assert ordinary.world_size == native.world_size == 2
        assert ordinary.view_slice == native.view_slice
        assert native.projection_shape == ((3 if rank == 0 else 2), *DETECTOR)
        volume = _values(SHAPE)
        sinogram = _values((VIEWS, *DETECTOR), seed=53)[ordinary.view_slice].contiguous()
        with torch.no_grad():
            # Complete all collectives before assertions, so a local numerical
            # failure cannot leave the other rank waiting in its next operation.
            expected = ordinary.backproject(sinogram)
            repeated = ordinary.backproject(sinogram)
            actual = native.backproject(sinogram)
            projected = ordinary.project(volume)
            projected_again = ordinary.project(volume)
            native_projected = native.project(_native_view(volume))
        assert expected.is_contiguous()
        _assert_native(actual)
        _assert_atomic_close(actual, expected, repeated)
        _assert_atomic_close(native_projected, projected, projected_again)
        assert ordinary.last_execution_stats is None
        assert native.last_execution_stats is None
    finally:
        dist.destroy_process_group()


@pytest.mark.cuda
@pytest.mark.parametrize("mismatch", [False, True], ids=["matching", "mismatched-flags"])
def test_native_layout_two_rank_nccl(tmp_path, mismatch):
    if torch.cuda.device_count() < 2:
        pytest.skip("Two CUDA GPUs are required, one per NCCL rank")
    if not dist.is_available() or not dist.is_nccl_available():
        pytest.skip("PyTorch NCCL support is required")
    rendezvous = (tmp_path / "rendezvous").resolve().as_uri()
    context = mp.spawn(_worker, args=(rendezvous, mismatch), nprocs=2, join=False)
    deadline = time.monotonic() + 180
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise AssertionError("native-layout NCCL checks exceeded 180 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
