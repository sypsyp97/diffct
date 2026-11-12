"""Real mixed-device transfer accounting and empty portable SUM contributors."""

from contextlib import ExitStack
from datetime import timedelta
import json
from pathlib import Path
import time
from unittest import mock

import pytest
import torch
import torch.distributed as dist

import diffct
from diffct.backends import ReferenceProjector
from tests._multiprocessing import spawn
from tests.test_execution_schedules import _CopyTrace
from tests.test_reference_backend import _operator
from tests.test_runtime_store_failures import _problem
from tests.test_store_runtime_contract import _portable_case


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="physical CUDA required")
GLOO = pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(),
                         reason="two real Gloo ranks required")


@pytest.mark.cuda
@CUDA
@pytest.mark.parametrize("method", ["project", "backproject"])
@pytest.mark.parametrize("source_device", ["cpu", "cuda"])
def test_portable_into_mixed_stores_report_observed_execution_transfer_bytes(method, source_device):
    torch.set_num_threads(1)
    case, _, data, expected = _portable_case(method)
    cuda = torch.device("cuda", torch.cuda.current_device())
    source_device = torch.device("cpu") if source_device == "cpu" else cuda
    output_device = cuda if source_device.type == "cpu" else torch.device("cpu")
    owner = _operator(case, devices=[cuda.index] if source_device.type == "cpu" else None,
                      volume_chunk_shape=(1, 1), view_chunk_size=1, detector_chunk_shape=(1,))
    source = diffct.TensorStore(data.to(source_device))
    output = diffct.TensorStore(torch.full_like(expected, torch.nan, device=output_device))
    trace, observed = _CopyTrace(case), []
    original_stats = ReferenceProjector._stats
    # Reports start after admission/geometry validation. Observe that same
    # execution boundary while forwarding every actual tensor copy and kernel.
    with ExitStack() as patches:
        def begin_execution(projector, *args, **kwargs):
            stats = original_stats(projector, *args, **kwargs)
            observed.append(stats)
            patches.enter_context(trace)
            return stats

        patches.enter_context(mock.patch.object(ReferenceProjector, "_stats", begin_execution))
        assert getattr(owner, method + "_into")(source, output) is output
    torch.cuda.synchronize(cuda)
    assert len(observed) == 1 and observed[0] is owner.last_execution_stats
    assert owner.last_execution_stats["devices"] == [str(cuda)]
    assert output.tensor.device == output_device
    torch.testing.assert_close(output.tensor.cpu(), expected, rtol=3e-11, atol=3e-12)
    actual = {direction: sum(item["bytes"] for item in trace.copies
                             if item["direction"] == direction)
              for direction in ("h2d", "d2h", "p2p")}
    required_direction = "h2d" if source_device.type == "cpu" else "d2h"
    assert actual[required_direction] > 0, "fixture did not exercise the mixed-store transfer"
    for direction, size in actual.items():
        assert owner.last_execution_stats[direction + "_bytes"] == size, (direction, actual)


def _empty_sum_worker(rank, rendezvous, directory, method, partition):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=15))
    try:
        call, _, data, shape, expected = _problem(method, partition, distributed=True, empty=True)
        source = diffct.TensorStore(data)
        output = diffct.TensorStore(torch.full(shape, torch.nan, dtype=torch.float64))
        reductions = []
        original_reduce = dist.all_reduce

        def reduce(value, *args, **kwargs):
            operation = kwargs.get("op", args[0] if args else dist.ReduceOp.SUM)
            if value.is_floating_point() and operation == dist.ReduceOp.SUM:
                group = kwargs.get("group", args[1] if len(args) > 1 else None)
                dtypes = [None] * dist.get_world_size(group)
                # Inspect actual submitted contributions before forwarding SUM.
                # This keeps an incorrect mixed-dtype request out of Gloo's
                # fatal transport path while every rank observes the mismatch.
                dist.all_gather_object(dtypes, str(value.dtype), group=group)
                assert dtypes == ["torch.float64"] * len(dtypes), \
                    f"portable float64 SUM contributors changed dtype: {dtypes}"
                reductions.append(dtypes)
            return original_reduce(value, *args, **kwargs)

        with mock.patch.object(dist, "all_reduce", reduce):
            assert call(source, output) is output
        torch.testing.assert_close(output.tensor, expected, rtol=3e-11, atol=3e-12)
        assert reductions, "no real floating SUM was observed"
        record = dict(empty=data.numel() == 0, source_shape=list(data.shape), reductions=reductions)
        (Path(directory) / f"rank-{rank}.json").write_text(json.dumps(record), encoding="utf-8")
    finally:
        dist.destroy_process_group()


@GLOO
@pytest.mark.parametrize("method,partition", [("project", "space"), ("backproject", "views")])
def test_portable_into_float64_empty_rank_preserves_real_sum_dtype_and_result(method, partition, tmp_path):
    context = spawn(_empty_sum_worker,
                    args=((tmp_path / "empty-sum").resolve().as_uri(), str(tmp_path), method, partition),
                    nprocs=2, join=False)
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            assert time.monotonic() < deadline, "empty-contributor workers exceeded 60 seconds"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    records = [json.loads((tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8"))
               for rank in range(2)]
    assert sum(record["empty"] for record in records) == 1, records
    assert all(record["reductions"] for record in records), records
