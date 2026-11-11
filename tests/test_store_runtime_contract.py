"""Portable store protocol, observed I/O accounting and parent-group ownership."""

from dataclasses import replace
from datetime import timedelta
import json
import math
from pathlib import Path
import time
from unittest import mock

import numpy as np
import pytest
import torch
import torch.distributed as dist

import diffct
from tests.test_execution_features import _run_four
from tests.test_reference_backend import _case, _data, _matrix, _operator
from tests.test_runtime_store_failures import _problem, _stores


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="physical CUDA required")
GLOO = pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(),
                         reason="four real Gloo ranks required")


class _OpaqueCPUStore:
    """Expose the five required store attributes, without optional device metadata."""

    def __init__(self, raw):
        self._raw = raw
        self.reads, self.writes = [], []

    @property
    def shape(self):
        return self._raw.shape

    @property
    def dtype(self):
        return self._raw.dtype

    def read(self, selected):
        value = self._raw.read(selected)
        self.reads.append((selected, value.numel() * value.element_size()))
        return value

    def write(self, selected, value, *, accumulate=False):
        self._raw.write(selected, value, accumulate=accumulate)
        self.writes.append((selected, value.numel() * value.element_size()))

    def flush(self):
        self._raw.flush()


class _DeviceStore(_OpaqueCPUStore):
    @property
    def device(self):
        return self._raw.device


def _portable_case(method):
    case = replace(_case("parallel"), shape=(3, 2), detector=(3,))
    operator = _operator(case, devices=None, volume_chunk_shape=(1, 1),
                         view_chunk_size=1, detector_chunk_shape=(1,))
    source_shape = case.shape if method == "project" else case.projection_shape
    output_shape = case.projection_shape if method == "project" else case.shape
    data = _data(source_shape, phase=.27)
    matrix = _matrix(case)
    expected = ((matrix if method == "project" else matrix.T) @ data.flatten()).reshape(output_shape)
    return case, operator, data, expected


def _assert_io_stats(owner, source, output, case, method, elapsed, device):
    stats = owner.last_execution_stats
    assert stats is not None
    assert stats["operation"] == method
    assert stats["backend"] == "torch"
    assert stats["schedule"] == ("views" if method == "project" else "spatial")
    assert stats["devices"] == [str(device)]
    assert stats["store_reads"] == len(source.reads) > 0
    assert stats["store_writes"] == len(output.writes) > 0
    assert stats["read_bytes"] == sum(size for _, size in source.reads)
    assert stats["write_bytes"] == sum(size for _, size in output.writes)
    # One actual read supplies each ray-batch/spatial-tile pair in either direction.
    expected_pairs = math.prod(case.shape) * math.prod(case.projection_shape)
    assert stats["ray_tile_pairs"] == len(source.reads) == expected_pairs
    assert math.isfinite(stats["elapsed_seconds"])
    assert 0 <= stats["elapsed_seconds"] <= elapsed


@pytest.mark.parametrize("method", ["project", "backproject"])
@pytest.mark.parametrize("kind", ["tensor", "npy"])
def test_portable_into_accepts_opaque_cpu_stores_and_accounts_for_actual_io(method, kind, tmp_path):
    torch.set_num_threads(1)
    case, owner, data, expected = _portable_case(method)
    if kind == "tensor":
        raw_source = diffct.TensorStore(data.clone())
        raw_output = diffct.TensorStore(torch.full_like(expected, torch.nan))
    else:
        np.save(tmp_path / "input.npy", data.numpy())
        raw_source = diffct.NpyStore(tmp_path / "input.npy", mode="r")
        raw_output = diffct.NpyStore.create(tmp_path / "output.npy", expected.shape, dtype=expected.dtype)
    source, output = _OpaqueCPUStore(raw_source), _OpaqueCPUStore(raw_output)
    assert not hasattr(source, "device") and not hasattr(output, "device")
    start = time.perf_counter()
    result = getattr(owner, method + "_into")(source, output)
    elapsed = time.perf_counter() - start
    assert result is output
    torch.testing.assert_close(raw_output.read(tuple(slice(None) for _ in expected.shape)), expected,
                               rtol=3e-11, atol=3e-12)
    torch.testing.assert_close(raw_source.read(tuple(slice(None) for _ in data.shape)), data, rtol=0, atol=0)
    _assert_io_stats(owner, source, output, case, method, elapsed, torch.device("cpu"))


@pytest.mark.cuda
@CUDA
@pytest.mark.parametrize("method", ["project", "backproject"])
def test_portable_into_infers_physical_cuda_from_tensor_stores_and_reports_actual_compute(method):
    torch.set_num_threads(1)
    case, owner, data, expected = _portable_case(method)
    device = torch.device("cuda", torch.cuda.current_device())
    raw_source = diffct.TensorStore(data.to(device))
    raw_output = diffct.TensorStore(torch.full_like(expected, torch.nan, device=device))
    source, output = _DeviceStore(raw_source), _DeviceStore(raw_output)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = getattr(owner, method + "_into")(source, output)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    assert result is output
    assert raw_output.tensor.device == raw_source.tensor.device == device
    torch.testing.assert_close(raw_output.tensor.cpu(), expected, rtol=3e-11, atol=3e-12)
    _assert_io_stats(owner, source, output, case, method, elapsed, device)
    assert owner.last_execution_stats["d2h_bytes"] == 0


def _isolated_parent_worker(rank, rendezvous, directory, method):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=15))
    try:
        groups = [dist.new_group(members, timeout=timedelta(seconds=15))
                  for members in ([0, 1], [2, 3])]
        group = groups[rank // 2]
        call, owner, data, shape, expected = _problem(
            method, "space" if method == "project" else "views", distributed=True, group=group)
        source, output = _stores(data, shape, "read" if rank == 0 else None)
        reductions = []
        original_reduce = dist.all_reduce

        def reduce(value, *args, **kwargs):
            used_group = kwargs.get("group", args[1] if len(args) > 1 else None)
            reductions.append(dict(parent=used_group is group, shape=list(value.shape),
                                   dtype=str(value.dtype)))
            # Check provenance before transport, so a wrong default group fails
            # deterministically rather than waiting for an unrelated group's ranks.
            assert used_group is group, "store execution escaped the caller's parent process group"
            return original_reduce(value, *args, **kwargs)

        record = dict(error=None, returned=False, correct=False, reductions=reductions)
        with mock.patch.object(dist, "all_reduce", reduce):
            try:
                record["returned"] = call(source, output) is output
                torch.testing.assert_close(output.tensor, expected, rtol=3e-11, atol=3e-12)
                record["correct"] = True
            except Exception as error:
                record.update(error=type(error).__name__, message=str(error))
        # Successful reuse proves a failed call did not poison either parent.
        reached = torch.tensor(1)
        dist.all_reduce(reached, group=group)
        record["parent_reused"] = reached.item() == 2
        record["injected"] = source.injected + output.injected
        record["successful_stats"] = owner is not None and owner.last_execution_stats is not None
        (Path(directory) / f"rank-{rank}.json").write_text(json.dumps(record), encoding="utf-8")
    finally:
        dist.destroy_process_group()


@GLOO
@pytest.mark.parametrize("method", ["project", "fbp"])
def test_two_disjoint_parent_groups_confine_store_failure_and_complete_healthy_work(method, tmp_path):
    _run_four(_isolated_parent_worker,
              ((tmp_path / "isolated-parents").resolve().as_uri(), str(tmp_path), method), timeout=70)
    records = [json.loads((tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8"))
               for rank in range(4)]
    for record in records:
        assert record["parent_reused"], records
        assert record["reductions"] and all(item["parent"] for item in record["reductions"]), records
        assert any(item["dtype"] == "torch.int64" and item["shape"] == [1]
                   for item in record["reductions"]), "real store-round control was not observed"
    assert sum(record["injected"] for record in records) == 1, records
    for record in records[:2]:
        assert record["error"] == "PermissionError", records
        assert record["message"] == "injected runtime read failure at call 1", records
        assert not record["returned"] and not record["successful_stats"], records
    for record in records[2:]:
        assert record["error"] is None and record["returned"] and record["correct"], records
