"""Rank-local numerical store failures settle every parent participant."""

from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist

import diffct
from tests._multiprocessing import spawn
from tests.test_execution_features import _scan
from tests.test_reference_backend import _case, _operator


GLOO = pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(),
                         reason="runtime store failure agreement requires Gloo")
METHODS = ("project", "backproject", "fbp", "fdk")
FAILURES = (("read", 1), ("read", 2), ("write", 1), ("write", 2), ("flush", 1))
GROUP_TIMEOUT = 6


class _FailingStore:
    """Delegate real storage; fail one numbered runtime operation only."""

    def __init__(self, tensor, operation=None, occurrence=1):
        self.store = diffct.TensorStore(tensor)
        self.operation, self.occurrence = operation, occurrence
        self.calls = dict(read=0, write=0, flush=0)
        self.error = PermissionError(f"injected runtime {operation} failure at call {occurrence}")
        self.injected = 0

    def __getattr__(self, name):
        return getattr(self.store, name)

    def _visit(self, operation):
        self.calls[operation] += 1
        if operation == self.operation and self.calls[operation] == self.occurrence:
            self.injected += 1
            raise self.error

    def read(self, selected):
        self._visit("read")
        return self.store.read(selected)

    def write(self, selected, value, *, accumulate=False):
        self._visit("write")
        return self.store.write(selected, value, accumulate=accumulate)

    def flush(self):
        self._visit("flush")
        return self.store.flush()


def _data(shape):
    return torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float64).reshape(shape) / 11 + .3


def _problem(method, partition="views", *, distributed=False, group=None, empty=False, grid=None):
    options = dict(distributed=distributed, process_group=group, partition=partition,
                   partition_grid=grid, view_chunk_size=1)
    if method in ("project", "backproject"):
        case = _case("parallel")
        case = replace(case, shape=(1 if empty else 3, 2), detector=(3,))
        if empty:
            case = replace(case, trajectory=tuple(value[:1] for value in case.trajectory))
        operator = _operator(case, volume_chunk_shape=(1, 1), detector_chunk_shape=(1,), **options)
        full_source = _data(case.shape if method == "project" else case.projection_shape)
        source = full_source[operator.volume_slice if method == "project" else operator.view_slice].clone()
        output_shape = operator.projection_shape if method == "project" else operator.volume_shape
        serial = _operator(case, volume_chunk_shape=(1, 1), view_chunk_size=1, detector_chunk_shape=(1,))
        expected = getattr(serial, method)(full_source)
        expected = expected[operator.view_slice if method == "project" else operator.volume_slice]
        return getattr(operator, method + "_into"), operator, source, output_shape, expected

    case = _scan("cone" if method == "fdk" else "parallel")
    count = 1 if empty else 3
    shape = ((1 if empty else 3, 2, 2) if method == "fdk" else (1 if empty else 3, 2))
    detector = (3, 2) if method == "fdk" else (3,)
    case = {**case, "shape": shape, "angles": case["angles"][:count],
            "trajectory": tuple(value[:count] for value in case["trajectory"]),
            "sino_shape": (count, *detector)}
    # Public Projector ownership gives exactly the input/output stores required
    # by the corresponding analytical public entrypoint, including empty ranks.
    owner = diffct.Projector(case["trajectory"], shape, detector, beam=case["beam"],
                             backend="torch", precision="float64", **options)
    full_source = _data(case["sino_shape"])
    source = full_source[owner.view_slice].clone()
    kwargs = dict(backend="torch", precision="float64", voxel_spacing=case["spacing"],
                  detector_spacing=case["pitch"] if method == "fdk" else case["pitch"][0],
                  volume_chunk_shape=(1,) * len(shape), isocenter=case["center"])
    expected = getattr(diffct, method)(full_source, case["trajectory"], shape, case["angles"], **kwargs)

    def call(data, output):
        return getattr(diffct, method + "_into")(data, output, case["trajectory"], shape,
                                                 case["angles"], **kwargs, **options)

    return call, None, source, owner.volume_shape, expected[owner.volume_slice]


def _stores(source, output_shape, operation=None, occurrence=1):
    return (_FailingStore(source.clone(), operation if operation == "read" else None, occurrence),
            _FailingStore(torch.full(output_shape, torch.nan, dtype=torch.float64),
                          operation if operation in ("write", "flush") else None, occurrence))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("operation,occurrence", FAILURES)
def test_single_rank_store_failure_and_healthy_retry(method, operation, occurrence):
    call, owner, data, shape, expected = _problem(method)
    source, output = _stores(data, shape, operation, occurrence)
    failure = source if operation == "read" else output
    with pytest.raises(PermissionError) as caught:
        call(source, output)
    assert caught.value is failure.error
    assert failure.injected == 1
    if owner is not None:
        assert owner.last_execution_stats is None
    source, output = _stores(data, shape)
    assert call(source, output) is output
    torch.testing.assert_close(output.tensor, expected, rtol=3e-11, atol=3e-12)


def _wait_for(paths, message):
    deadline = time.monotonic() + 3 * GROUP_TIMEOUT
    while not all(path.exists() for path in paths):
        assert time.monotonic() < deadline, message
        time.sleep(.01)


def _publish(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)


def _read_record(path):
    deadline = time.monotonic() + GROUP_TIMEOUT
    while True:
        try:
            content = path.read_text(encoding="utf-8")
        except PermissionError:
            # Windows scanners may briefly deny a newly published file.
            if time.monotonic() >= deadline:
                raise
            time.sleep(.01)
        else:
            return json.loads(content)


def _worker(rank, rendezvous, directory, world, method, operation, occurrence, partition, empty, subset):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(seconds=GROUP_TIMEOUT))
    root = Path(directory)
    try:
        members = list(range(1, world)) if subset else list(range(world))
        group = (dist.new_group(members, timeout=timedelta(seconds=GROUP_TIMEOUT)) if subset else None)
        if rank not in members:
            _wait_for([root / f"done-{peer}.json" for peer in members], "parent participants did not finish")
            return
        parent_rank = members.index(rank)
        grid = (2, 2) if partition == "hybrid" else None
        call, owner, data, shape, expected = _problem(
            method, partition, distributed=True, group=group, empty=empty, grid=grid)
        source, output = _stores(data, shape, operation if parent_rank == 0 else None, occurrence)
        record = dict(error=None, message=None, returned=False, recovery=False,
                      source_shape=list(data.shape), output_shape=list(shape))
        try:
            result = call(source, output)
            record["returned"] = result is output
        except Exception as error:
            record.update(error=type(error).__name__, message=str(error))
        record.update(injected=source.injected + output.injected,
                      calls=dict(source=source.calls, output=output.calls),
                      successful_stats=owner is not None and owner.last_execution_stats is not None)
        path = root / f"attempt-{rank}.json"
        _publish(path, record)
        attempts = [root / f"attempt-{peer}.json" for peer in members]
        _wait_for(attempts, "peer did not return from the bounded store call")
        # The unmodified implementation may have broken Gloo by timing out.
        # Retry only when every rank actually observed the original failure.
        records = [_read_record(path) for path in attempts]
        message = f"injected runtime {operation} failure at call {occurrence}"
        if all(item["error"] == "PermissionError" and item["message"] == message for item in records):
            source, output = _stores(data, shape)
            try:
                assert call(source, output) is output
                torch.testing.assert_close(output.tensor, expected, rtol=3e-11, atol=3e-12)
                # Reuse the caller's parent group itself after the healthy call.
                total = torch.tensor(1)
                dist.all_reduce(total, group=group)
                assert total.item() == len(members)
                record["recovery"] = True
            except Exception as error:
                record["recovery_error"] = f"{type(error).__name__}: {error}"
        _publish(root / f"done-{rank}.json", record)
        _wait_for([root / f"done-{peer}.json" for peer in members], "peer did not finish healthy retry")
    finally:
        dist.destroy_process_group()


def _run(tmp_path, method, operation, occurrence, partition, *, empty=False, subset=False):
    world = 5 if subset else 2
    context = spawn(_worker, args=((tmp_path / "rendezvous").resolve().as_uri(), str(tmp_path), world,
                                   method, operation, occurrence, partition, empty, subset),
                    nprocs=world, join=False)
    deadline = time.monotonic() + 55
    try:
        while not context.join(timeout=1):
            assert time.monotonic() < deadline, "runtime store workers exceeded 55 seconds"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    members = range(1, world) if subset else range(world)
    records = [_read_record(tmp_path / f"done-{rank}.json") for rank in members]
    assert sum(item["injected"] for item in records) == 1, records
    message = f"injected runtime {operation} failure at call {occurrence}"
    assert all(item["error"] == "PermissionError" and item["message"] == message for item in records), records
    assert all(not item["returned"] and not item["successful_stats"] for item in records), records
    assert all(item["recovery"] for item in records), records
    return records


@GLOO
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("operation,occurrence", FAILURES)
def test_two_rank_store_failure_agrees_before_peer_collective(method, operation, occurrence, tmp_path):
    partition = "space" if method == "project" else "views"
    _run(tmp_path, method, operation, occurrence, partition)


@GLOO
@pytest.mark.parametrize("method,partition", [("project", "views"), ("backproject", "space"),
                                               ("fbp", "space"), ("fdk", "space")])
def test_parent_failure_agreement_when_numerical_sum_has_one_rank(method, partition, tmp_path):
    _run(tmp_path, method, "read", 2, partition)


@GLOO
@pytest.mark.parametrize("method,partition", [("project", "views"), ("backproject", "space"),
                                               ("fbp", "views"), ("fdk", "space")])
def test_empty_views_or_slabs_still_observe_parent_store_failure(method, partition, tmp_path):
    records = _run(tmp_path, method, "read", 1, partition, empty=True)
    assert any(0 in record["source_shape"] or 0 in record["output_shape"] for record in records)


@GLOO
@pytest.mark.parametrize("method,operation,occurrence", [("project", "read", 2),
                                                        ("backproject", "write", 2),
                                                        ("fbp", "flush", 1), ("fdk", "read", 2)])
def test_uneven_hybrid_subset_store_failure_reaches_entire_parent(method, operation, occurrence, tmp_path):
    _run(tmp_path, method, operation, occurrence, "hybrid", subset=True)
