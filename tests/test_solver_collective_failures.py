"""Real two-rank solver rejection before numerical or caller-state mutation."""

from contextlib import contextmanager
from datetime import timedelta
import errno
import json
from pathlib import Path
import time
from unittest import mock

import numpy as np
import pytest
import torch
import torch.distributed as dist
from tests._multiprocessing import spawn

import diffct
from tests.test_execution_features import GLOO
from tests.test_solver_api import _cgls_reference, _checkpoint
from tests.test_solver_checkpoint_edges import _numerical_calls, _real_problem


# Fixture I/O may delay a peer before the measured rejection begins.
GROUP_TIMEOUT = 30
REJECTION_TIMEOUT = 5
SENTINEL = 37.


def _run_two(worker, tmp_path, *args):
    rendezvous = (tmp_path / "rendezvous").resolve().as_uri()
    context = spawn(worker, args=(rendezvous, str(tmp_path), *args), nprocs=2, join=False)
    deadline = time.monotonic() + 40
    try:
        while not context.join(timeout=1):
            assert time.monotonic() < deadline, "two-rank solver rejection exceeded 40s"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    return [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]


def _finish_record(root, rank, record):
    (root / f"rank-{rank}.json").write_text(json.dumps(record), encoding="utf-8")
    # Keep the failing rank alive until its peer returns. Early teardown would
    # turn the actual missing agreement into a connection-closed error.
    deadline = time.monotonic() + 2 * REJECTION_TIMEOUT + 5
    while not all((root / f"rank-{peer}.json").exists() for peer in range(2)):
        assert time.monotonic() < deadline, "peer did not return from the bounded solver call"
        time.sleep(.01)


def _io(root, rank, operator, measurements, mapped, prefix):
    output = torch.full(operator.volume_shape, SENTINEL, dtype=operator.dtype)
    if not mapped:
        return measurements.clone(), output
    np.save(root / f"{prefix}-data-{rank}.npy", measurements.numpy())
    source = diffct.NpyStore(root / f"{prefix}-data-{rank}.npy", mode="r")
    sink = diffct.NpyStore.create(root / f"{prefix}-output-{rank}.npy", operator.volume_shape,
                                  dtype=operator.dtype)
    sink.write(tuple(slice(0, size) for size in sink.shape), output)
    sink.flush()
    return source, sink


def _snapshot(value):
    return (value.detach().clone() if isinstance(value, torch.Tensor)
            else value.read(tuple(slice(0, size) for size in value.shape)))


@contextmanager
def _output_writes(output):
    writes = []
    store_type = diffct.TensorStore if isinstance(output, torch.Tensor) else diffct.NpyStore
    original = store_type.write

    def write(store, selected, value, **kwargs):
        target = (store.tensor is output if isinstance(output, torch.Tensor)
                  else store.path == output.path)
        if target:
            writes.append(value.numel())
        return original(store, selected, value, **kwargs)

    with mock.patch.object(store_type, "write", write):
        yield writes


def _rejected(records):
    assert all(record["error"] == "ValueError" for record in records), records
    assert records[0]["message"] == records[1]["message"], records
    for record in records:
        assert record["calls"] == [], ("rejection executed a numerical operator", record)
        assert record["output_writes"] == [], ("rejection wrote caller output", record)
        assert record["output_unchanged"], ("rejection changed caller output", record)
        assert record["input_unchanged"], ("rejection changed measurements", record)
        assert record["elapsed"] < REJECTION_TIMEOUT, ("rejection exceeded its five-second bound", record)


def _workspace_worker(rank, rendezvous, directory, mapped, failure):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=GROUP_TIMEOUT))
    try:
        root = Path(directory)
        case, matrix, data, operator = _real_problem("views")
        local = data[operator.view_slice].clone()
        source, positive_output = _io(root, rank, operator, local, mapped, "positive")
        positive = diffct.cgls_into(operator, source, positive_output, 1,
                                    workspace=root / f"positive-work-{rank}", block_elements=5,
                                    rtol=0., atol=0.)
        expected, history, _ = _cgls_reference(matrix, data.numpy(), 1)
        torch.testing.assert_close(_snapshot(positive.volume),
                                   torch.from_numpy(expected.reshape(case.shape))[operator.volume_slice],
                                   rtol=3e-11, atol=3e-12)
        np.testing.assert_allclose(positive.residual_history, history, rtol=3e-11, atol=3e-12)

        source, output = _io(root, rank, operator, local, mapped, "failure")
        before_input, before_output = _snapshot(source), _snapshot(output)
        workspace = root / f"failure-work-{rank}"
        if failure == "parent-file" and rank == 1:
            blocked = root / "regular-file"
            blocked.write_bytes(b"workspace parent is a real regular file")
            workspace = blocked / "workspace"
            assert blocked.is_file() and not workspace.exists()
        injected = []
        original_create = diffct.NpyStore.create

        def create(path, shape, *, dtype=torch.float32):
            if failure == "state-create" and rank == 1 and Path(path).resolve() == workspace / "p.npy":
                injected.append(str(path))
                raise OSError(errno.ENOSPC, "injected workspace state creation failure", str(path))
            return original_create(path, shape, dtype=dtype)

        error, message = None, None
        with _numerical_calls(operator) as calls, _output_writes(output) as writes, \
                mock.patch.object(diffct.NpyStore, "create", side_effect=create):
            started = time.monotonic()
            try:
                diffct.cgls_into(operator, source, output, 2, workspace=workspace,
                                 block_elements=5, rtol=0., atol=0.)
            except Exception as caught:
                error, message = type(caught).__name__, str(caught)
            elapsed = time.monotonic() - started
        _finish_record(root, rank, dict(error=error, message=message, elapsed=elapsed, calls=calls,
                                       output_writes=writes, injected=len(injected),
                                       output_unchanged=torch.equal(_snapshot(output), before_output),
                                       input_unchanged=torch.equal(_snapshot(source), before_input)))
    finally:
        dist.destroy_process_group()


@GLOO
@pytest.mark.parametrize("mapped", [False, True], ids=["tensor", "mapped"])
@pytest.mark.parametrize("failure", ["parent-file", "state-create"])
def test_two_rank_workspace_failure_rejects_before_operators_or_output_mutation(mapped, failure, tmp_path):
    records = _run_two(_workspace_worker, tmp_path, mapped, failure)
    if failure == "state-create":
        assert [record["injected"] for record in records] == [0, 1], records
    _rejected(records)


def _resume_worker(rank, rendezvous, directory, mapped):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=GROUP_TIMEOUT))
    try:
        root = Path(directory)
        case, matrix, data, operator = _real_problem("views")
        local = data[operator.view_slice].clone()
        source, first_output = _io(root, rank, operator, local, mapped, "partial")
        options = dict(rtol=0., atol=0., checkpoint_directory=root / "checkpoint")
        if mapped:
            partial = diffct.cgls_into(operator, source, first_output, 1,
                                       workspace=root / f"first-work-{rank}", block_elements=5, **options)
        else:
            partial = diffct.cgls(operator, source, 1, **options)
        expected, history, _ = _cgls_reference(matrix, data.numpy(), 1)
        assert partial.iterations == 1
        torch.testing.assert_close(_snapshot(partial.volume),
                                   torch.from_numpy(expected.reshape(case.shape))[operator.volume_slice],
                                   rtol=3e-11, atol=3e-12)
        np.testing.assert_allclose(partial.residual_history, history, rtol=3e-11, atol=3e-12)
        saved = _checkpoint(root / "checkpoint", 1, rank)
        assert json.loads((saved / "metadata.json").read_text())["iteration"] == 1
        assert tuple(np.load(saved / "x.npy").shape) == operator.volume_shape
        if rank == 1:
            (saved / "x.npy").write_bytes(b"")
        dist.barrier()
        assert (saved / "x.npy").stat().st_size == 0 if rank == 1 else (saved / "x.npy").stat().st_size > 0
        before_checkpoint = {path.name: path.read_bytes() for path in saved.iterdir()}
        source, output = _io(root, rank, operator, local, mapped, "resumed")
        before_input, before_output = _snapshot(source), _snapshot(output)
        workspace, publication = root / f"resume-work-{rank}", root / "resume-checkpoint"
        error, message = None, None
        with _numerical_calls(operator) as calls, _output_writes(output) as writes:
            started = time.monotonic()
            try:
                options = dict(rtol=0., atol=0., resume=saved, checkpoint_directory=publication)
                if mapped:
                    diffct.cgls_into(operator, source, output, 2, workspace=workspace,
                                     block_elements=5, **options)
                else:
                    diffct.cgls(operator, source, 2, x0=output, **options)
            except Exception as caught:
                error, message = type(caught).__name__, str(caught)
            elapsed = time.monotonic() - started
        _finish_record(root, rank, dict(error=error, message=message, elapsed=elapsed, calls=calls,
                                       output_writes=writes, workspace_created=workspace.exists(),
                                       publication_created=publication.exists(),
                                       checkpoint_unchanged=before_checkpoint == {
                                           path.name: path.read_bytes() for path in saved.iterdir()},
                                       output_unchanged=torch.equal(_snapshot(output), before_output),
                                       input_unchanged=torch.equal(_snapshot(source), before_input)))
    finally:
        dist.destroy_process_group()


@GLOO
@pytest.mark.parametrize("mapped", [False, True], ids=["tensor", "mapped"])
def test_two_rank_zero_byte_checkpoint_state_rejects_before_resume_updates(mapped, tmp_path):
    records = _run_two(_resume_worker, tmp_path, mapped)
    _rejected(records)
    for record in records:
        assert record["checkpoint_unchanged"], ("resume changed the rejected checkpoint", record)
        assert not record["workspace_created"], ("resume initialized recurrence state before rejection", record)
        assert not record["publication_created"], ("resume published a later checkpoint before rejection", record)
