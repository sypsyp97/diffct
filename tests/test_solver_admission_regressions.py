"""Finite streamed data and rank-consistent optional solver branches."""

from contextlib import contextmanager
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
from tests._multiprocessing import spawn

import diffct
from tests.test_detector_surfaces import _case, _grids, _matrix
from tests.test_execution_features import GLOO
from tests.test_solver_api import _DiskAllocations, _SolverStore, _cgls_reference, _checkpoint
from tests.test_solver_checkpoint_edges import _numerical_calls, _real_problem
from tests.test_solver_collective_failures import _finish_record, _output_writes, _snapshot


BLOCK = 5
SENTINEL = 37.
GROUP_TIMEOUT = 6
STORE_KINDS = ("tensor-store", "npy-store", "bounded-store")
BRANCHES = [(name, branch) for name in ("cgls", "sirt", "tv_reconstruction")
            for branch in ("x0", "resume", "checkpoint_directory")]
BRANCHES += [("cgls_into", branch) for branch in ("resume", "checkpoint_directory")]


def _local_problem():
    case = _case("parallel", views=5)
    case.shape, case.detector = (4, 3), (5,)
    u, v = _grids(case)
    matrix = _matrix(case, np.stack((u, v, np.zeros_like(u)), -1))
    data = torch.from_numpy((matrix @ np.linspace(-.3, .9, math.prod(case.shape))).reshape(case.sino_shape))
    operator = diffct.Projector(case.trajectory, case.shape, case.detector, beam=case.beam,
                                voxel_spacing=case.spacing, detector_spacing=case.pitch[0],
                                backend="torch", precision="float64", view_chunk_size=1,
                                volume_chunk_shape=(1, 1))
    return case, matrix, data, operator


def _source(kind, data, root, prefix):
    if kind == "tensor-store":
        return diffct.TensorStore(data.clone())
    path = root / f"{prefix}.npy"
    np.save(path, data.numpy())
    store = diffct.NpyStore(path, mode="r")
    return _SolverStore(store, BLOCK) if kind == "bounded-store" else store


def _backing(source):
    return source.store if isinstance(source, _SolverStore) else source


def _source_snapshot(source):
    return _snapshot(_backing(source))


@contextmanager
def _source_reads(source):
    """Observe the real backing-store reads, including delegated wrappers."""
    backing, reads = _backing(source), []
    original = type(backing).read

    def read(store, selected):
        if store is backing:
            size = math.prod(len(range(*part.indices(length)))
                             for part, length in zip(selected, store.shape))
            reads.append(size)
            assert size <= BLOCK, "measurement read exceeded block_elements"
        return original(store, selected)

    with mock.patch.object(type(backing), "read", read):
        yield reads


@pytest.mark.parametrize("kind", STORE_KINDS)
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")],
                         ids=["nan", "positive-inf", "negative-inf"])
def test_cgls_into_rejects_nonfinite_late_block_before_workspace_or_output(kind, invalid, tmp_path):
    _, _, data, operator = _local_problem()
    data[-1, -1] = invalid
    source = _source(kind, data, tmp_path, "measurements")
    before = _source_snapshot(source)
    output = torch.full(operator.volume_shape, SENTINEL, dtype=operator.dtype)
    workspace = tmp_path / "work"
    with _source_reads(source) as reads, _output_writes(output) as writes, \
            _numerical_calls(operator) as calls, \
            _DiskAllocations(BLOCK, forbid_full=(operator.projection_shape, operator.volume_shape)):
        with pytest.raises(ValueError):
            diffct.cgls_into(operator, source, output, 2, workspace=workspace,
                             block_elements=BLOCK, rtol=0., atol=0.)
    assert len(reads) > 1, "the invalid last block was not reached through bounded reads"
    assert calls == [] and writes == []
    assert not workspace.exists(), "finite validation initialized recurrence storage"
    torch.testing.assert_close(output, torch.full_like(output, SENTINEL), rtol=0, atol=0)
    torch.testing.assert_close(_source_snapshot(source), before, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("kind", STORE_KINDS)
def test_finite_streamed_cgls_matches_independent_matrix_recurrence(kind, tmp_path):
    case, matrix, data, operator = _local_problem()
    source = _source(kind, data, tmp_path, "measurements")
    before = _source_snapshot(source)
    output = torch.full(operator.volume_shape, SENTINEL, dtype=operator.dtype)
    expected, history, _ = _cgls_reference(matrix, data.numpy(), 2)
    with _source_reads(source) as reads:
        result = diffct.cgls_into(operator, source, output, 2, workspace=tmp_path / "work",
                                  block_elements=BLOCK, rtol=0., atol=0.)
    assert result.volume is output and result.iterations == 2 and reads
    torch.testing.assert_close(output, torch.from_numpy(expected.reshape(case.shape)), rtol=3e-11, atol=3e-12)
    np.testing.assert_allclose(result.residual_history, history, rtol=3e-11, atol=3e-12)
    torch.testing.assert_close(_source_snapshot(source), before, rtol=0, atol=0)


def _run_two(worker, tmp_path, *args):
    rendezvous = (tmp_path / "rendezvous").resolve().as_uri()
    context = spawn(worker, args=(rendezvous, str(tmp_path), *args), nprocs=2, join=False)
    deadline = time.monotonic() + 150
    try:
        while not context.join(timeout=1):
            assert time.monotonic() < deadline, "two-rank solver admission exceeded 150s"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    return [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]


def _record(root, rank, record):
    # Persist the actual solver outcome before testing the group's health.
    path = root / f"rank-{rank}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    try:
        dist.barrier()
        record["group_usable"] = True
    except RuntimeError as caught:
        record["group_usable"] = False
        record["barrier_error"] = str(caught)
    _finish_record(root, rank, record)


def _invoke(name, operator, source, output, workspace, iterations, **options):
    if name == "cgls_into":
        return diffct.cgls_into(operator, source, output, iterations, workspace=workspace,
                                block_elements=BLOCK, rtol=0., atol=0., **options)
    return getattr(diffct, name)(operator, source, iterations, rtol=0., atol=0., **options)


def _admission_worker(rank, rendezvous, directory, name, branch, kind, invalid):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=GROUP_TIMEOUT))
    try:
        root = Path(directory)
        _, _, data, operator = _real_problem("views")
        local = data[operator.view_slice].clone()
        options, saved = {}, None
        if branch == "resume":
            checkpoint = root / f"prepared-checkpoint-{rank}"
            partial_output = torch.full(operator.volume_shape, SENTINEL, dtype=operator.dtype)
            partial = _invoke(name, operator, local, partial_output, root / f"prepared-work-{rank}",
                              1, checkpoint_directory=checkpoint)
            assert partial.iterations == 1
            saved = _checkpoint(checkpoint, 1, rank)
            assert (saved / "metadata.json").is_file()
            assert np.isfinite(np.load(saved / "x.npy")).all()
            options["resume"] = saved if rank == 1 else None
        elif branch == "checkpoint_directory":
            options["checkpoint_directory"] = root / f"candidate-checkpoint-{rank}" if rank == 1 else None
        elif branch == "x0":
            options["x0"] = torch.zeros(operator.volume_shape, dtype=operator.dtype) if rank == 1 else None
        else:
            if rank == 1:
                local[-1, -1] = invalid
        source = _source(kind, local, root, f"candidate-data-{rank}") if name == "cgls_into" else local
        before_source = _source_snapshot(source) if name == "cgls_into" else source.clone()
        output = torch.full(operator.volume_shape, SENTINEL, dtype=operator.dtype)
        before_x0 = options["x0"].clone() if options.get("x0") is not None else None
        workspace = root / f"candidate-work-{rank}"
        before_checkpoint = ({path.name: path.read_bytes() for path in saved.iterdir()}
                             if saved is not None else None)
        error, message = None, None
        with _numerical_calls(operator) as calls, _output_writes(output) as writes:
            started = time.monotonic()
            try:
                _invoke(name, operator, source, output, workspace, 2, **options)
            except Exception as caught:
                error, message = type(caught).__name__, str(caught)
            elapsed = time.monotonic() - started
        after_source = _source_snapshot(source) if name == "cgls_into" else source
        _record(root, rank, dict(error=error, message=message, elapsed=elapsed, calls=calls,
                                output_writes=writes, workspace_created=workspace.exists(),
                                publication_created=(root / f"candidate-checkpoint-{rank}").exists(),
                                source_unchanged=torch.allclose(after_source, before_source,
                                                               rtol=0, atol=0, equal_nan=True),
                                output_unchanged=torch.equal(output, torch.full_like(output, SENTINEL)),
                                x0_unchanged=before_x0 is None or torch.equal(options["x0"], before_x0),
                                checkpoint_unchanged=before_checkpoint is None or before_checkpoint == {
                                    path.name: path.read_bytes() for path in saved.iterdir()}))
    finally:
        dist.destroy_process_group()


def _assert_rejected(records):
    assert all(record["error"] == "ValueError" for record in records), records
    assert records[0]["message"] == records[1]["message"], records
    for record in records:
        assert record["elapsed"] < GROUP_TIMEOUT, ("admission waited for the Gloo deadline", record)
        assert record["calls"] == [] and record["output_writes"] == [], record
        assert record["source_unchanged"] and record["output_unchanged"] and record["x0_unchanged"], record
        assert record["checkpoint_unchanged"] and record["group_usable"], record
        assert not record["workspace_created"] and not record["publication_created"], record


@GLOO
@pytest.mark.parametrize("kind,invalid", [("tensor-store", float("nan")),
                                         ("npy-store", float("inf")),
                                         ("bounded-store", -float("inf"))],
                         ids=["tensor-nan", "mapped-positive-inf", "bounded-negative-inf"])
def test_one_rank_nonfinite_measurements_reject_collectively_before_writes(kind, invalid, tmp_path):
    _assert_rejected(_run_two(_admission_worker, tmp_path, "cgls_into", "finite", kind, invalid))


@GLOO
@pytest.mark.parametrize("name,branch", BRANCHES, ids=[f"{name}-{branch}" for name, branch in BRANCHES])
def test_mixed_optional_branch_presence_rejects_before_collective_divergence(name, branch, tmp_path):
    _assert_rejected(_run_two(_admission_worker, tmp_path, name, branch, "npy-store", None))


def _matching_branches_worker(rank, rendezvous, directory, name):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=GROUP_TIMEOUT))
    try:
        root = Path(directory)
        case, matrix, data, operator = _real_problem("views")
        local = data[operator.view_slice].clone()
        checkpoint, publication = root / f"checkpoint-{rank}", root / f"later-checkpoint-{rank}"
        initial = {} if name == "cgls_into" else dict(x0=torch.zeros(operator.volume_shape, dtype=operator.dtype))
        output = torch.full(operator.volume_shape, SENTINEL, dtype=operator.dtype)
        partial = _invoke(name, operator, local, output, root / f"first-work-{rank}",
                          1, checkpoint_directory=checkpoint, **initial)
        saved = _checkpoint(checkpoint, 1, rank)
        assert partial.iterations == 1 and (saved / "metadata.json").is_file()
        output = torch.full(operator.volume_shape, SENTINEL, dtype=operator.dtype)
        result = _invoke(name, operator, local, output, root / f"resume-work-{rank}",
                         2, resume=saved, checkpoint_directory=publication, **initial)
        expected, history, _ = _cgls_reference(matrix, data.numpy(), 2)
        torch.testing.assert_close(_snapshot(result.volume),
                                   torch.from_numpy(expected.reshape(case.shape))[operator.volume_slice],
                                   rtol=3e-11, atol=3e-12)
        np.testing.assert_allclose(result.residual_history, history, rtol=3e-11, atol=3e-12)
        assert (_checkpoint(publication, 2, rank) / "metadata.json").is_file()
        _record(root, rank, dict(iterations=result.iterations, checkpoint=str(checkpoint),
                                resume=str(saved), publication=str(publication)))
    finally:
        dist.destroy_process_group()


@GLOO
@pytest.mark.parametrize("name", ["cgls", "cgls_into"])
def test_matching_branch_presence_allows_distinct_rank_local_checkpoint_paths(name, tmp_path):
    records = _run_two(_matching_branches_worker, tmp_path, name)
    assert all(record["iterations"] == 2 and record["group_usable"] for record in records), records
    for key in ("checkpoint", "resume", "publication"):
        assert records[0][key] != records[1][key]
