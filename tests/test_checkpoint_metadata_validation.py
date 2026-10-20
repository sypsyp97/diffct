"""Malformed checkpoint metadata rejects before solver work on every rank."""

from datetime import timedelta
import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch
import torch.distributed as dist

import diffct
from tests.test_execution_features import GLOO
from tests.test_solver_api import _cgls_reference, _checkpoint
from tests.test_solver_checkpoint_edges import _matrix_problem, _numerical_calls, _real_problem
from tests.test_solver_collective_failures import (
    GROUP_TIMEOUT, _finish_record, _io, _output_writes, _run_two, _snapshot,
)


def _solve(mapped, operator, source, output, iterations, workspace, **options):
    if mapped:
        return diffct.cgls_into(operator, source, output, iterations, workspace=workspace,
                               block_elements=5, rtol=0., atol=0., **options)
    return diffct.cgls(operator, source, iterations, x0=output, rtol=0., atol=0., **options)


def _corrupt_metadata(saved, structure):
    path = saved / "metadata.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    if structure == "root":
        document = []
    elif structure == "manifest":
        # Preserve the state names so rejection cannot rely only on key equality.
        document["states"] = list(document["states"])
    else:
        document["states"]["x"] = None
    path.write_text(json.dumps(document), encoding="utf-8")


@pytest.mark.parametrize("mapped", [False, True], ids=["cgls", "cgls_into"])
@pytest.mark.parametrize("structure", ["root", "manifest", "specification"])
def test_malformed_checkpoint_metadata_rejects_before_resume_mutation(mapped, structure, tmp_path):
    operator, measurements = _matrix_problem(torch.float64)
    source, first_output = _io(tmp_path, 0, operator, measurements, mapped, "partial")
    partial = _solve(mapped, operator, source, first_output if mapped else None, 1,
                     tmp_path / "first-work", checkpoint_directory=tmp_path / "checkpoint")
    assert partial.iterations == 1
    saved = _checkpoint(tmp_path / "checkpoint", 1)

    source, control_output = _io(tmp_path, 0, operator, measurements, mapped, "control")
    control = _solve(mapped, operator, source, control_output, 2, tmp_path / "control-work",
                     resume=saved)
    expected, history, _ = _cgls_reference(operator.matrix.numpy(), measurements.numpy(), 2)
    assert control.iterations == 2
    torch.testing.assert_close(_snapshot(control.volume), torch.from_numpy(expected.reshape(operator.volume_shape)),
                               rtol=3e-11, atol=3e-12)
    np.testing.assert_allclose(control.residual_history, history, rtol=3e-11, atol=3e-12)

    _corrupt_metadata(saved, structure)
    before_checkpoint = {path.name: path.read_bytes() for path in saved.iterdir()}
    source, output = _io(tmp_path, 0, operator, measurements, mapped, "rejected")
    before_input, before_output = _snapshot(source), _snapshot(output)
    workspace, publication = tmp_path / "resume-work", tmp_path / "resume-checkpoint"
    with _numerical_calls(operator) as calls, _output_writes(output) as writes:
        with pytest.raises((ValueError, TypeError)):
            _solve(mapped, operator, source, output, 2, workspace,
                   resume=saved, checkpoint_directory=publication)
    assert calls == [], "metadata rejection executed a numerical operator"
    assert writes == [], "metadata rejection wrote caller output"
    assert torch.equal(_snapshot(source), before_input)
    assert torch.equal(_snapshot(output), before_output)
    assert not workspace.exists()
    assert not publication.exists()
    assert before_checkpoint == {path.name: path.read_bytes() for path in saved.iterdir()}


def _metadata_worker(rank, rendezvous, directory, mapped):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=GROUP_TIMEOUT))
    try:
        root = Path(directory)
        case, matrix, data, operator = _real_problem("views")
        local = data[operator.view_slice].clone()
        source, first_output = _io(root, rank, operator, local, mapped, "partial")
        partial = _solve(mapped, operator, source, first_output if mapped else None, 1,
                         root / f"first-work-{rank}", checkpoint_directory=root / "checkpoint")
        assert partial.iterations == 1
        saved = _checkpoint(root / "checkpoint", 1, rank)

        source, control_output = _io(root, rank, operator, local, mapped, "control")
        control = _solve(mapped, operator, source, control_output, 2, root / f"control-work-{rank}",
                         resume=saved)
        expected, history, _ = _cgls_reference(matrix, data.numpy(), 2)
        assert control.iterations == 2
        torch.testing.assert_close(_snapshot(control.volume),
                                   torch.from_numpy(expected.reshape(case.shape))[operator.volume_slice],
                                   rtol=3e-11, atol=3e-12)
        np.testing.assert_allclose(control.residual_history, history, rtol=3e-11, atol=3e-12)
        if rank == 1:
            _corrupt_metadata(saved, "specification")
        dist.barrier()

        before_checkpoint = {path.name: path.read_bytes() for path in saved.iterdir()}
        source, output = _io(root, rank, operator, local, mapped, "rejected")
        before_input, before_output = _snapshot(source), _snapshot(output)
        workspace, publication = root / f"resume-work-{rank}", root / "resume-checkpoint"
        error, message = None, None
        with _numerical_calls(operator) as calls, _output_writes(output) as writes:
            started = time.monotonic()
            try:
                _solve(mapped, operator, source, output, 2, workspace,
                       resume=saved, checkpoint_directory=publication)
            except Exception as caught:
                error, message = type(caught).__name__, str(caught)
            elapsed = time.monotonic() - started
        record = dict(error=error, message=message, elapsed=elapsed, calls=calls,
                      output_writes=writes,
                      input_unchanged=torch.equal(_snapshot(source), before_input),
                      output_unchanged=torch.equal(_snapshot(output), before_output),
                      workspace_created=workspace.exists(), publication_created=publication.exists(),
                      checkpoint_unchanged=before_checkpoint == {
                          path.name: path.read_bytes() for path in saved.iterdir()})
        # Wait until both solver calls returned before testing group usability.
        # Otherwise the barrier could collide with the missing loader agreement.
        _finish_record(root, rank, record)
        reached = torch.tensor(1)
        record["group_error"] = None
        try:
            dist.barrier()
            dist.all_reduce(reached)
            record["group_usable"] = reached.item() == 2
        except RuntimeError as caught:
            record["group_usable"] = False
            record["group_error"] = str(caught)
        (root / f"rank-{rank}.json").write_text(json.dumps(record), encoding="utf-8")
    finally:
        dist.destroy_process_group()


@GLOO
@pytest.mark.parametrize("mapped", [False, True], ids=["cgls", "cgls_into"])
def test_two_rank_malformed_checkpoint_metadata_rejects_together_and_preserves_group(mapped, tmp_path):
    records = _run_two(_metadata_worker, tmp_path, mapped)
    assert all(record["error"] in ("ValueError", "TypeError") for record in records), records
    assert all(record["group_usable"] for record in records), records
    for record in records:
        assert record["calls"] == [], ("metadata rejection executed a numerical operator", record)
        assert record["output_writes"] == [], ("metadata rejection wrote caller output", record)
        assert record["input_unchanged"] and record["output_unchanged"], record
        assert record["checkpoint_unchanged"], record
        assert not record["workspace_created"] and not record["publication_created"], record
