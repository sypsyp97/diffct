"""Successful workers skip finalizers; worker and teardown errors still fail."""

import atexit
import os
from pathlib import Path
import sys
import time

import pytest
from torch.multiprocessing.spawn import ProcessExitedException, ProcessRaisedException

from tests._multiprocessing import spawn


def _successful_worker(rank, directory):
    # A normal interpreter exit would turn this successful worker into a failure.
    atexit.register(os._exit, 86)
    try:
        print(f"rank-{rank} stdout", end="")
        print(f"rank-{rank} stderr", end="", file=sys.stderr)
        return
    finally:
        (Path(directory) / f"teardown-{rank}").write_text("complete", encoding="utf-8")


def _failing_worker(rank, directory, failure):
    try:
        if failure == "body":
            raise AssertionError("intentional worker failure")
        if failure == "exit":
            sys.exit(17)
        if failure == "interrupt":
            raise KeyboardInterrupt("intentional worker interrupt")
        assert failure == "teardown"
    finally:
        (Path(directory) / f"teardown-{rank}").write_text("complete", encoding="utf-8")
        if failure == "teardown":
            raise RuntimeError("intentional teardown failure")


def _join(context):
    if context is None:
        return
    deadline = time.monotonic() + 30
    try:
        while not context.join(timeout=1):
            assert time.monotonic() < deadline, "spawn regression exceeded 30 seconds"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


@pytest.mark.parametrize("join", [True, False])
def test_successful_workers_finish_teardown_and_flush_before_skipping_finalizers(tmp_path, capfd, join):
    context = spawn(_successful_worker, args=(str(tmp_path),), nprocs=2, join=join)
    _join(context)
    if not join:
        assert [process.exitcode for process in context.processes] == [0, 0]
    captured = capfd.readouterr()
    for rank in range(2):
        assert (tmp_path / f"teardown-{rank}").read_text(encoding="utf-8") == "complete"
        assert f"rank-{rank} stdout" in captured.out
        assert f"rank-{rank} stderr" in captured.err


@pytest.mark.parametrize("join", [True, False])
@pytest.mark.parametrize("failure", ["body", "teardown", "exit", "interrupt"])
def test_worker_failures_reach_parent_after_teardown(tmp_path, join, failure):
    exception = ProcessExitedException if failure == "exit" else ProcessRaisedException
    message = {"body": "intentional worker failure", "teardown": "intentional teardown failure",
               "exit": "exit code 17", "interrupt": "spawned test worker interrupted"}[failure]
    with pytest.raises(exception, match=message) as raised:
        _join(spawn(_failing_worker, args=(str(tmp_path), failure), nprocs=1, join=join))
    assert (tmp_path / "teardown-0").read_text(encoding="utf-8") == "complete"
    if failure == "exit":
        assert raised.value.exit_code == 17
