"""A numeric-stage failure on one rank reaches every rank of the same call.

Two Gloo ranks share one portable Projector. After admission agrees, one rank's
ray/voxel interval computation raises once (``torch.OutOfMemoryError`` or a
``ValueError``). Every rank must raise from that same call (forward, or the
adjoint inside backward), well before the process-group timeout. The failing
rank keeps its exception; peers raise a ``RuntimeError`` that names the failing
rank. The same Projector and process group then produce results bit-identical
to a fresh Projector and to its own result before the failure.

Values of the valid calls are checked in four ways. Three hold on every
platform: bit-exact equalities inside one run, bit-exact equality with a
non-distributed Projector (view partition), and closeness to an independent
cell matrix. The fourth, bit-exact values recorded on the pre-fix tree
(d260ec1), applies only on a recorded platform and torch build.

Workers write one JSON record per phase, so a blocked rank leaves a missing or
late record instead of hanging the test.
"""

from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import platform
import re
import time
from unittest import mock

import numpy
import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing import ProcessExitedException
from tests._multiprocessing import spawn

from diffct import Projector, backends
from tests.test_reference_backend import _case, _close, _data, _matrix


GLOO = pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(),
                          reason="real Gloo process groups required")
WORLD_SIZE = 2
# The process-group timeout also covers rendezvous, warm-up and admission, so it
# is long. Promptness is a separate bound on the failing call alone: a healthy
# call takes milliseconds, a rank that waits for the process-group timeout
# takes PG_TIMEOUT_SECONDS. PROMPT_SECONDS sits between them with room for a
# loaded or emulated CI machine.
PG_TIMEOUT_SECONDS = 20.0
PROMPT_SECONDS = 4.0
# Covers spawn, imports and one full process-group timeout in a blocked rank.
DEADLINE_SECONDS = 90
INJECTED_TEXT = "injected allocation failure in the numeric stage"
INJECTED_ERRORS = {"OutOfMemoryError": torch.OutOfMemoryError, "ValueError": ValueError}
PHASES = ("warmup", "failure", "recovery")

# SHA-256 of the local output and input gradient of the valid distributed call,
# per (partition, rank), recorded on the pre-fix tree (d260ec1). Last bits
# depend on the platform and the torch build, so the table is keyed by both. Only
# natively recorded platforms have an entry. A platform without an entry runs
# every other check.
_PRE_FIX_DIGESTS = {
    "Darwin-arm64-torch2.10.0": {
        ("views", 0): dict(
            output="b824fb9f9309663ce47417dfdb004fef4265589ea6b0e73bd09d96135df8d39a",
            gradient="22fffaa4d946640d8649f6eb2733e3c5977f156ed4287c79400df1b71e8c58ee"),
        ("views", 1): dict(
            output="88f08aedf2ea1800b775763166855e69f50525a48bbdeaa3544e331ea0236eb7",
            gradient="22fffaa4d946640d8649f6eb2733e3c5977f156ed4287c79400df1b71e8c58ee"),
        ("space", 0): dict(
            output="bc43b0532985d998666d0546cea4f1ddd62d94f2b11cc548f1d9fc42c3bc419c",
            gradient="7f69f0a57d0f3236029d50bede9b137aa540bf4e08066b4ac003bf5f71d37271"),
        ("space", 1): dict(
            output="bc43b0532985d998666d0546cea4f1ddd62d94f2b11cc548f1d9fc42c3bc419c",
            gradient="5480dcd13a235ebba2be75366ffc4306251e6a33859bdb7b10b4a53ca04e6b56"),
    },
}


def _platform_key():
    return f"{platform.system()}-{platform.machine()}-torch{torch.__version__}"


def _injected_failure(state, error_type):
    """Replace ``_intervals``: raise once, then run the real method."""
    original = backends.ReferenceProjector._intervals

    def fake(self, *args, **kwargs):
        state["calls"] += 1
        if state["raised"] == 0:
            state["raised"] += 1
            raise error_type(INJECTED_TEXT)
        return original(self, *args, **kwargs)

    return mock.patch.object(backends.ReferenceProjector, "_intervals", fake)


def _operator(case, partition, distributed=True):
    return Projector(
        case.trajectory, case.shape, case.detector, beam=case.beam,
        detector_spacing=case.pitch[0], voxel_spacing=case.spacing,
        backend="torch", precision="float64", distributed=distributed, partition=partition,
    )


def _inputs(case, operator):
    volume = _data(case.shape, phase=.13)[operator.volume_slice].clone()
    rays = (operator.view_slice, *(slice(0, size) for size in case.detector))
    weight = _data(case.projection_shape, phase=.31)[rays].clone()
    return volume, weight


def _write(directory, rank, phase, record):
    path = Path(directory) / f"{phase}-rank{rank}.json"
    path.with_suffix(".tmp").write_text(json.dumps(record))
    path.with_suffix(".tmp").replace(path)


def _outcome(function):
    start = time.monotonic()
    try:
        function()
    except BaseException as error:  # the record carries every failure type
        return dict(raised=True, type=type(error).__name__,
                    runtime_error=isinstance(error, RuntimeError), message=str(error),
                    elapsed=time.monotonic() - start)
    return dict(raised=False, elapsed=time.monotonic() - start)


def _forward_and_gradient(operator, volume, weight):
    leaf = volume.clone().requires_grad_()
    output = operator.project(leaf)
    (output * weight).sum().backward()
    return output.detach(), leaf.grad


def _bits(tensor):
    """Exact value of a float64 tensor: shape and little-endian bytes."""
    assert tensor.dtype == torch.float64
    return dict(shape=list(tensor.shape),
                hex=tensor.detach().cpu().contiguous().numpy().astype("<f8").tobytes().hex())


def _tensor(bits):
    values = numpy.frombuffer(bytes.fromhex(bits["hex"]), dtype="<f8").reshape(bits["shape"])
    return torch.tensor(values, dtype=torch.float64)


def _digest(bits):
    return hashlib.sha256(f"{bits['shape']}:{bits['hex']}".encode()).hexdigest()


def _worker(rank, rendezvous, directory, partition, stage, failing_rank, error_name):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=WORLD_SIZE,
                            timeout=timedelta(seconds=PG_TIMEOUT_SECONDS))
    try:
        case = _case("parallel")
        operator = _operator(case, partition)
        volume, weight = _inputs(case, operator)
        # Warm-up: one complete valid forward and backward on every rank.
        warm = _forward_and_gradient(operator, volume, weight)
        record = dict(output=_bits(warm[0]), gradient=_bits(warm[1]),
                      views=operator.view_slice.indices(case.projection_shape[0])[:2],
                      slab=operator.volume_slice[0].indices(case.shape[0])[:2])
        if partition == "views":
            # A view rank holds the whole volume: the same rows of a
            # non-distributed Projector in this process are the reference.
            single = _operator(case, partition, distributed=False)
            record["single_output"] = _bits(single.project(volume)[operator.view_slice])
        _write(directory, rank, "warmup", record)
        if stage != "none":
            state = dict(calls=0, raised=0)
            patch = (_injected_failure(state, INJECTED_ERRORS[error_name]) if rank == failing_rank
                     else nullcontext())
            leaf = volume.clone().requires_grad_()
            dist.barrier()
            if stage == "forward":
                with patch:
                    record = _outcome(lambda: operator.project(leaf))
            else:
                output = operator.project(leaf)
                loss = (output * weight).sum()
                dist.barrier()
                with patch:
                    record = _outcome(loss.backward)
            record["injected"] = state["raised"]
            _write(directory, rank, "failure", record)

        # The same Projector and process group stay usable after the failure.
        def recover():
            after = _forward_and_gradient(operator, volume, weight)
            fresh = _forward_and_gradient(_operator(case, partition), volume, weight)
            recovery.update(output=_bits(after[0]), gradient=_bits(after[1]),
                            fresh_output=_bits(fresh[0]), fresh_gradient=_bits(fresh[1]))

        recovery = {}
        record = _outcome(recover)
        record.update(recovery)
        _write(directory, rank, "recovery", record)
    finally:
        dist.destroy_process_group()


def _read(directory, phase, rank):
    path = Path(directory) / f"{phase}-rank{rank}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _run(tmp_path, partition, stage, failing_rank=None, error_name=None):
    """Run both ranks; return (records by phase and rank, worker error, how the workers ended)."""
    rendezvous = (tmp_path / "rendezvous").absolute().as_uri()
    context = spawn(_worker, args=(rendezvous, str(tmp_path), partition, stage, failing_rank, error_name),
                    nprocs=WORLD_SIZE, join=False)
    processes = list(context.processes)
    deadline = time.monotonic() + DEADLINE_SECONDS
    worker_error = None
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                break
    except Exception as error:  # reported after the per-rank records
        worker_error = error
    finally:
        blocked = [rank for rank, process in enumerate(processes) if process.is_alive()]
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    # ``join`` stops the remaining ranks when one rank ends with an error, so
    # only the rank named by the error is known to have ended by itself.
    if isinstance(worker_error, ProcessExitedException):
        cause = (f"signal {worker_error.signal_name}" if worker_error.signal_name
                 else f"exit code {worker_error.exit_code}")
        ending = (f"the worker process of rank {worker_error.error_index} exited with {cause}; "
                  "the remaining ranks were then stopped")
    elif worker_error is not None:
        ending = f"a worker raised: {worker_error}"
    elif blocked:
        ending = f"ranks {blocked} were still running after {DEADLINE_SECONDS} s"
    else:
        ending = "all workers exited normally"
    records = {phase: {rank: _read(tmp_path, phase, rank) for rank in range(WORLD_SIZE)}
               for phase in PHASES}
    return records, worker_error, ending


def _assert_phase_complete(records, phase, label, ending):
    missing = [rank for rank, record in records[phase].items() if record is None]
    assert not missing, f"{label}: ranks {missing} wrote no {phase} record; {ending}"


def _assert_recovery(records, label, ending):
    _assert_phase_complete(records, "recovery", label, ending)
    for rank, record in records["recovery"].items():
        assert not record["raised"], f"{label} rank {rank}: following valid call raised: {record}"
        after = dict(output=record["output"], gradient=record["gradient"])
        fresh = dict(output=record["fresh_output"], gradient=record["fresh_gradient"])
        warm = {name: records["warmup"][rank][name] for name in ("output", "gradient")}
        assert after == fresh, f"{label} rank {rank}: result differs from a fresh Projector"
        assert after == warm, (
            f"{label} rank {rank}: result differs from the same Projector's earlier valid call"
        )


def _assert_valid_call_values(records, partition):
    """Success-path values of the warm-up call (recovery equals it bit-exactly)."""
    warmups = records["warmup"]
    case = _case("parallel")
    views, rows = case.projection_shape[0], case.shape[0]
    for name, size in (("views", views), ("slab", rows)):
        ranges = [tuple(warmups[rank][name]) for rank in range(WORLD_SIZE)]
        owned = (0, size) if (name == "views") != (partition == "views") else None
        if owned is not None:
            assert ranges == [owned] * WORLD_SIZE, (partition, name, ranges)
        else:
            assert ranges[0][0] == 0 and ranges[-1][1] == size and all(
                first[1] == second[0] for first, second in zip(ranges, ranges[1:])), (partition, name, ranges)

    # Bit-exact relations that do not depend on the platform.
    if partition == "views":
        for rank, record in warmups.items():
            assert record["output"] == record["single_output"], (
                f"views rank {rank}: local output differs from the same rows of a non-distributed Projector"
            )
        assert warmups[0]["gradient"] == warmups[1]["gradient"], "view ranks hold different input gradients"
    else:
        assert warmups[0]["output"] == warmups[1]["output"], "space ranks hold different replicated outputs"

    # Independent per-voxel box intersections, within the repository tolerance.
    matrix = _matrix(case)
    projection = (matrix @ _data(case.shape, phase=.13).reshape(-1)).reshape(case.projection_shape)
    adjoint = (matrix.T @ _data(case.projection_shape, phase=.31).reshape(-1)).reshape(case.shape)
    for rank, record in warmups.items():
        _close(_tensor(record["output"]), projection[slice(*record["views"])])
        # A space rank's loss counts the replicated projection once per rank, so
        # its gradient carries the factor WORLD_SIZE (documented in Projector).
        replicas = WORLD_SIZE if partition == "space" else 1
        _close(_tensor(record["gradient"]), replicas * adjoint[slice(*record["slab"])])

    # Bit-exact pre-fix values, on recorded platforms only.
    recorded = _PRE_FIX_DIGESTS.get(_platform_key())
    if recorded is not None:
        for rank, record in warmups.items():
            actual = dict(output=_digest(record["output"]), gradient=_digest(record["gradient"]))
            assert actual == recorded[partition, rank], (
                f"{partition} rank {rank}: valid distributed call differs from the values recorded "
                f"on the pre-fix tree for {_platform_key()}"
            )


@GLOO
@pytest.mark.parametrize("partition", ["views", "space"])
def test_distributed_success_path_keeps_pre_fix_values(partition, tmp_path):
    records, worker_error, ending = _run(tmp_path, partition, "none")
    _assert_phase_complete(records, "warmup", partition, ending)
    _assert_recovery(records, partition, ending)
    _assert_valid_call_values(records, partition)
    if worker_error is not None:
        raise worker_error
    assert ending == "all workers exited normally", ending


@GLOO
@pytest.mark.parametrize("partition", ["views", "space"])
@pytest.mark.parametrize("stage", ["forward", "backward"])
@pytest.mark.parametrize("failing_rank", [0, 1])
@pytest.mark.parametrize("error_name", list(INJECTED_ERRORS))
def test_numeric_failure_on_one_rank_raises_promptly_on_every_rank(
        partition, stage, failing_rank, error_name, tmp_path):
    label = f"{stage}/{partition}/failing rank {failing_rank}/{error_name}"
    records, worker_error, ending = _run(tmp_path, partition, stage, failing_rank, error_name)
    _assert_phase_complete(records, "warmup", label, ending)

    # The failing call: every rank raises, promptly, with the required error.
    _assert_phase_complete(records, "failure", label, ending)
    failures = records["failure"]
    injected = {rank: record["injected"] for rank, record in failures.items()}
    assert injected == {rank: int(rank == failing_rank) for rank in range(WORLD_SIZE)}, (
        f"{label}: fixture must fail the numeric stage once, on rank {failing_rank} only: {injected}"
    )
    succeeded = [rank for rank, record in failures.items() if not record["raised"]]
    assert not succeeded, (
        f"{label}: mixed outcome, ranks {succeeded} returned success from a call "
        f"that failed on rank {failing_rank}: {failures}"
    )
    slow = {rank: record["elapsed"] for rank, record in failures.items()
            if record["elapsed"] >= PROMPT_SECONDS}
    assert not slow, (
        f"{label}: ranks raised only after {slow} s; process-group timeout is "
        f"{PG_TIMEOUT_SECONDS} s, every rank must raise within {PROMPT_SECONDS} s: {failures}"
    )
    for rank, record in failures.items():
        if rank == failing_rank:
            assert record["type"] == error_name, f"{label}: failing rank lost its exception type: {record}"
            assert INJECTED_TEXT in record["message"], f"{label}: failing rank lost its message: {record}"
            continue
        assert record["runtime_error"], f"{label}: peer rank {rank} raised a non-RuntimeError: {record}"
        # The failing rank number must follow the word "rank" ("rank 1", "ranks [1]",
        # "rank(s): 1"); the same digit elsewhere in the message does not count.
        assert re.search(rf"\brank\D{{0,16}}{failing_rank}\b", record["message"], re.IGNORECASE), (
            f"{label}: peer rank {rank} error does not name failing rank {failing_rank}: {record['message']!r}"
        )

    _assert_recovery(records, label, ending)
    _assert_valid_call_values(records, partition)
    if worker_error is not None:
        raise worker_error
    assert ending == "all workers exited normally", ending
