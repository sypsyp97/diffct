"""A native kernel-launch failure on one NCCL rank reaches every rank of the same call.

Two NCCL ranks, one CUDA device each, share one ``backend="numba"`` Projector.
After admission agrees, the native kernel launch of one rank raises once
(``torch.OutOfMemoryError`` or a ``ValueError``). Every rank must raise from
that same call (forward, or the adjoint inside backward), well before the
process-group timeout. The failing rank keeps its exception; peers raise a
``RuntimeError`` that names the failing rank. The same Projector and process
group then give the results of a fresh Projector.

The kernel launch has two entry points. A view partition whose whole problem
fits the devices calls ``Projector._project_raw`` / ``Projector._backproject_raw``.
A space partition, and a view partition that streams, launch through the
module functions ``diffct.operators._project_into`` / ``_backproject_into``
and never call the ``_raw`` methods. The fixture replaces both entry points of
the direction under test; one shared counter makes the failure happen once, at
the entry point the call reaches first.

The streamed path also allocates its work buffers for every call, through
``diffct.pipeline._DevicePipeline.empty``: the volume tile of a launch in
``Projector._native_tile`` and the ray buffer in ``Projector._ray_buffers``.
A second fixture makes the first allocation of one of these buffers raise
``torch.OutOfMemoryError`` (spec item F1.8). In the adjoint of a space
partition the ray buffer is allocated before the rays are summed over the
ranks and the volume tile after it, so the two buffers fail at different
points of the call.

Workers write one JSON record per phase, so a blocked or aborted rank leaves a
missing or late record instead of hanging the test.
"""

from contextlib import ExitStack, nullcontext
from datetime import timedelta
import inspect
import json
from pathlib import Path
import re
import time
from unittest import mock

import numpy
import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing import ProcessExitedException
from tests._multiprocessing import spawn

from diffct import Projector, operators, pipeline
from tests.test_native_volume_layout import _assert_atomic_close
from tests.test_reference_backend import _case, _close, _data, _matrix


NCCL = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None
    or torch.cuda.device_count() < 2
    or not dist.is_available() or not dist.is_nccl_available(),
    reason="two NVIDIA CUDA devices and PyTorch NCCL support are required, one device per rank",
)
WORLD_SIZE = 2
# The process-group timeout also covers the warm-up call and admission, so it
# is not shorter. Kernel compilation runs before the process group exists.
# Promptness is a separate bound on the failing call alone: a healthy native
# call on this fixture takes milliseconds once the kernels are compiled, a rank
# that waits for the process-group timeout takes PG_TIMEOUT_SECONDS.
# PROMPT_SECONDS sits between them with room for a loaded node.
PG_TIMEOUT_SECONDS = 20.0
PROMPT_SECONDS = 4.0
# Covers spawn, CUDA and NCCL start, and Numba compilation of the forward and
# adjoint kernels in both ranks, up to the warm-up records.
DEADLINE_SECONDS = 600
# After both warm-up records exist the kernels are compiled. The remaining
# phases take milliseconds when healthy; a blocked rank needs at most a few
# process-group timeouts to raise or abort.
AFTER_WARMUP_SECONDS = 4 * PG_TIMEOUT_SECONDS
INJECTED_TEXT = "injected allocation failure in the native kernel launch"
INJECTED_ERRORS = {"OutOfMemoryError": torch.OutOfMemoryError, "ValueError": ValueError}
ALLOCATION_TEXT = "injected allocation failure of a streamed native work buffer"
PHASES = ("warmup", "failure", "recovery")
# Kernel-launch entry points per direction: (Projector method, module function).
ENTRY_POINTS = {"forward": ("_project_raw", "_project_into"),
                "backward": ("_backproject_raw", "_backproject_into")}
# Streamed work buffers: the Projector method that allocates each of them.
BUFFER_OWNERS = {"tile": "_native_tile", "rays": "_ray_buffers"}


def _injected_failure(state, error_type, stage):
    """Replace the kernel-launch entry points: raise once, then run the real code."""
    method_name, function_name = ENTRY_POINTS[stage]
    method = getattr(Projector, method_name)
    function = getattr(operators, function_name)

    def fail_once(name):
        state["calls"] += 1
        if state["raised"] == 0:
            state["raised"] += 1
            state["entry"] = name
            # Rank 0 of a streamed call can launch pilot kernels while it
            # selects the schedule, before the selection is broadcast.
            state["pilot"] = any(frame.function == "_pilot_candidate" for frame in inspect.stack())
            raise error_type(INJECTED_TEXT)

    def fake_method(self, *args, **kwargs):
        fail_once(method_name)
        return method(self, *args, **kwargs)

    def fake_function(*args, **kwargs):
        fail_once(function_name)
        return function(*args, **kwargs)

    stack = ExitStack()
    stack.enter_context(mock.patch.object(Projector, method_name, fake_method))
    stack.enter_context(mock.patch.object(operators, function_name, fake_function))
    return stack


def _injected_allocation_failure(state, buffer):
    """Replace the streamed buffer allocation: raise once for ``buffer``, then run the real code."""
    owner = BUFFER_OWNERS[buffer]
    empty = pipeline._DevicePipeline.empty

    def fake_empty(self, *args, **kwargs):
        callers, frame = [], inspect.currentframe().f_back
        while frame is not None:
            callers.append(frame.f_code.co_name)
            frame = frame.f_back
        # Rank 0 of a streamed call can allocate pilot buffers while it selects
        # the schedule. Those allocations pass: the failure belongs to the
        # buffers of the selected schedule, on rank 0 as on every other rank.
        if owner in callers and "_pilot_candidate" not in callers:
            state["calls"] += 1
            if state["raised"] == 0:
                state["raised"] += 1
                state["entry"], state["pilot"] = owner, False
                raise torch.OutOfMemoryError(ALLOCATION_TEXT)
        return empty(self, *args, **kwargs)

    return mock.patch.object(pipeline._DevicePipeline, "empty", fake_empty)


def _operator(case, partition, rank, distributed=True):
    return Projector(
        case.trajectory, case.shape, case.detector, beam=case.beam,
        detector_spacing=case.pitch[0], voxel_spacing=case.spacing,
        backend="numba", precision="fast", distributed=distributed, partition=partition,
        devices=[rank],
    )


def _inputs(case, operator):
    # CUDA inputs on the rank's device: a CPU input would stream every call.
    volume = _data(case.shape, dtype=torch.float32, device="cuda", phase=.13)[operator.volume_slice].clone()
    rays = (operator.view_slice, *(slice(0, size) for size in case.detector))
    weight = _data(case.projection_shape, dtype=torch.float32, device="cuda", phase=.31)[rays].clone()
    return volume, weight


def _write(directory, rank, phase, record):
    path = Path(directory) / f"{phase}-rank{rank}.json"
    path.with_suffix(".tmp").write_text(json.dumps(record))
    path.with_suffix(".tmp").replace(path)


def _outcome(function):
    start = time.monotonic()
    try:
        function()
        # An NCCL collective can return before it completes. Wait for the
        # device, so a rank whose collective never completes does not count
        # as a prompt success.
        torch.cuda.synchronize()
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
    """Exact value of a float32 tensor: shape and little-endian bytes."""
    assert tensor.dtype == torch.float32
    return dict(shape=list(tensor.shape),
                hex=tensor.detach().cpu().contiguous().numpy().astype("<f4").tobytes().hex())


def _tensor(bits):
    values = numpy.frombuffer(bytes.fromhex(bits["hex"]), dtype="<f4").reshape(bits["shape"])
    return torch.tensor(values, dtype=torch.float32)


def _worker(rank, rendezvous, directory, partition, stage, failing_rank, error_name, buffer):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    case = _case("parallel", scalar=True)
    # Compile the forward and adjoint kernels before the process group exists,
    # so that no rank waits in a collective while its peer still compiles.
    # A single-rank Projector with the same partition takes the same launch path.
    single = _operator(case, partition, rank, distributed=False)
    _forward_and_gradient(single, *_inputs(case, single))
    torch.cuda.synchronize()
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=WORLD_SIZE,
                            timeout=timedelta(seconds=PG_TIMEOUT_SECONDS))
    try:
        operator = _operator(case, partition, rank)
        volume, weight = _inputs(case, operator)
        # Warm-up: one complete valid forward and backward on every rank.
        warm = _forward_and_gradient(operator, volume, weight)
        _write(directory, rank, "warmup", dict(
            output=_bits(warm[0]), gradient=_bits(warm[1]), backend=operator.backend,
            views=operator.view_slice.indices(case.projection_shape[0])[:2],
            slab=operator.volume_slice[0].indices(case.shape[0])[:2],
        ))

        if stage != "none":
            state = dict(calls=0, raised=0, entry=None, pilot=None)
            patch = (nullcontext() if rank != failing_rank
                     else _injected_failure(state, INJECTED_ERRORS[error_name], stage) if buffer is None
                     else _injected_allocation_failure(state, buffer))
            leaf = volume.clone().requires_grad_()
            dist.barrier()
            if stage == "forward":
                with patch:
                    record = _outcome(lambda: operator.project(leaf))
            else:
                output = operator.project(leaf)
                loss = (output * weight).sum()
                torch.cuda.synchronize()
                dist.barrier()
                with patch:
                    record = _outcome(loss.backward)
            record.update(injected=state["raised"], entry=state["entry"], pilot=state["pilot"])
            _write(directory, rank, "failure", record)

        # The same Projector and process group stay usable after the failure.
        def recover():
            after = _forward_and_gradient(operator, volume, weight)
            fresh_operator = _operator(case, partition, rank)
            fresh = _forward_and_gradient(fresh_operator, volume, weight)
            repeated = _forward_and_gradient(fresh_operator, volume, weight)
            recovery.update(output=_bits(after[0]), gradient=_bits(after[1]),
                            fresh_output=_bits(fresh[0]), fresh_gradient=_bits(fresh[1]),
                            repeated_output=_bits(repeated[0]), repeated_gradient=_bits(repeated[1]))

        recovery = {}
        record = _outcome(recover)
        record.update(recovery)
        _write(directory, rank, "recovery", record)
    finally:
        dist.destroy_process_group()


def _read(directory, phase, rank):
    path = Path(directory) / f"{phase}-rank{rank}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _run(tmp_path, partition, stage, failing_rank=None, error_name=None, buffer=None):
    """Run both ranks; return (records by phase and rank, worker error, how the workers ended)."""
    rendezvous = (tmp_path / "rendezvous").absolute().as_uri()
    context = spawn(_worker, args=(rendezvous, str(tmp_path), partition, stage, failing_rank, error_name, buffer),
                    nprocs=WORLD_SIZE, join=False)
    processes = list(context.processes)
    start = time.monotonic()
    deadline, warm = start + DEADLINE_SECONDS, False
    worker_error = None
    try:
        while not context.join(timeout=1):
            if not warm and all(_read(tmp_path, "warmup", rank) is not None for rank in range(WORLD_SIZE)):
                warm, deadline = True, time.monotonic() + AFTER_WARMUP_SECONDS
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
            if process.is_alive():
                # A rank blocked inside an NCCL collective can ignore SIGTERM.
                process.kill()
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
        ending = (f"ranks {blocked} were still running after {time.monotonic() - start:.0f} s "
                  f"({AFTER_WARMUP_SECONDS:.0f} s allowed after the warm-up records, {DEADLINE_SECONDS} s in total)")
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
        warm = records["warmup"][rank]
        # The forward kernel writes each ray from one thread, so the projection
        # of one configuration and input is reproducible bit for bit.
        assert record["fresh_output"] == record["repeated_output"], (
            f"{label} rank {rank}: two calls of a fresh Projector differ; the bit-exact check has no basis"
        )
        assert record["output"] == record["fresh_output"], (
            f"{label} rank {rank}: projection differs from a fresh Projector"
        )
        assert record["output"] == warm["output"], (
            f"{label} rank {rank}: projection differs from the same Projector's earlier valid call"
        )
        # The adjoint kernel accumulates with atomic additions in an order that
        # changes between runs, so its result is not reproducible bit for bit.
        # The tolerance comes from two calls of the fresh Projector, with the
        # repository's rule for atomic accumulation.
        fresh, repeated = _tensor(record["fresh_gradient"]), _tensor(record["repeated_gradient"])
        _assert_atomic_close(_tensor(record["gradient"]), fresh, repeated)
        _assert_atomic_close(_tensor(warm["gradient"]), fresh, repeated)


def _assert_valid_call_values(records, partition):
    """The valid calls compute the projection and its adjoint, not only equal values."""
    warmups = records["warmup"]
    case = _case("parallel", scalar=True)
    views, rows = case.projection_shape[0], case.shape[0]
    for name, size in (("views", views), ("slab", rows)):
        ranges = [tuple(warmups[rank][name]) for rank in range(WORLD_SIZE)]
        owned = (0, size) if (name == "views") != (partition == "views") else None
        if owned is not None:
            assert ranges == [owned] * WORLD_SIZE, (partition, name, ranges)
        else:
            assert ranges[0][0] == 0 and ranges[-1][1] == size and all(
                first[1] == second[0] for first, second in zip(ranges, ranges[1:])), (partition, name, ranges)

    # Independent per-voxel box intersections, within the repository's
    # tolerance for the native float32 backend.
    matrix = _matrix(case)
    projection = (matrix @ _data(case.shape, phase=.13).reshape(-1)).reshape(case.projection_shape)
    adjoint = (matrix.T @ _data(case.projection_shape, phase=.31).reshape(-1)).reshape(case.shape)
    for rank, record in warmups.items():
        assert record["backend"] == "numba", f"rank {rank} did not use the native backend: {record['backend']}"
        _close(_tensor(record["output"]), projection[slice(*record["views"])], precision="fast")
        # A space rank's loss counts the replicated projection once per rank, so
        # its gradient carries the factor WORLD_SIZE (documented in Projector).
        replicas = WORLD_SIZE if partition == "space" else 1
        _close(_tensor(record["gradient"]), replicas * adjoint[slice(*record["slab"])], precision="fast")


@pytest.mark.cuda
@NCCL
@pytest.mark.parametrize("partition", ["views", "space"])
def test_native_valid_calls_pass_the_recovery_and_value_checks_without_a_failure(partition, tmp_path):
    """Control: the checks applied after a failure hold when no failure is injected."""
    records, worker_error, ending = _run(tmp_path, partition, "none")
    _assert_phase_complete(records, "warmup", partition, ending)
    _assert_recovery(records, partition, ending)
    _assert_valid_call_values(records, partition)
    if worker_error is not None:
        raise worker_error
    assert ending == "all workers exited normally", ending


def _assert_failure_reaches_every_rank(outcome, label, partition, failing_rank, error_name, entries, text):
    """One injected failure on ``failing_rank``: every rank raises promptly, then recovers."""
    records, worker_error, ending = outcome
    _assert_phase_complete(records, "warmup", label, ending)

    # The failing call: every rank raises, promptly, with the required error.
    _assert_phase_complete(records, "failure", label, ending)
    failures = records["failure"]
    injected = {rank: record["injected"] for rank, record in failures.items()}
    assert injected == {rank: int(rank == failing_rank) for rank in range(WORLD_SIZE)}, (
        f"{label}: fixture must fail once, on rank {failing_rank} only: {injected}"
    )
    assert failures[failing_rank]["entry"] in entries, failures[failing_rank]
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
            assert text in record["message"], f"{label}: failing rank lost its message: {record}"
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


@pytest.mark.cuda
@NCCL
@pytest.mark.parametrize("partition", ["views", "space"])
@pytest.mark.parametrize("stage", ["forward", "backward"])
@pytest.mark.parametrize("failing_rank", [0, 1])
@pytest.mark.parametrize("error_name", list(INJECTED_ERRORS))
def test_native_kernel_failure_on_one_rank_raises_promptly_on_every_rank(
        partition, stage, failing_rank, error_name, tmp_path):
    label = f"{stage}/{partition}/failing rank {failing_rank}/{error_name}"
    outcome = _run(tmp_path, partition, stage, failing_rank, error_name)
    _assert_failure_reaches_every_rank(outcome, label, partition, failing_rank, error_name,
                                       ENTRY_POINTS[stage], INJECTED_TEXT)


@pytest.mark.cuda
@NCCL
@pytest.mark.parametrize("stage", ["forward", "backward"])
@pytest.mark.parametrize("failing_rank", [0, 1])
@pytest.mark.parametrize("buffer", list(BUFFER_OWNERS))
def test_native_streamed_buffer_allocation_failure_on_one_rank_raises_promptly_on_every_rank(
        buffer, failing_rank, stage, tmp_path):
    """F1.8: a space partition always streams, so every call allocates both buffers."""
    label = f"{stage}/space/failing rank {failing_rank}/{buffer} buffer allocation"
    outcome = _run(tmp_path, "space", stage, failing_rank, "OutOfMemoryError", buffer)
    _assert_failure_reaches_every_rank(outcome, label, "space", failing_rank, "OutOfMemoryError",
                                       (BUFFER_OWNERS[buffer],), ALLOCATION_TEXT)
