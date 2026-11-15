"""Native ``beam="parallel3d"`` on two NVIDIA devices and over two NCCL ranks.

Each test needs two real NVIDIA CUDA devices. Expected values come from one
process of the portable float64 backend. No test starts a remote process.

The last test injects one allocation failure of a streamed work buffer on one
rank of a space partition. A space partition always streams, so every call
allocates the volume tile of a launch in ``Projector._native_tile`` and the ray
buffer in ``Projector._ray_buffers``, both through
``diffct.pipeline._DevicePipeline.empty``. Every rank must raise from that same
call, well before the process-group timeout. The failing rank keeps its
exception; the peer raises a ``RuntimeError`` that names the failing rank. The
same Projector and process group then give the results of a fresh Projector.
Workers write one JSON record per phase, so a blocked or aborted rank leaves a
missing or late record instead of hanging the test.
"""

from contextlib import nullcontext
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
from numba.core import config
from torch.multiprocessing import ProcessExitedException

from diffct import Projector, pipeline
from tests._multiprocessing import spawn
from tests._native_per_axis_spacing import (
    GEOMETRY_ATOL, GEOMETRY_RTOL, REAL_CUDA_REQUIRED, close, data, native_only, operator, surface,
)
from tests.test_parallel3d_native import _scan


pytestmark = [pytest.mark.cuda, REAL_CUDA_REQUIRED]
TWO_GPUS = pytest.mark.skipif(
    config.ENABLE_CUDASIM or not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="two real NVIDIA CUDA devices are required",
)
NCCL = pytest.mark.skipif(
    not dist.is_available() or not dist.is_nccl_available(), reason="NCCL is required",
)
OPERATIONS = ("project", "backproject")


def _reference(scan, detector, parameters, operation, value, weight):
    """One-process portable float64 output and its data and geometry gradients."""
    reference = operator(scan, backend="torch", precision="float64", detector_surface=detector)
    reference_data = value.double().requires_grad_()
    expected = getattr(reference, operation)(reference_data)
    gradients = torch.autograd.grad((expected * weight.double()).sum(),
                                    (reference_data, *scan.trajectory, *parameters))
    return expected, gradients


@TWO_GPUS
@pytest.mark.parametrize("kind", ["flat", "parameterized"])
def test_two_devices_in_one_process_match_portable(kind):
    scan = _scan(learnable=True)
    detector, parameters = surface(kind, learnable=True)
    image, rays = data(scan.shape), data(scan.projection_shape, phase=.31)
    for operation, value, weight in zip(OPERATIONS, (image, rays), (rays, image)):
        expected, wanted = _reference(scan, detector, parameters, operation, value, weight)
        native_data = value.cuda(0).requires_grad_()
        with native_only() as bridge:
            native = operator(scan, detector_surface=detector, devices=[0, 1])
            bridge.reset_mock()
            actual = getattr(native, operation)(native_data)
            gradients = torch.autograd.grad((actual * weight.cuda(0)).sum(),
                                            (native_data, *scan.trajectory, *parameters))
            assert native.backend == "numba" and bridge.call_count > 0
            used = {call.args[0].device.index for call in bridge.call_args_list if call.args[0].is_cuda}
            assert used == {0, 1}, "both requested devices must execute native work"
        close(actual, expected, kind=kind)
        close(gradients[0], wanted[0], kind=kind)
        for actual_gradient, expected_gradient in zip(gradients[1:], wanted[1:]):
            close(actual_gradient, expected_gradient, rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL)


def _nccl_worker(rank, rendezvous, kind, partition):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        scan = _scan(learnable=True)
        detector, parameters = surface(kind, learnable=True)
        image, rays = data(scan.shape, phase=.23), data(scan.projection_shape, phase=.47)
        for operation, full_data, full_weight in zip(OPERATIONS, (image, rays), (rays, image)):
            expected, wanted = _reference(scan, detector, parameters, operation, full_data, full_weight)
            with native_only() as bridge:
                native = Projector(
                    scan.trajectory, scan.shape, scan.detector, beam="parallel3d", backend="numba",
                    voxel_spacing=scan.spacing, detector_spacing=scan.pitch,
                    detector_surface=detector, devices=[rank], distributed=True, partition=partition,
                )
                assert native.world_size == 2 and native.rank == rank
                input_slice = native.volume_slice if operation == "project" else native.view_slice
                output_slice = native.view_slice if operation == "project" else native.volume_slice
                value = full_data[input_slice].clone().to(rank).requires_grad_()
                weight = full_weight[output_slice].to(rank)
                actual = getattr(native, operation)(value)
                # A replicated output counts once in the global loss.
                replicas = native.space_world_size if operation == "project" else native.view_world_size
                gradients = torch.autograd.grad((actual * weight).sum() / replicas,
                                                (value, *scan.trajectory, *parameters))
                assert native.backend == "numba" and bridge.call_count > 0
            close(actual, expected[output_slice], kind=kind)
            close(gradients[0], wanted[0][input_slice], kind=kind)
            for actual_gradient, expected_gradient in zip(gradients[1:], wanted[1:]):
                close(actual_gradient, expected_gradient, rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL)
    finally:
        dist.destroy_process_group()


@TWO_GPUS
@NCCL
@pytest.mark.parametrize("kind", ["flat", "parameterized"])
@pytest.mark.parametrize("partition", ["views", "space"])
def test_nccl_views_and_space_partitions_match_one_process(tmp_path, kind, partition):
    rendezvous = (tmp_path / "parallel3d-native-nccl").absolute().as_uri()
    context = spawn(_nccl_worker, args=(rendezvous, kind, partition), nprocs=2, join=False)
    deadline = time.monotonic() + 150
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise AssertionError("parallel3d native NCCL execution exceeded 150 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


WORLD_SIZE = 2
# The process-group timeout also covers the warm-up call and admission, so it
# is not shorter. Kernel compilation runs before the process group exists.
# Promptness is a separate bound on the failing call alone: a healthy native
# call on this fixture takes milliseconds once the kernels are compiled, a rank
# that waits for the process-group timeout takes PG_TIMEOUT_SECONDS.
# PROMPT_SECONDS sits between them with room for a loaded node.
PG_TIMEOUT_SECONDS = 20.0
PROMPT_SECONDS = 4.0
# Covers spawn, CUDA and NCCL start, the float64 reference and Numba
# compilation of the forward and adjoint kernels in both ranks, up to the
# warm-up records.
DEADLINE_SECONDS = 600
# After both warm-up records exist the kernels are compiled. The remaining
# phases take milliseconds when healthy; a blocked rank needs at most a few
# process-group timeouts to raise or abort.
AFTER_WARMUP_SECONDS = 4 * PG_TIMEOUT_SECONDS
ALLOCATION_TEXT = "injected allocation failure of a streamed native work buffer"
PHASES = ("warmup", "failure", "recovery")
# Streamed work buffers: the Projector method that allocates each of them.
BUFFER_OWNERS = {"tile": "_native_tile", "rays": "_ray_buffers"}


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
                state["entry"] = owner
                raise torch.OutOfMemoryError(ALLOCATION_TEXT)
        return empty(self, *args, **kwargs)

    return mock.patch.object(pipeline._DevicePipeline, "empty", fake_empty)


def _space_operator(scan, rank, distributed=True):
    return Projector(
        scan.trajectory, scan.shape, scan.detector, beam="parallel3d", backend="numba",
        voxel_spacing=scan.spacing, detector_spacing=scan.pitch,
        devices=[rank], distributed=distributed, partition="space",
    )


def _space_inputs(scan, native):
    # CUDA inputs on the rank's device, cut to the rank's part of the problem.
    volume = data(scan.shape, phase=.23, device="cuda")[native.volume_slice].clone()
    weight = data(scan.projection_shape, phase=.47, device="cuda")[native.view_slice].clone()
    return volume, weight


def _forward_and_gradient(native, volume, weight):
    leaf = volume.clone().requires_grad_()
    output = native.project(leaf)
    # A space rank holds the whole projection, so it counts once in the global loss.
    ((output * weight).sum() / native.space_world_size).backward()
    return output.detach(), leaf.grad


def _write(directory, rank, phase, record):
    path = Path(directory) / f"{phase}-rank{rank}.json"
    path.with_suffix(".tmp").write_text(json.dumps(record))
    path.with_suffix(".tmp").replace(path)


def _read(directory, phase, rank):
    path = Path(directory) / f"{phase}-rank{rank}.json"
    return json.loads(path.read_text()) if path.exists() else None


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


def _bits(tensor):
    """Exact value of a float32 tensor: shape and little-endian bytes."""
    assert tensor.dtype == torch.float32
    return dict(shape=list(tensor.shape),
                hex=tensor.detach().cpu().contiguous().numpy().astype("<f4").tobytes().hex())


def _tensor(bits):
    values = numpy.frombuffer(bytes.fromhex(bits["hex"]), dtype="<f4").reshape(bits["shape"])
    return torch.tensor(values, dtype=torch.float32)


def _allocation_failure_worker(rank, rendezvous, directory, stage, failing_rank, buffer):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    scan = _scan()
    # Compile the forward and adjoint kernels before the process group exists,
    # so that no rank waits in a collective while its peer still compiles.
    # A single-rank Projector with the same partition takes the same launch path.
    single = _space_operator(scan, rank, distributed=False)
    _forward_and_gradient(single, *_space_inputs(scan, single))
    torch.cuda.synchronize()
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=WORLD_SIZE,
                            timeout=timedelta(seconds=PG_TIMEOUT_SECONDS))
    try:
        native = _space_operator(scan, rank)
        volume, weight = _space_inputs(scan, native)
        # Warm-up: one complete valid forward and backward on every rank.
        warm = _forward_and_gradient(native, volume, weight)
        _write(directory, rank, "warmup", dict(
            output=_bits(warm[0]), gradient=_bits(warm[1]), backend=native.backend,
            world=[native.world_size, native.space_world_size],
            views=native.view_slice.indices(scan.projection_shape[0])[:2],
            slab=native.volume_slice[0].indices(scan.shape[0])[:2],
        ))

        state = dict(calls=0, raised=0, entry=None)
        patch = _injected_allocation_failure(state, buffer) if rank == failing_rank else nullcontext()
        leaf = volume.clone().requires_grad_()
        dist.barrier()
        if stage == "forward":
            with patch:
                record = _outcome(lambda: native.project(leaf))
        else:
            output = native.project(leaf)
            loss = (output * weight).sum() / native.space_world_size
            torch.cuda.synchronize()
            dist.barrier()
            with patch:
                record = _outcome(loss.backward)
        record.update(injected=state["raised"], entry=state["entry"])
        _write(directory, rank, "failure", record)

        # The same Projector and process group stay usable after the failure.
        def recover():
            after = _forward_and_gradient(native, volume, weight)
            fresh_operator = _space_operator(scan, rank)
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


def _run_allocation_failure(tmp_path, stage, failing_rank, buffer):
    """Run both ranks; return (records by phase and rank, worker error, how the workers ended)."""
    rendezvous = (tmp_path / "rendezvous").absolute().as_uri()
    context = spawn(_allocation_failure_worker,
                    args=(rendezvous, str(tmp_path), stage, failing_rank, buffer),
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


def _assert_failure_reaches_every_rank(records, label, failing_rank, buffer, ending):
    """The failing call: every rank raises, promptly, with the required error."""
    _assert_phase_complete(records, "failure", label, ending)
    failures = records["failure"]
    injected = {rank: record["injected"] for rank, record in failures.items()}
    assert injected == {rank: int(rank == failing_rank) for rank in range(WORLD_SIZE)}, (
        f"{label}: fixture must fail once, on rank {failing_rank} only: {injected}"
    )
    assert failures[failing_rank]["entry"] == BUFFER_OWNERS[buffer], failures[failing_rank]
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
            assert record["type"] == "OutOfMemoryError", f"{label}: failing rank lost its exception type: {record}"
            assert ALLOCATION_TEXT in record["message"], f"{label}: failing rank lost its message: {record}"
            continue
        assert record["runtime_error"], f"{label}: peer rank {rank} raised a non-RuntimeError: {record}"
        # The failing rank number must follow the word "rank" ("rank 1", "ranks [1]",
        # "rank(s): 1"); the same digit elsewhere in the message does not count.
        assert re.search(rf"\brank\D{{0,16}}{failing_rank}\b", record["message"], re.IGNORECASE), (
            f"{label}: peer rank {rank} error does not name failing rank {failing_rank}: {record['message']!r}"
        )


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
        # changes between runs, so its result is compared within the tolerance
        # this file uses for native float32 values.
        fresh = _tensor(record["fresh_gradient"])
        close(_tensor(record["repeated_gradient"]), fresh)
        close(_tensor(record["gradient"]), fresh)
        close(_tensor(warm["gradient"]), fresh)


def _assert_valid_call_values(records):
    """The valid calls compute the projection and its adjoint, not only equal values."""
    scan = _scan(learnable=True)  # the reference helper also differentiates the geometry
    views, rows = scan.projection_shape[0], scan.shape[0]
    image, rays = data(scan.shape, phase=.23), data(scan.projection_shape, phase=.47)
    projection, (adjoint, *_) = _reference(scan, None, (), "project", image, rays)
    warmups = records["warmup"]
    slabs = [tuple(warmups[rank]["slab"]) for rank in range(WORLD_SIZE)]
    assert slabs[0][0] == 0 and slabs[-1][1] == rows and all(
        first[0] < first[1] == second[0] for first, second in zip(slabs, slabs[1:])), slabs
    for rank, record in warmups.items():
        assert record["backend"] == "numba", f"rank {rank} did not use the native backend: {record['backend']}"
        assert record["world"] == [WORLD_SIZE, WORLD_SIZE], f"rank {rank} is not one of two space ranks: {record}"
        assert tuple(record["views"]) == (0, views), f"rank {rank} does not hold every view: {record['views']}"
        close(_tensor(record["output"]), projection)
        close(_tensor(record["gradient"]), adjoint[slice(*record["slab"])])


@TWO_GPUS
@NCCL
@pytest.mark.parametrize("stage", ["forward", "backward"])
@pytest.mark.parametrize("failing_rank", [0, 1])
@pytest.mark.parametrize("buffer", list(BUFFER_OWNERS))
def test_native_streamed_buffer_allocation_failure_on_one_rank_raises_promptly_on_every_rank(
        buffer, failing_rank, stage, tmp_path):
    """F1.8 for parallel3d: a space partition always streams, so every call allocates both buffers."""
    label = f"parallel3d/{stage}/space/failing rank {failing_rank}/{buffer} buffer allocation"
    records, worker_error, ending = _run_allocation_failure(tmp_path, stage, failing_rank, buffer)
    _assert_phase_complete(records, "warmup", label, ending)
    _assert_failure_reaches_every_rank(records, label, failing_rank, buffer, ending)
    _assert_recovery(records, label, ending)
    _assert_valid_call_values(records)
    if worker_error is not None:
        raise worker_error
    assert ending == "all workers exited normally", ending
