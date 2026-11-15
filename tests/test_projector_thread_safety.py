"""One portable Projector shared by several Python threads.

The automatic leading-dimension limit is forced to one element, so every call
runs the recursive piece path that publishes and reads per-call statistics.
Each threaded call must equal the same call run alone: bit-exact output and the
same ``last_execution_stats`` (except wall-clock time) read in the calling thread.

Spec items: F4.1 (no bookkeeping error, bit-exact results), F4.2 (per-thread
view: statistics read in thread T after its call returns are that call's
complete statistics, also after another thread has completed a later call),
F4.3 (serial values unchanged, None after a call that raises).
"""

import math
import sys
import threading
from unittest import mock

import pytest
import torch

from diffct import Projector, backends
from tests.test_reference_backend import _case, _matrix


THREADS = 4
ITERATIONS = 30
JOIN_SECONDS = 120
# Wall-clock time is the only statistic that differs between equal calls.
TIMING_KEYS = ("elapsed_seconds",)


@pytest.fixture
def one_element_pieces(monkeypatch):
    """Force the automatic leading-dimension split to one element per piece."""
    monkeypatch.setattr(backends, "_REFERENCE_WORK_BYTES", 1)


@pytest.fixture
def frequent_thread_switches():
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


def _projector():
    case = _case("parallel")
    # Volume and view hints are explicit; the detector hint stays automatic,
    # so the automatic leading-dimension limit applies.
    return Projector(
        case.trajectory, case.shape, case.detector, beam=case.beam,
        detector_spacing=case.pitch[0], voxel_spacing=case.spacing,
        backend="torch", precision="float64",
        volume_chunk_shape=case.shape, view_chunk_size=len(case.trajectory[0]),
    )


def _requests(projector):
    """Distinct inputs per (thread, iteration); operations and batch sizes differ per thread."""
    requests = {}
    for thread in range(THREADS):
        operation = "project" if thread % 2 == 0 else "backproject"
        shape = projector.volume_shape if operation == "project" else projector.projection_shape
        leading = (thread + 2,)
        for iteration in range(ITERATIONS):
            generator = torch.Generator().manual_seed(1000 * thread + iteration)
            value = torch.randn((*leading, *shape), dtype=torch.float64, generator=generator)
            requests[thread, iteration] = (operation, value)
    return requests


def _comparable(stats):
    assert isinstance(stats, dict), f"last_execution_stats is {stats!r}, expected a dict"
    return {key: value for key, value in stats.items() if key not in TIMING_KEYS}


def _solo(projector, operation, value):
    result = getattr(projector, operation)(value)
    return result, _comparable(projector.last_execution_stats)


# Statistics of the serial calls below, recorded on the pre-fix tree (d260ec1).
# Serial single-thread behaviour must keep these values.
_SERIAL_STATS = {
    operation: dict(operation=operation, backend="torch", precision="float64", dtype="torch.float64",
                    schedule="views", chunk_shape=(3, 4), view_chunk_size=3, detector_chunk_shape=(1,),
                    devices=["cpu"], ray_tile_pairs=5 * leading, store_reads=0, store_writes=0,
                    read_bytes=0, write_bytes=0, h2d_bytes=0, d2h_bytes=0, p2p_bytes=0,
                    collective_calls=0, collective_bytes=0)
    for operation, leading in (("project", 2), ("backproject", 3))
}


@pytest.mark.usefixtures("one_element_pieces")
def test_serial_piece_calls_keep_statistics_and_match_independent_matrix():
    case = _case("parallel")
    matrix = _matrix(case)  # Independent per-voxel box intersections.
    projector = _projector()
    requests = _requests(projector)
    for repeat in range(2):
        for thread in range(2):
            operation, value = requests[thread, repeat]
            result, stats = _solo(projector, operation, value)
            assert stats == _SERIAL_STATS[operation], (operation, repeat, stats)
            flat = value.reshape(value.shape[0], -1)
            expected = flat @ (matrix.T if operation == "project" else matrix)
            torch.testing.assert_close(result.reshape(value.shape[0], -1), expected,
                                       rtol=2e-11, atol=2e-11)
            # A fresh Projector gives the same result: serial reuse keeps no state.
            fresh_result, fresh_stats = _solo(_projector(), operation, value)
            assert torch.equal(result, fresh_result) and stats == fresh_stats


def _fail_at_call(state, failing_call):
    """Replace ``_intervals``: raise on one numbered call, run the real method otherwise."""
    original = backends.ReferenceProjector._intervals

    def fake(self, *args, **kwargs):
        state["calls"] += 1
        if state["calls"] == failing_call:
            state["raised"] += 1
            raise torch.OutOfMemoryError("injected allocation failure in the numeric stage")
        return original(self, *args, **kwargs)

    return mock.patch.object(backends.ReferenceProjector, "_intervals", fake)


@pytest.mark.usefixtures("one_element_pieces")
@pytest.mark.parametrize("position", ["first_piece", "last_piece"])
@pytest.mark.parametrize("thread", [0, 1], ids=["project", "backproject"])
def test_serial_call_that_raises_leaves_no_statistics_then_next_call_keeps_them(thread, position):
    """F4.3 error path: None after a call that raises, unchanged values after the next call."""
    projector = _projector()
    operation, value = _requests(projector)[thread, 0]
    # Count the interval computations of one complete call through the same boundary.
    count = dict(calls=0, raised=0)
    with _fail_at_call(count, failing_call=0):
        solo_result, solo_stats = _solo(projector, operation, value)
    assert solo_stats == _SERIAL_STATS[operation]
    assert count["calls"] >= 2 * value.shape[0], "fixture must compute intervals in every piece"
    # The last interval computation belongs to the last piece: earlier pieces
    # have already published their statistics when it raises.
    failing_call = 1 if position == "first_piece" else count["calls"]

    state = dict(calls=0, raised=0)
    with _fail_at_call(state, failing_call):
        with pytest.raises(torch.OutOfMemoryError, match="injected allocation failure"):
            getattr(projector, operation)(value)
        assert state["raised"] == 1
        assert projector.last_execution_stats is None
        # The replaced method now runs the real computation: a normal call.
        result, stats = _solo(projector, operation, value)
        assert state["raised"] == 1
    assert stats == _SERIAL_STATS[operation], (operation, position, stats)
    assert torch.equal(result, solo_result)
    result, stats = _solo(projector, operation, value)
    assert stats == _SERIAL_STATS[operation] and torch.equal(result, solo_result)


@pytest.mark.usefixtures("one_element_pieces", "frequent_thread_switches")
def test_concurrent_calls_match_solo_results_and_statistics():
    """F4.1 and F4.2: each thread reads its own call's result and complete statistics."""
    projector = _projector()
    requests = _requests(projector)
    # Solo references: the same calls, one at a time, in this thread.
    expected = {key: _solo(projector, *request) for key, request in requests.items()}
    pieces = {key: math.prod(request[1].shape[:1]) for key, request in requests.items()}
    assert all(count >= 2 for count in pieces.values()), "fixture must run the piece path"

    barrier = threading.Barrier(THREADS, timeout=JOIN_SECONDS)
    errors, mismatches = [], []

    def worker(thread):
        try:
            for iteration in range(ITERATIONS):
                operation, value = requests[thread, iteration]
                barrier.wait()
                try:
                    result = getattr(projector, operation)(value)
                    # F4.2: read in the calling thread, after its call returned,
                    # while the other threads keep calling the same Projector.
                    stats = projector.last_execution_stats
                except Exception as error:  # every failure is reported, not only the first
                    errors.append((thread, iteration, f"{type(error).__name__}: {error}"))
                    continue
                solo_result, solo_stats = expected[thread, iteration]
                if not torch.equal(result, solo_result):
                    mismatches.append((thread, iteration, "result differs from solo call"))
                if not isinstance(stats, dict):
                    mismatches.append((thread, iteration, f"last_execution_stats is {stats!r}"))
                elif _comparable(stats) != solo_stats:
                    mismatches.append((thread, iteration,
                                       f"stats {_comparable(stats)} != solo {solo_stats}"))
        except threading.BrokenBarrierError:
            errors.append((thread, None, "barrier broken (another thread stalled or failed)"))
        except BaseException as error:
            errors.append((thread, None, f"{type(error).__name__}: {error}"))
            barrier.abort()

    threads = [threading.Thread(target=worker, args=(index,), daemon=True) for index in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=JOIN_SECONDS)
    alive = [thread.name for thread in threads if thread.is_alive()]
    if alive:
        barrier.abort()
    assert not alive, f"threads did not finish within {JOIN_SECONDS} s: {alive}"
    assert not errors, f"{len(errors)} concurrent calls raised; first: {errors[:3]}"
    assert not mismatches, f"{len(mismatches)} concurrent calls differ from solo; first: {mismatches[:3]}"


@pytest.mark.usefixtures("one_element_pieces")
@pytest.mark.parametrize("first", [0, 1], ids=["project_then_backproject", "backproject_then_project"])
def test_statistics_read_after_another_thread_completed_a_call_are_the_reader_own(first):
    """F4.2 in a fixed order: A returns, B runs one complete call, then A reads.

    No call overlaps another, so serializing the calls cannot make A's read
    correct: only a per-thread view keeps A's statistics after B's call.
    """
    projector = _projector()
    requests = _requests(projector)
    request = {"A": requests[first, 0], "B": requests[1 - first, 0]}
    assert request["A"][0] != request["B"][0] and request["A"][1].shape[0] != request["B"][1].shape[0]
    solo = {name: _solo(projector, *value) for name, value in request.items()}
    assert solo["A"][1] != solo["B"][1], "fixture needs calls with different statistics"

    a_returned, b_returned = threading.Event(), threading.Event()
    seen, errors = {}, []

    def run(name, wait_for, done, read_after):
        try:
            if wait_for is not None and not wait_for.wait(JOIN_SECONDS):
                raise TimeoutError(f"thread {name} never started: the other thread did not finish")
            operation, value = request[name]
            result = getattr(projector, operation)(value)
            done.set()
            if read_after is not None and not read_after.wait(JOIN_SECONDS):
                raise TimeoutError(f"thread {name} never read: the other thread did not finish")
            seen[name] = (result, projector.last_execution_stats)
        except BaseException as error:
            errors.append((name, f"{type(error).__name__}: {error}"))
            done.set()

    threads = [threading.Thread(target=run, args=("A", None, a_returned, b_returned), daemon=True),
               threading.Thread(target=run, args=("B", a_returned, b_returned, None), daemon=True)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=JOIN_SECONDS)
    assert not any(thread.is_alive() for thread in threads), f"threads did not finish within {JOIN_SECONDS} s"
    assert not errors, errors
    for name in ("B", "A"):
        result, stats = seen[name]
        assert torch.equal(result, solo[name][0]), f"thread {name}: result differs from solo call"
        other = "A" if name == "B" else "B"
        assert _comparable(stats) == solo[name][1], (
            f"thread {name} read statistics that are not its own call's "
            f"(thread {other} ran {request[other][0]}, thread {name} ran {request[name][0]}): {stats}"
        )
