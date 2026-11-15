"""Two threads that ask for different CUDA streams each get their own stream wrapper.

Spec item F4.1 (native): ``diffct.utils._get_numba_external_stream_for`` is the
shared boundary where every native kernel launch turns a PyTorch CUDA stream
into a Numba stream wrapper. A wrapper built for another stream launches the
kernel on that other stream.

The interleaving is fixed, not left to the scheduler. Thread A is stopped
before one line of its request (line tracing, lines of the ``diffct`` package
only); thread B then makes one complete request for another stream; A
continues. Every line of A's request is used as the stopping point once.
A's stop has a time limit, so an implementation that makes B wait for A
(a lock) passes instead of blocking the test.

No CUDA device is needed, only an importable ``numba.cuda`` (on a host
without the NVIDIA bindings: ``NUMBA_ENABLE_CUDASIM=1
NUMBA_CUDA_USE_NVIDIA_BINDING=0``): the streams are plain objects with the two
attributes the boundary reads, and ``numba.cuda.external_stream`` is replaced
by a factory that records the stream pointer of each wrapper.
"""

from itertools import count
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

import diffct
from diffct import utils

# The boundary imports numba.cuda. A host without the NVIDIA bindings can
# import it only as the simulator.
cuda = pytest.importorskip(
    "numba.cuda", exc_type=ImportError,
    reason="numba.cuda is not importable: NVIDIA CUDA bindings or "
           "NUMBA_ENABLE_CUDASIM=1 NUMBA_CUDA_USE_NVIDIA_BINDING=0 are required")


PACKAGE = Path(diffct.__file__).absolute().parent
JOIN_SECONDS = 60
# How long A stays stopped when B cannot finish while A is stopped.
STOP_SECONDS = .25
_POINTERS = count(0x7F0000001000, 0x1000)


class _Wrapper:
    """Stands for a Numba external stream; remembers the stream pointer it wraps."""

    def __init__(self, pointer):
        self.pointer = int(pointer)


@pytest.fixture
def wrappers():
    created = []

    def external_stream(pointer):
        created.append(_Wrapper(pointer))
        return created[-1]

    with mock.patch.object(cuda, "external_stream", external_stream, create=True):
        yield created


def _stream():
    """A stream never requested before, so earlier requests cannot answer for it."""
    return SimpleNamespace(cuda_stream=next(_POINTERS), device=torch.device("cuda:0"))


def _request(stream):
    wrapper = utils._get_numba_external_stream_for(stream)
    assert isinstance(wrapper, _Wrapper), f"the boundary returned {wrapper!r}, not a wrapper from the factory"
    return wrapper.pointer


def _interleaved(stop_at, seeded):
    """A's request is stopped before its ``stop_at``-th line while B makes one request.

    Returns (number of lines A's request ran, list of wrong wrappers).
    """
    a, b = _stream(), _stream()
    names = {a.cuda_stream: "A's stream", b.cuda_stream: "B's stream"}
    if seeded:
        assert _request(a) == a.cuda_stream
    a_stopped, b_done = threading.Event(), threading.Event()
    lines, got, errors = [0], {}, []

    def on_line(frame, event, argument):
        if event == "line":
            if lines[0] == stop_at:
                a_stopped.set()
                b_done.wait(STOP_SECONDS)
            lines[0] += 1
        return on_line

    def on_call(frame, event, argument):
        inside = PACKAGE in Path(frame.f_code.co_filename).absolute().parents
        return on_line if inside else None

    def run_a():
        previous = sys.gettrace()
        try:
            sys.settrace(on_call)
            try:
                got["A"] = _request(a)
            finally:
                sys.settrace(previous)
        except BaseException as error:
            errors.append(f"A: {type(error).__name__}: {error}")
        finally:
            a_stopped.set()

    def run_b():
        try:
            if not a_stopped.wait(JOIN_SECONDS):
                raise TimeoutError("A never reached its stopping point")
            got["B"] = _request(b)
        except BaseException as error:
            errors.append(f"B: {type(error).__name__}: {error}")
        finally:
            b_done.set()

    threads = [threading.Thread(target=run_a, daemon=True), threading.Thread(target=run_b, daemon=True)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=JOIN_SECONDS)
    assert not any(thread.is_alive() for thread in threads), f"threads did not finish within {JOIN_SECONDS} s"
    assert not errors, errors
    # The requests that follow the interleaving read what it left behind.
    checks = [("A during the interleaving", got["A"], a.cuda_stream),
              ("B during the interleaving", got["B"], b.cuda_stream),
              ("B afterwards", _request(b), b.cuda_stream),
              ("A afterwards", _request(a), a.cuda_stream),
              ("B after A", _request(b), b.cuda_stream)]
    wrong = [f"{label} received the wrapper of {names.get(pointer, hex(pointer))}"
             for label, pointer, wanted in checks if pointer != wanted]
    return lines[0], wrong


def test_serial_requests_return_a_wrapper_of_the_requested_stream(wrappers):
    """Control: without threads every request is answered for its own stream."""
    first, second = _stream(), _stream()
    for stream in (first, first, second, first, second, second):
        assert _request(stream) == stream.cuda_stream
    assert {wrapper.pointer for wrapper in wrappers} == {first.cuda_stream, second.cuda_stream}


@pytest.mark.parametrize("seeded", [False, True], ids=["first_request", "repeated_request"])
def test_interleaved_requests_for_different_streams_return_their_own_wrappers(wrappers, seeded):
    # A run without a stopping point counts the lines of one request.
    total, wrong = _interleaved(None, seeded)
    assert not wrong, wrong
    assert total >= 1, "line tracing saw no line of the request inside the diffct package"
    failures = {}
    for stop_at in range(total):
        _, wrong = _interleaved(stop_at, seeded)
        if wrong:
            failures[stop_at] = wrong
    assert not failures, (
        f"a request for another stream, made while A was stopped before line index {sorted(failures)} "
        f"(0-based) of its {total} lines, mixed the stream wrappers: {failures}")
