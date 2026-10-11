"""Bounded event-reference ownership and completed-transfer retirement.

BOUNDED: all interleavings of 1..3 queued transfers, 1..2 reusable buffers,
completion, reference retirement, and stopping admission after a host error.
An unfinished transfer samples its source only at completion. Event-owned
references prevent early reuse; retiring completed references restores all
modeled buffer capacity after close. Two guard mutations have counterexamples.

Assumptions: fixed buffer pool; prior producer readiness and copy ordering;
event completion means the final device use finished; close finishes queued
work before releasing references. Upload/compute/download ordering is covered
separately by test_execution_protocol.py. Excluded: source equivalence, actual
CUDA events, native allocator bins, DMA timing, foreign-device synchronization,
arbitrary external allocations, and writes performed by a failing backend.
Independent runtime tests are required for those implementation properties.
"""

from collections import deque
from dataclasses import dataclass, replace

import pytest


@dataclass(frozen=True)
class _Event:
    job: int
    buffer: int
    complete: bool = False
    retained: bool = True


@dataclass(frozen=True)
class _State:
    next_job: int
    values: tuple
    events: tuple = ()
    observed: tuple = ()
    stopped: bool = False


def _owned(state):
    return {event.buffer for event in state.events if event.retained}


def _moves(state, jobs, mutation=None):
    if not state.stopped:
        yield "host error stops admission", replace(state, stopped=True)
        if state.next_job < jobs:
            for buffer in range(len(state.values)):
                if buffer not in _owned(state):
                    values = list(state.values)
                    values[buffer] = state.next_job
                    event = _Event(state.next_job, buffer)
                    yield "queue transfer", replace(
                        state, next_job=state.next_job + 1, values=tuple(values),
                        events=(*state.events, event),
                    )
    for index, event in enumerate(state.events):
        if not event.complete:
            events = list(state.events)
            events[index] = replace(event, complete=True)
            observed = tuple(sorted((*state.observed, (event.job, state.values[event.buffer]))))
            yield "complete transfer", replace(state, events=tuple(events), observed=observed)
            if mutation == "drop-before-wait" and event.retained:
                events[index] = replace(event, retained=False)
                yield "drop unfinished references", replace(state, events=tuple(events))
        elif mutation != "omit-completed-retirement":
            yield "retire completed references", replace(
                state, events=state.events[:index] + state.events[index + 1:],
            )


def _reachable(jobs, buffers, mutation=None):
    initial = _State(0, (-1,) * buffers)
    seen, frontier, terminal = {initial}, deque([initial]), set()
    while frontier:
        state = frontier.popleft()
        moves = list(_moves(state, jobs, mutation))
        if not moves:
            terminal.add(state)
        for _, following in moves:
            if following not in seen:
                seen.add(following)
                frontier.append(following)
    return seen, terminal


@pytest.mark.parametrize("jobs", [1, 2, 3])
@pytest.mark.parametrize("buffers", [1, 2])
def test_event_owned_transfers_preserve_values_and_restore_capacity_after_close(jobs, buffers):
    states, terminal = _reachable(jobs, buffers)
    assert states and terminal
    for state in states:
        assert len(_owned(state)) <= buffers
        assert all(event.retained for event in state.events if not event.complete)
        assert all(job == value for job, value in state.observed)
    for state in terminal:
        assert state.stopped or state.next_job == jobs
        assert len(state.observed) == state.next_job
        assert not state.events and not _owned(state)
        assert buffers - len(_owned(state)) == buffers
    if jobs > 1:
        assert any(state.stopped and 0 < state.next_job < jobs for state in terminal)
    else:
        assert any(state.stopped and state.next_job == jobs for state in terminal)


def test_dropping_oldest_references_before_barrier_has_reuse_counterexample():
    states, _ = _reachable(2, 1, "drop-before-wait")
    assert any(any(job != value for job, value in state.observed) for state in states)


def test_completed_events_without_reference_retirement_leave_capacity_pinned():
    _, terminal = _reachable(2, 2, "omit-completed-retirement")
    completed = [state for state in terminal if state.next_job == 2 and len(state.observed) == 2]
    assert completed
    assert all(all(event.complete for event in state.events) for state in completed)
    assert all(len(_owned(state)) == 2 for state in completed)
    assert all(2 - len(_owned(state)) == 0 for state in completed)
