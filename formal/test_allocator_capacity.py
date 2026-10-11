"""Native reservation admission under explicit idealized allocator assumptions.

PROOF: pooled-bin algebra when each prior fresh bin has occupied bytes greater
than segment minus the largest request; pinned reservations cannot bypass cap.
BOUNDED: every request permutation for 1..4 simultaneously live buffers from
nine size boundaries, native pool separation, and allocation rounding.
Assumptions: one stable owner, fresh unfragmented bins, monotonically added live
requests, standard native 512-byte requests / 2-MiB small segments / 20-MiB
middle segments / large requests rounded to 2 MiB. Middle and large requests
share the actual large pool; mixed requests receive a conservative fresh-segment
charge per request instead of treating them as separate pools.
Excluded: arbitrary free-hole patterns (explicit counterexample below), pending
event retirement, source equivalence, real CUDA streams, other processes,
allocator configuration overrides, callbacks and cudaMallocAsync internals.
Actual process ceilings and numerical results require the separate CUDA tests.
"""

from itertools import combinations_with_replacement, permutations
from unittest import mock

import pytest
import sympy as sp
import torch
import z3

from diffct import chunking


MIB = 1024 ** 2


def _declared(name):
    function = getattr(chunking, name, None)
    assert callable(function), f"INTERFACE_PENDING: diffct.chunking.{name}"
    return function


def _fresh_best_fit(sizes):
    """Independent finite allocator: one owner, no frees, each live request once."""
    pools = {"small": [], "large": []}
    reserved = 0
    for raw in sizes:
        size = (raw + 511) // 512 * 512
        if not size:
            continue
        name = "small" if size <= MIB else "large"
        segment = (2 * MIB if size <= MIB else 20 * MIB if size < 10 * MIB else
                   (size + 2 * MIB - 1) // (2 * MIB) * (2 * MIB))
        free = pools[name]
        fitting = [index for index, amount in enumerate(free) if amount >= size]
        if fitting:
            index = min(fitting, key=free.__getitem__)
            free[index] -= size
        else:
            free.append(segment - size)
            reserved += segment
    return reserved


def test_PROOF_pooled_bin_bound_from_strict_prior_bin_occupancy():
    # Euclidean division: total-min=q*d+r, d=segment-max, 0<=r<d.
    # If k=q+2+extra bins existed, strict old-bin occupancy would require
    # q*d+r > (q+1+extra)*d. Cancelling q*d yields an impossible remainder.
    q, d, remainder, extra = sp.symbols("q d r extra", integer=True)
    assert sp.expand(q * d + remainder - (q + 1 + extra) * d) == remainder - d - extra * d
    solver = z3.Solver()
    distance, rest, excess = z3.Ints("distance remainder extra_bins")
    solver.add(distance > 0, rest >= 0, rest < distance, excess >= 0,
               rest > distance + excess * distance)
    assert solver.check() == z3.unsat


def test_PROOF_pinned_reservations_plus_admitted_bound_cannot_exceed_cap():
    cap, pinned, needed, free, idle = z3.Ints("cap pinned needed free idle")
    solver = z3.Solver()
    solver.add(cap >= 0, pinned >= 0, needed >= 0, free >= 0, idle >= 0,
               needed <= cap - pinned, needed <= free + idle, pinned + needed > cap)
    assert solver.check() == z3.unsat
    # Using allocated payload instead of pinned segment bytes admits a real
    # 4-MiB caller / 20-MiB segment + a new 20-MiB segment under a 32-MiB cap.
    allocated = 4 * MIB
    assert 20 * MIB <= 32 * MIB - allocated
    assert 20 * MIB + 20 * MIB > 32 * MIB


def test_BOUNDED_free_holes_break_the_monotone_fresh_bin_assumption():
    # One middle segment held [8,8,4] MiB. Free first and last allocations;
    # two separated holes total 12 MiB, but neither admits a 9-MiB request.
    holes, live, incoming = [8 * MIB, 4 * MIB], 8 * MIB, 9 * MIB
    assert sum(holes) >= incoming and max(holes) < incoming
    assert live + incoming < 20 * MIB
    actual_reserved = 20 * MIB + 20 * MIB
    assert actual_reserved > _fresh_best_fit((live, incoming))


def test_BOUNDED_mixed_large_pool_mutation_has_a_fresh_monotone_counterexample():
    sizes = (6 * MIB, 10 * MIB, 6 * MIB)
    # The 10-MiB allocation consumes an existing 14-MiB tail, leaving only
    # 4 MiB for the last 6-MiB request. An independent-large-bin model misses it.
    actual = _fresh_best_fit(sizes)
    old_separate_pool_estimate = 20 * MIB + 10 * MIB
    assert actual == 40 * MIB > old_separate_pool_estimate
    reserve = _declared("_reservation_bytes")
    with mock.patch.object(torch.cuda, "get_allocator_backend", return_value="native"):
        assert reserve(sizes) >= actual


@pytest.mark.parametrize("sizes,expected", [
    ((), 0), ((MIB,), 2 * MIB), ((MIB + 1,), 20 * MIB),
    ((10 * MIB - 512,), 20 * MIB), ((10 * MIB,), 10 * MIB),
    ((10 * MIB + 1,), 12 * MIB),
    ((8 * MIB, 8 * MIB, 4 * MIB, 512), 22 * MIB),
    ((8 * MIB, 8 * MIB, 5 * MIB, 512), 42 * MIB),
    (((6 * MIB + 7 * MIB // 10),) * 5, 60 * MIB),
    ((6 * MIB, 10 * MIB, 6 * MIB), 50 * MIB),
])
def test_BOUNDED_native_reservation_pool_and_packing_boundaries(sizes, expected):
    reserve = _declared("_reservation_bytes")
    with mock.patch.object(torch.cuda, "get_allocator_backend", return_value="native"):
        assert reserve(sizes) == expected


def test_BOUNDED_production_bound_dominates_all_fresh_best_fit_request_orders():
    reserve = _declared("_reservation_bytes")
    domain = (512, MIB, MIB + 512, 4 * MIB, 6 * MIB, 8 * MIB,
              10 * MIB - 512, 10 * MIB, 10 * MIB + 512)
    with mock.patch.object(torch.cuda, "get_allocator_backend", return_value="native"):
        for count in range(1, 5):
            for sizes in combinations_with_replacement(domain, count):
                bound = reserve(sizes)
                separate_segments = sum(_fresh_best_fit((size,)) for size in sizes)
                assert bound <= separate_segments
                for order in set(permutations(sizes)):
                    assert _fresh_best_fit(order) <= bound, (order, bound)


def test_BOUNDED_async_uses_typed_bytes_without_native_pool_segments():
    reserve = _declared("_reservation_bytes")
    sizes = (0, 511, MIB + 1, 8 * MIB, 10 * MIB + 1)
    with mock.patch.object(torch.cuda, "get_allocator_backend", return_value="cudaMallocAsync"):
        assert reserve(sizes) == sum(sizes)


@pytest.mark.parametrize("state", ["active_allocated", "active_awaiting_free"])
def test_BOUNDED_snapshot_admission_counts_entire_pinned_segment_and_idle_reclaim(state):
    budget = _declared("_reservation_budget")
    # A pinned 20-MiB segment includes 16 MiB of unusable free splits. The
    # wholly idle 4-MiB segment can be reclaimed regardless of its old owner.
    segments = [dict(device=0, total_size=20 * MIB, stream=11, blocks=[
        dict(size=4 * MIB, state=state), dict(size=16 * MIB, state="inactive")]),
        dict(device=0, total_size=4 * MIB, stream=22,
             blocks=[dict(size=4 * MIB, state="inactive")]),
        dict(device=1, total_size=100 * MIB, stream=11,
             blocks=[dict(size=100 * MIB, state="active_allocated")])]
    with mock.patch.object(torch.cuda, "get_allocator_backend", return_value="native"), \
            mock.patch.object(torch.cuda, "mem_get_info", return_value=(MIB, 64 * MIB)), \
            mock.patch.object(torch.cuda, "get_per_process_memory_fraction", return_value=.5), \
            mock.patch.object(torch.cuda, "memory_snapshot", return_value=segments):
        assert budget(0) == 5 * MIB


def test_BOUNDED_async_reservation_admission_does_not_apply_native_snapshot_policy():
    budget = _declared("_reservation_budget")
    with mock.patch.object(torch.cuda, "get_allocator_backend", return_value="cudaMallocAsync"), \
            mock.patch.object(torch.cuda, "memory_snapshot", side_effect=AssertionError("native snapshot queried")):
        assert budget(0) >= (1 << 62)


def test_BOUNDED_pipeline_inventory_exposes_real_8_8_5mib_requests_before_admission():
    buffers = _declared("_pipeline_buffers")
    reserve = _declared("_reservation_bytes")
    with mock.patch.object(torch.cuda, "get_allocator_backend", return_value="native"):
        sizes = buffers("cone", (128, 128, 128), 5, (512, 512), False)
        assert sizes.count(8 * MIB) == 2 and sizes.count(5 * MIB) == 1
        assert sum(sizes) < 24 * MIB < reserve(sizes)
        assert reserve(sizes) == 42 * MIB
