"""Phase-consistent admission under an idealized standard native allocator.

PROOF: a bound on persistent pinned segments plus fresh pipeline reservations
implies the same pipeline fits after those states are allocated.
BOUNDED: the 2-MiB volume reproducer, 32/64-MiB ceilings and a reused idle
20/48-MiB segments, linked to production
planning with mocked CUDA metadata. No native kernel or real stream runs here.
Assumptions: stable owner, fresh monotone state allocation, standard 512-byte
rounding / 2-MiB small segments / 20-MiB middle segments, stable driver room.
Excluded: arbitrary holes, allocator overrides, event retirement, concurrent
owners and whole-system correctness. Real CUDA evidence lives in tests/.
"""

from contextlib import contextmanager
import importlib.util
from itertools import combinations, product
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import z3

from diffct import Projector, chunking


MIB = 1024 ** 2
SHAPE = (1, 524288)


def _blocks():
    path = Path(__file__).resolve().parents[1] / "examples" / "_block_cgls.py"
    spec = importlib.util.spec_from_file_location("_formal_state_admission_blocks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _store(shape, device):
    def no_io(*args):
        raise AssertionError("MOCKED allocator metadata must not execute store I/O")
    return SimpleNamespace(shape=shape, dtype=torch.float32, device=torch.device(device),
                           read=no_io, write=no_io, flush=no_io)


@contextmanager
def _allocator_metadata(cap, *, states_live=False, cached_mib=0):
    small_live = 512 * (3 if states_live else 1)
    segments = [dict(device=0, total_size=2 * MIB, stream=17, segment_type="small", blocks=[
        dict(size=small_live, state="active_allocated"),
        dict(size=2 * MIB - small_live, state="inactive")])]
    if states_live or cached_mib:
        middle = (cached_mib or 20) * MIB
        blocks = ([dict(size=6 * MIB, state="active_allocated"),
                   dict(size=middle - 6 * MIB, state="inactive")] if states_live else
                  [dict(size=middle, state="inactive")])
        segments.append(dict(device=0, total_size=middle, stream=17, segment_type="large", blocks=blocks))
    allocated = small_live + (6 * MIB if states_live else 0)
    reserved = sum(segment["total_size"] for segment in segments)
    with mock.patch.object(torch.cuda, "is_available", return_value=True), \
            mock.patch.object(torch.cuda, "device_count", return_value=1), \
            mock.patch.object(torch.cuda, "current_device", return_value=0), \
            mock.patch.object(torch.cuda, "get_allocator_backend", return_value="native"), \
            mock.patch.object(torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=17)), \
            mock.patch.object(torch.cuda, "mem_get_info", return_value=(1024 * MIB, 2048 * MIB)), \
            mock.patch.object(torch.cuda, "get_per_process_memory_fraction", return_value=cap / (2048 * MIB)), \
            mock.patch.object(torch.cuda, "memory_allocated", return_value=allocated), \
            mock.patch.object(torch.cuda, "memory_reserved", return_value=reserved), \
            mock.patch.object(torch.cuda, "memory_snapshot", return_value=segments):
        yield


def _admitted(report):
    return (report["estimated_gpu_bytes"] <= report["gpu_budget_bytes"]
            and report["largest_cuda_allocation_bytes"] <= report["allocation_limit"]
            and report["estimated_reserved_bytes"] <= report["reservation_budget_bytes"])


def test_PROOF_persistent_segment_bound_leaves_the_fresh_pipeline_room_at_every_phase():
    cap, caller, bound, actual, pipeline = z3.Ints("phase_cap phase_caller phase_bound phase_actual phase_pipeline")
    solver = z3.Solver()
    solver.add(cap >= 0, caller >= 0, bound >= 0, actual >= 0, pipeline >= 0,
               actual <= bound, caller + bound + pipeline <= cap,
               pipeline > cap - caller - actual)
    assert solver.check() == z3.unsat


def test_BOUNDED_joint_packing_mutation_admits_a_20mib_plan_that_poststate_replanning_rejects():
    # Three 2-MiB volume states and one 4-MiB dot block can pack with two
    # 2-MiB pipeline tiles. Separate phase admission excludes pinned state
    # segments, so the fresh middle-pool pipeline charge remains 20 MiB.
    with mock.patch.object(torch.cuda, "get_allocator_backend", return_value="native"):
        persistent = (2 * MIB,) * 3
        pipeline = (2 * MIB,) * 2
        packed = chunking._reservation_bytes(persistent + (4 * MIB,) + pipeline)
        pinned = chunking._reservation_bytes(persistent)
        fresh = chunking._reservation_bytes(pipeline)
    assert packed == pinned == fresh == 20 * MIB
    assert packed <= 32 * MIB < pinned + fresh
    assert fresh > 32 * MIB - pinned


def test_BOUNDED_idle_segment_reuse_breaks_a_fresh_only_persistent_state_bound():
    with mock.patch.object(torch.cuda, "get_allocator_backend", return_value="native"):
        fresh_state_bound = chunking._reservation_bytes((2 * MIB,) * 3)
        fresh_pipeline = chunking._reservation_bytes((2 * MIB,) * 2)
    reused_segment = 48 * MIB
    assert fresh_state_bound + fresh_pipeline <= 64 * MIB
    assert reused_segment > fresh_state_bound
    assert reused_segment + fresh_pipeline > 64 * MIB


def _monotone_pinned_bytes(requests, cached, fresh):
    """Independent best-fit pool, wholly idle cached bins, no frees or splits lost."""
    capacities, free = list(cached), list(cached)
    pinned = set()
    for request in requests:
        fitting = [index for index, remaining in enumerate(free) if remaining >= request]
        if fitting:
            index = min(fitting, key=free.__getitem__)
        else:
            index = len(free)
            capacities.append(fresh)
            free.append(fresh)
        free[index] -= request
        pinned.add(index)
    return sum(capacities[index] for index in pinned)


def test_BOUNDED_variable_capacity_monotone_bins_bound_actual_newly_pinned_segments():
    # Abstract integer units. Every possible request order of length 1..5,
    # requests 1..3, fresh capacity 8, and all subsets of idle bins 5/8/13.
    # Each bin's minimum capacity exceeds the largest request. Best-fit uses
    # a fitting live tail before activating a larger wholly idle cached bin.
    # This finite model establishes the pin bound rather than assuming it.
    fresh, capacities = 8, (5, 8, 13)
    checked = 0
    for cache_count in range(len(capacities) + 1):
        for cached in combinations(capacities, cache_count):
            minimum, maximum = min((fresh, *cached)), max((fresh, *cached))
            for count in range(1, 6):
                for requests in product((1, 2, 3), repeat=count):
                    total = sum(requests)
                    bins = (1 if total <= minimum else
                            (total - min(requests)) // (minimum - max(requests)) + 1)
                    actual = _monotone_pinned_bytes(requests, cached, fresh)
                    assert actual <= bins * maximum, (requests, cached, actual, bins, maximum)
                    checked += 1
    assert checked == 2904
    # Mutation: always charging the fresh capacity misses a larger idle bin.
    assert _monotone_pinned_bytes((1,), (13,), fresh) == 13 > fresh


@pytest.mark.parametrize("cap_mib,cached_mib", [(32, 0), (64, 0), (64, 20), (64, 48)],
                         ids=["cold32", "cold64-positive-control", "idle20-under64-positive-control", "idle48-under64"])
def test_BOUNDED_MOCKED_solver_report_agrees_with_first_poststate_bp_fp_planning(cap_mib, cached_mib):
    cap = cap_mib * MIB
    geometry = (torch.tensor([[1., 0.]]), torch.zeros(1, 2), torch.tensor([[0., 1.]]))
    rays, volume = _store((1, 1), "cuda:0"), _store(SHAPE, "cuda:0")
    with _allocator_metadata(cap, cached_mib=cached_mib):
        operator = Projector(geometry, SHAPE, (1,), beam="parallel", devices=[0],
                             schedule="spatial", volume_chunk_shape=SHAPE, view_chunk_size=1)
        report = _blocks()._solver_capacity(operator, rays, MIB, "cuda:0")
        cpu_report = _blocks()._solver_capacity(operator, rays, MIB, "cpu")
        # This is the first real BP/FP plan for host state: only the caller's
        # small measurement segment is pinned before native execution.
        for source, output, is_project in ((_store((1, 1), "cpu"), _store(SHAPE, "cpu"), False),
                                          (_store(SHAPE, "cpu"), _store((1, 1), "cpu"), True)):
            plan = operator._execution_plan(source, is_project, output)
            assert plan.estimated_reserved_bytes <= plan.reservation_budget_bytes
        assert cpu_report["estimated_cuda_state_bytes"] == 0
        assert _admitted(cpu_report)
    poststate_fits = []
    with _allocator_metadata(cap, states_live=True, cached_mib=cached_mib):
        assert chunking._reservation_budget(0) == cap - (2 + (cached_mib or 20)) * MIB
        for source, output, is_project in ((rays, volume, False), (volume, rays, True)):
            try:
                plan = operator._execution_plan(source, is_project, output)
            except RuntimeError as error:
                assert "cannot fit" in str(error)
                poststate_fits.append(False)
            else:
                poststate_fits.append(plan.estimated_reserved_bytes <= plan.reservation_budget_bytes)
    if cap_mib == 32 or cached_mib == 48:
        assert poststate_fits == [False, False], "fixture no longer has the phase mismatch"
        assert not _admitted(report), ("initial CGLS admission must reject states that pin the first BP/FP out", report)
    else:
        assert poststate_fits == [True, True]
        assert _admitted(report), "adequate phase capacity must retain CUDA state"
