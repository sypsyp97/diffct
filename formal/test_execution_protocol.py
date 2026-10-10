"""Idealized execution protocols: symbolic bounds and finite interleavings.

PROOF: integer admission bounds under the explicitly modeled slot, scratch,
geometry, snapshot and resident-accumulator assumptions.
BOUNDED: every reachable interleaving of 1..3 tagged jobs and 1..2 composite
slots, including host-input overwrite, upload, native computation, download,
CPU drain and reuse. Guard omissions have concrete counterexamples. Exact
symbolic resident rank reductions include an empty rank.
Excluded: production/source equivalence, real streams/events, allocators,
DMA overlap, CUDA atomics/fastmath, arbitrary sampler allocations and real NCCL.
Those require the independent runtime tests; this model cannot prove them.
"""

from collections import deque
from dataclasses import dataclass, replace
import itertools

import pytest
import sympy as sp
import z3


@dataclass(frozen=True)
class _Slot:
    job: int = -1
    stage: int = 0
    host: int = -1
    gpu: int = -1
    output: int = -1
    downloaded: int = -1


@dataclass(frozen=True)
class _State:
    next_job: int
    slots: tuple
    drained: tuple = ()


def _moves(state, jobs, missing=None):
    """Queued copies sample their source only when completion occurs."""
    for index, slot in enumerate(state.slots):
        changes = []
        if slot.stage == 0 and state.next_job < jobs:
            changes.append((replace(slot, job=state.next_job, stage=1, host=state.next_job), True, None))
        elif slot.stage == 1:
            changes.append((replace(slot, stage=2), False, None))  # enqueue upload
        elif slot.stage == 2:
            changes.append((replace(slot, stage=3, gpu=slot.host), False, None))
            if missing == "upload-before-compute":
                changes.append((replace(slot, stage=4), False, None))
        elif slot.stage == 3:
            changes.append((replace(slot, stage=4), False, None))
        elif slot.stage == 4:
            changes.append((replace(slot, stage=5, output=slot.gpu), False, None))
            if missing == "kernel-before-download":
                changes.append((replace(slot, stage=6), False, None))
        elif slot.stage == 5:
            changes.append((replace(slot, stage=6), False, None))  # enqueue download
        elif slot.stage == 6:
            changes.append((replace(slot, stage=7, downloaded=slot.output), False, None))
            if missing == "download-before-host-read":
                changes.append((replace(slot, stage=8), False, (slot.job, slot.downloaded)))
        elif slot.stage == 7:
            changes.append((replace(slot, stage=8), False, (slot.job, slot.downloaded)))
        elif slot.stage == 8:
            changes.append((_Slot(), False, None))
        if slot.host != -2 and (slot.stage >= 3 or missing == "host-overwrite-after-upload" and slot.stage == 2):
            changes.append((replace(slot, host=-2), False, None))
        if slot.output != -2 and (slot.stage >= 7 or missing == "output-reuse-after-download" and slot.stage == 6):
            changes.append((replace(slot, output=-2), False, None))
        if slot.gpu != -2 and (slot.stage >= 5 or missing == "gpu-input-reuse-after-kernel" and slot.stage == 4):
            changes.append((replace(slot, gpu=-2), False, None))
        for changed, assign, drain in changes:
            slots = list(state.slots)
            slots[index] = changed
            drained = state.drained if drain is None else (*state.drained, drain)
            yield _State(state.next_job + int(assign), tuple(slots), drained)


def _valid(state):
    occupied = [slot.job for slot in state.slots if slot.stage]
    if len(set(occupied)) != len(occupied):
        return False
    if any(3 <= slot.stage <= 4 and slot.gpu != slot.job for slot in state.slots):
        return False
    if any(5 <= slot.stage <= 6 and slot.output != slot.job for slot in state.slots):
        return False
    if any(slot.stage >= 7 and slot.downloaded != slot.job for slot in state.slots):
        return False
    return all(job == value for job, value in state.drained)


def _explore(jobs, slots, missing=None):
    initial = _State(0, (_Slot(),) * slots)
    pending, visited = deque([initial]), {initial}
    completed = set()
    while pending:
        state = pending.popleft()
        if not _valid(state):
            return len(visited), completed, state
        assert sum(slot.stage > 0 for slot in state.slots) <= slots
        if len(state.drained) == jobs and all(slot.stage == 0 for slot in state.slots):
            completed.add(tuple(sorted(state.drained)))
        for next_state in _moves(state, jobs, missing):
            if next_state not in visited:
                visited.add(next_state)
                pending.append(next_state)
    return len(visited), completed, None


@pytest.mark.parametrize("jobs,slots", tuple(itertools.product(range(1, 4), range(1, 3))))
def test_BOUNDED_all_two_slot_interleavings_preserve_tags_ownership_and_completed_outputs(jobs, slots):
    states, completed, counterexample = _explore(jobs, slots)
    assert counterexample is None
    assert completed == {tuple((job, job) for job in range(jobs))}
    assert states > jobs * 8


@pytest.mark.parametrize("missing", ["host-overwrite-after-upload", "upload-before-compute",
                                     "kernel-before-download", "download-before-host-read",
                                     "output-reuse-after-download", "gpu-input-reuse-after-kernel"])
def test_BOUNDED_omitting_each_lifetime_dependency_has_a_concrete_bad_execution(missing):
    states, _, counterexample = _explore(2, 2, missing)
    assert states > 1 and counterexample is not None and not _valid(counterexample)


def test_PROOF_independent_host_ceilings_include_geometry_and_explicit_snapshots_separately():
    slot, scratch, geometry, snapshot, first, second, active = z3.Ints(
        "slot scratch geometry snapshot first second active")
    solver = z3.Solver()
    solver.add(slot > 0, scratch >= 0, geometry >= 0, snapshot >= 0,
               first >= 0, second >= 0, first <= slot, second <= slot, active >= 0, active <= 2)
    live = z3.If(active >= 1, first, 0) + z3.If(active == 2, second, 0) + scratch + geometry + snapshot
    solver.add(live > 2 * slot + scratch + geometry + snapshot)
    assert solver.check() == z3.unsat


def test_PROOF_gpu_admission_accounts_all_resident_accumulators_before_view_first_execution():
    budget, tiles, voxel, rays, geometry, layout, extra = z3.Ints("budget tiles voxel rays geometry layout extra")
    base = rays + geometry + layout + extra
    admitted = tiles * voxel + base
    solver = z3.Solver()
    solver.add(budget > 0, tiles >= 1, voxel > 0, rays >= 0, geometry >= 0, layout >= 0, extra >= 0,
               admitted <= budget, z3.Or(tiles * voxel > budget, admitted - base > budget - base))
    assert solver.check() == z3.unsat
    # Omitting retained accumulators admits a concrete unsafe working set.
    solver = z3.Solver()
    solver.add(budget > 0, tiles > 1, voxel > 0, base >= 0,
               voxel + base <= budget, admitted > budget)
    assert solver.check() == z3.sat


@pytest.mark.parametrize("tile_order", [(0, 1), (1, 0)])
@pytest.mark.parametrize("view_order", [(0, 1), (1, 0)])
def test_PROOF_finite_symbolic_resident_rank_sum_then_download_matches_all_view_contributions(tile_order, view_order):
    """Two tiles, two nonempty ranks, one empty rank; exact symbolic SUM."""
    values = {(rank, tile, view): sp.Symbol(f"r{rank}t{tile}v{view}")
              for rank in range(2) for tile in range(2) for view in range(2)}
    downloads = []
    for tile in tile_order:
        resident = [sp.Integer(0), sp.Integer(0), sp.Integer(0)]
        for view in view_order:
            for rank in range(2):
                resident[rank] += values[rank, tile, view]
        # Empty rank participates in the same tile collective with its zero.
        combined = sum(resident)
        downloads.append((tile, combined))
    assert len(downloads) == 2 and len({tile for tile, _ in downloads}) == 2
    for tile, value in downloads:
        expected = sum(values[rank, tile, view] for rank in range(2) for view in range(2))
        assert sp.expand(value - expected) == 0


# Slice 4/5 extension. Prior protocol tests above remain unchanged. These
# statements concern exact algebra/idealized protocols, not implementation
# equivalence, floating-point convergence or physical multi-node performance.

from fractions import Fraction


def _slab_bounds(size, ranks, rank):
    base, extra = divmod(size, ranks)
    start = rank * base + min(rank, extra)
    return start, start + base + int(rank < extra)


@pytest.mark.parametrize("ranks", range(1, 6))
def test_BOUNDED_balanced_slabs_cover_without_overlap_including_empty_ranks(ranks):
    for size in range(10):
        intervals = [_slab_bounds(size, ranks, rank) for rank in range(ranks)]
        cells = [cell for begin, end in intervals for cell in range(begin, end)]
        counts = [end - begin for begin, end in intervals]
        assert cells == list(range(size)) and len(cells) == len(set(cells))
        assert max(counts) - min(counts) <= 1
        assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))
        if size < ranks:
            assert counts.count(0) == ranks - size


def test_PROOF_symbolic_slab_bounds_preserve_global_frame_and_adjacent_boundaries():
    size, ranks, rank = z3.Ints("slab_size slab_ranks slab_rank")
    base, extra = size / ranks, size % ranks
    begin = rank * base + z3.If(rank < extra, rank, extra)
    end = begin + base + z3.If(rank < extra, 1, 0)
    next_begin = (rank + 1) * base + z3.If(rank + 1 < extra, rank + 1, extra)
    solver = z3.Solver()
    solver.add(size >= 0, ranks > 0, rank >= 0, rank < ranks,
               z3.Or(begin < 0, end < begin, end > size,
                     z3.And(rank < ranks - 1, end != next_begin)))
    assert solver.check() == z3.unsat
    # The local cell is translated by its owned global offset, rather than
    # recentering its slab. Exact rational coordinates of either description.
    offset, local, global_size, pitch = sp.symbols("offset local global_size pitch")
    through_global = (offset + local + sp.Rational(1, 2) - global_size / 2) * pitch
    tile_centre = (offset - global_size / 2) * pitch
    through_tile = tile_centre + (local + sp.Rational(1, 2)) * pitch
    assert sp.expand(through_global - through_tile) == 0


@pytest.mark.parametrize("partition", ["views", "space"])
def test_PROOF_block_adjoint_and_replicated_cotangent_sum_include_an_empty_rank(partition):
    a = sp.Matrix(3, 3, lambda row, column: sp.Symbol(f"a{row}{column}"))
    x = sp.Matrix(sp.symbols("x0:3"))
    if partition == "space":
        blocks = (a[:, :1], a[:, 1:], sp.zeros(3, 0))
        pieces = (x[:1, :], x[1:, :], sp.zeros(0, 1))
        rays = sum((block * piece for block, piece in zip(blocks, pieces)), sp.zeros(3, 1))
        weights = [sp.Matrix([sp.Symbol(f"w{rank}_{row}") for row in range(3)]) for rank in range(3)]
        cotangent = sum(weights, sp.zeros(3, 1))
        lhs = sum((weight.T * rays)[0] for weight in weights)
        rhs = sum((piece.T * block.T * cotangent)[0] for block, piece in zip(blocks, pieces))
    else:
        blocks = (a[:1, :], a[1:, :], sp.zeros(0, 3))
        weights = [sp.Matrix(sp.symbols("w0:1")), sp.Matrix(sp.symbols("v0:2")), sp.zeros(0, 1)]
        lhs = sum((weight.T * block * x)[0] for weight, block in zip(weights, blocks))
        reduced = sum((block.T * weight for block, weight in zip(blocks, weights)), sp.zeros(3, 1))
        rhs = (x.T * reduced)[0]
    assert sp.expand(lhs - rhs) == 0
    # A second replicated volume SUM would multiply the intended adjoint.
    assert sp.expand(lhs - 3 * rhs) != 0


@pytest.mark.parametrize("partition", ["views", "space"])
def test_PROOF_norm_ownership_counts_replicated_state_once_and_owned_state_by_sum(partition):
    owned = sp.symbols("owned0 owned1", positive=True)
    replicated = sp.Symbol("replicated", positive=True)
    if partition == "space":
        gamma = sum((*owned, sp.Integer(0)))
        qq = sum((replicated, sp.Integer(0), sp.Integer(0)))
    else:
        gamma = sum((replicated, sp.Integer(0), sp.Integer(0)))
        qq = sum((*owned, sp.Integer(0)))
    expected = sum(owned) / replicated if partition == "space" else replicated / sum(owned)
    assert sp.simplify(gamma / qq - expected) == 0
    assert sp.simplify(gamma / (3 * qq) - expected) != 0


@pytest.mark.parametrize("tiles", [(0, 1, 3), (2, 0, 1), (1, 1, 0)])
def test_BOUNDED_common_collective_rounds_do_not_depend_on_tiles_gpus_or_local_grad_flags(tiles):
    rounds = (0, 1, 2)
    for local_gpus in ((1, 2, 1), (2, 1, 0)):
        for flags in itertools.product((False, True), repeat=3):
            # Local kernels differ. Every rank still reduces the same ray
            # rounds; a globally required backward supplies an internal
            # participation edge even for a rank with no differentiable input.
            forward = [tuple(("ray", batch) for batch in rounds) for _ in tiles]
            backward = [tuple(("cotangent", batch) for batch in rounds) if any(flags) else () for _ in tiles]
            assert len(set(forward)) == len(set(backward)) == 1
            native_work = [count * local_gpus[rank] for rank, count in enumerate(tiles)]
            assert native_work[tiles.index(0)] == 0
            if any(flags) and not all(flags):
                missing_participation = [backward[rank] if flag else () for rank, flag in enumerate(flags)]
                assert len(set(missing_participation)) > 1
    wrong_tile_loop = [tuple(("ray", batch) for _ in range(count) for batch in rounds) for count in tiles]
    assert len(set(wrong_tile_loop)) > 1


@dataclass(frozen=True)
class _QState:
    stage: int = 0
    q: int = 3
    gamma: int = 0
    qq: int = 0
    alpha: Fraction = Fraction(0)
    x: Fraction = Fraction(0)
    r: Fraction = Fraction(4, 3)


def _q_moves(state, early_reuse=False):
    if state.stage == 0:
        # Exact scalar instance A=3/2, y=4/3: s=p=2, gamma=4, q=3.
        yield replace(state, stage=1, gamma=4, qq=state.q ** 2)
    elif state.stage == 1:
        yield replace(state, stage=2)  # all rank qq contributions completed
    elif state.stage == 2:
        if state.qq:
            yield replace(state, stage=3, alpha=Fraction(state.gamma, state.qq))
        else:
            yield replace(state, stage=5)  # wrong overwritten q induces early stop
    elif state.stage == 3:
        yield replace(state, stage=4, x=state.x + state.alpha * 2)
    elif state.stage == 4:
        yield replace(state, stage=5, r=state.r - state.alpha * state.q)
    if state.q == 3 and (state.stage >= 5 or early_reuse):
        yield replace(state, q=0)


@pytest.mark.parametrize("early_reuse", [False, True])
def test_BOUNDED_q_survives_global_qq_alpha_and_residual_update_with_bad_reuse_counterexample(early_reuse):
    initial = _QState()
    pending, seen, bad, completed = deque([initial]), {initial}, [], []
    while pending:
        state = pending.popleft()
        if state.stage == 5:
            completed.append(state)
            if state.r != 0 or state.qq != 9 or state.x != Fraction(8, 9):
                bad.append(state)
        for next_state in _q_moves(state, early_reuse):
            if next_state not in seen:
                seen.add(next_state)
                pending.append(next_state)
    assert completed
    if early_reuse:
        assert bad, "q overwrite must produce a concrete wrong residual/qq"
    else:
        assert not bad and all(state.alpha == Fraction(4, 9) and state.qq == 9 for state in completed)


_CGLS_PHASE_REGIONS = {
    "projection": ("x", "s", "p", "r", "q", "pipeline"),
    "local-qq": ("x", "s", "p", "r", "q", "dot"),
    "global-qq": ("x", "s", "p", "r", "q"),
    "alpha": ("x", "s", "p", "r", "q"),
    "x-update": ("x", "s", "p", "r", "q"),
    "r-update": ("x", "s", "p", "r", "q"),
    "adjoint": ("x", "s", "p", "r", "q", "pipeline"),
    "gamma": ("x", "s", "p", "r", "q", "dot"),
    "p-update": ("x", "s", "p", "r", "q"),
}


def test_PROOF_idealized_cgls_region_ownership_bounds_each_phase_with_transient_scratch():
    """Independent named regions; exact bytes, excludes allocator rounding."""
    volume, ray, pipeline, dot = z3.Ints("region_volume region_ray region_pipeline region_dot")
    sizes = {"x": volume, "s": volume, "p": volume, "r": ray, "q": ray,
             "pipeline": pipeline, "dot": dot}
    estimate = 3 * volume + 2 * ray + pipeline + dot
    for phase, regions in _CGLS_PHASE_REGIONS.items():
        assert set(("x", "s", "p", "r", "q")) <= set(regions), phase
        assert len(regions) == len(set(regions)), "distinct live state regions cannot alias"
        actual = sum(sizes[name] for name in regions)
        solver = z3.Solver()
        solver.add(volume > 0, ray > 0, pipeline >= 0, dot >= 0, actual > estimate)
        assert solver.check() == z3.unsat, phase
    # qq/alpha/r-update retain all three volumes and both ray states; q cannot
    # be borrowed as dot/pipeline scratch while its residual use is outstanding.
    assert "q" in _CGLS_PHASE_REGIONS["r-update"]
    assert "dot" not in _CGLS_PHASE_REGIONS["r-update"]


@pytest.mark.parametrize("omitted", ["x", "s", "p", "r", "q", "pipeline", "dot"])
def test_PROOF_omitting_any_cgls_state_or_transient_has_an_actual_phase_budget_counterexample(omitted):
    volume, ray, pipeline, dot, budget = z3.Ints("omit_volume omit_ray omit_pipeline omit_dot omit_budget")
    sizes = {"x": volume, "s": volume, "p": volume, "r": ray, "q": ray,
             "pipeline": pipeline, "dot": dot}
    incomplete = sum(sizes[name] for name in sizes if name != omitted)
    phase = "local-qq" if omitted == "dot" else "projection"
    actual = sum(sizes[name] for name in _CGLS_PHASE_REGIONS[phase])
    solver = z3.Solver()
    solver.add(volume > 0, ray > 0, pipeline >= 0, dot >= 0,
               incomplete <= budget, actual > budget, budget > 0)
    assert solver.check() == z3.sat, (omitted, phase)
    model = solver.model()
    assert model.eval(actual).as_long() > model.eval(budget).as_long()
    assert model.eval(incomplete).as_long() <= model.eval(budget).as_long()
