"""Surface phase reservation under a finite, independent native allocator.

BOUNDED: enumerate every reachable allocation/free interleaving for the size
domains below. Each pipeline, composition and replay request is allocated once
and freed once. All composition requests retire before any replay allocation.
Pipeline requests can allocate and free anywhere, including between phases.

Assumptions: one owner, initially empty cache, best-fit splitting, adjacent free
block coalescing, 512-byte rounding, separate small/large pools, 2-MiB small
segments, 20-MiB middle segments and 2-MiB rounding for large segments. Ties use
segment birth order then block offset. Wholly idle segments remain reusable.
Excluded: preexisting holes/cache, repeated epochs, pending stream events,
concurrent owners, allocator overrides, sampler internals and actual CUDA.
Source guards bind selected admission expressions, not complete code semantics.
"""

import ast
import inspect
from itertools import product
from textwrap import dedent
from unittest import mock

import pytest
import torch

from diffct import Projector, chunking
from diffct.surfaces import ParameterizedSurface, _GeometrySpec


MIB = 1024 ** 2
FREE = -1


def _declared(owner, name):
    function = getattr(owner, name)
    assert callable(function), f"INTERFACE_PENDING: {owner.__name__}.{name}"
    return function


@pytest.fixture
def native_allocator():
    with mock.patch.object(torch.cuda, "get_allocator_backend", return_value="native"):
        yield


def _allocate(segments, owner, raw):
    """Immutable segments contain consecutive (owner, byte-count) blocks."""
    size = (raw + 511) // 512 * 512
    pool = "small" if size <= MIB else "large"
    fitting = [(amount, segment_index, block_index)
               for segment_index, (name, blocks) in enumerate(segments) if name == pool
               for block_index, (tag, amount) in enumerate(blocks)
               if tag == FREE and amount >= size]
    if fitting:
        amount, segment_index, block_index = min(fitting)
    else:
        amount = (2 * MIB if size <= MIB else 20 * MIB if size < 10 * MIB else
                  (size + 2 * MIB - 1) // (2 * MIB) * (2 * MIB))
        segment_index, block_index = len(segments), 0
        segments += ((pool, ((FREE, amount),)),)
    name, blocks = segments[segment_index]
    replacement = ((owner, size),) + (((FREE, amount - size),) if amount > size else ())
    changed = (name, blocks[:block_index] + replacement + blocks[block_index + 1:])
    return segments[:segment_index] + (changed,) + segments[segment_index + 1:]


def _free(segments, owner):
    found = 0
    result = []
    for pool, blocks in segments:
        merged = []
        for tag, amount in blocks:
            if tag == owner:
                found += 1
                tag = FREE
            if tag == FREE and merged and merged[-1][0] == FREE:
                merged[-1] = (FREE, merged[-1][1] + amount)
            else:
                merged.append((tag, amount))
        result.append((pool, tuple(merged)))
    assert found == 1, (owner, segments)
    return tuple(result)


def _reserved(segments):
    return sum(amount for _, blocks in segments for _, amount in blocks)


def _explore(pipeline, composition, replay, bound):
    """Merge identical states; explore every outgoing legal event from each."""
    sizes = pipeline + composition + replay
    composition_ids = range(len(pipeline), len(pipeline) + len(composition))
    replay_start = len(pipeline) + len(composition)
    pending = [(tuple(0 for _ in sizes), ())]
    seen = set()
    completed = 0
    while pending:
        state = pending.pop()
        if state in seen:
            continue
        seen.add(state)
        status, segments = state
        assert _reserved(segments) <= bound, (sizes, status, segments, bound)
        if all(value == 2 for value in status):
            completed += 1
        replay_ready = all(status[index] == 2 for index in composition_ids)
        assert not (any(status[index] == 1 for index in composition_ids)
                    and any(value == 1 for value in status[replay_start:]))
        for index, value in enumerate(status):
            if value == 2 or (value == 0 and index >= replay_start and not replay_ready):
                continue
            changed = list(status)
            changed[index] += 1
            blocks = (_allocate(segments, index, sizes[index]) if value == 0 else
                      _free(segments, index))
            pending.append((tuple(changed), blocks))
    assert completed > 0
    return len(seen)


@pytest.mark.parametrize("counts,domain", [
    ((1, 1, 1), (511, 512, MIB, MIB + 1, 4 * MIB,
                  10 * MIB - 512, 10 * MIB, 10 * MIB + 1)),
    ((2, 2, 2), (MIB // 2 + 1, MIB)),
], ids=["pool-and-rounding-boundaries", "split-holes-and-pipeline-pins"])
def test_BOUNDED_union_bounds_every_modeled_phase_and_pipeline_order(counts, domain, native_allocator):
    reserve = _declared(chunking, "_reservation_bytes")
    inventories = states = 0
    for sizes in product(domain, repeat=sum(counts)):
        first, second = counts[0], counts[0] + counts[1]
        pipeline, composition, replay = sizes[:first], sizes[first:second], sizes[second:]
        states += _explore(pipeline, composition, replay, reserve(sizes))
        inventories += 1
    assert inventories == len(domain) ** sum(counts)
    assert states > inventories


def _surface_inventory():
    # Metadata only: no sampler call, tensor transfer or native kernel is allowed.
    def sampler(*args):
        raise AssertionError("metadata-only surface inventory must not sample")
    operator = object.__new__(Projector)
    operator.beam = "fan"
    operator.volume_shape = (400, 400)
    operator.detector_shape = (5,)
    operator._geometry_trainable = True
    operator.detector_surface = ParameterizedSurface(
        sampler, (torch.tensor(0., dtype=torch.float64, requires_grad=True),))
    spec = _GeometrySpec("sampled", sampler, torch.float64)
    phases = _declared(Projector, "_surface_work_phases")(
        operator, 12000, (5,), 1, spec, streamed=True, batched=True)
    pipeline = _declared(chunking, "_pipeline_buffers")(
        "fan", (400, 400), 12000, (5,), True, slots=1, geometry_grad=True)
    return operator, spec, pipeline, phases


def test_BOUNDED_pinned_small_segments_disprove_the_per_phase_maximum(native_allocator):
    operator, spec, pipeline, phases = _surface_inventory()
    assert pipeline == (640000, 240128, 96256, 480256, 96256, 480256)
    assert phases == ((576000, 960512), (512,) * 14 + (1440256,))
    composition, replay = phases
    sizes = pipeline + composition + replay
    composition_ids = range(len(pipeline), len(pipeline) + len(composition))
    segments = ()
    # Compose first, then allocate persistent pipeline/gradient buffers. Their
    # small blocks pin both segments after every composition tensor is freed.
    for index in (*composition_ids, *range(len(pipeline))):
        segments = _allocate(segments, index, sizes[index])
    for index in composition_ids:
        segments = _free(segments, index)
    assert _reserved(segments) == 4 * MIB
    assert len(segments) == 2
    assert all(any(0 <= owner < len(pipeline) for owner, _ in blocks)
               for _, blocks in segments)
    for index in range(len(pipeline) + len(composition), len(sizes)):
        segments = _allocate(segments, index, sizes[index])
    actual = _reserved(segments)
    reserve = _declared(chunking, "_reservation_bytes")
    per_phase = max(reserve(pipeline + phase) for phase in phases)
    union = reserve(pipeline + tuple(size for phase in phases for size in phase))
    assert actual == 24 * MIB > per_phase == 22 * MIB
    assert union == 28 * MIB >= actual
    reserved = _declared(Projector, "_surface_reserved_buffers")(
        operator, 12000, (5,), 1, spec, streamed=True, batched=True)
    assert reserved == composition + replay
    assert reserve(pipeline + reserved) == union


@pytest.mark.parametrize("phases", [
    ((),), ((512,), ()), ((512, 512, 512), (1024,)),
    ((576000, 960512), (512,) * 14 + (1440256,)),
])
def test_BOUNDED_payload_peak_and_largest_allocation_keep_separate_charges(phases):
    operator = object.__new__(Projector)
    with mock.patch.object(Projector, "_surface_work_phases", return_value=phases):
        workspace = _declared(Projector, "_surface_work_buffers")(operator)
        union = _declared(Projector, "_surface_reserved_buffers")(operator)
    assert sum(workspace) == max(map(sum, phases))
    assert max(workspace, default=0) == max((size for phase in phases for size in phase), default=0)
    assert union == tuple(size for phase in phases for size in phase)


def _union_source_sites(name, source):
    """Guard the selected AST expressions, without claiming full equivalence."""
    tree = ast.parse(dedent(source))
    sites = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_reservation_bytes" and len(node.args) == 1):
            inventory = node.args[0]
        elif name == "_plan_buffers" and isinstance(node, ast.Return):
            inventory = node.value
        else:
            continue
        if not (isinstance(inventory, ast.BinOp) and isinstance(inventory.op, ast.Add)
                and isinstance(inventory.right, ast.Call)
                and isinstance(inventory.right.func, ast.Attribute)
                and ast.unparse(inventory.right.func) == "self._surface_reserved_buffers"):
            continue
        left = inventory.left
        label = left.id if isinstance(left, ast.Name) else ast.unparse(left.func)
        assert any(keyword.arg == "streamed" and ast.unparse(keyword.value) == "True"
                   for keyword in inventory.right.keywords), "union reservation must use streamed phases"
        sites.append((label, node, inventory))
    expected = {"_execution_plan": ["_pipeline_buffers", "buffers"],
                "_candidate_plans": ["_pipeline_buffers"],
                "_plan_buffers": ["_pipeline_buffers"]}
    assert sorted(label for label, _, _ in sites) == expected[name], f"{name}: union reservation source drift"
    return tree, sites


def test_BOUNDED_reserved_inventory_source_flattens_every_phase():
    source = dedent(inspect.getsource(_declared(Projector, "_surface_reserved_buffers")))
    returns = [ast.unparse(node.value) for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.Return)]
    assert returns == ["tuple((size for phase in self._surface_work_phases(*args, **kwargs) for size in phase))"]


@pytest.mark.parametrize("name,label", [
    ("_execution_plan", "buffers"),
    ("_execution_plan", "_pipeline_buffers"),
    ("_candidate_plans", "_pipeline_buffers"),
    ("_plan_buffers", "_pipeline_buffers"),
])
def test_BOUNDED_union_admission_source_guard_rejects_per_phase_mutation(name, label):
    source = inspect.getsource(_declared(Projector, name))
    tree, sites = _union_source_sites(name, source)
    target, inventory = next((node, inventory) for key, node, inventory in sites if key == label)
    # Mutate only the in-memory AST; production files are never written.
    phases = inventory.right
    phases.func.attr = "_surface_work_phases"
    one_phase = ast.BinOp(left=inventory.left, op=ast.Add(), right=ast.Name(id="phase", ctx=ast.Load()))
    element = (one_phase if name == "_plan_buffers" else
               ast.Call(func=ast.Name(id="_reservation_bytes", ctx=ast.Load()), args=[one_phase], keywords=[]))
    generator = ast.GeneratorExp(elt=element, generators=[ast.comprehension(
        target=ast.Name(id="phase", ctx=ast.Store()), iter=phases, ifs=[], is_async=0)])
    maximum = ast.Call(func=ast.Name(id="max", ctx=ast.Load()), args=[generator],
                       keywords=([ast.keyword(arg="key", value=ast.Name(id="_reservation_bytes", ctx=ast.Load()))]
                                 if name == "_plan_buffers" else []))
    if isinstance(target, ast.Return):
        target.value = maximum
    else:
        target.func, target.args, target.keywords = maximum.func, maximum.args, maximum.keywords
    with pytest.raises(AssertionError, match="union reservation source drift"):
        _union_source_sites(name, ast.unparse(ast.fix_missing_locations(tree)))
