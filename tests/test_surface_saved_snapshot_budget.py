"""Bound callback-saved snapshots independently of callback-owned allocations."""

from collections import Counter
from contextlib import contextmanager
import math
from unittest import mock

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from diffct import operators
from tests.test_detector_surfaces import _arc_offsets, _case, _projector
from tests.test_surface_snapshot_admission import _CudaValue, _metadata_inventory


def _snapshot_sizes(case, allocator):
    raw = [8 * case.views * math.prod(case.detector) * 3,
           *([8 * case.views * len(case.shape)] * len(case.trajectory))]
    return tuple((size + 511) // 512 * 512 if allocator == "native" else size for size in raw)


def _metadata_operator(detector=(4, 47)):
    case = _case("cone", views=3)
    case.shape, case.detector, case.pitch = (3, 4, 5), detector, (.52, .08)
    offsets = torch.from_numpy(_arc_offsets(case, 3.7, .037))
    operator = _projector(case, lambda u, v: offsets, backend="torch")
    operator.backend, operator._devices = "numba", (torch.device("cuda:0"),)
    return case, operator, _CudaValue.wrap(torch.ones(case.shape), "cuda:0", [])


@pytest.mark.parametrize("allocator", ("native", "cudaMallocAsync"))
def test_preliminary_full_inventory_charges_one_saved_arena_and_independent_geometry_snapshots(allocator):
    case, operator, source = _metadata_operator()
    snapshot_sizes = _snapshot_sizes(case, allocator)
    capacity = sum(snapshot_sizes)
    inventories = []
    reserve = operators._reservation_bytes

    def observed(sizes):
        inventories.append(tuple(sizes))
        return reserve(sizes)

    with _metadata_inventory({}), mock.patch.object(torch.cuda, "get_allocator_backend", return_value=allocator), \
            mock.patch.object(operators, "_reservation_bytes", observed):
        workspace = operator._surface_work_buffers(case.views, case.detector)
        plan = operator._execution_plan(source, True)
    assert plan is None and inventories
    required = Counter((*snapshot_sizes, capacity))
    assert not (required - Counter(workspace)), (
        "preliminary surface inventory omits rounded geometry snapshots or its one saved-tensor arena",
        tuple(workspace), tuple(required.elements()),
    )
    assert all(not (required - Counter(inventory)) for inventory in inventories), \
        "full reservation admission omitted the saved arena or independent geometry snapshots"


@pytest.mark.parametrize("constraint", ("payload", "allocation"))
def test_saved_arena_participates_in_full_payload_and_single_allocation_admission(constraint):
    case, operator, source = _metadata_operator()
    capacity = sum(_snapshot_sizes(case, "native"))
    assert capacity == 15872
    unlimited = 1 << 40
    budget = 100000 if constraint == "payload" else unlimited
    limit = capacity - 1 if constraint == "allocation" else unlimited
    with _metadata_inventory({}), mock.patch.object(torch.cuda, "get_allocator_backend", return_value="native"), \
            mock.patch.object(operators, "_memory_lower_bounds", return_value=(budget, limit, unlimited)), \
            mock.patch.object(operators, "_memory_budget", return_value=(budget, limit)):
        plan = operator._execution_plan(source, True)
    assert plan is not None, (
        f"full execution ignored the {capacity}-byte saved arena under {constraint} constraint "
        f"(payload={budget}, allocation={limit})"
    )
    assert 0 < plan.estimated_gpu_bytes <= plan.gpu_budget_bytes


def test_saved_arena_participates_in_native_segment_reservation_admission():
    case, operator, source = _metadata_operator((64, 64))
    assert sum(_snapshot_sizes(case, "native")) == 296960
    unlimited, reservation = 1 << 40, 2 * 1024 ** 2
    # This complete inventory fits one native 2 MiB small-pool segment only
    # when the arena is omitted. The arena requires a second segment. Once
    # full execution is rejected, the independent CUDA output consumes the
    # one available segment, leaving no reservation for even a streamed tile.
    with _metadata_inventory({}), mock.patch.object(torch.cuda, "get_allocator_backend", return_value="native"), \
            mock.patch.object(operators, "_memory_lower_bounds", return_value=(unlimited, unlimited, reservation)), \
            mock.patch.object(operators, "_reservation_budget", return_value=reservation):
        with pytest.raises(RuntimeError, match="budget.*minimum"):
            operator._execution_plan(source, True)


class _SavedSurface(torch.autograd.Function):
    """Real differentiable surface with explicitly saved caller-owned values."""

    @staticmethod
    def forward(ctx, base, *values):
        ctx.save_for_backward(*values)
        ctx.active = [value.requires_grad for value in values]
        ctx.scale = 1e-4 / max(1, sum(ctx.active))
        depth = sum(value.abs().square().double().mean() for value, active in zip(values, ctx.active) if active)
        result = base.clone()
        result[..., 2].add_(ctx.scale * depth)
        return result

    @staticmethod
    def backward(ctx, cotangent):
        values = ctx.saved_tensors
        weight = cotangent[..., 2].double().sum()
        gradients = [((2 * value * (ctx.scale / value.numel()) * weight).to(value.dtype) if active else None)
                     for value, active in zip(values, ctx.active)]
        return (cotangent, *gradients)


class _HookAllocations(TorchDispatchMode):
    def __init__(self, sources):
        super().__init__()
        self.sources, self.packing = sources, None
        self.packs, self.restored, self.geometry_copies = [], [], []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        tensors = [value for value in tree_flatten((args, kwargs or {}))[0] if isinstance(value, torch.Tensor)]
        inputs = {(value.device, value.untyped_storage()._cdata) for value in tensors if value.layout == torch.strided}
        result = function(*args, **(kwargs or {}))
        outputs = [value for value in tree_flatten(result)[0] if isinstance(value, torch.Tensor)]
        if self.packing is not None:
            for value in outputs:
                if (value.is_cuda and value.layout == torch.strided
                        and (value.device, value.untyped_storage()._cdata) not in inputs):
                    self.packing["gpu_allocations"].append(value.untyped_storage().nbytes())
            if (str(function) == "aten._to_copy.default" and args[0].is_cuda
                    and isinstance(result, torch.Tensor) and result.device.type == "cpu"):
                self.packing["cpu_copies"] += 1
        elif (str(function) == "aten.clone.default" and args[0].is_cuda
              and args[0].untyped_storage()._cdata in self.sources):
            self.geometry_copies.append(result.nbytes)
        return result


@contextmanager
def _snapshot_hooks(sources):
    original = torch.autograd.graph.saved_tensors_hooks
    trace = _HookAllocations(sources)

    def observed(pack, unpack):
        def packing(value):
            record = dict(shape=tuple(value.shape), dtype=value.dtype, layout=value.layout,
                          bytes=value.numel() * value.element_size(), element_size=value.element_size(),
                          gpu_allocations=[], cpu_copies=0)
            trace.packs.append(record)
            trace.packing = record
            try:
                return pack(value)
            finally:
                trace.packing = None

        def unpacking(saved):
            value = unpack(saved)
            record = dict(shape=tuple(value.shape), dtype=value.dtype, device=value.device,
                          layout=value.layout, value=value.detach().cpu().clone())
            if value.layout == torch.strided:
                record.update(storage=value.untyped_storage()._cdata,
                              offset=value.storage_offset() * value.element_size(),
                              pointer=value.data_ptr())
            trace.restored.append(record)
            return value

        return original(packing, unpacking)

    with trace, mock.patch.object(torch.autograd.graph, "saved_tensors_hooks", observed):
        yield trace


def _saved_values(kind, capacity):
    if kind == "oversized":
        values = [torch.full((100000,), .02, dtype=torch.float64, device="cuda"),
                  torch.tensor(.03, dtype=torch.float64, device="cuda")]
    elif kind == "cumulative":
        # Every save fits individually; their cumulative payload exceeds the arena.
        count = capacity // 512 + 2
        values = [torch.full((8, 8), .01 * (index + 1), dtype=torch.float64, device="cuda")
                  for index in range(count)]
    elif kind == "mixed":
        values = [torch.tensor([1, 2, 3], dtype=torch.uint8, device="cuda"),
                  (torch.arange(4, dtype=torch.float64, device="cuda").reshape(2, 2)
                   .to(torch.complex128) * (.01 + .02j)).T,
                  torch.linspace(.01, .2, 15, dtype=torch.float32, device="cuda").reshape(3, 5).T,
                  torch.arange(6, dtype=torch.float16, device="cuda")[::2] * .01,
                  torch.tensor([True], device="cuda"),
                  torch.arange(8, dtype=torch.bfloat16, device="cuda")[::2] * .01,
                  torch.arange(6, dtype=torch.float64, device="cuda").reshape(2, 3).T * .01]
    else:
        values = [torch.sparse_coo_tensor(torch.tensor([[0, 2], [1, 3]], device="cuda"),
                                         torch.tensor([.1, .2], dtype=torch.float64, device="cuda"),
                                         (4, 5), check_invariants=True),
                  torch.tensor(.03, dtype=torch.float64, device="cuda")]
    return [value.requires_grad_() if value.layout == torch.strided
            and (value.is_floating_point() or value.is_complex()) else value for value in values]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None,
                    reason="real NVIDIA CUDA snapshots are required")
@pytest.mark.parametrize("kind", ("oversized", "cumulative", "mixed", "sparse"))
def test_callback_saved_values_use_one_bounded_arena_and_independent_cpu_fallback(kind):
    case = _case("cone", views=3)
    case.trajectory = tuple(value.cuda() for value in case.trajectory)
    allocator = torch.cuda.get_allocator_backend()
    capacity = sum(_snapshot_sizes(case, allocator))
    base = torch.from_numpy(_arc_offsets(case, 3.7, .037)).cuda().requires_grad_()
    values = _saved_values(kind, capacity)
    if kind == "mixed":
        assert any(value.layout == torch.strided and not value.is_contiguous() for value in values)
    originals = [value.detach().clone() for value in values]
    active = [value.requires_grad for value in values]
    scale = 1e-4 / sum(active)
    expected = base.detach().clone()
    expected[..., 2].add_(scale * sum(value.abs().square().double().mean()
                                    for value, needed in zip(originals, active) if needed))
    operator = _projector(case, lambda u, v: base, backend="numba", distributed=False)
    sources = {value.untyped_storage()._cdata for value in operator._trajectory}

    def callback(u, v):
        result = _SavedSurface.apply(base, *values)
        sources.add(result.untyped_storage()._cdata)
        return result

    operator.detector_surface = callback
    source = torch.ones(case.shape, device="cuda")
    unlimited = 1 << 40
    with mock.patch.object(operators, "_memory_lower_bounds", return_value=(100000, unlimited, unlimited)), \
            mock.patch.object(operators, "_memory_budget", return_value=(100000, unlimited)), \
            mock.patch.object(operators, "_reservation_budget", return_value=unlimited), _snapshot_hooks(sources) as trace:
        plan, spec, geometry = operator._prepare_execution(source, True)
    assert plan is None and spec.kind == "offsets"
    torch.testing.assert_close(geometry[-1], expected, rtol=0, atol=0)
    with torch.no_grad():
        for value in values:
            if value.layout != torch.strided:
                value._values().add_(.5)
            elif value.dtype == torch.bool:
                value.logical_not_()
            else:
                value.add_(1)
    weight = torch.linspace(.2, 1.1, base.numel(), device="cuda", dtype=torch.float64).reshape(base.shape)
    gradients = torch.autograd.grad((geometry[-1] * weight).sum(), (base, *(value for value in values if value.requires_grad)))
    torch.testing.assert_close(gradients[0], weight, rtol=0, atol=0)
    for gradient, original in zip(gradients[1:], (value for value, needed in zip(originals, active) if needed)):
        wanted = (2 * original * (scale / original.numel()) * weight[..., 2].sum()).to(original.dtype)
        torch.testing.assert_close(gradient, wanted, rtol=0, atol=0)
    assert len(trace.packs) == len(trace.restored) == len(values)
    for restored, original in zip(trace.restored, originals):
        assert restored["shape"] == tuple(original.shape) and restored["dtype"] == original.dtype
        assert restored["device"] == original.device and restored["layout"] == original.layout
        torch.testing.assert_close(restored["value"], original.cpu(), rtol=0, atol=0)
    allocations = [size for record in trace.packs for size in record["gpu_allocations"]]
    assert allocations == [capacity], f"saved snapshots allocated {allocations}; expected one {capacity}-byte arena"
    used, packed_storages = 0, set()
    for record, restored in zip(trace.packs, trace.restored):
        aligned = math.ceil(used / record["element_size"]) * record["element_size"]
        fits = record["layout"] == torch.strided and aligned + record["bytes"] <= capacity
        if fits:
            assert record["cpu_copies"] == 0, "an admitted local saved tensor was staged on CPU"
            packed_storages.add(restored["storage"])
            assert restored["offset"] == aligned and restored["pointer"] % record["element_size"] == 0
            used = aligned + record["bytes"]
        else:
            assert record["cpu_copies"] > 0 and not record["gpu_allocations"], (
                "oversized, excess, or non-strided saved tensor did not use independent CPU fallback", record,
            )
    assert len(packed_storages) == 1 and used <= capacity
    assert len(trace.geometry_copies) == len(case.trajectory) + 1, \
        "frame/offset snapshots no longer have independent storage"
