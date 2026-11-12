"""Live portable store plans agree with actual CUDA work and peer SUMs.

Only memory-inventory metadata is constrained. Stores, traversal, CUDA tensor
operations, copies, and Gloo transport remain real; no physical pressure is used.
"""

from dataclasses import replace
from datetime import timedelta
import math
from unittest import mock

import pytest
import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode

from diffct import TensorStore, backends
from tests._multiprocessing import spawn
from tests.test_chunked_projector import _memory_inventory
from tests.test_reference_backend import _case, _data, _matrix, _operator


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="physical CUDA required")
GLOO = pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(),
                         reason="two real Gloo ranks required")
HOST_BYTES = 8 * 1024 ** 2  # Documented minimum portable host-staging limit.


def _work_bytes(tile, rays, *, host=False):
    """Documented FP64, unbatched work model, applied to observed rectangles.

    This is a bound on live work, rather than the largest individual tensor.
    The oracle never invokes the production planner or its work-model methods.
    """
    if host:
        return math.prod(tile) * (2 * 8 + 32) + rays * (2 * 8 + 64)
    planes = sum(tile) + len(tile) + 2
    return (math.prod(tile) * (3 * 8 + 32)
            + rays * (planes * ((4 + len(tile) + 2) * 8 + 40) + 2 * 8 + 1024))


# Leave half the work budget for rays, with a nontrivial two-voxel tile.
SMALL_BYTES = 2 * _work_bytes((1, 2), 1)


class _Store(TensorStore):
    def __init__(self, tensor):
        super().__init__(tensor)
        self.reads, self.writes = [], []
        self.flushes = 0

    def read(self, selected):
        value = super().read(selected)
        self.reads.append(tuple(value.shape))
        return value

    def write(self, selected, value, *, accumulate=False):
        super().write(selected, value, accumulate=accumulate)
        self.writes.append(tuple(value.shape))

    def flush(self):
        super().flush()
        self.flushes += 1


class _Allocations(TorchDispatchMode):
    def __init__(self, *caller_arrays):
        super().__init__()
        self.caller = {value.untyped_storage().data_ptr() for value in caller_arrays}
        self.largest = {}

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        pending = [result]
        while pending:
            value = pending.pop()
            if isinstance(value, torch.Tensor):
                if value.untyped_storage().data_ptr() not in self.caller:
                    kind = value.device.type
                    self.largest[kind] = max(self.largest.get(kind, 0),
                                             value.numel() * value.element_size())
            elif isinstance(value, (tuple, list)):
                pending.extend(value)
        return result


def _observe_intervals(records):
    original = backends.ReferenceProjector._intervals

    def observed(reference, origins, directions, tile):
        indices, lengths = original(reference, origins, directions, tile)
        assert indices.shape == lengths.shape and indices.shape[0] == len(origins)
        records.append((tuple(part.stop - part.start for part in tile),
                        len(origins), origins.device.type))
        return indices, lengths

    return mock.patch.object(backends.ReferenceProjector, "_intervals", observed)


def _assert_work(owner, source, output, pairs, allocations, method, budget):
    stats = owner.last_execution_stats
    assert stats is not None and stats["backend"] == "torch"
    assert stats["operation"] == method
    assert stats["devices"] == [str(torch.device("cuda", torch.cuda.current_device()))]
    spatial = source.reads if method == "project" else output.writes
    ray_batches = output.writes if method == "project" else source.reads
    declared_tile = tuple(stats["chunk_shape"])
    declared_rays = (stats["view_chunk_size"], *stats["detector_chunk_shape"])
    for observed, declared, extent in ((spatial, declared_tile, owner.volume_shape),
                                       (ray_batches, declared_rays, owner.projection_shape)):
        assert observed
        assert all(all(size <= limit for size, limit in zip(shape, declared))
                   for shape in observed), "actual store rectangle exceeds reported plan"
        assert tuple(max(shape[axis] for shape in observed) for axis in range(len(declared))) == tuple(
            min(limit, size) for limit, size in zip(declared, extent)
        ), "reported plan differs from actual store rectangles"
    assert pairs and all(device == "cuda" for _, _, device in pairs)
    for tile, rays, _ in pairs:
        assert all(size <= limit for size, limit in zip(tile, declared_tile))
        assert rays <= math.prod(declared_rays)
        assert _work_bytes(tile, rays) <= budget, "actual traversal exceeds live CUDA work budget"
        assert _work_bytes(tile, rays, host=True) <= HOST_BYTES, "actual staging exceeds host work budget"
    assert allocations.largest.get("cuda", 0) > 0
    assert allocations.largest["cuda"] <= budget
    assert allocations.largest.get("cpu", 0) <= HOST_BYTES
    assert stats["ray_tile_pairs"] == stats["store_reads"] == len(source.reads) == len(pairs)
    assert stats["store_writes"] == len(output.writes) > 0
    assert output.flushes == 1


def _tiny_case():
    return replace(_case("parallel"), shape=(4, 4))


def _trial(case, owner, method, budget, *, expected=None, data=None):
    source_shape = owner.volume_shape if method == "project" else owner.projection_shape
    output_shape = owner.projection_shape if method == "project" else owner.volume_shape
    if data is None:
        data = _data(source_shape, phase=.27)
    if expected is None:
        matrix = _matrix(case)
        expected = ((matrix if method == "project" else matrix.T) @ data.flatten()).reshape(output_shape)
    source = _Store(data.clone())
    output = _Store(torch.full(output_shape, torch.nan, dtype=torch.float64))
    pairs = []
    allocations = _Allocations(source.tensor, output.tensor)
    # Quarter of available CUDA memory is the portable work limit. Host stores
    # use the documented 8 MiB floor even under this small logical inventory.
    with _memory_inventory({torch.cuda.current_device(): 4 * budget}), \
            mock.patch.object(backends, "_host_free_bytes", return_value=4 * HOST_BYTES), \
            _observe_intervals(pairs), allocations:
        result = getattr(owner, method + "_into")(source, output)
    torch.cuda.synchronize()
    assert result is output
    torch.testing.assert_close(output.tensor, expected, rtol=3e-11, atol=3e-11)
    torch.testing.assert_close(source.tensor, data, rtol=0, atol=0)
    _assert_work(owner, source, output, pairs, allocations, method, budget)
    return owner.last_execution_stats


@pytest.mark.cuda
@CUDA
@pytest.mark.parametrize("method", ["project", "backproject"])
def test_portable_into_automatic_plan_uses_call_time_memory_and_reports_actual_work(method):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        case = _tiny_case()
        with _memory_inventory({torch.cuda.current_device(): 16 * SMALL_BYTES}):
            owner = _operator(case, devices=[torch.cuda.current_device()])
        assert (owner.volume_chunk_shape, owner.view_chunk_size, owner.detector_chunk_shape) == (None, None, None)
        _trial(case, owner, method, SMALL_BYTES)
    finally:
        torch.set_num_threads(previous)


@pytest.mark.cuda
@CUDA
def test_portable_into_cpu_stores_keep_host_staging_bound_with_larger_cuda_room():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # A 1.5 MiB caller volume: its full host work model exceeds 8 MiB,
        # while its CUDA work fits 32 MiB. Host inventory must affect the plan.
        directions = torch.tensor([[1., 0.], [-1., 0.]], dtype=torch.float64)
        centers = torch.tensor([[.17, .21], [-.13, -.17]], dtype=torch.float64)
        axes = torch.tensor([[0., 1.], [0., 1.]], dtype=torch.float64)
        case = replace(_case("parallel"), shape=(512, 384), detector=(1,),
                       trajectory=(directions, centers, axes), spacing=(1., 1.), pitch=(1.,))
        owner = _operator(case, devices=[torch.cuda.current_device()])
        data = torch.full(case.shape, 1.25, dtype=torch.float64)
        expected = torch.full(case.projection_shape, case.shape[1] * 1.25, dtype=torch.float64)
        _trial(case, owner, "project", 4 * HOST_BYTES, expected=expected, data=data)
    finally:
        torch.set_num_threads(previous)


def _shared_plan_worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.cuda.set_device(0)  # Both ranks intentionally share one physical GPU.
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        case = _tiny_case()
        owner = _operator(case, devices=[0], distributed=True, partition="space")
        image = _data(case.shape, phase=.27)
        expected = (_matrix(case) @ image.flatten()).reshape(case.projection_shape)
        original = dist.all_reduce
        sums = []

        def checked_sum(value, *args, **kwargs):
            if value.is_floating_point():
                group = kwargs.get("group")
                signatures = [None, None]
                signature = (tuple(value.shape), str(value.dtype))
                dist.all_gather_object(signatures, signature, group=group)
                assert all(item == signature for item in signatures), "peer numerical SUM shapes differ"
                sums.append(signature)
            return original(value, *args, **kwargs)

        # Unequal logical inventories must produce the smallest shared work
        # plan before real numerical SUMs. This is single-GPU Gloo evidence.
        local_budget = SMALL_BYTES * (1 if rank == 0 else 16)
        with mock.patch.object(dist, "all_reduce", checked_sum):
            stats = _trial(case, owner, "project", local_budget, expected=expected,
                           data=image[owner.volume_slice].clone())
        assert sums
        plans = [None, None]
        plan = (stats["chunk_shape"], stats["view_chunk_size"], stats["detector_chunk_shape"])
        dist.all_gather_object(plans, plan)
        assert plans[0] == plans[1], "ranks reported different automatic work plans"
    finally:
        dist.destroy_process_group()


@pytest.mark.cuda
@CUDA
@GLOO
def test_portable_into_two_gloo_ranks_share_smallest_live_plan_before_numerical_sum(tmp_path):
    spawn(_shared_plan_worker, args=((tmp_path / "shared-plan").resolve().as_uri(),),
          nprocs=2, join=True)
