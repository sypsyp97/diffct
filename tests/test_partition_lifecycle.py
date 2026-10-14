"""Public analytical calls must bound groups without invalidating live graphs."""

from datetime import timedelta
import gc
import json
from pathlib import Path
from unittest import mock

import torch
import torch.distributed as dist

from tests.test_execution_features import (
    GLOO, _analytical_oracle, _overrides, _owned, _random, _reconstruct,
    _run_four, _scan,
)


def _initialize(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=15))


def _hybrid(case, data, weights, *, group=None, grid=(2, 2)):
    return _reconstruct(case, data, backend="torch", precision="float64",
                        distributed=True, process_group=group, partition="hybrid",
                        partition_grid=grid, view_chunk_size=2,
                        volume_chunk_shape=(2, 3), **weights)


def _ownership(case, rank, grid=(2, 2)):
    view_rank, space_rank = divmod(rank, grid[1])
    views = _owned(len(case["angles"]), view_rank, grid[0])
    slab = (_owned(case["shape"][0], space_rank, grid[1]), slice(None))
    return views, slab


def _registered_group_count():
    # Diagnostic only: this registry strongly retains groups after tensor/GC
    # cleanup, unlike weak references to the temporary execution object.
    return len(dist.distributed_c10d._world.pg_map)


def _repeated_forward_worker(rank, rendezvous, directory):
    _initialize(rank, rendezvous)
    try:
        case, weights = _scan("parallel"), _overrides(_scan("parallel"))
        views, slab = _ownership(case, rank)
        counts = [_registered_group_count()]
        with torch.no_grad():
            for iteration in range(5):
                full_data = _random(case["sino_shape"], 670 + iteration)
                expected = _analytical_oracle(case, full_data, **weights)
                actual = _hybrid(case, full_data[views], weights)
                torch.testing.assert_close(actual, expected[slab], rtol=2e-11, atol=2e-12)
                del actual
                gc.collect()
                counts.append(_registered_group_count())
        (Path(directory) / f"groups-{rank}.json").write_text(json.dumps(counts), encoding="utf-8")
    finally:
        dist.destroy_process_group()


@GLOO
def test_repeated_public_hybrid_fbp_keeps_registered_groups_bounded_after_warmup(tmp_path):
    _run_four(_repeated_forward_worker,
              ((tmp_path / "reuse-rendezvous").resolve().as_uri(), str(tmp_path)), timeout=60)
    counts = [json.loads((tmp_path / f"groups-{rank}.json").read_text()) for rank in range(4)]
    assert all(len(rank_counts) == 6 for rank_counts in counts)
    assert all(max(rank_counts[1:]) <= rank_counts[1] for rank_counts in counts), \
        f"registered hybrid groups grew after warmup despite output deletion and gc: {counts}"


def _retained_graph_worker(rank, rendezvous):
    _initialize(rank, rendezvous)
    try:
        case = _scan("parallel")
        views, slab = _ownership(case, rank)
        cotangent = _random(case["shape"], 680)
        retained = []
        for seed in (681, 682):
            full_data = _random(case["sino_shape"], seed, grad=True)
            oracle_weights = _overrides(case, grad=True)
            expected = _analytical_oracle(case, full_data, **oracle_weights)
            expected_gradients = torch.autograd.grad(
                expected, (full_data, *oracle_weights.values()), cotangent)
            data = full_data.detach()[views].clone().requires_grad_()
            weights = {name: value.detach().clone().requires_grad_()
                       for name, value in oracle_weights.items()}
            actual = _hybrid(case, data, weights)
            torch.testing.assert_close(actual, expected.detach()[slab], rtol=2e-11, atol=2e-12)
            retained.append((actual, data, weights, expected_gradients))
        with torch.no_grad():
            for seed in (683, 684, 685):
                data = _random(case["sino_shape"], seed)
                weights = _overrides(case)
                actual = _hybrid(case, data[views], weights)
                expected = _analytical_oracle(case, data, **weights)
                torch.testing.assert_close(actual, expected[slab], rtol=2e-11, atol=2e-12)
                del actual
                gc.collect()
        # Backward crosses newer calls and runs the oldest retained graph twice.
        for index in (1, 0, 0):
            actual, data, weights, expected_gradients = retained[index]
            gradients = torch.autograd.grad((actual * cotangent[slab]).sum() / 2,
                                            (data, *weights.values()), retain_graph=True)
            torch.testing.assert_close(gradients[0], expected_gradients[0][views],
                                       rtol=3e-11, atol=3e-12)
            for gradient, expected_gradient in zip(gradients[1:], expected_gradients[1:]):
                assert expected_gradient.abs().max() > 0
                torch.testing.assert_close(gradient, expected_gradient, rtol=3e-11, atol=3e-12)
    finally:
        dist.destroy_process_group()


@GLOO
def test_retained_hybrid_fbp_graphs_backpropagate_after_later_calls_and_gc(tmp_path):
    _run_four(_retained_graph_worker, ((tmp_path / "graphs-rendezvous").resolve().as_uri(),),
              timeout=60)


def _explicit_parent_worker(rank, rendezvous):
    _initialize(rank, rendezvous)
    try:
        # Two independent parent handles have identical, noncontiguous members.
        # Caching by ranks/grid alone would improperly share their collectives.
        parents = []
        for _ in range(2):
            for members in ([0, 2], [1, 3]):
                group = dist.new_group(members, backend="gloo", use_local_synchronization=True)
                if rank in members:
                    parents.append(group)
        case = _scan("parallel")
        observed = []
        original = dist.all_reduce
        with torch.no_grad():
            for index, parent in enumerate(parents):
                views, slab = _ownership(case, dist.get_rank(parent), grid=(2, 1))
                data = _random(case["sino_shape"], 690 + index)
                weights = _overrides(case)
                expected = _analytical_oracle(case, data, **weights)
                groups = []

                def reduce(tensor, *args, **kwargs):
                    if tensor.is_floating_point():
                        groups.append(kwargs.get("group"))
                    return original(tensor, *args, **kwargs)

                with mock.patch.object(dist, "all_reduce", reduce):
                    actual = _hybrid(case, data[views], weights, group=parent, grid=(2, 1))
                torch.testing.assert_close(actual, expected[slab], rtol=2e-11, atol=2e-12)
                assert groups
                assert all(dist.get_process_group_ranks(group) == dist.get_process_group_ranks(parent)
                           for group in groups)
                observed.append(groups)
                del actual
                gc.collect()
        assert all(first is not second for first in observed[0] for second in observed[1]), \
            "independent parent process groups shared a hybrid numerical collective group"
    finally:
        dist.destroy_process_group()


@GLOO
def test_hybrid_fbp_respects_distinct_public_subgroup_parent_identities(tmp_path):
    _run_four(_explicit_parent_worker, ((tmp_path / "parents-rendezvous").resolve().as_uri(),),
              timeout=60)


def _reinitialize_worker(rank, rendezvous):
    case = _scan("parallel")
    views, slab = _ownership(case, rank)
    previous_world = None
    for epoch in range(2):
        _initialize(rank, f"{rendezvous}-{epoch}")
        try:
            assert dist.group.WORLD is not previous_world
            with torch.no_grad():
                for iteration in range(2):
                    data = _random(case["sino_shape"], 700 + 2 * epoch + iteration)
                    weights = _overrides(case)
                    actual = _hybrid(case, data[views], weights)
                    expected = _analytical_oracle(case, data, **weights)
                    torch.testing.assert_close(actual, expected[slab], rtol=2e-11, atol=2e-12)
                    del actual
                    gc.collect()
            dist.barrier()
            previous_world = dist.group.WORLD
        finally:
            dist.destroy_process_group()
        assert _registered_group_count() == 0


@GLOO
def test_hybrid_fbp_works_after_default_group_destruction_and_reinitialization(tmp_path):
    _run_four(_reinitialize_worker, ((tmp_path / "epochs-rendezvous").resolve().as_uri(),),
              timeout=60)
