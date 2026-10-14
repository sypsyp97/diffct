"""Real Gloo analytical ownership, gradients and bounded numerical storage."""

from contextlib import contextmanager
from datetime import timedelta
import json
import math
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.distributed as dist

import diffct
from tests.test_detector_surfaces import _case, _grids, _matrix
from tests.test_execution_features import (
    BEAMS, GLOO, _BoundedStore, _analytical_oracle, _no_whole_volume_collectives,
    _overrides, _owned, _random, _reconstruct, _run_four, _scan,
)


@contextmanager
def _bounded_reductions(case, phase, records):
    """Observe real reductions; one voxel forward, one ray batch backward."""
    original = dist.all_reduce

    def reduce(tensor, *args, **kwargs):
        if tensor.is_floating_point():
            limit = 2 * math.prod(case["sino_shape"][1:]) if phase[0] == "backward" else 1
            assert tensor.numel() <= limit, "analytical reduction exceeded its requested tile"
            op = kwargs.get("op", args[0] if args else dist.ReduceOp.SUM)
            assert op == dist.ReduceOp.SUM, "analytical numerical contributions must SUM"
            records.append((phase[0], tensor.numel()))
        return original(tensor, *args, **kwargs)

    with _no_whole_volume_collectives(), mock.patch.object(dist, "all_reduce", reduce):
        yield


def _analytical_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=25))
    try:
        records = []
        scenarios = [(beam, partition, grid, False) for beam in BEAMS
                     for partition, grid in (("views", None), ("space", None), ("hybrid", (2, 2)))]
        scenarios += [("parallel", partition, grid, True)
                      for partition, grid in (("views", None), ("space", None), ("hybrid", (2, 2)))]
        for beam, partition, grid, empty in scenarios:
            case = _scan(beam)
            if empty:
                case = {**case, "shape": (1, 4), "angles": case["angles"][:1],
                        "trajectory": tuple(value[:1] for value in case["trajectory"]),
                        "sino_shape": (1, *case["sino_shape"][1:])}
            view_ranks, space_ranks = grid or ((4, 1) if partition == "views" else (1, 4))
            view_rank, space_rank = divmod(rank, space_ranks)
            shard = _owned(len(case["angles"]), view_rank, view_ranks)
            slab = (_owned(case["shape"][0], space_rank, space_ranks),
                    *(slice(0, size) for size in case["shape"][1:]))
            full_data = _random(case["sino_shape"], 451, grad=True)
            overrides = _overrides(case)
            expected = _analytical_oracle(case, full_data, **overrides)
            cotangent = _random(case["shape"], 452)
            expected_gradient = torch.autograd.grad(expected, full_data, cotangent)[0]
            local_data = full_data.detach()[shard].clone().requires_grad_()
            phase, reductions = ["forward"], []
            options = dict(backend="torch", precision="float64", distributed=True,
                           partition=partition, partition_grid=grid, view_chunk_size=2,
                           volume_chunk_shape=(1,) * len(case["shape"]),
                           isocenter=case["center"], **overrides)
            with _bounded_reductions(case, phase, reductions):
                actual = _reconstruct(case, local_data, **options)
                assert actual.dtype == torch.float64 and actual.device.type == "cpu"
                assert tuple(actual.shape) == tuple(expected[slab].shape)
                torch.testing.assert_close(actual, expected.detach()[slab], rtol=2e-11, atol=2e-12)
                phase[0] = "backward"
                ((actual * cotangent[slab]).sum() / view_ranks).backward()
                assert local_data.grad is not None
                torch.testing.assert_close(local_data.grad, expected_gradient[shard], rtol=3e-11, atol=3e-12)
                phase[0] = "into"
                raw_source = diffct.TensorStore(local_data.detach())
                raw_output = diffct.TensorStore(torch.full_like(expected.detach()[slab], torch.nan))
                source = _BoundedStore(raw_source, view_limit=2)
                output = _BoundedStore(raw_output, spatial_limit=(1,) * len(case["shape"]))
                assert _reconstruct(case, source, into=output, **options) is output
                torch.testing.assert_close(raw_output.tensor, expected.detach()[slab], rtol=2e-11, atol=2e-12)
                torch.testing.assert_close(raw_source.tensor, full_data.detach()[shard], rtol=0, atol=0)
                if actual.numel():
                    assert output.writes
                    if local_data.numel():
                        assert source.reads
            if view_ranks > 1 and actual.numel():
                assert any(phase == "forward" and size for phase, size in reductions)
                assert any(phase == "into" and size for phase, size in reductions)
            if space_ranks > 1 and local_data.numel():
                assert any(phase == "backward" and size for phase, size in reductions)
            records.append(dict(beam=beam, partition=partition, empty=empty,
                                views=local_data.shape[0], first=actual.shape[0],
                                reads=len(source.reads), writes=len(output.writes), reductions=reductions))
        (Path(directory) / f"analytical-{rank}.json").write_text(json.dumps(records), encoding="utf-8")
    finally:
        dist.destroy_process_group()


@GLOO
def test_four_rank_analytical_views_space_hybrid_forward_data_gradients_and_bounded_into(tmp_path):
    _run_four(_analytical_worker, ((tmp_path / "analytical-rendezvous").resolve().as_uri(), str(tmp_path)),
              timeout=60)
    records = [json.loads((tmp_path / f"analytical-{rank}.json").read_text()) for rank in range(4)]
    assert all(len(rank_records) == 12 for rank_records in records)
    assert any(record["views"] == 0 for rank_records in records for record in rank_records)
    assert any(record["first"] == 0 for rank_records in records for record in rank_records)


def _asymmetric_grid_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=25))
    try:
        records = []
        for grid in ((1, 4), (4, 1)):
            for first, views in ((5, 5), (1, 1)):
                case = _case("parallel", views=views)
                case.shape, case.detector = (first, 4), (7,)
                u, v = _grids(case)
                matrix = torch.from_numpy(_matrix(case, np.stack((u, v, np.zeros_like(u)), -1)))
                truth, rays = _random(case.shape, 453), _random(case.sino_shape, 454)
                view_rank, space_rank = divmod(rank, grid[1])
                slab = (_owned(first, space_rank, grid[1]), slice(0, case.shape[1]))
                shard = _owned(views, view_rank, grid[0])
                with _no_whole_volume_collectives():
                    operator = diffct.Projector(case.trajectory, case.shape, case.detector, beam="parallel",
                                                detector_spacing=case.pitch[0], voxel_spacing=case.spacing,
                                                backend="torch", precision="float64", distributed=True,
                                                partition="hybrid", partition_grid=grid, view_chunk_size=1,
                                                volume_chunk_shape=(1, 1))
                    assert operator.global_volume_shape == case.shape
                    assert operator.volume_slice == slab and operator.view_slice == shard
                    assert operator.volume_shape == operator.local_volume_shape == tuple(truth[slab].shape)
                    assert operator.projection_shape == tuple(rays[shard].shape)
                    expected_rays = (matrix @ truth.flatten()).reshape(case.sino_shape)[shard]
                    expected_volume = (matrix.T @ rays.flatten()).reshape(case.shape)[slab]
                    torch.testing.assert_close(operator.project(truth[slab]), expected_rays, rtol=2e-10, atol=2e-11)
                    torch.testing.assert_close(operator.backproject(rays[shard]), expected_volume, rtol=2e-10, atol=2e-11)
                records.append(dict(grid=grid, views=operator.projection_shape[0], first=operator.volume_shape[0]))
        (Path(directory) / f"grid-{rank}.json").write_text(json.dumps(records), encoding="utf-8")
    finally:
        dist.destroy_process_group()


@GLOO
def test_four_rank_asymmetric_hybrid_grid_public_ownership_and_cell_oracle(tmp_path):
    _run_four(_asymmetric_grid_worker, ((tmp_path / "grid-rendezvous").resolve().as_uri(), str(tmp_path)),
              timeout=60)
    records = [json.loads((tmp_path / f"grid-{rank}.json").read_text()) for rank in range(4)]
    assert all(len(rank_records) == 4 for rank_records in records)
    assert any(record["views"] == 0 for rank_records in records for record in rank_records)
    assert any(record["first"] == 0 for rank_records in records for record in rank_records)
