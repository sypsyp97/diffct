"""Real slab TV remains finite when an unused boundary-face sum overflows."""

from datetime import timedelta
import json
import math
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from diffct import Projector, total_variation
from tests.test_detector_surfaces import _case
from tests.test_spatial_partition import _run_processes, gloo_required


EPS = 1e-6


def _constant_tv_worker(rank, rendezvous, directory, constant):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=10))
    try:
        case = _case("parallel", views=5)
        case.shape, case.detector = (4, 5), (5,)
        operator = Projector(case.trajectory, case.shape, case.detector, beam=case.beam,
                             voxel_spacing=case.spacing, detector_spacing=case.pitch[0],
                             backend="torch", precision="float64", distributed=True,
                             partition="space", view_chunk_size=1, volume_chunk_shape=(1, 1))
        assert operator.world_size == operator.space_world_size == 2
        assert operator.volume_shape == (2, 5)
        assert operator.volume_slice == (slice(rank * 2, rank * 2 + 2), slice(0, 5))
        records = []
        for reduction in ("mean", "sum"):
            volume = torch.full(operator.volume_shape, constant, dtype=torch.float64,
                                device="cpu", requires_grad=True)
            before = volume.detach().clone()
            assert torch.isfinite(volume).all()
            # Equal finite neighbors have exactly zero physical differences,
            # so neither the actual TV magnitude nor its gradient can overflow.
            assert all(torch.count_nonzero(torch.diff(volume.detach(), dim=axis)) == 0
                       for axis in range(volume.ndim))
            actual = total_variation(volume, operator=operator,
                                     voxel_spacing=(.53, .71), eps=EPS, reduction=reduction)
            gradient, = torch.autograd.grad(actual, volume)
            records.append(dict(rank=rank, reduction=reduction, tv=actual.detach().item(),
                                face_sum_overflows=torch.isinf(before[0].sum()).item(),
                                source_unchanged=torch.equal(volume.detach(), before),
                                finite_gradient=torch.isfinite(gradient).all().item(),
                                zero_gradient=torch.count_nonzero(gradient).item() == 0))
        dist.barrier()
        (Path(directory) / f"rank-{rank}.json").write_text(json.dumps(records), encoding="utf-8")
    finally:
        dist.destroy_process_group()


@gloo_required
@pytest.mark.parametrize("constant", [1.25, torch.finfo(torch.float64).max / 2],
                         ids=["moderate-constant", "overflowing-face-sum"])
def test_space_tv_constant_has_finite_scalar_and_zero_gradient(constant, tmp_path):
    _run_processes(_constant_tv_worker,
                   ((tmp_path / "tv-rendezvous").resolve().as_uri(), str(tmp_path), constant),
                   timeout=150)
    for rank in range(2):
        records = json.loads((tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8"))
        assert len(records) == 2
        for record in records:
            assert record["face_sum_overflows"] == math.isinf(5 * constant), record
            assert record["source_unchanged"], record
            assert record["finite_gradient"] and record["zero_gradient"], record
            # Every global voxel contributes sqrt(eps), including outer faces.
            expected = math.sqrt(EPS) * (20 if record["reduction"] == "sum" else 1)
            assert math.isfinite(record["tv"]), record
            assert record["tv"] == pytest.approx(expected, rel=2e-12, abs=2e-13), record
