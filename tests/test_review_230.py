"""Angle precision, input order and distributed chunk-hint contracts."""

from datetime import timedelta
import math

import pytest
import torch
import torch.distributed as dist

from diffct import Projector, angular_integration_weights, parker_weights
from tests.test_spatial_partition import _run_processes, gloo_required


@pytest.mark.parametrize("order", [[0, 1, 2, 3, 4], [3, 0, 4, 2, 1]],
                         ids=["sorted", "shuffled"])
def test_fp32_angular_weights_preserve_single_ulp_gaps(order):
    angles = torch.tensor([-3, 1, 1 + 2**-23, 1 + 2**-22, 3], dtype=torch.float32)
    # Compute trapezoidal widths from the represented input in higher precision.
    gaps = angles.double().diff()
    expected = torch.cat((gaps[:1] / 2, (gaps[:-1] + gaps[1:]) / 2,
                          gaps[-1:] / 2)).float()
    actual = angular_integration_weights(angles[order])
    assert actual.dtype == torch.float32
    assert actual[order.index(2)].item() == 2**-23
    torch.testing.assert_close(actual, expected[order], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_parker_short_scan_rows_follow_input_permutation(dtype):
    detectors, spacing, sdd = 64, 0.1, 10.0
    half_fan = math.atan(((detectors - 1) / 2) * spacing / sdd)
    angles = torch.linspace(0, math.pi + 2 * half_fan, 101, dtype=dtype)
    permutation = torch.randperm(len(angles), generator=torch.Generator().manual_seed(230))
    expected = parker_weights(angles, detectors, spacing, sdd)
    actual = parker_weights(angles[permutation], detectors, spacing, sdd)
    assert expected.shape == (len(angles), detectors)
    assert torch.isfinite(expected).all()
    assert expected.max() > expected.min()
    torch.testing.assert_close(actual, expected[permutation], rtol=0, atol=0)


def _parallel_projector(**options):
    angles = torch.tensor([0.0, 0.4, 0.9, 1.5, 2.2, 2.8], dtype=torch.float64)
    direction = torch.stack((-angles.sin(), angles.cos()), dim=1)
    detector_u = torch.stack((angles.cos(), angles.sin()), dim=1)
    trajectory = (direction, torch.zeros_like(direction), detector_u)
    return Projector(trajectory, (4, 4), (6,), beam="parallel", backend="torch",
                     precision="float64", partition="views", **options)


def _chunk_hints_worker(rank, rendezvous, differing_hint, control_index):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=15))
    try:
        alternatives = {
            "volume_chunk_shape": ((1, 4), (2, 2)),
            "view_chunk_size": (1, 2),
            "detector_chunk_shape": ((2,), (3,)),
        }
        hints = {name: values[control_index] for name, values in alternatives.items()}
        if differing_hint is not None:
            hints[differing_hint] = alternatives[differing_hint][rank]
            rejected = False
            try:
                _parallel_projector(distributed=True, **hints)
            except ValueError:
                rejected = True
            # Collect each construction outcome before asserting on either rank.
            outcomes = [None, None]
            dist.all_gather_object(outcomes, rejected)
            assert outcomes == [True, True], (
                f"{differing_hint} mismatch must raise ValueError at construction "
                f"on every rank; rejection outcomes={outcomes}"
            )
        else:
            reference = _parallel_projector(distributed=False)
            sinogram = torch.ones((6, 6), dtype=torch.float64)
            expected = reference.backproject(sinogram)
            operator = _parallel_projector(distributed=True, **hints)
            actual = operator.backproject(sinogram[operator.view_slice])
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    finally:
        dist.destroy_process_group()


@gloo_required
@pytest.mark.parametrize("hint", ["volume_chunk_shape", "view_chunk_size",
                                  "detector_chunk_shape"])
def test_distributed_projector_rejects_chunk_hint_mismatch_at_construction(hint, tmp_path):
    _run_processes(_chunk_hints_worker,
                   ((tmp_path / "mismatch").resolve().as_uri(), hint, 0), timeout=60)


@gloo_required
@pytest.mark.parametrize("control_index", [0, 1])
def test_distributed_projector_matching_chunk_hints_match_single_process(control_index, tmp_path):
    _run_processes(_chunk_hints_worker,
                   ((tmp_path / "matching").resolve().as_uri(), None, control_index), timeout=60)
