"""Fan-beam kernel benchmarks (arbitrary-trajectory API)."""

from __future__ import annotations

import pytest
import torch

from diffct import (
    FanBackprojectorFunction,
    FanProjectorFunction,
    angular_integration_weights,
    circular_trajectory_2d_fan,
    fan_cosine_weights,
    fan_weighted_backproject,
    ramp_filter_1d,
)

from ._common import (
    full_scan_angles,
    make_phantom_2d,
    make_sinogram_2d,
    skip_if_no_cuda,
    sync_call,
)


SDD = 800.0
SID = 500.0

FAN_SIZES = [
    (128, 180, 256, "small"),
    (256, 360, 512, "medium"),
    (512, 720, 1024, "large"),
]


def _trajectory(n_ang, device):
    return circular_trajectory_2d_fan(n_ang, sid=SID, sdd=SDD, device=device)


@pytest.mark.benchmark(group="fan-forward")
@pytest.mark.parametrize(
    "n_img,n_ang,n_det,label",
    FAN_SIZES,
    ids=[s[3] for s in FAN_SIZES],
)
def test_bench_fan_forward(benchmark, n_img, n_ang, n_det, label):
    skip_if_no_cuda()
    device = torch.device("cuda")
    img = make_phantom_2d(n_img, device)
    src_pos, det_center, det_u_vec = _trajectory(n_ang, device)

    benchmark(
        sync_call,
        FanProjectorFunction.apply,
        img,
        src_pos,
        det_center,
        det_u_vec,
        n_det,
        1.0,
    )


@pytest.mark.benchmark(group="fan-adjoint")
@pytest.mark.parametrize(
    "n_img,n_ang,n_det,label",
    FAN_SIZES,
    ids=[s[3] for s in FAN_SIZES],
)
def test_bench_fan_adjoint(benchmark, n_img, n_ang, n_det, label):
    skip_if_no_cuda()
    device = torch.device("cuda")
    sino = make_sinogram_2d(n_ang, n_det, device)
    src_pos, det_center, det_u_vec = _trajectory(n_ang, device)

    benchmark(
        sync_call,
        FanBackprojectorFunction.apply,
        sino,
        src_pos,
        det_center,
        det_u_vec,
        1.0,
        n_img,
        n_img,
    )


@pytest.mark.benchmark(group="fan-fbp-pipeline")
@pytest.mark.parametrize(
    "n_img,n_ang,n_det,label",
    FAN_SIZES,
    ids=[s[3] for s in FAN_SIZES],
)
def test_bench_fan_fbp_pipeline(benchmark, n_img, n_ang, n_det, label):
    """Full analytical FBP: cosine pre-weight + ramp + angular weight +
    voxel-gather + sdd/(2*pi*sid) scale."""
    skip_if_no_cuda()
    device = torch.device("cuda")
    img = make_phantom_2d(n_img, device)
    src_pos, det_center, det_u_vec = _trajectory(n_ang, device)
    angles = full_scan_angles(n_ang, device)
    sino = FanProjectorFunction.apply(
        img, src_pos, det_center, det_u_vec, n_det, 1.0
    )

    def run():
        cos_w = fan_cosine_weights(
            n_det, 1.0, SDD, device=device, dtype=torch.float32
        ).unsqueeze(0)
        sino_filt = ramp_filter_1d(
            sino * cos_w, dim=1, sample_spacing=1.0, pad_factor=2, window="hann"
        ).contiguous()
        d_beta = angular_integration_weights(angles, redundant_full_scan=True).view(-1, 1)
        sino_filt = sino_filt * d_beta
        return fan_weighted_backproject(
            sino_filt, src_pos, det_center, det_u_vec, 1.0, n_img, n_img
        )

    benchmark(sync_call, run)
