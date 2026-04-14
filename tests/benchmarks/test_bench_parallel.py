"""Parallel-beam kernel benchmarks (arbitrary-trajectory API)."""

from __future__ import annotations

import pytest
import torch

from diffct import (
    ParallelBackprojectorFunction,
    ParallelProjectorFunction,
    angular_integration_weights,
    circular_trajectory_2d_parallel,
    parallel_weighted_backproject,
    ramp_filter_1d,
)

from ._common import (
    full_scan_angles,
    make_phantom_2d,
    make_sinogram_2d,
    skip_if_no_cuda,
    sync_call,
)


PARALLEL_SIZES = [
    (128, 180, 192, "small"),
    (256, 360, 384, "medium"),
    (512, 720, 768, "large"),
]


def _trajectory(n_ang, device):
    return circular_trajectory_2d_parallel(n_ang, device=device)


@pytest.mark.benchmark(group="parallel-forward")
@pytest.mark.parametrize(
    "n_img,n_ang,n_det,label",
    PARALLEL_SIZES,
    ids=[s[3] for s in PARALLEL_SIZES],
)
def test_bench_parallel_forward(benchmark, n_img, n_ang, n_det, label):
    skip_if_no_cuda()
    device = torch.device("cuda")
    img = make_phantom_2d(n_img, device)
    ray_dir, det_origin, det_u_vec = _trajectory(n_ang, device)

    benchmark(
        sync_call,
        ParallelProjectorFunction.apply,
        img,
        ray_dir,
        det_origin,
        det_u_vec,
        n_det,
        1.0,
    )


@pytest.mark.benchmark(group="parallel-adjoint")
@pytest.mark.parametrize(
    "n_img,n_ang,n_det,label",
    PARALLEL_SIZES,
    ids=[s[3] for s in PARALLEL_SIZES],
)
def test_bench_parallel_adjoint(benchmark, n_img, n_ang, n_det, label):
    skip_if_no_cuda()
    device = torch.device("cuda")
    sino = make_sinogram_2d(n_ang, n_det, device)
    ray_dir, det_origin, det_u_vec = _trajectory(n_ang, device)

    benchmark(
        sync_call,
        ParallelBackprojectorFunction.apply,
        sino,
        ray_dir,
        det_origin,
        det_u_vec,
        1.0,
        n_img,
        n_img,
    )


@pytest.mark.benchmark(group="parallel-fbp-pipeline")
@pytest.mark.parametrize(
    "n_img,n_ang,n_det,label",
    PARALLEL_SIZES,
    ids=[s[3] for s in PARALLEL_SIZES],
)
def test_bench_parallel_fbp_pipeline(benchmark, n_img, n_ang, n_det, label):
    """Full analytical FBP: ramp + angular weight + gather."""
    skip_if_no_cuda()
    device = torch.device("cuda")
    img = make_phantom_2d(n_img, device)
    ray_dir, det_origin, det_u_vec = _trajectory(n_ang, device)
    angles = full_scan_angles(n_ang, device)
    sino = ParallelProjectorFunction.apply(
        img, ray_dir, det_origin, det_u_vec, n_det, 1.0
    )

    def run():
        sino_filt = ramp_filter_1d(
            sino, dim=1, sample_spacing=1.0, pad_factor=2, window="hann"
        ).contiguous()
        d_beta = angular_integration_weights(angles, redundant_full_scan=True).view(-1, 1)
        sino_filt = sino_filt * d_beta
        return parallel_weighted_backproject(
            sino_filt, ray_dir, det_origin, det_u_vec, 1.0, n_img, n_img
        )

    benchmark(sync_call, run)
