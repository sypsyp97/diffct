"""Parallel-beam kernel benchmarks.

Covers:
- ``ParallelProjectorFunction`` (forward Siddon ray-march)
- ``ParallelBackprojectorFunction`` (pure adjoint P^T Siddon scatter)
- ``parallel_weighted_backproject`` (full analytical FBP: ramp +
  angular weight + voxel-gather + 1/(2*pi) scale).

Every benchmark runs on CUDA and auto-skips when CUDA is unavailable.
Image / detector sizes sweep small / medium / large so the suite
captures the cost scaling with problem size.
"""

from __future__ import annotations

import pytest
import torch

from diffct.differentiable import (
    ParallelBackprojectorFunction,
    ParallelProjectorFunction,
    angular_integration_weights,
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
    # (n_img, n_ang, n_det, label)
    (128, 180, 192, "small"),
    (256, 360, 384, "medium"),
    (512, 720, 768, "large"),
]


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
    angles = full_scan_angles(n_ang, device)

    benchmark(
        sync_call,
        ParallelProjectorFunction.apply,
        img,
        angles,
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
    angles = full_scan_angles(n_ang, device)

    benchmark(
        sync_call,
        ParallelBackprojectorFunction.apply,
        sino,
        angles,
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
    """Full analytical FBP: ramp filter + angular weights + gather.

    What a user actually pays for when reconstructing, as opposed to
    just one of the four stages in isolation.
    """
    skip_if_no_cuda()
    device = torch.device("cuda")
    img = make_phantom_2d(n_img, device)
    angles = full_scan_angles(n_ang, device)
    sino = ParallelProjectorFunction.apply(img, angles, n_det, 1.0)

    def run():
        sino_filt = ramp_filter_1d(
            sino, dim=1, sample_spacing=1.0, pad_factor=2, window="hann"
        ).contiguous()
        d_beta = angular_integration_weights(angles, redundant_full_scan=True).view(-1, 1)
        sino_filt = sino_filt * d_beta
        return parallel_weighted_backproject(sino_filt, angles, 1.0, n_img, n_img)

    benchmark(sync_call, run)
