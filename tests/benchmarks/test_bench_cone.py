"""Cone-beam kernel benchmarks.

Covers:
- ``ConeProjectorFunction`` (forward 3D Siddon ray-march)
- ``ConeBackprojectorFunction`` (pure adjoint P^T Siddon scatter)
- ``cone_weighted_backproject`` (full analytical FDK: cosine pre-weight +
  ramp + angular weight + voxel-gather + ``sdd/(2*pi*sid)`` scale).

Volume sizes are deliberately smaller than the 2D cases because 3D
reconstruction scales as ``O(N^3)``. A 128^3 cone FDK at 360 views
already takes hundreds of milliseconds.
"""

from __future__ import annotations

import pytest
import torch

from diffct.differentiable import (
    ConeBackprojectorFunction,
    ConeProjectorFunction,
    angular_integration_weights,
    cone_cosine_weights,
    cone_weighted_backproject,
    ramp_filter_1d,
)

from ._common import (
    full_scan_angles,
    make_phantom_3d,
    make_sinogram_3d,
    skip_if_no_cuda,
    sync_call,
)


SDD = 900.0
SID = 600.0

CONE_SIZES = [
    # (n_vol, n_views, det_u, det_v, label)
    (64,  180, 128, 128, "small"),
    (96,  270, 192, 192, "medium"),
    (128, 360, 256, 256, "large"),
]


@pytest.mark.benchmark(group="cone-forward")
@pytest.mark.parametrize(
    "n_vol,n_views,det_u,det_v,label",
    CONE_SIZES,
    ids=[s[4] for s in CONE_SIZES],
)
def test_bench_cone_forward(benchmark, n_vol, n_views, det_u, det_v, label):
    skip_if_no_cuda()
    device = torch.device("cuda")
    vol = make_phantom_3d(n_vol, device)
    angles = full_scan_angles(n_views, device)

    benchmark(
        sync_call,
        ConeProjectorFunction.apply,
        vol,
        angles,
        det_u,
        det_v,
        1.0,
        1.0,
        SDD,
        SID,
    )


@pytest.mark.benchmark(group="cone-adjoint")
@pytest.mark.parametrize(
    "n_vol,n_views,det_u,det_v,label",
    CONE_SIZES,
    ids=[s[4] for s in CONE_SIZES],
)
def test_bench_cone_adjoint(benchmark, n_vol, n_views, det_u, det_v, label):
    skip_if_no_cuda()
    device = torch.device("cuda")
    sino = make_sinogram_3d(n_views, det_u, det_v, device)
    angles = full_scan_angles(n_views, device)

    benchmark(
        sync_call,
        ConeBackprojectorFunction.apply,
        sino,
        angles,
        n_vol,
        n_vol,
        n_vol,
        1.0,
        1.0,
        SDD,
        SID,
    )


@pytest.mark.benchmark(group="cone-fdk-pipeline")
@pytest.mark.parametrize(
    "n_vol,n_views,det_u,det_v,label",
    CONE_SIZES,
    ids=[s[4] for s in CONE_SIZES],
)
def test_bench_cone_fdk_pipeline(benchmark, n_vol, n_views, det_u, det_v, label):
    """Full analytical cone FDK: cosine pre-weight + ramp + angular
    weights + voxel-gather. What a user actually pays for."""
    skip_if_no_cuda()
    device = torch.device("cuda")
    vol = make_phantom_3d(n_vol, device)
    angles = full_scan_angles(n_views, device)
    sino = ConeProjectorFunction.apply(
        vol, angles, det_u, det_v, 1.0, 1.0, SDD, SID
    )

    def run():
        cos_w = cone_cosine_weights(
            det_u, det_v, 1.0, 1.0, SDD, device=device, dtype=torch.float32
        ).unsqueeze(0)
        sino_filt = ramp_filter_1d(
            sino * cos_w,
            dim=1,
            sample_spacing=1.0,
            pad_factor=2,
            window="hann",
        ).contiguous()
        d_beta = angular_integration_weights(
            angles, redundant_full_scan=True
        ).view(-1, 1, 1)
        sino_filt = sino_filt * d_beta
        return cone_weighted_backproject(
            sino_filt,
            angles,
            n_vol,
            n_vol,
            n_vol,
            1.0,
            1.0,
            SDD,
            SID,
        )

    benchmark(sync_call, run)
