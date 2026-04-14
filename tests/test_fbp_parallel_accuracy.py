"""Quantitative accuracy test for the parallel-beam FBP pipeline.

Runs ``ParallelProjectorFunction -> ramp_filter_1d ->
angular_integration_weights -> parallel_weighted_backproject`` on a
256x256 Shepp-Logan phantom and asserts RMSE + amplitude bounds that
would catch a regression in the analytical constant or ramp scale.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from diffct import (
    ParallelProjectorFunction,
    circular_trajectory_2d_parallel,
    parallel_weighted_backproject,
    ramp_filter_1d,
    angular_integration_weights,
)

from ._phantoms import shepp_logan_2d


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def _run_parallel_fbp(image, num_detectors, num_angles):
    device = image.device
    ray_dir, det_origin, det_u_vec = circular_trajectory_2d_parallel(
        num_angles, device=device
    )
    angles = torch.linspace(0.0, math.pi, num_angles + 1, device=device)[:-1]

    sino = ParallelProjectorFunction.apply(
        image, ray_dir, det_origin, det_u_vec, num_detectors, 1.0, 1.0
    )
    sino_filt = ramp_filter_1d(
        sino, dim=1, sample_spacing=1.0, pad_factor=2, window="hann"
    ).contiguous()
    d_beta = angular_integration_weights(angles, redundant_full_scan=False).view(-1, 1)
    sino_filt = sino_filt * d_beta
    reco = parallel_weighted_backproject(
        sino_filt, ray_dir, det_origin, det_u_vec, 1.0, image.shape[0], image.shape[1], 1.0
    )
    return reco


@pytest.mark.cuda
def test_fbp_parallel_shepp_logan_rmse_is_small():
    _skip_if_no_cuda()
    device = torch.device("cuda")

    N = 256
    phantom = shepp_logan_2d(N)
    image = torch.tensor(phantom, device=device, dtype=torch.float32)

    reco_raw = _run_parallel_fbp(image, num_detectors=512, num_angles=360)
    reco = F.relu(reco_raw)

    assert torch.isfinite(reco_raw).all()
    assert reco_raw.shape == image.shape

    rmse = torch.sqrt(torch.mean((reco_raw - image) ** 2)).item()
    assert rmse < 0.1, f"parallel FBP RMSE too high: {rmse:.5f}"

    # Amplitude sanity
    assert reco_raw.min().item() > -0.2
    assert reco.max().item() < 1.5
