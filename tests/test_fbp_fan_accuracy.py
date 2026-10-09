"""Quantitative accuracy test for the 2D fan-beam FBP pipeline.

Runs ``FanProjectorFunction -> fan_cosine_weights -> ramp_filter_1d ->
angular_integration_weights -> fan_weighted_backproject`` on a 256x256
Shepp-Logan phantom and asserts RMSE + amplitude bounds.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from diffct import (
    FanProjectorFunction,
    circular_trajectory_2d_fan,
    fan_weighted_backproject,
    fan_cosine_weights,
    ramp_filter_1d,
    angular_integration_weights,
)

from ._phantoms import shepp_logan_2d


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def _run_fan_fbp(image, geom):
    device = image.device
    src_pos, det_center, det_u_vec = circular_trajectory_2d_fan(
        geom["num_angles"], sid=geom["sid"], sdd=geom["sdd"], device=device
    )
    angles = torch.linspace(
        0.0, 2.0 * math.pi, geom["num_angles"] + 1, device=device
    )[:-1]

    sino = FanProjectorFunction.apply(
        image, src_pos, det_center, det_u_vec,
        geom["num_detectors"], geom["detector_spacing"], geom["voxel_spacing"],
    )
    weights = fan_cosine_weights(
        geom["num_detectors"], geom["detector_spacing"], geom["sdd"],
        device=device, dtype=image.dtype,
    ).unsqueeze(0)
    sino_filt = ramp_filter_1d(
        sino * weights,
        dim=1,
        sample_spacing=geom["detector_spacing"],
        pad_factor=2,
        window="hann",
    ).contiguous()
    d_beta = angular_integration_weights(angles, redundant_full_scan=True).view(-1, 1)
    sino_filt = sino_filt * d_beta
    return fan_weighted_backproject(
        sino_filt, src_pos, det_center, det_u_vec,
        geom["detector_spacing"], image.shape[0], image.shape[1],
        voxel_spacing=geom["voxel_spacing"],
    )


@pytest.mark.cuda
def test_fbp_fan_shepp_logan_rmse_is_small():
    _skip_if_no_cuda()
    device = torch.device("cuda")

    N = 256
    phantom = shepp_logan_2d(N)
    image = torch.tensor(phantom, device=device, dtype=torch.float32)

    geom = dict(
        num_angles=360,
        num_detectors=600,
        detector_spacing=1.0,
        voxel_spacing=1.0,
        sdd=800.0,
        sid=500.0,
    )

    reco_raw = _run_fan_fbp(image, geom)
    reco = F.relu(reco_raw)

    assert torch.isfinite(reco_raw).all()
    assert reco_raw.shape == image.shape

    rmse = torch.sqrt(torch.mean((reco_raw - image) ** 2)).item()
    assert rmse < 0.1, f"fan FBP RMSE too high: {rmse:.5f}"

    assert reco_raw.min().item() > -0.2
    assert reco.max().item() < 1.5
