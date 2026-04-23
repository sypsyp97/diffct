"""Quantitative accuracy test for the cone-beam FDK pipeline.

Runs ``ConeProjectorFunction -> cone_cosine_weights -> ramp_filter_1d ->
angular_integration_weights -> cone_weighted_backproject`` on a 128^3
Shepp-Logan phantom and asserts RMSE, amplitude, and z-profile bounds
that would catch regressions in the analytical constant, the
``(sid/U)^2`` FDK weight, or the ramp-filter scale.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from diffct import (
    ConeProjectorFunction,
    circular_trajectory_3d,
    cone_weighted_backproject,
    cone_cosine_weights,
    ramp_filter_1d,
    angular_integration_weights,
)

from ._phantoms import shepp_logan_3d


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def _run_fdk(vol, geom):
    device = vol.device
    src_pos, det_center, det_u_vec, det_v_vec = circular_trajectory_3d(
        geom["num_views"], sid=geom["sid"], sdd=geom["sdd"], device=device
    )
    angles = torch.linspace(
        0.0, 2.0 * math.pi, geom["num_views"] + 1, device=device
    )[:-1]

    sino = ConeProjectorFunction.apply(
        vol, src_pos, det_center, det_u_vec, det_v_vec,
        geom["det_u"], geom["det_v"], geom["du"], geom["dv"], geom["voxel_spacing"],
    )
    weights = cone_cosine_weights(
        geom["det_u"], geom["det_v"], geom["du"], geom["dv"], geom["sdd"],
        device=device, dtype=vol.dtype,
    ).unsqueeze(0)
    sino_filt = ramp_filter_1d(
        sino * weights,
        dim=1,
        sample_spacing=geom["du"],
        pad_factor=2,
        window="hann",
    ).contiguous()
    d_beta = angular_integration_weights(
        angles, redundant_full_scan=True
    ).view(-1, 1, 1)
    sino_filt = sino_filt * d_beta

    D, H, W = vol.shape
    return cone_weighted_backproject(
        sino_filt, src_pos, det_center, det_u_vec, det_v_vec,
        D, H, W, geom["du"], geom["dv"], voxel_spacing=geom["voxel_spacing"],
    )


@pytest.mark.cuda
def test_fdk_cone_shepp_logan_rmse_is_small():
    """FDK reconstruction of a 128^3 Shepp-Logan should stay within calibrated
    RMSE and amplitude bounds (guards against amplitude overshoot from a
    dropped ``(sid/U)^2`` weight or wrong Fourier-convention constant)."""
    _skip_if_no_cuda()
    device = torch.device("cuda")

    Nx = Ny = Nz = 128
    phantom = shepp_logan_3d(Nx)
    vol = torch.tensor(phantom, device=device, dtype=torch.float32).contiguous()

    geom = dict(
        num_views=360,
        det_u=256,
        det_v=256,
        du=1.0,
        dv=1.0,
        sdd=900.0,
        sid=600.0,
        voxel_spacing=1.0,
    )

    reco_raw = _run_fdk(vol, geom)
    reco = F.relu(reco_raw)

    assert torch.isfinite(reco_raw).all()
    assert reco_raw.shape == vol.shape

    rmse_raw = torch.sqrt(torch.mean((reco_raw - vol) ** 2)).item()
    assert rmse_raw < 0.1, f"raw FDK RMSE too high: {rmse_raw:.5f}"

    mid = Nz // 2
    rmse_center = torch.sqrt(
        torch.mean((reco_raw[mid] - vol[mid]) ** 2)
    ).item()
    assert rmse_center < 0.1, f"center-slice RMSE too high: {rmse_center:.5f}"

    z_profile_err = torch.abs(
        reco_raw[:, Ny // 2, Nx // 2] - vol[:, Ny // 2, Nx // 2]
    )
    # Cell-constant forward projection gives a slightly sharper one-sample
    # transition error at ellipsoid boundaries while preserving global FDK RMSE.
    assert z_profile_err.max().item() < 0.65

    assert reco.max().item() < 1.5
