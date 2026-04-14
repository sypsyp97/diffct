"""Offset-handling tests for the cone-beam FDK pipeline.

Adapts main-branch offset tests to the dev-branch arbitrary-trajectory
API: offsets are applied by shifting the trajectory arrays directly.
See ``test_fbp_fan_offsets.py`` for the rationale.
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


def _apply_offsets(src_pos, det_center, det_u_vec, det_v_vec,
                   det_offset_u=0.0, det_offset_v=0.0,
                   center_offset_xyz=(0.0, 0.0, 0.0)):
    if det_offset_u != 0.0:
        det_center = det_center + det_offset_u * det_u_vec
    if det_offset_v != 0.0:
        det_center = det_center + det_offset_v * det_v_vec
    cx, cy, cz = center_offset_xyz
    if (cx, cy, cz) != (0.0, 0.0, 0.0):
        shift = torch.tensor([cx, cy, cz], device=src_pos.device, dtype=src_pos.dtype)
        src_pos = src_pos + shift
        det_center = det_center + shift
    return src_pos, det_center


def _run_fdk(vol, det_offset_u=0.0, det_offset_v=0.0,
             center_offset_xyz=(0.0, 0.0, 0.0)):
    device = vol.device
    D, H, W = vol.shape
    num_views = 120
    det_u = det_v = 96
    du = dv = 1.0
    sdd, sid = 900.0, 600.0

    src_pos, det_center, det_u_vec, det_v_vec = circular_trajectory_3d(
        num_views, sid=sid, sdd=sdd, device=device
    )
    src_pos, det_center = _apply_offsets(
        src_pos, det_center, det_u_vec, det_v_vec,
        det_offset_u=det_offset_u, det_offset_v=det_offset_v,
        center_offset_xyz=center_offset_xyz,
    )
    angles = torch.linspace(0.0, 2.0 * math.pi, num_views + 1, device=device)[:-1]

    sino = ConeProjectorFunction.apply(
        vol, src_pos, det_center, det_u_vec, det_v_vec,
        det_u, det_v, du, dv, 1.0,
    )
    weights = cone_cosine_weights(
        det_u, det_v, du, dv, sdd, device=device, dtype=vol.dtype,
    ).unsqueeze(0)
    sino_filt = ramp_filter_1d(
        sino * weights, dim=1, sample_spacing=du, pad_factor=2, window="hann"
    ).contiguous()
    d_beta = angular_integration_weights(angles, redundant_full_scan=True).view(-1, 1, 1)
    sino_filt = sino_filt * d_beta
    return cone_weighted_backproject(
        sino_filt, src_pos, det_center, det_u_vec, det_v_vec,
        D, H, W, du, dv, 1.0,
    )


def _phantom():
    device = torch.device("cuda")
    phantom = shepp_logan_3d(64)
    return torch.tensor(phantom, device=device, dtype=torch.float32).contiguous()


@pytest.mark.cuda
def test_fdk_cone_with_detector_offset():
    _skip_if_no_cuda()
    vol = _phantom()
    reco = _run_fdk(vol, det_offset_u=2.0, det_offset_v=-1.5)
    assert torch.isfinite(reco).all()
    assert reco.shape == vol.shape
    assert reco.abs().max().item() < 5.0


@pytest.mark.cuda
def test_fdk_cone_with_center_offset():
    _skip_if_no_cuda()
    vol = _phantom()
    reco = _run_fdk(vol, center_offset_xyz=(1.0, -0.8, 0.3))
    assert torch.isfinite(reco).all()
    assert F.relu(reco).max().item() < 5.0


@pytest.mark.cuda
def test_fdk_cone_with_combined_offsets():
    _skip_if_no_cuda()
    vol = _phantom()
    reco = _run_fdk(
        vol,
        det_offset_u=1.5, det_offset_v=0.8,
        center_offset_xyz=(-0.7, 0.4, -0.2),
    )
    assert torch.isfinite(reco).all()
    assert reco.abs().max().item() < 5.0
