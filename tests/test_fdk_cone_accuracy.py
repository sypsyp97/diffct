"""Quantitative accuracy tests for the cone-beam FDK pipeline.

The public call chain exercised here is the same one used by
``examples/fdk_cone.py``::

    ConeProjectorFunction -> cone_cosine_weights -> ramp_filter_1d
        -> angular_integration_weights -> cone_weighted_backproject

The tests build a 3D Shepp-Logan phantom, run a full FDK
reconstruction, and assert that the raw and center-slice RMSE, the
z-profile error and the reconstructed amplitude all fall inside
conservative bounds. These thresholds are intentionally loose: they
flag real regressions (for example if the FDK gather kernel drops
the ``(sid/U)^2`` weight or if the ramp filter scale changes) without
reacting to numerical jitter between runs.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from diffct.differentiable import (
    ConeProjectorFunction,
    angular_integration_weights,
    cone_cosine_weights,
    cone_weighted_backproject,
    ramp_filter_1d,
)


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def _shepp_logan_3d(shape):
    """Return a 3D Shepp-Logan phantom as a float32 numpy array."""
    zz, yy, xx = np.mgrid[: shape[0], : shape[1], : shape[2]]
    xx = (xx - (shape[2] - 1) / 2) / ((shape[2] - 1) / 2)
    yy = (yy - (shape[1] - 1) / 2) / ((shape[1] - 1) / 2)
    zz = (zz - (shape[0] - 1) / 2) / ((shape[0] - 1) / 2)
    el_params = np.array(
        [
            [0,     0,       0,     0.69,   0.92,  0.81, 0,             0, 0,  1.0],
            [0,    -0.0184,  0,     0.6624, 0.874, 0.78, 0,             0, 0, -0.8],
            [0.22,  0,       0,     0.11,   0.31,  0.22, -np.pi / 10.0, 0, 0, -0.2],
            [-0.22, 0,       0,     0.16,   0.41,  0.28,  np.pi / 10.0, 0, 0, -0.2],
            [0,     0.35,   -0.15,  0.21,   0.25,  0.41, 0,             0, 0,  0.1],
            [0,     0.10,    0.25,  0.046,  0.046, 0.05, 0,             0, 0,  0.1],
            [0,    -0.10,    0.25,  0.046,  0.046, 0.05, 0,             0, 0,  0.1],
            [-0.08,-0.605,   0,     0.046,  0.023, 0.05, 0,             0, 0,  0.1],
            [0,    -0.605,   0,     0.023,  0.023, 0.02, 0,             0, 0,  0.1],
            [0.06, -0.605,   0,     0.023,  0.046, 0.02, 0,             0, 0,  0.1],
        ],
        dtype=np.float32,
    )

    x_pos = el_params[:, 0][:, None, None, None]
    y_pos = el_params[:, 1][:, None, None, None]
    z_pos = el_params[:, 2][:, None, None, None]
    a_axis = el_params[:, 3][:, None, None, None]
    b_axis = el_params[:, 4][:, None, None, None]
    c_axis = el_params[:, 5][:, None, None, None]
    phi = el_params[:, 6][:, None, None, None]
    val = el_params[:, 9][:, None, None, None]

    xc = xx[None, ...] - x_pos
    yc = yy[None, ...] - y_pos
    zc = zz[None, ...] - z_pos
    c = np.cos(phi)
    s = np.sin(phi)
    xp = c * xc - s * yc
    yp = s * xc + c * yc
    zp = zc

    mask = (
        (xp ** 2) / (a_axis ** 2)
        + (yp ** 2) / (b_axis ** 2)
        + (zp ** 2) / (c_axis ** 2)
        <= 1.0
    )
    phantom = np.sum(mask * val, axis=0)
    return np.clip(phantom, 0.0, 1.0).astype(np.float32)


def _run_fdk(phantom_torch, geom):
    """Run the analytical FDK pipeline and return the raw (unclamped) volume."""
    angles = torch.linspace(
        0.0,
        2.0 * math.pi,
        geom["num_views"] + 1,
        device=phantom_torch.device,
        dtype=torch.float32,
    )[:-1]

    sino = ConeProjectorFunction.apply(
        phantom_torch,
        angles,
        geom["det_u"],
        geom["det_v"],
        geom["du"],
        geom["dv"],
        geom["sdd"],
        geom["sid"],
        geom["voxel_spacing"],
    )

    cos_w = cone_cosine_weights(
        geom["det_u"],
        geom["det_v"],
        geom["du"],
        geom["dv"],
        geom["sdd"],
        device=phantom_torch.device,
        dtype=phantom_torch.dtype,
    ).unsqueeze(0)

    sino_filt = ramp_filter_1d(
        sino * cos_w,
        dim=1,
        sample_spacing=geom["du"],
        pad_factor=2,
        window="hann",
    ).contiguous()

    d_beta = angular_integration_weights(angles, redundant_full_scan=True).view(-1, 1, 1)
    sino_filt = sino_filt * d_beta

    D, H, W = phantom_torch.shape
    return cone_weighted_backproject(
        sino_filt,
        angles,
        D,
        H,
        W,
        geom["du"],
        geom["dv"],
        geom["sdd"],
        geom["sid"],
        voxel_spacing=geom["voxel_spacing"],
    )


@pytest.mark.cuda
def test_fdk_cone_shepp_logan_rmse_is_small():
    """FDK reconstruction of a 128^3 Shepp-Logan should stay within calibrated
    RMSE and amplitude bounds (guards against the pre-fix ~10x overshoot)."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
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
    Nx = Ny = Nz = 128
    phantom = _shepp_logan_3d((Nz, Ny, Nx))
    phantom_torch = torch.tensor(phantom, device=device, dtype=torch.float32).contiguous()

    reco_raw = _run_fdk(phantom_torch, geom)
    reco = F.relu(reco_raw)

    assert torch.isfinite(reco_raw).all()
    assert reco_raw.shape == phantom_torch.shape

    # Raw volume RMSE: with the voxel-driven FDK gather + ``(sid/U)^2`` weight
    # + calibrated ramp filter, a 128^3 Shepp-Logan lands well under 0.1. The
    # broken pre-fix pipeline sat near 1.4.
    rmse_raw = torch.sqrt(torch.mean((reco_raw - phantom_torch) ** 2)).item()
    assert rmse_raw < 0.1, f"raw FDK RMSE too high: {rmse_raw:.5f}"

    # The central slice lives on the midplane of the circular orbit and
    # should track the phantom even more closely than the full volume.
    mid = Nz // 2
    rmse_center = torch.sqrt(
        torch.mean((reco_raw[mid] - phantom_torch[mid]) ** 2)
    ).item()
    assert rmse_center < 0.1, f"center-slice RMSE too high: {rmse_center:.5f}"

    # z-profile along the central column: catches gross amplitude drift
    # along the axial direction (a common symptom of wrong FDK constants).
    z_profile_err = torch.abs(
        reco_raw[:, Ny // 2, Nx // 2] - phantom_torch[:, Ny // 2, Nx // 2]
    )
    # Cell-constant forward projection gives a slightly sharper one-sample
    # transition error at ellipsoid boundaries while preserving global FDK RMSE.
    assert z_profile_err.max().item() < 0.65

    # Amplitude sanity: phantom is clipped to [0, 1]. Real FDK output has
    # a small negative ringing lobe near sharp edges, so we only assert the
    # clamped max stays reasonable.
    assert reco.max().item() < 1.5
