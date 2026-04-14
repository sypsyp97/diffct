"""Quantitative accuracy tests for the fan-beam FBP pipeline.

The public call chain exercised here mirrors ``examples/fbp_fan.py``::

    FanProjectorFunction -> fan_cosine_weights -> ramp_filter_1d
        -> angular_integration_weights -> fan_weighted_backproject

Thresholds are calibrated conservatively on a 256x256 Shepp-Logan at
a typical magnification (sdd/sid = 1.6). They protect against
amplitude regressions (historic ~6x blow-up) and against the
``distance_weight=1.0`` bug reappearing inside the analytical path.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from diffct.differentiable import (
    FanProjectorFunction,
    angular_integration_weights,
    fan_cosine_weights,
    fan_weighted_backproject,
    ramp_filter_1d,
)


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def _shepp_logan_2d(Nx, Ny):
    phantom = np.zeros((Ny, Nx), dtype=np.float32)
    ellipses = [
        (0.0,    0.0,    0.69,   0.92,    0.0,  1.0),
        (0.0,   -0.0184, 0.6624, 0.8740,  0.0, -0.8),
        (0.22,   0.0,    0.11,   0.31,  -18.0, -0.8),
        (-0.22,  0.0,    0.16,   0.41,   18.0, -0.8),
        (0.0,    0.35,   0.21,   0.25,    0.0,  0.7),
    ]
    cx = (Nx - 1) * 0.5
    cy = (Ny - 1) * 0.5
    for ix in range(Nx):
        for iy in range(Ny):
            xnorm = (ix - cx) / (Nx / 2)
            ynorm = (iy - cy) / (Ny / 2)
            val = 0.0
            for (x0, y0, a, b, angdeg, ampl) in ellipses:
                th = np.deg2rad(angdeg)
                xp = (xnorm - x0) * np.cos(th) + (ynorm - y0) * np.sin(th)
                yp = -(xnorm - x0) * np.sin(th) + (ynorm - y0) * np.cos(th)
                if xp * xp / (a * a) + yp * yp / (b * b) <= 1.0:
                    val += ampl
            phantom[iy, ix] = val
    return np.clip(phantom, 0.0, 1.0)


def _run_fan_fbp(image_torch, geom):
    """Run the analytical fan FBP pipeline and return the raw volume."""
    angles = torch.linspace(
        0.0,
        2.0 * math.pi,
        geom["num_angles"] + 1,
        device=image_torch.device,
        dtype=torch.float32,
    )[:-1]

    sino = FanProjectorFunction.apply(
        image_torch,
        angles,
        geom["num_detectors"],
        geom["detector_spacing"],
        geom["sdd"],
        geom["sid"],
        geom["voxel_spacing"],
    )

    cos_w = fan_cosine_weights(
        geom["num_detectors"],
        geom["detector_spacing"],
        geom["sdd"],
        device=image_torch.device,
        dtype=image_torch.dtype,
    ).unsqueeze(0)

    sino_filt = ramp_filter_1d(
        sino * cos_w,
        dim=1,
        sample_spacing=geom["detector_spacing"],
        pad_factor=2,
        window="hann",
    ).contiguous()

    d_beta = angular_integration_weights(angles, redundant_full_scan=True).view(-1, 1)
    sino_filt = sino_filt * d_beta

    H, W = image_torch.shape
    return fan_weighted_backproject(
        sino_filt,
        angles,
        geom["detector_spacing"],
        H,
        W,
        geom["sdd"],
        geom["sid"],
        voxel_spacing=geom["voxel_spacing"],
    )


@pytest.mark.cuda
def test_fbp_fan_shepp_logan_rmse_is_small():
    """Fan FBP of a 256x256 Shepp-Logan should stay within calibrated RMSE
    and amplitude bounds (guards against the pre-fix ~6x overshoot)."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    geom = dict(
        num_angles=360,
        num_detectors=600,
        detector_spacing=1.0,
        sdd=800.0,
        sid=500.0,
        voxel_spacing=1.0,
    )
    Nx = Ny = 256
    phantom = _shepp_logan_2d(Nx, Ny)
    image_torch = torch.tensor(phantom, device=device, dtype=torch.float32)

    reco_raw = _run_fan_fbp(image_torch, geom)
    reco = F.relu(reco_raw)

    assert torch.isfinite(reco_raw).all()
    assert reco_raw.shape == image_torch.shape

    # RMSE bound: with the new voxel-gather FBP kernel + (sid/U)^2 weight
    # + calibrated scale, a 256x256 Shepp-Logan drops well under 0.1.
    rmse_raw = torch.sqrt(torch.mean((reco_raw - image_torch) ** 2)).item()
    assert rmse_raw < 0.1, f"raw fan FBP RMSE too high: {rmse_raw:.5f}"

    # Central-row profile error: catches axial amplitude drift.
    center_row_err = torch.abs(reco_raw[Ny // 2] - image_torch[Ny // 2])
    assert center_row_err.max().item() < 0.6

    # Amplitude sanity (historic bug: ~6x).
    assert reco.max().item() < 1.5
