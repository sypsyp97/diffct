"""Quantitative accuracy tests for the parallel-beam FBP pipeline.

Exercises the public call chain from ``examples/fbp_parallel.py``::

    ParallelProjectorFunction -> ramp_filter_1d
        -> angular_integration_weights -> parallel_weighted_backproject

Parallel beam has no cosine pre-weight and no distance weighting in
the backprojection - the FBP pipeline needs only ramp filtering,
angular weights, and the ``1/(2*pi)`` analytical constant that
``parallel_weighted_backproject`` applies internally.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from diffct.differentiable import (
    ParallelProjectorFunction,
    angular_integration_weights,
    parallel_weighted_backproject,
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


def _run_parallel_fbp(image_torch, geom):
    angles = torch.linspace(
        0.0,
        2.0 * math.pi,
        geom["num_angles"] + 1,
        device=image_torch.device,
        dtype=torch.float32,
    )[:-1]

    sino = ParallelProjectorFunction.apply(
        image_torch,
        angles,
        geom["num_detectors"],
        geom["detector_spacing"],
        geom["voxel_spacing"],
    )

    sino_filt = ramp_filter_1d(
        sino,
        dim=1,
        sample_spacing=geom["detector_spacing"],
        pad_factor=2,
        window="hann",
    ).contiguous()

    d_beta = angular_integration_weights(angles, redundant_full_scan=True).view(-1, 1)
    sino_filt = sino_filt * d_beta

    H, W = image_torch.shape
    return parallel_weighted_backproject(
        sino_filt,
        angles,
        geom["detector_spacing"],
        H,
        W,
        voxel_spacing=geom["voxel_spacing"],
    )


@pytest.mark.cuda
def test_fbp_parallel_shepp_logan_rmse_is_small():
    """Parallel FBP of a 256x256 Shepp-Logan should stay within calibrated
    RMSE and amplitude bounds (historic bug gave ~6.5x overshoot)."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    geom = dict(
        num_angles=360,
        num_detectors=512,
        detector_spacing=1.0,
        voxel_spacing=1.0,
    )
    Nx = Ny = 256
    phantom = _shepp_logan_2d(Nx, Ny)
    image_torch = torch.tensor(phantom, device=device, dtype=torch.float32)

    reco_raw = _run_parallel_fbp(image_torch, geom)
    reco = F.relu(reco_raw)

    assert torch.isfinite(reco_raw).all()
    assert reco_raw.shape == image_torch.shape

    rmse_raw = torch.sqrt(torch.mean((reco_raw - image_torch) ** 2)).item()
    assert rmse_raw < 0.1, f"raw parallel FBP RMSE too high: {rmse_raw:.5f}"

    center_row_err = torch.abs(reco_raw[Ny // 2] - image_torch[Ny // 2])
    assert center_row_err.max().item() < 0.6

    assert reco.max().item() < 1.5


@pytest.mark.cuda
def test_fbp_parallel_with_center_offset():
    """Non-zero image center offset should still produce a finite result."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    num_angles = 128
    angles = torch.linspace(
        0.0, 2.0 * math.pi, num_angles + 1, device=device, dtype=torch.float32
    )[:-1]

    H, W = 64, 64
    img = torch.zeros(H, W, device=device, dtype=torch.float32)
    img[16:48, 16:48] = 1.0

    sino = ParallelProjectorFunction.apply(
        img, angles, 96, 1.0, 1.0, 0.0, 0.5, -0.3
    )
    sino_filt = ramp_filter_1d(
        sino, dim=1, sample_spacing=1.0, pad_factor=2, window="hann"
    ).contiguous()
    d_beta = angular_integration_weights(angles, redundant_full_scan=True).view(-1, 1)
    sino_filt = sino_filt * d_beta

    reco = parallel_weighted_backproject(
        sino_filt,
        angles,
        1.0,
        H,
        W,
        center_offset_x=0.5,
        center_offset_y=-0.3,
    )
    assert reco.shape == img.shape
    assert reco.device == img.device
    assert torch.isfinite(reco).all()
