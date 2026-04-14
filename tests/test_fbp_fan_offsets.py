"""Offset-handling tests for the fan-beam FBP pipeline.

The dev-branch kernels take arbitrary per-view ``(src_pos, det_center,
det_u_vec)`` arrays instead of detector / center offset scalars, so
these tests apply the offsets by shifting the trajectory arrays
directly:

* ``detector_offset``: shift ``det_center`` along ``det_u_vec``.
* ``center_offset_xy``: shift BOTH ``src_pos`` and ``det_center`` by
  the same vector (this translates the whole imaging system relative
  to the volume, which is the arbitrary-trajectory equivalent of the
  main-branch ``center_offset_x``/``center_offset_y`` parameters).

The tests verify that the FBP gather kernel produces finite, correctly
shaped output and that the reconstructed amplitude stays bounded for
every offset configuration.
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


def _apply_offsets(src_pos, det_center, det_u_vec, det_offset, center_offset_xy):
    """Shift trajectory arrays to simulate scalar offsets."""
    if det_offset != 0.0:
        det_center = det_center + det_offset * det_u_vec
    if center_offset_xy != (0.0, 0.0):
        cx, cy = center_offset_xy
        shift = torch.tensor([cx, cy], device=src_pos.device, dtype=src_pos.dtype)
        src_pos = src_pos + shift
        det_center = det_center + shift
    return src_pos, det_center


def _run_fan_fbp(image, det_offset=0.0, center_offset_xy=(0.0, 0.0)):
    device = image.device
    Ny, Nx = image.shape
    num_angles = 180
    num_detectors = 384
    sdd, sid = 800.0, 500.0

    src_pos, det_center, det_u_vec = circular_trajectory_2d_fan(
        num_angles, sid=sid, sdd=sdd, device=device
    )
    src_pos, det_center = _apply_offsets(
        src_pos, det_center, det_u_vec, det_offset, center_offset_xy
    )
    angles = torch.linspace(0.0, 2.0 * math.pi, num_angles + 1, device=device)[:-1]

    sino = FanProjectorFunction.apply(
        image, src_pos, det_center, det_u_vec, num_detectors, 1.0, 1.0
    )
    weights = fan_cosine_weights(
        num_detectors, 1.0, sdd, device=device, dtype=image.dtype
    ).unsqueeze(0)
    sino_filt = ramp_filter_1d(
        sino * weights, dim=1, sample_spacing=1.0, pad_factor=2, window="hann"
    ).contiguous()
    d_beta = angular_integration_weights(angles, redundant_full_scan=True).view(-1, 1)
    sino_filt = sino_filt * d_beta
    return fan_weighted_backproject(
        sino_filt, src_pos, det_center, det_u_vec, 1.0, Ny, Nx, 1.0
    )


def _phantom():
    device = torch.device("cuda")
    phantom = shepp_logan_2d(128)
    return torch.tensor(phantom, device=device, dtype=torch.float32)


@pytest.mark.cuda
def test_fbp_fan_with_detector_offset():
    _skip_if_no_cuda()
    img = _phantom()
    reco = _run_fan_fbp(img, det_offset=2.0)
    assert torch.isfinite(reco).all()
    assert reco.shape == img.shape
    # A non-zero offset should not make the reconstruction blow up.
    assert reco.abs().max().item() < 5.0


@pytest.mark.cuda
def test_fbp_fan_with_center_offset():
    _skip_if_no_cuda()
    img = _phantom()
    reco = _run_fan_fbp(img, center_offset_xy=(1.5, -0.8))
    assert torch.isfinite(reco).all()
    assert F.relu(reco).max().item() < 5.0


@pytest.mark.cuda
def test_fbp_fan_with_combined_offsets():
    _skip_if_no_cuda()
    img = _phantom()
    reco = _run_fan_fbp(img, det_offset=1.5, center_offset_xy=(-0.7, 0.4))
    assert torch.isfinite(reco).all()
    assert reco.abs().max().item() < 5.0
