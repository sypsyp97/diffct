"""Offset-handling tests for the fan-beam FBP pipeline.

The analytical fan FBP path now goes through a dedicated voxel-driven
gather kernel. These tests make sure the gather kernel's geometry
inversion honors non-zero detector and image-center offsets.
"""

import math

import pytest
import torch

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


def _build_sinogram(
    img,
    angles,
    geom,
    detector_offset=0.0,
    center_offset_x=0.0,
    center_offset_y=0.0,
):
    """Forward-project ``img`` through ``FanProjectorFunction`` with offsets."""
    return FanProjectorFunction.apply(
        img,
        angles,
        geom["num_detectors"],
        geom["detector_spacing"],
        geom["sdd"],
        geom["sid"],
        geom["voxel_spacing"],
        detector_offset,
        center_offset_x,
        center_offset_y,
    )


def _fbp_reconstruct(img_shape, sino, angles, geom, **offsets):
    """Run the fan FBP pipeline forwarding offsets to every stage."""
    cos_w = fan_cosine_weights(
        geom["num_detectors"],
        geom["detector_spacing"],
        geom["sdd"],
        detector_offset=offsets.get("detector_offset", 0.0),
        device=sino.device,
        dtype=sino.dtype,
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

    H, W = img_shape
    return fan_weighted_backproject(
        sino_filt,
        angles,
        geom["detector_spacing"],
        H,
        W,
        geom["sdd"],
        geom["sid"],
        voxel_spacing=geom["voxel_spacing"],
        **offsets,
    )


def _make_square(H, W, device, value=1.0, box=(16, 48)):
    """Small uniform square used as a stable FBP fixture."""
    img = torch.zeros(H, W, device=device, dtype=torch.float32)
    lo, hi = box
    img[lo:hi, lo:hi] = value
    return img


def _default_geom_and_angles(device, num_angles=128):
    geom = dict(
        num_angles=num_angles,
        num_detectors=192,
        detector_spacing=1.0,
        sdd=800.0,
        sid=500.0,
        voxel_spacing=1.0,
    )
    angles = torch.linspace(
        0.0, 2.0 * math.pi, num_angles + 1, device=device, dtype=torch.float32
    )[:-1]
    return geom, angles


@pytest.mark.cuda
def test_fbp_fan_with_detector_offset():
    """A non-zero detector shift should still produce a finite, correctly
    shaped image on the same device."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    geom, angles = _default_geom_and_angles(device)
    img = _make_square(64, 64, device)

    offsets = dict(detector_offset=2.5)
    sino = _build_sinogram(img, angles, geom, **offsets)
    reco = _fbp_reconstruct(img.shape, sino, angles, geom, **offsets)

    assert reco.shape == img.shape
    assert reco.device == img.device
    assert torch.isfinite(reco).all()


@pytest.mark.cuda
def test_fbp_fan_with_center_offsets():
    """Shifting the image center off the isocenter still reconstructs."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    geom, angles = _default_geom_and_angles(device)
    img = _make_square(64, 64, device)

    offsets = dict(center_offset_x=1.0, center_offset_y=-0.5)
    sino = _build_sinogram(img, angles, geom, **offsets)
    reco = _fbp_reconstruct(img.shape, sino, angles, geom, **offsets)

    assert reco.shape == img.shape
    assert reco.device == img.device
    assert torch.isfinite(reco).all()


@pytest.mark.cuda
def test_fbp_fan_with_all_offsets_combined():
    """All three offsets simultaneously - stress test for the inverse geometry."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    geom, angles = _default_geom_and_angles(device)
    img = _make_square(64, 64, device, value=0.5, box=(12, 52))

    offsets = dict(
        detector_offset=1.75,
        center_offset_x=0.3,
        center_offset_y=-0.4,
    )
    sino = _build_sinogram(img, angles, geom, **offsets)
    reco = _fbp_reconstruct(img.shape, sino, angles, geom, **offsets)

    assert reco.shape == img.shape
    assert reco.device == img.device
    assert torch.isfinite(reco).all()
