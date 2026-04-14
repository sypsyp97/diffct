"""Offset-handling tests for the cone-beam FDK pipeline.

The analytical FDK path now goes through a dedicated voxel-driven
gather kernel. These tests make sure the gather kernel's geometry
inversion honors non-zero detector and volume-center offsets, i.e.
that the reconstructed volume is still finite, correctly shaped, and
stays on the expected CUDA device.
"""

import math

import pytest
import torch

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


def _build_sinogram(
    vol,
    angles,
    geom,
    detector_offset_u=0.0,
    detector_offset_v=0.0,
    center_offset_x=0.0,
    center_offset_y=0.0,
    center_offset_z=0.0,
):
    """Forward-project ``vol`` through ``ConeProjectorFunction`` with offsets."""
    return ConeProjectorFunction.apply(
        vol,
        angles,
        geom["det_u"],
        geom["det_v"],
        geom["du"],
        geom["dv"],
        geom["sdd"],
        geom["sid"],
        geom["voxel_spacing"],
        detector_offset_u,
        detector_offset_v,
        center_offset_x,
        center_offset_y,
        center_offset_z,
    )


def _fdk_reconstruct(vol_shape, sino, angles, geom, **offsets):
    """Run the analytical FDK pipeline forwarding offsets to every stage."""
    cos_w = cone_cosine_weights(
        geom["det_u"],
        geom["det_v"],
        geom["du"],
        geom["dv"],
        geom["sdd"],
        detector_offset_u=offsets.get("detector_offset_u", 0.0),
        detector_offset_v=offsets.get("detector_offset_v", 0.0),
        device=sino.device,
        dtype=sino.dtype,
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

    D, H, W = vol_shape
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
        **offsets,
    )


def _make_cube(D, H, W, device, value=1.0, box=(10, 22)):
    """Small uniform cube used as a stable FDK fixture."""
    vol = torch.zeros(D, H, W, device=device, dtype=torch.float32).contiguous()
    lo, hi = box
    vol[lo:hi, lo:hi, lo:hi] = value
    return vol


def _default_geom_and_angles(device, num_views=64):
    geom = dict(
        num_views=num_views,
        det_u=48,
        det_v=48,
        du=1.0,
        dv=1.0,
        sdd=900.0,
        sid=600.0,
        voxel_spacing=1.0,
    )
    angles = torch.linspace(
        0.0, 2.0 * math.pi, num_views + 1, device=device, dtype=torch.float32
    )[:-1]
    return geom, angles


@pytest.mark.cuda
def test_fdk_cone_with_detector_offsets():
    """Non-zero detector shifts (half-fan style) still produce a finite volume."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    geom, angles = _default_geom_and_angles(device)
    vol = _make_cube(32, 32, 32, device)

    offsets = dict(detector_offset_u=2.5, detector_offset_v=-1.5)
    sino = _build_sinogram(vol, angles, geom, **offsets)
    reco = _fdk_reconstruct(vol.shape, sino, angles, geom, **offsets)

    assert reco.shape == vol.shape
    assert reco.device == vol.device
    assert torch.isfinite(reco).all()


@pytest.mark.cuda
def test_fdk_cone_with_center_offsets():
    """Shifting the reconstruction volume center off the isocenter still works."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    geom, angles = _default_geom_and_angles(device)
    vol = _make_cube(32, 32, 32, device)

    offsets = dict(center_offset_x=1.0, center_offset_y=-0.5, center_offset_z=0.25)
    sino = _build_sinogram(vol, angles, geom, **offsets)
    reco = _fdk_reconstruct(vol.shape, sino, angles, geom, **offsets)

    assert reco.shape == vol.shape
    assert reco.device == vol.device
    assert torch.isfinite(reco).all()


@pytest.mark.cuda
def test_fdk_cone_with_all_offsets_combined():
    """All five offsets simultaneously - stress test for the gather inverse geometry."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    geom, angles = _default_geom_and_angles(device)
    vol = _make_cube(32, 32, 32, device, value=0.5, box=(8, 24))

    offsets = dict(
        detector_offset_u=1.75,
        detector_offset_v=-0.9,
        center_offset_x=0.3,
        center_offset_y=-0.4,
        center_offset_z=0.2,
    )
    sino = _build_sinogram(vol, angles, geom, **offsets)
    reco = _fdk_reconstruct(vol.shape, sino, angles, geom, **offsets)

    assert reco.shape == vol.shape
    assert reco.device == vol.device
    assert torch.isfinite(reco).all()


@pytest.mark.cuda
def test_fdk_cone_consistent_offsets_match_amplitude():
    """Reconstructing with the *same* offsets used during projection must
    still land within the right amplitude window - this catches scale
    regressions that slip through pure finiteness checks."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    geom, angles = _default_geom_and_angles(device, num_views=180)
    vol = _make_cube(48, 48, 48, device, value=1.0, box=(16, 32))

    offsets = dict(
        detector_offset_u=1.5,
        detector_offset_v=0.75,
        center_offset_x=0.4,
        center_offset_y=-0.2,
        center_offset_z=0.1,
    )
    sino = _build_sinogram(vol, angles, geom, **offsets)
    reco = _fdk_reconstruct(vol.shape, sino, angles, geom, **offsets)

    assert reco.shape == vol.shape
    assert torch.isfinite(reco).all()

    # Peak amplitude should stay in the analytical-FDK ballpark of the
    # unit-density cube (small overshoot from ramp ringing is allowed,
    # but anything above ~1.5 would indicate a dropped scale factor).
    assert reco.max().item() < 1.5
    # And there has to be real signal inside the cube (sanity).
    cube_interior = reco[18:30, 18:30, 18:30]
    assert cube_interior.mean().item() > 0.5
