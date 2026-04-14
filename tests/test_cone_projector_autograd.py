"""Autograd tests for ``ConeProjectorFunction``.

These tests intentionally exercise only the differentiable forward
path. They protect the autograd-facing kernel from being broken by
the analytical FDK changes, which live on a separate kernel branch.
"""

import math

import pytest
import torch

from diffct.differentiable import ConeProjectorFunction


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


@pytest.mark.cuda
def test_cone_projector_forward_and_backward_grad():
    """Forward-project a random volume, back-propagate and check that the
    gradient has the right shape, is finite, and is not identically zero."""
    _skip_if_no_cuda()
    device = torch.device("cuda")

    D, H, W = 24, 24, 24
    vol = torch.randn(D, H, W, device=device, dtype=torch.float32).contiguous()
    vol.requires_grad_(True)

    angles = torch.linspace(0.0, 2.0 * math.pi, 48, device=device, dtype=torch.float32)
    det_u, det_v = 32, 32
    sino = ConeProjectorFunction.apply(
        vol,
        angles,
        det_u,
        det_v,
        1.0,
        1.0,
        900.0,
        600.0,
        1.0,
    )

    assert sino.shape == (angles.numel(), det_u, det_v)
    assert torch.isfinite(sino).all()

    sino.mean().backward()

    assert vol.grad is not None
    assert vol.grad.shape == vol.shape
    assert torch.isfinite(vol.grad).all()
    # Random input, random angles: at least some voxels must receive a
    # non-zero contribution from the adjoint.
    assert vol.grad.abs().sum().item() > 0.0


@pytest.mark.cuda
def test_cone_projector_gradient_with_offsets():
    """The autograd path must still produce finite gradients when every
    detector and center offset is set to a non-trivial value."""
    _skip_if_no_cuda()
    device = torch.device("cuda")

    D, H, W = 20, 20, 20
    vol = torch.randn(D, H, W, device=device, dtype=torch.float32).contiguous()
    vol.requires_grad_(True)

    angles = torch.linspace(0.0, 2.0 * math.pi, 32, device=device, dtype=torch.float32)
    sino = ConeProjectorFunction.apply(
        vol,
        angles,
        24,
        24,
        1.0,
        1.0,
        900.0,
        600.0,
        1.0,
        0.3,    # detector_offset_u
        -0.2,   # detector_offset_v
        0.1,    # center_offset_x
        -0.15,  # center_offset_y
        0.2,    # center_offset_z
    )
    assert torch.isfinite(sino).all()

    (sino ** 2).mean().backward()
    assert vol.grad is not None
    assert torch.isfinite(vol.grad).all()
