"""Autograd tests for ``ConeProjectorFunction``.

These tests exercise the differentiable forward path and verify the
gradient has the right shape, is finite, and is not identically zero.
They protect the autograd-facing kernel from accidental regressions
(e.g. a detach inside the Function).
"""

import pytest
import torch

from diffct import ConeProjectorFunction, circular_trajectory_3d, spiral_trajectory_3d


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

    src_pos, det_center, det_u_vec, det_v_vec = circular_trajectory_3d(
        48, sid=600.0, sdd=900.0, device=device
    )
    det_u, det_v = 32, 32
    sino = ConeProjectorFunction.apply(
        vol, src_pos, det_center, det_u_vec, det_v_vec,
        det_u, det_v, 1.0, 1.0, 1.0,
    )

    assert sino.shape == (src_pos.shape[0], det_u, det_v)
    assert torch.isfinite(sino).all()

    sino.mean().backward()

    assert vol.grad is not None
    assert vol.grad.shape == vol.shape
    assert torch.isfinite(vol.grad).all()
    # Random input, random angles: at least some voxels must receive a
    # non-zero contribution from the adjoint.
    assert vol.grad.abs().sum().item() > 0.0


@pytest.mark.cuda
def test_cone_projector_gradient_on_spiral_trajectory():
    """The autograd path must still produce finite gradients on a non
    circular trajectory (spiral orbit)."""
    _skip_if_no_cuda()
    device = torch.device("cuda")

    D, H, W = 20, 20, 20
    vol = torch.randn(D, H, W, device=device, dtype=torch.float32).contiguous()
    vol.requires_grad_(True)

    src_pos, det_center, det_u_vec, det_v_vec = spiral_trajectory_3d(
        32, sid=600.0, sdd=900.0, z_range=10.0, n_turns=1.0, device=device,
    )
    sino = ConeProjectorFunction.apply(
        vol, src_pos, det_center, det_u_vec, det_v_vec,
        24, 24, 1.0, 1.0, 1.0,
    )
    assert torch.isfinite(sino).all()

    (sino ** 2).mean().backward()
    assert vol.grad is not None
    assert torch.isfinite(vol.grad).all()
    assert vol.grad.abs().sum().item() > 0.0
