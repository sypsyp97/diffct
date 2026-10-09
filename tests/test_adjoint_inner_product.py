"""Adjoint inner-product tests for every projector/backprojector pair.

For any linear operator ``A`` and its adjoint ``A^T`` we must have

    <A x, y>_S  =  <x, A^T y>_V

for arbitrary ``x`` in volume space and ``y`` in sinogram space. These
tests pick random ``x`` and ``y`` on CUDA, run both sides through the
autograd ``forward`` of the projector and the autograd ``forward`` of
the backprojector, and check that the two inner products agree to a
few parts in a thousand.
"""

import pytest
import torch

from diffct import (
    ParallelProjectorFunction,
    ParallelBackprojectorFunction,
    FanProjectorFunction,
    FanBackprojectorFunction,
    ConeProjectorFunction,
    ConeBackprojectorFunction,
    circular_trajectory_2d_parallel,
    circular_trajectory_2d_fan,
    circular_trajectory_3d,
)


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def _assert_adjoint(lhs, rhs, rtol):
    scale = max(abs(float(lhs)), abs(float(rhs)), 1e-6)
    err = abs(float(lhs) - float(rhs)) / scale
    assert err < rtol, (
        f"adjoint inner products disagree: lhs={lhs:.6f}, rhs={rhs:.6f}, "
        f"rel_err={err:.2e} (tol {rtol:.0e})"
    )


@pytest.mark.cuda
def test_parallel_adjoint_inner_product():
    _skip_if_no_cuda()
    device = torch.device("cuda")
    torch.manual_seed(0)

    H, W = 32, 32
    num_views, num_det = 45, 48
    x = torch.randn(H, W, device=device, dtype=torch.float32).contiguous()
    y = torch.randn(num_views, num_det, device=device, dtype=torch.float32).contiguous()

    ray_dir, det_origin, det_u_vec = circular_trajectory_2d_parallel(
        num_views, device=device
    )

    Ax = ParallelProjectorFunction.apply(
        x, ray_dir, det_origin, det_u_vec, num_det, 1.0, 1.0
    )
    Aty = ParallelBackprojectorFunction.apply(
        y, ray_dir, det_origin, det_u_vec, 1.0, H, W, 1.0
    )

    lhs = torch.sum(Ax * y).item()
    rhs = torch.sum(x * Aty).item()
    _assert_adjoint(lhs, rhs, rtol=5e-3)


@pytest.mark.cuda
def test_fan_adjoint_inner_product():
    _skip_if_no_cuda()
    device = torch.device("cuda")
    torch.manual_seed(1)

    H, W = 32, 32
    num_views, num_det = 45, 48
    sdd, sid = 900.0, 600.0
    x = torch.randn(H, W, device=device, dtype=torch.float32).contiguous()
    y = torch.randn(num_views, num_det, device=device, dtype=torch.float32).contiguous()

    src_pos, det_center, det_u_vec = circular_trajectory_2d_fan(
        num_views, sid=sid, sdd=sdd, device=device
    )

    Ax = FanProjectorFunction.apply(
        x, src_pos, det_center, det_u_vec, num_det, 1.0, 1.0
    )
    Aty = FanBackprojectorFunction.apply(
        y, src_pos, det_center, det_u_vec, 1.0, H, W, 1.0
    )

    lhs = torch.sum(Ax * y).item()
    rhs = torch.sum(x * Aty).item()
    _assert_adjoint(lhs, rhs, rtol=5e-3)


@pytest.mark.cuda
def test_cone_adjoint_inner_product():
    _skip_if_no_cuda()
    device = torch.device("cuda")
    torch.manual_seed(2)

    D, H, W = 16, 16, 16
    num_views, det_u, det_v = 32, 20, 20
    sdd, sid = 900.0, 600.0
    x = torch.randn(D, H, W, device=device, dtype=torch.float32).contiguous()
    y = torch.randn(num_views, det_u, det_v, device=device, dtype=torch.float32).contiguous()

    src_pos, det_center, det_u_vec, det_v_vec = circular_trajectory_3d(
        num_views, sid=sid, sdd=sdd, device=device
    )

    Ax = ConeProjectorFunction.apply(
        x, src_pos, det_center, det_u_vec, det_v_vec,
        det_u, det_v, 1.0, 1.0, 1.0,
    )
    Aty = ConeBackprojectorFunction.apply(
        y, src_pos, det_center, det_u_vec, det_v_vec,
        D, H, W, 1.0, 1.0, 1.0,
    )

    lhs = torch.sum(Ax * y).item()
    rhs = torch.sum(x * Aty).item()
    _assert_adjoint(lhs, rhs, rtol=5e-3)


@pytest.mark.cuda
def test_cone_autograd_backward_matches_backprojector_forward():
    """``ConeProjectorFunction.backward`` (triggered by autograd) and
    ``ConeBackprojectorFunction.forward`` should produce the same volume for
    the same input sinogram - they both run the pure-adjoint Siddon scatter.
    If this ever diverges, one of them has regressed to a weighted path."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    torch.manual_seed(3)

    D, H, W = 16, 16, 16
    num_views, det_u, det_v = 32, 20, 20
    sdd, sid = 900.0, 600.0

    x_zero = torch.zeros(D, H, W, device=device, dtype=torch.float32, requires_grad=True)
    src_pos, det_center, det_u_vec, det_v_vec = circular_trajectory_3d(
        num_views, sid=sid, sdd=sdd, device=device
    )
    y = torch.randn(num_views, det_u, det_v, device=device, dtype=torch.float32).contiguous()

    # Route 1: <P(x), y> and autograd d/dx gives P^T(y).
    Ax = ConeProjectorFunction.apply(
        x_zero, src_pos, det_center, det_u_vec, det_v_vec,
        det_u, det_v, 1.0, 1.0, 1.0,
    )
    scalar = torch.sum(Ax * y)
    scalar.backward()
    adjoint_via_autograd = x_zero.grad.detach().clone()

    # Route 2: the standalone backprojector Function.
    adjoint_via_backprojector = ConeBackprojectorFunction.apply(
        y, src_pos, det_center, det_u_vec, det_v_vec,
        D, H, W, 1.0, 1.0, 1.0,
    ).detach()

    assert adjoint_via_autograd.shape == adjoint_via_backprojector.shape
    diff = (adjoint_via_autograd - adjoint_via_backprojector).abs().max().item()
    scale = adjoint_via_backprojector.abs().max().item() + 1e-6
    assert diff / scale < 1e-3, (
        f"autograd backward and backprojector forward differ: "
        f"max abs diff={diff:.4e}, scale={scale:.4e}"
    )
