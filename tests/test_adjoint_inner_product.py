"""Adjoint inner-product tests for every diffct projector/backprojector pair.

For any linear operator ``A`` and its adjoint ``A^T`` we must have

    <A x, y>_S  =  <x, A^T y>_V

for arbitrary ``x`` in volume space and ``y`` in sinogram space. These
tests pick random ``x`` and ``y`` on CUDA, run both sides of the
identity through the autograd ``forward`` of the projector and the
autograd ``forward`` of the backprojector, and check that the two inner
products agree to a few parts in a thousand (the tolerance budget is
wide on purpose: float32 atomic adds and the per-segment chord-length
accumulation in the cell-constant Siddon kernels introduce small
rounding that differs between the two paths).

These tests protect against any regression that would make the
autograd ``backward`` diverge from the true adjoint - for example a
reintroduction of the historical ``distance_weight=1.0`` bug in the
cone/fan Siddon backward kernels.
"""

import math

import pytest
import torch

from diffct.differentiable import (
    ConeBackprojectorFunction,
    ConeProjectorFunction,
    FanBackprojectorFunction,
    FanProjectorFunction,
    ParallelBackprojectorFunction,
    ParallelProjectorFunction,
)


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def _assert_adjoint(lhs, rhs, rtol):
    """Assert <A x, y> == <x, A^T y> to within a relative tolerance."""
    scale = max(abs(float(lhs)), abs(float(rhs)), 1e-6)
    err = abs(float(lhs) - float(rhs)) / scale
    assert err < rtol, (
        f"adjoint inner products disagree: lhs={lhs:.6f}, rhs={rhs:.6f}, "
        f"rel_err={err:.2e} (tol {rtol:.0e})"
    )


@pytest.mark.cuda
def test_parallel_adjoint_inner_product():
    """Sanity check: parallel-beam kernels should already be exact adjoints."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    torch.manual_seed(0)

    H, W = 32, 32
    num_angles, num_det = 45, 48
    x = torch.randn(H, W, device=device, dtype=torch.float32).contiguous()
    y = torch.randn(num_angles, num_det, device=device, dtype=torch.float32).contiguous()
    angles = torch.linspace(0.0, math.pi, num_angles, device=device, dtype=torch.float32)

    Ax = ParallelProjectorFunction.apply(x, angles, num_det, 1.0)
    Aty = ParallelBackprojectorFunction.apply(y, angles, 1.0, H, W)

    lhs = torch.sum(Ax * y).item()
    rhs = torch.sum(x * Aty).item()
    _assert_adjoint(lhs, rhs, rtol=5e-3)


@pytest.mark.cuda
def test_fan_adjoint_inner_product():
    """Fan: after removing the stray distance_weight=1.0, the autograd
    backprojector must be the true adjoint of the autograd projector."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    torch.manual_seed(1)

    H, W = 32, 32
    num_angles, num_det = 45, 48
    sdd, sid = 900.0, 600.0
    x = torch.randn(H, W, device=device, dtype=torch.float32).contiguous()
    y = torch.randn(num_angles, num_det, device=device, dtype=torch.float32).contiguous()
    angles = torch.linspace(0.0, 2.0 * math.pi, num_angles, device=device, dtype=torch.float32)

    Ax = FanProjectorFunction.apply(x, angles, num_det, 1.0, sdd, sid)
    Aty = FanBackprojectorFunction.apply(y, angles, 1.0, H, W, sdd, sid)

    lhs = torch.sum(Ax * y).item()
    rhs = torch.sum(x * Aty).item()
    _assert_adjoint(lhs, rhs, rtol=5e-3)


@pytest.mark.cuda
def test_fan_sf_adjoint_inner_product():
    """SF-TR backend: voxel-driven trapezoidal forward paired with its
    voxel-driven gather adjoint. Protects the SF forward/backward pair
    and the ``backend='sf'`` dispatch in both Function classes."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    torch.manual_seed(11)

    H, W = 32, 32
    num_angles, num_det = 45, 48
    sdd, sid = 900.0, 600.0
    x = torch.randn(H, W, device=device, dtype=torch.float32).contiguous()
    y = torch.randn(num_angles, num_det, device=device, dtype=torch.float32).contiguous()
    angles = torch.linspace(0.0, 2.0 * math.pi, num_angles, device=device, dtype=torch.float32)

    Ax = FanProjectorFunction.apply(
        x, angles, num_det, 1.0, sdd, sid, 1.0, 0.0, 0.0, 0.0, "sf"
    )
    Aty = FanBackprojectorFunction.apply(
        y, angles, 1.0, H, W, sdd, sid, 1.0, 0.0, 0.0, 0.0, "sf"
    )

    lhs = torch.sum(Ax * y).item()
    rhs = torch.sum(x * Aty).item()
    _assert_adjoint(lhs, rhs, rtol=5e-3)


@pytest.mark.cuda
def test_fan_sf_autograd_backward_matches_backprojector_forward():
    """SF autograd backward must match the standalone SF backprojector
    Function on the same input. Both should compute the exact same
    ``A_sf^T y`` through the same kernel."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    torch.manual_seed(12)

    H, W = 32, 32
    num_angles, num_det = 45, 48
    sdd, sid = 900.0, 600.0

    x_zero = torch.zeros(H, W, device=device, dtype=torch.float32, requires_grad=True)
    angles = torch.linspace(0.0, 2.0 * math.pi, num_angles, device=device, dtype=torch.float32)
    y = torch.randn(num_angles, num_det, device=device, dtype=torch.float32).contiguous()

    Ax = FanProjectorFunction.apply(
        x_zero, angles, num_det, 1.0, sdd, sid, 1.0, 0.0, 0.0, 0.0, "sf"
    )
    torch.sum(Ax * y).backward()
    via_autograd = x_zero.grad.detach().clone()

    via_backprojector = FanBackprojectorFunction.apply(
        y, angles, 1.0, H, W, sdd, sid, 1.0, 0.0, 0.0, 0.0, "sf"
    ).detach()

    diff = (via_autograd - via_backprojector).abs().max().item()
    scale = via_backprojector.abs().max().item() + 1e-6
    assert diff / scale < 1e-3, (
        f"SF autograd backward and backprojector forward differ: "
        f"max abs diff={diff:.4e}, scale={scale:.4e}"
    )


@pytest.mark.cuda
def test_cone_adjoint_inner_product():
    """Cone: same adjoint identity for the 3D Siddon projector/backprojector."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    torch.manual_seed(2)

    D, H, W = 16, 16, 16
    num_views, det_u, det_v = 32, 20, 20
    sdd, sid = 900.0, 600.0
    x = torch.randn(D, H, W, device=device, dtype=torch.float32).contiguous()
    y = torch.randn(num_views, det_u, det_v, device=device, dtype=torch.float32).contiguous()
    angles = torch.linspace(0.0, 2.0 * math.pi, num_views, device=device, dtype=torch.float32)

    Ax = ConeProjectorFunction.apply(x, angles, det_u, det_v, 1.0, 1.0, sdd, sid)
    Aty = ConeBackprojectorFunction.apply(y, angles, D, H, W, 1.0, 1.0, sdd, sid)

    lhs = torch.sum(Ax * y).item()
    rhs = torch.sum(x * Aty).item()
    _assert_adjoint(lhs, rhs, rtol=5e-3)


@pytest.mark.cuda
@pytest.mark.parametrize("backend", ["sf_tr", "sf_tt"])
def test_cone_sf_adjoint_inner_product(backend):
    """3D SF backends: voxel-driven scatter forward paired with voxel-driven
    gather adjoint. Both SF-TR (rectangle axial) and SF-TT (trapezoid axial)
    must be byte-accurate adjoints of themselves."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    torch.manual_seed(30 + hash(backend) % 100)

    D, H, W = 16, 16, 16
    num_views, det_u, det_v = 24, 20, 20
    sdd, sid = 900.0, 600.0
    x = torch.randn(D, H, W, device=device, dtype=torch.float32).contiguous()
    y = torch.randn(num_views, det_u, det_v, device=device, dtype=torch.float32).contiguous()
    angles = torch.linspace(0.0, 2.0 * math.pi, num_views, device=device, dtype=torch.float32)

    Ax = ConeProjectorFunction.apply(
        x, angles, det_u, det_v, 1.0, 1.0, sdd, sid,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, backend,
    )
    Aty = ConeBackprojectorFunction.apply(
        y, angles, D, H, W, 1.0, 1.0, sdd, sid,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, backend,
    )

    lhs = torch.sum(Ax * y).item()
    rhs = torch.sum(x * Aty).item()
    _assert_adjoint(lhs, rhs, rtol=5e-3)


@pytest.mark.cuda
@pytest.mark.parametrize("backend", ["sf_tr", "sf_tt"])
def test_cone_sf_autograd_backward_matches_backprojector_forward(backend):
    """SF cone autograd backward must match the standalone SF backprojector
    Function on the same input."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    torch.manual_seed(40 + hash(backend) % 100)

    D, H, W = 16, 16, 16
    num_views, det_u, det_v = 24, 20, 20
    sdd, sid = 900.0, 600.0

    x_zero = torch.zeros(D, H, W, device=device, dtype=torch.float32, requires_grad=True)
    angles = torch.linspace(0.0, 2.0 * math.pi, num_views, device=device, dtype=torch.float32)
    y = torch.randn(num_views, det_u, det_v, device=device, dtype=torch.float32).contiguous()

    Ax = ConeProjectorFunction.apply(
        x_zero, angles, det_u, det_v, 1.0, 1.0, sdd, sid,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, backend,
    )
    torch.sum(Ax * y).backward()
    via_autograd = x_zero.grad.detach().clone()

    via_backprojector = ConeBackprojectorFunction.apply(
        y, angles, D, H, W, 1.0, 1.0, sdd, sid,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, backend,
    ).detach()

    diff = (via_autograd - via_backprojector).abs().max().item()
    scale = via_backprojector.abs().max().item() + 1e-6
    assert diff / scale < 1e-3, (
        f"SF cone ({backend}) autograd backward and backprojector forward "
        f"differ: max abs diff={diff:.4e}, scale={scale:.4e}"
    )


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
    angles = torch.linspace(0.0, 2.0 * math.pi, num_views, device=device, dtype=torch.float32)

    y = torch.randn(num_views, det_u, det_v, device=device, dtype=torch.float32).contiguous()

    # Route 1: <P(x), y> and then autograd d/dx gives P^T(y).
    Ax = ConeProjectorFunction.apply(
        x_zero, angles, det_u, det_v, 1.0, 1.0, sdd, sid
    )
    scalar = torch.sum(Ax * y)
    scalar.backward()
    adjoint_via_autograd = x_zero.grad.detach().clone()

    # Route 2: the standalone backprojector Function.
    adjoint_via_backprojector = ConeBackprojectorFunction.apply(
        y, angles, D, H, W, 1.0, 1.0, sdd, sid
    ).detach()

    assert adjoint_via_autograd.shape == adjoint_via_backprojector.shape
    diff = (adjoint_via_autograd - adjoint_via_backprojector).abs().max().item()
    scale = adjoint_via_backprojector.abs().max().item() + 1e-6
    assert diff / scale < 1e-3, (
        f"autograd backward and backprojector forward differ: "
        f"max abs diff={diff:.4e}, scale={scale:.4e}"
    )
