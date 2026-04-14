"""``torch.autograd.gradcheck`` tests for every projector Function.

These tests build the numerical Jacobian by finite differences and
compare it to the analytical Jacobian that our CUDA backward kernel
returns. Tolerances are tuned for float32 Siddon kernels so the runtime
stays reasonable on a consumer GPU.
"""

import pytest
import torch

from diffct import (
    ParallelProjectorFunction,
    FanProjectorFunction,
    ConeProjectorFunction,
    circular_trajectory_2d_parallel,
    circular_trajectory_2d_fan,
    circular_trajectory_3d,
)


# Tolerances tuned for float32 + Siddon ray-march + atomic-add backward.
_GC_EPS = 1e-2
_GC_ATOL = 5e-2
_GC_RTOL = 5e-2
_GC_NONDET = 5e-3


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


@pytest.mark.cuda
def test_parallel_projector_gradcheck():
    _skip_if_no_cuda()
    device = torch.device("cuda")
    torch.manual_seed(0)

    H, W = 6, 6
    img = torch.randn(H, W, device=device, dtype=torch.float32).contiguous()
    img.requires_grad_(True)
    num_detectors = 10

    ray_dir, det_origin, det_u_vec = circular_trajectory_2d_parallel(
        6, device=device
    )

    def fn(x):
        return ParallelProjectorFunction.apply(
            x, ray_dir, det_origin, det_u_vec, num_detectors, 1.0, 1.0
        )

    assert torch.autograd.gradcheck(
        fn,
        (img,),
        eps=_GC_EPS,
        atol=_GC_ATOL,
        rtol=_GC_RTOL,
        nondet_tol=_GC_NONDET,
        check_undefined_grad=False,
    )


@pytest.mark.cuda
def test_fan_projector_gradcheck():
    _skip_if_no_cuda()
    device = torch.device("cuda")
    torch.manual_seed(1)

    H, W = 6, 6
    img = torch.randn(H, W, device=device, dtype=torch.float32).contiguous()
    img.requires_grad_(True)
    num_detectors = 10

    src_pos, det_center, det_u_vec = circular_trajectory_2d_fan(
        8, sid=600.0, sdd=900.0, device=device
    )

    def fn(x):
        return FanProjectorFunction.apply(
            x, src_pos, det_center, det_u_vec, num_detectors, 1.0, 1.0
        )

    assert torch.autograd.gradcheck(
        fn,
        (img,),
        eps=_GC_EPS,
        atol=_GC_ATOL,
        rtol=_GC_RTOL,
        nondet_tol=_GC_NONDET,
        check_undefined_grad=False,
    )


@pytest.mark.cuda
def test_cone_projector_gradcheck():
    _skip_if_no_cuda()
    device = torch.device("cuda")
    torch.manual_seed(2)

    D, H, W = 4, 4, 4
    vol = torch.randn(D, H, W, device=device, dtype=torch.float32).contiguous()
    vol.requires_grad_(True)
    det_u = det_v = 6

    src_pos, det_center, det_u_vec, det_v_vec = circular_trajectory_3d(
        6, sid=600.0, sdd=900.0, device=device
    )

    def fn(x):
        return ConeProjectorFunction.apply(
            x, src_pos, det_center, det_u_vec, det_v_vec,
            det_u, det_v, 1.0, 1.0, 1.0,
        )

    assert torch.autograd.gradcheck(
        fn,
        (vol,),
        eps=_GC_EPS,
        atol=_GC_ATOL,
        rtol=_GC_RTOL,
        nondet_tol=_GC_NONDET,
        check_undefined_grad=False,
    )
