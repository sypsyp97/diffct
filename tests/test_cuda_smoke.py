"""CUDA smoke tests for every geometry.

For each geometry builds trajectory arrays via the circular-trajectory
helper, runs a forward projection, a pure-adjoint backprojection, and
the analytical weighted backprojection wrapper, and asserts that every
output tensor is finite.
"""

import pytest
import torch

from diffct import (
    ConeBackprojectorFunction,
    ConeProjectorFunction,
    FanBackprojectorFunction,
    FanProjectorFunction,
    ParallelBackprojectorFunction,
    ParallelProjectorFunction,
    cone_weighted_backproject,
    fan_weighted_backproject,
    parallel_weighted_backproject,
    circular_trajectory_2d_parallel,
    circular_trajectory_2d_fan,
    circular_trajectory_3d,
)


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


@pytest.mark.cuda
def test_parallel_cuda_smoke():
    _skip_if_no_cuda()
    device = torch.device("cuda")
    img = torch.zeros((16, 16), device=device, dtype=torch.float32)
    img[6:10, 6:10] = 1.0
    ray_dir, det_origin, det_u_vec = circular_trajectory_2d_parallel(64, device=device)

    sino = ParallelProjectorFunction.apply(
        img, ray_dir, det_origin, det_u_vec, 32, 1.0
    )
    reco_adj = ParallelBackprojectorFunction.apply(
        sino, ray_dir, det_origin, det_u_vec, 1.0, 16, 16
    )
    reco_fbp = parallel_weighted_backproject(
        sino, ray_dir, det_origin, det_u_vec, 1.0, 16, 16
    )

    assert torch.isfinite(sino).all()
    assert torch.isfinite(reco_adj).all()
    assert torch.isfinite(reco_fbp).all()


@pytest.mark.cuda
def test_autograd_smoke_parallel():
    """Autograd round trip on parallel beam."""
    _skip_if_no_cuda()
    device = torch.device("cuda")

    img = torch.randn((12, 12), device=device, dtype=torch.float32, requires_grad=True)
    ray_dir, det_origin, det_u_vec = circular_trajectory_2d_parallel(48, device=device)
    sino = ParallelProjectorFunction.apply(
        img, ray_dir, det_origin, det_u_vec, 24, 1.0
    )
    sino.mean().backward()
    assert torch.isfinite(img.grad).all()


@pytest.mark.cuda
def test_fan_cuda_smoke():
    _skip_if_no_cuda()
    device = torch.device("cuda")
    img = torch.zeros((16, 16), device=device, dtype=torch.float32)
    img[6:10, 6:10] = 1.0
    src_pos, det_center, det_u_vec = circular_trajectory_2d_fan(
        64, sid=500.0, sdd=800.0, device=device
    )

    sino = FanProjectorFunction.apply(
        img, src_pos, det_center, det_u_vec, 32, 1.0
    )
    reco_adj = FanBackprojectorFunction.apply(
        sino, src_pos, det_center, det_u_vec, 1.0, 16, 16
    )
    reco_fbp = fan_weighted_backproject(
        sino, src_pos, det_center, det_u_vec, 1.0, 16, 16
    )

    assert torch.isfinite(sino).all()
    assert torch.isfinite(reco_adj).all()
    assert torch.isfinite(reco_fbp).all()


@pytest.mark.cuda
def test_cone_cuda_smoke():
    _skip_if_no_cuda()
    device = torch.device("cuda")
    vol = torch.zeros((12, 12, 12), device=device, dtype=torch.float32).contiguous()
    vol[4:8, 4:8, 4:8] = 1.0
    src_pos, det_center, det_u_vec, det_v_vec = circular_trajectory_3d(
        48, sid=600.0, sdd=900.0, device=device
    )

    sino = ConeProjectorFunction.apply(
        vol, src_pos, det_center, det_u_vec, det_v_vec,
        20, 20, 1.0, 1.0,
    )
    reco_adj = ConeBackprojectorFunction.apply(
        sino, src_pos, det_center, det_u_vec, det_v_vec,
        12, 12, 12, 1.0, 1.0,
    )
    reco_fdk = cone_weighted_backproject(
        sino, src_pos, det_center, det_u_vec, det_v_vec,
        12, 12, 12, 1.0, 1.0,
    )

    assert torch.isfinite(sino).all()
    assert torch.isfinite(reco_adj).all()
    assert torch.isfinite(reco_fdk).all()
