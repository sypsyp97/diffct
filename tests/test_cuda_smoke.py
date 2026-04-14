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
    cone_weighted_backproject,
    fan_weighted_backproject,
    parallel_weighted_backproject,
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
    angles = torch.linspace(0.0, 2.0 * math.pi, 64, device=device, dtype=torch.float32)

    sino = ParallelProjectorFunction.apply(img, angles, 32, 1.0)
    reco = ParallelBackprojectorFunction.apply(sino, angles, 1.0, 16, 16)
    reco_fbp = parallel_weighted_backproject(sino, angles, 1.0, 16, 16)

    assert torch.isfinite(sino).all()
    assert torch.isfinite(reco).all()
    assert torch.isfinite(reco_fbp).all()


@pytest.mark.cuda
def test_autograd_with_offsets_smoke():
    _skip_if_no_cuda()
    device = torch.device("cuda")

    img = torch.randn((12, 12), device=device, dtype=torch.float32, requires_grad=True)
    angles = torch.linspace(0.0, 2.0 * math.pi, 48, device=device, dtype=torch.float32)
    sino = ParallelProjectorFunction.apply(img, angles, 24, 1.0, 1.0, 0.1, 0.2, -0.1)
    sino.mean().backward()
    assert torch.isfinite(img.grad).all()


@pytest.mark.cuda
def test_fan_cuda_smoke():
    _skip_if_no_cuda()
    device = torch.device("cuda")
    img = torch.zeros((16, 16), device=device, dtype=torch.float32)
    img[6:10, 6:10] = 1.0
    angles = torch.linspace(0.0, 2.0 * math.pi, 64, device=device, dtype=torch.float32)

    sino = FanProjectorFunction.apply(img, angles, 32, 1.0, 800.0, 500.0)
    reco_adj = FanBackprojectorFunction.apply(sino, angles, 1.0, 16, 16, 800.0, 500.0)
    reco_fbp = fan_weighted_backproject(sino, angles, 1.0, 16, 16, 800.0, 500.0)

    assert torch.isfinite(sino).all()
    assert torch.isfinite(reco_adj).all()
    assert torch.isfinite(reco_fbp).all()


@pytest.mark.cuda
def test_cone_cuda_smoke():
    _skip_if_no_cuda()
    device = torch.device("cuda")
    vol = torch.zeros((12, 12, 12), device=device, dtype=torch.float32).contiguous()
    vol[4:8, 4:8, 4:8] = 1.0
    angles = torch.linspace(0.0, 2.0 * math.pi, 48, device=device, dtype=torch.float32)

    sino = ConeProjectorFunction.apply(vol, angles, 20, 20, 1.0, 1.0, 900.0, 600.0)
    reco_adj = ConeBackprojectorFunction.apply(sino, angles, 12, 12, 12, 1.0, 1.0, 900.0, 600.0)
    reco_fdk = cone_weighted_backproject(sino, angles, 12, 12, 12, 1.0, 1.0, 900.0, 600.0)

    assert torch.isfinite(sino).all()
    assert torch.isfinite(reco_adj).all()
    assert torch.isfinite(reco_fdk).all()
