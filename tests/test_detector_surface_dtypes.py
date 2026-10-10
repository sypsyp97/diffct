"""Floating surface dtypes and caller-controlled CUDA callback outputs."""

import numpy as np
import pytest
import torch

from .test_detector_surfaces import (
    _arc_offsets, _axis_case, _case, _close, _data, _finite_difference,
    _grids, _matrix, _numpy, _projector,
)


@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.cuda
def test_finite_half_inputs_accept_world_points_beyond_half_range(beam):
    case = _axis_case(beam)
    case.trajectory = tuple(g[:1].half() for g in case.trajectory)
    case.trajectory[1][:, 0] = 50000.0  # The actual Half value is 49984.

    def surface(u, v):
        return torch.stack((u, v, torch.full_like(u, 40000.0)), dim=-1).half()

    # Quantize the independent physical grids before doing float64 frame math.
    u, v = _grids(case)
    offsets = np.stack((u.astype(np.float16).astype(np.float64),
                        v.astype(np.float16).astype(np.float64),
                        np.full_like(u, 40000.0)), axis=-1)
    assert case.trajectory[1][0, 0].item() == 49984.0
    assert case.trajectory[1][0, 0].item() + offsets[..., 2].min() == 89984.0
    assert np.isfinite(offsets).all()
    matrix = _matrix(case, offsets)
    projector = _projector(case, surface)  # CPU construction must accept this.
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the numerical checks after construction")
    image, sino = _data(case)
    projected, backprojected = projector.project(image), projector.backproject(sino)
    for output in (projected, backprojected):
        assert output.device == image.device and output.dtype == torch.float32
    _close(projected, (matrix @ _numpy(image).ravel()).reshape(case.sino_shape))
    _close(backprojected, (matrix.T @ _numpy(sino).ravel()).reshape(case.shape))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_captured_radius_with_explicit_grid_device_transfer():
    case = _case("cone")
    assert all(g.device.type == "cpu" and not g.requires_grad
               for g in case.trajectory)
    image, sino = _data(case)
    radius = torch.tensor(3.7, device=image.device, dtype=torch.float64,
                          requires_grad=True)

    def surface(u, v):
        assert u.device.type == v.device.type == "cpu"
        assert u.dtype == v.dtype == torch.float64
        u, v = u.to(device=radius.device), v.to(device=radius.device)
        return torch.stack((radius * torch.sin(u / radius) + 0.037 * u.square(),
                            v, radius * (1 - torch.cos(u / radius))), dim=-1)

    image_np, sino_np = _numpy(image).ravel(), _numpy(sino).ravel()

    def objective(value):
        return float(sino_np @ _matrix(case, _arc_offsets(case, value, 0.037))
                     @ image_np)

    expected_gradient = _finite_difference(objective, 3.7)
    assert abs(expected_gradient) > 1e-3
    matrix = _matrix(case, _arc_offsets(case, 3.7, 0.037))
    projector = _projector(case, surface)
    for operation, data, weight, expected in (
        ("project", image, sino, (matrix @ image_np).reshape(case.sino_shape)),
        ("backproject", sino, image, (matrix.T @ sino_np).reshape(case.shape)),
    ):
        output = getattr(projector, operation)(data)
        _close(output, expected)
        gradient, = torch.autograd.grad((output.double() * weight.double()).sum(),
                                        radius)
        assert gradient.device == radius.device and gradient.is_cuda
        assert torch.isfinite(gradient).all()
        _close(gradient, expected_gradient, rtol=5e-3, atol=3e-4)
