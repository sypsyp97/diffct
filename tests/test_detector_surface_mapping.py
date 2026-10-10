"""Joint cone u/v/n mapping, including surface-only captured v gradients."""

import numpy as np
import pytest
import torch

from .test_detector_surfaces import (
    _case, _close, _data, _finite_difference, _grids, _matrix, _numpy, _projector,
)


def _mapped_offsets(case, coefficient):
    """Independent NumPy equations, never calling the Torch callback."""
    u, v = _grids(case)
    b = np.asarray(coefficient, dtype=np.float64)
    if b.ndim:
        b = b.reshape(-1, 1, 1)
    local_u = 0.96 * u + 0.08 * v + 0.035 * u * v
    local_v = 0.89 * v + b * u + 0.025 * u * v
    local_n = 0.10 + 0.06 * u ** 2 + 0.09 * v ** 2 + 0.025 * u * v
    return np.stack(np.broadcast_arrays(local_u, local_v, local_n), axis=-1)


@pytest.mark.parametrize("per_view", [False, True], ids=["shared", "per-view"])
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_joint_cone_surface_mapping_and_captured_v_gradient(per_view):
    case = _case("cone")  # Tilted frame, asymmetric volume, physical pitches.
    values = np.array([0.11, -0.19, 0.23]) if per_view else np.array(0.16)
    coefficient = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    assert not any(g.requires_grad for g in case.trajectory)

    def surface(u, v):
        u, v = u.double(), v.double()
        b = coefficient.to(device=u.device)
        if b.ndim:
            b = b.reshape(-1, 1, 1)
        local_u = 0.96 * u + 0.08 * v + 0.035 * u * v
        local_v = 0.89 * v + b * u + 0.025 * u * v
        local_n = 0.10 + 0.06 * u.square() + 0.09 * v.square() + 0.025 * u * v
        return torch.stack(torch.broadcast_tensors(local_u, local_v, local_n),
                           dim=-1)

    image, sino = _data(case)
    assert not image.requires_grad and not sino.requires_grad
    image_np, sino_np = _numpy(image).ravel(), _numpy(sino).ravel()

    def objective(perturbed):
        return float(sino_np @ _matrix(case, _mapped_offsets(case, perturbed))
                     @ image_np)

    # Calibrate the oracle before checking production: both step sizes must
    # agree, and every captured v parameter must have a measurable derivative.
    directions = np.eye(case.views) if per_view else [1.0]
    expected_gradient = np.array([
        _finite_difference(objective, values, direction)
        for direction in directions
    ]).reshape(values.shape)
    assert np.all(np.abs(expected_gradient) > 1e-3)

    matrix = _matrix(case, _mapped_offsets(case, values))
    expected_projection = (matrix @ image_np).reshape(case.sino_shape)
    expected_backprojection = (matrix.T @ sino_np).reshape(case.shape)
    projector = _projector(case, surface)
    projected = projector.project(image)
    backprojected = projector.backproject(sino)

    for name, actual, expected, weight in (
        ("project", projected, expected_projection, sino),
        ("backproject", backprojected, expected_backprojection, image),
    ):
        assert actual.device == image.device and actual.dtype == torch.float32
        _close(actual, expected)
        assert actual.requires_grad, f"{name} lost surface-only autograd"
        gradient, = torch.autograd.grad((actual.double() * weight.double()).sum(),
                                        coefficient, allow_unused=True)
        assert gradient is not None, f"{name} discarded the returned local v"
        assert torch.isfinite(gradient).all()
        _close(gradient, expected_gradient, rtol=5e-3, atol=3e-4)
