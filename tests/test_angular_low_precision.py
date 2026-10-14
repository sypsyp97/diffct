"""Public angular weights compute safely without discarding FP64 precision."""

import math

import pytest
import torch

from diffct import angular_integration_weights, parker_weights


def _parker(angles):
    return parker_weights(angles, 64, 0.1, 10.0)


_WEIGHTS = [
    pytest.param(angular_integration_weights, id="angular"),
    pytest.param(_parker, id="parker"),
]
_LOW_PRECISION = [
    pytest.param(torch.float16, id="fp16"),
    pytest.param(torch.bfloat16, id="bf16"),
]


def _short_scan_angles(dtype):
    # The outer detector centre is 3.15, giving a maximum fan angle atan(0.315).
    return torch.linspace(0.0, math.pi + 2.0 * math.atan(0.315), 101, dtype=dtype)


@pytest.mark.parametrize("weight", _WEIGHTS)
@pytest.mark.parametrize("dtype", _LOW_PRECISION)
def test_low_precision_short_scan_weights_match_quantized_fp32_reference(weight, dtype):
    angles = _short_scan_angles(dtype)
    actual = weight(angles)
    # Promotion cannot recover bits already lost in the input angles.
    expected = weight(angles.to(torch.float32))
    assert torch.isfinite(actual).all()
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("weight", _WEIGHTS)
@pytest.mark.parametrize("dtype", _LOW_PRECISION)
def test_low_precision_angle_gradients_match_quantized_fp32_reference(weight, dtype):
    angles = _short_scan_angles(dtype).requires_grad_()
    reference_angles = angles.detach().to(torch.float32).requires_grad_()
    actual = weight(angles)
    expected = weight(reference_angles)
    probe = torch.linspace(-0.7, 1.3, actual.numel(), dtype=torch.float32).reshape(actual.shape)
    actual_gradient, = torch.autograd.grad(actual, angles, probe)
    expected_gradient, = torch.autograd.grad(expected, reference_angles, probe)
    assert torch.isfinite(expected_gradient).all()
    assert torch.isfinite(actual_gradient).all()
    assert actual_gradient.abs().max() > 0
    assert actual_gradient.dtype == dtype
    tolerance = torch.finfo(dtype).eps
    torch.testing.assert_close(actual_gradient, expected_gradient.to(dtype),
                               rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("weight", _WEIGHTS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
def test_public_weight_dtype_preserves_fp32_fp64_and_promotes_integers(weight, dtype):
    angles = torch.tensor([0, 1, 2, 4], dtype=dtype)
    actual = weight(angles)
    expected_dtype = torch.float32 if dtype == torch.int64 else dtype
    assert actual.dtype == expected_dtype
    assert torch.isfinite(actual).all()
    if dtype == torch.int64:
        torch.testing.assert_close(actual, weight(angles.to(torch.float32)))
