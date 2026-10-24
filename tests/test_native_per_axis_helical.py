"""N3 amendment: helical entry points accept physical per-axis voxel pitches.

The oracle composes single-slice FDK calls through the portable float64 path.
The public helical entry points retain their existing signatures.
"""

import math

import pytest
import torch

import diffct
from tests._native_per_axis_spacing import (
    ANALYTICAL_ATOL, ANALYTICAL_RTOL, NATIVE_REQUIRED,
    close, cuda_device, data, native_device, native_only,
)


@NATIVE_REQUIRED
@pytest.mark.parametrize("spacing", [.75, (.7, 1., 1.9), [.7, 1., 1.9]],
                         ids=["scalar-oracle-control", "tuple", "list"])
@pytest.mark.parametrize("mode", ["fdk-default", "fdk-custom", "weighted"])
def test_n3_helical_values_and_existing_autograd_match_portable_float64(native_device, spacing, mode):
    shape, detector, pitch = (3, 3, 4), (9, 9), (2., 3.)
    angles = torch.linspace(-3 * math.pi, 3 * math.pi, 17, dtype=torch.float64)
    zero, height = torch.zeros_like(angles), .6 * angles
    source = torch.stack((-20 * angles.sin(), 20 * angles.cos(), height), -1)
    center = torch.stack((20 * angles.sin(), -20 * angles.cos(), height), -1)
    u = torch.stack((angles.cos(), angles.sin(), zero), -1)
    v = torch.stack((zero, zero, torch.ones_like(angles)), -1)
    trajectory = tuple(value.float().double() for value in (source, center, u, v))
    isocenter = torch.zeros(3, dtype=torch.float64)
    axis_spacing = tuple(spacing) if isinstance(spacing, (tuple, list)) else (spacing,) * 3
    slice_z = (torch.arange(shape[0], dtype=torch.float64) + .5 - shape[0] / 2) * axis_spacing[0]

    rays = data((len(angles), *detector), phase=.31).double().requires_grad_()
    overrides = {}
    if mode != "fdk-default":
        overrides["view_weights"] = (.1 + .02 * data((shape[0], len(angles)))).double().requires_grad_()
    if mode == "fdk-custom":
        overrides["projection_weights"] = (.8 + .05 * data(rays.shape)).double().requires_grad_()
        overrides["filter_response"] = (.3 + .03 * data((detector[0] + 1,))).double().requires_grad_()

    if "view_weights" in overrides:
        weights = overrides["view_weights"]
    else:
        weights = diffct.helical_view_weights(angles, trajectory[0][:, 2], slice_z)
        close(weights.sum(1), torch.full((shape[0],), math.pi, dtype=torch.float64), rtol=0, atol=1e-12)
    oracle_options = {name: value for name, value in overrides.items() if name != "view_weights"}
    if mode == "weighted":
        # Unit projection weights and a unit FFT response leave filtered data
        # unchanged. Angular weights then give the discrete weighted gather.
        oracle_options = dict(projection_weights=torch.ones((), dtype=torch.float64),
                              filter_response=torch.ones(detector[0] + 1, dtype=torch.float64))
    planes = []
    for index, z in enumerate(slice_z):
        shift = torch.tensor([0., 0., z], dtype=torch.float64)
        shifted = (trajectory[0] - shift, trajectory[1] - shift, *trajectory[2:])
        plane = diffct.fdk(
            rays, shifted, (1, *shape[1:]), angles, isocenter=isocenter - shift,
            detector_spacing=pitch, voxel_spacing=axis_spacing, angular_weights=weights[index],
            backend="torch", precision="float64", **oracle_options,
        )
        planes.append(plane[0])
    expected = torch.stack(planes)
    cotangent = data(shape, phase=.47, dtype=torch.float64)
    inputs = (rays, *overrides.values())
    wanted_gradients = torch.autograd.grad((expected * cotangent).sum(), inputs)
    assert torch.isfinite(expected).all() and expected.abs().max() > 0
    for gradient in wanted_gradients:
        assert torch.isfinite(gradient).all() and gradient.abs().max() > 0

    native_inputs = tuple(value.detach().float().to(native_device).requires_grad_() for value in inputs)
    native_options = dict(zip(overrides, native_inputs[1:]))
    with native_only() as bridge:
        if mode == "weighted":
            actual = diffct.helical_weighted_backproject(
                native_inputs[0], trajectory, shape, native_options["view_weights"],
                isocenter=isocenter, detector_spacing=pitch, voxel_spacing=spacing,
            )
        else:
            actual = diffct.helical_fdk(
                native_inputs[0], trajectory, shape, angles, isocenter=isocenter,
                detector_spacing=pitch, voxel_spacing=spacing, **native_options,
            )
        gradients = torch.autograd.grad((actual * cotangent.to(native_device)).sum(), native_inputs)
        assert bridge.call_count > 0
    assert actual.dtype == torch.float32 and actual.device == native_inputs[0].device
    assert torch.isfinite(actual).all()
    close(actual, expected, rtol=ANALYTICAL_RTOL, atol=ANALYTICAL_ATOL)
    for gradient, wanted in zip(gradients, wanted_gradients):
        assert torch.isfinite(gradient).all()
        close(gradient, wanted, rtol=ANALYTICAL_RTOL, atol=ANALYTICAL_ATOL)
