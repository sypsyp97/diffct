"""N3 amendment: unsafe FP32 pitches fail clearly or remain numerically correct.

Run with NUMBA_ENABLE_CUDASIM=1 NUMBA_CUDA_USE_NVIDIA_BINDING=0. Extreme
anisotropy may be rejected by numba, but auto must execute without an error.
Pitch ratios of 100 must execute the production native kernels.
"""

from contextlib import nullcontext
from dataclasses import replace
import re

import pytest
import torch

from tests._native_per_axis_spacing import (
    GEOMETRY_ATOL, GEOMETRY_RTOL, NATIVE_REQUIRED,
    case, close, cuda_device, data, native_device, native_only, operator,
)


def extreme_scan(name, *, learnable=False):
    if name == "fan-ratio-1e8":
        beam, shape, spacing = "fan", (3, 5), (1e8, 1.)
        frames = ([[-4., -1.]], [[4., 1.]], [[0., 1.]])
    elif name == "cone-subnormal-pitch":
        beam, shape, spacing = "cone", (3, 3, 3), (1., 1., 2e-38)
        frames = ([[0., -4., 0.]], [[0., 4., 0.]], [[1., 0., 0.]], [[0., 0., 1.]])
    elif name == "fan-vjp-1e40":
        beam, shape, spacing = "fan", (3, 5), (1e20, 1e-20)
        frames = ([[-4e-20, -.25]], [[4e-20, .75]], [[1., 0.]])
    else:
        raise ValueError(name)
    # Preserve identical represented coordinates in both numerical paths.
    trajectory = tuple(torch.tensor(frame, dtype=torch.float32).double().requires_grad_(learnable)
                       for frame in frames)
    return replace(case(beam), shape=shape, spacing=spacing, trajectory=trajectory,
                   detector=(1,) * (len(shape) - 1))


def assert_fp32_range_error(error):
    message = str(error)
    assert "scalar" not in message.lower(), message
    assert re.search(r"(?i)float32|fp32|overflow|underflow|represent|range|ratio", message), message
    assert re.search(r"(?i)spacing|pitch|anisotrop|geometry|coordinate|ratio|voxel", message), message


@NATIVE_REQUIRED
@pytest.mark.parametrize("backend", ["numba", "auto"])
@pytest.mark.parametrize("name", ["fan-ratio-1e8", "cone-subnormal-pitch"])
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_n3_extreme_pitches_match_float64_or_reject_safely(native_device, backend, name, operation):
    scan = extreme_scan(name)
    image, rays = .5 + data(scan.shape).abs(), .5 + data(scan.projection_shape, phase=.31).abs()
    value, weight = (image, rays) if operation == "project" else (rays, image)
    reference_data = value.double().requires_grad_()
    reference = operator(scan, backend="torch", precision="float64")
    expected = getattr(reference, operation)(reference_data)
    expected_gradient, = torch.autograd.grad((expected * weight.double()).sum(), reference_data)
    assert torch.isfinite(expected).all() and expected.abs().max() > 0
    assert torch.isfinite(expected_gradient).all() and expected_gradient.abs().max() > 0
    try:
        native = operator(scan, backend=backend)
        assert native.backend in (("numba",) if backend == "numba" else ("numba", "torch"))
        selected_device = native_device if native.backend == "numba" else torch.device("cpu")
        native_data = value.to(selected_device).requires_grad_()
        with native_only() if native.backend == "numba" else nullcontext():
            actual = getattr(native, operation)(native_data)
            gradient, = torch.autograd.grad((actual * weight.to(selected_device)).sum(), native_data)
    except (ValueError, RuntimeError, OverflowError) as error:
        if backend != "numba":
            raise
        assert_fp32_range_error(error)
        fallback = operator(scan, backend="auto")
        assert fallback.backend == "torch", "auto must keep torch after a native FP32 range rejection"
        return
    assert torch.isfinite(actual).all() and torch.isfinite(gradient).all()
    close(actual, expected)
    close(gradient, expected_gradient)


@NATIVE_REQUIRED
@pytest.mark.parametrize("backend", ["numba", "auto"])
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_n3_extreme_fan_geometry_vjp_is_finite_and_correct_or_rejects(native_device, backend, operation):
    scan = extreme_scan("fan-vjp-1e40", learnable=True)
    image, rays = .5 + data(scan.shape).abs(), .5 + data(scan.projection_shape, phase=.31).abs()
    value, weight = (image, rays) if operation == "project" else (rays, image)
    reference_data = value.double().requires_grad_()
    reference = operator(scan, backend="torch", precision="float64")
    expected = getattr(reference, operation)(reference_data)
    wanted = torch.autograd.grad((expected * weight.double()).sum(), (reference_data, *scan.trajectory))
    assert torch.isfinite(expected).all() and expected.abs().max() > 0
    assert all(torch.isfinite(gradient).all() for gradient in wanted)
    assert wanted[0].abs().max() > 0 and any(gradient.abs().max() > 1 for gradient in wanted[1:])
    try:
        native = operator(scan, backend=backend)
        assert native.backend in (("numba",) if backend == "numba" else ("numba", "torch"))
        selected_device = native_device if native.backend == "numba" else torch.device("cpu")
        native_data = value.to(selected_device).requires_grad_()
        with native_only() if native.backend == "numba" else nullcontext():
            actual = getattr(native, operation)(native_data)
            gradients = torch.autograd.grad((actual * weight.to(selected_device)).sum(),
                                            (native_data, *scan.trajectory))
    except (ValueError, RuntimeError, OverflowError) as error:
        if backend != "numba":
            raise
        assert_fp32_range_error(error)
        fallback = operator(scan, backend="auto")
        assert fallback.backend == "torch", "auto must keep torch after a native FP32 range rejection"
        return
    assert torch.isfinite(actual).all()
    close(actual, expected)
    for index, (gradient, expected_gradient) in enumerate(zip(gradients, wanted)):
        assert torch.isfinite(gradient).all()
        if index == 0:
            close(gradient, expected_gradient)
        else:
            close(gradient, expected_gradient, rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL)


@NATIVE_REQUIRED
@pytest.mark.parametrize("backend", ["numba", "auto"])
@pytest.mark.parametrize("beam,axis", [("parallel", 0), ("parallel", 1), ("fan", 0), ("fan", 1),
                                      ("cone", 0), ("cone", 1), ("cone", 2)])
def test_n3_ratio_100_runs_natively_and_matches_float64(native_device, backend, beam, axis):
    scan = case(beam)
    rank = len(scan.shape)
    spacing = tuple(100. if index == axis else 1. for index in range(rank))
    shape = (3, 5, 7) if rank == 3 else (3, 5)
    directions = torch.eye(rank, dtype=torch.float64)
    centers = 4 * max(size * pitch for size, pitch in zip(shape, spacing)) * directions
    u = torch.roll(directions, 1, dims=1)
    trajectory = ((directions, centers, u) if beam == "parallel" else (-centers, centers, u))
    if rank == 3:
        trajectory += (torch.linalg.cross(directions, u, dim=-1),)
    scan = replace(scan, trajectory=trajectory, shape=shape, spacing=spacing, detector=(1,) * (rank - 1))
    image, rays = data(shape), data(scan.projection_shape, phase=.31)
    reference = operator(scan, backend="torch", precision="float64")
    expected = reference.project(image.double()), reference.backproject(rays.double())
    with native_only() as bridge:
        native = operator(scan, backend=backend)
        assert native.backend == "numba"
        actual = native.project(image.to(native_device)), native.backproject(rays.to(native_device))
        assert bridge.call_count > 0
    for result, wanted in zip(actual, expected):
        assert torch.isfinite(result).all() and wanted.abs().max() > 0
        close(result, wanted)
