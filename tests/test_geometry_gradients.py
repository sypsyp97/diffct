"""Geometry derivatives and image/sinogram Hessians for Siddon operators."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pytest
import torch

from diffct import (
    ConeBackprojectorFunction,
    ConeProjectorFunction,
    FanBackprojectorFunction,
    FanProjectorFunction,
    ParallelBackprojectorFunction,
    ParallelProjectorFunction,
    Projector,
)


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]
BEAMS = ("parallel", "fan", "cone")
APIS = ("projector", "lowlevel")


@dataclass
class Case:
    beam: str
    geometry: tuple[torch.Tensor, ...]
    volume: torch.Tensor
    detector: tuple[int, ...]
    pitch: tuple[float, ...]
    spacing: float = 0.8

    @property
    def sino_shape(self):
        return (self.geometry[0].shape[0], *self.detector)


def _case(beam, *, requires_grad=True, device="cuda:0"):
    # Offset angles and coordinates avoid exact voxel edges and corners.
    generator = torch.Generator().manual_seed(619)
    angle = torch.arange(12, dtype=torch.float64) * (math.pi / 12) + 0.137
    radial = torch.stack((angle.cos(), angle.sin()), dim=1)
    tangent = torch.stack((-angle.sin(), angle.cos()), dim=1)
    offset = torch.tensor([0.193, -0.271], dtype=torch.float64)
    if beam == "parallel":
        geometry = (radial, offset.expand_as(radial).clone(), tangent)
    elif beam == "fan":
        geometry = (-24 * radial + offset, 16 * radial + offset, tangent)
    else:
        tilt = 0.173
        radial3 = torch.cat((radial * math.cos(tilt),
                             torch.full((12, 1), math.sin(tilt))), dim=1)
        u = torch.cat((tangent, torch.zeros(12, 1)), dim=1)
        v = torch.linalg.cross(radial3, u, dim=1)
        offset3 = torch.cat((offset.expand(12, 2),
                             torch.linspace(-0.63, 0.79, 12)[:, None]), dim=1)
        geometry = (-24 * radial3 + offset3, 16 * radial3 + offset3, u, v)
    geometry = tuple(t.to(device=device, dtype=torch.float32).contiguous()
                     .requires_grad_(requires_grad) for t in geometry)
    shape = (10, 12, 14) if beam == "cone" else (14, 16)
    axes = [(torch.arange(n, dtype=torch.float64) + 0.5 - n / 2) / n
            for n in shape]
    coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    volume = torch.zeros(shape, dtype=torch.float64)
    for _ in range(5):
        center = (torch.rand(len(shape), generator=generator) - 0.5) * 0.3
        width = 0.065 + 0.045 * torch.rand((), generator=generator)
        amplitude = 0.4 + torch.rand((), generator=generator)
        volume += amplitude * torch.exp(-((coordinates - center) ** 2).sum(-1)
                                       / (2 * width ** 2))
    return Case(beam, geometry, volume.to(device=device, dtype=torch.float32),
                (23, 17) if beam == "cone" else (23,),
                (1.1, 1.05) if beam == "cone" else (1.1,))


def _operators(case, api, *, devices=None):
    if api == "projector":
        projector = Projector(
            case.geometry, tuple(case.volume.shape), case.detector, beam=case.beam,
            detector_spacing=case.pitch if case.beam == "cone" else case.pitch[0],
            voxel_spacing=case.spacing, devices=devices,
        )
        return projector.project, projector.backproject
    g, n, p, s = case.geometry, case.detector, case.pitch, case.spacing
    if case.beam == "parallel":
        return (lambda x: ParallelProjectorFunction.apply(x, *g, n[0], p[0], s),
                lambda y: ParallelBackprojectorFunction.apply(
                    y, *g, p[0], *case.volume.shape, s))
    if case.beam == "fan":
        return (lambda x: FanProjectorFunction.apply(x, *g, n[0], p[0], s),
                lambda y: FanBackprojectorFunction.apply(
                    y, *g, p[0], *case.volume.shape, s))
    return (lambda x: ConeProjectorFunction.apply(x, *g, *n, *p, s),
            lambda y: ConeBackprojectorFunction.apply(
                y, *g, *case.volume.shape, *p, s))


def _random(shape, device, seed):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)).to(device)


def _weighted_sum(value, weight):
    # CUDA kernels use float32; subtract and reduce their outputs in float64.
    return (value.double() * weight.double()).sum()


def _reference_case(beam, *, api):
    """Small random cell-constant volumes with generic detector geometry."""
    views = 5
    angle = torch.arange(views, dtype=torch.float64) * (math.pi / views) + 0.137
    radial = torch.stack((angle.cos(), angle.sin()), dim=1)
    tangent = torch.stack((-angle.sin(), angle.cos()), dim=1)
    offset = torch.tensor([0.193, -0.271], dtype=torch.float64)
    u = tangent + 0.047 * radial
    if beam == "parallel":
        # Low-level nonunit directions also check the line-parameter convention.
        geometry = (1.13 * radial, offset.expand_as(radial).clone(), u)
    elif beam == "fan":
        geometry = (-24 * radial + offset, 16 * radial + offset, u)
    else:
        tilt = 0.173
        radial3 = torch.cat((radial * math.cos(tilt),
                             torch.full((views, 1), math.sin(tilt))), dim=1)
        tangent3 = torch.cat((tangent, torch.zeros(views, 1)), dim=1)
        vertical = torch.linalg.cross(radial3, tangent3, dim=1)
        u = tangent3 + 0.047 * radial3 + 0.031 * vertical
        v = vertical - 0.029 * radial3 + 0.023 * tangent3
        offset3 = torch.cat((offset.expand(views, 2),
                             torch.linspace(-0.63, 0.79, views)[:, None]), dim=1)
        geometry = (-24 * radial3 + offset3, 16 * radial3 + offset3, u, v)
    if api == "projector":
        # Apply Gram-Schmidt in float64 before casting to kernel precision.
        geometry = list(geometry)
        axes = (0, 2) if beam == "parallel" else ((2, 3) if beam == "cone" else (2,))
        orthonormal = []
        for axis in axes:
            direction = geometry[axis]
            for previous in orthonormal:
                direction = direction - (direction * previous).sum(
                    dim=1, keepdim=True) * previous
            direction = direction / torch.linalg.vector_norm(
                direction, dim=1, keepdim=True)
            geometry[axis] = direction
            orthonormal.append(direction)
    geometry = tuple(t.to(device="cuda:0", dtype=torch.float32).contiguous()
                     .requires_grad_() for t in geometry)
    shape = (8, 10, 9) if beam == "cone" else (14, 12)
    volume = _random(shape, geometry[0].device, 619)
    return Case(beam, geometry, volume,
                (7, 5) if beam == "cone" else (9,),
                (0.9, 0.93) if beam == "cone" else (0.9,))


def _numpy(value):
    return value.detach().cpu().double().numpy()


def _siddon_ray(volume, origin, direction, spacing, *, segment):
    """Integrate exact cell lengths using face crossings and piece midpoints."""
    # Geometry coordinates are xyz; array coordinates are [z,] y, x.
    sizes = np.asarray(volume.shape[::-1])
    faces = [(np.arange(n + 1, dtype=np.float64) - n / 2) * spacing
             for n in sizes]
    enter, leave = (0.0, 1.0) if segment else (-np.inf, np.inf)
    crossings = []
    for axis, planes in enumerate(faces):
        if direction[axis] == 0:
            if not planes[0] <= origin[axis] < planes[-1]:
                return 0.0
            continue
        times = (planes - origin[axis]) / direction[axis]
        enter = max(enter, min(times[0], times[-1]))
        leave = min(leave, max(times[0], times[-1]))
        crossings.extend(times)
    if leave <= enter:
        return 0.0
    crossings = np.asarray(crossings, dtype=np.float64)
    times = np.unique(np.concatenate(([enter, leave],
                                     crossings[(crossings > enter) &
                                               (crossings < leave)])))
    midpoints = origin + ((times[:-1] + times[1:]) / 2)[:, None] * direction
    cells = np.floor(midpoints / spacing + sizes / 2).astype(np.int64)
    assert np.all((cells >= 0) & (cells < sizes))
    values = volume[tuple(cells[:, ::-1].T)]
    # Segment parameter is fractional length; parallel t is used as given.
    length_scale = np.linalg.norm(direction) if segment else 1.0
    return float(np.dot(values, np.diff(times)) * length_scale)


def _reference_forward(case, volume, geometry):
    """Float64 Siddon model defined only by the public geometry contract."""
    volume = np.asarray(volume, dtype=np.float64)
    geometry = tuple(np.asarray(g, dtype=np.float64) for g in geometry)
    projection = np.empty(case.sino_shape, dtype=np.float64)
    for view in range(case.sino_shape[0]):
        for cell in np.ndindex(case.detector):
            detector_point = geometry[1][view].copy()
            for axis, (index, count, pitch) in enumerate(
                    zip(cell, case.detector, case.pitch)):
                detector_point += ((index - (count - 1) / 2) * pitch *
                                   geometry[2 + axis][view])
            if case.beam == "parallel":
                origin, direction = detector_point, geometry[0][view]
            else:
                origin = geometry[0][view]
                direction = detector_point - origin
            projection[(view, *cell)] = _siddon_ray(
                volume, origin, direction, case.spacing,
                segment=case.beam != "parallel")
    return projection


def _check_reference_forward(actual, expected):
    actual = _numpy(actual)
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    error = np.max(np.abs(actual - expected))
    limit = 1e-4 * np.max(np.abs(expected))
    assert error <= limit, f"forward reference: max error={error:.8g}, limit={limit:.8g}"


def _check_directional_derivatives(case, gradients, evaluate, *, rtol=1e-4):
    geometry = tuple(_numpy(component) for component in case.geometry)
    eps = 1e-6
    for group, (component, gradient) in enumerate(zip(case.geometry, gradients)):
        assert gradient.shape == component.shape
        assert gradient.device == component.device
        assert gradient.dtype == component.dtype
        assert torch.isfinite(gradient).all()
        gradient = _numpy(gradient)
        # Check each probe separately so cancellation cannot hide a failed probe.
        for probe in range(3):
            rng = np.random.default_rng(730 + 13 * group + probe)
            direction = rng.standard_normal(tuple(component.shape))
            plus, minus = list(geometry), list(geometry)
            plus[group] = geometry[group] + eps * direction
            minus[group] = geometry[group] - eps * direction
            numerical = (evaluate(tuple(plus)) - evaluate(tuple(minus))) / (2 * eps)
            analytical = float(np.sum(gradient * direction))
            scale = max(abs(numerical),
                        1e-3 * np.linalg.norm(gradient) * np.linalg.norm(direction))
            limit = rtol * scale
            assert np.isfinite(numerical) and np.isfinite(analytical)
            assert abs(numerical - analytical) <= limit, (
                f"{case.beam} geometry group {group}, probe {probe}: "
                f"autograd={analytical:.10g}, reference={numerical:.10g}, "
                f"error={abs(numerical - analytical):.8g}, limit={limit:.8g}"
            )


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("operation", ("project", "backproject"))
def test_geometry_vjp_matches_forward_finite_difference(beam, api, operation):
    case = _reference_case(beam, api=api)
    project, backproject = _operators(case, api)
    weight = _random(case.sino_shape, case.volume.device, 811)
    volume, reference_weight = _numpy(case.volume), _numpy(weight)
    geometry = tuple(_numpy(component) for component in case.geometry)
    projection = project(case.volume)
    _check_reference_forward(projection, _reference_forward(case, volume, geometry))
    if operation == "project":
        loss = _weighted_sum(projection, weight)
    else:
        loss = _weighted_sum(backproject(weight), case.volume)
    gradients = torch.autograd.grad(loss, case.geometry)
    # <v, A^T w> = <A v, w> uses the same independent forward reference.
    reference_loss = lambda g: float(np.sum(
        _reference_forward(case, volume, g) * reference_weight))
    _check_directional_derivatives(case, gradients, reference_loss)
    if operation == "backproject":
        # The geometry VJP of A^T is the VJP of the same bilinear form y^T A v.
        adjoint_gradients = torch.autograd.grad(
            _weighted_sum(project(case.volume), weight), case.geometry)
        for actual, expected in zip(gradients, adjoint_gradients):
            scale = expected.abs().max().item()
            torch.testing.assert_close(actual, expected, rtol=1e-4,
                                       atol=1e-5 * max(scale, 1e-12))


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("operation", ("project", "backproject"))
@pytest.mark.parametrize("trainable_group", ("axis", "position"))
def test_projector_geometry_snapshot_and_live_references(beam, operation, trainable_group):
    frozen = _case(beam, requires_grad=False)
    frozen_ops = _operators(frozen, "projector")
    value = (frozen.volume if operation == "project" else
             _random(frozen.sino_shape, frozen.volume.device, 909))
    index = 0 if operation == "project" else 1
    before = frozen_ops[index](value).detach()
    with torch.no_grad():
        frozen.geometry[1].add_(0.29)
    torch.testing.assert_close(frozen_ops[index](value), before, rtol=1e-5, atol=1e-5)

    live = _case(beam, requires_grad=False)
    trainable = 1 if trainable_group == "position" else (0 if beam == "parallel" else 2)
    live.geometry[trainable].requires_grad_()
    live_ops = _operators(live, "projector")
    before = live_ops[index](value).detach()
    # One trainable component must retain every component, including this position.
    with torch.no_grad():
        live.geometry[1].add_(0.29)
    after = live_ops[index](value)
    assert (after.detach() - before).abs().max() > 1e-4
    fresh = Case(beam, tuple(t.detach().clone() for t in live.geometry), live.volume,
                 live.detector, live.pitch, live.spacing)
    torch.testing.assert_close(after, _operators(fresh, "projector")[index](value),
                               rtol=1e-5, atol=1e-5)
    weight = _random(after.shape, after.device, 910)
    gradient = torch.autograd.grad(_weighted_sum(after, weight),
                                   live.geometry[trainable])[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().max() > 0


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("operation", ("project", "backproject"))
def test_image_and_sinogram_double_backward(beam, api, operation):
    case = _case(beam, requires_grad=False)
    project, backproject = _operators(case, api)
    apply, adjoint = ((project, backproject) if operation == "project"
                      else (backproject, project))
    shape = case.volume.shape if operation == "project" else case.sino_shape
    x = _random(shape, case.volume.device, 1021).requires_grad_()
    v = _random(shape, case.volume.device, 1022)
    output = apply(x)
    target = _random(output.shape, output.device, 1023)
    loss = 0.5 * (output - target).square().sum()
    gradient = torch.autograd.grad(loss, x, create_graph=True)[0]
    hessian_v = torch.autograd.grad((gradient * v).sum(), x)[0]
    expected = adjoint(apply(v))
    torch.testing.assert_close(hessian_v, expected, rtol=1e-4,
                               atol=1e-5 * max(expected.abs().max().item(), 1e-12))


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("api", APIS)
def test_mixed_image_geometry_derivative(beam, api):
    case = _reference_case(beam, api=api)
    project, _ = _operators(case, api)
    x = case.volume.detach().clone().requires_grad_()
    v = _random(x.shape, x.device, 1121)
    target = _random(case.sino_shape, x.device, 1122)
    reference_x, reference_v, reference_target = _numpy(x), _numpy(v), _numpy(target)
    geometry = tuple(_numpy(component) for component in case.geometry)
    projection = project(x)
    _check_reference_forward(projection, _reference_forward(case, reference_x, geometry))
    _check_reference_forward(project(v), _reference_forward(case, reference_v, geometry))
    residual = projection - target
    gradient = torch.autograd.grad(0.5 * residual.square().sum(), x,
                                   create_graph=True)[0]
    mixed = torch.autograd.grad(_weighted_sum(gradient, v), case.geometry)
    # Differentiate <A v, A x - y>, including both occurrences of A.
    reference_loss = lambda g: float(np.sum(
        _reference_forward(case, reference_v, g) *
        (_reference_forward(case, reference_x, g) - reference_target)))
    # The mixed term multiplies two float32 GPU projections.
    _check_directional_derivatives(case, mixed, reference_loss, rtol=1e-3)


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("operation", ("project", "backproject"))
def test_pure_geometry_second_derivative_has_clear_error(beam, api, operation):
    case = _case(beam)
    project, backproject = _operators(case, api)
    weight = _random(case.sino_shape, case.volume.device, 1221)
    loss = (_weighted_sum(project(case.volume), weight) if operation == "project"
            else _weighted_sum(backproject(weight), case.volume))
    gradients = torch.autograd.grad(loss, case.geometry, create_graph=True)
    probe = sum(_weighted_sum(g, _random(g.shape, g.device, 1222 + i))
                for i, g in enumerate(gradients))
    with pytest.raises(RuntimeError, match="second derivatives with respect to the geometry"):
        torch.autograd.grad(probe, case.geometry)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices are required")
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("operation", ("project", "backproject"))
def test_multigpu_geometry_gradients_match_single_gpu(beam, operation):
    single, multi = _case(beam), _case(beam)
    single_ops = _operators(single, "projector", devices=[0])
    multi_ops = _operators(multi, "projector", devices=[0, 1])
    weight = _random(single.sino_shape, single.volume.device, 1321)

    def gradients(case, operators):
        loss = (_weighted_sum(operators[0](case.volume), weight)
                if operation == "project" else
                _weighted_sum(operators[1](weight), case.volume))
        return torch.autograd.grad(loss, case.geometry)

    for actual, expected in zip(gradients(multi, multi_ops), gradients(single, single_ops)):
        torch.testing.assert_close(actual, expected, rtol=1e-4,
                                   atol=1e-5 * max(expected.abs().max().item(), 1e-12))
