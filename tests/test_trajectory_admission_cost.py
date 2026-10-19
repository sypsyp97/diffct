"""Per-call validation stays bounded without weakening actual-pixel guards."""

import math

import pytest
import torch

from diffct import ParameterizedSurface, Projector, circular_trajectory_2d_fan, circular_trajectory_3d
from diffct.backends import ReferenceProjector


NATIVE = pytest.param("numba", marks=[pytest.mark.cuda, pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None, reason="NVIDIA CUDA is required",
)])


def _frames(beam, views=3, device="cpu"):
    generator = circular_trajectory_3d if beam == "cone" else circular_trajectory_2d_fan
    return generator(views, 1000.0, 1500.0, device=device, dtype=torch.float64)


def _admission_only(monkeypatch, operation):
    calls = []

    def execute(owner, value, *args):
        calls.append(owner)
        return args[0] if args else value

    seam = "_execute_into" if operation.endswith("_into") else "_" + operation
    monkeypatch.setattr(Projector, seam, execute)
    return calls


def _call(operator, operation, trajectory):
    projecting = operation.startswith("project")
    value = torch.zeros(operator.volume_shape if projecting else operator.projection_shape, dtype=operator.dtype)
    if operation.endswith("_into"):
        output = torch.full(operator.projection_shape if projecting else operator.volume_shape, 73.0, dtype=operator.dtype)
        return getattr(operator, operation)(value, output, trajectory=trajectory)
    return getattr(operator, operation)(value, trajectory=trajectory)


class _AdmissionTrace:
    def __init__(self, monkeypatch):
        self.ray_batches = 0
        self.ray_inputs = 0
        self.d2h_calls = 0
        rays, tensor_to, tensor_cpu = ReferenceProjector._rays, torch.Tensor.to, torch.Tensor.cpu

        def traced_rays(reference, views, pixels, device, *args, **kwargs):
            if kwargs.get("validate", args[0] if args else False):
                self.ray_batches += 1
                self.ray_inputs += (views.stop - views.start) * math.prod(pixel.stop - pixel.start for pixel in pixels)
            return rays(reference, views, pixels, device, *args, **kwargs)

        def traced_to(value, *args, **kwargs):
            result = tensor_to(value, *args, **kwargs)
            self.d2h_calls += int(value.device.type != "cpu" and result.device.type == "cpu")
            return result

        def traced_cpu(value, *args, **kwargs):
            result = tensor_cpu(value, *args, **kwargs)
            self.d2h_calls += int(value.device.type != "cpu")
            return result

        monkeypatch.setattr(ReferenceProjector, "_rays", traced_rays)
        monkeypatch.setattr(torch.Tensor, "to", traced_to)
        monkeypatch.setattr(torch.Tensor, "cpu", traced_cpu)


@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("backend", ["torch", NATIVE])
@pytest.mark.parametrize("operation", ["project", "backproject", "project_into", "backproject_into"])
def test_well_separated_flat_admission_does_not_walk_detector_pixels(beam, backend, operation, monkeypatch):
    frames = _frames(beam)
    calls = _admission_only(monkeypatch, operation)
    # Detector size must not turn an ordinary flat detector into a pixel
    # scan. Lightweight reference owners are allowed.
    for detector in (((7, 5), (43, 29)) if beam == "cone" else ((7,), (509,))):
        operator = Projector(frames, (3, 4, 5) if beam == "cone" else (4, 5), detector,
                             beam=beam, backend=backend)
        with monkeypatch.context() as local:
            trace = _AdmissionTrace(local)
            _call(operator, operation, frames)
        assert trace.ray_inputs <= 8 * len(frames[0]), (
            f"flat admission enumerated {trace.ray_inputs} ray inputs in {trace.ray_batches} batches "
            f"for {len(frames[0])} views and detector {detector}"
        )
    assert len(calls) == 2


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("backend", ["torch", NATIVE])
@pytest.mark.parametrize("surface_kind", ["legacy", "parameterized"])
def test_surface_admission_stages_cuda_frames_and_parameters_once(backend, surface_kind, monkeypatch):
    frames = _frames("cone", views=3, device="cuda")
    sag = torch.tensor(.03, dtype=torch.float64, device="cuda")

    def sampler(u, v, ids, coefficient):
        assert coefficient.device.type == "cpu"
        return torch.stack((u, v, coefficient * u.square()), dim=-1)

    if surface_kind == "parameterized":
        surface = ParameterizedSurface(sampler, (sag,))
    else:
        def surface(u, v):
            return torch.stack((u, v, .03 * u.square()), dim=-1)

    for chunks in ((5, 4), (2, 2)):
        operator = Projector(frames, (3, 4, 5), (5, 4), beam="cone", backend=backend,
                             detector_surface=surface, view_chunk_size=2, detector_chunk_shape=chunks)
        _admission_only(monkeypatch, "project")
        with monkeypatch.context() as local:
            trace = _AdmissionTrace(local)
            _call(operator, "project", frames)
        expected_components = len(frames) + int(surface_kind == "parameterized")
        assert 0 < trace.d2h_calls <= expected_components, (
            f"surface admission made {trace.d2h_calls} accelerator-to-CPU tensor transfers "
            f"for {expected_components} geometry components and detector chunks {chunks}"
        )


def _assert_rejected_before_write(operator, frames, operation, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("numerical execution reached invalid geometry")

    monkeypatch.setattr(Projector, "_project", forbidden)
    monkeypatch.setattr(Projector, "_backproject", forbidden)
    monkeypatch.setattr(Projector, "_execute_into", forbidden)
    projecting = operation.startswith("project")
    value = torch.zeros(operator.volume_shape if projecting else operator.projection_shape, dtype=operator.dtype)
    output = torch.full(operator.projection_shape if projecting else operator.volume_shape, 73.0, dtype=operator.dtype)
    with pytest.raises(ValueError, match="finite|differ|coincid|within"):
        if operation.endswith("_into"):
            getattr(operator, operation)(value, output, trajectory=frames)
        else:
            getattr(operator, operation)(value, trajectory=frames)
    torch.testing.assert_close(output, torch.full_like(output, 73.0))


@pytest.mark.parametrize("backend", ["torch", NATIVE])
@pytest.mark.parametrize("operation", ["project", "backproject", "project_into", "backproject_into"])
def test_flat_interior_pixel_coincident_only_after_fp32_staging_is_rejected(backend, operation, monkeypatch):
    frames = (torch.tensor([[1e6, 0.0]], dtype=torch.float64),
              torch.tensor([[1e6 + .01, 1.0]], dtype=torch.float64),
              torch.tensor([[0.0, 1.0]], dtype=torch.float64))
    # Raw frame admission and both outer corners are noncoincident. Only the
    # interior u=-1 pixel collapses to the source after staging components.
    for coordinate in (-2.0, 2.0):
        assert not torch.equal(frames[1].float() + coordinate * frames[2].float(), frames[0].float())
    assert torch.equal(frames[1].float() - frames[2].float(), frames[0].float())
    operator = Projector(_frames("fan", 1), (2, 3), (5,), beam="fan", backend=backend)
    _assert_rejected_before_write(operator, frames, operation, monkeypatch)


@pytest.mark.parametrize("backend", ["torch", NATIVE])
@pytest.mark.parametrize("operation", ["project", "backproject", "project_into", "backproject_into"])
def test_valid_flat_frame_with_source_inside_corner_bounds_is_admitted(backend, operation, monkeypatch):
    diagonal = math.sqrt(.5)
    frames = (torch.tensor([[0.0, 0.0, -5.0]], dtype=torch.float64),
              torch.tensor([[0.0, 0.0, 5.0]], dtype=torch.float64),
              torch.tensor([[diagonal, 0.0, diagonal]], dtype=torch.float64),
              torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64))
    corners = torch.stack([frames[1][0] + u * frames[2][0] + v * frames[3][0]
                           for u in (-20.0, 20.0) for v in (-1.0, 1.0)])
    assert torch.all(frames[0][0] >= corners.amin(0)) and torch.all(frames[0][0] <= corners.amax(0))
    # The source lies outside the detector plane. An inconclusive cheap
    # noncoincidence certificate must accept by another check/fallback.
    normal = torch.linalg.cross(frames[2], frames[3], dim=-1)
    assert torch.abs(((frames[1] - frames[0]) * normal).sum()) > 1
    operator = Projector(_frames("cone", 1), (2, 3, 4), (41, 3), beam="cone", backend=backend)
    calls = _admission_only(monkeypatch, operation)
    _call(operator, operation, frames)
    assert len(calls) == 1


@pytest.mark.parametrize("backend", ["torch", NATIVE])
@pytest.mark.parametrize("overflow", ["world-point", "direction"])
def test_flat_corner_and_direction_overflow_are_rejected_before_write(backend, overflow, monkeypatch):
    source = torch.tensor([[0.0 if overflow == "world-point" else -3e38, 0.0]], dtype=torch.float64)
    center = torch.tensor([[3e38 if overflow == "world-point" else 0.0, 0.0]], dtype=torch.float64)
    axis = torch.full((1, 2), math.sqrt(.5), dtype=torch.float64)
    frames = (source, center, axis)
    baseline = (torch.tensor([[-1.0, 0.0]], dtype=torch.float64), torch.zeros((1, 2), dtype=torch.float64),
                torch.tensor([[0.0, 1.0]], dtype=torch.float64))
    operator = Projector(baseline, (2, 3), (3,), beam="fan", backend=backend,
                         voxel_spacing=1e25, detector_spacing=3e38)
    if overflow == "direction":
        corner = center + 3e38 * axis
        assert torch.isfinite(corner.float()).all() and torch.isfinite(source.float()).all()
        assert not torch.isfinite(corner.float() - source.float()).all()
    _assert_rejected_before_write(operator, frames, "project_into", monkeypatch)


@pytest.mark.parametrize("backend", ["torch", NATIVE])
@pytest.mark.parametrize("surface_kind", ["legacy", "parameterized"])
@pytest.mark.parametrize("invalid", ["nan", "coincident", "float32-overflow"])
def test_surface_invalid_interior_pixels_remain_rejected(backend, surface_kind, invalid, monkeypatch):
    active = torch.tensor(0.0, dtype=torch.float64)

    def offsets(u, v, enabled):
        normal = torch.zeros_like(u)
        if bool(enabled):
            normal = torch.where((u == 0) & (v == 0),
                                 normal.new_tensor({"nan": float("nan"), "coincident": -15.0,
                                                    "float32-overflow": 1e39}[invalid]), normal)
        return torch.stack((u, v, normal), dim=-1)

    if surface_kind == "parameterized":
        def sampler(u, v, ids, enabled):
            return offsets(u, v, enabled)

        surface = ParameterizedSurface(sampler, (active,))
    else:
        def surface(u, v):
            return offsets(u, v, active)
    # For this axis-aligned circular frame, cross(U,V) points from the source
    # to the center. An interior normal displacement -SDD reaches the source.
    frames = (torch.tensor([[-9.0, 0.0, 0.0]], dtype=torch.float64),
              torch.tensor([[6.0, 0.0, 0.0]], dtype=torch.float64),
              torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64),
              torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64))
    operator = Projector(frames, (2, 3, 4), (5, 5), beam="cone", backend=backend,
                         detector_surface=surface, detector_chunk_shape=(2, 2))
    active.fill_(1.0)
    _assert_rejected_before_write(operator, frames, "project_into", monkeypatch)


@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_parameterized_surface_admission_preserves_differentiable_potential_sampler(operation):
    parameter = torch.tensor(.6, dtype=torch.float64, requires_grad=True)

    def sampler(u, v, ids, p):
        # A pure sampler may derive its offsets from a differentiable
        # potential. Its explicit parameter must still support autograd when
        # admission invokes the same callable as numerical execution.
        with torch.enable_grad():
            offset, = torch.autograd.grad(p.pow(3) / 3, p, create_graph=True)
        return torch.stack((u + offset, v, torch.zeros_like(u)), dim=-1)

    frames = (torch.tensor([[-2.0, 0.0]], dtype=torch.float64),
              torch.tensor([[2.0, 0.0]], dtype=torch.float64),
              torch.tensor([[0.0, 1.0]], dtype=torch.float64))
    operator = Projector(frames, (1, 1), (1,), beam="fan", backend="torch", precision="float64",
                         detector_surface=ParameterizedSurface(sampler, (parameter,)))
    data = torch.full((1, 1), 3.0, dtype=torch.float64, requires_grad=True)
    result = getattr(operator, operation)(data, trajectory=frames)
    data_gradient, geometry_gradient = torch.autograd.grad(result.sum(), (data, parameter))
    # The ray joins (-2,0) to (2,p²). At p=.6 it crosses both x faces of
    # the unit voxel, so its exact chord is sqrt(1 + p⁴/16).
    p = parameter.detach()
    chord = (1 + p.pow(4) / 16).sqrt()
    chord_derivative = p.pow(3) / (8 * chord)
    torch.testing.assert_close(result, torch.full_like(result, 3.0 * chord), rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(data_gradient, torch.full_like(data_gradient, chord), rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(geometry_gradient, 3.0 * chord_derivative, rtol=2e-12, atol=2e-12)
