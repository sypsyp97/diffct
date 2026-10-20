"""Per-call frames keep their own values and graphs on a reusable operator."""

from dataclasses import replace
import inspect

import pytest
import torch

from diffct import ParameterizedSurface, Projector, TensorStore
from tests.test_reference_backend import _case, _data, _matrix, _operator


def _rotated(case, angles):
    c, s = angles.cos(), angles.sin()
    if case.beam == "cone":
        z, o = torch.zeros_like(c), torch.ones_like(c)
        rotation = torch.stack((c, -s, z, s, c, z, z, z, o), dim=-1).reshape(-1, 3, 3)
    else:
        rotation = torch.stack((c, -s, s, c), dim=-1).reshape(-1, 2, 2)
    return tuple(torch.einsum("vij,vj->vi", rotation, value) for value in case.trajectory)


def _objective(case, angles, image, rays):
    matrix = _matrix(replace(case, trajectory=_rotated(case, angles)))
    return rays.flatten() @ matrix @ image.flatten()


def _angle_fd(case, angles, image, rays):
    result = torch.empty_like(angles)
    for index in range(len(angles)):
        delta = torch.zeros_like(angles)
        delta[index] = 1e-6
        result[index] = (_objective(case, angles + delta, image, rays)
                         - _objective(case, angles - delta, image, rays)) / 2e-6
    return result


@pytest.mark.parametrize("method", ["project", "backproject", "__call__", "project_into", "backproject_into"])
def test_override_is_an_optional_keyword_only_argument(method):
    parameter = inspect.signature(getattr(Projector, method)).parameters.get("trajectory")
    assert parameter is not None, f"{method} has no per-call trajectory argument"
    assert parameter.kind == inspect.Parameter.KEYWORD_ONLY and parameter.default is None
    assert inspect.signature(Projector).parameters["trajectory"].default is inspect.Parameter.empty


@pytest.mark.parametrize("beam", ["parallel", "fan", "cone"])
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_rebuilt_nonleaf_trajectory_trains_repeatedly_without_reconstructing_projector(beam, operation, monkeypatch):
    case = _case(beam)
    operator = _operator(case, view_chunk_size=2)
    partition = operator._partition
    image, rays = _data(case.shape), _data(case.projection_shape, phase=.31)
    angles = torch.tensor([.071, -.093, .127], dtype=torch.float64, requires_grad=True)
    original = tuple(value.clone() for value in case.trajectory)

    def no_reconstruction(*args, **kwargs):
        raise AssertionError("per-call geometry reconstructed the Projector")

    monkeypatch.setattr(Projector, "__init__", no_reconstruction)
    for _ in range(3):
        frames = _rotated(case, angles)
        assert all(value.grad_fn is not None for value in frames)
        matrix = _matrix(replace(case, trajectory=frames))
        value, weight = (image, rays) if operation == "project" else (rays, image)
        actual = getattr(operator, operation)(value, trajectory=frames)
        expected = matrix @ image.flatten() if operation == "project" else matrix.T @ rays.flatten()
        torch.testing.assert_close(actual.flatten(), expected, rtol=2e-11, atol=2e-12)
        gradient, = torch.autograd.grad((actual * weight).sum(), angles)
        expected_gradient = _angle_fd(case, angles.detach(), image, rays)
        assert expected_gradient.abs().max() > 1e-4, "uninformative angle fixture"
        torch.testing.assert_close(gradient, expected_gradient, rtol=5e-5, atol=2e-6)
        with torch.no_grad():
            angles.add_(-1e-4 * gradient)
        assert operator._partition is partition
    for actual, expected in zip(case.trajectory, original):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_interleaved_outstanding_calls_backward_independently_and_leave_default_intact(operation):
    case = _case("parallel")
    operator = _operator(case)
    image, rays = _data(case.shape), _data(case.projection_shape, phase=.17)
    first = torch.tensor([.07, -.09, .13], dtype=torch.float64, requires_grad=True)
    second = torch.tensor([-.11, .05, .19], dtype=torch.float64, requires_grad=True)
    value, weight = (image, rays) if operation == "project" else (rays, image)
    apply = getattr(operator, operation)
    default = apply(value).detach().clone()
    a = apply(value, trajectory=_rotated(case, first))
    b = apply(value, trajectory=list(_rotated(case, second)))
    torch.testing.assert_close(apply(value, trajectory=None), default)
    for result, angles in ((a, first), (b, second)):
        gradient, = torch.autograd.grad((result * weight).sum(), angles)
        torch.testing.assert_close(gradient, _angle_fd(case, angles.detach(), image, rays),
                                   rtol=5e-5, atol=2e-6)
    torch.testing.assert_close(apply(value), default)


def test_override_values_are_snapshotted_before_caller_mutation():
    case = _case("parallel")
    frames = tuple(value.clone().requires_grad_() for value in case.trajectory)
    image, rays = _data(case.shape).requires_grad_(), _data(case.projection_shape, phase=.4)
    matrix = _matrix(replace(case, trajectory=frames))
    operator = _operator(case)
    output = operator(image, trajectory=frames)
    with torch.no_grad():
        frames[1].add_(.37)
    later = operator.project(image, trajectory=frames)
    assert (later.detach() - output.detach()).abs().max() > 1e-3
    gradient, = torch.autograd.grad((output * rays).sum(), image)
    torch.testing.assert_close(output.flatten(), matrix @ image.detach().flatten(), rtol=2e-11, atol=2e-12)
    torch.testing.assert_close(gradient.flatten(), matrix.T @ rays.flatten(), rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("beam", ["parallel", "fan", "cone"])
def test_override_batch_chunks_and_tensor_stores_match_independent_matrix(beam):
    case = _case(beam)
    frames = _rotated(case, torch.tensor([.07, -.09, .13], dtype=torch.float64))
    matrix = _matrix(replace(case, trajectory=frames))
    operator = _operator(case, view_chunk_size=2,
                         volume_chunk_shape=(2,) * len(case.shape),
                         detector_chunk_shape=(2,) * len(case.detector))
    image = _data((2, 3, *case.shape)).transpose(0, 1)
    rays = _data((2, 3, *case.projection_shape), phase=.31).transpose(0, 1)
    forward = (image.reshape(6, -1) @ matrix.T).reshape(3, 2, *case.projection_shape)
    adjoint = (rays.reshape(6, -1) @ matrix).reshape(3, 2, *case.shape)
    torch.testing.assert_close(operator(image, trajectory=frames), forward, rtol=2e-11, atol=2e-12)
    torch.testing.assert_close(operator.backproject(rays, trajectory=frames), adjoint, rtol=2e-11, atol=2e-12)
    for operation, value, expected in (("project_into", image, forward), ("backproject_into", rays, adjoint)):
        sample, expected_sample = value[0, 0], expected[0, 0]
        sink = TensorStore(torch.full_like(expected_sample, 73.))
        assert getattr(operator, operation)(TensorStore(sample), sink, trajectory=frames) is sink
        torch.testing.assert_close(sink.tensor, expected_sample, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("beam", ["parallel", "cone"])
@pytest.mark.parametrize("surface_kind", ["callback", "parameterized"])
def test_override_with_detector_surface_keeps_global_frames_and_pixels(beam, surface_kind):
    case = _case(beam)
    frames = _rotated(case, torch.tensor([.07, -.09, .13], dtype=torch.float64))
    shift = torch.tensor(0., dtype=torch.float64, requires_grad=True)

    def offsets(u, v, displacement=0.):
        return torch.stack((u - .03 * u ** 3 + displacement, v, .15 * u.square()), dim=-1)

    surface = offsets if surface_kind == "callback" else ParameterizedSurface(
        lambda u, v, ids, displacement: offsets(u, v, displacement), (shift,),
    )
    operator = _operator(case, detector_surface=surface, view_chunk_size=2,
                         detector_chunk_shape=(2,) * len(case.detector))
    matrix = _matrix(replace(case, trajectory=frames), curved=True)
    image, rays = _data(case.shape), _data(case.projection_shape, phase=.31)
    assert (matrix - _matrix(case, curved=True)).abs().max() > 1e-3
    projected = operator.project(image, trajectory=frames)
    torch.testing.assert_close(projected.flatten(), matrix @ image.flatten(), rtol=2e-11, atol=2e-12)
    torch.testing.assert_close(operator.backproject(rays, trajectory=frames).flatten(), matrix.T @ rays.flatten(), rtol=2e-11, atol=2e-12)
    if surface_kind == "parameterized":
        gradient, = torch.autograd.grad((projected * rays).sum(), shift)
        # A shared local-u displacement is exactly a world center displacement
        # along det_u; differentiate the independent curved cell matrix.
        plus = (frames[0], frames[1] + 1e-6 * frames[2], *frames[2:])
        minus = (frames[0], frames[1] - 1e-6 * frames[2], *frames[2:])
        difference = (_matrix(replace(case, trajectory=plus), curved=True)
                      - _matrix(replace(case, trajectory=minus), curved=True)) / 2e-6
        expected_gradient = rays.flatten() @ difference @ image.flatten()
        assert expected_gradient.abs() > 1e-4, "uninformative surface-parameter fixture"
        torch.testing.assert_close(gradient, expected_gradient, rtol=5e-5, atol=2e-6)


@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_override_preserves_data_second_derivatives(operation):
    case = _case("parallel")
    frames = _rotated(case, torch.tensor([.07, -.09, .13], dtype=torch.float64))
    matrix = _matrix(replace(case, trajectory=frames))
    matrix = matrix if operation == "project" else matrix.T
    shape = case.shape if operation == "project" else case.projection_shape
    value, tangent = _data(shape).requires_grad_(), _data(shape, phase=.71)
    operator = _operator(case)
    output = getattr(operator, operation)(value, trajectory=frames)
    first, = torch.autograd.grad(output.square().sum() / 2, value, create_graph=True)
    second, = torch.autograd.grad((first * tangent).sum(), value)
    torch.testing.assert_close(second.flatten(), matrix.T @ matrix @ tangent.flatten(), rtol=2e-11, atol=2e-12)


class _WriteTrace(TensorStore):
    def __init__(self, tensor):
        super().__init__(tensor)
        self.writes = 0

    def write(self, *args, **kwargs):
        self.writes += 1
        return super().write(*args, **kwargs)


@pytest.mark.parametrize("invalid", ["count", "partial", "nan", "nonunit", "wrong_beam"])
def test_invalid_override_fails_before_execution_or_output_writes_and_recovers(invalid):
    case = _case("parallel")
    operator = _operator(case)
    image = _data(case.shape)
    baseline = operator(image).detach().clone()
    frames = [value.clone() for value in case.trajectory]
    if invalid == "count":
        frames = [value[:2] for value in frames]
    elif invalid == "partial":
        frames = frames[:2]
    elif invalid == "nan":
        frames[1][0, 0] = float("nan")
    elif invalid == "nonunit":
        frames[0] *= 2
    else:
        frames = list(_case("cone").trajectory)
    sink = _WriteTrace(torch.full(case.projection_shape, 73., dtype=torch.float64))
    with pytest.raises((TypeError, ValueError)):
        operator.project_into(image, sink, trajectory=frames)
    assert sink.writes == 0
    torch.testing.assert_close(sink.tensor, torch.full_like(sink.tensor, 73.))
    with pytest.raises((TypeError, ValueError)):
        operator(image, trajectory=frames)
    torch.testing.assert_close(operator(image), baseline)


def test_into_override_retains_no_autograd_contract():
    case = _case("parallel")
    operator = _operator(case)
    image = _data(case.shape).requires_grad_()
    output = torch.full(case.projection_shape, 73., dtype=torch.float64)
    with pytest.raises((ValueError, RuntimeError)) as error:
        operator.project_into(image, output, trajectory=case.trajectory)
    assert not isinstance(error.value, NotImplementedError), "into autograd validation is still pending"
    torch.testing.assert_close(output, torch.full_like(output, 73.))
    with torch.no_grad():
        assert operator.project_into(image, output, trajectory=case.trajectory) is output
    torch.testing.assert_close(output.flatten(), _matrix(case) @ image.detach().flatten(), rtol=2e-11, atol=2e-12)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("backend", ["torch", "numba"])
def test_cuda_override_keeps_geometry_leaf_and_data_gradients(backend):
    case = _case("parallel", scalar=True)
    operator = _operator(case, backend=backend, precision="fast")
    angles = torch.tensor([.07, -.09, .13], device="cuda", dtype=torch.float64, requires_grad=True)
    frames = _rotated(replace(case, trajectory=tuple(v.cuda() for v in case.trajectory)), angles)
    image, rays = _data(case.shape, device="cuda", dtype=torch.float32).requires_grad_(), _data(case.projection_shape, phase=.31)
    output = operator.project(image, trajectory=frames)
    assert output.device == image.device and output.dtype == torch.float32
    matrix = _matrix(replace(case, trajectory=frames))
    data_gradient, angle_gradient = torch.autograd.grad((output * rays.cuda()).sum(), (image, angles))
    torch.testing.assert_close(output.cpu().double().flatten(), matrix @ image.detach().cpu().double().flatten(), rtol=1e-4, atol=2e-5)
    torch.testing.assert_close(data_gradient.cpu().double().flatten(), matrix.T @ rays.flatten(), rtol=1e-4, atol=2e-5)
    expected = _angle_fd(case, angles.detach().cpu(), image.detach().cpu().double(), rays)
    torch.testing.assert_close(angle_gradient.cpu(), expected, rtol=3e-3, atol=3e-4)
