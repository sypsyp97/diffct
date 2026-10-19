"""Rigid transforms accept frames whose actual detector is a valid surface."""

from dataclasses import replace

import pytest
import torch

from diffct import Projector, transform_trajectory
from tests.test_detector_surfaces import _Case, _matrix
from tests.test_geometry_conversions import _rotation


def _surface_case(beam, reference, *, dtype=torch.float64, device="cpu"):
    dimensions = 3 if beam == "cone" else 2
    source = torch.tensor([[-3.0, .17, -.23][:dimensions]], dtype=dtype, device=device).repeat(2, 1)
    axis_u = torch.zeros_like(source)
    axis_u[:, 1] = 1
    center = source.clone() if reference == "same-center" else source + 2 * axis_u
    frames = (source, center, axis_u)
    if beam == "cone":
        axis_v = torch.zeros_like(source)
        axis_v[:, 2] = 1
        frames += (axis_v,)
    return _Case(beam, frames, (2, 3, 4) if beam == "cone" else (3, 4),
                 (3, 2) if beam == "cone" else (3,),
                 (.71, .83) if beam == "cone" else (.71,))


def _surface(u, v):
    return torch.stack((u, v, torch.full_like(u, 6.0)), dim=-1)


@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("reference", ["same-center", "edge-on"])
@pytest.mark.parametrize("transform", ["identity", "shared", "per-view"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_surface_supported_frames_accept_identity_and_proper_rigid_transforms(beam, reference, transform, dtype):
    case = _surface_case(beam, reference, dtype=dtype)
    dimensions = len(case.shape)
    rotation, translation = None, None
    if transform != "identity":
        angles = torch.tensor([.13, -.19] if transform == "per-view" else .13, dtype=dtype)
        rotation = _rotation(angles, dimensions)
        translation = torch.linspace(.07, .31, (2 if transform == "per-view" else 1) * dimensions,
                                     dtype=dtype).reshape((2, dimensions) if transform == "per-view" else (dimensions,))
    actual = transform_trajectory(case.trajectory, rotation, translation, beam=beam)
    for index, (original, value) in enumerate(zip(case.trajectory, actual)):
        expected = original if rotation is None else (rotation @ original.unsqueeze(-1)).squeeze(-1)
        if translation is not None and index < 2:
            expected = expected + translation
        assert value.dtype == dtype and value.device == original.device
        torch.testing.assert_close(value, expected)
    for axis in actual[2:]:
        torch.testing.assert_close(axis.norm(dim=-1), torch.ones(2, dtype=dtype))
    if beam == "cone":
        torch.testing.assert_close((actual[2] * actual[3]).sum(-1), torch.zeros(2, dtype=dtype))


@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("reference", ["same-center", "edge-on"])
@pytest.mark.parametrize("per_view", [False, True])
def test_surface_supported_transform_preserves_frame_and_transform_gradients(beam, reference, per_view):
    case = _surface_case(beam, reference)
    dimensions = len(case.shape)
    frames = tuple(value.clone().requires_grad_() for value in case.trajectory)
    angles = torch.tensor([.13, -.19] if per_view else .13, dtype=torch.float64, requires_grad=True)
    translation = torch.linspace(.07, .31, (2 if per_view else 1) * dimensions, dtype=torch.float64)
    translation = translation.reshape((2, dimensions) if per_view else (dimensions,)).requires_grad_()

    def converted(*values):
        return transform_trajectory(values[:-2], _rotation(values[-2], dimensions), values[-1], beam=beam)

    assert torch.autograd.gradcheck(converted, (*frames, angles, translation))


@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("reference", ["same-center", "edge-on"])
def test_transformed_surface_frames_project_and_backproject_against_cell_oracle(beam, reference):
    case = _surface_case(beam, reference)
    dimensions = len(case.shape)
    angle = torch.tensor(.137, dtype=torch.float64, requires_grad=True)
    translation = torch.linspace(.047, .113, dimensions, dtype=torch.float64).requires_grad_()
    transformed = transform_trajectory(case.trajectory, _rotation(angle, dimensions), translation, beam=beam)
    selected = replace(case, trajectory=transformed)
    operator = Projector(case.trajectory, case.shape, case.detector, beam=beam,
                         detector_spacing=case.pitch if beam == "cone" else case.pitch[0],
                         voxel_spacing=case.spacing, detector_surface=_surface,
                         backend="torch", precision="float64")
    # A real surface remains necessary: the reference center/plane is not a
    # valid flat detector, even though every sampled surface pixel is valid.
    with pytest.raises(ValueError):
        Projector(transformed, case.shape, case.detector, beam=beam,
                  backend="torch", precision="float64")
    axes = [(torch.arange(size, dtype=torch.float64) + .5 - size / 2) * pitch
            for size, pitch in zip(case.detector, case.pitch)]
    u, v = torch.meshgrid(*axes, indexing="ij") if beam == "cone" else (axes[0], torch.zeros_like(axes[0]))
    offsets = _surface(u, v).numpy()
    matrix = torch.from_numpy(_matrix(selected, offsets))
    image = torch.sin(torch.arange(torch.tensor(case.shape).prod(), dtype=torch.float64) + .31).reshape(case.shape).requires_grad_()
    rays = torch.cos(torch.arange(torch.tensor(case.sino_shape).prod(), dtype=torch.float64) + .17).reshape(case.sino_shape)
    projection = operator.project(image, trajectory=transformed)
    adjoint = operator.backproject(rays, trajectory=transformed)
    torch.testing.assert_close(projection.flatten(), matrix @ image.detach().flatten(), rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(adjoint.flatten(), matrix.T @ rays.flatten(), rtol=2e-11, atol=2e-11)
    data_gradient, angle_gradient, translation_gradient = torch.autograd.grad(
        (projection * rays).sum(), (image, angle, translation),
    )
    torch.testing.assert_close(data_gradient, adjoint, rtol=2e-11, atol=2e-11)

    def objective(a, t):
        rotation = _rotation(a, dimensions)
        frames = tuple((rotation @ value.unsqueeze(-1)).squeeze(-1) + (t if index < 2 else 0)
                       for index, value in enumerate(case.trajectory))
        oracle = torch.from_numpy(_matrix(replace(case, trajectory=frames), offsets))
        return rays.flatten() @ oracle @ image.detach().flatten()

    step = 1e-6
    expected_angle = (objective(angle.detach() + step, translation.detach())
                      - objective(angle.detach() - step, translation.detach())) / (2 * step)
    expected_translation = torch.empty_like(translation)
    for index in range(dimensions):
        delta = torch.zeros_like(translation)
        delta[index] = step
        expected_translation[index] = (objective(angle.detach(), translation.detach() + delta)
                                       - objective(angle.detach(), translation.detach() - delta)) / (2 * step)
    assert expected_angle.abs() > 1e-4 and expected_translation.abs().max() > 1e-4
    torch.testing.assert_close(angle_gradient, expected_angle, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(translation_gradient, expected_translation, rtol=3e-5, atol=3e-6)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_surface_supported_transform_keeps_cuda_dtype_device_and_graph(beam):
    case = _surface_case(beam, "same-center", device="cuda")
    angle = torch.tensor(.13, dtype=torch.float64, device="cuda", requires_grad=True)
    translation = torch.full((len(case.shape),), .07, dtype=torch.float64, device="cuda", requires_grad=True)
    result = transform_trajectory(case.trajectory, _rotation(angle, len(case.shape)), translation, beam=beam)
    assert all(value.device.type == "cuda" and value.dtype == torch.float64 for value in result)
    gradients = torch.autograd.grad(sum((value * torch.arange(value.numel(), device="cuda").reshape_as(value)).sum()
                                        for value in result), (angle, translation))
    assert all(torch.isfinite(value).all() and value.abs().max() > 1e-4 for value in gradients)
