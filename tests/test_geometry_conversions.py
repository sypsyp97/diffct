"""Angle tensors and rigid world transforms use ordinary differentiable tensors."""

import math

import pytest
import torch

import diffct


GENERATORS = ("circular_trajectory_2d_parallel", "circular_trajectory_2d_fan", "circular_trajectory_3d")


def _generate(name, angles, **kwargs):
    function = getattr(diffct, name)
    return function(angles, **kwargs) if name.endswith("parallel") else function(angles, 5., 9., **kwargs)


def _expected(name, angles, distance=0.):
    c, s = angles.cos(), angles.sin()
    if name.endswith("parallel"):
        tangent = torch.stack((-s, c), dim=-1)
        return torch.stack((c, s), dim=-1), distance * tangent, tangent
    source = torch.stack((-5 * s, 5 * c), dim=-1)
    center = torch.stack((4 * s, -4 * c), dim=-1)
    u = torch.stack((c, s), dim=-1)
    if name == "circular_trajectory_3d":
        zero = torch.zeros_like(angles)[:, None]
        return *(torch.cat((value, zero), dim=1) for value in (source, center, u)), torch.stack((0 * c, 0 * c, torch.ones_like(c)), dim=-1)
    return source, center, u


@pytest.mark.parametrize("name", GENERATORS)
def test_arbitrary_unsorted_angle_tensor_owns_dtype_device_and_gradient(name):
    angles = torch.tensor([1.13, -.27, .41], dtype=torch.float64, requires_grad=True)
    # Legacy device/dtype keywords govern count generation only.
    actual = _generate(name, angles, device="cuda", dtype=torch.float32)
    expected = _expected(name, angles)
    for value, reference in zip(actual, expected):
        assert value.dtype == angles.dtype and value.device == angles.device
        torch.testing.assert_close(value, reference, rtol=1e-13, atol=1e-13)
    weights = [torch.linspace(.1, .9, value.numel(), dtype=value.dtype).reshape(value.shape)
               for value in actual]
    gradient, = torch.autograd.grad(sum((value * weight).sum() for value, weight in zip(actual, weights)), angles)
    reference_gradient, = torch.autograd.grad(sum((value * weight).sum() for value, weight in zip(expected, weights)), angles)
    assert reference_gradient.abs().max() > .1
    torch.testing.assert_close(gradient, reference_gradient, rtol=1e-12, atol=1e-12)
    assert torch.autograd.gradcheck(lambda a: _generate(name, a), (angles,))


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize("custom_interval", [False, True])
def test_count_generation_preserves_exact_legacy_orbit_and_defaults(name, custom_interval):
    options = {"device": "cpu", "dtype": torch.float64}
    start, end = (0.23, 2.47) if custom_interval else (0., math.pi if name.endswith("parallel") else 2 * math.pi)
    if custom_interval:
        options.update(start_angle=start, end_angle=end)
    if name.endswith("parallel"):
        options["detector_distance"] = .73
    actual = _generate(name, 4, **options)
    expected = _expected(name, start + torch.arange(4, dtype=torch.float64) * (end - start) / 4, distance=.73)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("bad", [torch.empty(0), torch.ones(2, 1), torch.tensor([1, 2]), torch.tensor([0., float("nan")]), torch.tensor([float("inf")])])
@pytest.mark.parametrize("name", GENERATORS)
def test_angle_tensor_rejects_empty_nonvector_integer_or_nonfinite_input(name, bad):
    with pytest.raises((TypeError, ValueError)):
        _generate(name, bad, device="cpu")


def _transform():
    function = getattr(diffct, "transform_trajectory", None)
    assert callable(function), "required diffct.transform_trajectory export is missing"
    assert "transform_trajectory" in diffct.__all__
    return function


def _rotation(angle, dimensions):
    c, s = angle.cos(), angle.sin()
    if dimensions == 2:
        return torch.stack((c, -s, s, c), dim=-1).reshape(*angle.shape, 2, 2)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, z, s, c, z, z, z, o), dim=-1).reshape(*angle.shape, 3, 3)


def _frames(beam, angles):
    if beam in ("parallel", "fan", "cone"):
        name = {"parallel": GENERATORS[0], "fan": GENERATORS[1], "cone": GENERATORS[2]}[beam]
        # Count generation is already supported; keep transform fixtures independent of angle API.
        return _expected(name, angles, distance=.4)
    base = _expected(GENERATORS[0], angles, distance=.4)
    zero = torch.zeros_like(angles)[:, None]
    return *(torch.cat((value, zero), dim=1) for value in base), torch.stack((0 * angles, 0 * angles, torch.ones_like(angles)), dim=-1)


@pytest.mark.parametrize("beam", ["parallel", "fan", "cone", "parallel3d"])
@pytest.mark.parametrize("per_view", [False, True])
def test_rigid_transform_moves_points_and_rotates_directions_about_world_origin(beam, per_view):
    dimensions = 3 if beam in ("cone", "parallel3d") else 2
    angles = torch.tensor([.23, -.41, 1.17], dtype=torch.float64)
    frames = _frames(beam, angles)
    rotation = _rotation(torch.tensor([.31, -.19, .47] if per_view else .31, dtype=torch.float64), dimensions)
    translation = torch.arange(1, (3 if per_view else 1) * dimensions + 1, dtype=torch.float64).reshape((3, dimensions) if per_view else (dimensions,)) * .17
    actual = _transform()(frames, rotation, translation, beam=beam)
    for index, (value, original) in enumerate(zip(actual, frames)):
        expected = (torch.einsum("vij,vj->vi", rotation, original) if per_view else original @ rotation.T)
        if index == 1 or (index == 0 and beam in ("fan", "cone")):
            expected = expected + translation
        torch.testing.assert_close(value, expected, rtol=1e-12, atol=1e-12)
    for value, original in zip(_transform()(frames, beam=beam), frames):
        torch.testing.assert_close(value, original)
    # Mixed broadcast forms also remain valid.
    mixed = _transform()(frames, rotation if per_view else None, translation if not per_view else None, beam=beam)
    assert all(value.shape == (3, dimensions) for value in mixed)


@pytest.mark.parametrize("beam", ["fan", "parallel3d"])
def test_rigid_transform_gradcheck_reaches_frame_rotation_and_translation_leaves(beam):
    dimensions = 3 if beam == "parallel3d" else 2
    angles = torch.tensor([.23, .67], dtype=torch.float64, requires_grad=True)
    rotation_angles = torch.tensor([.31, -.19], dtype=torch.float64, requires_grad=True)
    translation = torch.linspace(.1, .6, 2 * dimensions, dtype=torch.float64).reshape(2, dimensions).requires_grad_()
    function = _transform()

    def converted(a, r, t):
        return function(_frames(beam, a), _rotation(r, dimensions), t, beam=beam)

    assert torch.autograd.gradcheck(converted, (angles, rotation_angles, translation))
    outputs = converted(angles, rotation_angles, translation)
    weights = [torch.linspace(.1, .9, x.numel(), dtype=x.dtype).reshape(x.shape) for x in outputs]
    gradients = torch.autograd.grad(sum((x * w).sum() for x, w in zip(outputs, weights)), (angles, rotation_angles, translation))
    assert all(torch.isfinite(g).all() and g.abs().max() > 1e-3 for g in gradients)


@pytest.mark.parametrize("invalid", ["rotation_shape", "rotation_count", "reflection", "nonorthogonal", "rotation_nan", "translation_shape", "translation_count", "translation_nan", "dtype", "bad_frame", "beam"])
def test_transform_rejects_invalid_rigid_transform_or_frame(invalid):
    frames = _frames("cone", torch.tensor([.23, .67], dtype=torch.float64))
    rotation, translation, beam = torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), "cone"
    if invalid == "rotation_shape":
        rotation = torch.eye(2, dtype=torch.float64)
    elif invalid == "rotation_count":
        rotation = rotation.expand(3, -1, -1)
    elif invalid == "reflection":
        rotation[0, 0] = -1
    elif invalid == "nonorthogonal":
        rotation[0, 1] = .2
    elif invalid == "rotation_nan":
        rotation[0, 0] = float("nan")
    elif invalid == "translation_shape":
        translation = torch.zeros(2, dtype=torch.float64)
    elif invalid == "translation_count":
        translation = torch.zeros(3, 3, dtype=torch.float64)
    elif invalid == "translation_nan":
        translation[0] = float("nan")
    elif invalid == "dtype":
        rotation = rotation.float()
    elif invalid == "bad_frame":
        frames = (*frames[:2], frames[2] * 2, frames[3])
    else:
        beam = "invalid"
    with pytest.raises((TypeError, ValueError, RuntimeError)) as error:
        _transform()(frames, rotation, translation, beam=beam)
    assert not isinstance(error.value, NotImplementedError), "transform validation is still pending"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_angle_generation_and_transform_stay_on_cuda_with_backward():
    angles = torch.tensor([.23, -.41], dtype=torch.float64, device="cuda", requires_grad=True)
    frames = diffct.circular_trajectory_3d(angles, 5., 9., device="cpu", dtype=torch.float32)
    angle = torch.tensor(.31, dtype=torch.float64, device="cuda", requires_grad=True)
    translation = torch.tensor([.1, -.2, .3], dtype=torch.float64, device="cuda", requires_grad=True)
    result = _transform()(frames, _rotation(angle, 3), translation)
    assert all(value.device == angles.device and value.dtype == angles.dtype for value in result)
    gradients = torch.autograd.grad(sum((value * torch.arange(value.numel(), device="cuda").reshape_as(value)).sum() for value in result), (angles, angle, translation))
    assert all(torch.isfinite(gradient).all() and gradient.abs().max() > 1e-4 for gradient in gradients)
