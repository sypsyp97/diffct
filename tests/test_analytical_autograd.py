"""Sinogram derivatives of the discrete analytical gather, independent of Siddon."""

import math

import pytest
import torch

from diffct import (
    angular_integration_weights,
    cone_cosine_weights,
    cone_weighted_backproject,
    fan_cosine_weights,
    fan_weighted_backproject,
    parallel_weighted_backproject,
    ramp_filter_1d,
)


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required",
)]
BEAMS = ("parallel", "fan", "cone")


def _case(beam, scenario="offset"):
    """Fixed CPU geometry; edge cases include exact end bins and outside samples."""
    if scenario == "edge":
        if beam == "parallel":
            trajectory = ([[0.0, 1.0]], [[0.0, 0.0]], [[1.0, 0.0]])
        elif beam == "fan":
            trajectory = ([[0.0, 4.0]], [[0.0, -4.0]], [[1.0, 0.0]])
        else:
            trajectory = (
                [[0.0, 4.0, 0.0]], [[0.0, -4.0, 0.0]],
                [[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]],
            )
        return {
            "trajectory": tuple(torch.tensor(t) for t in trajectory),
            "sino_shape": (1, 5, 3) if beam == "cone" else (1, 5),
            "output_shape": (5, 3, 7) if beam == "cone" else (3, 7),
            "pitch": (1.0, 2.0) if beam == "cone" else (1.0,),
            "voxel_spacing": 1.0,
            "isocenter": torch.zeros(3 if beam == "cone" else 2),
        }

    angles = torch.tensor([0.2, 1.15, 2.55], dtype=torch.float64)
    u = torch.stack((angles.cos(), angles.sin()), dim=1)
    if beam == "parallel":
        ray = torch.stack((-angles.sin(), angles.cos()), dim=1)
        origin = torch.tensor([[0.18, -0.27], [-0.33, 0.12], [0.29, 0.36]])
        trajectory = (ray, origin, u)
        isocenter = torch.zeros(2)
    else:
        isocenter = torch.tensor([0.25, -0.35], dtype=torch.float64)
        normal = torch.stack((-u[:, 1], u[:, 0]), dim=1)
        if beam == "cone":
            isocenter = torch.tensor([0.2, -0.3, 0.15], dtype=torch.float64)
            tilt = torch.tensor([0.12, -0.2, 0.3], dtype=torch.float64)
            u = torch.cat((u, torch.zeros(3, 1)), dim=1)
            v = torch.stack((-angles.sin() * tilt.sin(),
                             angles.cos() * tilt.sin(), tilt.cos()), dim=1)
            normal = torch.linalg.cross(u, v)
        sid = torch.tensor([4.2, 5.1, 4.6], dtype=torch.float64)
        sdd = torch.tensor([8.0, 9.0, 8.5], dtype=torch.float64)
        source = isocenter - normal * sid[:, None]
        detector = isocenter + normal * (sdd - sid)[:, None]
        detector = detector + u * torch.tensor([0.15, -0.2, 0.3])[:, None]
        if beam == "cone":
            detector = detector + v * torch.tensor([-0.2, 0.15, 0.25])[:, None]
        trajectory = (source, detector, u, v) if beam == "cone" else (source, detector, u)
    return {
        "trajectory": tuple(t.float() for t in trajectory),
        "sino_shape": (3, 5, 4) if beam == "cone" else (3, 6),
        "output_shape": (2, 3, 4) if beam == "cone" else (3, 5),
        "pitch": (0.8, 0.65) if beam == "cone" else (0.7,),
        "voxel_spacing": 0.6,
        "isocenter": isocenter.float(),
    }


def _backproject(beam, sinogram, case):
    trajectory, shape = case["trajectory"], case["output_shape"]
    pitch, spacing = case["pitch"], case["voxel_spacing"]
    if beam == "parallel":
        return parallel_weighted_backproject(sinogram, *trajectory, *pitch, *shape, spacing)
    if beam == "fan":
        return fan_weighted_backproject(sinogram, *trajectory, *pitch, *shape, spacing,
                                       isocenter=case["isocenter"])
    return cone_weighted_backproject(sinogram, *trajectory, *shape, *pitch, spacing,
                                    isocenter=case["isocenter"])


def _reference(beam, sinogram, case):
    """Float64 PyTorch reference for physical voxel centres and detector gathers."""
    axes = [torch.arange(n, dtype=torch.float64) + 0.5 - n / 2
            for n in case["output_shape"]]
    grids = torch.meshgrid(*axes, indexing="ij")
    points = torch.stack(tuple(reversed(grids)), dim=-1).reshape(-1, len(axes))
    points = points * case["voxel_spacing"]
    trajectory = tuple(t.detach().cpu().double() for t in case["trajectory"])
    if beam == "parallel":
        _, detector, u = trajectory
        detector_points = points[None, :, :].expand(sinogram.shape[0], -1, -1)
        weight = torch.ones(detector_points.shape[:2], dtype=torch.float64)
        scale = 1 / (2 * math.pi)
    else:
        source, detector, u = trajectory[:3]
        normal = (torch.stack((-u[:, 1], u[:, 0]), dim=1) if beam == "fan"
                  else torch.linalg.cross(u, trajectory[3]))
        normal = normal / torch.linalg.vector_norm(normal, dim=1, keepdim=True)
        align = ((detector - source) * normal).sum(dim=1)
        normal = torch.where(align[:, None] < 0, -normal, normal)
        sid = ((case["isocenter"].double() - source) * normal).sum(dim=1)
        sdd = ((detector - source) * normal).sum(dim=1)
        relative = points[None, :, :] - source[:, None, :]
        distance = (relative * normal[:, None, :]).sum(dim=2)
        valid_distance = (distance > 1e-6) & (sid[:, None] > 1e-6) & (sdd[:, None] > 1e-6)
        safe_distance = torch.where(valid_distance, distance, torch.ones_like(distance))
        detector_points = source[:, None, :] + relative * (sdd[:, None] / safe_distance)[:, :, None]
        weight = torch.where(valid_distance, (sid[:, None] / safe_distance).square(), 0)
        scale = (sdd / (2 * math.pi * sid))[:, None]

    relative = detector_points - detector[:, None, :]
    fu = (relative * u[:, None, :]).sum(dim=2) / case["pitch"][0]
    fu = fu + (sinogram.shape[1] - 1) / 2
    valid = (fu >= 0) & (fu <= sinogram.shape[1] - 1)
    iu = fu.floor().long().clamp(0, sinogram.shape[1] - 2)
    tu = (fu - iu).clamp(0, 1)
    views = torch.arange(sinogram.shape[0])[:, None]
    if beam != "cone":
        sample = sinogram[views, iu] * (1 - tu) + sinogram[views, iu + 1] * tu
    else:
        v = trajectory[3]
        fv = (relative * v[:, None, :]).sum(dim=2) / case["pitch"][1]
        fv = fv + (sinogram.shape[2] - 1) / 2
        valid = valid & (fv >= 0) & (fv <= sinogram.shape[2] - 1)
        iv = fv.floor().long().clamp(0, sinogram.shape[2] - 2)
        tv = (fv - iv).clamp(0, 1)
        sample = (sinogram[views, iu, iv] * (1 - tu) * (1 - tv)
                  + sinogram[views, iu + 1, iv] * tu * (1 - tv)
                  + sinogram[views, iu, iv + 1] * (1 - tu) * tv
                  + sinogram[views, iu + 1, iv + 1] * tu * tv)
    return (torch.where(valid, weight * sample, 0) * scale).sum(dim=0).reshape(case["output_shape"])


def _random(shape, seed, *, device="cuda", dtype=torch.float64, noncontiguous=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    if noncontiguous:
        return torch.randn((*shape, 2), generator=generator, device=device, dtype=dtype)[..., 0]
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


def _production_matrix(beam, case):
    """Columns of the actual float32 CUDA gather, including its wrapper scaling."""
    basis = torch.eye(math.prod(case["sino_shape"]), device="cuda")
    with torch.no_grad():
        return torch.stack([
            _backproject(beam, column.reshape(case["sino_shape"]), case).reshape(-1)
            for column in basis
        ], dim=1).double()


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("scenario", ("edge", "offset"))
def test_analytical_gather_matches_independent_reference(beam, scenario):
    case = _case(beam, scenario)
    sinogram = _random(case["sino_shape"], 100, noncontiguous=True)
    actual = _backproject(beam, sinogram, case)
    expected = _reference(beam, sinogram.cpu(), case)
    assert not sinogram.is_contiguous()
    assert actual.dtype == torch.float32 and actual.shape == case["output_shape"]
    assert actual.device == sinogram.device
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=2e-5, atol=2e-6)
    if scenario == "edge":
        assert (expected == 0).any() and (expected != 0).any()


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("scenario", ("edge", "offset"))
def test_sinogram_gradient_is_transpose_of_actual_gather(beam, scenario):
    case = _case(beam, scenario)
    sinogram = _random(case["sino_shape"], 101, noncontiguous=True).requires_grad_()
    output = _backproject(beam, sinogram, case)
    cotangent = _random(case["output_shape"], 102, dtype=torch.float32)
    matrix = _production_matrix(beam, case)
    expected = (matrix.T @ cotangent.double().reshape(-1)).reshape(sinogram.shape)
    reference_input = sinogram.detach().cpu().requires_grad_()
    reference_output = _reference(beam, reference_input, case)
    reference_grad, = torch.autograd.grad(reference_output, reference_input, cotangent.cpu().double())
    torch.testing.assert_close(expected.cpu(), reference_grad, rtol=2e-5, atol=2e-6)
    actual, = torch.autograd.grad(output, sinogram, cotangent)
    assert actual.dtype == sinogram.dtype and actual.device == sinogram.device
    assert not sinogram.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("beam", BEAMS)
def test_transpose_supports_derivative_with_respect_to_cotangent(beam):
    case = _case(beam)
    sinogram = _random(case["sino_shape"], 103).requires_grad_()
    cotangent = _random(case["output_shape"], 104, dtype=torch.float32).requires_grad_()
    output = _backproject(beam, sinogram, case)
    gradient, = torch.autograd.grad(output, sinogram, cotangent, create_graph=True)
    probe = _random(case["sino_shape"], 105)
    actual, = torch.autograd.grad(gradient, cotangent, probe)
    expected = (_production_matrix(beam, case) @ probe.reshape(-1)).reshape(cotangent.shape)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("beam", BEAMS)
def test_quadratic_reconstruction_loss_has_correct_hessian_vector_product(beam):
    case = _case(beam)
    sinogram = _random(case["sino_shape"], 106).requires_grad_()
    direction = _random(case["sino_shape"], 107)
    output = _backproject(beam, sinogram, case)
    gradient, = torch.autograd.grad(output.square().sum() / 2, sinogram, create_graph=True)
    actual, = torch.autograd.grad(gradient, sinogram, direction)
    matrix = _production_matrix(beam, case)
    expected = (matrix.T @ (matrix @ direction.reshape(-1))).reshape(sinogram.shape)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


def _preprocess(beam, sinogram, detector_weights, view_weights, case, use_rfft):
    n_views, n_det = case["sino_shape"][:2]
    device, dtype = sinogram.device, sinogram.dtype
    if beam == "parallel":
        cosine = torch.ones(n_det, device=device, dtype=dtype)
    elif beam == "fan":
        cosine = fan_cosine_weights(n_det, *case["pitch"], 8.0, device=device, dtype=dtype)
    else:
        cosine = cone_cosine_weights(*case["sino_shape"][1:], *case["pitch"], 8.0,
                                     device=device, dtype=dtype)
    broadcast_shape = (n_views,) + (1,) * (sinogram.ndim - 1)
    weighted = sinogram * cosine * detector_weights * view_weights.reshape(broadcast_shape)
    filtered = ramp_filter_1d(weighted, dim=1, sample_spacing=case["pitch"][0],
                              pad_factor=2, window="hann", use_rfft=use_rfft)
    angles = torch.tensor([0.2, 1.15, 2.55], device=device)
    return filtered * angular_integration_weights(angles).reshape(broadcast_shape)


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("use_rfft", (False, True))
def test_fft_native_and_learned_preweights_retain_gradients(beam, use_rfft):
    case = _case(beam)
    inputs = (
        _random(case["sino_shape"], 108).requires_grad_(),
        (_random(case["sino_shape"][1:], 109) * 0.1 + 1).requires_grad_(),
        (_random((case["sino_shape"][0],), 110) * 0.1 + 1).requires_grad_(),
    )
    filtered = _preprocess(beam, *inputs, case, use_rfft)
    output = _backproject(beam, filtered, case)
    cotangent = _random(case["output_shape"], 111, dtype=torch.float32)
    reference_inputs = tuple(t.detach().cpu().requires_grad_() for t in inputs)
    reference_filtered = _preprocess(beam, *reference_inputs, case, use_rfft)
    expected = _reference(beam, reference_filtered, case)
    reference_gradients = torch.autograd.grad(expected, reference_inputs, cotangent.cpu().double())
    actual_gradients = torch.autograd.grad(output, inputs, cotangent)
    for actual, reference in zip(actual_gradients, reference_gradients):
        assert actual.dtype == torch.float64 and torch.isfinite(actual).all()
        assert reference.abs().max() > 0
        torch.testing.assert_close(actual.cpu(), reference, rtol=3e-5, atol=3e-6)


GEOMETRY_COMPONENTS = [(beam, component) for beam in BEAMS
                       for component in range(4 if beam == "cone" else 3)]


@pytest.mark.parametrize(("beam", "component"), GEOMETRY_COMPONENTS)
@pytest.mark.parametrize("geometry_device", ("cpu", "cuda"))
def test_geometry_requiring_gradients_is_rejected_explicitly(beam, component, geometry_device):
    case = _case(beam)
    trajectory = [t.to(geometry_device).clone() for t in case["trajectory"]]
    trajectory[component].requires_grad_()
    case["trajectory"] = tuple(trajectory)
    sinogram = _random(case["sino_shape"], 112).requires_grad_()
    with pytest.raises((ValueError, RuntimeError, NotImplementedError),
                       match=r"(?i)(gradient|requires_grad|differentiab)"):
        _backproject(beam, sinogram, case)


@pytest.mark.parametrize("beam", ("fan", "cone"))
@pytest.mark.parametrize("geometry_device", ("cpu", "cuda"))
def test_isocenter_requiring_gradients_is_rejected_explicitly(beam, geometry_device):
    case = _case(beam)
    case["isocenter"] = case["isocenter"].to(geometry_device).requires_grad_()
    with pytest.raises((ValueError, RuntimeError, NotImplementedError),
                       match=r"(?i)(gradient|requires_grad|differentiab)"):
        _backproject(beam, _random(case["sino_shape"], 113), case)


@pytest.mark.parametrize("beam", BEAMS)
def test_cpu_sinogram_is_rejected(beam):
    case = _case(beam)
    with pytest.raises(ValueError, match=r"(?i)cuda"):
        _backproject(beam, _random(case["sino_shape"], 114, device="cpu"), case)


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("malformed", ("sinogram_rank", "view_count", "coordinate_dimension"))
def test_malformed_analytical_shapes_are_rejected(beam, malformed):
    case = _case(beam, "edge")
    sinogram = _random(case["sino_shape"], 115)
    if malformed == "sinogram_rank":
        sinogram = sinogram.unsqueeze(-1)
    elif malformed == "view_count":
        # Extra geometry rows are safe for the old kernel to read, but invalid.
        case["trajectory"] = tuple(t.repeat(2, 1) for t in case["trajectory"])
    else:
        # Extra coordinates likewise avoid out-of-bounds CUDA reads in RED.
        case["trajectory"] = tuple(torch.cat((t, torch.zeros(t.shape[0], 1)), dim=1)
                                   for t in case["trajectory"])
    with pytest.raises((ValueError, RuntimeError)):
        _backproject(beam, sinogram, case)
