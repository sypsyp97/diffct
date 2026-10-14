"""Finite portable inputs must survive zero dependencies and BF16 metadata."""

import math

import pytest
import torch

from diffct import Projector, TensorStore, fbp, fbp_into, fdk, fdk_into
from diffct.analytical_execution import AnalyticalExecution
from diffct.distribution import Partition


def _parallel(views, *, missed=False):
    return tuple(torch.tensor([vector] * views, dtype=torch.float64)
                 for vector in ((1.0, 0.0), (0.0, 100.0 if missed else 0.0), (0.0, 1.0)))


def _scan(beam, angles):
    angles = angles.float()
    radial = torch.stack((angles.cos(), angles.sin()), dim=-1)
    tangent = torch.stack((-angles.sin(), angles.cos()), dim=-1)
    if beam == "parallel":
        return radial, torch.zeros_like(radial), tangent
    radial = torch.cat((radial, torch.zeros(len(angles), 1)), dim=-1)
    tangent = torch.cat((tangent, torch.zeros(len(angles), 1)), dim=-1)
    vertical = torch.tensor([[0.0, 0.0, 1.0]] * len(angles))
    return -10 * radial, 10 * radial, tangent, vertical


def test_fp16_projection_keeps_finite_chord_gradient_and_jvp():
    operator = Projector(_parallel(1), (256, 256), (1,), beam="parallel",
                         backend="torch", precision="float32", distributed=False)
    image = torch.ones(operator.volume_shape, dtype=torch.float16, requires_grad=True)
    assert torch.isfinite(image).all() and torch.isinf(image.sum())
    expected = torch.tensor([[256.0]])
    actual = operator.project(image)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient, = torch.autograd.grad(actual.sum(), image)
    expected_gradient = torch.zeros_like(image)
    # y=0 is the lower face of row 128 on this even, half-open voxel grid.
    expected_gradient[128] = 1
    torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
    primal, tangent = torch.func.jvp(operator.project, (image.detach(),),
                                    (torch.ones_like(image),))
    torch.testing.assert_close(primal, expected, rtol=0, atol=0)
    torch.testing.assert_close(tangent, expected, rtol=0, atol=0)


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_missed_rays_keep_zero_gradient_when_finite_input_sum_overflows(precision, operation):
    dtype = torch.float64 if precision == "float64" else torch.float32
    views = 1 if operation == "project" else 64
    operator = Projector(_parallel(views, missed=True), (8, 8), (1,), beam="parallel",
                         backend="torch", precision=precision, distributed=False)
    shape = operator.volume_shape if operation == "project" else operator.projection_shape
    value = torch.full(shape, torch.finfo(dtype).max / 16, dtype=dtype, requires_grad=True)
    assert torch.isfinite(value).all() and torch.isinf(value.sum())
    function = getattr(operator, operation)
    actual = function(value)
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)
    gradient, = torch.autograd.grad(actual.sum(), value)
    torch.testing.assert_close(gradient, torch.zeros_like(value), rtol=0, atol=0)
    primal, tangent = torch.func.jvp(function, (value.detach(),), (torch.ones_like(value),))
    torch.testing.assert_close(primal, torch.zeros_like(primal), rtol=0, atol=0)
    torch.testing.assert_close(tangent, torch.zeros_like(tangent), rtol=0, atol=0)


@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_empty_projector_batch_keeps_connected_zero_gradient(operation):
    operator = Projector(_parallel(1), (3, 5), (1,), beam="parallel",
                         backend="torch", precision="float32", distributed=False)
    shape = operator.volume_shape if operation == "project" else operator.projection_shape
    value = torch.empty((0, *shape), dtype=torch.float16, requires_grad=True)
    actual = getattr(operator, operation)(value)
    assert actual.shape[0] == actual.numel() == 0
    assert actual.dtype == torch.float32 and actual.requires_grad
    gradient, = torch.autograd.grad(actual.sum(), value)
    assert gradient.shape == value.shape and gradient.numel() == 0
    assert gradient.dtype == value.dtype


def test_fp16_fbp_matches_same_quantized_fp32_data_and_gradient():
    angles = torch.linspace(0, math.pi, 257)[:-1]
    geometry = _scan("parallel", angles)
    source = torch.ones((256, 512), dtype=torch.float16, requires_grad=True)
    reference_source = source.detach().float().requires_grad_()
    options = dict(backend="torch", precision="float32", view_chunk_size=32,
                   volume_chunk_shape=(3, 5), distributed=False)
    reference = fbp(reference_source, geometry, (3, 5), angles, **options)
    assert torch.isfinite(source).all() and torch.isinf(source.sum())
    assert torch.isfinite(reference).all() and reference.abs().max() > 0
    actual = fbp(source, geometry, (3, 5), angles, **options)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    actual_gradient, = torch.autograd.grad(actual.sum(), source)
    reference_gradient, = torch.autograd.grad(reference.sum(), reference_source)
    expected_gradient = reference_gradient.to(source.dtype)
    assert expected_gradient.abs().max() > 0
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-3, atol=1e-6)


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_zero_angular_weights_keep_finite_fbp_with_overflowing_source_sum(precision):
    dtype = torch.float64 if precision == "float64" else torch.float32
    source = torch.full((64, 8), torch.finfo(dtype).max / 256,
                        dtype=dtype, requires_grad=True)
    angular = torch.zeros(64, dtype=dtype, requires_grad=True)
    angles = torch.linspace(0, math.pi, 65, dtype=dtype)[:-1]
    assert torch.isfinite(source).all() and torch.isinf(source.sum())
    actual = fbp(source, _scan("parallel", angles), (1, 1), angles,
                 angular_weights=angular, backend="torch", precision=precision,
                 view_chunk_size=8, distributed=False)
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)
    source_gradient, angular_gradient = torch.autograd.grad(actual.sum(), (source, angular))
    torch.testing.assert_close(source_gradient, torch.zeros_like(source), rtol=0, atol=0)
    # The zero angular leaf remains learnable: finite data gives nonzero
    # derivatives with respect to replacing those zero quadrature weights.
    assert torch.isfinite(angular_gradient).all() and angular_gradient.abs().max() > 0


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("option", ["projection_weights", "angular_weights", "filter_response"])
def test_zero_data_with_large_learned_weights_keeps_source_and_option_gradients(precision, option):
    dtype = torch.float64 if precision == "float64" else torch.float32
    source = torch.zeros((4, 8), dtype=dtype, requires_grad=True)
    shapes = {"projection_weights": source.shape, "angular_weights": (4,), "filter_response": (9,)}
    learned = torch.full(shapes[option], torch.finfo(dtype).max / 2,
                         dtype=dtype, requires_grad=True)
    angles = torch.linspace(0, math.pi, 5, dtype=dtype)[:-1]
    assert torch.isfinite(learned).all() and torch.isinf(learned.sum())
    options = {"angular_weights": torch.full((4,), .25, dtype=dtype), option: learned}
    actual = fbp(source, _scan("parallel", angles), (1, 1), angles,
                 backend="torch", precision=precision, distributed=False, **options)
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)
    source_gradient, option_gradient = torch.autograd.grad(actual.sum(), (source, learned))
    torch.testing.assert_close(option_gradient, torch.zeros_like(learned), rtol=0, atol=0)
    assert torch.isfinite(source_gradient).all() and source_gradient.abs().max() > 0


@pytest.mark.parametrize("beam", ["parallel", "cone"])
@pytest.mark.parametrize("streamed", [False, True], ids=["tensor", "into"])
def test_bf16_angles_match_their_fp32_quantized_baseline(beam, streamed):
    angles = torch.linspace(0, math.pi, 6)[:-1].to(torch.bfloat16)
    geometry = _scan(beam, angles)
    shape = (2, 3, 5) if beam == "cone" else (3, 5)
    source_shape = (5, 9, 4) if beam == "cone" else (5, 9)
    source = (torch.arange(math.prod(source_shape)).float().sin() + .3).reshape(source_shape)
    reconstruct = fdk if beam == "cone" else fbp
    reconstruct_into = fdk_into if beam == "cone" else fbp_into
    options = dict(backend="torch", precision="float32", view_chunk_size=2,
                   volume_chunk_shape=(1,) * len(shape), distributed=False)
    if beam == "cone":
        options["isocenter"] = (0.0, 0.0, 0.0)
    reference = reconstruct(source, geometry, shape, angles.float(), **options)
    assert torch.isfinite(reference).all() and reference.abs().max() > 0
    if streamed:
        actual = torch.full_like(reference, torch.nan)
        sink = TensorStore(actual)
        assert reconstruct_into(TensorStore(source), sink, geometry, shape, angles, **options) is sink
    else:
        actual = reconstruct(source, geometry, shape, angles, **options)
    assert actual.device.type == "cpu" and actual.dtype == torch.float32
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_acquisition_metadata_distinguishes_angle_dtype_and_values(monkeypatch):
    metadata = []
    original = Partition.preflight

    def capture(partition, error=None, request=None):
        original(partition, error, request)
        if request is not None:
            metadata.append(request)

    monkeypatch.setattr(Partition, "preflight", capture)
    source = torch.zeros((4, 8))
    geometry = _parallel(4)
    angle_vectors = (torch.zeros(4, dtype=torch.bfloat16), torch.zeros(4, dtype=torch.float16),
                     torch.tensor([0, 0, 0, .25], dtype=torch.bfloat16))
    for angles in angle_vectors:
        AnalyticalExecution(source, geometry, (1, 1), angles, beam="parallel",
                            angular_weights=torch.ones(4), backend="torch", precision="float32")
    # The first two arrays have identical raw zero bytes but different dtypes.
    # Numerical equality must not erase the collective acquisition contract.
    assert len(metadata) == 3 and len(set(metadata)) == 3
