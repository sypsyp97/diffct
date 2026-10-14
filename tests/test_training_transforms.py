"""Default native data transforms and empty batches for training workloads."""

import math

import pytest
import torch

from diffct import Projector
from tests.test_reference_backend import BEAMS, CUDA_REQUIRED, _case, _data, _matrix


def _operator(case, backend):
    # Omit precision so these checks exercise the public default fast mode.
    options = {} if backend == "auto" else {"backend": backend}
    return Projector(
        case.trajectory, case.shape, case.detector, beam=case.beam,
        detector_spacing=case.pitch if case.beam == "cone" else case.pitch[0],
        voxel_spacing=case.spacing, **options,
    )


def _close(actual, expected, device):
    assert actual.dtype == torch.float32
    assert actual.device == device
    torch.testing.assert_close(actual.detach().cpu().double(), expected.double(),
                               rtol=4e-5, atol=4e-5)


@pytest.mark.cuda
@CUDA_REQUIRED
@pytest.mark.parametrize("backend", ["auto", "numba"])
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_default_native_data_transforms_match_independent_linear_map(backend, beam, operation):
    case = _case(beam, scalar=True)
    operator = _operator(case, backend)
    assert operator.backend == "numba" and operator.precision == "fast"
    assert operator.dtype == torch.float32
    matrix = _matrix(case)
    input_shape, output_shape = case.shape, case.projection_shape
    if operation == "backproject":
        matrix = matrix.T
        input_shape, output_shape = output_shape, input_shape
    function = getattr(operator, operation)
    value = _data(input_shape, dtype=torch.float32, device="cuda")
    tangent = _data(input_shape, dtype=torch.float32, device="cuda", phase=0.79)
    weight = _data(output_shape, dtype=torch.float32, device="cuda", phase=0.31)

    def linear_loss(x):
        return (function(x) * weight).sum()

    _close(torch.func.grad(linear_loss)(value),
           (matrix.T @ weight.cpu().double().flatten()).reshape(input_shape), value.device)
    primal, derivative = torch.func.jvp(function, (value,), (tangent,))
    _close(primal, (matrix @ value.cpu().double().flatten()).reshape(output_shape), value.device)
    _close(derivative, (matrix @ tangent.cpu().double().flatten()).reshape(output_shape), value.device)
    jacobian = matrix.reshape(*output_shape, *input_shape)
    _close(torch.func.jacrev(function)(value), jacobian, value.device)
    _close(torch.func.jacfwd(function)(value), jacobian, value.device)
    batches = _data((2, *input_shape), dtype=torch.float32, device="cuda", phase=0.43)
    expected_batches = (batches.cpu().double().reshape(2, -1) @ matrix.T).reshape(2, *output_shape)
    _close(torch.func.vmap(function)(batches), expected_batches, value.device)

    def quadratic_loss(x):
        return function(x).square().sum() / 2

    gram = matrix.T @ matrix
    _close(torch.func.jacfwd(torch.func.grad(quadratic_loss))(value),
           gram.reshape(*input_shape, *input_shape), value.device)
    leaf = value.clone().requires_grad_()
    first, = torch.autograd.grad(quadratic_loss(leaf), leaf, create_graph=True)
    second, = torch.autograd.grad((first * tangent).sum(), leaf)
    _close(second, (gram @ tangent.cpu().double().flatten()).reshape(input_shape), value.device)


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("device", [
    "cpu", pytest.param("cuda", marks=[pytest.mark.cuda, CUDA_REQUIRED]),
])
def test_empty_batch_and_channel_dimensions_preserve_output_contract(beam, device):
    case = _case(beam, scalar=True)
    backends = ("torch",) if device == "cpu" else ("numba", "auto", "torch")
    for backend in backends:
        operator = _operator(case, backend)
        assert operator.precision == "fast" and operator.dtype == torch.float32
        for leading in ((0,), (2, 0)):
            for operation, input_shape, output_shape in (
                ("project", case.shape, case.projection_shape),
                ("backproject", case.projection_shape, case.shape),
            ):
                value = torch.empty((*leading, *input_shape), dtype=torch.float64, device=device)
                actual = getattr(operator, operation)(value)
                assert actual.shape == (*leading, *output_shape)
                assert actual.dtype == torch.float32 and actual.device == value.device
                assert actual.numel() == math.prod(leading) == 0
