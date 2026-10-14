"""Native surface data transforms preserve the eager single-ray linear map."""

import pytest
import torch

from diffct import ParameterizedSurface, Projector


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.version.hip is not None,
        reason="native surface transforms require NVIDIA CUDA",
    ),
]


def _cancellation_operator(beam, surface_kind, frame_dtype):
    dimensions = 3 if beam == "cone" else 2
    source = torch.zeros((1, dimensions), dtype=frame_dtype)
    center = torch.zeros_like(source)
    center[0, 0] = 1e8
    axis_u = torch.zeros_like(source)
    axis_u[0, 1] = 1.
    trajectory = (source, center, axis_u)
    if beam == "cone":
        axis_v = torch.zeros_like(source)
        axis_v[0, 2] = 1.
        trajectory += (axis_v,)
    calls = []
    displacement = torch.tensor(-1e8 + 1, dtype=torch.float64)

    def offsets(u, v, normal_offset):
        calls.append(1)
        return torch.stack((torch.zeros_like(u), torch.zeros_like(v),
                            torch.zeros_like(u) + normal_offset), dim=-1)

    if surface_kind == "legacy":
        def surface(u, v):
            return offsets(u, v, displacement)
    else:
        def sampler(u, v, view_indices, normal_offset):
            return offsets(u, v, normal_offset)

        surface = ParameterizedSurface(sampler, (displacement,))

    operator = Projector(
        trajectory, (1,) * dimensions, (1,) * (dimensions - 1),
        beam=beam, detector_surface=surface, backend="numba",
    )
    assert operator.backend == "numba" and operator.precision == "fast"
    assert len(calls) == 1
    return operator, calls


def _shapes(operator, operation):
    if operation == "project":
        return operator.volume_shape, operator.projection_shape
    return operator.projection_shape, operator.volume_shape


def _assert_linear_result(actual, data, output_shape):
    assert actual.dtype == torch.float32 and actual.device == data.device
    # FP64 center + normal_offset places the pixel at x=1. The segment [0, 1]
    # intersects the centered unit voxel for .5 units, so A = A.T = [.5].
    expected = (.5 * data).reshape(output_shape)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("surface_kind", ["legacy", "parameterized"])
@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("frame_dtype", [torch.float32, torch.float64], ids=["frames-fp32", "frames-fp64"])
def test_native_surface_jvp_preserves_eager_primal_and_data_tangent(beam, surface_kind, operation, frame_dtype):
    operator, calls = _cancellation_operator(beam, surface_kind, frame_dtype)
    input_shape, output_shape = _shapes(operator, operation)
    function = getattr(operator, operation)
    value = torch.ones(input_shape, dtype=torch.float32, device="cuda")
    tangent = torch.full_like(value, 1.5)
    before = len(calls)
    eager = function(value)
    assert len(calls) == before + 1
    _assert_linear_result(eager, value, output_shape)

    before = len(calls)
    primal, derivative = torch.func.jvp(function, (value,), (tangent,))
    assert len(calls) == before + 1
    _assert_linear_result(torch.stack((primal, derivative)),
                          torch.stack((value, tangent)), (2, *output_shape))


@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("surface_kind", ["legacy", "parameterized"])
@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("frame_dtype", [torch.float32, torch.float64], ids=["frames-fp32", "frames-fp64"])
def test_native_surface_vmap_preserves_eager_data_operator(beam, surface_kind, operation, frame_dtype):
    operator, calls = _cancellation_operator(beam, surface_kind, frame_dtype)
    input_shape, output_shape = _shapes(operator, operation)
    function = getattr(operator, operation)
    value = torch.ones(input_shape, dtype=torch.float32, device="cuda")
    before = len(calls)
    eager = function(value)
    assert len(calls) == before + 1
    _assert_linear_result(eager, value, output_shape)

    batch = torch.stack((value, 2 * value))
    before = len(calls)
    actual = torch.func.vmap(function)(batch)
    assert len(calls) == before + 1
    _assert_linear_result(actual, batch, (2, *output_shape))
