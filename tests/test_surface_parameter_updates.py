"""Validate current sampled pixels before execution after parameter updates."""

import math

import pytest
import torch

from diffct import ParameterizedSurface, Projector


OPERATIONS = ("project", "backproject", "project_into", "backproject_into")
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
NATIVE = pytest.param(
    "numba",
    marks=pytest.mark.skipif(
        torch.version.hip is not None, reason="native execution requires NVIDIA CUDA",
    ),
)


def _frames(beam, *, views=1):
    dimensions = 3 if beam == "cone" else 2
    source = torch.zeros((views, dimensions), dtype=torch.float64)
    center, axis_u = torch.zeros_like(source), torch.zeros_like(source)
    source[:, 0], center[:, 0], axis_u[:, 1] = -2.0, 2.0, 1.0
    trajectory = (source, center, axis_u)
    if beam == "cone":
        axis_v = torch.zeros_like(source)
        axis_v[:, 2] = 1.0
        trajectory += (axis_v,)
    return trajectory


def _input_and_sink(operator, operation, device="cpu"):
    projecting = operation.startswith("project")
    source_shape = operator.volume_shape if projecting else operator.projection_shape
    output_shape = operator.projection_shape if projecting else operator.volume_shape
    value = torch.ones(source_shape, dtype=operator.dtype, device=device)
    sink = (torch.full(output_shape, 73.0, dtype=operator.dtype, device=device)
            if operation.endswith("_into") else None)
    return value, sink


def _call(operator, operation, value, sink, **kwargs):
    function = getattr(operator, operation)
    return function(value, sink, **kwargs) if sink is not None else function(value, **kwargs)


def _assert_rejected(operator, operation, *, device="cpu", **kwargs):
    value, sink = _input_and_sink(operator, operation, device)
    # Check the sink even when the expected exception is missing. A late
    # invalid pixel must not leave earlier valid chunks in caller storage.
    try:
        with torch.no_grad(), pytest.raises(ValueError):
            _call(operator, operation, value, sink, **kwargs)
    finally:
        if sink is not None:
            torch.testing.assert_close(sink, torch.full_like(sink, 73.0), rtol=0, atol=0)


def _late_pixel_case(beam, *, kind="coincident", precision="float64", backend="torch"):
    trajectory = _frames(beam, views=2)
    dimensions = len(trajectory[0][0])
    parameter = torch.tensor(
        [1.0, 1.0 if beam == "cone" else 0.0, 1.0 if kind in ("nan", "inf") else 0.0],
        dtype=torch.float64,
    )

    def sampler(u, v, ids, offsets):
        local = torch.stack((u, v, torch.zeros_like(u)), dim=-1)
        last_pixel = (u == 1) & ((v == 1) if beam == "cone" else (v == 0))
        last_view = (ids == 1).reshape(-1, *([1] * u.ndim))
        depth = offsets[2].log() if kind in ("nan", "inf") else offsets[2]
        replacement = torch.stack((offsets[0], offsets[1], depth))
        return torch.where((last_view & last_pixel)[..., None], replacement, local)

    operator = Projector(
        trajectory, (3,) * dimensions, (3,) * (dimensions - 1), beam=beam,
        detector_surface=ParameterizedSurface(sampler, (parameter,)),
        backend=backend, precision=precision, view_chunk_size=1,
        detector_chunk_shape=(1,) * (dimensions - 1),
    )
    return operator, parameter, trajectory


def _make_invalid(parameter, kind):
    with torch.no_grad():
        if kind == "coincident":
            parameter.copy_(parameter.new_tensor([0.0, 0.0, -4.0]))
        elif kind in ("nan", "inf"):
            # Keep parameters finite; the sampled logarithm becomes NaN/-inf.
            parameter[2] = -1.0 if kind == "nan" else 0.0
        else:
            parameter[1] = 1e-12


@pytest.mark.parametrize("beam", ("fan", "cone"))
@pytest.mark.parametrize("precision", ("float32", "float64"))
@pytest.mark.parametrize("operation", OPERATIONS)
def test_updated_coincident_pixel_rejected_on_every_cpu_entry_point(beam, precision, operation):
    operator, parameter, trajectory = _late_pixel_case(beam, precision=precision)
    value, sink = _input_and_sink(operator, operation)
    with torch.no_grad():
        valid = _call(operator, operation, value, sink)
    assert torch.isfinite(valid).all() and valid.abs().sum() > 0
    _make_invalid(parameter, "coincident")

    # The identical original frames already reject through explicit admission.
    # Omitting the override must impose the same sampled-pixel contract.
    _assert_rejected(operator, operation, trajectory=trajectory)
    _assert_rejected(operator, operation)


@pytest.mark.parametrize("kind", ("nan", "inf", "middle"))
@pytest.mark.parametrize("operation", OPERATIONS)
def test_updated_invalid_offsets_rejected_before_any_cpu_output_write(kind, operation):
    operator, parameter, trajectory = _late_pixel_case("fan", kind=kind)
    _make_invalid(parameter, kind)
    assert torch.isfinite(parameter).all()
    _assert_rejected(operator, operation, trajectory=trajectory)
    _assert_rejected(operator, operation)


def _arithmetic_case(beam, kind):
    trajectory = _frames(beam)
    source, center, axis_u = trajectory[:3]
    source.zero_()
    center.zero_()
    if kind == "cancellation":
        center[0, 0] = 1e8
        parameter = torch.tensor([0.0, 0.0, -1e8 + 16], dtype=torch.float64)
        update = parameter.new_tensor([0.0, 0.0, -1e8 + 1])
        spacing = 1.0
        assert center[0, 0] + update[2] == 1
        assert center[0, 0].float() + update[2].float() == 0
    else:
        diagonal = math.sqrt(.5)
        center[0, :2] = center.new_tensor([3e38, -1e38])
        axis_u[0, :2] = axis_u.new_tensor([diagonal, diagonal])
        parameter = torch.zeros(3, dtype=torch.float64)
        update = parameter.new_tensor([3e38, 0.0, -3e38])
        spacing = 1e38
        # Completed FP64 points fit FP32, but center + u * axis_u overflows
        # before the normal displacement cancels that intermediate sum.
        completed = torch.tensor([3e38, (6 * diagonal - 1) * 1e38], dtype=torch.float64)
        assert torch.isfinite(completed.float()).all()
        assert not torch.isfinite(center[0].float() + update[0].float() * axis_u[0].float()).all()

    def sampler(u, v, ids, offsets):
        return torch.zeros((*u.shape, 3), dtype=u.dtype) + offsets

    dimensions = source.shape[-1]
    operator = Projector(
        trajectory, (1,) * dimensions, (1,) * (dimensions - 1),
        beam=beam, detector_surface=ParameterizedSurface(sampler, (parameter,)),
        backend="torch", precision="float32", voxel_spacing=spacing,
    )
    parameter.copy_(update)
    return operator, trajectory


@pytest.mark.parametrize("beam", ("fan", "cone"))
@pytest.mark.parametrize("kind", ("cancellation", "overflow"))
@pytest.mark.parametrize("operation", OPERATIONS)
def test_updated_surface_uses_execution_precision_for_validation(beam, kind, operation):
    operator, trajectory = _arithmetic_case(beam, kind)
    _assert_rejected(operator, operation, trajectory=trajectory)
    _assert_rejected(operator, operation)


@pytest.mark.cuda
@CUDA
@pytest.mark.parametrize("backend", ("torch", NATIVE))
@pytest.mark.parametrize("beam,operation", (
    ("fan", "project"), ("fan", "project_into"),
    ("cone", "backproject"), ("cone", "backproject_into"),
))
def test_updated_late_pixel_rejected_on_cuda(backend, beam, operation):
    operator, parameter, trajectory = _late_pixel_case(
        beam, backend=backend, precision="fast" if backend == "numba" else "float32",
    )
    value, sink = _input_and_sink(operator, operation, "cuda")
    with torch.no_grad():
        valid = _call(operator, operation, value, sink)
    assert torch.isfinite(valid).all() and valid.abs().sum() > 0
    _make_invalid(parameter, "coincident")
    _assert_rejected(operator, operation, device="cuda", trajectory=trajectory)
    _assert_rejected(operator, operation, device="cuda")


@pytest.mark.cuda
@CUDA
@pytest.mark.skipif(torch.version.hip is not None, reason="native execution requires NVIDIA CUDA")
@pytest.mark.parametrize("operation", ("project", "backproject"))
@pytest.mark.parametrize("transform", ("jvp", "vmap"))
def test_native_data_transform_rejects_updated_invalid_surface(operation, transform):
    operator, parameter, _ = _late_pixel_case("fan", backend="numba", precision="fast")
    value, _ = _input_and_sink(operator, operation, "cuda")
    function = getattr(operator, operation)
    _make_invalid(parameter, "coincident")
    with pytest.raises(ValueError):
        if transform == "jvp":
            torch.func.jvp(function, (value,), (torch.ones_like(value),))
        else:
            torch.func.vmap(function)(torch.stack((value, 2 * value)))


@pytest.mark.parametrize("beam", ("fan", "cone"))
@pytest.mark.parametrize("precision", ("float32", "float64"))
@pytest.mark.parametrize("operation", ("project", "backproject"))
def test_valid_updates_preserve_numeric_changes_and_outstanding_parameter_gradients(
    beam, precision, operation,
):
    trajectory = _frames(beam)
    dimensions = trajectory[0].shape[-1]
    parameter = torch.tensor(.4, dtype=torch.float64, requires_grad=True)

    def sampler(u, v, ids, tangent):
        return torch.stack((torch.zeros_like(u) + tangent, v, torch.zeros_like(u)), dim=-1)

    operator = Projector(
        trajectory, (1,) * dimensions, (1,) * (dimensions - 1), beam=beam,
        detector_surface=ParameterizedSurface(sampler, (parameter,)),
        backend="torch", precision=precision,
    )
    value, _ = _input_and_sink(operator, operation)
    value = (1.25 * value).requires_grad_()
    old = getattr(operator, operation)(value)
    with torch.no_grad():
        parameter.fill_(.6)
    new = getattr(operator, operation)(value)
    assert not torch.equal(old, new)

    # Each segment crosses the complete unit voxel in x and stays inside its
    # other faces. Its length is sqrt(1 + (tangent / 4)^2).
    tolerance = 2e-6 if precision == "float32" else 2e-12
    for result, tangent in ((old, .4), (new, .6)):
        length = math.sqrt(1 + (tangent / 4) ** 2)
        torch.testing.assert_close(result, torch.full_like(result, 1.25 * length),
                                   rtol=tolerance, atol=tolerance)
        data_gradient, surface_gradient = torch.autograd.grad(result.sum(), (value, parameter))
        torch.testing.assert_close(data_gradient, torch.full_like(value, length),
                                   rtol=tolerance, atol=tolerance)
        expected = parameter.new_tensor(1.25 * tangent / (16 * length))
        torch.testing.assert_close(surface_gradient, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("beam", ("fan", "cone"))
def test_live_surface_range_limit_uses_original_coordinates_before_float32_rounding(beam):
    trajectory = _frames(beam)
    trajectory[1].requires_grad_()
    dimensions = trajectory[0].shape[-1]

    def sampler(u, v, ids):
        return torch.stack((u, v, torch.zeros_like(u)), dim=-1)

    operator = Projector(
        trajectory, (1,) * dimensions, (1,) * (dimensions - 1), beam=beam,
        detector_surface=ParameterizedSurface(sampler),
        backend="torch", precision="float32", distributed=False,
    )
    with torch.no_grad():
        trajectory[0][0, 0] = -1000000.01
        trajectory[1][0, 0] = 1000000.01
    # Both raw endpoints exceed the 1e6-voxel bound, but FP32 rounds them
    # onto the boundary. The updated default must retain raw admission.
    for endpoint in trajectory[:2]:
        assert torch.linalg.vector_norm(endpoint).item() > 1e6
        assert torch.linalg.vector_norm(endpoint.float()).item() == 1e6
    _assert_rejected(operator, "project_into", trajectory=trajectory)
    _assert_rejected(operator, "project_into")
