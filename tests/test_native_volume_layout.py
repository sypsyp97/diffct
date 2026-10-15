"""Public native cone layout, copy avoidance, and autograd contracts."""

import inspect
import math

import pytest
import torch

import diffct


cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
SHAPE = (7, 9, 11)
DETECTOR = (9, 7)
VIEWS = 5


def _geometry(device="cuda"):
    # Offsets avoid symmetric rays and exact voxel corners. Unequal dimensions
    # expose axis swaps that a cubic, constant-volume fixture would hide.
    angle = torch.arange(VIEWS, dtype=torch.float64) * (math.pi / VIEWS) + 0.137
    radial = torch.stack((angle.cos(), angle.sin(), torch.zeros_like(angle)), dim=1)
    u = torch.stack((-angle.sin(), angle.cos(), torch.zeros_like(angle)), dim=1)
    v = torch.tensor([0., 0., 1.], dtype=torch.float64).expand(VIEWS, -1)
    offset = torch.stack((torch.full_like(angle, 0.193),
                          torch.full_like(angle, -0.271),
                          torch.linspace(-0.63, 0.79, VIEWS, dtype=torch.float64)), dim=1)
    return tuple(t.to(device=device, dtype=torch.float32).contiguous()
                 for t in (-80 * radial + offset, 60 * radial + offset, u, v))


def _projector(native, *, geometry=None, shape=SHAPE, **kwargs):
    return diffct.Projector(
        _geometry() if geometry is None else geometry, shape, DETECTOR,
        beam="cone", native_volume_layout=native, **kwargs,
    )


def _values(shape, seed=19):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)).cuda()


def _native_view(volume):
    return volume.permute(2, 1, 0).contiguous().permute(2, 1, 0)


def _assert_native(volume, shape=SHAPE):
    assert tuple(volume.shape) == shape
    assert volume.stride() == (1, shape[0], shape[0] * shape[1])
    assert not volume.is_contiguous()
    assert volume.permute(2, 1, 0).is_contiguous()


def _assert_bits(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == torch.float32
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    # Compare representations, including signed zero, rather than a tolerance.
    assert torch.equal(actual.contiguous().view(torch.int32),
                       expected.contiguous().view(torch.int32))


def test_native_layout_default_is_false_without_cuda():
    assert inspect.signature(diffct.Projector).parameters["native_volume_layout"].default is False
    projector = diffct.Projector(_geometry("cpu"), SHAPE, DETECTOR, beam="cone")
    assert projector.native_volume_layout is False


@pytest.mark.cuda
@cuda_required
def test_full_cone_backprojection_native_view_is_bitwise_equal_to_default():
    geometry = _geometry()
    default = diffct.Projector(geometry, SHAPE, DETECTOR, beam="cone")
    ordinary = _projector(False, geometry=geometry)
    native = _projector(True, geometry=geometry)
    sinogram = _values((VIEWS, *DETECTOR))
    with torch.no_grad():
        expected = default.backproject(sinogram)
        explicit = ordinary.backproject(sinogram)
        actual = native.backproject(sinogram)
    assert expected.is_contiguous() and explicit.is_contiguous()
    _assert_native(actual)
    _assert_bits(actual, expected)
    _assert_bits(explicit, expected)
    for projector in (default, ordinary, native):
        assert projector.last_execution_stats is None, "fixture unexpectedly streamed"


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("native", [False, True])
def test_project_accepts_both_volume_layouts_bitwise(native):
    volume = _values(SHAPE)
    view = _native_view(volume)
    _assert_native(view)
    projector = _projector(native)
    with torch.no_grad():
        expected = _projector(False).project(volume)
        contiguous_result = projector.project(volume)
        view_result = projector.project(view)
    _assert_bits(contiguous_result, expected)
    _assert_bits(view_result, expected)
    assert projector.last_execution_stats is None, "fixture unexpectedly streamed"


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("native", [False, True])
def test_project_native_view_does_not_allocate_a_volume_copy(native):
    # Measure live allocations, not reserved cache blocks or end-of-call bytes:
    # even a temporary clone freed before return raises the allocator peak.
    # The 240 KiB volume dwarfs the 1260-byte sinogram and five-view geometry.
    # A half-volume ceiling leaves generous metadata space but rejects a copy.
    shape = (32, 40, 48)
    volume = _native_view(_values(shape))
    _assert_native(volume, shape)
    projector = _projector(native, shape=shape)
    with torch.no_grad():
        warm = projector.project(volume)
        torch.cuda.synchronize()
        del warm
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = projector.project(volume)
        torch.cuda.synchronize()
        increment = torch.cuda.max_memory_allocated() - baseline
    assert projector.last_execution_stats is None, "copy measurement must use the full path"
    assert result.shape == (VIEWS, *DETECTOR)
    assert torch.isfinite(result).all()
    assert increment < volume.numel() * volume.element_size() // 2, (
        f"native view allocated {increment} bytes for a {volume.numel() * volume.element_size()}-byte volume"
    )


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("native_input", [False, True], ids=["contiguous", "strided"])
def test_data_and_geometry_gradients_match_across_layouts(operation, native_input):
    geometry = _geometry()
    input_shape = SHAPE if operation == "project" else (VIEWS, *DETECTOR)
    output_shape = (VIEWS, *DETECTOR) if operation == "project" else SHAPE
    values = _values(input_shape)
    weight = _values(output_shape, seed=43)

    def evaluate(native, strided):
        source, detector = (t.detach().clone().requires_grad_() for t in geometry[:2])
        operator = _projector(native, geometry=(source, detector, *geometry[2:]))
        data = (_native_view(values) if strided else values.clone()).detach().requires_grad_()
        cotangent = _native_view(weight) if strided else weight
        output = getattr(operator, operation)(data)
        gradients = torch.autograd.grad(output, (data, source, detector), grad_outputs=cotangent)
        assert operator.last_execution_stats is None, "gradient fixture unexpectedly streamed"
        return output.detach(), tuple(g.detach() for g in gradients)

    reference, expected = evaluate(False, False)
    for native in (False, True):
        actual, gradients = evaluate(native, native_input)
        _assert_bits(actual, reference)
        _assert_bits(gradients[0], expected[0])
        for gradient, target in zip(gradients[1:], expected[1:]):
            assert torch.isfinite(gradient).all()
            assert target.abs().max() > 0, "geometry fixture produced a vacuous gradient"
            # Geometry VJPs reduce products; storage order can change float32
            # reduction rounding. Allow 16 eps at the gradient's global scale.
            tolerance = 16 * torch.finfo(torch.float32).eps
            torch.testing.assert_close(gradient, target, rtol=tolerance,
                                       atol=tolerance * target.abs().max().item())


@pytest.mark.cuda
@cuda_required
def test_cg_style_updates_remain_native_and_match_contiguous_loop():
    ordinary, native = _projector(False), _projector(True)
    contiguous = _values(SHAPE)
    view = _native_view(contiguous)
    with torch.no_grad():
        for _ in range(4):
            _assert_native(view)
            ordinary_sino = ordinary.project(contiguous)
            native_sino = native.project(view)
            _assert_bits(native_sino, ordinary_sino)
            ordinary_update = ordinary.backproject(ordinary_sino)
            native_update = native.backproject(native_sino)
            _assert_native(native_update)
            _assert_bits(native_update, ordinary_update)
            contiguous = contiguous + 0.0001 * ordinary_update
            view = view + 0.0001 * native_update
            assert contiguous.is_contiguous()
            _assert_native(view)
            _assert_bits(view, contiguous)
            assert ordinary.last_execution_stats is None and native.last_execution_stats is None


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam,streamed", [
    ("parallel", False), ("fan", False),
    ("parallel", True), ("fan", True), ("cone", True),
])
def test_other_beams_and_streaming_ignore_native_layout(beam, streamed):
    if beam == "cone":
        trajectory, shape, detector = _geometry(), SHAPE, DETECTOR
    else:
        source, center, u, _ = _geometry()
        direction = -source[:, :2] + center[:, :2]
        direction = direction / direction.norm(dim=1, keepdim=True)
        trajectory = (direction if beam == "parallel" else source[:, :2].contiguous(),
                      center[:, :2].contiguous(), u[:, :2].contiguous())
        shape, detector = (9, 11), (9,)
    options = {"volume_chunk_shape": tuple(max(1, n // 2) for n in shape),
               "schedule": "spatial"} if streamed else {}
    ordinary = diffct.Projector(trajectory, shape, detector, beam=beam,
                               native_volume_layout=False, **options)
    native = diffct.Projector(trajectory, shape, detector, beam=beam,
                             native_volume_layout=True, **options)
    volume = _values(shape)
    sinogram = _values(ordinary.projection_shape, seed=53)
    with torch.no_grad():
        for operation, data in (("project", volume), ("backproject", sinogram)):
            expected = getattr(ordinary, operation)(data)
            actual = getattr(native, operation)(data)
            assert expected.is_contiguous() and actual.is_contiguous()
            _assert_bits(actual, expected)
            if streamed:
                assert ordinary.last_execution_stats is not None
                assert native.last_execution_stats is not None
