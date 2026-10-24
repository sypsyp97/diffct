"""Portable Siddon behavior against an independent per-cell slab matrix.

Geometry is fixed; data derivatives and torch.func transforms are checked
against explicit linear maps. The oracle does not call a DiffCT projector or
use the production ray-plane traversal. Its dense matrix is deliberately tiny
and confined to tests; production intermediates are checked separately.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import itertools
import math

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from diffct import Projector
from diffct.utils import TorchCUDABridge


BEAMS = ("parallel", "fan", "cone")
CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="native parity requires NVIDIA CUDA",
)


@dataclass(frozen=True)
class _Case:
    beam: str
    trajectory: tuple[torch.Tensor, ...]
    shape: tuple[int, ...]
    detector: tuple[int, ...]
    pitch: tuple[float, ...]
    spacing: float | tuple[float, ...]

    @property
    def projection_shape(self):
        return (len(self.trajectory[0]), *self.detector)


def _case(beam, *, scalar=False):
    angles = torch.tensor([0.231, 0.811, 1.431], dtype=torch.float64)
    radial = torch.stack((angles.cos(), angles.sin()), dim=-1)
    tangent = torch.stack((-angles.sin(), angles.cos()), dim=-1)
    # Noncircular, translated views exercise arbitrary geometry, not a helper
    # whose implementation could share the projector's coordinate mistake.
    offsets = torch.tensor([[0.173, -0.239], [-0.119, 0.281], [0.317, 0.113]],
                           dtype=torch.float64)
    if beam == "parallel":
        geometry = (radial, offsets + 3.7 * radial, tangent)
    elif beam == "fan":
        geometry = (-7.1 * radial + offsets, 5.3 * radial + offsets, tangent)
    else:
        tilts = torch.tensor([0.197, -0.137, 0.283], dtype=torch.float64)
        radial = torch.cat((radial * tilts.cos()[:, None], tilts.sin()[:, None]), dim=-1)
        tangent = torch.cat((tangent, torch.zeros(3, 1, dtype=torch.float64)), dim=-1)
        vertical = torch.linalg.cross(radial, tangent, dim=-1)
        offsets = torch.cat((offsets, torch.tensor([[-0.317], [0.113], [0.413]],
                                                  dtype=torch.float64)), dim=-1)
        geometry = (-7.1 * radial + offsets, 5.3 * radial + offsets, tangent, vertical)
    return _Case(
        beam, geometry, (2, 3, 4) if beam == "cone" else (3, 4),
        (3, 2) if beam == "cone" else (5,),
        (0.91, 0.67) if beam == "cone" else (0.91,),
        0.73 if scalar else ((1.3, 0.9, 0.6) if beam == "cone" else (1.1, 0.7)),
    )


def _operator(case, *, backend="torch", precision="float64", **kwargs):
    return Projector(
        case.trajectory, case.shape, case.detector, beam=case.beam,
        detector_spacing=case.pitch if case.beam == "cone" else case.pitch[0],
        voxel_spacing=case.spacing, backend=backend, precision=precision, **kwargs,
    )


def _slab_length(origin, direction, lower, upper, *, segment):
    """Intersect one cell box; half-open constant axes choose one boundary cell."""
    enter, leave = (0.0, 1.0) if segment else (-math.inf, math.inf)
    for p, delta, lo, hi in zip(origin, direction, lower, upper):
        if delta == 0.0:
            if not lo <= p < hi:
                return 0.0
        else:
            first, last = (lo - p) / delta, (hi - p) / delta
            enter, leave = max(enter, min(first, last)), min(leave, max(first, last))
    return max(0.0, leave - enter) * math.sqrt(sum(d * d for d in direction))


def _matrix(case, *, curved=False):
    """Float64 lengths in tensor-order columns and view/U/V-order rows."""
    geometry = tuple(component.detach().cpu().double().tolist() for component in case.trajectory)
    spacing = ((case.spacing,) * len(case.shape) if isinstance(case.spacing, (int, float))
               else case.spacing)
    rows = []
    for pixel in itertools.product(*(range(n) for n in case.projection_shape)):
        view = pixel[0]
        local = [(index - (n - 1) / 2) * pitch
                 for index, n, pitch in zip(pixel[1:], case.detector, case.pitch)]
        u, v = local[0], local[1] if case.beam == "cone" else 0.0
        n = 0.0
        if curved:
            u, n = u - 0.03 * u ** 3, 0.15 * u ** 2
        axis_u = geometry[2][view]
        if case.beam == "cone":
            axis_v = geometry[3][view]
            normal = [axis_u[1] * axis_v[2] - axis_u[2] * axis_v[1],
                      axis_u[2] * axis_v[0] - axis_u[0] * axis_v[2],
                      axis_u[0] * axis_v[1] - axis_u[1] * axis_v[0]]
        else:
            axis_v, normal = (0.0, 0.0), (axis_u[1], -axis_u[0])
        endpoint = [center + u * du + v * dv + n * dn
                    for center, du, dv, dn in zip(geometry[1][view], axis_u, axis_v, normal)]
        origin = endpoint if case.beam == "parallel" else geometry[0][view]
        direction = (geometry[0][view] if case.beam == "parallel"
                     else [p - s for p, s in zip(endpoint, origin)])
        row = []
        for cell in itertools.product(*(range(n) for n in case.shape)):
            lower = [(index - size / 2) * pitch
                     for index, size, pitch in zip(cell, case.shape, spacing)][::-1]
            upper = [lo + pitch for lo, pitch in zip(lower, spacing[::-1])]
            row.append(_slab_length(origin, direction, lower, upper,
                                    segment=case.beam != "parallel"))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float64)


def _data(shape, *, dtype=torch.float64, device="cpu", phase=0.0):
    values = torch.arange(math.prod(shape), dtype=torch.float64)
    return (torch.sin(values * 0.713 + phase) + 0.2 * torch.cos(values * 0.317)).reshape(shape).to(
        dtype=dtype, device=device,
    )


def _close(actual, expected, *, precision="float64"):
    tolerance = 2e-11 if precision == "float64" else 4e-5
    torch.testing.assert_close(actual.detach().cpu().double(), expected.double(),
                               rtol=tolerance, atol=tolerance)


def _no_numba(*args, **kwargs):
    raise AssertionError("portable backend called the native CUDA bridge")


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("scalar", [False, True], ids=["anisotropic", "scalar"])
def test_cpu_forward_and_adjoint_match_independent_cell_matrix(beam, scalar, monkeypatch):
    monkeypatch.setattr(TorchCUDABridge, "tensor_to_cuda_array", _no_numba)
    case = _case(beam, scalar=scalar)
    matrix = _matrix(case)
    assert matrix.abs().sum() > 0
    operator = _operator(case)
    assert operator.backend == "torch" and operator.precision == "float64"
    assert operator.dtype == torch.float64
    image, sino = _data(case.shape), _data(case.projection_shape, phase=0.31)
    forward, adjoint = operator.project(image), operator.backproject(sino)
    assert forward.dtype == adjoint.dtype == torch.float64
    assert forward.device == image.device and adjoint.device == sino.device
    _close(forward, (matrix @ image.flatten()).reshape(case.projection_shape))
    _close(adjoint, (matrix.T @ sino.flatten()).reshape(case.shape))
    _close((forward * sino).sum(), (image * adjoint).sum())


@pytest.mark.parametrize("beam", BEAMS)
def test_all_basis_vectors_reproduce_independent_matrix_and_transpose(beam):
    case = _case(beam)
    matrix = _matrix(case)
    operator = _operator(case, view_chunk_size=2, detector_chunk_shape=(2,) * len(case.detector),
                         volume_chunk_shape=(2,) * len(case.shape))
    images = torch.eye(math.prod(case.shape), dtype=torch.float64).reshape(-1, *case.shape)
    sinograms = torch.eye(math.prod(case.projection_shape), dtype=torch.float64).reshape(
        -1, *case.projection_shape,
    )
    _close(operator.project(images).reshape(len(images), -1), matrix.T)
    _close(operator.backproject(sinograms).reshape(len(sinograms), -1), matrix)


@pytest.mark.parametrize("beam", BEAMS)
def test_anisotropic_axis_chords_use_tensor_order_physical_spacing(beam):
    dimensions = 3 if beam == "cone" else 2
    shape = (3, 5, 7) if beam == "cone" else (3, 5)
    spacing = (1.3, 0.9, 0.6) if beam == "cone" else (1.1, 0.7)
    directions = torch.cat((torch.eye(dimensions), -torch.eye(dimensions))).double()
    u = torch.roll(directions, 1, dims=1)
    centers = 10 * directions
    geometry = ((directions, centers, u) if beam == "parallel" else
                (-centers, centers, u) if beam == "fan" else
                (-centers, centers, u, torch.linalg.cross(directions, u, dim=-1)))
    detector = (1, 1) if beam == "cone" else (1,)
    case = _Case(beam, geometry, shape, detector, (1.0,) * len(detector), spacing)
    expected = torch.tensor([n * pitch for n, pitch in zip(shape[::-1], spacing[::-1])] * 2,
                            dtype=torch.float64)
    _close(_operator(case).project(torch.ones(shape, dtype=torch.float64)).flatten(), expected)


@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_finite_rays_clip_at_source_and_detector_inside_volume(beam):
    dimensions = 3 if beam == "cone" else 2
    source = torch.zeros(1, dimensions, dtype=torch.float64)
    center, u = source.clone(), source.clone()
    source[0, 0], center[0, 0], source[0, 1], center[0, 1], u[0, 1] = -0.5, 0.75, 0.13, 0.13, 1.0
    geometry = (source, center, u)
    if beam == "cone":
        v = torch.zeros_like(u)
        v[0, 2] = 1.0
        geometry += (v,)
    detector = (1, 1) if beam == "cone" else (1,)
    shape = (3, 3, 5) if beam == "cone" else (3, 5)
    case = _Case(beam, geometry, shape, detector, (1.0,) * len(detector), 1.0)
    _close(_operator(case).project(torch.ones(shape, dtype=torch.float64)),
           torch.full(case.projection_shape, 1.25, dtype=torch.float64))


@pytest.mark.parametrize("precision", ["fast", "float32", "float64"])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32, torch.float64])
def test_precision_controls_output_and_public_metadata(precision, input_dtype):
    case = _case("parallel", scalar=True)
    operator = _operator(case, precision=precision)
    dtype = torch.float64 if precision == "float64" else torch.float32
    assert operator.backend == "torch" and operator.precision == precision
    assert operator.dtype == dtype
    image, sino = _data(case.shape, dtype=input_dtype), _data(case.projection_shape, dtype=input_dtype)
    matrix = _matrix(case)
    forward, adjoint = operator.project(image), operator.backproject(sino)
    assert forward.dtype == adjoint.dtype == dtype
    _close(forward, (matrix @ image.double().flatten()).reshape(case.projection_shape), precision=precision)
    _close(adjoint, (matrix.T @ sino.double().flatten()).reshape(case.shape), precision=precision)


def test_float64_retains_geometry_and_both_accumulators():
    angle = 0.37123456789123
    ray = torch.tensor([[math.cos(angle), math.sin(angle)]], dtype=torch.float64)
    u = torch.tensor([[-math.sin(angle), math.cos(angle)]], dtype=torch.float64)
    offset = torch.tensor([[0.137123456789, -0.239234567891]], dtype=torch.float64)
    case = _Case("parallel", (ray, offset, u), (3, 5), (1,), (1.0,), (0.73, 0.61))
    # This ray exits the two x faces: analytic box chord, no Siddon oracle.
    analytic_chord = 5 * 0.61 / math.cos(angle)
    _close(_operator(case).project(torch.ones(case.shape, dtype=torch.float64)),
           torch.tensor([[analytic_chord]], dtype=torch.float64))

    geometry = tuple(torch.tensor([vector] * 3, dtype=torch.float64)
                     for vector in ((1.0, 0.0), (0.0, 0.0), (0.0, 1.0)))
    case = _Case("parallel", geometry, (1, 3), (1,), (1.0,), 1.0)
    operator = _operator(case, view_chunk_size=2, volume_chunk_shape=(1, 2))
    values = torch.tensor([2 ** 30, 1.0, -(2 ** 30)], dtype=torch.float64)
    _close(operator.project(values.reshape(case.shape)), torch.ones(case.projection_shape, dtype=torch.float64))
    _close(operator.backproject(values.reshape(case.projection_shape)), torch.ones(case.shape, dtype=torch.float64))


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("leading", [(), (2,), (2, 3)])
def test_project_and_adjoint_preserve_arbitrary_batch_channel_dimensions(beam, leading):
    case = _case(beam)
    matrix = _matrix(case)
    operator = _operator(case)
    image, sino = _data((*leading, *case.shape)), _data((*leading, *case.projection_shape), phase=0.17)
    if len(leading) == 2:
        image, sino = image.transpose(0, 1), sino.transpose(0, 1)
        leading = leading[::-1]
        assert not image.is_contiguous() and not sino.is_contiguous()
    projected, adjoint = operator.project(image), operator.backproject(sino)
    assert projected.shape == (*leading, *case.projection_shape)
    assert adjoint.shape == (*leading, *case.shape)
    _close(projected, (image.reshape(-1, math.prod(case.shape)) @ matrix.T).reshape(projected.shape))
    _close(adjoint, (sino.reshape(-1, math.prod(case.projection_shape)) @ matrix).reshape(adjoint.shape))


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_torch_func_transforms_and_second_data_derivatives_match_linear_map(beam, operation):
    case = _case(beam)
    operator, matrix = _operator(case), _matrix(case)
    forward = operation == "project"
    matrix = matrix if forward else matrix.T
    input_shape, output_shape = ((case.shape, case.projection_shape) if forward else
                                 (case.projection_shape, case.shape))
    function = getattr(operator, operation)
    value, tangent = _data(input_shape), _data(input_shape, phase=0.79)
    weight = _data(output_shape, phase=0.31)
    linear_loss = lambda x: (function(x) * weight).sum()
    _close(torch.func.grad(linear_loss)(value), (matrix.T @ weight.flatten()).reshape(input_shape))
    primal, derivative = torch.func.jvp(function, (value,), (tangent,))
    _close(primal, (matrix @ value.flatten()).reshape(output_shape))
    _close(derivative, (matrix @ tangent.flatten()).reshape(output_shape))
    expected_jacobian = matrix.reshape(*output_shape, *input_shape)
    _close(torch.func.jacrev(function)(value), expected_jacobian)
    _close(torch.func.jacfwd(function)(value), expected_jacobian)
    batches = _data((2, 3, *input_shape), phase=0.43)
    expected_batches = (batches.reshape(-1, matrix.shape[1]) @ matrix.T).reshape(2, 3, *output_shape)
    _close(torch.func.vmap(torch.func.vmap(function))(batches), expected_batches)

    quadratic_loss = lambda x: function(x).square().sum() / 2
    expected_hessian = (matrix.T @ matrix).reshape(*input_shape, *input_shape)
    _close(torch.func.jacfwd(torch.func.grad(quadratic_loss))(value), expected_hessian)
    leaf = value.clone().requires_grad_()
    first, = torch.autograd.grad(quadratic_loss(leaf), leaf, create_graph=True)
    second, = torch.autograd.grad((first * tangent).sum(), leaf)
    _close(second, (matrix.T @ matrix @ tangent.flatten()).reshape(input_shape))


@pytest.mark.parametrize("beam", BEAMS)
def test_curved_detector_callback_remains_supported_by_torch(beam):
    case = _case(beam)
    matrix, flat = _matrix(case, curved=True), _matrix(case)
    assert (matrix - flat).abs().max() > 1e-4

    def surface(u, v):
        return torch.stack((u - 0.03 * u ** 3, v, 0.15 * u ** 2), dim=-1)

    operator = _operator(case, detector_surface=surface, view_chunk_size=2,
                         detector_chunk_shape=(2,) * len(case.detector))
    image, sino = _data(case.shape), _data(case.projection_shape)
    _close(operator.project(image), (matrix @ image.flatten()).reshape(case.projection_shape))
    _close(operator.backproject(sino), (matrix.T @ sino.flatten()).reshape(case.shape))


class _IntermediateBudget(TorchDispatchMode):
    """Observe actual tensor outputs; reject hidden dense ray-by-volume work."""

    def __init__(self, limit):
        super().__init__()
        self.limit = limit
        self.largest = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        pending = [output]
        while pending:
            item = pending.pop()
            if isinstance(item, torch.Tensor):
                self.largest = max(self.largest, item.numel())
                assert item.numel() <= self.limit, (
                    f"{func} created {tuple(item.shape)} ({item.numel()} elements); "
                    f"chunked intermediate budget is {self.limit}"
                )
            elif isinstance(item, (tuple, list)):
                pending.extend(item)
        return output


def test_reference_chunks_bound_intermediates_and_keep_global_coordinates():
    case = replace(_case("cone"), shape=(5, 7, 9), detector=(4, 3))
    matrix = _matrix(case)
    image, sino = _data(case.shape), _data(case.projection_shape)
    # Eight rays fit one view/pixel batch. Even four simultaneous arrays of
    # every global plane crossing fit; a ray-by-full-volume matrix does not.
    budget = max(image.numel(), sino.numel(), 4 * 8 * (sum(case.shape) + 3))
    assert budget < matrix.numel() and budget < 8 * image.numel()
    with _IntermediateBudget(budget) as allocations:
        operator = _operator(case, view_chunk_size=2, detector_chunk_shape=(2, 2),
                             volume_chunk_shape=(2, 3, 4))
        forward, adjoint = operator.project(image), operator.backproject(sino)
    assert allocations.largest > 0
    _close(forward, (matrix @ image.flatten()).reshape(case.projection_shape))
    _close(adjoint, (matrix.T @ sino.flatten()).reshape(case.shape))


@pytest.mark.parametrize("backend", [None, "", "Torch", "invalid", 1])
def test_invalid_backend_is_rejected(backend):
    with pytest.raises((TypeError, ValueError)):
        _operator(_case("parallel", scalar=True), backend=backend)


@pytest.mark.parametrize("precision", [None, "", "fp64", "half", 1])
def test_invalid_precision_is_rejected(precision):
    with pytest.raises((TypeError, ValueError)):
        _operator(_case("parallel", scalar=True), precision=precision)


@pytest.mark.parametrize("beam,spacing", [
    ("parallel", 0.0), ("parallel", -1.0), ("parallel", math.nan),
    ("parallel", math.inf), ("parallel", True), ("parallel", (1.0,)),
    ("parallel", (1.0, 2.0, 3.0)), ("parallel", (1.0, 0.0)),
    ("parallel", (math.nan, 1.0)), ("cone", (1.0, 2.0)),
    ("cone", (1.0, math.inf, 1.0)), ("cone", (1.0, 1.0, -1.0)),
])
def test_invalid_scalar_or_per_axis_voxel_spacing_is_rejected(beam, spacing):
    with pytest.raises((TypeError, ValueError)):
        _operator(replace(_case(beam), spacing=spacing))


@pytest.mark.parametrize("precision,spacing", [
    ("float32", 1.0), ("float64", 1.0), ("fast", (1.0, 1.0)),
    ("float64", (1.0, 2.0)),
])
def test_numba_validates_precision_and_tuple_spacing(precision, spacing):
    if precision == "fast" and torch.cuda.is_available() and torch.version.hip is None:
        operator = _operator(replace(_case("parallel"), spacing=spacing),
                             backend="numba", precision=precision)
        assert operator.backend == "numba" and operator.precision == precision
        return
    with pytest.raises((ValueError, RuntimeError)) as error:
        _operator(replace(_case("parallel"), spacing=spacing), backend="numba", precision=precision)
    assert not isinstance(error.value, NotImplementedError), "native configuration validation is still pending"


@pytest.mark.parametrize("platform", ["no-cuda", "hip"])
def test_numba_rejects_unavailable_native_platform(platform, monkeypatch):
    if platform == "no-cuda":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    else:
        monkeypatch.setattr(torch.version, "hip", "7.0-test")
    with pytest.raises((ValueError, RuntimeError)) as error:
        _operator(_case("parallel", scalar=True), backend="numba", precision="fast")
    assert not isinstance(error.value, NotImplementedError), "native platform validation is still pending"


@pytest.mark.parametrize("mode", ["no-cuda", "hip", "float32", "float64", "anisotropic"])
def test_auto_routes_configurations_to_expected_backend(mode, monkeypatch):
    case = _case("parallel", scalar=mode != "anisotropic")
    precision = mode if mode in ("float32", "float64") else "fast"
    if mode == "no-cuda":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if mode == "hip":
        # Dispatch simulation only; real HIP tensor execution needs ROCm hardware.
        monkeypatch.setattr(torch.version, "hip", "7.0-test")
    expected_backend = ("numba" if mode == "anisotropic" and torch.cuda.is_available()
                        and torch.version.hip is None else "torch")
    if expected_backend == "torch":
        monkeypatch.setattr(TorchCUDABridge, "tensor_to_cuda_array", _no_numba)
    operator = _operator(case, backend="auto", precision=precision)
    assert operator.backend == expected_backend and operator.precision == precision
    image = _data(case.shape)
    _close(operator.project(image), (_matrix(case) @ image.flatten()).reshape(case.projection_shape),
           precision=precision)


@pytest.mark.cuda
@CUDA_REQUIRED
@pytest.mark.parametrize("beam", BEAMS)
def test_native_cuda_and_portable_backend_match_independent_reference_with_batches(beam, monkeypatch):
    case = _case(beam, scalar=True)
    matrix = _matrix(case)
    images = _data((2, 3, *case.shape), dtype=torch.float32, device="cuda")
    sinograms = _data((2, 3, *case.projection_shape), dtype=torch.float32, device="cuda", phase=0.31)
    expected = (images.cpu().double().reshape(-1, math.prod(case.shape)) @ matrix.T).reshape(
        2, 3, *case.projection_shape,
    )
    expected_adjoint = (sinograms.cpu().double().reshape(-1, math.prod(case.projection_shape)) @ matrix).reshape(
        2, 3, *case.shape,
    )
    native_bridge = TorchCUDABridge.tensor_to_cuda_array
    native_calls = []

    def record_native(tensor):
        native_calls.append(tuple(tensor.shape))
        return native_bridge(tensor)

    monkeypatch.setattr(TorchCUDABridge, "tensor_to_cuda_array", record_native)
    for backend in ("numba", "auto", "torch"):
        operator = _operator(case, backend=backend, precision="fast")
        assert operator.backend == ("torch" if backend == "torch" else "numba")
        assert operator.dtype == torch.float32
        native_calls.clear()
        # Unbatched CUDA must still use the existing NVIDIA native fast path.
        _close(operator.project(images[0, 0]), expected[0, 0], precision="fast")
        _close(operator.backproject(sinograms[0, 0]), expected_adjoint[0, 0], precision="fast")
        assert bool(native_calls) == (backend != "torch")
        forward, adjoint = operator.project(images), operator.backproject(sinograms)
        assert forward.device == images.device and adjoint.device == sinograms.device
        _close(forward, expected, precision="fast")
        _close(adjoint, expected_adjoint, precision="fast")


@pytest.mark.cuda
@CUDA_REQUIRED
@pytest.mark.parametrize("beam", BEAMS)
def test_portable_float64_cuda_matches_cpu_reference_without_native_bridge(beam, monkeypatch):
    monkeypatch.setattr(TorchCUDABridge, "tensor_to_cuda_array", _no_numba)
    case = _case(beam)
    matrix = _matrix(case)
    operator = _operator(case)
    image, sino = _data(case.shape, device="cuda"), _data(case.projection_shape, device="cuda")
    forward, adjoint = operator.project(image), operator.backproject(sino)
    assert forward.dtype == adjoint.dtype == torch.float64
    assert forward.device == image.device and adjoint.device == sino.device
    _close(forward, (matrix @ image.cpu().flatten()).reshape(case.projection_shape))
    _close(adjoint, (matrix.T @ sino.cpu().flatten()).reshape(case.shape))
