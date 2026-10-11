"""Explicit batched geometry and global detector rectangles, with cell oracles."""

from collections import Counter
import gc
import inspect
import math
from unittest import mock

import numpy as np
import pytest
import torch
from torch.utils._pytree import tree_flatten

import diffct
from tests.test_block_storage import _BlockAllocations, _GuardedStore, _api, _rejection
from tests.test_chunked_projector import _memory_inventory, _offsets
from tests.test_detector_surfaces import (
    _case, _close, _data, _finite_difference, _grids, _matrix, _numpy, _projector,
)
from tests.test_execution_buffers import _ExecutionTrace


BEAMS = ("parallel", "fan", "cone")
cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _settings(case):
    return ((3, 4, 5), (4, 3)) if case.beam == "cone" else ((4, 5), (3,))


def _fixture(beam, *, views=5, learnable=False):
    case = _case(beam, views=views, learnable=learnable)
    case.shape = (5, 7, 9) if beam == "cone" else (7, 9)
    parameters = (torch.tensor(3.7, dtype=torch.float64, requires_grad=learnable),
                  torch.tensor(.037, dtype=torch.float64, requires_grad=learnable),
                  torch.linspace(-.2, .3, views, dtype=torch.float64, requires_grad=learnable))
    return case, parameters


def _torch_offsets(beam, u, v, ids, radius, coefficient, coupling, *, batched=True, factor=1.):
    if batched:
        scale = factor * (1 + .11 * (ids.to(dtype=u.dtype) + 1) + .19 * coupling[ids] + .07 * coupling.mean())
        scale = scale.reshape(len(ids), *([1] * u.ndim))
    else:
        scale = factor * (1 + .07 * coupling.mean())
    a = radius * torch.sin(u / radius) + coefficient * scale * (u.square() + .13 * u * v + .07 * v.square())
    b = v + coefficient * scale * (.11 * u * v + .05 * u.square()) if beam == "cone" else torch.zeros_like(a)
    c = scale * radius * (1 - torch.cos(u / radius)) + coefficient * scale * (.09 * u * v + .12 * v.square())
    return torch.stack((a, b + torch.zeros_like(a), c), dim=-1)


def _numpy_offsets(case, parameters, *, batched=True, factor=1.):
    u, v = _grids(case)
    radius, coefficient, coupling = (np.asarray(value, dtype=np.float64) for value in parameters)
    if batched:
        scale = factor * (1 + .11 * (np.arange(case.views) + 1) + .19 * coupling + .07 * coupling.mean())
        scale = scale.reshape(case.views, *([1] * u.ndim))
    else:
        scale = factor * (1 + .07 * coupling.mean())
    a = radius * np.sin(u / radius) + coefficient * scale * (u ** 2 + .13 * u * v + .07 * v ** 2)
    b = v + coefficient * scale * (.11 * u * v + .05 * u ** 2) if case.beam == "cone" else np.zeros_like(a)
    c = scale * radius * (1 - np.cos(u / radius)) + coefficient * scale * (.09 * u * v + .12 * v ** 2)
    return np.stack((a, np.broadcast_to(b, a.shape), c), axis=-1)


class _Sampler:
    """Pure geometry formula; records only observe calls, never affect offsets."""
    def __init__(self, case, *, batched=True, factor=1., pixels=None, batch=None):
        self.case, self.batched, self.factor = case, batched, factor
        self.pixels, self.batch = pixels, batch
        self.phase, self.records = "construction", []

    def __call__(self, u, v, ids, *parameters):
        assert u.device.type == v.device.type == ids.device.type == "cpu"
        assert ids.ndim == 1 and ids.dtype in (torch.int32, torch.int64)
        assert all(parameter.device.type == "cpu" for parameter in parameters), "streamed parameters were not CPU snapshots"
        if self.pixels is not None:
            assert len(u.shape) == len(self.pixels) and all(a <= b for a, b in zip(u.shape, self.pixels))
        if self.batch is not None:
            assert len(ids) <= self.batch, "sampler received all global views instead of a bounded batch"
        self.records.append({"phase": self.phase, "u": _numpy(u), "v": _numpy(v), "ids": ids.tolist(),
                             "parameter_dtypes": tuple(parameter.dtype for parameter in parameters)})
        return _torch_offsets(self.case.beam, u, v, ids, *parameters,
                              batched=self.batched, factor=self.factor)


def _batched_projector(case, sampler, parameters, *, chunk=None, pixels=None, batch=2, **kwargs):
    assert "detector_chunk_shape" in inspect.signature(diffct.Projector).parameters, \
        "required declared Projector(detector_chunk_shape=...) interface is missing"
    surface = _api("ParameterizedSurface")(sampler, parameters=parameters)
    return _projector(case, surface, volume_chunk_shape=chunk, view_chunk_size=batch,
                      detector_chunk_shape=pixels, **kwargs)


class _GeometryTrace(_ExecutionTrace):
    def __init__(self, case, chunk, pixels, batch, parameters=()):
        super().__init__(case)
        self.volume_chunk, self.pixels, self.view_batch = chunk, pixels, batch
        self.parameter_sizes = [parameter.numel() for parameter in parameters]
        self.origins = {(parameter.device, parameter.untyped_storage()._cdata): index
                        for index, parameter in enumerate(parameters)}
        self.snapshots, self.points = Counter(), []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = super().__torch_dispatch__(function, types, args, kwargs)
        operation = self.operations[-1]
        if self.phase == "forward" and operation["inputs"]:
            source = operation["inputs"][0]
            index = self.origins.get((source["device"], source["storage"]))
            if index is not None and (
                    str(function) == "aten.clone.default" and source["device"].type == "cpu" or
                    str(function) == "aten._to_copy.default" and source["device"].type == "cuda"
                    and operation["outputs"][0]["device"].type == "cpu"):
                self.snapshots[index] += 1
        rank = len(self.volume_chunk)
        maximum = max(math.prod(self.volume_chunk), self.view_batch * math.prod(self.pixels) * rank,
                      self.view_batch * rank * (4 if self.case.beam == "cone" else 3),
                      *self.parameter_sizes)
        for tensor in tree_flatten(result)[0]:
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.is_cuda:
                assert tensor.numel() <= maximum, ("unbounded CUDA geometry/data tensor", str(function), tensor.shape)
                assert tensor.untyped_storage().nbytes() <= maximum * tensor.element_size(), \
                    "full CUDA backing storage hidden behind a bounded view"
            if self.case.views > self.view_batch or any(a > b for a, b in zip(self.case.detector, self.pixels)):
                assert tuple(tensor.shape) not in ((*self.case.sino_shape, rank), (*self.case.sino_shape, 3)), \
                    "global detector offsets/world points were expanded or saved"
        return result

    def _proxy(self, kernel, beam, kind):
        parent = super()._proxy(kernel, beam, kind)
        tracer = self

        class Proxy:
            def __getitem__(self, configuration):
                launch = parent[configuration]

                def observed(*args, **kwargs):
                    positions = args[-2] if kind == "geometry_vjp" else args[-1]
                    record = {"phase": tracer.phase, "beam": beam, "kind": kind,
                              "shape": tuple(positions.shape), "elements": int(positions.size)}
                    result = launch(*args, **kwargs)
                    tracer.points.append(record)
                    return result
                return observed
        return Proxy()


def _check_samples(case, records, pixels, batch):
    expected_u, expected_v = _grids(case)
    u_axis = expected_u[:, 0] if case.beam == "cone" else expected_u
    v_axis = expected_v[0, :] if case.beam == "cone" else None
    coverage = set()
    for record in records:
        u, v, ids = record["u"], record["v"], record["ids"]
        assert len(ids) <= batch and all(0 <= index < case.views for index in ids)
        assert all(a <= b for a, b in zip(u.shape, pixels))
        rows = np.argmin(np.abs(u_axis[:, None] - (u[:, 0] if case.beam == "cone" else u)[None, :]), axis=0)
        assert np.array_equal(rows, np.arange(rows[0], rows[0] + len(rows)))
        if case.beam == "cone":
            columns = np.argmin(np.abs(v_axis[:, None] - v[0, :][None, :]), axis=0)
            assert np.array_equal(columns, np.arange(columns[0], columns[0] + len(columns)))
            np.testing.assert_allclose(u, expected_u[np.ix_(rows, columns)], rtol=2e-7, atol=2e-7)
            np.testing.assert_allclose(v, expected_v[np.ix_(rows, columns)], rtol=2e-7, atol=2e-7)
            coverage.update((view, int(row), int(column)) for view in ids for row in rows for column in columns)
        else:
            np.testing.assert_allclose(u, expected_u[rows], rtol=2e-7, atol=2e-7)
            np.testing.assert_allclose(v, 0, atol=0)
            coverage.update((view, int(row)) for view in ids for row in rows)
    assert coverage == set(np.ndindex(case.sino_shape)), "some global view/pixel coordinates were never sampled"


def _check_parameter_and_frame_fd(case, values, geometry, gradients, image, sino, *, batched=True, factor=1.):
    generator = np.random.default_rng(922)

    def objective(parameters, frame):
        matrix = _matrix(case, _numpy_offsets(case, parameters, batched=batched, factor=factor), frame)
        return float(_numpy(sino).ravel() @ matrix @ _numpy(image).ravel())

    for index, (value, gradient) in enumerate(zip(values, gradients[:len(values)])):
        direction = generator.normal(size=value.shape) if value.ndim else np.asarray(1.)
        direction /= np.linalg.norm(direction)

        def evaluate(changed):
            parameters = list(values)
            parameters[index] = changed
            return objective(parameters, geometry)

        expected = _finite_difference(evaluate, value, direction)
        actual = float((_numpy(gradient) * direction).sum())
        assert abs(expected) > 1e-3, ("uninformative surface parameter", index)
        assert actual == pytest.approx(expected, rel=5e-3, abs=4e-4)
    for index, (value, gradient) in enumerate(zip(geometry, gradients[len(values):])):
        direction = generator.normal(size=value.shape)
        if case.beam == "parallel" and index == 0:
            direction -= (direction * value).sum(-1, keepdims=True) * value
        direction /= np.linalg.norm(direction)

        def evaluate(changed):
            frame = list(geometry)
            frame[index] = changed
            return objective(values, frame)

        expected = _finite_difference(evaluate, value, direction)
        assert abs(expected) > 1e-3, ("uninformative frame component", index)
        assert float((_numpy(gradient) * direction).sum()) == pytest.approx(expected, rel=5e-3, abs=4e-4)


@pytest.mark.parametrize("beam", BEAMS)
def test_sampler_fixture_matches_numpy_for_global_rectangle_and_omitted_view_parameter_coupling(beam):
    case, parameters = _fixture(beam)
    u, v = _grids(case)
    selected = (slice(1, 4), slice(1, 4)) if beam == "cone" else (slice(2, 6),)
    ids = torch.tensor([1, 4], dtype=torch.int64)
    actual = _torch_offsets(beam, torch.tensor(u[selected]), torch.tensor(v[selected]), ids, *parameters)
    expected = _numpy_offsets(case, tuple(_numpy(value) for value in parameters))
    _close(actual, expected[(ids.numpy(), *selected)], rtol=2e-12, atol=2e-12)
    altered = list(parameters)
    altered[-1] = parameters[-1].clone()
    altered[-1][0] += .4  # Global explicit parameter, although view 0 is not sampled.
    changed = _torch_offsets(beam, torch.tensor(u[selected]), torch.tensor(v[selected]), ids, *altered)
    assert not torch.allclose(changed, actual), "fixture lost explicit omitted-view coupling"


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
def test_legacy_callback_fixture_real_numerics_and_each_fd_before_new_interface(beam):
    case, parameters = _fixture(beam, learnable=True)
    chunk, pixels = _settings(case)

    def legacy(u, v):
        return _torch_offsets(beam, u, v, torch.arange(case.views), *parameters)

    projector = _projector(case, legacy, volume_chunk_shape=chunk, view_chunk_size=2)
    image, sino = _data(case, device="cpu")
    output = projector.project(image)
    values, geometry = tuple(_numpy(value) for value in parameters), tuple(_numpy(value) for value in case.trajectory)
    matrix = _matrix(case, _numpy_offsets(case, values), geometry)
    _close(output, (matrix @ image.numpy().ravel()).reshape(case.sino_shape))
    gradients = torch.autograd.grad((output.double() * sino.double()).sum(), (*parameters, *case.trajectory))
    _check_parameter_and_frame_fd(case, values, geometry, gradients, image, sino)


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("kind", ["short", "long", "zero", "negative", "bool", "float"])
def test_detector_chunks_reject_wrong_rank_or_nonpositive_nonintegral_limits(beam, kind):
    case, _ = _fixture(beam)
    pixels = list(_settings(case)[1])
    if kind == "short":
        pixels = pixels[:-1]
    elif kind == "long":
        pixels += [2]
    else:
        pixels[-1] = {"zero": 0, "negative": -1, "bool": True, "float": 2.0}[kind]
    with _rejection("detector|chunk|rank|shape|integer|positive"):
        _projector(case, detector_chunk_shape=tuple(pixels))


@pytest.mark.parametrize("beam", BEAMS)
def test_detector_chunks_constructor_accepts_integral_rank_without_cuda(beam):
    case, _ = _fixture(beam)
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        projector = _projector(case, detector_chunk_shape=tuple(np.int64(value + 10) for value in case.detector))
    assert projector.projection_shape == case.sino_shape and projector.volume_shape == case.shape


@pytest.mark.parametrize("kind", ["non-tensor", "integer", "nan", "infinite"])
def test_parameterized_surface_requires_explicit_finite_floating_tensor_parameters(kind):
    value = {"non-tensor": 3.7, "integer": torch.tensor(3), "nan": torch.tensor(float("nan")),
             "infinite": torch.tensor(float("inf"))}[kind]
    with _rejection("parameter|tensor|float|finite"):
        _api("ParameterizedSurface")(lambda u, v, ids, *parameters: torch.stack((u, v, u * 0), -1), parameters=(value,))


def test_parameterized_surface_does_not_impose_radius_semantics_on_arbitrary_parameters():
    def arbitrary(u, v, ids, parameter):
        return torch.stack((u, v, torch.zeros_like(u)), dim=-1)
    parameter = torch.tensor(-2., dtype=torch.float64)
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        surface = _api("ParameterizedSurface")(arbitrary, parameters=(parameter,))
    assert isinstance(surface, _api("ParameterizedSurface"))


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
def test_flat_detector_rectangles_use_global_world_points_and_match_full_cell_matrix(beam):
    case, _ = _fixture(beam)
    chunk, pixels = _settings(case)
    projector = _projector(case, volume_chunk_shape=chunk, view_chunk_size=2, detector_chunk_shape=pixels)
    image, sino = _data(case, device="cpu")
    with _GeometryTrace(case, chunk, pixels, 2) as trace:
        trace.phase = "forward"
        forward = projector.project(image)
        trace.phase = "backproject"
        backward = projector.backproject(sino)
    matrix = _matrix(case, _offsets(case, flat=True))
    _close(forward, (matrix @ image.numpy().ravel()).reshape(case.sino_shape))
    _close(backward, (matrix.T @ sino.numpy().ravel()).reshape(case.shape))
    assert trace.points and all(record["elements"] > 0 for record in trace.points), \
        "flat subsets did not dispatch the existing explicit-point native kernels"
    assert {record["views"] for record in trace.launches} == {1, 2}
    assert len({record["rays"]["shape"][1:] for record in trace.launches}) > 1, "pixel tails were not dispatched"


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("batched", [False, True], ids=["shared-offsets", "per-view-offsets"])
def test_explicit_surface_global_pixel_and_view_tails_match_cell_oracle_and_adjoint(beam, batched):
    case, parameters = _fixture(beam)
    chunk, pixels = _settings(case)
    sampler = _Sampler(case, batched=batched, pixels=pixels, batch=2)
    projector = _batched_projector(case, sampler, parameters, chunk=chunk, pixels=pixels)
    image, sino = _data(case, device="cpu")
    with _GeometryTrace(case, chunk, pixels, 2, parameters) as trace:
        sampler.phase = trace.phase = "project"
        forward = projector.project(image)
        sampler.phase = trace.phase = "backproject"
        backward = projector.backproject(sino)
    matrix = _matrix(case, _numpy_offsets(case, tuple(_numpy(value) for value in parameters), batched=batched))
    _close(forward, (matrix @ image.numpy().ravel()).reshape(case.sino_shape))
    _close(backward, (matrix.T @ sino.numpy().ravel()).reshape(case.shape))
    torch.testing.assert_close((forward.double() * sino.double()).sum(),
                               (image.double() * backward.double()).sum(), rtol=4e-5, atol=7e-5)
    for phase in ("project", "backproject"):
        _check_samples(case, [record for record in sampler.records if record["phase"] == phase], pixels, 2)
    assert trace.points and all(record["elements"] > 0 for record in trace.points)


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_surface_dense_and_structural_store_into_share_real_native_geometry_path(beam, operation):
    case, parameters = _fixture(beam)
    chunk, pixels = _settings(case)
    sampler = _Sampler(case, pixels=pixels, batch=2)
    projector = _batched_projector(case, sampler, parameters, chunk=chunk, pixels=pixels)
    image, sino = _data(case, device="cpu")
    data, output_shape = (image, case.sino_shape) if operation == "project" else (sino, case.shape)
    input_block, output_block = (chunk, (2, *pixels)) if operation == "project" else ((2, *pixels), chunk)
    source = _GuardedStore(data, input_block)
    output = _GuardedStore(torch.full(output_shape, torch.nan), output_block)
    with _GeometryTrace(case, chunk, pixels, 2, parameters) as trace:
        trace.phase = sampler.phase = "dense"
        dense = getattr(projector, operation)(data)
        trace.phase = sampler.phase = "store"
        with _BlockAllocations((case.shape, case.sino_shape), max(math.prod(chunk), 2 * math.prod(pixels) * len(chunk))):
            returned = getattr(projector, f"{operation}_into")(source, output)
    assert returned is output
    matrix = _matrix(case, _numpy_offsets(case, tuple(_numpy(value) for value in parameters)))
    operator = matrix if operation == "project" else matrix.T
    _close(output.tensor, (operator @ data.numpy().ravel()).reshape(output_shape))
    torch.testing.assert_close(output.tensor, dense, rtol=4e-4, atol=6e-5)
    assert torch.all(output.overwrites == 1)
    for phase in ("dense", "store"):
        kind = "forward" if operation == "project" else "backward"
        assert any(record["phase"] == phase and record["kind"] == kind for record in trace.points)


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_saved_sampler_parameter_frame_snapshots_fd_and_data_hessians(beam, operation):
    case, parameters = _fixture(beam, learnable=True)
    chunk, pixels = _settings(case)
    sampler = _Sampler(case, pixels=pixels, batch=2)
    projector = _batched_projector(case, sampler, parameters, chunk=chunk, pixels=pixels)
    image, sino = _data(case, device="cpu")
    data, weight = (image, sino) if operation == "project" else (sino, image)
    data = data.double().requires_grad_()
    values, geometry = tuple(_numpy(value) for value in parameters), tuple(_numpy(value) for value in case.trajectory)
    matrix = _matrix(case, _numpy_offsets(case, values), geometry)
    operator = matrix if operation == "project" else matrix.T
    direction = torch.linspace(-.4, .7, data.numel(), dtype=data.dtype).reshape_as(data)
    originals = (*parameters, *case.trajectory)
    with _GeometryTrace(case, chunk, pixels, 2, originals) as trace:
        trace.phase = sampler.phase = "forward"
        output = getattr(projector, operation)(data)
        assert trace.snapshots == Counter({index: 1 for index in range(len(originals))}), \
            "surface/frame values must snapshot once per invocation, independently of batch/pixel count"
        with torch.no_grad():
            parameters[0].add_(.19)
            parameters[1].mul_(1.17)
            parameters[2].add_(.11)
            case.trajectory[1].add_(.071)
        replacement = _Sampler(case, factor=2.3, pixels=pixels, batch=2)
        projector.detector_surface = _api("ParameterizedSurface")(replacement, parameters=parameters)
        trace.phase = sampler.phase = "backward"
        gradients = torch.autograd.grad((output.double() * weight.double()).sum(), (data, *originals), retain_graph=True)
        trace.phase = "quadratic-backward"
        gradient, = torch.autograd.grad(output.double().square().sum() / 2, data, create_graph=True)
        trace.phase = "data-hessian"
        hessian, = torch.autograd.grad((gradient * direction).sum(), data)
    _close(output, (operator @ _numpy(data).ravel()).reshape(weight.shape))
    _close(gradients[0], (operator.T @ _numpy(weight).ravel()).reshape(data.shape))
    normal = operator.T @ operator
    _close(gradient, (normal @ _numpy(data).ravel()).reshape(data.shape))
    _close(hessian, (normal @ _numpy(direction).ravel()).reshape(data.shape))
    assert all(value.dtype == original.dtype and value.device == original.device and torch.isfinite(value).all()
               for value, original in zip(gradients, (data, *originals)))
    _check_parameter_and_frame_fd(case, values, geometry, gradients[1:], image, sino)
    assert any(record["kind"] == "geometry_vjp" for record in trace.points)


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
def test_original_cuda_parameter_dtype_and_gradient_device_survive_cpu_snapshots(beam):
    case, parameters = _fixture(beam, learnable=True)
    case.trajectory = tuple(value.detach().to(device="cuda:0" if index % 2 else "cpu",
                                             dtype=torch.float32 if index % 2 else torch.float64).requires_grad_()
                            for index, value in enumerate(case.trajectory))
    parameters = tuple(value.detach().to(device="cuda:0" if index != 1 else "cpu",
                                         dtype=torch.float64 if index == 0 else torch.float32).requires_grad_()
                       for index, value in enumerate(parameters))
    chunk, pixels = _settings(case)
    sampler = _Sampler(case, pixels=pixels, batch=2)
    projector = _batched_projector(case, sampler, parameters, chunk=chunk, pixels=pixels)
    image, sino = _data(case, device="cpu")
    originals = (*parameters, *case.trajectory)
    values, geometry = tuple(_numpy(value) for value in parameters), tuple(_numpy(value) for value in case.trajectory)
    with _GeometryTrace(case, chunk, pixels, 2, originals) as trace:
        trace.phase = sampler.phase = "forward"
        output = projector.project(image)
        with torch.no_grad():
            parameters[0].add_(.19)
            case.trajectory[1].add_(.071)
        trace.phase = "backward"
        gradients = torch.autograd.grad((output.double() * sino.double()).sum(), originals)
    assert trace.snapshots == Counter({index: 1 for index in range(len(originals))})
    assert all(record["parameter_dtypes"] == tuple(parameter.dtype for parameter in parameters) for record in sampler.records)
    for gradient, original in zip(gradients, originals):
        assert gradient.dtype == original.dtype and gradient.device == original.device
    _check_parameter_and_frame_fd(case, values, geometry, gradients, image, sino)


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("kind", ["shape", "view-count", "nan", "integer"])
def test_invalid_sampled_last_view_batch_is_rejected(beam, kind):
    case, parameters = _fixture(beam)
    chunk, pixels = _settings(case)

    def invalid_tail(u, v, ids, *values):
        offsets = _torch_offsets(beam, u, v, ids, *values)
        if torch.any(ids == case.views - 1):
            if kind == "shape":
                return offsets[..., :2]
            if kind == "view-count":
                return torch.cat((offsets, offsets[:1]), dim=0)
            if kind == "integer":
                return offsets.long()
            offsets = offsets.clone()
            offsets.reshape(-1, 3)[-1, -1] = torch.nan
        return offsets

    image, _ = _data(case, device="cpu")
    with _rejection("shape|view|batch|finite|float|tensor|surface|coordinate"):
        projector = _batched_projector(case, invalid_tail, parameters, chunk=chunk, pixels=pixels)
        projector.project(image)


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("parameter", ["surface", "frame"])
def test_geometry_second_derivative_limit_remains_explicit(parameter):
    case, parameters = _fixture("cone", learnable=True)
    chunk, pixels = _settings(case)
    projector = _batched_projector(case, _Sampler(case, pixels=pixels, batch=2), parameters,
                                   chunk=chunk, pixels=pixels)
    image, sino = _data(case, device="cpu")
    output = projector.project(image)
    chosen = parameters[0] if parameter == "surface" else case.trajectory[1]
    first, = torch.autograd.grad((output.double() * sino.double()).sum(), chosen, create_graph=True)
    with pytest.raises(RuntimeError, match="second derivatives.*geometry|geometry.*second derivatives"):
        torch.autograd.grad(first.sum(), chosen)


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
def test_automatic_geometry_budget_shrinks_pixels_when_one_full_view_cannot_fit(beam):
    case, parameters = _fixture(beam)
    case.shape = (3, 4, 5) if beam == "cone" else (4, 5)
    case.detector = (37, 29) if beam == "cone" else (129,)
    sampler = _Sampler(case)
    image = torch.linspace(.13, .7, math.prod(case.shape)).reshape(case.shape)
    # Native CUDA allocation rounding makes a 512-byte total budget unable
    # to hold even the distinct tile/ray/frame/point buffers. 3072 bytes fits
    # a serial bounded rectangle while a complete 129-point 2D view cannot fit.
    free = 4096 * 4 // 3 if beam == "cone" else 3072 * 4 // 3
    with _memory_inventory({torch.cuda.current_device(): free}):
        projector = _batched_projector(case, sampler, parameters, batch=None)
        with _ExecutionTrace(case) as trace:
            output = projector.project(image)
    records = [record for record in trace.launches if record["kind"] == "forward"]
    assert records and all(math.prod(record["rays"]["shape"][1:]) < math.prod(case.detector) for record in records)
    matrix = _matrix(case, _numpy_offsets(case, tuple(_numpy(value) for value in parameters)))
    _close(output, (matrix @ image.numpy().ravel()).reshape(case.sino_shape))


@pytest.mark.cuda
@cuda_required
def test_automatic_geometry_budget_reduces_view_batches_before_detector_pixels():
    case, parameters = _fixture("cone")
    case.shape, case.detector = (3, 4, 5), (9, 7)
    image, _ = _data(case, device="cpu")
    with _memory_inventory({torch.cuda.current_device(): 6400}):
        projector = _batched_projector(case, _Sampler(case), parameters, chunk=case.shape, batch=None)
        with _ExecutionTrace(case) as trace:
            output = projector.project(image)
    assert trace.launches and all(record["rays"]["shape"][1:] == case.detector for record in trace.launches)
    assert max(record["views"] for record in trace.launches) < case.views
    matrix = _matrix(case, _numpy_offsets(case, tuple(_numpy(value) for value in parameters)))
    _close(output, (matrix @ image.numpy().ravel()).reshape(case.sino_shape))


@pytest.mark.cuda
@cuda_required
def test_explicit_pixel_view_and_spatial_limits_fail_clearly_if_minimum_cannot_fit():
    case, parameters = _fixture("cone")
    chunk, pixels = _settings(case)
    with _memory_inventory({torch.cuda.current_device(): 64}), _ExecutionTrace(case) as trace, \
            _rejection("memory|budget|fit|minimum|limit"):
        projector = _batched_projector(case, _Sampler(case), parameters, chunk=chunk, pixels=pixels)
        projector.project(_data(case, device="cpu")[0])
    assert not trace.launches


def _geometry_memory_trial(views, detector):
    case, parameters = _fixture("cone", views=views, learnable=True)
    case.detector = detector
    chunk, pixels = _settings(case)
    sampler = _Sampler(case, pixels=pixels, batch=2)
    projector = _batched_projector(case, sampler, parameters, chunk=chunk, pixels=pixels)
    image, sino = _data(case, device="cpu")
    image.requires_grad_()
    projector.project(image.detach())
    gc.collect()
    torch.cuda.synchronize()
    initial = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with _GeometryTrace(case, chunk, pixels, 2, (*parameters, *case.trajectory)) as trace:
        trace.phase = "forward"
        output = projector.project(image)
        trace.phase = "backward"
        (output * sino).sum().backward()
    torch.cuda.synchronize()
    assert image.grad is not None and all(value.grad is not None and torch.isfinite(value.grad).all() for value in parameters)
    assert any(record["kind"] == "geometry_vjp" for record in trace.points)
    return torch.cuda.max_memory_allocated() - initial


@pytest.mark.cuda
@cuda_required
def test_geometry_peak_is_bounded_across_global_view_and_detector_growth_including_vjp():
    small = _geometry_memory_trial(5, (6, 5))
    more_views = _geometry_memory_trial(37, (6, 5))
    more_pixels = _geometry_memory_trial(5, (10, 9))
    # Two bounded native tiles and four bounded point/cotangent batches allow
    # allocator/layout scratch without allowing global view/pixel expansion.
    slack = 2 * 4 * (3 * 4 * 5) + 4 * 4 * 2 * (4 * 3) * 3
    assert more_views <= small + slack, (small, more_views, slack)
    assert more_pixels <= small + slack, (small, more_pixels, slack)


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("layout", ["float", "double", "half", "source-double", "mixed-axes", "double-offsets"])
def test_sampled_geometry_common_arithmetic_dtype_preserves_cpu_graph_and_original_gradient_dtypes(beam, layout):
    from diffct.surfaces import _GeometrySpec, _sample_geometry

    case = _case(beam, views=4)
    count = len(case.trajectory)
    dtype = {"half": torch.float16, "double": torch.float64}.get(layout, torch.float32)
    dtypes = [dtype] * count
    if layout == "source-double":
        dtypes[0] = torch.float64
    elif layout == "mixed-axes":
        dtypes[-1] = torch.float64
    frames = tuple(value.to(dtype=dtype).detach().requires_grad_()
                   for value, dtype in zip(case.trajectory, dtypes))
    gain = torch.tensor(.17, dtype=torch.float64 if layout == "double-offsets" else dtypes[1], requires_grad=True)
    selected_views = slice(1, 3)
    selected_pixels = (slice(1, 4), slice(1, 3)) if beam == "cone" else (slice(2, 6),)
    samples = []

    def sampler(u, v, ids, parameter):
        assert u.device.type == v.device.type == ids.device.type == parameter.device.type == "cpu"
        assert u.dtype == v.dtype == torch.float64 and ids.tolist() == [1, 2]
        base = torch.stack((u, v if beam == "cone" else torch.zeros_like(v),
                            .13 * u + .09 * v + .21), dim=-1).to(parameter.dtype)
        samples.append(_numpy(base))
        return parameter * base

    source, points = _sample_geometry(beam, case.detector, case.pitch, (*frames, gain),
                                     _GeometrySpec("sampled", sampler), selected_views, selected_pixels, case.spacing)
    expected_dtype = torch.float64 if any(value.dtype == torch.float64 for value in (*frames, gain)) else torch.float32
    assert source.dtype == points.dtype == expected_dtype
    assert source.device.type == points.device.type == "cpu" and source.grad_fn is not None and points.grad_fn is not None
    values = tuple(_numpy(value) for value in frames)
    # The independent world-point oracle uses quantized caller values and a
    # NumPy frame cross product; it never calls the production geometry builder.
    offsets = _numpy(gain * torch.from_numpy(samples[0]).to(gain.dtype))

    def world(frame, local):
        center = frame[1][selected_views].reshape(2, *([1] * len(selected_pixels)), -1)
        axis = frame[2][selected_views].reshape(center.shape)
        if beam == "cone":
            vertical = frame[3][selected_views].reshape(center.shape)
            return center + local[..., 0, None] * axis + local[..., 1, None] * vertical + local[..., 2, None] * np.cross(axis, vertical)
        normal = np.stack((axis[..., 1], -axis[..., 0]), axis=-1)
        return center + local[..., 0, None] * axis + local[..., 2, None] * normal

    expected = world(values, offsets)
    np.testing.assert_allclose(_numpy(source), values[0][selected_views], rtol=0, atol=0)
    np.testing.assert_allclose(_numpy(points), expected, rtol=2e-6, atol=2e-6)
    weight = np.linspace(.011, .073, points.numel()).reshape(points.shape)
    source_weight = np.linspace(.019, .053, source.numel()).reshape(source.shape)
    loss = (points * torch.from_numpy(weight).to(points.dtype)).sum() + (source * torch.from_numpy(source_weight).to(source.dtype)).sum()
    gradients = torch.autograd.grad(loss, (*frames, gain))
    for original, gradient in zip((*frames, gain), gradients):
        assert gradient.dtype == original.dtype and gradient.device == original.device and torch.isfinite(gradient).all()
    for index, value in enumerate(values):
        direction = np.linspace(.13, .37, value.size).reshape(value.shape)

        def objective(changed):
            frame = list(values)
            frame[index] = changed
            return float((world(frame, offsets) * weight).sum() + (frame[0][selected_views] * source_weight).sum())

        expected_gradient = _finite_difference(objective, value, direction)
        assert float((_numpy(gradients[index]) * direction).sum()) == pytest.approx(expected_gradient, rel=2e-3, abs=2e-4)
    expected_gain = float(((world(values, samples[0]) - world(values, np.zeros_like(samples[0]))) * weight).sum())
    assert float(gradients[-1]) == pytest.approx(expected_gain, rel=2e-3, abs=2e-4)


@pytest.mark.parametrize("beam", BEAMS)
def test_sampled_half_geometry_accepts_finite_world_points_beyond_half_range(beam):
    from diffct.surfaces import _GeometrySpec, _sample_geometry
    from tests.test_detector_surfaces import _axis_case

    case = _axis_case(beam)
    frames = tuple(value[:1].half() for value in case.trajectory)
    frames[1][:, 0] = 50000.

    def sampler(u, v, ids):
        return torch.stack((u, v, torch.full_like(u, 40000.)), dim=-1).half()

    source, points = _sample_geometry(beam, case.detector, case.pitch, frames,
                                     _GeometrySpec("sampled", sampler), slice(0, 1),
                                     tuple(slice(0, size) for size in case.detector), case.spacing)
    assert source.dtype == points.dtype == torch.float32 and torch.isfinite(points).all()
    torch.testing.assert_close(points[..., 0], torch.full(points.shape[:-1], 89984.), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["nan", "infinite", "outside-float32"])
def test_sampled_common_dtype_still_rejects_invalid_world_coordinates(kind):
    from diffct.surfaces import _GeometrySpec, _sample_geometry

    case = _case("cone", views=1)

    def sampler(u, v, ids):
        value = {"nan": float("nan"), "infinite": float("inf"), "outside-float32": 1e40}[kind]
        return torch.stack((u, v, torch.full_like(u, value)), dim=-1)

    with pytest.raises(ValueError, match="finite|float32"):
        _sample_geometry(case.beam, case.detector, case.pitch, case.trajectory,
                         _GeometrySpec("sampled", sampler), slice(0, 1),
                         tuple(slice(0, size) for size in case.detector), case.spacing)


@pytest.mark.cuda
@cuda_required
def test_sampled_float32_offsets_mixed_axes_real_projection_and_geometry_vjp():
    case, parameters = _fixture("cone", learnable=True)
    case.trajectory = tuple(value.detach().to(torch.float64 if index == 3 else torch.float32).requires_grad_()
                            for index, value in enumerate(case.trajectory))
    parameters = tuple(value.detach().float().requires_grad_() for value in parameters)

    class FloatSampler(_Sampler):
        def __call__(self, *args):
            return super().__call__(*args).float()

    chunk, pixels = _settings(case)
    projector = _batched_projector(case, FloatSampler(case), parameters, chunk=chunk, pixels=pixels)
    image, sino = _data(case, device="cpu")
    values = tuple(_numpy(value) for value in parameters)
    geometry = tuple(_numpy(value) for value in case.trajectory)
    with _GeometryTrace(case, chunk, pixels, 2, (*parameters, *case.trajectory)) as trace:
        output = projector.project(image)
        gradients = torch.autograd.grad((output.double() * sino.double()).sum(), (*parameters, *case.trajectory))
    matrix = _matrix(case, _numpy_offsets(case, values), geometry)
    _close(output, (matrix @ _numpy(image).ravel()).reshape(case.sino_shape))
    assert any(record["kind"] == "geometry_vjp" for record in trace.points)
    assert all(gradient.dtype == original.dtype for original, gradient in zip((*parameters, *case.trajectory), gradients))
    _check_parameter_and_frame_fd(case, values, geometry, gradients, image, sino)
