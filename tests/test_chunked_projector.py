"""CPU-backed spatial tiles: independent cell references and real CUDA evidence.

Hardware tests execute production kernels; launch proxies only observe and
forward calls. CPU configuration tests remain runnable without a CUDA machine.
The slab oracle is reused read-only from the detector-surface contract tests.
"""

from contextlib import ExitStack
from datetime import timedelta
import gc
import math
import sys
from unittest import mock

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from diffct import Projector, kernels
from tests.test_detector_surfaces import (
    _Case, _case, _close, _data, _finite_difference, _flat_surface,
    _grids, _matrix, _numpy, _projector,
)


BEAMS = ("parallel", "fan", "cone")
cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _chunk(case):
    return (2, 3, 4) if case.beam == "cone" else (3, 4)


def _surface(case, strength=1.0, per_view=False):
    def surface(u, v):
        s = torch.as_tensor(strength, dtype=u.dtype, device=u.device)
        if per_view:
            s = s * (1 + .17 * torch.arange(case.views, dtype=u.dtype))
            s = s.reshape(case.views, *([1] * u.ndim))
        if case.beam == "cone":
            a = u + s * (.13 * u * v + .07 * v.square())
            b = v + s * (.11 * u * v + .05 * u.square())
            c = s * (.19 * u.square() + .09 * u * v + .12 * v.square())
        else:
            a = 3.7 * torch.sin(u / 3.7) + .037 * s * u.square()
            b = torch.zeros_like(a)
            c = s * 3.7 * (1 - torch.cos(u / 3.7))
        return torch.stack((a, b + torch.zeros_like(a), c), dim=-1)
    return surface


def _offsets(case, strength=1.0, per_view=False, flat=False):
    u, v = _grids(case)
    if flat:
        return np.stack((u, v, np.zeros_like(u)), axis=-1)
    s = np.asarray(strength, dtype=np.float64)
    if per_view:
        s = (s * (1 + .17 * np.arange(case.views))).reshape(
            case.views, *([1] * u.ndim))
    if case.beam == "cone":
        a = u + s * (.13 * u * v + .07 * v ** 2)
        b = v + s * (.11 * u * v + .05 * u ** 2)
        c = s * (.19 * u ** 2 + .09 * u * v + .12 * v ** 2)
    else:
        a = 3.7 * np.sin(u / 3.7) + .037 * s * u ** 2
        b = np.zeros_like(a)
        c = s * 3.7 * (1 - np.cos(u / 3.7))
    return np.stack((a, np.broadcast_to(b, a.shape), c), axis=-1)


def _noncontiguous(tensor):
    return tensor.transpose(-1, -2).contiguous().transpose(-1, -2)


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("chunk", ["tuple", "list", "oversized"])
def test_cpu_constructor_accepts_rank_exact_integral_chunks_without_cuda(beam, chunk):
    case = _case(beam)
    dims = _chunk(case)
    dims = list(dims) if chunk == "list" else dims
    dims = tuple(np.int64(100) for _ in dims) if chunk == "oversized" else dims
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        projector = _projector(case, volume_chunk_shape=dims, view_chunk_size=np.int64(2))
    assert projector.volume_shape == case.shape
    assert projector.projection_shape == case.sino_shape
    assert projector.view_slice == slice(0, case.views)


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("kind", ["scalar", "short", "long", "zero", "negative", "bool", "float"])
def test_cpu_constructor_rejects_invalid_volume_chunks(beam, kind):
    case = _case(beam)
    dims = list(_chunk(case))
    if kind == "scalar":
        dims = 2
    elif kind == "short":
        dims = dims[:-1]
    elif kind == "long":
        dims += [2]
    else:
        dims[-1] = {"zero": 0, "negative": -1, "bool": True, "float": 2.0}[kind]
    with pytest.raises((TypeError, ValueError)):
        _projector(case, volume_chunk_shape=dims)


@pytest.mark.parametrize("value", [0, -2, True, 1.5, (2,), [2], "2"])
def test_cpu_constructor_rejects_invalid_view_batches(value):
    case = _case("parallel")
    with pytest.raises((TypeError, ValueError)):
        _projector(case, volume_chunk_shape=(3, 4), view_chunk_size=value)


@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("kind", ["shape", "integer", "complex", "not-tensor"])
def test_cpu_chunk_input_validation(operation, kind):
    case = _case("fan")
    projector = _projector(case, volume_chunk_shape=(3, 4))
    shape = case.shape if operation == "project" else case.sino_shape
    data = torch.ones(shape)
    if kind == "shape":
        data = data[:-1]
    elif kind == "integer":
        data = data.long()
    elif kind == "complex":
        data = data.to(torch.complex64)
    else:
        data = data.tolist()
    message = "shape" if kind == "shape" else ("floating|dtype" if kind in ("integer", "complex") else "[Tt]ensor")
    with pytest.raises((TypeError, ValueError), match=message):
        getattr(projector, operation)(data)


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("per_view,dtype", [(False, torch.float32), (True, torch.float64)])
def test_cpu_forward_adjoint_noncontiguous_match_cell_oracle_and_full(beam, per_view, dtype):
    case = _case(beam, views=5)
    if beam == "cone":
        case.shape = (5, 7, 9)
    surface = _surface(case, per_view=per_view)
    tiled = _projector(case, surface, volume_chunk_shape=_chunk(case), view_chunk_size=2)
    full = _projector(case, surface)
    image, sino = (_noncontiguous(t.to(dtype)) for t in _data(case, device="cpu"))
    assert not image.is_contiguous() and not sino.is_contiguous()
    original_image, original_sino = image.clone(), sino.clone()
    matrix = _matrix(case, _offsets(case, per_view=per_view))
    projected, backprojected = tiled.project(image), tiled.backproject(sino)
    for output in (projected, backprojected):
        assert output.device.type == "cpu" and output.dtype == torch.float32
    _close(projected, (matrix @ _numpy(image).ravel()).reshape(case.sino_shape))
    _close(backprojected, (matrix.T @ _numpy(sino).ravel()).reshape(case.shape))
    _close(projected, _numpy(full.project(image.cuda())))
    _close(backprojected, _numpy(full.backproject(sino.cuda())))
    lhs = (projected.double() * sino.double()).sum()
    rhs = (image.double() * backprojected.double()).sum()
    assert abs(lhs.item()) > .01, "uninformative adjoint fixture"
    torch.testing.assert_close(lhs, rhs, rtol=4e-5, atol=7e-5)
    torch.testing.assert_close(image, original_image, rtol=0, atol=0)
    torch.testing.assert_close(sino, original_sino, rtol=0, atol=0)


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
def test_flat_default_path_unchanged_and_oversized_tiles_clipped(beam):
    case = _case(beam)
    image, sino = _data(case, device="cpu")
    full = _projector(case)
    chunks = tuple(dim + 5 for dim in case.shape)
    tiled = _projector(case, volume_chunk_shape=chunks)
    flat = _projector(case, _flat_surface, volume_chunk_shape=chunks, view_chunk_size=2)
    matrix = _matrix(case, _offsets(case, flat=True))
    for operation, data, reference in (
        ("project", image, (matrix @ _numpy(image).ravel()).reshape(case.sino_shape)),
        ("backproject", sino, (matrix.T @ _numpy(sino).ravel()).reshape(case.shape)),
    ):
        expected = getattr(full, operation)(data.cuda())
        _close(getattr(tiled, operation)(data), reference)
        _close(getattr(flat, operation)(data), reference)
        actual_cuda = getattr(tiled, operation)(data.double().cuda().requires_grad_())
        assert actual_cuda.is_cuda and actual_cuda.dtype == torch.float32
        _close(actual_cuda, _numpy(expected))


def _boundary_case(beam):
    dimensions = 3 if beam == "cone" else 2
    points = [[8, -.5], [-.5, 8], [8, 8], [8, 1.39], [10, 8]]
    sources = [[-8, -.5], [-.5, -8], [-8, -8], [.23, -.11], [8, 8]]
    if dimensions == 3:
        points = [p + [z] for p, z in zip(points, [-.5, -.5, 8, .83, 8])]
        sources = [p + [z] for p, z in zip(sources, [-.5, -.5, -8, .17, 8])]
    source, center = (torch.tensor(x, dtype=torch.float64) for x in (sources, points))
    direction = center - source
    direction /= torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    u = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
    if dimensions == 3:
        u = torch.cat((u, torch.zeros(5, 1)), dim=-1)
        u /= torch.linalg.vector_norm(u, dim=-1, keepdim=True)
        geometry = source, center, u, torch.linalg.cross(direction, u, dim=-1)
    elif beam == "parallel":
        # Fifth parallel line misses, fourth tests an asymmetric interior chord.
        center[-1, 1] = 8
        geometry = direction, center, u
    else:
        geometry = source, center, u
    return _Case(beam, geometry, (5, 5, 7) if dimensions == 3 else (5, 7),
                 (1, 1) if dimensions == 3 else (1,),
                 (1., 1.) if dimensions == 3 else (1.,), spacing=1.)


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("one_cell", [False, True])
def test_internal_tile_faces_edges_corners_inside_sources_and_misses(beam, one_cell):
    case = _boundary_case(beam)
    chunk = (1,) * len(case.shape) if one_cell else ((2, 2, 3) if beam == "cone" else (2, 3))
    projector = _projector(case, volume_chunk_shape=chunk, view_chunk_size=2)
    image, sino = _data(case, device="cpu")
    matrix = _matrix(case, _offsets(case, flat=True))
    assert not matrix[-1].any() and matrix[0].sum() > 0
    _close(projector.project(image), (matrix @ _numpy(image).ravel()).reshape(case.sino_shape))
    _close(projector.backproject(sino), (matrix.T @ _numpy(sino).ravel()).reshape(case.shape))


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_cpu_data_gradient_dtype_and_hessian_with_trainable_surface(beam, operation):
    case = _case(beam)
    strength = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    projector = _projector(case, _surface(case, strength),
                           volume_chunk_shape=_chunk(case), view_chunk_size=2)
    image, sino = _data(case, device="cpu")
    data = (image if operation == "project" else sino).double().requires_grad_()
    direction = torch.linspace(-.4, .7, data.numel(), dtype=data.dtype).reshape_as(data)
    output = getattr(projector, operation)(data)
    gradient, = torch.autograd.grad(output.double().square().sum() / 2, data, create_graph=True)
    hessian, = torch.autograd.grad((gradient * direction).sum(), data)
    assert gradient.device == data.device and gradient.dtype == data.dtype
    assert hessian.device == data.device and hessian.dtype == data.dtype
    matrix = _matrix(case, _offsets(case))
    operator = matrix if operation == "project" else matrix.T
    normal = operator.T @ operator
    _close(gradient, (normal @ _numpy(data).ravel()).reshape(data.shape))
    _close(hessian, (normal @ _numpy(direction).ravel()).reshape(data.shape))


def _objective(case, image, sino, strength=1., geometry=None, per_view=False, flat=False):
    matrix = _matrix(case, _offsets(case, strength, per_view, flat), geometry)
    return float(_numpy(sino).ravel() @ matrix @ _numpy(image).ravel())


def _check_geometry_grads(case, image, sino, gradients, strength, geometry, per_view):
    expected = _finite_difference(lambda s: _objective(case, image, sino, s, geometry, per_view), strength)
    assert abs(expected) > 1e-3
    assert gradients[0].item() == pytest.approx(expected, rel=5e-3, abs=4e-4)
    _check_frame_grads(case, image, sino, gradients[1:], geometry, strength, per_view)


def _check_frame_grads(case, image, sino, gradients, geometry, strength=1., per_view=False, flat=False):
    generator = np.random.default_rng(922)
    for index, gradient in enumerate(gradients):
        assert gradient is not None and gradient.device.type == "cpu"
        direction = generator.normal(size=geometry[index].shape)
        if case.beam == "parallel" and index == 0:
            direction -= (direction * geometry[index]).sum(-1, keepdims=True) * geometry[index]
        direction /= np.linalg.norm(direction)

        def value(perturbed):
            changed = list(geometry)
            changed[index] = perturbed
            return _objective(case, image, sino, strength, changed, per_view, flat)

        expected = _finite_difference(value, geometry[index], direction)
        assert abs(expected) > 1e-3
        assert float((_numpy(gradient) * direction).sum()) == pytest.approx(
            expected, rel=5e-3, abs=4e-4)


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_surface_and_all_frame_vjps_and_saved_forward_geometry(beam, operation):
    case = _case(beam, learnable=True)
    strength = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    per_view = operation == "backproject"
    projector = _projector(case, _surface(case, strength, per_view),
                           volume_chunk_shape=_chunk(case), view_chunk_size=2)
    image, sino = _data(case, device="cpu")
    data, weight = (image, sino) if operation == "project" else (sino, image)
    previous = None
    for current in (1., 1.17):
        with torch.no_grad():
            strength.fill_(current)
            if current != 1.:
                case.trajectory[1][:, 0].add_(.083)
        geometry = tuple(_numpy(g) for g in case.trajectory)
        saved_matrix = _matrix(case, _offsets(case, current, per_view), geometry)
        variable = data.double().requires_grad_()
        output = getattr(projector, operation)(variable)
        expected = ((saved_matrix @ _numpy(image).ravel()).reshape(case.sino_shape)
                    if operation == "project" else
                    (saved_matrix.T @ _numpy(sino).ravel()).reshape(case.shape))
        _close(output, expected)
        if previous is not None:
            assert not torch.allclose(output, previous), "updated geometry was cached"
        previous = output.detach().clone()
        # Mutable parameters must not invalidate the forward snapshot or alter its VJP.
        with torch.no_grad():
            strength.add_(.19)
            case.trajectory[1][:, 1].sub_(.071)
        loss = (output.double() * weight.double()).sum()
        gradients = torch.autograd.grad(loss, (variable, strength, *case.trajectory))
        expected_data = ((saved_matrix.T @ _numpy(sino).ravel()).reshape(case.shape)
                         if operation == "project" else
                         (saved_matrix @ _numpy(image).ravel()).reshape(case.sino_shape))
        _close(gradients[0], expected_data)
        assert gradients[0].device == variable.device and gradients[0].dtype == variable.dtype
        _check_geometry_grads(case, image, sino, gradients[1:], current, geometry, per_view)


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_flat_frame_vjps_updated_calls_and_saved_forward_geometry(beam, operation):
    case = _case(beam, learnable=True)
    projector = _projector(case, volume_chunk_shape=_chunk(case), view_chunk_size=2)
    image, sino = _data(case, device="cpu")
    data, weight = (image, sino) if operation == "project" else (sino, image)
    previous = None
    for _ in range(2):
        geometry = tuple(_numpy(g) for g in case.trajectory)
        matrix = _matrix(case, _offsets(case, flat=True), geometry)
        variable = data.double().requires_grad_()
        output = getattr(projector, operation)(variable)
        operator = matrix if operation == "project" else matrix.T
        _close(output, (operator @ _numpy(data).ravel()).reshape(output.shape))
        if previous is not None:
            assert not torch.allclose(output, previous), "updated flat frame was cached"
        previous = output.detach().clone()
        with torch.no_grad():
            case.trajectory[1][:, 0].add_(.083)
        gradients = torch.autograd.grad((output.double() * weight.double()).sum(),
                                        (variable, *case.trajectory))
        _close(gradients[0], (operator.T @ _numpy(weight).ravel()).reshape(data.shape))
        _check_frame_grads(case, image, sino, gradients[1:], geometry, flat=True)
    output = getattr(projector, operation)(data)
    first, = torch.autograd.grad((output.double() * weight.double()).sum(),
                                 case.trajectory[1], create_graph=True)
    with pytest.raises(RuntimeError, match="second derivatives.*geometry|geometry.*second derivatives"):
        torch.autograd.grad(first.sum(), case.trajectory[1])


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
def test_fixed_original_frames_frozen_and_geometry_second_derivatives_raise(beam):
    case = _case(beam)
    fixed = _projector(case, volume_chunk_shape=_chunk(case), view_chunk_size=2)
    image, sino = _data(case, device="cpu")
    before = fixed.project(image)
    case.trajectory[1].add_(.37)
    torch.testing.assert_close(fixed.project(image), before, rtol=0, atol=0)
    case = _case(beam, learnable=True)
    strength = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    projector = _projector(case, _surface(case, strength), volume_chunk_shape=_chunk(case))
    for chosen in (strength, case.trajectory[1]):
        output = projector.project(image)
        gradient, = torch.autograd.grad((output.double() * sino.double()).sum(), chosen,
                                        create_graph=True)
        with pytest.raises(RuntimeError, match="second derivatives.*geometry|geometry.*second derivatives"):
            torch.autograd.grad(gradient.sum(), chosen)


@pytest.mark.cuda
@cuda_required
def test_cuda_callback_offsets_rejected_before_world_point_expansion():
    case = _case("cone")

    def callback(u, v):
        return _flat_surface(u, v).cuda()

    # The supplied CUDA offset itself is caller-owned; frame expansion would
    # create the complete (views,U,V,xyz) CUDA geometry and is forbidden.
    class NoWorldExpansion(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            result = function(*args, **(kwargs or {}))
            for tensor in tree_flatten(result)[0]:
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    assert tuple(tensor.shape) != (*case.sino_shape, 3), "expanded full CUDA surface"
            return result

    with NoWorldExpansion(), pytest.raises((TypeError, ValueError)):
        _projector(case, callback, volume_chunk_shape=_chunk(case), view_chunk_size=2)


class _Launches:
    """Observe actual kernel arrays/devices; always execute the original kernel."""
    def __init__(self):
        self.records = []
        self.phase = None
        self.patches = ExitStack()

    def __enter__(self):
        originals = {}
        for beam in BEAMS:
            prefix = "_cone_3d" if beam == "cone" else f"_{beam}_2d"
            for kind in ("forward", "backward", "geometry_vjp"):
                name = f"{prefix}_{kind}_kernel"
                original = getattr(kernels, name)
                originals[id(original)] = self._proxy(original, beam, kind)
        for name, module in list(sys.modules.items()):
            if name == "diffct" or name.startswith("diffct."):
                for attribute, value in list(vars(module).items()):
                    if id(value) in originals:
                        self.patches.enter_context(mock.patch.object(module, attribute, originals[id(value)]))
        return self

    def _proxy(self, kernel, beam, kind):
        tracer = self

        class Proxy:
            def __getitem__(self, config):
                configured = kernel[config]

                def launch(*args, **kwargs):
                    backward = kind == "backward"
                    volume = args[4 if beam == "cone" else 3] if backward else args[0]
                    sino = args[0] if backward else args[4 if beam == "cone" else 3]
                    record = (tracer.phase, beam, kind, torch.cuda.current_device(),
                              tuple(volume.shape), tuple(sino.shape))
                    result = configured(*args, **kwargs)
                    tracer.records.append(record)
                    return result
                return launch
        return Proxy()

    def __exit__(self, *exception):
        return self.patches.__exit__(*exception)


class _StagingGuard(TorchDispatchMode):
    def __init__(self, case, chunk, views):
        super().__init__()
        self.case, self.chunk, self.views = case, chunk, views
        self.limit = max(math.prod(chunk), views * math.prod(case.detector) * 3,
                         views * 12)

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        for tensor in tree_flatten(result)[0]:
            if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                continue
            shape = tuple(tensor.shape)
            assert tensor.numel() <= self.limit, f"unbounded CUDA tensor: {function}, {shape}"
            if self.case.views > self.views:
                assert shape != self.case.sino_shape, "full sinogram staged on CUDA"
                assert shape != (*self.case.sino_shape, len(self.case.shape)), "full detector geometry on CUDA"
        return result


def _assert_bounded_launches(records, case, chunk, batch, phases, devices=None):
    for phase in phases:
        observed = [record for record in records if record[0] == phase]
        assert observed, f"no real CUDA launch in {phase}"
        for _, beam, _, device, volume, sino in observed:
            assert beam == case.beam
            assert all(a <= b for a, b in zip(volume, chunk[::-1] if beam == "cone" else chunk)), volume
            assert 0 < sino[0] <= batch, sino
            assert sino[1:] == case.detector
            if devices is not None:
                assert device in devices


@pytest.mark.cuda
@cuda_required
def test_real_tiles_and_default_32_view_batches_in_both_directions():
    case = _case("cone", views=35)
    case.shape = (5, 7, 9)
    chunk = (2, 3, 4)
    projector = _projector(case, volume_chunk_shape=chunk)
    image, sino = _data(case, device="cpu")
    with _Launches() as launches, _StagingGuard(case, chunk, 32):
        launches.phase = "project"
        output = projector.project(image)
        launches.phase = "backproject"
        back = projector.backproject(sino)
    assert output.device.type == back.device.type == "cpu"
    _assert_bounded_launches(launches.records, case, chunk, 32, ("project", "backproject"))
    assert {record[-1][0] for record in launches.records} == {3, 32}
    assert len({record[-2] for record in launches.records}) > 1, "final partial tiles unobserved"


def _reachable_cuda(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        return value.numel() if value.is_cuda else 0
    if isinstance(value, dict):
        return sum(_reachable_cuda(v, seen) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_reachable_cuda(v, seen) for v in value)
    return 0


def _memory_trial(shape, views):
    case = _case("cone", views=views, learnable=True)
    case.shape, case.detector = shape, (16, 12)
    chunk, batch = (24, 24, 24), 3
    strength = torch.tensor(.21, dtype=torch.float64, requires_grad=True)
    projector = _projector(case, _surface(case, strength, per_view=True),
                           volume_chunk_shape=chunk, view_chunk_size=batch)
    image = torch.ones(shape, dtype=torch.float64, requires_grad=True)
    sino = torch.linspace(-.4, .8, math.prod(case.sino_shape), dtype=torch.float64).reshape(case.sino_shape)
    sino.requires_grad_()
    # Warm compilation outside the peak window, then measure live allocations.
    projector.project(image.detach())
    gc.collect()
    torch.cuda.synchronize()
    initial = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    saved_cuda = []

    def pack(tensor):
        if tensor.is_cuda:
            saved_cuda.append(tensor.numel())
        return tensor

    with _Launches() as launches, _StagingGuard(case, chunk, batch), \
            torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        launches.phase = "forward"
        projected = projector.project(image)
        launches.phase = "forward-backward"
        (projected.double() * sino.detach()).sum().backward()
        launches.phase = "adjoint"
        back = projector.backproject(sino)
        launches.phase = "adjoint-backward"
        (back.double() * image.detach()).sum().backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - initial
    assert image.grad.device.type == sino.grad.device.type == "cpu"
    assert strength.grad is not None and torch.isfinite(strength.grad)
    assert all(torch.isfinite(g).all() for g in (projected, back, image.grad, sino.grad))
    _assert_bounded_launches(launches.records, case, chunk, batch,
                            ("forward", "forward-backward", "adjoint", "adjoint-backward"))
    assert any(record[2] == "geometry_vjp" for record in launches.records), "learnable VJP did not dispatch"
    live_limit = 4 * math.prod(chunk) + 16 * batch * math.prod(case.detector)
    assert sum(saved_cuda) <= live_limit, "autograd retained streamed CUDA tiles"
    assert _reachable_cuda(vars(projector)) <= live_limit, "projector cache accumulated CUDA tiles/batches"
    return peak


@pytest.mark.cuda
@cuda_required
def test_peak_cuda_memory_bounded_across_volume_and_view_growth_including_backward():
    small = _memory_trial((24, 24, 24), 5)
    larger_volume = _memory_trial((72, 80, 96), 5)
    more_views = _memory_trial((24, 24, 24), 73)
    # Same tile/view limits. Allow two tile buffers and four endpoint batches
    # of allocator/layout scratch; growth must not track either full input.
    slack = 2 * 4 * 24 ** 3 + 4 * 4 * 3 * 16 * 12 * 3
    assert larger_volume <= small + slack, (small, larger_volume, slack)
    assert more_views <= small + slack, (small, more_views, slack)


@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA GPUs are required")
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("views", [1, 5])
def test_two_gpu_cpu_tiles_view_order_and_gradients(beam, views):
    case = _case(beam, views=views)
    projector = _projector(case, _surface(case, per_view=True), devices=[1, 0],
                           volume_chunk_shape=_chunk(case), view_chunk_size=2)
    image, sino = _data(case, device="cpu")
    image.requires_grad_()
    matrix = _matrix(case, _offsets(case, per_view=True))
    with _Launches() as launches:
        launches.phase = "project"
        output = projector.project(image)
        launches.phase = "backward"
        (output.double() * sino.double()).sum().backward()
    _close(output, (matrix @ _numpy(image).ravel()).reshape(case.sino_shape))
    _close(image.grad, (matrix.T @ _numpy(sino).ravel()).reshape(case.shape))
    _assert_bounded_launches(launches.records, case, _chunk(case), 2,
                            ("project", "backward"), {0, 1})
    assert {record[3] for record in launches.records} == ({1} if views == 1 else {0, 1})


def _nccl_worker(rank, beam, views, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        case = _case(beam, views=views)
        strength = torch.tensor(1., dtype=torch.float64, requires_grad=True)
        projector = _projector(case, _surface(case, strength), distributed=True,
                               volume_chunk_shape=_chunk(case), view_chunk_size=2)
        start = rank * (views // 2) + min(rank, views % 2)
        stop = start + views // 2 + int(rank < views % 2)
        assert projector.view_slice == slice(start, stop)
        assert projector.projection_shape == (stop - start, *case.detector)
        image, sino = _data(case, device="cpu")
        image.requires_grad_()
        local = sino[start:stop].clone().requires_grad_()
        matrix = _matrix(case, _offsets(case))
        local_matrix = matrix[start * math.prod(case.detector):stop * math.prod(case.detector)]
        real_reduce, reductions = dist.all_reduce, []

        def reduction(tensor, *args, **kwargs):
            assert tensor.is_cuda and tensor.device.index == rank
            assert tensor.numel() <= max(math.prod(_chunk(case)), 2 * math.prod(case.detector) * 3, 24)
            reductions.append(tensor.numel())
            return real_reduce(tensor, *args, **kwargs)

        with mock.patch.object(dist, "all_reduce", reduction), _StagingGuard(case, _chunk(case), 2):
            output = projector.project(image)
            _close(output, (local_matrix @ _numpy(image).ravel()).reshape(local.shape))
            (output.double() * local.detach().double()).sum().backward()
            _close(image.grad, (matrix.T @ _numpy(sino).ravel()).reshape(case.shape))
            expected = _finite_difference(lambda s: _objective(case, image, sino, s), 1.)
            assert strength.grad.item() == pytest.approx(expected, rel=5e-3, abs=4e-4)
            strength.grad = None
            back = projector.backproject(local)
            _close(back, (matrix.T @ _numpy(sino).ravel()).reshape(case.shape))
            weight = image.detach() * (rank + 1) / 2
            (back.double() * weight.double()).sum().backward()
            _close(local.grad, (local_matrix @ (_numpy(image) * 1.5).ravel()).reshape(local.shape))
            assert strength.grad.item() == pytest.approx(expected * 1.5, rel=5e-3, abs=4e-4)
        assert reductions, "no production NCCL reduction observed"
    finally:
        dist.destroy_process_group()


@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2 or not dist.is_nccl_available(),
                    reason="Two CUDA GPUs and real NCCL are required")
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("views", [1, 5], ids=["empty-rank", "uneven-ranks"])
def test_nccl_cpu_backed_bounded_collectives_and_sum_semantics(beam, views, tmp_path):
    rendezvous = (tmp_path / "chunk-nccl").resolve().as_uri()
    mp.spawn(_nccl_worker, args=(beam, views, rendezvous), nprocs=2, join=True)


# Automatic defaults below follow frozen automatic-cpu-backed-volume-tiles-v2.
def _memory_inventory(free_by_device):
    """Mock memory observations only. Numerical kernels and outputs stay real."""
    patches = ExitStack()
    allocated = torch.cuda.memory_allocated

    def info(device=None):
        index = torch.cuda.current_device() if device is None else (
            device if isinstance(device, int) else torch.device(device).index)
        index = torch.cuda.current_device() if index is None else index
        return free_by_device[index], torch.cuda.get_device_properties(index).total_memory

    patches.enter_context(mock.patch.object(torch.cuda, "mem_get_info", side_effect=info))
    # Exclude pre-existing reusable cache from this deliberately constrained
    # inventory; otherwise earlier tests could make a small free-memory mock fit.
    patches.enter_context(mock.patch.object(torch.cuda, "memory_reserved", side_effect=allocated))
    return patches


def _chosen_shape(records, beam, device=None):
    shapes = [record[-2][::-1] if beam == "cone" else record[-2]
              for record in records if device is None or record[3] == device]
    assert shapes
    return tuple(max(axis) for axis in zip(*shapes))


def _halving_shapes(shape):
    """Allowed largest-axis halving states; permit either integer rounding."""
    seen, pending = {shape}, [shape]
    while pending:
        current = pending.pop()
        largest = max(current)
        if largest == 1:
            continue
        for axis, size in enumerate(current):
            if size != largest:
                continue
            for half in {max(1, size // 2), (size + 1) // 2}:
                changed = list(current)
                changed[axis] = half
                changed = tuple(changed)
                if changed not in seen:
                    seen.add(changed)
                    pending.append(changed)
    return seen


def _chords(case, offsets):
    """Independent whole-box chord reference for a constant-one volume."""
    geometry = tuple(_numpy(g) for g in case.trajectory)
    offsets = np.broadcast_to(offsets, (*case.sino_shape, 3))
    lower, upper = -np.asarray(case.shape[::-1]) * case.spacing / 2, np.asarray(case.shape[::-1]) * case.spacing / 2
    result = np.zeros(case.sino_shape, dtype=np.float64)
    for index in np.ndindex(case.sino_shape):
        view = index[0]
        u = geometry[2][view]
        normal = np.cross(u, geometry[3][view]) if case.beam == "cone" else np.array([u[1], -u[0]])
        point = geometry[1][view] + offsets[index][0] * u + offsets[index][2] * normal
        if case.beam == "cone":
            point = point + offsets[index][1] * geometry[3][view]
        origin = point if case.beam == "parallel" else geometry[0][view]
        direction = geometry[0][view] if case.beam == "parallel" else point - origin
        enter, leave = (-np.inf, np.inf) if case.beam == "parallel" else (0., 1.)
        for p, d, lo, hi in zip(origin, direction, lower, upper):
            if d == 0:
                if not lo <= p < hi:
                    leave = -np.inf
            else:
                crossings = (lo - p) / d, (hi - p) / d
                enter, leave = max(enter, min(crossings)), min(leave, max(crossings))
        result[index] = max(leave - enter, 0) * np.linalg.norm(direction)
    return result


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("curved", [False, True])
def test_default_cpu_oracle_dtype_gradients_and_real_cuda_dispatch(beam, curved):
    case = _case(beam, views=5)
    surface = _surface(case, per_view=True) if curved else None
    projector = _projector(case, surface)
    image, sino = (_noncontiguous(t.double()).requires_grad_() for t in _data(case, device="cpu"))
    matrix = _matrix(case, _offsets(case, per_view=curved, flat=not curved))
    with _Launches() as launches:
        launches.phase = "project"
        projected = projector.project(image)
        launches.phase = "project-backward"
        (projected.double() * sino.detach()).sum().backward()
        launches.phase = "backproject"
        back = projector.backproject(sino)
        launches.phase = "backproject-backward"
        (back.double() * image.detach()).sum().backward()
    for output in (projected, back):
        assert output.device.type == "cpu" and output.dtype == torch.float32
    for gradient in (image.grad, sino.grad):
        assert gradient.device.type == "cpu" and gradient.dtype == torch.float64
    _close(projected, (matrix @ _numpy(image).ravel()).reshape(case.sino_shape))
    _close(back, (matrix.T @ _numpy(sino).ravel()).reshape(case.shape))
    _close(image.grad, _numpy(back))
    _close(sino.grad, _numpy(projected))
    _assert_bounded_launches(launches.records, case, case.shape, 32,
                            ("project", "project-backward", "backproject", "backproject-backward"))


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
def test_fitting_default_cuda_keeps_full_path_and_view_only_override_streams(beam):
    case = _case(beam, views=5)
    image, sino = _data(case)
    full = _projector(case)
    view_only = _projector(case, view_chunk_size=2)
    device = torch.cuda.current_device()
    with _memory_inventory({device: 8 * 1024 ** 3}), _Launches() as launches:
        launches.phase = "full-project"
        expected = full.project(image)
        launches.phase = "full-adjoint"
        expected_back = full.backproject(sino)
        launches.phase = "view-project"
        projected = view_only.project(image)
        launches.phase = "view-adjoint"
        back = view_only.backproject(sino)
    for phase in ("full-project", "full-adjoint"):
        records = [r for r in launches.records if r[0] == phase]
        assert len(records) == 1 and records[0][-1][0] == case.views
        assert _chosen_shape(records, beam) == case.shape
    _assert_bounded_launches(launches.records, case, case.shape, 2, ("view-project", "view-adjoint"))
    assert projected.device == back.device == image.device
    _close(projected, _numpy(expected))
    _close(back, _numpy(expected_back))


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("input_device", ["cpu", "cuda"])
def test_constrained_automatic_budget_halves_largest_axis_and_matches_oracle(input_device):
    case = _case("cone", views=5)
    case.shape = (17, 19, 23)
    projector = _projector(case, _surface(case, per_view=True))
    image, sino = _data(case, device=input_device)
    matrix = _matrix(case, _offsets(case, per_view=True))
    device, free = torch.cuda.current_device(), 64 * 1024
    with _memory_inventory({device: free}), _Launches() as launches:
        launches.phase = "project"
        projected = projector.project(image)
        launches.phase = "adjoint"
        back = projector.backproject(sino)
    chosen = _chosen_shape(launches.records, case.beam)
    assert chosen != case.shape, "constrained automatic path staged the full volume"
    assert chosen in _halving_shapes(case.shape), chosen
    # One float32 tile alone cannot exceed the usable 75-percent budget.
    assert math.prod(chosen) * 4 <= free * .75
    assert projected.device == image.device and back.device == sino.device
    _close(projected, (matrix @ _numpy(image).ravel()).reshape(case.sino_shape))
    _close(back, (matrix.T @ _numpy(sino).ravel()).reshape(case.shape))


@pytest.mark.cuda
@cuda_required
def test_automatic_view_batch_shrinks_for_endpoint_and_vjp_scratch():
    case = _case("cone", views=35, learnable=True)
    case.shape, case.detector = (5, 7, 9), (24, 20)
    strength = torch.tensor(.21, dtype=torch.float64, requires_grad=True)
    projector = _projector(case, _surface(case, strength, per_view=True))
    image = torch.ones(case.shape, dtype=torch.float64, requires_grad=True)
    sino = torch.ones(case.sino_shape, dtype=torch.float64, requires_grad=True)
    free, device = 64 * 1024, torch.cuda.current_device()
    maximum = int(free * .75 // (math.prod(case.detector) * 3 * 4))
    with _memory_inventory({device: free}), _Launches() as launches, _StagingGuard(case, case.shape, maximum):
        launches.phase = "project"
        projected = projector.project(image)
        launches.phase = "project-backward"
        projected.double().sum().backward()
        launches.phase = "adjoint"
        back = projector.backproject(sino)
        launches.phase = "adjoint-backward"
        back.double().sum().backward()
    _assert_bounded_launches(launches.records, case, case.shape, maximum,
                            ("project", "project-backward", "adjoint", "adjoint-backward"))
    assert max(record[-1][0] for record in launches.records) < 32
    _close(projected, _chords(case, _offsets(case, .21, per_view=True)))
    assert back.sum().item() == pytest.approx(projected.sum().item(), rel=4e-5, abs=4e-4)
    assert image.grad.device.type == sino.grad.device.type == "cpu"
    assert strength.grad is not None and torch.isfinite(strength.grad)


@pytest.mark.cuda
@cuda_required
def test_automatic_gpu_surface_constructor_cpu_validation_and_early_stream_rejection():
    case = _case("cone", views=5)

    def callback(u, v):
        return _flat_surface(u, v).cuda()

    # Constructor may accept GPU offsets for a later fitting full CUDA call,
    # but its validation must never expand the complete CUDA world geometry.
    with _StagingGuard(case, case.shape, 1):
        projector = _projector(case, callback)
    with _StagingGuard(case, case.shape, 1), pytest.raises((TypeError, ValueError), match="CPU|cpu"):
        projector.project(torch.ones(case.shape))
    image = torch.ones(case.shape, device="cuda")
    device = torch.cuda.current_device()
    with _memory_inventory({device: 8 * 1024 ** 3}), _Launches() as launches:
        launches.phase = "full"
        output = projector.project(image)
    _close(output, _chords(case, _offsets(case, flat=True)))
    assert len(launches.records) == 1
    with _memory_inventory({device: 4096}), _StagingGuard(case, case.shape, 1), \
            pytest.raises((TypeError, ValueError), match="CPU|cpu"):
        projector.project(image)


@pytest.mark.cuda
@cuda_required
def test_automatic_budget_too_small_for_one_view_fails_clearly():
    case = _case("cone")
    projector = _projector(case, _surface(case))
    with _memory_inventory({torch.cuda.current_device(): 64}), \
            pytest.raises((RuntimeError, ValueError), match="memory|budget|fit|voxel"):
        projector.project(torch.ones(case.shape))


def _fraction_worker(rank, ceiling_bytes):
    torch.cuda.set_device(0)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(ceiling_bytes / total, 0)
    case = _case("cone", views=5)
    case.shape = (144, 152, 160)
    projector = _projector(case, _surface(case, per_view=True))
    image = torch.ones(case.shape, dtype=torch.float64, requires_grad=True)
    assert image.numel() * 4 > ceiling_bytes, "full float32 volume must exceed the real allocator ceiling"
    torch.cuda.reset_peak_memory_stats()
    with _Launches() as launches:
        launches.phase = "project"
        output = projector.project(image)
        launches.phase = "backward"
        output.double().sum().backward()
    _close(output, _chords(case, _offsets(case, per_view=True)))
    assert image.grad.device.type == "cpu" and torch.isfinite(image.grad).all()
    assert _chosen_shape(launches.records, case.beam) != case.shape
    assert torch.cuda.max_memory_allocated() <= ceiling_bytes
    _assert_bounded_launches(launches.records, case, case.shape, 32, ("project", "backward"))


@pytest.mark.cuda
@cuda_required
def test_actual_per_process_fraction_constrains_default_streaming_in_isolated_process():
    # Fresh process: real allocator ceiling, no global/device configuration or
    # numerical mocks, no unrelated CUDA cache/process touched.
    mp.spawn(_fraction_worker, args=(12 * 1024 ** 2,), nprocs=1, join=True)


@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA GPUs are required")
def test_automatic_two_gpu_common_tile_uses_smallest_asymmetric_budget():
    case = _case("cone", views=7)
    case.shape = (17, 19, 23)
    projector = _projector(case, _surface(case, per_view=True), devices=[1, 0])
    image, sino = _data(case, device="cpu")
    matrix = _matrix(case, _offsets(case, per_view=True))
    with _memory_inventory({0: 1024 ** 2, 1: 64 * 1024}), _Launches() as launches:
        launches.phase = "project"
        output = projector.project(image)
        launches.phase = "adjoint"
        back = projector.backproject(sino)
    first, second = (_chosen_shape(launches.records, case.beam, device) for device in (0, 1))
    assert first == second and first != case.shape
    assert first in _halving_shapes(case.shape)
    assert math.prod(first) * 4 <= .75 * 64 * 1024
    _close(output, (matrix @ _numpy(image).ravel()).reshape(case.sino_shape))
    _close(back, (matrix.T @ _numpy(sino).ravel()).reshape(case.shape))


def _remaining_room_worker(rank):
    from diffct import circular_trajectory_3d

    torch.cuda.set_device(0)
    ceiling = 32 * 1024 ** 2
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(ceiling / total, 0)
    # Keep another caller's live allocation throughout both operations. The
    # process ceiling exceeds the native allocator's large-pool segment, but
    # its remaining room does not; small streamed allocations are still viable.
    held = torch.empty(22 * 1024 ** 2, dtype=torch.uint8, device="cuda:0")
    held.fill_(41)
    geometry = circular_trajectory_3d(5, sid=240., sdd=480., device="cpu")
    case = _Case("cone", geometry, (144, 152, 160), (9, 7), (1., 1.), spacing=1.)
    projector = _projector(case)
    image = torch.ones(case.shape, dtype=torch.float64, requires_grad=True)
    cotangent = torch.linspace(.17, .83, math.prod(case.sino_shape), dtype=torch.float64).reshape(case.sino_shape)
    expected = _chords(case, _offsets(case, flat=True))
    initial = torch.cuda.memory_allocated()
    remaining = ceiling - initial
    assert initial >= held.numel() and 0 < remaining < 12 * 1024 ** 2
    torch.cuda.reset_peak_memory_stats()
    with _Launches() as launches:
        launches.phase = "project"
        output = projector.project(image)
        launches.phase = "backward"
        (output.double() * cotangent).sum().backward()
    torch.cuda.synchronize()
    assert output.device.type == "cpu" and output.dtype == torch.float32
    assert image.grad.device.type == "cpu" and image.grad.dtype == torch.float64
    _close(output, expected)
    # A constant-one volume's chord sum independently determines the total
    # matched input gradient, without materializing an enormous cell matrix.
    assert image.grad.sum().item() == pytest.approx(
        float(np.sum(expected * cotangent.numpy())), rel=4e-5, abs=4e-4)
    assert torch.isfinite(image.grad).all() and image.grad.abs().sum().item() > .01
    assert torch.cuda.max_memory_allocated() <= ceiling
    chosen = _chosen_shape(launches.records, case.beam)
    assert math.prod(chosen) * 4 <= remaining
    _assert_bounded_launches(launches.records, case, chosen, 32, ("project", "backward"))
    assert held[0].item() == held[-1].item() == 41


@pytest.mark.cuda
@cuda_required
def test_default_cpu_streaming_respects_live_allocator_remaining_room():
    # Isolated real allocator; no memory observations, kernels or outputs mocked.
    mp.spawn(_remaining_room_worker, nprocs=1, join=True)


@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA GPUs are required")
def test_automatic_cpu_empty_gpu_shard_ignores_idle_zero_budget():
    case = _case("cone", views=1)
    projector = _projector(case, devices=[0, 1])
    image, sino = (t.double().requires_grad_() for t in _data(case, device="cpu"))
    matrix = _matrix(case, _offsets(case, flat=True))
    # With one view, card 1 receives no rays. Its observed lack of free memory
    # must not constrain spatial tiles or prevent card 0 from executing.
    with _memory_inventory({0: 32 * 1024 ** 2, 1: 0}), _Launches() as launches:
        launches.phase = "project"
        projected = projector.project(image)
        launches.phase = "project-backward"
        (projected.double() * sino.detach()).sum().backward()
        launches.phase = "adjoint"
        back = projector.backproject(sino)
        launches.phase = "adjoint-backward"
        (back.double() * image.detach()).sum().backward()
    _close(projected, (matrix @ _numpy(image).ravel()).reshape(case.sino_shape))
    _close(back, (matrix.T @ _numpy(sino).ravel()).reshape(case.shape))
    _close(image.grad, _numpy(back))
    _close(sino.grad, _numpy(projected))
    assert projected.device.type == back.device.type == "cpu"
    assert image.grad.dtype == sino.grad.dtype == torch.float64
    _assert_bounded_launches(launches.records, case, case.shape, 1,
                            ("project", "project-backward", "adjoint", "adjoint-backward"), {0})
    assert {record[3] for record in launches.records} == {0}


@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA GPUs are required")
def test_two_gpu_full_path_counts_total_cuda_surface_world_sampling_before_expansion():
    case = _case("cone", views=8)
    case.shape, case.detector = (2, 3, 4), (24, 20)

    def callback(u, v):
        # Shared float64 offsets are caller-provided. World positions are
        # sampled for all eight views on this device before four-view dispatch.
        return _flat_surface(u, v).to(device="cuda:0")

    class NoTotalWorldExpansion(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            result = function(*args, **(kwargs or {}))
            for tensor in tree_flatten(result)[0]:
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    assert tuple(tensor.shape) != (*case.sino_shape, 3), \
                        "full CUDA world surface expanded before the memory decision"
            return result

    with NoTotalWorldExpansion():
        projector = _projector(case, callback, devices=[0, 1])
    image, _ = _data(case, device="cuda:0")
    matrix = _matrix(case, _offsets(case, flat=True))
    # 400 KiB observations give 300 KiB after the documented reserve. A
    # four-view estimate fits, but eager float64 world-point arithmetic for
    # eight views has overlapping full tensors exceeding that usable room.
    with _memory_inventory({0: 400 * 1024, 1: 400 * 1024}), NoTotalWorldExpansion(), \
            _Launches() as denied:
        with pytest.raises((TypeError, ValueError), match="CPU|cpu"):
            projector.project(image)
    assert not denied.records, "unaffordable GPU surface should fail before any ray kernel"
    # The same callback remains usable when actual full-world sampling fits.
    with _memory_inventory({0: 32 * 1024 ** 2, 1: 32 * 1024 ** 2}), _Launches() as launches:
        launches.phase = "project"
        output = projector.project(image)
    _close(output, (matrix @ _numpy(image).ravel()).reshape(case.sino_shape))
    assert output.device == image.device
    _assert_bounded_launches(launches.records, case, case.shape, 4, ("project",), {0, 1})
    assert {record[3] for record in launches.records} == {0, 1}
