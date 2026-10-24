"""Bounded autograd memory on the single-process portable (PyTorch) backend.

A1: data gradients use the matched adjoint; the state autograd keeps after a
forward call does not grow with the number of internal execution batches.
A2: first-order trajectory and ParameterizedSurface parameter gradients keep
their values (central finite differences, agreement across chunk settings),
with the same bounded saved state; surface admission still runs.
A3: geometry second derivatives and forward-mode derivatives with respect to
trajectory tensors raise a clear error instead of returning silent values.

Two memory observers are used. ``saved_tensors_hooks`` sums the storages that
autograd packs during the forward call. A dispatch observer additionally
measures every storage created during forward that the returned graph keeps
alive, which includes tensors attached to autograd contexts as attributes.
"""

from __future__ import annotations

import gc
import math
import re
import weakref

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from diffct import ParameterizedSurface, Projector, circular_trajectory_3d
from tests._phantoms import shepp_logan_3d
from tests.test_reference_backend import BEAMS, _case, _data, _matrix


CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

# Unsupported-derivative errors must name the limitation, not a generic
# autograd failure such as "does not require grad".
SECOND_ORDER_MESSAGE = (
    r"(?i)(second|higher|double)[- ]?(order|deriv|backward)|not supported|unsupported"
)
FORWARD_MODE_MESSAGE = r"(?i)forward[- ]?mode|jvp|jacfwd|not supported|unsupported"
DERIVATIVE_ERRORS = (RuntimeError, NotImplementedError, ValueError)


# ---------------------------------------------------------------------------
# Geometry and operator construction
# ---------------------------------------------------------------------------

_SIZES = {
    # beam: (volume shape, detector shape, views, spacing, detector pitch)
    "fan": ((40, 40), (56,), 16, (0.9, 0.8), 0.95),
    "cone": ((12, 16, 16), (20, 14), 8, (0.9, 0.8, 0.85), (0.95, 0.9)),
}
_SMALL_SIZES = {
    "fan": ((8, 9), (7,), 5, (0.9, 0.8), 0.95),
    "cone": ((4, 6, 5), (5, 4), 4, (0.9, 0.8, 0.85), (0.95, 0.9)),
}


def _trajectory(beam, parameters):
    """Differentiable fan/cone trajectory from per-view angles and shifts.

    Axes stay unit length and mutually orthogonal for every parameter value.
    Offsets and the angle phase keep rays away from exact voxel faces.
    """
    angles, source_shift, center_shift = parameters
    radial = torch.stack((angles.cos(), angles.sin()), dim=-1)
    tangent = torch.stack((-angles.sin(), angles.cos()), dim=-1)
    if beam == "fan":
        return (-30 * radial + source_shift, 20 * radial + center_shift, tangent)
    tilt = 0.071
    views = angles.shape[0]
    radial = torch.cat((radial * math.cos(tilt),
                        torch.full((views, 1), math.sin(tilt), dtype=angles.dtype)), dim=-1)
    axis_u = torch.cat((tangent, torch.zeros((views, 1), dtype=angles.dtype)), dim=-1)
    axis_v = torch.linalg.cross(radial, axis_u, dim=-1)
    return (-30 * radial + source_shift, 20 * radial + center_shift, axis_u, axis_v)


def _trajectory_parameters(beam, views, *, dtype=torch.float64, seed=0):
    generator = torch.Generator().manual_seed(1231 + seed)
    dimensions = 2 if beam == "fan" else 3
    angles = (torch.arange(views, dtype=torch.float64) * (2 * math.pi / views) + 0.137
              + 0.01 * torch.rand(views, generator=generator, dtype=torch.float64))
    source_shift = 0.4 * (torch.rand(views, dimensions, generator=generator,
                                     dtype=torch.float64) - 0.5)
    center_shift = 0.4 * (torch.rand(views, dimensions, generator=generator,
                                     dtype=torch.float64) - 0.5)
    return tuple(value.to(dtype) for value in (angles, source_shift, center_shift))


def _chunks(setting, shape, detector, views):
    if setting == "few":
        return dict(view_chunk_size=views, volume_chunk_shape=tuple(shape),
                    detector_chunk_shape=tuple(detector))
    if setting == "many":
        # One view per batch and half-size tiles: tens to hundreds of batches.
        return dict(view_chunk_size=1,
                    volume_chunk_shape=tuple(-(-n // 2) for n in shape),
                    detector_chunk_shape=tuple(-(-n // 2) for n in detector))
    raise AssertionError(setting)


def _operator(beam, setting, *, sizes=_SIZES, precision="float64", trajectory=None,
              detector_surface=None):
    shape, detector, views, spacing, pitch = sizes[beam]
    dtype = torch.float64 if precision == "float64" else torch.float32
    if trajectory is None:
        trajectory = _trajectory(beam, _trajectory_parameters(beam, views, dtype=dtype))
    operator = Projector(
        trajectory, shape, detector, beam=beam, detector_spacing=pitch,
        voxel_spacing=spacing, backend="torch", precision=precision,
        detector_surface=detector_surface, **_chunks(setting, shape, detector, views),
    )
    assert operator.backend == "torch"
    return operator


def _random(shape, *, seed, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=generator, dtype=torch.float64).to(dtype) + 0.1


def _tensor_bytes(values):
    return sum(value.numel() * value.element_size() for value in values)


def _assert_relative(actual, expected, tolerance, label=""):
    actual, expected = actual.detach().double(), expected.detach().double()
    scale = expected.abs().max().clamp_min(1e-300)
    error = (actual - expected).abs().max() / scale
    assert error <= tolerance, f"{label}: relative max error {error.item():.3g} > {tolerance}"


# ---------------------------------------------------------------------------
# Memory observers
# ---------------------------------------------------------------------------

def _physical(tensor):
    """Strip functorch wrappers so storage sizes are physical bytes."""
    functorch = torch._C._functorch
    while functorch.is_batchedtensor(tensor) or functorch.is_gradtrackingtensor(tensor):
        tensor = functorch.get_unwrapped(tensor)
    return tensor


def _storage(tensor):
    """Physical (data pointer, bytes); (0, 0) for tensors that own no memory."""
    tensor = _physical(tensor)
    if tensor._is_zerotensor():  # forward-mode zero tangents
        return 0, 0
    storage = tensor.untyped_storage()
    return storage.data_ptr(), storage.nbytes()


def _saved_bytes(function):
    """Run ``function`` and sum distinct storages packed by autograd."""
    storages = {}

    def pack(tensor):
        pointer, size = _storage(tensor)
        if pointer:
            storages[pointer] = size
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        result = function()
    return sum(storages.values()), result


class _CreatedTensors(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.references = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for value in tree_leaves(result):
            if isinstance(value, torch.Tensor):
                self.references.append(weakref.ref(value))
        return result


def _live_bytes(references, excluded):
    gc.collect()
    storages = {}
    for reference in references:
        tensor = reference()
        if tensor is None:
            continue
        pointer, size = _storage(tensor)
        if pointer and pointer not in excluded:
            storages[pointer] = size
    return sum(storages.values())


def _graph_bytes(function, caller):
    """Bytes created in forward and kept alive only by the returned graph."""
    observer = _CreatedTensors()
    with observer:
        outputs = function()
    outputs = (outputs,) if isinstance(outputs, torch.Tensor) else tuple(outputs)
    excluded = {_storage(value)[0] for value in (*caller, *outputs)}
    with_graph = _live_bytes(observer.references, excluded)
    outputs = tuple(value.detach() for value in outputs)
    without_graph = _live_bytes(observer.references, excluded)
    return with_graph - without_graph


def _memory(beam, operation, setting, *, geometry_grad=False, surface=False):
    """Return saved bytes, graph-retained bytes and the size allowance."""
    shape, detector, views, _, _ = _SIZES[beam]
    parameters = tuple(value.clone().requires_grad_(geometry_grad)
                       for value in _trajectory_parameters(beam, views))
    trajectory = _trajectory(beam, parameters)
    detector_surface, surface_parameters = None, ()
    if surface:
        detector_surface, surface_parameters = _surface(views)
    operator = _operator(beam, setting, trajectory=tuple(t.detach() for t in trajectory),
                         detector_surface=detector_surface)
    input_shape = shape if operation == "project" else operator.projection_shape
    data = _random(input_shape, seed=7).requires_grad_()
    kwargs = {"trajectory": trajectory} if geometry_grad else {}

    def forward():
        return getattr(operator, operation)(data, **kwargs)

    saved, output = _saved_bytes(forward)
    graph = _graph_bytes(forward, (data, *trajectory, *surface_parameters))
    # The loss must reach every requested leaf through the bounded graph.
    leaves = (data, *(parameters if geometry_grad else ()), *surface_parameters)
    gradients = torch.autograd.grad((output * _random(output.shape, seed=8)).sum(), leaves)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    geometry_bytes = _tensor_bytes(trajectory) + _tensor_bytes(surface_parameters)
    allowance = 4 * (data.numel() + output.numel()) * 8 + 16 * geometry_bytes + 4096
    return saved, graph, allowance, geometry_bytes


# ---------------------------------------------------------------------------
# A1: data gradients and bounded saved state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_data_gradient_saved_state_is_bounded_and_batch_independent(beam, operation):
    results = {setting: _memory(beam, operation, setting) for setting in ("few", "many")}
    for setting, (saved, graph, allowance, _) in results.items():
        assert saved <= allowance, (
            f"{setting}: autograd saved {saved} bytes; bound is {allowance} bytes")
        assert graph <= allowance, (
            f"{setting}: graph keeps {graph} bytes alive; bound is {allowance} bytes")
    geometry_bytes = results["few"][3]
    slack = geometry_bytes + 4096
    assert abs(results["many"][0] - results["few"][0]) <= slack, results
    assert abs(results["many"][1] - results["few"][1]) <= slack, results


@pytest.mark.parametrize("setting", ["few", "many"])
@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_data_gradients_equal_matched_adjoint(beam, setting):
    operator = _operator(beam, setting)
    image = _random(operator.volume_shape, seed=1).requires_grad_()
    rays = _random(operator.projection_shape, seed=2).requires_grad_()
    image_weight = _random(operator.volume_shape, seed=3)
    ray_weight = _random(operator.projection_shape, seed=4)

    projection = operator.project(image)
    image_gradient, = torch.autograd.grad((projection * ray_weight).sum(), image)
    adjoint = operator.backproject(ray_weight)
    _assert_relative(image_gradient, adjoint, 1e-10, "grad <Ax,w>")

    backprojection = operator.backproject(rays)
    ray_gradient, = torch.autograd.grad((backprojection * image_weight).sum(), rays)
    _assert_relative(ray_gradient, operator.project(image_weight), 1e-10, "grad <A^T y,v>")

    # Matched adjoint identity <A x, w> == <x, A^T w>.
    left = (projection.detach() * ray_weight).sum()
    right = (image.detach() * adjoint).sum()
    _assert_relative(left, right, 1e-10, "<Ax,w> == <x,A^T w>")


@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_float32_data_gradient_equals_matched_adjoint(beam):
    operator = _operator(beam, "many", precision="float32")
    assert operator.dtype == torch.float32
    image = _random(operator.volume_shape, seed=1, dtype=torch.float32).requires_grad_()
    weight = _random(operator.projection_shape, seed=4, dtype=torch.float32)
    gradient, = torch.autograd.grad((operator.project(image) * weight).sum(), image)
    _assert_relative(gradient, operator.backproject(weight), 1e-5, "float32 grad")


def _tiny_operator(beam, setting):
    case = _case(beam)
    if setting == "few":
        chunks = {}
    else:
        chunks = dict(view_chunk_size=1, detector_chunk_shape=(2,) * len(case.detector),
                      volume_chunk_shape=(2,) * len(case.shape))
    operator = Projector(
        case.trajectory, case.shape, case.detector, beam=case.beam,
        detector_spacing=case.pitch if case.beam == "cone" else case.pitch[0],
        voxel_spacing=case.spacing, backend="torch", precision="float64", **chunks,
    )
    return case, operator


@pytest.mark.parametrize("setting", ["few", "many"])
@pytest.mark.parametrize("beam", BEAMS)
def test_higher_order_data_derivatives_and_transforms_match_linear_map(beam, setting):
    case, operator = _tiny_operator(beam, setting)
    matrix = _matrix(case)
    image_shape, ray_shape = case.shape, case.projection_shape
    image, tangent = _data(image_shape), _data(image_shape, phase=0.79)
    weight = _data(ray_shape, phase=0.31)
    measured = _data(ray_shape, phase=1.7)
    gram = matrix.T @ matrix

    def linear(x):
        return (operator.project(x) * weight).sum()

    def quadratic(x):
        return (operator.project(x) - measured).square().sum() / 2

    _assert_relative(torch.func.grad(linear)(image),
                     (matrix.T @ weight.flatten()).reshape(image_shape), 1e-10, "func.grad")

    primal, derivative = torch.func.jvp(operator.project, (image,), (tangent,))
    _assert_relative(primal, (matrix @ image.flatten()).reshape(ray_shape), 1e-10, "jvp primal")
    _assert_relative(derivative, (matrix @ tangent.flatten()).reshape(ray_shape), 1e-10, "jvp")

    batch = torch.stack((image, tangent, image - 2 * tangent))
    _assert_relative(torch.func.vmap(operator.project)(batch),
                     (batch.reshape(3, -1) @ matrix.T).reshape(3, *ray_shape), 1e-10, "vmap")

    hessian = torch.func.jacfwd(torch.func.grad(quadratic))(image)
    _assert_relative(hessian, gram.reshape(*image_shape, *image_shape), 1e-10,
                     "jacfwd(grad)")

    # Hessian-vector product through ordinary double backward.
    leaf = image.clone().requires_grad_()
    first, = torch.autograd.grad(quadratic(leaf), leaf, create_graph=True)
    _assert_relative(first, (matrix.T @ (matrix @ image.flatten() - measured.flatten()))
                     .reshape(image_shape), 1e-10, "first derivative")
    second, = torch.autograd.grad((first * tangent).sum(), leaf)
    _assert_relative(second, (gram @ tangent.flatten()).reshape(image_shape), 1e-10, "HVP")

    # Double backward through data: d/dw of (d<Ax, w>/dx . t) is A t.
    leaf = image.clone().requires_grad_()
    ray_weight = weight.clone().requires_grad_()
    first, = torch.autograd.grad((operator.project(leaf) * ray_weight).sum(), leaf,
                                 create_graph=True)
    mixed, = torch.autograd.grad((first * tangent).sum(), ray_weight)
    _assert_relative(mixed, (matrix @ tangent.flatten()).reshape(ray_shape), 1e-10,
                     "double backward through data")

    # The same identities for the adjoint operator.
    rays, ray_tangent = _data(ray_shape, phase=0.5), _data(ray_shape, phase=1.1)
    _, adjoint_derivative = torch.func.jvp(operator.backproject, (rays,), (ray_tangent,))
    _assert_relative(adjoint_derivative, (matrix.T @ ray_tangent.flatten()).reshape(image_shape),
                     1e-10, "backproject jvp")
    leaf = rays.clone().requires_grad_()
    first, = torch.autograd.grad(operator.backproject(leaf).square().sum() / 2, leaf,
                                 create_graph=True)
    second, = torch.autograd.grad((first * ray_tangent).sum(), leaf)
    _assert_relative(second, (matrix @ matrix.T @ ray_tangent.flatten()).reshape(ray_shape),
                     1e-10, "backproject HVP")


@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_hessian_vector_product_matches_finite_difference_of_gradient(beam):
    operator = _operator(beam, "many", sizes=_SMALL_SIZES)
    image = _random(operator.volume_shape, seed=5)
    direction = _random(operator.volume_shape, seed=6) - 0.6
    measured = _random(operator.projection_shape, seed=9)

    def gradient(x):
        leaf = x.clone().requires_grad_()
        value, = torch.autograd.grad(
            (operator.project(leaf) - measured).square().sum() / 2, leaf, create_graph=True)
        return leaf, value

    leaf, first = gradient(image)
    hvp, = torch.autograd.grad((first * direction).sum(), leaf)
    # The gradient is affine in the image, so a central difference is exact up
    # to rounding for any step.
    step = 0.5
    finite = (gradient(image + step * direction)[1] - gradient(image - step * direction)[1]) / (2 * step)
    _assert_relative(hvp, finite.detach(), 1e-10, "HVP vs finite difference")


# Transient CUDA work of one portable call may use this fraction of the device
# memory that is usable when the call starts (work-budget contract C1, stated
# in docs/EXECUTION.md). Caller inputs and returned outputs are excluded.
CUDA_WORK_FRACTION = 0.25


def _usable_cuda_bytes(device):
    free, _ = torch.cuda.mem_get_info(device)
    cached = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    return free + cached


@pytest.mark.cuda
@CUDA_REQUIRED
def test_cuda_shepp_logan_256_cone_forward_backward_has_bounded_memory():
    device = torch.device("cuda")
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    volume = torch.from_numpy(shepp_logan_3d(256)).to(device).requires_grad_()
    trajectory = circular_trajectory_3d(64, sid=600.0, sdd=1000.0, start_angle=0.1,
                                        device=device, dtype=torch.float32)
    operator = Projector(trajectory, (256, 256, 256), (768, 512), beam="cone",
                         detector_spacing=1.0, voxel_spacing=(1.0, 1.0, 1.0),
                         backend="torch", precision="float32")
    assert operator.backend == "torch"
    input_bytes = volume.numel() * volume.element_size()
    geometry_bytes = _tensor_bytes(trajectory)

    # Forward: the transient peak stays within the documented work budget and
    # the memory kept after the call is the output plus bounded saved state.
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    usable = _usable_cuda_bytes(device)
    torch.cuda.reset_peak_memory_stats(device)
    projection = operator.project(volume)
    torch.cuda.synchronize(device)
    assert projection.shape == (64, 768, 512)
    output_bytes = projection.numel() * projection.element_size()
    saved_state = torch.cuda.memory_allocated(device) - baseline - output_bytes
    peak_work = torch.cuda.max_memory_allocated(device) - baseline - output_bytes
    saved_bound = 4 * (input_bytes + output_bytes) + 16 * geometry_bytes + 4096
    work_bound = CUDA_WORK_FRACTION * usable
    assert saved_state <= saved_bound, (
        f"after forward {saved_state} bytes besides the output stay allocated; "
        f"bound is {saved_bound} bytes (input {input_bytes}, output {output_bytes})")
    assert peak_work <= work_bound, (
        f"forward transient peak {peak_work} bytes; work budget is {work_bound:.0f} bytes "
        f"({CUDA_WORK_FRACTION} of {usable} usable bytes)")

    # Backward: one adjoint call with the same work budget. Its returned arrays
    # are the volume gradient and, at most, a materialized upstream gradient.
    loss = projection.sum()
    torch.cuda.synchronize(device)
    before_backward = torch.cuda.memory_allocated(device)
    usable = _usable_cuda_bytes(device)
    torch.cuda.reset_peak_memory_stats(device)
    loss.backward()
    torch.cuda.synchronize(device)
    backward_work = (torch.cuda.max_memory_allocated(device) - before_backward
                     - input_bytes - output_bytes)
    work_bound = CUDA_WORK_FRACTION * usable
    assert backward_work <= work_bound, (
        f"backward transient peak {backward_work} bytes; work budget is {work_bound:.0f} "
        f"bytes ({CUDA_WORK_FRACTION} of {usable} usable bytes)")
    assert volume.grad is not None and volume.grad.shape == volume.shape
    assert torch.isfinite(volume.grad).all()
    del projection, loss
    with torch.no_grad():
        adjoint = operator.backproject(torch.ones((64, 768, 512), device=device))
    _assert_relative(volume.grad, adjoint, 1e-5, "Shepp-Logan 256 gradient")


# ---------------------------------------------------------------------------
# A2: geometry first-order gradients
# ---------------------------------------------------------------------------

def _geometry_loss(beam, setting, sizes=_SIZES, *, operation="project"):
    """Return L(parameters) = <op_theta(data), weight> and the parameters."""
    shape, detector, views, _, _ = sizes[beam]
    base = _trajectory_parameters(beam, views)
    operator = _operator(beam, setting, sizes=sizes)
    input_shape = shape if operation == "project" else operator.projection_shape
    output_shape = operator.projection_shape if operation == "project" else shape
    data, weight = _random(input_shape, seed=11), _random(output_shape, seed=12)

    def loss(parameters):
        output = getattr(operator, operation)(data, trajectory=_trajectory(beam, parameters))
        return (output * weight).sum()

    return loss, base


def _autograd_gradient(loss, base):
    leaves = tuple(value.clone().requires_grad_() for value in base)
    return torch.autograd.grad(loss(leaves), leaves)


def _central_differences(loss, base, step=1e-6):
    gradients = []
    for index, value in enumerate(base):
        gradient = torch.zeros_like(value)
        for position in range(value.numel()):
            values = []
            for sign in (1.0, -1.0):
                shifted = [v.clone() for v in base]
                shifted[index].view(-1)[position] += sign * step
                with torch.no_grad():
                    values.append(loss(tuple(shifted)))
            gradient.view(-1)[position] = (values[0] - values[1]) / (2 * step)
        gradients.append(gradient)
    return gradients


@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_trajectory_gradients_match_finite_differences_and_chunking(beam, operation):
    results = {}
    for setting in ("few", "many"):
        loss, base = _geometry_loss(beam, setting, _SMALL_SIZES, operation=operation)
        results[setting] = _autograd_gradient(loss, base)
    finite = _central_differences(loss, base)
    names = ("angles", "source shift", "center shift")
    for name, many, few, reference in zip(names, results["many"], results["few"], finite):
        assert reference.abs().max() > 1e-6, name
        _assert_relative(many, few, 1e-10, f"{name}: many vs few chunks")
        _assert_relative(many, reference, 1e-6, f"{name}: autograd vs central difference")


def test_constructor_trajectory_leaves_receive_the_override_gradient():
    beam = "fan"
    shape, detector, views, spacing, pitch = _SMALL_SIZES[beam]
    base = _trajectory_parameters(beam, views)
    leaves = tuple(t.detach().clone().requires_grad_() for t in _trajectory(beam, base))
    operator = Projector(leaves, shape, detector, beam=beam, detector_spacing=pitch,
                         voxel_spacing=spacing, backend="torch", precision="float64",
                         **_chunks("many", shape, detector, views))
    data = _random(shape, seed=11)
    weight = _random(operator.projection_shape, seed=12)
    retained = torch.autograd.grad((operator.project(data) * weight).sum(), leaves)
    override = tuple(t.detach().clone().requires_grad_() for t in leaves)
    expected = torch.autograd.grad(
        (operator.project(data, trajectory=override) * weight).sum(), override)
    for actual, reference in zip(retained, expected):
        _assert_relative(actual, reference, 1e-10, "retained vs override trajectory")


@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_trajectory_gradient_saved_state_is_bounded_and_batch_independent(beam, operation):
    results = {setting: _memory(beam, operation, setting, geometry_grad=True)
               for setting in ("few", "many")}
    for setting, (saved, graph, allowance, _) in results.items():
        assert saved <= allowance, (
            f"{setting}: autograd saved {saved} bytes; bound is {allowance} bytes")
        assert graph <= allowance, (
            f"{setting}: graph keeps {graph} bytes alive; bound is {allowance} bytes")
    slack = results["few"][3] + 4096
    assert abs(results["many"][0] - results["few"][0]) <= slack, results
    assert abs(results["many"][1] - results["few"][1]) <= slack, results


def _surface_sampler(u, v, view_indices, radius, shift):
    """Cylindrical offsets with a per-view lateral shift (global view IDs)."""
    view_shift = shift[view_indices].reshape(-1, *([1] * u.dim()))
    local_u = radius * torch.sin(u / radius) + view_shift
    normal = (radius * (torch.cos(u / radius) - 1)).expand_as(local_u)
    local_v = v.expand_as(local_u)
    return torch.stack((local_u, local_v, normal), dim=-1)


def _surface(views, *, radius=8.0, seed=3):
    generator = torch.Generator().manual_seed(seed)
    parameters = (
        torch.tensor(radius, dtype=torch.float64, requires_grad=True),
        (0.3 * (torch.rand(views, generator=generator, dtype=torch.float64) - 0.5))
        .requires_grad_(),
    )
    return ParameterizedSurface(_surface_sampler, parameters=parameters), parameters


def _surface_loss(beam, setting, *, operation="project"):
    shape, detector, views, _, _ = _SMALL_SIZES[beam]
    surface, parameters = _surface(views)
    operator = _operator(beam, setting, sizes=_SMALL_SIZES, detector_surface=surface)
    input_shape = shape if operation == "project" else operator.projection_shape
    output_shape = operator.projection_shape if operation == "project" else shape
    data, weight = _random(input_shape, seed=21), _random(output_shape, seed=22)

    def loss(values=None):
        if values is not None:
            with torch.no_grad():
                for target, value in zip(parameters, values):
                    target.copy_(value)
        return (getattr(operator, operation)(data) * weight).sum()

    return loss, parameters


@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_surface_parameter_gradients_match_finite_differences_and_chunking(beam, operation):
    gradients = {}
    for setting in ("few", "many"):
        loss, parameters = _surface_loss(beam, setting, operation=operation)
        initial = tuple(p.detach().clone() for p in parameters)
        gradients[setting] = torch.autograd.grad(loss(), parameters)
    finite = _central_differences(loss, initial, step=1e-5)
    loss(initial)
    for name, many, few, reference in zip(("radius", "shift"), gradients["many"],
                                          gradients["few"], finite):
        assert reference.abs().max() > 1e-6, name
        _assert_relative(many, few, 1e-10, f"{name}: many vs few chunks")
        _assert_relative(many, reference, 1e-6, f"{name}: autograd vs central difference")


@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_surface_gradient_saved_state_is_bounded_and_batch_independent(beam, operation):
    results = {setting: _memory(beam, operation, setting, surface=True)
               for setting in ("few", "many")}
    for setting, (saved, graph, allowance, _) in results.items():
        assert saved <= allowance, (
            f"{setting}: autograd saved {saved} bytes; bound is {allowance} bytes")
        assert graph <= allowance, (
            f"{setting}: graph keeps {graph} bytes alive; bound is {allowance} bytes")
    slack = results["few"][3] + 4096
    assert abs(results["many"][0] - results["few"][0]) <= slack, results
    assert abs(results["many"][1] - results["few"][1]) <= slack, results


@pytest.mark.parametrize("setting", ["few", "many"])
@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_surface_admission_rejects_nonfinite_parameters_with_gradients(beam, setting):
    shape, _, views, _, _ = _SMALL_SIZES[beam]
    surface, parameters = _surface(views)
    operator = _operator(beam, setting, sizes=_SMALL_SIZES, detector_surface=surface)
    image = _random(shape, seed=31).requires_grad_()
    operator.project(image).sum().backward()
    with torch.no_grad():
        parameters[1][1] = float("nan")
    with pytest.raises(ValueError, match="(?i)finite"):
        operator.project(image)
    with pytest.raises(ValueError, match="(?i)finite"):
        operator.backproject(_random(operator.projection_shape, seed=32).requires_grad_())


def _coincident_sampler(u, v, view_indices, normal_offset):
    local = torch.stack((u, torch.zeros_like(u), torch.zeros_like(u)), dim=-1)
    shift = torch.stack((torch.zeros_like(normal_offset), torch.zeros_like(normal_offset),
                         normal_offset))
    return local + shift


@pytest.mark.parametrize("setting", ["few", "many"])
def test_surface_admission_rejects_pixel_on_source_with_gradients(setting):
    # Source (0, 0); detector centre (2, 0); local normal (u_y, -u_x) = (1, 0).
    # A normal offset of -2 places the single pixel exactly on the source.
    trajectory = (torch.tensor([[0.0, 0.0]], dtype=torch.float64),
                  torch.tensor([[2.0, 0.0]], dtype=torch.float64),
                  torch.tensor([[0.0, 1.0]], dtype=torch.float64))
    offset = torch.tensor(-1.0, dtype=torch.float64, requires_grad=True)
    surface = ParameterizedSurface(_coincident_sampler, parameters=(offset,))
    chunks = {} if setting == "few" else dict(view_chunk_size=1, volume_chunk_shape=(1, 1),
                                               detector_chunk_shape=(1,))
    operator = Projector(trajectory, (2, 2), (1,), beam="fan", detector_surface=surface,
                         backend="torch", precision="float64", **chunks)
    image = torch.ones((2, 2), dtype=torch.float64, requires_grad=True)
    operator.project(image).sum().backward()
    assert offset.grad is not None
    with torch.no_grad():
        offset.fill_(-2.0)
    with pytest.raises(ValueError, match="(?i)coincid|distinct|differ|same|equal|source"):
        operator.project(image)


# ---------------------------------------------------------------------------
# A3: unsupported geometry derivatives raise
# ---------------------------------------------------------------------------

def _geometry_case(beam, operation="project"):
    shape, detector, views, _, _ = _SMALL_SIZES[beam]
    operator = _operator(beam, "many", sizes=_SMALL_SIZES)
    base = _trajectory(beam, _trajectory_parameters(beam, views))
    input_shape = shape if operation == "project" else operator.projection_shape
    output_shape = operator.projection_shape if operation == "project" else shape
    return operator, base, _random(input_shape, seed=41), _random(output_shape, seed=42)


@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_trajectory_second_derivative_raises(beam, operation):
    operator, base, data, weight = _geometry_case(beam, operation)
    source = base[0].clone().requires_grad_()
    trajectory = (source, *base[1:])
    with pytest.raises(DERIVATIVE_ERRORS, match=SECOND_ORDER_MESSAGE):
        loss = (getattr(operator, operation)(data, trajectory=trajectory) * weight).sum()
        first, = torch.autograd.grad(loss, source, create_graph=True)
        torch.autograd.grad(first.square().sum(), source)


@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_mixed_data_geometry_second_derivative_raises_or_is_exact(beam):
    """d/d(source) of the data gradient: an explicit error or the true value."""
    operator, base, image, weight = _geometry_case(beam)
    direction = torch.zeros_like(base[0])
    direction[:, 0] = 1.0

    def data_gradient(source):
        leaf = image.clone().requires_grad_()
        projection = operator.project(leaf, trajectory=(source, *base[1:]))
        return torch.autograd.grad((projection * weight).sum(), leaf, create_graph=True)[0]

    probe = _random(image.shape, seed=43)
    source = base[0].clone().requires_grad_()
    try:
        first = data_gradient(source)
        mixed, = torch.autograd.grad((first * probe).sum(), source, allow_unused=True)
    except DERIVATIVE_ERRORS as error:
        assert re.search(SECOND_ORDER_MESSAGE, str(error)), str(error)
        return
    assert mixed is not None, "mixed derivative silently disconnected from the trajectory"
    step = 1e-6
    with torch.no_grad():
        plus = (operator.backproject(weight, trajectory=(base[0] + step * direction, *base[1:]))
                * probe).sum()
        minus = (operator.backproject(weight, trajectory=(base[0] - step * direction, *base[1:]))
                 * probe).sum()
    finite = (plus - minus) / (2 * step)
    assert finite.abs() > 1e-6
    _assert_relative((mixed * direction).sum(), finite, 1e-6, "mixed derivative")


@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_surface_parameter_second_derivative_raises(beam, operation):
    loss, parameters = _surface_loss(beam, "many", operation=operation)
    radius = parameters[0]
    with pytest.raises(DERIVATIVE_ERRORS, match=SECOND_ORDER_MESSAGE):
        first, = torch.autograd.grad(loss(), radius, create_graph=True)
        torch.autograd.grad(first, radius)


@pytest.mark.parametrize("transform", ["jvp", "jacfwd"])
@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_forward_mode_with_respect_to_trajectory_raises(beam, operation, transform):
    operator, base, data, _ = _geometry_case(beam, operation)

    def function(source):
        return getattr(operator, operation)(data, trajectory=(source, *base[1:]))

    source = base[0].clone()
    with pytest.raises(DERIVATIVE_ERRORS, match=FORWARD_MODE_MESSAGE):
        if transform == "jvp":
            torch.func.jvp(function, (source,), (torch.ones_like(source),))
        else:
            torch.func.jacfwd(function)(source)


# ---------------------------------------------------------------------------
# All differentiable call forms: vmap and differentiable jvp (A1 measurement)
# ---------------------------------------------------------------------------

def _transform_memory(beam, operation, transform, setting):
    shape, _, _, _, _ = _SIZES[beam]
    operator = _operator(beam, setting)
    function = getattr(operator, operation)
    input_shape = shape if operation == "project" else operator.projection_shape
    adjoint = operator.backproject if operation == "project" else operator.project
    if transform == "vmap":
        data = _random((3, *input_shape), seed=51).requires_grad_()
        inputs = (data,)

        def forward():
            return torch.vmap(function)(data)
    else:
        data = _random(input_shape, seed=51).requires_grad_()
        tangent = (_random(input_shape, seed=52) - 0.6).requires_grad_()
        inputs = (data, tangent)

        def forward():
            return torch.func.jvp(function, (data,), (tangent,))

    saved, outputs = _saved_bytes(forward)
    graph = _graph_bytes(forward, inputs)
    outputs = (outputs,) if isinstance(outputs, torch.Tensor) else tuple(outputs)
    # The bounded graph must still give matched-adjoint gradients.
    weights = [_random(value.shape, seed=53 + index) for index, value in enumerate(outputs)]
    loss = sum((value * weight).sum() for value, weight in zip(outputs, weights))
    gradients = torch.autograd.grad(loss, inputs)
    if transform == "vmap":
        _assert_relative(gradients[0], torch.stack([adjoint(w) for w in weights[0]]), 1e-10,
                         "vmap gradient")
    else:
        _assert_relative(gradients[0], adjoint(weights[0]), 1e-10, "jvp primal gradient")
        _assert_relative(gradients[1], adjoint(weights[1]), 1e-10, "jvp tangent gradient")
    io_elements = sum(v.numel() for v in inputs) + sum(v.numel() for v in outputs)
    geometry_bytes = _tensor_bytes(_trajectory(beam, _trajectory_parameters(beam, _SIZES[beam][2])))
    allowance = 4 * io_elements * 8 + 16 * geometry_bytes + 4096
    return saved, graph, allowance, geometry_bytes


@pytest.mark.parametrize("transform", ["vmap", "jvp"])
@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_transform_saved_state_is_bounded_and_batch_independent(beam, operation, transform):
    results = {setting: _transform_memory(beam, operation, transform, setting)
               for setting in ("few", "many")}
    for setting, (saved, graph, allowance, _) in results.items():
        assert saved <= allowance, (
            f"{setting}: autograd saved {saved} bytes; bound is {allowance} bytes")
        assert graph <= allowance, (
            f"{setting}: graph keeps {graph} bytes alive; bound is {allowance} bytes")
    slack = results["few"][3] + 4096
    assert abs(results["many"][0] - results["few"][0]) <= slack, results
    assert abs(results["many"][1] - results["few"][1]) <= slack, results


# ---------------------------------------------------------------------------
# Geometry VJP working memory for a large leading batch
# ---------------------------------------------------------------------------

# docs/EXECUTION.md: automatically sized portable intermediates use an 8 MiB
# ceiling, excluding caller data and returned arrays.
WORK_BYTES = 8 * 1024 * 1024


class _AllocationTimeline(TorchDispatchMode):
    """Record each created storage and when its last observed tensor dies."""

    def __init__(self):
        super().__init__()
        self.events = []
        self._tensors = {}

    def _released(self, pointer):
        self._tensors[pointer] -= 1
        if self._tensors[pointer] == 0:
            del self._tensors[pointer]
            self.events.append(("free", pointer, 0))

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for value in tree_leaves(result):
            if not isinstance(value, torch.Tensor):
                continue
            pointer, size = _storage(value)
            if not pointer:
                continue
            if pointer not in self._tensors:
                self._tensors[pointer] = 0
                self.events.append(("alloc", pointer, size))
            self._tensors[pointer] += 1
            weakref.finalize(value, self._released, pointer)
        return result

    def usage(self, excluded):
        largest = live = peak = 0
        sizes = {}
        for kind, pointer, size in self.events:
            if pointer in excluded:
                continue
            if kind == "alloc":
                sizes[pointer] = size
                largest = max(largest, size)
                live += size
                peak = max(peak, live)
            elif pointer in sizes:
                live -= sizes.pop(pointer)
        return largest, peak


def _large_batch_case(geometry, batch=192):
    beam = "fan"
    shape, detector, views, spacing, pitch = _SIZES[beam]
    parameters = tuple(value.clone().requires_grad_(geometry == "trajectory")
                       for value in _trajectory_parameters(beam, views))
    trajectory = _trajectory(beam, parameters)
    surface, surface_parameters = (_surface(views) if geometry == "surface" else (None, ()))
    operator = Projector(tuple(t.detach() for t in trajectory), shape, detector, beam=beam,
                         detector_spacing=pitch, voxel_spacing=spacing, backend="torch",
                         precision="float64", detector_surface=surface)
    kwargs = {"trajectory": trajectory} if geometry == "trajectory" else {}
    leaves = parameters if geometry == "trajectory" else surface_parameters
    return operator, kwargs, leaves, (*trajectory, *surface_parameters), batch


@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("geometry", ["trajectory", "surface"])
def test_geometry_backward_with_large_batch_stays_within_work_bound(geometry, operation):
    operator, kwargs, leaves, geometry_tensors, batch = _large_batch_case(geometry)
    input_shape = (operator.volume_shape if operation == "project"
                   else operator.projection_shape)
    data = _random((batch, *input_shape), seed=61)
    output = getattr(operator, operation)(data, **kwargs)
    weight = _random(output.shape, seed=62)
    # Caller arrays stay below the ceiling, so any larger allocation is
    # workspace that scales with the batch instead of the automatic bound.
    assert data.numel() * 8 < WORK_BYTES and output.numel() * 8 < WORK_BYTES
    loss = (output * weight).sum()
    timeline = _AllocationTimeline()
    with timeline:
        gradients = torch.autograd.grad(loss, leaves)
    excluded = {value.untyped_storage().data_ptr()
                for value in (data, output, weight, *geometry_tensors, *gradients)}
    largest, peak = timeline.usage(excluded)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    assert largest <= WORK_BYTES, (
        f"geometry backward allocated {largest} bytes at once; ceiling is {WORK_BYTES}")
    assert peak <= 4 * WORK_BYTES, (
        f"geometry backward peak live workspace {peak} bytes; bound is {4 * WORK_BYTES}")


# ---------------------------------------------------------------------------
# Empty leading shapes with trainable geometry
# ---------------------------------------------------------------------------

def _closure_surface(radius):
    def cylinder(u, v):
        return torch.stack((radius * torch.sin(u / radius), v,
                            radius * (torch.cos(u / radius) - 1)), dim=-1)
    return cylinder


@pytest.mark.parametrize("leading", [(0,), (2, 0)], ids=["0", "2x0"])
@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("geometry", ["trajectory", "surface", "closure"])
@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_empty_leading_shape_with_trainable_geometry(beam, geometry, operation, leading):
    shape, detector, views, spacing, pitch = _SMALL_SIZES[beam]
    parameters = tuple(value.clone().requires_grad_(geometry == "trajectory")
                       for value in _trajectory_parameters(beam, views))
    trajectory = _trajectory(beam, parameters)
    surface, leaves = None, parameters
    if geometry == "surface":
        surface, leaves = _surface(views)
    elif geometry == "closure":
        radius = torch.tensor(8.0, dtype=torch.float64, requires_grad=True)
        surface, leaves = _closure_surface(radius), (radius,)
    operator = Projector(tuple(t.detach() for t in trajectory), shape, detector, beam=beam,
                         detector_spacing=pitch, voxel_spacing=spacing, backend="torch",
                         precision="float64", detector_surface=surface)
    kwargs = {"trajectory": trajectory} if geometry == "trajectory" else {}
    input_shape = shape if operation == "project" else operator.projection_shape
    output_shape = operator.projection_shape if operation == "project" else shape
    data = torch.empty((*leading, *input_shape), dtype=torch.float64, requires_grad=True)

    output = getattr(operator, operation)(data, **kwargs)
    assert output.shape == (*leading, *output_shape) and output.numel() == 0
    output.sum().backward()
    assert data.grad is not None and data.grad.shape == data.shape
    for leaf in leaves:
        assert leaf.grad is not None, "trainable geometry received no gradient"
        assert leaf.grad.shape == leaf.shape
        assert torch.count_nonzero(leaf.grad) == 0
