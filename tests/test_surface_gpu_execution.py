"""Native curved surfaces use admitted GPU batches without host pixel replay."""

from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from diffct import Projector
from diffct import operators, surface_gradients
from tests.test_chunked_projector import _memory_inventory
from tests.test_detector_surfaces import (
    _arc_offsets, _case, _close, _data, _finite_difference, _matrix, _numpy, _projector,
)
from tests.test_surface_compute_shards import _surface


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.version.hip is not None,
        reason="native surface execution requires NVIDIA CUDA",
    ),
]


@contextmanager
def _execution_trace():
    """Observe real planners, composition, and nested autograd; forward all calls."""
    trace = SimpleNamespace(plans=[], compositions=[], replay_calls=0,
                            nested_grad_calls=0, geometry_vjp_calls=0)
    planning = Projector._execution_plan
    composing = Projector._surface_geometry_into
    replay = surface_gradients._sample_geometry
    geometry_vjp = operators._geometry_vjp
    autograd_grad = torch.autograd.grad
    grad_depth = 0

    def plan(owner, *args, **kwargs):
        result = planning(owner, *args, **kwargs)
        trace.plans.append(result)
        return result

    def compose(owner, geometry, spec, views, pixels, centre, targets, pipe, groups, offsets=None):
        trace.compositions.append((tuple(targets[1].shape), targets[1].device))
        return composing(owner, geometry, spec, views, pixels, centre, targets, pipe, groups, offsets)

    def sample(*args, **kwargs):
        trace.replay_calls += 1
        return replay(*args, **kwargs)

    def vjp(*args, **kwargs):
        trace.geometry_vjp_calls += 1
        return geometry_vjp(*args, **kwargs)

    def grad(*args, **kwargs):
        nonlocal grad_depth
        if grad_depth:
            trace.nested_grad_calls += 1
        grad_depth += 1
        try:
            return autograd_grad(*args, **kwargs)
        finally:
            grad_depth -= 1

    with ExitStack() as patches:
        patches.enter_context(mock.patch.object(Projector, "_execution_plan", plan))
        patches.enter_context(mock.patch.object(Projector, "_surface_geometry_into", compose))
        patches.enter_context(mock.patch.object(surface_gradients, "_sample_geometry", sample))
        patches.enter_context(mock.patch.object(operators, "_geometry_vjp", vjp))
        patches.enter_context(mock.patch.object(torch.autograd, "grad", grad))
        yield trace


def _native_operator(case, kind, radius, **kwargs):
    return _projector(
        case, _surface(kind, radius), backend="numba", precision="fast",
        distributed=False, devices=[torch.cuda.current_device()], **kwargs,
    )


@pytest.mark.parametrize("beam", ("fan", "cone"))
@pytest.mark.parametrize("operation", ("project", "backproject"))
def test_parameterized_cuda_surface_uses_full_execution_when_legacy_workload_fits(beam, operation):
    case = _case(beam)
    radius = torch.tensor(3.7, dtype=torch.float64)
    image, rays = _data(case)
    source = image if operation == "project" else rays
    matrix = _matrix(case, _arc_offsets(case, 3.7, .037))
    linear = matrix if operation == "project" else matrix.T
    observations = []
    for kind in ("legacy", "parameterized"):
        operator = _native_operator(case, kind, radius)
        with _execution_trace() as trace:
            result = getattr(operator, operation)(source)
        _close(result, (linear @ _numpy(source).ravel()).reshape(result.shape))
        assert result.device == source.device and result.dtype == torch.float32
        observations.append(trace.plans)
    # The same real device budget already admits the legacy representation.
    # A pure explicit sampler with these identical pixels must be eligible too.
    assert observations[0] and all(plan is None for plan in observations[0]), (
        "the small legacy control did not fit the real device budget", observations[0],
    )
    assert observations[1] and all(plan is None for plan in observations[1]), (
        "parameterized surface forced streaming despite an admitted full workload", observations[1],
    )


def _gradient_reference(case, image, rays):
    image_np, rays_np = _numpy(image).ravel(), _numpy(rays).ravel()
    offsets = _arc_offsets(case, 3.7, .037)
    matrix = _matrix(case, offsets)
    radius_gradient = _finite_difference(
        lambda radius: float(rays_np @ _matrix(case, _arc_offsets(case, radius, .037)) @ image_np),
        3.7,
    )
    geometry = tuple(_numpy(value) for value in case.trajectory)
    generator = np.random.default_rng(922)
    frames = []
    for index, value in enumerate(geometry):
        direction = generator.normal(size=value.shape)
        direction /= np.linalg.norm(direction)

        def objective(changed):
            values = list(geometry)
            values[index] = changed
            return float(rays_np @ _matrix(case, offsets, values) @ image_np)

        frames.append((direction, _finite_difference(objective, value, direction)))
    return matrix, radius_gradient, frames


@pytest.mark.parametrize("beam", ("fan", "cone"))
@pytest.mark.parametrize("kind", ("legacy", "parameterized"))
def test_surface_geometry_gradients_replay_one_admitted_batch_not_detector_rows(beam, kind):
    case = _case(beam, learnable=True)
    radius = torch.tensor(3.7, dtype=torch.float64, requires_grad=True)
    operator = _native_operator(case, kind, radius)
    image, rays = _data(case)
    matrix, expected_radius, frame_checks = _gradient_reference(case, image, rays)
    traces, outputs = [], []
    for operation, source, weight, linear in (
        ("project", image, rays, matrix), ("backproject", rays, image, matrix.T),
    ):
        data = source.clone().requires_grad_()
        with _execution_trace() as trace:
            result = getattr(operator, operation)(data)
            gradients = torch.autograd.grad(
                (result.double() * weight.double()).sum(), (data, radius, *case.trajectory),
            )
        _close(result, (linear @ _numpy(source).ravel()).reshape(result.shape))
        _close(gradients[0], (linear.T @ _numpy(weight).ravel()).reshape(data.shape))
        assert abs(expected_radius) > 1e-3
        assert gradients[1].item() == pytest.approx(expected_radius, rel=5e-3, abs=3e-4)
        for gradient, original, (direction, expected) in zip(
            gradients[2:], case.trajectory, frame_checks,
        ):
            assert gradient.device == original.device and gradient.dtype == original.dtype
            assert torch.isfinite(gradient).all() and abs(expected) > 1e-3
            actual = float((_numpy(gradient) * direction).sum())
            assert actual == pytest.approx(expected, rel=5e-3, abs=3e-4)
        traces.append(trace)
        outputs.append(result.detach())
    torch.testing.assert_close((outputs[0].double() * rays.double()).sum(),
                               (image.double() * outputs[1].double()).sum(),
                               rtol=4e-4, atol=6e-5)
    for trace in traces:
        # One full batch can compose a single replay graph and pull back once
        # through geometry plus once through the user sampler. An analytic GPU
        # pullback may use fewer calls. Counts must not grow with rows or views.
        assert trace.replay_calls <= 1, f"full-batch replay fragmented into {trace.replay_calls} calls"
        assert trace.nested_grad_calls <= 2, (
            f"full-batch backward built {trace.nested_grad_calls} nested host autograd graphs"
        )
        assert trace.geometry_vjp_calls <= 1, (
            f"full-batch geometry VJP split into {trace.geometry_vjp_calls} kernel jobs"
        )


@pytest.mark.parametrize("constraint", ("explicit-chunks", "payload-budget"))
def test_parameterized_cuda_surface_keeps_bounded_fallback_and_correct_adjoint(constraint):
    case = _case("cone")
    radius = torch.tensor(3.7, dtype=torch.float64)
    hints = (dict(volume_chunk_shape=(3, 4, 5), view_chunk_size=1,
                  detector_chunk_shape=(3, 2)) if constraint == "explicit-chunks" else {})
    operator = _native_operator(case, "parameterized", radius, **hints)
    image, rays = _data(case)
    matrix = _matrix(case, _arc_offsets(case, 3.7, .037))
    outputs = []
    # Existing inventory helper injects only logical payload pressure. Kernels
    # remain real; physical allocator reservation tests live in their own suite.
    free = 8192
    for operation, source, linear in (
        ("project", image, matrix), ("backproject", rays, matrix.T),
    ):
        memory = (_memory_inventory({torch.cuda.current_device(): free})
                  if constraint == "payload-budget" else nullcontext())
        with memory, _execution_trace() as trace:
            result = getattr(operator, operation)(source)
        assert trace.plans and all(plan is not None for plan in trace.plans)
        assert trace.compositions, "streamed fallback did not expose a real geometry batch"
        for plan in trace.plans:
            if constraint == "explicit-chunks":
                assert all(a <= b for a, b in zip(plan.chunk_shape, hints["volume_chunk_shape"]))
                assert plan.view_chunk_size <= hints["view_chunk_size"]
                assert all(a <= b for a, b in zip(plan.detector_chunk_shape, hints["detector_chunk_shape"]))
            else:
                assert 0 < plan.estimated_gpu_bytes <= plan.gpu_budget_bytes <= .75 * free
        maximum_views = max(plan.view_chunk_size for plan in trace.plans)
        maximum_pixels = tuple(max(plan.detector_chunk_shape[axis] for plan in trace.plans)
                               for axis in range(len(case.detector)))
        for shape, device in trace.compositions:
            assert device == source.device and shape[0] <= maximum_views
            assert all(size <= limit for size, limit in zip(shape[1:-1], maximum_pixels))
        _close(result, (linear @ _numpy(source).ravel()).reshape(result.shape))
        outputs.append(result.detach())
    torch.testing.assert_close((outputs[0].double() * rays.double()).sum(),
                               (image.double() * outputs[1].double()).sum(),
                               rtol=4e-4, atol=6e-5)
