"""Bound host geometry work for a shared legacy CUDA detector surface.

Count real calls, including call-time admission and schedule selection. Only
observe execution; constructors run outside the counters and kernels still run.
"""

from contextlib import ExitStack, contextmanager
import math
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from diffct import Projector, operators, surface_gradients, surfaces


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.version.hip is not None,
        reason="native surface host budgets require NVIDIA CUDA",
    ),
]

DETECTOR = (888, 64)
VOLUME = (3, 4, 5)
PITCH = (.012, .07)
VIEWS = (64, 256)
STREAM_HINTS = dict(view_chunk_size=16, detector_chunk_shape=(444, 32))
STREAM_DETECTOR_TILES = math.prod(
    math.ceil(size / chunk)
    for size, chunk in zip(DETECTOR, STREAM_HINTS["detector_chunk_shape"])
)
# Allow four additional samples per operation for admission checks or bounded
# pilot/validation setup. This slack must not grow with views, batches, or tiles.
SAMPLE_SETUP_SLACK = 4
# Match the established float32 native surface acceptance tolerance.
RTOL, ATOL = 4e-4, 6e-5


def _trajectory(views):
    # The 64-view trajectory is a prefix of the 256-view trajectory.
    angle = .217 + torch.arange(views, dtype=torch.float32) * (2 * math.pi / 256)
    zero = torch.zeros_like(angle)
    radial = torch.stack((angle.cos(), angle.sin(), zero), dim=-1)
    tangent = torch.stack((-angle.sin(), angle.cos(), zero), dim=-1)
    vertical = torch.stack((zero, zero, torch.ones_like(angle)), dim=-1)
    return -9 * radial, 6 * radial, tangent, vertical


def _shared_surface(detector, device, pitch=PITCH):
    axes = [(torch.arange(size, device=device, dtype=torch.float32) + .5 - size / 2) * spacing
            for size, spacing in zip(detector, pitch)]
    u, v = torch.meshgrid(*axes, indexing="ij")
    radius = 8.
    offsets = torch.stack((radius * torch.sin(u / radius), v,
                           radius * (1 - torch.cos(u / radius))), dim=-1)
    assert offsets.shape == (*detector, 3)
    assert offsets.is_cuda and offsets.dtype == torch.float32
    assert not offsets.requires_grad

    def surface(u, v):
        # Return the same complete shared CUDA tensor for every invocation.
        return offsets

    return surface


class _LaunchCounter:
    """Forward dispatcher configuration and count actual launches."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.calls = 0

    def __getitem__(self, configuration):
        launch = self.kernel[configuration]

        def run(*args, **kwargs):
            self.calls += 1
            return launch(*args, **kwargs)

        return run


@contextmanager
def _host_work():
    # Import only after the hardware skip: CUDASIM does not supply torch CUDA.
    from diffct.kernels import surface_geometry

    launches = _LaunchCounter(surface_geometry.compose_surface)
    trace = SimpleNamespace(launches=launches, samples=0, plans=[])
    sample_geometry = surfaces._sample_geometry
    execution_plan = Projector._execution_plan

    def sample(*args, **kwargs):
        trace.samples += 1
        return sample_geometry(*args, **kwargs)

    def plan(owner, *args, **kwargs):
        result = execution_plan(owner, *args, **kwargs)
        trace.plans.append(result)
        return result

    with ExitStack() as patches:
        patches.enter_context(mock.patch.object(surface_geometry, "compose_surface", launches))
        # Also cover a module-level import if composition moves out of the method.
        if "compose_surface" in vars(operators):
            patches.enter_context(mock.patch.object(operators, "compose_surface", launches))
        # Patch both the definition and its existing imported aliases. Each
        # wrapper calls the saved original, so one sample counts only once.
        for module in (surfaces, operators, surface_gradients):
            patches.enter_context(mock.patch.object(module, "_sample_geometry", sample))
        patches.enter_context(mock.patch.object(Projector, "_execution_plan", plan))
        yield trace


@pytest.fixture(scope="module")
def large_runs():
    device = torch.device("cuda", torch.cuda.current_device())
    surface = _shared_surface(DETECTOR, device)
    image = torch.linspace(.1, 1.9, math.prod(VOLUME), device=device,
                           dtype=torch.float32).reshape(VOLUME)
    runs = {}
    for views in VIEWS:
        frames = _trajectory(views)
        assert all(not frame.requires_grad for frame in frames)
        rays = torch.ones((views, *DETECTOR), dtype=torch.float32, device=device)
        for name, hints in (("default", {}), ("streamed", STREAM_HINTS)):
            operator = Projector(
                frames, VOLUME, DETECTOR, beam="cone", detector_spacing=PITCH,
                voxel_spacing=.6, detector_surface=surface, backend="numba",
                precision="fast", devices=[device], distributed=False, **hints,
            )
            assert operator.backend == "numba" and not operator._geometry_trainable
            for operation, source in (("project", image), ("backproject", rays)):
                torch.cuda.synchronize(device)
                with torch.no_grad(), _host_work() as trace:
                    output = getattr(operator, operation)(source)
                    torch.cuda.synchronize(device)
                # Keep results on CPU so earlier calls do not consume the next
                # call's device budget. Copying is outside the measured region.
                trace.output = output.detach().cpu()
                trace.execution_stats = operator.last_execution_stats
                runs[name, operation, views] = trace
                del output
    return runs


@pytest.mark.parametrize("plan", ("default", "streamed"))
@pytest.mark.parametrize("operation", ("project", "backproject"))
def test_compose_surface_launches_stay_within_plan_budget(large_runs, plan, operation):
    for views in VIEWS:
        trace = large_runs[plan, operation, views]
        assert trace.plans, "native execution did not reach its planner"
        if plan == "streamed":
            for actual in trace.plans:
                assert actual is not None, "explicit hints did not force streaming"
                assert actual.view_chunk_size <= STREAM_HINTS["view_chunk_size"]
                assert all(size <= limit for size, limit in zip(
                    actual.detector_chunk_shape, STREAM_HINTS["detector_chunk_shape"],
                ))
        # Geometry uses each volume tile's centre, so every planned ray job may
        # compose once: view batches x detector tiles x volume tiles.
        stats = trace.execution_stats
        if stats is None:
            assert all(actual is None for actual in trace.plans)
            maximum = 0  # Full batched tensor geometry needs no compose launch.
        else:
            batches = math.ceil(views / stats["view_chunk_size"])
            detector_tiles = math.prod(math.ceil(size / chunk) for size, chunk in zip(
                DETECTOR, stats["detector_chunk_shape"],
            ))
            volume_tiles = math.prod(math.ceil(size / chunk) for size, chunk in zip(
                VOLUME, stats["chunk_shape"],
            ))
            maximum = batches * detector_tiles * volume_tiles
            # Schedule selection runs one warmup job, then a two-job prefix
            # for spatial/views or a three-job prefix for window. Each job
            # gets one launch, independent of its views, rows, or segments.
            maximum += sum(1 + {"spatial": 2, "views": 2, "window": 3}[pilot["schedule"]]
                           for pilot in stats["pilot"])
        assert trace.launches.calls <= maximum, (
            f"{plan} {operation}: {trace.launches.calls} compose_surface launches "
            f"for {views} views and detector {DETECTOR}; maximum is {maximum}"
        )
        assert torch.isfinite(trace.output).all()
        assert torch.count_nonzero(trace.output).item() > 0


@pytest.mark.parametrize("operation", ("project", "backproject"))
def test_default_geometry_sampling_does_not_scale_with_views(large_runs, operation):
    small = large_runs["default", operation, 64].samples
    large = large_runs["default", operation, 256].samples
    # Four times as many views may use at most twice as many host samples.
    # A path that needs no sample at 64 views still gets a two-call allowance.
    assert large <= 2 * max(1, small), (
        f"default {operation}: _sample_geometry calls grew from {small} at 64 views "
        f"to {large} at 256 views; maximum is {2 * max(1, small)}"
    )
    assert large <= 8, (
        f"default {operation}: {large} _sample_geometry calls for 256 views; maximum is 8"
    )


@pytest.mark.parametrize("views", VIEWS)
@pytest.mark.parametrize("operation", ("project", "backproject"))
def test_streamed_geometry_sampling_is_bounded_by_explicit_chunks(large_runs, views, operation):
    # Validation may sample once per explicit view batch and detector tile.
    # Derive the budget from the requested chunks, not observed sample counts.
    batches = math.ceil(views / STREAM_HINTS["view_chunk_size"])
    maximum = batches * STREAM_DETECTOR_TILES + SAMPLE_SETUP_SLACK
    samples = large_runs["streamed", operation, views].samples
    assert samples <= maximum, (
        f"streamed {operation}: {samples} _sample_geometry calls for {views} views; "
        f"maximum is {maximum} ({batches} view batches x {STREAM_DETECTOR_TILES} "
        f"detector tiles + {SAMPLE_SETUP_SLACK} setup calls)"
    )


@pytest.mark.parametrize("views", VIEWS)
def test_streamed_projection_matches_default(large_runs, views):
    default = large_runs["default", "project", views].output
    streamed = large_runs["streamed", "project", views].output
    assert default.shape == streamed.shape == (views, *DETECTOR)
    assert default.dtype == streamed.dtype == torch.float32
    torch.testing.assert_close(streamed, default, rtol=RTOL, atol=ATOL)


def test_small_native_projection_matches_torch():
    device = torch.device("cuda", torch.cuda.current_device())
    detector = (17, 9)
    pitch = (.4, .3)
    frames = tuple(frame[::51][:5].contiguous() for frame in _trajectory(256))
    surface = _shared_surface(detector, device, pitch)
    image = torch.linspace(.1, 1.9, math.prod(VOLUME), device=device,
                           dtype=torch.float32).reshape(VOLUME)
    results = {}
    for backend, hints in (
        ("torch", {}),
        ("default", {}),
        ("streamed", dict(view_chunk_size=2, detector_chunk_shape=(8, 4))),
    ):
        operator = Projector(
            frames, VOLUME, detector, beam="cone", detector_spacing=pitch,
            voxel_spacing=.6, detector_surface=surface,
            backend="torch" if backend == "torch" else "numba", precision="fast",
            devices=[device], distributed=False, **hints,
        )
        with torch.no_grad():
            results[backend] = operator.project(image)
    reference = results["torch"]
    assert reference.shape == (5, *detector) and reference.dtype == torch.float32
    assert torch.isfinite(reference).all() and torch.count_nonzero(reference).item() > 0
    for name in ("default", "streamed"):
        torch.testing.assert_close(results[name], reference, rtol=RTOL, atol=ATOL)
