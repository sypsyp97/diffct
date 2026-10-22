"""Count real native admission, surface launches, and device transfers."""

import math
from contextlib import nullcontext
from unittest import mock

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from diffct import ParameterizedSurface, Projector
from tests.test_chunked_projector import _memory_inventory
from tests.test_detector_surfaces import (
    _arc_offsets, _case, _close, _data, _matrix, _numpy, _projector,
)
from tests.test_surface_host_sampling_budget import _host_work


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.version.hip is not None,
        reason="native performance contracts require NVIDIA CUDA",
    ),
]


class _HostObservations(TorchDispatchMode):
    """Observe scalar reads and copies without replacing tensor operations."""

    def __init__(self):
        super().__init__()
        self.admitting = True
        self.scalars = []
        self.bulk_d2h = []
        self.admission_d2h = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        name = str(function)
        if self.admitting and name in ("aten._local_scalar_dense.default", "aten.allclose.default"):
            if any(isinstance(value, torch.Tensor) and value.is_cuda for value in args):
                self.scalars.append(name)
        source = None
        if name == "aten._to_copy.default" and result.device.type == "cpu":
            source = args[0]
        elif name == "aten.copy_.default" and args[0].device.type == "cpu":
            source = args[1]
        if isinstance(source, torch.Tensor) and source.is_cuda and source.numel() > 1:
            record = (name, tuple(source.shape), source.numel() * source.element_size())
            self.bulk_d2h.append(record)
            if self.admitting:
                self.admission_d2h.append(record)
        return result


@pytest.mark.parametrize("operation", ("project", "backproject"))
@pytest.mark.parametrize("override", (False, True), ids=("retained", "explicit"))
def test_live_native_cone_admission_aggregates_host_observations(operation, override):
    case = _case("cone", views=3)
    case.trajectory = tuple(value.cuda().requires_grad_() for value in case.trajectory)
    operator = _projector(case, backend="numba", distributed=False)
    image, rays = _data(case)
    source = image if operation == "project" else rays
    reference = _projector(case, backend="torch", precision="float64", distributed=False)
    expected = getattr(reference, operation)(source.double())
    execute = getattr(Projector, "_" + operation)
    reached = []

    def numerical(owner, *args):
        reached.append(True)
        trace.admitting = False
        return execute(owner, *args)

    with _HostObservations() as trace, mock.patch.object(Projector, "_" + operation, numerical):
        actual = getattr(operator, operation)(source, **({"trajectory": case.trajectory} if override else {}))
    torch.testing.assert_close(actual.double(), expected, rtol=4e-4, atol=6e-5)
    assert reached == [True], "measurement did not reach native numerical execution"
    assert not trace.admission_d2h, (
        f"native CUDA admission staged geometry on CPU: {trace.admission_d2h}"
    )
    assert len(trace.scalars) <= 6, (
        f"ordinary three-view cone admission made {len(trace.scalars)} CUDA host observations; "
        f"maximum is 6: {trace.scalars}"
    )


@pytest.mark.parametrize("operation", ("project_into", "backproject_into"))
@pytest.mark.parametrize("invalid", ("nan-axis", "edge-on"))
def test_live_native_invalid_axes_reject_before_into_writes(operation, invalid):
    case = _case("cone", views=3)
    case.trajectory = tuple(value.cuda().requires_grad_() for value in case.trajectory)
    operator = _projector(case, backend="numba", distributed=False)
    image, rays = _data(case)
    source, shape = ((image, case.sino_shape) if operation == "project_into" else (rays, case.shape))
    output = torch.full(shape, 73.0, device=source.device, dtype=source.dtype)
    with torch.no_grad():
        if invalid == "nan-axis":
            case.trajectory[2][1, 0] = float("nan")
        else:
            direction = case.trajectory[1][1] - case.trajectory[0][1]
            case.trajectory[2][1].copy_(direction / torch.linalg.vector_norm(direction))
    with pytest.raises(ValueError):
        getattr(operator, operation)(source, output)
    torch.testing.assert_close(output, torch.full_like(output, 73.0), rtol=0, atol=0)


@pytest.mark.parametrize("operation", ("project", "backproject"))
@pytest.mark.parametrize("kind", ("legacy-per-view", "sampled-shared", "sampled-per-view"))
@pytest.mark.parametrize("memory", ("ample", "tiny-budget"))
def test_streamed_surface_composes_each_admitted_batch_once(operation, kind, memory):
    case = _case("cone", views=3)
    case.shape, case.detector, case.pitch = (3, 4, 5), (4, 47), (.52, .08)
    radii = (torch.tensor(3.7, dtype=torch.float64) if kind == "sampled-shared"
             else torch.tensor([3.3, 3.7, 4.1], dtype=torch.float64))
    samples = []

    def arc(u, v, radius):
        if radius.ndim:
            radius = radius[:, None, None]
        tangent = radius * torch.sin(u / radius) + .037 * u.square()
        return torch.stack((tangent, v.expand_as(tangent), radius * (1 - torch.cos(u / radius))), -1)

    def sampler(u, v, ids, radius):
        assert u.device.type == v.device.type == ids.device.type == radius.device.type == "cpu"
        samples.append(tuple(ids.tolist()))
        return arc(u, v, radius[ids] if radius.ndim else radius)

    surface = ((lambda u, v: arc(u, v, radii)) if kind == "legacy-per-view"
               else ParameterizedSurface(sampler, (radii,)))
    operator = _projector(
        case, surface, backend="numba", distributed=False, schedule="spatial",
        view_chunk_size=2, detector_chunk_shape=(3, 31),
    )
    image, rays = _data(case, device="cpu" if memory == "tiny-budget" else "cuda")
    source = image if operation == "project" else rays
    matrix = _matrix(case, _arc_offsets(case, _numpy(radii), .037))
    linear = matrix if operation == "project" else matrix.T
    compose = Projector._surface_geometry_into
    jobs = []

    def observed(owner, geometry, spec, views, pixels, centre, targets, pipe, groups, offsets=None):
        ready = compose(owner, geometry, spec, views, pixels, centre, targets, pipe, groups, offsets)
        jobs.append((views.stop - views.start, tuple(part.stop - part.start for part in pixels)))
        # Compare the allocated reusable workspace with the planner's maximum
        # job inventory; edge batches may reuse the larger preceding buffers.
        actual_bytes = sum(math.ceil(value.nbytes / 512) * 512
                           for value in groups["_surface_workspace"][:2])
        plan = trace.plans[-1]
        batching = ({"batched": plan.surface_batch} if hasattr(plan, "surface_batch") else {})
        budget = sum(owner._surface_work_buffers(2, (3, 31), spec=spec, streamed=True, **batching))
        assert actual_bytes <= budget, ("surface workspace exceeded its admitted bytes", actual_bytes, budget)
        return ready

    samples.clear()
    # Both explicit chunks fit the old bounded workspace in this 6144-byte
    # payload budget. A larger composition workspace must retain that fallback.
    pressure = (_memory_inventory({torch.cuda.current_device(): 8192})
                if memory == "tiny-budget" else nullcontext())
    with pressure, torch.no_grad(), _host_work() as trace, \
            mock.patch.object(Projector, "_surface_geometry_into", observed):
        result = getattr(operator, operation)(source)
    _close(result, (linear @ _numpy(source).ravel()).reshape(result.shape))
    assert torch.count_nonzero(result).item() > 0
    assert trace.plans and all(plan is not None for plan in trace.plans)
    assert jobs and any(views == 2 and math.prod(pixels) > 21 for views, pixels in jobs)
    assert all(views <= 2 and all(size <= limit for size, limit in zip(pixels, (3, 31)))
               for views, pixels in jobs)
    for plan in trace.plans:
        assert 0 < plan.estimated_gpu_bytes <= plan.gpu_budget_bytes
        if memory == "tiny-budget":
            assert plan.gpu_budget_bytes <= 6144
            assert plan.view_chunk_size == 2 and plan.detector_chunk_shape == (3, 31)
    if kind.startswith("sampled"):
        assert samples and all(0 < len(ids) <= 2 for ids in samples)
        assert set().union(*map(set, samples)) == set(range(case.views))
    if memory == "ample":
        assert trace.launches.calls <= len(jobs), (
            f"{kind} {operation}: {trace.launches.calls} compose launches for {len(jobs)} admitted jobs; "
            "launches must not scale with views, detector rows, or 21-pixel segments"
        )


@pytest.mark.parametrize("operation", ("project", "backproject"))
def test_full_legacy_cuda_snapshots_and_outstanding_gradients_stay_device_local(operation):
    case = _case("cone", views=3)
    initial_frames = case.trajectory
    initial_offsets = torch.from_numpy(_arc_offsets(case, 3.7, .037))
    image, rays = _data(case)
    source, weight = (image, rays) if operation == "project" else (rays, image)

    def setup(device, backend):
        case.trajectory = tuple(value.detach().to(device).clone().requires_grad_() for value in initial_frames)
        offsets = initial_offsets.to(device).clone().requires_grad_()
        gain = torch.tensor(1.0, dtype=torch.float64, device=device, requires_grad=True)

        def callback(u, v):
            # Both multiplicands are saved for backward and must retain their
            # forward values if the caller updates captured tensors in place.
            return offsets * gain.square()

        owner = _projector(case, callback, backend=backend,
                           precision="float64" if backend == "torch" else "fast", distributed=False)
        data = source.detach().to(device=device, dtype=owner.dtype).clone().requires_grad_()
        return owner, (data, offsets, gain, *case.trajectory)

    reference, controls = setup("cpu", "torch")
    expected = getattr(reference, operation)(controls[0])
    expected_gradients = torch.autograd.grad((expected * weight.cpu().double()).sum(), controls)
    operator, originals = setup("cuda", "numba")
    with _HostObservations() as trace, _host_work() as execution:
        result = getattr(operator, operation)(originals[0])
        with torch.no_grad():
            originals[1].add_(.19)
            originals[2].fill_(1.3)
            originals[4].add_(.07)
        operator.detector_surface = lambda u, v: (_ for _ in ()).throw(
            AssertionError("outstanding backward resampled a replaced callback")
        )
        gradients = torch.autograd.grad((result.double() * weight.double()).sum(), originals)
    assert execution.plans and all(plan is None for plan in execution.plans), "fixture did not admit full execution"
    torch.testing.assert_close(result.cpu().double(), expected, rtol=4e-4, atol=6e-5)
    for actual, wanted, original in zip(gradients, expected_gradients, originals):
        assert actual.device == original.device and actual.dtype == original.dtype
        torch.testing.assert_close(actual.cpu().double(), wanted.double(), rtol=5e-3, atol=4e-4)
    assert not trace.bulk_d2h, (
        f"full CUDA surface forward/backward copied {sum(size for _, _, size in trace.bulk_d2h)} bytes "
        f"to CPU: {trace.bulk_d2h}"
    )
