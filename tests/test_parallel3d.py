"""Infinite 3D parallel rays, checked by independent scalar cell intersections."""

from dataclasses import replace
from datetime import timedelta
import itertools
import math
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import diffct
from diffct import ParameterizedSurface, Projector, TensorStore
from diffct.utils import TorchCUDABridge
from tests.test_reference_backend import _Case, _data, _slab_length


def _case():
    angles = torch.tensor([.231, .811, 1.431], dtype=torch.float64)
    tilts = torch.tensor([.197, -.137, .283], dtype=torch.float64)
    normal = torch.stack((angles.cos() * tilts.cos(), angles.sin() * tilts.cos(), tilts.sin()), dim=1)
    u = torch.stack((-angles.sin(), angles.cos(), torch.zeros_like(angles)), dim=1)
    v = torch.linalg.cross(normal, u, dim=1)
    # Unit ray directions deliberately oblique to both detector axes.
    direction = normal + .23 * u - .17 * v
    direction = direction / direction.norm(dim=1, keepdim=True)
    offsets = torch.tensor([[.173, -.239, -.317], [-.119, .281, .113], [.317, .113, .413]], dtype=torch.float64)
    center = offsets + torch.tensor([-5., 0., 5.], dtype=torch.float64)[:, None] * direction
    return _Case("parallel3d", (direction, center, u, v), (3, 4, 5), (4, 3), (.71, .63), (1.3, .9, .6))


def _operator(case, **options):
    defaults = dict(beam="parallel3d", detector_spacing=case.pitch, voxel_spacing=case.spacing,
                    backend="torch", precision="float64")
    defaults.update(options)
    return Projector(case.trajectory, case.shape, case.detector, **defaults)


def _matrix(case, *, curved=False, coefficient=.12):
    """One independent scalar slab intersection for every detector ray/cell pair."""
    frames = [value.detach().cpu().double().tolist() for value in case.trajectory]
    spacing = (case.spacing,) * 3 if isinstance(case.spacing, (float, int)) else case.spacing
    rows = []
    for view, iu, iv in itertools.product(range(len(frames[0])), range(case.detector[0]), range(case.detector[1])):
        a = (iu - (case.detector[0] - 1) / 2) * case.pitch[0]
        b = (iv - (case.detector[1] - 1) / 2) * case.pitch[1]
        sag = coefficient * (a * a + .31 * a * b + .19 * b * b) if curved else 0.
        u, v = frames[2][view], frames[3][view]
        n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        origin = [c + a * du + b * dv + sag * dn for c, du, dv, dn in zip(frames[1][view], u, v, n)]
        row = []
        for cell in itertools.product(*(range(size) for size in case.shape)):
            lower = [(index - size / 2) * pitch for index, size, pitch in zip(cell, case.shape, spacing)][::-1]
            upper = [lo + pitch for lo, pitch in zip(lower, spacing[::-1])]
            row.append(_slab_length(origin, frames[0][view], lower, upper, segment=False))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float64)


def _no_native(*args, **kwargs):
    raise AssertionError("3D parallel entered a native CUDA bridge")


def test_oblique_anisotropic_noncubic_forward_and_adjoint_match_cell_matrix(monkeypatch):
    monkeypatch.setattr(TorchCUDABridge, "tensor_to_cuda_array", _no_native)
    case = _case()
    matrix, operator = _matrix(case), _operator(case, backend="auto")
    assert operator.backend == "torch" and operator.dtype == torch.float64
    assert operator.volume_shape == case.shape and operator.projection_shape == case.projection_shape
    assert all((case.trajectory[0] * axis).sum(dim=1).abs().min() > .1 for axis in case.trajectory[2:])
    assert matrix.abs().sum() > 0, "fixture needs nonzero intersections"
    image, rays = _data(case.shape), _data(case.projection_shape, phase=.31)
    projected, adjoint = operator(image), operator.backproject(rays)
    torch.testing.assert_close(projected.flatten(), matrix @ image.flatten(), rtol=2e-11, atol=2e-12)
    torch.testing.assert_close(adjoint.flatten(), matrix.T @ rays.flatten(), rtol=2e-11, atol=2e-12)
    torch.testing.assert_close((projected * rays).sum(), (image * adjoint).sum(), rtol=2e-11, atol=2e-12)


def test_infinite_lines_ignore_origin_position_and_direction_sign_and_report_misses():
    direction = torch.tensor([[1., 0., 0.]] * 5, dtype=torch.float64)
    center = torch.tensor([[-100., .13, .17], [0., .13, .17], [100., .13, .17], [100., .13, .17], [0., 20., .17]], dtype=torch.float64)
    direction[3] *= -1
    u = torch.tensor([[0., 1., 0.]] * 5, dtype=torch.float64)
    v = torch.tensor([[0., 0., 1.]] * 5, dtype=torch.float64)
    case = _Case("parallel3d", (direction, center, u, v), (3, 4, 7), (1, 1), (1., 1.), (1.3, .9, .6))
    expected = torch.tensor([4.2, 4.2, 4.2, 4.2, 0.], dtype=torch.float64)
    torch.testing.assert_close(_operator(case).project(torch.ones(case.shape, dtype=torch.float64)).flatten(), expected, rtol=1e-12, atol=1e-12)


def test_near_tangent_nonzero_direction_is_valid_and_has_analytic_box_chord():
    direction = torch.tensor([[1., 0., 1e-5]], dtype=torch.float64)
    direction = direction / direction.norm(dim=1, keepdim=True)
    center = torch.zeros(1, 3, dtype=torch.float64)
    u = torch.tensor([[1., 0., 0.]], dtype=torch.float64)
    v = torch.tensor([[0., 1., 0.]], dtype=torch.float64)
    case = _Case("parallel3d", (direction, center, u, v), (1, 1, 1), (1, 1), (1., 1.), 1.)
    # The centered infinite line exits the cube through x=+-1/2.
    expected = (1. / direction[0, 0]).reshape(1, 1, 1)
    actual = _operator(case).project(torch.ones(case.shape, dtype=torch.float64))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_batched_noncontiguous_chunks_and_stores_match_cell_matrix(operation):
    case, leading = _case(), (3, 2)
    matrix = _matrix(case)
    operator = _operator(case, view_chunk_size=2, detector_chunk_shape=(2, 2), volume_chunk_shape=(2, 2, 3))
    shape = case.shape if operation == "project" else case.projection_shape
    output_shape = case.projection_shape if operation == "project" else case.shape
    value = _data((2, 3, *shape), phase=.31).transpose(0, 1)
    assert not value.is_contiguous()
    expected = (value.reshape(6, -1) @ (matrix.T if operation == "project" else matrix)).reshape(*leading, *output_shape)
    actual = getattr(operator, operation)(value)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    sample, expected_sample = value[0, 0], expected[0, 0]
    sink = TensorStore(torch.full_like(expected_sample, 73.))
    assert getattr(operator, operation + "_into")(TensorStore(sample), sink) is sink
    torch.testing.assert_close(sink.tensor, expected_sample, rtol=2e-11, atol=2e-12)
    torch.testing.assert_close(actual, getattr(_operator(case), operation)(value), rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("surface_kind", ["callback", "parameterized"])
def test_detector_surfaces_use_three_dimensional_frame_and_parameter_gradients(surface_kind):
    case = _case()
    coefficient = torch.tensor(.12, dtype=torch.float64, requires_grad=True)

    def offsets(u, v, strength):
        return torch.stack((u, v, strength * (u.square() + .31 * u * v + .19 * v.square())), dim=-1)

    surface = ((lambda u, v: offsets(u, v, coefficient)) if surface_kind == "callback" else
               ParameterizedSurface(lambda u, v, ids, strength: offsets(u, v, strength), (coefficient,)))
    operator = _operator(case, detector_surface=surface, view_chunk_size=2,
                         detector_chunk_shape=(2, 2), volume_chunk_shape=(2, 2, 3))
    image, rays = _data(case.shape), _data(case.projection_shape, phase=.31)
    matrix = _matrix(case, curved=True)
    assert (matrix - _matrix(case)).abs().max() > 1e-3
    projected, adjoint = operator(image), operator.backproject(rays)
    torch.testing.assert_close(projected.flatten(), matrix @ image.flatten(), rtol=2e-11, atol=2e-12)
    torch.testing.assert_close(adjoint.flatten(), matrix.T @ rays.flatten(), rtol=2e-11, atol=2e-12)
    gradient, = torch.autograd.grad((projected * rays).sum(), coefficient)
    plus = rays.flatten() @ _matrix(case, curved=True, coefficient=.120001) @ image.flatten()
    minus = rays.flatten() @ _matrix(case, curved=True, coefficient=.119999) @ image.flatten()
    expected = (plus - minus) / 2e-6
    assert expected.abs() > 1e-3
    torch.testing.assert_close(gradient, expected, rtol=4e-5, atol=2e-6)


@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_all_geometry_components_match_independent_finite_differences(operation):
    base = _case()
    frames = tuple(value.clone().requires_grad_() for value in base.trajectory)
    case = replace(base, trajectory=frames)
    image, rays = _data(case.shape), _data(case.projection_shape, phase=.31)
    operator = _operator(case, view_chunk_size=2, detector_chunk_shape=(2, 2))
    value, weight = (image, rays) if operation == "project" else (rays, image)
    output = getattr(operator, operation)(value)
    gradients = torch.autograd.grad((output * weight).sum(), frames)
    for component, gradient in enumerate(gradients):
        expected = torch.empty_like(gradient)
        for view, axis in itertools.product(range(gradient.shape[0]), range(3)):
            plus, minus = [value.detach().clone() for value in frames], [value.detach().clone() for value in frames]
            plus[component][view, axis] += 1e-6
            minus[component][view, axis] -= 1e-6
            high = rays.flatten() @ _matrix(replace(case, trajectory=tuple(plus))) @ image.flatten()
            low = rays.flatten() @ _matrix(replace(case, trajectory=tuple(minus))) @ image.flatten()
            expected[view, axis] = (high - low) / 2e-6
        assert expected.abs().max() > 1e-4, f"uninformative component {component}"
        torch.testing.assert_close(gradient, expected, rtol=5e-5, atol=3e-6)


@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_data_autograd_func_transforms_and_second_derivatives_use_matched_linear_map(operation):
    case = replace(_case(), shape=(2, 2, 3), detector=(2, 1))
    matrix, operator = _matrix(case), _operator(case)
    shape = case.shape if operation == "project" else case.projection_shape
    matrix = matrix if operation == "project" else matrix.T
    value, tangent = _data(shape), _data(shape, phase=.73)
    function = getattr(operator, operation)
    output_shape = case.projection_shape if operation == "project" else case.shape
    weight = _data(output_shape, phase=.31)
    torch.testing.assert_close(torch.func.grad(lambda x: (function(x) * weight).sum())(value).flatten(), matrix.T @ weight.flatten(), rtol=2e-11, atol=2e-12)
    _, jvp = torch.func.jvp(function, (value,), (tangent,))
    torch.testing.assert_close(jvp.flatten(), matrix @ tangent.flatten(), rtol=2e-11, atol=2e-12)
    torch.testing.assert_close(torch.func.jacrev(function)(value), matrix.reshape(*output_shape, *shape), rtol=2e-11, atol=2e-12)
    torch.testing.assert_close(torch.func.jacfwd(function)(value), matrix.reshape(*output_shape, *shape), rtol=2e-11, atol=2e-12)
    batch = torch.stack((value, tangent))
    torch.testing.assert_close(torch.func.vmap(function)(batch).reshape(2, -1), batch.reshape(2, -1) @ matrix.T, rtol=2e-11, atol=2e-12)
    leaf = value.clone().requires_grad_()
    gradient, = torch.autograd.grad(function(leaf).square().sum() / 2, leaf, create_graph=True)
    second, = torch.autograd.grad((gradient * tangent).sum(), leaf)
    torch.testing.assert_close(second.flatten(), matrix.T @ matrix @ tangent.flatten(), rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("invalid", ["components", "dimensions", "views", "nonfinite", "ray_norm", "u_norm", "nonorthogonal", "tangent"])
def test_invalid_3d_parallel_frames_reject_before_execution(invalid):
    case = _case()
    frames = [value.clone() for value in case.trajectory]
    if invalid == "components":
        frames.pop()
    elif invalid == "dimensions":
        frames = [value[:, :2] for value in frames]
    elif invalid == "views":
        frames[3] = frames[3][:2]
    elif invalid == "nonfinite":
        frames[1][0, 0] = math.nan
    elif invalid == "ray_norm":
        frames[0] *= 2
    elif invalid == "u_norm":
        frames[2] *= 2
    elif invalid == "nonorthogonal":
        frames[3] = frames[2].clone()
    else:
        frames[0] = frames[2].clone()
    with pytest.raises((TypeError, ValueError)):
        _operator(replace(case, trajectory=tuple(frames)))


def test_each_single_view_rejects_a_ray_equal_to_its_detector_axis():
    case = _case()
    # Check views separately: an exactly-zero dot in another view must not
    # hide cancellation error when the ray lies in this detector plane.
    for view in range(len(case.trajectory[0])):
        frames = tuple(value[view:view + 1].clone() for value in case.trajectory)
        tangent = (frames[2].clone(), *frames[1:])
        with pytest.raises((TypeError, ValueError)):
            _operator(replace(case, trajectory=tangent))


def test_auto_fast_and_scalar_pitch_route_to_torch_and_explicit_numba_rejects(monkeypatch):
    monkeypatch.setattr(TorchCUDABridge, "tensor_to_cuda_array", _no_native)
    case = replace(_case(), spacing=.73, pitch=(.71, .71))
    operator = _operator(case, backend="auto", precision="fast", detector_spacing=.71)
    assert operator.backend == "torch" and operator.dtype == torch.float32
    image = _data(case.shape, dtype=torch.float32)
    torch.testing.assert_close(operator(image).double().flatten(), _matrix(case) @ image.double().flatten(), rtol=5e-5, atol=5e-6)
    with pytest.raises((ValueError, RuntimeError), match="(?i)(parallel3d|parallel.?3d|3d.parallel|numba)"):
        _operator(case, backend="numba", precision="fast")


def test_circular_3d_parallel_lifts_legacy_xy_convention_and_angle_gradients():
    generator = getattr(diffct, "circular_trajectory_3d_parallel", None)
    assert callable(generator), "required diffct.circular_trajectory_3d_parallel export is missing"
    assert "circular_trajectory_3d_parallel" in diffct.__all__
    angles = torch.tensor([1.13, -.27, .41], dtype=torch.float64, requires_grad=True)
    actual = generator(angles, detector_distance=.73, device="cuda", dtype=torch.float32)
    two_d = diffct.circular_trajectory_2d_parallel(angles, detector_distance=.73)
    for value, xy in zip(actual[:3], two_d):
        assert value.dtype == angles.dtype and value.device == angles.device
        torch.testing.assert_close(value[:, :2], xy, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(value[:, 2], torch.zeros_like(angles))
    torch.testing.assert_close(actual[3], torch.tensor([[0., 0., 1.]] * 3, dtype=torch.float64))
    assert torch.autograd.gradcheck(lambda a: generator(a, .73), (angles,))
    count = generator(4, .73, device="cpu", dtype=torch.float64)
    legacy = diffct.circular_trajectory_2d_parallel(4, .73, device="cpu", dtype=torch.float64)
    for value, xy in zip(count[:3], legacy):
        torch.testing.assert_close(value[:, :2], xy, rtol=1e-13, atol=1e-13)
    custom = generator(3, .73, start_angle=.23, end_angle=2.47, device="cpu", dtype=torch.float64)
    legacy_custom = diffct.circular_trajectory_2d_parallel(3, .73, start_angle=.23, end_angle=2.47, device="cpu", dtype=torch.float64)
    for value, xy in zip(custom[:3], legacy_custom):
        torch.testing.assert_close(value[:, :2], xy, rtol=1e-13, atol=1e-13)
    for bad in (torch.empty(0), torch.ones(2, 1), torch.tensor([1, 2]), torch.tensor([math.nan]), torch.tensor([math.inf])):
        with pytest.raises((TypeError, ValueError)):
            generator(bad, device="cpu")


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_cuda_forward_adjoint_geometry_and_data_gradients_match_cpu_cell_oracle(precision, monkeypatch):
    monkeypatch.setattr(TorchCUDABridge, "tensor_to_cuda_array", _no_native)
    case = _case()
    dtype = torch.float64 if precision == "float64" else torch.float32
    matrix = _matrix(case)
    frames = tuple(value.cuda().requires_grad_() for value in case.trajectory)
    operator = _operator(replace(case, trajectory=frames), precision=precision,
                         view_chunk_size=2, detector_chunk_shape=(2, 2), volume_chunk_shape=(2, 2, 3))
    image = _data(case.shape, device="cuda", dtype=dtype).requires_grad_()
    rays = _data(case.projection_shape, device="cuda", dtype=dtype, phase=.31).requires_grad_()
    forward, adjoint = operator(image), operator.backproject(rays)
    assert forward.device == adjoint.device == image.device
    assert forward.dtype == adjoint.dtype == dtype
    tolerance = 2e-11 if precision == "float64" else 8e-5
    torch.testing.assert_close(forward.detach().cpu().double().flatten(), matrix @ image.detach().cpu().double().flatten(), rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(adjoint.detach().cpu().double().flatten(), matrix.T @ rays.detach().cpu().double().flatten(), rtol=tolerance, atol=tolerance)
    gradients = torch.autograd.grad((forward * rays).sum(), (image, *frames))
    torch.testing.assert_close(gradients[0].cpu().double().flatten(), matrix.T @ rays.detach().cpu().double().flatten(), rtol=tolerance, atol=tolerance)
    cpu_frames = tuple(value.clone().requires_grad_() for value in case.trajectory)
    cpu_output = _operator(replace(case, trajectory=cpu_frames), precision=precision).project(image.detach().cpu())
    cpu_gradients = torch.autograd.grad((cpu_output * rays.detach().cpu()).sum(), cpu_frames)
    for actual, expected in zip(gradients[1:], cpu_gradients):
        torch.testing.assert_close(actual.cpu(), expected, rtol=3e-4 if precision == "float32" else 2e-10, atol=3e-5 if precision == "float32" else 2e-11)


def _distributed_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=15))
    try:
        case = _case()
        image, rays = _data(case.shape), _data(case.projection_shape, phase=.31)
        frames = tuple(value.clone().requires_grad_() for value in case.trajectory)
        # Moving centers changes physical rays; exercise override rather than the default.
        replacement = (frames[0], frames[1] + torch.tensor([.037, -.019, .023]), *frames[2:])
        matrix = _matrix(replace(case, trajectory=replacement))
        for partition in ("views", "space", "hybrid"):
            options = {"distributed": True, "partition": partition}
            if partition == "hybrid":
                options["partition_grid"] = (1, 2)
            operator = _operator(case, view_chunk_size=2, volume_chunk_shape=(2, 2, 3), **options)
            local_image = image[operator.volume_slice].clone().requires_grad_()
            local_rays = rays[operator.view_slice]
            output = operator.project(local_image, trajectory=replacement)
            expected = (matrix @ image.flatten()).reshape(case.projection_shape)[operator.view_slice]
            torch.testing.assert_close(output, expected, rtol=2e-11, atol=2e-12)
            gradients = torch.autograd.grad((output * local_rays).sum() / operator.space_world_size, (local_image, *frames))
            torch.testing.assert_close(gradients[0], (matrix.T @ rays.flatten()).reshape(case.shape)[operator.volume_slice], rtol=2e-11, atol=2e-12)
            # Independent finite difference for the detector-center x coordinate.
            plus, minus = [x.detach().clone() for x in replacement], [x.detach().clone() for x in replacement]
            plus[1][0, 0] += 1e-6
            minus[1][0, 0] -= 1e-6
            expected_gradient = (rays.flatten() @ (_matrix(replace(case, trajectory=tuple(plus))) - _matrix(replace(case, trajectory=tuple(minus)))) @ image.flatten()) / 2e-6
            torch.testing.assert_close(gradients[2][0, 0], expected_gradient, rtol=5e-5, atol=3e-6)
            local_sino = local_rays.clone().requires_grad_()
            back = operator.backproject(local_sino, trajectory=tuple(x.detach() for x in replacement))
            expected_back = (matrix.T @ rays.flatten()).reshape(case.shape)[operator.volume_slice]
            torch.testing.assert_close(back, expected_back, rtol=2e-11, atol=2e-12)
            data_gradient, = torch.autograd.grad((back * image[operator.volume_slice]).sum() / operator.view_world_size, local_sino)
            torch.testing.assert_close(data_gradient, expected, rtol=2e-11, atol=2e-12)
        # Admission failure on just one rank must propagate before any execution/write.
        operator = _operator(case, distributed=True)
        invalid = tuple(x[:2] for x in case.trajectory) if rank == 0 else case.trajectory
        sink = TensorStore(torch.full(operator.projection_shape, 73., dtype=torch.float64))
        rejected = False
        try:
            operator.project_into(image, sink, trajectory=invalid)
        except (TypeError, ValueError, RuntimeError):
            rejected = True
        flags = [None, None]
        dist.all_gather_object(flags, rejected)
        assert all(flags), "one-rank invalid geometry was not rejected on every rank"
        torch.testing.assert_close(sink.tensor, torch.full_like(sink.tensor, 73.))
        # Mixed None/override and mixed geometry-gradient participation: reject
        # coherently or run the same physical geometry and finish backward.
        mixed = frames if rank == 0 else None
        value = image.clone().requires_grad_()
        rejected = False
        try:
            output = operator.project(value, trajectory=mixed)
        except (TypeError, ValueError, RuntimeError):
            rejected = True
        dist.all_gather_object(flags, rejected)
        assert len(set(flags)) == 1, "mixed override admission differed between ranks"
        if not rejected:
            torch.testing.assert_close(output, (_matrix(case) @ image.flatten()).reshape(case.projection_shape)[operator.view_slice], rtol=2e-11, atol=2e-12)
            output.sum().backward()
        torch.testing.assert_close(operator.project(image), (_matrix(case) @ image.flatten()).reshape(case.projection_shape)[operator.view_slice], rtol=2e-11, atol=2e-12)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is required")
def test_gloo_partitions_override_gradients_and_collective_validation_do_not_hang(tmp_path):
    rendezvous = (tmp_path / "parallel3d-rendezvous").resolve().as_uri()
    context = mp.spawn(_distributed_worker, args=(rendezvous,), nprocs=2, join=False)
    deadline = time.monotonic() + 75
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise AssertionError("two-process parallel3d checks exceeded 75 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
