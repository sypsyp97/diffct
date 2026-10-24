"""Contract N1--N3 and Unchanged for native per-axis voxel spacing.

CPU run (only the new tests):
    NUMBA_ENABLE_CUDASIM=1 NUMBA_CUDA_USE_NVIDIA_BINDING=0 OMP_NUM_THREADS=1 \
        .venv/bin/python -m pytest -q tests/test_native_per_axis_spacing.py \
        tests/test_native_per_axis_spacing_gpu.py

The fixed scalar .npz reference and its main-only generation command are
documented in tests/_native_per_axis_spacing.py. These are acceptance tests:
per-axis native failures on the pre-feature main are deliberately not xfailed.
"""

from dataclasses import replace
import math

import numpy as np
import pytest
import torch
from numba.core import config

import diffct
from diffct import Projector, TensorStore
from diffct.utils import TorchCUDABridge
from tests._native_per_axis_spacing import (
    ANALYTICAL_ATOL, ANALYTICAL_RTOL, BEAMS, CHORD_ATOL, CHORD_RTOL,
    FLAT_ATOL, FLAT_RTOL, GEOMETRY_ATOL, GEOMETRY_RTOL, NATIVE_REQUIRED,
    PORTABLE_ATOL, PORTABLE_RTOL, REFERENCE, SCALAR_BEAMS, SURFACES,
    case, chunks, close, cuda_device, data, native_device, native_only,
    operator, scalar_outputs, signatures, surface,
)


EXCLUDED_MODES = [(precision, partition) for precision in ("fast", "float32", "float64")
                  for partition in ("views", "space", "hybrid")
                  if precision != "fast" or partition == "hybrid"]


@NATIVE_REQUIRED
@pytest.mark.parametrize("backend", ["numba", "auto"])
@pytest.mark.parametrize("partition", ["views", "space"])
@pytest.mark.parametrize("container", [tuple, list], ids=["tuple", "list"])
def test_n1_fast_per_axis_selects_native(native_device, backend, partition, container):
    scan = case("cone")
    scan = replace(scan, spacing=container(scan.spacing))
    with native_only():
        result = operator(scan, backend=backend, partition=partition)
    assert result.backend == "numba" and result.precision == "fast"
    assert result.dtype == torch.float32


@pytest.mark.parametrize("precision,partition", EXCLUDED_MODES)
@pytest.mark.parametrize("scalar", [False, True], ids=["per-axis", "scalar"])
def test_unchanged_n1_excluded_modes_remain_portable(precision, partition, scalar, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    result = operator(case("fan", scalar=scalar), backend="auto", precision=precision, partition=partition)
    assert result.backend == "torch"
    assert result.dtype == (torch.float64 if precision == "float64" else torch.float32)


@pytest.mark.parametrize("precision,partition", EXCLUDED_MODES)
def test_n1_numba_rejection_no_longer_requires_scalar_spacing(precision, partition, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError) as caught:
        operator(case("fan"), precision=precision, partition=partition)
    assert "scalar" not in str(caught.value).lower()
    assert "precision" in str(caught.value).lower() or "partition" in str(caught.value).lower()


@pytest.mark.parametrize("available,hip", [(False, None), (True, "6.0")], ids=["cpu", "rocm"])
def test_unchanged_auto_without_nvidia_is_portable(available, hip, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.version, "hip", hip)
    assert operator(case("fan"), backend="auto").backend == "torch"


@NATIVE_REQUIRED
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("kind", SURFACES)
@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("leading", [(), (2,)], ids=["unbatched", "batched"])
def test_n2_n3_values_and_data_autograd_match_portable(native_device, beam, kind, operation, leading):
    scan = case(beam)
    detector, _ = surface(kind)
    reference = operator(scan, backend="torch", precision="float64", detector_surface=detector)
    image, rays = data((*leading, *scan.shape)), data((*leading, *scan.projection_shape), phase=.31)
    value, weight = (image, rays) if operation == "project" else (rays, image)
    reference_data = value.double().requires_grad_()
    expected = getattr(reference, operation)(reference_data)
    expected_gradient, = torch.autograd.grad((expected * weight.double()).sum(), reference_data)
    native_data = value.to(native_device).requires_grad_()
    with native_only() as bridge:
        native = operator(scan, detector_surface=detector)
        assert native.backend == "numba"
        actual = getattr(native, operation)(native_data)
        gradient, = torch.autograd.grad((actual * weight.to(native_device)).sum(), native_data)
        assert bridge.call_count > 0
    assert actual.dtype == gradient.dtype == torch.float32
    assert actual.device == gradient.device == native_data.device
    assert expected.abs().max() > 0 and expected_gradient.abs().max() > 0
    close(actual, expected, kind=kind)
    close(gradient, expected_gradient, kind=kind)


@NATIVE_REQUIRED
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("kind", ["flat", "parameterized"])
def test_n3_anisotropic_exact_transpose(native_device, beam, kind):
    scan = case(beam)
    detector, _ = surface(kind)
    image = data(scan.shape, device=native_device)
    rays = data(scan.projection_shape, phase=.31, device=native_device)
    with native_only() as bridge:
        native = operator(scan, detector_surface=detector)
        forward, adjoint = native.project(image), native.backproject(rays)
        assert native.backend == "numba" and bridge.call_count > 0
    lhs, rhs = (forward.double() * rays.double()).sum(), (image.double() * adjoint.double()).sum()
    assert forward.abs().max() > 0 and adjoint.abs().max() > 0
    # test_detector_surfaces.py::test_curved_adjoint_inner_product.
    close(lhs, rhs, rtol=3e-5, atol=5e-5)


@NATIVE_REQUIRED
@pytest.mark.parametrize("beam", SCALAR_BEAMS)
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_n2_public_lowlevel_functions_and_autograd(native_device, beam, operation):
    scan = case(beam)
    image, rays = data(scan.shape), data(scan.projection_shape, phase=.31)
    value, weight = (image, rays) if operation == "project" else (rays, image)
    reference = operator(scan, backend="torch", precision="float64")
    reference_data = value.double().requires_grad_()
    expected = getattr(reference, operation)(reference_data)
    expected_gradient, = torch.autograd.grad((expected * weight.double()).sum(), reference_data)
    native_data = value.to(native_device).requires_grad_()
    if operation == "project":
        if beam == "cone":
            function, arguments = diffct.ConeProjectorFunction, (*scan.detector, *scan.pitch, scan.spacing)
        else:
            function = diffct.ParallelProjectorFunction if beam == "parallel" else diffct.FanProjectorFunction
            arguments = (scan.detector[0], scan.pitch[0], scan.spacing)
    elif beam == "cone":
        function, arguments = diffct.ConeBackprojectorFunction, (*scan.shape, *scan.pitch, scan.spacing)
    else:
        function = diffct.ParallelBackprojectorFunction if beam == "parallel" else diffct.FanBackprojectorFunction
        arguments = (scan.pitch[0], *scan.shape, scan.spacing)
    with native_only() as bridge:
        actual = function.apply(native_data, *scan.trajectory, *arguments)
        gradient, = torch.autograd.grad((actual * weight.to(native_device)).sum(), native_data)
        assert bridge.call_count > 0
    close(actual, expected)
    close(gradient, expected_gradient)


@NATIVE_REQUIRED
@pytest.mark.parametrize("beam", BEAMS)
def test_n3_axis_chords_and_voxel_weights_have_physical_tensor_axis_units(native_device, beam):
    scan = case(beam)
    rank = len(scan.shape)
    # Odd dimensions put the exact axis rays through voxel centers.
    shape = (3, 5, 7) if rank == 3 else (3, 5)
    directions = torch.cat((torch.eye(rank), -torch.eye(rank))).double()
    centers, u = 32 * directions, torch.roll(directions, 1, dims=1)
    geometry = ((directions, centers, u) if beam in ("parallel", "parallel3d") else (-centers, centers, u))
    if rank == 3:
        geometry += (torch.linalg.cross(directions, u, dim=-1),)
    scan = replace(scan, shape=shape, detector=(1,) * (rank - 1), trajectory=geometry)
    expected = torch.tensor([size * pitch for size, pitch in zip(shape[::-1], scan.spacing[::-1])] * 2)
    # An explicit physical matrix also tests each voxel weight, beyond a box sum.
    matrix = torch.zeros((2 * rank, *shape), dtype=torch.float64)
    for view in range(2 * rank):
        axis = rank - 1 - view % rank
        selected = [size // 2 for size in shape]
        selected[axis] = slice(None)
        matrix[(view, *selected)] = scan.spacing[axis]
    rays = data(scan.projection_shape, phase=.31, device=native_device)
    with native_only():
        native = operator(scan, **chunks(scan))
        actual = native.project(torch.ones(shape).to(native_device))
        adjoint = native.backproject(rays)
    close(actual.flatten(), expected, rtol=CHORD_RTOL, atol=CHORD_ATOL)
    close(adjoint.flatten(), matrix.reshape(2 * rank, -1).T @ rays.cpu().double().flatten(),
          rtol=CHORD_RTOL, atol=CHORD_ATOL)


def _forward_objective(scan, kind, image, rays, parameters=(3.7, .037)):
    detector, _ = surface(kind, values=parameters)
    reference = operator(scan, backend="torch", precision="float64", detector_surface=detector)
    return (reference.project(image.double()) * rays.double()).sum()


@NATIVE_REQUIRED
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("kind", ["flat", "parameterized"])
@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("tiled", [False, True], ids=["full", "chunked"])
def test_n3_geometry_vjp_matches_portable_and_portable_forward_fd(native_device, beam, kind, operation, tiled):
    scan = case(beam, learnable=True)
    detector, parameters = surface(kind, learnable=True)
    image, rays = data(scan.shape), data(scan.projection_shape, phase=.31)
    value, weight = (image, rays) if operation == "project" else (rays, image)
    leaves = (*scan.trajectory, *parameters)
    reference = operator(scan, backend="torch", precision="float64", detector_surface=detector)
    expected = getattr(reference, operation)(value.double())
    expected_gradients = torch.autograd.grad((expected * weight.double()).sum(), leaves)
    with native_only() as bridge:
        native = operator(scan, detector_surface=detector, **(chunks(scan) if tiled else {}))
        actual = getattr(native, operation)(value.to(native_device))
        gradients = torch.autograd.grad((actual.double() * weight.to(native_device).double()).sum(), leaves)
        assert native.backend == "numba" and bridge.call_count > 0
    assert any(expected.abs().max() > 1e-3 for expected in expected_gradients)
    for actual_grad, reference_grad in zip(gradients, expected_gradients):
        assert torch.isfinite(actual_grad).all()
        close(actual_grad, reference_grad, rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL)
    _check_forward_finite_differences(scan, kind, image, rays, gradients)


def _check_forward_finite_differences(scan, kind, image, rays, gradients):
    # Central differences of the independent portable FLOAT64 FORWARD, even
    # when the native operation under test was backproject.
    base = replace(scan, trajectory=tuple(value.detach() for value in scan.trajectory))
    for group, gradient in enumerate(gradients):
        estimates = []
        if group < len(scan.trajectory):
            component = base.trajectory[group]
            direction = data(component.shape, phase=.29 + group, dtype=torch.float64)
            if scan.beam in ("parallel", "parallel3d") and group == 0:
                direction -= (direction * component).sum(-1, keepdim=True) * component
            direction /= direction.norm()
            for step in (1e-6, 5e-7):
                objectives = []
                for sign in (-1, 1):
                    frames = list(base.trajectory)
                    frames[group] = component + sign * step * direction
                    objectives.append(_forward_objective(replace(base, trajectory=tuple(frames)), kind, image, rays))
                estimates.append((objectives[1] - objectives[0]) / (2 * step))
            derivative = (gradient.cpu() * direction).sum()
        else:
            index = group - len(scan.trajectory)
            for step in (1e-6, 5e-7):
                objectives = []
                for sign in (-1, 1):
                    values = [3.7, .037]
                    values[index] += sign * step
                    objectives.append(_forward_objective(base, kind, image, rays, tuple(values)))
                estimates.append((objectives[1] - objectives[0]) / (2 * step))
            derivative = gradient.cpu()
        # test_detector_surfaces.py::_finite_difference uses two FD steps to
        # exclude a changed voxel topology before comparing a native gradient.
        close(estimates[0], estimates[1], rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL)
        close(derivative, estimates[0], rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL)


@NATIVE_REQUIRED
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("kind", ["flat", "parameterized"])
@pytest.mark.parametrize("execution", ["chunked", "host-store"])
def test_n2_chunked_and_host_resident_streaming(native_device, beam, kind, execution):
    scan = case(beam)
    detector, _ = surface(kind)
    image, rays = data(scan.shape), data(scan.projection_shape, phase=.31)
    reference = operator(scan, backend="torch", precision="float64", detector_surface=detector)
    expected = reference.project(image.double()), reference.backproject(rays.double())
    with native_only() as bridge:
        native = operator(scan, detector_surface=detector, **chunks(scan))
        assert native.backend == "numba"
        for method, source, wanted in zip(("project", "backproject"), (image, rays), expected):
            if execution == "host-store":
                # Plain CPU stores must stay on CPU; no fake CUDA metadata.
                source_store = TensorStore(source.clone())
                sink = TensorStore(torch.full(wanted.shape, math.nan, dtype=torch.float32))
                assert getattr(native, method + "_into")(source_store, sink) is sink
                actual = sink.tensor
                assert actual.device.type == "cpu"
                assert torch.equal(source_store.tensor, source)
            else:
                actual = getattr(native, method)(source.to(native_device))
            close(actual, wanted, kind=kind)
        assert bridge.call_count > 0


@NATIVE_REQUIRED
def test_n2_native_volume_layout_round_trip(native_device):
    scan = case("cone")
    rays = data(scan.projection_shape, phase=.31, device=native_device)
    reference = operator(scan, backend="torch", precision="float64")
    wanted = reference.backproject(rays.cpu().double())
    with native_only():
        native = operator(scan, native_volume_layout=True)
        volume = native.backproject(rays)
        depth, height, _ = scan.shape
        assert volume.stride() == (1, depth, depth * height)
        assert volume.permute(2, 1, 0).is_contiguous()
        projected = native.project(volume)
    close(volume, wanted)
    close(projected, reference.project(wanted))


@NATIVE_REQUIRED
@pytest.mark.parametrize("beam", SCALAR_BEAMS)
@pytest.mark.parametrize("tiled", [False, True], ids=["full", "chunked"])
@pytest.mark.parametrize("scalar", [False, True], ids=["per-axis", "unchanged-scalar"])
def test_n2_n3_fbp_fdk_values_and_autograd(native_device, beam, tiled, scalar):
    scan = case(beam, scalar=scalar)
    angles = torch.tensor([.231, .811, 1.431], dtype=torch.float64)
    reconstruct = diffct.fdk if beam == "cone" else diffct.fbp
    options = dict(detector_spacing=scan.pitch if beam == "cone" else scan.pitch[0],
                   voxel_spacing=scan.spacing, isocenter=torch.zeros(len(scan.shape)),
                   view_chunk_size=2 if tiled else None,
                   volume_chunk_shape=tuple(max(1, n - 1) for n in scan.shape) if tiled else None)
    if beam != "cone":
        options["beam"] = beam
    reference_data = data(scan.projection_shape, phase=.31, dtype=torch.float64).requires_grad_()
    expected = reconstruct(reference_data, scan.trajectory, scan.shape, angles,
                           backend="torch", precision="float64", **options)
    weight = data(scan.shape, dtype=torch.float64)
    expected_gradient, = torch.autograd.grad((expected * weight).sum(), reference_data)
    native_data = reference_data.detach().float().to(native_device).requires_grad_()
    with native_only() as bridge:
        actual = reconstruct(native_data, scan.trajectory, scan.shape, angles, backend="numba", **options)
        gradient, = torch.autograd.grad((actual * weight.to(native_device)).sum(), native_data)
        assert bridge.call_count > 0
    close(actual, expected, rtol=ANALYTICAL_RTOL, atol=ANALYTICAL_ATOL)
    close(gradient, expected_gradient, rtol=ANALYTICAL_RTOL, atol=ANALYTICAL_ATOL)


@NATIVE_REQUIRED
@pytest.mark.parametrize("algorithm", ["cgls", "sirt", "tv_reconstruction", "cgls_into"])
@pytest.mark.parametrize("scalar", [False, True], ids=["per-axis", "unchanged-scalar"])
def test_n2_solvers_use_native_per_axis_operator(native_device, algorithm, tmp_path, scalar):
    scan = case("fan", scalar=scalar)
    reference = operator(scan, backend="torch", precision="float64")
    rays = reference.project(data(scan.shape, dtype=torch.float64).abs()).detach()
    solver = getattr(diffct, algorithm)
    options = dict(iterations=2, rtol=0., atol=0.)
    if algorithm == "tv_reconstruction":
        options.update(weight=.01)
    if algorithm == "cgls_into":
        expected_store = TensorStore(torch.zeros(scan.shape, dtype=torch.float64))
        expected = solver(reference, TensorStore(rays), expected_store,
                          workspace=tmp_path / "reference", **options).volume.tensor
    else:
        expected = solver(reference, rays, **options).volume
    with native_only() as bridge:
        native = operator(scan)
        if algorithm == "cgls_into":
            output = TensorStore(torch.zeros(scan.shape))
            result = solver(native, TensorStore(rays.float()), output,
                            workspace=tmp_path / "native", **options)
            assert result.volume is output
            actual = output.tensor
        else:
            actual = solver(native, rays.float().to(native_device), **options).volume
        assert native.backend == "numba" and bridge.call_count > 0
    # test_solver_api.py::test_matrix_solver_matches_independent_recurrence_and_records_initial_plus_completed_residuals.
    close(actual, expected, rtol=3e-5, atol=3e-6)


@NATIVE_REQUIRED
@pytest.mark.parametrize("beam", SCALAR_BEAMS)
@pytest.mark.parametrize("scalar", [False, True], ids=["per-axis", "unchanged-scalar"])
def test_n2_fbp_fdk_host_store_streaming(native_device, beam, scalar):
    scan = case(beam, scalar=scalar)
    angles = torch.tensor([.231, .811, 1.431], dtype=torch.float64)
    options = dict(detector_spacing=scan.pitch if beam == "cone" else scan.pitch[0],
                   voxel_spacing=scan.spacing, isocenter=torch.zeros(len(scan.shape)),
                   view_chunk_size=2, volume_chunk_shape=tuple(max(1, size - 1) for size in scan.shape))
    if beam != "cone":
        options["beam"] = beam
    source = data(scan.projection_shape, phase=.31)
    reconstruct = diffct.fdk if beam == "cone" else diffct.fbp
    reconstruct_into = diffct.fdk_into if beam == "cone" else diffct.fbp_into
    expected = reconstruct(source.double(), scan.trajectory, scan.shape, angles,
                           backend="torch", precision="float64", **options)
    source_store = TensorStore(source.clone())
    sink = TensorStore(torch.full(scan.shape, math.nan))
    with native_only() as bridge:
        assert reconstruct_into(source_store, sink, scan.trajectory, scan.shape, angles,
                                backend="numba", **options) is sink
        assert bridge.call_count > 0
    assert sink.tensor.device.type == "cpu" and sink.tensor.dtype == torch.float32
    assert torch.equal(source_store.tensor, source)
    close(sink.tensor, expected, rtol=ANALYTICAL_RTOL, atol=ANALYTICAL_ATOL)


@NATIVE_REQUIRED
@pytest.mark.parametrize("beam", SCALAR_BEAMS)
@pytest.mark.parametrize("kind", ["flat", "parameterized"])
def test_n3_equal_tuple_matches_scalar(native_device, beam, kind):
    scan = case(beam, scalar=True)
    detector, _ = surface(kind)
    image, rays = data(scan.shape, device=native_device), data(scan.projection_shape, device=native_device)
    with native_only():
        scalar = operator(scan, detector_surface=detector)
        isotropic = operator(replace(scan, spacing=(scan.spacing,) * len(scan.shape)), detector_surface=detector)
        for method, value in (("project", image), ("backproject", rays)):
            close(getattr(isotropic, method)(value), getattr(scalar, method)(value),
                  rtol=FLAT_RTOL, atol=FLAT_ATOL)


@NATIVE_REQUIRED
@pytest.mark.parametrize("beam,axis", [("fan", 0), ("fan", 1), ("cone", 0), ("cone", 1), ("cone", 2)])
@pytest.mark.parametrize("limit", ["1e6", "1e15", "float32"])
@pytest.mark.parametrize("kind", ["flat", "parameterized"])
def test_n3_fp32_geometry_admission_uses_each_axis_pitch(native_device, beam, axis, limit, kind):
    scan = case(beam)
    rank = len(scan.shape)
    direction = torch.eye(rank, dtype=torch.float64)[axis:axis + 1]
    u = torch.roll(direction, 1, dims=1)
    pitch = scan.spacing[rank - 1 - axis]
    source_distance, detector_distance = {"1e6": (2e6 * pitch, 2e6 * pitch),
                                         "1e15": (2e15 * pitch, 100 * pitch),
                                         "float32": (1e39, 100 * pitch)}[limit]
    frames = (-source_distance * direction, detector_distance * direction, u)
    if rank == 3:
        frames += (torch.linalg.cross(direction, u, dim=-1),)
    scan = replace(scan, trajectory=frames, detector=(1,) * (rank - 1))
    detector, _ = surface(kind)
    with native_only(), pytest.raises(ValueError, match=r"1e6|1e15|float32|finite"):
        native = operator(scan, detector_surface=detector)
        native.project(torch.ones(scan.shape).to(native_device))


@NATIVE_REQUIRED
@pytest.mark.parametrize("beam,axis", [("fan", 0), ("fan", 1), ("cone", 0), ("cone", 1), ("cone", 2)])
@pytest.mark.parametrize("kind", ["flat", "parameterized"])
def test_n3_in_range_distant_endpoints_are_admitted_per_axis(native_device, beam, axis, kind):
    scan = case(beam)
    rank = len(scan.shape)
    direction = torch.eye(rank, dtype=torch.float64)[axis:axis + 1]
    u = torch.roll(direction, 1, dims=1)
    tensor_axis = rank - 1 - axis
    # A coarse-pitch axis can have a physical distance above 1e6 while its
    # voxel distance remains valid. Using the smallest pitch must not reject it.
    radius = 875000 * scan.spacing[tensor_axis]
    frames = (-radius * direction, radius * direction, u)
    if rank == 3:
        frames += (torch.linalg.cross(direction, u, dim=-1),)
    shape = (3, 5, 7) if rank == 3 else (3, 5)
    scan = replace(scan, trajectory=frames, shape=shape, detector=(1,) * (rank - 1))
    detector, _ = surface(kind)
    with native_only():
        native = operator(scan, detector_surface=detector)
        actual = native.project(torch.ones(shape).to(native_device))
    expected = torch.full(scan.projection_shape, shape[tensor_axis] * scan.spacing[tensor_axis])
    close(actual, expected, rtol=CHORD_RTOL, atol=CHORD_ATOL)


@pytest.mark.parametrize("beam", BEAMS)
def test_unchanged_portable_anisotropic_values_and_gradients(beam, monkeypatch):
    from tests.test_parallel3d import _matrix as parallel3d_matrix
    from tests.test_reference_backend import _matrix

    def forbidden(*args, **kwargs):
        raise AssertionError("portable execution called a native CUDA bridge")

    monkeypatch.setattr(TorchCUDABridge, "tensor_to_cuda_array", forbidden)
    scan = case(beam)
    matrix = (parallel3d_matrix if beam == "parallel3d" else _matrix)(scan)
    portable = operator(scan, backend="torch", precision="float64")
    image = data(scan.shape, dtype=torch.float64).requires_grad_()
    rays = data(scan.projection_shape, phase=.31, dtype=torch.float64).requires_grad_()
    forward, adjoint = portable.project(image), portable.backproject(rays)
    expected_forward = (matrix @ image.flatten()).reshape(scan.projection_shape)
    expected_adjoint = (matrix.T @ rays.flatten()).reshape(scan.shape)
    image_gradient, = torch.autograd.grad((forward * rays.detach()).sum(), image)
    ray_gradient, = torch.autograd.grad((adjoint * image.detach()).sum(), rays)
    for actual, expected in ((forward, expected_forward), (adjoint, expected_adjoint),
                             (image_gradient, expected_adjoint), (ray_gradient, expected_forward)):
        close(actual, expected, rtol=PORTABLE_RTOL, atol=PORTABLE_ATOL)


def test_unchanged_public_api_signatures():
    with np.load(REFERENCE, allow_pickle=False) as snapshot:
        for name, signature in signatures().items():
            assert signature == snapshot["signature__" + name].item(), name


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("kind", ["flat", "parameterized"])
def test_unchanged_portable_geometry_gradients_and_fd_oracle(beam, kind):
    scan = case(beam, learnable=True)
    detector, parameters = surface(kind, learnable=True)
    image, rays = data(scan.shape), data(scan.projection_shape, phase=.31)
    reference = operator(scan, backend="torch", precision="float64", detector_surface=detector)
    loss = (reference.project(image.double()) * rays.double()).sum()
    gradients = torch.autograd.grad(loss, (*scan.trajectory, *parameters))
    _check_forward_finite_differences(scan, kind, image, rays, gradients)


@NATIVE_REQUIRED
@pytest.mark.skipif(not config.ENABLE_CUDASIM, reason="bitwise main reference was generated under CUDASIM")
@pytest.mark.parametrize("beam", SCALAR_BEAMS)
@pytest.mark.parametrize("kind", ["flat", "parameterized"])
@pytest.mark.parametrize("tiled", [False, True], ids=["full", "chunked"])
def test_unchanged_scalar_native_is_bitwise_main(native_device, beam, kind, tiled):
    prefix = f"{beam}__{kind}__{'chunked' if tiled else 'full'}__"
    with native_only():
        actual = scalar_outputs(case(beam, scalar=True), kind, tiled, native_device)
    with np.load(REFERENCE, allow_pickle=False) as snapshot:
        assert snapshot["source_commit"].item() == "c4ef543"
        for name, value in actual.items():
            expected = snapshot[prefix + name]
            assert value.dtype == expected.dtype == np.float32
            assert value.shape == expected.shape
            # Compare representations, including signed zero, rather than floats.
            np.testing.assert_array_equal(value.view(np.uint32), expected.view(np.uint32))
