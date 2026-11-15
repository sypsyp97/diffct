"""Native (Numba CUDA) ``beam="parallel3d"`` contract.

Reference values come from the portable float64 backend, from independent
scalar cell intersections, or from closed-form box chords. No expected value
comes from a native parallel3d implementation.

Admission tests run on every host. Native tests need NVIDIA CUDA or the
simulator:

    NUMBA_ENABLE_CUDASIM=1 NUMBA_CUDA_USE_NVIDIA_BINDING=0 OMP_NUM_THREADS=1 \
        .venv/bin/python -m pytest -q tests/test_parallel3d_native.py

Tolerances are the constants that the native cone, fan and parallel tests
already use (tests/_native_per_axis_spacing.py). These are acceptance tests:
failures on the pre-feature tree are deliberately not xfailed.
"""

from contextlib import contextmanager
from dataclasses import replace
import math
import warnings

import numpy as np
import pytest
import torch

from diffct import NpyStore, Projector, TensorStore
from diffct.utils import TorchCUDABridge
from tests._native_per_axis_spacing import (
    CHORD_ATOL, CHORD_RTOL, FLAT_ATOL, FLAT_RTOL, GEOMETRY_ATOL, GEOMETRY_RTOL,
    NATIVE_REQUIRED, PORTABLE_ATOL, PORTABLE_RTOL, SURFACES,
    Case, case, chunks, close, cuda_device, data, native_device as native_device,
    native_only, operator, surface,
)
from tests.test_native_per_axis_admission import boundary_scan
from tests.test_parallel3d import _case as _oblique_case, _matrix as _cell_matrix


EXCLUDED_MODES = [(precision, partition) for precision in ("fast", "float32", "float64")
                  for partition in ("views", "space", "hybrid")
                  if precision != "fast" or partition == "hybrid"]
SPACING_LIMITS = {"ratio": (1e-3, 1., 20.), "range": (.5e-6, .5e-6, .5e-6)}
mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS device required")


def _scan(*, learnable=False):
    """Oblique rays, a non-cubic volume and three different voxel pitches."""
    base = _oblique_case()
    # Both backends receive exactly the same represented trajectory coordinates.
    frames = tuple(value.float().double().requires_grad_(learnable) for value in base.trajectory)
    return Case("parallel3d", frames, base.shape, base.detector, base.pitch, base.spacing)


def _replacement(scan, *, learnable=False):
    """A second valid geometry: other view order and moved detector centers."""
    frames = [value.detach().flip(0).clone() for value in scan.trajectory]
    frames[1] = frames[1] + torch.tensor([.037, -.019, .023], dtype=torch.float64)
    return tuple(value.float().double().requires_grad_(learnable) for value in frames)


def _portable(scan, **options):
    return operator(scan, backend="torch", precision="float64", **options)


def _patch_nvidia(monkeypatch, *, available=True, hip=None):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.version, "hip", hip)


def _backend_messages(caught):
    return [str(item.message) for item in caught
            if "parallel3d" in str(item.message) or "portable PyTorch backend" in str(item.message)]


def _check_ray_convention(directions, gradients, label):
    """Clarified item 3: the operator depends on ray_dir only through its direction."""
    directions = directions.detach().cpu().double()
    for name, gradient in zip(("ray_dir", "det_center"), gradients[:2]):
        gradient = gradient.detach().cpu().double()
        along = (gradient * directions).sum(-1).abs()
        # The elementwise geometry tolerance, applied to the per-view gradient length.
        allowed = GEOMETRY_ATOL + GEOMETRY_RTOL * gradient.norm(dim=-1)
        assert (along <= allowed).all(), (
            f"{label}: the {name} gradient must have no component along ray_dir "
            f"(portable convention: infinite lines, ray_dir used only as a direction); "
            f"along-ray components {along.tolist()}, allowed {allowed.tolist()}")


def _error(scan, **options):
    with pytest.raises(Exception) as caught:
        operator(scan, backend="numba", **options)
    return type(caught.value), str(caught.value)


def test_fixture_is_oblique_noncubic_and_anisotropic():
    scan = _scan()
    direction, _, u, v = scan.trajectory
    normal = torch.linalg.cross(u, v, dim=-1)
    # Item 3: the ray direction is not the detector normal in any view.
    assert (torch.linalg.cross(direction, normal, dim=-1).norm(dim=-1) > .2).all()
    assert all((direction * axis).sum(-1).abs().min() > .1 for axis in (u, v))
    assert len(set(scan.shape)) == 3 and len(set(scan.spacing)) == 3
    assert len(set(scan.detector)) == 2 and len(set(scan.pitch)) == 2


# Item 1: explicit numba requests fail exactly as they do for cone.

@pytest.mark.parametrize("available,hip", [(False, None), (True, "6.0")], ids=["cpu", "rocm"])
def test_numba_without_nvidia_cuda_raises_the_cone_error(available, hip, monkeypatch):
    _patch_nvidia(monkeypatch, available=available, hip=hip)
    expected = _error(case("cone"))
    assert "NVIDIA" in expected[1], "cone reference error changed"
    assert _error(_scan()) == expected


@pytest.mark.parametrize("precision,partition", EXCLUDED_MODES)
def test_numba_with_excluded_precision_or_partition_raises_the_cone_error(precision, partition, monkeypatch):
    _patch_nvidia(monkeypatch)
    expected = _error(case("cone"), precision=precision, partition=partition)
    assert "precision" in expected[1] or "partition" in expected[1], "cone reference error changed"
    assert _error(_scan(), precision=precision, partition=partition) == expected


@pytest.mark.parametrize("limit", sorted(SPACING_LIMITS))
def test_numba_with_incompatible_voxel_spacing_raises_the_cone_error(limit, monkeypatch):
    _patch_nvidia(monkeypatch)
    spacing = SPACING_LIMITS[limit]
    expected = _error(replace(case("cone"), spacing=spacing))
    assert "voxel_spacing" in expected[1], "cone reference error changed"
    assert _error(replace(_scan(), spacing=spacing)) == expected


SCALAR_OUTSIDE_PER_AXIS_LIMITS = (.999e-6, 1.001e6)


@pytest.mark.parametrize("backend", ["numba", "auto"])
@pytest.mark.parametrize("spacing", SCALAR_OUTSIDE_PER_AXIS_LIMITS)
def test_scalar_spacing_outside_per_axis_pitch_limits_is_admitted_as_for_cone(backend, spacing, monkeypatch):
    # Mirrors tests/test_native_per_axis_admission.py::
    # test_unchanged_scalar_admission_outside_per_axis_pitch_limits. Construction only.
    _patch_nvidia(monkeypatch)
    assert spacing < 1e-6 or spacing > 1e6
    # The cone scan of that test: axis rays, detector centers 4 volume extents away.
    cone = boundary_scan("cone", spacing)
    _, centers, u, v = cone.trajectory
    directions = centers / centers.norm(dim=-1, keepdim=True)
    scan = replace(cone, beam="parallel3d", trajectory=(directions, centers, u, v))
    assert operator(scan, backend="torch", precision="float64").backend == "torch", "invalid fixture"
    with native_only():
        reference = operator(cone, backend=backend)
        assert reference.backend == "numba", "cone reference admission changed"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            native = operator(scan, backend=backend)
    assert native.backend == "numba" and native.precision == "fast"
    assert not _backend_messages(caught)


# Item 2: automatic selection.

@pytest.mark.parametrize("backend", ["numba", "auto"])
@pytest.mark.parametrize("partition", ["views", "space"])
@pytest.mark.parametrize("scalar", [False, True], ids=["per-axis", "scalar"])
def test_reported_nvidia_cuda_constructs_native_without_a_warning(backend, partition, scalar, monkeypatch):
    # Construction only, so this runs on hosts without a GPU and without the simulator.
    _patch_nvidia(monkeypatch)
    scan = replace(_scan(), spacing=.75) if scalar else _scan()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        native = operator(scan, backend=backend, partition=partition)
    assert native.backend == "numba" and native.precision == "fast"
    assert native.dtype == torch.float32
    assert native.volume_shape == scan.shape and native.projection_shape == scan.projection_shape
    assert not _backend_messages(caught)

@pytest.mark.parametrize("available,hip", [(False, None), (True, "6.0")], ids=["cpu", "rocm"])
def test_unchanged_auto_without_nvidia_cuda_is_portable_and_silent(available, hip, monkeypatch):
    # Regression guard: this already holds on the pre-feature tree.
    _patch_nvidia(monkeypatch, available=available, hip=hip)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = operator(_scan(), backend="auto")
    assert result.backend == "torch" and result.dtype == torch.float32
    assert not [item for item in caught if "backend" in str(item.message)]


@mps
def test_unchanged_auto_on_mps_is_portable_and_matches_cpu():
    # Regression guard: this already holds on the pre-feature tree.
    scan = _scan()
    image = data(scan.shape, phase=0.31)
    frames = tuple(value.float().to("mps") for value in scan.trajectory)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = operator(replace(scan, trajectory=frames), backend="auto")
    assert result.backend == "torch"
    assert not [item for item in caught if "backend" in str(item.message)]
    expected = operator(scan, backend="torch").project(image)
    # tests/test_mps_backend.py::test_mps_parallel3d_matches_cpu.
    torch.testing.assert_close(result.project(image.to("mps")).cpu(), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("precision,partition", EXCLUDED_MODES)
def test_auto_fallback_warning_names_the_mode_and_not_the_beam(precision, partition, monkeypatch):
    _patch_nvidia(monkeypatch)
    with pytest.warns(UserWarning, match="portable PyTorch backend") as caught:
        result = operator(_scan(), backend="auto", precision=precision, partition=partition)
    assert result.backend == "torch"
    messages = [str(item.message) for item in caught if "portable PyTorch backend" in str(item.message)]
    assert len(messages) == 1
    if precision != "fast":
        assert f"precision={precision!r}" in messages[0]
    if partition == "hybrid":
        assert "partition='hybrid'" in messages[0]
    assert "parallel3d" not in messages[0]


@pytest.mark.parametrize("limit", sorted(SPACING_LIMITS))
def test_auto_fallback_warning_names_the_spacing_and_not_the_beam(limit, monkeypatch):
    _patch_nvidia(monkeypatch)
    with pytest.warns(UserWarning, match="portable PyTorch backend") as caught:
        result = operator(replace(_scan(), spacing=SPACING_LIMITS[limit]), backend="auto")
    assert result.backend == "torch"
    messages = [str(item.message) for item in caught if "portable PyTorch backend" in str(item.message)]
    assert len(messages) == 1 and "voxel_spacing" in messages[0]
    assert "parallel3d" not in messages[0]


@NATIVE_REQUIRED
@pytest.mark.parametrize("backend", ["numba", "auto"])
@pytest.mark.parametrize("partition", ["views", "space"])
def test_numba_and_auto_select_native_without_a_fallback_warning(native_device, backend, partition):
    scan = _scan()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        native = operator(scan, backend=backend, partition=partition)
    assert native.backend == "numba" and native.precision == "fast"
    assert native.dtype == torch.float32
    assert native.volume_shape == scan.shape and native.projection_shape == scan.projection_shape
    assert not _backend_messages(caught)
    image = data(scan.shape, phase=0.46, device=native_device)
    rays = data(scan.projection_shape, phase=0.96, device=native_device)
    # The frames hold 9 elements each, so the data sizes identify the data arrays.
    assert image.numel() != rays.numel() and scan.trajectory[0].numel() not in (image.numel(), rays.numel())
    with native_only() as bridge:
        projected = native.project(image)
        sizes = {call.args[0].numel() for call in bridge.call_args_list}
        assert image.numel() in sizes, "the volume did not reach the native array bridge"
        bridge.reset_mock()
        adjoint = native.backproject(rays)
        sizes = {call.args[0].numel() for call in bridge.call_args_list}
        assert rays.numel() in sizes, "the sinogram did not reach the native array bridge"
    reference = _portable(scan)
    close(projected, reference.project(image.cpu().double()))
    close(adjoint, reference.backproject(rays.cpu().double()))


@NATIVE_REQUIRED
def test_auto_fast_scalar_spacings_run_native_and_match_cell_matrix(native_device):
    # NVIDIA counterpart of tests/test_parallel3d.py::
    # test_auto_fast_and_scalar_pitch_route_to_torch_without_nvidia_cuda.
    scan = replace(_scan(), spacing=.73, pitch=(.71, .71))
    matrix = _cell_matrix(scan)
    assert matrix.abs().sum() > 0, "fixture needs nonzero intersections"
    image, rays = data(scan.shape, phase=0.48), data(scan.projection_shape, phase=0.98)
    with native_only() as bridge:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            native = Projector(scan.trajectory, scan.shape, scan.detector, beam="parallel3d",
                               detector_spacing=.71, voxel_spacing=.73, backend="auto", precision="fast")
        assert native.backend == "numba" and native.dtype == torch.float32
        assert not _backend_messages(caught)
        projected = native(image.to(native_device))
        adjoint = native.backproject(rays.to(native_device))
        assert bridge.call_count > 0
    close(projected.flatten(), matrix @ image.double().flatten())
    close(adjoint.flatten(), matrix.T @ rays.double().flatten())


# Items 3 and 5: values, data gradients and detector surfaces against portable.

@NATIVE_REQUIRED
@pytest.mark.parametrize("kind", SURFACES)
@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("leading", [(), (2,)], ids=["unbatched", "batched"])
def test_values_and_data_autograd_match_portable(native_device, kind, operation, leading):
    scan = _scan()
    detector, _ = surface(kind)
    image, rays = data((*leading, *scan.shape), phase=0.51), data((*leading, *scan.projection_shape), phase=1.01)
    value, weight = (image, rays) if operation == "project" else (rays, image)
    reference_data = value.double().requires_grad_()
    expected = getattr(_portable(scan, detector_surface=detector), operation)(reference_data)
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
    if kind != "flat":
        flat = getattr(_portable(scan), operation)(value.double())
        assert (expected - flat).abs().max() > 1e-3, "surface fixture must change the rays"
    close(actual, expected, kind=kind)
    close(gradient, expected_gradient, kind=kind)


@NATIVE_REQUIRED
@pytest.mark.parametrize("operation", ["__call__", "project", "backproject"])
def test_call_and_trajectory_override_match_portable(native_device, operation):
    scan = _scan()
    frames = _replacement(scan, learnable=True)
    moved = replace(scan, trajectory=frames)
    image, rays = data(scan.shape, phase=0.56), data(scan.projection_shape, phase=1.08)
    value, weight = (rays, image) if operation == "backproject" else (image, rays)
    method = "backproject" if operation == "backproject" else "project"
    reference_data = value.double().requires_grad_()
    default = getattr(_portable(scan), method)(value.double())
    expected = getattr(_portable(moved), method)(reference_data)
    wanted = torch.autograd.grad((expected * weight.double()).sum(), (reference_data, *frames))
    assert (expected - default).abs().max() > 1e-3, "override fixture must change the rays"
    native_data = value.to(native_device).requires_grad_()
    with native_only() as bridge:
        native = operator(scan)
        assert native.backend == "numba"
        apply = native if operation == "__call__" else getattr(native, operation)
        before = apply(native_data.detach())
        actual = apply(native_data, trajectory=frames)
        gradients = torch.autograd.grad((actual * weight.to(native_device)).sum(), (native_data, *frames))
        after = apply(native_data.detach(), trajectory=None)
        assert bridge.call_count > 0
    close(before, default)
    close(after, default)
    close(actual, expected)
    close(gradients[0], wanted[0])
    assert any(gradient.abs().max() > 1e-3 for gradient in wanted[1:])
    for actual_gradient, expected_gradient in zip(gradients[1:], wanted[1:]):
        assert torch.isfinite(actual_gradient).all()
        close(actual_gradient, expected_gradient, rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL)
    _check_ray_convention(frames[0], gradients[1:], "native override")


@NATIVE_REQUIRED
@pytest.mark.parametrize("kind", ["flat", "parameterized"])
@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("tiled", [False, True], ids=["full", "chunked"])
def test_geometry_and_surface_parameter_gradients_match_portable(native_device, kind, operation, tiled):
    scan = _scan(learnable=True)
    detector, parameters = surface(kind, learnable=True)
    image, rays = data(scan.shape, phase=0.61), data(scan.projection_shape, phase=1.15)
    value, weight = (image, rays) if operation == "project" else (rays, image)
    leaves = (*scan.trajectory, *parameters)
    expected = getattr(_portable(scan, detector_surface=detector), operation)(value.double())
    expected_gradients = torch.autograd.grad((expected * weight.double()).sum(), leaves)
    with native_only() as bridge:
        native = operator(scan, detector_surface=detector, **(chunks(scan) if tiled else {}))
        actual = getattr(native, operation)(value.to(native_device))
        gradients = torch.autograd.grad((actual.double() * weight.to(native_device).double()).sum(), leaves)
        assert native.backend == "numba" and bridge.call_count > 0
    assert len(gradients) == 4 + len(parameters)
    for actual_gradient, expected_gradient in zip(gradients, expected_gradients):
        assert expected_gradient.abs().max() > 1e-3, "uninformative geometry component"
        assert torch.isfinite(actual_gradient).all()
        close(actual_gradient, expected_gradient, rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL)
    _check_portable_ray_convention(scan.trajectory[0], expected_gradients)
    _check_ray_convention(scan.trajectory[0], gradients, "native")


def _check_portable_ray_convention(directions, gradients):
    # Fixture guard: the reference follows the convention to float64 rounding,
    # and both gradients are far from zero, so the native check is informative.
    for gradient in gradients[:2]:
        along = (gradient * directions.detach()).sum(-1).abs()
        assert (along <= 1e-11 * gradient.norm(dim=-1)).all(), "portable reference convention changed"
        assert gradient.norm(dim=-1).min() > .5, "uninformative ray convention fixture"


# Item 3 and clarified item 1: a scalar voxel_spacing takes its own native branch.

@NATIVE_REQUIRED
@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("tiled", [False, True], ids=["full", "chunked"])
def test_scalar_voxel_spacing_values_data_and_geometry_gradients_match_portable(native_device, operation, tiled):
    scan = replace(_scan(learnable=True), spacing=.75)
    assert isinstance(scan.spacing, float)
    image, rays = data(scan.shape, phase=0.63), data(scan.projection_shape, phase=1.17)
    value, weight = (image, rays) if operation == "project" else (rays, image)
    reference_data = value.double().requires_grad_()
    expected = getattr(_portable(scan), operation)(reference_data)
    wanted = torch.autograd.grad((expected * weight.double()).sum(), (reference_data, *scan.trajectory))
    # The scalar pitch is not a relabelled per-axis fixture.
    other = getattr(_portable(_scan()), operation)(value.double())
    assert (expected - other).abs().max() > 1e-2, "scalar fixture must change the values"
    native_data = value.to(native_device).requires_grad_()
    with native_only() as bridge:
        native = operator(scan, **(chunks(scan) if tiled else {}))
        assert native.backend == "numba"
        actual = getattr(native, operation)(native_data)
        gradients = torch.autograd.grad((actual.double() * weight.to(native_device).double()).sum(),
                                        (native_data, *scan.trajectory))
        assert bridge.call_count > 0
    assert actual.dtype == gradients[0].dtype == torch.float32
    assert expected.abs().max() > 0 and wanted[0].abs().max() > 0
    close(actual, expected)
    close(gradients[0], wanted[0])
    for actual_gradient, expected_gradient in zip(gradients[1:], wanted[1:]):
        assert expected_gradient.abs().max() > 1e-3, "uninformative geometry component"
        assert torch.isfinite(actual_gradient).all()
        close(actual_gradient, expected_gradient, rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL)
    _check_portable_ray_convention(scan.trajectory[0], wanted[1:])
    _check_ray_convention(scan.trajectory[0], gradients[1:], "native scalar spacing")


CHUNKINGS = {
    "volume": lambda scan: dict(volume_chunk_shape=tuple(max(1, size - 1) for size in scan.shape)),
    "views": lambda scan: dict(view_chunk_size=2),
    "detector": lambda scan: dict(detector_chunk_shape=(2, 2)),
    "all": chunks,
}


@NATIVE_REQUIRED
@pytest.mark.parametrize("chunking", sorted(CHUNKINGS))
@pytest.mark.parametrize("kind", ["flat", "parameterized"])
def test_volume_view_and_detector_chunking_match_portable(native_device, chunking, kind):
    scan = _scan()
    detector, _ = surface(kind)
    image, rays = data(scan.shape, phase=0.66), data(scan.projection_shape, phase=1.22)
    reference = _portable(scan, detector_surface=detector)
    expected = reference.project(image.double()), reference.backproject(rays.double())
    with native_only() as bridge:
        native = operator(scan, detector_surface=detector, **CHUNKINGS[chunking](scan))
        assert native.backend == "numba"
        actual = native.project(image.to(native_device)), native.backproject(rays.to(native_device))
        assert bridge.call_count > 0
    for result, wanted in zip(actual, expected):
        close(result, wanted, kind=kind)


def _store(kind, tensor, path):
    if kind == "tensor":
        return TensorStore(tensor)
    np.save(path, tensor.numpy())
    return NpyStore(path, mode="r+")


@NATIVE_REQUIRED
@pytest.mark.parametrize("store", ["tensor", "npy"])
@pytest.mark.parametrize("kind", ["flat", "parameterized"])
def test_streamed_into_tensor_and_block_stores_match_portable(native_device, store, kind, tmp_path):
    scan = _scan()
    detector, _ = surface(kind)
    image, rays = data(scan.shape, phase=0.71), data(scan.projection_shape, phase=1.29)
    reference = _portable(scan, detector_surface=detector)
    expected = reference.project(image.double()), reference.backproject(rays.double())
    with native_only() as bridge:
        native = operator(scan, detector_surface=detector, **chunks(scan))
        assert native.backend == "numba"
        for method, source, wanted in zip(("project", "backproject"), (image, rays), expected):
            source_store = _store(store, source.clone(), tmp_path / f"{method}-input.npy")
            sink = _store(store, torch.full(wanted.shape, math.nan, dtype=torch.float32),
                          tmp_path / f"{method}-output.npy")
            # Plain CPU stores must stay on CPU.
            assert getattr(native, method + "_into")(source_store, sink) is sink
            sink.flush()
            actual = (sink.tensor if store == "tensor"
                      else torch.from_numpy(np.load(tmp_path / f"{method}-output.npy")))
            assert actual.dtype == torch.float32 and actual.device.type == "cpu"
            assert torch.equal(source_store.read((slice(None),) * source.ndim), source)
            close(actual, wanted, kind=kind)
        assert bridge.call_count > 0


@NATIVE_REQUIRED
def test_native_volume_layout_option_keeps_portable_values(native_device):
    # Values only. The element order of the result is not part of this test.
    scan = _scan()
    rays = data(scan.projection_shape, phase=1.36, device=native_device)
    reference = _portable(scan)
    wanted = reference.backproject(rays.cpu().double())
    with native_only() as bridge:
        native = operator(scan, native_volume_layout=True)
        assert native.backend == "numba"
        volume = native.backproject(rays)
        projected = native.project(volume)
        assert bridge.call_count > 0
    assert tuple(volume.shape) == scan.shape
    close(volume, wanted)
    close(projected, reference.project(wanted))


# Item 4: the native projector and backprojector are a transpose pair.

@NATIVE_REQUIRED
@pytest.mark.parametrize("kind", ["flat", "parameterized"])
@pytest.mark.parametrize("tiled", [False, True], ids=["full", "chunked"])
def test_adjoint_inner_product(native_device, kind, tiled):
    scan = _scan()
    detector, _ = surface(kind)
    image = data(scan.shape, phase=0.81, device=native_device)
    rays = data(scan.projection_shape, phase=1.43, device=native_device)
    with native_only() as bridge:
        native = operator(scan, detector_surface=detector, **(chunks(scan) if tiled else {}))
        forward, adjoint = native.project(image), native.backproject(rays)
        assert native.backend == "numba" and bridge.call_count > 0
    lhs, rhs = (forward.double() * rays.double()).sum(), (image.double() * adjoint.double()).sum()
    assert forward.abs().max() > 0 and adjoint.abs().max() > 0 and lhs.abs() > 1e-2
    # tests/test_detector_surfaces.py::test_curved_adjoint_inner_product.
    close(lhs, rhs, rtol=3e-5, atol=5e-5)


# Item 7: closed-form box chords, independent of both backends.

CHORD_BACKENDS = [pytest.param("torch", id="portable-float64"),
                  pytest.param("numba", id="native", marks=NATIVE_REQUIRED)]
DENSITY = 1.25


@contextmanager
def _chord_operator(request, backend, scan):
    """Yield the operator, its input device and its data dtype."""
    if backend == "torch":
        # This parameter checks the closed forms themselves on every host.
        yield _portable(scan), torch.device("cpu"), torch.float64
        return
    device = request.getfixturevalue("native_device")
    # Construction and every operation must stay out of the portable backend.
    with native_only() as bridge:
        native = operator(scan)
        assert native.backend == "numba"
        yield native, device, torch.float32
        assert bridge.call_count > 0, "the chord operations made no native array bridge call"


def _box_chord(origin, direction, half):
    """Length of an infinite line inside the box [-half, half], world (x, y, z)."""
    enter, leave = -math.inf, math.inf
    for p, d, h in zip(origin, direction, half):
        if d == 0.:
            if not -h <= p < h:
                return 0.
            continue
        first, last = (-h - p) / d, (h - p) / d
        enter, leave = max(enter, min(first, last)), min(leave, max(first, last))
    return max(0., leave - enter) * math.sqrt(sum(d * d for d in direction))


@pytest.mark.parametrize("backend", CHORD_BACKENDS)
def test_axis_aligned_chords_of_a_constant_box(request, backend):
    # Odd dimensions put the exact axis rays through voxel centers.
    shape, spacing = (3, 5, 7), (.7, 1., 1.9)
    axes = torch.cat((torch.eye(3), -torch.eye(3))).double()
    # An infinite line gives the same chord for a detector behind, inside or in front of the box.
    distances = (32., 0., -32.)
    directions = axes.repeat(len(distances), 1)
    centers = torch.cat([distance * axes for distance in distances])
    u = torch.roll(directions, 1, dims=1)
    v = torch.linalg.cross(directions, u, dim=-1)
    scan = Case("parallel3d", (directions, centers, u, v), shape, (1, 1), (.91, .67), spacing)
    lengths = [size * pitch for size, pitch in zip(shape[::-1], spacing[::-1])]
    assert lengths == pytest.approx([13.3, 5., 2.1])
    expected = DENSITY * torch.tensor(lengths * 2 * len(distances), dtype=torch.float64)
    # Each central ray crosses one full row of voxels along one tensor axis.
    matrix = torch.zeros((len(directions), *shape), dtype=torch.float64)
    for view in range(len(directions)):
        axis = 2 - view % 3
        selected = [size // 2 for size in shape]
        selected[axis] = slice(None)
        matrix[(view, *selected)] = spacing[axis]
    with _chord_operator(request, backend, scan) as (native, device, dtype), torch.no_grad():
        rays = data(scan.projection_shape, phase=1.50, device=device, dtype=dtype)
        actual = native.project(torch.full(shape, DENSITY, dtype=dtype).to(device))
        adjoint = native.backproject(rays)
    close(actual.flatten(), expected, rtol=CHORD_RTOL, atol=CHORD_ATOL)
    close(adjoint.flatten(), matrix.reshape(len(directions), -1).T @ rays.cpu().double().flatten(),
          rtol=CHORD_RTOL, atol=CHORD_ATOL)


def _oblique_chord_scan():
    shape, detector, pitch, spacing = (4, 5, 6), (3, 2), (.83, .61), (1.3, .9, .6)
    u = torch.tensor([.3, -.8, .52], dtype=torch.float64)
    u = u / u.norm()
    v = torch.linalg.cross(u, torch.tensor([.2, .1, 1.], dtype=torch.float64), dim=-1)
    v = v / v.norm()
    direction = torch.linalg.cross(u, v, dim=-1) + .23 * u - .17 * v
    direction = direction / direction.norm()
    offset = torch.tensor([.113, -.071, .057], dtype=torch.float64)
    # Views 0-2: one line, detector behind, inside and in front of the box.
    # View 3: the opposite direction sign. View 4: every ray misses the box.
    signs = (1., 1., 1., -1., 1.)
    centers = [offset - 12. * direction, offset, offset + 12. * direction,
               offset + torch.tensor([.041, .029, -.037], dtype=torch.float64) - 12. * direction,
               offset + 30. * u]
    frames = (torch.stack([sign * direction for sign in signs]), torch.stack(centers),
              u.expand(5, 3).clone(), v.expand(5, 3).clone())
    frames = tuple(value.float().double() for value in frames)
    return Case("parallel3d", frames, shape, detector, pitch, spacing)


def _oblique_chords(scan):
    half = [size * pitch / 2 for size, pitch in zip(scan.shape[::-1], scan.spacing[::-1])]
    direction, centers, u, v = (value.tolist() for value in scan.trajectory)
    expected = torch.empty(scan.projection_shape, dtype=torch.float64)
    for view in range(len(direction)):
        for iu in range(scan.detector[0]):
            for iv in range(scan.detector[1]):
                a = (iu - (scan.detector[0] - 1) / 2) * scan.pitch[0]
                b = (iv - (scan.detector[1] - 1) / 2) * scan.pitch[1]
                origin = [c + a * du + b * dv for c, du, dv in zip(centers[view], u[view], v[view])]
                expected[view, iu, iv] = _box_chord(origin, direction[view], half)
    return expected


@pytest.mark.parametrize("backend", CHORD_BACKENDS)
def test_oblique_chords_of_a_constant_box(request, backend):
    scan = _oblique_chord_scan()
    chords = _oblique_chords(scan)
    direction, _, u, v = scan.trajectory
    # The direction is oblique to every volume axis and to the detector normal.
    assert direction.abs().min() > .1
    assert torch.linalg.cross(direction, torch.linalg.cross(u, v, dim=-1), dim=-1).norm(dim=-1).min() > .2
    assert chords[:4].min() > 1. and torch.all(chords[4] == 0)
    # The rays leave the box through different faces, so the chords differ.
    assert len({round(value, 6) for value in chords[0].flatten().tolist()}) == math.prod(scan.detector)
    assert (chords[3] - chords[0]).abs().max() > 1e-2
    # The same infinite lines, wherever the detector lies along them.
    torch.testing.assert_close(chords[0], chords[1], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(chords[0], chords[2], rtol=1e-6, atol=1e-6)
    with _chord_operator(request, backend, scan) as (native, device, dtype), torch.no_grad():
        actual = native.project(torch.full(scan.shape, DENSITY, dtype=dtype).to(device))
    assert tuple(actual.shape) == scan.projection_shape
    close(actual, DENSITY * chords)


# Item 8: tolerance guards for the beams and the backend that must not change.
# Bit identity with the pre-feature tree is a final-gate comparison of outputs
# from both trees on one host; no snapshot is committed for it. Scalar native
# parallel, fan and cone also stay pinned bitwise to an earlier tree by
# tests/test_native_per_axis_spacing.py::test_unchanged_scalar_native_is_bitwise_main.

@NATIVE_REQUIRED
@pytest.mark.parametrize("beam", ["parallel", "fan", "cone"])
@pytest.mark.parametrize("tiled", [False, True], ids=["full", "chunked"])
def test_unchanged_native_parallel_fan_cone_match_portable(native_device, beam, tiled):
    # Regression guard: this already holds on the pre-feature tree.
    scan = case(beam)
    image, rays = data(scan.shape, phase=1.50), data(scan.projection_shape, phase=1.86)
    reference = _portable(scan)
    expected = reference.project(image.double()), reference.backproject(rays.double())
    with native_only() as bridge:
        native = operator(scan, **(chunks(scan) if tiled else {}))
        assert native.backend == "numba"
        actual = native.project(image.to(native_device)), native.backproject(rays.to(native_device))
        assert bridge.call_count > 0
    for result, wanted in zip(actual, expected):
        assert wanted.abs().max() > 0
        close(result, wanted)


@pytest.mark.parametrize("precision", ["fast", "float32", "float64"])
def test_unchanged_portable_parallel3d_matches_cell_matrix_without_native_calls(precision, monkeypatch):
    # Regression guard: this already holds on the pre-feature tree.
    def forbidden(*args, **kwargs):
        raise AssertionError("portable parallel3d called a native CUDA bridge")

    monkeypatch.setattr(TorchCUDABridge, "tensor_to_cuda_array", forbidden)
    scan = _scan()
    matrix = _cell_matrix(scan)
    portable = operator(scan, backend="torch", precision=precision)
    dtype = torch.float64 if precision == "float64" else torch.float32
    assert portable.backend == "torch" and portable.dtype == dtype
    image = data(scan.shape, phase=1.57, dtype=dtype)
    rays = data(scan.projection_shape, phase=1.93, dtype=dtype)
    forward, adjoint = portable.project(image), portable.backproject(rays)
    tolerance = ((PORTABLE_RTOL, PORTABLE_ATOL) if precision == "float64" else (FLAT_RTOL, FLAT_ATOL))
    close(forward.flatten(), matrix @ image.double().flatten(), rtol=tolerance[0], atol=tolerance[1])
    close(adjoint.flatten(), matrix.T @ rays.double().flatten(), rtol=tolerance[0], atol=tolerance[1])


# Clarified item 3 (accuracy): one long ray through a volume whose values
# alternate along x. The line integral is a sum of 8192 terms of about +-1 that
# cancel, so float32 error that grows along the ray is not hidden by the result.

LONG_SHAPE = (1, 512, 8192)  # (D, H, W); native layout (W, H, D) = (8192, 512, 1)
LONG_SLOPE = .03


def _long_ray_scan(beam):
    """One detector pixel whose ray is the line through the origin along (1, .03, 0)."""
    direction = torch.tensor([1., LONG_SLOPE, 0.], dtype=torch.float64)
    direction = direction / direction.norm()
    u = torch.tensor([-LONG_SLOPE, 1., 0.], dtype=torch.float64)
    u = u / u.norm()
    v = torch.tensor([0., 0., 1.], dtype=torch.float64)
    if beam == "parallel3d":
        frames = (direction, torch.zeros(3, dtype=torch.float64), u, v)
    else:
        # The central cone ray is the same line: source and detector on it, outside the volume.
        frames = (-float(LONG_SHAPE[2]) * direction, float(LONG_SHAPE[2]) * direction, u, v)
    # Both backends receive exactly the same represented trajectory coordinates.
    frames = tuple(value.float().double()[None] for value in frames)
    return Case(beam, frames, LONG_SHAPE, (1, 1), (1., 1.), 1.)


def _long_ray_values(beam, native_device):
    """Native float32 and portable float64 projections of the alternating and the constant volume."""
    scan = _long_ray_scan(beam)
    depth, height, width = scan.shape
    alternating = torch.tensor([-1., 1.]).repeat(width // 2).expand(depth, height, width).contiguous()
    constant = torch.ones(scan.shape)
    reference = _portable(scan)
    expected = [reference.project(image.double()) for image in (alternating, constant)]
    # Fixture: the ray stays inside the volume from the first to the last x column,
    # so it crosses every column with the same length and the alternating sum cancels.
    path = width * math.sqrt(1 + LONG_SLOPE ** 2)
    assert width * LONG_SLOPE / 2 < height / 2 - 1 and width % 2 == 0
    close(expected[1].flatten(), torch.tensor([path], dtype=torch.float64), rtol=1e-6, atol=0.)
    assert expected[0].abs().max() < 1e-9, "the portable float64 alternating sum must cancel"
    with native_only() as bridge:
        native = operator(scan)
        assert native.backend == "numba"
        actual = [native.project(image.to(native_device)) for image in (alternating, constant)]
        assert bridge.call_count > 0
    assert all(value.dtype == torch.float32 and tuple(value.shape) == (1, 1, 1) for value in actual)
    # The native ray crosses the whole volume: a ray that misses it, or a
    # kernel that writes nothing, would also give a zero alternating sum.
    close(actual[1], expected[1])
    return actual[0].double().cpu().flatten(), expected[0].flatten()


@NATIVE_REQUIRED
def test_long_ray_error_is_no_worse_than_native_cone_on_the_same_line(native_device):
    cone, cone_expected = _long_ray_values("cone", native_device)
    actual, expected = _long_ray_values("parallel3d", native_device)
    error, cone_error = (actual - expected).abs().item(), (cone - cone_expected).abs().item()
    assert error <= cone_error + FLAT_ATOL, (
        f"native parallel3d error {error} on a ray of length {LONG_SHAPE[2]} voxels is worse than "
        f"the native cone error {cone_error} on the same line")
