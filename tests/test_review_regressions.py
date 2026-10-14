"""Per-view analytical scaling, strided storage safety and source packaging."""

import math
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
from unittest import mock

import numpy as np
import pytest
import torch

from diffct import (
    NpyStore, TensorStore, angular_integration_weights, cone_weighted_backproject,
    fan_weighted_backproject, parker_weights, ramp_filter_1d,
)
from diffct.streaming import _validate_stores


cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
ROOT = Path(__file__).resolve().parents[1]


def _orbit(beam, distances="both"):
    angles = torch.tensor([0.0, 1.1, 2.7], device="cuda")
    u = torch.stack((angles.cos(), angles.sin()), dim=1)
    normal = torch.stack((-angles.sin(), angles.cos()), dim=1)
    sid = torch.tensor([4.0, 8.0, 6.0], device="cuda")
    sdd = torch.tensor([8.0, 12.0, 18.0], device="cuda")
    if distances == "sdd":
        sid.fill_(4.0)
    elif distances == "sid":
        sdd.fill_(18.0)
    if beam == "cone":
        u = torch.cat((u, torch.zeros(3, 1, device="cuda")), dim=1)
        normal = torch.cat((normal, torch.zeros(3, 1, device="cuda")), dim=1)
        v = torch.tensor([[0.0, 0.0, 1.0]], device="cuda").expand(3, -1).contiguous()
    trajectory = (-normal * sid[:, None], normal * (sdd - sid)[:, None], u)
    return (*trajectory, v) if beam == "cone" else trajectory, sid, sdd


def _weighted(beam, sinogram, trajectory):
    if beam == "fan":
        return fan_weighted_backproject(sinogram, *trajectory, 1.0, 3, 3,
                                        isocenter=(0.0, 0.0))
    return cone_weighted_backproject(sinogram, *trajectory, 3, 3, 3, 1.0, 1.0,
                                     isocenter=(0.0, 0.0, 0.0))


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("distances", ["sdd", "sid", "both"])
def test_analytical_normalization_is_per_view_and_additive(beam, distances):
    trajectory, sid, sdd = _orbit(beam, distances)
    shape = (3, 9, 7) if beam == "cone" else (3, 9)
    amplitudes = torch.tensor([0.25, 1.5, -0.7], device="cuda")
    sinogram = amplitudes.reshape((3,) + (1,) * (len(shape) - 1)).expand(shape)
    actual = _weighted(beam, sinogram, trajectory)
    # At the physical isocenter the distance weight is exactly one. This
    # oracle contains neither detector interpolation nor production kernels.
    expected_center = (amplitudes * sdd / sid).sum() / (2 * math.pi)
    center = (1,) * actual.ndim
    torch.testing.assert_close(actual[center], expected_center, rtol=2e-6, atol=2e-7)
    individual = sum(_weighted(beam, sinogram[i:i + 1], tuple(t[i:i + 1] for t in trajectory))
                     for i in range(3))
    torch.testing.assert_close(actual, individual, rtol=2e-5, atol=2e-6)


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("inactive", ["zero-data", "zero-weight"])
def test_inactive_views_cannot_rescale_an_active_analytical_view(beam, inactive):
    trajectory, _, _ = _orbit(beam)
    shape = (3, 9, 7) if beam == "cone" else (3, 9)
    sinogram = torch.ones(shape, device="cuda")
    if inactive == "zero-data":
        sinogram[1:] = 0
        weighted = sinogram
    else:
        sinogram.requires_grad_()
        view_weights = torch.tensor([1.0, 0.0, 0.0], device="cuda")
        weighted = sinogram * view_weights.reshape((3,) + (1,) * (len(shape) - 1))
    actual = _weighted(beam, weighted, trajectory)
    expected = _weighted(beam, weighted[:1], tuple(t[:1] for t in trajectory))
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    if inactive == "zero-weight":
        gradient, = torch.autograd.grad(actual.sum(), sinogram)
        assert torch.count_nonzero(gradient[1:]) == 0
        assert gradient[0].abs().sum() > 0


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_varying_distances_gradient_matches_view_sum_and_finite_difference(beam):
    trajectory, _, _ = _orbit(beam)
    shape = (3, 9, 7) if beam == "cone" else (3, 9)
    generator = torch.Generator(device="cuda").manual_seed(142)
    sinogram = torch.randn(shape, generator=generator, device="cuda").requires_grad_()
    direction = torch.randn(shape, generator=generator, device="cuda")
    output = _weighted(beam, sinogram, trajectory)
    cotangent = torch.randn(output.shape, generator=generator, device="cuda")
    actual, = torch.autograd.grad(output, sinogram, cotangent)
    individual = sum(_weighted(beam, sinogram[i:i + 1], tuple(t[i:i + 1] for t in trajectory))
                     for i in range(3))
    expected, = torch.autograd.grad(individual, sinogram, cotangent)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    epsilon = 0.003
    with torch.no_grad():
        plus = _weighted(beam, sinogram + epsilon * direction, trajectory)
        minus = _weighted(beam, sinogram - epsilon * direction, trajectory)
    finite_difference = ((plus - minus) * cotangent).double().sum() / (2 * epsilon)
    adjoint_product = (actual * direction).double().sum()
    torch.testing.assert_close(adjoint_product, finite_difference, rtol=2e-3, atol=2e-4)


@pytest.mark.cuda
@cuda_required
def test_analytic_disk_has_unit_center_with_varying_sid_and_sdd():
    views, detectors, pitch, radius = 256, 512, 0.125, 4.0
    angles = torch.arange(views, device="cuda") * (2 * math.pi / views)
    alternate = torch.arange(views, device="cuda") % 2 == 0
    sid = torch.where(alternate, 20.0, 30.0)
    sdd = torch.where(alternate, 40.0, 90.0)
    normal = torch.stack((-angles.sin(), angles.cos()), dim=1)
    u = torch.stack((angles.cos(), angles.sin()), dim=1)
    trajectory = (-normal * sid[:, None], normal * (sdd - sid)[:, None], u)
    coordinates = (torch.arange(detectors, device="cuda") - (detectors - 1) / 2) * pitch
    # Exact line integrals of a unit disk; no rasterized forward projector.
    distance = sid[:, None] * torch.sin(torch.atan(coordinates / sdd[:, None]))
    projection = 2 * torch.sqrt((radius**2 - distance.square()).clamp(min=0))
    cosine = sdd[:, None] / torch.sqrt(sdd[:, None].square() + coordinates.square())
    filtered = ramp_filter_1d(projection * cosine, dim=1, sample_spacing=pitch, pad_factor=2)
    filtered = filtered * angular_integration_weights(angles)[:, None]
    result = fan_weighted_backproject(filtered, *trajectory, pitch, 3, 3,
                                      isocenter=(0.0, 0.0))
    assert result[1, 1].item() == pytest.approx(1.0, abs=0.02)


def _angles(coverage, order):
    angles = torch.arange(720, dtype=torch.float64) * (math.radians(coverage) / 720)
    if order == "decreasing":
        angles = angles.flip(0)
    elif order == "shuffled":
        angles = angles[torch.randperm(720, generator=torch.Generator().manual_seed(143))]
    return angles


@pytest.mark.parametrize("coverage", [240, 360])
@pytest.mark.parametrize("order", ["increasing", "decreasing", "shuffled"])
@pytest.mark.parametrize("weight", ["angular", "parker"])
def test_fp64_angle_offset_preserves_weights(coverage, order, weight):
    angles = _angles(coverage, order)
    shifted = angles + 1e6
    function = angular_integration_weights if weight == "angular" else (
        lambda values: parker_weights(values, 17, 1.0, 80.0)
    )
    expected, actual = function(angles), function(shifted)
    if weight == "angular":
        assert torch.all(actual > 0)
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=2e-7)
    if coverage == 360 and weight == "angular":
        assert actual.sum().item() == pytest.approx(math.pi, abs=1e-6)


@pytest.mark.parametrize("weight", ["angular", "parker"])
def test_short_scan_fp64_offset_preserves_supported_angle_autograd(weight):
    angles = _angles(240, "shuffled").requires_grad_()
    shifted = (angles.detach() + 1e6).requires_grad_()
    function = angular_integration_weights if weight == "angular" else (
        lambda values: parker_weights(values, 17, 1.0, 80.0)
    )
    expected, actual = function(angles), function(shifted)
    probe = torch.linspace(-0.7, 1.3, expected.numel()).reshape(expected.shape)
    expected_gradient, = torch.autograd.grad((expected * probe).sum(), angles)
    actual_gradient, = torch.autograd.grad((actual * probe).sum(), shifted)
    assert torch.isfinite(actual_gradient).all() and expected_gradient.abs().max() > 0
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-5, atol=1e-6)


class _TinyMatrix:
    """Real CPU linear map; CGLS state still uses the example's mapped stores."""
    volume_shape = projection_shape = global_volume_shape = local_volume_shape = (2, 2)
    volume_slice = (slice(0, 2), slice(0, 2))
    world_size, rank, process_group, partition = 1, 0, None, "views"
    beam, voxel_spacing, detector_spacing, detector_shape = "parallel", 1.0, (1.0,), (2,)

    def __init__(self):
        self.matrix = torch.tensor([[2., 0., 0., 0.], [1., 3., 0., 0.],
                                    [0., 1., 4., 0.], [0., 0., 1., 5.]])
        self.calls = []

    def _apply(self, source, output, transpose):
        self.calls.append("backproject" if transpose else "project")
        value = source.read((slice(0, 2), slice(0, 2)))
        matrix = self.matrix.T if transpose else self.matrix
        result = (matrix @ value.reshape(-1)).reshape(2, 2)
        output.write((slice(0, 2), slice(0, 2)), result)
        return output

    def project_into(self, source, output):
        return self._apply(source, output, False)

    def backproject_into(self, source, output):
        return self._apply(source, output, True)


def _overlapping_output(layout):
    if layout == "expand":
        return torch.full((1, 2), 7.0).expand(2, 2)
    return torch.full((3,), 7.0).as_strided((2, 2), (1, 1))


def _input_store(backend, tmp_path):
    if backend == "tensor":
        return TensorStore(torch.arange(1.0, 5.0).reshape(2, 2))
    path = tmp_path / "input.npy"
    np.save(path, np.arange(1, 5, dtype=np.float32).reshape(2, 2))
    store = NpyStore(path, mode="r")
    assert not store.writable and store.dtype == torch.float32
    return store


@pytest.mark.parametrize("layout", ["expand", "as-strided"])
@pytest.mark.parametrize("is_project", [False, True])
@pytest.mark.parametrize("backend", ["tensor", "npy"])
def test_validate_stores_rejects_internally_overlapping_output(layout, is_project, backend, tmp_path):
    source = _input_store(backend, tmp_path)
    output = TensorStore(_overlapping_output(layout))
    with mock.patch.object(TensorStore, "read", autospec=True, side_effect=TensorStore.read) as tensor_reads, \
            mock.patch.object(TensorStore, "write", autospec=True, side_effect=TensorStore.write) as tensor_writes, \
            mock.patch.object(NpyStore, "read", autospec=True, side_effect=NpyStore.read) as npy_reads, \
            mock.patch.object(NpyStore, "write", autospec=True, side_effect=NpyStore.write) as npy_writes:
        with pytest.raises(ValueError, match="overlap"):
            _validate_stores(_TinyMatrix(), source, output, is_project)
        for operation in (tensor_reads, tensor_writes, npy_reads, npy_writes):
            operation.assert_not_called()


@pytest.mark.parametrize("layout", ["expand", "as-strided"])
@pytest.mark.parametrize("backend", ["tensor", "npy"])
def test_cgls_rejects_internal_output_overlap_before_io_or_workspace(layout, backend, tmp_path):
    from examples import _block_cgls

    operator = _TinyMatrix()
    source = _input_store(backend, tmp_path)
    output = TensorStore(_overlapping_output(layout))
    original_source = torch.arange(1.0, 5.0).reshape(2, 2)
    original_output = output.tensor.clone()
    workspace = tmp_path / "solver-work"
    with mock.patch.object(_block_cgls, "_solver_capacity", wraps=_block_cgls._solver_capacity) as capacity, \
            mock.patch.object(NpyStore, "create", wraps=NpyStore.create) as create, \
            mock.patch.object(TensorStore, "read", autospec=True, side_effect=TensorStore.read) as tensor_reads, \
            mock.patch.object(TensorStore, "write", autospec=True, side_effect=TensorStore.write) as tensor_writes, \
            mock.patch.object(NpyStore, "read", autospec=True, side_effect=NpyStore.read) as npy_reads, \
            mock.patch.object(NpyStore, "write", autospec=True, side_effect=NpyStore.write) as npy_writes:
        with pytest.raises(ValueError, match="overlap"):
            _block_cgls.cgls_into(operator, source, output, 4, workspace=workspace, block_elements=2)
        capacity.assert_not_called()
        create.assert_not_called()
        for operation in (tensor_reads, tensor_writes, npy_reads, npy_writes):
            operation.assert_not_called()
    assert not operator.calls and not workspace.exists()
    torch.testing.assert_close(source.read((slice(0, 2), slice(0, 2))), original_source)
    torch.testing.assert_close(output.tensor, original_output)


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "slice", "as-strided"])
def test_cgls_accepts_nonoverlapping_tensor_output(layout, tmp_path):
    from examples import _block_cgls

    operator = _TinyMatrix()
    source = torch.arange(1.0, 5.0).reshape(2, 2)
    if layout == "transpose":
        output = torch.full((2, 2), 7.0).T
    elif layout == "slice":
        output = torch.full((2, 4), 7.0)[:, ::2]
    elif layout == "as-strided":
        output = torch.full((5,), 7.0).as_strided((2, 2), (3, 1))
    else:
        output = torch.full((2, 2), 7.0)
    source_store, output_store = TensorStore(source), TensorStore(output)
    expected = torch.linalg.solve(operator.matrix.double(), source.double().reshape(-1)).reshape(2, 2)
    returned = _block_cgls.cgls_into(operator, source_store, output_store, 4,
                                     workspace=tmp_path / "solver-work", block_elements=2)
    assert returned is output_store and operator.calls
    torch.testing.assert_close(output.double(), expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(source, torch.arange(1.0, 5.0).reshape(2, 2))


def test_cgls_accepts_disjoint_interleaved_tensor_stores(tmp_path):
    from examples import _block_cgls

    operator = _TinyMatrix()
    backing = torch.full((8,), 7.0)
    source, output = backing[::2].reshape(2, 2), backing[1::2].reshape(2, 2)
    source.copy_(torch.arange(1.0, 5.0).reshape(2, 2))
    original = source.clone()
    stores = TensorStore(source), TensorStore(output)
    assert _validate_stores(operator, *stores, False) == stores
    _block_cgls.cgls_into(operator, *stores, 4, workspace=tmp_path / "solver-work", block_elements=2)
    expected = torch.linalg.solve(operator.matrix.double(), original.double().reshape(-1)).reshape(2, 2)
    torch.testing.assert_close(output.double(), expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(source, original)


@pytest.mark.parametrize("is_project", [False, True])
def test_tensor_store_true_strided_overlap_is_rejected(is_project):
    backing = torch.arange(10, dtype=torch.float32)
    original = backing.clone()
    source = TensorStore(backing[:8:2].reshape(2, 2))
    output = TensorStore(backing[2:10:2].reshape(2, 2))
    with pytest.raises(ValueError, match="overlap"):
        _validate_stores(_TinyMatrix(), source, output, is_project)
    torch.testing.assert_close(backing, original)


def test_sdist_contains_example_helpers_required_by_packaged_tests(tmp_path, monkeypatch):
    build = pytest.importorskip("hatchling.build", reason="sdist check requires Hatch build tooling")
    monkeypatch.chdir(ROOT)
    archive = tmp_path / build.build_sdist(str(tmp_path))
    required = {"examples/_common.py", "examples/_block_cgls.py", "examples/data/preprocess_walnut.py",
                "examples/walnut_reconstruction.py"}
    with tarfile.open(archive) as contents:
        paths = {PurePosixPath(*PurePosixPath(item.name).parts[1:]).as_posix()
                 for item in contents.getmembers()}
        assert required <= paths, f"sdist missing packaged-test helpers: {sorted(required - paths)}"
        contents.extractall(tmp_path / "unpacked", filter="data")
    source_root, = (tmp_path / "unpacked").iterdir()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_example_psnr.py", "tests/test_release_2_0_1.py",
         "tests/test_block_cgls.py", "--collect-only", "-q"], cwd=source_root,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run(
        [sys.executable, "-I", "-m", "pytest",
         "tests/test_release_2_0_1.py::test_load_scan_detector_offsets",
         "tests/test_release_2_0_1.py::test_circular_geometry_moves_detector_along_axes", "-q"],
        cwd=source_root, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
