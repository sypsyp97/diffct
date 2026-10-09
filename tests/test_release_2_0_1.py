"""Release contracts for device selection, geometry, and walnut examples."""

import doctest
import importlib
import io
import math
import sys
import zipfile
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
FUNCTIONS = [
    "ParallelProjectorFunction", "ParallelBackprojectorFunction",
    "FanProjectorFunction", "FanBackprojectorFunction",
    "ConeProjectorFunction", "ConeBackprojectorFunction",
]
BEAMS = ["parallel", "fan", "cone"]
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)
requires_two_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires two CUDA devices"
)


@pytest.fixture
def examples(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    return importlib.import_module("data.preprocess_walnut")


def trajectory(beam, device):
    geometry = importlib.import_module("diffct.geometry")
    if beam == "parallel":
        return geometry.circular_trajectory_2d_parallel(5, start_angle=0.13, device=device)
    if beam == "fan":
        return geometry.circular_trajectory_2d_fan(5, 20.0, 35.0, start_angle=0.13, device=device)
    return geometry.circular_trajectory_3d(5, 20.0, 35.0, start_angle=0.13, device=device)


def function_arguments(name, geometry):
    if name == "ConeProjectorFunction":
        return (*geometry, 9, 7, 0.9, 1.1, 0.8)
    if name == "ConeBackprojectorFunction":
        return (*geometry, 4, 6, 8, 0.9, 1.1, 0.8)
    if "Backprojector" in name:
        return (*geometry, 0.9, 6, 8, 0.8)
    return (*geometry, 9, 0.9, 0.8)


@pytest.mark.cuda
@requires_two_gpus
@pytest.mark.parametrize("name", FUNCTIONS)
def test_function_uses_input_cuda_device_for_output_and_gradients(name):
    projectors = importlib.import_module("diffct.projectors")
    beam = next(beam for beam in BEAMS if name.lower().startswith(beam))
    geometry = trajectory(beam, "cuda:1")
    if "Backprojector" in name:
        shape = (5, 9, 7) if beam == "cone" else (5, 9)
    else:
        shape = (4, 6, 8) if beam == "cone" else (6, 8)
    data = torch.linspace(0.1, 0.9, math.prod(shape), device="cuda:1").reshape(shape)

    def evaluate(current_device):
        with torch.cuda.device(current_device):
            inputs = [value.detach().clone().requires_grad_() for value in (data, *geometry)]
            result = getattr(projectors, name).apply(
                inputs[0], *function_arguments(name, inputs[1:])
            )
            assert result.device == torch.device("cuda:1")
            assert torch.isfinite(result).all()
            weights = torch.linspace(0.2, 1.0, result.numel(), device="cuda:1").reshape(result.shape)
            gradients = torch.autograd.grad(result, inputs, grad_outputs=weights)
            torch.cuda.synchronize(1)
            assert torch.cuda.current_device() == current_device
            for gradient in gradients:
                assert gradient.device == torch.device("cuda:1")
                assert torch.isfinite(gradient).all()
            return result.detach(), tuple(gradient.detach() for gradient in gradients)

    expected, expected_gradients = evaluate(1)
    actual, actual_gradients = evaluate(0)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-5, atol=1e-6)


@pytest.mark.cuda
@requires_two_gpus
@pytest.mark.parametrize("beam", BEAMS)
def test_analytical_backprojection_uses_input_cuda_device(beam):
    analytical = importlib.import_module("diffct.analytical")
    geometry = trajectory(beam, "cuda:1")
    shape = (5, 9, 7) if beam == "cone" else (5, 9)
    sinogram = torch.linspace(0.1, 0.9, math.prod(shape), device="cuda:1").reshape(shape)
    sizes = (4, 6, 8, 0.9, 1.1) if beam == "cone" else (0.9, 6, 8)
    function = getattr(analytical, f"{beam}_weighted_backproject")

    def evaluate(current_device):
        with torch.cuda.device(current_device):
            result = function(sinogram, *geometry, *sizes, voxel_spacing=0.8)
            assert result.device == torch.device("cuda:1")
            torch.cuda.synchronize(1)
            assert torch.isfinite(result).all()
            assert torch.cuda.current_device() == current_device
            return result

    torch.testing.assert_close(evaluate(0), evaluate(1), rtol=1e-5, atol=1e-6)


def weighted_backprojection_arguments(beam, device):
    geometry = trajectory(beam, device)
    shape = (5, 9, 7) if beam == "cone" else (5, 9)
    sinogram = torch.linspace(0.1, 0.9, math.prod(shape), device=device).reshape(shape)
    if beam == "cone":
        names = ("sinogram", "src_pos", "det_center", "det_u_vec", "det_v_vec",
                 "D", "H", "W", "du", "dv", "voxel_spacing")
        args = (sinogram, *geometry, 4, 6, 8, 0.9, 1.1, 0.8)
    else:
        geometry_names = ("ray_dir", "det_origin", "det_u_vec") if beam == "parallel" else (
            "src_pos", "det_center", "det_u_vec"
        )
        names = ("sinogram", *geometry_names, "detector_spacing", "H", "W", "voxel_spacing")
        args = (sinogram, *geometry, 0.9, 6, 8, 0.8)
    keyword_only = {} if beam == "parallel" else {
        "isocenter": torch.zeros(3 if beam == "cone" else 2, device=device)
    }
    return args, dict(zip(names, args), **keyword_only), keyword_only


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("beam", BEAMS)
def test_analytical_backprojection_accepts_all_arguments_by_keyword(beam):
    analytical = importlib.import_module("diffct.analytical")
    function = getattr(analytical, f"{beam}_weighted_backproject")
    args, kwargs, keyword_only = weighted_backprojection_arguments(beam, "cuda:0")
    with torch.cuda.device(0):
        expected = function(*args, **keyword_only)
        actual = function(**kwargs)
        assert actual.device == args[0].device
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.cuda
@requires_two_gpus
@pytest.mark.parametrize("beam", BEAMS)
def test_analytical_keyword_backprojection_uses_input_cuda_device(beam):
    analytical = importlib.import_module("diffct.analytical")
    function = getattr(analytical, f"{beam}_weighted_backproject")
    args, kwargs, keyword_only = weighted_backprojection_arguments(beam, "cuda:1")
    with torch.cuda.device(1):
        expected = function(*args, **keyword_only)
    with torch.cuda.device(0):
        actual = function(**kwargs)
        assert actual.device == torch.device("cuda:1")
        torch.cuda.synchronize(1)
        assert torch.cuda.current_device() == 0
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dimensions", [2, 3])
@pytest.mark.parametrize("bad_value", [0.0, math.nan, math.inf, -math.inf], ids=["origin", "nan", "inf", "negative-inf"])
@pytest.mark.parametrize("bad_view", [0, 2, 4])
def test_custom_trajectory_rejects_invalid_source(dimensions, bad_value, bad_view):
    geometry = importlib.import_module("diffct.geometry")
    function = geometry.custom_trajectory_3d if dimensions == 3 else geometry.custom_trajectory_2d_fan

    def source_path(angles, sid):
        positions = torch.zeros((len(angles), dimensions), device=angles.device, dtype=angles.dtype)
        positions[:, 0] = -sid * torch.sin(angles)
        positions[:, 1] = sid * torch.cos(angles)
        positions[bad_view] = bad_value
        return positions

    with pytest.raises(ValueError):
        function(5, 20.0, 35.0, source_path, device="cpu")


def test_figure8_docstring_example_is_finite_and_constructs_projector():
    import diffct

    function = diffct.custom_trajectory_3d
    snippets = doctest.DocTestParser().get_examples(function.__doc__)
    assert any("def figure8_path" in snippet.source for snippet in snippets)
    namespace = {"torch": torch, "custom_trajectory_3d": partial(function, device="cpu")}
    for snippet in snippets:
        exec(compile(snippet.source, "<custom_trajectory_3d docstring>", "exec"), namespace)
    geometry = tuple(namespace[name] for name in ("src_pos", "det_center", "det_u_vec", "det_v_vec"))
    assert all(value.device.type == "cpu" for value in geometry)
    assert all(torch.isfinite(value).all() for value in geometry)
    assert (torch.linalg.vector_norm(geometry[0], dim=1) > 0).all()
    operator = diffct.Projector(geometry, (8, 8, 8), (9, 7))
    assert operator is not None


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("name", FUNCTIONS)
def test_function_class_docstring_example(name):
    projectors = importlib.import_module("diffct.projectors")
    function = getattr(projectors, name)
    example = doctest.DocTestParser().get_doctest(
        function.__doc__, {}, name, str(ROOT / "diffct" / "projectors.py"), 0
    )
    assert example.examples
    runner = doctest.DocTestRunner()
    result = runner.run(example)
    assert result.failed == 0
    assert result.attempted == len(example.examples)


def kept_region_offset(raw_size, bin_factor, crop):
    """Compute the midpoint of the kept raw-pixel interval from the contract."""
    binned = raw_size // bin_factor
    size = binned if crop == 0 else crop
    start = (binned - size) // 2
    left_edge = start * bin_factor
    right_edge = (start + size) * bin_factor
    return (left_edge + right_edge) / 2 - raw_size / 2


@pytest.mark.parametrize("raw_size,bin_factor,crop", [
    (2240, 8, 256), (2368, 8, 256), (2240, 8, 255),
    (2368, 8, 255), (11, 2, 0), (11, 2, 4), (12, 2, 3),
    (13, 3, 3), (14, 3, 0), (9, 1, 4), (8, 1, 3), (9, 1, 3),
])
def test_detector_center_offset(examples, raw_size, bin_factor, crop):
    expected = kept_region_offset(raw_size, bin_factor, crop)
    assert examples.detector_center_offset(raw_size, bin_factor, crop) == expected


def test_preprocess_without_projection_zips_raises(examples, tmp_path, monkeypatch):
    output = tmp_path / "result.npz"
    monkeypatch.setattr(sys, "argv", ["preprocess_walnut.py", str(output), str(tmp_path), "--view-stride", "721"])
    with pytest.raises(FileNotFoundError):
        examples.main()
    assert not output.exists()


@pytest.mark.parametrize("projection_number", [1, 2, 721])
def test_preprocess_kept_projection_presence(examples, tmp_path, monkeypatch, projection_number):
    tifffile = pytest.importorskip("tifffile")
    image = np.full((2368, 2240), 1000, dtype=np.uint16)
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, image)
    with zipfile.ZipFile(tmp_path / "20201111_walnut_projections_1.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"nested/20201111_walnut_{projection_number:04d}.tif", buffer.getvalue())
    output = tmp_path / "result.npz"
    monkeypatch.setattr(sys, "argv", ["preprocess_walnut.py", str(output), str(tmp_path), "--view-stride", "721"])
    if projection_number == 1:
        examples.main()
        with np.load(output) as data:
            assert data["sinogram"].shape == (1, 256, 256)
            np.testing.assert_array_equal(data["sinogram"], np.zeros((1, 256, 256)))
            np.testing.assert_array_equal(data["view_indices"], [0])
    else:
        with pytest.raises(ValueError, match="missing"):
            examples.main()
        assert not output.exists()


@pytest.mark.parametrize("bin_factor,crop_h,crop_w,du,dv", [
    (8, 255, 255, 0.4, 0.4), (7, 0, 0, 0.35, 0.35),
    (8, 255, 256, 0.4, 0.4), (8, 255, 255, 0.4, 0.6),
])
def test_load_scan_detector_offsets(examples, tmp_path, bin_factor, crop_h, crop_w, du, dv):
    walnut = importlib.import_module("walnut_reconstruction")
    path = tmp_path / "scan.npz"
    stored = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
    angles = np.array([0.0, 0.7], dtype=np.float64)
    np.savez(path, sinogram=stored, angles=angles, sid=20.0, sdd=35.0,
             du=du, dv=dv, detector_bin=bin_factor, crop_h=crop_h, crop_w=crop_w)
    result = walnut.load_scan(path)
    assert len(result) == 8
    sinogram, actual_angles, sid, sdd, actual_du, actual_dv, offset_u, offset_v = result
    np.testing.assert_array_equal(sinogram, stored.astype(np.float32).transpose(0, 2, 1))
    np.testing.assert_array_equal(actual_angles, angles)
    assert (sid, sdd, actual_du, actual_dv) == (20.0, 35.0, du, dv)
    assert offset_u == pytest.approx(kept_region_offset(2240, bin_factor, crop_w) * du / bin_factor)
    assert offset_v == pytest.approx(kept_region_offset(2368, bin_factor, crop_h) * dv / bin_factor)


@pytest.mark.parametrize("offset_u,offset_v", [(0.0, 0.0), (1.25, 0.0), (0.0, -2.5), (-1.25, 2.5)])
def test_circular_geometry_moves_detector_along_axes(examples, offset_u, offset_v):
    walnut = importlib.import_module("walnut_reconstruction")
    angles = np.array([0.0, math.pi / 2, 0.37])
    source, detector, u_axis, v_axis = walnut.circular_geometry(angles, 20.0, 35.0, offset_u, offset_v)
    sin, cos = np.sin(angles), np.cos(angles)
    zero = np.zeros_like(angles)
    expected_u = np.column_stack((cos, sin, zero))
    expected_v = np.column_stack((zero, zero, np.ones_like(angles)))
    nominal_detector = np.column_stack((15 * sin, -15 * cos, zero))
    np.testing.assert_allclose(source.numpy(), np.column_stack((-20 * sin, 20 * cos, zero)), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(u_axis.numpy(), expected_u, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(v_axis.numpy(), expected_v, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(detector.numpy(), nominal_detector + offset_u * expected_u + offset_v * expected_v,
                               rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("values", [[0.0, 0.0], [0.1, 0.7, -0.2]])
def test_psnr_identical_images_is_positive_infinity(examples, values):
    common = importlib.import_module("_common")
    image = torch.tensor(values, dtype=torch.float32)
    assert common.psnr(image.clone(), image) == math.inf


@pytest.mark.parametrize("beam", BEAMS)
def test_projector_process_group_is_none_and_read_only(beam):
    import diffct

    shape = (4, 6, 8) if beam == "cone" else (6, 8)
    detector = (9, 7) if beam == "cone" else 9
    operator = diffct.Projector(trajectory(beam, "cpu"), shape, detector, beam=beam)
    assert operator.process_group is None
    with pytest.raises(AttributeError):
        operator.process_group = object()
    assert operator.process_group is None
