"""Analytical backprojection safety, short-scan amplitude and translation."""

import math

import numpy as np
import pytest
import torch
from numba import cuda

from diffct import (
    angular_integration_weights, circular_trajectory_2d_fan, circular_trajectory_3d,
    cone_weighted_backproject, detector_coordinates_1d, fan_cosine_weights,
    fan_weighted_backproject, parallel_weighted_backproject, parker_weights,
    ramp_filter_1d,
)
from diffct.analytical import _analytical_geometry
from diffct.kernels import (
    _cone_3d_fdk_backproject_kernel, _fan_2d_fbp_backproject_kernel,
    _parallel_2d_fbp_backproject_kernel,
)


@pytest.mark.parametrize("n_det", [0, 1])
@pytest.mark.parametrize("beam", ["parallel", "fan"])
def test_2d_analytical_rejects_too_few_bins_before_cuda(beam, n_det):
    source = torch.tensor([[0.0, 10.0]])
    detector = -source
    u = torch.tensor([[1.0, 0.0]])
    backproject = (parallel_weighted_backproject if beam == "parallel"
                   else fan_weighted_backproject)
    with pytest.raises(ValueError, match="at least two detector bins"):
        backproject(torch.ones(1, n_det), source, detector, u, 1.0, 2, 2)


@pytest.mark.parametrize("det_shape", [(0, 4), (1, 4), (4, 0), (4, 1), (1, 1)])
def test_cone_analytical_rejects_too_few_bins_before_cuda(det_shape):
    source = torch.tensor([[0.0, 10.0, 0.0]])
    detector = -source
    u, v = torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="at least two detector bins"):
        cone_weighted_backproject(torch.ones(1, *det_shape), source, detector, u, v,
                                  2, 2, 2, 1.0, 1.0)


@pytest.mark.parametrize("beam", ["parallel", "fan", "cone_u", "cone_v", "cone_uv"])
def test_single_bin_kernel_body_never_reads_a_neighbor(monkeypatch, beam):
    # Execute the original kernel body with NumPy's bounds checking. Geometry
    # places the output voxel exactly on bin zero, so the old body reads bin 1.
    is_cone = beam.startswith("cone")
    monkeypatch.setattr(cuda, "grid", lambda dim: (0,) * dim)
    output = np.ones((1, 1, 1) if is_cone else (1, 1), dtype=np.float32)
    if not is_cone:
        sino = np.ones((1, 1), dtype=np.float32)
        source = np.array([[0.0, 4.0]], dtype=np.float32)
        detector = np.array([[0.0, -4.0]], dtype=np.float32)
        u = np.array([[1.0, 0.0]], dtype=np.float32)
        if beam == "parallel":
            detector[:] = 0
            kernel = _parallel_2d_fbp_backproject_kernel
            args = (sino, 1, 1, output, 1, 1, 1.0, source, detector, u, 0.5, 0.5, 1.0)
        else:
            kernel = _fan_2d_fbp_backproject_kernel
            args = (sino, 1, 1, output, 1, 1, 1.0, source, detector, u,
                    0.5, 0.5, 1.0, 0.0, 0.0)
    else:
        n_u, n_v = (2, 1) if beam == "cone_v" else ((1, 2) if beam == "cone_u" else (1, 1))
        sino = np.ones((1, n_u, n_v), dtype=np.float32)
        source = np.array([[0.0, 4.0, 0.0]], dtype=np.float32)
        detector = -source
        u, v = np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 1.0]])
        kernel = _cone_3d_fdk_backproject_kernel
        args = (sino, 1, n_u, n_v, output, 1, 1, 1, 1.0, 1.0, source, detector, u, v,
                0.5 + (n_u - 1) / 4, 0.5, 0.5 + (n_v - 1) / 4,
                1.0, 0.0, 0.0, 0.0)
    kernel.py_func(*args)
    assert (output == 0).all()


@pytest.mark.cuda
@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("coverage_degrees", [240, 360])
@pytest.mark.parametrize("voxel_spacing", [0.5, 1.0])
@pytest.mark.parametrize("explicit_isocenter", [False, True])
def test_analytical_backprojection_is_translation_invariant(
    beam, coverage_degrees, voxel_spacing, explicit_isocenter,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    factory = circular_trajectory_2d_fan if beam == "fan" else circular_trajectory_3d
    trajectory = factory(128, 16.0, 32.0, end_angle=math.radians(coverage_degrees),
                         device="cuda")
    shift = voxel_spacing * torch.tensor(
        [8.0, 0.0] if beam == "fan" else [8.0, 0.0, 3.0], device="cuda",
    )
    moved = (trajectory[0] + shift, trajectory[1] + shift, *trajectory[2:])
    base_kwargs = {"isocenter": torch.zeros_like(shift)} if explicit_isocenter else {}
    moved_kwargs = {"isocenter": shift} if explicit_isocenter else {}
    if beam == "fan":
        sinogram = torch.ones(128, 64, device="cuda") / 128
        base = fan_weighted_backproject(sinogram, *trajectory, 1.0, 32, 32,
                                        voxel_spacing, **base_kwargs)
        translated = fan_weighted_backproject(sinogram, *moved, 1.0, 32, 32,
                                              voxel_spacing, **moved_kwargs)
        expected, actual = base[13:20, 13:20], translated[13:20, 21:28]
    else:
        sinogram = torch.ones(128, 64, 32, device="cuda") / 128
        base = cone_weighted_backproject(sinogram, *trajectory, 16, 32, 32, 1.0, 1.0,
                                         voxel_spacing, **base_kwargs)
        translated = cone_weighted_backproject(sinogram, *moved, 16, 32, 32, 1.0, 1.0,
                                               voxel_spacing, **moved_kwargs)
        expected, actual = base[6:11, 13:20, 13:20], translated[9:14, 13:20, 21:28]
    assert expected.min() > 0
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("angle", [0.0, 0.3, 1.3])
def test_isocenter_inference_reports_ambiguous_views(angle):
    direction = torch.tensor([math.sin(angle), math.cos(angle)])
    source = torch.stack((10 * direction, -10 * direction))
    detector = -source
    normals = torch.stack((-direction, direction))
    with pytest.raises(ValueError, match="supply isocenter explicitly"):
        _analytical_geometry(source, detector, normals, None)
    isocenter, sid, sdd = _analytical_geometry(source, detector, normals, (0.0, 0.0))
    torch.testing.assert_close(isocenter, torch.zeros(2))
    assert sid == pytest.approx(10.0) and sdd == pytest.approx(20.0)


@pytest.mark.cuda
@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_explicit_isocenter_allows_one_view(beam):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    factory = circular_trajectory_2d_fan if beam == "fan" else circular_trajectory_3d
    trajectory = factory(1, 16.0, 32.0, device="cuda")
    if beam == "fan":
        reconstruction = fan_weighted_backproject(
            torch.ones(1, 16, device="cuda"), *trajectory, 1.0, 3, 3,
            isocenter=(0.0, 0.0),
        )
        center = reconstruction[1, 1]
    else:
        reconstruction = cone_weighted_backproject(
            torch.ones(1, 16, 8, device="cuda"), *trajectory, 3, 3, 3, 1.0, 1.0,
            isocenter=(0.0, 0.0, 0.0),
        )
        center = reconstruction[1, 1, 1]
    assert center.item() == pytest.approx(1 / math.pi, rel=1e-6)


@pytest.mark.cuda
@pytest.mark.parametrize("coverage_degrees", [330, 350, 360])
def test_parker_fan_disk_center_amplitude(coverage_degrees):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    n_views, n_det, pitch, sid, sdd, radius = 720, 256, 0.1, 20.0, 40.0, 4.0
    coverage = math.radians(coverage_degrees)
    angles = torch.arange(n_views, device="cuda") * (coverage / n_views)
    trajectory = circular_trajectory_2d_fan(n_views, sid, sdd, end_angle=coverage,
                                           device="cuda")
    u = detector_coordinates_1d(n_det, pitch, device="cuda")
    # Exact line integrals of a unit disk, independent of Siddon/rasterization.
    distance = sid * torch.sin(torch.atan(u / sdd))
    projection = 2 * torch.sqrt((radius**2 - distance**2).clamp(min=0))
    weights = parker_weights(angles, n_det, pitch, sdd)
    weighted = projection[None, :] * weights * fan_cosine_weights(
        n_det, pitch, sdd, device="cuda",
    )
    filtered = ramp_filter_1d(weighted, dim=1, sample_spacing=pitch, pad_factor=2)
    filtered *= angular_integration_weights(
        angles, redundant_full_scan=coverage_degrees == 360,
    )[:, None]
    reconstruction = fan_weighted_backproject(filtered, *trajectory, pitch, 7, 7)
    assert abs(reconstruction[3, 3].item() - 1.0) < 0.02
