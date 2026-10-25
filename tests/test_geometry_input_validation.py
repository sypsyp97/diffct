"""Trajectory constructors reject degenerate user inputs instead of building NaN geometry."""

import numpy as np
import pytest
import torch

from diffct.geometry import (
    circular_trajectory_2d_parallel,
    circular_trajectory_3d,
    custom_trajectory_2d_fan,
    custom_trajectory_2d_parallel,
    custom_trajectory_3d,
    random_trajectory_3d,
    saddle_trajectory_3d,
    sinusoidal_trajectory_2d_fan,
    sinusoidal_trajectory_2d_parallel,
    sinusoidal_trajectory_3d,
    spiral_trajectory_3d,
)


def _unit_dirs(angles):
    dirs = torch.zeros((len(angles), 2))
    dirs[:, 0] = torch.cos(angles)
    dirs[:, 1] = torch.sin(angles)
    return dirs


def _origins(angles):
    origins = torch.zeros((len(angles), 2))
    origins[:, 1] = -50.0
    return origins


def _source_path_2d(angles, sid):
    return sid * _unit_dirs(angles)


def _source_path_3d(angles, sid):
    return torch.cat((_source_path_2d(angles, sid),
                      torch.zeros_like(angles).unsqueeze(1)), dim=1)


@pytest.fixture(
    params=[
        (circular_trajectory_2d_parallel, {}),
        (circular_trajectory_3d, {"sid": 1000.0, "sdd": 1500.0}),
        (spiral_trajectory_3d, {"sid": 1000.0, "sdd": 1500.0}),
        (sinusoidal_trajectory_3d, {"sid": 1000.0, "sdd": 1500.0}),
        (saddle_trajectory_3d, {"sid": 1000.0, "sdd": 1500.0}),
        (custom_trajectory_2d_parallel,
         {"ray_dir_fn": _unit_dirs, "det_origin_fn": _origins}),
        (custom_trajectory_2d_fan,
         {"sid": 1000.0, "sdd": 1500.0, "source_path_fn": _source_path_2d}),
        (custom_trajectory_3d,
         {"sid": 1000.0, "sdd": 1500.0, "source_path_fn": _source_path_3d}),
        (sinusoidal_trajectory_2d_fan, {"sid": 1000.0, "sdd": 1500.0}),
        (sinusoidal_trajectory_2d_parallel, {}),
    ],
    ids=lambda case: case[0].__name__,
)
def view_count_constructor(request):
    return request.param


def test_custom_parallel_rejects_zero_ray_direction():
    def dirs(angles):
        out = _unit_dirs(angles)
        out[3] = 0.0
        return out

    with pytest.raises(ValueError, match="zero ray direction"):
        custom_trajectory_2d_parallel(8, dirs, _origins, device="cpu")


def test_custom_parallel_rejects_non_finite_ray_direction():
    def dirs(angles):
        out = _unit_dirs(angles)
        out[2, 0] = float("nan")
        return out

    with pytest.raises(ValueError, match="non-finite"):
        custom_trajectory_2d_parallel(8, dirs, _origins, device="cpu")


def test_custom_parallel_accepts_unit_directions():
    ray_dir, det_origin, det_u_vec = custom_trajectory_2d_parallel(
        8, _unit_dirs, _origins, device="cpu")
    assert torch.isfinite(ray_dir).all()
    assert torch.allclose(torch.linalg.vector_norm(ray_dir, dim=1), torch.ones(8))
    assert torch.isfinite(det_u_vec).all()


def test_random_trajectory_sources_stay_valid():
    src_pos, det_center, det_u_vec, det_v_vec = random_trajectory_3d(
        32, sid_mean=1000.0, sdd_mean=1500.0, sid_std=10.0, device="cpu", seed=7)
    assert torch.isfinite(src_pos).all()
    assert (torch.linalg.vector_norm(src_pos, dim=1) > 0).all()
    assert torch.isfinite(det_u_vec).all() and torch.isfinite(det_v_vec).all()


@pytest.mark.parametrize("n_views", [0, -1])
def test_nonpositive_view_count_rejected(n_views):
    with pytest.raises(ValueError, match="n_views"):
        circular_trajectory_2d_parallel(n_views, device="cpu")
    with pytest.raises(ValueError, match="n_views"):
        spiral_trajectory_3d(n_views, 1000.0, 1500.0, device="cpu")
    with pytest.raises(ValueError, match="n_views"):
        custom_trajectory_2d_parallel(n_views, _unit_dirs, _origins, device="cpu")


def test_angle_tensor_view_counts_still_accepted():
    angles = torch.linspace(0.0, torch.pi, 16)
    ray_dir, det_origin, det_u_vec = circular_trajectory_2d_parallel(
        angles, device="cpu")
    assert ray_dir.shape == (16, 2)


@pytest.mark.parametrize("n_views", [
    pytest.param(1, id="python-int-one"),
    pytest.param(4, id="python-int-many"),
    pytest.param(np.int32(1), id="numpy-int32-one"),
    pytest.param(np.int64(4), id="numpy-int64-many"),
    pytest.param(np.uint32(1), id="numpy-uint32-one"),
    pytest.param(np.uint64(4), id="numpy-uint64-many"),
])
def test_integer_view_counts_accepted(view_count_constructor, n_views):
    constructor, kwargs = view_count_constructor
    trajectory = constructor(n_views, device="cpu", **kwargs)
    assert trajectory
    for component in trajectory:
        assert component.shape[0] == int(n_views)
        assert component.device.type == "cpu"


@pytest.mark.parametrize("n_views", [
    pytest.param(2.5, id="fractional-float"),
    pytest.param(3.0, id="integral-float"),
    pytest.param(np.float32(3.0), id="numpy-integral-float"),
    pytest.param(np.float64(2.5), id="numpy-fractional-float"),
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
])
def test_noninteger_view_counts_raise_type_error(view_count_constructor, n_views):
    constructor, kwargs = view_count_constructor
    with pytest.raises(TypeError, match="n_views"):
        constructor(n_views, device="cpu", **kwargs)


@pytest.mark.parametrize("n_views", [
    pytest.param(0, id="python-zero"),
    pytest.param(-1, id="python-negative"),
    pytest.param(np.int32(0), id="numpy-zero"),
    pytest.param(np.int64(-1), id="numpy-negative"),
    pytest.param(np.uint64(0), id="numpy-unsigned-zero"),
])
def test_nonpositive_integer_view_counts_raise_value_error(
        view_count_constructor, n_views):
    constructor, kwargs = view_count_constructor
    with pytest.raises(ValueError, match="n_views"):
        constructor(n_views, device="cpu", **kwargs)


@pytest.mark.parametrize("constructor,kwargs", [
    pytest.param(circular_trajectory_2d_parallel, {}, id="circular-2d-parallel"),
    pytest.param(circular_trajectory_3d, {"sid": 1000.0, "sdd": 1500.0},
                 id="circular-3d"),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("angle_values", [[0.25], [0.1, 0.7, 2.6]],
                         ids=["one-angle", "multiple-angles"])
def test_circular_trajectories_accept_explicit_angle_tensors(
        constructor, kwargs, dtype, angle_values):
    angles = torch.tensor(angle_values, device="cpu", dtype=dtype)
    trajectory = constructor(angles, device="cpu", **kwargs)
    assert trajectory
    for component in trajectory:
        assert component.shape[0] == len(angle_values)
        assert component.device.type == "cpu"
