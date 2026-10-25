"""Trajectory constructors reject degenerate user inputs instead of building NaN geometry."""

import pytest
import torch

from diffct.geometry import (
    circular_trajectory_2d_parallel,
    custom_trajectory_2d_parallel,
    random_trajectory_3d,
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
