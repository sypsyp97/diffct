"""CPU construction checks for source separation and detector facing."""

import math

import pytest
import torch

from diffct import (
    Projector,
    circular_trajectory_2d_fan,
    circular_trajectory_2d_parallel,
    circular_trajectory_3d,
    spiral_trajectory_3d,
)


def _trajectory(beam, n_views=3):
    dimensions = 3 if beam == "cone" else 2
    source = torch.zeros(n_views, dimensions, dtype=torch.float32)
    source[:, 0] = -20.0
    center = torch.zeros_like(source)
    center[:, 0] = 20.0
    u = torch.zeros_like(source)
    u[:, 1] = 1.0
    if beam == "fan":
        return source, center, u
    v = torch.zeros_like(source)
    v[:, 2] = 1.0
    return source, center, u, v


def _construct(beam, trajectory, voxel_spacing=1.0):
    return Projector(
        trajectory,
        (4, 6, 7) if beam == "cone" else (6, 7),
        (5, 3) if beam == "cone" else (5,),
        beam=beam,
        voxel_spacing=voxel_spacing,
        devices=None,
        distributed=False,
    )


@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_valid_detector_facing_constructs(beam):
    projector = _construct(beam, _trajectory(beam))
    assert projector.projection_shape[0] == 3


def _mixed_dtype_trajectory(beam):
    dtypes = (torch.float64, torch.float32, torch.float32, torch.float64)
    return tuple(
        component.to(dtype=dtype)
        for component, dtype in zip(_trajectory(beam), dtypes)
    )


@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_valid_mixed_dtype_geometry_constructs(beam):
    projector = _construct(beam, _mixed_dtype_trajectory(beam))
    assert projector.projection_shape[0] == 3


@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("degeneracy", ["coincident", "edge-on"])
def test_mixed_dtype_degenerate_geometry_raises(beam, degeneracy):
    trajectory = _mixed_dtype_trajectory(beam)
    source, center, u = trajectory[:3]
    if degeneracy == "coincident":
        center[1] = source[1]
        message = "source and detector center must differ"
    else:
        u[1] = 0.0
        u[1, 0] = 1.0
        message = "edge-on"
    with pytest.raises(ValueError, match=message):
        _construct(beam, trajectory)


@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_float16_large_coordinate_edge_on_detector_raises(beam):
    trajectory = tuple(component.to(torch.float16) for component in _trajectory(beam))
    source, center, u = trajectory[:3]
    source[:, 0] = -40000.0
    center[:, 0] = 40000.0
    u[:] = 0.0
    u[:, 0] = 1.0
    with pytest.raises(ValueError, match="edge-on"):
        _construct(beam, trajectory)


@pytest.mark.parametrize(
    ("beam", "factory"),
    [
        ("fan", circular_trajectory_2d_fan),
        ("cone", circular_trajectory_3d),
        ("cone", spiral_trajectory_3d),
    ],
    ids=["fan-circular", "cone-circular", "cone-helical"],
)
def test_geometry_helpers_construct_on_cpu(beam, factory):
    trajectory = factory(7, sid=30.0, sdd=45.0, device="cpu")
    projector = _construct(beam, trajectory)
    assert projector.projection_shape[0] == 7


@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("bad_view", [None, 1], ids=["all-views", "one-bad-view"])
def test_coincident_source_and_detector_raise(beam, bad_view):
    trajectory = _trajectory(beam)
    source, center = trajectory[:2]
    if bad_view is None:
        center.copy_(source)
    else:
        center[bad_view] = source[bad_view]
    with pytest.raises(ValueError, match="source and detector center must differ"):
        _construct(beam, trajectory)


@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("bad_view", [None, 1], ids=["all-views", "one-bad-view"])
def test_edge_on_detector_raises(beam, bad_view):
    trajectory = _trajectory(beam)
    u = trajectory[2]
    # The principal ray points along x. In cone geometry, x cross z points
    # along y, perpendicular to the principal ray.
    edge_on_u = torch.zeros(u.shape[1], dtype=u.dtype)
    edge_on_u[0] = 1.0
    if bad_view is None:
        u[:] = edge_on_u
    else:
        u[bad_view] = edge_on_u
    with pytest.raises(ValueError, match="edge-on"):
        _construct(beam, trajectory)


@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("facing", [5e-5, -5e-5])
def test_nearly_edge_on_detector_below_threshold_raises(beam, facing):
    trajectory = _trajectory(beam)
    u = trajectory[2]
    u[:, 0] = math.sqrt(1.0 - facing * facing)
    u[:, 1] = facing
    with pytest.raises(ValueError, match="edge-on"):
        _construct(beam, trajectory)


@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("facing", [1e-2, -1e-2])
def test_slight_tilt_above_threshold_constructs(beam, facing):
    trajectory = _trajectory(beam)
    u = trajectory[2]
    u[:, 0] = math.sqrt(1.0 - facing * facing)
    u[:, 1] = facing
    projector = _construct(beam, trajectory)
    assert projector.projection_shape[0] == 3


def test_parallel_geometry_is_unaffected():
    trajectory = circular_trajectory_2d_parallel(3, device="cpu")
    projector = _construct("parallel", trajectory)
    assert projector.projection_shape == (3, 5)


@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("bad_view", [None, 1], ids=["all-views", "one-bad-view"])
def test_both_endpoints_beyond_soft_distance_limit_raise(beam, bad_view):
    trajectory = _trajectory(beam)
    source, center = trajectory[:2]
    views = slice(None) if bad_view is None else bad_view
    source[views, 0] = -1.5e6
    center[views, 0] = 1.5e6
    with pytest.raises(ValueError, match="within 1e6 voxels"):
        _construct(beam, trajectory)


@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_endpoint_distance_uses_euclidean_norm(beam):
    trajectory = _trajectory(beam)
    source, center = trajectory[:2]
    source[1, :2] = torch.tensor([-8e5, 8e5])
    center[1, :2] = torch.tensor([8e5, 8e5])
    with pytest.raises(ValueError, match="within 1e6 voxels"):
        _construct(beam, trajectory)


@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_endpoint_distance_is_measured_from_volume_origin(beam):
    trajectory = _trajectory(beam)
    source, center = trajectory[:2]
    source[1, 0] = 1.5e6
    center[1, 0] = 1.5e6 + 40.0
    with pytest.raises(ValueError, match="within 1e6 voxels"):
        _construct(beam, trajectory)


@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("endpoint", [0, 1], ids=["source", "detector"])
@pytest.mark.parametrize("bad_view", [None, 1], ids=["all-views", "one-bad-view"])
def test_any_endpoint_beyond_hard_distance_limit_raises(beam, endpoint, bad_view):
    trajectory = tuple(component.to(torch.float64) for component in _trajectory(beam))
    views = slice(None) if bad_view is None else bad_view
    trajectory[endpoint][views, 0] = -2e15 if endpoint == 0 else 2e15
    with pytest.raises(ValueError, match="within 1e15 voxels"):
        _construct(beam, trajectory)


@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_endpoint_distance_is_scaled_by_voxel_spacing(beam):
    trajectory = _trajectory(beam)
    source, center = trajectory[:2]
    source[:, 0] = -1.5e6
    center[:, 0] = 1.5e6
    projector = _construct(beam, trajectory, voxel_spacing=2.0)
    assert projector.projection_shape[0] == 3


@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("endpoint", [0, 1], ids=["source", "detector"])
@pytest.mark.parametrize("distance", [1e7, 1e15], ids=["far", "hard-limit"])
def test_one_near_endpoint_allows_far_endpoint(beam, endpoint, distance):
    trajectory = tuple(component.to(torch.float64) for component in _trajectory(beam))
    source, center = trajectory[:2]
    source[:, 0] = -100.0
    center[:, 0] = 100.0
    trajectory[endpoint][:, 0] = -distance if endpoint == 0 else distance
    projector = _construct(beam, trajectory)
    assert projector.projection_shape[0] == 3


@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_both_endpoints_at_soft_distance_limit_construct(beam):
    trajectory = _trajectory(beam)
    source, center = trajectory[:2]
    source[:, 0] = -1e6
    center[:, 0] = 1e6
    projector = _construct(beam, trajectory)
    assert projector.projection_shape[0] == 3


@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_near_endpoint_can_alternate_between_views(beam):
    trajectory = _trajectory(beam)
    source, center = trajectory[:2]
    source[:, 0] = -100.0
    center[:, 0] = 1e7
    source[1, 0] = -1e7
    center[1, 0] = 100.0
    projector = _construct(beam, trajectory)
    assert projector.projection_shape[0] == 3


def test_parallel_detector_beyond_distance_limits_constructs():
    trajectory = circular_trajectory_2d_parallel(3, device="cpu")
    trajectory[1][:, 0] = 2e15
    projector = _construct("parallel", trajectory)
    assert projector.projection_shape == (3, 5)
