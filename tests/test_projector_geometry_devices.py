"""Projector accepts valid trajectory components on mixed devices."""

from __future__ import annotations

import math

import pytest
import torch

from diffct import Projector
from tests.test_projector_api import (
    _cpu_cone_trajectory,
    _cpu_parallel_trajectory,
)


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("beam", "trajectory_factory", "cuda_axis", "volume_shape", "detector_shape"),
    [
        ("parallel", _cpu_parallel_trajectory, 0, (4, 7), 5),
        ("cone", _cpu_cone_trajectory, 2, (3, 4, 5), (4, 3)),
    ],
    ids=("parallel", "cone"),
)
def test_mixed_device_trajectory_matches_cpu_geometry(
    beam, trajectory_factory, cuda_axis, volume_shape, detector_shape
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Projector execution")
    device = torch.device("cuda", torch.cuda.current_device())
    trajectory = trajectory_factory()
    mixed_trajectory = list(trajectory)
    mixed_trajectory[cuda_axis] = mixed_trajectory[cuda_axis].to(device)

    kwargs = {
        "beam": beam,
        "detector_spacing": 1.25,
        "voxel_spacing": 0.75,
    }
    cpu_geometry_projector = Projector(
        trajectory, volume_shape, detector_shape, **kwargs
    )
    mixed_geometry_projector = Projector(
        tuple(mixed_trajectory), volume_shape, detector_shape, **kwargs
    )

    volume = torch.linspace(
        -0.2, 0.3, steps=math.prod(volume_shape), device=device
    ).reshape(volume_shape)
    projection_shape = (4, *(
        (detector_shape,) if isinstance(detector_shape, int) else detector_shape
    ))
    sinogram = torch.linspace(
        -0.15, 0.25, steps=math.prod(projection_shape), device=device
    ).reshape(projection_shape)

    torch.testing.assert_close(
        mixed_geometry_projector.project(volume),
        cpu_geometry_projector.project(volume),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        mixed_geometry_projector.backproject(sinogram),
        cpu_geometry_projector.backproject(sinogram),
        rtol=1e-5,
        atol=1e-5,
    )
