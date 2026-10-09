"""PROPERTY TEST / BOUNDED CPU checks of unchanged geometry.py functions.

Loaded directly with importlib because diffct.__init__ imports CUDA code.
This executes the real module; no CUDA implementation is replaced.
Sources: geometry.py:16,205,290,376,463,547,618,779,848,917.
The finite input ranges and tolerances are numerical evidence, not proofs.
"""

import importlib.util
import math
from pathlib import Path

import pytest
import torch
from hypothesis import given, settings, strategies as st

PATH = Path(__file__).resolve().parents[1] / "diffct" / "geometry.py"
SPEC = importlib.util.spec_from_file_location("formal_cpu_geometry", PATH)
G = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(G)
PROPERTY = settings(max_examples=60, deadline=None, derandomize=True)


def _frame(source, detector, u, v=None, sdd=30):
    ray = detector - source
    torch.testing.assert_close(torch.linalg.vector_norm(ray, dim=1), torch.full_like(source[:, 0], sdd), atol=1e-11, rtol=1e-12)
    torch.testing.assert_close(torch.linalg.vector_norm(u, dim=1), torch.ones_like(source[:, 0]), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close((ray * u).sum(1), torch.zeros_like(source[:, 0]), atol=1e-11, rtol=0)
    if v is not None:
        torch.testing.assert_close(torch.linalg.vector_norm(v, dim=1), torch.ones_like(source[:, 0]), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close((ray * v).sum(1), torch.zeros_like(source[:, 0]), atol=1e-11, rtol=0)
        torch.testing.assert_close((u * v).sum(1), torch.zeros_like(source[:, 0]), atol=1e-12, rtol=0)


@PROPERTY
@given(st.integers(2, 40), st.floats(-3, 3, allow_nan=False, allow_infinity=False),
       st.floats(5, 50, allow_nan=False, allow_infinity=False))
def test_PROPERTY_circular_2d_3d_frames_and_endpoint_exclusion(n, start, sid):
    sdd = sid + 20
    source, detector, u, v = G.circular_trajectory_3d(n, sid, sdd, start_angle=start, end_angle=start + 2 * math.pi, device="cpu", dtype=torch.float64)
    _frame(source, detector, u, v, sdd)
    torch.testing.assert_close(source.norm(dim=1), torch.full((n,), sid, dtype=torch.float64))
    angles = torch.atan2(-source[:, 0], source[:, 1])
    step = 2 * math.pi / n
    phase = torch.atan2(torch.sin(angles - start), torch.cos(angles - start))
    expected = torch.arange(n, dtype=torch.float64) * step
    torch.testing.assert_close(torch.sin(phase), torch.sin(expected), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(torch.cos(phase), torch.cos(expected), atol=1e-12, rtol=1e-12)
    assert not torch.allclose(source[0], source[-1])
    source2, detector2, u2 = G.circular_trajectory_2d_fan(n, sid, sdd, start_angle=start, end_angle=start + 2 * math.pi, device="cpu", dtype=torch.float64)
    torch.testing.assert_close(source[:, :2], source2)
    torch.testing.assert_close(detector[:, :2], detector2)
    torch.testing.assert_close(u[:, :2], u2)


@pytest.mark.parametrize("name", ["spiral_trajectory_3d", "sinusoidal_trajectory_3d", "saddle_trajectory_3d"])
@PROPERTY
@given(st.integers(2, 32), st.floats(-2, 2, allow_nan=False, allow_infinity=False))
def test_PROPERTY_noncircular_frames(name, n, start):
    kwargs = {"spiral_trajectory_3d": {"z_range": 7, "n_turns": 1.3},
              "sinusoidal_trajectory_3d": {"amplitude": 2, "frequency": 3},
              "saddle_trajectory_3d": {"z_amplitude": 4, "radial_amplitude": 2}}[name]
    source, detector, u, v = getattr(G, name)(n, 10, 30, start_angle=start, device="cpu", dtype=torch.float64, **kwargs)
    _frame(source, detector, u, v)
    torch.testing.assert_close(source[:, 2], detector[:, 2])
    if name == "spiral_trajectory_3d":
        assert float(source[0, 2]) == pytest.approx(-3.5)
        assert float(source[-1, 2]) == pytest.approx(3.5)
        assert torch.all(source[1:, 2] > source[:-1, 2])


@PROPERTY
@given(st.integers(2, 40), st.floats(-2, 2, allow_nan=False, allow_infinity=False),
       st.floats(-10, 10, allow_nan=False, allow_infinity=False))
def test_PROPERTY_parallel_unit_orthogonal_frames(n, start, distance):
    ray, detector, u = G.circular_trajectory_2d_parallel(n, distance, start_angle=start, device="cpu", dtype=torch.float64)
    torch.testing.assert_close(ray.norm(dim=1), torch.ones(n, dtype=torch.float64))
    torch.testing.assert_close(u.norm(dim=1), torch.ones(n, dtype=torch.float64))
    torch.testing.assert_close((ray * u).sum(1), torch.zeros(n, dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(detector, distance * u)
    # The default angular interval is pi, with the final angle excluded.
    end_ray = torch.tensor([math.cos(start + math.pi), math.sin(start + math.pi)], dtype=torch.float64)
    assert not torch.allclose(ray[-1], end_ray)


@PROPERTY
@given(st.integers(1, 24), st.floats(1, 10, allow_nan=False, allow_infinity=False))
def test_PROPERTY_custom_3d_frame_including_exact_z_axis(n, radius):
    def path(angles, sid):
        source = torch.stack([radius * torch.sin(angles), radius * torch.cos(angles), torch.full_like(angles, 2)], dim=1)
        # Include the explicitly supported z-axis special case.
        source[0] = source.new_tensor([0, 0, radius])
        return source
    source, detector, u, v = G.custom_trajectory_3d(n, radius, 30, path, device="cpu", dtype=torch.float64)
    _frame(source, detector, u, v)
    torch.testing.assert_close(u[0], torch.tensor([1, 0, 0], dtype=torch.float64))


def test_BOUNDED_custom_parallel_normalizes_direction_and_validates_shape():
    def ray(angles):
        return torch.stack([2 * torch.cos(angles), 2 * torch.sin(angles)], dim=1)
    def origin(angles):
        return torch.zeros((len(angles), 2), dtype=angles.dtype)
    direction, _, u = G.custom_trajectory_2d_parallel(7, ray, origin, device="cpu", dtype=torch.float64)
    torch.testing.assert_close(direction.norm(dim=1), torch.ones(7, dtype=torch.float64))
    torch.testing.assert_close((direction * u).sum(1), torch.zeros(7, dtype=torch.float64))
    with pytest.raises(ValueError, match="ray_dir_fn must return tensor"):
        G.custom_trajectory_2d_parallel(7, lambda a: torch.zeros(7, 3), origin, device="cpu")
