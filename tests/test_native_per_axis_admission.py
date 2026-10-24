"""N3 practical pitch limits apply only to the native per-axis path.

Run with NUMBA_ENABLE_CUDASIM=1 NUMBA_CUDA_USE_NVIDIA_BINDING=0.
The pre-feature library must fail the per-axis acceptance tests.
"""

from dataclasses import replace

import pytest
import torch

from tests._native_per_axis_spacing import (
    NATIVE_BEAMS, NATIVE_REQUIRED,
    case, close, cuda_device as cuda_device, data, native_device as native_device, native_only, operator,
)


BEAM_AXES = [("parallel", 0), ("parallel", 1), ("fan", 0), ("fan", 1),
             ("cone", 0), ("cone", 1), ("cone", 2)]
LIMIT_REASONS = {
    "ratio": r"(?i)(?=.*(?:spacing|pitch|voxel))(?=.*ratio)",
    "range": (r"(?i)(?=.*(?:spacing|pitch|voxel))"
              r"(?=.*(?:range|bounds|between|outside|minimum|maximum|1e[-+]?0?6|0\.000001|1000000))"),
}


def boundary_scan(beam, spacing):
    scan = case(beam)
    rank = len(scan.shape)
    shape = (3, 5, 7) if rank == 3 else (3, 5)
    pitches = (spacing,) * rank if isinstance(spacing, float) else spacing
    directions = torch.eye(rank, dtype=torch.float64)
    # World coordinate axes reverse the tensor axis order. Scale each central
    # ray to its own axis extent to keep the geometry well within FP32 guards.
    radii = [4 * size * pitch for size, pitch in zip(shape, pitches)][::-1]
    centers = torch.tensor(radii, dtype=torch.float64)[:, None] * directions
    u = torch.roll(directions, 1, dims=1)
    trajectory = ((directions, centers, u) if beam == "parallel" else (-centers, centers, u))
    if rank == 3:
        trajectory += (torch.linalg.cross(directions, u, dim=-1),)
    return replace(scan, shape=shape, spacing=spacing, trajectory=trajectory,
                   detector=(1,) * (rank - 1))


@NATIVE_REQUIRED
@pytest.mark.parametrize("backend", ["numba", "auto"])
@pytest.mark.parametrize("beam,axis", BEAM_AXES)
@pytest.mark.parametrize("limit", ["ratio", "minimum", "maximum"])
def test_n3_inclusive_pitch_limits_run_native_and_match_float64(native_device, backend, beam, axis, limit):
    other, changed = {"ratio": (1., 1e4), "minimum": (1e-2, 1e-6), "maximum": (1e2, 1e6)}[limit]
    rank = len(case(beam).shape)
    spacing = tuple(changed if index == axis else other for index in range(rank))
    assert min(spacing) >= 1e-6 and max(spacing) <= 1e6
    assert max(spacing) / min(spacing) == 1e4
    scan = boundary_scan(beam, spacing)
    image = .5 + data(scan.shape).abs()
    rays = .5 + data(scan.projection_shape, phase=.31).abs()
    reference = operator(scan, backend="torch", precision="float64")
    expected = reference.project(image.double()), reference.backproject(rays.double())
    with native_only() as bridge:
        native = operator(scan, backend=backend)
        assert native.backend == "numba"
        actual = native.project(image.to(native_device)), native.backproject(rays.to(native_device))
        assert bridge.call_count > 0
    for result, wanted in zip(actual, expected):
        assert torch.isfinite(result).all() and torch.isfinite(wanted).all()
        assert wanted.abs().max() > 0
        # Measure in the smallest pitch unit so the absolute tolerance cannot
        # hide zero output for a physical length near the lower pitch limit.
        close(result / min(spacing), wanted / min(spacing))


@NATIVE_REQUIRED
@pytest.mark.parametrize("beam,axis", BEAM_AXES)
@pytest.mark.parametrize("limit", ["ratio", "minimum", "maximum"])
def test_n3_outside_pitch_limits_raise_or_select_warned_torch(native_device, beam, axis, limit):
    other, changed = {"ratio": (1., 10001.), "minimum": (1e-6, .999e-6),
                      "maximum": (1e6, 1.001e6)}[limit]
    rank = len(case(beam).shape)
    spacing = tuple(changed if index == axis else other for index in range(rank))
    scan = boundary_scan(beam, spacing)
    reason = LIMIT_REASONS["ratio" if limit == "ratio" else "range"]
    with pytest.raises((ValueError, RuntimeError, OverflowError), match=reason) as caught:
        operator(scan, backend="numba")
    assert "scalar" not in str(caught.value).lower()
    with pytest.warns(Warning, match=reason):
        fallback = operator(scan, backend="auto")
    assert fallback.backend == "torch"


@NATIVE_REQUIRED
@pytest.mark.parametrize("backend", ["numba", "auto"])
@pytest.mark.parametrize("beam", NATIVE_BEAMS)
@pytest.mark.parametrize("spacing", [.999e-6, 1.001e6])
def test_unchanged_scalar_admission_outside_per_axis_pitch_limits(native_device, backend, beam, spacing):
    scan = boundary_scan(beam, spacing)
    with native_only():
        native = operator(scan, backend=backend)
    assert native.backend == "numba"
