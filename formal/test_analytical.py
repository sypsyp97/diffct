"""PROPERTY TEST / BOUNDED checks of actual CPU analytical helper bodies.

The AST loader compiles unchanged CPU FunctionDefs from analytical.py.
It avoids importing CUDA package initializers on CPU-only platforms, and
never substitutes CUDA implementations. CUDA wrappers are not executed.
Source: analytical.py:51-356. No CPU result is GPU validation evidence.
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st


def _load_cpu_helpers():
    path = Path(__file__).resolve().parents[1] / "diffct" / "analytical.py"
    names = {
        "detector_coordinates_1d", "_angular_scan_period", "angular_integration_weights",
        "fan_cosine_weights", "cone_cosine_weights", "parker_weights", "_ramp_window", "ramp_filter_1d",
    }
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == names
    # Preserve the real supported-window constant, including error behavior.
    selected += [node for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "_RAMP_WINDOWS" for t in node.targets)]
    namespace = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


A = _load_cpu_helpers()
FINITE = st.floats(-10, 10, allow_nan=False, allow_infinity=False)
PROPERTY = settings(max_examples=80, deadline=None, derandomize=True)


@pytest.mark.parametrize("period", [math.pi, 2 * math.pi])
@pytest.mark.parametrize("included", [False, True])
@pytest.mark.parametrize("redundant", [False, True])
def test_BOUNDED_angular_half_and_full_totals(period, included, redundant):
    n = 33
    angles = torch.linspace(0, period, n + (not included), dtype=torch.float64)
    if not included:
        angles = angles[:-1]
    weights = A.angular_integration_weights(angles, redundant)
    expected = period * (0.5 if redundant and period == 2 * math.pi else 1)
    assert torch.all(weights >= 0)
    assert float(weights.sum()) == pytest.approx(expected, rel=2e-6)


@PROPERTY
@given(st.lists(st.floats(0, 1, allow_nan=False, allow_infinity=False), min_size=2, max_size=32))
def test_PROPERTY_open_angular_weights_nonnegative_and_total(samples):
    angles = torch.tensor(samples, dtype=torch.float64)
    weights = A.angular_integration_weights(angles)
    assert torch.all(weights >= 0)
    expected = float(angles.float().max() - angles.float().min())
    assert float(weights.sum()) == pytest.approx(expected, abs=2e-7)
    # Reverse order, including repeated angles: compare aggregate mass per angle.
    reordered = A.angular_integration_weights(angles.flip(0)).flip(0)
    for angle in angles.float().unique():
        mask = angles.float() == angle
        torch.testing.assert_close(weights[mask].sum(), reordered[mask].sum(), atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("period", [math.pi, 2 * math.pi])
@pytest.mark.parametrize("included", [False, True])
def test_BOUNDED_angular_permutation_with_duplicate_endpoints(period, included):
    angles = torch.linspace(0, period, 25 + (not included))
    if not included:
        angles = angles[:-1]
    angles = torch.cat([angles, angles[:1], angles[-1:]])
    perm = torch.randperm(len(angles), generator=torch.Generator().manual_seed(17))
    original = A.angular_integration_weights(angles)
    shuffled = A.angular_integration_weights(angles[perm])
    assert torch.all(shuffled >= 0)
    assert float(shuffled.sum()) == pytest.approx(math.pi, rel=2e-6)
    # Individual weights of coincident views need not agree; the sampled measure does.
    for angle in angles.unique():
        torch.testing.assert_close(original[angles == angle].sum(), shuffled[angles[perm] == angle].sum())


@PROPERTY
@given(st.integers(2, 64), st.floats(0.1, 5, allow_nan=False, allow_infinity=False))
def test_PROPERTY_unique_angle_permutation(n, offset):
    angles = torch.arange(n, dtype=torch.float64) * (0.7 / n) + offset
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(n))
    torch.testing.assert_close(A.angular_integration_weights(angles)[perm], A.angular_integration_weights(angles[perm]))


@pytest.mark.parametrize("window", [None, "ram-lak", "hann", "hanning", "hamming", "cosine", "shepp-logan"])
def test_BOUNDED_ramp_windows_even_nonnegative_and_unit_dc(window):
    frequencies = torch.linspace(-0.5, 0.5, 101, dtype=torch.float64)
    weights = A._ramp_window(window, frequencies)
    torch.testing.assert_close(weights, weights.flip(0), atol=2e-15, rtol=2e-15)
    assert float(weights[50]) == pytest.approx(1.0, abs=2e-15)
    assert torch.all(weights >= -2e-15)
    assert torch.all(weights <= 1 + 2e-15)


def _direct_filter(values, pad, spacing):
    """Independent circular convolution, evaluated in the retained sample domain."""
    n, m = len(values), len(values) * pad
    output = np.zeros(n)
    for j in range(n):
        for i in range(n):
            lag = min((j - i) % m, (i - j) % m)
            h = math.pi / 2 if lag == 0 else (-2 / (math.pi * lag ** 2) if lag % 2 else 0)
            output[j] += float(values[i]) * h / spacing
    return torch.from_numpy(output)


@pytest.mark.parametrize("n", [3, 4, 7, 8])
@pytest.mark.parametrize("pad", [1, 2, 3])
@pytest.mark.parametrize("real_fft", [False, True])
def test_BOUNDED_fft_ramp_matches_direct_convolution(n, pad, real_fft):
    values = torch.tensor([math.sin(i + 0.3) + i / 7 for i in range(n)], dtype=torch.float64)
    actual = A.ramp_filter_1d(values, sample_spacing=0.7, pad_factor=pad, use_rfft=real_fft)
    torch.testing.assert_close(actual, _direct_filter(values, pad, 0.7), atol=4e-14, rtol=4e-14)


@PROPERTY
@given(st.lists(FINITE, min_size=2, max_size=20), st.floats(0.1, 5, allow_nan=False, allow_infinity=False))
def test_PROPERTY_ramp_reversal_spacing_and_fft_path_agreement(samples, spacing):
    values = torch.tensor(samples, dtype=torch.float64)
    base = A.ramp_filter_1d(values, pad_factor=2)
    reverse = A.ramp_filter_1d(values.flip(0), pad_factor=2).flip(0)
    torch.testing.assert_close(base, reverse, atol=5e-13, rtol=5e-13)
    scaled = A.ramp_filter_1d(values, sample_spacing=spacing, pad_factor=2, use_rfft=False)
    torch.testing.assert_close(scaled, base / spacing, atol=5e-12, rtol=5e-12)


@PROPERTY
@given(st.integers(1, 40), st.floats(0.01, 3, allow_nan=False, allow_infinity=False),
       st.floats(1, 100, allow_nan=False, allow_infinity=False))
def test_PROPERTY_detector_centres_and_cosine_symmetries(n, pitch, sdd):
    coordinates = A.detector_coordinates_1d(n, pitch, dtype=torch.float64)
    torch.testing.assert_close(coordinates, -coordinates.flip(0), atol=1e-13, rtol=1e-13)
    fan = A.fan_cosine_weights(n, pitch, sdd, dtype=torch.float64)
    torch.testing.assert_close(fan, fan.flip(0), atol=1e-14, rtol=1e-14)
    assert torch.all((fan > 0) & (fan <= 1))
    cone = A.cone_cosine_weights(n, 7, pitch, 0.3, sdd, dtype=torch.float64)
    torch.testing.assert_close(cone, cone.flip((0, 1)), atol=1e-14, rtol=1e-14)
    assert torch.all((cone > 0) & (cone <= 1))
    # The central cone row equals the fan weights because v=0.
    torch.testing.assert_close(cone[:, 3], fan)


def test_BOUNDED_parker_taper_pair_uses_plus_gamma_and_full_scan_passthrough():
    n, pitch, sdd, delta = 9, 0.3, 10.0, 0.5
    base = torch.linspace(0, math.pi + 2 * delta, 401)
    gamma = math.atan((n // 2 - 2) * pitch / sdd)
    beta = 0.4 * 2 * (delta - gamma)
    pair = beta + math.pi + 2 * gamma
    angles = torch.cat([base, torch.tensor([beta, pair])])
    weights = A.parker_weights(angles, n, pitch, sdd)
    k = n // 2 + (n // 2 - 2)
    assert float(weights[-2, k] + weights[-1, n - 1 - k]) == pytest.approx(1, abs=2e-6)
    assert torch.all((weights >= 0) & (weights <= 1))
    # Numerical check only: epsilon denominators exclude an exact runtime identity.
    full = torch.arange(64) * (2 * math.pi / 64)
    torch.testing.assert_close(A.parker_weights(full, n, pitch, sdd), torch.ones(64, n))


@pytest.mark.parametrize("degrees", [240, 330, 350])
def test_BOUNDED_parker_weighted_angular_mass(degrees):
    angles = torch.linspace(0, math.radians(degrees), 1601)
    weights = A.parker_weights(angles, 11, 0.5, 10)
    mass = (weights * A.angular_integration_weights(angles)[:, None]).sum(0)
    torch.testing.assert_close(mass, torch.full_like(mass, math.pi), atol=2e-5, rtol=2e-5)
