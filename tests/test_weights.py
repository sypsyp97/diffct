"""Unit tests for the analytical-reconstruction weight helpers.

Checks ``detector_coordinates_1d``, ``angular_integration_weights``,
``fan_cosine_weights``, ``cone_cosine_weights``, and ``parker_weights``.

Detector cell centres use ``u[k] = (k - (N - 1)/2) * ds``.
This convention applies to the weight helpers, Siddon forward/backward
kernels, and FBP/FDK gather kernels.
"""

import math

import pytest
import torch

from diffct import (
    angular_integration_weights,
    cone_cosine_weights,
    detector_coordinates_1d,
    fan_cosine_weights,
    parker_weights,
)


def test_detector_coordinates_even_odd_match_cell_center_convention():
    even = detector_coordinates_1d(4, 1.0)
    odd = detector_coordinates_1d(5, 1.0)
    assert torch.allclose(even, torch.tensor([-1.5, -0.5, 0.5, 1.5]))
    assert torch.allclose(odd, torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))


def test_angular_integration_weights_full_scan_redundant():
    n = 360
    angles = torch.linspace(0.0, 2.0 * math.pi, n + 1)[:-1]
    w = angular_integration_weights(angles, redundant_full_scan=True)
    assert torch.allclose(w, torch.full_like(w, math.pi / n), atol=1e-6)
    assert abs(w.sum().item() - math.pi) < 1e-6


def test_angular_integration_weights_short_scan_not_redundant():
    # Endpoint-excluded parallel half scan [0, pi) with no redundancy factor.
    n = 180
    angles = torch.linspace(0.0, math.pi, n + 1)[:-1]
    w = angular_integration_weights(angles, redundant_full_scan=False)
    assert torch.allclose(w, torch.full_like(w, math.pi / n), atol=1e-6)
    assert abs(w.sum().item() - math.pi) < 1e-6


def test_angular_integration_weights_parallel_half_scan_ignores_redundancy_flag():
    n = 180
    angles = torch.linspace(0.0, math.pi, n + 1)[:-1]
    w = angular_integration_weights(angles, redundant_full_scan=True)
    assert torch.allclose(w, torch.full_like(w, math.pi / n), atol=1e-6)
    assert abs(w.sum().item() - math.pi) < 1e-6


def test_angular_integration_weights_downsampled_full_scan_is_periodic():
    angles_all = torch.linspace(0.0, 2.0 * math.pi, 721 + 1)[:-1]
    angles = angles_all[::3]
    w = angular_integration_weights(angles, redundant_full_scan=True)
    assert abs(w.sum().item() - math.pi) < 1e-6
    assert w[0] < w[1]
    assert w[-1] < w[1]


def test_angular_integration_weights_endpoint_included_full_scan():
    angles = torch.deg2rad(torch.arange(721, dtype=torch.float32) * 0.5)
    w = angular_integration_weights(angles, redundant_full_scan=True)
    assert abs(w.sum().item() - math.pi) < 1e-6
    assert torch.isclose(w[0], w[1] * 0.5, atol=1e-6)
    assert torch.isclose(w[-1], w[-2] * 0.5, atol=1e-6)


def test_angular_integration_weights_open_short_scan_uses_trapezoid():
    angles = torch.linspace(0.0, 0.75 * math.pi, 11)
    w = angular_integration_weights(angles, redundant_full_scan=False)
    assert torch.allclose(w[0], w[1] * 0.5)
    assert torch.allclose(w[-1], w[-2] * 0.5)
    assert abs(w.sum().item() - 0.75 * math.pi) < 1e-6


def test_fan_cosine_weights_peak_at_origin():
    w = fan_cosine_weights(7, 1.0, 1000.0)
    # cos(gamma) = sdd / sqrt(sdd^2 + u^2): max at u closest to 0.
    # For odd N=7, bin 3 is exactly at the origin.
    argmax = int(torch.argmax(w).item())
    assert argmax == 3


def test_cone_cosine_weights_peak_at_detector_center():
    w = cone_cosine_weights(9, 9, 1.0, 1.0, 1200.0)
    # For odd N=9, cell (4, 4) is exactly at (u=0, v=0).
    flat = w.flatten()
    peak = int(flat.argmax().item())
    pu, pv = peak // 9, peak % 9
    assert (pu, pv) == (4, 4)


def test_parker_full_scan_is_one():
    n = 360
    angles = torch.linspace(0.0, 2.0 * math.pi, n + 1)[:-1]
    pw = parker_weights(angles, num_detectors=64, detector_spacing=1.0, sdd=800.0)
    assert torch.allclose(pw, torch.ones_like(pw))


def test_parker_short_scan_range_is_bounded():
    # Minimal short scan: pi + 2*gamma_max.
    n_det = 128
    spacing = 1.0
    sdd = 400.0
    u_max = ((n_det - 1) * 0.5) * spacing
    gamma_max = math.atan(u_max / sdd)
    coverage = math.pi + 2.0 * gamma_max

    n_views = 240
    step = coverage / n_views
    angles = torch.arange(n_views, dtype=torch.float32) * step

    pw = parker_weights(angles, n_det, spacing, sdd)
    assert pw.min() >= 0.0
    assert pw.max() <= 1.0 + 1e-5


@pytest.mark.parametrize("coverage_degrees", [240, 330, 350])
def test_parker_overscan_integrates_each_detector_to_pi(coverage_degrees):
    # The fan width must not turn a 330/350 degree acquisition into a full scan.
    angles = torch.linspace(0.3, 0.3 + math.radians(coverage_degrees), 2049)
    weights = parker_weights(angles, 64, 1.0, 80.0)
    d_beta = angular_integration_weights(angles, redundant_full_scan=False)
    integrated = (weights * d_beta[:, None]).sum(dim=0)
    torch.testing.assert_close(integrated, torch.full_like(integrated, math.pi),
                               rtol=0, atol=2e-5)
    assert weights[0].max() < 1e-6
    assert weights[-1].max() < 1e-6


def test_parker_narrow_fan_full_scan_is_one():
    angles = torch.arange(360) * (2 * math.pi / 360)
    weights = parker_weights(angles, 2, 0.01, 800.0)
    torch.testing.assert_close(weights, torch.ones_like(weights))


@pytest.mark.parametrize("coverage_degrees", [359.915, 359.92093, 359.925, 360.0])
@pytest.mark.parametrize("endpoint_included", [False, True])
def test_parker_and_angular_weights_preserve_amplitude_near_full_scan(
    coverage_degrees, endpoint_included,
):
    n = 7200
    coverage = math.radians(coverage_degrees)
    angles = torch.linspace(0.0, coverage, n if endpoint_included else n + 1)[:n]
    parker = parker_weights(angles, 5, 1.0, 80.0)
    angular = angular_integration_weights(angles)
    # Both short-scan tapers and full-scan redundancy must integrate each
    # ray exactly once. Disagreeing classifications halve the amplitude.
    integral = (parker * angular[:, None]).sum(dim=0)
    torch.testing.assert_close(integral, torch.full_like(integral, math.pi),
                               rtol=0, atol=2e-5)
