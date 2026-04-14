"""Unit tests for the analytical-reconstruction weight helpers.

Checks ``detector_coordinates_1d``, ``angular_integration_weights``,
``fan_cosine_weights``, ``cone_cosine_weights``, and ``parker_weights``.

Note: the dev-branch detector grid convention is
``u[k] = (k - N/2) * ds``, which differs from the main branch's
``(k - (N-1)/2)`` cell-centre convention. Dev's convention is built
into every kernel (Siddon forward/backward, FBP/FDK gather) so the
helper mirrors it. Expected values here reflect the dev convention.
"""

import math

import torch

from diffct import (
    angular_integration_weights,
    cone_cosine_weights,
    detector_coordinates_1d,
    fan_cosine_weights,
    parker_weights,
)


def test_detector_coordinates_even_odd_match_dev_convention():
    even = detector_coordinates_1d(4, 1.0)
    odd = detector_coordinates_1d(5, 1.0)
    # Dev convention: u[k] = (k - N/2) * ds
    assert torch.allclose(even, torch.tensor([-2.0, -1.0, 0.0, 1.0]))
    assert torch.allclose(odd, torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5]))


def test_angular_integration_weights_full_scan_redundant():
    n = 360
    angles = torch.linspace(0.0, 2.0 * math.pi, n + 1)[:-1]
    w = angular_integration_weights(angles, redundant_full_scan=True)
    # Trapezoidal rule over [0, 2*pi - step] with 1/2 redundancy factor.
    # For large n this converges to pi.
    assert abs(w.sum().item() - math.pi) < 1e-2


def test_angular_integration_weights_short_scan_not_redundant():
    # Half scan [0, pi] with no redundancy factor
    n = 180
    angles = torch.linspace(0.0, math.pi, n + 1)[:-1]
    w = angular_integration_weights(angles, redundant_full_scan=False)
    # Trapezoidal rule over [0, pi - step], for large n converges to pi.
    assert abs(w.sum().item() - math.pi) < 1e-1


def test_fan_cosine_weights_peak_at_origin():
    w = fan_cosine_weights(7, 1.0, 1000.0)
    # cos(gamma) = sdd / sqrt(sdd^2 + u^2): max at u closest to 0.
    # Dev convention: for N=7 (odd), bin 3 has u = -0.5 (closest to 0).
    # Actually wait, for N=7, (k - 3.5)*1 so bins are [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    # Two bins (3 and 4) are equidistant from origin.
    argmax = int(torch.argmax(w).item())
    assert argmax in (3, 4)


def test_cone_cosine_weights_peak_near_detector_center():
    w = cone_cosine_weights(9, 9, 1.0, 1.0, 1200.0)
    # For N=9 (odd) the closest bins to (u=0, v=0) are the 4 around index (4, 4).
    # Peak index should be one of those.
    flat = w.flatten()
    peak = int(flat.argmax().item())
    pu, pv = peak // 9, peak % 9
    assert pu in (3, 4) and pv in (3, 4)


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
    u_max = (n_det * 0.5) * spacing  # dev convention
    gamma_max = math.atan(u_max / sdd)
    coverage = math.pi + 2.0 * gamma_max

    n_views = 240
    step = coverage / n_views
    angles = torch.arange(n_views, dtype=torch.float32) * step

    pw = parker_weights(angles, n_det, spacing, sdd)
    assert pw.min() >= 0.0
    assert pw.max() <= 1.0 + 1e-5
