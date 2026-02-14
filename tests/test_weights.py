import math

import torch

from diffct.differentiable import (
    angular_integration_weights,
    cone_cosine_weights,
    detector_coordinates_1d,
    fan_cosine_weights,
    parker_weights,
)


def test_detector_coordinates_centering_even_odd():
    even = detector_coordinates_1d(4, 1.0)
    odd = detector_coordinates_1d(5, 1.0)
    assert torch.allclose(even, torch.tensor([-1.5, -0.5, 0.5, 1.5]))
    assert torch.allclose(odd, torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))


def test_angular_integration_weights_full_scan_redundant():
    n = 360
    angles = torch.linspace(0.0, 2.0 * math.pi, n + 1)[:-1]
    w = angular_integration_weights(angles, redundant_full_scan=True)
    # Redundant full scan should integrate to pi.
    assert torch.isclose(w.sum(), torch.tensor(math.pi, dtype=w.dtype), atol=1e-4)


def test_fan_cosine_weights_are_symmetric():
    w = fan_cosine_weights(7, 1.0, 1000.0)
    assert torch.allclose(w, torch.flip(w, dims=[0]), atol=1e-6)


def test_cone_cosine_weights_peak_at_detector_center():
    w = cone_cosine_weights(9, 9, 1.0, 1.0, 1200.0)
    center = w[4, 4]
    assert torch.all(center >= w)


def test_parker_full_scan_is_one():
    n = 360
    angles = torch.linspace(0.0, 2.0 * math.pi, n + 1)[:-1]
    pw = parker_weights(angles, num_detectors=64, detector_spacing=1.0, sdd=800.0)
    assert torch.allclose(pw, torch.ones_like(pw))


def test_parker_short_scan_range_is_bounded():
    # Create a minimal short scan that satisfies pi + 2*gamma_max.
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
