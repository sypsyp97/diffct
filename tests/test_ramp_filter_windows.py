"""Tests for ``diffct.analytical.ramp_filter_1d`` and the window helper.

Covers:
    * every documented window name at the direct ``_ramp_window`` layer
      (DC=1, Nyquist value, monotonicity),
    * the full ``ramp_filter_1d`` end-to-end for shape, finite DC response,
      rfft vs complex-fft parity, and ``sample_spacing`` scaling,
    * a high-frequency attenuation sanity check on a step input.
"""

import math

import pytest
import torch

from diffct import ramp_filter_1d
from diffct.analytical import _ramp_window


_WINDOWS = [None, "ram-lak", "hann", "hanning", "hamming", "cosine", "shepp-logan"]


def _skip_if_no_cuda():
    # ramp_filter_1d is pure PyTorch and CPU-capable, so the CUDA check
    # is optional - but we still want to run it on CUDA when available
    # so the tests mirror the production path.
    pass


@pytest.mark.parametrize("window", _WINDOWS)
def test_ramp_window_dc_gain_is_one(window):
    freqs = torch.linspace(-0.5, 0.5, 513)
    w = _ramp_window(window, freqs)
    dc_idx = int(torch.argmin(freqs.abs()))
    assert abs(float(w[dc_idx]) - 1.0) < 1e-5, (
        f"window {window} DC gain = {float(w[dc_idx])}"
    )


@pytest.mark.parametrize("window", ["hann", "hanning", "hamming", "cosine"])
def test_ramp_window_nyquist_attenuation(window):
    freqs = torch.linspace(0.0, 0.5, 257)
    w = _ramp_window(window, freqs)
    assert float(w[-1]) < 0.15, (
        f"window {window} should strongly suppress Nyquist, got {float(w[-1])}"
    )


@pytest.mark.parametrize("window", _WINDOWS)
def test_ramp_window_non_negative(window):
    freqs = torch.linspace(-0.5, 0.5, 513)
    w = _ramp_window(window, freqs)
    assert float(w.min()) >= -1e-5


def test_ramp_filter_shape_matches_input():
    x = torch.randn(8, 128)
    y = ramp_filter_1d(x, dim=1, pad_factor=2, window="hann")
    assert y.shape == x.shape


def test_ramp_filter_finite_dc_response():
    # A finite Ram-Lak kernel has a small positive DC response. Forcing it
    # to zero biases finite, zero-extended projections.
    n = 128
    odd = torch.arange(1, n // 2, 2, dtype=torch.float64)
    expected = math.pi / 2 - 4 / math.pi * (1 / odd.square()).sum()
    y = ramp_filter_1d(torch.ones(2, n, dtype=torch.float64), window="hann")
    torch.testing.assert_close(y, torch.full_like(y, expected), rtol=0, atol=1e-12)


@pytest.mark.parametrize("n", [15, 16])
@pytest.mark.parametrize("use_rfft", [False, True])
@pytest.mark.parametrize("pad_factor", [2, 3])
def test_ramp_matches_independent_linear_convolution(n, use_rfft, pad_factor):
    x = torch.linspace(-0.4, 0.9, n, dtype=torch.float64)
    # h[0] = pi/2; h[odd] = -2/(pi*k^2), h[nonzero even] = 0.
    expected = torch.zeros_like(x)
    for i in range(n):
        for j in range(n):
            lag = i - j
            coefficient = math.pi / 2 if lag == 0 else (
                -2 / (math.pi * lag**2) if lag % 2 else 0.0
            )
            expected[i] += x[j] * coefficient / 0.7
    actual = ramp_filter_1d(x, sample_spacing=0.7, pad_factor=pad_factor,
                            use_rfft=use_rfft)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)


def test_ramp_gaussian_matches_continuous_fourier_integral():
    sigma = 6.0
    u = torch.arange(-64, 64, dtype=torch.float64)
    projection = math.sqrt(2 * math.pi) * sigma * torch.exp(-u.square() / (2 * sigma**2))
    # Independent continuous integral: no discrete kernel or FFT in reference.
    omega = torch.linspace(0, 8 / sigma, 20001, dtype=torch.float64)
    integrand = (omega * torch.exp(-sigma**2 * omega.square() / 2))[:, None]
    reference = 2 * sigma**2 * torch.trapezoid(
        integrand * torch.cos(omega[:, None] * u[None, :]), omega, dim=0,
    )
    actual = ramp_filter_1d(projection, pad_factor=2)
    assert torch.sqrt((actual - reference).square().mean()).item() < 1e-6


@pytest.mark.parametrize("window", _WINDOWS)
def test_ramp_filter_rfft_vs_complex_match(window):
    torch.manual_seed(42)
    x = torch.randn(4, 64)
    y_rfft = ramp_filter_1d(x, dim=1, pad_factor=2, window=window, use_rfft=True)
    y_cfft = ramp_filter_1d(x, dim=1, pad_factor=2, window=window, use_rfft=False)
    max_diff = (y_rfft - y_cfft).abs().max().item()
    assert max_diff < 5e-5, f"window={window} rfft vs cfft diff={max_diff}"


def test_ramp_filter_sample_spacing_scaling():
    torch.manual_seed(7)
    x = torch.randn(4, 64)
    y1 = ramp_filter_1d(x, dim=1, sample_spacing=1.0, window="hann")
    y2 = ramp_filter_1d(x, dim=1, sample_spacing=2.0, window="hann")
    # sample_spacing=2 halves the output compared to sample_spacing=1.
    # Compare with elementwise absolute error instead of a brittle ratio.
    expected = y1 * 0.5
    max_diff = (y2 - expected).abs().max().item()
    assert max_diff < 1e-5, f"expected |y2 - y1/2| ~ 0, got {max_diff}"


def test_ramp_filter_step_high_frequency_attenuation():
    """A band-limited high-frequency sinusoid should survive the ramp
    filter (it boosts high frequencies) while the DC is suppressed."""
    n = 128
    t = torch.arange(n, dtype=torch.float32)
    high_freq = torch.cos(2.0 * torch.pi * 0.25 * t).view(1, n)  # f = 0.25 cycles/sample
    dc = torch.ones(1, n)
    y_hi = ramp_filter_1d(high_freq, dim=1, pad_factor=1, window="hann")
    y_dc = ramp_filter_1d(dc, dim=1, pad_factor=1, window="hann")
    assert y_hi.abs().max().item() > 5 * y_dc.abs().max().item()
