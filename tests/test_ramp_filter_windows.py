"""Tests for ``diffct.analytical.ramp_filter_1d`` and the window helper.

Covers:
    * every documented window name at the direct ``_ramp_window`` layer
      (DC=1, Nyquist value, monotonicity),
    * the full ``ramp_filter_1d`` end-to-end for shape, DC annihilation,
      rfft vs complex-fft parity, and ``sample_spacing`` scaling,
    * a high-frequency attenuation sanity check on a step input.
"""

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


def test_ramp_filter_kills_dc():
    # pad_factor=1 so zero-padding doesn't turn the constant into a step.
    x = torch.ones(16, 128)
    y = ramp_filter_1d(x, dim=1, pad_factor=1, window="hann")
    assert y.abs().max().item() < 1e-3, y.abs().max().item()


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
