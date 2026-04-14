"""Tests for every window option supported by ``ramp_filter_1d``.

Covers the five public window names
(``None``/``"ram-lak"``, ``"hann"``, ``"hamming"``, ``"cosine"``,
``"shepp-logan"``) at three layers:

1. ``_ramp_window`` directly (DC gain, Nyquist value, monotonicity,
   value range).
2. ``ramp_filter_1d`` end-to-end on a clean 1D signal, checking
   shape preservation, reality, DC annihilation (the ramp kills the
   DC component regardless of window), pad_factor correctness, and
   rfft vs complex-fft parity.
3. Sanity check that windowed variants really do attenuate
   high-frequency content compared to the bare Ram-Lak ramp.
"""

import math

import pytest
import torch

from diffct.differentiable import ramp_filter_1d
from diffct.differentiable import _ramp_window


# Every window name this module supports, plus their *expected*
# values at DC (freq=0) and at Nyquist (|freq|=0.5).
# Values come from the window definitions in ``_ramp_window``.
WINDOW_SPECS = [
    # name,                   dc_gain, nyquist_value
    (None,                     1.0,    1.0),       # unwindowed
    ("ram-lak",                1.0,    1.0),
    ("hann",                   1.0,    0.0),       # 0.5*(1 + cos(pi))
    ("hanning",                1.0,    0.0),
    ("hamming",                1.0,    0.08),      # 0.54 - 0.46
    ("cosine",                 1.0,    0.0),       # cos(pi/2)
    ("shepp-logan",            1.0,    2.0 / math.pi),  # sinc(1/2)
]


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


# ---------------------------------------------------------------------------
# 1. Direct tests of the window helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,dc,nyq", WINDOW_SPECS)
def test_ramp_window_endpoints_and_shape(name, dc, nyq):
    """DC bin equals ``dc``, last positive frequency approaches ``nyq``."""
    n = 64
    freqs = torch.fft.rfftfreq(n, dtype=torch.float32)
    w = _ramp_window(name, freqs)

    assert w.shape == freqs.shape
    # DC must be exactly preserved.
    assert torch.isclose(w[0], torch.tensor(dc, dtype=torch.float32), atol=1e-5), (
        f"window {name!r}: DC gain {w[0].item():.4f} != {dc}"
    )
    # Nyquist bin is freqs[-1] == 0.5 for even n.
    assert torch.isclose(
        w[-1], torch.tensor(nyq, dtype=torch.float32), atol=1e-5
    ), f"window {name!r}: Nyquist value {w[-1].item():.4f} != {nyq}"
    # Every documented window is a non-negative envelope in [0, 1].
    assert w.min().item() >= -1e-6
    assert w.max().item() <= 1.0 + 1e-6


@pytest.mark.parametrize("name,_dc,_nyq", WINDOW_SPECS)
def test_ramp_window_monotonic_from_dc_to_nyquist(name, _dc, _nyq):
    """All supported windows are non-increasing from DC to Nyquist."""
    n = 128
    freqs = torch.fft.rfftfreq(n, dtype=torch.float32)
    w = _ramp_window(name, freqs)
    diffs = w[1:] - w[:-1]
    # Allow a 1e-6 numerical slack; the Ram-Lak constant window should
    # give exact zero diffs.
    assert (diffs <= 1e-6).all(), f"window {name!r} not monotonic: {diffs.max().item()}"


def test_ramp_window_unknown_name_raises():
    """A bogus window name should raise, not silently fall back."""
    freqs = torch.fft.rfftfreq(16, dtype=torch.float32)
    with pytest.raises(ValueError, match="Unknown ramp window"):
        _ramp_window("not-a-real-window", freqs)


# ---------------------------------------------------------------------------
# 2. End-to-end ramp_filter_1d tests
# ---------------------------------------------------------------------------


def _reference_signal(n, device):
    """Smooth-ish 1D signal with non-zero DC and non-trivial bandwidth."""
    t = torch.linspace(-1.0, 1.0, n, device=device, dtype=torch.float32)
    # Triangle + gaussian bump - has a clear peak the ramp filter
    # will sharpen.
    return torch.clamp(1.0 - t.abs(), min=0.0) + 0.3 * torch.exp(-(t * 4.0) ** 2)


@pytest.mark.cuda
@pytest.mark.parametrize("name", [s[0] for s in WINDOW_SPECS])
def test_ramp_filter_1d_each_window_runs(name):
    """``ramp_filter_1d`` with every window option produces a finite
    real tensor of the same shape as the input."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    n = 64
    sino = _reference_signal(n, device).view(1, n).repeat(8, 1)

    out = ramp_filter_1d(
        sino,
        dim=1,
        sample_spacing=1.0,
        pad_factor=2,
        window=name,
    )
    assert out.shape == sino.shape
    assert torch.isfinite(out).all()
    assert out.dtype == sino.dtype


@pytest.mark.cuda
def test_ramp_filter_1d_kills_dc_component():
    """The ramp filter has ``|omega|=0`` at DC, so without padding the
    output must have numerically zero mean along ``dim`` regardless of
    window. (With ``pad_factor > 1`` only the *padded* signal has zero
    mean, because cropping back drops part of the filtered support.)"""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    n = 128
    sino = _reference_signal(n, device).view(1, n).repeat(4, 1)

    for name in [s[0] for s in WINDOW_SPECS]:
        out = ramp_filter_1d(
            sino, dim=1, sample_spacing=1.0, pad_factor=1, window=name
        )
        row_mean = out.mean(dim=1).abs().max().item()
        assert row_mean < 1e-4, (
            f"window {name!r}: row-mean should be zero but is {row_mean:.4e}"
        )


@pytest.mark.cuda
def test_ramp_filter_1d_rfft_matches_complex_fft():
    """``use_rfft=True`` (default) and ``use_rfft=False`` must agree
    to within float32 precision for real-valued input."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    n = 96
    sino = _reference_signal(n, device).view(1, n).repeat(5, 1)

    for name in [None, "hann", "shepp-logan"]:
        a = ramp_filter_1d(sino, dim=1, pad_factor=2, window=name, use_rfft=True)
        b = ramp_filter_1d(sino, dim=1, pad_factor=2, window=name, use_rfft=False)
        diff = (a - b).abs().max().item()
        assert diff < 1e-4, f"rfft vs complex fft mismatch for {name!r}: {diff:.2e}"


@pytest.mark.cuda
def test_ramp_filter_1d_pad_factor_preserves_length():
    """Zero-padding must not change the output length along ``dim``."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    n = 50  # odd-ish length to catch off-by-one bugs in the symmetric pad
    sino = _reference_signal(n, device).view(1, n).repeat(3, 1)

    for pad in [1, 2, 4]:
        out = ramp_filter_1d(sino, dim=1, pad_factor=pad, window="hann")
        assert out.shape == sino.shape, (
            f"pad_factor={pad} produced shape {tuple(out.shape)}"
        )


@pytest.mark.cuda
def test_ramp_filter_1d_sample_spacing_scaling():
    """Doubling ``sample_spacing`` should halve the output amplitude
    (the filter is rescaled by ``1/sample_spacing``)."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    n = 128
    sino = _reference_signal(n, device).view(1, n)

    a = ramp_filter_1d(sino, dim=1, sample_spacing=1.0, pad_factor=2, window=None)
    b = ramp_filter_1d(sino, dim=1, sample_spacing=2.0, pad_factor=2, window=None)

    # b should equal a / 2 to within float32 precision.
    ratio = (a / 2.0 - b).abs().max().item()
    assert ratio < 1e-5, f"sample_spacing scaling off: max abs diff {ratio:.2e}"


@pytest.mark.cuda
def test_ramp_filter_1d_windowed_attenuates_high_frequency():
    """Apodized windows (hann/hamming/cosine/shepp-logan) should have
    strictly smaller peak absolute output than the bare Ram-Lak ramp
    on a signal with broadband content - the windows kill the high
    frequencies that would otherwise create ringing."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    n = 128
    # A step function has infinite bandwidth - perfect stress test.
    sino = torch.zeros(1, n, device=device, dtype=torch.float32)
    sino[0, n // 2 :] = 1.0

    ramlak = ramp_filter_1d(sino, dim=1, pad_factor=2, window=None)
    rl_peak = ramlak.abs().max().item()

    for name in ["hann", "hamming", "cosine", "shepp-logan"]:
        out = ramp_filter_1d(sino, dim=1, pad_factor=2, window=name)
        peak = out.abs().max().item()
        assert peak < rl_peak, (
            f"window {name!r} did not attenuate high freq: peak {peak:.4f} >= "
            f"ram-lak peak {rl_peak:.4f}"
        )
