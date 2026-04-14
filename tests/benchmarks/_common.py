"""Shared helpers for the benchmark suite.

Keeps the per-geometry test files small: phantom builders, CUDA sync
wrapping, and the common ``skip if no CUDA`` guard all live here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch


def skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


def sync_call(fn, *args, **kwargs):
    """Run ``fn(*args, **kwargs)``, then ``torch.cuda.synchronize()``.

    Without the sync, ``pytest-benchmark`` would measure asynchronous
    kernel-launch latency instead of real GPU execution time.
    """
    out = fn(*args, **kwargs)
    torch.cuda.synchronize()
    return out


def make_phantom_2d(n, device):
    """Deterministic random phantom of shape ``(n, n)`` on ``device``."""
    torch.manual_seed(0xD1FFC7)
    return torch.randn(n, n, device=device, dtype=torch.float32).contiguous()


def make_phantom_3d(n, device):
    """Deterministic random phantom of shape ``(n, n, n)`` on ``device``."""
    torch.manual_seed(0xD1FFC7)
    return torch.randn(n, n, n, device=device, dtype=torch.float32).contiguous()


def make_sinogram_2d(n_ang, n_det, device):
    """Deterministic random sinogram of shape ``(n_ang, n_det)``."""
    torch.manual_seed(0xC7D1FF)
    return torch.randn(n_ang, n_det, device=device, dtype=torch.float32).contiguous()


def make_sinogram_3d(n_views, det_u, det_v, device):
    """Deterministic random sinogram of shape ``(n_views, det_u, det_v)``."""
    torch.manual_seed(0xC7D1FF)
    return torch.randn(
        n_views, det_u, det_v, device=device, dtype=torch.float32
    ).contiguous()


def full_scan_angles(num_angles, device):
    """Uniformly-spaced angles on ``[0, 2*pi)`` as a float32 tensor."""
    return torch.linspace(
        0.0, 2.0 * math.pi, num_angles + 1, device=device, dtype=torch.float32
    )[:-1].contiguous()
