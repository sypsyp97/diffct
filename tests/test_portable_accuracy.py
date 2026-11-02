"""Quantitative FBP/FDK accuracy on the portable PyTorch backend.

``tests/test_fbp_parallel_accuracy.py``, ``test_fbp_fan_accuracy.py`` and
``test_fdk_cone_accuracy.py`` assert the RMSE, amplitude-floor and
amplitude-ceiling bounds that pin the analytical constant, the ``(sid/U)^2``
FDK weight and the ramp-filter scale. All three call ``_skip_if_no_cuda()``,
and the ``cpu`` CI job runs on a runner without a GPU, so none of those
bounds is checked by CI today.

This module asserts the same three bounds through the public ``Projector``,
``fbp`` and ``fdk`` entry points on CPU, at sizes that finish in seconds. A
regression in the analytical constant, the cone weight or the filter scale
moves the RMSE well past the thresholds below, so the portable path gets the
guarantee the native tests already have.

Measured on a 64-core CPU worker (PyTorch 2.4), with the thresholds used by
the native tests:

    parallel 128^3, 360 angles, 256 bins ......... RMSE 0.0495
    fan      96^2, 360 angles, 192 bins .......... RMSE 0.0251
    cone     32^3, 180 views, 128x128 at du 1.0 .. RMSE 0.0376

The cone scan is the smallest of the three because its cost grows with the
detector area; 128 bins at unit spacing keep the detector inside the volume's
cone while sampling finely enough for the threshold.

The last test covers Parker weighting end to end. ``tests/test_weights.py``
checks ``parker_weights`` as a unit and the native suite checks that ``fbp``
composes it before filtering, but nothing checks that it actually repairs a
    short scan: without it the redundant-ray artefact drives the reconstruction
    down to about -0.38, well past the amplitude floor the native tests use,
    while Parker brings the minimum back to about -0.10.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from diffct import Projector, fbp, fdk
from diffct.geometry import (circular_trajectory_2d_fan,
                             circular_trajectory_2d_parallel,
                             circular_trajectory_3d)

from ._phantoms import shepp_logan_2d, shepp_logan_3d

# Bounds copied from the native accuracy tests: RMSE against the phantom, a
# floor on the signed reconstruction and a ceiling on its rectified amplitude.
RMSE_BOUND = 0.1
FLOOR_BOUND = -0.2
CEILING_BOUND = 1.5


def _check(reco, truth):
    assert torch.isfinite(reco).all()
    assert reco.shape == truth.shape
    rmse = torch.sqrt(torch.mean((reco - truth) ** 2)).item()
    assert rmse < RMSE_BOUND, f"portable RMSE too high: {rmse:.5f}"
    assert reco.min().item() > FLOOR_BOUND, f"amplitude undershoot: {reco.min().item():.5f}"
    assert F.relu(reco).max().item() < CEILING_BOUND, \
        f"amplitude overshoot: {F.relu(reco).max().item():.5f}"


def test_parallel_fbp_shepp_logan_rmse_is_small():
    """Portable parallel-beam FBP keeps the native RMSE and amplitude bounds."""
    n, angles, bins = 128, 360, 256
    image = torch.tensor(shepp_logan_2d(n), dtype=torch.float32)
    trajectory = circular_trajectory_2d_parallel(angles, device="cpu", dtype=torch.float32)
    operator = Projector(trajectory, (n, n), (bins,), beam="parallel",
                         detector_spacing=1.0, backend="torch")
    sinogram = operator.project(image)
    view_angles = torch.linspace(0.0, math.pi, angles + 1)[:-1]
    _check(fbp(sinogram, trajectory, (n, n), view_angles, beam="parallel"), image)


def test_fan_fbp_shepp_logan_rmse_is_small():
    """Portable fan-beam FBP keeps the native RMSE and amplitude bounds."""
    n, angles, bins, sid, sdd = 96, 360, 192, 500.0, 800.0
    image = torch.tensor(shepp_logan_2d(n), dtype=torch.float32)
    trajectory = circular_trajectory_2d_fan(angles, sid, sdd, device="cpu", dtype=torch.float32)
    operator = Projector(trajectory, (n, n), (bins,), beam="fan",
                         detector_spacing=1.0, backend="torch")
    sinogram = operator.project(image)
    view_angles = torch.linspace(0.0, math.tau, angles + 1)[:-1]
    _check(fbp(sinogram, trajectory, (n, n), view_angles, beam="fan",
               detector_spacing=1.0), image)


def test_cone_fdk_shepp_logan_rmse_is_small():
    """Portable cone-beam FDK keeps the native RMSE and amplitude bounds."""
    n, views, bins, spacing, sid, sdd = 32, 180, 128, 1.0, 300.0, 450.0
    volume = torch.tensor(shepp_logan_3d(n), dtype=torch.float32)
    trajectory = circular_trajectory_3d(views, sid, sdd, end_angle=math.tau,
                                        device="cpu", dtype=torch.float32)
    operator = Projector(trajectory, (n, n, n), (bins, bins), beam="cone",
                         detector_spacing=spacing, backend="torch")
    sinogram = operator.project(volume)
    view_angles = torch.linspace(0.0, math.tau, views + 1)[:-1]
    _check(fdk(sinogram, trajectory, (n, n, n), view_angles,
               detector_spacing=spacing), volume)


def _short_scan(short_scan):
    """Fan-beam short scan of pi + fan angle, with and without Parker."""
    n, views, bins, pitch, sid, sdd = 80, 180, 112, 1.0, 500.0, 800.0
    image = torch.tensor(shepp_logan_2d(n), dtype=torch.float32)
    gamma_max = math.atan((bins - 1) * pitch / 2 / sdd)
    # The angles drop their last sample, so stretch the requested span to keep
    # the realised one above pi + 2 * gamma_max.
    span = (math.pi + 2 * gamma_max + 1e-3) * views / (views - 1)
    trajectory = circular_trajectory_2d_fan(views, sid, sdd, end_angle=span,
                                            device="cpu", dtype=torch.float32)
    operator = Projector(trajectory, (n, n), (bins,), beam="fan",
                         detector_spacing=pitch, backend="torch")
    sinogram = operator.project(image)
    view_angles = torch.linspace(0.0, span, views + 1)[:-1]
    reco = fbp(sinogram, trajectory, (n, n), view_angles, beam="fan",
               detector_spacing=pitch, short_scan=short_scan)
    return reco, image


def test_parker_short_scan_keeps_the_signed_amplitude_floor():
    """Parker weighting is what lets a short scan stay inside the amplitude floor.

    ``tests/test_weights.py`` checks ``parker_weights`` as a unit and the
    native suite checks that ``fbp`` composes it before filtering, but nothing
    checks that it actually fixes the redundant-ray artefact. Without it the
    short scan undershoots to about -0.54; with it the minimum is about -0.10,
    inside the ``FLOOR_BOUND`` the native accuracy tests use.
    """
    plain, truth = _short_scan(short_scan=False)
    parker, _ = _short_scan(short_scan=True)
    assert torch.isfinite(parker).all()
    assert plain.min().item() < FLOOR_BOUND, \
        f"short scan without Parker should break the floor: {plain.min().item():.5f}"
    assert parker.min().item() > FLOOR_BOUND, \
        f"Parker left the reconstruction below the floor: {parker.min().item():.5f}"
    plain_rmse = torch.sqrt(torch.mean((plain - truth) ** 2)).item()
    parker_rmse = torch.sqrt(torch.mean((parker - truth) ** 2)).item()
    assert parker_rmse < plain_rmse, f"Parker did not improve RMSE: {parker_rmse} vs {plain_rmse}"
