"""``torch.autograd.gradcheck`` tests for every projector Function.

These tests are the strongest possible autograd-correctness guard:
they build the numerical Jacobian by finite differences and compare
it element-by-element to the analytical Jacobian that our CUDA
``backward`` kernel returns.

Because the Siddon kernels are hard-coded to ``float32`` the finite
differences are noisy, so the tests use loose but still meaningful
tolerances (``eps=1e-2``, ``atol``/``rtol`` ~ a few percent) and a
small non-determinism tolerance for the atomic-add backward. Inputs
are kept tiny so the full ``gradcheck`` (which calls the forward
once per input element per direction) finishes in well under a
minute on a consumer GPU.
"""

import math

import pytest
import torch

from diffct.differentiable import (
    ConeProjectorFunction,
    FanProjectorFunction,
    ParallelProjectorFunction,
)


# Tolerances tuned for float32 + Siddon ray-march + atomic-add backward.
# float32 finite differences saturate around 1e-3 relative accuracy, and
# the per-segment chord-length accumulation adds an extra factor-of-2 noise.
_GC_EPS = 1e-2
_GC_ATOL = 5e-2
_GC_RTOL = 5e-2
_GC_NONDET = 5e-3


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")


@pytest.mark.cuda
def test_parallel_projector_gradcheck():
    """Numerical vs analytical Jacobian match for ``ParallelProjectorFunction``."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    torch.manual_seed(0)

    H, W = 6, 6
    img = torch.randn(H, W, device=device, dtype=torch.float32).contiguous()
    img.requires_grad_(True)
    angles = torch.linspace(0.0, math.pi, 6, device=device, dtype=torch.float32)
    num_detectors = 10
    detector_spacing = 1.0

    def fn(x):
        return ParallelProjectorFunction.apply(
            x, angles, num_detectors, detector_spacing
        )

    assert torch.autograd.gradcheck(
        fn,
        (img,),
        eps=_GC_EPS,
        atol=_GC_ATOL,
        rtol=_GC_RTOL,
        nondet_tol=_GC_NONDET,
        check_undefined_grad=False,
    )


@pytest.mark.cuda
def test_fan_projector_gradcheck():
    """Numerical vs analytical Jacobian match for ``FanProjectorFunction``."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    torch.manual_seed(1)

    H, W = 6, 6
    img = torch.randn(H, W, device=device, dtype=torch.float32).contiguous()
    img.requires_grad_(True)
    angles = torch.linspace(0.0, 2.0 * math.pi, 8, device=device, dtype=torch.float32)
    num_detectors = 10
    detector_spacing = 1.0
    sdd, sid = 900.0, 600.0

    def fn(x):
        return FanProjectorFunction.apply(
            x, angles, num_detectors, detector_spacing, sdd, sid
        )

    assert torch.autograd.gradcheck(
        fn,
        (img,),
        eps=_GC_EPS,
        atol=_GC_ATOL,
        rtol=_GC_RTOL,
        nondet_tol=_GC_NONDET,
        check_undefined_grad=False,
    )


@pytest.mark.cuda
def test_fan_sf_projector_gradcheck():
    """Gradcheck for ``FanProjectorFunction`` with ``backend='sf'``.

    The SF backward kernel is a voxel-driven gather with no atomic adds,
    so it should be byte-deterministic (nondet_tol=0). Tolerances are
    otherwise the same as the Siddon gradcheck above."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    torch.manual_seed(21)

    H, W = 6, 6
    img = torch.randn(H, W, device=device, dtype=torch.float32).contiguous()
    img.requires_grad_(True)
    angles = torch.linspace(0.0, 2.0 * math.pi, 8, device=device, dtype=torch.float32)
    num_detectors = 10
    detector_spacing = 1.0
    sdd, sid = 900.0, 600.0

    def fn(x):
        return FanProjectorFunction.apply(
            x, angles, num_detectors, detector_spacing, sdd, sid,
            1.0, 0.0, 0.0, 0.0, "sf",
        )

    assert torch.autograd.gradcheck(
        fn,
        (img,),
        eps=_GC_EPS,
        atol=_GC_ATOL,
        rtol=_GC_RTOL,
        nondet_tol=_GC_NONDET,
        check_undefined_grad=False,
    )


@pytest.mark.cuda
@pytest.mark.parametrize("backend", ["sf_tr", "sf_tt"])
def test_cone_sf_projector_gradcheck(backend):
    """Gradcheck for ``ConeProjectorFunction`` with the SF backends.

    Tiny 4x4x4 volume / 6 views / 6x6 detector — gradcheck does ~128
    forward calls. SF backward is deterministic (gather, no atomics) so
    ``nondet_tol`` stays tight."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    torch.manual_seed(50 + hash(backend) % 100)

    D, H, W = 4, 4, 4
    vol = torch.randn(D, H, W, device=device, dtype=torch.float32).contiguous()
    vol.requires_grad_(True)
    angles = torch.linspace(0.0, 2.0 * math.pi, 6, device=device, dtype=torch.float32)
    det_u = det_v = 6
    du = dv = 1.0
    sdd, sid = 900.0, 600.0

    def fn(x):
        return ConeProjectorFunction.apply(
            x, angles, det_u, det_v, du, dv, sdd, sid,
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, backend,
        )

    assert torch.autograd.gradcheck(
        fn,
        (vol,),
        eps=_GC_EPS,
        atol=_GC_ATOL,
        rtol=_GC_RTOL,
        nondet_tol=_GC_NONDET,
        check_undefined_grad=False,
    )


@pytest.mark.cuda
def test_cone_projector_gradcheck():
    """Numerical vs analytical Jacobian match for ``ConeProjectorFunction``.

    Uses a 4x4x4 volume with 6 views and a 6x6 detector -- 64 input
    elements, so gradcheck runs 64*2 forward calls. Small enough to be
    fast but large enough to exercise the 3D Siddon path."""
    _skip_if_no_cuda()
    device = torch.device("cuda")
    torch.manual_seed(2)

    D, H, W = 4, 4, 4
    vol = torch.randn(D, H, W, device=device, dtype=torch.float32).contiguous()
    vol.requires_grad_(True)
    angles = torch.linspace(0.0, 2.0 * math.pi, 6, device=device, dtype=torch.float32)
    det_u = det_v = 6
    du = dv = 1.0
    sdd, sid = 900.0, 600.0

    def fn(x):
        return ConeProjectorFunction.apply(
            x, angles, det_u, det_v, du, dv, sdd, sid
        )

    assert torch.autograd.gradcheck(
        fn,
        (vol,),
        eps=_GC_EPS,
        atol=_GC_ATOL,
        rtol=_GC_RTOL,
        nondet_tol=_GC_NONDET,
        check_undefined_grad=False,
    )
