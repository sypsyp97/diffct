"""Native cylindrical detector: projection, matched adjoint and gradients.

    python examples/curved_detector.py

Uses a circular cone-beam trajectory and one ray per detector pixel. This
example uses the native curved operators directly; it does not run FDK.
"""

import torch

from diffct import Projector, circular_trajectory_3d
from _common import shepp_logan_3d


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("diffct needs a CUDA GPU")
    torch.manual_seed(0)
    trajectory = circular_trajectory_3d(90, sid=80.0, sdd=128.0, device="cpu")
    # Radius equals SDD: this cylinder is initially centred on each view's source.
    radius = torch.tensor(128.0, dtype=torch.float64, requires_grad=True)

    def cylinder(u, v):
        u, v = u.to(radius.device), v.to(radius.device)
        angle = u / radius
        # The generated detector normal points away from the source.
        return torch.stack((radius * angle.sin(), v,
                            radius * (angle.cos() - 1)), dim=-1)

    operator = Projector(trajectory, (32, 32, 32), (96, 64),
                         detector_spacing=(0.8, 1.0), detector_surface=cylinder)
    volume = torch.from_numpy(shepp_logan_3d(32)).cuda().requires_grad_()
    sinogram = operator.project(volume)
    y = torch.rand_like(sinogram)
    adjoint = operator.backproject(y)
    lhs = torch.sum(sinogram.double() * y.double())
    rhs = torch.sum(volume.double() * adjoint.double())
    torch.testing.assert_close(lhs, rhs, rtol=3e-5, atol=3e-5)

    # d(sum(A x))/dx = A^T 1, with the same native curved geometry.
    (volume_gradient,) = torch.autograd.grad(sinogram.sum(), volume, retain_graph=True)
    expected = operator.backproject(torch.ones_like(sinogram))
    torch.testing.assert_close(volume_gradient, expected, rtol=3e-5, atol=3e-5)
    (radius_gradient,) = torch.autograd.grad(sinogram.square().sum(), radius)
    assert torch.isfinite(radius_gradient) and radius_gradient != 0

    print(f"volume {tuple(volume.shape)} -> curved sinogram {tuple(sinogram.shape)}")
    print(f"matched adjoint {tuple(adjoint.shape)}, "
          f"relative mismatch {(abs(lhs - rhs) / abs(lhs)).item():.3e}")
    print(f"volume gradient matches native A^T 1; "
          f"radius gradient {radius_gradient.item():.6g}")


if __name__ == "__main__":
    main()
