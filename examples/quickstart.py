"""The Projector API in one file: parallel, fan and cone beams.

For each beam this projects a phantom, backprojects, checks the adjoint
identity <A x, y> = <x, A^T y>, and differentiates a loss with respect to the
volume and to the trajectory.

    python examples/quickstart.py
"""

import torch

from diffct import (
    Projector,
    circular_trajectory_2d_fan,
    circular_trajectory_2d_parallel,
    spiral_trajectory_3d,
)
from _common import shepp_logan_3d


def demo(name, operator, volume):
    sinogram = operator.project(volume)                 # A x
    y = torch.rand_like(sinogram)
    adjoint = operator.backproject(y)                   # A^T y
    lhs = torch.sum(sinogram.double() * y.double())
    rhs = torch.sum(volume.double() * adjoint.double())
    print(f"{name:8s} volume {tuple(volume.shape)} -> sinogram {tuple(sinogram.shape)}, "
          f"adjoint mismatch {abs(lhs - rhs).item() / abs(lhs).item():.1e}")

    # Gradient with respect to the volume: d/dx 0.5 ||A x - y||^2 = A^T (A x - y).
    x = volume.clone().requires_grad_()
    loss = 0.5 * (operator.project(x) - y).square().sum()
    (grad,) = torch.autograd.grad(loss, x)
    print(f"{'':8s} volume gradient norm {grad.norm().item():.3e}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("diffct needs a CUDA GPU")
    phantom = torch.from_numpy(shepp_logan_3d(128)).cuda()

    # 2D beams use (H, W) images. Trajectories are per-view tensors of shape (views, 2).
    image = phantom[64]
    demo("parallel", Projector(circular_trajectory_2d_parallel(180, device="cpu"),
                               (128, 128), 192, beam="parallel"), image)
    demo("fan", Projector(circular_trajectory_2d_fan(360, sid=300.0, sdd=500.0, device="cpu"),
                          (128, 128), 256, beam="fan"), image)

    # 3D cone beam: (D, H, W) volume, (views, U, V) sinogram, any per-view geometry.
    trajectory = spiral_trajectory_3d(360, sid=320.0, sdd=512.0, z_range=40.0, n_turns=1.0, device="cpu")
    demo("cone", Projector(trajectory, (128, 128, 128), (384, 256), detector_spacing=0.8), phantom)

    # Geometry gradients: make trajectory tensors require gradients.
    source, det_center, det_u, det_v = (t.clone() for t in trajectory)
    source.requires_grad_()
    operator = Projector((source, det_center, det_u, det_v), (128, 128, 128), (384, 256),
                         detector_spacing=0.8)
    operator.project(phantom).sum().backward()
    print(f"geometry gradient: d(sum of projections)/d(source) has shape {tuple(source.grad.shape)}, "
          f"norm {source.grad.norm().item():.3e}")


if __name__ == "__main__":
    main()
