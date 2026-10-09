"""Custom trajectory orientation retains the source-path gradient."""

import pytest
import torch

from diffct import Projector, custom_trajectory_3d


@pytest.mark.parametrize("n_views", [1, 3])
@pytest.mark.parametrize("axis", [2, 3], ids=["detector_u", "detector_v"])
def test_custom_trajectory_detector_axes_gradcheck(n_views, axis):
    source = torch.tensor([2.0, -3.0, 1.0], dtype=torch.float64, requires_grad=True)

    def orientation(position):
        return custom_trajectory_3d(
            n_views, 5.0, 9.0, lambda angles, sid: position.expand(n_views, -1),
            device="cpu", dtype=torch.float64,
        )[axis]

    assert orientation(source).requires_grad
    assert torch.autograd.gradcheck(orientation, (source,))


def test_custom_trajectory_axis_aligned_source_has_valid_frame_and_backward():
    source = torch.tensor([[0.0, 0.0, 3.0], [2.0, -3.0, 1.0]],
                          dtype=torch.float64, requires_grad=True)
    _, detector, u, v = custom_trajectory_3d(
        2, 5.0, 9.0, lambda angles, sid: source, device="cpu", dtype=torch.float64,
    )
    torch.testing.assert_close(u[0], torch.tensor([1.0, 0.0, 0.0], dtype=u.dtype))
    torch.testing.assert_close(u.norm(dim=1), torch.ones(2, dtype=u.dtype))
    torch.testing.assert_close(v.norm(dim=1), torch.ones(2, dtype=v.dtype))
    torch.testing.assert_close((u * v).sum(dim=1), torch.zeros(2, dtype=u.dtype))
    (detector.sum() + u.sum() + v.sum()).backward()
    assert torch.isfinite(source.grad).all()


@pytest.mark.cuda
def test_custom_source_path_backward_through_projector():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    source = torch.tensor([1.2, 5.1, 1.4], device="cuda", requires_grad=True)
    trajectory = custom_trajectory_3d(
        1, 6.0, 9.0, lambda angles, sid: source[None, :], device="cuda",
    )
    projector = Projector(trajectory, (3, 4, 5), (4, 3), detector_spacing=0.7)
    volume = torch.linspace(0.1, 1.0, 60, device="cuda").reshape(3, 4, 5)
    projection = projector.project(volume)
    gradient, = torch.autograd.grad(projection.square().sum(), source)
    assert torch.isfinite(gradient).all()
    assert gradient.abs().max() > 1e-6
