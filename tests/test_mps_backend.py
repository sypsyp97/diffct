"""The torch backend runs on Apple Silicon GPUs (MPS) with float32 working precision."""

import pytest
import torch

from diffct import Projector, SolverResult, fbp, helical_view_weights
from diffct.geometry import (circular_trajectory_2d_parallel, circular_trajectory_3d,
                             circular_trajectory_3d_parallel)

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS device required")


@mps
def test_mps_project_matches_cpu_parallel():
    torch.manual_seed(0)
    volume = torch.rand(64, 64)
    def run(device):
        trajectory = circular_trajectory_2d_parallel(36, device=device, dtype=torch.float32)
        projector = Projector(trajectory, volume_shape=(64, 64), detector_shape=(64,),
                              beam="parallel", backend="torch")
        sinogram = projector(volume.to(device))
        return sinogram.detach().cpu()
    cpu, mps_sino = run("cpu"), run("mps")
    assert torch.allclose(cpu, mps_sino, rtol=1e-4, atol=1e-4)


@mps
def test_mps_adjoint_inner_product():
    torch.manual_seed(1)
    trajectory = circular_trajectory_2d_parallel(48, device="mps", dtype=torch.float32)
    projector = Projector(trajectory, volume_shape=(64, 64), detector_shape=(96,),
                          beam="parallel", backend="torch")
    volume = torch.rand(64, 64, device="mps")
    sinogram = projector(volume)
    probe = torch.rand_like(sinogram)
    lhs = (sinogram * probe).sum()
    rhs = (volume * projector.backproject(probe)).sum()
    assert torch.allclose(lhs, rhs, rtol=1e-4, atol=1e-2)


@mps
def test_mps_projection_autograd():
    trajectory = circular_trajectory_2d_parallel(24, device="mps", dtype=torch.float32)
    projector = Projector(trajectory, volume_shape=(32, 32), detector_shape=(48,),
                          beam="parallel", backend="torch")
    volume = torch.rand(32, 32, device="mps", requires_grad=True)
    projector(volume).sum().backward()
    assert volume.grad is not None and torch.isfinite(volume.grad).all()


@mps
def test_mps_cone_geometry_gradients():
    src, det, u, v = circular_trajectory_3d(30, sid=800.0, sdd=1200.0,
                                            device="mps", dtype=torch.float32)
    src = src.detach().clone().requires_grad_(True)
    projector = Projector((src, det, u, v), volume_shape=(48, 48, 48),
                          detector_shape=(64, 64), backend="torch")
    volume = torch.rand(48, 48, 48, device="mps")
    projector(volume).sum().backward()
    assert src.grad is not None and torch.isfinite(src.grad).all()


@mps
def test_mps_fbp_matches_cpu():
    torch.manual_seed(2)
    volume = torch.rand(64, 64)
    angles = torch.linspace(0, torch.pi, 91)[:-1]
    def run(device):
        trajectory = circular_trajectory_2d_parallel(90, device=device, dtype=torch.float32)
        projector = Projector(trajectory, volume_shape=(64, 64), detector_shape=(96,),
                              beam="parallel", backend="torch")
        sinogram = projector(volume.to(device))
        return fbp(sinogram, trajectory, (64, 64), angles.to(device)).float().cpu()
    cpu, mps_rec = run("cpu"), run("mps")
    assert torch.allclose(cpu, mps_rec, rtol=1e-3, atol=1e-4)


@mps
def test_mps_solvers_run():
    from diffct import cgls, sirt, tv_reconstruction
    trajectory = circular_trajectory_2d_parallel(24, device="mps", dtype=torch.float32)
    projector = Projector(trajectory, volume_shape=(32, 32), detector_shape=(48,),
                          beam="parallel", backend="torch")
    volume = torch.rand(32, 32, device="mps")
    sinogram = projector(volume)
    for solver in (cgls, sirt, tv_reconstruction):
        result = solver(projector, sinogram, iterations=2)
        image = result.volume if isinstance(result, SolverResult) else result
        assert torch.isfinite(image).all()


@mps
def test_mps_detector_surface():
    from diffct.surfaces import ParameterizedSurface
    trajectory = circular_trajectory_3d(16, sid=800.0, sdd=1200.0,
                                        device="mps", dtype=torch.float32)
    gain = torch.ones(1, device="mps")
    surface = ParameterizedSurface(
        lambda u, v, ids, g: torch.stack((u, v, g.expand_as(u) * u), -1), (gain,))
    projector = Projector(trajectory, volume_shape=(32, 32, 32), detector_shape=(48, 48),
                          backend="torch", detector_surface=surface)
    sinogram = projector(torch.rand(32, 32, 32, device="mps"))
    assert torch.isfinite(sinogram).all()


@mps
def test_mps_streaming_stores():
    from diffct import TensorStore
    trajectory = circular_trajectory_2d_parallel(24, device="mps", dtype=torch.float32)
    projector = Projector(trajectory, volume_shape=(48, 48), detector_shape=(64,),
                          beam="parallel", backend="torch")
    volume = torch.rand(48, 48, device="mps")
    sinogram = torch.empty(24, 64, device="mps", dtype=torch.float32)
    projector.project_into(TensorStore(volume), TensorStore(sinogram))
    assert torch.isfinite(sinogram).all()
    image = torch.empty(48, 48, device="mps", dtype=torch.float32)
    projector.backproject_into(TensorStore(sinogram), TensorStore(image))
    assert torch.isfinite(image).all()


@mps
def test_mps_parallel3d_matches_cpu():
    torch.manual_seed(3)
    volume = torch.rand(24, 24, 24)
    def run(device):
        trajectory = circular_trajectory_3d_parallel(12, detector_distance=800.0,
                                                     device=device, dtype=torch.float32)
        projector = Projector(trajectory, volume_shape=(24, 24, 24), detector_shape=(32, 32),
                              beam="parallel3d", backend="torch")
        return projector(volume.to(device)).detach().cpu()
    torch.testing.assert_close(run("cpu"), run("mps"), rtol=1e-4, atol=1e-4)


@mps
def test_mps_transform_trajectory_rotation_matches_cpu():
    from diffct import transform_trajectory
    def run(device):
        trajectory = circular_trajectory_3d(8, sid=800.0, sdd=1200.0,
                                            device=device, dtype=torch.float32)
        angle = torch.tensor(0.3, device=device)
        c, s = torch.cos(angle), torch.sin(angle)
        z, o = torch.zeros_like(angle), torch.ones_like(angle)
        rotation = torch.stack((c, -s, z, s, c, z, z, z, o)).reshape(3, 3)
        return tuple(value.detach().cpu() for value in
                     transform_trajectory(trajectory, rotation=rotation))
    cpu, transformed = run("cpu"), run("mps")
    for expected, actual in zip(cpu, transformed):
        torch.testing.assert_close(expected, actual, rtol=1e-4, atol=1e-4)
    # A reflection must still be rejected by the proper-rotation check.
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0, ], device="mps"))
    with pytest.raises(ValueError, match="orthogonal"):
        transform_trajectory(circular_trajectory_3d(8, sid=800.0, sdd=1200.0,
                                                    device="mps", dtype=torch.float32),
                             rotation=reflection)


@mps
def test_mps_surface_with_cpu_trajectory_matches_cpu():
    from diffct.surfaces import ParameterizedSurface
    torch.manual_seed(4)
    volume = torch.rand(24, 24, 24)
    gain = torch.ones(1, device="cpu")
    surface = ParameterizedSurface(
        lambda u, v, ids, g: torch.stack((u, v, g.expand_as(u) * u), -1), (gain,))
    def run(device):
        trajectory = circular_trajectory_3d(8, sid=800.0, sdd=1200.0,
                                            device="cpu", dtype=torch.float32)
        projector = Projector(trajectory, volume_shape=(24, 24, 24), detector_shape=(32, 32),
                              backend="torch", detector_surface=surface)
        return projector(volume.to(device)).detach().cpu()
    torch.testing.assert_close(run("cpu"), run("mps"), rtol=1e-4, atol=1e-4)


@mps
def test_mps_solver_norm_scaling_extremes():
    from diffct import cgls
    torch.manual_seed(5)
    def solve(scale, spacing=1.0):
        trajectory = circular_trajectory_2d_parallel(16, device="mps", dtype=torch.float32)
        projector = Projector(trajectory, volume_shape=(32, 32), detector_shape=(48,),
                              beam="parallel", backend="torch",
                              voxel_spacing=spacing, detector_spacing=spacing)
        truth = (torch.rand(32, 32) > 0.7).float().to("mps")
        sinogram = projector(truth) * scale
        result = cgls(projector, sinogram, iterations=5)
        return truth, result

    truth, large = solve(1e19)
    assert large.iterations == 5 and not large.converged
    assert large.residual_history[0] > large.residual_history[-1] > 0.0
    truth, tiny = solve(1.0, spacing=1e-8)
    assert tiny.iterations == 5 and "breakdown" not in tiny.reason.lower()
    assert tiny.residual_history[0] > tiny.residual_history[-1] > 0.0


@mps
def test_mps_cpu_fp64_surface_forward_and_backward():
    from diffct.surfaces import ParameterizedSurface
    torch.manual_seed(6)
    volume = torch.rand(8, 8, 8)
    surface = ParameterizedSurface(
        lambda u, v, ids: torch.stack((u, v, torch.zeros_like(u)), -1))
    def run(device):
        trajectory = circular_trajectory_3d(4, sid=800.0, sdd=1200.0,
                                            device="cpu", dtype=torch.float64)
        projector = Projector(trajectory, volume_shape=(8, 8, 8), detector_shape=(12, 12),
                              backend="torch", detector_surface=surface)
        leaf = volume.to(device).clone().requires_grad_()
        sinogram = projector(leaf)
        sinogram.sum().backward()
        return sinogram.detach().cpu(), leaf.grad.detach().cpu()
    cpu_sino, cpu_grad = run("cpu")
    mps_sino, mps_grad = run("mps")
    torch.testing.assert_close(cpu_sino, mps_sino, rtol=2e-3, atol=2e-2)
    torch.testing.assert_close(cpu_grad, mps_grad, rtol=2e-3, atol=2e-2)


@mps
def test_mps_learnable_cpu_fp64_trajectory_stays_consistent():
    # A learnable FP64 trajectory on the host projecting an MPS volume keeps
    # working end to end; its forward matches the learnable FP32 trajectory
    # on the same device (both clamp to the MPS working dtype), and the
    # geometry gradients stay finite and non-zero.
    def forward(leaves):
        torch.manual_seed(7)
        projector = Projector(leaves, volume_shape=(8, 8, 8), detector_shape=(12, 12),
                              backend="torch", precision="fast")
        sinogram = projector(torch.rand(8, 8, 8, device="mps"))
        sinogram.sum().backward()
        return sinogram.detach().cpu(), [leaf.grad.cpu() for leaf in leaves]

    trajectory = circular_trajectory_3d(6, sid=800.0, sdd=1200.0,
                                        device="cpu", dtype=torch.float64)
    leaves64 = tuple(c.clone().requires_grad_() for c in trajectory)
    leaves32 = tuple(c.float().clone().requires_grad_() for c in trajectory)
    sinogram64, grads64 = forward(leaves64)
    sinogram32, _ = forward(leaves32)
    assert (sinogram64 - sinogram32).norm() / sinogram32.norm() < 1e-4
    for gradient in grads64:
        assert torch.isfinite(gradient).all() and gradient.abs().max() > 0


@mps
def test_mps_norm_result_keeps_the_float64_range():
    from diffct._solver_state import squared_norm
    class Solo:
        world_size = 1
    value = torch.full((100,), 2.0, device="mps", dtype=torch.float32)
    result = squared_norm(Solo(), value, "rays", return_tensor=True)
    assert isinstance(result, torch.Tensor) and result.device.type == "cpu"
    assert result.dtype == torch.float64 and result.ndim == 0
    assert abs(result.item() - 400.0) < 1e-2


@mps
def test_mps_norms_survive_a_float64_default_dtype():
    from diffct._solver_state import squared_norm
    class Solo:
        world_size = 1
    data = torch.full((100,), 2.0, device="mps", dtype=torch.float32)
    torch.set_default_dtype(torch.float64)
    try:
        assert abs(float(squared_norm(Solo(), data, "rays")) - 400.0) < 1e-2
    finally:
        torch.set_default_dtype(torch.float32)


def test_mps_float64_precision_rejected_with_guidance():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device required")
    trajectory = circular_trajectory_3d_parallel(4, device="mps", dtype=torch.float32)
    with pytest.raises(ValueError, match="fast' or 'float32'"):
        Projector(trajectory, volume_shape=(8, 8, 8), detector_shape=(12, 12),
                  beam="parallel3d", backend="torch", precision="float64")


def test_tangent_parallel3d_frame_rejected_in_float64():
    direction = torch.tensor([[1.0, 0.0, 5e-16]])
    axis = torch.tensor([[1.0, 0.0, 0.0]])
    vertical = torch.tensor([[0.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="tangent"):
        Projector((direction, direction, axis, vertical), volume_shape=(4, 4, 4),
                  detector_shape=(4, 4), beam="parallel3d", backend="torch")


@mps
def test_mps_helical_view_weights_match_cpu():
    from diffct import helical_view_weights
    angles = torch.linspace(0, 4 * torch.pi, 41)[:-1]
    source_z = torch.linspace(-25.0, 25.0, 40)
    slice_z = torch.linspace(-10.0, 10.0, 20)
    cpu = helical_view_weights(angles.clone(), source_z.clone(), slice_z.clone())
    device = helical_view_weights(angles.to("mps"), source_z.to("mps"), slice_z.to("mps"))
    assert device.device.type == "mps"
    torch.testing.assert_close(cpu, device.cpu(), rtol=0.0, atol=0.0)
    torch.testing.assert_close(device.sum().cpu(), torch.tensor(torch.pi * 20.0), rtol=1e-5, atol=1e-4)


def test_helical_weights_list_and_tensor_agree():
    # Large source heights lose precision when a list is converted through
    # the default dtype. Both input forms must select the same views and
    # produce identical weights; the check is device-agnostic and runs on
    # every CI runner.
    angles = torch.linspace(0, 4 * torch.pi, 42)[:-1]
    source_z_tensor = 1e8 + torch.arange(41, dtype=torch.float64)
    source_z_list = [1e8 + i for i in range(41)]
    slice_z = [100000020.0]
    from_tensor = helical_view_weights(angles.clone(), source_z_tensor.clone(), slice_z)
    from_list = helical_view_weights(angles.clone(), list(source_z_list), slice_z)
    selected = (from_tensor.sum(dim=0) > 0).nonzero().flatten()
    assert selected.min() == 10 and selected.max() == 30
    torch.testing.assert_close(from_tensor, from_list, rtol=0.0, atol=0.0)


def test_mps_precision_guard_fires_before_the_reference_path():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device required")
    trajectory = circular_trajectory_3d(4, sid=800.0, sdd=1200.0,
                                        device="cpu", dtype=torch.float64)
    projector = Projector(trajectory, volume_shape=(8, 8, 8), detector_shape=(12, 12),
                          backend="torch", precision="float64")
    for call in (lambda: projector.project(torch.rand(8, 8, 8, device="mps")),
                 lambda: projector.backproject(torch.rand(4, 12, 12, device="mps"))):
        with pytest.raises(ValueError, match="fast' or 'float32'"):
            call()
