"""The torch backend runs on Apple Silicon GPUs (MPS) with float32 working precision."""

import pytest
import torch

from diffct import Projector, SolverResult, fbp
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
