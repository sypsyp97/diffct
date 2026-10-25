"""The torch backend runs on Apple Silicon GPUs (MPS) with float32 working precision."""

import pytest
import torch

from diffct import Projector, SolverResult, fbp
from diffct.geometry import circular_trajectory_2d_parallel, circular_trajectory_3d

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
