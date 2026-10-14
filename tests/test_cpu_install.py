"""Public CPU APIs also work when a plain wheel has no CUDA runtime available."""

import importlib.metadata
import math
import os
from pathlib import Path
import sysconfig

import pytest
import torch


def _assert_clean_cpu_install(diffct):
    assert torch.version.cuda is None and torch.version.hip is None
    assert not torch.cuda.is_available()
    site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
    installed = Path(diffct.__file__).resolve()
    assert installed.is_relative_to(site_packages), installed
    assert diffct.__version__ == importlib.metadata.version("diffct")
    runtime_packages = [
        distribution.metadata["Name"]
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"].lower().replace("_", "-").startswith("nvidia-cuda-runtime")
    ]
    assert not runtime_packages, runtime_packages
    runtime_libraries = list(site_packages.rglob("libcudart.so*"))
    assert not runtime_libraries, runtime_libraries
    loaded = [line for line in Path("/proc/self/maps").read_text().splitlines() if "libcudart" in line]
    assert not loaded, loaded


@pytest.fixture(scope="module", autouse=True)
def _clean_install_boundary():
    # Normal test hosts may have CUDA. The isolated installation job opts in.
    clean_install = os.environ.get("DIFFCT_REQUIRE_CLEAN_CPU") == "1"
    if clean_install:
        import diffct
        _assert_clean_cpu_install(diffct)
    yield
    if clean_install:
        _assert_clean_cpu_install(diffct)


def _axis_trajectory():
    return tuple(torch.tensor([vector], dtype=torch.float64)
                 for vector in [(1.0, 0.0), (8.0, 0.0), (0.0, 1.0)])


def test_cpu_projection_adjoint_and_gradient_match_seven_cell_chord():
    import diffct

    operator = diffct.Projector(
        _axis_trajectory(), (1, 7), (1,), beam="parallel",
        backend="torch", precision="float64", distributed=False,
    )
    image = torch.ones((1, 7), dtype=torch.float64, requires_grad=True)
    sinogram = torch.tensor([[2.0]], dtype=torch.float64)
    projected = operator.project(image)
    adjoint = operator.backproject(sinogram)

    assert projected.dtype == adjoint.dtype == torch.float64
    assert projected.device.type == adjoint.device.type == "cpu"
    # A central x ray crosses all seven unit-width cells. Its adjoint puts the
    # sinogram value in each cell; no production traversal supplies this oracle.
    torch.testing.assert_close(projected, torch.tensor([[7.0]], dtype=torch.float64))
    torch.testing.assert_close(adjoint, torch.full_like(image, 2.0))
    torch.testing.assert_close((projected * sinogram).sum(), (image * adjoint).sum())
    projected.sum().backward()
    assert image.grad is not None and torch.count_nonzero(image.grad).item() == 7
    torch.testing.assert_close(image.grad, torch.ones_like(image))


def test_cpu_fbp_and_sinogram_gradient_are_finite_and_nonzero():
    import diffct

    angles = torch.arange(4, dtype=torch.float64) * (math.pi / 4)
    directions = torch.stack((angles.cos(), angles.sin()), dim=1)
    trajectory = directions, 8 * directions, torch.stack((-angles.sin(), angles.cos()), dim=1)
    image = torch.zeros((5, 5), dtype=torch.float64)
    image[2, 2] = 1.0
    operator = diffct.Projector(
        trajectory, image.shape, (7,), beam="parallel", backend="torch", precision="float64",
    )
    sinogram = operator.project(image).detach().requires_grad_()
    reconstruction = diffct.fbp(
        sinogram, trajectory, image.shape, angles, backend="torch", precision="float64",
    )
    assert reconstruction.shape == image.shape
    assert reconstruction.dtype == torch.float64 and reconstruction.device.type == "cpu"
    assert torch.isfinite(reconstruction).all() and torch.count_nonzero(reconstruction).item() > 0
    gradient, = torch.autograd.grad(reconstruction.square().sum(), sinogram)
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient).item() > 0


def test_explicit_native_backend_reports_unavailable_nvidia_cuda(monkeypatch):
    import diffct

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        diffct.Projector(_axis_trajectory(), (1, 7), (1,), beam="parallel", backend="numba")
