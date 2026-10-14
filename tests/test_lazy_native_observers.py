"""Cold native dispatch keeps existing launch observers scoped and reusable."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


def _run_cold(mode):
    repository = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(repository), env.get("PYTHONPATH", "")))
    env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "1"
    env.pop("NUMBA_ENABLE_CUDASIM", None)
    if mode == "cudasim":
        env["NUMBA_ENABLE_CUDASIM"] = "1"
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), mode],
        cwd=repository, env=env, capture_output=True, text=True, timeout=120,
    )
    assert child.returncode == 0, child.stdout + child.stderr


def _axis_trajectory(device="cpu"):
    return tuple(torch.tensor([vector], device=device)
                 for vector in [(1.0, 0.0), (8.0, 0.0), (0.0, 1.0)])


def _assert_cuda_unloaded():
    assert not any(name == "numba.cuda" or name.startswith("numba.cuda.")
                   for name in sys.modules)
    assert "diffct.kernels.parallel_beam" not in sys.modules


def _cpu_worker():
    import diffct

    _assert_cuda_unloaded()
    projector = diffct.Projector(
        _axis_trajectory(), (1, 7), (1,), beam="parallel", backend="torch",
    )
    image = torch.ones((1, 7), requires_grad=True)
    projected = projector.project(image)
    adjoint = projector.backproject(torch.tensor([[2.0]]))
    # A central x ray crosses seven unit-width cells; the adjoint gives each 2.
    torch.testing.assert_close(projected, torch.tensor([[7.0]]), rtol=0, atol=0)
    torch.testing.assert_close(adjoint, torch.full_like(image, 2.0), rtol=0, atol=0)
    projected.sum().backward()
    torch.testing.assert_close(image.grad, torch.ones_like(image), rtol=0, atol=0)
    _assert_cuda_unloaded()


def _cudasim_worker():
    import diffct.projectors as projectors
    from diffct import kernels
    from tests.test_chunked_projector import _Launches

    assert "diffct.kernels.parallel_beam" not in sys.modules
    name = "_parallel_2d_forward_kernel"
    # Real simulator dispatchers exercise export/patch lifetimes without CUDA.
    # No substitute kernel or simulated public Projector execution is involved.
    with _Launches():
        assert getattr(projectors, name) is getattr(kernels, name)
    assert getattr(projectors, name) is getattr(kernels, name), \
        "projector export retained the exited launch observer"
    with _Launches():
        assert getattr(projectors, name) is getattr(kernels, name), \
            "projector export bypassed the second launch observer"
    assert getattr(projectors, name) is getattr(kernels, name)


def _native_worker(mode):
    import diffct

    _assert_cuda_unloaded()
    if mode == "scan":
        from tests.test_chunked_projector import _Launches as Observer
    else:
        from tests.test_projector_api import _KernelLaunchTracer as Observer
    assert "diffct.kernels.parallel_beam" not in sys.modules
    projector = diffct.Projector(
        _axis_trajectory("cuda:0"), (1, 7), (1,), beam="parallel",
        backend="numba", devices=["cuda:0"], distributed=False,
    )
    image = torch.ones((1, 7), device="cuda:0")
    sinogram = torch.tensor([[2.0]], device="cuda:0")

    def launch():
        projected = projector.project(image)
        adjoint = projector.backproject(sinogram)
        torch.cuda.synchronize()
        torch.testing.assert_close(projected, torch.tensor([[7.0]], device=image.device),
                                   rtol=0, atol=0)
        torch.testing.assert_close(adjoint, torch.full_like(image, 2.0), rtol=0, atol=0)

    with Observer() as first:
        launch()
    first_count = len(first.records)
    with Observer() as second:
        launch()
    first_after_second = len(first.records)
    second_count = len(second.records)
    launch()
    counts = (first_count, first_after_second, second_count,
              len(first.records), len(second.records))
    print(f"observer={mode}, launches={counts}", flush=True)
    assert counts == (2, 2, 2, 2, 2), \
        "each observer must see its own native launches and stop recording on exit"
    if mode == "scan":
        kinds = [(record[1], record[2]) for record in second.records]
    else:
        kinds = [(record["beam"], record["kernel_kind"]) for record in second.records]
    assert kinds == [("parallel", "forward"), ("parallel", "backward")]


def test_cpu_import_and_projection_do_not_load_cuda():
    _run_cold("cpu")


def test_cudasim_observer_exports_restore_and_accept_next_observer():
    _run_cold("cudasim")


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None,
                    reason="NVIDIA CUDA is required")
@pytest.mark.parametrize("observer", ["scan", "direct"])
def test_cold_native_launch_observers_are_scoped_and_reusable(observer):
    _run_cold(observer)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "cpu":
        _cpu_worker()
    elif mode == "cudasim":
        _cudasim_worker()
    else:
        _native_worker(mode)
