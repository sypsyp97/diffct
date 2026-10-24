"""Small, asymmetric inputs for the native per-axis spacing contract.

Generate the scalar reference from unmodified main (c4ef543, diffct 2.4.3):

    NUMBA_ENABLE_CUDASIM=1 NUMBA_CUDA_USE_NVIDIA_BINDING=0 \
        .venv/bin/python -m tests._native_per_axis_spacing

The generator uses CPU storage and the production Numba CUDA kernels. It writes
tests/fixtures/native_scalar_spacing_main.npz. Never regenerate this reference
from a feature implementation to make a failed regression test pass.
"""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import inspect
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from numba.core import config

import diffct
from diffct import ParameterizedSurface, Projector
from tests.test_cone_backprojection_transpose import cuda_device


BEAMS = ("parallel", "fan", "cone", "parallel3d")
NATIVE_BEAMS = BEAMS[:3]  # parallel3d remains portable-only.
SCALAR_BEAMS = NATIVE_BEAMS
SURFACES = ("flat", "offset", "parameterized")
REFERENCE = Path(__file__).with_name("fixtures") / "native_scalar_spacing_main.npz"
NATIVE_REQUIRED = pytest.mark.skipif(
    not config.ENABLE_CUDASIM
    and (not torch.cuda.is_available() or torch.version.hip is not None),
    reason="NVIDIA CUDA or NUMBA_ENABLE_CUDASIM=1 is required",
)
REAL_CUDA_REQUIRED = pytest.mark.skipif(
    config.ENABLE_CUDASIM or not torch.cuda.is_available() or torch.version.hip is not None,
    reason="real NVIDIA CUDA is required",
)

# test_reference_backend.py::_close, used by
# test_native_cuda_and_portable_backend_match_independent_reference_with_batches.
FLAT_RTOL = FLAT_ATOL = 4e-5
# test_detector_surfaces.py::_close and
# test_curved_projection_and_adjoint_match_independent_cell_oracle.
SURFACE_RTOL, SURFACE_ATOL = 4e-4, 6e-5
# test_detector_surfaces.py::test_curve_and_each_trajectory_component_gradients_against_independent_fd.
GEOMETRY_RTOL, GEOMETRY_ATOL = 5e-3, 3e-4
# test_analytical_autograd.py::test_analytical_gather_matches_independent_reference
# and test_sinogram_gradient_is_transpose_of_actual_gather.
ANALYTICAL_RTOL, ANALYTICAL_ATOL = 2e-5, 2e-6
# test_axis_aligned_chords.py::test_central_chords_and_adjoint_with_short_final_tiles.
CHORD_RTOL, CHORD_ATOL = 3e-5, 3e-6
# test_reference_backend.py::_close for portable float64 cell-matrix comparisons.
PORTABLE_RTOL = PORTABLE_ATOL = 2e-11


@dataclass(frozen=True)
class Case:
    beam: str
    trajectory: tuple
    shape: tuple
    detector: tuple
    pitch: tuple
    spacing: object

    @property
    def projection_shape(self):
        return (len(self.trajectory[0]), *self.detector)


def case(beam, *, scalar=False, learnable=False):
    angle = torch.tensor([.231, .811, 1.431], dtype=torch.float64)
    radial = torch.stack((angle.cos(), angle.sin()), -1)
    tangent = torch.stack((-angle.sin(), angle.cos()), -1)
    offset = torch.tensor([[.173, -.239], [-.119, .281], [.317, .113]], dtype=torch.float64)
    if beam in ("cone", "parallel3d"):
        tilt = torch.tensor([.197, -.137, .283], dtype=torch.float64)
        radial = torch.cat((radial * tilt.cos()[:, None], tilt.sin()[:, None]), -1)
        tangent = torch.cat((tangent, torch.zeros(3, 1)), -1)
        vertical = torch.linalg.cross(radial, tangent, dim=-1)
        offset = torch.cat((offset, torch.tensor([[-.317], [.113], [.413]])), -1)
        geometry = ((radial, 3.7 * radial + offset, tangent, vertical) if beam == "parallel3d"
                    else (-7.1 * radial + offset, 5.3 * radial + offset, tangent, vertical))
        shape, detector, pitch, spacing = (2, 3, 4), (3, 2), (.91, .67), (.7, 1., 1.9)
    else:
        geometry = ((radial, 3.7 * radial + offset, tangent) if beam == "parallel"
                    else (-7.1 * radial + offset, 5.3 * radial + offset, tangent))
        shape, detector, pitch, spacing = (3, 4), (5,), (.91,), (.6, 1.3)
    # Both oracles receive exactly the same represented trajectory coordinates.
    geometry = tuple(value.float().double().requires_grad_(learnable) for value in geometry)
    return Case(beam, geometry, shape, detector, pitch, .75 if scalar else spacing)


def surface(kind, *, learnable=False, values=(3.7, .037)):
    if kind == "flat":
        return None, ()
    if kind == "offset":
        def offsets(u, v):
            return torch.stack((u + .137, v, .023 * u.square()), -1)
        return offsets, ()
    if kind != "parameterized":
        raise ValueError(kind)
    parameters = tuple(torch.tensor(value, dtype=torch.float64, requires_grad=learnable) for value in values)

    def sample(u, v, ids, radius, coefficient):
        sag = radius * (1 - torch.cos(u / radius))
        sag = sag + coefficient * (u.square() + .31 * u * v + .19 * v.square())
        return torch.stack((radius * torch.sin(u / radius), v, sag), -1)

    return ParameterizedSurface(sample, parameters), parameters


def operator(scan, *, backend="numba", precision="fast", detector_surface=None, **options):
    return Projector(scan.trajectory, scan.shape, scan.detector, beam=scan.beam,
                     detector_spacing=scan.pitch if len(scan.pitch) == 2 else scan.pitch[0],
                     voxel_spacing=scan.spacing, detector_surface=detector_surface,
                     backend=backend, precision=precision, distributed=False, **options)


def data(shape, *, phase=0., device="cpu", dtype=torch.float32):
    index = torch.arange(math.prod(shape), dtype=torch.float64)
    return (torch.sin(.713 * index + phase) + .2 * torch.cos(.317 * index)).reshape(shape).to(
        device=device, dtype=dtype,
    )


def chunks(scan):
    return dict(volume_chunk_shape=tuple(max(1, size - 1) for size in scan.shape),
                view_chunk_size=2, detector_chunk_shape=(2,) * len(scan.detector), schedule="spatial")


def close(actual, expected, *, kind="flat", rtol=None, atol=None):
    if rtol is None:
        rtol, atol = ((FLAT_RTOL, FLAT_ATOL) if kind == "flat" else (SURFACE_RTOL, SURFACE_ATOL))
    torch.testing.assert_close(actual.detach().cpu().double(), expected.detach().cpu().double(),
                               rtol=rtol, atol=atol)


@contextmanager
def native_only():
    """Reject portable execution and observe the unchanged native array bridge."""
    from diffct import backends
    from diffct.analytical_execution import AnalyticalExecution
    from diffct.utils import TorchCUDABridge

    def forbidden(*args, **kwargs):
        raise AssertionError("native per-axis call entered the portable backend")

    with mock.patch.object(backends.ReferenceProjector, "__init__", forbidden), \
            mock.patch.object(AnalyticalExecution, "_portable_gather", forbidden), \
            mock.patch.object(TorchCUDABridge, "tensor_to_cuda_array",
                              wraps=TorchCUDABridge.tensor_to_cuda_array) as bridge:
        yield bridge


def extend_cpu_transport(monkeypatch):
    """Extend the existing simulator fixture for analytical and streamed I/O."""
    if not config.ENABLE_CUDASIM:
        return
    from diffct import analytical, projectors, utils
    from tests.test_cone_backprojection_transpose import _SimulatorTensor

    original_run = Projector._run

    def factory(original):
        def allocate(*args, **kwargs):
            requested = kwargs.get("device")
            simulated = requested is not None and torch.device(requested).type == "cuda"
            if simulated:
                kwargs["device"] = "cpu"
            kwargs.pop("pin_memory", None)
            result = original(*args, **kwargs)
            return result.as_subclass(_SimulatorTensor) if simulated else result
        return allocate

    def run(owner, tensor, *args, **kwargs):
        # The CPU autograd engine creates plain CPU cotangents. Restore only
        # simulated device metadata before the production CUDA entry point.
        if not tensor.is_cuda:
            tensor = tensor.as_subclass(_SimulatorTensor)
        return original_run(owner, tensor, *args, **kwargs)

    stream = SimpleNamespace(device=torch.device("cuda:0"), cuda_stream=0,
                             wait_event=lambda event: None, wait_stream=lambda other: None,
                             synchronize=lambda: None)
    for module, name in ((torch, "empty"), (torch, "empty_strided"), (torch, "as_tensor"),
                         (torch.fft, "fftfreq"), (torch.fft, "rfftfreq")):
        monkeypatch.setattr(module, name, factory(getattr(module, name)))
    monkeypatch.setattr(Projector, "_run", run)
    monkeypatch.setattr(torch.cuda, "Stream", lambda *args, **kwargs: stream)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "memory_snapshot", lambda: [])
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *args: 0)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)
    monkeypatch.setattr(analytical, "_get_numba_external_stream_for", lambda *args: 0)
    # CPU autograd can strip subclass metadata inside a backward kernel call.
    # The simulator has one context; production dispatch and arithmetic remain.
    for module in (projectors, utils):
        monkeypatch.setattr(module, "_cuda_context", lambda *args: nullcontext())


@pytest.fixture
def native_device(cuda_device, monkeypatch):
    extend_cpu_transport(monkeypatch)
    return cuda_device


PUBLIC_APIS = ("Projector", "Projector.project", "Projector.backproject", "Projector.project_into",
               "Projector.backproject_into", "fbp", "fdk", "fbp_into", "fdk_into", "helical_fdk",
               "cgls", "sirt", "tv_reconstruction", "cgls_into", "ParameterizedSurface")


def signatures():
    result = {}
    for name in PUBLIC_APIS:
        target = diffct
        for part in name.split("."):
            target = getattr(target, part)
        result[name] = str(inspect.signature(target))
    return result


def scalar_outputs(scan, kind, tiled, device):
    detector, _ = surface(kind)
    native = operator(scan, detector_surface=detector, **(chunks(scan) if tiled else {}))
    image = data(scan.shape, device=device).requires_grad_()
    rays = data(scan.projection_shape, phase=.31, device=device).requires_grad_()
    projected, adjoint = native.project(image), native.backproject(rays)
    image_grad, = torch.autograd.grad((projected * rays.detach()).sum(), image)
    ray_grad, = torch.autograd.grad((adjoint * image.detach()).sum(), rays)
    return {name: value.detach().cpu().numpy().copy() for name, value in
            dict(project=projected, backproject=adjoint, volume_vjp=image_grad, sinogram_vjp=ray_grad).items()}


if __name__ == "__main__":
    if not config.ENABLE_CUDASIM:
        raise RuntimeError("scalar reference generation requires NUMBA_ENABLE_CUDASIM=1")
    if subprocess.run(["git", "diff", "--quiet", "c4ef543", "--", "diffct"], check=False).returncode:
        raise RuntimeError("reference generation requires the unmodified c4ef543 library tree")
    arrays = {"source_commit": np.array("c4ef543"), "source_version": np.array("2.4.3")}
    arrays.update({"signature__" + name: np.array(value) for name, value in signatures().items()})
    with pytest.MonkeyPatch.context() as patch:
        device = cuda_device.__wrapped__(patch)
        extend_cpu_transport(patch)
        for beam in SCALAR_BEAMS:
            for kind in ("flat", "parameterized"):
                for tiled in (False, True):
                    prefix = f"{beam}__{kind}__{'chunked' if tiled else 'full'}__"
                    arrays.update({prefix + key: value for key, value in
                                   scalar_outputs(case(beam, scalar=True), kind, tiled, device).items()})
    REFERENCE.parent.mkdir(exist_ok=True)
    np.savez_compressed(REFERENCE, **arrays)
    print(f"Wrote {REFERENCE.relative_to(Path.cwd())}: {len(arrays)} arrays")
