"""Check the public cone adjoint against columns of its forward matrix."""

from contextlib import nullcontext
from functools import wraps
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from numba.core import config

from diffct import Projector


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not config.ENABLE_CUDASIM
        and (not torch.cuda.is_available() or torch.version.hip is not None),
        reason="NVIDIA CUDA or NUMBA_ENABLE_CUDASIM=1 is required",
    ),
]

VOLUME_SHAPE = (2, 3, 4)
VOXEL_SPACING = 0.75
DETECTOR_COLUMNS = 5


class _SimulatorTensor(torch.Tensor):
    """Expose CUDA metadata over CPU storage used by Numba's simulator."""

    @property
    def device(self):
        return torch.device("cuda:0")

    @property
    def is_cuda(self):
        return True

    def cpu(self):
        return self.as_subclass(torch.Tensor)

    def record_stream(self, stream):
        pass


@pytest.fixture
def cuda_device(monkeypatch):
    if not config.ENABLE_CUDASIM:
        return torch.device("cuda:0")

    # Replace only CUDA transport. Projector planning, geometry, dispatch,
    # and both production ray kernels run unchanged in the simulator.
    from diffct import operators, projectors
    from diffct.utils import TorchCUDABridge

    def factory(original):
        @wraps(original)
        def allocate(*args, **kwargs):
            device = kwargs.get("device")
            simulated = device is not None and torch.device(device).type == "cuda"
            if simulated:
                kwargs["device"] = "cpu"
            result = original(*args, **kwargs)
            return result.as_subclass(_SimulatorTensor) if simulated else result

        return allocate

    original_to = torch.Tensor.to

    def to(tensor, *args, **kwargs):
        device = kwargs.get("device")
        if args and isinstance(args[0], (str, torch.device)):
            device = args[0]
        simulated = device is not None and torch.device(device).type == "cuda"
        if simulated:
            if args and isinstance(args[0], (str, torch.device)):
                args = (torch.device("cpu"), *args[1:])
            else:
                kwargs["device"] = "cpu"
        result = original_to(tensor, *args, **kwargs)
        if simulated:
            return result.as_subclass(_SimulatorTensor)
        if device is not None and isinstance(result, _SimulatorTensor):
            return result.as_subclass(torch.Tensor)
        return result

    stream = SimpleNamespace(device=torch.device("cuda:0"), cuda_stream=0,
                             wait_event=lambda event: None, wait_stream=lambda other: None)
    event = lambda: SimpleNamespace(record=lambda *args: None, synchronize=lambda: None,
                                   query=lambda: True)
    monkeypatch.setattr(torch.Tensor, "to", to)
    for name in ("empty", "zeros", "arange", "tensor"):
        monkeypatch.setattr(torch, name, factory(getattr(torch, name)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: stream)
    monkeypatch.setattr(torch.cuda, "Event", event)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args: (2**30, 2**30))
    monkeypatch.setattr(torch.cuda, "memory_stats_as_nested_dict", lambda *args: {})
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", lambda *args: 1.0)
    monkeypatch.setattr(TorchCUDABridge, "tensor_to_cuda_array",
                        lambda tensor: tensor.detach().cpu().numpy())
    for module in (operators, projectors):
        monkeypatch.setattr(module, "_get_numba_external_stream_for", lambda *args: 0)
    return torch.device("cuda:0")


def _curved_detector(u, v):
    # The central pixel stays on each exact axis or grazing line.
    return torch.stack((u, v, (u.square() + v.square()) / 32), dim=-1)


def _orbit(helical):
    angles = torch.tensor([0.31, 1.17], dtype=torch.float32)
    radial = torch.stack((angles.cos(), angles.sin(), torch.zeros_like(angles)), dim=-1)
    tangent = torch.stack((-angles.sin(), angles.cos(), torch.zeros_like(angles)), dim=-1)
    vertical = torch.tensor([[0.0, 0.0, 1.0]]).repeat(2, 1)
    offset = torch.zeros_like(radial)
    if helical:
        offset[:, 2] = torch.tensor([-0.45, 0.6])
    return -6 * radial + offset, 6 * radial + offset, tangent, vertical


def _axis_and_face_rays():
    # Coordinates are binary-exact: faces are x=+-1.5, y=+-1.125, z=+-.75.
    # Central pixels give x-, y-, and z-parallel rays, then lines grazing
    # the lower/upper y faces and the upper z face. Wide outer columns miss.
    source = [(-6, 0, 0), (0, -6, 0), (0, 0, -6),
              (-6, -1.125, 0), (-6, 1.125, 0), (-6, 0, 0.75)]
    center = [(6, 0, 0), (0, 6, 0), (0, 0, 6),
              (6, -1.125, 0), (6, 1.125, 0), (6, 0, 0.75)]
    u = [(0, 1, 0), (1, 0, 0), (1, 0, 0)] + [(0, 1, 0)] * 3
    v = [(0, 0, 1), (0, 0, 1), (0, 1, 0)] + [(0, 0, 1)] * 3
    return tuple(torch.tensor(values, dtype=torch.float32)
                 for values in (source, center, u, v))


def _check_transpose(trajectory, rows, surface, device):
    operator = Projector(
        trajectory, VOLUME_SHAPE, (DETECTOR_COLUMNS, rows), beam="cone",
        detector_spacing=(3.0, 1.5), voxel_spacing=VOXEL_SPACING,
        detector_surface=surface, backend="numba", devices=[device], distributed=False,
    )
    columns = []
    with torch.no_grad():
        for index in range(math.prod(VOLUME_SHAPE)):
            basis = torch.zeros(VOLUME_SHAPE, dtype=torch.float32, device=device)
            basis.flatten()[index] = 1.0
            projected = operator.project(basis)
            assert projected.dtype == torch.float32
            assert tuple(projected.shape) == operator.projection_shape
            columns.append(projected.detach().cpu().numpy().reshape(-1).astype(np.float64))
        matrix = np.column_stack(columns)
        assert np.isfinite(matrix).all()
        hit = np.any(matrix != 0, axis=1)
        assert hit.any() and (~hit).any(), "detector must both hit and miss the volume"
        rng = np.random.default_rng(713)
        y = rng.standard_normal(operator.projection_shape).astype(np.float32)
        expected = matrix.T @ y.reshape(-1).astype(np.float64)
        actual = operator.backproject(torch.from_numpy(y).to(device=device))

    assert tuple(actual.shape) == VOLUME_SHAPE
    assert actual.dtype == torch.float32
    actual = actual.detach().cpu().numpy().reshape(-1).astype(np.float64)
    assert np.isfinite(actual).all()
    scale = np.max(np.abs(expected))
    assert scale > 0, "the transpose comparison must not be vacuous"
    # Float32 atomic accumulation changes summation order. Use one global
    # scale so cancellation near zero does not require relative voxel accuracy.
    relative_max_error = np.max(np.abs(actual - expected)) / scale
    assert relative_max_error <= 1e-5, f"relative max error: {relative_max_error:.9g}"


@pytest.mark.parametrize("surface", [None, _curved_detector], ids=["flat", "callable"])
@pytest.mark.parametrize("rows,helical", [(1, False), (3, True), (33, False)],
                         ids=["rows1-circular", "rows3-helical", "rows33-circular"])
def test_cone_backprojection_matches_forward_transpose(cuda_device, surface, rows, helical):
    _check_transpose(_orbit(helical), rows, surface, cuda_device)


@pytest.mark.parametrize("surface", [None, _curved_detector], ids=["flat", "callable"])
def test_cone_axis_parallel_grazing_and_missed_rays_match_transpose(cuda_device, surface):
    _check_transpose(_axis_and_face_rays(), 3, surface, cuda_device)
