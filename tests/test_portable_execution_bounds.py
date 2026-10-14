"""Postimplementation checks for portable defaults and legacy full-grid surfaces.

Memory checks observe actual tensor payloads and store rectangles, rather than
repeating the planner's sizing formula. Physical results use analytic chords
or the independent voxel-box matrix from the detector-surface regression suite.
"""

import math

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from diffct import Projector, TensorStore
from tests.test_detector_surfaces import _case, _grids, _matrix


# The portable automatic working-table ceiling is 8 MiB. Caller-owned arrays
# and their views are excluded; every newly allocated intermediate is observed.
WORK_BYTES = 8 * 1024 * 1024


class _Allocations(TorchDispatchMode):
    def __init__(self, *caller_arrays):
        super().__init__()
        self.caller_storage = {value.untyped_storage().data_ptr() for value in caller_arrays}
        self.largest = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        pending = [result]
        while pending:
            value = pending.pop()
            if isinstance(value, torch.Tensor):
                if value.untyped_storage().data_ptr() in self.caller_storage:
                    continue
                payload = value.numel() * value.element_size()
                self.largest = max(self.largest, payload)
                assert payload <= WORK_BYTES, (
                    f"{func} allocated {tuple(value.shape)} ({payload} bytes); "
                    f"portable working-table ceiling is {WORK_BYTES} bytes"
                )
            elif isinstance(value, (tuple, list)):
                pending.extend(value)
        return result


class _RecordedStore(TensorStore):
    def __init__(self, tensor):
        super().__init__(tensor)
        self.reads, self.writes = [], []
        self.flushes = 0

    def read(self, slices):
        block = super().read(slices)
        self.reads.append((tuple(slices), tuple(block.shape), block.numel() * block.element_size()))
        return block

    def write(self, slices, value, *, accumulate=False):
        self.writes.append((tuple(slices), tuple(value.shape), value.numel() * value.element_size()))
        return super().write(slices, value, accumulate=accumulate)

    def flush(self):
        self.flushes += 1


@pytest.fixture
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _project_uniform(shape, detectors, precision, **chunks):
    dtype = torch.float64 if precision == "float64" else torch.float32
    directions = torch.tensor([[1., 0.], [-1., 0.], [1., 0.], [-1., 0.]], dtype=torch.float64)
    origins = torch.tensor([[.17, -.23], [-.31, .19], [.27, .41], [-.13, -.37]], dtype=torch.float64)
    axes = torch.tensor([[0., 1.]] * 4, dtype=torch.float64)
    source = _RecordedStore(torch.full(shape, 1.25, dtype=dtype))
    sink = _RecordedStore(torch.empty((4, detectors), dtype=dtype))
    with _Allocations(source.tensor, sink.tensor) as allocations:
        operator = Projector(
            (directions, origins, axes), shape, (detectors,), beam="parallel",
            detector_spacing=.31, voxel_spacing=(.7, .31),
            backend="auto", precision=precision, **chunks,
        )
        assert operator.backend == "torch"
        assert operator.project_into(source, sink) is sink
    # All rays cross both x faces inside the y extent. The exact uniform chord
    # is physical width times density, including translated/reversed rays.
    expected = torch.full(sink.shape, shape[1] * .31 * 1.25, dtype=dtype)
    tolerance = 2e-11 if dtype == torch.float64 else 4e-5
    torch.testing.assert_close(sink.tensor, expected, rtol=tolerance, atol=tolerance)
    assert source.reads and sink.writes and sink.flushes == 1
    assert max(payload for _, _, payload in source.reads) <= WORK_BYTES
    assert max(payload for _, _, payload in sink.writes) <= WORK_BYTES
    assert sum(math.prod(size) for _, size, _ in sink.writes) == sink.tensor.numel()
    rectangles = {tuple((part.start, part.stop) for part in slices): size
                  for slices, size, _ in source.reads}
    assert sum(math.prod(size) for size in rectangles.values()) == source.tensor.numel()
    assert operator.last_execution_stats["ray_tile_pairs"] == len(source.reads)
    return source, sink, allocations


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_portable_defaults_bound_actual_tables_and_store_chunks(precision, bounded_cpu_threads):
    source, sink, allocations = _project_uniform((2049, 513), 257, precision)
    assert allocations.largest > 100_000, "long traversal did not exercise working tables"
    assert len(sink.writes) > 1
    assert max(size[1] for _, size, _ in sink.writes) < 257
    if precision == "float64":
        # The caller's volume exceeds the cap. Actual read rectangles must
        # partition it; checking only reported planner metadata is insufficient.
        assert source.tensor.numel() * source.tensor.element_size() > WORK_BYTES
        assert max(math.prod(size) for _, size, _ in source.reads) < source.tensor.numel()

    source, sink, _ = _project_uniform(
        (19, 31), 17, precision, volume_chunk_shape=(7, 11),
        view_chunk_size=2, detector_chunk_shape=(5,),
    )
    assert all(height <= 7 and width <= 11 for _, (height, width), _ in source.reads)
    assert all(views <= 2 and pixels <= 5 for _, (views, pixels), _ in sink.writes)
    assert any(size == (5, 9) for _, size, _ in source.reads), "short final spatial tile was not read"
    assert any(size[1] == 2 for _, size, _ in sink.writes), "short final detector rectangle was not written"


@pytest.mark.parametrize("kind", ["extent-dependent", "fixed-per-view"])
def test_legacy_surface_full_grids_snapshot_once_and_refresh_between_calls(kind):
    case = _case("cone")
    grid_u, grid_v = _grids(case)
    parameter = torch.tensor(.071, dtype=torch.float64, requires_grad=True)
    calls = []
    full_u, full_v = torch.from_numpy(grid_u.copy()), torch.from_numpy(grid_v.copy())
    base = np.stack((grid_u, grid_v, .03 * grid_u ** 2), axis=-1)
    view_scale = np.arange(case.views, dtype=np.float64)[:, None, None]
    fixed = np.broadcast_to(base, (*case.sino_shape, 3)).copy()
    fixed[..., 0] += .037 * view_scale
    fixed[..., 1] -= .029 * view_scale
    fixed[..., 2] *= 1 + .2 * view_scale
    shifts = np.zeros_like(fixed)
    shifts[..., 0], shifts[..., 1], shifts[..., 2] = 1 + .13 * view_scale, -.2, .11

    def surface(u, v):
        torch.testing.assert_close(u, full_u, rtol=0, atol=0)
        torch.testing.assert_close(v, full_v, rtol=0, atol=0)
        calls.append(float(parameter.detach()))
        if kind == "fixed-per-view":
            # A legacy closure may return an already assembled global array;
            # it cannot be reinterpreted as a rectangle-local sampler.
            return torch.from_numpy(fixed) + parameter * torch.from_numpy(shifts)
        return torch.stack((u + parameter * u / u.abs().max(),
                            v - .2 * parameter * v / v.abs().max(), .03 * u.square()), dim=-1)

    def oracle(value):
        if kind == "fixed-per-view":
            offsets = fixed + value * shifts
        else:
            offsets = np.stack((grid_u + value * grid_u / np.max(np.abs(grid_u)),
                                grid_v - .2 * value * grid_v / np.max(np.abs(grid_v)),
                                .03 * grid_u ** 2), axis=-1)
        return torch.from_numpy(_matrix(case, offsets))

    operator = Projector(
        case.trajectory, case.shape, case.detector, beam=case.beam,
        detector_spacing=case.pitch, voxel_spacing=case.spacing,
        backend="torch", precision="float64", detector_surface=surface,
        view_chunk_size=2, detector_chunk_shape=(2, 2), volume_chunk_shape=(2, 3, 4),
    )
    calls.clear()  # Constructor validation is separate from invocation sampling.
    image = torch.sin(torch.arange(math.prod(case.shape), dtype=torch.float64) * .713).reshape(case.shape)
    weight = torch.cos(torch.arange(math.prod(case.sino_shape), dtype=torch.float64) * .317).reshape(case.sino_shape)
    first_matrix, second_matrix = oracle(.071), oracle(.193)
    assert (first_matrix - second_matrix).abs().max() > 1e-4

    first_image = image.clone().requires_grad_()
    first = operator.project(first_image)
    assert calls == [.071]
    torch.testing.assert_close(first, (first_matrix @ image.flatten()).reshape(case.sino_shape),
                               rtol=2e-11, atol=2e-11)
    with torch.no_grad():
        parameter.fill_(.193)
    first_gradient, = torch.autograd.grad((first * weight).sum(), first_image)
    assert calls == [.071], "backward resampled the changed legacy closure"
    torch.testing.assert_close(first_gradient, (first_matrix.T @ weight.flatten()).reshape(case.shape),
                               rtol=2e-11, atol=2e-11)

    second_image = image.clone().requires_grad_()
    second = operator.project(second_image)
    assert calls == [.071, .193]
    second_gradient, = torch.autograd.grad((second * weight).sum(), second_image)
    assert calls == [.071, .193]
    torch.testing.assert_close(second, (second_matrix @ image.flatten()).reshape(case.sino_shape),
                               rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(second_gradient, (second_matrix.T @ weight.flatten()).reshape(case.shape),
                               rtol=2e-11, atol=2e-11)
    adjoint = operator.backproject(weight)
    assert calls == [.071, .193, .193]
    torch.testing.assert_close(adjoint, (second_matrix.T @ weight.flatten()).reshape(case.shape),
                               rtol=2e-11, atol=2e-11)
