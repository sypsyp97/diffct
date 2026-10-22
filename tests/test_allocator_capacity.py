"""Real native CUDA segment ceilings, with numerical and disk-solver evidence.

Each capacity trial owns a fresh spawned process. The ceiling covers PyTorch's
allocator, not the CUDA context, Numba module storage, or another process.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from tests._multiprocessing import spawn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from diffct import NpyStore, Projector, circular_trajectory_3d
from diffct import chunking
import diffct.operators as operators


MIB = 1024 ** 2
cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _box_chords(trajectory, shape, detector, pitch):
    """FP64 whole-box slab intersections; no production projector is an oracle."""
    u, v = np.meshgrid(*[(np.arange(size) + .5 - size / 2) * pitch
                         for size in detector], indexing="ij")
    lower, upper = -np.asarray(shape[::-1]) / 2, np.asarray(shape[::-1]) / 2
    result = []
    for source, centre, du, dv in zip(*(value.double().numpy() for value in trajectory)):
        direction = centre + u[..., None] * du + v[..., None] * dv - source
        enter, leave = np.zeros(detector), np.ones(detector)
        for axis in range(3):
            delta = direction[..., axis]
            moving = delta != 0
            first = np.divide(lower[axis] - source[axis], delta,
                              out=np.zeros_like(delta), where=moving)
            second = np.divide(upper[axis] - source[axis], delta,
                               out=np.zeros_like(delta), where=moving)
            enter = np.maximum(enter, np.where(moving, np.minimum(first, second), -np.inf))
            leave = np.minimum(leave, np.where(moving, np.maximum(first, second), np.inf))
            if not lower[axis] <= source[axis] < upper[axis]:
                leave[~moving] = -np.inf
        result.append(np.maximum(leave - enter, 0) * np.linalg.norm(direction, axis=-1))
    return np.asarray(result)


def _whole(store):
    return store.read(tuple(slice(0, size) for size in store.shape))


def _check_allocator(ceiling, records, label, operator):
    torch.cuda.synchronize()
    stats = torch.cuda.memory_stats()
    peaks = {name: stats[name + "_bytes.all.peak"]
             for name in ("allocated", "active", "reserved")}
    assert all(value <= ceiling for value in peaks.values()), (label, peaks, ceiling)
    report = operator.last_execution_stats
    assert report is not None and report["kernel_launches"] > 0
    records.append(dict(operation=label, peaks=peaks, execution=report))


def _example(name):
    path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"allocator_capacity_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def _cold_native_worker(rank, directory, foreign_stream=False):
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    assert torch.cuda.get_allocator_backend() == "native"
    ceiling = 32 * MIB
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(ceiling / total, 0)
    assert abs(torch.cuda.get_per_process_memory_fraction(0) * total - ceiling) < 1
    assert torch.cuda.memory_reserved() == 0, "trial must start with a cold PyTorch allocator"
    held = None
    if foreign_stream:
        caller_stream = torch.cuda.Stream()
        with torch.cuda.stream(caller_stream):
            held = torch.full((4 * MIB,), 41, dtype=torch.uint8, device="cuda:0")
        caller_stream.synchronize()
        assert torch.cuda.memory_allocated() == 4 * MIB
        assert torch.cuda.memory_reserved() == 20 * MIB

    directory = Path(directory)
    shape, detector, views, pitch = (256, 256, 256), (512, 512), 4, .8
    trajectory = circular_trajectory_3d(views, 640., 1024., device="cpu")
    operator = Projector(trajectory, shape, detector, detector_spacing=pitch,
                         devices=[0], view_chunk_size=4)
    truth = NpyStore.create(directory / "truth.npy", shape)
    truth.write(tuple(slice(0, size) for size in shape), torch.ones(shape))
    truth.flush()
    assert 4 * np.prod(shape) > ceiling
    rays = NpyStore.create(directory / "rays.npy", operator.projection_shape)
    expected = _box_chords(trajectory, shape, detector, pitch)
    assert np.isfinite(expected).all() and expected.max() > 200
    weights = torch.linspace(.17, .83, expected.size).reshape(operator.projection_shape)
    cotangent = NpyStore.create(directory / "cotangent.npy", operator.projection_shape)
    cotangent.write(tuple(slice(0, size) for size in cotangent.shape), weights)
    back = NpyStore.create(directory / "back.npy", shape)
    records = []
    torch.cuda.reset_peak_memory_stats()

    # Repetition retains the actual allocator caches and schedule choices.
    # Never empty the cache between operations to manufacture a fitting result.
    for repetition in range(2):
        assert operator.project_into(truth, rays) is rays
        _check_allocator(ceiling, records, f"project-{repetition}", operator)
        torch.testing.assert_close(_whole(rays).double(), torch.from_numpy(expected),
                                   rtol=4e-4, atol=2e-3)
        assert operator.last_execution_stats["view_chunk_size"] == 4
        assert operator.backproject_into(cotangent, back) is back
        _check_allocator(ceiling, records, f"backproject-{repetition}", operator)
        volume = _whole(back).double()
        assert torch.isfinite(volume).all() and volume.abs().sum() > 0
        # A(1) supplies an independent scalar adjoint oracle for this large grid.
        assert volume.sum().item() == pytest.approx(
            float(np.sum(expected * weights.numpy())), rel=4e-4, abs=2e-3)
        if held is not None:
            assert torch.all(held.cpu() == 41), "caller-owned foreign-stream data changed"

    if foreign_stream:
        (directory / "evidence.json").write_text(json.dumps(dict(
            allocator_ceiling_bytes=ceiling, caller_live_bytes=held.numel(), calls=records)), encoding="utf-8")
        return

    reconstruction = NpyStore.create(directory / "reconstruction.npy", shape)
    _example("_block_cgls").cgls_into(operator, rays, reconstruction, 2,
                                     workspace=directory / "cgls", block_elements=MIB)
    _check_allocator(ceiling, records, "disk-cgls", operator)
    prediction = NpyStore.create(directory / "prediction.npy", operator.projection_shape)
    operator.project_into(reconstruction, prediction)
    _check_allocator(ceiling, records, "reproject", operator)
    residual = np.linalg.norm(_whole(prediction).double().numpy() - expected) / np.linalg.norm(expected)
    assert np.isfinite(residual) and residual < .5
    assert torch.isfinite(_whole(reconstruction)).all()
    (directory / "evidence.json").write_text(json.dumps(dict(
        allocator_ceiling_bytes=ceiling, relative_residual=float(residual), calls=records)), encoding="utf-8")


@pytest.mark.cuda
@cuda_required
def test_cold_native_32mib_large_flat_cone_repeats_and_disk_cgls(tmp_path):
    if torch.cuda.get_allocator_backend() != "native":
        pytest.skip("native segment admission regression requires the native CUDA allocator")
    spawn(_cold_native_worker, args=(str(tmp_path),), nprocs=1, join=True)


@pytest.mark.cuda
@cuda_required
def test_live_foreign_stream_segment_leaves_room_for_automatic_small_pool_plan(tmp_path):
    if torch.cuda.get_allocator_backend() != "native":
        pytest.skip("foreign-stream segment admission requires the native CUDA allocator")
    spawn(_cold_native_worker, args=(str(tmp_path), True), nprocs=1, join=True)


class _IndividualCudaAllocations(TorchDispatchMode):
    """Observe actual storage requests; no allocator or numerical operation mocked."""
    def __init__(self):
        super().__init__()
        self.largest = 0

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        inputs = {value.untyped_storage()._cdata for value in tree_flatten(args)[0]
                  if isinstance(value, torch.Tensor)}
        result = function(*args, **(kwargs or {}))
        for value in tree_flatten(result)[0]:
            if isinstance(value, torch.Tensor) and value.is_cuda and value.untyped_storage()._cdata not in inputs:
                self.largest = max(self.largest, value.untyped_storage().nbytes())
                assert value.untyped_storage().nbytes() <= MIB, \
                    ("individual native allocation exceeds feasible small-pool limit", str(function), value.shape)
        return result


@torch.no_grad()
def _cgls_individual_worker(rank, entries):
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    ceiling = 12 * MIB
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(ceiling / total, 0)
    geometry = (torch.tensor([[1., 0.]]), torch.zeros(1, 2), torch.tensor([[0., 1.]]))
    operator = Projector(geometry, (1, entries), (1,), beam="parallel",
                         devices=[0], schedule="spatial", view_chunk_size=1)
    measurements = torch.tensor([[2.]], device="cuda:0")
    original = measurements.cpu()
    # Logical state and FP64 scratch total < 6 MiB. Each native middle-pool
    # request still needs a 20 MiB segment, larger than this real process cap.
    assert 3 * 4 * entries + 8 * entries + 32 < 6 * MIB < ceiling
    torch.cuda.reset_peak_memory_stats()
    with _IndividualCudaAllocations() as observed:
        result = _example("_common").cgls(operator, measurements, 1)
    torch.cuda.synchronize()
    if entries > MIB // 4:
        assert result.device.type == "cpu", "over-limit full volume state requires host fallback"
    torch.testing.assert_close(result.cpu().double(),
                               torch.full((1, entries), 2 / entries, dtype=torch.float64),
                               rtol=5e-4, atol=5e-8)
    torch.testing.assert_close(measurements.cpu(), original)
    assert observed.largest > 0 and torch.cuda.max_memory_reserved() <= ceiling
    report = operator.last_solver_stats
    assert report is not None and report["state_backend"] in ("cpu", "cuda")


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("entries", [262145, 262144], ids=["state-over-1mib", "fp64-scratch-over-1mib"])
def test_cgls_checks_individual_state_and_fp64_scratch_against_native_limit(entries):
    if torch.cuda.get_allocator_backend() != "native":
        pytest.skip("individual segment admission requires the native CUDA allocator")
    spawn(_cgls_individual_worker, args=(entries,), nprocs=1, join=True)


def _native_boundary_worker(rank, allocation_size, reservation):
    torch.cuda.set_device(0)
    ceiling = 64 * MIB
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(ceiling / total, 0)
    assert torch.cuda.memory_reserved() == 0
    value = torch.empty(allocation_size, dtype=torch.uint8, device="cuda:0")
    assert value.numel() == allocation_size
    assert torch.cuda.memory_allocated() == (allocation_size + 511) // 512 * 512
    assert torch.cuda.memory_reserved() == reservation


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("allocation_size,reservation", [
    (MIB, 2 * MIB), (MIB + 1, 20 * MIB),
    (10 * MIB - 512, 20 * MIB), (10 * MIB, 10 * MIB),
    (10 * MIB + 1, 12 * MIB),
], ids=["small-upper", "middle-lower", "middle-upper", "large-lower", "large-round-up"])
def test_actual_standard_native_segment_boundaries(allocation_size, reservation):
    if torch.cuda.get_allocator_backend() != "native":
        pytest.skip("segment sizes belong to the native CUDA allocator")
    spawn(_native_boundary_worker, args=(allocation_size, reservation), nprocs=1, join=True)


def _mixed_native_worker(rank):
    torch.cuda.set_device(0)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(64 * MIB / total, 0)
    assert torch.cuda.memory_reserved() == 0
    sizes = (6 * MIB, 10 * MIB, 6 * MIB)
    values = []
    for index, size in enumerate(sizes):
        values.append(torch.full((size,), index + 3, dtype=torch.uint8, device="cuda:0"))
    assert torch.cuda.memory_allocated() == sum(sizes)
    assert torch.cuda.memory_reserved() == 40 * MIB
    assert chunking._reservation_bytes(sizes) >= torch.cuda.memory_reserved()
    assert all(value[0].item() == value[-1].item() == index + 3
               for index, value in enumerate(values))


@pytest.mark.cuda
@cuda_required
def test_actual_shared_large_pool_6_10_6mib_requires_two_20mib_segments():
    if torch.cuda.get_allocator_backend() != "native":
        pytest.skip("shared large pool belongs to the native CUDA allocator")
    spawn(_mixed_native_worker, nprocs=1, join=True)


@torch.no_grad()
def _packing_worker(rank, explicit):
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    ceiling = 32 * MIB
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(ceiling / total, 0)
    shape, detector = (128, 128, 128), (512, 512)
    trajectory = circular_trajectory_3d(5, 320., 512., device="cpu")
    options = dict(volume_chunk_shape=shape, view_chunk_size=5) if explicit else {}
    operator = Projector(trajectory, shape, detector, detector_spacing=.8,
                         devices=[0], schedule="spatial", **options)
    output = torch.full(operator.projection_shape, -13.)
    launches = []
    original = operators._project_into

    def launched(*args, **kwargs):
        launches.append(True)
        return original(*args, **kwargs)

    torch.cuda.reset_peak_memory_stats()
    with mock.patch.object(operators, "_project_into", side_effect=launched):
        try:
            result = operator.project_into(torch.ones(shape), output)
        except RuntimeError as error:
            assert explicit, "automatic limits failed instead of finding a feasible smaller plan"
            assert not isinstance(error, torch.OutOfMemoryError), "unsafe explicit plan reached the allocator"
            assert not launches and operator.last_execution_stats is None
            assert torch.all(output == -13.), "rejected explicit plan modified caller output"
            return
    assert result is output and launches
    expected = _box_chords(trajectory, shape, detector, .8)
    torch.testing.assert_close(output.double(), torch.from_numpy(expected), rtol=4e-4, atol=2e-3)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_reserved() <= ceiling
    report = operator.last_execution_stats
    if not explicit:
        assert (tuple(report["chunk_shape"]) != shape or report["view_chunk_size"] < 5
                or tuple(report["detector_chunk_shape"]) != detector)
    else:
        assert tuple(report["chunk_shape"]) == shape and report["view_chunk_size"] == 5


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("explicit", [False, True], ids=["automatic-reduction", "explicit-fit-or-early-rejection"])
def test_real_8_8_5mib_inventory_uses_feasible_plan_or_preserves_rejected_output(explicit):
    if torch.cuda.get_allocator_backend() != "native":
        pytest.skip("native packing admission requires the native CUDA allocator")
    spawn(_packing_worker, args=(explicit,), nprocs=1, join=True)


def test_solver_admission_includes_forward_view_accumulators_when_rays_dominate_volume():
    """Metadata-only admission; independent live-buffer count, no fake GPU proof."""
    geometry = (torch.tensor([[1., 0.]]).repeat(4, 1), torch.zeros(4, 2),
                torch.tensor([[0., 1.]]).repeat(4, 1))
    with mock.patch.object(torch.cuda, "is_available", return_value=True), \
            mock.patch.object(torch.cuda, "device_count", return_value=1):
        operator = Projector(geometry, (2, 4), (4096,), beam="parallel", devices=[0],
                             schedule="views", volume_chunk_shape=(1, 4), view_chunk_size=1)
    plan = chunking._ChunkPlan((1, 4), 1, (torch.device("cuda:0"),), (4096,),
                               gpu_budget_bytes=320000, allocation_limit=MIB,
                               reservation_budget_bytes=8 * MIB)
    source = SimpleNamespace(shape=operator.projection_shape, dtype=torch.float32,
                             device=torch.device("cuda:0"))
    with mock.patch.object(torch.cuda, "is_available", return_value=True), \
            mock.patch.object(torch.cuda, "get_allocator_backend", return_value="native"), \
            mock.patch.object(torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=17)), \
            mock.patch.object(torch.cuda, "memory_snapshot", return_value=[]), \
            mock.patch.object(operator, "_execution_plan", return_value=plan):
        report = _example("_block_cgls")._solver_capacity(operator, source, MIB, "cuda:0")
    # Two producer tiles, four ray buffers (two slots + two resident SUM
    # accumulators), three frame buffers per slot. Native requests round to512.
    minimum_forward_bytes = 2 * 512 + 4 * (4096 * 4) + 2 * 3 * 512
    assert report["estimated_pipeline_and_geometry_bytes"] >= minimum_forward_bytes
    assert report["estimated_gpu_bytes"] > report["gpu_budget_bytes"], \
        "solver admitted CUDA states without the larger forward working inventory"
