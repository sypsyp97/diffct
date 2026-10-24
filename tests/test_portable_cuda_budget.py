"""Portable-backend work budget, projection ray reuse and auto-fallback warning.

The checks follow the public contract only:

* C1: on a CUDA compute device the automatic portable work budget follows the
  device memory; on CPU it stays at 8 MiB. Results (and autograd gradients)
  must not depend on the budget beyond summation order, so every automatic run
  is compared with a run that uses explicit small chunks.
* C2: projection rays are set up once per ray batch. Outputs for the same
  chunking stay bitwise identical, so automatic and explicit-equivalent plans,
  repeated calls and recorded pre-change digests must agree exactly.
* C3: ``backend="auto"`` warns once per instance when NVIDIA CUDA is available
  but the native backend cannot be used, and stays silent otherwise.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import math
from pathlib import Path
import platform
import re
import time
import warnings

import pytest
import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode

from diffct import backends
from diffct import (
    ParameterizedSurface, Projector, circular_trajectory_2d_fan, circular_trajectory_3d,
)
from tests._multiprocessing import spawn


CPU_WORK_BYTES = 8 * 1024 * 1024
FLOAT32_TOLERANCE = 1e-5
FLOAT64_TOLERANCE = 1e-10
# FP32 traversal parameters are measured locally from each tile entry, so a
# different tile grid also changes segment rounding, not only summation order.
# Before this change, CPU automatic (8 MiB) and small explicit chunks already
# differed by up to 4.6e-5 x max for the larger cases below (fan 160x192,
# 24 views, data gradient), so those cases use this bound.
FLOAT32_TILE_GRID_TOLERANCE = 1e-4
CUDA = torch.cuda.is_available()
NVIDIA_CUDA = CUDA and torch.version.hip is None
REPOSITORY = Path(__file__).resolve().parents[1]

requires_cuda = pytest.mark.skipif(not CUDA, reason="portable CUDA budget needs a CUDA device")
requires_two_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="two CUDA devices are required",
)
requires_nvidia = pytest.mark.skipif(not NVIDIA_CUDA, reason="real NVIDIA CUDA is required")


# --------------------------------------------------------------------------
# Shared fixtures
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class _Case:
    beam: str
    shape: tuple[int, ...]
    detector: tuple[int, ...]
    views: int
    spacing: tuple[float, ...]
    precision: str
    surface: bool = False
    trainable: bool = False
    float32_tolerance: float = FLOAT32_TOLERANCE

    @property
    def dtype(self):
        return torch.float64 if self.precision == "float64" else torch.float32

    @property
    def tolerance(self):
        return FLOAT64_TOLERANCE if self.precision == "float64" else self.float32_tolerance

    @property
    def sino_shape(self):
        return (self.views, *self.detector)

    def geometry(self):
        """Source distances and detector pitch that cover most of the volume."""
        extent = max(size * pitch for size, pitch in zip(self.shape, self.spacing))
        sid = 2.5 * extent
        sdd = 2.0 * sid
        magnified = 1.6 * extent * sdd / sid
        pitch = tuple(magnified / size for size in self.detector)
        return sid, sdd, pitch

    def small_chunks(self):
        return dict(
            volume_chunk_shape=tuple(max(1, size // 3 + 1) for size in self.shape),
            view_chunk_size=max(1, min(3, self.views - 1)),
            detector_chunk_shape=tuple(max(1, size // 2) for size in self.detector),
        )


def _cylinder(u, v, view_indices, radius):
    # Source-centred cylinder: offsets along u bend toward the source.
    angle = u / radius
    return torch.stack((radius * torch.sin(angle), v + torch.zeros_like(u),
                        radius * (torch.cos(angle) - 1)), dim=-1)


def _trajectory(case):
    sid, sdd, _ = case.geometry()
    if case.beam == "fan":
        return circular_trajectory_2d_fan(case.views, sid, sdd, start_angle=0.1,
                                          device="cpu", dtype=torch.float64)
    return circular_trajectory_3d(case.views, sid, sdd, start_angle=0.1,
                                  device="cpu", dtype=torch.float64)


def _projector(case, *, geometry_device="cpu", trajectory=None, surface=None, **options):
    _, _, pitch = case.geometry()
    if trajectory is None:
        trajectory = tuple(value.to(geometry_device) for value in _trajectory(case))
    return Projector(
        trajectory, case.shape, case.detector, beam=case.beam,
        detector_spacing=pitch if case.beam == "cone" else pitch[0],
        voxel_spacing=case.spacing, backend="torch", precision=case.precision,
        detector_surface=surface, **options,
    )


def _data(case, device, seed=1234):
    generator = torch.Generator().manual_seed(seed)
    volume = torch.rand(case.shape, generator=generator, dtype=torch.float64)
    sinogram = torch.rand(case.sino_shape, generator=generator, dtype=torch.float64)
    return volume.to(device=device, dtype=case.dtype), sinogram.to(device=device, dtype=case.dtype)


def _evaluate(case, operation, *, data_device, geometry_device=None, **options):
    """Return the output and every autograd gradient of one call, on CPU."""
    geometry_device = data_device if geometry_device is None else geometry_device
    leaves = {}
    trajectory = [value.to(geometry_device) for value in _trajectory(case)]
    if case.trainable:
        names = ("source", "detector_center")
        for index, name in enumerate(names):
            trajectory[index] = trajectory[index].clone().requires_grad_()
            leaves[name] = trajectory[index]
    surface = None
    if case.surface:
        _, sdd, _ = case.geometry()
        radius = torch.tensor(sdd, dtype=torch.float64, requires_grad=case.trainable)
        if case.trainable:
            leaves["surface_radius"] = radius
        surface = ParameterizedSurface(_cylinder, (radius,))
    operator = _projector(case, trajectory=tuple(trajectory), surface=surface, **options)
    volume, sinogram = _data(case, data_device)
    if operation == "project":
        source, weight = volume.requires_grad_(), sinogram
    else:
        source, weight = sinogram.requires_grad_(), volume
    leaves["data"] = source
    output = getattr(operator, operation)(source)
    names = list(leaves)
    gradients = torch.autograd.grad((output * weight).sum(), [leaves[name] for name in names])
    result = {"output": output.detach().cpu()}
    result.update({f"grad[{name}]": value.detach().cpu() for name, value in zip(names, gradients)})
    return result, operator


def _assert_max_relative(actual, expected, tolerance, label):
    assert actual.keys() == expected.keys(), f"{label}: result sets differ"
    for key in expected:
        reference = expected[key].double()
        scale = reference.abs().max().item()
        assert scale > 0, f"{label}/{key}: uninformative all-zero reference"
        error = (actual[key].double() - reference).abs().max().item()
        assert error <= tolerance * scale, (
            f"{label}/{key}: max |difference| {error:.3e} exceeds "
            f"{tolerance:g} x max |reference| ({scale:.3e})"
        )


# CPU cases stay small; CUDA cases need several 8 MiB work arrays so that a
# device-sized budget actually changes the automatic plan.
CPU_CASES = {
    "fan-anisotropic-float32": _Case("fan", (13, 15), (17,), 5, (0.9, 0.7), "float32"),
    "cone-anisotropic-float32": _Case("cone", (7, 9, 11), (10, 8), 5, (1.1, 0.9, 0.8), "float32"),
    "cone-float64-trainable": _Case("cone", (7, 9, 11), (10, 8), 5, (1.0, 1.0, 1.0), "float64",
                                    trainable=True),
    "cone-surface-float32-trainable": _Case("cone", (7, 9, 11), (10, 8), 5, (1.1, 0.9, 0.8),
                                            "float32", surface=True, trainable=True),
    "cone-surface-float64-trainable": _Case("cone", (7, 9, 11), (10, 8), 5, (1.1, 0.9, 0.8),
                                            "float64", surface=True, trainable=True),
}
CUDA_CASES = {
    "fan-anisotropic-float32": _Case("fan", (160, 192), (256,), 24, (0.9, 0.7), "float32",
                                     float32_tolerance=FLOAT32_TILE_GRID_TOLERANCE),
    "cone-anisotropic-float32": _Case("cone", (25, 64, 72), (64, 48), 10, (1.1, 0.9, 0.8),
                                      "float32", float32_tolerance=FLOAT32_TILE_GRID_TOLERANCE),
    "cone-float64-trainable": _Case("cone", (25, 64, 72), (64, 48), 10, (1.0, 1.0, 1.0),
                                    "float64", trainable=True),
    "cone-surface-float32-trainable": _Case("cone", (25, 64, 72), (64, 48), 10, (1.1, 0.9, 0.8),
                                            "float32", surface=True, trainable=True,
                                            float32_tolerance=FLOAT32_TILE_GRID_TOLERANCE),
    "cone-surface-float64-trainable": _Case("cone", (25, 64, 72), (64, 48), 10, (1.1, 0.9, 0.8),
                                            "float64", surface=True, trainable=True),
}
OPERATIONS = ("project", "backproject")


@pytest.fixture
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


# --------------------------------------------------------------------------
# C1: automatic budget changes the plan, never the result
# --------------------------------------------------------------------------

@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("name", list(CPU_CASES))
def test_cpu_automatic_plan_matches_explicit_small_chunks_with_gradients(name, operation):
    case = CPU_CASES[name]
    automatic, _ = _evaluate(case, operation, data_device="cpu")
    explicit, _ = _evaluate(case, operation, data_device="cpu", **case.small_chunks())
    _assert_max_relative(automatic, explicit, case.tolerance, f"cpu/{name}/{operation}")


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("placement", ["cuda-input", "cpu-input-devices0"])
@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("name", list(CUDA_CASES))
def test_cuda_automatic_budget_matches_explicit_small_chunks_with_gradients(name, operation, placement):
    case = CUDA_CASES[name]
    if placement == "cuda-input":
        options = dict(data_device="cuda")
    else:
        # CPU caller data, CUDA compute device: the CUDA budget applies.
        options = dict(data_device="cpu", geometry_device="cpu", devices=[0])
    automatic, operator = _evaluate(case, operation, **options)
    stats = operator.last_execution_stats
    assert stats is not None and any(device.startswith("cuda") for device in stats["devices"]), (
        f"portable call did not compute on CUDA: {stats}"
    )
    explicit, _ = _evaluate(case, operation, **options, **case.small_chunks())
    _assert_max_relative(automatic, explicit, case.tolerance, f"{placement}/{name}/{operation}")


@pytest.mark.cuda
@requires_two_gpus
@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("name", ["cone-anisotropic-float32", "cone-surface-float64-trainable"])
def test_multi_gpu_devices_match_single_device_with_gradients(name, operation):
    case = CUDA_CASES[name]
    single, _ = _evaluate(case, operation, data_device="cuda:0", devices=[0])
    # The contract does not require every listed device to receive work: one
    # automatic batch can cover the whole problem and run on one device.
    multiple, _ = _evaluate(case, operation, data_device="cuda:0", devices=[0, 1])
    _assert_max_relative(multiple, single, case.tolerance, f"devices=[0,1]/{name}/{operation}")
    explicit, _ = _evaluate(case, operation, data_device="cuda:0", devices=[0, 1],
                            **case.small_chunks())
    _assert_max_relative(multiple, explicit, case.tolerance,
                         f"devices=[0,1] small chunks/{name}/{operation}")


class _LargestAllocation(TorchDispatchMode):
    """Record the largest tensor payload allocated inside a call."""

    def __init__(self, *caller_arrays):
        super().__init__()
        self.caller = {value.untyped_storage().data_ptr() for value in caller_arrays}
        self.largest = 0
        self.operation = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        pending = [result]
        while pending:
            value = pending.pop()
            if isinstance(value, torch.Tensor):
                if value.untyped_storage().data_ptr() in self.caller:
                    continue
                payload = value.numel() * value.element_size()
                if payload > self.largest:
                    self.largest, self.operation = payload, (str(func), tuple(value.shape))
            elif isinstance(value, (tuple, list)):
                pending.extend(value)
        return result


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_cpu_automatic_work_arrays_keep_8_mib_ceiling(precision, operation):
    # Full traversal tables would need about 100 MiB; caller data and the
    # returned array are each below 8 MiB, so every observed tensor counts.
    case = _Case("fan", (512, 512), (512,), 48, (0.9, 0.7), precision)
    operator = _projector(case)
    volume, sinogram = _data(case, "cpu")
    source = volume if operation == "project" else sinogram
    with torch.no_grad(), _LargestAllocation(volume, sinogram) as observed:
        output = getattr(operator, operation)(source)
    assert output.numel() * output.element_size() <= CPU_WORK_BYTES
    assert observed.largest <= CPU_WORK_BYTES, (
        f"CPU portable {operation} allocated {observed.largest} bytes "
        f"({observed.operation}); the CPU automatic ceiling is {CPU_WORK_BYTES} bytes"
    )
    assert observed.largest > CPU_WORK_BYTES // 8, "the case did not exercise the work tables"


def _documented_cuda_fraction():
    """Largest fraction stated next to free CUDA memory for the portable path."""
    path = REPOSITORY / "docs" / "EXECUTION.md"
    if not path.is_file():
        assert not (REPOSITORY / ".git").exists(), f"source checkout missing required documentation {path}"
        pytest.skip("docs/EXECUTION.md is not copied into installed-wheel test layouts")
    text = path.read_text(encoding="utf-8")
    words = {"half": .5, "quarter": .25, "third": 1 / 3}
    # Wording for "free memory" varies; accept common synonyms.
    free = re.compile(r"\bfree\b|\bavailable\b|\busable\b|\bunused\b|can still use")
    fractions = []
    for paragraph in re.split(r"\n\s*\n", text):
        flat = " ".join(paragraph.split())
        lowered = flat.lower()
        if not (free.search(lowered) and "cuda" in lowered and "memory" in lowered
                and ("portable" in lowered or "pytorch" in lowered)):
            continue
        fractions += [float(value) / 100 for value in re.findall(r"(\d+(?:\.\d+)?)\s*%", flat)]
        fractions += [float(numerator) / float(denominator) for numerator, denominator
                      in re.findall(r"\b(\d+)\s*/\s*(\d+)\b", flat) if float(denominator) > 0]
        fractions += [float(value) for value in re.findall(r"(?<![\d.])(0?\.\d+)(?![\d.])", flat)]
        fractions += [value for word, value in words.items() if re.search(rf"\b{word}\b", lowered)]
    fractions = [value for value in fractions if 0 < value <= 1]
    return max(fractions) if fractions else None


def test_execution_docs_state_the_portable_cuda_free_memory_fraction():
    fraction = _documented_cuda_fraction()
    assert fraction is not None, (
        "docs/EXECUTION.md must state the fraction of free CUDA memory that the "
        "portable backend's automatic work arrays may use"
    )


def _large_cone():
    return _Case("cone", (64, 256, 256), (384, 256), 16, (0.5, 0.3, 0.3), "float32")


def _bound_fraction():
    documented = _documented_cuda_fraction()
    return 1.0 if documented is None else documented


def _usable_cuda_bytes(device):
    """Memory the PyTorch allocator can still use on ``device`` (after emptying its cache)."""
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reusable = max(0, torch.cuda.memory_reserved(device) - allocated)
    getter = getattr(torch.cuda, "get_per_process_memory_fraction", None)
    cap = getter(device) if getter is not None else 1.0
    return max(0, min(free + reusable, int(total * cap) - allocated))


@contextmanager
def _allocator_room(device, room):
    """Cap this process so that its allocator can use about ``room`` more bytes.

    A small room makes caller-sized transients (for example a second full
    output) visible next to the work arrays, without multi-GiB test arrays.
    """
    getter = getattr(torch.cuda, "get_per_process_memory_fraction", None)
    if getter is None:
        pytest.skip("needs torch.cuda.get_per_process_memory_fraction")
    previous = getter(device)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    if free < 2 * room:
        pytest.skip(f"needs {2 * room} free bytes on {device}")
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, (torch.cuda.memory_reserved(device) + room) / total), device,
    )
    try:
        yield
    finally:
        torch.cuda.set_per_process_memory_fraction(previous, device)
        torch.cuda.empty_cache()


def _assert_within_bound(extra, usable, label):
    fraction = _bound_fraction()
    assert extra <= fraction * usable, (
        f"{label}: the call allocated {extra} transient CUDA bytes; the documented bound is "
        f"{fraction:g} x {usable} usable bytes at call start"
    )


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("operation", OPERATIONS)
def test_cuda_automatic_work_memory_is_bounded_by_free_memory(operation):
    case = _large_cone()
    device = torch.device("cuda", torch.cuda.current_device())
    operator = _projector(case, geometry_device=device)
    volume, sinogram = _data(case, device)
    source = volume if operation == "project" else sinogram
    usable = _usable_cuda_bytes(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        output = getattr(operator, operation)(source)
    torch.cuda.synchronize(device)
    extra = (torch.cuda.max_memory_allocated(device) - baseline
             - output.numel() * output.element_size())
    _assert_within_bound(extra, usable, f"CUDA {operation}")
    # The budget follows device memory, not the fixed 8 MiB CPU ceiling. This
    # problem needs several GiB of traversal tables when unchunked.
    if _bound_fraction() * usable >= 1 << 30:
        assert extra > 4 * CPU_WORK_BYTES, (
            f"portable CUDA {operation} peak work memory is {extra} bytes with "
            f"{usable} bytes usable; the automatic budget still looks fixed at 8 MiB"
        )


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("operation", OPERATIONS)
def test_cpu_data_with_cuda_compute_uses_a_device_sized_plan(operation):
    # CPU caller arrays, CUDA compute device: CUDA work arrays get the device
    # budget. All CUDA allocations are transients of the call.
    case = _large_cone()
    device = torch.device("cuda", 0)
    operator = _projector(case, devices=[0])
    volume, sinogram = _data(case, "cpu")
    source = volume if operation == "project" else sinogram
    usable = _usable_cuda_bytes(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        output = getattr(operator, operation)(source)
    torch.cuda.synchronize(device)
    assert output.device.type == "cpu"
    extra = torch.cuda.max_memory_allocated(device) - baseline
    _assert_within_bound(extra, usable, f"CPU data, CUDA {operation}")
    if _bound_fraction() * usable >= 1 << 30:
        assert extra > 4 * CPU_WORK_BYTES, (
            f"CPU data with devices=[0]: peak CUDA work memory of {operation} is {extra} bytes "
            f"with {usable} bytes usable; the plan still looks fixed at 8 MiB"
        )


# Outputs are large relative to the allowed room, so a second full output or
# a saved autograd graph outside the budget breaks the bound.
_TRANSIENT_CASES = {
    False: _Case("cone", (96, 384, 384), (384, 256), 16, (0.5, 0.3, 0.3), "float32"),
    True: _Case("cone", (33, 96, 96), (64, 64), 8, (0.5, 0.3, 0.3), "float32"),
}


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("autograd", [False, True], ids=["no-grad", "autograd"])
@pytest.mark.parametrize("operation", OPERATIONS)
def test_cuda_bound_covers_output_copies_and_the_recorded_graph(operation, autograd):
    case = _TRANSIENT_CASES[autograd]
    device = torch.device("cuda", torch.cuda.current_device())
    operator = _projector(case, geometry_device=device)
    volume, sinogram = _data(case, device)
    source = (volume if operation == "project" else sinogram).requires_grad_(autograd)
    with _allocator_room(device, 1 << 30):
        usable = _usable_cuda_bytes(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.set_grad_enabled(autograd):
            output = getattr(operator, operation)(source)
        torch.cuda.synchronize(device)
        # The returned array is the caller's; everything else counts, including
        # the graph that stays saved for backward.
        extra = (torch.cuda.max_memory_allocated(device) - baseline
                 - output.numel() * output.element_size())
    assert output.requires_grad == autograd
    _assert_within_bound(extra, usable, f"{operation} ({'autograd' if autograd else 'no grad'})")


def _contract_example():
    """Volume (81, 512, 512), per-axis spacing, cylindrical surface, 888x625 detector, 24 views."""
    shape, detector, views = (81, 512, 512), (888, 625), 24
    sid, sdd = 600.0, 1000.0
    trajectory = circular_trajectory_3d(views, sid, sdd, start_angle=0.1,
                                        device="cuda", dtype=torch.float32)

    def cylinder(u, v):
        angle = u / sdd
        return torch.stack((sdd * angle.sin(), v, sdd * (angle.cos() - 1)), dim=-1)

    operator = Projector(trajectory, shape, detector, beam="cone",
                         detector_spacing=(0.4, 0.15), voxel_spacing=(0.5, 0.25, 0.25),
                         backend="torch", precision="float32", detector_surface=cylinder)
    warm = Projector(tuple(value[:2] for value in trajectory),
                     (4, 8, 8), (8, 4), beam="cone", detector_spacing=(0.4, 0.15),
                     voxel_spacing=(0.5, 0.25, 0.25), backend="torch", precision="float32")
    with torch.no_grad():
        warm.project(torch.ones(4, 8, 8, device="cuda"))
    volume = torch.rand(shape, device="cuda", generator=torch.Generator("cuda").manual_seed(5))
    return operator, volume


def _timed_project(operator, volume):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        sinogram = operator.project(volume)
    torch.cuda.synchronize()
    return sinogram, time.perf_counter() - start


@pytest.mark.cuda
@pytest.mark.benchmark
@requires_cuda
def test_large_curved_anisotropic_cuda_projection_finishes_in_seconds():
    operator, volume = _contract_example()
    sinogram, elapsed = _timed_project(operator, volume)
    assert sinogram.shape == (24, 888, 625)
    assert torch.isfinite(sinogram).all() and sinogram.abs().max() > 0
    assert elapsed < 60.0, f"one portable CUDA projection of the contract example took {elapsed:.1f} s"


@pytest.mark.cuda
@pytest.mark.benchmark
@requires_cuda
def test_contract_example_is_20x_faster_than_the_fixed_8_mib_plan(monkeypatch):
    # The old behaviour is the fixed 8 MiB plan. The portable backend selects
    # it when its device budget hook returns None (the host/CPU branch).
    if not hasattr(backends.ReferenceProjector, "_work_budget"):
        pytest.skip("this tree has no device-sized portable budget to compare")
    operator, volume = _contract_example()
    _timed_project(operator, volume)
    automatic, elapsed = _timed_project(operator, volume)
    with monkeypatch.context() as patch:
        patch.setattr(backends.ReferenceProjector, "_work_budget",
                      lambda self, *args, **kwargs: None)
        fixed, fixed_elapsed = _timed_project(operator, volume)
    _assert_max_relative({"output": automatic.cpu()}, {"output": fixed.cpu()},
                         FLOAT32_TILE_GRID_TOLERANCE, "automatic vs fixed 8 MiB plan")
    assert fixed_elapsed >= 20 * elapsed, (
        f"automatic plan {elapsed:.2f} s vs fixed 8 MiB plan {fixed_elapsed:.2f} s: "
        f"speedup {fixed_elapsed / elapsed:.1f}x < 20x"
    )


# --------------------------------------------------------------------------
# C1: distributed ranks use their own device and match one process
# --------------------------------------------------------------------------

DISTRIBUTED_CASE = _Case("cone", (9, 12, 14), (13, 10), 6, (1.1, 0.9, 0.8), "float64")
DISTRIBUTED_CUDA_CASE = _Case("cone", (25, 64, 72), (64, 48), 10, (1.1, 0.9, 0.8), "float32",
                              float32_tolerance=FLOAT32_TILE_GRID_TOLERANCE)


def _distributed_worker(rank, transport, partition, rendezvous, case):
    torch.set_num_threads(1)
    if transport == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        devices = [rank]
    else:
        device = torch.device("cpu")
        devices = None
    dist.init_process_group(transport, init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        reference = _projector(case, geometry_device=device, devices=devices, **case.small_chunks())
        operator = _projector(case, geometry_device=device, devices=devices,
                              distributed=True, partition=partition)
        volume, sinogram = _data(case, device)
        volume_slice, view_slice = operator.volume_slice, operator.view_slice

        local_volume = volume[volume_slice].clone().requires_grad_()
        projection = operator.project(local_volume)
        expected_projection = reference.project(volume)
        divisor = operator.space_world_size
        if partition == "views":
            expected_projection = expected_projection[view_slice]
            weight = sinogram[view_slice]
        else:
            weight = sinogram
        ((projection * weight).sum() / divisor).backward()
        expected_volume_gradient = reference.backproject(sinogram)[volume_slice]

        local_sinogram = (sinogram[view_slice] if partition == "views" else sinogram).clone()
        local_sinogram.requires_grad_()
        adjoint = operator.backproject(local_sinogram)
        expected_adjoint = reference.backproject(sinogram)[volume_slice]
        weight = volume[volume_slice]
        ((adjoint * weight).sum() / operator.view_world_size).backward()
        expected_sinogram_gradient = reference.project(volume)
        if partition == "views":
            expected_sinogram_gradient = expected_sinogram_gradient[view_slice]

        actual = {"project": projection.detach().cpu(), "grad[volume]": local_volume.grad.cpu(),
                  "backproject": adjoint.detach().cpu(), "grad[sinogram]": local_sinogram.grad.cpu()}
        expected = {"project": expected_projection.cpu(), "grad[volume]": expected_volume_gradient.cpu(),
                    "backproject": expected_adjoint.cpu(),
                    "grad[sinogram]": expected_sinogram_gradient.cpu()}
        for key in expected:
            if expected[key].numel():
                _assert_max_relative({key: actual[key]}, {key: expected[key]}, case.tolerance,
                                     f"{transport}/{partition}/rank {rank}")
            else:
                assert actual[key].shape == expected[key].shape
        if transport == "nccl":
            stats = operator.last_execution_stats
            assert stats["devices"] == [f"cuda:{rank}"], f"rank {rank} computed on {stats['devices']}"
    finally:
        dist.destroy_process_group()


def _spawn_two(worker, args, seconds, label):
    context = spawn(worker, args=args, nprocs=2, join=False)
    deadline = time.monotonic() + seconds
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise AssertionError(f"two-rank {label} check exceeded {seconds} seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def _run_two_ranks(transport, partition, case, tmp_path, seconds):
    rendezvous = (tmp_path / f"{transport}-{partition}-rendezvous").resolve().as_uri()
    _spawn_two(_distributed_worker, (transport, partition, rendezvous, case), seconds, transport)


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(),
                    reason="CPU distributed check requires Gloo")
@pytest.mark.parametrize("partition", ["views", "space"])
def test_two_rank_gloo_cpu_matches_single_process(partition, tmp_path):
    _run_two_ranks("gloo", partition, DISTRIBUTED_CASE, tmp_path, seconds=120)


@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2 or not dist.is_available()
                    or not dist.is_nccl_available(), reason="two GPUs and NCCL are required")
@pytest.mark.parametrize("partition", ["views", "space"])
def test_two_rank_nccl_each_rank_uses_its_gpu_and_matches_single_process(partition, tmp_path):
    _run_two_ranks("nccl", partition, DISTRIBUTED_CUDA_CASE, tmp_path, seconds=180)


def _backward_memory_worker(rank, rendezvous):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=120))
    try:
        # Views partition: backproject returns a replicated volume, so its
        # backward reduces the volume cotangent across the two ranks.
        case = _Case("cone", (128, 512, 512), (384, 256), 16, (0.5, 0.25, 0.25), "float32")
        operator = _projector(case, geometry_device=device, devices=[rank],
                              distributed=True, partition="views")
        generator = torch.Generator(device).manual_seed(11 + rank)
        sinogram = torch.rand(operator.projection_shape, device=device, generator=generator)
        sinogram.requires_grad_()
        cotangent = torch.rand(operator.volume_shape, device=device, generator=generator)
        cotangent /= operator.view_world_size
        with _allocator_room(device, 1 << 30):
            adjoint = operator.backproject(sinogram)
            usable = _usable_cuda_bytes(device)
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            adjoint.backward(cotangent)
            torch.cuda.synchronize(device)
            extra = (torch.cuda.max_memory_allocated(device) - baseline
                     - sinogram.grad.numel() * sinogram.grad.element_size())
        volume_bytes = cotangent.numel() * cotangent.element_size()
        _assert_within_bound(extra, usable, f"rank {rank} backproject backward "
                                            f"(volume {volume_bytes} bytes)")
    finally:
        dist.destroy_process_group()


@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2 or not dist.is_available()
                    or not dist.is_nccl_available(), reason="two GPUs and NCCL are required")
def test_two_rank_nccl_backproject_backward_stays_within_the_bound(tmp_path):
    # Backward may keep one reduced tile at a time, not a whole extra volume.
    rendezvous = (tmp_path / "backward-memory-rendezvous").resolve().as_uri()
    _spawn_two(_backward_memory_worker, (rendezvous,), 300, "nccl backward memory")


# --------------------------------------------------------------------------
# C2: same chunking, same bits
# --------------------------------------------------------------------------

def _plan_from_stats(stats):
    keys = ("chunk_shape", "view_chunk_size", "detector_chunk_shape")
    if stats is None or any(stats.get(key) is None for key in keys):
        pytest.skip("last_execution_stats does not report the automatic portable plan")
    return dict(volume_chunk_shape=tuple(stats["chunk_shape"]),
                view_chunk_size=int(stats["view_chunk_size"]),
                detector_chunk_shape=tuple(stats["detector_chunk_shape"]))


def _project_once(case, device, **options):
    surface = None
    if case.surface:
        _, sdd, _ = case.geometry()
        surface = ParameterizedSurface(_cylinder, (torch.tensor(sdd, dtype=torch.float64),))
    operator = _projector(case, geometry_device=device, surface=surface, **options)
    volume, _ = _data(case, device)
    with torch.no_grad():
        return operator.project(volume), operator


# Names ending in "-auto" run the automatic plan (many ray batches on CPU);
# the others use explicit small chunks, so every ray batch meets several
# volume tiles and projection ray reuse is exercised.
C2_CPU_CASES = {
    "cone-anisotropic-float32-auto": _Case("cone", (40, 96, 96), (96, 64), 12, (1.1, 0.9, 0.8),
                                           "float32"),
    "cone-anisotropic-float32": _Case("cone", (7, 9, 10), (11, 8), 5, (1.1, 0.9, 0.8), "float32"),
    "cone-surface-float64": _Case("cone", (7, 9, 10), (11, 8), 5, (1.1, 0.9, 0.8), "float64",
                                  surface=True),
    "fan-anisotropic-float32": _Case("fan", (13, 15), (17,), 5, (0.9, 0.7), "float32"),
}


# Bitwise checks stay on CPU: the docs do not promise bitwise determinism of
# floating-point scatter accumulation on accelerators.
@pytest.mark.parametrize("name", list(C2_CPU_CASES))
def test_cpu_project_with_the_automatic_plan_as_explicit_hints_is_bitwise_identical(
    name, single_cpu_thread,
):
    case = C2_CPU_CASES[name]
    automatic, operator = _project_once(case, "cpu")
    plan = _plan_from_stats(operator.last_execution_stats)
    explicit, _ = _project_once(case, "cpu", **plan)
    assert torch.equal(automatic, explicit), (
        f"{name}: explicit hints equal to the automatic plan {plan} changed the bits"
    )


RAY_SETUP_CASES = {
    "cone-anisotropic-float32": _Case("cone", (7, 9, 10), (11, 8), 5, (1.1, 0.9, 0.8), "float32"),
    "cone-float64": _Case("cone", (7, 9, 10), (11, 8), 5, (1.0, 1.0, 1.0), "float64"),
    "fan-anisotropic-float32": _Case("fan", (13, 15), (17,), 5, (0.9, 0.7), "float32"),
}


@pytest.mark.parametrize("name", list(RAY_SETUP_CASES))
def test_project_sets_up_each_ray_batch_once_for_all_tiles(name, monkeypatch):
    # Spy on the portable ray-setup step. A flat detector in one process has
    # no separate validation pass, so every call belongs to the numerical pass.
    case = RAY_SETUP_CASES[name]
    chunks = case.small_chunks()
    operator = _projector(case, **chunks)
    volume, _ = _data(case, "cpu")
    calls = []
    original = backends.ReferenceProjector._rays

    def spy(self, views, pixels, *args, **kwargs):
        calls.append((views.start, views.stop, tuple((part.start, part.stop) for part in pixels)))
        return original(self, views, pixels, *args, **kwargs)

    monkeypatch.setattr(backends.ReferenceProjector, "_rays", spy)
    with torch.no_grad():
        operator.project(volume)
    tiles = math.prod(math.ceil(size / chunk) for size, chunk
                      in zip(case.shape, chunks["volume_chunk_shape"]))
    batches = math.ceil(case.views / chunks["view_chunk_size"]) * math.prod(
        math.ceil(size / chunk) for size, chunk in zip(case.detector, chunks["detector_chunk_shape"])
    )
    assert tiles > 1 and batches > 1
    assert operator.last_execution_stats["ray_tile_pairs"] == tiles * batches
    assert len(calls) == batches and len(set(calls)) == batches, (
        f"{len(calls)} ray setups for {batches} ray batches x {tiles} tiles; "
        f"expected one setup per ray batch"
    )


@pytest.mark.parametrize("name", list(C2_CPU_CASES))
def test_cpu_project_is_bitwise_repeatable(name, single_cpu_thread):
    case = C2_CPU_CASES[name]
    chunks = case.small_chunks() if not name.endswith("-auto") else {}
    first, operator = _project_once(case, "cpu", **chunks)
    second, _ = _project_once(case, "cpu", **chunks)
    third = operator.project(_data(case, "cpu")[0]).detach()
    assert torch.equal(first, second) and torch.equal(first, third)


ADJOINT_CASES = {
    "fan-anisotropic": _Case("fan", (13, 15), (17,), 5, (0.9, 0.7), "float64"),
    "cone-anisotropic": _Case("cone", (7, 9, 10), (11, 8), 5, (1.1, 0.9, 0.8), "float64"),
    "cone-surface": _Case("cone", (7, 9, 10), (11, 8), 5, (1.1, 0.9, 0.8), "float64", surface=True),
}


@pytest.mark.parametrize("chunking", ["automatic", "small"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda, requires_cuda])])
@pytest.mark.parametrize("name", list(ADJOINT_CASES))
def test_tiled_projection_is_the_adjoint_of_backprojection(name, device, chunking):
    case = ADJOINT_CASES[name]
    surface = None
    if case.surface:
        _, sdd, _ = case.geometry()
        surface = ParameterizedSurface(_cylinder, (torch.tensor(sdd, dtype=torch.float64),))
    options = case.small_chunks() if chunking == "small" else {}
    operator = _projector(case, geometry_device=device, surface=surface, **options)
    volume, sinogram = _data(case, device, seed=77)
    with torch.no_grad():
        forward = torch.dot(operator.project(volume).flatten(), sinogram.flatten()).item()
        adjoint = torch.dot(volume.flatten(), operator.backproject(sinogram).flatten()).item()
    assert abs(forward) > 1e-3
    assert abs(forward - adjoint) <= FLOAT64_TOLERANCE * max(abs(forward), abs(adjoint)), (
        f"<A x, y> = {forward!r} but <x, A^T y> = {adjoint!r}"
    )


# SHA-256 digests of projections from the implementation before this change
# (main 5cde9f0), one CPU thread. Bits can depend on the CPU, the BLAS/vector
# code paths and the PyTorch version, so only a matching platform compares.
PRE_CHANGE_DIGESTS = {
    "Darwin-arm64-torch2.10.0": {
        "cone-anisotropic-float32-auto": "f76bb5af13d1422efff2bae28f5a27e727355bdff8800e990241ed108419217b",
        "cone-anisotropic-float32": "8eccb5487f7de46e857e785d9347dbebb80bb3ed5ec27b842edc6ffd746cb719",
        "cone-surface-float64": "b8b60d95ddc6f28549a11684efb979bf9c0fb24a43f04c91336014a5167b8c7b",
        "fan-anisotropic-float32": "2db033fdd4f85ac22059b5128c4eae7e51baed89f983cbd4f5cc92c0ebbc66ac",
    },
}


def _platform_key():
    return f"{platform.system()}-{platform.machine()}-torch{torch.__version__}"


def _digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@pytest.mark.parametrize("name", list(C2_CPU_CASES))
def test_cpu_project_bits_equal_the_pre_change_implementation(name, single_cpu_thread):
    recorded = PRE_CHANGE_DIGESTS.get(_platform_key())
    if recorded is None:
        pytest.skip(f"no pre-change digests recorded for {_platform_key()}")
    case = C2_CPU_CASES[name]
    chunks = case.small_chunks() if not name.endswith("-auto") else {}
    output, _ = _project_once(case, "cpu", **chunks)
    assert _digest(output) == recorded[name], (
        f"{name}: projection bits differ from the pre-change implementation for the same chunking"
    )


# --------------------------------------------------------------------------
# C3: auto-fallback warning
# --------------------------------------------------------------------------

_SMALL = _Case("cone", (7, 9, 11), (10, 8), 5, (1.0, 1.0, 1.0), "fast")
_REASON_OPTIONS = {
    "voxel_spacing": dict(voxel_spacing=(1.1, 0.9, 0.8)),
    "precision-float32": dict(precision="float32"),
    "precision-float64": dict(precision="float64"),
    "hybrid": dict(distributed=True, partition="hybrid", partition_grid=(1, 1)),
}
_REASON_WORDS = {"voxel_spacing": "voxel_spacing", "precision-float32": "precision",
                 "precision-float64": "precision", "hybrid": "hybrid"}
_TORCH_HINT = re.compile(r"""backend\s*=\s*["']torch["']""")


@pytest.fixture
def single_rank_group(tmp_path):
    if dist.is_initialized():
        pytest.skip("a default process group already exists in this process")
    dist.init_process_group("gloo", init_method=(tmp_path / "warning-group").resolve().as_uri(),
                            rank=0, world_size=1, timeout=timedelta(seconds=30))
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.fixture(params=["simulated-nvidia", pytest.param("real-nvidia", marks=[pytest.mark.cuda,
                                                                               requires_nvidia])])
def nvidia_platform(request, monkeypatch):
    if request.param == "simulated-nvidia":
        # Dispatch simulation only: these configurations select the portable
        # backend, whose CPU execution needs no CUDA runtime.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.version, "hip", None)
    return request.param


def _construct(backend, reasons, **extra):
    options = {}
    for reason in reasons:
        options.update(_REASON_OPTIONS[reason])
    spacing = options.pop("voxel_spacing", _SMALL.spacing[0])
    precision = options.pop("precision", "fast")
    _, _, pitch = _SMALL.geometry()
    return Projector(_trajectory(_SMALL), _SMALL.shape, _SMALL.detector, beam="cone",
                     detector_spacing=pitch, voxel_spacing=spacing, backend=backend,
                     precision=precision, **options, **extra)


def _user_warnings(records):
    return [record for record in records if issubclass(record.category, UserWarning)]


REASON_SETS = [("voxel_spacing",), ("precision-float32",), ("precision-float64",), ("hybrid",),
               ("voxel_spacing", "precision-float64"), ("voxel_spacing", "precision-float32", "hybrid")]


@pytest.mark.parametrize("reasons", [value for value in REASON_SETS if value != ("voxel_spacing",)],
                         ids=lambda value: "+".join(value))
def test_auto_fallback_on_nvidia_warns_once_per_instance_and_names_each_reason(
    reasons, nvidia_platform, request,
):
    if "hybrid" in reasons:
        request.getfixturevalue("single_rank_group")
    for instance in range(2):
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            operator = _construct("auto", reasons)
        found = _user_warnings(records)
        assert operator.backend == "torch"
        assert len(found) == 1, (
            f"instance {instance}: expected one UserWarning, got {[str(r.message) for r in found]}"
        )
        message = str(found[0].message)
        for reason in reasons:
            if reason == "voxel_spacing":
                assert _REASON_WORDS[reason] not in message, f"spacing is not a fallback reason: {message}"
            else:
                assert _REASON_WORDS[reason] in message, f"warning does not name {reason!r}: {message}"
        assert _TORCH_HINT.search(message), f"warning does not mention backend='torch': {message}"
        volume = torch.ones(operator.volume_shape)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            sinogram = operator.project(volume)
            operator.backproject(sinogram)
        assert not _user_warnings(records), (
            f"calls repeated the fallback warning: {[str(r.message) for r in records]}"
        )


@pytest.mark.parametrize("reasons", REASON_SETS, ids=["+".join(value) for value in REASON_SETS])
def test_explicit_torch_backend_never_warns(reasons, nvidia_platform, request):
    if "hybrid" in reasons:
        request.getfixturevalue("single_rank_group")
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        operator = _construct("torch", reasons)
        operator.project(torch.ones(operator.volume_shape))
    assert operator.backend == "torch"
    assert not _user_warnings(records), [str(record.message) for record in records]


@pytest.mark.parametrize("platform_name", ["no-cuda", "rocm"])
@pytest.mark.parametrize("reasons", REASON_SETS, ids=["+".join(value) for value in REASON_SETS])
def test_auto_fallback_without_nvidia_cuda_does_not_warn(reasons, platform_name, monkeypatch, request):
    if platform_name == "no-cuda":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    else:
        # Dispatch simulation of a PyTorch ROCm build; no HIP execution occurs.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.version, "hip", "7.0-test")
    if "hybrid" in reasons:
        request.getfixturevalue("single_rank_group")
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        operator = _construct("auto", reasons)
    assert operator.backend == "torch"
    assert not _user_warnings(records), [str(record.message) for record in records]


@pytest.mark.cuda
@requires_nvidia
@pytest.mark.parametrize("backend", ["auto", "numba"])
@pytest.mark.parametrize("reasons", [(), ("voxel_spacing",)], ids=["scalar-spacing", "per-axis-spacing"])
def test_native_selection_on_nvidia_does_not_warn(backend, reasons):
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        operator = _construct(backend, reasons)
    assert operator.backend == "numba"
    assert not _user_warnings(records), [str(record.message) for record in records]
