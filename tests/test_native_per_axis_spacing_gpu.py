"""Real-NVIDIA portions of N2--N4 and unchanged native memory contracts.

Timing, NCCL, multiple devices and large geometry are deliberately skipped
under CUDASIM. No test submits work or starts an external process remotely.
N4 uses scalar spacing .7 and the median of seven warmed, synchronized,
end-to-end calls. Alternating measurement order limits drift between cases.
"""

from datetime import timedelta
import math
import statistics
import time
from unittest import mock

import pytest
import torch
import torch.distributed as dist
from numba.core import config
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from diffct import Projector
from tests._multiprocessing import spawn
from tests._native_per_axis_spacing import (
    BEAMS, GEOMETRY_ATOL, GEOMETRY_RTOL, REAL_CUDA_REQUIRED,
    case, close, data, native_only, operator, surface,
)


pytestmark = [pytest.mark.cuda, REAL_CUDA_REQUIRED]
TWO_GPUS = pytest.mark.skipif(
    config.ENABLE_CUDASIM or not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="two real NVIDIA CUDA devices are required",
)
NCCL = pytest.mark.skipif(
    not dist.is_available() or not dist.is_nccl_available(), reason="NCCL is required",
)


@TWO_GPUS
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("kind", ["flat", "parameterized"])
def test_n2_multi_device_values_data_and_geometry_gradients(beam, kind):
    scan = case(beam, learnable=True)
    detector, parameters = surface(kind, learnable=True)
    image, rays = data(scan.shape), data(scan.projection_shape, phase=.31)
    reference = operator(scan, backend="torch", precision="float64", detector_surface=detector)
    for operation, value, weight in (("project", image, rays), ("backproject", rays, image)):
        reference_data = value.double().requires_grad_()
        expected = getattr(reference, operation)(reference_data)
        wanted = torch.autograd.grad((expected * weight.double()).sum(),
                                     (reference_data, *scan.trajectory, *parameters))
        native_data = value.cuda(0).requires_grad_()
        with native_only() as bridge:
            native = operator(scan, detector_surface=detector, devices=[0, 1])
            bridge.reset_mock()
            actual = getattr(native, operation)(native_data)
            gradients = torch.autograd.grad((actual * weight.cuda(0)).sum(),
                                           (native_data, *scan.trajectory, *parameters))
            assert native.backend == "numba" and bridge.call_count > 0
            used = {call.args[0].device.index for call in bridge.call_args_list if call.args[0].is_cuda}
            assert used == {0, 1}, "both requested devices must execute native work"
        close(actual, expected, kind=kind)
        close(gradients[0], wanted[0], kind=kind)
        for actual_gradient, expected_gradient in zip(gradients[1:], wanted[1:]):
            close(actual_gradient, expected_gradient, rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL)


def _nccl_worker(rank, rendezvous, beam, kind, partition):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        scan = case(beam, learnable=True)
        detector, parameters = surface(kind, learnable=True)
        image, rays = data(scan.shape), data(scan.projection_shape, phase=.31)
        reference = operator(scan, backend="torch", precision="float64", detector_surface=detector)
        for operation, full_data, full_weight in (("project", image, rays), ("backproject", rays, image)):
            reference_data = full_data.double().requires_grad_()
            expected = getattr(reference, operation)(reference_data)
            expected_gradients = torch.autograd.grad((expected * full_weight.double()).sum(),
                                                    (reference_data, *scan.trajectory, *parameters))
            with native_only() as bridge:
                native = Projector(
                    scan.trajectory, scan.shape, scan.detector, beam=beam, backend="numba",
                    voxel_spacing=scan.spacing, detector_spacing=scan.pitch if len(scan.pitch) == 2 else scan.pitch[0],
                    detector_surface=detector, devices=[rank], distributed=True, partition=partition,
                )
                input_slice = native.volume_slice if operation == "project" else native.view_slice
                output_slice = native.view_slice if operation == "project" else native.volume_slice
                value = full_data[input_slice].clone().to(rank).requires_grad_()
                weight = full_weight[output_slice].to(rank)
                actual = getattr(native, operation)(value)
                replicas = native.space_world_size if operation == "project" else native.view_world_size
                gradients = torch.autograd.grad((actual * weight).sum() / replicas,
                                               (value, *scan.trajectory, *parameters))
                assert native.backend == "numba" and bridge.call_count > 0
            close(actual, expected[output_slice], kind=kind)
            close(gradients[0], expected_gradients[0][input_slice], kind=kind)
            for actual_gradient, expected_gradient in zip(gradients[1:], expected_gradients[1:]):
                close(actual_gradient, expected_gradient, rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL)
    finally:
        dist.destroy_process_group()


@TWO_GPUS
@NCCL
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("kind", ["flat", "parameterized"])
@pytest.mark.parametrize("partition", ["views", "space"])
def test_n2_nccl_views_and_space_values_and_autograd(tmp_path, beam, kind, partition):
    rendezvous = (tmp_path / "per-axis-nccl").resolve().as_uri()
    context = spawn(_nccl_worker, args=(rendezvous, beam, kind, partition), nprocs=2, join=False)
    deadline = time.monotonic() + 150
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise AssertionError("per-axis NCCL execution exceeded 150 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def _large_operator(spacing):
    views = 24
    angle = .217 + torch.arange(views, dtype=torch.float32) * (2 * math.pi / views)
    zero = torch.zeros_like(angle)
    radial = torch.stack((angle.cos(), angle.sin(), zero), -1)
    tangent = torch.stack((-angle.sin(), angle.cos(), zero), -1)
    vertical = torch.stack((zero, zero, torch.ones_like(angle)), -1)

    def cylinder(u, v):
        radius = 1040.
        return torch.stack((radius * torch.sin(u / radius), v, radius * (1 - torch.cos(u / radius))), -1)

    return Projector((-600 * radial, 440 * radial, tangent, vertical), (81, 512, 512), (888, 625),
                     # Keep the vertical field inside both physical boxes.
                     # Timing must compare rays with the same hit coverage.
                     beam="cone", voxel_spacing=spacing, detector_spacing=(.7, .05),
                     detector_surface=cylinder, backend="auto", devices=[0], distributed=False)


def test_n4_cylindrical_project_within_twenty_percent_of_scalar():
    image = torch.ones((81, 512, 512), device="cuda:0")
    with native_only() as bridge:
        scalar = _large_operator(.7)
        anisotropic = _large_operator((2.5, .7, .7))
        assert scalar.backend == anisotropic.backend == "numba"
        # Warm both specializations and plans before measuring complete calls.
        for native in (scalar, anisotropic):
            for _ in range(2):
                native.project(image)
        torch.cuda.synchronize()
        samples = {"scalar": [], "per-axis": []}
        for repetition in range(7):
            order = (("scalar", scalar), ("per-axis", anisotropic))
            if repetition % 2:
                order = order[::-1]
            for name, native in order:
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = native.project(image)
                torch.cuda.synchronize()
                samples[name].append(time.perf_counter() - start)
                assert result.shape == (24, 888, 625) and torch.isfinite(result).all()
        assert bridge.call_count > 0
    scalar_seconds = statistics.median(samples["scalar"])
    per_axis_seconds = statistics.median(samples["per-axis"])
    ratio = per_axis_seconds / scalar_seconds
    assert .8 <= ratio <= 1.2, (scalar_seconds, per_axis_seconds, ratio, samples)


class _HostGeometryAllocations(TorchDispatchMode):
    """Measure new FP64 host buffers without replacing tensor arithmetic."""

    def __init__(self):
        super().__init__()
        self.largest = 0

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        inputs = {value.untyped_storage()._cdata for value in tree_flatten((args, kwargs))[0]
                  if isinstance(value, torch.Tensor)}
        result = function(*args, **(kwargs or {}))
        for value in tree_flatten(result)[0]:
            if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.float64:
                continue
            storage = value.untyped_storage()
            if storage._cdata not in inputs:
                self.largest = max(self.largest, storage.nbytes())
                assert storage.nbytes() <= 8 * 1024 ** 2, (str(function), tuple(value.shape), storage.nbytes())
        return result


@pytest.mark.parametrize("spacing", [.7, (2.5, .7, .7)], ids=["unchanged-scalar", "per-axis"])
def test_unchanged_native_host_geometry_limit_and_memory_admission(spacing):
    from diffct import operators, pipeline
    from diffct.chunking import _host_geometry_bytes

    assert pipeline._GEOMETRY_WORK_BYTES == 8 * 1024 ** 2
    original = Projector._surface_geometry_into
    records = []

    def observed(owner, geometry, spec, views, pixels, centre, targets, pipe, groups, offsets=None):
        count = views.stop - views.start
        sizes = tuple(len(range(*selected.indices(size))) for selected, size in zip(pixels, owner.detector_shape))
        host_bytes = _host_geometry_bytes(owner.beam, count, sizes, owner._host_world_points(spec))
        assert host_bytes <= 8 * 1024 ** 2
        assert pipe.stats["estimated_gpu_bytes"] <= pipe.stats["gpu_budget_bytes"]
        assert pipe.stats["estimated_reserved_bytes"] <= pipe.stats["reservation_budget_bytes"]
        with _HostGeometryAllocations() as allocations:
            result = original(owner, geometry, spec, views, pixels, centre, targets, pipe, groups, offsets)
        records.append((host_bytes, allocations.largest))
        return result

    # Observe allocations and production admission; do not replace the planner.
    with native_only(), mock.patch.object(operators.Projector, "_surface_geometry_into", observed):
        native = _large_operator(spacing)
        image = torch.ones((81, 512, 512), device="cuda:0")
        rays = torch.ones((24, 888, 625), device="cuda:0")
        for method, value in (("project", image), ("backproject", rays)):
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated(0)
            torch.cuda.reset_peak_memory_stats(0)
            result = getattr(native, method)(value)
            torch.cuda.synchronize()
            working_peak = torch.cuda.max_memory_allocated(0) - before
            output_bytes = result.numel() * result.element_size()
            assert working_peak <= native.last_execution_stats["gpu_budget_bytes"] + output_bytes
            del result
        assert native.backend == "numba"
    assert records, "large native geometry must use bounded batches"
