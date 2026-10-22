"""CPU regressions for surface workspace admission and validation order."""

import importlib
import math
from unittest import mock

import pytest
import torch
from numba.core import config

from diffct import ParameterizedSurface, Projector, operators
from diffct.surfaces import _sample_offsets
from tests.test_detector_surfaces import _case, _projector
from tests.test_surface_geometry_cudasim import (
    _CpuGeometryPipe,
    _CudaOffsetMetadata,
    _cpu_geometry_transport,
)
from tests.test_surface_pipeline_contracts import _native_plan_budget


class _CompositionObserved(Exception):
    pass


@pytest.mark.skipif(not config.ENABLE_CUDASIM,
                    reason="CPU composition requires NUMBA_ENABLE_CUDASIM=1")
def test_streamed_geometry_gradient_composition_respects_fallback_workspace():
    case = _case("cone", views=2, learnable=True)
    case.shape, case.detector = (3, 4, 5), (16, 32)
    offsets = torch.zeros((case.views, *case.detector, 3), dtype=torch.float64)
    offsets[..., 0] = torch.linspace(-1., 1., case.detector[0])[:, None]
    offsets[..., 1] = torch.linspace(-1., 1., case.detector[1])
    offsets.requires_grad_()
    projector = _projector(case, lambda u, v: offsets, backend="torch", view_chunk_size=2)
    projector._devices = (torch.device("cuda:0"),)
    spec, geometry = projector._effective_geometry(None, torch.device("cpu"))
    assert projector._geometry_trainable and spec.kind == "offsets" and not spec.shared

    with _native_plan_budget(50000):
        plan = projector._execution_plan(torch.ones(case.shape), True, geometry_spec=spec)
        assert plan is not None and not plan.surface_batch
        assert plan.view_chunk_size == 2 and plan.detector_chunk_shape == case.detector
        admitted = projector._surface_work_buffers(
            plan.view_chunk_size, plan.detector_chunk_shape, spec=spec,
            streamed=True, batched=plan.surface_batch,
        )

    pipe = _CpuGeometryPipe()
    transfer = mock.MagicMock()
    transfer.__enter__.return_value.for_device.return_value = pipe

    def ray_job(source, native, geometry, centre, local, views, pixels, spec,
                pipe, groups, index, is_project, **kwargs):
        count, rank = views.stop - views.start, len(case.shape)
        targets = (torch.empty(count * rank),
                   torch.empty(count * math.prod(part.stop - part.start for part in pixels) * rank))
        # Run the production compositor with groups supplied by backward replay.
        # Stop before the CT VJP, which requires real CUDA tensor storage.
        projector._surface_geometry_into(
            geometry, spec, views, pixels, centre, targets, pipe, groups,
            offsets=kwargs["sampled_offsets"],
        )
        allocated = tuple((value.numel() * value.element_size() + 511) // 512 * 512
                          for value in pipe.allocations)
        assert allocated, "backward replay did not allocate composition storage"
        assert max(allocated) <= max(admitted), (allocated, admitted)
        assert sum(allocated) <= sum(admitted), (allocated, admitted)
        raise _CompositionObserved

    kernel = importlib.import_module("diffct.kernels.surface_geometry").compose_surface
    with _cpu_geometry_transport(kernel), \
            mock.patch.object(operators._pipeline, "_TransferPipeline", return_value=transfer), \
            mock.patch.object(projector, "_native_tile", return_value=torch.ones(case.shape)), \
            mock.patch.object(projector, "_ray_job", side_effect=ray_job), \
            pytest.raises(_CompositionObserved):
        projector._geometry_grad_streamed(
            torch.ones(case.shape), torch.ones(case.sino_shape), geometry, plan, spec,
        )


def test_surface_workspace_covers_cuda_fp64_sampler_cotangent_and_allocation_limit(monkeypatch):
    case = _case("fan", views=12000)
    case.detector = (5,)
    gain = torch.tensor(.01, dtype=torch.float64, requires_grad=True)

    def sampler(u, v, ids, gain):
        tangent = u.expand(len(ids), *u.shape)
        return torch.stack((tangent, torch.zeros_like(tangent), gain * tangent.square()), -1)

    surface = ParameterizedSurface(sampler, (gain,))
    projector = _projector(
        case, surface, backend="torch", view_chunk_size=case.views,
        detector_chunk_shape=case.detector,
    )
    projector._devices = (torch.device("cuda:0"),)
    # Retain CPU storage while exposing the CUDA sampler's output metadata.
    monkeypatch.setattr(surface, "sampler", lambda *args: sampler(*args).as_subclass(_CudaOffsetMetadata))
    spec, geometry = projector._effective_geometry(None, torch.device("cpu"))
    offsets = _sample_offsets(
        case.beam, case.detector, case.pitch, geometry, spec,
        slice(0, case.views), (slice(0, case.detector[0]),),
    )
    assert projector._geometry_trainable and offsets.is_cuda and offsets.dtype == torch.float64
    cotangent_bytes = offsets.numel() * offsets.element_size()
    allocation_limit = 1024 ** 2
    assert cotangent_bytes > allocation_limit

    with _native_plan_budget(64 * 1024 ** 2):
        buffers = projector._surface_work_buffers(
            case.views, case.detector, spec=spec, streamed=True, batched=True,
        )
        assert max(buffers) >= cotangent_bytes, (buffers, cotangent_bytes)
        with pytest.raises(RuntimeError, match="memory|budget|minimum|fit|limit"):
            projector._execution_plan(torch.ones(case.shape), True, geometry_spec=spec)


def test_trajectory_nonfinite_error_precedes_later_shape_error():
    case = _case("fan", views=2)
    source, centre, axis = case.trajectory
    source[0, 0] = torch.nan
    trajectory = (source, centre[:, :1], axis)
    with pytest.raises(ValueError, match="^trajectory tensors must contain only finite values$"):
        Projector(trajectory, case.shape, case.detector, beam="fan", backend="torch")


@pytest.mark.parametrize("budget,reservation_budget,admitted", [
    pytest.param(16_799_000, 22_400_000, True, id="reported-admission"),
    pytest.param(16_796_672, 18_874_368, True, id="exact-phase-peaks"),
    pytest.param(16_796_671, 18_874_368, False, id="total-one-byte-short"),
    pytest.param(16_796_672, 18_874_367, False, id="reservation-one-byte-short"),
])
def test_streamed_trainable_shared_surface_admission_checks_phase_peaks(
        budget, reservation_budget, admitted):
    case = _case("fan", views=100, learnable=True)
    case.shape, case.detector = (2048, 2048), (5,)

    def surface(u, v):
        return torch.stack((u, v, torch.zeros_like(u)), -1)

    projector = _projector(
        case, surface, backend="torch", volume_chunk_shape=case.shape,
        view_chunk_size=case.views, detector_chunk_shape=case.detector,
    )
    projector._devices = (torch.device("cuda:0"),)
    spec, _ = projector._effective_geometry(None, torch.device("cpu"))
    assert projector._geometry_trainable and spec.kind == "offsets" and spec.shared
    unlimited = (1 << 63) - 1

    with _native_plan_budget(budget), \
            mock.patch.object(operators, "_memory_budget", return_value=(budget, unlimited)), \
            mock.patch.object(operators, "_reservation_budget", return_value=reservation_budget):
        if not admitted:
            with pytest.raises(RuntimeError, match="memory|budget|minimum|fit|limit"):
                projector._execution_plan(torch.empty(case.shape), True, geometry_spec=spec)
            return
        plan = projector._execution_plan(torch.empty(case.shape), True, geometry_spec=spec)
        workspace = projector._surface_work_buffers(case.views, case.detector, spec=spec, streamed=True)

    assert plan is not None and plan.surface_batch
    assert plan.chunk_shape == case.shape
    assert plan.view_chunk_size == case.views and plan.detector_chunk_shape == case.detector
    # The 16,789,504-byte pipeline outlives both workspaces. Their peaks are
    # 5,632 bytes for composition and 7,168 bytes for gradient replay.
    assert sum(workspace) == max(5_632, 7_168)
    assert max(workspace) == max(5_120, 512)
    assert plan.estimated_gpu_bytes == 16_789_504 + max(5_632, 7_168)
    assert plan.estimated_gpu_bytes <= plan.gpu_budget_bytes == budget
    assert plan.estimated_reserved_bytes == 18_874_368
    assert plan.estimated_reserved_bytes <= plan.reservation_budget_bytes == reservation_budget
    assert plan.allocation_limit == unlimited


@pytest.mark.parametrize("allocation_limit,reservation_budget,admitted", [
    pytest.param(1_440_256, 23_068_672, True, id="exact-phase-reservation-and-allocation"),
    pytest.param(1_440_256, 23_068_671, False, id="reservation-one-byte-short"),
    pytest.param(1_440_255, 23_068_672, False, id="sampler-cotangent-one-byte-short"),
])
def test_streamed_cuda_sampler_reservation_uses_each_real_phase(
        monkeypatch, allocation_limit, reservation_budget, admitted):
    case = _case("fan", views=12000)
    case.shape, case.detector = (400, 400), (5,)
    gain = torch.tensor(.01, dtype=torch.float64, requires_grad=True)

    def sampler(u, v, ids, gain):
        tangent = u.expand(len(ids), *u.shape)
        return torch.stack((tangent, torch.zeros_like(tangent), gain * tangent.square()), -1)

    surface = ParameterizedSurface(sampler, (gain,))
    projector = _projector(
        case, surface, backend="torch", volume_chunk_shape=case.shape,
        view_chunk_size=case.views, detector_chunk_shape=case.detector,
    )
    projector._devices = (torch.device("cuda:0"),)
    monkeypatch.setattr(surface, "sampler", lambda *args: sampler(*args).as_subclass(_CudaOffsetMetadata))
    spec, geometry = projector._effective_geometry(None, torch.device("cpu"))
    offsets = _sample_offsets(
        case.beam, case.detector, case.pitch, geometry, spec,
        slice(0, case.views), (slice(0, case.detector[0]),),
    )
    assert projector._geometry_trainable and offsets.is_cuda and offsets.dtype == torch.float64
    budget = 64 * 1024 ** 2

    with _native_plan_budget(budget), \
            mock.patch.object(operators, "_memory_budget", return_value=(budget, allocation_limit)), \
            mock.patch.object(operators, "_reservation_budget", return_value=reservation_budget):
        if not admitted:
            with pytest.raises(RuntimeError, match="memory|budget|minimum|fit|limit"):
                projector._execution_plan(torch.empty(case.shape), True, geometry_spec=spec)
            return
        plan = projector._execution_plan(torch.empty(case.shape), True, geometry_spec=spec)
        workspace = projector._surface_work_buffers(case.views, case.detector, spec=spec, streamed=True)

    assert plan is not None and plan.surface_batch
    assert plan.chunk_shape == case.shape
    assert plan.view_chunk_size == case.views and plan.detector_chunk_shape == case.detector
    # Composition has the larger total; replay has the larger allocation.
    assert sum(workspace) == max(1_536_512, 1_447_424)
    cotangent_bytes = offsets.numel() * offsets.element_size()
    assert max(workspace) == (cotangent_bytes + 511) // 512 * 512 == allocation_limit
    assert plan.estimated_gpu_bytes == 2_033_152 + max(1_536_512, 1_447_424)
    # Native reservations are 8 MiB for composition and 22 MiB for replay,
    # each including the pipeline. A merged inventory incorrectly needs 24 MiB.
    assert plan.estimated_reserved_bytes == max(8, 22) * 1024 ** 2
    assert plan.estimated_reserved_bytes == plan.reservation_budget_bytes == reservation_budget
