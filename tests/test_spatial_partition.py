"""Fixed slab ownership, real process collectives and independent cell adjoints.

Gloo cases use two real processes sharing GPU 0. NCCL cases require two actual
GPUs. Replicated forward losses count once for Hessians; weighted losses sum
their explicit rank cotangents. No whole-volume gather is a numerical oracle.
"""

from contextlib import ExitStack
from datetime import timedelta
import inspect
import json
import math
from pathlib import Path
import time
from unittest import mock

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils._pytree import tree_flatten

import diffct
from tests.test_batched_surfaces import _Sampler, _numpy_offsets, _settings
from tests.test_block_storage import _GuardedStore, _rejection
from tests.test_chunked_projector import _offsets
from tests.test_detector_surfaces import (
    _case, _close, _data, _finite_difference, _matrix, _numpy, _projector,
)
from tests.test_execution_schedules import _CopyTrace


BEAMS = ("parallel", "fan", "cone")
gloo_required = pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(),
                                 reason="real two-process Gloo is required")
cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _space_projector(case, surface=None, *, partition="space", **kwargs):
    assert "partition" in inspect.signature(diffct.Projector).parameters, \
        "INTERFACE_PENDING: Projector(..., partition='views') declaration"
    operator = _projector(case, surface, partition=partition, **kwargs)
    for name in ("global_volume_shape", "volume_slice", "local_volume_shape"):
        assert hasattr(operator, name), f"INTERFACE_PENDING: Projector.{name} declaration"
    return operator


def _slab(shape, rank, world=2):
    base, extra = divmod(shape[0], world)
    start = rank * base + min(rank, extra)
    return (slice(start, start + base + int(rank < extra)),
            *(slice(0, size) for size in shape[1:]))


def _owned_columns(shape, index):
    return np.arange(math.prod(shape)).reshape(shape)[index].ravel()


def _fixture(beam, first=5, *, views=5, trainable=False):
    case = _case(beam, views=views, learnable=trainable)
    case.shape = (first, 5, 7) if beam == "cone" else (first, 7)
    case.detector = (4, 3) if beam == "cone" else (7,)
    parameters = (torch.tensor(3.7, dtype=torch.float64, requires_grad=trainable),
                  torch.tensor(.037, dtype=torch.float64, requires_grad=trainable),
                  torch.linspace(-.2, .3, views, dtype=torch.float64, requires_grad=trainable))
    return case, parameters


def _run_processes(worker, args, *, timeout=90):
    """Bound hangs and terminate only this test's own worker processes."""
    context = mp.spawn(worker, args=args, nprocs=2, join=False)
    deadline = time.monotonic() + timeout
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise AssertionError(f"two-process collective test exceeded {timeout}s")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


class _OwnedTrace(_CopyTrace):
    def __init__(self, case, chunk, pixels):
        super().__init__(case)
        self.pixels = pixels
        # Base resource guard is independent of the new scheduling policy.
        # This subclass sets its own limits instead of using the base chunk field.
        self.chunk = None
        self.tile_limit = math.prod(chunk)
        self.ray_limit = 2 * math.prod(pixels)

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        inputs = [item for item in tree_flatten(args)[0] if isinstance(item, torch.Tensor)]
        if self.phase in ("forward-backward", "views-cotangent") and str(function) in ("aten.clone.default", "aten._to_copy.default"):
            cotangent_shape = self.case.shape if self.phase == "views-cotangent" else self.case.sino_shape
            for tensor in inputs:
                if tensor.device.type == "cpu" and tuple(tensor.shape) == cotangent_shape:
                    assert tensor.numel() <= max(self.tile_limit, self.ray_limit), \
                        "complete distributed cotangent cloned for communication"
        result = super().__torch_dispatch__(function, types, args, kwargs)
        for tensor in tree_flatten(result)[0]:
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                # Explicit parameter snapshots/returned gradients are separate
                # from staged volume/ray data, as in the geometry contract.
                if tuple(tensor.shape) in (self.case.shape, self.case.sino_shape):
                    assert tensor.numel() <= max(self.tile_limit, self.ray_limit), \
                        "complete global volume/ray data staged on CUDA"
        return result


def _reductions(records, phase, chunk, pixels, parameters, backend):
    original = dist.all_reduce
    limit = max(math.prod(chunk), 2 * math.prod(pixels),
                sum(parameter.numel() for parameter in parameters))

    def observe(tensor, *args, **kwargs):
        operation = kwargs.get("op", args[0] if args else dist.ReduceOp.SUM)
        if tensor.is_floating_point() and operation == dist.ReduceOp.SUM:
            assert tensor.numel() <= limit, "full data collective escaped resident tiles/batches"
            if backend == "nccl":
                assert tensor.is_cuda, "NCCL data reduction staged through CPU"
            records.append((phase[0], tuple(tensor.shape), tensor.numel(), str(tensor.dtype)))
        return original(tensor, *args, **kwargs)
    return mock.patch.object(dist, "all_reduce", observe)


@pytest.mark.parametrize("beam", BEAMS)
def test_views_default_exposes_full_volume_ownership_metadata(beam):
    case, _ = _fixture(beam)
    operator = _space_projector(case, partition="views")
    assert operator.partition == "views"
    assert operator.global_volume_shape == operator.local_volume_shape == operator.volume_shape == case.shape
    assert operator.volume_slice == tuple(slice(0, size) for size in case.shape)
    assert operator.view_slice == slice(0, case.views)


@pytest.mark.parametrize("value", [None, True, 1, "Space", "slabs", ""])
def test_partition_rejects_unknown_or_nonstring_values(value):
    with _rejection("partition|views|space"):
        _space_projector(_fixture("cone")[0], partition=value)


@pytest.mark.parametrize("beam", BEAMS)
def test_single_rank_space_owns_global_frame_and_all_views(beam):
    case, _ = _fixture(beam)
    operator = _space_projector(case)
    assert operator.partition == "space"
    assert operator.global_volume_shape == operator.local_volume_shape == operator.volume_shape == case.shape
    assert operator.volume_slice == _slab(case.shape, 0, 1)
    assert operator.view_slice == slice(0, case.views) and operator.projection_shape == case.sino_shape


def _metadata_worker(rank, beam, first, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        case, _ = _fixture(beam, first)
        operator = _space_projector(case, distributed=True)
        owned = _slab(case.shape, rank)
        local = (owned[0].stop - owned[0].start, *case.shape[1:])
        assert operator.global_volume_shape == case.shape
        assert operator.volume_slice == owned
        assert operator.volume_shape == operator.local_volume_shape == local
        assert operator.view_slice == slice(0, case.views)
        assert operator.projection_shape == case.sino_shape
    finally:
        dist.destroy_process_group()


@gloo_required
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("first", [1, 5], ids=["empty-slab", "uneven-slabs"])
def test_real_gloo_metadata_uses_balanced_global_slabs_and_replicated_rays(beam, first, tmp_path):
    _run_processes(_metadata_worker, (beam, first, (tmp_path / "metadata").resolve().as_uri()))


def _spatial_worker(rank, beam, first, curved, flags, backend, rendezvous, directory):
    torch.cuda.set_device(rank if backend == "nccl" else 0)
    dist.init_process_group(backend, init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        trainable = flags == "all" or rank == 0
        case, parameters = _fixture(beam, first, trainable=trainable)
        sampler = _Sampler(case)
        surface = diffct.ParameterizedSurface(sampler, parameters=parameters) if curved else None
        chunk = (2, 3, 4) if beam == "cone" else (2, 4)
        pixels = (3, 2) if beam == "cone" else (3,)
        operator = _space_projector(case, surface, distributed=True, devices=[torch.cuda.current_device()],
                                    volume_chunk_shape=chunk, detector_chunk_shape=pixels,
                                    view_chunk_size=2)
        owned = _slab(case.shape, rank)
        image, sino = _data(case, device="cpu")
        image_numpy, sino_numpy = _numpy(image), _numpy(sino)
        columns = _owned_columns(case.shape, owned)
        local = image[owned].clone().double().requires_grad_(trainable)
        rays = sino.clone().double().requires_grad_(trainable)
        saved_frames = tuple(_numpy(value) for value in case.trajectory)
        saved_parameters = tuple(_numpy(value) for value in parameters)
        offsets = _numpy_offsets(case, saved_parameters) if curved else _offsets(case, flat=True)
        matrix = _matrix(case, offsets, saved_frames)
        # Caller cotangents prepared outside the communication observer. Keep
        # the loss float32 so engine dtype casts are not misread as staging.
        forward_weight = sino * ((rank + 1) / 2)
        records, phase = [], ["forward"]
        with _reductions(records, phase, chunk, pixels, (*case.trajectory, *parameters), backend), \
                _OwnedTrace(case, chunk, pixels) as trace:
            forward = operator.project(local)
            _close(forward, (matrix @ image_numpy.ravel()).reshape(case.sino_shape))
            assert tuple(forward.shape) == operator.projection_shape == case.sino_shape
            assert forward.requires_grad, "rank without local trainable inputs lost collective backward participation"
            # Invocation snapshots include both values and the callback identity.
            with torch.no_grad():
                case.trajectory[1].add_(.019)
                parameters[1].add_(.023)
            if surface is not None:
                surface.sampler = _Sampler(case, factor=1.7)
            phase[0], trace.phase = "forward-backward", "forward-backward"
            (forward * forward_weight).sum().backward()
            expected_local = (matrix[:, columns].T @ (sino_numpy * 1.5).ravel()).reshape(local.shape)
            if trainable:
                _close(local.grad, expected_local)
            else:
                assert local.grad is None
                assert all(value.grad is None for value in (*case.trajectory, *parameters))
        if trainable:
            direction = np.random.default_rng(711).normal(size=saved_frames[1].shape)
            direction /= np.linalg.norm(direction)

            def forward_objective(step):
                frames = list(saved_frames)
                frames[1] = frames[1] + step * direction
                changed = _matrix(case, offsets, frames)
                return float((sino_numpy * 1.5).ravel() @ changed @ image_numpy.ravel())

            expected = _finite_difference(forward_objective, 0.)
            actual = float((_numpy(case.trajectory[1].grad) * direction).sum())
            assert abs(expected) > 1e-3
            assert actual == pytest.approx(expected, rel=6e-3, abs=6e-4)
            if curved:
                def surface_objective(step):
                    values = list(saved_parameters)
                    values[1] = values[1] + step
                    changed = _matrix(case, _numpy_offsets(case, values), saved_frames)
                    return float((sino_numpy * 1.5).ravel() @ changed @ image_numpy.ravel())
                expected = _finite_difference(surface_objective, 0.)
                assert parameters[1].grad.item() == pytest.approx(expected, rel=6e-3, abs=6e-4)
        with torch.no_grad():
            for component, saved in zip(case.trajectory, saved_frames):
                component.copy_(torch.from_numpy(saved))
            for parameter, saved in zip(parameters, saved_parameters):
                parameter.copy_(torch.as_tensor(saved))
                parameter.grad = None
            for component in case.trajectory:
                component.grad = None
        if surface is not None:
            surface.sampler = sampler
        phase[0] = "backproject"
        with _reductions(records, phase, chunk, pixels, (*case.trajectory, *parameters), backend), \
                _OwnedTrace(case, chunk, pixels) as trace:
            back = operator.backproject(rays)
            _close(back, (matrix[:, columns].T @ sino_numpy.ravel()).reshape(local.shape))
            assert tuple(back.shape) == operator.local_volume_shape == local.shape
            assert back.requires_grad, "empty/nontrainable slab lost backward collective participation"
            weight = image[owned].double() * ((rank + 1) / 2)
            phase[0], trace.phase = "backproject-backward", "backproject-backward"
            (back.double() * weight).sum().backward()
        volume_weight = image_numpy.copy()
        for other in range(2):
            volume_weight[_slab(case.shape, other)] *= (other + 1) / 2
        if trainable:
            _close(rays.grad, (matrix @ volume_weight.ravel()).reshape(case.sino_shape))

            def back_objective(step):
                frames = list(saved_frames)
                frames[1] = frames[1] + step * direction
                return float(sino_numpy.ravel() @ _matrix(case, offsets, frames) @ volume_weight.ravel())

            expected = _finite_difference(back_objective, 0.)
            actual = float((_numpy(case.trajectory[1].grad) * direction).sum())
            assert actual == pytest.approx(expected, rel=6e-3, abs=6e-4)
            if curved:
                def back_surface_objective(step):
                    values = list(saved_parameters)
                    values[1] = values[1] + step
                    changed = _matrix(case, _numpy_offsets(case, values), saved_frames)
                    return float(sino_numpy.ravel() @ changed @ volume_weight.ravel())
                expected = _finite_difference(back_surface_objective, 0.)
                assert parameters[1].grad.item() == pytest.approx(expected, rel=6e-3, abs=6e-4)
        else:
            assert rays.grad is None
            assert all(value.grad is None for value in (*case.trajectory, *parameters))
        # Numerical store path uses rank-local I/O and never creates a global slab.
        source = _GuardedStore(local.detach().float(), chunk)
        result = _GuardedStore(torch.full(case.sino_shape, float("nan")), (2, *pixels))
        phase[0] = "project-into"
        with _reductions(records, phase, chunk, pixels, (*case.trajectory, *parameters), backend):
            assert operator.project_into(source, result) is result
        _close(result.tensor, (matrix @ image_numpy.ravel()).reshape(case.sino_shape))
        assert torch.all(result.overwrites == 1)
        output = _GuardedStore(torch.full(local.shape, float("nan")), chunk)
        phase[0] = "backproject-into"
        with _reductions(records, phase, chunk, pixels, (*case.trajectory, *parameters), backend):
            assert operator.backproject_into(_GuardedStore(sino, (2, *pixels)), output) is output
        _close(output.tensor, (matrix[:, columns].T @ sino_numpy.ravel()).reshape(local.shape))
        assert torch.all(output.overwrites == 1)
        if flags == "all":
            data = local.detach().requires_grad_()
            phase[0] = "project-hessian"
            with _reductions(records, phase, chunk, pixels, (*case.trajectory, *parameters), backend):
                forward = operator.project(data)
                canonical_loss = forward.double().square().sum() / 2 * int(rank == 0)
                gradient, = torch.autograd.grad(canonical_loss, data, create_graph=True)
                direction = torch.linspace(-.3, .6, image.numel()).reshape(case.shape)[owned].double()
                hessian, = torch.autograd.grad((gradient * direction).sum(), data)
            global_direction = np.linspace(-.3, .6, image.numel()).reshape(case.shape)
            _close(gradient, (matrix[:, columns].T @ matrix @ image_numpy.ravel()).reshape(local.shape))
            _close(hessian, (matrix[:, columns].T @ matrix @ global_direction.ravel()).reshape(local.shape))
            data = rays.detach().requires_grad_()
            phase[0] = "backproject-hessian"
            with _reductions(records, phase, chunk, pixels, (*case.trajectory, *parameters), backend):
                back = operator.backproject(data)
                gradient, = torch.autograd.grad(back.double().square().sum() / 2, data, create_graph=True)
                direction = torch.linspace(-.4, .7, sino.numel()).reshape(case.sino_shape).double()
                hessian, = torch.autograd.grad((gradient * direction).sum() * int(rank == 0), data)
            _close(gradient, (matrix @ matrix.T @ sino_numpy.ravel()).reshape(case.sino_shape))
            _close(hessian, (matrix @ matrix.T @ _numpy(direction).ravel()).reshape(case.sino_shape))
        (Path(directory) / f"rank-{rank}.json").write_text(json.dumps(records), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def _check_common_rounds(directory):
    records = [json.loads((Path(directory) / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert records[0] == records[1], "collective order depends on local tile count/empty slab/trainability"
    assert records[0], "no real spatial data SUM observed"
    assert not any(record[0] in ("backproject", "backproject-into") for record in records[0]), \
        "space backprojection unnecessarily SUMs the owned volume"


@pytest.mark.cuda
@cuda_required
@gloo_required
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("curved", [False, True], ids=["flat", "batched-surface"])
@pytest.mark.parametrize("first,flags", [(5, "all"), (5, "mixed"), (1, "mixed")],
                         ids=["uneven-hessian", "mixed-trainability", "empty-nontrainable"])
def test_real_gloo_spatial_forward_adjoint_snapshots_hessians_and_store_ownership(beam, curved, first, flags, tmp_path):
    _run_processes(_spatial_worker, (beam, first, curved, flags, "gloo",
                                   (tmp_path / "spatial").resolve().as_uri(), str(tmp_path)))
    _check_common_rounds(tmp_path)


@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2 or not dist.is_nccl_available(),
                    reason="two actual CUDA GPUs and real NCCL are required")
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("first", [1, 5], ids=["empty-slab", "uneven-slabs"])
def test_real_nccl_spatial_collectives_stay_resident_and_match_empty_rank_protocol(beam, first, tmp_path):
    _run_processes(_spatial_worker, (beam, first, True, "mixed", "nccl",
                                   (tmp_path / "spatial-nccl").resolve().as_uri(), str(tmp_path)))
    _check_common_rounds(tmp_path)


def _agreement_worker(rank, disagreement, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        schedule_mismatch = disagreement in ("views-schedule", "space-schedule")
        case, parameters = _fixture("cone", trainable=rank == 0 and not schedule_mismatch)
        partition = "views" if disagreement in ("views-trainability", "views-schedule") else "space"
        schedule = ("window" if rank == 0 else "spatial") if schedule_mismatch else "auto"
        if disagreement == "partition" and rank:
            partition = "views"
        if disagreement == "shape" and rank:
            case.shape = (6, *case.shape[1:])
        if disagreement == "spacing" and rank:
            case.spacing *= 1.2
        if disagreement == "parameter-layout" and rank:
            parameters = (*parameters, torch.tensor(1.))
        surface = diffct.ParameterizedSurface(lambda u, v, ids, *p: torch.stack((u, v, u * 0), -1),
                                              parameters=parameters)
        with _rejection("agree|metadata|partition|shape|geometry|parameter|spacing|scan|schedule"):
            operator = _space_projector(case, surface, partition=partition, schedule=schedule, distributed=True)
            # Some compatibility checks are necessarily admission-time checks.
            if disagreement != "views-trainability" and not schedule_mismatch:
                operator.project(torch.zeros(operator.volume_shape))
    finally:
        dist.destroy_process_group()


@gloo_required
@pytest.mark.parametrize("disagreement", ["partition", "shape", "spacing", "parameter-layout", "views-trainability",
                                         "views-schedule", "space-schedule"])
def test_distributed_global_settings_reject_mismatch_without_deadlock_and_preserve_views_rule(disagreement, tmp_path):
    _run_processes(_agreement_worker, (disagreement, (tmp_path / "agreement").resolve().as_uri()))


def _views_cotangent_worker(rank, beam, views, rendezvous):
    torch.cuda.set_device(0)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        case, _ = _fixture(beam, views=views)
        chunk = (2, 3, 4) if beam == "cone" else (2, 4)
        pixels = (3, 2) if beam == "cone" else (3,)
        operator = _space_projector(case, partition="views", distributed=True, volume_chunk_shape=chunk,
                                    detector_chunk_shape=pixels, view_chunk_size=2, devices=[0])
        image, sino = _data(case, device="cpu")
        local = sino[operator.view_slice].clone().double().requires_grad_()
        matrix = _matrix(case, _offsets(case, flat=True))
        first = operator.view_slice.start * math.prod(case.detector)
        stop = operator.view_slice.stop * math.prod(case.detector)
        weight = image * ((rank + 1) / 2)
        with _OwnedTrace(case, chunk, pixels) as trace:
            back = operator.backproject(local)
            _close(back, (matrix.T @ _numpy(sino).ravel()).reshape(case.shape))
            trace.phase = "views-cotangent"
            (back * weight).sum().backward()
        _close(local.grad, (matrix[first:stop] @ (_numpy(image) * 1.5).ravel()).reshape(local.shape))
    finally:
        dist.destroy_process_group()


@pytest.mark.cuda
@cuda_required
@gloo_required
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("views", [1, 5], ids=["empty-view-rank", "uneven-view-ranks"])
def test_views_backward_sums_replicated_volume_cotangents_in_bounded_tiles_including_empty_rank(beam, views, tmp_path):
    _run_processes(_views_cotangent_worker, (beam, views, (tmp_path / "views-cotangent").resolve().as_uri()))
