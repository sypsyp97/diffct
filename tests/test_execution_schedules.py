"""Actual native schedules, copy payloads and independent cell references."""

import inspect
import math
import sys
from unittest import mock

import numpy as np
import pytest
import torch

import diffct
import diffct.operators as operators
import diffct.projectors as native
from tests.test_block_storage import _GuardedStore, _rejection
from tests.test_chunked_projector import _memory_inventory, _offsets
from tests.test_detector_surfaces import _close, _data, _matrix, _numpy, _projector
from tests.test_execution_buffers import BEAMS, _ExecutionTrace, _chunk, _regular_case


cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
SCHEDULES = ("spatial", "views", "window")
REPORT_KEYS = {
    "operation", "schedule", "chunk_shape", "view_chunk_size", "detector_chunk_shape",
    "window_size", "gpu_budget_bytes", "estimated_gpu_bytes", "pinned_bytes_peak",
    "h2d_bytes", "d2h_bytes", "p2p_bytes", "kernel_launches", "layout_preparations", "pilot",
}


def _scheduled_projector(case, schedule="auto", *, surface=None, **kwargs):
    assert "schedule" in inspect.signature(diffct.Projector).parameters, \
        "INTERFACE_PENDING: Projector(..., schedule='auto') declaration"
    projector = _projector(case, surface, schedule=schedule, **kwargs)
    assert hasattr(projector, "last_execution_stats"), \
        "INTERFACE_PENDING: Projector.last_execution_stats declaration"
    return projector


def _stream():
    current = torch.cuda.current_stream()
    return current.device.index, int(current.cuda_stream)


def _schedule_case(beam, *, views=7):
    case = _regular_case(beam, views=views)
    # Equal-dtype transfer fixtures allow exact payload observations without
    # guessing where PyTorch performs a simultaneous dtype conversion.
    case.trajectory = tuple(value.float() for value in case.trajectory)
    return case


class _CopyTrace(_ExecutionTrace):
    """Forward actual operations; metadata never keeps a tensor or graph alive."""
    def __init__(self, case):
        super().__init__(case)
        self.sequence, self.copies, self.layouts = 0, [], []

    def _step(self):
        self.sequence += 1
        return self.sequence

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        sequence = self._step()
        result = super().__torch_dispatch__(function, types, args, kwargs)
        operation = self.operations[-1]
        source = target = None
        if str(function) == "aten._to_copy.default":
            source, target = operation["inputs"][0], operation["outputs"][0]
        elif str(function) == "aten.copy_.default":
            target, source = operation["inputs"][:2]
        if source is not None and source["device"] != target["device"]:
            if source["device"].type == "cpu":
                direction = "h2d"
            elif target["device"].type == "cpu":
                direction = "d2h"
            else:
                direction = "p2p"
            # The contract counts logical destination payload, including copies
            # that also cast. This does not estimate internal physical DMA.
            if str(function) == "aten._to_copy.default":
                source_dtype, target_dtype = args[0].dtype, result.dtype
            else:
                target_dtype, source_dtype = args[0].dtype, args[1].dtype
            non_blocking = (kwargs or {}).get("non_blocking", args[2] if len(args) > 2 else False)
            self.copies.append({"sequence": sequence, "phase": self.phase, "direction": direction,
                                "bytes": target["bytes"], "source": source, "target": target,
                                "stream": _stream(), "non_blocking": bool(non_blocking),
                                "equal_dtype": source_dtype == target_dtype})
        return result

    def __enter__(self):
        super().__enter__()
        original = native._prepare_volume

        def prepare(beam, tensor):
            sequence = self._step()
            result = original(beam, tensor)
            self.layouts.append({"sequence": sequence, "phase": self.phase, "beam": beam,
                                 "shape": tuple(result.shape), "bytes": result.numel() * result.element_size()})
            return result

        for name, module in list(sys.modules.items()):
            if name == "diffct" or name.startswith("diffct."):
                for attribute, value in list(vars(module).items()):
                    if value is original:
                        self.patches.enter_context(mock.patch.object(module, attribute, prepare))
        return self

    def _proxy(self, kernel, beam, kind):
        parent = super()._proxy(kernel, beam, kind)
        tracer = self

        class Proxy:
            def __getitem__(self, configuration):
                launch = parent[configuration]

                def observed(*args, **kwargs):
                    sequence, stream = tracer._step(), _stream()
                    handle = configuration[2].handle
                    assert int(getattr(handle, "value", handle) or 0) == stream[1], \
                        "native Numba launch lost the active PyTorch stream"
                    result = launch(*args, **kwargs)
                    tracer.launches[-1].update(sequence=sequence, stream=stream)
                    return result
                return observed
        return Proxy()


def _metadata_only(value):
    assert not isinstance(value, (torch.Tensor, np.ndarray)), "report retained numerical data or a graph"
    if isinstance(value, dict):
        assert all(isinstance(key, str) for key in value)
        for item in value.values():
            _metadata_only(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _metadata_only(item)
    else:
        assert value is None or isinstance(value, (str, bool, int, float))


def _check_report(projector, trace, operation, *, schedule=None):
    report = projector.last_execution_stats
    assert isinstance(report, dict), "instrumented streamed execution produced no report"
    assert REPORT_KEYS <= report.keys()
    _metadata_only(report)
    assert report["operation"] == operation
    assert report["schedule"] in SCHEDULES
    if schedule is not None:
        assert report["schedule"] == schedule
    assert isinstance(report["window_size"], int) and report["window_size"] > 0
    if report["schedule"] == "spatial":
        assert report["window_size"] == 1
    elif report["schedule"] == "window":
        assert report["window_size"] <= 2
    assert 0 < report["estimated_gpu_bytes"] <= report["gpu_budget_bytes"]
    pilot = report["pilot"]
    assert isinstance(pilot, list) and len(pilot) <= 8
    pilot_keys = {"schedule", "chunk_shape", "view_chunk_size", "detector_chunk_shape", "window_size",
                  "elapsed_ms", "h2d_bytes", "d2h_bytes", "p2p_bytes", "kernel_launches",
                  "layout_preparations", "estimated_h2d_bytes", "estimated_d2h_bytes"}
    for candidate in pilot:
        assert pilot_keys <= candidate.keys()
        assert math.isfinite(candidate["elapsed_ms"]) and candidate["elapsed_ms"] >= 0
        assert 0 < candidate["kernel_launches"] <= 4
        assert candidate["h2d_bytes"] + candidate["d2h_bytes"] > 0
        assert candidate["estimated_h2d_bytes"] >= 0 and candidate["estimated_d2h_bytes"] >= 0
    for direction in ("h2d", "d2h", "p2p"):
        key = f"{direction}_bytes"
        assert isinstance(report[key], int) and report[key] >= 0
        actual = sum(item["bytes"] for item in trace.copies if item["direction"] == direction)
        assert report[key] + sum(item[key] for item in pilot) == actual, (key, report, actual)
    assert report["kernel_launches"] + sum(item["kernel_launches"] for item in pilot) == len(trace.launches)
    assert report["layout_preparations"] + sum(item["layout_preparations"] for item in pilot) == len(trace.layouts)
    return report


@pytest.mark.parametrize("value", ["tiles", "view", "", None, 1, True])
def test_schedule_constructor_rejects_unknown_or_nonstring_values(value):
    with _rejection("schedule|auto|spatial|views|window"):
        _scheduled_projector(_regular_case("cone"), value)


def test_schedule_default_and_report_start_without_execution_or_cuda():
    assert inspect.signature(diffct.Projector).parameters["schedule"].default == "auto"
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        projector = _scheduled_projector(_regular_case("cone"))
    assert projector.schedule == "auto" and projector.last_execution_stats is None


def test_METADATA_foreign_output_device_budget_is_admitted_before_any_store_io_or_kernel():
    """Synthetic inventory admission only; no second GPU execution is claimed."""
    case = _schedule_case("parallel", views=5)
    calls, inventory = [], []

    class MetadataStore:
        def __init__(self, shape, device):
            self.shape, self.dtype, self.device = shape, torch.float32, torch.device(device)
            self.writable = True

        def read(self, index):
            calls.append("read")
            raise AssertionError("metadata admission reached store I/O")

        def write(self, index, value, *, accumulate=False):
            calls.append("write")
            raise AssertionError("metadata admission reached store I/O")

        def flush(self):
            pass

    def budget(device):
        index = torch.device(device).index
        inventory.append(index)
        return (0 if index == 1 else 1 << 30), (1 << 63) - 1

    source = MetadataStore(case.shape, "cpu")
    output = MetadataStore(case.sino_shape, "cuda:1")
    with mock.patch.object(torch.cuda, "is_available", return_value=True), \
            mock.patch.object(torch.cuda, "device_count", return_value=2), \
            mock.patch.object(operators, "_memory_budget", side_effect=budget), _CopyTrace(case) as trace:
        projector = _scheduled_projector(case, devices=[0], volume_chunk_shape=(2, 3), view_chunk_size=1)
        with _rejection("memory|budget|fit|minimum|limit"):
            projector.project_into(source, output)
    assert 1 in inventory, "foreign output capacity was omitted from the minimum budget"
    assert not calls and not trace.launches


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_forced_schedules_use_real_native_kernels_and_exact_copy_reports(beam, schedule, operation):
    case = _schedule_case(beam)
    chunk = _chunk(case)
    projector = _scheduled_projector(case, schedule, volume_chunk_shape=chunk, view_chunk_size=2)
    image, sino = _data(case, device="cpu")
    data = image if operation == "project" else sino
    with _CopyTrace(case) as trace:
        result = getattr(projector, operation)(data)
    matrix = _matrix(case, _offsets(case, flat=True))
    operator = matrix if operation == "project" else matrix.T
    shape = case.sino_shape if operation == "project" else case.shape
    _close(result, (operator @ _numpy(data).ravel()).reshape(shape))
    report = _check_report(projector, trace, operation, schedule=schedule)
    assert not report["pilot"], "forced schedule ran an unrelated selection pilot"
    assert all(a <= b for a, b in zip(report["chunk_shape"], chunk))
    assert report["view_chunk_size"] <= 2
    assert {record["views"] for record in trace.launches} == {1, 2}
    if operation == "backproject":
        assert report["d2h_bytes"] == 4 * math.prod(case.shape), "per-view partial volumes were downloaded"
    elif schedule == "views":
        assert report["d2h_bytes"] == 4 * math.prod(case.sino_shape), "completed ray batches were downloaded repeatedly"
    elif schedule == "spatial":
        tiles = math.prod(math.ceil(size / part) for size, part in zip(case.shape, report["chunk_shape"]))
        assert report["d2h_bytes"] == tiles * 4 * math.prod(case.sino_shape)
        assert len(trace.layouts) == tiles, "native layouts were recreated per view batch"


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_schedule_into_and_data_hessian_preserve_same_independent_operator(schedule):
    case = _schedule_case("cone")
    chunk = _chunk(case)
    projector = _scheduled_projector(case, schedule, volume_chunk_shape=chunk, view_chunk_size=2)
    image, sino = _data(case, device="cpu")
    source = _GuardedStore(image, chunk)
    output = _GuardedStore(torch.full(case.sino_shape, torch.nan), (2, *case.detector))
    with _CopyTrace(case) as trace:
        assert projector.project_into(source, output) is output
    _check_report(projector, trace, "project", schedule=schedule)
    assert torch.all(output.overwrites == 1)
    matrix = _matrix(case, _offsets(case, flat=True))
    _close(output.tensor, (matrix @ _numpy(image).ravel()).reshape(case.sino_shape))
    image.requires_grad_()
    dense = projector.project(image)
    gradient, = torch.autograd.grad(dense.square().sum() / 2, image, create_graph=True)
    direction = torch.linspace(-.3, .4, image.numel()).reshape_as(image)
    hessian, = torch.autograd.grad((gradient * direction).sum(), image)
    _close(hessian, (matrix.T @ matrix @ _numpy(direction).ravel()).reshape(case.shape))


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_auto_measures_bounded_pilots_and_keeps_final_outputs_clean(operation):
    case = _schedule_case("cone")
    projector = _scheduled_projector(case, volume_chunk_shape=_chunk(case), view_chunk_size=2)
    image, sino = _data(case, device="cpu")
    data = image if operation == "project" else sino
    with _CopyTrace(case) as trace:
        result = getattr(projector, operation)(data)
    matrix = _matrix(case, _offsets(case, flat=True))
    operator = matrix if operation == "project" else matrix.T
    _close(result, (operator @ _numpy(data).ravel()).reshape(result.shape))
    report = _check_report(projector, trace, operation)
    assert report["pilot"], "multiple eligible schedules were chosen without real native/copy measurements"
    tiles = math.prod(math.ceil(size / part) for size, part in zip(case.shape, report["chunk_shape"]))
    batches = math.ceil(case.views / report["view_chunk_size"])
    pixels = math.prod(math.ceil(size / part) for size, part in zip(case.detector, report["detector_chunk_shape"]))
    assert report["kernel_launches"] == tiles * batches * pixels
    # Every launch in this bounded fixture also fits the universal pilot cap.
    assert all(math.prod(record["native_shape"]) <= 4096 and record["views"] <= 4
               and math.prod(record["rays"]["shape"]) <= 4096 for record in trace.launches)


@pytest.mark.cuda
@cuda_required
def test_single_tile_single_ray_batch_avoids_selection_pilot():
    case = _schedule_case("parallel", views=1)
    case.shape, case.detector = (3, 4), (3,)
    projector = _scheduled_projector(case, volume_chunk_shape=case.shape, view_chunk_size=1)
    with _CopyTrace(case) as trace:
        projector.project(_data(case, device="cpu")[0])
    report = _check_report(projector, trace, "project")
    assert report["pilot"] == []


@pytest.mark.cuda
@cuda_required
def test_automatic_candidate_family_includes_contiguous_slabs_and_cartesian_blocks():
    case = _schedule_case("cone")
    case.shape, case.detector = (17, 5, 7), (3, 2)
    image, _ = _data(case, device="cpu")
    with _memory_inventory({torch.cuda.current_device(): 8000}):
        projector = _scheduled_projector(case, view_chunk_size=2)
        with _CopyTrace(case) as trace:
            result = projector.project(image)
    report = _check_report(projector, trace, "project")
    shapes = [tuple(candidate["chunk_shape"]) for candidate in report["pilot"]]
    assert any(shape[0] < case.shape[0] and shape[1:] == case.shape[1:] for shape in shapes), shapes
    assert any(sum(a < b for a, b in zip(shape, case.shape)) >= 2 for shape in shapes), shapes
    assert report["estimated_gpu_bytes"] <= 6000
    matrix = _matrix(case, _offsets(case, flat=True))
    _close(result, (matrix @ _numpy(image).ravel()).reshape(case.sino_shape))


@pytest.mark.cuda
@cuda_required
def test_report_replaced_by_next_operation_and_cleared_on_full_cuda_or_error():
    case = _schedule_case("parallel", views=5)
    projector = _scheduled_projector(case)
    image, sino = _data(case, device="cpu")
    projector.project(image)
    prior = projector.last_execution_stats
    assert isinstance(prior, dict) and prior["operation"] == "project"
    projector.backproject(sino)
    current = projector.last_execution_stats
    assert isinstance(current, dict) and current is not prior and current["operation"] == "backproject"
    projector.project(image.cuda())
    report = projector.last_execution_stats
    assert report is None or (report is not current and report["operation"] == "project")
    with pytest.raises(ValueError, match="shape"):
        projector.project(torch.empty(1))
    assert projector.last_execution_stats is None, "failed operation left stale accounting"


@pytest.mark.cuda
@cuda_required
def test_forced_view_backprojection_never_downloads_partial_volumes_when_retention_does_not_fit():
    case = _schedule_case("cone", views=5)
    image, sino = _data(case, device="cpu")
    with _memory_inventory({torch.cuda.current_device(): 1600}), _CopyTrace(case) as trace:
        try:
            projector = _scheduled_projector(case, "views", volume_chunk_shape=(1, 2, 3), view_chunk_size=1)
            result = projector.backproject(sino)
        except (ValueError, RuntimeError) as error:
            assert not isinstance(error, NotImplementedError)
            assert any(word in str(error).lower() for word in ("memory", "budget", "fit", "view", "schedule", "accum"))
            assert not trace.launches
            return
    report = _check_report(projector, trace, "backproject")
    assert report["schedule"] != "views", "all retained accumulators exceeded the live budget"
    assert report.get("fallback_reason"), "forced schedule fallback was not explained"
    assert report["d2h_bytes"] == 4 * math.prod(case.shape)
    matrix = _matrix(case, _offsets(case, flat=True))
    _close(result, (matrix.T @ _numpy(sino).ravel()).reshape(case.shape))


def _cpu_ranking_metadata(operation, equal_time):
    """Synthetic CPU ranking fixture; no numerical output or GPU timing claim."""
    from types import SimpleNamespace

    case = _schedule_case("cone", views=7)
    projector = _scheduled_projector(case, volume_chunk_shape=(3, 4, 5), view_chunk_size=2)
    plan = operators._ChunkPlan((3, 4, 5), 2, (torch.device("cuda:0"),),
                                detector_chunk_shape=case.detector, gpu_budget_bytes=1048576)
    source = SimpleNamespace(dtype=torch.float32, device=torch.device("cpu"))
    sink = SimpleNamespace(device=torch.device("cpu"))
    candidates = projector._candidate_plans(plan, operation == "project", None)
    assert {candidate.schedule for candidate in candidates} == set(SCHEDULES)
    assert all(0 < candidate.estimated_gpu_bytes <= plan.gpu_budget_bytes for candidate in candidates)
    times = {"spatial": 160., "views": 48., "window": 48. if equal_time else 40.}
    payloads = {"spatial": (1200, 1200), "views": (700, 100), "window": (100, 1500)}
    records = []

    def pilot(actual_source, actual_sink, is_project, geometry, candidate, spec):
        assert actual_source is source and actual_sink is sink
        assert is_project == (operation == "project") and geometry is None and spec is None
        h2d, d2h = payloads[candidate.schedule]
        measured = 3 if candidate.schedule == "window" else 2
        record = dict(schedule=candidate.schedule, chunk_shape=candidate.chunk_shape,
                      view_chunk_size=candidate.view_chunk_size, detector_chunk_shape=candidate.detector_chunk_shape,
                      window_size=candidate.window_size, estimated_elapsed_ms=times[candidate.schedule],
                      elapsed_ms=times[candidate.schedule] * measured / 32,
                      estimated_h2d_bytes=h2d, estimated_d2h_bytes=d2h,
                      fixture_scope="synthetic CPU ranking metadata; no native execution")
        records.append(record)
        return record

    return projector, plan, source, sink, candidates, records, pilot


@pytest.mark.parametrize("operation", ["project", "backproject"])
@pytest.mark.parametrize("equal_time", [False, True], ids=["faster-higher-payload", "equal-time-lower-payload"])
def test_CPU_synthetic_auto_ranking_prioritizes_estimated_time_then_total_copy_bytes_and_caches_winner(operation, equal_time):
    # CUDA device objects below are planning metadata only. Fail immediately if
    # this deterministic decision fixture attempts any actual CUDA initialization.
    with mock.patch.object(torch.cuda, "is_available", return_value=False), \
            mock.patch.object(torch.cuda, "get_allocator_backend", return_value="native"), \
            mock.patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU ranking fixture initialized CUDA")):
        projector, plan, source, sink, candidates, records, pilot = _cpu_ranking_metadata(operation, equal_time)
        with mock.patch.object(projector, "_pilot_candidate", side_effect=pilot) as measured:
            chosen, pilots, cached = projector._choose_schedule(source, sink, operation == "project", None, plan, None)
            assert not cached and pilots == records and measured.call_count == len(candidates)
            winner, replay, cached = projector._choose_schedule(source, sink, operation == "project", None, plan, None)
            assert cached and replay == [] and winner is chosen
            assert measured.call_count == len(candidates), "cached winner repeated the synthetic pilots"
        assert len(projector._schedule_cache) == 1 and projector.last_execution_stats is None
        assert winner in candidates and winner.estimated_gpu_bytes <= plan.gpu_budget_bytes
        for record in pilots:
            _metadata_only(record)
        expected = "views" if equal_time else "window"
        assert chosen.schedule == expected, ("CPU ranking policy chose copy bytes before estimated elapsed time", operation, pilots, chosen)
        assert winner.schedule == expected and winner.chunk_shape == plan.chunk_shape
        assert winner.view_chunk_size == plan.view_chunk_size and winner.detector_chunk_shape == plan.detector_chunk_shape
