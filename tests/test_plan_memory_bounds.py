"""Real allocator bounds and equivalence to snapshot-based CUDA admission.

Trials use isolated native allocator processes, including expandable segments.
The only admission override forces the exact planner; capacities remain real.
"""

from contextlib import contextmanager
from dataclasses import asdict
import math
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest
import torch

from diffct import chunking
import diffct.operators as operators
from diffct.storage import TensorStore
from tests.test_detector_surfaces import _case, _projector


MIB = 1024 ** 2
cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _subprocess(function, *args, expandable=False):
    env = os.environ.copy()
    configuration = f"backend:native,expandable_segments:{expandable}"
    env["PYTORCH_CUDA_ALLOC_CONF"] = configuration
    env["PYTORCH_ALLOC_CONF"] = configuration
    command = (f"from tests.test_plan_memory_bounds import {function}; "
               f"{function}(*{args!r})")
    result = subprocess.run([sys.executable, "-c", command], env=env,
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr


@contextmanager
def _allocator():
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    assert torch.cuda.get_allocator_backend() == "native"
    previous = torch.cuda.get_per_process_memory_fraction(0)
    try:
        yield torch.cuda.get_device_properties(0).total_memory
    finally:
        torch.cuda.synchronize(0)
        torch.cuda.set_per_process_memory_fraction(previous, 0)
        torch.cuda.empty_cache()


def _assert_bounds():
    device = torch.device("cuda:0")
    before = (torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device))
    lower = chunking._memory_lower_bounds(device)
    exact = (*chunking._memory_budget(device), chunking._reservation_budget(device))
    after = (torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device))
    assert before == after, "comparison changed the allocator state"
    assert all(0 <= bound <= limit for bound, limit in zip(lower, exact)), (lower, exact)


def _bounds_trial(state):
    with _allocator() as total:
        assert torch.cuda.memory_reserved(0) == 0, "trial must start cold"
        _assert_bounds()
        if state == "fresh":
            return
        # Adjacent allocations split a large-pool segment; a separate large
        # allocation supplies a wholly cached segment after release.
        live = torch.empty(3 * MIB, dtype=torch.uint8, device="cuda:0")
        released = torch.empty(5 * MIB, dtype=torch.uint8, device="cuda:0")
        large = torch.empty(24 * MIB, dtype=torch.uint8, device="cuda:0")
        del released, large
        segments = chunking._native_segments(0)
        assert any(any(b["state"] == "inactive" for b in s["blocks"])
                   and any(b["state"] == "active_allocated" for b in s["blocks"])
                   for s in segments)
        assert torch.cuda.memory_reserved(0) > torch.cuda.memory_allocated(0)
        _assert_bounds()
        if state == "streams":
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                foreign_live = torch.empty(3 * MIB, dtype=torch.uint8, device="cuda:0")
                foreign_free = torch.empty(5 * MIB, dtype=torch.uint8, device="cuda:0")
            del foreign_free
            segments = chunking._native_segments(0)
            assert any(s["stream"] == stream.cuda_stream
                       and any(b["state"] == "inactive" for b in s["blocks"])
                       and any(b["state"] == "active_allocated" for b in s["blocks"])
                       for s in segments)
            _assert_bounds()
            small_live = torch.empty(MIB // 2, dtype=torch.uint8, device="cuda:0")
            used = torch.empty(MIB // 2, dtype=torch.uint8, device="cuda:0")
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                torch.cuda._sleep(50000000)
                used.fill_(1)
                used.record_stream(stream)
            del used
            assert any(any(b["state"] == "active_awaiting_free" for b in s["blocks"])
                       and any(b["state"] == "active_allocated" for b in s["blocks"])
                       for s in chunking._native_segments(0))
            _assert_bounds()
            stream.synchronize()
            _assert_bounds()
            del foreign_live, small_live
        elif state == "fraction":
            torch.cuda.set_per_process_memory_fraction(64 * MIB / total, 0)
            _assert_bounds()
        elif state == "expandable":
            assert any(s["is_expandable"] for s in segments), \
                "expandable segments requested but not exercised"
        else:
            assert state == "mixed"
        del live


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("state", ["fresh", "mixed", "streams", "fraction", "expandable"])
def test_lower_bounds_never_exceed_exact_real_allocator_limits(state):
    _subprocess("_bounds_trial", state, expandable=state == "expandable")


def _compare_plans(operator, source, is_project, output=None):
    actual = operator._execution_plan(source, is_project, output)
    with mock.patch.object(operators, "_memory_lower_bounds", return_value=(0, 0, 0)):
        exact = operator._execution_plan(source, is_project, output)
    assert (actual is None) == (exact is None)
    if exact is not None:
        assert asdict(actual) == asdict(exact)
    return exact


@torch.no_grad()
def _equality_trial(beam, pressure):
    with _allocator() as total:
        case = _case(beam, views=5)
        case.shape = (48, 64, 64) if beam == "cone" else (384, 512)
        case.detector = (24, 24) if beam == "cone" else (64,)
        operator = _projector(case, devices=[0])
        for is_project in (True, False):
            torch.cuda.set_per_process_memory_fraction(1.0, 0)
            shape = case.shape if is_project else case.sino_shape
            source = torch.linspace(.1, 1., steps=math.prod(shape)).reshape(shape).cuda()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if pressure:
                # Search real process ceilings, without fabricating allocator
                # statistics. Require a feasible plan that actually splits work.
                found = False
                for ceiling in (8, 12, 16, 24, 32, 48, 64):
                    torch.cuda.set_per_process_memory_fraction(ceiling * MIB / total, 0)
                    with mock.patch.object(operators, "_memory_lower_bounds", return_value=(0, 0, 0)):
                        try:
                            plan = operator._execution_plan(source, is_project)
                        except RuntimeError as error:
                            assert "minimum tile/view/pixel batch" in str(error)
                            continue
                    if plan is not None and (plan.chunk_shape != case.shape
                                             or plan.view_chunk_size < case.views
                                             or plan.detector_chunk_shape != case.detector):
                        found = True
                        break
                assert found, "real ceilings did not exercise a feasible chunked plan"
            else:
                torch.cuda.set_per_process_memory_fraction(256 * MIB / total, 0)
            plan = _compare_plans(operator, source, is_project)
            assert (plan is not None) == pressure
            operation = operator.project if is_project else operator.backproject
            actual = operation(source).cpu()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            _compare_plans(operator, source, is_project)
            with mock.patch.object(operators, "_memory_lower_bounds", return_value=(0, 0, 0)):
                expected = operation(source).cpu()
            assert torch.isfinite(actual).all() and actual.abs().sum() > 0
            torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-4)
            del source
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", ["parallel", "fan", "cone"])
@pytest.mark.parametrize("pressure", [False, True], ids=["ample", "fraction-chunks"])
def test_snapshot_free_and_exact_plans_and_outputs_match(beam, pressure):
    _subprocess("_equality_trial", beam, pressure)


@torch.no_grad()
def _snapshot_trial(mode):
    with _allocator() as total:
        torch.cuda.set_per_process_memory_fraction(256 * MIB / total, 0)
        case = _case("cone")
        options = {"volume_chunk_shape": (2, 3, 4)} if mode == "explicit" else {}
        operator = _projector(case, devices=[0], **options)
        source = torch.ones(case.shape, device="cuda:0")
        output = TensorStore(torch.zeros(case.sino_shape)) if mode == "store" else None
        plan = _compare_plans(operator, source, True, output)
        assert (plan is None) == (mode == "full")
        with mock.patch.object(torch.cuda, "memory_snapshot", wraps=torch.cuda.memory_snapshot) as snapshots, \
                mock.patch.object(operators, "_memory_lower_bounds", wraps=operators._memory_lower_bounds) as bounds:
            if output is None:
                actual = operator.project(source).cpu()
            else:
                assert operator.project_into(source, output) is output
                actual = output.tensor.clone()
        if mode == "full":
            assert bounds.call_count > 0
            assert snapshots.call_count == 0
        else:
            assert bounds.call_count == 0
            assert snapshots.call_count > 0
        with mock.patch.object(operators, "_memory_lower_bounds", return_value=(0, 0, 0)):
            if output is None:
                expected = operator.project(source).cpu()
            else:
                operator.project_into(source, output)
                expected = output.tensor
        assert torch.isfinite(actual).all() and actual.abs().sum() > 0
        torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-4)


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("mode", ["full", "explicit", "store"])
def test_snapshot_elision_only_for_eligible_full_launch(mode):
    _subprocess("_snapshot_trial", mode)
