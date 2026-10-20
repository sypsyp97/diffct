"""Retained constructor frames obey per-call geometry admission after updates."""

import json
import math
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from diffct import Projector
from tests.test_call_trajectory import _WriteTrace
from tests.test_surface_parameter_updates import (
    OPERATIONS, _assert_rejected, _call, _frames as _divergent_frames, _input_and_sink,
)
from tests.test_trajectory_admission_cost import _AdmissionTrace, _admission_only
from tests.test_trajectory_surface_integration import _run_two, _start_group


_NVIDIA = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA is required",
)
EXECUTIONS = (
    pytest.param("torch", "float32", "cpu", id="torch-cpu-f32"),
    pytest.param("torch", "float64", "cpu", id="torch-cpu-f64"),
    pytest.param("numba", "fast", "cuda", id="native-cuda",
                 marks=[pytest.mark.cuda, _NVIDIA]),
)


def _frames(beam, *, views=1, device="cpu", trainable=True):
    frames = _divergent_frames("cone" if beam in ("cone", "parallel3d") else "fan", views=views)
    if beam in ("parallel", "parallel3d"):
        frames[0].zero_()
        frames[0][:, 0] = 1.0
    frames = tuple(value.to(device) for value in frames)
    frames[1].requires_grad_(trainable)
    return frames


def _operator(frames, beam, backend="torch", precision="float64", *, shape=None):
    dimensions = frames[0].shape[-1]
    return Projector(
        frames, shape or (2,) * dimensions, (1,) * (dimensions - 1),
        beam=beam, backend=backend, precision=precision, distributed=False,
    )


@pytest.mark.parametrize("backend,precision,device", EXECUTIONS)
@pytest.mark.parametrize("beam", ("fan", "cone"))
@pytest.mark.parametrize("operation", OPERATIONS)
def test_live_center_coincidence_rejected_like_explicit_trajectory(
    backend, precision, device, beam, operation,
):
    frames = _frames(beam, device=device)
    operator = _operator(frames, beam, backend, precision)
    value, sink = _input_and_sink(operator, operation, device)
    with torch.no_grad():
        valid = _call(operator, operation, value, sink)
        assert torch.isfinite(valid).all() and valid.abs().sum() > 0
        frames[1].copy_(frames[0])
    _assert_rejected(operator, operation, device=device, trajectory=frames)
    _assert_rejected(operator, operation, device=device)


@pytest.mark.parametrize("backend,precision,device", (EXECUTIONS[0], EXECUTIONS[2]))
def test_live_interior_pixel_coincident_only_after_float32_staging_rejects_before_writes(
    backend, precision, device,
):
    frames = _frames("fan", device=device)
    operator = Projector(
        frames, (2, 2), (5,), beam="fan", backend=backend, precision=precision,
        distributed=False,
    )
    with torch.no_grad():
        frames[0].copy_(frames[0].new_tensor([[1e6, 0.0]]))
        frames[1].copy_(frames[1].new_tensor([[1e6 + .01, 1.0]]))
    # Every raw FP64 pixel is distinct from the source. The FP32 corners
    # also remain distinct; only the interior u=-1 pixel becomes coincident.
    coordinates = torch.arange(-2, 3, dtype=torch.float64, device=device)
    raw_pixels = frames[1] + coordinates[:, None] * frames[2]
    assert torch.all(torch.any(raw_pixels != frames[0], dim=-1))
    staged_pixels = frames[1].float() + coordinates.float()[:, None] * frames[2].float()
    assert torch.all(torch.any(staged_pixels[[0, -1]] != frames[0].float(), dim=-1))
    assert torch.equal(staged_pixels[1], frames[0][0].float())
    _assert_rejected(operator, "project_into", device=device, trajectory=frames)
    _assert_rejected(operator, "project_into", device=device)


@pytest.mark.parametrize("beam", ("parallel", "parallel3d"))
@pytest.mark.parametrize("kind", ("nonunit", "nan"))
@pytest.mark.parametrize("operation", ("project", "backproject_into"))
def test_non_grad_axis_is_still_live_when_another_frame_requires_grad(beam, kind, operation):
    frames = _frames(beam)
    operator = _operator(frames, beam)
    assert frames[1].requires_grad and not frames[0].requires_grad
    with torch.no_grad():
        if kind == "nonunit":
            frames[0].mul_(2.0)
        else:
            frames[2][0, 1] = float("nan")
    _assert_rejected(operator, operation, trajectory=frames)
    _assert_rejected(operator, operation)


@pytest.mark.parametrize("backend,precision,device", (EXECUTIONS[0], EXECUTIONS[2]))
@pytest.mark.parametrize("bound", ("near-endpoint", "absolute-distance"))
@pytest.mark.parametrize("operation", ("backproject", "project_into"))
def test_live_float32_frames_recheck_acquisition_distance_bounds(
    backend, precision, device, bound, operation,
):
    frames = _frames("fan", device=device)
    operator = _operator(frames, "fan", backend, precision)
    with torch.no_grad():
        if bound == "near-endpoint":
            frames[0][0, 0], frames[1][0, 0] = -2e6, 2e6
        else:
            frames[1][0, 0] = 2e15
    _assert_rejected(operator, operation, device=device, trajectory=frames)
    _assert_rejected(operator, operation, device=device)


@pytest.mark.parametrize("backend,precision,device", (EXECUTIONS[1], EXECUTIONS[2]))
@pytest.mark.parametrize("beam", ("fan", "cone"))
@pytest.mark.parametrize("operation", ("project", "backproject"))
def test_valid_live_updates_keep_values_and_outstanding_frame_gradients(
    backend, precision, device, beam, operation,
):
    frames = _frames(beam, device=device)
    with torch.no_grad():
        frames[1][0, 1] = .4
    operator = _operator(frames, beam, backend, precision, shape=(1,) * frames[0].shape[-1])
    value, _ = _input_and_sink(operator, operation, device)
    value = (1.25 * value).requires_grad_()
    old = getattr(operator, operation)(value)
    with torch.no_grad():
        # Updating a non-grad component must take effect too: the constructor
        # retained the entire tuple because center requires gradients.
        frames[0][0, 0] = -3.0
        frames[1][0, 1] = .6
    new = getattr(operator, operation)(value)
    assert not torch.equal(old, new)

    # Both rays cross both x faces of a unit voxel, away from edge ties.
    tolerance = 2e-5 if backend == "numba" else 2e-12
    for result, distance, tangent in ((old, 4.0, .4), (new, 5.0, .6)):
        length = math.sqrt(1 + (tangent / distance) ** 2)
        torch.testing.assert_close(result, torch.full_like(result, 1.25 * length),
                                   rtol=tolerance, atol=tolerance)
        data_gradient, center_gradient = torch.autograd.grad(result.sum(), (value, frames[1]))
        torch.testing.assert_close(data_gradient, torch.full_like(value, length),
                                   rtol=tolerance, atol=tolerance)
        expected = torch.zeros_like(frames[1])
        expected[0, 0] = -1.25 * tangent ** 2 / (distance ** 3 * length)
        expected[0, 1] = 1.25 * tangent / (distance ** 2 * length)
        torch.testing.assert_close(center_gradient, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("beam", ("fan", "cone"))
@pytest.mark.parametrize("operation", OPERATIONS)
def test_constructor_without_grad_keeps_its_frozen_trajectory(beam, operation):
    frames = _frames(beam, trainable=False)
    operator = _operator(frames, beam)
    value, sink = _input_and_sink(operator, operation)
    with torch.no_grad():
        expected = _call(operator, operation, value, sink).clone()
        frames[1].copy_(frames[0])
        actual = _call(operator, operation, value, sink)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _assert_rejected(operator, operation, trajectory=frames)


@pytest.mark.cuda
@_NVIDIA
def test_flat_cuda_admission_does_not_stage_bulk_frames_through_cpu(monkeypatch):
    frames = _frames("fan", device="cuda")
    operator = _operator(frames, "fan", "numba", "fast")
    calls = _admission_only(monkeypatch, "project")
    value, _ = _input_and_sink(operator, "project", "cuda")
    for kwargs in ({}, {"trajectory": frames}):
        with monkeypatch.context() as local:
            trace = _AdmissionTrace(local)
            operator.project(value, **kwargs)
        assert trace.d2h_calls == 0, (
            f"flat CUDA admission staged {trace.d2h_calls} tensors on CPU; "
            "frame checks can reduce on their existing device"
        )
    assert len(calls) == 2


def _collective_worker(rank, rendezvous, directory):
    _start_group(rank, rendezvous, "gloo")
    records = []
    try:
        for mode in ("mixed-request", "explicit-invalid", "default-invalid"):
            frames = _frames("fan", views=2)
            operator = Projector(
                frames, (2, 2), (1,), beam="fan", backend="torch", precision="float64",
                distributed=True, partition="views",
            )
            if mode != "mixed-request" and rank == 1:
                with torch.no_grad():
                    frames[1][1].copy_(frames[0][1])
            kwargs = ({"trajectory": frames}
                      if mode == "explicit-invalid" or mode == "mixed-request" and rank == 1
                      else {})
            value = torch.ones(operator.volume_shape, dtype=operator.dtype)
            sink = _WriteTrace(torch.full(operator.projection_shape, 73.0, dtype=operator.dtype))
            error = None
            try:
                with torch.no_grad():
                    operator.project_into(value, sink, **kwargs)
            except (TypeError, ValueError, RuntimeError) as failure:
                error = type(failure).__name__
            records.append({
                "mode": mode, "error": error, "writes": sink.writes,
                "unchanged": bool(torch.all(sink.tensor == 73.0)),
            })
        (Path(directory) / f"rank-{rank}.json").write_text(json.dumps(records), encoding="utf-8")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is required",
)
def test_gloo_live_frame_rejection_is_collective_before_writes(tmp_path):
    _run_two(_collective_worker, tmp_path)
    for rank in range(2):
        records = json.loads((tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for record in records:
            assert record["error"] == "ValueError", (rank, record)
            assert record["writes"] == 0 and record["unchanged"], (rank, record)
