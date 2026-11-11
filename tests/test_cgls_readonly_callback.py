"""CGLS callbacks observe recurrence-owned iterates; writes fail collectively."""

from datetime import timedelta
import json
from pathlib import Path
import time
from unittest import mock

import numpy as np
import pytest
import torch
import torch.distributed as dist

import diffct
from diffct import _solver_state
from diffct.pipeline import _fragments
from tests.test_block_storage import _numpy_blocks
from tests.test_detector_surfaces import _case, _projector
from tests.test_execution_features import GLOO
from tests.test_solver_api import _DiskAllocations, _SolverStore, _checkpoint, _fixture
from tests.test_solver_checkpoint_edges import _numerical_calls, _real_problem
from tests.test_solver_collective_failures import (
    GROUP_TIMEOUT, REJECTION_TIMEOUT, _finish_record, _run_two,
)


READ_ONLY = r"(?i)read[- ]only"
ONE = (slice(0, 1), slice(0, 1))


def _output(kind, operator, directory):
    tensor = torch.empty(operator.volume_shape)
    if kind == "tensor":
        return tensor
    if kind in ("tensorstore", "delegate-tensor"):
        store = diffct.TensorStore(tensor)
    else:
        store = diffct.NpyStore.create(directory / "output.npy", operator.volume_shape)
    return _SolverStore(store, 3) if kind.startswith("delegate-") else store


def _snapshot(volume):
    if isinstance(volume, torch.Tensor):
        return volume.detach().clone()
    result = torch.empty(volume.shape, dtype=volume.dtype)
    for selected in _fragments(tuple(volume.shape), 3):
        result[selected] = volume.read(selected)
    return result


def _mutate(volume, mode):
    if mode == "zero":
        volume.zero_()
    elif mode == "data":
        volume.data.zero_()
    elif mode == "numpy":
        volume.numpy().fill(0.)
    elif mode == "write":
        volume.write(ONE, torch.full((1, 1), -37., dtype=volume.dtype))
    elif mode == "read-alias":
        volume.read(ONE).numpy().fill(-37.)
    else:
        volume._array[0, 0] = -37.


def _solve(api, operator, data, directory, *, kind="tensor", iterations=3, **options):
    options = dict(rtol=0., atol=0., **options)
    if api == "cgls":
        return diffct.cgls(operator, data, iterations, **options)
    output = _output(kind, operator, directory)
    return diffct.cgls_into(operator, data, output, iterations,
                            workspace=directory / "workspace", block_elements=3, **options)


@pytest.mark.parametrize("api,kind,mode", [
    ("cgls", "tensor", "zero"),
    ("cgls", "tensor", "data"),
    ("cgls", "tensor", "numpy"),
    ("cgls_into", "tensor", "zero"),
    ("cgls_into", "tensor", "numpy"),
    ("cgls_into", "tensorstore", "write"),
    ("cgls_into", "tensorstore", "read-alias"),
    ("cgls_into", "npy", "write"),
    ("cgls_into", "npy", "array"),
    ("cgls_into", "delegate-tensor", "read-alias"),
    ("cgls_into", "delegate-npy", "write"),
])
@pytest.mark.parametrize("stop", [False, True], ids=["continue", "return-False"])
def test_mutated_callback_iterate_rejected_before_next_numerical_call(api, kind, mode, stop, tmp_path):
    operator, data, _ = _fixture()
    boundary = []

    def callback(iteration, residual, volume):
        if iteration == 1:
            boundary.append(len(operator.calls))
            _mutate(volume, mode)
        return False if stop else None

    with pytest.raises(ValueError, match=READ_ONLY):
        _solve(api, operator, data, tmp_path, kind=kind, callback=callback)
    assert boundary == [3]
    assert len(operator.calls) == boundary[0], "mutation crossed the next projection/backprojection boundary"


@pytest.mark.parametrize("api", ["cgls", "cgls_into"])
@pytest.mark.parametrize("dtype", [torch.int64, torch.complex64], ids=["integer", "complex"])
def test_callback_replacing_tensor_storage_dtype_rejects_as_readonly_before_false_stop(api, dtype, tmp_path):
    operator, data, _ = _fixture()
    boundary = []

    def callback(iteration, residual, volume):
        boundary.append((iteration, len(operator.calls)))
        volume.data = volume.to(dtype=dtype)
        return False

    with pytest.raises(ValueError, match=READ_ONLY):
        _solve(api, operator, data, tmp_path, iterations=1, callback=callback)
    assert boundary == [(1, 3)]
    assert len(operator.calls) == 3, "storage replacement crossed a numerical boundary"


@pytest.mark.parametrize("api", ["cgls", "cgls_into"])
def test_mutation_cannot_escape_through_last_iteration_success_or_false_stop(api, tmp_path):
    operator, data, _ = _fixture()
    seen = []

    def callback(iteration, residual, volume):
        seen.append(iteration)
        _mutate(volume, "zero")
        return False

    with pytest.raises(ValueError, match=READ_ONLY):
        _solve(api, operator, data, tmp_path, iterations=1, callback=callback)
    assert seen == [1]


@pytest.mark.parametrize("api,kind", [("cgls", "tensor"), ("cgls_into", "npy")])
def test_checkpoint_before_rejected_callback_keeps_coherent_resumable_recurrence(api, kind, tmp_path):
    operator, data, _ = _fixture()
    checkpoint, observed = tmp_path / "checkpoint", []

    def callback(iteration, residual, volume):
        observed.append(_snapshot(volume))
        _mutate(volume, "zero" if api == "cgls" else "write")
        return False

    failed = tmp_path / "failed"
    failed.mkdir()
    with pytest.raises(ValueError, match=READ_ONLY):
        _solve(api, operator, data, failed, kind=kind, callback=callback,
               checkpoint_directory=checkpoint)
    saved = _checkpoint(checkpoint, 1)
    torch.testing.assert_close(torch.from_numpy(np.load(saved / "x.npy")), observed[0], rtol=0, atol=0)
    document = json.loads((saved / "metadata.json").read_text())
    assert document["iteration"] == 1 and len(document["residual_history"]) == 2
    assert not (checkpoint / "step-000002").exists()
    resumed_dir, baseline_dir = tmp_path / "resumed", tmp_path / "baseline"
    resumed_dir.mkdir()
    baseline_dir.mkdir()
    resumed = _solve(api, operator, data, resumed_dir, kind=kind, iterations=4, resume=saved)
    baseline = _solve(api, operator, data, baseline_dir, kind=kind, iterations=4)
    torch.testing.assert_close(_snapshot(resumed.volume), _snapshot(baseline.volume), rtol=0, atol=0)
    assert resumed.residual_history == baseline.residual_history


@pytest.mark.parametrize("api,kind", [("cgls", "tensor"), ("cgls_into", "tensor"),
                                      ("cgls_into", "tensorstore"), ("cgls_into", "npy"),
                                      ("cgls_into", "delegate-npy")])
def test_observer_callback_keeps_numerics_history_identity_and_resume(api, kind, tmp_path):
    operator, data, _ = _fixture()
    baseline_dir, partial_dir, resumed_dir = (tmp_path / name for name in ("baseline", "partial", "resumed"))
    for directory in (baseline_dir, partial_dir, resumed_dir):
        directory.mkdir()
    baseline = _solve(api, operator, data, baseline_dir, kind=kind, iterations=4)
    observed = []

    def callback(iteration, residual, volume):
        current = _snapshot(volume)
        actual = torch.linalg.vector_norm(data - (operator.matrix @ current.flatten()).reshape(data.shape)).item()
        assert residual == pytest.approx(actual, rel=3e-5, abs=3e-6)
        observed.append((iteration, residual, volume))
        return iteration < 2

    partial = _solve(api, operator, data, partial_dir, kind=kind, iterations=4,
                     callback=callback, checkpoint_directory=tmp_path / "checkpoint")
    assert partial.iterations == 2 and not partial.converged
    assert [item[0] for item in observed] == [1, 2]
    assert all(item[2] is observed[0][2] for item in observed)
    if api == "cgls_into":
        assert partial.volume is observed[-1][2]
    assert partial.residual_history[1:] == [item[1] for item in observed]
    final = _solve(api, operator, data, resumed_dir, kind=kind, iterations=4,
                   resume=_checkpoint(tmp_path / "checkpoint", 2), callback=lambda *args: None)
    torch.testing.assert_close(_snapshot(final.volume), _snapshot(baseline.volume), rtol=0, atol=0)
    assert final.residual_history == baseline.residual_history


@pytest.mark.parametrize("kind", ["npy", "delegate-npy"])
def test_mutating_npy_read_copy_is_a_harmless_readonly_observer(kind, tmp_path):
    operator, data, _ = _fixture()
    baseline_dir, observed_dir = tmp_path / "baseline", tmp_path / "observed"
    baseline_dir.mkdir()
    observed_dir.mkdir()
    baseline = _solve("cgls_into", operator, data, baseline_dir, kind=kind)

    def callback(iteration, residual, volume):
        # NpyStore.read deliberately returns a copy; this cannot change x.
        volume.read(ONE).zero_()

    observed = _solve("cgls_into", operator, data, observed_dir, kind=kind, callback=callback)
    torch.testing.assert_close(_snapshot(observed.volume), _snapshot(baseline.volume), rtol=0, atol=0)
    assert observed.residual_history == baseline.residual_history


@pytest.mark.parametrize("api", ["cgls", "cgls_into"])
def test_callback_exception_preserves_original_type_object_and_stops_work(api, tmp_path):
    operator, data, _ = _fixture()
    sentinel = RuntimeError("observer sentinel")

    def callback(*args):
        raise sentinel

    with pytest.raises(RuntimeError) as error:
        _solve(api, operator, data, tmp_path, callback=callback)
    assert error.value is sentinel
    assert len(operator.calls) == 3


@pytest.mark.parametrize("api", ["cgls", "cgls_into"])
def test_only_literal_false_stops_an_unmodified_callback(api, tmp_path):
    operator, data, _ = _fixture()
    result = _solve(api, operator, data, tmp_path, callback=lambda *args: 0)
    assert result.iterations == 3


@pytest.mark.parametrize("api", ["cgls", "cgls_into"])
def test_no_callback_does_not_add_iterate_fingerprinting(api, tmp_path):
    operator, data, _ = _fixture()
    with mock.patch.object(_solver_state, "fingerprint", side_effect=AssertionError("callback-free fingerprint")):
        result = _solve(api, operator, data, tmp_path)
    assert result.iterations == 3


def test_disk_observer_guard_preserves_block_bound_for_all_new_work(tmp_path):
    # Import-time caches are outside the solver's bounded working-set contract.
    import torch._dynamo

    operator, data, _ = _fixture()
    source = _SolverStore(diffct.TensorStore(data), 3)
    output = _output("delegate-npy", operator, tmp_path)
    with _DiskAllocations(3, forbid_full=(operator.volume_shape, operator.projection_shape, (6,), (15,))), \
            _numpy_blocks(3):
        result = diffct.cgls_into(operator, source, output, 3, workspace=tmp_path / "workspace",
                                  block_elements=3, rtol=0., atol=0., callback=lambda *args: None)
    assert result.volume is output and result.iterations == 3


@pytest.mark.parametrize("name", ["sirt", "tv_reconstruction"])
def test_other_solver_callbacks_retain_mutable_iterate_semantics(name):
    operator, data, _ = _fixture()

    def callback(iteration, residual, volume):
        volume.zero_()
        return False

    result = getattr(diffct, name)(operator, data, 3, rtol=0., atol=0., callback=callback)
    assert result.iterations == 1 and not result.converged
    assert torch.count_nonzero(result.volume).item() == 0


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="real native CUDA is required")
def test_native_cuda_cgls_observer_matches_baseline_and_mutation_is_rejected():
    case = _case("parallel", views=3)
    case.shape, case.detector = (2, 3), (4,)
    operator = _projector(case, backend="numba", devices=[0])
    truth = torch.tensor([[-.3, .7, 1.1], [.2, -.15, .9]], device="cuda:0")
    data = operator.project(truth)
    baseline = diffct.cgls(operator, data, 2, rtol=0., atol=0.)
    seen = []

    def observer(iteration, residual, volume):
        assert volume.device == torch.device("cuda:0")
        seen.append(iteration)

    observed = diffct.cgls(operator, data, 2, rtol=0., atol=0., callback=observer)
    assert seen == [1, 2]
    torch.testing.assert_close(observed.volume, baseline.volume, rtol=0, atol=0)
    assert observed.residual_history == baseline.residual_history

    def mutate(iteration, residual, volume):
        volume.zero_()
        return False

    with pytest.raises(ValueError, match=READ_ONLY):
        diffct.cgls(operator, data, 1, rtol=0., atol=0., callback=mutate)


def _observer_worker(rank, rendezvous, directory, api):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=GROUP_TIMEOUT))
    try:
        root = Path(directory)
        _, _, data, operator = _real_problem("views")
        local, observed_iterations = data[operator.view_slice].clone(), []

        def callback(iteration, residual, volume):
            observed_iterations.append(iteration)

        def solve(prefix, observer):
            options = dict(rtol=0., atol=0., callback=observer)
            if api == "cgls":
                return diffct.cgls(operator, local, 3, **options)
            output = diffct.NpyStore.create(root / f"{prefix}-output-{rank}.npy",
                                            operator.volume_shape, dtype=operator.dtype)
            return diffct.cgls_into(operator, local, output, 3,
                                    workspace=root / f"{prefix}-workspace-{rank}",
                                    block_elements=5, **options)

        baseline = solve("baseline", None)
        observed = solve("observer", callback if rank == 1 else None)
        _finish_record(root, rank, dict(iterations=observed.iterations,
                                       baseline_iterations=baseline.iterations,
                                       callback_iterations=observed_iterations,
                                       same_volume=torch.equal(_snapshot(observed.volume),
                                                               _snapshot(baseline.volume)),
                                       same_history=observed.residual_history == baseline.residual_history))
    finally:
        dist.destroy_process_group()


@GLOO
@pytest.mark.parametrize("api", ["cgls", "cgls_into"])
def test_clean_observer_on_one_rank_preserves_complete_solver_progress(api, tmp_path):
    records = _run_two(_observer_worker, tmp_path, api)
    assert all(record["iterations"] == record["baseline_iterations"] == 3 for record in records), records
    assert [record["callback_iterations"] for record in records] == [[], [1, 2, 3]], records
    assert all(record["same_volume"] and record["same_history"] for record in records), records


def _mutation_worker(rank, rendezvous, directory, api, stop):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=GROUP_TIMEOUT))
    try:
        root = Path(directory)
        _, _, data, operator = _real_problem("views")
        local, boundary = data[operator.view_slice].clone(), []

        def callback(iteration, residual, volume):
            boundary.append(iteration)
            _mutate(volume, "zero" if api == "cgls" else "write")
            return False if stop else None

        error, message = None, None
        output = (diffct.NpyStore.create(root / f"output-{rank}.npy", operator.volume_shape,
                                         dtype=operator.dtype) if api == "cgls_into" else None)
        with _numerical_calls(operator) as calls:
            started = time.monotonic()
            try:
                options = dict(rtol=0., atol=0., callback=callback if rank == 1 else None)
                if api == "cgls":
                    diffct.cgls(operator, local, 3, **options)
                else:
                    diffct.cgls_into(operator, local, output, 3,
                                     workspace=root / f"workspace-{rank}", block_elements=5, **options)
            except Exception as caught:
                error, message = type(caught).__name__, str(caught)
            elapsed = time.monotonic() - started
        _finish_record(root, rank, dict(error=error, message=message, calls=calls,
                                       callback_iterations=boundary, elapsed=elapsed))
    finally:
        dist.destroy_process_group()


@GLOO
@pytest.mark.parametrize("api", ["cgls", "cgls_into"])
@pytest.mark.parametrize("stop", [False, True], ids=["continue", "return-False"])
def test_one_rank_callback_mutation_rejects_every_rank_before_more_numerical_collectives(api, stop, tmp_path):
    records = _run_two(_mutation_worker, tmp_path, api, stop)
    assert [record["error"] for record in records] == ["ValueError", "ValueError"], records
    assert records[0]["message"] == records[1]["message"], records
    assert "read-only" in records[0]["message"].lower() or "read only" in records[0]["message"].lower()
    expected = (["backproject", "project", "backproject"] if api == "cgls" else
                ["backproject_into", "project_into", "backproject_into"])
    assert all(record["calls"] == expected for record in records), records
    assert [record["callback_iterations"] for record in records] == [[], [1]], records
    assert all(record["elapsed"] < REJECTION_TIMEOUT for record in records), records
