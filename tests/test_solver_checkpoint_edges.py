"""Precision and distributed checkpoint boundaries, with real solver numerics."""

from contextlib import ExitStack, contextmanager
from datetime import timedelta
import json
import os
from pathlib import Path
import time
from unittest import mock

import numpy as np
import pytest
import torch
import torch.distributed as dist

import diffct
from tests.test_block_storage import _numpy_blocks
from tests.test_detector_surfaces import _arc_offsets, _arc_surface, _case, _grids, _matrix
from tests.test_execution_features import GLOO, _hybrid, _run_four
from tests.test_solver_api import (
    _CheckpointWrites, _DiskAllocations, _MatrixOperator, _SolverStore,
    _cgls_reference, _checkpoint,
)


def _matrix_problem(dtype):
    operator = _MatrixOperator(dtype=dtype)
    operator.dtype, operator.precision, operator.backend = dtype, str(dtype).removeprefix("torch."), "torch"
    truth = np.array([-.3, .7, 1.1, .2, -.15, .9], dtype=np.float64)
    measurements = torch.from_numpy((operator.matrix.numpy() @ truth).reshape(operator.projection_shape)).to(dtype)
    return operator, measurements


@contextmanager
def _numerical_calls(operator):
    calls = []
    with ExitStack() as stack:
        for name in ("project", "backproject", "project_into", "backproject_into"):
            original = getattr(operator, name)

            def observe(*args, _name=name, _original=original, **kwargs):
                calls.append(_name)
                return _original(*args, **kwargs)

            stack.enter_context(mock.patch.object(operator, name, observe))
        yield calls


def test_double_disk_cgls_keeps_recurrence_state_and_resume_double_and_block_bounded(tmp_path):
    operator, measurements = _matrix_problem(torch.float64)
    expected, history, snapshots = _cgls_reference(operator.matrix.numpy(), measurements.numpy(), 4)
    np.save(tmp_path / "measurements.npy", measurements.numpy())
    source = _SolverStore(diffct.NpyStore(tmp_path / "measurements.npy", mode="r"), 5)
    outputs = {name: diffct.NpyStore.create(tmp_path / f"{name}.npy", operator.volume_shape,
                                          dtype=torch.float64)
               for name in ("full", "partial", "resumed")}
    original_read, original_write = diffct.NpyStore.read, diffct.NpyStore.write

    def read(store, selected):
        value = original_read(store, selected)
        assert value.numel() <= 5, "workspace/checkpoint/fingerprint read exceeded block_elements"
        return value

    def write(store, selected, value, **kwargs):
        assert value.numel() <= 5, "workspace/checkpoint update exceeded block_elements"
        return original_write(store, selected, value, **kwargs)

    with mock.patch.object(diffct.NpyStore, "read", read), mock.patch.object(diffct.NpyStore, "write", write), \
            _DiskAllocations(5, forbid_full=(operator.volume_shape, operator.projection_shape, (6,), (15,))), \
            _numpy_blocks(5):
        options = dict(block_elements=5, rtol=0., atol=0.)
        full = diffct.cgls_into(operator, source, outputs["full"], 4,
                                workspace=tmp_path / "full-work", **options)
        partial = diffct.cgls_into(operator, source, outputs["partial"], 2,
                                   workspace=tmp_path / "partial-work",
                                   checkpoint_directory=tmp_path / "checkpoint", **options)
        resumed = diffct.cgls_into(operator, source, outputs["resumed"], 4,
                                   workspace=tmp_path / "resume-work",
                                   resume=_checkpoint(tmp_path / "checkpoint", 2), **options)
    for name, result, reference in (("full", full, expected), ("partial", partial, snapshots[1]),
                                     ("resumed", resumed, expected)):
        assert result.volume is outputs[name]
        np.testing.assert_allclose(np.load(outputs[name].path).ravel(), reference, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(result.residual_history, history[:result.iterations + 1], rtol=1e-11, atol=1e-12)
    for path in (*tmp_path.glob("*-work/*.npy"), *tmp_path.glob("checkpoint/step-*/rank-*/*.npy")):
        assert np.load(path, mmap_mode="r").dtype == np.float64, f"double solver state was narrowed: {path}"
    np.testing.assert_array_equal(np.load(tmp_path / "measurements.npy"), measurements.numpy())


@pytest.mark.parametrize("disk", [False, True], ids=["tensor", "mapped"])
def test_same_float32_measurements_cannot_resume_with_changed_compute_precision(disk, tmp_path):
    operator, measurements = _matrix_problem(torch.float32)
    if disk:
        np.save(tmp_path / "measurements.npy", measurements.numpy())
        source = diffct.NpyStore(tmp_path / "measurements.npy", mode="r")
        first = diffct.NpyStore.create(tmp_path / "first.npy", operator.volume_shape)
        diffct.cgls_into(operator, source, first, 2, workspace=tmp_path / "first-work",
                         block_elements=5, rtol=0., atol=0., checkpoint_directory=tmp_path / "checkpoint")
    else:
        diffct.cgls(operator, measurements, 2, rtol=0., atol=0., checkpoint_directory=tmp_path / "checkpoint")
    operator.dtype, operator.precision = torch.float64, "float64"
    operator.calls.clear()
    sentinel = torch.full(operator.volume_shape, 83., dtype=torch.float64)
    with pytest.raises((ValueError, TypeError)):
        if disk:
            diffct.cgls_into(operator, source, sentinel, 3, workspace=tmp_path / "changed-work",
                             block_elements=5, rtol=0., atol=0., resume=_checkpoint(tmp_path / "checkpoint", 2))
        else:
            diffct.cgls(operator, measurements, 3, rtol=0., atol=0., resume=_checkpoint(tmp_path / "checkpoint", 2))
    assert operator.calls == [], "precision mismatch was rejected after numerical execution"
    torch.testing.assert_close(sentinel, torch.full_like(sentinel, 83.), rtol=0, atol=0)


def _sample_arc(u, v, views, radius):
    return torch.stack((radius * torch.sin(u / radius) + .02 * u.square(), v,
                        radius * (1 - torch.cos(u / radius))), dim=-1)


def _sample_changed_arc(u, v, views, radius):
    return torch.stack((radius * torch.sin(u / radius) + .12 * u.square(), v,
                        radius * (1 - torch.cos(u / radius))), dim=-1)


@pytest.mark.parametrize("beam", ["parallel", "fan"])
@pytest.mark.parametrize("surface_kind", ["parameterized", "captured-callback"])
def test_resume_fingerprints_effective_surface_pixels_even_when_explicit_parameters_are_unchanged(
    beam, surface_kind, tmp_path,
):
    case = _case(beam, views=5)
    case.shape, case.detector = (3, 4), (5,)
    radius = torch.tensor(3.7, dtype=torch.float64)
    coefficient = torch.tensor(.02, dtype=torch.float64)
    surface = (diffct.ParameterizedSurface(_sample_arc, (radius,)) if surface_kind == "parameterized"
               else _arc_surface(radius, coefficient))
    operator = diffct.Projector(case.trajectory, case.shape, case.detector, beam=beam,
                                detector_spacing=case.pitch[0], voxel_spacing=case.spacing,
                                detector_surface=surface, backend="torch", precision="float64",
                                view_chunk_size=2, detector_chunk_shape=(2,), volume_chunk_shape=(2, 2))
    matrix = _matrix(case, _arc_offsets(case, float(radius), .02))
    truth = np.linspace(-.3, .9, np.prod(case.shape))
    measurements = torch.from_numpy((matrix @ truth).reshape(case.sino_shape))
    diffct.cgls(operator, measurements, 2, rtol=0., atol=0., checkpoint_directory=tmp_path / "checkpoint")
    saved = _checkpoint(tmp_path / "checkpoint", 2)
    unchanged = diffct.cgls(operator, measurements, 3, rtol=0., atol=0., resume=saved)
    expected, history, _ = _cgls_reference(matrix, measurements.numpy(), 3)
    np.testing.assert_allclose(unchanged.volume.numpy().ravel(), expected, rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(unchanged.residual_history, history, rtol=2e-11, atol=2e-12)
    if surface_kind == "parameterized":
        operator.detector_surface = diffct.ParameterizedSurface(_sample_changed_arc, surface.parameters)
        assert operator.detector_surface.parameters[0] is radius
    else:
        coefficient.add_(.10)
        assert operator.detector_surface is surface  # Same function code and closure identities.
    changed_matrix = _matrix(case, _arc_offsets(case, float(radius), .12))
    assert np.max(np.abs(changed_matrix - matrix)) > 1e-3, "surface change does not change physical rays"
    with _numerical_calls(operator) as calls:
        with pytest.raises((ValueError, TypeError)):
            diffct.cgls(operator, measurements, 3, rtol=0., atol=0., resume=saved)
    assert calls == [], "effective surface mismatch was rejected after numerical projection"


def _real_problem(partition):
    case = _case("parallel", views=5)
    case.shape, case.detector = (4, 3), (5,)
    u, v = _grids(case)
    matrix = _matrix(case, np.stack((u, v, np.zeros_like(u)), -1))
    truth = np.linspace(-.3, .9, np.prod(case.shape))
    data = torch.from_numpy((matrix @ truth).reshape(case.sino_shape))
    operator = (_hybrid(case) if partition == "hybrid" else diffct.Projector(
        case.trajectory, case.shape, case.detector, beam=case.beam,
        voxel_spacing=case.spacing, detector_spacing=case.pitch[0], backend="torch", precision="float64",
        distributed=True, partition=partition, view_chunk_size=1, volume_chunk_shape=(1, 1)))
    return case, matrix, data, operator


def _usable_parent_group():
    reached = torch.tensor(1)
    try:
        dist.all_reduce(reached)
    except RuntimeError:
        return False
    return reached.item() == 4


def _initial_worker(rank, rendezvous, directory, partition):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=15))
    records = []
    try:
        root = Path(directory)
        case, matrix, data, operator = _real_problem(partition)
        local = data[operator.view_slice].clone()
        expected, history, _ = _cgls_reference(matrix, data.numpy(), 2)
        positive = diffct.cgls(operator, local, 2, rtol=0., atol=0., checkpoint_directory=root / "positive")
        torch.testing.assert_close(positive.volume, torch.from_numpy(expected.reshape(case.shape))[operator.volume_slice],
                                   rtol=3e-11, atol=3e-12)
        np.testing.assert_allclose(positive.residual_history, history, rtol=3e-11, atol=3e-12)
        records.append(dict(case="valid", iterations=positive.iterations, usable=_usable_parent_group(),
                            view_rank=operator.view_rank, measurement_norm=float(local.norm())))
        for failure in (("replica", "scan") if partition in ("space", "hybrid") else ("scan",)):
            invalid = local.clone()
            if rank == 3:
                if failure == "replica":
                    invalid[0, 0] += .1
                else:
                    operator._trajectory[1][operator.view_slice.start, 0] += .08
            error = None
            with _numerical_calls(operator) as calls:
                try:
                    diffct.cgls(operator, invalid, 2, rtol=0., atol=0., checkpoint_directory=root / failure)
                except (ValueError, TypeError, OSError, RuntimeError) as caught:
                    error = type(caught).__name__
            records.append(dict(case=failure, error=error, calls=calls, usable=_usable_parent_group()))
        (root / f"initial-rank-{rank}.json").write_text(json.dumps(records), encoding="utf-8")
    finally:
        dist.destroy_process_group()


@GLOO
@pytest.mark.parametrize("partition", ["views", "space", "hybrid"])
def test_four_rank_initial_checkpoint_validates_scan_and_measurement_replicas_before_calls(partition, tmp_path):
    _run_four(_initial_worker, ((tmp_path / "initial-rendezvous").resolve().as_uri(), str(tmp_path), partition), timeout=45)
    records = [json.loads((tmp_path / f"initial-rank-{rank}.json").read_text()) for rank in range(4)]
    assert all(rank[0]["iterations"] == 2 and rank[0]["usable"] for rank in records)
    if partition != "space":
        assert len({rank[0]["measurement_norm"] for rank in records}) > 1, "positive fixture must contain distinct view shards"
    for rank, checks in enumerate(records):
        for check in checks[1:]:
            assert check["error"] in ("ValueError", "TypeError"), (partition, rank, check)
            assert check["calls"] == [], ("initial validation executed a numerical operator", partition, rank, check)
            assert check["usable"], ("initial validation damaged the parent group", partition, rank, check)


def _publication_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=15))
    try:
        root = Path(directory)
        case, _, data, operator = _real_problem("hybrid")
        original_path_replace, original_os_replace = Path.replace, os.replace
        observed = {}
        error = None

        def publication(target):
            target = Path(target).resolve()
            if target.name == "metadata.json" and target.parent.parent.name == "step-000002":
                observed.setdefault("calls_at_publication", len(calls))
                if rank == 3:
                    raise OSError("injected rank3 atomic metadata publication failure")

        def replace_path(source, target):
            publication(target)
            return original_path_replace(source, target)

        def replace_os(source, target, *args, **kwargs):
            publication(target)
            return original_os_replace(source, target, *args, **kwargs)

        with _numerical_calls(operator) as calls, \
                mock.patch.object(Path, "replace", replace_path), mock.patch.object(os, "replace", replace_os), \
                _CheckpointWrites(root, case, rank) as writes:
            started = time.monotonic()
            try:
                diffct.cgls(operator, data[operator.view_slice].clone(), 4, rtol=0., atol=0.,
                            checkpoint_directory=root / "checkpoint")
            except (ValueError, TypeError, OSError, RuntimeError) as caught:
                error = type(caught).__name__
            elapsed = time.monotonic() - started
        record = dict(error=error, elapsed=elapsed, calls=calls, **observed,
                      complete_state_observations=len(writes.commits), usable=_usable_parent_group())
        (root / f"publication-rank-{rank}.json").write_text(json.dumps(record), encoding="utf-8")
    finally:
        dist.destroy_process_group()


@GLOO
def test_four_rank_atomic_metadata_failure_rejects_collectively_before_more_operators_and_keeps_parent_group(tmp_path):
    _run_four(_publication_worker, ((tmp_path / "publication-rendezvous").resolve().as_uri(), str(tmp_path)), timeout=45)
    records = [json.loads((tmp_path / f"publication-rank-{rank}.json").read_text()) for rank in range(4)]
    assert all(record["error"] in ("ValueError", "OSError") and record["usable"] for record in records), records
    for record in records:
        assert record["calls_at_publication"] == 5, "failure fixture must complete exactly two CGLS iterations"
        assert len(record["calls"]) == record["calls_at_publication"], "rank executed another operator after publication failure"
        assert record["complete_state_observations"] >= 2, "missing all-rank write/flush evidence before step2 publication"
        assert record["elapsed"] < 15, "publication failure waited for the Gloo timeout"
