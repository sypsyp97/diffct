"""Completed CUDA transfers restore capacity; unfinished transfers retain data.

Native tests own an isolated process and an actual per-process allocator cap.
They bound allocation/launch inventory rather than machine-dependent runtime.
"""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from diffct import Projector, circular_trajectory_3d
from diffct import chunking
from tests.test_allocator_capacity import _box_chords


MIB = 1024 ** 2
cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _allocator_state():
    stats = torch.cuda.memory_stats()
    segments = torch.cuda.memory_snapshot()
    states = Counter()
    for segment in segments:
        for block in segment["blocks"]:
            states[block["state"]] += block["size"]
    return dict(allocated_bytes=stats["allocated_bytes.all.current"],
                active_bytes=stats["active_bytes.all.current"],
                reserved_bytes=stats["reserved_bytes.all.current"],
                peak_reserved_bytes=stats["reserved_bytes.all.peak"],
                snapshot_block_bytes_by_state=dict(states),
                reservation_room_bytes=chunking._reservation_budget(torch.device("cuda:0")))


@torch.no_grad()
def _completed_reuse_worker(rank, directory):
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    ceiling = 32 * MIB
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(ceiling / total, 0)
    shape, detector, views = (256, 256, 256), (512, 512), 4
    trajectory = circular_trajectory_3d(views, 640., 1024., device="cpu")
    operator = Projector(trajectory, shape, detector, detector_spacing=.8,
                         devices=[0], view_chunk_size=4)
    expected = _box_chords(trajectory, shape, detector, .8)
    weights = torch.linspace(.17, .83, expected.size).reshape(operator.projection_shape)
    image = torch.ones(shape)
    back, rays = torch.empty(shape), torch.empty(operator.projection_shape)
    records = []
    evidence = Path(directory) / "completed-reuse-evidence.json"
    torch.cuda.reset_peak_memory_stats()
    for repetition in range(2):
        assert operator.backproject_into(weights, back) is back
        torch.cuda.synchronize()
        assert torch.isfinite(back).all() and back.abs().sum() > 0
        assert back.double().sum().item() == pytest.approx(
            float(np.sum(expected * weights.numpy())), rel=4e-4, abs=2e-3)
        baseline = dict(operator.last_execution_stats)
        state = _allocator_state()
        records.append(dict(repetition=repetition, operation="backproject", allocator=state, execution=baseline))
        evidence.write_text(json.dumps(records, indent=2), encoding="utf-8")
        assert state["allocated_bytes"] == 0, "CPU-only invocation retained live CUDA payload"
        # A completed call has no caller CUDA allocation. The same known-feasible
        # 20-MiB middle segment plus 2-MiB geometry segment must be admissible.
        assert state["reservation_room_bytes"] >= 22 * MIB, \
            ("completed transfers left stale pinned allocator events", state)
        assert operator.project_into(image, rays) is rays
        torch.testing.assert_close(rays.double(), torch.from_numpy(expected), rtol=4e-4, atol=2e-3)
        current = operator.last_execution_stats
        assert tuple(current["detector_chunk_shape"]) == detector
        assert current["view_chunk_size"] == 4
        assert math.prod(current["chunk_shape"]) >= math.prod(baseline["chunk_shape"])
        assert current["kernel_launches"] <= 16, "completed-call admission collapsed into tiny spatial/pixel jobs"
        state = _allocator_state()
        records.append(dict(repetition=repetition, operation="project", allocator=state, execution=current))
        evidence.write_text(json.dumps(records, indent=2), encoding="utf-8")
        assert state["peak_reserved_bytes"] <= ceiling


@pytest.mark.cuda
@cuda_required
def test_completed_backproject_restores_capacity_for_full_detector_forward_and_repetition(tmp_path):
    if torch.cuda.get_allocator_backend() != "native":
        pytest.skip("completed allocator-event retirement regression requires the native CUDA allocator")
    mp.spawn(_completed_reuse_worker, args=(str(tmp_path),), nprocs=1, join=True)
