"""Learned analytical overrides must share collective order on empty ranks."""

from datetime import timedelta
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.test_execution_features import (
    BEAMS, GLOO, _analytical_oracle, _overrides, _owned, _random,
    _reconstruct, _run_four, _scan,
)


def _learned_weights_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=10))
    path = Path(directory) / f"weights-{rank}.json"
    completed = []
    try:
        partitions = (("views", None), ("space", None), ("hybrid", (2, 2)))
        scenarios = [(beam, partition, grid, False) for beam in BEAMS for partition, grid in partitions]
        scenarios += [("parallel", partition, grid, True) for partition, grid in partitions]
        for beam, partition, grid, empty in scenarios:
            case = _scan(beam)
            # Five slabs and seven views make every control rank nonempty,
            # with unequal ownership. One slab/view then exercises empty ranks.
            case = {**case, "shape": (1, 4) if empty else (5, *case["shape"][1:])}
            if empty:
                case = {**case, "angles": case["angles"][:1],
                        "trajectory": tuple(value[:1] for value in case["trajectory"]),
                        "sino_shape": (1, *case["sino_shape"][1:])}
            view_ranks, space_ranks = grid or ((4, 1) if partition == "views" else (1, 4))
            view_rank, space_rank = divmod(rank, space_ranks)
            shard = _owned(len(case["angles"]), view_rank, view_ranks)
            slab = (_owned(case["shape"][0], space_rank, space_ranks),
                    *(slice(0, size) for size in case["shape"][1:]))
            oracle_data = _random(case["sino_shape"], 451, grad=True)
            oracle_weights = _overrides(case, grad=True)
            expected = _analytical_oracle(case, oracle_data, **oracle_weights)
            cotangent = _random(case["shape"], 452)
            expected_gradients = torch.autograd.grad(expected, (oracle_data, *oracle_weights.values()), cotangent)
            data = oracle_data.detach()[shard].clone().requires_grad_()
            weights = {name: value.detach().clone().requires_grad_() for name, value in oracle_weights.items()}
            actual = _reconstruct(case, data, backend="torch", precision="float64", distributed=True,
                                  partition=partition, partition_grid=grid, view_chunk_size=2,
                                  volume_chunk_shape=(1,) * len(case["shape"]),
                                  isocenter=case["center"], **weights)
            torch.testing.assert_close(actual, expected.detach()[slab], rtol=2e-11, atol=2e-12)
            current = dict(beam=beam, partition=partition, empty=empty,
                           views=data.shape[0], first=actual.shape[0], phase="backward")
            path.write_text(json.dumps(dict(completed=completed, current=current)), encoding="utf-8")
            gradients = torch.autograd.grad((actual * cotangent[slab]).sum() / view_ranks,
                                            (data, *weights.values()))
            torch.testing.assert_close(gradients[0], expected_gradients[0][shard], rtol=3e-11, atol=3e-12)
            for actual_gradient, expected_gradient in zip(gradients[1:], expected_gradients[1:]):
                assert expected_gradient.abs().max() > 0
                torch.testing.assert_close(actual_gradient, expected_gradient, rtol=3e-11, atol=3e-12)
            completed.append({**current, "phase": "passed"})
            path.write_text(json.dumps(dict(completed=completed, current=None)), encoding="utf-8")
    finally:
        dist.destroy_process_group()


@GLOO
def test_four_rank_learned_analytical_weight_and_data_gradients_survive_uneven_and_empty_ownership(tmp_path):
    try:
        # The shared parent deadline includes Windows spawning and child-process exit.
        _run_four(_learned_weights_worker,
                  ((tmp_path / "weights-rendezvous").resolve().as_uri(), str(tmp_path)))
    except (mp.ProcessExitedException, mp.ProcessRaisedException) as error:
        states = [json.loads(path.read_text()) for path in sorted(tmp_path.glob("weights-*.json"))]
        progress = [(len(state["completed"]), state["current"]) for state in states]
        pytest.fail(f"real Gloo analytical learned-weight backward failed; rank progress={progress}; {error}",
                    pytrace=False)
    states = [json.loads((tmp_path / f"weights-{rank}.json").read_text()) for rank in range(4)]
    assert all(len(state["completed"]) == 12 and state["current"] is None for state in states)
    assert any(record["views"] == 0 for state in states for record in state["completed"])
    assert any(record["first"] == 0 for state in states for record in state["completed"])
