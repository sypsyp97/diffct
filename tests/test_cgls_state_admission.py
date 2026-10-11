"""CGLS admission must survive the first replan with its five states live.

Native trials own isolated processes and a real PyTorch allocator ceiling.
The CPU sum map is an independent numerical solver check, not CUDA evidence.
"""

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
import torch.multiprocessing as mp

from diffct import Projector
from diffct import chunking


MIB = 1024 ** 2
ENTRIES = 524288
cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="real native CUDA is required")


def _example(name):
    path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
    if not path.is_file():
        assert not (path.parents[1] / ".git").exists(), f"source checkout missing required example {path}"
        pytest.skip("example-side tests require checkout/copied examples; examples are excluded from standalone sdist")
    spec = importlib.util.spec_from_file_location(f"_cgls_admission_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _ObservedProjector(Projector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def _observe(self, operation, value, output):
        if not self.calls:
            self.first_call_segments = [dict(total_size=segment["total_size"], stream=segment["stream"],
                                            segment_type=segment["segment_type"],
                                            blocks=[dict(size=block["size"], state=block["state"])
                                                    for block in segment["blocks"]])
                                        for segment in chunking._native_segments(0)]
        self.calls.append(dict(
            operation=operation, input_device=value.device.type, output_device=output.device.type,
            allocated_bytes=torch.cuda.memory_allocated(0), reserved_bytes=torch.cuda.memory_reserved(0),
            reservation_budget_bytes=chunking._reservation_budget(0),
        ))

    def backproject_into(self, value, output):
        self._observe("backproject", value, output)
        return super().backproject_into(value, output)

    def project_into(self, value, output):
        self._observe("project", value, output)
        return super().project_into(value, output)


@torch.no_grad()
def _native_worker(rank, ceiling_mib, cached_mib, backend, directory):
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    assert torch.cuda.get_allocator_backend() == "native"
    ceiling = ceiling_mib * MIB
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(ceiling / total, 0)
    assert abs(torch.cuda.get_per_process_memory_fraction(0) * total - ceiling) < 1
    assert torch.cuda.memory_reserved(0) == 0, "trial requires a cold PyTorch allocator"
    if cached_mib:
        cached = torch.empty(cached_mib * MIB, dtype=torch.uint8, device="cuda:0")
        assert torch.cuda.memory_reserved(0) == cached_mib * MIB
        del cached
        torch.cuda.synchronize(0)
        assert torch.cuda.memory_allocated(0) == 0
        segments = chunking._native_segments(0)
        assert len(segments) == 1 and segments[0]["total_size"] == cached_mib * MIB
        assert all(block["state"] == "inactive" for block in segments[0]["blocks"])
    shape = (1, ENTRIES)
    # Kernel indexing is image[y, x]. This central horizontal ray traverses
    # every unit-width cell exactly once: A = [1,...,1], A* y = [y,...,y].
    geometry = (torch.tensor([[1., 0.]]), torch.zeros(1, 2), torch.tensor([[0., 1.]]))
    operator = _ObservedProjector(geometry, shape, (1,), beam="parallel", devices=[0],
                                 schedule="spatial", volume_chunk_shape=shape, view_chunk_size=1)
    measurements = torch.tensor([[2.]], device="cuda:0")
    original = measurements.cpu()
    initial_allocated = torch.cuda.memory_allocated(0)
    assert initial_allocated == 512 and torch.cuda.memory_reserved(0) == (cached_mib + 2) * MIB
    torch.cuda.reset_peak_memory_stats(0)
    evidence = dict(allocator_ceiling_bytes=ceiling, volume_bytes=4 * ENTRIES,
                    cached_segment_bytes=cached_mib * MIB, initial_segments=chunking._native_segments(0),
                    expected_state_backend=backend, calls=operator.calls)
    try:
        result = _example("_common").cgls(operator, measurements, 1)
        torch.cuda.synchronize(0)
        assert result.device.type == backend
        assert operator.last_solver_stats["state_backend"] == backend
        # Together result x and these actual p/q/r/s arguments observe all
        # five state devices. No allocator or numerical operation is mocked.
        assert [call["operation"] for call in operator.calls] == ["backproject", "project", "backproject"]
        assert all(call["input_device"] == call["output_device"] == backend for call in operator.calls)
        first = operator.calls[0]
        if backend == "cpu":
            assert first["allocated_bytes"] == initial_allocated, "CUDA states allocated before host fallback"
        else:
            assert first["allocated_bytes"] >= initial_allocated + 3 * 4 * ENTRIES
            assert first["reserved_bytes"] >= 22 * MIB, "positive control did not pin a middle-pool state segment"
        torch.testing.assert_close(result.cpu().double(),
                                   torch.full(shape, 2 / ENTRIES, dtype=torch.float64), rtol=5e-4, atol=1e-9)
        torch.testing.assert_close(measurements.cpu(), original)
        execution = operator.last_execution_stats
        assert execution is not None and execution["kernel_launches"] > 0
        evidence["execution"] = execution
    except BaseException as error:
        evidence["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        evidence["solver"] = getattr(operator, "last_solver_stats", None)
        evidence["first_call_segments"] = getattr(operator, "first_call_segments", [])
        evidence["peaks"] = {name: torch.cuda.memory_stats(0)[name + "_bytes.all.peak"]
                             for name in ("allocated", "active", "reserved")}
        Path(directory, "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    assert all(value <= ceiling for value in evidence["peaks"].values()), evidence


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("ceiling_mib,cached_mib,backend", [(32, 0, "cpu"), (64, 0, "cuda"),
                                                        (64, 20, "cuda"), (64, 48, "cpu")],
                         ids=["32mib-host-fallback-before-state-allocation", "64mib-cuda-positive-control",
                              "64mib-idle-20mib-segment-cuda-positive-control",
                              "64mib-idle-48mib-segment-host-fallback"])
def test_native_cgls_admission_survives_first_backprojection_with_live_states(ceiling_mib, cached_mib, backend, tmp_path):
    if torch.cuda.get_allocator_backend() != "native":
        pytest.skip("phase admission regression requires the native CUDA allocator")
    _example("_common")  # Preserve standalone-sdist skipping before starting a worker.
    mp.spawn(_native_worker, args=(ceiling_mib, cached_mib, backend, str(tmp_path)), nprocs=1, join=True)


class _CpuSum:
    """Independent real CPU map A=sum and A*=replicate for one measurement."""
    world_size, rank, process_group, partition = 1, 0, None, "views"
    volume_shape, projection_shape = (1, 7), (1, 1)

    def __init__(self):
        self.calls = []

    def backproject_into(self, value, output):
        assert value.device.type == output.device.type == "cpu"
        self.calls.append("backproject")
        output.copy_(value.expand(self.volume_shape))
        return output

    def project_into(self, value, output):
        assert value.device.type == output.device.type == "cpu"
        self.calls.append("project")
        output.copy_(value.sum().reshape(self.projection_shape))
        return output


def test_cpu_cgls_sum_map_keeps_all_states_on_host_and_matches_one_step_solution():
    operator = _CpuSum()
    measurements = torch.tensor([[2.]])
    result = _example("_common").cgls(operator, measurements, 1)
    torch.testing.assert_close(result, torch.full(operator.volume_shape, 2 / 7))
    torch.testing.assert_close(measurements, torch.tensor([[2.]]))
    assert result.device.type == operator.last_solver_stats["state_backend"] == "cpu"
    assert operator.last_solver_stats["estimated_cuda_state_bytes"] == 0
    assert operator.calls == ["backproject", "project", "backproject"]
