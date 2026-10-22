"""Fixed-geometry distributed data transforms use the global physical map.

All ranks supply coherent owned pieces of the same values/directions. Output
replicas divide the scalar loss once; input replicas divide the ordinary
second-derivative cotangent once. No per-rank Jacobian convention is assumed.
"""

from contextlib import nullcontext
from datetime import timedelta
import json
from pathlib import Path
import time
from unittest import mock

import pytest
import torch
import torch.distributed as dist
from tests._multiprocessing import spawn

from diffct import Projector
from diffct.utils import TorchCUDABridge
from tests.test_detector_surfaces import _case as _geometry_case
from tests.test_reference_backend import CUDA_REQUIRED, _Case, _data, _matrix


GLOO = pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(),
                          reason="real Gloo process groups required")
TRANSFORMS = ("ordinary", "grad", "jvp", "vmap", "hvp")


def _surface(u, v):
    return torch.stack((u - .03 * u.pow(3), v, .15 * u.square()), dim=-1)


def _case(beam, first, views):
    geometry = _geometry_case(beam, views=views)
    return _Case(beam, geometry.trajectory,
                 (first, 3, 4) if beam == "cone" else (first, 4),
                 (3, 2) if beam == "cone" else (5,), geometry.pitch, geometry.spacing)


def _operator(case, partition, backend):
    return Projector(
        case.trajectory, case.shape, case.detector, beam=case.beam,
        detector_spacing=case.pitch if case.beam == "cone" else case.pitch[0],
        voxel_spacing=case.spacing, backend=backend,
        precision="float64" if backend == "torch" else "fast",
        distributed=True, partition=partition,
        partition_grid=(2, 2) if partition == "hybrid" else None,
        detector_surface=None if case.beam == "parallel" else _surface,
        view_chunk_size=1, volume_chunk_shape=(1,) * len(case.shape),
        detector_chunk_shape=(2,) * len(case.detector),
    )


def _close(actual, expected, value):
    assert actual.shape == expected.shape
    assert actual.dtype == value.dtype and actual.device == value.device
    tolerance = 4e-5 if value.dtype == torch.float32 else 3e-11
    torch.testing.assert_close(actual.detach().cpu().double(), expected.double(),
                               rtol=tolerance, atol=tolerance)


def _check(operator, case, matrix, operation, transform, device):
    forward = operation == "project"
    function = getattr(operator, operation)
    matrix = matrix if forward else matrix.T
    global_input, global_output = ((case.shape, case.projection_shape) if forward else
                                   (case.projection_shape, case.shape))
    ray_slice = (operator.view_slice, *(slice(0, size) for size in case.detector))
    input_slice, output_slice = ((operator.volume_slice, ray_slice) if forward else
                                 (ray_slice, operator.volume_slice))
    output_replicas = operator.space_world_size if forward else operator.view_world_size
    input_replicas = operator.view_world_size if forward else operator.space_world_size
    full_value = _data(global_input, phase=.13)
    full_direction = _data(global_input, phase=.79)
    full_weight = _data(global_output, phase=.31)
    value = full_value[input_slice].to(device=device, dtype=operator.dtype)
    direction = full_direction[input_slice].to(device=device, dtype=operator.dtype)
    weight = full_weight[output_slice].to(device=device, dtype=operator.dtype)
    expected_value = (matrix @ full_value.flatten()).reshape(global_output)[output_slice]
    expected_gradient = (matrix.T @ full_weight.flatten()).reshape(global_input)[input_slice]

    def loss(x, w=weight):
        return (function(x) * w).sum() / output_replicas

    # Each rank validates the ordinary map and VJP before testing its transform.
    leaf = value.clone().requires_grad_()
    ordinary_value = function(leaf)
    ordinary_gradient, = torch.autograd.grad(loss(leaf), leaf)
    _close(ordinary_value, expected_value, value)
    _close(ordinary_gradient, expected_gradient, value)
    if transform == "ordinary":
        return

    guard = (mock.patch.object(TorchCUDABridge, "tensor_to_cuda_array",
                               side_effect=AssertionError("data transform entered native CUDA bridge"))
             if operator.backend == "numba" else nullcontext())
    if transform == "grad":
        with guard:
            actual = torch.func.grad(loss)(value)
        _close(actual, expected_gradient, value)
        _close(actual, ordinary_gradient.detach().cpu().double(), value)
    elif transform == "jvp":
        ordinary_direction = function(direction)
        expected_direction = (matrix @ full_direction.flatten()).reshape(global_output)[output_slice]
        _close(ordinary_direction, expected_direction, value)
        with guard:
            primal, derivative = torch.func.jvp(function, (value,), (direction,))
        _close(primal, expected_value, value)
        _close(derivative, expected_direction, value)
        _close(derivative, ordinary_direction.detach().cpu().double(), value)
    elif transform == "vmap":
        full_batches = _data((2, 3, *global_input), phase=.43)
        batches = full_batches[(..., *input_slice)].to(device=device, dtype=operator.dtype)
        ordinary = torch.stack([function(x) for x in batches.reshape(6, *value.shape)]).reshape(
            2, 3, *ordinary_value.shape,
        )
        expected = (full_batches.reshape(6, -1) @ matrix.T).reshape(2, 3, *global_output)
        expected = expected[(..., *output_slice)]
        _close(ordinary, expected, value)
        full_weights = _data((2, *global_output), phase=.27)
        weights = full_weights[(..., *output_slice)].to(device=device, dtype=operator.dtype)
        expected_gradients = (full_weights.reshape(2, -1) @ matrix).reshape(2, *global_input)
        expected_gradients = expected_gradients[(..., *input_slice)]
        with guard:
            actual = torch.func.vmap(torch.func.vmap(function))(batches)
            gradients = torch.func.vmap(torch.func.grad(loss, argnums=0))(batches[:, 0], weights)
        _close(actual, expected, value)
        _close(actual, ordinary.detach().cpu().double(), value)
        _close(gradients, expected_gradients, value)
    else:
        assert transform == "hvp"

        def quadratic_loss(x):
            return function(x).square().sum() / (2 * output_replicas)

        gram = matrix.T @ matrix
        expected_first = (gram @ full_value.flatten()).reshape(global_input)[input_slice]
        expected_second = (gram @ full_direction.flatten()).reshape(global_input)[input_slice]
        leaf = value.clone().requires_grad_()
        first, = torch.autograd.grad(quadratic_loss(leaf), leaf, create_graph=True)
        second, = torch.autograd.grad((first * direction).sum() / input_replicas, leaf)
        _close(first, expected_first, value)
        _close(second, expected_second, value)
        with guard:
            actual_first, actual_second = torch.func.jvp(torch.func.grad(quadratic_loss),
                                                       (value,), (direction,))
        _close(actual_first, expected_first, value)
        _close(actual_second, expected_second, value)
        _close(actual_second, second.detach().cpu().double(), value)
    assert operator.last_execution_stats["backend"] == "torch"


def _worker(rank, rendezvous, world_size, partition, transform, native, directory):
    torch.set_num_threads(1)
    if native:
        # Two processes share one physical GPU; tensors/tiles are only tiny fixtures.
        torch.cuda.set_device(0)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=60))
    try:
        records = []
        configurations = (("parallel", 5, 5), ("fan", 3, 5), ("cone", 3, 5))
        if not native:
            configurations += (("parallel", 1, 1),)
        for backend in (("auto", "numba") if native else ("torch",)):
            for beam, first, views in configurations:
                case = _case(beam, first, views)
                matrix = _matrix(case, curved=beam != "parallel")
                assert matrix.abs().sum() > 0
                assert all(not component.requires_grad for component in case.trajectory)
                operator = _operator(case, partition, backend)
                assert operator.backend == ("numba" if native else "torch")
                assert operator.precision == ("fast" if native else "float64")
                assert operator.dtype == (torch.float32 if native else torch.float64)
                for operation in ("project", "backproject"):
                    _check(operator, case, matrix, operation, transform,
                           "cuda:0" if native else "cpu")
                records.append(dict(beam=beam, first=first, views=views, backend=backend,
                                    volume_shape=operator.volume_shape,
                                    projection_shape=operator.projection_shape))
        (Path(directory) / f"rank-{rank}.json").write_text(json.dumps(records), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def _run(tmp_path, partition, transform, *, native=False):
    world_size = 2 if native else 4
    rendezvous = (tmp_path / "rendezvous").resolve().as_uri()
    context = spawn(_worker, args=(rendezvous, world_size, partition, transform,
                                     native, str(tmp_path)), nprocs=world_size, join=False)
    deadline = time.monotonic() + 180
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise AssertionError("distributed data transform exceeded 180 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    records = [json.loads((tmp_path / f"rank-{rank}.json").read_text())
               for rank in range(world_size)]
    assert all(len(record) == (6 if native else 4) for record in records)
    if not native:
        if partition in ("space", "hybrid"):
            assert any(record[-1]["volume_shape"][0] == 0 for record in records)
        if partition in ("views", "hybrid"):
            assert any(record[-1]["projection_shape"][0] == 0 for record in records)


@GLOO
@pytest.mark.parametrize("partition", ["views", "space", "hybrid"])
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_four_rank_data_transforms_match_global_linear_map(partition, transform, tmp_path):
    _run(tmp_path, partition, transform)


@GLOO
@pytest.mark.cuda
@CUDA_REQUIRED
@pytest.mark.parametrize("partition", ["views", "space"])
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_native_distributed_data_transforms_route_portable_and_preserve_metadata(
    partition, transform, tmp_path,
):
    _run(tmp_path, partition, transform, native=True)
