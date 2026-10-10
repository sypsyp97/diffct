"""Resident tile/ray contract observed through real public Projector calls.

Kernel and bridge proxies always call the originals. Dispatch records contain
metadata only, so observations do not retain CUDA tensors or replace numerics.
The independent float64 cell oracle is reused from detector-surface tests.
"""

from collections import Counter, defaultdict
from contextlib import ExitStack
import gc
from itertools import product
import math
import sys
from unittest import mock

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from diffct import kernels
from diffct import pipeline
from diffct.chunking import _working_set_bytes
from diffct.utils import TorchCUDABridge
from tests.test_chunked_projector import _boundary_case, _offsets, _surface
from tests.test_detector_surfaces import (
    _case, _close, _data, _finite_difference, _matrix, _numpy, _projector,
)


BEAMS = ("parallel", "fan", "cone")
cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _tensor_metadata(tensor):
    storage = tensor.untyped_storage()
    return {
        "device": tensor.device,
        "shape": tuple(tensor.shape),
        "strides": tuple(tensor.stride()),
        "contiguous": tensor.is_contiguous(),
        "storage": storage._cdata,
        "pointer": tensor.data_ptr(),
        "bytes": tensor.numel() * tensor.element_size(),
        "storage_bytes": storage.nbytes(),
        "pinned": tensor.device.type == "cpu" and tensor.is_pinned(),
    }


class _ExecutionTrace(TorchDispatchMode):
    def __init__(self, case, chunk=None, batch=None, host_inputs=()):
        super().__init__()
        self.case, self.chunk, self.batch = case, chunk, batch
        self.phase = "operation"
        self.operations, self.launches = [], []
        self.allocations, self.bridged = {}, {}
        self.generation = 0
        self.patches = ExitStack()
        self.source_storages = {(tensor.device, tensor.untyped_storage()._cdata) for tensor in host_inputs}

    def _metadata(self, tensor):
        metadata = _tensor_metadata(tensor)
        key = tensor.device, metadata["storage"]
        if tensor.is_cuda and key not in self.allocations:
            # A reusable pool may predate this observed invocation. Identify
            # its live storage on first use; native epochs identify tile reuse.
            self.generation += 1
            self.allocations[key] = self.generation
        metadata["allocation"] = self.allocations.get(key)
        return metadata

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        inputs = [self._metadata(value) for value in tree_flatten(args)[0]
                  if isinstance(value, torch.Tensor)]
        result = function(*args, **(kwargs or {}))
        outputs = []
        for tensor in tree_flatten(result)[0]:
            if not isinstance(tensor, torch.Tensor):
                continue
            metadata = _tensor_metadata(tensor)
            key = tensor.device, metadata["storage"]
            if not any((item["device"], item["storage"]) == key for item in inputs):
                # Allocator reuse can recycle both data_ptr and StorageImpl ids.
                # Each actual allocation gets a new generation, independent of them.
                self.generation += 1
                self.allocations[key] = self.generation
            metadata["allocation"] = self.allocations.get(key)
            outputs.append(metadata)
            if self.chunk is not None and tensor.is_cuda:
                rank = len(self.chunk)
                limit = max(math.prod(self.chunk), self.batch * math.prod(self.case.detector) * rank,
                            self.batch * rank * (4 if self.case.beam == "cone" else 3))
                assert tensor.numel() <= limit, ("unbounded CUDA tensor", str(function), metadata)
                assert metadata["storage_bytes"] <= limit * tensor.element_size(), \
                    ("unbounded CUDA storage behind a tile view", str(function), metadata)
                if math.prod(self.case.shape) > math.prod(self.chunk):
                    assert metadata["shape"] not in (self.case.shape, self.case.shape[::-1]), \
                        "complete CPU volume was staged on CUDA"
                if self.case.views > self.batch:
                    assert metadata["shape"] not in (self.case.sino_shape, (*self.case.sino_shape, rank)), \
                        "complete CPU sinogram/surface was staged on CUDA"
            if self.chunk is not None and metadata["pinned"]:
                if metadata["shape"] in (self.case.shape, self.case.sino_shape):
                    assert metadata["storage_bytes"] <= pipeline._HOST_SLOT_BYTES, \
                        "complete large CPU input was pinned beyond the bounded slot ceiling"
        name = str(function)
        if self.chunk is not None and name == "aten.clone.default":
            for item in inputs:
                if (item["device"], item["storage"]) in self.source_storages and item["shape"] in (
                        self.case.shape, self.case.sino_shape):
                    assert item["bytes"] <= 4 * max(math.prod(self.chunk),
                                                   self.batch * math.prod(self.case.detector)), \
                        "complete CPU input was cloned"
        self.operations.append({"phase": self.phase, "name": name, "inputs": inputs, "outputs": outputs,
                                "fill": args[1] if name == "aten.fill_.Scalar" else None})
        return result

    def __enter__(self):
        original_bridge = TorchCUDABridge.tensor_to_cuda_array

        def bridge(tensor):
            result = original_bridge(tensor)
            self.bridged[tensor.device, tensor.data_ptr()] = self._metadata(tensor)
            return result

        self.patches.enter_context(mock.patch.object(TorchCUDABridge, "tensor_to_cuda_array", bridge))
        originals = {}
        for beam in BEAMS:
            prefix = "_cone_3d" if beam == "cone" else f"_{beam}_2d"
            for kind in ("forward", "backward", "geometry_vjp"):
                original = getattr(kernels, f"{prefix}_{kind}_kernel")
                originals[id(original)] = self._proxy(original, beam, kind)
        for name, module in list(sys.modules.items()):
            if name == "diffct" or name.startswith("diffct."):
                for attribute, value in list(vars(module).items()):
                    if id(value) in originals:
                        self.patches.enter_context(mock.patch.object(module, attribute, originals[id(value)]))
        return super().__enter__()

    def _proxy(self, kernel, beam, kind):
        tracer = self

        class Proxy:
            def __getitem__(self, config):
                configured = kernel[config]

                def launch(*args, **kwargs):
                    backward = kind == "backward"
                    volume = args[4 if beam == "cone" else 3] if backward else args[0]
                    rays = args[0] if backward else args[4 if beam == "cone" else 3]
                    device = torch.device("cuda", torch.cuda.current_device())

                    def metadata(array):
                        pointer = array.__cuda_array_interface__["data"][0]
                        assert (device, pointer) in tracer.bridged, "kernel buffer bypassed the real Torch bridge"
                        return dict(tracer.bridged[device, pointer])

                    if beam == "cone":
                        native_shape = tuple(int(value) for value in (args[5:8] if backward else args[1:4]))
                        views = int(args[1] if backward else args[5])
                    else:
                        native_shape = tuple(int(value) for value in (args[5], args[4])) if backward else (int(args[2]), int(args[1]))
                        views = int(args[1] if backward else args[4])
                    record = {"phase": tracer.phase, "beam": beam, "kind": kind,
                              "device": device, "native_shape": native_shape, "views": views,
                              "volume": metadata(volume), "rays": metadata(rays),
                              "operation_index": len(tracer.operations)}
                    result = configured(*args, **kwargs)
                    tracer.launches.append(record)
                    return result
                return launch
        return Proxy()

    def __exit__(self, *exception):
        try:
            return super().__exit__(*exception)
        finally:
            self.patches.__exit__(*exception)


def _regular_case(beam, *, views=5, learnable=False):
    case = _case(beam, views=views, learnable=learnable)
    case.shape = (5, 7, 9) if beam == "cone" else (7, 9)
    return case


def _chunk(case):
    # Every anisotropic tail remains nondegenerate, so real WHD copies are visible.
    return (3, 4, 5) if case.beam == "cone" else (4, 5)


def _tile_shapes(case, chunk):
    return [tuple(min(size - start, part) for size, start, part in zip(case.shape, starts, chunk))
            for starts in product(*(range(0, size, part) for size, part in zip(case.shape, chunk)))]


def _records(trace, phase, kind):
    return [record for record in trace.launches if record["phase"] == phase and record["kind"] == kind]


def _span(metadata):
    if not math.prod(metadata["shape"]):
        return metadata["pointer"], metadata["pointer"]
    element = metadata["bytes"] // math.prod(metadata["shape"])
    extent = element * (1 + sum((size - 1) * stride for size, stride in zip(metadata["shape"], metadata["strides"])))
    return metadata["pointer"], metadata["pointer"] + extent


def _overlaps(first, second):
    return first["device"] == second["device"] and first["allocation"] == second["allocation"] and \
        max(_span(first)[0], _span(second)[0]) < min(_span(first)[1], _span(second)[1])


def _native_epoch(trace, record, kind):
    native, epoch = record["volume"], -1
    for index, operation in enumerate(trace.operations[:record["operation_index"]]):
        if operation["phase"] != record["phase"]:
            continue
        outputs = [item for item in operation["outputs"] if _overlaps(item, native)]
        if kind == "backward":
            if _is_clear(operation) and any(_span(item)[0] <= _span(native)[0] and
                                             _span(item)[1] >= _span(native)[1] for item in outputs):
                epoch = index
        elif operation["name"] in ("aten._to_copy.default", "aten.copy_.default", "aten.clone.default") and outputs:
            epoch = index
    return epoch


def _groups(trace, phase, kind, case, chunk, devices=(0,)):
    groups = defaultdict(list)
    for record in _records(trace, phase, kind):
        assert record["beam"] == case.beam
        assert record["volume"]["contiguous"], "kernel volume lost native contiguous HW/WHD layout"
        assert record["volume"]["allocation"] is not None
        epoch = _native_epoch(trace, record, kind)
        assert epoch >= 0, "native tile has no observed preparation/clear epoch"
        groups[record["device"].index, record["native_shape"], epoch].append(record)
    active = devices[:min(len(devices), case.views)]
    expected = Counter((device, shape[::-1] if case.beam == "cone" else shape)
                       for device in active for shape in _tile_shapes(case, chunk))
    assert Counter((device, shape) for device, shape, _ in groups) == expected, \
        "real native launches missed or reprepared a spatial tile/device"
    return groups


def _is_clear(operation):
    return operation["name"] in ("aten.zeros.default", "aten.zeros_like.default", "aten.zero_.default") or (
        operation["name"] == "aten.fill_.Scalar" and operation.get("fill") == 0
    )


def _transfers(trace, phase, shapes, source, destination):
    transfers = []
    for operation in trace.operations:
        if operation["phase"] != phase or operation["name"] not in ("aten._to_copy.default", "aten.copy_.default"):
            continue
        if not operation["inputs"] or not operation["outputs"]:
            continue
        origin = operation["inputs"][-1]
        output = operation["outputs"][0]
        if origin["device"].type == source and output["device"].type == destination and output["shape"] in shapes:
            transfers.append(output)
    return transfers


def _native_transfers(trace, phase, native_arrays, source, destination):
    """Follow real native data ranges through layouts/copies/local sums.

    Allocation identity alone is insufficient for typed composite buffers:
    frame/ray fields can share storage while occupying disjoint byte ranges.
    """
    ranges = list(native_arrays)

    def covered(item):
        return any(_overlaps(item, known) and _span(known)[0] <= _span(item)[0] and
                   _span(item)[1] <= _span(known)[1] for known in ranges)

    copies = []
    for operation in trace.operations:
        if operation["phase"] != phase:
            continue
        inputs, outputs = operation["inputs"], operation["outputs"]
        if operation["name"] in ("aten.clone.default", "aten._to_copy.default", "aten.copy_.default",
                                  "aten.add_.Tensor", "aten.add.Tensor"):
            data = [item for item in inputs if item["device"].type == "cuda" and covered(item)]
            for output in outputs:
                if output["device"].type == "cuda" and any(output["bytes"] == item["bytes"] for item in data):
                    ranges.append(output)
            if operation["name"] in ("aten._to_copy.default", "aten.copy_.default") and inputs and outputs:
                origin, output = inputs[-1], outputs[0]
                if origin["device"].type == source and output["device"].type == destination and (
                        covered(origin) if source == "cuda" else covered(output)):
                    copies.append(output)
    return copies


def _layout_copies(trace, phase, allocations=None, source_allocations=None):
    return [operation for operation in trace.operations
            if operation["phase"] == phase and operation["name"] in (
                "aten.clone.default", "aten._to_copy.default", "aten.copy_.default")
            and operation["inputs"] and operation["outputs"]
            and len(operation["inputs"][-1]["shape"]) == 3
            and not operation["inputs"][-1]["contiguous"]
            and operation["outputs"][0]["contiguous"]
            and (allocations is None or operation["outputs"][0]["allocation"] in allocations)
            and (source_allocations is None or operation["inputs"][-1]["allocation"] in source_allocations)]


def _assert_accumulators(trace, phase, case, chunk, devices=(0,)):
    groups = _groups(trace, phase, "backward", case, chunk, devices)
    allocations = {record["volume"]["allocation"] for group in groups.values() for record in group}
    native_arrays = [record["volume"] for group in groups.values() for record in group]
    clears = [operation for operation in trace.operations
              if operation["phase"] == phase and _is_clear(operation)
              and any(_overlaps(output, native) for output in operation["outputs"]
                      for native in native_arrays)]
    downloads = _native_transfers(trace, phase, native_arrays, "cuda", "cpu")
    tile_count = len(groups)
    spatial_tiles = len(_tile_shapes(case, chunk))
    expected_bytes = 4 * math.prod(case.shape)
    counts = {"tiles": tile_count, "launches": sum(map(len, groups.values())), "clears": len(clears),
              "downloads": len(downloads), "download_bytes": sum(item["bytes"] for item in downloads),
              "layouts": len(_layout_copies(trace, phase, source_allocations=allocations)) if case.beam == "cone" else 0}
    issues = []
    if any(len({record["volume"]["allocation"] for record in group}) != 1 for group in groups.values()):
        issues.append("one accumulator allocation must span all view batches of a tile/device")
    if len(clears) != tile_count:
        issues.append("accumulators must clear exactly once per tile/device")
    if counts["download_bytes"] != expected_bytes:
        issues.append(f"completed combined native tile payload must download once ({expected_bytes} bytes)")
    if case.beam == "cone" and counts["layouts"] > spatial_tiles:
        issues.append("WHD-to-DHW materialization must follow completed tile accumulation")
    assert not issues, f"{phase}: {counts}; " + "; ".join(issues)
    return counts


def _assert_prepared(trace, phase, kind, case, chunk, devices=(0,)):
    groups = _groups(trace, phase, kind, case, chunk, devices)
    allocations = {record["volume"]["allocation"] for group in groups.values() for record in group}
    layouts = _layout_copies(trace, phase, allocations) if case.beam == "cone" else []
    issues = []
    if any(len({record["volume"]["allocation"] for record in group}) != 1 for group in groups.values()):
        issues.append("native prepared volume must survive every view batch")
    if len(layouts) > len(groups):
        issues.append("DHW-to-WHD materialization repeated across view batches")
    if kind == "forward" and any(len({record["rays"]["allocation"] for record in group}) > 2
                                 for group in groups.values()):
        issues.append("at most two reusable ray outputs must span view batches, including the short tail")
    assert not issues, f"{phase}/{kind}: tiles={len(groups)}, layouts={len(layouts)}; " + "; ".join(issues)


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("curved", [False, True], ids=["flat", "coupled-surface"])
def test_backprojection_accumulates_native_tiles_before_layout_and_host_transfer(beam, curved):
    case = _regular_case(beam)
    chunk = _chunk(case)
    surface = _surface(case, per_view=True) if curved else None
    projector = _projector(case, surface, volume_chunk_shape=chunk, view_chunk_size=2, schedule="spatial")
    image, sino = _data(case, device="cpu")
    matrix = _matrix(case, _offsets(case, per_view=curved, flat=not curved))
    with _ExecutionTrace(case, chunk, 2, host_inputs=(sino,)) as trace:
        actual = projector.backproject(sino)
    _close(actual, (matrix.T @ _numpy(sino).ravel()).reshape(case.shape))
    assert actual.device.type == "cpu" and actual.dtype == torch.float32
    _assert_accumulators(trace, "operation", case, chunk)


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("curved", [False, True], ids=["flat", "coupled-surface"])
def test_forward_prepares_native_volume_once_and_reuses_ray_storage(beam, curved):
    case = _regular_case(beam)
    chunk = _chunk(case)
    surface = _surface(case, per_view=True) if curved else None
    projector = _projector(case, surface, volume_chunk_shape=chunk, view_chunk_size=2, schedule="spatial")
    image, sino = _data(case, device="cpu")
    matrix = _matrix(case, _offsets(case, per_view=curved, flat=not curved))
    with _ExecutionTrace(case, chunk, 2, host_inputs=(image,)) as trace:
        actual = projector.project(image)
    _close(actual, (matrix @ _numpy(image).ravel()).reshape(case.sino_shape))
    assert {record["views"] for record in trace.launches} == {1, 2}, "short final view batch was not dispatched"
    native_arrays = [record["volume"] for record in _records(trace, "operation", "forward")]
    # Reverse causal GPU layouts to include DHW upload scratch as well as WHD.
    for operation in reversed(trace.operations):
        if operation["name"] in ("aten.clone.default", "aten.copy_.default", "aten._to_copy.default") and \
                any(_overlaps(output, known) for output in operation["outputs"] for known in native_arrays):
            native_arrays.extend(item for item in operation["inputs"] if item["device"].type == "cuda")
    uploads = _native_transfers(trace, "operation", native_arrays, "cpu", "cuda")
    assert sum(item["bytes"] for item in uploads) == image.numel() * 4
    _assert_prepared(trace, "operation", "forward", case, chunk)


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("curved", [False, True], ids=["flat", "coupled-surface"])
def test_reused_ray_buffer_overwrites_hits_with_misses_and_inside_source_short_tail(beam, curved):
    case = _boundary_case(beam)
    case.detector = (3, 3) if beam == "cone" else (3,)
    case.pitch = (.3, .2) if beam == "cone" else (.3,)
    surface = _surface(case, .21, per_view=True) if curved else None
    matrix = _matrix(case, _offsets(case, .21, per_view=curved, flat=not curved))
    pixels = math.prod(case.detector)
    assert matrix[:pixels].any() and not matrix[-pixels:].any(), "fixture lacks hit-to-miss reuse"
    if beam != "parallel":
        assert np.all(np.abs(_numpy(case.trajectory[0])[3]) < np.asarray(case.shape[::-1]) / 2)
        assert matrix[3 * pixels:4 * pixels].any(), "source-inside rays must exercise a nonzero partial chord"
    image = torch.linspace(.13, 1.7, math.prod(case.shape)).reshape(case.shape)
    projector = _projector(case, surface, volume_chunk_shape=case.shape, view_chunk_size=2, schedule="spatial")
    with _ExecutionTrace(case, case.shape, 2, host_inputs=(image,)) as trace:
        actual = projector.project(image)
    _close(actual, (matrix @ _numpy(image).ravel()).reshape(case.sino_shape))
    assert torch.count_nonzero(actual[-1]) == 0, "missed short-tail rays retained earlier hits"
    _assert_prepared(trace, "operation", "forward", case, case.shape)
    # Active kernel exits already overwrite every valid ray. No unconditional
    # ray zero_ count is imposed; correct reset/overwrite is measured numerically.


@pytest.mark.cuda
@cuda_required
def test_720_views_32_batch_backprojection_downloads_one_actual_tile():
    case = _case("cone", views=720)
    case.shape, case.detector = (3, 4, 5), (3, 2)
    projector = _projector(case, volume_chunk_shape=case.shape, view_chunk_size=32, schedule="spatial")
    image, sino = _data(case, device="cpu")
    matrix = _matrix(case, _offsets(case, flat=True))
    with _ExecutionTrace(case, case.shape, 32, host_inputs=(sino,)) as trace:
        actual = projector.backproject(sino)
    _close(actual, (matrix.T @ _numpy(sino).ravel()).reshape(case.shape), rtol=8e-4, atol=2e-4)
    records = _records(trace, "operation", "backward")
    assert Counter(record["views"] for record in records) == Counter({32: 22, 16: 1})
    counts = _assert_accumulators(trace, "operation", case, case.shape)
    assert counts["download_bytes"] == 240
    # Arithmetic illustration only, never a measured 1024^3 transfer/benchmark:
    # 1024^3 float32 voxels = 4 GiB; ceil(720/32) = 23 batches.
    gib = 1024 ** 3
    assert 4 * 1024 ** 3 == 4 * gib
    assert 4 * 1024 ** 3 * math.ceil(720 / 32) == 92 * gib


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("curved", [False, True], ids=["flat", "coupled-surface"])
@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_data_backward_geometry_vjp_and_hessian_keep_bounded_reusable_tiles(beam, curved, operation):
    case = _regular_case(beam, learnable=True)
    chunk = _chunk(case)
    strength = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    surface = _surface(case, strength, per_view=True) if curved else None
    projector = _projector(case, surface, volume_chunk_shape=chunk, view_chunk_size=2, schedule="spatial")
    image, sino = _data(case, device="cpu")
    data, weight = (image, sino) if operation == "project" else (sino, image)
    data = data.double().requires_grad_()
    parameters = (*case.trajectory, strength) if curved else case.trajectory
    saved_geometry = tuple(_numpy(component) for component in case.trajectory)
    matrix = _matrix(case, _offsets(case, per_view=curved, flat=not curved), saved_geometry)
    operator = matrix if operation == "project" else matrix.T
    direction = torch.linspace(-.4, .7, data.numel(), dtype=data.dtype).reshape_as(data)
    saved_cuda = []

    def pack(tensor):
        if tensor.is_cuda:
            saved_cuda.append(tensor.numel() * tensor.element_size())
        return tensor

    with _ExecutionTrace(case, chunk, 2, host_inputs=(data, weight)) as trace, \
            torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        trace.phase = "forward"
        output = getattr(projector, operation)(data)
        # Backward must use the forward geometry snapshot despite later edits.
        with torch.no_grad():
            case.trajectory[1].add_(.071)
            strength.add_(.19)
        trace.phase = "weighted-backward"
        gradients = torch.autograd.grad((output.double() * weight.double()).sum(),
                                        (data, *parameters), retain_graph=True)
        trace.phase = "quadratic-backward"
        gradient, = torch.autograd.grad(output.double().square().sum() / 2, data,
                                        create_graph=True)
        trace.phase = "data-hessian"
        hessian, = torch.autograd.grad((gradient * direction).sum(), data)
    _close(output, (operator @ _numpy(data).ravel()).reshape(weight.shape))
    _close(gradients[0], (operator.T @ _numpy(weight).ravel()).reshape(data.shape))
    normal = operator.T @ operator
    _close(gradient, (normal @ _numpy(data).ravel()).reshape(data.shape))
    _close(hessian, (normal @ _numpy(direction).ravel()).reshape(data.shape))
    assert gradients[0].dtype == gradient.dtype == hessian.dtype == data.dtype
    assert gradients[0].device == gradient.device == hessian.device == data.device
    assert all(value is not None and torch.isfinite(value).all() for value in gradients[1:])
    generator = np.random.default_rng(922)
    directions = [generator.normal(size=value.shape) for value in saved_geometry]
    if beam == "parallel":
        directions[0] -= (directions[0] * saved_geometry[0]).sum(-1, keepdims=True) * saved_geometry[0]
    directions = [value / np.linalg.norm(value) for value in directions]

    def objective(step):
        geometry = [value + step * delta for value, delta in zip(saved_geometry, directions)]
        offsets = _offsets(case, 1. + step * .13 if curved else 1., per_view=curved, flat=not curved)
        changed = _matrix(case, offsets, geometry)
        return float(_numpy(sino).ravel() @ changed @ _numpy(image).ravel())

    expected = _finite_difference(objective, 0.)
    actual = sum(float((_numpy(value) * delta).sum()) for value, delta in zip(gradients[1:], directions))
    if curved:
        actual += gradients[-1].item() * .13
    assert abs(expected) > 1e-3, "uninformative geometry directional derivative"
    assert actual == pytest.approx(expected, rel=5e-3, abs=4e-4)
    assert sum(saved_cuda) <= _working_set_bytes(beam, chunk, 2, case.detector, curved), \
        "autograd retained more CUDA tile storage than the live plan"
    _assert_prepared(trace, "weighted-backward", "geometry_vjp", case, chunk)
    if operation == "project":
        _assert_accumulators(trace, "weighted-backward", case, chunk)
    else:
        _assert_prepared(trace, "weighted-backward", "forward", case, chunk)


@pytest.mark.cuda
@cuda_required
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("curved", [False, True], ids=["flat", "coupled-surface"])
def test_full_single_gpu_returns_native_result_without_singleton_cat_or_extra_add_copy(beam, curved):
    case = _regular_case(beam)
    surface = _surface(case, per_view=True) if curved else None
    projector = _projector(case, surface, devices=[0])
    image, sino = _data(case)
    matrix = _matrix(case, _offsets(case, per_view=curved, flat=not curved))
    with _ExecutionTrace(case) as trace:
        trace.phase = "project"
        forward = projector.project(image)
        trace.phase = "backproject"
        backward = projector.backproject(sino)
    _close(forward, (matrix @ _numpy(image).ravel()).reshape(case.sino_shape))
    _close(backward, (matrix.T @ _numpy(sino).ravel()).reshape(case.shape))
    singleton_cats = [record for record in trace.operations if record["phase"] == "project"
                      and record["name"] == "aten.cat.default" and len(record["inputs"]) == 1]
    shapes = {case.shape, case.shape[::-1] if beam == "cone" else case.shape}
    clears = [value for record in trace.operations if record["phase"] == "backproject" and _is_clear(record)
              for value in record["outputs"] if value["device"].type == "cuda" and value["shape"] in shapes]
    adds = [record for record in trace.operations if record["phase"] == "backproject"
            and record["name"] == "aten.add_.Tensor" and record["outputs"][0]["shape"] == case.shape]
    assert not singleton_cats and len(clears) == 1 and not adds, \
        f"single GPU: singleton_cats={len(singleton_cats)}, volume_clears={len(clears)}, extra_volume_adds={len(adds)}"


@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA GPUs are required")
@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("views", [1, 7], ids=["empty-GPU-shard", "uneven-batches"])
def test_each_active_gpu_has_one_resident_accumulator_and_prepared_tile(beam, views):
    case = _regular_case(beam, views=views)
    chunk, devices = _chunk(case), (1, 0)
    projector = _projector(case, _surface(case, per_view=True), devices=list(devices),
                           volume_chunk_shape=chunk, view_chunk_size=2, schedule="spatial")
    image, sino = _data(case, device="cpu")
    matrix = _matrix(case, _offsets(case, per_view=True))
    with _ExecutionTrace(case, chunk, 2, host_inputs=(image, sino)) as trace:
        trace.phase = "project"
        forward = projector.project(image)
        trace.phase = "backproject"
        backward = projector.backproject(sino)
    _close(forward, (matrix @ _numpy(image).ravel()).reshape(case.sino_shape))
    _close(backward, (matrix.T @ _numpy(sino).ravel()).reshape(case.shape))
    assert {record["device"].index for record in trace.launches} == set(devices[:min(views, 2)])
    _assert_prepared(trace, "project", "forward", case, chunk, devices)
    _assert_accumulators(trace, "backproject", case, chunk, devices)


def _memory_trial(shape, views):
    case = _case("cone", views=views, learnable=True)
    case.shape, case.detector = shape, (4, 3)
    chunk, batch = (8, 9, 10), 2
    strength = torch.tensor(.21, dtype=torch.float64, requires_grad=True)
    projector = _projector(case, _surface(case, strength, per_view=True),
                           volume_chunk_shape=chunk, view_chunk_size=batch, schedule="spatial")
    image, sino = _data(case, device="cpu")
    image.requires_grad_()
    sino.requires_grad_()
    projector.project(image.detach())
    gc.collect()
    torch.cuda.synchronize()
    initial = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with _ExecutionTrace(case, chunk, batch, host_inputs=(image, sino)) as trace:
        trace.phase = "project"
        forward = projector.project(image)
        trace.phase = "project-backward"
        (forward * sino.detach()).sum().backward()
        trace.phase = "backproject"
        back = projector.backproject(sino)
        trace.phase = "backproject-backward"
        (back * image.detach()).sum().backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - initial
    assert all(torch.isfinite(value).all() for value in (forward, back, image.grad, sino.grad, strength.grad))
    assert {record["kind"] for record in trace.launches} == {"forward", "backward", "geometry_vjp"}
    return peak, _working_set_bytes("cone", chunk, batch, case.detector, True)


@pytest.mark.cuda
@cuda_required
def test_live_cuda_tile_buffers_remain_bounded_with_volume_and_view_growth_through_vjp():
    small, budget = _memory_trial((8, 9, 10), 5)
    larger_volume, _ = _memory_trial((16, 18, 20), 5)
    more_views, _ = _memory_trial((8, 9, 10), 37)
    # One complete planned working set allows bounded allocator/layout scratch,
    # while retaining full inputs or every completed tile grows past this bound.
    assert larger_volume <= small + budget, (small, larger_volume, budget)
    assert more_views <= small + budget, (small, more_views, budget)
