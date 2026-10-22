"""Portable analytical execution and real four-rank hybrid ownership contracts."""

from contextlib import ExitStack
from datetime import timedelta
import inspect
import json
import math
from pathlib import Path
import time
from unittest import mock

import numpy as np
import pytest
import torch
import torch.distributed as dist
from tests._multiprocessing import spawn

import diffct
from tests.test_detector_surfaces import _case, _grids, _matrix


BEAMS = ("parallel", "fan", "cone")
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA required")
GLOO = pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(),
                          reason="real four-process Gloo required")
EXECUTION = dict(backend="auto", precision="fast", devices=None, view_chunk_size=None,
                 volume_chunk_shape=None, distributed=False, process_group=None,
                 partition="views", partition_grid=None)


def _api(name):
    value = getattr(diffct, name, None)
    assert callable(value), f"INTERFACE_PENDING: diffct.{name} export"
    return value


def _scan(beam, *, uniform=False):
    # Unequal spacings and translated, variable-radius source lines expose
    # per-chunk quadrature, mean-distance and isocenter mistakes.
    angles = (.17 + torch.arange(8, dtype=torch.float64) * (2 * math.pi / 8) if uniform
              else torch.tensor([.17, .49, .99, 1.72, 2.62, 3.97, 5.38], dtype=torch.float64))
    radial = torch.stack((angles.cos(), angles.sin()), 1)
    u = torch.stack((-angles.sin(), angles.cos()), 1)
    center = torch.tensor([.19, -.23], dtype=torch.float64)
    shape, detector, pitch = (3, 5), (9,), (.71,)
    if beam == "cone":
        radial = torch.nn.functional.pad(radial, (0, 1))
        u = torch.nn.functional.pad(u, (0, 1))
        v = torch.tensor([0., 0., 1.], dtype=torch.float64).expand(len(angles), -1)
        center = torch.tensor([.19, -.23, .13], dtype=torch.float64)
        shape, detector, pitch = (3, 4, 5), (9, 7), (.71, .93)
    sid = 6. + torch.arange(len(angles), dtype=torch.float64) * .13
    sdd = 11. + torch.arange(len(angles), dtype=torch.float64) * .17
    if beam == "parallel":
        trajectory = (radial, center.expand_as(radial) + .27 * u, u)
    else:
        source = center - sid[:, None] * radial
        detector_center = source + sdd[:, None] * radial + .27 * u
        if beam == "cone":
            detector_center = detector_center - .19 * v
        trajectory = ((source, detector_center, u, v) if beam == "cone"
                      else (source, detector_center, u))
    return dict(beam=beam, trajectory=trajectory, shape=shape, angles=angles,
                sino_shape=(len(angles), *detector), pitch=pitch, spacing=.53,
                center=center)


def _reconstruct(case, value, *, into=None, **kwargs):
    function = _api("fdk" if case["beam"] == "cone" else "fbp")
    options = dict(detector_spacing=case["pitch"] if case["beam"] == "cone" else case["pitch"][0],
                   voxel_spacing=case["spacing"], **kwargs)
    if case["beam"] != "cone":
        options["beam"] = case["beam"]
    if into is not None:
        function = _api("fdk_into" if case["beam"] == "cone" else "fbp_into")
        return function(value, into, case["trajectory"], case["shape"], case["angles"], **options)
    return function(value, case["trajectory"], case["shape"], case["angles"], **options)


def _random(shape, seed, *, device="cpu", grad=False):
    value = torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return value.to(device).requires_grad_(grad)


def _overrides(case, *, device="cpu", grad=False):
    return dict(projection_weights=(_random(case["sino_shape"], 410) * .1 + .8).to(device).requires_grad_(grad),
                angular_weights=_random((len(case["angles"]),), 411).abs().to(device).requires_grad_(grad),
                filter_response=(_random((case["sino_shape"][1] + 1,), 412) * .1 + .4).to(device).requires_grad_(grad))


def _analytical_oracle(case, data, projection_weights, angular_weights, filter_response):
    """Physical voxel-centre gather; independent of all production backprojectors."""
    beam = case["beam"]
    weighted = data * projection_weights
    n = data.shape[1]
    left = n // 2
    pad = (0, 0, left, n - left) if beam == "cone" else (left, n - left)
    padded = torch.nn.functional.pad(weighted, pad)
    response = filter_response.reshape((1, n + 1) + (1,) * (data.ndim - 2))
    filtered = torch.fft.irfft(torch.fft.rfft(padded, dim=1) * response, n=2 * n, dim=1)
    filtered = filtered.narrow(1, left, n) * angular_weights.reshape((-1,) + (1,) * (data.ndim - 1))
    return _gather_oracle(case, filtered)


def _gather_oracle(case, filtered):
    beam, shape, pitch = case["beam"], case["shape"], case["pitch"]
    n = filtered.shape[1]
    axes = [(torch.arange(size, dtype=filtered.dtype, device=filtered.device) + .5 - size / 2)
            * case["spacing"] for size in shape]
    positions = torch.stack(torch.meshgrid(*axes, indexing="ij")[::-1], -1).reshape(-1, len(shape))
    geometry = tuple(value.to(filtered) for value in case["trajectory"])
    output = torch.zeros(len(positions), dtype=filtered.dtype, device=filtered.device)
    if beam != "parallel":
        source, detector, u = geometry[:3]
        normal = (torch.stack((-u[:, 1], u[:, 0]), 1) if beam == "fan"
                  else torch.linalg.cross(u, geometry[3], dim=1))
        normal = torch.nn.functional.normalize(normal, dim=1)
        normal = torch.where(((detector - source) * normal).sum(1)[:, None] < 0, -normal, normal)
        sid = ((case["center"].to(filtered) - source) * normal).sum(1)
        sdd = ((detector - source) * normal).sum(1)
    for view in range(len(case["angles"])):
        if beam == "parallel":
            hit = positions - geometry[1][view]
            weight = torch.ones_like(output)
        else:
            distance = ((positions - source[view]) * normal[view]).sum(1)
            hit = source[view] + (sdd[view] / distance)[:, None] * (positions - source[view]) - detector[view]
            weight = (sid[view] / distance).square()
        fu = (hit * geometry[2][view]).sum(1) / pitch[0] + (n - 1) / 2
        valid = (fu >= 0) & (fu <= n - 1)
        iu = fu.floor().long().clamp(0, n - 2)
        tu = (fu - iu).clamp(0, 1)
        if beam == "cone":
            m = filtered.shape[2]
            fv = (hit * geometry[3][view]).sum(1) / pitch[1] + (m - 1) / 2
            valid &= (fv >= 0) & (fv <= m - 1)
            iv = fv.floor().long().clamp(0, m - 2)
            tv = (fv - iv).clamp(0, 1)
            sample = ((1 - tu) * ((1 - tv) * filtered[view, iu, iv] + tv * filtered[view, iu, iv + 1])
                      + tu * ((1 - tv) * filtered[view, iu + 1, iv] + tv * filtered[view, iu + 1, iv + 1]))
        else:
            sample = (1 - tu) * filtered[view, iu] + tu * filtered[view, iu + 1]
        scale = 1 / (2 * math.pi) if beam == "parallel" else sdd[view] / (2 * math.pi * sid[view])
        output = output + torch.where(valid, sample * weight * scale, 0)
    return output.reshape(shape)


def _default_oracle(case, data):
    """Default cosine, finite Ram-Lak convolution and uniform full-turn quadrature."""
    beam, pitch = case["beam"], case["pitch"]
    geometry = tuple(value.to(data) for value in case["trajectory"])
    weighted = data
    if beam != "parallel":
        source, detector, u = geometry[:3]
        offset_u = ((torch.arange(data.shape[1], device=data.device, dtype=data.dtype)
                     - (data.shape[1] - 1) / 2) * pitch[0])
        if beam == "cone":
            v = geometry[3]
            offset_v = ((torch.arange(data.shape[2], device=data.device, dtype=data.dtype)
                         - (data.shape[2] - 1) / 2) * pitch[1])
            normal = torch.linalg.cross(u, v, dim=1)
            rays = (detector[:, None, None, :] - source[:, None, None, :]
                    + offset_u[None, :, None, None] * u[:, None, None, :]
                    + offset_v[None, None, :, None] * v[:, None, None, :])
        else:
            normal = torch.stack((-u[:, 1], u[:, 0]), 1)
            rays = (detector[:, None, :] - source[:, None, :]
                    + offset_u[None, :, None] * u[:, None, :])
        normal = normal / torch.linalg.vector_norm(normal, dim=1, keepdim=True)
        distance = ((detector - source) * normal).sum(1).abs()
        cosine = distance.reshape((-1,) + (1,) * (data.ndim - 1)) / torch.linalg.vector_norm(rays, dim=-1)
        weighted = data * cosine
    bins = []
    for target in range(data.shape[1]):
        value = weighted[:, target] * (math.pi / (2 * pitch[0]))
        for source in range(data.shape[1]):
            lag = target - source
            if lag % 2:
                value = value + weighted[:, source] * (-2 / (math.pi * lag * lag * pitch[0]))
        bins.append(value)
    # Eight equally spaced, endpoint-excluded angles cover 2*pi. Default
    # redundancy is one half, hence every acquired view carries pi/views.
    filtered = torch.stack(bins, 1) * (math.pi / len(case["angles"]))
    return _gather_oracle(case, filtered)


@pytest.mark.parametrize("name", ["fbp", "fdk"])
def test_analytical_execution_keyword_defaults_preserve_existing_callers(name):
    signature = inspect.signature(_api(name))
    for parameter, default in EXECUTION.items():
        assert parameter in signature.parameters, f"INTERFACE_PENDING: {name}(..., {parameter}=...)"
        assert signature.parameters[parameter].default == default


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda, CUDA])])
def test_strict_analytical_output_and_parameter_gradients_match_physical_gather(beam, device):
    case = _scan(beam)
    data = _random(case["sino_shape"], 413, device=device, grad=True)
    overrides = _overrides(case, device=device, grad=True)
    expected = _analytical_oracle(case, data, **overrides)
    actual = _reconstruct(case, data, backend="torch", precision="float64", isocenter=case["center"],
                          view_chunk_size=2, volume_chunk_shape=(1,) * len(case["shape"]), **overrides)
    assert actual.dtype == torch.float64 and actual.device == data.device
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    leaves = (data, *overrides.values())
    cotangent = _random(case["shape"], 414, device=device)
    expected_gradients = torch.autograd.grad(expected, leaves, cotangent)
    actual_gradients = torch.autograd.grad(actual, leaves, cotangent)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(actual_gradient).all() and actual_gradient.abs().max() > 0
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=3e-11, atol=3e-12)


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda, CUDA])])
def test_default_analytical_math_matches_independent_convolution_cosine_and_global_quadrature(beam, precision, device):
    case = _scan(beam, uniform=True)
    dtype = torch.float64 if precision == "float64" else torch.float32
    data = _random(case["sino_shape"], 434, device=device).to(dtype).requires_grad_()
    expected = _default_oracle(case, data.double())
    actual = _reconstruct(case, data, backend="torch", precision=precision,
                          view_chunk_size=2, volume_chunk_shape=(2,) * len(case["shape"]))
    assert actual.dtype == dtype and actual.device == data.device
    rtol, atol = ((2e-11, 2e-12) if precision == "float64" else (4e-5, 4e-6))
    torch.testing.assert_close(actual.double(), expected, rtol=rtol, atol=atol)
    cotangent = _random(case["shape"], 435, device=device)
    expected_gradient = torch.autograd.grad(expected, data, cotangent)[0]
    actual_gradient = torch.autograd.grad(actual, data, cotangent.to(dtype))[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=rtol, atol=atol)


@pytest.mark.parametrize("beam", BEAMS)
def test_analytical_chunks_use_full_scan_quadrature_isocenter_and_global_voxel_positions(beam):
    case = _scan(beam)
    data = _random(case["sino_shape"], 415)
    full = _reconstruct(case, data, backend="torch", precision="float64", isocenter=case["center"])
    for chunk in (1, 2, 4):
        actual = _reconstruct(case, data, backend="torch", precision="float64", view_chunk_size=chunk,
                              volume_chunk_shape=(2,) * len(case["shape"]))
        torch.testing.assert_close(actual, full, rtol=2e-11, atol=2e-12)


class _BoundedStore:
    """Observe actual TensorStore/NpyStore I/O without substituting their data."""
    def __init__(self, store, *, view_limit=None, spatial_limit=None):
        self.store, self.view_limit, self.spatial_limit = store, view_limit, spatial_limit
        self.reads, self.writes = [], []

    def __getattr__(self, name):
        return getattr(self.store, name)

    def _check(self, index):
        index = (index,) if isinstance(index, slice) else tuple(index)
        index += (slice(None),) * (len(self.shape) - len(index))
        lengths = tuple(len(range(*item.indices(size))) for item, size in zip(index, self.shape))
        if self.view_limit is not None:
            assert lengths[0] <= self.view_limit, "analytical source read exceeded view_chunk_size"
        if self.spatial_limit is not None:
            assert all(size <= limit for size, limit in zip(lengths, self.spatial_limit)), \
                "analytical output write exceeded volume_chunk_shape"
        return index

    def read(self, index):
        index = self._check(index)
        self.reads.append(index)
        return self.store.read(index)

    def write(self, index, value, *, accumulate=False):
        index = self._check(index)
        self.writes.append(index)
        return self.store.write(index, value, accumulate=accumulate)

    def flush(self):
        self.store.flush()


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("kind", ["tensor", "npy"])
def test_analytical_into_performs_bounded_real_store_io_and_returns_original_output(beam, kind, tmp_path):
    case = _scan(beam)
    data = _random(case["sino_shape"], 416)
    overrides = _overrides(case)
    expected = _analytical_oracle(case, data, **overrides)
    chunk = (2,) * len(case["shape"])
    if kind == "npy":
        np.save(tmp_path / "input.npy", data.numpy())
        raw_source = diffct.NpyStore(tmp_path / "input.npy", mode="r")
        raw_output = diffct.NpyStore.create(tmp_path / "output.npy", case["shape"], dtype=torch.float64)
    else:
        raw_source = diffct.TensorStore(data)
        raw_output = diffct.TensorStore(torch.full(case["shape"], torch.nan, dtype=torch.float64))
    source = _BoundedStore(raw_source, view_limit=2)
    output = _BoundedStore(raw_output, spatial_limit=chunk)
    result = _reconstruct(case, source, into=output, backend="torch", precision="float64",
                          view_chunk_size=2, volume_chunk_shape=chunk,
                          isocenter=case["center"], **overrides)
    assert result is output and source.reads and output.writes
    actual = raw_output.read(tuple(slice(None) for _ in case["shape"]))
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    if kind == "npy":
        np.testing.assert_array_equal(np.load(tmp_path / "input.npy"), data.numpy())
        np.testing.assert_allclose(np.load(tmp_path / "output.npy"), expected.numpy(), rtol=2e-11, atol=2e-12)
    else:
        torch.testing.assert_close(raw_source.tensor, data, rtol=0, atol=0)


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("problem", ["shape", "readonly", "alias", "autograd", "angles", "chunk"])
def test_analytical_into_rejects_invalid_request_before_any_source_read_or_output_write(beam, problem, tmp_path):
    case = _scan(beam)
    _api("fdk_into" if beam == "cone" else "fbp_into")
    data = _random(case["sino_shape"], 417)
    output_tensor = torch.full(case["shape"], 73., dtype=torch.float64)
    source = _BoundedStore(diffct.TensorStore(data), view_limit=2)
    output = _BoundedStore(diffct.TensorStore(output_tensor))
    options = dict(backend="torch", precision="float64", view_chunk_size=2)
    if problem == "shape":
        output = _BoundedStore(diffct.TensorStore(torch.full((1,) * len(case["shape"]), 73., dtype=torch.float64)))
    elif problem == "readonly":
        np.save(tmp_path / "readonly.npy", output_tensor.numpy())
        output = _BoundedStore(diffct.NpyStore(tmp_path / "readonly.npy", mode="r"))
    elif problem == "alias":
        case = {**case, "shape": case["sino_shape"]}
        source = data
        output = data
    elif problem == "autograd":
        source = data.requires_grad_()
    elif problem == "angles":
        case = {**case, "angles": case["angles"][:-1]}
    else:
        options["volume_chunk_shape"] = (0,) * len(case["shape"])
    with pytest.raises((ValueError, TypeError, PermissionError)):
        _reconstruct(case, source, into=output, **options)
    assert getattr(source, "reads", []) == [] and getattr(output, "writes", []) == []
    torch.testing.assert_close(output_tensor, torch.full_like(output_tensor, 73.), rtol=0, atol=0)
    torch.testing.assert_close(data.detach(), _random(case["sino_shape"], 417), rtol=0, atol=0)


@pytest.mark.cuda
@CUDA
@pytest.mark.parametrize("beam", BEAMS)
def test_native_cuda_analytical_full_and_streamed_paths_match_existing_pipeline(beam):
    from tests.test_reconstruction_api import _acquisition, _manual, _random as legacy_random
    case = _acquisition(beam)
    data = legacy_random(case["sino_shape"], 418)
    expected = _manual(beam, data, case, window="hamming")
    options = dict(detector_spacing=case["pitch"] if beam == "cone" else case["pitch"][0],
                   voxel_spacing=case["voxel_spacing"], isocenter=case["isocenter"], window="hamming")
    function = _api("fdk" if beam == "cone" else "fbp")
    if beam != "cone":
        options["beam"] = beam
    for execution in ({}, dict(backend="numba", devices=[0], view_chunk_size=2,
                              volume_chunk_shape=(1,) * len(case["shape"]))):
        actual = function(data, case["trajectory"], case["shape"], case["angles"], **options, **execution)
        torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-6)


@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two physical CUDA devices required")
@pytest.mark.parametrize("beam", BEAMS)
def test_analytical_two_physical_devices_preserve_geometry_and_global_weights(beam):
    case = _scan(beam)
    data = _random(case["sino_shape"], 419, device="cuda")
    expected = _reconstruct(case, data, backend="numba", isocenter=case["center"])
    actual = _reconstruct(case, data, backend="numba", devices=[0, 1], view_chunk_size=2,
                          volume_chunk_shape=(2,) * len(case["shape"]))
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-6)


def _owned(size, rank, count):
    base, extra = divmod(size, count)
    start = rank * base + min(rank, extra)
    return slice(start, start + base + int(rank < extra))


def _run_four(worker, args, *, timeout=150):
    context = spawn(worker, args=args, nprocs=4, join=False)
    deadline = time.monotonic() + timeout
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise AssertionError(f"four-process Gloo contract exceeded {timeout}s")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def _hybrid(case, **kwargs):
    assert "partition_grid" in inspect.signature(diffct.Projector).parameters, \
        "INTERFACE_PENDING: Projector(..., partition_grid=None)"
    return diffct.Projector(case.trajectory, case.shape, case.detector, beam=case.beam,
                            detector_spacing=case.pitch if case.beam == "cone" else case.pitch[0],
                            voxel_spacing=case.spacing, backend="torch", precision="float64",
                            distributed=True, partition="hybrid", partition_grid=(2, 2),
                            view_chunk_size=1, volume_chunk_shape=(1,) * len(case.shape), **kwargs)


def _no_whole_volume_collectives(face_elements=0, *, case=None, phase=None, reductions=None):
    """Permit bounded boundary-face exchange; forbid gathering owned volumes."""
    stack = ExitStack()
    original_gather = dist.all_gather
    def gather(output, tensor, *args, **kwargs):
        assert tensor.numel() <= face_elements, "hybrid/TV execution gathered complete numerical data"
        return original_gather(output, tensor, *args, **kwargs)
    stack.enter_context(mock.patch.object(dist, "all_gather", gather))
    for name in ("all_gather", "all_gather_into_tensor", "gather"):
        if name != "all_gather" and hasattr(dist, name):
            original_function = getattr(dist, name)
            def bounded(*args, _original=original_function, _name=name, **kwargs):
                tensor = args[1] if _name == "all_gather_into_tensor" else args[0]
                assert tensor.numel() <= face_elements, "hybrid/TV execution gathered complete numerical data"
                return _original(*args, **kwargs)
            stack.enter_context(mock.patch.object(dist, name, bounded))
    original = dist.all_gather_object
    def metadata(output, value, *args, **kwargs):
        def tensors(item):
            if isinstance(item, torch.Tensor):
                return item.numel() > max(1, face_elements)
            if isinstance(item, dict):
                return any(tensors(child) for child in item.values())
            if isinstance(item, (tuple, list)):
                return any(tensors(child) for child in item)
            if isinstance(item, np.ndarray):
                return item.size > max(1, face_elements)
            return False
        assert not tensors(value), "metadata collective carried numerical volume/rays"
        return original(output, value, *args, **kwargs)
    stack.enter_context(mock.patch.object(dist, "all_gather_object", metadata))
    if case is not None:
        original_reduce = dist.all_reduce
        def reduce(tensor, *args, **kwargs):
            if tensor.is_floating_point():
                shape = tuple(tensor.shape)
                if reductions is not None:
                    reductions.append(dict(phase=phase[0], shape=list(shape), elements=tensor.numel()))
                assert shape != case.shape or tensor.numel() <= 1, \
                    "hybrid all_reduce communicated the complete global volume"
                if phase[0] == "data" and tensor.numel() > 1:
                    # Requested volume tiles contain one voxel. Resident ray
                    # batches contain one view and the full detector rectangle.
                    rays = (len(shape) == len(case.detector) + 1 and shape[0] <= 1
                            and shape[1:] == case.detector)
                    flat_rays = len(shape) == 1 and tensor.numel() == math.prod(case.detector)
                    assert rays or flat_rays, \
                        "hybrid volume reduction exceeded the requested one-voxel tile"
            return original_reduce(tensor, *args, **kwargs)
        stack.enter_context(mock.patch.object(dist, "all_reduce", reduce))
    return stack


def _bounded_tv_faces(face_elements):
    stack = ExitStack()
    # P2POp validates async function identity; inspect its real batch rather
    # than replacing isend/irecv and invalidating otherwise legal P2POps.
    for name in ("send", "recv", "all_reduce"):
        original = getattr(dist, name)
        def bounded(tensor, *args, _original=original, **kwargs):
            assert tensor.numel() <= max(1, face_elements), "TV communicated an owned volume instead of a boundary face"
            return _original(tensor, *args, **kwargs)
        stack.enter_context(mock.patch.object(dist, name, bounded))
    original_batch = dist.batch_isend_irecv
    def batch(operations):
        assert all(operation.tensor.numel() <= face_elements for operation in operations), \
            "TV batch communicated an owned volume instead of a boundary face"
        return original_batch(operations)
    stack.enter_context(mock.patch.object(dist, "batch_isend_irecv", batch))
    return stack


def _tv_oracle(volume, spacing, eps=1e-6, reduction="mean"):
    spacing = (spacing,) * volume.ndim if isinstance(spacing, (int, float)) else spacing
    squared = torch.zeros_like(volume)
    for axis, pitch in enumerate(spacing):
        current = [slice(None)] * volume.ndim
        following = current.copy()
        current[axis], following[axis] = slice(0, -1), slice(1, None)
        difference = (volume[tuple(following)] - volume[tuple(current)]) / pitch
        padded = torch.zeros_like(volume)
        padded[tuple(current)] = difference
        squared = squared + padded.square()
    magnitude = torch.sqrt(squared + eps)
    return magnitude.sum() if reduction == "sum" else magnitude.mean()


def _hybrid_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=45))
    try:
        records = []
        for beam, first, views in (("parallel", 5, 5), ("cone", 3, 5), ("parallel", 1, 1)):
            case = _case(beam, views=views)
            case.shape = (first, 3, 4) if beam == "cone" else (first, 4)
            case.detector = (5, 3) if beam == "cone" else (7,)
            u, v = _grids(case)
            offsets = np.stack((u, v, np.zeros_like(u)), -1)
            matrix = torch.from_numpy(_matrix(case, offsets))
            truth = _random(case.shape, 420)
            rays = _random(case.sino_shape, 421)
            volume_cotangent = _random(case.shape, 422)
            view_rank, space_rank = divmod(rank, 2)
            slab = (_owned(first, space_rank, 2), *(slice(0, size) for size in case.shape[1:]))
            shard = _owned(views, view_rank, 2)
            phase, reductions = ["data"], []
            with _no_whole_volume_collectives(math.prod(case.shape[1:]), case=case,
                                              phase=phase, reductions=reductions):
                operator = _hybrid(case)
                assert operator.global_volume_shape == case.shape
                assert operator.volume_slice == slab and operator.view_slice == shard
                assert operator.volume_shape == operator.local_volume_shape == tuple(truth[slab].shape)
                assert operator.projection_shape == tuple(rays[shard].shape)
                local_x = truth[slab].clone().requires_grad_()
                projected = operator.project(local_x)
                expected_projection = (matrix @ truth.flatten()).reshape(case.sino_shape)[shard]
                torch.testing.assert_close(projected, expected_projection, rtol=2e-10, atol=2e-11)
                ((projected * rays[shard]).sum() / 2).backward()
                expected_x_gradient = (matrix.T @ rays.flatten()).reshape(case.shape)[slab]
                torch.testing.assert_close(local_x.grad, expected_x_gradient, rtol=3e-10, atol=3e-11)
                local_y = rays[shard].clone().requires_grad_()
                backprojected = operator.backproject(local_y)
                torch.testing.assert_close(backprojected, expected_x_gradient, rtol=2e-10, atol=2e-11)
                ((backprojected * volume_cotangent[slab]).sum() / 2).backward()
                expected_y_gradient = (matrix @ volume_cotangent.flatten()).reshape(case.sino_shape)[shard]
                torch.testing.assert_close(local_y.grad, expected_y_gradient, rtol=3e-10, atol=3e-11)
                # Explicit output methods exercise the same subgroup ownership.
                out_y, out_x = torch.empty_like(projected), torch.empty_like(backprojected)
                assert operator.project_into(truth[slab], out_y) is out_y
                assert operator.backproject_into(rays[shard], out_x) is out_x
                torch.testing.assert_close(out_y, expected_projection, rtol=2e-10, atol=2e-11)
                torch.testing.assert_close(out_x, expected_x_gradient, rtol=2e-10, atol=2e-11)
                lhs = (projected.detach() * rays[shard]).sum() / 2
                rhs = (truth[slab] * backprojected.detach()).sum() / 2
                dist.all_reduce(lhs)
                dist.all_reduce(rhs)
                torch.testing.assert_close(lhs, rhs, rtol=2e-11, atol=2e-12)
                phase[0] = "tv"
                for reduction in ("mean", "sum"):
                    full_tv = truth.clone().requires_grad_()
                    spacing = tuple(.61 + .23 * axis for axis in range(truth.ndim))
                    expected_tv = _tv_oracle(full_tv, spacing, reduction=reduction)
                    expected_gradient = torch.autograd.grad(expected_tv, full_tv)[0]
                    local_tv = truth[slab].clone().requires_grad_()
                    with _bounded_tv_faces(math.prod(case.shape[1:])):
                        actual_tv = _api("total_variation")(local_tv, operator=operator,
                                                             voxel_spacing=spacing, reduction=reduction)
                        torch.testing.assert_close(actual_tv, expected_tv, rtol=2e-11, atol=2e-12)
                        actual_tv.backward()
                    torch.testing.assert_close(local_tv.grad, expected_gradient[slab], rtol=3e-10, atol=3e-11)
            assert reductions, "real hybrid numerical reductions were not observed"
            records.append(dict(beam=beam, first=first, views=views,
                                volume_shape=list(operator.volume_shape), projection_shape=list(operator.projection_shape),
                                reductions=reductions))
        (Path(directory) / f"rank-{rank}.json").write_text(json.dumps(records), encoding="utf-8")
    finally:
        dist.destroy_process_group()


@GLOO
def test_real_four_rank_hybrid_forward_adjoint_replicated_loss_gradients_and_boundary_tv(tmp_path):
    assert "partition_grid" in inspect.signature(diffct.Projector).parameters, \
        "INTERFACE_PENDING: Projector.partition_grid"
    _api("total_variation")
    _run_four(_hybrid_worker, ((tmp_path / "hybrid-rendezvous").resolve().as_uri(), str(tmp_path)))
    records = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(4)]
    assert all(len(record) == 3 for record in records)
    assert any(record[2]["volume_shape"][0] == 0 for record in records)
    assert any(record[2]["projection_shape"][0] == 0 for record in records)


def _invalid_grid_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=20))
    try:
        case = _case("parallel", views=3)
        for grid in ((2, 3), (1, 4) if rank == 3 else (2, 2)):
            with pytest.raises((ValueError, TypeError)):
                diffct.Projector(case.trajectory, case.shape, case.detector, beam="parallel",
                                 backend="torch", distributed=True, partition="hybrid", partition_grid=grid)
            # All ranks survived rejection and can use the original group.
            reached = torch.tensor(1)
            dist.all_reduce(reached)
            assert reached.item() == 4
    finally:
        dist.destroy_process_group()


@GLOO
def test_real_four_rank_invalid_product_and_disagreeing_partition_grids_reject_collectively(tmp_path):
    assert "partition_grid" in inspect.signature(diffct.Projector).parameters, \
        "INTERFACE_PENDING: Projector.partition_grid"
    _run_four(_invalid_grid_worker, ((tmp_path / "invalid-grid").resolve().as_uri(),), timeout=70)


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (2, 1, 4)])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_total_variation_uses_every_voxel_zero_outer_differences_and_physical_axis_spacing(shape, reduction):
    data = _random(shape, 423, grad=True)
    spacing = tuple(.47 + .31 * axis for axis in range(len(shape)))
    expected = _tv_oracle(data, spacing, reduction=reduction)
    actual = _api("total_variation")(data, voxel_spacing=spacing, eps=1e-6, reduction=reduction)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-13)
    actual_gradient = torch.autograd.grad(actual, data)[0]
    expected_gradient = torch.autograd.grad(expected, data)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-12, atol=2e-13)


@pytest.mark.parametrize("option", [dict(voxel_spacing=0), dict(voxel_spacing=(1.,)),
                                  dict(voxel_spacing=(1., float("nan"))),
                                  dict(eps=-1), dict(eps=float("nan")), dict(reduction="median")])
def test_total_variation_rejects_invalid_spacing_smoothing_and_reduction(option):
    function = _api("total_variation")
    with pytest.raises((ValueError, TypeError)):
        function(torch.ones(2, 3, dtype=torch.float64), **option)
