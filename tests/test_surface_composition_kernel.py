"""Check native surface composition against the original tensor expression."""

import math

import numpy as np
import pytest
import torch
from numba.core import config

if not config.ENABLE_CUDASIM and (not torch.cuda.is_available() or torch.version.hip is not None):
    pytest.skip("surface composition requires CUDA or NUMBA_ENABLE_CUDASIM=1", allow_module_level=True)

from numba import cuda

from diffct.kernels import surface_geometry


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not config.ENABLE_CUDASIM and not cuda.is_available(),
        reason="surface composition requires CUDA or NUMBA_ENABLE_CUDASIM=1",
    ),
]

# Fix the acceptance bounds before observing the new kernel's results.
RTOL = 2e-6
ATOL = 2e-7
SENTINEL = np.float32(-713.25)
ORACLE_DEVICE = "cpu" if config.ENABLE_CUDASIM else "cuda"


def _arithmetic_inputs(frames, offsets, common):
    if common:
        dtype = torch.float32
        for value in (*frames, offsets):
            dtype = torch.promote_types(dtype, value.dtype)
        return tuple(value.to(dtype) for value in frames), offsets.to(dtype)
    return tuple(value.to(torch.promote_types(torch.float32, value.dtype))
                 for value in frames), offsets


def _reference(frames, offsets, beam, common, centre, device=ORACLE_DEVICE):
    """Use tensor broadcasting, promotion, and cross products as the oracle."""
    frames, offsets = _arithmetic_inputs(frames, offsets, common)
    frames = tuple(value.to(device) for value in frames)
    offsets = offsets.to(device)
    rank = frames[0].shape[-1]
    local = offsets.unsqueeze(0) if offsets.ndim == 2 else offsets
    tangent, vertical, depth = local.unbind(-1)
    source, center, axis_u = frames[:3]
    points = center[:, None] + tangent[..., None] * axis_u[:, None]
    if beam == "cone":
        axis_v = frames[3]
        normal_dtype = torch.promote_types(axis_u.dtype, axis_v.dtype)
        normal = torch.linalg.cross(axis_u.to(normal_dtype), axis_v.to(normal_dtype), dim=-1)
        points = (points + vertical[..., None] * axis_v[:, None]
                  + depth[..., None] * normal[:, None])
    else:
        normal = torch.stack((axis_u[:, 1], -axis_u[:, 0]), dim=-1)
        points = points + depth[..., None] * normal[:, None]
    raw_source, raw_points = source, points
    shift = torch.tensor(centre[:rank], dtype=points.dtype, device=device)
    points = (points - shift).float()
    if beam != "parallel":
        source = source - torch.tensor(centre[:rank], dtype=source.dtype, device=device)
    return raw_source, raw_points, source.float().cpu().numpy(), points.cpu().numpy()


def _reference_status(source, points, offsets, beam, spacing):
    statuses = []
    for point, offset in zip(points, offsets):
        finite = all(torch.isfinite(value).all().item()
                     for value in (source, point, source.float(), point.float(), offset))
        if not finite:
            statuses.append(1)
            continue
        if beam == "parallel":
            statuses.append(0)
            continue
        status = 2 if torch.equal(source.float(), point.float()) else 0
        distances = torch.linalg.vector_norm(torch.stack((source.double(), point.double())), dim=-1) / spacing
        if distances.min().item() > 1e6:
            status = 3
        if distances.max().item() > 1e15:
            status = 4
        statuses.append(status)
    return max(statuses, default=0)


def _pack(frames, offsets, view, start, count, capacity, common, device):
    frames, offsets = _arithmetic_inputs(frames, offsets, common)
    rank = frames[0].shape[-1]
    selected = offsets if offsets.ndim == 2 else offsets[view]
    selected = selected[start:start + count]
    packed_frames = np.concatenate([value[view].double().numpy() for value in frames])
    if rank == 3:
        dtype = torch.promote_types(frames[2].dtype, frames[3].dtype)
        normal = torch.linalg.cross(frames[2][view].to(device=device, dtype=dtype),
                                    frames[3][view].to(device=device, dtype=dtype), dim=-1)
        packed_frames = np.concatenate((packed_frames, normal.double().cpu().numpy()))
    packed_offsets = np.full(rank * capacity + 1, np.nan, dtype=np.float64)
    for column, component in enumerate((0, 1, 2) if rank == 3 else (0, 2)):
        packed_offsets[column * capacity:column * capacity + count] = selected[:, component].double().numpy()
    packed_offsets[-1] = 0.
    flags = tuple(value.dtype != torch.float64 for value in frames)
    flags += (False,) * (4 - len(flags))
    return packed_frames, packed_offsets, flags, offsets.dtype != torch.float64


def _exercise(frames, offsets, beam, *, common=False, centre=(0., 0., 0.),
              spacing=1., start=0, count=None, capacity=None, pixel_start=3,
              exact=False, reference_device=ORACLE_DEVICE):
    count = offsets.shape[-2] - start if count is None else count
    capacity = count + 3 if capacity is None else capacity
    views, rank = frames[0].shape
    output_shape = (views + 1, pixel_start + count + 2, rank)
    expected_source = np.full((views + 1, rank), SENTINEL, dtype=np.float32)
    expected_points = np.full(output_shape, SENTINEL, dtype=np.float32)
    source = cuda.to_device(expected_source.copy())
    points = cuda.to_device(expected_points.copy())
    raw_source, raw_points, reference_source, reference_points = _reference(
        frames, offsets, beam, common, centre, reference_device,
    )
    stream = cuda.stream()
    actual_status = []
    expected_status = []
    for view in reversed(range(views)):
        packed_frames, packed_offsets, flags, offset32 = _pack(
            frames, offsets, view, start, count, capacity, common, reference_device,
        )
        frame_buffer = cuda.to_device(packed_frames, stream=stream)
        offset_buffer = cuda.to_device(packed_offsets, stream=stream)
        threads = 8
        surface_geometry.compose_surface[math.ceil(capacity / threads), threads, stream](
            frame_buffer, offset_buffer, source, points, view, pixel_start,
            count, capacity, rank, beam == "parallel", flags, offset32, centre, spacing,
        )
        stream.synchronize()
        actual_status.append(offset_buffer.copy_to_host()[-1])
        local = offsets if offsets.ndim == 2 else offsets[view]
        expected_status.append(_reference_status(
            raw_source[view], raw_points[view, start:start + count],
            local[start:start + count], beam, spacing,
        ))
        expected_source[view] = reference_source[view]
        expected_points[view, pixel_start:pixel_start + count] = reference_points[view, start:start + count]
    actual_source, actual_points = source.copy_to_host(), points.copy_to_host()
    if exact:
        np.testing.assert_array_equal(actual_source, expected_source)
        np.testing.assert_array_equal(actual_points, expected_points)
    else:
        np.testing.assert_allclose(actual_source, expected_source, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(actual_points, expected_points, rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(actual_status, expected_status)
    return actual_source, actual_points, actual_status


@pytest.mark.parametrize("beam", ["fan", "cone", "parallel"])
@pytest.mark.parametrize("per_view", [False, True], ids=["shared", "per-view"])
@pytest.mark.parametrize("common", [False, True], ids=["native", "sampled"])
def test_random_mixed_dtypes_use_original_tensor_promotion(beam, per_view, common):
    rng = np.random.default_rng(1947)
    rank = 3 if beam == "cone" else 2
    dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    # Each input occupies each dtype; later cases mix all four independently.
    layouts = [tuple([dtype] * (rank + 2)) for dtype in dtypes]
    layouts += [tuple(dtypes[(case + component) % 4] for component in range(rank + 2))
                for case in range(4)]
    for layout in layouts:
        values = [rng.normal(size=(3, rank)) for _ in range(rank + 1)]
        values[0][:, 0] -= 9.
        values[1][:, 0] += 8.
        frames = tuple(torch.tensor(value, dtype=dtype)
                       for value, dtype in zip(values, layout[:-1]))
        local = rng.normal(size=((3, 9, 3) if per_view else (9, 3)))
        if rank == 2:
            local[..., 1] = 0.
        offsets = torch.tensor(local, dtype=layout[-1])
        _exercise(frames, offsets, beam, common=common, centre=(.375, -.625, .25),
                  spacing=.7, start=2, count=5, capacity=11)


@pytest.mark.parametrize("beam", ["fan", "cone", "parallel"])
@pytest.mark.parametrize("common", [False, True], ids=["native", "sampled"])
def test_product_rounding_precedes_promotion_into_double_center(beam, common):
    rank = 3 if beam == "cone" else 2
    source = torch.tensor([[-1.] + [0.] * (rank - 1)], dtype=torch.float32)
    center = torch.tensor([[-16785408.] + [0.] * (rank - 1)], dtype=torch.float64)
    axis_u = torch.tensor([[4097.] + [0.] * (rank - 1)], dtype=torch.float32)
    frames = (source, center, axis_u)
    if rank == 3:
        frames += (torch.tensor([[0., 0., 1.]], dtype=torch.float32),)
    offsets = torch.tensor([[4097., 0., 0.]], dtype=torch.float32)
    _, actual, _ = _exercise(frames, offsets, beam, common=common, exact=True)
    assert actual[0, 3, 0] == (1. if common else 0.)


@pytest.mark.parametrize("common", [False, True], ids=["native", "sampled"])
def test_cross_product_rounding_precedes_double_depth_multiply(common):
    frames = (
        torch.tensor([[-1., 0., 0.]], dtype=torch.float32),
        torch.tensor([[0., 0., -16785408.]], dtype=torch.float64),
        torch.tensor([[4097., 0., 0.]], dtype=torch.float32),
        torch.tensor([[0., 4097., 0.]], dtype=torch.float32),
    )
    offsets = torch.tensor([[0., 0., 1.]], dtype=torch.float64)
    _, actual, _ = _exercise(frames, offsets, "cone", common=common, exact=True)
    assert actual[0, 3, 2] == (1. if common else 0.)


@pytest.mark.parametrize("rotation", ["known", *range(8)])
@pytest.mark.parametrize("depth", [1e7, -3e7])
def test_non_axis_aligned_cross_rounding_survives_deep_double_cancellation(rotation, depth):
    if rotation == "known":
        axis_u = torch.tensor([[-.7559716702, .3963489830, -.5209743977]], dtype=torch.float32)
        axis_v = torch.tensor([[-.1834404022, -.8922420740, -.4126181304]], dtype=torch.float32)
    else:
        orthogonal, _ = np.linalg.qr(np.random.default_rng(3917 + rotation).normal(size=(3, 3)))
        axis_u = torch.tensor(orthogonal[:, 0][None], dtype=torch.float32)
        axis_v = torch.tensor(orthogonal[:, 1][None], dtype=torch.float32)
    reference_device = ORACLE_DEVICE
    normal = torch.linalg.cross(axis_u.to(reference_device), axis_v.to(reference_device), dim=-1)
    target = torch.tensor([[.125, -.25, .375]], dtype=torch.float64)
    center = -depth * normal.double().cpu() + target
    frames = (torch.tensor([[-1., -2., -3.]], dtype=torch.float32), center, axis_u, axis_v)
    offsets = torch.tensor([[.03125, -.0625, depth]], dtype=torch.float64)
    _exercise(frames, offsets, "cone", centre=(.0625, -.125, .25), exact=True,
              reference_device=reference_device)


@pytest.mark.parametrize("beam", ["fan", "cone", "parallel"])
def test_single_precision_addition_is_not_replaced_by_all_double_math(beam):
    rank = 3 if beam == "cone" else 2
    frames = (
        torch.tensor([[-4., 0.] + ([0.] if rank == 3 else [])], dtype=torch.float32),
        torch.tensor([[2. ** 24, 2. ** 24] + ([0.] if rank == 3 else [])], dtype=torch.float32),
        torch.tensor([[1., 1.] + ([0.] if rank == 3 else [])], dtype=torch.float32),
    )
    if rank == 3:
        frames += (torch.tensor([[0., 0., 1.]], dtype=torch.float32),)
    offsets = torch.tensor([[1., 0., -(2. ** 24)]], dtype=torch.float32)
    _, actual, _ = _exercise(frames, offsets, beam, exact=True)
    assert actual[0, 3, 0] == 0.


@pytest.mark.parametrize("beam", ["fan", "cone"])
@pytest.mark.parametrize("frame_dtype,offset_dtype", [
    (torch.float32, torch.float64), (torch.float64, torch.float32),
    (torch.float64, torch.float64),
])
def test_large_components_cancel_before_native_float32_cast(beam, frame_dtype, offset_dtype):
    rank = 3 if beam == "cone" else 2
    center_x = 1e8 + (.125 if frame_dtype == torch.float64 else 0.)
    depth = -1e8 + (.125 if frame_dtype == torch.float32 else 0.)
    frames = (
        torch.tensor([[-1.] + [0.] * (rank - 1)], dtype=frame_dtype),
        torch.tensor([[center_x] + [0.] * (rank - 1)], dtype=frame_dtype),
        torch.tensor([[0., 1.] + ([0.] if rank == 3 else [])], dtype=frame_dtype),
    )
    if rank == 3:
        frames += (torch.tensor([[0., 0., 1.]], dtype=frame_dtype),)
    offsets = torch.tensor([[0., 0., depth]], dtype=offset_dtype)
    _, actual, status = _exercise(frames, offsets, beam, exact=True)
    assert actual[0, 3, 0] == .125
    assert status == [0.]


@pytest.mark.parametrize("beam", ["fan", "cone", "parallel"])
def test_translation_follows_composition_and_preserves_parallel_direction(beam):
    rank = 3 if beam == "cone" else 2
    source_x = 1. if beam == "parallel" else 1e8 + .25
    frames = (
        torch.tensor([[source_x] + [0.] * (rank - 1)], dtype=torch.float64),
        torch.tensor([[1e8] + [0.] * (rank - 1)], dtype=torch.float64),
        torch.tensor([[0., 1.] + ([0.] if rank == 3 else [])], dtype=torch.float64),
    )
    if rank == 3:
        frames += (torch.tensor([[0., 0., 1.]], dtype=torch.float64),)
    offsets = torch.tensor([[0., 0., .125]], dtype=torch.float64)
    source, actual, _ = _exercise(frames, offsets, beam, centre=(1e8, 0., 0.), exact=True)
    assert source[0, 0] == (1. if beam == "parallel" else .25)
    assert actual[0, 3, 0] == .125


@pytest.mark.parametrize("beam", ["fan", "cone", "parallel"])
@pytest.mark.parametrize("source_x,center_x,depth,spacing,expected", [
    (-10., 10., 0., 1., 0),
    (1e4, 1e4 + 1e-4, 0., 1., 2),
    (-2e6, 2e6, 0., 1., 3),
    (0., 2e15, 0., 1., 4),
    (-1e6, 1e6, 0., 1., 0),
    (0., 1e15, 0., 1., 0),
    (-1.5e6, 1.5e6, 0., 2., 0),
    (0., 1.5e15, 0., 2., 0),
    (2e15, 2e15, 0., 1., 4),
    (0., 0., 4e38, 1., 1),
    (float("inf"), 10., 0., 1., 1),
    (-10., 10., float("nan"), 1., 1),
], ids=["valid", "fp32-coincident", "near", "far", "near-boundary", "far-boundary",
        "spacing-near", "spacing-far", "far-overrides-coincident", "fp32-overflow",
        "nonfinite-source", "nonfinite-offset"])
def test_validation_status_matches_original_admission(beam, source_x, center_x, depth, spacing, expected):
    rank = 3 if beam == "cone" else 2
    frames = (
        torch.tensor([[source_x] + [0.] * (rank - 1)], dtype=torch.float64),
        torch.tensor([[center_x] + [0.] * (rank - 1)], dtype=torch.float64),
        torch.tensor([[0., 1.] + ([0.] if rank == 3 else [])], dtype=torch.float64),
    )
    if rank == 3:
        frames += (torch.tensor([[0., 0., 1.]], dtype=torch.float64),)
    offsets = torch.tensor([[0., 0., depth]], dtype=torch.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        _, _, status = _exercise(frames, offsets, beam, spacing=spacing)
    assert status == [0 if beam == "parallel" and expected != 1 else expected]


def test_validation_status_accumulates_across_launches():
    offsets = cuda.to_device(np.array([0., 0., 0.], dtype=np.float64))
    source = cuda.to_device(np.zeros((1, 2), dtype=np.float32))
    points = cuda.to_device(np.zeros((1, 1, 2), dtype=np.float32))
    for source_x, center_x, expected in [(1., 1., 2), (float("inf"), 1., 2),
                                         (-2e6, 2e6, 3), (0., 2e15, 4), (-1., 1., 4)]:
        frames = cuda.to_device(np.array([source_x, 0., center_x, 0., 0., 1.]))
        surface_geometry.compose_surface[1, 8, 0](
            frames, offsets, source, points, 0, 0, 1, 1, 2, False,
            (False,) * 4, False, (0., 0., 0.), 1.,
        )
        cuda.synchronize()
        assert offsets.copy_to_host()[-1] == expected


@pytest.mark.parametrize("beam", ["fan", "cone"])
def test_oracle_detects_mutated_dispatch_under_observer(monkeypatch, beam):
    original = surface_geometry.compose_surface
    calls = []

    class ShiftedDispatch:
        def __getitem__(self, configuration):
            def launch(*args):
                calls.append((configuration, args[0].shape, args[1].shape))
                changed = list(args)
                changed[-2] = (.5, .5, .5)
                original[configuration](*changed)
            return launch

    monkeypatch.setattr(surface_geometry, "compose_surface", ShiftedDispatch())
    values = ([-1., 0.], [1., 0.], [0., 1.]) if beam == "fan" else (
        [-1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
    )
    frames = tuple(torch.tensor([value], dtype=torch.float64) for value in values)
    offsets = torch.tensor([[.25, 0., .5]], dtype=torch.float64)
    with pytest.raises(AssertionError):
        _exercise(frames, offsets, beam)
    assert len(calls) == 1
    assert len(calls[0][0]) == 3
    assert calls[0][1] == ((6,) if beam == "fan" else (15,))
    assert calls[0][2] == ((9,) if beam == "fan" else (13,))
