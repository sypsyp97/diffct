"""Compare bounded surface pullbacks with an independent whole-batch graph."""

import math
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from diffct.surface_gradients import surface_batch_vjp
from diffct.surfaces import ParameterizedSurface

DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
LAYOUTS = [(f"all-{str(dtype).split('.')[-1]}", (dtype,) * 5) for dtype in DTYPES]
LAYOUTS += [(f"mixed-{rotation}", tuple(DTYPES[(rotation + index) % 4]
                                      for index in range(5)))
            for rotation in range(4)]
TOLERANCES = {
    torch.float16: (2e-3, 2e-3),
    torch.bfloat16: (1.6e-2, 1.6e-2),
    torch.float32: (3e-6, 3e-6),
    torch.float64: (3e-12, 3e-12),
}


def _case(beam, per_view, layout, *, sampled=False, translated=False):
    rank = 3 if beam == "cone" else 2
    detector = (5, 9) if beam == "cone" else (13,)
    spacing = (.7, 1.1) if beam == "cone" else (.7,)
    pixels = (slice(1, 4), slice(2, 8)) if beam == "cone" else (slice(3, 11),)
    views = slice(2, 5)
    shape = tuple(selected.stop - selected.start for selected in pixels)
    generator = torch.Generator().manual_seed(9107)
    frames = []
    shift = torch.tensor((8192.25, -4096.5, 2048.125)[:rank], dtype=torch.float64)
    for index in range(rank + 1):
        value = .35 * torch.randn((3, rank), generator=generator, dtype=torch.float64)
        if index == 0:
            value[:, 0] -= 5.
        elif index == 1:
            value[:, 0] += 4.
        else:
            value[:, index - 1] += .9
        if translated and (index == 1 or index == 0 and beam != "parallel"):
            value += shift
        frames.append(value.to(layout[index]).requires_grad_())
    frames = tuple(frames)
    axes = [(torch.arange(selected.start, selected.stop, dtype=torch.float64)
             + .5 - size / 2) * pitch
            for selected, size, pitch in zip(pixels, detector, spacing)]
    u, v = (torch.meshgrid(*axes, indexing="ij") if beam == "cone"
            else (axes[0], torch.zeros_like(axes[0])))
    ids = torch.arange(views.start, views.stop)
    offset_dtype = layout[rank + 1]
    if sampled:
        parameter = torch.tensor((1.15, -.13, .27), dtype=offset_dtype, requires_grad=True)
        coupling = torch.tensor((.21, -.11, .08, .14, .32, .17),
                                dtype=layout[0], requires_grad=True)
        unused = torch.tensor((.4, -.2), dtype=layout[1], requires_grad=True)

        def sampler(u, v, view_indices, parameter, coupling, unused):
            parameter, coupling = parameter.double(), coupling.double()
            frame_term = (.07 * frames[2].double().square().sum(-1)
                          + .03 * frames[1].double().sum(-1)
                          + .02 * frames[0].double().sin().sum(-1))
            if beam == "cone":
                frame_term = frame_term + .04 * frames[3].double().square().sum(-1)
            scale = 1 + .1 * coupling[view_indices] + .05 * coupling.mean() + frame_term
            scale = (scale.reshape(3, *([1] * u.ndim)) if per_view else scale.mean())
            tangent = parameter[0] * torch.sin(u / parameter[0]) + parameter[1] * scale * u.square()
            vertical = (v + .03 * scale * u * v if beam == "cone"
                        else torch.zeros_like(tangent))
            depth = parameter[2] * scale * (.2 + u.square() + .1 * v.square())
            return torch.stack((tangent, vertical + torch.zeros_like(tangent), depth), -1).to(offset_dtype)

        surface = ParameterizedSurface(sampler, (parameter, coupling, unused))
        offsets = surface(u, v, ids)
        inputs = (*frames, *surface.parameters)
    else:
        offset_shape = (3, *shape, 3) if per_view else (*shape, 3)
        raw = .4 * torch.randn(offset_shape, generator=generator, dtype=torch.float64)
        if beam != "cone":
            raw[..., 1] = 0.
        offsets = raw.to(offset_dtype).requires_grad_()
        inputs = (*frames, offsets)
    point_count = 3 * math.prod(shape) * rank
    source_weights = torch.cos(.31 * torch.arange(3 * rank, dtype=torch.float32)).reshape(3, rank)
    point_weights = torch.sin(.17 + .23 * torch.arange(point_count, dtype=torch.float32)).reshape(3, *shape, rank)
    return SimpleNamespace(beam=beam, rank=rank, detector=detector, spacing=spacing,
                           pixels=pixels, views=views, shape=shape, inputs=inputs,
                           offsets=offsets, weights=(source_weights, point_weights),
                           legacy=not sampled, shift=shift if translated else None)


def _whole_batch(case):
    """Build the complete tensor expression without the replay helper."""
    frames = case.inputs[:case.rank + 1]
    if case.legacy:
        frames = tuple(value.to(torch.promote_types(torch.float32, value.dtype))
                       for value in frames)
        offsets = case.offsets
    else:
        dtype = torch.float32
        for value in (*frames, case.offsets):
            dtype = torch.promote_types(dtype, value.dtype)
        frames = tuple(value.to(dtype) for value in frames)
        offsets = case.offsets.to(dtype)
    if offsets.ndim == len(case.shape) + 1:
        offsets = offsets.unsqueeze(0)
    tangent, vertical, depth = offsets.unbind(-1)
    frame_shape = (3, *([1] * len(case.shape)), case.rank)
    source = frames[0]
    center, axis_u = (value.reshape(frame_shape) for value in frames[1:3])
    points = center + tangent[..., None] * axis_u
    if case.beam == "cone":
        axis_v = frames[3].reshape(frame_shape)
        normal_dtype = torch.promote_types(axis_u.dtype, axis_v.dtype)
        normal = torch.linalg.cross(axis_u.to(normal_dtype), axis_v.to(normal_dtype), dim=-1)
        points = points + vertical[..., None] * axis_v + depth[..., None] * normal
    else:
        normal = torch.stack((axis_u[..., 1], -axis_u[..., 0]), -1)
        points = points + depth[..., None] * normal
    if case.shift is not None:
        points = points - case.shift.to(points.dtype)
        if case.beam != "parallel":
            source = source - case.shift.to(source.dtype)
    loss = ((source.float() * case.weights[0]).sum()
            + (points.float() * case.weights[1]).sum())
    return torch.autograd.grad(loss, case.inputs, allow_unused=True)


def _replay(case, capacity):
    return surface_batch_vjp(case.beam, case.detector, case.spacing, case.inputs,
                             case.offsets, case.views, case.pixels, .8,
                             torch.device("cpu"), case.weights, capacity,
                             native_legacy=case.legacy)


def _assert_gradients(actual, expected, inputs):
    assert len(actual) == len(expected) == len(inputs)
    for index, (gradient, reference, original) in enumerate(zip(actual, expected, inputs)):
        if reference is None:
            if gradient is not None:
                assert gradient.device.type == "cpu" and gradient.dtype == original.dtype
                assert gradient.shape == original.shape
                torch.testing.assert_close(gradient, torch.zeros_like(original), rtol=0, atol=0)
            continue
        assert gradient is not None, f"missing gradient for input {index}"
        assert gradient.device.type == "cpu" and gradient.dtype == original.dtype
        assert gradient.shape == original.shape and torch.isfinite(gradient).all()
        rtol, atol = TOLERANCES[original.dtype]
        torch.testing.assert_close(gradient, reference, rtol=rtol, atol=atol,
                                   msg=lambda message, index=index: f"input {index}: {message}")


@pytest.mark.parametrize("beam", ("cone", "fan", "parallel"))
@pytest.mark.parametrize("per_view", (False, True), ids=("shared", "per-view"))
@pytest.mark.parametrize("layout", [values for _, values in LAYOUTS],
                         ids=[name for name, _ in LAYOUTS])
@pytest.mark.parametrize("sampled", (False, True), ids=("legacy", "parameterized"))
def test_replay_matches_whole_batch_autograd_for_mixed_dtypes(beam, per_view, layout, sampled):
    reference = _case(beam, per_view, layout, sampled=sampled)
    expected = _whole_batch(reference)
    # These limits include one pixel, a partial row, a row-crossing tail, and the whole batch.
    capacities = (1, 4, 7, 3 * math.prod(reference.shape) + 1)
    for capacity in capacities:
        case = _case(beam, per_view, layout, sampled=sampled)
        actual = _replay(case, capacity)
        _assert_gradients(actual, expected, case.inputs)
    if sampled:
        assert expected[-1] is None
        assert expected[-2].abs().sum() > 0
        assert expected[-2][0] != 0, "the omitted global view must affect the sampler"


@pytest.mark.parametrize("beam", ("cone", "fan", "parallel"))
@pytest.mark.parametrize("sampled", (False, True), ids=("legacy", "parameterized"))
def test_source_cotangent_is_added_once_per_view(beam, sampled):
    case = _case(beam, True, (torch.float64,) * 5, sampled=sampled)
    case.weights = (case.weights[0], torch.zeros_like(case.weights[1]))
    actual = _replay(case, 1)
    torch.testing.assert_close(actual[0], case.weights[0].double(), rtol=0, atol=0)
    for gradient, original in zip(actual[1:], case.inputs[1:]):
        if gradient is not None:
            torch.testing.assert_close(gradient, torch.zeros_like(original), rtol=0, atol=0)


@pytest.mark.parametrize("beam", ("cone", "fan", "parallel"))
@pytest.mark.parametrize("per_view", (False, True), ids=("shared", "per-view"))
def test_translation_keeps_native_cotangents_and_original_frame_gradients(beam, per_view):
    layout = (torch.float64,) * 5
    original = _case(beam, per_view, layout)
    expected = _whole_batch(original)
    translated = _case(beam, per_view, layout, translated=True)
    translated_expected = _whole_batch(translated)
    _assert_gradients(translated_expected, expected, translated.inputs)
    translated = _case(beam, per_view, layout, translated=True)
    _assert_gradients(_replay(translated, 4), expected, translated.inputs)


class _CompositionBudget(TorchDispatchMode):
    """Observe point-sized products while autograd records the replay graph."""

    def __init__(self, rank, capacity):
        super().__init__()
        self.rank, self.capacity = rank, capacity

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        if (function == torch.ops.aten.mul.Tensor and torch.is_grad_enabled()
                and isinstance(result, torch.Tensor) and result.ndim >= 3
                and result.shape[-1] == self.rank):
            pixels = result.numel() // self.rank
            assert pixels <= self.capacity, "surface replay composed more pixels than its capacity"
        return result


@pytest.mark.parametrize("beam", ("cone", "fan", "parallel"))
@pytest.mark.parametrize("sampled", (False, True), ids=("legacy", "parameterized"))
def test_replay_point_arithmetic_obeys_pixel_capacity(beam, sampled):
    layout = (torch.float64,) * 5
    expected = _whole_batch(_case(beam, True, layout, sampled=sampled))
    case = _case(beam, True, layout, sampled=sampled)
    with _CompositionBudget(case.rank, 4):
        actual = _replay(case, 4)
    _assert_gradients(actual, expected, case.inputs)
