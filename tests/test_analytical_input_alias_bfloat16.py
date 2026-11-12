"""Analytical output admission must compare logical bytes across tensor dtypes."""

import itertools
import math

import pytest
import torch

from .test_analytical_input_alias import (
    ALIAS_ERROR,
    BEAMS,
    WEIGHTS,
    _ObservedStore,
    _case,
    _reconstruct,
    _weight_shape,
)

import diffct


def _logical_bytes(tensor):
    """Enumerate byte addresses for these small fixtures without a NumPy cast."""
    size = tensor.element_size()
    start = tensor.data_ptr()
    return {
        start + sum(index * stride for index, stride in zip(indices, tensor.stride())) * size + byte
        for indices in itertools.product(*(range(length) for length in tensor.shape))
        for byte in range(size)
    }


def _shared_views(shape, weight_shape, halfword_offset):
    count = max(math.prod(shape), math.prod(weight_shape))
    backing = torch.full((3 * count,), -19., dtype=torch.float32)
    output = backing[::3][:math.prod(shape)].reshape(shape)
    weight = backing.view(torch.bfloat16)[halfword_offset::6][:math.prod(weight_shape)]
    return backing, output, weight.reshape(weight_shape)


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("name", WEIGHTS)
@pytest.mark.parametrize("layout", ["independent", "strided-shared"])
@pytest.mark.parametrize("delegated", [False, True])
def test_into_accepts_bfloat16_weights_with_float32_output(beam, name, layout, delegated):
    data, trajectory, shape, angles, options = _case(beam)
    weight_shape = _weight_shape(name, data)
    if layout == "independent":
        output_tensor = torch.full(shape, -19., dtype=torch.float32)
        weight = torch.empty(weight_shape, dtype=torch.bfloat16)
        assert output_tensor.untyped_storage().data_ptr() != weight.untyped_storage().data_ptr()
    else:
        # Each 12-byte slot contains four output bytes and a separate two-byte weight.
        backing, output_tensor, weight = _shared_views(shape, weight_shape, 2)
        assert output_tensor.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()
        assert not output_tensor.is_contiguous() and not weight.is_contiguous()
        assert backing.dtype == torch.float32
    weight.copy_(torch.linspace(.625, 1.25, weight.numel()).reshape(weight_shape))
    output_bytes, weight_bytes = _logical_bytes(output_tensor), _logical_bytes(weight)
    assert output_bytes.isdisjoint(weight_bytes)
    if layout == "strided-shared":
        assert max(min(output_bytes), min(weight_bytes)) <= min(max(output_bytes), max(weight_bytes))
    assert weight.dtype == torch.bfloat16 and weight.device.type == "cpu"
    assert options["precision"] == "float32" and options["backend"] == "torch"
    saved_weight, saved_data = weight.clone(), data.clone()
    expected = _reconstruct(
        beam, data, trajectory, shape, angles, {**options, name: weight.clone()},
    )
    assert expected.dtype == torch.float32
    assert torch.isfinite(expected).all() and expected.abs().max() > 0
    source = _ObservedStore(diffct.TensorStore(data))
    output = (
        _ObservedStore(_ObservedStore(diffct.TensorStore(output_tensor)))
        if delegated else output_tensor
    )

    result = _reconstruct(
        beam, source, trajectory, shape, angles, {**options, name: weight}, output=output,
    )

    assert result is output and source.reads
    assert source.writes == [] and source.flushes == 0
    if delegated:
        for store in (output, output.store):
            assert store.writes and store.flushes == 1
    assert output_tensor.dtype == torch.float32
    torch.testing.assert_close(output_tensor, expected, rtol=3e-6, atol=3e-7)
    torch.testing.assert_close(weight, saved_weight, rtol=0, atol=0)
    torch.testing.assert_close(data, saved_data, rtol=0, atol=0)


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("name", WEIGHTS)
@pytest.mark.parametrize("halfword_offset", [0, 1], ids=["same-start", "upper-half"])
def test_into_rejects_true_bfloat16_byte_overlap_before_numerical_io(
    beam, name, halfword_offset,
):
    data, trajectory, shape, angles, options = _case(beam)
    backing, output_tensor, weight = _shared_views(shape, _weight_shape(name, data), halfword_offset)
    weight.copy_(torch.linspace(.625, 1.25, weight.numel()).reshape(weight.shape))
    assert output_tensor.dtype == torch.float32 and weight.dtype == torch.bfloat16
    assert not output_tensor.is_contiguous() and not weight.is_contiguous()
    assert output_tensor.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()
    assert _logical_bytes(output_tensor) & _logical_bytes(weight)
    if halfword_offset == 1:
        assert weight.data_ptr() == output_tensor.data_ptr() + weight.element_size()
    saved_bytes = backing.view(torch.uint8).clone()
    saved_weight, saved_data = weight.clone(), data.clone()
    source = _ObservedStore(diffct.TensorStore(data))
    output = _ObservedStore(_ObservedStore(diffct.TensorStore(output_tensor)))

    with pytest.raises(ValueError, match=ALIAS_ERROR):
        _reconstruct(
            beam, source, trajectory, shape, angles, {**options, name: weight}, output=output,
        )

    for store in (source, output, output.store):
        assert store.reads == [] and store.writes == [] and store.flushes == 0
    torch.testing.assert_close(backing.view(torch.uint8), saved_bytes, rtol=0, atol=0)
    torch.testing.assert_close(weight, saved_weight, rtol=0, atol=0)
    torch.testing.assert_close(data, saved_data, rtol=0, atol=0)
