"""Streamed analytical output must not overwrite parameters reread by later tiles."""

import math

import numpy as np
import pytest
import torch

import diffct


BEAMS = ("parallel", "fan", "cone")
WEIGHTS = ("projection_weights", "angular_weights", "filter_response")
ALIAS_ERROR = "overlap|alias|backing"


class _ObservedStore:
    """Delegate real storage while recording all numerical I/O."""

    def __init__(self, store):
        self.store = store
        self.reads, self.writes, self.flushes = [], [], 0

    def __getattr__(self, name):
        return getattr(self.store, name)

    def read(self, index):
        self.reads.append(index)
        return self.store.read(index)

    def write(self, index, value, *, accumulate=False):
        self.writes.append(index)
        return self.store.write(index, value, accumulate=accumulate)

    def flush(self):
        self.flushes += 1
        self.store.flush()


def _case(beam):
    angles = torch.arange(4, dtype=torch.float32) * (2 * math.pi / 4)
    if beam == "parallel":
        trajectory = diffct.circular_trajectory_2d_parallel(4, device="cpu")
    elif beam == "fan":
        trajectory = diffct.circular_trajectory_2d_fan(4, 6., 12., device="cpu")
    else:
        trajectory = diffct.circular_trajectory_3d(4, 6., 12., device="cpu")
    shape = (2, 4, 4) if beam == "cone" else (4, 4)
    data_shape = (4, 4, 2) if beam == "cone" else (4, 4)
    data = torch.arange(math.prod(data_shape), dtype=torch.float32).reshape(data_shape) / 10
    options = dict(backend="torch", precision="float32", voxel_spacing=.4, view_chunk_size=2,
                   volume_chunk_shape=(1, *shape[1:]))
    if beam != "cone":
        options["beam"] = beam
    return data, trajectory, shape, angles, options


def _weight_shape(name, data):
    return {"projection_weights": tuple(data.shape), "angular_weights": (len(data),),
            "filter_response": (data.shape[1] + 1,)}[name]


def _reconstruct(beam, source, trajectory, shape, angles, options, *, output=None):
    name = "fdk" if beam == "cone" else "fbp"
    function = getattr(diffct, name if output is None else name + "_into")
    arguments = (source,) if output is None else (source, output)
    return function(*arguments, trajectory, shape, angles, **options)


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("name", WEIGHTS)
@pytest.mark.parametrize("output_kind", ["tensor", "tensor-store", "delegated", "npy"])
def test_into_rejects_reread_weight_alias_before_any_numerical_io(
    beam, name, output_kind, tmp_path,
):
    data, trajectory, shape, angles, options = _case(beam)
    source = _ObservedStore(diffct.TensorStore(data))
    if output_kind == "npy":
        raw_output = diffct.NpyStore.create(tmp_path / "output.npy", shape)
        raw_output._array[...] = 1.
        # Independent tensor storage still addresses the writable mapping.
        output_tensor = torch.from_numpy(raw_output._array)
        output = _ObservedStore(_ObservedStore(raw_output))
    else:
        output_tensor = torch.ones(shape)
        raw_output = diffct.TensorStore(output_tensor)
        output = {"tensor": output_tensor, "tensor-store": raw_output,
                  "delegated": _ObservedStore(_ObservedStore(raw_output))}[output_kind]
    weight_shape = _weight_shape(name, data)
    weight = output_tensor.reshape(-1)[:math.prod(weight_shape)].reshape(weight_shape)
    saved_output, saved_weight, saved_data = output_tensor.clone(), weight.clone(), data.clone()

    with pytest.raises(ValueError, match=ALIAS_ERROR):
        _reconstruct(beam, source, trajectory, shape, angles,
                     {**options, name: weight}, output=output)

    assert source.reads == [] and source.writes == [] and source.flushes == 0
    if isinstance(output, _ObservedStore):
        assert output.reads == [] and output.writes == [] and output.flushes == 0
        assert output.store.reads == [] and output.store.writes == [] and output.store.flushes == 0
    torch.testing.assert_close(output_tensor, saved_output, rtol=0, atol=0)
    torch.testing.assert_close(weight, saved_weight, rtol=0, atol=0)
    torch.testing.assert_close(data, saved_data, rtol=0, atol=0)
    if output_kind == "npy":
        np.testing.assert_array_equal(np.load(raw_output.path), saved_output.numpy())


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("name", WEIGHTS)
def test_into_rejects_dtype_reinterpretation_alias_even_when_each_tile_casts_weight(beam, name):
    data, trajectory, shape, angles, options = _case(beam)
    weight_shape = _weight_shape(name, data)
    count = math.prod(weight_shape)
    backing = torch.ones(max(math.prod(shape), 2 * count), dtype=torch.float32)
    output_tensor = backing[:math.prod(shape)].reshape(shape)
    weight = backing.view(torch.float64)[:count].reshape(weight_shape)
    assert weight.dtype != output_tensor.dtype and torch.isfinite(weight).all()
    source = _ObservedStore(diffct.TensorStore(data))
    output = _ObservedStore(diffct.TensorStore(output_tensor))
    saved = backing.clone()

    with pytest.raises(ValueError, match=ALIAS_ERROR):
        _reconstruct(beam, source, trajectory, shape, angles,
                     {**options, name: weight}, output=output)

    assert source.reads == [] and output.writes == [] and output.flushes == 0
    torch.testing.assert_close(backing, saved, rtol=0, atol=0)


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("name", WEIGHTS)
@pytest.mark.parametrize("delegated", [False, True])
def test_into_accepts_disjoint_interleaved_output_and_weight_views(beam, name, delegated):
    data, trajectory, shape, angles, options = _case(beam)
    weight_shape = _weight_shape(name, data)
    backing = torch.full((2 * max(math.prod(shape), math.prod(weight_shape)),), -19.)
    output_tensor = backing[::2][:math.prod(shape)].reshape(shape)
    weight = backing[1::2][:math.prod(weight_shape)].reshape(weight_shape)
    weight.copy_(torch.linspace(.6, 1.2, weight.numel()).reshape(weight_shape))
    saved_weight, saved_data = weight.clone(), data.clone()
    expected = _reconstruct(beam, data, trajectory, shape, angles,
                            {**options, name: weight.clone()})
    assert torch.isfinite(expected).all() and expected.abs().max() > 0
    output = (_ObservedStore(_ObservedStore(diffct.TensorStore(output_tensor)))
              if delegated else output_tensor)
    source = _ObservedStore(diffct.TensorStore(data))

    result = _reconstruct(beam, source, trajectory, shape, angles,
                          {**options, name: weight}, output=output)

    assert result is output and source.reads
    if delegated:
        assert output.writes and output.flushes == 1
    torch.testing.assert_close(output_tensor, expected, rtol=3e-6, atol=3e-7)
    torch.testing.assert_close(weight, saved_weight, rtol=0, atol=0)
    torch.testing.assert_close(data, saved_data, rtol=0, atol=0)


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("consumed_input", ["geometry", "angles"])
def test_into_accepts_alias_of_snapshotted_geometry_or_consumed_angles(beam, consumed_input):
    data, trajectory, shape, angles, options = _case(beam)
    output = torch.zeros(shape)
    if consumed_input == "geometry":
        component = output.reshape(-1)[:trajectory[1].numel()].reshape_as(trajectory[1])
        component.copy_(trajectory[1])
        trajectory = (trajectory[0], component, *trajectory[2:])
    else:
        aliased_angles = output.reshape(-1)[:len(angles)]
        aliased_angles.copy_(angles)
        angles = aliased_angles
        # Parker beta must also be a constructor-time derived value.
        options["short_scan"] = beam != "parallel"
    expected = _reconstruct(beam, data, tuple(value.clone() for value in trajectory), shape,
                            angles.clone(), options)
    assert torch.isfinite(expected).all() and expected.abs().max() > 0

    result = _reconstruct(beam, data, trajectory, shape, angles, options, output=output)

    assert result is output
    torch.testing.assert_close(output, expected, rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("name", WEIGHTS)
def test_into_rejects_autograd_weights_before_numerical_io(beam, name):
    data, trajectory, shape, angles, options = _case(beam)
    weight = torch.ones(_weight_shape(name, data), requires_grad=True)
    output_tensor = torch.full(shape, 73.)
    source = _ObservedStore(diffct.TensorStore(data))
    output = _ObservedStore(diffct.TensorStore(output_tensor))
    saved_data, saved_weight, saved_output = data.clone(), weight.detach().clone(), output_tensor.clone()

    with pytest.raises(ValueError):
        _reconstruct(beam, source, trajectory, shape, angles,
                     {**options, name: weight}, output=output)

    assert source.reads == [] and source.writes == [] and source.flushes == 0
    assert output.reads == [] and output.writes == [] and output.flushes == 0
    assert weight.requires_grad and weight.grad is None
    torch.testing.assert_close(data, saved_data, rtol=0, atol=0)
    torch.testing.assert_close(weight.detach(), saved_weight, rtol=0, atol=0)
    torch.testing.assert_close(output_tensor, saved_output, rtol=0, atol=0)
