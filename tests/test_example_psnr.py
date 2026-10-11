"""Peak-one PSNR with bounded work for mixed CPU/CUDA state devices."""

from contextlib import nullcontext
import math

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from examples._common import psnr


class _NoWholeVolumeWork(TorchDispatchMode):
    def __init__(self, elements):
        super().__init__()
        self.elements = elements

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        arguments = tree_flatten((args, kwargs or {}))[0]
        storage = {(value.device, value.untyped_storage()._cdata)
                   for value in arguments if isinstance(value, torch.Tensor)}
        result = function(*args, **(kwargs or {}))
        for value in tree_flatten(result)[0]:
            if isinstance(value, torch.Tensor) and value.numel() >= self.elements:
                assert (value.device, value.untyped_storage()._cdata) in storage, \
                    ("complete-volume allocation or copy", str(function), value.shape)
        return result


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_psnr_cpu_peak_one_and_identical_inputs(noncontiguous):
    truth = torch.arange(30, dtype=torch.float64).reshape(2, 3, 5) / 17
    reconstruction = truth + torch.linspace(-.25, .5, 30).reshape(2, 3, 5)
    expected = -10 * math.log10(((reconstruction - truth).square().sum() / truth.numel()).item())
    if noncontiguous:
        reconstruction, truth = reconstruction.transpose(0, 2), truth.transpose(0, 2)
        assert not reconstruction.is_contiguous() and not truth.is_contiguous()
    assert psnr(reconstruction, truth) == pytest.approx(expected, abs=1e-10)
    assert psnr(truth, truth.clone()) == math.inf


@pytest.mark.parametrize("devices", [
    ("cpu", "cpu"),
    pytest.param(("cuda", "cpu"), marks=pytest.mark.cuda),
    pytest.param(("cpu", "cuda"), marks=pytest.mark.cuda),
    pytest.param(("cuda", "cuda"), marks=pytest.mark.cuda),
])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_psnr_device_pairs_weight_all_elements_with_bounded_mixed_device_work(devices, noncontiguous):
    if "cuda" in devices and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    shape = (67, 131, 257)
    truth = torch.linspace(-.5, 1.5, math.prod(shape)).reshape(shape)
    reconstruction = truth.clone()
    reconstruction[:23] += .125
    reconstruction[23:] -= .375
    expected = -10 * math.log10((reconstruction.double() - truth.double()).square().mean().item())
    reconstruction, truth = reconstruction.to(devices[0]), truth.to(devices[1])
    if noncontiguous:
        reconstruction, truth = reconstruction.transpose(0, 2), truth.transpose(0, 2)
        assert not reconstruction.is_contiguous() and not truth.is_contiguous()
    allocation_guard = _NoWholeVolumeWork(math.prod(shape)) if devices[0] != devices[1] else nullcontext()
    with allocation_guard:
        actual = psnr(reconstruction, truth)
    assert actual == pytest.approx(expected, abs=1e-6)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("devices", [("cuda", "cpu"), ("cpu", "cuda")])
def test_psnr_identical_mixed_device_inputs_return_infinity(devices):
    data = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5)
    reconstruction, truth = data.to(devices[0]).transpose(0, 2), data.to(devices[1]).transpose(0, 2)
    assert psnr(reconstruction, truth) == math.inf
