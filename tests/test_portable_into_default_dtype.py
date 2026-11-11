"""Explicit portable precision is independent of later global dtype changes."""

import pytest
import torch

import diffct
from tests.test_portable_execution_bounds import _RecordedStore


@pytest.mark.parametrize("method", ["project", "backproject"])
def test_explicit_float32_into_result_ignores_default_dtype_changed_after_construction(method):
    torch.set_num_threads(1)
    previous = torch.get_default_dtype()
    results = []
    try:
        torch.set_default_dtype(torch.float32)
        views = 1 if method == "project" else 4
        shape = (1, 4) if method == "project" else (1, 1)
        direction = torch.tensor([[1., 0.]], dtype=torch.float64).expand(views, -1).clone()
        centre = torch.zeros((views, 2), dtype=torch.float64)
        axis = torch.tensor([[0., 1.]], dtype=torch.float64).expand(views, -1).clone()
        owner = diffct.Projector((direction, centre, axis), shape, (1,), beam="parallel",
                                 backend="torch", precision="float32", voxel_spacing=1.,
                                 detector_spacing=1., volume_chunk_shape=(1, 1),
                                 view_chunk_size=1, detector_chunk_shape=(1,))
        source_shape = owner.volume_shape if method == "project" else owner.projection_shape
        output_shape = owner.projection_shape if method == "project" else owner.volume_shape
        # Unit chords expose cancellation across four spatial tiles or views.
        data = torch.tensor([2 ** 24, 1., -(2 ** 24), 2.], dtype=torch.float32).reshape(source_shape)
        for default in (torch.float32, torch.float64):
            torch.set_default_dtype(default)
            source = _RecordedStore(data.clone())
            output = _RecordedStore(torch.full(output_shape, torch.nan, dtype=torch.float32))
            assert getattr(owner, method + "_into")(source, output) is output
            assert source.reads and output.writes
            assert output.tensor.dtype == torch.float32
            assert torch.isfinite(output.tensor).all() and torch.count_nonzero(output.tensor) > 0
            torch.testing.assert_close(source.tensor, data, rtol=0, atol=0)
            results.append(output.tensor.clone())
    finally:
        torch.set_default_dtype(previous)
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
