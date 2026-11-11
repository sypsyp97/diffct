"""Numerical stores reuse one legacy surface sample per public invocation."""

import math

import pytest
import torch

from tests.test_portable_execution_bounds import _RecordedStore
from tests.test_trajectory_surface_integration import _data, _frames, _offsets, _operator, _ray_matrix


@pytest.fixture
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("operation", ["project", "backproject"])
def test_public_into_samples_legacy_surface_once_and_refreshes_next_call(operation, single_cpu_thread):
    coefficient = .037
    calls = []

    def surface(u, v):
        calls.append(coefficient)
        return _offsets(u, v, coefficient)

    frames = _frames(3)
    operator = _operator(frames, surface, volume_chunk_shape=(2, 3), view_chunk_size=2,
                         detector_chunk_shape=(3,))
    calls.clear()  # Constructor admission is independent of numerical calls.
    data = _data(operator.volume_shape if operation == "project" else operator.projection_shape)
    expected_results = []
    for coefficient in (.037, .193):
        matrix = _ray_matrix(frames, coefficient)
        expected = ((matrix @ data.flatten()).reshape(operator.projection_shape) if operation == "project"
                    else (matrix.T @ data.flatten()).reshape(operator.volume_shape))
        source = _RecordedStore(data.clone())
        sink = _RecordedStore(torch.full_like(expected, torch.nan))

        assert getattr(operator, operation + "_into")(source, sink) is sink

        assert calls == [.037, .193][:len(expected_results) + 1], \
            "numerical into must reuse the invocation's full legacy surface sample"
        torch.testing.assert_close(sink.tensor, expected, rtol=3e-11, atol=3e-12)
        torch.testing.assert_close(source.tensor, data, rtol=0, atol=0)
        assert len(source.reads) > 1 and len(sink.writes) > 1
        assert all(rows <= 2 and columns <= 3 for _, (rows, columns), _ in source.reads)
        assert all(rows <= 2 and columns <= 3 for _, (rows, columns), _ in sink.writes)
        assert sum(math.prod(shape) for _, shape, _ in sink.writes) == expected.numel()
        assert sink.flushes == 1
        expected_results.append(expected)

    assert (expected_results[0] - expected_results[1]).abs().max() > 1e-3, \
        "surface refresh fixture must change the physical result"
