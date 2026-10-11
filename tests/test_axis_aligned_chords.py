"""Public projection contract for odd grids and shortened final spatial tiles.

Central axis rays cross one complete voxel row. Their weights are exactly the
voxel spacing, independent of traversal order or the spatial partition.
The public API supports 2D parallel rays and 3D cone rays, not 3D parallel rays.
"""

import math

import pytest
import torch

from diffct import NpyStore, Projector, TensorStore

SHAPES = [(1, 7), (7, 1), (5, 7), (1, 5, 7), (5, 1, 7), (5, 7, 1), (3, 5, 7)]
DTYPES = [torch.float32, torch.float64]
CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for public projection")


def _operator(shape, geometry_dtype):
    # Build positive and negative central rays along each physical axis.
    dimensions = len(shape)
    directions = torch.cat((torch.eye(dimensions), -torch.eye(dimensions))).to(geometry_dtype)
    centers = 8 * directions
    u = torch.roll(directions, 1, dims=1)
    if dimensions == 2:
        trajectory = directions, centers, u
    else:
        v = torch.linalg.cross(directions, u)
        trajectory = -centers, centers, u, v
    return Projector(
        trajectory, shape,
        (1,) if len(shape) == 2 else (1, 1),
        beam="parallel" if len(shape) == 2 else "cone",
        voxel_spacing=0.7, devices=[0] if torch.cuda.is_available() else None,
        distributed=False, volume_chunk_shape=tuple(min(3, size) for size in shape),
        schedule="spatial",
    )


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("geometry_dtype", DTYPES)
def test_axis_aligned_public_geometry_constructs_on_cpu(shape, geometry_dtype):
    operator = _operator(shape, geometry_dtype)
    assert operator.volume_shape == shape
    assert operator.projection_shape == (2 * len(shape), *([1] * (len(shape) - 1)))
    assert operator.volume_chunk_shape[-1] == min(3, shape[-1])


@pytest.mark.cuda
@CUDA_REQUIRED
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("geometry_dtype", DTYPES)
@pytest.mark.parametrize("data_dtype", DTYPES)
@pytest.mark.parametrize("input_device", ["cpu", "cuda"])
def test_central_chords_and_adjoint_with_short_final_tiles(shape, geometry_dtype, data_dtype, input_device):
    operator = _operator(shape, geometry_dtype)
    constant = torch.full(shape, 1.25, dtype=data_dtype, device=input_device)
    projected = operator.project(constant)
    expected_chords = torch.tensor(list(shape[::-1]) * 2, dtype=torch.float64) * 0.7 * 1.25
    # Public outputs are float32 even when data and geometry are float64.
    assert projected.dtype == torch.float32
    assert projected.device == constant.device
    torch.testing.assert_close(projected.cpu().double().flatten(), expected_chords, rtol=3e-5, atol=3e-6)

    # Exact physical weights: each central ray crosses a full voxel row.
    dimensions = len(shape)
    matrix = torch.zeros((2 * dimensions, *shape), dtype=torch.float64)
    for view in range(2 * dimensions):
        tensor_axis = dimensions - 1 - view % dimensions
        index = [size // 2 for size in shape]
        index[tensor_axis] = slice(None)
        matrix[(view, *index)] = 0.7
    matrix = matrix.reshape(2 * dimensions, -1)
    image = (torch.arange(math.prod(shape), dtype=torch.float64).sin() + 0.4).reshape(shape)
    image = image.to(device=input_device, dtype=data_dtype)
    sino = torch.linspace(-0.7, 1.3, 2 * len(shape), dtype=data_dtype, device=input_device)
    sino = sino.reshape(operator.projection_shape)
    forward = operator.project(image)
    backward = operator.backproject(sino)
    image64 = image.cpu().double().flatten()
    sino64 = sino.cpu().double().flatten()
    torch.testing.assert_close(forward.cpu().double().flatten(), matrix @ image64, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(backward.cpu().double().flatten(), matrix.T @ sino64, rtol=3e-5, atol=3e-6)
    assert backward.dtype == torch.float32 and backward.device == sino.device
    lhs = torch.dot(forward.cpu().double().flatten(), sino64)
    rhs = torch.dot(image64, backward.cpu().double().flatten())
    torch.testing.assert_close(lhs, rhs, rtol=3e-5, atol=3e-6)


@pytest.mark.cuda
@CUDA_REQUIRED
@pytest.mark.parametrize("geometry_dtype", DTYPES)
@pytest.mark.parametrize("data_dtype", DTYPES)
@pytest.mark.parametrize("chunk_width", [3, 4])
@pytest.mark.parametrize("output_kind", ["tensor", "npy"])
def test_singleton_parallel_public_example(geometry_dtype, data_dtype, chunk_width, output_kind, tmp_path):
    trajectory = tuple(torch.tensor([vector], dtype=geometry_dtype)
                       for vector in [(1, 0), (8, 0), (0, 1)])
    operator = Projector(
        trajectory, (1, 7), (1,), beam="parallel", voxel_spacing=0.7,
        devices=[0], distributed=False, volume_chunk_shape=(1, chunk_width), schedule="spatial",
    )
    image = torch.ones((1, 7), dtype=data_dtype)
    sino = torch.ones((1, 1), dtype=data_dtype)
    if output_kind == "tensor":
        result = operator.project(image)
        adjoint = operator.backproject(sino)
    else:
        projected_store = NpyStore.create(tmp_path / "projection.npy", (1, 1))
        adjoint_store = NpyStore.create(tmp_path / "adjoint.npy", (1, 7))
        assert operator.project_into(TensorStore(image), projected_store) is projected_store
        assert operator.backproject_into(TensorStore(sino), adjoint_store) is adjoint_store
        # Reopen files to check the public storage result, including its flush.
        result = NpyStore(projected_store.path, mode="r").read((slice(None), slice(None)))
        adjoint = NpyStore(adjoint_store.path, mode="r").read((slice(None), slice(None)))
    torch.testing.assert_close(result, torch.tensor([[4.9]]), rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(adjoint, torch.full((1, 7), 0.7), rtol=3e-5, atol=3e-6)
