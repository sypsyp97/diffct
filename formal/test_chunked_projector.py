"""Exact tile models and finite production-kernel evidence, not whole-system proof.

PROOF: positive-integer half-open partition ownership (Z3), exact centred-grid
and detector-frame translation, and finite symbolic block/distributed adjoints.
BOUNDED: every forward/adjoint basis on tensor grids (H,W)=(3,4) and
(D,H,W)=(3,2,4), three views, 3 or (2,2) detectors, spacing 1.3, flat
asymmetric tiles or per-view curved/coupled one-cell tiles. Actual production
Siddon kernels run on each spatial block and batches of at most two views.
Three boundary domains use internal faces/edges/corners, inside sources and
misses. Three VJP domains sum actual block endpoint/source derivatives and
compare independent float64 differences with unchanged cell support at h/h2.
Excluded: scheduler/source equivalence, PyTorch dispatch and autograd, GPU
allocation bounds, hardware atomics/fastmath, and real distributed collectives.
"""

from fractions import Fraction
import itertools
import math

import numpy as np
import pytest
import sympy as sp
import z3

from test_cudasim import _matrix_geometry
from test_detector_surfaces import _cell_matrix, _launch, _world_points


BEAMS = ("parallel", "fan", "cone")


def _tiles(shape, chunk):
    for start in itertools.product(*(range(0, size, limit) for size, limit in zip(shape, chunk))):
        yield tuple(slice(i, min(i + limit, size)) for i, size, limit in zip(start, shape, chunk))


def test_PROOF_positive_integer_partition_has_one_half_open_owner():
    """All positive integer extents/limits; the model applies independently per axis."""
    size, limit, index, other = z3.Ints("size limit index other")
    owner = index / limit
    start = owner * limit
    end = z3.If(start + limit < size, start + limit, size)
    constraints = [size > 0, limit > 0, index >= 0, index < size]
    solver = z3.Solver()
    solver.add(*constraints, z3.Not(z3.And(start <= index, index < end)))
    assert solver.check() == z3.unsat
    solver = z3.Solver()
    other_end = z3.If((other + 1) * limit < size, (other + 1) * limit, size)
    solver.add(*constraints, other >= 0, other * limit <= index, index < other_end,
               other != owner)
    assert solver.check() == z3.unsat


@pytest.mark.parametrize("rank", [2, 3])
def test_BOUNDED_exact_cartesian_partition_and_cell_centres(rank):
    """Enumerate extents 1..3, limits 1..4, every cell, and exact rational spacing."""
    spacing = Fraction(13, 10)
    for shape in itertools.product(range(1, 4), repeat=rank):
        for chunk in itertools.product(range(1, 5), repeat=rank):
            counts = {}
            for block in _tiles(shape, chunk):
                lengths = tuple(s.stop - s.start for s in block)
                shift = tuple((Fraction(s.start) + Fraction(length - size, 2)) * spacing
                              for s, length, size in zip(block, lengths, shape))
                for local in itertools.product(*(range(length) for length in lengths)):
                    global_index = tuple(s.start + i for s, i in zip(block, local))
                    counts[global_index] = counts.get(global_index, 0) + 1
                    for i, j, size, length, centre in zip(global_index, local, shape, lengths, shift):
                        global_centre = (Fraction(i) + Fraction(1, 2) - Fraction(size, 2)) * spacing
                        local_centre = (Fraction(j) + Fraction(1, 2) - Fraction(length, 2)) * spacing
                        assert global_centre == local_centre + centre
            assert len(counts) == math.prod(shape) and set(counts.values()) == {1}


def test_PROOF_frame_translation_preserves_global_cells_and_ray_segments():
    size, length, start, index, spacing = sp.symbols("N L b i s", real=True)
    shift = (start + (length - size) / 2) * spacing
    global_cell = (start + index + sp.Rational(1, 2) - size / 2) * spacing
    local_cell = (index + sp.Rational(1, 2) - length / 2) * spacing
    assert sp.expand(local_cell + shift - global_cell) == 0
    centre = sp.Matrix(sp.symbols("cx cy cz", real=True))
    source = sp.Matrix(sp.symbols("sx sy sz", real=True))
    translation = sp.Matrix(sp.symbols("tx ty tz", real=True))
    u = sp.Matrix(sp.symbols("ux uy uz", real=True))
    v = sp.Matrix(sp.symbols("vx vy vz", real=True))
    a, b, c, tau = sp.symbols("a b c tau", real=True)
    point = centre + a * u + b * v + c * u.cross(v)
    local_point = centre - translation + a * u + b * v + c * u.cross(v)
    assert sp.expand(local_point - (point - translation)) == sp.zeros(3, 1)
    local_ray = source - translation + tau * (local_point - (source - translation))
    assert sp.expand(local_ray + translation - (source + tau * (point - source))) == sp.zeros(3, 1)


def test_PROOF_block_adjoint_and_replicated_distributed_sum():
    """Finite symbolic model: four cells, three rays, two disjoint block embeddings."""
    first = sp.Matrix(3, 2, sp.symbols("a0:6"))
    second = sp.Matrix(3, 2, sp.symbols("b0:6"))
    r0, r1 = sp.eye(4)[:2, :], sp.eye(4)[2:, :]
    x = sp.Matrix(sp.symbols("x0:4"))
    y = sp.Matrix(sp.symbols("y0:3"))
    operator = first * r0 + second * r1
    adjoint = r0.T * first.T * y + r1.T * second.T * y
    assert sp.expand(y.dot(operator * x) - x.dot(adjoint)) == 0
    assert sp.expand(adjoint - operator.T * y) == sp.zeros(4, 1)
    rank0, rank1, empty = operator[:1, :], operator[1:, :], sp.zeros(0, 4)
    w0, w1 = sp.Matrix(sp.symbols("w0:4")), sp.Matrix(sp.symbols("v0:4"))
    replicated = rank0.T * y[:1, :] + rank1.T * y[1:, :] + empty.T * sp.zeros(0, 1)
    expected = y[:1, :].dot(rank0 * (w0 + w1)) + y[1:, :].dot(rank1 * (w0 + w1))
    assert sp.expand(w0.dot(replicated) + w1.dot(replicated) - expected) == 0


def _domain(beam):
    shape = (3, 2, 4) if beam == "cone" else (3, 4)
    chunk = (2, 1, 3) if beam == "cone" else (2, 3)
    detector = (2, 2) if beam == "cone" else (3,)
    return shape, chunk, detector


def _kernel_layout(image, beam):
    return image.transpose(2, 1, 0).copy() if beam == "cone" else image.copy()


def _tensor_layout(image, beam):
    return image.transpose(2, 1, 0).copy() if beam == "cone" else image.copy()


def _reference(beam, shape, spacing, first, points):
    kernel_shape = shape[::-1] if beam == "cone" else shape
    matrix = _cell_matrix(beam, kernel_shape, spacing, first, points)
    if beam == "cone":
        columns = np.arange(math.prod(shape)).reshape(kernel_shape).transpose(2, 1, 0).ravel()
        matrix = matrix[:, columns]
    return matrix


def _local_frame(beam, shape, block, spacing, geometry, points, view_slice):
    lengths = np.asarray([s.stop - s.start for s in block])
    starts = np.asarray([s.start for s in block])
    shift = (starts + (lengths - np.asarray(shape)) / 2) * spacing
    shift = shift[::-1]  # Tensor order [z,]y,x -> physical x,y[,z].
    local = []
    for index, component in enumerate(geometry):
        value = component[view_slice].astype(np.float64)
        if index == 1 or (index == 0 and beam != "parallel"):
            value = value - shift
        local.append(value.astype(np.float32))
    return tuple(local), (points[view_slice].astype(np.float64) - shift).astype(np.float32)


def _block_forward(kernels, beam, image, shape, block, spacing, geometry, points, pitches, endpoints):
    result = np.zeros(points.shape[:-1], dtype=np.float32)
    for start in range(0, len(geometry[0]), 2):
        views = slice(start, start + 2)
        local, positions = _local_frame(beam, shape, block, spacing, geometry, points, views)
        part = np.zeros(positions.shape[:-1], dtype=np.float32)
        _launch(kernels, beam, _kernel_layout(image, beam), part, spacing, local, pitches,
                positions if endpoints else None)
        result[views] = part
    return result


def _block_adjoint(kernels, beam, sino, shape, block, spacing, geometry, points, pitches, endpoints):
    block_shape = tuple(s.stop - s.start for s in block)
    image = np.zeros(block_shape, dtype=np.float32)
    for start in range(0, len(geometry[0]), 2):
        views = slice(start, start + 2)
        local, positions = _local_frame(beam, shape, block, spacing, geometry, points, views)
        part = _kernel_layout(np.zeros(block_shape, dtype=np.float32), beam)
        _launch(kernels, beam, part, sino[views].copy(), spacing, local, pitches,
                positions if endpoints else None, backward=True)
        image += _tensor_layout(part, beam)
    return image


@pytest.mark.parametrize("beam", BEAMS)
@pytest.mark.parametrize("curved", [False, True], ids=["flat-asymmetric", "curved-one-cell"])
def test_BOUNDED_all_tiled_forward_and_adjoint_basis_matrices(simulated_kernels, beam, curved):
    shape, chunk, detector = _domain(beam)
    spacing, pitches = 1.3, (.67 * 1.3, .43 * 1.3)
    geometry = _matrix_geometry(beam, spacing)
    surface = ("coupled" if beam == "cone" else "arc") if curved else "flat"
    points = _world_points(beam, geometry, detector, pitches, spacing, surface, curved)
    chunk = (1,) * len(shape) if curved else chunk
    reference = _reference(beam, shape, spacing, geometry[0], points)
    forward, adjoint = np.zeros_like(reference), np.zeros_like(reference.T)
    for block in _tiles(shape, chunk):
        block_shape = tuple(s.stop - s.start for s in block)
        for index in np.ndindex(block_shape):
            image = np.zeros(block_shape, dtype=np.float32)
            image[index] = 1
            column = np.ravel_multi_index(tuple(s.start + i for s, i in zip(block, index)), shape)
            forward[:, column] = _block_forward(simulated_kernels, beam, image, shape, block,
                                                spacing, geometry, points, pitches, curved).ravel()
        for column in range(math.prod(points.shape[:-1])):
            sino = np.zeros(points.shape[:-1], dtype=np.float32)
            sino.flat[column] = 1
            back = _block_adjoint(simulated_kernels, beam, sino, shape, block, spacing,
                                  geometry, points, pitches, curved)
            assembled = adjoint[:, column].reshape(shape)
            assembled[block] = back
    np.testing.assert_allclose(forward, reference, rtol=4e-5, atol=5e-6)
    np.testing.assert_allclose(adjoint, reference.T, rtol=4e-5, atol=5e-6)
    x, y = np.linspace(-.6, 1.2, forward.shape[1]), np.linspace(-.7, .9, forward.shape[0])
    np.testing.assert_allclose(y @ forward @ x, x @ adjoint @ y, rtol=4e-5, atol=5e-6)


def _boundary_geometry(beam):
    first = np.array([[-5., .5], [1., -5.], [-5., -5.], [.13, .17], [5., 5.]])
    centre = np.array([[5., .5], [1., 5.], [5., 5.], [5., 1.7], [6., 5.]])
    if beam == "cone":
        first = np.column_stack((first, [.5, .5, -5., .21, 5.]))
        centre = np.column_stack((centre, [.5, .5, 5., -.83, 5.]))
    direction = centre - first
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    u = np.column_stack((-direction[:, 1], direction[:, 0]))
    if beam == "cone":
        u = np.column_stack((u, np.zeros(5)))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        geometry = first, centre, u, np.cross(direction, u)
    else:
        geometry = (direction if beam == "parallel" else first), centre, u
    geometry = tuple(g.astype(np.float32) for g in geometry)
    points = centre[:, None, None, :] if beam == "cone" else centre[:, None, :]
    return geometry, points.astype(np.float32)


@pytest.mark.parametrize("beam", BEAMS)
def test_BOUNDED_internal_block_boundaries_source_inside_and_miss(simulated_kernels, beam):
    shape, chunk, _ = _domain(beam)
    geometry, points = _boundary_geometry(beam)
    reference = _reference(beam, shape, 1., geometry[0], points)
    assert reference[0].sum() > 0 and not reference[-1].any()
    image = np.linspace(-.3, 1.7, math.prod(shape), dtype=np.float32).reshape(shape)
    sino = np.linspace(-.4, .8, len(points), dtype=np.float32).reshape(points.shape[:-1])
    forward, adjoint = np.zeros_like(sino), np.zeros_like(image)
    for block in _tiles(shape, chunk):
        forward += _block_forward(simulated_kernels, beam, image[block], shape, block,
                                   1., geometry, points, (1., 1.), True)
        adjoint[block] = _block_adjoint(simulated_kernels, beam, sino, shape, block,
                                       1., geometry, points, (1., 1.), True)
    np.testing.assert_allclose(forward.ravel(), reference @ image.ravel(), rtol=4e-5, atol=5e-6)
    np.testing.assert_allclose(adjoint.ravel(), reference.T @ sino.ravel(), rtol=4e-5, atol=5e-6)


def _vjp_launch(kernels, beam, image, sino, spacing, geometry, points, pitches):
    image = _kernel_layout(image, beam)
    f = np.float32
    views = len(geometry[0])
    gradients = [np.zeros_like(g) for g in geometry]
    endpoint_gradient = np.zeros_like(points)
    if beam == "cone":
        nx, ny, nz = image.shape
        nu, nv = sino.shape[1:]
        args = (image, nx, ny, nz, sino, views, nu, nv, f(pitches[0]), f(pitches[1]),
                *geometry, f(nx / 2), f(ny / 2), f(nz / 2), f(spacing),
                *gradients, points, endpoint_gradient)
        kernels._cone_3d_geometry_vjp_kernel[(nv, nu, views), (1, 1, 1)](*args)
    else:
        ny, nx = image.shape
        ndet = sino.shape[1]
        args = (image, nx, ny, sino, views, ndet, f(pitches[0]), *geometry,
                f(nx / 2), f(ny / 2), f(spacing), *gradients, points, endpoint_gradient)
        getattr(kernels, f"_{beam}_2d_geometry_vjp_kernel")[(views, ndet), (1, 1)](*args)
    return gradients[0], endpoint_gradient


@pytest.mark.parametrize("beam", BEAMS)
def test_BOUNDED_tile_entry_exit_vjp_cancellation_against_slab_differences(simulated_kernels, beam):
    shape, chunk, detector = _domain(beam)
    spacing, pitches = 1.3, (.67 * 1.3, .43 * 1.3)
    geometry = _matrix_geometry(beam, spacing)
    points = _world_points(beam, geometry, detector, pitches, spacing,
                           "coupled" if beam == "cone" else "arc", True)
    image = np.sin(np.arange(math.prod(shape)) * .7).astype(np.float32).reshape(shape)
    sino = np.linspace(-.9, 1.2, math.prod(points.shape[:-1]), dtype=np.float32).reshape(points.shape[:-1])
    first_gradient, point_gradient = np.zeros_like(geometry[0], dtype=np.float64), np.zeros_like(points, dtype=np.float64)
    for block in _tiles(shape, chunk):
        for start in range(0, len(geometry[0]), 2):
            views = slice(start, start + 2)
            local, positions = _local_frame(beam, shape, block, spacing, geometry, points, views)
            first, endpoint = _vjp_launch(simulated_kernels, beam, image[block], sino[views].copy(),
                                          spacing, local, positions, pitches)
            first_gradient[views] += first
            point_gradient[views] += endpoint
    base = _reference(beam, shape, spacing, geometry[0], points)

    def finite_difference(first, endpoints, first_direction=None, point_direction=None):
        estimates = []
        for step in (1e-4 * spacing, .5e-4 * spacing):
            plus_first, minus_first = first.copy(), first.copy()
            plus_points, minus_points = endpoints.copy(), endpoints.copy()
            if first_direction is not None:
                plus_first += step * first_direction
                minus_first -= step * first_direction
            if point_direction is not None:
                plus_points += step * point_direction
                minus_points -= step * point_direction
            plus = _reference(beam, shape, spacing, plus_first, plus_points)
            minus = _reference(beam, shape, spacing, minus_first, minus_points)
            assert np.array_equal(plus > 0, base > 0) and np.array_equal(minus > 0, base > 0)
            estimates.append(sino.ravel().astype(np.float64) @ ((plus - minus) @ image.ravel()) / (2 * step))
        np.testing.assert_allclose(estimates[0], estimates[1], rtol=1e-5, atol=1e-6)
        return estimates[1]

    expected = np.zeros_like(point_gradient)
    for index in np.ndindex(points.shape):
        direction = np.zeros_like(points, dtype=np.float64)
        direction[index] = 1
        expected[index] = finite_difference(geometry[0].astype(np.float64), points.astype(np.float64),
                                            point_direction=direction)
    assert np.max(np.abs(expected)) > 1e-3
    np.testing.assert_allclose(point_gradient, expected, rtol=4e-4, atol=4e-5)
    direction = np.random.default_rng(415).normal(size=geometry[0].shape)
    if beam == "parallel":
        first = geometry[0].astype(np.float64)
        direction -= (direction * first).sum(-1, keepdims=True) * first
    direction /= np.linalg.norm(direction)
    expected_first = finite_difference(geometry[0].astype(np.float64), points.astype(np.float64),
                                       first_direction=direction)
    assert abs(expected_first) > 1e-3
    np.testing.assert_allclose(np.sum(first_gradient * direction), expected_first, rtol=4e-4, atol=4e-5)
