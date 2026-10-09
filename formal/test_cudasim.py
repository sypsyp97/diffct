"""BOUNDED checks of the actual Siddon kernels in Numba's CPU simulator.

The finite matrix domain has three beam types, two grids, two voxel pitches,
three views, and every image and sinogram basis vector. There is no sampling.
The chord domain enumerates the named rays below on the same grids and pitches.
These checks do not establish unbounded correctness or CUDA fastmath behavior.
"""

import itertools
import math

import pytest


BEAMS = ("parallel", "fan", "cone")
GRID_IDS = (0, 1)
PITCHES = (1.0, 1.7)
MATRIX_DOMAIN = tuple(itertools.product(BEAMS, GRID_IDS, PITCHES))
RTOL = 3e-5
ATOL = 3e-6


def _shape(beam, grid):
    # Two-dimensional arrays use (y, x); cone kernel arrays use (x, y, z).
    return ((2, 3, 2), (3, 2, 3))[grid] if beam == "cone" else ((2, 3), (3, 2))[grid]


def _slab_length(origin, direction, lower, upper, segment):
    """Independent float64 slab intersection, in physical coordinates."""
    lo, hi = (0.0, 1.0) if segment else (-math.inf, math.inf)
    for p, d, a, b in zip(origin, direction, lower, upper):
        if d == 0:
            # Half-open cells select a unique voxel on an internal grid plane.
            if not a <= p < b:
                return 0.0
        else:
            crossings = ((a - p) / d, (b - p) / d)
            lo = max(lo, min(crossings))
            hi = min(hi, max(crossings))
    return max(0.0, hi - lo) * math.sqrt(sum(d * d for d in direction))


def _rays(beam, geometry, detector_shape, detector_pitch):
    """Construct physical rays from the centred detector definition."""
    import numpy as np

    first, centre, u, *v = [array.astype(np.float64) for array in geometry]
    for view in range(len(first)):
        for pixel in np.ndindex(detector_shape):
            detector = centre[view].copy()
            detector += (pixel[0] - (detector_shape[0] - 1) / 2) * detector_pitch * u[view]
            if v:
                detector += (pixel[1] - (detector_shape[1] - 1) / 2) * detector_pitch * v[0][view]
            if beam == "parallel":
                yield (view, *pixel), detector, first[view], False
            else:
                yield (view, *pixel), first[view], detector - first[view], True


def _cell_matrix(beam, shape, pitch, geometry, detectors, detector_pitch):
    """Intersect each ray with each voxel; do not implement Siddon traversal."""
    import numpy as np

    xyz_shape = shape if beam == "cone" else shape[::-1]
    half = np.asarray(xyz_shape, dtype=np.float64) * pitch / 2
    matrix = np.zeros((len(geometry[0]) * math.prod(detectors), math.prod(shape)))
    for ray_index, origin, direction, segment in _rays(beam, geometry, detectors, detector_pitch):
        row = np.ravel_multi_index(ray_index, (len(geometry[0]), *detectors))
        for index in np.ndindex(shape):
            xyz_index = index if beam == "cone" else index[::-1]
            lower = np.asarray(xyz_index, dtype=np.float64) * pitch - half
            column = np.ravel_multi_index(index, shape)
            matrix[row, column] = _slab_length(origin, direction, lower, lower + pitch, segment)
    return matrix


def _launch(kernels, beam, image, sino, pitch, geometry, detector_pitch, backward=False):
    """Launch the production kernel directly on NumPy float32 arrays."""
    import numpy as np

    f = np.float32
    if beam == "cone":
        nx, ny, nz = image.shape
        views, nu, nv = sino.shape
        tail = (f(detector_pitch), f(detector_pitch), *geometry,
                f(nx / 2), f(ny / 2), f(nz / 2), f(pitch))
        args = ((sino, views, nu, nv, image, nx, ny, nz) if backward else
                (image, nx, ny, nz, sino, views, nu, nv))
        kernel = getattr(kernels, "_cone_3d_backward_kernel" if backward else "_cone_3d_forward_kernel")
        kernel[(nv, nu, views), (1, 1, 1)](*args, *tail)
    else:
        ny, nx = image.shape
        views, ndet = sino.shape
        tail = (f(detector_pitch), *geometry, f(nx / 2), f(ny / 2), f(pitch))
        args = ((sino, views, ndet, image, nx, ny) if backward else
                (image, nx, ny, sino, views, ndet))
        kernel = getattr(kernels, f"_{beam}_2d_{'backward' if backward else 'forward'}_kernel")
        kernel[(views, ndet), (1, 1)](*args, *tail)


def _matrix_geometry(beam, pitch):
    import numpy as np

    source = [(-5, .17), (.23, 5), (-4, -3)]
    detector = [(5, .17), (.23, -5), (4, 3)]
    u = [(0, 1), (1, 0), (-.6, .8)]
    if beam == "parallel":
        return tuple(np.asarray(a, dtype=np.float32) for a in (
            [(1, 0), (0, -1), (.8, .6)],
            np.asarray(source) * pitch, u))
    if beam == "cone":
        source = [(*p, .19) for p in source]
        detector = [(*p, -.13) for p in detector]
        u = [(*p, 0) for p in u]
    geometry = [np.asarray(source) * pitch, np.asarray(detector) * pitch, u]
    if beam == "cone":
        geometry.append([(0, 0, 1)] * 3)
    return tuple(np.asarray(a, dtype=np.float32) for a in geometry)


@pytest.mark.parametrize("beam,grid,pitch", MATRIX_DOMAIN)
def test_bounded_all_basis_matrices(simulated_kernels, beam, grid, pitch):
    """Enumerate every column of A and A^T, then compare an independent oracle."""
    import numpy as np

    shape = _shape(beam, grid)
    detectors = (2 + grid, 2) if beam == "cone" else (2 + grid,)
    sino_shape = (3, *detectors)
    geometry = _matrix_geometry(beam, pitch)
    detector_pitch = .7 * pitch
    forward = np.zeros((math.prod(sino_shape), math.prod(shape)))
    adjoint = np.zeros(forward.T.shape)
    for column in range(math.prod(shape)):
        image = np.zeros(shape, dtype=np.float32)
        image.flat[column] = 1
        sino = np.zeros(sino_shape, dtype=np.float32)
        _launch(simulated_kernels, beam, image, sino, pitch, geometry, detector_pitch)
        forward[:, column] = sino.ravel()
    for column in range(math.prod(sino_shape)):
        image = np.zeros(shape, dtype=np.float32)
        sino = np.zeros(sino_shape, dtype=np.float32)
        sino.flat[column] = 1
        _launch(simulated_kernels, beam, image, sino, pitch, geometry, detector_pitch, backward=True)
        adjoint[:, column] = image.ravel()
    np.testing.assert_allclose(adjoint, forward.T, rtol=RTOL, atol=ATOL)
    oracle = _cell_matrix(beam, shape, pitch, geometry, detectors, detector_pitch)
    np.testing.assert_allclose(forward, oracle, rtol=RTOL, atol=ATOL)


# Finite source/detector coordinates in voxel units. z is added for cone rays.
RAY_CASES = (
    ("axis", (-4, .19), (4, .19)),
    ("reverse_axis", (4, .19), (-4, .19)),
    ("miss", (-4, 4), (4, 4)),
    ("source_inside", (.21, .19), (4, .19)),
    ("detector_inside", (-4, .19), (.21, .19)),
    ("both_inside", (-.21, .19), (.21, .19)),
    ("distant_source", (-1e6, .19), (4, .23)),
    ("distant_detector", (-4, .19), (1e6, .23)),
    ("distant_small_direction", (-1e6, 1.6), (4, .9)),
    ("grazing_grid_plane", (-4, -1e-7), (4, 1e-7)),
)


@pytest.mark.parametrize("beam,grid,pitch", MATRIX_DOMAIN)
@pytest.mark.parametrize("case,start,end", RAY_CASES, ids=[case[0] for case in RAY_CASES])
def test_bounded_uniform_box_chords(simulated_kernels, beam, grid, pitch, case, start, end):
    """Check physical chord lengths, finite endpoints, misses, and small directions."""
    import numpy as np

    shape = _shape(beam, grid)
    if beam == "cone":
        start, end = (*start, .17), (*end, .17)
    start = np.asarray(start, dtype=np.float32) * np.float32(pitch)
    end = np.asarray(end, dtype=np.float32) * np.float32(pitch)
    if beam == "parallel":
        direction = end.astype(np.float64) - start.astype(np.float64)
        direction /= np.linalg.norm(direction)
        geometry = (direction[None].astype(np.float32), start[None], np.asarray([(0, 1)], dtype=np.float32))
    else:
        u = (0, 1, 0) if beam == "cone" else (0, 1)
        geometry = (start[None], end[None], np.asarray([u], dtype=np.float32))
        if beam == "cone":
            geometry += (np.asarray([(0, 0, 1)], dtype=np.float32),)
    detectors = (1, 1) if beam == "cone" else (1,)
    image = np.ones(shape, dtype=np.float32)
    sino = np.zeros((1, *detectors), dtype=np.float32)
    _launch(simulated_kernels, beam, image, sino, pitch, geometry, .7 * pitch)
    xyz_shape = shape if beam == "cone" else shape[::-1]
    half = np.asarray(xyz_shape) * pitch / 2
    _, origin, direction, segment = next(_rays(beam, geometry, detectors, .7 * pitch))
    expected = _slab_length(origin, direction, -half, half, segment)
    np.testing.assert_allclose(sino.item(), expected, rtol=RTOL, atol=ATOL)

    # A row-dependent image also detects assignment to the wrong row at grazing crossings.
    if case == "grazing_grid_plane":
        image = (np.indices(shape)[1 if beam == "cone" else 0] + 1).astype(np.float32)
        sino.fill(0)
        _launch(simulated_kernels, beam, image, sino, pitch, geometry, .7 * pitch)
        weights = _cell_matrix(beam, shape, pitch, geometry, detectors, .7 * pitch)
        np.testing.assert_allclose(sino.ravel(), weights @ image.ravel(), rtol=RTOL, atol=ATOL)
