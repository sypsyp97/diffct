"""PROOF / BOUNDED evidence for the parameterized-detector endpoint contract.

PROOF checks exact local-frame, arc and first-order chain-rule identities.
They concern the stated mathematical model, not complete source equivalence,
CUDA compilation, floating-point arithmetic or traversal completeness. The
flat-grid expressions already have selected source guards in test_proofs.py;
the flat comparisons here also bind that limit to actual kernel execution.

BOUNDED checks launch production forward/backward/VJP kernels in CUDASIM on
NumPy float32 arrays. Every image and sinogram basis is enumerated for 16
domains: parallel/fan arcs and cone arcs/coupled surfaces, shared/per-view
local offsets, and grids (y,x)=(2,3)/(3,2) or (x,y,z)=(2,3,2)/(3,2,3),
with voxel spacings 0.8/1.7 respectively. Three views use 3 detector samples
in 2D or a (3,2) detector in 3D, with unequal u/v pitches in matrix checks.
The coupled surface has nonzero u*v terms in all three local offsets.
Both operators are compared separately with an independent float64 per-cell
slab oracle (not Siddon traversal); rtol=3e-5, atol=3e-6. Six flat cases
compare both operators with their existing default branch at these grids.

Three endpoint VJP cases use spacing 1.3, all point coordinates, nonuniform
image/cotangent values and independent float64 central differences at h and
h/2 (h=1e-4*spacing). Perturbed rays must keep their intersected cells, and
the differences must agree; VJP tolerance is rtol=2e-4, atol=2e-5.
Excluded: PyTorch dispatch/autograd, callback sampling and validation,
geometry second derivatives, hardware atomics/fastmath and multi-GPU.
The private endpoint interfaces may initially be INTERFACE_PENDING.
"""

import inspect
import math

import pytest
import sympy as sp

from test_cudasim import _matrix_geometry, _shape, _slab_length


BEAM_SURFACES = (("parallel", "arc"), ("fan", "arc"),
                 ("cone", "arc"), ("cone", "coupled"))
RTOL, ATOL = 3e-5, 3e-6


def test_PROOF_clockwise_2d_normal_and_local_recovery():
    """Unit tangent (cos(t),sin(t)); n=(u_y,-u_x); arbitrary real offsets."""
    t, a, c = sp.symbols("t a c", real=True)
    tangent = sp.Matrix([sp.cos(t), sp.sin(t)])
    normal = sp.Matrix([tangent[1], -tangent[0]])
    displacement = a * tangent + c * normal
    assert sp.trigsimp(tangent.dot(normal)) == 0
    assert sp.trigsimp(normal.dot(normal) - 1) == 0
    assert sp.trigsimp(displacement.dot(tangent) - a) == 0
    assert sp.trigsimp(displacement.dot(normal) - c) == 0


def test_PROOF_3d_cross_normal_and_flat_mapping():
    """Cross normal is unit if the real u/v vectors are unit and orthogonal."""
    u_axis = sp.Matrix(sp.symbols("ux uy uz", real=True))
    v_axis = sp.Matrix(sp.symbols("vx vy vz", real=True))
    centre = sp.Matrix(sp.symbols("cx cy cz", real=True))
    a, b, c = sp.symbols("a b c", real=True)
    normal = u_axis.cross(v_axis)
    assert sp.expand(u_axis.dot(normal)) == 0
    assert sp.expand(v_axis.dot(normal)) == 0
    assert sp.expand(normal.dot(normal) - u_axis.dot(u_axis) * v_axis.dot(v_axis)
                     + u_axis.dot(v_axis) ** 2) == 0
    point = centre + a * u_axis + b * v_axis + c * normal
    assert point.subs(c, 0) == centre + a * u_axis + b * v_axis


def test_PROOF_centered_grid_arc_circle_and_flat_limit():
    """R>0, real u/v: centred grid equivalence, circular section and R->infty."""
    i, count, pitch, u, v = sp.symbols("i count pitch u v", real=True)
    radius = sp.symbols("radius", positive=True)
    assert sp.expand((i + sp.Rational(1, 2) - count / 2) * pitch
                     - (i - (count - 1) / 2) * pitch) == 0
    arc_u = radius * sp.sin(u / radius)
    arc_n = radius * (1 - sp.cos(u / radius))
    assert sp.trigsimp(arc_u ** 2 + (arc_n - radius) ** 2 - radius ** 2) == 0
    assert sp.trigsimp(sp.diff(arc_u, u) ** 2 + sp.diff(arc_n, u) ** 2 - 1) == 0
    assert sp.Matrix([sp.limit(arc_u, radius, sp.oo), v,
                      sp.limit(arc_n, radius, sp.oo)]) == sp.Matrix([u, v, 0])


def test_PROOF_world_point_vjp_and_moving_frame_chain_rule():
    """Exact differential for arbitrary real C,U,V and local offsets (a,b,c)."""
    centre = sp.Matrix(sp.symbols("cx cy cz", real=True))
    u_axis = sp.Matrix(sp.symbols("ux uy uz", real=True))
    v_axis = sp.Matrix(sp.symbols("vx vy vz", real=True))
    gradient = sp.Matrix(sp.symbols("gx gy gz", real=True))
    dc = sp.Matrix(sp.symbols("dcx dcy dcz", real=True))
    du = sp.Matrix(sp.symbols("dux duy duz", real=True))
    dv = sp.Matrix(sp.symbols("dvx dvy dvz", real=True))
    a, b, c, da, db, dn, t = sp.symbols("a b c da db dn t", real=True)
    normal = u_axis.cross(v_axis)
    point = centre + a * u_axis + b * v_axis + c * normal
    actual_local = sp.Matrix([sp.diff(gradient.dot(point), q) for q in (a, b, c)])
    expected_local = sp.Matrix([gradient.dot(axis) for axis in (u_axis, v_axis, normal)])
    assert actual_local == expected_local
    moving = (centre + t * dc + (a + t * da) * (u_axis + t * du)
              + (b + t * db) * (v_axis + t * dv)
              + (c + t * dn) * (u_axis + t * du).cross(v_axis + t * dv))
    differential = (dc + da * u_axis + a * du + db * v_axis + b * dv
                    + dn * normal + c * (du.cross(v_axis) + u_axis.cross(dv)))
    assert sp.expand(sp.diff(gradient.dot(moving), t).subs(t, 0)
                     - gradient.dot(differential)) == 0


def test_PROOF_arc_radius_chain_rule():
    """R>0; parameter u and frame fixed; local cotangent (g_u,0,g_n)."""
    u, gu, gn = sp.symbols("u gu gn", real=True)
    radius = sp.symbols("radius", positive=True)
    theta = u / radius
    loss = gu * radius * sp.sin(theta) + gn * radius * (1 - sp.cos(theta))
    expected = (gu * (sp.sin(theta) - theta * sp.cos(theta))
                + gn * (1 - sp.cos(theta) - theta * sp.sin(theta)))
    assert sp.simplify(sp.diff(loss, radius) - expected) == 0


def _world_points(beam, geometry, detectors, pitches, spacing, surface, per_view):
    """Sample the frozen local-coordinate contract, independent of production."""
    import numpy as np

    du, dv = pitches
    u = (np.arange(detectors[0], dtype=np.float64) + .5 - detectors[0] / 2) * du
    if beam == "cone":
        v = (np.arange(detectors[1], dtype=np.float64) + .5 - detectors[1] / 2) * dv
        u, v = np.meshgrid(u, v, indexing="ij")
    else:
        v = np.zeros_like(u)
    points = []
    for view in range(len(geometry[0])):
        variant = view if per_view else 0
        radius = (1.25 + .31 * variant) * spacing
        if surface == "flat":
            a, b, c = u, v, np.zeros_like(u)
        elif surface == "arc":
            a = radius * np.sin(u / radius) + .09 * variant * spacing
            b = v + (.04 * variant * spacing if beam == "cone" else 0)
            c = radius * (1 - np.cos(u / radius)) + .07 * variant * spacing
        else:
            strength = 1 + .23 * variant
            a = u + strength * (.27 * u * v + .13 * v * v) / spacing
            b = v + strength * (.19 * u * v + .11 * u * u) / spacing
            c = strength * (.34 * u * u + .21 * u * v + .17 * v * v) / spacing
        centre, u_axis = geometry[1][view].astype(np.float64), geometry[2][view].astype(np.float64)
        if beam == "cone":
            v_axis = geometry[3][view].astype(np.float64)
            normal = np.cross(u_axis, v_axis)
            point = centre + a[..., None] * u_axis + b[..., None] * v_axis + c[..., None] * normal
        else:
            normal = np.asarray([u_axis[1], -u_axis[0]])
            point = centre + a[..., None] * u_axis + c[..., None] * normal
        points.append(point)
    return np.asarray(points, dtype=np.float32)


def _cell_matrix(beam, shape, spacing, first, points):
    """Physical float64 intersections with each cell; no traversal or kernel use."""
    import numpy as np

    xyz_shape = shape if beam == "cone" else shape[::-1]
    half = np.asarray(xyz_shape, dtype=np.float64) * spacing / 2
    points = points.astype(np.float64)
    first = first.astype(np.float64)
    sino_shape = points.shape[:-1]
    matrix = np.zeros((math.prod(sino_shape), math.prod(shape)), dtype=np.float64)
    for ray_index in np.ndindex(sino_shape):
        view = ray_index[0]
        origin = points[ray_index] if beam == "parallel" else first[view]
        direction = first[view] if beam == "parallel" else points[ray_index] - origin
        row = np.ravel_multi_index(ray_index, sino_shape)
        for index in np.ndindex(shape):
            xyz_index = index if beam == "cone" else index[::-1]
            lower = np.asarray(xyz_index, dtype=np.float64) * spacing - half
            column = np.ravel_multi_index(index, shape)
            matrix[row, column] = _slab_length(origin, direction, lower, lower + spacing,
                                               beam != "parallel")
    return matrix


def _dispatch(kernel, grid, threads, args, points=None, point_gradient=None):
    """CUDASIM accepts positional launch arguments only; additions are final."""
    if points is not None:
        names = inspect.signature(kernel.fn).parameters
        required = ["d_detector_positions"]
        if point_gradient is not None:
            required.append("d_grad_detector_positions")
        for name in required:
            if name not in names:
                pytest.fail(f"INTERFACE_PENDING: {kernel.fn.__name__} lacks {name}", pytrace=False)
        args += (points,)
        if point_gradient is not None:
            args += (point_gradient,)
    kernel[grid, threads](*args)


def _launch(kernels, beam, image, sino, spacing, geometry, pitches, points=None,
            backward=False, point_gradient=None):
    """Execute actual forward, adjoint or geometry-VJP kernel on NumPy arrays."""
    import numpy as np

    f = np.float32
    du, dv = pitches
    if beam == "cone":
        nx, ny, nz = image.shape
        views, nu, nv = sino.shape
        tail = (f(du), f(dv), *geometry, f(nx / 2), f(ny / 2), f(nz / 2), f(spacing))
        args = ((sino, views, nu, nv, image, nx, ny, nz) if backward else
                (image, nx, ny, nz, sino, views, nu, nv))
        prefix, grid, threads = "_cone_3d", (nv, nu, views), (1, 1, 1)
    else:
        ny, nx = image.shape
        views, ndet = sino.shape
        tail = (f(du), *geometry, f(nx / 2), f(ny / 2), f(spacing))
        args = ((sino, views, ndet, image, nx, ny) if backward else
                (image, nx, ny, sino, views, ndet))
        prefix, grid, threads = f"_{beam}_2d", (views, ndet), (1, 1)
    suffix = "geometry_vjp" if point_gradient is not None else ("backward" if backward else "forward")
    if point_gradient is not None:
        tail += tuple(np.zeros_like(array) for array in geometry)
    _dispatch(getattr(kernels, f"{prefix}_{suffix}_kernel"), grid, threads,
              args + tail, points, point_gradient)


@pytest.mark.parametrize("beam,surface", BEAM_SURFACES)
@pytest.mark.parametrize("grid,spacing", [(0, .8), (1, 1.7)])
@pytest.mark.parametrize("per_view", [False, True], ids=["shared", "per_view"])
def test_BOUNDED_all_curved_endpoint_basis_matrices(simulated_kernels, beam, surface,
                                                    grid, spacing, per_view):
    """Every basis for both actual operators versus separate slab-oracle A/A^T."""
    import numpy as np

    shape = _shape(beam, grid)
    detectors = (3, 2) if beam == "cone" else (3,)
    geometry = _matrix_geometry(beam, spacing)
    pitches = (.67 * spacing, .43 * spacing)
    points = _world_points(beam, geometry, detectors, pitches, spacing, surface, per_view)
    reference = _cell_matrix(beam, shape, spacing, geometry[0], points)
    flat = _world_points(beam, geometry, detectors, pitches, spacing, "flat", False)
    flat_reference = _cell_matrix(beam, shape, spacing, geometry[0], flat)
    assert np.max(np.abs(reference - flat_reference)) > 1e-3 * spacing
    sino_shape = points.shape[:-1]
    forward = np.zeros_like(reference)
    adjoint = np.zeros_like(reference.T)
    for column in range(math.prod(shape)):
        image, sino = np.zeros(shape, dtype=np.float32), np.zeros(sino_shape, dtype=np.float32)
        image.flat[column] = 1
        _launch(simulated_kernels, beam, image, sino, spacing, geometry, pitches, points)
        forward[:, column] = sino.ravel()
    for column in range(math.prod(sino_shape)):
        image, sino = np.zeros(shape, dtype=np.float32), np.zeros(sino_shape, dtype=np.float32)
        sino.flat[column] = 1
        _launch(simulated_kernels, beam, image, sino, spacing, geometry, pitches, points, backward=True)
        adjoint[:, column] = image.ravel()
    np.testing.assert_allclose(forward, reference, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(adjoint, reference.T, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("beam", ["parallel", "fan", "cone"])
@pytest.mark.parametrize("grid,spacing", [(0, .8), (1, 1.7)])
def test_BOUNDED_flat_points_match_both_default_operators(simulated_kernels, beam, grid, spacing):
    """Bind the exact flat limit to both actual operators, with nonuniform data."""
    import numpy as np

    shape = _shape(beam, grid)
    detectors = (3, 2) if beam == "cone" else (3,)
    geometry = _matrix_geometry(beam, spacing)
    pitches = (.61 * spacing, .47 * spacing)
    points = _world_points(beam, geometry, detectors, pitches, spacing, "flat", False)
    image = np.linspace(-.6, 1.3, math.prod(shape), dtype=np.float32).reshape(shape)
    expected, actual = np.zeros(points.shape[:-1], dtype=np.float32), np.zeros(points.shape[:-1], dtype=np.float32)
    _launch(simulated_kernels, beam, image, expected, spacing, geometry, pitches)
    _launch(simulated_kernels, beam, image, actual, spacing, geometry, pitches, points)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
    sino = np.linspace(-.7, 1.1, expected.size, dtype=np.float32).reshape(expected.shape)
    expected_image, actual_image = np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
    _launch(simulated_kernels, beam, expected_image, sino, spacing, geometry, pitches, backward=True)
    _launch(simulated_kernels, beam, actual_image, sino, spacing, geometry, pitches, points, backward=True)
    np.testing.assert_allclose(actual_image, expected_image, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("beam", ["parallel", "fan", "cone"])
def test_BOUNDED_endpoint_vjp_against_independent_finite_differences(simulated_kernels, beam):
    """All endpoint coordinates, with unchanged cell support at both step sizes."""
    import numpy as np

    spacing = 1.3
    shape = _shape(beam, 0)
    detectors = (3, 2) if beam == "cone" else (3,)
    geometry = _matrix_geometry(beam, spacing)
    pitches = (.67 * spacing, .43 * spacing)
    surface = "coupled" if beam == "cone" else "arc"
    points = _world_points(beam, geometry, detectors, pitches, spacing, surface, True)
    image = np.linspace(-.5, 1.7, math.prod(shape), dtype=np.float32).reshape(shape)
    cotangent = np.linspace(-.9, 1.2, math.prod(points.shape[:-1]), dtype=np.float32).reshape(points.shape[:-1])
    base = _cell_matrix(beam, shape, spacing, geometry[0], points)
    expected = np.zeros_like(points, dtype=np.float64)
    image64, cotangent64 = image.ravel().astype(np.float64), cotangent.ravel().astype(np.float64)
    for index in np.ndindex(points.shape):
        estimates = []
        for step in (1e-4 * spacing, .5e-4 * spacing):
            plus, minus = points.astype(np.float64), points.astype(np.float64)
            plus[index] += step
            minus[index] -= step
            plus_matrix = _cell_matrix(beam, shape, spacing, geometry[0], plus)
            minus_matrix = _cell_matrix(beam, shape, spacing, geometry[0], minus)
            assert np.array_equal(plus_matrix > 0, base > 0), index
            assert np.array_equal(minus_matrix > 0, base > 0), index
            estimates.append(cotangent64 @ ((plus_matrix - minus_matrix) @ image64) / (2 * step))
        np.testing.assert_allclose(estimates[0], estimates[1], rtol=1e-5, atol=1e-6)
        expected[index] = estimates[1]
    assert np.max(np.abs(expected)) > 1e-3
    actual = np.zeros_like(points)
    _launch(simulated_kernels, beam, image, cotangent, spacing, geometry, pitches,
            points, point_gradient=actual)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
