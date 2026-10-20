"""Curved 3D parallel rays checked with independent Siddon plane crossings."""

import itertools
import math

import pytest
import torch

from diffct import ParameterizedSurface, Projector

_VOLUME_SHAPE = (3, 4, 5)
_VOXEL_SPACING = (1.17, .83, .61)
_DETECTOR_SHAPE = (4, 3)
_DETECTOR_SPACING = (.67, .59)


def _frames(*, override=False):
    angles = [.41, .77, 1.39] if override else [.29, .92, 1.57]
    tilts = [.13, -.24, .36] if override else [-.22, .19, .31]
    centers = [[-.23, .17, .11], [.19, -.13, .27], [-.11, .31, -.23]]
    shifts = [[.07, -.11, .09], [-.09, .17, -.13], [.13, -.07, .19]]
    directions, origins, axes_u, axes_v = [], [], [], []
    for view, (angle, tilt) in enumerate(zip(angles, tilts)):
        normal = [math.cos(angle) * math.cos(tilt),
                  math.sin(angle) * math.cos(tilt), math.sin(tilt)]
        u = [-math.sin(angle), math.cos(angle), 0.]
        v = [-math.cos(angle) * math.sin(tilt),
             -math.sin(angle) * math.sin(tilt), math.cos(tilt)]
        direction = [n + (.24 + .03 * view) * du - .17 * dv
                     for n, du, dv in zip(normal, u, v)]
        norm = math.sqrt(sum(value * value for value in direction))
        direction = [value / norm for value in direction]
        origin = [center + (shift if override else 0.) + (-3. + 3.5 * view) * ray
                  for center, shift, ray in zip(centers[view], shifts[view], direction)]
        directions.append(direction)
        origins.append(origin)
        axes_u.append(u)
        axes_v.append(v)
    return tuple(torch.tensor(value, dtype=torch.float64)
                 for value in (directions, origins, axes_u, axes_v))


def _surface(u, v, displacement=.21):
    return torch.stack((.94 * u + .06 * v,
                        v + displacement + .08 * u - .04 * v.square(),
                        .14 * u.square() + .09 * v.square() + .05 * u * v), dim=-1)


def _siddon_row(origin, direction):
    """Split an infinite line at voxel planes and assign each open interval."""
    shape_xyz = _VOLUME_SHAPE[::-1]
    pitch_xyz = _VOXEL_SPACING[::-1]
    planes = [[(index - size / 2) * pitch for index in range(size + 1)]
              for size, pitch in zip(shape_xyz, pitch_xyz)]
    row = [0.] * math.prod(_VOLUME_SHAPE)
    entry, exit = -math.inf, math.inf
    for point, ray, boundaries in zip(origin, direction, planes):
        if ray == 0.:
            if not boundaries[0] < point < boundaries[-1]:
                return row
            continue
        a = (boundaries[0] - point) / ray
        b = (boundaries[-1] - point) / ray
        entry, exit = max(entry, min(a, b)), min(exit, max(a, b))
    if exit <= entry:
        return row
    crossings = [entry, exit]
    for point, ray, boundaries in zip(origin, direction, planes):
        if ray != 0.:
            crossings.extend((plane - point) / ray for plane in boundaries[1:-1]
                             if entry < (plane - point) / ray < exit)
    crossings = sorted(set(crossings))
    ray_norm = math.sqrt(sum(value * value for value in direction))
    for start, stop in itertools.pairwise(crossings):
        middle = (start + stop) / 2
        cell = [math.floor((point + middle * ray - boundaries[0]) / pitch)
                for point, ray, boundaries, pitch in zip(origin, direction, planes, pitch_xyz)]
        x, y, z = cell
        assert all(0 <= index < size for index, size in zip(cell, shape_xyz))
        row[(z * shape_xyz[1] + y) * shape_xyz[0] + x] += (stop - start) * ray_norm
    return row


def _cell_matrix(frames, *, displacement=.21, flat=False):
    directions, centers, axes_u, axes_v = [value.detach().cpu().tolist() for value in frames]
    rows = []
    for view, iu, iv in itertools.product(range(len(directions)),
                                        range(_DETECTOR_SHAPE[0]), range(_DETECTOR_SHAPE[1])):
        a = (iu + .5 - _DETECTOR_SHAPE[0] / 2) * _DETECTOR_SPACING[0]
        b = (iv + .5 - _DETECTOR_SHAPE[1] / 2) * _DETECTOR_SPACING[1]
        # These scalar equations are independent of the Torch surface callback.
        local_u = a if flat else .94 * a + .06 * b
        local_v = b if flat else b + displacement + .08 * a - .04 * b * b
        local_n = 0. if flat else .14 * a * a + .09 * b * b + .05 * a * b
        u, v = axes_u[view], axes_v[view]
        normal = [u[1] * v[2] - u[2] * v[1],
                  u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        origin = [center + local_u * du + local_v * dv + local_n * dn
                  for center, du, dv, dn in zip(centers[view], u, v, normal)]
        rows.append(_siddon_row(origin, directions[view]))
    return torch.tensor(rows, dtype=torch.float64)


def test_siddon_oracle_has_hand_computed_axis_cells_and_diagonal_chord():
    # The line crosses the five x cells at z=1, y=2, each with width .61.
    axis = torch.tensor(_siddon_row([0., .11, .13], [2., 0., 0.]), dtype=torch.float64)
    expected = torch.zeros(_VOLUME_SHAPE, dtype=torch.float64)
    expected[1, 2, :] = .61
    torch.testing.assert_close(axis, expected.flatten(), rtol=0., atol=1e-14)
    # The centered diagonal exits through x=+-1.525, before the y/z faces.
    diagonal = sum(_siddon_row([0., 0., 0.], [1., 1., 1.]))
    assert diagonal == pytest.approx(3.05 * math.sqrt(3.), rel=1e-14, abs=1e-14)


@pytest.mark.parametrize("surface_kind", ["callback", "parameterized"])
def test_curved_parallel3d_override_matches_independent_cell_matrix_and_adjoint(surface_kind):
    default, replacement = _frames(), _frames(override=True)
    matrix, default_matrix = _cell_matrix(replacement), _cell_matrix(default)
    assert matrix.sum() > 1.
    assert (matrix - default_matrix).abs().max() > .05
    assert (matrix - _cell_matrix(replacement, displacement=0.)).abs().max() > .05
    assert (matrix - _cell_matrix(replacement, flat=True)).abs().max() > .05
    surface = _surface
    if surface_kind == "parameterized":
        surface = ParameterizedSurface(
            lambda u, v, ids, displacement: _surface(u, v, displacement),
            (torch.tensor(.21, dtype=torch.float64),),
        )
    operator = Projector(default, _VOLUME_SHAPE, _DETECTOR_SHAPE,
                         beam="parallel3d", detector_spacing=_DETECTOR_SPACING,
                         voxel_spacing=_VOXEL_SPACING, detector_surface=surface,
                         backend="torch", precision="float64", view_chunk_size=2,
                         detector_chunk_shape=(2, 2), volume_chunk_shape=(2, 3, 3))
    image = torch.sin(torch.arange(math.prod(_VOLUME_SHAPE), dtype=torch.float64) * .37 + .13)
    image = image.reshape(_VOLUME_SHAPE)
    projection_shape = (3, *_DETECTOR_SHAPE)
    rays = torch.cos(torch.arange(math.prod(projection_shape), dtype=torch.float64) * .43 + .29)
    rays = rays.reshape(projection_shape)

    projected = operator.project(image, trajectory=replacement)
    backprojected = operator.backproject(rays, trajectory=list(replacement))
    torch.testing.assert_close(projected.flatten(), matrix @ image.flatten(), rtol=2e-11, atol=2e-12)
    torch.testing.assert_close(backprojected.flatten(), matrix.T @ rays.flatten(), rtol=2e-11, atol=2e-12)
    torch.testing.assert_close((projected * rays).sum(), (image * backprojected).sum(),
                               rtol=2e-11, atol=2e-12)
    torch.testing.assert_close(operator.project(image).flatten(), default_matrix @ image.flatten(),
                               rtol=2e-11, atol=2e-12)
    torch.testing.assert_close(operator.backproject(rays).flatten(), default_matrix.T @ rays.flatten(),
                               rtol=2e-11, atol=2e-12)
