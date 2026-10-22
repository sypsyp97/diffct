"""Portable geometry VJPs against independent float64 voxel-box differences."""

from __future__ import annotations

from datetime import timedelta
import time

import numpy as np
import pytest
import torch
import torch.distributed as dist
from tests._multiprocessing import spawn

from diffct import ParameterizedSurface, Projector
from tests.test_detector_surfaces import _arc_offsets, _case, _finite_difference, _matrix, _numpy


BEAMS = ("parallel", "fan", "cone")
OPERATIONS = ("project", "backproject")


def _sample_arc(u, v, view_indices, radius, coefficients):
    coefficient = coefficients[view_indices].reshape(-1, *([1] * u.ndim))
    x = radius * torch.sin(u / radius) + coefficient * u.square()
    sag = radius * (1 - torch.cos(u / radius)) + torch.zeros_like(x)
    return torch.stack((x, v + torch.zeros_like(x), sag), dim=-1)


def _inputs(beam):
    case = _case(beam, views=5, learnable=True)
    radius = torch.tensor(3.7, dtype=torch.float64, requires_grad=True)
    coefficients = torch.linspace(.021, .053, case.views, dtype=torch.float64,
                                  requires_grad=True)
    generator = torch.Generator().manual_seed(416)
    image = torch.randn(case.shape, dtype=torch.float64, generator=generator)
    rays = torch.randn(case.sino_shape, dtype=torch.float64, generator=generator)
    return case, radius, coefficients, image, rays


def _change_positions_and_surface(case, radius, coefficients, step):
    # Update live position leaves after construction. Unit axes stay valid.
    with torch.no_grad():
        for index in ((1,) if case.beam == "parallel" else (0, 1)):
            case.trajectory[index][:, 0].add_(.019 * (step + 1))
            case.trajectory[index][:, 1].sub_(.011 * (step + 1))
        radius.add_(.013 * (step + 1))
        coefficients.add_(torch.linspace(.001, .003, case.views,
                                         dtype=torch.float64) * (step + 1))


def _operator(case, radius, coefficients, **options):
    return Projector(
        case.trajectory, case.shape, case.detector, beam=case.beam,
        detector_spacing=case.pitch if case.beam == "cone" else case.pitch[0],
        voxel_spacing=case.spacing, backend="torch", precision="float64",
        detector_surface=ParameterizedSurface(_sample_arc, (radius, coefficients)),
        volume_chunk_shape=(2,) * len(case.shape), view_chunk_size=2,
        detector_chunk_shape=(3,) * len(case.detector), **options,
    )


def _reference(case, radius, coefficients, image, rays):
    geometry = tuple(_numpy(value) for value in case.trajectory)
    radius_value, coefficient_values = float(radius.detach()), _numpy(coefficients)
    image_np, rays_np = _numpy(image).ravel(), _numpy(rays).ravel()

    def objective(frames=geometry, r=radius_value, c=coefficient_values):
        matrix = _matrix(case, _arc_offsets(case, np.full(case.views, r), c),
                         geometry=frames)
        return float(rays_np @ matrix @ image_np)

    gradients = []
    for component, value in enumerate(geometry):
        gradient = np.empty_like(value)
        for index in np.ndindex(value.shape):
            direction = np.zeros_like(value)
            direction[index] = 1

            def changed(perturbed):
                frames = list(geometry)
                frames[component] = perturbed
                return objective(frames=frames)

            gradient[index] = _finite_difference(changed, value, direction)
        gradients.append(gradient)
    gradients.append(np.asarray(_finite_difference(lambda r: objective(r=r), radius_value)))
    coefficient_gradient = np.empty_like(coefficient_values)
    for index in range(case.views):
        direction = np.zeros_like(coefficient_values)
        direction[index] = 1
        coefficient_gradient[index] = _finite_difference(
            lambda c: objective(c=c), coefficient_values, direction,
        )
    gradients.append(coefficient_gradient)
    assert all(np.isfinite(value).all() and np.max(np.abs(value)) > 1e-5
               for value in gradients), "uninformative geometry finite-difference fixture"
    matrix = _matrix(case, _arc_offsets(case, np.full(case.views, radius_value), coefficient_values),
                     geometry=geometry)
    return matrix, gradients


@pytest.fixture(scope="module")
def references():
    result = {}
    for beam in BEAMS:
        case, radius, coefficients, image, rays = _inputs(beam)
        for step, operation in enumerate(OPERATIONS):
            _change_positions_and_surface(case, radius, coefficients, step)
            result[beam, operation] = _reference(case, radius, coefficients, image, rays)
    return result


def _check_call(case, radius, coefficients, image, rays, operator, operation,
                reference, *, data_requires_grad):
    matrix_np, gradients = reference
    matrix = torch.from_numpy(matrix_np)
    volume_slice, view_slice = operator.volume_slice, operator.view_slice
    for parameter in (*case.trajectory, radius, coefficients):
        parameter.grad = None
    if operation == "project":
        data = image[volume_slice].clone().requires_grad_(data_requires_grad)
        weight = rays[view_slice]
        repetitions = operator.space_world_size
        expected = (matrix @ image.flatten()).reshape(case.sino_shape)[view_slice]
        expected_data_gradient = (matrix.T @ rays.flatten()).reshape(case.shape)[volume_slice]
    else:
        data = rays[view_slice].clone().requires_grad_(data_requires_grad)
        weight = image[volume_slice]
        repetitions = operator.view_world_size
        expected = (matrix.T @ rays.flatten()).reshape(case.shape)[volume_slice]
        expected_data_gradient = (matrix @ image.flatten()).reshape(case.sino_shape)[view_slice]
    actual = getattr(operator, operation)(data)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    assert actual.requires_grad, "portable output lost trainable geometry participation"
    # Replicated outputs count the same global objective exactly once.
    ((actual * weight).sum() / repetitions).backward()
    names = ("source/direction", "detector_center", "detector_u", "detector_v")[:len(case.trajectory)]
    for name, parameter, expected_gradient in zip(
        (*names, "surface_radius", "per_view_surface_coefficient"),
        (*case.trajectory, radius, coefficients), gradients,
    ):
        assert parameter.grad is not None, (
            f"{case.beam}/{operation} rank {operator.rank} lost {name} gradient"
        )
        assert torch.isfinite(parameter.grad).all()
        # Full global tensors include views unowned by this rank. Every rank
        # must receive one global SUM, including per-view surface parameters.
        torch.testing.assert_close(parameter.grad, torch.from_numpy(expected_gradient),
                                   rtol=3e-5, atol=3e-7)
    if data_requires_grad:
        torch.testing.assert_close(data.grad, expected_data_gradient, rtol=3e-11, atol=3e-12)
    else:
        assert data.grad is None


def test_single_cpu_geometry_and_surface_gradients_match_independent_finite_differences(references):
    for beam in BEAMS:
        case, radius, coefficients, image, rays = _inputs(beam)
        operator = _operator(case, radius, coefficients)
        for step, operation in enumerate(OPERATIONS):
            _change_positions_and_surface(case, radius, coefficients, step)
            _check_call(case, radius, coefficients, image, rays, operator, operation,
                        references[beam, operation], data_requires_grad=False)


def _distributed_worker(rank, rendezvous, partition, references):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=15))
    try:
        for beam in BEAMS:
            case, radius, coefficients, image, rays = _inputs(beam)
            options = dict(distributed=True, partition=partition)
            if partition == "hybrid":
                options["partition_grid"] = (2, 2)
            operator = _operator(case, radius, coefficients, **options)
            for step, operation in enumerate(OPERATIONS):
                _change_positions_and_surface(case, radius, coefficients, step)
                # All ranks backpropagate geometry; one rank's data is fixed.
                # Skipping its adjoint collectives must fail within 15 seconds.
                fixed_rank = 0 if operation == "project" else 3
                _check_call(case, radius, coefficients, image, rays, operator, operation,
                            references[beam, operation], data_requires_grad=rank != fixed_rank)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(),
                    reason="CPU distributed geometry requires Gloo")
@pytest.mark.parametrize("partition", ["views", "space", "hybrid"])
def test_four_rank_geometry_gradients_sum_once_with_mixed_data_participation(
    partition, references, tmp_path,
):
    rendezvous = (tmp_path / f"geometry-{partition}-rendezvous").resolve().as_uri()
    context = spawn(_distributed_worker, args=(rendezvous, partition, references),
                       nprocs=4, join=False)
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise AssertionError("four-process geometry backward exceeded 60 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
