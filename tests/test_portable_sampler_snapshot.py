"""Each portable forward owns its sampler and explicit parameter snapshots."""

from dataclasses import replace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from diffct import ParameterizedSurface, Projector
from tests.test_reference_backend import _Case, _matrix
from tests.test_spatial_partition import _run_processes


def _old_sampler(u, v, ids, parameter):
    return torch.stack((u + parameter, v, torch.zeros_like(u)), dim=-1)


def _new_sampler(u, v, ids, parameter):
    return torch.stack((u + 3 * parameter + .6, v, torch.zeros_like(u)), dim=-1)


def _case():
    frames = tuple(torch.tensor(value, dtype=torch.float64) for value in (
        [[-10., -3.], [-10., -3.]],
        [[8., 1.7], [8., 1.7]],
        [[0., 1.], [0., 1.]],
    ))
    return _Case("fan", frames, (3, 5), (1,), (1.,), 1.)


def _reference(case, parameter, scale, offset, image, rays):
    def matrix_at(value):
        source, center, axis = case.trajectory
        shifted = (source, center + (scale * value + offset) * axis, axis)
        return _matrix(replace(case, trajectory=shifted))

    step = 1e-5
    derivative = rays.flatten() @ (
        matrix_at(parameter + step) - matrix_at(parameter - step)
    ) @ image.flatten() / (2 * step)
    assert torch.isfinite(derivative) and derivative.abs() > 1e-3
    return matrix_at(parameter), derivative


def _call(operator, operation, image, rays, kwargs):
    if operation == "project":
        data = image[operator.volume_slice].clone().requires_grad_()
        weight, repetitions = rays[operator.view_slice], operator.space_world_size
    else:
        data = rays[operator.view_slice].clone().requires_grad_()
        weight, repetitions = image[operator.volume_slice], operator.view_world_size
    output = getattr(operator, operation)(data, **kwargs)
    return data, output, (output * weight).sum() / repetitions


def _checks(label, operation, operator, output, gradients, reference, image, rays):
    matrix, parameter_gradient = reference
    projection = (matrix @ image.flatten()).reshape(rays.shape)[operator.view_slice]
    backprojection = (matrix.T @ rays.flatten()).reshape(image.shape)[operator.volume_slice]
    expected_output, expected_data = (
        (projection, backprojection) if operation == "project"
        else (backprojection, projection)
    )
    return (
        (f"{label}/output", output, expected_output, 2e-11),
        (f"{label}/data-gradient", gradients[0], expected_data, 2e-11),
        (f"{label}/parameter-gradient", gradients[1], parameter_gradient, 3e-7),
    )


def _exercise(operation, *, distributed):
    checks = []
    for explicit in (False, True):
        for change in ("unchanged", "sampler", "sampler-and-parameters"):
            case = _case()
            parameter = torch.tensor(.11, dtype=torch.float64, requires_grad=True)
            surface = ParameterizedSurface(_old_sampler, (parameter,))
            operator = Projector(
                case.trajectory, case.shape, case.detector, beam=case.beam,
                detector_surface=surface, backend="torch", precision="float64",
                distributed=distributed,
            )
            kwargs = {}
            if explicit:
                frames = tuple(value.clone() for value in case.trajectory)
                frames[1][:, 1].add_(.04)
                case = replace(case, trajectory=frames)
                kwargs["trajectory"] = frames
            image = (torch.arange(15, dtype=torch.float64).sin() + .4).reshape(case.shape)
            rays = torch.ones(case.projection_shape, dtype=torch.float64)
            old_reference = _reference(case, .11, 1., 0., image, rays)
            old_data, old_output, old_loss = _call(operator, operation, image, rays, kwargs)

            if change != "unchanged":
                surface.sampler = _new_sampler
            if change == "sampler-and-parameters":
                # Both mutation of the original leaf and replacement of the
                # parameter tuple must leave the outstanding graph intact.
                with torch.no_grad():
                    parameter.add_(.08)
                surface.parameters = (torch.tensor(.27, dtype=torch.float64, requires_grad=True),)

            current_parameter = surface.parameters[0]
            scale, offset = (1., 0.) if change == "unchanged" else (3., .6)
            fresh_reference = _reference(
                case, current_parameter.item(), scale, offset, image, rays,
            )
            if change != "unchanged":
                assert (fresh_reference[0] - old_reference[0]).abs().max() > .1
            # Keep both graphs outstanding: a new invocation must not replace
            # the sampler or values owned by the earlier invocation.
            fresh_data, fresh_output, fresh_loss = _call(operator, operation, image, rays, kwargs)
            old_gradients = torch.autograd.grad(old_loss, (old_data, parameter))
            fresh_gradients = torch.autograd.grad(fresh_loss, (fresh_data, current_parameter))
            label = f"{operation}/{'explicit' if explicit else 'default'}/{change}"
            checks.extend(_checks(f"{label}/old", operation, operator, old_output,
                                  old_gradients, old_reference, image, rays))
            checks.extend(_checks(f"{label}/fresh", operation, operator, fresh_output,
                                  fresh_gradients, fresh_reference, image, rays))
    return checks


def _assert_checks(checks):
    failures = []
    for label, actual, expected, tolerance in checks:
        try:
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        except AssertionError:
            failures.append(f"{label}: max error {(actual - expected).abs().max().item():.12g}")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("operation", ("project", "backproject"))
def test_single_cpu_forward_captures_sampler_and_parameter_values(operation):
    _assert_checks(_exercise(operation, distributed=False))


def _worker(rank, rendezvous, operation):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=20))
    try:
        checks = _exercise(operation, distributed=True)
        # All numerical calls finish before assertions; verify that the same
        # process group remains usable after both graphs have backpropagated.
        probe = torch.tensor(rank + 1)
        dist.all_reduce(probe)
        assert probe.item() == 3
        _assert_checks(checks)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(),
                    reason="sampler snapshot regression requires two Gloo ranks")
@pytest.mark.parametrize("operation", ("project", "backproject"))
def test_two_rank_forward_captures_sampler_and_parameter_values(operation, tmp_path):
    rendezvous = (tmp_path / "sampler-snapshot-rendezvous").resolve().as_uri()
    _run_processes(_worker, (rendezvous, operation), timeout=60)
