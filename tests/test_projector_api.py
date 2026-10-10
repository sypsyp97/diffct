"""Behavioral tests for the high-level arbitrary-trajectory ``Projector``.

The declaration-only operator is intentionally exercised through the public API
and compared with the existing low-level autograd functions.  CUDA tests are
small so they cover argument plumbing and collectives without becoming another
kernel accuracy suite.
"""

from __future__ import annotations

import math
from contextlib import ExitStack, contextmanager
from unittest import mock

import pytest
import torch
from numba import cuda as numba_cuda

import diffct.projectors as projectors_module

from diffct import (
    ConeBackprojectorFunction,
    ConeProjectorFunction,
    FanBackprojectorFunction,
    FanProjectorFunction,
    ParallelBackprojectorFunction,
    ParallelProjectorFunction,
    Projector,
    circular_trajectory_2d_fan,
    circular_trajectory_2d_parallel,
    spiral_trajectory_3d,
)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Projector execution")


def _cpu_parallel_trajectory(n_views: int = 4):
    ray_dir = torch.tensor([[1.0, 0.0]] * n_views)
    det_origin = torch.zeros(n_views, 2)
    det_u_vec = torch.tensor([[0.0, 1.0]] * n_views)
    return ray_dir, det_origin, det_u_vec


def _cpu_fan_trajectory(n_views: int = 4):
    src_pos = torch.tensor([[-20.0, 0.0]] * n_views)
    det_center = torch.tensor([[20.0, 0.0]] * n_views)
    det_u_vec = torch.tensor([[0.0, 1.0]] * n_views)
    return src_pos, det_center, det_u_vec


def _cpu_cone_trajectory(n_views: int = 4):
    src_pos = torch.tensor([[-20.0, 0.0, 0.0]] * n_views)
    det_center = torch.tensor([[20.0, 0.0, 0.0]] * n_views)
    det_u_vec = torch.tensor([[0.0, 1.0, 0.0]] * n_views)
    det_v_vec = torch.tensor([[0.0, 0.0, 1.0]] * n_views)
    return src_pos, det_center, det_u_vec, det_v_vec


def _base_constructor_args(beam: str = "parallel"):
    if beam == "parallel":
        return _cpu_parallel_trajectory(), (6, 7), 5
    if beam == "fan":
        return _cpu_fan_trajectory(), (6, 7), (5,)
    return _cpu_cone_trajectory(), (4, 6, 7), (5, 3)


@pytest.mark.parametrize(
    ("beam", "overrides"),
    [
        ("parallel", {"beam": "invalid"}),
        ("parallel", {"volume_shape": (6,)}),
        ("parallel", {"detector_shape": (2, 3)}),
        ("parallel", {"detector_spacing": 0.0}),
        ("parallel", {"detector_spacing": -1.0}),
        ("parallel", {"voxel_spacing": 0.0}),
        ("parallel", {"voxel_spacing": (1.0, 2.0)}),
        ("fan", {"detector_shape": (5, 3)}),
        ("cone", {"detector_shape": (5,)}),
        ("cone", {"detector_spacing": (1.0, 0.0)}),
        ("cone", {"detector_spacing": (1.0, -1.0)}),
        ("cone", {"voxel_spacing": (1.0, 1.0)}),
        ("parallel", {"devices": []}),
        ("parallel", {"devices": ["cpu"]}),
        (
            "parallel",
            {"devices": [torch.device("cuda:0"), torch.device("cuda:0")]},
        ),
    ],
)
def test_constructor_rejects_invalid_public_arguments(beam, overrides):
    """Invalid configuration fails before any CUDA work is attempted."""
    trajectory, volume_shape, detector_shape = _base_constructor_args(beam)
    kwargs = {
        "beam": beam,
        "volume_shape": volume_shape,
        "detector_shape": detector_shape,
    }
    kwargs.update(overrides)
    with pytest.raises((TypeError, ValueError)):
        Projector(trajectory, **kwargs)


@pytest.mark.parametrize("beam", ["parallel", "fan", "cone"])
def test_constructor_rejects_malformed_or_nonfinite_geometry(beam):
    trajectory, volume_shape, detector_shape = _base_constructor_args(beam)

    mismatched = list(trajectory)
    mismatched[-1] = mismatched[-1][:-1]
    with pytest.raises((TypeError, ValueError)):
        Projector(
            tuple(mismatched),
            volume_shape,
            detector_shape,
            beam=beam,
        )

    nonfinite = list(trajectory)
    nonfinite[0] = nonfinite[0].clone()
    nonfinite[0][0, 0] = math.nan
    with pytest.raises((TypeError, ValueError)):
        Projector(
            tuple(nonfinite),
            volume_shape,
            detector_shape,
            beam=beam,
        )

    empty = tuple(component[:0] for component in trajectory)
    with pytest.raises((TypeError, ValueError)):
        Projector(empty, volume_shape, detector_shape, beam=beam)

    integer_geometry = tuple(component.to(torch.int64) for component in trajectory)
    with pytest.raises((TypeError, ValueError)):
        Projector(integer_geometry, volume_shape, detector_shape, beam=beam)


def test_constructor_rejects_nonorthogonal_or_nonunit_axes():
    parallel = list(_cpu_parallel_trajectory())
    parallel[0] = parallel[0].clone()
    parallel[0][0] = torch.tensor([2.0, 0.0])
    with pytest.raises((TypeError, ValueError)):
        Projector(tuple(parallel), (6, 7), 5, beam="parallel")

    fan = list(_cpu_fan_trajectory())
    fan[2] = fan[2].clone()
    fan[2][0] = torch.tensor([0.0, 2.0])
    with pytest.raises((TypeError, ValueError)):
        Projector(tuple(fan), (6, 7), 5, beam="fan")

    parallel = list(_cpu_parallel_trajectory())
    parallel[2] = parallel[2].clone()
    parallel[2][0] = torch.tensor([1.0, 0.0])
    with pytest.raises((TypeError, ValueError)):
        Projector(tuple(parallel), (6, 7), 5, beam="parallel")

    cone = list(_cpu_cone_trajectory())
    cone[3] = cone[3].clone()
    cone[3][0] = cone[2][0]
    with pytest.raises((TypeError, ValueError)):
        Projector(tuple(cone), (4, 6, 7), (5, 3), beam="cone")


def test_process_group_requires_distributed_mode():
    trajectory, volume_shape, detector_shape = _base_constructor_args()
    with pytest.raises((TypeError, ValueError)):
        Projector(
            trajectory,
            volume_shape,
            detector_shape,
            beam="parallel",
            process_group=object(),
        )


def test_distributed_mode_requires_an_initialized_process_group():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        pytest.skip("This process already has a process group")
    trajectory, volume_shape, detector_shape = _base_constructor_args()
    with pytest.raises((RuntimeError, TypeError, ValueError)) as error:
        Projector(
            trajectory,
            volume_shape,
            detector_shape,
            beam="parallel",
            distributed=True,
        )
    # INTERFACE_PENDING: the declaration-only scaffold raises this sentinel;
    # do not let that RuntimeError subclass count as validation evidence.
    assert not isinstance(error.value, NotImplementedError)


@pytest.mark.cuda
def test_projector_default_cpu_input_matches_cuda_forward_and_adjoint():
    _require_cuda()
    device = torch.device("cuda", torch.cuda.current_device())
    trajectory = tuple(t.to(device) for t in _cpu_parallel_trajectory(3))
    projector = Projector(trajectory, (6, 7), 5, beam="parallel")
    image = torch.linspace(-.3, 1.1, 42, dtype=torch.float64).reshape(6, 7)
    sino = torch.linspace(-.7, .9, 15, dtype=torch.float64).reshape(3, 5)
    projected = projector.project(image)
    back = projector.backproject(sino)
    assert projected.device.type == back.device.type == "cpu"
    assert projected.dtype == back.dtype == torch.float32
    torch.testing.assert_close(projected, projector.project(image.to(device)).cpu(), rtol=4e-5, atol=5e-5)
    torch.testing.assert_close(back, projector.backproject(sino.to(device)).cpu(), rtol=4e-5, atol=5e-5)


@pytest.mark.parametrize("beam", ["parallel", "fan", "cone"])
def test_projector_accepts_geometry_gradients(beam):
    trajectory, volume_shape, detector_shape = _base_constructor_args(beam)
    trajectory = tuple(component.clone().requires_grad_() for component in trajectory)
    Projector(trajectory, volume_shape, detector_shape, beam=beam)


def _cuda_case(
    beam: str, device: torch.device, n_views: int = 5, make_projector: bool = True
):
    if beam == "parallel":
        trajectory = circular_trajectory_2d_parallel(n_views, device=device)
        volume_base = torch.randn(7, 9, device=device)
        volume = volume_base.transpose(0, 1)
        detector_shape = 7
        detector_spacing = 1.25
        raw_project = lambda value: ParallelProjectorFunction.apply(
            value,
            *trajectory,
            detector_shape,
            detector_spacing,
            0.75,
        )
        raw_backproject = lambda sino: ParallelBackprojectorFunction.apply(
            sino,
            *trajectory,
            detector_spacing,
            *volume.shape,
            0.75,
        )
    elif beam == "fan":
        trajectory = circular_trajectory_2d_fan(
            n_views, sid=30.0, sdd=45.0, device=device
        )
        volume_base = torch.randn(7, 9, device=device)
        volume = volume_base.transpose(0, 1)
        detector_shape = (7,)
        detector_spacing = 1.25
        raw_project = lambda value: FanProjectorFunction.apply(
            value,
            *trajectory,
            detector_shape[0],
            detector_spacing,
            0.75,
        )
        raw_backproject = lambda sino: FanBackprojectorFunction.apply(
            sino,
            *trajectory,
            detector_spacing,
            *volume.shape,
            0.75,
        )
    else:
        trajectory = spiral_trajectory_3d(
            n_views,
            sid=30.0,
            sdd=45.0,
            z_range=2.0,
            n_turns=1.25,
            device=device,
        )
        volume_base = torch.randn(9, 7, 5, device=device)
        volume = volume_base.permute(2, 1, 0)
        detector_shape = (6, 4)
        detector_spacing = (1.25, 0.8)
        raw_project = lambda value: ConeProjectorFunction.apply(
            value,
            *trajectory,
            detector_shape[0],
            detector_shape[1],
            *detector_spacing,
            0.75,
        )
        raw_backproject = lambda sino: ConeBackprojectorFunction.apply(
            sino,
            *trajectory,
            *volume.shape,
            *detector_spacing,
            0.75,
        )

    projector = None
    if make_projector:
        projector = Projector(
            trajectory,
            tuple(volume.shape),
            detector_shape,
            beam=beam,
            detector_spacing=detector_spacing,
            voxel_spacing=0.75,
        )
    return projector, volume, raw_project, raw_backproject, trajectory, detector_shape


@pytest.mark.cuda
@pytest.mark.parametrize("beam", ["parallel", "fan", "cone"])
def test_projector_matches_lowlevel_forward_and_backproject(beam):
    _require_cuda()
    device = torch.device("cuda", torch.cuda.current_device())
    projector, volume, raw_project, raw_backproject, trajectory, detector_shape = _cuda_case(
        beam, device
    )

    projection = projector.project(volume)
    expected_projection = raw_project(volume)
    torch.testing.assert_close(projection, expected_projection, rtol=1e-5, atol=1e-5)
    assert projection.device == volume.device
    expected_shape = (5, detector_shape) if isinstance(detector_shape, int) else (5, *detector_shape)
    assert projection.shape == expected_shape

    explicit_device_projector = Projector(
        trajectory,
        tuple(volume.shape),
        detector_shape,
        beam=beam,
        detector_spacing=1.25 if beam != "cone" else (1.25, 0.8),
        voxel_spacing=0.75,
        devices=[device],
    )
    torch.testing.assert_close(
        explicit_device_projector.project(volume),
        expected_projection,
        rtol=1e-5,
        atol=1e-5,
    )

    if beam == "cone":
        scalar_spacing_projector = Projector(
            trajectory,
            tuple(volume.shape),
            detector_shape,
            beam=beam,
            detector_spacing=1.25,
            voxel_spacing=0.75,
        )
        scalar_spacing_expected = ConeProjectorFunction.apply(
            volume,
            *trajectory,
            detector_shape[0],
            detector_shape[1],
            1.25,
            1.25,
            0.75,
        )
        torch.testing.assert_close(
            scalar_spacing_projector.project(volume),
            scalar_spacing_expected,
            rtol=1e-5,
            atol=1e-5,
        )

    # Noncontiguous sinograms exercise the same input route in the adjoint.
    if projection.ndim == 2:
        sino = torch.randn(
            projection.shape[1], projection.shape[0], device=device
        ).transpose(0, 1)
    else:
        sino = torch.randn(
            projection.shape[2], projection.shape[1], projection.shape[0], device=device
        ).permute(2, 1, 0)
    backprojection = projector.backproject(sino)
    expected_backprojection = raw_backproject(sino)
    torch.testing.assert_close(
        backprojection, expected_backprojection, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(projector(volume), projection, rtol=1e-5, atol=1e-5)


@pytest.mark.cuda
@pytest.mark.parametrize("beam", ["parallel", "fan", "cone"])
def test_projector_adjoint_and_both_autograd_directions(beam):
    _require_cuda()
    device = torch.device("cuda", torch.cuda.current_device())
    projector, volume, _, _, _, _ = _cuda_case(beam, device, n_views=4)
    probe_volume = torch.randn_like(volume).contiguous()
    projection = projector.project(probe_volume)
    probe_sino = torch.randn_like(projection).contiguous()

    lhs = torch.sum(projection * probe_sino)
    rhs = torch.sum(probe_volume * projector.backproject(probe_sino))
    torch.testing.assert_close(lhs, rhs, rtol=5e-3, atol=5e-4)

    project_input = probe_volume.detach().clone().requires_grad_()
    project_grad = torch.autograd.grad(
        (projector.project(project_input) * probe_sino).sum(), project_input
    )[0]
    torch.testing.assert_close(
        project_grad,
        projector.backproject(probe_sino),
        rtol=5e-3,
        atol=5e-4,
    )

    sino_input = probe_sino.detach().clone().requires_grad_()
    backproject_grad = torch.autograd.grad(
        (projector.backproject(sino_input) * probe_volume).sum(), sino_input
    )[0]
    torch.testing.assert_close(
        backproject_grad,
        projector.project(probe_volume),
        rtol=5e-3,
        atol=5e-4,
    )


@pytest.mark.cuda
def test_projector_metadata_and_current_stream():
    _require_cuda()
    device = torch.device("cuda", torch.cuda.current_device())
    projector, volume, raw_project, _, _, detector_shape = _cuda_case(
        "cone", device, n_views=5
    )

    # INTERFACE_PENDING: declaration-only scaffold has not exposed these
    # metadata properties yet; the values define the frozen public contract.
    assert projector.rank == 0
    assert projector.world_size == 1
    assert projector.view_slice == slice(0, 5)
    assert projector.projection_shape == (5, *detector_shape)

    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        projected = projector.project(volume)
    stream.synchronize()
    torch.testing.assert_close(projected, raw_project(volume), rtol=1e-5, atol=1e-5)


def _multi_gpu_devices():
    if torch.cuda.device_count() < 2:
        pytest.skip("at least two CUDA devices are required")
    # Keep the input on cuda:1 while the caller's current device stays cuda:0.
    return [torch.device("cuda:0"), torch.device("cuda:1")]


_KERNEL_SPECS = (
    ("_parallel_2d_forward_kernel", "parallel", "forward", 4),
    ("_parallel_2d_backward_kernel", "parallel", "backward", 1),
    ("_fan_2d_forward_kernel", "fan", "forward", 4),
    ("_fan_2d_backward_kernel", "fan", "backward", 1),
    ("_cone_3d_forward_kernel", "cone", "forward", 5),
    ("_cone_3d_backward_kernel", "cone", "backward", 1),
)


def _stream_handle(stream):
    return int(stream.cuda_stream)


class _KernelLaunchProxy:
    def __init__(self, kernel, name, beam, kind, views_arg, tracer):
        self.kernel = kernel
        self.name = name
        self.beam = beam
        self.kind = kind
        self.views_arg = views_arg
        self.tracer = tracer

    def __getitem__(self, launch_config):
        configured_kernel = self.kernel[launch_config]
        stream = launch_config[2]
        handle = getattr(stream, "handle", stream)
        stream_handle = int(getattr(handle, "value", handle))

        def launch(*args, **kwargs):
            torch_device = torch.cuda.current_device()
            numba_device = numba_cuda.current_context().device.id
            n_views = int(args[self.views_arg])
            result = configured_kernel(*args, **kwargs)
            self.tracer.records.append(
                {
                    "phase": self.tracer.phase,
                    "stream_handle": stream_handle,
                    "torch_device": torch_device,
                    "numba_device": numba_device,
                    "beam": self.beam,
                    "kernel": self.name,
                    "kernel_kind": self.kind,
                    "n_views": n_views,
                }
            )
            return result

        return launch


class _KernelLaunchTracer:
    def __init__(self):
        self.records = []
        self.phase = None
        self._patches = None

    def __enter__(self):
        self._patches = ExitStack()
        for name, beam, kind, views_arg in _KERNEL_SPECS:
            kernel = getattr(projectors_module, name)
            proxy = _KernelLaunchProxy(
                kernel, name, beam, kind, views_arg, self
            )
            self._patches.enter_context(
                mock.patch.object(projectors_module, name, proxy)
            )
        return self

    def __exit__(self, *exc_info):
        return self._patches.__exit__(*exc_info)

    @contextmanager
    def in_phase(self, phase):
        previous = self.phase
        self.phase = phase
        try:
            yield
        finally:
            self.phase = previous


def _expected_view_counts(n_views, devices):
    base, remainder = divmod(n_views, len(devices))
    return {
        device.index: base + int(rank < remainder)
        for rank, device in enumerate(devices)
        if base + int(rank < remainder) > 0
    }


def _launch_phase_issues(
    records, phase, beam, kernel_kind, expected_view_counts, expected_streams
):
    observed = [record for record in records if record["phase"] == phase]
    expected = sorted(expected_view_counts.items())
    actual = sorted(
        (record["torch_device"], record["n_views"]) for record in observed
    )
    issues = []
    if actual != expected:
        issues.append(f"{phase}: observed shards {actual}, expected {expected}")
    for record in observed:
        if record["beam"] != beam or record["kernel_kind"] != kernel_kind:
            issues.append(
                f"{phase}: observed {record['beam']} {record['kernel_kind']} "
                f"{record['kernel']}, expected {beam} {kernel_kind} kernel"
            )
        if record["torch_device"] != record["numba_device"]:
            issues.append(
                f"{phase}: Torch device {record['torch_device']} and Numba "
                f"device {record['numba_device']} disagree"
            )
        expected_stream = expected_streams.get(record["torch_device"])
        if record["stream_handle"] != expected_stream:
            issues.append(
                f"{phase}: {record['kernel']} on cuda:{record['torch_device']} "
                f"launched on stream {record['stream_handle']}, expected selected "
                f"stream {expected_stream}"
            )
    return issues


def _assert_launch_phases(records, expectations):
    issues = []
    for phase, beam, kind, view_counts, streams in expectations:
        issues.extend(
            _launch_phase_issues(records, phase, beam, kind, view_counts, streams)
        )
    assert not issues, "CUDA kernel launch observations:\n" + "\n".join(issues)


def _multi_gpu_projector(beam, devices, n_views):
    input_device = devices[1]
    with torch.cuda.device(input_device):
        torch.manual_seed(100 + n_views)
        _, volume, raw_project, raw_backproject, trajectory, detector_shape = _cuda_case(
            beam, input_device, n_views=n_views, make_projector=False
        )
    projector = Projector(
        trajectory,
        tuple(volume.shape),
        detector_shape,
        beam=beam,
        detector_spacing=(1.25, 0.8) if beam == "cone" else 1.25,
        voxel_spacing=0.75,
        devices=devices,
    )
    return projector, volume, raw_project, raw_backproject


@pytest.mark.cuda
@pytest.mark.parametrize("beam", ["parallel", "fan", "cone"])
@pytest.mark.parametrize("n_views", [5, 1])
def test_multigpu_numeric_parity_gradients_and_real_device_calls(beam, n_views):
    _require_cuda()
    devices = _multi_gpu_devices()
    current_device, input_device = devices
    projector, volume, raw_project, raw_backproject = _multi_gpu_projector(
        beam, devices, n_views
    )
    expected_views = _expected_view_counts(n_views, devices)

    with torch.cuda.device(current_device):
        assert torch.cuda.current_device() == current_device.index
        sinogram_shape = (n_views, *projector.projection_shape[1:])
        sinogram = torch.linspace(
            -0.2,
            0.2,
            steps=math.prod(sinogram_shape),
            device=input_device,
        ).reshape(sinogram_shape)
        image_weight = torch.linspace(
            -0.15,
            0.15,
            steps=math.prod(projector.projection_shape),
            device=input_device,
        ).reshape(projector.projection_shape)
        image_input = volume.detach().clone().requires_grad_()
        sino_input = sinogram.detach().clone().requires_grad_()
        sino_weight = torch.linspace(
            -0.1,
            0.1,
            steps=volume.numel(),
            device=input_device,
        ).reshape(volume.shape)
        torch.cuda.synchronize(input_device)
        stream = torch.cuda.Stream(device=input_device)
        selected_streams = {
            device.index: _stream_handle(torch.cuda.current_stream(device=device))
            for device in devices
        }
        selected_streams[input_device.index] = _stream_handle(stream)
        with _KernelLaunchTracer() as tracer:
            with torch.cuda.stream(stream):
                with tracer.in_phase("project_forward"):
                    torch.cuda.set_device(current_device.index)
                    assert torch.cuda.current_device() == current_device.index
                    assert (
                        _stream_handle(torch.cuda.current_stream(input_device))
                        == _stream_handle(stream)
                    )
                    projection = projector.project(volume)
                    assert torch.cuda.current_device() == current_device.index

                with tracer.in_phase("backproject_forward"):
                    torch.cuda.set_device(current_device.index)
                    assert torch.cuda.current_device() == current_device.index
                    assert (
                        _stream_handle(torch.cuda.current_stream(input_device))
                        == _stream_handle(stream)
                    )
                    backprojection = projector.backproject(sinogram)
                    assert torch.cuda.current_device() == current_device.index

                with tracer.in_phase("project_gradient_forward"):
                    torch.cuda.set_device(current_device.index)
                    assert torch.cuda.current_device() == current_device.index
                    assert (
                        _stream_handle(torch.cuda.current_stream(input_device))
                        == _stream_handle(stream)
                    )
                    gradient_projection = projector.project(image_input)
                    assert torch.cuda.current_device() == current_device.index
                project_loss = (gradient_projection * image_weight).sum()
                with tracer.in_phase("project_backward"):
                    torch.cuda.set_device(current_device.index)
                    assert torch.cuda.current_device() == current_device.index
                    assert (
                        _stream_handle(torch.cuda.current_stream(input_device))
                        == _stream_handle(stream)
                    )
                    project_loss.backward()
                    assert torch.cuda.current_device() == current_device.index

                with tracer.in_phase("backproject_gradient_forward"):
                    torch.cuda.set_device(current_device.index)
                    assert torch.cuda.current_device() == current_device.index
                    assert (
                        _stream_handle(torch.cuda.current_stream(input_device))
                        == _stream_handle(stream)
                    )
                    gradient_backprojection = projector.backproject(sino_input)
                    assert torch.cuda.current_device() == current_device.index
                backproject_loss = (gradient_backprojection * sino_weight).sum()
                with tracer.in_phase("backproject_backward"):
                    torch.cuda.set_device(current_device.index)
                    assert torch.cuda.current_device() == current_device.index
                    assert (
                        _stream_handle(torch.cuda.current_stream(input_device))
                        == _stream_handle(stream)
                    )
                    backproject_loss.backward()
                    assert torch.cuda.current_device() == current_device.index

        stream.synchronize()
        for device in devices:
            torch.cuda.synchronize(device)

    _assert_launch_phases(
        tracer.records,
        [
            ("project_forward", beam, "forward", expected_views, selected_streams),
            ("backproject_forward", beam, "backward", expected_views, selected_streams),
            ("project_gradient_forward", beam, "forward", expected_views, selected_streams),
            ("project_backward", beam, "backward", expected_views, selected_streams),
            ("backproject_gradient_forward", beam, "backward", expected_views, selected_streams),
            ("backproject_backward", beam, "forward", expected_views, selected_streams),
        ],
    )

    with torch.cuda.device(input_device):
        expected_projection = raw_project(volume)
        expected_backprojection = raw_backproject(sinogram)
        expected_image_input = volume.detach().clone().requires_grad_()
        (raw_project(expected_image_input) * image_weight).sum().backward()
        expected_sino_input = sinogram.detach().clone().requires_grad_()
        (raw_backproject(expected_sino_input) * sino_weight).sum().backward()
    for device in devices:
        torch.cuda.synchronize(device)

    torch.testing.assert_close(
        projection, expected_projection, rtol=5e-4, atol=5e-5
    )
    torch.testing.assert_close(
        backprojection, expected_backprojection, rtol=5e-4, atol=5e-5
    )
    assert projection.device == input_device
    assert backprojection.device == input_device
    torch.testing.assert_close(
        image_input.grad,
        expected_image_input.grad,
        rtol=5e-4,
        atol=5e-5,
    )
    torch.testing.assert_close(
        sino_input.grad,
        expected_sino_input.grad,
        rtol=5e-4,
        atol=5e-5,
    )


@pytest.mark.cuda
@pytest.mark.parametrize("beam", ["parallel", "fan", "cone"])
def test_projector_real_kernels_use_nondefault_stream(beam):
    _require_cuda()
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed({"parallel": 201, "fan": 202, "cone": 203}[beam])
    projector, volume, _, _, _, _ = _cuda_case(
        beam, device, n_views=4
    )
    projection_shape = projector.projection_shape
    sinogram = torch.linspace(
        -0.2, 0.2, steps=math.prod(projection_shape), device=device
    ).reshape(projection_shape)

    with torch.cuda.device(device):
        torch.cuda.synchronize(device)
        stream = torch.cuda.Stream(device=device)
        selected_streams = {device.index: _stream_handle(stream)}
        with _KernelLaunchTracer() as tracer:
            with torch.cuda.stream(stream):
                with tracer.in_phase("project"):
                    projected = projector.project(volume)
                with tracer.in_phase("backproject"):
                    backprojection = projector.backproject(sinogram)
        stream.synchronize()
        torch.cuda.synchronize(device)

    expected_views = {device.index: projection_shape[0]}
    _assert_launch_phases(
        tracer.records,
        [
            ("project", beam, "forward", expected_views, selected_streams),
            ("backproject", beam, "backward", expected_views, selected_streams),
        ],
    )
    assert projected.shape == projection_shape
    assert backprojection.shape == volume.shape
@pytest.mark.cuda
@pytest.mark.parametrize("beam", ["parallel", "fan", "cone"])
def test_projector_rejects_input_shape_mismatch(beam):
    _require_cuda()
    device = torch.device("cuda", torch.cuda.current_device())
    projector, volume, _, _, _, _ = _cuda_case(beam, device, n_views=4)
    wrong_volume = torch.zeros(
        (volume.shape[0] + 1, *volume.shape[1:]), device=device
    )
    with pytest.raises((TypeError, ValueError)):
        projector.project(wrong_volume)

    projection_shape = projector.projection_shape
    wrong_sinogram = torch.zeros(
        (projection_shape[0] + 1, *projection_shape[1:]), device=device
    )
    with pytest.raises((TypeError, ValueError)):
        projector.backproject(wrong_sinogram)
