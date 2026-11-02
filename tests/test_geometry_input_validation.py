"""Trajectory constructors reject degenerate user inputs instead of building NaN geometry."""

from contextlib import contextmanager

import numpy as np
import pytest
import torch

from diffct.geometry import (
    circular_trajectory_2d_parallel,
    circular_trajectory_3d,
    custom_trajectory_2d_fan,
    custom_trajectory_2d_parallel,
    custom_trajectory_3d,
    random_trajectory_3d,
    saddle_trajectory_3d,
    sinusoidal_trajectory_2d_fan,
    sinusoidal_trajectory_2d_parallel,
    sinusoidal_trajectory_3d,
    spiral_trajectory_3d,
)


def _unit_dirs(angles):
    dirs = torch.zeros((len(angles), 2))
    dirs[:, 0] = torch.cos(angles)
    dirs[:, 1] = torch.sin(angles)
    return dirs


def _origins(angles):
    origins = torch.zeros((len(angles), 2))
    origins[:, 1] = -50.0
    return origins


def _source_path_2d(angles, sid):
    return sid * _unit_dirs(angles)


def _source_path_3d(angles, sid):
    return torch.cat((_source_path_2d(angles, sid),
                      torch.zeros_like(angles).unsqueeze(1)), dim=1)


@pytest.fixture(
    params=[
        (circular_trajectory_2d_parallel, {}),
        (circular_trajectory_3d, {"sid": 1000.0, "sdd": 1500.0}),
        (spiral_trajectory_3d, {"sid": 1000.0, "sdd": 1500.0}),
        (sinusoidal_trajectory_3d, {"sid": 1000.0, "sdd": 1500.0}),
        (saddle_trajectory_3d, {"sid": 1000.0, "sdd": 1500.0}),
        (custom_trajectory_2d_parallel,
         {"ray_dir_fn": _unit_dirs, "det_origin_fn": _origins}),
        (custom_trajectory_2d_fan,
         {"sid": 1000.0, "sdd": 1500.0, "source_path_fn": _source_path_2d}),
        (custom_trajectory_3d,
         {"sid": 1000.0, "sdd": 1500.0, "source_path_fn": _source_path_3d}),
        (sinusoidal_trajectory_2d_fan, {"sid": 1000.0, "sdd": 1500.0}),
        (sinusoidal_trajectory_2d_parallel, {}),
    ],
    ids=lambda case: case[0].__name__,
)
def view_count_constructor(request):
    return request.param


def test_custom_parallel_rejects_zero_ray_direction():
    def dirs(angles):
        out = _unit_dirs(angles)
        out[3] = 0.0
        return out

    with pytest.raises(ValueError, match="zero ray direction"):
        custom_trajectory_2d_parallel(8, dirs, _origins, device="cpu")


def test_custom_parallel_rejects_non_finite_ray_direction():
    def dirs(angles):
        out = _unit_dirs(angles)
        out[2, 0] = float("nan")
        return out

    with pytest.raises(ValueError, match="non-finite"):
        custom_trajectory_2d_parallel(8, dirs, _origins, device="cpu")


def test_custom_parallel_accepts_unit_directions():
    ray_dir, det_origin, det_u_vec = custom_trajectory_2d_parallel(
        8, _unit_dirs, _origins, device="cpu")
    assert torch.isfinite(ray_dir).all()
    assert torch.allclose(torch.linalg.vector_norm(ray_dir, dim=1), torch.ones(8))
    assert torch.isfinite(det_u_vec).all()


def test_random_trajectory_sources_stay_valid():
    src_pos, det_center, det_u_vec, det_v_vec = random_trajectory_3d(
        32, sid_mean=1000.0, sdd_mean=1500.0, sid_std=10.0, device="cpu", seed=7)
    assert torch.isfinite(src_pos).all()
    assert (torch.linalg.vector_norm(src_pos, dim=1) > 0).all()
    assert torch.isfinite(det_u_vec).all() and torch.isfinite(det_v_vec).all()


@pytest.mark.parametrize("n_views", [0, -1])
def test_nonpositive_view_count_rejected(n_views):
    with pytest.raises(ValueError, match="n_views"):
        circular_trajectory_2d_parallel(n_views, device="cpu")
    with pytest.raises(ValueError, match="n_views"):
        spiral_trajectory_3d(n_views, 1000.0, 1500.0, device="cpu")
    with pytest.raises(ValueError, match="n_views"):
        custom_trajectory_2d_parallel(n_views, _unit_dirs, _origins, device="cpu")


def test_angle_tensor_view_counts_still_accepted():
    angles = torch.linspace(0.0, torch.pi, 16)
    ray_dir, det_origin, det_u_vec = circular_trajectory_2d_parallel(
        angles, device="cpu")
    assert ray_dir.shape == (16, 2)


@pytest.mark.parametrize("n_views", [
    pytest.param(1, id="python-int-one"),
    pytest.param(4, id="python-int-many"),
    pytest.param(np.int32(1), id="numpy-int32-one"),
    pytest.param(np.int64(4), id="numpy-int64-many"),
    pytest.param(np.uint32(1), id="numpy-uint32-one"),
    pytest.param(np.uint64(4), id="numpy-uint64-many"),
])
def test_integer_view_counts_accepted(view_count_constructor, n_views):
    constructor, kwargs = view_count_constructor
    trajectory = constructor(n_views, device="cpu", **kwargs)
    assert trajectory
    for component in trajectory:
        assert component.shape[0] == int(n_views)
        assert component.device.type == "cpu"


@pytest.mark.parametrize("n_views", [
    pytest.param(2.5, id="fractional-float"),
    pytest.param(3.0, id="integral-float"),
    pytest.param(np.float32(3.0), id="numpy-integral-float"),
    pytest.param(np.float64(2.5), id="numpy-fractional-float"),
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
])
def test_noninteger_view_counts_raise_type_error(view_count_constructor, n_views):
    constructor, kwargs = view_count_constructor
    with pytest.raises(TypeError, match="n_views"):
        constructor(n_views, device="cpu", **kwargs)


@pytest.mark.parametrize("n_views", [
    pytest.param(0, id="python-zero"),
    pytest.param(-1, id="python-negative"),
    pytest.param(np.int32(0), id="numpy-zero"),
    pytest.param(np.int64(-1), id="numpy-negative"),
    pytest.param(np.uint64(0), id="numpy-unsigned-zero"),
])
def test_nonpositive_integer_view_counts_raise_value_error(
        view_count_constructor, n_views):
    constructor, kwargs = view_count_constructor
    with pytest.raises(ValueError, match="n_views"):
        constructor(n_views, device="cpu", **kwargs)


@pytest.mark.parametrize("constructor,kwargs", [
    pytest.param(circular_trajectory_2d_parallel, {}, id="circular-2d-parallel"),
    pytest.param(circular_trajectory_3d, {"sid": 1000.0, "sdd": 1500.0},
                 id="circular-3d"),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("angle_values", [[0.25], [0.1, 0.7, 2.6]],
                         ids=["one-angle", "multiple-angles"])
def test_circular_trajectories_accept_explicit_angle_tensors(
        constructor, kwargs, dtype, angle_values):
    angles = torch.tensor(angle_values, device="cpu", dtype=dtype)
    trajectory = constructor(angles, device="cpu", **kwargs)
    assert trajectory
    for component in trajectory:
        assert component.shape[0] == len(angle_values)
        assert component.device.type == "cpu"


def test_random_trajectory_seed_leaves_the_global_generator_untouched():
    torch.manual_seed(1234)
    expected = torch.randn(4)
    torch.manual_seed(1234)
    random_trajectory_3d(32, sid_mean=1000.0, sdd_mean=1500.0, sid_std=10.0,
                         pos_std=1.0, angle_std=1e-3, device="cpu", seed=7)
    torch.testing.assert_close(torch.randn(4), expected)


def test_random_trajectory_seed_is_reproducible():
    kwargs = dict(sid_mean=1000.0, sdd_mean=1500.0, sid_std=10.0, pos_std=1.0,
                  angle_std=1e-3, device="cpu", seed=7)
    first = random_trajectory_3d(32, **kwargs)
    second = random_trajectory_3d(32, **kwargs)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b)


def test_random_trajectory_seed_still_replays_the_previous_stream():
    """A private generator must not move any trajectory that a seed pinned down.

    Seeding used to reach the numbers through the process-wide generator, so a
    caller could replay them. A generator seeded on the same device produces
    that very stream, which is why switching to a private one is value-free.
    """
    torch.manual_seed(2026)
    expected = [torch.randn(16), torch.randn(16, 3)]
    generator = torch.Generator(device="cpu").manual_seed(2026)
    for shape, want in zip([(16,), (16, 3)], expected):
        torch.testing.assert_close(torch.randn(shape, generator=generator), want)


@contextmanager
def _reference_random_trajectory(n_views, sid_mean, sdd_mean, sid_std=0.0,
                                 pos_std=0.0, angle_std=0.0, device="cuda",
                                 dtype=torch.float32, seed=None):
    """Replay the random algorithm from 062bddd without leaking RNG state.

    Keep subsequent stream checks inside this context, before RNG restoration.
    The deterministic circular constructor is shared with the original algorithm.
    """
    with torch.random.fork_rng(), torch.random.fork_rng(
            devices=[0], device_type="mps", enabled=torch.backends.mps.is_available()):
        if seed is not None:
            torch.manual_seed(seed)
        src_pos, det_center, det_u_vec, det_v_vec = circular_trajectory_3d(
            n_views, sid_mean, sdd_mean, device=device, dtype=dtype)

        if sid_std > 0.0:
            sid_perturbations = torch.randn(n_views, device=device, dtype=dtype) * sid_std
            sdd_perturbations = sid_perturbations.clone()
            angles = torch.atan2(-src_pos[:, 0], src_pos[:, 1])
            cos_angles = torch.cos(angles)
            sin_angles = torch.sin(angles)
            sid_actual = sid_mean + sid_perturbations
            src_pos[:, 0] = -sid_actual * sin_angles
            src_pos[:, 1] = sid_actual * cos_angles
            sdd_actual = sdd_mean + sdd_perturbations
            idd = sdd_actual - sid_actual
            det_center[:, 0] = idd * sin_angles
            det_center[:, 1] = -idd * cos_angles

        if pos_std > 0.0:
            src_pos += torch.randn_like(src_pos) * pos_std
            det_center += torch.randn_like(det_center) * pos_std

        if angle_std > 0.0:
            angle_perturbations = torch.randn(n_views, device=device, dtype=dtype) * angle_std
            cos_perturb = torch.cos(angle_perturbations)
            sin_perturb = torch.sin(angle_perturbations)
            u_new_x = det_u_vec[:, 0] * cos_perturb - det_u_vec[:, 1] * sin_perturb
            u_new_y = det_u_vec[:, 0] * sin_perturb + det_u_vec[:, 1] * cos_perturb
            det_u_vec[:, 0] = u_new_x
            det_u_vec[:, 1] = u_new_y

        det_u_vec = det_u_vec / torch.norm(det_u_vec, dim=1, keepdim=True)
        det_v_vec = det_v_vec / torch.norm(det_v_vec, dim=1, keepdim=True)
        yield src_pos, det_center, det_u_vec, det_v_vec


@pytest.fixture
def preserved_random_states():
    cpu = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    mps = torch.mps.get_rng_state().clone() if torch.backends.mps.is_available() else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda:
            torch.cuda.set_rng_state_all(cuda)
        if mps is not None:
            torch.mps.set_rng_state(mps)


@pytest.fixture(params=[
    pytest.param(("cpu", torch.float32), id="cpu-float32"),
    pytest.param(("cpu", torch.float64), id="cpu-float64"),
    pytest.param(("mps", torch.float32), id="mps-float32",
                 marks=pytest.mark.skipif(not torch.backends.mps.is_available(),
                                          reason="MPS is unavailable")),
])
def random_device_dtype(request):
    # MPS does not support float64 tensors.
    return request.param


def assert_trajectory_bits_equal(actual, expected):
    assert len(actual) == len(expected) == 4
    for got, want in zip(actual, expected):
        assert got.shape == want.shape
        assert got.dtype == want.dtype
        assert got.device == want.device
        assert torch.equal(got.detach().cpu().contiguous().view(torch.uint8),
                           want.detach().cpu().contiguous().view(torch.uint8))


@pytest.mark.parametrize("seed", [
    pytest.param(73, id="int"),
    pytest.param(True, id="bool-true"),
    pytest.param(False, id="bool-false"),
    pytest.param(np.int64(73), id="numpy-int"),
    pytest.param(torch.tensor(73), id="scalar-tensor"),
])
@pytest.mark.parametrize("noise", [
    pytest.param({}, id="no-noise"),
    pytest.param(dict(sid_std=7.0), id="sid"),
    pytest.param(dict(pos_std=0.75), id="position"),
    pytest.param(dict(angle_std=0.03), id="angle"),
    pytest.param(dict(sid_std=7.0, pos_std=0.75, angle_std=0.03), id="all"),
])
def test_random_seeded_matches_main_bits(
        random_device_dtype, preserved_random_states, seed, noise):
    device, dtype = random_device_dtype
    kwargs = dict(sid_mean=1000.0, sdd_mean=1500.0, device=device,
                  dtype=dtype, seed=seed, **noise)
    with _reference_random_trajectory(17, **kwargs) as expected:
        pass
    actual = random_trajectory_3d(17, **kwargs)
    assert_trajectory_bits_equal(actual, expected)


@pytest.mark.parametrize("seed", [73, True, np.int64(73), torch.tensor(73), None],
                         ids=["int", "bool", "numpy-int", "scalar-tensor", "global"])
@pytest.mark.parametrize("pos_std", [0.0, 0.75])
def test_random_angle_tensor_matches_main_bits(
        random_device_dtype, preserved_random_states, seed, pos_std):
    device, angle_dtype = random_device_dtype
    angles = torch.tensor([0.1, 0.7, 2.6, 4.9], device=device, dtype=angle_dtype)
    # The reference accepts angle vectors for position noise, but not SID or
    # angular noise, whose random draws require an integer view count.
    kwargs = dict(sid_mean=1000.0, sdd_mean=1500.0, pos_std=pos_std,
                  device=device, dtype=torch.float32, seed=seed)
    torch.manual_seed(2026)
    with _reference_random_trajectory(angles, **kwargs) as expected:
        pass
    torch.manual_seed(2026)
    actual = random_trajectory_3d(angles, **kwargs)
    assert_trajectory_bits_equal(actual, expected)


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(),
                                                 reason="CUDA is unavailable")),
    pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(),
                                                reason="MPS is unavailable")),
])
def test_random_seed_preserves_all_global_rng_states(device, preserved_random_states):
    torch.manual_seed(1234)
    cpu = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    mps = torch.mps.get_rng_state().clone() if torch.backends.mps.is_available() else None
    random_trajectory_3d(17, 1000.0, 1500.0, sid_std=7.0, pos_std=0.75,
                         angle_std=0.03, device=device, seed=73)
    assert torch.equal(torch.get_rng_state(), cpu)
    for actual, expected in zip(torch.cuda.get_rng_state_all() if cuda else [], cuda):
        assert torch.equal(actual, expected)
    if mps is not None:
        assert torch.equal(torch.mps.get_rng_state(), mps)


def test_random_unseeded_matches_main_global_stream(
        random_device_dtype, preserved_random_states):
    device, dtype = random_device_dtype
    kwargs = dict(sid_mean=1000.0, sdd_mean=1500.0, sid_std=7.0,
                  pos_std=0.75, angle_std=0.03, device=device, dtype=dtype, seed=None)
    torch.manual_seed(2026)
    with _reference_random_trajectory(17, **kwargs) as expected:
        expected_next = torch.randn(9, device=device, dtype=dtype)
    torch.manual_seed(2026)
    actual = random_trajectory_3d(17, **kwargs)
    actual_next = torch.randn(9, device=device, dtype=dtype)
    assert_trajectory_bits_equal(actual, expected)
    assert torch.equal(actual_next, expected_next)


@pytest.mark.parametrize("error_type", [RuntimeError, TypeError, ValueError])
def test_random_device_generator_failure_is_not_silently_replaced(monkeypatch, error_type):
    original_generator = torch.Generator

    def unavailable_device_generator(*args, **kwargs):
        if "device" in kwargs:
            raise error_type("per-device generator unavailable")
        return original_generator(*args, **kwargs)

    monkeypatch.setattr(torch, "Generator", unavailable_device_generator)
    with pytest.raises(error_type, match="per-device generator unavailable"):
        random_trajectory_3d(17, 1000.0, 1500.0, sid_std=7.0, pos_std=0.75,
                             angle_std=0.03, device="cpu", seed=73)
