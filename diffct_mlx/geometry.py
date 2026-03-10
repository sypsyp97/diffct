"""Geometry and trajectory generation for CT scanning.

This module provides functions for generating circular, spiral, sinusoidal,
saddle, and custom trajectories for CT scanning geometries, using MLX arrays
for Apple Silicon acceleration.
"""

import math
import mlx.core as mx
from typing import Callable, Tuple
import json
import numpy as np
import warnings
from .real_measured_data_helper import (
    _as_3d_float_array,
    _get_first_present,
    _normalize_detector_vectors,
    diagnose_cone_geometry,
    estimate_cone_isocenter,
)


# ============================================================================
# 3D Cone Beam Trajectory Generation
# ============================================================================

def circular_trajectory_3d(n_views, sid, sdd, start_angle=0.0, end_angle=None,
                          dtype=mx.float32):
    """Generate circular trajectory geometry for cone-beam CT.

    Parameters
    ----------
    n_views : int
        Number of projection views.
    sid : float
        Source-to-Isocenter Distance (SID).
    sdd : float
        Source-to-Detector Distance (SDD).
    start_angle : float, optional
        Starting angle in radians (default: 0.0).
    end_angle : float, optional
        Ending angle in radians (default: 2*pi).
    dtype : mx.Dtype, optional
        Data type (default: mx.float32).

    Returns
    -------
    src_pos, det_center, det_u_vec, det_v_vec : mx.array
        Each of shape ``(n_views, 3)``.
    """
    if end_angle is None:
        end_angle = 2 * math.pi

    step = (end_angle - start_angle) / n_views
    angles = start_angle + mx.arange(n_views, dtype=dtype) * step

    cos_a = mx.cos(angles)
    sin_a = mx.sin(angles)
    zeros = mx.zeros(n_views, dtype=dtype)
    ones = mx.ones(n_views, dtype=dtype)

    src_pos = mx.stack([-sid * sin_a, sid * cos_a, zeros], axis=1)

    idd = sdd - sid
    det_center = mx.stack([idd * sin_a, -idd * cos_a, zeros], axis=1)

    det_u_vec = mx.stack([cos_a, sin_a, zeros], axis=1)
    det_v_vec = mx.stack([zeros, zeros, ones], axis=1)

    return src_pos, det_center, det_u_vec, det_v_vec


def random_trajectory_3d(n_views, sid_mean, sdd_mean, sid_std=0.0, pos_std=0.0,
                        angle_std=0.0, dtype=mx.float32, seed=None):
    """Generate random trajectory geometry for cone-beam CT.

    Parameters
    ----------
    n_views : int
        Number of projection views.
    sid_mean : float
        Mean Source-to-Isocenter Distance.
    sdd_mean : float
        Mean Source-to-Detector Distance.
    sid_std : float, optional
        Standard deviation for SID variations (default: 0.0).
    pos_std : float, optional
        Standard deviation for position offsets (default: 0.0).
    angle_std : float, optional
        Standard deviation for angular perturbations (default: 0.0).
    dtype : mx.Dtype, optional
        Data type (default: mx.float32).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    src_pos, det_center, det_u_vec, det_v_vec : mx.array
        Each of shape ``(n_views, 3)``.
    """
    if seed is not None:
        mx.random.seed(seed)

    src_pos, det_center, det_u_vec, det_v_vec = circular_trajectory_3d(
        n_views, sid_mean, sdd_mean, dtype=dtype
    )

    if sid_std > 0.0:
        sid_pert = mx.random.normal(shape=(n_views,)) * sid_std
        angles = mx.arctan2(-src_pos[:, 0], src_pos[:, 1])
        cos_a = mx.cos(angles)
        sin_a = mx.sin(angles)

        sid_actual = sid_mean + sid_pert
        src_x = -sid_actual * sin_a
        src_y = sid_actual * cos_a
        src_pos = mx.stack([src_x, src_y, src_pos[:, 2]], axis=1)

        sdd_actual = sdd_mean + sid_pert
        idd = sdd_actual - sid_actual
        det_x = idd * sin_a
        det_y = -idd * cos_a
        det_center = mx.stack([det_x, det_y, det_center[:, 2]], axis=1)

    if pos_std > 0.0:
        src_pos = src_pos + mx.random.normal(shape=src_pos.shape) * pos_std
        det_center = det_center + mx.random.normal(shape=det_center.shape) * pos_std

    if angle_std > 0.0:
        angle_pert = mx.random.normal(shape=(n_views,)) * angle_std
        cos_p = mx.cos(angle_pert)
        sin_p = mx.sin(angle_pert)

        u_new_x = det_u_vec[:, 0] * cos_p - det_u_vec[:, 1] * sin_p
        u_new_y = det_u_vec[:, 0] * sin_p + det_u_vec[:, 1] * cos_p
        det_u_vec = mx.stack([u_new_x, u_new_y, det_u_vec[:, 2]], axis=1)

    # Renormalize
    det_u_vec = det_u_vec / mx.linalg.norm(det_u_vec, axis=1, keepdims=True)
    det_v_vec = det_v_vec / mx.linalg.norm(det_v_vec, axis=1, keepdims=True)

    return src_pos, det_center, det_u_vec, det_v_vec


def spiral_trajectory_3d(n_views, sid, sdd, z_range=100.0, n_turns=2.0,
                        start_angle=0.0, dtype=mx.float32):
    """Generate spiral (helical) trajectory geometry for cone-beam CT.

    Parameters
    ----------
    n_views : int
        Number of projection views.
    sid : float
        Source-to-Isocenter Distance.
    sdd : float
        Source-to-Detector Distance.
    z_range : float, optional
        Total z-axis range (default: 100.0).
    n_turns : float, optional
        Number of complete rotations (default: 2.0).
    start_angle : float, optional
        Starting angle in radians (default: 0.0).
    dtype : mx.Dtype, optional
        Data type (default: mx.float32).

    Returns
    -------
    src_pos, det_center, det_u_vec, det_v_vec : mx.array
        Each of shape ``(n_views, 3)``.
    """
    end_angle = start_angle + 2 * math.pi * n_turns
    step = (end_angle - start_angle) / n_views
    angles = start_angle + mx.arange(n_views, dtype=dtype) * step
    z_positions = mx.linspace(-z_range / 2, z_range / 2, n_views)

    cos_a = mx.cos(angles)
    sin_a = mx.sin(angles)
    zeros = mx.zeros(n_views, dtype=dtype)
    ones = mx.ones(n_views, dtype=dtype)

    src_pos = mx.stack([-sid * sin_a, sid * cos_a, z_positions], axis=1)

    idd = sdd - sid
    det_center = mx.stack([idd * sin_a, -idd * cos_a, z_positions], axis=1)

    det_u_vec = mx.stack([cos_a, sin_a, zeros], axis=1)
    det_v_vec = mx.stack([zeros, zeros, ones], axis=1)

    return src_pos, det_center, det_u_vec, det_v_vec


def sinusoidal_trajectory_3d(n_views, sid, sdd, amplitude=50.0, frequency=2.0,
                            start_angle=0.0, dtype=mx.float32):
    """Generate sinusoidal trajectory geometry for cone-beam CT.

    Parameters
    ----------
    n_views : int
        Number of projection views.
    sid : float
        Mean Source-to-Isocenter Distance.
    sdd : float
        Source-to-Detector Distance.
    amplitude : float, optional
        Amplitude of radial variation (default: 50.0).
    frequency : float, optional
        Oscillation cycles per rotation (default: 2.0).
    start_angle : float, optional
        Starting angle in radians (default: 0.0).
    dtype : mx.Dtype, optional
        Data type (default: mx.float32).

    Returns
    -------
    src_pos, det_center, det_u_vec, det_v_vec : mx.array
        Each of shape ``(n_views, 3)``.
    """
    end_angle = start_angle + 2 * math.pi
    step = (end_angle - start_angle) / n_views
    angles = start_angle + mx.arange(n_views, dtype=dtype) * step

    radial_var = amplitude * mx.sin(frequency * angles)
    sid_varying = sid + radial_var

    cos_a = mx.cos(angles)
    sin_a = mx.sin(angles)
    zeros = mx.zeros(n_views, dtype=dtype)
    ones = mx.ones(n_views, dtype=dtype)

    src_pos = mx.stack([-sid_varying * sin_a, sid_varying * cos_a, zeros], axis=1)

    idd = sdd - sid_varying
    det_center = mx.stack([idd * sin_a, -idd * cos_a, zeros], axis=1)

    det_u_vec = mx.stack([cos_a, sin_a, zeros], axis=1)
    det_v_vec = mx.stack([zeros, zeros, ones], axis=1)

    return src_pos, det_center, det_u_vec, det_v_vec


def saddle_trajectory_3d(n_views, sid, sdd, z_amplitude=50.0, radial_amplitude=30.0,
                        start_angle=0.0, dtype=mx.float32):
    """Generate saddle-shaped trajectory geometry for cone-beam CT.

    Parameters
    ----------
    n_views : int
        Number of projection views.
    sid : float
        Mean Source-to-Isocenter Distance.
    sdd : float
        Source-to-Detector Distance.
    z_amplitude : float, optional
        Amplitude of z-axis variation (default: 50.0).
    radial_amplitude : float, optional
        Amplitude of radial variation (default: 30.0).
    start_angle : float, optional
        Starting angle in radians (default: 0.0).
    dtype : mx.Dtype, optional
        Data type (default: mx.float32).

    Returns
    -------
    src_pos, det_center, det_u_vec, det_v_vec : mx.array
        Each of shape ``(n_views, 3)``.
    """
    end_angle = start_angle + 2 * math.pi
    step = (end_angle - start_angle) / n_views
    angles = start_angle + mx.arange(n_views, dtype=dtype) * step

    z_positions = z_amplitude * mx.cos(2 * angles)
    radial_var = radial_amplitude * mx.sin(2 * angles)
    sid_varying = sid + radial_var

    cos_a = mx.cos(angles)
    sin_a = mx.sin(angles)
    zeros = mx.zeros(n_views, dtype=dtype)
    ones = mx.ones(n_views, dtype=dtype)

    src_pos = mx.stack([-sid_varying * sin_a, sid_varying * cos_a, z_positions], axis=1)

    idd = sdd - sid_varying
    det_center = mx.stack([idd * sin_a, -idd * cos_a, z_positions], axis=1)

    det_u_vec = mx.stack([cos_a, sin_a, zeros], axis=1)
    det_v_vec = mx.stack([zeros, zeros, ones], axis=1)

    return src_pos, det_center, det_u_vec, det_v_vec


def custom_trajectory_3d(n_views, sid, sdd,
                        source_path_fn: Callable,
                        start_angle=0.0, dtype=mx.float32):
    """Generate custom trajectory geometry for cone-beam CT.

    Parameters
    ----------
    n_views : int
        Number of projection views.
    sid : float
        Source-to-Isocenter Distance.
    sdd : float
        Source-to-Detector Distance.
    source_path_fn : callable
        Function ``(angles, sid) -> src_pos`` with shape ``(n_views, 3)``.
    start_angle : float, optional
        Starting angle in radians (default: 0.0).
    dtype : mx.Dtype, optional
        Data type (default: mx.float32).

    Returns
    -------
    src_pos, det_center, det_u_vec, det_v_vec : mx.array
        Each of shape ``(n_views, 3)``.
    """
    end_angle = start_angle + 2 * math.pi
    step = (end_angle - start_angle) / n_views
    angles = start_angle + mx.arange(n_views, dtype=dtype) * step

    src_pos = source_path_fn(angles, sid)
    if src_pos.shape != (n_views, 3):
        raise ValueError(f"source_path_fn must return shape ({n_views}, 3), got {src_pos.shape}")

    src_norm = mx.linalg.norm(src_pos, axis=1)
    mx.eval(src_norm)
    min_src_norm = float(mx.min(src_norm))
    max_src_norm = float(mx.max(src_norm))
    if min_src_norm <= 1e-6:
        raise ValueError("source_path_fn must keep the source away from the isocenter")
    if max_src_norm >= sdd:
        raise ValueError("source_path_fn must keep the source radius smaller than sdd")

    det_center_list = []
    det_u_list = []
    det_v_list = []

    for i in range(n_views):
        src_vec = src_pos[i]
        src_norm = mx.linalg.norm(src_vec)
        src_unit = src_vec / src_norm

        det_c = -src_unit * (sdd - src_norm)
        det_center_list.append(det_c)

        # u-direction: perpendicular to source in xy-plane
        if abs(float(src_vec[0])) < 1e-6 and abs(float(src_vec[1])) < 1e-6:
            u = mx.array([1.0, 0.0, 0.0], dtype=dtype)
        else:
            u_unnorm = mx.array([-float(src_vec[1]), float(src_vec[0]), 0.0], dtype=dtype)
            u = u_unnorm / mx.linalg.norm(u_unnorm)
        det_u_list.append(u)

        # v-direction: cross product
        v = mx.array([
            float(src_unit[1]) * float(u[2]) - float(src_unit[2]) * float(u[1]),
            float(src_unit[2]) * float(u[0]) - float(src_unit[0]) * float(u[2]),
            float(src_unit[0]) * float(u[1]) - float(src_unit[1]) * float(u[0]),
        ], dtype=dtype)
        v = v / mx.linalg.norm(v)
        det_v_list.append(v)

    det_center = mx.stack(det_center_list, axis=0)
    det_u_vec = mx.stack(det_u_list, axis=0)
    det_v_vec = mx.stack(det_v_list, axis=0)

    return src_pos, det_center, det_u_vec, det_v_vec



def _compute_detector_from_source_and_sdd(src_np, sdd):
    src_norm = np.linalg.norm(src_np, axis=1, keepdims=True)
    src_norm = np.maximum(src_norm, 1e-6)
    src_unit = src_np / src_norm

    sdd_np = np.asarray(sdd, dtype=np.float32)
    if sdd_np.ndim == 0:
        sdd_np = np.full((src_np.shape[0], 1), float(sdd_np), dtype=np.float32)
    elif sdd_np.ndim == 1:
        sdd_np = sdd_np[:, None]

    det_c_np = -src_unit * (sdd_np - src_norm)

    det_u_np = np.stack(
        [-src_np[:, 1], src_np[:, 0], np.zeros(src_np.shape[0], dtype=np.float32)],
        axis=1,
    )
    det_u_np = _normalize_detector_vectors(
        det_u_np,
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )

    det_v_np = np.cross(src_unit, det_u_np)
    det_v_np = _normalize_detector_vectors(
        det_v_np,
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    return det_c_np, det_u_np, det_v_np




def load_arbitrary_cone_geometry_from_json(json_path, *, flip_det_u=False,
                                           flip_det_v=False,
                                           warn_on_inconsistency=True,
                                           recenter_to_isocenter=False):
    """Load arbitrary cone trajectory from JSON and return MLX geometry arrays.

    If detector center and basis vectors are present in the JSON, they are used
    directly. ``flip_det_v=True`` is useful for scanners whose protocol stores
    the detector row direction with opposite sign.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        src_raw = _get_first_present(
            payload,
            ("src_pos",),
            ("source_positions",),
            ("source_path",),
            ("trajectory", "src_pos"),
            ("trajectory", "source_positions_mm"),
        )
        if src_raw is None:
            raise ValueError("Missing source positions in JSON.")
        src_np = _as_3d_float_array(src_raw, "src_pos")

        det_c = _get_first_present(
            payload,
            ("det_center",),
            ("detector_center",),
            ("trajectory", "det_center"),
            ("trajectory", "detector_centers_mm"),
        )
        det_u = _get_first_present(
            payload,
            ("det_u_vec",),
            ("detector_u_vec",),
            ("trajectory", "det_u_vec"),
            ("trajectory", "detector_u_vec"),
        )
        det_v = _get_first_present(
            payload,
            ("det_v_vec",),
            ("detector_v_vec",),
            ("trajectory", "det_v_vec"),
            ("trajectory", "detector_v_vec"),
        )
        sdd = _get_first_present(
            payload,
            ("sdd",),
            ("source", "source_to_detector_distance_mm"),
            ("source", "source_to_detector_distance"),
        )
    else:
        src_np = _as_3d_float_array(payload, "src_pos")
        det_c = det_u = det_v = sdd = None

    if det_c is None or det_u is None or det_v is None:
        if sdd is None:
            raise ValueError(
                "Missing SDD in JSON. Provide detector geometry or 'sdd'."
            )
        det_c_np, det_u_np, det_v_np = _compute_detector_from_source_and_sdd(src_np, sdd)
        if warn_on_inconsistency:
            warnings.warn(
                "Detector center/u/v missing in JSON; falling back to geometry "
                "constructed from source positions and SDD.",
                stacklevel=2,
            )
    else:
        det_c_np = _as_3d_float_array(det_c, "det_center")
        det_u_np = _normalize_detector_vectors(
            _as_3d_float_array(det_u, "det_u_vec"),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        det_v_np = _normalize_detector_vectors(
            _as_3d_float_array(det_v, "det_v_vec"),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )

    if flip_det_u:
        det_u_np = -det_u_np
    if flip_det_v:
        det_v_np = -det_v_np

    if recenter_to_isocenter:
        # Measured geometries may be expressed in a world frame whose origin is
        # not the reconstruction isocenter. We translate source and detector
        # centers together so that the central rays intersect near (0, 0, 0)
        # without changing the relative scanner geometry.
        isocenter = estimate_cone_isocenter(src_np, det_c_np)
        print(
            "Recentering geometry: "
            f"estimated isocenter = ({isocenter[0]:.3f}, "
            f"{isocenter[1]:.3f}, {isocenter[2]:.3f}) mm -> (0.000, 0.000, 0.000) mm"
        )
        src_np = src_np - isocenter
        det_c_np = det_c_np - isocenter

    if warn_on_inconsistency and sdd is not None:
        diag = diagnose_cone_geometry(src_np, det_c_np, det_u_np, det_v_np)
        sdd_np = np.asarray(sdd, dtype=np.float32)
        sdd_min = float(np.min(sdd_np))
        sdd_max = float(np.max(sdd_np))
        if (
            abs(diag["sdd_min_mm"] - sdd_min) > 1e-3
            or abs(diag["sdd_max_mm"] - sdd_max) > 1e-3
        ):
            warnings.warn(
                "Loaded detector centers imply an effective SDD range "
                f"[{diag['sdd_min_mm']:.3f}, {diag['sdd_max_mm']:.3f}] mm, "
                f"but JSON metadata reports [{sdd_min:.3f}, {sdd_max:.3f}] mm.",
                stacklevel=2,
            )

    return (
        mx.array(src_np, dtype=mx.float32),
        mx.array(det_c_np, dtype=mx.float32),
        mx.array(det_u_np, dtype=mx.float32),
        mx.array(det_v_np, dtype=mx.float32),
    )



# ============================================================================
# 2D Fan Beam Trajectory Generation
# ============================================================================

def circular_trajectory_2d_fan(n_views, sid, sdd, start_angle=0.0, end_angle=None,
                              dtype=mx.float32):
    """Generate circular trajectory for 2D fan-beam CT.

    Parameters
    ----------
    n_views : int
        Number of projection views.
    sid : float
        Source-to-Isocenter Distance.
    sdd : float
        Source-to-Detector Distance.
    start_angle : float, optional
        Starting angle in radians (default: 0.0).
    end_angle : float, optional
        Ending angle in radians (default: 2*pi).
    dtype : mx.Dtype, optional
        Data type (default: mx.float32).

    Returns
    -------
    src_pos, det_center, det_u_vec : mx.array
        Shapes ``(n_views, 2)``.
    """
    if end_angle is None:
        end_angle = 2 * math.pi

    step = (end_angle - start_angle) / n_views
    angles = start_angle + mx.arange(n_views, dtype=dtype) * step

    cos_a = mx.cos(angles)
    sin_a = mx.sin(angles)

    src_pos = mx.stack([-sid * sin_a, sid * cos_a], axis=1)

    idd = sdd - sid
    det_center = mx.stack([idd * sin_a, -idd * cos_a], axis=1)

    det_u_vec = mx.stack([cos_a, sin_a], axis=1)

    return src_pos, det_center, det_u_vec


def sinusoidal_trajectory_2d_fan(n_views, sid, sdd, amplitude=50.0, frequency=2.0,
                                start_angle=0.0, dtype=mx.float32):
    """Generate sinusoidal trajectory for 2D fan-beam CT.

    Parameters
    ----------
    n_views : int
        Number of projection views.
    sid : float
        Mean Source-to-Isocenter Distance.
    sdd : float
        Source-to-Detector Distance.
    amplitude : float, optional
        Amplitude of radial variation (default: 50.0).
    frequency : float, optional
        Oscillation cycles per rotation (default: 2.0).
    start_angle : float, optional
        Starting angle in radians (default: 0.0).
    dtype : mx.Dtype, optional
        Data type (default: mx.float32).

    Returns
    -------
    src_pos, det_center, det_u_vec : mx.array
        Shapes ``(n_views, 2)``.
    """
    end_angle = start_angle + 2 * math.pi
    step = (end_angle - start_angle) / n_views
    angles = start_angle + mx.arange(n_views, dtype=dtype) * step

    radial_var = amplitude * mx.sin(frequency * angles)
    sid_varying = sid + radial_var

    cos_a = mx.cos(angles)
    sin_a = mx.sin(angles)

    src_pos = mx.stack([-sid_varying * sin_a, sid_varying * cos_a], axis=1)

    idd = sdd - sid_varying
    det_center = mx.stack([idd * sin_a, -idd * cos_a], axis=1)

    det_u_vec = mx.stack([cos_a, sin_a], axis=1)

    return src_pos, det_center, det_u_vec


def custom_trajectory_2d_fan(n_views, sid, sdd,
                            source_path_fn: Callable,
                            start_angle=0.0, dtype=mx.float32):
    """Generate custom trajectory for 2D fan-beam CT.

    Parameters
    ----------
    n_views : int
        Number of projection views.
    sid : float
        Source-to-Isocenter Distance.
    sdd : float
        Source-to-Detector Distance.
    source_path_fn : callable
        Function ``(angles, sid) -> src_pos`` with shape ``(n_views, 2)``.
    start_angle : float, optional
        Starting angle in radians (default: 0.0).
    dtype : mx.Dtype, optional
        Data type (default: mx.float32).

    Returns
    -------
    src_pos, det_center, det_u_vec : mx.array
        Shapes ``(n_views, 2)``.
    """
    end_angle = start_angle + 2 * math.pi
    step = (end_angle - start_angle) / n_views
    angles = start_angle + mx.arange(n_views, dtype=dtype) * step

    src_pos = source_path_fn(angles, sid)
    if src_pos.shape != (n_views, 2):
        raise ValueError(f"source_path_fn must return shape ({n_views}, 2), got {src_pos.shape}")

    src_norm = mx.linalg.norm(src_pos, axis=1)
    mx.eval(src_norm)
    min_src_norm = float(mx.min(src_norm))
    max_src_norm = float(mx.max(src_norm))
    if min_src_norm <= 1e-6:
        raise ValueError("source_path_fn must keep the source away from the isocenter")
    if max_src_norm >= sdd:
        raise ValueError("source_path_fn must keep the source radius smaller than sdd")

    det_center_list = []
    det_u_list = []

    for i in range(n_views):
        src_vec = src_pos[i]
        src_norm = mx.linalg.norm(src_vec)
        src_unit = src_vec / src_norm

        det_c = -src_unit * (sdd - src_norm)
        det_center_list.append(det_c)

        # u-direction: perpendicular (-y, x)
        det_u_list.append(mx.array([-float(src_unit[1]), float(src_unit[0])], dtype=dtype))

    det_center = mx.stack(det_center_list, axis=0)
    det_u_vec = mx.stack(det_u_list, axis=0)

    return src_pos, det_center, det_u_vec


# ============================================================================
# 2D Parallel Beam Trajectory Generation
# ============================================================================

def circular_trajectory_2d_parallel(n_views, detector_distance=0.0, start_angle=0.0,
                                   end_angle=None, dtype=mx.float32):
    """Generate circular trajectory for 2D parallel-beam CT.

    Parameters
    ----------
    n_views : int
        Number of projection views.
    detector_distance : float, optional
        Distance from isocenter to detector origin (default: 0.0).
    start_angle : float, optional
        Starting angle in radians (default: 0.0).
    end_angle : float, optional
        Ending angle in radians (default: pi).
    dtype : mx.Dtype, optional
        Data type (default: mx.float32).

    Returns
    -------
    ray_dir, det_origin, det_u_vec : mx.array
        Shapes ``(n_views, 2)``.
    """
    if end_angle is None:
        end_angle = math.pi

    step = (end_angle - start_angle) / n_views
    angles = start_angle + mx.arange(n_views, dtype=dtype) * step

    cos_a = mx.cos(angles)
    sin_a = mx.sin(angles)

    ray_dir = mx.stack([cos_a, sin_a], axis=1)
    det_origin = mx.stack([-detector_distance * sin_a, detector_distance * cos_a], axis=1)
    det_u_vec = mx.stack([-sin_a, cos_a], axis=1)

    return ray_dir, det_origin, det_u_vec


def sinusoidal_trajectory_2d_parallel(n_views, amplitude=50.0, frequency=2.0,
                                     start_angle=0.0, dtype=mx.float32):
    """Generate sinusoidal trajectory for 2D parallel-beam CT.

    Parameters
    ----------
    n_views : int
        Number of projection views.
    amplitude : float, optional
        Amplitude of detector displacement (default: 50.0).
    frequency : float, optional
        Oscillation cycles per rotation (default: 2.0).
    start_angle : float, optional
        Starting angle in radians (default: 0.0).
    dtype : mx.Dtype, optional
        Data type (default: mx.float32).

    Returns
    -------
    ray_dir, det_origin, det_u_vec : mx.array
        Shapes ``(n_views, 2)``.
    """
    end_angle = start_angle + math.pi
    step = (end_angle - start_angle) / n_views
    angles = start_angle + mx.arange(n_views, dtype=dtype) * step

    det_offset = amplitude * mx.sin(frequency * angles)

    cos_a = mx.cos(angles)
    sin_a = mx.sin(angles)

    ray_dir = mx.stack([cos_a, sin_a], axis=1)
    det_origin = mx.stack([-det_offset * sin_a, det_offset * cos_a], axis=1)
    det_u_vec = mx.stack([-sin_a, cos_a], axis=1)

    return ray_dir, det_origin, det_u_vec


def custom_trajectory_2d_parallel(n_views,
                                  ray_dir_fn: Callable,
                                  det_origin_fn: Callable,
                                  start_angle=0.0, dtype=mx.float32):
    """Generate custom trajectory for 2D parallel-beam CT.

    Parameters
    ----------
    n_views : int
        Number of projection views.
    ray_dir_fn : callable
        Function ``(angles) -> ray_dir`` with shape ``(n_views, 2)``.
    det_origin_fn : callable
        Function ``(angles) -> det_origin`` with shape ``(n_views, 2)``.
    start_angle : float, optional
        Starting angle in radians (default: 0.0).
    dtype : mx.Dtype, optional
        Data type (default: mx.float32).

    Returns
    -------
    ray_dir, det_origin, det_u_vec : mx.array
        Shapes ``(n_views, 2)``.
    """
    end_angle = start_angle + math.pi
    step = (end_angle - start_angle) / n_views
    angles = start_angle + mx.arange(n_views, dtype=dtype) * step

    ray_dir = ray_dir_fn(angles)
    det_origin = det_origin_fn(angles)

    if ray_dir.shape != (n_views, 2):
        raise ValueError(f"ray_dir_fn must return shape ({n_views}, 2), got {ray_dir.shape}")
    if det_origin.shape != (n_views, 2):
        raise ValueError(f"det_origin_fn must return shape ({n_views}, 2), got {det_origin.shape}")

    ray_dir = ray_dir / mx.linalg.norm(ray_dir, axis=1, keepdims=True)

    # u-direction: perpendicular to ray (rotate 90°)
    det_u_vec = mx.stack([-ray_dir[:, 1], ray_dir[:, 0]], axis=1)

    return ray_dir, det_origin, det_u_vec
