"""Helpers for loading and validating measured cone-beam geometries."""

import numpy as np

__all__ = [
    "diagnose_cone_geometry",
    "estimate_cone_isocenter",
]


def _get_first_present(mapping, *key_paths):
    """Return the first value found for any candidate key path."""
    for key_path in key_paths:
        current = mapping
        found = True
        for key in key_path:
            if not isinstance(current, dict) or key not in current:
                found = False
                break
            current = current[key]
        if found:
            return current
    return None


def _as_3d_float_array(value, name):
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n_views, 3), got {arr.shape}")
    return arr


def _normalize_detector_vectors(vec_np, fallback):
    norm = np.linalg.norm(vec_np, axis=1, keepdims=True)
    out = vec_np.copy()
    non_zero = norm[:, 0] >= 1e-6
    out[non_zero] /= norm[non_zero]
    out[~non_zero] = fallback
    return out


def estimate_cone_isocenter(src_pos, det_center):
    """Estimate the common intersection point of the central rays."""
    src_np = _as_3d_float_array(src_pos, "src_pos")
    det_c_np = _as_3d_float_array(det_center, "det_center")

    ray = det_c_np - src_np
    ray_norm = np.linalg.norm(ray, axis=1, keepdims=True)
    ray_unit = ray / np.maximum(ray_norm, 1e-6)

    identity = np.eye(3, dtype=np.float64)
    system = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    for point, direction in zip(src_np.astype(np.float64), ray_unit.astype(np.float64)):
        proj = identity - np.outer(direction, direction)
        system += proj
        rhs += proj @ point
    return np.linalg.solve(system, rhs).astype(np.float32)


def diagnose_cone_geometry(src_pos, det_center, det_u_vec, det_v_vec):
    """Return compact geometry diagnostics for debugging cone trajectories."""
    src_np = _as_3d_float_array(src_pos, "src_pos")
    det_c_np = _as_3d_float_array(det_center, "det_center")
    det_u_np = _normalize_detector_vectors(
        _as_3d_float_array(det_u_vec, "det_u_vec"),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    det_v_np = _normalize_detector_vectors(
        _as_3d_float_array(det_v_vec, "det_v_vec"),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )

    src_norm = np.linalg.norm(src_np, axis=1)
    ray = det_c_np - src_np
    ray_norm = np.linalg.norm(ray, axis=1)
    ray_unit = ray / np.maximum(ray_norm[:, None], 1e-6)
    src_unit = src_np / np.maximum(src_norm[:, None], 1e-6)

    normal = np.cross(det_u_np, det_v_np)
    normal = _normalize_detector_vectors(
        normal,
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    isocenter = estimate_cone_isocenter(src_np, det_c_np)
    dist_to_iso = np.linalg.norm(np.cross(ray_unit, src_np - isocenter), axis=1)

    return {
        "n_views": int(src_np.shape[0]),
        "sid_min_mm": float(src_norm.min()),
        "sid_max_mm": float(src_norm.max()),
        "sdd_min_mm": float(ray_norm.min()),
        "sdd_max_mm": float(ray_norm.max()),
        "det_u_dot_det_v_max_abs": float(np.abs(np.sum(det_u_np * det_v_np, axis=1)).max()),
        "det_u_dot_ray_max_abs": float(np.abs(np.sum(det_u_np * ray_unit, axis=1)).max()),
        "det_v_dot_ray_max_abs": float(np.abs(np.sum(det_v_np * ray_unit, axis=1)).max()),
        "normal_dot_ray_mean": float(np.mean(np.sum(normal * ray_unit, axis=1))),
        "ray_vs_minus_source_mean": float(np.mean(np.sum(ray_unit * (-src_unit), axis=1))),
        "estimated_isocenter_x_mm": float(isocenter[0]),
        "estimated_isocenter_y_mm": float(isocenter[1]),
        "estimated_isocenter_z_mm": float(isocenter[2]),
        "ray_to_isocenter_max_mm": float(dist_to_iso.max()),
        "ray_to_isocenter_mean_mm": float(dist_to_iso.mean()),
    }
