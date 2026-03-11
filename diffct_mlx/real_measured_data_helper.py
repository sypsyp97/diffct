"""Helpers for loading and validating measured cone-beam geometries."""

from pathlib import Path
import re

import numpy as np

__all__ = [
    "apply_upper_left_detector_transform",
    "apply_detector_array_convention",
    "apply_detector_geometry_convention",
    "auto_voxel_spacing_from_detector",
    "build_upper_left_detector_transform",
    "diagnose_cone_geometry",
    "estimate_cone_isocenter",
    "load_tiff_projections",
    "normalize_volume",
    "resize_volume_to_shape",
    "shift_detector_center",
    "transform_detector_offsets",
]

_NATURAL_SORT_RE = re.compile(r"\d+|\D+")


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


def _read_tiff_2d(path):
    try:
        import tifffile

        image = tifffile.imread(path)
    except ImportError:
        from PIL import Image

        with Image.open(path) as img:
            image = np.asarray(img)

    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"Projection at {path} must be 2D, got shape {image.shape}")
    return image


def build_upper_left_detector_transform(
    detector_pixels_u,
    detector_pixels_v,
    *,
    detector_binning_u=1,
    detector_binning_v=1,
    flip_v_axis=True,
):
    """Build the legacy detector transform from centered detector coords to upper-left pixel coords.

    The returned 3x3 homogeneous transform matches the legacy convention used in
    `arbitrary_projection_matrix_vox2pix.py`, including the optional sign flip on
    the second detector axis.
    """
    eff_u = max(1, int(detector_pixels_u) // max(1, int(detector_binning_u)))
    eff_v = max(1, int(detector_pixels_v) // max(1, int(detector_binning_v)))

    transform = np.eye(3, dtype=np.float64)
    transform[0, 2] = (float(eff_v) - 1.0) / 2.0
    transform[1, 2] = (float(eff_u) - 1.0) / 2.0
    if flip_v_axis:
        transform[1, 1] *= -1.0
    return transform


def apply_upper_left_detector_transform(
    projection_matrix,
    detector_pixels_u,
    detector_pixels_v,
    *,
    detector_binning_u=1,
    detector_binning_v=1,
    flip_v_axis=True,
):
    """Left-multiply a 3x4 projection matrix with the optional upper-left detector transform."""
    projection_matrix = np.asarray(projection_matrix, dtype=np.float64)
    if projection_matrix.shape != (3, 4):
        raise ValueError(
            f"projection_matrix must have shape (3, 4), got {projection_matrix.shape}"
        )
    transform = build_upper_left_detector_transform(
        detector_pixels_u,
        detector_pixels_v,
        detector_binning_u=detector_binning_u,
        detector_binning_v=detector_binning_v,
        flip_v_axis=flip_v_axis,
    )
    return transform @ projection_matrix


def apply_detector_array_convention(
    projections,
    det_u_vec=None,
    det_v_vec=None,
    *,
    du=None,
    dv=None,
    flip_u=False,
    flip_v=False,
    transpose_uv=False,
):
    """Apply detector-axis convention changes to image data and optional geometry.

    This is the practical counterpart to the legacy projection-matrix transform:
    it changes the detector array convention directly on the measured projections
    and keeps detector basis vectors and pixel pitches consistent.

    Parameters
    ----------
    projections : array-like
        Projection stack with shape ``(n_views, det_u, det_v)``.
    det_u_vec, det_v_vec : array-like or None
        Optional detector basis vectors, each with shape ``(n_views, 3)``.
    du, dv : float or None
        Optional detector pixel pitches. Swapped when ``transpose_uv=True``.
    flip_u, flip_v : bool
        Mirror the detector data along the corresponding detector axis.
    transpose_uv : bool
        Swap the detector u/v axes in both the data and the geometry metadata.

    Returns
    -------
    projections_out, det_u_vec_out, det_v_vec_out, du_out, dv_out
    """
    projections_out = np.asarray(projections)
    if projections_out.ndim != 3:
        raise ValueError(
            f"projections must have shape (n_views, det_u, det_v), got {projections_out.shape}"
        )

    det_u_out = None if det_u_vec is None else _as_3d_float_array(det_u_vec, "det_u_vec").copy()
    det_v_out = None if det_v_vec is None else _as_3d_float_array(det_v_vec, "det_v_vec").copy()
    du_out = du
    dv_out = dv

    if flip_u:
        projections_out = projections_out[:, ::-1, :]
        if det_u_out is not None:
            det_u_out *= -1.0

    if flip_v:
        projections_out = projections_out[:, :, ::-1]
        if det_v_out is not None:
            det_v_out *= -1.0

    if transpose_uv:
        projections_out = np.transpose(projections_out, (0, 2, 1))
        det_u_out, det_v_out = det_v_out, det_u_out
        du_out, dv_out = dv_out, du_out

    return projections_out, det_u_out, det_v_out, du_out, dv_out


def apply_detector_geometry_convention(
    det_u_vec=None,
    det_v_vec=None,
    *,
    du=None,
    dv=None,
    det_u=None,
    det_v=None,
    flip_u=False,
    flip_v=False,
    transpose_uv=False,
):
    """Apply detector-axis convention changes to geometry metadata only."""
    det_u_out = None if det_u_vec is None else _as_3d_float_array(det_u_vec, "det_u_vec").copy()
    det_v_out = None if det_v_vec is None else _as_3d_float_array(det_v_vec, "det_v_vec").copy()
    du_out = du
    dv_out = dv
    det_u_out_px = det_u
    det_v_out_px = det_v

    if flip_u and det_u_out is not None:
        det_u_out *= -1.0

    if flip_v and det_v_out is not None:
        det_v_out *= -1.0

    if transpose_uv:
        det_u_out, det_v_out = det_v_out, det_u_out
        du_out, dv_out = dv_out, du_out
        det_u_out_px, det_v_out_px = det_v_out_px, det_u_out_px

    return det_u_out, det_v_out, du_out, dv_out, det_u_out_px, det_v_out_px


def auto_voxel_spacing_from_detector(
    volume_shape,
    detector_shape_uv,
    detector_pitch_u_mm,
    detector_pitch_v_mm,
    magnification,
    fov_margin_mm=8.0,
):
    """Choose a voxel spacing whose object-space FOV fits inside the volume."""
    eff_u, eff_v = detector_shape_uv
    if magnification <= 0.0:
        raise ValueError(f"Invalid magnification: {magnification}")

    raw_fov_u_mm = (eff_u * detector_pitch_u_mm) / magnification
    raw_fov_v_mm = (eff_v * detector_pitch_v_mm) / magnification
    fov_u_mm = raw_fov_u_mm - 2.0 * float(fov_margin_mm)
    fov_v_mm = raw_fov_v_mm - 2.0 * float(fov_margin_mm)
    if fov_u_mm <= 0.0:
        fov_u_mm = raw_fov_u_mm
    if fov_v_mm <= 0.0:
        fov_v_mm = raw_fov_v_mm

    depth, height, width = volume_shape
    return float(
        min(
            fov_u_mm / float(width),
            fov_v_mm / float(height),
            fov_v_mm / float(depth),
        )
    )


def transform_detector_offsets(horizontal_px, vertical_px, config):
    """Map header pixel offsets into a detector convention."""
    offset_u_px = float(horizontal_px)
    offset_v_px = float(vertical_px)
    if config["flip_u"]:
        offset_u_px *= -1.0
    if config["flip_v"]:
        offset_v_px *= -1.0
    if config["transpose_uv"]:
        offset_u_px, offset_v_px = offset_v_px, offset_u_px
    return offset_u_px, offset_v_px


def shift_detector_center(det_center, det_u_vec, det_v_vec, du, dv, offset_u_px=0.0, offset_v_px=0.0):
    """Shift detector centers by pixel offsets along the detector basis vectors."""
    det_center_np = np.asarray(det_center, dtype=np.float32)
    det_u_np = np.asarray(det_u_vec, dtype=np.float32)
    det_v_np = np.asarray(det_v_vec, dtype=np.float32)
    return (
        det_center_np
        + float(offset_u_px * du) * det_u_np
        + float(offset_v_px * dv) * det_v_np
    ).astype(np.float32, copy=False)


def resize_volume_to_shape(volume, target_shape):
    from scipy.ndimage import zoom

    volume_np = np.asarray(volume, dtype=np.float32)
    target_shape = tuple(int(x) for x in target_shape)
    if volume_np.shape == target_shape:
        return volume_np

    zoom_factors = tuple(t / s for s, t in zip(volume_np.shape, target_shape))
    return zoom(volume_np, zoom_factors, order=1).astype(np.float32, copy=False)


def normalize_volume(volume):
    volume_np = np.asarray(volume, dtype=np.float32)
    vol_min = float(np.min(volume_np))
    vol_max = float(np.max(volume_np))
    scale = vol_max - vol_min
    if scale <= 0.0:
        return np.zeros_like(volume_np, dtype=np.float32)
    return ((volume_np - vol_min) / scale).astype(np.float32, copy=False)


def _bin_mean_2d(image, bin_u=1, bin_v=1):
    bin_u = max(1, int(bin_u))
    bin_v = max(1, int(bin_v))
    image = np.asarray(image, dtype=np.float32)
    if bin_u == 1 and bin_v == 1:
        return image

    height, width = image.shape
    eff_height = (height // bin_v) * bin_v
    eff_width = (width // bin_u) * bin_u
    if eff_height <= 0 or eff_width <= 0:
        raise ValueError(
            f"Invalid detector binning ({bin_u}, {bin_v}) for image shape {image.shape}"
        )

    if eff_height != height or eff_width != width:
        off_v = (height - eff_height) // 2
        off_u = (width - eff_width) // 2
        image = image[off_v : off_v + eff_height, off_u : off_u + eff_width]

    return image.reshape(
        eff_height // bin_v,
        bin_v,
        eff_width // bin_u,
        bin_u,
    ).mean(axis=(1, 3), dtype=np.float32)


def _save_projection_debug_figure(before, after, view_indices, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(view_indices), figsize=(4 * len(view_indices), 8))
    axes = np.asarray(axes, dtype=object)
    if axes.ndim == 1:
        axes = axes[:, None]

    for col, view_idx in enumerate(view_indices):
        axes[0, col].imshow(before[col], cmap="gray")
        axes[0, col].set_title(f"Raw view {view_idx}")
        axes[0, col].axis("off")

        axes[1, col].imshow(after[col], cmap="gray")
        axes[1, col].set_title(f"Log view {view_idx}")
        axes[1, col].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _projection_sort_key(path):
    return tuple(
        int(part) if part[0].isdigit() else part.lower()
        for part in _NATURAL_SORT_RE.findall(path.name)
    )


def load_tiff_projections(
    proj_dir,
    log_transform=False,
    i0_percentile=99.9,
    view_stride=1,
    detector_binning_u=1,
    detector_binning_v=1,
    debug_visualization=False,
    debug_output_path=None,
):
    proj_path = Path(proj_dir)
    projection_paths = sorted(
        (
            path
            for path in proj_path.iterdir()
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        ),
        key=_projection_sort_key,
    )
    if not projection_paths:
        raise FileNotFoundError(f"No TIFF projections found in {proj_path}")
    projection_paths = projection_paths[:: max(1, int(view_stride))]
    detector_binning_u = max(1, int(detector_binning_u))
    detector_binning_v = max(1, int(detector_binning_v))

    first_proj = np.asarray(
        _bin_mean_2d(
            _read_tiff_2d(projection_paths[0]),
            bin_u=detector_binning_u,
            bin_v=detector_binning_v,
        ),
        dtype=np.float32,
    )
    stack = np.empty((len(projection_paths),) + first_proj.shape, dtype=np.float32)
    stack[0] = first_proj

    for idx, path in enumerate(projection_paths[1:], start=1):
        proj = np.asarray(
            _bin_mean_2d(
                _read_tiff_2d(path),
                bin_u=detector_binning_u,
                bin_v=detector_binning_v,
            ),
            dtype=np.float32,
        )
        if proj.shape != first_proj.shape:
            raise ValueError(
                f"Projection at {path} has shape {proj.shape}, expected {first_proj.shape}"
            )
        stack[idx] = proj

    if log_transform:
        debug_indices = None
        debug_before = None
        if debug_visualization:
            debug_indices = sorted({0, len(stack) // 2, len(stack) - 1})
            debug_before = stack[debug_indices].copy()
        i0 = max(float(np.percentile(stack, i0_percentile)), 1.0)
        np.divide(stack, i0, out=stack)
        np.clip(stack, 1e-6, 1.0, out=stack)
        np.log(stack, out=stack)
        stack *= -10.0
        if debug_before is not None:
            output_path = (
                Path(debug_output_path)
                if debug_output_path is not None
                else proj_path / "projection_log_transform_debug.png"
            )
            _save_projection_debug_figure(
                debug_before,
                stack[debug_indices],
                debug_indices,
                output_path,
            )
    return stack


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
