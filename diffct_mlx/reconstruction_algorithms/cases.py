"""Reusable reconstruction-case builders for example and benchmark scripts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np

from ..geometry import (
    circular_trajectory_2d_fan,
    circular_trajectory_2d_parallel,
    circular_trajectory_3d,
    load_arbitrary_cone_geometry_from_json,
)
from ..phantoms import shepp_logan_2d, shepp_logan_3d
from ..projectors import (
    cone_backward,
    cone_forward,
    fan_backward,
    fan_forward,
    parallel_backward,
    parallel_forward,
)
from ..real_measured_data_helper import (
    apply_detector_geometry_convention,
    auto_voxel_spacing_from_detector,
    load_tiff_projections,
    normalize_volume,
    resize_volume_to_shape,
    shift_detector_center,
    transform_detector_offsets,
)


Array = mx.array


@dataclass
class ReconstructionCase:
    """Container describing one reconstruction problem instance."""

    name: str
    sinogram: Array
    volume_shape: tuple[int, ...]
    forward_single: Callable[[Array, int], Array]
    back_single: Callable[[Array, int], Array]
    back_project_all: Callable[[Array], Array]
    reference: Array | None = None
    reference_title: str | None = None
    supports_fbp: bool = False
    supports_fdk: bool = False
    fbp_normalization_scale: float | None = None
    iterative_iteration_count: int = 5
    sirt_iteration_count: int = 15
    iterative_sart_iteration_count: int = 2
    iterative_backprojection_scale: float = 0.215
    pocs_iterative_update_method: str = "sart"
    iterative_positivity_mode: str = "per_iteration"
    iterative_detector_border_u: int = 0
    iterative_detector_border_v: int = 0
    iterative_preserve_unmasked_computed_projection: bool = False
    tv_reg_iteration_count: int = 6
    tv_alpha: float = 0.12
    asd_reg_iteration_count: int = 6
    asd_alpha: float = 0.12
    asd_epsilon: float = 0.05
    awtv_reg_iteration_count: int = 6
    awtv_alpha: float = 0.12
    awtv_epsilon: float = 0.08
    awtv_delta: float = 0.6e-2
    fdk_normalization_scale: float | None = None
    fbp_weight: Callable[[Array], Array] | None = None
    fdk_weight: Callable[[Array], Array] | None = None


@dataclass
class MeasuredConeDataConfig:
    """Configuration for loading measured cone-beam data."""

    data_dir: str | Path
    volume_shape: tuple[int, int, int] = (128, 128, 128)
    target_view_count: int = 360
    target_detector_shape: tuple[int, int] = (256, 256)
    trajectory_json_path: str | Path | None = None
    reference_volume_path: str | Path | None = None
    reference_meta_path: str | Path | None = None
    recenter_to_isocenter: bool = True
    flip_det_u: bool = False
    flip_det_v: bool = False
    transpose_uv: bool = True
    flip_u: bool = False
    flip_v: bool = False
    log_transform: bool = True
    revert: bool = False
    viewwise_i0: bool = True
    air_border_px: int = 16
    subtract_air_baseline: bool = True
    air_baseline_percentile: float = 50.0
    measured_fov_margin_mm: float = 8.0
    iterative_backprojection_scale: float = 0.215


def make_parallel_2d_operators(
    ray_dir: Array,
    det_origin: Array,
    det_u_vec: Array,
    *,
    image_shape: tuple[int, int],
    num_detectors: int,
    detector_spacing: float = 1.0,
    voxel_spacing: float = 1.0,
) -> tuple[Callable[[Array, int], Array], Callable[[Array, int], Array], Callable[[Array], Array]]:
    """Create single-view and all-view parallel-beam projector wrappers."""
    ny, nx = image_shape

    def forward_single(volume: Array, projection_index: int) -> Array:
        return parallel_forward(
            volume,
            ray_dir[projection_index : projection_index + 1],
            det_origin[projection_index : projection_index + 1],
            det_u_vec[projection_index : projection_index + 1],
            num_detectors=num_detectors,
            detector_spacing=detector_spacing,
            voxel_spacing=voxel_spacing,
        )[0]

    def forward_slice(volume: Array, start: int, stop: int) -> Array:
        return parallel_forward(
            volume,
            ray_dir[start:stop],
            det_origin[start:stop],
            det_u_vec[start:stop],
            num_detectors=num_detectors,
            detector_spacing=detector_spacing,
            voxel_spacing=voxel_spacing,
        )

    def back_single(projection: Array, projection_index: int) -> Array:
        return parallel_backward(
            mx.array(projection, dtype=mx.float32)[None, :],
            ray_dir[projection_index : projection_index + 1],
            det_origin[projection_index : projection_index + 1],
            det_u_vec[projection_index : projection_index + 1],
            detector_spacing=detector_spacing,
            H=ny,
            W=nx,
            voxel_spacing=voxel_spacing,
        )

    def back_slice(projection: Array, start: int, stop: int) -> Array:
        return parallel_backward(
            mx.array(projection, dtype=mx.float32),
            ray_dir[start:stop],
            det_origin[start:stop],
            det_u_vec[start:stop],
            detector_spacing=detector_spacing,
            H=ny,
            W=nx,
            voxel_spacing=voxel_spacing,
        )

    def back_project_all(filtered_sinogram: Array) -> Array:
        return parallel_backward(
            filtered_sinogram,
            ray_dir,
            det_origin,
            det_u_vec,
            detector_spacing=detector_spacing,
            H=ny,
            W=nx,
            voxel_spacing=voxel_spacing,
        )

    forward_single.project_slice = forward_slice  # type: ignore[attr-defined]
    back_single.project_slice = back_slice  # type: ignore[attr-defined]
    return forward_single, back_single, back_project_all


def make_fan_2d_operators(
    src_pos: Array,
    det_center: Array,
    det_u_vec: Array,
    *,
    image_shape: tuple[int, int],
    num_detectors: int,
    detector_spacing: float = 1.0,
    voxel_spacing: float = 1.0,
) -> tuple[Callable[[Array, int], Array], Callable[[Array, int], Array], Callable[[Array], Array]]:
    """Create single-view and all-view fan-beam projector wrappers."""
    ny, nx = image_shape

    def forward_single(volume: Array, projection_index: int) -> Array:
        return fan_forward(
            volume,
            src_pos[projection_index : projection_index + 1],
            det_center[projection_index : projection_index + 1],
            det_u_vec[projection_index : projection_index + 1],
            num_detectors=num_detectors,
            detector_spacing=detector_spacing,
            voxel_spacing=voxel_spacing,
        )[0]

    def forward_slice(volume: Array, start: int, stop: int) -> Array:
        return fan_forward(
            volume,
            src_pos[start:stop],
            det_center[start:stop],
            det_u_vec[start:stop],
            num_detectors=num_detectors,
            detector_spacing=detector_spacing,
            voxel_spacing=voxel_spacing,
        )

    def back_single(projection: Array, projection_index: int) -> Array:
        return fan_backward(
            mx.array(projection, dtype=mx.float32)[None, :],
            src_pos[projection_index : projection_index + 1],
            det_center[projection_index : projection_index + 1],
            det_u_vec[projection_index : projection_index + 1],
            detector_spacing=detector_spacing,
            H=ny,
            W=nx,
            voxel_spacing=voxel_spacing,
        )

    def back_slice(projection: Array, start: int, stop: int) -> Array:
        return fan_backward(
            mx.array(projection, dtype=mx.float32),
            src_pos[start:stop],
            det_center[start:stop],
            det_u_vec[start:stop],
            detector_spacing=detector_spacing,
            H=ny,
            W=nx,
            voxel_spacing=voxel_spacing,
        )

    def back_project_all(filtered_sinogram: Array) -> Array:
        return fan_backward(
            filtered_sinogram,
            src_pos,
            det_center,
            det_u_vec,
            detector_spacing=detector_spacing,
            H=ny,
            W=nx,
            voxel_spacing=voxel_spacing,
        )

    forward_single.project_slice = forward_slice  # type: ignore[attr-defined]
    back_single.project_slice = back_slice  # type: ignore[attr-defined]
    return forward_single, back_single, back_project_all


def make_cone_3d_operators(
    src_pos: Array,
    det_center: Array,
    det_u_vec: Array,
    det_v_vec: Array,
    *,
    volume_shape: tuple[int, int, int],
    detector_shape: tuple[int, int],
    du: float = 1.0,
    dv: float = 1.0,
    voxel_spacing: float = 1.0,
) -> tuple[Callable[[Array, int], Array], Callable[[Array, int], Array], Callable[[Array], Array]]:
    """Create single-view and all-view cone-beam projector wrappers."""
    nz, ny, nx = volume_shape
    det_u_count, det_v_count = detector_shape

    def forward_single(volume: Array, projection_index: int) -> Array:
        return cone_forward(
            volume,
            src_pos[projection_index : projection_index + 1],
            det_center[projection_index : projection_index + 1],
            det_u_vec[projection_index : projection_index + 1],
            det_v_vec[projection_index : projection_index + 1],
            det_u=det_u_count,
            det_v=det_v_count,
            du=du,
            dv=dv,
            voxel_spacing=voxel_spacing,
        )[0]

    def forward_slice(volume: Array, start: int, stop: int) -> Array:
        return cone_forward(
            volume,
            src_pos[start:stop],
            det_center[start:stop],
            det_u_vec[start:stop],
            det_v_vec[start:stop],
            det_u=det_u_count,
            det_v=det_v_count,
            du=du,
            dv=dv,
            voxel_spacing=voxel_spacing,
        )

    def back_single(projection: Array, projection_index: int) -> Array:
        return cone_backward(
            mx.array(projection, dtype=mx.float32)[None, :, :],
            src_pos[projection_index : projection_index + 1],
            det_center[projection_index : projection_index + 1],
            det_u_vec[projection_index : projection_index + 1],
            det_v_vec[projection_index : projection_index + 1],
            D=nz,
            H=ny,
            W=nx,
            du=du,
            dv=dv,
            voxel_spacing=voxel_spacing,
        )

    def back_slice(projection: Array, start: int, stop: int) -> Array:
        return cone_backward(
            mx.array(projection, dtype=mx.float32),
            src_pos[start:stop],
            det_center[start:stop],
            det_u_vec[start:stop],
            det_v_vec[start:stop],
            D=nz,
            H=ny,
            W=nx,
            du=du,
            dv=dv,
            voxel_spacing=voxel_spacing,
        )

    def back_project_all(filtered_sinogram: Array) -> Array:
        return cone_backward(
            filtered_sinogram,
            src_pos,
            det_center,
            det_u_vec,
            det_v_vec,
            D=nz,
            H=ny,
            W=nx,
            du=du,
            dv=dv,
            voxel_spacing=voxel_spacing,
        )

    forward_single.project_slice = forward_slice  # type: ignore[attr-defined]
    back_single.project_slice = back_slice  # type: ignore[attr-defined]
    return forward_single, back_single, back_project_all


def build_parallel_2d_case(
    *,
    image_shape: tuple[int, int] = (96, 96),
    num_views: int = 180,
    num_detectors: int = 160,
    detector_spacing: float = 1.0,
    voxel_spacing: float = 1.0,
) -> ReconstructionCase:
    """Build a Shepp-Logan 2D parallel-beam case."""
    ny, nx = image_shape
    reference = mx.array(shepp_logan_2d((ny, nx)))
    ray_dir, det_origin, det_u_vec = circular_trajectory_2d_parallel(num_views)
    sinogram = parallel_forward(
        reference,
        ray_dir,
        det_origin,
        det_u_vec,
        num_detectors=num_detectors,
        detector_spacing=detector_spacing,
        voxel_spacing=voxel_spacing,
    )

    forward_single, back_single, back_project_all = make_parallel_2d_operators(
        ray_dir,
        det_origin,
        det_u_vec,
        image_shape=(ny, nx),
        num_detectors=num_detectors,
        detector_spacing=detector_spacing,
        voxel_spacing=voxel_spacing,
    )

    return ReconstructionCase(
        name="Parallel 2D",
        sinogram=sinogram,
        volume_shape=(ny, nx),
        forward_single=forward_single,
        back_single=back_single,
        back_project_all=back_project_all,
        reference=reference,
        reference_title="Phantom",
        supports_fbp=True,
        fbp_normalization_scale=math.pi / (2.0 * num_views),
    )


def build_fan_2d_case(
    *,
    image_shape: tuple[int, int] = (256, 256),
    num_views: int = 360,
    num_detectors: int = 600,
    detector_spacing: float = 1.0,
    voxel_spacing: float = 1.0,
    sid: float = 500.0,
    sdd: float = 800.0,
) -> ReconstructionCase:
    """Build a Shepp-Logan 2D fan-beam case."""
    ny, nx = image_shape
    reference = mx.array(shepp_logan_2d((ny, nx)))
    src_pos, det_center, det_u_vec = circular_trajectory_2d_fan(num_views, sid, sdd)
    sinogram = fan_forward(
        reference,
        src_pos,
        det_center,
        det_u_vec,
        num_detectors=num_detectors,
        detector_spacing=detector_spacing,
        voxel_spacing=voxel_spacing,
    )

    detector_coords = (mx.arange(num_detectors) - (num_detectors - 1) / 2) * detector_spacing
    cosine_weights = mx.cos(mx.arctan(detector_coords / sdd)).reshape(1, -1)

    forward_single, back_single, back_project_all = make_fan_2d_operators(
        src_pos,
        det_center,
        det_u_vec,
        image_shape=(ny, nx),
        num_detectors=num_detectors,
        detector_spacing=detector_spacing,
        voxel_spacing=voxel_spacing,
    )

    return ReconstructionCase(
        name="Fan 2D",
        sinogram=sinogram,
        volume_shape=(ny, nx),
        forward_single=forward_single,
        back_single=back_single,
        back_project_all=back_project_all,
        reference=reference,
        reference_title="Phantom",
        supports_fbp=True,
        fbp_normalization_scale=math.pi / (2.0 * num_views),
        fbp_weight=lambda raw: raw * cosine_weights,
    )


def build_cone_3d_case(
    *,
    volume_shape: tuple[int, int, int] = (128, 128, 128),
    num_views: int = 360,
    detector_shape: tuple[int, int] = (256, 256),
    du: float = 1.0,
    dv: float = 1.0,
    voxel_spacing: float = 1.0,
    sid: float = 600.0,
    sdd: float = 900.0,
) -> ReconstructionCase:
    """Build a Shepp-Logan 3D cone-beam case."""
    nz, ny, nx = volume_shape
    det_u_count, det_v_count = detector_shape
    reference = mx.array(shepp_logan_3d((nz, ny, nx)))
    src_pos, det_center, det_u_vec, det_v_vec = circular_trajectory_3d(num_views, sid, sdd)
    sinogram = cone_forward(
        reference,
        src_pos,
        det_center,
        det_u_vec,
        det_v_vec,
        det_u=det_u_count,
        det_v=det_v_count,
        du=du,
        dv=dv,
        voxel_spacing=voxel_spacing,
    )

    u_coords = (mx.arange(det_u_count) - (det_u_count - 1) / 2) * du
    v_coords = (mx.arange(det_v_count) - (det_v_count - 1) / 2) * dv
    fdk_weights = sdd / mx.sqrt(
        sdd**2 + u_coords.reshape(1, det_u_count, 1) ** 2 + v_coords.reshape(1, 1, det_v_count) ** 2
    )
    cone_weight = lambda raw: raw * fdk_weights

    forward_single, back_single, back_project_all = make_cone_3d_operators(
        src_pos,
        det_center,
        det_u_vec,
        det_v_vec,
        volume_shape=(nz, ny, nx),
        detector_shape=(det_u_count, det_v_count),
        du=du,
        dv=dv,
        voxel_spacing=voxel_spacing,
    )

    normalization = (math.pi * sid) / (2.0 * sdd * num_views)
    return ReconstructionCase(
        name="Cone 3D",
        sinogram=sinogram,
        volume_shape=(nz, ny, nx),
        forward_single=forward_single,
        back_single=back_single,
        back_project_all=back_project_all,
        reference=reference,
        reference_title="Phantom",
        supports_fbp=True,
        supports_fdk=True,
        fbp_normalization_scale=normalization,
        fdk_normalization_scale=normalization,
        fbp_weight=cone_weight,
        fdk_weight=cone_weight,
    )


def build_measured_cone_3d_case(config: MeasuredConeDataConfig) -> ReconstructionCase:
    """Build a 3D cone-beam case from measured TIFF projections and geometry."""
    data_dir = Path(config.data_dir)
    trajectory_json_path = Path(config.trajectory_json_path)
    reference_volume_path = None if config.reference_volume_path is None else Path(config.reference_volume_path)
    reference_meta_path = None if config.reference_meta_path is None else Path(config.reference_meta_path)

    nz, ny, nx = config.volume_shape
    target_det_u, target_det_v = config.target_detector_shape

    with trajectory_json_path.open("r", encoding="utf-8") as handle:
        geometry_payload = json.load(handle)

    src_pos, det_center, det_u_vec, det_v_vec = load_arbitrary_cone_geometry_from_json(
        trajectory_json_path,
        flip_det_u=config.flip_det_u,
        flip_det_v=config.flip_det_v,
        recenter_to_isocenter=config.recenter_to_isocenter,
    )

    view_stride = max(1, int(np.ceil(src_pos.shape[0] / config.target_view_count)))
    det_v_binning = max(1, int(np.ceil(geometry_payload["detector"]["num_pixels"]["v"] / target_det_v)))
    det_u_binning = max(1, int(np.ceil(geometry_payload["detector"]["num_pixels"]["u"] / target_det_u)))

    measured_sino_np = load_tiff_projections(
        data_dir,
        log_transform=config.log_transform,
        revert=config.revert,
        viewwise_i0=config.viewwise_i0,
        air_border_px=config.air_border_px,
        subtract_air_baseline=config.subtract_air_baseline,
        air_baseline_percentile=config.air_baseline_percentile,
        view_stride=view_stride,
        detector_binning_u=det_u_binning,
        detector_binning_v=det_v_binning,
        debug_visualization=False,
    )

    src_pos = src_pos[::view_stride]
    det_center = det_center[::view_stride]
    det_u_vec = det_u_vec[::view_stride]
    det_v_vec = det_v_vec[::view_stride]

    measured_du = float(geometry_payload["detector"]["pixel_size_mm"]["u"]) * det_u_binning
    measured_dv = float(geometry_payload["detector"]["pixel_size_mm"]["v"]) * det_v_binning
    header_offset_u_px = float(geometry_payload["detector"]["offset_px"].get("horizontal", 0.0) or 0.0) / det_u_binning
    header_offset_v_px = float(geometry_payload["detector"]["offset_px"].get("vertical", 0.0) or 0.0) / det_v_binning
    measured_det_v = measured_sino_np.shape[1]
    measured_det_u = measured_sino_np.shape[2]

    det_u_vec, det_v_vec, measured_du, measured_dv, measured_det_u, measured_det_v = apply_detector_geometry_convention(
        det_u_vec,
        det_v_vec,
        du=measured_du,
        dv=measured_dv,
        det_u=measured_det_u,
        det_v=measured_det_v,
        flip_u=config.flip_u,
        flip_v=config.flip_v,
        transpose_uv=config.transpose_uv,
    )
    header_offset_u_px, header_offset_v_px = transform_detector_offsets(
        header_offset_u_px,
        header_offset_v_px,
        {
            "flip_u": config.flip_u,
            "flip_v": config.flip_v,
            "transpose_uv": config.transpose_uv,
        },
    )
    det_center = shift_detector_center(
        det_center,
        det_u_vec,
        det_v_vec,
        measured_du,
        measured_dv,
        offset_u_px=header_offset_u_px,
        offset_v_px=header_offset_v_px,
    )

    reference_np = None
    reference_title = None
    resized_reference_voxel_spacing = None
    if reference_volume_path is not None and reference_meta_path is not None:
        if reference_volume_path.exists() and reference_meta_path.exists():
            reference_meta = json.loads(reference_meta_path.read_text())
            source_shape = tuple(int(value) for value in reference_meta["shape_zyx"])
            resize_factors = tuple(src / dst for src, dst in zip(source_shape, config.volume_shape))
            if max(resize_factors) - min(resize_factors) > 1e-6:
                raise ValueError(
                    f"Reference volume resize must stay isotropic, got factors {resize_factors}."
                )
            resized_reference_voxel_spacing = float(reference_meta["voxel_size_mm"]) * resize_factors[0]
            reference_np = resize_volume_to_shape(np.load(reference_volume_path), config.volume_shape)
            reference_np = normalize_volume(reference_np, upper_percentile=None)
            reference_title = "Reference Volume"

    if resized_reference_voxel_spacing is not None:
        measured_voxel_spacing = float(resized_reference_voxel_spacing)
    else:
        measured_voxel_spacing = auto_voxel_spacing_from_detector(
            config.volume_shape,
            (measured_det_u, measured_det_v),
            measured_du,
            measured_dv,
            magnification=float(geometry_payload["source"]["magnification"]),
            fov_margin_mm=config.measured_fov_margin_mm,
        )

    forward_single, back_single, back_project_all = make_cone_3d_operators(
        src_pos,
        det_center,
        det_u_vec,
        det_v_vec,
        volume_shape=(nz, ny, nx),
        detector_shape=(measured_det_u, measured_det_v),
        du=measured_du,
        dv=measured_dv,
        voxel_spacing=measured_voxel_spacing,
    )

    reference = None if reference_np is None else mx.array(reference_np, dtype=mx.float32)
    sinogram = mx.array(measured_sino_np, dtype=mx.float32)
    return ReconstructionCase(
        name="Measured Cone 3D",
        sinogram=sinogram,
        volume_shape=config.volume_shape,
        forward_single=forward_single,
        back_single=back_single,
        back_project_all=back_project_all,
        reference=reference,
        reference_title=reference_title,
    )
