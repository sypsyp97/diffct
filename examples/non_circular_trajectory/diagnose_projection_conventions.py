#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

import diffct_mlx
from diffct_mlx.real_measured_data_helper import load_tiff_projections

try:
    _load_arbitrary_cone_geometry_from_json = diffct_mlx.load_arbitrary_cone_geometry_from_json
except AttributeError:
    from diffct_mlx.geometry import load_arbitrary_cone_geometry_from_json as _load_arbitrary_cone_geometry_from_json


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GEOMETRY_JSON = SCRIPT_DIR / "sample_data" / "sim_obj_1_tif" / "sim_obj_1_geometry_diffct.json"
DEFAULT_REFERENCE_VOLUME = SCRIPT_DIR / "sample_data" / "reko" / "sim_obj_1_diffct.npy"
DEFAULT_REFERENCE_META = SCRIPT_DIR / "sample_data" / "reko" / "sim_obj_1_diffct.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "diagnostics" / "sim_obj_1_projection_convention"
FIT_TIE_ABS_TOL = 1e-4
FORWARD_GEOMETRY_CONVENTION = {
    "transpose_uv": True,
    "flip_u": False,
    "flip_v": False,
}
IDENTITY_DETECTOR_CONVENTION = {
    "transpose_uv": False,
    "flip_u": False,
    "flip_v": False,
    "delta_u_px": 0.0,
    "delta_v_px": 0.0,
}


def _center_crop_to_factor_1d(length: int, factor: int) -> tuple[int, int]:
    eff = (length // factor) * factor
    start = max(0, (length - eff) // 2)
    return start, start + eff


def downsample_volume_mean(volume: np.ndarray, factor: int) -> np.ndarray:
    factor = max(1, int(factor))
    volume = np.asarray(volume, dtype=np.float32)
    if factor == 1:
        return volume

    depth, height, width = volume.shape
    z0, z1 = _center_crop_to_factor_1d(depth, factor)
    y0, y1 = _center_crop_to_factor_1d(height, factor)
    x0, x1 = _center_crop_to_factor_1d(width, factor)
    volume = volume[z0:z1, y0:y1, x0:x1]
    depth, height, width = volume.shape

    return volume.reshape(
        depth // factor,
        factor,
        height // factor,
        factor,
        width // factor,
        factor,
    ).mean(axis=(1, 3, 5), dtype=np.float32)


def subtract_reference_background(volume: np.ndarray, percentile: float) -> tuple[np.ndarray, float]:
    volume = np.asarray(volume, dtype=np.float32)
    percentile = float(percentile)
    if percentile <= 0.0:
        return volume, 0.0
    baseline = float(np.percentile(volume, percentile))
    if baseline <= 0.0:
        return volume, baseline
    return np.maximum(volume - baseline, 0.0), baseline


def transform_detector_offsets(horizontal_px: float, vertical_px: float, config: dict[str, bool]) -> tuple[float, float]:
    offset_u_px = float(horizontal_px)
    offset_v_px = float(vertical_px)
    if config["flip_u"]:
        offset_u_px *= -1.0
    if config["flip_v"]:
        offset_v_px *= -1.0
    if config["transpose_uv"]:
        offset_u_px, offset_v_px = offset_v_px, offset_u_px
    return offset_u_px, offset_v_px


def shift_detector_center(
    det_center,
    det_u_vec,
    det_v_vec,
    du: float,
    dv: float,
    *,
    offset_u_px: float = 0.0,
    offset_v_px: float = 0.0,
) -> np.ndarray:
    det_center_np = np.asarray(det_center, dtype=np.float32)
    det_u_np = np.asarray(det_u_vec, dtype=np.float32)
    det_v_np = np.asarray(det_v_vec, dtype=np.float32)
    return (
        det_center_np
        + float(offset_u_px * du) * det_u_np
        + float(offset_v_px * dv) * det_v_np
    ).astype(np.float32, copy=False)


def fit_affine(reference: np.ndarray, measured: np.ndarray, sample_step: int = 1) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.float64).ravel()[::sample_step]
    mea = np.asarray(measured, dtype=np.float64).ravel()[::sample_step]

    design = np.column_stack([ref, np.ones_like(ref)])
    scale, bias = np.linalg.lstsq(design, mea, rcond=None)[0]
    fitted = scale * ref + bias
    residual = fitted - mea

    ref_std = float(np.std(ref))
    mea_std = float(np.std(mea))
    if ref_std <= 0.0 or mea_std <= 0.0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(ref, mea)[0, 1])

    return {
        "scale": float(scale),
        "bias": float(bias),
        "mse_affine": float(np.mean(residual ** 2)),
        "mae_affine": float(np.mean(np.abs(residual))),
        "corrcoef": corr,
        "reference_mean": float(np.mean(ref)),
        "measured_mean": float(np.mean(mea)),
    }


def candidate_preference_key(config: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        0 if config["transpose_uv"] and not config["flip_u"] and not config["flip_v"] else 1,
        int(bool(config["flip_u"])) + int(bool(config["flip_v"])),
        0 if config["transpose_uv"] else 1,
        int(bool(config["flip_u"])),
        int(bool(config["flip_v"])),
    )


def result_sort_key(item: dict[str, Any]) -> tuple[float, tuple[int, int, int, int, int]]:
    return float(item["mse_affine"]), candidate_preference_key(item)


def is_better_result(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    candidate_mse = float(candidate["mse_affine"])
    incumbent_mse = float(incumbent["mse_affine"])
    if candidate_mse < incumbent_mse - FIT_TIE_ABS_TOL:
        return True
    if abs(candidate_mse - incumbent_mse) <= FIT_TIE_ABS_TOL:
        return candidate_preference_key(candidate) < candidate_preference_key(incumbent)
    return False


def candidate_configs(offset_values_px: list[float]) -> list[dict[str, Any]]:
    del offset_values_px
    return [dict(IDENTITY_DETECTOR_CONVENTION)]


def build_voxel_spacing_values_mm(args: argparse.Namespace, base_voxel_spacing_mm: float) -> list[float]:
    if args.voxel_spacing_values_mm:
        values = [float(v) for v in args.voxel_spacing_values_mm]
    else:
        values = [base_voxel_spacing_mm * float(scale) for scale in args.voxel_spacing_scale_values]

    deduped: list[float] = []
    seen: set[float] = set()
    for value in values:
        if value <= 0.0:
            raise ValueError(f"voxel spacing must be positive, got {value}")
        rounded = round(value, 9)
        if rounded in seen:
            continue
        seen.add(rounded)
        deduped.append(float(value))
    return deduped


def candidate_label(config: dict[str, Any], offset_u_px: float, offset_v_px: float) -> str:
    return (
        f"voxel_spacing_mm={config['voxel_spacing_mm']:.6f}, "
        f"transpose_uv={config['transpose_uv']}, "
        f"flip_u={config['flip_u']}, "
        f"flip_v={config['flip_v']}, "
        f"offset_u_px={offset_u_px:+.2f}, "
        f"offset_v_px={offset_v_px:+.2f}"
    )


def save_best_preview(
    *,
    output_path: Path,
    measured: np.ndarray,
    forward: np.ndarray,
    scale: float,
    bias: float,
    title: str,
) -> None:
    fitted = scale * forward + bias
    diff = fitted - measured
    view_indices = sorted({0, measured.shape[0] // 2, measured.shape[0] - 1})
    display_vmin = float(min(np.percentile(measured, 1.0), np.percentile(fitted, 1.0)))
    display_vmax = float(max(np.percentile(measured, 99.5), np.percentile(fitted, 99.5)))
    diff_abs_max = float(np.percentile(np.abs(diff), 99.5))
    if diff_abs_max <= 0.0:
        diff_abs_max = float(np.max(np.abs(diff))) if diff.size else 1.0

    fig, axes = plt.subplots(3, len(view_indices), figsize=(4 * len(view_indices), 10))
    for col, view_idx in enumerate(view_indices):
        axes[0, col].imshow(
            measured[view_idx].T,
            cmap="gray",
            origin="lower",
            vmin=display_vmin,
            vmax=display_vmax,
        )
        axes[0, col].set_title(f"Measured view {view_idx}")
        axes[0, col].axis("off")

        axes[1, col].imshow(
            fitted[view_idx].T,
            cmap="gray",
            origin="lower",
            vmin=display_vmin,
            vmax=display_vmax,
        )
        axes[1, col].set_title(f"Fitted forward view {view_idx}")
        axes[1, col].axis("off")

        axes[2, col].imshow(
            diff[view_idx].T,
            cmap="bwr",
            origin="lower",
            vmin=-diff_abs_max,
            vmax=diff_abs_max,
        )
        axes[2, col].set_title(f"Residual view {view_idx}")
        axes[2, col].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose detector/image convention mismatches by forward-projecting a reference volume and comparing it to measured TIFF projections."
    )
    parser.add_argument("--geometry-json", type=Path, default=DEFAULT_GEOMETRY_JSON)
    parser.add_argument("--projection-dir", type=Path, default=None)
    parser.add_argument("--reference-volume", type=Path, default=DEFAULT_REFERENCE_VOLUME)
    parser.add_argument("--reference-meta", type=Path, default=DEFAULT_REFERENCE_META)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--view-stride", type=int, default=4)
    parser.add_argument("--max-views", type=int, default=200)
    parser.add_argument("--detector-binning", type=int, default=1)
    parser.add_argument("--volume-binning", type=int, default=2)
    parser.add_argument("--voxel-spacing-scale-values", type=float, nargs="+", default=[1.0])
    parser.add_argument("--voxel-spacing-values-mm", type=float, nargs="+", default=None)
    parser.add_argument("--reference-background-percentile", type=float, default=0.5)
    parser.add_argument("--i0-percentile", type=float, default=99.9)
    parser.add_argument("--offset-search-values", type=float, nargs="+", default=[0.0])
    parser.add_argument("--flip-det-u", action="store_true")
    parser.add_argument("--flip-det-v", action="store_true")
    parser.add_argument("--no-recenter", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    geometry_json = args.geometry_json.resolve()
    reference_volume_path = args.reference_volume.resolve()
    reference_meta_path = args.reference_meta.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry_payload = json.loads(geometry_json.read_text())
    reference_meta = json.loads(reference_meta_path.read_text())
    projection_dir = args.projection_dir.resolve() if args.projection_dir is not None else geometry_json.parent

    volume_np = np.load(reference_volume_path).astype(np.float32, copy=False)
    volume_np = downsample_volume_mean(volume_np, int(args.volume_binning))
    volume_np, reference_background_baseline = subtract_reference_background(
        volume_np,
        percentile=float(args.reference_background_percentile),
    )
    voxel_spacing_mm = float(reference_meta["voxel_size_mm"]) * max(1, int(args.volume_binning))
    voxel_spacing_values_mm = build_voxel_spacing_values_mm(args, voxel_spacing_mm)
    volume_mx = mx.array(volume_np, dtype=mx.float32)

    src_pos, det_center, det_u_vec, det_v_vec = _load_arbitrary_cone_geometry_from_json(
        geometry_json,
        flip_det_u=bool(args.flip_det_u),
        flip_det_v=bool(args.flip_det_v),
        recenter_to_isocenter=not bool(args.no_recenter),
    )

    view_stride = max(1, int(args.view_stride))
    detector_binning = max(1, int(args.detector_binning))
    measured_sino = load_tiff_projections(
        projection_dir,
        log_transform=True,
        i0_percentile=float(args.i0_percentile),
        view_stride=view_stride,
        detector_binning_u=detector_binning,
        detector_binning_v=detector_binning,
    )

    src_pos = src_pos[::view_stride]
    det_center = det_center[::view_stride]
    det_u_vec = det_u_vec[::view_stride]
    det_v_vec = det_v_vec[::view_stride]

    if args.max_views is not None and int(args.max_views) > 0:
        max_views = min(int(args.max_views), measured_sino.shape[0], src_pos.shape[0])
        measured_sino = measured_sino[:max_views]
        src_pos = src_pos[:max_views]
        det_center = det_center[:max_views]
        det_u_vec = det_u_vec[:max_views]
        det_v_vec = det_v_vec[:max_views]

    det_pitch_u_mm = float(geometry_payload["detector"]["pixel_size_mm"]["u"]) * detector_binning
    det_pitch_v_mm = float(geometry_payload["detector"]["pixel_size_mm"]["v"]) * detector_binning
    header_offset_u_px = float(geometry_payload["detector"]["offset_px"].get("horizontal", 0.0) or 0.0) / detector_binning
    header_offset_v_px = float(geometry_payload["detector"]["offset_px"].get("vertical", 0.0) or 0.0) / detector_binning
    forward_u_vec, forward_v_vec, forward_du, forward_dv, forward_det_u, forward_det_v = diffct_mlx.apply_detector_geometry_convention(
        det_u_vec,
        det_v_vec,
        du=det_pitch_u_mm,
        dv=det_pitch_v_mm,
        det_u=measured_sino.shape[1],
        det_v=measured_sino.shape[2],
        flip_u=FORWARD_GEOMETRY_CONVENTION["flip_u"],
        flip_v=FORWARD_GEOMETRY_CONVENTION["flip_v"],
        transpose_uv=FORWARD_GEOMETRY_CONVENTION["transpose_uv"],
    )
    forward_offset_u_px, forward_offset_v_px = transform_detector_offsets(
        header_offset_u_px,
        header_offset_v_px,
        FORWARD_GEOMETRY_CONVENTION,
    )

    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for base_config in candidate_configs([float(v) for v in args.offset_search_values]):
        for trial_voxel_spacing_mm in voxel_spacing_values_mm:
            config = {
                **base_config,
                "voxel_spacing_mm": float(trial_voxel_spacing_mm),
            }
            trial_sino = measured_sino
            trial_u_vec = forward_u_vec
            trial_v_vec = forward_v_vec
            trial_du = forward_du
            trial_dv = forward_dv
            offset_u_px = forward_offset_u_px + float(config["delta_u_px"])
            offset_v_px = forward_offset_v_px + float(config["delta_v_px"])
            trial_det_center = shift_detector_center(
                det_center,
                trial_u_vec,
                trial_v_vec,
                trial_du,
                trial_dv,
                offset_u_px=offset_u_px,
                offset_v_px=offset_v_px,
            )

            forward_sino = diffct_mlx.cone_forward(
                volume_mx,
                src_pos,
                trial_det_center,
                trial_u_vec,
                trial_v_vec,
                det_u=forward_det_u,
                det_v=forward_det_v,
                du=trial_du,
                dv=trial_dv,
                voxel_spacing=float(config["voxel_spacing_mm"]),
            )
            mx.eval(forward_sino)
            forward_sino_np = np.array(forward_sino)

            stats = fit_affine(forward_sino_np, trial_sino)
            result = {
                "label": candidate_label(config, offset_u_px, offset_v_px),
                "voxel_spacing_mm": float(config["voxel_spacing_mm"]),
                "voxel_spacing_scale": float(config["voxel_spacing_mm"]) / voxel_spacing_mm,
                "transpose_uv": bool(config["transpose_uv"]),
                "flip_u": bool(config["flip_u"]),
                "flip_v": bool(config["flip_v"]),
                "offset_u_px": float(offset_u_px),
                "offset_v_px": float(offset_v_px),
                "detector_shape_nuv": [int(trial_sino.shape[0]), int(trial_sino.shape[1]), int(trial_sino.shape[2])],
                "detector_pitch_mm": {"u": float(trial_du), "v": float(trial_dv)},
                **stats,
            }
            results.append(result)
            if is_better_result(result, best):
                best = {
                    **result,
                    "_measured": trial_sino,
                    "_forward": forward_sino_np,
                }
            print(f"{result['label']} -> mse_affine={result['mse_affine']:.6f}, corr={result['corrcoef']:.4f}")

    assert best is not None
    results.sort(key=result_sort_key)

    summary = {
        "geometry_json": str(geometry_json),
        "projection_dir": str(projection_dir),
        "reference_volume": str(reference_volume_path),
        "reference_meta": str(reference_meta_path),
        "volume_shape_zyx": [int(x) for x in volume_np.shape],
        "voxel_spacing_mm": float(voxel_spacing_mm),
        "voxel_spacing_values_mm": [float(v) for v in voxel_spacing_values_mm],
        "reference_background_percentile": float(args.reference_background_percentile),
        "reference_background_baseline": float(reference_background_baseline),
        "view_stride": int(view_stride),
        "max_views": None if args.max_views is None else int(args.max_views),
        "detector_binning": int(detector_binning),
        "header_offset_px_binned": {
            "u": float(header_offset_u_px),
            "v": float(header_offset_v_px),
        },
        "forward_geometry_convention": {
            "transpose_uv": bool(FORWARD_GEOMETRY_CONVENTION["transpose_uv"]),
            "flip_u": bool(FORWARD_GEOMETRY_CONVENTION["flip_u"]),
            "flip_v": bool(FORWARD_GEOMETRY_CONVENTION["flip_v"]),
            "offset_u_px": float(forward_offset_u_px),
            "offset_v_px": float(forward_offset_v_px),
        },
        "candidate_count": len(results),
        "best_candidate": {k: v for k, v in best.items() if not k.startswith("_")},
        "candidates": results,
    }

    summary_path = output_dir / "diagnosis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    save_best_preview(
        output_path=output_dir / "best_candidate_preview.png",
        measured=best["_measured"],
        forward=best["_forward"],
        scale=float(best["scale"]),
        bias=float(best["bias"]),
        title=best["label"],
    )

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote preview: {output_dir / 'best_candidate_preview.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
