"""Utility script to visualise trajectory generators from diffct.geometry.

This script provides a command-line interface to plot individual trajectories,
and when invoked without arguments it generates a gallery of common trajectories
under the ``plots/`` directory.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
GEOMETRY_PATH = ROOT_DIR / "diffct" / "geometry.py"
GEOMETRY_SPEC = importlib.util.spec_from_file_location("diffct_geometry", GEOMETRY_PATH)
if GEOMETRY_SPEC is None or GEOMETRY_SPEC.loader is None:
    raise RuntimeError(f"Unable to load geometry module from {GEOMETRY_PATH}")
geometry = importlib.util.module_from_spec(GEOMETRY_SPEC)
GEOMETRY_SPEC.loader.exec_module(geometry)


# Mapping from CLI name to geometry function and argument mapping.
_TRAJECTORY_SPECS: Dict[str, Tuple] = {
    "circular3d": (
        geometry.circular_trajectory_3d,
        {
            "n_views": "n_views",
            "sid": "sid",
            "sdd": "sdd",
            "start_angle": "start_angle",
            "end_angle": "end_angle",
        },
    ),
    "spiral3d": (
        geometry.spiral_trajectory_3d,
        {
            "n_views": "n_views",
            "sid": "sid",
            "sdd": "sdd",
            "z_range": "z_range",
            "n_turns": "n_turns",
        },
    ),
    "sinusoidal3d": (
        geometry.sinusoidal_trajectory_3d,
        {
            "n_views": "n_views",
            "sid": "sid",
            "sdd": "sdd",
            "amplitude": "amplitude",
            "frequency": "frequency",
        },
    ),
    "saddle3d": (
        geometry.saddle_trajectory_3d,
        {
            "n_views": "n_views",
            "sid": "sid",
            "sdd": "sdd",
            "z_amplitude": "z_amplitude",
            "radial_amplitude": "radial_amplitude",
            "start_angle": "start_angle",
        },
    ),
    "random3d": (
        geometry.random_trajectory_3d,
        {
            "n_views": "n_views",
            "sid": "sid_mean",
            "sdd": "sdd_mean",
            "sid_std": "sid_std",
            "pos_std": "pos_std",
            "angle_std": "angle_std",
            "seed": "seed",
        },
    ),
}

_DEFAULT_JOBS: List[Dict[str, object]] = [
    {
        "trajectory": "circular3d",
        "output": "circular3d.png",
        "kwargs": {"n_views": 360, "sid": 1000.0, "sdd": 1500.0},
    },
    {
        "trajectory": "spiral3d",
        "output": "spiral3d.png",
        "kwargs": {"n_views": 720, "sid": 1000.0, "sdd": 1500.0, "z_range": 200.0, "n_turns": 3.0},
    },
    {
        "trajectory": "sinusoidal3d",
        "output": "sinusoidal3d.png",
        "kwargs": {"n_views": 360, "sid": 1000.0, "sdd": 1500.0, "amplitude": 60.0, "frequency": 2.5},
        "show_detectors": True,
        "sample_step": 20,
        "arrow_scale": 30.0,
    },
    {
        "trajectory": "saddle3d",
        "output": "saddle3d.png",
        "kwargs": {"n_views": 360, "sid": 1000.0, "sdd": 1500.0, "z_amplitude": 80.0, "radial_amplitude": 40.0},
    },
    {
        "trajectory": "random3d",
        "output": "random3d.png",
        "kwargs": {
            "n_views": 360,
            "sid": 1000.0,
            "sdd": 1500.0,
            "sid_std": 80.0,
            "pos_std": 10.0,
            "angle_std": 0.05,
            "seed": 42,
        },
        "show_detectors": True,
        "sample_step": 30,
        "arrow_scale": 25.0,
    },
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        choices=sorted(_TRAJECTORY_SPECS.keys()),
        default="circular3d",
        help="Trajectory generator to visualise.",
    )
    parser.add_argument("--n-views", type=int, default=360, help="Number of projection views to sample.")
    parser.add_argument("--sid", type=float, default=1000.0, help="Source-to-isocenter distance.")
    parser.add_argument("--sdd", type=float, default=1500.0, help="Source-to-detector distance.")
    parser.add_argument("--start-angle", type=float, default=0.0, help="Starting angle in radians.")
    parser.add_argument(
        "--end-angle",
        type=float,
        default=None,
        help="Optional end angle in radians (defaults to a full rotation when omitted).",
    )
    parser.add_argument("--z-range", type=float, default=100.0, help="Axial extent for spiral trajectories.")
    parser.add_argument("--n-turns", type=float, default=2.0, help="Number of turns for spiral trajectories.")
    parser.add_argument("--amplitude", type=float, default=50.0, help="Amplitude for sinusoidal trajectories.")
    parser.add_argument("--frequency", type=float, default=2.0, help="Frequency for sinusoidal trajectories.")
    parser.add_argument("--z-amplitude", type=float, default=50.0, help="Vertical amplitude for saddle trajectories.")
    parser.add_argument(
        "--radial-amplitude",
        type=float,
        default=30.0,
        help="Radial amplitude for saddle trajectories.",
    )
    parser.add_argument("--sid-std", type=float, default=0.0, help="SID variation for random trajectories.")
    parser.add_argument("--pos-std", type=float, default=0.0, help="Positional noise for random trajectories.")
    parser.add_argument("--angle-std", type=float, default=0.0, help="Angular noise for random trajectories.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for random trajectories.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device to use for generating the trajectory.",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=25,
        help="Stride for plotting detector orientation arrows.",
    )
    parser.add_argument(
        "--arrow-scale",
        type=float,
        default=50.0,
        help="Scale factor for detector orientation arrows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file path to save the figure instead of displaying it.",
    )
    parser.add_argument(
        "--show-detectors",
        action="store_true",
        help="Also plot detector orientation arrows along the trajectory.",
    )
    return parser.parse_args()


def _build_kwargs(param_values: Dict[str, object], param_map: Dict[str, str], *, device: str, dtype: torch.dtype) -> Dict[str, object]:
    kwargs: Dict[str, object] = {"device": device, "dtype": dtype}
    for arg_name, param_name in param_map.items():
        if arg_name not in param_values:
            continue
        value = param_values[arg_name]
        if value is None:
            continue
        kwargs[param_name] = value
    return kwargs


def _set_equal_axes(ax: plt.Axes, *arrays: np.ndarray) -> None:
    stacked = np.vstack(arrays)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    centers = 0.5 * (mins + maxs)
    max_range = 0.5 * (maxs - mins).max()
    if max_range == 0.0:
        max_range = 1.0
    ax.set_xlim(centers[0] - max_range, centers[0] + max_range)
    ax.set_ylim(centers[1] - max_range, centers[1] + max_range)
    ax.set_zlim(centers[2] - max_range, centers[2] + max_range)


def _plot_detector_orientations(
    ax: plt.Axes,
    det_center: np.ndarray,
    det_u_vec: np.ndarray,
    det_v_vec: np.ndarray,
    step: int,
    scale: float,
) -> None:
    step = max(1, step)
    for idx in range(0, len(det_center), step):
        center = det_center[idx]
        u_vec = det_u_vec[idx] * scale
        v_vec = det_v_vec[idx] * scale
        ax.quiver(*center, *u_vec, color="tab:orange", length=1.0, normalize=False)
        ax.quiver(*center, *v_vec, color="tab:green", length=1.0, normalize=False)


def _generate_plot(
    trajectory: str,
    *,
    param_values: Dict[str, object],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    show_detectors: bool = False,
    sample_step: int = 25,
    arrow_scale: float = 50.0,
    output: Path | None = None,
) -> Path | None:
    fn, param_map = _TRAJECTORY_SPECS[trajectory]
    kwargs = _build_kwargs(param_values, param_map, device=device, dtype=dtype)
    src_pos, det_center, det_u_vec, det_v_vec = fn(**kwargs)

    src = src_pos.detach().cpu().numpy()
    det = det_center.detach().cpu().numpy()
    det_u = det_u_vec.detach().cpu().numpy()
    det_v = det_v_vec.detach().cpu().numpy()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(src[:, 0], src[:, 1], src[:, 2], label="Source path", color="tab:blue")
    ax.plot(det[:, 0], det[:, 1], det[:, 2], label="Detector path", color="tab:red")

    if show_detectors:
        _plot_detector_orientations(ax, det, det_u, det_v, sample_step, arrow_scale)

    ax.scatter([0], [0], [0], color="black", marker="x", s=50, label="Isocenter")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{trajectory} trajectory")
    ax.legend()
    _set_equal_axes(ax, src, det)
    fig.tight_layout()

    saved_path: Path | None = None
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        saved_path = output
    else:
        plt.show()
    plt.close(fig)
    return saved_path


def _run_cli() -> None:
    args = _parse_args()
    param_names = _TRAJECTORY_SPECS[args.trajectory][1].keys()
    param_values = {name: getattr(args, name) for name in param_names}
    saved_path = _generate_plot(
        args.trajectory,
        param_values=param_values,
        device=args.device,
        show_detectors=args.show_detectors,
        sample_step=args.sample_step,
        arrow_scale=args.arrow_scale,
        output=args.output,
    )
    if saved_path is not None:
        print(f"Saved figure to {saved_path}")


def _run_all() -> None:
    gallery_dir = ROOT_DIR / "plots"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    for job in _DEFAULT_JOBS:
        trajectory = job["trajectory"]
        param_values = dict(job.get("kwargs", {}))
        output_path = gallery_dir / job["output"]
        saved_path = _generate_plot(
            trajectory,
            param_values=param_values,
            device=str(job.get("device", "cpu")),
            show_detectors=bool(job.get("show_detectors", False)),
            sample_step=int(job.get("sample_step", 25)),
            arrow_scale=float(job.get("arrow_scale", 50.0)),
            output=output_path,
        )
        if saved_path is not None:
            print(f"[✓] {trajectory} saved to {saved_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _run_cli()
    else:
        _run_all()
