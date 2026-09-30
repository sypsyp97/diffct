"""Real NCCL checks for distributed Projector forward and autograd paths.

Launch with torchrun. This runner only exercises the production Projector and
the existing raw CUDA autograd functions used as single-device references.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import socket
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

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


RTOL = 5e-4
ATOL = 5e-5
INIT_TIMEOUT = timedelta(seconds=120)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-cross-node", action="store_true")
    parser.add_argument("--expected-world-size", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _backend_blocker():
    if not torch.cuda.is_available():
        return "CUDA is unavailable"
    if not dist.is_available():
        return "torch.distributed is unavailable"
    if not dist.is_nccl_available():
        return "PyTorch NCCL support is unavailable"
    return None


def _slice_for_rank(n_views, world_size, rank):
    base, remainder = divmod(n_views, world_size)
    start = rank * base + min(rank, remainder)
    return start, start + base + int(rank < remainder)


def _distributed_any(failed, device):
    value = torch.tensor([int(failed)], dtype=torch.int32, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value.item())


def _record_tensor_error(errors, name, actual, expected):
    actual = actual.detach()
    expected = expected.detach()
    same_shape = tuple(actual.shape) == tuple(expected.shape)
    result = {
        "passed": False,
        "actual_shape": list(actual.shape),
        "reference_shape": list(expected.shape),
        "rtol": RTOL,
        "atol": ATOL,
        "max_abs": None,
        "reference_scale": None,
        "relative_error": None,
    }
    if same_shape:
        if actual.numel() == 0:
            result.update(
                passed=True,
                max_abs=0.0,
                reference_scale=0.0,
                relative_error=0.0,
            )
        else:
            actual32 = actual.to(dtype=torch.float32)
            expected32 = expected.to(dtype=torch.float32)
            difference = (actual32 - expected32).abs()
            scale = expected32.abs().max()
            if torch.isfinite(difference).all() and torch.isfinite(scale):
                max_abs = float(difference.max().item())
                reference_scale = float(scale.item())
                relative_error = (
                    max_abs / reference_scale if reference_scale > 0 else None
                )
                result.update(
                    passed=torch.allclose(
                        actual32, expected32, rtol=RTOL, atol=ATOL
                    ),
                    max_abs=max_abs,
                    reference_scale=reference_scale,
                    relative_error=relative_error,
                )
    errors[name] = result
    return result["passed"]


def _run_stage(record, device, name, operation):
    value = None
    local_failure = None
    try:
        value = operation()
    except Exception as error:  # keep every rank in the same test sequence
        local_failure = f"{type(error).__name__}: {error}"

    any_failure = _distributed_any(local_failure is not None, device)
    record["stages"][name] = {
        "passed": not any_failure,
        "error": local_failure,
    }
    if any_failure and local_failure is None:
        record["stages"][name]["error"] = "another rank failed this stage"
    if any_failure:
        record["passed"] = False
    return value, any_failure


def _beam_case(beam, n_views, device):
    if beam == "parallel":
        trajectory = circular_trajectory_2d_parallel(n_views, device=device)
        volume_shape = (6, 7)
        detector_shape = 5
        spacing = 1.25
        raw_project = lambda image: ParallelProjectorFunction.apply(
            image, *trajectory, detector_shape, spacing, 0.75
        )
        raw_backproject = lambda sino: ParallelBackprojectorFunction.apply(
            sino, *trajectory, spacing, *volume_shape, 0.75
        )
    elif beam == "fan":
        trajectory = circular_trajectory_2d_fan(
            n_views, sid=30.0, sdd=45.0, device=device
        )
        volume_shape = (6, 7)
        detector_shape = (5,)
        spacing = 1.25
        raw_project = lambda image: FanProjectorFunction.apply(
            image, *trajectory, detector_shape[0], spacing, 0.75
        )
        raw_backproject = lambda sino: FanBackprojectorFunction.apply(
            sino, *trajectory, spacing, *volume_shape, 0.75
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
        volume_shape = (4, 6, 7)
        detector_shape = (5, 3)
        spacing = (1.25, 0.8)
        raw_project = lambda volume: ConeProjectorFunction.apply(
            volume,
            *trajectory,
            detector_shape[0],
            detector_shape[1],
            *spacing,
            0.75,
        )
        raw_backproject = lambda sino: ConeBackprojectorFunction.apply(
            sino, *trajectory, *volume_shape, *spacing, 0.75
        )

    volume = torch.linspace(
        -0.1, 0.1, steps=math.prod(volume_shape), device=device
    ).reshape(volume_shape)
    volume.requires_grad_()
    projection_shape = (n_views, *(
        (detector_shape,) if isinstance(detector_shape, int) else detector_shape
    ))
    full_sinogram = torch.linspace(
        -0.08, 0.08, steps=math.prod(projection_shape), device=device
    ).reshape(projection_shape)
    detector_pixels = math.prod(projection_shape[1:])
    pixel_pattern = torch.linspace(
        0.4, 1.0, steps=detector_pixels, device=device
    ).reshape((1, *projection_shape[1:]))
    view_factors = torch.empty(n_views, device=device)
    for owner in range(dist.get_world_size()):
        start, stop = _slice_for_rank(n_views, dist.get_world_size(), owner)
        view_factors[start:stop] = float(owner + 1)
    full_projection_weight = view_factors.reshape(
        (n_views, *([1] * (len(projection_shape) - 1)))
    ) * pixel_pattern
    start, stop = _slice_for_rank(
        n_views, dist.get_world_size(), dist.get_rank()
    )
    local_weight = full_projection_weight[start:stop].contiguous()
    probe_volume = torch.linspace(
        0.05, 0.15, steps=math.prod(volume_shape), device=device
    ).reshape(volume_shape)
    projector = Projector(
        trajectory,
        volume_shape,
        detector_shape,
        beam=beam,
        detector_spacing=spacing,
        voxel_spacing=0.75,
        distributed=True,
    )
    return {
        "beam": beam,
        "n_views": n_views,
        "start": start,
        "stop": stop,
        "projection_shape": projection_shape,
        "volume": volume,
        "trajectory": trajectory,
        "full_sinogram": full_sinogram,
        "full_projection_weight": full_projection_weight,
        "local_weight": local_weight,
        "probe_volume": probe_volume,
        "projector": projector,
        "raw_project": raw_project,
        "raw_backproject": raw_backproject,
    }


def _check_metadata(case, record):
    projector = case["projector"]
    expected_slice = slice(case["start"], case["stop"])
    actual_slice = projector.view_slice
    passed = (
        projector.rank == dist.get_rank()
        and projector.world_size == dist.get_world_size()
        and actual_slice == expected_slice
    )
    record["metadata"] = {
        "passed": passed,
        "rank": projector.rank,
        "world_size": projector.world_size,
        "view_slice": [actual_slice.start, actual_slice.stop],
        "expected_view_slice": [case["start"], case["stop"]],
        "projection_shape": list(projector.projection_shape),
    }
    if not passed:
        record["passed"] = False


def _skip_case_metrics(case_record, from_stage, reason):
    remaining = {
        "project_forward": (
            "project_local",
            "project_volume_gradient",
            "backproject_replicated",
            "backproject_sinogram_gradient",
        ),
        "project_backward": (
            "project_volume_gradient",
            "backproject_replicated",
            "backproject_sinogram_gradient",
        ),
        "backproject_forward": (
            "backproject_replicated",
            "backproject_sinogram_gradient",
        ),
        "backproject_backward": ("backproject_sinogram_gradient",),
    }[from_stage]
    for name in remaining:
        case_record["errors"].setdefault(
            name, {"passed": False, "skipped": reason}
        )
    case_record["passed"] = False


def _run_case(beam, n_views, device, record):
    case_record = {
        "beam": beam,
        "global_views": n_views,
        "view_slice": None,
        "local_views": None,
        "errors": {},
        "stages": {},
        "backward_calls": {"project": False, "backproject": False},
        "rank_local_loss_weight": dist.get_rank() + 1,
        "passed": True,
    }
    case = None
    setup_error = None
    try:
        case = _beam_case(beam, n_views, device)
        case_record["view_slice"] = [case["start"], case["stop"]]
        case_record["local_views"] = case["stop"] - case["start"]
        _check_metadata(case, case_record)
    except Exception as error:
        setup_error = f"{type(error).__name__}: {error}"
    if _distributed_any(setup_error is not None, device):
        case_record["passed"] = False
        case_record["stages"]["setup"] = {
            "passed": False,
            "error": setup_error or "another rank failed setup",
        }
        _skip_case_metrics(case_record, "project_forward", "setup failed")
        record["passed"] = False
        record["cases"].append(case_record)
        return

    start, stop = case["start"], case["stop"]
    projection, failed = _run_stage(
        case_record,
        device,
        "project_forward",
        lambda: _project_and_compare(case, case_record),
    )
    if failed:
        _skip_case_metrics(case_record, "project_forward", "a rank failed")
        record["passed"] = False
        record["cases"].append(case_record)
        return

    def project_backward():
        loss = (projection * case["local_weight"]).sum()
        case_record["rank_local_projection_loss"] = float(loss.detach().item())
        loss.backward()
        case_record["backward_calls"]["project"] = True
        reference_volume = case["volume"].detach().clone().requires_grad_()
        reference_projection = case["raw_project"](reference_volume)
        (reference_projection * case["full_projection_weight"]).sum().backward()
        _record_tensor_error(
            case_record["errors"],
            "project_volume_gradient",
            case["volume"].grad,
            reference_volume.grad,
        )

    _, failed = _run_stage(
        case_record, device, "project_backward", project_backward
    )
    if failed:
        _skip_case_metrics(case_record, "project_backward", "a rank failed")
        record["passed"] = False
        record["cases"].append(case_record)
        return

    local_sinogram = case["full_sinogram"][start:stop].clone().requires_grad_()

    def backproject_forward():
        backprojection = case["projector"].backproject(local_sinogram)
        reference = case["raw_backproject"](case["full_sinogram"])
        _record_tensor_error(
            case_record["errors"], "backproject_replicated", backprojection, reference
        )
        return backprojection

    backprojection, failed = _run_stage(
        case_record,
        device,
        "backproject_forward",
        backproject_forward,
    )
    if failed:
        _skip_case_metrics(case_record, "backproject_forward", "a rank failed")
        record["passed"] = False
        record["cases"].append(case_record)
        return

    def backproject_backward():
        rank_weight = float(dist.get_rank() + 1)
        rank_scale = rank_weight / dist.get_world_size()
        case_record["rank_image_loss_weight"] = rank_weight
        case_record["rank_image_loss_scale"] = rank_scale
        weighted_probe = case["probe_volume"] * rank_weight
        loss = (backprojection * weighted_probe).sum() / dist.get_world_size()
        case_record["replicated_image_loss"] = float(loss.detach().item())
        loss.backward()
        case_record["backward_calls"]["backproject"] = True
        mean_rank_weight = (dist.get_world_size() + 1) / 2.0
        case_record["reference_image_cotangent_scale"] = mean_rank_weight
        reference_gradient = case["raw_project"](
            case["probe_volume"] * mean_rank_weight
        )
        _record_tensor_error(
            case_record["errors"],
            "backproject_sinogram_gradient",
            local_sinogram.grad,
            reference_gradient[start:stop],
        )
        if start == stop:
            zero_sized = (
                local_sinogram.grad is not None
                and local_sinogram.grad.shape == local_sinogram.shape
                and local_sinogram.grad.numel() == 0
            )
            case_record["errors"]["empty_sinogram_gradient_zero"] = {
                "passed": zero_sized,
                "shape": list(
                    local_sinogram.grad.shape
                    if local_sinogram.grad is not None
                    else ()
                ),
                "numel": (
                    local_sinogram.grad.numel()
                    if local_sinogram.grad is not None
                    else None
                ),
            }
            if not zero_sized:
                case_record["passed"] = False

    _, failed = _run_stage(
        case_record, device, "backproject_backward", backproject_backward
    )
    if failed:
        _skip_case_metrics(case_record, "backproject_backward", "a rank failed")
        record["passed"] = False
    if not all(item.get("passed", False) for item in case_record["errors"].values()):
        case_record["passed"] = False
    record["passed"] = record["passed"] and case_record["passed"]
    record["cases"].append(case_record)


def _project_and_compare(case, case_record):
    projection = case["projector"].project(case["volume"])
    reference = case["raw_project"](case["volume"].detach())
    _record_tensor_error(
        case_record["errors"],
        "project_local",
        projection,
        reference[case["start"] : case["stop"]],
    )
    return projection


def _write_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _check_distributed(args):
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
    missing = [key for key in required if key not in os.environ]
    if missing:
        raise RuntimeError(
            "launch this runner with torchrun; missing environment: "
            + ", ".join(missing)
        )
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if args.expected_world_size is not None and world_size != args.expected_world_size:
        raise RuntimeError(
            f"expected world size {args.expected_world_size}, got {world_size}"
        )
    if world_size < 2:
        raise RuntimeError("distributed projector checks require world size >= 2")
    if rank < 0 or rank >= world_size:
        raise RuntimeError(f"rank {rank} is outside world size {world_size}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled")
    if not dist.is_available() or not dist.is_nccl_available():
        raise RuntimeError("PyTorch with NCCL support is required; Gloo fallback is disabled")
    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} has no matching local CUDA device "
            f"(device_count={torch.cuda.device_count()})"
        )
    return rank, world_size, local_rank


def main(argv=None):
    args = _parse_args(argv)
    blocker = _backend_blocker()
    if blocker is not None:
        reason = f"{blocker}; no CPU/Gloo fallback is used"
        if args.require_cuda or args.require_cross_node:
            print(f"ERROR: {reason}", file=sys.stderr)
            return 2
        print(f"SKIP: {reason}")
        return 0

    try:
        rank, world_size, local_rank = _check_distributed(args)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    try:
        dist.init_process_group(
            backend="nccl", init_method="env://", timeout=INIT_TIMEOUT
        )
    except Exception as error:
        print(f"ERROR: NCCL process-group initialization failed: {error}", file=sys.stderr)
        return 1

    host = socket.gethostname()
    hosts = [None] * world_size
    local_record = {
        "host": host,
        "rank": rank,
        "local_gpu_index": local_rank,
        "local_gpu_name": torch.cuda.get_device_name(local_rank),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "numba_versions": {
            "numba": importlib.metadata.version("numba"),
            "numba-cuda": importlib.metadata.version("numba-cuda"),
        },
        "world_size": world_size,
        "distinct_node_count": None,
        "passed": True,
        "errors": {},
        "stages": {},
        "cases": [],
    }
    final_result = None
    output_error = None
    try:
        dist.all_gather_object(hosts, host)
        node_count = len(set(hosts))
        local_record["distinct_node_count"] = node_count
        local_record["nodes"] = sorted(set(hosts))
        n_uneven = world_size * 2 + 1
        if args.require_cross_node and node_count < 2:
            local_record["errors"]["cross_node_requirement"] = {
                "passed": False,
                "error": (
                    "--require-cross-node needs at least two distinct hosts; "
                    f"observed {node_count}"
                ),
            }
            local_record["passed"] = False
        else:
            for beam in ("parallel", "fan", "cone"):
                for n_views in (n_uneven, 1):
                    _run_case(beam, n_views, device, local_record)

        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_record)
        passed = all(item["passed"] for item in gathered)
        final_result = {
            "passed": passed,
            "backend": "nccl",
            "world_size": world_size,
            "distinct_node_count": node_count,
            "nodes": sorted(set(hosts)),
            "checks": {
                "rtol": RTOL,
                "atol": ATOL,
                "global_view_counts": [n_uneven, 1],
                "beams": ["parallel", "fan", "cone"],
                "init_timeout_seconds": int(INIT_TIMEOUT.total_seconds()),
            },
            "ranks": gathered,
        }
        if rank == 0 and args.output is not None:
            try:
                _write_result(args.output, final_result)
            except Exception as error:
                output_error = f"{type(error).__name__}: {error}"
        write_status = [output_error]
        dist.broadcast_object_list(write_status, src=0)
        output_error = write_status[0]
        if output_error:
            final_result["passed"] = False
            final_result["output_error"] = output_error
        if rank == 0:
            print(json.dumps(final_result, indent=2, sort_keys=True, allow_nan=False))
        return 0 if final_result["passed"] else 1
    except Exception as error:
        # A per-stage failure is synchronized above. This path covers setup or
        # reporting failures; the process-group timeout bounds peer waits.
        local_record["passed"] = False
        local_record["errors"]["runner"] = {
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
        }
        print(
            f"rank {rank}: NCCL check failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
