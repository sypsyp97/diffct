#!/usr/bin/env python3
"""Observe real diffct calls. Instrumentation is confined to this process."""

import argparse
from copy import copy
from datetime import timedelta
import functools
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import uuid


class Recorder:
    def __init__(self, path, rank):
        self.output = path.open("w", buffering=1)
        self.rank = rank
        self.lock = threading.Lock()
        self.local = threading.local()
        self.bounded = 0
        self.phase = "forward"
        self.device = "cpu"
        self.torch = None

    def emit(self, kind, **fields):
        with self.lock:
            self.output.write(json.dumps(dict(kind=kind, rank=self.rank, ns=time.monotonic_ns(),
                                              thread=threading.current_thread().name,
                                              call=getattr(self.local, "call", "none"), **fields)) + "\n")


def instrument(rec, backends, distribution, pipeline, torch):
    dist = torch.distributed
    for name in ("all_reduce", "all_gather", "broadcast"):
        original = getattr(dist, name)

        def wrap(original, name):
            @functools.wraps(original)
            def observed(*args, **kwargs):
                group = kwargs["group"] if "group" in kwargs else None
                members = dist.get_process_group_ranks(group or dist.group.WORLD)
                tensor = args[1] if name == "all_gather" else args[0]
                operation = str(kwargs["op"]) if "op" in kwargs else "default"
                fields = dict(group=",".join(map(str, members)), members=members,
                              signature=f"{name}:{operation}:{tuple(tensor.shape)}:{tensor.dtype}")
                rec.emit("collective_enter", **fields)
                try:
                    result = original(*args, **kwargs)
                except BaseException as error:
                    rec.emit("collective_error", message=str(error), **fields)
                    raise
                rec.emit("collective_exit", **fields)
                return result
            return observed
        setattr(dist, name, wrap(original, name))
    original_group = dist.new_group

    @functools.wraps(original_group)
    def new_group(*args, **kwargs):
        result = original_group(*args, **kwargs)
        members = args[0] if args else kwargs["ranks"]
        rec.emit("group_create", group=",".join(map(str, members)), members=members)
        return result
    dist.new_group = new_group

    for name in ("_PortableApply", "_DistributedApply"):
        if hasattr(backends, name):
            function = getattr(backends, name)
            original = function.forward

            def wrap_forward(original, name):
                @functools.wraps(original)
                def forward(*args, **kwargs):
                    rec.bounded += 1
                    rec.emit("bounded_function", function=name)
                    return original(*args, **kwargs)
                return forward
            function.forward = staticmethod(wrap_forward(original, name))
    reference = backends.ReferenceProjector
    original_rays, original_intervals = reference._rays, reference._intervals

    @functools.wraps(original_rays)
    def rays(self, views, pixels, *args, **kwargs):
        rec.local.rays = [views.start, views.stop, *[(p.start, p.stop) for p in pixels]]
        return original_rays(self, views, pixels, *args, **kwargs)

    @functools.wraps(original_intervals)
    def intervals(self, origins, directions, tile):
        fields = dict(phase=rec.phase, tile=[(p.start, p.stop) for p in tile],
                      rays=rec.local.rays, leading=getattr(rec.local, "leading", []))
        rec.emit("pair_begin", **fields)
        result = original_intervals(self, origins, directions, tile)
        rec.emit("pair_end", **fields)
        return result
    reference._rays, reference._intervals = rays, intervals

    original_geometry = reference._geometry_grad
    original_chunks = reference._work_chunks

    def chunks(self, *args, **kwargs):
        result = original_chunks(self, *args, **kwargs)
        rec.emit("chunk_plan", phase=rec.phase, plan=result,
                 batch_elements=args[0] if args else 1)
        return result

    def geometry(self, tensor, cotangent, is_project, *args, **kwargs):
        previous = rec.phase
        rec.phase = "geometry"
        rec.emit("geometry_begin", project=is_project, volume_shape=self.owner.volume_shape,
                 volume_slice=[(p.start, p.stop) for p in self.owner.volume_slice],
                 detector_shape=self.owner.detector_shape,
                 view_slice=(self.owner.view_slice.start, self.owner.view_slice.stop),
                 leading=list(tensor.shape[:-len(self.owner.volume_shape if is_project
                                                 else self.owner.projection_shape)]))
        try:
            return original_geometry(self, tensor, cotangent, is_project, *args, **kwargs)
        finally:
            rec.emit("geometry_end")
            rec.phase = previous
    reference._geometry_grad, reference._work_chunks = geometry, chunks

    original_apply = reference._tensor_apply

    @functools.wraps(original_apply)
    def apply(self, tensor, is_project, *args, **kwargs):
        rec.local.leading = list(tensor.shape[:-len(self.owner.volume_shape if is_project
                                                   else self.owner.projection_shape)])
        result = original_apply(self, tensor, is_project, *args, **kwargs)
        rec.emit("plan", stats=self.owner.last_execution_stats)
        return result
    reference._tensor_apply = apply

    original_slot = pipeline._DevicePipeline.slot

    def slot(self):
        result = original_slot(self)
        rec.emit("host_reuse", ready=result.event is None)
        return result
    pipeline._DevicePipeline.slot = slot
    original_finish = pipeline._HostSlot.finish

    def finish(self):
        drain, event = self.drain, self.event
        if drain is not None:
            def observed_drain():
                rec.emit("host_read", ready=event is None or event.query())
                return drain()
            self.drain = observed_drain
        return original_finish(self)
    pipeline._HostSlot.finish = finish
    original_close = pipeline._DevicePipeline.close

    def close(self):
        try:
            return original_close(self)
        finally:
            rec.emit("pool_close", remaining=len(self.host) + len(self.pending) + len(self.outputs)
                     + len(self.output_streams))
    pipeline._DevicePipeline.close = close
    original_event = pipeline._DevicePipeline.event

    def event(stream):
        result = original_event(stream)
        rec.emit("event_record", event=id(result), stream=int(stream.cuda_stream))
        return result
    pipeline._DevicePipeline.event = staticmethod(event)
    original_copy = pipeline._copy

    def copy(target, source, stats):
        rec.emit("copy_begin", source=str(source.device), target=str(target.device),
                 bytes=target.numel() * target.element_size())
        return original_copy(target, source, stats)
    pipeline._copy = copy


def projector(torch, diffct, device, distributed=False, partition="views", surface=False,
              trajectory_grad=False, explicit=True, native=False):
    frames = (torch.tensor([[-4., 0.]] * 3, dtype=torch.float64),
              torch.tensor([[4., 0.]] * 3, dtype=torch.float64),
              torch.tensor([[0., 1.]] * 3, dtype=torch.float64))
    if trajectory_grad:
        frames = tuple(value.requires_grad_() for value in frames)
    options = dict(beam="fan", volume_shape=(2, 3), detector_shape=(3,), trajectory=frames,
                   detector_spacing=1., voxel_spacing=1. if native else (1., 1.),
                   precision="fast" if native else "float64", backend="numba" if native else "torch",
                   distributed=distributed, partition=partition)
    if explicit:
        options.update(volume_chunk_shape=(1, 2), view_chunk_size=1, detector_chunk_shape=(2,))
    if surface:
        parameter = torch.tensor(.02, dtype=torch.float64, requires_grad=True)

        def sampler(u, v, ids, offset):
            return torch.stack((u + offset, torch.zeros_like(u), torch.zeros_like(u)), dim=-1)
        options["detector_surface"] = diffct.ParameterizedSurface(sampler, (parameter,))
    return diffct.Projector(**options)


def call(rec, name, function, *, required=False, allowed_error=False, require_rejection=False):
    rec.local.call = name
    rec.emit("call_begin", name=name)
    if rec.device.startswith("cuda"):
        rec.torch.cuda.synchronize()
        rec.torch.cuda.reset_peak_memory_stats()
        rec.emit("memory_begin", allocated=rec.torch.cuda.memory_allocated())
    start, before = time.monotonic(), rec.bounded
    try:
        result = function()
    except BaseException as error:
        message = str(error)
        kind = "runtime_timeout" if "timed out" in message.lower() or "timeout" in message.lower() else "call_error"
        expected = allowed_error and not name.startswith("thread-")
        rec.emit(kind, name=name, message=message, type=type(error).__name__,
                 allowed=expected, traceback=traceback.format_exc(), seconds=time.monotonic() - start)
        if require_rejection:
            if "forward-mode" not in message or "geometry" not in message:
                raise RuntimeError("geometry derivative raised an unrelated error") from error
            rec.emit("route", name=name, required=False, bounded=rec.bounded > before,
                     rejected=True, require_rejection=True)
        if not allowed_error:
            raise
        return None
    rec.emit("route", name=name, required=required, bounded=rec.bounded > before,
             pure=name in ("jvp-project", "jvp-backproject"),
             rejected=False, require_rejection=require_rejection)
    rec.emit("call_end", name=name, seconds=time.monotonic() - start)
    if rec.device.startswith("cuda"):
        rec.torch.cuda.synchronize()
        rec.emit("memory_end", allocated=rec.torch.cuda.memory_allocated(),
                 peak=rec.torch.cuda.max_memory_allocated())
    return result


def serial(rec, torch, diffct, device, native):
    p = projector(torch, diffct, device, native=native)
    dtype = torch.float32 if native else torch.float64
    data = torch.ones(p.volume_shape, device=device, dtype=dtype)
    for index in range(2):
        y = call(rec, f"forward-{index}", lambda: p.project(data))
        call(rec, f"adjoint-{index}", lambda: p.backproject(y))
    if device.startswith("cuda"):
        # Real stores exercise native host slots and CUDA events.
        host = torch.ones(p.volume_shape, dtype=dtype)
        for index in range(2):
            output = torch.empty(p.projection_shape, dtype=dtype)
            call(rec, f"host-store-{index}", lambda: p.project_into(diffct.TensorStore(host),
                                                                   diffct.TensorStore(output)))
    return p


def routing(rec, torch, diffct, device, backends):
    bounded_contract = hasattr(backends, "_PortableApply")
    for empty in (False, True):
        for geometry_kind in ("fixed", "trajectory", "surface"):
            p = projector(torch, diffct, device, trajectory_grad=geometry_kind == "trajectory",
                          surface=geometry_kind == "surface")
            for operation in ("project", "backproject"):
                shape = p.volume_shape if operation == "project" else p.projection_shape
                function = getattr(p, operation)
                for needs_grad in (False, True):
                    data = torch.ones((0, *shape) if empty else shape, device=device,
                                      dtype=torch.float64, requires_grad=needs_grad)
                    name = f"{geometry_kind}-{operation}-grad{int(needs_grad)}-empty{int(empty)}"
                    result = call(rec, name, lambda: function(data),
                                  required=bounded_contract and (needs_grad or geometry_kind != "fixed"))
                    if result.requires_grad:
                        rec.phase = "backward"
                        call(rec, name + "-backward", lambda: result.sum().backward())
                        rec.phase = "forward"
                if not empty and geometry_kind == "fixed":
                    data = torch.ones(shape, device=device, dtype=torch.float64)
                    with torch.no_grad():
                        call(rec, f"no-grad-{operation}", lambda: function(data.requires_grad_()), required=False)
                    call(rec, f"jvp-{operation}", lambda: torch.func.jvp(function, (data.detach(),),
                                                                         (torch.ones_like(data),)))
                    tangent = torch.ones_like(data, requires_grad=True)
                    call(rec, f"jvp-grad-tangent-{operation}", lambda: torch.func.jvp(
                        function, (data.detach(),), (tangent,)), required=bounded_contract)
                    batched = torch.ones((2, *shape), device=device, dtype=torch.float64, requires_grad=True)
                    call(rec, f"vmap-grad-{operation}", lambda: torch.vmap(function)(batched),
                         required=bounded_contract)
    # Geometry forward derivatives must raise on PR2.
    if bounded_contract:
        p = projector(torch, diffct, device)
        data = torch.ones(p.volume_shape, dtype=torch.float64, device=device)
        frames = p._trajectory
        call(rec, "geometry-jvp", lambda: torch.func.jvp(
            lambda *args: p.project(data, trajectory=args), frames,
            tuple(torch.ones_like(frame) for frame in frames)), allowed_error=True, require_rejection=True)


def memory_floor(rec, torch, diffct, device, backends):
    if not hasattr(backends.ReferenceProjector, "_work_budget"):
        return
    p = projector(torch, diffct, device, explicit=False)
    original = backends._memory_lower_bounds
    # Exercise the real scalar planner at a specified allocator state.
    # This is an admission counterexample, not a CUDA allocation measurement.
    usable = 1024 ** 2
    backends._memory_lower_bounds = lambda device: (usable * 3 // 4,) * 3
    try:
        rec.local.call = "planner-floor"
        budget = p._reference._work_budget((torch.device("cuda:0"),))
        value = budget[0] if isinstance(budget, tuple) else budget
        owner = copy(p)
        owner.global_volume_shape = owner.volume_shape = (128, 128)
        owner.volume_slice = (slice(0, 128), slice(0, 128))
        owner.projection_shape = (3, 512)
        owner.detector_shape = (512,)
        reference = copy(p._reference)
        reference.owner = owner
        plan = reference._work_chunks(1, budget)
        geometry = reference._snapshot_geometry()
        views, pixels = next(reference._batches(*plan[1:]))
        origins, directions, unused = reference._rays(views, pixels, torch.device("cpu"), geometry=geometry)
        unused, world = next(reference._tiles(plan[0]))
        indices, lengths = reference._intervals(origins, directions, world)
        payload = max(tensor.numel() * tensor.element_size() for tensor in (indices, lengths))
        rec.emit("budget_observation", available=usable, bound=usable // 4, admitted=value,
                 minimum_array_bytes=payload, plan=plan, simulated_allocator=True, probe_device="cpu")
        rec.local.call = "planner-graph"
        usable = 4 * 1024 ** 2
        graph_budget = reference._work_budget((torch.device("cuda:0"),), graph=(1, False))
        plan = reference._work_chunks(1, graph_budget)
        saved = reference._graph_bytes(plan[0], 1, False)
        data = torch.ones(owner.volume_shape, dtype=owner.dtype, requires_grad=True)
        kept = {}

        def pack(tensor):
            storage = tensor.untyped_storage()
            if tensor.numel() > data.numel() and storage._cdata != data.untyped_storage()._cdata:
                kept[storage._cdata] = storage.nbytes()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            result = reference._tensor_apply(data, True, geometry=geometry, _plan=(plan, graph_budget))
        rec.emit("budget_observation", available=usable, bound=usable // 4,
                 admitted=sum(kept.values()), estimated_graph=saved, work_budget=graph_budget[0],
                 saved_arrays=len(kept), simulated_allocator=True, probe_device="cpu")
        del result
    finally:
        backends._memory_lower_bounds = original


def thread_race(rec, torch, diffct, device, backends):
    p = projector(torch, diffct, device, explicit=False)
    reference = p._reference
    original_apply, original_preflight = reference._tensor_apply, p._partition.preflight
    original_budget = backends._REFERENCE_WORK_BYTES
    child_finished, other_reset, release_other = threading.Event(), threading.Event(), threading.Event()
    state = threading.local()
    failures = []
    backends._REFERENCE_WORK_BYTES = 256

    def apply(*args, **kwargs):
        depth = getattr(state, "depth", 0)
        state.depth = depth + 1
        try:
            result = original_apply(*args, **kwargs)
            if threading.current_thread().name == "A" and depth == 1 and not child_finished.is_set():
                rec.emit("stats_child_published")
                child_finished.set()
                if not other_reset.wait(5):
                    raise RuntimeError("thread schedule did not reach the reset")
            return result
        finally:
            state.depth = depth

    def preflight(*args, **kwargs):
        if threading.current_thread().name == "B":
            rec.emit("stats_other_reset", is_none=p.last_execution_stats is None)
            other_reset.set()
            if not release_other.wait(5):
                raise RuntimeError("thread schedule did not release B")
        return original_preflight(*args, **kwargs)

    def run_a():
        try:
            call(rec, "thread-A", lambda: p.project(torch.ones((8, *p.volume_shape), dtype=torch.float64)),
                 allowed_error=True)
        finally:
            release_other.set()

    def run_b():
        if not child_finished.wait(5):
            failures.append("A did not split")
            release_other.set()
            return
        call(rec, "thread-B", lambda: p.project(torch.ones(p.volume_shape, dtype=torch.float64)),
             allowed_error=True)
    reference._tensor_apply = apply
    p._partition.preflight = preflight
    try:
        threads = [threading.Thread(target=run_a, name="A"), threading.Thread(target=run_b, name="B")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(12)
        if failures or any(thread.is_alive() for thread in threads):
            raise RuntimeError(str(failures or "thread watchdog expired"))
        rec.output.flush()
        recorded = [json.loads(line) for line in Path(rec.output.name).read_text().splitlines()]
        errors = [(row["name"], row["type"], row["message"]) for row in recorded
                  if row["kind"] in ("call_error", "runtime_timeout") and row["name"].startswith("thread-")]
        if errors != [("thread-A", "TypeError", "'NoneType' object is not subscriptable")]:
            raise RuntimeError(f"statistics race witness changed: {errors}")
    finally:
        reference._tensor_apply, p._partition.preflight = original_apply, original_preflight
        backends._REFERENCE_WORK_BYTES = original_budget


def worker(args):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.source))
    import torch
    torch.set_num_threads(1)
    import diffct
    from diffct import backends, distribution, pipeline
    rank = args.rank
    rec = Recorder(args.out / f"rank-{rank}.ndjson", rank)
    rec.emit("source", path=str(Path(diffct.__file__).resolve()), torch=torch.__version__,
             has_budget=hasattr(backends.ReferenceProjector, "_work_budget"),
             has_bounded=hasattr(backends, "_PortableApply"),
             backends_sha256=hashlib.sha256(Path(backends.__file__).read_bytes()).hexdigest())
    instrument(rec, backends, distribution, pipeline, torch)
    device = f"cuda:{rank}" if args.device == "cuda" else "cpu"
    rec.device, rec.torch = device, torch
    if args.device == "cuda":
        torch.cuda.set_device(rank)
        torch.cuda.memory._record_memory_history(max_entries=100000, record_pinned_host_memory=True)
    if args.scenario in ("distributed", "planner"):
        torch.distributed.init_process_group(args.backend, init_method=args.init, rank=rank,
                                             world_size=args.world, timeout=timedelta(seconds=3))
        if args.scenario == "planner":
            p = projector(torch, diffct, device, distributed=True, explicit=False)
            original_memory, original_host = backends._memory_lower_bounds, backends._host_free_bytes
            backends._memory_lower_bounds = lambda unused: ((96 if rank == 0 else 192) * 1024 ** 2,) * 3
            backends._host_free_bytes = lambda: (256 if rank == 0 else 512) * 1024 ** 2
            try:
                for cpu_compute in (False, True):
                    compute = torch.device("cpu" if cpu_compute and rank == 0 else "cuda:0")
                    data = torch.device("cpu" if rank == 0 else "cuda:0")
                    budget = p._reference._work_budget((compute,), (data,), agree=True)
                    plan = p._reference._work_chunks(1, budget)
                    rec.emit("plan_agreement", signature=json.dumps([budget, plan]),
                             group=",".join(map(str, range(args.world))), members=list(range(args.world)),
                             simulated_allocator=True)
            finally:
                backends._memory_lower_bounds, backends._host_free_bytes = original_memory, original_host
            torch.distributed.destroy_process_group()
            rec.emit("process_end")
            rec.output.close()
            os._exit(0)
        p = projector(torch, diffct, device, distributed=True, partition=args.partition,
                      trajectory_grad=True)
        for index in range(2 if args.fault == "none" else 1):
            data = torch.ones(p.volume_shape, dtype=torch.float64, device=device,
                              requires_grad=rank == 0)
            if args.fault == "admission" and rank == 0:
                data = torch.ones((7,), dtype=torch.float64, device=device)
            if args.fault == "numerical" and rank == 0:
                def fail(*unused, **kw):
                    raise torch.OutOfMemoryError("formal injected OOM after admission")
                backends.ReferenceProjector._intervals = fail
            result = call(rec, f"distributed-{index}", lambda: p.project(data), allowed_error=args.fault != "none")
            if result is not None:
                rec.phase = "backward"
                call(rec, f"backward-{index}", lambda: result.sum().backward(),
                     allowed_error=args.fault != "none")
                rec.phase = "forward"
        if args.fault == "numerical" and rank == 0:
            # Keep the failed participant alive to separate peer timeouts from exit detection.
            threading.Event().wait(4)
        if args.fault == "none":
            # A second owner must reuse the process groups from the first owner.
            projector(torch, diffct, device, distributed=True, partition=args.partition)
        (args.out / f"flight-{rank}.pickle").write_bytes(torch._C._distributed_c10d._dump_fr_trace())
        torch.distributed.destroy_process_group()
    else:
        serial(rec, torch, diffct, device, args.native)
        if not args.native:
            routing(rec, torch, diffct, device, backends)
            memory_floor(rec, torch, diffct, device, backends)
            thread_race(rec, torch, diffct, device, backends)
    if args.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.memory._dump_snapshot(str(args.out / f"memory-{rank}.pickle"))
        rec.emit("cuda_memory", peak=torch.cuda.max_memory_allocated())
        torch.cuda.memory._record_memory_history(enabled=None)
    rec.emit("process_end")
    rec.output.close()
    # Gloo destructor failures must not hide successfully collected traces.
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--scenario", default="serial")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--init", default="")
    parser.add_argument("--partition", default="views")
    parser.add_argument("--fault", default="none")
    args = parser.parse_args()
    args.source, args.out = args.source.resolve(), args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    manifest = []
    cases = [("serial", 1, "none", "views"), ("views", 2, "none", "views"),
             ("space", 2, "none", "space"), ("admission", 2, "admission", "views"),
             ("numerical", 2, "numerical", "views")]
    if args.hybrid:
        cases.append(("hybrid", 4, "none", "hybrid"))
    if "def _work_budget(" in (args.source / "diffct" / "backends.py").read_text():
        cases.append(("planner", 2, "none", "views"))
    for name, world, fault, partition in cases:
        out = args.out / name
        out.mkdir(exist_ok=True)
        processes = []
        logs = []
        rendezvous = out / ("rendezvous-" + uuid.uuid4().hex)
        for rank in range(world):
            log = (out / f"rank-{rank}.log").open("w")
            logs.append(log)
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--source", str(args.source),
                   "--out", str(out), "--scenario", ("planner" if name == "planner" else
                                                     "serial" if world == 1 else "distributed"),
                   "--rank", str(rank), "--world", str(world), "--partition", partition,
                   "--fault", fault, "--init", f"file://{rendezvous}",
                   "--device", args.device, "--backend", args.backend]
            if args.native:
                cmd.append("--native")
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                       TORCH_NCCL_TRACE_BUFFER_SIZE="100000", TORCH_FR_BUFFER_SIZE="100000",
                       TORCH_NCCL_DUMP_ON_TIMEOUT="1",
                       TORCH_DISTRIBUTED_DEBUG="DETAIL")
            processes.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env))
        try:
            for process in processes:
                process.wait(timeout=45)
        finally:
            # These handles refer only to workers created by this invocation.
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            for log in logs:
                log.close()
        codes = [process.returncode for process in processes]
        if any(codes) and args.backend == "nccl" and fault == "numerical":
            # NCCL's watchdog can abort the process before Python catches an
            # exception. Record the observed exit, without inventing call exits.
            for rank, code in enumerate(codes):
                if code == 0:
                    continue
                message = (out / f"rank-{rank}.log").read_text()
                if "timeout" not in message.lower() and "timed out" not in message.lower():
                    raise RuntimeError(f"unexpected NCCL worker exit {code}: {out}")
                with (out / f"rank-{rank}.ndjson").open("a") as target:
                    target.write(json.dumps(dict(kind="runtime_timeout", rank=rank, ns=time.monotonic_ns(),
                                                 observer="parent", exitcode=code, message=message)) + "\n")
                    target.write(json.dumps(dict(kind="process_end", rank=rank, ns=time.monotonic_ns(),
                                                 observer="parent", exitcode=code)) + "\n")
            codes = [0] * world
        if any(codes):
            raise RuntimeError(f"collector {name} failed: {codes}; see {out}")
        source = json.loads((out / "rank-0.ndjson").read_text().splitlines()[0])
        expected = []
        if name == "serial" and not args.native:
            expected.append("NoUnexpectedCrash")
            if source["has_budget"]:
                expected.append("BudgetBound")
        if name == "numerical":
            expected.append("NoRuntimeTimeout")
        manifest.append(dict(name=name, files=[str((out / f"rank-{rank}.ndjson").relative_to(args.out))
                                              for rank in range(world)], world=world, fault=fault,
                             expected=expected))
        print(json.dumps(dict(case=name, workers=world, returncodes=codes)), flush=True)
    (args.out / "traces.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
