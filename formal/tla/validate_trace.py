"""Constrain the observed protocol to each complete NDJSON execution."""

import copy
import hashlib
import json
import itertools
import math
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent
INVARIANTS = ["NoMixedOutcomes", "PlanAgreement", "CollectiveAgreement", "CompletionCounts", "CollectivesFinished", "NoRuntimeTimeout",
              "NoUnexpectedCrash", "BoundedRoute", "PureForwardRoute", "GeometryRejected", "NoUnfinishedRead", "NoLeaks",
              "BudgetBound", "BudgetMinimum", "NativeSelection", "FallbackWarning", "AdmissionRejected", "CallsFinished", "GroupReuse"]
KINDS = {"call_begin", "call_end", "plan_agreement", "collective_enter", "collective_exit", "group_create", "runtime_timeout", "call_error",
         "route", "host_read", "host_reuse", "pool_close", "budget_observation", "process_end", "budget_result",
         "selection_observation", "fallback_warning", "request_rejection"}


def literal(value):
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "<<" + ",".join(literal(item) for item in value) + ">>"
    if isinstance(value, dict):
        return "[" + ",".join(f"{key} |-> {literal(item)}" for key, item in value.items()) + "]"
    raise TypeError(type(value))


def event_record(row):
    # Fixed record shape makes missing observations visible to SANY/TLC.
    fields = dict(kind=row["kind"], rank=row["rank"], name="none", group="none", signature="none", allowed=False,
                  required=False, bounded=False, pure=False, require_rejection=False, rejected=False,
                  ready=True, remaining=0, admitted=0, bound=0, correct=True, required_warning=False, warned=False)
    for key in fields:
        if key in row:
            fields[key] = row[key]
    return fields


def replay(rows, world, out, name, java, jar, tlc, invariants, expected=None, incomplete=False):
    directory = out / name
    directory.mkdir(exist_ok=True)
    for module in ("ObservedProtocol", "TraceValidation"):
        shutil.copyfile(HERE / f"{module}.tla", directory / f"{module}.tla")
    groups = {row["group"]: row["members"] for row in rows if "group" in row}
    groups["none"] = []
    records = [event_record(row) for row in rows if row["kind"] in KINDS]
    module = directory / "RunTrace.tla"
    module.write_text("---- MODULE RunTrace ----\nEXTENDS TraceValidation\n"
                      + f"TraceRanks == 0..{world - 1}\n"
                      + "TraceGroups == {" + ",".join(literal(g) for g in groups) + "}\n"
                      + "TraceMembers == [g \\in TraceGroups |-> CASE "
                      + " [] ".join(f"g = {literal(g)} -> {{{','.join(map(str, members))}}}"
                                    for g, members in groups.items()) + "]\n"
                      + f"TraceEvents == {literal(records)}\n====\n")
    config = directory / "RunTrace.cfg"
    config.write_text("SPECIFICATION Spec\nCONSTANTS Ranks <- TraceRanks Groups <- TraceGroups\n"
                      + "Members <- TraceMembers Events <- TraceEvents\n"
                      + f"AllowIncomplete = {'TRUE' if incomplete else 'FALSE'}\n"
                      + "INVARIANTS " + " ".join(invariants) + "\nPROPERTY TraceConsumed\n")
    return tlc(java, jar, module, config, out, name,
               f"Invariant {expected} is violated" if expected else None)


def validate_directory(trace_dir, out, java, jar, tlc):
    manifest = json.loads((trace_dir / "traces.json").read_text())
    if not manifest:
        raise RuntimeError("empty trace manifest")
    root = HERE.parents[1] / "diffct"
    source_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(root.rglob("*.py"))}
    results = []
    for case in manifest:
        rows = []
        if len(case["files"]) != case["world"]:
            raise RuntimeError("trace manifest has the wrong participant count")
        for rank, name in enumerate(case["files"]):
            rank_rows = [json.loads(line) for line in (trace_dir / name).read_text().splitlines()]
            if not rank_rows or rank_rows[-1]["kind"] != "process_end":
                raise RuntimeError(f"incomplete trace: {name}")
            source = rank_rows[0]
            if source["kind"] != "source" or source["schema"] != 2 or source["library_sha256"] != source_hashes:
                raise RuntimeError(f"trace source differs from this library: {name}")
            if any(item["rank"] != rank for item in rank_rows):
                raise RuntimeError(f"trace participant differs from its manifest: {name}")
            observed = {item["kind"] for item in rank_rows}
            required = {"call_begin", "call_end"} if case["name"] != "planner" else {"budget_result", "plan_agreement"}
            if case["name"] in ("views", "space", "hybrid"):
                required |= {"geometry_begin", "leading_plan", "geometry_reduce", "pair_begin", "bounded_function"}
                if not any(item["kind"] == "leading_plan" and item["leading"] == [3, 2]
                           and item["limit"] is not None and item["limit"] < 6 for item in rank_rows):
                    raise RuntimeError(f"trace never splits a leading batch: {name}")
            if case["name"] == "serial" and case["native"]:
                required |= {"host_read", "host_reuse", "pool_close", "event_record", "copy_begin"}
            if case["fault"] == "admission":
                required = {"call_begin", "call_error", "request_rejection"}
            elif case["fault"] == "numerical":
                required = {"call_begin"}
            if not required <= observed:
                raise RuntimeError(f"missing observations in {name}: {sorted(required - observed)}")
            rows.extend(rank_rows)
        rows.sort(key=lambda row: row["ns"])
        expected = list(case["expected"])
        if case["fault"] == "numerical":
            expected.append("NoMixedOutcomes")
        invariants = [name for name in INVARIANTS if name not in expected]
        prefix = trace_dir.name + "." + case["name"]
        results.append(replay(rows, case["world"], out, prefix, java, jar, tlc, invariants,
                              incomplete=case["fault"] == "numerical"))
        for property_name in expected:
            results.append(replay(rows, case["world"], out, prefix + "." + property_name,
                                  java, jar, tlc, [property_name], expected=property_name,
                                  incomplete=case["fault"] == "numerical"))
        if case["name"] == "views":
            mutated = copy.deepcopy(rows)
            event = next(row for row in mutated if row["kind"] == "collective_enter")
            event["signature"] += ":removed-agreement-guard"
            results.append(replay(mutated, case["world"], out, prefix + ".mutation", java, jar, tlc,
                                  ["CollectiveAgreement"], expected="CollectiveAgreement"))
        if rows[0]["has_bounded"]:
            results.extend(recompute(rows, out, prefix, java, jar, tlc))
    results.extend(guard_mutations(out, trace_dir.name, java, jar, tlc))
    return results


def guard_mutations(out, prefix, java, jar, tlc):
    """Reject a minimal malformed execution for each observation invariant."""
    group = dict(group="0,1", members=[0, 1], signature="A", name="call")
    cases = {
        "NoMixedOutcomes": [dict(kind="call_end", rank=0, name="call"),
                            dict(kind="call_error", rank=1, name="call", allowed=True)],
        "PlanAgreement": [dict(kind="plan_agreement", rank=0, **group),
                          dict(kind="plan_agreement", rank=1, **{**group, "signature": "B"})],
        "CollectiveAgreement": [dict(kind="collective_enter", rank=0, **group),
                                dict(kind="collective_enter", rank=1, **{**group, "signature": "B"})],
        "CompletionCounts": [dict(kind="collective_exit", rank=0, **group)],
        "CollectivesFinished": [dict(kind="collective_enter", rank=0, **group)],
        "NoRuntimeTimeout": [dict(kind="runtime_timeout", rank=0)],
        "NoUnexpectedCrash": [dict(kind="call_error", rank=0, allowed=False)],
        "BoundedRoute": [dict(kind="route", rank=0, required=True, bounded=False)],
        "PureForwardRoute": [dict(kind="route", rank=0, pure=True, bounded=True)],
        "GeometryRejected": [dict(kind="route", rank=0, require_rejection=True, rejected=False)],
        "NoUnfinishedRead": [dict(kind="host_read", rank=0, ready=False)],
        "NoLeaks": [dict(kind="pool_close", rank=0, remaining=1)],
        "BudgetBound": [dict(kind="budget_observation", rank=0, admitted=2, bound=1)],
        "BudgetMinimum": [dict(kind="budget_result", rank=0, correct=False)],
        "NativeSelection": [dict(kind="selection_observation", rank=0, correct=False)],
        "FallbackWarning": [dict(kind="fallback_warning", rank=0, required_warning=True, warned=False)],
        "AdmissionRejected": [dict(kind="request_rejection", rank=0, rejected=False)],
        "CallsFinished": [dict(kind="call_begin", rank=0, name="unfinished")],
        "GroupReuse": [dict(kind="group_create", rank=0, **group)] * 2,
    }
    return [replay(rows, 2, out, prefix + ".guard-" + name, java, jar, tlc, [name], expected=name)
            for name, rows in cases.items()]


def leading_pieces(shape, limit):
    """Enumerate the contract's row-major pieces independently of observed IDs."""
    pending, result = [tuple((0, size) for size in shape)], []
    while pending:
        piece = pending.pop(0)
        sizes = [end - start for start, end in piece]
        count = math.prod(sizes)
        if limit is None or count <= limit:
            result.append(piece)
        else:
            axis = max(range(len(sizes)), key=sizes.__getitem__)
            step = max(1, limit // (count // sizes[axis]))
            children = [(*piece[:axis], (first, min(first + step, piece[axis][1])), *piece[axis + 1:])
                        for first in range(piece[axis][0], piece[axis][1], step)]
            pending[:0] = children
    return result


def recompute(rows, out, prefix, java, jar, tlc):
    """Bind output SUMs, leading pieces, input SUMs, pairs and parameter SUMs."""
    results, seen = [], set()
    for start, row in enumerate(rows):
        if row["kind"] != "geometry_begin":
            continue
        identity = (row["rank"], row["project"], row["reduce_input"], row["replicated"],
                    tuple(row["leading"]), row["parameter_blocks"])
        if identity in seen:
            continue
        seen.add(identity)
        end = next(i for i in range(start + 1, len(rows))
                   if rows[i]["kind"] == "geometry_end" and rows[i]["rank"] == row["rank"])
        segment = [item for item in rows[start + 1:end] if item["rank"] == row["rank"]]
        plans = [item["plan"] for item in segment if item["kind"] == "chunk_plan"
                 and item["phase"] == "geometry"]
        if not plans:
            raise RuntimeError("geometry replay has no observed plan")
        chunks, view_step, detector_step = plans[0]
        tiles = list(itertools.product(*(list((first, min(first + step, size))
                                              for first in range(0, size, step))
                                         for size, step in zip(row["volume_shape"], chunks))))
        tiles = [tuple((a + owned[0], b + owned[0]) for (a, b), owned in zip(tile, row["volume_slice"]))
                 for tile in tiles]
        pixels = list(itertools.product(*(list((first, min(first + step, size))
                                               for first in range(0, size, step))
                                          for size, step in zip(row["detector_shape"], detector_step))))
        batches = [(first, min(first + view_step, row["view_slice"][1]), *pixel)
                   for first in range(row["view_slice"][0], row["view_slice"][1], view_step) for pixel in pixels]
        leading_plan, = [item for item in segment if item["kind"] == "leading_plan"]
        pieces = leading_pieces(row["leading"], leading_plan["limit"])
        actual = []
        parameter = 0
        for item in segment:
            if item["kind"] not in ("pair_begin", "geometry_reduce"):
                continue
            stage = "pair" if item["kind"] == "pair_begin" else item["stage"]
            if stage == "parameter":
                parameter += 1
                actual.append(["parameter", parameter, 0, 0])
                continue
            piece = 0 if stage == "output" else pieces.index(tuple(map(tuple, item["piece"]))) + 1
            tile = (tiles.index(tuple(map(tuple, item["tile"]))) + 1
                    if stage == "pair" or (stage == "input" and row["project"])
                    or (stage == "output" and not row["project"]) else 0)
            rays = (batches.index(tuple(tuple(part) if isinstance(part, list) else part for part in item["rays"])) + 1
                    if stage == "pair" or (stage == "input" and not row["project"])
                    or (stage == "output" and row["project"]) else 0)
            actual.append([stage, piece, tile, rays])
        name = prefix + f".recompute-{len(results)}"
        directory = out / name
        directory.mkdir(exist_ok=True)
        for module in ("RecomputeOrder", "RecomputeTrace"):
            shutil.copyfile(HERE / f"{module}.tla", directory / f"{module}.tla")
        module = directory / "RunRecompute.tla"
        module.write_text("---- MODULE RunRecompute ----\nEXTENDS RecomputeTrace\n"
                          + f"Observed == {literal(actual)}\n====\n")
        config = directory / "RunRecompute.cfg"
        config.write_text("SPECIFICATION TraceSpec\nCONSTANTS Ranks = {r1} Mutation = \"none\"\n"
                          + f"Tiles = {len(tiles)} Rays = {len(batches)} "
                          + f"Pieces = {len(pieces)} Parameters = {row['parameter_blocks']} "
                          + f"Replicated = {literal(row['replicated'])} ReduceInput = {literal(row['reduce_input'])} "
                          + f"Project = {'TRUE' if row['project'] else 'FALSE'} Actual <- Observed\n"
                          + "INVARIANTS ObservedLength ForwardOrder\nPROPERTY Termination\n")
        results.append(tlc(java, jar, module, config, out, name))
        if len(actual) > 1 and not any(item["name"].endswith("mutation") for item in results):
            actual[0], actual[1] = actual[1], actual[0]
            mutant = directory / "MutatedRecompute.tla"
            mutant.write_text("---- MODULE MutatedRecompute ----\nEXTENDS RecomputeTrace\n"
                               + f"Observed == {literal(actual)}\n====\n")
            results.append(tlc(java, jar, mutant, config, out, name + ".mutation", "Deadlock reached"))
            truncated = directory / "TruncatedRecompute.tla"
            truncated.write_text("---- MODULE TruncatedRecompute ----\nEXTENDS RecomputeTrace\n"
                                  + f"Observed == {literal(actual[:-1])}\n====\n")
            results.append(tlc(java, jar, truncated, config, out, name + ".truncated",
                               "Invariant ObservedLength is violated"))
    return results
