#!/usr/bin/env python3
"""Run pinned model checkers, mutation witnesses, and observed trace checks."""

import argparse
import functools
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request


HERE = Path(__file__).resolve().parent


def download(cache, name, lock):
    destination = cache / name
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".download")
        with urllib.request.urlopen(lock["url"], timeout=120) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        temporary.replace(destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if digest != lock["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch: {destination}: {digest}")
    return destination


def command(args, log, timeout):
    start = time.monotonic()
    with log.open("w") as output:
        result = subprocess.run(args, stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
    return result.returncode, log.read_text(), round(time.monotonic() - start, 3)


def tlc(java, jar, module, config, out, name, expected=None, timeout=120):
    # Stable TLC resolves external configs relative to the root module.
    if module.parent != config.parent:
        staged = out / (name + ".sources")
        staged.mkdir(exist_ok=True)
        shutil.copyfile(module, staged / module.name)
        shutil.copyfile(config, staged / config.name)
        module, config = staged / module.name, staged / config.name
    log = out / f"{name}.log"
    cmd = [java, "-Xmx768m", "-XX:+UseParallelGC", "-cp", str(jar), "tlc2.TLC",
           "-workers", "1", "-seed", "1", "-metadir", str(out / f"{name}.states"),
           "-config", str(config), str(module)]
    code, text, elapsed = command(cmd, log, timeout)
    if expected is None:
        passed = code == 0 and "Model checking completed. No error has been found." in text
    else:
        passed = code != 0 and expected in text and "Error: Parsing" not in text
    matches = re.findall(r"([\d,]+) states generated, ([\d,]+) distinct states found", text)
    counts = matches[-1] if matches else ("0", "0")
    result = dict(name=name, passed=passed, expected=expected or "success", returncode=code,
                  seconds=elapsed, generated=int(counts[0].replace(",", "")),
                  distinct=int(counts[1].replace(",", "")), log=str(log))
    print(json.dumps(result), flush=True)
    return result


def models(java, jar, out, large):
    results = []
    modules = ["BudgetAgreement", "NativeAdmission", "Distributed", "MemoryEnvelope", "PipelineLifecycle", "AutogradRouting",
               "RecomputeOrder", "ThreadSafety"]
    for module in modules:
        results.append(tlc(java, jar, HERE / f"{module}.tla", HERE / f"{module}.small.cfg",
                           out, f"{module}.small"))
    if large:
        results.append(tlc(java, jar, HERE / "Distributed.tla", HERE / "Distributed.large.cfg",
                           out, "Distributed.large"))
    mutations = {
        "BudgetAgreement": [("device", "Invariant DeviceAgreement is violated"),
                            ("host", "Invariant HostAgreement is violated"),
                            ("deviceMax", "Invariant DeviceMinimum is violated"),
                            ("hostMax", "Invariant HostMinimum is violated"),
                            ("localLeading", "Invariant LeadingAgreement is violated")],
        "NativeAdmission": [("range", "Invariant NativeRange is violated"),
                            ("selection", "Invariant SelectionAgreement is violated"),
                            ("spacing", "Invariant SpacingAgreement is violated"),
                            ("dtype", "Invariant TrajectoryAgreement is violated"),
                            ("warning", "Invariant FallbackWarning is violated")],
        "Distributed": [("localBudget", "Invariant BudgetAgreement is violated"),
                        ("skipEmpty", "Deadlock reached"), ("skipUntrained", "Deadlock reached"),
                        ("localError", "Deadlock reached")],
        "MemoryEnvelope": [(field, "Invariant DeviceBound is violated")
                           for field in ("work", "copy", "graph", "reduction")]
                          + [("host", "Invariant HostBound is violated"),
                             ("twoTiles", "Invariant OneReducedTile is violated"),
                             ("cotangent", "Invariant CotangentException is violated"),
                             ("leak", "Invariant NoLeaks is violated")],
        "PipelineLifecycle": [("reuseHost", "Invariant NoEarlyReuse is violated"),
                              ("unfinishedUpload", "Invariant NoUnfinishedRead is violated"),
                              ("unfinishedDownload", "Invariant NoUnfinishedRead is violated"),
                              ("releaseEarly", "Invariant NoEarlyReuse is violated"),
                              ("leak", "Invariant NoLeaks is violated"),
                              ("growPool", "Invariant PoolBound is violated")],
        "AutogradRouting": [("shallow", "Invariant BoundedRecording is violated"),
                            ("empty", "Invariant BoundedRecording is violated"),
                            ("forceFunction", "Invariant PureForwardJvp is violated"),
                            ("geometryJvp", "Invariant GeometryForwardRejected is violated")],
        "RecomputeOrder": [("rankOrder", "Invariant ForwardOrder is violated")],
    }
    for module, cases in mutations.items():
        base = (HERE / f"{module}.small.cfg").read_text()
        # Safety witnesses do not need temporal exploration.
        base = re.sub(r"^PROPERT(?:Y|IES) .*\n?", "", base, flags=re.M)
        for mutation, expected in cases:
            config = out / f"{module}.{mutation}.cfg"
            config.write_text(base.replace('Mutation = "none"', f'Mutation = "{mutation}"'))
            results.append(tlc(java, jar, HERE / f"{module}.tla", config, out,
                               f"{module}.{mutation}", expected))
    for check, expected in (("stats", "Invariant StatsIsolation is violated"),
                             ("groups", "Invariant GroupReuse is violated")):
        config = out / f"ThreadSafety.{check}.cfg"
        config.write_text((HERE / "ThreadSafety.small.cfg").read_text()
                          .replace("Serialized = TRUE", "Serialized = FALSE")
                          .replace("INVARIANTS StatsIsolation GroupReuse",
                                   "INVARIANT " + ("StatsIsolation" if check == "stats" else "GroupReuse")))
        results.append(tlc(java, jar, HERE / "ThreadSafety.tla", config, out,
                           f"ThreadSafety.{check}", expected))
    config = out / "Distributed.order.cfg"
    config.write_text((out / "Distributed.localBudget.cfg").read_text()
                      .replace("INVARIANTS BudgetAgreement CollectiveOrder", "INVARIANT CollectiveOrder"))
    results.append(tlc(java, jar, HERE / "Distributed.tla", config, out, "Distributed.order",
                       "Invariant CollectiveOrder is violated"))
    # Symmetry is used only for safety. Liveness runs above use no symmetry.
    config = out / "Distributed.symmetry.cfg"
    config.write_text(re.sub(r"^PROPERTIES .*\n?", "", (HERE / "Distributed.small.cfg").read_text(),
                             flags=re.M) + "\nSYMMETRY RankSymmetry\n")
    results.append(tlc(java, jar, HERE / "Distributed.tla", config, out, "Distributed.symmetry"))
    return results


def symbolic(java, archive, cache, out):
    jar = cache / "apalache-0.62.4" / "lib" / "apalache.jar"
    if not jar.exists():
        # Releases contain only regular files/directories under this directory.
        with tarfile.open(archive) as source:
            source.extractall(cache, filter="data")
    results = []
    for mutation, expected in (("none", None), ("graph", "Invariant")):
        config = out / f"MemoryEnvelope.symbolic.{mutation}.cfg"
        config.write_text((HERE / "MemoryEnvelope.symbolic.cfg").read_text()
                          .replace('Mutation = "none"', f'Mutation = "{mutation}"'))
        log = out / f"apalache.{mutation}.log"
        cmd = [java, "-Xmx1024m", "-XX:+UseParallelGC", "-jar", str(jar),
               f"--out-dir={out / ('apalache-' + mutation)}", "check",
               f"--config={config}", "--length=8",
               "--inv=DeviceBound,HostBound,OneReducedTile,CotangentException,NoLeaks", str(HERE / "MemoryEnvelope.tla")]
        code, text, seconds = command(cmd, log, 120)
        passed = ((code == 0 and "The outcome is: NoError" in text) if expected is None
                  else code == 12 and "state invariant 0 violated" in text)
        row = dict(name=f"apalache.{mutation}", passed=passed, returncode=code,
                   seconds=seconds, expected=expected or "success", bound=8, log=str(log))
        results.append(row)
        print(json.dumps(row), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=HERE / ".cache")
    parser.add_argument("--out", type=Path, default=HERE / ".results")
    parser.add_argument("--java", default=("/opt/homebrew/opt/openjdk/bin/java"
                        if Path("/opt/homebrew/opt/openjdk/bin/java").exists() else shutil.which("java")))
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--skip-apalache", action="store_true")
    parser.add_argument("--only", choices=("models", "traces", "all"), default="all")
    parser.add_argument("--trace-dir", type=Path)
    args = parser.parse_args()
    args.cache = args.cache.resolve()
    args.out = args.out.resolve()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    lock = json.loads((HERE / "tools.json").read_text())
    jar = download(args.cache, "tla2tools-" + lock["tlc"]["version"] + ".jar", lock["tlc"])
    results = []
    if args.only in ("models", "all"):
        results.extend(models(args.java, jar, args.out, args.large))
        if not args.skip_apalache:
            archive = download(args.cache, "apalache-0.62.4.tgz", lock["apalache"])
            results.extend(symbolic(args.java, archive, args.cache, args.out))
    if args.only in ("traces", "all"):
        from validate_trace import validate_directory
        if args.trace_dir is None:
            parser.error("--trace-dir is required for trace validation")
        # Replay TLC runs over real distributed traces vary in state count with
        # the observed schedules; give them a larger budget than the models.
        tlc_traces = functools.partial(tlc, timeout=300)
        results.extend(validate_directory(args.trace_dir.resolve(), args.out, args.java, jar, tlc_traces))
    (args.out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0 if results and all(row["passed"] for row in results) else 1


if __name__ == "__main__":
    sys.exit(main())
