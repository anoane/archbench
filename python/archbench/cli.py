# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from .catalog import generate, select, PROFILES, EXTENSIONS
from .config import load_cases
from .model import Case, canonical
from .runner import run_suite, atomic_json, read_jsonl
from .report import render_report, compare, summary
from .build import build, doctor, ROOT
from .pairing import operation_id, with_software_counterparts


def selection_options(parser):
    parser.add_argument("--profile", choices=PROFILES, default="standard")
    parser.add_argument("--matrix", type=Path, help="custom JSON matrix or an exact JSONL plan")
    parser.add_argument("--families", default="", help="comma-separated family filter")
    parser.add_argument("--dtypes", default="", help="comma-separated dtype/quant format filter")
    parser.add_argument("--ops", default="", help="comma-separated operation filter")
    parser.add_argument("--match", default="", help="substring in the full case label")
    parser.add_argument("--shard", help="zero-based INDEX/COUNT; keeps software pairs together by operation ID (case ID for exact replay)")
    parser.add_argument("--limit", type=int, default=0, help="maximum selected cases; 0 means all")
    parser.add_argument("--software-pairs", action=argparse.BooleanOptionalAction, default=None,
                        help="include software tensor/GEMM/BLAS/FFT/CSR/quantized GEMM counterparts (default on; exact JSONL plans replay unchanged unless explicitly enabled)")


def software_pairs_enabled(args):
    explicit = getattr(args, "software_pairs", None)
    if explicit is not None: return explicit
    return not (args.matrix and args.matrix.suffix == ".jsonl")


def selected(args):
    source = load_cases(args.matrix) if args.matrix else generate(args.profile)
    shard = tuple(map(int, args.shard.split("/"))) if args.shard else None
    if shard is not None and len(shard) != 2: raise ValueError("shard is INDEX/COUNT")
    if args.limit < 0: raise ValueError("limit must be nonnegative")
    csv = lambda value: tuple(x for x in value.split(",") if x)
    filtered = select(source, families=csv(args.families), dtypes=csv(args.dtypes), ops=csv(args.ops), match=args.match)
    paired = software_pairs_enabled(args)
    expanded = with_software_counterparts(filtered) if paired else filtered
    def final_selection():
        emitted = 0
        for item in expanded:
            if shard is not None:
                index, count = shard
                if count < 1 or index < 0 or index >= count: raise ValueError("shard must be INDEX/COUNT, zero based")
                key = operation_id(item) if paired else item.id
                if int(key.split(".")[-1], 16) % count != index: continue
            yield item
            emitted += 1
            if args.limit and emitted >= args.limit: return
    return final_selection()


MEASUREMENT_PRESETS = {
    "quick": {"samples": 3, "max_samples": 3, "warmup": 1, "warmup_ms": 0,
              "target_ms": 5, "target_cv": 0},
    "standard": {"samples": 7, "max_samples": 21, "warmup": 3, "warmup_ms": 0,
                 "target_ms": 20, "target_cv": .05},
    "stable": {"samples": 11, "max_samples": 31, "warmup": 10, "warmup_ms": 200,
               "target_ms": 50, "target_cv": .03},
}


def resolve_measurement_policy(args):
    preset = MEASUREMENT_PRESETS[args.quality]
    for name, value in preset.items():
        if getattr(args, name) is None:
            setattr(args, name, max(value, args.samples) if name == "max_samples" else value)
    return args


def chosen_case(args):
    if args.case_json:
        value = json.loads(Path(args.case_json).read_text())
        return Case.from_dict(value)
    if not args.from_run or not args.case_id: raise ValueError("provide --case-json or both --from-run and --case-id")
    for value in read_jsonl(Path(args.from_run) / "plan.jsonl"):
        if value["id"] == args.case_id: return Case.from_dict(value)
    raise ValueError("case ID not in run plan")


def make_parser():
    parser = argparse.ArgumentParser(prog="bench", description="ArchBench: explicit architecture and numeric-path GPU benchmarks")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor_parser = sub.add_parser("doctor", help="toolkit, hardware, topology and architecture-target inventory")
    doctor_parser.add_argument("--output", type=Path)
    builder = sub.add_parser("build", help="build separate architecture binaries with CUDA 13.5")
    builder.add_argument("--arch", default="80,89,120a")
    builder.add_argument("--build-dir", default=str(ROOT / "build"))
    builder.add_argument("--nvcc")
    builder.add_argument("--jobs", type=int, default=4)
    builder.add_argument("--cpu-only", action="store_true")
    builder.add_argument("--allow-older-cuda", action="store_true")
    listing = sub.add_parser("list", help="inspect or count the finite case matrix")
    selection_options(listing)
    listing.add_argument("--count", action="store_true")
    listing.add_argument("--json", action="store_true")
    plan = sub.add_parser("plan", help="write exact case identities without GPU execution")
    selection_options(plan)
    plan.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run", help="execute isolated, validated CUDA workers")
    selection_options(run)
    run.add_argument("--binary", required=True)
    run.add_argument("--output", required=True)
    gpu = run.add_mutually_exclusive_group()
    gpu.add_argument("--device", type=int, default=0, help="CUDA ordinal among visible devices")
    gpu.add_argument("--gpu-uuid", help="full GPU UUID; resolves the matching visible CUDA ordinal without hiding peers")
    run.add_argument("--quality", choices=MEASUREMENT_PRESETS, default="stable",
                     help="measurement preset; stable targets low variation but never hides unstable samples")
    run.add_argument("--samples", type=int, help="minimum timed sample count; each sample is an average over repeated calls")
    run.add_argument("--max-samples", type=int, help="bounded adaptive sample count")
    run.add_argument("--target-cv", type=float, help="coefficient-of-variation target; 0 disables adaptive extension")
    run.add_argument("--warmup", type=int, help="minimum untimed warmup calls")
    run.add_argument("--warmup-ms", type=float, help="minimum untimed warmup wall duration, additional to call count")
    run.add_argument("--target-ms", type=float, help="target device duration per sample batch")
    run.add_argument("--repeats", type=int, default=0, help="fixed calls per sample; 0 calibrates automatically")
    run.add_argument("--max-repeats", type=int, default=10000)
    run.add_argument("--memory-fraction", type=float, default=.70)
    run.add_argument("--memory-limit-mib", type=int, default=0)
    run.add_argument("--timeout", type=float, default=120)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument("--fail-fast", action="store_true")
    run.add_argument("--shuffle-seed", type=int, default=20261125)
    run.add_argument("--progress-every", type=int, default=20)
    run.add_argument("--telemetry", action="store_true")
    run.add_argument("--telemetry-interval", type=float, default=.2)
    run.add_argument("--report", action=argparse.BooleanOptionalAction, default=True,
                     help="automatically generate HTML, CSV and coverage JSON on completion/interruption")
    report = sub.add_parser("report", help="generate self-contained HTML, CSV and coverage JSON")
    report.add_argument("run", type=Path)
    report.add_argument("--output", type=Path)
    comparison = sub.add_parser("compare", help="compare matched verified case IDs across runs")
    comparison.add_argument("runs", nargs="+", type=Path)
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument("--allow-software-mismatch", action="store_true")
    coverage = sub.add_parser("coverage", help="honest per-family validation and missing-case counts")
    coverage.add_argument("run", type=Path)
    sub.add_parser("extensions", help="list adapters and explicit scope gaps")
    probes = sub.add_parser("probe", help="compile-only PTX capability probes, never benchmark results")
    probes.add_argument("--arch", default="80,89,120a")
    probes.add_argument("--output", type=Path, required=True)
    probes.add_argument("--ptxas")
    for name in ("profile", "sanitize"):
        item = sub.add_parser(name, help="collect profiler evidence" if name == "profile" else "run Compute Sanitizer")
        item.add_argument("--binary", required=True)
        item.add_argument("--case-json", type=Path)
        item.add_argument("--from-run", type=Path)
        item.add_argument("--case-id")
        item.add_argument("--output", type=Path, required=True)
        item.add_argument("--device", type=int, default=0)
        item.add_argument("--timeout", type=float, default=300)
        item.add_argument("--tool", choices=("memcheck", "racecheck", "initcheck", "synccheck"), default="memcheck")
        item.add_argument("--kernel-regex")
    assembly = sub.add_parser("disassemble", help="save SASS and resource usage with binary SHA256")
    assembly.add_argument("--binary", required=True)
    assembly.add_argument("--output", type=Path, required=True)
    cut_build = sub.add_parser("cutlass-build", help="build a supplied CUTLASS checkout and record exact revision")
    cut_build.add_argument("--source", required=True)
    cut_build.add_argument("--build-dir", required=True)
    cut_build.add_argument("--arch", choices=("80", "89", "120a"), required=True)
    cut_build.add_argument("--kernels", default="all")
    cut_build.add_argument("--jobs", type=int, default=4)
    cut_build.add_argument("--nvcc")
    cut = sub.add_parser("cutlass", help="verified profiler adapter; extra arguments follow --")
    cut.add_argument("--profiler", required=True)
    cut.add_argument("--output", type=Path, required=True)
    cut.add_argument("--timeout", type=float, default=3600)
    cut.add_argument("--enumerate", action="store_true")
    cut.add_argument("args", nargs=argparse.REMAINDER)
    external = sub.add_parser("external", help="run a provider argv JSON and preserve raw results without inferring validation")
    external.add_argument("spec", type=Path)
    external.add_argument("--output", type=Path, required=True)
    external.add_argument("--timeout", type=float, default=3600)
    test = sub.add_parser("selftest", help="host references and harness tests; requires no GPU")
    test.add_argument("--verbose", action="store_true")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor()
            if args.output: atomic_json(args.output, result)
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "build":
            if args.jobs < 1: raise ValueError("jobs must be positive")
            print(json.dumps(build(args), indent=2)); return 0
        if args.command == "list":
            counts = Counter()
            for item in selected(args):
                counts[item.family] += 1
                if not args.count: print(canonical(item.as_dict()) if args.json else f"{item.id} {item.label}")
            if args.count: print(json.dumps({"total": sum(counts.values()), "families": counts}, indent=2))
            return 0
        if args.command == "plan":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            count = 0
            with args.output.open("w", encoding="utf-8") as stream:
                for item in selected(args): stream.write(canonical(item.as_dict()) + "\n"); count += 1
            print(f"Wrote {count} exact cases to {args.output}; none have been executed."); return 0
        if args.command == "run":
            resolve_measurement_policy(args)
            if not (1 <= args.samples <= 10000 and 0 <= args.warmup <= 10000 and args.timeout > 0
                    and args.target_ms > 0 and args.max_repeats > 0 and args.progress_every > 0
                    and args.samples <= args.max_samples <= 10000 and 0 <= args.target_cv <= 1
                    and 0 <= args.warmup_ms <= 60000 and 0 <= args.repeats <= args.max_repeats
                    and args.telemetry_interval >= .02 and 0 < args.memory_fraction <= .95 and args.device >= 0):
                raise ValueError("invalid timing, memory, device or progress parameter")
            result = run_suite(selected(args), args)
            print(json.dumps(result, indent=2))
            nonfatal = {"ok", "measurement_only", "unsupported", "memory_budget"}
            return 1 if any(count for status, count in result["status_counts"].items() if status not in nonfatal) or result["missing"] else 0
        if args.command == "report": print(render_report(args.run, args.output)); return 0
        if args.command == "compare": print(compare(args.runs, args.output, allow_software_mismatch=args.allow_software_mismatch)); return 0
        if args.command == "coverage": print(json.dumps(summary(args.run), indent=2)); return 0
        if args.command == "extensions": print(json.dumps(EXTENSIONS, indent=2)); return 0
        if args.command == "probe":
            from .probes import compile_probes
            print(json.dumps(compile_probes(args.output, args.arch.split(","), args.ptxas), indent=2)); return 0
        if args.command in ("profile", "sanitize"):
            from .adapters import instrument
            result = instrument("sanitizer" if args.command == "sanitize" else "profiler", args.binary,
                     chosen_case(args), args.output, tool=args.tool, kernel_regex=args.kernel_regex, device=args.device, timeout=args.timeout)
            print(json.dumps(result, indent=2)); return 0 if result["returncode"] == 0 and (args.command != "sanitize" or result.get("verification_passed")) else 1
        if args.command == "disassemble":
            from .adapters import disassemble
            result = disassemble(args.binary, args.output)
            print(json.dumps(result, indent=2)); return 0 if result["returncode"] == 0 else 1
        if args.command == "cutlass-build":
            from .adapters import cutlass_build
            result = cutlass_build(args.source, args.build_dir, args.arch, args.kernels, args.jobs, args.nvcc)
            print(json.dumps(result, indent=2)); return 0 if result["returncode"] == 0 else 1
        if args.command == "cutlass":
            from .adapters import cutlass_run
            extra = args.args[1:] if args.args[:1] == ["--"] else args.args
            result = cutlass_run(args.profiler, args.output, extra, timeout=args.timeout, enumerate_only=args.enumerate)
            print(json.dumps(result, indent=2)); return 0 if result["returncode"] == 0 else 1
        if args.command == "external":
            from .adapters import external_run
            result = external_run(args.spec, args.output, args.timeout)
            print(json.dumps(result, indent=2)); return 0 if result["returncode"] == 0 else 1
        if args.command == "selftest":
            import unittest
            suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
            result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
            return 0 if result.wasSuccessful() else 1
    except (ValueError, OSError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted. Completed records are preserved.", file=sys.stderr)
        return 130
    return 0
