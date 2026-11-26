#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run reviewable hardware acceptance and sanitizer evidence, never infer passes."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from archbench.runner import atomic_json, read_jsonl, utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("80", "89", "120a"), required=True)
    device = parser.add_mutually_exclusive_group()
    device.add_argument("--device", type=int, default=0)
    device.add_argument("--gpu-uuid", help="full visible GPU UUID; preserves peer visibility")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--sanitizers", default="memcheck,racecheck,initcheck,synccheck")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()): parser.error("choose an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    binary = (args.binary or ROOT / "build/bin" / f"archbench_sm{args.arch}").resolve()
    if not binary.exists(): parser.error("build the matching worker first with ./bench build")
    commands = []
    def invoke(argv, label):
        command = [sys.executable, str(ROOT / "bench"), *map(str, argv)]
        code = subprocess.run(command, cwd=ROOT).returncode
        commands.append({"stage": label, "argv": command, "returncode": code})
        return code
    invoke(["doctor", "--output", output / "doctor.json"], "environment")
    suite = output / "acceptance"
    selector = ["--gpu-uuid", args.gpu_uuid] if args.gpu_uuid else ["--device", args.device]
    code = invoke(["run", "--matrix", ROOT / "configs/acceptance.json", "--binary", binary, "--output", suite,
                   *selector, "--quality", "quick", "--samples", 3, "--warmup", 1, "--target-ms", 2,
                   "--max-repeats", 128, "--timeout", 120, "--progress-every", 1], "numerical_acceptance")
    if (suite / "results.jsonl").exists():
        invoke(["report", suite], "report")
        selected_device = json.loads((suite / "run.json").read_text())["identity"]["policy"]["device"]
        # Sanitize executed representatives only. Unsupported is retained in acceptance,
        # and is never transformed into a sanitizer pass by an exit-zero skip.
        representatives = {}
        for row in read_jsonl(suite / "results.jsonl"):
            if row["status"] == "ok": representatives.setdefault(row["case"]["family"], row["case"]["id"])
        for family, case_id in representatives.items():
            for tool in filter(None, args.sanitizers.split(",")):
                if tool not in ("memcheck", "racecheck", "initcheck", "synccheck"): parser.error("invalid sanitizer")
                invoke(["sanitize", "--binary", binary, "--from-run", suite, "--case-id", case_id,
                        "--device", selected_device, "--tool", tool, "--timeout", 300,
                        "--output", output / "sanitizer" / family / tool], f"{family}_{tool}")
    report = {"created_at": utc_now(), "architecture_requested": args.arch, "commands": commands,
              "all_commands_succeeded": all(item["returncode"] == 0 for item in commands),
              "warning": "Command success does not mean every feature is supported. See acceptance/coverage.json for unsupported, skipped and unvalidated cases."}
    atomic_json(output / "acceptance_gate.json", report)
    return 0 if report["all_commands_succeeded"] else 1

if __name__ == "__main__": raise SystemExit(main())
