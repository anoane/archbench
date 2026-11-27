# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from archbench.model import Case
from archbench.runner import run_suite, invoke_worker, read_jsonl, repair_tail
from archbench.report import render_report, compare

MOCK = Path(__file__).parent / "fixtures/mock_worker.py"


def args(output, **updates):
    base = dict(binary=str(MOCK), output=str(output), resume=False, retry_failed=False,
                samples=3, warmup=0, target_ms=1, max_repeats=1, memory_fraction=.7,
                memory_limit_mib=0, device=0, timeout=2, profile="smoke", shuffle_seed=17,
                telemetry=False, telemetry_interval=.2, progress_every=1000, fail_fast=False)
    base.update(updates)
    return argparse.Namespace(**base)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="archbench_TEST_ONLY_")
        self.root = Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()

    def test_resume_does_not_duplicate(self):
        cases = [Case("scalar", {"op": "good"}), Case("scalar", {"op": "skip"})]
        out = self.root / "run"
        first = run_suite(cases, args(out))
        second = run_suite(cases, args(out, resume=True))
        self.assertEqual(first["recorded"], 2)
        self.assertEqual(second["validated_measured"], 1)
        self.assertEqual(len(list(read_jsonl(out / "results.jsonl"))), 2)

    def test_completed_and_partial_runs_automatically_report_every_selected_case(self):
        out = self.root / "complete"
        result = run_suite([Case("scalar", {"op": "good"}), Case("scalar", {"op": "skip"})], args(out))
        for name in ("report.html", "results.csv", "coverage.json", "pairs.json"):
            self.assertTrue((out / name).is_file(), name)
        self.assertEqual(
            Path(result["report_html"]).resolve(),
            (out / "report.html").resolve(),
        )
        self.assertEqual(json.loads((out / "coverage.json").read_text())["selected"], 2)
        partial = self.root / "partial"
        result = run_suite([Case("scalar", {"op": "fail", "seed": seed}) for seed in range(3)],
                           args(partial, fail_fast=True))
        self.assertEqual((result["selected"], result["recorded"], result["missing"]), (3, 1, 2))
        self.assertTrue((partial / "report.html").is_file())
        coverage = json.loads((partial / "coverage.json").read_text())
        self.assertEqual(coverage["missing"], 2)
        # Reporting must not append measurements for missing requested cases.
        self.assertEqual(len(list(read_jsonl(partial / "results.jsonl"))), 1)

    def test_resume_rejects_policy_change(self):
        out = self.root / "run"; cases = [Case("scalar", {"op": "good"})]
        run_suite(cases, args(out))
        with self.assertRaises(ValueError): run_suite(cases, args(out, resume=True, samples=4))

    def test_resume_rejects_case_change(self):
        out = self.root / "run"
        run_suite([Case("scalar", {"op": "good"})], args(out))
        with self.assertRaises(ValueError): run_suite([Case("scalar", {"op": "different"})], args(out, resume=True))

    def test_existing_run_requires_resume(self):
        out = self.root / "run"; cases = [Case("scalar", {"op": "good"})]
        run_suite(cases, args(out))
        with self.assertRaises(ValueError): run_suite(cases, args(out))

    def test_retry_failed_preserves_attempt_history(self):
        out = self.root / "run"; cases = [Case("scalar", {"op": "fail"})]
        run_suite(cases, args(out))
        result = run_suite(cases, args(out, resume=True, retry_failed=True))
        records = list(read_jsonl(out / "results.jsonl"))
        self.assertEqual([x["attempt"] for x in records], [1, 2])
        self.assertEqual(result["recorded"], 1)
        self.assertEqual(result["validated_measured"], 0)

    def test_timeout_record(self):
        result = invoke_worker(MOCK, ["--op", "timeout"], self.root, "timeout", .05)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["samples_ms"], [])

    def test_malformed_worker_output(self):
        result = invoke_worker(MOCK, ["--op", "invalid"], self.root, "invalid", 2)
        self.assertEqual(result["status"], "error")

    def test_partial_tail_repair_only(self):
        path = self.root / "partial.jsonl"
        path.write_text('{"a":1}\n{"a":')
        self.assertTrue(repair_tail(path))
        self.assertEqual(list(read_jsonl(path)), [{"a": 1}])
        self.assertFalse(repair_tail(path))
        path.write_text('bad\n')
        self.assertFalse(repair_tail(path))
        with self.assertRaises(ValueError): list(read_jsonl(path))

    def test_html_escapes_script_injection(self):
        out = self.root / "run"
        run_suite([Case("scalar", {"op": "xss"})], args(out))
        path = render_report(out)
        text = path.read_text()
        self.assertNotIn('</script><img src=x', text)
        self.assertIn('\\u003c/script>', text)
        self.assertTrue((out / "coverage.json").exists())
        self.assertTrue((out / "results.csv").exists())

    def test_comparison_omits_unverified_speedups(self):
        cases = [Case("scalar", {"op": "good"}), Case("scalar", {"op": "skip"})]
        runs = [self.root / "a", self.root / "b"]
        for out in runs: run_suite(cases, args(out))
        destination = self.root / "compare"
        compare(runs, destination)
        result = json.loads((destination / "comparison.json").read_text())
        good = next(x for x in result["rows"] if x["case"]["params"]["op"] == "good")
        skipped = next(x for x in result["rows"] if x["case"]["params"]["op"] == "skip")
        self.assertEqual(good["speedup_vs_first"], [1.0, 1.0])
        self.assertEqual(skipped["speedup_vs_first"], [None, None])

    def test_comparison_software_guard(self):
        runs = [self.root / "a", self.root / "b"]
        for out in runs: run_suite([Case("scalar", {"op": "good"})], args(out))
        path = runs[1] / "run.json"
        value = json.loads(path.read_text()); value["identity"]["software"]["cublas_version"] = "changed"
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError): compare(runs, self.root / "c")
        compare(runs, self.root / "c", allow_software_mismatch=True)
        self.assertTrue(json.loads((self.root / "c/comparison.json").read_text())["software_mismatch"])
