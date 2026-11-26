# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for result identity and evidence handling; no GPU is used.

Every executable fixture and every result directory lives in TemporaryDirectory.
The pass-through sanitizer fixture tests adapter classification only and performs
no CUDA instrumentation. Synthetic fixture timings are never benchmark evidence.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from archbench.adapters import capture, instrument, parse_cutlass_csv
from archbench.config import load_cases
from archbench.model import Case, is_comparable, is_measured_pass, validate_record
from archbench.report import compare, summary
from archbench.runner import read_jsonl, read_run, run_suite


FIXTURE = Path(__file__).parent / "fixtures" / "mock_worker.py"


class ReviewRegressions(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="archbench_REVIEW_TEST_ONLY_")
        self.root = Path(self.temporary.name)
        self.worker = self.root / "mock_worker.py"
        shutil.copy2(FIXTURE, self.worker)
        self.worker.chmod(self.worker.stat().st_mode | 0o100)
        self.good_case = Case("scalar", {"op": "good"})

    def tearDown(self):
        self.temporary.cleanup()

    def arguments(self, output, **updates):
        values = dict(binary=str(self.worker), output=str(output), resume=False,
                      retry_failed=False, samples=3, warmup=0, target_ms=1,
                      max_repeats=1, memory_fraction=.7, memory_limit_mib=0,
                      device=0, timeout=5, profile="smoke", shuffle_seed=17,
                      telemetry=False, telemetry_interval=.2, progress_every=1000,
                      fail_fast=False)
        values.update(updates)
        return argparse.Namespace(**values)

    def run_cases(self, output, cases=None, **updates):
        # Tests must not query an installed system GPU or depend on its driver.
        with patch("archbench.runner.shutil.which", return_value=None), contextlib.redirect_stderr(io.StringIO()):
            return run_suite(cases or [self.good_case], self.arguments(output, **updates))

    def change_first_result(self, output, mutate):
        path = output / "results.jsonl"
        rows = list(read_jsonl(path))
        mutate(rows[0])
        path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
        return path.read_bytes()

    def sanitizer_fixture(self, worker=None):
        wrapper = self.root / "mock_sanitizer.py"
        wrapper.write_text(
            f"#!{sys.executable}\n"
            "# TEST ONLY: transparent process wrapper; no sanitizer work occurs.\n"
            "import os, sys\n"
            "assert sys.argv[1:5] == ['--tool', 'memcheck', '--error-exitcode', '99']\n"
            "os.execv(sys.argv[5], sys.argv[5:])\n"
        )
        wrapper.chmod(0o700)
        chosen_worker = worker or self.worker

        def executable(value):
            return str(wrapper if value == "compute-sanitizer" else chosen_worker)

        return patch("archbench.adapters.checked_executable", side_effect=executable)

    def test_reserved_options_cannot_override_case_identity_or_run_policy(self):
        reserved = ("family", "info", "help", "device", "samples", "warmup",
                    "target-ms", "max-repeats", "memory-fraction", "memory-limit-mib")
        for name in reserved:
            with self.subTest(name=name), self.assertRaises(ValueError):
                Case("gemm", {name: "scalar"})
        valid = Case("gemm", {"dtype": "f32", "trans-a": "T", "m": 128})
        self.assertEqual(valid.argv().count("--family"), 1)
        self.assertIn("--trans-a", valid.argv())

    def test_non_core_provider_is_not_dispatched_to_core_worker(self):
        output = self.root / "wrong_provider"
        with self.assertRaises(ValueError):
            self.run_cases(output, [Case("gemm", {"m": 64}, provider="cutlass")])
        self.assertFalse((output / "results.jsonl").exists())
        self.assertFalse((output / "run.json").exists())

    def test_duplicate_jsonl_plan_is_rejected_before_execution(self):
        plan = self.root / "duplicate_plan.jsonl"
        line = json.dumps(self.good_case.as_dict()) + "\n"
        plan.write_text(line + line)
        output = self.root / "duplicates"
        with self.assertRaises(ValueError):
            self.run_cases(output, list(load_cases(plan)))
        self.assertFalse((output / "results.jsonl").exists())

    def test_nonfinite_cutlass_runtime_never_counts_and_remains_serializable(self):
        path = self.root / "provider.csv"
        path.write_text("Status,Disposition,Runtime\n"
                        "Success,Passed,inf\nSuccess,Passed,-inf\n"
                        "Success,Passed,nan\nSuccess,Passed,0\n"
                        "Success,Passed,-1\nSuccess,Passed,0.25\n")
        rows = parse_cutlass_csv(path)
        self.assertEqual([row["validated_measured"] for row in rows],
                         [False, False, False, False, False, True])
        self.assertEqual([row["runtime_ms"] for row in rows[:3]], [None, None, None])
        # Regression: NaN used to survive parsing and crash the adapter write.
        self.assertEqual(len(json.loads(json.dumps(rows, allow_nan=False))), 6)

    def test_boolean_samples_are_not_timings(self):
        for status, validation in (("ok", "passed"), ("measurement_only", "not_applicable")):
            with self.subTest(status=status):
                row = dict(schema_version=1, status=status, validation=validation, samples_ms=[True])
                self.assertFalse(is_measured_pass(row))
                self.assertFalse(is_comparable(row))
                self.assertTrue(validate_record(row))
        self.assertTrue(is_measured_pass(dict(status="ok", validation="passed", samples_ms=[1, .5])))

    def test_adapter_refuses_to_overwrite_existing_evidence(self):
        output = self.root / "evidence"
        capture([str(self.worker), "--op", "good"], output, timeout=5)
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        with self.assertRaises(ValueError):
            capture([str(self.worker), "--op", "skip"], output, timeout=5)
        after = {path.name: path.read_bytes() for path in output.iterdir()}
        self.assertEqual(after, before)

    def test_partial_evidence_is_preserved_if_invocation_metadata_is_missing(self):
        output = self.root / "partial_evidence"
        output.mkdir()
        (output / "stdout.txt").write_text("preexisting partial evidence\n")
        with self.assertRaises(ValueError):
            capture([str(self.worker), "--op", "good"], output, timeout=5)
        self.assertEqual((output / "stdout.txt").read_text(), "preexisting partial evidence\n")

    def test_resume_rejects_changed_shuffle_seed_without_appending_results(self):
        output = self.root / "changed_seed"
        self.run_cases(output)
        before = (output / "results.jsonl").read_bytes()
        with self.assertRaises(ValueError):
            self.run_cases(output, resume=True, shuffle_seed=18)
        self.assertEqual((output / "results.jsonl").read_bytes(), before)

    def test_first_run_rejects_malformed_device_metadata_cleanly(self):
        for index, metadata in enumerate((None, [], {}, {"uuid_hex": "bad"}, {"uuid_hex": False})):
            with self.subTest(metadata=metadata):
                output = self.root / f"bad_info_{index}"
                reply = dict(schema_version=1, status="ok", execution="metadata_only", metadata=metadata)
                with patch("archbench.runner.invoke_worker", return_value=reply), self.assertRaises(ValueError):
                    self.run_cases(output)
                self.assertFalse((output / "run.json").exists())

    def test_tampered_case_contents_are_rejected_by_resume_and_reader(self):
        output = self.root / "case_tamper"
        self.run_cases(output)
        before = self.change_first_result(output, lambda row: row["case"]["params"].update(op="different"))
        with self.assertRaises(ValueError):
            read_run(output)
        with self.assertRaises(ValueError):
            self.run_cases(output, resume=True)
        self.assertEqual((output / "results.jsonl").read_bytes(), before)

    def test_foreign_run_id_is_rejected_by_resume_and_comparison(self):
        first, second = self.root / "run_a", self.root / "run_b"
        self.run_cases(first)
        self.run_cases(second)
        first_record = next(read_jsonl(first / "results.jsonl"))
        (second / "results.jsonl").write_text(json.dumps(first_record) + "\n")
        with self.assertRaises(ValueError):
            self.run_cases(second, resume=True)
        with self.assertRaises(ValueError):
            compare([first, second], self.root / "forbidden_comparison")

    def test_unplanned_case_cannot_inflate_recorded_coverage(self):
        output = self.root / "outside_plan"
        self.run_cases(output)
        self.change_first_result(output, lambda row: row.update(case=Case("scalar", {"op": "other"}).as_dict()))
        with self.assertRaises(ValueError):
            summary(output)
        with self.assertRaises(ValueError):
            self.run_cases(output, resume=True)

    def test_invalid_saved_numeric_result_is_not_accepted_on_resume(self):
        output = self.root / "bad_samples"
        self.run_cases(output)
        self.change_first_result(output, lambda row: row.update(samples_ms=[True]))
        with self.assertRaises(ValueError):
            self.run_cases(output, resume=True)
        with self.assertRaises(ValueError):
            read_run(output)

    def test_saved_plan_tampering_is_rejected_by_reader_and_resume(self):
        output = self.root / "changed_plan"
        self.run_cases(output)
        (output / "plan.jsonl").write_text(json.dumps(Case("scalar", {"op": "different"}).as_dict()) + "\n")
        with self.assertRaises(ValueError):
            read_run(output)
        with self.assertRaises(ValueError):
            self.run_cases(output, resume=True)

    def test_skipped_case_is_recorded_but_never_a_validated_measurement(self):
        output = self.root / "skipped"
        outcome = self.run_cases(output, [Case("scalar", {"op": "skip"})])
        self.assertEqual(outcome["recorded"], 1)
        self.assertEqual(outcome["missing"], 0)
        self.assertEqual(outcome["validated_measured"], 0)
        self.assertEqual(summary(output)["validated_measured"], 0)

    def test_sanitizer_exit_zero_does_not_validate_a_skipped_case(self):
        with self.sanitizer_fixture():
            outcome = instrument("sanitizer", str(self.worker), Case("scalar", {"op": "skip"}),
                                 self.root / "sanitizer_skip", timeout=5)
        self.assertEqual(outcome["returncode"], 0)
        self.assertFalse(outcome["benchmark_executed"])
        self.assertFalse(outcome["verification_passed"])
        self.assertEqual(outcome["worker_outcomes"][0]["status"], "unsupported")
        self.assertFalse(outcome["counts_as_core_benchmark_result"])

    def test_sanitizer_does_not_accept_ok_without_numerical_validation(self):
        unvalidated_worker = self.root / "unvalidated_mock_worker.py"
        text = FIXTURE.read_text().replace("'validation': 'passed'", "'validation': 'not_run'")
        self.assertNotEqual(text, FIXTURE.read_text())
        unvalidated_worker.write_text(text)
        unvalidated_worker.chmod(0o700)
        with self.sanitizer_fixture(unvalidated_worker):
            outcome = instrument("sanitizer", str(unvalidated_worker), self.good_case,
                                 self.root / "sanitizer_unvalidated", timeout=5)
        self.assertEqual(outcome["returncode"], 0)
        self.assertFalse(outcome["verification_passed"])

    def test_sanitizer_positive_control_remains_separate_from_core_results(self):
        with self.sanitizer_fixture():
            outcome = instrument("sanitizer", str(self.worker), self.good_case,
                                 self.root / "sanitizer_good", timeout=5)
        self.assertEqual(outcome["returncode"], 0)
        self.assertTrue(outcome["benchmark_executed"])
        self.assertTrue(outcome["verification_passed"])
        self.assertFalse(outcome["counts_as_core_benchmark_result"])


if __name__ == "__main__":
    unittest.main()
