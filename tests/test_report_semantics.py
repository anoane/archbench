# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reporting must distinguish absent measurements from actual numerical zero."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import tempfile
import unittest

from archbench.model import Case
from archbench.report import compare, render_report


def record(family, params, *, samples=(0.002593085713140162,) * 3,
           status="ok", validation="passed", execution="compiler_lowered",
           unit="Gop/s", work=4099, byte_count=0, absolute=0, relative=0, metadata=None):
    metrics = {}
    if samples:
        median = statistics.median(samples)
        metrics = {"median_ms": median, "useful_gops": work / (median * 1e6),
                   "logical_gbytes_s": byte_count / (median * 1e6)}
    return {"schema_version": 1, "case": Case(family, params).as_dict(),
            "status": status, "validation": validation, "execution": execution,
            "unit": unit, "work_per_iteration": work, "bytes_per_iteration": byte_count,
            "samples_ms": list(samples), "metrics": metrics, "metadata": metadata or {},
            "max_abs_error": absolute, "max_rel_error": relative, "reason": ""}


class ReportSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="archbench_report_TEST_ONLY_")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def save_run(self, records, name="run"):
        directory = self.root / name
        directory.mkdir()
        records = copy.deepcopy(records)
        plan = [row["case"] for row in records]
        selection = hashlib.sha256("\n".join(row["id"] for row in plan).encode()).hexdigest()
        metadata = {"schema_version": 1, "run_id": name, "selected_cases": len(plan),
                    "device": {"device_name": "Synthetic test device"},
                    "identity": {"selection_sha256": selection, "policy": {},
                                 "environment": {}, "software": {}}}
        for row in records: row["run_id"] = name
        (directory / "run.json").write_text(json.dumps(metadata))
        (directory / "plan.jsonl").write_text("".join(json.dumps(row) + "\n" for row in plan))
        (directory / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
        return directory

    def report(self, records):
        directory = self.save_run(records)
        original = {name: (directory / name).read_bytes()
                    for name in ("run.json", "plan.jsonl", "results.jsonl")}
        document = render_report(directory).read_text()
        payload = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>',
                                      document, re.DOTALL).group(1))
        for name, content in original.items():
            self.assertEqual((directory / name).read_bytes(), content)
        with (directory / "results.csv").open(newline="") as stream:
            exported = list(csv.DictReader(stream))
        return payload["rows"], exported, document

    def test_copy_has_bandwidth_but_no_arithmetic_rate(self):
        original = record("memory", {"op": "cp_async", "elements": 4100},
                          unit="GB/s", work=0, byte_count=32800)
        rows, exported, document = self.report([original])
        row, csv_row = rows[0], exported[0]
        self.assertIsNone(row["useful_gops"])
        self.assertEqual(row["display"]["useful_rate"], "N/A")
        self.assertEqual(row["metrics"]["useful_gops"], 0)  # Original value remains inspectable.
        self.assertAlmostEqual(row["logical_gbytes_s"], 12.649022681274973)
        self.assertEqual(row["display"]["median_time"], "2.59309 µs")
        self.assertEqual(csv_row["useful_gops"], "")
        self.assertEqual(float(csv_row["median_ms"]), original["metrics"]["median_ms"])
        self.assertEqual(csv_row["unit"], "GB/s")
        self.assertEqual(csv_row["work_per_iteration"], "0")
        self.assertEqual(csv_row["bytes_per_iteration"], "32800")
        self.assertIn("Median time", document)
        self.assertIn("Useful rate (unit shown)", document)
        self.assertIn("CSV times remain in milliseconds", document)

    def test_atomic_rate_uses_its_actual_unit_and_no_byte_model(self):
        rows, exported, _ = self.report([record("atomic", {"op": "add", "dtype": "u16"},
                                              unit="Gatomic/s", work=4099, byte_count=0)])
        row = rows[0]
        self.assertTrue(row["display"]["useful_rate"].endswith(" Gatomic/s"))
        self.assertEqual(row["display"]["logical_rate"], "N/A")
        self.assertEqual(exported[0]["logical_gbytes_s"], "")
        self.assertEqual(row["max_abs_error"], 0)
        self.assertTrue(row["abs_error_available"])
        self.assertIn("max abs error: 0; max rel error: 0", row["notes"])
        self.assertNotIn("0.00000", row["notes"])

    def test_skipped_record_does_not_claim_execution_or_zero_error(self):
        rows, exported, _ = self.report([record("tensor", {"dtype": "e2m1"}, samples=(),
                                              status="unsupported", validation="not_run", work=0)])
        row = rows[0]
        self.assertEqual(row["execution"], "not_executed")
        self.assertEqual(row["recorded_execution"], "compiler_lowered")
        self.assertEqual(row["recorded_max_abs_error"], 0)
        self.assertIsNone(row["max_abs_error"])
        self.assertIsNone(row["max_rel_error"])
        self.assertEqual(row["display"]["median_time"], "N/A")
        self.assertEqual(exported[0]["max_abs_error"], "")
        self.assertEqual(exported[0]["abs_error_available"], "False")
        self.assertEqual(exported[0]["recorded_execution"], "compiler_lowered")

    def test_measured_launch_keeps_small_nonzero_rate_and_no_numeric_error(self):
        original = record("launch", {"op": "graph", "graph-nodes": 8, "iterations": 16},
                          samples=(0.11607516089151072,) * 3, status="measurement_only",
                          validation="not_applicable", execution="runtime_api", unit="Glaunch/s", work=16)
        rows, exported, _ = self.report([original])
        self.assertEqual(rows[0]["display"]["useful_rate"], "0.000137842 Glaunch/s")
        self.assertEqual(rows[0]["display"]["median_time"], "116.075 µs")
        self.assertEqual(rows[0]["execution"], "runtime_api")
        self.assertIsNone(rows[0]["max_abs_error"])
        self.assertEqual(exported[0]["max_abs_error"], "")

    def test_validation_failure_retains_error_but_unrun_error_does_not(self):
        failed = record("gemm", {"dtype": "f32"}, samples=(), status="validation_failed",
                        validation="failed", absolute=0.25, relative=0.0625)
        failed["reason"] = "CPU reference mismatch"
        unrun = record("gemm", {"dtype": "f64"}, samples=(), status="error",
                       validation="not_run", absolute=0, relative=0)
        rows, exported, _ = self.report([failed, unrun])
        self.assertEqual(rows[0]["max_abs_error"], 0.25)
        self.assertEqual(rows[0]["max_rel_error"], 0.0625)
        self.assertIn("CPU reference mismatch", rows[0]["notes"])
        self.assertIn("max abs error: 0.25", rows[0]["notes"])
        self.assertIsNone(rows[1]["max_abs_error"])
        self.assertEqual(exported[1]["max_rel_error"], "")

    def test_legacy_quant_relative_error_is_unavailable(self):
        legacy = record("quant", {"format": "u4"}, absolute=2.384185791015625e-7,
                        relative=0, unit="Gvalue/s")
        modern = record("quant", {"format": "s4"}, absolute=0, relative=0, unit="Gvalue/s",
                        metadata={"relative_error_definition": "max |actual-reference| / max(1e-8, |reference|)"})
        rows, exported, _ = self.report([legacy, modern])
        self.assertEqual(rows[0]["recorded_max_rel_error"], 0)
        self.assertIsNone(rows[0]["max_rel_error"])
        self.assertFalse(rows[0]["rel_error_available"])
        self.assertIn("legacy quant record", rows[0]["error_note"])
        self.assertEqual(exported[0]["max_rel_error"], "")
        self.assertEqual(rows[1]["max_rel_error"], 0)
        self.assertTrue(rows[1]["rel_error_available"])
        self.assertEqual(exported[1]["max_rel_error"], "0")

    def test_primary_bandwidth_label_does_not_turn_loads_into_arithmetic(self):
        rows, exported, _ = self.report([record("memory", {"op": "shared_bank", "stride": 32},
                                              unit="GB/s", work=8192, byte_count=32768)])
        self.assertEqual(rows[0]["unit"], "GB/s")
        self.assertEqual(rows[0]["useful_rate_unit"], "Gload/s")
        self.assertTrue(rows[0]["display"]["useful_rate"].endswith(" Gload/s"))
        self.assertEqual(exported[0]["useful_rate_unit"], "Gload/s")

    def test_csv_statistics_keep_original_millisecond_scale(self):
        samples = (1.0, 1.0, 1.125, 1.25, 1.5, 1.75, 2.0)
        rows, exported, _ = self.report([record("scalar", {"op": "fma"}, samples=samples)])
        row = exported[0]
        self.assertEqual(rows[0]["display"]["median_time"], "1250 µs")
        self.assertEqual(float(row["median_ms"]), 1.25)
        self.assertEqual(float(row["min_ms"]), 1.0)
        self.assertEqual(float(row["max_ms"]), 2.0)
        self.assertEqual(int(row["sample_count"]), 7)
        self.assertEqual(float(row["mean_ms"]), statistics.mean(samples))
        self.assertLessEqual(float(row["median_ci95_low_ms"]), 1.25)
        self.assertGreaterEqual(float(row["median_ci95_high_ms"]), 1.25)

    def test_comparison_preserves_raw_execution_and_eligibility(self):
        skipped = record("tensor", {"dtype": "e2m1"}, samples=(), status="unsupported",
                         validation="not_run", work=0)
        measured = record("scalar", {"op": "fma"})
        runs = [self.save_run([skipped, measured], name) for name in ("first", "second")]
        compare(runs, self.root / "comparison")
        rows = json.loads((self.root / "comparison/comparison.json").read_text())["rows"]
        skipped_row = next(row for row in rows if row["case"]["family"] == "tensor")
        self.assertEqual(skipped_row["execution"], ["not_executed", "not_executed"])
        self.assertEqual(skipped_row["recorded_execution"], ["compiler_lowered", "compiler_lowered"])
        self.assertEqual(skipped_row["speedup_vs_first"], [None, None])
        measured_row = next(row for row in rows if row["case"]["family"] == "scalar")
        self.assertEqual(measured_row["speedup_vs_first"], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
