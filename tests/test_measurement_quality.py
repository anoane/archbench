# SPDX-License-Identifier: AGPL-3.0-or-later
"""Statistical definitions, work-normalized ratios, and host timing policy tests.

Synthetic samples here are test data, never benchmark evidence.
"""
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from archbench.report import _display_row, _partition_cases, comparison_basis, render_report, compare
from archbench.statistics import describe, assess_quality, rate_summary
import test_report_semantics as report_fixtures

record = report_fixtures.record


class TimingStatisticsTests(unittest.TestCase):
    def test_exact_median_interval_matches_independent_binomial_coverage(self):
        for n in range(1, 80):
            result = describe(range(1, n + 1))
            # Probability that a fixed interval [X_(k), X_(n-k+1)]
            # brackets the population median for a continuous distribution.
            candidates = []
            for k in range(1, n // 2 + 1):
                coverage = 1 - 2 * sum(math.comb(n, j) for j in range(k)) / 2**n
                if coverage >= .95:
                    candidates.append((k, coverage))
            if candidates:
                k, coverage = candidates[-1]
                self.assertEqual(result["median_ci95_ms"], [k, n - k + 1])
                self.assertAlmostEqual(result["median_ci_achieved_coverage"], coverage)
            else:
                self.assertEqual(result["median_ci95_ms"], [None, None])
                self.assertFalse(result["median_ci_finite"])
            self.assertEqual(result["median_ci_method"], "distribution_free_order_statistic")

    def test_small_sample_uncertainty_is_not_fabricated(self):
        one = describe([.002])
        self.assertIsNone(one["cv"])
        self.assertIsNone(one["stddev_ms"])
        self.assertEqual(assess_quality(one)["label"], "insufficient_samples")
        self.assertFalse(describe([1] * 5)["median_ci_finite"])
        self.assertTrue(describe([1] * 6)["median_ci_finite"])
        self.assertEqual(describe([1] * 6)["median_ci95_ms"], [1, 1])
        self.assertIn("adaptive stopping", one["median_ci_coverage_scope"])

    def test_descriptive_statistics_and_input_order_drift(self):
        result = describe([1, 2, 3, 4])
        self.assertEqual(result["median_us"], 2500)
        self.assertEqual(result["median_ns"], 2500000)
        self.assertEqual(result["mean_ms"], 2.5)
        self.assertEqual(result["mad_ms"], 1)
        self.assertAlmostEqual(result["stddev_ms"], math.sqrt(5 / 3))
        self.assertAlmostEqual(result["cv"], math.sqrt(5 / 3) / 2.5)
        self.assertAlmostEqual(result["p10_batch_ms"], 1.3)
        self.assertAlmostEqual(result["p90_batch_ms"], 3.7)
        self.assertEqual(result["half_run_drift_fraction"], .8)
        self.assertEqual(describe([4, 3, 2, 1])["half_run_drift_fraction"], -.8)
        self.assertTrue(assess_quality(result)["warnings"])

    def test_rate_confidence_bounds_reverse_timing_endpoints(self):
        result = rate_summary([1, 2, 3, 4, 5, 6, 7], 1e9, scale=1e12, unit="TFLOP/s")
        self.assertAlmostEqual(result["value"], .25)
        self.assertAlmostEqual(result["ci95"][0], 1 / 7)
        self.assertEqual(result["ci95"][1], 1)
        self.assertEqual(result["samples"], [1 / n for n in range(1, 8)])
        even = rate_summary([1, 2], 1e9)
        self.assertAlmostEqual(even["value"], 1000 / 1.5)
        self.assertEqual(even["mean"], 750)
        self.assertEqual(even["ci95"], [None, None])
        self.assertIsNone(rate_summary([1, 2, 3], 0))

    def test_quality_does_not_remove_noisy_samples_or_fabricate_stability(self):
        values = [1, 1, 1, 1, 9, 1, 1]
        result = describe(values)
        self.assertEqual(result["max_ms"], 9)
        self.assertEqual(result["count"], 7)
        self.assertEqual(assess_quality(result)["label"], "variable")
        self.assertEqual(assess_quality(result, target_cv=0)["label"], "not_requested")
        self.assertEqual(assess_quality(describe([1] * 7))["label"], "stable")

    def test_large_fixed_count_interval_is_finite_and_symmetric(self):
        result = describe(range(1, 10001))
        lower, upper = result["median_ci95_ms"]
        self.assertLess(lower, 5000.5)
        self.assertGreater(upper, 5000.5)
        self.assertEqual(lower + upper, 10001)
        self.assertGreaterEqual(result["median_ci_achieved_coverage"], .95)


class SemanticMetricTests(unittest.TestCase):
    def test_only_floating_arithmetic_has_tflops(self):
        samples = (1,) * 7
        floating = _display_row(record("tensor", {"dtype": "f32"}, work=2e12, samples=samples))
        integer = _display_row(record("tensor", {"dtype": "i8"}, work=2e12, samples=samples))
        function = _display_row(record("sfu", {"dtype": "f32", "op": "sin"}, work=2e12, samples=samples,
                                       unit="Gcall/s", metadata={"work_domain": "special_function"}))
        self.assertEqual(floating["tflops"], 2000)
        self.assertIsNone(floating["integer_tops"])
        self.assertEqual(integer["integer_tops"], 2000)
        self.assertIsNone(integer["tflops"])
        self.assertIsNone(function["tflops"])
        self.assertEqual(function["useful_rate_unit"], "Gcall/s")
        self.assertNotIn("FLOP", function["display"]["useful_rate"])

    def test_logical_decimal_and_binary_bandwidth(self):
        row = _display_row(record("transfer", {"op": "h2d"}, samples=(1000,) * 7,
                                  work=0, byte_count=2**30, unit="GB/s"))
        self.assertEqual(row["logical_gibytes_s"], 1)
        self.assertAlmostEqual(row["logical_gbytes_s"], 1.073741824)
        self.assertIsNone(row["tflops"])
        self.assertEqual(row["display"]["median_time"], "1e+06 µs")

    def test_blas_abs_norm_and_extrema_are_scoped_reduction_work_not_tflops(self):
        for operation in ("asum", "nrm2", "iamax", "iamin"):
            for dtype in ("f32", "c64"):
                row = _display_row(record("blas", {"op": operation, "dtype": dtype},
                                          unit="Gop/s", work=4096, samples=(.01,) * 7,
                                          metadata={"work_domain": "floating_point"}))
                self.assertEqual(row["work_domain"], "reduction_work")
                self.assertIsNone(row["tflops"])
                self.assertIsNone(row["integer_tops"])
                self.assertEqual(row["useful_rate_unit"], "Gop/s (reduction model)")
                self.assertEqual(row["useful_tera_rate_unit"], "Top/s (reduction model)")
                self.assertNotIn("FLOP", row["display"]["useful_rate"])

    def test_work_normalization_prevents_sm_count_latency_error(self):
        first = record("scalar", {"op": "fma"}, samples=(2,) * 7, work=100)
        second = record("scalar", {"op": "fma"}, samples=(3,) * 7, work=200)
        result = comparison_basis(first, second)
        self.assertEqual(result["basis"], "useful_work_rate")
        self.assertAlmostEqual(result["speedup"], 4 / 3)
        self.assertNotEqual(result["speedup"], 2 / 3)
        same = comparison_basis(first, first)
        self.assertEqual(same["basis"], "equal_work_latency")
        self.assertEqual(same["endpoint_bounds"], [1, 1])
        first.pop("work_per_iteration")
        self.assertEqual(comparison_basis(first, second)["basis"], "unavailable")

    def test_byte_rate_normalization_and_validation_exclusion(self):
        first = record("transfer", {"op": "h2d"}, samples=(2,) * 7, work=0, byte_count=1000)
        second = record("transfer", {"op": "h2d"}, samples=(4,) * 7, work=0, byte_count=4000)
        result = comparison_basis(first, second)
        self.assertEqual(result["basis"], "logical_bandwidth")
        self.assertEqual(result["speedup"], 2)
        second.update(status="validation_failed", validation="failed")
        self.assertIsNone(comparison_basis(first, second)["speedup"])
        row = _display_row(second)
        self.assertEqual(row["quality"]["label"], "stable")
        self.assertEqual(row["validation"], "failed")
        self.assertEqual(row["capability"]["availability"], "available")

    def test_unsupported_is_not_globally_impossible(self):
        row = _display_row(record("gemm", {"backend": "cublaslt", "dtype": "e2m1"},
                                  samples=(), status="unsupported", validation="not_run"))
        self.assertEqual(row["capability"]["availability"], "unavailable")
        self.assertEqual(row["capability"]["scope"], "backend_request")
        self.assertIn("does not mean globally impossible", row["capability"]["claim"])
        self.assertEqual(row["implementation"]["class"], "not_executed")

    def test_compiler_expansion_and_build_proof_are_not_runtime_proof(self):
        expanded = _display_row(record("tensor", {"dtype": "u4"}, execution="compiler_expanded_tensor_ptx",
                                       metadata={"codegen_evidence": "compiler_expanded_mma"}))
        self.assertEqual(expanded["implementation"]["class"], "compiler_emulation")
        native = _display_row(record("tensor", {"dtype": "f16"}, execution="requested_tensor_ptx",
                                     metadata={"tensor_codegen_evidence": "direct_mma_sass"}))
        self.assertEqual(native["implementation"]["class"], "native_build_instruction")
        self.assertIn("runtime", native["implementation"]["label"])


class HostTimingPolicyTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "host C++ compiler unavailable")
    def test_systems_validator_preserves_nonfinite_failure_without_false_zero(self):
        root = Path(__file__).resolve().parents[1]
        function = (root / "src/systems.cu").read_text().split("void check_value(result&", 1)[1].split("\nvoid begin_validation", 1)[0]
        source = r'''
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <cassert>
struct result { double max_abs_error=0,max_rel_error=0; std::string validation="passed",status="ok",reason;
  std::map<std::string,double> metrics; };
void check_value(result&''' + function + r'''
int main() {
  result exact; check_value(exact,1,1); assert(exact.max_abs_error == 0 && exact.validation == "passed");
  result mismatch; check_value(mismatch,1.25,1); assert(mismatch.max_abs_error == .25 && mismatch.validation == "failed");
  result bad; check_value(bad,std::numeric_limits<long double>::quiet_NaN(),1);
  assert(bad.validation == "failed" && bad.status == "validation_failed");
  assert(std::isnan(bad.max_abs_error) && std::isnan(bad.max_rel_error));
  assert(bad.metrics.at("nonfinite_validation_values") == 1);
  check_value(bad,1,1); assert(std::isnan(bad.max_abs_error));
}
'''
        with tempfile.TemporaryDirectory(prefix="archbench_host_validation_TEST_ONLY_") as directory:
            path, binary = Path(directory) / "validation.cpp", Path(directory) / "validation"
            path.write_text(source)
            subprocess.run(["g++", "-std=c++17", "-O2", str(path), "-o", str(binary)],
                           check=True, capture_output=True, text=True)
            subprocess.run([str(binary)], check=True, capture_output=True, text=True)

    @unittest.skipUnless(shutil.which("g++"), "host C++ compiler unavailable")
    def test_real_host_statistics_and_adaptive_stopping(self):
        source = r'''
#include "archbench/timing_stats.hpp"
#include <cassert>
#include <cmath>
int main() {
  const auto a = archbench::analyze_timing({1,2,3,4});
  assert(a.valid && a.count == 4 && a.median == 2.5 && a.mean == 2.5 && a.mad == 1);
  assert(std::abs(a.stddev - std::sqrt(5./3)) < 1e-14);
  assert(std::abs(a.p10 - 1.3) < 1e-14 && std::abs(a.p90 - 3.7) < 1e-14);
  assert(!archbench::analyze_timing({1,0,2}).valid);
  assert(std::isnan(archbench::analyze_timing({1}).cv));
  archbench::timing_policy p; p.minimum_samples=3; p.maximum_samples=5; p.target_cv=.05;
  assert(p.needs_more({1,1}));
  assert(!p.needs_more({1,1,1}));
  assert(p.needs_more({1,2,1}));
  assert(!p.needs_more({1,2,1,2,1}));
  p.target_cv=0; assert(!p.needs_more({1,2,1}));
  p.repeats=20; assert(p.repeats == 20);
}
'''
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="archbench_host_timing_TEST_ONLY_") as directory:
            source_path, binary = Path(directory) / "timing.cpp", Path(directory) / "timing"
            source_path.write_text(source)
            subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-pedantic", "-I", str(root / "include"),
                            str(source_path), "-o", str(binary)], check=True, capture_output=True, text=True)
            subprocess.run([str(binary)], check=True, capture_output=True, text=True)


class LargeReportTests(unittest.TestCase):
    setUp = report_fixtures.ReportSemanticsTests.setUp
    tearDown = report_fixtures.ReportSemanticsTests.tearDown
    save_run = report_fixtures.ReportSemanticsTests.save_run
    report = report_fixtures.ReportSemanticsTests.report

    def test_every_case_survives_old_two_thousand_row_boundary(self):
        records = [record("scalar", {"op": "fma", "elements": index + 1}, samples=(1,) * 7)
                   for index in range(2007)]
        rows, exported, document = self.report(records)
        self.assertEqual(len(rows), 2007)
        self.assertEqual(len(exported), 2007)
        self.assertEqual({row["id"] for row in rows}, {row["case"]["id"] for row in records})
        self.assertIn('id="page-size"', document)
        self.assertIn('id="page"', document)
        self.assertIn('id="case-details"', document)
        self.assertNotIn('slice(0,2000)', document)
        if shutil.which("node"):
            script = re.findall(r'<script>(.*?)</script>', document, re.DOTALL)[0]
            path = self.root / "report-script.js"
            path.write_text(script)
            subprocess.run(["node", "--check", str(path)], check=True, capture_output=True, text=True)

    def test_saved_plan_derives_legacy_pairs_without_changing_raw_records(self):
        requested = record("gemm", {"backend": "cublas", "dtype": "f32", "m": 8}, samples=(1,) * 7)
        software = record("gemm", {"backend": "software", "dtype": "f32", "m": 8}, samples=(2,) * 7,
                          execution="software_simt")
        rows, _, document = self.report([requested, software])
        pairs = json.loads((self.root / "run/pairs.json").read_text())
        self.assertEqual(pairs["counts"]["both_validated"], 1)
        pair = pairs["rows"][0]
        self.assertEqual(pair["software_speedup_vs_requested"]["speedup"], .5)
        self.assertTrue(pair["numerical_problem_covered"])
        self.assertEqual(rows[0]["pairing"]["software_case_id"], software["case"]["id"])
        self.assertIn("Inspect counterpart", document)

    def test_hash_partition_has_bounded_pages_and_exact_case_routing(self):
        from archbench.model import Case
        cases = [Case("scalar", {"elements": index}) for index in range(1500)]
        planned = {case.id: case for case in cases}
        widths, groups = _partition_cases(planned, maximum=17)
        members = [identifier for group in groups.values() for identifier in group]
        self.assertEqual(len(members), len(set(members)))
        self.assertEqual(set(members), set(planned))
        self.assertLessEqual(max(map(len, groups.values())), 17)
        for filename, identifiers in groups.items():
            for identifier in identifiers:
                family, digest = identifier.split(".")
                self.assertEqual(filename, f"{family}.{digest[:widths[family]]}.html")

    def test_large_report_and_comparison_keep_all_cases_in_bounded_offline_shards(self):
        records = [record("scalar", {"op": "fma", "elements": index + 1}, samples=(1,) * 7)
                   for index in range(120)]
        first, second = [self.save_run(records, name) for name in ("large-a", "large-b")]
        # One planned case has no result in either run; it must remain reachable.
        for directory in (first, second):
            path = directory / "results.jsonl"
            path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
        with patch("archbench.report.SINGLE_REPORT_LIMIT", 10):
            index_path = render_report(first)
            compare([first, second], self.root / "large-comparison")
        manifest = json.loads((first / "report_index.json").read_text())
        self.assertEqual(manifest["overview"]["selected"], 120)
        self.assertEqual(manifest["overview"]["missing"], 1)
        seen, documents = [], [index_path.read_text()]
        for page in manifest["pages"]:
            text = (first / page["file"]).read_text()
            documents.append(text)
            data = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', text, re.DOTALL).group(1))
            self.assertLessEqual(len(data["rows"]), manifest["maximum_rows_per_page"])
            seen.extend(data["rows"])
        self.assertEqual(len(seen), 120)
        missing = [row for row in seen if row["status"] == "missing"]
        self.assertEqual(len(missing), 1)
        self.assertIsNone(missing[0]["max_abs_error"])
        comparison_root = self.root / "large-comparison"
        comparison = json.loads((comparison_root / "comparison.json").read_text())
        self.assertNotIn("rows", comparison)
        self.assertEqual(comparison["selected_cases"], 120)
        compared = []
        for page in comparison["pages"]:
            text = (comparison_root / page["file"]).read_text()
            documents.append(text)
            part = json.loads((comparison_root / page["file"]).with_suffix(".json").read_text())
            compared.extend(part["rows"])
        self.assertEqual(len(compared), 120)
        self.assertEqual(sum(row["statuses"] == ["missing", "missing"] for row in compared), 1)
        # Parse each distinct generated script, without requiring a GPU or browser.
        if shutil.which("node"):
            scripts = set(script for text in documents for script in re.findall(r'<script>(.*?)</script>', text, re.DOTALL))
            for index, script in enumerate(scripts):
                path = self.root / f"shard-script-{index}.js"
                path.write_text(script)
                subprocess.run(["node", "--check", str(path)], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
