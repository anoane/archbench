# SPDX-License-Identifier: AGPL-3.0-or-later
import json
from pathlib import Path
import tempfile
import unittest
from archbench.catalog import generate, select, FAMILIES
from archbench.config import load_cases
from archbench.model import Case, is_measured_pass, is_comparable, validate_record


class CatalogTests(unittest.TestCase):
    def test_stable_identity_order_independent(self):
        self.assertEqual(Case("gemm", {"m": 4, "n": 8}).id, Case("gemm", {"n": 8, "m": 4}).id)
        self.assertNotEqual(Case("gemm", {"m": 4}).id, Case("gemm", {"m": 8}).id)
        self.assertNotEqual(Case("gemm", {"m": 4}, provider="x").id, Case("gemm", {"m": 4}).id)

    def test_roundtrip_identity_validation(self):
        item = Case("scalar", {"dtype": "f32", "op": "fma"})
        self.assertEqual(Case.from_dict(item.as_dict()), item)
        bad = {**item.as_dict(), "id": "scalar.invalid"}
        with self.assertRaises(ValueError): Case.from_dict(bad)

    def test_standard_all_families_unique(self):
        cases = list(generate("standard"))
        self.assertEqual(len(cases), len({item.id for item in cases}))
        self.assertEqual({item.family for item in cases}, set(FAMILIES))
        self.assertGreater(len(cases), 2000)

    def test_smoke_not_synthetic_results(self):
        for item in generate("smoke"):
            self.assertNotIn("samples_ms", item.as_dict())
            self.assertEqual(item.provider, "core")

    def test_shards_partition_exactly(self):
        cases = list(generate("smoke"))
        parts = [list(select(iter(cases), shard=(i, 7))) for i in range(7)]
        ids = [item.id for part in parts for item in part]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {item.id for item in cases})

    def test_filters_and_limit(self):
        result = list(select(generate("standard"), families=("scalar",), dtypes=("f64",), ops=("fma",), limit=1))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].params["dtype"], "f64")

    def test_bad_shard(self):
        with self.assertRaises(ValueError): list(select(iter([Case("x")]), shard=(3, 2)))

    def test_custom_matrix_expansion_and_deduplication(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "matrix.json"
            path.write_text(json.dumps({"schema_version": 1, "sweeps": [{"family": "scalar",
                "params": {"op": "fma"}, "axes": {"dtype": ["f32", "f64"], "ilp": [1, 4]}}]}))
            self.assertEqual(len(list(load_cases(path))), 4)

    def test_matrix_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "matrix.json"
            path.write_text(json.dumps({"schema_version": 1, "max_cases": 1, "sweeps": [{"family": "scalar", "axes": {"dtype": ["f32", "f64"]}}]}))
            with self.assertRaises(ValueError): list(load_cases(path))

    def test_latency_configuration_is_one_warp_one_chain(self):
        for item in generate("standard"):
            if item.params.get("mode") == "latency":
                self.assertEqual(item.params.get("threads", 32), 32)
                self.assertEqual(item.params.get("blocks", 1), 1)
                self.assertEqual(item.params.get("ilp", 1), 1)

    def test_conversion_uses_stream_size_not_ignored_iterations(self):
        for item in select(generate("smoke"), families=("convert",)):
            self.assertIn("size", item.params)
            self.assertNotIn("iterations", item.params)

    def test_vector_blas_uses_effective_length(self):
        for item in select(generate("standard"), families=("blas",)):
            if item.params.get("op") in ("axpy", "dot", "nrm2", "scal", "asum", "iamax", "iamin", "copy", "swap"):
                self.assertIn("elements", item.params)
                self.assertNotIn("m", item.params)
                self.assertNotIn("k", item.params)

    def test_argv_never_interpolates_shell(self):
        item = Case("scalar", {"dtype": "$(touch malicious)", "x": True})
        self.assertIn("$(touch malicious)", item.argv())
        self.assertIn("true", item.argv())


class ResultAccountingTests(unittest.TestCase):
    def baseline(self):
        return {"schema_version": 1, "status": "ok", "validation": "passed", "samples_ms": [1., 1.1, .9]}

    def test_pass_requires_correctness_and_timing(self):
        good = self.baseline()
        self.assertTrue(is_measured_pass(good)); self.assertEqual(validate_record(good), [])
        for change in ({"validation": "not_checked"}, {"samples_ms": []}, {"status": "unsupported"}, {"samples_ms": [0]}, {"samples_ms": [float("nan")]}):
            self.assertFalse(is_measured_pass({**good, **change}))

    def test_skipped_does_not_count(self):
        row = {"schema_version": 1, "status": "unsupported", "validation": "not_run", "samples_ms": []}
        self.assertFalse(is_measured_pass(row)); self.assertEqual(validate_record(row), [])

    def test_launch_has_no_numerical_validation(self):
        row = {"schema_version": 1, "status": "measurement_only", "validation": "not_applicable", "samples_ms": [.002]}
        self.assertFalse(is_measured_pass(row)); self.assertTrue(is_comparable(row)); self.assertEqual(validate_record(row), [])

    def test_unknown_status_rejected(self):
        self.assertIn("unknown status", validate_record({**self.baseline(), "status": "seems_fast"}))
