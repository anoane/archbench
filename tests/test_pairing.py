# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check experiment identity and software coverage without inventing GPU timings."""
from pathlib import Path
import tempfile
import unittest

from archbench.cli import make_parser, resolve_measurement_policy, selected
from archbench.config import load_cases
from archbench.model import Case, canonical
from archbench.pairing import (operation_id, pair_summary, pairing_index,
                              pairing_metadata, software_counterpart,
                              with_software_counterparts)

ROOT = Path(__file__).resolve().parents[1]


class CounterpartTests(unittest.TestCase):
    def test_all_numeric_controls_are_preserved_and_ids_remain_distinct(self):
        params = {"backend": "cublaslt", "dtype": "fp8_e4m3", "dtype-b": "fp8_e5m2",
                  "m": 31, "n": 65, "k": 128, "batch": 3, "batch-gap": 7,
                  "trans-a": "T", "trans-b": "N", "ld-pad": 16, "alpha": .75,
                  "beta": .25, "epilogue": "bias_relu", "output-dtype": "f32",
                  "scale-mode": "block32", "seed": 37, "algorithms": 8}
        original = Case("gemm", params, ("regression",))
        software = software_counterpart(original)
        self.assertEqual(software.params, {**params, "backend": "software"})
        self.assertEqual(software.tags, original.tags)
        self.assertNotEqual(original.id, software.id)
        self.assertEqual(operation_id(original), operation_id(software))
        different = Case("gemm", {**params, "seed": 38})
        self.assertNotEqual(operation_id(original), operation_id(different))

    def test_library_internal_emulation_has_only_a_mathematical_reference(self):
        for controls in ({"compute": "fixedpoint"}, {"compute": "bf16x9"},
                         {"fast-accum": True}, {"fast-accum": "true"}):
            with self.subTest(controls=controls):
                source = Case("gemm", {"backend": "cublas", "dtype": "f64", **controls})
                baseline = software_counterpart(source)
                self.assertEqual(baseline.params["software-policy"], "mathematical-reference")
                index = pairing_index([source, baseline])
                for case in (source, baseline):
                    info = pairing_metadata(case, index)
                    self.assertEqual(info["relation"], "mathematical_reference")
                    self.assertFalse(info["bitwise_equivalence_claim"])
                    self.assertFalse(info["library_algorithm_equivalence_claim"])

    def test_quantized_gemm_pairs_preserve_preprocessing_scope(self):
        for mode in ("predequant_gemm", "end_to_end"):
            source = Case("quant_gemm", {"format": "nf4", "mode": mode, "group": 32, "k": 65})
            pair = software_counterpart(source)
            self.assertEqual(pair.params, {**source.params, "backend": "software"})
        self.assertIsNone(software_counterpart(Case("quant_gemm", {"mode": "fused"})))

    def test_acceptance_requests_are_retained_and_have_explicit_counterparts(self):
        original = list(load_cases(ROOT / "configs/acceptance.json"))
        expanded = list(with_software_counterparts(original))
        self.assertEqual(len(original), 192)
        self.assertEqual(len({case.id for case in expanded}), len(expanded))
        self.assertTrue({case.id for case in original}.issubset(case.id for case in expanded))
        counterparts = [software_counterpart(case) for case in original]
        expected = {case.id for case in original} | {case.id for case in counterparts if case}
        self.assertEqual({case.id for case in expanded}, expected)
        self.assertGreater(len(expanded), len(original))
        self.assertEqual([case.as_dict() for case in with_software_counterparts(expanded)],
                         [case.as_dict() for case in expanded])

    def test_unsupported_native_plus_valid_software_is_not_a_native_pass(self):
        requested = Case("tensor", {"dtype": "fp8_e4m3", "backend": "native"})
        software = software_counterpart(requested)
        records = {requested.id: {"status": "unsupported", "validation": "not_run", "samples_ms": []},
                   software.id: {"status": "ok", "validation": "passed", "samples_ms": [1, 2, 3]}}
        report = pair_summary([requested, software], records)
        self.assertEqual(report["counts"]["software_only_validated"], 1)
        row = report["rows"][0]
        self.assertFalse(row["requested_validated"])
        self.assertTrue(row["software_validated"])
        self.assertTrue(row["numerical_problem_covered"])
        self.assertEqual(row["requested_status"], "unsupported")

    def test_missing_software_is_not_invented(self):
        requested = Case("fft", {"dtype": "f32", "n": 1024})
        report = pair_summary([requested], {})
        self.assertEqual(report["counts"]["software_pairs_not_selected"], 1)
        self.assertEqual(report["rows"][0]["software_status"], "not_selected")
        self.assertFalse(report["rows"][0]["numerical_problem_covered"])

    def test_shared_software_has_separate_pair_and_operation_denominators(self):
        a = Case("gemm", {"backend": "cublas", "dtype": "f32", "m": 32, "n": 32, "k": 32})
        b = Case("gemm", {**a.params, "backend": "cublaslt"})
        software = software_counterpart(a)
        self.assertEqual(software.id, software_counterpart(b).id)
        records = {software.id: {"status": "ok", "validation": "passed", "samples_ms": [1]}}
        report = pair_summary([a, b, software], records)
        self.assertEqual(report["counts"]["software_only_validated"], 2)
        self.assertEqual(report["unique_operation_counts"]["selected"], 1)
        self.assertEqual(report["unique_operation_counts"]["numerically_covered"], 1)


class SelectionAndPolicyTests(unittest.TestCase):
    def setUp(self):
        self.parser = make_parser()

    def test_exact_plan_replay_does_not_silently_expand(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.jsonl"
            original = Case("gemm", {"dtype": "f64", "m": 8, "n": 8, "k": 8})
            path.write_text(canonical(original.as_dict()) + "\n")
            args = self.parser.parse_args(["list", "--matrix", str(path)])
            self.assertEqual(list(selected(args)), [original])
            args = self.parser.parse_args(["list", "--matrix", str(path), "--software-pairs"])
            self.assertEqual(list(selected(args)), [original, software_counterpart(original)])

    def test_semantic_sharding_keeps_pairs_together(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.jsonl"
            originals = [Case("gemm", {"dtype": "f32", "m": m, "n": 17, "k": 31}) for m in range(1, 40)]
            path.write_text("".join(canonical(case.as_dict()) + "\n" for case in originals))
            seen = set()
            for shard in range(7):
                args = self.parser.parse_args(["list", "--matrix", str(path), "--software-pairs", "--shard", f"{shard}/7"])
                result = list(selected(args)); ids = {case.id for case in result}
                self.assertFalse(seen & ids)
                seen |= ids
                for case in result:
                    pair = software_counterpart(case)
                    if pair: self.assertIn(pair.id, ids)
            self.assertEqual(len(seen), 2 * len(originals))

    def test_stable_defaults_and_explicit_overrides(self):
        base = ["run", "--binary", "worker", "--output", "out"]
        args = resolve_measurement_policy(self.parser.parse_args(base))
        self.assertEqual((args.samples, args.max_samples, args.target_ms, args.target_cv,
                          args.warmup, args.warmup_ms), (11, 31, 50, .03, 10, 200))
        args = resolve_measurement_policy(self.parser.parse_args(base + ["--quality", "quick", "--samples", "40", "--target-cv", "0.02"]))
        self.assertEqual((args.samples, args.max_samples, args.target_cv, args.target_ms), (40, 40, .02, 5))

    def test_gpu_selection_flags_are_mutually_exclusive(self):
        import contextlib, io
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parser.parse_args(["run", "--binary", "worker", "--output", "out", "--device", "1", "--gpu-uuid", "GPU-0123456789abcdef0123456789abcdef"])
