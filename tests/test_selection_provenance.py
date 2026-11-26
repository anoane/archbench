# SPDX-License-Identifier: AGPL-3.0-or-later
"""Selection provenance and acceptance companion regressions; no GPU is used."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from archbench.config import load_cases
from archbench.model import Case
from archbench.runner import read_jsonl, run_suite


ROOT = Path(__file__).resolve().parents[1]


class SelectionProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="archbench_SELECTION_TEST_ONLY_")
        self.root = Path(self.temporary.name)
        self.worker = self.root / "mock_worker.py"
        shutil.copy2(ROOT / "tests/fixtures/mock_worker.py", self.worker)
        self.worker.chmod(0o700)
        self.case = Case("scalar", {"op": "good"})

    def tearDown(self):
        self.temporary.cleanup()

    def run_cases(self, output, cases, **updates):
        values = dict(binary=str(self.worker), output=str(output), resume=False,
                      retry_failed=False, samples=3, warmup=0, target_ms=1,
                      max_repeats=1, memory_fraction=.7, memory_limit_mib=0,
                      device=0, timeout=5, profile="standard", shuffle_seed=17,
                      telemetry=False, telemetry_interval=.2, progress_every=1000,
                      fail_fast=False)
        values.update(updates)
        # Mock output only tests orchestration. It is never performance evidence.
        with patch("archbench.runner.shutil.which", return_value=None), contextlib.redirect_stderr(io.StringIO()):
            return run_suite(cases, argparse.Namespace(**values))

    def test_matrix_and_exact_plan_record_the_effective_source(self):
        for extension, expected_kind in (("json", "json_matrix"), ("jsonl", "jsonl_plan")):
            with self.subTest(extension=extension):
                source = self.root / f"selection.{extension}"
                if extension == "json":
                    source.write_text(json.dumps({"schema_version": 1, "cases": [self.case.as_dict()]}))
                else:
                    source.write_text(json.dumps(self.case.as_dict()) + "\n")
                output = self.root / extension
                self.run_cases(output, list(load_cases(source)), matrix=source,
                               profile="exhaustive", families="scalar")
                metadata = json.loads((output / "run.json").read_text())
                provenance = metadata["selection_source"]
                self.assertEqual(provenance["kind"], expected_kind)
                self.assertEqual(provenance["path"], str(source.resolve()))
                self.assertEqual(provenance["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
                self.assertTrue(provenance["profile_argument_ignored"])
                self.assertEqual(provenance["filters"], {"families": "scalar"})
                self.assertEqual(metadata["selected_cases"], 1)
                # Resume remains governed by selected case identities, not a path.
                self.assertNotIn("selection_source", metadata["identity"])

    def test_generated_profile_records_profile_and_active_filters(self):
        output = self.root / "profile"
        self.run_cases(output, [self.case], profile="smoke", limit=1)
        metadata = json.loads((output / "run.json").read_text())
        self.assertEqual(metadata["selection_source"],
                         {"kind": "profile", "profile": "smoke", "filters": {"limit": 1},
                          "software_pairs": True, "shard_basis": "operation_id"})

    def test_legacy_run_resumes_without_rewriting_or_requiring_new_metadata(self):
        source = self.root / "original_plan.jsonl"
        source.write_text(json.dumps(self.case.as_dict()) + "\n")
        output = self.root / "legacy"
        self.run_cases(output, list(load_cases(source)), matrix=source)
        metadata_path = output / "run.json"
        metadata = json.loads(metadata_path.read_text())
        del metadata["selection_source"]  # The schema of previously saved runs.
        metadata_path.write_text(json.dumps(metadata) + "\n")
        before = metadata_path.read_bytes()
        relocated = self.root / "relocated_plan.jsonl"
        source.rename(relocated)
        result = self.run_cases(output, list(load_cases(relocated)), matrix=relocated, resume=True)
        self.assertEqual(result["recorded"], 1)
        self.assertEqual(result["missing"], 0)
        self.assertEqual(len(list(read_jsonl(output / "results.jsonl"))), 1)
        self.assertEqual(metadata_path.read_bytes(), before)


class TensorEmulationCompanionTests(unittest.TestCase):
    def test_companion_matches_every_native_acceptance_case_exactly(self):
        original = list(load_cases(ROOT / "configs/acceptance.json"))
        self.assertEqual(len(original), 192)
        native = [case for case in original if case.family == "tensor"
                  and case.params.get("backend", "native") in ("native", "ptx")]
        expected = [Case(case.family, {**case.params, "backend": "emulated"},
                         case.tags, case.provider) for case in native]
        companion = list(load_cases(ROOT / "configs/acceptance_tensor_emulation.json"))
        self.assertEqual(len(native), 17)
        self.assertEqual(len(companion), 17)
        self.assertEqual([case.as_dict() for case in companion],
                         [case.as_dict() for case in expected])
        self.assertTrue(all(case.family == "tensor" and case.params["backend"] == "emulated"
                            for case in companion))
        self.assertTrue({case.id for case in companion}.isdisjoint(case.id for case in original))


if __name__ == "__main__":
    unittest.main()
