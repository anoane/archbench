# SPDX-License-Identifier: AGPL-3.0-or-later
from pathlib import Path
import tempfile
import unittest
from archbench.adapters import parse_cutlass_csv
from archbench.statistics import describe


class AdapterTests(unittest.TestCase):
    def test_cutlass_only_passed_results_are_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "provider.csv"
            path.write_text('Operation,Status,Disposition,Runtime\nreal_kernel,Success,Passed,1.5\nunchecked,Success,NotVerified,0.1\nfailed,Success,Incorrect,0.2\nskipped,NotSupported,NotRun,0\n')
            rows = parse_cutlass_csv(path)
            self.assertEqual([x["validated_measured"] for x in rows], [True, False, False, False])
            self.assertEqual(len(rows), 4)

    def test_statistics_even_median(self):
        result = describe([1., 2., 3., 4.], bootstrap=0)
        self.assertEqual(result["median_ms"], 2.5)
        self.assertEqual(result["mad_ms"], 1.0)

    def test_statistics_rejects_invalid(self):
        self.assertEqual(describe([]), {})
        self.assertEqual(describe([0]), {})
        self.assertEqual(describe([float("nan")]), {})

    def test_bootstrap_is_reproducible(self):
        self.assertEqual(describe([1, 2, 3, 4]), describe([1, 2, 3, 4]))
