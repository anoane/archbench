# SPDX-License-Identifier: AGPL-3.0-or-later
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = shutil.which("g++") or shutil.which("c++")

@unittest.skipUnless(COMPILER, "C++17 host compiler required")
class QuantGemmSoftwareTests(unittest.TestCase):
    def test_ragged_decoded_products_exclude_poisoned_group_padding(self):
        with tempfile.TemporaryDirectory(prefix="archbench_quant_host_") as directory:
            binary = Path(directory) / "quant_host"
            subprocess.run([COMPILER, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT / "include"), str(ROOT / "tests/quant_gemm_host.cpp"),
                            "-o", str(binary)], check=True, capture_output=True, text=True)
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
            self.assertEqual(result.stdout, "PASS 245 decoded products; poisoned padding and active NaN\n")
