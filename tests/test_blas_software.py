# SPDX-License-Identifier: AGPL-3.0-or-later
"""Host algebra properties of the exact portable software BLAS implementation."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = shutil.which("g++") or shutil.which("c++")

@unittest.skipUnless(COMPILER, "C++17 host compiler required")
class SoftwareBlasAlgebra(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="archbench_blas_algebra_")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.binary = Path(cls.temp.name) / "blas_host"
        subprocess.run([COMPILER, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-I", str(ROOT / "include"), str(ROOT / "tests/blas_software_host.cpp"),
                        "-o", str(cls.binary)], check=True, capture_output=True, text=True)

    def scenario(self, name):
        result = subprocess.run([str(self.binary), name], check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout, f"PASS {name}\n")

    def test_triangular_residual_both_sides_triangles_transposes_and_diagonals(self):
        self.scenario("solve")

    def test_symmetric_hermitian_and_triangular_products_match_dense_algebra(self):
        self.scenario("structure")

    def test_vector_and_matrix_transposes_conjugation_padding_and_strides(self):
        self.scenario("general")
