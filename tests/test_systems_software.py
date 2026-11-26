# SPDX-License-Identifier: AGPL-3.0-or-later
"""Host arithmetic/layout checks; no CUDA stubs or simulated GPU performance."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = shutil.which("g++") or shutil.which("c++")


@unittest.skipUnless(COMPILER, "host C++17 compiler required")
class SystemsSoftwareTests(unittest.TestCase):
    def test_fft_helpers_against_independent_dft_and_native_input_contract(self):
        source = (ROOT / "src/systems.cu").read_text()
        match = re.search(r"float input_value\(std::size_t i, unsigned salt = 0\) \{.*?\n\}", source, re.S)
        self.assertIsNotNone(match, "native stored-input generator must be checked, not copied silently")
        native_function = match.group(0).replace("input_value(", "native_library_input(", 1)
        with tempfile.TemporaryDirectory(prefix="archbench_systems_HOST_ONLY_") as directory:
            native, binary = Path(directory) / "native_input.cpp", Path(directory) / "check"
            native.write_text("#include <cstddef>\n#include <cstdint>\n" + native_function + "\n")
            subprocess.run([COMPILER, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT / "include"), str(ROOT / "tests/systems_software_host.cpp"),
                            str(native), "-o", str(binary)], check=True, capture_output=True, text=True)
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
            self.assertIn("passed: 96 transforms;", result.stdout)
            self.assertIn("host execution only", result.stdout)


if __name__ == "__main__":
    unittest.main()
