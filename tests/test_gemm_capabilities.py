# SPDX-License-Identifier: AGPL-3.0-or-later
"""Host checks of the documented GEMM capability policy, without CUDA stubs.

These checks compile and execute the policy header used by the GEMM dispatcher.
They do not compile CUDA, verify library APIs, or supply GPU performance evidence.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
COMPILER = shutil.which("g++") or shutil.which("c++")


@unittest.skipUnless(COMPILER, "a C++17 host compiler is required for capability policy checks")
class GemmCapabilityPolicyTests(unittest.TestCase):
    def test_bf16x9_admits_only_cuda_13_5_documented_capabilities(self):
        # Include the real predicate, and compare its full output against the
        # documented finite set. Neighboring and future major/minor values must
        # not silently gain support through a broad >= architecture test.
        with tempfile.TemporaryDirectory(prefix="archbench_gemm_policy_") as temporary:
            directory = Path(temporary)
            source = directory / "capabilities.cpp"
            binary = directory / "capabilities"
            source.write_text(
                '#include "src/gemm_capabilities.hpp"\n'
                '#include <iostream>\n'
                'int main() {\n'
                '  for (int major = -1; major <= 15; ++major)\n'
                '    for (int minor = -1; minor <= 15; ++minor)\n'
                '      if (archbench::gemm_detail::cublas_bf16x9_supported(major, minor))\n'
                '        std::cout << major << "." << minor << "\\n";\n'
                '}\n'
            )
            subprocess.run(
                [COMPILER, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT),
                 str(source), "-o", str(binary)],
                check=True, capture_output=True, text=True,
            )
            completed = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
        self.assertEqual(completed.stdout.splitlines(), ["10.0", "10.3"])


if __name__ == "__main__":
    unittest.main()
