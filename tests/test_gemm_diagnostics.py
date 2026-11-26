# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compile and exercise the actual host-only GEMM failure-diagnostic block.

No CUDA headers, GPU, or API stubs are used. This tests evidence selection and
bounded serialization; it does not compile the CUDA translation unit or validate
cuBLAS algorithms, epilogue behavior, or the numerical tolerance itself.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
COMPILER = shutil.which("g++") or shutil.which("c++")


@unittest.skipUnless(COMPILER, "a C++17 host compiler is required for diagnostic checks")
class GemmFailureDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (ROOT / "src/gemm.cu").read_text()
        begin = "// ARCHBENCH_HOST_GEMM_DIAGNOSTICS_BEGIN"
        end = "// ARCHBENCH_HOST_GEMM_DIAGNOSTICS_END"
        block = source.split(begin, 1)[1].split(end, 1)[0]
        cls.temporary = tempfile.TemporaryDirectory(prefix="archbench_gemm_diagnostics_")
        cls.addClassCleanup(cls.temporary.cleanup)
        directory = Path(cls.temporary.name)
        cls.binary = directory / "diagnostics"
        probe = directory / "diagnostics.cpp"
        probe.write_text(r'''
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
using wide = std::complex<long double>;
''' + block + r'''
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
validation_failure_sample sample(uint64_t index, long double observed,
                                 long double expected, long double bound) {
    return {index,3,5,7,{observed,0},{expected,0},{2,0},9,bound};
}
int main(int argc, char** argv) {
    require(argc == 2, "scenario required");
    validation_failure_details details;
    std::map<std::string,std::string> metadata;
    std::map<std::string,double> metrics;
    const std::string scenario = argv[1];
    if (scenario == "bounded") {
        // A large absolute error can still pass its own condition-aware bound.
        // It must not replace evidence from a smaller failed output.
        details.consider(true,sample(1,1001,1,2000));
        details.consider(false,sample(11,1.02L,1,.01L));
        details.consider(false,sample(22,101,1,90));
        details.consider(false,sample(33,1.003L,1,.0001L));
        for (int i = 0; i < 20000; ++i)
            details.consider(false,sample(100+i,101,1,90));
        require(details.failed_samples == 20003, "failed count lost samples");
        require(details.first.logical_index == 11, "first failure overwritten");
        require(details.worst.logical_index == 33, "ranked by absolute error instead of bound violation");
        details.record(metadata,metrics);
        require(metadata.size() == 31, "diagnostic output grows with number of failures");
        require(metrics.at("validation_failed_samples") == 20003, "recorded count differs");
        require(metadata.at("validation_first_failure.logical_index") == "11", "wrong first index");
        require(metadata.at("validation_worst_failure.logical_index") == "33", "wrong worst index");
        require(metadata.at("validation_worst_failure.row") == "3", "wrong row");
        require(metadata.at("validation_worst_failure.column") == "5", "wrong column");
        require(metadata.at("validation_worst_failure.batch") == "7", "wrong batch");
        require(std::stold(metadata.at("validation_worst_failure.preactivation_real")) == 2,
                "preactivation evidence missing");
        require(std::stold(metadata.at("validation_worst_failure.bound_sum_abs")) == 9,
                "bound input evidence missing");
    } else if (scenario == "nonfinite") {
        details.consider(false,sample(11,1.02L,1,.01L));
        details.consider(false,sample(22,std::numeric_limits<long double>::quiet_NaN(),1,.01L));
        details.consider(false,sample(33,std::numeric_limits<long double>::infinity(),1,.01L));
        require(details.worst.logical_index == 22, "nonfinite value not ranked first or tie not stable");
        require(std::isinf(details.worst.error_over_bound()), "nonfinite violation is not unbounded");
        details.record(metadata,metrics);
        require(metadata.at("validation_worst_failure.observed_real").find("nan") != std::string::npos,
                "NaN evidence was lost");
        require(metadata.at("validation_worst_failure.error_over_bound").find("inf") != std::string::npos,
                "unbounded ratio was lost");
        require(std::isinf(sample(44,2,1,0).error_over_bound()), "zero bound divides incorrectly");
    } else if (scenario == "passed") {
        details.consider(true,sample(1,1,1,0));
        details.record(metadata,metrics);
        require(details.failed_samples == 0, "passing value counted as failure");
        require(metadata.empty(), "passing case invents failure evidence");
        require(metrics.at("validation_failed_samples") == 0, "passing count not recorded");
    } else throw std::runtime_error("unknown scenario");
    std::cout << "passed\n";
}
''')
        subprocess.run(
            [COMPILER, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
             str(probe), "-o", str(cls.binary)],
            check=True, capture_output=True, text=True,
        )

    def check_scenario(self, scenario):
        completed = subprocess.run([str(self.binary), scenario], check=True,
                                   capture_output=True, text=True)
        self.assertEqual(completed.stdout, "passed\n")

    def test_first_and_worst_failed_samples_are_bounded_and_distinct(self):
        self.check_scenario("bounded")

    def test_nonfinite_and_zero_bound_evidence_is_preserved(self):
        self.check_scenario("nonfinite")

    def test_passing_case_has_no_failure_evidence(self):
        self.check_scenario("passed")


if __name__ == "__main__":
    unittest.main()
