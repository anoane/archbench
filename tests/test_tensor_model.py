# SPDX-License-Identifier: AGPL-3.0-or-later
"""Host checks for work counting/segments, not CUDA compilation or GPU evidence."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from archbench.config import load_cases

ROOT = Path(__file__).resolve().parents[1]
COMPILER = shutil.which("g++") or shutil.which("c++")


class TensorThroughputPlanTests(unittest.TestCase):
    def test_plan_pairs_backends_and_bounds_half_accumulators(self):
        cases = list(load_cases(ROOT / "configs/tensor_throughput.json"))
        self.assertEqual(len(cases), 140)
        sides = {backend: set() for backend in ("native", "emulated")}
        for case in cases:
            params = dict(case.params)
            backend = params.pop("backend")
            sides[backend].add(tuple(sorted(params.items())))
            self.assertEqual(params["mode"], "throughput")
            self.assertEqual(params["ilp"], 8)
            self.assertEqual(params["iterations"], 20000)
            if params["accumulator"] == "f16":
                self.assertEqual(params["reset-every"], 512)
            self.assertNotEqual("blocks" in params, "blocks-per-sm" in params)
        self.assertEqual(sides["native"], sides["emulated"])
        linked = [case.params for case in cases if "linked_fp8_launch_controls" in case.tags]
        self.assertEqual(len(linked), 2)
        for params in linked:
            self.assertEqual((params["blocks"], params["threads"], params["dtype"], params["accumulator"]),
                             (1056, 256, "e4m3", "f32"))


@unittest.skipUnless(COMPILER, "host C++17 compiler required")
class TensorAccountingTests(unittest.TestCase):
    def test_actual_accounting_header_matches_independent_work_and_segment_models(self):
        source = r'''
#include "archbench/tensor_model.hpp"
#include <iostream>
#include <stdexcept>
using namespace archbench::tensor_model;
void require(bool ok) { if (!ok) throw std::runtime_error("tensor accounting mismatch"); }
int main() {
    const auto linked = count_work(1056ULL * 256, 8, 20000, 16, 8, 32, false);
    require(linked.warps == 8448 && linked.logical_tiles == 1351680000ULL);
    require(linked.operations_per_tile == 8192 && linked.operations == 11072962560000ULL);
    require(count_work(32, 1, 1, 8, 8, 4, false).operations == 512);
    require(count_work(32, 1, 1, 16, 8, 256, true).operations == 32768);
    require(count_work(32, 1, 1, 16, 8, 256, false).operations == 65536);
    for (int iterations : {1, 2, 3, 31, 512, 513, 1024, 4099, 20000}) {
        for (int reset : {0, 1, 3, 512, 1024, 65536}) {
            int accumulated = 0, longest = 0, resets = 0;
            for (int step = 0; step < iterations; ++step) {
                if (reset && accumulated == reset) { accumulated = 0; ++resets; }
                ++accumulated;
                if (accumulated > longest) longest = accumulated;
            }
            require(final_segment_iterations(iterations, reset) == accumulated);
            require(largest_segment_iterations(iterations, reset) == longest);
            require(reset_events(iterations, reset) == static_cast<uint64_t>(resets));
            for (int ilp : {1, 2, 4, 8}) {
                const auto count = count_work(128, ilp, iterations, 16, 8, 32, false);
                const uint64_t independent = uint64_t(4) * ilp * iterations * 8192;
                require(count.operations == independent);
            }
        }
    }
    bool overflow = false, invalid = false;
    try { (void) count_work(UINT64_MAX & ~uint64_t(31), 8, 1000000, 16, 8, 256, false); }
    catch (const std::overflow_error&) { overflow = true; }
    try { (void) count_work(33, 8, 1, 16, 8, 32, false); }
    catch (const std::invalid_argument&) { invalid = true; }
    require(overflow && invalid);
    require(reset_interval(0) == 0 && reset_interval(1000000) == 1000000);
    for (int64_t bad : {int64_t(-1), int64_t(1000001), INT64_MAX}) {
        bool rejected = false;
        try { (void) reset_interval(bad); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected);
    }
    std::cout << "passed: upstream work, units, 54 segment schedules, overflow and bounds\n";
}
'''
        with tempfile.TemporaryDirectory(prefix="archbench_tensor_HOST_ONLY_") as directory:
            cpp, binary = Path(directory) / "check.cpp", Path(directory) / "check"
            cpp.write_text(source)
            subprocess.run([COMPILER, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT / "include"), str(cpp), "-o", str(binary)],
                           check=True, capture_output=True, text=True)
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
            self.assertIn("passed: upstream work", result.stdout)


if __name__ == "__main__":
    unittest.main()
