# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exercise actual ALU host validation helpers without CUDA API/type stubs."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = shutil.which("g++") or shutil.which("c++")


def function(source, signature):
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@unittest.skipUnless(COMPILER, "host C++17 compiler required")
class AluValidationTests(unittest.TestCase):
    def test_sampling_all_chains_and_nonfinite_or_exact_integer_diagnostics(self):
        source = (ROOT / "src/alu.cu").read_text()
        helpers = [
            function(source, "std::vector<size_t> micro_sample_indices("),
            function(source, "void validate_number("),
            function(source, "template<class U>\nvoid validate_integer_bits("),
            function(source, "template<class T>\nvoid validate_all_finite("),
        ]
        fixture = r'''
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
// A minimal result-data fixture, not a CUDA runtime or device emulation.
struct result {
    std::string status="ok", validation="passed", reason;
    double max_abs_error=0, max_rel_error=0;
    std::map<std::string,std::string> metadata;
    std::map<std::string,double> metrics;
};
template<class T> double as_double(T value) { return static_cast<double>(value); }
void require(bool ok) { if (!ok) throw std::runtime_error("ALU validation regression"); }
'''
        checks = r'''
int main() {
    int shapes = 0;
    for (size_t lanes : {size_t(1),size_t(2),size_t(31),size_t(32),size_t(64),size_t(256),
                        size_t(1024),size_t(135168),size_t(270336),size_t(65535)*1024}) {
        for (size_t width : {size_t(1),size_t(2),size_t(4),size_t(8),size_t(16)}) {
            const auto indices=micro_sample_indices(lanes,width);
            require(!indices.empty() && indices.size()<=256);
            require(std::is_sorted(indices.begin(),indices.end()));
            require(std::adjacent_find(indices.begin(),indices.end())==indices.end());
            std::map<size_t,std::vector<bool>> covered;
            for (size_t index : indices) {
                require(index<lanes*width);
                auto& values=covered[index/width];
                if(values.empty())values.resize(width,false);
                values[index%width]=true;
            }
            require(covered.begin()->first==0);
            if(covered.size()>1)require(covered.rbegin()->first==lanes-1);
            for(const auto& lane:covered)
                require(std::all_of(lane.second.begin(),lane.second.end(),[](bool present){return present;}));
            if(lanes*width<=256)require(indices.size()==lanes*width);
            ++shapes;
        }
    }
    // The old stride for this allocation visited only chain zero.
    const auto indices=micro_sample_indices(1024,8);
    for(size_t chain=0;chain<8;++chain)
        require(std::any_of(indices.begin(),indices.end(),[&](size_t q){return q%8==chain;}));
    for(auto bad:std::vector<std::pair<size_t,size_t>>{{0,1},{1,0},{1,257}}) {
        bool rejected=false;
        try{(void)micro_sample_indices(bad.first,bad.second);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected);
    }
    result finite;
    validate_number(finite,1.000001,1,2e-6,0);
    require(finite.status=="ok"&&finite.max_abs_error>0);
    validate_number(finite,1.1,1,2e-6,0);
    require(finite.status=="validation_failed"&&finite.max_abs_error>.09);
    const double infinity=std::numeric_limits<double>::infinity(),nan=std::numeric_limits<double>::quiet_NaN();
    for(auto values:std::vector<std::pair<double,double>>{{nan,0},{infinity,1},{-infinity,1},{infinity,infinity},{1,nan}}) {
        result nonfinite;
        validate_number(nonfinite,values.first,values.second,0,0);
        require(nonfinite.status=="validation_failed");
        require(std::isinf(nonfinite.max_abs_error)&&std::isinf(nonfinite.max_rel_error));
    }
    result scan;
    std::vector<double> outputs(8192,1);
    outputs[1]=nan;outputs[3]=infinity;outputs[5]=-infinity;
    validate_all_finite(scan,outputs);
    require(scan.status=="validation_failed"&&scan.metrics["nonfinite_output_values"]==3);
    require(scan.metadata["all_downloaded_outputs_finite"]=="false");
    result clean;validate_all_finite(clean,std::vector<double>{0,1,-1});
    require(clean.status=="ok"&&clean.max_abs_error==0&&clean.metrics["nonfinite_output_values"]==0);
    result exact;validate_integer_bits(exact,uint64_t(0),uint64_t(0),"bits");
    require(exact.status=="ok"&&exact.max_abs_error==0);
    result nearby64;validate_integer_bits(nearby64,UINT64_MAX-1,UINT64_MAX,"bits");
    require(nearby64.status=="validation_failed"&&nearby64.max_abs_error==1);
    require(nearby64.max_rel_error>0&&nearby64.metrics["bit_pattern_mismatches"]==1);
    using U128=unsigned __int128;
    const U128 base=(U128(1)<<127)+42;
    result nearby128;validate_integer_bits(nearby128,base-1,base,"bits");
    require(nearby128.status=="validation_failed"&&nearby128.max_abs_error==1&&nearby128.max_rel_error>0);
    result narrow;validate_integer_bits(narrow,uint8_t(0),uint8_t(255),"bits");
    require(narrow.max_abs_error==255&&narrow.max_rel_error==1);
    std::cout<<"passed: "<<shapes<<" ILP/packed sampling shapes; finite/nonfinite and u8/u64/u128 diagnostics\n";
}
'''
        with tempfile.TemporaryDirectory(prefix="archbench_alu_validation_HOST_ONLY_") as directory:
            cpp, binary = Path(directory) / "check.cpp", Path(directory) / "check"
            cpp.write_text(fixture + "\n".join(helpers) + checks)
            subprocess.run([COMPILER, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                            str(cpp), "-o", str(binary)], check=True, capture_output=True, text=True)
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
            self.assertIn("passed: 50 ILP/packed sampling shapes", result.stdout)


if __name__ == "__main__":
    unittest.main()
