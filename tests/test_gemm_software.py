# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exercise actual portable GEMM arithmetic/layout and tuning-selection code.

These tests use a real host C++ compiler, not CUDA/API stubs. They do not claim
to compile or execute the CUDA kernels or prove a cuBLAS implementation correct.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
COMPILER = shutil.which("g++") or shutil.which("c++")


@unittest.skipUnless(COMPILER, "a C++17 host compiler is required")
class GemmSoftwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="archbench_software_gemm_")
        cls.addClassCleanup(cls.temporary.cleanup)
        folder = Path(cls.temporary.name)
        source = folder / "software.cpp"
        cls.binary = folder / "software"
        source.write_text(r'''
#include "gemm_software_model.hpp"
#include <array>
#include <complex>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
using namespace archbench::gemm_software;
namespace codec = archbench::codec;
void require(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}
bool same_float(float x,float y) {
    return (std::isnan(x)&&std::isnan(y)) || codec::float_bits(x)==codec::float_bits(y);
}
float independent_fp8(unsigned code,bool e4m3) {
    const int fraction_bits=e4m3?3:2;
    const int exponent_bits=e4m3?4:5;
    const int bias=e4m3?7:15;
    const int exponent=(code>>fraction_bits)&((1<<exponent_bits)-1);
    const int fraction=code&((1<<fraction_bits)-1);
    if((e4m3&&((code&127)==127))||(!e4m3&&exponent==31&&fraction))
        return std::numeric_limits<float>::quiet_NaN();
    float value=!e4m3&&exponent==31?std::numeric_limits<float>::infinity():
        exponent?std::ldexp(1.0f+float(fraction)/(1<<fraction_bits),exponent-bias):
                 std::ldexp(float(fraction)/(1<<fraction_bits),1-bias);
    return code&128?-value:value;
}
void storage_checks() {
    const std::array<float,8> magnitudes{0,.5,1,1.5,2,3,4,6};
    std::array<std::uint8_t,9> packed{};
    for(unsigned i=0;i<17;++i)codec::pack(packed.data(),i,4,(i*7)&15);
    for(unsigned i=0;i<17;++i) {
        unsigned code=(i*7)&15;
        float expected=magnitudes[code&7]*(code&8?-1:1);
        require(same_float(load_component<float>(packed.data(),storage_kind::e2m1,i),expected),
                "FP4 sign or high/low nibble ordering is wrong");
    }
    // The odd last logical value must not accidentally depend on a tail nibble.
    packed.back()|=0xf0;
    require(load_component<float>(packed.data(),storage_kind::e2m1,16)==0,"FP4 odd tail read uses wrong nibble");
    std::array<std::uint8_t,256> fp8{};
    for(unsigned i=0;i<256;++i)fp8[i]=i;
    for(unsigned i=0;i<256;++i) {
        require(same_float(load_component<float>(fp8.data(),storage_kind::e4m3,i),independent_fp8(i,true)),"E4M3 decode mismatch");
        require(same_float(load_component<float>(fp8.data(),storage_kind::e5m2,i),independent_fp8(i,false)),"E5M2 decode mismatch");
    }
    // Every binary16 bit pattern is compared with an independent host-native
    // IEEE half conversion, including signed zero, subnormals and specials.
    for(unsigned i=0;i<65536;++i) {
        const std::uint16_t bits=i;
        _Float16 native;
        std::memcpy(&native,&bits,sizeof(bits));
        const std::array<std::uint8_t,2> bytes{std::uint8_t(i),std::uint8_t(i>>8)};
        require(same_float(load_component<float>(bytes.data(),storage_kind::f16,0),float(native)),"binary16 storage decode mismatch");
        require(same_float(load_component<float>(bytes.data(),storage_kind::bf16,0),codec::from_bits(i<<16)),"bfloat16 storage decode mismatch");
    }
    const std::array<double,4> z{1.25,-3.5,-7.75,9.5};
    require(load_component<double>(z.data(),storage_kind::c128,1)==-7.75,"complex real offset mismatch");
    require(load_component<double>(z.data(),storage_kind::c128,1,true)==9.5,"complex imaginary offset mismatch");
}
void rounding_checks() {
    for(unsigned bits=0;bits<0x7bff;++bits) {
        std::uint16_t a_bits=bits,b_bits=bits+1;
        _Float16 a,b;
        std::memcpy(&a,&a_bits,2);std::memcpy(&b,&b_bits,2);
        const float midpoint=(float(a)+float(b))*.5f;
        for(float x:{std::nextafter(midpoint,-INFINITY),midpoint,std::nextafter(midpoint,INFINITY),-midpoint}) {
            const _Float16 native=static_cast<_Float16>(x);
            std::uint16_t expected;std::memcpy(&expected,&native,2);
            require(half_bits_rne(x)==expected,"half RNE midpoint/subnormal/sign conversion mismatch");
        }
    }
    require(half_bits_rne(65504)==0x7bff,"half maximum finite corrupted");
    require(half_bits_rne(65520)==0x7c00,"half overflow must produce infinity, not finite saturation");
    require(half_bits_rne(INFINITY)==0x7c00&&half_bits_rne(-INFINITY)==0xfc00,"half infinity corrupted");
    require((half_bits_rne(NAN)&0x7fff)>0x7c00,"half NaN changed to infinity");
    for(unsigned removed:{13u,16u}) {
        for(unsigned high:{0u,1u,0x100u,0x3f80u,0x7f7fu}) {
            const unsigned even=(high<<(removed+1));
            if((even&0x7f800000u)==0x7f800000u)continue;
            const unsigned half=1u<<(removed-1);
            for(unsigned sign:{0u,0x80000000u}) {
                require(codec::float_bits(round_mantissa_rne(codec::from_bits(sign|even|half),removed))==(sign|even),"RNE even tie rounded upward");
                const unsigned odd=even|(1u<<removed);
                if((odd&0x7f800000u)!=0x7f800000u)
                    require(codec::float_bits(round_mantissa_rne(codec::from_bits(sign|odd|half),removed))==((sign|odd)+(1u<<removed)),"RNE odd tie did not round upward");
            }
        }
    }
    require(std::isnan(round_input(NAN,input_rounding::bf16)),"BF16 NaN lost");
    require(round_input(1.0006f,input_rounding::tf32)==1.0009765625f,"TF32 precision is not ten fraction bits");
    require(round_input(1.004f,input_rounding::bf16)==1.0078125f,"BF16 precision is not seven fraction bits");
}
void layout_checks() {
    // Ragged 3x7 by 7x5 products, three batches, padding, and an odd packed
    // stride exercise physical nibble addressing across every transpose pair.
    constexpr int m=3,n=5,k=7,batches=3;
    for(bool ta:{false,true})for(bool tb:{false,true}) {
        const int lda=(ta?k:m)+2,ldb=(tb?n:k)+2;
        const std::size_t sa=std::size_t(lda)*(ta?m:k)+6;
        const std::size_t sb=std::size_t(ldb)*(tb?k:n)+8;
        std::vector<std::uint8_t> a((sa*batches+1)/2),b((sb*batches+1)/2);
        auto code_a=[](int row,int q,int batch){return unsigned(row*3+q*5+batch*7)&15u;};
        auto code_b=[](int q,int col,int batch){return unsigned(q*7+col*3+batch*5)&15u;};
        for(int batch=0;batch<batches;++batch) {
            for(int row=0;row<m;++row)for(int q=0;q<k;++q) {
                const std::size_t idx=batch*sa+(ta?std::size_t(row)*lda+q:std::size_t(q)*lda+row);
                codec::pack(a.data(),idx,4,code_a(row,q,batch));
            }
            for(int col=0;col<n;++col)for(int q=0;q<k;++q) {
                const std::size_t idx=batch*sb+(tb?std::size_t(q)*ldb+col:std::size_t(col)*ldb+q);
                codec::pack(b.data(),idx,4,code_b(q,col,batch));
            }
        }
        const std::array<double,8> magnitudes{0,.5,1,1.5,2,3,4,6};
        auto decode=[&](unsigned code){return magnitudes[code&7]*(code&8?-1:1);};
        for(int batch=0;batch<batches;++batch)for(int col=0;col<n;++col)for(int row=0;row<m;++row) {
            double actual=0,expected=0;
            for(int q=0;q<k;++q) {
                actual+=load_component<double>(a.data(),storage_kind::e2m1,matrix_index(row,q,batch,lda,sa,ta))*
                        load_component<double>(b.data(),storage_kind::e2m1,matrix_index(q,col,batch,ldb,sb,tb));
                expected+=decode(code_a(row,q,batch))*decode(code_b(q,col,batch));
            }
            require(actual==expected,"packed mixed-transpose/batch GEMM addressing mismatch");
        }
    }
    for(int vector:{16,32}) {
        constexpr int outer=259,reduction=131;
        const std::size_t count=((outer+127)/128)*((reduction+vector-1)/vector+3)/4*512;
        std::set<std::size_t> occupied;
        for(int row=0;row<outer;++row)for(int q=0;q<reduction;q+=vector) {
            const auto offset=scale_offset(row,q,reduction,vector);
            require(offset<count,"block scale offset exceeds allocated tiled storage");
            require(occupied.insert(offset).second,"distinct block scales alias in tiled layout");
            require(scale_offset(row,std::min(q+vector-1,reduction-1),reduction,vector)==offset,"scale changes within one block");
        }
    }
    require(scale_offset(32,0,128,16)==4,"scale row interleave wrong");
    require(scale_offset(127,48,128,16)==511,"scale tile end wrong");
    require(scale_offset(0,64,128,16)==512,"scale K tile crossing wrong");
    require(scale_offset(128,0,128,16)==1024,"scale row tile crossing wrong");
    std::vector<std::uint8_t> s(2048,127);
    s[scale_offset(127,48,128,32)]=126;
    require(load_scale(s.data(),127,48,128,32)==.5f,"UE8M0 scale loaded from wrong physical location");
    s.assign(2048,56);s[scale_offset(32,17,128,16)]=64;
    require(load_scale(s.data(),32,17,128,16)==2.0f,"UE4M3 scale loaded from wrong physical location");
}
void candidate_checks() {
    const auto nan=std::numeric_limits<double>::quiet_NaN();
    const auto inf=std::numeric_limits<double>::infinity();
    std::vector<candidate_score> candidates{
        {0,false,{.01,.01,.01}}, // fast but numerically wrong
        {1,true,{2,2,900}},     // one timing outlier must not dominate selection
        {2,true,{3,3,3}},
        {3,true,{nan}}, {4,true,{0}}, {5,true,{inf}}};
    require(fastest_validated(candidates)==1,"selector accepted invalid numerics/timing or used the mean");
    require(candidate_median({9,true,{8,2,4,6}})==5,"even-count median incorrect");
    require(fastest_validated({{9,true,{2}},{3,true,{2}}})==3,"equal medians do not select a stable lower index");
    require(fastest_validated({{0,false,{.01}},{1,true,{nan}},{2,true,{}}})==-1,"all rejected candidates produced a selected result");
}
void rank1_checks() {
    for(int k:{3,1024,4096,8192,65536})for(int row=0;row<13;++row)for(int col=0;col<11;++col) {
        float sum=0;
        for(int q=0;q<k;++q)sum=std::fma(rank1_input(true,row,q),rank1_input(false,q,col),sum);
        const double expected=double(k)*(row%13-6)*(col%11-5)/64.0;
        require(sum==expected&&rank1_expected(row,col,k)==expected,"rank1 analytic result differs from full FP32 dot product");
    }
}
int main(int argc,char** argv) {
    require(argc==2,"scenario required");
    const std::string scenario=argv[1];
    if(scenario=="storage")storage_checks();
    else if(scenario=="rounding")rounding_checks();
    else if(scenario=="layout")layout_checks();
    else if(scenario=="candidates")candidate_checks();
    else if(scenario=="rank1")rank1_checks();
    else throw std::runtime_error("unknown scenario");
    std::cout<<"passed\n";
}
''')
        subprocess.run(
            [COMPILER, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
             "-I", str(ROOT / "include"), "-I", str(ROOT / "src"),
             str(source), "-o", str(cls.binary)],
            check=True, capture_output=True, text=True,
        )

    def check_scenario(self, scenario):
        result = subprocess.run([str(self.binary), scenario], check=True,
                                capture_output=True, text=True)
        self.assertEqual(result.stdout, "passed\n")

    def test_actual_storage_decoders_and_packed_nibbles(self):
        self.check_scenario("storage")

    def test_rounding_against_native_host_ieee_half_and_ties(self):
        self.check_scenario("rounding")

    def test_ragged_transposed_batched_packed_and_block_scale_addresses(self):
        self.check_scenario("layout")

    def test_fastest_selection_requires_validation_and_finite_median(self):
        self.check_scenario("candidates")

    def test_rank1_analytic_reference_matches_full_float_reduction(self):
        self.check_scenario("rank1")


if __name__ == "__main__":
    unittest.main()
