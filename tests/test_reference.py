# SPDX-License-Identifier: AGPL-3.0-or-later
import math
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest
from archbench import reference as ref

ROOT = Path(__file__).resolve().parents[1]


class NumericReferenceTests(unittest.TestCase):
    def test_known_float_values(self):
        cases = (("e2m1", 7, 6), ("e2m3", 31, 7.5), ("e3m2", 31, 28),
                 ("e4m3", 126, 448), ("e5m2", 123, 57344), ("f16", 0x7bff, 65504),
                 ("bf16", 0x3f80, 1))
        for fmt, code, expected in cases:
            with self.subTest(fmt=fmt): self.assertEqual(ref.decode(code, fmt), expected)

    def test_ieee_special_values(self):
        self.assertTrue(math.isinf(ref.decode(0x7c, "e5m2")))
        self.assertTrue(math.isnan(ref.decode(0x7f, "e4m3")))
        self.assertTrue(math.isnan(ref.decode(0x7c01, "f16")))
        self.assertEqual(math.copysign(1, ref.decode(8, "e2m1")), -1)

    def test_ties_to_even_and_saturation(self):
        self.assertEqual(ref.encode(1.25, "e2m1"), 2)
        self.assertEqual(ref.encode(1.75, "e2m1"), 4)
        self.assertEqual(ref.encode(math.inf, "e4m3"), 126)
        self.assertEqual(ref.encode(-math.inf, "e4m3"), 254)
        self.assertEqual(ref.encode(math.inf, "nf4"), 15)
        self.assertEqual(ref.encode(-math.inf, "nf4"), 0)

    def test_nf4_is_codebook_not_e2m1(self):
        self.assertEqual(ref.decode(7, "nf4"), 0)
        self.assertEqual(ref.decode(7, "e2m1"), 6)
        self.assertNotEqual(ref.encode(.5, "nf4"), ref.encode(.5, "e2m1"))

    def test_roundtrip_all_small_float_codes(self):
        for fmt in ("e2m1", "e2m3", "e3m2", "e4m3", "e5m2", "nf4"):
            for code in range(2 ** ref.width(fmt)):
                value = ref.decode(code, fmt)
                if math.isfinite(value): self.assertEqual(ref.encode(value, fmt), code)

    def test_pack_every_width_and_ragged_tail(self):
        rng = random.Random(777)
        for bits in range(1, 65):
            codes = [rng.randrange(2 ** bits) for _ in range(131)]
            payload = ref.pack(codes, bits)
            self.assertEqual(ref.unpack(payload, len(codes), bits), codes)
            self.assertEqual(len(payload), (len(codes) * bits + 7) // 8)

    def test_pack_rejects_bad_inputs(self):
        for bits in (0, 65):
            with self.assertRaises(ValueError): ref.pack([0], bits)
        with self.assertRaises(ValueError): ref.pack([-1], 4)
        with self.assertRaises(ValueError): ref.pack([16], 4)
        with self.assertRaises(ValueError): ref.unpack(b"", 1, 4)

    def test_signed_one_bit_is_twos_complement(self):
        self.assertEqual([ref.decode(x, "s1") for x in (0, 1)], [0, -1])
        self.assertEqual(ref.encode(1, "s1"), 0)

    def test_modular_signed_overflow(self):
        self.assertEqual(ref.modular_integer(127, 1, 8, True, "add"), 128)
        self.assertEqual(ref.modular_integer(128, 7, 8, True, "shr"), 255)
        self.assertEqual(ref.modular_integer(255, 0, 8, True, "min"), 255)
        self.assertEqual(ref.modular_integer(255, 0, 8, False, "min"), 0)
        self.assertEqual(ref.modular_integer(2 ** 64 - 1, 2, 64, False, "mul"), 2 ** 64 - 2)

    def test_block_quantization_zero_and_constant(self):
        for fmt in ("s1", "s4", "u4", "nf4", "e2m1", "e4m3", "f16", "bf16"):
            zero = ref.quantize([0.] * 35, fmt, 32)
            self.assertEqual(zero["decoded"], [0.] * 35)
            result = ref.quantize([.375] * 35, fmt, 16)
            self.assertTrue(all(math.isfinite(x) for x in result["decoded"]))
            if fmt == "u4": self.assertEqual(result["decoded"], [.375] * 35)

    def test_scale_encoding(self):
        for x, code in ((1, 127), (.5, 126), (1.1, 128), (2, 128), (0, 0)):
            self.assertEqual(ref.scale_e8m0(x), code)

    def test_reference_rejects_nonfinite_quant_group(self):
        with self.assertRaises(ValueError): ref.quantize([math.nan], "nf4")


@unittest.skipUnless(shutil.which("g++"), "g++ required for C++/Python independent codec comparison")
class CppCodecCrossChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="archbench_codec_")
        cls.binary = Path(cls.temp.name) / "codec_host"
        subprocess.run([shutil.which("g++"), "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-I", str(ROOT / "include"), str(ROOT / "tests/codec_host.cpp"), "-o", str(cls.binary)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_cpp_exhaustive_roundtrip_and_pack(self):
        result = subprocess.run([str(self.binary)], capture_output=True, text=True, check=True)
        self.assertIn("393666 passed", result.stdout)

    def test_independent_decode_every_binary_float_code(self):
        checks = 0
        for fmt in (*ref.FLOATS, "nf4"):
            result = subprocess.run([str(self.binary), "decode", fmt], capture_output=True, text=True, check=True)
            for line in result.stdout.splitlines():
                code, text = line.split()
                actual = float.fromhex(text)
                expected = ref.f32(ref.decode(int(code), fmt))
                if math.isnan(expected): self.assertTrue(math.isnan(actual))
                else:
                    self.assertEqual(actual, expected, (fmt, code))
                    if expected == 0: self.assertEqual(math.copysign(1, actual), math.copysign(1, expected))
                checks += 1
        self.assertGreater(checks, 131000)

    def test_independent_encode_random_values_and_midpoints(self):
        rng = random.Random(9207)
        for fmt in (*ref.FLOATS, "nf4", "s1", "s4", "u4", "s16", "u16"):
            values = [ref.f32(rng.uniform(-2000, 2000)) for _ in range(200)] + [0., -0., 1.25, 1.75, .001, -.001]
            if fmt in ("e2m1", "e2m3", "e3m2", "e4m3", "e5m2", "nf4"):
                table = ref.positive_table(fmt)
                values += [ref.f32((a + b) / 2) for a, b in zip(table, table[1:])]
            text = "\n".join(format(value, ".9g") for value in values)
            result = subprocess.run([str(self.binary), "encode", fmt], input=text, capture_output=True, text=True, check=True)
            actual = list(map(int, result.stdout.splitlines()))
            expected = [ref.encode(value, fmt) for value in values]
            self.assertEqual(actual, expected, fmt)
