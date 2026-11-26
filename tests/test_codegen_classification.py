# SPDX-License-Identifier: AGPL-3.0-or-later
"""A tensor opcode of a different arithmetic width is a compiler expansion."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("classify_mma_sass", Path(__file__).resolve().parents[1]/"tools/classify_mma_sass.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CodegenClassificationTests(unittest.TestCase):
    def test_int8_tensor_instructions_are_not_native_int4_or_binary(self):
        sass = "/*0040*/ IMMA.16832.S8.U8 R4, R8, R12, RZ ;\n/*0050*/ IMAD R2, R2, R3, R4 ;"
        for probe in ("int4_mma", "binary_mma"):
            report = MODULE.classify(sass, probe)
            self.assertEqual(report["codegen_evidence"], "compiler_expanded_mma")
            self.assertFalse(report["runtime_verified"])

    def test_direct_opcode_matching_and_predicates(self):
        for probe, mnemonic in (("fp64_mma", "DMMA.8x8x4"), ("int4_mma", "IMMA.16864.S4.U4"),
                                ("binary_mma", "BMMA.168256.XOR.POPC")):
            result = MODULE.classify(f"/*0120*/ @!P0 {mnemonic} R0, R1, R2, R3 ;", probe)
            self.assertEqual(result["codegen_evidence"], "direct_mma_sass")
            self.assertEqual(result["instruction_mnemonics"][mnemonic], 1)

    def test_empty_or_failed_disassembly_is_not_an_emulation_claim(self):
        self.assertEqual(MODULE.classify("cuobjdump fatal: missing nvdisasm", "fp64_mma")["codegen_evidence"], "not_inspected")
        self.assertEqual(MODULE.classify("/*0010*/ DFMA R0, R1, R2, R3 ;", "fp64_mma")["codegen_evidence"], "scalar_expansion")
