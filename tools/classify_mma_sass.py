#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Classify isolated probe disassembly, never claim device runtime support.

This checks the requested arithmetic width/opcode, not just the presence of any
Tensor Core instruction: an INT4 request expanded into INT8 MMA is emulated.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def classify(text: str, probe: str) -> dict:
    mnemonics = Counter(re.findall(r'/\*[0-9a-fA-F]+\*/\s+(?:@!?[A-Za-z0-9]+\s+)?([A-Z][A-Za-z0-9_.]*)\s', text))
    mma = {name: count for name, count in mnemonics.items() if 'MMA' in name.split('.')[0]}
    if probe == 'fp64_mma':
        exact = any(name.startswith('DMMA.') or 'F64' in name for name in mma)
    elif probe == 'int4_mma':
        exact = any(name.startswith('IMMA.') and ('.S4' in name or '.U4' in name) for name in mma)
    elif probe == 'binary_mma':
        exact = any(name.startswith('BMMA.') or '.B1' in name for name in mma)
    else:
        raise ValueError('unknown probe: '+probe)
    lowering = ('direct_mma_sass' if exact else 'compiler_expanded_mma' if mma
                else 'scalar_expansion' if mnemonics else 'not_inspected')
    return {'schema_version': 1, 'probe': probe, 'codegen_evidence': lowering,
            'sass_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'instruction_mnemonics': dict(mnemonics), 'mma_mnemonics': mma,
            'runtime_verified': False,
            'scope': 'Isolated exact-target probe machine-code inspection; not a GPU execution, full benchmark-kernel inspection, or hardware throughput claim.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',choices=('fp64_mma','int4_mma','binary_mma'),required=True)
    parser.add_argument('--sass',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=classify(args.sass.read_text(),args.probe)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(result['codegen_evidence'])

if __name__=='__main__': main()
