# ArchBench 0.2.0 source-release audit

Audit date: 2026-11-26 (Europe/Rome)

This repository was prepared from the reviewed source-only ArchBench 0.2.0
snapshot used by 170HX-XRAY. It intentionally excludes generated reports,
historical validation output, archives, SASS captures, compiled binaries,
caches, bytecode and macOS metadata. The original CI workflow is included.

## Licensing boundary

- `LICENSE` is the complete GNU AGPL version 3 text; the repository offer is
  AGPL-3.0-or-later.
- Every first-party source file carrying an SPDX identifier uses
  `AGPL-3.0-or-later`; no alternative project license is offered.
- `THIRD_PARTY_NOTICES.md` remains the dependency-license inventory.

## Validation boundary

The source package contains no prebuilt CUDA worker. Host self-tests and
independent arithmetic checks are run before publication. CUDA compilation and
physical-GPU acceptance remain environment-specific gates and are not inferred
from a source-only release.

Local source-release results:

- `./bench selftest`: 132/132 Python host tests passed on Python 3.15.2;
- `tools/check_alu_reference.py --compiler g++`: 532,480 host-executed limb
  transitions passed for widths 1–128 and both signedness modes;
- CUDA-disabled CMake/CTest: 3/3 tests passed with AppleClang 21;
- the macOS `/var` versus `/private/var` temporary-path assertion was corrected
  to compare canonical paths rather than platform aliases;
- the final tree contains no report, compiled object, archive, cache, bytecode,
  macOS metadata, credential pattern, symlink or file larger than 10 MiB.

Final commit, test counts and clean-tree checks are recorded by Git/CI; a
commit cannot embed its own hash.
