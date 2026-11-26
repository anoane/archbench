# Generated coverage inventory

Counts below are distinct selected requests, not executions or passes. Software expansion is on for profiles and JSON matrices; exact JSONL plans replay unchanged.

| Family | Smoke with pairs | Standard with pairs | Exhaustive with pairs | Full with pairs |
|---|---:|---:|---:|---:|
| integer | 60 | 832 | 209,664 | 209,700 |
| scalar | 16 | 72 | 2,268 | 2,272 |
| bit | 18 | 18 | 567 | 570 |
| packed | 36 | 36 | 1,134 | 1,138 |
| sfu | 24 | 24 | 756 | 768 |
| convert | 144 | 144 | 339 | 495 |
| integer_width | 120 | 280 | 1,280 | 1,401 |
| tensor | 368 | 388 | 6,268 | 6,811 |
| gemm | 268 | 1,502 | 157,800 | 158,365 |
| blas | 124 | 528 | 900 | 1,034 |
| quant | 42 | 1,206 | 7,206 | 7,220 |
| quant_gemm | 84 | 2,520 | 10,080 | 11,847 |
| memory | 13 | 35 | 179 | 573 |
| collective | 8 | 22 | 68 | 82 |
| atomic | 228 | 228 | 532 | 772 |
| workload | 13 | 292 | 516 | 626 |
| launch | 2 | 2 | 4 | 6 |
| transfer | 8 | 24 | 72 | 85 |
| sparse | 8 | 24 | 72 | 82 |
| fft | 24 | 192 | 240 | 282 |
| **Total** | **1,608** | **8,369** | **399,945** | **404,129** |

`full` is the deduplicated union of all original profiles plus every bundled configuration. The arithmetic parameter matrix dominates the count; case count is not a feature-quality or peak-performance score.

| Profile | Without added pairs | With pairs |
|---|---:|---:|
| smoke | 1,360 | 1,608 |
| standard | 6,156 | 8,369 |
| exhaustive | 315,894 | 399,945 |
| full | 318,948 | 404,129 |

| Configuration | Source entries | With pairs |
|---|---:|---:|
| `acceptance.json` | 192 | 235 |
| `acceptance_tensor_emulation.json` | 17 | 17 |
| `gemm_throughput.json` | 6 | 12 |
| `llm_shapes.json` | 372 | 717 |
| `memory_sweep.json` | 408 | 408 |
| `tensor_throughput.json` | 140 | 140 |

Some library requests share a software counterpart. Pair coverage counts requested implementation pairs separately from unique numerical operation IDs. Every original requested case is retained; a software pass never establishes a missing native/library pass.

Reproduce the default selection with `./bench list --profile full --count`; add `--no-software-pairs` for unexpanded counts. Use `./bench plan` to freeze exact identities before an architecture comparison. See [scope](scope.md), [methodology](methodology.md) and [result format](result-format.md).
