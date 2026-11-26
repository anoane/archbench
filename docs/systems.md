# Memory, synchronization, transfers, and composite workloads

`src/systems.cu` implements eight worker families. It is a collection of
measurement baselines, with explicit numerical references and explicit timing
scope. A portable baseline is not claimed to be the fastest implementation of
the corresponding algorithm.

The worker receives flat `--key value` arguments. The suite's planner emits
these arguments automatically. All sizes count **elements**, not bytes, unless
an option or metric explicitly says bytes. A transfer element is four bytes.

## Operation inventory

| Family | Operations | Accepted data types | Principal parameters |
|---|---|---|---|
| `memory` | `read`, `write`, `copy`, `triad`, `strided`, `gather`, `scatter` | `f32` | `elements`, `stride`, `threads`, `blocks`, `seed` |
| `memory` | `pointer_chase` | `u32` | `elements`, `iterations`, `seed` |
| `memory` | `shared_bank` | `f32` | `elements`, `stride`, `iterations`, `threads`, `blocks` |
| `memory` | `cp_async`, `shared_copy` | `f32` | `elements`, `threads`, `blocks` |
| `atomic` | `add`, `min`, `max`, `exchange`, `cas`, `and`, `or`, `xor` | `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64` | `elements`, `addresses`, `iterations`, `threads`, `blocks` |
| `atomic` | `add`, `min`, `max`, `exchange`, `cas` | `f32`, `f64` | same |
| `atomic` | `add` | `f16`, `bf16` | same |
| `collective` | `shuffle`, `ballot`, `barrier`, `block_reduce`, `block_scan`, `reduce_sum`, `scan`, `sort` | `f32`; `ballot`/`barrier` also accept `u32` output labels | `elements`, `threads`, `blocks`, `iterations` where applicable |
| `workload` | `softmax`, `rmsnorm`, `layernorm`, `rope` | `f16`, `bf16`, `f32`, `f64` | `rows`, `cols`, `threads`, `blocks` |
| `workload` | `relu`, `silu`, `gelu`, `gelu_tanh`, `swiglu` | `f16`, `bf16`, `f32`, `f64` | `elements`, `threads`, `blocks` |
| `workload` | `attention` | `f16`, `bf16`, `f32`, `f64` | `sequence`, `head-dim`, `heads`, `batch`, `causal`, `check-rows` |
| `launch` | `empty`, `graph` | not applicable | `iterations`, `graph-nodes` |
| `transfer` | `h2d`, `d2h`, `d2d` | `f32` | `elements`, `pinned` |
| `transfer` | `managed_prefetch`, `managed_first_touch` | `f32` | `elements`, `threads`, `blocks` |
| `transfer` | `p2p` | `f32` | `elements`, `peer` |
| `sparse` | `csr_spmv`, `csr_spmm` | `f32`, `f64` | `rows`, `cols`, `nnz-per-row`, `n` for SpMM, `preprocess` |
| `fft` | `c2c`, `r2c`, `c2r` | `f16`, `bf16`, `f32`/`c64`, `f64`/`c128` | `rank`, `n` or `nx`, `ny`, `nz`, `batch`, `direction`, `check-bins` |

`--threads` defaults to 256 and must be a multiple of 32 within the device
limit. Reduction kernels require a power of two. `--blocks` defaults to four
times the SM count for grid-stride kernels, capped by the amount of work.
Row-wise and block-collective kernels derive their grid from the shape, so a
requested block count does not override those grids. Results record actual
dimensions where relevant; the case parameters remain available in the run
record.

## Memory measurements

The streaming cases use deterministic nonconstant values. Gather and scatter
use a seeded random **permutation**. Scatter therefore has a single writer per
destination, unlike an intentionally racy scatter benchmark. `strided` performs
`output[i] = input[(i * stride) % elements]`; the modulo/index arithmetic is part
of its measured cost. It is not a raw address-generation-free instruction test.

`read` reduces all input into one sum per CUDA thread. Its byte count includes
those sink stores, and its results are checked against independently computed
CPU sums. A repeated load that the compiler can delete is not used to claim
bandwidth. `triad` computes `x + 0.5*z`, counting two arithmetic operations per
element.

Memory byte counts describe requested logical traffic. They are not a claim
about physical DRAM transactions, cache-line utilization, write allocation, or
compression. Working-set sweeps expose changes in behavior, but the suite does
not label a measurement “L1”, “L2”, or “DRAM” solely from an assumed cache size.
Use actual hardware counters and the separate cache-state metadata to make
that attribution.

`pointer_chase` constructs one random Hamiltonian cycle through all indices.
One GPU thread follows it with dependent volatile loads. It reports event
timing and a separate `cycles_per_dependent_load` measurement from a
post-timing `clock64()` sample. That sample includes loop and clock-read
overhead. The number of iterations need not traverse the entire allocation:
choose it deliberately relative to the working set. The final index is
checked exactly against the known cycle.

`shared_bank` places lane `l` at word address `l * stride`. Lanes access
different words, so same-bank access is not accidentally a broadcast. The
nominal conflict degree is `gcd(stride, 32)` under the 32-bank, four-byte word
mapping. The kernel includes initialization, synchronization, accumulation,
and output stores. It is a shared-memory stress case, not a claim that every
event cycle is a bank-conflict penalty.

`cp_async` and `shared_copy` have identical single-stage shared-memory copy
layouts. `cp_async` emits 16-byte `cp.async.ca.shared.global`, commit, and wait
instructions on the target architectures. `elements` must be divisible by
four. Tail vectors in a partially populated block are handled without an
out-of-range read. These cases compare asynchronous instructions with a
synchronous staging baseline; they do not pretend to be a fully tuned
multistage GEMM pipeline. Inspect the produced SASS to confirm lowering.

## Atomics and contention

`elements` logical workers each perform `iterations` updates to
`worker_index % addresses`. Set `addresses=1` for global contention, a small
power of two for a hot set, and `addresses=elements` for distinct destinations.
Every address must be touched, so `addresses <= elements` is required.

Integer add uses defined hardware modular arithmetic. Signed 64-bit addition
is performed through the corresponding unsigned atomic representation.
Eight- and sixteen-bit signed/unsigned atomics use CAS on the aligned
containing 32-bit word, preserving neighboring fields. Allocations are padded
to complete words. These subword cases are labelled `emulated`; they include
false sharing between adjacent fields and retry overhead.
Integer bit operations and floating-point min/max/exchange/CAS are kept
separate: floating-point min/max use explicit CAS loops and are labelled
`emulated`, while floating-point add uses the CUDA atomic API. The CUDA API
may itself lower a type to a hardware instruction or an emulation sequence on
different architectures. A source-level intrinsic is not sufficient evidence
for a native-instruction claim.

All additions add one. Half precision unit increments stop changing at 2048,
bfloat16 increments at 256, and FP32 increments at 2^24. The reference checks
this rounding behavior instead of treating it as a missing update. This is
intentional and recorded in metadata; it does not model a general floating-point
atomic reduction with an arbitrary distribution of summands.

Timed repetitions retain their state to avoid hiding a reset in the atomic
throughput figure. Validation uploads the initial state again and executes a
fresh single workload. For `cas`, the operation compares zero and exchanges
one, making the steady-state timing a failed-comparison benchmark. For an
application with frequent successful CAS updates, use a separate workload
and do not reuse this result as its throughput.

The reported rate counts attempted atomic operations. No physical bandwidth
is inferred from an atomic instruction count. Legacy CUDA atomics here have
device scope; these cases do not test every memory order or system scope.

## Collectives

Warp shuffle is a dependent lane-rotation chain. Padding lanes participate
with zero and a full-warp mask, so nonmultiples of the warp size remain valid.
Ballot checks the entire expected bit mask. Barrier testing includes
dependent shared-memory communication around two barriers per iteration;
its operation count is **CTA barrier instances**, not lane participation.

CUB provides block reduction, block inclusive scan, device reduction,
device inclusive scan, and radix sort. Device primitives query workspace
before timing and reuse it. Block scan restarts at each block boundary;
`scan` is the device-wide version. Sort validation compares every output key
to a CPU sort, checking order and multiplicity together. Logical byte counts
exclude internal CUB passes, so they must not be interpreted as bus traffic.

## Composite workloads

FP16, BF16, and FP32 storage use FP32 accumulation. FP64 storage uses FP64
accumulation. Softmax subtracts the row maximum. LayerNorm includes generated
nonconstant gamma and beta. RMSNorm includes gamma. Both use epsilon=1e-5.
GELU has separate erf and tanh-approximation operations. SwiGLU uses two
separate input arrays.

RoPE uses interleaved pairs, base 10000, and the row index as the position.
`cols` must be even. Its timed kernel evaluates powers and trigonometric
functions; it is not a precomputed sine/cosine-cache lookup benchmark.

The attention workload explicitly runs three stages: naive QK, stable
softmax, and naive AV. The score matrix is materialized. It supports causal
masking, batched heads, and FP16/BF16/FP32/FP64 storage. It is a correctness
baseline and a deliberately different memory/computation pattern from
FlashAttention. It must not be reported as FlashAttention-2 or -3.

All outputs are checked for the simple workloads. Attention checks all
channels of evenly spaced output rows using independent long-double CPU
dot products and softmax; `check-rows` defaults to 16 and may be raised to
the full row count. The output record distinguishes sampled validation from
full validation. FLOP counts for attention cover the implemented QK and AV
multiply-add loops; softmax, SFU operations, and indexing are not silently
counted as FLOPs. For causal attention, the QK stage skips masked pairs while
the AV stage still visits all keys, matching the implementation's count.

## Timing host interaction correctly

Pinned H2D/D2H and device-to-device copies use CUDA event timing. Pageable
copies use host wall time through stream synchronization because host staging
can be invisible to device event intervals. Transfer bandwidth counts the
one-way payload; it does not double the bytes simply because the copy reads
one allocation and writes another.

Managed-memory cases migrate to the CPU and write **every element before
each sample**, outside timing. `managed_prefetch` times migration to the GPU.
`managed_first_touch` times a GPU increment kernel plus demand migration.
Both use host completion timing and record that difference explicitly. The
CUDA 13 `cudaMemLocation` prefetch API is used on CUDA 13 builds. The runtime's
managed-memory and concurrent-access capabilities are checked.

P2P validates visibility, peer-access capability, allocation, and the copied
payload. `peer` is a visible CUDA device ordinal. The suite does not infer
that a P2P path is NVLink: PCIe, NVSwitch, driver configuration, topology,
virtualization, and the exact product still matter. Record topology alongside
the benchmark. A missing peer path yields an unsupported case, not zero
bandwidth.

`launch empty` measures a group of ordinary empty-kernel launches.
`launch graph` captures `graph-nodes` empty kernels and repeatedly launches
the executable graph. Graph capture/instantiate/upload cost is reported
separately. Both report device event measurements and separate host
submission and synchronized-completion medians. Rates count CUDA API launch
calls; graph node count is a separate metric. Empty kernels have no numerical
output, so their validation is explicitly `not_applicable`, with successful
CUDA completion recorded as the only validation scope.

## Sparse and FFT libraries

cuSPARSE cases require `ARCHBENCH_HAS_CUSPARSE=1` and linking cuSPARSE.
CSR rows have sorted unique column indices, deterministic values, and an
adjustable `nnz-per-row`. SpMV and column-major SpMM both use beta=0, so timed
repetitions cannot accumulate into C. Preprocessing is enabled by default,
timed separately, and excluded from steady-state execution. Every output is
validated against a CPU CSR product. General CSR sparsity is **not** the
structured 2:4 Tensor Core sparse-MMA path; that requires a different backend.

cuFFT cases require `ARCHBENCH_HAS_CUFFT=1` and linking cuFFT. Rank may be
one, two, or three. Dimensions are `[nx]`, `[ny,nx]`, or `[nz,ny,nx]` and batch
is contiguous. C2C supports forward and inverse directions; R2C requires
`direction=forward`, C2R requires `direction=inverse`. Transforms are not
normalized. Plan and workspace creation occur outside steady-state timing.
C2R defaults to inverse, while C2C/R2C default to forward. FP16 and BF16
use the cuFFT Xt APIs with matching complex execution types and require
power-of-two dimensions. Their inputs are scaled by `1/sqrt(volume)` to
control overflow; reference calculations use the actual rounded stored
values. The input scale and numerical validation tolerances are recorded
with each result, and maximum observed errors are retained.

C2R input is generated as a valid Hermitian spectrum directly, rather than
using the same FFT implementation to produce the reference. cuFFT may modify
C2R input even out of place, so this case restores an immutable device copy
before each transform and **includes that restoration in the measured time**.
The metadata states this scope. Compare it only with an equivalent workload.

`check-bins` defaults to 32 evenly spaced output elements. Each is checked
against a direct independent long-double multidimensional DFT. This catches
transform direction, normalization, packing, multidimensional indexing,
batch-distance, and real/complex interpretation mistakes; it is stronger than
only checking a forward/inverse round trip. Raising `check-bins` can make CPU
validation expensive: its cost is outside GPU timing. The traditional
5*N*log2(N) complex-FFT FLOP estimate is a separate model field, while the
primary rate counts transformed points.

## Verification status and source references

This module was written for CUDA 13.5. The authoring environment has neither
NVCC nor a GPU. A host C++ syntax pass with deliberately minimal API stubs
checks ordinary C++ structure, but does **not** validate CUDA compilation,
PTX assembly, CUDA library signatures, or hardware correctness. The archive's
real GPU build, correctness smoke runs, and Compute Sanitizer runs are the
required validation gates on each target machine. No GPU performance numbers
are supplied as if they had been measured here.

Primary API references consulted:

- [CUDA 13.5 runtime memory management](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html)
- [CUDA 13.5 graph management](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__GRAPH.html)
- [CUDA language extensions and atomics](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)
- [CUDA BF16 arithmetic functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__ARITHMETIC.html)
- [CUB device primitives](https://nvidia.github.io/cccl/cub/api/device.html)
- [cuSPARSE generic API](https://docs.nvidia.com/cuda/cusparse/)
- [cuFFT documentation](https://docs.nvidia.com/cuda/cufft/index.html)

