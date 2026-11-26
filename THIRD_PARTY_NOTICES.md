# Third-party interfaces and numeric constants

The CUDA, cuBLAS, cuBLASLt, cuFFT, cuSPARSE, Nsight Compute, and Compute
Sanitizer libraries/tools are not redistributed. They remain subject to
NVIDIA's licenses. CUB is used from the installed CUDA toolkit. CUTLASS is an
optional user-supplied checkout; its source is not vendored here. The adapter
records the actual checkout revision and dirty state before building it.

## NF4 codebook

The 16 NF4 numeric constants match bitsandbytes' public format table:
https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py
and https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/csrc/kernels.cu
(accessed 2026-11-25).

The surrounding encoder/decoder is an independent implementation. ArchBench's
packing is not the bitsandbytes tensor layout and must not be used to decode a
bitsandbytes artifact without a layout adapter.

Relevant bitsandbytes license notice:

MIT License

Copyright (c) Facebook, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
