# SPDX-License-Identifier: AGPL-3.0-or-later
# NVIDIA CUDA 13.5 development image, Linux x86_64. The immutable digest was
# verified against NVIDIA's official Docker Hub tag on 2026-11-25.
FROM nvidia/cuda:13.5.0-devel-ubuntu24.04@sha256:13adcab240e4d8fde48262f293881afd5cc61ac639a3f6bc911d7b8d8e8bc4d0
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 cmake ninja-build g++ git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/archbench
COPY . .
ARG CUDA_ARCHS=80,89,120a
ARG BUILD_JOBS=2
RUN ./bench selftest && python3 tools/check_alu_reference.py --compiler g++ \
    && ./bench build --arch ${CUDA_ARCHS} --jobs ${BUILD_JOBS}
CMD ["./bench", "doctor"]
