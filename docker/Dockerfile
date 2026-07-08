
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nvcr.io/nvidia/base/ubuntu:jammy-20250619 AS base

WORKDIR /workspace

ENV VIRTUAL_ENV=/opt/aiconfigurator/venv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN mkdir /opt/aiconfigurator && \
    uv venv --python 3.12 ${VIRTUAL_ENV} && \
    chmod a+rx /root && \
    chmod -R a+rX /root/.local
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV MPLCONFIGDIR=/tmp/matplotlib

# The `aiconfigurator` wheel owns the Rust extension, SDK/data payload, CLI,
# generator, webapp, and Spica packages. It is built inside the PyPA
# manylinux_2_28 image so the native (Rust/PyO3) `_aiconfigurator_core.*.so`
# links against glibc 2.28; maturin's built-in auditwheel then checks the
# manylinux_2_28 policy, bundles any non-whitelisted shared libs (the abi3
# extension needs only glibc/libgcc, so none), and stamps the portable
# `manylinux_2_28_<arch>` tag. The small `aiconfigurator-core` wheel built
# afterward is a pure-Python compatibility metapackage and needs no repair.
#
# The manylinux image already ships auditwheel, patchelf, a gcc toolset, and
# ca-certificates; we add the static `uv` build frontend and a rustup toolchain
# so maturin's cargo build can fetch crates from crates.io over TLS.
#
# The base is selected per target architecture: BuildKit populates TARGETARCH
# (amd64/arm64) from `--platform` (default: the build host's arch), and only the
# matching stage below is built. Build an aarch64 wheel with:
#   docker build -f docker/Dockerfile --platform linux/arm64 --target build .
# natively on ARM, or via QEMU emulation / `docker buildx` on an x86 host. The
# crate is architecture-agnostic (no target_arch code); the parquet data payload
# is identical across arches.
FROM quay.io/pypa/manylinux_2_28_x86_64 AS manylinux-amd64
FROM quay.io/pypa/manylinux_2_28_aarch64 AS manylinux-arm64
ARG TARGETARCH
FROM manylinux-${TARGETARCH} AS build
WORKDIR /workspace
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV RUSTUP_HOME=/opt/rust/rustup \
    CARGO_HOME=/opt/rust/cargo \
    PATH=/opt/rust/cargo/bin:${PATH}
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain stable

# Pin the build interpreter to the image's CPython 3.12 so uv/maturin never fetch
# a standalone one. abi3 (abi3-py310) means the single wheel serves CPython >=3.10.
ENV UV_PYTHON=/opt/python/cp312-cp312/bin/python3

COPY src/ /workspace/src/
COPY rust/ /workspace/rust/
COPY LICENSE pyproject.toml README.md /workspace/
COPY tools/verify_release_wheels.py /workspace/tools/verify_release_wheels.py
# Fail fast if Git LFS content wasn't materialized before the build — otherwise
# the wheel ships pointer stubs in place of perf data (see DYN-2725).
RUN set -e; \
    bad=$(grep -rlE "^version https://git-lfs" /workspace/src/aiconfigurator/systems 2>/dev/null || true); \
    if [ -n "$bad" ]; then \
        echo "ERROR: Unmaterialized Git LFS pointer files found:" >&2; \
        echo "$bad" >&2; \
        echo "Run 'git lfs install && git lfs pull' in the source tree before building." >&2; \
        exit 1; \
    fi
# Build the main wheel. Under a frontend like `uv build`, maturin's PEP517
# backend injects `--compatibility off` (plain linux) by default, and that CLI
# flag overrides `[tool.maturin] compatibility`. MATURIN_PEP517_ARGS re-asserts
# the manylinux_2_28 target -- maturin appends it after the injected `off`, so it
# wins -- which drives maturin's built-in auditwheel to check the 2_28 policy and
# stamp the `manylinux_2_28_<arch>` tag. The `aiconfigurator-core` metapackage is
# pure-Python. `python` is not on PATH in the manylinux image, so the verifier
# runs via ${UV_PYTHON}.
RUN MATURIN_PEP517_ARGS="--compatibility manylinux_2_28" \
        uv build --wheel --out-dir /workspace/dist && \
    ( cd src/aiconfigurator-core && uv build --wheel --out-dir /workspace/dist ) && \
    "${UV_PYTHON}" tools/verify_release_wheels.py /workspace/dist

FROM base AS runtime
COPY --from=build /workspace/dist /wheelhouse
RUN uv pip install /wheelhouse/*.whl && \
    rm -rf /wheelhouse

FROM runtime AS test

# The `aiconfigurator_core` extension and all Python packages ship inside the
# main wheel, so the test stage only adds that wheel's dev extras and sources.
COPY --from=build /workspace/dist /wheelhouse
RUN AIC_WHL=$(ls /wheelhouse/aiconfigurator-*.whl) && \
    uv pip install "${AIC_WHL}[dev]" && \
    rm -rf /wheelhouse

COPY pytest.ini /workspace/
COPY tests/ /workspace/tests/
COPY docs/spica/examples/ /workspace/docs/spica/examples/
COPY src/ /workspace/src/
COPY tools/ /workspace/tools/
COPY collector/ /workspace/collector/
