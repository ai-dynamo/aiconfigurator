
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

FROM base AS build
# The lower `aiconfigurator-core` wheel owns the `aiconfigurator_core`
# extension and SDK/data payload, so build it before the upper CLI/generator
# wheel. maturin needs the Rust crate sources, a C linker (gcc), and
# ca-certificates so cargo can fetch crates from crates.io over TLS. Rust itself
# is bootstrapped by maturin's build backend.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/*
COPY src/ /workspace/src/
COPY spica/ /workspace/spica/
COPY rust/ /workspace/rust/
COPY LICENSE pyproject.toml README.md /workspace/
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
# Build the native `aiconfigurator-core` wheel AND an sdist. The wheel is
# linux_x86_64-only (native PyO3 extension); the sdist is the fallback that lets
# `pip install` build from source on other platforms (macOS, Linux aarch64) when
# a Rust toolchain is present. Without it, ARM64/macOS users cannot install AIC
# via the documented pip command. The upper `aiconfigurator` wheel is
# py3-none-any, so it needs no sdist.
RUN cd src/aiconfigurator-core && uv build --wheel --out-dir /workspace/dist && \
    uv build --sdist --out-dir /workspace/dist && \
    cd /workspace && uv build --wheel --out-dir /workspace/dist

FROM base AS runtime
COPY --from=build /workspace/dist /wheelhouse
RUN uv pip install /wheelhouse/*.whl && \
    rm -rf /wheelhouse

FROM runtime AS test

# The `aiconfigurator_core` extension ships inside the lower wheel built in the
# `build` stage, so the test stage only adds the upper wheel's dev extras and
# test sources.
COPY --from=build /workspace/dist /wheelhouse
RUN AIC_WHL=$(ls /wheelhouse/aiconfigurator-*.whl) && \
    uv pip install /wheelhouse/aiconfigurator_core-*.whl "${AIC_WHL}[dev]" && \
    rm -rf /wheelhouse

COPY pytest.ini /workspace/
COPY tests/ /workspace/tests/
COPY src/ /workspace/src/
COPY tools/ /workspace/tools/
COPY collector/ /workspace/collector/
