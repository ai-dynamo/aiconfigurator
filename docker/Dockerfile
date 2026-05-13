
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

FROM rust:1-slim-bookworm AS rust-toolchain

FROM base AS build
# setuptools-rust compiles the aiconfigurator-core cdylib into the wheel during
# `uv build --wheel`, so the Rust toolchain + C compiler must be present here.
COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo
COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
ENV PATH="${CARGO_HOME}/bin:${PATH}"
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/*

COPY src/ /workspace/src/
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
RUN uv build --wheel --out-dir /workspace/dist

# Generate a CycloneDX SBOM for the Rust crate and inject it into every wheel
# under /workspace/dist per PEP 770.
COPY tools/wheel_build/ /workspace/tools/wheel_build/
RUN bash /workspace/tools/wheel_build/install_build_tools.sh
RUN cargo cyclonedx \
      --manifest-path /workspace/rust/aiconfigurator-core/Cargo.toml \
      --format json \
      --output-pattern bom
RUN for whl in /workspace/dist/*.whl; do \
      python3 /workspace/tools/wheel_build/inject_sbom.py "$whl" \
        --sbom /workspace/rust/aiconfigurator-core/bom.cdx.json; \
    done

FROM base AS runtime
COPY --from=build /workspace/dist /wheelhouse
RUN WHL=$(ls -d /wheelhouse/*) && \
    uv pip install $WHL && \
    rm -rf /wheelhouse

FROM runtime AS test

COPY --from=build /workspace/dist /wheelhouse
RUN WHL=$(ls -d /wheelhouse/*) && \
    uv pip install "$WHL[dev]" && \
    rm -rf /wheelhouse

COPY pytest.ini /workspace/
COPY tests/ /workspace/tests/
COPY src/ /workspace/src/
COPY tools/ /workspace/tools/
COPY collector/ /workspace/collector/
