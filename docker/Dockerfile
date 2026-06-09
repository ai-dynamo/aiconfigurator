
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Single multi-stage Dockerfile that builds BOTH the pure-Python
# `aiconfigurator` wheel and the platform-specific `aiconfigurator-core`
# extension wheel, plus a `combined-test` stage that installs both into one
# venv and exercises the rust backend + python fallback end-to-end.
#
# Local / CI invocations (override WHEEL_BUILD_BASE for non-default platforms):
#   docker buildx build --target wheel-out-pure --output type=local,dest=./dist -f docker/Dockerfile .
#   docker buildx build --target wheel-out-rust --output type=local,dest=./dist \
#       --build-arg WHEEL_BUILD_BASE=quay.io/pypa/manylinux_2_28_aarch64 \
#       --platform linux/arm64 -f docker/Dockerfile .
#   docker buildx build --target combined-test \
#       --build-arg WHEEL_BUILD_BASE=quay.io/pypa/manylinux_2_28_aarch64 \
#       --platform linux/arm64 -f docker/Dockerfile .

# The wheel-build base image. CI sets this per matrix row to one of:
#   - quay.io/pypa/manylinux_2_28_{x86_64,aarch64}     (glibc 2.28, RHEL 8+)
#   - quay.io/pypa/manylinux2014_{x86_64,aarch64}      (glibc 2.17, EL7-era)
#   - quay.io/pypa/musllinux_1_2_{x86_64,aarch64}      (musl, Alpine/distroless)
# Default keeps single-platform local builds on x86_64 hosts working without
# an extra --build-arg flag. Auditwheel ≥ 5 auto-detects the libc family
# from the binary, so the same downstream `auditwheel repair` step works
# across all three families.
ARG WHEEL_BUILD_BASE=quay.io/pypa/manylinux_2_28_x86_64

# ----------------------------------------------------------------------------
# Rust-side wheel build (aiconfigurator-core)
# ----------------------------------------------------------------------------

# rust-toolchain: install rustup + the pinned cargo-cyclonedx into the same
# wheel-build base so the binaries are linkable against the build stage's
# libc. cargo-cyclonedx is on PATH so maturin's PEP 770 SBOM emission fires
# automatically when the rust-core wheel is built.
FROM ${WHEEL_BUILD_BASE} AS rust-toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
      sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"
RUN cargo install --locked cargo-cyclonedx --version 0.5.7

# rust-build: produce the aiconfigurator-core wheel. auditwheel
# repair stamps the wheel with the appropriate platform tag based on the
# actual symbol versions it observes — manylinux_2_28_<arch>,
# manylinux2014_<arch>, or musllinux_1_2_<arch>.
FROM ${WHEEL_BUILD_BASE} AS rust-build
COPY --from=rust-toolchain /root/.cargo /root/.cargo
COPY --from=rust-toolchain /root/.rustup /root/.rustup
ENV CARGO_HOME=/root/.cargo
ENV RUSTUP_HOME=/root/.rustup
# cp310 is the build interpreter — matches maturin's abi3-py310 tag.
ENV PATH="${CARGO_HOME}/bin:/opt/python/cp310-cp310/bin:${PATH}"
RUN pip install --no-cache-dir 'maturin>=1.10,<2'

WORKDIR /workspace/rust/aiconfigurator-core
COPY rust/aiconfigurator-core/ ./
RUN maturin build --release --out /workspace/rust-raw-dist
RUN mkdir -p /workspace/rust-dist && \
    for whl in /workspace/rust-raw-dist/*.whl; do \
      auditwheel repair --wheel-dir /workspace/rust-dist "$whl"; \
    done

# Single-purpose stage: only the rust-core wheel(s).
FROM scratch AS wheel-out-rust
COPY --from=rust-build /workspace/rust-dist/ /

# ----------------------------------------------------------------------------
# Pure-Python wheel build (aiconfigurator)
# ----------------------------------------------------------------------------

# pure-build: setuptools wheel, py3-none-any, platform-independent. No Rust
# toolchain, no Docker-image-base constraint — anything with Python 3.10+
# and uv works. Use the same base for consistency with the test stage.
FROM python:3.12-slim AS pure-build
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /workspace
COPY src/ /workspace/src/
COPY LICENSE pyproject.toml README.md /workspace/
# Fail fast if Git LFS content wasn't materialized — otherwise the wheel
# ships pointer stubs in place of perf data (see DYN-2725).
RUN set -e; \
    bad=$(grep -rlE "^version https://git-lfs" /workspace/src/aiconfigurator/systems 2>/dev/null || true); \
    if [ -n "$bad" ]; then \
        echo "ERROR: Unmaterialized Git LFS pointer files found:" >&2; \
        echo "$bad" >&2; \
        echo "Run 'git lfs install && git lfs pull' in the source tree before building." >&2; \
        exit 1; \
    fi
RUN uv build --wheel --out-dir /workspace/pure-dist

# Single-purpose stage: only the pure-Python wheel.
FROM scratch AS wheel-out-pure
COPY --from=pure-build /workspace/pure-dist/ /

# ----------------------------------------------------------------------------
# Unified combined smoke test
# ----------------------------------------------------------------------------
# Installs BOTH wheels into one venv and exercises the rust backend +
# python fallback. Buildx auto-resolves the pure-build + rust-build
# upstreams, so a single `--target combined-test` builds and verifies the
# whole pipeline. Identical command works locally and in CI.

FROM python:3.12-slim AS combined-test
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY --from=pure-build /workspace/pure-dist /wheels/pure
COPY --from=rust-build /workspace/rust-dist /wheels/rust

ENV PATH="/opt/v/bin:${PATH}"
ENV MPLCONFIGDIR=/tmp/matplotlib

RUN uv venv /opt/v && \
    uv pip install /wheels/pure/*.whl /wheels/rust/*.whl

# --- Rust path ---
RUN aiconfigurator cli default \
      --model-path Qwen/Qwen3-32B --total-gpus 1 --system h200_sxm \
      --engine-step-backend rust --save-dir /tmp/out-rust

# --- Python fallback path ---
# Uninstall the rust extension and re-run with the python backend, proving
# the bare-aiconfigurator install path still works.
RUN uv pip uninstall aiconfigurator-core && \
    aiconfigurator cli default \
      --model-path Qwen/Qwen3-32B --total-gpus 1 --system h200_sxm \
      --engine-step-backend python --save-dir /tmp/out-python

# ----------------------------------------------------------------------------
# Existing runtime + test stages (NVIDIA Ubuntu base)
# ----------------------------------------------------------------------------

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

# runtime: install both wheels (pure + rust). The rust-core wheel is
# platform-specific (manylinux_2_28); the base image (Ubuntu jammy, glibc
# 2.35) is compatible.
FROM base AS runtime
COPY --from=pure-build /workspace/pure-dist /wheelhouse-pure
COPY --from=rust-build /workspace/rust-dist /wheelhouse-rust
RUN uv pip install /wheelhouse-pure/*.whl /wheelhouse-rust/*.whl && \
    rm -rf /wheelhouse-pure /wheelhouse-rust

FROM runtime AS test
COPY --from=pure-build /workspace/pure-dist /wheelhouse-pure
RUN WHL=$(ls /wheelhouse-pure/*.whl) && \
    uv pip install "${WHL}[dev]" && \
    rm -rf /wheelhouse-pure

COPY pytest.ini /workspace/
COPY tests/ /workspace/tests/
COPY src/ /workspace/src/
COPY tools/ /workspace/tools/
COPY collector/ /workspace/collector/
