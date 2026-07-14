
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
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV MPLCONFIGDIR=/tmp/matplotlib

# The `aiconfigurator` wheel (Rust extension + SDK/data payload + CLI, generator,
# webapp, Spica) is built in the PyPA manylinux_2_28 image so the native PyO3
# `.so` links glibc 2.28 and maturin's auditwheel stamps the portable
# `manylinux_2_28_<arch>` tag. We add `uv` and a rustup toolchain so cargo can
# fetch crates over TLS.
#
# TARGETARCH (amd64/arm64) selects the base per `--platform`; only the matching
# stage builds. aarch64: `docker build --platform linux/arm64 --target build .`.
FROM quay.io/pypa/manylinux_2_28_x86_64 AS manylinux-amd64
FROM quay.io/pypa/manylinux_2_28_aarch64 AS manylinux-arm64
ARG TARGETARCH
FROM manylinux-${TARGETARCH} AS build
WORKDIR /workspace
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV RUSTUP_HOME=/opt/rust/rustup \
    CARGO_HOME=/opt/rust/cargo \
    PATH=/opt/rust/cargo/bin:${PATH}
# Download to disk (not `curl | sh`, which hides a failed download behind sh's
# exit status); pin the toolchain for reproducible release wheels.
RUN curl --proto '=https' --tlsv1.2 -sSf --retry 5 --retry-connrefused --retry-delay 2 \
        https://sh.rustup.rs -o /tmp/rustup-init.sh \
    && sh /tmp/rustup-init.sh -y --profile minimal --default-toolchain 1.96.0 \
    && rm /tmp/rustup-init.sh

# Pin to the image's CPython 3.12 so uv/maturin never fetch a standalone one.
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
# `uv build` runs maturin's PEP517 backend, which defaults to `--compatibility
# off`; MATURIN_PEP517_ARGS overrides it to stamp the manylinux_2_28 tag (maturin
# appends it last, so it wins). `python` isn't on PATH here -> verify via UV_PYTHON.
RUN MATURIN_PEP517_ARGS="--compatibility manylinux_2_28" \
        uv build --wheel --out-dir /workspace/dist && \
    "${UV_PYTHON}" tools/verify_release_wheels.py /workspace/dist

FROM base AS runtime
COPY --from=build /workspace/dist /wheelhouse
RUN uv pip install /wheelhouse/aiconfigurator-*.whl && \
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
