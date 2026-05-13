
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
# Build-time tooling required to install pinned cargo binaries.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/*
# Install pinned cargo binaries (currently: cargo-cyclonedx) into the cargo
# image's CARGO_HOME, so a single `COPY --from=rust-toolchain /usr/local/cargo`
# in downstream stages brings rustc + cargo + every pinned tool over together.
# This stage is only invalidated when install_build_tools.sh changes, not on
# every source edit.
COPY tools/wheel_build/install_build_tools.sh /tmp/install_build_tools.sh
RUN bash /tmp/install_build_tools.sh && rm /tmp/install_build_tools.sh

FROM base AS build
# setuptools-rust compiles the aiconfigurator-core cdylib into the wheel during
# `uv build --wheel`, so the Rust toolchain + C compiler must be present here.
# cargo-cyclonedx comes along inside CARGO_HOME from the rust-toolchain stage.
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
COPY tools/wheel_build/ /workspace/tools/wheel_build/
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
# under /workspace/dist per PEP 770. The SBOM is located by `find` so the
# build doesn't depend on cargo-cyclonedx's exact output-naming behavior.
RUN cargo cyclonedx \
      --manifest-path /workspace/rust/aiconfigurator-core/Cargo.toml \
      --format json
RUN set -e; \
    sbom=$(find /workspace/rust -maxdepth 5 -name '*.cdx.json' -type f -print -quit); \
    if [ -z "$sbom" ]; then \
      echo "ERROR: cargo cyclonedx did not produce a *.cdx.json file under /workspace/rust" >&2; \
      exit 1; \
    fi; \
    echo "Found SBOM at: $sbom"; \
    for whl in /workspace/dist/*.whl; do \
      python3 /workspace/tools/wheel_build/inject_sbom.py "$whl" --sbom "$sbom"; \
    done

# Single-purpose stage that contains only the built wheels. Use this target
# with `docker buildx build --target wheel-out --output type=local,dest=./dist`
# to copy wheels onto the host without exporting the whole build-stage rootfs.
FROM scratch AS wheel-out
COPY --from=build /workspace/dist/ /

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
