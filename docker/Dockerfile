
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

# The standalone `aiconfigurator-core` wheel owns the Rust extension and the
# SDK/data payload. The upper `aiconfigurator` wheel owns the application and
# depends on the exact matching core version. Core is built in the PyPA
# manylinux_2_28 image so the native PyO3 `.so` links glibc 2.28 and maturin's
# auditwheel stamps the portable
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
COPY aic-core/ /workspace/aic-core/
COPY LICENSE pyproject.toml README.md /workspace/
COPY tools/verify_release_wheels.py /workspace/tools/verify_release_wheels.py
COPY tools/verify_installed_package_layers.py /workspace/tools/verify_installed_package_layers.py
COPY tools/build_legacy_monolith_fixture.py /workspace/tools/build_legacy_monolith_fixture.py
# Fail fast if Git LFS content wasn't materialized before the build — otherwise
# the wheel ships pointer stubs in place of perf data (see DYN-2725).
RUN set -e; \
    bad=$(grep -rlE "^version https://git-lfs" /workspace/aic-core/src/aiconfigurator_core/systems 2>/dev/null || true); \
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
        uv build --project aic-core --wheel --out-dir /workspace/dist && \
    cd /workspace && uv build --wheel --out-dir /workspace/dist && \
    cd /workspace && "${UV_PYTHON}" tools/verify_release_wheels.py /workspace/dist

# Exercise the real wheel boundary outside the checkout: core-only (including
# the aiconfigurator_core.sdk import contract), full install, upper uninstall
# (core survives), and core uninstall (upper files survive).
RUN uv venv --python 3.12 /tmp/aic-wheel-test && \
    CORE_WHL=$(ls /workspace/dist/aiconfigurator_core-*.whl) && \
    AIC_WHL=$(ls /workspace/dist/aiconfigurator-*.whl) && \
    uv pip install --python /tmp/aic-wheel-test/bin/python "${CORE_WHL}" && \
    cd /tmp && /tmp/aic-wheel-test/bin/python \
        /workspace/tools/verify_installed_package_layers.py --expect core --exercise-engine && \
    uv pip install --python /tmp/aic-wheel-test/bin/python --find-links /workspace/dist "${AIC_WHL}" && \
    uv pip check --python /tmp/aic-wheel-test/bin/python && \
    cd /tmp && /tmp/aic-wheel-test/bin/python \
        /workspace/tools/verify_installed_package_layers.py --expect full && \
    uv pip uninstall --python /tmp/aic-wheel-test/bin/python aiconfigurator && \
    cd /tmp && /tmp/aic-wheel-test/bin/python \
        /workspace/tools/verify_installed_package_layers.py --expect core && \
    uv pip install --python /tmp/aic-wheel-test/bin/python --find-links /workspace/dist "${AIC_WHL}" && \
    uv pip uninstall --python /tmp/aic-wheel-test/bin/python aiconfigurator-core && \
    cd /tmp && /tmp/aic-wheel-test/bin/python \
        /workspace/tools/verify_installed_package_layers.py --expect upper

# Start from another empty environment and install only the upper wheel argument.
# Dependency resolution must pull the matching core wheel from the wheelhouse,
# after which the canonical aiconfigurator.sdk import contract must work.
RUN uv venv --python 3.12 /tmp/aic-upper-install-test && \
    AIC_WHL=$(ls /workspace/dist/aiconfigurator-*.whl) && \
    uv pip install --python /tmp/aic-upper-install-test/bin/python \
        --find-links /workspace/dist "${AIC_WHL}" && \
    uv pip check --python /tmp/aic-upper-install-test/bin/python && \
    cd /tmp && /tmp/aic-upper-install-test/bin/python \
        /workspace/tools/verify_installed_package_layers.py --expect full

# Exercise the supported clean migration from the historical monolithic 0.9
# wheel. The old distribution owns `aiconfigurator_core/__init__.py`, so it must
# be uninstalled before the split wheels are installed; otherwise pip can remove
# the newly installed core file while uninstalling the monolith.
RUN "${UV_PYTHON}" tools/build_legacy_monolith_fixture.py /tmp/legacy-dist && \
    LEGACY_WHL=$(ls /tmp/legacy-dist/aiconfigurator-0.9.0-*.whl) && \
    CORE_WHL=$(ls /workspace/dist/aiconfigurator_core-*.whl) && \
    AIC_WHL=$(ls /workspace/dist/aiconfigurator-*.whl) && \
    "${UV_PYTHON}" -m venv /tmp/aic-pip-migration && \
    /tmp/aic-pip-migration/bin/pip install "${LEGACY_WHL}" && \
    /tmp/aic-pip-migration/bin/pip uninstall -y aiconfigurator aiconfigurator-core && \
    /tmp/aic-pip-migration/bin/pip install "${CORE_WHL}" "${AIC_WHL}" && \
    cd /tmp && /tmp/aic-pip-migration/bin/python \
        /workspace/tools/verify_installed_package_layers.py --expect full --exercise-engine

FROM base AS runtime
COPY --from=build /workspace/dist /wheelhouse
RUN uv pip install --find-links /wheelhouse /wheelhouse/aiconfigurator-*.whl && \
    rm -rf /wheelhouse

FROM runtime AS test

# The perf-database tool tests shell out to git (changed_ops revision diffing,
# migrate_* execute paths use git mv/rm/add). Test stage only — the shipped
# runtime image stays git-free.
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Runtime already installed both payload-owning wheels. Add the upper wheel's
# development extras before copying the test sources.
COPY --from=build /workspace/dist /wheelhouse
RUN AIC_WHL=$(ls /wheelhouse/aiconfigurator-*.whl) && \
    uv pip install --find-links /wheelhouse "${AIC_WHL}[dev]" && \
    rm -rf /wheelhouse

COPY pytest.ini /workspace/
COPY tests/ /workspace/tests/
COPY src/ /workspace/src/
COPY aic-core/ /workspace/aic-core/
COPY tools/ /workspace/tools/
COPY collector/ /workspace/collector/
