
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
# The standalone `aiconfigurator-core` wheel owns the Rust extension and the
# SDK/data payload. The upper `aiconfigurator` wheel owns CLI/generator/webapp/
# Spica and depends on the exact matching core version. Maturin needs the Rust
# crate sources, a C linker (gcc), and ca-certificates so cargo can fetch crates
# from crates.io over TLS. Rust itself is bootstrapped by maturin's build backend.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/*
COPY src/ /workspace/src/
COPY rust/ /workspace/rust/
COPY LICENSE pyproject.toml README.md /workspace/
COPY tools/verify_release_wheels.py /workspace/tools/verify_release_wheels.py
COPY tools/verify_installed_package_layers.py /workspace/tools/verify_installed_package_layers.py
COPY tools/build_legacy_monolith_fixture.py /workspace/tools/build_legacy_monolith_fixture.py
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
RUN cd src/aiconfigurator-core && uv build --wheel --out-dir /workspace/dist && \
    cd /workspace && uv build --wheel --out-dir /workspace/dist && \
    cd /workspace && python tools/verify_release_wheels.py /workspace/dist

# Exercise the real wheel boundary outside the checkout: core-only, full install,
# upper uninstall (core survives), and core uninstall (upper files survive).
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

# Pip installs dependencies before dependents, so crossing from a monolithic
# 0.9 wheel requires removing the old payload owner first. A minimal local 0.9
# fixture records representative upper/core paths and makes this ownership
# transfer deterministic without relying on the historical wheel bootstrap.
RUN python tools/build_legacy_monolith_fixture.py /tmp/legacy-dist && \
    LEGACY_WHL=$(ls /tmp/legacy-dist/aiconfigurator-0.9.0-*.whl) && \
    CORE_WHL=$(ls /workspace/dist/aiconfigurator_core-*.whl) && \
    AIC_WHL=$(ls /workspace/dist/aiconfigurator-*.whl) && \
    python -m venv /tmp/aic-pip-migration && \
    /tmp/aic-pip-migration/bin/pip install "${LEGACY_WHL}" && \
    /tmp/aic-pip-migration/bin/pip uninstall -y aiconfigurator aiconfigurator-core && \
    /tmp/aic-pip-migration/bin/pip install "${CORE_WHL}" "${AIC_WHL}" && \
    cd /tmp && /tmp/aic-pip-migration/bin/python \
        /workspace/tools/verify_installed_package_layers.py --expect full --exercise-engine && \
    python -m venv /tmp/aic-pip-repair && \
    /tmp/aic-pip-repair/bin/pip install "${LEGACY_WHL}" && \
    /tmp/aic-pip-repair/bin/pip install --upgrade "${CORE_WHL}" "${AIC_WHL}" && \
    if cd /tmp && /tmp/aic-pip-repair/bin/python \
        /workspace/tools/verify_installed_package_layers.py --expect full; then \
        echo "ERROR: naive monolith-to-split pip upgrade unexpectedly preserved core files" >&2; \
        exit 1; \
    fi && \
    /tmp/aic-pip-repair/bin/pip install --force-reinstall --no-deps "${CORE_WHL}" && \
    cd /tmp && /tmp/aic-pip-repair/bin/python \
        /workspace/tools/verify_installed_package_layers.py --expect full --exercise-engine

FROM base AS runtime
COPY --from=build /workspace/dist /wheelhouse
RUN uv pip install /wheelhouse/*.whl && \
    rm -rf /wheelhouse

FROM runtime AS test

# Runtime already installed both payload-owning wheels. Add the upper wheel's
# development extras before copying the test sources.
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
