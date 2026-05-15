
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Pass --build-arg MANYLINUX_ARCH=x86_64 or aarch64 to match the docker buildx
# --platform target. CI sets this from matrix.archs (which already uses the
# x86_64 / aarch64 names manylinux uses). Default keeps single-platform local
# builds on x86_64 hosts working without an extra flag.
ARG MANYLINUX_ARCH=x86_64

# Rust toolchain stage, sourced from the same manylinux_2_28 image so that
# rustup-installed cargo/rustc and any compiled cargo binaries run inside the
# downstream `build` stage (matching glibc 2.28).
FROM quay.io/pypa/manylinux_2_28_${MANYLINUX_ARCH} AS rust-toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
      sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"
# cargo-cyclonedx is required by maturin's built-in PEP 770 SBOM emission —
# maturin invokes it as a subprocess and embeds the result at
# <dist-info>/sboms/<crate-name>-py3.cyclonedx.json. Pinned here so a single
# COPY --from=rust-toolchain pulls cargo, rustc, and the SBOM tool together.
RUN cargo install --locked cargo-cyclonedx --version 0.5.7

# Wheel-build stage: manylinux_2_28 (AlmaLinux 8, glibc 2.28). Building here
# means the cdylib references at most glibc 2.28 symbols, so `auditwheel
# repair` can stamp the wheel `manylinux_2_28_<arch>` for broad compatibility
# with RHEL 8 / Ubuntu 20.04 / Debian 11 hosts.
FROM quay.io/pypa/manylinux_2_28_${MANYLINUX_ARCH} AS build
COPY --from=rust-toolchain /root/.cargo /root/.cargo
COPY --from=rust-toolchain /root/.rustup /root/.rustup
ENV CARGO_HOME=/root/.cargo
ENV RUSTUP_HOME=/root/.rustup
# Pick cp310 as the build interpreter so maturin's abi3-py310 tag is honest
# (the wheel works on cp310+, but maturin needs an interpreter ≥ 3.10 to
# introspect during build).
ENV PATH="${CARGO_HOME}/bin:/opt/python/cp310-cp310/bin:${PATH}"
RUN pip install --no-cache-dir 'maturin>=1.10,<2'

WORKDIR /workspace
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

# Build the wheel via maturin. Maturin produces a cp310-abi3-linux_<arch>
# wheel and, because cargo-cyclonedx is on PATH, embeds the CycloneDX SBOM
# at <dist-info>/sboms/aiconfigurator-core-py3.cyclonedx.json — no separate
# cargo cyclonedx step, no custom SBOM injector needed.
RUN maturin build --release --out /workspace/raw-dist

# auditwheel repair rewrites the platform tag honestly based on observed
# symbol versions (linux_<arch> -> manylinux_2_28_<arch>) and bundles any
# external shared-library deps inside the wheel.
RUN mkdir -p /workspace/dist && \
    for whl in /workspace/raw-dist/*.whl; do \
      auditwheel repair --wheel-dir /workspace/dist "$whl"; \
    done

# Single-purpose stage that contains only the built wheels. Use this target
# with `docker buildx build --target wheel-out --output type=local,dest=./dist`
# to copy wheels onto the host without exporting the whole build-stage rootfs.
FROM scratch AS wheel-out
COPY --from=build /workspace/dist/ /

# Runtime / test stages keep the NVIDIA Ubuntu base — only the wheel-build
# pipeline cares about the manylinux baseline.
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
