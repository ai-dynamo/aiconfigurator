
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
# Pinned binary cargo tools (currently cargo-cyclonedx). Installed once here
# so every downstream stage picks them up via `COPY --from=rust-toolchain`.
COPY tools/wheel_build/install_build_tools.sh /tmp/install_build_tools.sh
RUN bash /tmp/install_build_tools.sh && rm /tmp/install_build_tools.sh

# Wheel-build stage: manylinux_2_28 (AlmaLinux 8, glibc 2.28). Building here
# means the cdylib references at most glibc 2.28 symbols, so `auditwheel
# repair` can stamp the wheel `manylinux_2_28_<arch>` for broad compatibility
# with RHEL 8 / Ubuntu 20.04 / Debian 11 hosts.
FROM quay.io/pypa/manylinux_2_28_${MANYLINUX_ARCH} AS build
COPY --from=rust-toolchain /root/.cargo /root/.cargo
COPY --from=rust-toolchain /root/.rustup /root/.rustup
ENV CARGO_HOME=/root/.cargo
ENV RUSTUP_HOME=/root/.rustup
ENV PATH="${CARGO_HOME}/bin:/opt/python/cp312-cp312/bin:${PATH}"
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace
ENV VIRTUAL_ENV=/opt/aiconfigurator/venv
RUN uv venv --python 3.12 ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV MPLCONFIGDIR=/tmp/matplotlib

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

# `auditwheel repair` rewrites the wheel's platform tag honestly based on the
# actual symbol versions it observes (glibc, libstdc++ etc.) and bundles any
# external shared-library deps inside the wheel. Lets auditwheel choose the
# tag rather than hard-coding it — that way a future glibc-symbol bump from
# Rust's std surfaces in the tag instead of silently breaking installs.
RUN mkdir -p /workspace/dist-repaired && \
    for whl in /workspace/dist/*.whl; do \
      auditwheel repair --wheel-dir /workspace/dist-repaired "$whl" && \
      rm "$whl"; \
    done && \
    mv /workspace/dist-repaired/*.whl /workspace/dist/ && \
    rmdir /workspace/dist-repaired

# Retag the wheel as `py3-none-<plat>`. The aiconfigurator-core cdylib is
# loaded via ctypes and never links libpython, so a single platform wheel
# covers all supported CPython versions (>= 3.10 per pyproject.toml).
RUN uv pip install 'wheel>=0.42'
RUN for whl in /workspace/dist/*.whl; do \
      python3 -m wheel tags --python-tag py3 --abi-tag none --remove "$whl"; \
    done

# Generate a CycloneDX SBOM for the Rust crate and inject it into every wheel
# under /workspace/dist per PEP 770.
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
