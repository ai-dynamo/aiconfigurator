#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Installs binary cargo tools required by the wheel-build pipeline.
# Pinned so CI, Docker, and local developer builds agree on the SBOM tool
# version. Bumping is a one-line PR — verify the produced bom.cdx.json
# still parses and the component set is sensible before merging.

set -euo pipefail

CARGO_CYCLONEDX_VERSION="0.5.7"

require_cargo() {
    if ! command -v cargo >/dev/null 2>&1; then
        echo "install_build_tools.sh: cargo is not on PATH; install Rust via https://rustup.rs/ first" >&2
        exit 1
    fi
}

want_version() {
    local tool=$1 expected=$2
    command -v "$tool" >/dev/null 2>&1 \
        && "$tool" --version 2>/dev/null | grep -q "$expected"
}

require_cargo

if want_version cargo-cyclonedx "$CARGO_CYCLONEDX_VERSION"; then
    echo "cargo-cyclonedx $CARGO_CYCLONEDX_VERSION already installed"
else
    echo "Installing cargo-cyclonedx $CARGO_CYCLONEDX_VERSION..."
    cargo install --locked --force \
        cargo-cyclonedx --version "$CARGO_CYCLONEDX_VERSION"
fi
