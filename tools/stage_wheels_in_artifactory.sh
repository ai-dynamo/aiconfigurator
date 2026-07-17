#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${ARTIFACTORY_URL:?ARTIFACTORY_URL is not configured}"
: "${ARTIFACTORY_TOKEN:?ARTIFACTORY_TOKEN is not configured}"
: "${ARTIFACTORY_PYPI_REPO_NAME:?ARTIFACTORY_PYPI_REPO_NAME is not configured}"
: "${ARTIFACTORY_SUBPATH:?ARTIFACTORY_SUBPATH is not configured}"
: "${UPLOAD_UPPER:?UPLOAD_UPPER is not configured}"

if [[ "${ARTIFACTORY_SUBPATH}" == /* || "${ARTIFACTORY_SUBPATH}" == *..* ]]; then
    echo "::error::ARTIFACTORY_SUBPATH must be a relative path without parent traversal"
    exit 1
fi

shopt -s nullglob
core_wheels=(wheelhouse/aiconfigurator_core-*.whl)
if [[ "${#core_wheels[@]}" -ne 1 ]]; then
    echo "::error::Expected exactly one aiconfigurator-core wheel, found ${#core_wheels[@]}"
    exit 1
fi

wheels=("${core_wheels[0]}")
if [[ "${UPLOAD_UPPER}" == "true" ]]; then
    upper_wheels=(wheelhouse/aiconfigurator-*.whl)
    if [[ "${#upper_wheels[@]}" -ne 1 ]]; then
        echo "::error::Expected exactly one aiconfigurator wheel, found ${#upper_wheels[@]}"
        exit 1
    fi
    wheels+=("${upper_wheels[0]}")
elif [[ "${UPLOAD_UPPER}" != "false" ]]; then
    echo "::error::UPLOAD_UPPER must be true or false"
    exit 1
fi

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
        echo "### Artifactory wheels"
        echo
        echo "- Path: \`${ARTIFACTORY_SUBPATH}/\`"
    } >> "${GITHUB_STEP_SUMMARY}"
fi

for wheel in "${wheels[@]}"; do
    filename="$(basename "${wheel}")"
    target="${ARTIFACTORY_URL%/}/${ARTIFACTORY_PYPI_REPO_NAME}/${ARTIFACTORY_SUBPATH}/${filename}"
    curl --fail-with-body --show-error --silent \
        --connect-timeout 30 --max-time 900 \
        --retry 3 --retry-delay 5 --retry-all-errors \
        --header "Authorization: Bearer ${ARTIFACTORY_TOKEN}" \
        --upload-file "${wheel}" \
        "${target}"
    echo "Uploaded ${filename} to ${ARTIFACTORY_SUBPATH}/"
done
