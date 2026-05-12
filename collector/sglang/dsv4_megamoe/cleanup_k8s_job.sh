#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

JOB_NAME="${1:?usage: cleanup_k8s_job.sh JOB_NAME NAMESPACE}"
NAMESPACE="${2:?usage: cleanup_k8s_job.sh JOB_NAME NAMESPACE}"

kubectl -n "${NAMESPACE}" delete job "${JOB_NAME}" --ignore-not-found=true
kubectl -n "${NAMESPACE}" delete service "${JOB_NAME}" --ignore-not-found=true
kubectl -n "${NAMESPACE}" delete pod -l "app=${JOB_NAME},aic.nvidia.com/collector=dsv4-megamoe" --ignore-not-found=true
kubectl -n "${NAMESPACE}" delete computedomains.resource.nvidia.com "${JOB_NAME}-compute-domain" --ignore-not-found=true
kubectl -n "${NAMESPACE}" delete resourceclaimtemplates.resource.k8s.io "${JOB_NAME}-compute-domain-channel" --ignore-not-found=true

echo "Remaining owned resources:"
kubectl -n "${NAMESPACE}" get \
  "job/${JOB_NAME}" \
  "service/${JOB_NAME}" \
  "computedomains.resource.nvidia.com/${JOB_NAME}-compute-domain" \
  "resourceclaimtemplates.resource.k8s.io/${JOB_NAME}-compute-domain-channel" \
  --ignore-not-found=true || true
kubectl -n "${NAMESPACE}" get job,svc,pod,computedomains.resource.nvidia.com -l \
  "app=${JOB_NAME},aic.nvidia.com/collector=dsv4-megamoe" || true
