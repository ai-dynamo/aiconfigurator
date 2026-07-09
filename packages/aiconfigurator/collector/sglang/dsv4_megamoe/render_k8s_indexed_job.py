#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a Kubernetes Indexed Job for DSv4 MegaMoE collection.

The script only renders YAML.  Applying and cleanup stay explicit so cluster
resources are not created accidentally.
"""

from __future__ import annotations

import argparse
import sys
from textwrap import indent

DEFAULT_GPUS_PER_NODE = {
    "B200": 8,
    "GB200": 4,
    "B300": 8,
    "GB300": 4,
}

DEFAULT_IMAGES = {
    "B200": "lmsysorg/sglang:deepseek-v4-blackwell",
    "GB200": "lmsysorg/sglang:deepseek-v4-grace-blackwell",
    "B300": "lmsysorg/sglang:deepseek-v4-b300",
    "GB300": "lmsysorg/sglang:deepseek-v4-grace-blackwell",
}


def _default_gpus_per_node(system_name: str) -> int:
    try:
        return DEFAULT_GPUS_PER_NODE[system_name.upper()]
    except KeyError as exc:
        raise SystemExit(f"--gpus-per-node is required for unknown system {system_name}") from exc


def _default_image(system_name: str) -> str:
    try:
        return DEFAULT_IMAGES[system_name.upper()]
    except KeyError as exc:
        raise SystemExit(f"--image is required for unknown system {system_name}") from exc


def _env(name: str, value: str) -> str:
    return f"        - name: {name}\n          value: {value!r}"


def _parse_extra_env(values: list[str]) -> list[tuple[str, str]]:
    envs = []
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--env must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise SystemExit(f"--env key must not be empty, got {item!r}")
        envs.append((key, value))
    return envs


def _parse_node_selector(value: str) -> list[tuple[str, str]]:
    pairs = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"--node-selector entries must be KEY=VALUE, got {item!r}")
        key, selector_value = item.split("=", 1)
        key = key.strip()
        selector_value = selector_value.strip()
        if not key or not selector_value:
            raise SystemExit(f"--node-selector entries must be non-empty KEY=VALUE pairs, got {item!r}")
        pairs.append((key, selector_value))
    return pairs


def render(args: argparse.Namespace) -> str:
    gpus_per_node = args.gpus_per_node or _default_gpus_per_node(args.system_name)
    image = args.image or _default_image(args.system_name)
    if args.ep_size % gpus_per_node != 0:
        raise SystemExit("--ep-size must be divisible by --gpus-per-node")
    nnodes = args.ep_size // gpus_per_node
    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    master_addr = f"{args.job_name}-0.{args.job_name}.{args.namespace}.svc.cluster.local"
    compute_domain_name = args.compute_domain_name or f"{args.job_name}-compute-domain"
    compute_domain_channel = args.compute_domain_channel or f"{args.job_name}-compute-domain-channel"
    claim_name = args.compute_domain_claim_name
    labels = {
        "app": args.job_name,
        "aic.nvidia.com/collector": "dsv4-megamoe",
    }
    labels_yaml = "\n".join(f"    {key}: {value}" for key, value in labels.items())
    selector_yaml = "\n".join(f"    {key}: {value}" for key, value in labels.items())
    compute_domain = ""
    pod_resource_claims = ""
    container_claims = ""
    if args.compute_domain:
        compute_domain = f"""apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata:
  name: {compute_domain_name}
  namespace: {args.namespace}
  labels:
{labels_yaml}
spec:
  numNodes: {args.compute_domain_num_nodes or nnodes}
  channel:
    allocationMode: {args.compute_domain_allocation_mode}
    resourceClaimTemplate:
      name: {compute_domain_channel}
---
"""
        pod_resource_claims = f"""
      resourceClaims:
      - name: {claim_name}
        resourceClaimTemplateName: {compute_domain_channel}"""
        container_claims = f"""
          claims:
          - name: {claim_name}"""
    env_entries = [
        _env("SYSTEM_NAME", args.system_name),
        _env("GPUS_PER_NODE", str(gpus_per_node)),
        _env("EP_SIZE", str(args.ep_size)),
        _env("NNODES", str(nnodes)),
        _env("MASTER_ADDR", master_addr),
        _env("MASTER_PORT", str(args.master_port)),
        _env("MODEL_CONFIG", args.model_config),
        _env("OUTPUT_PATH", args.output_path),
        _env("PERF_FILE", args.perf_file),
        _env("DISTRIBUTIONS", args.distributions),
        _env("SOURCE_POLICY", args.source_policy),
        _env("ROUTING_SEED", str(args.routing_seed)),
        _env("ROUTING_SEEDS", args.routing_seeds),
        _env("PHASES", args.phases),
        _env("PREFILL_TOKENS", args.prefill_tokens),
        _env("DECODE_TOKENS", args.decode_tokens),
        _env("PRE_DISPATCH", args.pre_dispatch),
        _env("INCLUDE_ROUTED_SCALE", str(args.include_routed_scale)),
        _env("RENORMALIZE_TOPK_WEIGHTS", str(args.renormalize_topk_weights)),
        _env("NUM_WARMUP", str(args.num_warmup)),
        _env("NUM_ITERATIONS", str(args.num_iterations)),
        _env("NUM_MAX_TOKENS_PER_RANK", str(num_max_tokens_per_rank)),
        "\n".join(
            [
                "        - name: NODE_RANK",
                "          valueFrom:",
                "            fieldRef:",
                "              fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']",
            ]
        ),
    ]
    if args.system_name.upper() == "GB200" and nnodes > 1:
        env_entries.extend(
            [
                _env("NCCL_MNNVL_ENABLE", "1"),
                _env("NCCL_CUMEM_ENABLE", "1"),
            ]
        )
    for key, value in _parse_extra_env(args.env):
        env_entries = [entry for entry in env_entries if not entry.startswith(f"        - name: {key}\n")]
        env_entries.append(_env(key, value))
    env_yaml = "\n".join(env_entries)
    volume_mounts = ""
    volumes = ""
    if args.pvc_name:
        volume_mounts = f"""
        volumeMounts:
        - name: aic-workspace
          mountPath: {args.pvc_mount!r}"""
        volumes = f"""
      volumes:
      - name: aic-workspace
        persistentVolumeClaim:
          claimName: {args.pvc_name}"""
    node_selector = ""
    if args.node_selector:
        pairs = _parse_node_selector(args.node_selector)
        node_selector = "\n      nodeSelector:\n" + "\n".join(f"        {key}: {value}" for key, value in pairs)

    tolerations = ""
    if args.toleration_key:
        entries = "\n".join(
            f"      - key: {key!r}\n        operator: Exists\n        effect: NoSchedule" for key in args.toleration_key
        )
        tolerations = f"""
      tolerations:
{entries}"""

    priority_class = ""
    if args.priority_class_name:
        priority_class = f"""
      priorityClassName: {args.priority_class_name}"""

    security_context = ""
    if args.ipc_lock:
        security_context = """
        securityContext:
          capabilities:
            add: ["IPC_LOCK"]"""

    command = f"""
set -euo pipefail
cd {args.working_dir}
bash packages/aiconfigurator/collector/sglang/dsv4_megamoe/run_torchrun.sh
"""

    return f"""{compute_domain}apiVersion: v1
kind: Service
metadata:
  name: {args.job_name}
  namespace: {args.namespace}
  labels:
{labels_yaml}
spec:
  clusterIP: None
  selector:
{selector_yaml}
---
apiVersion: batch/v1
kind: Job
metadata:
  name: {args.job_name}
  namespace: {args.namespace}
  labels:
{labels_yaml}
spec:
  completions: {nnodes}
  parallelism: {nnodes}
  completionMode: Indexed
  backoffLimit: 0
  template:
    metadata:
      labels:
{indent(labels_yaml, "      ")}
    spec:
      restartPolicy: Never
      subdomain: {args.job_name}
{priority_class}
{pod_resource_claims}
      containers:
      - name: collector
        image: {image}
        imagePullPolicy: {args.image_pull_policy}
        workingDir: {args.working_dir}
{security_context}
        env:
{env_yaml}
        command: ["/bin/bash", "-lc"]
        args:
        - |
{indent(command.rstrip(), "          ")}
        resources:
{container_claims}
          limits:
            nvidia.com/gpu: {gpus_per_node}
          requests:
            nvidia.com/gpu: {gpus_per_node}{volume_mounts}{volumes}{node_selector}{tolerations}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--image-pull-policy", default="IfNotPresent")
    parser.add_argument("--system-name", default="gb200")
    parser.add_argument("--gpus-per-node", type=int, default=None)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--model-config", default="dsv4_pro")
    parser.add_argument("--working-dir", default="/workspace/aiconfigurator")
    parser.add_argument(
        "--output-path",
        default="/workspace/aiconfigurator/packages/aiconfigurator/collector/sglang/dsv4_megamoe/results",
    )
    parser.add_argument("--perf-file", default="dsv4_megamoe_module_perf.txt")
    parser.add_argument(
        "--distributions",
        default="balanced,power_law_1.01,power_law_1.2,power_law_sampled_1.9",
    )
    parser.add_argument("--source-policy", choices=["random"], default="random")
    parser.add_argument("--routing-seed", type=int, default=0)
    parser.add_argument(
        "--routing-seeds",
        default="",
        help="optional comma-separated routing seeds; empty lets collection expand power_law_sampled_1.9 to 10 seeds",
    )
    parser.add_argument("--phases", default="context,generation")
    parser.add_argument("--prefill-tokens", default="1024,2048,4096,8192,16384,32768")
    parser.add_argument("--decode-tokens", default="1,2,4,8,16,32,64,128,256,512")
    parser.add_argument("--pre-dispatch", choices=["sglang_jit", "copy"], default="sglang_jit")
    parser.add_argument("--include-routed-scale", type=int, choices=[0, 1], default=1)
    parser.add_argument("--renormalize-topk-weights", type=int, choices=[0, 1], default=1)
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-iterations", type=int, default=20)
    parser.add_argument("--num-max-tokens-per-rank", type=int, required=True)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--priority-class-name", default="")
    parser.add_argument("--compute-domain", action="store_true")
    parser.add_argument("--compute-domain-name", default="")
    parser.add_argument("--compute-domain-channel", default="")
    parser.add_argument("--compute-domain-claim-name", default="compute-domain-channel")
    parser.add_argument("--compute-domain-num-nodes", type=int, default=0)
    parser.add_argument("--compute-domain-allocation-mode", choices=["Single", "All"], default="Single")
    parser.add_argument("--ipc-lock", type=int, choices=[0, 1], default=0)
    parser.add_argument("--pvc-name", default="")
    parser.add_argument("--pvc-mount", default="/mnt/shared")
    parser.add_argument("--node-selector", default="")
    parser.add_argument("--toleration-key", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    sys.stdout.write(render(parse_args()))


if __name__ == "__main__":
    main()
