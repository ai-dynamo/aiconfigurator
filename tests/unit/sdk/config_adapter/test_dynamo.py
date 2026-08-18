# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aiconfigurator.sdk.config_adapter import (
    AdapterOverrides,
    DynamoRecipeSource,
    WorkloadPointOverride,
    adapt_config,
    to_cli_estimate_kwargs,
)

pytestmark = pytest.mark.unit

_NIGHTLY_FIXTURES = Path(__file__).parents[3] / "fixtures" / "config_adapter" / "dynamo_nightly"


def _dynamo_ci_recipe() -> dict:
    return {
        "name": "gb200-fp4-test",
        "model": {"path": "dsfp4", "container": "0.5.5.post2", "precision": "fp4"},
        "resources": {
            "gpu_type": "gb200",
            "prefill_nodes": 1,
            "prefill_workers": 1,
            "decode_nodes": 2,
            "decode_workers": 2,
            "gpus_per_node": 4,
        },
        "backend": {
            "sglang_config": {
                "prefill": {
                    "served-model-name": "deepseek-ai/DeepSeek-R1",
                    "tp-size": 4,
                    "dp-size": 1,
                    "ep-size": 1,
                    "kv-cache-dtype": "fp8_e4m3",
                    "quantization": "modelopt_fp4",
                    "context-length": 8192,
                    "mem-fraction-static": 0.9,
                },
                "decode": {
                    "served-model-name": "deepseek-ai/DeepSeek-R1",
                    "tp-size": 4,
                    "dp-size": 4,
                    "ep-size": 4,
                    "enable-dp-attention": True,
                    "kv-cache-dtype": "fp8_e4m3",
                    "quantization": "modelopt_fp4",
                    "context-length": 8192,
                    "mem-fraction-static": 0.9,
                },
            }
        },
        "benchmark": {"isl": 1024, "osl": 256, "concurrencies": "8x10"},
    }


def _agg_yaml(backend: str = "vllm", *, args_as_string: bool = False, extra: str = "") -> str:
    module = backend
    model_flag = "--model-path" if backend in {"sglang", "trtllm"} else "--model"
    tp_flag = "--tp" if backend == "sglang" else "--tensor-parallel-size"
    args = f"{model_flag} $MODEL_PATH {tp_flag} 2 --gpu-memory-utilization 0.9"
    if args_as_string:
        rendered_args = f'          args: ["python3 -m dynamo.{module} {args}"]'
        rendered_command = ""
    else:
        rendered_command = f"""          command: [python3, -m, dynamo.{module}]
"""
        rendered_args = f'''          args:
            - {model_flag}
            - $MODEL_PATH
            - {tp_flag}
            - "2"
            - --gpu-memory-utilization
            - "0.9"'''
    return f"""
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: test-{backend}
spec:
  backendFramework: {backend}
  services:
    worker:
      componentType: worker
      replicas: 1
      resources:
        limits:
          gpu: "2"
      extraPodSpec:
        nodeSelector:
          nvidia.com/gpu.product: NVIDIA-H200-SXM
        mainContainer:
          env:
            - name: MODEL_PATH
              value: QWEN/QWEN3-32B
{rendered_command}{rendered_args}
{extra}
"""


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
@pytest.mark.parametrize("args_as_string", [False, True])
def test_agg_backends_support_string_and_list_arguments(backend, args_as_string):
    report = adapt_config(
        DynamoRecipeSource(_agg_yaml(backend, args_as_string=args_as_string)),
        AdapterOverrides(isl=1024, osl=128, concurrency=16),
    )
    request = report.requests[0]

    assert request.backend.name == backend
    assert request.model.path == "QWEN/QWEN3-32B"
    assert request.systems.prefill == "h200_sxm"
    assert request.topology.worker.gpus_per_replica == 2
    assert request.provenance.assumptions[-1] == (
        "Aggregated worker replicas are omitted during cli_estimate lowering because "
        "cli_estimate has no aggregated worker-count parameter."
    )


def test_override_only_speculation_requires_explicit_acceptance():
    outcome = adapt_config(
        DynamoRecipeSource(_agg_yaml("trtllm")),
        AdapterOverrides(isl=1024, osl=128, concurrency=16, nextn=1, nextn_accepted=None),
    ).outcomes[0]

    assert outcome.status == "rejected"
    assert outcome.diagnostics[-1].message == "speculative decoding requires an explicit nextn_accepted override"


def test_resolved_dsv4_shell_comments_do_not_change_engine_arguments():
    fixture = _NIGHTLY_FIXTURES / "dsv4"

    outcome = adapt_config(
        DynamoRecipeSource(fixture / "deployment.yaml", fixture / "benchmark-job.yaml"),
        AdapterOverrides(
            system_name="gb200",
            free_gpu_memory_fraction=0.9,
            workload_points=(WorkloadPointOverride(point_id="concurrency-8", isl=8192, osl=1024, concurrency=8),),
        ),
    ).outcomes[0]

    assert outcome.status == "adapted"
    assert outcome.request is not None
    assert outcome.request.topology.kind == "disagg"


def test_resolved_dsv4_sweep_preserves_points_and_role_memory_fractions():
    fixture = _NIGHTLY_FIXTURES / "dsv4"

    report = adapt_config(
        DynamoRecipeSource(fixture / "deployment.yaml", fixture / "benchmark-job.yaml"),
        AdapterOverrides(system_name="gb200"),
    )

    assert [outcome.point_id for outcome in report.outcomes] == [
        "concurrency-1",
        "concurrency-8",
        "concurrency-64",
        "concurrency-128",
        "concurrency-256",
        "concurrency-512",
        "concurrency-1024",
    ]
    assert report.outcomes[0].status == "rejected"
    assert "must be divisible" in report.outcomes[0].diagnostics[-1].message
    assert all(outcome.request is not None for outcome in report.outcomes[1:])
    for outcome in report.outcomes[1:]:
        assert outcome.request is not None
        assert outcome.request.runtime.prefill_free_gpu_memory_fraction == 0.8
        assert outcome.request.runtime.decode_free_gpu_memory_fraction == 0.9


def test_resolved_kimi_engine_config_requires_speculative_acceptance():
    fixture = _NIGHTLY_FIXTURES / "kimi"

    outcome = adapt_config(
        DynamoRecipeSource(fixture / "deployment.yaml", fixture / "benchmark-job.yaml"),
        AdapterOverrides(
            system_name="gb200",
            workload_points=(WorkloadPointOverride(point_id="concurrency-3", isl=1000, osl=1500, concurrency=3),),
        ),
    ).outcomes[0]

    assert outcome.status == "rejected"
    assert "nextn_accepted" in outcome.diagnostics[-1].message


def test_resolved_kimi_sweep_preserves_all_points_and_uneven_distribution_diagnostics():
    fixture = _NIGHTLY_FIXTURES / "kimi"

    report = adapt_config(
        DynamoRecipeSource(fixture / "deployment.yaml", fixture / "benchmark-job.yaml"),
        AdapterOverrides(system_name="gb200"),
    )

    assert [outcome.point_id for outcome in report.outcomes] == [
        "concurrency-1",
        "concurrency-2",
        "concurrency-4",
        "concurrency-8",
        "concurrency-16",
        "concurrency-32",
        "concurrency-64",
        "concurrency-128",
        "concurrency-256",
    ]
    assert all(outcome.status == "rejected" for outcome in report.outcomes)
    assert all("must be divisible" in outcome.diagnostics[-1].message for outcome in report.outcomes)


def test_resolved_kimi_engine_speculation_is_preserved_with_explicit_acceptance():
    fixture = _NIGHTLY_FIXTURES / "kimi"

    outcome = adapt_config(
        DynamoRecipeSource(fixture / "deployment.yaml", fixture / "benchmark-job.yaml"),
        AdapterOverrides(
            system_name="gb200",
            nextn_accepted=1.5,
            workload_points=(WorkloadPointOverride(point_id="concurrency-3", isl=1000, osl=1500, concurrency=3),),
        ),
    ).outcomes[0]

    assert outcome.status == "adapted"
    assert outcome.request is not None
    assert outcome.request.model.nextn == 3
    assert outcome.request.model.nextn_accepted == 1.5


def test_resolved_kimi_unmounted_engine_config_is_rejected():
    fixture = _NIGHTLY_FIXTURES / "kimi"
    documents = list(yaml.safe_load_all((fixture / "deployment.yaml").read_text()))
    deployment = next(document for document in documents if document.get("kind") == "DynamoGraphDeployment")
    del deployment["spec"]["services"]["agg"]["extraPodSpec"]["mainContainer"]["volumeMounts"]

    outcome = adapt_config(
        DynamoRecipeSource(documents, fixture / "benchmark-job.yaml"),
        AdapterOverrides(
            system_name="gb200",
            nextn_accepted=1.5,
            workload_points=(WorkloadPointOverride(point_id="concurrency-3", isl=1000, osl=1500, concurrency=3),),
        ),
    ).outcomes[0]

    assert outcome.status == "rejected"
    assert "does not resolve to a mounted ConfigMap file" in outcome.diagnostics[-1].message


def test_disagg_env_substitution_and_role_sizing():
    deployment = """
kind: DynamoGraphDeployment
metadata: {name: disagg}
spec:
  backendFramework: vllm
  services:
    prefill:
      componentType: worker
      subComponentType: prefill
      replicas: 2
      resources: {limits: {gpu: "2"}}
      extraPodSpec:
        nodeSelector: {nvidia.com/gpu.product: NVIDIA-H200}
        mainContainer:
          env: [{name: MODEL_PATH, value: QWEN/QWEN3-32B}]
          args:
            - >-
              python -m dynamo.vllm --model ${MODEL_PATH} --tensor-parallel-size 2
              --gpu-memory-utilization 0.85 --disaggregation-mode prefill
    decode:
      componentType: worker
      subComponentType: decode
      replicas: 1
      resources: {limits: {gpu: "4"}}
      extraPodSpec:
        nodeSelector: {nvidia.com/gpu.product: NVIDIA-H200}
        mainContainer:
          env: [{name: MODEL_PATH, value: QWEN/QWEN3-32B}]
          args:
            - >-
              python -m dynamo.vllm --model ${MODEL_PATH} --tensor-parallel-size 4
              --gpu-memory-utilization 0.7 --disaggregation-mode decode
"""
    report = adapt_config(
        DynamoRecipeSource(deployment),
        AdapterOverrides(isl=1024, osl=128, concurrency=16),
    )
    kwargs = to_cli_estimate_kwargs(report.requests[0])

    assert kwargs["prefill_num_workers"] == 2
    assert kwargs["prefill_batch_size"] == 1
    assert kwargs["decode_num_workers"] == 1
    assert kwargs["decode_batch_size"] == 16
    assert kwargs["prefill_free_gpu_memory_fraction"] == 0.85
    assert kwargs["decode_free_gpu_memory_fraction"] == 0.7


def test_global_env_uses_served_model_name_for_local_model_path():
    deployment = """
kind: DynamoGraphDeployment
metadata: {name: nemotron-ultra}
spec:
  backendFramework: vllm
  env:
    - {name: MODEL_PATH, value: /opt/models/patched/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4}
    - {name: SERVED_MODEL_NAME, value: nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4}
  services:
    worker:
      componentType: worker
      replicas: 1
      resources: {limits: {gpu: "2"}}
      extraPodSpec:
        nodeSelector: {nvidia.com/gpu.product: NVIDIA-B200}
        mainContainer:
          args:
            - python -m dynamo.vllm --model ${MODEL_PATH} --served-model-name ${SERVED_MODEL_NAME}
              --tensor-parallel-size 2
"""

    outcome = adapt_config(
        DynamoRecipeSource(deployment),
        AdapterOverrides(isl=1024, osl=128, concurrency=16),
    ).outcomes[0]

    assert outcome.request is not None
    assert outcome.request.model.path == "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"


def test_workload_concurrency_is_preserved_when_active_batch_is_capped():
    deployment = _agg_yaml(args_as_string=True).replace(
        "--gpu-memory-utilization 0.9",
        "--max-num-seqs 32 --gpu-memory-utilization 0.9",
    )

    outcome = adapt_config(
        DynamoRecipeSource(deployment),
        AdapterOverrides(isl=1024, osl=128, concurrency=1080),
    ).outcomes[0]

    assert outcome.request is not None
    assert outcome.request.workload.concurrency == 1080
    assert outcome.request.topology.worker.batch_size == 32
    assert "workload concurrency 1080 remains unchanged" in outcome.request.provenance.assumptions[-2]


def test_explicit_batch_override_above_engine_max_is_rejected():
    deployment = _agg_yaml(args_as_string=True).replace(
        "--gpu-memory-utilization 0.9",
        "--max-num-seqs 32 --gpu-memory-utilization 0.9",
    )

    outcome = adapt_config(
        DynamoRecipeSource(deployment),
        AdapterOverrides(isl=1024, osl=128, concurrency=1080, batch_size=33),
    ).outcomes[0]

    assert outcome.status == "rejected"
    assert outcome.diagnostics[-1].message == "active batch size 33 exceeds worker max batch size 32"


def test_multidocument_configmap_and_multinode_sizing():
    deployment = """
kind: ConfigMap
metadata: {name: engine}
data:
  config.yaml: |
    tensor_parallel_size: 8
    pipeline_parallel_size: 1
    max_batch_size: 32
---
kind: DynamoGraphDeployment
metadata: {name: multinode}
spec:
  backendFramework: trtllm
  services:
    worker:
      componentType: worker
      replicas: 1
      multinode: {nodeCount: 2}
      resources: {limits: {gpu: "4"}}
      extraPodSpec:
        nodeSelector: {nvidia.com/gpu.product: NVIDIA-B200}
        volumes: [{name: engine, configMap: {name: engine}}]
        mainContainer:
          volumeMounts: [{name: engine, mountPath: /configs}]
          env:
            - {name: MODEL_PATH, value: QWEN/QWEN3-32B}
            - {name: ENGINE_ARGS, value: /configs/config.yaml}
          args: ["python -m dynamo.trtllm --model-path $MODEL_PATH"]
"""
    outcome = adapt_config(
        DynamoRecipeSource(deployment),
        AdapterOverrides(isl=1024, osl=128, concurrency=16),
    ).outcomes[0]

    assert outcome.request is not None
    assert outcome.request.topology.worker.gpus_per_replica == 8
    assert outcome.request.topology.worker.tp_size == 8
    assert outcome.request.systems.prefill == "b200_sxm"


def test_gb200_product_does_not_alias_to_b200():
    deployment = _agg_yaml().replace("NVIDIA-H200-SXM", "NVIDIA-GB200-NVL72")
    outcome = adapt_config(
        DynamoRecipeSource(deployment),
        AdapterOverrides(isl=1024, osl=128, concurrency=16),
    ).outcomes[0]

    assert outcome.request is not None
    assert outcome.request.systems.prefill == "gb200"


def test_sglang_dp_attention_world_size_maps_to_attention_dp():
    deployment = _agg_yaml("sglang", args_as_string=True).replace(
        "--tp 2 --gpu-memory-utilization 0.9",
        "--tp 2 --dp-size 2 --enable-dp-attention --gpu-memory-utilization 0.9",
    )
    outcome = adapt_config(
        DynamoRecipeSource(deployment),
        AdapterOverrides(isl=1024, osl=128, concurrency=8),
    ).outcomes[0]

    assert outcome.request is not None
    assert outcome.request.topology.worker.tp_size == 1
    assert outcome.request.topology.worker.attention_dp_size == 2
    assert outcome.request.topology.worker.batch_size == 4


def test_sglang_runtime_flags_are_preserved():
    deployment = _agg_yaml("sglang", args_as_string=True).replace(
        "--gpu-memory-utilization 0.9",
        "--mem-fraction-static 0.73 --context-length 4096",
    )
    outcome = adapt_config(
        DynamoRecipeSource(deployment),
        AdapterOverrides(isl=1024, osl=128, concurrency=8),
    ).outcomes[0]

    assert outcome.request is not None
    assert outcome.request.runtime.free_gpu_memory_fraction == 0.73
    assert outcome.request.runtime.max_seq_len == 4096
    kwargs = to_cli_estimate_kwargs(outcome.request)
    assert kwargs["free_gpu_memory_fraction"] == 0.73
    assert kwargs["max_seq_len"] == 4096


def test_extra_engine_args_selects_one_file_from_multi_file_configmap():
    deployment = """
kind: ConfigMap
metadata: {name: engine}
data:
  selected.yaml: |
    tensor_parallel_size: 2
  other.yaml: |
    tensor_parallel_size: 4
---
kind: DynamoGraphDeployment
metadata: {name: selected-config}
spec:
  backendFramework: trtllm
  services:
    worker:
      componentType: worker
      resources: {limits: {gpu: "2"}}
      extraPodSpec:
        nodeSelector: {nvidia.com/gpu.product: NVIDIA-H200}
        volumes: [{name: engine, configMap: {name: engine}}]
        mainContainer:
          volumeMounts: [{name: engine, mountPath: /config}]
          args:
            - >-
              python -m dynamo.trtllm --model-path QWEN/QWEN3-32B
              --extra-engine-args /config/selected.yaml
"""
    outcome = adapt_config(
        DynamoRecipeSource(deployment),
        AdapterOverrides(isl=1024, osl=128, concurrency=8),
    ).outcomes[0]

    assert outcome.request is not None
    assert outcome.request.topology.worker.tp_size == 2


def test_configmap_filename_match_does_not_accept_suffixes():
    deployment = """
kind: ConfigMap
metadata: {name: engine}
data:
  selected.yaml: |
    tensor_parallel_size: 2
---
kind: DynamoGraphDeployment
metadata: {name: wrong-config}
spec:
  backendFramework: trtllm
  services:
    worker:
      componentType: worker
      resources: {limits: {gpu: "2"}}
      extraPodSpec:
        nodeSelector: {nvidia.com/gpu.product: NVIDIA-H200}
        volumes: [{name: engine, configMap: {name: engine}}]
        mainContainer:
          volumeMounts: [{name: engine, mountPath: /config}]
          args:
            - >-
              python -m dynamo.trtllm --model-path QWEN/QWEN3-32B
              --extra-engine-args /config/my-selected.yaml
"""
    outcome = adapt_config(
        DynamoRecipeSource(deployment),
        AdapterOverrides(isl=1024, osl=128, concurrency=8),
    ).outcomes[0]

    assert outcome.status == "rejected"
    assert "does not resolve to a mounted ConfigMap file" in outcome.diagnostics[-1].message


def test_non_integral_gpu_count_is_rejected():
    deployment = _agg_yaml().replace('gpu: "2"', "gpu: 2.9")
    outcome = adapt_config(
        DynamoRecipeSource(deployment),
        AdapterOverrides(isl=1024, osl=128, concurrency=8),
    ).outcomes[0]

    assert outcome.status == "rejected"
    assert "GPU limit must be an integer" in outcome.diagnostics[-1].message


def test_false_expert_parallel_flag_does_not_enable_ep():
    deployment = _agg_yaml(args_as_string=True).replace(
        "--gpu-memory-utilization 0.9",
        "--enable-expert-parallel false --gpu-memory-utilization 0.9",
    )
    outcome = adapt_config(
        DynamoRecipeSource(deployment),
        AdapterOverrides(isl=1024, osl=128, concurrency=8),
    ).outcomes[0]

    assert outcome.request is not None
    assert outcome.request.topology.worker.moe_ep_size == 1
    assert outcome.request.topology.worker.moe_tp_size == 2


def test_plain_string_is_yaml_text_not_an_implicit_file_path(tmp_path):
    deployment = tmp_path / "deployment.yaml"
    deployment.write_text(_agg_yaml())

    outcome = adapt_config(
        DynamoRecipeSource(str(deployment)),
        AdapterOverrides(isl=1024, osl=128, concurrency=8),
    ).outcomes[0]

    assert outcome.status == "rejected"
    assert "must be an object" in outcome.diagnostics[-1].message


def test_multiple_points_preserve_order_and_rejections():
    performance = {
        "points": [
            {"point_id": "valid", "isl": 1024, "osl": 128, "concurrency": 16},
            {"point_id": "invalid", "isl": 1024, "osl": 128, "concurrency": 3},
        ]
    }
    deployment = _agg_yaml(args_as_string=True).replace(
        "--tensor-parallel-size 2 --gpu-memory-utilization 0.9",
        "--tensor-parallel-size 1 --data-parallel-size 2 --gpu-memory-utilization 0.9",
    )
    report = adapt_config(DynamoRecipeSource(deployment, performance))

    assert [(outcome.point_id, outcome.status) for outcome in report.outcomes] == [
        ("valid", "adapted"),
        ("invalid", "rejected"),
    ]
    assert len(report.requests) == 1
    assert len(report.rejections) == 1


def test_workload_overrides_do_not_heal_malformed_explicit_point():
    performance = {
        "points": [
            {"point_id": "first", "isl": 1024, "osl": 128, "concurrency": 8},
            "malformed",
            {"point_id": "last", "isl": 2048, "osl": 256, "concurrency": 16},
        ]
    }

    report = adapt_config(
        DynamoRecipeSource(_agg_yaml(), performance),
        AdapterOverrides(isl=512, osl=64, concurrency=4),
    )

    assert [(outcome.point_id, outcome.status) for outcome in report.outcomes] == [
        ("first", "adapted"),
        ("point-1", "rejected"),
        ("last", "adapted"),
    ]
    assert "workload point 1 must be an object" in report.outcomes[1].diagnostics[-1].message


def test_explicit_workload_points_override_perf_points():
    overrides = AdapterOverrides(
        workload_points=(WorkloadPointOverride(point_id="override", isl=512, osl=64, concurrency=8),)
    )
    report = adapt_config(
        DynamoRecipeSource(_agg_yaml(), {"points": [{"point_id": "source", "isl": 1, "osl": 1, "concurrency": 1}]}),
        overrides,
    )

    assert [outcome.point_id for outcome in report.outcomes] == ["override"]
    assert report.requests[0].workload.isl == 512


def test_helm_benchmark_values_expand_literal_tool_pipeline_points():
    performance = {
        "toolPipeline": [
            {
                "tool": "aiperf",
                "config": {"name": "4k_500", "isl": 4000, "osl": 500, "concurrency": [4, 8]},
            }
        ]
    }

    report = adapt_config(DynamoRecipeSource(_agg_yaml(), performance))

    assert [(outcome.point_id, outcome.status) for outcome in report.outcomes] == [
        ("4k_500-concurrency-4", "adapted"),
        ("4k_500-concurrency-8", "adapted"),
    ]
    assert [request.workload.concurrency for request in report.requests] == [4, 8]


def test_run_perf_workload_is_discovered_without_execution():
    performance = {
        "spec": {
            "containers": [
                {"args": ["run_perf 8 1024 128"]},
            ]
        }
    }

    outcome = adapt_config(DynamoRecipeSource(_agg_yaml(), performance)).outcomes[0]

    assert outcome.request is not None
    assert outcome.request.workload.isl == 1024
    assert outcome.request.workload.osl == 128
    assert outcome.request.workload.concurrency == 8


def test_concurrency_per_gpu_workload_is_expanded():
    performance = {
        "spec": {
            "containers": [
                {
                    "env": [
                        {"name": "ISL", "value": "1024"},
                        {"name": "OSL", "value": "128"},
                        {"name": "CONCURRENCY_PER_GPU", "value": "4"},
                        {"name": "DEPLOYMENT_GPU_COUNT", "value": "2"},
                    ]
                }
            ]
        }
    }

    outcome = adapt_config(DynamoRecipeSource(_agg_yaml(), performance)).outcomes[0]

    assert outcome.request is not None
    assert outcome.request.workload.isl == 1024
    assert outcome.request.workload.osl == 128
    assert outcome.request.workload.concurrency == 8


def test_missing_workload_fallback_preserves_one_rejected_point():
    report = adapt_config(DynamoRecipeSource(_agg_yaml()))

    assert [(outcome.point_id, outcome.status) for outcome in report.outcomes] == [("point-0", "rejected")]
    assert "workload ISL must be an integer" in report.outcomes[0].diagnostics[-1].message


def test_configmap_command_conflict_is_rejected():
    deployment = """
kind: ConfigMap
metadata: {name: engine}
data:
  config.yaml: |
    tensor_parallel_size: 1
---
kind: DynamoGraphDeployment
metadata: {name: conflict}
spec:
  backendFramework: trtllm
  services:
    worker:
      componentType: worker
      resources: {limits: {gpu: "2"}}
      extraPodSpec:
        nodeSelector: {nvidia.com/gpu.product: NVIDIA-H200}
        volumes: [{name: engine, configMap: {name: engine}}]
        mainContainer:
          volumeMounts: [{name: engine, mountPath: /configs}]
          env:
            - {name: MODEL_PATH, value: QWEN/QWEN3-32B}
            - {name: ENGINE_ARGS, value: /configs/config.yaml}
          args: ["python -m dynamo.trtllm --model-path $MODEL_PATH --tensor-parallel-size 2"]
"""
    outcome = adapt_config(
        DynamoRecipeSource(deployment),
        AdapterOverrides(isl=1024, osl=128, concurrency=8),
    ).outcomes[0]

    assert outcome.status == "rejected"
    assert "conflicting tensor parallelism" in outcome.diagnostics[-1].message


@pytest.mark.parametrize(
    ("deployment", "message"),
    [
        (
            "kind: DynamoGraphDeployment\nspec: {backendFramework: vllm, services: {main: {componentType: main}}}\n",
            "componentType: main",
        ),
        (
            """
kind: DynamoGraphDeployment
spec:
  backendFramework: vllm
  services:
    AfdWorker:
      componentType: worker
      resources: {limits: {gpu: 1}}
      extraPodSpec: {mainContainer: {args: ["python -m dynamo.vllm --model QWEN/QWEN3-32B"]}}
""",
            "special topology",
        ),
    ],
)
def test_unsupported_special_topologies_are_rejected(deployment, message):
    outcome = adapt_config(
        DynamoRecipeSource(deployment),
        AdapterOverrides(system_name="h200_sxm", isl=1, osl=1, concurrency=1),
    ).outcomes[0]

    assert outcome.status == "rejected"
    assert message in outcome.diagnostics[-1].message


def test_missing_system_and_arbitrary_shell_values_are_rejected():
    missing_system = _agg_yaml().replace("nvidia.com/gpu.product: NVIDIA-H200-SXM", "nvidia.com/gpu.present: 'true'")
    shell_value = _agg_yaml().replace("value: QWEN/QWEN3-32B", "value: $(discover-model)")

    system_outcome = adapt_config(
        DynamoRecipeSource(missing_system),
        AdapterOverrides(isl=1, osl=1, concurrency=2),
    ).outcomes[0]
    shell_outcome = adapt_config(
        DynamoRecipeSource(shell_value),
        AdapterOverrides(isl=1, osl=1, concurrency=2),
    ).outcomes[0]

    assert system_outcome.status == "rejected"
    assert "GPU model is not declared" in system_outcome.diagnostics[-1].message
    assert shell_outcome.status == "rejected"
    assert "shell-derived" in shell_outcome.diagnostics[-1].message


@pytest.mark.parametrize("shell_parameter", ["${MODEL_PATH:-QWEN/QWEN3-32B}", "$1"])
def test_shell_parameter_expansions_are_rejected(shell_parameter):
    deployment = _agg_yaml(args_as_string=True).replace("$MODEL_PATH", shell_parameter)

    outcome = adapt_config(
        DynamoRecipeSource(deployment),
        AdapterOverrides(isl=1024, osl=128, concurrency=8),
    ).outcomes[0]

    assert outcome.status == "rejected"
    assert "unsupported shell parameter expansion" in outcome.diagnostics[-1].message


def test_zero_speculative_depth_remains_disabled():
    deployment = _agg_yaml(args_as_string=True).replace(
        "--gpu-memory-utilization 0.9",
        "--num-speculative-tokens 0 --gpu-memory-utilization 0.9",
    )

    outcome = adapt_config(
        DynamoRecipeSource(deployment),
        AdapterOverrides(isl=1024, osl=128, concurrency=8),
    ).outcomes[0]

    assert outcome.request is not None
    assert outcome.request.model.nextn == 0
    assert outcome.request.model.nextn_accepted is None


def test_zero_speculative_depth_in_engine_config_remains_disabled():
    deployment = """
kind: ConfigMap
metadata: {name: engine}
data:
  config.yaml: |
    tensor_parallel_size: 2
    speculative_config:
      num_nextn_predict_layers: 0
---
kind: DynamoGraphDeployment
metadata: {name: zero-speculation}
spec:
  backendFramework: trtllm
  services:
    worker:
      componentType: worker
      resources: {limits: {gpu: "2"}}
      extraPodSpec:
        nodeSelector: {nvidia.com/gpu.product: NVIDIA-H200}
        volumes: [{name: engine, configMap: {name: engine}}]
        mainContainer:
          volumeMounts: [{name: engine, mountPath: /config}]
          args:
            - >-
              python -m dynamo.trtllm --model-path QWEN/QWEN3-32B
              --extra-engine-args /config/config.yaml
"""

    outcome = adapt_config(
        DynamoRecipeSource(deployment),
        AdapterOverrides(isl=1024, osl=128, concurrency=8),
    ).outcomes[0]

    assert outcome.status == "adapted"
    assert outcome.request is not None
    assert outcome.request.model.nextn == 0
    assert outcome.request.model.nextn_accepted is None


def test_conflicting_engine_config_speculative_depths_are_rejected():
    deployment = """
kind: ConfigMap
metadata: {name: prefill-engine}
data:
  config.yaml: |
    tensor_parallel_size: 1
    speculative_config: {num_nextn_predict_layers: 2}
---
kind: ConfigMap
metadata: {name: decode-engine}
data:
  config.yaml: |
    tensor_parallel_size: 1
    speculative_config: {num_nextn_predict_layers: 3}
---
kind: DynamoGraphDeployment
metadata: {name: conflicting-speculation}
spec:
  backendFramework: trtllm
  services:
    prefill:
      componentType: worker
      subComponentType: prefill
      resources: {limits: {gpu: "1"}}
      extraPodSpec:
        nodeSelector: {nvidia.com/gpu.product: NVIDIA-H200}
        volumes: [{name: engine, configMap: {name: prefill-engine}}]
        mainContainer:
          volumeMounts: [{name: engine, mountPath: /config}]
          args:
            - >-
              python -m dynamo.trtllm --model-path QWEN/QWEN3-32B
              --disaggregation-mode prefill --extra-engine-args /config/config.yaml
    decode:
      componentType: worker
      subComponentType: decode
      resources: {limits: {gpu: "1"}}
      extraPodSpec:
        nodeSelector: {nvidia.com/gpu.product: NVIDIA-H200}
        volumes: [{name: engine, configMap: {name: decode-engine}}]
        mainContainer:
          volumeMounts: [{name: engine, mountPath: /config}]
          args:
            - >-
              python -m dynamo.trtllm --model-path QWEN/QWEN3-32B
              --disaggregation-mode decode --extra-engine-args /config/config.yaml
"""

    outcome = adapt_config(
        DynamoRecipeSource(deployment),
        AdapterOverrides(isl=1024, osl=128, concurrency=8, nextn_accepted=1.5),
    ).outcomes[0]

    assert outcome.status == "rejected"
    assert "different speculative depths" in outcome.diagnostics[-1].message


def test_speculative_model_flag_remains_active():
    deployment = _agg_yaml(args_as_string=True).replace(
        "--gpu-memory-utilization 0.9",
        "--speculative-model Qwen/Qwen3-0.6B --gpu-memory-utilization 0.9",
    )

    outcome = adapt_config(
        DynamoRecipeSource(deployment),
        AdapterOverrides(isl=1024, osl=128, concurrency=8),
    ).outcomes[0]

    assert outcome.status == "rejected"
    assert "nextn_accepted" in outcome.diagnostics[-1].message


def test_dynamo_ci_concrete_recipe_expands_points_and_maps_topology():
    report = adapt_config(DynamoRecipeSource(_dynamo_ci_recipe()))

    assert [(outcome.point_id, outcome.status) for outcome in report.outcomes] == [
        ("concurrency-8", "adapted"),
        ("concurrency-10", "rejected"),
    ]
    request = report.requests[0]
    assert request.model.path == "deepseek-ai/DeepSeek-R1"
    assert request.backend.name == "sglang"
    assert request.backend.version == "0.5.5.post2"
    assert request.systems.prefill == "gb200"
    assert request.quantization.gemm == "nvfp4"
    assert request.quantization.kvcache == "fp8"
    assert request.topology.prefill.gpus_per_replica == 4
    assert request.topology.prefill.batch_size == 1
    assert request.topology.decode.replicas == 2
    assert request.topology.decode.attention_dp_size == 4
    assert request.topology.decode.batch_size == 1
    assert request.provenance.source_ids["format"] == "dynamo-ci-concrete"
    assert "must be divisible" in report.rejections[0].diagnostics[-1].message


def test_dynamo_ci_decode_batch_override_preserves_nondivisible_concurrency():
    recipe = _dynamo_ci_recipe()
    recipe["benchmark"]["concurrencies"] = "10"

    rejected = adapt_config(DynamoRecipeSource(recipe)).outcomes[0]
    adapted = adapt_config(
        DynamoRecipeSource(recipe),
        AdapterOverrides(decode_batch_size=2),
    ).outcomes[0]

    assert rejected.status == "rejected"
    assert "must be divisible" in rejected.diagnostics[-1].message
    assert adapted.request is not None
    assert adapted.request.workload.concurrency == 10
    assert adapted.request.topology.decode.batch_size == 2
    assert "decode batch size is explicitly overridden to 2" in adapted.request.provenance.assumptions[-1]


def test_dynamo_ci_rejects_workload_exceeding_declared_decode_context_length():
    recipe = _dynamo_ci_recipe()
    recipe["benchmark"] = {"isl": 8192, "osl": 1024, "concurrencies": "8"}

    outcome = adapt_config(DynamoRecipeSource(recipe)).outcomes[0]

    assert outcome.status == "rejected"
    assert "runtime.decode_max_seq_len (8192)" in outcome.diagnostics[-1].message
    assert "workload.isl + workload.osl (9216)" in outcome.diagnostics[-1].message


def test_dynamo_ci_preserves_role_specific_memory_fractions():
    recipe = _dynamo_ci_recipe()
    recipe["backend"]["sglang_config"]["decode"]["mem-fraction-static"] = 0.8

    adapted = adapt_config(DynamoRecipeSource(recipe)).outcomes[0]
    overridden = adapt_config(DynamoRecipeSource(recipe), AdapterOverrides(free_gpu_memory_fraction=0.85)).outcomes[0]
    role_overridden = adapt_config(
        DynamoRecipeSource(recipe),
        AdapterOverrides(
            free_gpu_memory_fraction=0.85,
            prefill_free_gpu_memory_fraction=0.75,
            decode_free_gpu_memory_fraction=0.7,
        ),
    ).outcomes[0]

    assert adapted.request is not None
    assert adapted.request.runtime.free_gpu_memory_fraction is None
    assert adapted.request.runtime.prefill_free_gpu_memory_fraction == 0.9
    assert adapted.request.runtime.decode_free_gpu_memory_fraction == 0.8
    assert overridden.request is not None
    assert overridden.request.runtime.free_gpu_memory_fraction == 0.85
    assert overridden.request.runtime.prefill_free_gpu_memory_fraction is None
    assert overridden.request.runtime.decode_free_gpu_memory_fraction is None
    assert role_overridden.request is not None
    assert role_overridden.request.runtime.free_gpu_memory_fraction == 0.85
    assert role_overridden.request.runtime.prefill_free_gpu_memory_fraction == 0.75
    assert role_overridden.request.runtime.decode_free_gpu_memory_fraction == 0.7


def test_dynamo_ci_declared_unsupported_quantization_is_rejected_even_for_fp8_model():
    recipe = _dynamo_ci_recipe()
    recipe["benchmark"]["concurrencies"] = "8"
    recipe["model"]["precision"] = "fp8"
    recipe["backend"]["sglang_config"]["prefill"]["quantization"] = "awq"
    recipe["backend"]["sglang_config"]["decode"]["quantization"] = "awq"

    outcome = adapt_config(DynamoRecipeSource(recipe)).outcomes[0]

    assert outcome.status == "rejected"
    assert "unsupported dynamo-ci quantization mode 'awq'" in outcome.diagnostics[-1].message


def test_dynamo_ci_resource_parallelism_mismatch_is_rejected_for_every_point():
    recipe = _dynamo_ci_recipe()
    recipe["resources"]["decode_workers"] = 1

    report = adapt_config(DynamoRecipeSource(recipe))

    assert [outcome.status for outcome in report.outcomes] == ["rejected", "rejected"]
    assert all("does not match GPUs per worker" in outcome.diagnostics[-1].message for outcome in report.outcomes)


def test_dynamo_ci_speculation_requires_acceptance():
    recipe = _dynamo_ci_recipe()
    recipe["benchmark"]["concurrencies"] = "8"
    recipe["backend"]["sglang_config"]["decode"]["speculative-num-steps"] = 2

    rejected = adapt_config(DynamoRecipeSource(recipe)).outcomes[0]
    adapted = adapt_config(
        DynamoRecipeSource(recipe),
        AdapterOverrides(nextn_accepted=1.5),
    ).outcomes[0]

    assert rejected.status == "rejected"
    assert "nextn_accepted" in rejected.diagnostics[-1].message
    assert adapted.request is not None
    assert adapted.request.model.nextn == 2
    assert adapted.request.model.nextn_accepted == 1.5


def test_components_schema_resolves_configmap_placeholders():
    deployment = """
kind: ConfigMap
metadata: {name: engine-config}
data:
  speculative-config: '{"method":"mtp","num_speculative_tokens":3}'
---
kind: DynamoGraphDeployment
metadata: {name: components-agg}
spec:
  backendFramework: vllm
  components:
    - name: Frontend
      type: frontend
      podTemplate: {spec: {containers: [{name: main, args: [frontend]}]}}
    - name: agg
      type: worker
      replicas: 2
      podTemplate:
        spec:
          nodeSelector: {nvidia.com/gpu.product: NVIDIA-B200}
          containers:
            - name: main
              command: [python3, -m, dynamo.vllm]
              args:
                - --model=Qwen/Qwen3-32B
                - --tensor-parallel-size=1
                - --speculative-config=$(SPEC_CONFIG)
              env:
                - name: SPEC_CONFIG
                  valueFrom: {configMapKeyRef: {name: engine-config, key: speculative-config}}
              resources: {limits: {nvidia.com/gpu: "1"}}
"""

    outcome = adapt_config(
        DynamoRecipeSource(deployment),
        AdapterOverrides(isl=1024, osl=128, concurrency=8, nextn=3, nextn_accepted=1.5),
    ).outcomes[0]

    assert outcome.request is not None
    assert outcome.request.systems.prefill == "b200_sxm"
    assert outcome.request.topology.worker.replicas == 2
    assert outcome.request.topology.worker.batch_size == 4


def test_components_schema_extracts_literal_engine_command_from_shell_wrapper():
    deployment = """
kind: ConfigMap
metadata: {name: sglang-config}
data:
  prefill.yaml: |-
    model-path: Qwen/Qwen3-32B
    tensor-parallel-size: 2
  decode.yaml: |-
    model-path: Qwen/Qwen3-32B
    tensor-parallel-size: 2
    max-running-requests: 16
---
kind: DynamoGraphDeployment
metadata: {name: components-disagg}
spec:
  backendFramework: sglang
  components:
    - name: prefill
      type: prefill
      replicas: 1
      podTemplate:
        spec:
          nodeSelector: {nvidia.com/gpu.product: NVIDIA-H200}
          containers:
            - name: main
              command: [/bin/bash, -lc]
              args:
                - |
                  DEVICE="$(discover-device --dynamic)"
                  exec python3 -m dynamo.sglang --config /etc/sglang/prefill.yaml
              volumeMounts:
                - {name: config, mountPath: /etc/sglang}
              resources: {limits: {nvidia.com/gpu: "2"}}
          volumes:
            - {name: config, configMap: {name: sglang-config}}
    - name: decode
      type: decode
      replicas: 2
      podTemplate:
        spec:
          nodeSelector: {nvidia.com/gpu.product: NVIDIA-H200}
          containers:
            - name: main
              command: [/bin/bash, -lc]
              args:
                - |
                  DEVICE="$(discover-device --dynamic)"
                  exec python3 -m dynamo.sglang --config /etc/sglang/decode.yaml
              volumeMounts:
                - {name: config, mountPath: /etc/sglang}
              resources: {limits: {nvidia.com/gpu: "2"}}
          volumes:
            - {name: config, configMap: {name: sglang-config}}
"""

    outcome = adapt_config(
        DynamoRecipeSource(deployment),
        AdapterOverrides(isl=1024, osl=128, concurrency=16),
    ).outcomes[0]

    assert outcome.request is not None
    assert outcome.request.topology.kind == "disagg"
    assert outcome.request.topology.prefill.tp_size == 2
    assert outcome.request.topology.decode.replicas == 2
    assert outcome.request.topology.decode.batch_size == 8


def test_explicit_batch_override_preserves_nondivisible_source_concurrency():
    deployment = _agg_yaml().replace("replicas: 1", "replicas: 3")

    rejected = adapt_config(
        DynamoRecipeSource(deployment),
        AdapterOverrides(isl=1024, osl=128, concurrency=32),
    ).outcomes[0]
    adapted = adapt_config(
        DynamoRecipeSource(deployment),
        AdapterOverrides(isl=1024, osl=128, concurrency=32, batch_size=11),
    ).outcomes[0]

    assert rejected.status == "rejected"
    assert adapted.request is not None
    assert adapted.request.workload.concurrency == 32
    assert adapted.request.topology.worker.batch_size == 11
    assert "active batch size is explicitly overridden" in adapted.request.provenance.assumptions[-2]


def test_trtllm_extra_engine_args_override_command_runtime_values():
    deployment = """
kind: ConfigMap
metadata: {name: engine}
data:
  config.yaml: |
    tensor_parallel_size: 1
    max_batch_size: 32
    max_seq_len: 4096
    kv_cache_config: {free_gpu_memory_fraction: 0.7}
---
kind: DynamoGraphDeployment
metadata: {name: trtllm-precedence}
spec:
  backendFramework: trtllm
  services:
    worker:
      componentType: worker
      replicas: 1
      resources: {limits: {gpu: 1}}
      extraPodSpec:
        nodeSelector: {nvidia.com/gpu.product: NVIDIA-H200}
        volumes: [{name: engine, configMap: {name: engine}}]
        mainContainer:
          volumeMounts: [{name: engine, mountPath: /config}]
          env: [{name: MODEL_PATH, value: Qwen/Qwen3-32B}]
          args:
            - >-
              python -m dynamo.trtllm --model-path ${MODEL_PATH}
              --extra-engine-args /config/config.yaml
              --free-gpu-memory-fraction 0.9
"""

    outcome = adapt_config(
        DynamoRecipeSource(deployment),
        AdapterOverrides(isl=1024, osl=128, concurrency=16),
    ).outcomes[0]

    assert outcome.request is not None
    assert outcome.request.runtime.free_gpu_memory_fraction == 0.7


def test_explicit_backend_main_components_map_as_disagg_workers():
    deployment = """
kind: ConfigMap
metadata: {name: engine}
data:
  prefill.yaml: |
    tensor_parallel_size: 1
    max_batch_size: 1
    max_seq_len: 9000
  decode.yaml: |
    tensor_parallel_size: 2
    max_batch_size: 32
    max_seq_len: 11000
---
kind: DynamoGraphDeployment
metadata: {name: main-components}
spec:
  backendFramework: trtllm
  services:
    prefill:
      componentType: main
      replicas: 1
      resources: {limits: {gpu: 1}}
      extraPodSpec:
        nodeSelector: {nvidia.com/gpu.product: NVIDIA-H200}
        volumes: [{name: engine, configMap: {name: engine}}]
        mainContainer:
          volumeMounts: [{name: engine, mountPath: /config}]
          env: [{name: MODEL_PATH, value: openai/gpt-oss-120b}]
          args:
            - >-
              python -m dynamo.trtllm --model-path ${MODEL_PATH}
              --extra-engine-args /config/prefill.yaml --disaggregation-mode prefill
    decode:
      componentType: main
      replicas: 1
      resources: {limits: {gpu: 2}}
      extraPodSpec:
        nodeSelector: {nvidia.com/gpu.product: NVIDIA-H200}
        volumes: [{name: engine, configMap: {name: engine}}]
        mainContainer:
          volumeMounts: [{name: engine, mountPath: /config}]
          env: [{name: MODEL_PATH, value: openai/gpt-oss-120b}]
          args:
            - >-
              python -m dynamo.trtllm --model-path ${MODEL_PATH}
              --extra-engine-args /config/decode.yaml --disaggregation-mode decode
"""

    outcome = adapt_config(
        DynamoRecipeSource(deployment),
        AdapterOverrides(isl=1024, osl=128, concurrency=16),
    ).outcomes[0]

    assert outcome.request is not None
    assert outcome.request.topology.kind == "disagg"
    assert outcome.request.topology.prefill.gpus_per_replica == 1
    assert outcome.request.topology.decode.gpus_per_replica == 2
    assert outcome.request.runtime.max_seq_len is None
    assert outcome.request.runtime.prefill_max_seq_len == 9000
    assert outcome.request.runtime.decode_max_seq_len == 11000
    kwargs = to_cli_estimate_kwargs(outcome.request)
    assert kwargs["prefill_max_seq_len"] == 9000
    assert kwargs["decode_max_seq_len"] == 11000
