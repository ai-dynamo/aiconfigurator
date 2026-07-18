# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest

from collector.fpm_forward.config import FPMCollectionOptions, PrefillSamplingProfile, add_fpm_arguments
from collector.fpm_forward.database import aggregate_cell, write_formal_database
from collector.fpm_forward.planner import BackendPolicy, FPMCell, build_collection_plan
from collector.fpm_forward.topology import enumerate_fpm_topologies
from collector.fpm_forward.types import ParallelTopology

pytestmark = pytest.mark.unit


def _write_provenance(path, *, cell_id: str, plan_sha256: str = "plan-sha", attempt_id: str = "attempt"):
    path.write_text(
        json.dumps(
            {
                "schema_name": "aic_fpm_collector_provenance",
                "schema_version": 1,
                "cell_id": cell_id,
                "plan_sha256": plan_sha256,
                "attempt_id": attempt_id,
                "runtime": {"backend": "vllm", "backend_version": "0.24.0"},
            }
        )
    )


def _concurrent_database_writer(root: str, row: dict, start_event) -> None:
    start_event.wait(timeout=10)
    plan = SimpleNamespace(system="b200_sxm", backend="vllm", aic_revision="revision")
    write_formal_database(plan, [row], systems_root=Path(root))


def _args(**overrides):
    values = {
        "fpm_max_gpus": 4,
        "fpm_gpu_counts": [4],
        "fpm_parallel_presets": None,
        "fpm_parallel_axes": None,
        "fpm_backend_axes": None,
        "fpm_weight_quantizations": None,
        "fpm_kv_cache_dtypes": None,
        "fpm_tp_sizes": None,
        "fpm_pp_sizes": None,
        "fpm_dp_sizes": None,
        "fpm_moe_tp_sizes": None,
        "fpm_moe_ep_sizes": None,
        "fpm_cp_sizes": None,
        "fpm_warmup_iterations": None,
        "fpm_max_prefill_isl": None,
        "fpm_max_prefill_batch_size": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_options_leave_point_generation_to_dynamo():
    options = FPMCollectionOptions.from_args(_args())

    assert options.warmup_iterations == 5
    assert options.gpu_counts == (4,)
    assert options.parallel_presets == ("auto",)
    assert options.to_dict()["point_source"] == "dynamo_native_self_benchmark"
    assert options.to_dict()["measurement_repeats"] == 1
    assert options.max_prefill_isl == 8192
    assert options.max_prefill_batch_size is None
    assert options.vllm_max_model_len == -1
    assert options.prefill_sampling.max_total_prefill_tokens == 8192
    assert len(options.prefill_sampling.cudagraph_capture_sizes) == 99
    assert options.prefill_sampling.max_cudagraph_capture_size == 2048
    assert len(options.prefill_sampling.new_token_axis_points) == 199
    assert options.prefill_sampling.max_new_token_samples == 199
    assert "sampling_budget" not in options.to_dict()
    assert "kv_block_size" not in options.to_dict()

    warmed = FPMCollectionOptions.from_args(_args(fpm_warmup_iterations=3))
    assert warmed.to_dict()["global_warmup_iterations"] == 3
    disabled = FPMCollectionOptions.from_args(_args(fpm_warmup_iterations=0))
    assert disabled.to_dict()["global_warmup_iterations"] == 0

    with pytest.raises(ValueError, match="exceed"):
        FPMCollectionOptions.from_args(_args(fpm_gpu_counts=[4, 8]))


def test_prefill_limits_expose_no_cli_aliases():
    parser = argparse.ArgumentParser()
    add_fpm_arguments(parser)

    help_text = parser.format_help()
    assert "--fpm-max-prefill-isl" in help_text
    assert "--fpm-max-prefill-batch-size" in help_text
    assert "--fpm-max-isl" not in help_text
    assert "--fpm-max-prefill-bs" not in help_text


def test_prefill_sampling_profile_keeps_vllm_strides_and_exact_endpoint():
    short = PrefillSamplingProfile.build(max_isl=1000, max_batch_size=16)

    assert short.max_cudagraph_capture_size == 1000
    assert short.cudagraph_capture_sizes[:7] == (1, 2, 4, 8, 16, 24, 32)
    assert short.cudagraph_capture_sizes[-4:] == (928, 960, 992, 1000)
    assert len(short.cudagraph_capture_sizes) == 67
    assert len(short.new_token_axis_points) == 132
    assert short.max_new_token_samples == 132

    long = PrefillSamplingProfile.build(max_isl=8192, max_batch_size=None)
    assert long.cudagraph_capture_sizes[-4:] == (1952, 1984, 2016, 2048)
    assert long.new_token_axis_points[-4:] == (2048, 2049, 4096, 8192)
    assert long.to_dict()["new_token_axis_point_count"] == 199


def test_parallel_topologies_are_delegated_to_aic_enumerator():
    options = FPMCollectionOptions.from_args(_args())
    topologies = enumerate_fpm_topologies(backend="vllm", is_moe=True, options=options)
    assert topologies == (
        ParallelTopology(tp=1, pp=1, dp=4, moe_tp=1, moe_ep=4, cp=1),
        ParallelTopology(tp=4, pp=1, dp=1, moe_tp=1, moe_ep=4, cp=1),
    )

    with_pure_tp = enumerate_fpm_topologies(
        backend="vllm",
        is_moe=True,
        options=options,
        allow_pure_tp=True,
    )
    assert with_pure_tp == (
        ParallelTopology(tp=1, pp=1, dp=4, moe_tp=1, moe_ep=4, cp=1),
        ParallelTopology(tp=4, pp=1, dp=1, moe_tp=4, moe_ep=1, cp=1),
        ParallelTopology(tp=4, pp=1, dp=1, moe_tp=1, moe_ep=4, cp=1),
    )


def test_pure_tp_requires_explicit_model_runtime_capability():
    options = FPMCollectionOptions.from_args(
        _args(
            fpm_parallel_presets=["pure_tp"],
            fpm_moe_tp_sizes=[4],
        )
    )

    with pytest.raises(ValueError, match="does not explicitly admit"):
        enumerate_fpm_topologies(backend="vllm", is_moe=True, options=options)


def test_plan_contains_only_cell_matrix_and_native_point_contract():
    options = FPMCollectionOptions.from_args(
        _args(
            fpm_parallel_axes=["dp", "moe_ep"],
            fpm_dp_sizes=[4],
            fpm_moe_ep_sizes=[4],
        )
    )
    kwargs = {
        "backend": "vllm",
        "model_path": "nvidia/GLM-5.2-NVFP4",
        "system": "b200_sxm",
        "selected_ops": {"dsa_context_module", "dsa_generation_module"},
        "options": options,
    }
    first = build_collection_plan(
        **kwargs,
        generator_overrides={"K8sConfig": {"k8s_image": "example/vllm-runtime:first"}},
    )
    second = build_collection_plan(
        **kwargs,
        generator_overrides={"K8sConfig": {"k8s_image": "example/vllm-runtime:second"}},
    )

    assert first.sha256 != second.sha256
    assert first.dtype_profile.gemm_quant_mode == "nvfp4"
    assert first.dtype_profile.kv_cache_dtypes == ("fp8",)
    assert len(first.cells) == 2
    assert {cell.workload_kind for cell in first.cells} == {"prefill", "decode"}
    assert {cell.parallel_strategy for cell in first.cells} == {"dep"}
    payload = first.to_dict()
    assert payload["schema_version"] == 9
    point_generation = dict(payload["point_generation"])
    prefill_sampling = point_generation.pop("prefill_sampling")
    assert point_generation == {
        "owner": "dynamo.vllm.instrumented_scheduler.InstrumentedScheduler",
        "method": "native_self_benchmark",
        "coordinates": ["batch_size", "total_prefill_tokens", "total_kv_read_tokens"],
        "partition_policy": "balanced_v1",
        "point_admission": "dynamo_live_scheduler",
        "precondition": "vllm_engine_initialized",
        "planned_point_count": None,
    }
    assert prefill_sampling["cudagraph_capture_size_count"] == 99
    assert prefill_sampling["new_token_axis_point_count"] == 199
    assert prefill_sampling["prefill_max_new_token_samples"] == 199
    assert "prefix_max_batch_size_samples" not in prefill_sampling
    assert payload["counts"]["prefill_cudagraph_capture_sizes"] == 99
    assert payload["counts"]["prefill_new_token_axis_points"] == 199
    assert payload["counts"]["points"] == "runtime-determined"
    assert "population" not in payload
    assert "sampling" not in payload
    assert "capacity_admission" not in payload
    assert payload["counts"]["candidate_topologies"] == 1
    assert payload["counts"]["memory_rejected_topologies"] == 0
    assert payload["topology_memory_admission"][0]["disposition"] == "admitted"
    assert "runtime_overlay" not in payload


def test_backend_policy_is_deeply_immutable():
    source = {"nested": {"values": [1, 2]}}
    policy = BackendPolicy("baseline", "baseline", source, {"runtime.mode": "FULL"})
    original = policy.to_dict()

    source["nested"]["values"].append(3)
    detached = policy.generator_overrides
    detached["nested"]["values"].append(4)

    assert policy.to_dict() == original


def test_glm_auto_matrix_keeps_aic_parallel_and_dtype_resolution():
    plan = build_collection_plan(
        backend="vllm",
        model_path="nvidia/GLM-5.2-NVFP4",
        model_architecture="GlmMoeDsaForCausalLM",
        system="b200_sxm",
        selected_ops={"dsa_context_module", "dsa_generation_module"},
        options=FPMCollectionOptions.from_args(_args()),
    )

    assert plan.capability.allow_pure_tp is True
    assert {cell.parallel_strategy for cell in plan.cells} == {"pure_tp", "tep", "dep"}
    assert ParallelTopology(tp=4, pp=1, dp=1, moe_tp=4, moe_ep=1, cp=1) in plan.topologies
    assert len(plan.cells) == len(plan.topologies) * 2
    assert all(cell.to_dict()["point_source"] == "dynamo_native_self_benchmark" for cell in plan.cells)


def test_glm_memory_admission_uses_configured_max_new_tokens_and_warns_on_drops(caplog):
    plan = build_collection_plan(
        backend="vllm",
        model_path="nvidia/GLM-5.2-NVFP4",
        model_architecture="GlmMoeDsaForCausalLM",
        system="b200_sxm",
        selected_ops={"dsa_context_module", "dsa_generation_module"},
        options=FPMCollectionOptions.from_args(
            _args(
                fpm_gpu_counts=[1, 2, 4],
                fpm_max_prefill_isl=16384,
            )
        ),
    )

    assert {topology.total_gpus for topology in plan.topologies} == {4}
    assert len(plan.topologies) == 3
    payload = plan.to_dict()
    assert payload["counts"]["candidate_topologies"] == 7
    assert payload["counts"]["memory_rejected_topologies"] == 4
    assert {
        decision["topology"]["tp"] * decision["topology"]["dp"]
        for decision in payload["topology_memory_admission"]
        if decision["disposition"] == "rejected"
    } == {1, 2}
    assert "fpm_forward: dropped 4/7 topologies" in caplog.text
    assert "max_new_tokens=16384" in caplog.text
    assert {decision["activation_envelope"]["max_new_tokens"] for decision in payload["topology_memory_admission"]} == {
        16384
    }


def test_glm_memory_admission_fails_after_warning_when_every_topology_is_impossible(caplog):
    with pytest.raises(ValueError, match="rejected every structurally valid FPM topology"):
        build_collection_plan(
            backend="vllm",
            model_path="nvidia/GLM-5.2-NVFP4",
            model_architecture="GlmMoeDsaForCausalLM",
            system="b200_sxm",
            selected_ops={"dsa_context_module", "dsa_generation_module"},
            options=FPMCollectionOptions.from_args(
                _args(
                    fpm_gpu_counts=[1, 2],
                )
            ),
        )

    assert "fpm_forward: dropped 4/4 topologies" in caplog.text


def test_memory_admission_keeps_unknown_estimates_for_runtime_verification(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("model is not supported by the AIC memory estimator")

    monkeypatch.setattr(
        "collector.fpm_forward.memory_admission.KVCacheEstimator.from_request",
        unavailable,
    )
    plan = build_collection_plan(
        backend="vllm",
        model_path="nvidia/GLM-5.2-NVFP4",
        model_architecture="GlmMoeDsaForCausalLM",
        system="b200_sxm",
        selected_ops={"dsa_context_module", "dsa_generation_module"},
        options=FPMCollectionOptions.from_args(_args()),
    )

    assert len(plan.topologies) == 3
    assert {decision.disposition for decision in plan.topology_memory_admission} == {"unknown"}


def test_memory_admission_drops_only_the_rejected_dtype_cells(monkeypatch):
    class Estimate:
        def __init__(self, *, admitted: bool):
            self.breakdown = {
                "non_kv_bytes": 50 if admitted else 150,
                "gpu_memory_capacity_bytes": 100,
            }

    def estimate(*_args, **kwargs):
        return Estimate(admitted=kwargs["kvcache_quant_mode"] == "fp8")

    monkeypatch.setattr(
        "collector.fpm_forward.memory_admission.KVCacheEstimator.from_request",
        estimate,
    )
    plan = build_collection_plan(
        backend="vllm",
        model_path="nvidia/GLM-5.2-NVFP4",
        model_architecture="GlmMoeDsaForCausalLM",
        system="b200_sxm",
        selected_ops={"dsa_context_module", "dsa_generation_module"},
        options=FPMCollectionOptions.from_args(_args(fpm_kv_cache_dtypes=["bfloat16", "fp8"])),
    )

    assert {cell.kv_cache_dtype for cell in plan.cells} == {"fp8"}
    assert {
        estimate.kv_cache_dtype
        for decision in plan.topology_memory_admission
        for estimate in decision.estimates
        if estimate.disposition == "rejected"
    } == {"bfloat16"}


def test_minimax_m3_keeps_family_dtype_and_parallel_capabilities():
    plan = build_collection_plan(
        backend="vllm",
        model_path="MiniMaxAI/MiniMax-M3",
        model_architecture="MiniMaxM3ForCausalLM",
        system="b200_sxm",
        selected_ops={"attention_context", "attention_generation"},
        has_model_cases=False,
        options=FPMCollectionOptions.from_args(
            _args(
                fpm_max_gpus=16,
                fpm_gpu_counts=[8, 16],
            )
        ),
    )

    assert plan.capability.support_level == "family_template"
    assert plan.capability.template_id == "aic_family:minimaxm3:moe_msa"
    assert plan.capability.attention_kind == "moe_msa"
    assert plan.capability.attention_source == "dsa_module"
    assert plan.capability.allow_pure_tp is True
    assert {cell.parallel_strategy for cell in plan.cells} == {"pure_tp", "tep", "dep"}


_DSV4_ATTENTION_OPS = {
    "dsv4_csa_context_module",
    "dsv4_hca_context_module",
    "dsv4_csa_generation_module",
    "dsv4_hca_generation_module",
}


@pytest.mark.parametrize(
    ("model_path", "expected_strategies", "expected_memory_rejections"),
    [
        ("sgl-project/DeepSeek-V4-Pro-FP8", {"pure_tp", "tep"}, 1),
        ("sgl-project/DeepSeek-V4-Flash-FP8", {"pure_tp", "tep", "dep"}, 0),
    ],
)
def test_dsv4_fp8_keeps_exact_capabilities_and_applies_max_new_token_memory_admission(
    model_path,
    expected_strategies,
    expected_memory_rejections,
):
    plan = build_collection_plan(
        backend="vllm",
        model_path=model_path,
        model_architecture="DeepseekV4ForCausalLM",
        system="b200_sxm",
        selected_ops=_DSV4_ATTENTION_OPS,
        options=FPMCollectionOptions.from_args(
            _args(
                fpm_max_gpus=16,
                fpm_gpu_counts=[16],
            )
        ),
    )

    assert plan.capability.support_level == "exact"
    assert plan.capability.template_id == "aic_exact:dsv4_module"
    assert plan.capability.attention_kind == "moe_dsv4"
    assert plan.capability.attention_source == "dsv4_module"
    assert plan.capability.allow_pure_tp is True
    assert plan.dtype_profile.fmha_quant_mode == "bfloat16"
    assert plan.dtype_profile.kv_cache_dtypes == ("fp8",)
    assert {cell.parallel_strategy for cell in plan.cells} == expected_strategies
    assert plan.to_dict()["counts"]["memory_rejected_topologies"] == expected_memory_rejections
    assert all(cell.to_dict()["point_source"] == "dynamo_native_self_benchmark" for cell in plan.cells)


@pytest.mark.parametrize(
    "model_path",
    [
        "deepseek-ai/DeepSeek-V4-Pro",
        "deepseek-ai/DeepSeek-V4-Flash",
    ],
)
def test_dsv4_native_fp4_is_rejected_by_vllm_dtype_capability(model_path):
    with pytest.raises(ValueError, match="does not support moe dtype 'w4a8_mxfp4_mxfp8'"):
        build_collection_plan(
            backend="vllm",
            model_path=model_path,
            model_architecture="DeepseekV4ForCausalLM",
            system="b200_sxm",
            selected_ops=_DSV4_ATTENTION_OPS,
            options=FPMCollectionOptions.from_args(
                _args(
                    fpm_max_gpus=16,
                    fpm_gpu_counts=[16],
                )
            ),
        )


def test_unknown_model_keeps_auditable_capability_template(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["NewMoeDsaForCausalLM"],
                "num_attention_heads": 32,
                "num_key_value_heads": 32,
                "hidden_size": 4096,
                "intermediate_size": 8192,
                "num_hidden_layers": 4,
                "vocab_size": 32000,
                "n_routed_experts": 8,
                "kv_lora_rank": 512,
                "qk_rope_head_dim": 64,
                "max_position_embeddings": 4096,
            }
        )
    )
    plan = build_collection_plan(
        backend="vllm",
        model_path=str(tmp_path),
        model_architecture="NewMoeDsaForCausalLM",
        system="b200_sxm",
        selected_ops={"attention_context", "attention_generation"},
        has_model_cases=False,
        options=FPMCollectionOptions.from_args(_args()),
    )

    assert plan.capability.support_level == "bootstrap_template"
    assert plan.capability.template_id == "generic:moe_dsa"
    assert {cell.parallel_strategy for cell in plan.cells} == {"dep", "pure_tp", "tep"}


def test_unregistered_dense_model_keeps_gqa_bootstrap_template(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Phi3ForCausalLM"],
                "hidden_size": 3072,
                "intermediate_size": 8192,
                "num_hidden_layers": 32,
                "num_attention_heads": 24,
                "num_key_value_heads": 8,
                "vocab_size": 200064,
                "max_position_embeddings": 131072,
                "torch_dtype": "bfloat16",
            }
        )
    )
    plan = build_collection_plan(
        backend="vllm",
        model_path=str(tmp_path),
        model_architecture="Phi3ForCausalLM",
        system="b200_sxm",
        selected_ops={"attention_context", "attention_generation"},
        has_model_cases=False,
        options=FPMCollectionOptions.from_args(_args()),
    )

    assert plan.capability.model_family is None
    assert plan.capability.support_level == "bootstrap_template"
    assert plan.capability.template_id == "generic:dense_gqa"
    assert plan.capability.attention_source == "dense_attention"
    assert plan.capability.allow_pure_tp is False
    assert {cell.parallel_strategy for cell in plan.cells} == {"tp"}


def test_registered_dense_model_without_case_file_keeps_family_template():
    plan = build_collection_plan(
        backend="vllm",
        model_path="Qwen/Qwen3-32B",
        model_architecture="Qwen3ForCausalLM",
        system="b200_sxm",
        selected_ops={"attention_context", "attention_generation"},
        has_model_cases=False,
        options=FPMCollectionOptions.from_args(_args()),
    )

    assert plan.capability.support_level == "family_template"
    assert plan.capability.template_id == "aic_family:llama:dense_gqa"
    assert plan.capability.attention_source == "dense_attention"
    assert {cell.parallel_strategy for cell in plan.cells} == {"tp"}


def test_exact_dense_mla_model_does_not_become_moe(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepSeekForCausalLM"],
                "hidden_size": 2048,
                "intermediate_size": 8192,
                "num_hidden_layers": 4,
                "num_attention_heads": 16,
                "num_key_value_heads": 16,
                "kv_lora_rank": 512,
                "vocab_size": 32000,
                "max_position_embeddings": 4096,
                "torch_dtype": "bfloat16",
            }
        )
    )
    plan = build_collection_plan(
        backend="vllm",
        model_path=str(tmp_path),
        model_architecture="DeepSeekForCausalLM",
        system="b200_sxm",
        selected_ops={"mla_context_module", "mla_generation_module"},
        has_model_cases=True,
        options=FPMCollectionOptions.from_args(_args()),
    )

    assert plan.capability.support_level == "exact"
    assert plan.capability.attention_kind == "dense_mla"
    assert plan.capability.is_moe is False
    assert {cell.parallel_strategy for cell in plan.cells} == {"tp"}


def test_explicit_kv_dtype_still_requires_aic_runtime_capability():
    options = FPMCollectionOptions.from_args(_args(fpm_kv_cache_dtypes=["int8"]))
    with pytest.raises(ValueError, match="does not support KV-cache dtype"):
        build_collection_plan(
            backend="vllm",
            model_path="nvidia/GLM-5.2-NVFP4",
            system="b200_sxm",
            selected_ops={"dsa_context_module", "dsa_generation_module"},
            options=options,
        )


def test_arbitrary_backend_variant_is_not_a_capability_declaration():
    with pytest.raises(ValueError, match="no longer an admission mechanism"):
        build_collection_plan(
            backend="vllm",
            model_path="nvidia/GLM-5.2-NVFP4",
            system="b200_sxm",
            selected_ops={"dsa_context_module", "dsa_generation_module"},
            options=FPMCollectionOptions.from_args(_args()),
            collector_config={"backend_variants": {"moe": [{"id": "invented"}]}},
        )


def _synthetic_plan_and_cell(tmp_path):
    topology = ParallelTopology(tp=1, pp=1, dp=2, moe_tp=1, moe_ep=2, cp=1)
    cell = FPMCell(
        cell_id="fpm-test",
        workload_kind="prefill",
        topology=topology,
        weight_quantization="nvfp4",
        kv_cache_dtype="fp8",
        backend_policy=BackendPolicy("baseline", "baseline_auto", {}, {}),
        parallel_strategy="dep",
        gemm_quant_mode="nvfp4",
        moe_quant_mode="nvfp4",
        fmha_quant_mode="fp8",
        comm_quant_mode="half",
    )
    plan = SimpleNamespace(
        sha256="plan-sha",
        aic_revision="revision",
        model_path="org/model",
        system="b200_sxm",
        backend="vllm",
        options=SimpleNamespace(warmup_iterations=0),
        capability=SimpleNamespace(
            support_level="exact",
            template_id="aic_exact:dsa_module",
            template_version=1,
            aic_database_version="0.24.0",
        ),
    )
    point = {
        "point_type": "prefill",
        "benchmark_id": 1,
        "total_prefill_tokens": 257,
        "total_kv_read_tokens": 128,
        "batch_size": 4,
        "expected_cudagraph_mode": "PIECEWISE",
        "expected_capture_size": 272,
        "padding_tokens": 15,
        "sample_reasons": ["post_capture"],
    }
    cell_dir = tmp_path / "cell"
    rank_fpms = []
    for rank, latency in ((0, 0.004), (1, 0.006)):
        rank_fpms.append(
            {
                "counter_id": 1,
                "dp_rank": rank,
                "wall_time": latency,
                "scheduled_requests": {
                    "num_prefill_requests": 4,
                    "sum_prefill_tokens": 257,
                    "sum_prefill_kv_tokens": 128,
                    "num_decode_requests": 0,
                    "sum_decode_kv_tokens": 0,
                },
            }
        )
    iteration_group = {
        "benchmark_id": 1,
        "point": point,
        "expected_dp_ranks": [0, 1],
        "complete": True,
        "wall_time": 0.006,
        "rank_results": [{"dp_rank": rank, "fpms": [fpm]} for rank, fpm in enumerate(rank_fpms)],
    }
    for rank, fpm in enumerate(rank_fpms):
        output = cell_dir / "raw" / f"pod-{rank}" / ("benchmark.json" if rank == 0 else f"benchmark_dp{rank}.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_provenance(output.parent / "collector-provenance.json", cell_id=cell.cell_id)
        output.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "artifact_type": "rank",
                    "status": "complete",
                    "valid": True,
                    "usable": True,
                    "timing_valid": True,
                    "stop_reason": None,
                    "error": None,
                    "run_id": "run",
                    "grid_digest": "grid",
                    "config": {"mode": "prefill"},
                    "coverage": {"expected_points": 1, "completed_points": 1, "skipped_points": 0},
                    "dp": {"rank": rank, "size": 2},
                    "results": [{"point": point, "fpms": [fpm]}],
                    "iteration_groups": [iteration_group],
                    "skipped_points": [],
                    "missing_phases": [],
                    "timing": {
                        "benchmark_elapsed_seconds": 1.0 + rank,
                        "measured_iteration_seconds": 0.006,
                    },
                }
            )
        )
    return plan, cell, cell_dir


def test_native_aggregation_preserves_iteration_totals(tmp_path):
    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    rows = aggregate_cell(plan, cell, cell_dir)

    assert len(rows) == 1
    assert rows[0]["latency_ms"] == pytest.approx(6.0)
    assert rows[0]["batch_size"] == 4
    assert rows[0]["total_prefill_tokens"] == 257
    assert rows[0]["total_kv_read_tokens"] == 128
    assert rows[0]["partition_policy"] == "balanced_v1"
    assert rows[0]["measurement_policy"] == "dynamo_native_single_sample_v1"
    assert rows[0]["backend_version"] == "0.24.0"
    assert "suffix_length" not in rows[0]
    assert "prefix_length" not in rows[0]


def test_native_aggregation_rejects_rank_grid_drift(tmp_path):
    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    second_rank = next((cell_dir / "raw" / "pod-1").glob("benchmark*.json"))
    payload = json.loads(second_rank.read_text())
    payload["grid_digest"] = "different"
    second_rank.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="different run identities"):
        aggregate_cell(plan, cell, cell_dir)


def test_native_aggregation_rejects_stale_collector_attempt(tmp_path):
    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    second_provenance = cell_dir / "raw" / "pod-1" / "collector-provenance.json"
    payload = json.loads(second_provenance.read_text())
    payload["attempt_id"] = "stale-attempt"
    second_provenance.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="provenance differs across pods"):
        aggregate_cell(plan, cell, cell_dir)


def test_formal_database_uses_schema_v4_and_rejects_conflicts(tmp_path):
    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    rows = aggregate_cell(plan, cell, cell_dir)
    parquet, metadata = write_formal_database(plan, rows, systems_root=tmp_path / "systems")
    write_formal_database(plan, rows, systems_root=tmp_path / "systems")

    assert parquet.exists()
    metadata_payload = json.loads(metadata.read_text())
    assert metadata_payload["schema_version"] == 4
    assert metadata_payload["coordinate_system"] == "iteration_totals_balanced_v1"
    assert metadata_payload["backend_version"] == "0.24.0"
    assert parquet.parent.name == "0.24.0"

    conflicting = [{**rows[0], "latency_ms": 7.0}]
    with pytest.raises(ValueError, match="conflicting"):
        write_formal_database(plan, conflicting, systems_root=tmp_path / "systems")


def test_formal_database_serializes_concurrent_publishers(tmp_path):
    plan, _cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    base_row = aggregate_cell(plan, _cell, cell_dir)[0]
    rows = [
        {
            **base_row,
            "cell_id": f"fpm-concurrent-{index}",
            "source_plan_sha256": f"plan-{index}",
        }
        for index in range(4)
    ]
    systems_root = tmp_path / "systems"
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    processes = [
        context.Process(
            target=_concurrent_database_writer,
            args=(str(systems_root), row, start_event),
        )
        for row in rows
    ]
    try:
        for process in processes:
            process.start()
        start_event.set()
        for process in processes:
            process.join(timeout=30)
        assert [process.exitcode for process in processes] == [0] * len(processes)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    import pyarrow.parquet as pq

    destination = systems_root / "b200_sxm" / "vllm" / "0.24.0"
    table = pq.read_table(destination / "fpm_forward_perf.parquet")
    metadata = json.loads((destination / "fpm_forward_perf.metadata.json").read_text())
    assert table.num_rows == len(rows)
    assert metadata["row_count"] == len(rows)
