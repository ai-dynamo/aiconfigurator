# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

from collector.fpm_forward.config import FPMCollectionOptions
from collector.fpm_forward.database import aggregate_cell, write_formal_database
from collector.fpm_forward.planner import (
    RUNTIME_OVERLAY_FILES,
    BackendPolicy,
    FPMCell,
    _runtime_overlay_files,
    build_collection_plan,
)
from collector.fpm_forward.population import build_attention_population
from collector.fpm_forward.sampling import build_sampling_design
from collector.fpm_forward.topology import enumerate_fpm_topologies
from collector.fpm_forward.types import FPMPoint, ParallelTopology
from collector.model_cases import build_collection_case_plan

pytestmark = pytest.mark.unit


def _args(**overrides):
    values = {
        "fpm_max_gpus": 4,
        "fpm_gpu_counts": [4],
        "fpm_parallel_presets": None,
        "fpm_parallel_axes": None,
        "fpm_backend_axes": None,
        "fpm_weight_quantizations": None,
        "fpm_kv_cache_dtypes": None,
        "fpm_sampling_budget": "one_quarter",
        "fpm_tp_sizes": None,
        "fpm_pp_sizes": None,
        "fpm_dp_sizes": None,
        "fpm_moe_tp_sizes": None,
        "fpm_smoke_points": None,
        "fpm_moe_ep_sizes": None,
        "fpm_cp_sizes": None,
        "fpm_kv_block_size": 64,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_options_freeze_single_sample_policy_and_gpu_bound():
    options = FPMCollectionOptions.from_args(_args())
    assert options.warmup_repeats == 1
    assert options.measurement_repeats == 1
    assert options.gpu_counts == (4,)
    assert options.parallel_presets == ("auto",)

    with pytest.raises(ValueError, match="exceed"):
        FPMCollectionOptions.from_args(_args(fpm_gpu_counts=[4, 8]))


def test_runtime_overlay_is_complete_and_hash_frozen(tmp_path):
    for name in RUNTIME_OVERLAY_FILES:
        (tmp_path / name).write_text(f"# {name}\n")
    original_sha256 = dict.fromkeys(RUNTIME_OVERLAY_FILES, "0" * 64)

    manifest, base_manifest = _runtime_overlay_files(
        {
            "runtime_overlay_dir": str(tmp_path),
            "runtime_overlay_original_sha256": original_sha256,
        }
    )

    assert [name for name, _sha256 in manifest] == list(RUNTIME_OVERLAY_FILES)
    assert all(len(sha256) == 64 for _name, sha256 in manifest)
    assert dict(base_manifest) == original_sha256

    (tmp_path / RUNTIME_OVERLAY_FILES[0]).unlink()
    with pytest.raises(ValueError, match="missing"):
        _runtime_overlay_files(
            {
                "runtime_overlay_dir": str(tmp_path),
                "runtime_overlay_original_sha256": original_sha256,
            }
        )


def test_prefill_sampling_is_deterministic_nested_and_latency_blind():
    points = tuple(
        FPMPoint("prefill", batch, suffix, prefix)
        for prefix in (0, 64, 128)
        for batch in (1, 2, 4)
        for suffix in (16, 64, 256)
    )
    first = build_sampling_design(points, block_size=64, active_budget="one_quarter")
    second = build_sampling_design(points, block_size=64, active_budget="one_quarter")
    assert first.sha256 == second.sha256
    assert first.ordered_points == second.ordered_points
    assert set(first.selected).isdisjoint(first.reserve)
    assert set(first.selected) | set(first.reserve) == set(points)
    counts = first.budget_counts
    assert counts["one_eighth"] <= counts["one_quarter"] <= counts["one_half"] <= counts["full"]
    assert first.to_dict()["latency_labels_used"] is False


def test_decode_sampling_uses_only_aic_population_members():
    points = tuple(FPMPoint("decode", batch, 1, length) for batch in (1, 2, 4, 8) for length in (64, 256, 1024, 4096))
    design = build_sampling_design(points, block_size=64, active_budget="one_half")
    assert len(design.ordered_points) == len(points)
    assert set(design.ordered_points) == set(points)


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


def test_plan_identity_includes_generator_execution_config():
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
    first = build_collection_plan(**kwargs, generator_overrides={"params": {"agg": {"max_seq_len": 66}}})
    second = build_collection_plan(**kwargs, generator_overrides={"params": {"agg": {"max_seq_len": 65537}}})

    assert first.generator_config_sha256 != second.generator_config_sha256
    assert first.sha256 != second.sha256
    assert first.capability.support_level == "exact"
    assert first.capability.attention_kind == "moe_dsa"
    assert first.dtype_profile.gemm_quant_mode == "nvfp4"
    assert first.dtype_profile.kv_cache_dtypes == ("fp8",)
    assert {cell.parallel_strategy for cell in first.cells} == {"dep"}


def test_glm_auto_matrix_includes_capacity_admitted_pure_tp():
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


def test_minimax_m3_uses_dsa_transfer_population_and_registered_pure_tp():
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
    assert plan.population.source_ops == ("dsa_context_module", "dsa_generation_module")
    assert {cell.parallel_strategy for cell in plan.cells} == {"pure_tp", "tep", "dep"}


_DSV4_ATTENTION_OPS = {
    "dsv4_csa_context_module",
    "dsv4_hca_context_module",
    "dsv4_csa_generation_module",
    "dsv4_hca_generation_module",
}


@pytest.mark.parametrize(
    "model_path",
    [
        "sgl-project/DeepSeek-V4-Pro-FP8",
        "sgl-project/DeepSeek-V4-Flash-FP8",
    ],
)
def test_dsv4_fp8_uses_exact_compressed_attention_population(model_path):
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
    assert plan.capability.allow_pure_tp is False
    assert plan.dtype_profile.fmha_quant_mode == "bfloat16"
    assert plan.dtype_profile.kv_cache_dtypes == ("fp8",)
    assert set(plan.population.source_ops) == _DSV4_ATTENTION_OPS
    assert FPMPoint("prefill", 1, 64, 65472) in plan.population.prefill_points
    assert FPMPoint("decode", 1, 1, 65536) in plan.population.decode_points
    assert {cell.parallel_strategy for cell in plan.cells} == {"tep", "dep"}


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


def test_unknown_model_uses_auditable_bootstrap_template(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["NewMoeDsaForCausalLM"],
                "num_attention_heads": 32,
                "num_key_value_heads": 32,
                "head_dim": 128,
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
    assert plan.capability.attention_source == "dsa_module"
    assert plan.population.source_name == "dsa_module"
    assert {cell.parallel_strategy for cell in plan.cells} == {"dep", "tep"}


def test_unregistered_phi4_mini_uses_dense_gqa_bootstrap_template(tmp_path):
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

    assert plan.capability.architecture == "Phi3ForCausalLM"
    assert plan.capability.model_family is None
    assert plan.capability.support_level == "bootstrap_template"
    assert plan.capability.template_id == "generic:dense_gqa"
    assert plan.capability.attention_source == "dense_attention"
    assert plan.capability.allow_pure_tp is False
    assert {cell.parallel_strategy for cell in plan.cells} == {"tp"}


def test_registered_model_without_case_file_uses_family_template():
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


def test_explicit_kv_dtype_requires_aic_attention_data():
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
    options = FPMCollectionOptions.from_args(_args())
    with pytest.raises(ValueError, match="no longer an admission mechanism"):
        build_collection_plan(
            backend="vllm",
            model_path="nvidia/GLM-5.2-NVFP4",
            system="b200_sxm",
            selected_ops={"dsa_context_module", "dsa_generation_module"},
            options=options,
            collector_config={"backend_variants": {"moe": [{"id": "invented"}]}},
        )


@pytest.mark.parametrize(
    ("model_path", "expected_source"),
    [
        ("nvidia/GLM-5.2-NVFP4", "dsa_module"),
        ("Qwen/Qwen3-32B", "dense_attention"),
    ],
)
def test_population_uses_model_selected_aic_attention_family(model_path, expected_source):
    case_plan = build_collection_case_plan(backend="vllm", model_path=model_path, gpu_type="b200_sxm")
    population = build_attention_population(
        backend="vllm",
        model_path=model_path,
        selected_ops=case_plan.selected_ops,
        kv_block_size=64,
    )
    assert population.source_name == expected_source
    assert population.prefill_points
    assert population.decode_points
    assert any(point.prefix_length > 0 for point in population.prefill_points)


def _synthetic_plan_and_cell(tmp_path):
    topology = ParallelTopology(tp=1, pp=1, dp=2, moe_tp=1, moe_ep=2, cp=1)
    cell = FPMCell(
        cell_id="fpm-test",
        workload_kind="prefill",
        topology=topology,
        weight_quantization="nvfp4",
        kv_cache_dtype="fp8_e4m3",
        backend_policy=BackendPolicy("baseline", "baseline_auto", {}, {}),
        sampling_sha256="sampling",
        parallel_strategy="dep",
        gemm_quant_mode="nvfp4",
        moe_quant_mode="nvfp4",
        fmha_quant_mode="fp8",
        comm_quant_mode="half",
    )
    point = FPMPoint("prefill", 2, 64, 128)
    plan = SimpleNamespace(
        sha256="plan-sha",
        aic_revision="revision",
        model_path="org/model",
        system="b200_sxm",
        backend="vllm",
        capability=SimpleNamespace(
            support_level="exact",
            template_id="aic_exact:dsa_module",
            template_version=1,
            aic_database_version="0.24.0",
        ),
        prefill_design=SimpleNamespace(selected=(point,)),
        decode_design=SimpleNamespace(selected=()),
    )
    cell_dir = tmp_path / "cell"
    for rank, latency in ((0, 0.004), (1, 0.006)):
        output = cell_dir / "raw" / f"pod-{rank}" / ("benchmark.json" if rank == 0 else f"benchmark_dp{rank}.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        fpm = {
            "dp_rank": rank,
            "wall_time": latency,
            "scheduled_requests": {
                "num_prefill_requests": 2,
                "sum_prefill_tokens": 128,
                "sum_prefill_kv_tokens": 256,
                "num_decode_requests": 0,
                "sum_decode_kv_tokens": 0,
            },
        }
        output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "complete",
                    "valid": True,
                    "collector": {
                        "plan_sha256": "plan-sha",
                        "cell_id": "fpm-test",
                        "warmup_repeats": 1,
                        "measured_repeats": 1,
                    },
                    "campaign_results": [
                        {
                            "point": point.to_dict(),
                            "warmup_fpms": [{**fpm, "wall_time": latency * 2}],
                            "fpms": [fpm],
                        }
                    ],
                }
            )
        )
    return plan, cell, cell_dir


def test_single_measurement_dp_aggregation_uses_max_rank(tmp_path):
    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    rows = aggregate_cell(plan, cell, cell_dir)
    assert len(rows) == 1
    assert rows[0]["latency_ms"] == pytest.approx(6.0)
    assert rows[0]["measurement_policy"] == "single_sample_v1"
    assert rows[0]["parallel_strategy"] == "dep"
    assert rows[0]["gemm_quant_mode"] == "nvfp4"
    assert rows[0]["moe_quant_mode"] == "nvfp4"
    assert rows[0]["fmha_quant_mode"] == "fp8"
    assert rows[0]["comm_quant_mode"] == "half"
    assert rows[0]["model_support_level"] == "exact"


def test_formal_database_is_idempotent_and_rejects_conflicts(tmp_path):
    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    rows = aggregate_cell(plan, cell, cell_dir)
    parquet, metadata = write_formal_database(plan, rows, systems_root=tmp_path / "systems")
    write_formal_database(plan, rows, systems_root=tmp_path / "systems")
    assert parquet.exists()
    metadata_payload = json.loads(metadata.read_text())
    assert metadata_payload["measurement_repeats"] == 1
    assert metadata_payload["schema_version"] == 2

    conflicting = [{**rows[0], "latency_ms": 7.0}]
    with pytest.raises(ValueError, match="conflicting"):
        write_formal_database(plan, conflicting, systems_root=tmp_path / "systems")
