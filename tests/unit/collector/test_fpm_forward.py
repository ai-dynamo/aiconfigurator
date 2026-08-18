# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiconfigurator.sdk.utils import HuggingFaceDownloadError
from collector.fpm_forward.capabilities import resolve_model_capability
from collector.fpm_forward.config import FPMCollectionOptions, PrefillSamplingProfile, add_fpm_arguments
from collector.fpm_forward.database import aggregate_cell, write_formal_database
from collector.fpm_forward.memory_admission import filter_memory_infeasible_topologies
from collector.fpm_forward.model_capability import load_model_config
from collector.fpm_forward.planner import (
    BackendPolicy,
    FPMCell,
    backend_identity_columns,
    build_collection_plan,
)
from collector.fpm_forward.topology import enumerate_fpm_topologies
from collector.fpm_forward.types import ParallelTopology

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _pinned_git_revision(monkeypatch):
    """Keep plan-identity tests hermetic: no git binary or checkout required
    (CI test containers ship without git), and plan hashes must not depend on
    the ambient repository HEAD. Yields the real function for the tests that
    exercise revision derivation itself."""

    from collector.fpm_forward import planner as planner_module

    real = planner_module._git_revision
    monkeypatch.setattr(planner_module, "_git_revision", lambda: "test-revision")
    yield real


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
        "fpm_moe_backend": None,
        "fpm_attention_backend": None,
        "fpm_enable_wideep": None,
        "fpm_enable_eplb": None,
        "fpm_strict_admission": None,
        "fpm_weight_quantizations": None,
        "fpm_kv_cache_dtypes": None,
        "fpm_model_config": None,
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
    assert "--fpm-model-config" in help_text
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
    assert payload["schema_version"] == 10
    assert payload["capability"]["model_config"]["source_kind"] == "aic_cache"
    assert len(payload["capability"]["model_config"]["sha256"]) == 64
    assert payload["capability"]["model_config"]["payload"]["architectures"] == ["GlmMoeDsaForCausalLM"]
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
    policy = BackendPolicy("baseline", source, {"runtime.mode": "FULL"})
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


def test_memory_admission_drops_only_the_rejected_dtype_cells(monkeypatch, caplog):
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
    # Per-dtype drops must be counted, never silent (collector rules: the
    # memory filter's drops are logged; whole-topology logging alone would
    # hide these).
    assert "fpm_forward: dropped 3/6 (topology, kv_dtype) cell groups (memory budget" in caplog.text


def test_plan_identity_ignores_memory_estimator_error_text(monkeypatch):
    """Estimator failure diagnostics vary across runs and hosts (transient
    network errors, host paths); hashing them would spuriously invalidate
    resume for identical plans. Only dispositions belong to the identity."""

    kwargs = {
        "backend": "vllm",
        "model_path": "nvidia/GLM-5.2-NVFP4",
        "model_architecture": "GlmMoeDsaForCausalLM",
        "system": "b200_sxm",
        "selected_ops": {"dsa_context_module", "dsa_generation_module"},
        "options": FPMCollectionOptions.from_args(_args()),
    }

    def failing(message):
        def from_request(*_args, **_kwargs):
            raise RuntimeError(message)

        return from_request

    monkeypatch.setattr(
        "collector.fpm_forward.memory_admission.KVCacheEstimator.from_request",
        failing("connection timed out to huggingface.co"),
    )
    first = build_collection_plan(**kwargs)
    monkeypatch.setattr(
        "collector.fpm_forward.memory_admission.KVCacheEstimator.from_request",
        failing("HTTP Error 503: Service Unavailable under /Users/someone"),
    )
    second = build_collection_plan(**kwargs)

    assert first.sha256 == second.sha256
    # The human-readable plan still carries the full diagnostic.
    assert "connection timed out to huggingface.co" in json.dumps(first.to_dict())


def test_git_revision_folds_dirty_tracked_state_into_identity(_pinned_git_revision, monkeypatch):
    """Uncommitted tracked edits change collector behavior without moving
    HEAD; the revision must distinguish them, resume identically for an
    unchanged tree, and ignore untracked noise (artifact dirs)."""

    from collector.fpm_forward import planner as fpm_planner

    real_git_revision = _pinned_git_revision
    outputs = {
        ("rev-parse", "HEAD"): "abc123\n",
        ("status", "--porcelain", "--untracked-files=no"): "",
        ("diff-index", "--no-ext-diff", "--full-index", "-p", "HEAD"): "",
    }

    def fake_run(args, **_kwargs):
        return SimpleNamespace(stdout=outputs[tuple(args[1:])], returncode=0)

    monkeypatch.setattr(fpm_planner.subprocess, "run", fake_run)

    assert real_git_revision() == "abc123"

    outputs[("status", "--porcelain", "--untracked-files=no")] = " M collector/fpm_forward/runner.py\n"
    outputs[("diff-index", "--no-ext-diff", "--full-index", "-p", "HEAD")] = "-a\n+b\n"
    dirty = real_git_revision()
    assert dirty.startswith("abc123-dirty-")
    assert real_git_revision() == dirty

    outputs[("diff-index", "--no-ext-diff", "--full-index", "-p", "HEAD")] = "-a\n+c\n"
    changed = real_git_revision()
    assert changed.startswith("abc123-dirty-")
    assert changed != dirty


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
def test_dsv4_native_fp4_uses_vllm_sm100_dtype_capability(model_path):
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
    assert plan.capability.allow_pure_tp is True
    assert plan.dtype_profile.gemm_quant_mode == "fp8_block"
    assert plan.dtype_profile.moe_quant_mode == "w4a8_mxfp4_mxfp8"
    assert plan.dtype_profile.fmha_quant_mode == "bfloat16"
    assert plan.dtype_profile.kv_cache_dtypes == ("fp8",)
    assert {cell.parallel_strategy for cell in plan.cells} == {"pure_tp", "tep", "dep"}
    assert all(cell.to_dict()["point_source"] == "dynamo_native_self_benchmark" for cell in plan.cells)


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


def _unknown_fp8_moe_config(*, hidden_size: int = 4096) -> dict[str, object]:
    return {
        "architectures": ["NewMoeDsaForCausalLM"],
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "hidden_size": hidden_size,
        "intermediate_size": 8192,
        "num_hidden_layers": 4,
        "vocab_size": 32000,
        "n_routed_experts": 8,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "max_position_embeddings": 4096,
    }


def test_explicit_real_config_keeps_unregistered_fp8_moe_bootstrap(monkeypatch, tmp_path):
    config_dir = tmp_path / "private-model-config"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(json.dumps(_unknown_fp8_moe_config()))
    (config_dir / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "FP8", "kv_cache_quant_algo": "FP8"}})
    )

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("the checkpoint is accessible only inside the runtime Pod")

    monkeypatch.setattr(
        "collector.fpm_forward.memory_admission.KVCacheEstimator.from_request",
        unavailable,
    )
    plan = build_collection_plan(
        backend="vllm",
        model_path="private-org/runtime-only-model",
        model_config_path=str(config_dir),
        model_architecture="NewMoeDsaForCausalLM",
        system="b200_sxm",
        selected_ops={"attention_context", "attention_generation"},
        has_model_cases=False,
        options=FPMCollectionOptions.from_args(_args()),
    )

    evidence = plan.capability.model_config
    assert evidence.source_kind == "explicit"
    assert len(evidence.sha256) == 64
    assert evidence.payload["hf_quant_config"]["quantization"]["quant_algo"] == "FP8"
    detached = evidence.payload
    detached["architectures"] = ["MutatedForCausalLM"]
    assert evidence.payload["architectures"] == ["NewMoeDsaForCausalLM"]
    assert plan.capability.support_level == "bootstrap_template"
    assert plan.capability.is_moe is True
    assert plan.dtype_profile.gemm_quant_mode == "fp8_static"
    assert plan.dtype_profile.moe_quant_mode == "fp8"
    assert plan.dtype_profile.kv_cache_dtypes == ("fp8",)


def test_huggingface_config_is_used_when_model_is_not_in_aic_cache(monkeypatch):
    calls = []

    def download(model_path, filename, *, raise_on_404):
        calls.append((model_path, filename, raise_on_404))
        if filename == "config.json":
            return _unknown_fp8_moe_config()
        return None

    monkeypatch.setattr("collector.fpm_forward.model_capability._download_hf_json", download)

    resolved = load_model_config("new-org/New-MoE-Model")

    assert resolved.source_kind == "huggingface"
    assert resolved.payload["architectures"] == ["NewMoeDsaForCausalLM"]
    assert calls == [
        ("new-org/New-MoE-Model", "config.json", True),
        ("new-org/New-MoE-Model", "hf_quant_config.json", False),
    ]


def test_unresolvable_model_config_fails_loudly(monkeypatch):
    def missing(*_args, **_kwargs):
        raise HuggingFaceDownloadError("Hugging Face returned HTTP error 404")

    monkeypatch.setattr("collector.fpm_forward.model_capability._download_hf_json", missing)

    with pytest.raises(ValueError, match="cannot resolve a real model config"):
        build_collection_plan(
            backend="vllm",
            model_path="missing-org/does-not-exist",
            model_architecture="MissingForCausalLM",
            system="b200_sxm",
            selected_ops={"attention_context", "attention_generation"},
            has_model_cases=False,
            options=FPMCollectionOptions.from_args(_args()),
        )


def test_empty_explicit_model_config_is_rejected(tmp_path):
    config = tmp_path / "config.json"
    config.write_text("{}\n")

    with pytest.raises(ValueError, match="must not be empty"):
        load_model_config("private-org/model", explicit_config_path=str(config))


def test_empty_base_config_with_sibling_quant_config_is_rejected(tmp_path):
    """The emptiness gate must fire BEFORE the quantization merge: a sibling
    hf_quant_config.json would otherwise make an empty config.json non-empty
    and admit a contentless model identity into the frozen plan."""

    (tmp_path / "config.json").write_text("{}\n")
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "FP8", "kv_cache_quant_algo": "FP8"}})
    )

    with pytest.raises(ValueError, match="must not be empty"):
        load_model_config(str(tmp_path))


def test_empty_huggingface_config_with_quant_config_is_rejected(monkeypatch):
    def download(_model_path, filename, *, raise_on_404):
        if filename == "config.json":
            return {}
        return {"quantization": {"quant_algo": "FP8"}}

    monkeypatch.setattr("collector.fpm_forward.model_capability._download_hf_json", download)

    with pytest.raises(ValueError, match="must not be empty"):
        load_model_config("new-org/empty-config-model")


def test_empty_kv_dtype_request_fails_before_any_resolution():
    """The CLI maps an empty --fpm-kv-cache-dtypes to ("auto",); a direct
    caller with an empty tuple must get an actionable error instead of a
    bare IndexError after config resolution."""

    with pytest.raises(ValueError, match="at least one KV-cache dtype"):
        resolve_model_capability(
            backend="vllm",
            model_path="nvidia/GLM-5.2-NVFP4",
            model_architecture="GlmMoeDsaForCausalLM",
            selected_ops={"dsa_context_module", "dsa_generation_module"},
            has_model_cases=True,
            system="b200_sxm",
            requested_weight_quantizations=(),
            requested_kv_cache_dtypes=(),
        )


def test_memory_admission_fails_loudly_on_capability_invariant_violation():
    """A kv dtype without an fmha mapping violates resolve_model_capability's
    invariant; it must raise instead of being recorded as a fail-open
    'unknown' memory-estimate outcome."""

    capability = SimpleNamespace(
        aic_database_version="0.24.0",
        dtype=SimpleNamespace(
            gemm_quant_mode="nvfp4",
            moe_quant_mode="nvfp4",
            comm_quant_mode="half",
            kv_cache_dtypes=("fp8",),
            fmha_by_kv_dtype={},
        ),
    )

    with pytest.raises(ValueError, match="no fmha mapping"):
        filter_memory_infeasible_topologies(
            backend="vllm",
            model_path="org/model",
            system="b200_sxm",
            capability=capability,
            topologies=(ParallelTopology(tp=1, pp=1, dp=1, moe_tp=1, moe_ep=1, cp=1),),
            max_new_tokens=8192,
        )


def test_model_config_content_changes_the_frozen_plan_hash(monkeypatch, tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    same_as_first = tmp_path / "same-as-first.json"
    first.write_text(json.dumps(_unknown_fp8_moe_config(hidden_size=4096)))
    second.write_text(json.dumps(_unknown_fp8_moe_config(hidden_size=5120)))
    same_as_first.write_text(json.dumps(_unknown_fp8_moe_config(hidden_size=4096)))

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("memory estimate intentionally unavailable")

    monkeypatch.setattr(
        "collector.fpm_forward.memory_admission.KVCacheEstimator.from_request",
        unavailable,
    )
    common = {
        "backend": "vllm",
        "model_path": "private-org/runtime-only-model",
        "model_architecture": "NewMoeDsaForCausalLM",
        "system": "b200_sxm",
        "selected_ops": {"attention_context", "attention_generation"},
        "has_model_cases": False,
        "options": FPMCollectionOptions.from_args(_args()),
    }

    first_plan = build_collection_plan(**common, model_config_path=str(first))
    second_plan = build_collection_plan(**common, model_config_path=str(second))
    same_content_plan = build_collection_plan(**common, model_config_path=str(same_as_first))

    assert first_plan.capability.model_config.sha256 != second_plan.capability.model_config.sha256
    assert first_plan.sha256 != second_plan.sha256
    assert first_plan.sha256 == same_content_plan.sha256


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
        backend_policy=BackendPolicy("baseline_auto", {}, {}),
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
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")

    assert len(rows) == 1
    assert rows[0]["latency_ms"] == pytest.approx(6.0)
    assert rows[0]["batch_size"] == 4
    assert rows[0]["total_prefill_tokens"] == 257
    assert rows[0]["total_kv_read_tokens"] == 128
    assert rows[0]["partition_policy"] == "balanced_v1"
    assert rows[0]["measurement_policy"] == "dynamo_native_single_sample_v1"
    assert rows[0]["backend_version"] == "0.24.0"
    assert rows[0]["collector_attempt_id"] == "attempt"
    assert rows[0]["runtime_run_id"] == "run"
    assert rows[0]["runtime_grid_digest"] == "grid"
    assert "suffix_length" not in rows[0]
    assert "prefix_length" not in rows[0]


def test_native_aggregation_rejects_rank_grid_drift(tmp_path):
    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    second_rank = next((cell_dir / "raw" / "pod-1").glob("benchmark*.json"))
    payload = json.loads(second_rank.read_text())
    payload["grid_digest"] = "different"
    second_rank.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="different run identities"):
        aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")


def test_native_aggregation_rejects_stale_collector_attempt(tmp_path):
    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    for provenance in (cell_dir / "raw").glob("*/collector-provenance.json"):
        payload = json.loads(provenance.read_text())
        payload["attempt_id"] = "stale-attempt"
        provenance.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="attempt mismatch"):
        aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")


def test_native_aggregation_requires_attempt_identity(tmp_path):
    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)

    with pytest.raises(ValueError, match="without an expected Collector attempt"):
        aggregate_cell(plan, cell, cell_dir, expected_attempt_id="")


@pytest.mark.parametrize("bad_value", [4.0, True, "4"], ids=["float", "bool", "str"])
def test_native_validation_rejects_non_integer_point_dimensions(tmp_path, bad_value):
    """Native grid coordinates are integers by contract; a float/bool/str
    dimension means a noncompliant engine or tampered artifact and must fail
    validation instead of being silently truncated into a database row."""

    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    artifact = cell_dir / "raw" / "pod-0" / "benchmark.json"
    payload = json.loads(artifact.read_text())
    payload["results"][0]["point"]["batch_size"] = bad_value
    artifact.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="'batch_size' must be an integer"):
        aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")


def test_native_validation_rejects_sub_batch_token_totals(tmp_path):
    """Every scheduled request contributes at least one prefill token (prefill)
    or reads at least one KV token (decode), so totals below the request count
    are impossible coordinates and must fail instead of becoming database rows."""

    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    artifact = cell_dir / "raw" / "pod-0" / "benchmark.json"
    payload = json.loads(artifact.read_text())
    payload["results"][0]["point"]["total_prefill_tokens"] = 3
    artifact.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="fewer tokens than requests"):
        aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")

    # The decode floor cannot be reached through the synthetic prefill cell
    # (workload-kind contract fires first), so exercise the guard directly.
    from collector.fpm_forward.native_artifact import _expected_scheduled

    decode_point = {
        "point_type": "decode",
        "batch_size": 4,
        "total_prefill_tokens": 0,
        "total_kv_read_tokens": 2,
    }
    with pytest.raises(ValueError, match="fewer KV tokens than requests"):
        _expected_scheduled(decode_point)


def test_formal_database_uses_schema_v6_and_rejects_conflicts(tmp_path):
    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")
    parquet, metadata, skipped = write_formal_database(plan, rows, systems_root=tmp_path / "systems")
    write_formal_database(plan, rows, systems_root=tmp_path / "systems")

    assert parquet.exists()
    metadata_payload = json.loads(metadata.read_text())
    assert metadata_payload["schema_version"] == 6
    assert metadata_payload["coordinate_system"] == "iteration_totals_balanced_v1"
    assert metadata_payload["backend_version"] == "0.24.0"
    assert metadata_payload["collector_attempt_ids"] == ["attempt"]
    assert metadata_payload["runtime_run_ids"] == ["run"]
    assert metadata_payload["runtime_grid_digests"] == ["grid"]
    assert parquet.parent.name == "0.24.0"

    conflicting = [{**rows[0], "latency_ms": 7.0}]
    with pytest.raises(ValueError, match="conflicting"):
        write_formal_database(plan, conflicting, systems_root=tmp_path / "systems")


def test_formal_database_first_publisher_wins_on_rerun_overlap(tmp_path):
    """A cell republished under a different run identity is skipped whole
    (first publisher wins, sealed rows never mixed or overwritten) while the
    database file stays byte-stable and the skip is reported to the caller."""

    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")
    systems_root = tmp_path / "systems"
    parquet, _metadata, first_skipped = write_formal_database(plan, rows, systems_root=systems_root)
    assert first_skipped == ()
    sealed = parquet.read_bytes()

    rerun_rows = [
        {
            **rows[0],
            "total_prefill_tokens": rows[0]["total_prefill_tokens"] + 1,
            "latency_ms": 999.0,
            "collector_attempt_id": "different-attempt",
            "runtime_run_id": "different-run",
        }
    ]
    parquet2, _metadata2, skipped = write_formal_database(plan, rerun_rows, systems_root=systems_root)

    assert skipped == (rows[0]["cell_id"],)
    assert parquet2.read_bytes() == sealed


def test_formal_database_mixed_overlap_skips_sealed_cell_and_lands_the_rest(tmp_path):
    """A plan overlapping one sealed cell publishes its fresh cells while the
    sealed cell's rows stay first-publisher-owned: partial skip, rest lands."""

    import pyarrow.parquet as pq

    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")
    systems_root = tmp_path / "systems"
    write_formal_database(plan, rows, systems_root=systems_root)

    overlapping = {
        **rows[0],
        "latency_ms": 123.0,
        "collector_attempt_id": "attempt-2",
        "runtime_run_id": "run-2",
    }
    fresh = {
        **rows[0],
        "cell_id": "fpm-test-other",
        "batch_size": rows[0]["batch_size"] + 4,
        "collector_attempt_id": "attempt-2",
        "runtime_run_id": "run-2",
    }
    parquet, _metadata, skipped = write_formal_database(plan, [overlapping, fresh], systems_root=systems_root)

    assert skipped == (rows[0]["cell_id"],)
    published = pq.read_table(parquet).to_pylist()
    assert {row["cell_id"] for row in published} == {rows[0]["cell_id"], "fpm-test-other"}
    sealed_rows = [row for row in published if row["cell_id"] == rows[0]["cell_id"]]
    assert {row["collector_attempt_id"] for row in sealed_rows} == {"attempt"}
    fresh_rows = [row for row in published if row["cell_id"] == "fpm-test-other"]
    assert {row["collector_attempt_id"] for row in fresh_rows} == {"attempt-2"}


def test_formal_database_refuses_parquet_without_commit_record(tmp_path):
    """Published rows are sha-sealed evidence: a parquet without its metadata
    commit record could be a partial write or a foreign file, so the merge
    must refuse instead of building on unvouched bytes."""

    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")
    systems_root = tmp_path / "systems"
    _parquet, metadata, _skipped = write_formal_database(plan, rows, systems_root=systems_root)

    metadata.unlink()
    with pytest.raises(ValueError, match="no commit record"):
        write_formal_database(plan, rows, systems_root=systems_root)


def test_formal_database_refuses_parquet_that_mismatches_commit_record(tmp_path):
    """A parquet whose bytes disagree with the recorded sha256 (manual edit,
    torn write) must abort the merge instead of silently absorbing it."""

    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")
    systems_root = tmp_path / "systems"
    _parquet, metadata, _skipped = write_formal_database(plan, rows, systems_root=systems_root)

    committed = json.loads(metadata.read_text())
    committed["parquet_sha256"] = "0" * 64
    metadata.write_text(json.dumps(committed))
    with pytest.raises(ValueError, match="does not match its commit record"):
        write_formal_database(plan, rows, systems_root=systems_root)


def test_backend_marker_validation_tolerates_native_json_types(tmp_path):
    """Markers are declared as strings ("True") while the resolved config
    stores native JSON types (true); validation compares canonical string
    forms, so the type gap is not a mismatch but a real value gap still is."""

    from collector.fpm_forward.database import _validate_backend_markers

    cell = FPMCell(
        cell_id="fpm-test",
        workload_kind="prefill",
        topology=ParallelTopology(tp=1, pp=1, dp=2, moe_tp=1, moe_ep=2, cp=1),
        weight_quantization="nvfp4",
        kv_cache_dtype="fp8",
        backend_policy=BackendPolicy("eplb_pinned", {}, {"config.engine_args.enable_eplb": "True"}),
        parallel_strategy="dep",
        gemm_quant_mode="nvfp4",
        moe_quant_mode="nvfp4",
        fmha_quant_mode="fp8",
        comm_quant_mode="half",
    )
    cell_dir = tmp_path / "cell"
    config_path = cell_dir / "raw" / "pod-0" / "resolved-config-node0.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"config": {"engine_args": {"enable_eplb": True}}}))

    _validate_backend_markers(cell, cell_dir)

    config_path.write_text(json.dumps({"config": {"engine_args": {"enable_eplb": False}}}))
    with pytest.raises(ValueError, match="backend marker mismatch"):
        _validate_backend_markers(cell, cell_dir)


def test_formal_database_merge_gate_names_missing_row_key_columns(tmp_path):
    """An existing parquet that satisfies the run-identity columns but lacks a
    _ROW_KEY column must be rejected with the actionable schema ValueError,
    never a bare KeyError from the merge index."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")

    destination = tmp_path / "systems" / "b200_sxm" / "vllm" / "0.24.0"
    destination.mkdir(parents=True)
    stale_rows = [{key: value for key, value in rows[0].items() if key != "weight_quantization"}]
    parquet_path = destination / "fpm_forward_perf.parquet"
    pq.write_table(pa.Table.from_pylist(stale_rows), parquet_path)
    (destination / "fpm_forward_perf.metadata.json").write_text(
        json.dumps({"parquet_sha256": hashlib.sha256(parquet_path.read_bytes()).hexdigest()})
    )

    with pytest.raises(ValueError, match=r"missing columns: \['weight_quantization'\]"):
        write_formal_database(plan, rows, systems_root=tmp_path / "systems")


def test_formal_database_serializes_concurrent_publishers(tmp_path):
    plan, _cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    base_row = aggregate_cell(plan, _cell, cell_dir, expected_attempt_id="attempt")[0]
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


def test_minimax_m3_family_routing_survives_inherited_base_attention_ops():
    """include_base-inherited dense ops must not count as exact evidence."""

    plan = build_collection_plan(
        backend="vllm",
        model_path="MiniMaxAI/MiniMax-M3",
        model_architecture="MiniMaxM3ForCausalLM",
        system="b200_sxm",
        selected_ops={"attention_context", "attention_generation"},
        has_model_cases=True,
        options=FPMCollectionOptions.from_args(
            _args(
                fpm_max_gpus=16,
                fpm_gpu_counts=[8, 16],
            )
        ),
    )

    assert plan.capability.support_level == "family_template"
    assert plan.capability.template_id == "aic_family:minimaxm3:moe_msa"
    assert plan.capability.attention_source == "dsa_module"


def test_fmha_resolves_per_kv_dtype_against_joint_evidence(monkeypatch):
    """fp8 fmha exists only under the fp8 kv slice: bf16-kv cells must carry
    the bfloat16 transfer slice (recorded via fmha_resolution), never a flat
    fp8 label the database cannot serve jointly."""

    class Estimate:
        def __init__(self):
            self.breakdown = {"non_kv_bytes": 50, "gpu_memory_capacity_bytes": 100}

    monkeypatch.setattr(
        "collector.fpm_forward.memory_admission.KVCacheEstimator.from_request",
        lambda *_args, **_kwargs: Estimate(),
    )
    plan = build_collection_plan(
        backend="vllm",
        model_path="nvidia/GLM-5.2-NVFP4",
        model_architecture="GlmMoeDsaForCausalLM",
        system="b200_sxm",
        selected_ops={"dsa_context_module", "dsa_generation_module"},
        options=FPMCollectionOptions.from_args(_args(fpm_kv_cache_dtypes=["bfloat16", "fp8"])),
    )

    assert plan.dtype_profile.fmha_by_kv_dtype == {"bfloat16": "bfloat16", "fp8": "fp8"}
    assert plan.dtype_profile.fmha_resolution_by_kv_dtype == {
        # fp8 FMHA exists only with an fp8 KV cache, so the bf16-kv cells run
        # (and are labeled with) the engine's kv-coupled bfloat16 dispatch.
        "bfloat16": "kv_dtype_dispatch_from_fp8",
        "fp8": "checkpoint_native",
    }
    labels = {(cell.kv_cache_dtype, cell.fmha_quant_mode, cell.fmha_resolution) for cell in plan.cells}
    assert labels == {
        ("bfloat16", "bfloat16", "kv_dtype_dispatch_from_fp8"),
        ("fp8", "fp8", "checkpoint_native"),
    }


def test_formal_database_rejects_path_bearing_backend_versions(tmp_path):
    """backend_version comes from pod provenance; a path-bearing value must
    never become a directory component under the database root."""

    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")
    for row in rows:
        row["backend_version"] = "0.24.0/../../evil"

    with pytest.raises(ValueError, match="not a safe database directory name"):
        write_formal_database(plan, rows, systems_root=tmp_path / "systems")


def test_generator_overrides_reject_malformed_model_cache():
    from collector.fpm_forward.entry import _load_generator_overrides

    args = argparse.Namespace(
        generator_config=None,
        generator_set=None,
        generator_dynamo_version=None,
        generated_config_version=None,
        namespace=None,
        transport=None,
        image_pull_secret=None,
        model_cache="pvc:mount:sub:extra",
    )
    with pytest.raises(ValueError, match="NAME\\[:MOUNT\\[:SUBPATH\\]\\]"):
        _load_generator_overrides(args)


def test_formal_database_requires_family_measured_version_in_curated_tree(tmp_path, monkeypatch):
    """Default publication targets <system>/<backend>/<version> (the fpm
    forward-model consumer's path), but only for versions the curated tree
    already measures under a family dir: the SDK treats any populated version
    dir as a declared database, so an undeclared (or marker-only) version must
    not be materialized."""

    from collector.fpm_forward import database as fpm_database

    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")
    curated = tmp_path / "curated"
    monkeypatch.setattr(fpm_database, "_curated_systems_root", lambda: curated)

    # Nothing declares the version anywhere in the curated tree.
    with pytest.raises(ValueError, match="not a curated AIC database version"):
        write_formal_database(plan, rows, systems_root=None)

    # Evidence under a dot-prefixed dir is invisible to SDK version discovery
    # and must not count.
    hidden = curated / plan.system / ".backup" / plan.backend / "0.24.0"
    hidden.mkdir(parents=True)
    (hidden / "attention_perf.parquet").write_bytes(b"stub")
    with pytest.raises(ValueError, match="not a curated AIC database version"):
        write_formal_database(plan, rows, systems_root=None)

    # A marker-only family dir declares the version but holds no measured
    # data; publishing would flip a marker-only version into a "has data"
    # version and change op-level default-version resolution.
    family_dir = curated / plan.system / "attention" / plan.backend / "0.24.0"
    family_dir.mkdir(parents=True)
    (family_dir / "reuse.yaml").write_text("reuse: []\n")
    with pytest.raises(ValueError, match="not a curated AIC database version"):
        write_formal_database(plan, rows, systems_root=None)

    # Measured files in a mid-collection (partial) dir are vetoed too: the SDK
    # treats a partial version dir as undeclared regardless of its perf files,
    # so publishing would newly declare the version to version discovery.
    (family_dir / "attention_perf.parquet").write_bytes(b"stub")
    (family_dir / "collection_meta.yaml").write_text("tables:\n  attention:\n    status: partial\n")
    with pytest.raises(ValueError, match="not a curated AIC database version"):
        write_formal_database(plan, rows, systems_root=None)

    # A completed collection admits the version; publication creates the
    # two-level consumer path next to the family layout.
    (family_dir / "collection_meta.yaml").write_text("tables:\n  attention:\n    status: complete\n")
    parquet, _metadata, _skipped = write_formal_database(plan, rows, systems_root=None)
    assert parquet.is_file()
    assert parquet == curated / plan.system / plan.backend / "0.24.0" / "fpm_forward_perf.parquet"

    explicit = tmp_path / "explicit"
    parquet2, _metadata2, _skipped2 = write_formal_database(plan, rows, systems_root=explicit)
    assert parquet2.is_file()


def test_curated_systems_root_resolves_to_the_sdk_default_tree():
    """The default publication root must be the tree the SDK's
    --systems-paths default actually reads (the aiconfigurator_core package
    data), not a repo-relative guess."""

    from collector.fpm_forward.database import _curated_systems_root

    root = _curated_systems_root()
    assert root.parts[-2:] == ("systems", "data")
    assert "aiconfigurator_core" in root.parts
    assert root.is_dir()


def _args_cell(workload_kind: str) -> FPMCell:
    return FPMCell(
        cell_id=f"fpm-args-{workload_kind}",
        workload_kind=workload_kind,
        topology=ParallelTopology(tp=4, pp=1, dp=1, moe_tp=1, moe_ep=4, cp=1),
        weight_quantization="nvfp4",
        kv_cache_dtype="fp8",
        backend_policy=BackendPolicy("baseline_auto", {}, {}),
        parallel_strategy="tep",
        gemm_quant_mode="nvfp4",
        moe_quant_mode="nvfp4",
        fmha_quant_mode="fp8",
        comm_quant_mode="half",
    )


def _args_plan():
    return SimpleNamespace(
        sha256="plan-sha",
        model_path="org/model",
        system="b200_sxm",
        backend="vllm",
        options=SimpleNamespace(
            warmup_iterations=5,
            vllm_max_model_len=-1,
            prefill_sampling=PrefillSamplingProfile.build(max_isl=8192, max_batch_size=None),
        ),
    )


def _cell_cli_args(workload_kind: str) -> list[str]:
    from collector.fpm_forward.runner import _cell_generator_overrides

    merged = _cell_generator_overrides(_args_plan(), _args_cell(workload_kind), {})
    return merged["params"]["agg"]["extra_cli_args"]


def test_decode_cells_keep_prefix_caching_and_async_overlap():
    """Decode measures a production-shaped steady step on engine defaults.

    Prefix caching stays enabled: the engine's KV warm-up reuses warmed
    prefixes across points and refuses to warm without it
    (skip_reason="prefix_caching_disabled"), which would collapse every
    decode point into the fake-KV fallback regime convicted of
    underestimating capture-mode decode. The old pin's block-hasher
    contamination (26.5 ms -> 121 ms at (batch 256, 2.1M KV)) applied to the
    fallback protocol only; under KV warm-up the hashing runs at seed time,
    outside the measured step. Async scheduling also stays on: production
    overlaps scheduler CPU work with the GPU (async 26.5 ms = 1.03x of the
    25.8 ms measured on real traffic, sync 31.3 ms = 1.21x).
    """
    args = _cell_cli_args("decode")

    assert "--no-enable-prefix-caching" not in args
    assert "--no-async-scheduling" not in args


def test_prefill_cells_keep_prefix_caching_for_seeded_kv_reads():
    """Prefill must NOT disable prefix caching.

    Points with total_kv_read_tokens > 0 stage their context through the fake
    prefix cache, and ``_bench_cached_kv_read_tokens`` reads
    ``Request.block_hashes`` -- only populated while prefix caching installs a
    block hasher. Disabling it fails every cached-prefill point's seed
    validation. Prefill is insensitive to both flags anyway (100-108 ms across
    all four combinations at 8192 new tokens).
    """
    args = _cell_cli_args("prefill")

    assert "--no-enable-prefix-caching" not in args
    assert "--no-async-scheduling" in args


def _decode_cell_with_coordinate_collision(tmp_path, *, clamp_first):
    """Two decode plan points that measure the same physical coordinate.

    Under the steady-state decode policy a context-clamped point (requested
    ctx=1, measured at ctx=2) lands on the coordinate the native ctx=2 point
    already samples.
    """
    topology = ParallelTopology(tp=1, pp=1, dp=1, moe_tp=1, moe_ep=1, cp=1)
    cell = FPMCell(
        cell_id="fpm-test-decode",
        workload_kind="decode",
        topology=topology,
        weight_quantization="nvfp4",
        kv_cache_dtype="fp8",
        backend_policy=BackendPolicy("baseline_auto", {}, {}),
        parallel_strategy="single",
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
    if clamp_first == "both":
        reason_pairs = (["capture", "context_clamped"], ["capture", "context_clamped"])
    elif clamp_first:
        reason_pairs = (["capture", "context_clamped"], ["capture"])
    else:
        reason_pairs = (["capture"], ["capture"])
    points = []
    for benchmark_id, (reasons, wall) in enumerate(zip(reason_pairs, (0.0070, 0.0068), strict=True), start=1):
        points.append(
            (
                {
                    "point_type": "decode",
                    "benchmark_id": benchmark_id,
                    "total_prefill_tokens": 0,
                    "total_kv_read_tokens": 2,
                    "batch_size": 1,
                    "expected_cudagraph_mode": "FULL",
                    "expected_capture_size": 1,
                    "padding_tokens": 0,
                    "sample_reasons": reasons,
                },
                {
                    "counter_id": benchmark_id,
                    "dp_rank": 0,
                    "wall_time": wall,
                    "scheduled_requests": {
                        "num_prefill_requests": 0,
                        "sum_prefill_tokens": 0,
                        "sum_prefill_kv_tokens": 0,
                        "num_decode_requests": 1,
                        "sum_decode_kv_tokens": 2,
                    },
                },
            )
        )
    cell_dir = tmp_path / "cell"
    output = cell_dir / "raw" / "pod-0" / "benchmark.json"
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
                "config": {"mode": "decode"},
                "coverage": {"expected_points": 2, "completed_points": 2, "skipped_points": 0},
                "dp": {"rank": 0, "size": 1},
                "results": [{"point": point, "fpms": [fpm]} for point, fpm in points],
                "iteration_groups": [
                    {
                        "benchmark_id": point["benchmark_id"],
                        "point": point,
                        "expected_dp_ranks": [0],
                        "complete": True,
                        "wall_time": fpm["wall_time"],
                        "rank_results": [{"dp_rank": 0, "fpms": [fpm]}],
                    }
                    for point, fpm in points
                ],
                "skipped_points": [],
                "missing_phases": [],
                "timing": {
                    "benchmark_elapsed_seconds": 1.0,
                    "measured_iteration_seconds": 0.0138,
                },
            }
        )
    )
    return plan, cell, cell_dir


def test_native_aggregation_drops_clamped_duplicate_of_native_coordinate(tmp_path):
    plan, cell, cell_dir = _decode_cell_with_coordinate_collision(tmp_path, clamp_first=True)
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")

    assert len(rows) == 1
    assert rows[0]["latency_ms"] == pytest.approx(6.8)
    assert rows[0]["batch_size"] == 1
    assert rows[0]["total_kv_read_tokens"] == 2

    write_formal_database(plan, rows, systems_root=tmp_path / "systems")


def test_native_aggregation_keeps_first_when_all_duplicates_are_clamped(tmp_path):
    plan, cell, cell_dir = _decode_cell_with_coordinate_collision(tmp_path, clamp_first="both")
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")

    assert len(rows) == 1
    assert rows[0]["latency_ms"] == pytest.approx(7.0)


def test_native_aggregation_rejects_native_coordinate_collision(tmp_path):
    plan, cell, cell_dir = _decode_cell_with_coordinate_collision(tmp_path, clamp_first=False)

    with pytest.raises(ValueError, match="unclamped samples share one key"):
        aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")


def test_backend_identity_defaults_record_auto_everywhere():
    """v6: unspecified knobs record "auto" (engine decided), plumb nothing."""

    options = FPMCollectionOptions.from_args(_args())
    plan = build_collection_plan(
        backend="vllm",
        model_path="nvidia/GLM-5.2-NVFP4",
        system="b200_sxm",
        selected_ops={"dsa_context_module"},
        options=options,
        generator_overrides={},
    )
    policy = plan.cells[0].backend_policy
    assert backend_identity_columns(policy) == {
        "moe_backend": "auto",
        "attention_backend": "auto",
        "enable_wideep": False,
        "enable_eplb": False,
    }
    assert policy.policy_id == "baseline_auto"
    assert policy.generator_overrides == {}
    assert policy.expected_markers == {}


def test_pinned_moe_backend_plumbs_kernel_config_and_moves_cell_identity():
    """v6: a pinned value must reach the engine (kernel-config), demand
    resolved-config evidence, land in the row columns, and change cell ids."""

    kwargs = {
        "backend": "vllm",
        "model_path": "nvidia/GLM-5.2-NVFP4",
        "system": "b200_sxm",
        "selected_ops": {"dsa_context_module"},
        "generator_overrides": {},
    }
    auto_plan = build_collection_plan(**kwargs, options=FPMCollectionOptions.from_args(_args()))
    pinned_plan = build_collection_plan(
        **kwargs, options=FPMCollectionOptions.from_args(_args(fpm_moe_backend="flashinfer_cutlass"))
    )

    policy = pinned_plan.cells[0].backend_policy
    assert policy.expected_markers == {"config.engine_args.kernel_config.moe_backend": "flashinfer_cutlass"}
    cli_args = policy.generator_overrides["params"]["agg"]["extra_cli_args"]
    assert cli_args[0] == "--kernel-config"
    assert json.loads(cli_args[1]) == {"moe_backend": "flashinfer_cutlass"}
    assert backend_identity_columns(policy)["moe_backend"] == "flashinfer_cutlass"
    assert policy.policy_id == "explicit-moe_backend=flashinfer_cutlass"
    assert {cell.cell_id for cell in pinned_plan.cells}.isdisjoint({cell.cell_id for cell in auto_plan.cells})


def test_pinned_eplb_sets_engine_flag_and_marker():
    options = FPMCollectionOptions.from_args(_args(fpm_enable_eplb="true"))
    plan = build_collection_plan(
        backend="vllm",
        model_path="nvidia/GLM-5.2-NVFP4",
        system="b200_sxm",
        selected_ops={"dsa_context_module"},
        options=options,
        generator_overrides={},
    )
    policy = plan.cells[0].backend_policy
    assert policy.expected_markers == {"config.engine_args.enable_eplb": "True"}
    assert "--enable-eplb" in policy.generator_overrides["params"]["agg"]["extra_cli_args"]
    assert backend_identity_columns(policy)["enable_eplb"] is True


def test_unplumbed_backend_identity_fails_closed():
    """A pinned value the collector cannot deliver to the engine must be
    rejected up front - a row claiming an unapplied backend would be a lie."""

    for overrides, match in (
        ({"fpm_enable_wideep": "true"}, "SGLang-only"),  # true only; false is the default
        ({"fpm_attention_backend": "fa3"}, "no verified vllm plumbing"),
    ):
        options = FPMCollectionOptions.from_args(_args(**overrides))
        with pytest.raises(ValueError, match=match):
            build_collection_plan(
                backend="vllm",
                model_path="nvidia/GLM-5.2-NVFP4",
                system="b200_sxm",
                selected_ops={"dsa_context_module"},
                options=options,
                generator_overrides={},
            )


def test_strict_admission_rejects_aic_structural_invalidity_by_default(monkeypatch):
    """A structural ValueError from the AIC estimator means the modeling
    consumer would fail on the same math - the topology is dropped up front
    (default --fpm-strict-admission true)."""

    def structurally_invalid(*_args, **_kwargs):
        raise ValueError(
            "Invalid quantized MoE configuration: (moe_intermediate_size=1536 / moe_tp_size=8) "
            "% weight_block_size=128 != 0"
        )

    monkeypatch.setattr(
        "collector.fpm_forward.memory_admission.KVCacheEstimator.from_request",
        structurally_invalid,
    )
    with pytest.raises(ValueError, match="rejected every structurally valid FPM topology"):
        build_collection_plan(
            backend="vllm",
            model_path="nvidia/GLM-5.2-NVFP4",
            model_architecture="GlmMoeDsaForCausalLM",
            system="b200_sxm",
            selected_ops={"dsa_context_module", "dsa_generation_module"},
            options=FPMCollectionOptions.from_args(_args()),
        )


def test_strict_admission_selective_rejection_keeps_valid_topologies(monkeypatch, caplog):
    """Structural invalidity is per-topology: only the invalid shapes drop,
    the rest stay collectable, and the drop log attributes the structural
    cause instead of blaming memory capacity."""

    import logging
    from types import SimpleNamespace

    def selective(*_args, **kwargs):
        if kwargs.get("moe_tp_size", 1) > 1:
            raise ValueError("Invalid quantized MoE configuration: 1536 % 128 != 0")
        return SimpleNamespace(breakdown={"non_kv_bytes": 10 * 2**30, "gpu_memory_capacity_bytes": 100 * 2**30})

    monkeypatch.setattr(
        "collector.fpm_forward.memory_admission.KVCacheEstimator.from_request",
        selective,
    )
    with caplog.at_level(logging.WARNING, logger="collector.fpm_forward.memory_admission"):
        plan = build_collection_plan(
            backend="vllm",
            model_path="nvidia/GLM-5.2-NVFP4",
            model_architecture="GlmMoeDsaForCausalLM",
            system="b200_sxm",
            selected_ops={"dsa_context_module", "dsa_generation_module"},
            options=FPMCollectionOptions.from_args(_args()),
        )

    assert plan.topologies
    assert all(topology.moe_tp == 1 for topology in plan.topologies)
    assert {decision.disposition for decision in plan.topology_memory_admission} == {"admitted", "rejected"}
    rejected = [d for d in plan.topology_memory_admission if d.disposition == "rejected"]
    assert all("structural validation" in d.reason for d in rejected)
    assert "AIC structural validation predicts the configuration is invalid" in caplog.text
    assert "exceeds GPU capacity" not in caplog.text


def test_strict_admission_false_keeps_predicted_invalid_for_runtime_verification(monkeypatch):
    def structurally_invalid(*_args, **_kwargs):
        raise ValueError("Invalid quantized MoE configuration: 192 % 128 != 0")

    monkeypatch.setattr(
        "collector.fpm_forward.memory_admission.KVCacheEstimator.from_request",
        structurally_invalid,
    )
    plan = build_collection_plan(
        backend="vllm",
        model_path="nvidia/GLM-5.2-NVFP4",
        model_architecture="GlmMoeDsaForCausalLM",
        system="b200_sxm",
        selected_ops={"dsa_context_module", "dsa_generation_module"},
        options=FPMCollectionOptions.from_args(_args(fpm_strict_admission="false")),
    )

    assert len(plan.topologies) == 3
    decisions = plan.topology_memory_admission
    assert {decision.disposition for decision in decisions} == {"unknown"}
    assert all("predicts this configuration is invalid" in decision.reason for decision in decisions)


def test_missing_perf_data_stays_runnable_under_strict_admission(monkeypatch):
    """Coverage gaps are not structural invalidity: collection may be exactly
    what fills them, so strict admission must not reject them."""

    from aiconfigurator_core.sdk.errors import PerfDataNotAvailableError

    def unavailable(*_args, **_kwargs):
        raise PerfDataNotAvailableError("no perf rows for this shape")

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
