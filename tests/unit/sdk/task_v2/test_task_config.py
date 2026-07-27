# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the new flat Task in sdk/task.py.

End-to-end sweep correctness is covered by the integration parity test
against the legacy CLI; these tests focus on construction, defaulting,
prefix discipline, and the build_* helpers.
"""

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.task_v2 import Task

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Construction defaults
# ---------------------------------------------------------------------------


def test_default_task_config_is_agg_with_default_workload():
    t = Task()
    assert t.serving_mode == "agg"
    assert t.isl == 4000
    assert t.osl == 1000
    assert t.ttft == 1000.0
    assert t.tpot == 50.0


def test_agg_with_model_resolves_identity_and_backend():
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        total_gpus=8,
    )
    assert t.is_moe is True
    assert t.model_family == "DEEPSEEK"
    assert t.nextn is not None
    assert t.backend_version is not None  # resolved to latest
    # Search space defaults populated
    assert t.agg_tp_candidates == [1, 2, 4, 8]
    assert t.agg_pp_candidates == [1]


def test_agg_resolves_quant_from_hf():
    t = Task(serving_mode="agg", model_path="deepseek-ai/DeepSeek-V3", system_name="h200_sxm")
    # DeepSeek-V3 is fp8_block from HF config
    assert t.gemm_quant_mode == common.GEMMQuantMode.fp8_block


def test_agg_explicit_quant_overrides_hf():
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
    )
    assert t.gemm_quant_mode == common.GEMMQuantMode.bfloat16  # explicit wins
    # unset modes fall back to HF inference (DeepSeek-V3)
    assert t.kvcache_quant_mode == common.KVCacheQuantMode.fp8


# ---------------------------------------------------------------------------
# Disagg construction
# ---------------------------------------------------------------------------


def test_disagg_with_separate_role_specs():
    t = Task(
        serving_mode="disagg",
        prefill_model_path="deepseek-ai/DeepSeek-V3",
        prefill_system_name="h200_sxm",
        decode_model_path="deepseek-ai/DeepSeek-V3",
        decode_system_name="h200_sxm",
        total_gpus=32,
    )
    assert t.is_moe is True
    assert t.prefill_tp_candidates is not None
    assert t.decode_tp_candidates is not None
    assert t.num_gpu_per_replica is not None
    assert t.max_gpu_per_replica == 32  # clamped to total_gpus=32, matches v1 _finalize_disagg
    assert t.max_prefill_workers == 32


def test_disagg_wideep_sets_larger_replica_budget():
    t = Task(
        serving_mode="disagg",
        prefill_model_path="deepseek-ai/DeepSeek-V3",
        prefill_system_name="gb200",
        prefill_enable_wideep=True,
        decode_model_path="deepseek-ai/DeepSeek-V3",
        decode_system_name="gb200",
        decode_enable_wideep=True,
    )
    assert t.max_gpu_per_replica == 512
    assert t.num_gpu_per_replica is None  # wideep doesn't set a fixed list


# ---------------------------------------------------------------------------
# Strict prefix discipline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("enable_wideep", True),
        ("enable_chunked_prefill", True),
        ("enable_eplb", True),
        ("gemm_quant_mode", common.GEMMQuantMode.fp8),
    ],
)
def test_disagg_rejects_top_level_worker_field_leakage(field, value):
    """Setting top-level worker fields in disagg mode must raise (no silent override)."""
    with pytest.raises(ValueError, match="top-level worker fields"):
        Task(
            serving_mode="disagg",
            prefill_model_path="x",
            prefill_system_name="h200_sxm",
            decode_model_path="x",
            decode_system_name="h200_sxm",
            **{field: value},
        )


# ---------------------------------------------------------------------------
# from_yaml: flat format (the new canonical YAML)
# ---------------------------------------------------------------------------


def test_from_yaml_flat_agg():
    yaml_data = {
        "serving_mode": "agg",
        "model_path": "deepseek-ai/DeepSeek-V3",
        "system_name": "h200_sxm",
        "backend_name": "trtllm",
        "backend_version": "1.2.0rc5",
        "total_gpus": 8,
        "isl": 4000,
        "osl": 1000,
        "ttft": 1000.0,
        "tpot": 40.0,
        "nextn": 1,
        "nextn_accepted": 0.85,
        "gemm_quant_mode": "fp8_block",
        "kvcache_quant_mode": "bfloat16",
        "agg_num_gpu_candidates": [4, 8],
        "agg_tp_candidates": [1, 2, 4, 8],
        "agg_pp_candidates": [1],
    }
    t = Task.from_yaml(yaml_data)
    assert t.serving_mode == "agg"
    assert t.model_path == "deepseek-ai/DeepSeek-V3"
    assert t.backend_version == "1.2.0rc5"
    assert t.gemm_quant_mode == common.GEMMQuantMode.fp8_block
    assert t.kvcache_quant_mode == common.KVCacheQuantMode.bfloat16
    assert t.agg_num_gpu_candidates == [4, 8]
    assert t.agg_tp_candidates == [1, 2, 4, 8]
    assert t.nextn == 1


def test_from_yaml_flat_agg_minimal():
    """Just model_path + system_name; everything else defaults."""
    t = Task.from_yaml(
        {
            "serving_mode": "agg",
            "model_path": "deepseek-ai/DeepSeek-V3",
            "system_name": "h200_sxm",
        }
    )
    assert t.serving_mode == "agg"
    # Latest backend_version auto-resolved
    assert t.backend_version is not None
    # Quant inferred from HF config
    assert t.gemm_quant_mode == common.GEMMQuantMode.fp8_block


def test_from_yaml_flat_disagg():
    yaml_data = {
        "serving_mode": "disagg",
        "isl": 4000,
        "osl": 1000,
        "ttft": 1000.0,
        "tpot": 40.0,
        "total_gpus": 32,
        "prefill_model_path": "deepseek-ai/DeepSeek-V3",
        "prefill_system_name": "h200_sxm",
        "prefill_backend_name": "trtllm",
        "prefill_gemm_quant_mode": "fp8_block",
        "prefill_kvcache_quant_mode": "bfloat16",
        "decode_model_path": "deepseek-ai/DeepSeek-V3",
        "decode_system_name": "h200_sxm",
        "decode_backend_name": "trtllm",
        "decode_gemm_quant_mode": "fp8_block",
        "decode_kvcache_quant_mode": "bfloat16",
        "num_gpu_per_replica": [8, 16, 32, 64, 128],
        "max_gpu_per_replica": 128,
        "max_prefill_workers": 32,
        "max_decode_workers": 32,
        "prefill_latency_correction": 1.1,
        "decode_latency_correction": 1.08,
        "prefill_max_batch_size": 1,
        "decode_max_batch_size": 512,
    }
    t = Task.from_yaml(yaml_data)
    assert t.serving_mode == "disagg"
    assert t.prefill_model_path == "deepseek-ai/DeepSeek-V3"
    assert t.decode_model_path == "deepseek-ai/DeepSeek-V3"
    assert t.prefill_gemm_quant_mode == common.GEMMQuantMode.fp8_block
    assert t.decode_gemm_quant_mode == common.GEMMQuantMode.fp8_block
    assert t.max_gpu_per_replica == 32  # clamped to total_gpus=32 (min(32, 128)), matches v1
    assert t.prefill_latency_correction == 1.1


def test_from_yaml_with_cli_overrides():
    t = Task.from_yaml(
        {
            "serving_mode": "agg",
            "model_path": "deepseek-ai/DeepSeek-V3",
            "system_name": "h200_sxm",
            "isl": 4000,
            "ttft": 1000.0,
        },
        isl=8000,
        ttft=500.0,
    )
    assert t.isl == 8000
    assert t.ttft == 500.0


def test_from_yaml_rejects_unknown_keys():
    """Unknown keys hard-fail -- no silent-ignore path. Both bad keys are named."""
    with pytest.raises(ValueError) as exc:
        Task.from_yaml(
            {
                "serving_mode": "agg",
                "model_path": "deepseek-ai/DeepSeek-V3",
                "system_name": "h200_sxm",
                "totally_made_up_field": 42,
                "another_typo": "value",
            }
        )
    assert "totally_made_up_field" in str(exc.value)
    assert "another_typo" in str(exc.value)


def test_from_yaml_rejects_non_yaml_expressible_strategy_field(monkeypatch):
    """A valid-but-not-YAML-expressible field (predictor) written in YAML hard-fails."""
    monkeypatch.setattr(Task, "__post_init__", lambda self: None)
    with pytest.raises(ValueError) as exc:
        Task.from_yaml({"serving_mode": "agg", "model_path": "x", "system_name": "h200_sxm", "predictor": "nope"})
    assert "predictor" in str(exc.value)


def test_attention_backend_and_wideep_num_slots_reach_model_config():
    """attention_backend / wideep_num_slots flow into every role's ModelConfig
    (fa3 vs flashinfer selects different MLA perf tables; slots feeds EPLB)."""
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        attention_backend="fa3",
        wideep_num_slots=288,
    )
    mc = t.build_model_config(role="agg")
    assert mc.attention_backend == "fa3"
    assert mc.wideep_num_slots == 288


def test_invalid_attention_backend_rejected():
    t = Task(
        serving_mode="agg", model_path="deepseek-ai/DeepSeek-V3", system_name="h200_sxm", attention_backend="torch"
    )
    with pytest.raises(ValueError, match="attention_backend"):
        t.validate()


def test_invalid_wideep_num_slots_rejected():
    t = Task(serving_mode="agg", model_path="deepseek-ai/DeepSeek-V3", system_name="h200_sxm", wideep_num_slots=0)
    with pytest.raises(ValueError, match="wideep_num_slots"):
        t.validate()


def test_from_yaml_disagg_rejects_legacy_shared_model_path():
    """Legacy YAML shape with top-level model_path is not silently mirrored to roles."""
    with pytest.raises(ValueError, match="top-level worker fields"):
        Task.from_yaml(
            {
                "serving_mode": "disagg",
                "model_path": "deepseek-ai/DeepSeek-V3",  # legacy shared form
                "system_name": "h200_sxm",
                "total_gpus": 32,
            }
        )


# ---------------------------------------------------------------------------
# Builders consumed by sweep.py
# ---------------------------------------------------------------------------


def test_build_runtime_config_carries_workload():
    t = Task(isl=2048, osl=512, ttft=300.0, tpot=20.0)
    rt = t.build_runtime_config(batch_size=64)
    assert rt.isl == 2048
    assert rt.osl == 512
    assert rt.ttft == 300.0
    assert rt.tpot == 20.0
    assert rt.batch_size == 64


def test_build_model_config_agg_uses_resolved_quant():
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        nextn=2,
        nextn_accepted=1.2,
    )
    mc = t.build_model_config(role="agg")
    assert mc.gemm_quant_mode == common.GEMMQuantMode.bfloat16
    assert mc.nextn == t.nextn == 2
    assert not hasattr(mc, "nextn_accepted")
    assert t.nextn_accepted == 1.2
    assert t.build_speculative_profile().expected_accepted_tokens == 1.2


def test_sweep_agg_kwargs_shape():
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
    )
    kwargs = t.sweep_agg_kwargs(database="fake-db")
    assert kwargs["model_path"] == "deepseek-ai/DeepSeek-V3"
    assert kwargs["backend_name"] == "trtllm"
    assert kwargs["database"] == "fake-db"
    assert isinstance(kwargs["parallel_config_list"], list)
    assert len(kwargs["parallel_config_list"]) > 0


def test_sweep_disagg_kwargs_shape():
    t = Task(
        serving_mode="disagg",
        prefill_model_path="deepseek-ai/DeepSeek-V3",
        prefill_system_name="h200_sxm",
        decode_model_path="deepseek-ai/DeepSeek-V3",
        decode_system_name="h200_sxm",
    )
    kwargs = t.sweep_disagg_kwargs(prefill_database="p-db", decode_database="d-db")
    assert kwargs["model_path"] == "deepseek-ai/DeepSeek-V3"
    assert kwargs["prefill_database"] == "p-db"
    assert kwargs["decode_database"] == "d-db"
    assert kwargs["prefill_latency_correction"] == 1.1
    assert kwargs["decode_latency_correction"] == 1.08
    assert kwargs["decode_max_num_tokens"] == 512
    assert len(kwargs["prefill_num_worker_list"]) == 32
    assert len(kwargs["decode_num_worker_list"]) == 32
    # Rate-match degradation and autoscale TTFT correction defaults flow through.
    assert kwargs["rate_matching_prefill_degradation"] == 0.9
    assert kwargs["rate_matching_decode_degradation"] == 0.92
    assert kwargs["autoscale_ttft_correction_factor"] == 1.8


def test_sweep_disagg_require_same_tp_sglang_non_wideep():
    """SGLang non-wideep disagg must enforce prefill/decode TP equality (dynamo#5870).

    WideEP relaxes it; other backends never get the SGLang-specific constraint.
    """

    def mk(backend, **extra):
        return Task(
            serving_mode="disagg",
            prefill_model_path="deepseek-ai/DeepSeek-V3",
            prefill_system_name="h200_sxm",
            prefill_backend_name=backend,
            decode_model_path="deepseek-ai/DeepSeek-V3",
            decode_system_name="h200_sxm",
            decode_backend_name=backend,
            **extra,
        ).sweep_disagg_kwargs(prefill_database=None, decode_database=None)

    assert mk("sglang")["require_same_tp"] is True
    assert mk("sglang", prefill_enable_wideep=True, decode_enable_wideep=True)["require_same_tp"] is False
    assert mk("trtllm")["require_same_tp"] is False


def test_deepseek_prefill_downgrades_decode_keeps_fp8_fmha(caplog):
    """DeepSeek context attention (prefill) downgrades fp8 FMHA to bf16 via the
    data fallback (the packaged context_mla tables carry no fp8 slice), with
    one warning for the task.  Decode keeps the checkpoint-inferred fp8 label:
    no generation table keys on fmha, so the label is inert for decode modeling
    and the fallback skips decode roles.
    """
    import logging

    from aiconfigurator.sdk import common

    with caplog.at_level(logging.WARNING):
        t = Task(
            serving_mode="disagg",
            prefill_model_path="deepseek-ai/DeepSeek-V3",
            prefill_system_name="h200_sxm",
            prefill_backend_name="sglang",
            decode_model_path="deepseek-ai/DeepSeek-V3",
            decode_system_name="h200_sxm",
            decode_backend_name="sglang",
        )
    assert t.prefill_fmha_quant_mode == common.FMHAQuantMode.bfloat16
    assert t.decode_fmha_quant_mode == common.FMHAQuantMode.fp8
    fallback_msgs = [r.message for r in caplog.records if "falling back to bfloat16 FMHA" in r.message]
    assert len(fallback_msgs) == 1 and fallback_msgs[0].startswith("prefill ")


def test_fmha_data_fallback_unknown_arch_downgrades_with_warning(caplog):
    """A checkpoint-inferred fp8 FMHA on a platform whose context-attention table
    has no fp8 slice (a100: no fp8 hardware) falls back to bfloat16 with a warning,
    instead of leaving a mode that validate would reject.  The same model on a
    platform WITH fp8 data (h200) keeps fp8 and does not warn.

    kv is pinned to bfloat16 on a100: capability is judged jointly with the kv
    mode, and a100's only slice is (bf16 fmha, bf16 kv) -- with an fp8 kv there
    is nothing to fall back to and the inference is kept for validate to
    report.
    """
    import logging

    from aiconfigurator.sdk import common

    with caplog.at_level(logging.WARNING):
        t = Task(
            serving_mode="agg",
            model_path="Qwen/Qwen3-32B-FP8-Static-PerTensor",
            system_name="a100_sxm",
            backend_name="sglang",
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        )
    assert t.fmha_quant_mode == common.FMHAQuantMode.bfloat16
    assert any("falling back to bfloat16 FMHA" in r.message for r in caplog.records)

    # With the inferred fp8 kv, no a100 slice can serve at all -> keep fp8
    # (validate reports the gap); no misleading "falling back" warning.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        t_kv_fp8 = Task(
            serving_mode="agg",
            model_path="Qwen/Qwen3-32B-FP8-Static-PerTensor",
            system_name="a100_sxm",
            backend_name="sglang",
        )
    assert t_kv_fp8.fmha_quant_mode == common.FMHAQuantMode.fp8
    assert not any("falling back to bfloat16 FMHA" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        t2 = Task(
            serving_mode="agg",
            model_path="Qwen/Qwen3-32B-FP8-Static-PerTensor",
            system_name="h200_sxm",
            backend_name="trtllm",
        )
    assert t2.fmha_quant_mode == common.FMHAQuantMode.fp8
    assert not any("falling back to bfloat16 FMHA" in r.message for r in caplog.records)


def test_fmha_data_fallback_skips_generation_only_decode(caplog):
    """A generic-attention decode role never reads fmha data (generation tables
    key on kv dtype; validate checks fmha only for context roles), so the
    fallback must not warn about or downgrade decode even on a system whose
    context table lacks fp8.  The prefill role on the same system DOES fall back.
    """
    import logging

    from aiconfigurator.sdk import common

    with caplog.at_level(logging.WARNING):
        t = Task(
            serving_mode="disagg",
            prefill_model_path="Qwen/Qwen3-32B-FP8-Static-PerTensor",
            prefill_system_name="a100_sxm",
            prefill_backend_name="sglang",
            prefill_kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            decode_model_path="Qwen/Qwen3-32B-FP8-Static-PerTensor",
            decode_system_name="a100_sxm",
            decode_backend_name="sglang",
            decode_kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        )
    assert t.prefill_fmha_quant_mode == common.FMHAQuantMode.bfloat16
    assert t.decode_fmha_quant_mode == common.FMHAQuantMode.fp8
    fallback_msgs = [r.message for r in caplog.records if "falling back to bfloat16 FMHA" in r.message]
    assert len(fallback_msgs) == 1 and fallback_msgs[0].startswith("prefill ")


def test_fmha_data_fallback_without_bf16_slice_left_untouched(monkeypatch, caplog):
    """When the context table has fmha data but neither fp8 nor bf16 slices
    (e.g. wideep tables carry only fp8_block), the fallback must leave the
    inferred fp8 alone -- there is nothing safe to fall back to, so validate
    reports the gap instead."""
    import logging

    from aiconfigurator.sdk import common

    monkeypatch.setattr(Task, "_context_fmha_supported_modes", lambda self, role: ["fp8_block"])
    with caplog.at_level(logging.WARNING):
        t = Task(
            serving_mode="agg",
            model_path="Qwen/Qwen3-32B-FP8-Static-PerTensor",
            system_name="a100_sxm",
            backend_name="sglang",
        )
    assert t.fmha_quant_mode == common.FMHAQuantMode.fp8
    assert not any("falling back to bfloat16 FMHA" in r.message for r in caplog.records)


def test_deepseek_v32_v4_context_fmha_downgrade_is_data_driven():
    """DeepSeek-V3.2 / V4 context fmha resolves bf16 on sglang b200 because the
    dsa/dsv4 context module tables there carry only bf16 slices -- decided by
    the data fallback, not a hand-written model rule.  Decode keeps the fp8
    label (no generation table keys on fmha).
    """
    from aiconfigurator.sdk import common

    for mp in ("deepseek-ai/DeepSeek-V3.2", "deepseek-ai/DeepSeek-V4-Pro"):
        t = Task(
            serving_mode="disagg",
            prefill_model_path=mp,
            prefill_system_name="b200_sxm",
            prefill_backend_name="sglang",
            decode_model_path=mp,
            decode_system_name="b200_sxm",
            decode_backend_name="sglang",
        )
        assert t.prefill_fmha_quant_mode == common.FMHAQuantMode.bfloat16
        assert t.decode_fmha_quant_mode == common.FMHAQuantMode.fp8


def test_get_model_preserves_task_resolved_fmha():
    """get_model()'s legacy FMHA guards fire only when fmha arrives unset: a
    Task-resolved fp8 (data-backed -- b200 vLLM ships native fp8 dsa_context
    slices) must survive model build.  Review regression: the guards in
    _apply_model_quant_defaults used to re-downgrade on the value, silently
    undoing the data-driven resolution at every sweep point."""
    from aiconfigurator.sdk import common
    from aiconfigurator.sdk.models import get_model

    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3.2",
        system_name="b200_sxm",
        backend_name="vllm",
    )
    assert t.fmha_quant_mode == common.FMHAQuantMode.fp8
    mc = t.build_model_config(role="agg")
    mc.tp_size = 8
    mc.moe_ep_size = 8
    mc.moe_tp_size = 1
    get_model("deepseek-ai/DeepSeek-V3.2", mc, "vllm")
    assert mc.fmha_quant_mode == common.FMHAQuantMode.fp8


def test_fmha_fallback_uses_joint_fmha_kv_capability(caplog):
    """Capability is judged jointly with the role's kv mode: on b200 trtllm the
    fp8 context_mla slice exists only under kv=fp8 (shared-layer module rows),
    so inferred fp8 fmha survives with kv=fp8 but must downgrade with an
    explicit bf16 kv -- the flat per-op list would keep fp8 and crash at query
    time (review finding)."""
    import logging

    from aiconfigurator.sdk import common

    base = dict(serving_mode="agg", model_path="deepseek-ai/DeepSeek-V3", system_name="b200_sxm", backend_name="trtllm")
    assert Task(**base).fmha_quant_mode == common.FMHAQuantMode.fp8  # kv inferred fp8 -> joint slice present
    with caplog.at_level(logging.WARNING):
        t = Task(**base, kvcache_quant_mode=common.KVCacheQuantMode.bfloat16)
    assert t.fmha_quant_mode == common.FMHAQuantMode.bfloat16
    assert any("falling back to bfloat16 FMHA" in r.message for r in caplog.records)


def test_wideep_trtllm_context_fmha_capability_uses_granular_table(caplog):
    """trtllm wideep context queries the granular context_mla table directly, so
    the fmha fallback must key capability off the granular slices -- the merged
    context_mla list can contain module-only fp8 rows inherited cross-framework
    via the shared layer (gb200: vllm module data), which the wideep path can
    never hit.  Regression for the deepseek_wideep_trtllm.yaml e2e failure.
    """
    import logging

    from aiconfigurator.sdk import common

    with caplog.at_level(logging.WARNING):
        t = Task(
            serving_mode="agg",
            model_path="deepseek-ai/DeepSeek-V3",
            system_name="gb200",
            backend_name="trtllm",
            enable_wideep=True,
        )
    assert t.fmha_quant_mode == common.FMHAQuantMode.bfloat16
    assert any("context_mla_granular" in r.message for r in caplog.records)


def test_nextn_never_auto_enabled(caplog):
    """MTP is never auto-enabled: nextn stays 0 even for checkpoints that ship
    MTP layers; a hint log surfaces the unused capability."""
    import logging

    def mk(mp):
        return Task(serving_mode="agg", model_path=mp, system_name="h200_sxm", backend_name="trtllm").nextn

    with caplog.at_level(logging.INFO, logger="aiconfigurator.sdk.task_v2"):
        assert mk("deepseek-ai/DeepSeek-V3") == 0  # HF declares 1 -- still off by default
        assert mk("moonshotai/Kimi-K2.5") == 0
        assert mk("Qwen/Qwen3.5-27B") == 0
    assert any("ships MTP" in r.message for r in caplog.records)


def test_nextn_auto_resolves_depth_from_checkpoint():
    """nextn='auto' takes the draft depth from num_nextn_predict_layers; the
    acceptance value is still required -- it is never inferred."""
    import pytest as _pytest

    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        backend_name="trtllm",
        nextn="auto",
        nextn_accepted=0.7,
    )
    assert t.nextn == 1  # checkpoint declares num_nextn_predict_layers=1
    assert t.nextn_accepted == 0.7

    # No MTP layers in the checkpoint -> auto resolves to disabled, no
    # acceptance value needed.
    t2 = Task(
        serving_mode="agg",
        model_path="Qwen/Qwen3-32B",
        system_name="h200_sxm",
        backend_name="trtllm",
        nextn="auto",
    )
    assert t2.nextn == 0

    # auto resolving to a positive depth still demands nextn_accepted, and the
    # error says what the depth resolved to.
    with _pytest.raises(ValueError, match=r"auto.*resolved to nextn=1.*nextn_accepted"):
        Task(
            serving_mode="agg",
            model_path="deepseek-ai/DeepSeek-V3",
            system_name="h200_sxm",
            backend_name="trtllm",
            nextn="auto",
        )

    # auto cannot resolve without a checkpoint to read.
    with _pytest.raises(ValueError, match="requires a model path"):
        Task(serving_mode="agg", model_path="", system_name="h200_sxm", backend_name="trtllm", nextn="auto")


def test_nextn_requires_nextn_accepted():
    """nextn > 0 without nextn_accepted is a hard error -- no built-in acceptance assumption."""
    import pytest as _pytest

    with _pytest.raises(ValueError, match="nextn_accepted"):
        Task(
            serving_mode="agg",
            model_path="deepseek-ai/DeepSeek-V3",
            system_name="h200_sxm",
            backend_name="trtllm",
            nextn=1,
        )
    with _pytest.raises(ValueError, match="within"):
        Task(
            serving_mode="agg",
            model_path="deepseek-ai/DeepSeek-V3",
            system_name="h200_sxm",
            backend_name="trtllm",
            nextn=1,
            nextn_accepted=1.5,
        )
    with _pytest.raises(ValueError, match="within"):
        Task(
            serving_mode="agg",
            model_path="deepseek-ai/DeepSeek-V3",
            system_name="h200_sxm",
            backend_name="trtllm",
            nextn=1,
            nextn_accepted=-0.1,
        )
    # Validation must not depend on model-identity resolution (which is skipped
    # when no primary model path is set).
    with _pytest.raises(ValueError, match="nextn_accepted"):
        Task(serving_mode="agg", model_path="", system_name="h200_sxm", backend_name="trtllm", nextn=1)
    with _pytest.raises(ValueError, match=">= 0"):
        Task(
            serving_mode="agg",
            model_path="deepseek-ai/DeepSeek-V3",
            system_name="h200_sxm",
            backend_name="trtllm",
            nextn=-1,
        )
    with _pytest.raises(ValueError, match="integer"):
        Task(
            serving_mode="agg",
            model_path="deepseek-ai/DeepSeek-V3",
            system_name="h200_sxm",
            backend_name="trtllm",
            nextn=1.5,
            nextn_accepted=0.5,
        )


def test_nextn_explicit_override_warns(caplog):
    """An explicit nextn diverging from the checkpoint warns (MTP module reuse)."""
    import logging

    with caplog.at_level(logging.WARNING, logger="aiconfigurator.sdk.task_v2"):
        t = Task(
            serving_mode="agg",
            model_path="deepseek-ai/DeepSeek-V3",
            system_name="h200_sxm",
            backend_name="trtllm",
            nextn=3,
            nextn_accepted=1.8,
        )
    assert t.nextn == 3
    assert t.nextn_accepted == 1.8
    assert any("differs from" in r.message for r in caplog.records)


def test_moe_backend_flows_into_model_config():
    """Task.moe_backend must reach the per-role ModelConfig so get_model selects the
    right MoE kernel (v1 set it; v2 build_model_config used to drop it -> None)."""
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        backend_name="sglang",
        moe_backend="deepep_moe",
    )
    assert t.build_model_config(role="agg").moe_backend == "deepep_moe"


def test_dsv4_native_sglang_moe_remap():
    """Native DeepSeek-V4 (Pro AND Flash) on sglang remaps MoE to the arch-specific
    kernel (v1 dsv4pro-moe-arch): Blackwell -> w4a8_mxfp4_mxfp8_trtllm; on Hopper
    the native FP4-expert checkpoints are rejected outright. megamoe, non-sglang
    backends, and the sgl-project FP8 requant artifacts are exempt.
    """
    from aiconfigurator.sdk import common

    def moe(be, mp="deepseek-ai/DeepSeek-V4-Pro", system="b200_sxm", **kw):
        return Task(
            serving_mode="agg",
            model_path=mp,
            system_name=system,
            backend_name=be,
            **kw,
        ).moe_quant_mode

    for mp in ("deepseek-ai/DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Flash"):
        assert moe("sglang", mp=mp) == common.MoEQuantMode.w4a8_mxfp4_mxfp8_trtllm
        # Native FP4-expert checkpoints are rejected outright on Hopper.
        with pytest.raises(ValueError, match="native FP4 routed-expert"):
            moe("sglang", mp=mp, system="h200_sxm")
        assert moe("trtllm", mp=mp) != common.MoEQuantMode.w4a8_mxfp4_mxfp8_trtllm
    # megamoe keys its own quant table (packaged data exists only for V4-Pro).
    assert moe("sglang", moe_backend="megamoe") != common.MoEQuantMode.w4a8_mxfp4_mxfp8_trtllm
    # FP8 requant artifacts keep their own (fp8_block-family) resolution.
    assert moe("sglang", mp="sgl-project/DeepSeek-V4-Flash-FP8") != common.MoEQuantMode.w4a8_mxfp4_mxfp8_trtllm


def test_dsv4_third_party_fp4_sglang_moe_remap_on_hopper():
    """Third-party FP4-expert DSV4 checkpoints (e.g. RedHatAI) get the sglang
    MoE remap based on expert_dtype rather than hardcoded model paths.
    Hopper -> w4a16_mxfp4_cutlass; Blackwell -> w4a8_mxfp4_mxfp8_trtllm."""
    hopper = Task(
        serving_mode="agg",
        model_path="RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8",
        system_name="h200_sxm",
        backend_name="sglang",
    )
    assert hopper.moe_quant_mode == common.MoEQuantMode.w4a16_mxfp4_cutlass
    blackwell = Task(
        serving_mode="agg",
        model_path="RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8",
        system_name="b200_sxm",
        backend_name="sglang",
    )
    assert blackwell.moe_quant_mode == common.MoEQuantMode.w4a8_mxfp4_mxfp8_trtllm
    # FP8-only requants (no expert_dtype=fp4) are NOT remapped.
    fp8_requant = Task(
        serving_mode="agg",
        model_path="sgl-project/DeepSeek-V4-Flash-FP8",
        system_name="h200_sxm",
        backend_name="sglang",
    )
    assert fp8_requant.moe_quant_mode not in (
        common.MoEQuantMode.w4a16_mxfp4_cutlass,
        common.MoEQuantMode.w4a8_mxfp4_mxfp8_trtllm,
    )


@pytest.mark.parametrize(
    "model_path",
    [
        "RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8",
        "sgl-project/DeepSeek-V4-Flash-FP8",
        "deepseek-ai/DeepSeek-V4-Flash",
    ],
)
def test_dsv4_fp8_kvcache_enforced(model_path):
    """All DSV4 models use FP8 KV cache, regardless of HF config."""
    t = Task(
        serving_mode="agg",
        model_path=model_path,
        system_name="b200_sxm",
        backend_name="trtllm",
    )
    assert t.kvcache_quant_mode == common.KVCacheQuantMode.fp8


def test_pareto_sweep_controls_tpot_grid():
    """pareto_sweep=True (default) sweeps the legacy TPOT grid (matches v1); False
    evaluates only the single tpot target (Planner path)."""
    t = Task(serving_mode="agg", model_path="Qwen/Qwen3-32B", system_name="h200_sxm", backend_name="trtllm")
    assert t.pareto_sweep is True
    assert isinstance(t.sweep_agg_kwargs(database=None)["runtime_config"].tpot, list)
    t2 = Task(
        serving_mode="agg",
        model_path="Qwen/Qwen3-32B",
        system_name="h200_sxm",
        backend_name="trtllm",
        tpot=42.0,
        pareto_sweep=False,
    )
    assert t2.sweep_agg_kwargs(database=None)["runtime_config"].tpot == 42.0


def test_total_gpus_budget_filters_and_validates():
    """total_gpus clamps the num_gpu search space and validates it (v1 _finalize_*)."""
    # agg: filters num_gpu candidates above the budget
    t = Task(serving_mode="agg", model_path="Qwen/Qwen3-32B", system_name="gb200", backend_name="trtllm", total_gpus=4)
    assert all(n <= 4 for n in t.agg_num_gpu_candidates)
    assert t.agg_num_gpu_candidates  # not emptied
    # agg: negative total_gpus rejected
    with pytest.raises(ValueError, match="no smaller than 0"):
        Task(
            serving_mode="agg",
            model_path="Qwen/Qwen3-32B",
            system_name="h200_sxm",
            backend_name="trtllm",
            total_gpus=-1,
        )
    # disagg: total_gpus < 2 rejected
    with pytest.raises(ValueError, match="greater than 2"):
        Task(
            serving_mode="disagg",
            prefill_model_path="Qwen/Qwen3-32B",
            prefill_system_name="h200_sxm",
            prefill_backend_name="trtllm",
            decode_model_path="Qwen/Qwen3-32B",
            decode_system_name="h200_sxm",
            decode_backend_name="trtllm",
            total_gpus=1,
        )
    # disagg: max_gpu_per_replica clamped to total_gpus, candidates filtered
    td = Task(
        serving_mode="disagg",
        prefill_model_path="Qwen/Qwen3-32B",
        prefill_system_name="h200_sxm",
        prefill_backend_name="trtllm",
        decode_model_path="Qwen/Qwen3-32B",
        decode_system_name="h200_sxm",
        decode_backend_name="trtllm",
        total_gpus=8,
    )
    assert td.max_gpu_per_replica == 8  # min(8, 128)
    assert all(n <= 8 for n in td.prefill_num_gpu_candidates)
    assert all(n <= 8 for n in td.decode_num_gpu_candidates)
    # num_gpu_per_replica keeps v1's full list at construct time; the cap is applied at sweep
    # time via get_working_list (matches v1 construct state, not a construct-time filter).
    assert 128 in td.num_gpu_per_replica
    assert all(n <= 8 for n in td.sweep_disagg_kwargs(prefill_database=None, decode_database=None)["num_gpu_list"])


def test_large_pipeline_parallel_augments_dsv32_blackwell_defaults():
    """DeepSeek-V3.2 MoE on Blackwell, non-wideep, total_gpus>=16 gets PP=2/TP=8/16-GPU
    added to the DEFAULT search space (v1 _include_large_pipeline_parallel_worker)."""
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3.2",
        system_name="b200_sxm",
        backend_name="trtllm",
        total_gpus=64,
    )
    assert 16 in t.agg_num_gpu_candidates
    assert 8 in t.agg_tp_candidates
    assert 2 in t.agg_pp_candidates
    # Not applied below 16 GPUs
    t2 = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3.2",
        system_name="b200_sxm",
        backend_name="trtllm",
        total_gpus=8,
    )
    assert 2 not in t2.agg_pp_candidates
    # Not applied on non-Blackwell
    t3 = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3.2",
        system_name="h200_sxm",
        backend_name="trtllm",
        total_gpus=64,
    )
    assert 2 not in t3.agg_pp_candidates
    # User-supplied candidates are NOT augmented (v1 yaml-over-defaults)
    t4 = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3.2",
        system_name="b200_sxm",
        backend_name="trtllm",
        total_gpus=64,
        agg_pp_candidates=[1],
    )
    assert t4.agg_pp_candidates == [1]


def test_megamoe_sglang_parallel_lists_and_validation():
    """SGLang MegaMoE (initial support): DeepSeek-V4-Pro on Blackwell gets EP-only parallel
    lists; non-sglang / non-DeepSeek-V4 are rejected (v1 _validate_megamoe_backend_support)."""
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V4-Pro",
        system_name="b200_sxm",
        backend_name="sglang",
        moe_backend="megamoe",
    )
    assert t.agg_moe_tp_candidates == [1]  # EP-only
    assert t.agg_moe_ep_candidates  # populated from the megamoe lists
    with pytest.raises(ValueError, match="SGLang backend"):
        Task(
            serving_mode="agg",
            model_path="deepseek-ai/DeepSeek-V4-Pro",
            system_name="b200_sxm",
            backend_name="trtllm",
            moe_backend="megamoe",
        )
    with pytest.raises(ValueError, match="DeepSeek-V4"):
        Task(
            serving_mode="agg",
            model_path="deepseek-ai/DeepSeek-V3",
            system_name="b200_sxm",
            backend_name="sglang",
            moe_backend="megamoe",
        )


def test_sglang_agg_default_moe_ep_search():
    """SGLang non-wideep MoE agg DEFAULT search must include moe_ep>1 (standard comm) /
    EP-only for deepep_moe (v1 standard vs deepep_moe branches). Was moe_ep=[1] — a bug
    masked by always passing explicit candidates, caught by default-path parity."""
    t = Task(serving_mode="agg", model_path="deepseek-ai/DeepSeek-V3", system_name="h200_sxm", backend_name="sglang")
    assert t.agg_moe_tp_candidates == [1, 2, 4, 8]
    assert t.agg_moe_ep_candidates == [1, 2, 4, 8]
    t2 = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        backend_name="sglang",
        moe_backend="deepep_moe",
    )
    assert t2.agg_moe_tp_candidates == [1]
    assert t2.agg_moe_ep_candidates == [1, 2, 4, 8]


def test_run_validates_by_default():
    """run() validates first (v1 fail-fast); validate=False skips it. SGLang WideEP DeepSeek
    has no wideep_context_mla data for fp8/bf16 -> validate raises."""
    from aiconfigurator.sdk.errors import UnsupportedWideepConfigError

    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        backend_name="sglang",
        enable_wideep=True,
        total_gpus=64,
    )
    with pytest.raises(UnsupportedWideepConfigError):
        t.run()  # default validate=True


def test_enable_wideep_normalizes_moe_backend():
    """enable_wideep implies the deepep_moe MoE backend (mirrors v1 __init__), so DB
    validation selects the wideep_*_moe ops."""
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        backend_name="sglang",
        enable_wideep=True,
    )
    assert t.moe_backend == "deepep_moe"


def test_wideep_replica_size_is_bounded():
    """WideEP num_gpu_list (replica sizes) must be range(1, max_gpu_per_replica+1), not
    unbounded -- v2 sweep gates replica size by this list, mirroring v1 get_working_list."""
    t = Task(
        serving_mode="disagg",
        prefill_model_path="Qwen/Qwen3-235B-A22B",
        prefill_system_name="b200_sxm",
        prefill_backend_name="trtllm",
        prefill_enable_wideep=True,
        decode_model_path="Qwen/Qwen3-235B-A22B",
        decode_system_name="b200_sxm",
        decode_backend_name="trtllm",
        decode_enable_wideep=True,
        total_gpus=64,
    )
    kw = t.sweep_disagg_kwargs(prefill_database=None, decode_database=None)
    assert kw["num_gpu_list"] == list(range(1, t.max_gpu_per_replica + 1))
    assert max(kw["num_gpu_list"]) <= t.total_gpus


def test_explicit_fmha_fp8_not_downgraded():
    """V3/Kimi context fmha fp8->bf16 downgrade fires only on HF-inferred fp8 (matches v1's
    `not explicit_fmha_mode` guard). An explicit fp8 is kept, so validate can fail fast
    (instead of silently modelling bf16)."""
    from aiconfigurator.sdk import common

    base = dict(serving_mode="agg", model_path="deepseek-ai/DeepSeek-V3", system_name="h200_sxm", backend_name="trtllm")
    assert Task(**base).fmha_quant_mode == common.FMHAQuantMode.bfloat16  # HF-inferred -> downgraded
    assert (
        Task(**base, fmha_quant_mode=common.FMHAQuantMode.fp8).fmha_quant_mode == common.FMHAQuantMode.fp8
    )  # explicit -> kept


def test_database_mode_is_forwarded_to_view_loader(monkeypatch):
    """Task delegates mode selection to the configured database-view boundary."""
    calls = []
    database = object()

    def fake_get_database_view(*args, **kwargs):
        calls.append((args, kwargs))
        return database

    monkeypatch.setattr("aiconfigurator.sdk.perf_database.get_database_view", fake_get_database_view)
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        backend_name="trtllm",
        database_mode="EMPIRICAL",
    )
    assert t._load_database("h200_sxm", "trtllm", "1.3.0rc10") is database
    assert calls[-1][1]["database_mode"] == "EMPIRICAL"
    assert calls[-1][1]["allow_missing_data"] is True

    t2 = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        backend_name="trtllm",
        database_mode="SILICON",
    )
    assert t2._load_database("h200_sxm", "trtllm", "1.3.0rc10") is database
    assert calls[-1][1]["database_mode"] == "SILICON"
    assert calls[-1][1]["allow_missing_data"] is False


def test_no_orphan_fields():
    """Every tunable Task field must be consumed in task_v2/sweep -- guards against the
    'accepted-but-ignored field' class of bug (e.g. database_mode, which was defined +
    converted from YAML but never read at runtime). Candidate lists are read dynamically via
    getattr(f"{role}_{dim}_candidates") so they're whitelisted; everything else must appear
    as self.<field> or _role_attr(role, "<bare>")."""
    import dataclasses
    import pathlib
    import re

    import aiconfigurator.sdk.sweep as sweep_mod
    import aiconfigurator.sdk.task_v2 as tv2_mod

    srcs = pathlib.Path(tv2_mod.__file__).read_text() + pathlib.Path(sweep_mod.__file__).read_text()
    orphans = []
    for f in [x.name for x in dataclasses.fields(Task) if x.init and not x.name.startswith("_")]:
        if f.endswith("_candidates"):
            continue  # read dynamically via getattr(f"{role}_{dim}_candidates")
        bare = re.sub(r"^(prefill_|decode_)", "", f)
        read = (
            re.search(rf"self\.{f}\b", srcs)
            or re.search(rf'_role_attr\([^,]+,\s*"{bare}"', srcs)
            or re.search(rf'"{f}"', srcs)
            or re.search(rf'"{bare}"', srcs)
        )
        if not read:
            orphans.append(f)
    assert not orphans, f"orphan Task fields (defined but never consumed): {orphans}"


def test_scalar_config_fields_reach_sweep_consumers():
    """Scalar config knobs must actually reach sweep consumers (regression guard for the
    accepted-but-ignored class of bug, e.g. database_mode). Values must flow into the
    sweep kwargs / runtime_config rather than being silently dropped."""
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        backend_name="trtllm",
        isl=1234,
        osl=567,
        ttft=111.0,
        tpot=22.0,
        prefix=333,
        request_latency=8888.0,
        free_gpu_memory_fraction=0.55,
        max_seq_len=4321,
    )
    kw = t.sweep_agg_kwargs(database=None)
    rt = kw["runtime_config"]
    assert (rt.isl, rt.osl, rt.prefix, rt.request_latency) == (1234, 567, 333, 8888.0)
    assert kw["free_gpu_memory_fraction"] == 0.55
    assert kw["max_seq_len"] == 4321
    assert isinstance(rt.tpot, list)  # pareto_sweep=True default -> legacy grid reaches the sweep

    td = Task(
        serving_mode="disagg",
        prefill_model_path="deepseek-ai/DeepSeek-V3",
        prefill_system_name="h200_sxm",
        prefill_backend_name="trtllm",
        decode_model_path="deepseek-ai/DeepSeek-V3",
        decode_system_name="h200_sxm",
        decode_backend_name="trtllm",
        total_gpus=16,
        isl=1234,
        osl=567,
        ttft=111.0,
        tpot=22.0,
        prefix=333,
        request_latency=8888.0,
    )
    drt = td.sweep_disagg_kwargs(prefill_database=None, decode_database=None)["runtime_config"]
    assert (drt.isl, drt.osl, drt.prefix, drt.request_latency) == (1234, 567, 333, 8888.0)


def test_disagg_calibration_overrides_flow_into_sweep_kwargs():
    """Overriding the new Task fields propagates to sweep_disagg_kwargs."""
    t = Task(
        serving_mode="disagg",
        prefill_model_path="deepseek-ai/DeepSeek-V3",
        prefill_system_name="h200_sxm",
        decode_model_path="deepseek-ai/DeepSeek-V3",
        decode_system_name="h200_sxm",
        prefill_latency_correction=1.3,
        decode_latency_correction=1.15,
        rate_match_prefill_degradation=0.85,
        rate_match_decode_degradation=0.88,
        autoscale_ttft_correction_factor=2.0,
    )
    kwargs = t.sweep_disagg_kwargs(prefill_database=None, decode_database=None)
    assert kwargs["prefill_latency_correction"] == 1.3
    assert kwargs["decode_latency_correction"] == 1.15
    assert kwargs["rate_matching_prefill_degradation"] == 0.85
    assert kwargs["rate_matching_decode_degradation"] == 0.88
    assert kwargs["autoscale_ttft_correction_factor"] == 2.0


def test_sweep_kwargs_mode_mismatch_raises():
    t_agg = Task(serving_mode="agg", model_path="deepseek-ai/DeepSeek-V3", system_name="h200_sxm")
    with pytest.raises(ValueError, match="got 'agg'"):
        t_agg.sweep_disagg_kwargs(prefill_database=None, decode_database=None)


# ---------------------------------------------------------------------------
# Task.run() entry point
# ---------------------------------------------------------------------------


def test_run_dispatches_to_sweep_agg(monkeypatch):
    """run() loads DB internally and dispatches to sweep_agg for agg mode."""
    from aiconfigurator.sdk import sweep

    captured: dict = {}

    def fake_get_database(system, backend, version, **kwargs):
        captured.setdefault("dbs", []).append((system, backend, version))
        return f"db-{system}-{backend}-{version}"

    def fake_sweep_agg(**kwargs):
        captured["agg_kwargs"] = kwargs
        return "agg-result"

    monkeypatch.setattr("aiconfigurator.sdk.perf_database.get_database_view", fake_get_database)
    monkeypatch.setattr(sweep, "sweep_agg", fake_sweep_agg)

    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
    )
    result = t.run(validate=False)  # this test isolates dispatch; validate() is covered separately
    assert result == "agg-result"
    # DB loaded for the (system, backend, version) triple (the resolve-time
    # fmha fallback loads the same view earlier; dispatch adds one more).
    assert set(captured["dbs"]) == {("h200_sxm", t.backend_name, t.backend_version)}
    assert captured["agg_kwargs"]["model_path"] == "deepseek-ai/DeepSeek-V3"
    assert captured["agg_kwargs"]["database"] == f"db-h200_sxm-{t.backend_name}-{t.backend_version}"


def test_run_dispatches_to_sweep_disagg_with_two_dbs(monkeypatch):
    """run() loads two databases (prefill + decode) for disagg and dispatches."""
    from aiconfigurator.sdk import sweep

    captured: dict = {}

    def fake_get_database(system, backend, version, **kwargs):
        captured.setdefault("dbs", []).append((system, backend, version))
        return f"db-{system}"

    def fake_sweep_disagg(**kwargs):
        captured["disagg_kwargs"] = kwargs
        return "disagg-result"

    monkeypatch.setattr("aiconfigurator.sdk.perf_database.get_database_view", fake_get_database)
    monkeypatch.setattr(sweep, "sweep_disagg", fake_sweep_disagg)

    t = Task(
        serving_mode="disagg",
        prefill_model_path="deepseek-ai/DeepSeek-V3",
        prefill_system_name="h200_sxm",
        decode_model_path="deepseek-ai/DeepSeek-V3",
        decode_system_name="h100_sxm",
    )
    result = t.run()
    assert result == "disagg-result"
    # Both DBs loaded
    systems = [d[0] for d in captured["dbs"]]
    assert "h200_sxm" in systems and "h100_sxm" in systems
    # autoscale defaults to False
    assert captured["disagg_kwargs"]["autoscale"] is False


def test_run_passes_autoscale_flag(monkeypatch):
    from aiconfigurator.sdk import sweep

    captured: dict = {}

    def fake_get_database(*a, **kw):
        return "db"

    def fake_sweep_disagg(**kwargs):
        captured["autoscale"] = kwargs.get("autoscale")
        return "result"

    monkeypatch.setattr("aiconfigurator.sdk.perf_database.get_database_view", fake_get_database)
    monkeypatch.setattr(sweep, "sweep_disagg", fake_sweep_disagg)

    t = Task(
        serving_mode="disagg",
        prefill_model_path="deepseek-ai/DeepSeek-V3",
        prefill_system_name="h200_sxm",
        decode_model_path="deepseek-ai/DeepSeek-V3",
        decode_system_name="h200_sxm",
    )
    t.run(autoscale=True)
    assert captured["autoscale"] is True


def test_run_rejects_autoscale_in_agg_mode():
    t = Task(serving_mode="agg", model_path="deepseek-ai/DeepSeek-V3", system_name="h200_sxm")
    with pytest.raises(ValueError, match="autoscale is only supported in disagg mode"):
        t.run(autoscale=True)


def test_run_forwards_predictor_field_to_sweep_agg(monkeypatch):
    """Task.predictor is plumbed into sweep_agg's predictor kwarg."""
    from aiconfigurator.sdk import sweep

    captured: dict = {}

    def fake_get_database(*a, **kw):
        return "db"

    def fake_sweep_agg(**kwargs):
        captured["predictor"] = kwargs.get("predictor")
        return "agg-result"

    monkeypatch.setattr("aiconfigurator.sdk.perf_database.get_database_view", fake_get_database)
    monkeypatch.setattr(sweep, "sweep_agg", fake_sweep_agg)

    from aiconfigurator.sdk.predictor import AnalyticPredictor

    custom = AnalyticPredictor()  # any Predictor-compatible object
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        predictor=custom,
    )
    t.run()
    assert captured["predictor"] is custom


def test_run_forwards_predictor_field_to_sweep_disagg(monkeypatch):
    """Task.predictor is plumbed into sweep_disagg's predictor kwarg."""
    from aiconfigurator.sdk import sweep

    captured: dict = {}

    def fake_get_database(*a, **kw):
        return "db"

    def fake_sweep_disagg(**kwargs):
        captured["predictor"] = kwargs.get("predictor")
        return "disagg-result"

    monkeypatch.setattr("aiconfigurator.sdk.perf_database.get_database_view", fake_get_database)
    monkeypatch.setattr(sweep, "sweep_disagg", fake_sweep_disagg)

    from aiconfigurator.sdk.predictor import AnalyticPredictor

    custom = AnalyticPredictor()
    t = Task(
        serving_mode="disagg",
        prefill_model_path="deepseek-ai/DeepSeek-V3",
        prefill_system_name="h200_sxm",
        decode_model_path="deepseek-ai/DeepSeek-V3",
        decode_system_name="h200_sxm",
        predictor=custom,
    )
    t.run()
    assert captured["predictor"] is custom


def test_to_dict_skips_predictor_strategy_field():
    """Strategy fields (Python objects) shouldn't appear in to_dict / YAML output."""
    from aiconfigurator.sdk.predictor import AnalyticPredictor

    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        predictor=AnalyticPredictor(),
    )
    d = t.to_dict()
    assert "predictor" not in d


# ---------------------------------------------------------------------------
# Task.run_single_* (single-point evaluation, subsumes cli_estimate)
# ---------------------------------------------------------------------------


def _build_fake_summary(result_dict: dict | None = None, oom: bool = False):
    """Return a MagicMock InferenceSummary."""
    from unittest.mock import MagicMock

    s = MagicMock(name="InferenceSummary")
    s.check_oom.return_value = oom
    s.check_kv_cache_oom.return_value = False
    s.get_result_dict.return_value = (
        result_dict if result_dict is not None else {"tokens/s/gpu": 100.0, "ttft": 50.0, "tpot": 20.0}
    )
    # For disagg: get_summary_df().iloc[0].to_dict()
    import pandas as pd

    s.get_summary_df.return_value = pd.DataFrame([result_dict or {"tokens/s/gpu": 100.0, "ttft": 50.0, "tpot": 20.0}])
    return s


def test_run_single_agg_calls_predict_agg_worker_with_fixed_point(monkeypatch):
    """run_single_agg builds ModelConfig with given dims and calls predict_agg_worker once."""
    from aiconfigurator.sdk import predict

    captured: dict = {}

    def fake_get_database(*a, **kw):
        return "db"

    def fake_get_backend(name):
        from unittest.mock import MagicMock

        return MagicMock(name=f"backend-{name}")

    def fake_get_model(model_path, model_config, backend_name):
        captured["model_config"] = model_config
        from unittest.mock import MagicMock

        return MagicMock(name="model")

    def fake_predict_agg_worker(**kwargs):
        captured["predict_kwargs"] = kwargs
        return _build_fake_summary(result_dict={"tokens/s/gpu": 999.0, "ttft": 42.0, "tpot": 7.0})

    monkeypatch.setattr("aiconfigurator.sdk.perf_database.get_database_view", fake_get_database)
    monkeypatch.setattr("aiconfigurator.sdk.backends.factory.get_backend", fake_get_backend)
    monkeypatch.setattr("aiconfigurator.sdk.models.get_model", fake_get_model)
    monkeypatch.setattr(predict, "predict_agg_worker", fake_predict_agg_worker)

    t = Task(serving_mode="agg", model_path="deepseek-ai/DeepSeek-V3", system_name="h200_sxm")
    result = t.run_single_agg(tp=4, pp=1, dp=1, moe_tp=1, moe_ep=1, batch_size=64)

    # Result is the fake_predict_agg_worker's result_dict
    assert result["tokens/s/gpu"] == 999.0
    assert result["ttft"] == 42.0
    # ModelConfig built with the requested parallelism
    mc = captured["model_config"]
    assert mc.tp_size == 4 and mc.pp_size == 1 and mc.moe_tp_size == 1 and mc.moe_ep_size == 1
    # ctx_tokens defaults to isl
    assert captured["predict_kwargs"]["ctx_tokens"] == t.isl


def test_run_single_agg_rejects_disagg_task():
    t = Task(
        serving_mode="disagg",
        prefill_model_path="deepseek-ai/DeepSeek-V3",
        prefill_system_name="h200_sxm",
        decode_model_path="deepseek-ai/DeepSeek-V3",
        decode_system_name="h200_sxm",
    )
    with pytest.raises(ValueError, match="requires serving_mode='agg'"):
        t.run_single_agg(tp=4, batch_size=64)


def test_run_single_agg_raises_on_oom(monkeypatch):
    """OOM in single-point eval should surface as a clear RuntimeError."""
    from aiconfigurator.sdk import predict

    def fake_get_database(*a, **kw):
        return "db"

    def fake_get_backend(name):
        from unittest.mock import MagicMock

        return MagicMock()

    def fake_get_model(model_path, model_config, backend_name):
        from unittest.mock import MagicMock

        return MagicMock()

    def fake_predict_agg_worker(**kwargs):
        return _build_fake_summary(oom=True)

    monkeypatch.setattr("aiconfigurator.sdk.perf_database.get_database_view", fake_get_database)
    monkeypatch.setattr("aiconfigurator.sdk.backends.factory.get_backend", fake_get_backend)
    monkeypatch.setattr("aiconfigurator.sdk.models.get_model", fake_get_model)
    monkeypatch.setattr(predict, "predict_agg_worker", fake_predict_agg_worker)

    t = Task(serving_mode="agg", model_path="deepseek-ai/DeepSeek-V3", system_name="h200_sxm")
    with pytest.raises(RuntimeError, match="OOM"):
        t.run_single_agg(tp=1, batch_size=999)


def test_run_single_disagg_invokes_both_phases_and_rate_matches(monkeypatch):
    """run_single_disagg calls predict_disagg_worker twice (prefill + decode)
    then rate-matches the pair into one ColumnsDisagg row."""
    from aiconfigurator.sdk import predict

    call_roles: list[str] = []

    def fake_get_database(*a, **kw):
        return "db"

    def fake_get_backend(name):
        from unittest.mock import MagicMock

        return MagicMock()

    def fake_get_model(model_path, model_config, backend_name):
        from unittest.mock import MagicMock

        return MagicMock()

    # Build per-role summary dicts that satisfy _rate_match_dict's schema.
    def _phase_summary(role: str):
        base = {
            "model": "m",
            "isl": 4000,
            "osl": 500,
            "prefix": 0,
            "concurrency": 1,
            "bs": 1,
            "global_bs": 1,
            "tp": 4,
            "pp": 1,
            "dp": 1,
            "moe_tp": 1,
            "moe_ep": 1,
            "parallel": "tp4pp1dp1",
            "ttft": 80.0 if role == "prefill" else 0.0,
            "tpot": 0.0 if role == "prefill" else 25.0,
            "seq/s": 10.0 if role == "prefill" else 5.0,
            "tokens/s/user": 0.0 if role == "prefill" else 40.0,
            "gemm": "fp8",
            "kvcache": "fp8",
            "fmha": "fp8",
            "moe": "fp8",
            "comm": "half",
            "memory": 12.3,
            "backend": "trtllm",
            "version": "1.3.0",
            "system": "h200_sxm",
            "power_w": 500.0,
        }
        return _build_fake_summary(result_dict=base)

    def fake_predict_disagg_worker(**kwargs):
        call_roles.append(kwargs["role"])
        return _phase_summary(kwargs["role"])

    monkeypatch.setattr("aiconfigurator.sdk.perf_database.get_database_view", fake_get_database)
    monkeypatch.setattr("aiconfigurator.sdk.backends.factory.get_backend", fake_get_backend)
    monkeypatch.setattr("aiconfigurator.sdk.models.get_model", fake_get_model)
    monkeypatch.setattr(predict, "predict_disagg_worker", fake_predict_disagg_worker)

    t = Task(
        serving_mode="disagg",
        prefill_model_path="deepseek-ai/DeepSeek-V3",
        prefill_system_name="h200_sxm",
        decode_model_path="deepseek-ai/DeepSeek-V3",
        decode_system_name="h200_sxm",
    )
    row = t.run_single_disagg(
        prefill_tp=4,
        prefill_batch_size=1,
        prefill_num_workers=2,
        decode_tp=2,
        decode_batch_size=64,
        decode_num_workers=4,
    )

    # Both phases invoked
    assert call_roles == ["prefill", "decode"]
    # Rate-matched row has (p)/(d) prefixed columns plus encoder placeholders
    assert "(p)workers" in row and "(d)workers" in row
    assert row["(p)workers"] == 2
    assert row["(d)workers"] == 4
    assert "(e)workers" in row  # encoder placeholders preserved


def test_run_single_disagg_rejects_agg_task():
    t = Task(serving_mode="agg", model_path="deepseek-ai/DeepSeek-V3", system_name="h200_sxm")
    with pytest.raises(ValueError, match="requires serving_mode='disagg'"):
        t.run_single_disagg(prefill_tp=4, decode_tp=2, decode_batch_size=64)


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


def test_validate_agg_happy_path():
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
    )
    t.validate()  # no raise


def test_validate_agg_requires_model_path():
    t = Task(serving_mode="agg")
    with pytest.raises(ValueError, match="agg mode requires model_path"):
        t.validate()


def test_validate_agg_requires_system_name():
    t = Task(serving_mode="agg", model_path="deepseek-ai/DeepSeek-V3")
    with pytest.raises(ValueError, match="agg mode requires system_name"):
        t.validate()


def test_validate_agg_fp8_static_on_sglang_is_data_driven():
    """fp8_static is no longer hard-gated to trtllm; support is decided by the
    perf DB.  h200_sxm/sglang has fp8 GEMM data but no compute_scale/scale_matrix
    overhead tables, so fp8_static is rejected by the DB-side check rather than a
    backend allowlist."""
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        backend_name="sglang",
        backend_version="0.5.10",
        gemm_quant_mode=common.GEMMQuantMode.fp8_static,
    )
    with pytest.raises(ValueError, match="Unsupported gemm quant mode 'fp8_static'"):
        t.validate()


def test_validate_disagg_happy_path():
    t = Task(
        serving_mode="disagg",
        prefill_model_path="deepseek-ai/DeepSeek-V3",
        prefill_system_name="h200_sxm",
        decode_model_path="deepseek-ai/DeepSeek-V3",
        decode_system_name="h200_sxm",
    )
    t.validate()  # no raise


def test_validate_disagg_requires_both_role_model_paths():
    t = Task(
        serving_mode="disagg",
        prefill_model_path="deepseek-ai/DeepSeek-V3",
        prefill_system_name="h200_sxm",
        decode_system_name="h200_sxm",
        # decode_model_path missing
    )
    with pytest.raises(ValueError, match="both prefill_model_path and decode_model_path"):
        t.validate()


def test_validate_disagg_rejects_mismatched_prefill_decode_model_paths():
    """Hetero-model disagg is not supported; mismatch must fail loud."""
    t = Task(
        serving_mode="disagg",
        prefill_model_path="deepseek-ai/DeepSeek-V3",
        prefill_system_name="h200_sxm",
        decode_model_path="Qwen/Qwen3-32B",
        decode_system_name="h200_sxm",
    )
    with pytest.raises(ValueError, match="prefill_model_path == decode_model_path"):
        t.validate()


def test_validate_disagg_fp8_static_is_data_driven_per_role():
    """Per-role fp8_static support is decided by the perf DB, not a trtllm
    allowlist.  h200_sxm/sglang lacks the overhead tables, so the prefill role's
    fp8_static is rejected by the DB-side check."""
    t = Task(
        serving_mode="disagg",
        prefill_model_path="deepseek-ai/DeepSeek-V3",
        prefill_system_name="h200_sxm",
        prefill_backend_name="sglang",
        prefill_backend_version="0.5.10",
        prefill_gemm_quant_mode=common.GEMMQuantMode.fp8_static,
        decode_model_path="deepseek-ai/DeepSeek-V3",
        decode_system_name="h200_sxm",
    )
    with pytest.raises(ValueError, match="Unsupported gemm quant mode 'fp8_static'"):
        t.validate()


def test_validate_invalid_serving_mode_raises():
    t = Task(serving_mode="agg", model_path="deepseek-ai/DeepSeek-V3", system_name="h200_sxm")
    t.serving_mode = "weird"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Invalid serving_mode"):
        t.validate()


# ---------------------------------------------------------------------------
# validate() database-dependent checks
# ---------------------------------------------------------------------------


def test_validate_database_check_passes_for_supported_quant():
    """Valid quant modes against a real perf DB should pass full validate()."""
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        backend_name="trtllm",
        # HF-inferred quant modes are by definition supported by the DB
    )
    t.validate()  # no raise


def test_validate_database_check_rejects_unsupported_quant():
    """Setting a quant mode the DB doesn't list should raise from the DB layer."""
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        backend_name="trtllm",
    )
    # Force a quant mode that doesn't exist for context_mla on this DB.
    # int4_awq is implausible for DeepSeek MLA.
    t.fmha_quant_mode = common.FMHAQuantMode.bfloat16  # may or may not be in DB
    # Use a clearly unsupported gemm mode via direct field write.
    t.gemm_quant_mode = common.GEMMQuantMode.int4_wo
    # Either passes (if DB happens to have mxfp4) or raises a clear ValueError.
    try:
        t.validate()
    except ValueError as exc:
        assert "Unsupported gemm" in str(exc) or "Supported gemm" in str(exc)


def test_validate_moe_quant_transfer_reachable_in_hybrid():
    """supported_quant_mode is a data-presence list. A MoE quant absent from the DB
    but sharing a (memory, compute) profile with a supported quant is XQUANT-reachable
    in HYBRID (operations/moe.py), so HYBRID validate must admit it while SILICON stays
    strict. Kimi-K2.5 infers int4_wo MoE (profile (0.5,1)) which b200/trtllm has no data
    for, but w4a16_mxfp4 (same profile) is collected."""

    def make(mode, policy=None):
        return Task(
            serving_mode="agg",
            model_path="moonshotai/Kimi-K2.5",
            system_name="b200_sxm",
            backend_name="trtllm",
            backend_version="1.3.0rc10",
            database_mode=mode,
            transfer_policy=policy,
        )

    with pytest.raises(ValueError, match="Unsupported moe quant mode 'int4_wo'"):
        make("SILICON").validate()
    make("HYBRID").validate()  # default policy (all on) -> XQUANT reachable -> no raise
    make("HYBRID", "xquant").validate()  # XQUANT explicitly enabled -> no raise

    # The admission must respect the transfer policy: if XQUANT is disabled, moe.py would
    # reject the quant at query time, so validate must NOT pre-admit it.
    for disabled in ("off", "conservative"):
        with pytest.raises(ValueError, match="Unsupported moe quant mode 'int4_wo'"):
            make("HYBRID", disabled).validate()


def test_validate_skips_db_check_when_database_unavailable():
    """If DB can't be loaded, DB validation silently skips (caller sees other errors)."""
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        backend_name="trtllm",
    )
    # Force a non-existent backend_version → DB load fails → DB check skipped
    t.backend_version = "9.99.99-nonexistent"
    t.validate()  # static checks pass, DB silently skipped


def test_validate_fp8_static_fails_fast_when_db_unavailable_in_silicon():
    """SILICON mode can't confirm fp8_static overhead data without the DB, so an
    unloadable (system, backend, version) must fail fast instead of deferring."""
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        backend_name="trtllm",
        gemm_quant_mode=common.GEMMQuantMode.fp8_static,
    )
    t.backend_version = "9.99.99-nonexistent"  # DB load fails -> can't confirm support
    with pytest.raises(ValueError, match="fp8_static GEMM mode requires perf data"):
        t.validate()


def test_validate_skips_db_check_for_deepseekv4_synthetic_mode():
    """DeepSeek-V4 in synthetic database modes skips DB validation entirely."""
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",  # use V3 since we just need is_moe + family
        system_name="h200_sxm",
        backend_name="trtllm",
    )
    # Manually force model_family to simulate DeepSeek-V4
    t._model_family = "DEEPSEEKV4"
    t.database_mode = "SOL"
    # Set an obviously unsupported quant mode; should be skipped because of synthetic mode
    t.gemm_quant_mode = common.GEMMQuantMode.int4_wo
    t.validate()  # no raise — synthetic mode allowance kicks in


# ---------------------------------------------------------------------------
# to_dict() / to_yaml()
# ---------------------------------------------------------------------------


def test_to_dict_emits_resolved_state_with_enum_names():
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        gemm_quant_mode=common.GEMMQuantMode.fp8,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
    )
    d = t.to_dict()
    assert d["serving_mode"] == "agg"
    assert d["model_path"] == "deepseek-ai/DeepSeek-V3"
    # Enums emitted as .name strings (round-trippable through from_yaml)
    assert d["gemm_quant_mode"] == "fp8"
    assert d["kvcache_quant_mode"] == "fp8"
    # Backend version resolved automatically
    assert d["backend_version"] is not None
    # Search candidates populated
    assert d["agg_tp_candidates"] == [1, 2, 4, 8]


def test_to_dict_excludes_internal_fields():
    t = Task(serving_mode="agg", model_path="deepseek-ai/DeepSeek-V3", system_name="h200_sxm")
    d = t.to_dict()
    # Internal underscore-prefixed fields not exposed
    assert not any(k.startswith("_") for k in d)


def test_to_yaml_round_trips_through_from_yaml():
    """Ensure to_yaml output is parseable by from_yaml (modulo HF re-resolution)."""
    import yaml

    t1 = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
    )
    yaml_text = t1.to_yaml()
    yaml_data = yaml.safe_load(yaml_text)
    t2 = Task.from_yaml(yaml_data)

    # Core fields preserved
    assert t2.serving_mode == t1.serving_mode
    assert t2.model_path == t1.model_path
    assert t2.gemm_quant_mode == t1.gemm_quant_mode
    assert t2.agg_tp_candidates == t1.agg_tp_candidates
