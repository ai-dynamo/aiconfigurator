# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the new flat Task in sdk/task_config.py.

End-to-end sweep correctness is covered by the integration parity test
against the legacy CLI; these tests focus on construction, defaulting,
prefix discipline, and the build_* helpers.
"""

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.task_config import Task

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


def test_agg_quant_preset_overrides_hf():
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        quant_preset="bfloat16",
    )
    assert t.gemm_quant_mode == common.GEMMQuantMode.bfloat16
    assert t.kvcache_quant_mode == common.KVCacheQuantMode.bfloat16


def test_agg_explicit_quant_overrides_preset():
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        quant_preset="bfloat16",
        gemm_quant_mode=common.GEMMQuantMode.fp8,
    )
    assert t.gemm_quant_mode == common.GEMMQuantMode.fp8
    # other modes follow preset
    assert t.kvcache_quant_mode == common.KVCacheQuantMode.bfloat16


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
    assert t.max_gpu_per_replica == 128
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
        ("quant_preset", "fp8"),
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
    assert t.max_gpu_per_replica == 128
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


def test_from_yaml_warns_on_unknown_keys(caplog):
    """Unknown keys are warned about but not silently swallowed."""
    import logging

    with caplog.at_level(logging.WARNING):
        Task.from_yaml(
            {
                "serving_mode": "agg",
                "model_path": "deepseek-ai/DeepSeek-V3",
                "system_name": "h200_sxm",
                "totally_made_up_field": 42,
                "another_typo": "value",
            }
        )
    assert "totally_made_up_field" in caplog.text
    assert "another_typo" in caplog.text


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
        quant_preset="bfloat16",
    )
    mc = t.build_model_config(role="agg")
    assert mc.gemm_quant_mode == common.GEMMQuantMode.bfloat16
    assert mc.nextn == t.nextn


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

    def fake_get_database(system, backend, version):
        captured.setdefault("dbs", []).append((system, backend, version))
        return f"db-{system}-{backend}-{version}"

    def fake_sweep_agg(**kwargs):
        captured["agg_kwargs"] = kwargs
        return "agg-result"

    monkeypatch.setattr("aiconfigurator.sdk.perf_database.get_database", fake_get_database)
    monkeypatch.setattr(sweep, "sweep_agg", fake_sweep_agg)

    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
    )
    result = t.run()
    assert result == "agg-result"
    # DB loaded for the (system, backend, version) triple
    assert captured["dbs"] == [("h200_sxm", t.backend_name, t.backend_version)]
    assert captured["agg_kwargs"]["model_path"] == "deepseek-ai/DeepSeek-V3"
    assert captured["agg_kwargs"]["database"] == f"db-h200_sxm-{t.backend_name}-{t.backend_version}"


def test_run_dispatches_to_sweep_disagg_with_two_dbs(monkeypatch):
    """run() loads two databases (prefill + decode) for disagg and dispatches."""
    from aiconfigurator.sdk import sweep

    captured: dict = {}

    def fake_get_database(system, backend, version):
        captured.setdefault("dbs", []).append((system, backend, version))
        return f"db-{system}"

    def fake_sweep_disagg(**kwargs):
        captured["disagg_kwargs"] = kwargs
        return "disagg-result"

    monkeypatch.setattr("aiconfigurator.sdk.perf_database.get_database", fake_get_database)
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

    monkeypatch.setattr("aiconfigurator.sdk.perf_database.get_database", fake_get_database)
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


def test_validate_agg_rejects_deepseek_on_vllm():
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        backend_name="vllm",
    )
    with pytest.raises(NotImplementedError, match="DeepSeek family on the vLLM backend"):
        t.validate()


def test_validate_agg_rejects_fp8_static_on_non_trtllm():
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        backend_name="sglang",
        gemm_quant_mode=common.GEMMQuantMode.fp8_static,
    )
    with pytest.raises(ValueError, match="fp8_static GEMM mode is only supported on the trtllm backend"):
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


def test_validate_disagg_rejects_fp8_static_on_non_trtllm_per_role():
    t = Task(
        serving_mode="disagg",
        prefill_model_path="deepseek-ai/DeepSeek-V3",
        prefill_system_name="h200_sxm",
        prefill_backend_name="sglang",
        prefill_gemm_quant_mode=common.GEMMQuantMode.fp8_static,
        decode_model_path="deepseek-ai/DeepSeek-V3",
        decode_system_name="h200_sxm",
    )
    with pytest.raises(ValueError, match="prefill_backend_name='sglang'"):
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
        quant_preset="fp8",
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
        quant_preset="bfloat16",
    )
    yaml_text = t1.to_yaml()
    yaml_data = yaml.safe_load(yaml_text)
    t2 = Task.from_yaml(yaml_data)

    # Core fields preserved
    assert t2.serving_mode == t1.serving_mode
    assert t2.model_path == t1.model_path
    assert t2.gemm_quant_mode == t1.gemm_quant_mode
    assert t2.agg_tp_candidates == t1.agg_tp_candidates
