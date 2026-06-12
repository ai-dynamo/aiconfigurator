import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "collector" / "layerwise" / "diagnostics"))

from compare_aic_layerwise_fpm import (
    GENERATION_PER_OP_FIELDS,
    MIXED_PER_OP_FIELDS,
    _add_generation_per_op_columns,
    _add_mixed_per_op_columns,
    _context_pathology_reasons,
    _decode_pathology_reasons,
    _infer_observed_fpm_context_budget,
    _load_fpm,
    _load_fpm_max_num_batched_tokens,
    _match_decode,
    _mixed_pathology_reasons,
    _model_defaults,
    _nearest_available_generation_kv,
    _nonterminal_mixed_chunk_counter_ids,
    _prepare_moe_overlay_systems_root,
)
from compare_layerwise_fpm import compare_layerwise_to_fpm
from summarize_layerwise_fpm_comparisons import summarize_manifest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations.layerwise import _interpolated_layer_scale_metadata


def _write(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n")


def test_diagnostic_model_defaults_include_canonical_deepseek_v4_flash() -> None:
    model = _model_defaults("deepseek-ai/DeepSeek-V4-Flash", tp=1, moe_tp=1, ep=1)

    assert model.model_path == "deepseek-ai/DeepSeek-V4-Flash"
    assert model._num_layers == 43
    assert model._hidden_size == 4096
    assert model._topk == 6
    assert model._num_experts == 256
    assert model._moe_inter_size == 2048
    assert model.config.gemm_quant_mode == common.GEMMQuantMode.fp8_block
    assert model.config.moe_quant_mode == common.MoEQuantMode.w4a8_mxfp4_mxfp8
    assert model.config.kvcache_quant_mode == common.KVCacheQuantMode.fp8
    assert model.config.fmha_quant_mode == common.FMHAQuantMode.fp8
    assert model.config.nextn == 0
    assert isinstance(model.extra_params, common.DeepSeekV4Config)


def test_prepare_moe_overlay_systems_root_keeps_base_data_and_overlays_moe(tmp_path) -> None:
    systems_root = tmp_path / "systems"
    version_root = systems_root / "data" / "b300_sxm" / "vllm" / "0.20.1"
    version_root.mkdir(parents=True)
    _write(
        systems_root / "b300_sxm.yaml",
        """data_dir: data/b300_sxm
gpu:
  sm_version: 103
node:
  num_gpus_per_node: 8
misc:
  nccl_version: '2.27'
""",
    )
    _write(version_root / "layerwise_perf.csv", "framework,version\nvLLM,0.20.1")
    local_moe = tmp_path / "moe_perf.txt"
    _write(
        local_moe,
        """framework,version,device,op_name,kernel_source,moe_dtype,num_tokens,hidden_size,inter_size,topk,num_experts,moe_tp_size,moe_ep_size,distribution,latency
VLLM,0.20.1,test,moe,vllm_mxfp4_moe,w4a8_mxfp4_mxfp8,1,4096,2048,6,256,1,1,power_law_1.2,0.1
""",
    )

    overlay = Path(
        _prepare_moe_overlay_systems_root(
            systems_root=str(systems_root),
            moe_perf_file=local_moe,
            output=tmp_path / "compare.csv",
        )
    )

    overlay_version = overlay / "data" / "b300_sxm" / "vllm" / "0.20.1"
    assert (overlay / "b300_sxm.yaml").is_file()
    assert (overlay_version / "layerwise_perf.csv").exists()
    assert (overlay_version / "moe_perf.txt").read_text().count("w4a8_mxfp4_mxfp8") == 1
    assert not (overlay_version / "moe_perf.parquet").exists()


def test_add_mixed_per_op_columns_emits_stable_schema() -> None:
    row: dict[str, object] = {}

    _add_mixed_per_op_columns(
        row,
        {"mixed_layerwise_context_combined": 12.5},
        {"mixed_layerwise_context_combined": "silicon"},
    )

    for op_name in MIXED_PER_OP_FIELDS:
        assert f"aic_op_{op_name}" in row
        assert f"aic_source_{op_name}" in row
    assert row["aic_op_mixed_layerwise_context_combined"] == 12.5
    assert row["aic_source_mixed_layerwise_context_combined"] == "silicon"
    assert row["aic_op_mixed_moe"] == 0.0
    assert row["aic_source_mixed_moe"] == ""


def test_add_generation_per_op_columns_emits_stable_schema() -> None:
    row: dict[str, object] = {}

    _add_generation_per_op_columns(
        row,
        {"generation_layerwise": 3.25, "generation_moe": 0.5},
        {"generation_layerwise": "silicon", "generation_moe": "estimated"},
    )

    for op_name in GENERATION_PER_OP_FIELDS:
        assert f"aic_op_{op_name}" in row
        assert f"aic_source_{op_name}" in row
    assert row["aic_op_generation_layerwise"] == 3.25
    assert row["aic_source_generation_layerwise"] == "silicon"
    assert row["aic_op_generation_moe"] == 0.5
    assert row["aic_source_generation_moe"] == "estimated"
    assert row["aic_op_generation_moe_ep_alltoall"] == 0.0
    assert row["aic_source_generation_moe_ep_alltoall"] == ""


def test_summarize_layerwise_fpm_comparisons_manifest(tmp_path) -> None:
    comparison = tmp_path / "compare.csv"
    _write(
        comparison,
        """phase,shape,error_pct
mixed,ctx128_gen1,10.0
mixed,ctx256_gen1,-5.0
gen,bs1_past4096,2.0
""",
    )
    manifest = tmp_path / "manifest.csv"
    _write(
        manifest,
        f"""model,tp,moe_tp,ep,phase,source_phase,comparison_csv
Qwen/Test,TP8,1,8,mixed_broader,mixed,{comparison.name}
""",
    )

    rows = summarize_manifest(manifest)

    assert rows == [
        {
            "model": "Qwen/Test",
            "tp": "TP8",
            "moe_tp": "1",
            "ep": "8",
            "phase": "mixed_broader",
            "rows": 2,
            "mape_pct": pytest.approx(7.5),
            "max_abs_error_pct": pytest.approx(10.0),
            "worst_shape": "ctx128_gen1",
            "worst_error_pct": pytest.approx(10.0),
            "comparison_csv": comparison.name,
        }
    ]


def test_mixed_pathology_filter_flags_low_latency_large_context_row() -> None:
    rows = [
        {"phase": "mixed", "ctx_tokens": "8000", "latency_ms": "410.0"},
        {"phase": "mixed", "ctx_tokens": "8050", "latency_ms": "420.0"},
        {"phase": "mixed", "ctx_tokens": "8100", "latency_ms": "430.0"},
        {"phase": "mixed", "ctx_tokens": "8150", "latency_ms": "440.0"},
        {"phase": "mixed", "ctx_tokens": "8200", "latency_ms": "450.0"},
        {"phase": "mixed", "ctx_tokens": "8188", "latency_ms": "11.6"},
        {"phase": "mixed", "ctx_tokens": "200", "latency_ms": "2.0"},
    ]

    reasons = _mixed_pathology_reasons(
        rows,
        tiny_ctx_tokens=128,
        min_ctx_tokens=1024,
        peer_ctx_fraction=0.05,
        peer_ctx_min_window=128,
        min_peer_count=5,
        latency_fraction=0.25,
        high_latency_factor=1.75,
    )

    assert set(reasons) == {5}
    assert reasons[5].startswith("mixed_latency_below_peer_envelope:")


def test_mixed_pathology_filter_flags_high_latency_large_context_row() -> None:
    rows = [
        {"phase": "mixed", "ctx_tokens": "8000", "latency_ms": "410.0"},
        {"phase": "mixed", "ctx_tokens": "8050", "latency_ms": "420.0"},
        {"phase": "mixed", "ctx_tokens": "8100", "latency_ms": "430.0"},
        {"phase": "mixed", "ctx_tokens": "8150", "latency_ms": "440.0"},
        {"phase": "mixed", "ctx_tokens": "8200", "latency_ms": "450.0"},
        {"phase": "mixed", "ctx_tokens": "8188", "latency_ms": "815.0"},
        {"phase": "mixed", "ctx_tokens": "200", "latency_ms": "2.0"},
    ]

    reasons = _mixed_pathology_reasons(
        rows,
        tiny_ctx_tokens=128,
        min_ctx_tokens=1024,
        peer_ctx_fraction=0.05,
        peer_ctx_min_window=128,
        min_peer_count=5,
        latency_fraction=0.25,
        high_latency_factor=1.75,
    )

    assert set(reasons) == {5}
    assert reasons[5].startswith("mixed_latency_above_peer_envelope:")


def test_mixed_pathology_filter_flags_isolated_transition_spike() -> None:
    rows = [
        {"phase": "mixed", "ctx_tokens": "1360", "latency_ms": "50.0"},
        {"phase": "mixed", "ctx_tokens": "1632", "latency_ms": "1002.0"},
        {"phase": "mixed", "ctx_tokens": "1904", "latency_ms": "46.0"},
        {"phase": "mixed", "ctx_tokens": "2002", "latency_ms": "52.0"},
    ]

    reasons = _mixed_pathology_reasons(
        rows,
        tiny_ctx_tokens=128,
        min_ctx_tokens=128,
        peer_ctx_fraction=0.05,
        peer_ctx_min_window=512,
        min_peer_count=3,
        latency_fraction=0.25,
        high_latency_factor=1.75,
    )

    assert reasons[1].startswith("mixed_latency_above_peer_envelope:")


def test_mixed_pathology_filter_flags_adjacent_sequence_spike() -> None:
    rows = [
        {
            "phase": "mixed",
            "ctx_tokens": "8160",
            "ctx_kv_tokens": "0",
            "decode_requests": "15",
            "mean_decode_kv_tokens": "5339.8",
            "latency_ms": "47.3",
        },
        {
            "phase": "mixed",
            "ctx_tokens": "4352",
            "ctx_kv_tokens": "8160",
            "decode_requests": "15",
            "mean_decode_kv_tokens": "5340.8",
            "latency_ms": "87.7",
        },
        {
            "phase": "mixed",
            "ctx_tokens": "216",
            "ctx_kv_tokens": "12512",
            "decode_requests": "15",
            "mean_decode_kv_tokens": "5341.8",
            "latency_ms": "48.5",
        },
    ]

    reasons = _mixed_pathology_reasons(
        rows,
        tiny_ctx_tokens=128,
        min_ctx_tokens=128,
        peer_ctx_fraction=0.05,
        peer_ctx_min_window=512,
        min_peer_count=3,
        latency_fraction=0.25,
        high_latency_factor=1.75,
    )

    assert set(reasons) == {1}
    assert reasons[1].startswith("mixed_latency_above_adjacent_sequence_envelope:")


def test_mixed_pathology_filter_flags_tiny_continuation_tail() -> None:
    rows = [
        {
            "phase": "mixed",
            "ctx_tokens": "49",
            "ctx_kv_tokens": "4047",
            "decode_requests": "13",
            "latency_ms": "12.0",
        },
        {
            "phase": "mixed",
            "ctx_tokens": "4096",
            "ctx_kv_tokens": "0",
            "decode_requests": "1",
            "latency_ms": "208.0",
        },
    ]

    reasons = _mixed_pathology_reasons(
        rows,
        tiny_ctx_tokens=128,
        min_ctx_tokens=1024,
        peer_ctx_fraction=0.05,
        peer_ctx_min_window=128,
        min_peer_count=5,
        latency_fraction=0.25,
        high_latency_factor=1.75,
    )

    assert set(reasons) == {0}
    assert reasons[0].startswith("mixed_tiny_continuation_tail:")


def test_nonterminal_mixed_chunk_counter_ids_flags_following_prefix() -> None:
    rows = [
        {
            "phase": "mixed",
            "counter_id": "10",
            "ctx_tokens": "4096",
            "ctx_requests": "1",
            "ctx_kv_tokens": "8192",
        },
        {
            "phase": "context",
            "counter_id": "11",
            "ctx_tokens": "128",
            "ctx_requests": "1",
            "ctx_kv_tokens": "12288",
        },
        {
            "phase": "mixed",
            "counter_id": "12",
            "ctx_tokens": "8192",
            "ctx_requests": "1",
            "ctx_kv_tokens": "0",
        },
    ]

    assert _nonterminal_mixed_chunk_counter_ids(rows) == {"10"}


def test_mixed_pathology_filter_flags_rows_below_decode_floor() -> None:
    rows = [
        {
            "phase": "mixed",
            "ctx_tokens": "224",
            "ctx_kv_tokens": "0",
            "decode_requests": "31",
            "mean_decode_kv_tokens": "4096.0",
            "latency_ms": "15.0",
        }
    ]
    decode_rows = [
        {
            "phase": "decode",
            "decode_requests": "31",
            "mean_decode_kv_tokens": str(4096 + offset),
            "latency_ms": "20.0",
        }
        for offset in (0, 1, 2)
    ]

    reasons = _mixed_pathology_reasons(
        rows,
        decode_rows=decode_rows,
        tiny_ctx_tokens=128,
        min_ctx_tokens=1024,
        peer_ctx_fraction=0.05,
        peer_ctx_min_window=128,
        min_peer_count=5,
        latency_fraction=0.25,
        high_latency_factor=1.75,
    )

    assert set(reasons) == {0}
    assert reasons[0].startswith("mixed_latency_below_decode_floor:")


def test_mixed_pathology_filter_flags_large_context_rows_below_decode_floor() -> None:
    rows = [
        {
            "phase": "mixed",
            "ctx_tokens": "928",
            "ctx_kv_tokens": "3168",
            "decode_requests": "3",
            "mean_decode_kv_tokens": "4096.3",
            "latency_ms": "1.6",
        }
    ]
    decode_rows = [
        {
            "phase": "decode",
            "decode_requests": "3",
            "mean_decode_kv_tokens": str(4096 + offset),
            "latency_ms": "4.0",
        }
        for offset in (0, 1, 2)
    ]

    reasons = _mixed_pathology_reasons(
        rows,
        decode_rows=decode_rows,
        tiny_ctx_tokens=128,
        min_ctx_tokens=128,
        peer_ctx_fraction=0.05,
        peer_ctx_min_window=128,
        min_peer_count=5,
        latency_fraction=0.25,
        high_latency_factor=1.75,
    )

    assert set(reasons) == {0}
    assert reasons[0].startswith("mixed_latency_below_decode_floor:")


def test_context_pathology_filter_flags_continuation_floor() -> None:
    rows = [
        {
            "phase": "context",
            "ctx_tokens": "928",
            "ctx_kv_tokens": "3168",
            "latency_ms": "1.4",
        },
        {
            "phase": "context",
            "ctx_tokens": "928",
            "ctx_kv_tokens": "3168",
            "latency_ms": "43.0",
        },
    ]

    reasons = _context_pathology_reasons(
        rows,
        min_continuation_ctx_tokens=128,
        continuation_min_latency_ms=5.0,
    )

    assert set(reasons) == {0}
    assert reasons[0].startswith("context_continuation_below_latency_floor:")


def test_context_pathology_filter_flags_nonterminal_prefill_chunk() -> None:
    rows = [
        {
            "phase": "context",
            "ctx_tokens": "3168",
            "ctx_requests": "1",
            "ctx_kv_tokens": "0",
            "latency_ms": "80.0",
        },
        {
            "phase": "context",
            "ctx_tokens": "928",
            "ctx_requests": "1",
            "ctx_kv_tokens": "3168",
            "latency_ms": "1.4",
        },
        {
            "phase": "context",
            "ctx_tokens": "1024",
            "ctx_requests": "1",
            "ctx_kv_tokens": "0",
            "latency_ms": "35.0",
        },
    ]

    reasons = _context_pathology_reasons(
        rows,
        min_continuation_ctx_tokens=128,
        continuation_min_latency_ms=5.0,
    )

    assert set(reasons) == {0, 1}
    assert reasons[0].startswith("context_nonterminal_prefill_chunk:")
    assert reasons[1].startswith("context_continuation_below_latency_floor:")


def test_fpm_context_budget_uses_observed_scheduler_rows(tmp_path: Path) -> None:
    phase_csv = tmp_path / "fpm_metrics_phase.csv"
    _write(
        phase_csv,
        """
phase,counter_id,ctx_tokens,ctx_requests,ctx_kv_tokens,decode_requests,mean_decode_kv_tokens,latency_ms
context,1,3696,1,0,0,0,40.0
context,2,400,1,3696,0,0,40.0
mixed,3,8096,1,0,4,4096,45.0
decode,4,0,0,0,4,4096,4.0
""",
    )
    _write(
        tmp_path / "effective_vllm_config.json",
        """
{"scheduler_config.max_num_batched_tokens": 2048}
""",
    )

    assert _infer_observed_fpm_context_budget(phase_csv) == 8096
    assert _load_fpm_max_num_batched_tokens(phase_csv) == 8096


def test_interpolated_layer_scale_metadata_avoids_double_scaling() -> None:
    data = {
        128: {
            0: {
                "latency": 13.5,
                "measured_layer_count": 40,
                "layer_multiplier": 40,
            }
        },
        400: {
            3696: {
                "latency": 55.0,
                "measured_layer_count": 1,
                "layer_multiplier": 1,
            }
        },
    }

    assert _interpolated_layer_scale_metadata(data) == (1.0, 1.0)


def test_interpolated_layer_scale_metadata_preserves_legacy_missing_scale() -> None:
    data = {
        4096: {
            2048: {
                "latency": 0.46,
            }
        },
        8192: {
            2048: {
                "latency": 0.91,
            },
            0: {
                "latency": 66.8,
                "measured_layer_count": 64,
                "layer_multiplier": 64,
            }
        },
    }

    assert _interpolated_layer_scale_metadata(data) is None


def test_decode_pathology_filter_flags_isolated_high_latency_row() -> None:
    rows = [
        {
            "phase": "decode",
            "decode_requests": "2",
            "mean_decode_kv_tokens": "4101.500",
            "latency_ms": "803.961",
        },
        {
            "phase": "decode",
            "decode_requests": "2",
            "mean_decode_kv_tokens": "4102.000",
            "latency_ms": "8.733",
        },
        {
            "phase": "decode",
            "decode_requests": "1",
            "mean_decode_kv_tokens": "4101.000",
            "latency_ms": "8.215",
        },
    ]

    reasons = _decode_pathology_reasons(
        rows,
        peer_kv_window=8.0,
        peer_batch_window=2,
        min_peer_count=1,
        latency_factor=5.0,
        min_latency_ms=20.0,
    )

    assert set(reasons) == {0}
    assert reasons[0].startswith("decode_latency_above_peer_envelope:")


def test_decode_pathology_filter_flags_decode_tail_after_prefill() -> None:
    rows = [
        {
            "phase": "mixed",
            "ctx_tokens": "79",
            "decode_requests": "3",
            "mean_decode_kv_tokens": "4099.000",
            "latency_ms": "29.344",
        },
        {
            "phase": "decode",
            "decode_requests": "4",
            "mean_decode_kv_tokens": "4099.000",
            "latency_ms": "30.084",
        },
        {
            "phase": "decode",
            "decode_requests": "3",
            "mean_decode_kv_tokens": "4099.000",
            "latency_ms": "8.950",
        },
        {
            "phase": "decode",
            "decode_requests": "2",
            "mean_decode_kv_tokens": "4100.000",
            "latency_ms": "7.677",
        },
        {
            "phase": "decode",
            "decode_requests": "1",
            "mean_decode_kv_tokens": "4101.000",
            "latency_ms": "7.752",
        },
    ]

    reasons = _decode_pathology_reasons(
        rows,
        peer_kv_window=8.0,
        peer_batch_window=2,
        min_peer_count=1,
        latency_factor=5.0,
        min_latency_ms=20.0,
    )

    assert set(reasons) == {1, 2, 3, 4}
    assert reasons[1].startswith("decode_segment_after_prefill:")


def test_load_fpm_returns_filtered_row_audit(tmp_path) -> None:
    fpm = tmp_path / "fpm.csv"
    _write(
        fpm,
        """
phase,counter_id,worker_id,dp_rank,ctx_tokens,ctx_requests,ctx_kv_tokens,decode_tokens,decode_requests,decode_kv_tokens,mean_decode_kv_tokens,queued_ctx_tokens,queued_ctx_requests,queued_decode_requests,queued_decode_kv_tokens,latency_ms
decode,1,w,0,0,0,0,2,2,8203,4101.500,0,0,0,0,803.961
decode,2,w,0,0,0,0,2,2,8204,4102.000,0,0,0,0,8.733
context,3,w,0,128,1,0,0,0,0,0.000,0,0,0,0,17.0
""",
    )

    context, decode, filtered = _load_fpm(fpm, filter_pathological_decode=True)

    assert context[(1, 128, 0)] == [17.0]
    assert decode[(2, 4102.0)] == [8.733]
    assert (2, 4101.5) not in decode
    assert len(filtered) == 1
    assert filtered[0]["counter_id"] == "1"
    assert filtered[0]["reason"].startswith("decode_latency_above_peer_envelope:")


def test_load_fpm_reconstructs_leading_chunked_context_requests(tmp_path) -> None:
    fpm = tmp_path / "fpm.csv"
    _write(
        fpm,
        """
phase,counter_id,worker_id,dp_rank,ctx_tokens,ctx_requests,ctx_kv_tokens,decode_tokens,decode_requests,decode_kv_tokens,mean_decode_kv_tokens,queued_ctx_tokens,queued_ctx_requests,queued_decode_requests,queued_decode_kv_tokens,latency_ms
context,1,w,0,128,1,0,0,0,0,0.000,0,0,0,0,17.0
context,2,w,0,2048,1,0,0,0,0,0.000,0,0,0,0,100.0
context,3,w,0,2048,1,2048,0,0,0,0.000,0,0,0,0,110.0
decode,4,w,0,0,0,0,1,1,4096,4096.000,0,0,0,0,8.0
context,5,w,0,2048,1,0,0,0,0,0.000,0,0,0,0,999.0
context,6,w,0,2048,1,2048,0,0,0,0.000,0,0,0,0,999.0
""",
    )

    context, decode, _ = _load_fpm(fpm, filter_pathological_context=True)

    assert context[(1, 128, 0)] == [17.0]
    assert context[(1, 4096, 0)] == [210.0]
    assert (1, 2048, 0) not in context
    assert decode[(1, 4096.0)] == [8.0]


def test_decode_comparison_uses_exact_kv_bin(tmp_path) -> None:
    layerwise = tmp_path / "layerwise.csv"
    fpm = tmp_path / "fpm.csv"
    _write(
        layerwise,
        """
framework,framework_version,system,model,attn_tp,moe_tp,ep,num_slots,gemm_quant,moe_quant,attn_quant,kv_quant,phase,batch_size,new_tokens,past_kv,layer_type,layer_index,measured_layer_count,layer_multiplier,latency_ms,rms_latency_ms,rms_kernel_count,includes_moe,vllm_config_hash
vLLM,0.20.1,gpu,model,1,1,1,,bf16,bf16,bf16,bf16,gen,1,1,4096,dense,0,1,4,1.0,0,0,false,hash
""",
    )
    _write(
        fpm,
        """
phase,counter_id,worker_id,dp_rank,ctx_tokens,ctx_requests,ctx_kv_tokens,decode_tokens,decode_requests,decode_kv_tokens,mean_decode_kv_tokens,queued_ctx_tokens,queued_ctx_requests,queued_decode_requests,queued_decode_kv_tokens,latency_ms
decode,1,w,0,0,0,0,1,1,4096,4096.000,0,0,0,0,4.0
decode,2,w,0,0,0,0,1,1,4097,4097.000,0,0,0,0,40.0
""",
    )

    rows = compare_layerwise_to_fpm(
        layerwise,
        fpm,
        aggregation="trimmed_mean",
        decode_match="nearest",
        max_decode_kv_distance=4.0,
        decode_pool_forward_window=6.0,
    )

    assert len(rows) == 1
    assert rows[0]["fpm_decode_kv"] == "4096.000"
    assert rows[0]["fpm_match"] == "exact"
    assert rows[0]["layerwise_ms"] == pytest.approx(4.0)
    assert rows[0]["fpm_ms"] == pytest.approx(4.0)
    assert rows[0]["error_pct"] == pytest.approx(0.0)


def test_decode_comparison_labels_nearest_kv_bin(tmp_path) -> None:
    layerwise = tmp_path / "layerwise.csv"
    fpm = tmp_path / "fpm.csv"
    _write(
        layerwise,
        """
framework,framework_version,system,model,attn_tp,moe_tp,ep,num_slots,gemm_quant,moe_quant,attn_quant,kv_quant,phase,batch_size,new_tokens,past_kv,layer_type,layer_index,measured_layer_count,layer_multiplier,latency_ms,rms_latency_ms,rms_kernel_count,includes_moe,vllm_config_hash
vLLM,0.20.1,gpu,model,1,1,1,,bf16,bf16,bf16,bf16,gen,4,1,4096,dense,0,4,4,12.0,0,0,false,hash
""",
    )
    _write(
        fpm,
        """
phase,counter_id,worker_id,dp_rank,ctx_tokens,ctx_requests,ctx_kv_tokens,decode_tokens,decode_requests,decode_kv_tokens,mean_decode_kv_tokens,queued_ctx_tokens,queued_ctx_requests,queued_decode_requests,queued_decode_kv_tokens,latency_ms
decode,1,w,0,0,0,0,4,4,16388,4097.000,0,0,0,0,10.0
decode,2,w,0,0,0,0,4,4,16392,4098.000,0,0,0,0,20.0
""",
    )

    rows = compare_layerwise_to_fpm(
        layerwise,
        fpm,
        aggregation="mean",
        decode_match="nearest",
        max_decode_kv_distance=4.0,
        decode_pool_forward_window=6.0,
    )

    assert len(rows) == 1
    assert rows[0]["fpm_decode_kv"] == "4097.000"
    assert rows[0]["fpm_match"] == "nearest"
    assert rows[0]["layerwise_ms"] == pytest.approx(12.0)
    assert rows[0]["fpm_ms"] == pytest.approx(10.0)
    assert rows[0]["error_pct"] == pytest.approx(20.0)


def test_decode_comparison_can_apply_explicit_kv_offset(tmp_path) -> None:
    layerwise = tmp_path / "layerwise.csv"
    fpm = tmp_path / "fpm.csv"
    _write(
        layerwise,
        """
framework,framework_version,system,model,attn_tp,moe_tp,ep,num_slots,gemm_quant,moe_quant,attn_quant,kv_quant,phase,batch_size,new_tokens,past_kv,layer_type,layer_index,measured_layer_count,layer_multiplier,latency_ms,rms_latency_ms,rms_kernel_count,includes_moe,vllm_config_hash
vLLM,0.20.1,gpu,model,1,1,1,,bf16,bf16,bf16,bf16,gen,1,1,4096,dense,0,1,4,1.0,0,0,false,hash
""",
    )
    _write(
        fpm,
        """
phase,counter_id,worker_id,dp_rank,ctx_tokens,ctx_requests,ctx_kv_tokens,decode_tokens,decode_requests,decode_kv_tokens,mean_decode_kv_tokens,queued_ctx_tokens,queued_ctx_requests,queued_decode_requests,queued_decode_kv_tokens,latency_ms
decode,1,w,0,0,0,0,1,1,4096,4096.000,0,0,0,0,40.0
decode,2,w,0,0,0,0,1,1,4097,4097.000,0,0,0,0,4.0
""",
    )

    rows = compare_layerwise_to_fpm(
        layerwise,
        fpm,
        aggregation="mean",
        decode_match="nearest",
        max_decode_kv_distance=4.0,
        decode_pool_forward_window=6.0,
        decode_kv_offset=1.0,
    )

    assert len(rows) == 1
    assert rows[0]["fpm_decode_kv"] == "4097.000"
    assert rows[0]["fpm_match"] == "exact"
    assert rows[0]["fpm_ms"] == pytest.approx(4.0)


def test_decode_comparison_can_pool_forward_kv_window(tmp_path) -> None:
    layerwise = tmp_path / "layerwise.csv"
    fpm = tmp_path / "fpm.csv"
    _write(
        layerwise,
        """
framework,framework_version,system,model,attn_tp,moe_tp,ep,num_slots,gemm_quant,moe_quant,attn_quant,kv_quant,phase,batch_size,new_tokens,past_kv,layer_type,layer_index,measured_layer_count,layer_multiplier,latency_ms,rms_latency_ms,rms_kernel_count,includes_moe,vllm_config_hash
vLLM,0.20.1,gpu,model,1,1,1,,bf16,bf16,bf16,bf16,gen,4,1,4096,dense,0,4,4,12.0,0,0,false,hash
""",
    )
    _write(
        fpm,
        """
phase,counter_id,worker_id,dp_rank,ctx_tokens,ctx_requests,ctx_kv_tokens,decode_tokens,decode_requests,decode_kv_tokens,mean_decode_kv_tokens,queued_ctx_tokens,queued_ctx_requests,queued_decode_requests,queued_decode_kv_tokens,latency_ms
decode,1,w,0,0,0,0,4,4,16388,4097.000,0,0,0,0,10.0
decode,2,w,0,0,0,0,4,4,16392,4098.000,0,0,0,0,14.0
decode,3,w,0,0,0,0,4,4,16420,4105.000,0,0,0,0,100.0
""",
    )

    rows = compare_layerwise_to_fpm(
        layerwise,
        fpm,
        aggregation="mean",
        decode_match="pooled",
        max_decode_kv_distance=4.0,
        decode_pool_forward_window=4.0,
    )

    assert len(rows) == 1
    assert rows[0]["fpm_decode_kv"] == "4097.000..4098.000"
    assert rows[0]["fpm_match"] == "pooled"
    assert rows[0]["fpm_samples"] == 2
    assert rows[0]["fpm_ms"] == pytest.approx(12.0)
    assert rows[0]["error_pct"] == pytest.approx(0.0)


def test_aic_decode_pooled_match_returns_representative_kv() -> None:
    decode = {
        (4, 4097.0): [10.0],
        (4, 4098.0): [14.0, 16.0],
        (4, 4105.0): [100.0],
    }

    matched = _match_decode(
        decode,
        batch_size=4,
        past_kv=4096,
        mode="pooled",
        max_distance=4.0,
        pool_forward_window=4.0,
    )

    assert matched is not None
    kv_label, samples, match, representative_kv = matched
    assert kv_label == "4097.000..4098.000"
    assert samples == [10.0, 14.0, 16.0]
    assert match == "pooled"
    assert representative_kv == 4098


def test_aic_decode_query_snaps_to_nearest_collected_layerwise_kv() -> None:
    layerwise_data = {
        "qwen/test": {
            "GEN": {
                1: {
                    1: {4096: {"latency": 1.0}},
                    4: {4096: {"latency": 1.5}},
                }
            }
        }
    }

    assert (
        _nearest_available_generation_kv(
            layerwise_data,
            model="Qwen/Test",
            tp_size=1,
            requested_kv=4098,
            max_distance=6.0,
        )
        == 4096
    )
    assert (
        _nearest_available_generation_kv(
            layerwise_data,
            model="Qwen/Test",
            tp_size=1,
            requested_kv=8192,
            max_distance=6.0,
        )
        is None
    )


def test_comparison_keeps_multi_model_rows_separate(tmp_path) -> None:
    layerwise = tmp_path / "layerwise.csv"
    fpm = tmp_path / "fpm.csv"
    _write(
        layerwise,
        """
framework,framework_version,system,model,attn_tp,moe_tp,ep,num_slots,gemm_quant,moe_quant,attn_quant,kv_quant,phase,batch_size,new_tokens,past_kv,layer_type,layer_index,measured_layer_count,layer_multiplier,latency_ms,rms_latency_ms,rms_kernel_count,includes_moe,vllm_config_hash
vLLM,0.20.1,gpu,model-a,1,1,1,,bf16,bf16,bf16,bf16,ctx,1,128,0,dense,0,4,4,10.0,0,0,false,hash
vLLM,0.20.1,gpu,model-b,1,1,1,,bf16,bf16,bf16,bf16,ctx,1,128,0,dense,0,4,4,30.0,0,0,false,hash
""",
    )
    _write(
        fpm,
        """
phase,counter_id,worker_id,dp_rank,ctx_tokens,ctx_requests,ctx_kv_tokens,decode_tokens,decode_requests,decode_kv_tokens,mean_decode_kv_tokens,queued_ctx_tokens,queued_ctx_requests,queued_decode_requests,queued_decode_kv_tokens,latency_ms
context,1,w,0,128,1,0,0,0,0,0.000,0,0,0,0,10.0
""",
    )

    rows = compare_layerwise_to_fpm(
        layerwise,
        fpm,
        aggregation="mean",
        decode_match="nearest",
        max_decode_kv_distance=4.0,
        decode_pool_forward_window=6.0,
    )
    filtered = compare_layerwise_to_fpm(
        layerwise,
        fpm,
        aggregation="mean",
        decode_match="nearest",
        max_decode_kv_distance=4.0,
        decode_pool_forward_window=6.0,
        model_filter="model-a",
    )

    assert [row["model"] for row in rows] == ["model-a", "model-b"]
    assert [row["layerwise_ms"] for row in rows] == [10.0, 30.0]
    assert len(filtered) == 1
    assert filtered[0]["model"] == "model-a"
