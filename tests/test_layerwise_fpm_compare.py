import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "collector" / "layerwise" / "diagnostics"))

from compare_layerwise_fpm import compare_layerwise_to_fpm


def _write(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n")


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
