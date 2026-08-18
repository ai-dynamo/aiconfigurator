# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 module tests: data loading, op weight accounting, and model
memory estimation.

The attention/mHC SOL formulas and the silicon interpolation ladder that used
to be tested here (``_deepseek_v4_attention_sol``, kNN past-frontier holds,
prefix-resolved table reads, rank-local head-bucket resolution) retired to the
compiled engine with #1357 PR-5; that behaviour is anchored by
the frozen parity
goldens.
"""

import pytest

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk import operations as ops
from aiconfigurator.sdk.backends.sglang_backend import SGLANGBackend
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import PerfDatabase

pytestmark = pytest.mark.unit


def _mhc_view_db(tmp_path, rows: list[dict] | None):
    """Minimal systems tree serving the ``_mhc_module_data`` engine view
    (the Python mhc parser retired with the deprecation-cleanup PR)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml

    root = tmp_path / "systems"
    root.mkdir(exist_ok=True)
    (root / "h100_sxm.yaml").write_text(
        yaml.safe_dump(
            {
                "data_dir": "data/h100_sxm",
                "gpu": {
                    "sm_version": 90,
                    "mem_bw": 4_800_000_000_000.0,
                    "mem_bw_empirical_scaling_factor": 0.8,
                    "mem_empirical_constant_latency": 0.000003,
                    "bfloat16_tc_flops": 989_000_000_000_000.0,
                    "fp8_tc_flops": 1_978_000_000_000_000.0,
                },
                "node": {
                    "num_gpus_per_node": 8,
                    "inter_node_bw": 50_000_000_000.0,
                    "intra_node_bw": 450_000_000_000.0,
                    "p2p_latency": 0.00001,
                },
                "misc": {"nccl_version": "2.26.2"},
            }
        ),
        encoding="utf-8",
    )
    if rows is not None:
        path = root / "data/h100_sxm/sparse_attention/vllm/1.0.0/mhc_module_perf.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({k: [r[k] for r in rows] for k in rows[0]}), path)
    return PerfDatabase("h100_sxm", "vllm", "1.0.0", str(root), database_mode="HYBRID")


def _mhc_row(hidden_size: int, latency: float) -> dict:
    return {
        "framework": "VLLM",
        "version": "test",
        "device": "H20",
        "op_name": "pre",
        "kernel_source": "mhc",
        "architecture": "DeepseekV4ForCausalLM",
        "num_tokens": 512,
        "hc_mult": 4,
        "hidden_size": hidden_size,
        "latency": latency,
    }


def test_mhc_module_view_returns_none_for_missing_file(tmp_path):
    from aiconfigurator_core.sdk.engine_table_view import fetch_table_view

    assert fetch_table_view(_mhc_view_db(tmp_path, None), "_mhc_module_data") is None


def test_mhc_view_keys_by_op_hc_mult_hidden_size_num_tokens(tmp_path):
    from aiconfigurator_core.sdk.engine_table_view import fetch_table_view

    db = _mhc_view_db(tmp_path, [_mhc_row(4096, 1.5), _mhc_row(7168, 2.5)])
    data = fetch_table_view(db, "_mhc_module_data")

    # data[op][hc_mult][hidden_size][num_tokens] — hidden_size distinguishes rows.
    assert set(data.keys()) == {"pre"}
    assert set(data["pre"][4].keys()) == {4096, 7168}
    assert data["pre"][4][4096][512]["latency"] == pytest.approx(1.5)
    assert data["pre"][4][7168][512]["latency"] == pytest.approx(2.5)
    assert data["pre"][4][7168][512]["power"] == pytest.approx(0.0)


def test_mhc_weight_memory_uses_quant_mode():
    bf16_op = ops.DeepSeekV4MHCModule(
        "mhc",
        1,
        "pre",
        7168,
        4,
        20,
        common.GEMMQuantMode.bfloat16,
        architecture="DeepseekV4ProForCausalLM",
    )
    fp8_op = ops.DeepSeekV4MHCModule(
        "mhc",
        1,
        "pre",
        7168,
        4,
        20,
        common.GEMMQuantMode.fp8_block,
        architecture="DeepseekV4ProForCausalLM",
    )
    assert fp8_op.get_weights() == pytest.approx(bf16_op.get_weights() / 2)


def test_deepseek_v4_per_op_sol_queries_run_end_to_end():
    """Every DSV4 op must answer a per-op SOL query through the per-call
    ``op._engine_query()`` surface (the PERMANENT internal single-op plumbing,
    routed through the compiled engine's model-less probe — not a deprecation
    shim). The probe engine loads perf tables from disk,
    so this runs on a real shipped database rather than the synthetic-stuffed
    fixture (whose in-memory tables the engine cannot see)."""
    from aiconfigurator.sdk.perf_database import get_database_view

    # b200_sxm/sglang/0.5.14 ships the full DSV4 table set (csa modules, mhc),
    # which the probe engine loads eagerly per op family even under SOL.
    db = get_database_view("b200_sxm", "sglang", "0.5.14", database_mode="SOL")
    model_config = config.ModelConfig(
        tp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        nextn=1,
        overwrite_num_layers=2,
    )
    model = get_model("sgl-project/DeepSeek-V4-Flash-FP8", model_config, backend_name="sglang")

    context_total = sum(
        float(op._engine_query(db, x=128, batch_size=1, beam_width=1, s=128, prefix=0)) for op in model.context_ops
    )
    generation_total = sum(
        float(op._engine_query(db, x=2, batch_size=2, beam_width=1, s=129)) for op in model.generation_ops
    )
    assert context_total > 0
    assert generation_total > 0


def test_sglang_deepseek_v4_pro_moe_workspace_uses_residual_hidden_size(mutable_comprehensive_perf_db):
    db = mutable_comprehensive_perf_db
    db.system_spec["gpu"]["mem_capacity"] = 198674743296  # GB200 189471 MiB
    db.system_spec["misc"]["nccl_mem"] = {1: 0, 2: 358612992, 4: 411041792, 8: 411041792}
    db.system_spec["misc"]["other_mem"] = 3758096384

    model_config = config.ModelConfig(
        tp_size=1,
        pp_size=1,
        attention_dp_size=8,
        moe_tp_size=1,
        moe_ep_size=8,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        moe_quant_mode=common.MoEQuantMode.w4a8_mxfp4_mxfp8,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        comm_quant_mode=common.CommQuantMode.half,
        moe_backend="megamoe",
        nextn=0,
    )
    model = get_model("deepseek-ai/DeepSeek-V4-Pro", model_config, backend_name="sglang")

    memory = SGLANGBackend()._get_memory_usage(
        model,
        db,
        batch_size=1,
        beam_width=1,
        isl=8192,
        osl=1024,
    )

    num_tokens = 8192
    attention_width = model._num_heads * model._head_size
    residual_width = model._hidden_size
    assert model.activation_hidden_size == residual_width
    assert attention_width > residual_width

    tp_activation_factor = 28
    attention_workspace = 2 * num_tokens * attention_width * tp_activation_factor
    moe_scale_workspace = (
        num_tokens
        * residual_width
        * model.config.attention_dp_size
        * model._num_experts
        * model._topk
        / model.config.moe_ep_size
        / 128
        * 4
    )
    expected_activation_gib = (attention_workspace + moe_scale_workspace) * 1.15 / (1 << 30)

    assert memory["activations"] == pytest.approx(expected_activation_gib)

    old_moe_scale_workspace = (
        num_tokens
        * attention_width
        * model.config.attention_dp_size
        * model._num_experts
        * model._topk
        / model.config.moe_ep_size
        / 128
        * 4
    )
    old_activation_gib = (attention_workspace + old_moe_scale_workspace) * 1.15 / (1 << 30)
    assert memory["activations"] < old_activation_gib
