# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end tests verifying that cli_estimate().get() produces the same
metrics as the corresponding rows in cli_default() best_config_topn output.

For each row in the best_configs DataFrame (agg and disagg modes), the test
calls cli_estimate with the same parameters and asserts that:
  1. EstimateResult.get() returns data matching the CSV row.
  2. EstimateResult property accessors are consistent with get().

Parametrized across dense and MoE models to cover both code paths.
"""

import math

import pandas as pd
import pytest

from aiconfigurator.cli.api import cli_default, cli_estimate

pytestmark = pytest.mark.e2e

RTOL = 1e-3  # relative tolerance
# The backend rounds the DataFrame to 3 decimal places (.round(3)) before
# writing the CSV, but result_dict (used by .get()) stores unrounded values.
# Allow up to 0.5e-3 absolute tolerance to cover the rounding gap.
ATOL = 5e-4  # absolute tolerance

CONFIGS = [
    {"model": "QWEN/QWEN3-32B", "system": "h100_sxm", "total_gpus": 8},
    {"model": "QWEN/QWEN3-235B-A22B", "system": "h100_sxm", "total_gpus": 32},
]


def _close(a: float, b: float) -> bool:
    if a == b:
        return True
    return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)


def _config_id(cfg: dict) -> str:
    return f"{cfg['model'].split('/')[-1]}_{cfg['system']}"


@pytest.fixture(scope="module", params=CONFIGS, ids=[_config_id(c) for c in CONFIGS])
def default_result(request):
    cfg = request.param
    result = cli_default(
        model_path=cfg["model"],
        total_gpus=cfg["total_gpus"],
        system=cfg["system"],
    )
    result._test_cfg = cfg
    return result


def _assert_get_matches_csv(est, got, best_df, row, idx):
    """Assert .get() column set and values match the CSV row."""
    csv_cols = set(best_df.columns)
    got_cols = set(got.keys())

    assert got_cols == csv_cols, (
        f"Row {idx}: column set mismatch. "
        f"Extra in get(): {got_cols - csv_cols}, "
        f"Missing from get(): {csv_cols - got_cols}"
    )

    for col in best_df.columns:
        if col not in got:
            continue
        if pd.api.types.is_numeric_dtype(best_df[col]):
            csv_val = float(row[col])
            got_val = float(got[col])
            assert _close(csv_val, got_val), (
                f"Row {idx}: column '{col}' mismatch: csv={csv_val}, estimate.get()={got_val}"
            )
        else:
            assert str(got[col]) == str(row[col]), (
                f"Row {idx}: column '{col}' mismatch: csv={row[col]!r}, estimate.get()={got[col]!r}"
            )


def _assert_properties_consistent(est, got):
    """Assert @property accessors and dataclass fields match .get()."""
    from dataclasses import fields as dc_fields

    est_type = type(est)
    property_names = [name for name in dir(est_type) if isinstance(getattr(est_type, name, None), property)]
    assert len(property_names) > 0, "EstimateResult has no @property accessors"

    for prop_name in property_names:
        prop_val = getattr(est, prop_name)
        if prop_name not in got:
            continue
        assert prop_val == got[prop_name], f"Property '{prop_name}': attribute={prop_val}, get()={got[prop_name]}"

    skip_fields = {"raw", "mode", "per_ops_data"}
    for f in dc_fields(est):
        if f.name in skip_fields or f.name not in got:
            continue
        field_val = getattr(est, f.name)
        assert field_val == got[f.name], f"Field '{f.name}': attribute={field_val}, get()={got[f.name]}"


def test_agg_estimate_matches_csv(default_result):
    """Each agg best_config row should match a cli_estimate .get() call.

    Also verifies that @property accessors and dataclass fields are
    consistent with .get() on the first row.
    """
    cfg = default_result._test_cfg
    best_df = default_result.best_configs.get("agg")
    assert best_df is not None and not best_df.empty, "No agg best_configs found"

    for idx, row in best_df.iterrows():
        est = cli_estimate(
            model_path=cfg["model"],
            system_name=cfg["system"],
            backend_name="trtllm",
            mode="agg",
            tp_size=int(row["tp"]),
            pp_size=int(row["pp"]),
            attention_dp_size=int(row["dp"]),
            moe_tp_size=int(row["moe_tp"]),
            moe_ep_size=int(row["moe_ep"]),
            isl=int(row["isl"]),
            osl=int(row["osl"]),
            batch_size=int(row["bs"]),
            ctx_tokens=int(row["ctx_tokens"]),
            gemm_quant_mode=str(row["gemm"]),
            kvcache_quant_mode=str(row["kvcache"]),
            fmha_quant_mode=str(row["fmha"]),
            moe_quant_mode=str(row["moe"]),
            comm_quant_mode=str(row["comm"]),
        )

        got = est.get()
        _assert_get_matches_csv(est, got, best_df, row, idx)

        if idx == best_df.index[0]:
            _assert_properties_consistent(est, got)


def test_disagg_estimate_matches_csv(default_result):
    """Each disagg best_config row should match a cli_estimate .get() call.

    Same as agg: column set and all values (including TTFT-corrected throughput
    metrics) must match.  Also verifies @property and dataclass consistency
    on the first row.
    """
    cfg = default_result._test_cfg
    best_df = default_result.best_configs.get("disagg")
    if best_df is None or best_df.empty:
        pytest.skip("No disagg best_configs produced for this model/system")

    for idx, row in best_df.iterrows():
        est = cli_estimate(
            model_path=cfg["model"],
            system_name=cfg["system"],
            backend_name="trtllm",
            mode="disagg",
            isl=int(row["isl"]),
            osl=int(row["osl"]),
            prefill_tp_size=int(row["(p)tp"]),
            prefill_pp_size=int(row["(p)pp"]),
            prefill_attention_dp_size=int(row["(p)dp"]),
            prefill_moe_tp_size=int(row["(p)moe_tp"]),
            prefill_moe_ep_size=int(row["(p)moe_ep"]),
            prefill_batch_size=int(row["(p)bs"]),
            prefill_num_workers=int(row["(p)workers"]),
            decode_tp_size=int(row["(d)tp"]),
            decode_pp_size=int(row["(d)pp"]),
            decode_attention_dp_size=int(row["(d)dp"]),
            decode_moe_tp_size=int(row["(d)moe_tp"]),
            decode_moe_ep_size=int(row["(d)moe_ep"]),
            decode_batch_size=int(row["(d)bs"]),
            decode_num_workers=int(row["(d)workers"]),
            gemm_quant_mode=str(row["(p)gemm"]),
            kvcache_quant_mode=str(row["(p)kvcache"]),
            fmha_quant_mode=str(row["(p)fmha"]),
            moe_quant_mode=str(row["(p)moe"]),
            comm_quant_mode=str(row["(p)comm"]),
        )

        got = est.get()
        _assert_get_matches_csv(est, got, best_df, row, idx)

        if idx == best_df.index[0]:
            _assert_properties_consistent(est, got)
