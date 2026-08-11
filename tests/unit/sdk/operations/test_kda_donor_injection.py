# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-backend donors must not defeat the KDA per-key fused routing.

The sglang K3 fused-decode routing selects ``kda_fused_decode`` for a shard
exactly when that shard has NO Triton-pair generation key — "The SDK's
per-key fused routing requires the Triton key to be ABSENT for the covered
shard" (b200_sxm kda sglang collection_meta.yaml). vLLM genuinely collects
rows under the same physical kernel names, so a manifest that lists those
kernel_sources as cross-backend-inheritable lets the shared layer fill the
deliberately-absent sglang keys and reroute decode onto the wrong kernel
(measured: b300 K3 agg spec bs1 tpot 3.296 -> 3.336 ms, review 2026-08-04).

This test builds the manifest THROUGH the generator (render_manifest), so it
fails if the ABSENCE_LOAD_BEARING exclusion in
tools/perf_database/generate_perf_data_reuse_manifest.py is ever removed: the entries
would come back as multi-framework `shared`, the vllm donor below would fill
the 12-head Triton generation key, and the assertion on the fused latency
would break. Rust needs no twin — it consumes Python's serialized source
chain (engine.py `_compute_perf_db_sources`).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from aiconfigurator.sdk.perf_database import (
    PerfDatabase,
    _load_perf_data_reuse_manifest_entries,
)
from aiconfigurator_core.sdk.operations.mamba import KDAKernel

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]

_KDA_HEADER = (
    "framework,version,device,op_name,kernel_source,phase,batch_size,seq_len,"
    "num_tokens,d_model,d_conv,num_k_heads,head_k_dim,num_v_heads,head_v_dim,model_name,latency\n"
)
# The K3 TP8 12-head shard (d_model, heads, dims match the shipped datasets).
_SHARD = "7168,4,12,128,12,128"
_FUSED_LATENCY = 0.006
_DONOR_LATENCY = 0.015


def _generator():
    name = "generate_perf_data_reuse_manifest_under_test"
    spec = importlib.util.spec_from_file_location(
        name, _REPO_ROOT / "tools" / "perf_database" / "generate_perf_data_reuse_manifest.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Registered so the module's @dataclass processing can resolve itself.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _kda_rows(framework: str, kernel_source: str, latency: float) -> list[str]:
    return [
        f"{framework},1.0,h100,kda,{kernel_source},generation,{batch},0,{batch},{_SHARD},kimi,{latency}\n"
        for batch in (1, 8, 64)
    ]


def _write_kda_csv(path: Path, chunks: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(_KDA_HEADER)
        for chunk in chunks:
            f.writelines(chunk)


def _summary(op_file: str, kernel_source: str, tier: str, frameworks: list[str]) -> dict:
    return {
        "op_file": op_file,
        "kernel_source": kernel_source,
        "tier": tier,
        "system": "h100_sxm",
        "frameworks": set(frameworks),
        "total_raw_rows": 3,
        "within_framework_dedup_rows": 0,
        "median_pct_divergence": None,
    }


def test_vllm_donor_cannot_defeat_the_fused_decode_reroute(tmp_path, monkeypatch):
    gen = _generator()

    systems_root = tmp_path / "systems"
    systems_root.mkdir()
    # Minimal system yaml + required nccl stub (same shape as the shared-layer tests).
    (systems_root / "h100_sxm.yaml").write_text(
        "data_dir: data/h100_sxm\n"
        "gpu:\n"
        "  mem_bw: 4800000000000\n"
        "  mem_bw_empirical_scaling_factor: 0.8\n"
        "  mem_empirical_constant_latency: 0.000003\n"
        "  mem_capacity: 151397597184\n"
        "  bfloat16_tc_flops: 989000000000000\n"
        "  int8_tc_flops: 1978000000000000\n"
        "  fp8_tc_flops: 1978000000000000\n"
        "  power: 700\n"
        "  sm_version: 90\n"
        "node:\n"
        "  num_gpus_per_node: 8\n"
        "  inter_node_bw: 50000000000\n"
        "  intra_node_bw: 450000000000\n"
        "  pcie_bw: 64000000000\n"
        "  p2p_latency: 0.00001\n"
        "misc:\n"
        "  nccl_mem: {1: 0, 2: 358612992, 4: 411041792, 8: 411041792}\n"
        "  other_mem: 3758096384\n"
        "  nccl_version: '2.26.2'\n"
    )
    nccl_dir = systems_root / "data" / "h100_sxm" / "nccl" / "2.26.2"
    nccl_dir.mkdir(parents=True)
    (nccl_dir / "nccl_perf.txt").write_text(
        "framework,version,device,op_name,kernel_source,nccl_dtype,num_gpus,message_size,latency\n"
    )

    # sglang owns ONLY the fused generation row for the 12-head shard — the
    # Triton generation key is deliberately absent (fused dispatch truth).
    _write_kda_csv(
        systems_root / "data" / "h100_sxm" / "sglang" / "1.0" / "kda_perf.txt",
        [_kda_rows("SGLang", "kda_fused_decode", _FUSED_LATENCY)],
    )
    # The vllm sibling carries genuine Triton-pair rows for the SAME shard.
    _write_kda_csv(
        systems_root / "data" / "h100_sxm" / "vllm" / "1.0" / "kda_perf.txt",
        [
            _kda_rows("VLLM", "causal_conv1d_update", _DONOR_LATENCY),
            _kda_rows("VLLM", "fused_recurrent_kda_packed_decode", _DONOR_LATENCY),
        ],
    )

    # Manifest built THROUGH the generator: the exclusion must demote the
    # absence-load-bearing pairs to a loader-inert tier even though both
    # backends genuinely produce rows under those kernel_sources.
    summaries = [
        _summary("kda_perf.parquet", "kda_fused_decode", "shared", ["sglang"]),
        _summary("kda_perf.parquet", "causal_conv1d_update", "shared", ["sglang", "vllm"]),
        _summary("kda_perf.parquet", "fused_recurrent_kda_packed_decode", "shared", ["sglang", "vllm"]),
    ]
    (systems_root / "perf_data_reuse_manifest.yaml").write_text(gen.render_manifest(summaries))
    _load_perf_data_reuse_manifest_entries.cache_clear()
    KDAKernel.clear_cache()

    db = PerfDatabase("h100_sxm", "sglang", "1.0", str(systems_root), database_mode="SILICON")

    def _query(kernel_source):
        return KDAKernel._query_kda_table(
            db,
            phase="generation",
            kernel_source=kernel_source,
            batch_size=8,
            seq_len=None,
            d_model=7168,
            num_k_heads=12,
            head_k_dim=128,
            num_v_heads=12,
            head_v_dim=128,
            d_conv=4,
        )

    recurrence = _query("fused_recurrent_kda_packed_decode")
    assert float(recurrence) == pytest.approx(_FUSED_LATENCY), (
        "vllm donor rows filled the deliberately-absent sglang Triton key and "
        "defeated the fused-decode reroute — the ABSENCE_LOAD_BEARING manifest "
        "exclusion is not holding"
    )
    assert recurrence.source == "silicon"
    conv = _query("causal_conv1d_update")
    assert float(conv) == 0.0  # folded into the fused row, not donor-priced

    # And the exclusion is visible in the generated manifest itself.
    manifest_text = (systems_root / "perf_data_reuse_manifest.yaml").read_text()
    assert manifest_text.count("tier: absence_load_bearing") == 2
