# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

pytestmark = pytest.mark.unit

BACKEND_FACTS = Path(__file__).resolve().parents[3] / "tools" / "perf_database" / "backend_facts.py"


@pytest.fixture
def backend_facts_module():
    spec = importlib.util.spec_from_file_location("backend_facts", BACKEND_FACTS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def data_tree(tmp_path):
    """Tiny data tree: one parquet table (two dtype slices, one multi-backend)
    and one legacy txt table with only the placeholder 'default' label."""
    sglang_dir = tmp_path / "h200_sxm" / "sglang" / "0.5.12"
    sglang_dir.mkdir(parents=True)
    rows = [
        # bf16 slice: single named backend
        {
            "framework": "sglang",
            "version": "0.5.12",
            "device": "h200",
            "op_name": "context_attention",
            "kernel_source": "flash_attention",
            "attn_dtype": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "batch_size": 1,
            "isl": 512,
            "latency": 1.0,
        },
        {
            "framework": "sglang",
            "version": "0.5.12",
            "device": "h200",
            "op_name": "context_attention",
            "kernel_source": "flash_attention",
            "attn_dtype": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "batch_size": 2,
            "isl": 512,
            "latency": 2.0,
        },
        # fp8 kv slice: two backends -> multi
        {
            "framework": "sglang",
            "version": "0.5.12",
            "device": "h200",
            "op_name": "context_attention",
            "kernel_source": "flash_attention",
            "attn_dtype": "bfloat16",
            "kv_cache_dtype": "fp8",
            "batch_size": 1,
            "isl": 512,
            "latency": 1.0,
        },
        {
            "framework": "sglang",
            "version": "0.5.12",
            "device": "h200",
            "op_name": "context_attention",
            "kernel_source": "triton",
            "attn_dtype": "bfloat16",
            "kv_cache_dtype": "fp8",
            "batch_size": 1,
            "isl": 8192,
            "latency": 9.0,
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), sglang_dir / "context_attention_perf.parquet")

    trtllm_dir = tmp_path / "h200_sxm" / "trtllm" / "1.0.0rc3"
    trtllm_dir.mkdir(parents=True)
    (trtllm_dir / "context_mla_perf.txt").write_text(
        "framework,version,device,op_name,kernel_source,mla_dtype,kv_cache_dtype,num_heads,latency\n"
        "trtllm,1.0.0rc3,h200,mla_context,default,float16,float16,128,1.0\n"
        "trtllm,1.0.0rc3,h200,mla_context,default,float16,float16,64,2.0\n"
    )
    # Marker file without kernel_source must be skipped, not crash the scan.
    (trtllm_dir / "SHARED_LAYER_REUSE.txt").write_text("see 1.0.0rc2\n")
    return tmp_path


def _facts_for(by_op, op_file):
    return by_op[op_file]


def test_scan_groups_by_precision_slice_and_classifies_status(backend_facts_module, data_tree):
    facts, axes_by_op = backend_facts_module.scan(data_tree)
    by_op = backend_facts_module._fact_entries(facts, axes_by_op)

    attn = _facts_for(by_op, "context_attention_perf")
    assert axes_by_op["context_attention_perf"] == ["op_name", "attn_dtype", "kv_cache_dtype"]
    assert len(attn) == 2
    bf16 = next(e for e in attn if e["kv_cache_dtype"] == "bfloat16")
    assert bf16["kernel_sources"] == {"flash_attention": 2}
    assert bf16["status"] == "single"
    fp8 = next(e for e in attn if e["kv_cache_dtype"] == "fp8")
    assert fp8["kernel_sources"] == {"flash_attention": 1, "triton": 1}
    assert fp8["status"] == "multi"

    mla = _facts_for(by_op, "context_mla_perf")
    assert len(mla) == 1
    assert mla[0]["framework"] == "trtllm"
    assert mla[0]["version"] == "1.0.0rc3"
    assert mla[0]["system"] == "h200_sxm"
    assert mla[0]["kernel_sources"] == {"default": 2}
    assert mla[0]["status"] == "default_only"


def test_yaml_output_is_valid_and_round_trips(backend_facts_module, data_tree):
    facts, axes_by_op = backend_facts_module.scan(data_tree)
    by_op = backend_facts_module._fact_entries(facts, axes_by_op)
    doc = yaml.safe_load(backend_facts_module.render_yaml(by_op, axes_by_op))

    assert doc["schema_version"] == 1
    ops = {o["op_file"]: o for o in doc["ops"]}
    assert set(ops) == {"context_attention_perf", "context_mla_perf"}
    fp8 = next(f for f in ops["context_attention_perf"]["facts"] if f["kv_cache_dtype"] == "fp8")
    assert fp8["kernel_sources"] == {"flash_attention": 1, "triton": 1}
    assert fp8["status"] == "multi"


def test_markdown_output_lists_every_slice(backend_facts_module, data_tree):
    facts, axes_by_op = backend_facts_module.scan(data_tree)
    by_op = backend_facts_module._fact_entries(facts, axes_by_op)
    md = backend_facts_module.render_markdown(by_op, axes_by_op)

    assert "## `context_attention_perf`" in md
    assert "`flash_attention` (1), `triton` (1)" in md
    assert "default_only" in md
    assert "Fact slices: **3**" in md


def test_classify_status(backend_facts_module):
    assert backend_facts_module.classify_status({"fa3": 10}) == "single"
    assert backend_facts_module.classify_status({"fa3": 10, "default": 1}) == "multi"
    assert backend_facts_module.classify_status({"default": 5}) == "default_only"
