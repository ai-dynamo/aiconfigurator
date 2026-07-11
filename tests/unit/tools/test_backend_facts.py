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
BACKEND_MAP = Path(__file__).resolve().parents[3] / "collector" / "kernel_source_backends.yaml"

# Synthetic translation table exercising exact, match-conditioned, and absent entries.
MAPPINGS = [
    {"framework": "sglang", "kernel_source": "flash_attention", "backend": "fa3"},
    {"framework": "sglang", "kernel_source": "triton", "backend": "triton"},
    {
        "framework": "trtllm",
        "kernel_source": "default",
        "match": {"op_file": "context_mla_perf", "mla_dtype": "float16"},
        "backend": "trtllm_internal",
    },
]


@pytest.fixture
def backend_facts_module():
    spec = importlib.util.spec_from_file_location("backend_facts", BACKEND_FACTS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _attn_row(kernel_source, kv_cache_dtype, batch_size):
    return {
        "framework": "sglang",
        "version": "0.5.12",
        "device": "h200",
        "op_name": "context_attention",
        "kernel_source": kernel_source,
        "attn_dtype": "bfloat16",
        "kv_cache_dtype": kv_cache_dtype,
        "batch_size": batch_size,
        "isl": 512,
        "latency": 1.0,
    }


@pytest.fixture
def data_tree(tmp_path):
    """Tiny data tree: one parquet table (two dtype slices, one multi-backend)
    and one legacy txt table with only the placeholder 'default' label."""
    sglang_dir = tmp_path / "h200_sxm" / "sglang" / "0.5.12"
    sglang_dir.mkdir(parents=True)
    rows = [
        # bf16 slice: single named backend
        _attn_row("flash_attention", "bfloat16", 1),
        _attn_row("flash_attention", "bfloat16", 2),
        # fp8 kv slice: two backends
        _attn_row("flash_attention", "fp8", 1),
        _attn_row("triton", "fp8", 2),
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


def test_scan_groups_by_precision_slice(backend_facts_module, data_tree):
    facts, axes_by_op = backend_facts_module.scan(data_tree)
    by_op = backend_facts_module._fact_entries(facts, axes_by_op, MAPPINGS)

    attn = by_op["context_attention_perf"]
    assert axes_by_op["context_attention_perf"] == ["op_name", "attn_dtype", "kv_cache_dtype"]
    assert len(attn) == 2
    bf16 = next(e for e in attn if e["kv_cache_dtype"] == "bfloat16")
    assert bf16["kernel_sources"] == ["flash_attention"]
    assert bf16["backends"] == ["fa3"]
    fp8 = next(e for e in attn if e["kv_cache_dtype"] == "fp8")
    assert fp8["kernel_sources"] == ["flash_attention", "triton"]
    assert fp8["backends"] == ["fa3", "triton"]

    mla = by_op["context_mla_perf"]
    assert len(mla) == 1
    assert mla[0]["framework"] == "trtllm"
    assert mla[0]["version"] == "1.0.0rc3"
    assert mla[0]["system"] == "h200_sxm"
    assert mla[0]["kernel_sources"] == ["default"]
    assert mla[0]["backends"] == ["trtllm_internal"]  # via the match-conditioned entry


def test_translate_match_and_unmapped(backend_facts_module):
    translate = backend_facts_module.translate
    # match-conditioned entry applies only when every match key equals the slice value
    assert translate(MAPPINGS, "trtllm", "context_mla_perf", {"mla_dtype": "float16"}, "default") == "trtllm_internal"
    assert translate(MAPPINGS, "trtllm", "context_mla_perf", {"mla_dtype": "bfloat16"}, "default") is None
    assert translate(MAPPINGS, "trtllm", "generation_mla_perf", {"mla_dtype": "float16"}, "default") is None
    # unmapped label -> None, and _fact_entries records it as unverified
    assert translate(MAPPINGS, "sglang", "gemm_perf", {}, "mystery_label") is None


def test_unmapped_labels_become_unverified_and_are_reported(backend_facts_module, data_tree):
    facts, axes_by_op = backend_facts_module.scan(data_tree)
    unmapped: set = set()
    by_op = backend_facts_module._fact_entries(facts, axes_by_op, MAPPINGS[:1], unmapped)
    fp8 = next(e for e in by_op["context_attention_perf"] if e["kv_cache_dtype"] == "fp8")
    assert fp8["backends"] == ["fa3", "unverified"]
    assert ("sglang", "triton") in unmapped and ("trtllm", "default") in unmapped


def test_committed_backend_map_covers_schema(backend_facts_module):
    mappings = backend_facts_module.load_backend_map(BACKEND_MAP)
    assert mappings, "translation table must not be empty"
    for m in mappings:
        assert set(m) <= {"framework", "kernel_source", "backend", "match", "source"}, m
        assert m["framework"] in {"sglang", "trtllm", "vllm"}, m
        assert m["kernel_source"] and m["backend"] and m["source"], m


def test_yaml_output_is_valid_and_round_trips(backend_facts_module, data_tree):
    facts, axes_by_op = backend_facts_module.scan(data_tree)
    by_op = backend_facts_module._fact_entries(facts, axes_by_op, MAPPINGS)
    doc = yaml.safe_load(backend_facts_module.render_yaml(by_op, axes_by_op))

    assert doc["schema_version"] == 1
    ops = {o["op_file"]: o for o in doc["ops"]}
    assert set(ops) == {"context_attention_perf", "context_mla_perf"}
    fp8 = next(f for f in ops["context_attention_perf"]["facts"] if f["kv_cache_dtype"] == "fp8")
    assert fp8["kernel_sources"] == ["flash_attention", "triton"]
    assert fp8["backends"] == ["fa3", "triton"]


def test_check_passes_when_registry_matches(backend_facts_module, data_tree, tmp_path):
    facts, axes_by_op = backend_facts_module.scan(data_tree)
    by_op = backend_facts_module._fact_entries(facts, axes_by_op, MAPPINGS)
    registry = tmp_path / "op_backend_facts.yaml"
    registry.write_text(backend_facts_module.render_yaml(by_op, axes_by_op))

    assert backend_facts_module.check(registry, by_op, axes_by_op) == []


def test_check_reports_drift(backend_facts_module, data_tree, tmp_path):
    facts, axes_by_op = backend_facts_module.scan(data_tree)
    by_op = backend_facts_module._fact_entries(facts, axes_by_op, MAPPINGS)
    registry = tmp_path / "op_backend_facts.yaml"
    registry.write_text(backend_facts_module.render_yaml(by_op, axes_by_op))

    # New data appears (kernel changed for the bf16 slice + a brand-new slice).
    sglang_dir = data_tree / "h200_sxm" / "sglang" / "0.5.14"
    sglang_dir.mkdir(parents=True)
    rows = [_attn_row("trtllm_mha", "bfloat16", 1)]
    rows[0]["version"] = "0.5.14"
    pq.write_table(pa.Table.from_pylist(rows), sglang_dir / "context_attention_perf.parquet")

    facts2, axes2 = backend_facts_module.scan(data_tree)
    by_op2 = backend_facts_module._fact_entries(facts2, axes2, MAPPINGS)
    drift = backend_facts_module.check(registry, by_op2, axes2)
    assert len(drift) == 1
    assert drift[0].startswith("data-not-in-registry:")
    assert "0.5.14" in drift[0] and "trtllm_mha" in drift[0]

    # And the reverse direction: registry entry with no backing data.
    registry.write_text(backend_facts_module.render_yaml(by_op2, axes2))
    drift = backend_facts_module.check(registry, by_op, axes_by_op)
    assert len(drift) == 1
    assert drift[0].startswith("registry-not-in-data:")


def test_check_reports_mismatched_kernel_sources(backend_facts_module, data_tree, tmp_path):
    facts, axes_by_op = backend_facts_module.scan(data_tree)
    by_op = backend_facts_module._fact_entries(facts, axes_by_op, MAPPINGS)
    text = backend_facts_module.render_yaml(by_op, axes_by_op)
    registry = tmp_path / "op_backend_facts.yaml"
    registry.write_text(text.replace('kernel_sources: ["default"]', 'kernel_sources: ["trtllm_mla"]'))

    drift = backend_facts_module.check(registry, by_op, axes_by_op)
    assert len(drift) == 1
    assert drift[0].startswith("mismatch:")
    assert "trtllm_mla" in drift[0] and "default" in drift[0]
