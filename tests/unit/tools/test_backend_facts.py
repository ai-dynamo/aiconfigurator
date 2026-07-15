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
CATALOG = Path(__file__).resolve().parents[3] / "collector" / "op_backend_catalog.yaml"
FAMILIES = {"context_attention_perf": "attention", "context_mla_perf": "mla"}

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
    doc = yaml.safe_load(backend_facts_module.render_yaml(by_op, axes_by_op, FAMILIES))

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
    registry.write_text(backend_facts_module.render_yaml(by_op, axes_by_op, FAMILIES))

    assert backend_facts_module.check(registry, by_op, axes_by_op) == []


def test_check_reports_drift(backend_facts_module, data_tree, tmp_path):
    facts, axes_by_op = backend_facts_module.scan(data_tree)
    by_op = backend_facts_module._fact_entries(facts, axes_by_op, MAPPINGS)
    registry = tmp_path / "op_backend_facts.yaml"
    registry.write_text(backend_facts_module.render_yaml(by_op, axes_by_op, FAMILIES))

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
    registry.write_text(backend_facts_module.render_yaml(by_op2, axes2, FAMILIES))
    drift = backend_facts_module.check(registry, by_op, axes_by_op)
    assert len(drift) == 1
    assert drift[0].startswith("registry-not-in-data:")


def test_check_reports_mismatched_kernel_sources(backend_facts_module, data_tree, tmp_path):
    facts, axes_by_op = backend_facts_module.scan(data_tree)
    by_op = backend_facts_module._fact_entries(facts, axes_by_op, MAPPINGS)
    text = backend_facts_module.render_yaml(by_op, axes_by_op, FAMILIES)
    registry = tmp_path / "op_backend_facts.yaml"
    registry.write_text(text.replace('kernel_sources: ["default"]', 'kernel_sources: ["trtllm_mla"]'))

    drift = backend_facts_module.check(registry, by_op, axes_by_op)
    assert len(drift) == 1
    assert drift[0].startswith("mismatch:")
    assert "trtllm_mla" in drift[0] and "default" in drift[0]


def test_committed_catalog_covers_every_registry_op_file(backend_facts_module):
    catalog = backend_facts_module.load_catalog(CATALOG)
    families = backend_facts_module.catalog_families(catalog)
    registry = yaml.safe_load((Path(__file__).resolve().parents[3] / "collector" / "op_backend_facts.yaml").read_text())
    missing = [op["op_file"] for op in registry["ops"] if op["op_file"] not in families]
    assert missing == []
    assert all(op["family"] == families[op["op_file"]] for op in registry["ops"])


def test_catalog_inconsistencies(backend_facts_module, data_tree):
    facts, axes_by_op = backend_facts_module.scan(data_tree)
    by_op = backend_facts_module._fact_entries(facts, axes_by_op, MAPPINGS)
    catalog = {
        "families": [
            {
                "family": "attention",
                "op_files": ["context_attention_perf"],
                "frameworks": {"sglang": {"choices": [{"backend": "fa3"}]}},
            }
        ]
    }
    problems = backend_facts_module.catalog_inconsistencies(by_op, catalog)
    # context_mla_perf has no family; observed triton is not in the enumerated
    # (attention, sglang) choice space
    assert "op-file-without-family: context_mla_perf" in problems
    assert any(
        p.startswith("backend-not-in-catalog: family=attention framework=sglang") and "triton" in p for p in problems
    )

    catalog["families"][0]["frameworks"]["sglang"]["choices"].append({"backend": "triton"})
    catalog["families"].append({"family": "mla", "op_files": ["context_mla_perf"]})
    assert backend_facts_module.catalog_inconsistencies(by_op, catalog) == []


def test_catalog_backend_vocab_keyed_by_family_and_framework(backend_facts_module):
    catalog = {
        "families": [
            {
                "family": "attention",
                "op_files": ["context_attention_perf"],
                "frameworks": {
                    "sglang": {"choices": [{"backend": "fa3"}, {"backend": "triton"}]},
                    "vllm": {"choices": [{"backend": "flash_attn"}]},
                },
            },
            {"family": "mla", "op_files": ["context_mla_perf"]},  # no enumerated choice space
        ]
    }
    vocab = backend_facts_module.catalog_backend_vocab(catalog)
    assert vocab == {
        ("attention", "sglang"): {"fa3", "triton"},
        ("attention", "vllm"): {"flash_attn"},
    }
    assert ("mla", "sglang") not in vocab
    assert ("attention", "trtllm") not in vocab


def test_catalog_inconsistencies_does_not_merge_choice_spaces_across_frameworks(backend_facts_module):
    # fa3 is valid for sglang but not for vllm within the same family: per-(family,
    # framework) keying must not let one framework's vocab validate another's facts.
    by_op = {
        "context_attention_perf": [
            {"framework": "sglang", "backends": ["fa3"]},
            {"framework": "vllm", "backends": ["fa3"]},
        ]
    }
    catalog = {
        "families": [
            {
                "family": "attention",
                "op_files": ["context_attention_perf"],
                "frameworks": {
                    "sglang": {"choices": [{"backend": "fa3"}]},
                    "vllm": {"choices": [{"backend": "flash_attn"}]},
                },
            }
        ]
    }
    problems = backend_facts_module.catalog_inconsistencies(by_op, catalog)
    assert len(problems) == 1
    assert "framework=vllm" in problems[0] and "fa3" in problems[0]


def test_catalog_inconsistencies_validates_trtllm_internal(backend_facts_module):
    # trtllm_internal is no longer a blanket exemption: it must appear in the
    # catalog's choice list like any other backend.
    by_op = {"moe_perf": [{"framework": "trtllm", "backends": ["trtllm_internal"]}]}
    catalog = {
        "families": [
            {
                "family": "moe",
                "op_files": ["moe_perf"],
                "frameworks": {"trtllm": {"choices": [{"backend": "cutlass"}]}},
            }
        ]
    }
    problems = backend_facts_module.catalog_inconsistencies(by_op, catalog)
    assert any("trtllm_internal" in p for p in problems)

    catalog["families"][0]["frameworks"]["trtllm"]["choices"].append({"backend": "trtllm_internal"})
    assert backend_facts_module.catalog_inconsistencies(by_op, catalog) == []


def test_catalog_inconsistencies_still_exempts_unverified(backend_facts_module):
    by_op = {"moe_perf": [{"framework": "trtllm", "backends": ["unverified"]}]}
    catalog = {
        "families": [
            {
                "family": "moe",
                "op_files": ["moe_perf"],
                "frameworks": {"trtllm": {"choices": [{"backend": "cutlass"}]}},
            }
        ]
    }
    assert backend_facts_module.catalog_inconsistencies(by_op, catalog) == []


def _write_inconsistent_catalog(tmp_path):
    """A catalog whose (attention, sglang) choice space is missing 'triton', so
    the fp8 slice in `data_tree` (backends [fa3, triton]) trips a catalog violation."""
    catalog_path = tmp_path / "op_backend_catalog.yaml"
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "families": [
                    {
                        "family": "attention",
                        "op_files": ["context_attention_perf"],
                        "frameworks": {"sglang": {"choices": [{"backend": "fa3"}]}},  # missing "triton"
                    },
                    {"family": "mla", "op_files": ["context_mla_perf"]},
                ]
            }
        )
    )
    return catalog_path


def _write_backend_map(tmp_path):
    backend_map_path = tmp_path / "kernel_source_backends.yaml"
    backend_map_path.write_text(yaml.safe_dump({"mappings": MAPPINGS}))
    return backend_map_path


def test_main_check_fails_on_catalog_inconsistencies(backend_facts_module, data_tree, tmp_path, monkeypatch, capsys):
    backend_map_path = _write_backend_map(tmp_path)
    catalog_path = _write_inconsistent_catalog(tmp_path)

    # Registry in sync with the data (no drift) so only the catalog check can fail.
    facts, axes_by_op = backend_facts_module.scan(data_tree)
    mappings = backend_facts_module.load_backend_map(backend_map_path)
    by_op = backend_facts_module._fact_entries(facts, axes_by_op, mappings)
    registry_path = tmp_path / "op_backend_facts.yaml"
    registry_path.write_text(backend_facts_module.render_yaml(by_op, axes_by_op, FAMILIES))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backend_facts.py",
            "--data-root",
            str(data_tree),
            "--registry",
            str(registry_path),
            "--backend-map",
            str(backend_map_path),
            "--catalog",
            str(catalog_path),
            "--check",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        backend_facts_module.main()
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "catalog: backend-not-in-catalog: family=attention framework=sglang" in out and "triton" in out


def test_main_write_refuses_when_catalog_inconsistent(backend_facts_module, data_tree, tmp_path, monkeypatch, capsys):
    backend_map_path = _write_backend_map(tmp_path)
    catalog_path = _write_inconsistent_catalog(tmp_path)
    registry_path = tmp_path / "op_backend_facts.yaml"  # does not exist yet

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backend_facts.py",
            "--data-root",
            str(data_tree),
            "--registry",
            str(registry_path),
            "--backend-map",
            str(backend_map_path),
            "--catalog",
            str(catalog_path),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        backend_facts_module.main()
    assert exc_info.value.code == 1
    # Refused before render_yaml: no registry written, so "family: unknown" can never land on disk.
    assert not registry_path.exists()
    out = capsys.readouterr().out
    assert "backend-not-in-catalog: family=attention framework=sglang" in out and "triton" in out
