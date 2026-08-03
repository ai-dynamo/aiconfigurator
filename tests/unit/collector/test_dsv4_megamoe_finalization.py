# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
from types import SimpleNamespace

import pytest
import yaml

from collector import provenance
from collector.sglang.dsv4_megamoe import finalize_validated

pytestmark = pytest.mark.unit


def _write_staging_csv(path, *, version: str = "0.5.15", num_tokens: str = "1024"):
    row = {
        "framework": "SGLang",
        "version": version,
        "phase": "context",
        "moe_ep_size": "4",
        "num_tokens": num_tokens,
        "distribution": "balanced",
        "latency": "1.25",
    }
    with path.open("w", newline="", encoding="utf-8") as perf_file:
        writer = csv.DictWriter(perf_file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _args(tmp_path, *, version: str = "0.5.15"):
    staging_csv = tmp_path / finalize_validated.PERF_FILE
    summary = tmp_path / "validation_summary.txt"
    summary.write_text("total_rows=1 expected=1\nVALIDATION=PASS\n", encoding="utf-8")
    return SimpleNamespace(
        staging_csv=staging_csv,
        validation_summary=summary,
        target_dir=tmp_path / "packaged" / version,
        target_sglang_version=version,
        image="lmsysorg/sglang:v0.5.15",
        collector_ref="a" * 40,
        collector_hash="sha256:" + "b" * 64,
        model_config="dsv4_pro",
        prefill_ep_sizes="4",
        decode_ep_sizes="",
        prefill_tokens="1024",
        decode_tokens="",
        distributions="balanced",
        phase_order="context",
        routing_seed=0,
        routing_seeds="",
        source_policy="random",
        pre_dispatch="sglang_jit",
        cap_policy="case_tokens",
        prefill_num_max_tokens_per_rank=32768,
        decode_num_max_tokens_per_rank=512,
        include_routed_scale=1,
        renormalize_topk_weights=1,
        num_warmup=5,
        num_iterations=20,
    )


def test_publish_converts_in_staging_and_writes_full_provenance(tmp_path):
    args = _args(tmp_path)
    _write_staging_csv(args.staging_csv)

    target_parquet, target_meta = finalize_validated.publish_validated(args)

    assert args.staging_csv.exists()
    assert args.staging_csv.with_suffix(".parquet").exists()
    assert target_parquet.exists()
    assert not (args.target_dir / finalize_validated.PERF_FILE).exists()

    doc = yaml.safe_load(target_meta.read_text(encoding="utf-8"))
    assert doc["runtime"] == {
        "framework": "sglang",
        "version": "0.5.15",
        "image": "lmsysorg/sglang:v0.5.15",
    }
    table = doc["tables"][finalize_validated.TABLE_STEM]
    assert table["collector_ref"] == "a" * 40
    assert table["collector_hash"] == "sha256:" + "b" * 64
    assert table["case_plan_hash"].startswith("sha256:")
    assert table["rows"] == 1
    assert table["status"] == "complete"


def test_publish_preserves_existing_full_provenance_entries(tmp_path):
    args = _args(tmp_path)
    _write_staging_csv(args.staging_csv)
    args.target_dir.mkdir(parents=True)
    existing_table = {
        "collector_ref": "c" * 40,
        "collector_hash": "sha256:" + "d" * 64,
        "case_plan_hash": "sha256:" + "e" * 64,
        "collected_at": "2026-07-20",
        "rows": 2,
        "status": "complete",
    }
    provenance.write_collection_meta(
        args.target_dir,
        {"framework": "sglang", "version": "0.5.15", "image": "lmsysorg/sglang:v0.5.15"},
        {"moe_perf": existing_table},
    )

    _, target_meta = finalize_validated.publish_validated(args)

    doc = yaml.safe_load(target_meta.read_text(encoding="utf-8"))
    assert doc["tables"]["moe_perf"] == existing_table
    assert finalize_validated.TABLE_STEM in doc["tables"]


def test_publish_rejects_legacy_sidecar_without_touching_target(tmp_path):
    args = _args(tmp_path)
    _write_staging_csv(args.staging_csv)
    args.target_dir.mkdir(parents=True)
    (args.target_dir / "collection_meta.yaml").write_text(
        "schema_version: 1\nprovenance: legacy\nruntime:\n  framework: sglang\n"
        "  version: 0.5.15\ntables:\n  dsv4_megamoe_module_perf:\n    status: complete\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="provenance: legacy"):
        finalize_validated.publish_validated(args)

    assert not (args.target_dir / "dsv4_megamoe_module_perf.parquet").exists()
    assert not args.staging_csv.with_suffix(".parquet").exists()


def test_publish_rejects_collected_version_mismatch(tmp_path):
    args = _args(tmp_path)
    _write_staging_csv(args.staging_csv, version="0.5.14")

    with pytest.raises(RuntimeError, match="exactly match destination"):
        finalize_validated.publish_validated(args)

    assert not args.target_dir.exists()


def test_publish_rejects_rows_that_do_not_match_requested_plan(tmp_path):
    args = _args(tmp_path)
    _write_staging_csv(args.staging_csv, num_tokens="2048")

    with pytest.raises(RuntimeError, match="do not exactly match"):
        finalize_validated.publish_validated(args)

    assert not args.target_dir.exists()
