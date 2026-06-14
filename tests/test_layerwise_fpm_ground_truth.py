import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "collector" / "layerwise" / "common"))
sys.path.insert(0, str(ROOT / "collector" / "layerwise" / "fpm_ground_truth"))

import send_requests
import summarize_fpm
from random_prompt_tokens import RandomPromptTokenConfig


def _send_args(tmp_path, **overrides):
    args = argparse.Namespace(
        endpoint="completions",
        vary_isl_osl=True,
        isl_values="4,8",
        osl_values="1",
        isl_min=1,
        isl_max=8,
        osl_min=1,
        osl_max=1,
        requests=3,
        max_tokens=1,
        prompt_token_seed=0,
        prompt_token_mode="random_vocab_excluding_special",
        prompt_token_pool=[],
        prompt_rng=random.Random(0),
        prompt_token_config=RandomPromptTokenConfig(100, frozenset({0, 1})),
        workload_output=str(tmp_path / "workload.csv"),
        workload_label="context",
        append_workload=False,
        request_index_offset=10,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_send_requests_labels_and_appends_workload_csv(tmp_path):
    first = _send_args(tmp_path)
    specs = send_requests.build_specs(first)

    second = _send_args(
        tmp_path,
        requests=1,
        isl_values="16",
        workload_label="decode_b1",
        append_workload=True,
        request_index_offset=20,
    )
    send_requests.build_specs(second)

    rows = list(csv.DictReader((tmp_path / "workload.csv").open()))
    assert [spec["index"] for spec in specs] == [10, 11, 12]
    assert [row["workload_label"] for row in rows] == ["context", "context", "context", "decode_b1"]
    assert [int(row["request_index"]) for row in rows] == [10, 11, 12, 20]
    assert [int(row["target_isl"]) for row in rows] == [4, 8, 4, 16]


def test_send_requests_prompt_seed_is_optional_and_reproducible_when_set(tmp_path):
    seeded_first = _send_args(tmp_path, prompt_token_seed=123, prompt_rng=random.Random(999))
    seeded_second = _send_args(tmp_path, prompt_token_seed=123, prompt_rng=random.Random(111))
    unseeded = _send_args(tmp_path, prompt_token_seed=None, prompt_rng=random.Random(123))

    assert send_requests.make_token_ids(seeded_first, 16, 10) == send_requests.make_token_ids(
        seeded_second,
        16,
        10,
    )
    assert send_requests.make_token_ids(seeded_first, 16, 10) != send_requests.make_token_ids(
        unseeded,
        16,
        10,
    )


def test_send_requests_safe_ascii_prompt_pool_limits_sampled_tokens(tmp_path):
    args = _send_args(
        tmp_path,
        prompt_token_mode="safe_ascii",
        prompt_token_pool=[7, 11, 13],
    )

    token_ids = send_requests.make_token_ids(args, 64, 10)

    assert len(token_ids) == 64
    assert set(token_ids) <= {7, 11, 13}


def test_send_requests_printable_ascii_token_filter():
    assert send_requests.is_printable_ascii_token_text(" hello")
    assert send_requests.is_printable_ascii_token_text("A")
    assert not send_requests.is_printable_ascii_token_text("")
    assert not send_requests.is_printable_ascii_token_text("   ")
    assert not send_requests.is_printable_ascii_token_text("Eva𠅁")


def test_send_requests_real_workload_uses_fallback_shapes_when_dataset_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(send_requests, "load_openassistant_shapes", lambda *args, **kwargs: [])
    args = _send_args(
        tmp_path,
        real_workload=True,
        requests=5,
        max_model_len=2048,
        real_workload_dataset="missing/dataset",
        real_workload_max_rows=10,
    )

    specs = send_requests.build_specs(args)

    rows = list(csv.DictReader((tmp_path / "workload.csv").open()))
    assert len(specs) == 5
    assert len({int(row["target_isl"]) for row in rows}) > 1
    assert all(int(row["target_isl"]) + int(row["target_osl"]) <= 2048 for row in rows)
    assert {row["shape_source"] for row in rows} == {"synthetic_large_shape_distribution"}


def test_send_requests_real_workload_scales_dataset_shapes_to_large_distribution(tmp_path, monkeypatch):
    monkeypatch.setattr(
        send_requests,
        "load_openassistant_shapes",
        lambda *args, **kwargs: [(5, 10), (100, 20), (20, 100), (300, 30)],
    )
    args = _send_args(
        tmp_path,
        real_workload=True,
        requests=4,
        max_model_len=32768,
        real_workload_dataset="OpenAssistant/oasst1",
        real_workload_max_rows=10,
        real_workload_shape_source="scaled_dataset",
    )

    specs = send_requests.build_specs(args)

    isls = [spec["target_isl"] for spec in specs]
    osls = [spec["target_osl"] for spec in specs]
    assert min(isls) >= 100
    assert max(isls) == 16384
    assert min(osls) >= 100
    assert max(osls) == 4096


def test_summarize_fpm_classifies_context_decode_and_mixed_rows(tmp_path):
    detail_path = tmp_path / "detail.csv"
    output_path = tmp_path / "phase.csv"
    fieldnames = [
        "workload_segment",
        "counter_id",
        "worker_id",
        "dp_rank",
        "sum_prefill_tokens",
        "num_prefill_requests",
        "sum_prefill_kv_tokens",
        "var_prefill_length",
        "num_decode_requests",
        "sum_decode_kv_tokens",
        "var_decode_kv_tokens",
        "queued_prefill_requests",
        "queued_prefill_tokens",
        "queued_var_prefill_length",
        "queued_decode_requests",
        "queued_decode_kv_tokens",
        "queued_var_decode_kv_tokens",
        "latency_ms",
    ]
    rows = [
        ["sweep", 1, "w", 0, 64, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "1.0"],
        ["sweep", 2, "w", 0, 0, 0, 0, 0, 4, 4096, 0, 0, 0, 0, 0, 0, 0, "2.0"],
        ["real", 3, "w", 0, 128, 1, 0, 0, 3, 3072, 0, 0, 0, 0, 0, 0, 0, "3.0"],
        ["real", 4, "w", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "4.0"],
    ]
    with detail_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        writer.writerows(rows)

    count = summarize_fpm.summarize_file(detail_path, output_path)

    summarized = list(csv.DictReader(output_path.open()))
    assert count == 3
    assert [row["phase"] for row in summarized] == ["context", "decode", "mixed"]
    assert [row["workload_segment"] for row in summarized] == ["sweep", "sweep", "real"]
    assert summarized[1]["decode_tokens"] == "4"
    assert summarized[1]["mean_decode_kv_tokens"] == "1024.000"
