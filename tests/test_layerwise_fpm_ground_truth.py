import argparse
import csv
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


def test_summarize_fpm_classifies_context_decode_and_mixed_rows(tmp_path):
    detail_path = tmp_path / "detail.csv"
    output_path = tmp_path / "phase.csv"
    fieldnames = [
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
        [1, "w", 0, 64, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "1.0"],
        [2, "w", 0, 0, 0, 0, 0, 4, 4096, 0, 0, 0, 0, 0, 0, 0, "2.0"],
        [3, "w", 0, 128, 1, 0, 0, 3, 3072, 0, 0, 0, 0, 0, 0, 0, "3.0"],
        [4, "w", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "4.0"],
    ]
    with detail_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        writer.writerows(rows)

    count = summarize_fpm.summarize_file(detail_path, output_path)

    summarized = list(csv.DictReader(output_path.open()))
    assert count == 3
    assert [row["phase"] for row in summarized] == ["context", "decode", "mixed"]
    assert summarized[1]["decode_tokens"] == "4"
    assert summarized[1]["mean_decode_kv_tokens"] == "1024.000"
