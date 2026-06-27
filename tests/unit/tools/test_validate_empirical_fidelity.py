# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools" / "accuracy_regression_testing"


@pytest.fixture(scope="module")
def fidelity_module():
    path = TOOLS_DIR / "validate_empirical_fidelity.py"
    spec = importlib.util.spec_from_file_location("validate_empirical_fidelity", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _case(points):
    return {
        "id": "dense-bf16",
        "family": "dense",
        "quant": "BF16",
        "model": "org/model",
        "system": "test_gpu",
        "backend": "test_backend",
        "version": "1.0",
        "model_config": {"tp_size": 2},
        "points": points,
    }


def test_generated_off_grid_points_are_deterministic(fidelity_module):
    spec = {
        "count": 12,
        "seed": 17,
        "distribution": "log_uniform",
        "ranges": {"batch_size": [2, 31], "isl": [1000, 9000], "osl": [2, 300]},
        "avoid": {
            "batch_size": [2, 4, 8, 16],
            "isl": [1024, 2048, 4096, 8192],
            "osl": [2, 4, 8, 16, 32, 64, 128, 256],
        },
    }

    first = fidelity_module.generate_points(spec, ["prefill", "decode"])
    second = fidelity_module.generate_points(spec, ["prefill", "decode"])

    assert first == second
    assert len(first) == 12
    assert len({tuple(point[field] for field in fidelity_module.POINT_FIELDS) for point in first}) == 12
    for point in first:
        assert point["batch_size"] not in spec["avoid"]["batch_size"]
        assert point["isl"] not in spec["avoid"]["isl"]
        assert point["osl"] not in spec["avoid"]["osl"]
        assert point["phases"] == ["prefill", "decode"]
        assert point["sample_kind"] == "offgrid"


def test_generated_points_respect_workload_envelope(fidelity_module):
    spec = {
        "count": 20,
        "seed": 19,
        "distribution": "log_uniform",
        "ranges": {"batch_size": [2, 64], "isl": [100, 16000], "osl": [2, 1024]},
        "max_batch_tokens": 32768,
        "max_sequence": 16384,
    }

    points = fidelity_module.generate_points(spec, ["prefill", "decode"])

    assert len(points) == 20
    assert all(point["batch_size"] * (point["isl"] + point["osl"]) <= 32768 for point in points)
    assert all(point["isl"] + point["osl"] <= 16384 for point in points)


def test_generated_point_id_is_unique_after_repeated_prefix_collision(fidelity_module):
    case = _case(
        [
            {"id": "g000", "batch_size": 3, "isl": 1500, "osl": 8},
            {"id": "generated-g000", "batch_size": 5, "isl": 1700, "osl": 9},
        ]
    )
    case["point_generation"] = {
        "count": 1,
        "seed": 1,
        "ranges": {"batch_size": [7, 7], "isl": [1900, 1900], "osl": [10, 10]},
    }

    points = fidelity_module.normalize_matrix({"cases": [case]})[0]["points"]

    assert [point["point_id"] for point in points] == ["g000", "generated-g000", "generated-generated-g000"]


def test_p90_uses_nearest_rank_observation(fidelity_module):
    assert fidelity_module._percentile([0.0] * 9 + [100.0], 0.9) == 0.0
    assert fidelity_module._percentile([0.0] * 8 + [50.0, 100.0], 0.9) == 50.0


def test_explicit_points_can_select_phases(fidelity_module):
    matrix = {
        "defaults": {"phases": ["prefill", "decode"]},
        "cases": [
            _case(
                [
                    {"id": "both", "batch_size": 3, "isl": 1500, "osl": 33},
                    {
                        "id": "decode-only",
                        "batch_size": 11,
                        "isl": 3000,
                        "osl": 257,
                        "phases": ["static_gen"],
                    },
                ]
            )
        ],
    }

    case = fidelity_module.normalize_matrix(matrix)[0]

    assert case["points"][0]["phases"] == ["prefill", "decode"]
    assert case["points"][0]["sample_kind"] == "explicit"
    assert case["points"][1]["phases"] == ["decode"]


def test_case_can_reclassify_one_phase(fidelity_module):
    case = _case([{"id": "point", "sample_kind": "offgrid", "batch_size": 3, "isl": 1500, "osl": 8}])
    case["phase_sample_kinds"] = {"decode": "extrapolation"}
    normalized = fidelity_module.normalize_matrix({"cases": [case]})[0]
    point = normalized["points"][0]

    assert fidelity_module._metadata(normalized, point, "prefill", "off")["sample_kind"] == "offgrid"
    assert fidelity_module._metadata(normalized, point, "decode", "off")["sample_kind"] == "extrapolation"


def test_transfer_policy_defaults_off_and_is_a_global_override(fidelity_module):
    assert fidelity_module.parse_args([]).transfer_policy == "off"
    assert fidelity_module.parse_args(["--transfer-policy", "aggressive"]).transfer_policy == "aggressive"

    case = _case([{"id": "point", "batch_size": 3, "isl": 1500, "osl": 8}])
    seen = []

    def factory(case, mode, transfer_policy):
        seen.append((mode, transfer_policy))
        return lambda point, phase: fidelity_module.Measurement(10.0, {"op": 10.0})

    report = fidelity_module.run_matrix(
        {"cases": [case]},
        mode_runner_factory=factory,
        transfer_policy="aggressive",
    )

    assert seen == [("SILICON", "aggressive"), ("EMPIRICAL", "aggressive")]
    assert {row["transfer_policy"] for row in report.observations} == {"aggressive"}
    assert {row["transfer_policy"] for row in report.pairs} == {"aggressive"}
    assert {row["transfer_policy"] for row in report.summary} == {"aggressive"}
    assert {row["transfer_policy"] for row in report.op_summary} == {"aggressive"}

    case["transfer_policy"] = "off"
    with pytest.raises(ValueError, match="cannot vary by case"):
        fidelity_module.normalize_matrix({"cases": [case]})
    with pytest.raises(ValueError, match="cannot be set in matrix defaults"):
        fidelity_module.normalize_matrix(
            {
                "defaults": {"transfer_policy": "off"},
                "cases": [_case([{"id": "point", "batch_size": 3, "isl": 1500, "osl": 8}])],
            }
        )


def test_strict_off_excludes_transfer_tagged_pairs_but_aggressive_keeps_them(fidelity_module):
    case = _case([{"id": "point", "batch_size": 3, "isl": 1500, "osl": 8}])

    def factory(case, mode, transfer_policy):
        def run(point, phase):
            tags = frozenset({"empirical", "xshape"}) if mode == "EMPIRICAL" else frozenset()
            return fidelity_module.Measurement(
                11.0 if mode == "EMPIRICAL" else 10.0,
                {"op": 11.0 if mode == "EMPIRICAL" else 10.0},
                provenance_tags=tags,
            )

        return run

    strict = fidelity_module.run_matrix({"cases": [case]}, mode_runner_factory=factory)
    assert {row["worst_provenance"] for row in strict.observations if row["mode"] == "EMPIRICAL"} == {"xshape"}
    assert all(row["pair_status"] == "transfer_excluded" for row in strict.pairs)
    assert all(row["transfer_tagged"] for row in strict.pairs)
    assert all(json.loads(row["transfer_tags_json"]) == ["xshape"] for row in strict.pairs)
    strict_all = strict.summary[0]
    assert strict_all["comparable"] == 0
    assert strict_all["transfer_tagged_pairs"] == 2
    assert strict_all["transfer_excluded_pairs"] == 2
    assert strict_all["mean_ape_pct"] is None

    aggressive = fidelity_module.run_matrix(
        {"cases": [_case([{"id": "point", "batch_size": 3, "isl": 1500, "osl": 8}])]},
        mode_runner_factory=factory,
        transfer_policy="aggressive",
    )
    assert all(row["pair_status"] == "comparable" for row in aggressive.pairs)
    aggressive_all = aggressive.summary[0]
    assert aggressive_all["comparable"] == 2
    assert aggressive_all["transfer_tagged_pairs"] == 2
    assert aggressive_all["transfer_excluded_pairs"] == 0
    assert aggressive_all["mean_ape_pct"] == pytest.approx(10.0)


def test_sdk_runner_sets_transfer_policy_before_backend_or_model_use(fidelity_module, monkeypatch):
    from aiconfigurator.sdk import common, models, perf_database
    from aiconfigurator.sdk.backends import factory as backend_factory

    events = []

    class Database:
        def set_transfer_policy(self, policy):
            events.append(("set_transfer_policy", policy))

        def set_default_database_mode(self, mode):
            events.append(("set_default_database_mode", mode))

    def get_database(**kwargs):
        events.append("get_database")
        return Database()

    monkeypatch.setattr(perf_database, "get_database", get_database)
    monkeypatch.setattr(
        backend_factory,
        "get_backend",
        lambda backend: events.append("get_backend") or object(),
    )
    monkeypatch.setattr(
        models,
        "get_model",
        lambda *args, **kwargs: events.append("get_model") or object(),
    )

    fidelity_module._SdkModeRunner(
        _case([{"id": "point", "batch_size": 3, "isl": 1500, "osl": 8}]),
        "EMPIRICAL",
        "aggressive",
    )

    assert events == [
        "get_database",
        ("set_transfer_policy", "aggressive"),
        ("set_default_database_mode", common.DatabaseMode.EMPIRICAL),
        "get_backend",
    ]


def test_run_matrix_records_failures_and_computes_metrics_and_attribution(fidelity_module):
    matrix = {
        "cases": [
            _case(
                [
                    {"id": "a", "batch_size": 3, "isl": 1500, "osl": 11},
                    {"id": "b", "batch_size": 5, "isl": 3000, "osl": 21},
                    {"id": "c", "batch_size": 7, "isl": 5500, "osl": 41},
                    {"id": "error", "batch_size": 9, "isl": 9000, "osl": 81},
                ]
            )
        ]
    }
    silicon_values = {"a": 100.0, "b": 200.0, "c": 400.0, "error": 800.0}
    empirical_values = {"a": 110.0, "b": 180.0, "c": 440.0}
    factory_calls = []

    def factory(case, mode, transfer_policy):
        factory_calls.append((case["id"], mode, transfer_policy))

        def run(point, phase):
            if mode == "EMPIRICAL" and point["point_id"] == "error" and phase == "decode":
                raise NotImplementedError("missing empirical data")
            value = (silicon_values if mode == "SILICON" else empirical_values).get(
                point["point_id"], silicon_values[point["point_id"]]
            )
            if phase == "decode":
                value /= 10
            return fidelity_module.Measurement(
                value_ms=value,
                op_latencies={"attention": value * 0.25, "ffn": value * 0.75},
            )

        return run

    report = fidelity_module.run_matrix(matrix, mode_runner_factory=factory, worst_n=1)

    assert factory_calls == [("dense-bf16", "SILICON", "off"), ("dense-bf16", "EMPIRICAL", "off")]
    assert len(report.observations) == 4 * 2 * 2
    assert len(report.pairs) == 4 * 2
    failed = [row for row in report.observations if row["status"] == "error"]
    assert len(failed) == 1
    assert failed[0]["error_type"] == "NotImplementedError"
    assert failed[0]["error_message"] == "missing empirical data"

    prefill = next(row for row in report.summary if row["scope"] == "case_phase" and row["phase"] == "prefill")
    assert prefill["paired_coverage_pct"] == 100.0
    assert prefill["empirical_given_silicon_coverage_pct"] == 100.0
    assert prefill["eligible_coverage_pct"] == 100.0
    assert prefill["mean_ape_pct"] == pytest.approx(7.5)
    assert prefill["median_ape_pct"] == pytest.approx(10.0)
    assert prefill["p90_ape_pct"] == pytest.approx(10.0)
    assert prefill["max_ape_pct"] == pytest.approx(10.0)
    assert prefill["wape_pct"] == pytest.approx(70 / 1500 * 100)
    assert prefill["signed_bias_pct"] == pytest.approx(30 / 1500 * 100)

    decode = next(row for row in report.summary if row["scope"] == "case_phase" and row["phase"] == "decode")
    assert decode["empirical_coverage_pct"] == 75.0
    assert decode["paired_coverage_pct"] == 75.0
    assert decode["comparable"] == 3
    assert len(report.attribution) == 2 * 2  # one worst point per phase, two ops each
    assert {row["op"] for row in report.attribution} == {"attention", "ffn"}
    attention_prefill = next(
        row
        for row in report.op_summary
        if row["scope"] == "phase_op" and row["phase"] == "prefill" and row["op"] == "attention"
    )
    assert attention_prefill["count"] == 4  # all comparable points, independent of worst_n=1
    assert attention_prefill["mean_op_ape_pct"] == pytest.approx(7.5)


def test_op_summary_uses_full_op_union_and_surfaces_missing_ops(fidelity_module):
    case = _case(
        [
            {"id": "a", "batch_size": 3, "isl": 1500, "osl": 11},
            {"id": "b", "batch_size": 5, "isl": 3000, "osl": 21},
        ]
    )
    case["phases"] = ["prefill"]
    matrix = {"cases": [case]}

    def factory(case, mode, transfer_policy):
        def run(point, phase):
            if point["point_id"] == "a":
                if mode == "SILICON":
                    return fidelity_module.Measurement(20.0, {"attention": 10.0, "only_silicon": 5.0})
                return fidelity_module.Measurement(22.0, {"attention": 12.0, "only_empirical": 3.0})
            if mode == "SILICON":
                return fidelity_module.Measurement(40.0, {"attention": 20.0, "only_silicon": 4.0})
            return fidelity_module.Measurement(38.0, {"attention": 18.0, "only_silicon": 6.0})

        return run

    report = fidelity_module.run_matrix(matrix, mode_runner_factory=factory, worst_n=0)
    phase_rows = {
        row["op"]: row for row in report.op_summary if row["scope"] == "phase_op" and row["phase"] == "prefill"
    }

    attention = phase_rows["attention"]
    assert attention["count"] == 2
    assert attention["both_present_count"] == 2
    assert attention["mean_op_ape_pct"] == pytest.approx(15.0)
    assert attention["median_op_ape_pct"] == pytest.approx(15.0)
    assert attention["p90_op_ape_pct"] == pytest.approx(20.0)
    assert attention["max_op_ape_pct"] == pytest.approx(20.0)
    assert attention["wape_pct"] == pytest.approx(4 / 30 * 100)
    assert attention["signed_bias_pct"] == pytest.approx(0.0)
    assert attention["mean_silicon_latency_share_pct"] == pytest.approx(50.0)

    only_silicon = phase_rows["only_silicon"]
    assert only_silicon["count"] == 2
    assert only_silicon["empirical_missing_count"] == 1
    assert only_silicon["op_ape_count"] == 2
    assert only_silicon["mean_op_ape_pct"] == pytest.approx(75.0)
    assert only_silicon["wape_pct"] == pytest.approx(7 / 9 * 100)
    assert only_silicon["signed_bias_pct"] == pytest.approx(-3 / 9 * 100)

    only_empirical = phase_rows["only_empirical"]
    assert only_empirical["count"] == 1
    assert only_empirical["silicon_missing_count"] == 1
    assert only_empirical["op_ape_count"] == 0
    assert only_empirical["mean_op_ape_pct"] is None
    assert only_empirical["wape_pct"] is None
    assert only_empirical["mean_silicon_latency_share_pct"] == pytest.approx(0.0)

    case_attention = next(
        row for row in report.op_summary if row["scope"] == "case_phase_op" and row["op"] == "attention"
    )
    assert case_attention["case_id"] == "dense-bf16"
    assert case_attention["family"] == "dense"
    assert case_attention["quant"] == "BF16"


def test_mode_build_error_is_materialized_for_every_point(fidelity_module):
    matrix = {
        "cases": [
            _case(
                [
                    {"id": "a", "batch_size": 3, "isl": 1500, "osl": 11},
                    {"id": "b", "batch_size": 5, "isl": 3000, "osl": 21},
                ]
            )
        ]
    }

    def factory(case, mode, transfer_policy):
        if mode == "EMPIRICAL":
            raise RuntimeError("database unavailable")
        return lambda point, phase: fidelity_module.Measurement(1.0, {"op": 1.0})

    report = fidelity_module.run_matrix(matrix, mode_runner_factory=factory)

    empirical = [row for row in report.observations if row["mode"] == "EMPIRICAL"]
    assert len(empirical) == 4
    assert all(row["status"] == "error" for row in empirical)
    assert all(row["error_message"] == "database unavailable" for row in empirical)
    assert all(row["pair_status"] == "empirical_error" for row in report.pairs)
    assert len(fidelity_module.unexpected_errors(report)) == 4


def test_expected_empirical_gap_is_not_an_unexpected_error(fidelity_module):
    matrix = {"cases": [_case([{"id": "gap", "batch_size": 3, "isl": 1500, "osl": 8}])]}

    def factory(case, mode, transfer_policy):
        def run(point, phase):
            if mode == "EMPIRICAL":
                error = type("EmpiricalNotImplementedError", (RuntimeError,), {})
                raise error("no empirical data")
            return fidelity_module.Measurement(10.0, {"op": 10.0})

        return run

    report = fidelity_module.run_matrix(matrix, mode_runner_factory=factory)

    assert all(row["pair_status"] == "empirical_gap" for row in report.pairs)
    assert fidelity_module.unexpected_errors(report) == []


def test_mean_ape_threshold_fails_without_comparable_pairs(fidelity_module):
    summary = {
        "eligible_coverage_pct": 0.0,
        "eligible": 10,
        "silicon_coverage_pct": 100.0,
        "mean_ape_pct": None,
    }

    failures = fidelity_module.threshold_failures(summary, max_mean_ape=5.0)

    assert failures == ["mean APE is unavailable because there are no comparable pairs"]


def test_coverage_thresholds_guard_denominator_and_count(fidelity_module):
    summary = {
        "eligible_coverage_pct": 99.0,
        "eligible": 1,
        "silicon_coverage_pct": 10.0,
        "mean_ape_pct": 1.0,
    }

    failures = fidelity_module.threshold_failures(
        summary,
        min_eligible_coverage=100.0,
        min_eligible_count=10,
        min_silicon_coverage=90.0,
        max_mean_ape=5.0,
    )

    assert len(failures) == 3
    assert failures[0].startswith("eligible coverage")
    assert failures[1] == "eligible count 1 < 10"
    assert failures[2].startswith("silicon coverage")


def test_oom_observations_are_retained_but_excluded_from_fidelity(fidelity_module):
    matrix = {"cases": [_case([{"id": "oom", "batch_size": 3, "isl": 1500, "osl": 8}])]}

    def factory(case, mode, transfer_policy):
        return lambda point, phase: fidelity_module.Measurement(10.0, {"op": 10.0}, oom=True)

    report = fidelity_module.run_matrix(matrix, mode_runner_factory=factory)

    assert all(row["status"] == "oom" for row in report.observations)
    assert all(row["value_ms"] == 10.0 for row in report.observations)
    assert all(row["pair_status"] == "oom" for row in report.pairs)
    case_phase = next(row for row in report.summary if row["scope"] == "case_phase")
    assert case_phase["comparable"] == 0
    assert case_phase["silicon_ooms"] == 1
    assert case_phase["empirical_ooms"] == 1


def test_image_point_and_encoder_phase_are_preserved(fidelity_module):
    matrix = {
        "cases": [
            _case(
                [
                    {
                        "id": "image",
                        "sample_kind": "image",
                        "batch_size": 3,
                        "isl": 1500,
                        "osl": 8,
                        "image_height": 448,
                        "image_width": 448,
                        "num_images_per_request": 2,
                        "phases": ["prefill", "encoder"],
                    }
                ]
            )
        ]
    }

    point = fidelity_module.normalize_matrix(matrix)[0]["points"][0]

    assert point["sample_kind"] == "image"
    assert point["image_height"] == 448
    assert point["image_width"] == 448
    assert point["num_images_per_request"] == 2
    assert point["phases"] == ["prefill", "encoder"]


def test_full_value_and_ops_use_the_same_phase_boundary(fidelity_module):
    class Summary:
        def get_summary_df(self):
            return [{"context_latency": 9.877, "encoder_latency": 4.5, "tpot": 1.25}]

        def get_context_latency_dict(self):
            return {"context_op": 9.8765}

        def get_encoder_latency_dict(self):
            return {"encoder_op": 4.5}

        def get_generation_latency_dict(self):
            return {"generation_op": 10.0}

    summary = Summary()

    assert fidelity_module._summary_value(summary, "prefill") == 9.877
    assert fidelity_module._summary_ops(summary, "prefill", 9) == {"context_op": 9.8765}
    assert fidelity_module._summary_value(summary, "encoder") == 4.5
    assert fidelity_module._summary_ops(summary, "encoder", 9) == {"encoder_op": 4.5}
    assert fidelity_module._summary_value(summary, "decode") == 1.25
    assert fidelity_module._summary_ops(summary, "decode", 9) == {"generation_op": 1.25}


def test_summary_separates_sample_kinds_and_conditions_coverage_on_silicon(fidelity_module):
    matrix = {
        "cases": [
            _case(
                [
                    {"id": "offgrid", "sample_kind": "offgrid", "batch_size": 3, "isl": 1500, "osl": 8},
                    {"id": "boundary", "sample_kind": "boundary", "batch_size": 9, "isl": 2049, "osl": 8},
                ]
            )
        ]
    }

    def factory(case, mode, transfer_policy):
        def run(point, phase):
            if point["point_id"] == "boundary" and mode == "SILICON":
                raise RuntimeError("bad silicon reference")
            if point["point_id"] == "boundary" and mode == "EMPIRICAL":
                raise NotImplementedError("empirical gap")
            return fidelity_module.Measurement(10.0, {"op": 10.0})

        return run

    report = fidelity_module.run_matrix(matrix, mode_runner_factory=factory)
    offgrid = next(
        row
        for row in report.summary
        if row["scope"] == "case_phase_sample_kind" and row["phase"] == "prefill" and row["sample_kind"] == "offgrid"
    )
    boundary = next(
        row
        for row in report.summary
        if row["scope"] == "case_phase_sample_kind" and row["phase"] == "prefill" and row["sample_kind"] == "boundary"
    )

    assert offgrid["paired_coverage_pct"] == 100.0
    assert offgrid["empirical_given_silicon_coverage_pct"] == 100.0
    assert boundary["silicon_coverage_pct"] == 0.0
    assert boundary["empirical_given_silicon_coverage_pct"] == 0.0


def test_write_report_emits_csv_and_json_for_each_dataset(fidelity_module, tmp_path):
    matrix = {"cases": [_case([{"id": "a", "batch_size": 3, "isl": 1500, "osl": 11}])]}

    def factory(case, mode, transfer_policy):
        multiplier = 1.1 if mode == "EMPIRICAL" else 1.0
        return lambda point, phase: fidelity_module.Measurement(10 * multiplier, {"attention": 10 * multiplier})

    report = fidelity_module.run_matrix(matrix, mode_runner_factory=factory, worst_n=1)
    fidelity_module.write_report(report, tmp_path)

    for dataset in ("observations", "pairs", "summary", "op_summary", "attribution"):
        assert (tmp_path / f"{dataset}.csv").is_file()
        json_path = tmp_path / f"{dataset}.json"
        assert json_path.is_file()
        assert isinstance(json.loads(json_path.read_text()), list)
    combined = json.loads((tmp_path / "report.json").read_text())
    assert set(combined) == {"observations", "pairs", "summary", "op_summary", "attribution"}


def test_default_matrix_covers_model_families_and_quantizations(fidelity_module):
    matrix = fidelity_module.load_matrix(fidelity_module.DEFAULT_MATRIX)

    assert "transfer_policy" not in matrix.get("defaults", {})
    assert all("transfer_policy" not in case for case in matrix["cases"])
    assert {case["family"] for case in matrix["cases"]} >= {"dense", "MoE", "DSA"}
    assert {case["quant"] for case in matrix["cases"]} >= {"BF16", "FP8", "NVFP4"}


def test_stack_matrix_is_normalizable_and_keeps_reference_axes_separate(fidelity_module):
    path = TOOLS_DIR / "empirical_fidelity_stack_matrix.json"
    matrix = fidelity_module.load_matrix(path)
    assert "transfer_policy" not in matrix.get("defaults", {})
    assert all("transfer_policy" not in case for case in matrix["cases"])
    cases = fidelity_module.normalize_matrix(matrix)

    assert {case["backend"] for case in cases} == {"sglang", "trtllm", "vllm"}
    assert {case["system"] for case in cases} >= {"b200_sxm", "h100_sxm", "b300_sxm"}
    assert all(len(case["points"]) == 9 for case in cases)
    assert all(len({point["point_id"] for point in case["points"]}) == 9 for case in cases)
