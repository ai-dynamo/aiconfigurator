# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from collector.fpm_forward.trtllm_grid import (
    TRTLLM_FULL_GRID_V1,
    TrtllmGridAdmissionSummary,
    TrtllmGridBuild,
    build_trtllm_full_grid,
    build_trtllm_grid,
)
from collector.fpm_forward.trtllm_state import (
    TrtllmCoordinate,
    TrtllmManifest,
    TrtllmRuntimeLimits,
)

pytestmark = pytest.mark.unit


def _kimi_runtime_limits() -> TrtllmRuntimeLimits:
    return TrtllmRuntimeLimits(
        max_seq_len=262_144,
        max_num_requests=128,
        max_batch_size=128,
        max_num_tokens=8_192,
        kv_cache_max_num_blocks=72_485,
        kv_cache_tokens_per_block=32,
        decode_cuda_graph_batch_sizes=(1, 2, 4, 8, 16, 32, 64, 128),
    )


def test_versioned_profile_declares_and_controls_the_grid_shape():
    limits = TrtllmRuntimeLimits(
        max_seq_len=100,
        max_num_requests=1,
        max_batch_size=1,
        max_num_tokens=10,
        kv_cache_max_num_blocks=1_000,
        kv_cache_tokens_per_block=1,
        decode_cuda_graph_batch_sizes=(1,),
    )
    ternary_prefill = replace(
        TRTLLM_FULL_GRID_V1,
        profile_id="test-ternary-prefill-v1",
        prefill_total_tokens_geometric_base=3,
    )

    default_totals = {
        point.total_prefill_tokens
        for point in build_trtllm_grid(limits, profile=TRTLLM_FULL_GRID_V1).coordinates
        if point.workload_kind == "prefill"
    }
    ternary_totals = {
        point.total_prefill_tokens
        for point in build_trtllm_grid(limits, profile=ternary_prefill).coordinates
        if point.workload_kind == "prefill"
    }

    assert TRTLLM_FULL_GRID_V1.to_dict() == {
        "profile_id": "trtllm_full_grid_v1",
        "prefill_total_tokens_geometric_base": 2,
        "prefill_batch_size_geometric_base": 2,
        "prefill_kv_read_blocks_geometric_base": 2,
        "prefill_prefix_blocks_per_request_anchors": [0, 1],
        "decode_graph_batch_offsets": [0, 1],
        "decode_eager_batch_geometric_base": 2,
        "decode_kv_read_tokens_geometric_base": 2,
        "include_direct_limit_endpoints": True,
        "include_kv_capacity_endpoints": True,
        "prefill_max_new_tokens": 1,
        "decode_max_new_tokens": 2,
        "phase_order": ["prefill", "decode"],
        "descending_phases": ["prefill"],
    }
    assert default_totals == {1, 2, 4, 8, 10}
    assert ternary_totals == {1, 3, 9, 10}


def test_grid_drops_phases_that_cannot_reserve_generation_blocks():
    limits = TrtllmRuntimeLimits(
        max_seq_len=4,
        max_num_requests=2,
        max_batch_size=2,
        max_num_tokens=2,
        kv_cache_max_num_blocks=1,
        kv_cache_tokens_per_block=1,
        decode_cuda_graph_batch_sizes=(1, 2),
    )

    assert build_trtllm_full_grid(limits) == ()


def test_grid_reports_aggregate_kv_capacity_admission():
    limits = TrtllmRuntimeLimits(
        max_seq_len=4,
        max_num_requests=2,
        max_batch_size=2,
        max_num_tokens=2,
        kv_cache_max_num_blocks=1,
        kv_cache_tokens_per_block=1,
        decode_cuda_graph_batch_sizes=(1, 2),
    )

    build = build_trtllm_grid(limits, profile=TRTLLM_FULL_GRID_V1)

    assert build.admission_summary.to_dict() == {
        "candidate_count": 12,
        "admitted_count": 0,
        "dropped_count": 12,
        "dropped_by_reason": {"kv_cache_capacity": 12},
    }


def test_grid_admission_reserves_the_profile_max_new_tokens():
    limits = TrtllmRuntimeLimits(
        max_seq_len=10,
        max_num_requests=1,
        max_batch_size=1,
        max_num_tokens=4,
        kv_cache_max_num_blocks=2,
        kv_cache_tokens_per_block=4,
        decode_cuda_graph_batch_sizes=(1,),
    )

    grid = build_trtllm_full_grid(limits)
    prefill_at_four = [point for point in grid if point.workload_kind == "prefill" and point.total_prefill_tokens == 4]
    decode_maximum = max(point.total_kv_read_tokens for point in grid if point.workload_kind == "decode")

    assert prefill_at_four == [TrtllmCoordinate("prefill", 1, 4, 0)]
    assert decode_maximum == 6


def test_decode_is_absent_when_a_context_token_plus_two_tokens_cannot_fit():
    limits = TrtllmRuntimeLimits(
        max_seq_len=2,
        max_num_requests=1,
        max_batch_size=1,
        max_num_tokens=1,
        kv_cache_max_num_blocks=10,
        kv_cache_tokens_per_block=2,
        decode_cuda_graph_batch_sizes=(1,),
    )

    assert all(point.workload_kind == "prefill" for point in build_trtllm_full_grid(limits))


def test_decode_keeps_the_kv_capacity_batch_boundary():
    limits = TrtllmRuntimeLimits(
        max_seq_len=3,
        max_num_requests=4,
        max_batch_size=4,
        max_num_tokens=4,
        kv_cache_max_num_blocks=3,
        kv_cache_tokens_per_block=4,
        decode_cuda_graph_batch_sizes=(1,),
    )
    decode_batch_sizes = {
        point.batch_size for point in build_trtllm_full_grid(limits) if point.workload_kind == "decode"
    }

    assert decode_batch_sizes == {1, 2, 3}


def test_decode_capacity_batch_boundary_uses_reserved_blocks_per_request():
    limits = TrtllmRuntimeLimits(
        max_seq_len=10,
        max_num_requests=8,
        max_batch_size=8,
        max_num_tokens=8,
        kv_cache_max_num_blocks=9,
        kv_cache_tokens_per_block=1,
        decode_cuda_graph_batch_sizes=(1,),
    )
    decode_batch_sizes = {
        point.batch_size for point in build_trtllm_full_grid(limits) if point.workload_kind == "decode"
    }

    assert decode_batch_sizes == {1, 2, 3}


def test_kimi_grid_has_the_frozen_full_phase_counts():
    build = build_trtllm_grid(_kimi_runtime_limits(), profile=TRTLLM_FULL_GRID_V1)
    grid = build.coordinates

    prefill_count = sum(point.workload_kind == "prefill" for point in grid)
    decode_count = sum(point.workload_kind == "decode" for point in grid)

    assert (prefill_count, decode_count, len(grid)) == (1_214, 260, 1_474)
    assert build.admission_summary.to_dict() == {
        "candidate_count": 1_574,
        "admitted_count": 1_474,
        "dropped_count": 100,
        "dropped_by_reason": {"kv_cache_capacity": 100},
    }


def test_full_grid_logs_one_aggregate_kv_capacity_drop_summary(caplog):
    with caplog.at_level("INFO", logger="collector.fpm_forward.trtllm_grid"):
        build_trtllm_full_grid(_kimi_runtime_limits())

    records = [
        record
        for record in caplog.records
        if record.name == "collector.fpm_forward.trtllm_grid" and "trtllm_fpm_forward: dropped" in record.message
    ]
    assert [(record.levelname, record.message) for record in records] == [
        (
            "INFO",
            "trtllm_fpm_forward: dropped 100/1574 coordinates "
            "(KV-cache capacity, device=72485 blocks x 32 tokens/block=2319520 tokens)",
        )
    ]


def test_full_grid_does_not_log_a_drop_summary_when_every_candidate_is_admitted(caplog):
    limits = TrtllmRuntimeLimits(
        max_seq_len=100,
        max_num_requests=1,
        max_batch_size=1,
        max_num_tokens=10,
        kv_cache_max_num_blocks=1_000,
        kv_cache_tokens_per_block=1,
        decode_cuda_graph_batch_sizes=(1,),
    )
    build = build_trtllm_grid(limits, profile=TRTLLM_FULL_GRID_V1)

    with caplog.at_level("INFO", logger="collector.fpm_forward.trtllm_grid"):
        coordinates = build_trtllm_full_grid(limits)

    assert build.admission_summary.dropped_count == 0
    assert coordinates == build.coordinates
    assert not [
        record
        for record in caplog.records
        if record.name == "collector.fpm_forward.trtllm_grid" and "trtllm_fpm_forward: dropped" in record.message
    ]


def test_kimi_grid_build_is_persistable_with_a_campaign_independent_digest():
    limits = _kimi_runtime_limits()
    build = build_trtllm_grid(limits, profile=TRTLLM_FULL_GRID_V1)
    document = build.to_dict()
    digest_payload = {key: value for key, value in document.items() if key != "runtime_grid_digest"}
    expected_digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()

    assert set(document) == {
        "schema_name",
        "schema_version",
        "profile",
        "runtime_limits",
        "coordinates",
        "admission_summary",
        "runtime_grid_digest",
    }
    assert document["schema_name"] == "aic_trtllm_fpm_runtime_grid"
    assert document["schema_version"] == 1
    assert build.runtime_grid_digest == expected_digest
    assert build.runtime_grid_digest == "54a72f919768e35bf7a57b220f3e2248d8219d55282c1f46664a0825b957276c"
    assert document["profile"] == TRTLLM_FULL_GRID_V1.to_dict()
    assert document["runtime_limits"] == limits.to_dict()
    assert document["admission_summary"] == build.admission_summary.to_dict()
    assert document["runtime_grid_digest"] == build.runtime_grid_digest
    assert json.loads(json.dumps(document)) == document


def test_grid_digest_binds_the_full_profile_not_only_its_id():
    limits = _kimi_runtime_limits()
    coordinates = (TrtllmCoordinate("prefill", 1, 1, 0),)
    admission_summary = TrtllmGridAdmissionSummary(
        candidate_count=1,
        admitted_count=1,
        dropped_by_reason=(),
    )
    first_profile = TRTLLM_FULL_GRID_V1
    second_profile = replace(
        first_profile,
        prefill_max_new_tokens=first_profile.prefill_max_new_tokens + 1,
    )

    first = TrtllmGridBuild(first_profile, limits, coordinates, admission_summary)
    second = TrtllmGridBuild(second_profile, limits, coordinates, admission_summary)

    assert first.profile.profile_id == second.profile.profile_id
    assert first.coordinates == second.coordinates
    assert first.runtime_grid_digest != second.runtime_grid_digest


@pytest.mark.parametrize(
    "field_name",
    (
        "prefill_prefix_blocks_per_request_anchors",
        "decode_graph_batch_offsets",
        "phase_order",
        "descending_phases",
    ),
)
def test_grid_profile_rejects_mutable_sequence_fields(field_name):
    mutable_value = list(getattr(TRTLLM_FULL_GRID_V1, field_name))

    with pytest.raises(TypeError, match=rf"TrtllmGridProfile\.{field_name} must be a tuple"):
        replace(TRTLLM_FULL_GRID_V1, **{field_name: mutable_value})


def test_grid_profile_rejects_a_blank_profile_id():
    with pytest.raises(ValueError, match="profile_id must be a non-blank string"):
        replace(TRTLLM_FULL_GRID_V1, profile_id=" \t")


@pytest.mark.parametrize(
    "field_name",
    ("include_direct_limit_endpoints", "include_kv_capacity_endpoints"),
)
def test_grid_profile_endpoint_flags_are_strict_booleans(field_name):
    with pytest.raises(TypeError, match=rf"TrtllmGridProfile\.{field_name} must be a boolean"):
        replace(TRTLLM_FULL_GRID_V1, **{field_name: 1})


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("prefill_total_tokens_geometric_base", True),
        ("prefill_batch_size_geometric_base", 2.0),
        ("prefill_kv_read_blocks_geometric_base", True),
        ("decode_eager_batch_geometric_base", 2.0),
        ("decode_kv_read_tokens_geometric_base", True),
        ("prefill_max_new_tokens", 1.0),
        ("decode_max_new_tokens", True),
        ("prefill_prefix_blocks_per_request_anchors", (False,)),
        ("decode_graph_batch_offsets", (0.0,)),
    ),
)
def test_grid_profile_integer_fields_reject_booleans_and_nonintegers(field_name, invalid_value):
    with pytest.raises(TypeError, match=rf"TrtllmGridProfile\.{field_name}.*integer"):
        replace(TRTLLM_FULL_GRID_V1, **{field_name: invalid_value})


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    (
        ("prefill_prefix_blocks_per_request_anchors", (1, 0), "sorted and unique"),
        ("prefill_prefix_blocks_per_request_anchors", (0, 0), "sorted and unique"),
        ("decode_graph_batch_offsets", (1, 0), "sorted and unique"),
        ("decode_graph_batch_offsets", (0, 0), "sorted and unique"),
        ("phase_order", ("prefill", "prefill"), "prefill and decode exactly once"),
        ("descending_phases", ("prefill", "prefill"), "unique subset of phase_order"),
        ("descending_phases", ("other",), "unique subset of phase_order"),
    ),
)
def test_grid_profile_requires_canonical_axes_and_exact_phase_semantics(field_name, invalid_value, message):
    with pytest.raises(ValueError, match=message):
        replace(TRTLLM_FULL_GRID_V1, **{field_name: invalid_value})


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("phase_order", (1, "decode")),
        ("descending_phases", (1,)),
    ),
)
def test_grid_profile_phase_names_are_strict_strings(field_name, invalid_value):
    with pytest.raises(TypeError, match=rf"TrtllmGridProfile\.{field_name} values must be strings"):
        replace(TRTLLM_FULL_GRID_V1, **{field_name: invalid_value})


def test_grid_profile_descending_phases_follow_phase_order():
    with pytest.raises(ValueError, match="descending_phases must follow phase_order"):
        replace(TRTLLM_FULL_GRID_V1, descending_phases=("decode", "prefill"))


def test_grid_admission_summary_rejects_a_mutable_drop_reason_list():
    with pytest.raises(TypeError, match="dropped_by_reason must be a tuple"):
        TrtllmGridAdmissionSummary(
            candidate_count=1,
            admitted_count=0,
            dropped_by_reason=[("kv_cache_capacity", 1)],
        )


def test_grid_admission_summary_rejects_mutable_drop_reason_entries():
    with pytest.raises(TypeError, match="each dropped_by_reason entry must be a 2-tuple"):
        TrtllmGridAdmissionSummary(
            candidate_count=1,
            admitted_count=0,
            dropped_by_reason=(["kv_cache_capacity", 1],),
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("candidate_count", True),
        ("candidate_count", 1.0),
        ("admitted_count", True),
        ("admitted_count", 1.0),
    ),
)
def test_grid_admission_counts_are_strict_integers(field_name, invalid_value):
    values = {
        "candidate_count": 1,
        "admitted_count": 1,
        "dropped_by_reason": (),
    }
    values[field_name] = invalid_value

    with pytest.raises(TypeError, match=rf"TrtllmGridAdmissionSummary\.{field_name} must be an integer"):
        TrtllmGridAdmissionSummary(**values)


@pytest.mark.parametrize(
    ("invalid_reason", "error_type", "message"),
    (
        (1, TypeError, "drop reasons must be strings"),
        (" \t", ValueError, "drop reasons must be non-blank strings"),
    ),
)
def test_grid_admission_drop_reasons_are_nonblank_strings(invalid_reason, error_type, message):
    with pytest.raises(error_type, match=message):
        TrtllmGridAdmissionSummary(
            candidate_count=1,
            admitted_count=0,
            dropped_by_reason=((invalid_reason, 1),),
        )


def test_grid_admission_drop_reasons_are_canonically_sorted():
    with pytest.raises(ValueError, match="drop reasons must be sorted and unique"):
        TrtllmGridAdmissionSummary(
            candidate_count=2,
            admitted_count=0,
            dropped_by_reason=(("sequence_limit", 1), ("kv_cache_capacity", 1)),
        )


@pytest.mark.parametrize("invalid_count", (True, 1.0))
def test_grid_admission_drop_counts_are_strict_integers(invalid_count):
    with pytest.raises(TypeError, match="drop counts must be integers"):
        TrtllmGridAdmissionSummary(
            candidate_count=1,
            admitted_count=0,
            dropped_by_reason=(("kv_cache_capacity", invalid_count),),
        )


@pytest.mark.parametrize(
    ("candidate_count", "admitted_count", "dropped_by_reason", "message"),
    (
        (-1, 0, (), "admission counts must be non-negative"),
        (0, -1, (("kv_cache_capacity", 1),), "admission counts must be non-negative"),
        (0, 1, (), "admitted count cannot exceed candidate count"),
        (1, 0, (("kv_cache_capacity", 0),), "drop counts must be positive"),
        (2, 0, (("kv_cache_capacity", 1),), "account for every dropped candidate"),
        (2, 0, (("kv_cache_capacity", 1), ("kv_cache_capacity", 1)), "sorted and unique"),
    ),
)
def test_grid_admission_summary_rejects_inconsistent_counts(
    candidate_count,
    admitted_count,
    dropped_by_reason,
    message,
):
    with pytest.raises(ValueError, match=message):
        TrtllmGridAdmissionSummary(candidate_count, admitted_count, dropped_by_reason)


def test_grid_build_rejects_a_mutable_coordinate_list():
    with pytest.raises(TypeError, match=r"TrtllmGridBuild\.coordinates must be a tuple"):
        TrtllmGridBuild(
            profile=TRTLLM_FULL_GRID_V1,
            runtime_limits=_kimi_runtime_limits(),
            coordinates=[TrtllmCoordinate("prefill", 1, 1, 0)],
            admission_summary=TrtllmGridAdmissionSummary(1, 1, ()),
        )


@pytest.mark.parametrize("field_name", ("profile", "runtime_limits", "admission_summary"))
def test_grid_build_rejects_wrong_record_types(field_name):
    values = {
        "profile": TRTLLM_FULL_GRID_V1,
        "runtime_limits": _kimi_runtime_limits(),
        "coordinates": (TrtllmCoordinate("prefill", 1, 1, 0),),
        "admission_summary": TrtllmGridAdmissionSummary(1, 1, ()),
    }
    values[field_name] = object()

    with pytest.raises(TypeError, match=rf"TrtllmGridBuild\.{field_name} must be a"):
        TrtllmGridBuild(**values)


def test_grid_build_rejects_non_coordinate_records():
    with pytest.raises(TypeError, match="coordinates must contain only TrtllmCoordinate records"):
        TrtllmGridBuild(
            profile=TRTLLM_FULL_GRID_V1,
            runtime_limits=_kimi_runtime_limits(),
            coordinates=(object(),),
            admission_summary=TrtllmGridAdmissionSummary(1, 1, ()),
        )


def test_grid_build_rejects_duplicate_physical_coordinates():
    coordinate = TrtllmCoordinate("prefill", 1, 1, 0)

    with pytest.raises(ValueError, match="coordinates contain a duplicate physical coordinate"):
        TrtllmGridBuild(
            profile=TRTLLM_FULL_GRID_V1,
            runtime_limits=_kimi_runtime_limits(),
            coordinates=(coordinate, coordinate),
            admission_summary=TrtllmGridAdmissionSummary(2, 2, ()),
        )


def test_grid_build_rejects_an_admitted_count_coordinate_mismatch():
    with pytest.raises(ValueError, match="admitted_count must equal the coordinate count"):
        TrtllmGridBuild(
            profile=TRTLLM_FULL_GRID_V1,
            runtime_limits=_kimi_runtime_limits(),
            coordinates=(TrtllmCoordinate("prefill", 1, 1, 0),),
            admission_summary=TrtllmGridAdmissionSummary(2, 2, ()),
        )


def test_prefill_runs_largest_first_before_ascending_decode():
    grid = build_trtllm_full_grid(_kimi_runtime_limits())
    first_decode = next(index for index, point in enumerate(grid) if point.workload_kind == "decode")
    prefill = grid[:first_decode]
    decode = grid[first_decode:]
    prefill_keys = [(point.total_prefill_tokens, point.batch_size, point.total_kv_read_tokens) for point in prefill]
    decode_keys = [(point.batch_size, point.total_kv_read_tokens) for point in decode]

    assert prefill_keys == sorted(prefill_keys, reverse=True)
    assert decode_keys == sorted(decode_keys)
    assert (prefill[0], prefill[-1], decode[0], decode[-1]) == (
        TrtllmCoordinate("prefill", 128, 8_192, 2_307_232),
        TrtllmCoordinate("prefill", 1, 1, 0),
        TrtllmCoordinate("decode", 1, 0, 1),
        TrtllmCoordinate("decode", 128, 0, 2_318_117),
    )


def test_kimi_prefill_sites_cover_power_of_two_new_token_and_batch_axes():
    prefill = [point for point in build_trtllm_full_grid(_kimi_runtime_limits()) if point.workload_kind == "prefill"]
    sites = {(point.total_prefill_tokens, point.batch_size) for point in prefill}

    assert len(sites) == 84
    assert {batch for total, batch in sites if total == 1} == {1}
    assert {batch for total, batch in sites if total == 64} == {1, 2, 4, 8, 16, 32, 64}
    assert {batch for total, batch in sites if total == 8_192} == {
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
    }


def test_kimi_prefill_kv_axis_is_block_aligned_with_exact_capacity_anchors():
    prefill = [point for point in build_trtllm_full_grid(_kimi_runtime_limits()) if point.workload_kind == "prefill"]

    def kv_tokens_at(total_new_tokens: int, batch_size: int) -> list[int]:
        return sorted(
            point.total_kv_read_tokens
            for point in prefill
            if point.total_prefill_tokens == total_new_tokens and point.batch_size == batch_size
        )

    assert all(point.total_kv_read_tokens % 32 == 0 for point in prefill)
    assert kv_tokens_at(1, 1) == [
        block_count * 32 for block_count in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1_024, 2_048, 4_096, 8_191)
    ]
    assert kv_tokens_at(8_192, 128) == [
        block_count * 32
        for block_count in (0, 128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 72_101)
    ]


def test_kimi_decode_sites_straddle_cuda_graph_boundaries_and_keep_exact_maxima():
    decode = [point for point in build_trtllm_full_grid(_kimi_runtime_limits()) if point.workload_kind == "decode"]
    batch_sizes = {point.batch_size for point in decode}
    maxima = {
        batch_size: max(point.total_kv_read_tokens for point in decode if point.batch_size == batch_size)
        for batch_size in batch_sizes
    }

    assert batch_sizes == {1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65, 128}
    assert {batch_size: maxima[batch_size] for batch_size in (1, 9, 128)} == {
        1: 262_142,
        9: 2_319_254,
        128: 2_318_117,
    }
    assert [point.total_kv_read_tokens for point in decode if point.batch_size == 9] == [
        9,
        16,
        32,
        64,
        128,
        256,
        512,
        1_024,
        2_048,
        4_096,
        8_192,
        16_384,
        32_768,
        65_536,
        131_072,
        262_144,
        524_288,
        1_048_576,
        2_097_152,
        2_319_254,
    ]


def test_kimi_grid_is_unique_and_has_a_deterministic_manifest_identity():
    limits = _kimi_runtime_limits()
    first_grid = build_trtllm_full_grid(limits)
    second_grid = build_trtllm_full_grid(limits)

    manifest = TrtllmManifest.build(
        campaign_id="kimi-k2.5-tep8-trtllm-full-grid-v1",
        timing_rank_count=8,
        runtime_limits=limits,
        coordinates=first_grid,
    )
    rebuilt_manifest = TrtllmManifest.build(
        campaign_id="kimi-k2.5-tep8-trtllm-full-grid-v1",
        timing_rank_count=8,
        runtime_limits=limits,
        coordinates=second_grid,
    )

    assert first_grid == second_grid == manifest.coordinates
    assert len({point.point_id for point in first_grid}) == len(first_grid)
    assert manifest.sha256 == rebuilt_manifest.sha256
    assert manifest.sha256 == "44870e33f919566b2895a665e84bcb672e9a9d04f78e98be63fc789190621763"


def test_non_power_engine_limits_are_exact_grid_endpoints():
    limits = TrtllmRuntimeLimits(
        max_seq_len=101,
        max_num_requests=6,
        max_batch_size=6,
        max_num_tokens=10,
        kv_cache_max_num_blocks=1_000,
        kv_cache_tokens_per_block=8,
        decode_cuda_graph_batch_sizes=(1, 2, 4),
    )
    grid = build_trtllm_full_grid(limits)
    prefill_sites = {
        (point.total_prefill_tokens, point.batch_size) for point in grid if point.workload_kind == "prefill"
    }
    decode_batch_sizes = {point.batch_size for point in grid if point.workload_kind == "decode"}

    assert {total for total, _batch in prefill_sites} == {1, 2, 4, 8, 10}
    assert {batch for total, batch in prefill_sites if total == 10} == {1, 2, 4, 6}
    assert decode_batch_sizes == {1, 2, 3, 4, 5, 6}


def test_prefill_keeps_the_one_block_per_request_prefix_endpoint():
    limits = TrtllmRuntimeLimits(
        max_seq_len=3,
        max_num_requests=5,
        max_batch_size=5,
        max_num_tokens=5,
        kv_cache_max_num_blocks=100,
        kv_cache_tokens_per_block=1,
        decode_cuda_graph_batch_sizes=(1, 2, 4),
    )
    prefix_totals = sorted(
        point.total_kv_read_tokens
        for point in build_trtllm_full_grid(limits)
        if point.workload_kind == "prefill" and point.total_prefill_tokens == 5 and point.batch_size == 5
    )

    assert prefix_totals == [0, 5]
