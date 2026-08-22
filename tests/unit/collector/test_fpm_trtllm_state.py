# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import stat as stat_module
from dataclasses import FrozenInstanceError, replace

import pytest

from collector.fpm_forward.trtllm_state import (
    TRTLLM_COORDINATE_SYSTEM,
    TRTLLM_MEASUREMENT_POLICY,
    TrtllmAcceptedMeasurement,
    TrtllmCoordinate,
    TrtllmLedger,
    TrtllmManifest,
    TrtllmRuntimeLimits,
)

pytestmark = pytest.mark.unit

REQUIRED_IBS_FIELDS = (
    "numContextRequests",
    "numCtxTokens",
    "numCtxKvTokens",
    "numGenRequests",
    "numGenKvTokens",
    "numQueuedContextRequests",
    "numQueuedCtxTokens",
    "numQueuedGenRequests",
    "numQueuedGenKvTokens",
    "numPausedRequests",
    "numPausedKvTokens",
)
QUEUE_OR_PAUSE_IBS_FIELDS = (
    "numQueuedContextRequests",
    "numQueuedCtxTokens",
    "numQueuedGenRequests",
    "numQueuedGenKvTokens",
    "numPausedRequests",
    "numPausedKvTokens",
)


def _runtime_limits() -> TrtllmRuntimeLimits:
    return TrtllmRuntimeLimits(
        max_seq_len=262_144,
        max_num_requests=128,
        max_batch_size=128,
        max_num_tokens=8_192,
        kv_cache_max_num_blocks=72_485,
        kv_cache_tokens_per_block=32,
        decode_cuda_graph_batch_sizes=(1, 2, 4, 8, 16, 32, 64, 128),
    )


def _prefill_coordinate() -> TrtllmCoordinate:
    return TrtllmCoordinate(
        workload_kind="prefill",
        batch_size=2,
        total_prefill_tokens=64,
        total_kv_read_tokens=0,
    )


def _manifest(*coordinates: TrtllmCoordinate) -> TrtllmManifest:
    return TrtllmManifest.build(
        campaign_id="campaign",
        timing_rank_count=8,
        runtime_limits=_runtime_limits(),
        coordinates=coordinates or (_prefill_coordinate(),),
    )


def _prefill_ibs() -> dict[str, int | float]:
    return {
        "numScheduledRequests": 2,
        "numContextRequests": 2,
        "numCtxTokens": 64,
        "numCtxKvTokens": 0,
        "numGenRequests": 0,
        "numGenKvTokens": 0,
        "numQueuedContextRequests": 0,
        "numQueuedCtxTokens": 0,
        "numQueuedGenRequests": 0,
        "numQueuedGenKvTokens": 0,
        "numPausedRequests": 0,
        "numPausedKvTokens": 0,
        "microBatchId": 0,
        "avgNumDecodedTokensPerIter": 0.0,
    }


def _rank_stats(
    *,
    times: tuple[float, ...] | None = None,
    iteration: int = 42,
    ibs: dict[str, int | float] | None = None,
) -> list[dict[str, object]]:
    if times is None:
        times = (4.0, 4.1, 4.2, 4.3, 4.4, 5.5, 4.6, 4.7)
    return [
        {
            "rank": rank,
            "iter": iteration,
            "gpuForwardTimeMS": times[rank],
            "inflightBatchingStats": dict(ibs if ibs is not None else _prefill_ibs()),
        }
        for rank in reversed(range(8))
    ]


def test_manifest_build_has_stable_content_addressed_identity():
    coordinate = TrtllmCoordinate(
        workload_kind="prefill",
        batch_size=2,
        total_prefill_tokens=64,
        total_kv_read_tokens=0,
    )

    manifest = TrtllmManifest.build(
        campaign_id="kimi-k2.5-tep8",
        timing_rank_count=8,
        runtime_limits=_runtime_limits(),
        coordinates=(coordinate,),
    )

    assert TRTLLM_COORDINATE_SYSTEM == "iteration_totals_balanced_v1"
    assert TRTLLM_MEASUREMENT_POLICY == "trtllm_gpu_forward_rank_max_single_sample_v1"
    assert coordinate.point_id == "045c04bd3b85a608ea53d5f0993b6d22da43585837d3515bcf55a746f6d360cd"
    assert manifest.sha256 == "42075e015ba90f5bed026bb9bceb94f74956b1599b385bb4458450f27b0ad404"
    assert manifest.to_dict() == {
        "schema_name": "aic_trtllm_fpm_coordinate_manifest",
        "schema_version": 1,
        "campaign_id": "kimi-k2.5-tep8",
        "timing_rank_count": 8,
        "coordinate_system": TRTLLM_COORDINATE_SYSTEM,
        "measurement_policy": TRTLLM_MEASUREMENT_POLICY,
        "runtime_limits": {
            "max_seq_len": 262_144,
            "max_num_requests": 128,
            "max_batch_size": 128,
            "max_num_tokens": 8_192,
            "kv_cache_max_num_blocks": 72_485,
            "kv_cache_tokens_per_block": 32,
            "decode_cuda_graph_batch_sizes": [1, 2, 4, 8, 16, 32, 64, 128],
        },
        "coordinates": [
            {
                "workload_kind": "prefill",
                "batch_size": 2,
                "total_prefill_tokens": 64,
                "total_kv_read_tokens": 0,
                "point_id": coordinate.point_id,
            }
        ],
        "sha256": manifest.sha256,
    }


def test_manifest_identity_includes_order_and_detaches_the_input_sequence():
    prefill = TrtllmCoordinate(
        workload_kind="prefill",
        batch_size=2,
        total_prefill_tokens=64,
        total_kv_read_tokens=0,
    )
    decode = TrtllmCoordinate(
        workload_kind="decode",
        batch_size=2,
        total_prefill_tokens=0,
        total_kv_read_tokens=64,
    )
    source = [prefill, decode]
    manifest = TrtllmManifest.build(
        campaign_id="campaign",
        timing_rank_count=8,
        runtime_limits=_runtime_limits(),
        coordinates=source,
    )
    reversed_manifest = TrtllmManifest.build(
        campaign_id="campaign",
        timing_rank_count=8,
        runtime_limits=_runtime_limits(),
        coordinates=list(reversed(source)),
    )

    source.reverse()

    assert manifest.sha256 != reversed_manifest.sha256
    assert manifest.coordinates == (prefill, decode)


def test_manifest_to_dict_returns_a_detached_document():
    manifest = _manifest()
    original = manifest.to_dict()
    detached = manifest.to_dict()

    detached["runtime_limits"]["decode_cuda_graph_batch_sizes"].append(256)
    detached["coordinates"][0]["total_prefill_tokens"] = 1

    assert manifest.to_dict() == original


@pytest.mark.parametrize(
    "case",
    [
        "campaign",
        "rank_count",
        "max_seq_len",
        "max_num_requests",
        "max_batch_size",
        "max_num_tokens",
        "kv_cache_max_num_blocks",
        "kv_cache_tokens_per_block",
        "decode_cuda_graph_batch_sizes",
        "coordinate",
    ],
)
def test_manifest_digest_changes_with_every_frozen_identity_dimension(case):
    baseline = _manifest()
    campaign_id = "campaign"
    timing_rank_count = 8
    runtime_limits = _runtime_limits()
    coordinate = _prefill_coordinate()
    if case == "campaign":
        campaign_id = "campaign-2"
    elif case == "rank_count":
        timing_rank_count = 9
    elif case == "coordinate":
        coordinate = replace(coordinate, total_prefill_tokens=65)
    elif case == "decode_cuda_graph_batch_sizes":
        runtime_limits = replace(
            runtime_limits,
            decode_cuda_graph_batch_sizes=(1, 2, 4, 8, 16, 32, 64, 127, 128),
        )
    else:
        runtime_limits = replace(
            runtime_limits,
            **{case: getattr(runtime_limits, case) + 1},
        )

    changed = TrtllmManifest.build(
        campaign_id=campaign_id,
        timing_rank_count=timing_rank_count,
        runtime_limits=runtime_limits,
        coordinates=(coordinate,),
    )

    assert changed.sha256 != baseline.sha256


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("batch_size", True),
        ("batch_size", 2.0),
        ("batch_size", "2"),
        ("total_prefill_tokens", True),
        ("total_prefill_tokens", 64.0),
        ("total_prefill_tokens", "64"),
        ("total_kv_read_tokens", True),
        ("total_kv_read_tokens", 64.0),
        ("total_kv_read_tokens", "64"),
    ],
)
def test_coordinate_dimensions_are_strict_integers(field_name, bad_value):
    with pytest.raises(TypeError, match=field_name):
        replace(_prefill_coordinate(), **{field_name: bad_value})


@pytest.mark.parametrize(
    "field_name",
    [
        "max_seq_len",
        "max_num_requests",
        "max_batch_size",
        "max_num_tokens",
        "kv_cache_max_num_blocks",
        "kv_cache_tokens_per_block",
    ],
)
@pytest.mark.parametrize("bad_value", [True, 8.0, "8"])
def test_runtime_limit_dimensions_are_strict_integers(field_name, bad_value):
    with pytest.raises(TypeError, match=field_name):
        replace(_runtime_limits(), **{field_name: bad_value})


@pytest.mark.parametrize(
    "bad_sizes",
    [
        [1, 2],
        (1, True),
        (1, 2.0),
        (1, "2"),
    ],
)
def test_decode_cuda_graph_batch_sizes_are_an_immutable_strict_integer_tuple(bad_sizes):
    with pytest.raises(TypeError, match="decode_cuda_graph_batch_sizes"):
        replace(_runtime_limits(), decode_cuda_graph_batch_sizes=bad_sizes)


@pytest.mark.parametrize(
    ("coordinate_kwargs", "message"),
    [
        (
            {"workload_kind": "prefill", "batch_size": 0, "total_prefill_tokens": 0, "total_kv_read_tokens": 0},
            "batch_size",
        ),
        (
            {"workload_kind": "prefill", "batch_size": 2, "total_prefill_tokens": -1, "total_kv_read_tokens": 0},
            "total_prefill_tokens",
        ),
        (
            {"workload_kind": "prefill", "batch_size": 2, "total_prefill_tokens": 2, "total_kv_read_tokens": -1},
            "total_kv_read_tokens",
        ),
        (
            {"workload_kind": "prefill", "batch_size": 2, "total_prefill_tokens": 1, "total_kv_read_tokens": 0},
            "at least batch_size",
        ),
        (
            {"workload_kind": "prefill", "batch_size": 2, "total_prefill_tokens": 2, "total_kv_read_tokens": 1},
            "zero or at least batch_size",
        ),
        (
            {"workload_kind": "decode", "batch_size": 2, "total_prefill_tokens": 1, "total_kv_read_tokens": 64},
            "zero total_prefill_tokens",
        ),
        (
            {"workload_kind": "decode", "batch_size": 2, "total_prefill_tokens": 0, "total_kv_read_tokens": 1},
            "at least batch_size",
        ),
    ],
)
def test_coordinate_phase_rules_reject_impossible_physical_totals(coordinate_kwargs, message):
    with pytest.raises(ValueError, match=message):
        TrtllmCoordinate(**coordinate_kwargs)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_seq_len": 0},
        {"max_num_requests": 0},
        {"max_batch_size": 0},
        {"max_num_tokens": 0},
        {"kv_cache_max_num_blocks": 0},
        {"kv_cache_tokens_per_block": 0},
        {"decode_cuda_graph_batch_sizes": ()},
        {"decode_cuda_graph_batch_sizes": (0, 1)},
        {"decode_cuda_graph_batch_sizes": (1, 1)},
        {"decode_cuda_graph_batch_sizes": (2, 1)},
        {"decode_cuda_graph_batch_sizes": (1, 129)},
    ],
)
def test_runtime_limits_reject_nonpositive_or_invalid_graph_boundaries(overrides):
    with pytest.raises(ValueError):
        replace(_runtime_limits(), **overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"campaign_id": ""},
        {"campaign_id": 7},
        {"timing_rank_count": 0},
        {"timing_rank_count": True},
        {"timing_rank_count": 8.0},
        {"timing_rank_count": "8"},
        {"runtime_limits": _runtime_limits().to_dict()},
        {"coordinates": ()},
        {"coordinates": ({"workload_kind": "prefill"},)},
    ],
)
def test_manifest_build_rejects_invalid_identity_types(overrides):
    arguments = {
        "campaign_id": "campaign",
        "timing_rank_count": 8,
        "runtime_limits": _runtime_limits(),
        "coordinates": (_prefill_coordinate(),),
    }
    arguments.update(overrides)

    with pytest.raises((TypeError, ValueError)):
        TrtllmManifest.build(**arguments)


def test_manifest_build_rejects_duplicate_physical_coordinates():
    coordinate = _prefill_coordinate()

    with pytest.raises(ValueError, match="duplicate physical coordinate"):
        TrtllmManifest.build(
            campaign_id="campaign",
            timing_rank_count=8,
            runtime_limits=_runtime_limits(),
            coordinates=(coordinate, coordinate),
        )


@pytest.mark.parametrize(
    "coordinate",
    [
        TrtllmCoordinate("prefill", 5, 5, 0),
        TrtllmCoordinate("prefill", 2, 17, 0),
        TrtllmCoordinate("prefill", 2, 2, 257),
        TrtllmCoordinate("prefill", 2, 16, 120),
        TrtllmCoordinate("decode", 2, 0, 129),
    ],
)
def test_manifest_build_rejects_coordinates_outside_direct_runtime_bounds(coordinate):
    limits = TrtllmRuntimeLimits(
        max_seq_len=64,
        max_num_requests=4,
        max_batch_size=4,
        max_num_tokens=16,
        kv_cache_max_num_blocks=32,
        kv_cache_tokens_per_block=8,
        decode_cuda_graph_batch_sizes=(1, 2, 4),
    )

    with pytest.raises(ValueError, match="runtime limit"):
        TrtllmManifest.build(
            campaign_id="campaign",
            timing_rank_count=8,
            runtime_limits=limits,
            coordinates=(coordinate,),
        )


def test_manifest_rejects_prefill_that_exceeds_kv_blocks_after_per_request_rounding():
    coordinate = TrtllmCoordinate("prefill", 2, 8, 0)
    limits = TrtllmRuntimeLimits(
        max_seq_len=64,
        max_num_requests=2,
        max_batch_size=2,
        max_num_tokens=8,
        kv_cache_max_num_blocks=1,
        kv_cache_tokens_per_block=8,
        decode_cuda_graph_batch_sizes=(1, 2),
    )

    with pytest.raises(ValueError, match="KV-cache blocks"):
        TrtllmManifest.build(
            campaign_id="campaign",
            timing_rank_count=8,
            runtime_limits=limits,
            coordinates=(coordinate,),
        )


@pytest.mark.parametrize(
    ("coordinate", "max_num_tokens"),
    [
        (TrtllmCoordinate("prefill", 1, 2**53 + 1, 0), 2**53 + 1),
        (TrtllmCoordinate("decode", 1, 0, 2**53), 1),
    ],
    ids=("prefill", "decode"),
)
def test_manifest_uses_exact_integer_ceiling_for_large_kv_block_counts(coordinate, max_num_tokens):
    limits = TrtllmRuntimeLimits(
        max_seq_len=2**53 + 2,
        max_num_requests=1,
        max_batch_size=1,
        max_num_tokens=max_num_tokens,
        kv_cache_max_num_blocks=2**52,
        kv_cache_tokens_per_block=2,
        decode_cuda_graph_batch_sizes=(1,),
    )

    with pytest.raises(ValueError, match="KV-cache blocks"):
        TrtllmManifest.build(
            campaign_id="campaign",
            timing_rank_count=8,
            runtime_limits=limits,
            coordinates=(coordinate,),
        )


def test_manifest_accepts_prefill_at_exact_scheduler_sequence_and_kv_block_boundaries():
    coordinate = TrtllmCoordinate("prefill", 2, 16, 0)
    limits = TrtllmRuntimeLimits(
        max_seq_len=9,
        max_num_requests=2,
        max_batch_size=2,
        max_num_tokens=16,
        kv_cache_max_num_blocks=2,
        kv_cache_tokens_per_block=8,
        decode_cuda_graph_batch_sizes=(1, 2),
    )

    manifest = TrtllmManifest.build(
        campaign_id="campaign",
        timing_rank_count=8,
        runtime_limits=limits,
        coordinates=(coordinate,),
    )

    assert manifest.coordinates == (coordinate,)


def test_manifest_requires_prefill_kv_totals_to_align_to_runtime_blocks():
    coordinate = TrtllmCoordinate("prefill", 2, 2, 10)
    limits = TrtllmRuntimeLimits(
        max_seq_len=64,
        max_num_requests=2,
        max_batch_size=2,
        max_num_tokens=2,
        kv_cache_max_num_blocks=10,
        kv_cache_tokens_per_block=8,
        decode_cuda_graph_batch_sizes=(1, 2),
    )

    with pytest.raises(ValueError, match="whole KV-cache blocks"):
        TrtllmManifest.build(
            campaign_id="campaign",
            timing_rank_count=8,
            runtime_limits=limits,
            coordinates=(coordinate,),
        )


def test_manifest_requires_one_prefill_kv_block_per_request_when_kv_is_present():
    coordinate = TrtllmCoordinate("prefill", 2, 2, 8)
    limits = TrtllmRuntimeLimits(
        max_seq_len=64,
        max_num_requests=2,
        max_batch_size=2,
        max_num_tokens=2,
        kv_cache_max_num_blocks=10,
        kv_cache_tokens_per_block=8,
        decode_cuda_graph_batch_sizes=(1, 2),
    )

    with pytest.raises(ValueError, match="one KV-cache block per request"):
        TrtllmManifest.build(
            campaign_id="campaign",
            timing_rank_count=8,
            runtime_limits=limits,
            coordinates=(coordinate,),
        )


def test_manifest_rejects_balanced_prefill_request_that_cannot_fit_one_output_token():
    coordinate = TrtllmCoordinate("prefill", 2, 19, 0)
    limits = TrtllmRuntimeLimits(
        max_seq_len=10,
        max_num_requests=2,
        max_batch_size=2,
        max_num_tokens=19,
        kv_cache_max_num_blocks=10,
        kv_cache_tokens_per_block=8,
        decode_cuda_graph_batch_sizes=(1, 2),
    )

    with pytest.raises(ValueError, match="prefill request sequence length"):
        TrtllmManifest.build(
            campaign_id="campaign",
            timing_rank_count=8,
            runtime_limits=limits,
            coordinates=(coordinate,),
        )


def test_manifest_balances_prefill_kv_as_whole_blocks_before_checking_each_sequence():
    coordinate = TrtllmCoordinate("prefill", 2, 3, 24)
    limits = TrtllmRuntimeLimits(
        max_seq_len=18,
        max_num_requests=2,
        max_batch_size=2,
        max_num_tokens=3,
        kv_cache_max_num_blocks=10,
        kv_cache_tokens_per_block=8,
        decode_cuda_graph_batch_sizes=(1, 2),
    )

    with pytest.raises(ValueError, match="prefill request sequence length"):
        TrtllmManifest.build(
            campaign_id="campaign",
            timing_rank_count=8,
            runtime_limits=limits,
            coordinates=(coordinate,),
        )


def test_manifest_rejects_balanced_decode_context_that_cannot_fit_two_runner_tokens():
    coordinate = TrtllmCoordinate("decode", 2, 0, 17)
    limits = TrtllmRuntimeLimits(
        max_seq_len=10,
        max_num_requests=2,
        max_batch_size=2,
        max_num_tokens=2,
        kv_cache_max_num_blocks=10,
        kv_cache_tokens_per_block=8,
        decode_cuda_graph_batch_sizes=(1, 2),
    )

    with pytest.raises(ValueError, match="decode request sequence length"):
        TrtllmManifest.build(
            campaign_id="campaign",
            timing_rank_count=8,
            runtime_limits=limits,
            coordinates=(coordinate,),
        )


def test_manifest_counts_the_measured_decode_write_when_checking_kv_blocks():
    coordinate = TrtllmCoordinate("decode", 1, 0, 8)
    limits = TrtllmRuntimeLimits(
        max_seq_len=10,
        max_num_requests=1,
        max_batch_size=1,
        max_num_tokens=1,
        kv_cache_max_num_blocks=1,
        kv_cache_tokens_per_block=8,
        decode_cuda_graph_batch_sizes=(1,),
    )

    with pytest.raises(ValueError, match="KV-cache blocks"):
        TrtllmManifest.build(
            campaign_id="campaign",
            timing_rank_count=8,
            runtime_limits=limits,
            coordinates=(coordinate,),
        )


def test_manifest_accepts_decode_at_exact_sequence_and_measured_kv_block_boundaries():
    coordinate = TrtllmCoordinate("decode", 1, 0, 7)
    limits = TrtllmRuntimeLimits(
        max_seq_len=9,
        max_num_requests=1,
        max_batch_size=1,
        max_num_tokens=1,
        kv_cache_max_num_blocks=1,
        kv_cache_tokens_per_block=8,
        decode_cuda_graph_batch_sizes=(1,),
    )

    manifest = TrtllmManifest.build(
        campaign_id="campaign",
        timing_rank_count=8,
        runtime_limits=limits,
        coordinates=(coordinate,),
    )

    assert manifest.coordinates == (coordinate,)


def test_manifest_load_round_trips_the_verified_public_document(tmp_path):
    manifest = _manifest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest.to_dict()))

    loaded = TrtllmManifest.load(path)

    assert loaded == manifest
    assert loaded.to_dict() == manifest.to_dict()


def test_manifest_load_rechecks_runtime_feasibility_before_accepting_frozen_coordinates(tmp_path):
    payload = _manifest().to_dict()
    payload["runtime_limits"]["max_seq_len"] = 32
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="prefill request sequence length"):
        TrtllmManifest.load(path)


@pytest.mark.parametrize("case", ["free_blocks", "point_id", "policy", "sha256", "unknown_key"])
def test_manifest_load_rejects_tampering_and_never_accepts_free_blocks_as_identity(tmp_path, case):
    manifest = _manifest()
    payload = manifest.to_dict()
    if case == "free_blocks":
        payload["runtime_limits"]["kv_cache_free_num_blocks"] = 70_000
    elif case == "point_id":
        payload["coordinates"][0]["point_id"] = "0" * 64
    elif case == "policy":
        payload["measurement_policy"] = "different-policy"
    elif case == "sha256":
        payload["sha256"] = "0" * 64
    else:
        payload["unexpected"] = True
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))

    with pytest.raises((TypeError, ValueError)):
        TrtllmManifest.load(path)


def test_new_ledger_persists_the_manifest_and_reports_every_coordinate_pending(tmp_path):
    decode = TrtllmCoordinate("decode", 2, 0, 64)
    manifest = _manifest(_prefill_coordinate(), decode)

    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=manifest,
        current_runtime_limits=_runtime_limits(),
    )

    assert ledger.pending_coordinates == manifest.coordinates
    assert ledger.accepted_measurements == ()
    assert json.loads((tmp_path / "ledger" / "manifest.json").read_text()) == manifest.to_dict()


def test_ledger_exposes_its_frozen_manifest_without_a_mutable_copy(tmp_path):
    manifest = _manifest()
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=manifest,
        current_runtime_limits=_runtime_limits(),
    )

    assert ledger.manifest is manifest


def test_new_ledger_durably_persists_the_manifest_before_open_returns(monkeypatch, tmp_path):
    events = []
    real_fsync = os.fsync
    real_replace = os.replace
    real_close = os.close

    def record_fsync(file_descriptor):
        descriptor_kind = "directory" if stat_module.S_ISDIR(os.fstat(file_descriptor).st_mode) else "file"
        events.append(f"{descriptor_kind} fsync")
        real_fsync(file_descriptor)

    def record_replace(source, destination):
        events.append("replace")
        real_replace(source, destination)

    def record_close(file_descriptor):
        if stat_module.S_ISDIR(os.fstat(file_descriptor).st_mode):
            events.append("directory close")
        real_close(file_descriptor)

    monkeypatch.setattr(os, "fsync", record_fsync)
    monkeypatch.setattr(os, "replace", record_replace)
    monkeypatch.setattr(os, "close", record_close)

    TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=_manifest(),
        current_runtime_limits=_runtime_limits(),
    )

    assert events == ["file fsync", "replace", "directory fsync", "directory close"]


def test_accept_aggregates_exactly_eight_ranks_with_the_maximum_gpu_forward_time(tmp_path):
    coordinate = _prefill_coordinate()
    manifest = _manifest(coordinate)
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=manifest,
        current_runtime_limits=_runtime_limits(),
    )

    measurement = ledger.accept(
        coordinate,
        attempt_id="attempt-1",
        rank_stats=_rank_stats(),
    )

    assert isinstance(measurement, TrtllmAcceptedMeasurement)
    assert measurement.manifest_sha256 == manifest.sha256
    assert measurement.campaign_id == "campaign"
    assert measurement.point_id == coordinate.point_id
    assert measurement.attempt_id == "attempt-1"
    assert measurement.iteration_id == 42
    assert measurement.coordinate == coordinate
    assert measurement.inflight_batching_stats == _prefill_ibs()
    assert measurement.rank_times_ms == (
        (0, 4.0),
        (1, 4.1),
        (2, 4.2),
        (3, 4.3),
        (4, 4.4),
        (5, 5.5),
        (6, 4.6),
        (7, 4.7),
    )
    assert measurement.latency_ms == 5.5
    assert measurement.timing_source == "trtllm_iteration_stats"
    assert measurement.measurement_policy == TRTLLM_MEASUREMENT_POLICY
    assert ledger.pending_coordinates == ()
    assert ledger.accepted_measurements == (measurement,)

    accepted_path = tmp_path / "ledger" / "accepted" / f"{coordinate.point_id}.json"
    accepted_payload = json.loads(accepted_path.read_text())
    assert accepted_payload == {
        "schema_name": "aic_trtllm_fpm_accepted_measurement",
        "schema_version": 1,
        "manifest_sha256": manifest.sha256,
        "campaign_id": "campaign",
        "point_id": coordinate.point_id,
        "attempt_id": "attempt-1",
        "iteration_id": 42,
        "coordinate": coordinate.to_dict(),
        "inflight_batching_stats": _prefill_ibs(),
        "rank_times_ms": [{"rank": rank, "gpu_forward_time_ms": time} for rank, time in measurement.rank_times_ms],
        "latency_ms": 5.5,
        "timing_source": "trtllm_iteration_stats",
        "measurement_policy": TRTLLM_MEASUREMENT_POLICY,
    }


def test_accepted_measurement_is_deeply_immutable(tmp_path):
    coordinate = _prefill_coordinate()
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=_manifest(coordinate),
        current_runtime_limits=_runtime_limits(),
    )
    measurement = ledger.accept(coordinate, attempt_id="attempt-1", rank_stats=_rank_stats())

    detached_ibs = measurement.inflight_batching_stats
    detached_ibs["numCtxTokens"] = 1

    assert measurement.inflight_batching_stats == _prefill_ibs()
    with pytest.raises(FrozenInstanceError):
        measurement.latency_ms = 1.0


@pytest.mark.parametrize("case", ["missing", "duplicate", "out_of_range", "not_sequence", "row_not_mapping"])
def test_accept_rejects_any_input_that_is_not_exactly_ranks_zero_through_seven(tmp_path, case):
    coordinate = _prefill_coordinate()
    manifest = _manifest(coordinate)
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=manifest,
        current_runtime_limits=_runtime_limits(),
    )
    stats = _rank_stats()
    if case == "missing":
        stats.pop()
    elif case == "duplicate":
        stats[0]["rank"] = stats[1]["rank"]
    elif case == "out_of_range":
        stats[0]["rank"] = 8
    elif case == "not_sequence":
        stats = iter(stats)
    else:
        stats[0] = "rank-seven"

    with pytest.raises((TypeError, ValueError)):
        ledger.accept(coordinate, attempt_id="attempt-1", rank_stats=stats)

    assert ledger.pending_coordinates == (coordinate,)
    assert ledger.accepted_measurements == ()
    assert list((tmp_path / "ledger" / "accepted").glob("*.json")) == []


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("rank", True),
        ("rank", 7.0),
        ("rank", "7"),
        ("iter", True),
        ("iter", 42.0),
        ("iter", "42"),
        ("iter", 43),
    ],
)
def test_accept_requires_strict_rank_and_one_identical_iteration_id(tmp_path, field_name, bad_value):
    coordinate = _prefill_coordinate()
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=_manifest(coordinate),
        current_runtime_limits=_runtime_limits(),
    )
    stats = _rank_stats()
    stats[0][field_name] = bad_value

    with pytest.raises((TypeError, ValueError)):
        ledger.accept(coordinate, attempt_id="attempt-1", rank_stats=stats)

    assert ledger.pending_coordinates == (coordinate,)


def test_accept_rejects_a_negative_iteration_id(tmp_path):
    coordinate = _prefill_coordinate()
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=_manifest(coordinate),
        current_runtime_limits=_runtime_limits(),
    )

    with pytest.raises(ValueError, match="non-negative"):
        ledger.accept(
            coordinate,
            attempt_id="attempt-1",
            rank_stats=_rank_stats(iteration=-1),
        )

    assert ledger.pending_coordinates == (coordinate,)


@pytest.mark.parametrize("bad_time", [None, True, 0, -1.0, float("nan"), float("inf"), "4.0"])
def test_accept_requires_a_finite_positive_gpu_forward_time_on_every_rank(tmp_path, bad_time):
    coordinate = _prefill_coordinate()
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=_manifest(coordinate),
        current_runtime_limits=_runtime_limits(),
    )
    stats = _rank_stats()
    stats[0]["gpuForwardTimeMS"] = bad_time

    with pytest.raises((TypeError, ValueError), match="gpuForwardTimeMS"):
        ledger.accept(coordinate, attempt_id="attempt-1", rank_stats=stats)

    assert ledger.pending_coordinates == (coordinate,)


def test_accept_never_falls_back_to_iter_latency(tmp_path):
    coordinate = _prefill_coordinate()
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=_manifest(coordinate),
        current_runtime_limits=_runtime_limits(),
    )
    stats = _rank_stats()
    del stats[0]["gpuForwardTimeMS"]
    stats[0]["iterLatencyMS"] = 99.0

    with pytest.raises(TypeError, match="gpuForwardTimeMS"):
        ledger.accept(coordinate, attempt_id="attempt-1", rank_stats=stats)

    assert ledger.pending_coordinates == (coordinate,)


@pytest.mark.parametrize("field_name", REQUIRED_IBS_FIELDS)
def test_accept_requires_all_eleven_inflight_batching_fields(tmp_path, field_name):
    coordinate = _prefill_coordinate()
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=_manifest(coordinate),
        current_runtime_limits=_runtime_limits(),
    )
    stats = _rank_stats()
    del stats[0]["inflightBatchingStats"][field_name]

    with pytest.raises(TypeError, match=field_name):
        ledger.accept(coordinate, attempt_id="attempt-1", rank_stats=stats)

    assert ledger.pending_coordinates == (coordinate,)


@pytest.mark.parametrize("field_name", REQUIRED_IBS_FIELDS)
@pytest.mark.parametrize("bad_value", [True, 1.0, "1"])
def test_accept_requires_strict_integers_for_all_inflight_batching_fields(tmp_path, field_name, bad_value):
    coordinate = _prefill_coordinate()
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=_manifest(coordinate),
        current_runtime_limits=_runtime_limits(),
    )
    stats = _rank_stats()
    stats[0]["inflightBatchingStats"][field_name] = bad_value

    with pytest.raises(TypeError, match=field_name):
        ledger.accept(coordinate, attempt_id="attempt-1", rank_stats=stats)

    assert ledger.pending_coordinates == (coordinate,)


@pytest.mark.parametrize("field_name", QUEUE_OR_PAUSE_IBS_FIELDS)
def test_accept_rejects_every_queued_or_paused_work_indicator(tmp_path, field_name):
    coordinate = _prefill_coordinate()
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=_manifest(coordinate),
        current_runtime_limits=_runtime_limits(),
    )
    stats = _rank_stats()
    for stat in stats:
        stat["inflightBatchingStats"][field_name] = 1

    with pytest.raises(ValueError, match="requested coordinate"):
        ledger.accept(coordinate, attempt_id="attempt-1", rank_stats=stats)

    assert ledger.pending_coordinates == (coordinate,)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("numContextRequests", 1),
        ("numCtxTokens", 63),
        ("numCtxKvTokens", 2),
        ("numGenRequests", 1),
        ("numGenKvTokens", 1),
        ("numScheduledRequests", 1),
    ],
)
def test_accept_rejects_a_rank_identical_but_wrong_or_mixed_prefill_coordinate(tmp_path, field_name, bad_value):
    coordinate = _prefill_coordinate()
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=_manifest(coordinate),
        current_runtime_limits=_runtime_limits(),
    )
    stats = _rank_stats()
    for stat in stats:
        stat["inflightBatchingStats"][field_name] = bad_value

    with pytest.raises(ValueError, match="requested coordinate"):
        ledger.accept(coordinate, attempt_id="attempt-1", rank_stats=stats)

    assert ledger.pending_coordinates == (coordinate,)


def test_accept_requires_the_complete_inflight_batching_mapping_to_match_across_ranks(tmp_path):
    coordinate = _prefill_coordinate()
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=_manifest(coordinate),
        current_runtime_limits=_runtime_limits(),
    )
    stats = _rank_stats()
    stats[0]["inflightBatchingStats"]["microBatchId"] = 1

    with pytest.raises(ValueError, match="mappings differ"):
        ledger.accept(coordinate, attempt_id="attempt-1", rank_stats=stats)

    assert ledger.pending_coordinates == (coordinate,)


def test_accept_rejects_a_coordinate_not_frozen_in_the_manifest(tmp_path):
    coordinate = _prefill_coordinate()
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=_manifest(coordinate),
        current_runtime_limits=_runtime_limits(),
    )
    unknown = replace(coordinate, total_prefill_tokens=65)

    with pytest.raises(ValueError, match="not present in the frozen manifest"):
        ledger.accept(unknown, attempt_id="attempt-1", rank_stats=_rank_stats())

    assert ledger.pending_coordinates == (coordinate,)


@pytest.mark.parametrize("bad_attempt_id", [None, True, 1, "", "   "])
def test_accept_requires_a_nonempty_string_attempt_identity(tmp_path, bad_attempt_id):
    coordinate = _prefill_coordinate()
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=_manifest(coordinate),
        current_runtime_limits=_runtime_limits(),
    )

    with pytest.raises((TypeError, ValueError), match="attempt_id"):
        ledger.accept(coordinate, attempt_id=bad_attempt_id, rank_stats=_rank_stats())

    assert ledger.pending_coordinates == (coordinate,)


def test_accept_validates_a_pure_decode_coordinate(tmp_path):
    coordinate = TrtllmCoordinate("decode", 2, 0, 64)
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=_manifest(coordinate),
        current_runtime_limits=_runtime_limits(),
    )
    ibs = _prefill_ibs()
    ibs.update(
        numContextRequests=0,
        numCtxTokens=0,
        numGenRequests=2,
        numGenKvTokens=64,
    )

    measurement = ledger.accept(
        coordinate,
        attempt_id="attempt-1",
        rank_stats=_rank_stats(ibs=ibs),
    )

    assert measurement.coordinate == coordinate
    assert measurement.inflight_batching_stats == ibs


def test_reopen_derives_completed_and_pending_state_from_accepted_files(tmp_path):
    prefill = _prefill_coordinate()
    decode = TrtllmCoordinate("decode", 2, 0, 64)
    manifest = _manifest(prefill, decode)
    root = tmp_path / "ledger"
    first_ledger = TrtllmLedger.open(
        root,
        manifest=manifest,
        current_runtime_limits=_runtime_limits(),
    )
    accepted = first_ledger.accept(
        prefill,
        attempt_id="attempt-1",
        rank_stats=_rank_stats(),
    )

    reopened = TrtllmLedger.open(
        root,
        manifest=manifest,
        current_runtime_limits=_runtime_limits(),
    )

    assert reopened.accepted_measurements == (accepted,)
    assert reopened.pending_coordinates == (decode,)
    accepted_path = root / "accepted" / f"{prefill.point_id}.json"
    original_bytes = accepted_path.read_bytes()
    retried = reopened.accept(prefill, attempt_id="attempt-1", rank_stats=_rank_stats())
    assert retried == accepted
    assert accepted_path.read_bytes() == original_bytes


def test_reopen_ignores_an_uncommitted_same_directory_atomic_temp_file(tmp_path):
    manifest = _manifest()
    root = tmp_path / "ledger"
    TrtllmLedger.open(
        root,
        manifest=manifest,
        current_runtime_limits=_runtime_limits(),
    )
    (root / "accepted" / ".interrupted.json.abc.tmp").write_text("{")

    reopened = TrtllmLedger.open(
        root,
        manifest=manifest,
        current_runtime_limits=_runtime_limits(),
    )

    assert reopened.accepted_measurements == ()
    assert reopened.pending_coordinates == manifest.coordinates


def test_failed_accept_stays_pending_and_can_be_retried_with_valid_rank_stats(tmp_path):
    coordinate = _prefill_coordinate()
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=_manifest(coordinate),
        current_runtime_limits=_runtime_limits(),
    )

    with pytest.raises(ValueError, match="exactly 8 rows"):
        ledger.accept(coordinate, attempt_id="attempt-1", rank_stats=_rank_stats()[:-1])

    assert ledger.pending_coordinates == (coordinate,)
    accepted = ledger.accept(coordinate, attempt_id="attempt-1", rank_stats=_rank_stats())
    assert ledger.accepted_measurements == (accepted,)
    assert ledger.pending_coordinates == ()


def test_identical_retry_is_idempotent_and_conflicting_retries_never_overwrite(tmp_path):
    coordinate = _prefill_coordinate()
    root = tmp_path / "ledger"
    ledger = TrtllmLedger.open(
        root,
        manifest=_manifest(coordinate),
        current_runtime_limits=_runtime_limits(),
    )
    first = ledger.accept(coordinate, attempt_id="attempt-1", rank_stats=_rank_stats())
    accepted_path = root / "accepted" / f"{coordinate.point_id}.json"
    original_bytes = accepted_path.read_bytes()

    identical = ledger.accept(coordinate, attempt_id="attempt-1", rank_stats=_rank_stats())

    assert identical is first
    assert accepted_path.read_bytes() == original_bytes

    with pytest.raises(ValueError, match="conflicting accepted measurement"):
        ledger.accept(coordinate, attempt_id="attempt-2", rank_stats=_rank_stats())
    changed_times = (9.0, 4.1, 4.2, 4.3, 4.4, 5.5, 4.6, 4.7)
    with pytest.raises(ValueError, match="conflicting accepted measurement"):
        ledger.accept(
            coordinate,
            attempt_id="attempt-1",
            rank_stats=_rank_stats(times=changed_times),
        )

    assert accepted_path.read_bytes() == original_bytes
    assert ledger.accepted_measurements == (first,)


def test_ledger_open_rejects_changed_static_runtime_limits(tmp_path):
    manifest = _manifest()
    root = tmp_path / "ledger"
    TrtllmLedger.open(
        root,
        manifest=manifest,
        current_runtime_limits=_runtime_limits(),
    )
    changed_limits = replace(
        _runtime_limits(),
        kv_cache_max_num_blocks=_runtime_limits().kv_cache_max_num_blocks - 1,
    )

    with pytest.raises(ValueError, match="runtime limits"):
        TrtllmLedger.open(
            root,
            manifest=manifest,
            current_runtime_limits=changed_limits,
        )


def test_ledger_open_rejects_a_different_manifest_even_when_runtime_limits_match(tmp_path):
    root = tmp_path / "ledger"
    first_manifest = _manifest()
    TrtllmLedger.open(
        root,
        manifest=first_manifest,
        current_runtime_limits=_runtime_limits(),
    )
    different_manifest = TrtllmManifest.build(
        campaign_id="different-campaign",
        timing_rank_count=8,
        runtime_limits=_runtime_limits(),
        coordinates=first_manifest.coordinates,
    )

    with pytest.raises(ValueError, match="ledger manifest"):
        TrtllmLedger.open(
            root,
            manifest=different_manifest,
            current_runtime_limits=_runtime_limits(),
        )


def test_ledger_open_fails_closed_on_an_unknown_accepted_point(tmp_path):
    root = tmp_path / "ledger"
    manifest = _manifest()
    TrtllmLedger.open(
        root,
        manifest=manifest,
        current_runtime_limits=_runtime_limits(),
    )
    (root / "accepted" / f"{'0' * 64}.json").write_text("{}")

    with pytest.raises(ValueError, match="unknown point_id"):
        TrtllmLedger.open(
            root,
            manifest=manifest,
            current_runtime_limits=_runtime_limits(),
        )


def test_ledger_open_fails_closed_on_a_corrupt_accepted_file(tmp_path):
    coordinate = _prefill_coordinate()
    manifest = _manifest(coordinate)
    root = tmp_path / "ledger"
    ledger = TrtllmLedger.open(
        root,
        manifest=manifest,
        current_runtime_limits=_runtime_limits(),
    )
    ledger.accept(coordinate, attempt_id="attempt-1", rank_stats=_rank_stats())
    accepted_path = root / "accepted" / f"{coordinate.point_id}.json"
    accepted_path.write_text("{")

    with pytest.raises(ValueError, match="cannot load accepted"):
        TrtllmLedger.open(
            root,
            manifest=manifest,
            current_runtime_limits=_runtime_limits(),
        )


@pytest.mark.parametrize(
    "case",
    [
        "manifest",
        "campaign",
        "point",
        "coordinate",
        "inflight",
        "rank_order",
        "latency",
        "timing_source",
        "policy",
        "unknown_key",
    ],
)
def test_ledger_open_revalidates_every_accepted_record_contract(case, tmp_path):
    coordinate = _prefill_coordinate()
    manifest = _manifest(coordinate)
    root = tmp_path / "ledger"
    ledger = TrtllmLedger.open(
        root,
        manifest=manifest,
        current_runtime_limits=_runtime_limits(),
    )
    ledger.accept(coordinate, attempt_id="attempt-1", rank_stats=_rank_stats())
    accepted_path = root / "accepted" / f"{coordinate.point_id}.json"
    payload = json.loads(accepted_path.read_text())
    if case == "manifest":
        payload["manifest_sha256"] = "0" * 64
    elif case == "campaign":
        payload["campaign_id"] = "other"
    elif case == "point":
        payload["point_id"] = "0" * 64
    elif case == "coordinate":
        payload["coordinate"]["total_prefill_tokens"] = 65
    elif case == "inflight":
        payload["inflight_batching_stats"]["numQueuedContextRequests"] = 1
    elif case == "rank_order":
        payload["rank_times_ms"].reverse()
    elif case == "latency":
        payload["latency_ms"] = 4.0
    elif case == "timing_source":
        payload["timing_source"] = "gpuForwardTimeMS"
    elif case == "policy":
        payload["measurement_policy"] = "different-policy"
    else:
        payload["unexpected"] = True
    accepted_path.write_text(json.dumps(payload))

    with pytest.raises((TypeError, ValueError)):
        TrtllmLedger.open(
            root,
            manifest=manifest,
            current_runtime_limits=_runtime_limits(),
        )
