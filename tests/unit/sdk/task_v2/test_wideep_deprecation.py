# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecation surface for the legacy enable_wideep / deepep_moe flags.

The flags are behaviorally inert (large-EP participation is coverage-driven,
per tuple); this module pins the USER-FACING contract:

- warn once per key per process (DeprecationWarning + logger.warning);
- the parsed values are KEPT (the resolved exp_config.yaml is a captured
  regression artifact; ``enable_wideep -> moe_backend == "deepep_moe"``
  normalization is deliberately retained);
- results are identical with and without the flags (trtllm; the sglang
  deepep_moe EP-only ladder is the retained normalization, pinned elsewhere);
- the old disagg prefix-discipline ValueError for top-level enable_wideep is
  downgraded to the same warning (key kept, ignored).
"""

import warnings

import pytest

from aiconfigurator.sdk import task_v2
from aiconfigurator.sdk.perf_database import PerfDataNotAvailableError
from aiconfigurator.sdk.task_v2 import Task, _warn_large_ep_flag

pytestmark = pytest.mark.unit

MODEL = "deepseek-ai/DeepSeek-V3"


@pytest.fixture(autouse=True)
def _fresh_warned_keys():
    """Snapshot/clear/restore the process-global warn-once dedupe set.

    Warn-once is per process by design; each test needs a clean slate to
    observe its own first warning, and must not leave the set altered for
    tests that run after it (mirrors _one_shot_log_state in the coverage
    tests)."""
    before = set(task_v2._warned_large_ep_keys)
    task_v2._warned_large_ep_keys.clear()
    try:
        yield
    finally:
        task_v2._warned_large_ep_keys.clear()
        task_v2._warned_large_ep_keys.update(before)


def _deprecations(record) -> list[str]:
    """Our large-EP deprecation messages captured by catch_warnings(record=True)."""
    return [
        str(w.message)
        for w in record
        if w.category is DeprecationWarning and "deprecated and ignored" in str(w.message)
    ]


# ---------------------------------------------------------------------------
# Warn-once helper
# ---------------------------------------------------------------------------


def test_warn_helper_fires_exactly_once_per_key():
    with pytest.warns(DeprecationWarning, match="'enable_wideep' is deprecated and ignored"):
        _warn_large_ep_flag("enable_wideep")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _warn_large_ep_flag("enable_wideep")  # same key: silenced
        _warn_large_ep_flag("decode_enable_wideep")  # new key: fires
    msgs = _deprecations(record)
    assert len(msgs) == 1
    assert "'decode_enable_wideep'" in msgs[0]


# ---------------------------------------------------------------------------
# Constructor / from_yaml paths (both fire from __post_init__)
# ---------------------------------------------------------------------------


def test_agg_constructor_warns_once_and_keeps_flag():
    with pytest.warns(DeprecationWarning, match="'enable_wideep' is deprecated and ignored"):
        t = Task(serving_mode="agg", model_path=MODEL, system_name="h200_sxm", enable_wideep=True)
    # Values are KEPT: the field survives, and the deliberately-retained
    # normalization still spells the deprecated moe_backend on the resolved task.
    assert t.enable_wideep is True
    assert t.moe_backend == "deepep_moe"
    # Second construction with the same key: warn-once per process.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        Task(serving_mode="agg", model_path=MODEL, system_name="h200_sxm", enable_wideep=True)
    assert _deprecations(record) == []


def test_from_yaml_warns_and_keeps_flag():
    yaml_data = {
        "serving_mode": "agg",
        "model_path": MODEL,
        "system_name": "h200_sxm",
        "backend_name": "trtllm",
        "total_gpus": 8,
        "enable_wideep": True,
    }
    with pytest.warns(DeprecationWarning, match="'enable_wideep' is deprecated and ignored"):
        t = Task.from_yaml(yaml_data)
    assert t.enable_wideep is True
    assert t.to_dict()["enable_wideep"] is True  # the exp_config.yaml artifact keeps the key
    # from_yaml and the constructor share one warn point (__post_init__):
    # the key is already spent for this process.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        Task(serving_mode="agg", model_path=MODEL, system_name="h200_sxm", enable_wideep=True)
    assert _deprecations(record) == []


def test_disagg_role_flags_warn_per_key():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        Task(
            serving_mode="disagg",
            prefill_model_path=MODEL,
            prefill_system_name="h200_sxm",
            decode_model_path=MODEL,
            decode_system_name="h200_sxm",
            prefill_enable_wideep=True,
            decode_enable_wideep=True,
        )
    msgs = _deprecations(record)
    assert len(msgs) == 2  # one per key, not one per flag read
    assert any("'prefill_enable_wideep'" in m for m in msgs)
    assert any("'decode_enable_wideep'" in m for m in msgs)


def test_false_flags_never_warn():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        Task(serving_mode="agg", model_path=MODEL, system_name="h200_sxm", enable_wideep=False)
    assert _deprecations(record) == []


# ---------------------------------------------------------------------------
# moe_backend: deepep_moe warns; megamoe never does
# ---------------------------------------------------------------------------


def test_moe_backend_deepep_moe_warns():
    with pytest.warns(DeprecationWarning, match="'moe_backend=deepep_moe' is deprecated and ignored"):
        t = Task(
            serving_mode="agg",
            model_path=MODEL,
            system_name="h200_sxm",
            backend_name="sglang",
            moe_backend="deepep_moe",
        )
    assert t.moe_backend == "deepep_moe"  # value kept


def test_enable_wideep_normalization_does_not_double_warn_moe_backend():
    """enable_wideep=True still resolves moe_backend='deepep_moe' (retained
    normalization) but only the user-set key warns -- the normalized value
    must not trigger a second 'moe_backend=deepep_moe' warning."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        t = Task(serving_mode="agg", model_path=MODEL, system_name="h200_sxm", enable_wideep=True)
    assert t.moe_backend == "deepep_moe"
    msgs = _deprecations(record)
    assert len(msgs) == 1
    assert "'enable_wideep'" in msgs[0]


def test_megamoe_never_warns():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        Task(
            serving_mode="agg",
            model_path="deepseek-ai/DeepSeek-V4-Pro",
            system_name="b200_sxm",
            backend_name="sglang",
            moe_backend="megamoe",
        )
    assert _deprecations(record) == []


# ---------------------------------------------------------------------------
# Prefix-discipline downgrade: top-level enable_wideep in disagg
# ---------------------------------------------------------------------------


def test_disagg_top_level_enable_wideep_warns_instead_of_raising():
    """Was a prefix-discipline ValueError; now the same deprecation warning.
    The key is kept but ignored: disagg normalization reads only the role
    flags, so moe_backend stays None."""
    with pytest.warns(DeprecationWarning, match="'enable_wideep' is deprecated and ignored"):
        t = Task(
            serving_mode="disagg",
            prefill_model_path=MODEL,
            prefill_system_name="h200_sxm",
            decode_model_path=MODEL,
            decode_system_name="h200_sxm",
            enable_wideep=True,
        )
    assert t.enable_wideep is True  # kept
    assert t.moe_backend is None  # ignored (no role flag set)


def test_disagg_other_leakage_still_raises():
    """Only the deprecated flag is downgraded; real prefix-discipline leakage
    (e.g. model_path) still fails loudly."""
    with pytest.raises(ValueError, match="top-level worker fields"):
        Task(
            serving_mode="disagg",
            model_path=MODEL,
            prefill_model_path=MODEL,
            prefill_system_name="h200_sxm",
            decode_model_path=MODEL,
            decode_system_name="h200_sxm",
        )


# ---------------------------------------------------------------------------
# Flag vs flagless: identical search space and per-tuple modeling (trtllm)
# ---------------------------------------------------------------------------

_AGG_DIMS = ("num_gpu", "tp", "pp", "dp", "moe_tp", "moe_ep", "cp")


def _agg_task(**overrides) -> Task:
    return Task(
        serving_mode="agg",
        model_path=MODEL,
        system_name="h200_sxm",
        backend_name="trtllm",
        total_gpus=64,
        **overrides,
    )


def _model_config_outcome(task: Task, role: str, parallel: tuple) -> tuple:
    """Comparable result for tuples that may hit a strict cross-node gate."""
    try:
        model_config = task.build_model_config(role=role, parallel=parallel)
    except (PerfDataNotAvailableError, ValueError) as exc:
        return ("error", type(exc), str(exc))
    return ("ok", model_config.moe_comm_backend, model_config.moe_backend)


def test_flag_vs_flagless_agg_trtllm_identical():
    """Same Task inputs +- enable_wideep -> identical resolved candidate lists,
    identical sweep_agg_kwargs parallel_config_list, and identical per-tuple
    moe_comm_backend resolution (the flag has zero behavioral effect)."""
    flagged = _agg_task(enable_wideep=True)
    flagless = _agg_task()

    for dim in _AGG_DIMS:
        attr = f"agg_{dim}_candidates"
        assert getattr(flagged, attr) == getattr(flagless, attr), attr

    kw_flagged = flagged.sweep_agg_kwargs(database=None)
    kw_flagless = flagless.sweep_agg_kwargs(database=None)
    assert kw_flagged["parallel_config_list"] == kw_flagless["parallel_config_list"]

    for tup in kw_flagless["parallel_config_list"]:
        flagged_outcome = _model_config_outcome(flagged, "agg", tuple(tup))
        flagless_outcome = _model_config_outcome(flagless, "agg", tuple(tup))
        assert flagged_outcome == flagless_outcome, tup


def _disagg_task(**overrides) -> Task:
    return Task(
        serving_mode="disagg",
        prefill_model_path=MODEL,
        prefill_system_name="h200_sxm",
        prefill_backend_name="trtllm",
        decode_model_path=MODEL,
        decode_system_name="h200_sxm",
        decode_backend_name="trtllm",
        total_gpus=64,
        **overrides,
    )


def test_flag_vs_flagless_disagg_trtllm_identical():
    flagged = _disagg_task(prefill_enable_wideep=True, decode_enable_wideep=True)
    flagless = _disagg_task()

    for role in ("prefill", "decode"):
        for dim in _AGG_DIMS:
            attr = f"{role}_{dim}_candidates"
            assert getattr(flagged, attr) == getattr(flagless, attr), attr
    assert flagged.num_gpu_per_replica == flagless.num_gpu_per_replica
    assert flagged.max_gpu_per_replica == flagless.max_gpu_per_replica

    for role in ("prefill", "decode"):
        tuples_flagged = list(flagged.iter_parallel(role))
        tuples_flagless = list(flagless.iter_parallel(role))
        assert tuples_flagged == tuples_flagless, role
        for tup in tuples_flagless:
            flagged_outcome = _model_config_outcome(flagged, role, tuple(tup))
            flagless_outcome = _model_config_outcome(flagless, role, tuple(tup))
            assert flagged_outcome == flagless_outcome, (role, tup)


# ---------------------------------------------------------------------------
# V1 YAML with enable_wideep: converts + warns + behaves identically
# ---------------------------------------------------------------------------


def _v1_agg(enable_wideep: bool | None) -> dict:
    v1 = {
        "mode": "patch",
        "serving_mode": "agg",
        "model_path": MODEL,
        "system_name": "h200_sxm",
        "backend_name": "trtllm",
        "total_gpus": 64,
    }
    if enable_wideep is not None:
        v1["enable_wideep"] = enable_wideep
    return v1


def test_v1_yaml_with_enable_wideep_converts_warns_and_is_inert():
    with pytest.warns(DeprecationWarning, match="'enable_wideep' is deprecated and ignored"):
        flagged = Task.from_yaml(_v1_agg(enable_wideep=True))
    assert flagged.enable_wideep is True  # mapping mechanics unchanged

    flagless = Task.from_yaml(_v1_agg(enable_wideep=None))
    for dim in _AGG_DIMS:
        attr = f"agg_{dim}_candidates"
        assert getattr(flagged, attr) == getattr(flagless, attr), attr
