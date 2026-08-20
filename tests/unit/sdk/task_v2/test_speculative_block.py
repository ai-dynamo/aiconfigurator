# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-level `speculative:` block — the scheme-based generalization of the
legacy nextn/nextn_accepted pair."""

from __future__ import annotations

import pytest

from aiconfigurator.sdk.task_v2 import Task

pytestmark = pytest.mark.unit

DRAFT_CONFIG = {
    "dspark_block_size": 5,
    "dspark_target_layer_ids": [40, 41, 42],
    "dspark_markov_rank": 256,
}


def _task(**kw) -> Task:
    # b200_sxm: the task-level native-FP4-on-Hopper gate predates the Hopper
    # w4a16 serving identities (now with native data) — relaxing it is a
    # separate upstream item; these tests target block parsing/profile math.
    return Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V4-Flash",
        system_name="b200_sxm",
        backend_name="sglang",
        **kw,
    )


def _dspark_block(**over) -> dict:
    block = {
        "method": "dspark",
        "params": {"num_draft_tokens": 7},
        "draft_config": dict(DRAFT_CONFIG),
        "accepted_tokens": 4.09,
    }
    block.update(over)
    return block


class TestSpeculativeBlock:
    def test_dspark_block_resolves_scheme_config(self):
        t = _task(speculative=_dspark_block())
        cfg = t.build_model_config(role="agg")
        assert cfg.speculation is not None
        assert cfg.speculation.kind == "dspark"
        assert cfg.nextn == 0  # dspark never triggers mtp scaling
        profile = t.build_speculative_profile()
        assert profile.tokens_per_iteration == pytest.approx(5.09)

    def test_mtp_method_desugars_to_legacy_pair(self):
        t = _task(speculative={"method": "mtp", "params": {"depth": 2}, "accepted_tokens": 0.8})
        assert t.nextn == 2
        assert t.nextn_accepted == 0.8
        legacy = _task(nextn=2, nextn_accepted=0.8)
        assert t.build_speculative_profile() == legacy.build_speculative_profile()

    def test_conflict_with_legacy_nextn_raises(self):
        with pytest.raises(ValueError):
            _task(nextn=2, nextn_accepted=0.5, speculative=_dspark_block())

    def test_accepted_tokens_required(self):
        with pytest.raises(ValueError):
            _task(speculative=_dspark_block(accepted_tokens=None))

    def test_accepted_bound_is_scheme_derived(self):
        with pytest.raises(ValueError):
            _task(speculative=_dspark_block(accepted_tokens=7.5))  # > N drafted

    def test_unknown_keys_rejected(self):
        with pytest.raises(ValueError):
            _task(speculative=_dspark_block(typo_key=1))

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError):
            _task(speculative={"method": "warpdrive", "accepted_tokens": 1.0})

    def test_none_method_is_noop(self):
        t = _task(speculative={"method": "none"})
        assert t.build_model_config(role="agg").speculation is None
        assert t.build_speculative_profile().tokens_per_iteration == 1.0
