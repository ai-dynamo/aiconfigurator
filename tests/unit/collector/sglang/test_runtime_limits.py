# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from collector.sglang import runtime_limits

pytestmark = pytest.mark.unit


class _ChunkedIndexer:
    def _should_chunk_mqa_logits(self):
        return True

    def _get_topk_ragged(self, start, q_offset):
        while start < q_offset:
            logits_chunk = start
            start += 1
        return logits_chunk


def test_dsa_chunking_source_contract_is_detected(monkeypatch):
    module = SimpleNamespace(Indexer=_ChunkedIndexer)
    monkeypatch.setattr(runtime_limits.importlib, "import_module", lambda _name: module)

    assert runtime_limits.sglang_dsa_mqa_logits_chunking_supported()


def test_dsa_chunking_detection_failure_does_not_enable_a_heuristic_filter(monkeypatch):
    def missing(_name):
        raise ModuleNotFoundError

    monkeypatch.setattr(runtime_limits.importlib, "import_module", missing)

    with pytest.raises(RuntimeError, match="chunking source contract was not detected"):
        runtime_limits.sglang_dsa_mqa_logits_chunking_supported()
