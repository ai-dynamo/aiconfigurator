# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk.backends.sglang_backend import SGLANGBackend
from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend
from aiconfigurator.sdk.config import RuntimeConfig

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "backend_cls",
    [
        SGLANGBackend,
        TRTLLMBackend,
        VLLMBackend,
    ],
)
def test_agg_sweep_preserves_engine_step_backend(monkeypatch, backend_cls) -> None:
    backend = backend_cls()
    captured_backends = []

    monkeypatch.setattr(
        backend,
        "_get_ctx_tokens_list_for_agg_sweep",
        lambda *_args, **_kwargs: [8],
    )

    def _fake_run_agg(model, database, runtime_config, **_kwargs):
        del model, database
        captured_backends.append(runtime_config.engine_step_backend)
        summary = MagicMock()
        summary.check_oom.return_value = False
        summary.check_kv_cache_oom.return_value = False
        summary.get_result_dict.return_value = {"ttft": 1.0, "tpot": 1.0, "seq/s": 1.0}
        summary.get_per_ops_source.return_value = None
        return summary

    monkeypatch.setattr(backend, "run_agg", _fake_run_agg)

    backend.find_best_agg_result_under_constraints(
        model=MagicMock(),
        database=MagicMock(),
        runtime_config=RuntimeConfig(
            isl=8,
            osl=4,
            ttft=10.0,
            tpot=10.0,
            prefix=0,
            engine_step_backend="rust",
        ),
        max_batch_size=1,
        ctx_stride=8,
    )

    assert captured_backends == ["rust"]
