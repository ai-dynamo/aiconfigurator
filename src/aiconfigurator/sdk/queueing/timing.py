# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TimingModel adapter over AIC's own phase estimators.

Thin delegation to ``BaseBackend._run_context_phase`` /
``_run_generation_phase`` — the same estimators behind ``run_static`` —
so backend-specific overrides, correction scales, and any future evolution
of the phase runners apply automatically. This module deliberately contains
NO per-op query logic of its own: the phase runners are the single source
of truth for "how long does a prefill batch / decode step take".
"""

from __future__ import annotations

from aiconfigurator.sdk.config import RuntimeConfig


class DatabaseTimingModel:
    """Build prefill/decode timing callables from (model, database, backend).

    Args:
        model: an SDK model (from ``models.get_model``)
        database: a perf database (from ``perf_database.get_database``)
        backend: a backend instance (from ``backends.factory.get_backend``) —
            its phase runners are the authority for timing semantics
    """

    def __init__(self, model, database, backend):
        self._model = model
        self._database = database
        self._backend = backend
        self._cache: dict = {}

    def prefill_ms(self, batch_size: int, mean_isl: int, mean_prefix: int) -> float:
        key = ("pf", batch_size, mean_isl, mean_prefix)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        runtime_config = RuntimeConfig(batch_size=batch_size, beam_width=1, isl=mean_isl, osl=1, prefix=mean_prefix)
        latency_dict, _, _ = self._backend._run_context_phase(
            self._model, self._database, runtime_config, batch_size, mean_isl, mean_prefix
        )
        total = float(sum(latency_dict.values()))
        self._cache[key] = total
        return total

    def decode_ms(self, batch_size: int, context_len: int) -> float:
        key = ("dec", batch_size, context_len)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        # one decode iteration at the given context length (osl=2, stride=1
        # evaluates a single generation step — same shape run_static uses)
        runtime_config = RuntimeConfig(batch_size=batch_size, beam_width=1, isl=context_len, osl=2)
        latency_dict, _, _ = self._backend._run_generation_phase(
            self._model, self._database, runtime_config, batch_size, 1, context_len, 2, 1
        )
        total = max(float(sum(latency_dict.values())), 1e-6)
        self._cache[key] = total
        return total
