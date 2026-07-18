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

    #: timing inputs are quantized to this granularity before lookup so the
    #: evaluator's per-pass queries (context grows by one token per pass)
    #: hit the cache instead of issuing thousands of perf-DB queries; the
    #: induced timing error is <= one grain over the sequence length
    _GRAIN = 64

    def __init__(self, model, database, backend):
        self._model = model
        self._database = database
        self._backend = backend
        self._cache: dict = {}

    @classmethod
    def _q(cls, v: int) -> int:
        return max(1, round(v / cls._GRAIN) * cls._GRAIN)

    def prefill_ms(self, batch_size: int, mean_isl: int, mean_prefix: int) -> float:
        # quantize the EFFECTIVE length (isl - prefix), not isl and prefix
        # independently — independent rounding can collapse the difference
        # to zero for small trailing chunks
        eff = max(1, mean_isl - mean_prefix)
        eff = eff if eff < self._GRAIN else self._q(eff)
        mean_prefix = 0 if mean_prefix <= 0 else self._q(mean_prefix)
        mean_isl = mean_prefix + eff
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
        context_len = self._q(context_len)
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

    def mixed_pass_ms(self, ctx_tokens: int, gen_tokens: int, isl: int, osl: int, prefix: int) -> float:
        """Duration of one mixed (prefill chunk + decode) pass.

        Thin delegation to ``BaseBackend._get_mix_step_latency`` — the same
        runner behind run_agg's mix-step estimate, including its batching
        efficiency factor — so the evaluator's mixed passes cost exactly
        what the screening tier's ``t_mix`` costs at the same operating
        point. A prefill_ms + decode_ms sum would double-count the
        non-attention cost (weights are loaded from HBM once for the
        combined batch). The calendar falls back to the sum when a timing
        model does not provide this hook.

        Args mirror the runner: ctx_tokens = prefill compute tokens this
        pass, gen_tokens = decode rows this pass, (isl, osl, prefix) = the
        workload shape the runner uses for attention sizing.
        """
        ctx_tokens = ctx_tokens if ctx_tokens < self._GRAIN else self._q(ctx_tokens)
        key = ("mix", ctx_tokens, gen_tokens, isl, osl, prefix)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        runtime_config = RuntimeConfig(batch_size=max(1, gen_tokens), beam_width=1, isl=isl, osl=osl, prefix=prefix)
        latency_ms, _, _, _ = self._backend._get_mix_step_latency(
            self._model, self._database, runtime_config, ctx_tokens, gen_tokens, isl, osl, prefix
        )
        total = max(float(latency_ms) * self._backend._mix_step_efficiency(ctx_tokens, gen_tokens), 1e-6)
        self._cache[key] = total
        return total
