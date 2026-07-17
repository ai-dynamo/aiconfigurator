# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TimingModel adapter over the AIC perf database.

Query semantics are identical to the Dynamo mocker bridge
(dynamo lib/bindings/python/src/dynamo/_internal/aic.py) and verified to
float precision against it, so the queueing model, the DES oracle, and the
mocker share one timing basis.
"""

from __future__ import annotations


class DatabaseTimingModel:
    """Build prefill/decode timing callables from (model, database)."""

    def __init__(self, model, database):
        self._model = model
        self._database = database
        self._model_name = getattr(model, "model_name", None) or model.model_path
        self._cache: dict = {}

    def prefill_ms(self, batch_size: int, mean_isl: int, mean_prefix: int) -> float:
        effective_isl = max(1, mean_isl - mean_prefix)
        key = ("pf", batch_size, effective_isl, mean_prefix)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        total = 0.0
        for op in self._model.context_ops:
            op_name = getattr(op, "_name", "")
            x = batch_size if "logits_gemm" in op_name else batch_size * effective_isl
            total += float(
                op.query(
                    self._database,
                    x=x,
                    batch_size=batch_size,
                    beam_width=1,
                    s=effective_isl,
                    prefix=mean_prefix,
                    model_name=self._model_name,
                    seq_imbalance_correction_scale=1.0,
                )
            )
        self._cache[key] = total
        return total

    def decode_ms(self, batch_size: int, context_len: int) -> float:
        eff_bs = batch_size * (self._model._nextn + 1)
        key = ("dec", eff_bs, context_len)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        total = 0.0
        for op in self._model.generation_ops:
            total += float(
                op.query(
                    self._database,
                    x=eff_bs,
                    batch_size=eff_bs,
                    beam_width=1,
                    s=context_len + 1,
                    model_name=self._model_name,
                    gen_seq_imbalance_correction_scale=1.0,
                )
            )
        total = max(total, 1e-6)
        self._cache[key] = total
        return total
