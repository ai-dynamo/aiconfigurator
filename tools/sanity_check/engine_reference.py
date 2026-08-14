# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiled-engine per-op reference for the sanity-check charts.

Each ``query_*`` method builds the corresponding single-op list and evaluates
it through the compiled Rust engine's ad-hoc op-list FFI —
``EngineHandle.evaluate_ops_json`` for SILICON values and
``EngineHandle.evaluate_ops_sol_json`` for the ``(sol_time, sol_math,
sol_mem)`` SOL_FULL triples. The engine handle is bound to the same live
database view (shared-layer sources, query mode, transfer policy) via
``EngineHandle.for_database``.

The charted values are the engine's OP-LEVEL estimates — the single per-op
oracle that remains once #1357 retires the ``PerfDatabase.query_*`` per-call
stack. For most families that is exactly the old facade value; two families
deliberately differ from the retired raw-table facade semantics:

- context attention includes the fused rope/kv-write extras the engine
  charges on top of the attention table (small shape-independent additive);
- gemm ``fp8_static`` charts the engine's model (dynamic-fp8 row minus the
  overhead tables); the facade's fp8_static curve was a duplicate of the
  fp8 curve (its lookup normalized to the fp8 table).

Only the two modes the charts use are supported: ``DatabaseMode.SILICON``
(the probe engine's configured mode) and ``DatabaseMode.SOL_FULL``. Rust
raises the same typed SDK errors (``PerfDataNotAvailableError`` etc.), so
the notebook's probe-and-skip handling works unchanged.

``query_trtllm_alltoall`` is deliberately NOT re-oracled: its chart walks the
raw per-phase table (prepare/dispatch/combine/combine_lp), which no op-level
evaluation expresses — it stays on the Python facade as a documented PR-5
residual (like the AFD comm op).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.common import DatabaseMode
from aiconfigurator_core.sdk.engine import EngineHandle, build_ops_json
from aiconfigurator_core.sdk.operations.attention import ContextAttention, GenerationAttention
from aiconfigurator_core.sdk.operations.communication import NCCL, CustomAllReduce
from aiconfigurator_core.sdk.operations.dsa import (
    DEFAULT_DSA_ARCHITECTURE,
    ContextDSAModule,
    GenerationDSAModule,
)
from aiconfigurator_core.sdk.operations.gemm import GEMM
from aiconfigurator_core.sdk.operations.mla import ContextMLA, GenerationMLA
from aiconfigurator_core.sdk.operations.moe import MoE

# Ad-hoc probe shapes must fit the FFI's u32 token count. Comm charts sweep
# message sizes up to 2**32 elements, so the element count is factored into
# `x * per_token_elements` with both factors in range.
_MAX_X = 2**31


def _factor_message_size(size: int) -> tuple[int, int]:
    """Factor ``size`` into ``(x, per_token_elements)`` with ``x <= 2**31``.

    The comm ops compute ``message_size = x * per_token_elements``; the chart
    sweeps powers of two, so an exact power-of-two split always exists.
    """
    per_token = 1
    x = int(size)
    while x > _MAX_X:
        if x % 2:
            raise ValueError(f"message size {size} cannot be factored into u32 range")
        x //= 2
        per_token *= 2
    return x, per_token


class EngineReference:
    """Per-op query surface backed by the compiled Rust engine."""

    def __init__(self, database: Any) -> None:
        self._database = database
        self._engine = EngineHandle.for_database(database)
        # `build_ops_json` only reads `architecture` off the model.
        self._model_stub = SimpleNamespace(architecture="")

    def _eval(
        self,
        op: Any,
        *,
        is_context: bool,
        batch_size: int,
        s: int,
        prefix: int = 0,
        x: int | None = None,
        database_mode: DatabaseMode,
    ):
        ops_json = build_ops_json(
            [op],
            model=self._model_stub,
            backend=self._database.backend,
            database=self._database,
        )
        if database_mode == DatabaseMode.SOL_FULL:
            entries = self._engine.evaluate_ops_sol_json(
                ops_json, is_context=is_context, batch_size=batch_size, s=s, prefix=prefix, x=x
            )
            (_, sol_time, sol_math, sol_mem) = entries[0]
            return sol_time, sol_math, sol_mem
        if database_mode != DatabaseMode.SILICON:
            raise ValueError(f"EngineReference supports SILICON and SOL_FULL only, got {database_mode}")
        entries = self._engine.evaluate_ops_json(
            ops_json, is_context=is_context, batch_size=batch_size, s=s, prefix=prefix, x=x
        )
        return entries[0][1]

    # ------------------------------------------------------------------ #
    # query_* mirrors (facade-compatible signatures, charts' subset)
    # ------------------------------------------------------------------ #

    def query_gemm(self, m, n, k, quant_mode, database_mode):
        # fp8_static charts the ENGINE's model for it (dynamic-fp8 row minus
        # the collected overhead tables, floored at the roofline). The old
        # facade normalized the lookup to fp8, so its fp8_static curve was a
        # pixel-identical duplicate of the fp8 curve — no information lost.
        op = GEMM("gemm", 1.0, n, k, quant_mode)
        return self._eval(op, is_context=True, batch_size=1, s=1, x=int(m), database_mode=database_mode)

    def query_context_attention(self, b, s, n, n_kv, kvcache_quant_mode, fmha_quant_mode, database_mode, prefix=0):
        # Op-level estimate: includes the fused rope/kv-write extras the
        # engine charges on top of the attention table (a small, shape-
        # independent additive term the old raw-table facade did not chart).
        op = ContextAttention("context_attention", 1.0, n, n_kv, kvcache_quant_mode, fmha_quant_mode)
        return self._eval(
            op, is_context=True, batch_size=int(b), s=int(s), prefix=int(prefix), database_mode=database_mode
        )

    def query_generation_attention(self, b, s, n, n_kv, kvcache_quant_mode, database_mode):
        op = GenerationAttention("generation_attention", 1.0, n, n_kv, kvcache_quant_mode)
        return self._eval(op, is_context=False, batch_size=int(b), s=int(s), database_mode=database_mode)

    def query_context_mla(self, b, s, num_heads, kvcache_quant_mode, fmha_quant_mode, database_mode, prefix=0):
        op = ContextMLA("context_mla", 1.0, num_heads, kvcache_quant_mode, fmha_quant_mode)
        return self._eval(
            op, is_context=True, batch_size=int(b), s=int(s), prefix=int(prefix), database_mode=database_mode
        )

    def query_generation_mla(self, b, s, num_heads, kvcache_quant_mode, database_mode):
        op = GenerationMLA("generation_mla", 1.0, num_heads, kvcache_quant_mode)
        return self._eval(op, is_context=False, batch_size=int(b), s=int(s), database_mode=database_mode)

    def query_context_dsa_module(
        self,
        b,
        s,
        num_heads,
        kvcache_quant_mode,
        fmha_quant_mode,
        database_mode,
        prefix=0,
        architecture=DEFAULT_DSA_ARCHITECTURE,
    ):
        # The op layer derives the index dims (index_n_heads / index_head_dim
        # / index_topk) from `architecture` — pass the architecture instead of
        # raw index dims (the pre-re-oracle facade's calling convention).
        op = ContextDSAModule(
            "context_dsa",
            1.0,
            num_heads,
            kvcache_quant_mode,
            fmha_quant_mode,
            common.GEMMQuantMode.bfloat16,
            architecture=architecture,
        )
        return self._eval(
            op, is_context=True, batch_size=int(b), s=int(s), prefix=int(prefix), database_mode=database_mode
        )

    def query_generation_dsa_module(
        self,
        b,
        s,
        num_heads,
        kv_cache_dtype,
        database_mode,
        architecture=DEFAULT_DSA_ARCHITECTURE,
    ):
        op = GenerationDSAModule(
            "generation_dsa",
            1.0,
            num_heads,
            kv_cache_dtype,
            common.GEMMQuantMode.bfloat16,
            architecture=architecture,
        )
        return self._eval(op, is_context=False, batch_size=int(b), s=int(s), database_mode=database_mode)

    def query_moe(
        self,
        num_tokens,
        hidden_size,
        inter_size,
        topk,
        num_experts,
        moe_tp_size,
        moe_ep_size,
        quant_mode,
        workload_distribution,
        database_mode,
    ):
        op = MoE(
            "moe",
            1.0,
            hidden_size,
            inter_size,
            topk,
            num_experts,
            moe_tp_size,
            moe_ep_size,
            quant_mode,
            workload_distribution,
            attention_dp_size=1,
        )
        return self._eval(op, is_context=True, batch_size=1, s=1, x=int(num_tokens), database_mode=database_mode)

    def query_custom_allreduce(self, quant_mode, tp_size, size, database_mode):
        if quant_mode != common.CommQuantMode.half:
            # The op-level spec pins custom AR to half (its table has no
            # other dtype); the chart only sweeps half.
            raise ValueError(f"custom allreduce reference supports CommQuantMode.half only, got {quant_mode}")
        x, per_token = _factor_message_size(int(size))
        op = CustomAllReduce("custom_allreduce", 1.0, per_token, tp_size)
        return self._eval(op, is_context=True, batch_size=1, s=1, x=x, database_mode=database_mode)

    def query_nccl(self, quant_mode, num_gpus, operation, size, database_mode):
        x, per_token = _factor_message_size(int(size))
        op = NCCL("nccl", 1.0, operation, per_token, num_gpus, quant_mode)
        return self._eval(op, is_context=True, batch_size=1, s=1, x=x, database_mode=database_mode)
