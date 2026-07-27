# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke parity checks for Python SDK engine-step latency versus Rust core.

Each test compares an apples-to-apples Python and Rust value for the
surface under test (static_ctx + static_gen, the agg/disagg pipelines
through `cli_estimate`, and Python's `_get_mix_step_latency` vs Rust's
mix-step FPM). A drift outside ``PARITY_RTOL`` fails the assertion with
a per-metric delta report so the failure mode is informative.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import pytest

from aiconfigurator.cli.api import cli_estimate
from aiconfigurator.sdk import common, config, errors, perf_database, rust_engine_step
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.operations import util_empirical

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class EngineStepParityCase:
    model_path: str
    system_name: str = "b200_sxm"
    backend_name: str = "vllm"
    backend_version: str = "0.19.0"
    batch_size: int = 1
    isl: int = 1024
    osl: int = 2
    prefix: int = 0
    tp_size: int = 8
    pp_size: int = 1
    attention_dp_size: int = 1
    moe_tp_size: int = 1
    moe_ep_size: int = 8
    cp_size: int = 1
    agg_batch_size: int = 2
    agg_ctx_tokens: int | None = None
    disagg_prefill_batch_size: int = 1
    disagg_prefill_num_workers: int = 1
    disagg_decode_batch_size: int = 4
    disagg_decode_num_workers: int = 1
    # HYBRID/EMPIRICAL parity knobs: database mode, transfer-policy preset
    # (None = default ALL_TRANSFERS), and a forced MoE quant used to steer a
    # query into a specific transfer tier (xquant/xprofile) on real data.
    database_mode: str = "SILICON"
    transfer_policy: str | None = None
    moe_quant_mode: str | None = None


SMOKE_CASES = [
    # Original 3 smoke cases (Phase 3).
    pytest.param(
        EngineStepParityCase(model_path="MiniMaxAI/MiniMax-M2.5"),
        id="minimax-m25-b200-vllm-019-isl1024-osl2",
    ),
    pytest.param(
        EngineStepParityCase(model_path="moonshotai/Kimi-K2.5"),
        id="kimi-k25-b200-vllm-019-isl1024-osl2",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="MiniMaxAI/MiniMax-M2.5",
            batch_size=2,
            isl=2048,
            osl=5,
            prefix=256,
        ),
        id="minimax-m25-b200-vllm-019-sampled-prefix",
    ),
    # Phase 4 D1: extra MoE coverage on b200_sxm/vllm/0.19.0.
    pytest.param(
        EngineStepParityCase(model_path="MiniMaxAI/MiniMax-M2.7"),
        id="minimax-m27-b200-vllm-019-isl1024-osl2",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3-30B-A3B",
            tp_size=4,
            moe_ep_size=4,
        ),
        id="qwen3-30b-a3b-b200-vllm-019-isl1024-osl2",
    ),
    pytest.param(
        EngineStepParityCase(model_path="Qwen/Qwen3-235B-A22B"),
        id="qwen3-235b-a22b-b200-vllm-019-isl1024-osl2",
    ),
    # Phase 4 D1: dense (Llama-family) coverage on b200_sxm/vllm/0.19.0.
    # The smoke MoE defaults (`moe_ep_size=8`) are unused by the dense path
    # but pass through `cli_estimate` without harm.
    pytest.param(
        EngineStepParityCase(model_path="Qwen/Qwen3-32B"),
        id="qwen3-32b-b200-vllm-019-isl1024-osl2",
    ),
    pytest.param(
        EngineStepParityCase(model_path="meta-llama/Meta-Llama-3.1-70B"),
        id="llama31-70b-b200-vllm-019-isl1024-osl2",
    ),
    pytest.param(
        EngineStepParityCase(model_path="meta-llama/Meta-Llama-3.1-8B"),
        id="llama31-8b-b200-vllm-019-isl1024-osl2",
    ),
    # Phase 4 D1: cross-system coverage on the smoke MiniMax model.
    pytest.param(
        EngineStepParityCase(
            model_path="MiniMaxAI/MiniMax-M2.5",
            system_name="h200_sxm",
        ),
        id="minimax-m25-h200-vllm-019-isl1024-osl2",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="MiniMaxAI/MiniMax-M2.5",
            system_name="h100_sxm",
        ),
        id="minimax-m25-h100-vllm-019-isl1024-osl2",
    ),
    # Phase 4 D4: DeepSeek-family coverage unlocked by the `Op::Overlap`
    # variant + `128 // tp_size` MLA head count fix.
    pytest.param(
        EngineStepParityCase(model_path="deepseek-ai/DeepSeek-V3"),
        id="deepseek-v3-b200-vllm-019-isl1024-osl2",
    ),
    pytest.param(
        EngineStepParityCase(model_path="deepseek-ai/DeepSeek-R1"),
        id="deepseek-r1-b200-vllm-019-isl1024-osl2",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="moonshotai/Kimi-K2.5",
            system_name="h200_sxm",
        ),
        id="kimi-k25-h200-vllm-019-isl1024-osl2",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="moonshotai/Kimi-K2.5",
            system_name="h100_sxm",
        ),
        id="kimi-k25-h100-vllm-019-isl1024-osl2",
    ),
    # Phase 4 D4: cross-backend (SGLang non-DeepEP path) coverage.
    pytest.param(
        EngineStepParityCase(
            model_path="MiniMaxAI/MiniMax-M2.5",
            backend_name="sglang",
            backend_version="0.5.10",
        ),
        id="minimax-m25-b200-sglang-0510-isl1024-osl2",
    ),
    # Phase 4 D5: DeepSeek-family on SGLang, unlocked by the
    # `Op::Fallback` variant that mirrors Python's `FallbackOp` (primary
    # `MLAModule` -> granular `MlaBmm + ContextMla/GenerationMla + MlaBmm`
    # chain when the module-level perf data is absent, which is the case
    # for SGLang and TRT-LLM).
    pytest.param(
        EngineStepParityCase(
            model_path="moonshotai/Kimi-K2.5",
            backend_name="sglang",
            backend_version="0.5.10",
        ),
        id="kimi-k25-b200-sglang-0510-isl1024-osl2",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="moonshotai/Kimi-K2.5",
            system_name="h200_sxm",
            backend_name="sglang",
            backend_version="0.5.10",
        ),
        id="kimi-k25-h200-sglang-0510-isl1024-osl2",
    ),
    # Phase 4 D6: NemotronNas (Puzzle / DeciLM per-block architecture).
    pytest.param(
        EngineStepParityCase(model_path="nvidia/Llama-3_3-Nemotron-Super-49B-v1"),
        id="nemotron-nas-b200-vllm-019-isl1024-osl2",
    ),
    # Phase 4 D7-B: Qwen3.5 hybrid GDN + full-attention.
    pytest.param(
        EngineStepParityCase(model_path="Qwen/Qwen3.5-27B"),
        id="qwen35-27b-b200-vllm-019-isl1024-osl2",
    ),
    pytest.param(
        EngineStepParityCase(model_path="Qwen/Qwen3.5-397B-A17B"),
        id="qwen35-397b-a17b-b200-vllm-019-isl1024-osl2",
    ),
    # Phase 4 D7-D: NemotronH hybrid Mamba2 + attention + MLP.
    pytest.param(
        EngineStepParityCase(model_path="nvidia/Nemotron-H-56B-Base-8K"),
        id="nemotron-h-56b-b200-vllm-019-isl1024-osl2",
    ),
    # Phase 4 D7-E: DeepSeekV32 family (DSA attention + MoE).
    pytest.param(
        EngineStepParityCase(model_path="deepseek-ai/DeepSeek-V3.2"),
        id="deepseek-v32-b200-vllm-019-isl1024-osl2",
    ),
    pytest.param(
        EngineStepParityCase(model_path="zai-org/GLM-5"),
        id="glm5-b200-vllm-019-isl1024-osl2",
    ),
    # Tripwire for the DSA kernel_source bucket contract (review B1, both
    # halves): sglang 0.5.14 records executed-kernel names whose bucket
    # classification must match between the Python and Rust loaders — a
    # long-context bf16-KV DSA query diverges ~30% if either side falls back
    # to the bare substring rule.
    pytest.param(
        EngineStepParityCase(
            model_path="zai-org/GLM-5",
            backend_name="sglang",
            backend_version="0.5.14",
            isl=16384,
        ),
        id="glm5-b200-sglang-0514-isl16384-osl2",
    ),
    # GLM-5.2 shared-index amortization (full_frac = 21/78): per-layer DSA is
    # w*full + (1-w)*skip using the skip_indexer rows collected in the same
    # parquet. Tripwire for the Rust skip-table port — dropping the skip rows
    # (the pre-port behavior) silently overestimates every GLM-5.2 sweep.
    pytest.param(
        EngineStepParityCase(
            model_path="nvidia/GLM-5.2-NVFP4",
            backend_name="sglang",
            backend_version="0.5.14",
        ),
        id="glm52-b200-sglang-0514-isl1024-osl2",
    ),
    # Phase 4 D7-F: backend coverage for newly-ported families. The
    # builders are backend-independent (the per-backend conditional
    # `_attn_dp` / `_tp_allreduce` branches are handled inside each
    # family's `build_*` function), but the perf-DB tables live in
    # per-backend directories — these cases prove the same Rust
    # builder matches Python on sglang and trtllm tables, not just
    # vllm.
    pytest.param(
        EngineStepParityCase(
            model_path="nvidia/Llama-3_3-Nemotron-Super-49B-v1",
            backend_name="sglang",
            backend_version="0.5.10",
        ),
        id="nemotron-nas-b200-sglang-0510-isl1024-osl2",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="nvidia/Llama-3_3-Nemotron-Super-49B-v1",
            backend_name="trtllm",
            backend_version="1.3.0rc10",
        ),
        id="nemotron-nas-b200-trtllm-130rc10-isl1024-osl2",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3.5-27B",
            backend_name="sglang",
            backend_version="0.5.10",
        ),
        id="qwen35-27b-b200-sglang-0510-isl1024-osl2",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3.5-27B",
            backend_name="trtllm",
            backend_version="1.3.0rc10",
        ),
        id="qwen35-27b-b200-trtllm-130rc10-isl1024-osl2",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3.5-397B-A17B",
            backend_name="sglang",
            backend_version="0.5.10",
        ),
        id="qwen35-397b-a17b-b200-sglang-0510-isl1024-osl2",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3.5-397B-A17B",
            backend_name="trtllm",
            backend_version="1.3.0rc10",
        ),
        id="qwen35-397b-a17b-b200-trtllm-130rc10-isl1024-osl2",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="nvidia/Nemotron-H-56B-Base-8K",
            backend_name="sglang",
            backend_version="0.5.10",
        ),
        id="nemotron-h-56b-b200-sglang-0510-isl1024-osl2",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="nvidia/Nemotron-H-56B-Base-8K",
            backend_name="trtllm",
            backend_version="1.3.0rc10",
        ),
        id="nemotron-h-56b-b200-trtllm-130rc10-isl1024-osl2",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="deepseek-ai/DeepSeek-V3.2",
            backend_name="sglang",
            backend_version="0.5.10",
        ),
        id="deepseek-v32-b200-sglang-0510-isl1024-osl2",
    ),
    # Attention-DP coverage: sglang all-gathers the DP-sharded tokens before
    # the MoE, so MoE compute scales by attention_dp_size. Every other MoE case
    # runs at attention_dp=1, which left the Rust MoE token scaling untested;
    # this Qwen3-235B config (tp1 dp8 etp4 ep2) exercises attention_dp>1.
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3-235B-A22B",
            backend_name="sglang",
            backend_version="0.5.10",
            tp_size=1,
            attention_dp_size=8,
            moe_tp_size=4,
            moe_ep_size=2,
        ),
        id="qwen3-235b-b200-sglang-0510-adp8-etp4ep2",
    ),
    # Phase 4 D7-G: shape-variation coverage. All previous cases run at
    # `(batch=1, isl=1024, osl=2)` (plus one prefix variant). The four
    # cases below sweep the four parity-sensitive shape directions, each
    # placed on a different family + backend combination so a regression
    # in one direction also implicates the family/backend it lands on:
    #
    #  - decode-heavy: short isl, long osl -> stresses gen-side seq
    #    interpolation and per-step accumulation (MoE on vllm).
    #  - prefill-heavy: long isl, short osl -> exercises context-attention
    #    interp at the upper axis bound (dense on trtllm).
    #  - prefix coverage: mid isl + large prefix -> MLA prefix correction
    #    and prefix-slice picking (MLA family on vllm).
    #  - larger batch: bs > 1 -> Mamba2 batch interpolation away from
    #    bs=1 (state-space family on sglang); also bumps the agg/disagg
    #    batch defaults so the bs effect actually reaches those pipelines.
    pytest.param(
        EngineStepParityCase(
            model_path="MiniMaxAI/MiniMax-M2.5",
            isl=128,
            osl=64,
        ),
        id="minimax-m25-b200-vllm-019-shape-decode-heavy",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3-32B",
            backend_name="trtllm",
            backend_version="1.3.0rc10",
            isl=8192,
            osl=2,
        ),
        id="qwen3-32b-b200-trtllm-130rc10-shape-prefill-heavy",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="deepseek-ai/DeepSeek-V3",
            isl=2048,
            osl=16,
            prefix=1024,
        ),
        id="deepseek-v3-b200-vllm-019-shape-prefix-heavy",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="nvidia/Nemotron-H-56B-Base-8K",
            backend_name="sglang",
            backend_version="0.5.10",
            batch_size=8,
            agg_batch_size=8,
            disagg_decode_batch_size=8,
        ),
        id="nemotron-h-56b-b200-sglang-0510-shape-batch8",
    ),
    # Phase 4 D7-H: GPT family (gpt.py -> gpt.rs). Dense GQA transformer
    # with non-gated FFN; the b200/vllm case used to be error-symmetric
    # (no perf data) and now lands at full numeric parity through the
    # Rust builder. The b200/trtllm/1.3.0rc10 case (support matrix
    # PASS) verifies the same builder against the trtllm tables.
    pytest.param(
        EngineStepParityCase(model_path="openai/gpt-oss-20b"),
        id="gpt-oss-20b-b200-vllm-019-isl1024-osl2",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="openai/gpt-oss-20b",
            backend_name="trtllm",
            backend_version="1.3.0rc10",
        ),
        id="gpt-oss-20b-b200-trtllm-130rc10-isl1024-osl2",
    ),
    # Phase 4 D7-C: data-gap families. Python errors with
    # `PerfDataNotAvailableError` because the perf DB doesn't ship the
    # required tables for these shapes; Rust errors at the equivalent
    # query point (`AicError::PerfDatabase`). The error-symmetry contract
    # asserts both engines fail together — same outcome, even if the
    # exact failure point in the op graph differs.
    pytest.param(
        EngineStepParityCase(model_path="meta-llama/Llama-4-Scout-17B-16E-Instruct"),
        id="llama4-scout-b200-vllm-019-isl1024-osl2",
    ),
    pytest.param(
        EngineStepParityCase(model_path="deepseek-ai/DeepSeek-V4-Flash"),
        id="deepseek-v4-flash-b200-vllm-019-isl1024-osl2",
    ),
    # Phase 5 D8: smoke coverage for the 14 unique (model, system, backend,
    # version) tuples that surfaced as DRIFT in the 2026-06-01 full
    # support-matrix Pareto scan (16 entries; 14 unique combos because two
    # gb200/vllm/0.14.0 models drifted in both agg and disagg modes).
    # Each combo here exercises all four parity surfaces (static, mixed,
    # agg, disagg). Adding them at the default smoke shape means:
    #   - Tuples where the NCCL/OneCCL path fix (5ce469ff) was the root
    #     cause now compute and assert numeric parity going forward.
    #   - Tuples where the perf-DB lacks data for the smoke shape
    #     (`tp=8, moe_ep=8, isl=1024, osl=2`) error symmetrically in both
    #     engines, exercising the error-symmetry contract.
    # See the support-matrix scan triage notes in the project docs for the
    # full triage / cluster table.
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3-1.7B",
            system_name="h100_sxm",
            backend_name="vllm",
            backend_version="0.14.0",
        ),
        id="qwen3-17b-h100-vllm-014-scan-coverage",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3-30B-A3B",
            system_name="b60",
            backend_name="vllm",
            backend_version="0.12.0",
            tp_size=4,
            moe_ep_size=4,
        ),
        id="qwen3-30b-a3b-b60-vllm-012-scan-coverage",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3-30B-A3B",
            system_name="h200_sxm",
            backend_name="sglang",
            backend_version="0.5.9",
            tp_size=4,
            moe_ep_size=4,
        ),
        id="qwen3-30b-a3b-h200-sglang-059-scan-coverage",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3-30B-A3B",
            system_name="gb300",
            backend_name="sglang",
            backend_version="0.5.9",
            tp_size=4,
            moe_ep_size=4,
        ),
        id="qwen3-30b-a3b-gb300-sglang-059-scan-coverage",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3-8B",
            backend_name="trtllm",
            backend_version="1.2.0rc5",
        ),
        id="qwen3-8b-b200-trtllm-120rc5-scan-coverage",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3.5-27B",
            system_name="b300_sxm",
            backend_name="trtllm",
            backend_version="1.3.0rc10",
        ),
        id="qwen35-27b-b300-trtllm-130rc10-scan-coverage",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="deepseek-ai/DeepSeek-R1",
            system_name="gb200",
            backend_name="vllm",
            backend_version="0.14.0",
        ),
        id="deepseek-r1-gb200-vllm-014-scan-coverage",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="deepseek-ai/DeepSeek-V3",
            system_name="gb200",
            backend_name="vllm",
            backend_version="0.14.0",
        ),
        id="deepseek-v3-gb200-vllm-014-scan-coverage",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="meta-llama/Meta-Llama-3.1-405B",
            system_name="b300_sxm",
            backend_name="sglang",
            backend_version="0.5.10",
        ),
        id="llama31-405b-b300-sglang-0510-scan-coverage",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="meta-llama/Meta-Llama-3.1-8B",
            system_name="gb200",
            backend_name="trtllm",
            backend_version="1.3.0rc10",
        ),
        id="llama31-8b-gb200-trtllm-130rc10-scan-coverage",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="moonshotai/Kimi-K2.5",
            system_name="b300_sxm",
            backend_name="vllm",
            backend_version="0.19.0",
        ),
        id="kimi-k25-b300-vllm-019-scan-coverage",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="moonshotai/Kimi-K2.5",
            system_name="gb300",
            backend_name="vllm",
            backend_version="0.19.0",
        ),
        id="kimi-k25-gb300-vllm-019-scan-coverage",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="moonshotai/Kimi-K2.5",
            system_name="h200_sxm",
            backend_name="vllm",
            backend_version="0.14.0",
        ),
        id="kimi-k25-h200-vllm-014-scan-coverage",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="nvidia/Nemotron-H-56B-Base-8K",
            system_name="h200_sxm",
            backend_name="vllm",
            backend_version="0.19.0",
        ),
        id="nemotron-h-56b-h200-vllm-019-scan-coverage",
    ),
]

PARITY_RTOL = 0.01


# Error-symmetry contract: when Python raises one of these, Rust is
# expected to raise (Python's `PerfDataNotAvailableError` and friends
# travel through `cli_estimate` as `ValueError` / `RuntimeError`
# subclasses; the Rust FFI maps `AicError::PerfDatabase`/`Io` to the same
# `PerfDataNotAvailableError` and `AicError::EmpiricalNotImplemented` to
# `EmpiricalNotImplementedError` — see `TestRustTypedErrorsAcrossFfi` —
# with everything else as `ValueError`). Tests count any exception in
# either as a sentinel
# value `_ERROR` and assert that *both* engines either compute or
# error consistently. Numeric tolerance only applies when both
# compute.
class _ErrorSentinel:
    """Singleton marker for "this metric raised an exception"."""

    __slots__ = ("kind", "message")

    def __init__(self, exc: BaseException) -> None:
        self.kind = type(exc).__name__
        self.message = str(exc).splitlines()[0][:200]

    def __repr__(self) -> str:
        return f"ERROR({self.kind}: {self.message})"


def _quiet_call(func, *args, **kwargs):
    """Keep interpolation loader chatter out of parity test output."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return func(*args, **kwargs)


def _safe_value(thunk) -> float | _ErrorSentinel:
    """Run `thunk()` and return its numeric result, or an `_ErrorSentinel`
    capturing the exception type+message if it raised. The harness
    treats two sentinels as "matching" regardless of message — both
    engines erroring is the parity outcome we expect for data-gap
    families."""
    try:
        return float(thunk())
    except Exception as exc:
        return _ErrorSentinel(exc)


def _static_metrics(
    case: EngineStepParityCase,
    *,
    engine_step_backend: str,
    osl: int | None = None,
) -> dict[str, float | _ErrorSentinel]:
    kwargs = {
        "model_path": case.model_path,
        "system_name": case.system_name,
        "backend_name": case.backend_name,
        "backend_version": case.backend_version,
        "batch_size": case.batch_size,
        "isl": case.isl,
        "osl": case.osl if osl is None else osl,
        "prefix": case.prefix,
        "tp_size": case.tp_size,
        "pp_size": case.pp_size,
        "attention_dp_size": case.attention_dp_size,
        "moe_tp_size": case.moe_tp_size,
        "moe_ep_size": case.moe_ep_size,
        "stride": 1,
        "engine_step_backend": engine_step_backend,
        "database_mode": case.database_mode,
        "transfer_policy": case.transfer_policy,
        "moe_quant_mode": case.moe_quant_mode,
    }
    context_ms = _safe_value(
        lambda: (
            _quiet_call(cli_estimate, mode="static_ctx", **kwargs).summary.get_summary_df().iloc[0]["context_latency"]
        )
    )
    generation_ms = _safe_value(
        lambda: (
            _quiet_call(cli_estimate, mode="static_gen", **kwargs)
            .summary.get_summary_df()
            .iloc[0]["generation_latency"]
        )
    )
    if isinstance(context_ms, _ErrorSentinel) or isinstance(generation_ms, _ErrorSentinel):
        total: float | _ErrorSentinel = context_ms if isinstance(context_ms, _ErrorSentinel) else generation_ms
    else:
        total = context_ms + generation_ms
    return {
        "context_ms": context_ms,
        "generation_ms": generation_ms,
        "total_ms": total,
    }


def _agg_metrics(case: EngineStepParityCase, *, engine_step_backend: str) -> dict[str, float | _ErrorSentinel]:
    def call():
        return _quiet_call(
            cli_estimate,
            mode="agg",
            model_path=case.model_path,
            system_name=case.system_name,
            backend_name=case.backend_name,
            backend_version=case.backend_version,
            batch_size=case.agg_batch_size,
            ctx_tokens=case.agg_ctx_tokens or case.isl,
            isl=case.isl,
            osl=case.osl,
            prefix=case.prefix,
            tp_size=case.tp_size,
            pp_size=case.pp_size,
            attention_dp_size=case.attention_dp_size,
            moe_tp_size=case.moe_tp_size,
            moe_ep_size=case.moe_ep_size,
            engine_step_backend=engine_step_backend,
            database_mode=case.database_mode,
            transfer_policy=case.transfer_policy,
            moe_quant_mode=case.moe_quant_mode,
        )

    # Errors propagate from a single call site — capture once, surface
    # the same sentinel for every metric.
    err: _ErrorSentinel | None = None
    try:
        result = call()
    except Exception as exc:
        err = _ErrorSentinel(exc)
        result = None
    if err is not None:
        return {"ttft_ms": err, "tpot_ms": err, "request_latency_ms": err}
    return {
        "ttft_ms": float(result.ttft),
        "tpot_ms": float(result.tpot),
        "request_latency_ms": float(result.request_latency),
    }


def _disagg_metrics(case: EngineStepParityCase, *, engine_step_backend: str) -> dict[str, float | _ErrorSentinel]:
    def call():
        return _quiet_call(
            cli_estimate,
            mode="disagg",
            model_path=case.model_path,
            system_name=case.system_name,
            backend_name=case.backend_name,
            backend_version=case.backend_version,
            isl=case.isl,
            osl=case.osl,
            prefix=case.prefix,
            tp_size=case.tp_size,
            pp_size=case.pp_size,
            attention_dp_size=case.attention_dp_size,
            moe_tp_size=case.moe_tp_size,
            moe_ep_size=case.moe_ep_size,
            prefill_batch_size=case.disagg_prefill_batch_size,
            prefill_num_workers=case.disagg_prefill_num_workers,
            decode_batch_size=case.disagg_decode_batch_size,
            decode_num_workers=case.disagg_decode_num_workers,
            engine_step_backend=engine_step_backend,
            database_mode=case.database_mode,
            transfer_policy=case.transfer_policy,
            moe_quant_mode=case.moe_quant_mode,
        )

    err: _ErrorSentinel | None = None
    try:
        result = call()
    except Exception as exc:
        err = _ErrorSentinel(exc)
        result = None
    if err is not None:
        return {"ttft_ms": err, "tpot_ms": err, "request_latency_ms": err}
    return {
        "ttft_ms": float(result.ttft),
        "tpot_ms": float(result.tpot),
        "request_latency_ms": float(result.request_latency),
    }


def _mix_step_shape(case: EngineStepParityCase) -> dict:
    """Mix-step (chunked-prefill + decode) shape for a smoke case.

    Treats the case's single prefill request as one chunk with `case.isl`
    isl-equivalent tokens (matching Python's agg orchestration). Decode
    batch is `case.batch_size` (matches the FPM constructor below).
    """
    return {
        "ctx_tokens": case.isl,
        "gen_tokens": case.batch_size,
        "isl": case.isl,
        "osl": max(case.osl, 2),
        "prefix": case.prefix,
    }


def _case_database(case: EngineStepParityCase):
    """Perf database for a case: the plain cached database for SILICON
    defaults, or a mode/policy-configured query view for HYBRID/EMPIRICAL
    cases (mirrors what `cli_estimate` builds internally)."""
    if case.database_mode == "SILICON" and case.transfer_policy is None:
        return _quiet_call(perf_database.get_database, case.system_name, case.backend_name, case.backend_version)
    return _quiet_call(
        perf_database.get_database_view,
        case.system_name,
        case.backend_name,
        case.backend_version,
        # Mirror `cli_estimate`: non-SILICON modes tolerate missing tables at
        # load (the empirical layer covers the gaps at query time).
        allow_missing_data=case.database_mode != "SILICON",
        database_mode=case.database_mode,
        transfer_policy=case.transfer_policy,
    )


def _case_model_config(case: EngineStepParityCase) -> config.ModelConfig:
    return config.ModelConfig(
        tp_size=case.tp_size,
        pp_size=case.pp_size,
        attention_dp_size=case.attention_dp_size,
        moe_tp_size=case.moe_tp_size,
        moe_ep_size=case.moe_ep_size,
        cp_size=case.cp_size,
        moe_quant_mode=(common.MoEQuantMode[case.moe_quant_mode] if case.moe_quant_mode else None),
    )


def _python_mixed_step_ms(case: EngineStepParityCase) -> float:
    """Python `_get_mix_step_latency` for the case's mix-step shape."""
    database = _case_database(case)
    if database is None:
        raise RuntimeError(
            f"failed to load perf database for {case.system_name}/{case.backend_name}/{case.backend_version}"
        )
    model = _quiet_call(get_model, case.model_path, _case_model_config(case), case.backend_name)
    backend = get_backend(case.backend_name)
    runtime_config = config.RuntimeConfig(
        batch_size=case.batch_size,
        beam_width=1,
        isl=case.isl,
        osl=max(case.osl, 2),
        prefix=case.prefix,
    )
    shape = _mix_step_shape(case)
    latency_ms, _, _, _ = _quiet_call(
        backend._get_mix_step_latency,
        model,
        database,
        runtime_config,
        shape["ctx_tokens"],
        shape["gen_tokens"],
        shape["isl"],
        shape["osl"],
        shape["prefix"],
    )
    return float(latency_ms)


def _rust_mixed_step_ms(case: EngineStepParityCase) -> float:
    """Rust mix-step latency for the case's mix-step shape (same FPM)."""
    database = _case_database(case)
    if database is None:
        raise RuntimeError(
            f"failed to load perf database for {case.system_name}/{case.backend_name}/{case.backend_version}"
        )
    model = _quiet_call(get_model, case.model_path, _case_model_config(case), case.backend_name)
    shape = _mix_step_shape(case)
    return rust_engine_step.estimate_mixed_step_latency_with_rust(
        model,
        database,
        ctx_tokens=shape["ctx_tokens"],
        gen_tokens=shape["gen_tokens"],
        isl=shape["isl"],
        osl=shape["osl"],
        prefix=shape["prefix"],
    )


def _parity_mismatch_reason(
    comparisons: dict[str, tuple[float | _ErrorSentinel, float | _ErrorSentinel]],
    rtol: float = PARITY_RTOL,
) -> str | None:
    """Compare Python and Rust per-metric outputs with three valid pairings:

      - both compute                  -> numeric tolerance check
      - both error with the SAME kind -> pass (error-symmetry contract; the
        typed-FFI mapping raises the canonical sdk classes on the rust side,
        so `type(exc).__name__` must agree — a panic/TypeError paired with a
        typed miss is a real divergence, not symmetry)
      - anything else                 -> fail with the asymmetric reason

    Returns ``None`` when every metric in `comparisons` matches under
    one of those rules; otherwise returns a formatted multi-row diff.
    """
    rows = []
    has_mismatch = False
    metric_width = max([len("metric"), *(len(name) for name in comparisons)])
    for name, (python_value, rust_value) in comparisons.items():
        py_err = isinstance(python_value, _ErrorSentinel)
        rs_err = isinstance(rust_value, _ErrorSentinel)
        if py_err and rs_err:
            if python_value.kind != rust_value.kind:
                # Both errored, but with different exception classes.
                has_mismatch = True
                rows.append(
                    f"{name:<{metric_width}} {python_value!r:>10} {rust_value!r:>10} "
                    f"{'-':>10} {'-':>10} {'-':>10}  kind"
                )
                continue
            # Both errored with the same kind — symmetric. Pass.
            rows.append(f"{name:<{metric_width}} {'ERROR':>10} {'ERROR':>10} {'-':>10} {'-':>10} {'-':>10}    sym")
            continue
        if py_err != rs_err:
            # Asymmetric — one errored, the other didn't.
            has_mismatch = True
            py_repr = repr(python_value) if py_err else f"{python_value:.3f}"
            rs_repr = repr(rust_value) if rs_err else f"{rust_value:.3f}"
            rows.append(f"{name:<{metric_width}} {py_repr:>10} {rs_repr:>10} {'-':>10} {'-':>10} {'-':>10}  asym")
            continue
        # Both compute — apply numeric tolerance.
        allowed = max(abs(python_value) * rtol, 1e-9)
        delta = rust_value - python_value
        delta_pct = delta / abs(python_value) * 100 if python_value else float("inf")
        status = "drift" if abs(delta) > allowed else "ok"
        has_mismatch = has_mismatch or status == "drift"
        rows.append(
            f"{name:<{metric_width}} {python_value:>10.3f} {rust_value:>10.3f} "
            f"{delta:>10.3f} {delta_pct:>9.2f}% {rtol * 100:>9.2f}% {status:>6}"
        )
    if not has_mismatch:
        return None
    return "\n".join(
        [
            "parity drift (expected)",
            f"{'metric':<{metric_width}} {'python_ms':>10} {'rust_ms':>10} "
            f"{'delta_ms':>10} {'delta_pct':>10} {'tolerance':>10} {'status':>6}",
            *rows,
        ]
    )


def _static_comparison_metrics(case: EngineStepParityCase) -> dict[str, tuple[float, float]]:
    python_metrics = _static_metrics(case, engine_step_backend="python")
    rust_metrics = _static_metrics(case, engine_step_backend="rust")
    return {
        "static_ctx": (python_metrics["context_ms"], rust_metrics["context_ms"]),
        "static_gen": (python_metrics["generation_ms"], rust_metrics["generation_ms"]),
        "static_total": (python_metrics["total_ms"], rust_metrics["total_ms"]),
    }


def _mixed_step_comparison_metrics(
    case: EngineStepParityCase,
) -> dict[str, tuple[float | _ErrorSentinel, float | _ErrorSentinel]]:
    return {
        "mixed_step": (
            _safe_value(lambda: _python_mixed_step_ms(case)),
            _safe_value(lambda: _rust_mixed_step_ms(case)),
        ),
    }


def _agg_comparison_metrics(case: EngineStepParityCase) -> dict[str, tuple[float, float]]:
    python_metrics = _agg_metrics(case, engine_step_backend="python")
    rust_metrics = _agg_metrics(case, engine_step_backend="rust")
    return {
        "agg_ttft": (python_metrics["ttft_ms"], rust_metrics["ttft_ms"]),
        "agg_tpot": (python_metrics["tpot_ms"], rust_metrics["tpot_ms"]),
        "agg_request": (python_metrics["request_latency_ms"], rust_metrics["request_latency_ms"]),
    }


def _disagg_comparison_metrics(case: EngineStepParityCase) -> dict[str, tuple[float, float]]:
    python_metrics = _disagg_metrics(case, engine_step_backend="python")
    rust_metrics = _disagg_metrics(case, engine_step_backend="rust")
    return {
        "disagg_ttft": (python_metrics["ttft_ms"], rust_metrics["ttft_ms"]),
        "disagg_tpot": (python_metrics["tpot_ms"], rust_metrics["tpot_ms"]),
        "disagg_request": (python_metrics["request_latency_ms"], rust_metrics["request_latency_ms"]),
    }


def _prepare_rust_core(monkeypatch: pytest.MonkeyPatch) -> None:
    # The live path is the compiled-engine ``EngineHandle`` (Python builds the
    # ``EngineSpec``, the PyO3 ``aiconfigurator_core`` extension executes it).
    # The legacy ctypes dylib is gone, so the only requirement is that the
    # maturin-built extension is importable.
    pytest.importorskip(
        "aiconfigurator_core",
        reason="maturin-built aiconfigurator_core extension is required "
        "(`uv run maturin develop -m aic-core/rust/aiconfigurator-core/Cargo.toml`)",
    )
    rust_engine_step._engine_handle_cache_clear()


class TestRustEngineStepStaticParity:
    @pytest.mark.parametrize("case", SMOKE_CASES)
    def test_smoke_parity(
        self,
        case: EngineStepParityCase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _prepare_rust_core(monkeypatch)

        reason = _parity_mismatch_reason(_static_comparison_metrics(case))
        assert reason is None, reason


class TestRustEngineStepMixedStepParity:
    @pytest.mark.parametrize("case", SMOKE_CASES)
    def test_smoke_parity(
        self,
        case: EngineStepParityCase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _prepare_rust_core(monkeypatch)

        reason = _parity_mismatch_reason(_mixed_step_comparison_metrics(case))
        assert reason is None, reason


class TestRustEngineStepAggParity:
    @pytest.mark.parametrize("case", SMOKE_CASES)
    def test_smoke_parity(
        self,
        case: EngineStepParityCase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _prepare_rust_core(monkeypatch)

        reason = _parity_mismatch_reason(_agg_comparison_metrics(case))
        assert reason is None, reason


class TestRustEngineStepDisaggParity:
    @pytest.mark.parametrize("case", SMOKE_CASES)
    def test_smoke_parity(
        self,
        case: EngineStepParityCase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _prepare_rust_core(monkeypatch)

        reason = _parity_mismatch_reason(_disagg_comparison_metrics(case))
        assert reason is None, reason


# Context-parallelism (CP) parity cases. CP is SGLang-only and shards prefill
# sequence tokens: token-major ops divide their per-rank token count by cp
# (seq_split), ContextAttention models rank-0's zigzag chunk split, and
# MoEDispatch all-gathers (pre) / reduce-scatters (combine) the CP-sharded
# tokens. sglang CP requires tp_size=1 and attention_dp_size=1, so the width
# (tp*dp*cp) is carried entirely by cp and matched by moe_tp*moe_ep.
#
# Validated on the mix-step surface: the prefill chunk exercises the CP ops
# (context attention, GEMMs, comm, MoE dispatch). Without the Rust CP support
# these cases drift (Rust would evaluate the cp>1 config as cp=1); with it the
# Python and Rust engine steps match within PARITY_RTOL.
CP_CASES = [
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3-235B-A22B",
            system_name="b200_sxm",
            backend_name="sglang",
            backend_version="0.5.10",
            tp_size=1,
            attention_dp_size=1,
            moe_tp_size=8,
            moe_ep_size=1,
            cp_size=8,
        ),
        id="qwen3-235b-a22b-b200-sglang-0510-cp8",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3-235B-A22B",
            system_name="b200_sxm",
            backend_name="sglang",
            backend_version="0.5.10",
            tp_size=1,
            attention_dp_size=1,
            moe_tp_size=4,
            moe_ep_size=1,
            cp_size=4,
        ),
        id="qwen3-235b-a22b-b200-sglang-0510-cp4",
    ),
    # MLA context-parallelism: Kimi is MLA with bfloat16 FMHA (collected on
    # sglang), so it exercises the ContextMLA cp zigzag sharding. (DeepSeek-R1
    # would need the uncollected fp8-FMHA context-MLA slice; DSA/dsv4 CP need
    # uncollected sparse mqa/topk tables — both out of scope until collected.)
    pytest.param(
        EngineStepParityCase(
            model_path="moonshotai/Kimi-K2.5",
            system_name="b200_sxm",
            backend_name="sglang",
            backend_version="0.5.10",
            tp_size=1,
            attention_dp_size=1,
            moe_tp_size=8,
            moe_ep_size=1,
            cp_size=8,
        ),
        id="kimi-k25-b200-sglang-0510-cp8",
    ),
]


class TestRustEngineStepCpMixedStepParity:
    """CP parity on the mix-step surface (CP ops run in the prefill chunk)."""

    @pytest.mark.parametrize("case", CP_CASES)
    def test_cp_parity(
        self,
        case: EngineStepParityCase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _prepare_rust_core(monkeypatch)

        reason = _parity_mismatch_reason(_mixed_step_comparison_metrics(case))
        assert reason is None, reason


# HYBRID / EMPIRICAL parity cases: the util-space empirical layer
# (`sdk/operations/util_empirical.py`, PR #1238) ported to the compiled engine
# (issue #1333 §4.6 option b). Every case below was probed on the Python side
# with `capture_provenance()` so each transfer tier of the ladder
# (own-data empirical -> xshape -> xquant -> xprofile -> xop) is pinned by at
# least one real-data case, plus the two contract cases (HYBRID==SILICON
# invariance on covered configs; symmetric EmpiricalNotImplementedError on
# genuine coverage misses).
#
# These cases assert at a much tighter tolerance than the SILICON smoke suite.
# util-space empirical is close to silicon BY DESIGN on collected configs
# (offline study: ~1.9% mean APE, many rows < 1%), so at the 1% smoke rtol a
# Rust engine that silently ignored the mode and computed pure SILICON would
# still pass several EMPIRICAL cases. Both engines run the same f64 math over
# the same rows, so the faithful port agrees to ~1e-9; 1e-4 keeps headroom
# while making a silicon fallback (0.3-5% off) unmissable.
HYBRID_PARITY_RTOL = 1e-4
HYBRID_CASES = [
    # xop + xshape: MiniMax-M3 is all-MoE + MSA sparse attention with NO own
    # silicon data anywhere — MSA borrows DSA's util (xop, `msa.py`), the MoE
    # shape borrows the nearest collected sibling (xshape, `moe.py`). Probed:
    # static ctx/gen = 69.588/5.426 ms, tags {xop, xshape}.
    pytest.param(
        EngineStepParityCase(
            model_path="MiniMaxAI/MiniMax-M3",
            database_mode="HYBRID",
        ),
        id="minimax-m3-b200-vllm-019-hybrid-xop",
    ),
    # Same rescue on the sglang tables (DSA util source = sglang 0.5.14 data).
    # Probed: 39.490/2.994 ms, tags {xop, xshape}.
    pytest.param(
        EngineStepParityCase(
            model_path="MiniMaxAI/MiniMax-M3",
            backend_name="sglang",
            backend_version="0.5.14",
            database_mode="HYBRID",
        ),
        id="minimax-m3-b200-sglang-0514-hybrid-xop",
    ),
    # Transfer-policy gating: with xop disabled ("off" and "balanced" presets)
    # MSA must raise EmpiricalNotImplementedError on BOTH engines
    # (error-symmetry). Guards that the Rust port honours the policy at query
    # time instead of always transferring.
    pytest.param(
        EngineStepParityCase(
            model_path="MiniMaxAI/MiniMax-M3",
            database_mode="HYBRID",
            transfer_policy="off",
        ),
        id="minimax-m3-b200-vllm-019-hybrid-policy-off",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="MiniMaxAI/MiniMax-M3",
            database_mode="HYBRID",
            transfer_policy="balanced",
        ),
        id="minimax-m3-b200-vllm-019-hybrid-policy-balanced",
    ),
    # xquant: forced MoE quant w4a16_mxfp4_cutlass is uncollected on
    # b200/vllm/0.19.0 but shares the (memory=0.5, compute=1) profile with
    # collected int4_wo / w4a16_mxfp4 — the ladder lands on the xquant tier.
    # Probed: 90.578/10.908 ms, tags {xquant}.
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3-235B-A22B",
            database_mode="HYBRID",
            moe_quant_mode="w4a16_mxfp4_cutlass",
        ),
        id="qwen3-235b-a22b-b200-vllm-019-hybrid-xquant",
    ),
    # xprofile: forced MoE quant w4afp8 (memory=0.5, compute=2) has NO
    # collected same-profile sibling on b200/vllm/0.19.0 — the ladder falls
    # through to the cross-profile tier with the util-level rescale.
    # Probed: 47.755/8.450 ms, tags {xprofile}.
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3-235B-A22B",
            database_mode="HYBRID",
            moe_quant_mode="w4afp8",
        ),
        id="qwen3-235b-a22b-b200-vllm-019-hybrid-xprofile",
    ),
    # Attention cross-head_size xshape: MiMo-V2-Flash has head_dim=192 while
    # b200/vllm/0.19.0 collected only {128, 256} — SILICON raises, HYBRID
    # borrows the nearest collected head_size (`attention.py` ctx + gen
    # reference grids). Probed: 33.253/3.499 ms, tags {xshape}.
    pytest.param(
        EngineStepParityCase(
            model_path="XiaomiMiMo/MiMo-V2-Flash",
            database_mode="HYBRID",
        ),
        id="mimo-v2-flash-b200-vllm-019-hybrid-attn-xshape",
    ),
    # HYBRID==SILICON invariance: Kimi-K2.5 on b200/vllm/0.19.0 is fully
    # covered by silicon data (probed worst-provenance = silicon, no empirical
    # tier fires). The hybrid layer must not perturb covered queries; this
    # pins Rust-HYBRID == Python-HYBRID (== SILICON) on a collected config.
    pytest.param(
        EngineStepParityCase(
            model_path="moonshotai/Kimi-K2.5",
            database_mode="HYBRID",
        ),
        id="kimi-k25-b200-vllm-019-hybrid-invariant",
    ),
    # Ladder miss: NVFP4 MoE on Hopper has no own-shape, cross-shape, or
    # sibling reference anywhere in the h200/vllm/0.19.0 tables — Python
    # raises EmpiricalNotImplementedError; the Rust port must fail the same
    # query point (error-symmetry), never fabricate a SOL/constant value.
    pytest.param(
        EngineStepParityCase(
            model_path="nvidia/MiniMax-M2.5-NVFP4",
            system_name="h200_sxm",
            database_mode="HYBRID",
        ),
        id="minimax-m25-nvfp4-h200-vllm-019-hybrid-miss",
    ),
    # EMPIRICAL mode: every data-backed op answers SOL(query)/util from its
    # own collected slice — the broadest guard of the ported util math (grid
    # build, k=2 IDW in normalized log space, per-axis boundary clamp) across
    # op families: dense GEMM+GQA, MoE, MLA, DSA (vllm + sglang), fp8_block
    # MoE, and the state-space (Mamba2 SOL-degradation) path.
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3-32B",
            database_mode="EMPIRICAL",
        ),
        id="qwen3-32b-b200-vllm-019-empirical",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3-235B-A22B",
            database_mode="EMPIRICAL",
        ),
        id="qwen3-235b-a22b-b200-vllm-019-empirical",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="moonshotai/Kimi-K2.5",
            database_mode="EMPIRICAL",
        ),
        id="kimi-k25-b200-vllm-019-empirical",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="deepseek-ai/DeepSeek-V3.2",
            database_mode="EMPIRICAL",
        ),
        id="deepseek-v32-b200-vllm-019-empirical",
    ),
    # Off-grid shape on purpose: at the smoke shape (isl=1024, b=1) every GLM-5
    # op lands exactly on collected grid points, where util reconstruction
    # returns SOL/util == measured — EMPIRICAL degenerates to SILICON and the
    # case cannot distinguish a mode-ignoring engine. isl=1536 separates the
    # two by ~3.1% (probed) while both still compute.
    pytest.param(
        EngineStepParityCase(
            model_path="zai-org/GLM-5",
            backend_name="sglang",
            backend_version="0.5.14",
            isl=1536,
            database_mode="EMPIRICAL",
        ),
        id="glm5-b200-sglang-0514-empirical",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="MiniMaxAI/MiniMax-M2.5",
            database_mode="EMPIRICAL",
        ),
        id="minimax-m25-b200-vllm-019-empirical",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="nvidia/Nemotron-H-56B-Base-8K",
            database_mode="EMPIRICAL",
        ),
        id="nemotron-h-56b-b200-vllm-019-empirical",
    ),
]


class TestRustEngineStepHybridStaticParity:
    @pytest.mark.parametrize("case", HYBRID_CASES)
    def test_hybrid_parity(
        self,
        case: EngineStepParityCase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _prepare_rust_core(monkeypatch)

        reason = _parity_mismatch_reason(_static_comparison_metrics(case), rtol=HYBRID_PARITY_RTOL)
        assert reason is None, reason


class TestRustEngineStepHybridMixedStepParity:
    @pytest.mark.parametrize("case", HYBRID_CASES)
    def test_hybrid_parity(
        self,
        case: EngineStepParityCase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _prepare_rust_core(monkeypatch)

        reason = _parity_mismatch_reason(_mixed_step_comparison_metrics(case), rtol=HYBRID_PARITY_RTOL)
        assert reason is None, reason


class TestRustEngineStepHybridAggParity:
    @pytest.mark.parametrize("case", HYBRID_CASES)
    def test_hybrid_parity(
        self,
        case: EngineStepParityCase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _prepare_rust_core(monkeypatch)

        reason = _parity_mismatch_reason(_agg_comparison_metrics(case), rtol=HYBRID_PARITY_RTOL)
        assert reason is None, reason


class TestRustEngineStepHybridDisaggParity:
    @pytest.mark.parametrize("case", HYBRID_CASES)
    def test_hybrid_parity(
        self,
        case: EngineStepParityCase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _prepare_rust_core(monkeypatch)

        reason = _parity_mismatch_reason(_disagg_comparison_metrics(case), rtol=HYBRID_PARITY_RTOL)
        assert reason is None, reason


def _rust_static_breakdown(case: EngineStepParityCase):
    """Drive the rust engine-step bridge directly (no cli_estimate error
    wrapping) so the exception object crossing the FFI is what the test sees."""
    database = _case_database(case)
    model = _quiet_call(get_model, case.model_path, _case_model_config(case), case.backend_name)
    runtime_config = config.RuntimeConfig(
        batch_size=case.batch_size,
        beam_width=1,
        isl=case.isl,
        osl=case.osl,
        prefix=case.prefix,
    )
    return rust_engine_step.estimate_static_latency_breakdown_with_rust(
        model, database, runtime_config, "static", 1, 1.0
    )


class TestRustTypedErrorsAcrossFfi:
    """FFI typed-error contract: `aic_to_py` used to map
    every `AicError` to `ValueError`, so Python-side classifiers
    (`perf_database.has_perf_data_not_available_cause`, the support-matrix
    HYBRID-miss triage on `EmpiricalNotImplementedError`) could not recognize
    rust-path misses. The boundary now raises the canonical
    `aiconfigurator.sdk.errors` classes for the typed variants."""

    def test_silicon_data_gap_raises_typed_perf_data_miss(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # NVFP4 GEMM tables are not collected on h200/vllm/0.19.0: under
        # SILICON, Rust hits `AicError::PerfDatabase` ("GEMM perf data missing
        # for quant 'nvfp4'") at the query point — which must cross the FFI as
        # the SAME sdk class Python raises, recognized by the cause-chain
        # walker (the miss-classification the sweep/support-matrix rely on).
        _prepare_rust_core(monkeypatch)
        case = EngineStepParityCase(model_path="nvidia/MiniMax-M2.5-NVFP4", system_name="h200_sxm")
        with pytest.raises(errors.PerfDataNotAvailableError) as excinfo:
            _rust_static_breakdown(case)
        assert perf_database.has_perf_data_not_available_cause(excinfo.value)
        # The AicError display prefix pins the raise to the rust side of the
        # FFI (a Python-side miss would carry the sdk's own wording).
        message = str(excinfo.value)
        assert "perf database error" in message or "I/O error" in message, message

    def test_hybrid_ladder_miss_raises_typed_empirical_miss(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # NVFP4 MoE on Hopper: no own-shape/cross-shape/sibling reference
        # anywhere (the HYBRID_CASES ladder-miss config). Rust raises
        # `AicError::EmpiricalNotImplemented`, which must surface as the sdk's
        # EmpiricalNotImplementedError — the typed hybrid-miss — and NOT be
        # classified as a plain perf-data miss.
        _prepare_rust_core(monkeypatch)
        case = EngineStepParityCase(
            model_path="nvidia/MiniMax-M2.5-NVFP4",
            system_name="h200_sxm",
            database_mode="HYBRID",
        )
        with pytest.raises(errors.EmpiricalNotImplementedError) as excinfo:
            _rust_static_breakdown(case)
        message = str(excinfo.value)
        assert "empirical estimation not implemented" in message, message
        assert not perf_database.has_perf_data_not_available_cause(excinfo.value)


class TestRustProvenanceCapture:
    """FFI provenance contract: the compiled engine records
    the empirical tier that fired (max-rank, mirroring Python's
    `PROVENANCE_ORDER`) and the bridge forwards it into
    `util_empirical.capture_provenance`, so support-matrix HYBRID_PASS tier
    labelling works identically for rust-routed runs."""

    def test_hybrid_xop_run_records_tier(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # MiniMax-M3 HYBRID: MSA borrows DSA's util (xop) — the run's worst
        # tier. Python probes record {xop, xshape}; the rust path must land on
        # the same worst_provenance.
        _prepare_rust_core(monkeypatch)
        case = EngineStepParityCase(model_path="MiniMaxAI/MiniMax-M3", database_mode="HYBRID")
        with util_empirical.capture_provenance() as tags:
            metrics = _static_metrics(case, engine_step_backend="rust")
        assert not isinstance(metrics["total_ms"], _ErrorSentinel), repr(metrics)
        assert util_empirical.worst_provenance(tags) == "xop", tags

    def test_pure_silicon_run_records_nothing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Fully-collected config on SILICON: no empirical path fires, so the
        # capture stays empty and worst_provenance defaults to "silicon".
        _prepare_rust_core(monkeypatch)
        case = EngineStepParityCase(model_path="moonshotai/Kimi-K2.5")
        with util_empirical.capture_provenance() as tags:
            metrics = _static_metrics(case, engine_step_backend="rust")
        assert not isinstance(metrics["total_ms"], _ErrorSentinel), repr(metrics)
        assert util_empirical.worst_provenance(tags) == "silicon", tags
