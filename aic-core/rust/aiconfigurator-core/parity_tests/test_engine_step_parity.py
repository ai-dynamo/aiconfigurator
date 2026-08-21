# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke parity checks for the Rust engine step versus the frozen reference.

Each test compares the LIVE Rust value for the surface under test
(static_ctx + static_gen, the agg/disagg pipelines through `cli_estimate`,
the mix-step composition) against a FROZEN golden fixture. The goldens were
captured from the Python engine step while that path was still alive
(dedup-plan Gate 2); with the Python step removed, they are the permanent
regression oracle. A drift outside ``PARITY_RTOL`` fails the assertion with
a per-metric delta report so the failure mode is informative.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

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
    # Speculative block size (DSPARK/MTP). nextn > 0 switches the model's
    # generation ops onto the verify tables — for Kimi-K3 that crosses the
    # fused CuTeDSL verify reroute and the donor-absence contract it depends
    # on (kda_perf ABSENCE_LOAD_BEARING manifest exclusion) in CI.
    nextn: int = 0
    # Power-carrying identities only: extend the static surface with the
    # context/generation energy sums and the summary power averages. Leave
    # False on identities whose perf tables ship no power columns — there the
    # sums are 0.0 on both engines and the comparison would be vacuous.
    compare_energy: bool = False
    # AFD (attention-FFN disaggregation) topology, consumed only by the "afd"
    # surface (AFD_CASES). Names mirror the cli_estimate kwargs that
    # cli/main.py maps the --n-a-nodes / --n-f-nodes / --a-tp-size /
    # --a-batch-size / --f-moe-ep-size flags onto (afd_-prefixed here to keep
    # the case namespace readable); defaults track cli_estimate's.
    afd_n_a_nodes: int = 1
    afd_n_f_nodes: int = 1
    afd_a_tp_size: int = 1
    afd_a_batch_size: int = 128
    afd_f_moe_ep_size: int = 1
    # Attention kernel-lane override (`ModelConfig.attention_backend`,
    # AIC-1715/1716). `None` = no override, i.e. the framework-default lane
    # heads the precedence order. The op carries the RESOLVED walk order into
    # the spec, so this is the knob that makes Rust and Python have to agree on
    # which kernel's measurements answer a query.
    attention_backend: str | None = None


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
    # Phase 4 D7-C: Llama-4 Scout was originally a data-gap case, but the
    # tracked perf data now gives it full numeric parity on all four surfaces.
    pytest.param(
        EngineStepParityCase(model_path="meta-llama/Llama-4-Scout-17B-16E-Instruct"),
        id="llama4-scout-b200-vllm-019-isl1024-osl2",
    ),
    # DeepSeek-V4 Flash remains a data-gap case. Python errors with
    # `PerfDataNotAvailableError` because the perf DB doesn't ship the
    # required tables for this shape; Rust errors at the equivalent
    # query point (`AicError::PerfDatabase`). The error-symmetry contract
    # asserts both engines fail together — same outcome, even if the
    # exact failure point in the op graph differs.
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
    #     (`tp=8, moe_expert_compute=8, isl=1024, osl=2`) error symmetrically in both
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
    # Kimi-K3 (review Blocker 1 anchor): hybrid KDA + MLA LatentMoE. The
    # case defaults (tp8/ep8) put KDA on the fused 12-head shard — the exact
    # config the per-key kda_fused_decode routing and the exact-first mla_bmm
    # routing serve, freezing the K3 engine-step numbers themselves (the
    # donor-injection class of regression shifts BOTH engines through the
    # shared serialized source chain, so only a committed anchor like this
    # catches a silent shift in CI).
    pytest.param(
        EngineStepParityCase(
            model_path="moonshotai/Kimi-K3",
            system_name="b300_sxm",
            backend_name="sglang",
            backend_version="0.5.16",
        ),
        id="kimi-k3-b300-sglang-0516-nospec",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="moonshotai/Kimi-K3",
            backend_name="vllm",
            backend_version="0.1.dev19262",
        ),
        id="kimi-k3-b200-vllm-dev19262-nospec",
    ),
    # DSPARK speculative case: nextn=7 -> verify width 8 (the fused CuTeDSL
    # verify kernel's collected draft-width cap), crossing the fused verify
    # reroute + conv fold-to-zero end-to-end.
    pytest.param(
        EngineStepParityCase(
            model_path="moonshotai/Kimi-K3",
            system_name="b300_sxm",
            backend_name="sglang",
            backend_version="0.5.16",
            nextn=7,
        ),
        id="kimi-k3-b300-sglang-0516-dspark-nextn7",
    ),
    # Attention kernel-lane coverage (AIC-1715/1716). b200_sxm/sglang/0.5.14
    # collects three lanes for the dense attention ops (trtllm_mha, triton,
    # flashinfer) and 0.5.14 is the first version with a framework-default map
    # entry, so Qwen3.5-27B's full-attention layers walk a REAL multi-lane
    # table: the no-override case heads on the map lane (triton) and gap-fills
    # from trtllm_mha, the override case pins trtllm_mha first. Python resolves
    # the walk order and Rust replays it verbatim off the op spec, so a drift
    # in either the resolver or the replay shows up here as a latency split.
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3.5-27B",
            backend_name="sglang",
            backend_version="0.5.14",
        ),
        id="qwen35-27b-b200-sglang-0514-lanes-default",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3.5-27B",
            backend_name="sglang",
            backend_version="0.5.14",
            attention_backend="trtllm_mha",
        ),
        id="qwen35-27b-b200-sglang-0514-lanes-trtllm-mha",
    ),
    # AIC-1762: Qwen3.8-Max (Qwen3.5-family hybrid GDN + full-attention,
    # 2.4T-A95B) on gb300/sglang/0.5.17 -- the first end-to-end exercise of
    # the sparse 0.5.17 data design (moe/gdn/gemm collected; attention/comm/
    # quantize answer through shared-layer sibling inheritance from 0.5.14,
    # perf_database.get_database docstring). nextn: resolve_nextn_auto reads
    # num_nextn_predict_layers from the checkpoint's flat config (no
    # text_config nesting, common.py:716-718) and finds it absent -> 0; both
    # cases pin nextn=0 explicitly (the case struct's own field) rather than
    # relying silently on the dataclass default, so the "no MTP" resolution
    # is a reviewable decision, not an omission.
    #
    # fp8 lane (checkpoint-native fp8_block, auto-inferred from the bundled
    # -FP8 config's weight_block_size -- utils.py:1213-1215): tp=8 (every
    # other smoke case's default) OOMs -- the fp8_block checkpoint is
    # ~2.4TB of weights (1 byte/param) over 8 GPUs. tp=16 (still GDN-head-
    # divisible: 16 % 16 == 0, 128 % 16 == 0) is the smallest power-of-two
    # width that fits; moe_ep bumped to 16 to keep tp*dp*cp == moe_tp*moe_ep.
    # "agg-flavored": bumps agg_batch_size well past the trivial default (2)
    # so the IFB batch dimension is genuinely exercised at this model's scale.
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3.8-2.4T-A95B-FP8",
            system_name="gb300",
            backend_name="sglang",
            backend_version="0.5.17",
            tp_size=16,
            moe_tp_size=1,
            moe_ep_size=16,
            agg_batch_size=32,
            nextn=0,
        ),
        id="qwen38-max-gb300-sglang-0517-fp8-agg",
    ),
    # nvfp4 lane: the bf16 base checkpoint (no quantization_config) forced
    # onto nvfp4 MoE via the explicit override -- same pattern as the
    # existing xquant/xprofile HYBRID_CASES below (moe_quant_mode is the
    # only quant knob the case struct exposes; gemm/attention/kv stay at
    # their checkpoint-inferred default, bfloat16 here). nvfp4's 0.5625
    # bytes/param (vs fp8_block's 1) is small enough that the default tp=8
    # fits without adjustment. "disagg-flavored": bumps the disagg worker/
    # batch knobs past their trivial defaults (1 prefill worker, 4-request
    # decode batch) for a genuine disagg-shaped profile.
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3.8-2.4T-A95B",
            system_name="gb300",
            backend_name="sglang",
            backend_version="0.5.17",
            moe_quant_mode="nvfp4",
            disagg_prefill_num_workers=2,
            disagg_decode_batch_size=8,
            disagg_decode_num_workers=2,
            nextn=0,
        ),
        id="qwen38-max-gb300-sglang-0517-nvfp4-disagg",
    ),
    # AIC-1782: the vllm/0.27.1 twin of the sglang 0.5.17 cases above -- the
    # first end-to-end exercise of the sparse vllm/0.27.1 data design (moe/
    # gdn/gemm collected on Blackwell; attention/comm/quantize answer through
    # shared-layer sibling inheritance from 0.24.0). nextn: same resolution
    # as the sglang pair -- resolve_nextn_auto (config_builders.py) finds no
    # num_nextn_predict_layers in the checkpoint's flat config -> 0; pinned
    # explicitly so the "no MTP" decision stays reviewable.
    #
    # fp8 lane (checkpoint-native fp8_block from the bundled -FP8 config's
    # weight_block_size -- utils.py:1213-1215): tp=16 mirrors both the sglang
    # case's footprint rationale (~2.4TB of fp8 weights; tp=8 OOMs on the
    # agg/disagg memory gate here too) and the deployment reference (the
    # dynamo qwen3.8-2.4t-a95b-fp8 recipe deploys vllm at TP16); moe_ep=16
    # keeps tp*dp*cp == moe_tp*moe_ep. MoE prices from the collected
    # flashinfer_trtllm fp8 monolithic lane, GDN decode from the collected
    # fused_recurrent packed-decode rows (load-time label canonicalization),
    # and the GDN CONTEXT scan from the collected vllm
    # chunk_gated_delta_rule_flashinfer rows -- query_gdn's physical-lane
    # alias walk covers vllm 0.27.1 (state_space.rs, AIC-1782 V5b; the V5a
    # smoke had caught it resolving the sglang 0.5.17 chunk_gated_delta_rule
    # donor rows while the walk was still version-pinned to 0.24.0).
    # "agg-flavored" like its sglang twin: agg_batch_size well past the
    # trivial default.
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3.8-2.4T-A95B-FP8",
            system_name="gb300",
            backend_name="vllm",
            backend_version="0.27.1",
            tp_size=16,
            moe_tp_size=1,
            moe_ep_size=16,
            agg_batch_size=32,
            nextn=0,
        ),
        id="qwen38-max-gb300-vllm-0271-fp8-agg",
    ),
    # bfloat16 lane -- NOT nvfp4 like the sglang twin: the RadixArk NVFP4
    # row's vllm quant gate is fail-closed (allowed_modes [], AIC-1782 Task
    # V2) and only the base id was collected, so there is no vllm/0.27.1
    # nvfp4 moe silicon to pin against; bfloat16 is the other collected lane
    # (the flashinfer_trtllm bf16 monolithic kernel). Parallelism: the bf16
    # checkpoint is ~4.9TB (2 bytes/param), over tp=16 that is ~306GB/GPU >
    # gb300's 284GB -- the agg/disagg memory gate raises OOM -- and tp=32 has
    # ZERO collected moe rows (the kernel family rejects local inter_size
    # 2048/32 = 64 < 128), so the case goes tp=16 x pp=2 (~153GB/GPU): pp is
    # a modeled axis of the case struct (first SMOKE case to exercise pp>1),
    # and pp splits layers across stages without touching the per-stage MoE
    # width, so lookups stay on the collected tp<=16 grid and
    # tp*dp*cp == moe_tp*moe_ep still holds at 16. "disagg-flavored" like
    # its sglang twin: same non-trivial prefill/decode worker+batch knobs.
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3.8-2.4T-A95B",
            system_name="gb300",
            backend_name="vllm",
            backend_version="0.27.1",
            tp_size=16,
            pp_size=2,
            moe_tp_size=1,
            moe_ep_size=16,
            disagg_prefill_num_workers=2,
            disagg_decode_batch_size=8,
            disagg_decode_num_workers=2,
            nextn=0,
        ),
        id="qwen38-max-gb300-vllm-0271-bf16-disagg",
    ),
]

PARITY_RTOL = 0.01


# Power/energy parity cases: one per power-carrying database identity (the
# only shipped identities whose perf parquets carry measured power columns:
# b200_sxm/vllm/0.22.0, b200_sxm/trtllm/1.3.0rc15, gb200/vllm/0.22.0,
# gb200/trtllm/1.3.0rc17, h200_sxm/vllm/0.22.0). Every SMOKE_CASE sits on
# 0.19.0 / 0.5.x / 1.3.0rc10 tables, which are latency-only — without these
# cases the per-op energy that now crosses the FFI (PR-2) would have ZERO
# numeric coverage in the parity suites. ``compare_energy=True`` extends the
# static surface with the context/generation energy sums and the summary
# power averages; the mixed/agg/disagg surfaces run at the standard latency
# metrics. Models verified to have perf data on these versions (probed on
# the live Python engine): dense Qwen3-32B on four identities, MoE
# Qwen3-30B-A3B on b200_sxm/vllm/0.22.0 (gb200/trtllm ships no MoE tables at
# 1.3.0rc17, so that identity stays dense).
POWER_CASES = [
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3-30B-A3B",
            backend_version="0.22.0",
            tp_size=4,
            moe_ep_size=4,
            compare_energy=True,
        ),
        id="qwen3-30b-a3b-b200-vllm-022-power",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3-32B",
            backend_name="trtllm",
            backend_version="1.3.0rc15",
            compare_energy=True,
        ),
        id="qwen3-32b-b200-trtllm-130rc15-power",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3-32B",
            system_name="gb200",
            backend_version="0.22.0",
            compare_energy=True,
        ),
        id="qwen3-32b-gb200-vllm-022-power",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3-32B",
            system_name="gb200",
            backend_name="trtllm",
            backend_version="1.3.0rc17",
            compare_energy=True,
        ),
        id="qwen3-32b-gb200-trtllm-130rc17-power",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3-32B",
            system_name="h200_sxm",
            backend_version="0.22.0",
            compare_energy=True,
        ),
        id="qwen3-32b-h200-vllm-022-power",
    ),
]


# Site-transfer tie-break anchors (issue #1456): when a GEMM query lands off
# the collected grid, Python resolves equidistant candidate sites through a
# *stable* sort over the index's enumeration order; the pre-fix Rust selection
# broke that tie differently and silently drifted these exact configs. Each
# case pins the surface that originally exposed the divergence, so the
# tie-break stays anchored in CI:
#  - Qwen3-32B-FP8 @ h200_sxm/vllm/0.19.0, agg: off-grid fp8_block GEMM
#    n=1280 k=5120 (the tp8 fused-QKV projection).
#  - Llama-4-Scout @ gb200/vllm/0.24.0, disagg decode: the mirror bf16 shape
#    n=5120 k=1280 (the tp4 attention out-projection; tp4 * moe_ep4 keeps the
#    16-expert MoE mapping valid).
TIE_AGG_CASES = [
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3-32B-FP8",
            system_name="h200_sxm",
        ),
        id="qwen3-32b-fp8-h200-vllm-019-tiebreak-agg",
    ),
]
TIE_DISAGG_CASES = [
    pytest.param(
        EngineStepParityCase(
            model_path="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            system_name="gb200",
            backend_version="0.24.0",
            tp_size=4,
            moe_ep_size=4,
        ),
        id="llama4-scout-gb200-vllm-024-tiebreak-disagg",
    ),
]
TIE_CASES = [*TIE_AGG_CASES, *TIE_DISAGG_CASES]


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

    @classmethod
    def from_kind(cls, kind: str) -> _ErrorSentinel:
        """Rehydrate a sentinel from a golden-recorded exception class name.

        Goldens persist only ``kind`` (the error-symmetry contract matches on
        the exception class, never the message), so the message is a fixed
        marker here.
        """
        sentinel = cls.__new__(cls)
        sentinel.kind = kind
        sentinel.message = "recorded in golden fixture"
        return sentinel

    def __repr__(self) -> str:
        return f"ERROR({self.kind}: {self.message})"


class _MemoizedCall:
    """Memoize a zero-arg thunk *including* a raised exception, so several
    metric extractions can share one ``cli_estimate`` result while each
    metric's ``_safe_value`` still observes the same error kind."""

    __slots__ = ("_done", "_exc", "_thunk", "_value")

    def __init__(self, thunk) -> None:
        self._thunk = thunk
        self._value = None
        self._exc: BaseException | None = None
        self._done = False

    def __call__(self):
        if not self._done:
            try:
                self._value = self._thunk()
            except Exception as exc:
                self._exc = exc
            self._done = True
        if self._exc is not None:
            raise self._exc
        return self._value


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
        # The compiled engine is the only step executor; pinned so an ambient
        # env override can never flip the harness off it.
        "engine_step_backend": "rust",
        "database_mode": case.database_mode,
        "transfer_policy": case.transfer_policy,
        "moe_quant_mode": case.moe_quant_mode,
        "attention_backend": case.attention_backend,
    }
    ctx_result = _MemoizedCall(lambda: _quiet_call(cli_estimate, mode="static_ctx", **kwargs))
    gen_result = _MemoizedCall(lambda: _quiet_call(cli_estimate, mode="static_gen", **kwargs))
    context_ms = _safe_value(lambda: ctx_result().summary.get_summary_df().iloc[0]["context_latency"])
    generation_ms = _safe_value(lambda: gen_result().summary.get_summary_df().iloc[0]["generation_latency"])
    if isinstance(context_ms, _ErrorSentinel) or isinstance(generation_ms, _ErrorSentinel):
        total: float | _ErrorSentinel = context_ms if isinstance(context_ms, _ErrorSentinel) else generation_ms
    else:
        total = context_ms + generation_ms
    metrics: dict[str, float | _ErrorSentinel] = {
        "context_ms": context_ms,
        "generation_ms": generation_ms,
        "total_ms": total,
    }
    if case.compare_energy:
        # Power-carrying identities only: the phase energy sums (W*ms, the
        # per-op values the FFI folds into the summary dicts) and the summary
        # power averages, from the same memoized cli_estimate results.
        metrics["ctx_energy_wms"] = _safe_value(
            lambda: sum(ctx_result().summary.get_context_energy_wms_dict().values())
        )
        metrics["gen_energy_wms"] = _safe_value(
            lambda: sum(gen_result().summary.get_generation_energy_wms_dict().values())
        )
        metrics["ctx_power_w"] = _safe_value(lambda: ctx_result().summary.get_context_power_avg())
        metrics["gen_power_w"] = _safe_value(lambda: gen_result().summary.get_generation_power_avg())
    return metrics


def _agg_metrics(case: EngineStepParityCase) -> dict[str, float | _ErrorSentinel]:
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
            engine_step_backend="rust",
            database_mode=case.database_mode,
            transfer_policy=case.transfer_policy,
            moe_quant_mode=case.moe_quant_mode,
            attention_backend=case.attention_backend,
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


def _disagg_metrics(case: EngineStepParityCase) -> dict[str, float | _ErrorSentinel]:
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
            engine_step_backend="rust",
            database_mode=case.database_mode,
            transfer_policy=case.transfer_policy,
            moe_quant_mode=case.moe_quant_mode,
            attention_backend=case.attention_backend,
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


def _afd_metrics(case: EngineStepParityCase) -> dict[str, float | _ErrorSentinel]:
    """AFD (attention-FFN disaggregation) estimate for the case's topology.

    Mirrors how ``cli/main.py`` maps the AFD flags onto
    ``cli_estimate(mode="afd", ...)`` (n_a_nodes / n_f_nodes / a_tp_size /
    a_batch_size / f_moe_ep_size; the remaining AFD knobs stay at their
    cli_estimate defaults: afd_phase="decode" + afd_combined_with_pd=True, so
    tpot comes from the AFD decode pipeline and ttft from the static-ctx
    complement). The AFD session sources its per-op values through the
    op-list evaluate FFI (``AFDInferenceSession._sum_latency``) when the
    routing gate allows; that internal RuntimeConfig carries no explicit
    ``engine_step_backend`` (the kwarg reaches only the static complement),
    so callers pin the AFD side via the ``AICONFIGURATOR_ENGINE_STEP_BACKEND``
    env var (the parity test sets "rust"; the retired golden capture exported
    "python" while that path was still alive).
    """

    def call():
        return _quiet_call(
            cli_estimate,
            mode="afd",
            model_path=case.model_path,
            system_name=case.system_name,
            backend_name=case.backend_name,
            backend_version=case.backend_version,
            batch_size=case.batch_size,
            isl=case.isl,
            osl=case.osl,
            prefix=case.prefix,
            tp_size=case.tp_size,
            pp_size=case.pp_size,
            attention_dp_size=case.attention_dp_size,
            moe_tp_size=case.moe_tp_size,
            moe_ep_size=case.moe_ep_size,
            n_a_nodes=case.afd_n_a_nodes,
            n_f_nodes=case.afd_n_f_nodes,
            a_tp_size=case.afd_a_tp_size,
            a_batch_size=case.afd_a_batch_size,
            f_moe_ep_size=case.afd_f_moe_ep_size,
            engine_step_backend="rust",
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
        return {"ttft_ms": err, "tpot_ms": err}
    return {
        "ttft_ms": float(result.ttft),
        "tpot_ms": float(result.tpot),
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
        nextn=case.nextn,
        attention_backend=case.attention_backend,
    )


def _cp_static_ctx_ms(case: EngineStepParityCase) -> float:
    """Context-phase static sum through the cp-aware model builder.

    The "static" surface goes through `cli_estimate`, which has no cp knob —
    a cp>1 case fails model validation there before any op runs. CP cases
    anchor the context phase (the only phase the CP composition runs on)
    with the same `_case_model_config` construction the mixed surface uses.
    (The Python-side values were captured into the goldens before the Python
    step path was removed; only the rust side runs live.)
    """
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
        engine_step_backend="rust",
    )
    ctx_lat, _ctx_e, _gen_lat, _gen_e, _ctx_src, _gen_src = _quiet_call(
        backend._run_static_breakdown, model, database, runtime_config, "static_ctx", 1
    )
    return float(sum(ctx_lat.values()))


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


# --------------------------------------------------------------------------- #
# Surface plumbing. A "surface" is one of the apples-to-apples comparison
# granularities; `_surface_metrics` maps a (case, surface) pair to the live
# rust comparison-metric dict, keyed by the names the golden fixtures persist.
# (The Python side of every fixture was captured by the retired
# `regenerate_goldens.py` while the Python step path was still alive; the
# goldens are frozen artifacts now — see README.)
# --------------------------------------------------------------------------- #

ENGINE_STEP_SURFACES = ("static", "mixed", "agg", "disagg")


def _surface_metrics(
    case: EngineStepParityCase,
    surface: str,
) -> dict[str, float | _ErrorSentinel]:
    """LIVE compiled-engine metrics for a (case, surface) pair.

    There is no engine selector anymore: the helpers pin
    ``engine_step_backend="rust"`` internally (passing "python" would be a
    silent rust-vs-rust self-comparison via the deprecation no-op — exactly
    the vacuous-comparison class the golden rewiring exists to prevent).
    """
    if surface == "static":
        metrics = _static_metrics(case)
        out: dict[str, float | _ErrorSentinel] = {
            "static_ctx": metrics["context_ms"],
            "static_gen": metrics["generation_ms"],
            "static_total": metrics["total_ms"],
        }
        if case.compare_energy:
            out["static_ctx_energy"] = metrics["ctx_energy_wms"]
            out["static_gen_energy"] = metrics["gen_energy_wms"]
            out["static_ctx_power"] = metrics["ctx_power_w"]
            out["static_gen_power"] = metrics["gen_power_w"]
        return out
    if surface == "mixed":
        return {"mixed_step": _safe_value(lambda: _rust_mixed_step_ms(case))}
    if surface == "cp_static_ctx":
        return {"cp_static_ctx": _safe_value(lambda: _cp_static_ctx_ms(case))}
    if surface == "agg":
        metrics = _agg_metrics(case)
        return {
            "agg_ttft": metrics["ttft_ms"],
            "agg_tpot": metrics["tpot_ms"],
            "agg_request": metrics["request_latency_ms"],
        }
    if surface == "disagg":
        metrics = _disagg_metrics(case)
        return {
            "disagg_ttft": metrics["ttft_ms"],
            "disagg_tpot": metrics["tpot_ms"],
            "disagg_request": metrics["request_latency_ms"],
        }
    if surface == "afd":
        metrics = _afd_metrics(case)
        return {
            "afd_ttft": metrics["ttft_ms"],
            "afd_tpot": metrics["tpot_ms"],
        }
    raise ValueError(f"unknown parity surface: {surface!r}")


def _comparison_metrics(
    case: EngineStepParityCase,
    surface: str,
) -> dict[str, tuple[float | _ErrorSentinel, float | _ErrorSentinel]]:
    """(python-golden, live-rust) pairs for every metric of the surface."""
    rust_metrics = _surface_metrics(case, surface)
    python_metrics = _golden_python_metrics(case, surface, rust_metrics)
    return {name: (python_metrics[name], rust_metrics[name]) for name in rust_metrics}


def _static_comparison_metrics(case: EngineStepParityCase) -> dict[str, tuple[float, float]]:
    return _comparison_metrics(case, "static")


def _mixed_step_comparison_metrics(
    case: EngineStepParityCase,
) -> dict[str, tuple[float | _ErrorSentinel, float | _ErrorSentinel]]:
    return _comparison_metrics(case, "mixed")


def _agg_comparison_metrics(case: EngineStepParityCase) -> dict[str, tuple[float, float]]:
    return _comparison_metrics(case, "agg")


def _disagg_comparison_metrics(case: EngineStepParityCase) -> dict[str, tuple[float, float]]:
    return _comparison_metrics(case, "disagg")


# --------------------------------------------------------------------------- #
# Golden fixtures (dedup-plan Gate 2, frozen at Gate 3). The Python side of
# every comparison is a FROZEN reference captured while the Python latency
# path was still alive; only the Rust side runs live. The Python-era records
# never regenerate (the capture path is gone with the Python step). New cases
# and deliberate rust-side modeling changes pin values from the live rust
# engine via `pin_goldens.py` — the golden diff is the review artifact.
# --------------------------------------------------------------------------- #

GOLDEN_DIR = Path(__file__).resolve().parent / "goldens"
_REGENERATE_HINT = (
    "pin the missing/changed records with "
    "`.venv/bin/python aic-core/rust/aiconfigurator-core/parity_tests/pin_goldens.py` "
    "(appends records from the live rust engine; see parity_tests/README.md)"
)
_GOLDEN_CACHE: dict[str, dict] = {}


def load_parity_golden(filename: str) -> dict:
    """Load (and memoize) one golden JSON from ``parity_tests/goldens/``."""
    cached = _GOLDEN_CACHE.get(filename)
    if cached is None:
        path = GOLDEN_DIR / filename
        if not path.is_file():
            pytest.fail(f"missing golden fixture {path}; {_REGENERATE_HINT}", pytrace=False)
        cached = _GOLDEN_CACHE[filename] = json.loads(path.read_text())
    return cached


def _case_golden_id(case: EngineStepParityCase) -> str:
    case_id = _CASE_GOLDEN_IDS.get(case)
    if case_id is None:
        pytest.fail(
            f"case {case!r} is not part of ENGINE_STEP_GOLDEN_MATRIX — add it to a golden case list; "
            f"{_REGENERATE_HINT}",
            pytrace=False,
        )
    return case_id


def _golden_python_metrics(
    case: EngineStepParityCase,
    surface: str,
    rust_metrics: dict[str, float | _ErrorSentinel],
) -> dict[str, float | _ErrorSentinel]:
    """Golden (frozen Python) metrics for a (case, surface) pair.

    A surface-level ``{"error": kind}`` record (every metric raised with one
    kind — the single-call-site agg/disagg wrapping) expands over the live
    comparison's metric names; a ``{"values": ...}`` record must carry exactly
    the same metric names the live side computed, else the fixture predates a
    metric-set change and needs regeneration.
    """
    case_id = _case_golden_id(case)
    record = load_parity_golden("engine_step.json")["cases"].get(case_id, {}).get(surface)
    if record is None:
        pytest.fail(
            f"no engine-step golden for case '{case_id}' surface '{surface}'; {_REGENERATE_HINT}",
            pytrace=False,
        )
    if "error" in record:
        kind = record["error"]
        return {name: _ErrorSentinel.from_kind(kind) for name in rust_metrics}
    values = record["values"]
    if set(values) != set(rust_metrics):
        pytest.fail(
            f"golden metric names for '{case_id}::{surface}' ({sorted(values)}) do not match the live "
            f"comparison ({sorted(rust_metrics)}); {_REGENERATE_HINT}",
            pytrace=False,
        )
    return {
        name: (_ErrorSentinel.from_kind(value["error"]) if isinstance(value, dict) else float(value))
        for name, value in values.items()
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
    @pytest.mark.parametrize("case", [*SMOKE_CASES, *POWER_CASES])
    def test_smoke_parity(
        self,
        case: EngineStepParityCase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _prepare_rust_core(monkeypatch)

        reason = _parity_mismatch_reason(_static_comparison_metrics(case))
        assert reason is None, reason


class TestRustEngineStepMixedStepParity:
    @pytest.mark.parametrize("case", [*SMOKE_CASES, *POWER_CASES])
    def test_smoke_parity(
        self,
        case: EngineStepParityCase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _prepare_rust_core(monkeypatch)

        reason = _parity_mismatch_reason(_mixed_step_comparison_metrics(case))
        assert reason is None, reason


class TestRustEngineStepAggParity:
    @pytest.mark.parametrize("case", [*SMOKE_CASES, *POWER_CASES])
    def test_smoke_parity(
        self,
        case: EngineStepParityCase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _prepare_rust_core(monkeypatch)

        reason = _parity_mismatch_reason(_agg_comparison_metrics(case))
        assert reason is None, reason


class TestRustEngineStepDisaggParity:
    @pytest.mark.parametrize("case", [*SMOKE_CASES, *POWER_CASES])
    def test_smoke_parity(
        self,
        case: EngineStepParityCase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _prepare_rust_core(monkeypatch)

        reason = _parity_mismatch_reason(_disagg_comparison_metrics(case))
        assert reason is None, reason


class TestRustEngineStepTieBreakParity:
    """#1456 site-transfer tie-break anchors (see TIE_AGG/TIE_DISAGG_CASES):
    each trigger config runs on the surface that originally exposed the
    off-grid GEMM tie divergence, at the standard latency tolerance."""

    @pytest.mark.parametrize("case", TIE_AGG_CASES)
    def test_agg_tie_break_parity(
        self,
        case: EngineStepParityCase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _prepare_rust_core(monkeypatch)

        reason = _parity_mismatch_reason(_agg_comparison_metrics(case))
        assert reason is None, reason

    @pytest.mark.parametrize("case", TIE_DISAGG_CASES)
    def test_disagg_tie_break_parity(
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
# (tp*dp*cp) is carried entirely by cp and matched by moe_tp*moe_expert_compute.
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
    # would need the uncollected fp8-FMHA context-MLA slice; DSA CP needs
    # uncollected sparse mqa/topk tables — out of scope until collected.)
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

# DeepSeek-V4 CSA context-parallelism on a REUSE-carrying version (issue
# #1498): 0.5.12 ships no primary dsv4 sparse tables — the mqa/topk lookups
# resolve through the approved `reuse.yaml` donors (from_version 0.5.14) —
# so this case anchors the reuse-aware CP top_last loader on both engines
# AND the CP composition itself (the `_query_cp` full/cp deltas +
# all-gathers, plus the token-major mHC seq_split division whose absence in
# Rust was the original 34% static_ctx divergence). isl=8192 pins the
# adjudicated repro shape.
#
# Anchored on BOTH the `cp_static_ctx` surface (the adjudicated static repro,
# 41.737710 ms) and the mixed surface. cp_static_ctx was re-pinned from
# 42.430756 ms for AIC-1759's b200_sxm sglang/0.5.14 gemm_perf replacement:
# systems/data/b200_sxm/gemm/sglang/0.5.12/reuse.yaml declares gemm_perf
# from_version 0.5.14 as this case's own donor, so the deliberate GEMM data
# update shifts this anchor too, independent of the dsv4-sparse-table reuse
# the case otherwise pins (re-pinned via pin_goldens.py --refresh; mixed is
# unaffected, still bit-identical on both engines at the value below). The
# mixed anchor additionally pins the `run_mixed` pass-filter fix: Python's
# passes used to run the FULL op lists and discard the non-consumed values,
# so the generation-MoE singleton low-token miss in pass 3 (num_tokens=1,
# one measured point at 1024) raised on the Python side only, while the
# rust mixed step never queries the ops it does not consume. With the
# passes filtered to their consumed sets the case computes bit-identically
# on both engines (46.0492671363915 ms).
DSV4_CP_CASES = [
    pytest.param(
        EngineStepParityCase(
            model_path="deepseek-ai/DeepSeek-V4-Flash",
            system_name="b200_sxm",
            backend_name="sglang",
            backend_version="0.5.12",
            isl=8192,
            osl=8,
            tp_size=1,
            attention_dp_size=1,
            moe_tp_size=1,
            moe_ep_size=8,
            cp_size=8,
        ),
        id="dsv4-flash-b200-sglang-0512-cp8-reuse",
    ),
]


class TestRustEngineStepCpMixedStepParity:
    """CP parity on the mix-step surface (CP ops run in the prefill chunk)."""

    @pytest.mark.parametrize("case", [*CP_CASES, *DSV4_CP_CASES])
    def test_cp_parity(
        self,
        case: EngineStepParityCase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _prepare_rust_core(monkeypatch)

        reason = _parity_mismatch_reason(_mixed_step_comparison_metrics(case))
        assert reason is None, reason


class TestRustEngineStepCpStaticCtxParity:
    """CP parity on the context-phase static surface (issue #1498 anchor).

    Pins the adjudicated DSV4 CSA CP repro shape end to end on a
    reuse-carrying version: the composition's four sparse-gate lookups
    resolve through approved donors, the mHC ops divide by seq_split, and
    the static_ctx sums agree. The same case is ALSO exercised on the mixed
    surface (`TestRustEngineStepCpMixedStepParity` parametrizes
    DSV4_CP_CASES) — this surface exists because `cli_estimate` has no cp
    knob, so the plain "static" surface cannot express a cp>1 case.
    """

    @pytest.mark.parametrize("case", DSV4_CP_CASES)
    def test_cp_static_ctx_parity(
        self,
        case: EngineStepParityCase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _prepare_rust_core(monkeypatch)

        reason = _parity_mismatch_reason(_comparison_metrics(case, "cp_static_ctx"))
        assert reason is None, reason


class TestRustEngineHandleDatabasePolicyIdentity:
    """Handle-cache isolation across database-policy views (review finding).

    `build_engine_spec_json` bakes the database's policy-dependent
    `perf_db_sources` into the compiled handle, so a shared-layer-off view
    and a shared-layer-on view of the SAME on-disk identity must not alias
    one cached handle. The cache is cleared ONCE per ordering (not between
    the two queries — that is the point): whichever view warms the cache
    must not answer, or fail, for the other. Guards both directions on the
    adjudicated DSV4 CP shape: the false FAILURE (off-warmed cache raising
    for the reuse-carrying on-view) and the false SUCCESS (on-warmed cache
    computing for the primary-only off-view that must raise).
    """

    # The issue #1498 adjudicated static_ctx sum for DSV4_CP_CASES[0] (same
    # shape `TestRustEngineStepCpStaticCtxParity` pins as `cp_static_ctx`).
    # Re-pinned from 42.4307555484161 for AIC-1759's b200_sxm sglang/0.5.14
    # gemm_perf replacement -- see the DSV4_CP_CASES comment above for why a
    # deliberate GEMM data update shifts this reuse-carrying case's anchor.
    ANCHOR_MS = 41.73771024187921

    def _static_ctx_ms(self, model, view) -> float:
        rc = config.RuntimeConfig(batch_size=1, beam_width=1, isl=8192, osl=8, prefix=0, engine_step_backend="rust")
        ctx_latency, _gen, *_ = rust_engine_step.estimate_static_latency_breakdown_with_rust(
            model, view, rc, "static_ctx", 1, 1.0
        )
        return float(sum(ctx_latency.values()))

    def _build(self):
        (case,) = DSV4_CP_CASES[0].values
        model = _quiet_call(get_model, case.model_path, _case_model_config(case), case.backend_name)
        off = _quiet_call(
            perf_database.get_database_view,
            case.system_name,
            case.backend_name,
            case.backend_version,
            shared_layer=False,
        )
        on = _quiet_call(
            perf_database.get_database_view,
            case.system_name,
            case.backend_name,
            case.backend_version,
            shared_layer=True,
        )
        if off is None or on is None:
            pytest.skip("no perf database for the DSV4 CP case identity")
        return model, off, on

    def test_off_warmed_cache_does_not_fail_the_shared_on_view(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _prepare_rust_core(monkeypatch)  # the ONLY cache clear in this ordering
        model, off, on = self._build()
        with pytest.raises(errors.PerfDataNotAvailableError):
            self._static_ctx_ms(model, off)
        assert self._static_ctx_ms(model, on) == pytest.approx(self.ANCHOR_MS, rel=1e-9)

    def test_on_warmed_cache_does_not_answer_for_the_shared_off_view(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _prepare_rust_core(monkeypatch)  # the ONLY cache clear in this ordering
        model, off, on = self._build()
        assert self._static_ctx_ms(model, on) == pytest.approx(self.ANCHOR_MS, rel=1e-9)
        with pytest.raises(errors.PerfDataNotAvailableError):
            self._static_ctx_ms(model, off)


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
    # Full-model xprofile resolution: NVFP4 on Hopper has no collected GEMM
    # or MoE tables, but under the default (aggressive) policy the GEMM
    # quant-transfer ladder borrows across profiles (as does MoE's), so the
    # whole HYBRID breakdown computes on both sides — value parity. (Before
    # the GEMM ladder existed this case pinned the error-symmetric miss; the
    # miss contract moved to the "balanced" case below.)
    pytest.param(
        EngineStepParityCase(
            model_path="nvidia/MiniMax-M2.5-NVFP4",
            system_name="h200_sxm",
            database_mode="HYBRID",
        ),
        id="minimax-m25-nvfp4-h200-vllm-019-hybrid-xprofile",
    ),
    # Ladder miss (error-symmetry): with XPROFILE policy-disabled
    # ("balanced" = xshape+xquant), NVFP4 GEMM (profile (0.5625, 4)) has no
    # same-profile sibling anywhere in the h200/vllm/0.19.0 tables — Python
    # raises EmpiricalNotImplementedError; the Rust port must fail the same
    # query point, never fabricate a SOL/constant value.
    pytest.param(
        EngineStepParityCase(
            model_path="nvidia/MiniMax-M2.5-NVFP4",
            system_name="h200_sxm",
            database_mode="HYBRID",
            transfer_policy="balanced",
        ),
        id="minimax-m25-nvfp4-h200-vllm-019-hybrid-balanced-miss",
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


# SOL parity cases: the per-op speed-of-light dispatch ported with the
# SOL_FULL retirement (`DatabaseMode::Sol | SolFull` arms across the operator
# families; the routing gate no longer delegates SOL to the Python step).
# SOL is a pure analytic formula on BOTH sides — no tables, no interpolation,
# no transfer ladder — so the faithful port agrees to f64 rounding (~1e-12);
# HYBRID_PARITY_RTOL (1e-4) keeps headroom while making any silicon/hybrid
# fallthrough (orders of magnitude off the roofline) unmissable.
# Re-parameterized from existing SMOKE_CASES ids: two dense (Qwen3-32B,
# Llama-3.1-70B), one MoE (Qwen3-235B-A22B), one MLA (DeepSeek-V3 — also
# covers MLA BMM + the mode-aware mem_op extras).
SOL_CASES = [
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3-32B",
            database_mode="SOL",
        ),
        id="qwen3-32b-b200-vllm-019-sol",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="meta-llama/Meta-Llama-3.1-70B",
            database_mode="SOL",
        ),
        id="llama31-70b-b200-vllm-019-sol",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3-235B-A22B",
            database_mode="SOL",
        ),
        id="qwen3-235b-a22b-b200-vllm-019-sol",
    ),
    pytest.param(
        EngineStepParityCase(
            model_path="deepseek-ai/DeepSeek-V3",
            database_mode="SOL",
        ),
        id="deepseek-v3-b200-vllm-019-sol",
    ),
]


class TestRustEngineStepSolStaticParity:
    @pytest.mark.parametrize("case", SOL_CASES)
    def test_sol_parity(
        self,
        case: EngineStepParityCase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _prepare_rust_core(monkeypatch)

        reason = _parity_mismatch_reason(_static_comparison_metrics(case), rtol=HYBRID_PARITY_RTOL)
        assert reason is None, reason


class TestRustEngineStepSolMixedStepParity:
    @pytest.mark.parametrize("case", SOL_CASES)
    def test_sol_parity(
        self,
        case: EngineStepParityCase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _prepare_rust_core(monkeypatch)

        reason = _parity_mismatch_reason(_mixed_step_comparison_metrics(case), rtol=HYBRID_PARITY_RTOL)
        assert reason is None, reason


# AFD (attention-FFN disaggregation) parity case: the AFD orchestration (A/F
# partitioning, ping-pong pipeline, comm ops) is Python-side permanently, but
# the per-op values it sums come from the compiled engine through the op-list
# evaluate FFI (`AFDInferenceSession._sum_latency`). rust==python was verified
# manually (bit-identical) when that sourcing landed; this case pins it in CI.
# One MoE model with AFD support on a version with data: Qwen3-30B-A3B on
# h200_sxm/vllm/0.19.0, one A node + one F node (a_tp=4, a_batch=32,
# f_moe_ep=8) — verified end-to-end through `cli_estimate(mode="afd", ...)`.
AFD_CASES = [
    pytest.param(
        EngineStepParityCase(
            model_path="Qwen/Qwen3-30B-A3B",
            system_name="h200_sxm",
            tp_size=4,
            moe_ep_size=4,
            afd_n_a_nodes=1,
            afd_n_f_nodes=1,
            afd_a_tp_size=4,
            afd_a_batch_size=32,
            afd_f_moe_ep_size=8,
        ),
        id="qwen3-30b-a3b-h200-vllm-019-afd",
    ),
]


class TestRustEngineStepAfdParity:
    """AFD parity anchor (ttft from the static-ctx complement, tpot from the
    AFD decode pipeline whose per-op values cross the evaluate FFI)."""

    @pytest.mark.parametrize("case", AFD_CASES)
    def test_afd_parity(
        self,
        case: EngineStepParityCase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _prepare_rust_core(monkeypatch)
        # The AFD session's internal RuntimeConfig carries no explicit
        # engine_step_backend (cli_estimate's kwarg reaches only the static
        # complement), so pin the env-var backstop: the live side must stay
        # on the compiled engine even under an ambient =python override.
        monkeypatch.setenv(rust_engine_step.ENGINE_STEP_BACKEND_ENV, "rust")

        reason = _parity_mismatch_reason(_comparison_metrics(case, "afd"))
        assert reason is None, reason


# The full engine-step golden matrix: every (case, surface) pair the parity
# classes above compare, in one importable structure so `pin_goldens.py`
# pins exactly what the tests consume (single source of truth — the pin
# script defines no case of its own). Keep in lockstep with the class
# parametrizations.
ENGINE_STEP_GOLDEN_MATRIX: tuple[tuple[list, tuple[str, ...]], ...] = (
    (SMOKE_CASES, ENGINE_STEP_SURFACES),
    (POWER_CASES, ENGINE_STEP_SURFACES),
    (CP_CASES, ("mixed",)),
    (DSV4_CP_CASES, ("cp_static_ctx", "mixed")),
    (HYBRID_CASES, ENGINE_STEP_SURFACES),
    (SOL_CASES, ("static", "mixed")),
    (TIE_AGG_CASES, ("agg",)),
    (TIE_DISAGG_CASES, ("disagg",)),
    (AFD_CASES, ("afd",)),
)


def _build_case_golden_ids() -> dict[EngineStepParityCase, str]:
    mapping: dict[EngineStepParityCase, str] = {}
    for params, _surfaces in ENGINE_STEP_GOLDEN_MATRIX:
        for param in params:
            (case,) = param.values
            existing = mapping.get(case)
            if existing is not None and existing != param.id:
                raise AssertionError(f"case {case!r} carries two golden ids: {existing!r} / {param.id!r}")
            mapping[case] = param.id
    return mapping


# Golden records are keyed by the pytest param id; the frozen dataclass is
# hashable, so tests recover the id from the parametrized case value.
_CASE_GOLDEN_IDS = _build_case_golden_ids()


class TestGoldenComparisonGuards:
    """Anti-vacuous guard: prove the golden comparison actually bites.

    The golden rewiring replaced the live Python side with fixture lookups;
    if the lookup or the tolerance check ever degenerated (e.g. comparing the
    rust value to itself, or an over-wide tolerance), every parity test would
    stay green forever. Doctoring an in-memory copy of the golden by 5x the
    tolerance and watching the comparison fail proves drift is detectable.
    The fixture on disk is never touched.
    """

    def test_golden_comparison_detects_drift(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _prepare_rust_core(monkeypatch)
        param = SMOKE_CASES[0]
        (case,) = param.values

        original = load_parity_golden
        doctored = copy.deepcopy(original("engine_step.json"))
        record = doctored["cases"][param.id]["static"]
        assert "values" in record, f"guard needs a computing case, got {record}"
        record["values"]["static_ctx"] = float(record["values"]["static_ctx"]) * 1.05

        monkeypatch.setattr(
            sys.modules[__name__],
            "load_parity_golden",
            lambda filename: doctored if filename == "engine_step.json" else original(filename),
        )
        reason = _parity_mismatch_reason(_static_comparison_metrics(case))
        assert reason is not None, "5% golden drift on static_ctx was not detected"
        assert "static_ctx" in reason and "drift" in reason, reason

    def test_golden_error_asymmetry_detected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The symmetric twin: a golden that recorded an error while the live
        # rust side computes must fail (asym), with the same message shape
        # the live differential used.
        _prepare_rust_core(monkeypatch)
        param = SMOKE_CASES[0]
        (case,) = param.values

        original = load_parity_golden
        doctored = copy.deepcopy(original("engine_step.json"))
        doctored["cases"][param.id]["static"] = {"error": "PerfDataNotAvailableError"}

        monkeypatch.setattr(
            sys.modules[__name__],
            "load_parity_golden",
            lambda filename: doctored if filename == "engine_step.json" else original(filename),
        )
        reason = _parity_mismatch_reason(_static_comparison_metrics(case))
        assert reason is not None, "golden-error vs rust-value asymmetry was not detected"
        assert "asym" in reason, reason


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
        # MiMo-V2-Flash has head_dim=192 while b200/vllm/0.19.0 collected only
        # {128, 256}: under SILICON, Rust hits `AicError::PerfDatabase` at the
        # attention query point — which must cross the FFI as the SAME sdk
        # class Python raises, recognized by the cause-chain walker (the
        # miss-classification the sweep/support-matrix rely on). (The previous
        # vehicle, NVFP4 GEMM on h200, now classifies as the strict
        # MissingSystemFlopsError before any data lookup — h200 has no
        # fp4_tc_flops — see the missing-dtype test below.)
        _prepare_rust_core(monkeypatch)
        case = EngineStepParityCase(model_path="XiaomiMiMo/MiMo-V2-Flash")
        with pytest.raises(errors.PerfDataNotAvailableError) as excinfo:
            _rust_static_breakdown(case)
        assert perf_database.has_perf_data_not_available_cause(excinfo.value)
        # The AicError display prefix pins the raise to the rust side of the
        # FFI (a Python-side miss would carry the sdk's own wording).
        message = str(excinfo.value)
        assert "perf database error" in message or "I/O error" in message, message

    def test_missing_dtype_flops_raises_typed_missing_flops_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # NVFP4 on Hopper: h200 has no fp4 hardware and its YAML defines no
        # fp4_tc_flops, so the strict per-dtype resolver rejects the
        # configuration at query entry (#1398) — before the HYBRID ladder
        # (any transfer policy) is even consulted, which is why this vehicle
        # can no longer pin #1455's balanced-policy ladder-miss contract
        # (symmetric EmpiricalNotImplementedError coverage lives in the
        # MSA/GDN HYBRID tests above). Rust raises
        # `AicError::MissingSystemFlops`, which must surface as the sdk's
        # MissingSystemFlopsError — the expected-CLI-error ValueError class —
        # and NOT be classified as a plain perf-data miss.
        _prepare_rust_core(monkeypatch)
        case = EngineStepParityCase(
            model_path="nvidia/MiniMax-M2.5-NVFP4",
            system_name="h200_sxm",
            database_mode="HYBRID",
        )
        with pytest.raises(errors.MissingSystemFlopsError) as excinfo:
            _rust_static_breakdown(case)
        message = str(excinfo.value)
        assert "fp4_tc_flops" in message, message
        # The AicError display prefix pins the raise to the rust side of the
        # FFI (a Python-side raise would carry the sdk's own wording).
        assert "missing system flops" in message, message
        assert isinstance(excinfo.value, ValueError)
        assert not perf_database.has_perf_data_not_available_cause(excinfo.value)

    def test_silicon_missing_dtype_on_lazy_family_matches_python(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # SILICON x missing-dtype on a lazily-loading table family (nvfp4
        # MLA-BMM: h200 has no fp4_tc_flops). Both engines must classify this
        # as MissingSystemFlopsError, NOT as a perf-data miss — the quadrant
        # the original pins (SILICON with all dtypes present; HYBRID
        # missing-dtype on an eager op) left uncovered until the flops
        # resolution was hoisted before every load/key lookup.
        _prepare_rust_core(monkeypatch)
        case = EngineStepParityCase(model_path="nvidia/MiniMax-M2.5-NVFP4", system_name="h200_sxm")
        with pytest.raises(errors.MissingSystemFlopsError) as excinfo:
            _rust_static_breakdown(case)
        assert "missing system flops" in str(excinfo.value), str(excinfo.value)
        assert not perf_database.has_perf_data_not_available_cause(excinfo.value)
        # Python classifies the same query point identically.
        database = _case_database(case)
        from aiconfigurator_core.sdk.engine import _evaluate_single_op
        from aiconfigurator_core.sdk.operations.mla import MLABmm

        with pytest.raises(errors.MissingSystemFlopsError):
            # The retired query_mla_bmm shim's exact twin, through the
            # single-op plumbing (the database's live SILICON view).
            _evaluate_single_op(
                database,
                MLABmm("mla_bmm_query", 1.0, 16, common.GEMMQuantMode.nvfp4, True),
                is_context=False,
                batch_size=16,
                s=1,
            )

    def test_pre_sm89_fp8_kv_generation_returns_shipped_silicon(self) -> None:
        # a100_sxm ships 2,534 measured generation-MLA rows with fp8 KV
        # (trtllm/1.0.0): Ampere has no fp8 tensor cores — the decode kernel
        # dequantizes KV and issues the MMA on the bf16 pipeline, which is how
        # that data was collected at all. The sm-gated derivation
        # (generation_attn_mode) must keep those rows queryable instead of
        # demanding an fp8_tc_flops entry a100 must never define (the
        # support-matrix FP8 gate is keyed on that entry's presence).
        database = _quiet_call(perf_database.get_database, "a100_sxm", "trtllm", "1.0.0")
        from aiconfigurator_core.sdk.engine import _evaluate_single_op
        from aiconfigurator_core.sdk.operations.mla import GenerationMLA

        def _gen_mla(kv_mode):
            # The retired query_generation_mla shim's exact twin (the
            # database's live SILICON view).
            op = GenerationMLA("generation_mla_query", 1.0, 64, kv_mode)
            return _evaluate_single_op(database, op, is_context=False, batch_size=1, s=65)

        fp8_kv = _gen_mla(common.KVCacheQuantMode.fp8)
        bf16_kv = _gen_mla(common.KVCacheQuantMode.bfloat16)
        assert float(fp8_kv) > 0
        # Same silicon exact-hit region: the two dtypes' measured values sit
        # within a few percent of each other (dequant-to-bf16 pipeline).
        assert abs(float(fp8_kv) - float(bf16_kv)) / float(bf16_kv) < 0.25


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
            metrics = _static_metrics(case)
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
            metrics = _static_metrics(case)
        assert not isinstance(metrics["total_ms"], _ErrorSentinel), repr(metrics)
        assert util_empirical.worst_provenance(tags) == "silicon", tags


# --------------------------------------------------------------------------- #
# forward_model="fpm": whole-model op parity over a synthetic dataset in a
# temp systems root (no fpm_forward pair is checked in). Python resolves the
# pair via `set_systems_paths`; the compiled spec carries the same root in
# `engine.systems_path`, so the Rust `FpmForwardTable` reads the identical
# file. Covers static_ctx / static_gen / mixed / genonly plus out-of-domain
# error symmetry and the FpmForward op-transfer tags.
# --------------------------------------------------------------------------- #

_FPM_MODEL = "MiniMaxAI/MiniMax-M2.5"
_FPM_VERSION = "0.19.0"
# (workload_kind, batch, total_prefill, total_kv, latency_ms) — per-DP-rank
# iteration totals, batch domains sized for the runtime points below.
_FPM_ROWS = [
    ("prefill", 1, 512, 0, 10.0),
    ("prefill", 1, 1024, 0, 18.0),
    ("prefill", 1, 2048, 0, 34.0),
    ("prefill", 1, 1024, 1024, 21.0),
    ("prefill", 2, 1024, 0, 18.5),
    ("prefill", 2, 2048, 0, 35.0),
    ("prefill", 2, 4096, 0, 69.0),
    ("prefill", 4, 4096, 0, 68.0),
    ("decode", 1, 0, 1, 2.0),
    ("decode", 1, 0, 1025, 2.2),
    ("decode", 1, 0, 65536, 3.1),
    ("decode", 4, 0, 4, 4.0),
    ("decode", 4, 0, 4100, 4.5),
    ("decode", 4, 0, 262144, 7.9),
    # CUDA-graph cliff pair at capture=512 plus an eager anchor: exercises
    # the decode batch-axis regime routing on both engines.
    ("decode", 512, 0, 4096, 10.0),
    ("decode", 512, 0, 65536, 11.0),
    ("decode", 512, 0, 262144, 12.0),
    ("decode", 513, 0, 4096, 30.0),
    ("decode", 513, 0, 65536, 33.0),
    ("decode", 513, 0, 262144, 36.0),
    ("decode", 1024, 0, 4096, 60.0),
    ("decode", 1024, 0, 65536, 66.0),
    ("decode", 1024, 0, 262144, 72.0),
]


def _fpm_write_pair(data_dir) -> None:
    import hashlib
    import json as _json

    import pyarrow as pa
    import pyarrow.parquet as pq

    identity = {
        "gemm_quant_mode": "fp8_block",
        "moe_quant_mode": "fp8_block",
        "fmha_quant_mode": "bfloat16",
        "comm_quant_mode": "half",
        "kv_cache_dtype": "fp8",
        "tp": 2,
        "pp": 1,
        "dp": 1,
        "moe_tp": 1,
        "moe_ep": 2,
        "cp": 1,
    }
    rows = []
    for kind, batch, prefill, kv, lat in _FPM_ROWS:
        rows.append(
            {
                "cell_id": f"fpm-{kind}-{batch}-{prefill}-{kv}",
                "model_path": _FPM_MODEL,
                "system": "b200_sxm",
                "backend": "vllm",
                "backend_version": _FPM_VERSION,
                "weight_quantization": "fp8_block",
                **identity,
                "moe_backend": "auto",
                "attention_backend": "auto",
                "enable_wideep": False,
                "enable_eplb": False,
                "workload_kind": kind,
                "batch_size": batch,
                "total_prefill_tokens": prefill,
                "total_kv_read_tokens": kv,
                "partition_policy": "balanced_v1",
                "latency_ms": lat,
            }
        )
    parquet_path = data_dir / "fpm_forward_perf.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    digest = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    (data_dir / "fpm_forward_perf.metadata.json").write_text(
        _json.dumps(
            {
                "schema_name": "aic_fpm_forward_perf",
                "schema_version": 6,
                "coordinate_system": "iteration_totals_balanced_v1",
                "measurement_policy": "dynamo_native_single_sample_v1",
                "system": "b200_sxm",
                "backend": "vllm",
                "backend_version": _FPM_VERSION,
                "parquet_sha256": digest,
                "row_count": len(rows),
            }
        )
    )


@pytest.fixture(scope="class")
def fpm_systems_root(tmp_path_factory):
    import shutil

    root = tmp_path_factory.mktemp("fpm_systems")
    pkg_systems = perf_database.get_systems_paths()[-1]
    shutil.copy(f"{pkg_systems}/b200_sxm.yaml", root / "b200_sxm.yaml")
    data_dir = root / "data" / "b200_sxm" / "vllm" / _FPM_VERSION
    data_dir.mkdir(parents=True)
    _fpm_write_pair(data_dir)

    perf_database.set_systems_paths(f"{root},default")
    rust_engine_step._engine_handle_cache_clear()
    try:
        yield root
    finally:
        perf_database.set_systems_paths("default")
        rust_engine_step._engine_handle_cache_clear()


# Frozen Python-side references for the FPM parity class, captured from the
# live Python FPM walk immediately before its deletion (Phase 2 PR-3) on the
# deterministic synthetic fixture above — the same freeze-then-delete pattern
# as the golden fixtures. This class sits outside ENGINE_STEP_GOLDEN_MATRIX
# because its dataset is generated per-run (`_FPM_ROWS`), so the frozen
# values live inline; string entries are expected exception kinds
# (error-symmetry contract).
_FPM_STATIC_FROZEN = {
    ("static_ctx", 1, 1024, 1, 0): 18.0,
    ("static_ctx", 1, 2048, 1, 1024): 21.0,
    ("static_ctx", 2, 1500, 1, 0): 50.8046875,
    ("static_ctx", 3, 1024, 1, 0): 51.37999771084165,
    ("static_gen", 1, 1024, 2, 0): 2.2,
    ("static_gen", 4, 1024, 2, 0): 4.5,
    ("static_gen", 2, 1024, 2, 0): 2.8812035914364387,
    ("static_gen", 4, 9_000_000, 2, 0): "PerfDataNotAvailableError",
    ("static_ctx", 16, 256, 1, 0): 68.0,
    ("static_ctx", 16, 320, 1, 256): "PerfDataNotAvailableError",
}
_FPM_MIXED_FROZEN = {
    (1024, 4, 1024, 2): 18.562552704189983,
    (512, 4, 1024, 2): 10.587744831848209,
    # ctx_tokens == 0 raises at MixedStepInput construction on both engines.
    (0, 4, 1024, 2): "ValueError",
    (0, 600, 100, 2): "ValueError",
    (1024, 0, 1024, 2): 18.0,
    (4096, 0, 256, 1): 68.0,
}
_FPM_GENONLY_FROZEN = 4.5


class TestRustEngineStepFpmParity:
    """forward_model='fpm' regression vs the frozen Python reference.

    The Python FPM walk is gone (Phase 2 PR-3); the live side below is the
    compiled engine (Op::FpmForward), compared against `_FPM_*_FROZEN`.
    """

    def _build(self):
        from aiconfigurator.sdk.config_builders import build_model_config

        cfg = build_model_config(
            tp_size=2,
            pp_size=1,
            attention_dp_size=1,
            moe_tp_size=1,
            moe_ep_size=2,
            gemm_quant_mode="fp8_block",
            moe_quant_mode="fp8_block",
            kvcache_quant_mode="fp8",
            fmha_quant_mode="bfloat16",
            comm_quant_mode="half",
            forward_model="fpm",
        )
        model = get_model(_FPM_MODEL, cfg, "vllm")
        database = _quiet_call(perf_database.get_database, "b200_sxm", "vllm", _FPM_VERSION)
        return model, get_backend("vllm"), database

    def _static(self, model, backend, database, mode, batch, isl, osl, prefix):
        rc = config.RuntimeConfig(batch_size=batch, beam_width=1, isl=isl, osl=osl, prefix=prefix)

        def thunk():
            summary = backend.run_static(model, database, rc, mode=mode)
            d = summary.get_context_latency_dict() if mode == "static_ctx" else summary.get_generation_latency_dict()
            return sum(d.values())

        return _safe_value(thunk)

    @staticmethod
    def _assert_frozen(frozen, rs, point):
        if isinstance(frozen, str):
            assert isinstance(rs, _ErrorSentinel) and rs.kind == frozen, (
                f"{point}: expected symmetric {frozen}, got {rs!r}"
            )
            return
        assert not isinstance(rs, _ErrorSentinel), f"{point}: frozen={frozen} but live raised {rs!r}"
        allowed = max(abs(frozen) * PARITY_RTOL, 1e-9)
        assert abs(rs - frozen) <= allowed, f"{point}: frozen={frozen} rs={rs} delta={abs(rs - frozen)}"

    def test_fpm_arena_selects_the_fpm_engine(self, fpm_systems_root, monkeypatch):
        # Review finding (#1461): from_native() dropped forward_model, so the
        # FPM arena always compiled the op_level engine. A decode-only
        # estimate hitting the fpm_forward table's exact row proves the
        # whole-model engine was selected through the supported predictor API.
        _prepare_rust_core(monkeypatch)
        from aiconfigurator_core.sdk.rust_engine_step import RustForwardPassPerfModel

        config = {
            "schema_version": 1,
            "model_name": _FPM_MODEL,
            "system_name": "b200_sxm",
            "backend": "vllm",
            "backend_version": _FPM_VERSION,
            "systems_path": str(fpm_systems_root),
            "tp_size": 2,
            "pp_size": 1,
            "attention_dp_size": 1,
            "moe_tp_size": 1,
            "moe_ep_size": 2,
            "weight_dtype": "fp8_block",
            "moe_dtype": "fp8_block",
            "activation_dtype": "bfloat16",
            "kv_cache_dtype": "fp8",
            "kv_block_size": None,
            "nextn": None,
            "forward_model": "fpm",
        }
        model = RustForwardPassPerfModel.from_native(config)
        decode_only = [
            {
                "version": 1,
                "scheduled_requests": {
                    "num_decode_requests": 1,
                    "sum_decode_kv_tokens": 1025,
                },
            }
        ]
        ms = model.estimate_forward_pass_time_ms(decode_only)
        assert ms == pytest.approx(2.2)  # exact fpm decode row (1, 1025)

    def test_fpm_spec_tags(self, fpm_systems_root, monkeypatch):
        _prepare_rust_core(monkeypatch)
        import json as _json

        from aiconfigurator.sdk import engine as sdk_engine

        model, _backend, database = self._build()
        spec = _json.loads(
            _quiet_call(
                sdk_engine.build_engine_spec_json,
                model,
                model_path=_FPM_MODEL,
                system="b200_sxm",
                backend="vllm",
                backend_version=_FPM_VERSION,
                kv_block_size=None,
                systems_path=None,
                nextn=0,
                database=database,
            )
        )
        (ctx_tag,) = spec["context_ops"][0].keys()
        (gen_tag,) = spec["generation_ops"][0].keys()
        assert (ctx_tag, gen_tag) == ("FpmForward", "FpmForward")
        ctx_op = spec["context_ops"][0]["FpmForward"]
        assert ctx_op["phase"] == "prefill"
        assert spec["generation_ops"][0]["FpmForward"]["phase"] == "decode"
        assert len(ctx_op["match_identity"]) == 15
        assert ctx_op["sol_ops"], "sol_ops must carry the original granular list"

    @pytest.mark.parametrize(
        ("mode", "batch", "isl", "osl", "prefix"),
        [
            ("static_ctx", 1, 1024, 1, 0),  # exact prefill hit
            ("static_ctx", 1, 2048, 1, 1024),  # exact cached-prefill hit
            ("static_ctx", 2, 1500, 1, 0),  # in-curve lerp
            ("static_ctx", 3, 1024, 1, 0),  # uncollected batch -> transfer (SOL)
            ("static_gen", 1, 1024, 2, 0),  # exact decode hit (kv = isl+1)
            ("static_gen", 4, 1024, 2, 0),  # exact decode hit at B=4
            ("static_gen", 2, 1024, 2, 0),  # uncollected batch -> transfer (SOL)
            ("static_gen", 4, 9_000_000, 2, 0),  # out of domain -> both error
            ("static_ctx", 16, 256, 1, 0),  # above the batch ceiling -> pure clamp (kv/T = 0)
            ("static_ctx", 16, 320, 1, 256),  # high KV pressure -> SOL-rescaled clamp
        ],
    )
    def test_fpm_static_parity(self, fpm_systems_root, monkeypatch, mode, batch, isl, osl, prefix):
        _prepare_rust_core(monkeypatch)
        model, backend, database = self._build()
        rs = self._static(model, backend, database, mode, batch, isl, osl, prefix)
        frozen = _FPM_STATIC_FROZEN[(mode, batch, isl, osl, prefix)]
        self._assert_frozen(frozen, rs, f"{mode} B={batch} isl={isl} prefix={prefix}")

    @pytest.mark.parametrize(
        ("ctx_tokens", "gen_tokens", "isl", "osl"),
        [
            (1024, 4, 1024, 2),  # mixed: prefill chunk + marginal decode
            (512, 4, 1024, 2),  # partial chunk (chunk_scale > 1)
            (0, 4, 1024, 2),  # gen-only keeps full decode
            (0, 600, 100, 2),  # gen-only across the decode regime boundary (eager side)
            (1024, 0, 1024, 2),  # prefill-only chunk
            (4096, 0, 256, 1),  # 16 whole prefills: certified batch clamp to the ceiling
        ],
    )
    def test_fpm_mixed_step_parity(self, fpm_systems_root, monkeypatch, ctx_tokens, gen_tokens, isl, osl):
        _prepare_rust_core(monkeypatch)
        model, backend, database = self._build()

        rc = config.RuntimeConfig(batch_size=1, beam_width=1, isl=isl, osl=osl, prefix=0)
        rs = _safe_value(
            lambda: backend._get_mix_step_latency(model, database, rc, ctx_tokens, gen_tokens, isl, osl, 0)[0]
        )
        frozen = _FPM_MIXED_FROZEN[(ctx_tokens, gen_tokens, isl, osl)]
        self._assert_frozen(frozen, rs, f"mixed ctx={ctx_tokens} gen={gen_tokens}")

    def test_fpm_genonly_step_parity(self, fpm_systems_root, monkeypatch):
        _prepare_rust_core(monkeypatch)
        model, backend, database = self._build()

        rc = config.RuntimeConfig(batch_size=1, beam_width=1, isl=1024, osl=2, prefix=0)
        rs = _safe_value(lambda: backend._get_genonly_step_latency(model, database, rc, 4, 1023, 2)[0])
        self._assert_frozen(_FPM_GENONLY_FROZEN, rs, "genonly gen=4")
