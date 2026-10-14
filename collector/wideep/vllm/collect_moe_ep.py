# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM large-EP MoE expert-compute collector (op ``moe_ep``) — DORMANT (D3).

Benchmarks vLLM's fused-experts kernel — the expert-compute equivalent of
sglang's ``DeepEPMoE.run_moe_core`` — simulating an EP world of
``moe_ep_size`` ranks on one GPU by allocating only the rank-local expert
shard, and emits the same unified ``moe_expert_compute_perf`` rows (one table,
``inference_phase`` column) consumed by
``aiconfigurator_core.sdk.operations.moe_comm.load_moe_expert_compute_data``.

DORMANT per plan decision D3: the pinned ``wideep_vllm`` runtime and empty
registry serve standalone ``moe_a2a`` collection only. This module has no
``OpEntry`` and no ``hash_closures.yaml`` entry; a closure may not precede
registration. Activation is documented in ``collector/wideep/README.md``.
Enrollment MUST verify on the pinned image that the kernel invoked here is
the one vLLM serving dispatch selects for the large-EP DeepEP compute path
(layer_permissions.md: kernel_source records ground truth, manual pins need
source proof).

Shapes are DECLARED: every benchmarked geometry comes from the
``model_case_values.moe`` rows marked ``wideep: true`` crossed with the
``cases/base_ops/moe.yaml`` expert-parallel grid, under the declared vllm
quantization policy.
"""

import os
import sys

import torch
from vllm.model_executor.layers.fused_moe import fused_experts

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COLLECTOR_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
if COLLECTOR_ROOT not in sys.path:
    sys.path.append(COLLECTOR_ROOT)

try:
    from helper import benchmark_with_power, log_perf, power_law_deepep_decode, power_law_deepep_prefill
except ModuleNotFoundError:
    sys.path.append(COLLECTOR_ROOT)
    from helper import benchmark_with_power, log_perf, power_law_deepep_decode, power_law_deepep_prefill

from importlib.metadata import version as get_version

#: The only quantization this collector benchmarks: block-scaled FP8
#: (``fused_experts`` with ``use_fp8_w8a8=True`` and a 128x128
#: ``block_shape``) — the vllm serving path for fp8_block checkpoints, the
#: same precision the sglang moe_ep collector drives through deep_gemm.
#: Cases for models whose declared vllm ``allowed_modes`` exclude it are not
#: generated — a declaration-layer decision, never a relabelled invocation.
MOE_EP_QUANT_MODE = "fp8_block"

#: The unified table key the consumer resolves for vllm large-EP compute:
#: ``MoEExpertCompute._resolve_kernel_source`` returns ``"deepep_moe"`` for BOTH sglang
#: and vllm (aic-core sdk/operations/moe_comm.py — vllm's large-EP serving
#: path is DeepEP dispatch + fused-experts compute, the same kernel leg).
#: Enrollment (D3 activation) must verify on the pinned vLLM-DeepEP image
#: that this benchmark invokes exactly that serving path before rows are
#: collected under this label.
MOE_EP_KERNEL_SOURCE = "deepep_moe"

#: Written into the ``op_name`` prefix column of every row; the context /
#: generation split lives in the ``inference_phase`` payload column.
MOE_EP_OP_NAME = "moe_ep"

#: Fp8 block quantization geometry (weights and activations), matching the
#: DeepSeek-style block-scaled FP8 recipe vllm's fused_experts consumes.
MOE_EP_FP8_BLOCK_SHAPE = (128, 128)

#: Per-rank token sweeps, identical to the sglang moe_ep collector's phase
#: grids so both backends give the interpolator the same token support.
MOE_EP_CONTEXT_NUM_TOKENS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
MOE_EP_GENERATION_NUM_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128)
MOE_EP_POWER_LAW_ALPHAS = (0.6, 0.8, 1.01, 1.02, 1.2)


class MoeEpBenchmarkError(RuntimeError):
    """One queued moe_ep benchmark case failed to execute.

    ``layer_permissions.md``: a queued case is executed or raises. The executor
    records this with its case parameters in ``errors_<module>.json``.
    """


def _measure_power_enabled() -> bool:
    """Whether this run measures power (mirrors ``helper._parse_bool_env``)."""
    value = os.environ.get("COLLECTOR_MEASURE_POWER")
    return False if value is None else value.lower() in ("true", "1", "yes")


def _power_columns(power_stats) -> dict | None:
    """D7: emit power only where the bench measures it, and never as 0.0.

    Returns ``None`` (no power columns at all) when the run does not measure
    power; otherwise the measured stats, or NaN when the sampler produced no
    samples. A present-but-null power column would crash ``load_moe_expert_compute_data``.
    """
    if not _measure_power_enabled():
        return None
    if power_stats and power_stats.get("power") is not None:
        return power_stats
    return {"power": float("nan"), "power_limit": float("nan")}


def _moe_expert_compute_perf_path(output_path, perf_filename) -> str:
    """Resolve the registry-provided perf filename into the run's output dir."""
    directory = output_path if output_path is not None else os.getcwd()
    return os.path.join(directory, str(perf_filename))


def _global_num_tokens(num_tokens: int, moe_ep_size: int) -> int:
    """The GLOBAL token count for one per-rank token point.

    The single source for the family's token accounting, mirroring the sglang
    twin exactly (``collect_deepep_moe.py``: ``num_token * simulated_ep_size``
    sizes the hidden states, feeds ``power_law_deepep_prefill`` /
    ``power_law_deepep_decode`` AND is the persisted ``num_tokens`` column).
    Both the bench input and the persisted key MUST use this one expression —
    feeding the per-rank count into the distribution generators would land
    rows on keys shared with sglang rows that benchmarked an ep-fold larger
    token population.
    """
    return int(num_tokens) * int(moe_ep_size)


def _build_moe_ep_row(
    *,
    moe_dtype: str,
    distribution: str,
    inference_phase: str,
    num_tokens: int,
    hidden_size: int,
    inter_size: int,
    topk: int,
    num_experts: int,
    num_slots: int,
    moe_tp_size: int,
    moe_ep_size: int,
    latency_ms: float,
) -> dict:
    """Build one unified ``moe_expert_compute_perf`` payload row.

    Column set and order are the consumer contract
    (``sdk/operations/moe_comm.py::load_moe_expert_compute_data``); the
    ``framework/version/device/op_name/kernel_source`` prefix is added by
    ``helper.log_perf``. ``latency`` is milliseconds. Identical payload to the
    sglang and trtllm moe_ep collectors: one table, one schema.
    """
    return {
        "moe_dtype": moe_dtype,
        "distribution": distribution,
        "inference_phase": inference_phase,
        "num_tokens": int(num_tokens),
        "hidden_size": int(hidden_size),
        "inter_size": int(inter_size),
        "topk": int(topk),
        "num_experts": int(num_experts),
        # vllm has no EPLB redundant-expert axis here: every routed expert
        # owns exactly one slot (same stance as the sglang collector).
        "num_slots": int(num_slots),
        "moe_tp_size": int(moe_tp_size),
        "moe_ep_size": int(moe_ep_size),
        "latency": float(latency_ms),
    }


def _phase_cases(inference_phase: str):
    """The per-phase (num_tokens, distribution, alpha) sweep, sorted (D5).

    ``num_tokens`` here is the PER-RANK token count; rows persist the global
    count (``num_tokens * moe_ep_size``), mirroring the sglang collector.
    """
    if inference_phase == "context":
        token_counts = MOE_EP_CONTEXT_NUM_TOKENS
    elif inference_phase == "generation":
        token_counts = MOE_EP_GENERATION_NUM_TOKENS
    else:
        raise ValueError(f"unknown inference_phase {inference_phase!r}")

    cases = []
    for num_tokens in token_counts:
        cases.append({"num_tokens": num_tokens, "distributed": "uniform", "power_law_alpha": None})
        for alpha in MOE_EP_POWER_LAW_ALPHAS:
            cases.append({"num_tokens": num_tokens, "distributed": "power_law", "power_law_alpha": alpha})
    return sorted(
        cases,
        key=lambda case: (
            case["distributed"],
            case["power_law_alpha"] if case["power_law_alpha"] is not None else -1.0,
            case["num_tokens"],
        ),
    )


def _collect_phase_rows(
    *,
    inference_phase: str,
    phase_cases,
    bench,
    moe_ep_size: int,
    hidden_size: int,
    inter_size: int,
    topk: int,
    num_experts: int,
    num_slots: int,
) -> list[tuple[dict, dict | None]]:
    """Run one phase sweep through ``bench`` and build the persisted rows.

    ``bench(inference_phase, global_num_tokens, distributed, power_law_alpha)``
    must execute the fused-experts benchmark for one token point and return
    ``(latency_ms, power_stats_or_None)``. The bench receives the GLOBAL token
    count — the exact value persisted in the row's ``num_tokens`` column, both
    computed once here through :func:`_global_num_tokens` so the benchmarked
    population and the key can never diverge. Injected so the row/population
    contract is unit-testable without vllm or a GPU; the collection entrypoint
    passes :func:`_make_fused_experts_bench`'s callable.

    Execute-or-raise: a failing token point raises ``MoeEpBenchmarkError``
    with the case parameters — never a silent skip.
    """
    rows: list[tuple[dict, dict | None]] = []
    for case in phase_cases:
        num_tokens = case["num_tokens"]
        distributed = case["distributed"]
        power_law_alpha = case["power_law_alpha"]
        global_tokens = _global_num_tokens(num_tokens, moe_ep_size)
        try:
            latency_ms, power_stats = bench(inference_phase, global_tokens, distributed, power_law_alpha)
        except MoeEpBenchmarkError:
            raise
        except Exception as e:
            raise MoeEpBenchmarkError(
                f"moe_ep {inference_phase} case failed (num_tokens={num_tokens}, "
                f"global_num_tokens={global_tokens}, "
                f"distribution={distributed}, alpha={power_law_alpha}, "
                f"moe_ep_size={moe_ep_size}, num_experts={num_experts}): {e}"
            ) from e
        distribution_str = f"power_law_{power_law_alpha}" if distributed == "power_law" else distributed
        rows.append(
            (
                _build_moe_ep_row(
                    moe_dtype=MOE_EP_QUANT_MODE,
                    distribution=distribution_str,
                    inference_phase=inference_phase,
                    num_tokens=global_tokens,
                    hidden_size=hidden_size,
                    inter_size=inter_size,
                    topk=topk,
                    num_experts=num_experts,
                    num_slots=num_slots,
                    moe_tp_size=1,
                    moe_ep_size=moe_ep_size,
                    latency_ms=latency_ms,
                ),
                power_stats,
            )
        )
    return rows


def get_moe_ep_test_cases():
    """Declared large-EP MoE compute cases (vllm twin of the sglang getter).

    Filters, all in declared homes:

    * ``wideep: true`` on the model's moe row — the op is declared for the model.
    * ``tp == 1 and ep > 1`` — the large-EP identity of this table.
    * declared vllm quantization policy must allow :data:`MOE_EP_QUANT_MODE`,
      the only precision this benchmark actually runs.

    One benchmark invocation per ``(model, ep)``: this collector simulates the
    EP world on a single GPU and sweeps token counts and expert distributions
    internally. Emitted sorted (D5).

    Returns:
        list[list]: ``[num_local_experts, moe_ep_size, hidden_size, inter_size,
        topk, num_experts, num_slots, moe_dtype]`` per case — the same
        positional layout as the sglang ``run_moe_ep``.
    """
    try:
        # collect.py puts COLLECTOR_ROOT on sys.path (see the module header).
        from case_generator import (
            get_common_moe_test_cases,
            is_wideep_moe_model,
            moe_model_allows_quantization,
        )
    except ModuleNotFoundError:
        from collector.case_generator import (
            get_common_moe_test_cases,
            is_wideep_moe_model,
            moe_model_allows_quantization,
        )

    recipes = get_common_moe_test_cases(backend="vllm")
    dropped_not_declared = 0
    dropped_not_ep = 0
    dropped_quant = 0
    cases: dict[tuple[str, int], list] = {}

    for recipe in recipes:
        if not is_wideep_moe_model(recipe.model_name):
            dropped_not_declared += 1
            continue
        if recipe.tp != 1 or recipe.ep <= 1:
            dropped_not_ep += 1
            continue
        if not moe_model_allows_quantization("vllm", recipe.model_name, MOE_EP_QUANT_MODE):
            dropped_quant += 1
            continue
        num_experts = int(recipe.num_experts)
        ep_size = int(recipe.ep)
        # get_common_moe_test_cases already enforces num_experts % ep == 0.
        cases.setdefault(
            (recipe.model_name, ep_size),
            [
                num_experts // ep_size,
                ep_size,
                int(recipe.hidden_size),
                int(recipe.inter_size),
                int(recipe.topk),
                num_experts,
                num_experts,  # num_slots: no EPLB redundancy axis on vllm
                MOE_EP_QUANT_MODE,
            ],
        )

    print(
        f"moe_ep[vllm]: {len(cases)} cases from {len(recipes)} moe recipes "
        f"(dropped: {dropped_not_declared} not declared wideep, "
        f"{dropped_not_ep} not tp=1/ep>1, {dropped_quant} quant not allowed; "
        f"{len(recipes) - dropped_not_declared - dropped_not_ep - dropped_quant - len(cases)} "
        f"deduplicated onto (model, ep))"
    )
    return [cases[key] for key in sorted(cases)]


def _make_fused_experts_bench(
    *,
    num_local_experts: int,
    moe_ep_size: int,
    hidden_size: int,
    inter_size: int,
    topk: int,
    num_experts: int,
    device,
):
    """Build the real fused-experts bench callable for one declared case.

    Allocates the rank-local expert shard as block-scaled FP8 weights and
    times ``vllm.model_executor.layers.fused_moe.fused_experts`` over the
    rank-local routing produced by the shared power-law helpers (the same
    generators the sglang collector uses). Returns the ``bench`` callable
    :func:`_collect_phase_rows` consumes.
    """
    block_n, block_k = MOE_EP_FP8_BLOCK_SHAPE
    fp8 = torch.float8_e4m3fn

    w1 = torch.randn(num_local_experts, 2 * inter_size, hidden_size, dtype=torch.bfloat16, device=device).to(fp8)
    w2 = torch.randn(num_local_experts, hidden_size, inter_size, dtype=torch.bfloat16, device=device).to(fp8)
    w1_scale = torch.ones(
        num_local_experts,
        (2 * inter_size + block_n - 1) // block_n,
        (hidden_size + block_k - 1) // block_k,
        dtype=torch.float32,
        device=device,
    )
    w2_scale = torch.ones(
        num_local_experts,
        (hidden_size + block_n - 1) // block_n,
        (inter_size + block_k - 1) // block_k,
        dtype=torch.float32,
        device=device,
    )

    def _counts_to_topk_ids(tokens_per_local_expert, global_num_tokens: int):
        """Spread per-local-expert token counts over a [global, topk] id grid.

        Only the rank-local selections are active; every other slot is -1
        with zero weight — the same rank-local masking shape the sglang twin
        drives through its dispatch-output tensors.
        """
        flat = torch.repeat_interleave(
            torch.arange(num_local_experts, dtype=torch.int64), tokens_per_local_expert.to(torch.int64)
        )[: global_num_tokens * topk]
        if flat.numel() < global_num_tokens * topk:
            flat = torch.nn.functional.pad(flat, (0, global_num_tokens * topk - flat.numel()), value=-1)
        topk_ids = flat.reshape(global_num_tokens, topk).to(device=device, dtype=torch.int32)
        topk_weights = torch.where(
            topk_ids >= 0,
            torch.full_like(topk_ids, 1.0 / topk, dtype=torch.float32),
            torch.zeros((), dtype=torch.float32, device=device),
        )
        return topk_ids, topk_weights

    def _routing(inference_phase: str, global_num_tokens: int, distributed: str, power_law_alpha):
        """Rank-local (topk_ids, topk_weights) over the GLOBAL token count.

        Token accounting mirrors the sglang twin exactly: the distribution
        generators and the routing grid are sized with the GLOBAL count
        (``collect_deepep_moe.py`` feeds ``num_token * simulated_ep_size``
        into ``power_law_deepep_prefill``/``power_law_deepep_decode`` and its
        uniform arithmetic), never the per-rank count.

        Non-local expert selections are -1 with zero weight. ENROLLMENT-TIME
        VERIFICATION ITEM (D3, see the module docstring): confirm against the
        pinned vllm source that ``fused_experts`` skips out-of-range ids
        during alignment (the shape vllm's EP ``expert_map`` produces) — this
        is the behavior this routing relies on and it is UNVERIFIED here, no
        vllm runtime being pinned yet.
        """
        if distributed == "uniform":
            # Mirrors the sglang uniform arithmetic: global_tokens * topk
            # selections spread evenly over ALL experts; this rank computes
            # its num_local_experts share.
            tokens_per_local_expert = global_num_tokens * topk // num_experts
            counts = torch.full((num_local_experts,), tokens_per_local_expert, dtype=torch.int64)
            if tokens_per_local_expert == 0:
                # Fewer routed selections than experts: the first
                # per-rank-tokens x topk local experts get one token each —
                # the sglang decode fallback's arithmetic
                # (masked_m[: num_token * top_k] = 1, num_token per-rank).
                counts[: max(global_num_tokens * topk // moe_ep_size, 1)] = 1
            return _counts_to_topk_ids(counts, global_num_tokens)
        if inference_phase == "context":
            topk_idx, topk_weights, _ = power_law_deepep_prefill(
                global_num_tokens, num_experts, topk, moe_ep_size, power_law_alpha
            )
            return topk_idx.to(device=device, dtype=torch.int32), topk_weights.to(device=device)
        tokens_per_local_expert = power_law_deepep_decode(
            global_num_tokens, num_experts, topk, moe_ep_size, power_law_alpha
        )
        return _counts_to_topk_ids(tokens_per_local_expert, global_num_tokens)

    def bench(inference_phase: str, global_num_tokens: int, distributed: str, power_law_alpha):
        # `global_num_tokens` arrives pre-globalized from _collect_phase_rows
        # (_global_num_tokens) — the same value persisted in the row.
        hidden_states = torch.randn(global_num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
        topk_ids, topk_weights = _routing(inference_phase, global_num_tokens, distributed, power_law_alpha)

        def kernel_func():
            fused_experts(
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                inplace=False,
                use_fp8_w8a8=True,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                block_shape=list(MOE_EP_FP8_BLOCK_SHAPE),
            )

        with benchmark_with_power(
            device=device,
            kernel_func=kernel_func,
            num_warmups=3,
            num_runs=10,
            repeat_n=1,
        ) as results:
            pass
        return results["latency_ms"], results["power_stats"]

    return bench


def run_moe_ep(
    num_local_experts,
    moe_ep_size,
    hidden_size,
    inter_size,
    topk,
    num_experts,
    num_slots,
    moe_dtype,
    *,
    perf_filename,
    device="cuda:0",
    bench=None,
    output_path=None,
):
    """Run one declared large-EP MoE compute case through vllm fused_experts.

    ``perf_filename`` follows the sglang collector's convention
    (``PerfFile.MOE_EXPERT_COMPUTE``, bound by the registry OpEntry at enrollment).
    ``bench`` is injectable for tests; the default is the real
    :func:`_make_fused_experts_bench` callable on ``device``.
    """
    if moe_dtype != MOE_EP_QUANT_MODE:
        raise MoeEpBenchmarkError(
            f"moe_ep[vllm] benchmarks {MOE_EP_QUANT_MODE!r} only; case declared moe_dtype={moe_dtype!r}"
        )
    device = torch.device(device)
    if bench is None:
        torch.cuda.set_device(device)
        bench = _make_fused_experts_bench(
            num_local_experts=num_local_experts,
            moe_ep_size=moe_ep_size,
            hidden_size=hidden_size,
            inter_size=inter_size,
            topk=topk,
            num_experts=num_experts,
            device=device,
        )

    for inference_phase in ("context", "generation"):
        rows = _collect_phase_rows(
            inference_phase=inference_phase,
            phase_cases=_phase_cases(inference_phase),
            bench=bench,
            moe_ep_size=moe_ep_size,
            hidden_size=hidden_size,
            inter_size=inter_size,
            topk=topk,
            num_experts=num_experts,
            num_slots=num_slots,
        )
        for row, power_stats in rows:
            # log_perf reports write failures by returning False — fail
            # closed, mirroring the sglang/trtllm moe_ep writers.
            if not log_perf(
                item_list=[row],
                framework="VLLM",
                version=get_version("vllm"),
                device_name=torch.cuda.get_device_name(device),
                op_name=MOE_EP_OP_NAME,
                kernel_source=MOE_EP_KERNEL_SOURCE,
                perf_filename=_moe_expert_compute_perf_path(output_path, perf_filename),
                power_stats=_power_columns(power_stats),
            ):
                raise MoeEpBenchmarkError(
                    f"helper.log_perf failed to persist the measured {inference_phase} row "
                    f"(ep={moe_ep_size}, num_experts={num_experts})"
                )
