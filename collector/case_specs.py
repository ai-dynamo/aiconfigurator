# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import itertools
import os
import sys
from pathlib import Path
from typing import Optional

import yaml

COLLECTOR_ROOT = Path(__file__).resolve().parent
BASE_CASES_PATH = COLLECTOR_ROOT / "cases" / "base_model_cases.yaml"
MODEL_CASES_DIR = COLLECTOR_ROOT / "cases" / "models"


def _load_base_cases_data() -> dict:
    with open(BASE_CASES_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{BASE_CASES_PATH}: top-level YAML value must be a mapping")
    return data


def get_base_op_case_specs(op_name: str) -> list[dict[str, object]]:
    """Return dict-style case specs for a framework-agnostic base op."""
    try:
        cases = _load_base_cases_data().get("all_frameworks_op_cases", {}).get(op_name, {}).get("cases")
    except FileNotFoundError:
        return []
    if not isinstance(cases, list):
        return []
    return [case for case in cases if isinstance(case, dict)]


def get_base_framework_op_case_specs(backend: str, op_name: str) -> list[dict[str, object]]:
    """Return dict-style case specs for a backend-specific base op."""
    try:
        framework_cases = _load_base_cases_data().get("framework_specific_op_cases", {})
    except FileNotFoundError:
        return []
    cases = framework_cases.get(backend, {}).get(op_name, {}).get("cases")
    if not isinstance(cases, list):
        return []
    return [case for case in cases if isinstance(case, dict)]


def get_base_common_case_values(name: str) -> dict[str, object]:
    """Return shared scalar/list values from base_model_cases.yaml."""
    try:
        values = _load_base_cases_data().get("common_case_values", {}).get(name, {})
    except FileNotFoundError:
        return {}
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise TypeError(f"common_case_values.{name} must be a mapping")
    return values


def _required_base_common_case_values(name: str) -> dict[str, object]:
    values = get_base_common_case_values(name)
    if not values:
        raise RuntimeError(f"{BASE_CASES_PATH} is missing common_case_values.{name}")
    return values


def _get_model_path_filter() -> str | None:
    """Return the model-path filter from the environment, or None for 'all'."""
    val = os.environ.get("COLLECTOR_MODEL_PATH", "").strip()
    return val if val else None


def _load_model_cases_data() -> list[dict]:
    data = []
    for path in sorted(MODEL_CASES_DIR.glob("*_cases.yaml")):
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise TypeError(f"{path}: top-level YAML value must be a mapping")
        data.append(raw)
    return data


def _model_case_values(op_name: str, *, apply_model_filter: bool = True) -> list[dict]:
    values = []
    for data in _load_model_cases_data():
        op_values = data.get("model_case_values", {}).get(op_name, [])
        if op_values is None:
            continue
        if isinstance(op_values, dict):
            values.append(dict(op_values))
            continue
        if not isinstance(op_values, list):
            raise TypeError(f"model_case_values.{op_name} must be a list or mapping")
        values.extend(dict(item) for item in op_values)

    model_path = _get_model_path_filter() if apply_model_filter else None
    if model_path:
        values = [value for value in values if value.get("model_path") == model_path]
    return values


def is_wideep_moe_model(model_name: str) -> bool:
    """Return True if *model_name* needs WideEP MoE collection."""
    return any(
        value.get("model_path") == model_name and value.get("wideep")
        for value in _model_case_values("moe", apply_model_filter=False)
    )


def get_all_model_names() -> list[str]:
    """Return all known model names across all op types.

    Reads directly from model case YAML — does not instantiate test
    case objects or call generator functions, so pruning logic in the generators
    cannot accidentally exclude models from the allowlist.
    """
    model_names = []
    for data in _load_model_cases_data():
        primary = data.get("model_path")
        if primary and primary != "__base__":
            model_names.append(str(primary))
        model_names.extend(str(path) for path in data.get("model_paths", []) or [])
        for values in (data.get("model_case_values") or {}).values():
            if isinstance(values, dict):
                value_model_path = values.get("model_path")
                if value_model_path:
                    model_names.append(str(value_model_path))
                continue
            if isinstance(values, list):
                model_names.extend(
                    str(value["model_path"]) for value in values if isinstance(value, dict) and value.get("model_path")
                )

    deduped = []
    seen = set()
    for model_name in model_names:
        if model_name in seen:
            continue
        seen.add(model_name)
        deduped.append(model_name)
    return deduped


@dataclasses.dataclass
class MoeCommonTestCase:
    num_tokens_list: list[int]
    hidden_size: int
    inter_size: int
    topk: int
    num_experts: int
    tp: int
    ep: int
    model_name: str
    token_expert_distribution: str
    power_law_alpha: Optional[float]


def _moe_token_expert_distributions(moe_sweep: dict[str, object]) -> list[tuple[str, Optional[float]]]:
    raw_distributions = moe_sweep.get("token_expert_distributions")
    if not isinstance(raw_distributions, list):
        raise TypeError("common_case_values.moe.token_expert_distributions must be a list")

    distributions = []
    for item in raw_distributions:
        if not isinstance(item, dict):
            raise TypeError("common_case_values.moe.token_expert_distributions entries must be mappings")
        name = item.get("name") or item.get("distribution")
        if not name:
            raise ValueError("MoE token expert distribution entries need a name")
        alpha = item.get("power_law_alpha")
        distributions.append((str(name), None if alpha is None else float(alpha)))
    return distributions


def get_common_moe_test_cases():
    moe_sweep = _required_base_common_case_values("moe")
    num_tokens = _as_int_list(moe_sweep.get("token_counts"), field_name="moe.token_counts")
    tp_list = _as_int_list(moe_sweep.get("tensor_parallel_sizes"), field_name="moe.tensor_parallel_sizes")
    ep_list = _as_int_list(moe_sweep.get("expert_parallel_sizes"), field_name="moe.expert_parallel_sizes")
    num_gpu_list = _as_int_list(moe_sweep.get("gpu_counts"), field_name="moe.gpu_counts")
    token_distributions = _moe_token_expert_distributions(moe_sweep)

    model_config_list = _model_case_values("moe")

    test_cases: list[MoeCommonTestCase] = []

    for (
        num_gpu,  # starting from fewer gpus. workaround for potential buffer bug in moe impl.
        model_config,
        tp,
        ep,
        (token_distribution, power_law_alpha),
    ) in itertools.product(
        num_gpu_list,
        model_config_list,
        tp_list,
        ep_list,
        token_distributions,
    ):
        hs = int(model_config["hidden_size"])
        inter_s = int(model_config["inter_size"])
        topk = int(model_config["topk"])
        num_experts = int(model_config["num_experts"])
        model_name = str(model_config["model_path"])

        max_tp_exclusive = model_config.get("max_tp_exclusive")
        if max_tp_exclusive is not None and tp >= int(max_tp_exclusive):
            continue

        if tp * ep != num_gpu:
            continue
        if ep > num_experts:
            continue
        if num_experts % ep != 0:
            continue
        # we need to ensure inter_s can be divided by tp.
        if inter_s % tp != 0:
            continue

        test_cases.append(
            MoeCommonTestCase(
                num_tokens_list=num_tokens,
                hidden_size=hs,
                inter_size=inter_s,
                topk=topk,
                num_experts=num_experts,
                tp=tp,
                ep=ep,
                model_name=model_name,
                token_expert_distribution=token_distribution,
                power_law_alpha=power_law_alpha,
            )
        )

    return test_cases


@dataclasses.dataclass
class GemmCommonTestCase:
    x: int
    n: int
    k: int


def _as_int_list(value, *, field_name: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return [int(item) for item in value]


def _get_base_gemm_shape_sweeps() -> list[dict[str, object]]:
    shape_sweeps = get_base_op_case_specs("gemm")
    if not shape_sweeps:
        raise RuntimeError(f"{BASE_CASES_PATH} is missing all_frameworks_op_cases.gemm.cases")
    return shape_sweeps


def get_gemm_case_specs() -> list[GemmCommonTestCase]:
    test_cases = []
    for shape_sweep in _get_base_gemm_shape_sweeps():
        token_counts = _as_int_list(shape_sweep.get("token_counts"), field_name="gemm.token_counts")
        feature_sizes = shape_sweep.get("feature_sizes")
        input_feature_sizes = _as_int_list(
            shape_sweep.get("input_feature_sizes", feature_sizes),
            field_name="gemm.input_feature_sizes",
        )
        output_feature_sizes = _as_int_list(
            shape_sweep.get("output_feature_sizes", feature_sizes),
            field_name="gemm.output_feature_sizes",
        )
        skip_shapes = {
            (int(skip["output_features"]), int(skip["input_features"])) for skip in shape_sweep.get("skip_shapes", [])
        }

        for token_count in sorted(token_counts, reverse=True):
            for output_features in sorted(output_feature_sizes, reverse=True):
                for input_features in sorted(input_feature_sizes, reverse=True):
                    if (output_features, input_features) in skip_shapes:
                        continue
                    if output_features * input_features == 65536 * 65536:
                        continue
                    test_cases.append(GemmCommonTestCase(x=token_count, n=output_features, k=input_features))

    return test_cases


@dataclasses.dataclass
class ComputeScaleCommonTestCase:
    m: int
    k: int


def get_compute_scale_case_specs() -> list[ComputeScaleCommonTestCase]:
    shape_sweeps = get_base_framework_op_case_specs("trtllm", "compute_scale")
    if not shape_sweeps:
        seen_mk = set()
        test_cases = []
        for gemm_common_testcase in get_gemm_case_specs():
            key = (gemm_common_testcase.x, gemm_common_testcase.k)
            if key in seen_mk:
                continue
            seen_mk.add(key)
            test_cases.append(ComputeScaleCommonTestCase(m=key[0], k=key[1]))
        return test_cases

    test_cases = []
    for shape_sweep in shape_sweeps:
        token_counts = _as_int_list(shape_sweep.get("token_counts"), field_name="compute_scale.token_counts")
        input_feature_sizes = _as_int_list(
            shape_sweep.get("input_feature_sizes"),
            field_name="compute_scale.input_feature_sizes",
        )
        for token_count in sorted(token_counts, reverse=True):
            for input_features in input_feature_sizes:
                test_cases.append(ComputeScaleCommonTestCase(m=token_count, k=input_features))

    return test_cases


@dataclasses.dataclass
class MLACommonTestCase:
    num_heads: int
    batch_size: int
    input_len: int
    is_context_phase: bool
    kv_cache_block_size: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    model_name: str


def _get_mla_case_specs(is_context: bool):
    test_cases = []

    model_config_list = _model_case_values("mla")
    mla_sweep = _required_base_common_case_values("mla")

    if is_context:
        b_list = _as_int_list(mla_sweep.get("context_batch_sizes"), field_name="mla.context_batch_sizes")
        s_list = _as_int_list(mla_sweep.get("context_sequence_lengths"), field_name="mla.context_sequence_lengths")
        max_tokens = int(mla_sweep["max_context_tokens"])
    else:
        b_list = _as_int_list(mla_sweep.get("generation_batch_sizes"), field_name="mla.generation_batch_sizes")
        s_list = _as_int_list(
            mla_sweep.get("generation_target_sequence_lengths"),
            field_name="mla.generation_target_sequence_lengths",
        )
        max_tokens = int(mla_sweep["max_generation_tokens"])
    kv_cache_block_size_list = _as_int_list(
        mla_sweep.get("kv_cache_block_sizes"),
        field_name="mla.kv_cache_block_sizes",
    )

    for (
        s,
        b,
        kv_cache_block_size,
        model_config,
    ) in itertools.product(
        s_list,
        b_list,
        kv_cache_block_size_list,
        model_config_list,
    ):
        if b * s > max_tokens:
            continue

        test_cases.append(
            MLACommonTestCase(
                num_heads=int(model_config["num_heads"]),
                input_len=s if is_context else s - 1,
                batch_size=b,
                is_context_phase=is_context,
                kv_cache_block_size=kv_cache_block_size,
                q_lora_rank=int(model_config["q_lora_rank"]),
                kv_lora_rank=int(model_config["kv_lora_rank"]),
                qk_nope_head_dim=int(model_config["qk_nope_head_dim"]),
                qk_rope_head_dim=int(model_config["qk_rope_head_dim"]),
                v_head_dim=int(model_config["v_head_dim"]),
                model_name=str(model_config["model_path"]),
            )
        )

    return test_cases


def get_context_mla_case_specs():
    return _get_mla_case_specs(is_context=True)


def get_generation_mla_case_specs():
    return _get_mla_case_specs(is_context=False)


# =============================================================================
# Mamba2 SSM Test Cases
# =============================================================================


@dataclasses.dataclass
class Mamba2CommonTestCase:
    """Test case configuration for Mamba2 SSM benchmarking."""

    phase: str  # "context" or "generation"
    d_model: int  # hidden_size
    d_state: int  # SSM state dimension
    d_conv: int  # Conv1d kernel size
    nheads: int  # Number of Mamba heads
    head_dim: int  # Dimension per head
    n_groups: int  # Number of groups for B, C matrices
    chunk_size: int  # Chunk size for SSM scan
    num_tokens_list: Optional[list[int]]  # For context phase (continuous batching)
    batch_size_list: Optional[list[int]]  # For generation phase, or context static batching
    seq_len_list: Optional[list[int]]  # For context phase with static batching
    model_name: str


def get_common_mamba2_test_cases() -> list[Mamba2CommonTestCase]:
    """
    Generate common test cases for Mamba2 SSM benchmarking.

    Includes configurations for:
    - Nemotron-H 3-30B (primary target)
    - Other potential Mamba2-based models

    Returns:
        List of Mamba2CommonTestCase configurations
    """
    test_cases: list[Mamba2CommonTestCase] = []
    mamba2_sweep = _required_base_common_case_values("mamba2")
    context_seq_lens = _as_int_list(
        mamba2_sweep.get("context_sequence_lengths"),
        field_name="mamba2.context_sequence_lengths",
    )
    context_batch_sizes = _as_int_list(
        mamba2_sweep.get("context_batch_sizes"),
        field_name="mamba2.context_batch_sizes",
    )
    generation_batch_sizes = _as_int_list(
        mamba2_sweep.get("generation_batch_sizes"),
        field_name="mamba2.generation_batch_sizes",
    )

    model_config_list = _model_case_values("mamba2")

    for model_config in model_config_list:
        d_model = int(model_config["d_model"])
        d_state = int(model_config["d_state"])
        d_conv = int(model_config["d_conv"])
        nheads = int(model_config["nheads"])
        head_dim = int(model_config["head_dim"])
        n_groups = int(model_config["n_groups"])
        chunk_size = int(model_config["chunk_size"])
        model_name = str(model_config["model_path"])

        # Context (prefill) test case
        test_cases.append(
            Mamba2CommonTestCase(
                phase="context",
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                nheads=nheads,
                head_dim=head_dim,
                n_groups=n_groups,
                chunk_size=chunk_size,
                num_tokens_list=None,  # Not used for static batching
                batch_size_list=context_batch_sizes,
                seq_len_list=context_seq_lens,
                model_name=model_name,
            )
        )

        # Generation (decode) test case
        test_cases.append(
            Mamba2CommonTestCase(
                phase="generation",
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                nheads=nheads,
                head_dim=head_dim,
                n_groups=n_groups,
                chunk_size=chunk_size,
                num_tokens_list=None,
                batch_size_list=generation_batch_sizes,
                seq_len_list=None,  # Not used for generation
                model_name=model_name,
            )
        )

    return test_cases


# =============================================================================
# GDN (Gated DeltaNet) Test Cases  — Qwen3.5 linear_attention layers
# =============================================================================


@dataclasses.dataclass
class GdnCommonTestCase:
    """Test case configuration for GDN (Gated DeltaNet) kernel benchmarking."""

    phase: str  # "context" or "generation"
    d_model: int  # hidden_size
    d_conv: int  # Conv1d kernel size
    num_k_heads: int  # Number of GDN key heads
    head_k_dim: int  # Key head dimension
    num_v_heads: int  # Number of GDN value heads
    head_v_dim: int  # Value head dimension
    batch_size_list: Optional[list[int]]
    seq_len_list: Optional[list[int]]  # For context phase; None for generation
    model_name: str


# =============================================================================
# MHC (DeepSeek-V4 Hash-Compressed attention) Test Cases
# =============================================================================


@dataclasses.dataclass
class MhcCommonTestCase:
    """Test case configuration for DeepSeek-V4 mHC pre/post kernel benchmarking."""

    phase: str  # "pre" or "post"
    hidden_size: int
    hc_mult: int
    num_tokens_list: list[int]
    model_name: str


def get_common_mhc_test_cases() -> list[MhcCommonTestCase]:
    """Generate common test cases for mHC (pre/post) kernel benchmarking."""
    mhc_sweep = _required_base_common_case_values("mhc")
    num_tokens_list = _as_int_list(mhc_sweep.get("token_counts"), field_name="mhc.token_counts")

    model_config_list = _model_case_values("mhc")

    test_cases: list[MhcCommonTestCase] = []
    for model_config in model_config_list:
        hidden_size = int(model_config["hidden_size"])
        hc_mult = int(model_config["hc_mult"])
        model_name = str(model_config["model_path"])
        for phase in ("pre", "post"):
            test_cases.append(
                MhcCommonTestCase(
                    phase=phase,
                    hidden_size=hidden_size,
                    hc_mult=hc_mult,
                    num_tokens_list=num_tokens_list,
                    model_name=model_name,
                )
            )
    return test_cases


def get_common_gdn_test_cases() -> list[GdnCommonTestCase]:
    """
    Generate common test cases for GDN (Gated DeltaNet) kernel benchmarking.

    Covers all 8 unique dimension sets across the full Qwen3.5 collection
    for both context (prefill) and generation (decode) phases.
    """
    test_cases: list[GdnCommonTestCase] = []
    gdn_sweep = _required_base_common_case_values("gdn")
    context_seq_lens = _as_int_list(
        gdn_sweep.get("context_sequence_lengths"),
        field_name="gdn.context_sequence_lengths",
    )
    context_batch_sizes = _as_int_list(
        gdn_sweep.get("context_batch_sizes"),
        field_name="gdn.context_batch_sizes",
    )
    generation_batch_sizes = _as_int_list(
        gdn_sweep.get("generation_batch_sizes"),
        field_name="gdn.generation_batch_sizes",
    )

    model_config_list = _model_case_values("gdn")

    for model_config in model_config_list:
        d_model = int(model_config["d_model"])
        d_conv = int(model_config["d_conv"])
        num_k_heads = int(model_config["num_k_heads"])
        head_k_dim = int(model_config["head_k_dim"])
        num_v_heads = int(model_config["num_v_heads"])
        head_v_dim = int(model_config["head_v_dim"])
        model_name = str(model_config["model_path"])

        # Context (prefill) test case
        test_cases.append(
            GdnCommonTestCase(
                phase="context",
                d_model=d_model,
                d_conv=d_conv,
                num_k_heads=num_k_heads,
                head_k_dim=head_k_dim,
                num_v_heads=num_v_heads,
                head_v_dim=head_v_dim,
                batch_size_list=context_batch_sizes,
                seq_len_list=context_seq_lens,
                model_name=model_name,
            )
        )

        # Generation (decode) test case
        test_cases.append(
            GdnCommonTestCase(
                phase="generation",
                d_model=d_model,
                d_conv=d_conv,
                num_k_heads=num_k_heads,
                head_k_dim=head_k_dim,
                num_v_heads=num_v_heads,
                head_v_dim=head_v_dim,
                batch_size_list=generation_batch_sizes,
                seq_len_list=None,
                model_name=model_name,
            )
        )

    return test_cases


# ═══════════════════════════════════════════════════════════════════════
# DeepSeek-V4-Flash test cases
# ═══════════════════════════════════════════════════════════════════════
# Used by ``collector.sglang.collect_dsv4_flash_attn`` (full-module bench)
# and ``collector.sglang.deepseekv4_sparse_modules`` (sparse kernel bench).
# Both backends re-export the relevant ``get_*`` functions so collect.py
# can resolve them via getattr on each per-backend module.


def _dsv4_flash_config() -> dict:
    configs = _model_case_values("dsv4_flash", apply_model_filter=False)
    if not configs:
        raise RuntimeError("model_case_values.dsv4_flash is missing from model case YAML")
    return configs[0]


def _dsv4_flash_attention_kinds() -> tuple[str, ...]:
    return tuple(str(kind) for kind in _dsv4_flash_config().get("attention_kinds", ["csa", "hca"]))


_DSV4_FLASH_CONFIG = _dsv4_flash_config()
_DSV4_FLASH_MODEL_PATH = str(_DSV4_FLASH_CONFIG["model_path"])
DSV4_FLASH_ATTN_KINDS = _dsv4_flash_attention_kinds()
_DSV4_FLASH_MODULE_BATCH_SIZES = list(_DSV4_FLASH_CONFIG["module_batch_sizes"])
_DSV4_FLASH_MODULE_SEQ_LENGTHS = list(_DSV4_FLASH_CONFIG["module_sequence_lengths"])
_DSV4_FLASH_MODULE_TP_SIZES = list(_DSV4_FLASH_CONFIG["module_tp_sizes"])
_DSV4_FLASH_SPARSE_BS_LIST = list(_DSV4_FLASH_CONFIG["sparse_batch_sizes"])
_DSV4_FLASH_SPARSE_ISL_LIST = list(_DSV4_FLASH_CONFIG["sparse_input_lengths"])
_DSV4_FLASH_SPARSE_PAST_KV_LIST = list(_DSV4_FLASH_CONFIG["sparse_past_kv_lengths"])
_DSV4_FLASH_SPARSE_CHUNK_PREFILL_SIZE = int(_DSV4_FLASH_CONFIG["sparse_chunk_prefill_size"])
_DSV4_FLASH_SPARSE_MAX_FULL_S = int(_DSV4_FLASH_CONFIG["sparse_max_full_sequence_length"])
_DSV4_FLASH_SPARSE_TP_LIST_ATTN = list(_DSV4_FLASH_CONFIG["sparse_tp_sizes"]["hca_attn"])
_DSV4_FLASH_SPARSE_TP_LIST_INDEXER = list(_DSV4_FLASH_CONFIG["sparse_tp_sizes"]["paged_mqa_logits"])


def _dsv4_flash_active() -> bool:
    """Honour the ``--model-path`` (``COLLECTOR_MODEL_PATH``) filter.

    All V4-Flash test cases are V4-Flash-only by construction.  When the
    user filters to a different model, every V4-Flash op must emit zero
    cases so the collector skips it.
    """
    filt = _get_model_path_filter()
    return filt is None or filt == _dsv4_flash_config()["model_path"]


def _dsv4_flash_model_path() -> str:
    filt = _get_model_path_filter()
    return filt if filt is not None else str(_dsv4_flash_config()["model_path"])


def _has_native_fp4_experts() -> bool:
    """True when the device has native FP4 tensor cores (Blackwell sm_100+).

    DSv4-Flash's checkpoint ships routed experts in FP4. When the collector
    sets ``server_args.quantization="fp8"`` (our ``gemm_type="fp8_block"``
    sweep) sglang's FusedMoE quant_method tries to materialize these
    experts. The fork-specific check in ``sglang/srt/layers/quantization/
    fp8.py:process_weights_after_loading_block_quant`` raises
    ``NotImplementedError: DeepSeekV4 FP4 experts now require a native FP4
    MoE backend.`` before any ``ignored_layers`` / ``is_layer_skipped``
    logic gets a chance — both the upstream ``SGLANG_FP8_IGNORED_LAYERS``
    env var and ``quantization_config.ignored_layers`` are bypassed by
    that hard-coded check. The sglang error message suggests
    ``--moe-runner-backend marlin``, but Marlin software-emulates FP4 with
    INT4 unpack + bf16 compute, producing perf numbers that don't reflect
    how DSv4-Flash actually deploys on Hopper. So we skip the fp8_block
    sweep on pre-Blackwell entirely; the bfloat16 sweep still runs and
    measures the projection path Hopper would deploy.
    """
    try:
        import torch as _t

        if not _t.cuda.is_available():
            return False
        return _t.cuda.get_device_capability(0)[0] >= 10
    except Exception:
        return False


def _dsv4_flash_module_precision_combos(phase: str):
    """``(compute_dtype, kv_cache_dtype, gemm_type)`` triples.

    DeepseekV4ForCausalLM rejects bfloat16 KV cache (asserts at load time),
    so we only emit fp8 KV.  ``gemm_type`` switches projection dispatch:
      * ``bfloat16``  — projections through cuBLASLt nvjet kernels
      * ``fp8_block`` — fp8 block-quantised weights → DeepGEMM
                        ``sm90_fp8_gemm_1d2d_impl`` (matches production)

    ``fp8_block`` is omitted on pre-Blackwell parts; see
    ``_has_native_fp4_experts`` for the kernel-side rationale.
    """
    del phase
    combos = [("bfloat16", "fp8", "bfloat16")]
    if _has_native_fp4_experts():
        combos.append(("bfloat16", "fp8", "fp8_block"))
    else:
        print(
            "[dsv4-flash-test-cases] device lacks native FP4 experts (pre-Blackwell); "
            "omitting fp8_block from gemm_type sweep"
        )
    return combos


def _dsv4_flash_module_filter_pairs(mode: str, batch_sizes, seq_lens):
    """Drop ``(bs, sl)`` pairs that exceed KV pool / kernel limits.

    Context (b * s):
        ≤ 8192 — matches sglang's default ``chunked_prefill_size``.
    Generation (b * s):
        ≤ 1M overall, with per-sl batch caps for long contexts (sl≥8192→bs≤64,
        sl≥32768→bs≤16, sl≥65536→bs≤8, sl≥131072→bs≤4, sl≥262144→bs≤2,
        sl≥524288→bs==1).  Ensures bs=1 is always allowed at every sl.
    """
    is_context = mode == "context"
    pairs = []
    for bs in batch_sizes:
        for sl in seq_lens:
            if is_context:
                if bs * sl > 8192:
                    continue
            else:
                if bs * sl > 1024 * 1024:
                    continue
                if sl >= 524288 and bs > 1:
                    continue
                if sl >= 262144 and bs > 2:
                    continue
                if sl >= 131072 and bs > 4:
                    continue
                if sl >= 65536 and bs > 8:
                    continue
                if sl >= 32768 and bs > 16:
                    continue
                if sl >= 8192 and bs > 64:
                    continue
            pairs.append((bs, sl))
    return pairs


def _build_dsv4_flash_module_test_cases(mode: str, attn_kinds=None):
    """One case per ``(attn_kind, tp_size, gemm_type, batch_size)``.

    Test case shape (9 elements; ``perf_filename`` is bound by collect.py
    via OpEntry, NOT in the tuple)::

        [0, batch_size, tp_size, kv_cache_dtype, compute_dtype, gemm_type,
         model_path, attn_kind, attention_backend]

    Each spawned subprocess builds ONE ``ModelRunner`` for ``(bs, max_sl)``
    and sweeps every valid sl for that bs internally.
    """
    config = _dsv4_flash_config()
    attn_kinds = tuple(attn_kinds) if attn_kinds is not None else _dsv4_flash_attention_kinds()
    pairs = _dsv4_flash_module_filter_pairs(
        mode,
        config["module_batch_sizes"],
        config["module_sequence_lengths"],
    )
    bs_set = sorted({bs for bs, _ in pairs})
    model_path = _dsv4_flash_model_path()

    cases: list[list] = []
    for attn_kind in attn_kinds:
        for compute_dtype, kv_dtype, gemm_type in _dsv4_flash_module_precision_combos(mode):
            for tp_size in config["module_tp_sizes"]:
                for bs in bs_set:
                    cases.append(
                        [
                            0,
                            bs,
                            tp_size,
                            kv_dtype,
                            compute_dtype,
                            gemm_type,
                            model_path,
                            attn_kind,
                            None,
                        ]
                    )
    return cases


def get_dsv4_flash_csa_context_test_cases():
    if not _dsv4_flash_active():
        return []
    return _build_dsv4_flash_module_test_cases("context", ("csa",))


def get_dsv4_flash_hca_context_test_cases():
    if not _dsv4_flash_active():
        return []
    return _build_dsv4_flash_module_test_cases("context", ("hca",))


def get_dsv4_flash_csa_generation_test_cases():
    if not _dsv4_flash_active():
        return []
    return _build_dsv4_flash_module_test_cases("generation", ("csa",))


def get_dsv4_flash_hca_generation_test_cases():
    if not _dsv4_flash_active():
        return []
    return _build_dsv4_flash_module_test_cases("generation", ("hca",))


DSV4_FLASH_SPARSE_KERNELS = ("paged_mqa_logits", "hca_attn")


def _build_dsv4_flash_sparse_test_cases(
    kernels=DSV4_FLASH_SPARSE_KERNELS,
    bs_list=None,
    isl_list=None,
    past_kv_list=None,
    tp_list_attn=None,
    tp_list_indexer=None,
):
    """Generate ``(bs, isl, past_kv, tp_size, kernel, model)`` tuples.

    Filters mirror sglang prefill scheduler:
      * bs x isl ≤ chunked_prefill_size = 8192   — new-token budget per chunk
      * bs x (isl + past_kv) ≤ 1M                — model context cap
    """
    config = _dsv4_flash_config()
    sparse_tp_sizes = config["sparse_tp_sizes"]
    bs_list = list(bs_list) if bs_list is not None else list(config["sparse_batch_sizes"])
    isl_list = list(isl_list) if isl_list is not None else list(config["sparse_input_lengths"])
    past_kv_list = list(past_kv_list) if past_kv_list is not None else list(config["sparse_past_kv_lengths"])
    tp_list_attn = list(tp_list_attn) if tp_list_attn is not None else list(sparse_tp_sizes["hca_attn"])
    tp_list_indexer = (
        list(tp_list_indexer) if tp_list_indexer is not None else list(sparse_tp_sizes["paged_mqa_logits"])
    )
    sparse_chunk_prefill_size = int(config["sparse_chunk_prefill_size"])
    sparse_max_full_s = int(config["sparse_max_full_sequence_length"])
    model_path = _dsv4_flash_model_path()

    cases = []
    for kernel in kernels:
        tp_list = tp_list_attn if kernel == "hca_attn" else tp_list_indexer
        for tp_size in tp_list:
            for bs in bs_list:
                for isl in isl_list:
                    if bs * isl > sparse_chunk_prefill_size:
                        continue
                    for past_kv in past_kv_list:
                        if bs * (isl + past_kv) > sparse_max_full_s:
                            continue
                        full_s = isl + past_kv
                        if kernel == "paged_mqa_logits" and full_s < 4:
                            continue
                        if kernel == "hca_attn" and full_s < 64:
                            continue
                        cases.append(
                            [
                                bs,
                                isl,
                                past_kv,
                                tp_size,
                                kernel,
                                model_path,
                            ]
                        )
    return cases


def _dsv4_flash_sparse_smoke_or_full(kernel: str):
    if not _dsv4_flash_active():
        return []
    if "--smoke" in sys.argv:
        return [[1, 1024, 8192, 1, kernel, _dsv4_flash_model_path()]]
    return _build_dsv4_flash_sparse_test_cases(kernels=(kernel,))


def get_dsv4_flash_paged_mqa_logits_test_cases():
    """paged_mqa_logits sparse-kernel sweep (CSA indexer scoring)."""
    return _dsv4_flash_sparse_smoke_or_full("paged_mqa_logits")


def get_dsv4_flash_hca_attn_test_cases():
    """hca_attn sparse-kernel sweep (HCA c128 sparse FMLA)."""
    return _dsv4_flash_sparse_smoke_or_full("hca_attn")
