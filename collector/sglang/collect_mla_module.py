# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MLA Module Collector for SGLang — unified MLA and DSA benchmarking.

Profiles the complete attention module forward pass at the model-runner level,
using SGLang's own ServerArgs → ModelRunner → ForwardBatch pipeline with dummy
weights. Op names and data schema are aligned with vLLM and TRT-LLM
collect_mla_module.py so that perf_database queries work across frameworks.

Supported models, attention types, and micro-sweeps are defined in collector v2
YAML and loaded through collector.case_generator.

Usage:
    # DSA context phase (DeepSeek-V3.2 style)
    SGLANG_LOAD_FORMAT=dummy SGLANG_TEST_NUM_LAYERS=2 \
        python collect_mla_module.py --mode context --attn-type dsa

    # MLA generation phase (DeepSeek-V3 style)
    SGLANG_LOAD_FORMAT=dummy SGLANG_TEST_NUM_LAYERS=2 \
        python collect_mla_module.py --mode generation --attn-type mla
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import types
from importlib.metadata import version as get_version

import numpy as np
import torch

# The collector writes an AIC-local config.json for DeepSeek-V3.2/GLM-5 so
# SGLang can load them through its DeepSeek-V3 NSA path.  Letting SGLang's
# layer-count backup auto-rewrite run here can remap DeepSeek-V3.2 to the
# DeepSeek-V4 compressed-attention path before the DSA module is benchmarked.
os.environ.setdefault("SGLANG_APPLY_CONFIG_BACKUP", "none")


def _piecewise_forward_context(
    set_forward_context,
    forward_batch,
    attention_layers,
    quant_config,
    moe_layers,
    moe_fusions,
    *,
    dsa_indexers=None,
):
    kwargs = {}
    try:
        import inspect

        if "dsa_indexers" in inspect.signature(set_forward_context).parameters:
            kwargs["dsa_indexers"] = dsa_indexers
    except (TypeError, ValueError):
        pass

    return set_forward_context(
        forward_batch,
        attention_layers,
        quant_config,
        moe_layers,
        moe_fusions,
        **kwargs,
    )


try:
    from helper import benchmark_with_power, get_sm_version, log_perf
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from helper import benchmark_with_power, get_sm_version, log_perf

try:
    from collector.case_generator import get_mla_module_model_specs, get_mla_module_sweep_spec
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from case_generator import get_mla_module_model_specs, get_mla_module_sweep_spec

try:
    from collector.sglang.runtime_limits import (
        alloc_prefix_indices as _alloc_prefix_indices,
    )
    from collector.sglang.runtime_limits import (
        dsa_indexer_prefill_shape_is_supported,
        dsa_indexer_total_kv_tokens_supported,
        dsa_indexer_workspace_bytes,
        dsa_indexer_workspace_limit_bytes,
        required_kv_tokens,
        required_prefill_extend_tokens,
        sglang_dsa_mqa_logits_chunking_supported,
    )
    from collector.sglang.runtime_limits import (
        kv_pool_capacity_tokens as _kv_pool_capacity_tokens,
    )
    from collector.sglang.runtime_limits import (
        runtime_chunk_size as _runtime_chunk_size,
    )
    from collector.sglang.runtime_limits import (
        temporarily_chunked_alloc_extend as _temporarily_chunked_alloc_extend,
    )
except ModuleNotFoundError:
    from runtime_limits import (
        alloc_prefix_indices as _alloc_prefix_indices,
    )
    from runtime_limits import (
        dsa_indexer_prefill_shape_is_supported,
        dsa_indexer_total_kv_tokens_supported,
        dsa_indexer_workspace_bytes,
        dsa_indexer_workspace_limit_bytes,
        required_kv_tokens,
        required_prefill_extend_tokens,
        sglang_dsa_mqa_logits_chunking_supported,
    )
    from runtime_limits import (
        kv_pool_capacity_tokens as _kv_pool_capacity_tokens,
    )
    from runtime_limits import (
        runtime_chunk_size as _runtime_chunk_size,
    )
    from runtime_limits import (
        temporarily_chunked_alloc_extend as _temporarily_chunked_alloc_extend,
    )


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

# Perf-database-compatible dtype strings → SGLang ServerArgs kv_cache_dtype values.
# The perf DB uses enum names like "fp8"; SGLang uses "fp8_e4m3".
SGLANG_KV_DTYPE: dict[str, str] = {
    "bfloat16": "bfloat16",
    "fp8": "fp8_e4m3",
}

# AIC's cached HuggingFace model configs — avoids HF downloads in CI.
_MODEL_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
    "aiconfigurator",
    "model_configs",
)
_GLM5_DSA_ARCHITECTURE = "GlmMoeDsaForCausalLM"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _parse_int_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item) for item in value.split(",") if item]


def _filter_cases_from_env(test_cases, *, is_prefill: bool, attn_type: str):
    if attn_type != "dsa":
        return test_cases

    phase = "CONTEXT" if is_prefill else "GENERATION"
    seq_filter = _parse_int_list(os.environ.get(f"AIC_DSA_{phase}_SEQ_LENS"))
    batch_filter = _parse_int_list(os.environ.get(f"AIC_DSA_{phase}_BATCH_SIZES"))
    prefix_filter = _parse_int_list(os.environ.get("AIC_DSA_CONTEXT_PREFIX_LENS")) if is_prefill else None

    if seq_filter is None and prefix_filter is None and batch_filter is None:
        return test_cases

    seq_set = set(seq_filter) if seq_filter is not None else None
    prefix_set = set(prefix_filter) if prefix_filter is not None else None
    batch_set = set(batch_filter) if batch_filter is not None else None

    filtered = []
    for bs, seq_len, ip, prefix_len in test_cases:
        if batch_set is not None and bs not in batch_set:
            continue
        if seq_set is not None and seq_len not in seq_set:
            continue
        if prefix_set is not None and prefix_len not in prefix_set:
            continue
        filtered.append((bs, seq_len, ip, prefix_len))
    print(f"[DSA] Env-filtered {phase.lower()} cases: {len(filtered)}/{len(test_cases)}")
    return filtered


def _is_glm5_dsa_model(model_id: str) -> bool:
    return _module_model_architecture(model_id) == _GLM5_DSA_ARCHITECTURE


def _enable_glm5_dsa_piecewise_graph(attn_type: str, model_id: str) -> bool:
    if _env_flag("AIC_DISABLE_PIECEWISE_CUDA_GRAPH"):
        return False
    # SGLang 0.5.10 full-model piecewise capture rejects the GLM-5 DSA
    # attention path; module-level piecewise replay remains enabled below.
    return _env_flag("AIC_ENABLE_PIECEWISE_CUDA_GRAPH") and attn_type == "dsa" and _is_glm5_dsa_model(model_id)


def _piecewise_cuda_graph_tokens_for_cases(test_cases, is_prefill: bool) -> list[int] | None:
    env_tokens = _parse_int_list(os.environ.get("AIC_PIECEWISE_CUDA_GRAPH_TOKENS"))
    if env_tokens is not None:
        return env_tokens

    tokens = set()
    for batch_size, seq_len, *_rest in test_cases:
        token_count = int(batch_size) * int(seq_len) if is_prefill else int(batch_size)
        if is_prefill:
            token_count = ((token_count + 7) // 8) * 8
        tokens.add(token_count)
    return sorted(tokens)


def _generation_cuda_graph_enabled_for_tokens(num_tokens: int) -> bool:
    """Match SGLang decode CUDA graph coverage for module microbenchmarks."""
    if _env_flag("AIC_DISABLE_CUDA_GRAPH"):
        return False

    cuda_graph_bs = _parse_int_list(os.environ.get("AIC_CUDA_GRAPH_BS"))
    if cuda_graph_bs is not None:
        return int(num_tokens) in set(cuda_graph_bs)

    max_bs = int(os.environ.get("AIC_CUDA_GRAPH_MAX_BS", "256"))
    return 0 < int(num_tokens) <= max_bs


def _resolve_local_model_path(model_id: str) -> str:
    """Resolve a HuggingFace model ID to a local config directory.

    Uses AIC's cached model configs from src/aiconfigurator/model_configs/
    so that the collection pipeline never needs HuggingFace network access.
    The function patches model_type and architectures for sglang AutoConfig
    compatibility and strips auto_map to prevent any remote code download.

    SGLang's AutoConfig.from_pretrained() doesn't recognize "deepseek_v32" or
    "glm_moe_dsa" model types.  The workaround (matching sglang's own
    _load_deepseek_v32_model approach) is to present these as "deepseek_v3".
    DSA-specific fields (index_topk, index_head_dim, etc.) are preserved in
    the config and sglang uses those for DSA detection via is_deepseek_nsa().

    Falls back to the original HF model ID if local config is not found.
    """
    config_file = os.path.join(_MODEL_CONFIG_DIR, f"{model_id.replace('/', '--')}_config.json")
    if not os.path.exists(config_file):
        return model_id

    with open(config_file) as f:
        config = json.load(f)

    # Normalise model_type so sglang's AutoConfig recognises it.
    original_arch = None
    if config.get("model_type") in ("deepseek_v32", "glm_moe_dsa"):
        original_arch = (config.get("architectures") or [None])[0]
        config["architectures"] = ["DeepseekV3ForCausalLM"]
        config["model_type"] = "deepseek_v3"

    # Strip auto_map to prevent transformers from attempting to download
    # custom config/model classes from HuggingFace when trust_remote_code=True.
    config.pop("auto_map", None)

    # Re-apply GLM-5 arch-specific overrides that sglang would normally
    # apply in server_args.py:1609-1612 based on
    # hf_config.architectures[0] == "GlmMoeDsaForCausalLM" - but we just
    # rewrote that string to "DeepseekV3ForCausalLM" above so AutoConfig
    # can find the model_type, so sglang's gate never fires. Without
    # this, on Blackwell every GLM-5 DSA context probe with max_kv_len
    # <= 2048 (the default DENSE_ATTN threshold) dispatches to
    # _forward_standard_mha -> flashinfer.prefill.trtllm_ragged_attention_deepseek
    # which asserts on GLM-5's v_head_dim=256 and kills the subprocess
    # after ~14 rows/bucket.
    if (
        original_arch == "GlmMoeDsaForCausalLM"
        and get_sm_version() >= 100
        and "SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD" not in os.environ
    ):
        os.environ["SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD"] = "0"

    tmp_dir = os.path.join(
        tempfile.gettempdir(),
        f"aic_sglang_config_{model_id.replace('/', '_')}_{os.getpid()}",
    )
    os.makedirs(tmp_dir, exist_ok=True)
    with open(os.path.join(tmp_dir, "config.json"), "w") as f:
        json.dump(config, f)

    quant_config_file = os.path.join(
        _MODEL_CONFIG_DIR,
        f"{model_id.replace('/', '--')}_hf_quant_config.json",
    )
    if os.path.exists(quant_config_file):
        shutil.copyfile(quant_config_file, os.path.join(tmp_dir, "hf_quant_config.json"))

    return tmp_dir


# ═══════════════════════════════════════════════════════════════════════
# Precision Combos
# ═══════════════════════════════════════════════════════════════════════


def _get_precision_combos(phase: str):
    """Return (compute_dtype, kv_cache_dtype, gemm_type) triples for a phase.

    All strings are perf-database-compatible (not SGLang-native).

    SGLang precision axes:
      compute_dtype:  always "bfloat16" (DSA / NSA kernels run bf16 FMHA;
                      on B200 decode with fp8 KV the trtllm path runs fp8
                      FMHA internally, but the latency is captured under the
                      fp8 KV row).
      kv_cache_dtype: "bfloat16" always; "fp8" on SM >= 90 (Hopper+).
      gemm_type:      "bfloat16" for bf16 weights; "fp8_block" on SM >= 89,
                      in which case load_model_runner launches sglang with
                      quantization="fp8" so the inner attention projections
                      (q_b_proj, kv_b_proj, o_proj, wq_b, wk) and the fp8
                      paged MQA scoring kernel fire on real fp8 weights.

    fp4_e2m1 omitted — SGLang supports it on SM >= 100, but KVCacheQuantMode
    enum has no fp4 entry, so perf_database cannot consume it yet.
    """
    sm = get_sm_version()
    kv_dtypes = ["bfloat16"]
    if sm >= 90:
        kv_dtypes.append("fp8")
    combos = [("bfloat16", kv, "bfloat16") for kv in kv_dtypes]
    if sm >= 89:
        combos += [("bfloat16", kv, "fp8_block") for kv in kv_dtypes]
    return combos


def _get_backends(attn_type: str):
    """Return the attention backend string to use for a given attention type.

    For DSA: returns "nsa" — SGLang auto-detects nsa_prefill_backend /
    nsa_decode_backend based on SM + kv_cache_dtype.
    For MLA: returns the best available backend based on SM version.
    Aligns with sglang's _get_default_attn_backend() in server_args.py.
    """
    sm = get_sm_version()
    if attn_type == "dsa":
        return "nsa"
    else:
        if sm >= 100:
            return "trtllm_mla"
        elif sm >= 90:
            return "fa3"
        else:
            # sglang defaults MLA to "triton" on SM < 90; flashinfer MLA
            # (BatchMLAPagedAttentionWrapper) is not validated on these GPUs.
            return "triton"


def _get_mla_backend_list() -> list[str]:
    """Return all MLA backends to sweep for wideep MLA collection.

    Per-architecture backends:
      SM >= 100: ["trtllm_mla"]  — flashinfer MLA is not supported on Blackwell;
                  sglang auto-promotes to trtllm_mla and then fails kv_cache_dtype
                  validation.  Existing B200 perf data contains only trtllm_mla.
      SM >= 90:  ["flashinfer", "fa3"]
      SM < 90:   []  — sglang defaults MLA to "triton" on SM < 90 (A100 etc.);
                  flashinfer MLA kernels (BatchMLAPagedAttentionWrapper) are not
                  validated on SM < 90 and crash the subprocess.  Skip wideep MLA
                  collection on these GPUs; kernel-level collectors (collect_mla.py)
                  already capture per-kernel latency via mocks.
    """
    sm = get_sm_version()
    if sm >= 100:
        return ["trtllm_mla"]
    elif sm >= 90:
        return ["flashinfer", "fa3"]
    else:
        return []


# ═══════════════════════════════════════════════════════════════════════
# Test Case Generation
# ═══════════════════════════════════════════════════════════════════════


def _module_model_architecture(model_path: str) -> str:
    """Return the YAML-declared architecture for a module benchmark model."""
    for spec in get_mla_module_model_specs(apply_model_filter=False):
        if spec.model_path == model_path:
            return spec.architecture
    return "unknown"


def _module_model_native_heads(model_path: str) -> int:
    """Return native attention heads for TP-sim filtering."""
    for spec in get_mla_module_model_specs(apply_model_filter=False):
        if spec.model_path == model_path:
            return spec.native_num_heads
    return 128


def get_context_test_cases(attn_type: str):
    """Context-phase test cases.

    Returns list of [seq_len, batch_size, num_heads, kv_cache_dtype,
                     compute_dtype, gemm_type].
    """
    sweep = get_mla_module_sweep_spec("sglang")
    cases = []
    for compute_dtype, kv_dtype, gemm_type in _get_precision_combos("context"):
        for num_heads in sweep.inner_sweep_head_counts:
            for batch_size in sweep.context_batch_sizes:
                for seq_len in sweep.context_sequence_lengths:
                    if batch_size * seq_len > sweep.context_max_tokens:
                        continue
                    if (
                        seq_len >= sweep.context_large_sequence_min
                        and batch_size > sweep.context_large_sequence_max_batch_size
                    ):
                        continue
                    cases.append([seq_len, batch_size, num_heads, kv_dtype, compute_dtype, gemm_type])
    return cases


def get_generation_test_cases(attn_type: str):
    """Generation-phase test cases.

    Returns list of [kv_cache_len, batch_size, num_heads, kv_cache_dtype,
                     compute_dtype, gemm_type].
    """
    sweep = get_mla_module_sweep_spec("sglang")
    cases = []
    for compute_dtype, kv_dtype, gemm_type in _get_precision_combos("generation"):
        for num_heads in sweep.inner_sweep_head_counts:
            for batch_size in sweep.generation_batch_sizes:
                for seq_len in sweep.generation_sequence_lengths:
                    if batch_size * seq_len > sweep.generation_max_tokens:
                        continue
                    if (
                        seq_len >= sweep.generation_large_sequence_min
                        and batch_size > sweep.generation_large_sequence_max_batch_size
                    ):
                        continue
                    cases.append([seq_len, batch_size, num_heads, kv_dtype, compute_dtype, gemm_type])
    return cases


def _get_module_precision_combos():
    """Return a single baseline precision combo for module-level benchmarks.

    Module benchmarks spawn a subprocess per (model, precision, heads) group.
    Each subprocess loads a full ModelRunner (~15-20 s), so fewer groups means
    faster overall collection.  Use a single bfloat16 baseline — matching the
    wideep MLA pattern — so the total subprocess count stays manageable.

    Precision-specific kernel performance is already captured by kernel-level
    collectors (collect_mla.py, collect_attn.py).  The module benchmark
    captures module-level overhead (scheduling, memory management, attention
    dispatch) which is largely precision-independent with dummy weights.
    """
    return get_mla_module_sweep_spec("sglang").module_precision_combos


def _dsa_context_prefix_shape_is_valid(batch_size: int, seq_len: int, prefix_len: int) -> bool:
    """Return whether a DSA prefix-context sample is structurally valid.

    Single-token extension is a decode/generation shape.  SGLang's DSA prefill
    indexer can illegal-access on that shape, so context collection skips it
    before launching kernels.  Capacity is checked after ModelRunner
    construction against the actual runtime limits.
    """
    return prefix_len >= 0 and dsa_indexer_prefill_shape_is_supported(batch_size, seq_len)


def _build_module_test_cases(attn_type: str, mode: str):
    """Build one test case per unique (local heads, target TP, precision, model) group.

    Output format: [seq_len, batch_size, num_heads, kv_cache_dtype,
                    compute_dtype, gemm_type, model_path,
                    attn_type, attention_backend, target_tp_size]

    Each test case triggers a subprocess that sweeps all (batch_size, seq_len)
    combinations internally, so we only need one entry per group — not one per
    individual point. seq_len and batch_size are set to 0 as placeholders.

    Uses the YAML ``module_precision_combos`` and ``top_level_head_counts``
    instead of the full inner sweep to keep
    the subprocess count low — each subprocess loads a full ModelRunner which
    takes ~15-20 s.  With 4 precision combos x 5 head counts = 20 subprocesses,
    the collection exceeds typical container timeouts.  1 combo x 2 heads per
    model ≈ 3 subprocesses fits comfortably within the 120 s sample timeout.

    attention_backend is None for DSA (resolved at runtime by _get_backends()).
    perf_filename is supplied by collect.py via functools.partial as a keyword
    argument, so it is not included in the test case tuple.
    """
    if attn_type == "dsa" and _skip_sm120_deepgemm_attention_modules():
        return []
    model_specs = get_mla_module_model_specs(attention_type=attn_type)
    sweep = get_mla_module_sweep_spec("sglang")
    cases = []
    for model_spec in model_specs:
        for compute_dtype, kv_dtype, gemm_type in _get_module_precision_combos():
            target_tps = sweep.module_tp_sizes if attn_type == "dsa" else [1]
            for target_tp_size in target_tps:
                if model_spec.native_num_heads % target_tp_size != 0:
                    continue
                num_heads = model_spec.native_num_heads // target_tp_size
                if num_heads not in sweep.inner_sweep_head_counts:
                    continue
                batch_sizes = sweep.context_batch_sizes if attn_type == "dsa" and mode == "context" else [0]
                for batch_size in batch_sizes:
                    cases.append(
                        [
                            0,
                            batch_size,
                            num_heads,
                            kv_dtype,
                            compute_dtype,
                            gemm_type,
                            model_spec.model_path,
                            attn_type,
                            None,
                            target_tp_size,
                        ]
                    )
    return cases


def _build_wideep_mla_test_cases(mode: str):
    """Build test cases for wideep MLA collection (backward-compatible).

    Output format: [seq_len, batch_size, num_heads, kv_cache_dtype,
                    compute_dtype, gemm_type, model_path,
                    attn_type, attention_backend]

    Matches the old collect_wideep_attn.py behavior:
    - Single precision combo (bfloat16 run, logged as fp8_block/fp8)
    - Sweeps multiple attention backends per SM version
    - Only DeepSeek-V3 (the MLA model), not V3.2/GLM-5 (DSA models)

    perf_filename is supplied by collect.py via functools.partial as a keyword
    argument, so it is not included in the test case tuple.
    """
    if _skip_sm120_deepgemm_attention_modules():
        return []
    model_specs = get_mla_module_model_specs(attention_type="mla", wideep_mla=True)
    sweep = get_mla_module_sweep_spec("sglang")
    backends = _get_mla_backend_list()
    cases = []
    for model_spec in model_specs:
        for backend in backends:
            for num_heads in sweep.inner_sweep_head_counts:
                if num_heads > model_spec.native_num_heads:
                    continue
                # Single precision: run with bfloat16, log as fp8_block/fp8
                cases.append(
                    [
                        0,
                        0,
                        num_heads,
                        "bfloat16",
                        "bfloat16",
                        "bfloat16",
                        model_spec.model_path,
                        "mla",
                        backend,
                    ]
                )
    return cases


def _skip_sm120_deepgemm_attention_modules() -> bool:
    """Return True when SGLang's DeepGEMM-backed module path is unsupported.

    On SM120 with the SGLang 0.5.10 runtime container, forcing these module
    collectors produced no usable rows:
    - WideEP MLA context/generation subprocesses failed after every shape was
      skipped, ending with "MLA module ... produced no perf rows".
    - DSA context/generation failed the same way, and the underlying DeepGEMM
      calls reported unsupported Blackwell paths, e.g.
      attention.hpp:136 and gemm.hpp:376 "Unsupported architecture".

    Keep the skip as the default for this collector/runtime combo so the RTX
    PRO 6000 collection reflects runnable SGLang 0.5.10 paths. Developers can
    set COLLECTOR_FORCE_DEEPGEMM_ATTENTION_MODULES=1 to repro or validate a
    newer runtime.
    """
    return os.environ.get("COLLECTOR_FORCE_DEEPGEMM_ATTENTION_MODULES") != "1" and get_sm_version() >= 120


def get_wideep_mla_context_test_cases():
    """collect.py entrypoint for wideep MLA context collection."""
    return _build_wideep_mla_test_cases(mode="context")


def get_wideep_mla_generation_test_cases():
    """collect.py entrypoint for wideep MLA generation collection."""
    return _build_wideep_mla_test_cases(mode="generation")


def get_dsa_context_module_test_cases():
    """collect.py entrypoint for DSA context module collection."""
    # DSA (NSA) requires SM >= 90.
    if get_sm_version() < 90:
        return []
    return _build_module_test_cases(attn_type="dsa", mode="context")


def get_dsa_generation_module_test_cases():
    """collect.py entrypoint for DSA generation module collection."""
    # DSA (NSA) requires SM >= 90.
    if get_sm_version() < 90:
        return []
    return _build_module_test_cases(attn_type="dsa", mode="generation")


# ═══════════════════════════════════════════════════════════════════════
# SGLang Helpers
# ═══════════════════════════════════════════════════════════════════════


def cleanup_distributed():
    """Clean up SGLang distributed environment if it exists."""
    import sglang.srt.distributed.parallel_state as parallel_state

    for var_name in ["_TP", "_PP", "_MOE_EP", "_MOE_TP", "_WORLD", "_PDMUX_PREFILL_TP_GROUP"]:
        if hasattr(parallel_state, var_name):
            setattr(parallel_state, var_name, None)

    import sglang.srt.eplb.expert_location as expert_location

    if hasattr(expert_location, "_global_expert_location_metadata"):
        expert_location._global_expert_location_metadata = None


def _ensure_fp8_block_quant_config(hf_cfg) -> None:
    """Populate hf_config.quantization_config with weight_block_size for fp8_block.

    After _resolve_local_model_path rewrites the model_type to deepseek_v3, the
    JSON's ``quantization_config`` section may not be preserved as an attribute
    on the HF config object. sglang's _get_quantization_config then falls back
    to ``Fp8Config()`` with no weight_block_size, which flips Fp8LinearMethod
    to the channel-FP8 path — that path transposes the weight post-load and
    breaks downstream DeepseekV2 post_load_weights. Re-inject the block-scale
    fields so block_quant=True fires.
    """
    default_qc = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "fmt": "e4m3",
    }
    qc = getattr(hf_cfg, "quantization_config", None)
    if qc is None:
        hf_cfg.quantization_config = default_qc
        return
    if isinstance(qc, dict):
        for k, v in default_qc.items():
            if qc.get(k) is None:
                qc[k] = v
        return
    for k, v in default_qc.items():
        if getattr(qc, k, None) is None:
            setattr(qc, k, v)


def _patch_nsa_rope_contiguity(model_runner):
    """Workaround for sglang rope contiguity bugs on Blackwell (SM>=100).

    The NSA Indexer's _get_k_bf16 calls torch.split on q/k, producing
    non-contiguous views.  The JIT RoPE kernel rejects these:
      - <=0.5.9  rope.cuh:498 check_cuda_contiguous  (q/k assertion)
      - >=0.5.10 rope.cuh:299 TensorMatcher verify    (positions stride check)

    We monkey-patch rotary_emb.forward on each NSA Indexer layer to call
    .contiguous() on positions, q, and k before the kernel.
    """
    if get_sm_version() < 100:
        return

    for layer in model_runner.model.model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        indexer = getattr(attn, "indexer", None)
        if indexer is None:
            continue
        # MultiPlatformOp wraps the actual module
        actual = getattr(indexer, "_module", indexer)
        rotary_emb = getattr(actual, "rotary_emb", None)
        if rotary_emb is None:
            continue

        original_forward = rotary_emb.forward

        def _make_contiguous_forward(orig):
            def _forward(positions, query, key, fused_set_kv_buffer_arg=None):
                return orig(
                    positions.contiguous(),
                    query.contiguous(),
                    key.contiguous() if key is not None else key,
                    fused_set_kv_buffer_arg,
                )

            return _forward

        rotary_emb.forward = _make_contiguous_forward(original_forward)
        print(f"Patched rope contiguity for layer {layer}")


def _patch_nsa_indexer_compile_for_module_cuda_graph(attention_module) -> None:
    """Use eager indexer gate helpers during module-level CUDA graph capture.

    SGLang's piecewise runner compiles/captures the whole model path.  The AIC
    module collector captures just ``self_attn.forward``.  In that narrower
    capture, the NSA indexer's small ``@torch.compile`` gate helpers can lazily
    invoke inductor inside ``torch.cuda.graph(...)`` and fail on CPU->CUDA
    copies.  Binding the original wrapped functions keeps the same math while
    avoiding compile work inside graph capture.
    """
    indexer = getattr(attention_module, "indexer", None)
    if indexer is None:
        return
    actual_indexer = getattr(indexer, "_module", indexer)
    for name in (
        "_project_and_scale_head_gates",
        "_get_logits_head_gate",
        "_apply_q_scale_and_softmax_scale",
    ):
        bound = getattr(actual_indexer, name, None)
        if bound is None:
            continue
        raw = getattr(bound, "__wrapped__", None)
        if raw is None:
            raw = getattr(getattr(type(actual_indexer), name, None), "__wrapped__", None)
        if raw is not None:
            setattr(actual_indexer, name, types.MethodType(raw, actual_indexer))


class CudaIllegalAccessError(RuntimeError):
    """Stop the current subprocess after a CUDA illegal access poisons context."""


def _expect_module_attr(module, attr_name: str, expected: int, module_name: str) -> None:
    value = getattr(module, attr_name, None)
    if value != expected:
        raise RuntimeError(f"{module_name}.{attr_name}={value}, expected {expected}")


def _validate_dsa_tp_module_shapes(model_runner, local_num_heads: int, target_tp_size: int) -> None:
    """Verify the single-GPU DSA module emulates one target TP rank.

    The collector cannot launch a full distributed TP group for every
    micro-benchmark.  Instead it loads a single local-rank attention module with
    local ``num_attention_heads``.  That must also localize the TP-sharded
    projection GEMMs, not just the value logged as ``num_heads``.
    """
    if target_tp_size <= 1:
        return

    try:
        attn = model_runner.model.model.layers[0].self_attn
    except AttributeError as exc:
        raise RuntimeError("failed to locate DSA self_attn module for TP shape validation") from exc

    _expect_module_attr(attn, "num_heads", local_num_heads, "self_attn")
    _expect_module_attr(attn, "num_local_heads", local_num_heads, "self_attn")

    qk_nope_head_dim = int(attn.qk_nope_head_dim)
    qk_rope_head_dim = int(attn.qk_rope_head_dim)
    v_head_dim = int(attn.v_head_dim)
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    q_b_proj = getattr(attn, "q_b_proj", None)
    if q_b_proj is not None:
        q_b_out = local_num_heads * qk_head_dim
        _expect_module_attr(q_b_proj, "output_size", q_b_out, "q_b_proj")
        _expect_module_attr(q_b_proj, "output_size_per_partition", q_b_out, "q_b_proj")

    kv_b_proj = getattr(attn, "kv_b_proj", None)
    if kv_b_proj is not None:
        kv_b_out = local_num_heads * (qk_nope_head_dim + v_head_dim)
        _expect_module_attr(kv_b_proj, "output_size", kv_b_out, "kv_b_proj")
        _expect_module_attr(kv_b_proj, "output_size_per_partition", kv_b_out, "kv_b_proj")

    o_proj = getattr(attn, "o_proj", None)
    if o_proj is not None:
        o_proj_in = local_num_heads * v_head_dim
        _expect_module_attr(o_proj, "input_size", o_proj_in, "o_proj")
        _expect_module_attr(o_proj, "input_size_per_partition", o_proj_in, "o_proj")
        _expect_module_attr(o_proj, "output_size", int(attn.hidden_size), "o_proj")


def _import_sglang_forward_context():
    """Return SGLang's forward-context wrapper across runtime API versions."""
    try:
        from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
    except ModuleNotFoundError:
        from contextlib import nullcontext

        class ForwardContext:
            def __init__(self, *args, **kwargs):
                pass

        def forward_context(_context):
            return nullcontext()

    return ForwardContext, forward_context


def _sglang_forward_context(attn_backend):
    forward_context_type, forward_context = _import_sglang_forward_context()
    return forward_context(forward_context_type(attn_backend=attn_backend))


def load_model_runner(
    model_path: str,
    head_num: int,
    kv_cache_dtype: str,
    attention_backend: str,
    device: str = "cuda:0",
    tp_rank: int = 0,
    gemm_type: str = "bfloat16",
    target_tp_size: int = 1,
    enable_piecewise_cuda_graph: bool = False,
    piecewise_cuda_graph_tokens: list[int] | None = None,
    max_total_tokens: int | None = None,
):
    """Load SGLang ModelRunner with dummy weights.

    Args:
        model_path: HuggingFace model path (e.g. "deepseek-ai/DeepSeek-V3.2").
        head_num: Number of local attention heads to benchmark for one target TP rank.
        kv_cache_dtype: Perf-DB-compatible string ("bfloat16" or "fp8").
            Mapped to SGLang-native string via SGLANG_KV_DTYPE.
        attention_backend: Backend string for ServerArgs (e.g. "nsa", "fa3").
        gemm_type: Perf-DB-compatible string ("bfloat16" or "fp8_block").
            "fp8_block" launches sglang with quantization="fp8" so the
            attention module's linear projections and the paged MQA scoring
            kernel fire on real fp8 weights; "bfloat16" keeps quantization
            disabled and all projections run bf16 with dummy weights.

    Environment variables:
        SGLANG_TEST_NUM_LAYERS: Number of layers to load (default 2).
        SGLANG_LOAD_FORMAT: Weight format (default "dummy").
    """
    import random

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import suppress_other_loggers

    suppress_other_loggers()

    device_str = str(device)
    if ":" in device_str:
        gpu_id = int(device_str.split(":")[-1])
    else:
        gpu_id = tp_rank

    num_layers = int(os.environ.get("SGLANG_TEST_NUM_LAYERS", "2"))
    load_format = os.environ.get("SGLANG_LOAD_FORMAT", "dummy")

    # Map perf-DB dtype to SGLang-native dtype.
    # trtllm_mla accepts "bf16" but rejects "bfloat16" in its validation.
    sglang_kv_dtype = SGLANG_KV_DTYPE.get(kv_cache_dtype, kv_cache_dtype)
    if attention_backend == "trtllm_mla" and sglang_kv_dtype == "bfloat16":
        sglang_kv_dtype = "bf16"

    # Use AIC's local model configs to avoid HF downloads.  Also patches
    # model_type for sglang compatibility (glm_moe_dsa → deepseek_v3).
    local_model_path = _resolve_local_model_path(model_path)

    server_args = ServerArgs(
        model_path=local_model_path,
        dtype="auto",
        device="cuda",
        load_format=load_format,
        tp_size=1,
        trust_remote_code=True,
        disable_radix_cache=True,
        disable_cuda_graph=not enable_piecewise_cuda_graph,
        kv_cache_dtype=sglang_kv_dtype,
        max_total_tokens=max_total_tokens,
    )

    # Quantization control: bf16 (dummy weights) vs fp8 (real fp8 weight
    # paths so attention projections and indexer MQA scoring actually fire in
    # fp8).
    if gemm_type == "fp8_block":
        server_args.quantization = "fp8"
    elif gemm_type == "nvfp4":
        server_args.quantization = "modelopt_fp4"
    else:
        server_args.quantization = None

    if enable_piecewise_cuda_graph:
        server_args.disable_cuda_graph_padding = True
        server_args.cuda_graph_max_bs = int(os.environ.get("AIC_CUDA_GRAPH_MAX_BS", "256"))
        cuda_graph_bs = _parse_int_list(os.environ.get("AIC_CUDA_GRAPH_BS"))
        if cuda_graph_bs:
            server_args.cuda_graph_bs = cuda_graph_bs
        if piecewise_cuda_graph_tokens:
            server_args.piecewise_cuda_graph_tokens = piecewise_cuda_graph_tokens
        max_piecewise_tokens = os.environ.get("AIC_PIECEWISE_CUDA_GRAPH_MAX_TOKENS")
        if max_piecewise_tokens:
            server_args.piecewise_cuda_graph_max_tokens = int(max_piecewise_tokens)
        elif piecewise_cuda_graph_tokens:
            server_args.piecewise_cuda_graph_max_tokens = max(piecewise_cuda_graph_tokens)
        if hasattr(server_args, "enforce_piecewise_cuda_graph"):
            server_args.enforce_piecewise_cuda_graph = True

    if hasattr(server_args, "disable_piecewise_cuda_graph"):
        server_args.disable_piecewise_cuda_graph = not enable_piecewise_cuda_graph
    else:
        server_args.enable_piecewise_cuda_graph = enable_piecewise_cuda_graph

    server_args.attention_backend = attention_backend
    print(
        f"Using attention backend: {attention_backend}, kv_cache_dtype: {sglang_kv_dtype}, "
        f"gpu_id: {gpu_id}, chunked_prefill_size={server_args.chunked_prefill_size}, "
        f"max_prefill_tokens={server_args.max_prefill_tokens}, max_total_tokens={max_total_tokens}, "
        f"piecewise_cuda_graph={enable_piecewise_cuda_graph}"
    )

    if num_layers > 0 and load_format == "dummy":
        override_args = {
            "num_hidden_layers": num_layers,
            "num_attention_heads": head_num,
            "num_key_value_heads": head_num,
        }
        server_args.json_model_override_args = json.dumps(override_args)

    _set_envs_and_config(server_args)

    nccl_port = 29500 + random.randint(0, 10000) + gpu_id * 100

    model_config = ModelConfig.from_server_args(server_args)

    # Bug A fix: ensure hf_config.quantization_config carries weight_block_size
    # so sglang constructs Fp8Config with block_quant=True. Without this, the
    # Fp8LinearMethod post-load path at fp8.py:660 transposes kv_b_proj.weight
    # from (out=7168, in=512) to (in=512, out=7168), and then
    # deepseek_weight_loader.py:555 unflatten(dim0=512, 448) fails with
    # `448 ∤ 512`. Our _resolve_local_model_path writes quantization_config into
    # the tmp config.json, but after the model_type rewrite to deepseek_v3 the
    # attribute may not survive onto hf_config — re-inject it here.
    if gemm_type == "fp8_block":
        _ensure_fp8_block_quant_config(model_config.hf_config)

    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=gpu_id,
        tp_rank=gpu_id,
        tp_size=server_args.tp_size,
        pp_rank=0,
        pp_size=1,
        moe_ep_rank=0,
        moe_ep_size=1,
        nccl_port=nccl_port,
        server_args=server_args,
    )

    _patch_nsa_rope_contiguity(model_runner)
    _validate_dsa_tp_module_shapes(model_runner, head_num, target_tp_size)

    return model_runner


# ═══════════════════════════════════════════════════════════════════════
# Core Benchmarking
# ═══════════════════════════════════════════════════════════════════════


def run_attention_torch(
    model_runner,
    test_cases,
    head_num: int,
    test_layer: int,
    num_warmup: int,
    num_iterations: int,
    device: str,
    output_path: str | None,
    *,
    attn_type: str,
    model_path: str,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    target_tp_size: int = 1,
    use_module_cuda_graph: bool = False,
):
    """Run attention benchmark for both prefill and decode phases.

    Args:
        test_cases: List of (batch_size, seq_length, is_prefill) tuples.
        kv_cache_dtype: Perf-DB-compatible string for logging.
        compute_dtype: Perf-DB-compatible string for logging.
        gemm_type: Perf-DB-compatible string for logging.
    """
    attention_module = model_runner.model.model.layers[test_layer].self_attn
    architecture = _module_model_architecture(model_path)
    backend_name = model_runner.server_args.attention_backend
    version = get_version("sglang")
    device_name = torch.cuda.get_device_name(device)

    # Wideep MLA backward-compatibility: the old collect_wideep_attn.py logged
    # mla_dtype="fp8_block", kv_cache_dtype="fp8" for all runs, used the raw
    # backend name as kernel_source, and different op_name / filename patterns.
    # perf_database loaders (load_wideep_*_mla_data) expect these conventions.
    is_wideep_mla = attn_type == "mla"

    if is_wideep_mla:
        log_mla_dtype = "fp8_block"
        log_kv_dtype = "fp8"
        log_gemm_type = "fp8_block"
    else:
        # DSA: log dtype strings that match the common.*QuantMode enum member
        # names expected by perf_database loaders (e.g. "bfloat16", "fp8").
        log_mla_dtype = compute_dtype
        log_kv_dtype = kv_cache_dtype
        log_gemm_type = gemm_type

    # QKV latent dimensions for AttentionInputs
    q_lora_rank = getattr(attention_module, "q_lora_rank", 1536) or 1536
    kv_lora_rank = getattr(attention_module, "kv_lora_rank", 512)
    qk_rope_head_dim = getattr(attention_module, "qk_rope_head_dim", 64)
    qkv_latent_dim = q_lora_rank + kv_lora_rank + qk_rope_head_dim

    def dummy_qkv_latent_func(h, fb):
        return torch.randn(h.shape[0], qkv_latent_dim, dtype=h.dtype, device=h.device)

    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    logged_count = 0
    for test_case in test_cases:
        if len(test_case) == 4:
            batch_size, seq_length, is_prefill, prefix_len = test_case
        else:
            batch_size, seq_length, is_prefill = test_case
            prefix_len = 0

        if is_prefill:
            logged_count += int(
                _run_prefill(
                    model_runner=model_runner,
                    attention_module=attention_module,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    head_num=head_num,
                    num_warmup=num_warmup,
                    num_iterations=num_iterations,
                    device=device,
                    output_path=output_path,
                    dummy_qkv_latent_func=dummy_qkv_latent_func,
                    attn_type=attn_type,
                    model_path=model_path,
                    architecture=architecture,
                    backend_name=backend_name,
                    version=version,
                    device_name=device_name,
                    log_mla_dtype=log_mla_dtype,
                    log_kv_dtype=log_kv_dtype,
                    log_gemm_type=log_gemm_type,
                    target_tp_size=target_tp_size,
                    prefix_len=prefix_len,
                    use_module_cuda_graph=use_module_cuda_graph,
                )
            )
        else:
            logged_count += int(
                _run_decode(
                    model_runner=model_runner,
                    attention_module=attention_module,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    head_num=head_num,
                    num_warmup=num_warmup,
                    num_iterations=num_iterations,
                    device=device,
                    output_path=output_path,
                    dummy_qkv_latent_func=dummy_qkv_latent_func,
                    attn_type=attn_type,
                    model_path=model_path,
                    architecture=architecture,
                    backend_name=backend_name,
                    version=version,
                    device_name=device_name,
                    log_mla_dtype=log_mla_dtype,
                    log_kv_dtype=log_kv_dtype,
                    log_gemm_type=log_gemm_type,
                    target_tp_size=target_tp_size,
                    use_module_cuda_graph=use_module_cuda_graph,
                )
            )
    return logged_count


def _run_prefill(
    model_runner,
    attention_module,
    batch_size: int,
    seq_length: int,
    head_num: int,
    num_warmup: int,
    num_iterations: int,
    device: str,
    output_path: str | None,
    dummy_qkv_latent_func,
    attn_type: str,
    model_path: str,
    architecture: str,
    backend_name: str,
    version: str,
    device_name: str,
    log_mla_dtype: str,
    log_kv_dtype: str,
    log_gemm_type: str,
    target_tp_size: int,
    prefix_len: int = 0,
    use_module_cuda_graph: bool = False,
):
    """Run prefill (context) benchmark for a single (batch_size, seq_length) point."""
    is_wideep_mla = attn_type == "mla"
    from sglang.srt.layers.communicator import AttentionInputs, get_attn_tp_context
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.chunk_cache import ChunkCache
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.utils import BumpAllocator

    forward_context_type, forward_context = _import_sglang_forward_context()

    print(f"\nPrefill: batch_size={batch_size}, seq_length={seq_length}, prefix_len={prefix_len}")

    try:
        model_runner.req_to_token_pool.clear()
        model_runner.token_to_kv_pool_allocator.clear()

        prefix_indices = _alloc_prefix_indices(model_runner, batch_size, prefix_len)
        full_length = prefix_len + seq_length

        reqs = []
        for i in range(batch_size):
            req = Req(
                rid=str(i),
                origin_input_text="",
                origin_input_ids=list(torch.randint(0, 10000, (full_length,)).tolist()),
                sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
            )
            req.prefix_indices = prefix_indices[i]
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = seq_length if prefix_len else len(req.fill_ids)
            req.logprob_start_len = 0
            reqs.append(req)

        cache_params = CacheInitParams(
            disable=True,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
            page_size=model_runner.token_to_kv_pool_allocator.page_size,
        )
        tree_cache = ChunkCache(cache_params)

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        with _temporarily_chunked_alloc_extend(model_runner, batch_size * seq_length):
            batch.prepare_for_extend()
        if hasattr(batch, "get_model_worker_batch"):
            batch_for_forward = batch.get_model_worker_batch()
        else:
            batch_for_forward = batch
        forward_batch = ForwardBatch.init_new(batch_for_forward, model_runner)
        model_runner.attn_backend.init_forward_metadata(forward_batch)

        hidden_states = torch.randn(
            batch_size * seq_length,
            model_runner.model.config.hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        positions = (
            torch.arange(prefix_len, prefix_len + seq_length, device="cuda")
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
            .flatten()
        )
        zero_allocator = BumpAllocator(buffer_size=256, dtype=torch.float32, device="cuda")

        attn_inputs = AttentionInputs(hidden_states, forward_batch, dummy_qkv_latent_func)
        get_attn_tp_context().set_attn_inputs(attn_inputs)

        use_module_piecewise_replay = _env_flag("AIC_ENABLE_MODULE_PIECEWISE_REPLAY") or (
            attn_type == "dsa" and _is_glm5_dsa_model(model_path)
        )
        use_full_model_piecewise_replay = _env_flag("AIC_ENABLE_FULL_MODEL_PIECEWISE_REPLAY")
        use_module_cuda_graph = use_module_cuda_graph or _env_flag("AIC_ENABLE_MODULE_CUDA_GRAPH")
        use_module_piecewise_context = _env_flag("AIC_ENABLE_MODULE_PIECEWISE_CONTEXT")
        token_count = batch_size * seq_length
        max_piecewise_tokens = int(getattr(model_runner.server_args, "piecewise_cuda_graph_max_tokens", 0) or 0) or (
            max(int(x) for x in model_runner.server_args.piecewise_cuda_graph_tokens)
            if model_runner.server_args.piecewise_cuda_graph_tokens
            else 0
        )
        if use_module_piecewise_replay and max_piecewise_tokens and token_count > max_piecewise_tokens:
            print(
                "  Module piecewise CUDA graph replay disabled: "
                f"num_tokens={token_count} exceeds max={max_piecewise_tokens}; "
                "using piecewise context without graph replay"
            )
            use_module_piecewise_replay = False
            use_module_cuda_graph = False
            use_module_piecewise_context = True
        if use_module_piecewise_replay:
            print("  Module piecewise CUDA graph replay enabled")
            use_full_model_piecewise_replay = False
            use_module_cuda_graph = False
            use_module_piecewise_context = False
        if use_full_model_piecewise_replay:
            print("  Full model piecewise CUDA graph replay enabled")
            use_module_cuda_graph = False
            use_module_piecewise_context = False
        if use_module_piecewise_context:
            from sglang.srt.compilation.piecewise_context_manager import (
                enable_piecewise_cuda_graph,
            )
            from sglang.srt.compilation.piecewise_context_manager import (
                set_forward_context as set_piecewise_forward_context,
            )

            if use_module_cuda_graph:
                print("  Module CUDA graph disabled; using piecewise CUDA graph context")
                use_module_cuda_graph = False

        if use_module_cuda_graph:
            _patch_nsa_indexer_compile_for_module_cuda_graph(attention_module)

        if use_module_piecewise_replay:
            from sglang.srt.layers.logits_processor import LogitsProcessorOutput

            replay_token_count = getattr(model_runner, "_aic_module_piecewise_replay_token_count", None)
            if (
                getattr(model_runner, "_aic_module_piecewise_replay_initialized", False)
                and replay_token_count != token_count
            ):
                # The module-only wrapper compiles against the current token
                # shape. Reusing it for a different token_count can trigger
                # runtime recompilation outside the PCG capture stream, which
                # SGLang rejects with "PCG capture stream is not set".
                model_runner.piecewise_cuda_graph_runner = None
                model_runner._aic_module_piecewise_replay_initialized = False
                model_runner._aic_module_piecewise_replay_token_count = None

            if getattr(model_runner, "_aic_module_piecewise_replay_initialized", False):
                print("  Module piecewise runner=True")
            else:
                original_piecewise_tokens = getattr(model_runner.server_args, "piecewise_cuda_graph_tokens", None)
                original_piecewise_max_tokens = getattr(
                    model_runner.server_args, "piecewise_cuda_graph_max_tokens", None
                )
                model_runner.server_args.piecewise_cuda_graph_tokens = [token_count]
                model_runner.server_args.piecewise_cuda_graph_max_tokens = token_count
                module_hidden_states = torch.randn(
                    token_count,
                    model_runner.model.config.hidden_size,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                module_logits = torch.empty(
                    module_hidden_states.shape[0],
                    1,
                    dtype=torch.float32,
                    device="cuda",
                )

                class _AttentionOnlyLayer(torch.nn.Module):
                    def __init__(self, self_attn):
                        super().__init__()
                        self.self_attn = self_attn

                class _AttentionOnlyLanguageModel(torch.nn.Module):
                    def __init__(self, self_attn, hidden_buffer, logits_buffer):
                        super().__init__()
                        self.layers = torch.nn.ModuleList([_AttentionOnlyLayer(self_attn)])
                        self.hidden_states = hidden_buffer
                        self.logits = logits_buffer

                    def forward(
                        self,
                        input_ids: torch.Tensor,
                        positions: torch.Tensor,
                        forward_batch,
                        **kwargs,
                    ):
                        hidden = self.hidden_states[: input_ids.shape[0]]
                        attn_inputs = AttentionInputs(hidden, forward_batch, dummy_qkv_latent_func)
                        get_attn_tp_context().set_attn_inputs(attn_inputs)
                        out = self.layers[0].self_attn(
                            positions=positions,
                            hidden_states=hidden,
                            forward_batch=forward_batch,
                            zero_allocator=zero_allocator,
                        )
                        if isinstance(out, tuple):
                            out = out[0]
                        return LogitsProcessorOutput(
                            next_token_logits=self.logits[: input_ids.shape[0]],
                            hidden_states=out,
                        )

                class _AttentionOnlyModel(torch.nn.Module):
                    def __init__(self, self_attn):
                        super().__init__()
                        self.model = _AttentionOnlyLanguageModel(self_attn, module_hidden_states, module_logits)

                    def forward(
                        self,
                        input_ids: torch.Tensor,
                        positions: torch.Tensor,
                        forward_batch,
                        **kwargs,
                    ):
                        return self.model.forward(input_ids, positions, forward_batch, **kwargs)

                original_model = model_runner.model
                original_num_layers = model_runner.model_config.num_hidden_layers
                import sglang.srt.model_executor.piecewise_cuda_graph_runner as pcg_runner_mod
                from sglang.srt.compilation.piecewise_context_manager import (
                    set_forward_context as set_piecewise_forward_context,
                )
                from sglang.srt.layers.dp_attention import (
                    DpPaddingMode,
                    set_dp_buffer_len,
                    set_is_extend_in_batch,
                )

                original_install_torch_compiled = pcg_runner_mod.install_torch_compiled
                original_warmup_compile = pcg_runner_mod.PiecewiseCudaGraphRunner.warmup_compile
                original_capture_one_batch_size = pcg_runner_mod.PiecewiseCudaGraphRunner.capture_one_batch_size

                def _install_torch_compiled_with_module_dims(module, *args, **kwargs):
                    if isinstance(module, _AttentionOnlyLanguageModel) and kwargs.get("dynamic_arg_dims") is None:
                        kwargs["dynamic_arg_dims"] = {
                            "input_ids": 0,
                            "positions": 0,
                        }
                    return original_install_torch_compiled(module, *args, **kwargs)

                def _run_module_piecewise_target(runner, target_forward_batch):
                    static_forward_batch = runner.replay_prepare(target_forward_batch)
                    if static_forward_batch.dp_padding_mode is None:
                        static_forward_batch.dp_padding_mode = DpPaddingMode.get_default_mode_in_cuda_graph()
                    runner.model_runner.attn_backend.init_forward_metadata(target_forward_batch)
                    static_forward_batch.dp_local_start_pos = None
                    static_forward_batch.dp_local_num_tokens = None
                    set_dp_buffer_len(
                        None,
                        len(static_forward_batch.input_ids),
                        static_forward_batch.dp_padding_mode.is_max_len(),
                    )
                    set_is_extend_in_batch(False)
                    with (
                        forward_context(forward_context_type(attn_backend=runner.model_runner.attn_backend)),
                        _piecewise_forward_context(
                            set_piecewise_forward_context,
                            static_forward_batch,
                            runner.attention_layers,
                            runner.quant_config,
                            runner.moe_layers,
                            runner.moe_fusions,
                            dsa_indexers=getattr(runner, "dsa_indexers", None),
                        ),
                    ):
                        return runner.model_runner.model.forward(
                            static_forward_batch.input_ids,
                            static_forward_batch.positions,
                            static_forward_batch,
                        )

                def _warmup_compile_with_module_batch(runner, num_tokens: int):
                    target_forward_batch = getattr(
                        runner.model_runner,
                        "_aic_module_piecewise_forward_batch",
                        None,
                    )
                    if target_forward_batch is not None and num_tokens in runner.capture_num_tokens:
                        return _run_module_piecewise_target(runner, target_forward_batch)
                    return original_warmup_compile(runner, num_tokens)

                def _capture_one_batch_size_with_module_batch(runner, num_tokens: int):
                    target_forward_batch = getattr(
                        runner.model_runner,
                        "_aic_module_piecewise_forward_batch",
                        None,
                    )
                    if target_forward_batch is not None and num_tokens in runner.capture_num_tokens:
                        for _ in range(2):
                            runner.device_module.synchronize()
                            runner.model_runner.tp_group.barrier()
                            _run_module_piecewise_target(runner, target_forward_batch)
                        return
                    return original_capture_one_batch_size(runner, num_tokens)

                module_model = _AttentionOnlyModel(attention_module).to("cuda")
                module_model.config = getattr(original_model, "config", None)
                module_model.quant_config = getattr(original_model, "quant_config", None)
                module_model.model.config = getattr(
                    getattr(original_model, "model", original_model), "config", module_model.config
                )
                model_runner.model = module_model
                model_runner.model_config.num_hidden_layers = 1
                model_runner._aic_module_piecewise_forward_batch = forward_batch
                try:
                    pcg_runner_mod.install_torch_compiled = _install_torch_compiled_with_module_dims
                    pcg_runner_mod.PiecewiseCudaGraphRunner.warmup_compile = _warmup_compile_with_module_batch
                    pcg_runner_mod.PiecewiseCudaGraphRunner.capture_one_batch_size = (
                        _capture_one_batch_size_with_module_batch
                    )
                    with torch.no_grad():
                        model_runner.init_piecewise_cuda_graphs()
                finally:
                    pcg_runner_mod.install_torch_compiled = original_install_torch_compiled
                    pcg_runner_mod.PiecewiseCudaGraphRunner.warmup_compile = original_warmup_compile
                    pcg_runner_mod.PiecewiseCudaGraphRunner.capture_one_batch_size = original_capture_one_batch_size
                    model_runner.model_config.num_hidden_layers = original_num_layers
                    if hasattr(model_runner, "_aic_module_piecewise_forward_batch"):
                        delattr(model_runner, "_aic_module_piecewise_forward_batch")
                    model_runner.server_args.piecewise_cuda_graph_tokens = original_piecewise_tokens
                    model_runner.server_args.piecewise_cuda_graph_max_tokens = original_piecewise_max_tokens
                model_runner._aic_module_piecewise_replay_initialized = True
                model_runner._aic_module_piecewise_replay_token_count = token_count
                print(f"  Module piecewise runner={model_runner.piecewise_cuda_graph_runner is not None}")

        def call_attention_module():
            with forward_context(forward_context_type(attn_backend=model_runner.attn_backend)):
                if use_module_piecewise_context:
                    with (
                        enable_piecewise_cuda_graph(),
                        _piecewise_forward_context(
                            set_piecewise_forward_context,
                            forward_batch,
                            getattr(model_runner, "attention_layers", []),
                            getattr(model_runner.model, "quant_config", None),
                            getattr(model_runner, "moe_layers", []),
                            getattr(model_runner, "moe_fusions", []),
                            dsa_indexers=getattr(model_runner, "dsa_indexers", None),
                        ),
                    ):
                        # Keep DSA dispatch consistent with SGLang's
                        # piecewise path even when we do not replay a
                        # captured CUDA graph for this token count.
                        model_runner.attn_backend.init_forward_metadata(forward_batch)
                        attention_module(
                            positions=positions,
                            hidden_states=hidden_states,
                            forward_batch=forward_batch,
                            zero_allocator=zero_allocator,
                        )
                else:
                    attention_module(
                        positions=positions,
                        hidden_states=hidden_states,
                        forward_batch=forward_batch,
                        zero_allocator=zero_allocator,
                    )

        last_can_run_graph = None

        def call_full_model_piecewise_replay():
            nonlocal last_can_run_graph
            output = model_runner.forward(forward_batch)
            last_can_run_graph = getattr(output, "can_run_graph", None)

        call_target = (
            call_full_model_piecewise_replay
            if (use_full_model_piecewise_replay or use_module_piecewise_replay)
            else call_attention_module
        )

        # Warmup
        for _ in range(num_warmup):
            with torch.no_grad():
                call_target()
            torch.cuda.synchronize()
        if use_full_model_piecewise_replay or use_module_piecewise_replay:
            print(f"  Piecewise can_run_graph={last_can_run_graph}")

        module_cuda_graph = None
        if use_module_cuda_graph:
            try:
                torch.cuda.synchronize()
                module_cuda_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(module_cuda_graph):
                    call_target()
                torch.cuda.synchronize()
                print("  Module CUDA graph capture enabled")
            except Exception:
                print("  Module CUDA graph capture failed")
                raise

        # Timed runs — skip first 2 for stability
        cuda_times = []
        for i in range(num_iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.no_grad():
                if module_cuda_graph is not None:
                    module_cuda_graph.replay()
                else:
                    call_target()
            end_event.record()
            torch.cuda.synchronize()
            if i > 1:
                cuda_times.append(start_event.elapsed_time(end_event))

        avg_time_ms = np.mean(cuda_times)

        # Log perf — wideep MLA uses old filename/op_name/kernel_source conventions
        try:
            if is_wideep_mla:
                perf_fname = "wideep_context_mla_perf.txt"
                op_name = "mla_context"
                kernel_source = backend_name
            else:
                perf_fname = f"{attn_type}_context_module_perf.txt"
                op_name = f"{attn_type}_context_module"
                kernel_source = f"{attn_type}_{backend_name}"
            perf_filename = _resolve_perf_path(output_path, perf_fname)
            log_perf(
                item_list=[
                    {
                        "model": model_path,
                        "architecture": architecture,
                        "mla_dtype": log_mla_dtype,
                        "kv_cache_dtype": log_kv_dtype,
                        "gemm_type": log_gemm_type,
                        "num_heads": head_num,
                        "batch_size": batch_size,
                        "isl": seq_length,
                        "tp_size": target_tp_size,
                        "step": prefix_len,
                        "latency": f"{avg_time_ms:.4f}",
                    }
                ],
                framework="SGLang",
                version=version,
                device_name=device_name,
                op_name=op_name,
                kernel_source=kernel_source,
                perf_filename=perf_filename,
            )
        except Exception as e:
            print(f"  Warning: failed to log prefill metrics: {e}")
            return False

        print(
            f"  Prefill: {avg_time_ms:.3f} ms "
            f"(min: {np.min(cuda_times):.3f}, max: {np.max(cuda_times):.3f}, "
            f"std: {np.std(cuda_times):.3f})"
        )
        return True

    except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
        print(f"  OOM: b={batch_size}, s={seq_length} — skipping")
        torch.cuda.empty_cache()
        return False
    except Exception as e:
        traceback.print_exc()
        error_str = str(e).lower()
        if "out of memory" in error_str:
            print(f"  OOM: b={batch_size}, s={seq_length} — skipping")
            torch.cuda.empty_cache()
            return False
        if "cuda" in error_str and "illegal" in error_str:
            print("  CUDA illegal access detected — stopping this subprocess to preserve prior rows")
            raise CudaIllegalAccessError(f"CUDA illegal access at b={batch_size}, s={seq_length}") from e
        print("  Skipping this configuration...")
        return False
    finally:
        for cleanup_name, cleanup_fn in (
            ("req_to_token_pool.clear", model_runner.req_to_token_pool.clear),
            ("token_to_kv_pool_allocator.clear", model_runner.token_to_kv_pool_allocator.clear),
            ("torch.cuda.empty_cache", torch.cuda.empty_cache),
        ):
            try:
                cleanup_fn()
            except Exception as cleanup_exc:
                print(
                    f"  Warning: cleanup step {cleanup_name} failed after benchmark: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )


def _run_decode(
    model_runner,
    attention_module,
    batch_size: int,
    seq_length: int,
    head_num: int,
    num_warmup: int,
    num_iterations: int,
    device: str,
    output_path: str | None,
    dummy_qkv_latent_func,
    attn_type: str,
    model_path: str,
    architecture: str,
    backend_name: str,
    version: str,
    device_name: str,
    log_mla_dtype: str,
    log_kv_dtype: str,
    log_gemm_type: str,
    target_tp_size: int,
    use_module_cuda_graph: bool = False,
):
    """Run decode (generation) benchmark for a single (batch_size, kv_cache_length) point."""
    is_wideep_mla = attn_type == "mla"
    from sglang.srt.layers.communicator import AttentionInputs, get_attn_tp_context
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.chunk_cache import ChunkCache
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.utils import BumpAllocator

    forward_context_type, forward_context = _import_sglang_forward_context()

    print(f"\nDecode: batch_size={batch_size}, kv_cache_length={seq_length}")

    try:
        model_runner.req_to_token_pool.clear()
        model_runner.token_to_kv_pool_allocator.clear()

        reqs = []
        for i in range(batch_size):
            req = Req(
                rid=str(i),
                origin_input_text="",
                origin_input_ids=list(torch.randint(0, 10000, (seq_length,)).tolist()),
                sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
            )
            req.prefix_indices = torch.empty((0,), dtype=torch.int64)
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(req.fill_ids)
            req.logprob_start_len = 0
            req.cached_tokens = 0
            req.already_computed = 0
            reqs.append(req)

        cache_params = CacheInitParams(
            disable=True,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
            page_size=model_runner.token_to_kv_pool_allocator.page_size,
        )
        tree_cache = ChunkCache(cache_params)

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        # Allocate KV cache slots, then switch to decode
        batch.prepare_for_extend()
        batch.output_ids = torch.randint(0, 10000, (batch_size,), dtype=torch.int64, device="cuda")
        batch.prepare_for_decode()
        if hasattr(batch, "get_model_worker_batch"):
            batch_for_forward_decode = batch.get_model_worker_batch()
        else:
            batch_for_forward_decode = batch
        forward_batch_decode = ForwardBatch.init_new(batch_for_forward_decode, model_runner)
        model_runner.attn_backend.init_forward_metadata(forward_batch_decode)

        decode_hidden = torch.randn(
            batch_size,
            model_runner.model.config.hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        decode_positions = torch.full((batch_size,), seq_length, device="cuda")
        zero_allocator = BumpAllocator(buffer_size=2048, dtype=torch.float32, device="cuda")

        attn_inputs_decode = AttentionInputs(decode_hidden, forward_batch_decode, dummy_qkv_latent_func)
        get_attn_tp_context().set_attn_inputs(attn_inputs_decode)
        if use_module_cuda_graph or _env_flag("AIC_ENABLE_MODULE_CUDA_GRAPH"):
            _patch_nsa_indexer_compile_for_module_cuda_graph(attention_module)

        def kernel_func():
            with forward_context(forward_context_type(attn_backend=model_runner.attn_backend)):
                attention_module(
                    positions=decode_positions,
                    hidden_states=decode_hidden,
                    forward_batch=forward_batch_decode,
                    zero_allocator=zero_allocator,
                )

        # Pre-warm JIT / autotuning before CUDA graph capture.
        # DSA decode on Blackwell calls DeepGEMM fp8_paged_mqa_logits and
        # flashinfer trtllm_batch_decode_with_kv_cache_mla, both of which
        # JIT or autotune on first call to a new (heads, bs, kv_len) shape.
        # If that work spills into the graph-capture window inside
        # benchmark_with_power, it emits cudaMemcpy-like ops that aren't
        # permitted during capture and the whole sweep silently skips
        # (Issue #3 — reproduces only at reduced heads ∈ {8, 16, 32} where
        # the warmup-time JIT cache from heads=64 doesn't satisfy the new
        # template instantiation). A few extra eager kernel_func calls
        # with explicit syncs in between flush that path before capture.
        if torch.cuda.is_available():
            for _ in range(5):
                kernel_func()
                torch.cuda.synchronize()

        use_benchmark_cuda_graph = not (
            attn_type == "dsa" and not _generation_cuda_graph_enabled_for_tokens(batch_size)
        )
        print(f"  Decode module CUDA graph: {use_benchmark_cuda_graph} (tokens={batch_size})")

        with benchmark_with_power(
            device=device,
            kernel_func=kernel_func,
            num_warmups=num_warmup,
            num_runs=num_iterations,
            repeat_n=1,
            use_cuda_graph=use_benchmark_cuda_graph,
        ) as results:
            pass

        avg_time_ms = results["latency_ms"]
        power_stats = results["power_stats"]

        # Log perf — wideep MLA uses isl=seq_len, step=0 (old convention).
        # DSA uses isl=1, step=seq_len, where seq_len is past-KV length.
        # The wideep generation loader computes s = isl + step, so both
        # conventions yield the same effective key when step=0 → s=seq_len.
        try:
            if is_wideep_mla:
                perf_fname = "wideep_generation_mla_perf.txt"
                op_name = "mla_generation"
                kernel_source = backend_name
                log_isl = seq_length
                log_step = 0
            else:
                perf_fname = f"{attn_type}_generation_module_perf.txt"
                op_name = f"{attn_type}_generation_module"
                kernel_source = f"{attn_type}_{backend_name}"
                log_isl = 1
                log_step = seq_length
            perf_filename = _resolve_perf_path(output_path, perf_fname)
            log_perf(
                item_list=[
                    {
                        "model": model_path,
                        "architecture": architecture,
                        "mla_dtype": log_mla_dtype,
                        "kv_cache_dtype": log_kv_dtype,
                        "gemm_type": log_gemm_type,
                        "num_heads": head_num,
                        "batch_size": batch_size,
                        "isl": log_isl,
                        "tp_size": target_tp_size,
                        "step": log_step,
                        "latency": f"{avg_time_ms:.4f}",
                    }
                ],
                framework="SGLang",
                version=version,
                device_name=device_name,
                op_name=op_name,
                kernel_source=kernel_source,
                perf_filename=perf_filename,
                power_stats=power_stats,
            )
        except Exception as e:
            print(f"  Warning: failed to log decode metrics: {e}")
            return False

        print(f"  Decode: {avg_time_ms:.3f} ms")
        return True

    except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
        print(f"  OOM: b={batch_size}, s={seq_length} — skipping")
        torch.cuda.empty_cache()
        return False
    except Exception as e:
        traceback.print_exc()
        error_str = str(e).lower()
        if "out of memory" in error_str:
            print(f"  OOM: b={batch_size}, s={seq_length} — skipping")
            torch.cuda.empty_cache()
            return False
        if "cuda" in error_str and "illegal" in error_str:
            print("  CUDA illegal access detected — stopping this subprocess to preserve prior rows")
            raise CudaIllegalAccessError(f"CUDA illegal access at b={batch_size}, s={seq_length}") from e
        print("  Skipping this configuration...")
        return False
    finally:
        for cleanup_name, cleanup_fn in (
            ("req_to_token_pool.clear", model_runner.req_to_token_pool.clear),
            ("token_to_kv_pool_allocator.clear", model_runner.token_to_kv_pool_allocator.clear),
            ("torch.cuda.empty_cache", torch.cuda.empty_cache),
        ):
            try:
                cleanup_fn()
            except Exception as cleanup_exc:
                print(
                    f"  Warning: cleanup step {cleanup_name} failed after benchmark: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )


def _resolve_perf_path(output_path: str | None, filename: str) -> str:
    """Resolve the full path for a perf output file."""
    if output_path is not None:
        return os.path.join(output_path, filename)
    collector_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(collector_dir, filename)


# ═══════════════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════════════


def run_mla_module(
    attn_type: str,
    head_num: int,
    model_path: str,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    is_prefill: bool,
    gpu_id: int,
    output_path: str | None = None,
    attention_backend: str | None = None,
    batch_size_filter: int | None = None,
    target_tp_size: int = 1,
):
    """Run MLA/DSA module benchmark — called inside a subprocess.

    Sets up the model runner for the given configuration and runs all
    (batch_size, seq_length) combos for the specified phase.
    """
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    if attention_backend is None:
        attention_backend = _get_backends(attn_type)

    if is_prefill:
        all_cases = get_context_test_cases(attn_type)
        phase_name = "Context"
    else:
        all_cases = get_generation_test_cases(attn_type)
        phase_name = "Generation"

    # Filter to matching precision combo.
    # Test case format: [seq_len, batch_size, num_heads, kv_dtype, compute_dtype, gemm_type]
    # run_attention_torch expects: (batch_size, seq_length, is_prefill, prefix_len)
    base_cases = [
        (tc[1], tc[0], is_prefill)
        for tc in all_cases
        if tc[3] == kv_cache_dtype and tc[4] == compute_dtype and tc[5] == gemm_type and tc[2] == head_num
    ]
    if batch_size_filter:
        base_cases = [(bs, seq_len, ip) for bs, seq_len, ip in base_cases if bs == batch_size_filter]
    if is_prefill and attn_type == "dsa":
        prefix_lens = get_mla_module_sweep_spec("sglang").context_prefix_lengths
        # Run prefix as the outer loop so a late long-prefix illegal access does
        # not discard all larger-ISL rows for the current batch-size subprocess.
        cases = [
            (bs, seq_len, ip, prefix_len)
            for prefix_len in prefix_lens
            for bs, seq_len, ip in base_cases
            if _dsa_context_prefix_shape_is_valid(bs, seq_len, prefix_len)
        ]
    else:
        cases = [(bs, seq_len, ip, 0) for bs, seq_len, ip in base_cases]
    cases = _filter_cases_from_env(cases, is_prefill=is_prefill, attn_type=attn_type)

    # Known-crash skip: B200 DSv3.2 DSA generation at reduced heads ≤ 32 and
    # kv_cache_length ≥ 256 produces an async CUDA illegal memory access
    # inside the flashinfer trtllm_batch_decode_with_kv_cache_mla kernel —
    # the subprocess SIGABRTs and every remaining (bs, kv_len) in this task
    # is lost. Root cause is collector-setup divergence from production
    # (dummy weights + cleared KV pool + json_model_override_args heads
    # override) rather than a real kernel bug — InferenceX SemiAnalysis
    # benchmarks show GLM-5 tp=4 on B200 + fp8-KV works in production.
    # Skipping these points lets the subprocess complete the sweep and
    # populate rows for bs in {2..1024} x kv_len in {1..128} at reduced heads
    # instead of losing them behind the first SIGABRT at (bs=1, kv_len=256).
    if (
        not is_prefill
        and attn_type == "dsa"
        and model_path == "deepseek-ai/DeepSeek-V3.2"
        and head_num <= 32
        and get_sm_version() == 100
        and not _env_flag("AIC_FORCE_DSA_REDUCED_HEAD_GENERATION")
    ):
        before = len(cases)
        cases = [(bs, seq_len, ip, prefix_len) for (bs, seq_len, ip, prefix_len) in cases if seq_len < 256]
        skipped = before - len(cases)
        if skipped:
            print(
                f"[SKIP] B200 DSv3.2 DSA gen reduced-heads: dropping {skipped} "
                f"(bs, kv_len) cases with kv_len>=256 to avoid SIGABRT "
                f"(heads={head_num}, kv={kv_cache_dtype}, gemm={gemm_type})"
            )

    print(f"\n{'=' * 60}")
    print(
        f"{attn_type.upper()} Module {phase_name}: model={model_path}, backend={attention_backend}, "
        f"head_num={head_num}, target_tp={target_tp_size}, kv={kv_cache_dtype}, "
        f"compute={compute_dtype}, gemm={gemm_type}, GPU={gpu_id}"
    )
    print(f"Test cases: {len(cases)}")
    print(f"{'=' * 60}")

    use_module_cuda_graph = _enable_glm5_dsa_piecewise_graph(attn_type, model_path)
    enable_runner_piecewise_cuda_graph = _enable_glm5_dsa_piecewise_graph(attn_type, model_path)

    if is_prefill and attn_type == "dsa":
        before = len(cases)
        cases = [
            (bs, seq_len, ip, prefix_len)
            for (bs, seq_len, ip, prefix_len) in cases
            if dsa_indexer_total_kv_tokens_supported(bs, seq_len, prefix_len, is_prefill=True)
        ]
        skipped = before - len(cases)
        if skipped:
            print(f"[DSA] Dropped {skipped} context cases beyond DSA indexer total KV token limit")

    max_total_tokens = None
    if cases:
        max_total_tokens = max(
            required_kv_tokens(bs, seq_len, prefix_len, is_prefill=ip) for (bs, seq_len, ip, prefix_len) in cases
        )
        if max_total_tokens > 0:
            max_total_tokens += max(1024, max_total_tokens // 20)

    piecewise_cuda_graph_tokens = (
        _piecewise_cuda_graph_tokens_for_cases(cases, is_prefill) if enable_runner_piecewise_cuda_graph else None
    )
    if use_module_cuda_graph:
        print("[DSA] Module CUDA graph capture enabled for GLM-5 DSA piecewise parity")
    if enable_runner_piecewise_cuda_graph:
        token_msg = piecewise_cuda_graph_tokens if piecewise_cuda_graph_tokens is not None else "SGLang default"
        print(f"[DSA] SGLang piecewise CUDA graph enabled with tokens={token_msg}")
    if max_total_tokens is not None:
        print(f"[DSA] SGLang max_total_tokens capped for collector at {max_total_tokens}")

    cleanup_distributed()
    torch.cuda.empty_cache()

    try:
        model_runner = load_model_runner(
            model_path=model_path,
            head_num=head_num,
            kv_cache_dtype=kv_cache_dtype,
            attention_backend=attention_backend,
            device=device,
            gemm_type=gemm_type,
            target_tp_size=target_tp_size,
            enable_piecewise_cuda_graph=enable_runner_piecewise_cuda_graph,
            piecewise_cuda_graph_tokens=piecewise_cuda_graph_tokens,
            max_total_tokens=max_total_tokens,
        )

        if is_prefill and attn_type == "dsa":
            chunk_size = _runtime_chunk_size(model_runner)
            before = len(cases)
            cases = [
                (bs, seq_len, ip, prefix_len)
                for (bs, seq_len, ip, prefix_len) in cases
                if required_prefill_extend_tokens(bs, seq_len) <= chunk_size
            ]
            skipped = before - len(cases)
            print(
                f"[DSA] SGLang runtime chunked_prefill_size={chunk_size}; "
                f"dropped {skipped} context cases with fresh_tokens > chunked_prefill_size"
            )

            kv_capacity = _kv_pool_capacity_tokens(model_runner)
            if kv_capacity is not None:
                before = len(cases)
                cases = [
                    (bs, seq_len, ip, prefix_len)
                    for (bs, seq_len, ip, prefix_len) in cases
                    if required_kv_tokens(bs, seq_len, prefix_len, is_prefill=True) <= kv_capacity
                ]
                skipped = before - len(cases)
                if skipped:
                    print(f"[DSA] Dropped {skipped} context cases beyond actual KV pool capacity={kv_capacity} tokens")

            if sglang_dsa_mqa_logits_chunking_supported():
                print("[DSA] SGLang MQA logits chunk path detected; skipping collector indexer workspace pre-filter")
            else:
                free_mem, total_mem = torch.cuda.mem_get_info()
                workspace_limit = dsa_indexer_workspace_limit_bytes(free_mem, total_mem)
                before = len(cases)
                cases = [
                    (bs, seq_len, ip, prefix_len)
                    for (bs, seq_len, ip, prefix_len) in cases
                    if dsa_indexer_workspace_bytes(
                        bs,
                        seq_len,
                        prefix_len,
                        head_num,
                        target_tp_size,
                        is_prefill=True,
                    )
                    <= workspace_limit
                ]
                skipped = before - len(cases)
                if skipped:
                    print(
                        f"[DSA] Dropped {skipped} context cases beyond estimated indexer "
                        f"workspace limit={workspace_limit} bytes (free={free_mem}, total={total_mem})"
                    )

        logged_count = run_attention_torch(
            model_runner=model_runner,
            test_cases=cases,
            head_num=head_num,
            test_layer=0,
            num_warmup=3,
            num_iterations=10,
            device=device,
            output_path=output_path,
            attn_type=attn_type,
            model_path=model_path,
            kv_cache_dtype=kv_cache_dtype,
            compute_dtype=compute_dtype,
            gemm_type=gemm_type,
            target_tp_size=target_tp_size,
            use_module_cuda_graph=use_module_cuda_graph,
        )
        if cases and logged_count == 0:
            raise RuntimeError(
                f"{attn_type.upper()} module {phase_name.lower()} produced no perf rows "
                f"for model={model_path}, heads={head_num}, kv={kv_cache_dtype}, gemm={gemm_type}"
            )
    finally:
        cleanup_distributed()
        torch.cuda.empty_cache()
        gc.collect()


def _run_mla_subprocess(
    attn_type: str,
    head_num: int,
    model_path: str,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    is_prefill: bool,
    gpu_id: int,
    output_path: str | None = None,
    attention_backend: str | None = None,
    batch_size_filter: int | None = None,
    target_tp_size: int = 1,
):
    """Run MLA/DSA benchmark in a subprocess with CUDA_VISIBLE_DEVICES isolation."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    phase = "context" if is_prefill else "generation"
    output_repr = f'"{output_path}"' if output_path else "None"
    backend_repr = f'"{attention_backend}"' if attention_backend else "None"
    batch_filter_repr = "None" if batch_size_filter is None else str(batch_size_filter)
    code = (
        f'import sys; sys.path.insert(0, "{os.path.dirname(os.path.abspath(__file__))}")\n'
        f"from collect_mla_module import run_mla_module\n"
        f'run_mla_module("{attn_type}", {head_num}, "{model_path}", '
        f'"{kv_cache_dtype}", "{compute_dtype}", "{gemm_type}", {is_prefill}, '
        f"0, {output_repr}, {backend_repr}, {batch_filter_repr}, {target_tp_size})\n"
    )

    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    # Per-subprocess timeout: keep the historical 120 s default, but allow
    # GLM-5 DSA piecewise CUDA graph capture runs to raise it from the launch
    # script. First-time capture/compile can legitimately exceed 120 s.
    subprocess_timeout = int(os.environ.get("AIC_MLA_MODULE_SUBPROCESS_TIMEOUT_SEC", "120"))
    try:
        stdout, _ = proc.communicate(timeout=subprocess_timeout)
        if stdout:
            print(stdout.decode("utf-8", errors="replace"))
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        print(
            f"  WARNING: {attn_type.upper()} module {phase} subprocess timed out "
            f"after {subprocess_timeout}s "
            f"(heads={head_num}, model={model_path}, kv={kv_cache_dtype}, "
            f"gemm={gemm_type}, target_tp={target_tp_size}) — skipping"
        )
        return  # Skip this config instead of raising

    if proc.returncode != 0:
        # Include last lines of subprocess output in the error so the cause
        # is visible in the error group (subprocess stderr is merged into stdout).
        tail = ""
        if stdout:
            lines = stdout.decode("utf-8", errors="replace").strip().splitlines()
            tail = "\n".join(lines[-30:])  # last 30 lines should include traceback
        raise RuntimeError(
            f"{attn_type.upper()} module {phase} subprocess failed (exit code {proc.returncode})\n"
            f"--- subprocess output (last 30 lines) ---\n{tail}"
        )


def run_mla_module_worker(
    seq_len: int,
    batch_size: int,
    num_heads: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    model_path: str,
    attn_type: str,
    attention_backend: str | None = None,
    target_tp_size: int = 1,
    *,
    perf_filename: str,
    device: str = "cuda:0",
):
    """Worker-compatible wrapper used by collector/collect.py.

    Each call runs ALL (batch_size, seq_len) combos for the given
    (attn_type, num_heads, precision, model) combo in a subprocess.
    DSA context uses batch_size to shard the prefix sweep across subprocesses;
    other module cases keep batch_size as a placeholder.

    For wideep MLA test cases, attention_backend is the 9th positional
    element specifying which backend to benchmark (e.g. "flashinfer", "fa3").
    For DSA test cases, it defaults to None and _get_backends() is used.

    perf_filename and device are keyword-only arguments supplied by
    collect.py via functools.partial and the worker dispatch loop.
    """
    device_str = str(device) if not isinstance(device, str) else device
    gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0
    is_prefill = "context" in perf_filename
    batch_size_filter = batch_size if is_prefill and attn_type == "dsa" and batch_size > 0 else None

    print(f"\n{'=' * 60}")
    print(
        f"{attn_type.upper()} Module {'Context' if is_prefill else 'Generation'}: "
        f"model={model_path}, heads={num_heads}, target_tp={target_tp_size}, kv={kv_cache_dtype}, "
        f"compute={compute_dtype}, gemm={gemm_type}, "
        f"backend={attention_backend or 'auto'}, batch_filter={batch_size_filter or 'all'}, GPU={gpu_id}"
    )
    print(f"{'=' * 60}")

    # Resolve output directory for perf data files.  When collect.py provides
    # a bare filename (e.g. "dsa_context_module_perf.txt"), write to CWD so
    # the perf files land next to vllm's output — matching artifact collection.
    output_path = os.path.dirname(perf_filename) or os.getcwd()

    _run_mla_subprocess(
        attn_type=attn_type,
        head_num=num_heads,
        model_path=model_path,
        kv_cache_dtype=kv_cache_dtype,
        compute_dtype=compute_dtype,
        gemm_type=gemm_type,
        is_prefill=is_prefill,
        gpu_id=gpu_id,
        output_path=output_path,
        attention_backend=attention_backend,
        batch_size_filter=batch_size_filter,
        target_tp_size=target_tp_size,
    )


def _cleanup():
    torch.cuda.empty_cache()
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="SGLang MLA/DSA Module Benchmark")
    parser.add_argument("--mode", choices=["context", "generation"], required=True)
    parser.add_argument("--attn-type", choices=["mla", "dsa"], default=None, help="If not set, runs both")
    parser.add_argument("--model", type=str, default=None, help="HuggingFace model path")
    parser.add_argument("--num-heads", type=int, default=None, help="Filter by head count")
    parser.add_argument("--kv-cache-dtype", choices=["bfloat16", "fp8"], default=None)
    parser.add_argument("--output-path", default=None, help="Output directory for perf files")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    args = parser.parse_args()

    # Determine which attn_types to run
    if args.attn_type:
        attn_types = [args.attn_type]
    else:
        attn_types = sorted({spec.attention_type for spec in get_mla_module_model_specs(apply_model_filter=False)})

    for attn_type in attn_types:
        # Determine models
        if args.model:
            models = [args.model]
        else:
            models = [
                spec.model_path
                for spec in get_mla_module_model_specs(attention_type=attn_type, apply_model_filter=False)
            ]

        for model_path in models:
            print(f"\n{'=' * 60}")
            print(f"Model: {model_path}  |  Attention: {attn_type.upper()}  |  Mode: {args.mode}")
            print(f"{'=' * 60}")

            native_heads = _module_model_native_heads(model_path)
            head_nums = (
                [args.num_heads]
                if args.num_heads
                else [h for h in get_mla_module_sweep_spec("sglang").inner_sweep_head_counts if h <= native_heads]
            )

            for compute_dtype, kv_dtype, gemm_type in _get_precision_combos(args.mode):
                if args.kv_cache_dtype and kv_dtype != args.kv_cache_dtype:
                    continue

                for head_num in head_nums:
                    is_prefill = args.mode == "context"
                    gpu_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
                    try:
                        run_mla_module(
                            attn_type=attn_type,
                            head_num=head_num,
                            model_path=model_path,
                            kv_cache_dtype=kv_dtype,
                            compute_dtype=compute_dtype,
                            gemm_type=gemm_type,
                            is_prefill=is_prefill,
                            gpu_id=gpu_id,
                            output_path=args.output_path,
                        )
                    except Exception as e:
                        print(f"  FAILED: {e}")
                        traceback.print_exc()

    print(f"\n{'=' * 50}")
    print("ALL TESTS COMPLETED")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
