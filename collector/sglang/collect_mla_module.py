# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MLA Module Collector for SGLang — unified MLA and DSA benchmarking.

Profiles the complete attention module forward pass at the model-runner level,
using SGLang's own ServerArgs → ModelRunner → ForwardBatch pipeline with dummy
weights. Op names and data schema are aligned with vLLM and TRT-LLM
collect_mla_module.py so that perf_database queries work across frameworks.

Supported models and their attention types are defined in SUPPORTED_MODELS.

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
import subprocess
import sys
import tempfile
import traceback
from importlib.metadata import version as get_version

import numpy as np
import torch

try:
    from helper import benchmark_with_power, get_sm_version, log_perf
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from helper import benchmark_with_power, get_sm_version, log_perf


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

SUPPORTED_MODELS: dict[str, str] = {
    "deepseek-ai/DeepSeek-V3": "mla",
    "deepseek-ai/DeepSeek-V3.2": "dsa",
    "zai-org/GLM-5": "dsa",
}

MODEL_ARCHITECTURE: dict[str, str] = {
    "deepseek-ai/DeepSeek-V3": "DeepseekV3ForCausalLM",
    "deepseek-ai/DeepSeek-V3.2": "DeepseekV32ForCausalLM",
    "zai-org/GLM-5": "GlmMoeDsaForCausalLM",
}

# Native num_attention_heads per model — used to filter TP-sim head counts
# and to always override correctly when head_num != native.
MODEL_NATIVE_HEADS: dict[str, int] = {
    "deepseek-ai/DeepSeek-V3": 128,
    "deepseek-ai/DeepSeek-V3.2": 128,
    "zai-org/GLM-5": 64,
}

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
    if config.get("model_type") in ("deepseek_v32", "glm_moe_dsa"):
        config["architectures"] = ["DeepseekV3ForCausalLM"]
        config["model_type"] = "deepseek_v3"

    # Strip auto_map to prevent transformers from attempting to download
    # custom config/model classes from HuggingFace when trust_remote_code=True.
    config.pop("auto_map", None)

    tmp_dir = os.path.join(
        tempfile.gettempdir(),
        f"aic_sglang_config_{model_id.replace('/', '_')}_{os.getpid()}",
    )
    os.makedirs(tmp_dir, exist_ok=True)
    with open(os.path.join(tmp_dir, "config.json"), "w") as f:
        json.dump(config, f)

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
            return "flashinfer"


def _get_mla_backend_list() -> list[str]:
    """Return all MLA backends to sweep for wideep MLA collection.

    Per-architecture backends:
      SM >= 100: ["trtllm_mla"]  — flashinfer MLA is not supported on Blackwell;
                  sglang auto-promotes to trtllm_mla and then fails kv_cache_dtype
                  validation.  Existing B200 perf data contains only trtllm_mla.
      SM >= 90:  ["flashinfer", "fa3"]
      SM < 90:   ["flashinfer"]
    """
    sm = get_sm_version()
    if sm >= 100:
        return ["trtllm_mla"]
    elif sm >= 90:
        return ["flashinfer", "fa3"]
    else:
        return ["flashinfer"]


# ═══════════════════════════════════════════════════════════════════════
# Test Case Generation
# ═══════════════════════════════════════════════════════════════════════

# Sweep ranges — aligned with vllm/trtllm collect_mla_module.py
_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
_SEQ_LENGTHS = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
_HEAD_NUMS = [128, 64, 32, 16, 8]  # 8 covers GLM-5 (native 64) at tp=8


def get_context_test_cases(attn_type: str):
    """Context-phase test cases.

    Returns list of [seq_len, batch_size, num_heads, kv_cache_dtype,
                     compute_dtype, gemm_type, perf_filename].
    """
    cases = []
    base_fname = f"{attn_type}_context_module_perf.txt"
    for compute_dtype, kv_dtype, gemm_type in _get_precision_combos("context"):
        for num_heads in _HEAD_NUMS:
            for batch_size in _BATCH_SIZES:
                for seq_len in _SEQ_LENGTHS:
                    if batch_size * seq_len > 128 * 1024:
                        continue
                    if seq_len >= 8192 and batch_size > 8:
                        continue
                    cases.append([seq_len, batch_size, num_heads, kv_dtype, compute_dtype, gemm_type, base_fname])
    return cases


def get_generation_test_cases(attn_type: str):
    """Generation-phase test cases.

    Returns list of [kv_cache_len, batch_size, num_heads, kv_cache_dtype,
                     compute_dtype, gemm_type, perf_filename].
    """
    cases = []
    base_fname = f"{attn_type}_generation_module_perf.txt"
    for compute_dtype, kv_dtype, gemm_type in _get_precision_combos("generation"):
        for num_heads in _HEAD_NUMS:
            for batch_size in _BATCH_SIZES:
                for seq_len in _SEQ_LENGTHS:
                    if batch_size * seq_len > 256 * 1024:
                        continue
                    if seq_len >= 8192 and batch_size > 16:
                        continue
                    cases.append([seq_len, batch_size, num_heads, kv_dtype, compute_dtype, gemm_type, base_fname])
    return cases


def _build_module_test_cases(attn_type: str, mode: str):
    """Build one test case per unique (num_heads, precision, model) group.

    Output format: [seq_len, batch_size, num_heads, kv_cache_dtype,
                    compute_dtype, gemm_type, perf_filename, model_path,
                    attn_type, attention_backend]

    Each test case triggers a subprocess that sweeps all (batch_size, seq_len)
    combinations internally, so we only need one entry per group — not one per
    individual point. seq_len and batch_size are set to 0 as placeholders.

    attention_backend is None for DSA (resolved at runtime by _get_backends()).
    All test cases are 10 elements so that collect.py's ``func(*task, device)``
    maps positional args correctly to run_mla_module_worker().

    When ``AIC_DIAG_B200=1`` is set, this returns a single task
    (heads=16, kv=fp8, gemm=bfloat16, model=DSv3.2) so a B200 diagnostic
    run only has to pay DeepGEMM JIT + model-load cost once.
    """
    base_fname = f"{attn_type}_{mode}_module_perf.txt"
    if os.environ.get("AIC_DIAG_B200") == "1" and attn_type == "dsa" and mode == "generation":
        return [[0, 0, 16, "fp8", "bfloat16", "bfloat16", base_fname, "deepseek-ai/DeepSeek-V3.2", attn_type, None]]
    model_paths = [m for m, t in SUPPORTED_MODELS.items() if t == attn_type]
    cases = []
    for model_path in model_paths:
        native_heads = MODEL_NATIVE_HEADS.get(model_path, 128)
        for compute_dtype, kv_dtype, gemm_type in _get_precision_combos(mode):
            for num_heads in _HEAD_NUMS:
                if num_heads > native_heads:
                    continue  # Skip invalid TP-sim configs
                cases.append(
                    [
                        0,
                        0,
                        num_heads,
                        kv_dtype,
                        compute_dtype,
                        gemm_type,
                        base_fname,
                        model_path,
                        attn_type,
                        None,
                    ]
                )
    return cases


def _build_wideep_mla_test_cases(mode: str):
    """Build test cases for wideep MLA collection (backward-compatible).

    Output format: [seq_len, batch_size, num_heads, kv_cache_dtype,
                    compute_dtype, gemm_type, perf_filename, model_path,
                    attn_type, attention_backend]

    Matches the old collect_wideep_attn.py behavior:
    - Single precision combo (bfloat16 run, logged as fp8_block/fp8)
    - Sweeps multiple attention backends per SM version
    - Only DeepSeek-V3 (the MLA model), not V3.2/GLM-5 (DSA models)
    """
    phase = "context" if mode == "context" else "generation"
    base_fname = f"wideep_{phase}_mla_perf.txt"
    model_paths = [m for m, t in SUPPORTED_MODELS.items() if t == "mla"]
    backends = _get_mla_backend_list()
    cases = []
    for model_path in model_paths:
        native_heads = MODEL_NATIVE_HEADS.get(model_path, 128)
        for backend in backends:
            for num_heads in _HEAD_NUMS:
                if num_heads > native_heads:
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
                        base_fname,
                        model_path,
                        "mla",
                        backend,
                    ]
                )
    return cases


def get_wideep_mla_context_test_cases():
    """collect.py entrypoint for wideep MLA context collection."""
    return _build_wideep_mla_test_cases(mode="context")


def get_wideep_mla_generation_test_cases():
    """collect.py entrypoint for wideep MLA generation collection."""
    return _build_wideep_mla_test_cases(mode="generation")


def get_dsa_context_module_test_cases():
    """collect.py entrypoint for DSA context module collection."""
    return _build_module_test_cases(attn_type="dsa", mode="context")


def get_dsa_generation_module_test_cases():
    """collect.py entrypoint for DSA generation module collection."""
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


def _patch_channel_quant_contiguous():
    """Workaround for GLM-5 / DeepSeek-V3.2 dummy-weight fp8 init crash.

    sglang's ``channel_quant_to_tensor_quant`` path has a downstream ``.view(-1)``
    that trips on non-contiguous tensors when weights come from GLM-5 dummy
    load. Force both inputs contiguous before the call so the reshape
    succeeds. Idempotent — safe to call repeatedly.
    See SGLANG_DSA_COLLECTION_GAPS.md Phase 2a(ii).
    """
    from sglang.srt.layers.quantization import fp8_utils

    orig = fp8_utils.channel_quant_to_tensor_quant
    if getattr(orig, "_aic_contiguous_patched", False):
        return

    def patched(x_q_channel, x_s):
        if not x_q_channel.is_contiguous():
            x_q_channel = x_q_channel.contiguous()
        if x_s is not None and not x_s.is_contiguous():
            x_s = x_s.contiguous()
        return orig(x_q_channel, x_s)

    patched._aic_contiguous_patched = True
    fp8_utils.channel_quant_to_tensor_quant = patched


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


def _install_diag_hooks():
    """Instrument the Blackwell fp8 NSA decode path for root-causing the B200
    illegal memory access at reduced heads x fp8-KV x kv_len>=256.

    Gated on ``AIC_DIAG_B200=1``. Temporary; revert when the collector-setup
    divergence is understood.

    v2 (2026-04-17): pipeline #1146 showed that ``CUDA_LAUNCH_BLOCKING=1``
    masks the original race and produces "Offset increment outside graph
    capture" at every decode probe instead, because blocking mode is
    incompatible with CUDA graph capture. We drop that flag and instead
    record the CUDA stream + graph-capture status at every wrapped call,
    plus hook ``torch.cuda.CUDAGraph`` capture begin/end so we can see when
    capture windows open and close relative to the kernel launches.

    1. Monkey-patches ``deep_gemm.fp8_paged_mqa_logits`` (NSA indexer
       scoring) and ``flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla``
       (B200 decode) to print input shapes + dtypes + capture status
       before each call.
    2. Hooks ``torch.cuda.CUDAGraph.{__init__, capture_begin, capture_end}``
       to print when capture windows open and close.
    3. Leaves the illegal memory access timing-accurate so the original
       crash point still reproduces.
    """
    if os.environ.get("AIC_DIAG_B200") != "1":
        return
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

    def _capture_state_str():
        try:
            stream = torch.cuda.current_stream()
            try:
                cap = torch.cuda.is_current_stream_capturing()
            except Exception:
                cap = "?"
            return f"stream_id={stream.stream_id} capturing={cap}"
        except Exception as e:
            return f"capture_state_err={e}"

    try:
        import deep_gemm

        _orig_mqa = deep_gemm.fp8_paged_mqa_logits

        def _diag_mqa(q_fp8, kv_cache_fp8, weights, seqlens_32, block_tables, schedule_metadata, max_seq_len, **kw):
            # shape/dtype only — never call .tolist()/.item() here, both do
            # D2H syncs that are not permitted inside a CUDA graph capture
            # and would mask whatever non-captureable op the kernel itself
            # triggers.
            print(
                f"[DIAG indexer] {_capture_state_str()} "
                f"q_fp8={tuple(q_fp8.shape)}/{q_fp8.dtype} "
                f"kv={tuple(kv_cache_fp8.shape)}/{kv_cache_fp8.dtype} "
                f"weights={tuple(weights.shape)}/{weights.dtype} "
                f"seqlens={tuple(seqlens_32.shape)}/{seqlens_32.dtype} "
                f"block_tables={tuple(block_tables.shape)} "
                f"max_seq_len={max_seq_len}",
                flush=True,
            )
            return _orig_mqa(
                q_fp8, kv_cache_fp8, weights, seqlens_32, block_tables, schedule_metadata, max_seq_len, **kw
            )

        deep_gemm.fp8_paged_mqa_logits = _diag_mqa
    except Exception as e:
        print(f"[DIAG] failed to patch deep_gemm.fp8_paged_mqa_logits: {e}", flush=True)

    try:
        import flashinfer.decode as fid

        _orig_decode = fid.trtllm_batch_decode_with_kv_cache_mla

        def _diag_decode(**kw):
            # shape/dtype only — see note in _diag_mqa about D2H syncs.
            q = kw.get("query")
            kv = kw.get("kv_cache")
            bt = kw.get("block_tables")
            sl = kw.get("seq_lens")
            print(
                f"[DIAG trtllm_decode] {_capture_state_str()} "
                f"query={tuple(q.shape) if q is not None else None}/{q.dtype if q is not None else None} "
                f"kv_cache={tuple(kv.shape) if kv is not None else None}/{kv.dtype if kv is not None else None} "
                f"block_tables={tuple(bt.shape) if bt is not None else None} "
                f"seq_lens={tuple(sl.shape) if sl is not None else None}/{sl.dtype if sl is not None else None} "
                f"max_seq_len={kw.get('max_seq_len')} "
                f"sparse_mla_top_k={kw.get('sparse_mla_top_k')}",
                flush=True,
            )
            return _orig_decode(**kw)

        fid.trtllm_batch_decode_with_kv_cache_mla = _diag_decode
    except Exception as e:
        print(f"[DIAG] failed to patch flashinfer trtllm_batch_decode: {e}", flush=True)

    try:
        _orig_init = torch.cuda.CUDAGraph.__init__
        _orig_cb = torch.cuda.CUDAGraph.capture_begin
        _orig_ce = torch.cuda.CUDAGraph.capture_end

        def _diag_init(self, *a, **kw):
            print(f"[DIAG cuda_graph.__init__] {_capture_state_str()}", flush=True)
            return _orig_init(self, *a, **kw)

        def _diag_cb(self, *a, **kw):
            print(f"[DIAG cuda_graph.capture_begin] {_capture_state_str()}", flush=True)
            return _orig_cb(self, *a, **kw)

        def _diag_ce(self, *a, **kw):
            print(f"[DIAG cuda_graph.capture_end] {_capture_state_str()}", flush=True)
            return _orig_ce(self, *a, **kw)

        torch.cuda.CUDAGraph.__init__ = _diag_init
        torch.cuda.CUDAGraph.capture_begin = _diag_cb
        torch.cuda.CUDAGraph.capture_end = _diag_ce
    except Exception as e:
        print(f"[DIAG] failed to patch torch.cuda.CUDAGraph: {e}", flush=True)


def load_model_runner(
    model_path: str,
    head_num: int,
    kv_cache_dtype: str,
    attention_backend: str,
    device: str = "cuda:0",
    tp_rank: int = 0,
    gemm_type: str = "bfloat16",
):
    """Load SGLang ModelRunner with dummy weights.

    Args:
        model_path: HuggingFace model path (e.g. "deepseek-ai/DeepSeek-V3.2").
        head_num: Number of attention heads to benchmark.
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

    _install_diag_hooks()
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
        mem_fraction_static=0.5,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        kv_cache_dtype=sglang_kv_dtype,
    )

    # Quantization control: bf16 (dummy weights) vs fp8 (real fp8 weight
    # paths so attention projections and indexer MQA scoring actually fire in
    # fp8). The GLM-5 dummy-weight fp8 init crash
    # (channel_quant_to_tensor_quant .view(-1) on non-contiguous tensor) is
    # worked around by _patch_channel_quant_contiguous below.
    if gemm_type == "fp8_block":
        _patch_channel_quant_contiguous()
        server_args.quantization = "fp8"
    else:
        server_args.quantization = None

    # Disable piecewise CUDA graph — its warmup compile OOMs on large models
    # (e.g. 64 GiB allocation with fp8 + 128 heads on H200).
    # Not a ServerArgs constructor param; set post-init like collect_attn.py.
    # Field renamed in sglang 0.5.10: enable_piecewise_cuda_graph → disable_piecewise_cuda_graph.
    if hasattr(server_args, "disable_piecewise_cuda_graph"):
        server_args.disable_piecewise_cuda_graph = True  # sglang >=0.5.10
    else:
        server_args.enable_piecewise_cuda_graph = False  # sglang <=0.5.9

    server_args.attention_backend = attention_backend
    print(f"Using attention backend: {attention_backend}, kv_cache_dtype: {sglang_kv_dtype}, gpu_id: {gpu_id}")

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
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=0.5,
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
):
    """Run attention benchmark for both prefill and decode phases.

    Args:
        test_cases: List of (batch_size, seq_length, is_prefill) tuples.
        kv_cache_dtype: Perf-DB-compatible string for logging.
        compute_dtype: Perf-DB-compatible string for logging.
        gemm_type: Perf-DB-compatible string for logging.
    """
    attention_module = model_runner.model.model.layers[test_layer].self_attn
    architecture = MODEL_ARCHITECTURE.get(model_path, "unknown")
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

    for test_case in test_cases:
        batch_size, seq_length, is_prefill = test_case

        if is_prefill:
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
            )
        else:
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
            )


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

    print(f"\nPrefill: batch_size={batch_size}, seq_length={seq_length}")

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
        batch.prepare_for_extend()
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
        model_runner.attn_backend.init_forward_metadata(forward_batch)

        hidden_states = torch.randn(
            batch_size * seq_length,
            model_runner.model.config.hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        positions = torch.arange(seq_length, device="cuda").unsqueeze(0).expand(batch_size, -1).contiguous().flatten()
        zero_allocator = BumpAllocator(buffer_size=256, dtype=torch.float32, device="cuda")

        attn_inputs = AttentionInputs(hidden_states, forward_batch, dummy_qkv_latent_func)
        get_attn_tp_context().set_attn_inputs(attn_inputs)

        # Warmup
        for _ in range(num_warmup):
            with torch.no_grad():
                attention_module(
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                    zero_allocator=zero_allocator,
                )

        # Timed runs — skip first 2 for stability
        cuda_times = []
        for i in range(num_iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.no_grad():
                attention_module(
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                    zero_allocator=zero_allocator,
                )
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
                        "tp_size": 1,
                        "step": 0,
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

        print(
            f"  Prefill: {avg_time_ms:.3f} ms "
            f"(min: {np.min(cuda_times):.3f}, max: {np.max(cuda_times):.3f}, "
            f"std: {np.std(cuda_times):.3f})"
        )

    except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
        print(f"  OOM: b={batch_size}, s={seq_length} — skipping")
        torch.cuda.empty_cache()
        return
    except Exception as e:
        traceback.print_exc()
        error_str = str(e).lower()
        if "out of memory" in error_str:
            print(f"  OOM: b={batch_size}, s={seq_length} — skipping")
            torch.cuda.empty_cache()
            return
        if "cuda" in error_str and "illegal" in error_str:
            print("  CUDA illegal access detected — stopping to prevent cascading failures")
            raise
        print("  Skipping this configuration...")
        return
    finally:
        model_runner.req_to_token_pool.clear()
        model_runner.token_to_kv_pool_allocator.clear()
        torch.cuda.empty_cache()


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
        model_worker_batch_decode = batch.get_model_worker_batch()
        forward_batch_decode = ForwardBatch.init_new(model_worker_batch_decode, model_runner)
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

        def kernel_func():
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

        with benchmark_with_power(
            device=device,
            kernel_func=kernel_func,
            num_warmups=num_warmup,
            num_runs=num_iterations,
            repeat_n=1,
        ) as results:
            pass

        avg_time_ms = results["latency_ms"]
        power_stats = results["power_stats"]

        # Log perf — wideep MLA uses isl=seq_len, step=0 (old convention).
        # DSA uses isl=1, step=seq_len (matches vllm convention).
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
                        "tp_size": 1,
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

        print(f"  Decode: {avg_time_ms:.3f} ms")

    except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
        print(f"  OOM: b={batch_size}, s={seq_length} — skipping")
        torch.cuda.empty_cache()
        return
    except Exception as e:
        traceback.print_exc()
        error_str = str(e).lower()
        if "out of memory" in error_str:
            print(f"  OOM: b={batch_size}, s={seq_length} — skipping")
            torch.cuda.empty_cache()
            return
        if "cuda" in error_str and "illegal" in error_str:
            print("  CUDA illegal access detected — stopping to prevent cascading failures")
            raise
        print("  Skipping this configuration...")
        return
    finally:
        model_runner.req_to_token_pool.clear()
        model_runner.token_to_kv_pool_allocator.clear()
        torch.cuda.empty_cache()


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
    # Test case format: [seq_len, batch_size, num_heads, kv_dtype, compute_dtype, gemm_type, ...]
    # run_attention_torch expects: (batch_size, seq_length, is_prefill)
    cases = [
        (tc[1], tc[0], is_prefill)
        for tc in all_cases
        if tc[3] == kv_cache_dtype and tc[4] == compute_dtype and tc[5] == gemm_type and tc[2] == head_num
    ]

    print(f"\n{'=' * 60}")
    print(
        f"{attn_type.upper()} Module {phase_name}: model={model_path}, backend={attention_backend}, "
        f"head_num={head_num}, kv={kv_cache_dtype}, compute={compute_dtype}, gemm={gemm_type}, GPU={gpu_id}"
    )
    print(f"Test cases: {len(cases)}")
    print(f"{'=' * 60}")

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
        )

        run_attention_torch(
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
):
    """Run MLA/DSA benchmark in a subprocess with CUDA_VISIBLE_DEVICES isolation."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    phase = "context" if is_prefill else "generation"
    output_repr = f'"{output_path}"' if output_path else "None"
    backend_repr = f'"{attention_backend}"' if attention_backend else "None"
    code = (
        f'import sys; sys.path.insert(0, "{os.path.dirname(os.path.abspath(__file__))}")\n'
        f"from collect_mla_module import run_mla_module\n"
        f'run_mla_module("{attn_type}", {head_num}, "{model_path}", '
        f'"{kv_cache_dtype}", "{compute_dtype}", "{gemm_type}", {is_prefill}, 0, {output_repr}, {backend_repr})\n'
    )

    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    try:
        stdout, _ = proc.communicate(timeout=1800)  # 30 min for DeepGEMM JIT compile
        if stdout:
            print(stdout.decode("utf-8", errors="replace"))
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"{attn_type.upper()} module {phase} subprocess failed (exit code {proc.returncode})")


def run_mla_module_worker(
    seq_len: int,
    batch_size: int,
    num_heads: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    perf_filename: str,
    model_path: str,
    attn_type: str,
    attention_backend: str | None = None,
    device: str = "cuda:0",
):
    """Worker-compatible positional wrapper used by collector/collect.py.

    Each call runs ALL (batch_size, seq_len) combos for the given
    (attn_type, num_heads, precision, model) combo in a subprocess.
    The seq_len and batch_size args from the individual test case are
    ignored here because the subprocess sweeps all combos internally.

    For wideep MLA test cases, attention_backend is the 10th positional
    element specifying which backend to benchmark (e.g. "flashinfer", "fa3").
    For DSA test cases, it defaults to None and _get_backends() is used.
    """
    device_str = str(device) if not isinstance(device, str) else device
    gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0
    is_prefill = "context" in perf_filename

    print(f"\n{'=' * 60}")
    print(
        f"{attn_type.upper()} Module {'Context' if is_prefill else 'Generation'}: "
        f"model={model_path}, heads={num_heads}, kv={kv_cache_dtype}, "
        f"compute={compute_dtype}, gemm={gemm_type}, "
        f"backend={attention_backend or 'auto'}, GPU={gpu_id}"
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
        attn_types = list(set(SUPPORTED_MODELS.values()))

    for attn_type in attn_types:
        # Determine models
        if args.model:
            models = [args.model]
        else:
            models = [m for m, t in SUPPORTED_MODELS.items() if t == attn_type]

        for model_path in models:
            print(f"\n{'=' * 60}")
            print(f"Model: {model_path}  |  Attention: {attn_type.upper()}  |  Mode: {args.mode}")
            print(f"{'=' * 60}")

            native_heads = MODEL_NATIVE_HEADS.get(model_path, 128)
            head_nums = [args.num_heads] if args.num_heads else [h for h in _HEAD_NUMS if h <= native_heads]

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
