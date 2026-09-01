# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 CSA/HCA attention module collector for TensorRT-LLM.

Benchmarks the full DeepSeek-V4 attention module — q/kv/o projections,
rotary, compressor, indexer (top-k selection) and the sparse MLA kernels —
for the four module tables the SDK's ``ContextDeepSeekV4AttentionModule`` /
``GenerationDeepSeekV4AttentionModule`` consume:

    dsv4_csa_context_module     dsv4_hca_context_module
    dsv4_csa_generation_module  dsv4_hca_generation_module

TRT-LLM ships full DeepSeek-V4 support from 1.3.0rc21: the model class
(``_torch/models/modeling_deepseekv4.py``) and the sparse attention stack
(``_torch/attention_backend/sparse/deepseek_v4/``: DeepseekV4TrtllmAttention,
DeepseekV4Indexer, DeepseekV4CacheManager, Compressor).
``ModelConfig.from_pretrained`` auto-builds ``DeepSeekV4SparseAttentionConfig``
for architecture ``DeepseekV4ForCausalLM`` (model_config.py:948-990
@1.3.0rc23), so this collector constructs ``DeepseekV4Attention`` through the
framework's own config/dispatch path — CSA vs HCA is expressed exclusively by
the checkpoint config's per-layer ``compress_ratios`` value (4 = CSA c4 +
indexer top-k; 128 = HCA c128), mirroring the sglang/vllm module collectors.

Row semantics follow the unified #1429 convention (same as sglang/vllm):
``num_heads`` is the RANK-LOCAL head count with a mandatory ``tp_size``
column (the loader derives native = num_heads * tp_size and validates the
local semantics, aic-core .../operations/dsv4.py); context rows carry
isl = fresh chunk tokens / step = cached prefix, generation rows carry
isl = 1 / step = past KV length. TP shards are simulated by shrinking the
config head/o_groups counts (num_attention_heads = native // tp,
o_groups = max(1, o_groups // tp)) in a single process, matching the vllm
collector; collective latency is therefore excluded, like both references.

Like the vllm collector, weights are dummy, so the CSA indexer top-k selects
degenerate indices; the SDK's topK DELTA calibration
(``dsv4_csa_topk_calib``) corrects this at query time when present and is a
no-op otherwise ((calib or {}).get(...) -> delta 0.0, operations/dsv4.py).
No trtllm calib table is produced yet — same accepted state as vllm 0.24.0.
"""

from __future__ import annotations

__compat__ = "trtllm>=1.3.0rc21"

import gc
import json
import os
import shutil
import sys
import tempfile
import traceback
import weakref

import torch

try:
    from registry_types import PerfFile

    from helper import benchmark_with_power, log_perf
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from registry_types import PerfFile

    from helper import benchmark_with_power, log_perf


ARCHITECTURE = "DeepseekV4ForCausalLM"

# Same geometry constants the sglang/vllm collectors read from
# collector/cases/models/DeepseekV4ForCausalLM_cases.yaml via case_generator.
try:
    from case_generator import (
        _DSV4_MODULE_BATCH_SIZES,
        _DSV4_MODULE_BUDGETS,
        _DSV4_MODULE_SEQ_LENGTHS,
        _DSV4_MODULE_TP_SIZES,
        _selected_dsv4_models,
    )
except ModuleNotFoundError:
    from collector.case_generator import (
        _DSV4_MODULE_BATCH_SIZES,
        _DSV4_MODULE_BUDGETS,
        _DSV4_MODULE_SEQ_LENGTHS,
        _DSV4_MODULE_TP_SIZES,
        _selected_dsv4_models,
    )

ATTN_KIND_TO_COMPRESS_RATIO = {"csa": 4, "hca": 128}

# Universal sweep budgets are DECLARED at the base-op layer
# (cases/base_ops/dsv4_attention.yaml, in this module's hash closure) and
# enforced at generation time with counted drops. These are generator
# constraints — "the op's universal math (identities, budgets)" belongs to
# the base-op/generator layer (case_authoring.md §"Legitimate shape
# narrowing", row 2) — NOT memory-feasibility filters; the collector only
# reads the declared values.
MAX_SEQ_LEN = _DSV4_MODULE_BUDGETS["max_seq_len"]
MAX_CONTEXT_QUERY_TOKENS = _DSV4_MODULE_BUDGETS["max_context_query_tokens"]
MAX_GENERATION_KV_TOKENS = _DSV4_MODULE_BUDGETS["max_generation_kv_tokens"]
DECODE_BATCH_LADDER = _DSV4_MODULE_BUDGETS["decode_batch_ladder"]
CONTEXT_PREFIX_ANCHORS = (0, 128, 2048, 4096)

_MODEL_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
    "aiconfigurator",
    "model_configs",
)


# ═══════════════════════════════════════════════════════════════════════
# Case population
# ═══════════════════════════════════════════════════════════════════════


def _filter_shapes(mode: str, drops: dict[str, int] | None = None):
    """(bs, sl, prefix) grid with the vllm collector's budget filters
    (collector/vllm/collect_dsv4_attn.py:833-880).

    Budget drops are counted per reason (layer_permissions.md memory-filter
    rule 3: "drops are counted, never silent") — the caps are env-overridable,
    so a misconfigured cap must be visible, not a silently-empty population.
    """
    drops = drops if drops is not None else {}

    def _drop(reason: str) -> None:
        drops[reason] = drops.get(reason, 0) + 1

    smoke = "--smoke" in sys.argv
    batch_sizes = [1] if smoke else _DSV4_MODULE_BATCH_SIZES
    seq_lens = [64] if smoke else _DSV4_MODULE_SEQ_LENGTHS
    shapes = []
    for bs in batch_sizes:
        for sl in seq_lens:
            if sl > MAX_SEQ_LEN:
                _drop("seq_len_cap")
                continue
            if mode == "context":
                prefixes = (0, 128) if smoke else tuple(dict.fromkeys(CONTEXT_PREFIX_ANCHORS + (MAX_SEQ_LEN - sl,)))
                for prefix in prefixes:
                    if prefix < 0 or prefix + sl > MAX_SEQ_LEN:
                        _drop("prefix_bounds")
                        continue
                    if bs * sl > MAX_CONTEXT_QUERY_TOKENS:
                        _drop("context_query_tokens_cap")
                        continue
                    if bs * (prefix + sl) > MAX_GENERATION_KV_TOKENS:
                        _drop("context_kv_tokens_cap")
                        continue
                    shapes.append((bs, sl, prefix))
            else:
                if bs * sl > MAX_GENERATION_KV_TOKENS:
                    _drop("generation_kv_tokens_cap")
                    continue
                # Declared decode batch ladder (module_budgets).
                if any(sl >= floor and bs > max_bs for floor, max_bs in DECODE_BATCH_LADDER):
                    _drop("decode_batch_ladder")
                    continue
                shapes.append((bs, sl, 0))
    return shapes


def _build_dsv4_test_cases(mode: str, attn_kind: str) -> list[dict]:
    cases: list[dict] = []
    tp_sizes = [1] if "--smoke" in sys.argv else _DSV4_MODULE_TP_SIZES
    drops: dict[str, int] = {}
    # Loop-invariant: same shape grid for every (model, tp) pair, so compute
    # (and count drops) once. Case order is unchanged.
    shapes = _filter_shapes(mode, drops)
    if drops:
        total_dropped = sum(drops.values())
        print(
            f"[trtllm-dsv4] {mode}/{attn_kind}: dropped {total_dropped} shape(s) at generation (budget filter): {drops}"
        )
    for model_path in _selected_dsv4_models():
        for tp_size in tp_sizes:
            for bs, sl, prefix in shapes:
                params = [sl, bs, tp_size, "fp8", "bfloat16", "fp8_block", model_path, attn_kind]
                case_id = f"dsv4_{attn_kind}_{mode}_b{bs}_s{sl}_tp{tp_size}_{model_path.replace('/', '_')}"
                if mode == "context":
                    params.append(prefix)
                    case_id += f"_p{prefix}"
                cases.append({"id": case_id, "params": params})
    if not cases:
        # Zero cases with no logged reason is a population bug
        # (layer_permissions.md); with the reasons logged it is an explicit
        # configuration error (e.g. an env cap set below the whole grid).
        raise RuntimeError(f"dsv4 {mode}/{attn_kind}: budget filter dropped every shape (drops={drops})")
    return cases


def get_dsv4_csa_context_test_cases() -> list[dict]:
    return _build_dsv4_test_cases("context", "csa")


def get_dsv4_hca_context_test_cases() -> list[dict]:
    return _build_dsv4_test_cases("context", "hca")


def get_dsv4_csa_generation_test_cases() -> list[dict]:
    return _build_dsv4_test_cases("generation", "csa")


def get_dsv4_hca_generation_test_cases() -> list[dict]:
    return _build_dsv4_test_cases("generation", "hca")


# ═══════════════════════════════════════════════════════════════════════
# Module construction (framework dispatch — no manual backend pinning)
# ═══════════════════════════════════════════════════════════════════════

# Size-1 cache of the last constructed attention module, keyed by
# (model_path, attn_kind, tp_size). collect.py workers are persistent and the
# case order groups shapes into long same-geometry runs, so consecutive tasks
# reuse one module and skip the ~30s construction (from_pretrained + weight
# creation + fp8 post-processing) that dominated the 42s/case cost — the
# per-case KV cache manager, metadata, CUDA-graph capture and benchmark are
# untouched, and reusing one module across batches is exactly what serving
# does. Owner-approved perf change 2026-08-09.
_MODULE_CACHE: dict = {}


def _cached_dsv4_attention_module(model_path: str, attn_kind: str, tp_size: int, device: str):
    key = (model_path, attn_kind, int(tp_size), device)
    hit = _MODULE_CACHE.get(key)
    if hit is not None:
        return hit
    # Evict the previous geometry before building the next (bounded memory:
    # at most one module's weights are ever retained).
    if _MODULE_CACHE:
        _MODULE_CACHE.clear()
        gc.collect()
        torch.cuda.empty_cache()
    entry = create_dsv4_attention_module(model_path=model_path, attn_kind=attn_kind, tp_size=tp_size, device=device)
    _MODULE_CACHE[key] = entry
    return entry


def generation_request_geometry(seq_len: int) -> dict[str, int]:
    """Decode-request state triple, single-sourced so the serving invariant
    cannot drift between construction sites.

    Serving population (model_engine.py ORDINARY decode branch @1.3.0rc23):
    ``past_seen_token_num = request.max_beam_num_tokens - 1`` (:4308),
    ``position_id = past_seen_token_num`` (:4315),
    ``request.cached_tokens = past_seen_token_num`` (:4332) and
    ``num_cached_tokens_per_seq.append(past_seen_token_num - ...)``
    (:4333-4335) — i.e. the new token's POSITION equals the PAST-SEEN/CACHED
    token count. (The extend/spec-decode branch at :4148,4164-4169 keeps the
    same invariant; this collector's cases are ordinary decode.)
    The collector's decode dummy request registers ``seq_len + 1`` beam tokens
    (``request_tokens = max_seq = seq_len + 1``), so past-seen == cached ==
    position == ``seq_len``, and the persisted row ``step`` (past-KV length,
    #1429 row semantics) is the same value.
    """
    return {"num_cached_tokens": seq_len, "position": seq_len, "persisted_step": seq_len}


def _patched_config_dir(model_path: str, compress_ratio: int, tp_size: int) -> tuple[str, dict]:
    """Write a single-layer DSV4 config with the requested compress ratio and
    TP-local head counts into a temp dir.

    Uses the AIC-packaged config (no HF download / trust_remote_code lock —
    same rationale as collect_mla_module._resolve_local_model_path).
    ``compress_ratios=[compress_ratio]`` expresses CSA (4) vs HCA (128)
    exactly like the vllm collector (collect_dsv4_attn.py:133 @0.24.0);
    ModelConfig.from_pretrained reads it into
    DeepSeekV4SparseAttentionConfig (model_config.py:956-990 @1.3.0rc23).
    """
    cfg_fname = model_path.replace("/", "--") + "_config.json"
    config_file = os.path.join(_MODEL_CONFIG_DIR, cfg_fname)
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"AIC packaged config not found for {model_path!r}: expected {config_file}")
    with open(config_file) as f:
        config = json.load(f)

    native_heads = int(config["num_attention_heads"])
    if native_heads % tp_size != 0:
        raise ValueError(f"tp_size={tp_size} does not divide native head count {native_heads} of {model_path}")
    local_heads = native_heads // tp_size
    # o_groups shards with TP like the SDK model's
    # local_o_groups = max(1, o_groups // tp_size)
    # (aic-core .../sdk/models/deepseek_v4.py:116).
    if "o_groups" not in config:
        # Fail loudly (case_authoring.md "unresolvable declarations"): a
        # defaulted value would silently benchmark a different attention
        # geometry, and o_groups is not part of the logged row. Both packaged
        # DSV4 artifacts declare it (Flash 8, Pro 16).
        raise KeyError(
            f"AIC packaged config for {model_path!r} omits 'o_groups'; the TP-local geometry cannot be resolved"
        )
    o_groups = int(config["o_groups"])
    local_o_groups = max(1, o_groups // tp_size)

    config.pop("auto_map", None)
    config["architectures"] = [ARCHITECTURE]
    # The DSV4 release artifacts tag model_type "deepseek_ref" (remote-code
    # config class). TRT-LLM's native config class registers as
    # "deepseek_v4" (configs/__init__.py:38 @1.3.0rc23,
    # CONFIG_MAPPING.register) and its defaults supply the DSV4 dims the
    # artifact json omits (kv_lora_rank=448, v_head_dim=512,
    # qk_nope_head_dim=448 — configs/deepseekv4.py:35-39 @1.3.0rc23), so
    # rewrite the type instead of pulling remote code.
    config["model_type"] = "deepseek_v4"
    config["num_hidden_layers"] = 1
    config["compress_ratios"] = [compress_ratio]
    config["num_attention_heads"] = local_heads
    config["o_groups"] = local_o_groups

    tmp_dir = tempfile.mkdtemp(prefix=f"aic_dsv4_{model_path.replace('/', '_')}_")
    with open(os.path.join(tmp_dir, "config.json"), "w") as f:
        json.dump(config, f)
    return tmp_dir, {"native_heads": native_heads, "local_heads": local_heads}


def create_dsv4_attention_module(
    model_path: str,
    attn_kind: str,
    tp_size: int,
    device: str = "cuda:0",
):
    """Build DeepseekV4Attention through TRT-LLM's own config path.

    Mirrors collect_mla_module.create_attention_layer: framework-constructed
    module, dummy weights, serving quant-exclusion pass. The attention module
    (an MLA subclass) contains the projections, indexer and compressor — the
    unit the SDK module tables bill.
    """
    # Registers model_type "deepseek_v4" -> DeepseekV4Config into transformers
    # CONFIG_MAPPING at import (configs/__init__.py @1.3.0rc23); required
    # before ModelConfig.from_pretrained resolves the patched config.
    import tensorrt_llm._torch.configs  # noqa: F401
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4Attention
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig

    try:
        from .collect_mla_module import _apply_gemm_type_quant, initialize_dummy_weights
    except ImportError:
        from collect_mla_module import _apply_gemm_type_quant, initialize_dummy_weights

    compress_ratio = ATTN_KIND_TO_COMPRESS_RATIO[attn_kind]
    config_dir, head_info = _patched_config_dir(model_path, compress_ratio, tp_size)

    try:
        mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
        model_config = ModelConfig.from_pretrained(
            config_dir,
            mapping=mapping,
            enable_min_latency=False,
            use_cuda_graph=False,
            force_dynamic_quantization=False,
            spec_config=None,
            max_num_tokens=131072,
            max_seq_len=163840,
            moe_max_num_tokens=None,
            moe_load_balancer=None,
            lora_config=None,
            mm_encoder_only=False,
            attn_backend="TRTLLM",
            moe_backend="CUTLASS",
            skip_create_weights_in_init=True,
        )
    finally:
        shutil.rmtree(config_dir, ignore_errors=True)

    if (
        model_config.sparse_attention_config is None
        or getattr(model_config.sparse_attention_config, "algorithm", None) != "deepseek_v4"
    ):
        raise RuntimeError(
            "ModelConfig.from_pretrained did not build a deepseek_v4 sparse_attention_config "
            f"for {model_path!r} — got {model_config.sparse_attention_config!r}. "
            "The framework dispatch contract changed; re-audit against the runtime version."
        )

    _apply_gemm_type_quant(model_config, "fp8_block", use_fp8_kv_cache=True)

    # Provenance: print the RESOLVED kernel-selection knobs (same auto-build
    # code path default serving takes when the user sets no
    # sparse_attention_config; model_config.py:948-990 @1.3.0rc23). Audits
    # compare these against serving defaults — e.g. the 12.7x CSA
    # long-prefix gap vs vllm traces to cute-dsl topk/paged-mqa being
    # default-OFF here (2026-08-06 audit).
    _sc = model_config.sparse_attention_config
    print(
        "[dsv4-collector] resolved sparse config: "
        + ", ".join(
            f"{k}={getattr(_sc, k, None)}"
            for k in (
                "indexer_k_dtype",
                "use_cute_dsl_topk",
                "use_cute_dsl_paged_mqa_logits",
                "enable_heuristic_topk",
                "q_split_threshold",
                "skip_indexer_for_short_seqs",
                "index_topk",
                "compress_ratios",
            )
        )
    )

    aux_stream = torch.cuda.Stream(device=device)
    attn_module = DeepseekV4Attention(
        model_config=model_config,
        layer_idx=0,
        aux_stream=aux_stream,
    )

    # Serving applies QuantConfig.exclude_modules before weight creation
    # (apply_quant_config_exclude_modules, modeling_utils.py @runtime version);
    # mirror it like collect_mla_module does (see the long citation there).
    quant_config = model_config.quant_config
    if quant_config is not None and quant_config.exclude_modules is not None:
        excluded_replacement = QuantConfig(kv_cache_quant_algo=quant_config.kv_cache_quant_algo)
        for module_name, module in attn_module.named_modules():
            if getattr(module, "quant_config", None) is None:
                continue
            if quant_config.is_module_excluded_from_quantization(module_name):
                module.quant_config = excluded_replacement

    for module in attn_module.modules():
        if callable(getattr(module, "create_weights", None)):
            module.create_weights()
    attn_module.to(device)

    initialize_dummy_weights(attn_module)
    for module in attn_module.modules():
        if hasattr(module, "post_load_weights") and not getattr(module, "_weights_removed", False):
            module.post_load_weights()

    attn_module.eval()
    attn_module.requires_grad_(False)
    return attn_module, model_config, head_info


# ═══════════════════════════════════════════════════════════════════════
# KV cache + metadata
# ═══════════════════════════════════════════════════════════════════════


def create_dsv4_kv_cache_and_metadata(
    model_config,
    attn_module,
    batch_size: int,
    seq_len: int,
    is_context: bool,
    prefix_len: int = 0,
    device: str = "cuda:0",
):
    """DSV4 cache manager + attention metadata, following the serving
    construction path with a DIRECT pinned citation on every hand-set field
    (layer_permissions.md metadata-parity rule; audited against TensorRT-LLM
    1.3.0rc23 sources, 2026-08-19):

    - manager construction  -> pyexecutor/_util.py:1843-1867 (is_mla branch)
    - Metadata construction -> pyexecutor/model_engine.py:2475-2489
      (_set_up_attn_metadata)
    - mixed-batch metadata assignment -> model_engine.py:4735-4777
      (seq_lens :4735-4739; request_ids/prompt_lens/num_contexts
      :4759-4761; KVCacheParams(use_cache=True, num_cached_tokens_per_seq)
      :4772-4775)
    - context per-request population -> begin_compute/prompt slicing
      :3941-3945, position_ids.extend(range(begin_compute, ...)) :3946-3947,
      past-seen prefix accounting :3960-3998
    - ordinary decode per-request population -> :4308-4337
      (past_seen/position/cached/prompt_len)
    - decode dummy-request prompt_len proof -> resource_manager.py:988-995
      (is_gen: req.prompt_len = token_num - 1; py_prompt_len = prompt_len),
      consumed at model_engine.py:4336
    """
    from tensorrt_llm._torch.attention_backend.interface import (
        AttentionRuntimeFeatures,
        KVCacheParams,
    )
    from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType

    # DeepseekV4CacheManager derives from KVCacheManagerV2, which consumes the
    # llm-args pydantic KvCacheConfig (kv_cache_manager_v2.py reads fields
    # like enable_swa_scratch_reuse) — the config object serving forwards from
    # TorchLlmArgs, not the C++ executor binding the V1 managers take.
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    try:
        from .collect_mla_module import get_kv_cache_manager_cls
    except ImportError:
        from collect_mla_module import get_kv_cache_manager_cls

    config = model_config.pretrained_config
    mapping = model_config.mapping

    kv_lora_rank = config.kv_lora_rank
    qk_rope_head_dim = config.qk_rope_head_dim
    head_dim = kv_lora_rank + qk_rope_head_dim

    # DeepSeek-V4 serving default page size ("TensorRT LLM defaults
    # DeepSeek-V4 to tokens_per_block=128",
    # examples/models/core/deepseek_v4/README.md:66-67 @v1.3.0rc23;
    # DeepseekV4CacheManager asserts 128/256).
    tokens_per_block = 128

    prefix_len = int(prefix_len) if is_context else 0
    if is_context:
        max_seq = prefix_len + seq_len + 1
        total_tokens = seq_len * batch_size
        seq_len_q = seq_len
        kv_cache_len = prefix_len
    else:
        max_seq = seq_len + 1
        total_tokens = batch_size
        seq_len_q = 1
        kv_cache_len = generation_request_geometry(seq_len)["num_cached_tokens"]
    # Serving's max_seq_len is the ENGINE envelope, not the request length:
    # the DSV4 metadata derives num_sparse_topk = window(128) +
    # next_pow2(ceil(max_seq_len/128)) (sparse_deepseek_v4.py:435-444
    # @1.3.0rc23) and the trtllmGen fmha kernel asserts it is a multiple of 4
    # — request-sized envelopes (< 257) yield pow2 1/2 -> 129/130 and crashed
    # every tiny-KV HCA decode case on B200 (smoke round 3). Floor ONLY the
    # envelope; per-request state (add_dummy_requests token_nums, metadata
    # kv lens) stays the real request size, exactly as in serving — flooring
    # token_nums too registered 512-token dummy KV against 4-token metadata
    # and turned the crash into an IMA (smoke round 4).
    request_tokens = prefix_len + seq_len_q if is_context else max_seq
    max_seq = max(max_seq, 512)

    # KVCacheManagerV2 requires an explicit quota (max_tokens or
    # max_gpu_total_bytes; kv_cache_manager_v2.py "Quota not set" @1.3.0rc23)
    # — in serving, free_gpu_memory_fraction is converted to a byte quota by
    # the executor's KV estimation before the manager is built, so pass the
    # equivalent byte quota directly (half of currently-free device memory,
    # matching the DSV4 example's free_gpu_memory_fraction=0.5,
    # examples/models/core/deepseek_v4/README.md:148-151 @v1.3.0rc23). A
    # main-KV-shaped max_tokens cap under-counts on V2 (the byte quota spans
    # main KV + SWA + compressor/indexer caches) and made large-KV shapes
    # fail dummy-request allocation ("Request ID not found in IndexMapper",
    # B200 smoke round 1 2026-08-06).
    free_bytes, _ = torch.cuda.mem_get_info(torch.device(device))
    kv_cache_config = KvCacheConfig(
        tokens_per_block=tokens_per_block,
        max_gpu_total_bytes=int(free_bytes * 0.5),
        enable_block_reuse=False,
    )
    kv_cache_manager_cls = get_kv_cache_manager_cls(model_config, kv_cache_config)
    # fp8 KV rows -> DataType.FP8, matching serving's kv_cache_dtype
    # resolution from kv_cache_quant_algo (set by _apply_gemm_type_quant).
    kv_cache_dtype = DataType.FP8

    # Serving construction site: pyexecutor/_util.py:1843-1867 @1.3.0rc23
    # (is_mla branch): CacheType.SELFKONLY :1846, num_kv_heads=1 :1848,
    # head_dim=kv_lora_rank+qk_rope_head_dim :1849, dtype=kv_cache_dtype,
    # vocab_size=config.vocab_size :1856, sparse_attention_config :1862,
    # pretrained_config + layer_mask forwarded (single-layer here).
    kv_cache_manager = kv_cache_manager_cls(
        kv_cache_config,
        CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=kv_cache_dtype,
        vocab_size=config.vocab_size,
        layer_mask=[True],
        sparse_attention_config=model_config.sparse_attention_config,
        pretrained_config=config,
        model_config=model_config,
    )

    # From here on the manager owns device memory: release it on ANY
    # failure before re-raising, so a failed case cannot leak a
    # batch x max_seq fp8 pool into the worker's next cases (which would
    # then OOM for the wrong reason and pollute the failure log).
    try:
        request_ids = list(range(batch_size))
        token_nums = [request_tokens] * batch_size
        # add_dummy_requests is serving's own warmup-request mechanism
        # (model_engine.py:2141-2160, :2277, :2300 @1.3.0rc23 register dummy
        # requests the same way before capture/warmup); is_gen mirrors the
        # request phase (generation metadata below declares num_contexts=0
        # with cached KV).
        kv_cache_manager.add_dummy_requests(request_ids, token_nums, is_gen=not is_context)

        attention_cls = get_attention_backend(
            model_config.attn_backend,
            model_config.sparse_attention_config,
        )

        sparse_metadata_params = model_config.sparse_attention_config.to_sparse_metadata_params(
            pretrained_config=config
        )

        # Constructor kwargs mirror _set_up_attn_metadata
        # (model_engine.py:2475-2489 @1.3.0rc23); per-batch fields mirror the
        # _prepare_tp_inputs population sites cited per field below.
        attn_metadata = attention_cls.Metadata(
            # max_num_requests/max_num_tokens are CAPACITY BOUNDS
            # (model_engine.py:2476-2477 passes engine-lifetime capacity).
            # This collector prepares exactly one batch, so the tight bound
            # (= this batch's request/token counts) satisfies the same
            # buffer-sizing contract; the parity test asserts
            # max_num_tokens >= sum(seq_lens).
            max_num_requests=batch_size,
            max_num_tokens=total_tokens,
            kv_cache_manager=kv_cache_manager,
            mapping=mapping,
            # seq_lens: per-request current-step token counts — context
            # appends the fresh chunk length, decode appends 1; assigned as a
            # tensor at model_engine.py:4735-4739.
            seq_lens=torch.tensor([seq_len_q] * batch_size, dtype=torch.int32),
            # position_ids: interface dataclass default None
            # (attention_backend/interface.py:100) — serving never assigns it
            # on this metadata; positions travel as a model input (the tensor
            # built in the runner below, mirroring model_engine.py
            # :3946-3947 / :4315).
            position_ids=None,
            # num_contexts = scheduled context-request count
            # (model_engine.py:4761); ordinary decode batches carry 0.
            num_contexts=batch_size if is_context else 0,
            # num_cached_tokens_per_seq: context = begin_compute prefix
            # (:3973-3976); decode = past_seen - compressed (:4333-4335,
            # compressed offset 0 for these single-layer dummy weights);
            # wrapped into KVCacheParams(use_cache=True, ...) exactly as
            # model_engine.py:4772-4775 does.
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[kv_cache_len] * batch_size,
            ),
            # cross: interface default None (interface.py:135) — cross
            # metadata exists only for encoder-decoder attention.
            cross=None,
            request_ids=request_ids,
            # prompt_lens: context = chunk-local fresh length (:3941-3945
            # prompt slicing; the SM100 cached-KV walker consumes chunk-local
            # semantics); decode = py_prompt_len, which for is_gen dummy
            # requests IS the past-KV length: add_dummy_requests sets
            # req.prompt_len = token_num - 1 (resource_manager.py:988-995)
            # and we register token_num = seq_len + 1, so py_prompt_len ==
            # seq_len == kv_cache_len, appended at model_engine.py:4336.
            prompt_lens=[seq_len_q if is_context else kv_cache_len] * batch_size,
            # cached-KV context flag: is_mla AND cache_reuse|chunked_prefill
            # (model_engine.py:2419-2422) — true here exactly for
            # prefix-carrying context cases.
            enable_context_mla_with_cached_kv=bool(is_context and prefix_len > 0),
            runtime_features=AttentionRuntimeFeatures(
                chunked_prefill=False,
                cache_reuse=bool(is_context and prefix_len > 0),
            ),
            # all_rank_num_tokens stays None: attention-DP only
            # (model_engine.py:3280-3283); single-process collection.
            all_rank_num_tokens=None,
            # workspace: optional attention-kernel scratch
            # (attention_backend/trtllm.py:77, lazily grown at :306); an
            # empty tensor is the pre-warmup serving state.
            workspace=torch.tensor([], device=device, dtype=torch.int8),
            # sparse_metadata_params: same to_sparse_metadata_params call
            # serving makes (model_engine.py:2442-2446).
            sparse_metadata_params=sparse_metadata_params,
        )

        if hasattr(attn_module, "indexer") and attn_module.indexer is not None:
            attn_metadata.indexer = attn_module.indexer

        attn_metadata.prepare()
        return kv_cache_manager, attn_metadata, attention_cls
    except Exception:
        kv_cache_manager.shutdown()
        raise


# ═══════════════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════════


def run_dsv4_attn(
    seq_len: int,
    batch_size: int,
    tp_size: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    model_path: str,
    attn_kind: str,
    prefix_len: int = 0,
    *,
    mode: str,
    perf_filename: str,
    device: str = "cuda:0",
    warming_up: int = 10,
    test_ite: int = 6,
):
    import tensorrt_llm
    import tensorrt_llm._torch.utils as _trtllm_utils
    from tensorrt_llm._torch.utils import get_model_extra_attrs, model_extra_attrs

    try:
        from .collect_mla_module import _cleanup
    except ImportError:
        from collect_mla_module import _cleanup

    if attn_kind not in ATTN_KIND_TO_COMPRESS_RATIO:
        raise ValueError(f"unsupported DSV4 attn_kind: {attn_kind}")
    if kv_cache_dtype != "fp8":
        raise ValueError(f"DSV4 module rows are fp8-KV only (got {kv_cache_dtype!r})")
    if compute_dtype != "bfloat16":
        raise ValueError(f"DSV4 module rows are bfloat16-compute only (got {compute_dtype!r})")
    if gemm_type != "fp8_block":
        raise ValueError(f"DSV4 module rows are fp8_block-gemm only (got {gemm_type!r})")

    is_context = mode == "context"
    torch_device = torch.device(device)
    torch.cuda.set_device(torch_device)

    attn_module, model_config, head_info = _cached_dsv4_attention_module(model_path, attn_kind, tp_size, device)

    # Ownership: the KV pool and the process-global extra-attrs slot are
    # released/restored on EVERY exit path (success, dry-run failure,
    # benchmark failure) — workers survive a failed case and run the next
    # one, so a leaked pool would turn later cases into bogus OOM records
    # and a stale attention_metadata weakref would outlive the case.
    saved_extra_attrs = getattr(_trtllm_utils._model_extra_attrs, "attrs", None)
    kv_cache_manager = None
    try:
        kv_cache_manager, attn_metadata, attention_cls = create_dsv4_kv_cache_and_metadata(
            model_config=model_config,
            attn_module=attn_module,
            batch_size=batch_size,
            seq_len=seq_len,
            is_context=is_context,
            prefix_len=prefix_len,
            device=device,
        )

        hidden_size = model_config.pretrained_config.hidden_size
        # int32 position_ids: serving populates them int32 and the DSV4 indexer
        # rope op asserts it (sparse/deepseek_v4/deepseek_v4.py mla_rope_inplace
        # "position_ids must be int32" @1.3.0rc23).
        if is_context:
            num_tokens = seq_len * batch_size
            position_ids = (
                torch.arange(prefix_len, prefix_len + seq_len, device=torch_device, dtype=torch.int32)
                .unsqueeze(0)
                .expand(batch_size, -1)
                .reshape(-1)
                .contiguous()
            )
        else:
            num_tokens = batch_size
            # position == past-seen/cached count, NOT seq_len - 1: see
            # generation_request_geometry (ordinary decode,
            # model_engine.py:4308,4315,4332-4335 @1.3.0rc23). Off-by-one
            # here rotates rope one step early; it does not change kernel
            # shapes or measured cost.
            position_ids = torch.full(
                (batch_size,),
                generation_request_geometry(seq_len)["position"],
                device=torch_device,
                dtype=torch.int32,
            )

        hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=torch_device)

        # FIXME(kernel-limit): context shapes at bs*sl == 262144 query tokens
        # fail with cudaLaunchKernelEx "invalid argument" (grid-dim limit) on
        # B200/1.3.0rc23 — serving chunks prefill at max_num_tokens (<<262144),
        # so these sweep extremes exceed the serving envelope; unverified which
        # kernel hits the limit. Cases fail into the classified log.
        with model_extra_attrs(model_config.extra_attrs):
            get_model_extra_attrs()["attention_metadata"] = weakref.ref(attn_metadata)
            try:
                with torch.inference_mode():
                    attn_module.forward(position_ids, hidden_states, attn_metadata)
            except Exception:
                print("  Dry run failed:")
                traceback.print_exc()
                raise  # the finally below releases the KV pool

        _trtllm_utils._model_extra_attrs.attrs = model_config.extra_attrs
        _trtllm_utils._model_extra_attrs.attrs["attention_metadata"] = weakref.ref(attn_metadata)

        def kernel_func():
            attn_module.forward(position_ids, hidden_states, attn_metadata)

        with benchmark_with_power(
            device=torch_device,
            kernel_func=kernel_func,
            num_warmups=warming_up,
            num_runs=test_ite,
            repeat_n=1,
            allow_graph_fail=False,
        ) as results:
            pass

        latency = results["latency_ms"]

        if is_context:
            isl, step = seq_len, prefix_len
        else:
            isl, step = 1, generation_request_geometry(seq_len)["persisted_step"]

        log_perf(
            item_list=[
                {
                    "model": model_path,
                    "architecture": ARCHITECTURE,
                    "mla_dtype": compute_dtype,
                    "kv_cache_dtype": kv_cache_dtype,
                    "gemm_type": gemm_type,
                    # Rank-LOCAL heads + mandatory tp_size (unified #1429; the
                    # SDK loader derives native = num_heads * tp_size and
                    # validates local semantics).
                    "num_heads": head_info["local_heads"],
                    "batch_size": batch_size,
                    "isl": isl,
                    "tp_size": tp_size,
                    "step": step,
                    "compress_ratio": ATTN_KIND_TO_COMPRESS_RATIO[attn_kind],
                    "latency": f"{latency:.4f}",
                }
            ],
            framework="TRTLLM",
            version=tensorrt_llm.__version__,
            device_name=torch.cuda.get_device_name(device),
            op_name=f"dsv4_{attn_kind}_{mode}_module",
            # Ground truth: the attention backend class TRT-LLM's own selector
            # returned for this sparse config (get_attention_backend); the many
            # internal kernels (indexer deepgemm, sparse MLA, compressor) are
            # not observable from one label — see kernel_source_backends.yaml.
            kernel_source=attention_cls.__name__,
            perf_filename=perf_filename,
            power_stats=results["power_stats"],
        )

        print(
            f"  [dsv4_{attn_kind}_{mode}] b={batch_size} s={seq_len} prefix={prefix_len} "
            f"tp={tp_size} local_heads={head_info['local_heads']}: {latency:.4f} ms"
        )

        return latency
    finally:
        _trtllm_utils._model_extra_attrs.attrs = saved_extra_attrs
        _cleanup(kv_cache_manager)
        gc.collect()
        torch.cuda.empty_cache()


def run_dsv4_attn_worker(
    seq_len: int,
    batch_size: int,
    tp_size: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    model_path: str,
    attn_kind: str,
    prefix_len: int = 0,
    *,
    perf_filename: str,
    device: str = "cuda:0",
) -> None:
    """collect.py worker: one task = one shape = one row.

    Mode is derived from the target perf filename (same convention as the
    vllm collector, collect_dsv4_attn.py:968 @0.24.0).
    """
    mode = "context" if "context" in os.path.basename(perf_filename) else "generation"
    smoke = "--smoke" in sys.argv
    run_dsv4_attn(
        seq_len=seq_len,
        batch_size=batch_size,
        tp_size=tp_size,
        kv_cache_dtype=kv_cache_dtype,
        compute_dtype=compute_dtype,
        gemm_type=gemm_type,
        model_path=model_path,
        attn_kind=attn_kind,
        prefix_len=prefix_len,
        mode=mode,
        perf_filename=perf_filename,
        device=device,
        warming_up=3 if smoke else 10,
        test_ite=3 if smoke else 6,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Collect TRT-LLM DeepSeek-V4 CSA/HCA attention module latency.")
    parser.add_argument("--mode", choices=["context", "generation"], default="context")
    parser.add_argument("--attn-kind", choices=["csa", "hca"], default="csa")
    parser.add_argument("--model-path", default="sgl-project/DeepSeek-V4-Flash-FP8")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--prefix", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    perf_name = PerfFile[f"DSV4_{args.attn_kind.upper()}_{args.mode.upper()}_MODULE"].value
    perf_filename = os.path.join(args.output_path, perf_name) if args.output_path else perf_name
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)

    run_dsv4_attn(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        tp_size=args.tp_size,
        kv_cache_dtype="fp8",
        compute_dtype="bfloat16",
        gemm_type="fp8_block",
        model_path=args.model_path,
        attn_kind=args.attn_kind,
        prefix_len=args.prefix,
        mode=args.mode,
        perf_filename=perf_filename,
        device=args.device,
        warming_up=3,
        test_ite=3,
    )


if __name__ == "__main__":
    main()
