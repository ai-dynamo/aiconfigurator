# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# MiniMax-M3 support landed in TRT-LLM 1.3.0rc19 (modeling_minimaxm3.py +
# attention_backend/sparse/minimax_m3/); this collector follows the rc20 APIs.
__compat__ = "trtllm>=1.3.0rc19"

"""
MSA Module Collector for TRT-LLM — MiniMax-M3 sparse-attention benchmarking.

Profiles the complete MiniMax-M3 sparse-attention module forward pass
(qkv/index projections + per-head QK norm + partial RoPE + index attention +
top-k block selection + sparse GQA + output projection), not a bare kernel.
Uses TRT-LLM's own modeling code (``MiniMaxM3DecoderLayer`` →
``MiniMaxM3Attention``) to construct a single sparse layer with dummy
weights, then extracts the attention module for benchmarking — the same
framework-builder approach as ``collect_mla_module.py``.

The M3 sparse backend has two framework implementations, and this collector
follows whichever one the framework's own dispatch resolves:

* Triton reference (rc19/rc20 only path; rc23 ``implementation="triton"``):
  Python + Triton (sparse/minimax_m3/kernels.py@1.3.0rc20, split into
  triton_backend.py/triton_metadata.py@1.3.0rc23, adapted from SGLang), no SM
  gate — hardware-validated on SM90. Its metadata contract is the
  prebuilt ``attn_metadata.minimax_m3`` dict attachment.
* MSA / fmha_sm100 (1.3.0rc23 ``implementation="msa"``, SM100/103 only):
  ``MiniMaxM3MsaSparseAttention`` on the TrtllmAttention stack
  (sparse/minimax_m3/msa_backend.py@1.3.0rc23). Its metadata is a
  ``TrtllmAttentionMetadata`` subclass carrying flat CUDA-graph-stable MSA
  buffers — there is no ``minimax_m3`` attachment on this path.

``create_kv_cache_and_metadata`` detects the resolved Metadata class (never
the version string) and validates the matching contract; ``kernel_source``
records which backend actually ran.

Supported models and micro-sweeps come from collector v2 YAML
(``cases/models/MiniMaxM3ForCausalLM_cases.yaml`` `mla_module` rows with
``attention_type: msa`` + ``cases/base_ops/mla_module.yaml``).

Usage:
    python collect_msa_module.py --mode context --model MiniMaxAI/MiniMax-M3
    python collect_msa_module.py --mode generation --quick --batch-size 4 --seq-len 2048 --num-heads 64
"""

import argparse
import dataclasses
import gc
import inspect
import os
import sys
import traceback
import weakref

import tensorrt_llm
import torch
import transformers
from tensorrt_llm._torch.attention_backend.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3DecoderLayer
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

# ═══════════════════════════════════════════════════════════════════════
# Config registry patch — TRT-LLM's _CONFIG_REGISTRY has no "minimax_m3"
# entry and transformers 5.5.x AutoConfig doesn't know the model_type
# either (load_pretrained_config would raise in its AutoConfig fallback,
# config_utils.py:569-571@1.3.0rc20).  Serving loads the checkpoint via
# trust_remote_code; the bundled config dir has auto_map stripped (see
# helper._resolve_local_model_path), so mirror the collect_mla_module.py
# glm_moe_dsa precedent and route through a config class instead.  The M3
# modeling code needs attribute access only — its own VL path materialises
# a bare PretrainedConfig the same way (_wrap_dict_as_config,
# modeling_minimaxm3.py:65-78@1.3.0rc20) — so the base PretrainedConfig is
# the faithful offline stand-in.  This changes HOW the config object is
# constructed, never which kernel runs (the M3 sparse dispatch keys on
# sparse_attention_config / model_type, both preserved verbatim).
# ═══════════════════════════════════════════════════════════════════════
from tensorrt_llm._torch.pyexecutor.config_utils import _CONFIG_REGISTRY
from tensorrt_llm._torch.pyexecutor.model_loader import initialize_dummy_weights
from tensorrt_llm._torch.pyexecutor._util import get_kv_cache_manager_cls
from tensorrt_llm._torch.utils import AuxStreamType, get_model_extra_attrs, model_extra_attrs
from tensorrt_llm._utils import torch_dtype_to_binding

try:
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MiniMaxM3SparseAttentionConfig
except ImportError:  # pragma: no cover - rc19 layout
    from tensorrt_llm.bindings.executor import KvCacheConfig
    from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from collector.case_generator import get_mla_module_model_specs, get_mla_module_sweep_spec
from collector.helper import _resolve_local_model_path, benchmark_with_power, get_sm_version, log_perf
from collector.registry_types import PerfFile

# LazyConfigDict resolves entries via getattr(tensorrt_llm._torch.configs,
# name) (config_utils.py:493-498@1.3.0rc20), so expose the base class there
# before registering the model_type.
import tensorrt_llm._torch.configs as _trtllm_configs  # noqa: E402

if not hasattr(_trtllm_configs, "PretrainedConfig"):
    _trtllm_configs.PretrainedConfig = transformers.PretrainedConfig
if "minimax_m3" not in _CONFIG_REGISTRY:
    _CONFIG_REGISTRY["minimax_m3"] = "PretrainedConfig"


# ═══════════════════════════════════════════════════════════════════════
# Test Cases
# ═══════════════════════════════════════════════════════════════════════


def _get_precision_combos(phase: str):
    """Return (compute_dtype, kv_cache_dtype, gemm_type) triples for MSA.

    Precision axes:
      gemm_type    — linear-layer GEMMs (q/k/v/o + index projections)
        bfloat16:  always
        fp8_block: SM >= 89 (Ada / Hopper / Blackwell)
        nvfp4:     SM >= 100 (Blackwell)

      (compute_dtype, kv_cache_dtype) — attention compute + KV cache
        M3 sparse attention at 1.3.0rc20 is bf16-only end to end: the
        sparse backend gathers paged K/V and runs torch matmul GQA +
        Triton softmax in the cache dtype (sparse/minimax_m3/backend.py
        _sparse_gqa_masked@1.3.0rc20), and the index-K side cache is
        allocated bf16 (cache_manager._torch_dtype_for_index_cache
        falls back to bf16 for non-fp16/fp32 main dtypes). No fp8-KV
        combo is declared until the framework grows one.
    """
    sm = get_sm_version()

    gemm_types = ["bfloat16"]
    if sm >= 89:
        gemm_types.append("fp8_block")
    if sm >= 100:
        gemm_types.append("nvfp4")

    attn_combos = [("bfloat16", "bfloat16")]
    return [(c, kv, g) for g in gemm_types for c, kv in attn_combos]


def get_context_test_cases():
    """Context-phase test cases.

    Returns list of [seq_len, batch_size, num_heads, kv_cache_dtype,
                     compute_dtype, gemm_type, prefix_len].
    """
    cases = []
    sweep = get_mla_module_sweep_spec("trtllm")
    for compute_dtype, kv_dtype, gemm_type in _get_precision_combos("context"):
        for num_heads in sweep.inner_sweep_head_counts:
            for b in sweep.context_batch_sizes:
                for s in sweep.context_sequence_lengths:
                    if b * s > sweep.context_max_tokens:
                        continue
                    if (
                        sweep.context_large_sequence_min
                        and s >= sweep.context_large_sequence_min
                        and b > sweep.context_large_sequence_max_batch_size
                    ):
                        continue
                    for prefix_len in sweep.context_prefix_lengths:
                        cases.append([s, b, num_heads, kv_dtype, compute_dtype, gemm_type, prefix_len])
    return cases


def get_generation_test_cases():
    """Generation-phase test cases.

    Returns list of [kv_cache_len, batch_size, num_heads, kv_cache_dtype,
                     compute_dtype, gemm_type].
    """
    cases = []
    sweep = get_mla_module_sweep_spec("trtllm")
    for compute_dtype, kv_dtype, gemm_type in _get_precision_combos("generation"):
        for num_heads in sweep.inner_sweep_head_counts:
            for b in sweep.generation_batch_sizes:
                for s in sweep.generation_sequence_lengths:
                    if b * s > sweep.generation_max_tokens:
                        continue
                    if (
                        sweep.generation_large_sequence_min
                        and s >= sweep.generation_large_sequence_min
                        and b > sweep.generation_large_sequence_max_batch_size
                    ):
                        continue
                    cases.append([s, b, num_heads, kv_dtype, compute_dtype, gemm_type])
    return cases


def _build_module_test_cases(mode: str):
    """Build module-level test cases for one phase.

    Output test case format is positional args for run_msa_module_worker:
    [seq_len, batch_size, num_heads, kv_cache_dtype, compute_dtype, gemm_type,
     model_path (, prefix_len)]
    """
    base_cases = get_context_test_cases() if mode == "context" else get_generation_test_cases()
    model_specs = get_mla_module_model_specs(attention_type="msa", backend="trtllm")
    cases = []
    for model_spec in model_specs:
        for base_case in base_cases:
            s, b, h, kv_dtype, compute_dtype, gemm_type, *rest = base_case
            case = [s, b, h, kv_dtype, compute_dtype, gemm_type, model_spec.model_path]
            if rest:
                case.append(rest[0])
            cases.append(case)
    return cases


def get_msa_context_module_test_cases():
    """collect.py entrypoint for MSA context module collection."""
    return _build_module_test_cases(mode="context")


def get_msa_generation_module_test_cases():
    """collect.py entrypoint for MSA generation module collection."""
    return _build_module_test_cases(mode="generation")


# ═══════════════════════════════════════════════════════════════════════
# Layer Construction
# ═══════════════════════════════════════════════════════════════════════


def _ceil_div(a, b):
    return (a + b - 1) // b


def _round_up(a, b):
    return _ceil_div(a, b) * b


def _replace_quant_config(qc, **kwargs):
    """Replace fields in quant_config; supports both dataclass and Pydantic BaseModel."""
    import dataclasses

    if dataclasses.is_dataclass(qc):
        return dataclasses.replace(qc, **kwargs)
    return qc.model_copy(update=kwargs)


def _set_quant_config(model_config, new_qc):
    """Set quant_config, bypassing ModelConfig freeze if necessary."""
    try:
        model_config.quant_config = new_qc
    except AttributeError:
        object.__setattr__(model_config, "quant_config", new_qc)


def _apply_gemm_type_quant(model_config, gemm_type: str):
    """Apply GEMM quantization to model_config.quant_config.

    M3 sparse attention keeps the KV cache in bf16 (see
    _get_precision_combos), so kv_cache_quant_algo stays None on every
    combo. The index_q/index_k projections are built with
    ``quant_config=None`` regardless (modeling_minimaxm3.py:637-660
    @1.3.0rc20), so GEMM quant here covers qkv_proj / o_proj only —
    matching what serving quantizes for these checkpoints.
    """
    if gemm_type == "bfloat16":
        _set_quant_config(
            model_config,
            _replace_quant_config(
                model_config.quant_config,
                quant_algo=None,
                kv_cache_quant_algo=None,
                exclude_modules=None,
            ),
        )
    elif gemm_type == "fp8_block":
        _set_quant_config(
            model_config,
            _replace_quant_config(
                model_config.quant_config,
                quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
                group_size=128,
                kv_cache_quant_algo=None,
                exclude_modules=None,
            ),
        )
    elif gemm_type == "nvfp4":
        _set_quant_config(
            model_config,
            _replace_quant_config(
                model_config.quant_config,
                quant_algo=QuantAlgo.NVFP4,
                kv_cache_quant_algo=None,
                exclude_modules=None,
            ),
        )
    else:
        raise ValueError(f"Unknown gemm_type: {gemm_type!r}")


def create_msa_attention_layer(
    model_path: str,
    num_heads: int = 64,
    gemm_type: str = "bfloat16",
    device: str = "cuda:0",
):
    """Create a single MiniMax-M3 sparse attention layer from TRT-LLM's own
    modeling code.

    Builds ``MiniMaxM3DecoderLayer`` — the same constructor serving uses for
    layers 3..N-1 (modeling_minimaxm3.py:1191-1196@1.3.0rc20 passes
    ``is_sparse_attention_layer`` / ``disable_index_value`` per
    ``get_sparse_layer_ids`` / ``get_sparse_disable_index_value_layer_ids``) —
    and extracts ``layer.self_attn``.  The single benchmark layer is made
    sparse by truncating the config's per-layer pattern arrays to one sparse
    entry; every scalar sparse field (index heads/dim, block size, top-k) is
    taken from the checkpoint config verbatim.
    """
    mapping = tensorrt_llm.mapping.Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)

    # Resolve to the bundled auto_map-stripped config dir first (see
    # collect_mla_module.create_attention_layer for why: the remote-code
    # path's file lock serializes parallel workers).
    model_path = _resolve_local_model_path(model_path)

    _cfg_dict, _ = transformers.PretrainedConfig.get_config_dict(model_path)
    original_architecture = _cfg_dict.get("architectures", [_cfg_dict.get("model_type", "unknown")])[0]
    # The VL checkpoint nests the text model under text_config; the module
    # benchmark models the text decoder only, mirroring
    # modeling_minimaxm3.get_text_config@1.3.0rc20.
    text_cfg_dict = _cfg_dict.get("text_config", _cfg_dict)

    sparse_cfg_dict = text_cfg_dict.get("sparse_attention_config")
    if not sparse_cfg_dict:
        raise ValueError(
            f"model {model_path} has no sparse_attention_config; the MSA module "
            "collector only supports MiniMax-M3 sparse-attention checkpoints"
        )

    # Serving constructs the M3 runtime with an explicit
    # MiniMaxM3SparseAttentionConfig (llm_args.py:592-668@1.3.0rc20;
    # modeling_minimaxm3.py:1147 rejects sparse layers built without it).
    # ModelConfig.from_pretrained does NOT auto-build it for M3 (only the
    # DeepseekV32/GlmMoeDsa branch exists, model_config.py:617-647), so
    # build it here from the checkpoint's sparse dict — same field mapping
    # MiniMaxM3Attention.__init__ reads (modeling_minimaxm3.py:623-630).
    def _required_sparse_field(key: str) -> int:
        # Unresolvable artifact configuration fails loudly: substituting a
        # literal here would benchmark a different sparse geometry and persist
        # it under this model's identity.
        if key not in sparse_cfg_dict:
            raise RuntimeError(
                f"checkpoint sparse_attention_config is missing '{key}' "
                f"(present: {sorted(sparse_cfg_dict)}); refusing to default the sparse geometry"
            )
        return int(sparse_cfg_dict[key])

    sparse_kwargs = dict(
        sparse_num_index_heads=_required_sparse_field("sparse_num_index_heads"),
        sparse_index_dim=_required_sparse_field("sparse_index_dim"),
        sparse_block_size=_required_sparse_field("sparse_block_size"),
        sparse_topk_blocks=_required_sparse_field("sparse_topk_blocks"),
        sparse_init_blocks=_required_sparse_field("sparse_init_block"),
        sparse_local_blocks=_required_sparse_field("sparse_local_block"),
        sparse_disable_index_value=True,
    )
    # 1.3.0rc23 added implementation: Literal["triton","msa"] (default
    # "triton"; llm_args.py MiniMaxM3SparseAttentionConfig). The "msa"
    # (fmha_sm100) kernels are the performance path on the SM100 family and
    # hard-require it (sparse/minimax_m3/msa_availability.py:ensure_msa_available
    # raises off SM100/103), so collect "msa" there and the Triton reference
    # elsewhere. kernel_source records which one actually ran. Field-presence
    # probe keeps rc19/rc20 (no such field, pydantic strict) working.
    if "implementation" in getattr(MiniMaxM3SparseAttentionConfig, "model_fields", {}):
        sparse_kwargs["implementation"] = "msa" if get_sm_version() in (100, 103) else "triton"
    sparse_attention_config = MiniMaxM3SparseAttentionConfig(**sparse_kwargs)

    # NOTE: sparse_attention_config is intentionally NOT passed as a
    # from_pretrained kwarg. load_pretrained_config forwards every kwarg into
    # transformers' config load, and PretrainedConfig.from_dict setattr's any
    # kwarg whose name matches an existing config attribute — the M3
    # checkpoint's config.json carries a `sparse_attention_config` dict, so
    # the typed object would clobber the dict the modeling code reads
    # (MiniMaxM3Attention.__init__ sparse_cfg getattr,
    # modeling_minimaxm3.py:623@1.3.0rc20). Set it on the ModelConfig after
    # loading instead; the attention-backend dispatch reads it at layer
    # construction time, which happens later.
    model_config = ModelConfig.from_pretrained(
        model_path,
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
        allreduce_strategy=AllReduceStrategy.AUTO,
        mm_encoder_only=False,
        attn_backend="TRTLLM",
        moe_backend="CUTLASS",
        moe_disable_finalize_fusion=False,
        use_low_precision_moe_combine=False,
        skip_create_weights_in_init=True,
    )

    try:
        model_config.sparse_attention_config = sparse_attention_config
    except AttributeError:
        object.__setattr__(model_config, "sparse_attention_config", sparse_attention_config)

    pretrained_config = model_config.pretrained_config

    # The registry-patched load returns a base PretrainedConfig, which keeps
    # torch_dtype as the JSON string; the typed remote-code config serving
    # loads carries a torch.dtype. Normalize so downstream dtype= kwargs
    # (RMSNorm/Linear construction) see the same object serving sees.
    if isinstance(getattr(pretrained_config, "torch_dtype", None), str):
        pretrained_config.torch_dtype = getattr(torch, pretrained_config.torch_dtype)

    # Single-layer benchmark surgery: one SPARSE layer with a dense MLP so
    # layer.self_attn is the MSA module under test. get_sparse_layer_ids /
    # get_moe_layer_ids validate len(freq) == num_hidden_layers, so the
    # per-layer pattern arrays are truncated consistently. All sparse
    # scalar fields stay checkpoint-verbatim.
    pretrained_config.num_hidden_layers = 1
    sparse_cfg = dict(pretrained_config.sparse_attention_config)
    sparse_cfg["sparse_attention_freq"] = [1]
    if "sparse_disable_index_value" in sparse_cfg:
        flags = sparse_cfg["sparse_disable_index_value"]
        sparse_cfg["sparse_disable_index_value"] = [flags[-1] if isinstance(flags, list) else flags]
    pretrained_config.sparse_attention_config = sparse_cfg
    if getattr(pretrained_config, "moe_layer_freq", None) is not None:
        pretrained_config.moe_layer_freq = [0]

    # TP-shard emulation: shrink the head axes like serving's TP sharding
    # would. KV heads keep the checkpoint's GQA ratio (num_kv_heads =
    # num_heads / (native_q / native_kv)) and floor at 1, matching
    # replication for tp > native_kv. index_q/index_k projections are
    # replicated across TP ranks (modeling_minimaxm3.py:632-660@1.3.0rc20),
    # so the sparse index dims are left untouched.
    #
    # FIXME(kernel-limit): the M3 sparse backend requires num_kv_heads to
    # divide sparse_num_index_heads (=4) — MiniMaxM3SparseConfig
    # validation (sparse/minimax_m3/metadata.py:113 via backend.py:1052
    # @1.3.0rc20). The sweep grid's num_heads=128 rows derive kv=8 and
    # fail fast at construction (hardware-observed on SM90 2026-08-06,
    # smoke 3/3 failures all this class). Serving can never reach kv>4
    # (native 64q/4kv, TP>=1), so these rows stay classified failures per
    # layer_permissions.md (no generation-side filtering). Re-verify on
    # the next framework version bump.
    native_q = int(pretrained_config.num_attention_heads)
    native_kv = int(pretrained_config.num_key_value_heads)
    gqa_ratio = max(1, native_q // native_kv)
    pretrained_config.num_attention_heads = num_heads
    pretrained_config.num_key_value_heads = max(1, num_heads // gqa_ratio)

    _apply_gemm_type_quant(model_config, gemm_type)

    aux_stream = torch.cuda.Stream(device=device)
    aux_stream_dict = {
        AuxStreamType.Attention: aux_stream,
        AuxStreamType.MoeShared: aux_stream,
        AuxStreamType.MoeChunkingOverlap: torch.cuda.Stream(device=device),
    }

    layer = MiniMaxM3DecoderLayer(
        model_config=model_config,
        layer_idx=0,
        aux_stream_dict=aux_stream_dict,
    )

    # Mirror serving's exclude_modules pass (see
    # collect_mla_module.create_attention_layer for the full rationale);
    # M3 combos currently set exclude_modules=None so this is a no-op kept
    # for parity with future quant translations.
    quant_config = model_config.quant_config
    if quant_config is not None and quant_config.exclude_modules is not None:
        excluded_replacement = QuantConfig(kv_cache_quant_algo=quant_config.kv_cache_quant_algo)
        for module_name, module in layer.named_modules():
            if getattr(module, "quant_config", None) is None:
                continue
            if quant_config.is_module_excluded_from_quantization(module_name):
                module.quant_config = excluded_replacement

    for module in layer.modules():
        if callable(getattr(module, "create_weights", None)):
            module.create_weights()
    layer.to(device)

    initialize_dummy_weights(layer)
    for module in layer.modules():
        if hasattr(module, "post_load_weights") and not getattr(module, "_weights_removed", False):
            module.post_load_weights()

    layer.eval()
    layer.requires_grad_(False)

    attn_module = layer.self_attn
    if not getattr(attn_module, "is_sparse_attention_layer", False):
        raise RuntimeError(
            "benchmark layer came out dense; sparse_attention_freq surgery did not take"
        )
    return attn_module, model_config, original_architecture


# ═══════════════════════════════════════════════════════════════════════
# KV Cache + Metadata
# ═══════════════════════════════════════════════════════════════════════


def _resolve_msa_metadata_cls():
    """Return the MSA (fmha_sm100) Metadata class, or None where the framework
    has no MSA backend.

    1.3.0rc23 split the M3 package into triton_backend/triton_metadata (the
    old Triton path) and msa_backend (the fmha_sm100 path); rc19/rc20 ship
    neither module nor the ``implementation`` knob. Import-probe the module
    instead of sniffing the version string.
    """
    try:
        from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_backend import (
            MiniMaxM3MsaSparseAttentionMetadata,
        )
    except ImportError:  # rc19/rc20 layout — Triton contract only
        return None
    return MiniMaxM3MsaSparseAttentionMetadata


def _resolve_attention_backend_cls(model_config: ModelConfig):
    """Resolve the sparse attention backend class exactly like serving.

    1.3.0rc23 lowers the llm_args config into SparseParams first and passes
    ``sparse_params`` (model_engine.py:592-596@1.3.0rc23); the factory then
    routes ``implementation == "msa"`` to MiniMaxM3MsaSparseAttention after
    ensure_msa_available() (sparse/utils.py:45-60@1.3.0rc23) and everything
    else to the Triton reference. rc19/rc20 pass the llm_args config itself
    (model_engine.py:497-499@1.3.0rc20). Probe get_attention_backend's real
    signature so the framework's own dispatch stays the selector on both
    lines.
    """
    if "sparse_params" in inspect.signature(get_attention_backend).parameters:
        sparse_params = model_config.sparse_attention_config.to_sparse_params(
            pretrained_config=model_config.pretrained_config
        )
        return get_attention_backend(model_config.attn_backend, sparse_params=sparse_params)
    return get_attention_backend(
        model_config.attn_backend,
        sparse_attention_config=model_config.sparse_attention_config,
    )


def create_kv_cache_and_metadata(
    model_config: ModelConfig,
    batch_size: int,
    seq_len: int,
    is_context: bool,
    prefix_len: int = 0,
    device: str = "cuda:0",
):
    """Create the M3 KV cache manager and attention metadata via framework
    utilities, mirroring serving's construction.

    Cache manager: get_kv_cache_manager_cls routes sparse-attention models to
    MiniMaxM3KVCacheManagerV2 (pyexecutor/_util.py:75-94@1.3.0rc20;
    :90-166@1.3.0rc23 — at rc23 the manager is shared by the Triton and MSA
    backends, sparse/minimax_m3/__init__.py); kwargs mirror the generic
    non-MLA branch (_util.py:1859-1888@1.3.0rc20; :2079-2105@1.3.0rc23):
    CacheType.SELF, per-layer num_kv_heads list, tokens_per_block 32,
    sparse_attention_config + pretrained_config forwarded. The single
    benchmark layer is declared sparse explicitly (sparse_layer_ids=[0]) —
    the manager's num_layers-based default assumes the full checkpoint's
    "first 3 dense" convention (sparse/minimax_m3/cache_manager.py:162-166
    @1.3.0rc20; :170-176@1.3.0rc23) which does not apply to a one-layer
    build.

    Metadata: the resolved backend's Metadata class builds the sparse runtime
    metadata inside prepare() from the standard AttentionMetadata fields:
    seq_lens carries the current chunk's new-token counts, the cached prefix
    travels via kv_cache_params.num_cached_tokens_per_seq, and kv_lens =
    cached + seq_lens is derived internally — the same current-chunk-only
    prompt_lens semantics as serving (model_engine.py:2986-3017@1.3.0rc20).
    Two framework contracts exist, discriminated by the resolved Metadata
    class (never the version string):

    * Triton (rc19/rc20; rc23 implementation="triton"): prepare() attaches
      the prebuilt runtime metadata + out_cache_loc as the
      ``attn_metadata.minimax_m3`` dict (metadata.py:899-902@1.3.0rc20;
      unchanged at rc23: triton_metadata.py:860-864, kv_lens derivation
      :842-845 both versions), which the model layer's
      _sparse_attention_core reads (modeling_minimaxm3.py:1265-1275
      @1.3.0rc23).
    * MSA / fmha_sm100 (rc23 implementation="msa"): the Metadata subclasses
      TrtllmAttentionMetadata; __post_init__ reads sparse_metadata_params
      and allocates flat CUDA-graph-stable buffers (msa_backend.py:245-249,
      296-348@1.3.0rc23), prepare() fills the cache-write buffers
      (_build_msa_fields, :549-609, setting _msa_fields_ready) and, for a
      pure generation batch, the graph-safe fmha_sm100 decode plans
      (_build_decode_plans, :428-547; prefill/mixed leave the plans None
      and run eagerly). The model layer consumes it via run_indexer +
      the inherited TrtllmAttention forward (modeling_minimaxm3.py:
      1130-1141@1.3.0rc23).

    Serving passes max_num_sequences / num_heads_per_kv /
    sparse_metadata_params when constructing the Metadata on both lines
    (model_engine.py:2474-2487@1.3.0rc23; :1810-1830@1.3.0rc20, where the M3
    config's to_sparse_metadata_params is the base-class None). Those kwargs
    are mirrored here, field-presence-probed to keep the rc19 claim honest.

    Returns (kv_cache_manager, attn_metadata, kernel_source) where
    kernel_source records the resolved backend: "msa_fmha_sm100" for the MSA
    path, "default" for the Triton reference (the label the SM90 rc20 dataset
    already carries).
    """
    config = model_config.pretrained_config
    mapping = model_config.mapping

    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    num_kv_heads = int(config.num_key_value_heads)
    # Serving forces the KV page size per implementation: 128 for "msa"
    # (the fmha_sm100 SparseK2qCsrBuilderSm100 asserts blk_kv == 128), 32
    # for the Triton reference (py_executor_creator.py:427@1.3.0rc23:
    # `tokens_per_block = 128 if m3_sparse_config.implementation == "msa"
    # else 32`; rc19/rc20 have no implementation field and use 32).
    tokens_per_block = (
        128 if getattr(model_config.sparse_attention_config, "implementation", "triton") == "msa" else 32
    )

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
        kv_cache_len = seq_len - 1

    # 2x headroom: KVCacheManagerV2 draws the M3 INDEX_KEY side-cache pages
    # from the same max_tokens page budget as main K/V, and the is_gen dummy
    # path resizes to capacity+1 — an exact budget makes add_dummy_requests
    # fail (return None after releasing resources).
    #
    # NOTE(pool-size-dependent latency, 1.3.0rc23): oversizing is NOT free on
    # the MSA path. select_blocks does a permute().contiguous() copy of the
    # ENTIRE index-K pool every forward (msa_utils.cache_view_to_msa_paged
    # @1.3.0rc23, called from msa_indexer.select_blocks:147), so the measured
    # latency carries a term linear in the pool pages this budget allocates
    # (~35 ns/page measured on B200 2026-08-08; 92% of the gen b=256 kv=65536
    # point). Serving pays the same copy over its own (memory-budget-sized)
    # pool, so this is framework truth in kind but collector-sized in
    # magnitude — keep this factor stable across campaigns for comparability.
    # Upstream main removes the copy (get_index_k_buffer(kv_layout="HND")
    # zero-copy view); re-check on the next version bump.
    kv_cache_config = KvCacheConfig(
        max_tokens=2 * batch_size * (_round_up(max_seq, tokens_per_block) + 2 * tokens_per_block),
        enable_block_reuse=False,
    )
    kv_cache_manager_cls = get_kv_cache_manager_cls(model_config, kv_cache_config)

    kv_cache_manager = kv_cache_manager_cls(
        kv_cache_config,
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=[num_kv_heads],
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=torch_dtype_to_binding(torch.bfloat16),
        sparse_attention_config=model_config.sparse_attention_config,
        pretrained_config=config,
        sparse_layer_ids=[0],
        disable_index_value_layer_ids=[0],
        sparse_index_dim=model_config.sparse_attention_config.sparse_index_dim,
    )

    request_ids = list(range(batch_size))
    # token_nums = past_kv_len + input_len (kv_cache_manager_v2.py
    # add_dummy_requests docstring); is_gen marks decode requests so the
    # committed-history hint matches a decode step's cache state.
    token_nums = [prefix_len + seq_len_q] * batch_size if is_context else [seq_len] * batch_size
    dummy_result = kv_cache_manager.add_dummy_requests(token_nums=token_nums, request_ids=request_ids, is_gen=not is_context)
    if dummy_result is None:
        raise RuntimeError(
            f"KVCacheManagerV2.add_dummy_requests failed (returned None) for "
            f"b={batch_size}, max_seq={max_seq}: pool budget too small or "
            f"resource allocation failed"
        )

    attention_cls = _resolve_attention_backend_cls(model_config)
    metadata_cls = attention_cls.Metadata
    msa_metadata_cls = _resolve_msa_metadata_cls()
    is_msa = msa_metadata_cls is not None and issubclass(metadata_cls, msa_metadata_cls)

    # Serving-parity Metadata kwargs (model_engine.py:2474-2487@1.3.0rc23;
    # :1810-1830@1.3.0rc20), field-presence-probed for rc19:
    #   max_num_sequences = batch_size * max_beam_width, beam width 1 here;
    #   num_heads_per_kv  = GQA ratio (model_engine.py:2428-2440@1.3.0rc23) —
    #     computed from the (TP-shard-emulated) pretrained_config like serving
    #     computes it from its config;
    #   sparse_metadata_params carries the MSA sparse geometry the rc23 MSA
    #     metadata's __post_init__ reads (msa_backend.py:245-249); the rc20 M3
    #     config inherits the base to_sparse_metadata_params -> None
    #     (llm_args.py:570-572@1.3.0rc20), matching the field default.
    init_field_names = {f.name for f in dataclasses.fields(metadata_cls) if f.init}
    serving_metadata_kwargs = {}
    if "max_num_sequences" in init_field_names:
        serving_metadata_kwargs["max_num_sequences"] = batch_size
    if "num_heads_per_kv" in init_field_names:
        num_q_heads = int(getattr(config, "num_attention_heads", 0) or 0)
        serving_metadata_kwargs["num_heads_per_kv"] = (
            num_q_heads // num_kv_heads if num_q_heads and num_kv_heads else 1
        )
    if "sparse_metadata_params" in init_field_names:
        serving_metadata_kwargs["sparse_metadata_params"] = (
            model_config.sparse_attention_config.to_sparse_metadata_params(
                pretrained_config=config
            )
        )

    attn_metadata = metadata_cls(
        max_num_requests=batch_size,
        max_num_tokens=total_tokens,
        kv_cache_manager=kv_cache_manager,
        mapping=mapping,
        seq_lens=torch.tensor([seq_len_q] * batch_size, dtype=torch.int32),
        position_ids=None,
        num_contexts=batch_size if is_context else 0,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[kv_cache_len] * batch_size,
        ),
        cross=None,
        request_ids=request_ids,
        # Current-chunk token count only — the cached prefix travels via
        # num_cached_tokens_per_seq (model_engine.py:2986-3017@1.3.0rc20,
        # same as collect_mla_module).
        prompt_lens=[seq_len_q if is_context else kv_cache_len] * batch_size,
        runtime_features=AttentionRuntimeFeatures(
            chunked_prefill=False,
            cache_reuse=bool(is_context and prefix_len > 0),
        ),
        all_rank_num_tokens=None,
        **serving_metadata_kwargs,
    )

    attn_metadata.prepare()
    if is_msa:
        # MSA contract: prepare() must have populated the flat cache-write
        # buffers (msa_backend.py:549-609@1.3.0rc23 sets _msa_fields_ready on
        # success) and, for a pure generation batch, the graph-safe decode
        # plans (:442-460 — absent plans mean sparse_metadata_params never
        # reached the metadata and decode would silently take the eager
        # prefill path instead of serving's planned one).
        if not getattr(attn_metadata, "_msa_fields_ready", False):
            raise RuntimeError(
                "MiniMaxM3MsaSparseAttentionMetadata.prepare() did not build "
                "the MSA cache-write buffers (msa_backend.py:549-609@1.3.0rc23); "
                "the KV cache manager is not the M3 sparse manager"
            )
        if not is_context and attn_metadata.msa_decode_proxy_plan is None:
            raise RuntimeError(
                "MSA decode plans were not built for a pure generation batch "
                "(msa_backend.py:428-460@1.3.0rc23); sparse_metadata_params "
                "did not reach the metadata"
            )
        kernel_source = "msa_fmha_sm100"
    else:
        # Triton contract (rc19/rc20; rc23 implementation="triton"):
        # prepare() attaches the prebuilt runtime metadata dict
        # (metadata.py:899-902@1.3.0rc20; triton_metadata.py:860-864@1.3.0rc23).
        if getattr(attn_metadata, "minimax_m3", None) is None:
            raise RuntimeError(
                "MiniMaxM3AttentionMetadata.prepare() did not build the minimax_m3 "
                "attachment; the KV cache manager is not the M3 sparse manager"
            )
        # Keep the label the SM90 rc20 dataset already carries for this path.
        kernel_source = "default"

    return kv_cache_manager, attn_metadata, kernel_source


# ═══════════════════════════════════════════════════════════════════════
# Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════


def run_msa_module(
    seq_len: int,
    batch_size: int,
    num_heads: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    perf_filename: str,
    prefix_len: int = 0,
    *,
    model_path: str,
    device: str = "cuda:0",
    warming_up: int = 10,
    test_ite: int = 6,
):
    """Run a single MiniMax-M3 MSA module-level benchmark point."""
    torch.cuda.set_device(device)
    torch_device = torch.device(device)

    if kv_cache_dtype != "bfloat16" or compute_dtype != "bfloat16":
        raise ValueError(
            f"M3 sparse attention at {tensorrt_llm.__version__} is bf16-only "
            f"(see _get_precision_combos); got compute={compute_dtype}, kv={kv_cache_dtype}"
        )

    is_context = "context" in perf_filename
    prefix_len = int(prefix_len) if is_context else 0
    phase = "context" if is_context else "generation"
    print(
        f"\n[MSA module] {phase} b={batch_size}, s={seq_len}, "
        f"prefix={prefix_len}, heads={num_heads}, gemm={gemm_type}, "
        f"compute={compute_dtype}, kv={kv_cache_dtype}, model={model_path}"
    )

    attn_module, model_config, original_architecture = create_msa_attention_layer(
        model_path=model_path,
        num_heads=num_heads,
        gemm_type=gemm_type,
        device=device,
    )

    kv_cache_manager, attn_metadata, kernel_source = create_kv_cache_and_metadata(
        model_config=model_config,
        batch_size=batch_size,
        seq_len=seq_len,
        is_context=is_context,
        prefix_len=prefix_len,
        device=device,
    )

    # Cross-check: the metadata contract validated above must match the
    # backend instance the layer actually dispatches to — the model layer
    # branches on isinstance(self.attn, MiniMaxM3MsaSparseAttention)
    # (modeling_minimaxm3.py:1130-1147@1.3.0rc23). A mismatch would benchmark
    # one path against the other's metadata and mislabel kernel_source.
    layer_backend_name = type(getattr(attn_module, "attn", None)).__name__
    layer_is_msa = layer_backend_name == "MiniMaxM3MsaSparseAttention"
    if layer_is_msa != (kernel_source == "msa_fmha_sm100"):
        _cleanup(kv_cache_manager)
        raise RuntimeError(
            f"metadata contract ({kernel_source}) does not match the layer's "
            f"attention backend ({layer_backend_name})"
        )

    hidden_size = model_config.pretrained_config.hidden_size
    if is_context:
        num_tokens = seq_len * batch_size
        position_ids = (
            torch.arange(prefix_len, prefix_len + seq_len, device=torch_device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .reshape(-1)
            .contiguous()
        )
    else:
        num_tokens = batch_size
        position_ids = torch.full(
            (batch_size,),
            seq_len - 1,
            device=torch_device,
            dtype=torch.long,
        )

    hidden_states = torch.randn(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=torch_device,
    )

    import tensorrt_llm._torch.utils as _trtllm_utils

    _prev_extra_attrs = getattr(_trtllm_utils._model_extra_attrs, "attrs", None)
    try:
        with model_extra_attrs(model_config.extra_attrs):
            get_model_extra_attrs()["attention_metadata"] = weakref.ref(attn_metadata)
            try:
                with torch.inference_mode():
                    attn_module.forward(position_ids, hidden_states, attn_metadata)
            except Exception:
                print("  Dry run failed:")
                traceback.print_exc()
                raise

        _trtllm_utils._model_extra_attrs.attrs = model_config.extra_attrs
        _trtllm_utils._model_extra_attrs.attrs["attention_metadata"] = weakref.ref(attn_metadata)

        def kernel_func():
            attn_module.forward(position_ids, hidden_states, attn_metadata)

        # Measurement mode mirrors serving's execution mode per backend/phase.
        # MSA prefill runs eagerly in serving — decode plans are cleared for
        # prefill/mixed batches (msa_backend.py:434-436@1.3.0rc23) and the
        # indexer then plans fmha_sm100 inline with host-side work
        # (msa_indexer.py:149-191@1.3.0rc23), which is CUDA-graph-capture
        # unsafe — so MSA context is measured eagerly. MSA decode replays the
        # prebuilt graph-stable plans (built for capture) and keeps graph-mode
        # measurement, as does the Triton path for both phases (unchanged from
        # the SM90 rc20 collection).
        use_cuda_graph = not (kernel_source == "msa_fmha_sm100" and is_context)

        with benchmark_with_power(
            device=torch_device,
            kernel_func=kernel_func,
            num_warmups=warming_up,
            num_runs=test_ite,
            repeat_n=1,
            allow_graph_fail=False,
            use_cuda_graph=use_cuda_graph,
        ) as results:
            pass

        latency = results["latency_ms"]

        if is_context:
            isl = seq_len
            step = prefix_len
        else:
            isl = 1
            step = seq_len

        op_name = f"msa_{phase}_module"

        if not log_perf(
            item_list=[
                {
                    "model": model_path,
                    "architecture": original_architecture,
                    "mla_dtype": compute_dtype,
                    "kv_cache_dtype": kv_cache_dtype,
                    "gemm_type": gemm_type,
                    "num_heads": num_heads,
                    "batch_size": batch_size,
                    "isl": isl,
                    "tp_size": 1,
                    "step": step,
                    "latency": f"{latency:.4f}",
                }
            ],
            framework="TRTLLM",
            version=tensorrt_llm.__version__,
            device_name=torch.cuda.get_device_name(device),
            op_name=op_name,
            kernel_source=kernel_source,
            perf_filename=perf_filename,
            power_stats=results["power_stats"],
        ):
            # A failed write with a normal return would report the case as
            # successful while silently shrinking the dataset.
            raise RuntimeError(f"failed to persist MSA row to {perf_filename}")

        print(
            f"  [{phase}] b={batch_size}, s={seq_len}, heads={num_heads}, "
            f"prefix={prefix_len}, gemm={gemm_type}, backend={kernel_source}: "
            f"{latency:.4f} ms"
        )

        return latency
    finally:
        # Restore the process-global extra-attrs slot and always release the
        # KV pool: the CLI loop continues past a failed case, so a leaked
        # pool would surface as spurious OOM in later cases, and a stale
        # weakref here would point at the shut-down manager.
        if _prev_extra_attrs is not None:
            _trtllm_utils._model_extra_attrs.attrs = _prev_extra_attrs
        elif hasattr(_trtllm_utils._model_extra_attrs, "attrs"):
            del _trtllm_utils._model_extra_attrs.attrs
        _cleanup(kv_cache_manager)


def run_msa_module_worker(
    seq_len: int,
    batch_size: int,
    num_heads: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    model_path: str,
    prefix_len: int = 0,
    *,
    perf_filename: str,
    device: str = "cuda:0",
):
    """Worker-compatible positional wrapper used by collector/collect.py."""
    return run_msa_module(
        seq_len=seq_len,
        batch_size=batch_size,
        num_heads=num_heads,
        kv_cache_dtype=kv_cache_dtype,
        compute_dtype=compute_dtype,
        gemm_type=gemm_type,
        prefix_len=prefix_len,
        perf_filename=perf_filename,
        model_path=model_path,
        device=device,
    )


def _cleanup(kv_cache_manager):
    if kv_cache_manager is not None:
        kv_cache_manager.shutdown()
    torch.cuda.empty_cache()
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def _perf_file_for(mode: str) -> str:
    return PerfFile.MSA_CONTEXT_MODULE if mode == "context" else PerfFile.MSA_GENERATION_MODULE


def main():
    all_model_specs = get_mla_module_model_specs(attention_type="msa", apply_model_filter=False)
    model_names = [spec.model_path for spec in all_model_specs]

    parser = argparse.ArgumentParser(description="MiniMax-M3 MSA module-level collector for TRT-LLM")
    parser.add_argument("--mode", choices=["context", "generation"], required=True)
    parser.add_argument("--model", type=str, default=None, choices=model_names)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--prefix-len", type=int, default=0)
    parser.add_argument("--gemm-type", type=str, choices=["bfloat16", "fp8_block", "nvfp4"], default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    model_specs_to_run = (
        [spec for spec in all_model_specs if spec.model_path == args.model] if args.model else all_model_specs
    )

    for model_spec in model_specs_to_run:
        model_path = model_spec.model_path
        print(f"\n{'=' * 60}\nModel: {model_path}  |  Attention: MSA\n{'=' * 60}")

        perf_filename = _perf_file_for(args.mode)

        if args.quick:
            run_msa_module(
                seq_len=args.seq_len or 2048,
                batch_size=args.batch_size or 4,
                num_heads=args.num_heads or 64,
                kv_cache_dtype="bfloat16",
                compute_dtype="bfloat16",
                gemm_type=args.gemm_type or "bfloat16",
                prefix_len=args.prefix_len,
                perf_filename=perf_filename,
                model_path=model_path,
                device=args.device,
            )
            continue

        test_cases = get_context_test_cases() if args.mode == "context" else get_generation_test_cases()
        if args.num_heads is not None:
            test_cases = [tc for tc in test_cases if tc[2] == args.num_heads]
        if args.gemm_type is not None:
            test_cases = [tc for tc in test_cases if tc[5] == args.gemm_type]

        print(f"Running {len(test_cases)} {args.mode} MSA module test cases...")
        num_failed = 0
        for i, tc in enumerate(test_cases):
            s, b, h, kv_dtype, compute, gemm, *rest = tc
            print(f"[{i + 1}/{len(test_cases)}]", end="")
            try:
                run_msa_module(
                    seq_len=s,
                    batch_size=b,
                    num_heads=h,
                    kv_cache_dtype=kv_dtype,
                    compute_dtype=compute,
                    gemm_type=gemm,
                    prefix_len=rest[0] if rest else 0,
                    perf_filename=perf_filename,
                    model_path=model_path,
                    device=args.device,
                )
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM: b={b}, s={s}, heads={h}, gemm={gemm}")
                num_failed += 1
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"  FAILED: b={b}, s={s}, heads={h}, gemm={gemm}: {e}")
                traceback.print_exc()
                num_failed += 1
                torch.cuda.empty_cache()
                gc.collect()
        if num_failed:
            # Exit non-zero so an operator (and CI) can see failures without
            # scraping the log; per-case detail is printed above.
            print(f"{num_failed}/{len(test_cases)} cases failed")
            sys.exit(1)


if __name__ == "__main__":
    main()
