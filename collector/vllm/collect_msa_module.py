# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# MiniMax-M3 support landed in vLLM 0.24.0 (vllm/models/minimax_m3/ +
# MinimaxM3QKVParallelLinearWithIndexer + fused_minimax_m3_qknorm_rope_kv_insert);
# this collector follows the 0.24.0 APIs and is pinned to the manifest
# default runtime (vllm collectors pin exactly; see
# test_active_cuda_vllm_collectors_are_exactly_pinned_to_manifest_version).
__compat__ = "vllm==0.24.0"

"""
MSA Module Collector for vLLM — MiniMax-M3 sparse-attention benchmarking.

Profiles the complete MiniMax-M3 sparse-attention module forward pass
(fused qkv+index projection, fused per-head Gemma QK norm + partial RoPE +
paged KV/index-cache insert, lightning-indexer top-k block selection,
block-sparse GQA attend, output projection), not a bare kernel. Uses vLLM's
own modeling code (``MiniMaxM3SparseAttention``, the exact module
``MiniMaxM3DecoderLayer`` builds for sparse layers —
vllm/models/minimax_m3/nvidia/model.py:671-679@v0.24.0) with dummy weights,
the same framework-builder approach as ``collect_mla_module.py``.

Kernel dispatch is the framework's own: the attend impl is chosen by
``select_main_impl_cls`` (Triton on non-SM100, fmha_sm100 "MSA" attend on the
SM100 family — common/sparse_attention.py:391-422@v0.24.0) and the indexer
impl by ``select_indexer_impl_cls`` (common/indexer.py:457-507@v0.24.0), both
at module construction. ``kernel_source`` records the actually-selected impl
class. Hardware-validated on SM90; other SMs carry registry
``unverified_sms`` markers until validated.

Supported models and micro-sweeps come from collector v2 YAML
(``cases/models/MiniMaxM3ForCausalLM_cases.yaml`` `mla_module` rows with
``attention_type: msa`` + ``cases/base_ops/mla_module.yaml`` vllm override).

Usage:
    python collect_msa_module.py --mode context --model MiniMaxAI/MiniMax-M3
    python collect_msa_module.py --mode generation --quick --batch-size 4 --seq-len 2048 --num-heads 64
    python collect_msa_module.py --mode context --quick --kv-cache-dtype fp8
"""

import argparse
import gc
import json
import os
import sys
import traceback

import torch
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context

# ═══════════════════════════════════════════════════════════════════════
# Config registry shim — the MiniMax-M3 HF release config carries
# model_type "minimax_m3", but vLLM 0.24.0's _CONFIG_REGISTRY only maps
# "minimax_m3_vl" / "minimax_m3_mtp" (transformers_utils/config.py:106-107
# @v0.24.0) and transformers doesn't know the model_type either, so the
# AutoConfig fallback fails offline. Serving loads such checkpoints via
# trust_remote_code; the bundled config dir has auto_map stripped (see
# helper._resolve_local_model_path), so route the model_type to vLLM's own
# text-backbone config class ``MiniMaxM3TextConfig``
# (transformers_utils/configs/minimax_m3.py:8-105@v0.24.0), whose fields
# 1:1 mirror the checkpoint config (every bundled-config key is a named
# __init__ arg or **kwargs passthrough). This changes HOW the config
# object is constructed, never which kernel runs — sparse dispatch keys on
# sparse_attention_config / head geometry, preserved verbatim from JSON.
# ═══════════════════════════════════════════════════════════════════════
from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm.v1.worker.workspace import init_workspace_manager
from vllm.version import __version__ as vllm_version

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from collector.case_generator import get_mla_module_model_specs, get_mla_module_sweep_spec
from collector.helper import _resolve_local_model_path, benchmark_with_power, get_sm_version, log_perf
from collector.registry_types import PerfFile
from collector.vllm.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
    enable_engine_fused_ops,
    setup_distributed,
    with_exit_stack,
)

if "minimax_m3" not in _CONFIG_REGISTRY:
    _CONFIG_REGISTRY["minimax_m3"] = "MiniMaxM3TextConfig"


# ═══════════════════════════════════════════════════════════════════════
# Test Cases — aligned with TRT-LLM's collect_msa_module.py
# ═══════════════════════════════════════════════════════════════════════


def _get_precision_combos(phase: str):
    """Return (compute_dtype, kv_cache_dtype, gemm_type) triples for MSA.

    Precision axes:
      gemm_type    — linear-layer GEMMs (fused qkv+index projection + o_proj)
        bfloat16:  always (the only declared M3 artifact is BF16)
        fp8_block: SM >= 89 — vLLM's blockwise-FP8 checkpoint path
                   (Fp8Config → Fp8LinearMethod block_quant → DeepGEMM /
                   W8A8BlockFp8LinearOp); the M3 projections take the
                   generic quant path (MinimaxM3QKVParallelLinearWithIndexer
                   forwards quant_config into ColumnParallelLinear,
                   linear.py:1379-1389@v0.24.0; o_proj is a plain
                   RowParallelLinear, nvidia/model.py:449-456)
        nvfp4:     SM >= 100 — ModelOpt NVFP4 checkpoint path, same generic
                   linear quant dispatch

      (compute_dtype, kv_cache_dtype) — attention compute + KV cache
        bf16/bf16: the BF16 checkpoint's default serving config
                   (cache_dtype "auto" resolves to the model dtype).
        bf16/fp8:  serving's global ``--kv-cache-dtype fp8`` (vLLM
                   CacheDType "fp8" = fp8_e4m3 on CUDA, config/cache.py
                   :75-78@v0.24.0), declared on every SM this collector
                   runs on: the M3 sparse backend accepts fp8 KV on all
                   platforms (supported_kv_cache_dtypes, common/
                   sparse_attention.py:56-62@v0.24.0 — "bf16 or fp8
                   (e4m3/e5m2): the Triton kernels dequant fp8 before the
                   dots"). Off the SM100 family the Triton attend
                   reinterprets the cache as the platform fp8 dtype and
                   dequantizes in-kernel (:298-306, view at :352); on the
                   SM100 family select_main_impl_cls excludes only
                   "fp8_e5m2", so "fp8" (e4m3) rides the MSA attend
                   (:391-422, gate :407; the MSA impl shares the base
                   impl's fp8 view, nvidia/sparse_attention_msa.py:49).
                   The indexer's K side-cache is governed by the
                   independent attention_config.indexer_kv_dtype knob and
                   stays at its serving default "bf16" (config/attention
                   .py:55; nvidia/model.py:490-491) — an fp8 indexer
                   cache is a non-default opt-in, not collected here.
    """
    sm = get_sm_version()

    gemm_types = ["bfloat16"]
    if sm >= 89:
        gemm_types.append("fp8_block")
    if sm >= 100:
        gemm_types.append("nvfp4")

    attn_combos = [("bfloat16", "bfloat16"), ("bfloat16", "fp8")]
    return [(c, kv, g) for g in gemm_types for c, kv in attn_combos]


# Native MiniMax-M3 GQA ratio: 64 q heads / 4 kv heads (bundled config
# num_attention_heads / num_key_value_heads). Used by the TP-shard emulation
# and by the generation-time memory-feasibility filter below.
_NATIVE_GQA_RATIO = 16
_M3_HEAD_DIM = 128
_M3_INDEX_DIM = 128

_MEMORY_BUDGET_SAFETY_FACTOR = 0.9


def _emulated_kv_heads(num_heads: int) -> int:
    return max(1, num_heads // _NATIVE_GQA_RATIO)


def _device_total_memory_bytes():
    """Live device memory for the generation-time memory-feasibility filter."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory
    except Exception:
        pass
    return None


def _generation_kv_footprint_bytes(total_tokens: int, num_heads: int, kv_cache_dtype: str) -> int:
    """Lower bound of a generation case's peak device footprint.

    run_msa_module allocates the paged main K/V cache (2 x kv_heads x 128
    elems/token; 2 B/elem for bf16, 1 B/elem for the fp8 cache's uint8
    storage — STR_DTYPE_TO_TORCH_DTYPE["fp8"], utils/torch_utils.py:38
    @v0.24.0) plus the indexer's key side-cache (128 elems/token, always
    bf16). Module weights, metadata, and the top-k buffer are deliberately
    excluded so the estimate stays a provable lower bound: a case this
    filter drops cannot fit on the device, on any platform.
    """
    kv_heads = _emulated_kv_heads(num_heads)
    main_bytes_per_elem = 1 if kv_cache_dtype == "fp8" else 2
    entry_bytes = 2 * kv_heads * _M3_HEAD_DIM * main_bytes_per_elem + _M3_INDEX_DIM * 2
    return total_tokens * entry_bytes


def get_context_test_cases():
    """Context-phase test cases.

    Returns list of [seq_len, batch_size, num_heads, kv_cache_dtype,
                     compute_dtype, gemm_type, prefix_len].
    """
    cases = []
    sweep = get_mla_module_sweep_spec("vllm")
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

    Applies the one sanctioned in-collector filter (layer_permissions.md):
    a generation-time memory-feasibility drop, size vs live device capacity
    only, counted and logged.
    """
    cases = []
    sweep = get_mla_module_sweep_spec("vllm")
    total_memory = _device_total_memory_bytes()
    budget = None if total_memory is None else int(total_memory * _MEMORY_BUDGET_SAFETY_FACTOR)
    considered = 0
    dropped = 0
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
                    considered += 1
                    if budget is not None and _generation_kv_footprint_bytes(b * s, num_heads, kv_dtype) > budget:
                        dropped += 1
                        continue
                    cases.append([s, b, num_heads, kv_dtype, compute_dtype, gemm_type])
    if dropped:
        print(
            f"msa_generation_module: dropped {dropped}/{considered} cases "
            f"(memory budget, device={total_memory / 2**30:.0f}GiB)"
        )
    return cases


def _build_module_test_cases(mode: str):
    """Build module-level test cases for one phase.

    Output test case format is positional args for run_msa_module_worker:
    [seq_len, batch_size, num_heads, kv_cache_dtype, compute_dtype, gemm_type,
     model_path (, prefix_len)]
    """
    base_cases = get_context_test_cases() if mode == "context" else get_generation_test_cases()
    model_specs = get_mla_module_model_specs(attention_type="msa", backend="vllm")
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
# Module Construction
# ═══════════════════════════════════════════════════════════════════════


def _ceil_div(a, b):
    return (a + b - 1) // b


def _create_gemm_quant_config(gemm_type: str):
    """Create the vLLM QuantizationConfig for a given gemm_type.

    Returns None for bfloat16 (unquantised GEMMs). For fp8_block / nvfp4,
    returns a serialized-checkpoint config so dummy weights are created in
    the quantized layout and processed by ``process_weights_after_loading``
    — same rationale and citations as collect_mla_module.py.
    """
    if gemm_type == "bfloat16":
        return None
    if gemm_type == "fp8_block":
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config

        # vLLM requires is_checkpoint_fp8_serialized=True for block-scaled
        # FP8 (fp8.py raises ValueError otherwise). This routes through
        # Fp8LinearMethod (block_quant=True) → W8A8BlockFp8LinearOp →
        # DeepGEMM on SM>=89.
        return Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
    if gemm_type == "nvfp4":
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4Config,
        )

        return ModelOptNvFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        )
    raise ValueError(f"Unknown gemm_type: {gemm_type!r}")


_ATTN_LAYER_NAME = "model.layers.0.self_attn.attn"
_INDEX_CACHE_LAYER_NAME = "model.layers.0.self_attn.attn.index_cache"


def _create_msa_attention_module(
    model_path: str,
    num_heads: int,
    gemm_type: str,
    kv_cache_dtype: str,
    max_seq_len: int,
    max_batch_size: int,
    is_context: bool,
    device: str = "cuda:0",
):
    """Create a ``MiniMaxM3SparseAttention`` module from vLLM's own modeling
    code.

    Loads the real HF config from model_path, applies the TP-shard head
    emulation in-memory, and constructs the exact module serving builds for
    sparse layers (``MiniMaxM3DecoderLayer.__init__``,
    nvidia/model.py:671-679@v0.24.0). Backend selection happens inside the
    module constructor exactly as in serving (select_main_impl_cls /
    select_indexer_impl_cls); ``kv_cache_dtype`` must therefore be baked
    into cache_config before construction — "fp8" here models serving's
    global ``--kv-cache-dtype fp8``.
    """
    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        # This collector benchmarks the NVIDIA path (vllm/models/minimax_m3/
        # __init__.py:16-27@v0.24.0 routes ROCm to amd/model.py, a different
        # module with different kernels).
        raise RuntimeError("collect_msa_module benchmarks the NVIDIA MiniMax-M3 path; ROCm is unsupported")

    local_model_path = _resolve_local_model_path(model_path)

    with open(os.path.join(local_model_path, "config.json")) as f:
        cfg_dict = json.load(f)
    original_architecture = cfg_dict.get("architectures", [cfg_dict.get("model_type", "unknown")])[0]
    if not cfg_dict.get("sparse_attention_config"):
        raise ValueError(
            f"model {model_path} has no sparse_attention_config; the MSA module "
            "collector only supports MiniMax-M3 sparse-attention checkpoints"
        )

    native_q = int(cfg_dict["num_attention_heads"])
    native_kv = int(cfg_dict["num_key_value_heads"])
    gqa_ratio = max(1, native_q // native_kv)
    num_kv_heads = max(1, num_heads // gqa_ratio)

    # Page size == sparse block size: the M3 backend supports exactly one
    # kernel block size, 128 (MiniMaxM3SparseBackend
    # .get_supported_kernel_block_sizes, common/sparse_attention.py:81-84
    # @v0.24.0), so serving's cache block size resolves to 128.
    block_size = 128
    max_model_len = max(max_seq_len, 4096)

    vllm_config = create_vllm_config(
        model_name=local_model_path,
        max_model_len=max_model_len,
        block_size=block_size,
        num_gpu_blocks=1 + _ceil_div(max_seq_len + 1, block_size) * max_batch_size,
        max_num_seqs=max_batch_size,
        max_num_batched_tokens=max(max_batch_size * max_seq_len, 131072) if is_context else max_batch_size,
        # Serving's global --kv-cache-dtype: use_fp8_kv_cache=True sets
        # cache_config.cache_dtype "fp8" (= fp8_e4m3 on CUDA, config/cache.py
        # :75-78@v0.24.0), False leaves "auto" (model dtype, bf16). The module
        # reads it at construction (self.kv_cache_dtype =
        # cache_config.cache_dtype, nvidia/model.py:483-488@v0.24.0) and
        # selects the attend impl off it (select_main_impl_cls(
        # kv_cache_dtype=...), nvidia/model.py:501-513), so the dtype must be
        # set before the module is built for dispatch to match serving.
        use_fp8_kv_cache=(kv_cache_dtype == "fp8"),
        trust_remote_code=False,
        # The HF release config declares architectures
        # ["MiniMaxM3ForCausalLM"], which vLLM 0.24.0's model registry does
        # not know; the registered text-backbone architecture is
        # "MiniMaxM3SparseForCausalLM" (model_executor/models/registry.py
        # :163-166@v0.24.0) — the same class vLLM's own VL wrapper routes
        # the text backbone through (nvidia/model.py:1039-1044). Map it via
        # hf_overrides, vLLM's first-class --hf-overrides mechanism
        # (config/model.py:496-541 applies it before arch resolution).
        hf_overrides={"architectures": ["MiniMaxM3SparseForCausalLM"]},
        # TP-shard emulation: shrink the head axes exactly like serving's TP
        # sharding would. Per-rank q heads = total/tp and kv heads =
        # max(1, total_kv/tp) (MiniMaxM3SparseAttention.__init__,
        # nvidia/model.py:411-419@v0.24.0), so with tp_size=1 here we bake
        # the sharded counts into the config.
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )

    # Override quant_config to control linear-layer GEMM precision (the
    # BF16 checkpoint ships no quant config, so None means bf16 GEMMs).
    vllm_config.quant_config = _create_gemm_quant_config(gemm_type)

    hf_config = vllm_config.model_config.hf_text_config

    # Index-head TP emulation: index_q has the same head count as the KV
    # heads and shards identically — including replication when
    # tp > total_kv (num_idx_heads = num_kv_heads, nvidia/model.py:427-431;
    # MinimaxM3QKVParallelLinearWithIndexer asserts total_num_index_heads ==
    # total_num_kv_heads and rides the KV-head sharding/replication path,
    # linear.py:1317-1345@v0.24.0; index_k is a single head replicated to
    # every rank). Natively sparse_num_index_heads (4) == num_key_value_heads
    # (4), so the per-rank emulated counts are reproduced by setting the
    # config's index-head count to the emulated kv-head count. The indexer
    # metadata builder derives the same per-rank count from this field
    # (common/indexer.py:229-239), and the shared top-k buffer as well
    # (nvidia/model.py:769-781).
    sparse_cfg = dict(hf_config.sparse_attention_config)
    sparse_cfg["sparse_num_index_heads"] = num_kv_heads
    hf_config.sparse_attention_config = sparse_cfg

    # Reserved top-k indices buffer shared by the indexer and the attend —
    # mirrors MiniMaxM3Model.__init__ (nvidia/model.py:769-781@v0.24.0):
    # shape [num_index_heads, tokens padded to a multiple of 4 for
    # build_k2q_csr's int4 loads, sparse_topk_blocks], int32.
    max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    padded_num_tokens = (max_num_batched_tokens + 3) // 4 * 4
    topk_indices_buffer = torch.empty(
        num_kv_heads,
        padded_num_tokens,
        sparse_cfg["sparse_topk_blocks"],
        dtype=torch.int32,
        device=device,
    )

    # Build the module inside set_current_vllm_config():
    # MiniMaxM3SparseAttention.__init__ reads the current vllm config for
    # attention_config.indexer_kv_dtype and registers itself (and its
    # indexer side cache) in compilation_config.static_forward_context
    # (nvidia/model.py:481-534, common/indexer.py:145-148@v0.24.0).
    from vllm.models.minimax_m3.nvidia.model import MiniMaxM3SparseAttention
    from vllm.utils.torch_utils import set_default_torch_dtype

    with set_current_vllm_config(vllm_config), set_default_torch_dtype(vllm_config.model_config.dtype):
        attn_module = MiniMaxM3SparseAttention(
            config=hf_config,
            layer_id=0,
            quant_config=vllm_config.quant_config,
            prefix="model.layers.0.self_attn",
            cache_config=vllm_config.cache_config,
            topk_indices_buffer=topk_indices_buffer,
        )

    # Serialized quant configs create weight params on meta device; to()
    # cannot copy meta tensors, so use to_empty() when needed.
    if any(p.is_meta for p in attn_module.parameters()):
        attn_module = attn_module.to_empty(device=torch.device(device))
    else:
        attn_module = attn_module.to(device)
    attn_module.eval()
    attn_module.requires_grad_(False)

    # Deterministic dummy weights (fill_, not RNG — same precedent as
    # collect_mla_module.py): fp8/uint8 → 0, fp32 scales → 0.5, rest 0.01.
    # Kernel latency depends on shapes/dtypes, not values.
    with torch.no_grad():
        for name, tensor in list(attn_module.named_parameters()) + list(attn_module.named_buffers()):
            if tensor.is_meta:
                continue
            if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8):
                tensor.data.zero_()
            elif tensor.dtype == torch.float32 and "scale" in name:
                tensor.data.fill_(0.5)
            else:
                tensor.data.fill_(0.01)

    # Process weights, mimicking vLLM's model loader (quantized-layout
    # conversion; no-op for unquantized linears).
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

    with set_current_vllm_config(vllm_config):
        for _, module in attn_module.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if isinstance(quant_method, QuantizeMethodBase):
                quant_method.process_weights_after_loading(module)

    return attn_module, vllm_config, original_architecture


# ═══════════════════════════════════════════════════════════════════════
# KV Cache + Metadata
# ═══════════════════════════════════════════════════════════════════════


def _rebase_block_table_and_slots(common_attn_metadata, block_size: int):
    """Reserve block 0 as the null block and derive the slot mapping from the
    block table.

    Serving computes each group's slot mapping from that group's block table
    as ``block_id * block_size + intra_block_offset`` for the tokens being
    written (BlockTable.compute_slot_mapping, called from _prepare_inputs —
    gpu_model_runner.py:2126@v0.24.0), with block id 0 reserved as the null
    block. create_common_attn_metadata's arange path starts at block 0 and
    fills slot_mapping with a plain arange, so rebase both here.
    """
    block_table = common_attn_metadata.block_table_tensor
    block_table += 1

    query_start_loc = common_attn_metadata.query_start_loc_cpu
    context_lens = common_attn_metadata.num_computed_tokens_cpu
    slot_mapping = common_attn_metadata.slot_mapping
    device = slot_mapping.device
    for i in range(common_attn_metadata.num_reqs):
        start = int(query_start_loc[i])
        end = int(query_start_loc[i + 1])
        token_offsets = torch.arange(end - start, dtype=torch.long, device=device) + int(context_lens[i])
        block_ids = block_table[i, token_offsets // block_size].to(torch.long)
        slot_mapping[start:end] = block_ids * block_size + token_offsets % block_size
    return int(block_table.max().item()) + 1  # blocks needed incl. null block


def _create_kv_caches_and_metadata(
    vllm_config,
    attn_module,
    batch_size: int,
    seq_len: int,
    is_context: bool,
    prefix_len: int = 0,
    device: str = "cuda:0",
):
    """Create the main + indexer KV caches and both metadata objects via the
    framework's own specs and metadata builders, mirroring serving.

    Metadata: serving builds one metadata per KV-cache group via the layer's
    backend builder over CommonAttentionMetadata and hands the model a
    ``{layer_name: metadata}`` dict plus a per-layer slot-mapping dict
    (gpu_model_runner._get_slot_mappings, :3972-4045, passed into
    set_forward_context at :4315-4324@v0.24.0). The cached prefix travels as
    num_computed_tokens (context_lens) inside CommonAttentionMetadata; the
    builders derive decode/prefill splits themselves
    (split_decodes_and_prefills, common/sparse_attention.py:197-203).
    """
    torch_device = torch.device(device)
    block_size = vllm_config.cache_config.block_size

    prefix_len = int(prefix_len) if is_context else 0

    if is_context:
        batch_spec = BatchSpec(
            seq_lens=[prefix_len + seq_len] * batch_size,
            query_lens=[seq_len] * batch_size,
        )
    else:
        # Generation KV coordinate contract: `seq_len` is the CACHED length
        # (the getter's kv_cache_len), the current token decodes at position
        # seq_len, and the measured total is seq_len + 1 — the coordinate the
        # loader keys the row at (s = isl + step = 1 + seq_len).
        # create_common_attn_metadata derives num_computed_tokens =
        # seq_lens - query_lens, so seq_lens must carry the TOTAL. An earlier
        # revision passed seq_lens=[seq_len] (cached seq_len - 1, total
        # seq_len) while persisting the seq_len + 1 coordinate (one-token key
        # skew; shipped rows from that revision were re-keyed step -> step-1).
        batch_spec = BatchSpec(
            seq_lens=[seq_len + 1] * batch_size,
            query_lens=[1] * batch_size,
        )

    common_attn_metadata = create_common_attn_metadata(batch_spec, block_size, torch_device, arange_block_indices=True)
    num_blocks = _rebase_block_table_and_slots(common_attn_metadata, block_size)

    # Main paged K/V cache: shape from the layer's own backend
    # ((num_blocks, 2, block_size, num_kv_heads, head_size) — MiniMaxM3
    # SparseBackend.get_kv_cache_shape, common/sparse_attention.py:90-98
    # @v0.24.0; NHD stride order is the natural layout, :100-114), dtype
    # from the layer's KV-cache spec (nvidia/model.py:540-549@v0.24.0:
    # spec.dtype = kv_cache_dtype_str_to_dtype(cache_dtype) — bf16 for
    # "auto", uint8 storage for "fp8", utils/torch_utils.py:38,394-400).
    # Serving allocates the same way: a raw per-layer buffer viewed with the
    # spec dtype and backend shape (gpu_model_runner
    # ._reshape_kv_cache_tensors:7081-7180@v0.24.0); the impl then
    # reinterprets the uint8 storage as the platform fp8 (e4m3) at use
    # (common/sparse_attention.py:298-306, view at :352).
    backend_cls = attn_module.get_attn_backend()
    kv_cache_spec = attn_module.get_kv_cache_spec(vllm_config)
    kv_cache = torch.zeros(
        backend_cls.get_kv_cache_shape(num_blocks, block_size, kv_cache_spec.num_kv_heads, kv_cache_spec.head_size),
        dtype=kv_cache_spec.dtype,
        device=torch_device,
    )

    # Indexer key side-cache: (num_blocks, block_size, head_size), dtype from
    # the index cache module (bf16 for the default indexer_kv_dtype "bf16",
    # config/attention.py:55; MiniMaxM3IndexerBackend.get_kv_cache_shape,
    # common/indexer.py:91-99@v0.24.0).
    index_cache_layer = attn_module.indexer.index_cache
    index_backend_cls = index_cache_layer.get_attn_backend()
    index_spec = index_cache_layer.get_kv_cache_spec(vllm_config)
    index_kv_cache = torch.zeros(
        index_backend_cls.get_kv_cache_shape(num_blocks, block_size, 1, index_spec.head_size),
        dtype=index_cache_layer.dtype,
        device=torch_device,
    )

    main_builder = backend_cls.get_builder_cls()(kv_cache_spec, [_ATTN_LAYER_NAME], vllm_config, torch_device)
    attn_metadata = main_builder.build(
        common_prefix_len=prefix_len,
        common_attn_metadata=common_attn_metadata,
    )

    index_builder = index_backend_cls.get_builder_cls()(
        index_spec, [_INDEX_CACHE_LAYER_NAME], vllm_config, torch_device
    )
    index_metadata = index_builder.build(
        common_prefix_len=prefix_len,
        common_attn_metadata=common_attn_metadata,
    )

    return kv_cache, index_kv_cache, attn_metadata, index_metadata, common_attn_metadata


# ═══════════════════════════════════════════════════════════════════════
# Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════


@with_exit_stack
def run_msa_module(
    exit_stack,
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
    if kv_cache_dtype not in ("bfloat16", "fp8") or compute_dtype != "bfloat16":
        raise ValueError(
            f"MSA combos are declared with bf16 compute and bf16/fp8 KV cache at "
            f"vllm {vllm_version} (see _get_precision_combos); "
            f"got compute={compute_dtype}, kv={kv_cache_dtype}"
        )

    setup_distributed(device)
    torch.cuda.set_device(device)
    enable_engine_fused_ops()
    init_workspace_manager(torch.device(device))

    is_context = "context" in perf_filename
    prefix_len = int(prefix_len) if is_context else 0
    phase = "context" if is_context else "generation"
    print(
        f"\n[MSA module] {phase} b={batch_size}, s={seq_len}, "
        f"prefix={prefix_len}, heads={num_heads}, gemm={gemm_type}, "
        f"compute={compute_dtype}, kv={kv_cache_dtype}, model={model_path}"
    )

    # 1. Create attention module (framework builds and dispatches the impls).
    attn_module, vllm_config, original_architecture = _create_msa_attention_module(
        model_path=model_path,
        num_heads=num_heads,
        gemm_type=gemm_type,
        kv_cache_dtype=kv_cache_dtype,
        max_seq_len=prefix_len + seq_len,
        max_batch_size=batch_size,
        is_context=is_context,
        device=device,
    )

    # 2. Create KV caches + metadata via the framework's builders.
    with set_current_vllm_config(vllm_config):
        kv_cache, index_kv_cache, attn_metadata, index_metadata, common_attn_metadata = _create_kv_caches_and_metadata(
            vllm_config=vllm_config,
            attn_module=attn_module,
            batch_size=batch_size,
            seq_len=seq_len,
            is_context=is_context,
            prefix_len=prefix_len,
            device=device,
        )

    # 2b. Bind the caches to the registered layers — the benchmark analog of
    # serving's bind_kv_cache, which assigns each allocated per-layer tensor
    # to forward_context[layer_name].kv_cache (v1/worker/utils.py:462-530
    # @v0.24.0; the model reads self.kv_cache / index_cache.kv_cache directly,
    # nvidia/model.py:606-607).
    forward_ctx = vllm_config.compilation_config.static_forward_context
    forward_ctx[_ATTN_LAYER_NAME].kv_cache = kv_cache
    forward_ctx[_INDEX_CACHE_LAYER_NAME].kv_cache = index_kv_cache

    # 2c. Prove the requested KV precision actually reached the framework
    # path — a benchmark that silently ran bf16 under an "fp8" label would be
    # wrong data, worse than a crash. For "fp8": the layer must have been
    # built with cache_dtype "fp8" (nvidia/model.py:483-488@v0.24.0), the
    # cache storage must be the fp8 uint8 layout (utils/torch_utils.py:38),
    # and the framework-selected impl must be in fp8 mode (use_fp8_kv,
    # common/sparse_attention.py:298-306). The indexer side-cache stays bf16
    # in both combos (default indexer_kv_dtype, config/attention.py:55).
    want_fp8_kv = kv_cache_dtype == "fp8"
    expected_cache_dtype = torch.uint8 if want_fp8_kv else torch.bfloat16
    impl_use_fp8_kv = bool(getattr(attn_module.impl, "use_fp8_kv", False))
    if (
        kv_cache.dtype != expected_cache_dtype
        or impl_use_fp8_kv != want_fp8_kv
        or index_kv_cache.dtype != torch.bfloat16
    ):
        raise RuntimeError(
            f"KV-cache precision mismatch: requested kv={kv_cache_dtype}, allocated "
            f"main cache {kv_cache.dtype}, impl use_fp8_kv={impl_use_fp8_kv}, "
            f"index cache {index_kv_cache.dtype} "
            f"(layer cache_dtype={attn_module.kv_cache_dtype!r})"
        )

    # 3. Input tensors. Positions are absolute (cached prefix precedes the
    # current chunk); generation decodes token seq_len-1 with seq_len-1
    # cached tokens.
    hidden_size = vllm_config.model_config.hf_text_config.hidden_size
    torch_device = torch.device(device)
    if is_context:
        num_tokens = seq_len * batch_size
        positions = (
            torch.arange(prefix_len, prefix_len + seq_len, device=torch_device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .reshape(-1)
            .contiguous()
        )
    else:
        num_tokens = batch_size
        positions = torch.full((batch_size,), seq_len, device=torch_device, dtype=torch.long)

    hidden_states = torch.full(
        (num_tokens, hidden_size),
        0.01,
        dtype=torch.bfloat16,
        device=torch_device,
    )

    # 4. Forward context: metadata + slot-mapping dicts keyed by layer name,
    # exactly the structures serving passes into set_forward_context
    # (gpu_model_runner.py:4315-4324 with _get_slot_mappings :3972-4045
    # @v0.24.0). The model consumes slot_mapping[self.layer_name] and
    # slot_mapping[indexer.index_cache.prefix] (nvidia/model.py:574-583) and
    # attn_metadata[layer_name] inside the impls
    # (common/sparse_attention.py:337-340, common/indexer.py:395-398).
    attn_metadata_dict = {
        _ATTN_LAYER_NAME: attn_metadata,
        _INDEX_CACHE_LAYER_NAME: index_metadata,
    }
    slot_mapping_dict = {
        _ATTN_LAYER_NAME: common_attn_metadata.slot_mapping,
        _INDEX_CACHE_LAYER_NAME: common_attn_metadata.slot_mapping,
    }
    exit_stack.enter_context(set_current_vllm_config(vllm_config))
    exit_stack.enter_context(set_forward_context(attn_metadata_dict, vllm_config, slot_mapping=slot_mapping_dict))

    # Steps 5-7 run under one try/finally: collect.py reuses this worker
    # process for later cases, so the workspace singleton and CUDA
    # allocations must be released on EVERY exit path — a leak here turns
    # one failure into misleading follow-on OOMs.
    try:
        # 5. Dry run
        try:
            with torch.inference_mode():
                attn_module.forward(positions, hidden_states)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  Dry run OOM: {e}")
            raise
        except Exception as e:
            print(f"  Dry run failed: {e}")
            traceback.print_exc()
            # Propagate to collect.py's worker so the failure is recorded in
            # the error queue (no silent skip).
            raise

        # 6. Benchmark. CUDA graph capture is mandatory (allow_graph_fail=False):
        # the M3 Triton kernels are capture-safe (no host syncs; the builders
        # declare AttentionCGSupport.UNIFORM_BATCH, common/sparse_attention.py
        # :161-167, common/indexer.py:214-219@v0.24.0), verified on SM90.
        def kernel_func():
            attn_module.forward(positions, hidden_states)

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

        # 7. Log results — schema aligned with TRT-LLM's collect_msa_module.
        if is_context:
            isl = seq_len
            step = prefix_len
        else:
            isl = 1
            step = seq_len

        op_name = f"msa_{phase}_module"

        # Ground truth: the impl class the framework's own dispatch selected at
        # construction (select_main_impl_cls — Triton off-SM100, MSA on SM100).
        kernel_source = type(attn_module.impl).__name__

        _persist_msa_row(
            item={
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
            },
            vllm_version=vllm_version,
            device_name=torch.cuda.get_device_name(device),
            op_name=op_name,
            kernel_source=kernel_source,
            perf_filename=perf_filename,
            power_stats=results["power_stats"],
        )

        print(
            f"  [{phase}] b={batch_size}, s={seq_len}, heads={num_heads}, "
            f"prefix={prefix_len}, gemm={gemm_type}, kv={kv_cache_dtype} "
            f"(cache storage {kv_cache.dtype}), backend={kernel_source}: {latency:.4f} ms"
        )

        return latency
    finally:
        _cleanup()


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


def _cleanup():
    # Release vLLM's WorkspaceManager singleton scratch buffers between
    # tasks (same rationale as collect_mla_module._cleanup: the manager
    # only grows and pins its high-water mark for the worker's lifetime).
    import vllm.v1.worker.workspace as _ws_mod

    _ws_mod._manager = None
    gc.collect()
    torch.cuda.empty_cache()


def _persist_msa_row(
    *,
    item: dict,
    vllm_version: str,
    device_name: str,
    op_name: str,
    kernel_source: str,
    perf_filename: str,
    power_stats: dict | None,
) -> None:
    """Persist one MSA perf row, failing the case when the write fails.

    ``log_perf`` returns False after lock exhaustion or a write/fsync
    exception; discarding that would let the worker return normally and the
    checkpoint advance with no row persisted (a silently shrunk dataset).
    Mirrors the TRT-LLM and SGLang collectors."""
    if not log_perf(
        item_list=[item],
        framework="VLLM",
        version=vllm_version,
        device_name=device_name,
        op_name=op_name,
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=power_stats,
    ):
        raise RuntimeError(f"failed to persist MSA row to {perf_filename}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def _perf_file_for(mode: str) -> str:
    return PerfFile.MSA_CONTEXT_MODULE if mode == "context" else PerfFile.MSA_GENERATION_MODULE


def main():
    all_model_specs = get_mla_module_model_specs(attention_type="msa", apply_model_filter=False)
    model_names = [spec.model_path for spec in all_model_specs]

    parser = argparse.ArgumentParser(description="MiniMax-M3 MSA module-level collector for vLLM")
    parser.add_argument("--mode", choices=["context", "generation"], required=True)
    parser.add_argument("--model", type=str, default=None, choices=model_names)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--prefix-len", type=int, default=0)
    parser.add_argument("--gemm-type", type=str, choices=["bfloat16", "fp8_block", "nvfp4"], default=None)
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["bfloat16", "fp8"],
        default=None,
        help="KV cache dtype (default: run both bfloat16 and fp8)",
    )
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
                kv_cache_dtype=args.kv_cache_dtype or "bfloat16",
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
        if args.kv_cache_dtype is not None:
            test_cases = [tc for tc in test_cases if tc[3] == args.kv_cache_dtype]
        if args.gemm_type is not None:
            test_cases = [tc for tc in test_cases if tc[5] == args.gemm_type]
        # Honor the shape options outside --quick too, so a reproduction
        # command narrows to the named case instead of silently running the
        # whole sweep.
        if args.seq_len is not None:
            test_cases = [tc for tc in test_cases if tc[0] == args.seq_len]
        if args.batch_size is not None:
            test_cases = [tc for tc in test_cases if tc[1] == args.batch_size]
        if args.prefix_len is not None:
            test_cases = [tc for tc in test_cases if (tc[6] if len(tc) > 6 else 0) == args.prefix_len]

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
                print(f"  OOM: b={b}, s={s}, heads={h}, gemm={gemm}, kv={kv_dtype}")
                num_failed += 1
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"  FAILED: b={b}, s={s}, heads={h}, gemm={gemm}, kv={kv_dtype}: {e}")
                traceback.print_exc()
                num_failed += 1
                torch.cuda.empty_cache()
                gc.collect()
        if num_failed:
            # The registry/executor path records classified failures via the
            # raising worker; this standalone repro CLI at least exits
            # non-zero so a failed run is visible without scraping the log.
            print(f"{num_failed}/{len(test_cases)} cases failed")
            sys.exit(1)


if __name__ == "__main__":
    main()
