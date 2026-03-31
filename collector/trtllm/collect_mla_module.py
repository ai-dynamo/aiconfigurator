# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__compat__ = ">=1.2.0rc5"

"""
MLA Module Collector for TRT-LLM — unified MLA and DSA benchmarking.

Profiles the complete attention module forward pass (projections + attention +
output), not just the bare attention kernel.  Uses TRT-LLM's own modeling code
to construct a single-layer mock model with dummy weights, then extracts the
attention module for benchmarking.

Supported models and their attention types are defined in SUPPORTED_MODELS.
Model dimensions are loaded from the HF config.json via from_pretrained().

Usage:
    # MLA context phase (DeepSeek-V3)
    python collect_mla_module.py --mode context --model deepseek-ai/DeepSeek-V3

    # DSA generation phase (DeepSeek-V3.2)
    python collect_mla_module.py --mode generation --model deepseek-ai/DeepSeek-V3.2

    # All models, context phase
    python collect_mla_module.py --mode context

    # Quick single-point test
    python collect_mla_module.py --mode context --model deepseek-ai/DeepSeek-V3
    --quick --batch-size 4 --seq-len 2048 --num-heads 64

    # FP8 KV cache only
    python collect_mla_module.py --mode context --model deepseek-ai/DeepSeek-V3 --kv-cache-dtype fp8
"""

import argparse
import dataclasses
import gc
import os
import sys
import traceback
import weakref

import tensorrt_llm
import torch
from tensorrt_llm._torch.attention_backend.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import (
    DeepseekV3DecoderLayer,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.pyexecutor._util import get_kv_cache_manager_cls

# ═══════════════════════════════════════════════════════════════════════
# Config registry patch — TRT-LLM's _CONFIG_REGISTRY only maps
# "deepseek_v32" but GLM-5 uses model_type "glm_moe_dsa".  Without this
# patch load_pretrained_config() falls through to AutoConfig which also
# doesn't know "glm_moe_dsa", causing a KeyError.  The config layout is
# identical to DeepSeek-V3, so reusing DeepseekV3Config is safe.
# ═══════════════════════════════════════════════════════════════════════
from tensorrt_llm._torch.pyexecutor.config_utils import _CONFIG_REGISTRY
from tensorrt_llm._torch.pyexecutor.model_loader import initialize_dummy_weights
from tensorrt_llm._torch.utils import AuxStreamType, get_model_extra_attrs, model_extra_attrs
from tensorrt_llm._utils import torch_dtype_to_binding
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.mode import QuantAlgo

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from collector.helper import benchmark_with_power, get_sm_version, log_perf

if "glm_moe_dsa" not in _CONFIG_REGISTRY:
    _CONFIG_REGISTRY["glm_moe_dsa"] = "DeepseekV3Config"

# Fix NVFP4 weight_scale 2D bug in TRT-LLM <= rc6: weight_scale is allocated 1D
# but post_load_weights() asserts dim()==2. Patch only if defined on the class
# itself (rc9+ removed post_load_weights from DeepseekV32Attention).
try:
    from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV32Attention

    if "post_load_weights" in DeepseekV32Attention.__dict__:

        def _post_load_weights(self):
            assert self.kv_a_proj_with_mqa.weight.data.dtype == self.indexer.wk.weight.data.dtype
            offset = self.kv_lora_rank + self.qk_rope_head_dim + self.q_lora_rank
            self.kv_a_proj_with_mqa.weight.data[offset : offset + self.indexer.head_dim].copy_(
                self.indexer.wk.weight.data
            )
            if getattr(self.indexer.wk, "weight_scale", None) is not None:
                _pad = lambda x, n: (x + n - 1) // n * n
                for m in (self.kv_a_proj_with_mqa, self.indexer.wk):
                    ws = m.weight_scale
                    if ws.dim() == 1:
                        nr = _pad(m.weight.shape[0], 128)
                        m.weight_scale.data = ws.data.view(nr, ws.numel() // nr)
                scale = self.indexer.wk.weight_scale.data
                self.kv_a_proj_with_mqa.weight_scale.data[offset : offset + scale.shape[0]].copy_(scale)
            self.indexer.wk = None

        DeepseekV32Attention.post_load_weights = _post_load_weights
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════
# Supported Models — model_path → attention type
# ═══════════════════════════════════════════════════════════════════════

SUPPORTED_MODELS: dict[str, str] = {
    "deepseek-ai/DeepSeek-V3": "mla",
    "deepseek-ai/DeepSeek-V3.2": "dsa",
    "zai-org/GLM-5": "dsa",
}


# ═══════════════════════════════════════════════════════════════════════
# Test Cases
# ═══════════════════════════════════════════════════════════════════════


def _get_precision_combos(phase: str, attn_type: str):
    """Return (compute_dtype, kv_cache_dtype, gemm_type) triples for a phase.

    Each triple describes the full quantisation configuration for one
    benchmark sweep.  GPU capability (SM version) determines which
    combos are available.

    Precision axes:
      gemm_type    — linear-layer GEMMs (projections inside the module)
        bfloat16:  always
        fp8_block: SM >= 89 (Ada / Hopper / Blackwell)
        nvfp4:     SM >= 100 (Blackwell)

      (compute_dtype, kv_cache_dtype) — attention compute + KV cache
        TRT-LLM currently only supports bfloat16 attention compute.
        FP8 KV cache availability depends on attn_type and SM:
          MLA:  Hopper only (86 < SM < 100) — old FMHA FP8 MLA variants.
          DSA:  Blackwell (SM >= 100) — trtllm-gen sparseMla=1 FP8 kernels.
    """
    sm = get_sm_version()
    is_dsa = attn_type == "dsa"

    gemm_types = ["bfloat16"]
    if sm >= 89:
        gemm_types.append("fp8_block")
    if sm >= 100:
        gemm_types.append("nvfp4")

    # FIXME: FP8 KV cache and platform combinations are incomplete.
    attn_combos = [("bfloat16", "bfloat16")]
    if is_dsa:
        if sm >= 100:
            attn_combos.append(("bfloat16", "fp8"))
    else:
        if 86 < sm < 100:
            attn_combos.append(("bfloat16", "fp8"))

    return [(c, kv, g) for g in gemm_types for c, kv in attn_combos]


def get_context_test_cases(attn_type: str):
    """Context-phase test cases.

    Returns list of [seq_len, batch_size, num_heads, kv_cache_dtype,
                     compute_dtype, gemm_type, perf_filename].
    """
    cases = []
    b_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    s_list = [1, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 32768]
    base_fname = f"{attn_type}_context_module_perf.txt"
    for compute_dtype, kv_dtype, gemm_type in _get_precision_combos("context", attn_type):
        for num_heads in [128, 64, 32, 16, 8, 4, 2, 1]:
            for b in b_list:
                for s in s_list:
                    if b * s > 131072:
                        continue
                    cases.append([s, b, num_heads, kv_dtype, compute_dtype, gemm_type, base_fname])
    return cases


def get_generation_test_cases(attn_type: str):
    """Generation-phase test cases.

    Returns list of [kv_cache_len, batch_size, num_heads, kv_cache_dtype,
                     compute_dtype, gemm_type, perf_filename].
    """
    cases = []
    b_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    s_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    base_fname = f"{attn_type}_generation_module_perf.txt"
    for compute_dtype, kv_dtype, gemm_type in _get_precision_combos("generation", attn_type):
        for num_heads in [128, 64, 32, 16, 8, 4, 2, 1]:
            for b in b_list:
                for s in s_list:
                    if b * s > 1024 * 4096 * 2 * 2 * 2:
                        continue
                    cases.append([s, b, num_heads, kv_dtype, compute_dtype, gemm_type, base_fname])
    return cases


def _build_module_test_cases(attn_type: str, mode: str):
    """Build module-level test cases for a specific attention type and phase.

    Output test case format is positional args for run_mla_module_worker:
    [seq_len, batch_size, num_heads, kv_cache_dtype, compute_dtype, gemm_type,
     perf_filename, model_path, attn_type]
    """
    base_cases = get_context_test_cases(attn_type) if mode == "context" else get_generation_test_cases(attn_type)
    model_paths = [m for m, t in SUPPORTED_MODELS.items() if t == attn_type]
    cases = []
    for model_path in model_paths:
        for s, b, h, kv_dtype, compute_dtype, gemm_type, fname in base_cases:
            cases.append([s, b, h, kv_dtype, compute_dtype, gemm_type, fname, model_path, attn_type])
    return cases


def get_mla_context_module_test_cases():
    """collect.py entrypoint for MLA context module collection."""
    return _build_module_test_cases(attn_type="mla", mode="context")


def get_mla_generation_module_test_cases():
    """collect.py entrypoint for MLA generation module collection."""
    return _build_module_test_cases(attn_type="mla", mode="generation")


def get_dsa_context_module_test_cases():
    """collect.py entrypoint for DSA context module collection."""
    return _build_module_test_cases(attn_type="dsa", mode="context")


def get_dsa_generation_module_test_cases():
    """collect.py entrypoint for DSA generation module collection."""
    return _build_module_test_cases(attn_type="dsa", mode="generation")


# ═══════════════════════════════════════════════════════════════════════
# Layer Construction
# ═══════════════════════════════════════════════════════════════════════


def _ceil_div(a, b):
    return (a + b - 1) // b


def _round_up(a, b):
    return _ceil_div(a, b) * b


def _replace_quant_config(qc, **kwargs):
    """Replace fields in quant_config; supports both dataclass and Pydantic BaseModel."""
    if dataclasses.is_dataclass(qc):
        return dataclasses.replace(qc, **kwargs)
    # Pydantic BaseModel (TRT-LLM >= rc9 uses StrictBaseModel / Pydantic v2)
    return qc.model_copy(update=kwargs)


def _set_quant_config(model_config, new_qc):
    """Set quant_config, bypassing ModelConfig freeze if necessary."""
    try:
        model_config.quant_config = new_qc
    except AttributeError:
        object.__setattr__(model_config, "quant_config", new_qc)


def _apply_gemm_type_quant(model_config, gemm_type: str, use_fp8_kv_cache: bool):
    """Apply GEMM quantization to model_config.quant_config.

    Replaces the quant_config so that linear-layer GEMMs (and MLA absorption
    BMMs where supported) run at the requested precision.
    """
    kv_algo = QuantAlgo.FP8 if use_fp8_kv_cache else None

    if gemm_type == "bfloat16":
        if use_fp8_kv_cache:
            _set_quant_config(
                model_config,
                _replace_quant_config(
                    model_config.quant_config,
                    kv_cache_quant_algo=kv_algo,
                ),
            )
    elif gemm_type == "fp8_block":
        _set_quant_config(
            model_config,
            _replace_quant_config(
                model_config.quant_config,
                quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
                group_size=128,
                kv_cache_quant_algo=kv_algo,
                exclude_modules=None,
            ),
        )
    elif gemm_type == "nvfp4":
        _set_quant_config(
            model_config,
            _replace_quant_config(
                model_config.quant_config,
                quant_algo=QuantAlgo.NVFP4,
                kv_cache_quant_algo=kv_algo,
                exclude_modules=None,
            ),
        )
    else:
        raise ValueError(f"Unknown gemm_type: {gemm_type!r}")


def create_attention_layer(
    model_path: str,
    num_heads: int = 128,
    use_fp8_kv_cache: bool = False,
    gemm_type: str = "bfloat16",
    device: str = "cuda:0",
):
    """
    Create a single attention layer from TRT-LLM's own modeling code.

    Uses the official config.json from the HF model repo via from_pretrained()
    so the layer matches real inference behavior.  MLA vs DSA is determined by
    the model's config.json automatically (model_type, sparse_attention_config).
    """
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)

    # GLM-5 uses model_type "glm_moe_dsa" / arch "GlmMoeDsaForCausalLM" which
    # TRT-LLM doesn't recognise.  ModelConfig.from_pretrained() only auto-builds
    # sparse_attention_config for "DeepseekV32ForCausalLM", so we pre-read the
    # HF config and supply it manually for GLM-5.
    sparse_attention_config = None
    import transformers

    _cfg_dict, _ = transformers.PretrainedConfig.get_config_dict(model_path)
    if _cfg_dict.get("model_type") == "glm_moe_dsa":
        from tensorrt_llm._torch.model_config import DeepSeekSparseAttentionConfig

        sparse_attention_config = DeepSeekSparseAttentionConfig(
            index_n_heads=_cfg_dict["index_n_heads"],
            index_head_dim=_cfg_dict["index_head_dim"],
            index_topk=_cfg_dict["index_topk"],
        )

    # Capture the original HF architecture *before* from_pretrained() which
    # may remap the model_type.  We return it separately because ModelConfig
    # may be frozen (TRT-LLM >= 0.18, PR #4814) and cannot accept new attrs.
    original_architecture = _cfg_dict.get("architectures", [_cfg_dict.get("model_type", "unknown")])[0]

    model_config = ModelConfig.from_pretrained(
        model_path,
        mapping=mapping,
        enable_min_latency=False,
        use_cuda_graph=False,
        force_dynamic_quantization=False,
        spec_config=None,
        sparse_attention_config=sparse_attention_config,
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

    pretrained_config = model_config.pretrained_config

    # GLM-5 uses model_type "glm_moe_dsa" / arch "GlmMoeDsaForCausalLM" but
    # TRT-LLM only recognises "deepseek_v32" / "DeepseekV32ForCausalLM".
    # The architecture is identical, so we remap to avoid falling into the
    # wrong code path (MLA instead of DSA).
    if pretrained_config.model_type == "glm_moe_dsa":
        pretrained_config.model_type = "deepseek_v32"
        pretrained_config.architectures = ["DeepseekV32ForCausalLM"]

    # Override for single-layer, single-GPU benchmark
    pretrained_config.num_hidden_layers = 1
    pretrained_config.num_attention_heads = num_heads
    pretrained_config.num_key_value_heads = num_heads

    _apply_gemm_type_quant(model_config, gemm_type, use_fp8_kv_cache)

    aux_stream = torch.cuda.Stream(device=device)
    aux_stream_dict = {
        AuxStreamType.Attention: aux_stream,
        AuxStreamType.MoeShared: aux_stream,
        AuxStreamType.MoeChunkingOverlap: torch.cuda.Stream(device=device),
    }

    layer = DeepseekV3DecoderLayer(
        model_config=model_config,
        layer_idx=0,
        aux_stream_dict=aux_stream_dict,
    )

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

    next_ln = RMSNorm(
        hidden_size=pretrained_config.hidden_size,
        eps=pretrained_config.rms_norm_eps,
        dtype=pretrained_config.torch_dtype,
    ).to(device)
    next_ln.requires_grad_(False)
    initialize_dummy_weights(next_ln)
    layer.next_layer_layernorm = next_ln

    attn_module = layer.self_attn
    return attn_module, model_config, original_architecture


# ═══════════════════════════════════════════════════════════════════════
# KV Cache + Metadata
# ═══════════════════════════════════════════════════════════════════════


def create_kv_cache_and_metadata(
    model_config: ModelConfig,
    attn_module,
    batch_size: int,
    seq_len: int,
    is_context: bool,
    use_fp8_kv_cache: bool = False,
    device: str = "cuda:0",
):
    """
    Create KV cache manager and attention metadata using framework utilities.

    Follows the same pattern as TRT-LLM's ``layer_wise_benchmarks/runner_utils.py``.
    """
    config = model_config.pretrained_config
    mapping = model_config.mapping
    tokens_per_block = 64

    kv_lora_rank = config.kv_lora_rank
    qk_rope_head_dim = config.qk_rope_head_dim
    head_dim = kv_lora_rank + qk_rope_head_dim

    if is_context:
        max_seq = seq_len + 1
        total_tokens = seq_len * batch_size
        seq_len_q = seq_len
        kv_cache_len = 0
    else:
        max_seq = seq_len + 1
        total_tokens = batch_size
        seq_len_q = 1
        kv_cache_len = seq_len

    # --- KV Cache Manager ---
    kv_cache_config = KvCacheConfig(
        max_tokens=batch_size * _round_up(max_seq, tokens_per_block),
        enable_block_reuse=False,
    )
    import inspect as _inspect

    _kv_sig = _inspect.signature(get_kv_cache_manager_cls)
    if "kv_cache_config" in _kv_sig.parameters:
        # TRT-LLM >= rc9: requires kv_cache_config argument
        kv_cache_manager_cls = get_kv_cache_manager_cls(model_config, kv_cache_config)
    else:
        kv_cache_manager_cls = get_kv_cache_manager_cls(model_config)
    kv_cache_dtype = DataType.FP8 if use_fp8_kv_cache else torch_dtype_to_binding(torch.bfloat16)

    layer_mask = [True]  # single layer
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
        layer_mask=layer_mask,
        sparse_attn_config=model_config.sparse_attention_config,
        model_config=model_config,
    )

    request_ids = list(range(batch_size))
    token_nums = [max_seq] * batch_size
    kv_cache_manager.add_dummy_requests(request_ids, token_nums)

    # --- Attention Metadata ---
    attention_cls = get_attention_backend(
        model_config.attn_backend,
        model_config.sparse_attention_config,
    )

    sm_major = torch.cuda.get_device_capability()[0]
    _enable_flash_mla = (
        model_config.attn_backend == "TRTLLM" and (kv_lora_rank + qk_rope_head_dim) == 576 and sm_major == 9
    )

    attn_metadata = attention_cls.Metadata(
        max_num_requests=batch_size,
        max_num_tokens=total_tokens,
        kv_cache_manager=kv_cache_manager,
        mapping=mapping,
        enable_flash_mla=_enable_flash_mla,
        seq_lens=torch.tensor([seq_len_q] * batch_size, dtype=torch.int32),
        position_ids=None,
        num_contexts=batch_size if is_context else 0,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[kv_cache_len] * batch_size,
        ),
        cross=None,
        request_ids=request_ids,
        prompt_lens=[seq_len_q if is_context else kv_cache_len] * batch_size,
        runtime_features=AttentionRuntimeFeatures(
            chunked_prefill=False,
            cache_reuse=False,
        ),
        all_rank_num_tokens=None,
        workspace=torch.tensor([], device=device, dtype=torch.int8),
        sparse_attention_config=model_config.sparse_attention_config,
    )

    # DSA needs a reference to the indexer for prepare()
    if hasattr(attn_module, "indexer") and attn_module.indexer is not None:
        attn_metadata.indexer = attn_module.indexer

    attn_metadata.prepare()

    return kv_cache_manager, attn_metadata


# ═══════════════════════════════════════════════════════════════════════
# Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════


def run_mla_module(
    seq_len: int,
    batch_size: int,
    num_heads: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    perf_filename: str,
    *,
    model_path: str,
    attn_type: str,
    device: str = "cuda:0",
    warming_up: int = 10,
    test_ite: int = 6,
):
    """Run a single MLA / DSA module-level benchmark point."""
    torch.cuda.set_device(device)
    torch_device = torch.device(device)

    use_fp8_kv_cache = kv_cache_dtype == "fp8"

    is_context = "context" in perf_filename
    phase = "context" if is_context else "generation"
    variant = attn_type.upper()
    print(
        f"\n[{variant} module] {phase} b={batch_size}, s={seq_len}, "
        f"heads={num_heads}, gemm={gemm_type}, compute={compute_dtype}, kv={kv_cache_dtype}, model={model_path}"
    )

    # 1. Create attention layer (from_pretrained reads config.json)
    attn_module, model_config, original_architecture = create_attention_layer(
        model_path=model_path,
        num_heads=num_heads,
        use_fp8_kv_cache=use_fp8_kv_cache,
        gemm_type=gemm_type,
        device=device,
    )

    # 2. Create KV cache + metadata
    kv_cache_manager, attn_metadata = create_kv_cache_and_metadata(
        model_config=model_config,
        attn_module=attn_module,
        batch_size=batch_size,
        seq_len=seq_len,
        is_context=is_context,
        use_fp8_kv_cache=use_fp8_kv_cache,
        device=device,
    )

    # 3. Input tensors
    hidden_size = model_config.pretrained_config.hidden_size
    if is_context:
        num_tokens = seq_len * batch_size
        position_ids = (
            torch.arange(seq_len, device=torch_device, dtype=torch.long)
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

    # 4. Dry run
    with model_extra_attrs(model_config.extra_attrs):
        get_model_extra_attrs()["attention_metadata"] = weakref.ref(attn_metadata)
        try:
            with torch.inference_mode():
                attn_module.forward(position_ids, hidden_states, attn_metadata)
        except Exception as e:
            print(f"  Dry run failed: {e}")
            traceback.print_exc()
            _cleanup(kv_cache_manager)
            return

    # 5. Benchmark
    import tensorrt_llm._torch.utils as _trtllm_utils

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

    # 6. Log results
    if is_context:
        isl = seq_len
        step = 0
    else:
        isl = 1
        step = seq_len

    op_name = f"{attn_type}_{phase}_module"

    architecture = original_architecture

    log_perf(
        item_list=[
            {
                "model": model_path,
                "architecture": architecture,
                "mla_dtype": "float16" if compute_dtype == "bfloat16" else compute_dtype,
                "kv_cache_dtype": "float16" if kv_cache_dtype == "bfloat16" else kv_cache_dtype,
                "gemm_type": "float16" if gemm_type == "bfloat16" else gemm_type,
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
        kernel_source="default",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )

    print(
        f"  [{phase}] b={batch_size}, s={seq_len}, heads={num_heads}, "
        f"gemm={gemm_type}, compute={compute_dtype}, kv={kv_cache_dtype}: {latency:.4f} ms"
    )

    _cleanup(kv_cache_manager)
    return latency


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
    device: str = "cuda:0",
):
    """Worker-compatible positional wrapper used by collector/collect.py."""
    return run_mla_module(
        seq_len=seq_len,
        batch_size=batch_size,
        num_heads=num_heads,
        kv_cache_dtype=kv_cache_dtype,
        compute_dtype=compute_dtype,
        gemm_type=gemm_type,
        perf_filename=perf_filename,
        model_path=model_path,
        attn_type=attn_type,
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


def main():
    model_names = list(SUPPORTED_MODELS.keys())

    parser = argparse.ArgumentParser(
        description="MLA/DSA module-level collector for TRT-LLM",
    )
    parser.add_argument("--mode", choices=["context", "generation"], required=True)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=model_names,
        help=f"Model to benchmark. If not specified, runs all: {model_names}",
    )
    parser.add_argument("--num-heads", type=int, default=None, help="Filter by number of heads")
    parser.add_argument("--batch-size", type=int, default=None, help="Single batch size (for --quick)")
    parser.add_argument("--seq-len", type=int, default=None, help="Single seq len (for --quick)")
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["bfloat16", "fp8"],
        default=None,
        help="KV cache dtype (default: run both bfloat16 and fp8 when GPU supports it)",
    )
    parser.add_argument(
        "--compute-dtype",
        type=str,
        choices=["bfloat16"],
        default=None,
        help="Compute dtype for attention (default: bfloat16; reserved for future FP8 support)",
    )
    parser.add_argument(
        "--gemm-type",
        type=str,
        choices=["bfloat16", "fp8_block", "nvfp4"],
        default=None,
        help="GEMM quantisation type for linear layers (default: run all supported by GPU)",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--quick", action="store_true", help="Quick single-point test")
    args = parser.parse_args()

    # Select models to run
    if args.model:
        models_to_run = {args.model: SUPPORTED_MODELS[args.model]}
    else:
        models_to_run = SUPPORTED_MODELS

    for model_path, attn_type in models_to_run.items():
        print(f"\n{'=' * 60}")
        print(f"Model: {model_path}  |  Attention: {attn_type.upper()}")
        print(f"{'=' * 60}")

        if args.quick:
            b = args.batch_size or 4
            s = args.seq_len or 2048
            h = args.num_heads or 128
            kv_dtype = args.kv_cache_dtype or "bfloat16"
            compute = args.compute_dtype or "bfloat16"
            gemm = args.gemm_type or "bfloat16"
            fname = f"{attn_type}_{args.mode}_module_perf.txt"
            run_mla_module(
                seq_len=s,
                batch_size=b,
                num_heads=h,
                kv_cache_dtype=kv_dtype,
                compute_dtype=compute,
                gemm_type=gemm,
                perf_filename=fname,
                model_path=model_path,
                attn_type=attn_type,
                device=args.device,
            )
            continue

        if args.mode == "context":
            test_cases = get_context_test_cases(attn_type=attn_type)
        else:
            test_cases = get_generation_test_cases(attn_type=attn_type)

        if args.num_heads is not None:
            test_cases = [tc for tc in test_cases if tc[2] == args.num_heads]

        if args.kv_cache_dtype is not None:
            test_cases = [tc for tc in test_cases if tc[3] == args.kv_cache_dtype]

        if args.compute_dtype is not None:
            test_cases = [tc for tc in test_cases if tc[4] == args.compute_dtype]

        if args.gemm_type is not None:
            test_cases = [tc for tc in test_cases if tc[5] == args.gemm_type]

        print(f"Running {len(test_cases)} {args.mode} {attn_type.upper()} module test cases...")

        for i, (s, b, h, kv_dtype, compute, gemm, fname) in enumerate(test_cases):
            print(f"[{i + 1}/{len(test_cases)}]", end="")
            try:
                run_mla_module(
                    seq_len=s,
                    batch_size=b,
                    num_heads=h,
                    kv_cache_dtype=kv_dtype,
                    compute_dtype=compute,
                    gemm_type=gemm,
                    perf_filename=fname,
                    model_path=model_path,
                    attn_type=attn_type,
                    device=args.device,
                )
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM: b={b}, s={s}, heads={h}, gemm={gemm}, compute={compute}, kv={kv_dtype}")
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"  FAILED: b={b}, s={s}, heads={h}, gemm={gemm}, compute={compute}, kv={kv_dtype}: {e}")
                traceback.print_exc()
                torch.cuda.empty_cache()
                gc.collect()


if __name__ == "__main__":
    main()
