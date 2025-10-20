# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass

import tensorrt_llm
import torch
from helper import log_perf
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionInputType,
    MLAParams,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.sampling_params import SamplingParams


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@dataclass(kw_only=True, frozen=True)
class Scenario:
    dtype: torch.dtype = torch.bfloat16
    kv_cache_dtype: torch.dtype = torch.bfloat16
    num_layers: int = 1
    num_heads: int = 128
    num_kv_heads: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    hidden_size: int = 7168
    max_position_embeddings: int = 163840
    rope_theta: float = 10000.0
    rope_beta_fast: int = 32
    rope_beta_slow: int = 1
    rope_factor: float = 40.0
    rope_mscale: float = 1.0
    rope_mscale_all_dim: float = 1.0
    rope_original_max_position_embeddings: int = 4096
    rope_type: str = "yarn"
    model_type: str = "deepseek_v3"
    kv_cache_tokens_per_block: int = 64


@dataclass(kw_only=True, frozen=True)
class RopeConfig:
    hidden_size: int = 7168
    num_attention_heads: int = 128
    rope_scaling: dict = (
        {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
        },
    )
    max_position_embeddings: int = 163840
    rope_theta: float = 10000.0
    qk_rope_head_dim: int = 64
    model_type: str = "deepseek_v3"


def run_mla(
    input_len,
    batch_size,
    output_len,
    kv_cache_dtype,
    num_heads,
    world_size,
    tp_size,
    tokens_per_block,
    warming_up,
    test_ite,
    is_context_phase,
    perf_filename,
    device,
):
    scenario = Scenario()
    q_lora_rank = scenario.q_lora_rank
    kv_lora_rank = scenario.kv_lora_rank
    qk_nope_head_dim = scenario.qk_nope_head_dim
    qk_rope_head_dim = scenario.qk_rope_head_dim
    v_head_dim = scenario.v_head_dim
    rope_config = RopeConfig(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        rope_scaling={
            "beta_fast": scenario.rope_beta_fast,
            "beta_slow": scenario.rope_beta_slow,
            "factor": scenario.rope_factor,
            "mscale": scenario.rope_mscale,
            "mscale_all_dim": scenario.rope_mscale_all_dim,
            "original_max_position_embeddings": scenario.rope_original_max_position_embeddings,
            "type": scenario.rope_type,
        },
        max_position_embeddings=scenario.max_position_embeddings,
        rope_theta=scenario.rope_theta,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        model_type=scenario.model_type,
    )
    kv_cache_tokens_per_block = tokens_per_block
    # device = torch.device('cuda')
    dtype = scenario.dtype
    num_kv_heads = num_heads

    context_sequence_lengths = [input_len for _ in range(batch_size)]

    num_generation_steps = 0 if is_context_phase else 1

    _run_attn_for_backend(
        "TRTLLM",
        num_heads,
        num_kv_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        rope_config,
        kv_cache_tokens_per_block,
        device,
        dtype,
        kv_cache_dtype,
        context_sequence_lengths,
        output_len,
        num_generation_steps,
        world_size,
        tp_size,
        warming_up,
        test_ite,
        is_context_phase,
        perf_filename,
    )


def _run_attn_for_backend(
    backend_name,
    num_heads,
    num_kv_heads,
    q_lora_rank,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    rope_config,
    kv_cache_tokens_per_block,
    device,
    dtype,
    kv_cache_dtype,
    context_sequence_lengths,
    generation_seq_len_q,
    num_generation_steps,
    world_size,
    tp_size,
    warming_up,
    test_ite,
    is_context_phase,
    perf_filename,
):
    max_context_sequence_length = max(context_sequence_lengths)
    max_num_contexts = len(context_sequence_lengths)

    torch.cuda.set_device(device)
    attention_cls = get_attention_backend(backend_name)
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    # Setup attention module and metadata
    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        rope=RopeParams.from_config(rope_config),
        is_neox=False,
    )
    mla_params = MLAParams(
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=v_head_dim,
        predicted_tokens_per_seq=1,
    )

    def yarn_get_mscale(scale=1, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    mscale_all_dim = pos_embd_params.rope.mscale_all_dim
    scaling_factor = pos_embd_params.rope.scale
    mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
    q_scaling = 1.0 / (mscale * mscale)

    quant_config = None
    if kv_cache_dtype == tensorrt_llm.bindings.DataType.FP8:
        quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8.value)

    if is_context_phase:
        attn_mla = attention_cls(
            layer_idx=0,
            num_heads=num_heads,
            head_dim=qk_head_dim,
            num_kv_heads=num_kv_heads,
            quant_config=quant_config,
            q_scaling=q_scaling,
            pos_embd_params=pos_embd_params,
            mla_params=mla_params,
        )
    else:
        attn_mla = attention_cls(
            layer_idx=0,
            num_heads=num_heads,
            head_dim=kv_lora_rank + qk_rope_head_dim,
            num_kv_heads=1,
            quant_config=quant_config,
            q_scaling=q_scaling,
            pos_embd_params=pos_embd_params,
            mla_params=mla_params,
        )
    # NOTE: set up metadata, refer to tensorrt_llm/_torch/pyexecutor/model_engine.py
    # all layers share the same metadata
    mapping = Mapping(world_size=world_size, tp_size=tp_size, rank=0)
    max_tokens = (
        (
            max_context_sequence_length
            + (num_generation_steps + 1) * generation_seq_len_q
            + kv_cache_tokens_per_block
            - 1
        )
        // kv_cache_tokens_per_block
        * kv_cache_tokens_per_block
        * max_num_contexts
    )
    kv_cache_manager = KVCacheManager(
        KvCacheConfig(
            max_tokens=max_tokens,
            enable_block_reuse=False,
        ),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=kv_lora_rank + qk_rope_head_dim,
        tokens_per_block=kv_cache_tokens_per_block,
        max_seq_len=max(context_sequence_lengths) + (num_generation_steps + 1) * generation_seq_len_q,
        max_batch_size=len(context_sequence_lengths),
        mapping=mapping,
        dtype=kv_cache_dtype,
    )

    for req_id, ctx_len in enumerate(context_sequence_lengths):
        req = LlmRequest(
            request_id=req_id,
            max_new_tokens=num_generation_steps + 1,
            input_tokens=[1] * ctx_len,
            sampling_config=SamplingConfig(SamplingParams()._get_sampling_config()),
            is_streaming=False,
        )
        req.paged_kv_block_ids = []
        beam_width = 1
        kv_cache_manager.impl.add_sequence(req_id, ctx_len, beam_width, req)

    attn_metadata = attention_cls.Metadata(
        seq_lens=torch.tensor(context_sequence_lengths, dtype=torch.int),
        request_ids=list(range(len(context_sequence_lengths))),
        max_num_requests=len(context_sequence_lengths),
        num_contexts=len(context_sequence_lengths),
        prompt_lens=context_sequence_lengths,
        max_num_tokens=max(context_sequence_lengths),
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0 for _ in context_sequence_lengths],
        ),
        mapping=mapping,
    )
    attn_metadata.prepare()

    if not is_context_phase:
        for req_id in range(len(context_sequence_lengths)):
            for _ in range(generation_seq_len_q):
                kv_cache_manager.impl.add_token(req_id)
        attn_metadata = attention_cls.Metadata(
            seq_lens=torch.tensor([generation_seq_len_q] * len(context_sequence_lengths), dtype=torch.int),
            request_ids=list(range(len(context_sequence_lengths))),
            max_num_requests=len(context_sequence_lengths),
            num_contexts=0,
            prompt_lens=context_sequence_lengths,
            max_num_tokens=max(context_sequence_lengths),
            kv_cache_manager=kv_cache_manager,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=list(context_sequence_lengths),
            ),
            mapping=mapping,
            enable_flash_mla=torch.cuda.get_device_capability() == (9, 0),
        )
        attn_metadata.prepare()

    if is_context_phase:
        ctx_compressed_kv = torch.randn(
            [ctx_len * len(context_sequence_lengths), kv_lora_rank],
            dtype=dtype,
            device=device,
        )

        ctx_k_pe = torch.randn(
            [ctx_len * len(context_sequence_lengths), qk_rope_head_dim],
            dtype=dtype,
            device=device,
        )

        ctx_q = torch.randn(
            [ctx_len * len(context_sequence_lengths), num_heads * qk_head_dim],
            dtype=dtype,
            device=device,
        )

        ctx_kv = torch.randn(
            [
                ctx_len * len(context_sequence_lengths),
                num_kv_heads * (qk_nope_head_dim + v_head_dim),
            ],
            dtype=dtype,
            device=device,
        )
        # ctx_v.stride(0) == num_kv_heads * (qk_nope_head_dim + v_head_dim)
        ctx_k_nope, ctx_v = ctx_kv.split([num_kv_heads * qk_nope_head_dim, num_kv_heads * v_head_dim], dim=-1)
        ctx_k_nope = ctx_k_nope.view(-1, num_kv_heads, qk_nope_head_dim)
        ctx_k = torch.cat(
            [ctx_k_nope, ctx_k_pe.view(-1, 1, qk_rope_head_dim).expand(-1, num_kv_heads, -1)],
            dim=-1,
        )
        ctx_k = ctx_k.view(-1, num_kv_heads * qk_head_dim)

        q = ctx_q
        k = ctx_k
        v = ctx_v
        compressed_kv = ctx_compressed_kv
        k_pe = ctx_k_pe

        latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)
        attn_mla.forward(
            q,
            k,
            v,
            attn_metadata,
            attention_input_type=AttentionInputType.context_only,
            latent_cache=latent_cache,
        )
    else:
        compressed_kv = torch.randn(
            [generation_seq_len_q * len(context_sequence_lengths), kv_lora_rank],
            dtype=dtype,
            device=device,
        )

        k_pe = torch.randn(
            [generation_seq_len_q * len(context_sequence_lengths), qk_rope_head_dim],
            dtype=dtype,
            device=device,
        )

        fused_q = torch.randn(
            [
                generation_seq_len_q * len(context_sequence_lengths),
                num_heads * (kv_lora_rank + qk_rope_head_dim),
            ],
            dtype=dtype,
            device=device,
        )

        q_pe = torch.randn(
            [generation_seq_len_q * len(context_sequence_lengths), num_heads, qk_rope_head_dim],
            dtype=dtype,
            device=device,
        )

        latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)
        attn_mla.forward(
            fused_q,
            None,
            None,
            attn_metadata,
            attention_input_type=AttentionInputType.generation_only,
            latent_cache=latent_cache,
            q_pe=q_pe,
        )

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        if is_context_phase:
            attn_mla.forward(
                q,
                k,
                v,
                attn_metadata,
                attention_input_type=AttentionInputType.context_only,
                latent_cache=latent_cache,
            )
        else:
            attn_mla.forward(
                fused_q,
                None,
                None,
                attn_metadata,
                attention_input_type=AttentionInputType.generation_only,
                latent_cache=latent_cache,
                q_pe=q_pe,
            )

    # warmup
    for i in range(warming_up):
        g.replay()

    # collect
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(test_ite):
        g.replay()
    end_event.record()
    torch.cuda.synchronize()
    latency = start_event.elapsed_time(end_event) / test_ite

    # write result
    if is_context_phase:
        isl = max_context_sequence_length
        step = 0
    else:
        isl = 1
        step = max_context_sequence_length

    dtype_str = "float16"
    if kv_cache_dtype == tensorrt_llm.bindings.DataType.FP8:
        dtype_str = "fp8"

    log_perf(
        item_list=[
            {
                "mla_dtype": "float16",
                "kv_cache_dtype": dtype_str,
                "num_heads": num_heads,
                "batch_size": len(context_sequence_lengths),
                "isl": isl,
                "tp_size": tp_size,
                "step": step,
                "latency": latency,
            }
        ],
        framework="TRTLLM",
        version=tensorrt_llm.__version__,
        device_name=torch.cuda.get_device_name(device),
        op_name=f"mla_{'context' if is_context_phase else 'generation'}",
        kernel_source="default",
        perf_filename=perf_filename,
    )

    kv_cache_manager.shutdown()
