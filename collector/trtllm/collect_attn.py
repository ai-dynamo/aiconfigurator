# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The retired sm_exceptions "Blackwell rejects GQA ratio >= 32 unless
# divisible by 32" claim (PR #1302) is fully REFUTED on 1.3.0rc20: v3a
# out-of-plan probes (h/kv = 48/1, 40/1, 96/2; hd=128; ctx+gen) pass 6/6
# through framework dispatch on SM100 (B200, 2026-07-18), SM103 (B300,
# 2026-07-19) and SM120 (RTX PRO 6000, 2026-07-19). FIXME(kernel-limit)
# deleted per its lifecycle; never move this back into YAML
# (see .claude/rules/collector/layer_permissions.md).

"""TensorRT-LLM dense attention collector.

Constructs a single TRT-LLM torch-flow attention layer and synthetic metadata to
benchmark context and generation attention. This file owns TRT-LLM cache manager
setup, quantization flags, SM/version-specific skips, and perf-row formatting.
"""

import os

import tensorrt_llm
import torch
from tensorrt_llm._torch.attention_backend import TrtllmAttentionMetadata
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionRuntimeFeatures,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention_backend.utils import create_attention
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

from collector.case_generator import (
    get_attention_context_shape_sweeps,
    get_attention_generation_shape_sweeps,
    get_attention_head_configs,
)
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.registry_types import PerfFile


def _skip_trtllm_sm120_fp8_context_fmha(
    batch_size: int,
    input_len: int,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> bool:
    if not (tensorrt_llm.__version__.startswith(("1.3.0rc5", "1.3.0rc10")) and get_sm_version() >= 120):
        return False

    num_tokens = batch_size * input_len
    if tensorrt_llm.__version__.startswith("1.3.0rc5"):
        return (
            # MHA h=128 max-token cases crash with an illegal memory access in
            # the SM120 FP8 context FMHA kernel.
            (num_heads == num_key_value_heads == 96 and head_dim == 128 and num_tokens == 65536)
            # h=256 cases fail in the qkv_256 SM120 FP8 FMHA kernel.
            or head_dim == 256
        )

    if head_dim != 256:
        return False

    return (
        # TRT-LLM 1.3.0rc10 SM120 qkv_256 FP8 context FMHA crashes with
        # cudaErrorIllegalAddress for these verified high-token regions.
        (num_heads == 96 and num_key_value_heads == 8 and num_tokens >= 81920)
        or (num_heads == 48 and num_key_value_heads == 8 and num_tokens >= 131072)
        or (num_heads == num_key_value_heads == 96 and batch_size >= 2 and input_len >= 16384)
    )


def _skip_trtllm_sm89_rc15_long_context_gqa(
    batch_size: int,
    input_len: int,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> bool:
    # FIXME(kernel-limit): the SM89 long-context GQA IMA family REPRODUCES on
    # 1.3.0rc20 (L40S gate100 2026-07-21: cudaErrorIllegalAddress at
    # h=96/kv=8/hd=256/131072 tokens and h=48/kv=4/hd=256/131072 tokens, both
    # inside this predicate's region). The same IMA family also reproduces on
    # SM90/H20 (2026-07-24 full run: cudaErrorIllegalAddress at
    # h=96/kv=1/hd=128/131072 tokens, both bf16 and fp8-KV), so it is not
    # SM89-exclusive. Kept version-scoped to rc15 by owner decision (SM89
    # handoff): rc20 cases fail classified instead of being skipped; re-verify
    # against framework source on the next version bump.
    if not (tensorrt_llm.__version__.startswith("1.3.0rc15") and get_sm_version() == 89):
        return False

    if num_key_value_heads not in {1, 2, 4, 8}:
        return False

    num_tokens = batch_size * input_len
    if num_heads == 96:
        if head_dim == 128:
            return num_tokens >= 98304
        if head_dim >= 256:
            return num_tokens >= 49152
        return head_dim >= 192 and num_tokens >= 65536
    if num_heads == 64:
        if head_dim >= 256:
            return num_tokens >= 81920
        return head_dim >= 192 and num_tokens >= 98304
    if num_heads == 48:
        if head_dim >= 256:
            return num_tokens >= 98304
        return head_dim >= 192 and num_tokens >= 131072
    if num_heads == 40:
        return head_dim >= 256 and num_tokens >= 131072
    return False


def _skip_trtllm_sm89_rc15_fp8_context_mha(
    batch_size: int,
    input_len: int,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    use_fp8_kv_cache: bool,
    use_fp8_context_fmha: bool,
) -> bool:
    if not (tensorrt_llm.__version__.startswith("1.3.0rc15") and get_sm_version() == 89):
        return False

    if not (use_fp8_kv_cache and use_fp8_context_fmha and num_heads == num_key_value_heads):
        return False

    num_tokens = batch_size * input_len
    if num_heads == 96:
        if head_dim == 128:
            return num_tokens >= 65536
        if head_dim >= 256:
            return num_tokens >= 32768
        return head_dim >= 192 and num_tokens >= 40960
    if num_heads == 64:
        if head_dim >= 256:
            return num_tokens >= 49152
        return head_dim >= 192 and num_tokens >= 65536
    if num_heads == 48:
        return head_dim >= 256 and num_tokens >= 65536
    return False


def run_attention_torch(
    batch_size,
    input_len,
    num_heads,
    num_key_value_heads,  # keep same as num_heads for MHA
    head_dim,
    attention_window_size,
    use_fp8_kv_cache,
    use_fp8_context_fmha,
    is_context_phase,
    attn_backend_name="TRTLLM",
    *,
    perf_filename,
    device="cuda:0",
):
    device = torch.device(device)
    torch.set_default_device(device)
    torch.cuda.set_device(device)

    if attn_backend_name not in {"TRTLLM", "FLASHINFER"}:
        raise ValueError(f"Unsupported TRT-LLM dense attention backend: {attn_backend_name}")
    is_flashinfer = attn_backend_name == "FLASHINFER"
    # FIXME(kernel-limit): the FLASHINFER path pins the trtllm-gen FMHA sub-backend
    # below (attn.flashinfer_backend = "trtllm-gen"), mirroring the sole FLASHINFER-
    # routed model, Gemma4, which forces it unconditionally on every SM
    # (models/modeling_gemma4.py:263-270@1.3.0rc20). trtllm-gen's FMHA runner
    # hard-restricts to Blackwell — TllmGenFmhaRunner asserts
    # ``mSM == kSM_100 || mSM == kSM_103`` ("Unsupported architecture",
    # flashinfer/data/include/flashinfer/trtllm/fmha/fmhaRunner.cuh:37@1.3.0rc20) —
    # and fa2 lacks the head_dim-256 SWA / head_dim-512 cubins, so there is NO dense
    # attention kernel for Gemma4 on Hopper/SM90 (or any non-Blackwell) in rc20;
    # serving hits the identical assertion. Fail closed with a cited, classified skip
    # instead of paying model construction + a CUDA-level abort per case. Re-verify the
    # supported-arch set against fmhaRunner.cuh on the next framework version bump.
    if is_flashinfer and get_sm_version() not in (100, 103):
        raise ValueError(
            f"FlashInfer trtllm-gen FMHA is Blackwell-only (SM100/103); Gemma4 dense "
            f"attention has no kernel on SM{get_sm_version()} "
            f"(fmhaRunner.cuh:37 + modeling_gemma4.py:270 @1.3.0rc20)"
        )
    if is_flashinfer and use_fp8_context_fmha:
        # Mirror of the sglang collector's compute-dtype labeling rule:
        # TRT-LLM 1.3.0rc20 FlashInferAttention has no FP8 FMHA compute path,
        # including under the serving-pinned trtllm-gen sub-backend set below.
        # trtllm-gen's only fp8 handling is the fp8 KV cache: it casts Q to the
        # KV dtype to reuse the QkvE4m3OBfloat16 cubins but the FMHA output is
        # still BF16 (attention_backend/flashinfer.py:1789-1791@1.3.0rc20). The
        # collector's fp8-context-FMHA case (quant_algo=FP8 + out_scale) is a
        # TRTLLM-backend contract flashinfer does not consume, so an fp8-labeled
        # flashinfer row would record BF16 compute under an fp8 label. Fail closed.
        raise ValueError("TRT-LLM 1.3.0rc20 FlashInferAttention has no FP8 FMHA compute path")

    # if XQA JIT is enabled, the context phase will also trigger XQA prepare which causes the error
    # with specifc q/kv head and seq setting.
    if is_context_phase:
        os.environ["TRTLLM_ENABLE_XQA_JIT"] = "0"
    else:
        os.environ["TRTLLM_ENABLE_XQA_JIT"] = "1"

    layer_idx = 0
    world_size = 1
    tp_size = 1
    tokens_per_block = 64
    warming_up = 10
    test_ite = 6
    output_len = 1
    if use_fp8_context_fmha:
        assert use_fp8_kv_cache
        quant_algo = QuantAlgo.FP8
    else:
        quant_algo = None

    # out_scale mirrors serving's Attention._use_quantize_output()
    # (_torch/modules/attention.py:648-670,758-761@1.3.0rc20): an fp8-quantized
    # model supplies o_proj.inv_input_scale; a model with no weight/activation
    # quant (``not has_any_quant(exclude_kv_cache=True)``) supplies None, fp8 KV
    # cache or not. Passing a scale is not inert: the backend allocates fp8
    # attention output whenever out_scale is present (is_quantize_output,
    # attention_backend/trtllm.py:1452) and the op then takes the fp8-output path
    # (is_fp8_out, thop/attentionOp.cpp:1155:
    # ``is_fp8_out || is_fp4_out || (hasFp8KvCache() && use_paged_context_fmha)``).
    # Per phase:
    # - context: the fp8-model flavor is the dedicated use_fp8_context_fmha case,
    #   so the fp8-KV-only case means "unquantized model + fp8 KV" -> None. On
    #   SM90 the framework then aborts on its own ("attention output scale should
    #   be provided", common/attentionOp.cpp:681) because fp8-KV plus the forced
    #   paged-context FMHA (trtllm.py:1434, nvbug 5624818) select the fp8-output
    #   path with no scale — serving aborts identically for that config on
    #   SM90/rc20, so the case fails classified rather than being measured under
    #   the wrong label (the row records attn_dtype=bfloat16).
    # - generation: the sweep has no separate fp8-model case, so the fp8-KV row
    #   is the fp8-model deployment; its serving path provides the scale and
    #   writes fp8 attention output on every SM. A dummy unit scale mirrors
    #   o_proj.inv_input_scale for the bare-layer benchmark.
    if use_fp8_context_fmha or (use_fp8_kv_cache and not is_context_phase):
        out_scale = torch.tensor(
            [1.0],
            dtype=torch.float32,
            device=device,
        )
    else:
        out_scale = None

    if use_fp8_kv_cache:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.FP8
    else:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16

    # Serving hands pos_embd_params to create_attention only when the backend
    # fuses RoPE (modules/attention.py:601-610@1.3.0rc20). FlashInferAttention
    # does not (support_fused_rope=False, attention_backend/interface.py:976-977),
    # so serving applies RoPE outside the backend and a flashinfer attention
    # row excludes RoPE — mirrored here.
    if is_flashinfer:
        pos_embd_params = None
    else:
        pos_embd_params = PositionalEmbeddingParams(type=PositionEmbeddingType.rope_gpt_neox, rope=RopeParams(dim=128))

    quant_config = QuantConfig(
        quant_algo=quant_algo,  # fp8 fmha
        kv_cache_quant_algo=QuantAlgo.FP8 if use_fp8_kv_cache else None,  # fp8 kv,
        group_size=128,
        smoothquant_val=0.5,
        clamp_val=None,
        use_meta_recipe=False,
        has_zero_point=False,
        pre_quant_scale=False,
        exclude_modules=None,
    )

    attn = create_attention(
        backend_name=attn_backend_name,
        layer_idx=layer_idx,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_key_value_heads,
        pos_embd_params=pos_embd_params,
        quant_config=quant_config,
        is_mla_enable=False,
    )
    # get_attention_backend silently falls back to TrtllmAttention when the
    # requested backend is unavailable (attention_backend/utils.py:52-53
    # @1.3.0rc20, e.g. flashinfer not importable). Serving tolerates that with
    # a warning; a collector must not — the row would be labeled with a
    # backend that never ran. Verify the constructed class.
    expected_attn_cls = {"TRTLLM": "TrtllmAttention", "FLASHINFER": "FlashInferAttention"}[attn_backend_name]
    if type(attn).__name__ != expected_attn_cls:
        raise RuntimeError(
            f"create_attention resolved {type(attn).__name__} for backend {attn_backend_name} "
            f"(expected {expected_attn_cls}); refusing the silent fallback"
        )

    if is_flashinfer:
        # create_attention only selects the backend CLASS; it does not thread
        # the flashinfer *sub-backend*, so FlashInferAttention defaults it to
        # "fa2" (attention_backend/flashinfer.py:1372@1.3.0rc20). Serving does
        # not run fa2 for any FLASHINFER-routed model: the sole such model,
        # Gemma4, pins ``self.attn.flashinfer_backend = "trtllm-gen"`` for ALL
        # layers (models/modeling_gemma4.py:263-270@1.3.0rc20) because trtllm-gen
        # carries the pre-compiled cubins for both the head_dim-256 SWA and
        # head_dim-512 global geometries. Without this pin the collector would
        # benchmark fa2 — a kernel serving never invokes — yielding silently
        # wrong rows (e.g. hd256 SWA "passing" on fa2 while serving's trtllm-gen
        # rejects the architecture). Mirror serving's pin exactly; the attribute
        # flows into every metadata.plan() call (flashinfer.py:1848,1874).
        attn.flashinfer_backend = "trtllm-gen"

    total_num_tokens = (input_len + output_len) * batch_size

    mapping = Mapping(world_size=world_size, rank=0, tp_size=tp_size)

    num_hidden_layers = 1

    synthetic_max_seq_len = input_len + output_len + 1
    if attention_window_size > 0:
        synthetic_max_seq_len = max(synthetic_max_seq_len, attention_window_size + output_len + 1)

    kv_cache_config = KvCacheConfig(
        max_tokens=int((synthetic_max_seq_len - 1) / tokens_per_block + 1)
        * tokens_per_block
        * batch_size
        * 2,  # num_bloacks * block_size
        enable_block_reuse=False,
    )

    # Serving allocates dense-attention KV cache as CacheType.SELF
    # (pyexecutor/_util.py:1539@1.3.0rc20); SELFKONLY is reserved for MLA
    # (_util.py:1636). Regular GQA/MHA has separate K and V tensors, so both
    # TRTLLM and FlashInfer paths must benchmark the full serving layout.
    kv_cache_type = tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF
    kv_cache_manager = KVCacheManager(
        kv_cache_config=kv_cache_config,
        kv_cache_type=kv_cache_type,
        num_layers=num_hidden_layers,
        num_kv_heads=num_key_value_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=synthetic_max_seq_len,  # +1 for the magic fixme mentioned in trtllm xqa JIT path impl.
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=kv_cache_dtype,
    )

    input_seq_lens = [input_len for _ in range(batch_size)]
    total_seq_lens = [input_len + output_len for _ in range(batch_size)]
    request_ids = list(range(batch_size))
    kv_cache_manager.add_dummy_requests(request_ids, total_seq_lens)

    if is_context_phase:
        metadata_kwargs = dict(
            seq_lens=torch.tensor(input_seq_lens, dtype=torch.int32, device="cpu"),
            num_contexts=batch_size,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[0 for _ in range(batch_size)],
                block_ids_per_seq=None,
                host_max_attention_window_sizes=None,
                host_sink_token_length=None,
            ),
            runtime_features=AttentionRuntimeFeatures(
                chunked_prefill=False, cache_reuse=False, has_speculative_draft_tokens=False
            ),
        )
    else:
        gen_seq_lens = [1 for _ in range(batch_size)]
        metadata_kwargs = dict(
            seq_lens=torch.tensor(gen_seq_lens, dtype=torch.int32, device="cpu"),
            num_contexts=0,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=input_seq_lens,
                block_ids_per_seq=None,
                host_max_attention_window_sizes=None,
                host_sink_token_length=None,
            ),
            runtime_features=AttentionRuntimeFeatures(chunked_prefill=False, cache_reuse=False),
        )
    metadata_kwargs.update(
        max_num_requests=batch_size,
        max_num_tokens=total_num_tokens,
        kv_cache_manager=kv_cache_manager,
        mapping=mapping,
        enable_flash_mla=False,
        position_ids=None,
        cross=None,
        request_ids=request_ids,
        prompt_lens=input_seq_lens,
        all_rank_num_tokens=None,
    )
    if is_flashinfer:
        # Serving builds metadata from the selected backend's Metadata class
        # with backend-agnostic kwargs (pyexecutor/model_engine.py:1784,
        # 1818-1830@1.3.0rc20); ``workspace`` is a TrtllmAttentionMetadata-only
        # field, flashinfer manages its own workspace_buffer.
        from tensorrt_llm._torch.attention_backend.flashinfer import FlashInferAttentionMetadata

        attn_metadata = FlashInferAttentionMetadata(**metadata_kwargs)
    else:
        attn_metadata = TrtllmAttentionMetadata(
            **metadata_kwargs,
            workspace=torch.tensor([], device=device, dtype=torch.int8),
        )

    attn_metadata.prepare()

    if is_context_phase:
        num_tokens = input_len * batch_size
    else:
        num_tokens = batch_size

    sinks = torch.randn(num_heads, dtype=torch.float32) if head_dim == 64 else None
    q = torch.randn([num_tokens, num_heads * head_dim]).bfloat16().to(torch.device(device))
    k = torch.randn([num_tokens, num_key_value_heads * head_dim]).bfloat16().to(torch.device(device))
    v = torch.randn([num_tokens, num_key_value_heads * head_dim]).bfloat16().to(torch.device(device))
    if is_flashinfer:
        # Serving splits Q/K/V for backends without fused-QKV support
        # (modules/attention.py:619,641-646@1.3.0rc20; FlashInferAttention
        # support_fused_qkv=False, attention_backend/interface.py:980-981).
        # Sinks and out_scale are TRTLLM-backend contracts
        # (modules/attention.py:949; fp8 FMHA is refused above), and sink
        # profiles never route here (head_dim == 64 has no flashinfer pin).
        if sinks is not None:
            raise ValueError("attention sinks are a TRTLLM-backend contract (modules/attention.py:949)")
        forward_args = (q, k, v)
        forward_kwargs = {
            "attention_window_size": attention_window_size if attention_window_size > 0 else None,
        }
    else:
        forward_args = (torch.concat([q, k, v], dim=-1), None, None)
        forward_kwargs = {
            "attention_window_size": attention_window_size if attention_window_size > 0 else None,
            "attention_sinks": sinks,
            "out_scale": out_scale,
        }
    attn.forward(*forward_args, attn_metadata, **forward_kwargs)

    # Use benchmark_with_power context manager
    def kernel_func():
        attn.forward(*forward_args, attn_metadata, **forward_kwargs)

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=warming_up,
        num_runs=test_ite,
        repeat_n=1,
    ) as results:
        pass

    latency = results["latency_ms"]

    # write result
    if is_context_phase:
        isl = input_len
        step = 0
        op_name = "context_attention"
    else:
        isl = 1
        step = input_len
        op_name = "generation_attention"
    kv_cache_dtype_str = "bfloat16"
    if use_fp8_kv_cache:
        kv_cache_dtype_str = "fp8"
    if use_fp8_context_fmha:
        dtype_str = "fp8"
    else:
        dtype_str = "bfloat16"

    log_perf(
        item_list=[
            {
                "batch_size": batch_size,
                "isl": isl,
                "num_heads": num_heads,
                "num_key_value_heads": num_key_value_heads,
                "head_dim": head_dim,
                "window_size": attention_window_size,
                "beam_width": 1,
                "attn_dtype": dtype_str,
                "kv_cache_dtype": kv_cache_dtype_str,
                "step": step,
                "latency": latency,
            }
        ],
        framework="TRTLLM",
        version=tensorrt_llm.__version__,
        device_name=torch.cuda.get_device_name(device),
        op_name=op_name,
        kernel_source="torch_flow_flashinfer" if is_flashinfer else "torch_flow",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )
    kv_cache_manager.shutdown()


def _int_list(values):
    return [int(value) for value in values]


def get_context_attention_test_cases():
    has_fp8 = get_sm_version() > 86
    test_cases = []

    for shape_sweep in get_attention_context_shape_sweeps("trtllm"):
        batch_sizes = _int_list(shape_sweep["batch_sizes"])
        sequence_lengths = _int_list(shape_sweep["sequence_lengths"])
        max_tokens_self_attention = int(shape_sweep["max_tokens_self_attention"])
        max_tokens_grouped_query_attention = int(shape_sweep["max_tokens_grouped_query_attention"])
        max_batch_size_self_attention = int(shape_sweep["max_batch_size_self_attention"])
        max_kv_elements = int(shape_sweep["max_kv_elements"])

        for head_config in get_attention_head_configs(shape_sweep, phase="context", backend="trtllm"):
            n = head_config.num_heads
            num_kv_heads = head_config.num_kv_heads
            h = head_config.head_dim
            attention_window_size = head_config.window_size
            attn_backend = head_config.kernel_source or "TRTLLM"
            if num_kv_heads != n and (num_kv_heads >= n or n % num_kv_heads != 0):
                continue

            for s in sorted(sequence_lengths, reverse=True):
                for b in sorted(batch_sizes, reverse=True):
                    if num_kv_heads == n:
                        if b * s > max_tokens_self_attention or b > max_batch_size_self_attention:
                            continue
                    else:
                        if b * s > max_tokens_grouped_query_attention:
                            continue
                    if b * s * num_kv_heads * h * 2 >= max_kv_elements:
                        continue
                    # The XQA-prepare precheck below is a TRTLLM-kernel init
                    # constraint; flashinfer-routed configs never run it.
                    if get_sm_version() >= 100 and attn_backend == "TRTLLM":
                        # though it's a precheck of gen kernels during the attention op init,
                        # this cannot be skipped for now
                        # TLLM_CHECK_WITH_INFO((params.m_num_heads_q_per_kv < max_num_heads_q_per_kv_in_cta || params.m_num_heads_q_per_kv % max_num_heads_q_per_kv_in_cta == 0), # noqa: E501
                        m_num_heads_q_per_kv = n // num_kv_heads
                        max_num_heads_q_per_kv_in_cta = 32
                        if (
                            m_num_heads_q_per_kv >= max_num_heads_q_per_kv_in_cta
                            and m_num_heads_q_per_kv % max_num_heads_q_per_kv_in_cta != 0
                        ):
                            continue

                    skip_fp8_context_fmha = _skip_trtllm_sm120_fp8_context_fmha(
                        b,
                        s,
                        n,
                        num_kv_heads,
                        h,
                    )
                    if _skip_trtllm_sm89_rc15_long_context_gqa(b, s, n, num_kv_heads, h):
                        continue
                    for precision_case in shape_sweep["precision_cases"]:
                        use_fp8_kv_cache = bool(precision_case["fp8_kv_cache"])
                        use_fp8_context_fmha = bool(precision_case["fp8_context_fmha"])
                        if not has_fp8 and use_fp8_kv_cache:
                            continue
                        if skip_fp8_context_fmha and use_fp8_context_fmha:
                            continue
                        if _skip_trtllm_sm89_rc15_fp8_context_mha(
                            b,
                            s,
                            n,
                            num_kv_heads,
                            h,
                            use_fp8_kv_cache,
                            use_fp8_context_fmha,
                        ):
                            continue
                        test_cases.append(
                            [
                                b,
                                s,
                                n,
                                num_kv_heads,
                                h,
                                attention_window_size,
                                use_fp8_kv_cache,
                                use_fp8_context_fmha,
                                True,
                                attn_backend,
                            ]
                        )

    return test_cases


def _generation_target_sequence_lengths(batch_sizes, sequence_lengths, num_heads, max_tokens, shape_sweep):
    b_s_dict = {}
    s_b_dict = {}
    for s in sequence_lengths:
        max_b = max_tokens // s // num_heads
        for b in batch_sizes:
            if b > max_b:
                break
            if s not in s_b_dict:
                s_b_dict[s] = {b}
            else:
                s_b_dict[s].add(b)
    for s, b_set in s_b_dict.items():
        if len(b_set) < int(shape_sweep["min_batch_options_per_sequence"]):
            continue
        for b in b_set:
            if b not in b_s_dict:
                b_s_dict[b] = {s - 1}
            b_s_dict[b].add(s - 1)
    return b_s_dict


def get_generation_attention_test_cases():
    has_fp8 = get_sm_version() > 86
    test_cases = []

    for shape_sweep in get_attention_generation_shape_sweeps("trtllm"):
        batch_sizes = _int_list(shape_sweep["batch_sizes"])
        sequence_lengths = _int_list(shape_sweep["sequence_lengths"])
        min_drop_batch = int(shape_sweep["drop_largest_sequence_for_batch_at_least"])

        for head_config in get_attention_head_configs(shape_sweep, phase="generation", backend="trtllm"):
            n = head_config.num_heads
            n_kv = head_config.num_kv_heads
            h = head_config.head_dim
            attention_window_size = head_config.window_size
            attn_backend = head_config.kernel_source or "TRTLLM"
            max_tokens_key = "max_mha_tokens_per_step" if n == n_kv else "max_xqa_tokens_per_step"
            b_s_dict = _generation_target_sequence_lengths(
                batch_sizes,
                sequence_lengths,
                n,
                int(shape_sweep[max_tokens_key]),
                shape_sweep,
            )
            # The XQA precheck is a TRTLLM-kernel init constraint; flashinfer-
            # routed configs never run it.
            if n_kv != n and get_sm_version() >= 100 and attn_backend == "TRTLLM":
                # TLLM_CHECK_WITH_INFO((params.m_num_heads_q_per_kv < max_num_heads_q_per_kv_in_cta || params.m_num_heads_q_per_kv % max_num_heads_q_per_kv_in_cta == 0), # noqa: E501
                m_num_heads_q_per_kv = n // n_kv
                max_num_heads_q_per_kv_in_cta = 32
                if (
                    m_num_heads_q_per_kv >= max_num_heads_q_per_kv_in_cta
                    and m_num_heads_q_per_kv % max_num_heads_q_per_kv_in_cta != 0
                ):
                    continue
            for b, s_list_limited in b_s_dict.items():
                target_s_list = sorted(s_list_limited)
                if b >= min_drop_batch:
                    target_s_list = target_s_list[:-1]
                for s in target_s_list:
                    for precision_case in shape_sweep["precision_cases"]:
                        use_fp8_kv_cache = bool(precision_case["fp8_kv_cache"])
                        if not has_fp8 and use_fp8_kv_cache:
                            continue
                        test_cases.append(
                            [b, s, n, n_kv, h, attention_window_size, use_fp8_kv_cache, False, False, attn_backend]
                        )
    return test_cases


if __name__ == "__main__":
    test_cases = get_context_attention_test_cases()
    for test_case in test_cases:
        run_attention_torch(*test_case, perf_filename=PerfFile.CONTEXT_ATTENTION)

    test_cases = get_generation_attention_test_cases()
    for test_case in test_cases:
        run_attention_torch(*test_case, perf_filename=PerfFile.GENERATION_ATTENTION)
