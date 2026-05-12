# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__compat__ = "sglang>=0.5.10rc0"

import math
import os
import random

import pkg_resources
import sglang.srt.layers.dp_attention
import sglang.srt.server_args
import torch
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.registry_types import PerfFile

# Mocking for standalone collector script
# sglang >=0.5.10 removed _ATTN_TP_SIZE/_ATTN_TP_RANK/_ATTN_TP_GROUP from dp_attention;
# get_attention_tp_size() now delegates to sglang.srt.distributed. Set the remaining
# private variables for older versions and always override the public functions.
if hasattr(sglang.srt.layers.dp_attention, "_ATTN_TP_SIZE"):
    sglang.srt.layers.dp_attention._ATTN_TP_SIZE = 1
    sglang.srt.layers.dp_attention._ATTN_TP_RANK = 0
sglang.srt.layers.dp_attention._ATTN_DP_SIZE = 1
sglang.srt.layers.dp_attention._ATTN_DP_RANK = 0
sglang.srt.layers.dp_attention._LOCAL_ATTN_DP_SIZE = 1
sglang.srt.layers.dp_attention._LOCAL_ATTN_DP_RANK = 0
sglang.srt.layers.dp_attention.get_attention_tp_size = lambda: 1
sglang.srt.layers.dp_attention.get_attention_tp_rank = lambda: 0
sglang.srt.layers.dp_attention.get_attention_dp_size = lambda: 1
sglang.srt.layers.dp_attention.get_attention_dp_rank = lambda: 0
# Patch imported versions in other modules
import sglang.srt.layers.attention.flashinfer_mla_backend
import sglang.srt.layers.attention.triton_backend
import sglang.srt.layers.attention.trtllm_mla_backend

sglang.srt.layers.attention.flashinfer_mla_backend.get_attention_tp_size = lambda: 1
sglang.srt.layers.attention.triton_backend.get_attention_tp_size = lambda: 1
sglang.srt.layers.attention.trtllm_mla_backend.get_attention_tp_size = lambda: 1

DISABLE_BACKWARD = os.getenv("FLASH_ATTENTION_DISABLE_BACKWARD", "FALSE") == "TRUE"

# Default DeepSeek MLA dims (non-wide): latent=512, rope=64 (query=576, value=512).
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128  # DeepSeek configs
QK_ROPE_HEAD_DIM = 64
MLA_PAGE_SIZE = 64
# Scaling follows production: 1 / sqrt(qk_nope + qk_rope)
MLA_SCALING = 1 / math.sqrt(QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)
INT32_MAX = 2**31 - 1
# Largest kv slot index we can safely touch before the old flashmla kernels overflow
MAX_KV_LOC = (INT32_MAX // (KV_LORA_RANK + QK_ROPE_HEAD_DIM)) - MLA_PAGE_SIZE

# We only cover deepseek v3 in this collector script.


def _cuda_version_at_least(major: int, minor: int) -> bool:
    if torch.version.cuda is None:
        return False
    version = tuple(int(part) for part in torch.version.cuda.split(".")[:2])
    return version >= (major, minor)


def _select_default_mla_backend() -> str:
    """Match SGLang 0.5.10's default MLA backend for DeepSeek V3."""
    sm_version = get_sm_version()
    if 100 <= sm_version < 110 and _cuda_version_at_least(12, 8):
        # DeepSeek V3/R1/V3.1 special-case in SGLang server_args.py.
        return "trtllm_mla"
    if 90 <= sm_version < 100 and _cuda_version_at_least(12, 3):
        return "fa3"
    return "triton"


class MockModelConfig:
    def __init__(
        self,
        context_len: int = 32768,
        num_attention_heads: int = 128,
        kv_lora_rank: int = KV_LORA_RANK,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 512,
        scaling: float = 1.0,
    ):
        self.is_encoder_decoder = False
        self.context_len = context_len
        self.attention_arch = AttentionArch.MLA
        self.is_hybrid = False
        self.attention_chunk_size = None
        # Provide compatibility with newer sglang versions that expect hybrid-SWA metadata
        self.is_hybrid_swa = None
        self.swa_attention_layer_ids = None
        self.full_attention_layer_ids = None
        self.num_attention_heads = num_attention_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.scaling = scaling
        self.is_local_attention_model = False

    def get_num_kv_heads(self, tp_size: int):
        return 1


class MockServerArgs:
    def __init__(self, kv_cache_dtype: torch.dtype, page_size: int):
        self.enable_lora = False
        self.enable_deterministic_inference = False
        self.kv_cache_dtype = "fp8" if kv_cache_dtype == torch.float8_e4m3fn else "bfloat16"
        self.speculative_eagle_topk = 0
        self.speculative_num_draft_tokens = 0
        self.speculative_num_steps = 0
        self.speculative_attention_mode = "prefill"
        self.attention_backend = "fa3"
        self.prefill_attention_backend = "fa3"
        self.decode_attention_backend = "fa3"
        self.page_size = page_size
        self.device = "cuda"
        self.disable_chunked_prefix_cache = True
        self.disaggregation_mode = None
        self.flashinfer_mla_disable_ragged = False
        self.chunked_prefill_size = -1
        self.triton_attention_num_kv_splits = 8
        self.triton_attention_split_tile_size = None
        # sglang >=0.5.10: FlashAttentionBackend.__init__ reads disable_cuda_graph
        self.disable_cuda_graph = True


class MockModelRunner:
    def __init__(
        self,
        device: torch.device,
        kv_cache_dtype: torch.dtype,
        page_size: int,
        num_attention_heads: int = 128,
        scaling: float = 1.0,
    ):
        self.device = device
        self.gpu_id = device.index if device.index is not None else torch.cuda.current_device()
        self.kv_cache_dtype = kv_cache_dtype
        self.dtype = torch.bfloat16
        self.page_size = page_size
        self.req_to_token_pool = None
        self.token_to_kv_pool = None
        self.token_to_kv_pool_allocator = None
        self.attn_backend = None
        self.sliding_window_size = None
        self.is_hybrid = False
        self.hybrid_gdn_config = None
        self.kimi_linear_config = None
        self.model_config = MockModelConfig(num_attention_heads=num_attention_heads, scaling=scaling)
        # Keep attributes for compatibility across sglang versions (older code ignores them)
        self.is_hybrid_swa = self.model_config.is_hybrid_swa
        self.attn_cp_size = 1  # Context parallelism size; required by FlashAttentionBackend in sglang >=0.5.10
        self.server_args = MockServerArgs(kv_cache_dtype, page_size)
        self.use_mla_backend = True


def create_req_to_token_pool(
    batch_size: int,
    total_len: int,
    page_size: int,
    torch_device: torch.device,
    device_str: str,
) -> tuple[ReqToTokenPool, torch.Tensor]:
    pool = ReqToTokenPool(
        size=batch_size,
        max_context_len=total_len,
        device=device_str,
        enable_memory_saver=False,
    )
    req_indices = torch.arange(batch_size, dtype=torch.int32, device=torch_device).view(batch_size, 1)
    token_offsets = torch.arange(total_len, dtype=torch.int32, device=torch_device).view(1, total_len)
    token_matrix = (req_indices * total_len) + token_offsets + page_size
    pool.req_to_token[:batch_size, :total_len] = token_matrix
    return pool, token_matrix.contiguous()


def benchmark_layer(layer, forward_batch, q, k, v, q_rope, k_rope, **kwargs):
    # Use benchmark_with_power context manager
    device = q.device

    def kernel_func():
        extra_kwargs = dict(kwargs)
        if q_rope is not None:
            extra_kwargs["q_rope"] = q_rope
        if k_rope is not None:
            extra_kwargs["k_rope"] = k_rope
        layer(q, k, v, forward_batch, **extra_kwargs)

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=3,
        num_runs=20,
        repeat_n=1,
    ) as results:
        pass

    return results["latency_ms"], results["power_stats"]


def get_context_mla_test_cases():
    # This collector covers the CUDA MLA backends used by SGLang defaults on SM90+.
    sm_version = get_sm_version()
    if sm_version < 90:
        return []

    backend = _select_default_mla_backend()
    dtype_list = [torch.bfloat16] if backend == "triton" else [torch.bfloat16, torch.float8_e4m3fn]
    test_cases = []
    n_list = [64, 128]
    b_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    s_list = [1, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 32768]
    for n in n_list:
        for b in b_list:
            for s in s_list:
                for dtype in dtype_list:
                    for tp_size in [1, 2, 4, 8, 16, 32, 64]:
                        if b * s > 65536:
                            continue
                        test_cases.append(
                            [
                                s,
                                b,
                                1,
                                dtype,
                                n,
                                tp_size,
                                tp_size,
                                64,
                                10,
                                6,
                                True,
                            ]
                        )
    return test_cases


def get_generation_mla_test_cases():
    # This collector covers the CUDA MLA backends used by SGLang defaults on SM90+.
    sm_version = get_sm_version()
    if sm_version < 90:
        return []

    backend = _select_default_mla_backend()
    if backend == "triton":
        # SGLang's Triton MLA path stores BF16 MLA KV cache.
        dtype_list = [torch.bfloat16]
    else:
        dtype_list = [torch.bfloat16, torch.float8_e4m3fn]
    test_cases = []
    n_list = [64, 128]
    for n in n_list:
        for b in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            for s in [
                2,
                4,
                8,
                16,
                32,
                64,
                128,
                256,
                512,
                1024,
                2048,
                4096,
                8192,
                16384,
                32768,
                65536,
                131072,
            ]:
                for dtype in dtype_list:
                    for tp_size in [1, 2, 4, 8, 16, 32, 64]:
                        if backend == "trtllm_mla" and sm_version == 120 and n // tp_size != 128:
                            # XQA MLA kernel has head_group_ratio=128 hardcoded; it always reads
                            # 128 Q heads from the q tensor regardless of the runtime head count.
                            # local_heads != 128 causes out-of-bounds GPU memory reads and crashes.
                            # Only n=128, tp=1 (local_heads=128) is safe.
                            continue
                        if b * s > 1024 * 4096 * 4:
                            continue
                        total_len = s
                        # Guard against hitting int32 limits in the legacy flashmla kernel path.
                        if (b * total_len) + MLA_PAGE_SIZE > MAX_KV_LOC:
                            continue
                        test_cases.append(
                            [
                                s - 1,
                                b,
                                1,
                                dtype,
                                n,
                                tp_size,
                                tp_size,
                                64,
                                10,
                                6,
                                False,
                            ]
                        )
    return test_cases


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
    *,
    perf_filename,
    device="cuda:0",
):
    torch.cuda.set_device(device)
    torch_device = torch.device(device)
    random.seed(0)
    torch.manual_seed(0)
    del world_size, tokens_per_block, warming_up, test_ite, output_len

    assert kv_cache_dtype in [torch.bfloat16, torch.float8_e4m3fn], "Unsupported kv cache dtype"
    assert num_heads % tp_size == 0, "num_heads must be divisible by tp_size"
    local_num_heads = num_heads // tp_size

    model_runner = MockModelRunner(
        torch_device,
        kv_cache_dtype,
        MLA_PAGE_SIZE,
        num_attention_heads=num_heads,
        scaling=MLA_SCALING,
    )
    total_len = input_len if is_context_phase else input_len + 1
    req_to_token_pool, token_matrix = create_req_to_token_pool(
        batch_size=batch_size,
        total_len=total_len,
        page_size=MLA_PAGE_SIZE,
        torch_device=torch_device,
        device_str=str(torch_device),
    )
    model_runner.req_to_token_pool = req_to_token_pool

    selected_backend = _select_default_mla_backend()
    model_runner.server_args.attention_backend = selected_backend
    model_runner.server_args.prefill_attention_backend = selected_backend
    model_runner.server_args.decode_attention_backend = selected_backend
    # Set global args after potential overrides.
    sglang.srt.server_args.set_global_server_args_for_scheduler(model_runner.server_args)

    # Define dimensions based on phase
    kv_lora_rank = KV_LORA_RANK
    qk_rope_head_dim = QK_ROPE_HEAD_DIM
    qk_nope_head_dim = QK_NOPE_HEAD_DIM

    if selected_backend == "trtllm_mla":
        if is_context_phase:
            # Prefill: Non-absorbed, standard projected heads
            # q_nope (128) + q_rope (64) = 192
            v_head_dim = qk_nope_head_dim
            head_dim_total = qk_nope_head_dim + qk_rope_head_dim
        else:
            # Decode: Weight absorbed
            # latent (512) + rope (64) = 576
            v_head_dim = kv_lora_rank
            head_dim_total = kv_lora_rank + qk_rope_head_dim
    else:
        v_head_dim = kv_lora_rank
        head_dim_total = kv_lora_rank + qk_rope_head_dim

    # Keep model_config consistent with chosen dims
    # Must update config BEFORE creating attn_backend so it picks up the right v_head_dim
    model_runner.model_config.kv_lora_rank = kv_lora_rank
    model_runner.model_config.v_head_dim = v_head_dim
    model_runner.model_config.qk_nope_head_dim = qk_nope_head_dim
    model_runner.model_config.qk_rope_head_dim = qk_rope_head_dim
    model_runner.model_config.scaling = MLA_SCALING

    total_tokens = max(1, batch_size * total_len)
    kv_cache_size = max(MLA_PAGE_SIZE, math.ceil(total_tokens / MLA_PAGE_SIZE) * MLA_PAGE_SIZE)
    kv_pool = MLATokenToKVPool(
        size=kv_cache_size,
        page_size=MLA_PAGE_SIZE,
        dtype=kv_cache_dtype,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        layer_num=1,
        device=str(torch_device),
        enable_memory_saver=False,
    )
    model_runner.token_to_kv_pool = kv_pool

    if selected_backend == "trtllm_mla":
        # TRTLLMMLABackend inherits FlashInferMLAAttnBackend which creates
        # FlashInferMLAIndicesUpdaterDecode(model_runner, self) — a cyclic reference.
        # Without explicit GC, previous backends accumulate and corrupt shared workspace state.
        import gc

        gc.collect()
        attn_backend = TRTLLMMLABackend(model_runner)
        kernel_source = "trtllm_mla"
    elif selected_backend == "triton":
        attn_backend = TritonAttnBackend(model_runner)
        kernel_source = "triton"
    else:
        # Hopper: use FA3-compatible FlashAttention path for MLA.
        attn_backend = FlashAttentionBackend(model_runner)
        kernel_source = "flash_attention"

    layer = RadixAttention(
        num_heads=local_num_heads,
        head_dim=head_dim_total,
        scaling=MLA_SCALING,
        num_kv_heads=1,
        layer_id=0,
        v_head_dim=v_head_dim,
    ).to(torch_device)

    req_pool_indices = torch.arange(batch_size, dtype=torch.int32, device=torch_device)
    q_rope_arg = None
    k_rope_arg = None

    if is_context_phase:
        seq_lens = torch.full((batch_size,), input_len, dtype=torch.int32, device=torch_device)
        prefix_lens = torch.zeros_like(seq_lens)
        # TRTLLM/FlashInfer paths use projected heads here; Triton uses latent KV_LORA_RANK heads.
        q_shape = (batch_size * input_len, local_num_heads, v_head_dim)
        q_nope = torch.randn(q_shape, device=torch_device, dtype=torch.bfloat16)
        q_rope = torch.randn(
            batch_size * input_len,
            local_num_heads,
            qk_rope_head_dim,
            device=torch_device,
            dtype=torch.bfloat16,
        )
        k_shape = (batch_size * input_len, 1, v_head_dim)
        k_nope = torch.randn(k_shape, device=torch_device, dtype=torch.bfloat16)
        k_rope = torch.randn(
            batch_size * input_len,
            1,
            qk_rope_head_dim,
            device=torch_device,
            dtype=torch.bfloat16,
        )
        # v has the same head dimension as the non-rope K fragment for each backend path.
        v = k_nope
        if kernel_source == "triton":
            q = torch.cat([q_nope, q_rope], dim=-1)
            k = torch.cat([k_nope, k_rope], dim=-1)
        else:
            q = q_nope
            k = k_nope
            q_rope_arg = q_rope
            k_rope_arg = k_rope

        positions = torch.cat([torch.arange(input_len, device=torch_device) for _ in range(batch_size)])

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=batch_size,
            input_ids=torch.zeros(batch_size, input_len, dtype=torch.long, device=torch_device),
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=token_matrix.reshape(-1).to(torch.int32),
            seq_lens_sum=int(seq_lens.sum().item()),
            seq_lens_cpu=seq_lens.cpu(),
            extend_seq_lens=seq_lens,
            extend_prefix_lens=prefix_lens,
            extend_seq_lens_cpu=seq_lens.cpu().tolist(),
            extend_prefix_lens_cpu=prefix_lens.cpu().tolist(),
            extend_num_tokens=int(seq_lens.sum().item()),
            positions=positions,
        )
    else:
        history_len = input_len
        seq_lens = torch.full((batch_size,), history_len + 1, dtype=torch.int32, device=torch_device)
        positions = torch.full((batch_size,), history_len, device=torch_device)

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=batch_size,
            input_ids=torch.zeros(batch_size, 1, dtype=torch.long, device=torch_device),
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=token_matrix[:, history_len:].reshape(-1).to(torch.int32),
            seq_lens_sum=int(seq_lens.sum().item()),
            seq_lens_cpu=seq_lens.cpu(),
            positions=positions,
        )

        if history_len > 0:
            history_loc = token_matrix[:, :history_len].reshape(-1).contiguous()
            cache_k = torch.randn(
                history_loc.numel(),
                1,
                kv_lora_rank,
                device=torch_device,
                dtype=torch.bfloat16,
            )
            cache_k_rope = torch.randn(
                history_loc.numel(),
                1,
                qk_rope_head_dim,
                device=torch_device,
                dtype=torch.bfloat16,
            )
            kv_pool.set_mla_kv_buffer(
                layer,
                history_loc.to(torch.int64),
                cache_k,
                cache_k_rope,
            )

        q_nope = torch.randn(batch_size, local_num_heads, v_head_dim, device=torch_device, dtype=torch.bfloat16)
        q_rope = torch.randn(
            batch_size,
            local_num_heads,
            qk_rope_head_dim,
            device=torch_device,
            dtype=torch.bfloat16,
        )
        k_nope = torch.randn(batch_size, 1, v_head_dim, device=torch_device, dtype=torch.bfloat16)
        k_rope = torch.randn(batch_size, 1, qk_rope_head_dim, device=torch_device, dtype=torch.bfloat16)
        v = k_nope
        q_nope = q_nope.view(batch_size * 1, local_num_heads, v_head_dim)
        q_rope = q_rope.view(batch_size * 1, local_num_heads, qk_rope_head_dim)
        if kernel_source == "triton":
            q = torch.cat([q_nope, q_rope], dim=-1)
            k = torch.cat([k_nope, k_rope], dim=-1)
        else:
            q = q_nope
            k = k_nope
            q_rope_arg = q_rope
            k_rope_arg = k_rope

    # Add dummy cos_sin_cache only for TRTLLM MLA path (both prefill/decode)
    if kernel_source == "trtllm_mla" and qk_rope_head_dim > 0:
        # flashinfer.rope.mla_rope_quantize_fp8 requires cos_sin_cache to be float32
        cos_sin_cache = torch.randn(
            total_len,
            max(1, qk_rope_head_dim // 2),
            2,
            device=torch_device,
            dtype=torch.float32,
        )
        extra_kwargs = {"cos_sin_cache": cos_sin_cache}
    else:
        extra_kwargs = {}

    forward_batch.req_to_token_pool = req_to_token_pool
    forward_batch.token_to_kv_pool = kv_pool
    forward_batch.attn_backend = attn_backend
    attn_backend.init_forward_metadata(forward_batch)

    latency, power_stats = benchmark_layer(
        layer,
        forward_batch,
        q,
        k,
        v,
        q_rope_arg,
        k_rope_arg,
        **extra_kwargs,
    )

    if is_context_phase:
        isl = input_len
        step = 0
    else:
        isl = 1
        step = input_len

    str_type = "bfloat16" if kv_cache_dtype == torch.bfloat16 else "fp8"
    log_perf(
        item_list=[
            {
                "mla_dtype": "bfloat16",
                "kv_cache_dtype": str_type,
                "num_heads": local_num_heads,
                "batch_size": batch_size,
                "isl": isl,
                "tp_size": tp_size,
                "step": step,
                "latency": latency,
            }
        ],
        framework="SGLang",
        version=pkg_resources.get_distribution("sglang").version,
        device_name=torch.cuda.get_device_name(device),
        op_name=f"mla_{'context' if is_context_phase else 'generation'}",
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=power_stats,
    )


if __name__ == "__main__":
    test_cases = get_context_mla_test_cases()
    for test_case in test_cases[0:10]:
        print(test_case)
        run_mla(*test_case, perf_filename=PerfFile.CONTEXT_MLA)

    test_cases = get_generation_mla_test_cases()
    for test_case in test_cases[0:10]:
        print(test_case)
        run_mla(*test_case, perf_filename=PerfFile.GENERATION_MLA)
