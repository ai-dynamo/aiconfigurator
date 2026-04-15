# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__compat__ = "vllm>=0.14.0"

"""
MLA BMM collector for vLLM — benchmarks the two batch matrix multiplications
in MLA decode (generation) path.

In vLLM's MLA decode path (MLAAttention.forward_impl / _v_up_proj):

  Pre-BMM  (mla_gen_pre):
    q_nope [B,N,P]  →  transpose(0,1) → [N,B,P]
    W_UK_T [N,P,L]  (= W_UK.permute(1,2,0), stored in self.W_UK_T)
    torch.bmm(q_nope_trans, W_UK_T) → [N,B,L]   # ql_nope projection

  Post-BMM (mla_gen_post):
    attn_out [B,N,L] →  view + transpose(0,1) → [N,B,L]
    W_UV     [N,L,V]   (= W_UV.transpose(0,1), stored in self.W_UV)
    torch.bmm(attn_out_trans, W_UV) → [N,B,V]   # v up-projection

Both ops use bfloat16; vLLM explicitly does not have quantized BMM kernels for
W_UV / W_UK_T on NVIDIA hardware (comment in process_weights_after_loading).

Weight tensor layouts (production, float16 path):
  W_UK_T = W_UK.permute(1, 2, 0)  where W_UK has shape [L, H, P]
           -> logical [H, P, L], strides (P, 1, H*P)  [heads interleaved]
  W_UV   = W_UV.transpose(0, 1)   where W_UV has shape [L, H, V]
           -> logical [H, L, V], strides (V, H*V, 1)  [heads interleaved]

DeepSeek V3 / R1 dimensions:
  qk_nope_head_dim  P = 128
  kv_lora_rank      L = 512
  v_head_dim        V = 128

Reference:
  vllm>=0.17.0: vllm/model_executor/layers/attention/mla_attention.py
    MLAAttention.forward_impl (pre-bmm), MLAAttention._v_up_proj (post-bmm)
  vllm>=0.14.0: vllm/v1/attention/backends/mla/common.py
    MLACommonImpl.forward (pre-bmm), MLACommonImpl._v_up_proj (post-bmm)
  Weight layout is identical across versions.
"""

import torch
from vllm.version import __version__ as vllm_version

from collector.common_test_cases import (
    build_mla_gen_post_test_cases,
    build_mla_gen_pre_test_cases,
)
from collector.helper import benchmark_with_power, log_perf


# DeepSeek V3 / R1 MLA dimensions (fixed across all test cases)
_QK_NOPE_HEAD_DIM = 128   # P
_KV_LORA_RANK = 512       # L
_V_HEAD_DIM = 128         # V

# vLLM has no fp8 bmm kernel for W_UK_T / W_UV on NVIDIA hardware —
# process_weights_after_loading always dequantizes to bfloat16.
_DTYPE_LIST = ["float16"]


def get_mla_gen_pre_test_cases():
    return build_mla_gen_pre_test_cases(_DTYPE_LIST)


def get_mla_gen_post_test_cases():
    return build_mla_gen_post_test_cases(_DTYPE_LIST)


def run_mla_gen_pre(num_tokens, num_heads, dtype, num_warmups, num_runs, perf_filename, device="cuda:0"):
    """
    Benchmark the pre-BMM in vLLM's MLA decode path:

        q_nope [num_tokens, num_heads, P]  →  transpose  →  [num_heads, num_tokens, P]
        W_UK_T [num_heads, P, L]
        out    [num_heads, num_tokens, L]

        torch.bmm(q_nope_trans, W_UK_T, out=out)

    This mirrors MLAAttention.forward_impl in vLLM 0.17.0.
    """
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    assert dtype == "float16", "vLLM MLA BMM only supports float16 (bfloat16) on NVIDIA"

    qk_nope_head_dim = _QK_NOPE_HEAD_DIM
    kv_lora_rank = _KV_LORA_RANK

    # Inputs
    q_nope = torch.randn([num_tokens, num_heads, qk_nope_head_dim], dtype=torch.bfloat16, device=device)
    # Replicate production layout: W_UK [L, H, P] -> W_UK_T = W_UK.permute(1, 2, 0) -> [H, P, L]
    # Strides are (P, 1, H*P) — heads interleaved in underlying storage.
    W_UK_T = torch.randn([kv_lora_rank, num_heads, qk_nope_head_dim], dtype=torch.bfloat16, device=device).permute(1, 2, 0)
    out = torch.empty([num_heads, num_tokens, kv_lora_rank], dtype=torch.bfloat16, device=device)

    # Dry run
    q_nope_trans = q_nope.transpose(0, 1)  # [N, B, P]
    torch.bmm(q_nope_trans, W_UK_T, out=out)

    def kernel_func():
        q_nope_trans = q_nope.transpose(0, 1)  # [N, B, P]
        torch.bmm(q_nope_trans, W_UK_T, out=out)

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=num_warmups,
        num_runs=num_runs,
        repeat_n=1,
    ) as results:
        pass

    log_perf(
        item_list=[
            {
                "bmm_dtype": dtype,
                "num_tokens": num_tokens,
                "num_heads": num_heads,
                "latency": results["latency_ms"],
            }
        ],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name="mla_gen_pre",
        kernel_source="default",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


def run_mla_gen_post(num_tokens, num_heads, dtype, num_warmups, num_runs, perf_filename, device="cuda:0"):
    """
    Benchmark the post-BMM (v up-projection) in vLLM's MLA decode path:

        attn_out [num_tokens, num_heads, L]  →  transpose  →  [num_heads, num_tokens, L]
        W_UV     [num_heads, L, V]
        out      [num_heads, num_tokens, V]

        torch.bmm(attn_out_trans, W_UV, out=out)

    This mirrors MLAAttention._v_up_proj in vLLM 0.17.0.
    """
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    assert dtype == "float16", "vLLM MLA BMM only supports float16 (bfloat16) on NVIDIA"

    kv_lora_rank = _KV_LORA_RANK
    v_head_dim = _V_HEAD_DIM

    # Inputs
    attn_out = torch.randn([num_tokens, num_heads, kv_lora_rank], dtype=torch.bfloat16, device=device)
    # Replicate production layout: W_UV [L, H, V] -> W_UV = W_UV.transpose(0, 1) -> [H, L, V]
    # Strides are (V, H*V, 1) — heads interleaved in underlying storage.
    W_UV = torch.randn([kv_lora_rank, num_heads, v_head_dim], dtype=torch.bfloat16, device=device).transpose(0, 1)
    out = torch.empty([num_heads, num_tokens, v_head_dim], dtype=torch.bfloat16, device=device)

    # Dry run
    attn_out_trans = attn_out.transpose(0, 1)  # [N, B, L]
    torch.bmm(attn_out_trans, W_UV, out=out)

    def kernel_func():
        attn_out_trans = attn_out.transpose(0, 1)  # [N, B, L]
        torch.bmm(attn_out_trans, W_UV, out=out)

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=num_warmups,
        num_runs=num_runs,
        repeat_n=1,
    ) as results:
        pass

    log_perf(
        item_list=[
            {
                "bmm_dtype": dtype,
                "num_tokens": num_tokens,
                "num_heads": num_heads,
                "latency": results["latency_ms"],
            }
        ],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name="mla_gen_post",
        kernel_source="default",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


if __name__ == "__main__":
    print("=== mla_gen_pre test cases ===")
    for tc in get_mla_gen_pre_test_cases()[:3]:
        print(tc)
        run_mla_gen_pre(*tc)

    print("=== mla_gen_post test cases ===")
    for tc in get_mla_gen_post_test_cases()[:3]:
        print(tc)
        run_mla_gen_post(*tc)
