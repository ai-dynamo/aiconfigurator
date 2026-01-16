# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import tensorrt_llm
import torch

from helper import benchmark_with_power, get_sm_version, log_perf


def get_mla_gen_pre_test_cases():
    test_cases = []
    gen_num_tokens = [
        1,
        2,
        4,
        8,
        16,
        32,
        48,
        64,
        80,
        96,
        128,
        160,
        192,
        256,
        320,
        384,
        512,
        768,
        1024,
        1536,
        2048,
        3072,
        4096,
        6144,
        8192,
    ]
    num_heads = [128, 64, 32, 16, 8, 4, 2, 1]
    dtype_list = ["float16"]
    if 86 < get_sm_version() < 100:
        dtype_list += ["fp8"]
    for num_tokens in gen_num_tokens:
        for num_head in num_heads:
            for dtype in dtype_list:
                test_cases.append([num_tokens, num_head, dtype, 2, 10, "mla_bmm_perf.txt"])
    return test_cases


def get_mla_gen_post_test_cases():
    test_cases = []
    ctx_num_tokens = [
        1,
        2,
        4,
        8,
        16,
        32,
        48,
        64,
        80,
        96,
        128,
        160,
        192,
        256,
        320,
        384,
        512,
        768,
        1024,
        1536,
        2048,
        3072,
        4096,
        6144,
        8192,
        12288,
        16384,
        20480,
    ]
    num_heads = [128, 64, 32, 16, 8, 4, 2, 1]
    dtype_list = ["float16"]
    if 86 < get_sm_version() < 100:
        dtype_list += ["fp8"]
    for num_tokens in ctx_num_tokens:
        for num_head in num_heads:
            for dtype in dtype_list:
                test_cases.append([num_tokens, num_head, dtype, 2, 10, "mla_bmm_perf.txt"])
    return test_cases


def run_mla_gen_pre(num_tokens, num_heads, dtype, num_warmups, num_runs, perf_filename, device="cuda:0"):
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    # num_heads is already split by tp_size
    qk_nope_head_dim = 128
    kv_lora_rank = 512
    # record graph
    if dtype == "float16":
        q_nope = torch.randn([num_tokens, num_heads, qk_nope_head_dim]).bfloat16().to(torch.device(device))
        k_b_proj_trans = torch.randn([num_heads, kv_lora_rank, qk_nope_head_dim]).bfloat16().to(torch.device(device))
        out = torch.randn([num_tokens, num_heads, kv_lora_rank]).bfloat16().to(torch.device(device))
        # => num_heads, num_tokens, kv_lora_rank

        # Dry run
        q_nope_trans = q_nope.transpose(0, 1)
        k_b_proj_trans_trans = k_b_proj_trans.transpose(1, 2)
        out_trans = out.transpose(0, 1)
        torch.ops.trtllm.bmm_out(q_nope_trans, k_b_proj_trans_trans, out_trans)

        def kernel_func():
            q_nope_trans = q_nope.transpose(0, 1)
            k_b_proj_trans_trans = k_b_proj_trans.transpose(1, 2)
            out_trans = out.transpose(0, 1)
            torch.ops.trtllm.bmm_out(q_nope_trans, k_b_proj_trans_trans, out_trans)

        # Use benchmark_with_power context manager
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
            framework="TRTLLM",
            version=tensorrt_llm.__version__,
            device_name=torch.cuda.get_device_name(device),
            op_name="mla_gen_pre",
            kernel_source="default",
            perf_filename=perf_filename,
            power_stats=results["power_stats"],
        )
    elif dtype == "fp8":
        q_nope = torch.randn([num_tokens, num_heads, qk_nope_head_dim], dtype=torch.bfloat16).to(torch.device(device))
        # q_nope_fp8 = torch.randn(
        #     [num_heads, num_tokens, qk_nope_head_dim], dtype=torch.bfloat16, device=device
        # ).to(dtype=torch.float8_e4m3fn)
        k_b_proj_trans = torch.randn(
            [num_heads, kv_lora_rank, qk_nope_head_dim], dtype=torch.bfloat16, device=device
        ).to(dtype=torch.float8_e4m3fn)
        k_b_proj_trans_scale = torch.randn(
            [num_heads, kv_lora_rank // 128, qk_nope_head_dim // 128],
            dtype=torch.float32,
            device=device,
        )
        # q_nope_out = (
        #     torch.randn([num_heads, num_tokens, kv_lora_rank]).bfloat16().to(torch.device(device))
        # )
        fused_q = torch.randn([num_tokens, num_heads, kv_lora_rank], dtype=torch.bfloat16, device=device)
        # => num_heads, num_tokens, kv_lora_rank
        q_nope_fp8, q_nope_scales = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(q_nope)
        q_nope_out = fused_q.transpose(0, 1)
        torch.ops.trtllm.fp8_block_scaling_bmm_out(
            q_nope_fp8, k_b_proj_trans, q_nope_scales, k_b_proj_trans_scale, q_nope_out
        )

        def kernel_func():
            q_nope_fp8, q_nope_scales = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(q_nope)
            q_nope_out = fused_q.transpose(0, 1)
            torch.ops.trtllm.fp8_block_scaling_bmm_out(
                q_nope_fp8, k_b_proj_trans, q_nope_scales, k_b_proj_trans_scale, q_nope_out
            )

        # Use benchmark_with_power context manager
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
            framework="TRTLLM",
            version=tensorrt_llm.__version__,
            device_name=torch.cuda.get_device_name(device),
            op_name="mla_gen_pre",
            kernel_source="default",
            perf_filename=perf_filename,
            power_stats=results["power_stats"],
        )
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def run_mla_gen_post(num_tokens, num_heads, dtype, num_warmups, num_runs, perf_filename, device="cuda:0"):
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    # num_heads is already split by tp_size
    kv_lora_rank = 512
    v_head_dim = 128
    # record graph
    if dtype == "float16":
        attn_out_latent = torch.randn([num_tokens, num_heads, kv_lora_rank]).bfloat16().to(torch.device(device))
        v_b_proj = torch.randn([num_heads, v_head_dim, kv_lora_rank]).bfloat16().to(torch.device(device))
        attn_output = torch.randn([num_tokens, num_heads, v_head_dim]).bfloat16().to(torch.device(device))

        # Dry run
        torch.ops.trtllm.bmm_out(attn_out_latent.transpose(0, 1), v_b_proj.transpose(1, 2), attn_output.transpose(0, 1))

        def kernel_func():
            torch.ops.trtllm.bmm_out(
                attn_out_latent.transpose(0, 1),
                v_b_proj.transpose(1, 2),
                attn_output.transpose(0, 1),
            )

        # Use benchmark_with_power context manager
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
            framework="TRTLLM",
            version=tensorrt_llm.__version__,
            device_name=torch.cuda.get_device_name(device),
            op_name="mla_gen_post",
            kernel_source="default",
            perf_filename=perf_filename,
            power_stats=results["power_stats"],
        )
    elif dtype == "fp8":
        attn_out_latent = torch.randn([num_tokens, num_heads, kv_lora_rank], dtype=torch.bfloat16, device=device)
        v_b_proj = torch.randn([num_heads, v_head_dim, kv_lora_rank], dtype=torch.bfloat16, device=device).to(
            dtype=torch.float8_e4m3fn
        )
        v_b_proj_scale = torch.randn(
            [num_heads, v_head_dim // 128, kv_lora_rank // 128], dtype=torch.float32, device=device
        )
        attn_output = torch.randn([num_tokens, num_heads, v_head_dim]).bfloat16().to(torch.device(device))

        # dry run
        attn_out_latent_fp8, attn_out_latent_scales = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
            attn_out_latent
        )
        torch.ops.trtllm.fp8_block_scaling_bmm_out(
            attn_out_latent_fp8,
            v_b_proj,
            attn_out_latent_scales,
            v_b_proj_scale,
            attn_output.transpose(0, 1),
        )

        def kernel_func():
            attn_out_latent_fp8, attn_out_latent_scales = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(
                attn_out_latent
            )
            torch.ops.trtllm.fp8_block_scaling_bmm_out(
                attn_out_latent_fp8,
                v_b_proj,
                attn_out_latent_scales,
                v_b_proj_scale,
                attn_output.transpose(0, 1),
            )

        # Use benchmark_with_power context manager
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
            framework="TRTLLM",
            version=tensorrt_llm.__version__,
            device_name=torch.cuda.get_device_name(device),
            op_name="mla_gen_post",
            kernel_source="default",
            perf_filename=perf_filename,
            power_stats=results["power_stats"],
        )
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


if __name__ == "__main__":
    test_cases = get_mla_gen_pre_test_cases()
    for test_case in test_cases:
        print(test_case)
        run_mla_gen_pre(*test_case)
    test_cases = get_mla_gen_post_test_cases()
    for test_case in test_cases:
        print(test_case)
        run_mla_gen_post(*test_case)
