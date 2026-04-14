# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pkg_resources
import torch
from sgl_kernel import bmm_fp8
from sglang.srt.layers.quantization.fp8_kernel import (
    per_tensor_quant_mla_fp8,
)

from collector.common_test_cases import (
    build_mla_gen_post_test_cases,
    build_mla_gen_pre_test_cases,
)
from helper import benchmark_with_power, log_perf

_DTYPE_LIST = ["float16", "fp8"]


def get_mla_gen_pre_test_cases():
    return build_mla_gen_pre_test_cases(_DTYPE_LIST)


def get_mla_gen_post_test_cases():
    return build_mla_gen_post_test_cases(_DTYPE_LIST)


def run_mla_gen_pre(num_tokens, num_heads, dtype, num_warmups, num_runs, perf_filename, device="cuda:0"):
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    assert dtype == "fp8" or dtype == "float16", "only support fp8 and float16"

    qk_nope_head_dim = 128
    kv_lora_rank = 512

    q_nope = torch.randn((num_tokens, num_heads, qk_nope_head_dim), device=device, dtype=torch.bfloat16)

    if dtype == "fp8":
        zeroscale = torch.tensor([0], dtype=torch.float32, device=device)
        q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(q_nope.transpose(0, 1), zeroscale)
        w_kc = torch.randn((num_heads, kv_lora_rank, qk_nope_head_dim), dtype=torch.bfloat16, device=device).to(
            dtype=torch.float8_e4m3fn
        )
        w_kc = w_kc.transpose(1, 2)
        w_scale = torch.randn(
            (num_heads, kv_lora_rank // 128, qk_nope_head_dim // 128),
            dtype=torch.float32,
            device=device,
        )

        def kernel_func():
            q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(q_nope.transpose(0, 1), zeroscale)
            bmm_fp8(q_nope_val, w_kc, q_nope_scale, w_scale, torch.bfloat16)
    else:
        w_kc = torch.randn((num_heads, kv_lora_rank, qk_nope_head_dim), dtype=torch.bfloat16, device=device)
        w_kc = w_kc.transpose(1, 2)

        def kernel_func():
            torch.bmm(q_nope.transpose(0, 1), w_kc)

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
        framework="SGLang",
        version=pkg_resources.get_distribution("sglang").version,
        device_name=torch.cuda.get_device_name(device),
        op_name="mla_gen_pre",
        kernel_source="default",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


def run_mla_gen_post(num_tokens, num_heads, dtype, num_warmups, num_runs, perf_filename, device="cuda:0"):
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    assert dtype == "float16" or dtype == "fp8", "only support fp8 and float16"

    qk_nope_head_dim = 128
    kv_lora_rank = 512
    v_head_dim = 128

    if dtype == "float16":
        attn_output = torch.randn([num_tokens, num_heads, kv_lora_rank]).bfloat16().to(torch.device(device))
        w_vc = torch.randn([num_heads, v_head_dim, kv_lora_rank]).bfloat16().to(torch.device(device))
        w_vc = w_vc.transpose(1, 2)
        attn_bmm_output = torch.empty(
            (attn_output.shape[0], qk_nope_head_dim * v_head_dim),
            dtype=attn_output.dtype,
            device=attn_output.device,
        )

        torch.bmm(
            attn_output.transpose(0, 1),
            w_vc,
            out=attn_bmm_output.view(-1, qk_nope_head_dim, v_head_dim).transpose(0, 1),
        )

        def kernel_func():
            torch.bmm(
                attn_output.transpose(0, 1),
                w_vc,
                out=attn_bmm_output.view(-1, qk_nope_head_dim, v_head_dim).transpose(0, 1),
            )
    else:
        attn_output = torch.randn([num_tokens, num_heads, kv_lora_rank], dtype=torch.bfloat16, device=device)
        w_vc = torch.randn([num_heads, v_head_dim, kv_lora_rank], dtype=torch.bfloat16, device=device).to(
            dtype=torch.float8_e4m3fn
        )
        w_vc = w_vc.transpose(1, 2)
        w_scale = torch.randn([num_heads, v_head_dim // 128, kv_lora_rank // 128], dtype=torch.float32, device=device)
        attn_bmm_output = torch.randn([num_tokens, num_heads, v_head_dim]).bfloat16().to(torch.device(device))

        zeroscale = torch.tensor([1], dtype=torch.float32, device=device)

        def kernel_func():
            attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                attn_output.transpose(0, 1),
                zeroscale,
            )
            bmm_fp8(
                attn_output_val,
                w_vc,
                attn_output_scale,
                w_scale,
                torch.bfloat16,
            )

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
        framework="SGLang",
        version=pkg_resources.get_distribution("sglang").version,
        device_name=torch.cuda.get_device_name(device),
        op_name="mla_gen_post",
        kernel_source="default",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )
