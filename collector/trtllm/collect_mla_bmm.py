# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Verified against 1.3.0rc20 serving source (2026-07-18, B200/SM100 phase).
# Serving's MLA absorption BMM wrapper `fp8_block_scaling_bmm_out`
# (tensorrt_llm/_torch/modules/attention.py:1151-1191@1.3.0rc20) dispatches:
#   SM90/89  -> torch.ops.trtllm.fp8_block_scaling_bmm_out (measured here);
#   SM120    -> the same torch op with per_token_quant_and_transform;
#   SM100/103 (is_sm_100f) -> plain bf16 torch.bmm on dequantized weights by
#              default (opt-in cute_dsl_fp8_bmm_blackwell only).
# So the fp8 axis of THIS op (the Hopper quantize+bmm pair) is correctly
# closed on SM100/103: serving never invokes that torch op there, and the
# bf16 axis (torch.ops.trtllm.bmm_out, attention.py:2507) remains the honest
# SM100 measurement.
# SM120 half RESOLVED on 1.3.0rc20/SM120 (2026-07-19, RTX PRO 6000): the
# serving pair — fp8_utils.per_token_quant_and_transform(need_permute102=True)
# activation quant + resmooth_to_fp8_e8m0/transform_sf_into_required_layout
# weight scales (transform_weights, attention.py:3013-3028@1.3.0rc20) into the
# same torch.ops.trtllm.fp8_block_scaling_bmm_out — runs pre+post shapes
# (t in {1,64,512,2048}) with finite outputs and fp8-level numerics vs a bf16
# reference (mean rel err 6-8%, scale=1.0 isolation). The fp8 axis is
# therefore open on SM120 below, dispatching the SM120 quant helpers exactly
# as serving does. Never move this back into YAML.

"""TensorRT-LLM MLA generation BMM micro-collector.

Benchmarks the auxiliary MLA generation BMM shapes used by TRT-LLM modeling. It
consumes the YAML-backed shape grid, sets up BF16/FP8 tensors, runs benchmarks,
and logs MLA BMM perf rows for pre/post generation operations.
"""

import tensorrt_llm
import tensorrt_llm.quantization.utils.fp8_utils as fp8_utils
import torch
from case_generator import get_mla_bmm_case_specs

from helper import benchmark_with_power, get_sm_version, log_perf


def _supported_dtypes() -> set[str]:
    # fp8 axis mirrors serving's fp8_block_scaling_bmm_out wrapper dispatch
    # (attention.py:1159-1190@1.3.0rc20): SM89/90 and SM120 invoke the fp8
    # torch op (different activation-quant helpers); SM100/103 default to
    # plain bf16 torch.bmm on dequantized weights, so the axis stays closed
    # there and bf16 remains the honest measurement.
    dtype_list = ["bfloat16"]
    sm = get_sm_version()
    if 86 < sm < 100 or sm == 120:
        dtype_list += ["fp8"]
    return set(dtype_list)


def _prep_fp8_weight(weight_fp8: torch.Tensor, weight_scale: torch.Tensor):
    """Load-time weight-scale prep, mirroring serving per SM.

    SM120 serving resmooths block scales to e8m0 and transforms the layout at
    weight-load time (MLA.transform_weights → resmooth_parameters,
    attention.py:3013-3028@1.3.0rc20); SM89/90 consume the raw float32
    1x128 block scales directly.
    """
    if get_sm_version() == 120:
        weight_fp8, weight_scale = fp8_utils.resmooth_to_fp8_e8m0(weight_fp8, weight_scale)
        weight_scale = fp8_utils.transform_sf_into_required_layout(
            weight_scale,
            mn=weight_fp8.shape[1],
            k=weight_fp8.shape[2],
            recipe=(1, 128, 128),
            num_groups=weight_fp8.shape[0],
            is_sfa=False,
        )
    return weight_fp8, weight_scale


def _quantize_fp8_activation(x: torch.Tensor):
    """Per-forward activation quant, mirroring serving per SM.

    Serving's fp8_block_scaling_bmm_out wrapper (attention.py:1159-1176
    @1.3.0rc20) quantizes mat1 with fp8_batched_quantize_1x128_permute102 on
    SM89/90 and fp8_utils.per_token_quant_and_transform(need_permute102=True)
    on SM120, then calls the same torch.ops.trtllm.fp8_block_scaling_bmm_out.
    """
    if get_sm_version() == 120:
        return fp8_utils.per_token_quant_and_transform(x, need_permute102=True)
    return torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(x)


def _get_mla_bmm_test_cases(op_name: str):
    supported_dtypes = _supported_dtypes()
    return [
        [case.num_tokens, case.num_heads, case.dtype, case.num_warmups, case.num_runs]
        for case in get_mla_bmm_case_specs("trtllm", op_name)
        if case.dtype in supported_dtypes
    ]


def get_mla_gen_pre_test_cases():
    return _get_mla_bmm_test_cases("mla_bmm_gen_pre")


def get_mla_gen_post_test_cases():
    return _get_mla_bmm_test_cases("mla_bmm_gen_post")


def run_mla_gen_pre(num_tokens, num_heads, dtype, num_warmups, num_runs, *, perf_filename, device="cuda:0"):
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    # num_heads is already split by tp_size
    qk_nope_head_dim = 128
    kv_lora_rank = 512
    # record graph
    if dtype == "bfloat16":
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
        # positive scales: the SM120 prep path (resmooth to e8m0) works in
        # log2 domain and would NaN on randn's negative values
        k_b_proj_trans_scale = (
            torch.rand(
                [num_heads, kv_lora_rank // 128, qk_nope_head_dim // 128],
                dtype=torch.float32,
                device=device,
            )
            + 0.5
        )
        k_b_proj_trans, k_b_proj_trans_scale = _prep_fp8_weight(k_b_proj_trans, k_b_proj_trans_scale)
        # q_nope_out = (
        #     torch.randn([num_heads, num_tokens, kv_lora_rank]).bfloat16().to(torch.device(device))
        # )
        fused_q = torch.randn([num_tokens, num_heads, kv_lora_rank], dtype=torch.bfloat16, device=device)
        # => num_heads, num_tokens, kv_lora_rank
        q_nope_fp8, q_nope_scales = _quantize_fp8_activation(q_nope)
        q_nope_out = fused_q.transpose(0, 1)
        torch.ops.trtllm.fp8_block_scaling_bmm_out(
            q_nope_fp8, k_b_proj_trans, q_nope_scales, k_b_proj_trans_scale, q_nope_out
        )

        def kernel_func():
            q_nope_fp8, q_nope_scales = _quantize_fp8_activation(q_nope)
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


def run_mla_gen_post(num_tokens, num_heads, dtype, num_warmups, num_runs, *, perf_filename, device="cuda:0"):
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    # num_heads is already split by tp_size
    kv_lora_rank = 512
    v_head_dim = 128
    # record graph
    if dtype == "bfloat16":
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
        # positive scales: see the pre-BMM note (SM120 e8m0 resmooth)
        v_b_proj_scale = (
            torch.rand([num_heads, v_head_dim // 128, kv_lora_rank // 128], dtype=torch.float32, device=device) + 0.5
        )
        v_b_proj, v_b_proj_scale = _prep_fp8_weight(v_b_proj, v_b_proj_scale)
        attn_output = torch.randn([num_tokens, num_heads, v_head_dim]).bfloat16().to(torch.device(device))

        # dry run
        attn_out_latent_fp8, attn_out_latent_scales = _quantize_fp8_activation(attn_out_latent)
        torch.ops.trtllm.fp8_block_scaling_bmm_out(
            attn_out_latent_fp8,
            v_b_proj,
            attn_out_latent_scales,
            v_b_proj_scale,
            attn_output.transpose(0, 1),
        )

        def kernel_func():
            attn_out_latent_fp8, attn_out_latent_scales = _quantize_fp8_activation(attn_out_latent)
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
    from registry_types import PerfFile

    test_cases = get_mla_gen_pre_test_cases()
    for test_case in test_cases:
        print(test_case)
        run_mla_gen_pre(*test_case, perf_filename=PerfFile.MLA_BMM)
    test_cases = get_mla_gen_post_test_cases()
    for test_case in test_cases:
        print(test_case)
        run_mla_gen_post(*test_case, perf_filename=PerfFile.MLA_BMM)
