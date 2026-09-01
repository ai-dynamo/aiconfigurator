# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Declarative registry mapping ops to collector modules for vLLM.

Collector-v2 keeps active entries aligned with the current framework manifest.
Only add a versioned route when the manifest intentionally supports multiple
live framework APIs for the same op.
"""

from collector.registry_types import OpEntry, PerfFile

REGISTRY: list[OpEntry] = [
    OpEntry(
        op="gemm",
        module="collector.vllm.collect_gemm",
        get_func="get_gemm_test_cases",
        run_func="run_gemm",
        perf_filename=PerfFile.GEMM,
    ),
    OpEntry(
        op="compute_scale",
        module="collector.vllm.collect_computescale",
        get_func="get_computescale_test_cases",
        run_func="run_computescale",
        perf_filename=PerfFile.COMPUTESCALE,
    ),
    OpEntry(
        op="attention_context",
        module="collector.vllm.collect_attn",
        get_func="get_context_attention_test_cases",
        run_func="run_attention_torch",
        perf_filename=PerfFile.CONTEXT_ATTENTION,
    ),
    OpEntry(
        op="attention_generation",
        module="collector.vllm.collect_attn",
        get_func="get_generation_attention_test_cases",
        run_func="run_attention_torch",
        perf_filename=PerfFile.GENERATION_ATTENTION,
    ),
    OpEntry(
        op="encoder_attention",
        module="collector.vllm.collect_attn_encoder",
        get_func="get_encoder_attention_test_cases",
        run_func="run_encoder_attention_torch",
        perf_filename=PerfFile.ENCODER_ATTENTION,
    ),
    OpEntry(
        op="moe",
        module="collector.vllm.collect_moe",
        get_func="get_moe_test_cases",
        run_func="run_moe_torch",
        perf_filename=PerfFile.MOE,
    ),
    OpEntry(
        op="mla_bmm_gen_pre",
        module="collector.vllm.collect_mla_bmm",
        get_func="get_mla_gen_pre_test_cases",
        run_func="run_mla_gen_pre",
        perf_filename=PerfFile.MLA_BMM,
    ),
    OpEntry(
        op="mla_bmm_gen_post",
        module="collector.vllm.collect_mla_bmm",
        get_func="get_mla_gen_post_test_cases",
        run_func="run_mla_gen_post",
        perf_filename=PerfFile.MLA_BMM,
    ),
    OpEntry(
        op="mla_context_module",
        module="collector.vllm.collect_mla_module_027",
        get_func="get_mla_context_module_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.MLA_CONTEXT_MODULE,
    ),
    OpEntry(
        op="mla_generation_module",
        module="collector.vllm.collect_mla_module_027",
        get_func="get_mla_generation_module_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.MLA_GENERATION_MODULE,
    ),
    OpEntry(
        op="dsa_context_module",
        module="collector.vllm.collect_mla_module",
        get_func="get_dsa_context_module_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.DSA_CONTEXT_MODULE,
    ),
    OpEntry(
        op="dsa_generation_module",
        module="collector.vllm.collect_mla_module",
        get_func="get_dsa_generation_module_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.DSA_GENERATION_MODULE,
    ),
    OpEntry(
        op="msa_context_module",
        module="collector.vllm.collect_msa_module",
        get_func="get_msa_context_module_test_cases",
        run_func="run_msa_module_worker",
        perf_filename=PerfFile.MSA_CONTEXT_MODULE,
        # MiniMax-M3 MSA hardware-validated on SM90 (h100/h200, Triton
        # impls), SM100/103 (b200/b300/gb200/gb300 — vLLM's own dispatch
        # switches the attend + indexer to the fmha_sm100 "MSA" impls via
        # select_main_impl_cls / select_indexer_impl_cls @0.24.0), SM89
        # (l40s) and SM120 (Triton impls; fp8_block gemm cases fail
        # classified on SM120 via DeepGEMM's SM120 assert). SM121 has never
        # run on hardware and stays marked.
        unverified_sms=(121,),
    ),
    OpEntry(
        op="msa_generation_module",
        module="collector.vllm.collect_msa_module",
        get_func="get_msa_generation_module_test_cases",
        run_func="run_msa_module_worker",
        perf_filename=PerfFile.MSA_GENERATION_MODULE,
        # See msa_context_module above.
        unverified_sms=(121,),
    ),
    OpEntry(
        op="dsv4_csa_context_module",
        module="collector.vllm.collect_dsv4_attn",
        get_func="get_dsv4_csa_context_test_cases",
        run_func="run_dsv4_attn_worker",
        perf_filename=PerfFile.DSV4_CSA_CONTEXT_MODULE,
    ),
    OpEntry(
        op="dsv4_hca_context_module",
        module="collector.vllm.collect_dsv4_attn",
        get_func="get_dsv4_hca_context_test_cases",
        run_func="run_dsv4_attn_worker",
        perf_filename=PerfFile.DSV4_HCA_CONTEXT_MODULE,
    ),
    OpEntry(
        op="dsv4_csa_generation_module",
        module="collector.vllm.collect_dsv4_attn",
        get_func="get_dsv4_csa_generation_test_cases",
        run_func="run_dsv4_attn_worker",
        perf_filename=PerfFile.DSV4_CSA_GENERATION_MODULE,
    ),
    OpEntry(
        op="dsv4_hca_generation_module",
        module="collector.vllm.collect_dsv4_attn",
        get_func="get_dsv4_hca_generation_test_cases",
        run_func="run_dsv4_attn_worker",
        perf_filename=PerfFile.DSV4_HCA_GENERATION_MODULE,
    ),
    OpEntry(
        op="dsv4_paged_mqa_logits_module",
        module="collector.vllm.collect_dsv4_attn",
        get_func="get_dsv4_paged_mqa_logits_test_cases",
        run_func="run_dsv4_sparse_kernel_worker",
        perf_filename=PerfFile.DSV4_PAGED_MQA_LOGITS_MODULE,
    ),
    OpEntry(
        op="dsv4_hca_attn_module",
        module="collector.vllm.collect_dsv4_attn",
        get_func="get_dsv4_hca_attn_test_cases",
        run_func="run_dsv4_sparse_kernel_worker",
        perf_filename=PerfFile.DSV4_HCA_ATTN_MODULE,
    ),
    OpEntry(
        op="mhc_module",
        module="collector.vllm.collect_mhc_module",
        get_func="get_mhc_module_test_cases",
        run_func="run_mhc_module_worker",
        perf_filename=PerfFile.MHC_MODULE,
    ),
    OpEntry(
        op="gdn",
        module="collector.vllm.collect_gdn",
        get_func="get_gdn_test_cases",
        run_func="run_gdn_torch",
        perf_filename=PerfFile.GDN,
    ),
    OpEntry(
        op="kda",
        module="collector.vllm.collect_kda",
        get_func="get_kda_test_cases",
        run_func="run_kda_torch",
        perf_filename=PerfFile.KDA,
        # Kimi-K3 KDA kernels exist only on the vLLM kimi-k3 branch preview
        # image; verified on Hopper (SM90), Ada (SM89 — full grid, 1145
        # rows, chunk/Triton fallback lanes, no FlashKDA/fused-decode
        # below SM90), B200 (SM100), B300/GB200/GB300 (SM100/103) and RTX
        # PRO 6000 (SM120 — full grid, 1203 rows, all six kernel paths).
        # Only SM80 (no probe hardware) remains unverified.
        unverified_sms=(80,),
    ),
]

REGISTRY_XPU: list[OpEntry] = [
    OpEntry(
        op="gemm",
        module="collector.vllm.collect_gemm_xpu",
        get_func="get_gemm_test_cases",
        run_func="run_gemm",
        perf_filename=PerfFile.GEMM,
    ),
    OpEntry(
        op="attention_context",
        module="collector.vllm.collect_attn_xpu",
        get_func="get_context_attention_test_cases",
        run_func="run_attention_torch",
        perf_filename=PerfFile.CONTEXT_ATTENTION,
    ),
    OpEntry(
        op="attention_generation",
        module="collector.vllm.collect_attn_xpu",
        get_func="get_generation_attention_test_cases",
        run_func="run_attention_torch",
        perf_filename=PerfFile.GENERATION_ATTENTION,
    ),
    OpEntry(
        op="moe",
        module="collector.vllm.collect_moe_xpu",
        get_func="get_moe_test_cases",
        run_func="run_moe_torch",
        perf_filename=PerfFile.MOE,
    ),
]
