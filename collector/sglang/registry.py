# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Declarative registry mapping ops to collector modules for SGLang.

No version forks exist yet. When SGLang API changes require a fork,
add a ``versions`` tuple following the trtllm registry pattern.
"""

from collector.registry_types import OpEntry, PerfFile

REGISTRY: list[OpEntry] = [
    OpEntry(
        op="gemm",
        module="collector.sglang.collect_gemm",
        get_func="get_gemm_test_cases",
        run_func="run_gemm",
        perf_filename=PerfFile.GEMM,
    ),
    OpEntry(
        op="compute_scale",
        module="collector.sglang.collect_computescale",
        get_func="get_computescale_test_cases",
        run_func="run_computescale",
        perf_filename=PerfFile.COMPUTESCALE,
    ),
    OpEntry(
        op="mla_context",
        module="collector.sglang.collect_mla",
        get_func="get_context_mla_test_cases",
        run_func="run_mla",
        perf_filename=PerfFile.CONTEXT_MLA,
    ),
    OpEntry(
        op="mla_generation",
        module="collector.sglang.collect_mla",
        get_func="get_generation_mla_test_cases",
        run_func="run_mla",
        perf_filename=PerfFile.GENERATION_MLA,
    ),
    OpEntry(
        op="mla_bmm_gen_pre",
        module="collector.sglang.collect_mla_bmm",
        get_func="get_mla_gen_pre_test_cases",
        run_func="run_mla_gen_pre",
        perf_filename=PerfFile.MLA_BMM,
    ),
    OpEntry(
        op="mla_bmm_gen_post",
        module="collector.sglang.collect_mla_bmm",
        get_func="get_mla_gen_post_test_cases",
        run_func="run_mla_gen_post",
        perf_filename=PerfFile.MLA_BMM,
    ),
    OpEntry(
        op="moe",
        module="collector.sglang.collect_moe",
        get_func="get_moe_test_cases",
        run_func="run_moe_torch",
        perf_filename=PerfFile.MOE,
    ),
    OpEntry(
        op="attention_context",
        module="collector.sglang.collect_attn",
        get_func="get_context_attention_test_cases",
        run_func="run_attention_torch",
        perf_filename=PerfFile.CONTEXT_ATTENTION,
    ),
    OpEntry(
        op="attention_generation",
        module="collector.sglang.collect_attn",
        get_func="get_generation_attention_test_cases",
        run_func="run_attention_torch",
        perf_filename=PerfFile.GENERATION_ATTENTION,
    ),
    OpEntry(
        op="encoder_attention",
        module="collector.sglang.collect_attn_encoder",
        get_func="get_encoder_attention_test_cases",
        run_func="run_encoder_attention_torch",
        perf_filename=PerfFile.ENCODER_ATTENTION,
    ),
    OpEntry(
        op="dsa_context_module",
        module="collector.sglang.collect_mla_module",
        get_func="get_dsa_context_module_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.DSA_CONTEXT_MODULE,
    ),
    OpEntry(
        op="dsa_generation_module",
        module="collector.sglang.collect_mla_module",
        get_func="get_dsa_generation_module_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.DSA_GENERATION_MODULE,
    ),
    # DeepSeek-V4 module-level data (csa/hca x ctx/gen = 4 ops, 1 file each).
    OpEntry(
        op="dsv4_csa_context_module",
        module="collector.sglang.collect_dsv4_attn",
        get_func="get_dsv4_csa_context_test_cases",
        run_func="run_dsv4_attn_worker",
        perf_filename=PerfFile.DSV4_CSA_CONTEXT_MODULE,
    ),
    OpEntry(
        op="dsv4_hca_context_module",
        module="collector.sglang.collect_dsv4_attn",
        get_func="get_dsv4_hca_context_test_cases",
        run_func="run_dsv4_attn_worker",
        perf_filename=PerfFile.DSV4_HCA_CONTEXT_MODULE,
    ),
    OpEntry(
        op="dsv4_csa_generation_module",
        module="collector.sglang.collect_dsv4_attn",
        get_func="get_dsv4_csa_generation_test_cases",
        run_func="run_dsv4_attn_worker",
        perf_filename=PerfFile.DSV4_CSA_GENERATION_MODULE,
    ),
    OpEntry(
        op="dsv4_hca_generation_module",
        module="collector.sglang.collect_dsv4_attn",
        get_func="get_dsv4_hca_generation_test_cases",
        run_func="run_dsv4_attn_worker",
        perf_filename=PerfFile.DSV4_HCA_GENERATION_MODULE,
    ),
    # DeepSeek-V4 currently models CSA/HCA through full attention-module data
    # above.  Keep these kernel-level collectors as supporting data for future
    # prefix/past_kv correction and residual analysis; they are not the primary
    # modeling path.
    OpEntry(
        op="dsv4_paged_mqa_logits_module",
        module="collector.sglang.deepseekv4_sparse_modules",
        get_func="get_dsv4_paged_mqa_logits_test_cases",
        run_func="run_dsv4_sparse_kernel_worker",
        perf_filename=PerfFile.DSV4_PAGED_MQA_LOGITS_MODULE,
    ),
    OpEntry(
        op="dsv4_hca_attn_module",
        module="collector.sglang.deepseekv4_sparse_modules",
        get_func="get_dsv4_hca_attn_test_cases",
        run_func="run_dsv4_sparse_kernel_worker",
        perf_filename=PerfFile.DSV4_HCA_ATTN_MODULE,
    ),
    OpEntry(
        op="dsv4_csa_attn_module",
        module="collector.sglang.deepseekv4_sparse_modules",
        get_func="get_dsv4_csa_attn_test_cases",
        run_func="run_dsv4_sparse_kernel_worker",
        perf_filename=PerfFile.DSV4_CSA_ATTN_MODULE,
    ),
    # CSA topk_512 DELTA calibration (flat vs top_last) — feeds the
    # degenerate->representative topK correction in perf_database.
    OpEntry(
        op="dsv4_csa_topk_calib",
        module="collector.sglang.deepseekv4_sparse_modules",
        get_func="get_dsv4_topk_calib_test_cases",
        run_func="run_dsv4_sparse_kernel_worker",
        perf_filename=PerfFile.DSV4_CSA_TOPK_CALIB,
    ),
    # GLM-5 DSA sparse sub-kernels (mqa / topk / dsa_attn) — GLM-5 analogue of
    # the DSV4 sparse family; shapes 1:1 from the GLM-5 DSA module CSV.
    OpEntry(
        op="glm5_mqa_logits_module",
        module="collector.sglang.glm5_dsa_sparse_modules",
        get_func="get_glm5_mqa_test_cases",
        run_func="run_glm5_dsa_sparse_kernel_worker",
        perf_filename=PerfFile.GLM5_MQA_LOGITS_MODULE,
    ),
    OpEntry(
        op="glm5_topk_module",
        module="collector.sglang.glm5_dsa_sparse_modules",
        get_func="get_glm5_topk_test_cases",
        run_func="run_glm5_dsa_sparse_kernel_worker",
        perf_filename=PerfFile.GLM5_TOPK_MODULE,
    ),
    OpEntry(
        op="glm5_dsa_attn_module",
        module="collector.sglang.glm5_dsa_sparse_modules",
        get_func="get_glm5_dsa_attn_test_cases",
        run_func="run_glm5_dsa_sparse_kernel_worker",
        perf_filename=PerfFile.GLM5_DSA_ATTN_MODULE,
    ),
    OpEntry(
        op="gdn",
        module="collector.sglang.collect_gdn",
        get_func="get_gdn_test_cases",
        run_func="run_gdn_torch",
        perf_filename=PerfFile.GDN,
    ),
    OpEntry(
        op="mhc_module",
        module="collector.sglang.collect_mhc_module",
        get_func="get_mhc_module_test_cases",
        run_func="run_mhc_module_worker",
        perf_filename=PerfFile.MHC_MODULE,
    ),
]
