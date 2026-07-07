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
        # SM120 bring-up audit completed on RTX 6000 Pro (2026-07-05): every
        # (quant mode x backend) family in the SM120 plan was probed on
        # hardware with verified constructed-method provenance — bf16/triton,
        # bf16/flashinfer_cutlass (NemotronH), fp8_block/triton,
        # nvfp4/flashinfer_cutlass, w4a16_mxfp4/marlin (GPT-OSS) — and the
        # framework-auto NVFP4 trtllm-gen path was shown in-kernel broken on
        # SM120. See the SM120 entry in the Hopper/Blackwell ledger.
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
        # SM103's bundled TRTLLM-GEN capability check rejects the target.
        # SM120 stays parked: the RTX 6000 Pro probe (2026-07-06) confirmed
        # 0.5.14 forces the TRTLLM-GEN sub-backend there and every case
        # raises "Unsupported architecture" (fmhaRunner.cuh:37) — hardware
        # confirmation of the ledger's DSA-SUBBACKEND-SELECTOR row.
        unverified_sms=(103, 120),
    ),
    OpEntry(
        op="dsa_generation_module",
        module="collector.sglang.collect_mla_module",
        get_func="get_dsa_generation_module_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.DSA_GENERATION_MODULE,
        unverified_sms=(103, 120),
    ),
    # GLM-5.2 skip-indexer layers (index_topk_freq>1): same shapes as the full
    # DSA module, but the per-layer indexer (mqa+topk+index-K store) is patched
    # out so the captured cost is the reuse-layer cost. run_func derives a
    # skip_indexer bool from the "skip_indexer" perf_filename and passes it to
    # the benchmark subprocess as an explicit arg (no env var).
    OpEntry(
        op="dsa_context_module_skip_indexer",
        module="collector.sglang.collect_mla_module",
        get_func="get_dsa_context_module_skip_indexer_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.DSA_CONTEXT_MODULE_SKIP_INDEXER,
        # The only registered GLM-5.2 artifact is NVFP4 (SM100+); the exact
        # reuse-layer path is currently hardware-validated only on SM100.
        # SM120 shares the full-module TRTLLM-GEN "Unsupported architecture"
        # raise (RTX 6000 Pro probe 2026-07-06).
        unverified_sms=(90, 103, 120),
    ),
    OpEntry(
        op="dsa_generation_module_skip_indexer",
        module="collector.sglang.collect_mla_module",
        get_func="get_dsa_generation_module_skip_indexer_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.DSA_GENERATION_MODULE_SKIP_INDEXER,
        unverified_sms=(90, 103, 120),
    ),
    # DeepSeek-V4 module-level data (csa/hca x ctx/gen = 4 ops, 1 file each).
    OpEntry(
        op="dsv4_csa_context_module",
        module="collector.sglang.collect_dsv4_attn",
        get_func="get_dsv4_csa_context_test_cases",
        run_func="run_dsv4_attn_worker",
        perf_filename=PerfFile.DSV4_CSA_CONTEXT_MODULE,
        # RTX 6000 Pro probe 2026-07-06: every SM120 case raises the
        # collector's own fail-closed guard (_derive_csa_context_pool_cap —
        # SM120 forces a Torch logits leaf whose memory contract is not the
        # SM90/100/103 DeepGEMM workspace formula). Park the SM until that
        # workspace policy is separately validated. L40S probe 2026-07-07:
        # SM89 hits the same guard (named in its raise), and stock 0.5.14
        # serving cannot reach DSV4 attention on SM89 at all — the
        # server_args DeepseekV4 hook (server_args.py:3786-3810) disables
        # SGLANG_OPT_DEEPGEMM_HC_PRENORM only on SM120/HIP, so SM89 keeps
        # the default-True flag (environ.py:781) while ENABLE_JIT_DEEPGEMM
        # is False and mHC crashes with NameError first.
        unverified_sms=(89, 120),
    ),
    OpEntry(
        op="dsv4_hca_context_module",
        module="collector.sglang.collect_dsv4_attn",
        get_func="get_dsv4_hca_context_test_cases",
        run_func="run_dsv4_attn_worker",
        perf_filename=PerfFile.DSV4_HCA_CONTEXT_MODULE,
        # L40S probe 2026-07-07: every SM89 case dies in-kernel (CUDA
        # InternalError at the first shape) — the sgl-kernel compressed
        # FlashMLA family has no SM89 target, and serving is blocked earlier
        # by the mHC prenorm NameError (see the CSA context entry).
        unverified_sms=(89,),
    ),
    OpEntry(
        op="dsv4_csa_generation_module",
        module="collector.sglang.collect_dsv4_attn",
        get_func="get_dsv4_csa_generation_test_cases",
        run_func="run_dsv4_attn_worker",
        perf_filename=PerfFile.DSV4_CSA_GENERATION_MODULE,
        # L40S probe 2026-07-07: same SM89 in-kernel CUDA InternalError as
        # the HCA modules (no SM89 FlashMLA target; serving blocked by the
        # mHC prenorm NameError regardless).
        unverified_sms=(89,),
    ),
    OpEntry(
        op="dsv4_hca_generation_module",
        module="collector.sglang.collect_dsv4_attn",
        get_func="get_dsv4_hca_generation_test_cases",
        run_func="run_dsv4_attn_worker",
        perf_filename=PerfFile.DSV4_HCA_GENERATION_MODULE,
        # L40S probe 2026-07-07: see dsv4_hca_context_module — SM89 parked.
        unverified_sms=(89,),
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
        # SM120 selects the Torch/v1 indexer paths and its whole DSV4 module
        # family is source-derived, hardware-unvalidated; pre-Hopper is
        # excluded by the op_min_sm=90 capability floor.  The 2026-07 RTX
        # 6000 Pro bring-up left this marker in place: the op is
        # experimental-only (no default-plan scheduling, no SDK call-site —
        # see DeepseekV4ForCausalLM_cases.yaml), so there is nothing to
        # hardware-validate through the sanctioned plan.
        unverified_sms=(120,),
    ),
    # The standalone HCA/CSA FMLA sub-kernel collectors require the
    # ``flash_mla`` package, which the pinned 0.5.14 image does not ship (the
    # serving path uses the bundled sgl-kernel FlashMLA instead); no platform
    # has ever produced rows for them. Keep them parked as unverified until a
    # wired backend exists, instead of letting the getter silently enumerate
    # zero cases.
    OpEntry(
        op="dsv4_hca_attn_module",
        module="collector.sglang.deepseekv4_sparse_modules",
        get_func="get_dsv4_hca_attn_test_cases",
        run_func="run_dsv4_sparse_kernel_worker",
        perf_filename=PerfFile.DSV4_HCA_ATTN_MODULE,
        unverified=True,
    ),
    OpEntry(
        op="dsv4_csa_attn_module",
        module="collector.sglang.deepseekv4_sparse_modules",
        get_func="get_dsv4_csa_attn_test_cases",
        run_func="run_dsv4_sparse_kernel_worker",
        perf_filename=PerfFile.DSV4_CSA_ATTN_MODULE,
        unverified=True,
    ),
    # CSA topk_512 DELTA calibration (flat vs top_last) — feeds the
    # degenerate->representative topK correction in perf_database.
    OpEntry(
        op="dsv4_csa_topk_calib",
        module="collector.sglang.deepseekv4_sparse_modules",
        get_func="get_dsv4_topk_calib_test_cases",
        run_func="run_dsv4_sparse_kernel_worker",
        perf_filename=PerfFile.DSV4_CSA_TOPK_CALIB,
        # SM120 serving disables topk v2 and takes v1 for both phases (see
        # DSV4-TOPK-PHASE-AND-SM120). The RTX 6000 Pro probe (2026-07-06)
        # showed the un-reviewed producer emits v2 rows on SM120 anyway
        # (309/310 launched; one v2 shape fails a jit_kernel runtime check) —
        # wrong-variant rows for that platform. Keep SM120 parked until the
        # producer is made variant-aware and the whole-module CSA path is
        # validated; pre-Hopper is excluded by the op_min_sm=90 capability
        # floor.
        unverified_sms=(120,),
    ),
    # GLM-5 DSA sparse sub-kernels (mqa / topk / dsa_attn) — GLM-5 analogue of
    # the DSV4 sparse family; shapes 1:1 from the GLM-5 DSA module CSV.
    # GLM-5 sparse sub-kernels share the DeepGEMM fp8_mqa_logits / FlashMLA
    # sparse family: pre-Hopper is excluded by the op_min_sm=90 capability
    # floors. SM120 stays parked after the RTX 6000 Pro probe (2026-07-06):
    # mqa raises DeepGEMM "Unsupported architecture" (attention.hpp:184) and
    # dsa_attn raises "Sparse Attention Forward Kernel is only supported on
    # SM90a and SM100f", so the GLM-5 DSA serving family is unrunnable on
    # SM120 at 0.5.14. The topk kernel itself ran clean (32/32 smoke points,
    # fast_topk_transform_fused), but rows for a family serving can never run
    # there would be non-serving data — park all three together.
    OpEntry(
        op="glm5_mqa_logits_module",
        module="collector.sglang.glm5_dsa_sparse_modules",
        get_func="get_glm5_mqa_test_cases",
        run_func="run_glm5_dsa_sparse_kernel_worker",
        perf_filename=PerfFile.GLM5_MQA_LOGITS_MODULE,
        unverified_sms=(120,),
    ),
    OpEntry(
        op="glm5_topk_module",
        module="collector.sglang.glm5_dsa_sparse_modules",
        get_func="get_glm5_topk_test_cases",
        run_func="run_glm5_dsa_sparse_kernel_worker",
        perf_filename=PerfFile.GLM5_TOPK_MODULE,
        unverified_sms=(120,),
    ),
    OpEntry(
        op="glm5_dsa_attn_module",
        module="collector.sglang.glm5_dsa_sparse_modules",
        get_func="get_glm5_dsa_attn_test_cases",
        run_func="run_glm5_dsa_sparse_kernel_worker",
        perf_filename=PerfFile.GLM5_DSA_ATTN_MODULE,
        unverified_sms=(120,),
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
        # L40S probe 2026-07-07: both directions crash in stock 0.5.14 —
        # the server_args DeepseekV4 hook (server_args.py:3786-3810) turns
        # SGLANG_OPT_DEEPGEMM_HC_PRENORM off only for SM120/HIP, so SM89
        # keeps the default True (environ.py:781) and the pre/post paths
        # call deep_gemm.tf32_hc_prenorm_gemm with deep_gemm never imported
        # (ENABLE_JIT_DEEPGEMM=False on SM89) -> NameError at
        # deep_gemm_wrapper/entrypoint.py:197. Serving crashes identically;
        # park SM89 until a framework fix ships.
        unverified_sms=(89,),
    ),
]
