# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Declarative registry mapping ops to collector modules.

TRT-LLM collectors target the current manifest runtime. Each module file still
declares its precise ``__compat__`` constraint, which is validated at runtime.
"""

from collector.registry_types import OpEntry, PerfFile

REGISTRY: list[OpEntry] = [
    OpEntry(
        op="gemm",
        module="collector.trtllm.collect_gemm",
        get_func="get_gemm_test_cases",
        run_func="run_gemm",
        perf_filename=PerfFile.GEMM,
    ),
    OpEntry(
        op="compute_scale",
        module="collector.trtllm.collect_computescale",
        get_func="get_computescale_test_cases",
        run_func="run_computescale",
        perf_filename=PerfFile.COMPUTESCALE,
    ),
    OpEntry(
        op="mla_context",
        module="collector.trtllm.collect_mla",
        get_func="get_context_mla_test_cases",
        run_func="run_mla",
        perf_filename=PerfFile.CONTEXT_MLA,
    ),
    OpEntry(
        op="mla_generation",
        module="collector.trtllm.collect_mla",
        get_func="get_generation_mla_test_cases",
        run_func="run_mla",
        perf_filename=PerfFile.GENERATION_MLA,
    ),
    OpEntry(
        op="attention_context",
        module="collector.trtllm.collect_attn",
        get_func="get_context_attention_test_cases",
        run_func="run_attention_torch",
        perf_filename=PerfFile.CONTEXT_ATTENTION,
    ),
    OpEntry(
        op="attention_generation",
        module="collector.trtllm.collect_attn",
        get_func="get_generation_attention_test_cases",
        run_func="run_attention_torch",
        perf_filename=PerfFile.GENERATION_ATTENTION,
    ),
    OpEntry(
        op="encoder_attention",
        module="collector.trtllm.collect_attn_encoder",
        get_func="get_encoder_attention_test_cases",
        run_func="run_encoder_attention_torch",
        perf_filename=PerfFile.ENCODER_ATTENTION,
    ),
    OpEntry(
        op="mla_bmm_gen_pre",
        module="collector.trtllm.collect_mla_bmm",
        get_func="get_mla_gen_pre_test_cases",
        run_func="run_mla_gen_pre",
        perf_filename=PerfFile.MLA_BMM,
    ),
    OpEntry(
        op="mla_bmm_gen_post",
        module="collector.trtllm.collect_mla_bmm",
        get_func="get_mla_gen_post_test_cases",
        run_func="run_mla_gen_post",
        perf_filename=PerfFile.MLA_BMM,
    ),
    OpEntry(
        op="moe",
        module="collector.trtllm.collect_moe",
        get_func="get_moe_test_cases",
        run_func="run_moe_torch",
        perf_filename=PerfFile.MOE,
    ),
    OpEntry(
        op="mamba2",
        module="collector.trtllm.collect_mamba2",
        get_func="get_mamba2_test_cases",
        run_func="run_mamba2_torch",
        perf_filename=PerfFile.MAMBA2,
    ),
    OpEntry(
        op="gdn",
        module="collector.trtllm.collect_gdn",
        get_func="get_gdn_test_cases",
        run_func="run_gdn_torch",
        perf_filename=PerfFile.GDN,
    ),
    OpEntry(
        op="mla_context_module",
        module="collector.trtllm.collect_mla_module",
        get_func="get_mla_context_module_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.MLA_CONTEXT_MODULE,
        # fp8-KV MLA module combos are hardware-validated on SM90/100/103/120
        # (see collect_mla_module._get_precision_combos); SM121 has never run
        # them on hardware — cases are queued there and this marker records
        # the maturity gap (layer_permissions.md registry markers).
        unverified_sms=(121,),
    ),
    OpEntry(
        op="mla_generation_module",
        module="collector.trtllm.collect_mla_module",
        get_func="get_mla_generation_module_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.MLA_GENERATION_MODULE,
        # fp8-KV MLA module combos are hardware-validated on SM90/100/103/120
        # (see collect_mla_module._get_precision_combos); SM121 has never run
        # them on hardware — cases are queued there and this marker records
        # the maturity gap (layer_permissions.md registry markers).
        unverified_sms=(121,),
    ),
    OpEntry(
        op="dsa_context_module",
        module="collector.trtllm.collect_mla_module",
        get_func="get_dsa_context_module_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.DSA_CONTEXT_MODULE,
    ),
    OpEntry(
        op="dsa_generation_module",
        module="collector.trtllm.collect_mla_module",
        get_func="get_dsa_generation_module_test_cases",
        run_func="run_mla_module_worker",
        perf_filename=PerfFile.DSA_GENERATION_MODULE,
    ),
    OpEntry(
        op="mhc_module",
        module="collector.trtllm.collect_mhc_module",
        get_func="get_mhc_module_test_cases",
        run_func="run_mhc_module_worker",
        perf_filename=PerfFile.MHC_MODULE,
    ),
    # DeepSeek-V4 CSA/HCA attention modules. Requires trtllm>=1.3.0rc21
    # (module __compat__); the framework itself rejects pre-Blackwell GPUs
    # (mla.py forward_*_sparse_mla "DeepSeek-V4 is not supported on
    # pre-blackwell GPUs" @1.3.0rc23) — those platforms fail into the
    # classified log per observe-don't-predict.
    # SM120 probe (RTX PRO 6000, 1.3.0rc23, campaign 2026-08-07): 100% of
    # cases fail from case 0 (~5.4s/case classified errors with SIGABRT
    # worker resets), matching the DeepGEMM sparse-attention "Unsupported
    # architecture" family limit already documented for DSA in
    # collect_mla_module.py — park SM120 until a framework fix ships.
    OpEntry(
        op="dsv4_csa_context_module",
        module="collector.trtllm.collect_dsv4_attn",
        get_func="get_dsv4_csa_context_test_cases",
        run_func="run_dsv4_attn_worker",
        perf_filename=PerfFile.DSV4_CSA_CONTEXT_MODULE,
        unverified_sms=(120,),
    ),
    OpEntry(
        op="dsv4_hca_context_module",
        module="collector.trtllm.collect_dsv4_attn",
        get_func="get_dsv4_hca_context_test_cases",
        run_func="run_dsv4_attn_worker",
        perf_filename=PerfFile.DSV4_HCA_CONTEXT_MODULE,
        unverified_sms=(120,),
    ),
    OpEntry(
        op="dsv4_csa_generation_module",
        module="collector.trtllm.collect_dsv4_attn",
        get_func="get_dsv4_csa_generation_test_cases",
        run_func="run_dsv4_attn_worker",
        perf_filename=PerfFile.DSV4_CSA_GENERATION_MODULE,
        unverified_sms=(120,),
    ),
    OpEntry(
        op="dsv4_hca_generation_module",
        module="collector.trtllm.collect_dsv4_attn",
        get_func="get_dsv4_hca_generation_test_cases",
        run_func="run_dsv4_attn_worker",
        perf_filename=PerfFile.DSV4_HCA_GENERATION_MODULE,
        unverified_sms=(120,),
    ),
    OpEntry(
        op="msa_context_module",
        module="collector.trtllm.collect_msa_module",
        get_func="get_msa_context_module_test_cases",
        run_func="run_msa_module_worker",
        perf_filename=PerfFile.MSA_CONTEXT_MODULE,
        # MiniMax-M3 MSA modules: hardware-validated on SM90 (h100/h200,
        # rc23 Triton reference path) and SM100/103 (b200/b300/gb200/gb300,
        # rc23 implementation="msa" fmha_sm100 path — see collect_msa_module).
        # SM120 runs the Triton path; its table is pending (collection-pool
        # availability) and lands in a follow-up — until then M3-on-SM120
        # trtllm estimates fail with a typed EmpiricalNotImplemented (no MSA
        # table and no DSA xop donor on that system/backend): an explicitly
        # rejected cell, not a silent fallback. SM121 has never run on
        # hardware and stays marked.
        unverified_sms=(121,),
    ),
    OpEntry(
        op="msa_generation_module",
        module="collector.trtllm.collect_msa_module",
        get_func="get_msa_generation_module_test_cases",
        run_func="run_msa_module_worker",
        perf_filename=PerfFile.MSA_GENERATION_MODULE,
        # See msa_context_module marker rationale.
        unverified_sms=(121,),
    ),
]
