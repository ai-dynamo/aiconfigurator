# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM production-path MoE collector.

Shared MoE cases come from YAML. Every quantization mode is built through
``FusedMoE`` so vLLM owns routing, TP/EP sharding, weight post-processing, and
backend selection exactly as it does for model execution.

Runtime: stock vLLM v0.24.0 (owner decision 2026-08-02, reverting the
2026-08-01 family-wide preview pin — a temporary build must not own a
family default). K3 SITU cases collect on stock via the owner-approved
situ-as-silu Marlin approximation (2026-08-02): serving truth for the
ct-mxfp4+situ checkpoint is MarlinExperts W4A16 on every SM (the preview
build's CUTLASS activation whitelist rejects SITU), so the collector
forces the same Marlin selection on stock, runs the elementwise epilogue
as silu (<~3% of the op), and marks the rows' kernel_source with
`_situ_as_silu`. On the preview build itself the betas pass through
natively. Re-verify at every version bump: when upstream wires SITU into
a quantized-activation kernel (the preview already carries an unwired
trtllm-gen SITU mapping), the serving kernel class changes and this
approximation must be retired.
"""

__compat__ = "vllm==0.24.0"

import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from vllm.version import __version__ as vllm_version

from collector.case_generator import (
    get_common_moe_test_cases,
    get_moe_quantization_modes,
    get_moe_quantization_module_config,
    moe_model_allows_quantization,
)
from collector.helper import (
    balanced_logits,
    benchmark_with_power,
    get_sm_version,
    log_perf,
    power_law_logits_v3,
)

aic_debug = int(os.getenv("aic_moe_debug", "0"))  # noqa: SIM112
_MODEL_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "src/aiconfigurator/model_configs"


def _load_model_moe_config(model_name: str) -> dict:
    """Load the checked-in HF config used to derive vLLM's FusedMoE args."""
    config_path = _MODEL_CONFIG_ROOT / f"{model_name.replace('/', '--')}_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing packaged model config for vLLM MoE case {model_name!r}: {config_path}")

    with config_path.open() as config_file:
        config = json.load(config_file)
    return config.get("text_config", config)


def _resolve_moe_runtime_config(model_name: str, module_config: dict) -> dict:
    """Resolve the FusedMoE arguments used by vLLM 0.24 model code."""
    model_config = _load_model_moe_config(model_name)
    model_type = str(model_config.get("model_type", ""))
    activation = str(
        module_config.get("activation")
        or model_config.get("hidden_act")
        or model_config.get("hidden_activation")
        or "silu"
    )

    use_grouped_topk = model_type in {
        "deepseek_v3",
        "kimi_k2",
        # Kimi-K3 (kimi_linear): config declares use_grouped_topk=true with
        # one expert group (num_expert_group=1, topk_group=1, noaux_tc), and
        # the kimi-k3 branch model honors the flag directly
        # (models/kimi_k3/nvidia/model.py:411-413 @ 0.1.dev19262).
        "kimi_linear",
        "mimo_v2_flash",
        "glm_moe_dsa",
        "nemotron_h",
    }
    declares_grouped_routing = (
        model_config.get("n_group") is not None
        or model_config.get("topk_group") is not None
        or model_config.get("topk_method") == "noaux_tc"
    )
    # DeepSeek V4 declares topk_method=noaux_tc but routes UNGROUPED in vLLM
    # 0.24: its FusedMoE gets no expert grouping — only the noaux_tc
    # e_score_correction_bias with sqrtsoftplus scoring and float32 router
    # logits (models/deepseek_v4/nvidia/model.py:652-669, 557-561 @0.24.0).
    # The bias/scoring fields below already carry that. Both V4 model_types
    # share that model code: "deepseek_ref" (the sgl-project FP8 requants)
    # and "deepseek_v4" (the native expert_dtype=fp4 checkpoints).
    ungrouped_noaux_tc = model_type in ("deepseek_ref", "deepseek_v4")
    if declares_grouped_routing and not use_grouped_topk and not ungrouped_noaux_tc:
        raise ValueError(
            f"vLLM MoE model {model_name!r} (model_type={model_type!r}) declares grouped/noaux_tc "
            "routing fields but is not a recognized grouped-topk model type; verify how vLLM 0.24 "
            "routes this model and extend the mapping instead of silently benchmarking non-grouped routing"
        )
    # Some upstream configs spell the routing contract differently (Step3p5/3p7
    # use "use_moe_router_bias"/"moe_router_scaling_factor"). Read both spellings:
    # a curated config that drops the vendor key silently benchmarks unbiased
    # routing while vLLM serves biased routing, which the guard above cannot see
    # because the vendor key is not one of the grouped/noaux_tc fields it checks.
    declared_routing_bias = model_config.get("use_routing_bias")
    if declared_routing_bias is None:
        declared_routing_bias = model_config.get("use_moe_router_bias")
    use_routing_bias = (
        model_config.get("topk_method") == "noaux_tc"
        or bool(declared_routing_bias)
        or model_type in {"mimo_v2_flash", "glm_moe_dsa", "nemotron_h"}
    )
    scoring_func = str(model_config.get("scoring_func") or "softmax")
    if model_type == "nemotron_h":
        scoring_func = "sigmoid"

    custom_routing = None
    apply_router_weight_on_input = False
    renormalize = bool(model_config.get("norm_topk_prob", True))
    if model_type == "llama4_text":
        custom_routing = "llama4"
        apply_router_weight_on_input = True
        renormalize = False
    elif model_type == "gemma4_text":
        custom_routing = "gemma4"
        activation = "gelu_tanh"
    elif model_type == "nemotron_h":
        activation = str(model_config["mlp_hidden_act"])
        activation = {
            "gelu_pytorch_tanh": "gelu_tanh",
        }.get(activation, activation)
        activation = f"{activation}_no_mul"

    if use_grouped_topk:
        # DeepSeek-family configs spell the group count `n_group`; Kimi-K3
        # (kimi_linear) spells it `num_expert_group`, which is the attribute
        # the branch model reads (models/kimi_k3/nvidia/model.py:412).
        raw_n_group = model_config.get("n_group", model_config.get("num_expert_group"))
        missing_fields = [
            field
            for field, value in (("n_group", raw_n_group), ("topk_group", model_config.get("topk_group")))
            if value is None
        ]
        if missing_fields:
            raise ValueError(f"vLLM grouped-topk model {model_name!r} is missing {', '.join(missing_fields)}")
        try:
            num_expert_group = int(raw_n_group)
            topk_group = int(model_config["topk_group"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"vLLM grouped-topk model {model_name!r} requires integer group fields") from error
        if num_expert_group <= 0 or topk_group <= 0 or topk_group > num_expert_group:
            raise ValueError(
                f"vLLM grouped-topk model {model_name!r} has invalid "
                f"n_group={num_expert_group}, topk_group={topk_group}"
            )
    else:
        num_expert_group = None
        topk_group = None

    return {
        "renormalize": renormalize,
        "scoring_func": scoring_func,
        "activation": activation,
        # SiTU (Kimi-K3) is parameterized: FusedMoE requires the betas when
        # activation == "situ" (the TRTLLM experts assert beta > 0; the
        # marlin epilogue reads both). Sourced from the packaged config.
        "activation_situ_beta": (model_config.get("activation_situ_beta") if activation == "situ" else None),
        "activation_situ_linear_beta": (
            model_config.get("activation_situ_linear_beta") if activation == "situ" else None
        ),
        "routed_scaling_factor": float(
            model_config.get("routed_scaling_factor") or model_config.get("moe_router_scaling_factor") or 1.0
        ),
        "swiglu_limit": model_config.get("swiglu_limit") if model_type in ("deepseek_v4", "deepseek_ref") else None,
        "use_grouped_topk": use_grouped_topk,
        "num_expert_group": num_expert_group,
        "topk_group": topk_group,
        "apply_routed_scale_to_output": model_type in {"deepseek_v3", "kimi_k2", "glm_moe_dsa", "nemotron_h"},
        "use_routing_bias": use_routing_bias,
        "router_logits_float32": use_routing_bias or scoring_func in {"sigmoid", "sqrtsoftplus"},
        "custom_routing": custom_routing,
        "apply_router_weight_on_input": apply_router_weight_on_input,
        "has_bias": bool(module_config.get("has_bias", False)),
    }


def _moe_execution_key(common_moe_testcase, moe_type: str):
    module_config = get_moe_quantization_module_config(
        "vllm",
        moe_type,
        model_name=common_moe_testcase.model_name,
    )
    routing_config = _resolve_moe_runtime_config(common_moe_testcase.model_name, module_config)
    return (
        moe_type,
        tuple(common_moe_testcase.num_tokens_list),
        common_moe_testcase.hidden_size,
        common_moe_testcase.inter_size,
        common_moe_testcase.topk,
        common_moe_testcase.num_experts,
        common_moe_testcase.tp,
        common_moe_testcase.ep,
        common_moe_testcase.token_expert_distribution,
        common_moe_testcase.power_law_alpha,
        json.dumps(module_config, sort_keys=True, separators=(",", ":")),
        json.dumps(routing_config, sort_keys=True, separators=(",", ":")),
    )


def _moe_consumer_keys(common_moe_testcase, moe_type: str):
    """Return every consumer-visible key emitted by one getter task."""
    distribution = (
        f"power_law_{common_moe_testcase.power_law_alpha}"
        if common_moe_testcase.token_expert_distribution == "power_law"
        else common_moe_testcase.token_expert_distribution
    )
    return tuple(
        (
            moe_type,
            distribution,
            common_moe_testcase.topk,
            common_moe_testcase.num_experts,
            common_moe_testcase.hidden_size,
            common_moe_testcase.inter_size,
            common_moe_testcase.tp,
            common_moe_testcase.ep,
            num_tokens,
        )
        for num_tokens in common_moe_testcase.num_tokens_list
    )


def get_moe_test_cases():
    """Generate MoE test cases"""

    sm = get_sm_version()
    enabled_moe_types = get_moe_quantization_modes(
        "vllm",
        sm_version=sm,
        runtime_version=vllm_version,
        runtime_features={
            "per_block_fp8": True,
            "nvfp4": True,
            "mxfp4": True,
        },
    )

    test_cases = []
    seen = set()
    consumer_key_owners = {}
    quant_policy_drops = {}
    models_with_cases = set()

    for common_moe_testcase in get_common_moe_test_cases(backend="vllm"):
        model_name = common_moe_testcase.model_name

        for moe_type in enabled_moe_types:
            if not moe_model_allows_quantization("vllm", model_name, moe_type):
                quant_policy_drops[model_name] = quant_policy_drops.get(model_name, 0) + 1
                continue
            models_with_cases.add(model_name)

            execution_key = _moe_execution_key(common_moe_testcase, moe_type)
            if execution_key in seen:
                continue
            consumer_keys = _moe_consumer_keys(common_moe_testcase, moe_type)
            for consumer_key in consumer_keys:
                previous_owner = consumer_key_owners.get(consumer_key)
                if previous_owner is not None and previous_owner[0] != execution_key:
                    previous_model = previous_owner[1]
                    raise ValueError(
                        "vLLM MoE population collision: "
                        f"models {previous_model!r} and {model_name!r} map distinct benchmark "
                        f"invocations to consumer key {consumer_key!r}; "
                        "the current moe_perf consumer cannot represent both"
                    )
            for consumer_key in consumer_keys:
                consumer_key_owners[consumer_key] = (execution_key, model_name)
            seen.add(execution_key)

            test_cases.append(
                [
                    moe_type,
                    common_moe_testcase.num_tokens_list,
                    common_moe_testcase.hidden_size,
                    common_moe_testcase.inter_size,
                    common_moe_testcase.topk,
                    common_moe_testcase.num_experts,
                    common_moe_testcase.tp,
                    common_moe_testcase.ep,
                    common_moe_testcase.model_name,
                    common_moe_testcase.token_expert_distribution,
                    common_moe_testcase.power_law_alpha,
                ]
            )

    # Zero-case expansions must be explainable from logged drops
    # (case_authoring.md): name every planned model whose declared vllm quant
    # policy excluded all of its moe cases.
    fully_dropped = sorted(set(quant_policy_drops) - models_with_cases)
    if fully_dropped:
        print(
            f"moe: dropped all cases for {len(fully_dropped)} model(s) by declared vllm "
            f"quantization policy (allowed_modes): {', '.join(fully_dropped)}"
        )

    return test_cases


def run_moe_torch(
    moe_type,
    num_tokens_lists,
    hidden_size,
    inter_size,
    topk,
    num_experts,
    moe_tp_size,
    moe_ep_size,
    model_name,
    distributed="power_law",
    power_law_alpha=0.0,
    *,
    perf_filename,
    device="cuda:0",
):
    """Benchmark the vLLM 0.24.0 model-execution MoE path."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.forward_context import get_forward_context, set_forward_context
    from vllm.model_executor.layers.fused_moe.experts.fallback import FallbackExperts
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
    from vllm.v1.worker.workspace import init_workspace_manager

    from collector.vllm.utils import setup_distributed

    if moe_tp_size > 1 and moe_ep_size > 1:
        raise ValueError("vLLM MoE collector does not combine logical TP and EP")

    setup_distributed(device)
    torch.cuda.set_device(device)
    init_workspace_manager(torch.device(device))
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)

    module_config = get_moe_quantization_module_config("vllm", moe_type, model_name=model_name)
    runtime_config = _resolve_moe_runtime_config(model_name, module_config)
    if runtime_config["use_grouped_topk"]:
        num_expert_group = runtime_config["num_expert_group"]
        topk_group = runtime_config["topk_group"]
        if num_experts <= num_expert_group or num_experts % num_expert_group != 0:
            raise ValueError(
                f"vLLM grouped-topk model {model_name!r} cannot divide {num_experts} experts "
                f"into {num_expert_group} groups"
            )
        experts_per_group = num_experts // num_expert_group
        if topk > topk_group * experts_per_group:
            raise ValueError(
                f"vLLM grouped-topk model {model_name!r} requests topk={topk}, but "
                f"topk_group={topk_group} exposes only {topk_group * experts_per_group} experts"
            )
    e_score_correction_bias = (
        torch.zeros(num_experts, dtype=torch.float32, device=device) if runtime_config["use_routing_bias"] else None
    )
    router_logits_dtype = torch.float32 if runtime_config["router_logits_float32"] else None
    custom_routing_function = None
    if runtime_config["custom_routing"] == "llama4":
        from vllm.model_executor.models.llama4 import Llama4MoE

        custom_routing_function = Llama4MoE.custom_routing_function
    elif runtime_config["custom_routing"] == "gemma4":
        from vllm.model_executor.models.gemma4 import gemma4_fused_routing_kernel_triton

        per_expert_scale = torch.ones(num_experts, device=device)

        def gemma4_routing_function(hidden_states, gating_output, topk, renormalize):
            return gemma4_fused_routing_kernel_triton(gating_output, topk, per_expert_scale)

        custom_routing_function = gemma4_routing_function

    if moe_type == "bfloat16":
        quant_config = None
    elif moe_type == "fp8":
        # Per-tensor "fp8" rows are anchored on ModelOpt static checkpoints
        # (e.g. Nemotron Ultra FP8). Serving loads those via
        # ModelOptFp8MoEMethod: static/static quant keys
        # (vllm/model_executor/layers/quantization/modelopt.py:749-771
        # @0.24.0) and non-gated-aware weight sizing (w13_num_shards = 2 if
        # is_act_and_mul else 1, modelopt.py:812). Fp8Config/Fp8MoEMethod
        # would diverge on both: activation_scheme="dynamic" resolves
        # kFp8DynamicTensorSym and selects a different backend than serving
        # (fp8.py:514-533), and its create_weights hardcodes gated 2x w13
        # (fp8.py:580) which breaks non-gated models like Nemotron.
        quant_config = ModelOptFp8Config(
            quant_method="FP8",
            is_checkpoint_fp8_serialized=True,
            kv_cache_quant_method=None,
            exclude_modules=[],
        )
    elif moe_type == "fp8_block":
        # Block-FP8 serving (DeepSeek-style checkpoints) is per-128-block
        # weights with dynamic per-group activations; Fp8Config rejects
        # static for block quant (fp8.py:130-134).
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
    elif moe_type == "w4a16_mxfp4":
        checkpoint_qc = _load_model_moe_config(model_name).get("quantization_config") or {}
        is_ct_mxfp4 = checkpoint_qc.get("quant_method") == "compressed-tensors" and "mxfp4" in str(
            checkpoint_qc.get("format", "")
        )
        if is_ct_mxfp4:
            # Kimi-K3-style compressed-tensors mxfp4-pack checkpoints: vLLM
            # resolves this descriptor to CompressedTensorsW4A4Mxfp4MoEMethod
            # (compressed_tensors_moe.py:66-71 @0.1.dev19262). Its CUTLASS
            # W4A4 path rejects the SITU activation
            # (CutlassExpertsMxfp4._supports_activation, cutlass_moe.py:
            # 321-327 — SILU/GELU/GELU_TANH/SWIGLUOAI only), so the method
            # falls back to weight-only MarlinExperts, which supports SITU
            # (compressed_tensors_moe_w4a4_mxfp4.py:58-71) — the w4a16
            # identity this lane persists. Constructing the checkpoint's own
            # config (instead of the OCP Mxfp4Config) lets that selection run
            # exactly as serving; OCP mxfp4 checkpoints (gpt-oss family) keep
            # the Mxfp4Config path below.
            quant_config = CompressedTensorsConfig.from_config(checkpoint_qc)
        else:
            quant_config = Mxfp4Config()
    elif moe_type == "w4a8_mxfp4_mxfp8":
        # Native DeepSeek-V4 (expert_dtype=fp4) serving path: vLLM overrides
        # the checkpoint's fp8 quant_method to DeepseekV4FP8Config
        # (models/deepseek_v4/quant_config.py:120-132 @0.24.0), which returns
        # Mxfp4MoEMethod for the routed experts (quant_config.py:142-152);
        # backend selection then runs select_deepseek_v4_mxfp4_moe_backend
        # exactly as in serving. Constructor args mirror the checkpoints'
        # quantization_config (fp8-serialized, dynamic, 128x128 blocks).
        from vllm.models.deepseek_v4.quant_config import DeepseekV4FP8Config

        expert_dtype = _load_model_moe_config(model_name).get("expert_dtype")
        if expert_dtype != "fp4":
            raise ValueError(
                f"vLLM MoE case {model_name!r} requests w4a8_mxfp4_mxfp8 but its packaged "
                f"config declares expert_dtype={expert_dtype!r}; only the native fp4-expert "
                "DeepSeek-V4 checkpoints run this path"
            )
        quant_config = DeepseekV4FP8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
    elif moe_type == "nvfp4":
        nvfp4_args = {
            "num_bits": 4,
            "type": "float",
            "strategy": "tensor_group",
            "group_size": 16,
            "symmetric": True,
            "dynamic": False,
        }
        quant_config = CompressedTensorsConfig.from_config(
            {
                "quant_method": "compressed-tensors",
                "format": "nvfp4-pack-quantized",
                "config_groups": {
                    "group_0": {
                        "format": "nvfp4-pack-quantized",
                        "weights": nvfp4_args,
                        "input_activations": nvfp4_args,
                        "output_activations": None,
                        "targets": ["Linear"],
                    }
                },
            }
        )
    elif moe_type == "int4_wo":
        group_size = int(module_config.get("group_size", 128))
        quant_config = CompressedTensorsConfig.from_config(
            {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "config_groups": {
                    "group_0": {
                        "format": "pack-quantized",
                        "weights": {
                            "num_bits": 4,
                            "type": "int",
                            "strategy": "group",
                            "group_size": group_size,
                            "symmetric": True,
                            "dynamic": False,
                        },
                        "input_activations": None,
                        "output_activations": None,
                        "targets": ["Linear"],
                    }
                },
            }
        )
    else:
        raise ValueError(f"Unsupported vLLM MoE quantization mode: {moe_type}")

    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        hf_text_config=SimpleNamespace(model_type=""),
        model="collector_dummy",
        quantization_config=None,
    )
    if moe_type == "w4a8_mxfp4_mxfp8":
        # DeepseekV4FP8Config resolves expert_dtype lazily from the current
        # vllm_config's hf_config (models/deepseek_v4/quant_config.py:57-78
        # @0.24.0); mirror the native checkpoint field so the RoutedExperts
        # dispatch resolves to the MXFP4 expert path deterministically.
        vllm_config.model_config.hf_config = SimpleNamespace(
            model_type="deepseek_v4",
            expert_dtype="fp4",
        )
    vllm_config.scheduler_config.max_num_batched_tokens = max(num_tokens_lists)
    vllm_config.parallel_config.enable_expert_parallel = moe_ep_size > 1

    # vLLM represents EP by flattening the requested TP world and then
    # converting it to EP when enable_expert_parallel is set. The collector is
    # one physical rank, but requested logical sizes determine local weights
    # and the rank-0 expert map exactly as in a multi-rank model.
    logical_parallel_size = moe_ep_size if moe_ep_size > 1 else moe_tp_size

    # SITU activation (Kimi-K3) exists only on the kimi-k3 preview build
    # (stock v0.24.0 has no situ path — upstream v0.24.0 activation.py).
    # Probe the constructor instead of predicting: on the preview build,
    # pass the betas through. On stock, apply the owner-approved
    # approximation (2026-08-02): serving truth for (ct-mxfp4, situ) is
    # MarlinExperts weight-only W4A16 — the preview rejects SITU in its
    # CUTLASS activation whitelist and falls to Marlin
    # (compressed_tensors_moe_w4a4_mxfp4.py:58-71 @0.1.dev19262). Stock's
    # device-only gate would instead send this checkpoint to CUTLASS W4A4
    # on Blackwell (cutlass_moe.py:693-699 @v0.24.0) — a kernel K3 serving
    # never runs — so the CUTLASS gate is forced off to reproduce the
    # serving selection, and the epilogue runs as silu (silu_and_mul vs
    # situ_and_mul is an elementwise tail, <~3% of the op). The
    # approximation is recorded in kernel_source (`_situ_as_silu` suffix)
    # and hard-checked below: it is only valid if Marlin actually ran.
    situ_kwargs = {}
    situ_marlin_approx = False
    if runtime_config["activation"] == "situ":
        from inspect import signature as _sig

        if "activation_situ_beta" in _sig(FusedMoE.__init__).parameters:
            situ_kwargs = {
                "activation_situ_beta": runtime_config["activation_situ_beta"],
                "activation_situ_linear_beta": runtime_config["activation_situ_linear_beta"],
            }
        else:
            runtime_config["activation"] = "silu"
            situ_marlin_approx = True

    @contextlib.contextmanager
    def _serving_truth_marlin_selection():
        """Force the Marlin selection K3 serving makes for situ checkpoints.

        On Hopper this is a no-op (stock's device gate already excludes
        CUTLASS mxfp4 below SM100)."""
        if not situ_marlin_approx:
            yield
            return
        from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
            CutlassExpertsMxfp4,
        )

        orig = CutlassExpertsMxfp4.__dict__["_supports_current_device"]
        CutlassExpertsMxfp4._supports_current_device = staticmethod(lambda: False)
        try:
            yield
        finally:
            CutlassExpertsMxfp4._supports_current_device = orig

    with set_current_vllm_config(vllm_config), _serving_truth_marlin_selection():
        moe_module = FusedMoE(
            num_experts=num_experts,
            top_k=topk,
            hidden_size=hidden_size,
            intermediate_size=inter_size,
            params_dtype=torch.bfloat16,
            renormalize=runtime_config["renormalize"],
            use_grouped_topk=runtime_config["use_grouped_topk"],
            num_expert_group=runtime_config["num_expert_group"],
            topk_group=runtime_config["topk_group"],
            quant_config=quant_config,
            tp_size=logical_parallel_size,
            dp_size=1,
            pcp_size=1,
            prefix="collector_moe",
            custom_routing_function=custom_routing_function,
            scoring_func=runtime_config["scoring_func"],
            routed_scaling_factor=runtime_config["routed_scaling_factor"],
            swiglu_limit=runtime_config["swiglu_limit"],
            e_score_correction_bias=e_score_correction_bias,
            apply_router_weight_on_input=runtime_config["apply_router_weight_on_input"],
            has_bias=runtime_config["has_bias"],
            activation=runtime_config["activation"],
            router_logits_dtype=router_logits_dtype,
            apply_routed_scale_to_output=runtime_config["apply_routed_scale_to_output"],
            **situ_kwargs,
        )
        moe_module.to(device)
        moe_module.eval()
        moe_module.requires_grad_(False)

        # Populate the checkpoint-native tensors created by the selected
        # quantization method, then let that method convert them to the exact
        # runtime layout and construct the production kernel.
        routed_experts = moe_module.routed_experts
        with torch.no_grad():
            for name, parameter in routed_experts.named_parameters(recurse=False):
                if parameter.dtype == torch.uint8:
                    if "scale" in name:
                        parameter.fill_(127)
                    else:
                        parameter.random_(0, 256)
                elif parameter.dtype == torch.int32:
                    parameter.random_(0, 16)
                elif parameter.is_floating_point():
                    parameter.fill_(1.0 if "scale" in name else 0.01)
                else:
                    parameter.zero_()

        quant_method = routed_experts.quant_method
        # API-compat shim for stock v0.24.0 Marlin weight prep under EP:
        # prepare_moe_fp4_layer_for_marlin sizes its repack loop and shape
        # assert from moe_config.num_experts (GLOBAL), while create_weights
        # shards the expert dimension locally (marlin_utils_fp4.py:460-481
        # @v0.24.0: e=moe_config.num_experts vs w13 created with the local
        # count). The preview build preps EP-local weights fine (972 EP
        # rows collected there). Present the local count during weight prep
        # only, then restore the global count — routing needs it. Metadata
        # presentation only; the invoked kernels are unchanged.
        _mc = getattr(routed_experts, "moe_config", None)
        _local_e = getattr(routed_experts, "num_experts", None)
        _swap = situ_marlin_approx and _mc is not None and _local_e is not None and _mc.num_experts != _local_e
        if _swap:
            _global_e = _mc.num_experts
            try:
                _mc.num_experts = _local_e
            except AttributeError:
                object.__setattr__(_mc, "num_experts", _local_e)
        try:
            quant_method.process_weights_after_loading(routed_experts)
        finally:
            if _swap:
                try:
                    _mc.num_experts = _global_e
                except AttributeError:
                    object.__setattr__(_mc, "num_experts", _global_e)

        selected_backend = None
        for backend_attr in (
            "unquantized_backend",
            "fp8_backend",
            "nvfp4_backend",
            "mxfp4_backend",
            "wna16_backend",
        ):
            if hasattr(quant_method, backend_attr):
                selected_backend = getattr(quant_method, backend_attr)
                break
        backend_name = getattr(selected_backend, "value", str(selected_backend))
        moe_kernel = getattr(quant_method, "moe_kernel", None)
        experts_impl = getattr(moe_kernel, "fused_experts", None)
        wrapper_name = type(experts_impl).__name__ if experts_impl is not None else "direct"
        print(
            "vLLM MoE path:",
            type(quant_method).__name__,
            f"backend={backend_name}",
            f"tp={moe_module.moe_config.tp_size}",
            f"ep={moe_module.moe_config.ep_size}",
            f"routing={moe_module.router.routing_method_type}",
            f"activation={runtime_config['activation']}",
        )

        method_name = type(quant_method).__name__.removesuffix("Method")

        # Performance testing for each token count.
        for num_tokens in num_tokens_lists:
            print("num_tokens", num_tokens)
            print("topk", topk)
            hidden_states = torch.randn([num_tokens, hidden_size], dtype=torch.bfloat16, device=device)
            if isinstance(experts_impl, FallbackExperts):
                leaf_experts = experts_impl._select_experts_impl(
                    hidden_states,
                    routed_experts.w13_weight,
                    routed_experts.w2_weight,
                )
                if leaf_experts is None:
                    raise RuntimeError(f"vLLM MoE wrapper {wrapper_name} did not select an experts implementation")
            else:
                leaf_experts = experts_impl
            experts_name = type(leaf_experts).__name__ if leaf_experts is not None else "direct"
            print(f"vLLM MoE experts: wrapper={wrapper_name} leaf={experts_name}")
            source = f"vllm_{method_name}_{backend_name}_{experts_name}".lower()
            source = source.replace(" ", "_").replace("-", "_")
            if situ_marlin_approx:
                # The situ->silu approximation is only valid on the Marlin
                # path serving actually selects; anything else is wrong data.
                if "marlin" not in experts_name.lower():
                    raise RuntimeError(
                        f"situ-as-silu approximation requires MarlinExperts "
                        f"(serving truth for ct-mxfp4+situ), but the runtime "
                        f"selected {experts_name} despite the forced CUTLASS "
                        f"gate — refusing to record mislabeled rows."
                    )
                source += "_situ_as_silu"

            num_iter = 5 if distributed == "power_law" else 1
            logits_dtype = router_logits_dtype or torch.bfloat16
            if distributed == "power_law":
                # The workload generator's smoothing Conv1d consumes FP32.
                # Module construction uses a BF16 default dtype, so restore
                # FP32 only while generating logits and then put the collector
                # default back exactly as it was.
                previous_dtype = torch.get_default_dtype()
                torch.set_default_dtype(torch.float32)
                try:
                    router_logits_list = [
                        power_law_logits_v3(num_tokens, num_experts, topk, moe_ep_size, power_law_alpha)
                        .to(logits_dtype)
                        .to(device)
                        for _ in range(num_iter)
                    ]
                finally:
                    torch.set_default_dtype(previous_dtype)
            elif distributed == "balanced":
                router_logits_list = [balanced_logits(num_tokens, num_experts, topk).to(logits_dtype).to(device)]
            else:
                raise ValueError(f"Unsupported distributed mode: {distributed}")
            num_warmups = 1 if distributed == "power_law" else 3
            num_runs = 1 if distributed == "power_law" else 6

            def run_single_iteration():
                for router_logits in router_logits_list:
                    forward_context = get_forward_context()
                    forward_context.moe_layer_index = 0
                    moe_module(hidden_states, router_logits)

            with (
                set_forward_context({}, vllm_config),
                benchmark_with_power(
                    device=device,
                    kernel_func=run_single_iteration,
                    num_warmups=num_warmups,
                    num_runs=num_runs,
                    repeat_n=1,
                ) as results,
            ):
                pass

            latency = results["latency_ms"] / num_iter
            power_stats = results["power_stats"]
            print(f"moe latency: {latency}")

            log_perf(
                item_list=[
                    {
                        "moe_dtype": moe_type,
                        "num_tokens": num_tokens,
                        "hidden_size": hidden_size,
                        "inter_size": inter_size,
                        "topk": topk,
                        "num_experts": num_experts,
                        "moe_tp_size": moe_tp_size,
                        "moe_ep_size": moe_ep_size,
                        "distribution": (
                            "power_law_" + str(power_law_alpha) if distributed == "power_law" else distributed
                        ),
                        "latency": latency,
                    }
                ],
                framework="VLLM",
                version=vllm_version,
                device_name=torch.cuda.get_device_name(device),
                op_name="moe",
                kernel_source=source,
                perf_filename=perf_filename,
                power_stats=power_stats,
            )


if __name__ == "__main__":
    import traceback

    from collector.registry_types import PerfFile

    test_cases = get_moe_test_cases()
    print(f"Total test cases: {len(test_cases)}")

    # Standalone debug entrypoint: report each failing case and keep going,
    # like the collect.py worker path does, instead of aborting the sweep.
    for test_case in test_cases[:4]:
        print(f"Running test case: {test_case}")
        try:
            run_moe_torch(*test_case, perf_filename=PerfFile.MOE)
        except Exception as error:
            traceback.print_exc()
            print(f"Test case failed ({type(error).__name__}): {test_case}")
