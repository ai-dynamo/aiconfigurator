# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bridge SGLang 0.5.14 bypassed TopK to its routed FlashInfer kernel path.

The patch is intentionally fail-closed: it only applies to the exact upstream
``topk.py`` shipped by the pinned SGLang tag.  While the recorder is active,
``select_experts`` produces and records explicit expert IDs and those same IDs
are consumed by the routed FlashInfer MoE implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

SGLANG_VERSION = "0.5.14"
EXPECTED_SOURCE_SHA256 = "ca35e317522d85b391005f270df34ebd692120927e8c66d9ca22d4778dfa140e"
EXPECTED_MXFP4_SOURCE_SHA256 = "b514d889f7ef55a8dca5f941702ec2871c8980000b6da355e7f52eb414ca9e5f"

_ORIGINAL = """        elif output_format == TopKOutputFormat.BYPASSED:
            return BypassedTopKOutput(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_config=self.topk_config,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
            )
"""

_PATCHED = """        elif output_format == TopKOutputFormat.BYPASSED:
            bypassed_output = BypassedTopKOutput(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_config=self.topk_config,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
            )
            # Collector bridge: while recording, materialize routing once so
            # the recorder observes the exact IDs consumed by the routed
            # FlashInfer kernel path. Outside a recording window the original
            # fused internal-routing path remains unchanged.
            if get_global_expert_distribution_recorder().recording:
                return bypassed_output.to_standard(layer_id=self.layer_id)
            return bypassed_output
"""

_MXFP4_ORIGINAL = """            assert x_quant.shape[-1] == self.hidden_size
            assert TopKOutputChecker.format_is_bypassed(topk_output)

            top_k = topk_output.topk_config.top_k
            router_logits = topk_output.router_logits

            with use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            ):
                num_tokens = x_quant.shape[0]
                hidden_size = origin_hidden_states_dim
                symm_output = torch.empty(
                    num_tokens, hidden_size, dtype=torch.bfloat16, device=x_quant.device
                )
            trtllm_gen_output = trtllm_fp4_block_scale_moe(
                router_logits.to(torch.bfloat16),
                None,  # routing_bias
                x_quant,
                x_scale,
                layer.w13_weight,  # uint8 (e2m1 x 2)
                layer.w13_weight_scale,  # uint8 (e4m3 x 2)
                layer.w13_weight_bias,  # fp32 per expert per channel
                layer.gemm1_alpha,  # fp32 per expert
                layer.gemm1_beta,  # fp32 per expert
                layer.gemm1_clamp_limit,  # fp32 per expert
                layer.w2_weight,  # uint8 (e2m1 x 2)
                layer.w2_weight_scale,  # ue8m0
                layer.w2_weight_bias,  # fp32 per expert per channel
                None,  # output1_scale_scalar
                None,  # output1_scale_gate_scalar
                None,  # output2_scale_scalar
                layer.num_experts,
                top_k,
                None,  # n_group      # TODO: support n_group
                None,  # topk_group   # TODO: support topk_group
                self.intermediate_size_per_partition,  # padded to multiple of 256
                layer.moe_ep_rank * layer.num_local_experts,  # local_expert_offset
                layer.num_local_experts,  # local num experts
                None,  # routed_scaling_factor
                1,  # routing_method_type, renormalize
                True,  # do finalize
                tune_max_num_tokens=next_power_of_2(x_quant.shape[0]),
                output=symm_output,
            )[0]
"""

_MXFP4_PATCHED = """            assert x_quant.shape[-1] == self.hidden_size

            with use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            ):
                num_tokens = x_quant.shape[0]
                hidden_size = origin_hidden_states_dim
                symm_output = torch.empty(
                    num_tokens, hidden_size, dtype=torch.bfloat16, device=x_quant.device
                )

            if TopKOutputChecker.format_is_standard(topk_output):
                from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
                from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
                    PackTopkIds,
                )

                packed_topk = PackTopkIds.execute(
                    topk_output.topk_ids, topk_output.topk_weights
                )
                trtllm_gen_output = trtllm_fp4_block_scale_routed_moe(
                    topk_ids=packed_topk,
                    routing_bias=None,
                    hidden_states=x_quant,
                    hidden_states_scale=x_scale,
                    gemm1_weights=layer.w13_weight,
                    gemm1_weights_scale=layer.w13_weight_scale,
                    gemm1_bias=layer.w13_weight_bias,
                    gemm1_alpha=layer.gemm1_alpha,
                    gemm1_beta=layer.gemm1_beta,
                    gemm1_clamp_limit=layer.gemm1_clamp_limit,
                    gemm2_weights=layer.w2_weight,
                    gemm2_weights_scale=layer.w2_weight_scale,
                    gemm2_bias=layer.w2_weight_bias,
                    output1_scale_scalar=None,
                    output1_scale_gate_scalar=None,
                    output2_scale_scalar=None,
                    num_experts=layer.num_experts,
                    top_k=packed_topk.shape[1],
                    n_group=1,
                    topk_group=1,
                    intermediate_size=self.intermediate_size_per_partition,
                    local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
                    local_num_experts=layer.num_local_experts,
                    routed_scaling_factor=1.0,
                    routing_method_type=1,
                    do_finalize=True,
                    tune_max_num_tokens=next_power_of_2(x_quant.shape[0]),
                    output=symm_output,
                )[0]
            elif TopKOutputChecker.format_is_bypassed(topk_output):
                top_k = topk_output.topk_config.top_k
                router_logits = topk_output.router_logits
                trtllm_gen_output = trtllm_fp4_block_scale_moe(
                    router_logits.to(torch.bfloat16),
                    None,  # routing_bias
                    x_quant,
                    x_scale,
                    layer.w13_weight,  # uint8 (e2m1 x 2)
                    layer.w13_weight_scale,  # uint8 (e4m3 x 2)
                    layer.w13_weight_bias,  # fp32 per expert per channel
                    layer.gemm1_alpha,  # fp32 per expert
                    layer.gemm1_beta,  # fp32 per expert
                    layer.gemm1_clamp_limit,  # fp32 per expert
                    layer.w2_weight,  # uint8 (e2m1 x 2)
                    layer.w2_weight_scale,  # ue8m0
                    layer.w2_weight_bias,  # fp32 per expert per channel
                    None,  # output1_scale_scalar
                    None,  # output1_scale_gate_scalar
                    None,  # output2_scale_scalar
                    layer.num_experts,
                    top_k,
                    None,  # n_group
                    None,  # topk_group
                    self.intermediate_size_per_partition,
                    layer.moe_ep_rank * layer.num_local_experts,
                    layer.num_local_experts,
                    None,  # routed_scaling_factor
                    1,  # routing_method_type, renormalize
                    True,  # do finalize
                    tune_max_num_tokens=next_power_of_2(x_quant.shape[0]),
                    output=symm_output,
                )[0]
            else:
                raise ValueError(f"Unsupported topk output format: {topk_output.format}")
"""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def apply_bridge(report_path: Path) -> dict:
    installed_version = importlib.metadata.version("sglang")
    if installed_version != SGLANG_VERSION:
        raise RuntimeError(f"SGLang {installed_version!r} != pinned {SGLANG_VERSION!r}")

    topk_spec = importlib.util.find_spec("sglang.srt.layers.moe.topk")
    mxfp4_spec = importlib.util.find_spec("sglang.srt.layers.quantization.mxfp4")
    if topk_spec is None or topk_spec.origin is None:
        raise RuntimeError("could not locate sglang.srt.layers.moe.topk")
    if mxfp4_spec is None or mxfp4_spec.origin is None:
        raise RuntimeError("could not locate sglang.srt.layers.quantization.mxfp4")
    topk_path = Path(topk_spec.origin)
    mxfp4_path = Path(mxfp4_spec.origin)
    topk_source = topk_path.read_bytes()
    mxfp4_source = mxfp4_path.read_bytes()
    topk_original_sha256 = _sha256(topk_source)
    mxfp4_original_sha256 = _sha256(mxfp4_source)
    if topk_original_sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"refusing to patch unexpected SGLang topk.py {topk_original_sha256}; expected {EXPECTED_SOURCE_SHA256}"
        )
    if mxfp4_original_sha256 != EXPECTED_MXFP4_SOURCE_SHA256:
        raise RuntimeError(
            f"refusing to patch unexpected SGLang mxfp4.py {mxfp4_original_sha256}; "
            f"expected {EXPECTED_MXFP4_SOURCE_SHA256}"
        )

    topk_decoded = topk_source.decode("utf-8")
    mxfp4_decoded = mxfp4_source.decode("utf-8")
    if topk_decoded.count(_ORIGINAL) != 1:
        raise RuntimeError("expected bypassed TopK block was not uniquely present")
    if mxfp4_decoded.count(_MXFP4_ORIGINAL) != 1:
        raise RuntimeError("expected MXFP4 internal-routing block was not uniquely present")
    topk_patched = topk_decoded.replace(_ORIGINAL, _PATCHED, 1).encode("utf-8")
    mxfp4_patched = mxfp4_decoded.replace(_MXFP4_ORIGINAL, _MXFP4_PATCHED, 1).encode("utf-8")
    for source_path, patched in ((topk_path, topk_patched), (mxfp4_path, mxfp4_patched)):
        temporary = source_path.with_suffix(".py.collector-tmp")
        temporary.write_bytes(patched)
        temporary.replace(source_path)

    report = {
        "status": "APPLIED",
        "framework": "sglang",
        "framework_version": installed_version,
        "source_files": {
            "topk.py": {
                "path": str(topk_path),
                "original_sha256": topk_original_sha256,
                "patched_sha256": _sha256(topk_patched),
            },
            "mxfp4.py": {
                "path": str(mxfp4_path),
                "original_sha256": mxfp4_original_sha256,
                "patched_sha256": _sha256(mxfp4_patched),
            },
        },
        "behavior": "materialize_and_consume_explicit_topk_only_while_recorder_is_active",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(apply_bridge(args.report), sort_keys=True))


if __name__ == "__main__":
    main()
