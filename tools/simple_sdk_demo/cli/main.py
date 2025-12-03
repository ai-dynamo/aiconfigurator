# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.inference_session import InferenceSession
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import get_database


def parse(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="batch_size")
    parser.add_argument("--isl", type=int, help="input sequence length, max 256k")
    parser.add_argument("--osl", type=int, help="output sequence length")
    parser.add_argument("--stride", type=int, default=32, help="stride for fast generation, 32 by default")
    parser.add_argument(
        "--mode",
        type=str,
        default="static",
        choices=["static", "static_ctx", "static_gen", "agg", "disagg"],
        help="inference mode",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="trtllm",
        choices=list(common.BackendName.__members__.keys()),
        help="backend",
    )
    parser.add_argument("--version", type=str, default="0.20.0", help="backend version")
    parser.add_argument(
        "--system",
        type=str,
        default="h200_sxm",
        choices=["h100_sxm", "h200_sxm", "b200_sxm", "gb200_sxm", "a100_sxm"],
        help="GPU+system Type",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(common.SupportedModels.keys()),
        help="model name, with size",
    )
    parser.add_argument(
        "--overwrite_num_layers",
        type=int,
        default=0,
        help="if larger than 0, overwrite model layers to the value assigned",
    )
    parser.add_argument("--attention_dp_size", type=int, default=1, help="attention data parallel size  moe models")
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        choices=[1, 2, 4, 8, 16, 32],
        help="tensor parallel size, support 1,2,4,8,16,32",
    )
    parser.add_argument(
        "--pp_size",
        type=int,
        default=1,
        choices=[1, 2, 4, 8],
        help="pipeline parallel size, support 1,2,4,8",
    )
    parser.add_argument(
        "--gemm_quant_mode",
        type=str,
        default="float16",
        choices=list(common.GEMMQuantMode.__members__.keys()),
        help="gemm quantization type",
    )
    parser.add_argument(
        "--moe_quant_mode",
        type=str,
        default="float16",
        choices=list(common.MoEQuantMode.__members__.keys()),
        help="moe quantization type",
    )
    parser.add_argument(
        "--kvcache_quant_mode",
        type=str,
        default="float16",
        choices=list(common.KVCacheQuantMode.__members__.keys()),
        help="kvcache quantization type",
    )
    parser.add_argument(
        "--fmha_quant_mode",
        type=str,
        default="float16",
        choices=list(common.FMHAQuantMode.__members__.keys()),
        help="kvcache quantization type",
    )
    parser.add_argument("--moe_tp_size", type=int, default=1, help="tp size for moe")
    parser.add_argument("--moe_ep_size", type=int, default=1, help="ep size for moe")
    parser.add_argument(
        "--workload_distribution",
        type=str,
        default="uniform",
        choices=["uniform", "balanced", "mostUnbalanced", "power_law"],
        help="workload for moe models",
    )
    parser.add_argument("--ttft", type=float, default=300, help="SLA requirement: ttft limit (ms)")
    parser.add_argument("--tpot", type=float, default=50, help="SLA requirement: tpot limit (ms)")
    parser.add_argument(
        "--ctx_stride",
        type=int,
        default=64,
        help="Agg: ctx_stride, mimicking num_tokens_per_block param in TRT-LLM, defaulted as 64",
    )
    parser.add_argument(
        "--enable_wideep",
        action="store_true",
        help="Enable WideEP DeepSeek modeling (only effective when --backend sglang and model is DeepSeek).",
    )
    parser.add_argument(
        "--moe_backend",
        type=str,
        default="deepep_moe",
        choices=["deepep_moe", "none"],
        help="moe backend",
    )
    args = parser.parse_args(args=args)

    return args


def main(args):
    database = get_database(system=args.system, backend=args.backend, version=args.version)
    model_config = config.ModelConfig(
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        gemm_quant_mode=common.GEMMQuantMode[args.gemm_quant_mode],
        moe_quant_mode=common.MoEQuantMode[args.moe_quant_mode],
        kvcache_quant_mode=common.KVCacheQuantMode[args.kvcache_quant_mode],
        fmha_quant_mode=common.FMHAQuantMode[args.fmha_quant_mode],
        moe_tp_size=args.moe_tp_size,
        moe_ep_size=args.moe_ep_size,
        workload_distribution=args.workload_distribution,
        attention_dp_size=args.attention_dp_size,
        overwrite_num_layers=args.overwrite_num_layers,
        enable_wideep=args.enable_wideep,
        moe_backend=args.moe_backend,
    )
    runtime_config = config.RuntimeConfig(
        batch_size=args.batch_size,
        beam_width=1,
        isl=args.isl,
        osl=args.osl,
        ttft=args.ttft,
        tpot=args.tpot,
    )
    model = get_model(args.model, model_config, args.backend)
    backend = get_backend(args.backend)
    session = InferenceSession(model, database, backend)

    if args.mode == "agg":
        summary = session.find_best_agg_result_under_constraints(
            runtime_config=runtime_config, top_k=0, max_batch_size=512, ctx_stride=args.ctx_stride
        )
        print(summary.get_summary_df())
        summary.get_summary_df().to_csv("result.csv")
    elif args.mode == "disagg":
        # TODO, leave this for future as it requires too many params as input. It should call into
        # DisaggInferenceSession.run_disagg()
        summary = None
    else:
        summary = session.run_static(mode=args.mode, runtime_config=runtime_config, stride=32)
        print(summary.get_summary_df())
        for info in summary.get_static_info():
            print(info)


if __name__ == "__main__":
    args = parse(sys.argv[1:])
    main(args)
