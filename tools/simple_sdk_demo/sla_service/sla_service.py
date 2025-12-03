# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import copy
import logging
import sys
from typing import Any

import orjson
import pandas as pd
import uvicorn
from fastapi import Body, FastAPI, Response

from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.common import SupportedModels
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk.inference_session import InferenceSession
from aiconfigurator.sdk.models import check_is_moe, get_model
from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator.sdk.utils import enumerate_parallel_config

logger = logging.getLogger(__name__)


class PrettyJSONResponse(Response):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_INDENT_2)


app = FastAPI(
    title="Dynamo AIConfigurator SLA API",
    description="Dynamo AIConfigurator SLA API",
    default_response_class=PrettyJSONResponse,
)


@app.get("/sla/supported_models")
def get_supported_models():
    return Response(
        content=orjson.dumps({"model list:": list(SupportedModels.keys())}),
        media_type="application/json",
    )


@app.post("/sla")
def post_sla(
    system: str = Body("h200_sxm", description="hardware name, h200_sxm, h100_sxm, b200_sxm, gb200_sxm, a100_sxm"),
    backend: str = Body("trtllm", description="backend name, trtllm, sglang, vllm"),
    version: str = Body("0.20.0", description="trtllm version, 0.20.0"),
    model_name: str = Body("QWEN3_32B", description="model name"),
    isl: int = Body(4000, description="input sequence length"),
    osl: int = Body(500, description="output sequence length"),
    ttft: int = Body(300, description="first token latency limit"),
    tpot: int = Body(10, description="inter token latency limit"),
    quant: str = Body("fp8", description="quantization mode: fp8, fp8_block, float16"),
    kvcache_quant: str = Body("fp8", description="kvcache quantization mode, fp8, int8, float16"),
):
    logging.basicConfig(level=logging.INFO)
    result_dict = {}
    try:
        model_config = ModelConfig(
            gemm_quant_mode=common.GEMMQuantMode[quant],
            kvcache_quant_mode=common.KVCacheQuantMode[kvcache_quant],
            fmha_quant_mode=common.FMHAQuantMode.float16,
        )
        runtime_config = RuntimeConfig(batch_size=1, isl=isl, osl=osl, ttft=ttft, tpot=tpot)

        database = get_database(system, backend, version)
        backend_instance = get_backend(backend)

        # dense model
        is_moe = check_is_moe(model_name)
        agg_parallel_config_list = enumerate_parallel_config(
            num_gpu_list=[1, 2, 4, 8],
            tp_list=[1, 2, 4, 8],
            pp_list=[1],
            moe_tp_list=[1, 2, 4, 8],
            moe_ep_list=[1, 2, 4, 8],
            dp_list=[1],
            is_moe=is_moe,
            backend=common.BackendName(backend),
            enable_wideep=False,
        )

        concurrency_list_default = [
            2,
            4,
            8,
            16,
            32,
            48,
            64,
            96,
            128,
            192,
            256,
            384,
            512,
            768,
            1024,
            2048,
            3072,
            4096,
        ]
        max_num_tokens = 8192  # default as NIM
        min_cc = max_num_tokens // isl + 1
        cc_list = [cc for cc in concurrency_list_default if cc >= min_cc]
        results_df = pd.DataFrame(columns=common.ColumnsAgg)
        for parallel_config in agg_parallel_config_list:
            tp_size, pp_size, dp_size, moe_tp_size, moe_ep_size = parallel_config
            overwritten_model_config = copy.deepcopy(model_config)
            overwritten_model_config.pp_size = pp_size
            overwritten_model_config.tp_size = tp_size
            overwritten_model_config.moe_tp_size = moe_tp_size
            overwritten_model_config.moe_ep_size = moe_ep_size
            overwritten_model_config.attention_dp_size = dp_size
            model = get_model(model_name=model_name, model_config=overwritten_model_config, backend_name=backend)
            sess = InferenceSession(model, database, backend_instance)

            for cc in cc_list:
                runtime_config.batch_size = cc
                summary = sess.run_agg(runtime_config=runtime_config, ctx_tokens=max_num_tokens)
                result_df = summary.get_summary_df()
                if summary.check_oom():
                    logger.info(f"OOM for cc: {cc}")
                    break  # larger cc will cause oom
                if result_df.loc[0, "tpot"] <= tpot and result_df.loc[0, "ttft"] <= ttft:
                    logger.info(f"Found valid config for cc: {cc}")
                    if len(results_df) == 0:
                        results_df = result_df
                    else:
                        results_df = pd.concat([results_df, result_df], axis=0, ignore_index=True)
                else:
                    logger.info(
                        f"Invalid config for cc: {cc} tpot: {result_df.loc[0, 'tpot']} ttft: {result_df.loc[0, 'ttft']}"
                    )
                    break

        results_df = results_df.sort_values(by="tokens/s/gpu", ascending=False).reset_index(drop=True)

        if len(results_df) != 0:
            result_dict = results_df.loc[0].to_dict()
    except Exception as e:
        print(e)
        result_dict = {"error": str(e)}

    return result_dict


def parse(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_name", type=str, default="127.0.0.1", help="server name")
    parser.add_argument("--server_port", type=int, default=7860, help="server port")
    args = parser.parse_args(args=args)
    return args


if __name__ == "__main__":
    args = parse(sys.argv[1:])
    uvicorn.run(app, host=args.server_name, port=args.server_port)
