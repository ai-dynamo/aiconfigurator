# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Dict, Any
import yaml
import logging
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from typing import Optional
from aiconfigurator.sdk import common
from aiconfigurator.sdk.models import get_model_family
from aiconfigurator.cli.helpers import (DynamoConfig, allocate_disagg_nodes)

logger = logging.getLogger(__name__)

TEMPLATE_ROOT = Path(__file__).resolve().parent.parent / "templates" / "trtllm" / "dynamo"
RUN_SH_TPL = TEMPLATE_ROOT / "run.sh.j2"
ENG_TPL_DEFAULT = TEMPLATE_ROOT / "extra_engine_args.yaml.j2"


def _pick_engine_tpl(version: Optional[str]) -> Path:
    """
    Choose engine args template based on --version
    Fallback to default template if version is not found
    """
    if version:
        cand = TEMPLATE_ROOT / f"extra_engine_args.{version}.yaml.j2"
        if cand.exists():
            logger.debug("Using version-specific engine template %s", cand.name)
            return cand
        logger.warning("Engine template %s not found, use default",
                       cand.name)
    return ENG_TPL_DEFAULT


def generate_backend_config(cfg_res,
                            cfg_cfg,
                            dyn_cfg: DynamoConfig,
                            version: str) -> Dict[str, Dict[str, Any]]:
    """
    Generate trt-llm backend config:
    Args:
        cfg_res: aiconfiguratorResult
        cfg_cfg: aiconfiguratorConfig
        dyn_cfg: DynamoConfig
        version: cli passed version string
    Returns:
        { mode : { filename : yaml/str } }
        Where script content is **string**, yaml content is **dict** for later dump
    """
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_ROOT)),
                      undefined=StrictUndefined,
                      trim_blocks=True, lstrip_blocks=True)

    engine_tpl = env.get_template(_pick_engine_tpl(version).name)
    sh_tpl = env.get_template(RUN_SH_TPL.name)

    parse_eng = lambda ctx: yaml.safe_load(engine_tpl.render(**ctx))
    global_args, worker_args = dyn_cfg.split_by_worker_type()

    model_name = cfg_cfg.model_name

    output: Dict[str, Dict[str, Any]] = {}


    def _build_worker_config(prefix, cfg_res_best):
        dynamo_cfg = dict(
            tp=int(cfg_res_best[f"{prefix}tp"]),
            pp=int(cfg_res_best[f"{prefix}pp"]),
            dp=int(cfg_res_best[f"{prefix}dp"]),
            bs=int(cfg_res_best[f"{prefix}bs"]),
        )

        dynamo_cfg["gpu"] = dynamo_cfg["tp"] * dynamo_cfg["pp"] * dynamo_cfg["dp"]

        if prefix != "":
            dynamo_cfg["workers"] = int(cfg_res_best[f"{prefix}workers"])

        if cfg_cfg.is_moe:
            dynamo_cfg["is_moe"] = True
            dynamo_cfg["moe_tp"] = cfg_res_best[f"{prefix}moe_tp"]
            dynamo_cfg["moe_ep"] = cfg_res_best[f"{prefix}moe_ep"]
            if cfg_res_best[f"{prefix}dp"] > 1:
                dynamo_cfg["enable_attention_dp"] = True
                dynamo_cfg["tp"] = cfg_res_best[f"{prefix}dp"]
                dynamo_cfg["gpu"] = dynamo_cfg["tp"] * dynamo_cfg["pp"]
        else:
            dynamo_cfg["enable_attention_dp"] = False
        
        if cfg_cfg.agg_worker_config.kvcache_quant_mode == common.KVCacheQuantMode.fp8:
            dynamo_cfg["kv_cache_dtype"] = "fp8"

        # Generate mtp config for both prefill and decode to prevent kv transceiver error
        if get_model_family(cfg_cfg.model_name) == "DEEPSEEK":
            if cfg_cfg.nextn >= 1:
                dynamo_cfg["is_speculative"] = True
                dynamo_cfg["decoding_type"] = "MTP"
                dynamo_cfg["num_nextn_predict_layers"] = cfg_cfg.nextn

        
        dynamo_cfg["max_seq_len"] = int(cfg_cfg.isl + cfg_cfg.osl + 1000)
        return dynamo_cfg

    # Agg
    if not cfg_res.agg_best_config.empty:
        agg_config = _build_worker_config("", cfg_res.agg_best_config.iloc[0])
        # we're assuming no chunked prefill. It actually doesn't have large impact on the performance.
        # +1000 is for holding the largest isl. You can set a safe value to hold your longest isl.
        
        if cfg_cfg.nextn >= 1:
            agg_config["max_num_tokens"] = agg_config["bs"] * (1+cfg_cfg.nextn) + cfg_cfg.isl + 1500
        else:
            agg_config["max_num_tokens"] = agg_config["bs"] + cfg_cfg.isl + 1500

        agg_yaml = parse_eng({**agg_config, "dynamo_config": global_args})

        script = sh_tpl.render(
            mode="agg",
            router_mode=global_args.get("router_mode", ""),
            include_frontend=True,
            agg_engine_args="agg/agg_config.yaml",
            dynamo_config=global_args,
            model_name=model_name,
        )

        output["agg"] = {
            "agg_config.yaml": agg_yaml,
            "node_0_run.sh": script,
        }

  
    # Disagg
    if not cfg_res.disagg_best_config.empty:
        disagg_bc = cfg_res.disagg_best_config.iloc[0]
        disagg_prefill_config = _build_worker_config("(p)", disagg_bc)
        disagg_decode_config = _build_worker_config("(d)", disagg_bc)

        # setting max_num_tokens
        disagg_prefill_config["max_num_tokens"] = disagg_prefill_config["bs"] * cfg_cfg.isl + 1500
        disagg_decode_config["max_num_tokens"] = disagg_decode_config["bs"]
        if cfg_cfg.nextn >= 1:
            disagg_decode_config["max_num_tokens"] = agg_config["max_num_tokens"] * (1+cfg_cfg.nextn)
        

        pre_yaml = parse_eng({**disagg_prefill_config,
                              "dynamo_config": {**global_args, **worker_args["prefill"]}})
        dec_yaml = parse_eng({**disagg_decode_config,
                              "dynamo_config": {**global_args, **worker_args["decode"]}})

        plan = allocate_disagg_nodes(disagg_prefill_config["workers"],
                                     disagg_prefill_config["gpu"],
                                     disagg_decode_config["workers"],
                                     disagg_decode_config["gpu"])

        node_files: Dict[str, str] = {}
        for idx, cnt in enumerate(plan):
            pre_workers = cnt["p_worker"]
            dec_workers = cnt["d_worker"]
            script = sh_tpl.render(
                mode="disagg",
                router_mode=global_args.get("router_mode", ""),
                include_frontend=(idx == 0),
                prefill_gpu=disagg_prefill_config["gpu"],
                prefill_workers=pre_workers,
                decode_gpu=disagg_decode_config["gpu"],
                decode_workers=dec_workers,
                decode_gpu_offset=pre_workers * disagg_prefill_config["gpu"],
                prefill_engine_args="disagg/prefill_config.yaml",
                decode_engine_args="disagg/decode_config.yaml",
                dynamo_config=global_args,
                model_name=model_name,
            )
            node_files[f"node_{idx}_run.sh"] = script

        output["disagg"] = {
            "prefill_config.yaml": pre_yaml,
            "decode_config.yaml": dec_yaml,
            **node_files,
        }

    return output
