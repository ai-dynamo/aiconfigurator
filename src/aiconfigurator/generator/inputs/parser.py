# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Input parser that transforms AIConfiguratorConfig/Result into GeneratorContext.
"""
from typing import Optional

from .schema import (
    DynamoConfig,
    GeneratorContext,
    RuntimeView,
    ModeConfig,
)

class InputParser:
    """Factory of GeneratorContext from runtime objects."""

    @staticmethod
    def _enum_name_or_str(x) -> Optional[str]:
        if x is None:
            return None
        return getattr(x, "name", str(x))

    @staticmethod
    def _get_kv_mode(cfg) -> Optional[str]:
        for key in ("agg_worker_config", "disagg_prefill_worker_config", "disagg_decode_worker_config"):
            worker_config = cfg.get(key, None)
            if worker_config is not None and worker_config.get("kvcache_quant_mode", None) is not None:
                return InputParser._enum_name_or_str(worker_config.get("kvcache_quant_mode"))
        return None

    @staticmethod
    def _make_runtime(cfg) -> RuntimeView:
        return RuntimeView(
            isl=cfg["isl"],
            osl=cfg["osl"],
            is_moe=bool(cfg["is_moe"]),
            kv_cache_mode=InputParser._get_kv_mode(cfg),
            nextn=cfg.get("nextn", 0),
            nextn_accept_rates=cfg.get("nextn_accept_rates", None),
        )

    @staticmethod
    def from_runtime(
                    cfg: dict,
                    overrides: DynamoConfig,
                    backend: str,
                    version: str) -> GeneratorContext:

        if cfg["serving_mode"] == "agg":
            modes = {
                "agg": ModeConfig(
                    workers=cfg["agg_worker_config"].get("workers", 1),
                    bs=cfg["agg_worker_config"]["bs"],
                    tp=cfg["agg_worker_config"]["tp"],
                    pp=cfg["agg_worker_config"]["pp"],
                    dp=cfg["agg_worker_config"].get("dp", 1),
                    moe_tp=cfg["agg_worker_config"].get("moe_tp", 1),
                    moe_ep=cfg["agg_worker_config"].get("moe_ep", 1),
                    memory=cfg["agg_worker_config"].get("memory", None),
                ),
            }
        else:
            modes = {
                "disagg_prefill": ModeConfig(
                    workers=cfg["disagg_prefill_worker_config"]["workers"],
                    bs=cfg["disagg_prefill_worker_config"]["bs"],
                    tp=cfg["disagg_prefill_worker_config"]["tp"],
                    pp=cfg["disagg_prefill_worker_config"]["pp"],
                    dp=cfg["disagg_prefill_worker_config"].get("dp", 1),
                    moe_tp=cfg["disagg_prefill_worker_config"].get("moe_tp", 1),
                    moe_ep=cfg["disagg_prefill_worker_config"].get("moe_ep", 1),
                    memory=cfg["disagg_prefill_worker_config"].get("memory", None),
                ),
                "disagg_decode": ModeConfig(
                    workers=cfg["disagg_decode_worker_config"]["workers"],
                    bs=cfg["disagg_decode_worker_config"]["bs"],                    
                    tp=cfg["disagg_decode_worker_config"]["tp"],
                    pp=cfg["disagg_decode_worker_config"]["pp"],
                    dp=cfg["disagg_decode_worker_config"].get("dp", 1),
                    moe_tp=cfg["disagg_decode_worker_config"].get("moe_tp", 1),
                    moe_ep=cfg["disagg_decode_worker_config"].get("moe_ep", 1),
                    memory=cfg["disagg_decode_worker_config"].get("memory", None),
                ),
            }
        
        total_gpus = cfg["total_gpus"]
        if cfg["serving_mode"] == "disagg":
            num_gpus_per_replica = (modes["disagg_prefill"].workers * 
                                    modes["disagg_prefill"].tp * 
                                    modes["disagg_prefill"].dp * 
                                    modes["disagg_prefill"].pp + 
                                    modes["disagg_decode"].workers * 
                                    modes["disagg_decode"].tp * 
                                    modes["disagg_decode"].dp * 
                                    modes["disagg_decode"].pp)
        else:
            num_gpus_per_replica = modes["agg"].workers * modes["agg"].tp * modes["agg"].dp * modes["agg"].pp
        num_replicas = total_gpus // num_gpus_per_replica
        modes["total_gpus"] = total_gpus
        modes["num_gpus_per_replica"] = num_gpus_per_replica
        modes["num_replicas"] = num_replicas
        
        exp_config = cfg.get("exp_config", None)
        if exp_config is not None:
            modes["exp_config"] = {
                "ttft": exp_config.get("ttft", None),
                "tps_per_user": exp_config.get("tps_per_user", None),
                "tps_per_gpu": exp_config.get("tps_per_gpu", None),
                "concurrency_per_replica": exp_config.get("concurrency_per_replica", None),
                "num_requests_multiplier": exp_config.get("num_requests_multiplier", 10),
            }

        return GeneratorContext(
            model_name=cfg["model_name"],
            backend=backend,
            version=version,
            runtime=InputParser._make_runtime(cfg),
            overrides=overrides,
            modes=modes,
        )
