# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Input parser that transforms AIConfiguratorConfig/Result into GeneratorContext.
"""
from typing import Optional
from .schema import (
    DynamoConfig, GeneratorContext, RuntimeView,
    BestAggConfig, BestDisaggConfig, RoleParallelSpec
)

class InputParser:
    """Factory of GeneratorContext from runtime objects."""

    @staticmethod
    def _enum_name_or_str(x) -> Optional[str]:
        if x is None:
            return None
        return getattr(x, "name", str(x))

    @staticmethod
    def from_runtime(cfg, res, overrides: DynamoConfig,
                     version: Optional[str], backend: Optional[str]) -> GeneratorContext:
        """
        Build GeneratorContext from AIConfiguratorConfig (cfg) and AIConfiguratorResult (res).
        """
        agg_w = getattr(cfg, "agg_worker_config", None)
        pre_w = getattr(cfg, "disagg_prefill_worker_config", None)
        dec_w = getattr(cfg, "disagg_decode_worker_config", None)

        resolved_backend = backend or (
            getattr(agg_w, "backend_name", None)
            or getattr(pre_w, "backend_name", None)
            or getattr(dec_w, "backend_name", None)
        )
        resolved_version = version or (
            getattr(agg_w, "version", None)
            or getattr(pre_w, "version", None)
            or getattr(dec_w, "version", None)
        )

        kv_mode = None
        for w in (agg_w, pre_w, dec_w):
            if w is not None and getattr(w, "kvcache_quant_mode", None) is not None:
                kv_mode = InputParser._enum_name_or_str(w.kvcache_quant_mode)
                break

        runtime = RuntimeView(
            isl=cfg.isl,
            osl=cfg.osl,
            is_moe=bool(cfg.is_moe),
            kv_cache_mode=kv_mode,
            nextn=getattr(cfg, "nextn", 0),
            nextn_accept_rates=getattr(cfg, "nextn_accept_rates", None),
        )

        # Extract best agg
        best_agg = None
        if getattr(res, "agg_best_config", None) is not None and not res.agg_best_config.empty:
            row = res.agg_best_config.iloc[0]
            best_agg = BestAggConfig(
                tp=int(row["tp"]), pp=int(row["pp"]), dp=int(row["dp"]), bs=int(row["bs"]),
                moe_tp=int(row["moe_tp"]) if "moe_tp" in row else None,
                moe_ep=int(row["moe_ep"]) if "moe_ep" in row else None,
            )

        # Extract best disagg
        best_disagg = None
        if getattr(res, "disagg_best_config", None) is not None and not res.disagg_best_config.empty:
            row = res.disagg_best_config.iloc[0]
            pre = RoleParallelSpec(
                tp=int(row["(p)tp"]), pp=int(row["(p)pp"]), dp=int(row["(p)dp"]), bs=int(row["(p)bs"]),
                moe_tp=int(row["(p)moe_tp"]) if "(p)moe_tp" in row else None,
                moe_ep=int(row["(p)moe_ep"]) if "(p)moe_ep" in row else None,
                workers=int(row["(p)workers"]),
            )
            dec = RoleParallelSpec(
                tp=int(row["(d)tp"]), pp=int(row["(d)pp"]), dp=int(row["(d)dp"]), bs=int(row["(d)bs"]),
                moe_tp=int(row["(d)moe_tp"]) if "(d)moe_tp" in row else None,
                moe_ep=int(row["(d)moe_ep"]) if "(d)moe_ep" in row else None,
                workers=int(row["(d)workers"]),
            )
            best_disagg = BestDisaggConfig(prefill=pre, decode=dec)

        return GeneratorContext(
            model_name=cfg.model_name,
            backend=resolved_backend,
            version=resolved_version,
            runtime=runtime,
            overrides=overrides,
            best_agg=best_agg,
            best_disagg=best_disagg,
        )
