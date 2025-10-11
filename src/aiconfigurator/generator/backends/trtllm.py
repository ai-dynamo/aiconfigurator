# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
TRT-LLM backend generator.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from .base import BaseGenerator, register
from ..inputs.schema import GeneratorContext, ModeConfig
from ..types import ArtifactBundle
from ..utils.node_allocation import allocate_disagg_nodes
from aiconfigurator.sdk.models import get_model_family

TEMPLATE_ROOT = Path(__file__).resolve().parents[1] / "templates" / "trtllm"

def _pick_engine_tpl(version: Optional[str]) -> str:
    """Choose engine args template by version; fallback to default."""
    cand = TEMPLATE_ROOT / f"extra_engine_args.{version}.yaml.j2" if version else None
    return (cand if cand and cand.exists() else (TEMPLATE_ROOT / "extra_engine_args.yaml.j2")).name

@register("trtllm")
class TRTLLMGenerator(BaseGenerator):
    name = "trtllm"

    def __init__(self) -> None:
        self.env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_ROOT)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _apply_kv_cache_mode(self, cfg: Dict[str, Any], kv_mode: Optional[str]) -> None:
        """Map kv_cache quant mode into engine dtype if needed."""
        if not kv_mode:
            return
        if kv_mode.lower() == "fp8":
            cfg["kv_cache_dtype"] = "fp8"

    def _apply_moe_attention_dp_mapping(self, cfg: Dict[str, Any], spec: ModeConfig, is_moe: bool) -> None:
        """
        - If is_moe:
            - mark is_moe
            - attach moe_tp/moe_ep when available
            - if dp > 1: enable_attention_dp=True and remap tp := dp, gpu := tp * pp
        - Else:
            - enable_attention_dp=False
        """
        if is_moe:
            cfg["is_moe"] = True
            if spec.moe_tp is not None:
                cfg["moe_tp"] = spec.moe_tp
            if spec.moe_ep is not None:
                cfg["moe_ep"] = spec.moe_ep
            if spec.dp > 1:
                cfg["enable_attention_dp"] = True
                cfg["tp"] = spec.dp
                cfg["gpu"] = cfg["tp"] * cfg["pp"]
            else:
                cfg["enable_attention_dp"] = False
        else:
            cfg["enable_attention_dp"] = False

    def _apply_deepseek_mtp(self, cfg: Dict[str, Any], ctx: GeneratorContext, role: str) -> None:
        """
        Keep original behavior: Only enable MTP for DEEPSEEK family when nextn>=1.
        """
        if ctx.runtime.nextn and ctx.runtime.nextn >= 1 and get_model_family(ctx.model_name) == "DEEPSEEK":
            cfg["is_speculative"] = True
            cfg["decoding_type"] = "MTP"
            cfg["num_nextn_predict_layers"] = int(ctx.runtime.nextn)

    def _get_yaml_tpl(self, tpl_name: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        import yaml
        tpl = self.env.get_template(tpl_name)
        return yaml.safe_load(tpl.render(**ctx))

    def _build_worker_config(self, spec: ModeConfig, ctx: GeneratorContext, role: str) -> Dict[str, Any]:
        """
        Build a per-role engine args dict based on the original implementation,
        with common parts factored into helpers and reading from GeneratorContext.
        `role` is one of: 'agg', 'prefill', 'decode'.
        """
        cfg: Dict[str, Any] = dict(
            tp=int(spec.tp),
            pp=int(spec.pp),
            dp=int(spec.dp),
            bs=int(spec.bs),
            **spec.extra,
        )
        cfg["gpu"] = cfg["tp"] * cfg["pp"] * cfg["dp"]

        # workers only for disagg roles
        if spec.workers is not None and role in ("prefill", "decode"):
            cfg["workers"] = int(spec.workers)

        # MoE / attention-dp mapping
        self._apply_moe_attention_dp_mapping(cfg, spec, is_moe=ctx.runtime.is_moe)

        # kv cache dtype
        self._apply_kv_cache_mode(cfg, ctx.runtime.kv_cache_mode)

        # max seq len heuristic
        cfg["max_seq_len"] = int(ctx.runtime.isl + ctx.runtime.osl + 1000)

        # speculative decoding for DEEPSEEK family
        self._apply_deepseek_mtp(cfg, ctx, role)

        return cfg

    def generate(self, ctx: GeneratorContext) -> ArtifactBundle:
        engine_tpl = _pick_engine_tpl(ctx.version)
        run_tpl = self.env.get_template("run.sh.j2")
        k8s_tpl = (TEMPLATE_ROOT / "k8s_deploy.yaml.j2").name
        global_args, worker_args = ctx.overrides.split_by_worker_type()

        out: Dict[str, Dict[str, Any]] = {}

        agg_spec = ctx.modes.get("agg")
        if agg_spec:
            agg_cfg = self._build_worker_config(agg_spec, ctx, role="agg")

            # perf considerations
            agg_cfg["cuda_graph_batch_sizes"] = [i for i in range(1, agg_cfg["bs"] + 1)]
            agg_cfg["disable_overlap_scheduler"] = False

            # max_num_tokens heuristic
            if ctx.runtime.nextn and ctx.runtime.nextn >= 1 and get_model_family(ctx.model_name) == "DEEPSEEK":
                agg_cfg["max_num_tokens"] = agg_cfg["bs"] * (1 + ctx.runtime.nextn) + ctx.runtime.isl + 1500
            else:
                agg_cfg["max_num_tokens"] = agg_cfg["bs"] + ctx.runtime.isl + 1500

            agg_yaml = self._get_yaml_tpl(engine_tpl, {**agg_cfg, "dynamo_config": global_args})
            agg_yaml_str = yaml.safe_dump(agg_yaml, sort_keys=False)

            script = run_tpl.render(
                mode="agg",
                router_mode=global_args.get("router_mode", ""),
                include_frontend=True,
                agg_engine_args="agg/agg_config.yaml",
                dynamo_config=global_args,
                model_name=ctx.model_name,
            )

            k8s_agg = {
                "model_name": ctx.model_name,
                "mode": "agg",
                "agg_workers": int(global_args.get("agg_workers", 1)),
                "agg_gpu": int(agg_cfg["gpu"]),
                "agg_engine_args": "/workspace/engine_configs/agg_config.yaml",
                "dynamo_config": global_args,
                "agg_engine_args_inline": agg_yaml_str,
            }

            k8s_yaml_agg = self._get_yaml_tpl(k8s_tpl, k8s_agg)

            out["agg"] = {
                "agg_config.yaml": agg_yaml,
                "node_0_run.sh": script,
                "k8s_deploy.yaml": k8s_yaml_agg
            }

        pre_spec = ctx.modes.get("disagg_prefill")
        dec_spec = ctx.modes.get("disagg_decode")
        if pre_spec and dec_spec:

            pre_cfg = self._build_worker_config(pre_spec, ctx, role="prefill")
            dec_cfg = self._build_worker_config(dec_spec, ctx, role="decode")

            # overlap scheduler
            pre_cfg["disable_overlap_scheduler"] = True
            dec_cfg["disable_overlap_scheduler"] = False

            # tokens bound
            pre_cfg["max_num_tokens"] = pre_cfg["bs"] * ctx.runtime.isl + 1500
            if ctx.runtime.nextn and ctx.runtime.nextn >= 1 and get_model_family(ctx.model_name) == "DEEPSEEK":
                dec_cfg["is_speculative"] = True
                dec_cfg["decoding_type"] = "MTP"
                dec_cfg["num_nextn_predict_layers"] = int(ctx.runtime.nextn)
                dec_cfg["max_num_tokens"] = dec_cfg["bs"] * (1 + ctx.runtime.nextn)
            else:
                dec_cfg["max_num_tokens"] = dec_cfg["bs"]

            # cuda graph sizes for decode
            dec_cfg["cuda_graph_batch_sizes"] = [i for i in range(1, dec_cfg["bs"] + 1)]

            pre_yaml = self._get_yaml_tpl(engine_tpl, {**pre_cfg, "dynamo_config": {**global_args, **worker_args["prefill"]}})
            dec_yaml = self._get_yaml_tpl(engine_tpl, {**dec_cfg, "dynamo_config": {**global_args, **worker_args["decode"]}})
            pre_yaml_str = yaml.safe_dump(pre_yaml, sort_keys=False)
            dec_yaml_str = yaml.safe_dump(dec_yaml, sort_keys=False)



            plan = allocate_disagg_nodes(pre_cfg.get("workers", 1), pre_cfg["gpu"], dec_cfg.get("workers", 1), dec_cfg["gpu"])

            node_files: Dict[str, str] = {}
            for idx, cnt in enumerate(plan):
                script = run_tpl.render(
                    mode="disagg",
                    router_mode=global_args.get("router_mode", ""),
                    include_frontend=(idx == 0),
                    prefill_gpu=pre_cfg["gpu"],
                    prefill_workers=cnt["p_worker"],
                    decode_gpu=dec_cfg["gpu"],
                    decode_workers=cnt["d_worker"],
                    decode_gpu_offset=cnt["p_worker"] * pre_cfg["gpu"],
                    prefill_engine_args="disagg/prefill_config.yaml",
                    decode_engine_args="disagg/decode_config.yaml",
                    dynamo_config=global_args,
                    model_name=ctx.model_name,
                )
                node_files[f"node_{idx}_run.sh"] = script

            k8s_disagg = {
                "model_name": ctx.model_name,
                "mode": "disagg",
                "prefill_workers": int(pre_cfg.get("workers", 1)),
                "decode_workers": int(dec_cfg.get("workers", 1)),
                "prefill_gpu": int(pre_cfg["gpu"]),
                "decode_gpu": int(dec_cfg["gpu"]),
                "prefill_engine_args": "/workspace/engine_configs/prefill_config.yaml",
                "decode_engine_args": "/workspace/engine_configs/decode_config.yaml",
                "dynamo_config": global_args,
                "prefill_engine_args_inline": pre_yaml_str,
                "decode_engine_args_inline": dec_yaml_str,
            }
            k8s_yaml_disagg = self._get_yaml_tpl(k8s_tpl, k8s_disagg)

            out["disagg"] = {
                "prefill_config.yaml": pre_yaml,
                "decode_config.yaml": dec_yaml,
                "k8s_deploy.yaml": k8s_yaml_disagg,
                **node_files,
            }

        return ArtifactBundle(by_mode=out)
