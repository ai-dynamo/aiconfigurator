# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union
from collections.abc import Mapping

import yaml
import pandas as pd
from munch import DefaultMunch, Munch
import json
from typing import Dict, Any

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.models import check_is_moe
from aiconfigurator.sdk.perf_database import (
    PerfDatabase,
    get_database,
    get_latest_database_version,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfigLayer:
    name: str
    data: Union[dict, Callable[["TaskContext"], dict]]
    condition: Optional[Callable[["TaskContext"], bool]] = None

    def applies_to(self, ctx: "TaskContext") -> bool:
        if self.condition is None:
            return True
        try:
            return self.condition(ctx)
        except Exception:  # pragma: no cover
            logger.debug("Layer %s condition evaluation failed", self.name)
            return False

    def resolve(self, ctx: "TaskContext") -> dict:
        payload = self.data(ctx) if callable(self.data) else self.data
        return copy.deepcopy(payload)


@dataclass
class TaskContext:
    serving_mode: Literal["agg", "disagg"]
    model_name: str
    system_name: str
    decode_system_name: Optional[str]
    backend_name: str
    backend_version: Optional[str]
    use_specific_quant_mode: Optional[str]
    enable_wide_ep: bool
    isl: int
    osl: int
    ttft: float
    tpot: float
    total_gpus: Optional[int]
    profiles: list[str] = field(default_factory=list)
    yaml_patch: dict = field(default_factory=dict)
    yaml_mode: Literal["patch", "replace"] = "patch"

    @property
    def is_moe(self) -> bool:
        return check_is_moe(self.model_name)

    def resolved_backend_version_for(self, system_name: str) -> str:
        if self.backend_version is not None:
            return self.backend_version
        return get_latest_database_version(system=system_name, backend=self.backend_name)


def _deep_merge(target: dict, source: Mapping, *, allow_new: bool = True) -> dict:
    for key, value in source.items():
        if key not in target:
            if not allow_new:
                continue
            target[key] = copy.deepcopy(value)
            continue

        if isinstance(target[key], dict) and isinstance(value, Mapping):
            _deep_merge(target[key], value, allow_new=allow_new)
        else:
            target[key] = copy.deepcopy(value)
    return target


def _ensure_munch(obj: Union[dict, DefaultMunch, Munch]) -> DefaultMunch:
    if isinstance(obj, (DefaultMunch, Munch)):
        return DefaultMunch.fromDict(obj.toDict(), DefaultMunch)
    return DefaultMunch.fromDict(obj, DefaultMunch)


class TaskConfigFactory:
    PROFILE_REGISTRY: dict[str, list[ConfigLayer]] = {}

    @classmethod
    def register_profile(cls, name: str, layers: list[ConfigLayer]) -> None:
        cls.PROFILE_REGISTRY[name] = layers

    @classmethod
    def create(cls, ctx: TaskContext) -> tuple[DefaultMunch, list[str]]:
        config_dict: dict[str, Any] = {}
        applied_layers: list[str] = []

        for layer in cls._base_layers():
            if layer.applies_to(ctx):
                _deep_merge(config_dict, layer.resolve(ctx))
                applied_layers.append(layer.name)

        for layer in cls._mode_layers(ctx):
            if layer.applies_to(ctx):
                _deep_merge(config_dict, layer.resolve(ctx))
                applied_layers.append(layer.name)

        for profile in ctx.profiles:
            layers = cls.PROFILE_REGISTRY.get(profile)
            if not layers:
                logger.warning("Profile '%s' not found, skipping", profile)
                continue
            for layer in layers:
                if layer.applies_to(ctx):
                    _deep_merge(config_dict, layer.resolve(ctx))
                    applied_layers.append(f"profile:{profile}:{layer.name}")

        if ctx.yaml_patch:
            if ctx.yaml_mode == "replace":
                config_dict = copy.deepcopy(ctx.yaml_patch)
                applied_layers.append("yaml_replace")
            else:
                _deep_merge(config_dict, ctx.yaml_patch, allow_new=True)
                applied_layers.append("yaml_patch")

        config = DefaultMunch.fromDict(config_dict, DefaultMunch)

        if config.model_name != ctx.model_name:
            raise ValueError(
                f"Model name mismatch: base {ctx.model_name} vs. merged {config.model_name}"
            )

        if ctx.serving_mode == "agg":
            cls._finalize_agg(config, ctx)
        elif ctx.serving_mode == "disagg":
            cls._finalize_disagg(config, ctx)
        else:
            raise ValueError(f"Invalid serving mode: {ctx.serving_mode}")

        config.applied_layers = applied_layers
        return config, applied_layers

    @classmethod
    def _base_layers(cls) -> list[ConfigLayer]:
        return [ConfigLayer("base-common", cls._base_common_layer)]

    @classmethod
    def _mode_layers(cls, ctx: TaskContext) -> list[ConfigLayer]:
        if ctx.serving_mode == "agg":
            return [ConfigLayer("agg-defaults", cls._agg_defaults_layer)]
        if ctx.serving_mode == "disagg":
            return [ConfigLayer("disagg-defaults", cls._disagg_defaults_layer)]
        return []

    @staticmethod
    def _base_common_layer(ctx: TaskContext) -> dict:
        nextn = 1 if ctx.model_name in ["DEEPSEEK_V3", "KIMI_K2"] else 0
        return {
            "serving_mode": ctx.serving_mode,
            "model_name": ctx.model_name,
            "nextn": nextn,
            "nextn_accept_rates": [0.85, 0, 0, 0, 0],
            "runtime_config": {
                "isl": ctx.isl,
                "osl": ctx.osl,
                "ttft": ctx.ttft,
                "tpot": ctx.tpot,
            },
        }

    @staticmethod
    def _agg_defaults_layer(ctx: TaskContext) -> dict:
        should_enable_pp = False
        worker_config = {
            "system_name": ctx.system_name,
            "backend_name": ctx.backend_name,
            "backend_version": ctx.resolved_backend_version_for(ctx.system_name),
            "num_gpu_per_worker": [1, 2, 4, 8],
            "tp_list": [1, 2, 4, 8],
            "pp_list": [1, 2, 4, 8] if should_enable_pp else [1],
            "dp_list": [1, 2, 4, 8] if ctx.is_moe else [1],
            "moe_tp_list": [1],
            "moe_ep_list": [1, 2, 4, 8] if ctx.is_moe else [1],
        }

        if not ctx.is_moe:
            if ctx.system_name == "gb200_sxm":
                worker_config["num_gpu_per_worker"] = [1, 2, 4, 8, 16]
                worker_config["tp_list"] = [1, 2, 4, 8, 16]
                worker_config["pp_list"] = [1]
        else:
            if ctx.enable_wide_ep:
                worker_config["num_gpu_per_worker"] = [1, 2, 4, 8, 16, 32]
                worker_config["tp_list"] = [1, 2, 4, 8]
                worker_config["pp_list"] = [1, 2, 4, 8, 16, 32] if should_enable_pp else [1]
                worker_config["dp_list"] = [1, 2, 4, 8, 16, 32]
                worker_config["moe_ep_list"] = [1, 2, 4, 8, 16, 32]

        return {
            "is_moe": ctx.is_moe,
            "worker_config": worker_config,
        }

    @staticmethod
    def _disagg_defaults_layer(ctx: TaskContext) -> dict:
        should_enable_pp = False
        decode_system = ctx.decode_system_name or ctx.system_name

        prefill_worker_config = {
            "system_name": ctx.system_name,
            "backend_name": ctx.backend_name,
            "backend_version": ctx.resolved_backend_version_for(ctx.system_name),
            "num_gpu_per_worker": [1, 2, 4, 8],
            "tp_list": [1, 2, 4, 8],
            "pp_list": [1, 2, 4, 8] if should_enable_pp else [1],
            "dp_list": [1],
            "moe_tp_list": [1],
            "moe_ep_list": [1, 2, 4, 8] if ctx.is_moe else [1],
        }

        decode_worker_config = {
            "system_name": decode_system,
            "backend_name": ctx.backend_name,
            "backend_version": ctx.resolved_backend_version_for(decode_system),
            "num_gpu_per_worker": [1, 2, 4, 8],
            "tp_list": [1, 2, 4, 8],
            "pp_list": [1, 2, 4, 8] if should_enable_pp else [1],
            "dp_list": [1, 2, 4, 8] if ctx.is_moe else [1],
            "moe_tp_list": [1],
            "moe_ep_list": [1, 2, 4, 8] if ctx.is_moe else [1],
        }

        if not ctx.is_moe:
            if ctx.system_name == "gb200_sxm":
                prefill_worker_config["num_gpu_per_worker"] = [1, 2, 4, 8, 16]
                prefill_worker_config["tp_list"] = [1, 2, 4, 8, 16]
                prefill_worker_config["pp_list"] = [1]
            if decode_system == "gb200_sxm":
                decode_worker_config["num_gpu_per_worker"] = [1, 2, 4, 8, 16]
                decode_worker_config["tp_list"] = [1, 2, 4, 8, 16]
                decode_worker_config["pp_list"] = [1]
        else:
            if ctx.enable_wide_ep:
                prefill_worker_config["num_gpu_per_worker"] = [1, 2, 4, 8, 16]
                prefill_worker_config["tp_list"] = [1, 2, 4, 8]
                prefill_worker_config["pp_list"] = [1, 2, 4, 8, 16] if should_enable_pp else [1]
                prefill_worker_config["dp_list"] = [1, 2, 4]
                prefill_worker_config["moe_ep_list"] = [1, 2, 4, 8, 16]

                decode_worker_config["num_gpu_per_worker"] = [1, 2, 4, 8, 16, 32, 64]
                decode_worker_config["tp_list"] = [1, 2, 4, 8]
                decode_worker_config["pp_list"] = [1, 2, 4, 8, 16, 32, 64] if should_enable_pp else [1]
                decode_worker_config["dp_list"] = [1, 2, 4, 8, 16, 32, 64]
                decode_worker_config["moe_ep_list"] = [1, 2, 4, 8, 16, 32, 64]

        replica_config = {
            "num_gpu_per_replica": [
                1,
                2,
                4,
                8,
                16,
                24,
                32,
                40,
                48,
                56,
                64,
                72,
                80,
                88,
                96,
                104,
                112,
                120,
                128,
            ],
            "max_gpu_per_replica": 128,
            "max_prefill_worker": 32,
            "max_decode_worker": 32,
        }

        if ctx.enable_wide_ep:
            replica_config["num_gpu_per_replica"] = None
            replica_config["max_gpu_per_replica"] = 512

        advanced_tuning_config = {
            "prefill_latency_correction_scale": 1.1,
            "decode_latency_correction_scale": 1.08,
            "prefill_max_batch_size": 1,
            "decode_max_batch_size": 512,
        }

        return {
            "is_moe": ctx.is_moe,
            "prefill_worker_config": prefill_worker_config,
            "decode_worker_config": decode_worker_config,
            "replica_config": replica_config,
            "advanced_tuning_config": advanced_tuning_config,
        }

    @classmethod
    def _finalize_agg(cls, config: DefaultMunch, ctx: TaskContext) -> None:
        worker_config = config.worker_config

        if ctx.total_gpus is not None:
            if ctx.total_gpus < 0:
                raise ValueError(
                    f"total_gpus of agg must be no smaller than 0, got {ctx.total_gpus}"
                )
            worker_config.num_gpu_per_worker = [
                num for num in worker_config.num_gpu_per_worker if num <= ctx.total_gpus
            ]
            logger.debug(
                "Overwriting num gpu per worker to %s", worker_config.num_gpu_per_worker
            )

        cls._apply_quant_modes(
            target_cfg=worker_config,
            model_name=ctx.model_name,
            system=worker_config.system_name,
            backend=worker_config.backend_name,
            version=worker_config.backend_version,
            preferred_mode=ctx.use_specific_quant_mode,
        )

    @classmethod
    def _finalize_disagg(cls, config: DefaultMunch, ctx: TaskContext) -> None:
        prefill_cfg = config.prefill_worker_config
        decode_cfg = config.decode_worker_config
        replica_cfg = config.replica_config

        # if replica_cfg.max_gpu_per_replica is overwritten by patch, extend the num_gpu_per_replica if needed
        max_from_config = replica_cfg.get("max_gpu_per_replica")
        if max_from_config and max_from_config > 0:
            if replica_cfg.num_gpu_per_replica is not None:
                while max_from_config > max(replica_cfg.num_gpu_per_replica):
                    replica_cfg.num_gpu_per_replica.append(
                        max(replica_cfg.num_gpu_per_replica) + 8
                    )

        # using total gpus to limit the max gpu per replica
        if ctx.total_gpus is not None:
            if ctx.total_gpus < 2:
                raise ValueError(
                    f"total_gpus must be greater than 2 for disagg, got {ctx.total_gpus}"
                )
            replica_cfg.max_gpu_per_replica = min(ctx.total_gpus, replica_cfg.get("max_gpu_per_replica"))
            logger.debug("Using max gpu per replica %s", replica_cfg.max_gpu_per_replica)

        cls._apply_quant_modes(
            target_cfg=prefill_cfg,
            model_name=ctx.model_name,
            system=prefill_cfg.system_name,
            backend=prefill_cfg.backend_name,
            version=prefill_cfg.backend_version,
            preferred_mode=ctx.use_specific_quant_mode,
        )

        cls._apply_quant_modes(
            target_cfg=decode_cfg,
            model_name=ctx.model_name,
            system=decode_cfg.system_name,
            backend=decode_cfg.backend_name,
            version=decode_cfg.backend_version,
            preferred_mode=ctx.use_specific_quant_mode,
        )

    @staticmethod
    def _apply_quant_modes(
        target_cfg: DefaultMunch,
        model_name: str,
        system: str,
        backend: str,
        version: str,
        preferred_mode: Optional[str],
    ) -> None:
        quant_keys = [
            "gemm_quant_mode",
            "moe_quant_mode",
            "kvcache_quant_mode",
            "fmha_quant_mode",
            "comm_quant_mode",
        ]

        # Check if all quant modes are already set with string values
        existing = {key: getattr(target_cfg, key, None) for key in quant_keys}
        if all(value is not None and isinstance(value, str) for value in existing.values()):
            return

        database = get_database(system=system, backend=backend, version=version)
        defaults = TaskConfigFactory._get_quant_mode(
            model_name=model_name,
            database=database,
            use_specific_quant_mode=preferred_mode,
        )

        for key, value in zip(quant_keys, defaults):
            current = getattr(target_cfg, key, None)
            if current is None or not isinstance(current, str):
                setattr(target_cfg, key, value)

    @staticmethod
    def _get_quant_mode(
        model_name: str,
        database: PerfDatabase,
        use_specific_quant_mode: Optional[str] = None,
    ) -> tuple[str, str, str, str, str]:
        gemm_quant_mode = "fp8_block"
        moe_quant_mode = "fp8_block"
        kvcache_quant_mode = "fp8"
        fmha_quant_mode = "float16" if model_name in ["DEEPSEEK_V3", "KIMI_K2"] else "fp8"
        comm_quant_mode = "half"

        sm_version = database.system_spec["gpu"]["sm_version"]

        if sm_version >= 100:
            gemm_quant_mode = "nvfp4"
            moe_quant_mode = "nvfp4"
            kvcache_quant_mode = "fp8"
            fmha_quant_mode = "fp8"
        elif sm_version >= 89:
            gemm_quant_mode = "fp8_block"
            moe_quant_mode = "fp8_block"
            kvcache_quant_mode = "fp8"
            fmha_quant_mode = "fp8"
        else:
            gemm_quant_mode = "float16"
            moe_quant_mode = "float16"
            kvcache_quant_mode = "float16"
            fmha_quant_mode = "float16"

        if model_name in ["DEEPSEEK_V3", "KIMI_K2"]:
            fmha_quant_mode = "float16"
        if any(keyword in model_name for keyword in ["MOE_Mixtral", "QWEN2", "LLAMA"]):
            if sm_version < 100 and sm_version >= 89:
                gemm_quant_mode = "fp8"
                moe_quant_mode = "fp8"

        if use_specific_quant_mode is not None:
            if use_specific_quant_mode != "w4afp8":
                gemm_quant_mode = use_specific_quant_mode
            moe_quant_mode = use_specific_quant_mode

        return (
            gemm_quant_mode,
            moe_quant_mode,
            kvcache_quant_mode,
            fmha_quant_mode,
            comm_quant_mode,
        )


_quants = {
    "fp8_default": {
        "gemm_quant_mode": "fp8",
        "moe_quant_mode": "fp8",
        "kvcache_quant_mode": "fp8",
        "fmha_quant_mode": "fp8",
        "comm_quant_mode": "half",
    },
    "float16_default": {
        "gemm_quant_mode": "float16",
        "moe_quant_mode": "float16",
        "kvcache_quant_mode": "float16",
        "fmha_quant_mode": "float16",
        "comm_quant_mode": "half",
    },
    "nvfp4_default": {
        "gemm_quant_mode": "nvfp4",
        "moe_quant_mode": "nvfp4",
        "kvcache_quant_mode": "fp8",
        "fmha_quant_mode": "fp8",
        "comm_quant_mode": "half",
    },
}


def _quant_profile_layers(name: str, overrides: dict[str, str]) -> list[ConfigLayer]:
    def _quant_payload(target: str) -> dict[str, dict[str, str]]:
        return {target: overrides}

    return [
        ConfigLayer(
            name=f"{name}-agg",
            condition=lambda ctx: ctx.serving_mode == "agg",
            data=lambda ctx: _quant_payload("worker_config"),
        ),
        ConfigLayer(
            name=f"{name}-prefill",
            condition=lambda ctx: ctx.serving_mode == "disagg",
            data=lambda ctx: _quant_payload("prefill_worker_config"),
        ),
        ConfigLayer(
            name=f"{name}-decode",
            condition=lambda ctx: ctx.serving_mode == "disagg",
            data=lambda ctx: _quant_payload("decode_worker_config"),
        ),
    ]


def register_builtin_profiles() -> None:
    for name, overrides in _quants.items():
        TaskConfigFactory.register_profile(name, _quant_profile_layers(name, overrides))


register_builtin_profiles()

def task_config_to_generator_config(task_config: TaskConfig, result_df: pd.DataFrame) -> dict:
    """Convert a task config and result dataframe to a generator config."""
    def _build_worker_config_from_result_df(prefix: str, result_df: pd.Series) -> dict:
        return {
            "gemm_quant_mode": result_df[f'{prefix}gemm'],
            "moe_quant_mode": result_df[f'{prefix}moe'],
            "kvcache_quant_mode": result_df[f'{prefix}kvcache'],
            "fmha_quant_mode": result_df[f'{prefix}fmha'],
            "comm_quant_mode": result_df[f'{prefix}comm'],
            "bs": result_df[f'{prefix}bs'],
            "workers": result_df.get(f'{prefix}workers', 1),
            "tp": result_df[f'{prefix}tp'],
            "pp": result_df[f'{prefix}pp'],
            "dp": result_df[f'{prefix}dp'],
            "moe_tp": result_df[f'{prefix}moe_tp'],
            "moe_ep": result_df[f'{prefix}moe_ep'],
            "memory": result_df[f'{prefix}memory'],
        }

    cfg = {
        "model_name": task_config.model_name,
        "system_name": task_config.system_name,
        "decode_system_name": getattr(task_config, "decode_system_name", None),
        "total_gpus": task_config.total_gpus,
        "serving_mode": task_config.serving_mode,
        "nextn": task_config.config.nextn,
        "is_moe": task_config.config.is_moe,
        "isl": task_config.config.runtime_config.isl,
        "osl": task_config.config.runtime_config.osl,
    }

    if task_config.serving_mode == "agg":
        cfg["agg_worker_config"] = _build_worker_config_from_result_df("", result_df)
    else:
        cfg["disagg_prefill_worker_config"] = _build_worker_config_from_result_df("(p)", result_df)
        cfg["disagg_decode_worker_config"] = _build_worker_config_from_result_df("(d)", result_df)

    cfg["exp_config"] = {
        "ttft": result_df['ttft'],
        "tps_per_user": result_df['tokens/s/user'],                        
        "tps_per_gpu": result_df['tokens/s/gpu_cluster'] or result_df['tokens/s/gpu'],
        "concurrency_per_replica": result_df['concurrency'],
        "num_requests_multiplier": 10,
    }

    return cfg


class TaskConfig:
    def __init__(
        self,
        serving_mode: str,
        model_name: str,
        system_name: str,
        decode_system_name: Optional[str] = None,
        backend_name: str = "trtllm",
        backend_version: Optional[str] = None,
        use_specific_quant_mode: Optional[str] = None,
        isl: int = 4000,
        osl: int = 1000,
        ttft: float = 1000,
        tpot: float = 50,
        enable_wide_ep: bool = False,
        total_gpus: Optional[int] = None,
        profiles: Optional[list[str]] = None,
        yaml_config: Optional[dict] = None,
    ) -> None:
        yaml_mode = "patch"
        yaml_patch: dict = {}
        effective_profiles: list[str] = list(profiles or [])

        if yaml_config is not None:
            logger.info(
                "Task %s: Overwriting config from YAML: %s",
                f"{serving_mode}_{model_name}",
                yaml_config,
            )
            yaml_mode = yaml_config.get("mode", "patch")
            if yaml_mode not in {"patch", "replace"}:
                raise ValueError(f"Invalid yaml mode: {yaml_mode}")
            yaml_profiles = yaml_config.get("profiles", [])
            if profiles and yaml_profiles:
                logger.warning("Both constructor profiles and YAML profiles provided; combining them")
            effective_profiles = list(dict.fromkeys([*effective_profiles, *yaml_profiles]))
            yaml_patch = yaml_config.get("config", yaml_config)

        ctx = TaskContext(
            serving_mode=serving_mode,
            model_name=model_name,
            system_name=system_name,
            decode_system_name=decode_system_name,
            backend_name=backend_name,
            backend_version=backend_version,
            use_specific_quant_mode=use_specific_quant_mode,
            enable_wide_ep=enable_wide_ep,
            isl=isl,
            osl=osl,
            ttft=ttft,
            tpot=tpot,
            total_gpus=total_gpus,
            profiles=effective_profiles,
            yaml_patch=yaml_patch,
            yaml_mode=yaml_mode,
        )

        self.config, applied_layers = TaskConfigFactory.create(ctx)
        self.config.applied_layers = applied_layers

        self.serving_mode = serving_mode
        self.model_name = model_name
        self.system_name = system_name
        self.decode_system_name = decode_system_name
        self.backend_name = backend_name
        self.use_specific_quant_mode = use_specific_quant_mode
        self.enable_wide_ep = enable_wide_ep
        self.total_gpus = total_gpus
        self.yaml_mode = yaml_mode
        self.yaml_patch = yaml_patch
        self.profiles = list(effective_profiles)

        if serving_mode == "agg":
            effective_backend_version = self.config.worker_config.backend_version
            self.backend_version = effective_backend_version
        elif serving_mode == "disagg":
            prefill_backend_version = self.config.prefill_worker_config.backend_version
            decode_backend_version = self.config.decode_worker_config.backend_version
            self.prefill_backend_version = prefill_backend_version
            self.decode_backend_version = decode_backend_version
            if prefill_backend_version == decode_backend_version:
                effective_backend_version = prefill_backend_version
            else:
                effective_backend_version = f"{prefill_backend_version}-{decode_backend_version}"
            self.backend_version = effective_backend_version
        else:
            effective_backend_version = backend_version
            self.backend_version = backend_version

        self.task_name = (
            f"{serving_mode}_{model_name}_{system_name}_{decode_system_name}_{backend_name}_{effective_backend_version}_{isl}_{osl}_{ttft}_{tpot}"
        ) if serving_mode == "disagg" else (
            f"{serving_mode}_{model_name}_{system_name}_{backend_name}_{effective_backend_version}_{isl}_{osl}_{ttft}_{tpot}"
        )
        self.config.task_name = self.task_name

        if serving_mode == "agg":
            self._convert_worker_config_to_enum(self.config.worker_config)
            logger.info("Task %s: Runtime config: %s", self.task_name, self.config.runtime_config)
            logger.info("Task %s: Worker config: %s", self.task_name, self.config.worker_config)
        elif serving_mode == "disagg":
            self._convert_worker_config_to_enum(self.config.prefill_worker_config)
            self._convert_worker_config_to_enum(self.config.decode_worker_config)
            logger.info("Task %s: Runtime config: %s", self.task_name, self.config.runtime_config)
            logger.info("Task %s: Prefill worker config: %s", self.task_name, self.config.prefill_worker_config)
            logger.info("Task %s: Decode worker config: %s", self.task_name, self.config.decode_worker_config)
            logger.info("Task %s: Replica config: %s", self.task_name, self.config.replica_config)
            logger.info("Task %s: Advanced tuning config: %s", self.task_name, self.config.advanced_tuning_config)
        else:
            raise ValueError(f"Invalid serving mode: {serving_mode}")

    def pretty(self) -> str:
        def _convert(obj: Any) -> Any:
            if isinstance(obj, DefaultMunch):
                return {key: _convert(value) for key, value in obj.items()}
            if isinstance(obj, list):
                return [_convert(item) for item in obj]
            if isinstance(obj, tuple):
                return tuple(_convert(item) for item in obj)
            if hasattr(obj, "name"):
                return getattr(obj, "name")
            return obj

        printable: Dict[str, Any] = {
            "mode": self.yaml_mode,
            "serving_mode": self.serving_mode,
            "model_name": self.model_name,
            "total_gpus": self.total_gpus,
            "system_name": self.system_name,
        }

        if self.config.serving_mode == "disagg":
            printable["decode_system_name"] = self.decode_system_name

        printable["backend_name"] = self.backend_name
        printable["backend_version"] = self.backend_version

        runtime_dict = _convert(self.config.runtime_config)
        printable.update({k: runtime_dict.get(k) for k in ("isl", "osl", "ttft", "tpot") if runtime_dict.get(k) is not None})

        base_config = _convert(getattr(self.config, "yaml_patch", getattr(self, "yaml_patch", {})))
        printable["profiles"] = self.profiles

        def _ensure_dict(target: Dict[str, Any], key: str) -> Dict[str, Any]:
            value = target.setdefault(key, {})
            if not isinstance(value, dict):
                raise ValueError(f"Expected dict for config['{key}'], got {type(value)}")
            return value

        config_section: Dict[str, Any] = dict(base_config) if isinstance(base_config, dict) else {}

        if getattr(self.config, "nextn", None) is not None:
            config_section.setdefault("nextn", self.config.nextn)
        if getattr(self.config, "nextn_accept_rates", None) is not None:
            config_section.setdefault("nextn_accept_rates", self.config.nextn_accept_rates)

        if self.config.serving_mode == "agg" and hasattr(self.config, "worker_config"):
            wc = _convert(self.config.worker_config)
            _ensure_dict(config_section, "worker_config").update(wc)
        elif self.config.serving_mode == "disagg":
            for key in ("prefill_worker_config", "decode_worker_config", "replica_config", "advanced_tuning_config"):
                value = getattr(self.config, key, None)
                if value is not None:
                    cfg = _convert(value)
                    if isinstance(cfg, dict):
                        _ensure_dict(config_section, key).update(cfg)
                    else:
                        config_section[key] = cfg

        if config_section:
            printable["config"] = config_section
        
        final_dict = {self.task_name: printable}

        return json.dumps(final_dict, indent=2)

    def _convert_worker_config_to_enum(self, worker_config: Union[dict, DefaultMunch]) -> None:
        """Convert string quant mode values to enums, skip if already converted."""
        worker_cfg = _ensure_munch(worker_config)
        
        # Only convert if the value is a string
        if isinstance(worker_cfg.gemm_quant_mode, str):
            worker_cfg.gemm_quant_mode = common.GEMMQuantMode[worker_cfg.gemm_quant_mode]
        if isinstance(worker_cfg.moe_quant_mode, str):
            worker_cfg.moe_quant_mode = common.MoEQuantMode[worker_cfg.moe_quant_mode]
        if isinstance(worker_cfg.kvcache_quant_mode, str):
            worker_cfg.kvcache_quant_mode = common.KVCacheQuantMode[worker_cfg.kvcache_quant_mode]
        if isinstance(worker_cfg.fmha_quant_mode, str):
            worker_cfg.fmha_quant_mode = common.FMHAQuantMode[worker_cfg.fmha_quant_mode]
        if isinstance(worker_cfg.comm_quant_mode, str):
            worker_cfg.comm_quant_mode = common.CommQuantMode[worker_cfg.comm_quant_mode]
        
        if isinstance(worker_config, dict):
            worker_config.update(worker_cfg)
        else:
            worker_config.gemm_quant_mode = worker_cfg.gemm_quant_mode
            worker_config.moe_quant_mode = worker_cfg.moe_quant_mode
            worker_config.kvcache_quant_mode = worker_cfg.kvcache_quant_mode
            worker_config.fmha_quant_mode = worker_cfg.fmha_quant_mode
            worker_config.comm_quant_mode = worker_cfg.comm_quant_mode


class TaskRunner:
    def run_agg(self, task_config: DefaultMunch) -> dict[str, Optional[pd.DataFrame]]:
        logger.info("Task %s: Setting up runtime config", task_config.task_name)
        runtime_config = config.RuntimeConfig(
            isl=task_config.runtime_config.isl,
            osl=task_config.runtime_config.osl,
            ttft=task_config.runtime_config.ttft,
            tpot=list(range(1, 20, 1)) + list(range(20, 300, 5)),
        )
        logger.info("Task %s: Setting up database", task_config.task_name)
        try:
            database = copy.deepcopy(
                get_database(
                    system=task_config.worker_config.system_name,
                    backend=task_config.worker_config.backend_name,
                    version=task_config.worker_config.backend_version,
                )
            )
        except Exception as e:  # pragma: no cover
            logger.error(
                "Error getting database for %s %s %s: %s",
                task_config.worker_config.system_name,
                task_config.worker_config.backend_name,
                task_config.worker_config.backend_version,
                e,
            )
            return None
        logger.info("Task %s: Setting up model config", task_config.task_name)
        model_config = config.ModelConfig(
            gemm_quant_mode=task_config.worker_config.gemm_quant_mode,
            kvcache_quant_mode=task_config.worker_config.kvcache_quant_mode,
            fmha_quant_mode=task_config.worker_config.fmha_quant_mode,
            moe_quant_mode=task_config.worker_config.moe_quant_mode,
            comm_quant_mode=task_config.worker_config.comm_quant_mode,
            nextn=task_config.nextn,
            nextn_accept_rates=task_config.nextn_accept_rates,
        )
        logger.info("Task %s: Enumerating parallel config", task_config.task_name)
        try:
            from aiconfigurator.sdk import pareto_analysis as pa

            parallel_config_list = pa.enumerate_parallel_config(
                num_gpu_list=task_config.worker_config.num_gpu_per_worker,
                tp_list=task_config.worker_config.tp_list,
                pp_list=task_config.worker_config.pp_list,
                dp_list=task_config.worker_config.dp_list,
                moe_tp_list=task_config.worker_config.moe_tp_list,
                moe_ep_list=task_config.worker_config.moe_ep_list,
                is_moe=check_is_moe(task_config.model_name),
                backend=common.BackendName(task_config.worker_config.backend_name),
            )
        except Exception as e:  # pragma: no cover
            logger.error(
                "Error enumerating parallel config for %s %s %s: %s",
                task_config.worker_config.system_name,
                task_config.worker_config.backend_name,
                task_config.worker_config.backend_version,
                e,
            )
            return None
        logger.info("Task %s: Running agg pareto", task_config.task_name)
        result_df = pa.agg_pareto(
            model_name=task_config.model_name,
            runtime_config=runtime_config,
            database=database,
            backend_name=task_config.worker_config.backend_name,
            model_config=model_config,
            parallel_config_list=parallel_config_list,
        )
        return {"pareto_df": result_df, "pareto_frontier_df": pa.get_pareto_front(result_df, 'tokens/s/user', 'tokens/s/gpu').reset_index(drop=True).reset_index()}

    def run_disagg(self, task_config: DefaultMunch) -> dict[str, Optional[pd.DataFrame]]:
        logger.info("Task %s: Setting up runtime config", task_config.task_name)
        runtime_config = config.RuntimeConfig(
            isl=task_config.runtime_config.isl,
            osl=task_config.runtime_config.osl,
            ttft=task_config.runtime_config.ttft,
            tpot=list(range(1, 20, 1)) + list(range(20, 300, 5)),
        )

        logger.info("Task %s: Setting up prefill database", task_config.task_name)
        try:
            prefill_database = copy.deepcopy(
                get_database(
                    system=task_config.prefill_worker_config.system_name,
                    backend=task_config.prefill_worker_config.backend_name,
                    version=task_config.prefill_worker_config.backend_version,
                )
            )
        except Exception as e:  # pragma: no cover
            logger.error(
                "Error getting prefill database for %s %s %s: %s",
                task_config.prefill_worker_config.system_name,
                task_config.prefill_worker_config.backend_name,
                task_config.prefill_worker_config.backend_version,
                e,
            )
            return None
        logger.info("Task %s: Setting up prefill model config", task_config.task_name)
        prefill_model_config = config.ModelConfig(
            gemm_quant_mode=task_config.prefill_worker_config.gemm_quant_mode,
            kvcache_quant_mode=task_config.prefill_worker_config.kvcache_quant_mode,
            fmha_quant_mode=task_config.prefill_worker_config.fmha_quant_mode,
            moe_quant_mode=task_config.prefill_worker_config.moe_quant_mode,
            comm_quant_mode=task_config.prefill_worker_config.comm_quant_mode,
            nextn=task_config.nextn,
            nextn_accept_rates=task_config.nextn_accept_rates,
        )

        logger.info("Task %s: Enumerating prefill parallel config", task_config.task_name)
        try:
            from aiconfigurator.sdk import pareto_analysis as pa

            prefill_parallel_config_list = pa.enumerate_parallel_config(
                num_gpu_list=task_config.prefill_worker_config.num_gpu_per_worker,
                tp_list=task_config.prefill_worker_config.tp_list,
                pp_list=task_config.prefill_worker_config.pp_list,
                dp_list=task_config.prefill_worker_config.dp_list,
                moe_tp_list=task_config.prefill_worker_config.moe_tp_list,
                moe_ep_list=task_config.prefill_worker_config.moe_ep_list,
                is_moe=check_is_moe(task_config.model_name),
                backend=common.BackendName(task_config.prefill_worker_config.backend_name),
            )
        except Exception as e:  # pragma: no cover
            logger.error(
                "Error enumerating prefill parallel config for %s %s %s: %s",
                task_config.prefill_worker_config.system_name,
                task_config.prefill_worker_config.backend_name,
                task_config.prefill_worker_config.backend_version,
                e,
            )
            return None

        logger.info("Task %s: Setting up decode database", task_config.task_name)
        try:
            decode_database = copy.deepcopy(
                get_database(
                    system=task_config.decode_worker_config.system_name,
                    backend=task_config.decode_worker_config.backend_name,
                    version=task_config.decode_worker_config.backend_version,
                )
            )
        except Exception as e:  # pragma: no cover
            logger.error(
                "Error getting decode database for %s %s %s: %s",
                task_config.decode_worker_config.system_name,
                task_config.decode_worker_config.backend_name,
                task_config.decode_worker_config.backend_version,
                e,
            )
            return None
        logger.info("Task %s: Setting up decode model config", task_config.task_name)
        decode_model_config = config.ModelConfig(
            gemm_quant_mode=task_config.decode_worker_config.gemm_quant_mode,
            kvcache_quant_mode=task_config.decode_worker_config.kvcache_quant_mode,
            fmha_quant_mode=task_config.decode_worker_config.fmha_quant_mode,
            moe_quant_mode=task_config.decode_worker_config.moe_quant_mode,
            comm_quant_mode=task_config.decode_worker_config.comm_quant_mode,
            nextn=task_config.nextn,
            nextn_accept_rates=task_config.nextn_accept_rates,
        )

        logger.info("Task %s: Enumerating decode parallel config", task_config.task_name)
        try:
            from aiconfigurator.sdk import pareto_analysis as pa

            decode_parallel_config_list = pa.enumerate_parallel_config(
                num_gpu_list=task_config.decode_worker_config.num_gpu_per_worker,
                tp_list=task_config.decode_worker_config.tp_list,
                pp_list=task_config.decode_worker_config.pp_list,
                dp_list=task_config.decode_worker_config.dp_list,
                moe_tp_list=task_config.decode_worker_config.moe_tp_list,
                moe_ep_list=task_config.decode_worker_config.moe_ep_list,
                is_moe=check_is_moe(task_config.model_name),
                backend=common.BackendName(task_config.decode_worker_config.backend_name),
            )
        except Exception as e:  # pragma: no cover
            logger.error(
                "Error enumerating decode parallel config for %s %s %s: %s",
                task_config.decode_worker_config.system_name,
                task_config.decode_worker_config.backend_name,
                task_config.decode_worker_config.backend_version,
                e,
            )
            return None

        logger.info("Task %s: Running disagg pareto", task_config.task_name)
        result_df = pa.disagg_pareto(
            model_name=task_config.model_name,
            runtime_config=runtime_config,
            prefill_database=prefill_database,
            prefill_backend_name=task_config.prefill_worker_config.backend_name,
            prefill_model_config=prefill_model_config,
            prefill_parallel_config_list=prefill_parallel_config_list,
            decode_database=decode_database,
            decode_backend_name=task_config.decode_worker_config.backend_name,
            decode_model_config=decode_model_config,
            decode_parallel_config_list=decode_parallel_config_list,
            num_gpu_list=task_config.replica_config.num_gpu_per_replica,
            max_num_gpu=task_config.replica_config.max_gpu_per_replica,
            prefill_max_num_worker=task_config.replica_config.max_prefill_worker,
            decode_max_num_worker=task_config.replica_config.max_decode_worker,
            prefill_max_num_tokens=task_config.advanced_tuning_config.prefill_max_batch_size*task_config.runtime_config.isl,
            decode_max_num_tokens=task_config.advanced_tuning_config.decode_max_batch_size,
            prefill_latency_correction_scale=task_config.advanced_tuning_config.prefill_latency_correction_scale,
            decode_latency_correction_scale=task_config.advanced_tuning_config.decode_latency_correction_scale,
        )
        return {"pareto_df": result_df, "pareto_frontier_df": pa.get_pareto_front(result_df, 'tokens/s/user', 'tokens/s/gpu').reset_index(drop=True).reset_index()}

    def run(self, task_config: TaskConfig) -> dict[str, Optional[pd.DataFrame]]:
        serving_mode = task_config.config.serving_mode
        logger.info(
            "Starting Pareto Analysis for %s in %s mode...",
            task_config.task_name,
            serving_mode,
        )
        try:
            if serving_mode == "agg":
                result = self.run_agg(task_config.config)
            elif serving_mode == "disagg":
                result = self.run_disagg(task_config.config)
            else:
                raise ValueError(f"Invalid serving mode: {serving_mode}")
        except Exception as e:
            logger.error(
                "Error running pareto analysis for %s in %s mode: %s",
                task_config.task_name,
                serving_mode,
                e,
            )
            result = None

        if result is None:
            logger.warning("No result found for %s in %s mode.", task_config.task_name, serving_mode)

        return result

if __name__ == "__main__":

    task_agg = TaskConfig(
        serving_mode="agg",
        model_name="QWEN3_32B",
        system_name="h200_sxm",
        ttft=600,
        tpot=20,
        isl=4000,
        osl=500,
        total_gpus=8,
    )
    task_runner = TaskRunner()
    print("\n=== TaskConfig (agg) ===")
    print(task_agg.pretty())
    agg_df = task_runner.run(task_agg)["pareto_frontier_df"]
    agg_df.to_csv("agg_df.csv", index=False)
    print("\n=== agg pareto ===")
    print(agg_df)

    task_disagg = TaskConfig(
        serving_mode="disagg",
        model_name="QWEN3_32B",
        system_name="h200_sxm",
        ttft=600,
        tpot=20,
        isl=4000,
        osl=500,
        total_gpus=16,
        profiles=["fp8_default"],
        yaml_config={
            "mode": "patch",
            "config": {
                "advanced_tuning_config": {"prefill_latency_correction_scale": 1.1, "decode_latency_correction_scale": 1.08},
            },
        },
    )
    print("\n=== TaskConfig (disagg) ===")
    print(task_disagg.pretty())
    disagg_df = task_runner.run(task_disagg)["pareto_frontier_df"]
    disagg_df.to_csv("disagg_df.csv", index=False)
    print("\n=== disagg pareto ===")
    print(disagg_df)