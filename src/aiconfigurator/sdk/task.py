# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import pandas as pd
from munch import DefaultMunch, Munch

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.models import _apply_model_quant_defaults, check_is_moe, get_model_family
from aiconfigurator.sdk.pareto_analysis import get_pareto_front
from aiconfigurator.sdk.perf_database import get_database, get_latest_database_version
from aiconfigurator.sdk.utils import enumerate_parallel_config, get_model_config_from_model_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfigLayer:
    name: str
    data: dict | Callable[[TaskContext], dict]
    condition: Callable[[TaskContext], bool] | None = None

    def applies_to(self, ctx: TaskContext) -> bool:
        if self.condition is None:
            return True
        try:
            return self.condition(ctx)
        except Exception:  # pragma: no cover
            logger.debug("Layer %s condition evaluation failed", self.name)
            return False

    def resolve(self, ctx: TaskContext) -> dict:
        payload = self.data(ctx) if callable(self.data) else self.data
        return copy.deepcopy(payload)


@dataclass
class TaskContext:
    serving_mode: Literal["agg", "disagg"]
    model_path: str
    model_family: str
    system_name: str
    decode_system_name: str | None
    backend_name: str
    backend_version: str | None
    isl: int
    osl: int
    prefix: int
    ttft: float | None
    tpot: float | None
    request_latency: float | None
    enable_wideep: bool
    total_gpus: int | None
    profiles: list[str] = field(default_factory=list)
    yaml_patch: dict = field(default_factory=dict)
    yaml_mode: Literal["patch", "replace"] = "patch"

    @property
    def is_moe(self) -> bool:
        return check_is_moe(self.model_path)

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


def _ensure_munch(obj: dict | DefaultMunch | Munch) -> DefaultMunch:
    if isinstance(obj, (DefaultMunch, Munch)):
        return DefaultMunch.fromDict(obj.toDict(), DefaultMunch)
    return DefaultMunch.fromDict(obj, DefaultMunch)


def build_disagg_parallel_lists(
    backend_name: str,
    prefill_system: str,
    decode_system: str,
    is_moe: bool,
    enable_wideep: bool = False,
    should_enable_pp: bool = False,
) -> tuple[dict, dict]:
    """Build the TP/PP/DP/MoE-TP/MoE-EP search-space lists for disagg enumeration.

    This is the single source of truth shared by :class:`TaskConfigFactory` (for the
    default CLI sweep) and the profiling enumeration in
    ``aiconfigurator.generator.enumerate``.

    Args:
        backend_name: Backend identifier (``"trtllm"``, ``"sglang"``, ``"vllm"``).
        prefill_system: System name for the prefill worker (e.g. ``"h200_sxm"``).
        decode_system: System name for the decode worker.
        is_moe: Whether the model is a Mixture-of-Experts model.
        enable_wideep: Enable wide expert-parallelism search space.
        should_enable_pp: Enable pipeline-parallelism candidates (default ``False``).

    Returns:
        ``(prefill_worker_config, decode_worker_config)`` - two dicts each containing
        the keys ``num_gpu_per_worker``, ``tp_list``, ``pp_list``, ``dp_list``,
        ``moe_tp_list``, ``moe_ep_list``.
    """
    prefill_worker_config: dict = {
        "num_gpu_per_worker": [1, 2, 4, 8],
        "tp_list": [1, 2, 4, 8],
        "pp_list": [1, 2, 4, 8] if should_enable_pp else [1],
        "dp_list": [1],
        "moe_tp_list": [1],
        "moe_ep_list": [1, 2, 4, 8] if is_moe else [1],
    }

    decode_worker_config: dict = {
        "num_gpu_per_worker": [1, 2, 4, 8],
        "tp_list": [1, 2, 4, 8],
        "pp_list": [1, 2, 4, 8] if should_enable_pp else [1],
        "dp_list": [1, 2, 4, 8] if is_moe else [1],
        "moe_tp_list": [1],
        "moe_ep_list": [1, 2, 4, 8] if is_moe else [1],
    }

    if not is_moe:
        if prefill_system in ["gb200", "gb300"]:
            prefill_worker_config["num_gpu_per_worker"] = [1, 2, 4, 8, 16]
            prefill_worker_config["tp_list"] = [1, 2, 4, 8, 16]
            prefill_worker_config["pp_list"] = [1]
        if decode_system in ["gb200", "gb300"]:
            decode_worker_config["num_gpu_per_worker"] = [1, 2, 4, 8, 16]
            decode_worker_config["tp_list"] = [1, 2, 4, 8, 16]
            decode_worker_config["pp_list"] = [1]
    else:
        if backend_name == "trtllm":
            if enable_wideep:
                prefill_worker_config["num_gpu_per_worker"] = [4, 8, 16, 32]
                prefill_worker_config["tp_list"] = [1, 2, 4, 8]
                prefill_worker_config["pp_list"] = [1, 2, 4, 8, 16, 32] if should_enable_pp else [1]
                prefill_worker_config["dp_list"] = [4, 8, 16, 32]
                prefill_worker_config["moe_tp_list"] = [1]
                prefill_worker_config["moe_ep_list"] = [4, 8, 16, 32]

                decode_worker_config["num_gpu_per_worker"] = [4, 8, 16, 32, 64]
                decode_worker_config["tp_list"] = [1, 2, 4, 8]
                decode_worker_config["pp_list"] = [1, 2, 4, 8, 16, 32, 64] if should_enable_pp else [1]
                decode_worker_config["dp_list"] = [4, 8, 16, 32, 64]
                decode_worker_config["moe_tp_list"] = [1]
                decode_worker_config["moe_ep_list"] = [4, 8, 16, 32, 64]
            else:
                parallel_config_list = [1, 2, 4, 8]

                prefill_worker_config["num_gpu_per_worker"] = parallel_config_list
                prefill_worker_config["tp_list"] = parallel_config_list
                prefill_worker_config["pp_list"] = parallel_config_list if should_enable_pp else [1]
                prefill_worker_config["dp_list"] = parallel_config_list
                prefill_worker_config["moe_tp_list"] = parallel_config_list
                prefill_worker_config["moe_ep_list"] = parallel_config_list

                decode_worker_config["num_gpu_per_worker"] = parallel_config_list
                decode_worker_config["tp_list"] = parallel_config_list
                decode_worker_config["pp_list"] = parallel_config_list if should_enable_pp else [1]
                decode_worker_config["dp_list"] = parallel_config_list
                decode_worker_config["moe_tp_list"] = parallel_config_list
                decode_worker_config["moe_ep_list"] = parallel_config_list
        elif backend_name == "sglang":
            if enable_wideep:
                prefill_worker_config["num_gpu_per_worker"] = [8, 16, 32]
                prefill_worker_config["tp_list"] = [1, 2, 4, 8]
                prefill_worker_config["pp_list"] = [1, 2, 4, 8, 16, 32] if should_enable_pp else [1]
                prefill_worker_config["dp_list"] = [1, 2, 4, 8, 16, 32]
                prefill_worker_config["moe_tp_list"] = [1]
                prefill_worker_config["moe_ep_list"] = [8, 16, 32]

                decode_worker_config["num_gpu_per_worker"] = [8, 16, 32, 64]
                decode_worker_config["tp_list"] = [1, 2, 4, 8]
                decode_worker_config["pp_list"] = [1, 2, 4, 8, 16, 32, 64] if should_enable_pp else [1]
                decode_worker_config["dp_list"] = [1, 2, 4, 8, 16, 32, 64]
                decode_worker_config["moe_tp_list"] = [1]
                decode_worker_config["moe_ep_list"] = [8, 16, 32, 64]
            else:
                parallel_config_list = [1, 2, 4, 8]

                prefill_worker_config["num_gpu_per_worker"] = parallel_config_list
                prefill_worker_config["tp_list"] = parallel_config_list
                prefill_worker_config["pp_list"] = parallel_config_list if should_enable_pp else [1]
                prefill_worker_config["dp_list"] = parallel_config_list
                prefill_worker_config["moe_tp_list"] = parallel_config_list
                prefill_worker_config["moe_ep_list"] = [1]

                decode_worker_config["num_gpu_per_worker"] = parallel_config_list
                decode_worker_config["tp_list"] = parallel_config_list
                decode_worker_config["pp_list"] = parallel_config_list if should_enable_pp else [1]
                decode_worker_config["dp_list"] = parallel_config_list
                decode_worker_config["moe_tp_list"] = parallel_config_list
                decode_worker_config["moe_ep_list"] = [1]
        elif backend_name == "vllm":
            parallel_config_list = [1, 2, 4, 8]

            prefill_worker_config["num_gpu_per_worker"] = parallel_config_list
            prefill_worker_config["tp_list"] = parallel_config_list
            prefill_worker_config["pp_list"] = parallel_config_list if should_enable_pp else [1]
            prefill_worker_config["dp_list"] = parallel_config_list
            prefill_worker_config["moe_tp_list"] = parallel_config_list
            prefill_worker_config["moe_ep_list"] = parallel_config_list

            decode_worker_config["num_gpu_per_worker"] = parallel_config_list
            decode_worker_config["tp_list"] = parallel_config_list
            decode_worker_config["pp_list"] = parallel_config_list if should_enable_pp else [1]
            decode_worker_config["dp_list"] = parallel_config_list
            decode_worker_config["moe_tp_list"] = parallel_config_list
            decode_worker_config["moe_ep_list"] = parallel_config_list
        else:
            raise ValueError(f"Invalid backend: {backend_name}")

    return prefill_worker_config, decode_worker_config


class TaskConfigFactory:
    PROFILE_REGISTRY: ClassVar[dict[str, list[ConfigLayer]]] = {}

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

        # after initialize with args and defaults, apply the yaml patch if any
        if ctx.yaml_patch:
            if ctx.yaml_mode == "replace":
                config_dict = copy.deepcopy(ctx.yaml_patch)
                applied_layers.append("yaml_replace")
            else:
                _deep_merge(config_dict, ctx.yaml_patch, allow_new=True)
                applied_layers.append("yaml_patch")

        config = DefaultMunch.fromDict(config_dict, DefaultMunch)

        if config.model_path != ctx.model_path:
            raise ValueError(f"Model name mismatch: base {ctx.model_path} vs. merged {config.model_path}")

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
        nextn = 1 if ctx.model_family == "DEEPSEEK" else 0
        return {
            "serving_mode": ctx.serving_mode,
            "model_path": ctx.model_path,
            "nextn": nextn,
            "nextn_accept_rates": [0.85, 0, 0, 0, 0],
            "runtime_config": {
                "isl": ctx.isl,
                "osl": ctx.osl,
                "prefix": ctx.prefix,
                "ttft": ctx.ttft,
                "tpot": ctx.tpot,
                "request_latency": ctx.request_latency,
            },
            "enable_wideep": ctx.enable_wideep,
            "moe_backend": None,  # sglang wideep only
            "attention_backend": "flashinfer",  # sglang wideep only
        }

    @staticmethod
    def _agg_defaults_layer(ctx: TaskContext) -> dict:
        should_enable_pp = False  # FIXME: need to improve pp alignment and then enable
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
            if ctx.system_name in ["gb200", "gb300"]:
                worker_config["num_gpu_per_worker"] = [1, 2, 4, 8, 16]
                worker_config["tp_list"] = [1, 2, 4, 8, 16]
                worker_config["pp_list"] = [1]
        else:
            if ctx.backend_name == "trtllm":
                if ctx.enable_wideep:
                    # trtllm + wideep: dp > 1 and moe_ep > 1 required
                    worker_config["num_gpu_per_worker"] = [2, 4, 8, 16, 32, 64]
                    worker_config["tp_list"] = [1, 2, 4, 8]
                    worker_config["pp_list"] = [1, 2, 4, 8, 16, 32, 64] if should_enable_pp else [1]
                    worker_config["dp_list"] = [2, 4, 8, 16, 32, 64]
                    worker_config["moe_tp_list"] = [1]
                    worker_config["moe_ep_list"] = [2, 4, 8, 16, 32, 64]
                else:
                    worker_config["num_gpu_per_worker"] = [1, 2, 4, 8]
                    worker_config["tp_list"] = [1, 2, 4, 8]
                    worker_config["pp_list"] = [1, 2, 4, 8] if should_enable_pp else [1]
                    worker_config["dp_list"] = [1, 2, 4, 8]
                    worker_config["moe_tp_list"] = [1, 2, 4, 8]
                    worker_config["moe_ep_list"] = [1, 2, 4, 8]
            elif ctx.backend_name == "sglang":
                if ctx.enable_wideep:
                    # sglang + wideep (keep previous logic)
                    worker_config["num_gpu_per_worker"] = [8, 16, 32, 64]
                    worker_config["tp_list"] = [1, 2, 4, 8]
                    worker_config["pp_list"] = [1, 2, 4, 8, 16, 32, 64] if should_enable_pp else [1]
                    worker_config["dp_list"] = [1, 2, 4, 8, 16, 32, 64]
                    worker_config["moe_tp_list"] = [1]
                    worker_config["moe_ep_list"] = [8, 16, 32, 64]
                else:
                    worker_config["num_gpu_per_worker"] = [1, 2, 4, 8]
                    worker_config["tp_list"] = [1, 2, 4, 8]
                    worker_config["pp_list"] = [1, 2, 4, 8] if should_enable_pp else [1]
                    worker_config["dp_list"] = [1, 2, 4, 8]
                    worker_config["moe_tp_list"] = [1, 2, 4, 8]
                    worker_config["moe_ep_list"] = [1]
            elif ctx.backend_name == "vllm":
                worker_config["num_gpu_per_worker"] = [1, 2, 4, 8]
                worker_config["tp_list"] = [1, 2, 4, 8]
                worker_config["pp_list"] = [1, 2, 4, 8] if should_enable_pp else [1]
                worker_config["dp_list"] = [1, 2, 4, 8]
                worker_config["moe_tp_list"] = [1, 2, 4, 8]
                worker_config["moe_ep_list"] = [1, 2, 4, 8]
            else:
                raise ValueError(f"Invalid backend: {ctx.backend_name}")

        return {
            "is_moe": ctx.is_moe,
            "worker_config": worker_config,
        }

    @staticmethod
    def _disagg_defaults_layer(ctx: TaskContext) -> dict:
        decode_system = ctx.decode_system_name or ctx.system_name

        prefill_worker_config, decode_worker_config = build_disagg_parallel_lists(
            backend_name=ctx.backend_name,
            prefill_system=ctx.system_name,
            decode_system=decode_system,
            is_moe=ctx.is_moe,
            enable_wideep=ctx.enable_wideep,
        )

        # Attach runtime metadata that _disagg_defaults_layer needs but
        # build_disagg_parallel_lists does not own.
        prefill_worker_config["system_name"] = ctx.system_name
        prefill_worker_config["backend_name"] = ctx.backend_name
        prefill_worker_config["backend_version"] = ctx.resolved_backend_version_for(ctx.system_name)

        decode_worker_config["system_name"] = decode_system
        decode_worker_config["backend_name"] = ctx.backend_name
        decode_worker_config["backend_version"] = ctx.resolved_backend_version_for(decode_system)

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

        if ctx.enable_wideep:
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
                raise ValueError(f"total_gpus of agg must be no smaller than 0, got {ctx.total_gpus}")
            worker_config.num_gpu_per_worker = [
                num for num in worker_config.num_gpu_per_worker if num <= ctx.total_gpus
            ]
            logger.debug("Overwriting num gpu per worker to %s", worker_config.num_gpu_per_worker)

    @classmethod
    def _finalize_disagg(cls, config: DefaultMunch, ctx: TaskContext) -> None:
        replica_cfg = config.replica_config

        # if replica_cfg.max_gpu_per_replica is overwritten by patch, extend the num_gpu_per_replica
        # if needed
        max_from_config = replica_cfg.get("max_gpu_per_replica")
        if max_from_config and max_from_config > 0 and replica_cfg.num_gpu_per_replica is not None:
            while max_from_config > max(replica_cfg.num_gpu_per_replica):
                replica_cfg.num_gpu_per_replica.append(max(replica_cfg.num_gpu_per_replica) + 8)

        # using total gpus to limit the max gpu per replica
        if ctx.total_gpus is not None:
            if ctx.total_gpus < 2:
                raise ValueError(f"total_gpus must be greater than 2 for disagg, got {ctx.total_gpus}")
            replica_cfg.max_gpu_per_replica = min(ctx.total_gpus, replica_cfg.get("max_gpu_per_replica"))
            logger.debug("Using max gpu per replica %s", replica_cfg.max_gpu_per_replica)


_quants = {
    "fp8": {
        "gemm_quant_mode": "fp8",
        "moe_quant_mode": "fp8",
        "kvcache_quant_mode": "fp8",
        "fmha_quant_mode": "fp8",
        "comm_quant_mode": "half",
    },
    "fp8_static": {
        "gemm_quant_mode": "fp8_static",
        "moe_quant_mode": "fp8",
        "kvcache_quant_mode": "fp8",
        "fmha_quant_mode": "fp8",
        "comm_quant_mode": "half",
    },
    "float16": {
        "gemm_quant_mode": "float16",
        "moe_quant_mode": "float16",
        "kvcache_quant_mode": "float16",
        "fmha_quant_mode": "float16",
        "comm_quant_mode": "half",
    },
    "nvfp4": {
        "gemm_quant_mode": "nvfp4",
        "moe_quant_mode": "nvfp4",
        "kvcache_quant_mode": "fp8",
        "fmha_quant_mode": "fp8",
        "comm_quant_mode": "half",
    },
    "mxfp4": {
        "gemm_quant_mode": "float16",
        "moe_quant_mode": "w4a16_mxfp4",
        "kvcache_quant_mode": "float16",
        "fmha_quant_mode": "float16",
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


class TaskConfig:
    def __init__(
        self,
        serving_mode: str,
        model_path: str,
        system_name: str,
        decode_system_name: str | None = None,
        backend_name: str = "trtllm",
        backend_version: str | None = None,
        isl: int = 4000,
        osl: int = 1000,
        prefix: int = 0,
        ttft: float = 1000,
        tpot: float = 50,
        request_latency: float | None = None,
        enable_wideep: bool = False,
        total_gpus: int | None = None,
        profiles: list[str] | None = None,
        yaml_config: dict | None = None,
        database_mode: str | None = None,
    ) -> None:
        """
        Initialize a TaskConfig object.
        We use args to initialize and allow passing in a yaml file to do patch.
        The patch order:
        1. args + yaml config (yaml patch) as the ctx
        2. In create, initilize with args and defaults (defined in TaskConfigFactory)
        3. Apply the yaml patch if any
        4. Finalize the config (Do type conversion and logging)
        Add those necessary args to allow users to use args standalone without yaml file.
        TODO: To refactor this part to unify the final config

        Args:
            serving_mode: The serving mode of the task.
            model_path: The name of the model.
            system_name: The name of the system.
            decode_system_name: The name of the decode system.
            backend_name: The name of the backend.
            backend_version: The version of the backend.
            isl: The input sequence length.
            osl: The output sequence length.
            ttft: The target TTFT.
            tpot: The target TPOT.
            request_latency: The target end-to-end request latency.
            enable_wideep: Whether to enable wideep.
            total_gpus: The total number of GPUs.
            profiles: The profiles to use.
            yaml_config: The YAML configuration.
        """
        self.serving_mode = serving_mode
        self.model_path = model_path
        self.system_name = system_name
        self.decode_system_name = decode_system_name
        self.backend_name = backend_name
        self.backend_version = backend_version
        yaml_mode = "patch"
        yaml_patch: dict = {}
        effective_profiles: list[str] = list(profiles or [])

        if yaml_config is not None:
            logger.info(
                "Task %s: Overwriting config from YAML: %s",
                f"{serving_mode}_{model_path}",
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
            model_path=model_path,
            model_family=get_model_family(model_path),
            system_name=system_name,
            decode_system_name=decode_system_name,
            backend_name=backend_name,
            backend_version=backend_version,
            isl=isl,
            osl=osl,
            prefix=prefix,
            ttft=ttft,
            tpot=tpot,
            request_latency=request_latency,
            enable_wideep=enable_wideep,
            total_gpus=total_gpus,
            profiles=effective_profiles,
            yaml_patch=yaml_patch,
            yaml_mode=yaml_mode,
        )

        self.config, applied_layers = TaskConfigFactory.create(ctx)
        self.config.applied_layers = applied_layers
        self.config.database_mode = database_mode  # Store in config for TaskRunner access

        self.serving_mode = serving_mode
        self.model_path = model_path
        self.system_name = system_name
        self.decode_system_name = decode_system_name
        self.backend_name = backend_name
        self.enable_wideep = enable_wideep
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
            (
                f"{serving_mode}_{model_path}_{system_name}_{decode_system_name}_{backend_name}_{effective_backend_version}_{isl}_{osl}_{prefix}_{ttft}_{tpot}"
            )
            if serving_mode == "disagg"
            else (
                f"{serving_mode}_{model_path}_{system_name}_{backend_name}_{effective_backend_version}_{isl}_{osl}_{prefix}_{ttft}_{tpot}"
            )
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
            logger.info(
                "Task %s: Prefill worker config: %s",
                self.task_name,
                self.config.prefill_worker_config,
            )
            logger.info(
                "Task %s: Decode worker config: %s",
                self.task_name,
                self.config.decode_worker_config,
            )
            logger.info("Task %s: Replica config: %s", self.task_name, self.config.replica_config)
            logger.info(
                "Task %s: Advanced tuning config: %s",
                self.task_name,
                self.config.advanced_tuning_config,
            )
        else:
            raise ValueError(f"Invalid serving mode: {serving_mode}")

        self.validate()

    def validate(self):
        """
        Check that the task can be run by AIC.
        """

        # TODO: add more support matrix based validation
        if self.backend_name == "vllm" and get_model_family(self.model_path) == "DEEPSEEK":
            raise NotImplementedError("AIConfigurator does not yet support DEEPSEEK models for VLLM backend.")

        # fp8_static GEMM mode is currently TRTLLM-only.
        def _validate_fp8_static(worker_cfg: DefaultMunch, target: str) -> None:
            gemm_quant_mode = worker_cfg.get("gemm_quant_mode", None)
            if gemm_quant_mode is None:
                return
            mode_name = gemm_quant_mode.name if hasattr(gemm_quant_mode, "name") else str(gemm_quant_mode)
            if str(mode_name).lower() != common.GEMMQuantMode.fp8_static.name:
                return

            backend_name = worker_cfg.get("backend_name", None)
            if str(backend_name).lower() != common.BackendName.trtllm.value:
                raise ValueError(
                    f"fp8_static is currently only supported in trtllm backend. we got backend='{backend_name}'."
                )

        if self.serving_mode == "agg":
            _validate_fp8_static(self.config.worker_config, "worker_config")
        elif self.serving_mode == "disagg":
            _validate_fp8_static(self.config.prefill_worker_config, "prefill_worker_config")
            _validate_fp8_static(self.config.decode_worker_config, "decode_worker_config")

        # Validate requested quant modes against available perf data early, to avoid
        # late interpolation/assert failures and to provide actionable guidance.
        try:
            database = get_database(system=self.system_name, backend=self.backend_name, version=self.backend_version)
        except Exception:
            # If database can't be loaded at all, let downstream handle/report it.
            return

        supported = getattr(database, "supported_quant_mode", {}) or {}

        def _supported_or_raise(op: str, mode_name: str | None) -> None:
            if mode_name is None:
                return
            supported_modes = supported.get(op, []) or []
            if supported_modes and mode_name not in supported_modes:
                raise ValueError(
                    f"Unsupported {op} quant mode '{mode_name}' for system='{self.system_name}', "
                    f"backend='{self.backend_name}', version='{self.backend_version}'. "
                    f"Supported {op} modes: {sorted(supported_modes)}"
                )

        def _to_name(value: object) -> str | None:
            if value is None:
                return None
            return value.name if hasattr(value, "name") else str(value)

        def _get_cfg_value(cfg: object, key: str) -> object:
            if isinstance(cfg, Mapping):
                return cfg.get(key, None)
            return getattr(cfg, key, None)

        model_info = {}
        try:
            model_info = get_model_config_from_model_path(self.model_path) or {}
        except Exception:
            model_info = {}
        model_raw_config = model_info.get("raw_config")
        model_architecture = model_info.get("architecture")

        def _resolve_model_quant_modes(worker_cfg: object) -> dict[str, str | None]:
            if model_raw_config is None:
                return {}
            model_config = config.ModelConfig(
                gemm_quant_mode=_get_cfg_value(worker_cfg, "gemm_quant_mode"),
                moe_quant_mode=_get_cfg_value(worker_cfg, "moe_quant_mode"),
                kvcache_quant_mode=_get_cfg_value(worker_cfg, "kvcache_quant_mode"),
                fmha_quant_mode=_get_cfg_value(worker_cfg, "fmha_quant_mode"),
                comm_quant_mode=_get_cfg_value(worker_cfg, "comm_quant_mode"),
            )
            try:
                _apply_model_quant_defaults(
                    model_config,
                    model_raw_config,
                    model_architecture,
                    self.backend_name,
                )
            except Exception:
                return {}
            return {
                "gemm_quant_mode": _to_name(model_config.gemm_quant_mode),
                "moe_quant_mode": _to_name(model_config.moe_quant_mode),
                "kvcache_quant_mode": _to_name(model_config.kvcache_quant_mode),
                "fmha_quant_mode": _to_name(model_config.fmha_quant_mode),
                "comm_quant_mode": _to_name(model_config.comm_quant_mode),
            }

        is_deepseek = get_model_family(self.model_path) == "DEEPSEEK"
        enable_wideep = bool(getattr(self.config, "enable_wideep", self.enable_wideep))
        moe_backend = getattr(self.config, "moe_backend", None)

        # DeepSeek uses MLA perf tables; others use attention perf tables.
        if is_deepseek:
            if self.backend_name == "sglang" and enable_wideep:
                context_attn_key = "wideep_context_mla"
                generation_attn_key = "wideep_generation_mla"
            else:
                context_attn_key = "context_mla"
                generation_attn_key = "generation_mla"
        else:
            context_attn_key = "context_attention"
            generation_attn_key = "generation_attention"

        def _validate_worker_config(wc: object, *, validate_context: bool, validate_generation: bool) -> None:
            model_modes = _resolve_model_quant_modes(wc)
            gemm_mode = model_modes.get("gemm_quant_mode") or _to_name(_get_cfg_value(wc, "gemm_quant_mode"))
            _supported_or_raise("gemm", gemm_mode)

            moe_mode = model_modes.get("moe_quant_mode") or _to_name(_get_cfg_value(wc, "moe_quant_mode"))
            if self.backend_name == "sglang" and enable_wideep and moe_backend == "deepep_moe":
                if validate_context:
                    _supported_or_raise("wideep_context_moe", moe_mode)
                if validate_generation:
                    _supported_or_raise("wideep_generation_moe", moe_mode)
            else:
                _supported_or_raise("moe", moe_mode)

            if validate_context:
                fmha_mode = model_modes.get("fmha_quant_mode") or _to_name(_get_cfg_value(wc, "fmha_quant_mode"))
                _supported_or_raise(context_attn_key, fmha_mode)

            if validate_generation:
                kvcache_mode = model_modes.get("kvcache_quant_mode") or _to_name(
                    _get_cfg_value(wc, "kvcache_quant_mode")
                )
                _supported_or_raise(generation_attn_key, kvcache_mode)

        # agg/disagg worker configs use the same field names
        if self.config.serving_mode == "agg":
            _validate_worker_config(self.config.worker_config, validate_context=True, validate_generation=True)
        elif self.config.serving_mode == "disagg":
            _validate_worker_config(self.config.prefill_worker_config, validate_context=True, validate_generation=False)
            _validate_worker_config(self.config.decode_worker_config, validate_context=False, validate_generation=True)

    def pretty(self) -> str:
        def _convert(obj: Any) -> Any:
            if isinstance(obj, DefaultMunch):
                return {key: _convert(value) for key, value in obj.items()}
            if isinstance(obj, list):
                return [_convert(item) for item in obj]
            if isinstance(obj, tuple):
                return tuple(_convert(item) for item in obj)
            if hasattr(obj, "name"):
                return obj.name
            return obj

        printable: dict[str, Any] = {
            "mode": self.yaml_mode,
            "serving_mode": self.serving_mode,
            "model_path": self.model_path,
            "total_gpus": self.total_gpus,
            "system_name": self.system_name,
        }

        if self.config.serving_mode == "disagg":
            printable["decode_system_name"] = self.decode_system_name

        printable["backend_name"] = self.backend_name
        printable["backend_version"] = self.backend_version

        runtime_dict = _convert(self.config.runtime_config)
        printable.update(
            {
                k: runtime_dict.get(k)
                for k in ("isl", "osl", "prefix", "ttft", "tpot", "request_latency")
                if runtime_dict.get(k) is not None
            }
        )

        printable["enable_wideep"] = self.enable_wideep
        printable["moe_backend"] = self.config.moe_backend
        printable["attention_backend"] = self.config.attention_backend

        base_config = _convert(getattr(self.config, "yaml_patch", getattr(self, "yaml_patch", {})))
        printable["profiles"] = self.profiles

        def _ensure_dict(target: dict[str, Any], key: str) -> dict[str, Any]:
            value = target.setdefault(key, {})
            if not isinstance(value, dict):
                raise TypeError(f"Expected dict for config['{key}'], got {type(value)}")
            return value

        config_section: dict[str, Any] = dict(base_config) if isinstance(base_config, dict) else {}

        if getattr(self.config, "nextn", None) is not None:
            config_section.setdefault("nextn", self.config.nextn)
        if getattr(self.config, "nextn_accept_rates", None) is not None:
            config_section.setdefault("nextn_accept_rates", self.config.nextn_accept_rates)

        if self.config.serving_mode == "agg" and hasattr(self.config, "worker_config"):
            wc = _convert(self.config.worker_config)
            _ensure_dict(config_section, "worker_config").update(wc)
        elif self.config.serving_mode == "disagg":
            for key in (
                "prefill_worker_config",
                "decode_worker_config",
                "replica_config",
                "advanced_tuning_config",
            ):
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

    def _convert_worker_config_to_enum(self, worker_config: dict | DefaultMunch) -> None:
        """Convert string quant mode values to enums, skip if already converted."""
        worker_cfg = _ensure_munch(worker_config)

        # Ensure missing quant mode keys resolve to None instead of DefaultMunch.
        for key in (
            "gemm_quant_mode",
            "moe_quant_mode",
            "kvcache_quant_mode",
            "fmha_quant_mode",
            "comm_quant_mode",
        ):
            worker_cfg.setdefault(key, None)

        # Only convert if the value is a string
        gemm_quant_mode = worker_cfg.get("gemm_quant_mode", None)
        if isinstance(gemm_quant_mode, str):
            worker_cfg["gemm_quant_mode"] = common.GEMMQuantMode[gemm_quant_mode]

        moe_quant_mode = worker_cfg.get("moe_quant_mode", None)
        if isinstance(moe_quant_mode, str):
            worker_cfg["moe_quant_mode"] = common.MoEQuantMode[moe_quant_mode]

        kvcache_quant_mode = worker_cfg.get("kvcache_quant_mode", None)
        if isinstance(kvcache_quant_mode, str):
            worker_cfg["kvcache_quant_mode"] = common.KVCacheQuantMode[kvcache_quant_mode]

        fmha_quant_mode = worker_cfg.get("fmha_quant_mode", None)
        if isinstance(fmha_quant_mode, str):
            worker_cfg["fmha_quant_mode"] = common.FMHAQuantMode[fmha_quant_mode]

        comm_quant_mode = worker_cfg.get("comm_quant_mode", None)
        if isinstance(comm_quant_mode, str):
            worker_cfg["comm_quant_mode"] = common.CommQuantMode[comm_quant_mode]

        worker_config.update(worker_cfg)


class TaskRunner:
    @staticmethod
    def _get_database(system: str, backend: str, version: str, database_mode: str | None = None):
        """Fetch a database from the global cache.

        When *database_mode* is set the returned object is a deep copy (because
        `set_default_database_mode` mutates state).  Otherwise the cached
        instance is returned directly.
        """
        db = get_database(system=system, backend=backend, version=version)
        if database_mode is not None:
            db = copy.deepcopy(db)
            db.set_default_database_mode(common.DatabaseMode[database_mode])
        return db

    def run_agg(self, task_config: DefaultMunch) -> dict[str, pd.DataFrame | None]:
        logger.info("Task %s: Setting up runtime config", task_config.task_name)
        runtime_config = config.RuntimeConfig(
            isl=task_config.runtime_config.isl,
            osl=task_config.runtime_config.osl,
            prefix=task_config.runtime_config.prefix,
            ttft=task_config.runtime_config.ttft,
            tpot=list(range(1, 20, 1)) + list(range(20, 300, 5)),
            request_latency=getattr(task_config.runtime_config, "request_latency", None),
        )
        logger.info("Task %s: Setting up database", task_config.task_name)
        try:
            database_mode = getattr(task_config, "database_mode", None)
            database = self._get_database(
                system=task_config.worker_config.system_name,
                backend=task_config.worker_config.backend_name,
                version=task_config.worker_config.backend_version,
                database_mode=database_mode,
            )
            if database_mode is not None:
                logger.info("Task %s: Using database mode: %s", task_config.task_name, database_mode)
        except Exception:  # pragma: no cover
            logger.exception(
                "Error getting database for %s %s %s",
                task_config.worker_config.system_name,
                task_config.worker_config.backend_name,
                task_config.worker_config.backend_version,
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
            moe_backend=task_config.moe_backend,  # sglang wideep only
            attention_backend=task_config.attention_backend,  # sglang wideep only
            enable_wideep=task_config.enable_wideep,
        )
        logger.info("Task %s: Enumerating parallel config", task_config.task_name)
        try:
            from aiconfigurator.sdk import pareto_analysis as pa

            parallel_config_list = enumerate_parallel_config(
                num_gpu_list=task_config.worker_config.num_gpu_per_worker,
                tp_list=task_config.worker_config.tp_list,
                pp_list=task_config.worker_config.pp_list,
                dp_list=task_config.worker_config.dp_list,
                moe_tp_list=task_config.worker_config.moe_tp_list,
                moe_ep_list=task_config.worker_config.moe_ep_list,
                is_moe=check_is_moe(task_config.model_path),
                backend=common.BackendName(task_config.worker_config.backend_name),
                enable_wideep=task_config.enable_wideep,
            )
        except Exception:  # pragma: no cover
            logger.exception(
                "Error enumerating parallel config for %s %s %s",
                task_config.worker_config.system_name,
                task_config.worker_config.backend_name,
                task_config.worker_config.backend_version,
            )
            return None
        logger.info("Task %s: Running agg pareto", task_config.task_name)
        result_df = pa.agg_pareto(
            model_path=task_config.model_path,
            runtime_config=runtime_config,
            database=database,
            backend_name=task_config.worker_config.backend_name,
            model_config=model_config,
            parallel_config_list=parallel_config_list,
        )
        return {
            "pareto_df": result_df,
        }

    def run_disagg(self, task_config: DefaultMunch, autoscale: bool = False) -> dict[str, pd.DataFrame | None]:
        logger.info("Task %s: Setting up runtime config", task_config.task_name)
        runtime_config = config.RuntimeConfig(
            isl=task_config.runtime_config.isl,
            osl=task_config.runtime_config.osl,
            prefix=task_config.runtime_config.prefix,
            ttft=task_config.runtime_config.ttft,
            tpot=list(range(1, 20, 1)) + list(range(20, 300, 5)),
            request_latency=getattr(task_config.runtime_config, "request_latency", None),
        )

        # Get database mode from config
        database_mode = getattr(task_config, "database_mode", None)

        logger.info("Task %s: Setting up prefill database", task_config.task_name)
        try:
            prefill_database = self._get_database(
                system=task_config.prefill_worker_config.system_name,
                backend=task_config.prefill_worker_config.backend_name,
                version=task_config.prefill_worker_config.backend_version,
                database_mode=database_mode,
            )
            if database_mode is not None:
                logger.info("Task %s: Using prefill database mode: %s", task_config.task_name, database_mode)
        except Exception:  # pragma: no cover
            logger.exception(
                "Error getting prefill database for %s %s %s",
                task_config.prefill_worker_config.system_name,
                task_config.prefill_worker_config.backend_name,
                task_config.prefill_worker_config.backend_version,
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
            moe_backend=task_config.moe_backend,  # sglang wideep only
            attention_backend=task_config.attention_backend,  # sglang wideep only
            enable_wideep=task_config.enable_wideep,
        )

        logger.info("Task %s: Enumerating prefill parallel config", task_config.task_name)
        try:
            from aiconfigurator.sdk import pareto_analysis as pa

            prefill_parallel_config_list = enumerate_parallel_config(
                num_gpu_list=task_config.prefill_worker_config.num_gpu_per_worker,
                tp_list=task_config.prefill_worker_config.tp_list,
                pp_list=task_config.prefill_worker_config.pp_list,
                dp_list=task_config.prefill_worker_config.dp_list,
                moe_tp_list=task_config.prefill_worker_config.moe_tp_list,
                moe_ep_list=task_config.prefill_worker_config.moe_ep_list,
                is_moe=check_is_moe(task_config.model_path),
                backend=common.BackendName(task_config.prefill_worker_config.backend_name),
                enable_wideep=task_config.enable_wideep,
            )
        except Exception:  # pragma: no cover
            logger.exception(
                "Error enumerating prefill parallel config for %s %s %s",
                task_config.prefill_worker_config.system_name,
                task_config.prefill_worker_config.backend_name,
                task_config.prefill_worker_config.backend_version,
            )
            return None

        logger.info("Task %s: Setting up decode database", task_config.task_name)
        try:
            decode_database = self._get_database(
                system=task_config.decode_worker_config.system_name,
                backend=task_config.decode_worker_config.backend_name,
                version=task_config.decode_worker_config.backend_version,
                database_mode=database_mode,
            )
            if database_mode is not None:
                logger.info("Task %s: Using decode database mode: %s", task_config.task_name, database_mode)
        except Exception:  # pragma: no cover
            logger.exception(
                "Error getting decode database for %s %s %s",
                task_config.decode_worker_config.system_name,
                task_config.decode_worker_config.backend_name,
                task_config.decode_worker_config.backend_version,
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
            moe_backend=task_config.moe_backend,  # sglang wideep only
            attention_backend=task_config.attention_backend,  # sglang wideep only
            enable_wideep=task_config.enable_wideep,
        )

        logger.info("Task %s: Enumerating decode parallel config", task_config.task_name)
        try:
            from aiconfigurator.sdk import pareto_analysis as pa

            decode_parallel_config_list = enumerate_parallel_config(
                num_gpu_list=task_config.decode_worker_config.num_gpu_per_worker,
                tp_list=task_config.decode_worker_config.tp_list,
                pp_list=task_config.decode_worker_config.pp_list,
                dp_list=task_config.decode_worker_config.dp_list,
                moe_tp_list=task_config.decode_worker_config.moe_tp_list,
                moe_ep_list=task_config.decode_worker_config.moe_ep_list,
                is_moe=check_is_moe(task_config.model_path),
                backend=common.BackendName(task_config.decode_worker_config.backend_name),
                enable_wideep=task_config.enable_wideep,
            )
        except Exception:  # pragma: no cover
            logger.exception(
                "Error enumerating decode parallel config for %s %s %s",
                task_config.decode_worker_config.system_name,
                task_config.decode_worker_config.backend_name,
                task_config.decode_worker_config.backend_version,
            )
            return None

        # For SGLang non-wideep disaggregated serving
        # See: https://github.com/ai-dynamo/dynamo/issues/5870
        backend_name = str(task_config.prefill_worker_config.backend_name)
        enable_wideep = bool(getattr(task_config, "enable_wideep", False))
        require_same_tp = backend_name == "sglang" and not enable_wideep

        if require_same_tp:
            logger.warning(
                "SGLang non-wideep disaggregated serving requires the same TP size "
                "for prefill and decode workers. Configurations with different TP "
                "sizes will be filtered out. "
            )

        logger.info("Task %s: Running disagg pareto", task_config.task_name)
        result_df = pa.disagg_pareto(
            model_path=task_config.model_path,
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
            prefill_max_num_tokens=task_config.advanced_tuning_config.prefill_max_batch_size
            * task_config.runtime_config.isl,
            decode_max_num_tokens=task_config.advanced_tuning_config.decode_max_batch_size,
            prefill_latency_correction_scale=task_config.advanced_tuning_config.prefill_latency_correction_scale,
            decode_latency_correction_scale=task_config.advanced_tuning_config.decode_latency_correction_scale,
            require_same_tp=require_same_tp,
            autoscale=autoscale,
            target_tpot=task_config.runtime_config.tpot if autoscale else None,
        )
        return {"pareto_df": result_df}

    def run(
        self,
        task_config: TaskConfig,
        autoscale: bool = False,
    ) -> dict[str, pd.DataFrame | None]:
        serving_mode = task_config.config.serving_mode
        logger.info(
            "Starting Pareto Analysis for %s in %s mode (autoscale=%s)...",
            task_config.task_name,
            serving_mode,
            autoscale,
        )
        try:
            if serving_mode == "agg":
                if autoscale:
                    raise ValueError("autoscale mode is only supported for disagg serving mode.")
                result = self.run_agg(task_config.config)
            elif serving_mode == "disagg":
                result = self.run_disagg(task_config.config, autoscale=autoscale)
            else:
                raise ValueError(f"Invalid serving mode: {serving_mode}")
        except Exception:
            logger.exception(
                "Error running pareto analysis for %s in %s mode",
                task_config.task_name,
                serving_mode,
            )
            result = None
            raise

        if result is None:
            logger.warning("No result found for %s in %s mode.", task_config.task_name, serving_mode)

        return result


if __name__ == "__main__":
    task_agg = TaskConfig(
        serving_mode="agg",
        model_path="QWEN3_32B",
        system_name="h200_sxm",
        ttft=600,
        tpot=20,
        isl=4000,
        osl=500,
        prefix=0,
        total_gpus=8,
    )
    task_runner = TaskRunner()
    print("\n=== TaskConfig (agg) ===")
    print(task_agg.pretty())
    agg_df = task_runner.run(task_agg)["pareto_df"]
    agg_df = get_pareto_front(agg_df, "tokens/s/user", "tokens/s/gpu").reset_index(drop=True).reset_index()
    agg_df.to_csv("agg_df.csv", index=False)
    print("\n=== agg pareto ===")
    print(agg_df)

    task_disagg = TaskConfig(
        serving_mode="disagg",
        model_path="QWEN3_32B",
        system_name="h200_sxm",
        ttft=600,
        tpot=20,
        isl=4000,
        osl=500,
        prefix=0,
        total_gpus=16,
        profiles=["fp8"],
        yaml_config={
            "mode": "patch",
            "config": {
                "advanced_tuning_config": {
                    "prefill_latency_correction_scale": 1.1,
                    "decode_latency_correction_scale": 1.08,
                },
            },
        },
    )
    print("\n=== TaskConfig (disagg) ===")
    print(task_disagg.pretty())
    disagg_df = task_runner.run(task_disagg)["pareto_df"]
    disagg_df = get_pareto_front(disagg_df, "tokens/s/user", "tokens/s/gpu").reset_index(drop=True).reset_index()
    disagg_df.to_csv("disagg_df.csv", index=False)
    print("\n=== disagg pareto ===")
    print(disagg_df)
