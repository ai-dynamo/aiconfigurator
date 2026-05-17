# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TaskConfig — flat user-facing config for sweep_agg / sweep_disagg.

Replaces the legacy ``sdk.task.TaskConfig`` (V1) and the orphaned
``sdk.task_v2.TaskConfig`` (V2).  The legacy YAML format is NOT supported;
new YAML uses field names that map 1:1 to this dataclass.

Design:
- Flat dataclass, SGLang-style.  No nested DefaultMunch, no deep_merge.
- ``__post_init__`` resolves model identity, backend version, quant modes,
  search candidates.  After construction, every active field has a
  concrete value.
- Strict prefix discipline: in disagg mode, top-level worker-spec fields
  (model_path, system_name, backend_name, quant_*, enable_wideep, ...)
  are not used and setting them raises ValueError.  Use prefill_* /
  decode_* fields explicitly.
- ``from_yaml`` is a thin pass-through: YAML keys must equal field names.
- ``sweep_agg_kwargs()`` / ``sweep_disagg_kwargs()`` build the exact
  kwargs needed by :mod:`aiconfigurator.sdk.sweep` — no caller
  marshalling required.

See ``cli/exps/example_new.yaml`` for the canonical YAML format.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.models import (
    _infer_quant_modes_from_raw_config,
    check_is_moe,
    get_model_family,
)
from aiconfigurator.sdk.perf_database import get_latest_database_version
from aiconfigurator.sdk.utils import enumerate_parallel_config, get_model_config_from_model_path

logger = logging.getLogger(__name__)

ParallelChoice = tuple[int, int, int, int, int]  # (tp, pp, dp, moe_tp, moe_ep)


_DEFAULT_NEXTN_ACCEPT_RATES: list[float] = [0.85, 0.8, 0.6, 0.0, 0.0]

QUANT_PRESETS: dict[str, dict[str, str]] = {
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
    "bfloat16": {
        "gemm_quant_mode": "bfloat16",
        "moe_quant_mode": "bfloat16",
        "kvcache_quant_mode": "bfloat16",
        "fmha_quant_mode": "bfloat16",
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
        "gemm_quant_mode": "bfloat16",
        "moe_quant_mode": "w4a16_mxfp4",
        "kvcache_quant_mode": "bfloat16",
        "fmha_quant_mode": "bfloat16",
        "comm_quant_mode": "half",
    },
}

_QUANT_ENUM_TABLES: dict[str, type] = {
    "gemm_quant_mode": common.GEMMQuantMode,
    "moe_quant_mode": common.MoEQuantMode,
    "kvcache_quant_mode": common.KVCacheQuantMode,
    "fmha_quant_mode": common.FMHAQuantMode,
    "comm_quant_mode": common.CommQuantMode,
}

_QUANT_FALLBACKS: dict[str, object] = {
    "gemm_quant_mode": common.GEMMQuantMode.bfloat16,
    "moe_quant_mode": common.MoEQuantMode.bfloat16,
    "kvcache_quant_mode": common.KVCacheQuantMode.bfloat16,
    "fmha_quant_mode": common.FMHAQuantMode.bfloat16,
    "comm_quant_mode": common.CommQuantMode.half,
}


def _resolve_quant_str(key: str, value: Any) -> Any:
    # Accept role-prefixed keys (e.g. "prefill_gemm_quant_mode") by stripping
    # the prefix before looking up the enum table.
    bare = key
    for role in ("prefill_", "decode_"):
        if bare.startswith(role):
            bare = bare[len(role) :]
            break
    enum_cls = _QUANT_ENUM_TABLES.get(bare)
    if enum_cls is not None and isinstance(value, str):
        return enum_cls[value]
    return value


# ---------------------------------------------------------------------------
# Default disagg search space (mirror of legacy build_disagg_parallel_lists)
# ---------------------------------------------------------------------------


def _default_disagg_search(
    *,
    backend_name: str,
    is_moe: bool,
    prefill_system: str,
    decode_system: str,
    prefill_enable_wideep: bool,
    decode_enable_wideep: bool,
    moe_backend: str | None,
    should_enable_pp: bool = False,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Inlined version of legacy sdk.task.build_disagg_parallel_lists.

    Kept here so task_config.py does not depend on V1 task.py.  Algorithm
    identical; locked by integration parity test.
    """
    prefill_cfg: dict[str, list[int]] = {
        "num_gpu_per_worker": [1, 2, 4, 8],
        "tp_list": [1, 2, 4, 8],
        "pp_list": [1, 2, 4, 8] if should_enable_pp else [1],
        "dp_list": [1],
        "moe_tp_list": [1],
        "moe_ep_list": [1, 2, 4, 8] if is_moe else [1],
    }
    decode_cfg: dict[str, list[int]] = {
        "num_gpu_per_worker": [1, 2, 4, 8],
        "tp_list": [1, 2, 4, 8],
        "pp_list": [1, 2, 4, 8] if should_enable_pp else [1],
        "dp_list": [1, 2, 4, 8] if is_moe else [1],
        "moe_tp_list": [1],
        "moe_ep_list": [1, 2, 4, 8] if is_moe else [1],
    }
    if not is_moe:
        if prefill_system in ("gb200", "gb300"):
            prefill_cfg["num_gpu_per_worker"] = [1, 2, 4, 8, 16]
            prefill_cfg["tp_list"] = [1, 2, 4, 8, 16]
            prefill_cfg["pp_list"] = [1]
        if decode_system in ("gb200", "gb300"):
            decode_cfg["num_gpu_per_worker"] = [1, 2, 4, 8, 16]
            decode_cfg["tp_list"] = [1, 2, 4, 8, 16]
            decode_cfg["pp_list"] = [1]
        return prefill_cfg, decode_cfg

    if backend_name == "trtllm":
        if prefill_enable_wideep:
            prefill_cfg = {
                "num_gpu_per_worker": [4, 8, 16, 32],
                "tp_list": [1, 2, 4, 8],
                "pp_list": [1, 2, 4, 8, 16, 32] if should_enable_pp else [1],
                "dp_list": [4, 8, 16, 32],
                "moe_tp_list": [1],
                "moe_ep_list": [4, 8, 16, 32],
            }
        else:
            x = [1, 2, 4, 8]
            prefill_cfg = {
                "num_gpu_per_worker": x,
                "tp_list": x,
                "pp_list": x if should_enable_pp else [1],
                "dp_list": x,
                "moe_tp_list": x,
                "moe_ep_list": x,
            }
        if decode_enable_wideep:
            decode_cfg = {
                "num_gpu_per_worker": [4, 8, 16, 32, 64],
                "tp_list": [1, 2, 4, 8],
                "pp_list": [1, 2, 4, 8, 16, 32, 64] if should_enable_pp else [1],
                "dp_list": [4, 8, 16, 32, 64],
                "moe_tp_list": [1],
                "moe_ep_list": [4, 8, 16, 32, 64],
            }
        else:
            x = [1, 2, 4, 8]
            decode_cfg = {
                "num_gpu_per_worker": x,
                "tp_list": x,
                "pp_list": x if should_enable_pp else [1],
                "dp_list": x,
                "moe_tp_list": x,
                "moe_ep_list": x,
            }
    elif backend_name == "sglang":
        if prefill_enable_wideep or decode_enable_wideep:
            prefill_cfg = {
                "num_gpu_per_worker": [8, 16, 32],
                "tp_list": [1, 2, 4, 8],
                "pp_list": [1, 2, 4, 8, 16, 32] if should_enable_pp else [1],
                "dp_list": [1, 2, 4, 8, 16, 32],
                "moe_tp_list": [1],
                "moe_ep_list": [8, 16, 32],
            }
            decode_cfg = {
                "num_gpu_per_worker": [8, 16, 32, 64],
                "tp_list": [1, 2, 4, 8],
                "pp_list": [1, 2, 4, 8, 16, 32, 64] if should_enable_pp else [1],
                "dp_list": [1, 2, 4, 8, 16, 32, 64],
                "moe_tp_list": [1],
                "moe_ep_list": [8, 16, 32, 64],
            }
        elif moe_backend == "deepep_moe":
            x = [1, 2, 4, 8]
            for cfg in (prefill_cfg, decode_cfg):
                cfg["num_gpu_per_worker"] = x
                cfg["tp_list"] = x
                cfg["pp_list"] = x if should_enable_pp else [1]
                cfg["dp_list"] = x
                cfg["moe_tp_list"] = [1]
                cfg["moe_ep_list"] = [1, 2, 4, 8]
        else:
            x = [1, 2, 4, 8]
            prefill_cfg = {
                "num_gpu_per_worker": x,
                "tp_list": x,
                "pp_list": x if should_enable_pp else [1],
                "dp_list": x,
                "moe_tp_list": x,
                "moe_ep_list": [1, 2, 4, 8],
            }
            decode_cfg = {
                "num_gpu_per_worker": x,
                "tp_list": x,
                "pp_list": x if should_enable_pp else [1],
                "dp_list": x,
                "moe_tp_list": x,
                "moe_ep_list": [1, 2, 4, 8],
            }
    elif backend_name == "vllm":
        x = [1, 2, 4, 8]
        prefill_cfg = {
            "num_gpu_per_worker": x,
            "tp_list": x,
            "pp_list": x if should_enable_pp else [1],
            "dp_list": x,
            "moe_tp_list": x,
            "moe_ep_list": x,
        }
        decode_cfg = copy.deepcopy(prefill_cfg)
    else:
        raise ValueError(f"Invalid backend: {backend_name}")

    return prefill_cfg, decode_cfg


# ---------------------------------------------------------------------------
# TaskConfig
# ---------------------------------------------------------------------------


@dataclass
class TaskConfig:
    """Flat user-facing task config.  See module docstring."""

    # ====== 1. Mode + workload ======
    serving_mode: Literal["agg", "disagg"] = "agg"
    isl: int = 4000
    osl: int = 1000
    prefix: int = 0
    ttft: float = 1000.0
    tpot: float = 50.0
    request_latency: float | None = None
    total_gpus: int | None = None
    database_mode: str | None = None
    free_gpu_memory_fraction: float | None = None
    max_seq_len: int | None = None
    engine_step_backend: str | None = None

    # ====== 2. Agg worker spec (serving_mode='agg') ======
    model_path: str = ""
    system_name: str = ""
    backend_name: str = "trtllm"
    backend_version: str | None = None
    enable_wideep: bool = False
    enable_chunked_prefill: bool = False
    enable_eplb: bool = False
    nextn: int | None = None
    nextn_accept_rates: list[float] = field(default_factory=lambda: list(_DEFAULT_NEXTN_ACCEPT_RATES))
    moe_backend: str | None = None
    quant_preset: str | None = None
    gemm_quant_mode: common.GEMMQuantMode | None = None
    moe_quant_mode: common.MoEQuantMode | None = None
    kvcache_quant_mode: common.KVCacheQuantMode | None = None
    fmha_quant_mode: common.FMHAQuantMode | None = None
    comm_quant_mode: common.CommQuantMode | None = None

    # ====== 3. Agg search space ======
    agg_num_gpu_candidates: list[int] | None = None
    agg_tp_candidates: list[int] | None = None
    agg_pp_candidates: list[int] | None = None
    agg_dp_candidates: list[int] | None = None
    agg_moe_tp_candidates: list[int] | None = None
    agg_moe_ep_candidates: list[int] | None = None

    # ====== 4. Disagg prefill worker spec ======
    prefill_model_path: str = ""
    prefill_system_name: str = ""
    prefill_backend_name: str = "trtllm"
    prefill_backend_version: str | None = None
    prefill_enable_wideep: bool = False
    prefill_enable_chunked_prefill: bool = False
    prefill_enable_eplb: bool = False
    prefill_quant_preset: str | None = None
    prefill_gemm_quant_mode: common.GEMMQuantMode | None = None
    prefill_moe_quant_mode: common.MoEQuantMode | None = None
    prefill_kvcache_quant_mode: common.KVCacheQuantMode | None = None
    prefill_fmha_quant_mode: common.FMHAQuantMode | None = None
    prefill_comm_quant_mode: common.CommQuantMode | None = None

    # ====== 5. Disagg prefill search space ======
    prefill_num_gpu_candidates: list[int] | None = None
    prefill_tp_candidates: list[int] | None = None
    prefill_pp_candidates: list[int] | None = None
    prefill_dp_candidates: list[int] | None = None
    prefill_moe_tp_candidates: list[int] | None = None
    prefill_moe_ep_candidates: list[int] | None = None

    # ====== 6. Disagg decode worker spec ======
    decode_model_path: str = ""
    decode_system_name: str = ""
    decode_backend_name: str = "trtllm"
    decode_backend_version: str | None = None
    decode_enable_wideep: bool = False
    decode_enable_chunked_prefill: bool = False
    decode_enable_eplb: bool = False
    decode_quant_preset: str | None = None
    decode_gemm_quant_mode: common.GEMMQuantMode | None = None
    decode_moe_quant_mode: common.MoEQuantMode | None = None
    decode_kvcache_quant_mode: common.KVCacheQuantMode | None = None
    decode_fmha_quant_mode: common.FMHAQuantMode | None = None
    decode_comm_quant_mode: common.CommQuantMode | None = None

    # ====== 7. Disagg decode search space ======
    decode_num_gpu_candidates: list[int] | None = None
    decode_tp_candidates: list[int] | None = None
    decode_pp_candidates: list[int] | None = None
    decode_dp_candidates: list[int] | None = None
    decode_moe_tp_candidates: list[int] | None = None
    decode_moe_ep_candidates: list[int] | None = None

    # ====== 8. Disagg orchestration ======
    num_gpu_per_replica: list[int] | None = None
    max_gpu_per_replica: int | None = None
    max_prefill_workers: int | None = None
    max_decode_workers: int | None = None
    prefill_max_batch_size: int = 1
    decode_max_batch_size: int = 512
    prefill_latency_correction: float = 1.1
    decode_latency_correction: float = 1.08

    # ====== 9. Internal — resolved in __post_init__ ======
    _is_moe: bool = field(default=False, repr=False, init=False)
    _model_family: str = field(default="", repr=False, init=False)
    _raw_config: dict = field(default_factory=dict, repr=False, init=False)
    _architecture: str = field(default="", repr=False, init=False)

    # =====================================================================
    # Construction
    # =====================================================================

    @classmethod
    def from_yaml(cls, yaml_data: dict, **overrides: Any) -> TaskConfig:
        """Construct from a flat YAML dict.

        YAML keys must match TaskConfig field names directly.  String values
        for quant_mode fields are converted to the matching enum.  Unknown
        keys are warned about but ignored.  ``overrides`` (kwargs) win over
        YAML values.
        """
        valid_keys = {f.name for f in dataclasses.fields(cls) if f.init and not f.name.startswith("_")}
        kwargs: dict[str, Any] = {}
        for k, v in yaml_data.items():
            if k not in valid_keys:
                logger.warning("from_yaml: ignoring unknown key %r", k)
                continue
            kwargs[k] = _resolve_quant_str(k, v) if k.endswith("quant_mode") else v
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kwargs)

    @classmethod
    def from_cli(cls, **kwargs: Any) -> TaskConfig:
        """Construct from CLI kwargs.  Filters None to let __post_init__ defaults run."""
        return cls(**{k: v for k, v in kwargs.items() if v is not None})

    # =====================================================================
    # __post_init__
    # =====================================================================

    def __post_init__(self) -> None:
        self._check_prefix_discipline()
        self._resolve_model_identity()
        self._resolve_backend_version()
        self._resolve_quant_modes()
        self._resolve_search_space()

    def _check_prefix_discipline(self) -> None:
        """In disagg mode, top-level worker-spec fields must be at their defaults.

        Setting top-level ``enable_wideep=True`` while serving_mode='disagg'
        is the kind of silent override that the legacy V1/V2 paths swallowed
        without warning.  Be explicit here.
        """
        if self.serving_mode != "disagg":
            return
        leakage = []
        if self.model_path:
            leakage.append("model_path")
        if self.system_name:
            leakage.append("system_name")
        # Don't flag enable_wideep=False (default), only True.
        if self.enable_wideep:
            leakage.append("enable_wideep")
        if self.enable_chunked_prefill:
            leakage.append("enable_chunked_prefill")
        if self.enable_eplb:
            leakage.append("enable_eplb")
        if self.quant_preset is not None:
            leakage.append("quant_preset")
        for q in _QUANT_ENUM_TABLES:
            if getattr(self, q) is not None:
                leakage.append(q)
        if leakage:
            raise ValueError(
                f"Disagg mode: top-level worker fields are not used and must not be set "
                f"(got {leakage}).  Use prefill_* / decode_* variants instead."
            )

    def _resolve_model_identity(self) -> None:
        primary = self.model_path if self.serving_mode == "agg" else self.prefill_model_path
        if not primary:
            return
        info = get_model_config_from_model_path(primary)
        self._raw_config = info.get("raw_config", {})
        self._architecture = info["architecture"]
        self._model_family = get_model_family(primary)
        self._is_moe = check_is_moe(primary)

        text_key = common.MULTIMODAL_TEXT_CONFIG_KEY.get(self._architecture)
        cfg = self._raw_config[text_key] if text_key and text_key in self._raw_config else self._raw_config
        hf_nextn = cfg.get("num_nextn_predict_layers", 0)
        if self.nextn is None:
            self.nextn = hf_nextn
        elif self.nextn != hf_nextn:
            logger.debug("nextn overridden: HF config=%d, using user value=%d", hf_nextn, self.nextn)

    def _resolve_backend_version(self) -> None:
        def _resolve(system: str, backend: str, current: str | None) -> str | None:
            if current is not None:
                return current
            return get_latest_database_version(system=system, backend=backend)

        if self.serving_mode == "agg":
            if self.system_name and self.backend_name:
                self.backend_version = _resolve(self.system_name, self.backend_name, self.backend_version)
        else:
            if self.prefill_system_name and self.prefill_backend_name:
                self.prefill_backend_version = _resolve(
                    self.prefill_system_name, self.prefill_backend_name, self.prefill_backend_version
                )
            if self.decode_system_name and self.decode_backend_name:
                self.decode_backend_version = _resolve(
                    self.decode_system_name, self.decode_backend_name, self.decode_backend_version
                )

    def _resolve_quant_modes(self) -> None:
        """Resolve quant modes for the active role(s).

        Priority (highest wins): explicit field > preset > HF base > bfloat16 fallback.
        """
        roles = ["agg"] if self.serving_mode == "agg" else ["prefill", "decode"]
        base = _infer_quant_modes_from_raw_config(self._raw_config)

        for role in roles:
            preset_name = self._role_attr(role, "quant_preset")
            preset_overrides: dict[str, object] = {}
            if preset_name is not None:
                preset_def = QUANT_PRESETS.get(preset_name)
                if preset_def is None:
                    logger.warning("Unknown quant_preset %r for role %s, ignoring", preset_name, role)
                else:
                    for k, v in preset_def.items():
                        preset_overrides[k] = _resolve_quant_str(k, v)

            for key in _QUANT_ENUM_TABLES:
                explicit = self._role_attr(role, key)
                from_preset = preset_overrides.get(key)
                from_hf = base.get(key)
                fallback = _QUANT_FALLBACKS[key]

                if explicit is not None:
                    continue
                resolved = from_preset if from_preset is not None else (from_hf if from_hf is not None else fallback)
                self._set_role_attr(role, key, resolved)

        # Backend / architecture fixups
        for role in roles:
            fmha = self._role_attr(role, "fmha_quant_mode")
            if (
                self._architecture in ("DeepseekV3ForCausalLM", "KimiK25ForConditionalGeneration")
                and fmha == common.FMHAQuantMode.fp8
            ):
                self._set_role_attr(role, "fmha_quant_mode", common.FMHAQuantMode.bfloat16)
            backend_name = self._role_attr(role, "backend_name")
            if backend_name == "vllm" and self._role_attr(role, "fmha_quant_mode") == common.FMHAQuantMode.fp8:
                self._set_role_attr(role, "fmha_quant_mode", common.FMHAQuantMode.bfloat16)

    def _resolve_search_space(self) -> None:
        if self.serving_mode == "agg":
            self._resolve_agg_search()
        else:
            self._resolve_disagg_search()

    def _resolve_agg_search(self) -> None:
        def _set(name: str, values: list[int]) -> None:
            if getattr(self, name) is None:
                setattr(self, name, values)

        if not self._is_moe:
            blackwell = self.system_name in ("gb200", "gb300")
            wide = [1, 2, 4, 8, 16] if blackwell else [1, 2, 4, 8]
            _set("agg_num_gpu_candidates", wide)
            _set("agg_tp_candidates", wide)
            _set("agg_pp_candidates", [1])
            _set("agg_dp_candidates", [1])
            _set("agg_moe_tp_candidates", [1])
            _set("agg_moe_ep_candidates", [1])
            return

        if self.backend_name == "trtllm" and self.enable_wideep:
            _set("agg_num_gpu_candidates", [2, 4, 8, 16, 32, 64])
            _set("agg_tp_candidates", [1, 2, 4, 8])
            _set("agg_pp_candidates", [1])
            _set("agg_dp_candidates", [2, 4, 8, 16, 32, 64])
            _set("agg_moe_tp_candidates", [1])
            _set("agg_moe_ep_candidates", [2, 4, 8, 16, 32, 64])
        elif self.backend_name == "sglang" and self.enable_wideep:
            _set("agg_num_gpu_candidates", [8, 16, 32, 64])
            _set("agg_tp_candidates", [1, 2, 4, 8])
            _set("agg_pp_candidates", [1])
            _set("agg_dp_candidates", [1, 2, 4, 8, 16, 32, 64])
            _set("agg_moe_tp_candidates", [1])
            _set("agg_moe_ep_candidates", [8, 16, 32, 64])
        elif self.backend_name == "sglang" and not self.enable_wideep:
            _set("agg_num_gpu_candidates", [1, 2, 4, 8])
            _set("agg_tp_candidates", [1, 2, 4, 8])
            _set("agg_pp_candidates", [1])
            _set("agg_dp_candidates", [1, 2, 4, 8])
            _set("agg_moe_tp_candidates", [1, 2, 4, 8])
            _set("agg_moe_ep_candidates", [1])
        elif self.backend_name in ("trtllm", "vllm"):
            x = [1, 2, 4, 8]
            _set("agg_num_gpu_candidates", x)
            _set("agg_tp_candidates", x)
            _set("agg_pp_candidates", [1])
            _set("agg_dp_candidates", x)
            _set("agg_moe_tp_candidates", x)
            _set("agg_moe_ep_candidates", x)
        else:
            raise ValueError(f"Unsupported backend: {self.backend_name}")

    def _resolve_disagg_search(self) -> None:
        prefill_cfg, decode_cfg = _default_disagg_search(
            backend_name=self.prefill_backend_name,
            is_moe=self._is_moe,
            prefill_system=self.prefill_system_name,
            decode_system=self.decode_system_name,
            prefill_enable_wideep=self.prefill_enable_wideep,
            decode_enable_wideep=self.decode_enable_wideep,
            moe_backend=self.moe_backend,
        )
        for role, src in (("prefill", prefill_cfg), ("decode", decode_cfg)):
            self._fill_role_search(role, src)

        # Replica defaults
        if self.prefill_enable_wideep or self.decode_enable_wideep:
            if self.max_gpu_per_replica is None:
                self.max_gpu_per_replica = 512
        else:
            if self.num_gpu_per_replica is None:
                self.num_gpu_per_replica = [1, 2, 4, 8] + list(range(16, 129, 8))
            if self.max_gpu_per_replica is None:
                self.max_gpu_per_replica = 128
        if self.max_prefill_workers is None:
            self.max_prefill_workers = 32
        if self.max_decode_workers is None:
            self.max_decode_workers = 32

    def _fill_role_search(self, role: str, src: dict[str, list[int]]) -> None:
        map_to_attr = {
            "num_gpu_per_worker": f"{role}_num_gpu_candidates",
            "tp_list": f"{role}_tp_candidates",
            "pp_list": f"{role}_pp_candidates",
            "dp_list": f"{role}_dp_candidates",
            "moe_tp_list": f"{role}_moe_tp_candidates",
            "moe_ep_list": f"{role}_moe_ep_candidates",
        }
        for k_src, k_attr in map_to_attr.items():
            if getattr(self, k_attr) is None:
                setattr(self, k_attr, src[k_src])

    # =====================================================================
    # Role attribute access (no fallback across prefixes — strict discipline)
    # =====================================================================

    def _role_attr(self, role: str, name: str) -> Any:
        return getattr(self, name if role == "agg" else f"{role}_{name}")

    def _set_role_attr(self, role: str, name: str, value: Any) -> None:
        setattr(self, name if role == "agg" else f"{role}_{name}", value)

    # =====================================================================
    # Builders consumed by sweep.py
    # =====================================================================

    def build_runtime_config(self, batch_size: int | None = None) -> config.RuntimeConfig:
        rt = config.RuntimeConfig(
            isl=self.isl,
            osl=self.osl,
            prefix=self.prefix,
            ttft=self.ttft,
            tpot=self.tpot,
            request_latency=self.request_latency,
            engine_step_backend=self.engine_step_backend,
        )
        if batch_size is not None:
            rt.batch_size = batch_size
        return rt

    def build_model_config(self, *, role: Literal["agg", "prefill", "decode"]) -> config.ModelConfig:
        """Build a ModelConfig template for the given role (parallelism unset).

        ``sweep_agg`` / ``sweep_disagg`` overwrite tp/pp/dp/moe_tp/moe_ep per
        sweep point.  This template carries the resolved quant / nextn /
        feature flags only.
        """
        return config.ModelConfig(
            gemm_quant_mode=self._role_attr(role, "gemm_quant_mode"),
            moe_quant_mode=self._role_attr(role, "moe_quant_mode"),
            kvcache_quant_mode=self._role_attr(role, "kvcache_quant_mode"),
            fmha_quant_mode=self._role_attr(role, "fmha_quant_mode"),
            comm_quant_mode=self._role_attr(role, "comm_quant_mode"),
            nextn=self.nextn or 0,
            nextn_accept_rates=self.nextn_accept_rates,
            enable_wideep=self._role_attr(role, "enable_wideep"),
            enable_eplb=self._role_attr(role, "enable_eplb"),
        )

    def iter_parallel(self, role: Literal["agg", "prefill", "decode"]) -> Iterator[ParallelChoice]:
        """Yield (tp, pp, dp, moe_tp, moe_ep) tuples for the role.

        Uses sdk.utils.enumerate_parallel_config so MoE constraints match
        the legacy path exactly.
        """
        prefix = "agg_" if role == "agg" else f"{role}_"

        def _cands(dim: str) -> list[int]:
            return getattr(self, f"{prefix}{dim}_candidates")

        return iter(
            enumerate_parallel_config(
                num_gpu_list=_cands("num_gpu"),
                tp_list=_cands("tp"),
                pp_list=_cands("pp"),
                dp_list=_cands("dp"),
                moe_tp_list=_cands("moe_tp"),
                moe_ep_list=_cands("moe_ep"),
                is_moe=self._is_moe,
                backend=common.BackendName[self._role_attr(role, "backend_name")],
                enable_wideep=self._role_attr(role, "enable_wideep"),
                moe_backend=self.moe_backend,
            )
        )

    # =====================================================================
    # Properties
    # =====================================================================

    @property
    def is_moe(self) -> bool:
        return self._is_moe

    @property
    def model_family(self) -> str:
        return self._model_family

    # =====================================================================
    # sweep.py kwargs builders
    # =====================================================================

    def sweep_agg_kwargs(self, *, database) -> dict[str, Any]:
        """Return the exact kwargs needed for sweep.sweep_agg.

        Caller is responsible for loading the perf database (so it can be
        shared across multiple TaskConfigs).
        """
        if self.serving_mode != "agg":
            raise ValueError(f"sweep_agg_kwargs requires serving_mode='agg', got {self.serving_mode!r}")
        parallel_config_list = list(self.iter_parallel("agg"))
        return {
            "model_path": self.model_path,
            "runtime_config": self.build_runtime_config(),
            "database": database,
            "backend_name": self.backend_name,
            "model_config": self.build_model_config(role="agg"),
            "parallel_config_list": parallel_config_list,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "free_gpu_memory_fraction": self.free_gpu_memory_fraction,
            "max_seq_len": self.max_seq_len,
        }

    def sweep_disagg_kwargs(self, *, prefill_database, decode_database) -> dict[str, Any]:
        """Return the exact kwargs needed for sweep.sweep_disagg."""
        if self.serving_mode != "disagg":
            raise ValueError(f"sweep_disagg_kwargs requires serving_mode='disagg', got {self.serving_mode!r}")
        prefill_parallel = list(self.iter_parallel("prefill"))
        decode_parallel = list(self.iter_parallel("decode"))
        # Derive worker count ranges from replica constraints (legacy semantics).
        prefill_worker_list = list(range(1, (self.max_prefill_workers or 32) + 1))
        decode_worker_list = list(range(1, (self.max_decode_workers or 32) + 1))
        num_gpu_list = self.num_gpu_per_replica if self.num_gpu_per_replica else None
        return {
            "model_path": self.prefill_model_path,
            "runtime_config": self.build_runtime_config(),
            "prefill_database": prefill_database,
            "prefill_backend_name": self.prefill_backend_name,
            "prefill_model_config": self.build_model_config(role="prefill"),
            "prefill_parallel_config_list": prefill_parallel,
            "prefill_latency_correction": self.prefill_latency_correction,
            "decode_database": decode_database,
            "decode_backend_name": self.decode_backend_name,
            "decode_model_config": self.build_model_config(role="decode"),
            "decode_parallel_config_list": decode_parallel,
            "decode_latency_correction": self.decode_latency_correction,
            "prefill_max_num_tokens": max(self.prefill_max_batch_size, 1) * self.isl,
            "decode_max_num_tokens": self.decode_max_batch_size,
            "prefill_num_worker_list": prefill_worker_list,
            "decode_num_worker_list": decode_worker_list,
            "num_gpu_list": num_gpu_list,
        }


__all__ = ["QUANT_PRESETS", "ParallelChoice", "TaskConfig"]
