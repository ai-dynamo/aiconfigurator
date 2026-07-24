# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Task — flat user-facing config for sweep_agg / sweep_disagg.

Replaces the legacy ``sdk.task.TaskConfig`` (now deleted).  Legacy V1 YAML is
auto-detected and converted on load (see ``task_v1_compat``); the canonical new
YAML uses field names that map 1:1 to this dataclass.

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

See ``src/aiconfigurator/cli/example.yaml`` for the canonical YAML format.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.models import (
    _infer_quant_modes_from_raw_config,
    attention_op_keys,
    check_is_moe,
    get_model_family,
    resolve_dsv4_moe_arch_mode,
)
from aiconfigurator.sdk.perf_database import (
    get_latest_database_version,
    is_blackwell_system,
    is_hopper_system,
    load_system_spec,
)
from aiconfigurator.sdk.speculative import (
    SpeculativeDecodingProfile,
    normalize_speculative_decoding,
)
from aiconfigurator.sdk.utils import enumerate_parallel_config, get_model_config_from_model_path

logger = logging.getLogger(__name__)

ParallelChoice = tuple[int, int, int, int, int, int]  # (tp, pp, dp, moe_tp, moe_ep, cp)


def _default_cp_list_for(model_family: str, backend_name: str) -> list[int]:
    """Default prefill/agg ``cp_list`` for the CP auto-sweep; ``[1]`` otherwise.

    Capability-derived: any model whose class declares ``supports_cp`` on this
    backend is auto-swept over cp ∈ {1,2,4,8}. Keying off the registry (not a
    hardcoded family list) means the sweep policy never drifts from
    ``BaseModel.supports_cp``. Decode is always forced to cp=1 by iter_parallel.
    """
    from aiconfigurator.sdk.models.base import _MODEL_REGISTRY

    cls = _MODEL_REGISTRY.get(model_family)
    if cls is not None and cls.supports_cp(backend_name):
        return [1, 2, 4, 8]
    return [1]


# Legacy V1 TaskRunner swept TPOT over this fixed grid to build the latency/throughput
# Pareto frontier. Used when ``pareto_sweep=True`` (the default) so v2 matches v1.
_LEGACY_TPOT_SWEEP: list[int] = list(range(1, 20, 1)) + list(range(20, 300, 5))

# DeepSeek-V3.2 / V4 MoE on Blackwell get extra large-pipeline-parallel configs
# (PP=2/TP=8/16-GPU). Mirrors v1 _LARGE_PIPELINE_PARALLEL_MODEL_FAMILIES (backends were
# all three, i.e. unrestricted).
_LARGE_PIPELINE_PARALLEL_MODEL_FAMILIES = {"DEEPSEEKV32", "DEEPSEEKV4"}

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


# Models that get a Blackwell MoE-quant promotion on the TRT-LLM backend.
_GPTOSS_BLACKWELL_MODELS = frozenset({"openai/gpt-oss-120b", "openai/gpt-oss-20b"})

# Native FP4 routed-expert DeepSeek-V4 checkpoints and their FP8 replacements.
# The native FP4 weights are unsupported on Hopper.
_DEEPSEEK_V4_NATIVE_FP4_TO_FP8_MODEL = {
    "deepseek-ai/DeepSeek-V4-Flash": "sgl-project/DeepSeek-V4-Flash-FP8",
    "deepseek-ai/DeepSeek-V4-Pro": "sgl-project/DeepSeek-V4-Pro-FP8",
}


# SGLang MegaMoE (DeepSeek-V4) — only these checkpoints have packaged perf data.
_DEEPSEEK_V4_MEGAMOE_SUPPORTED_MODELS = {
    "deepseek-ai/DeepSeek-V4-Pro",
    "sgl-project/DeepSeek-V4-Pro-FP8",
}


def _sglang_megamoe_parallel_lists(system_name: str, should_enable_pp: bool = False) -> dict[str, list[int]]:
    """SGLang MegaMoE parallel search lists; rack-NVL aware. Mirrors v1 (initial support)."""
    spec = load_system_spec(system_name)
    has_rack_nvl = int(spec.get("node", {}).get("num_gpus_per_rack", 0) or 0) >= 32
    ep_list = [4, 8, 16, 32] if has_rack_nvl else [8]
    return {
        "num_gpu_per_worker": ep_list,
        "tp_list": [1, 2, 4, 8],
        "pp_list": ep_list if should_enable_pp else [1],
        "dp_list": [1, 2, 4, 8, 16, 32] if has_rack_nvl else [1, 2, 4, 8],
        "moe_tp_list": [1],
        "moe_ep_list": ep_list,
    }


# ---------------------------------------------------------------------------
# Default disagg search space (mirror of legacy build_disagg_parallel_lists)
# ---------------------------------------------------------------------------


def build_disagg_parallel_lists(
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

    Kept here so the new sdk.task_v2 module does not depend on V1 (sdk.task).
    Algorithm identical; locked by integration parity test.
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
        elif moe_backend == "megamoe":
            prefill_cfg = _sglang_megamoe_parallel_lists(prefill_system, should_enable_pp)
            decode_cfg = _sglang_megamoe_parallel_lists(decode_system, should_enable_pp)
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
# Task
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """Flat user-facing optimization task.

    Holds every knob the user controls (workload, model spec, search space,
    SLA targets) as a flat dataclass.  Construction (or ``__post_init__``)
    resolves model identity, backend version, quant modes, and search
    candidates so the resulting object is fully concrete.

    Entry point: ``.run()`` loads the perf database(s) internally and
    dispatches to :mod:`aiconfigurator.sdk.sweep` -- callers don't need to
    know about databases or which sweep function applies to their serving
    mode.

    See module docstring for design notes.
    """

    # ====== 1. Mode + workload ======
    serving_mode: Literal["agg", "disagg"] = "agg"
    isl: int = 4000
    osl: int = 1000
    prefix: int = 0
    # Multimodal image inputs (folded into the effective ISL by RuntimeConfig).
    image_height: int = 0
    image_width: int = 0
    num_images_per_request: int = 1
    ttft: float = 1000.0
    tpot: float = 50.0
    # When True (default), sweep TPOT over the legacy grid to build the full Pareto
    # frontier (matches v1). Set False to evaluate only the single ``tpot`` target --
    # used by the Planner, where Pareto selection happens elsewhere.
    pareto_sweep: bool = True
    request_latency: float | None = None
    total_gpus: int | None = None
    database_mode: str | None = None
    # Fine-grained HYBRID/EMPIRICAL transfer control: which empirical transfer kinds are
    # permitted (see common.TransferKind). None = all (default). Accepts a preset name
    # ("conservative"/"balanced"/"aggressive"/"off"), a kind ("xshape"), or a list thereof.
    transfer_policy: str | list | None = None
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
    # MTP speculative decoding is OFF unless explicitly requested: nextn is the
    # draft length (compute cost), nextn_accepted the average accepted draft tokens
    # per step (generation benefit, 0 <= nextn_accepted <= nextn). nextn_accepted is
    # required when nextn > 0 -- there is no built-in acceptance assumption.
    # nextn="auto" resolves the draft depth from the checkpoint's
    # num_nextn_predict_layers (absent/0 -> disabled); the acceptance value is
    # still never inferred.
    nextn: int | str = 0
    nextn_accepted: float | None = None
    moe_backend: str | None = None
    attention_backend: str | None = None  # 'flashinfer' (default) or 'fa3'; only consumed by MLA models
    wideep_num_slots: int | None = None  # EPLB slot count; defaults to num_experts when None
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
    agg_cp_candidates: list[int] | None = None

    # ====== 4. Disagg prefill worker spec ======
    prefill_model_path: str = ""
    prefill_system_name: str = ""
    prefill_backend_name: str = "trtllm"
    prefill_backend_version: str | None = None
    prefill_enable_wideep: bool = False
    prefill_enable_chunked_prefill: bool = False
    prefill_enable_eplb: bool = False
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
    prefill_cp_candidates: list[int] | None = None

    # ====== 6. Disagg decode worker spec ======
    decode_model_path: str = ""
    decode_system_name: str = ""
    decode_backend_name: str = "trtllm"
    decode_backend_version: str | None = None
    decode_enable_wideep: bool = False
    decode_enable_eplb: bool = False
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
    decode_cp_candidates: list[int] | None = None

    # ====== 8. Disagg orchestration ======
    num_gpu_per_replica: list[int] | None = None
    max_gpu_per_replica: int | None = None
    max_prefill_workers: int | None = None
    max_decode_workers: int | None = None
    prefill_max_batch_size: int = 1
    decode_max_batch_size: int = 512
    prefill_latency_correction: float = 1.1
    decode_latency_correction: float = 1.08
    # Rate-matching degradation factors: under (P_workers, D_workers) pairing,
    # neither phase delivers its standalone throughput perfectly; these model
    # the practical efficiency loss.  Calibrated against silicon (V1 default).
    rate_match_prefill_degradation: float = 0.9
    rate_match_decode_degradation: float = 0.92
    # TTFT pre-correction applied to prefill candidates before the SLA filter,
    # accounting for queueing-under-concurrency in the deployed system.
    # Used by both ``_find_best_disagg_under_constraint`` and
    # ``picking.pick_autoscale``; default 1.8 locked by parity test.
    autoscale_ttft_correction_factor: float = 1.8

    # ====== 8.5 Predictor strategy ======
    # Optional Predictor that decides how each single config point is
    # predicted.  None (default) uses sdk.predictor.AnalyticPredictor --
    # bit-identical to the pre-Predictor behavior.  Future implementations
    # (e.g. MockerPredictor wrapping Dynamo Mocker, DynamicPredictor) can
    # be injected here without touching sweep / predict / Task internals.
    # Excluded from to_dict / YAML serialization (it is a strategy object,
    # not a primitive value).
    predictor: Any = field(default=None, repr=False)

    # ====== 9. Internal — resolved in __post_init__ ======
    _is_moe: bool = field(default=False, repr=False, init=False)
    _model_family: str = field(default="", repr=False, init=False)
    _raw_config: dict = field(default_factory=dict, repr=False, init=False)
    _architecture: str = field(default="", repr=False, init=False)

    # =====================================================================
    # Construction
    # =====================================================================

    @classmethod
    def from_yaml(cls, yaml_data: dict, **overrides: Any) -> Task:
        """Construct from a flat YAML dict.

        YAML keys must match Task field names directly.  String values
        for quant_mode fields are converted to the matching enum.
        ``overrides`` (kwargs) win over YAML values.

        Any key that would not take effect is rejected with a
        ``ValueError`` -- there is no silent-ignore path.  This covers
        unknown/misspelled keys and strategy fields like ``predictor``
        that cannot be expressed in YAML (they're Python objects; pass
        them via ``overrides`` or assign after construction).

        Legacy V1 YAML (nested ``config:`` / ``mode`` / ``profiles``) is
        auto-detected and converted to the flat V2 schema, emitting a
        ``DeprecationWarning``.
        """
        from aiconfigurator.sdk.task_v1_compat import convert_v1_to_v2, is_v1_config

        if is_v1_config(yaml_data):
            warnings.warn(
                "Legacy V1 task YAML detected; auto-converting to the flat V2 schema. "
                "This compatibility path is deprecated -- migrate to the flat format "
                "(see cli/example.yaml).",
                DeprecationWarning,
                stacklevel=2,
            )
            logger.warning("from_yaml: legacy V1 YAML auto-converted to V2 (deprecated; migrate to the flat format).")
            yaml_data = convert_v1_to_v2(yaml_data)
        valid_keys = {f.name for f in dataclasses.fields(cls) if f.init and not f.name.startswith("_")}
        # Strategy objects (e.g. predictor) are valid fields but cannot be
        # constructed from YAML; writing them in YAML has no effect, so reject.
        _yaml_skip: frozenset[str] = frozenset({"predictor"})
        unknown = [k for k in yaml_data if k not in valid_keys]
        not_expressible = [k for k in yaml_data if k in _yaml_skip]
        if unknown or not_expressible:
            parts: list[str] = []
            if unknown:
                parts.append(f"unknown key(s): {', '.join(map(repr, sorted(unknown)))}")
            if not_expressible:
                parts.append(
                    f"not YAML-expressible, pass via overrides: {', '.join(map(repr, sorted(not_expressible)))}"
                )
            raise ValueError(
                "Task.from_yaml: rejecting config with key(s) that would not take effect -- "
                + "; ".join(parts)
                + ". Fix or remove them (keys are never silently ignored)."
            )
        kwargs: dict[str, Any] = {
            k: (_resolve_quant_str(k, v) if k.endswith("quant_mode") else v) for k, v in yaml_data.items()
        }
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kwargs)

    @classmethod
    def from_cli(cls, **kwargs: Any) -> Task:
        """Construct from CLI kwargs.  Filters None to let __post_init__ defaults run."""
        return cls(**{k: v for k, v in kwargs.items() if v is not None})

    # =====================================================================
    # Convenience read-only views (primary = prefill side in disagg)
    # =====================================================================
    # Disagg has no shared top-level worker fields (prefix discipline), so
    # callers that just want "the model / system / backend for this task"
    # (display, identity, file naming) read the prefill side. These never
    # set state, so they don't violate the discipline.

    @property
    def primary_model_path(self) -> str:
        return self.model_path if self.serving_mode == "agg" else self.prefill_model_path

    @property
    def primary_system_name(self) -> str:
        return self.system_name if self.serving_mode == "agg" else self.prefill_system_name

    @property
    def primary_backend_name(self) -> str:
        return self.backend_name if self.serving_mode == "agg" else self.prefill_backend_name

    @property
    def primary_backend_version(self) -> str | None:
        return self.backend_version if self.serving_mode == "agg" else self.prefill_backend_version

    # =====================================================================
    # __post_init__
    # =====================================================================

    def __post_init__(self) -> None:
        self._check_prefix_discipline()
        # Validate the MTP pair BEFORE model-identity resolution: the latter is
        # skipped when no primary model path is set, and the check must not
        # depend on it (non-negative integer nextn; finite acceptance in range).
        # nextn="auto" is the one exception: its depth comes from the checkpoint,
        # so it is resolved and validated in _resolve_model_identity.
        if self.nextn != "auto":
            self.nextn, self.nextn_accepted = normalize_speculative_decoding(self.nextn, self.nextn_accepted)
        self._validate_deepseek_v4_hardware()
        self._resolve_model_identity()
        if self.nextn == "auto":
            raise ValueError("nextn='auto' requires a model path to resolve num_nextn_predict_layers.")
        self._resolve_backend_version()
        self._normalize_wideep_moe_backend()
        self._resolve_quant_modes()
        self._resolve_search_space()
        self._validate_megamoe_backend_support()

    def _normalize_wideep_moe_backend(self) -> None:
        """enable_wideep implies the deepep_moe MoE backend (mirrors v1 __init__), so the
        DB validation picks the wideep_*_moe ops and ModelConfig gets the right kernel."""
        if self.moe_backend is not None:
            return
        wideep = (
            self.enable_wideep
            if self.serving_mode == "agg"
            else (self.prefill_enable_wideep or self.decode_enable_wideep)
        )
        if wideep:
            self.moe_backend = "deepep_moe"

    def _validate_megamoe_backend_support(self) -> None:
        """v1 _validate_megamoe_backend_support: megamoe is sglang + DeepSeek-V4-Pro + Blackwell only."""
        if self.moe_backend != "megamoe":
            return
        roles = ["agg"] if self.serving_mode == "agg" else ["prefill", "decode"]
        if self._role_attr(roles[0], "backend_name") != "sglang":
            raise ValueError("moe_backend='megamoe' is currently supported only for the SGLang backend.")
        if self._model_family != "DEEPSEEKV4":
            raise ValueError("moe_backend='megamoe' is currently supported only for DeepSeek-V4 models.")
        model = self._role_attr(roles[0], "model_path")
        if model not in _DEEPSEEK_V4_MEGAMOE_SUPPORTED_MODELS:
            raise ValueError(
                "moe_backend='megamoe' currently has packaged performance data only for "
                f"DeepSeek-V4-Pro; got model_path={model!r}."
            )
        non_blackwell = sorted(
            {
                self._role_attr(r, "system_name")
                for r in roles
                if not is_blackwell_system(self._role_attr(r, "system_name"))
            }
        )
        if non_blackwell:
            raise ValueError(
                f"moe_backend='megamoe' requires Blackwell-class systems (SM >= 100); non-Blackwell: {non_blackwell}."
            )

    def _validate_deepseek_v4_hardware(self) -> None:
        """Reject native DeepSeek-V4 FP4-expert checkpoints on Hopper (use the FP8 build)."""
        roles = ["agg"] if self.serving_mode == "agg" else ["prefill", "decode"]
        for role in roles:
            model = self._role_attr(role, "model_path")
            replacement = _DEEPSEEK_V4_NATIVE_FP4_TO_FP8_MODEL.get(model)
            if replacement and is_hopper_system(self._role_attr(role, "system_name")):
                raise ValueError(
                    f"{model} uses native FP4 routed-expert weights and is not supported on "
                    f"Hopper systems. Use {replacement} instead."
                )

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
        # MTP is never enabled implicitly: nextn defaults to 0 and must be set
        # explicitly. Surface a hint when the checkpoint ships MTP layers.
        hf_nextn = cfg.get("num_nextn_predict_layers")
        if self.nextn == "auto":
            # "auto" trusts the checkpoint for the draft DEPTH only; the
            # acceptance value is a workload measurement and is never inferred.
            resolved = int(hf_nextn or 0)
            if resolved > 0:
                try:
                    resolved, self.nextn_accepted = normalize_speculative_decoding(resolved, self.nextn_accepted)
                except ValueError as exc:
                    raise ValueError(
                        f"nextn='auto' resolved to nextn={resolved} from the checkpoint's "
                        f"num_nextn_predict_layers: {exc}"
                    ) from exc
                logger.info(
                    "nextn='auto': modeling MTP with nextn=%d from the checkpoint's num_nextn_predict_layers.",
                    resolved,
                )
            else:
                logger.info(
                    "nextn='auto': checkpoint ships no MTP layers (num_nextn_predict_layers absent or 0); "
                    "modeling WITHOUT speculative decoding."
                )
            self.nextn = resolved
        if self.nextn > 0:
            # Range/required-ness already validated in __post_init__ (validate_nextn).
            if hf_nextn is not None and self.nextn != hf_nextn:
                logger.warning(
                    "nextn=%d differs from the checkpoint's num_nextn_predict_layers=%d "
                    "(the single MTP module is reused for extra draft steps).",
                    self.nextn,
                    hf_nextn,
                )
        elif hf_nextn:
            logger.info(
                "Checkpoint ships MTP (num_nextn_predict_layers=%d) but nextn is not set; "
                "modeling WITHOUT speculative decoding. Pass nextn (or nextn='auto') and "
                "nextn_accepted to model it.",
                hf_nextn,
            )

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

        Priority (highest wins): explicit field > HF base > bfloat16 fallback.
        """
        roles = ["agg"] if self.serving_mode == "agg" else ["prefill", "decode"]
        base = _infer_quant_modes_from_raw_config(self._raw_config)

        # GPT-OSS on Blackwell (trtllm): default MoE to w4a8_mxfp4_mxfp8 for higher
        # tensor-core throughput, unless moe_quant_mode was set explicitly.  Applied
        # before the resolution loop so the explicit-wins check below preserves it.
        # (Mirrors the legacy V1 TaskConfigFactory gpt-oss-blackwell promotion; each
        # disagg role is promoted independently based on its own system.)
        for role in roles:
            if (
                self._role_attr(role, "moe_quant_mode") is None
                and self._role_attr(role, "backend_name") == "trtllm"
                and self._role_attr(role, "model_path") in _GPTOSS_BLACKWELL_MODELS
                and is_blackwell_system(self._role_attr(role, "system_name"))
            ):
                self._set_role_attr(role, "moe_quant_mode", common.MoEQuantMode.w4a8_mxfp4_mxfp8)

        # Track whether fmha came from an explicit field (vs HF/fallback): the
        # data-driven fallback below must NOT fire on an EXPLICIT fp8 -- explicit
        # values are the user's contract and validate fails fast on them.
        fmha_explicit: dict[str, bool] = {}
        for role in roles:
            for key in _QUANT_ENUM_TABLES:
                explicit = self._role_attr(role, key)
                from_hf = base.get(key)
                if key == "fmha_quant_mode":
                    fmha_explicit[role] = explicit is not None
                # Native DeepSeek-V4 on sglang uses arch-specific MoE kernels; the
                # shared helper (also called on the cli estimate path) returns the
                # dedicated perf-DB quant mode. Acts at the HF-base layer so an
                # explicit field still overrides it.
                if key == "moe_quant_mode":
                    arch_mode = resolve_dsv4_moe_arch_mode(
                        self._role_attr(role, "model_path"),
                        self._role_attr(role, "system_name"),
                        self._role_attr(role, "backend_name"),
                        self.moe_backend,
                    )
                    if arch_mode is not None:
                        from_hf = arch_mode
                fallback = _QUANT_FALLBACKS[key]

                if explicit is not None:
                    continue
                resolved = from_hf if from_hf is not None else fallback
                self._set_role_attr(role, key, resolved)

        # Data-driven FMHA resolution: if an inferred fp8 has no fp8 slice in
        # the role's fmha-keyed context-attention table, fall back to bfloat16
        # with a warning instead of failing validate later.  bf16-as-fp8 is
        # conservative: same kv-cache dtype, attention math modeled at bf16
        # throughput.  The data IS the capability statement -- there are no
        # per-model downgrade rules; when fp8 slices land for a combo (e.g.
        # DSA on Blackwell vLLM), the inference survives and uses them.
        # Explicit user fp8 is never overridden -- validate stays fail-fast
        # for it (including v1 profile-derived values).  Systems with no
        # packaged data keep the checkpoint inference untouched.
        #
        # Context-using roles only: NO generation table keys on fmha (decode
        # compute dtype follows the kv-cache dtype; the generation MLA module
        # loader drops the degenerate mla_dtype column), so an fp8 label is
        # inert on decode -- and validate likewise checks fmha only for
        # context-using roles.
        for role in roles:
            if role == "decode":
                continue
            if fmha_explicit.get(role, False):
                continue
            if self._role_attr(role, "fmha_quant_mode") != common.FMHAQuantMode.fp8:
                continue
            supported = self._context_fmha_supported_modes(role)
            if not supported or common.FMHAQuantMode.fp8.name in supported:
                continue  # fp8 data present, or no DB to consult -> keep fp8
            if common.FMHAQuantMode.bfloat16.name not in supported:
                continue  # no bf16 slice either -> let validate report the gap
            self._set_role_attr(role, "fmha_quant_mode", common.FMHAQuantMode.bfloat16)
            ctx_op, _ = self._attention_op_keys(role)
            field = "fmha_quant_mode" if self.serving_mode == "agg" else f"{role}_fmha_quant_mode"
            logger.warning(
                f"{role} fmha_quant_mode=fp8 (inferred from the model checkpoint) has no "
                f"{ctx_op!r} perf data for system={self._role_attr(role, 'system_name')!r}, "
                f"backend={self._role_attr(role, 'backend_name')!r}, "
                f"version={self._role_attr(role, 'backend_version')!r}; falling back to bfloat16 "
                f"FMHA data. Predictions are conservative if the deployed engine runs fp8 FMHA; "
                f"set {field} explicitly to override."
            )

    def _attention_op_keys(self, role: str) -> tuple[str, str]:
        """(context_op, generation_op) support-matrix keys for this role's model
        family / backend / wideep combination (shared by the resolve-time FMHA
        fallback and ``_check_role_against_db``; mapping lives in
        ``models.attention_op_keys``)."""
        return attention_op_keys(
            self._model_family,
            self._role_attr(role, "backend_name"),
            bool(self._role_attr(role, "enable_wideep")),
        )

    def _try_load_role_database(self, role: str):
        """Load the role's perf DB, returning None when the perf data is
        unavailable (missing system/backend/version data).  Programmer errors
        propagate; only data-availability failures are swallowed."""
        from aiconfigurator.sdk.perf_database import (
            PerfDataNotAvailableError,
            has_perf_data_not_available_cause,
        )

        system = self._role_attr(role, "system_name")
        backend = self._role_attr(role, "backend_name")
        version = self._role_attr(role, "backend_version")
        if not (system and backend and version):
            return None
        try:
            return self._load_database(system, backend, version)
        except (PerfDataNotAvailableError, FileNotFoundError) as exc:
            logger.debug("perf DB unavailable for %s role (%s/%s/%s): %s", role, system, backend, version, exc)
            return None
        except Exception as exc:
            # Match the legacy "DB error" envelope (e.g. wrapped FileNotFoundError
            # inside RuntimeError) without swallowing programmer typos.
            if not has_perf_data_not_available_cause(exc):
                raise
            logger.debug("perf DB unavailable for %s role (%s/%s/%s): %s", role, system, backend, version, exc)
            return None

    def _context_fmha_supported_modes(self, role: str) -> list[str]:
        """FMHA modes with perf data for this role's fmha-keyed context-attention
        op, jointly with the role's resolved kv-cache mode (an fmha slice that
        exists only under a different kv dtype cannot serve this role's
        queries).  Returns [] when the DB (or the op's table) is unavailable,
        meaning "no information" -- callers must not read that as "nothing
        supported"."""
        from aiconfigurator.sdk.perf_database import context_fmha_supported_modes

        database = self._try_load_role_database(role)
        if database is None:
            return []
        return context_fmha_supported_modes(
            database,
            self._attention_op_keys(role)[0],
            self._role_attr(role, "kvcache_quant_mode"),
        )

    def _resolve_search_space(self) -> None:
        roles = ["agg"] if self.serving_mode == "agg" else ["prefill", "decode"]
        # Candidate fields the user did NOT supply are eligible for default augmentation
        # (large-PP). User-supplied candidates win, matching v1's yaml-over-defaults order.
        defaulted = {
            f"{role}_{dim}_candidates"
            for role in roles
            for dim in ("num_gpu", "tp", "pp", "dp", "moe_tp", "moe_ep")
            if getattr(self, f"{role}_{dim}_candidates") is None
        }
        if self.serving_mode == "agg":
            self._resolve_agg_search()
        else:
            self._resolve_disagg_search()
        self._apply_large_pipeline_parallel(defaulted)
        self._apply_total_gpus_budget()

    def _large_pipeline_parallel_applies(self) -> bool:
        """v1 _large_pipeline_parallel_worker_defaults_apply: DeepSeek-V3.2/V4 MoE on
        Blackwell, non-wideep, total_gpus>=16 get extra PP=2 / TP=8 / 16-GPU configs."""
        if not self._is_moe or self._model_family not in _LARGE_PIPELINE_PARALLEL_MODEL_FAMILIES:
            return False
        if self.serving_mode == "agg":
            wideep = self.enable_wideep
            systems = [self.system_name]
        else:
            wideep = self.prefill_enable_wideep or self.decode_enable_wideep
            systems = [self.prefill_system_name, self.decode_system_name]
        if wideep or self.moe_backend in ("deepep_moe", "megamoe"):
            return False
        if self.total_gpus is None or self.total_gpus < 16:
            return False
        try:
            return all(is_blackwell_system(s) for s in systems)
        except Exception:
            return False

    def _apply_large_pipeline_parallel(self, defaulted: set[str]) -> None:
        if not self._large_pipeline_parallel_applies():
            return
        roles = ["agg"] if self.serving_mode == "agg" else ["prefill", "decode"]
        merges = {
            "num_gpu": [16],
            "tp": [8],
            "pp": [2],
            "dp": [1],
            "moe_tp": [1, 2, 4, 8],
            "moe_ep": [1, 2, 4, 8],
        }
        for role in roles:
            for dim, add in merges.items():
                attr = f"{role}_{dim}_candidates"
                if attr not in defaulted:
                    continue  # user supplied this explicitly; v1 yaml override wins
                cur = getattr(self, attr) or []
                setattr(self, attr, sorted(set(cur) | set(add)))

    def _apply_total_gpus_budget(self) -> None:
        """Clamp the per-worker GPU-count search space to the total_gpus budget and
        validate it. Mirrors v1 _finalize_agg / _finalize_disagg."""
        if self.total_gpus is None:
            return
        if self.serving_mode == "agg":
            if self.total_gpus < 0:
                raise ValueError(f"total_gpus of agg must be no smaller than 0, got {self.total_gpus}")
            self.agg_num_gpu_candidates = [n for n in self.agg_num_gpu_candidates if n <= self.total_gpus]
        else:
            if self.total_gpus < 2:
                raise ValueError(f"total_gpus must be greater than 2 for disagg, got {self.total_gpus}")
            if self.max_gpu_per_replica is not None:
                self.max_gpu_per_replica = min(self.total_gpus, self.max_gpu_per_replica)
            # num_gpu_per_replica is intentionally NOT filtered here: v1 keeps the full list
            # and applies max_gpu_per_replica as a ceiling at sweep time (get_working_list);
            # v2 mirrors that in sweep_disagg_kwargs, so construct-time state matches v1.
            self.prefill_num_gpu_candidates = [n for n in self.prefill_num_gpu_candidates if n <= self.total_gpus]
            self.decode_num_gpu_candidates = [n for n in self.decode_num_gpu_candidates if n <= self.total_gpus]

    def _resolve_agg_search(self) -> None:
        def _set(name: str, values: list[int]) -> None:
            if getattr(self, name) is None:
                setattr(self, name, values)

        # CP auto-sweep for validated families (sglang); [1] otherwise. agg runs
        # prefill in-worker, so cp applies; decode-cp=1 is enforced in iter_parallel.
        _set("agg_cp_candidates", _default_cp_list_for(self._model_family, self.backend_name))

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

        if self.backend_name == "sglang" and self.moe_backend == "megamoe":
            mm = _sglang_megamoe_parallel_lists(self.system_name)
            _set("agg_num_gpu_candidates", mm["num_gpu_per_worker"])
            _set("agg_tp_candidates", mm["tp_list"])
            _set("agg_pp_candidates", mm["pp_list"])
            _set("agg_dp_candidates", mm["dp_list"])
            _set("agg_moe_tp_candidates", mm["moe_tp_list"])
            _set("agg_moe_ep_candidates", mm["moe_ep_list"])
        elif self.backend_name == "trtllm" and self.enable_wideep:
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
            if self.moe_backend == "deepep_moe":
                # Intra-node DeepEP (ep 1-8, NVLink): EP-only
                _set("agg_moe_tp_candidates", [1])
                _set("agg_moe_ep_candidates", [1, 2, 4, 8])
            else:
                # Standard comm (fused_moe + allgather/RS)
                _set("agg_moe_tp_candidates", [1, 2, 4, 8])
                _set("agg_moe_ep_candidates", [1, 2, 4, 8])
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
        prefill_cfg, decode_cfg = build_disagg_parallel_lists(
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
            "cp_list": f"{role}_cp_candidates",
        }
        for k_src, k_attr in map_to_attr.items():
            if getattr(self, k_attr) is None:
                if k_src == "cp_list":
                    # Decode is always cp=1 (CP is prefill-only). prefill/agg
                    # auto-sweep cp for CP-validated families (else [1]); an
                    # explicit worker-config cp_list still wins. A user-supplied
                    # non-1 decode cp is rejected in iter_parallel.
                    if role == "decode":
                        value = [1]
                    else:
                        backend = self._role_attr(role, "backend_name")
                        value = src.get(k_src, _default_cp_list_for(self._model_family, backend))
                else:
                    value = src[k_src]
                setattr(self, k_attr, value)

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
            image_height=self.image_height,
            image_width=self.image_width,
            num_images_per_request=self.num_images_per_request,
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
            nextn=self.nextn,
            enable_wideep=self._role_attr(role, "enable_wideep"),
            enable_eplb=self._role_attr(role, "enable_eplb"),
            # moe_backend / attention_backend / wideep_num_slots are shared across roles
            # (Task has no per-role variant) and fed to ModelConfig so get_model selects the
            # right MoE kernel (deepep_moe / megamoe), MLA attention perf tables (fa3 vs
            # flashinfer), and EPLB slot count. workload_distribution remains non-configurable
            # in v2 and ModelConfig's default matches v1's.
            moe_backend=self.moe_backend,
            # None means "unspecified" -> fall back to flashinfer (matches v1 and ModelConfig's default).
            attention_backend=self.attention_backend or "flashinfer",
            wideep_num_slots=self.wideep_num_slots,
        )

    def build_speculative_profile(self) -> SpeculativeDecodingProfile:
        """Build the upper-layer expected-progress assumption for prediction."""
        return SpeculativeDecodingProfile.from_inputs(self.nextn, self.nextn_accepted)

    def iter_parallel(self, role: Literal["agg", "prefill", "decode"]) -> Iterator[ParallelChoice]:
        """Yield (tp, pp, dp, moe_tp, moe_ep, cp) tuples for the role.

        Uses sdk.utils.enumerate_parallel_config so MoE constraints match
        the legacy path exactly.
        """
        prefix = "agg_" if role == "agg" else f"{role}_"

        def _cands(dim: str) -> list[int]:
            return getattr(self, f"{prefix}{dim}_candidates")

        # CP is modeled for context/prefill only; decode must be cp=1. Fail loud
        # rather than silently coercing a user-supplied decode cp>1.
        cp_list = _cands("cp") or [1]
        if role == "decode" and any(c != 1 for c in cp_list):
            raise ValueError(
                f"decode CP must be 1 (CP is modeled for prefill only); got "
                f"decode_cp_candidates={cp_list}. Enable CP via prefill/agg instead."
            )

        return iter(
            enumerate_parallel_config(
                num_gpu_list=_cands("num_gpu"),
                tp_list=_cands("tp"),
                pp_list=_cands("pp"),
                dp_list=_cands("dp"),
                moe_tp_list=_cands("moe_tp"),
                moe_ep_list=_cands("moe_ep"),
                cp_list=cp_list,
                is_moe=self._is_moe,
                backend=common.BackendName[self._role_attr(role, "backend_name")],
                enable_wideep=self._role_attr(role, "enable_wideep"),
                moe_backend=self.moe_backend,
            )
        )

    # =====================================================================
    # Validation
    # =====================================================================

    def validate(self) -> None:
        """Check that the resolved task is internally consistent and supported.

        Two layers:
        - Static checks: required fields, DeepSeek+vLLM exclusion.  Always
          run, no I/O.
        - Database-dependent checks: each user-selected quant mode is in
          the perf database's ``supported_quant_mode`` list for its op
          (this is where fp8_static is gated by overhead-table availability)
          (gemm, moe / wideep_*_moe, context_attention / context_mla /
          dsa_context_module / deepseek_v4_context_module / wideep_context_mla,
          and the corresponding generation_* op).  Skipped silently if
          the DB cannot be loaded, or if the model is DeepSeek-V4 in a
          synthetic database mode (SOL / SOL_FULL / EMPIRICAL / HYBRID).

        Database load is cheap (``get_database`` is module-level cached),
        and the load happens later in sweep anyway — failing here just
        moves the error to a friendlier point.

        Raises:
            ValueError / NotImplementedError on a contradiction.
            UnsupportedWideepConfigError specifically for wideep_* ops
            (lets callers distinguish from generic ``ValueError``).
        """
        if self.attention_backend is not None and self.attention_backend not in ("flashinfer", "fa3"):
            raise ValueError(f"attention_backend must be 'flashinfer' or 'fa3', got {self.attention_backend!r}.")
        if self.wideep_num_slots is not None and self.wideep_num_slots <= 0:
            raise ValueError(f"wideep_num_slots must be a positive integer, got {self.wideep_num_slots!r}.")
        if self.serving_mode == "agg":
            self._validate_agg()
        elif self.serving_mode == "disagg":
            self._validate_disagg()
        else:
            raise ValueError(f"Invalid serving_mode: {self.serving_mode!r}")
        self._validate_database_quant_modes()

    def _validate_agg(self) -> None:
        if not self.model_path:
            raise ValueError("agg mode requires model_path")
        if not self.system_name:
            raise ValueError("agg mode requires system_name")
        # fp8_static is not hard-gated to trtllm: it is derived from the dynamic
        # fp8 GEMM minus compute_scale/scale_matrix overhead and works on any
        # backend whose perf DB carries those tables.  _validate_database_quant_modes
        # rejects it on backends/systems that lack the data.

    def _validate_disagg(self) -> None:
        if not self.prefill_model_path or not self.decode_model_path:
            raise ValueError("disagg mode requires both prefill_model_path and decode_model_path.")
        if self.prefill_model_path != self.decode_model_path:
            # sweep_disagg currently takes a single model_path used for both
            # phases (Task.sweep_disagg_kwargs passes self.prefill_model_path).
            # Hetero-disagg means different *systems*, not different models;
            # enforce that explicitly so cross-model setups fail loud instead
            # of silently using the prefill model on the decode side.
            raise ValueError(
                f"disagg mode requires prefill_model_path == decode_model_path; "
                f"got prefill={self.prefill_model_path!r}, decode={self.decode_model_path!r}.  "
                "Hetero-model disagg is not supported by sweep_disagg today."
            )
        if not self.prefill_system_name or not self.decode_system_name:
            raise ValueError("disagg mode requires both prefill_system_name and decode_system_name.")
        # fp8_static is not hard-gated to trtllm (see _validate_agg); the
        # per-role DB check in _validate_database_quant_modes governs support.

    def _validate_database_quant_modes(self) -> None:
        """Validate user's quant modes against the perf database's supported list.

        Mirrors the per-op check in V1's ``TaskConfig.validate``.  Skipped
        silently if the DB can't be loaded or for DeepSeek-V4 in synthetic
        modes (where the supported_quant_mode table is incomplete).
        """
        # DeepSeek-V4 in synthetic database modes: DB's supported_quant_mode
        # list is incomplete; skip entirely (V1 parity).
        if self._model_family == "DEEPSEEKV4" and self.database_mode in (
            "SOL",
            "SOL_FULL",
            "EMPIRICAL",
            "HYBRID",
        ):
            return

        if self.serving_mode == "agg":
            self._check_role_against_db("agg", validate_context=True, validate_generation=True)
        else:
            self._check_role_against_db("prefill", validate_context=True, validate_generation=False)
            self._check_role_against_db("decode", validate_context=False, validate_generation=True)

    def _check_role_against_db(
        self,
        role: str,
        *,
        validate_context: bool,
        validate_generation: bool,
    ) -> None:
        """For one role, fetch its perf DB and verify each quant mode is supported."""
        from aiconfigurator.sdk.errors import UnsupportedWideepConfigError

        system = self._role_attr(role, "system_name")
        backend = self._role_attr(role, "backend_name")
        version = self._role_attr(role, "backend_version")
        if not (system and backend and version):
            return  # nothing to validate against

        # DB unavailable; let sweep surface the real error later.
        database = self._try_load_role_database(role)

        if database is None:
            # In SILICON mode the DB must exist; fp8_static is derived from
            # compute_scale/scale_matrix overhead tables we can't confirm without it,
            # so fail fast rather than defer to a late run() failure.  Other modes
            # (and other quant modes) keep deferring to the sweep.
            if self.database_mode in (None, common.DatabaseMode.SILICON.name) and (
                self._role_attr(role, "gemm_quant_mode") == common.GEMMQuantMode.fp8_static
            ):
                raise ValueError(
                    f"fp8_static GEMM mode requires perf data that is unavailable for "
                    f"system={system!r}, backend={backend!r}, version={version!r}."
                )
            return

        supported: dict = getattr(database, "supported_quant_mode", {}) or {}
        moe_backend = self.moe_backend  # shared across roles
        is_moe = self._is_moe

        # Pick the attention-module op keys for this (model family, backend, wideep).
        ctx_op, gen_op = self._attention_op_keys(role)

        # supported_quant_mode is a DATA-PRESENCE list (which quants the DB carries
        # tables for), not a backend-capability list. In SILICON that equals what we
        # can model. In HYBRID/EMPIRICAL the MoE util-empirical path can synthesize a
        # quant from a collected quant that shares its (memory, compute) profile
        # (XQUANT cross-quant transfer, see operations/moe.py) -- only MoE implements
        # this, so only MoE relaxes. Truly-unreachable quants (no same-profile data)
        # still fail early here rather than crashing late in the sweep.
        # Admission via the XQUANT cross-quant transfer only holds if (a) we're in a
        # non-SILICON mode AND (b) the resolved transfer policy actually enables XQUANT.
        # Otherwise operations/moe.py rejects the quant at query time by policy, so
        # validate must not pre-admit it (e.g. transfer_policy="off"/"conservative").
        xquant_enabled = self.database_mode not in (
            None,
            common.DatabaseMode.SILICON.name,
        ) and common.TransferKind.XQUANT in common.resolve_transfer_policy(self.transfer_policy)

        def _profile_reachable(mode: Any, supported_names: list) -> bool:
            enum_cls = type(mode)
            val = getattr(mode, "value", None)
            qp = (getattr(val, "memory", None), getattr(val, "compute", None))
            for nm in supported_names:
                try:
                    other = enum_cls[nm].value
                except (KeyError, AttributeError):
                    continue
                if (getattr(other, "memory", None), getattr(other, "compute", None)) == qp:
                    return True
            return False

        def _check(op: str, mode: Any, *, profile_transfer: bool = False) -> None:
            if mode is None:
                return
            modes = supported.get(op, []) or []
            if not modes:
                return  # DB doesn't record support for this op; skip
            name = mode.name if hasattr(mode, "name") else str(mode)
            if name in modes:
                return
            # Modes that normalize to a different table name for perf queries
            # (nvfp4_wo -> bfloat16, w4a16_mxfp4_cutlass -> w4a16_mxfp4) are
            # accepted when the target table mode is supported.
            validation_aliases = {"nvfp4_wo": "bfloat16", "w4a16_mxfp4_cutlass": "w4a16_mxfp4"}
            alias = validation_aliases.get(name)
            if alias and alias in modes:
                return
            if profile_transfer and xquant_enabled and _profile_reachable(mode, modes):
                return  # transfer-reachable in HYBRID/EMPIRICAL with XQUANT enabled
            exc_type = UnsupportedWideepConfigError if op.startswith("wideep_") else ValueError
            raise exc_type(
                f"Unsupported {op} quant mode {name!r} for system={system!r}, "
                f"backend={backend!r}, version={version!r}. "
                f"Supported {op} modes: {sorted(modes)}"
            )

        # GEMM is always validated (applies to all worker shapes).
        _check("gemm", self._role_attr(role, "gemm_quant_mode"))

        # MoE — only when model is MoE.
        if is_moe:
            moe_mode = self._role_attr(role, "moe_quant_mode")
            if backend == "sglang" and moe_backend == "deepep_moe":
                # WideEP MoE: per-phase op keys (raises UnsupportedWideepConfigError).
                if validate_context:
                    _check("wideep_context_moe", moe_mode, profile_transfer=True)
                if validate_generation:
                    _check("wideep_generation_moe", moe_mode, profile_transfer=True)
            else:
                _check("moe", moe_mode, profile_transfer=True)

        # FMHA: only meaningful for context-using workers (agg, prefill).
        if validate_context:
            _check(ctx_op, self._role_attr(role, "fmha_quant_mode"))

        # KV cache: only meaningful for generation-using workers (agg, decode).
        if validate_generation:
            _check(gen_op, self._role_attr(role, "kvcache_quant_mode"))

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
    # Serialization
    # =====================================================================

    def to_dict(self) -> dict[str, Any]:
        """Return a flat dict snapshot of every user-facing field after resolution.

        Internal fields (those starting with ``_``) are excluded.  Enum
        values are emitted as their ``.name`` string (e.g.
        ``GEMMQuantMode.fp8_block`` → ``"fp8_block"``).  None values
        are kept (so the caller can see which fields are still unresolved).

        Useful for debugging ("what did the user actually get after
        __post_init__?") and for writing an "effective config" report.
        """
        # Strategy fields hold non-serializable objects; skip them.
        non_serializable: frozenset[str] = frozenset({"predictor"})

        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            if f.name.startswith("_") or not f.init:
                continue
            if f.name in non_serializable:
                continue
            value = getattr(self, f.name)
            if hasattr(value, "name") and hasattr(value, "value"):
                # Enum — emit its name
                value = value.name
            out[f.name] = value
        return out

    def to_yaml(self) -> str:
        """Return a YAML string of :func:`to_dict` output.

        The result is round-trippable through :func:`from_yaml` (modulo
        None fields which are accepted by the constructor as defaults).
        """
        import yaml

        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    # =====================================================================
    # sweep.py kwargs builders
    # =====================================================================

    def sweep_agg_kwargs(self, *, database) -> dict[str, Any]:
        """Return the exact kwargs needed for sweep.sweep_agg.

        Caller is responsible for loading the perf database (so it can be
        shared across multiple Tasks).
        """
        if self.serving_mode != "agg":
            raise ValueError(f"sweep_agg_kwargs requires serving_mode='agg', got {self.serving_mode!r}")
        parallel_config_list = list(self.iter_parallel("agg"))
        runtime_config = self.build_runtime_config()
        if self.pareto_sweep:
            runtime_config.tpot = _LEGACY_TPOT_SWEEP
        return {
            "model_path": self.model_path,
            "runtime_config": runtime_config,
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
        # Mirror v1 get_working_list(num_gpu_per_replica, max_gpu_per_replica): an explicit
        # list is filtered by the cap; a None list (WideEP) becomes range(1, cap+1) so the
        # replica size stays bounded (v2 sweep gates by this list, not a max ceiling).
        if self.num_gpu_per_replica:
            num_gpu_list = self.num_gpu_per_replica
            if self.max_gpu_per_replica is not None:
                num_gpu_list = [n for n in num_gpu_list if n <= self.max_gpu_per_replica]
        elif self.max_gpu_per_replica is not None:
            num_gpu_list = list(range(1, self.max_gpu_per_replica + 1))
        else:
            num_gpu_list = None
        # SGLang non-wideep disaggregated serving requires prefill/decode TP to match
        # (KV transfer layout constraint, ai-dynamo/dynamo#5870). WideEP relaxes it.
        require_same_tp = self.prefill_backend_name == "sglang" and not (
            self.prefill_enable_wideep or self.decode_enable_wideep
        )
        runtime_config = self.build_runtime_config()
        if self.pareto_sweep:
            runtime_config.tpot = _LEGACY_TPOT_SWEEP
        return {
            "model_path": self.prefill_model_path,
            "runtime_config": runtime_config,
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
            "rate_matching_prefill_degradation": self.rate_match_prefill_degradation,
            "rate_matching_decode_degradation": self.rate_match_decode_degradation,
            "autoscale_ttft_correction_factor": self.autoscale_ttft_correction_factor,
            "require_same_tp": require_same_tp,
        }

    # =====================================================================
    # Optimization entry point
    # =====================================================================

    def _load_database(self, system: str, backend: str, version: str):
        """Load the perf DB honoring database_mode (SILICON/HYBRID/EMPIRICAL). Non-SILICON
        modes allow missing measured data. Returns an immutable, configuration-scoped
        lightweight view so mode and transfer policy cannot mutate the process-cached
        data template."""
        from aiconfigurator.sdk.perf_database import get_database_view

        allow_missing = self.database_mode is not None and self.database_mode != common.DatabaseMode.SILICON.name
        return get_database_view(
            system,
            backend,
            version,
            allow_missing_data=allow_missing,
            database_mode=self.database_mode,
            transfer_policy=self.transfer_policy,
        )

    def run(self, *, autoscale: bool = False, validate: bool = True):
        """Run the sweep and return a feasible-candidate DataFrame.

        Loads the perf database(s) for the active role(s) internally and
        dispatches to ``sweep_agg`` or ``sweep_disagg`` based on
        ``serving_mode``.  Callers do not need to know about databases or
        which sweep function applies.

        Args:
            autoscale: disagg-only.  When True, prefill and decode workers
                are picked independently via ``picking.pick_autoscale`` --
                no rate matching is performed and the result has
                ``(p)workers=1`` and ``(d)workers=1``.  Ignored in agg mode.
            validate: when True (default), call ``validate()`` first to fail fast
                on unsupported quant / WideEP configs -- matches v1, which validates
                in ``__init__``.  Set False for a best-effort sweep that silently
                skips unsupported parallel configs (e.g. the Planner).

        Returns:
            pandas.DataFrame -- ``common.ColumnsAgg`` schema for agg,
            ``common.ColumnsDisagg`` for disagg.  This is the SLA-feasible
            candidate set; Pareto frontier computation is downstream in
            ``aiconfigurator.sdk.picking``.
        """
        if validate:
            self.validate()
        from aiconfigurator.sdk.sweep import sweep_agg, sweep_disagg

        if self.serving_mode == "agg":
            if autoscale:
                raise ValueError("autoscale is only supported in disagg mode")
            database = self._load_database(self.system_name, self.backend_name, self.backend_version)
            return sweep_agg(
                **self.sweep_agg_kwargs(database=database),
                predictor=self.predictor,
                speculative_profile=self.build_speculative_profile(),
            )
        if self.serving_mode == "disagg":
            prefill_database = self._load_database(
                self.prefill_system_name, self.prefill_backend_name, self.prefill_backend_version
            )
            decode_database = self._load_database(
                self.decode_system_name, self.decode_backend_name, self.decode_backend_version
            )
            return sweep_disagg(
                **self.sweep_disagg_kwargs(prefill_database=prefill_database, decode_database=decode_database),
                autoscale=autoscale,
                predictor=self.predictor,
                speculative_profile=self.build_speculative_profile(),
            )
        raise ValueError(f"Invalid serving_mode: {self.serving_mode!r}")

    # =====================================================================
    # Single-point evaluation (subsumes cli_estimate)
    # =====================================================================

    def run_single_agg(
        self,
        *,
        tp: int,
        pp: int = 1,
        dp: int = 1,
        moe_tp: int = 1,
        moe_ep: int = 1,
        batch_size: int,
        ctx_tokens: int | None = None,
    ) -> dict:
        """Evaluate one fixed agg config point and return its row dict.

        Subsumes the per-point use case that ``cli/api.cli_estimate``
        handles today (40 separate kwargs, custom model/backend wiring).
        Reads model_path / system_name / backend / quant / nextn / isl /
        osl from the Task itself; only the per-point dimensions are
        passed as method args.

        Args:
            tp / pp / dp / moe_tp / moe_ep: parallelism for this single point.
            batch_size: concurrency (max in-flight requests).
            ctx_tokens: per-step context-token budget for the IFB
                scheduler.  Defaults to ``self.isl`` (full prefill in
                one step) -- matching ``cli_estimate`` semantics.

        Returns:
            Row dict in ``common.ColumnsAgg`` schema, equivalent to one
            row of what ``run()`` would produce for the same point.

        Raises:
            ValueError: if called on a disagg Task.  Use
                :meth:`run_single_disagg` instead.
            RuntimeError: on OOM at this config point.
        """
        if self.serving_mode != "agg":
            raise ValueError(
                f"run_single_agg requires serving_mode='agg', got {self.serving_mode!r}; "
                "use run_single_disagg for disagg."
            )
        from aiconfigurator.sdk.backends.factory import get_backend
        from aiconfigurator.sdk.models import get_model
        from aiconfigurator.sdk.predict import predict_agg_worker

        model_config = self.build_model_config(role="agg")
        model_config.tp_size = tp
        model_config.pp_size = pp
        model_config.attention_dp_size = dp if self._is_moe else 1
        model_config.moe_tp_size = moe_tp
        model_config.moe_ep_size = moe_ep

        runtime_config = self.build_runtime_config(batch_size=batch_size)
        database = self._load_database(self.system_name, self.backend_name, self.backend_version)
        backend = get_backend(self.backend_name)
        model = get_model(self.model_path, model_config, self.backend_name)

        backend_kwargs: dict[str, Any] = {}
        if self.max_seq_len is not None:
            backend_kwargs["max_seq_len"] = self.max_seq_len
        if self.free_gpu_memory_fraction is not None:
            backend_kwargs["free_gpu_memory_fraction"] = self.free_gpu_memory_fraction

        summary = predict_agg_worker(
            model=model,
            backend=backend,
            database=database,
            runtime_config=runtime_config,
            ctx_tokens=ctx_tokens if ctx_tokens is not None else self.isl,
            predictor=self.predictor,
            speculative_profile=self.build_speculative_profile(),
            **backend_kwargs,
        )
        if summary.check_oom():
            raise RuntimeError(
                f"OOM at tp={tp} pp={pp} dp={dp} moe_tp={moe_tp} moe_ep={moe_ep} "
                f"batch_size={batch_size}.  Reduce batch_size, increase parallelism, "
                "or use a quantized model."
            )
        result = summary.get_result_dict()
        if result is None:
            raise RuntimeError("run_single_agg produced no result; configuration may be invalid.")
        return result

    def run_single_disagg(
        self,
        *,
        prefill_tp: int,
        prefill_pp: int = 1,
        prefill_dp: int = 1,
        prefill_moe_tp: int = 1,
        prefill_moe_ep: int = 1,
        prefill_batch_size: int = 1,
        prefill_num_workers: int = 1,
        decode_tp: int,
        decode_pp: int = 1,
        decode_dp: int = 1,
        decode_moe_tp: int = 1,
        decode_moe_ep: int = 1,
        decode_batch_size: int,
        decode_num_workers: int = 1,
    ) -> dict:
        """Evaluate one fixed disagg config point and return its row dict.

        Subsumes the disagg per-point use case from ``cli_estimate``.
        Reads workload + model_path + quant from the Task; per-role
        parallelism, batch_size, and num_workers come from args.

        Returns:
            Row dict in ``common.ColumnsDisagg`` schema (one rate-matched
            P/D pair).

        Raises:
            ValueError: if called on an agg Task.
            RuntimeError: on OOM in either phase.
        """
        if self.serving_mode != "disagg":
            raise ValueError(
                f"run_single_disagg requires serving_mode='disagg', got {self.serving_mode!r}; "
                "use run_single_agg for agg."
            )
        from aiconfigurator.sdk.backends.factory import get_backend
        from aiconfigurator.sdk.models import get_model
        from aiconfigurator.sdk.predict import predict_disagg_worker
        from aiconfigurator.sdk.sweep import _rate_match_dict

        # --- Prefill phase ---
        p_mc = self.build_model_config(role="prefill")
        p_mc.tp_size = prefill_tp
        p_mc.pp_size = prefill_pp
        p_mc.attention_dp_size = prefill_dp if self._is_moe else 1
        p_mc.moe_tp_size = prefill_moe_tp
        p_mc.moe_ep_size = prefill_moe_ep

        p_rt = self.build_runtime_config(batch_size=prefill_batch_size)
        p_db = self._load_database(self.prefill_system_name, self.prefill_backend_name, self.prefill_backend_version)
        p_backend = get_backend(self.prefill_backend_name)
        p_model = get_model(self.prefill_model_path, p_mc, self.prefill_backend_name)

        p_summary = predict_disagg_worker(
            model=p_model,
            backend=p_backend,
            database=p_db,
            runtime_config=p_rt,
            role="prefill",
            latency_correction=self.prefill_latency_correction,
            predictor=self.predictor,
            speculative_profile=self.build_speculative_profile(),
        )
        if p_summary.check_oom():
            raise RuntimeError(
                f"OOM in prefill phase at tp={prefill_tp} pp={prefill_pp} dp={prefill_dp} "
                f"batch_size={prefill_batch_size}."
            )

        # --- Decode phase ---
        d_mc = self.build_model_config(role="decode")
        d_mc.tp_size = decode_tp
        d_mc.pp_size = decode_pp
        d_mc.attention_dp_size = decode_dp if self._is_moe else 1
        d_mc.moe_tp_size = decode_moe_tp
        d_mc.moe_ep_size = decode_moe_ep

        d_rt = self.build_runtime_config(batch_size=decode_batch_size)
        d_db = self._load_database(self.decode_system_name, self.decode_backend_name, self.decode_backend_version)
        d_backend = get_backend(self.decode_backend_name)
        d_model = get_model(self.decode_model_path, d_mc, self.decode_backend_name)

        d_summary = predict_disagg_worker(
            model=d_model,
            backend=d_backend,
            database=d_db,
            runtime_config=d_rt,
            role="decode",
            latency_correction=self.decode_latency_correction,
            predictor=self.predictor,
            speculative_profile=self.build_speculative_profile(),
        )
        if d_summary.check_oom():
            raise RuntimeError(
                f"OOM in decode phase at tp={decode_tp} pp={decode_pp} dp={decode_dp} batch_size={decode_batch_size}."
            )

        # --- Rate-match the pair ---
        p_dict = p_summary.get_summary_df().iloc[0].to_dict()
        d_dict = d_summary.get_summary_df().iloc[0].to_dict()
        return _rate_match_dict(p_dict, prefill_num_workers, d_dict, decode_num_workers)


__all__ = ["ParallelChoice", "Task"]
