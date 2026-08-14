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
  (model_path, system_name, backend_name, quant_*, ...) are not used and
  setting them raises ValueError.  Use prefill_* / decode_* fields
  explicitly.  (Deprecated exception: enable_wideep only warns — it is
  ignored everywhere, see _warn_large_ep_flag.)
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
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.errors import NoFeasibleConfigError
from aiconfigurator.sdk.models import (
    _get_model_info,
    _infer_quant_modes_from_raw_config,
    attention_op_keys,
    check_is_moe,
    get_model_family,
    resolve_dsv4_moe_arch_mode,
    resolve_kimi_k3_moe_arch_mode,
)
from aiconfigurator.sdk.models.blocks.moe import LARGE_EP_READY_FAMILIES, MoEBlockShape
from aiconfigurator.sdk.operations.moe_comm import MOE_A2A_BACKENDS, nodes_for
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

# (model_path, system, backend, version) combos that already emitted the
# "no large-EP coverage" INFO log (spec section 6: one log per model/system,
# not one per Task or per probe).
_LARGE_EP_EMPTY_COVERAGE_LOGGED: set[tuple[str, str, str, str | None]] = set()

# Same dedupe key, for the "generation covered but context is not" warning
# (see Task._resolve_moe_comm_backend).
_LARGE_EP_ASYMMETRIC_COVERAGE_WARNED: set[tuple[str, str, str, str | None]] = set()

# ---------------------------------------------------------------------------
# Deprecated large-EP flags: accepted, warned once per key per process, ignored
# (large-EP participation is coverage-driven per tuple — see
# _resolve_moe_comm_backend). The parsed VALUES are kept: the resolved task /
# exp_config.yaml artifact still carries them, and enable_wideep still spells
# moe_backend="deepep_moe" (_normalize_wideep_moe_backend), both deliberately
# retained for artifact stability. Mirrors _warn_legacy_marker_once in
# aiconfigurator_core.sdk.perf_database.
# ---------------------------------------------------------------------------

_DEPRECATED_LARGE_EP_KEYS = ("enable_wideep", "prefill_enable_wideep", "decode_enable_wideep")
_LARGE_EP_DEPRECATION_MSG = (
    "'%s' is deprecated and ignored: large-EP is explored automatically from data "
    "coverage; restrict EP sizes with *_moe_ep_candidates."
)
_warned_large_ep_keys: set[str] = set()


def _warn_large_ep_flag(key: str) -> None:
    """One-time-per-key deprecation warning for the legacy large-EP flags."""
    if key in _warned_large_ep_keys:
        return
    _warned_large_ep_keys.add(key)
    warnings.warn(_LARGE_EP_DEPRECATION_MSG % key, DeprecationWarning, stacklevel=3)
    logger.warning(_LARGE_EP_DEPRECATION_MSG, key)


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
    base = [1, 2, 4, 8, 16] if is_moe else [1, 2, 4, 8]
    prefill_cfg: dict[str, list[int]] = {
        "num_gpu_per_worker": base,
        "tp_list": base,
        "pp_list": [1, 2, 4, 8] if should_enable_pp else [1],
        "dp_list": [1],
        "moe_tp_list": [1],
        "moe_ep_list": base if is_moe else [1],
    }
    decode_cfg: dict[str, list[int]] = {
        "num_gpu_per_worker": base,
        "tp_list": base,
        "pp_list": [1, 2, 4, 8] if should_enable_pp else [1],
        "dp_list": base if is_moe else [1],
        "moe_tp_list": [1],
        "moe_ep_list": base if is_moe else [1],
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
            x = [1, 2, 4, 8, 16]
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
            x = [1, 2, 4, 8, 16]
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
            x = [1, 2, 4, 8, 16]
            for cfg in (prefill_cfg, decode_cfg):
                cfg["num_gpu_per_worker"] = x
                cfg["tp_list"] = x
                cfg["pp_list"] = x if should_enable_pp else [1]
                cfg["dp_list"] = x
                cfg["moe_tp_list"] = [1]
                cfg["moe_ep_list"] = [1, 2, 4, 8, 16]
        else:
            x = [1, 2, 4, 8, 16]
            prefill_cfg = {
                "num_gpu_per_worker": x,
                "tp_list": x,
                "pp_list": x if should_enable_pp else [1],
                "dp_list": x,
                "moe_tp_list": x,
                "moe_ep_list": [1, 2, 4, 8, 16],
            }
            decode_cfg = {
                "num_gpu_per_worker": x,
                "tp_list": x,
                "pp_list": x if should_enable_pp else [1],
                "dp_list": x,
                "moe_tp_list": x,
                "moe_ep_list": [1, 2, 4, 8, 16],
            }
    elif backend_name == "vllm":
        x = [1, 2, 4, 8, 16]
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
# AFD search space helpers (migrated from legacy sdk.task)
# ---------------------------------------------------------------------------


def _is_valid_afd_moe_ep_size(f_moe_ep_size: int, tp_f: int, num_experts: int) -> bool:
    return (
        f_moe_ep_size >= 1
        and f_moe_ep_size <= tp_f
        and tp_f % f_moe_ep_size == 0
        and (num_experts <= 0 or (f_moe_ep_size <= num_experts and num_experts % f_moe_ep_size == 0))
    )


def build_afd_parallel_lists(
    total_gpus: int,
    gpus_per_node: int,
    is_moe: bool,
    num_experts: int = 0,
    *,
    search_config: Mapping[str, Any] | None = None,
) -> list[tuple[int, int, int, int, int, str]]:
    """Enumerate AFD candidate topologies for the default-mode sweep.

    Returns a list of ``(n_a_nodes, n_f_nodes, tp_a, f_moe_ep_size,
    num_microbatches, pipeline_model)`` tuples.

    Candidates satisfy the hard constraints:
    * GPU budget -- ``(n_a_nodes + n_f_nodes) * gpus_per_node <= total_gpus``
    * ``tp_a`` divides ``gpus_per_node``
    * ``f_moe_ep_size`` divides ``tp_f = n_f_nodes * gpus_per_node``
    * ``num_experts % f_moe_ep_size == 0`` when known
    """
    if gpus_per_node < 1 or total_gpus < 1:
        return []

    total_nodes = total_gpus // gpus_per_node
    if total_nodes < 2:
        return []

    search = dict(search_config or {})
    tp_a_candidates = search.get("tp_a_list")
    if tp_a_candidates is None:
        tp_a_candidates = sorted({d for d in (1, 2, 4, gpus_per_node) if d >= 1 and gpus_per_node % d == 0})
    else:
        tp_a_candidates = sorted({int(tp) for tp in tp_a_candidates if gpus_per_node % int(tp) == 0})

    microbatch_candidates = list(search.get("microbatch_list") or [2, 3, 4])
    pipeline_candidates = list(search.get("pipeline_model_list") or ["optimistic", "conservative"])
    f_moe_ep_size_list = search.get("f_moe_ep_size_list")
    max_af_ratio = float(search.get("max_af_ratio", 4.0))
    max_candidates = int(search.get("max_candidates", 10_000))
    candidate_overflow = str(search.get("candidate_overflow", "error"))
    if max_candidates < 1:
        raise ValueError(f"afd_config.search.max_candidates must be >= 1, got {max_candidates}.")
    if candidate_overflow not in {"error", "truncate"}:
        raise ValueError("afd_config.search.candidate_overflow must be 'error' or 'truncate'.")

    candidates: list[tuple[int, int, int, int, int, str]] = []
    for n_a_nodes in range(1, total_nodes):
        for n_f_nodes in range(1, total_nodes - n_a_nodes + 1):
            if n_a_nodes / n_f_nodes > max_af_ratio:
                continue
            tp_f = n_f_nodes * gpus_per_node
            if is_moe:
                raw_ep_candidates = f_moe_ep_size_list or [1, 2, "n_f_nodes", "tp_f"]
                resolved_ep_candidates: set[int] = set()
                for ep in raw_ep_candidates:
                    if ep == "n_f_nodes":
                        resolved_ep_candidates.add(n_f_nodes)
                    elif ep == "tp_f":
                        resolved_ep_candidates.add(tp_f)
                    else:
                        resolved_ep_candidates.add(int(ep))
                ep_candidates = sorted(
                    ep for ep in resolved_ep_candidates if _is_valid_afd_moe_ep_size(ep, tp_f, num_experts)
                )
            else:
                ep_candidates = [1]
            for tp_a in tp_a_candidates:
                for f_moe_ep_size in ep_candidates:
                    for num_microbatches in microbatch_candidates:
                        for pipeline_model in pipeline_candidates:
                            # Skip optimistic + mb < 3: the K=3 pipeline
                            # requires num_microbatches >= 2 + t_c/max(t_a,
                            # t_f), which is >= 3 whenever t_c > 0 (the
                            # normal case).  mb=2 + optimistic always degrades
                            # to conservative, producing a duplicate of the
                            # mb=2 + conservative candidate and a flood of
                            # per-stride warnings.
                            if pipeline_model == "optimistic" and num_microbatches < 3:
                                continue
                            candidates.append(
                                (n_a_nodes, n_f_nodes, tp_a, f_moe_ep_size, num_microbatches, pipeline_model)
                            )
    if len(candidates) > max_candidates:
        message = f"AFD default sweep produced {len(candidates)} candidates, exceeding max_candidates={max_candidates}."
        if candidate_overflow == "truncate":
            logger.warning("%s Truncating deterministically to the first %d candidates.", message, max_candidates)
            candidates = candidates[:max_candidates]
        else:
            raise ValueError(
                f"{message} Narrow the AFD search, increase afd_max_candidates "
                "(--afd-max-candidates), or explicitly set afd_candidate_overflow='truncate' "
                "(--afd-candidate-overflow truncate)."
            )
    logger.info("AFD default sweep candidate count: %d", len(candidates))
    return candidates


def _lookup_num_gpus_per_node(system_name: str) -> int | None:
    """Best-effort lookup of ``num_gpus_per_node`` from a system's yaml spec."""
    import os

    from aiconfigurator.sdk.perf_database import get_systems_paths

    for systems_root in get_systems_paths():
        yaml_path = os.path.join(systems_root, f"{system_name}.yaml")
        if not os.path.isfile(yaml_path):
            continue
        try:
            import yaml as _yaml

            with open(yaml_path) as fh:
                spec = _yaml.safe_load(fh) or {}
        except Exception:
            logger.debug("Could not read system yaml at %s", yaml_path, exc_info=True)
            continue
        node = spec.get("node") if isinstance(spec, dict) else None
        if isinstance(node, dict) and isinstance(node.get("num_gpus_per_node"), int):
            return int(node["num_gpus_per_node"])
    return None


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
    serving_mode: Literal["agg", "disagg", "afd"] = "agg"
    isl: int = 4000
    osl: int = 1000
    prefix: int = 0
    # Multimodal image inputs (folded into the effective ISL by RuntimeConfig).
    image_height: int = 0
    image_width: int = 0
    num_images_per_request: int = 1
    # Vision encoder data parallelism (ModelConfig default).
    enable_encoder_dp: bool = True
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
    # Forward-pass modeling mode: "op_level" (default) or "fpm" (whole-model
    # forward op backed by collected fpm_forward data). Threaded into every
    # ModelConfig this task builds; validated in models.get_model.
    forward_model: str = "op_level"

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

    # ====== 10. AFD config (serving_mode='afd') ======
    afd_total_gpus: int | None = None  # AFD GPU budget (defaults to total_gpus)
    afd_combined_with_pd: bool = True
    afd_comm_overhead_factor: float = 1.0
    afd_boundary_on_attn: bool = True
    afd_total_batch_size: int | None = None
    # Per-A-worker ceiling for the automatic A-batch search.
    afd_max_a_batch_size: int = 1024
    # AFD pinned topology (single-point mode: skip sweep, run AFDInferenceSession)
    afd_n_a_nodes: int | None = None
    afd_n_f_nodes: int | None = None
    afd_tp_a: int | None = None
    afd_a_batch_size: int | None = None
    # AFD search space config (used only when topology is NOT pinned)
    afd_tp_a_candidates: list[int] | None = None
    afd_microbatch_candidates: list[int] | None = None
    afd_pipeline_model_candidates: list[str] | None = None
    afd_f_moe_ep_size_candidates: list[int | str] | None = None
    afd_max_af_ratio: float = 4.0
    afd_max_candidates: int = 10_000
    afd_candidate_overflow: str = "error"
    # AFD prefill search config (used when combined_with_pd=True)
    afd_prefill_batch_size_list: list[int] | None = None
    afd_prefill_max_candidates: int = 256
    afd_prefill_candidate_overflow: str = "error"
    afd_max_prefill_gpus: int | None = None
    afd_max_prefill_workers: int | None = None
    # AFD calibration
    afd_prefill_degradation: float | None = None
    afd_decode_degradation: float | None = None
    afd_ttft_correction_factor: float | None = None
    afd_decode_latency_correction: float = 1.0

    # ====== 11. Internal — resolved in __post_init__ ======
    _is_moe: bool = field(default=False, repr=False, init=False)
    _model_family: str = field(default="", repr=False, init=False)
    _raw_config: dict = field(default_factory=dict, repr=False, init=False)
    _architecture: str = field(default="", repr=False, init=False)
    _num_experts: int = field(default=0, repr=False, init=False)
    _afd_parallel_config_list: list = field(default_factory=list, repr=False, init=False)
    _afd_gpus_per_node: int = field(default=8, repr=False, init=False)
    _afd_topology_pinned: bool = field(default=False, repr=False, init=False)
    # Which fmha_quant_mode values came from an explicit field (per role) --
    # handed from _resolve_quant_modes to _apply_fmha_data_fallback.
    _fmha_explicit: dict = field(default_factory=dict, repr=False, init=False)
    # role -> large-EP coverage (see _large_ep_coverage). Depends on the model /
    # system / backend / MoE quant mode only, never on the candidate lists, so
    # it survives post-construction edits to those.
    _large_ep_coverage_cache: dict = field(default_factory=dict, repr=False, init=False)

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
        return self.model_path if self.serving_mode in ("agg", "afd") else self.prefill_model_path

    @property
    def primary_system_name(self) -> str:
        return self.system_name if self.serving_mode in ("agg", "afd") else self.prefill_system_name

    @property
    def primary_backend_name(self) -> str:
        return self.backend_name if self.serving_mode in ("agg", "afd") else self.prefill_backend_name

    @property
    def primary_backend_version(self) -> str | None:
        return self.backend_version if self.serving_mode in ("agg", "afd") else self.prefill_backend_version

    @property
    def effective_total_gpus(self) -> int | None:
        """Return the GPU budget used by the active serving mode."""
        if self.serving_mode == "afd" and self.afd_total_gpus is not None:
            return self.afd_total_gpus
        return self.total_gpus

    # =====================================================================
    # __post_init__
    # =====================================================================

    def __post_init__(self) -> None:
        # Deprecation surface first, on the raw constructor values (both the
        # Task(...) and from_yaml paths land here): a truthy legacy flag or an
        # explicit user moe_backend="deepep_moe" warns once per key per
        # process. Checked BEFORE _normalize_wideep_moe_backend so the
        # retained enable_wideep -> deepep_moe normalization never triggers a
        # second warning for a value the user did not write.
        for _key in _DEPRECATED_LARGE_EP_KEYS:
            if getattr(self, _key):
                _warn_large_ep_flag(_key)
        if self.moe_backend == "deepep_moe":
            _warn_large_ep_flag("moe_backend=deepep_moe")
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
        # The search space is resolved BEFORE the data-driven FMHA fallback:
        # the attention-op keys the fallback consults depend on whether any
        # enumerated tuple is large-EP, which is only known once the candidate
        # lists exist. The fallback in turn touches nothing the search space
        # reads (it only rewrites fmha_quant_mode).
        self._resolve_search_space()
        self._apply_fmha_data_fallback()
        self._validate_megamoe_backend_support()

    def _normalize_wideep_moe_backend(self) -> None:
        """Field-level compat mapping: ``enable_wideep`` still spells the
        deprecated ``moe_backend="deepep_moe"`` (mirrors v1 ``__init__``), so a
        task round-trips through ``to_dict`` / effective-config reports exactly
        as before.

        The VALUE is inert in resolution as of the coverage-driven large-EP
        wiring: ``build_model_config`` never forwards ``deepep_moe`` (it used to
        select the wideEP model classes and the wideep MoE compute tables),
        ``enumerate_parallel_config`` only special-cases megamoe, and large-EP
        participation comes from perf-data coverage. What it still does is pick
        the fused EP-only sglang LADDER, exactly as on the flag-off path."""
        if self.moe_backend is not None:
            return
        wideep = (
            self.enable_wideep
            if self.serving_mode in ("agg", "afd")
            else (self.prefill_enable_wideep or self.decode_enable_wideep)
        )
        if wideep:
            self.moe_backend = "deepep_moe"

    def _validate_megamoe_backend_support(self) -> None:
        """v1 _validate_megamoe_backend_support: megamoe is sglang + DeepSeek-V4-Pro + Blackwell only."""
        if self.moe_backend != "megamoe":
            return
        roles = ["agg"] if self.serving_mode in ("agg", "afd") else ["prefill", "decode"]
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
        roles = ["agg"] if self.serving_mode in ("agg", "afd") else ["prefill", "decode"]
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

        Exception: top-level ``enable_wideep=True`` no longer raises. The flag
        is deprecated-and-ignored everywhere (disagg normalization reads only
        the prefill_/decode_ flags), so the leakage is harmless; the
        deprecation warning already fired from ``__post_init__``.
        """
        if self.serving_mode != "disagg":
            return
        leakage = []
        if self.model_path:
            leakage.append("model_path")
        if self.system_name:
            leakage.append("system_name")
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
        primary = self.model_path if self.serving_mode in ("agg", "afd") else self.prefill_model_path
        if not primary:
            return
        info = get_model_config_from_model_path(primary)
        self._raw_config = info.get("raw_config", {})
        self._architecture = info["architecture"]
        self._model_family = get_model_family(primary)
        self._is_moe = check_is_moe(primary)
        self._num_experts = int(info.get("num_experts", 0) or 0)

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
            elif self._architecture in common.DSPARK_ARCHITECTURES:
                logger.info(
                    "nextn='auto' cannot resolve a DSPARK block size (the draft is a "
                    "standalone model, not checkpoint MTP layers); modeling WITHOUT "
                    "speculative decoding. Pass an explicit nextn (block size, e.g. 7) "
                    "and nextn_accepted to model DSPARK."
                )
            else:
                logger.info(
                    "nextn='auto': checkpoint ships no MTP layers (num_nextn_predict_layers absent or 0); "
                    "modeling WITHOUT speculative decoding."
                )
            self.nextn = resolved
        if self.nextn > 0:
            # Range/required-ness already validated in __post_init__ (validate_nextn).
            if self._architecture in common.DSPARK_ARCHITECTURES:
                logger.info(
                    "nextn=%d is the DSPARK speculative block size (draft tokens per "
                    "step, served by a standalone trained draft model); the "
                    "checkpoint's num_nextn_predict_layers does not apply to this "
                    "architecture.",
                    self.nextn,
                )
            elif hf_nextn is not None and self.nextn != hf_nextn:
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

        if self.serving_mode in ("agg", "afd"):
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
        roles = ["agg"] if self.serving_mode in ("agg", "afd") else ["prefill", "decode"]
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
                    if arch_mode is None:
                        arch_mode = resolve_kimi_k3_moe_arch_mode(
                            self._role_attr(role, "model_path"),
                            self._role_attr(role, "system_name"),
                            self._role_attr(role, "backend_name"),
                        )
                    if arch_mode is not None:
                        from_hf = arch_mode
                fallback = _QUANT_FALLBACKS[key]

                if explicit is not None:
                    continue
                resolved = from_hf if from_hf is not None else fallback
                self._set_role_attr(role, key, resolved)

        self._fmha_explicit = fmha_explicit

    def _apply_fmha_data_fallback(self) -> None:
        """Data-driven FMHA resolution (second half of quant resolution).

        Split out of ``_resolve_quant_modes`` because it consults the role's
        attention-op keys, which depend on whether any enumerated parallel
        tuple is large-EP -- i.e. on the search space, which is resolved in
        between (see ``__post_init__``).
        """
        roles = ["agg"] if self.serving_mode in ("agg", "afd") else ["prefill", "decode"]
        fmha_explicit = self._fmha_explicit

        # nvfp4 → nvfp4_wo on non-Blackwell: no native FP4 tensor cores, so all
        # runtimes dequantize to BF16 before the MMA. The transfer ladder models
        # the Marlin-class memory savings via the (0.5625, 1) util-level entry.
        for role in roles:
            system = self._role_attr(role, "system_name")
            if not is_blackwell_system(system):
                if self._role_attr(role, "gemm_quant_mode") == common.GEMMQuantMode.nvfp4:
                    self._set_role_attr(role, "gemm_quant_mode", common.GEMMQuantMode.nvfp4_wo)
                if self._role_attr(role, "moe_quant_mode") == common.MoEQuantMode.nvfp4:
                    self._set_role_attr(role, "moe_quant_mode", common.MoEQuantMode.nvfp4_wo)

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
        # A task whose tuples span both regimes (fused AND large-EP) reaches TWO
        # context tables, and fmha_quant_mode is one per-task value: score both
        # candidate modes over the reachable tables and keep the inferred one
        # unless bfloat16 serves strictly more of them. Single-regime tasks --
        # every task before large EP became per-tuple -- score one table and
        # behave exactly as before.
        for role in roles:
            if role == "decode":
                continue
            if fmha_explicit.get(role, False):
                continue
            if self._role_attr(role, "fmha_quant_mode") != common.FMHAQuantMode.fp8:
                continue
            ctx_ops = [ctx for ctx, _gen in self._reachable_attention_op_keys(role)]
            supported = {op: self._context_fmha_supported_modes(role, op) for op in ctx_ops}
            informative = [modes for modes in supported.values() if modes]
            if not informative:
                continue  # no DB / no table to consult -> keep fp8
            fp8_hits = sum(common.FMHAQuantMode.fp8.name in modes for modes in informative)
            bf16_hits = sum(common.FMHAQuantMode.bfloat16.name in modes for modes in informative)
            if fp8_hits >= bf16_hits or bf16_hits == 0:
                continue  # fp8 data present, or nothing safe to fall back to
            self._set_role_attr(role, "fmha_quant_mode", common.FMHAQuantMode.bfloat16)
            missing = [op for op, modes in supported.items() if modes and common.FMHAQuantMode.fp8.name not in modes]
            field = "fmha_quant_mode" if self.serving_mode == "agg" else f"{role}_fmha_quant_mode"
            logger.warning(
                f"{role} fmha_quant_mode=fp8 (inferred from the model checkpoint) has no "
                f"{', '.join(repr(op) for op in missing)} perf data for "
                f"system={self._role_attr(role, 'system_name')!r}, "
                f"backend={self._role_attr(role, 'backend_name')!r}, "
                f"version={self._role_attr(role, 'backend_version')!r}; falling back to bfloat16 "
                f"FMHA data. Predictions are conservative if the deployed engine runs fp8 FMHA; "
                f"set {field} explicitly to override."
            )

    def _reachable_attention_op_keys(self, role: str) -> list[tuple[str, str]]:
        """(context_op, generation_op) key pairs this role's tuples can reach.

        Attention op keys are per-REGIME (the large-EP graphs query the wideep /
        granular MLA tables, the fused graphs the module ones), and large EP is
        a per-tuple property now, so one task can reach both: the fused pair
        when any enumerated tuple stays fused, the large-EP pair when any tuple
        resolves a comm backend. Both consumers are per-task data-availability
        checks -- the resolve-time FMHA fallback and ``_check_role_against_db``
        -- and both treat the list as "any of these tables can serve this task":
        a per-tuple gap is pruned by the sweep, not fatal here. Fused first, so
        the universally-reachable regime leads the diagnostics. Mapping lives in
        ``models.attention_op_keys``."""
        # AFD partitions the aggregate model across its A/F topology and does
        # not enumerate the standard agg TP/DP/EP candidate lists. It also
        # never assigns the standard per-tuple MoE comm backend, so its
        # attention surface is the fused aggregate one.
        if self.serving_mode == "afd" and role == "agg":
            return [attention_op_keys(self._model_family, self.backend_name, False)]

        regimes: set[bool] = set()
        for tup in self.iter_parallel(role):
            regimes.add(self._resolve_moe_comm_backend(role, tup) is not None)
            if len(regimes) == 2:
                break
        backend_name = self._role_attr(role, "backend_name")
        return [attention_op_keys(self._model_family, backend_name, large) for large in sorted(regimes or {False})]

    def _attention_op_keys(self, role: str) -> tuple[str, str]:
        """The role's primary (context_op, generation_op) keys: the large-EP
        flavor when any enumerated tuple is large-EP, else the fused one.

        Kept for callers that need ONE answer (diagnostics / single-op probes);
        capability checks use :meth:`_reachable_attention_op_keys` instead."""
        return self._reachable_attention_op_keys(role)[-1]

    # =====================================================================
    # Large-EP coverage (spec sections 4.4.3 / 4.5) -- no flag selects it
    # =====================================================================

    def _role_phases(self, role: str) -> tuple[str, ...]:
        """Inference phases a role runs: agg does both, prefill/decode one."""
        if role == "prefill":
            return ("context",)
        if role == "decode":
            return ("generation",)
        return ("context", "generation")

    def _required_large_ep_phases(self, role: str) -> tuple[str, ...]:
        """Extra phase-coverage requirement large EP puts on a role.

        Returns ``("context",)`` for the single-phase roles (prefill/decode)
        and the full phase pair for agg; callers union it with
        ``_role_phases(role)``, so the effective requirement is always the
        phases the role runs PLUS context. Context is required even for a
        decode worker because its model object holds the whole graph and the
        memory model sizes its weights from ``model.context_ops``
        (``base_backend._get_memory_usage``) — a fused context span under a
        large-EP decode step would be priced with the wrong (÷tp shared
        expert, router) weights."""
        return ("context",) if role != "agg" else self._role_phases(role)

    def _large_ep_coverage(self, role: str) -> dict[str, dict[str, set[int]]]:
        """``{phase: {comm_backend: {ep_size, ...}}}`` explorable with large EP.

        Per spec section 4.5, an EP size is explorable for a phase when its
        comm backend carries dispatch+combine rows for the model shape at
        ``(ep, nodes_for(ep, gpus_per_node))`` (topology check against THIS
        system), the backend's registry feasibility rules admit the config,
        and the EP expert-compute table covers the shape under the role's MoE
        quant mode for that phase. BOTH phases are probed for every role: a
        disagg worker only runs one of them, but its model object holds the
        whole graph and the memory model sizes weights off the context ops, so
        the phase the role does not run must be emitted in the same regime (the
        deleted wideEP classes built both phases off one flag). Empty sets and
        backends are dropped.

        Backends are kept in registry-insertion order (the tie-break when two
        backends cover the same EP); the caller picks the first one covering
        the tuple's EP.

        Never raises on missing DATA: an absent model shape, system spec,
        database or table yields ``{}`` -- the fused path then serves every
        tuple. A caller BUG still raises: a str-typed ``moe_quant_mode`` is a
        ``TypeError``, not empty coverage (it would silently miss every
        enum-keyed compute row and disable large-EP exploration).
        """
        cached = self._large_ep_coverage_cache.get(role)
        if cached is not None:
            return cached
        coverage = self._compute_large_ep_coverage(role)
        self._large_ep_coverage_cache[role] = coverage
        return coverage

    def _compute_large_ep_coverage(self, role: str) -> dict[str, dict[str, set[int]]]:
        if not self._is_moe or self._model_family not in LARGE_EP_READY_FAMILIES:
            return {}
        model_path = self._role_attr(role, "model_path")
        backend_name = self._role_attr(role, "backend_name")
        system_name = self._role_attr(role, "system_name")
        if not model_path:
            return {}
        try:
            shape = MoEBlockShape.from_model_info(_get_model_info(model_path))
        except Exception as exc:  # not a MoE checkpoint / unparsable config
            logger.debug("large-EP coverage: no MoE shape for %s: %s", model_path, exc)
            return {}

        spec = load_system_spec(system_name)
        gpus_per_node = int(spec.get("node", {}).get("num_gpus_per_node", 0) or 0)
        sm_version = spec.get("gpu", {}).get("sm_version")
        sm_version = int(sm_version) if sm_version is not None else None
        database = self._try_load_role_database(role)
        # The probes are a PerfDatabase contract; a database object without them
        # (a lightweight double injected by a caller) carries no coverage
        # information, which is the same answer as an absent table.
        a2a_probe = getattr(database, "moe_a2a_coverage", None)
        compute_probe = getattr(database, "moe_expert_compute_coverage", None)
        coverage: dict[str, dict[str, set[int]]] = {}
        if gpus_per_node and a2a_probe is not None and compute_probe is not None:
            a2a = a2a_probe(shape.hidden_size, shape.topk, shape.num_experts)
            quant_mode = self._role_attr(role, "moe_quant_mode")
            if quant_mode is not None and not isinstance(quant_mode, common.MoEQuantMode):
                # The compute table is keyed by MoEQuantMode members; any
                # other type (str, int, a sibling enum like
                # GEMMQuantMode.bfloat16) would miss every key and silently
                # report empty coverage, disabling large-EP exploration.
                raise TypeError(
                    f"moe_quant_mode must be a common.MoEQuantMode member, got "
                    f"{type(quant_mode).__name__} {quant_mode!r} "
                )
            for phase in ("context", "generation"):
                compute = compute_probe(
                    shape.hidden_size, shape.moe_inter_size, shape.topk, shape.num_experts, quant_mode, phase
                )
                per_backend: dict[str, set[int]] = {}
                for name, backend_spec in MOE_A2A_BACKENDS.items():
                    if backend_name not in backend_spec.frameworks or phase not in backend_spec.inference_phases:
                        continue
                    eps = {
                        ep
                        for ep, node_num in a2a.get(name, ())
                        if node_num == nodes_for(ep, gpus_per_node)
                        and backend_spec.feasible(
                            topk=shape.topk,
                            num_experts=shape.num_experts,
                            moe_tp_size=1,
                            moe_ep_size=ep,
                            sm_version=sm_version,
                        )
                    } & compute
                    if eps:
                        per_backend[name] = eps
                if per_backend:
                    coverage[phase] = per_backend

        if not coverage:
            log_key = (model_path, system_name, backend_name, self._role_attr(role, "backend_version"))
            if log_key not in _LARGE_EP_EMPTY_COVERAGE_LOGGED:
                _LARGE_EP_EMPTY_COVERAGE_LOGGED.add(log_key)
                logger.info(
                    "large-EP exploration is OFF for %s on %s/%s: no MoE all-to-all + EP-compute "
                    "coverage for this model shape (hidden=%d, topk=%d, experts=%d) under "
                    "moe_quant_mode=%s. Run the moe_a2a and moe_ep collectors for this "
                    "model/system to enable it; the fused (small-EP) path is unaffected.",
                    model_path,
                    system_name,
                    backend_name,
                    shape.hidden_size,
                    shape.topk,
                    shape.num_experts,
                    getattr(self._role_attr(role, "moe_quant_mode"), "name", None),
                )
        return coverage

    def _resolve_moe_comm_backend(self, role: str, parallel_tuple) -> dict[str, str] | None:
        """Per-phase MoE comm backend for ONE parallel tuple, or None (fused).

        Large EP needs ``moe_tp == 1`` and ``moe_ep > 1``; the phases the ROLE
        RUNS must cover that EP, otherwise the whole tuple stays fused (an agg
        worker cannot run context large-EP and generation fused -- it is one
        model graph). The CONTEXT phase is required on top of that for every
        role, decode included: a worker's model object holds the whole graph
        and the memory model sizes weights off ``model.context_ops``
        (``base_backend._get_memory_usage``), so a decode worker whose context
        span fell back to the fused emission would be priced with the fused
        shared-expert/router weights while its decode step runs large EP.
        Generation-only coverage therefore keeps the tuple fused, with one
        warning naming the asymmetry (shipped data covers both phases; this
        fires the day a collection lands one phase ahead of the other).

        The per-phase backend is the first registry entry covering this EP, so
        a phase served by two backends resolves deterministically (e.g. trtllm
        nvlink_two_sided before nvlink_one_sided) while an EP only one of them
        collected still resolves to that one.
        """
        _tp, _pp, _dp, moe_tp, moe_ep, _cp = tuple(parallel_tuple)
        if moe_tp != 1 or moe_ep <= 1:
            return None
        if _dp <= 1 and self._role_attr(role, "backend_name") == "trtllm":
            # TRT-LLM large-EP requires attention DP (model construction
            # enforces attention_dp_size > 1 in validate_trtllm_large_ep);
            # a dp=1 TEP tuple must stay on the fused path instead of
            # resolving a comm backend and then failing to build.
            return None
        coverage = self._large_ep_coverage(role)
        if not coverage:
            return None
        resolved: dict[str, str] = {}
        for phase in ("context", "generation"):
            for name, eps in coverage.get(phase, {}).items():
                if moe_ep in eps:
                    resolved[phase] = name
                    break
        required = set(self._role_phases(role)) | set(self._required_large_ep_phases(role))
        missing = required - set(resolved)
        if missing:
            if missing == {"context"}:
                self._warn_context_coverage_gap(role, moe_ep)
            return None
        return resolved

    def _warn_context_coverage_gap(self, role: str, moe_ep: int) -> None:
        """One-shot warning: the role's own phase is covered but context is not."""
        key = (
            self._role_attr(role, "model_path"),
            self._role_attr(role, "system_name"),
            self._role_attr(role, "backend_name"),
            self._role_attr(role, "backend_version"),
        )
        if key in _LARGE_EP_ASYMMETRIC_COVERAGE_WARNED:
            return
        _LARGE_EP_ASYMMETRIC_COVERAGE_WARNED.add(key)
        logger.warning(
            "large-EP coverage for %s on %s/%s is asymmetric: the %s phase is collected at "
            "moe_ep=%d but the context phase is not. Keeping those configs on the fused path -- "
            "a worker's weights are sized from its context ops, so a context-fused / "
            "generation-large-EP graph would be mis-priced. Collect the missing context rows "
            "(moe_a2a + moe_ep) to enable them.",
            key[0],
            key[1],
            key[2],
            " / ".join(self._role_phases(role)),
            moe_ep,
        )

    def _large_ep_eps(self, role: str) -> set[int]:
        """EP sizes this role could run large-EP: covered in every phase it runs
        AND in the context phase that sizes its weights (see
        ``_required_large_ep_phases``)."""
        coverage = self._large_ep_coverage(role)
        eps: set[int] | None = None
        for phase in set(self._role_phases(role)) | set(self._required_large_ep_phases(role)):
            per_backend = coverage.get(phase, {})
            phase_eps = set().union(*per_backend.values()) if per_backend else set()
            eps = phase_eps if eps is None else eps & phase_eps
        return eps or set()

    def _role_has_large_ep_tuple(self, role: str) -> bool:
        """Whether any enumerated tuple for this role resolves a comm backend."""
        if not self._large_ep_eps(role):
            return False
        return any(self._resolve_moe_comm_backend(role, tup) for tup in self.iter_parallel(role))

    def _num_gpus_per_node(self, role: str) -> int | None:
        """System node width (hardware fact) for this role, None when unknown.

        Large-EP model construction raises when it is missing rather than
        assuming 8 (``models.helpers.large_ep_gpus_per_node``)."""
        spec = load_system_spec(self._role_attr(role, "system_name"))
        return int(spec.get("node", {}).get("num_gpus_per_node", 0) or 0) or None

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

    def _context_fmha_supported_modes(self, role: str, ctx_op: str | None = None) -> list[str]:
        """FMHA modes with perf data for one fmha-keyed context-attention op
        (default: this role's primary one), jointly with the role's resolved
        kv-cache mode (an fmha slice that exists only under a different kv dtype
        cannot serve this role's queries).  Returns [] when the DB (or the op's
        table) is unavailable, meaning "no information" -- callers must not read
        that as "nothing supported"."""
        from aiconfigurator.sdk.perf_database import context_fmha_supported_modes

        database = self._try_load_role_database(role)
        if database is None:
            return []
        if ctx_op is None:
            ctx_op = self._attention_op_keys(role)[0]
        if ctx_op == "context_mla" and self._attention_quant_identity_mixed(role):
            # Mixed-projection checkpoints (e.g. V3.1-NVFP4: BF16 q/kv + NVFP4
            # o_proj) bypass the profiled MLA-module row — no single-gemm_type
            # module identity matches — so fmha availability must be judged on
            # the granular table alone: a module-only fp8 slice cannot serve
            # these models' queries.
            ctx_op = "context_mla_granular"
        return context_fmha_supported_modes(
            database,
            ctx_op,
            self._role_attr(role, "kvcache_quant_mode"),
        )

    def _attention_quant_identity_mixed(self, role: str) -> bool:
        """Whether the checkpoint's attention projections diverge in dtype
        (some excluded from quantization, some not) under this role's gemm
        mode — the condition that makes DeepSeek-family models bypass the
        profiled MLA-module row (see DeepSeekModel.__init__)."""
        from aiconfigurator_core.sdk.models.helpers import attention_projection_exclusions

        if self._role_attr(role, "gemm_quant_mode") == common.GEMMQuantMode.bfloat16:
            return False  # everything runs BF16 -> uniform identity
        excl = attention_projection_exclusions(self._raw_config) & {"q", "kv", "o"}
        return bool(excl) and excl != {"q", "kv", "o"}

    def _resolve_search_space(self) -> None:
        if self.serving_mode == "afd":
            self._resolve_afd_search()
            return
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
        Blackwell with total_gpus>=16 get extra PP=2 / TP=8 / 16-GPU configs.

        The wideep / deepep_moe exclusions are gone with the flags: large EP is
        a per-tuple property now, and a PP>1 tuple simply finds no (ep, node)
        comm data and stays fused. MegaMoE keeps its own parallel lists, so it
        still opts out."""
        if not self._is_moe or self._model_family not in _LARGE_PIPELINE_PARALLEL_MODEL_FAMILIES:
            return False
        if self.serving_mode == "agg":
            systems = [self.system_name]
        else:
            systems = [self.prefill_system_name, self.decode_system_name]
        if self.moe_backend == "megamoe":
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
            return
        if self.backend_name not in ("trtllm", "sglang", "vllm"):
            raise ValueError(f"Unsupported backend: {self.backend_name}")

        # Fused defaults: what a task without large-EP data explores today.
        if self.backend_name == "sglang":
            x = [1, 2, 4, 8, 16] if self._is_moe else [1, 2, 4, 8]
            fused = {
                "num_gpu": x,
                "tp": x,
                "pp": [1],
                "dp": x,
                # Intra-node DeepEP (ep 1-8, NVLink) is EP-only; standard comm
                # (fused_moe + allgather/RS) also explores MoE TP.
                "moe_tp": [1] if self.moe_backend == "deepep_moe" else x,
                "moe_ep": x,
            }
        else:
            x = [1, 2, 4, 8, 16] if self._is_moe else [1, 2, 4, 8]
            fused = {"num_gpu": x, "tp": x, "pp": [1], "dp": x, "moe_tp": x, "moe_ep": x}

        # Large-EP ladder, offered when the perf data covers this model shape on
        # this system (no flag): the single task explores BOTH regimes, so the
        # lists are the union of the fused defaults and the multi-node ladder.
        wide = None
        if self.backend_name == "trtllm":
            wide = {
                "num_gpu": [2, 4, 8, 16, 32, 64],
                "tp": [1, 2, 4, 8],
                "pp": [1],
                "dp": [2, 4, 8, 16, 32, 64],
                "moe_tp": [1],
                "moe_ep": [2, 4, 8, 16, 32, 64],
            }
        elif self.backend_name == "sglang":
            wide = {
                "num_gpu": [8, 16, 32, 64],
                "tp": [1, 2, 4, 8],
                "pp": [1],
                "dp": [1, 2, 4, 8, 16, 32, 64],
                "moe_tp": [1],
                "moe_ep": [8, 16, 32, 64],
            }
        elif self.backend_name == "vllm":
            # vLLM ships no static multi-node ladder (data-only enablement);
            # derive it from the covered EP sizes so coverage that lands is
            # actually explorable. Pure-EP tuples need num_gpu/dp/moe_ep
            # candidates at each covered EP (tp=1 => dp == ep by the width
            # identity tp*dp*cp == moe_tp*moe_ep).
            eps = sorted(self._large_ep_eps("agg"))
            if eps:
                wide = {
                    "num_gpu": eps,
                    "tp": [1],
                    "pp": [1],
                    "dp": eps,
                    "moe_tp": [1],
                    "moe_ep": eps,
                }
        if wide is not None and self._large_ep_eps("agg"):
            fused = {dim: sorted(set(values) | set(wide[dim])) for dim, values in fused.items()}

        for dim, values in fused.items():
            _set(f"agg_{dim}_candidates", values)

    def _resolve_disagg_search(self) -> None:
        def _lists(wide: bool) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
            return build_disagg_parallel_lists(
                backend_name=self.prefill_backend_name,
                is_moe=self._is_moe,
                prefill_system=self.prefill_system_name,
                decode_system=self.decode_system_name,
                prefill_enable_wideep=wide,
                decode_enable_wideep=wide,
                moe_backend=self.moe_backend,
            )

        # Per-role large-EP participation, from data coverage instead of the
        # flags: a role whose shape/system is covered explores the union of the
        # multi-node ladder and the fused defaults, its partner is unaffected.
        # (The shared builder keys its sglang branch on either role, so the
        # wide lists are fetched with both roles on and picked per role here.)
        fused_cfgs = dict(zip(("prefill", "decode"), _lists(wide=False), strict=True))
        large_ep = {role: bool(self._large_ep_eps(role)) for role in ("prefill", "decode")}
        if any(large_ep.values()):
            wide_cfgs = dict(zip(("prefill", "decode"), _lists(wide=True), strict=True))
        for role, src in fused_cfgs.items():
            if large_ep[role]:
                src = {dim: sorted(set(values) | set(wide_cfgs[role][dim])) for dim, values in src.items()}
                if self._role_attr(role, "backend_name") == "vllm":
                    # Data-only vLLM enablement: the shared builder has no
                    # vLLM wide branch, so derive the ladder from the covered
                    # EP sizes (see _resolve_agg_search for the identity).
                    eps = sorted(self._large_ep_eps(role))
                    ladder = {
                        "num_gpu_per_worker": eps,
                        "dp_list": eps,
                        "moe_ep_list": eps,
                    }
                    src = {dim: sorted(set(values) | set(ladder.get(dim, []))) for dim, values in src.items()}
            self._fill_role_search(role, src)

        # Replica defaults. Keyed on the resolved CANDIDATES, not on coverage:
        # a user list that pins the search to fused tuples (e.g. moe_ep=[1])
        # keeps the fused replica ladder even on a covered model/system.
        if any(self._role_has_large_ep_tuple(role) for role in ("prefill", "decode")):
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

    def _resolve_afd_search(self) -> None:
        """Resolve AFD search space: enumerate candidate topologies.

        When afd_n_a_nodes/afd_n_f_nodes/afd_tp_a are all set, the topology is
        pinned and no sweep is needed (single-point mode via AFDInferenceSession).
        """
        gpus_per_node = _lookup_num_gpus_per_node(self.system_name)
        if gpus_per_node is None:
            raise ValueError(
                f"Cannot resolve num_gpus_per_node for system '{self.system_name}'; "
                "AFD requires a valid system yaml spec."
            )
        self._afd_gpus_per_node = gpus_per_node

        pinned_fields = {
            "afd_n_a_nodes": self.afd_n_a_nodes,
            "afd_n_f_nodes": self.afd_n_f_nodes,
            "afd_tp_a": self.afd_tp_a,
        }
        pinned_count = sum(value is not None for value in pinned_fields.values())
        if 0 < pinned_count < len(pinned_fields):
            missing = [name for name, value in pinned_fields.items() if value is None]
            raise ValueError(
                f"AFD pinned topology requires afd_n_a_nodes, afd_n_f_nodes, and afd_tp_a together; missing {missing}."
            )

        effective_total_gpus = self.effective_total_gpus
        if effective_total_gpus is None:
            raise ValueError("total_gpus or afd_total_gpus is required for serving_mode='afd'.")

        # Pinned topology: validate it now and skip sweep enumeration.
        if pinned_count == len(pinned_fields):
            if self.afd_n_a_nodes < 1 or self.afd_n_f_nodes < 1:
                raise ValueError("afd_n_a_nodes and afd_n_f_nodes must both be positive.")
            if self.afd_tp_a < 1 or gpus_per_node % self.afd_tp_a != 0:
                raise ValueError(
                    f"afd_tp_a ({self.afd_tp_a}) must be a positive divisor of gpus_per_node ({gpus_per_node})."
                )
            pinned_gpus = (self.afd_n_a_nodes + self.afd_n_f_nodes) * gpus_per_node
            if pinned_gpus > effective_total_gpus:
                raise ValueError(
                    f"AFD pinned topology requires {pinned_gpus} GPUs, exceeding "
                    f"the configured budget of {effective_total_gpus}."
                )
            tp_f = self.afd_n_f_nodes * gpus_per_node
            if self._is_moe and not _is_valid_afd_moe_ep_size(tp_f, tp_f, self._num_experts):
                raise ValueError(
                    f"AFD pinned topology resolves f_moe_ep_size={tp_f}, but the model has "
                    f"{self._num_experts} experts. F-side EP must be a positive divisor of both "
                    "the available F ranks and the model expert count."
                )
            self._afd_topology_pinned = True
            self._afd_parallel_config_list = []
            return

        if effective_total_gpus < 2 * gpus_per_node:
            raise ValueError(
                "The current node-granular AFD topology requires one full A node and one full F node "
                f"(at least 2 nodes, {2 * gpus_per_node} GPUs at {gpus_per_node} GPUs/node); "
                f"got total_gpus={effective_total_gpus}."
            )

        # Obtain num_experts for MoE models
        num_experts = self._num_experts if self._is_moe else 0

        # Build search config from Task fields
        search_config: dict[str, Any] = {}
        if self.afd_tp_a_candidates is not None:
            search_config["tp_a_list"] = self.afd_tp_a_candidates
        if self.afd_microbatch_candidates is not None:
            search_config["microbatch_list"] = self.afd_microbatch_candidates
        if self.afd_pipeline_model_candidates is not None:
            search_config["pipeline_model_list"] = self.afd_pipeline_model_candidates
        if self.afd_f_moe_ep_size_candidates is not None:
            search_config["f_moe_ep_size_list"] = self.afd_f_moe_ep_size_candidates
        search_config["max_af_ratio"] = self.afd_max_af_ratio
        search_config["max_candidates"] = self.afd_max_candidates
        search_config["candidate_overflow"] = self.afd_candidate_overflow

        self._afd_parallel_config_list = build_afd_parallel_lists(
            total_gpus=effective_total_gpus,
            gpus_per_node=gpus_per_node,
            is_moe=self._is_moe,
            num_experts=num_experts,
            search_config=search_config,
        )
        if not self._afd_parallel_config_list:
            raise NoFeasibleConfigError(
                "AFD search produced no valid topology candidates. Check the GPU budget and "
                f"candidate filters (afd_tp_a_candidates={self.afd_tp_a_candidates!r}, "
                f"afd_microbatch_candidates={self.afd_microbatch_candidates!r}, "
                f"afd_pipeline_model_candidates={self.afd_pipeline_model_candidates!r}, "
                f"afd_f_moe_ep_size_candidates={self.afd_f_moe_ep_size_candidates!r})."
            )

        # Also resolve disagg-style prefill parallel lists for combined-with-PD
        if self.afd_combined_with_pd:
            prefill_cfg, _ = build_disagg_parallel_lists(
                backend_name=self.backend_name,
                is_moe=self._is_moe,
                prefill_system=self.system_name,
                decode_system=self.system_name,
                prefill_enable_wideep=self.enable_wideep,
                decode_enable_wideep=self.enable_wideep,
                moe_backend=self.moe_backend,
            )
            # The static prefill pool is an internal view of the same AFD task,
            # so it must use the same backend and feature semantics.
            self.prefill_model_path = self.model_path
            self.prefill_system_name = self.system_name
            self.prefill_backend_name = self.backend_name
            self.prefill_backend_version = self.backend_version
            self.prefill_enable_wideep = self.enable_wideep
            self.prefill_enable_chunked_prefill = self.enable_chunked_prefill
            self.prefill_enable_eplb = self.enable_eplb
            # Store prefill parallel for sweep_afd_kwargs
            candidate_defaults = {
                "prefill_num_gpu_candidates": prefill_cfg["num_gpu_per_worker"],
                "prefill_tp_candidates": prefill_cfg["tp_list"],
                "prefill_pp_candidates": prefill_cfg["pp_list"],
                "prefill_dp_candidates": prefill_cfg["dp_list"],
                "prefill_moe_tp_candidates": prefill_cfg["moe_tp_list"],
                "prefill_moe_ep_candidates": prefill_cfg["moe_ep_list"],
            }
            for attr, values in candidate_defaults.items():
                if getattr(self, attr) is None:
                    setattr(self, attr, values)
            # Propagate resolved agg quant modes to prefill role so
            # build_model_config(role="prefill") inherits the same promotions
            # (e.g. GPT-OSS Blackwell w4a8_mxfp4_mxfp8)
            for qkey in (
                "gemm_quant_mode",
                "moe_quant_mode",
                "kvcache_quant_mode",
                "fmha_quant_mode",
                "comm_quant_mode",
            ):
                if getattr(self, f"prefill_{qkey}") is None:
                    setattr(self, f"prefill_{qkey}", getattr(self, qkey))

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

    def build_model_config(
        self,
        *,
        role: Literal["agg", "prefill", "decode"],
        parallel: ParallelChoice | None = None,
    ) -> config.ModelConfig:
        """Build a ModelConfig for the given role, optionally for one tuple.

        Without ``parallel`` this is the template ``sweep_agg`` /
        ``sweep_disagg`` overwrite tp/pp/dp/moe_tp/moe_ep on per sweep point;
        with it, the MoE comm backend is resolved for that exact tuple (large
        EP is a per-config property -- see ``_resolve_moe_comm_backend``), so
        the model classes build the large-EP graph for the tuples the data
        covers and the fused one for the rest.
        """
        return config.ModelConfig(
            gemm_quant_mode=self._role_attr(role, "gemm_quant_mode"),
            moe_quant_mode=self._role_attr(role, "moe_quant_mode"),
            kvcache_quant_mode=self._role_attr(role, "kvcache_quant_mode"),
            fmha_quant_mode=self._role_attr(role, "fmha_quant_mode"),
            comm_quant_mode=self._role_attr(role, "comm_quant_mode"),
            nextn=self.nextn,
            enable_encoder_dp=self.enable_encoder_dp,
            enable_eplb=self._role_attr(role, "enable_eplb"),
            # attention_backend / wideep_num_slots are shared across roles (Task has no
            # per-role variant) and fed to ModelConfig so get_model selects the MLA
            # attention perf tables (fa3 vs flashinfer) and the EPLB slot count.
            # workload_distribution remains non-configurable in v2 and ModelConfig's
            # default matches v1's.
            #
            # moe_backend="deepep_moe" is NOT forwarded: it used to select both the
            # sglang wideEP model classes and the wideep MoE compute tables for the
            # FUSED op. Large EP is coverage-driven per tuple now, so passing it on
            # would make a fused tuple price itself off the large-EP tables. MegaMoE
            # is a real DeepSeek-V4 kernel selection and passes through.
            moe_backend=self.moe_backend if self.moe_backend != "deepep_moe" else None,
            # None means "unspecified" -> fall back to flashinfer (matches v1 and ModelConfig's default).
            attention_backend=self.attention_backend or "flashinfer",
            wideep_num_slots=self.wideep_num_slots,
            forward_model=self.forward_model or "op_level",
            moe_comm_backend=(self._resolve_moe_comm_backend(role, parallel) if parallel is not None else None),
            # Hardware fact, injected alongside the comm backend: the large-EP
            # ops take the comm node span at construction and would otherwise
            # have no channel to it (models.helpers.large_ep_gpus_per_node).
            num_gpus_per_node=self._num_gpus_per_node(role),
        )

    def _model_config_factory(self, role: Literal["agg", "prefill", "decode"]):
        """Per-tuple ModelConfig builder handed to ``sweep_*`` (see
        ``sweep._point_model_config``)."""

        def _build(parallel: ParallelChoice) -> config.ModelConfig:
            return self.build_model_config(role=role, parallel=parallel)

        return _build

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
        elif self.serving_mode == "afd":
            self._validate_afd()
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

    def _validate_afd(self) -> None:
        if not self.model_path:
            raise ValueError("afd mode requires model_path")
        if not self.system_name:
            raise ValueError("afd mode requires system_name")
        if self.effective_total_gpus is None:
            raise ValueError("afd mode requires total_gpus or afd_total_gpus")
        if (
            isinstance(self.afd_max_a_batch_size, bool)
            or not isinstance(self.afd_max_a_batch_size, int)
            or self.afd_max_a_batch_size < 32
        ):
            raise ValueError(f"afd_max_a_batch_size must be an integer >= 32, got {self.afd_max_a_batch_size!r}.")
        if (
            isinstance(self.afd_max_candidates, bool)
            or not isinstance(self.afd_max_candidates, int)
            or self.afd_max_candidates < 1
        ):
            raise ValueError(f"afd_max_candidates must be a positive integer, got {self.afd_max_candidates!r}.")
        if self.afd_candidate_overflow not in {"error", "truncate"}:
            raise ValueError(
                f"afd_candidate_overflow must be either 'error' or 'truncate', got {self.afd_candidate_overflow!r}."
            )
        if self.backend_name == "vllm" and self._model_family == "DEEPSEEK":
            raise NotImplementedError("AIConfigurator does not yet support the DeepSeek family on the vLLM backend.")

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
        elif self.serving_mode == "afd":
            # AFD uses the agg worker config for both A and F pools
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
        is_moe = self._is_moe

        # Attention-module op keys per regime this role's tuples can reach
        # (fused and/or large EP) -- see _reachable_attention_op_keys.
        attention_op_pairs = self._reachable_attention_op_keys(role)

        # supported_quant_mode is a DATA-PRESENCE list (which quants the DB carries
        # tables for), not a backend-capability list. In SILICON that equals what we
        # can model. In HYBRID/EMPIRICAL the util-empirical path (GEMM and MoE; the
        # shared quant-transfer primitive in operations/util_empirical.py) can
        # synthesize a quant from a collected sibling: XQUANT borrows within the
        # same (memory, compute) profile, XPROFILE borrows across profiles rescaled
        # by the op's util-LEVEL ratio. This gate mirrors exactly what the resolved
        # policy + DB contents make reachable at query time — truly-unreachable
        # quants still fail early here rather than crashing late in the sweep.
        # Admission only holds if (a) we're in a non-SILICON mode AND (b) the
        # resolved transfer policy actually enables that relation. Otherwise the
        # query path rejects the quant at run time by policy, so validate must not
        # pre-admit it (e.g. transfer_policy="off"/"conservative").
        _non_silicon = self.database_mode not in (
            None,
            common.DatabaseMode.SILICON.name,
        )
        _policy = common.resolve_transfer_policy(self.transfer_policy)
        xquant_enabled = _non_silicon and common.TransferKind.XQUANT in _policy
        xprofile_enabled = _non_silicon and common.TransferKind.XPROFILE in _policy

        from aiconfigurator.sdk.operations import gemm as gemm_ops
        from aiconfigurator.sdk.operations import moe as moe_ops

        _xprofile_level_known = {
            "gemm": gemm_ops.xprofile_util_level_known,
            "moe": moe_ops.xprofile_util_level_known,
            "wideep_context_moe": moe_ops.xprofile_util_level_known,
            "wideep_generation_moe": moe_ops.xprofile_util_level_known,
        }

        def _mode_profile(mode: Any) -> tuple:
            val = getattr(mode, "value", None)
            return (getattr(val, "memory", None), getattr(val, "compute", None))

        def _supported_profiles(mode: Any, supported_names: list) -> list[tuple]:
            enum_cls = type(mode)
            out = []
            for nm in supported_names:
                try:
                    out.append(_mode_profile(enum_cls[nm]))
                except (KeyError, AttributeError):
                    continue
            return out

        def _profile_reachable(mode: Any, supported_names: list) -> bool:
            return _mode_profile(mode) in _supported_profiles(mode, supported_names)

        def _xprofile_reachable(op: str, mode: Any, supported_names: list) -> bool:
            """XPROFILE admission: the op's util-LEVEL table must list the query
            profile (the runtime default fallback is deliberately NOT admitted —
            the one intentional way this gate is stricter than the ladder: it
            enforces the enum-line + level-line add-a-quant recipe), and the DB
            must carry at least one quant of a DIFFERENT profile to borrow from
            (any collected quant is a viable nearest-profile reference)."""
            level_known = _xprofile_level_known.get(op)
            if level_known is None or not level_known(mode):
                return False
            qp = _mode_profile(mode)
            return any(p != qp for p in _supported_profiles(mode, supported_names))

        def _check(op: str, mode: Any, *, profile_transfer: bool = False) -> None:
            if mode is None:
                return
            # Strict per-dtype FLOPS resolution happens at every query entry
            # (#1398) — BEFORE any table lookup or transfer ladder — so the
            # gate must mirror it: a mode whose compute dtype has no usable
            # *_tc_flops entry fails validate() here instead of on the first
            # sweep query, no matter how transfer-reachable its data is.
            # Memory-only modes (kv cache) carry compute_dtype None and skip;
            # the sm-gated generation derivation stays a query-time concern.
            # (getattr: unit tests stub _try_load_role_database with sentinel
            # objects that carry no system_spec — skip the check there, real
            # databases always have one.)
            _spec = getattr(database, "system_spec", None)
            if _spec is not None and getattr(getattr(mode, "value", None), "compute_dtype", None) is not None:
                common.get_quant_tc_flops(_spec, mode)
            modes = supported.get(op, []) or []
            if not modes:
                return  # DB doesn't record support for this op; skip
            name = mode.name if hasattr(mode, "name") else str(mode)
            if name in modes:
                return
            # Modes that normalize to a different table name for perf queries
            # (w4a16_mxfp4_cutlass -> w4a16_mxfp4, see operations/moe.py) are
            # accepted when the target table mode is supported. Data-less quants
            # are NOT aliased to a collected table — they are admitted (or not)
            # through the transfer reachability checks below, mirroring the
            # query-time ladder.
            validation_aliases = {"w4a16_mxfp4_cutlass": "w4a16_mxfp4"}
            alias = validation_aliases.get(name)
            if alias and alias in modes:
                return
            # fp8_static is a COMPOSITE mode: its base GEMM is transferable, but
            # it also requires compute_scale/scale_matrix overhead tables that
            # have no transfer ladder (by design). Its admission stays purely
            # data-driven (fp8_static in modes iff all three tables exist, see
            # _gemm_key_names).
            transfer_ok = profile_transfer and mode is not common.GEMMQuantMode.fp8_static
            if transfer_ok and xquant_enabled and _profile_reachable(mode, modes):
                return  # XQUANT-reachable in HYBRID/EMPIRICAL (same-profile data)
            if transfer_ok and xprofile_enabled and _xprofile_reachable(op, mode, modes):
                return  # XPROFILE-reachable (calibrated level + any other-profile data)
            exc_type = UnsupportedWideepConfigError if op.startswith("wideep_") else ValueError
            raise exc_type(
                f"Unsupported {op} quant mode {name!r} for system={system!r}, "
                f"backend={backend!r}, version={version!r}. "
                f"Supported {op} modes: {sorted(modes)}"
            )

        def _check_attention(ops: list[str], mode: Any) -> None:
            """Validate one quant mode against the reachable attention regimes.

            ``ops`` is ordered by authority (``_reachable_attention_op_keys``:
            fused first, then large EP), and the rule is "the FIRST INFORMATIVE
            op decides":

            - An op whose table the DB records no ``supported_quant_mode`` for
              carries no capability information: it abstains. It must neither
              green-light the check (an absent table is not a capability
              statement) nor fail it -- the legacy per-op check skipped it too.
            - The first informative op is the regime the task MUST be able to
              run: the fused one whenever any enumerated tuple is fused, the
              large-EP one for a large-EP-only task (which is what the flag
              used to select). Failing it is fatal, exactly as before this
              became a per-tuple property.
            - A later regime that cannot serve the mode is NOT fatal: its
              tuples are pruned by the sweep's per-tuple guard. Say so once, so
              losing them is not silent.
            - No informative op at all -> abstain (legacy behavior for ops the
              DB records nothing about)."""
            informative = [op for op in ops if supported.get(op)]
            if not informative:
                return
            _check(informative[0], mode)
            unusable = []
            for op in informative[1:]:
                try:
                    _check(op, mode)
                except (ValueError, UnsupportedWideepConfigError):
                    unusable.append(op)
            if unusable:
                logger.info(
                    "%s: %s has no %s data for %s; the parallel configs that need "
                    "%s will be skipped during the sweep (the %r regime still runs). "
                    "Set a quant mode those tables carry to model them.",
                    role,
                    ", ".join(unusable),
                    getattr(mode, "name", mode),
                    f"{system}/{backend}/{version}",
                    " / ".join(unusable),
                    informative[0],
                )

        # GEMM is always validated (applies to all worker shapes). It has the
        # same transfer ladder as MoE (shared primitive), so the same
        # HYBRID/EMPIRICAL relaxation applies.
        _check("gemm", self._role_attr(role, "gemm_quant_mode"), profile_transfer=True)

        # MoE — only when model is MoE. The large-EP expert-compute tables are
        # NOT validated here: coverage gating already decided, per tuple and per
        # quant mode, which EP sizes have data (a tuple that slips through a
        # partial-coverage hole is pruned by the sweep's per-tuple guard). Only
        # the fused MoE op, which every task can still reach, is checked.
        if is_moe:
            _check("moe", self._role_attr(role, "moe_quant_mode"), profile_transfer=True)

        # FMHA: only meaningful for context-using workers (agg, prefill).
        if validate_context:
            _check_attention([ctx for ctx, _gen in attention_op_pairs], self._role_attr(role, "fmha_quant_mode"))

        # KV cache: only meaningful for generation-using workers (agg, decode).
        if validate_generation:
            _check_attention([gen for _ctx, gen in attention_op_pairs], self._role_attr(role, "kvcache_quant_mode"))

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
            # Per-tuple builder: the MoE comm backend is resolved for each
            # parallel config, not once for the task.
            "model_config": self._model_config_factory("agg"),
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
        runtime_config = self.build_runtime_config()
        if self.pareto_sweep:
            runtime_config.tpot = _LEGACY_TPOT_SWEEP
        return {
            "model_path": self.prefill_model_path,
            "runtime_config": runtime_config,
            "prefill_database": prefill_database,
            "prefill_backend_name": self.prefill_backend_name,
            # Per-tuple builders (see sweep_agg_kwargs).
            "prefill_model_config": self._model_config_factory("prefill"),
            "prefill_parallel_config_list": prefill_parallel,
            "prefill_latency_correction": self.prefill_latency_correction,
            "decode_database": decode_database,
            "decode_backend_name": self.decode_backend_name,
            "decode_model_config": self._model_config_factory("decode"),
            "decode_parallel_config_list": decode_parallel,
            "decode_latency_correction": self.decode_latency_correction,
            "free_gpu_memory_fraction": self.free_gpu_memory_fraction,
            "prefill_max_num_tokens": max(self.prefill_max_batch_size, 1) * self.isl,
            "decode_max_num_tokens": self.decode_max_batch_size,
            "prefill_num_worker_list": prefill_worker_list,
            "decode_num_worker_list": decode_worker_list,
            "num_gpu_list": num_gpu_list,
            "rate_matching_prefill_degradation": self.rate_match_prefill_degradation,
            "rate_matching_decode_degradation": self.rate_match_decode_degradation,
            "autoscale_ttft_correction_factor": self.autoscale_ttft_correction_factor,
            "require_same_tp": self._require_same_tp_gate(),
        }

    def sweep_afd_kwargs(self, *, database) -> dict[str, Any]:
        """Return the exact kwargs needed for sweep.sweep_afd."""
        if self.serving_mode != "afd":
            raise ValueError(f"sweep_afd_kwargs requires serving_mode='afd', got {self.serving_mode!r}")

        runtime_config = self.build_runtime_config()
        if self.pareto_sweep:
            runtime_config.tpot = _LEGACY_TPOT_SWEEP

        # Build prefill parallel config list for combined-with-PD
        prefill_parallel_config_list = None
        prefill_model_config = None
        if self.afd_combined_with_pd:
            prefill_parallel_config_list = list(self.iter_parallel("prefill"))
            prefill_model_config = self.build_model_config(role="prefill")

        return {
            "model_path": self.model_path,
            "runtime_config": runtime_config,
            "database": database,
            "backend_name": self.backend_name,
            "model_config": self.build_model_config(role="agg"),
            "afd_parallel_config_list": [tuple(c) for c in self._afd_parallel_config_list],
            "gpus_per_node": self._afd_gpus_per_node,
            "total_gpus": self.effective_total_gpus,
            "combined_with_pd": self.afd_combined_with_pd,
            "comm_overhead_factor": self.afd_comm_overhead_factor,
            "boundary_on_attn": self.afd_boundary_on_attn,
            "total_batch_size": self.afd_total_batch_size,
            "max_a_batch_size": self.afd_max_a_batch_size,
            "target_ttft": self.ttft,
            "free_gpu_memory_fraction": self.free_gpu_memory_fraction,
            "max_seq_len": self.max_seq_len,
            # combined-with-PD prefill options
            "prefill_database": database if self.afd_combined_with_pd else None,
            "prefill_backend_name": self.backend_name if self.afd_combined_with_pd else None,
            "prefill_model_config": prefill_model_config,
            "prefill_parallel_config_list": prefill_parallel_config_list,
            "prefill_batch_size_list": self.afd_prefill_batch_size_list,
            "prefill_system_name": self.system_name if self.afd_combined_with_pd else None,
            "prefill_backend_version": self.backend_version if self.afd_combined_with_pd else None,
            "prefill_max_candidates": self.afd_prefill_max_candidates,
            "prefill_candidate_overflow": self.afd_prefill_candidate_overflow,
            "max_prefill_gpus": self.afd_max_prefill_gpus,
            "max_prefill_workers": self.afd_max_prefill_workers,
            # calibration
            "prefill_degradation": self.afd_prefill_degradation,
            "decode_degradation": self.afd_decode_degradation,
            "ttft_correction_factor": self.afd_ttft_correction_factor,
            "decode_latency_correction": self.afd_decode_latency_correction,
        }

    def _require_same_tp_gate(self):
        """Per-PAIR "prefill and decode TP must match" predicate for sweep_disagg.

        SGLang disaggregated serving requires matching TP (KV transfer layout,
        ai-dynamo/dynamo#5870); large EP relaxes it. Large EP is per tuple now,
        so the constraint is decided per candidate PAIR: a pair is exempt when
        either side resolves a MoE comm backend. Non-sglang tasks return the
        plain ``False`` the old bool API used (no predicate needed)."""
        if self.prefill_backend_name != "sglang":
            return False
        large_ep_eps = {
            role: {tup[4] for tup in self.iter_parallel(role) if self._resolve_moe_comm_backend(role, tup) is not None}
            for role in ("prefill", "decode")
        }
        if not (large_ep_eps["prefill"] or large_ep_eps["decode"]):
            return True  # nothing can be large EP -> the constraint always applies

        def _requires_same_tp(prefill_row, decode_row) -> bool:
            for role, row in (("prefill", prefill_row), ("decode", decode_row)):
                if int(row["moe_tp"]) == 1 and int(row["moe_ep"]) in large_ep_eps[role]:
                    return False
            return True

        return _requires_same_tp

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
        from aiconfigurator.sdk.sweep import sweep_afd, sweep_agg, sweep_disagg

        if self.serving_mode == "agg":
            if autoscale:
                raise ValueError("autoscale is only supported in disagg mode")
            database = self._load_database(self.system_name, self.backend_name, self.backend_version)
            return sweep_agg(
                **self.sweep_agg_kwargs(database=database),
                predictor=self.predictor,
                speculative_profile=self.build_speculative_profile(),
            )
        if self.serving_mode == "afd":
            if autoscale:
                raise ValueError("autoscale is not supported for afd serving mode")
            database = self._load_database(self.system_name, self.backend_name, self.backend_version)
            if self._afd_topology_pinned:
                # Pinned topology: single-point via AFDInferenceSession
                return self._run_afd_single_point(database)
            return sweep_afd(**self.sweep_afd_kwargs(database=database))
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

        model_config = self.build_model_config(role="agg", parallel=(tp, pp, dp, moe_tp, moe_ep, 1))
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
        p_mc = self.build_model_config(
            role="prefill", parallel=(prefill_tp, prefill_pp, prefill_dp, prefill_moe_tp, prefill_moe_ep, 1)
        )
        p_mc.tp_size = prefill_tp
        p_mc.pp_size = prefill_pp
        p_mc.attention_dp_size = prefill_dp if self._is_moe else 1
        p_mc.moe_tp_size = prefill_moe_tp
        p_mc.moe_ep_size = prefill_moe_ep

        p_rt = self.build_runtime_config(batch_size=prefill_batch_size)
        p_db = self._load_database(self.prefill_system_name, self.prefill_backend_name, self.prefill_backend_version)
        p_backend = get_backend(self.prefill_backend_name)
        p_model = get_model(self.prefill_model_path, p_mc, self.prefill_backend_name)

        worker_kwargs: dict[str, Any] = {}
        if self.free_gpu_memory_fraction is not None:
            worker_kwargs["free_gpu_memory_fraction"] = self.free_gpu_memory_fraction

        p_summary = predict_disagg_worker(
            model=p_model,
            backend=p_backend,
            database=p_db,
            runtime_config=p_rt,
            role="prefill",
            latency_correction=self.prefill_latency_correction,
            predictor=self.predictor,
            speculative_profile=self.build_speculative_profile(),
            **worker_kwargs,
        )
        if p_summary.check_oom() or p_summary.check_kv_cache_oom():
            raise RuntimeError(
                f"OOM in prefill phase at tp={prefill_tp} pp={prefill_pp} dp={prefill_dp} "
                f"batch_size={prefill_batch_size} (memory capacity or KV-cache budget exceeded)."
            )

        # --- Decode phase ---
        d_mc = self.build_model_config(
            role="decode", parallel=(decode_tp, decode_pp, decode_dp, decode_moe_tp, decode_moe_ep, 1)
        )
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
            **worker_kwargs,
        )
        if d_summary.check_oom() or d_summary.check_kv_cache_oom():
            raise RuntimeError(
                f"OOM in decode phase at tp={decode_tp} pp={decode_pp} dp={decode_dp} "
                f"batch_size={decode_batch_size} (memory capacity or KV-cache budget exceeded)."
            )

        # --- Rate-match the pair ---
        p_dict = p_summary.get_summary_df().iloc[0].to_dict()
        d_dict = d_summary.get_summary_df().iloc[0].to_dict()
        return _rate_match_dict(p_dict, prefill_num_workers, d_dict, decode_num_workers)

    def _run_afd_single_point(self, database):
        """Run a single pinned-topology AFD estimate via AFDInferenceSession."""
        import copy

        from aiconfigurator.sdk.backends.factory import get_backend
        from aiconfigurator.sdk.config import AFDConfig
        from aiconfigurator.sdk.inference_session import AFDInferenceSession

        gpus_per_node = self._afd_gpus_per_node
        n_a_nodes = self.afd_n_a_nodes
        n_f_nodes = self.afd_n_f_nodes
        tp_a = self.afd_tp_a

        backend = get_backend(self.backend_name)
        base_model_config = self.build_model_config(role="agg")

        tp_f = n_f_nodes * gpus_per_node
        n_a_workers = (n_a_nodes * gpus_per_node) // tp_a

        # Derive a_batch_size from total_batch_size if provided
        a_batch_size = self.afd_a_batch_size or 128
        if self.afd_total_batch_size is not None:
            if n_a_workers <= 0 or self.afd_total_batch_size % n_a_workers != 0:
                raise ValueError(
                    f"afd_total_batch_size={self.afd_total_batch_size} must be exactly divisible "
                    f"by n_a_workers={n_a_workers}."
                )
            derived = self.afd_total_batch_size // n_a_workers
            if self.afd_a_batch_size is not None and self.afd_a_batch_size != derived:
                raise ValueError(
                    f"afd_a_batch_size={self.afd_a_batch_size} conflicts with "
                    f"afd_total_batch_size={self.afd_total_batch_size} / n_a_workers={n_a_workers} = {derived}."
                )
            a_batch_size = derived

        # Determine f_moe_ep_size (default: tp_f for MoE, 1 for dense)
        f_moe_ep_size = tp_f if self._is_moe else 1
        if tp_f % f_moe_ep_size != 0:
            raise ValueError(f"f_moe_ep_size ({f_moe_ep_size}) must divide tp_f ({tp_f}).")
        f_moe_tp = tp_f // f_moe_ep_size

        afd_config = AFDConfig(
            n_a_nodes=n_a_nodes,
            n_f_nodes=n_f_nodes,
            tp_a=tp_a,
            tp_f=tp_f,
            a_batch_size=a_batch_size,
            gpus_per_node=gpus_per_node,
            f_moe_ep_size=f_moe_ep_size,
            comm_overhead_factor=self.afd_comm_overhead_factor,
            boundary_on_attn=self.afd_boundary_on_attn,
        )

        a_model_config = copy.deepcopy(base_model_config)
        a_model_config.tp_size = tp_a
        a_model_config.pp_size = 1
        a_model_config.moe_tp_size = tp_a
        a_model_config.moe_ep_size = 1
        a_model_config.attention_dp_size = 1

        f_model_config = copy.deepcopy(base_model_config)
        f_model_config.tp_size = tp_f
        f_model_config.pp_size = 1
        f_model_config.moe_tp_size = f_moe_tp
        f_model_config.moe_ep_size = f_moe_ep_size
        f_model_config.attention_dp_size = 1

        runtime_config = self.build_runtime_config(batch_size=afd_config.n_a_workers * afd_config.a_batch_size)

        session = AFDInferenceSession(
            model_path=self.model_path,
            a_model_config=a_model_config,
            f_model_config=f_model_config,
            database=database,
            backend=backend,
            afd_config=afd_config,
        )
        summary = session.run_afd(
            runtime_config,
            phase="both",
            free_gpu_memory_fraction=self.free_gpu_memory_fraction,
            max_seq_len=self.max_seq_len,
        )
        return summary.get_summary_df()


__all__ = ["ParallelChoice", "Task", "_lookup_num_gpus_per_node", "build_afd_parallel_lists"]
