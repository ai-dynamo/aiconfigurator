# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Parallelization enumeration for DGD candidate generation.

This module enumerates disaggregated prefill/decode parallelization candidates
and converts each into an aggregated DynamoGraphDeployment (DGD) config suitable
for real-silicon profiling sweeps.

The flow mirrors ``engine.py``:  AIC's rendering engine computes CLI args
(including rule-plugin heuristics), then dynamo's ``build_dgd_config`` and
``convert_config`` produce the final aggregated DGD.
"""

from __future__ import annotations

import logging
import shlex
from typing import Any

from aiconfigurator.generator.naive import (
    _calculate_min_tp,
    _estimate_model_weight_bytes,
    _get_system_config,
)
from aiconfigurator.generator.rendering.engine import render_backend_templates
from aiconfigurator.sdk import common
from aiconfigurator.sdk.models import check_is_moe
from aiconfigurator.sdk.task import build_disagg_parallel_lists
from aiconfigurator.sdk.utils import enumerate_parallel_config

logger = logging.getLogger(__name__)

# Default heuristics from AIC's advanced_tuning_config
_AIC_PREFILL_MAX_BATCH_SIZE = 1
_AIC_DECODE_MAX_BATCH_SIZE = 512


# ---------------------------------------------------------------------------
# Step 1: Support check
# ---------------------------------------------------------------------------


def check_model_hardware_support(
    model_path: str,
    system: str,
    backend: str,
    backend_version: str | None = None,
) -> bool:
    """Check whether AIC supports the model x hardware combination for disagg.

    This wraps :func:`common.check_support` and returns a single boolean
    indicating disaggregated-mode support.

    Args:
        model_path: HuggingFace model path or local path.
        system: System name (GPU type), e.g. ``"h200_sxm"``.
        backend: Backend name (``"trtllm"``, ``"sglang"``, ``"vllm"``).
        backend_version: Optional backend database version.

    Returns:
        ``True`` if disaggregated mode is supported, ``False`` otherwise.
    """
    from aiconfigurator.sdk.utils import get_model_config_from_model_path

    try:
        model_info = get_model_config_from_model_path(model_path)
        architecture = model_info.get("architecture")
    except Exception:
        architecture = None

    try:
        result = common.check_support(
            model=model_path,
            system=system,
            backend=backend,
            version=backend_version,
            architecture=architecture,
        )
        return result.disagg_supported
    except Exception:
        logger.warning("Support check failed for %s on %s/%s", model_path, system, backend)
        return False


# ---------------------------------------------------------------------------
# Step 2: Build param_values and get CLI args via AIC rendering engine
# ---------------------------------------------------------------------------


def _build_param_values(
    model_path: str,
    backend: str,
    image: str,
    num_gpus_per_node: int,
    isl: int,
    worker_type: str,
    tp: int,
    pp: int,
    dp: int,
    *,
    max_batch_size: int | None = None,
    max_num_tokens: int | None = None,
) -> dict[str, Any]:
    """Build an AIC ``param_values`` dict for a single candidate.

    The dict follows the structure expected by
    :func:`render_backend_templates`: ``ServiceConfig``, ``K8sConfig``,
    ``DynConfig``, ``params`` (with ``prefill`` / ``decode`` sub-dicts),
    ``WorkerConfig``, ``NodeConfig``, ``SlaConfig``.

    Only the ``worker_type`` side is populated with the real parallel config;
    the other side gets a minimal placeholder so the rendering engine can
    still produce both sets of CLI args without errors.
    """
    gpus_per_worker = tp * pp * dp

    # Build the worker params for the candidate side
    worker_params: dict[str, Any] = {
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": pp,
        "gpus_per_worker": gpus_per_worker,
    }
    if dp > 1:
        worker_params["data_parallel_size"] = dp
    if max_batch_size is not None:
        worker_params["max_batch_size"] = max_batch_size
    if max_num_tokens is not None:
        worker_params["max_num_tokens"] = max_num_tokens

    # Minimal placeholder for the other worker side
    placeholder: dict[str, Any] = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "gpus_per_worker": 1,
    }

    if worker_type == "prefill":
        params = {"prefill": worker_params, "decode": placeholder}
        worker_config = {
            "prefill_workers": 1,
            "decode_workers": 1,
            "prefill_gpus_per_worker": gpus_per_worker,
            "decode_gpus_per_worker": 1,
        }
    else:
        params = {"prefill": placeholder, "decode": worker_params}
        worker_config = {
            "prefill_workers": 1,
            "decode_workers": 1,
            "prefill_gpus_per_worker": 1,
            "decode_gpus_per_worker": gpus_per_worker,
        }

    return {
        "ServiceConfig": {
            "model_path": model_path,
            "served_model_path": model_path,
        },
        "K8sConfig": {
            "k8s_image": image,
            "name_prefix": "enum",
        },
        "DynConfig": {"mode": "disagg"},
        "params": params,
        "WorkerConfig": worker_config,
        "NodeConfig": {"num_gpus_per_node": num_gpus_per_node},
        "SlaConfig": {"isl": isl, "osl": 1000},
        "backend": backend,
    }


# ---------------------------------------------------------------------------
# Step 3: Build a single aggregated DGD for one candidate
# ---------------------------------------------------------------------------


def _build_agg_dgd_for_candidate(
    backend: str,
    model_path: str,
    image: str,
    num_gpus_per_node: int,
    isl: int,
    is_moe: bool,
    worker_type: str,
    tp: int,
    pp: int,
    dp: int,
    *,
    max_batch_size: int | None = None,
    max_num_tokens: int | None = None,
    k8s_pvc_name: str | None = None,
    k8s_pvc_mount_path: str = "/workspace/model_cache",
    k8s_model_path_in_pvc: str | None = None,
) -> dict:
    """Build an aggregated DGD config for a single prefill or decode candidate.

    The flow mirrors ``engine.py``:

    1. Build ``param_values`` and call :func:`render_backend_templates` to get
       CLI args (this applies AIC rule plugins / heuristics).
    2. Pass CLI args to ``modifier.build_dgd_config`` to build a disagg DGD
       (same approach as :func:`_generate_k8s_via_dynamo`).
    3. Call ``modifier.convert_config`` to collapse the disagg DGD into an
       aggregated single-worker DGD for profiling.

    Args:
        backend: Backend name (``"trtllm"``, ``"sglang"``, ``"vllm"``).
        model_path: HuggingFace ID or local model path.
        image: Container image for all DGD services.
        num_gpus_per_node: GPUs per physical node.
        isl: Input sequence length.
        is_moe: Whether the model is MoE.
        worker_type: ``"prefill"`` or ``"decode"``.
        tp: Tensor-parallel size.
        pp: Pipeline-parallel size (always 1 in real_silicon_sweep).
        dp: Data-parallel size.
        max_batch_size: Override max batch size in ``param_values``.
            When ``None`` the param is omitted and the rule plugins / engine
            defaults apply.
        max_num_tokens: Override max num tokens in ``param_values``.
            When ``None`` the param is omitted and the rule plugins / engine
            defaults apply.
        k8s_pvc_name: PVC claim name for model cache.
        k8s_pvc_mount_path: Where the PVC is mounted inside the container.
        k8s_model_path_in_pvc: Relative path to model inside PVC.

    Returns:
        Complete aggregated DGD config dict.
    """
    from dynamo.profiler.utils.config_modifiers import CONFIG_MODIFIERS
    from dynamo.profiler.utils.defaults import EngineType

    modifier = CONFIG_MODIFIERS[backend]
    gpus = tp * pp * dp

    # ------------------------------------------------------------------
    # 1. Compute CLI args via AIC rendering engine (rule plugins included)
    # ------------------------------------------------------------------
    param_values = _build_param_values(
        model_path=model_path,
        backend=backend,
        image=image,
        num_gpus_per_node=num_gpus_per_node,
        isl=isl,
        worker_type=worker_type,
        tp=tp,
        pp=pp,
        dp=dp,
        max_batch_size=max_batch_size,
        max_num_tokens=max_num_tokens,
    )

    rendered = render_backend_templates(param_values, backend)

    if worker_type == "prefill":
        cli_str = rendered.get("cli_args_prefill", "")
    else:
        cli_str = rendered.get("cli_args_decode", "")

    cli_args_list = shlex.split(cli_str) if cli_str else []

    # ------------------------------------------------------------------
    # 2. Build disagg DGD from CLI args (mirrors _generate_k8s_via_dynamo)
    # ------------------------------------------------------------------
    if worker_type == "prefill":
        dgd = modifier.build_dgd_config(
            mode="disagg",
            model_name=model_path,
            image=image,
            prefill_cli_args=cli_args_list,
            prefill_replicas=1,
            prefill_gpus=gpus,
            decode_cli_args=[],
            decode_replicas=1,
            decode_gpus=1,
        )
    else:
        dgd = modifier.build_dgd_config(
            mode="disagg",
            model_name=model_path,
            image=image,
            prefill_cli_args=[],
            prefill_replicas=1,
            prefill_gpus=1,
            decode_cli_args=cli_args_list,
            decode_replicas=1,
            decode_gpus=gpus,
        )

    # Apply PVC after build_dgd_config so we can pass the correct pvc_path
    # (build_dgd_config hardcodes pvc_path="" which loses the model subdirectory)
    if k8s_pvc_name:
        dgd = modifier.update_model_from_pvc(
            dgd,
            model_name=model_path,
            pvc_name=k8s_pvc_name,
            pvc_mount_path=k8s_pvc_mount_path,
            pvc_path=k8s_model_path_in_pvc or "",
        )

    # ------------------------------------------------------------------
    # 3. Convert disagg DGD → aggregated single-worker DGD
    # ------------------------------------------------------------------
    engine_type = EngineType.PREFILL if worker_type == "prefill" else EngineType.DECODE
    agg_config = modifier.convert_config(dgd, engine_type, is_moe_model=is_moe)

    return agg_config


# ---------------------------------------------------------------------------
# Step 4: Main enumeration entry point
# ---------------------------------------------------------------------------


def enumerate_profiling_configs(
    model_path: str,
    system: str,
    backend: str,
    *,
    image: str = "",
    isl: int = 4000,
    enable_wideep: bool = False,
    backend_version: str | None = None,
    num_gpus_per_node: int | None = None,
    k8s_pvc_name: str | None = None,
    k8s_pvc_mount_path: str = "/workspace/model_cache",
    k8s_model_path_in_pvc: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Enumerate parallelization candidates and return aggregated DGD configs.

    This function is the primary entry point for the Dynamo profiler to obtain
    a set of DGD configs for real-silicon prefill and decode sweeps.

    Only **disaggregated** mode is supported.  For each candidate GPU count
    and parallelization strategy the function:

    1. Checks AIC support for the model x hardware combination.
    2. Computes memory-based minimum GPUs per engine.
    3. Enumerates valid parallelization configs (TP / TEP / DEP, **no PP**).
    4. For each config, uses AIC's rendering engine (with rule plugins) to
       compute CLI args, then builds an aggregated DGD via dynamo's
       ``build_dgd_config`` + ``convert_config``.
    5. Applies AIC heuristics for ``max_batch_size`` / ``max_num_tokens``
       when the combination is supported; otherwise uses safe defaults.

    Args:
        model_path: HuggingFace model ID or local path.  Used for memory
            estimation and AIC support checks (reads ``config.json`` from
            HuggingFace or local filesystem).
        system: System name (GPU type), e.g. ``"h200_sxm"``.
        backend: Backend name (``"trtllm"``, ``"sglang"``, ``"vllm"``).
        image: Container image for all DGD services.
        isl: Input sequence length (used for prefill heuristic).
        enable_wideep: Enable wide expert-parallelism search space.
        backend_version: Optional backend database version.
        num_gpus_per_node: GPUs per physical node.  If ``None`` it is
            read from the system config YAML.
        k8s_pvc_name: PVC claim name for model cache.  When set, the
            generated DGDs will mount this PVC.
        k8s_pvc_mount_path: Where the PVC is mounted inside the container.
            Defaults to ``"/workspace/model_cache"``.
        k8s_model_path_in_pvc: Relative path to the model inside the PVC
            (e.g. ``"Qwen/Qwen3-32B"``).  The engine uses
            ``{k8s_pvc_mount_path}/{k8s_model_path_in_pvc}`` at runtime
            to load weights.

    Returns:
        ``(prefill_configs, decode_configs)`` -- two lists of aggregated DGD
        config dicts ready for YAML serialization and deployment.
    """
    # ------------------------------------------------------------------
    # 0. Resolve system config
    # ------------------------------------------------------------------
    sys_cfg = _get_system_config(system)
    if num_gpus_per_node is None:
        num_gpus_per_node = sys_cfg["gpus_per_node"]
    vram_per_gpu = sys_cfg["vram_per_gpu"]

    is_moe = check_is_moe(model_path)
    backend_enum = common.BackendName(backend)

    # Auto-enable wideEP for all MoE models
    if is_moe:
        enable_wideep = True

    # GQA+MoE models (e.g. Qwen3Moe) also allow pure TP; MLA+MoE (e.g. DeepSeek) do not
    allow_moe_pure_tp = False
    if is_moe:
        from aiconfigurator.sdk.utils import get_model_config_from_model_path

        try:
            model_info = get_model_config_from_model_path(model_path)
            architecture = model_info.get("architecture", "")
            # GQA+MoE architectures that support pure TP sweeping
            _gqa_moe_architectures = {"Qwen3MoeForCausalLM"}
            allow_moe_pure_tp = architecture in _gqa_moe_architectures
        except Exception:
            allow_moe_pure_tp = False

    # ------------------------------------------------------------------
    # 1. Support check
    # ------------------------------------------------------------------
    aic_supported = check_model_hardware_support(model_path, system, backend, backend_version=backend_version)
    logger.info(
        "AIC disagg support for %s on %s/%s: %s",
        model_path,
        system,
        backend,
        aic_supported,
    )

    # ------------------------------------------------------------------
    # 2. Memory-based minimum GPUs per engine
    # ------------------------------------------------------------------
    model_weight_bytes = _estimate_model_weight_bytes(model_path)
    min_gpus = _calculate_min_tp(
        model_weight_bytes=model_weight_bytes,
        vram_per_gpu=vram_per_gpu,
        gpus_per_node=num_gpus_per_node,
        total_gpus=num_gpus_per_node,  # cap at single node for min calculation
    )
    logger.info("Minimum GPUs per engine (memory fit): %d", min_gpus)

    # ------------------------------------------------------------------
    # 3. Enumerate parallel configs
    # ------------------------------------------------------------------
    prefill_wc, decode_wc = build_disagg_parallel_lists(
        backend_name=backend,
        prefill_system=system,
        decode_system=system,
        is_moe=is_moe,
        enable_wideep=enable_wideep,
    )

    # Determine max GPUs for real_silicon_sweep filtering
    if is_moe and enable_wideep:
        prefill_max_gpus = max(prefill_wc["num_gpu_per_worker"])
        decode_max_gpus = max(decode_wc["num_gpu_per_worker"])
    else:
        # Dense models (and MoE non-wideEP): cap at num_gpus_per_node
        prefill_max_gpus = num_gpus_per_node
        decode_max_gpus = num_gpus_per_node

    logger.info(
        "Parallel lists: is_moe=%s, enable_wideep=%s, min_gpus=%d, "
        "prefill_max_gpus=%d, decode_max_gpus=%d, "
        "prefill_num_gpu_list=%s, prefill_dp_list=%s, prefill_moe_ep_list=%s",
        is_moe,
        enable_wideep,
        min_gpus,
        prefill_max_gpus,
        decode_max_gpus,
        prefill_wc["num_gpu_per_worker"],
        prefill_wc["dp_list"],
        prefill_wc["moe_ep_list"],
    )

    prefill_parallel_configs = enumerate_parallel_config(
        num_gpu_list=prefill_wc["num_gpu_per_worker"],
        tp_list=prefill_wc["tp_list"],
        pp_list=prefill_wc["pp_list"],
        dp_list=prefill_wc["dp_list"],
        moe_tp_list=prefill_wc["moe_tp_list"],
        moe_ep_list=prefill_wc["moe_ep_list"],
        is_moe=is_moe,
        backend=backend_enum,
        enable_wideep=enable_wideep,
        real_silicon_sweep=True,
        min_num_gpus=min_gpus,
        max_num_gpus=prefill_max_gpus,
        allow_moe_pure_tp=allow_moe_pure_tp,
    )

    decode_parallel_configs = enumerate_parallel_config(
        num_gpu_list=decode_wc["num_gpu_per_worker"],
        tp_list=decode_wc["tp_list"],
        pp_list=decode_wc["pp_list"],
        dp_list=decode_wc["dp_list"],
        moe_tp_list=decode_wc["moe_tp_list"],
        moe_ep_list=decode_wc["moe_ep_list"],
        is_moe=is_moe,
        backend=backend_enum,
        enable_wideep=enable_wideep,
        real_silicon_sweep=True,
        min_num_gpus=min_gpus,
        max_num_gpus=decode_max_gpus,
        allow_moe_pure_tp=allow_moe_pure_tp,
    )

    # For GQA+MoE models, wideep mode only produces DEP/TEP (moe_tp is always 1).
    # Pure TP configs (moe_tp > 1, moe_ep == 1) require a second pass with
    # enable_wideep=False to get non-wideep parallel lists.
    if is_moe and allow_moe_pure_tp and enable_wideep:
        prefill_wc_tp, decode_wc_tp = build_disagg_parallel_lists(
            backend_name=backend,
            prefill_system=system,
            decode_system=system,
            is_moe=is_moe,
            enable_wideep=False,
        )
        for label, wc, max_gpus, config_list in [
            ("prefill", prefill_wc_tp, prefill_max_gpus, prefill_parallel_configs),
            ("decode", decode_wc_tp, decode_max_gpus, decode_parallel_configs),
        ]:
            tp_configs = enumerate_parallel_config(
                num_gpu_list=wc["num_gpu_per_worker"],
                tp_list=wc["tp_list"],
                pp_list=wc["pp_list"],
                dp_list=wc["dp_list"],
                moe_tp_list=wc["moe_tp_list"],
                moe_ep_list=wc["moe_ep_list"],
                is_moe=is_moe,
                backend=backend_enum,
                enable_wideep=False,
                real_silicon_sweep=True,
                min_num_gpus=min_gpus,
                max_num_gpus=max_gpus,
                allow_moe_pure_tp=True,
            )
            # Only keep pure TP configs (avoid duplicating DEP/TEP from wideep pass)
            for cfg in tp_configs:
                _tp, _pp, _dp, _moe_tp, _moe_ep = cfg
                if _moe_tp > 1 and _moe_ep == 1 and cfg not in config_list:
                    config_list.append(cfg)

    logger.info(
        "Enumerated %d prefill configs, %d decode configs",
        len(prefill_parallel_configs),
        len(decode_parallel_configs),
    )

    # ------------------------------------------------------------------
    # 4. Determine heuristic batch / token limits
    # ------------------------------------------------------------------
    # Prefill: always max_batch_size=1, max_num_tokens=isl (AIC heuristic)
    prefill_max_bs: int | None = _AIC_PREFILL_MAX_BATCH_SIZE
    prefill_max_tokens: int | None = _AIC_PREFILL_MAX_BATCH_SIZE * isl

    if aic_supported:
        # AIC heuristic for decode
        decode_max_bs: int | None = _AIC_DECODE_MAX_BATCH_SIZE
        decode_max_tokens: int | None = _AIC_DECODE_MAX_BATCH_SIZE
    else:
        # Unsupported: leave decode to engine defaults
        decode_max_bs = None
        decode_max_tokens = None

    # ------------------------------------------------------------------
    # 5. Build DGD configs via AIC rendering + dynamo build_dgd_config
    # ------------------------------------------------------------------
    prefill_configs: list[dict[str, Any]] = []
    for cfg in prefill_parallel_configs:
        tp, pp, dp, moe_tp, moe_ep = cfg
        logger.info(
            "Building prefill DGD: tp=%d pp=%d dp=%d moe_tp=%d moe_ep=%d",
            tp,
            pp,
            dp,
            moe_tp,
            moe_ep,
        )
        try:
            dgd = _build_agg_dgd_for_candidate(
                backend=backend,
                model_path=model_path,
                image=image,
                num_gpus_per_node=num_gpus_per_node,
                isl=isl,
                is_moe=is_moe,
                worker_type="prefill",
                tp=tp,
                pp=pp,
                dp=dp,
                max_batch_size=prefill_max_bs,
                max_num_tokens=prefill_max_tokens,
                k8s_pvc_name=k8s_pvc_name,
                k8s_pvc_mount_path=k8s_pvc_mount_path,
                k8s_model_path_in_pvc=k8s_model_path_in_pvc,
            )
            prefill_configs.append(dgd)
        except Exception:
            logger.exception("Failed to build prefill DGD for config %s", cfg)

    decode_configs: list[dict[str, Any]] = []
    for cfg in decode_parallel_configs:
        tp, pp, dp, moe_tp, moe_ep = cfg
        logger.info(
            "Building decode DGD: tp=%d pp=%d dp=%d moe_tp=%d moe_ep=%d",
            tp,
            pp,
            dp,
            moe_tp,
            moe_ep,
        )
        try:
            dgd = _build_agg_dgd_for_candidate(
                backend=backend,
                model_path=model_path,
                image=image,
                num_gpus_per_node=num_gpus_per_node,
                isl=isl,
                is_moe=is_moe,
                worker_type="decode",
                tp=tp,
                pp=pp,
                dp=dp,
                max_batch_size=decode_max_bs,
                max_num_tokens=decode_max_tokens,
                k8s_pvc_name=k8s_pvc_name,
                k8s_pvc_mount_path=k8s_pvc_mount_path,
                k8s_model_path_in_pvc=k8s_model_path_in_pvc,
            )
            decode_configs.append(dgd)
        except Exception:
            logger.exception("Failed to build decode DGD for config %s", cfg)

    logger.info(
        "Enumeration complete: %d prefill DGDs, %d decode DGDs",
        len(prefill_configs),
        len(decode_configs),
    )
    return prefill_configs, decode_configs
