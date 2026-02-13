# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

# Valid backend values accepted by TorchLlmArgs.  update_llm_args_with_extra_dict
# may silently discard the field during merge, so we validate explicitly.
_VALID_TRTLLM_BACKENDS = frozenset({"pytorch"})


def _import_trtllm_args():
    from tensorrt_llm.llmapi import llm_args as llm_args_mod

    return llm_args_mod


def _load_yaml_payload(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in YAML file {path}, got {type(payload).__name__}")
    return payload


def collect_config_paths(root_dir: Path) -> list[tuple[str, Path]]:
    targets = [
        ("agg", root_dir / "agg" / "top1" / "agg_config.yaml"),
        ("decode", root_dir / "disagg" / "top1" / "decode_config.yaml"),
        ("prefill", root_dir / "disagg" / "top1" / "prefill_config.yaml"),
    ]
    missing = [label for label, path in targets if not path.exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"Missing expected configs under {root_dir}: {missing_str}. "
            "Expected paths: agg/top1/agg_config.yaml, disagg/top1/decode_config.yaml, "
            "disagg/top1/prefill_config.yaml."
        )
    return [(label, path) for label, path in targets]


def validate_torchllm_engine_args(
    engine_args: dict[str, Any],
    model_path: Optional[str] = None,
    *,
    backend: str = "pytorch",
):
    llm_args_mod = _import_trtllm_args()
    model_path = model_path or "dummy-model"
    base_args = llm_args_mod.TorchLlmArgs(model=model_path, backend=backend)
    base_dict = base_args.model_dump()

    extra_args = dict(engine_args)
    extra_args.setdefault("backend", backend)

    # Validate the backend value early.  update_llm_args_with_extra_dict may
    # silently ignore or overwrite this field, masking invalid values.
    cfg_backend = extra_args.get("backend")
    if cfg_backend not in _VALID_TRTLLM_BACKENDS:
        raise ValueError(
            f"Invalid backend '{cfg_backend}' in engine config, expected one of {sorted(_VALID_TRTLLM_BACKENDS)}"
        )

    # Treat unknown keys as validation errors for clarity.
    dropped_keys = sorted(set(extra_args) - set(base_dict))
    if dropped_keys:
        raise ValueError("Unsupported TRT-LLM config keys: " + ", ".join(dropped_keys))
    merged = llm_args_mod.update_llm_args_with_extra_dict(
        base_dict,
        extra_args,
        extra_llm_api_options=None,
    )
    return llm_args_mod.TorchLlmArgs(**merged)


def validate_torchllm_engine_config_file(
    engine_config_path: str,
    model_path: Optional[str],
    service_key: Optional[str] = None,
) -> tuple[Any, Optional[str]]:
    _ = service_key
    engine_args = _load_yaml_payload(engine_config_path)
    llm_args = validate_torchllm_engine_args(engine_args, model_path)
    return llm_args, model_path
