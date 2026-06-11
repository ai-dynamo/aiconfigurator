# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM engine argument construction and engine creation for layerwise workers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_COMMON_DIR = _THIS_DIR.parent / "common"
sys.path.insert(0, str(_COMMON_DIR))

from vllm_deployment import VllmDeploymentConfig, build_engine_args, has_cli_flag

try:
    from .data import DataPoint
except ImportError:  # pragma: no cover - direct script compatibility
    from data import DataPoint


DEFAULT_EXTRA_VLLM_ARGS = (
    ("flag", "--skip-mm-profiling", ("--no-skip-mm-profiling",)),
    ("pair", "--limit-mm-per-prompt", '{"image":0,"video":0}'),
    ("pair", "--generation-config", "vllm"),
)


def _append_default_vllm_args(extra_vllm_args: list[str]) -> None:
    for kind, flag, value_or_aliases in DEFAULT_EXTRA_VLLM_ARGS:
        if kind == "flag":
            aliases = tuple(value_or_aliases)
            if not has_cli_flag(extra_vllm_args, flag, *aliases):
                extra_vllm_args.append(flag)
        elif not has_cli_flag(extra_vllm_args, flag):
            extra_vllm_args.extend([flag, value_or_aliases])

def _engine_tokens(
    *,
    model_dir: str,
    datapoints: list[DataPoint],
    extra_vllm_args: list[str],
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
    max_model_len: int | None = None,
    gen_driver: str = "prefix_cache",
) -> list[str]:
    tokens = build_engine_args(
        VllmDeploymentConfig(
            model=model_dir,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            gpu_memory_utilization=0.9,
        )
    )
    tokens.extend(extra_vllm_args)
    return tokens

def _create_llm(engine_tokens: list[str], *, enable_layerwise_nvtx_tracing: bool = True):
    """Create a vLLM LLM instance with collector-specific parser defaults."""
    from vllm.engine.arg_utils import EngineArgs

    parser = argparse.ArgumentParser(add_help=False)
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(
        load_format="dummy",
        trust_remote_code=True,
        enable_layerwise_nvtx_tracing=enable_layerwise_nvtx_tracing,
        skip_tokenizer_init=True,
    )
    args = parser.parse_args(engine_tokens)
    engine_args = EngineArgs.from_cli_args(args)
    from vllm import LLM

    return LLM.from_engine_args(engine_args)
