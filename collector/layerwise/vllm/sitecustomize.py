"""Install layerwise vLLM hooks inside spawned vLLM subprocesses.

vLLM physical tensor-parallel runs use spawned EngineCore/worker processes, so
patches imported by the parent layerwise worker do not automatically apply
where scheduling/model execution actually happens.  Python imports
``sitecustomize`` from ``PYTHONPATH`` during interpreter startup; the layerwise
worker places this directory on ``PYTHONPATH`` before creating the vLLM engine.
"""

from __future__ import annotations

import os
import sys


def _try_import(module: str) -> None:
    """Import a best-effort instrumentation module without breaking startup."""

    try:
        __import__(module)
    except Exception as exc:  # pragma: no cover - exercised in vLLM subprocesses
        print(f"[layerwise-sitecustomize] failed to import {module}: {exc!r}", file=sys.stderr)


if os.environ.get("LAYERWISE_SCHEDULER_TIMING", "0") == "1":
    _try_import("vllm_scheduler_timing_patch")

if os.environ.get("LAYERWISE_STEP_MARKER") == "1":
    _try_import("vllm_step_marker")
