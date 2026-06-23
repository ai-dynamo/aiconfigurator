"""PYTHONPATH-injected bootstrap for FPM expert-routing capture.

The Dynamo vLLM worker spawns EngineCore / TP-worker subprocesses; Python imports
``sitecustomize`` from ``PYTHONPATH`` at interpreter startup, so dropping this
directory on ``PYTHONPATH`` (``export PYTHONPATH=<inject_dir>:$PYTHONPATH``)
installs the capture hook in every worker process without any code mount.

Enabled only when ``FPM_ROUTING_STAGE`` is set (A or B). See
``fpm_routing_capture.py`` for the hook itself and ``EXPERT_ROUTING_CAPTURE.md``
for the runbook.
"""

from __future__ import annotations

import os
import sys

if os.environ.get("FPM_ROUTING_STAGE"):
    try:
        import fpm_routing_capture  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised in vLLM subprocesses
        print(
            f"[fpm-routing-sitecustomize] failed to import fpm_routing_capture: {exc!r}",
            file=sys.stderr,
        )
