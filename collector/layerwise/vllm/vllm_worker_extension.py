# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM worker extension that installs layerwise patches in spawned workers.

Physical tensor-parallel diagnostics run vLLM's normal multiprocess worker
path.  Parent-process monkey patches do not reach those spawned workers, so
this extension is loaded through vLLM's ``--worker-extension-cls`` hook and
imports the patch module before ``Worker.load_model`` runs.
"""

from __future__ import annotations

import vllm_layer_skip_patch  # noqa: F401  - import installs the load_model patch


class LayerwiseWorkerExtension:
    """Marker extension used only to trigger import-time layerwise patching."""

