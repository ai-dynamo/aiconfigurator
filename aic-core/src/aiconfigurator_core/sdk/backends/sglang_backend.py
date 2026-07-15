# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import ClassVar

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.backends.base_backend import BaseBackend

logger = logging.getLogger(__name__)


class SGLANGBackend(BaseBackend):
    """SGLANG backend.

    Carries higher activation/system overheads than TRT-LLM to reflect SGLANG's
    Python execution overhead, plus a larger minimum activation budget.
    """

    # Per-family activation scaling tuned for SGLANG (Python overhead +
    # dynamic execution => higher coefficients than TRT-LLM).
    ACTIVATION_COEFFICIENTS: ClassVar[dict[str, dict[int, float]]] = {
        "GPT": {1: 13, 2: 8, 4: 6.5, 8: 6.5},
        "LLAMA": {1: 14, 2: 8.5, 4: 6.5, 8: 6.5},
        "MOE": {1: 28, 2: 17, 4: 13, 8: 13},
        "GEMMA4MIX": {1: 28, 2: 17, 4: 13, 8: 13},
        "DEEPSEEK": {1: 28, 2: 17, 4: 13, 8: 13},
        "DEEPSEEKV32": {1: 28, 2: 17, 4: 13, 8: 13},
        "DEEPSEEKV4": {1: 28, 2: 17, 4: 13, 8: 13},
        "KIMIK25": {1: 28, 2: 17, 4: 13, 8: 13},
        "default": {1: 13, 2: 8, 4: 6.5, 8: 6.5},
    }
    MIN_ACTIVATION_BYTES = 90 * 1024 * 1024  # higher floor than TRT-LLM's 70 MiB
    ACTIVATION_OVERHEAD_FRAC = 0.15  # 15% additional activation overhead
    OTHERS_OVERHEAD_FRAC = 0.20  # 20% additional system overhead

    def __init__(self):
        super().__init__()
        self.name = common.BackendName.sglang

    def _tpot_mix_steps(self, num_mix_steps: int) -> int:
        # Same pipeline-drain correction as TRT-LLM: ~3 steps elapse before
        # new requests can be enqueued after the last prefill finishes.
        return max(1, num_mix_steps - 3)
