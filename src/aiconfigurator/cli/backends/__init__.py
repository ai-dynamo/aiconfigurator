# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module
from typing import Callable, Dict, Any

BackendGenerator = Callable[
    ["AIConfiguratorResult", "AIConfiguratorConfig", "DynamoConfig", str],
    Dict[str, Dict[str, Any]],
]

_REG: Dict[str, str] = {
    "trtllm": ".trtllm",
    "vllm": ".vllm",
    "sglang": ".sglang",
}

def get_config_generator(backend: str) -> BackendGenerator:
    if backend not in _REG:
        raise ValueError(f"Unknown backend {backend}")
    mod = import_module(__name__ + _REG[backend])
    return getattr(mod, "generate_backend_config")