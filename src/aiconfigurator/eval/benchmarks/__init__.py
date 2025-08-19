# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Will support different benchmark backend
from typing import Callable, Dict, Any

_REGISTRY: Dict[str, Dict[str, Any]] = {}

def register(name: str, *, parse=None):
    def _wrap(fn: Callable[..., None]):
        _REGISTRY[name] = {"run": fn, "parse": parse}
        return fn
    return _wrap

def get(name: str):
    return _REGISTRY[name]
