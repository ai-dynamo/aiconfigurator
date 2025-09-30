# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Backend generator base class and registry.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from ..inputs.schema import GeneratorContext
from ..types import ArtifactBundle

_REG = {}

class BaseGenerator(ABC):
    """Abstract generator for a backend."""
    name: str

    @abstractmethod
    def generate(self, ctx: GeneratorContext) -> ArtifactBundle:
        """Generate artifacts from a validated context."""

def register(backend: str):
    """Decorator to register a generator class."""
    def _wrap(cls):
        _REG[backend] = cls()
        return cls
    return _wrap

def get_generator(backend: str) -> BaseGenerator:
    if backend not in _REG:
        raise ValueError(f"Unknown backend '{backend}'. Available: {list(_REG)}")
    return _REG[backend]