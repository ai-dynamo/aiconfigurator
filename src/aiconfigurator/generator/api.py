# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
API for backend config generation (runtime-first).

This module exposes a single entry point that:
- Accepts runtime objects (AIConfiguratorConfig & AIConfiguratorResult).
- Selects the proper backend generator.
- Optionally saves the generated artifacts to disk.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
from .inputs.schema import DynamoConfig, GeneratorContext
from .inputs.parser import InputParser
from .backends import get_generator
from .utils.writers import save_artifacts
from .types import ArtifactBundle
from pandas import DataFrame
import yaml

class _GenerateAPI:
    @staticmethod
    def from_runtime(
        cfg: dict,
        backend: Optional[str],        
        version: Optional[str],
        overrides: DynamoConfig = DynamoConfig(),
        save_dir: Optional[str] = None,
    ) -> ArtifactBundle:
        """Generate artifacts from runtime objects."""
        ctx: GeneratorContext = InputParser.from_runtime(
            cfg=cfg,
            overrides=overrides,
            backend=backend,
            version=version,
        )
        gen = get_generator(ctx.backend)
        artifacts = gen.generate(ctx)
        if save_dir:
            save_artifacts(artifacts.by_mode, root_dir=save_dir)
        return artifacts

    @staticmethod
    def from_file(
        yaml_path: str,
        backend: Optional[str],
        version: Optional[str],
        overrides: DynamoConfig = DynamoConfig(),
        save_dir: Optional[str] = None,
    ) -> ArtifactBundle:
        """Generate artifacts from files."""
        try:
            with open(yaml_path, "r") as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to load yaml file: {yaml_path}") from e
        ctx: GeneratorContext = InputParser.from_runtime(
            cfg=cfg,
            overrides=overrides,
            version=version,
            backend=backend,
        )
        gen = get_generator(ctx.backend)
        artifacts = gen.generate(ctx)
        if save_dir:
            save_artifacts(artifacts.by_mode, root_dir=save_dir)
        return artifacts

# public alias
generate_backend_config = _GenerateAPI
