# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
API for backend config generation (runtime-first).

This module exposes a single entry point that:
- Accepts runtime objects (AIConfiguratorConfig & AIConfiguratorResult).
- Selects the proper backend generator.
- Optionally saves the generated artifacts to disk.
"""

import yaml

from .backends import get_generator
from .inputs.parser import InputParser
from .inputs.schema import DynamoConfig, GeneratorContext
from .types import ArtifactBundle
from .utils.writers import save_artifacts


class _GenerateAPI:
    @staticmethod
    def from_runtime(
        cfg: dict,
        backend: str | None,
        version: str | None,
        overrides: DynamoConfig | None = None,
        save_dir: str | None = None,
    ) -> ArtifactBundle:
        """Generate artifacts from runtime objects."""
        if overrides is None:
            overrides = DynamoConfig()

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
        backend: str | None,
        version: str | None,
        overrides: DynamoConfig | None = None,
        save_dir: str | None = None,
    ) -> ArtifactBundle:
        """Generate artifacts from files."""
        if overrides is None:
            overrides = DynamoConfig()

        try:
            with open(yaml_path) as f:
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
