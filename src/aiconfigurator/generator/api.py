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

class _GenerateAPI:
    @staticmethod
    def from_runtime(
        cfg,  # AIConfiguratorConfig
        res,  # AIConfiguratorResult
        overrides: DynamoConfig,
        version: Optional[str],
        backend: Optional[str] = None,
        save_dir: Optional[str] = None,
    ) -> ArtifactBundle:
        """Generate artifacts from runtime objects."""
        ctx: GeneratorContext = InputParser.from_runtime(
            cfg=cfg, res=res, overrides=overrides, version=version, backend=backend
        )
        gen = get_generator(ctx.backend)
        artifacts = gen.generate(ctx)
        if save_dir:
            save_artifacts(artifacts.by_mode, root_dir=save_dir)
        return artifacts

    @staticmethod
    def from_files(
        cfg_path: str,
        res_path: str,
        overrides: Optional[DynamoConfig] = None,
        *,
        backend: Optional[str] = None,
        version: Optional[str] = None,
        model_name: Optional[str] = None,
        is_moe: Optional[bool] = None,
        total_gpus: Optional[int] = None,
        agg_csv_path: Optional[str] = None,
        disagg_csv_path: Optional[str] = None,
        save_dir: Optional[str] = None,
    ) -> ArtifactBundle:
        """
        Generate artifacts from file paths.

        Args:
            cfg_path: Path to aiconfigurator_config.yaml
            res_path: Path to aiconfigurator_result.json
            overrides: Optional DynamoConfig with dynamo_config.* overrides
            backend/version: Optional explicit override; if None, infer from config YAML
            model_name: Optional model name for scripts; default 'unknown_model'
            is_moe: Optional explicit flag; if None, default False (or read from overrides.extras)
            total_gpus: If provided, choose best row using cluster-scale metric; else fall back to tokens/s/gpu
            agg_csv_path / disagg_csv_path: Optional explicit pareto CSV paths; defaults to sibling files of cfg_path
            save_dir: If provided, persist generated files immediately
        """
        overrides = overrides or DynamoConfig()
        ctx: GeneratorContext = InputParser.from_files(
            cfg_path=cfg_path,
            res_path=res_path,
            overrides=overrides,
            backend=backend,
            version=version,
            model_name=model_name,
            is_moe=is_moe,
            total_gpus=total_gpus,
            agg_csv_path=agg_csv_path,
            disagg_csv_path=disagg_csv_path,
        )
        gen = get_generator(ctx.backend)
        artifacts = gen.generate(ctx)
        if save_dir:
            save_artifacts(artifacts.by_mode, root_dir=save_dir)
        return artifacts

# public alias
generate_backend_config = _GenerateAPI
