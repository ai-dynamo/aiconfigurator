# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import stat
from dataclasses import dataclass


@dataclass
class ArtifactWriter:
    output_dir: str
    prefer_disagg: bool
    has_agg_role: bool

    def write(self, artifacts: dict[str, str]) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        for artifact_name, content in artifacts.items():
            destination = self._destination_for(artifact_name)
            if destination is None:
                continue
            self._emit_file(destination, content)

    def _destination_for(self, artifact_name: str) -> str | None:
        if artifact_name.startswith("cli_args_"):
            return None
        if artifact_name == "k8s_deploy.yaml":
            return os.path.join(self.output_dir, artifact_name)
        if artifact_name.startswith("extra_engine_args_"):
            if not self._should_emit_engine_file(artifact_name):
                return None
            mapped = self._map_engine_name(artifact_name)
            return os.path.join(self.output_dir, mapped)
        if artifact_name == "run.sh":
            mapped = "run_x.sh"
        else:
            mapped = artifact_name
        return os.path.join(self.output_dir, mapped)

    def _should_emit_engine_file(self, artifact_name: str) -> bool:
        if "agg" in artifact_name and self.prefer_disagg:
            return False
        has_prefill_or_decode = ("prefill" in artifact_name) or ("decode" in artifact_name)
        return not (self.has_agg_role and not self.prefer_disagg and has_prefill_or_decode)

    @staticmethod
    def _map_engine_name(artifact_name: str) -> str:
        if artifact_name == "extra_engine_args_agg.yaml":
            return "agg_config.yaml"
        if artifact_name == "extra_engine_args_prefill.yaml":
            return "prefill_config.yaml"
        if artifact_name == "extra_engine_args_decode.yaml":
            return "decode_config.yaml"
        return artifact_name

    def _emit_file(self, path: str, content: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        if path.endswith(".sh"):
            current_mode = os.stat(path).st_mode
            os.chmod(path, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
