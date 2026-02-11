# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass

import yaml


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
            self._emit_file(destination, content, artifact_name)

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

    def _emit_file(self, path: str, content: str, artifact_name: str | None = None) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        # Reformat file to remove unnecessary newlines from j2 rendering
        suffix = self._get_file_suffix(artifact_name or path)
        if suffix in ("yaml", "yml") and not self._should_skip_yaml_reformat(artifact_name):
            self._reformat_yaml(path)
        elif suffix == "json":
            self._reformat_json(path)

        if self._is_k8s_yaml(artifact_name):
            self._cleanup_k8s_yaml(path)

        if path.endswith(".sh"):
            current_mode = os.stat(path).st_mode
            os.chmod(path, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    @staticmethod
    def _should_skip_yaml_reformat(artifact_name: str | None) -> bool:
        if not artifact_name:
            return False
        return artifact_name in {"k8s_deploy.yaml", "k8s_bench.yaml"}

    @staticmethod
    def _is_k8s_yaml(artifact_name: str | None) -> bool:
        return artifact_name in {"k8s_deploy.yaml", "k8s_bench.yaml"}

    @staticmethod
    def _cleanup_k8s_yaml(path: str) -> None:
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
            # Drop only fully empty lines to avoid touching literal block content.
            cleaned = [line for line in lines if line not in {"\n", "\r\n"}]
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(cleaned)
        except Exception:
            # If cleanup fails, leave the file as-is
            pass

    @staticmethod
    def _get_file_suffix(name_or_path: str) -> str:
        """Extract file suffix from artifact name or path."""
        if "." not in name_or_path:
            return ""
        return name_or_path.rsplit(".", 1)[1].lower()

    @staticmethod
    def _reformat_yaml(path: str) -> None:
        """Load and reformat YAML file to remove unnecessary newlines."""
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data is not None:
                with open(path, "w", encoding="utf-8") as f:
                    yaml.dump(data, f, sort_keys=False, default_flow_style=False, width=4096)
        except Exception:
            # If parsing fails, leave the file as-is
            pass

    @staticmethod
    def _reformat_json(path: str) -> None:
        """Load and reformat JSON file to remove unnecessary newlines."""
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            # If parsing fails, leave the file as-is
            pass
