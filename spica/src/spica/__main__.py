# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local entry point: ``python -m spica --config smart_sweep.yaml``."""

from __future__ import annotations

import argparse
import sys

import yaml
from pydantic import ValidationError

from .config import SmartSearchConfig
from .search import run_smart_search


def main() -> None:
    parser = argparse.ArgumentParser(prog="spica", description="Spica smart sweeper")
    parser.add_argument("--config", required=True, help="Path to a SmartSearchConfig YAML file")
    args = parser.parse_args()

    try:
        config = SmartSearchConfig.from_yaml(args.config)
    except OSError as exc:  # missing file, a directory, unreadable, etc.
        parser.error(f"could not read config {args.config}: {exc}")
    except yaml.YAMLError as exc:
        parser.error(f"malformed YAML in {args.config}: {exc}")
    except ValidationError as exc:
        parser.error(f"invalid config {args.config}: {exc}")

    candidates = run_smart_search(config)
    if not candidates:
        print("no feasible candidate found (check backends / SLA / gpu_budget / replay errors)", file=sys.stderr)
        sys.exit(1)
    for i, candidate in enumerate(candidates):
        print(f"{i}: score={candidate.score} used_gpus={candidate.used_gpus}")


if __name__ == "__main__":
    main()
