# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local entry point: ``python -m spica --config smart_sweep.yaml``."""

from __future__ import annotations

import argparse

from .config import SmartSearchConfig
from .search import run_smart_search


def main() -> None:
    parser = argparse.ArgumentParser(prog="spica", description="Spica smart sweeper")
    parser.add_argument("--config", required=True, help="Path to a SmartSearchConfig YAML file")
    args = parser.parse_args()

    config = SmartSearchConfig.from_yaml(args.config)
    candidates = run_smart_search(config)
    for i, candidate in enumerate(candidates):
        print(f"{i}: score={candidate.score} used_gpus={candidate.used_gpus}")


if __name__ == "__main__":
    main()
