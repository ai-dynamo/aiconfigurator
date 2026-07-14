# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

import uvicorn

from aiconfigurator.logging_utils import setup_logging
from aiconfigurator.sdk import perf_database

from .app import create_app


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="AIConfigurator service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the service to.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the service to.")
    parser.add_argument("--log-level", default="info", help="Uvicorn log level.")
    parser.add_argument(
        "--systems-paths",
        default=None,
        help="Comma-separated systems database roots. Use 'default' to include built-in data.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging()
    if args.systems_paths is not None:
        perf_database.set_systems_paths(args.systems_paths)
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
