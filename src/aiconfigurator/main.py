# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import sys

from aiconfigurator import __version__
from aiconfigurator.cli.main import configure_parser as configure_cli_parser
from aiconfigurator.cli.main import main as cli_main
from aiconfigurator.generator.api import generator_cli_helper
from aiconfigurator.logging_utils import setup_logging


def _run_cli(extra_args: list[str]) -> None:
    if generator_cli_helper(extra_args):
        return
    cli_parser = argparse.ArgumentParser(description="AIConfigurator for disaggregated serving deployment.")
    configure_cli_parser(cli_parser)
    cli_args = cli_parser.parse_args(extra_args)
    cli_main(cli_args)


def _show_version(extra_args: list[str]) -> None:
    print(f"aiconfigurator {__version__}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AIConfigurator for disaggregated serving deployment.")
    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)

    # CLI subcommand
    cli_parser = subparsers.add_parser("cli", help="Run CLI interface", add_help=False)
    cli_parser.set_defaults(handler=_run_cli)

    # Version subcommand
    version_parser = subparsers.add_parser("version", help="Show version information", add_help=False)
    version_parser.set_defaults(handler=_show_version)

    args, extras = parser.parse_known_args(argv)

    setup_logging(level=logging.DEBUG if getattr(args, "debug", False) else logging.INFO)

    # extras contains the arguments for the selected sub-command
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("No sub-command handler registered.")
    handler(extras)


if __name__ == "__main__":
    main(sys.argv[1:])
