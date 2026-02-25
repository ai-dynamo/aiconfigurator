# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import sys

from aiconfigurator import __version__
from aiconfigurator.cli.main import configure_parser as configure_cli_parser
from aiconfigurator.cli.main import main as cli_main
from aiconfigurator.eval.main import configure_parser as configure_eval_parser
from aiconfigurator.eval.main import main as eval_main
from aiconfigurator.generator.api import generator_cli_helper

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _is_gradio_importable() -> bool:
    """Check that gradio is installed and actually importable (not just leftover files)."""
    try:
        import gradio

        return hasattr(gradio, "__version__")
    except Exception:
        return False


_GRADIO_AVAILABLE = _is_gradio_importable()


def _run_webapp(extra_args: list[str]) -> None:
    if not _GRADIO_AVAILABLE:
        print(
            "Error: The 'webapp' subcommand requires a working Gradio installation.\n"
            "Install it with:  pip install aiconfigurator[webapp]"
        )
        sys.exit(1)
    from aiconfigurator.webapp.main import configure_parser as configure_webapp_parser
    from aiconfigurator.webapp.main import main as webapp_main

    webapp_parser = argparse.ArgumentParser(description="Dynamo AIConfigurator web interface")
    configure_webapp_parser(webapp_parser)
    webapp_args = webapp_parser.parse_args(extra_args)
    webapp_main(webapp_args)


def _run_cli(extra_args: list[str]) -> None:
    if generator_cli_helper(extra_args):
        return
    cli_parser = argparse.ArgumentParser(description="Dynamo AIConfigurator for disaggregated serving deployment.")
    configure_cli_parser(cli_parser)
    cli_args = cli_parser.parse_args(extra_args)
    cli_main(cli_args)


def _run_eval(extra_args: list[str]) -> None:
    eval_parser = argparse.ArgumentParser(description="Generate config -> Launch Service -> Benchmarking -> Analysis")
    configure_eval_parser(eval_parser)
    eval_args = eval_parser.parse_args(extra_args)
    eval_main(eval_args)


def _show_version(extra_args: list[str]) -> None:
    print(f"aiconfigurator {__version__}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Dynamo AIConfigurator for disaggregated serving deployment.")
    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)

    # CLI subcommand
    cli_parser = subparsers.add_parser("cli", help="Run CLI interface", add_help=False)
    cli_parser.set_defaults(handler=_run_cli)

    # Webapp subcommand (always visible; prints install hint when Gradio is missing)
    webapp_help = (
        "Run Web interface" if _GRADIO_AVAILABLE else "Run Web interface (requires: pip install aiconfigurator[webapp])"
    )
    webapp_parser = subparsers.add_parser("webapp", help=webapp_help, add_help=False)
    webapp_parser.set_defaults(handler=_run_webapp)

    # Eval subcommand
    eval_parser = subparsers.add_parser(
        "eval", help="Generate config -> Launch Service -> Benchmarking -> Analysis", add_help=False
    )
    eval_parser.set_defaults(handler=_run_eval)

    # Version subcommand
    version_parser = subparsers.add_parser("version", help="Show version information", add_help=False)
    version_parser.set_defaults(handler=_show_version)

    args, extras = parser.parse_known_args(argv)

    # extras contains the arguments for the selected sub-command
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("No sub-command handler registered.")
    handler(extras)


if __name__ == "__main__":
    main(sys.argv[1:])
