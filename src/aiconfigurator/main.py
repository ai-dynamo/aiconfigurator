# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
from aiconfigurator.cli.main import main as cli_main, configure_parser as configure_cli_parser
from aiconfigurator.webapp.main import main as webapp_main, configure_parser as configure_webapp_parser
from aiconfigurator.eval.main import main as eval_main, configure_parser as configure_eval_parser


def _run_cli(extra_args: list[str]) -> None:
    cli_parser = argparse.ArgumentParser(
        description="Dynamo AIConfigurator for disaggregated serving deployment."
    )
    configure_cli_parser(cli_parser)
    cli_args = cli_parser.parse_args(extra_args)
    cli_main(cli_args)


def _run_webapp(extra_args: list[str]) -> None:
    webapp_parser = argparse.ArgumentParser(
        description="Dynamo AIConfigurator web interface"
    )
    configure_webapp_parser(webapp_parser)
    webapp_args = webapp_parser.parse_args(extra_args)
    webapp_main(webapp_args)


def _run_eval(extra_args: list[str]) -> None:
    eval_parser = argparse.ArgumentParser(
        description="Generate config -> Launch Service -> Benchmarking -> Analysis"
    )
    configure_eval_parser(eval_parser)
    eval_args = eval_parser.parse_args(extra_args)
    eval_main(eval_args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Dynamo AIConfigurator for disaggregated serving deployment.'
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run', required=True)
    
    # CLI subcommand
    cli_parser = subparsers.add_parser('cli', help='Run CLI interface', add_help=False)
    cli_parser.set_defaults(handler=_run_cli)
    
    # Webapp subcommand  
    webapp_parser = subparsers.add_parser('webapp', help='Run Web interface', add_help=False)
    webapp_parser.set_defaults(handler=_run_webapp)

    # Eval subcommand  
    eval_parser = subparsers.add_parser('eval', help='Generate config -> Launch Service -> Benchmarking -> Analysis', add_help=False)
    eval_parser.set_defaults(handler=_run_eval)

    args, extras = parser.parse_known_args(argv)

    # extras contains the arguments for the selected sub-command
    handler = getattr(args, 'handler', None)
    if handler is None:
        parser.error('No sub-command handler registered.')
    handler(extras)

if __name__ == "__main__":
    main(sys.argv[1:])