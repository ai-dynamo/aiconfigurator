# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
from aiconfigurator.cli.main import main as cli_main, configure_parser as configure_cli_parser
from aiconfigurator.webapp.main import main as webapp_main, configure_parser as configure_webapp_parser

def main():
    parser = argparse.ArgumentParser(
        description='Dynamo AIConfigurator for disaggregated serving deployment.'
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run', required=True)
    
    # CLI subcommand
    cli_parser = subparsers.add_parser('cli', help='Run CLI interface')
    configure_cli_parser(cli_parser)
    cli_parser.set_defaults(func=cli_main)
    
    # Webapp subcommand  
    webapp_parser = subparsers.add_parser('webapp', help='Run Web interface')
    configure_webapp_parser(webapp_parser)
    webapp_parser.set_defaults(func=webapp_main)

    # Parse args
    args = parser.parse_args()
    
    # Call appropriate function
    args.func(args)

if __name__ == "__main__":
    main()