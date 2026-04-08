# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logging utilities for aiconfigurator."""

import logging
import os
import sys


def _stdout_env_suggests_plain() -> bool:
    """True when NO_COLOR or non-TTY stdout indicates avoiding ANSI."""
    if "NO_COLOR" in os.environ:
        return True
    # Hack to support --no-color arg before argparse.parse_args() is completed.
    if "--no-color" in sys.argv:
        return True
    try:
        return not sys.stdout.isatty()
    except (AttributeError, OSError, ValueError):
        return True


def use_plain_cli_output() -> bool:
    """Return True if human-oriented output should avoid ANSI.

    True when the root logger was configured with setup_logging and
    no_color=True (--no-color), or when NO_COLOR is set, or when stdout
    is not a TTY.

    .. NO_COLOR: https://no-color.org/
    """
    # Check if the root logger was configured with setup_logging and no_color=True
    for handler in logging.getLogger().handlers:
        fmt = getattr(handler, "formatter", None)
        if isinstance(fmt, ColoredFormatter) and fmt.force_no_color:
            return True

    return _stdout_env_suggests_plain()


def _cli_bold(text: str) -> str:
    """Wrap *text* in bold SGR when use_plain_cli_output is false."""
    if use_plain_cli_output():
        return text
    return f"\033[1m{text}\033[0m"


def _cli_underline(text: str) -> str:
    """Wrap *text* in underline SGR when use_plain_cli_output is false."""
    if use_plain_cli_output():
        return text
    return f"\033[4m{text}\033[0m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log output.

    Example:
    17:56:43 [aiconfigurator] [utils.py:664] [I] Hello, world!

    Colors:
    - Header (time, [aiconfigurator], filename) in grey
    - Log level icon ([I], [W], [E], [D]) based on level:
      - INFO: Blue
      - WARNING: Yellow
      - ERROR: Red
      - DEBUG: Cyan
    - Message stays default color.
    """

    # ANSI color codes
    GREY = "\033[90m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    def __init__(self, *args, force_no_color: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_no_color = force_no_color
        self.use_colors = not force_no_color and not _stdout_env_suggests_plain()

    def format(self, record):
        # Get the base formatted message
        log_message = super().format(record)

        if not self.use_colors:
            return log_message

        # Check if format contains [aiconfigurator]
        if "[aiconfigurator]" in log_message:
            parts = log_message.split(" ", 3)
            if len(parts) >= 4:
                time_part = parts[0]
                aiconfig_part = parts[1]
                level_part = parts[2]  # [L]
                rest = parts[3]

                bracket_end = rest.find("]")
                if bracket_end != -1:
                    filename_part = rest[: bracket_end + 1]
                    message_part = rest[bracket_end + 1 :].lstrip()  # Skip "]" and any whitespace

                    colored_header = f"{self.GREY}{time_part} {aiconfig_part} {filename_part}{self.RESET}"
                    level_char = level_part[1] if len(level_part) > 1 else " "
                    colored_level = self._color_level(level_char, level_part)
                    return f"{colored_header} {colored_level} {message_part}"

        # Unhandled format. Return as is.
        return log_message

    def _color_level(self, level_char, level_part):
        """Color the log level based on the first character."""
        color_map = {
            "I": self.BLUE,
            "W": self.YELLOW,
            "E": self.RED,
            "D": self.CYAN,
        }
        if level_char in color_map:
            return f"{color_map[level_char]}{level_part}{self.RESET}"
        return level_part


def setup_logging(level=logging.INFO, *, no_color: bool = False):
    """Configure root logging to stdout.

    Args:
        level: Minimum log level for the root logger.
        no_color: If True (CLI --no-color), disable ANSI in log lines and treat all
            CLI summaries as plain; see use_plain_cli_output. TTY and NO_COLOR
            are combined inside ColoredFormatter and use_plain_cli_output.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create console handler with colored formatter
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = ColoredFormatter(
        "%(asctime)s [aiconfigurator] [%(levelname).1s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
        force_no_color=no_color,
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
