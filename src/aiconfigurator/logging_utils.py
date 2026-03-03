# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logging utilities for aiconfigurator."""

import logging
import os
import sys


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Check if colors should be disabled (e.g., when output is redirected)
        self.use_colors = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

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


def setup_logging(level=logging.INFO):
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create console handler with colored formatter
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = ColoredFormatter(
        "%(asctime)s [aiconfigurator] [%(levelname).1s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
