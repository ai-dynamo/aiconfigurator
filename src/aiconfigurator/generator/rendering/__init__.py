# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rendering package for generator inputs and backend artifacts.

This module exposes a single import surface so callers do not need to know
where schemas, mapping logic, or rule engines live internally.
"""

from .engine import (
    _cast_literal,
    evaluate_expression,
    get_param_keys,
    load_yaml_mapping,
    prepare_template_context,
    render_backend_parameters,
    render_backend_templates,
    render_parameters,
)
from .rule_engine import apply_rule_plugins
from .schemas import apply_defaults

__all__ = [
    "_cast_literal",
    "apply_defaults",
    "apply_rule_plugins",
    "evaluate_expression",
    "get_param_keys",
    "load_yaml_mapping",
    "prepare_template_context",
    "render_backend_parameters",
    "render_backend_templates",
    "render_parameters",
]
