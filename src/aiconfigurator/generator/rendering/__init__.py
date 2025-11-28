"""
Rendering package for generator inputs and backend artifacts.

This module exposes a single import surface so callers do not need to know
where schemas, mapping logic, or rule engines live internally.
"""

from .engine import (
    evaluate_expression,
    load_yaml_mapping,
    render_parameters,
    render_backend_parameters,
    render_backend_templates,
    prepare_template_context,
    _cast_literal,
    get_param_keys,
)
from .rule_engine import apply_rule_plugins
from .schemas import apply_defaults

__all__ = [
    "apply_defaults",
    "apply_rule_plugins",
    "evaluate_expression",
    "load_yaml_mapping",
    "render_parameters",
    "render_backend_parameters",
    "render_backend_templates",
    "prepare_template_context",
    "_cast_literal",
    "get_param_keys",
]

