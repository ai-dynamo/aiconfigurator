# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import logging
import math
import os
from numbers import Real
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from aiconfigurator.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)

_ALLOWED_FUNCS = {
    "min": min,
    "max": max,
    "ceil": math.ceil,
    "floor": math.floor,
}

_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}

_UNARY_OPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


class EmpiricalCorrectionResolver:
    """Load and evaluate backend-specific empirical correction expressions."""

    def __init__(
        self,
        default_expressions: dict[str, str],
        allowed_vars: dict[str, set[str]],
        config_dir_name: str = "empirical_correction",
        config_file_name: str = "empirical_correction.yaml",
        section_key: str = "corrections",
    ) -> None:
        # `default_expressions` and `allowed_vars` are backend-defined contracts.
        # YAML can override expression bodies, but variable names are still validated
        # against this contract to avoid accidental/unsafe symbol usage.
        self._default_expressions = default_expressions
        self._allowed_vars = allowed_vars
        self._config_dir_name = config_dir_name
        self._config_file_name = config_file_name
        self._section_key = section_key
        self._cache: dict[tuple[str, str, str, str], dict[str, str]] = {}

    def _get_config_path(self, database: "PerfDatabase") -> str | None:
        systems_root = getattr(database, "systems_root", None)
        if not systems_root:
            return None
        return os.path.join(
            systems_root,
            self._config_dir_name,
            database.system,
            database.backend,
            database.version,
            self._config_file_name,
        )

    def _load_overrides(self, database: "PerfDatabase") -> dict[str, str]:
        systems_root = getattr(database, "systems_root", None)
        if not systems_root:
            return {}

        cache_key = (systems_root, database.system, database.backend, database.version)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        config_path = self._get_config_path(database)
        if not config_path or not os.path.isfile(config_path):
            self._cache[cache_key] = {}
            return {}

        try:
            with open(config_path) as f:
                raw_config = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning(
                "Failed to load empirical correction config %s: %s. Falling back to defaults.",
                config_path,
                exc,
            )
            self._cache[cache_key] = {}
            return {}

        if not isinstance(raw_config, dict):
            logger.warning(
                "Invalid empirical correction config %s: expected a mapping at top level. Falling back to defaults.",
                config_path,
            )
            self._cache[cache_key] = {}
            return {}

        corrections = raw_config.get(self._section_key, raw_config)
        if not isinstance(corrections, dict):
            logger.warning(
                "Invalid corrections section in %s: expected a mapping. Falling back to defaults.",
                config_path,
            )
            self._cache[cache_key] = {}
            return {}

        sanitized_corrections: dict[str, str] = {}
        for key, value in corrections.items():
            if not isinstance(value, str):
                logger.warning(
                    "Invalid empirical correction '%s' in %s: expected string expression, got %s. Ignoring override.",
                    key,
                    config_path,
                    type(value).__name__,
                )
                continue
            sanitized_corrections[key] = value.strip()

        self._cache[cache_key] = sanitized_corrections
        return sanitized_corrections

    def _evaluate_expr(self, expr: str, variables: dict[str, float], allowed_vars: set[str]) -> float:
        """Evaluate one empirical-correction expression with a restricted AST evaluator.

        Args:
            expr: Expression string from default config or YAML override.
            variables: Runtime variable values available to the expression.
            allowed_vars: Per-formula allowlist of variable names.

        Returns:
            Numeric evaluation result as float.

        Raises:
            ValueError: If the expression uses disallowed syntax/functions/variables,
                references missing variables, or yields a non-numeric value.
        """
        expression_tree = ast.parse(expr, mode="eval")

        def _assert_numeric(value, context: str):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"Expression value for {context} must be numeric, got {type(value).__name__}")
            return float(value)

        def _eval_node(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return _eval_node(node.body)
            if isinstance(node, ast.Constant):
                return _assert_numeric(node.value, "constant")
            if isinstance(node, ast.Name):
                if node.id in allowed_vars:
                    if node.id not in variables:
                        # The formula references an allowed symbol, but caller did not
                        # provide a runtime value for it.
                        raise ValueError(f"Missing variable '{node.id}' for formula evaluation")
                    return _assert_numeric(variables[node.id], node.id)
                raise ValueError(f"Variable '{node.id}' is not allowed in this formula")
            if isinstance(node, ast.BinOp):
                op_type = type(node.op)
                if op_type not in _BIN_OPS:
                    raise ValueError(f"Operator '{op_type.__name__}' is not allowed in empirical corrections")
                return _BIN_OPS[op_type](_eval_node(node.left), _eval_node(node.right))
            if isinstance(node, ast.UnaryOp):
                op_type = type(node.op)
                if op_type not in _UNARY_OPS:
                    raise ValueError(f"Unary operator '{op_type.__name__}' is not allowed in empirical corrections")
                return _UNARY_OPS[op_type](_eval_node(node.operand))
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise ValueError("Only direct function calls are allowed in empirical corrections")
                func_name = node.func.id
                func = _ALLOWED_FUNCS.get(func_name)
                if func is None:
                    raise ValueError(f"Function '{func_name}' is not allowed in empirical corrections")
                if node.keywords:
                    raise ValueError("Keyword arguments are not allowed in empirical corrections")
                args = [_eval_node(arg) for arg in node.args]
                return _assert_numeric(func(*args), f"function '{func_name}'")
            raise ValueError(f"Expression node '{type(node).__name__}' is not allowed in empirical corrections")

        return _eval_node(expression_tree)

    def evaluate(self, name: str, database: "PerfDatabase", variables: dict[str, float]) -> float:
        """Resolve and evaluate one empirical correction entry.

        Resolution order:
            1. Use YAML override expression if present for ``name``.
            2. Otherwise use backend default expression for ``name``.

        If an override exists but fails to parse/evaluate, this method logs a warning
        and falls back to the default expression for the same ``name``.

        Args:
            name: Correction key, such as ``ttft_correction_factor``.
            database: PerfDatabase instance that provides system/backend/version
                context for locating override YAML.
            variables: Runtime variable values used by expression evaluation.

        Returns:
            Evaluated numeric value as float.
        """
        default_expr = self._default_expressions[name]
        allowed_vars = self._allowed_vars[name]
        corrections = self._load_overrides(database)
        override_expr = corrections.get(name)

        expr = override_expr if override_expr else default_expr
        if override_expr:
            logger.debug("Using override empirical correction '%s': %s", name, override_expr)

        try:
            return self._evaluate_expr(expr=expr, variables=variables, allowed_vars=allowed_vars)
        except Exception as exc:
            if override_expr:
                # Any override parsing/evaluation failure should be non-fatal; we always
                # fall back to the backend default expression to keep estimation running.
                logger.warning(
                    "Failed to evaluate override empirical correction '%s' (%s): %s. Falling back to default expression: %s",
                    name,
                    override_expr,
                    exc,
                    default_expr,
                )
                return self._evaluate_expr(
                    expr=default_expr,
                    variables=variables,
                    allowed_vars=allowed_vars,
                )
            raise
