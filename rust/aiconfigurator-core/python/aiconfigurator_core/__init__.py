# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Rust accelerator for ``aiconfigurator``'s forward-pass latency estimator.

This package ships the compiled PyO3 extension as the private submodule ``_core``
and re-exports its classes. The Python wrapper, public API, and the
``--engine-step-backend rust`` opt-in live in the ``aiconfigurator`` package
(``aiconfigurator.sdk.rust_engine_step``).

Maturin drops the cdylib next to this file as ``_core.abi3.so`` (or ``.pyd`` on
Windows).
"""

from ._core import PyEngineStepEstimator, PyForwardPassPerfModel

__all__ = ["PyEngineStepEstimator", "PyForwardPassPerfModel"]
