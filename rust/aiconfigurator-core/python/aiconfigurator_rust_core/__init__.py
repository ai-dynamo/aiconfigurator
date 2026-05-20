# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Rust accelerator for ``aiconfigurator``'s forward-pass latency estimator.

This package only ships the compiled PyO3 extension ``aiconfigurator_core``.
The Python wrapper, public API, and the ``--engine-step-backend rust`` opt-in
live in the ``aiconfigurator`` package (``aiconfigurator.sdk.rust_engine_step``).

Maturin drops the cdylib next to this file as
``aiconfigurator_core.abi3.so`` (or ``.pyd`` on Windows).
"""

from . import aiconfigurator_core

__all__ = ["aiconfigurator_core"]
