# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level package for the compiled ``aiconfigurator_core`` Rust extension.

The actual extension is built by maturin as the submodule
``aiconfigurator_core._aiconfigurator_core``. This package re-exports its
public surface so callers can ``import aiconfigurator_core`` directly. It
exports the compiled-engine pyclass ``AicEngine``
(``from_spec`` / ``run_static`` / ``predict_prefill_latency`` /
``predict_decode_latency`` / ``mixed_step_latency`` / ``decode_step_latency``),
the op-transfer ``#[pyfunction]`` ``engine_spec_bincode_from_json`` (JSON
``EngineSpec`` -> bincode bytes, the Python -> Rust op-transfer wire), and the
build smoke check ``_build_smoke``.

It also exports the forward-pass perf model pyclass
``RustForwardPassPerfModel`` (PR #1152, built on the compiled ``Engine``): the
tuned/fallback forward-pass latency model with online correction, regression
fallback, and diagnostics.

KV-cache capacity estimation is not exported here: it is computed in Python
(``aiconfigurator_core.sdk.memory``), and the Rust
``aiconfigurator_core::memory::estimate_kv_cache`` is a Rust-only forwarder for
embedded callers (e.g. the Dynamo Mocker), like ``build_aic_engine``.
"""

from ._aiconfigurator_core import (
    AicEngine,
    RustForwardPassPerfModel,
    _build_smoke,
    engine_spec_bincode_from_json,
)

__all__ = [
    "AicEngine",
    "RustForwardPassPerfModel",
    "_build_smoke",
    "engine_spec_bincode_from_json",
]
