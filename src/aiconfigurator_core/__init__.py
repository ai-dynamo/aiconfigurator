# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level package for the compiled ``aiconfigurator_core`` Rust extension.

The actual extension is built by maturin as the submodule
``aiconfigurator_core._aiconfigurator_core``. This package re-exports its
public surface so callers can ``import aiconfigurator_core`` directly. As of
Phase 1.5 commit E5 it exports the compiled-engine pyclass ``AicEngine``
(``from_spec`` / ``run_static`` / ``predict_prefill_latency`` /
``predict_decode_latency`` / ``mixed_step_latency`` / ``decode_step_latency``),
the op-transfer ``#[pyfunction]`` ``engine_spec_bincode_from_json`` (JSON
``EngineSpec`` -> bincode bytes, the Python -> Rust op-transfer wire), plus the
retained build smoke check ``_phase15_smoke``.

``build_aic_engine`` is intentionally NOT re-exported: it is a Rust-only entry
point for embedded callers (e.g. the Dynamo Mocker), not part of the Python
surface.
"""

from ._aiconfigurator_core import AicEngine, _phase15_smoke, engine_spec_bincode_from_json

__all__ = ["AicEngine", "_phase15_smoke", "engine_spec_bincode_from_json"]
