# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

class AicEngine:
    @staticmethod
    def from_spec(bytes: bytes, systems_path: str | None = None) -> AicEngine: ...  # noqa: A002
    def run_static(
        self,
        batch_size: int,
        beam_width: int,
        isl: int,
        osl: int,
        prefix: int,
        seq_imbalance_correction_scale: float,
        gen_seq_imbalance_correction_scale: float,
        mode: str = "static",
        stride: int = 32,
    ) -> tuple[float, float, float]: ...
    def predict_prefill_latency(self, bs: int, isl: int, prefix: int = 0) -> float: ...
    def predict_decode_latency(self, bs: int, isl: int, osl: int = 2) -> float: ...
    def mixed_step_latency(
        self,
        ctx_tokens: int,
        gen_tokens: int,
        isl: int,
        osl: int,
        prefix: int = 0,
        seq_imbalance_correction_scale: float = 1.0,
        gen_seq_imbalance_correction_scale: float = 1.0,
    ) -> float: ...
    def mixed_step_breakdown(
        self,
        ctx_tokens: int,
        gen_tokens: int,
        isl: int,
        osl: int,
        prefix: int = 0,
        seq_imbalance_correction_scale: float = 1.0,
        gen_seq_imbalance_correction_scale: float = 1.0,
    ) -> tuple[float, float, float, float]: ...
    def decode_step_latency(
        self,
        gen_tokens: int,
        isl: int,
        osl: int,
        gen_seq_imbalance_correction_scale: float = 1.0,
    ) -> float: ...
    def run_static_per_op(
        self,
        batch_size: int,
        beam_width: int,
        isl: int,
        osl: int,
        prefix: int,
        seq_imbalance_correction_scale: float,
        gen_seq_imbalance_correction_scale: float,
        mode: str = "static",
        stride: int = 32,
    ) -> tuple[
        list[tuple[str, float, float, str]],
        list[tuple[str, float, float, str]],
    ]: ...
    def mixed_step_breakdown_per_op(
        self,
        ctx_tokens: int,
        gen_tokens: int,
        isl: int,
        osl: int,
        prefix: int = 0,
        seq_imbalance_correction_scale: float = 1.0,
        gen_seq_imbalance_correction_scale: float = 1.0,
    ) -> tuple[
        list[tuple[str, float, float, str]],
        list[tuple[str, float, float, str]],
        list[tuple[str, float, float, str]],
    ]: ...
    def decode_step_per_op(
        self,
        gen_tokens: int,
        isl: int,
        osl: int,
        gen_seq_imbalance_correction_scale: float = 1.0,
    ) -> list[tuple[str, float, float, str]]: ...
    def evaluate_context_ops(
        self,
        indices: list[int],
        batch_size: int,
        s: int,
        prefix: int = 0,
        seq_imbalance_correction_scale: float = 1.0,
        x: int | None = None,
    ) -> list[tuple[str, float, float, str]]: ...
    def evaluate_generation_ops(
        self,
        indices: list[int],
        batch_size: int,
        s: int,
        gen_seq_imbalance_correction_scale: float = 1.0,
        prefix: int = 0,
        x: int | None = None,
    ) -> list[tuple[str, float, float, str]]: ...
    def evaluate_ops_json(
        self,
        ops_json: str,
        is_context: bool,
        batch_size: int,
        s: int,
        prefix: int = 0,
        imbalance_correction_scale: float = 1.0,
        x: int | None = None,
    ) -> list[tuple[str, float, float, str]]: ...
    def last_provenance(self) -> str | None: ...

class RustForwardPassPerfModel:
    @staticmethod
    def from_native(config_json: str, options_json: str | None = None) -> RustForwardPassPerfModel: ...
    @staticmethod
    def best_available(config_json: str, options_json: str | None = None) -> RustForwardPassPerfModel: ...
    @staticmethod
    def from_regression(options_json: str | None = None) -> RustForwardPassPerfModel: ...
    def estimate_forward_pass_time_ms(self, fpm_json: str) -> float | None: ...
    def tune_with_fpms(self, iterations_json: str) -> None: ...
    def diagnostics(self) -> str: ...
    def min_correction_factor(self) -> float | None: ...
    def max_correction_factor(self) -> float | None: ...
    def avg_correction_factor(self) -> float | None: ...

def engine_spec_bincode_from_json(spec_json: str) -> list[int]: ...
def _build_smoke() -> int: ...
