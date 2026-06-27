# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure, framework-free case builders for migrated collector operations."""

from collector.planner.schemas.attention import (
    ATTENTION_CONTEXT_SCHEMA,
    ATTENTION_GENERATION_SCHEMA,
    ENCODER_ATTENTION_SCHEMA,
    AttentionOpCaseSchema,
    build_attention_context_cases,
    build_attention_generation_cases,
    build_encoder_attention_cases,
    register_attention_schemas,
)

__all__ = [
    "ATTENTION_CONTEXT_SCHEMA",
    "ATTENTION_GENERATION_SCHEMA",
    "ENCODER_ATTENTION_SCHEMA",
    "AttentionOpCaseSchema",
    "build_attention_context_cases",
    "build_attention_generation_cases",
    "build_encoder_attention_cases",
    "register_attention_schemas",
]
