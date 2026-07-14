// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! (Retired.) The `overlap_composition` helper that lived here implemented
//! the OPPOSITE composition of Python\'s `OverlapOp` (max-within-group /
//! sum-across-groups instead of sum-within / max-across) and had no
//! production caller — the correct overlap semantics live on
//! `operators::op::OverlapOp` (`latency = max(sum(group_a), sum(group_b))`,
//! mirroring `aiconfigurator.sdk.operations.overlap.OverlapOp`). Deleted
//! rather than fixed so nobody wires the wrong composition by accident.
