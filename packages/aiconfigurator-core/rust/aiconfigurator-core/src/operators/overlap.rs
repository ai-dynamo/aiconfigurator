// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Overlap composition: max-of-groups latency aggregation.
//!
//! Mirrors `aiconfigurator_core.sdk.operations.overlap.Overlap`. When two
//! kernels can execute in parallel (e.g. compute on one stream while a
//! collective runs on another), AIC composes their effective latency as
//! the max of the group rather than the sum. This helper takes a list of
//! `(latency_ms, group_id)` entries and returns `sum(group_max)` across
//! groups.

use crate::operators::base::{PerformanceResult, Source};

/// Compose a list of latencies grouped by overlap-group identifier.
/// Within each group, the effective latency is the max; total is the
/// sum across groups. Sources combine the same way `Source::combine`
/// behaves — same tag across all entries keeps it; any disagreement
/// becomes `Source::Mixed`.
pub fn overlap_composition(entries: &[(PerformanceResult, u32)]) -> PerformanceResult {
    use std::collections::BTreeMap;

    let mut by_group: BTreeMap<u32, (f64, Source)> = BTreeMap::new();
    for &(result, group_id) in entries {
        let slot = by_group.entry(group_id).or_insert((0.0, result.source));
        if result.latency_ms > slot.0 {
            slot.0 = result.latency_ms;
        }
        slot.1 = slot.1.combine(result.source);
    }
    let mut total = 0.0_f64;
    let mut source = Source::Silicon;
    let mut first = true;
    for (_, (lat, src)) in by_group {
        total += lat;
        source = if first { src } else { source.combine(src) };
        first = false;
    }
    PerformanceResult::new(total, source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_within_group_then_sum_across_groups() {
        let entries = vec![
            (PerformanceResult::silicon(10.0), 0),
            (PerformanceResult::silicon(8.0), 0), // grouped with above -> max = 10.0
            (PerformanceResult::silicon(5.0), 1), // own group
        ];
        let result = overlap_composition(&entries);
        assert_eq!(result.latency_ms, 15.0); // 10 + 5
        assert_eq!(result.source, Source::Silicon);
    }

    #[test]
    fn mixed_sources_yield_mixed_tag() {
        let entries = vec![
            (PerformanceResult::silicon(5.0), 0),
            (PerformanceResult::new(3.0, Source::Empirical), 1),
        ];
        let result = overlap_composition(&entries);
        assert_eq!(result.source, Source::Mixed);
    }

    #[test]
    fn empty_list_is_zero() {
        let result = overlap_composition(&[]);
        assert_eq!(result.latency_ms, 0.0);
    }
}
