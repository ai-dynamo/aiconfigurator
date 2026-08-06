// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared categorical index for MoE-family token curves.

use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct MoeShapeKey {
    pub(super) topk: u32,
    pub(super) num_experts: u32,
    pub(super) hidden_size: u32,
    pub(super) inter_size: u32,
    pub(super) moe_tp_size: u32,
    pub(super) moe_ep_size: u32,
}

pub(super) struct MoeIndex<S, V> {
    by_quant: BTreeMap<String, MoeQuantIndex<S, V>>,
}

pub(super) struct MoeQuantIndex<S, V> {
    first_distribution: String,
    by_distribution: BTreeMap<String, BTreeMap<S, V>>,
}

impl<S, V> Default for MoeIndex<S, V> {
    fn default() -> Self {
        Self {
            by_quant: BTreeMap::new(),
        }
    }
}

impl<S: Ord, V> MoeIndex<S, V> {
    pub(super) fn is_empty(&self) -> bool {
        self.by_quant.is_empty()
    }

    pub(super) fn quant(&self, quant: &str) -> Option<&MoeQuantIndex<S, V>> {
        self.by_quant.get(quant)
    }

    pub(super) fn entry(&mut self, quant: String, distribution: String, shape: S) -> &mut V
    where
        V: Default,
    {
        let quant_index = self.by_quant.entry(quant).or_insert_with(|| MoeQuantIndex {
            first_distribution: distribution.clone(),
            by_distribution: BTreeMap::new(),
        });
        quant_index
            .by_distribution
            .entry(distribution)
            .or_default()
            .entry(shape)
            .or_default()
    }

    pub(super) fn map_values<U>(self, mut map: impl FnMut(V) -> U) -> MoeIndex<S, U> {
        let mut by_quant = BTreeMap::new();
        for (quant, index) in self.by_quant {
            let mut by_distribution = BTreeMap::new();
            for (distribution, by_shape) in index.by_distribution {
                by_distribution.insert(
                    distribution,
                    by_shape
                        .into_iter()
                        .map(|(shape, value)| (shape, map(value)))
                        .collect(),
                );
            }
            by_quant.insert(
                quant,
                MoeQuantIndex {
                    first_distribution: index.first_distribution,
                    by_distribution,
                },
            );
        }
        MoeIndex { by_quant }
    }

    pub(super) fn resolve_uniform(
        &self,
        quant: &str,
        requested_distribution: &str,
        shape: &S,
    ) -> (&str, Option<&V>) {
        let Some(index) = self.quant(quant) else {
            return ("uniform", None);
        };
        let (distribution, by_shape) = index.resolve_named_or(requested_distribution, "uniform");
        (
            distribution,
            by_shape.and_then(|by_shape| by_shape.get(shape)),
        )
    }

    pub(super) fn resolve_uniform_shapes(
        &self,
        quant: &str,
        requested_distribution: &str,
    ) -> (&str, Option<&BTreeMap<S, V>>) {
        let Some(index) = self.quant(quant) else {
            return ("uniform", None);
        };
        index.resolve_named_or(requested_distribution, "uniform")
    }
}

impl<S: Ord, V> MoeQuantIndex<S, V> {
    fn resolve_named_or(
        &self,
        requested_distribution: &str,
        fallback: &'static str,
    ) -> (&str, Option<&BTreeMap<S, V>>) {
        if let Some((distribution, by_shape)) =
            self.by_distribution.get_key_value(requested_distribution)
        {
            (distribution.as_str(), Some(by_shape))
        } else {
            (fallback, self.by_distribution.get(fallback))
        }
    }

    pub(super) fn resolve_first(
        &self,
        requested_distribution: &str,
        shape: &S,
    ) -> (&str, Option<&V>) {
        let distribution = self
            .by_distribution
            .get_key_value(requested_distribution)
            .map(|(distribution, _)| distribution.as_str())
            .unwrap_or(self.first_distribution.as_str());
        let value = self
            .by_distribution
            .get(distribution)
            .and_then(|by_shape| by_shape.get(shape));
        (distribution, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_named_uniform_and_first_distributions() {
        let shape = MoeShapeKey {
            topk: 2,
            num_experts: 8,
            hidden_size: 4096,
            inter_size: 2048,
            moe_tp_size: 1,
            moe_ep_size: 4,
        };
        let mut index = MoeIndex::<_, u32>::default();
        *index.entry("fp8".into(), "power_law".into(), shape) = 1;
        *index.entry("fp8".into(), "uniform".into(), shape) = 2;

        assert_eq!(
            index.resolve_uniform("fp8", "power_law", &shape),
            ("power_law", Some(&1))
        );
        assert_eq!(
            index.resolve_uniform("fp8", "missing", &shape),
            ("uniform", Some(&2))
        );
        assert_eq!(
            index.quant("fp8").unwrap().resolve_first("missing", &shape),
            ("power_law", Some(&1))
        );
    }
}
