// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed system hardware spec parsed from `src/aiconfigurator/systems/*.yaml`.
//!
//! Mirrors `aiconfigurator.sdk.system_spec.SystemSpec`. The Python type
//! subclasses `dict`; the Rust port uses a typed struct because every
//! downstream consumer here needs typed field access. The on-disk YAML schema
//! is the contract; struct layout follows the YAML shape one-to-one.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::common::error::AicError;

/// Hardware system spec deserialized from `systems/<system>.yaml`.
#[derive(Clone, Debug, Deserialize)]
pub struct SystemSpec {
    /// Path to per-system perf data, relative to the systems directory.
    pub data_dir: PathBuf,
    pub gpu: GpuSpec,
    pub node: NodeSpec,
    #[serde(default)]
    pub misc: MiscSpec,
}

/// GPU hardware spec.
#[derive(Clone, Debug, Deserialize)]
pub struct GpuSpec {
    /// HBM memory bandwidth, bytes/s.
    pub mem_bw: f64,
    /// Empirical scaling factor applied to `mem_bw`.
    #[serde(default = "default_mem_bw_scaling")]
    pub mem_bw_empirical_scaling_factor: f64,
    /// Constant latency added to memory-bound ops (seconds).
    #[serde(default)]
    pub mem_empirical_constant_latency: f64,
    /// HBM capacity in bytes.
    #[serde(default)]
    pub mem_capacity: Option<u64>,
    /// Peak TC-FLOPS at bf16.
    #[serde(default)]
    pub bfloat16_tc_flops: Option<f64>,
    /// Peak TC-FLOPS at int8.
    #[serde(default)]
    pub int8_tc_flops: Option<f64>,
    /// Peak TC-FLOPS at fp8.
    #[serde(default)]
    pub fp8_tc_flops: Option<f64>,
    /// Peak TC-FLOPS at fp4/nvfp4.
    #[serde(default)]
    pub fp4_tc_flops: Option<f64>,
    /// Per-GPU TDP, watts.
    #[serde(default)]
    pub power: Option<f64>,
    /// CUDA SM architecture version (e.g. 100 for Blackwell SM_100).
    #[serde(default)]
    pub sm_version: Option<u32>,
}

/// Node-level topology spec.
#[derive(Clone, Debug, Deserialize)]
pub struct NodeSpec {
    pub num_gpus_per_node: u32,
    /// Inter-node bandwidth, bytes/s per GPU, single direction.
    pub inter_node_bw: f64,
    /// Intra-node (NVLink) bandwidth, bytes/s per GPU, single direction.
    pub intra_node_bw: f64,
    /// PCIe bandwidth, bytes/s, single direction.
    #[serde(default)]
    pub pcie_bw: Option<f64>,
    /// Point-to-point latency (seconds).
    #[serde(default)]
    pub p2p_latency: f64,
    /// Rack-level fan-out (GB200/GB300). Optional.
    #[serde(default)]
    pub num_gpus_per_rack: Option<u32>,
    /// Inter-rack bandwidth, bytes/s. Optional.
    #[serde(default)]
    pub inter_rack_bw: Option<f64>,
}

/// Miscellaneous, mostly empirical, configuration.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct MiscSpec {
    /// NCCL memory overhead by per-rank count.
    #[serde(default)]
    pub nccl_mem: BTreeMap<u32, u64>,
    /// Other memory overhead in bytes.
    #[serde(default)]
    pub other_mem: Option<u64>,
    /// NCCL version string used during data collection. The corresponding
    /// `nccl_perf.parquet` lives under `<systems_root>/<data_dir>/nccl/
    /// <nccl_version>/` and is system-wide (NOT per backend/version),
    /// mirroring Python `sdk/operations/communication.py:294`.
    #[serde(default)]
    pub nccl_version: Option<String>,
    /// OneCCL version string used during data collection. Set on XPU
    /// systems (e.g. `b60`) where Intel oneCCL is the comm backend; the
    /// corresponding `oneccl_perf.parquet` lives under
    /// `<systems_root>/<data_dir>/oneccl/<oneccl_version>/`. Mirrors
    /// Python `sdk/operations/communication.py:301-303`.
    #[serde(default)]
    pub oneccl_version: Option<String>,
}

fn default_mem_bw_scaling() -> f64 {
    1.0
}

impl SystemSpec {
    /// Load a system YAML from disk.
    pub fn load(path: &Path) -> Result<Self, AicError> {
        let text = fs::read_to_string(path).map_err(|source| AicError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        serde_yaml::from_str(&text).map_err(|source| AicError::Yaml {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Point-to-point bandwidth (bytes/s) for a given collective fan-out.
    ///
    /// Mirrors Python's `SystemSpec.get_p2p_bandwidth` three-tier selection:
    /// `intra_node_bw` within a node, `inter_node_bw` within a rack,
    /// `inter_rack_bw` (or `inter_node_bw` fallback) across racks.
    pub fn get_p2p_bandwidth(&self, num_gpus: u32) -> f64 {
        let node = &self.node;
        if num_gpus <= node.num_gpus_per_node {
            return node.intra_node_bw;
        }
        let per_rack = node.num_gpus_per_rack.unwrap_or(u32::MAX);
        if num_gpus <= per_rack {
            return node.inter_node_bw;
        }
        node.inter_rack_bw.unwrap_or(node.inter_node_bw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REPO_ROOT_HINT: &str = env!("CARGO_MANIFEST_DIR");

    fn systems_root() -> PathBuf {
        // CARGO_MANIFEST_DIR points at rust/aiconfigurator-core; the systems
        // dir is two levels up.
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator/systems")
    }

    #[test]
    fn parse_b200_sxm() {
        let spec = SystemSpec::load(&systems_root().join("b200_sxm.yaml"))
            .expect("b200_sxm.yaml parse must succeed");
        assert_eq!(spec.data_dir, PathBuf::from("data/b200_sxm"));
        // 7.7 TB/s since PR #1246 (was 8.0; this pin went stale unnoticed
        // because cargo test was not in the gating CI).
        assert_eq!(spec.gpu.mem_bw, 7_700_000_000_000.0);
        assert_eq!(spec.gpu.mem_bw_empirical_scaling_factor, 0.8);
        assert_eq!(spec.gpu.bfloat16_tc_flops, Some(2_250_000_000_000_000.0));
        assert_eq!(spec.gpu.sm_version, Some(100));
        assert_eq!(spec.node.num_gpus_per_node, 8);
        assert!(spec.node.num_gpus_per_rack.is_none()); // b200_sxm has no rack tier
        assert_eq!(spec.misc.nccl_mem.get(&8), Some(&411_041_792));
        assert_eq!(spec.misc.nccl_version.as_deref(), Some("2.27.3"));
    }

    #[test]
    fn parse_gb200_rack_tier() {
        let spec = SystemSpec::load(&systems_root().join("gb200.yaml"))
            .expect("gb200.yaml parse must succeed");
        assert!(spec.node.num_gpus_per_rack.is_some());
        assert!(spec.node.inter_rack_bw.is_some());
    }

    #[test]
    fn p2p_bandwidth_three_tier_selection() {
        // Build a minimal in-memory spec with all three tiers set.
        let spec = SystemSpec {
            data_dir: PathBuf::from("data/synthetic"),
            gpu: GpuSpec {
                mem_bw: 1.0,
                mem_bw_empirical_scaling_factor: 1.0,
                mem_empirical_constant_latency: 0.0,
                mem_capacity: None,
                bfloat16_tc_flops: None,
                int8_tc_flops: None,
                fp8_tc_flops: None,
                fp4_tc_flops: None,
                power: None,
                sm_version: None,
            },
            node: NodeSpec {
                num_gpus_per_node: 8,
                intra_node_bw: 900.0,
                inter_node_bw: 100.0,
                pcie_bw: None,
                p2p_latency: 0.0,
                num_gpus_per_rack: Some(72),
                inter_rack_bw: Some(10.0),
            },
            misc: MiscSpec::default(),
        };

        // Within node -> intra
        assert_eq!(spec.get_p2p_bandwidth(1), 900.0);
        assert_eq!(spec.get_p2p_bandwidth(8), 900.0);
        // Within rack -> inter-node
        assert_eq!(spec.get_p2p_bandwidth(9), 100.0);
        assert_eq!(spec.get_p2p_bandwidth(72), 100.0);
        // Across racks -> inter-rack
        assert_eq!(spec.get_p2p_bandwidth(73), 10.0);
    }

    #[test]
    fn p2p_bandwidth_falls_back_to_inter_node_when_inter_rack_unset() {
        let spec = SystemSpec {
            data_dir: PathBuf::from("data/synthetic"),
            gpu: GpuSpec {
                mem_bw: 1.0,
                mem_bw_empirical_scaling_factor: 1.0,
                mem_empirical_constant_latency: 0.0,
                mem_capacity: None,
                bfloat16_tc_flops: None,
                int8_tc_flops: None,
                fp8_tc_flops: None,
                fp4_tc_flops: None,
                power: None,
                sm_version: None,
            },
            node: NodeSpec {
                num_gpus_per_node: 8,
                intra_node_bw: 900.0,
                inter_node_bw: 100.0,
                pcie_bw: None,
                p2p_latency: 0.0,
                num_gpus_per_rack: Some(72),
                inter_rack_bw: None, // unset
            },
            misc: MiscSpec::default(),
        };
        assert_eq!(spec.get_p2p_bandwidth(100), 100.0);
    }
}
