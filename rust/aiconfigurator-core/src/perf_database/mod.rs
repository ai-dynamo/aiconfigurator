// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Modular perf database with one table owner per op family.
//!
//! Each per-family submodule (`gemm`, etc.) owns its CSV loaders, query API,
//! and runtime cache. Loading is lazy: `PerfDatabase::load` only resolves
//! paths and parses the system YAML; each table's CSV is read on first
//! query via `OnceLock`. Submodules cover the full op-family set: gemm,
//! attention, mla, dsa, dsv4, mhc, moe, communication, state-space, and
//! the WideEP/DeepEP all-to-all variants.

use std::path::{Path, PathBuf};

use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;

pub mod attention;
pub mod communication;
pub mod dsa;
pub mod dsv4;
pub mod gemm;
mod interpolation;
pub mod mhc;
pub mod mla;
pub mod moe;
pub mod parquet_loader;
pub mod perf_interp;
pub mod state_space;
pub mod wideep;
pub mod wideep_mla;
pub mod wideep_moe;

pub use attention::AttentionTable;
pub use communication::CommunicationTable;
pub use dsa::DsaTable;
pub use dsv4::{AttnKind, Dsv4Table};
pub use gemm::GemmTable;
pub use mhc::MhcTable;
pub use mla::MlaTable;
pub use moe::MoeTable;
pub use state_space::StateSpaceTable;
pub use wideep::WideEpTable;
pub use wideep_mla::WideEpMlaTable;
pub use wideep_moe::WideEpMoeTable;

/// Modular performance database for a specific
/// `<system>/<backend>/<version>` tuple.
///
/// `load` does the cheap work: resolves the data directory from the system
/// YAML and constructs empty per-family tables. The first query on each
/// family triggers the CSV read.
pub struct PerfDatabase {
    pub system: String,
    pub backend: String,
    pub version: String,
    pub system_spec: SystemSpec,
    pub data_root: PathBuf,
    pub gemm: GemmTable,
    pub attention: AttentionTable,
    pub mla: MlaTable,
    pub moe: MoeTable,
    pub communication: CommunicationTable,
    pub dsa: DsaTable,
    pub dsv4: Dsv4Table,
    pub mhc: MhcTable,
    pub wideep: WideEpTable,
    pub wideep_mla: WideEpMlaTable,
    pub wideep_moe: WideEpMoeTable,
    pub state_space: StateSpaceTable,
}

impl PerfDatabase {
    /// Resolve and parse the system YAML, locate the per-version data
    /// directory, and construct lazy table owners.
    ///
    /// `systems_root` points at `src/aiconfigurator/systems`. `system` is a
    /// basename like `b200_sxm`. `backend` is `vllm` / `sglang` / `trtllm`.
    /// `version` is the backend version directory name (e.g. `0.19.0`).
    pub fn load(
        systems_root: &Path,
        system: &str,
        backend: &str,
        version: &str,
    ) -> Result<Self, AicError> {
        let system_yaml = systems_root.join(format!("{system}.yaml"));
        let spec = SystemSpec::load(&system_yaml)?;
        let system_data_root = systems_root.join(&spec.data_dir);
        let data_root = system_data_root.join(backend).join(version);
        if !data_root.is_dir() {
            return Err(AicError::PerfDatabase(format!(
                "perf data directory not found: {} (system={system}, backend={backend}, version={version})",
                data_root.display()
            )));
        }
        // NCCL/OneCCL parquet files live under `<system_data_root>/{nccl,
        // oneccl}/<version>/`, NOT under the backend/version data dir. The
        // version comes from `SystemSpec.misc.{nccl,oneccl}_version` and is
        // optional — XPU systems decl ``oneccl_version`` only, GPU systems
        // typically decl `nccl_version` only. Mirrors Python
        // `sdk/operations/communication.py:294, 301-303`.
        let nccl_root = spec
            .misc
            .nccl_version
            .as_ref()
            .map(|v| system_data_root.join("nccl").join(v));
        let oneccl_root = spec
            .misc
            .oneccl_version
            .as_ref()
            .map(|v| system_data_root.join("oneccl").join(v));
        Ok(Self {
            system: system.to_string(),
            backend: backend.to_string(),
            version: version.to_string(),
            gemm: GemmTable::new(data_root.clone(), spec.clone()),
            attention: AttentionTable::new(data_root.clone(), spec.clone()),
            system_spec: spec,
            mla: MlaTable::new(data_root.clone()),
            moe: MoeTable::new(data_root.clone()),
            communication: CommunicationTable::new(data_root.clone(), nccl_root, oneccl_root),
            dsa: DsaTable::new(data_root.clone()),
            dsv4: Dsv4Table::new(data_root.clone()),
            mhc: MhcTable::new(data_root.clone()),
            wideep: WideEpTable::new(data_root.clone()),
            wideep_mla: WideEpMlaTable::new(data_root.clone()),
            wideep_moe: WideEpMoeTable::new(data_root.clone()),
            state_space: StateSpaceTable::new(data_root.clone()),
            data_root,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REPO_ROOT_HINT: &str = env!("CARGO_MANIFEST_DIR");

    fn systems_root() -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator/systems")
    }

    #[test]
    fn load_b200_sxm_vllm_database() {
        let db = PerfDatabase::load(&systems_root(), "b200_sxm", "vllm", "0.19.0")
            .expect("b200_sxm/vllm/0.19.0 must load");
        assert_eq!(db.system, "b200_sxm");
        assert_eq!(db.backend, "vllm");
        assert_eq!(db.version, "0.19.0");
        assert!(db.data_root.is_dir(), "data_root must exist");
        assert!(db.data_root.join("gemm_perf.parquet").is_file());
    }

    #[test]
    fn load_unknown_version_errors() {
        match PerfDatabase::load(&systems_root(), "b200_sxm", "vllm", "99.99.99") {
            Err(AicError::PerfDatabase(_)) => {}
            Ok(_) => panic!("expected load to fail for missing version"),
            Err(other) => panic!("expected PerfDatabase error, got {other:?}"),
        }
    }
}
