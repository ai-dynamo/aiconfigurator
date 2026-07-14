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
use std::sync::Arc;

use crate::common::enums::{DatabaseMode, TransferPolicy};
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::config::{PerfDbSources, PerfSource};
use crate::operators::util_empirical::{DeltaLookupCache, UtilGridCache};

/// Resolve the ordered source list for one op-file basename: the Python-supplied
/// shared-layer sources when present, else a single primary `data_root/<basename>`
/// with no `kernel_source` filter (identical to the pre-shared-layer default).
pub(crate) fn resolve_op_sources(
    perf_db_sources: &PerfDbSources,
    basename: &str,
    data_root: &Path,
) -> Vec<PerfSource> {
    match perf_db_sources.get(basename) {
        Some(sources) if !sources.is_empty() => sources.clone(),
        _ => vec![PerfSource(data_root.join(basename), None)],
    }
}

/// Whether a row's `kernel_source` passes a source's filter, mirroring Python
/// `_read_filtered_rows`: `None` admits every row; a `Some` allowlist keeps only
/// rows whose `kernel_source` value is in the set (a row missing the column is
/// dropped, since Python's `row.get("kernel_source") in ks_filter` is `False`
/// for `None`). Shared by every op table's multi-source loader.
pub(crate) fn kernel_source_ok(
    filter: Option<&[String]>,
    ks_col: Option<usize>,
    row: &parquet_loader::PerfRow,
) -> Result<bool, AicError> {
    match filter {
        None => Ok(true),
        Some(allow) => match row.str_optional(ks_col)? {
            Some(ks) => Ok(allow.iter().any(|a| a == ks)),
            None => Ok(false),
        },
    }
}

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

/// The loaded per-family perf tables for one `<system>/<backend>/<version>`
/// tuple — the mode-independent, immutable-after-load data half of
/// [`PerfDatabase`]. Shared (via `Arc`) between mode views.
pub struct PerfTables {
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

/// Modular performance database for a specific
/// `<system>/<backend>/<version>` tuple.
///
/// `load` does the cheap work: resolves the data directory from the system
/// YAML and constructs empty per-family tables. The first query on each
/// family triggers the CSV read.
///
/// Structurally this is a mode-configured VIEW over shared [`PerfTables`]
/// (mirroring Python's configured query views): the tables deref through, so
/// `db.gemm` / `db.system_spec` read as before, while `database_mode` /
/// `transfer_policy` are per-view. [`PerfDatabase::silicon_view`] derives the
/// SILICON view `FallbackOp` (and MSA's cross-op DSA probe) evaluate against
/// under HYBRID.
pub struct PerfDatabase {
    tables: Arc<PerfTables>,
    /// Query mode (Python's `database._default_database_mode`). SILICON
    /// queries collected tables only; HYBRID falls back to the util-space
    /// empirical layer on a typed silicon miss; EMPIRICAL always answers
    /// `SOL/util`. Defaults to SILICON; `Engine::from_spec_bytes` overwrites
    /// it from the spec.
    pub database_mode: DatabaseMode,
    /// Enabled empirical transfer kinds (Python's `database.transfer_policy`).
    pub transfer_policy: TransferPolicy,
    /// Memo of built util-calibration grids, keyed by op/slice identity
    /// (see `operators::util_empirical`). Tables are immutable after load and
    /// grid keys name their slice, so the memo is shared across mode views
    /// and never needs invalidation.
    pub util_grids: Arc<UtilGridCache>,
    /// Memo of zero-aware delta lookups (the `compute_scale` empirical
    /// mechanism); same keying/lifetime rationale as `util_grids`.
    pub delta_lookups: Arc<DeltaLookupCache>,
}

impl std::ops::Deref for PerfDatabase {
    type Target = PerfTables;

    fn deref(&self) -> &PerfTables {
        &self.tables
    }
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
        Self::load_with_sources(systems_root, system, backend, version, &PerfDbSources::default())
    }

    /// Like [`PerfDatabase::load`], but honours the shared-layer
    /// (sibling/cross-version) `perf_db_sources` resolved in Python
    /// (`sdk/engine.py::_compute_perf_db_sources`). For op files present in the
    /// map, the ordered source list (with per-source `kernel_source` filters) is
    /// used instead of the single primary file so Rust inherits the same rows
    /// Python does under SILICON/HYBRID. Op files absent from the map fall back
    /// to the primary `data_root` (identical to [`PerfDatabase::load`]).
    pub fn load_with_sources(
        systems_root: &Path,
        system: &str,
        backend: &str,
        version: &str,
        perf_db_sources: &PerfDbSources,
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
        let tables = PerfTables {
            system: system.to_string(),
            backend: backend.to_string(),
            version: version.to_string(),
            // Every op table resolves its own file basenames from
            // `perf_db_sources` via `with_sources` (shared-layer aware); an
            // absent basename falls back to the primary `data_root` file.
            // NCCL/OneCCL are framework-agnostic and never inherit siblings, so
            // their roots stay as the direct system-wide dirs.
            gemm: GemmTable::with_sources(data_root.clone(), spec.clone(), perf_db_sources),
            attention: AttentionTable::with_sources(data_root.clone(), spec.clone(), perf_db_sources),
            mla: MlaTable::with_sources(data_root.clone(), spec.clone(), perf_db_sources),
            moe: MoeTable::with_sources(data_root.clone(), perf_db_sources),
            communication: CommunicationTable::with_sources(
                data_root.clone(),
                nccl_root,
                oneccl_root,
                perf_db_sources,
            ),
            dsa: DsaTable::with_sources(data_root.clone(), perf_db_sources),
            dsv4: Dsv4Table::with_sources(data_root.clone(), perf_db_sources),
            mhc: MhcTable::with_sources(data_root.clone(), perf_db_sources),
            wideep: WideEpTable::with_sources(data_root.clone(), perf_db_sources),
            wideep_mla: WideEpMlaTable::with_sources(data_root.clone(), spec.clone(), perf_db_sources),
            wideep_moe: WideEpMoeTable::with_sources(data_root.clone(), perf_db_sources),
            state_space: StateSpaceTable::with_sources(
                data_root.clone(),
                backend,
                version,
                perf_db_sources,
            ),
            system_spec: spec,
            data_root,
        };
        Ok(Self {
            tables: Arc::new(tables),
            database_mode: DatabaseMode::default(),
            transfer_policy: TransferPolicy::ALL,
            util_grids: Arc::new(UtilGridCache::new()),
            delta_lookups: Arc::new(DeltaLookupCache::new()),
        })
    }

    /// Configure the query mode + transfer policy (both immutable per
    /// database instance afterwards, mirroring Python's configured query
    /// views). Called by `Engine::from_spec_bytes` with the spec's values.
    pub fn with_mode(mut self, database_mode: DatabaseMode, transfer_policy: TransferPolicy) -> Self {
        self.database_mode = database_mode;
        self.transfer_policy = transfer_policy;
        self
    }

    /// Test-only mutable access to the shared tables (panics if the view has
    /// been cloned — synthetic-table injection must happen before any
    /// `silicon_view`). Production code never mutates loaded tables.
    #[cfg(test)]
    pub(crate) fn tables_mut(&mut self) -> &mut PerfTables {
        Arc::get_mut(&mut self.tables).expect("tables Arc must be unique for test mutation")
    }

    /// The SILICON view over the same loaded tables (cheap: `Arc` clones).
    ///
    /// Mirrors Python `FallbackOp.query`'s
    /// `_get_configured_database_view(database, SILICON, transfer_policy)`:
    /// under HYBRID the primary op is evaluated silicon-only so the module
    /// table's absence falls to the granular fallback chain instead of being
    /// hybrid-estimated at module level. Also used by MSA's cross-op DSA
    /// utilisation probe.
    pub fn silicon_view(&self) -> PerfDatabase {
        PerfDatabase {
            tables: Arc::clone(&self.tables),
            database_mode: DatabaseMode::Silicon,
            transfer_policy: self.transfer_policy,
            util_grids: Arc::clone(&self.util_grids),
            delta_lookups: Arc::clone(&self.delta_lookups),
        }
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
