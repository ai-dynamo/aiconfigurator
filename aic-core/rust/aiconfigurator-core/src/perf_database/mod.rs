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
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use crate::common::enums::{DatabaseMode, TransferPolicy};
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::config::{PerfDbSources, PerfSource};
use crate::operators::util_empirical::{DeltaLookupCache, ProvenanceTier, UtilGridCache};

/// The five known legacy/framework-agnostic backend directory names. Mirrors the
/// SDK loader's `KNOWN_BACKEND_DIRS`
/// (`aic-core/src/aiconfigurator_core/sdk/perf_database.py`): any other
/// first-level directory under a system's data dir is a family dir containing
/// `<backend>/<version>` subtrees.
/// Keep textually identical to the CANONICAL `_KNOWN_BACKEND_DIRS` in
/// `aic-core/src/aiconfigurator_core/sdk/operations/base.py`, which lists
/// every copy that must stay in sync (Rust cannot import the Python set).
const KNOWN_BACKEND_DIRS: [&str; 5] = ["trtllm", "sglang", "vllm", "nccl", "oneccl"];

/// Resolve the ordered source list for one op-file basename: the Python-supplied
/// shared-layer sources when present, else a single primary `data_root/<basename>`
/// with no `kernel_source` filter (identical to the pre-shared-layer default). When
/// that legacy file is absent (`data_root` migrated to the family-first layout),
/// falls back to scanning sibling family dirs for
/// `<family>/<backend>/<version>/<basename>`.
///
/// Minimal fallback: Python supplies `perf_db_sources` in practice, so this only
/// covers the plain (no shared-layer) default-source case.
pub(crate) fn resolve_op_sources(
    perf_db_sources: &PerfDbSources,
    basename: &str,
    data_root: &Path,
) -> Vec<PerfSource> {
    match perf_db_sources.get(basename) {
        Some(sources) if !sources.is_empty() => sources.clone(),
        _ => {
            let legacy = data_root.join(basename);
            let path = if legacy.is_file() {
                legacy
            } else {
                find_in_family_dirs(data_root, basename).unwrap_or(legacy)
            };
            vec![PerfSource(path, None)]
        }
    }
}

/// Scan family-first sibling dirs for `<family>/<backend>/<version>/<basename>`,
/// where `<data_dir>` (the family dirs' parent) and `<backend>/<version>` are
/// derived from `data_root` (`<data_dir>/<backend>/<version>`).
fn find_in_family_dirs(data_root: &Path, basename: &str) -> Option<PathBuf> {
    let version = data_root.file_name()?.to_str()?;
    let backend = data_root.parent()?.file_name()?.to_str()?;
    let data_dir = data_root.parent()?.parent()?;
    for entry in std::fs::read_dir(data_dir).ok()?.flatten() {
        let name = entry.file_name();
        let name = match name.to_str() {
            Some(name) => name,
            None => continue,
        };
        if KNOWN_BACKEND_DIRS.contains(&name) {
            continue;
        }
        let candidate = entry.path().join(backend).join(version).join(basename);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// True when at least one family-first dir directly under `system_data_root`
/// (any first-level entry whose name is not in `KNOWN_BACKEND_DIRS`) contains
/// a `<backend>/<version>` subdirectory. Lets [`PerfDatabase::load_with_sources`]
/// accept a tuple whose data has migrated entirely off the legacy
/// `<backend>/<version>` layout — no legacy dir needs to exist as long as some
/// family dir holds the tuple. The actual per-file resolution then happens
/// inside each table's `resolve_op_sources` call via `find_in_family_dirs`.
fn has_family_backend_version(system_data_root: &Path, backend: &str, version: &str) -> bool {
    let entries = match std::fs::read_dir(system_data_root) {
        Ok(entries) => entries,
        Err(_) => return false,
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = match name.to_str() {
            Some(name) => name,
            None => continue,
        };
        if KNOWN_BACKEND_DIRS.contains(&name) || !entry.path().is_dir() {
            continue;
        }
        if entry.path().join(backend).join(version).is_dir() {
            return true;
        }
    }
    false
}

/// NCCL/OneCCL system-wide perf data is migrating from the legacy
/// `<system_data_root>/{nccl,oneccl}/<version>` layout to family-first
/// `<system_data_root>/comm/{nccl,oneccl}/<version>` (the collector always
/// writes NCCL/OneCCL under the single `comm` family, unlike the per-backend
/// families gemm/attention/etc. use — there is only ever one comm family).
/// Python's loader (`operations/base.py::resolve_op_data_path`) discovers the
/// family dir structurally by scanning every first-level entry under
/// `system_data_root`; Rust hardcodes the one `comm` candidate instead, since
/// NCCL/OneCCL only ever land under that name in practice — a plain path
/// check is simpler than a directory scan and behaves identically here.
fn comm_root(system_data_root: &Path, backend_dir: &str, version: &str) -> PathBuf {
    let family_root = system_data_root.join("comm").join(backend_dir).join(version);
    if family_root.is_dir() {
        family_root
    } else {
        system_data_root.join(backend_dir).join(version)
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
pub mod dsv4_megamoe;
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
pub use dsv4_megamoe::Dsv4MegaMoeTable;
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
    pub dsv4_megamoe: Dsv4MegaMoeTable,
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
    /// Max-rank empirical provenance tier fired since the last
    /// [`PerfDatabase::reset_provenance`] (Rust mirror of Python's
    /// `capture_provenance` contextvar in `sdk/operations/util_empirical.py`).
    /// Shared across mode views (`silicon_view` clones) like `util_grids`, so
    /// notes made through a derived view land on the run's accumulator. The
    /// engine FFI resets it per `run_static`/step call and reads the worst
    /// tier back for the Python bridge / support-matrix HYBRID labelling.
    provenance: Arc<AtomicU8>,
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
    /// `systems_root` points at `src/aiconfigurator_core/systems`. `system` is a
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
        // Accept either layout: the legacy `<backend>/<version>` dir itself,
        // or at least one family-first `<family>/<backend>/<version>` dir
        // (family = any first-level dir under `system_data_root` other than
        // the known legacy backend names). `data_root` stays the legacy path
        // either way — each table's `resolve_op_sources` call resolves the
        // actual per-file location (legacy or family) independently.
        if !data_root.is_dir() && !has_family_backend_version(&system_data_root, backend, version) {
            return Err(AicError::PerfDatabase(format!(
                "perf data directory not found in either the legacy layout ({}) or a family-first layout \
                 (<family>/{backend}/{version} under {}) (system={system}, backend={backend}, version={version})",
                data_root.display(),
                system_data_root.display()
            )));
        }
        // NCCL/OneCCL parquet files live under `<system_data_root>/comm/
        // {nccl, oneccl}/<version>/` (family-first, preferred) or the legacy
        // `<system_data_root>/{nccl, oneccl}/<version>/`, NOT under the
        // backend/version data dir — see `comm_root`. The version comes from
        // `SystemSpec.misc.{nccl,oneccl}_version` and is optional — XPU
        // systems decl ``oneccl_version`` only, GPU systems typically decl
        // `nccl_version` only. Mirrors Python
        // `sdk/operations/communication.py:294, 301-303`.
        let nccl_root = spec
            .misc
            .nccl_version
            .as_ref()
            .map(|v| comm_root(&system_data_root, "nccl", v));
        let oneccl_root = spec
            .misc
            .oneccl_version
            .as_ref()
            .map(|v| comm_root(&system_data_root, "oneccl", v));
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
            // Single-primary by design: the Python MegaMoE loader reads one
            // unified path and never the shared-layer source list (see
            // `dsv4_megamoe.rs`).
            dsv4_megamoe: Dsv4MegaMoeTable::new(data_root.clone()),
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
            provenance: Arc::new(AtomicU8::new(ProvenanceTier::Silicon as u8)),
        })
    }

    /// Record that an empirical path of tier `tier` produced a value.
    /// Max-rank accumulation, so the cell always holds the run's
    /// least-confident (worst) tier — Python's `worst_provenance` semantics.
    ///
    /// Call sites mirror Python `note_provenance` exactly: after each
    /// successful `util_empirical::estimate` with the tier the site knows
    /// (own-data "empirical", attention head_size ref grid "xshape", the MoE
    /// ladder's `reference_provenance`, communication rank-overflow "xshape",
    /// MSA cross-op "xop", ...).
    pub fn note_provenance(&self, tier: ProvenanceTier) {
        self.provenance.fetch_max(tier as u8, Ordering::Relaxed);
    }

    /// Clear the accumulator back to `Silicon` (start of a run).
    pub fn reset_provenance(&self) {
        self.provenance
            .store(ProvenanceTier::Silicon as u8, Ordering::Relaxed);
    }

    /// The least-confident tier fired since the last reset; `Silicon` when no
    /// empirical path fired (Python `worst_provenance` of an empty capture).
    pub fn worst_provenance(&self) -> ProvenanceTier {
        ProvenanceTier::from_rank(self.provenance.load(Ordering::Relaxed))
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
            provenance: Arc::clone(&self.provenance),
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
            .join("src/aiconfigurator_core/systems")
    }

    #[test]
    fn load_b200_sxm_vllm_database() {
        let db = PerfDatabase::load(&systems_root(), "b200_sxm", "vllm", "0.19.0")
            .expect("b200_sxm/vllm/0.19.0 must load");
        assert_eq!(db.system, "b200_sxm");
        assert_eq!(db.backend, "vllm");
        assert_eq!(db.version, "0.19.0");
        let gemm_sources = resolve_op_sources(
            &PerfDbSources::default(),
            "gemm_perf.parquet",
            &db.data_root,
        );
        assert!(
            gemm_sources[0].0.is_file(),
            "resolved GEMM source must exist"
        );
    }

    #[test]
    fn provenance_cell_accumulates_worst_tier_and_is_shared_with_views() {
        let db = PerfDatabase::load(&systems_root(), "b200_sxm", "vllm", "0.19.0")
            .expect("b200_sxm/vllm/0.19.0 must load");
        assert_eq!(db.worst_provenance(), ProvenanceTier::Silicon);

        // Max-rank accumulation: a lower tier never overwrites a higher one.
        db.note_provenance(ProvenanceTier::XShape);
        db.note_provenance(ProvenanceTier::Empirical);
        assert_eq!(db.worst_provenance(), ProvenanceTier::XShape);

        // Notes through a derived silicon view land on the same accumulator
        // (the MSA xop probe evaluates against `silicon_view`).
        let view = db.silicon_view();
        view.note_provenance(ProvenanceTier::XOp);
        assert_eq!(db.worst_provenance(), ProvenanceTier::XOp);

        db.reset_provenance();
        assert_eq!(db.worst_provenance(), ProvenanceTier::Silicon);
        assert_eq!(view.worst_provenance(), ProvenanceTier::Silicon);
    }

    #[test]
    fn load_unknown_version_errors() {
        match PerfDatabase::load(&systems_root(), "b200_sxm", "vllm", "99.99.99") {
            Err(AicError::PerfDatabase(_)) => {}
            Ok(_) => panic!("expected load to fail for missing version"),
            Err(other) => panic!("expected PerfDatabase error, got {other:?}"),
        }
    }

    #[test]
    fn resolve_op_sources_falls_back_to_family_dir_when_legacy_file_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let backend = "sglang";
        let version = "0.5.14";

        // Legacy <data_dir>/<backend>/<version> dir exists but doesn't hold the file.
        let legacy_data_root = tmp.path().join(backend).join(version);
        std::fs::create_dir_all(&legacy_data_root).unwrap();

        // Family dir <data_dir>/gemm/<backend>/<version>/gemm_perf.parquet does.
        let family_data_root = tmp.path().join("gemm").join(backend).join(version);
        std::fs::create_dir_all(&family_data_root).unwrap();
        std::fs::write(family_data_root.join("gemm_perf.parquet"), b"stub").unwrap();

        let sources = resolve_op_sources(&PerfDbSources::default(), "gemm_perf.parquet", &legacy_data_root);
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].0, family_data_root.join("gemm_perf.parquet"));
        assert!(sources[0].1.is_none());
    }

    #[test]
    fn resolve_op_sources_skips_known_backend_dirs_when_scanning_families() {
        let tmp = tempfile::tempdir().unwrap();
        let backend = "nccl";
        let version = "2.19";
        let legacy_data_root = tmp.path().join(backend).join(version);
        std::fs::create_dir_all(&legacy_data_root).unwrap();

        // A sibling top-level dir named like a known legacy backend must not be
        // treated as a family dir, even though it holds a matching path.
        let decoy = tmp.path().join("vllm").join(backend).join(version);
        std::fs::create_dir_all(&decoy).unwrap();
        std::fs::write(decoy.join("nccl_perf.parquet"), b"stub").unwrap();

        let sources = resolve_op_sources(&PerfDbSources::default(), "nccl_perf.parquet", &legacy_data_root);
        // Falls back to the legacy (nonexistent) path since "vllm" is a known
        // backend dir, not a family dir, and must be skipped during the scan.
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].0, legacy_data_root.join("nccl_perf.parquet"));
    }

    /// Minimal `SystemSpec` YAML with `data_dir: data` (relative to
    /// `systems_root`) — just enough for `SystemSpec::load` to succeed so
    /// `PerfDatabase::load`'s existence-gate behavior can be exercised on a
    /// synthetic tree without a real systems fixture.
    fn write_synthetic_system_yaml(systems_root: &Path, system: &str) {
        let yaml = "\
data_dir: data
gpu:
  mem_bw: 1000000000000
node:
  num_gpus_per_node: 8
  inter_node_bw: 100000000000
  intra_node_bw: 900000000000
";
        std::fs::write(systems_root.join(format!("{system}.yaml")), yaml).unwrap();
    }

    #[test]
    fn load_succeeds_on_family_only_layout() {
        let tmp = tempfile::tempdir().unwrap();
        let systems_root = tmp.path();
        let (system, backend, version) = ("synth_family", "vllm", "1.2.3");
        write_synthetic_system_yaml(systems_root, system);

        // Family-first layout only: <data>/gemm/<backend>/<version>/gemm_perf.parquet.
        // No legacy <data>/<backend>/<version> dir exists at all.
        let family_dir = systems_root.join("data").join("gemm").join(backend).join(version);
        std::fs::create_dir_all(&family_dir).unwrap();
        std::fs::write(family_dir.join("gemm_perf.parquet"), b"stub").unwrap();

        let legacy_data_root = systems_root.join("data").join(backend).join(version);
        assert!(!legacy_data_root.is_dir(), "fixture must not have a legacy dir");

        let db = PerfDatabase::load(systems_root, system, backend, version)
            .expect("family-only layout must load");
        assert_eq!(db.system, system);
        // `data_root` stays the legacy path by contract; per-file resolution
        // (tested separately above) is what actually finds the family file.
        assert_eq!(db.data_root, legacy_data_root);
    }

    #[test]
    fn load_succeeds_on_legacy_only_layout() {
        let tmp = tempfile::tempdir().unwrap();
        let systems_root = tmp.path();
        let (system, backend, version) = ("synth_legacy", "vllm", "1.2.3");
        write_synthetic_system_yaml(systems_root, system);

        let legacy_dir = systems_root.join("data").join(backend).join(version);
        std::fs::create_dir_all(&legacy_dir).unwrap();
        std::fs::write(legacy_dir.join("gemm_perf.parquet"), b"stub").unwrap();

        let db = PerfDatabase::load(systems_root, system, backend, version)
            .expect("legacy-only layout must load");
        assert_eq!(db.data_root, legacy_dir);
    }

    #[test]
    fn load_errors_mentioning_both_layouts_on_total_miss() {
        let tmp = tempfile::tempdir().unwrap();
        let systems_root = tmp.path();
        let (system, backend, version) = ("synth_missing", "vllm", "9.9.9");
        write_synthetic_system_yaml(systems_root, system);
        std::fs::create_dir_all(systems_root.join("data")).unwrap();

        match PerfDatabase::load(systems_root, system, backend, version) {
            Err(AicError::PerfDatabase(msg)) => {
                assert!(msg.contains("legacy"), "error should mention legacy layout: {msg}");
                assert!(msg.contains("family"), "error should mention family layout: {msg}");
            }
            Ok(_) => panic!("expected load to fail for a totally missing tuple"),
            Err(other) => panic!("expected PerfDatabase error, got {other:?}"),
        }
    }

    #[test]
    fn comm_root_prefers_family_comm_dir_over_legacy_nccl() {
        let tmp = tempfile::tempdir().unwrap();
        let system_data_root = tmp.path();
        let version = "2.27.3";

        // Only the legacy dir exists: fall back to it.
        let legacy = system_data_root.join("nccl").join(version);
        std::fs::create_dir_all(&legacy).unwrap();
        assert_eq!(comm_root(system_data_root, "nccl", version), legacy);

        // Once the family-first `comm/nccl/<version>` dir also exists, it
        // takes priority over the legacy dir.
        let family = system_data_root.join("comm").join("nccl").join(version);
        std::fs::create_dir_all(&family).unwrap();
        assert_eq!(comm_root(system_data_root, "nccl", version), family);
    }
}
