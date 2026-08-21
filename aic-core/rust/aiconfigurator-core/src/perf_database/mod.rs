// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Modular perf database with one table owner per op family.
//!
//! Each per-family submodule (`gemm`, etc.) owns its parquet loaders, query API,
//! and runtime cache. Loading is lazy: `PerfDatabase::load` only resolves
//! paths and parses the system YAML; each table's parquet data is read on first
//! query via `OnceLock`. Submodules cover the full op-family set: gemm,
//! attention, mla, dsa, dsv4, mhc, moe, communication, state-space, and
//! the TRT-LLM all-to-all table.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

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

/// Test-only compatibility shim over [`SourceResolver::fixed`]'s map
/// semantics (present-but-empty = veto, absent = default-primary with the
/// family walk). Production code threads a [`SourceResolver`] instead.
#[cfg(test)]
pub(crate) fn resolve_op_sources(
    perf_db_sources: &PerfDbSources,
    basename: &str,
    data_root: &Path,
) -> Vec<PerfSource> {
    SourceResolver::fixed(perf_db_sources.clone())
        .sources_for(basename, data_root)
        .expect("fixed-map resolution is infallible")
}

/// Stable identity string for one load input set, used as the shared-tables
/// memo key. The resolver renders its own identity: live resolution is a pure
/// function of the load identity + policy; a fixed map renders its full
/// contents. `\x1f` (unit separator) keeps boundaries unambiguous; non-UTF-8
/// paths degrade lossily instead of failing.
/// Process-global (or test-private) memo behind [`PerfDatabase::load_resolved_shared`].
type SharedTablesMemo = Mutex<HashMap<String, Weak<PerfTables>>>;

fn shared_tables_key(
    systems_root: &Path,
    system: &str,
    backend: &str,
    version: &str,
    resolver: &SourceResolver,
) -> String {
    format!(
        "{}\x1f{system}\x1f{backend}\x1f{version}\x1f{}",
        systems_root.display(),
        resolver.identity_key()
    )
}

/// Whether a version dir is excluded from data loading wholesale: a legacy
/// `INCOMPLETE.txt` marker with NO structured `collection_meta.yaml` sidecar.
/// Mirrors the CANONICAL `operations/base.py::_version_dir_is_unusable`
/// (a structured sidecar supersedes a stale legacy marker; `status: partial`
/// validation is the Python admission layer's job, not this existence check).
pub(crate) fn version_dir_is_unusable(version_dir: &Path) -> bool {
    if version_dir.join("collection_meta.yaml").is_file() {
        return false;
    }
    version_dir.join("INCOMPLETE.txt").is_file()
}

/// Scan family-first sibling dirs for `<family>/<backend>/<version>/<basename>`,
/// where `<data_dir>` (the family dirs' parent) and `<backend>/<version>` are
/// derived from `data_root` (`<data_dir>/<backend>/<version>`). Mirrors
/// `operations/base.py::resolve_op_data_path`'s family walk: dot-dirs are
/// skipped and a version dir carrying the legacy INCOMPLETE veto is never
/// admitted (the legacy `<backend>/<version>` fallback stays UNvetoed there,
/// so callers falling back to `data_root` keep that behavior).
pub(crate) fn find_in_family_dirs(data_root: &Path, basename: &str) -> Option<PathBuf> {
    let version = data_root.file_name()?.to_str()?;
    let backend = data_root.parent()?.file_name()?.to_str()?;
    let data_dir = data_root.parent()?.parent()?;
    for entry in std::fs::read_dir(data_dir).ok()?.flatten() {
        let name = entry.file_name();
        let name = match name.to_str() {
            Some(name) => name,
            None => continue,
        };
        if name.starts_with('.') || KNOWN_BACKEND_DIRS.contains(&name) {
            continue;
        }
        let version_dir = entry.path().join(backend).join(version);
        if version_dir_is_unusable(&version_dir) {
            continue;
        }
        let candidate = version_dir.join(basename);
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
/// inside each table's `resolver.sources_for` call via `find_in_family_dirs`.
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
    let family_root = system_data_root
        .join("comm")
        .join(backend_dir)
        .join(version);
    // A family dir under the legacy INCOMPLETE veto is never admitted —
    // Python's resolve_op_data_path skipped it and fell through to the
    // legacy layout (which _build_op_sources admission-checks separately).
    if family_root.is_dir() && !version_dir_is_unusable(&family_root) {
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
mod axis_curve;
pub mod communication;
pub mod dsa;
pub mod dsv4;
pub mod dsv4_megamoe;
pub mod fpm_forward;
pub mod gemm;
mod interpolation;
pub mod mhc;
pub mod mla;
pub mod moe;
pub mod moe_a2a;
mod moe_index;
pub mod msa;
pub mod parquet_loader;
pub mod perf_interp;
pub mod source_resolution;
pub mod state_space;
pub mod table_view;
pub mod trtllm_alltoall;
pub mod wideep_mla;

pub use attention::AttentionTable;
pub use communication::CommunicationTable;
pub use dsa::DsaTable;
pub use dsv4::{AttnKind, Dsv4Table};
pub use dsv4_megamoe::Dsv4MegaMoeTable;
pub use fpm_forward::FpmForwardTable;
pub use gemm::GemmTable;
pub use mhc::MhcTable;
pub use mla::MlaTable;
pub use moe::MoeTable;
pub use moe_a2a::MoeA2aTable;
pub use msa::MsaTable;
pub use source_resolution::{resolve_one, ResolveCtx, ResolveReport, SourceResolver};
pub use state_space::StateSpaceTable;
pub use trtllm_alltoall::TrtllmAlltoallTable;
pub use wideep_mla::WideEpMlaTable;

/// The loaded per-family perf tables for one caller-facing
/// `<system>/<backend>/<version>` tuple — the mode-independent,
/// immutable-after-load data half of
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
    pub moe_a2a: MoeA2aTable,
    pub communication: CommunicationTable,
    pub dsa: DsaTable,
    pub msa: MsaTable,
    pub dsv4: Dsv4Table,
    pub dsv4_megamoe: Dsv4MegaMoeTable,
    pub mhc: MhcTable,
    pub trtllm_alltoall: TrtllmAlltoallTable,
    pub wideep_mla: WideEpMlaTable,
    pub state_space: StateSpaceTable,
    pub fpm_forward: FpmForwardTable,
    /// The load's source resolver, retained so the table views
    /// (`table_view.rs`) can resolve any basename on demand through the same
    /// channel logic the query tables used.
    pub source_resolver: Arc<SourceResolver>,
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
    /// Test-only: swap the fpm_forward table so fixtures can point it at a
    /// temp collector pair while every other table stays on the checked-in
    /// fixture DB. The fixture db must be uniquely owned (no clones yet).
    #[cfg(test)]
    pub(crate) fn set_fpm_forward_for_test(&mut self, table: FpmForwardTable) {
        Arc::get_mut(&mut self.tables)
            .expect("test fixture db must be uniquely owned")
            .fpm_forward = table;
    }

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
        Self::load_with_sources(
            systems_root,
            system,
            backend,
            version,
            &PerfDbSources::default(),
        )
    }

    /// Like [`PerfDatabase::load`], but honours an explicit pre-materialized
    /// shared-layer source map (tests / synthetic injections). For op files
    /// present in the map, the ordered source list (with per-source
    /// `kernel_source` filters) is used instead of the single primary file.
    /// Op files absent from the map fall back to the primary `data_root`
    /// (identical to [`PerfDatabase::load`]). Production engine loads use
    /// [`PerfDatabase::load_resolved`] instead — the engine owns source
    /// resolution (`source_resolution.rs`) since the deprecation-cleanup PR.
    pub fn load_with_sources(
        systems_root: &Path,
        system: &str,
        backend: &str,
        version: &str,
        perf_db_sources: &PerfDbSources,
    ) -> Result<Self, AicError> {
        Self::load_with_sources_opts(
            systems_root,
            system,
            backend,
            version,
            perf_db_sources,
            false,
        )
    }

    /// [`PerfDatabase::load_with_sources`] with an estimate-only escape hatch:
    /// `tolerate_missing_data` skips the perf-data-directory existence gate so
    /// a system that ships only a spec yaml (Python's `allow_missing_data`
    /// "estimate" databases) can still back a SOL view — every SOL answer is
    /// analytic from the system spec, and any table-backed lookup raises its
    /// own per-family miss lazily. Non-SOL callers must keep the loud gate:
    /// a typo'd version string should fail at load, not as per-op misses.
    pub fn load_with_sources_opts(
        systems_root: &Path,
        system: &str,
        backend: &str,
        version: &str,
        perf_db_sources: &PerfDbSources,
        tolerate_missing_data: bool,
    ) -> Result<Self, AicError> {
        Self::load_with_resolver(
            systems_root,
            system,
            backend,
            version,
            Arc::new(SourceResolver::fixed(perf_db_sources.clone())),
            tolerate_missing_data,
        )
    }

    /// The live-resolution context for a load identity. Reads the system
    /// yaml to locate the data dir; policy flags mirror the Python view's
    /// `enable_shared_layer` / `strict_provenance` attributes.
    pub fn resolve_ctx(
        systems_root: &Path,
        system: &str,
        backend: &str,
        version: &str,
        enable_shared_layer: bool,
        strict_provenance: bool,
    ) -> Result<ResolveCtx, AicError> {
        let system_yaml = systems_root.join(format!("{system}.yaml"));
        let spec = SystemSpec::load(&system_yaml)?;
        Ok(ResolveCtx {
            systems_root: systems_root.to_path_buf(),
            system_data_root: systems_root.join(&spec.data_dir),
            backend: backend.to_string(),
            version: version.to_string(),
            enable_shared_layer,
            strict: strict_provenance,
        })
    }

    /// Load with ENGINE-OWNED shared-layer source resolution (Collector V3
    /// design §6): every table's source list is resolved live from the
    /// perf-data tree (`source_resolution.rs`) — primary, declared reuse,
    /// nearest-earlier fallback, and manifest-gated cross-backend fill —
    /// exactly as the retired Python `_build_op_sources` did. This is the
    /// production path behind `Engine::from_spec_bytes`.
    pub fn load_resolved(
        systems_root: &Path,
        system: &str,
        backend: &str,
        version: &str,
        enable_shared_layer: bool,
        strict_provenance: bool,
        tolerate_missing_data: bool,
    ) -> Result<Self, AicError> {
        let ctx = Self::resolve_ctx(
            systems_root,
            system,
            backend,
            version,
            enable_shared_layer,
            strict_provenance,
        )?;
        Self::load_with_resolver(
            systems_root,
            system,
            backend,
            version,
            Arc::new(SourceResolver::live(ctx)),
            tolerate_missing_data,
        )
    }

    fn load_with_resolver(
        systems_root: &Path,
        system: &str,
        backend: &str,
        version: &str,
        resolver: Arc<SourceResolver>,
        tolerate_missing_data: bool,
    ) -> Result<Self, AicError> {
        let system_yaml = systems_root.join(format!("{system}.yaml"));
        let spec = SystemSpec::load(&system_yaml)?;
        let system_data_root = systems_root.join(&spec.data_dir);
        let data_root = system_data_root.join(backend).join(version);
        // Accept either layout: the legacy `<backend>/<version>` dir itself,
        // or at least one family-first `<family>/<backend>/<version>` dir
        // (family = any first-level dir under `system_data_root` other than
        // the known legacy backend names). `data_root` stays the legacy path
        // either way — each table's `resolver.sources_for` call resolves the
        // actual per-file location (legacy or family) independently.
        if !tolerate_missing_data
            && !data_root.is_dir()
            && !has_family_backend_version(&system_data_root, backend, version)
        {
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
            // Every op table resolves its own file basenames through the
            // resolver via `with_sources` (shared-layer aware). NCCL/OneCCL
            // are framework-agnostic and never inherit siblings, so their
            // roots stay as the direct system-wide dirs.
            gemm: GemmTable::with_sources(data_root.clone(), spec.clone(), &resolver)?,
            attention: AttentionTable::with_sources(data_root.clone(), spec.clone(), &resolver)?,
            mla: MlaTable::with_sources(data_root.clone(), spec.clone(), &resolver)?,
            moe: MoeTable::with_sources(data_root.clone(), &resolver)?,
            moe_a2a: MoeA2aTable::with_sources(data_root.clone(), &resolver)?,
            communication: CommunicationTable::with_sources(
                data_root.clone(),
                nccl_root,
                oneccl_root,
                &resolver,
            )?,
            dsa: DsaTable::with_sources(data_root.clone(), &resolver)?,
            dsv4: Dsv4Table::with_sources(data_root.clone(), &resolver)?,
            // Single-primary by design: the MegaMoE loader reads one unified
            // path and never the shared-layer source list (see
            // `dsv4_megamoe.rs`) — but that one path IS family-first
            // resolved, so take the head of the standard source resolution.
            dsv4_megamoe: Dsv4MegaMoeTable::with_primary(
                resolver
                    .sources_for("dsv4_megamoe_module_perf.parquet", &data_root)?
                    .into_iter()
                    .next()
                    .map(|PerfSource(path, _)| path)
                    .unwrap_or_else(|| data_root.join("dsv4_megamoe_module_perf.parquet")),
            ),
            msa: MsaTable::with_sources(data_root.clone(), &resolver)?,
            mhc: MhcTable::with_sources(data_root.clone(), &resolver)?,
            trtllm_alltoall: TrtllmAlltoallTable::with_sources(data_root.clone(), &resolver)?,
            wideep_mla: WideEpMlaTable::with_sources(data_root.clone(), spec.clone(), &resolver)?,
            state_space: StateSpaceTable::with_sources(
                data_root.clone(),
                backend,
                version,
                &resolver,
            )?,
            // Deliberately NOT shared-layer aware: FPM whole-model data is
            // valid only for its exact backend/version (fpm_forward.rs).
            fpm_forward: FpmForwardTable::new(data_root.clone(), system, backend, version),
            system_spec: spec,
            // Kept for the table-view folds (`table_view.rs`), which resolve
            // every basename themselves — including the wideep/deepep files
            // no query table loads.
            source_resolver: resolver,
            data_root,
        };
        Ok(Self::from_tables(Arc::new(tables)))
    }

    /// Like [`PerfDatabase::load_with_sources`], but shares the loaded
    /// [`PerfTables`] across databases with the same load identity
    /// (systems root, system, backend, version, shared-layer source map).
    ///
    /// The tables are the expensive half of a database: each family parses
    /// its parquet files behind a `OnceLock` on first query, so every
    /// separately-loaded instance re-parses the same files. Sharing them
    /// amortizes that parse across the many single-use engines a sweep
    /// compiles (one per model x parallelism x quant identity — the
    /// prediction-regression gate builds hundreds per combo). Only the
    /// immutable-after-load half is shared: every returned view gets fresh
    /// mode/policy fields, memo caches, and a fresh provenance accumulator,
    /// so two engines over the same tables never couple their runtime state.
    ///
    /// The memo holds `Weak` references and is swept on insert, so tables
    /// live exactly as long as some database still uses them. Concurrent
    /// first loads of the same identity may both build (benign: last insert
    /// wins and both instances work; the memo is an amortization, not a
    /// uniqueness guarantee).
    pub fn load_resolved_shared(
        systems_root: &Path,
        system: &str,
        backend: &str,
        version: &str,
        enable_shared_layer: bool,
        strict_provenance: bool,
        tolerate_missing_data: bool,
    ) -> Result<Self, AicError> {
        static SHARED_TABLES: OnceLock<SharedTablesMemo> = OnceLock::new();
        Self::load_resolved_shared_in(
            SHARED_TABLES.get_or_init(Default::default),
            systems_root,
            system,
            backend,
            version,
            enable_shared_layer,
            strict_provenance,
            tolerate_missing_data,
        )
    }

    /// [`Self::load_resolved_shared`] against an explicit memo. The public
    /// entry point passes the process-global memo; tests pass a private one
    /// so the same-identity sharing assertion is DETERMINISTIC — against the
    /// global memo it is only an amortization (parallel tests loading the
    /// same identity may overwrite or expire each other's entries, the
    /// documented concurrent-first-load behavior).
    #[allow(clippy::too_many_arguments)]
    fn load_resolved_shared_in(
        memo: &SharedTablesMemo,
        systems_root: &Path,
        system: &str,
        backend: &str,
        version: &str,
        enable_shared_layer: bool,
        strict_provenance: bool,
        tolerate_missing_data: bool,
    ) -> Result<Self, AicError> {
        if tolerate_missing_data {
            // Estimate-only loads bypass the memo entirely: caching a set of
            // empty tables under the plain identity key would let a later
            // STRICT load of the same identity silently succeed with empty
            // tables instead of raising the loud missing-directory error.
            return Self::load_resolved(
                systems_root,
                system,
                backend,
                version,
                enable_shared_layer,
                strict_provenance,
                true,
            );
        }
        let ctx = Self::resolve_ctx(
            systems_root,
            system,
            backend,
            version,
            enable_shared_layer,
            strict_provenance,
        )?;
        let resolver = Arc::new(SourceResolver::live(ctx));
        let key = shared_tables_key(systems_root, system, backend, version, &resolver);
        if let Some(tables) = memo.lock().unwrap().get(&key).and_then(Weak::upgrade) {
            return Ok(Self::from_tables(tables));
        }
        let db = Self::load_with_resolver(systems_root, system, backend, version, resolver, false)?;
        let mut map = memo.lock().unwrap();
        map.retain(|_, weak| weak.strong_count() > 0);
        map.insert(key, Arc::downgrade(&db.tables));
        Ok(db)
    }

    /// Fresh default-mode view over `tables`. Per-view state (query mode,
    /// transfer policy, memo caches, provenance accumulator) always starts
    /// fresh here — the invariant [`PerfDatabase::load_with_sources_shared`]
    /// relies on to share tables without coupling views.
    fn from_tables(tables: Arc<PerfTables>) -> Self {
        Self {
            tables,
            database_mode: DatabaseMode::default(),
            transfer_policy: TransferPolicy::ALL,
            util_grids: Arc::new(UtilGridCache::new()),
            delta_lookups: Arc::new(DeltaLookupCache::new()),
            provenance: Arc::new(AtomicU8::new(ProvenanceTier::Silicon as u8)),
        }
    }

    /// The shared tables `Arc` (the load identity of the underlying data).
    /// Exposed for the engine layer's table-sharing tests.
    #[cfg(test)]
    pub(crate) fn tables_arc(&self) -> &Arc<PerfTables> {
        &self.tables
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
    pub fn with_mode(
        mut self,
        database_mode: DatabaseMode,
        transfer_policy: TransferPolicy,
    ) -> Self {
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

    /// The SOL_FULL view over the same loaded tables (cheap: `Arc` clones).
    ///
    /// Mirrors passing `database_mode=DatabaseMode.SOL_FULL` per call on the
    /// Python `query_*` facade: every operator query takes its analytic SOL
    /// branch (and attaches `PerformanceResult::sol` components) regardless
    /// of the engine's configured mode. Used by the per-op SOL-decomposition
    /// FFI (`evaluate_ops_sol_json`).
    pub fn sol_full_view(&self) -> PerfDatabase {
        PerfDatabase {
            tables: Arc::clone(&self.tables),
            database_mode: DatabaseMode::SolFull,
            transfer_policy: self.transfer_policy,
            util_grids: Arc::clone(&self.util_grids),
            delta_lookups: Arc::clone(&self.delta_lookups),
            provenance: Arc::clone(&self.provenance),
        }
    }
}

/// Shared fixtures for the per-family ENERGY oracle tests.
///
/// The shipped perf databases carry no `power` column (their energies are
/// all 0.0), so the energy oracles run on synthetic power-carrying parquet
/// fixtures. Each test documents its fixture rows and query; the pinned
/// values were minted by rebuilding the identical fixture with pandas and
/// querying the frozen Python SDK, e.g.:
///
/// ```text
/// cd <repo> && uv run python - <<'PY'
/// import pandas as pd, yaml, tempfile, os
/// from aiconfigurator_core.sdk.perf_database import PerfDatabase
/// from aiconfigurator_core.sdk import common
/// root = tempfile.mkdtemp(); data = os.path.join(root, "data", "vllm", "1.0")
/// os.makedirs(data)
/// yaml.safe_dump({...the testsys spec below...},
///                open(os.path.join(root, "testsys.yaml"), "w"))
/// pd.DataFrame([...the test's rows...]).to_parquet(
///     os.path.join(data, "<family>_perf.parquet"))
/// db = PerfDatabase("testsys", "vllm", "1.0", root)
/// r = db.query_<family>(..., database_mode=common.DatabaseMode.SILICON)
/// print(float(r), r.energy)
/// PY
/// ```
///
/// The system spec here mirrors that script's `testsys.yaml` byte for byte
/// so SOL-dependent paths (the GEMM load clamp, the fp8_static latency
/// floor) agree across both engines.
#[cfg(test)]
pub(crate) mod energy_test_fixtures {
    use std::fs::File;
    use std::path::Path;
    use std::sync::Arc;

    use parquet::data_type::{BoolType, ByteArray, ByteArrayType, DataType, DoubleType, Int64Type};
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::{SerializedFileWriter, SerializedRowGroupWriter};
    use parquet::schema::parser::parse_message_type;

    use crate::common::system_spec::{GpuSpec, MiscSpec, NodeSpec, SystemSpec};

    /// One parquet column of the fixture: name + values (row-aligned).
    pub(crate) enum Col {
        Str(&'static str, Vec<&'static str>),
        I64(&'static str, Vec<i64>),
        F64(&'static str, Vec<f64>),
        Bool(&'static str, Vec<bool>),
    }

    fn write_typed<T: DataType>(rg: &mut SerializedRowGroupWriter<'_, File>, values: &[T::T]) {
        let mut col = rg.next_column().unwrap().expect("column");
        col.typed::<T>().write_batch(values, None, None).unwrap();
        col.close().unwrap();
    }

    /// Write one single-row-group parquet with the given columns.
    pub(crate) fn write_parquet(path: &Path, cols: &[Col]) {
        let fields: Vec<String> = cols
            .iter()
            .map(|c| match c {
                Col::Str(name, _) => format!("REQUIRED BYTE_ARRAY {name} (UTF8);"),
                Col::I64(name, _) => format!("REQUIRED INT64 {name};"),
                Col::F64(name, _) => format!("REQUIRED DOUBLE {name};"),
                Col::Bool(name, _) => format!("REQUIRED BOOLEAN {name};"),
            })
            .collect();
        let schema = format!("message energy_fixture {{ {} }}", fields.join(" "));
        let schema = Arc::new(parse_message_type(&schema).expect("schema must parse"));
        let file = File::create(path).expect("create parquet");
        let mut writer =
            SerializedFileWriter::new(file, schema, Arc::new(WriterProperties::builder().build()))
                .expect("writer");
        let mut rg = writer.next_row_group().expect("row group");
        for col in cols {
            match col {
                Col::Str(_, v) => {
                    let bytes: Vec<ByteArray> = v.iter().map(|s| ByteArray::from(*s)).collect();
                    write_typed::<ByteArrayType>(&mut rg, &bytes);
                }
                Col::I64(_, v) => write_typed::<Int64Type>(&mut rg, v),
                Col::F64(_, v) => write_typed::<DoubleType>(&mut rg, v),
                Col::Bool(_, v) => write_typed::<BoolType>(&mut rg, v),
            }
        }
        rg.close().unwrap();
        writer.close().unwrap();
    }

    /// The oracle script's `testsys.yaml` gpu/node numbers, as an in-code
    /// spec for family tables constructed directly from a data dir.
    pub(crate) fn energy_test_spec() -> SystemSpec {
        SystemSpec {
            data_dir: "data".into(),
            gpu: GpuSpec {
                mem_bw: 7.7e12,
                mem_bw_empirical_scaling_factor: 0.92,
                mem_empirical_constant_latency: 2e-6,
                mem_capacity: None,
                bfloat16_tc_flops: Some(2.25e15),
                int8_tc_flops: Some(4.5e15),
                fp8_tc_flops: Some(4.5e15),
                fp4_tc_flops: Some(9e15),
                power: None,
                sm_version: Some(100),
            },
            node: NodeSpec {
                num_gpus_per_node: 8,
                intra_node_bw: 900e9,
                inter_node_bw: 50e9,
                pcie_bw: None,
                p2p_latency: 2e-6,
                num_gpus_per_rack: None,
                inter_rack_bw: None,
            },
            misc: MiscSpec::default(),
        }
    }

    /// Write `testsys.yaml` (the YAML twin of [`energy_test_spec`]) into a
    /// synthetic systems root and return the fixture data dir
    /// `<root>/data/vllm/1.0` — the layout `PerfDatabase::load(root,
    /// "testsys", "vllm", "1.0")` resolves.
    pub(crate) fn write_energy_systems_root(root: &Path) -> std::path::PathBuf {
        let yaml = r#"
data_dir: data
gpu:
  mem_bw: 7.7e12
  mem_bw_empirical_scaling_factor: 0.92
  mem_empirical_constant_latency: 2.0e-6
  bfloat16_tc_flops: 2.25e15
  int8_tc_flops: 4.5e15
  fp8_tc_flops: 4.5e15
  fp4_tc_flops: 9.0e15
  sm_version: 100
node:
  num_gpus_per_node: 8
  intra_node_bw: 900.0e9
  inter_node_bw: 50.0e9
  p2p_latency: 2.0e-6
misc:
  nccl_version: test
"#;
        std::fs::write(root.join("testsys.yaml"), yaml).expect("write yaml");
        let data = root.join("data").join("vllm").join("1.0");
        std::fs::create_dir_all(&data).expect("mkdir data");
        data
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
        assert_eq!(gemm_sources.len(), 1);
        assert!(
            gemm_sources[0].0.is_file(),
            "resolved GEMM parquet must exist: {}",
            gemm_sources[0].0.display()
        );
    }

    #[test]
    fn shared_load_reuses_tables_and_isolates_view_state() {
        // A PRIVATE memo: against the process-global one this assertion is
        // flaky under parallel `cargo test` — other tests loading the same
        // identity overwrite/expire the entry (the documented
        // concurrent-first-load amortization), which is exactly what the
        // uniqueness assertion below must not race with.
        let memo = SharedTablesMemo::default();
        let db1 = PerfDatabase::load_resolved_shared_in(
            &memo,
            &systems_root(),
            "b200_sxm",
            "vllm",
            "0.19.0",
            false,
            false,
            false,
        )
        .expect("shared load must succeed");
        let db2 = PerfDatabase::load_resolved_shared_in(
            &memo,
            &systems_root(),
            "b200_sxm",
            "vllm",
            "0.19.0",
            false,
            false,
            false,
        )
        .expect("shared load must succeed");
        assert!(
            Arc::ptr_eq(&db1.tables, &db2.tables),
            "same load identity must share the parsed tables"
        );
        // Per-view state must stay isolated: a provenance note (and memo
        // caches) on one view never leaks into another view over the same
        // shared tables.
        db1.note_provenance(ProvenanceTier::Empirical);
        assert_eq!(db2.worst_provenance(), ProvenanceTier::Silicon);
        assert!(!Arc::ptr_eq(&db1.util_grids, &db2.util_grids));
        assert!(!Arc::ptr_eq(&db1.delta_lookups, &db2.delta_lookups));
    }

    #[test]
    fn missing_data_dir_is_tolerated_only_when_requested() {
        // Estimate-only escape hatch (#1552 review finding 8): a system with a
        // spec yaml but NO perf-data directory must load under the tolerant
        // flag (SOL answers are analytic from the spec) and must keep failing
        // loudly under the strict default.
        let strict = PerfDatabase::load_with_sources_opts(
            &systems_root(),
            "h100_pcie",
            "trtllm",
            "estimate",
            &PerfDbSources::default(),
            false,
        );
        assert!(
            strict
                .err()
                .map(|e| e.to_string().contains("perf data directory not found"))
                .unwrap_or(false),
            "strict load of a data-less tuple must raise the missing-directory error"
        );
        let tolerant = PerfDatabase::load_with_sources_opts(
            &systems_root(),
            "h100_pcie",
            "trtllm",
            "estimate",
            &PerfDbSources::default(),
            true,
        )
        .expect("tolerant load must succeed from the spec yaml alone");
        // Table-backed lookups still miss lazily per family.
        assert!(tolerant
            .gemm
            .query(
                crate::common::enums::GemmQuantMode::Bfloat16,
                64,
                4096,
                4096
            )
            .is_err());
    }

    #[test]
    fn shared_load_distinct_policies_load_fresh_tables() {
        let db_no_shared = PerfDatabase::load_resolved_shared(
            &systems_root(),
            "b200_sxm",
            "vllm",
            "0.19.0",
            false,
            false,
            false,
        )
        .expect("shared load must succeed");
        let db_shared = PerfDatabase::load_resolved_shared(
            &systems_root(),
            "b200_sxm",
            "vllm",
            "0.19.0",
            true,
            false,
            false,
        )
        .expect("shared load must succeed");
        assert!(
            !Arc::ptr_eq(&db_no_shared.tables, &db_shared.tables),
            "a different shared-layer policy is a different load identity"
        );
        // Map-based loads are never memoized: each is a fresh identity.
        let db_map_a = PerfDatabase::load_with_sources(
            &systems_root(),
            "b200_sxm",
            "vllm",
            "0.19.0",
            &PerfDbSources::default(),
        )
        .expect("map load must succeed");
        let db_map_b = PerfDatabase::load_with_sources(
            &systems_root(),
            "b200_sxm",
            "vllm",
            "0.19.0",
            &PerfDbSources::default(),
        )
        .expect("map load must succeed");
        assert!(
            !Arc::ptr_eq(&db_map_a.tables, &db_map_b.tables),
            "explicit-map loads bypass the shared-tables memo"
        );
    }

    #[test]
    fn shared_tables_are_dropped_with_their_last_view() {
        // A unique tempdir root gives this test its own memo identity, so
        // parallel tests holding the real-fixture tables can't keep this
        // entry alive.
        let tmp = tempfile::tempdir().unwrap();
        energy_test_fixtures::write_energy_systems_root(tmp.path());
        let db = PerfDatabase::load_resolved_shared(
            tmp.path(),
            "testsys",
            "vllm",
            "1.0",
            false,
            false,
            false,
        )
        .expect("fixture load must succeed");
        let weak = Arc::downgrade(&db.tables);
        drop(db);
        assert!(
            weak.upgrade().is_none(),
            "the memo must hold Weak refs only — tables die with their last view"
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

        let sources = resolve_op_sources(
            &PerfDbSources::default(),
            "gemm_perf.parquet",
            &legacy_data_root,
        );
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

        let sources = resolve_op_sources(
            &PerfDbSources::default(),
            "nccl_perf.parquet",
            &legacy_data_root,
        );
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
        let family_dir = systems_root
            .join("data")
            .join("gemm")
            .join(backend)
            .join(version);
        std::fs::create_dir_all(&family_dir).unwrap();
        std::fs::write(family_dir.join("gemm_perf.parquet"), b"stub").unwrap();

        let legacy_data_root = systems_root.join("data").join(backend).join(version);
        assert!(
            !legacy_data_root.is_dir(),
            "fixture must not have a legacy dir"
        );

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
                assert!(
                    msg.contains("legacy"),
                    "error should mention legacy layout: {msg}"
                );
                assert!(
                    msg.contains("family"),
                    "error should mention family layout: {msg}"
                );
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
