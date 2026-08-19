// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared-layer perf-data source resolution (Collector V3 design §6).
//!
//! Faithful port of the retired Python resolver
//! (`sdk/perf_database.py::_build_op_sources` and its helper graph). The
//! engine owns resolution now: Python passes only the load identity
//! (systems root, system, backend, version) plus policy
//! (`enable_shared_layer`, `strict_provenance`), and every table resolves its
//! source list through [`SourceResolver`]. The Python side re-derives the
//! same report over FFI (`py.rs::resolve_op_sources_report_json`) for its
//! `data_provenance` diagnostics and for re-emitting warnings through the
//! established logging registries — resolution is a pure function of the
//! tree, so the two derivations always agree.
//!
//! Ordering, in priority (design §6):
//!   1. Active backend/version (primary). No `kernel_source` filter.
//!   2. Declared donors from the REQUESTED version dir's `reuse.yaml`
//!      (design §6.3), in file order. The only channel that may borrow a
//!      version NEWER than requested. No filter.
//!   3. Same-backend siblings STRICTLY EARLIER than requested (design §6.2),
//!      nearest first. No filter.
//!   4. Cross-backend fill (design §6.4), kernel-identity gated by
//!      `perf_data_reuse_manifest.yaml`, newest-first per framework.
//!
//! Deliberate divergences from the retired Python resolver (both documented
//! at their sites): no `.parquet -> .txt` candidate fallback (the engine
//! reads parquet only — `.txt` retired with the Python parsers in PR-6), and
//! structurally malformed manifest/reuse documents error with Rust-built
//! messages whose *prefixes* mirror the Python `ValueError` texts (the
//! parser-detail suffix differs between PyYAML and serde_yaml).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Mutex;

use pep440_rs::Version;
use serde::Serialize;

use crate::common::error::AicError;
use crate::config::{PerfDbSources, PerfSource};

use super::{find_in_family_dirs, version_dir_is_unusable, KNOWN_BACKEND_DIRS};

const REUSE_YAML_MARKER: &str = "reuse.yaml";
const COLLECTION_META_MARKER: &str = "collection_meta.yaml";
const INCOMPLETE_MARKER: &str = "INCOMPLETE.txt";
const SHARED_LAYER_REUSE_MARKER: &str = "SHARED_LAYER_REUSE.txt";
const COMM_FAMILY_DIR: &str = "comm";
/// Framework-agnostic comm tables: never inherit siblings (op-name early
/// return, independent of the family-path check). Mirrors the Python
/// `PerfDataFilename.nccl / .oneccl` early exit.
const FRAMEWORK_AGNOSTIC_BASENAMES: [&str; 2] = ["nccl_perf.parquet", "oneccl_perf.parquet"];
const REUSE_ENTRY_REQUIRED_KEYS: [&str; 4] = ["table", "from_version", "reason", "approved_by"];

fn perf_err(msg: String) -> AicError {
    AicError::PerfDatabase(msg)
}

// ---------------------------------------------------------------------------
// Warnings: structured events, emitted by the PYTHON side through its
// existing warn-once registries / log formats. The engine-internal resolution
// discards them (the Python view re-derives and logs).
// ---------------------------------------------------------------------------

/// One structured warning event. `kind` selects the Python-side emitter:
///   primary_veto          args = [primary_path, op_file_basename, version_dir]
///   legacy_layout         args = [data_dir]
///   legacy_marker         args = [scope, marker_name, replacement]
///   malformed_sidecar     args = [version_path, message]
///   malformed_reuse       args = [reuse_path, message]
///   duplicate_declared    args = [table, from_version, system_data_root]  (DEBUG)
///   unparseable_sibling   args = [system_data_root, backend, version]
///   low_fidelity          args = [op_file_basename, sibling_path]
///   strict_provenance     args = [dedupe_kind, dedupe_key, message]
#[derive(Clone, Debug, Serialize)]
pub struct ResolverWarning {
    pub kind: &'static str,
    pub args: Vec<String>,
}

fn warn(kind: &'static str, args: Vec<String>) -> ResolverWarning {
    ResolverWarning { kind, args }
}

// ---------------------------------------------------------------------------
// Report shapes
// ---------------------------------------------------------------------------

/// One admitted source, tagged with its channel — the Rust twin of a
/// `data_provenance` entry plus the load-facing `kernel_source` filter.
#[derive(Clone, Debug, Serialize)]
pub struct ResolvedRecord {
    pub version: String,
    pub path: String,
    pub channel: &'static str,
    pub exists: bool,
    /// `None` admits every row; `Some` keeps only matching `kernel_source`
    /// rows (cross-backend fill). Sorted (BTreeSet), matching the Python
    /// wire projection `sorted(ks)`.
    pub ks_filter: Option<BTreeSet<String>>,
}

/// Resolution result for ONE op-file basename.
#[derive(Clone, Debug, Default, Serialize)]
pub struct ResolveReport {
    pub records: Vec<ResolvedRecord>,
    pub warnings: Vec<ResolverWarning>,
}

impl ResolveReport {
    /// Project the load-facing source list (path + optional filter),
    /// mirroring the Python `_finish()` return value.
    pub fn sources(&self) -> Vec<PerfSource> {
        self.records
            .iter()
            .map(|r| {
                PerfSource(
                    PathBuf::from(&r.path),
                    r.ks_filter
                        .as_ref()
                        .map(|set| set.iter().cloned().collect::<Vec<_>>()),
                )
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// YAML helpers (error texts mirror the Python ValueError prefixes)
// ---------------------------------------------------------------------------

/// Python-ish type name for error-message parity with `type(x).__name__`.
fn yaml_type_name(v: &serde_yaml::Value) -> &'static str {
    match v {
        serde_yaml::Value::Null => "NoneType",
        serde_yaml::Value::Bool(_) => "bool",
        serde_yaml::Value::Number(n) => {
            if n.is_f64() {
                "float"
            } else {
                "int"
            }
        }
        serde_yaml::Value::String(_) => "str",
        serde_yaml::Value::Sequence(_) => "list",
        serde_yaml::Value::Mapping(_) => "dict",
        serde_yaml::Value::Tagged(_) => "object",
    }
}

fn load_yaml_mapping(path: &Path, label: &str) -> Result<serde_yaml::Mapping, AicError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| perf_err(format!("{}: failed to parse {label}: {e}", path.display())))?;
    let value: serde_yaml::Value = serde_yaml::from_str(&text)
        .map_err(|e| perf_err(format!("{}: failed to parse {label}: {e}", path.display())))?;
    match value {
        serde_yaml::Value::Mapping(m) => Ok(m),
        other => Err(perf_err(format!(
            "{}: expected a YAML mapping at the top level, got {}",
            path.display(),
            yaml_type_name(&other)
        ))),
    }
}

fn record_strict_provenance_warning(
    warnings: &mut Vec<ResolverWarning>,
    kind: &str,
    key: &Path,
    message: String,
) {
    warnings.push(warn(
        "strict_provenance",
        vec![kind.to_string(), key.display().to_string(), message],
    ));
}

fn version_dir_data_stems(version_path: &Path) -> std::io::Result<BTreeSet<String>> {
    let entries = match std::fs::read_dir(version_path) {
        Ok(entries) => entries,
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
            ) =>
        {
            return Ok(BTreeSet::new())
        }
        Err(error) => return Err(error),
    };
    let mut stems = BTreeSet::new();
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.')
            || matches!(
                name.as_ref(),
                REUSE_YAML_MARKER
                    | COLLECTION_META_MARKER
                    | INCOMPLETE_MARKER
                    | SHARED_LAYER_REUSE_MARKER
            )
            || !entry.path().is_file()
        {
            continue;
        }
        if let Some(stem) = entry.path().file_stem() {
            stems.insert(stem.to_string_lossy().into_owned());
        }
    }
    Ok(stems)
}

/// Enforce Collector V3 sidecar coverage at native admission time. Primary
/// directories check every real data table; declared donors check the one
/// table being admitted. Callers exclude legacy-layout directories.
fn check_strict_provenance_coverage(
    version_path: &Path,
    strict: bool,
    only_table: Option<&str>,
    warnings: &mut Vec<ResolverWarning>,
) -> Result<(), AicError> {
    let stems = match only_table {
        Some(table) => BTreeSet::from([table.to_string()]),
        None => match version_dir_data_stems(version_path) {
            Ok(stems) => stems,
            Err(error) => {
                let message = format!(
                    "{}: cannot inspect perf-data files ({error}); strict provenance cannot verify \
                     sidecar coverage (Collector V3 design §5/§7.4)",
                    version_path.display()
                );
                if strict {
                    return Err(perf_err(message));
                }
                record_strict_provenance_warning(
                    warnings,
                    "unreadable-version-dir",
                    version_path,
                    message,
                );
                return Ok(());
            }
        },
    };
    if stems.is_empty() {
        return Ok(());
    }

    let meta_path = version_path.join(COLLECTION_META_MARKER);
    if !meta_path.is_file() {
        let tables: Vec<&String> = stems.iter().collect();
        let message = format!(
            "{}: holds table(s) {tables:?} with no collection_meta.yaml sidecar \
             (Collector V3 design §5/§7.4)",
            version_path.display()
        );
        if strict {
            return Err(perf_err(message));
        }
        record_strict_provenance_warning(warnings, "missing-sidecar", version_path, message);
        return Ok(());
    }

    let meta = match load_yaml_mapping(&meta_path, "collection_meta.yaml") {
        Ok(meta) => meta,
        Err(error) => {
            if strict {
                return Err(error);
            }
            record_strict_provenance_warning(
                warnings,
                "malformed-sidecar",
                &meta_path,
                error.to_string(),
            );
            return Ok(());
        }
    };
    let covered: BTreeSet<String> = match meta.get(serde_yaml::Value::String("tables".to_string())) {
        Some(serde_yaml::Value::Mapping(tables)) => tables
            .keys()
            .filter_map(|key| match key {
                serde_yaml::Value::String(name) => Some(name.clone()),
                _ => None,
            })
            .collect(),
        _ => BTreeSet::new(),
    };
    let uncovered: BTreeSet<String> = stems.difference(&covered).cloned().collect();
    if uncovered.is_empty() {
        return Ok(());
    }

    let tables: Vec<&String> = uncovered.iter().collect();
    let legacy = matches!(
        meta.get(serde_yaml::Value::String("provenance".to_string())),
        Some(serde_yaml::Value::String(value)) if value == "legacy"
    );
    if legacy {
        let message = format!(
            "{}: provenance: legacy sidecar does not list table(s) {tables:?}; \
             graced for one release (Collector V3 design §5)",
            meta_path.display()
        );
        record_strict_provenance_warning(warnings, "legacy-uncovered", &meta_path, message);
        return Ok(());
    }

    let message = format!(
        "{}: table(s) {tables:?} not covered by collection_meta.yaml 'tables' entries \
         (Collector V3 design §5/§7.4)",
        meta_path.display()
    );
    if strict {
        return Err(perf_err(message));
    }
    record_strict_provenance_warning(warnings, "uncovered-table", &meta_path, message);
    Ok(())
}

#[derive(Clone, Debug)]
struct ReuseEntry {
    table: String,
    from_version: String,
}

/// Parse+validate a `reuse.yaml` sidecar (design §6.3). Mirrors the Python
/// `_parse_reuse_yaml`: fail loudly on any structural or type mismatch; a
/// present-but-empty `reuse: []` is a valid "nothing declared" document; a
/// MISSING top-level `reuse` key is a schema error.
fn parse_reuse_yaml(path: &Path) -> Result<Vec<ReuseEntry>, AicError> {
    let mapping = load_yaml_mapping(path, "reuse.yaml")?;
    let entries_value = mapping
        .get(serde_yaml::Value::String("reuse".to_string()))
        .ok_or_else(|| {
            perf_err(format!(
                "{}: missing required top-level 'reuse' key",
                path.display()
            ))
        })?;
    let entries = match entries_value {
        serde_yaml::Value::Sequence(seq) => seq,
        other => {
            return Err(perf_err(format!(
                "{}: 'reuse' must be a list, got {}",
                path.display(),
                yaml_type_name(other)
            )))
        }
    };
    let mut validated = Vec::with_capacity(entries.len());
    for (i, entry) in entries.iter().enumerate() {
        let entry_map = match entry {
            serde_yaml::Value::Mapping(m) => m,
            other => {
                return Err(perf_err(format!(
                    "{}: reuse[{i}] must be a mapping, got {}",
                    path.display(),
                    yaml_type_name(other)
                )))
            }
        };
        let missing: Vec<&str> = REUSE_ENTRY_REQUIRED_KEYS
            .iter()
            .copied()
            .filter(|key| {
                !entry_map.contains_key(serde_yaml::Value::String((*key).to_string()))
            })
            .collect();
        if !missing.is_empty() {
            return Err(perf_err(format!(
                "{}: reuse[{i}] missing required key(s): {}",
                path.display(),
                missing.join(", ")
            )));
        }
        let mut fields = BTreeMap::new();
        for key in REUSE_ENTRY_REQUIRED_KEYS {
            let value = &entry_map[&serde_yaml::Value::String(key.to_string())];
            match value {
                serde_yaml::Value::String(s) if !s.trim().is_empty() => {
                    fields.insert(key, s.clone());
                }
                _ => {
                    return Err(perf_err(format!(
                        "{}: reuse[{i}].{key} must be a non-empty string",
                        path.display()
                    )))
                }
            }
        }
        validated.push(ReuseEntry {
            table: fields["table"].clone(),
            from_version: fields["from_version"].clone(),
        });
    }
    Ok(validated)
}

// ---------------------------------------------------------------------------
// Version-dir marker state (the ADMISSION-layer strict predicate; the
// resolver-side lenient existence check stays `version_dir_is_unusable`)
// ---------------------------------------------------------------------------

/// Whole-directory exclusion state, yaml-first with legacy `.txt` fallback.
/// Mirrors the Python `_version_dir_state` unusable branch INCLUDING its
/// side effects: both sidecars are parsed when present (so malformed
/// authored metadata surfaces loudly) and honoring a legacy marker records a
/// deprecation warning. `status: partial` is NOT grounds for rejection.
fn version_dir_state_unusable(
    version_path: &Path,
    warn_scope: &Path,
    warnings: &mut Vec<ResolverWarning>,
) -> Result<bool, AicError> {
    let reuse_yaml_path = version_path.join(REUSE_YAML_MARKER);
    let legacy_reuse_path = version_path.join(SHARED_LAYER_REUSE_MARKER);
    if reuse_yaml_path.is_file() {
        parse_reuse_yaml(&reuse_yaml_path)?;
    } else if legacy_reuse_path.is_file() {
        warnings.push(warn(
            "legacy_marker",
            vec![
                warn_scope.display().to_string(),
                SHARED_LAYER_REUSE_MARKER.to_string(),
                REUSE_YAML_MARKER.to_string(),
            ],
        ));
    }

    let meta_yaml_path = version_path.join(COLLECTION_META_MARKER);
    let legacy_incomplete_path = version_path.join(INCOMPLETE_MARKER);
    if meta_yaml_path.is_file() {
        // Parsed for fail-loud parity; partial tables are informational.
        load_yaml_mapping(&meta_yaml_path, "collection_meta.yaml")?;
        Ok(false)
    } else if legacy_incomplete_path.is_file() {
        warnings.push(warn(
            "legacy_marker",
            vec![
                warn_scope.display().to_string(),
                INCOMPLETE_MARKER.to_string(),
                COLLECTION_META_MARKER.to_string(),
            ],
        ));
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Whether a request must reject the whole version directory. A malformed
/// sidecar raises in strict mode; non-strict mode records a warning and lets
/// normal loading continue. Mirrors `_version_dir_unusable_for_request`.
fn version_dir_unusable_for_request(
    version_path: &Path,
    data_dir: &Path,
    strict: bool,
    warnings: &mut Vec<ResolverWarning>,
) -> Result<bool, AicError> {
    match version_dir_state_unusable(version_path, data_dir, warnings) {
        Ok(unusable) => Ok(unusable),
        Err(e) => {
            if strict {
                return Err(e);
            }
            warnings.push(warn(
                "malformed_sidecar",
                vec![version_path.display().to_string(), e.to_string()],
            ));
            Ok(false)
        }
    }
}

// ---------------------------------------------------------------------------
// Directory walks (readdir order, mirroring os.listdir order dependence)
// ---------------------------------------------------------------------------

/// Resolve one op table under the family-first layout, legacy fallback.
/// Rust twin of `operations/base.py::resolve_op_data_path`, minus the
/// `.parquet -> .txt` candidate fallback (the engine reads parquet only).
pub(crate) fn resolve_op_data_path(
    system_data_root: &Path,
    backend: &str,
    version: &str,
    op_filename: &str,
) -> PathBuf {
    if let Ok(read_dir) = std::fs::read_dir(system_data_root) {
        for entry in read_dir.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') || KNOWN_BACKEND_DIRS.contains(&name.as_ref()) {
                continue;
            }
            let version_dir = system_data_root
                .join(name.as_ref())
                .join(backend)
                .join(version);
            if !version_dir.is_dir() || version_dir_is_unusable(&version_dir) {
                continue;
            }
            let candidate = version_dir.join(op_filename);
            if candidate.exists() {
                return candidate;
            }
        }
    }
    system_data_root.join(backend).join(version).join(op_filename)
}

/// Yield `(version, version_path)` for a backend across BOTH tree layouts.
/// Mirrors `_iter_backend_version_dirs`: family layout
/// `<data_dir>/<family>/<backend>/<version>` (any first-level dir that is
/// not a known backend dir) plus the deprecated legacy layout
/// `<data_dir>/<backend>/<version>` (warns once per tree, deduped
/// Python-side).
fn iter_backend_version_dirs(
    data_dir: &Path,
    backend: &str,
    warnings: &mut Vec<ResolverWarning>,
) -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    let Ok(read_dir) = std::fs::read_dir(data_dir) else {
        return out;
    };
    for entry in read_dir.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let entry_path = data_dir.join(name.as_ref());
        if name.starts_with('.') || !entry_path.is_dir() {
            continue;
        }
        if name == backend {
            // legacy layout
            warnings.push(warn("legacy_layout", vec![data_dir.display().to_string()]));
            iter_version_subdirs(&entry_path, &mut out);
        } else if !KNOWN_BACKEND_DIRS.contains(&name.as_ref()) {
            // family dir
            let backend_path = entry_path.join(backend);
            if backend_path.is_dir() {
                iter_version_subdirs(&backend_path, &mut out);
            }
        }
    }
    out
}

fn iter_version_subdirs(backend_path: &Path, out: &mut Vec<(String, PathBuf)>) {
    let Ok(read_dir) = std::fs::read_dir(backend_path) else {
        return;
    };
    for entry in read_dir.flatten() {
        let version = entry.file_name();
        let version = version.to_string_lossy();
        let version_path = backend_path.join(version.as_ref());
        if !version.starts_with(['.', '_']) && version_path.is_dir() {
            out.push((version.into_owned(), version_path));
        }
    }
}

/// Best-effort family-dir name for an op's resolved primary path. Returns
/// `None` for legacy-layout (3-component) or otherwise-unresolved paths —
/// the comm exclusion then simply does not trigger (deliberate transition
/// exception, pinned by `test_legacy_layout_comm_op_keeps_pre_v3_siblings`).
fn op_file_family_from_path(primary_path: &Path, system_data_root: &Path) -> Option<String> {
    let rel = primary_path.strip_prefix(system_data_root).ok()?;
    let parts: Vec<_> = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    if parts.len() == 4 && !KNOWN_BACKEND_DIRS.contains(&parts[0].as_str()) {
        return Some(parts[0].clone());
    }
    None
}

// ---------------------------------------------------------------------------
// reuse.yaml scoping + manifest
// ---------------------------------------------------------------------------

/// Declared-reuse entries for one op file, scoped to the REQUESTED version
/// dir(s) only. Mirrors `_requested_version_reuse_entries`: every
/// `reuse.yaml` found at the (backend, version) pair is parsed in dir order;
/// only entries whose `table` names this op file are kept, in file order. A
/// malformed `reuse.yaml` raises in strict mode; non-strict warns (deduped
/// Python-side with the load-time check's key) and declares zero donors.
fn requested_version_reuse_entries(
    system_data_root: &Path,
    backend: &str,
    version: &str,
    op_file_basename: &str,
    strict: bool,
    warnings: &mut Vec<ResolverWarning>,
) -> Result<Vec<ReuseEntry>, AicError> {
    let mut matched = Vec::new();
    for (candidate_version, version_path) in
        iter_backend_version_dirs(system_data_root, backend, warnings)
    {
        if candidate_version != version {
            continue;
        }
        let reuse_path = version_path.join(REUSE_YAML_MARKER);
        if !reuse_path.is_file() {
            continue;
        }
        let entries = match parse_reuse_yaml(&reuse_path) {
            Ok(entries) => entries,
            Err(e) => {
                if strict {
                    return Err(e);
                }
                warnings.push(warn(
                    "malformed_reuse",
                    vec![reuse_path.display().to_string(), e.to_string()],
                ));
                continue;
            }
        };
        for entry in entries {
            if format!("{}.parquet", entry.table) == op_file_basename {
                matched.push(entry);
            }
        }
    }
    Ok(matched)
}

#[derive(Clone, Debug)]
struct ManifestEntry {
    kernel_source: Option<String>,
    tier: Option<String>,
    frameworks: Vec<String>,
}

/// Manifest entries for one op file from
/// `<systems_root>/perf_data_reuse_manifest.yaml`. Mirrors the retired Python
/// loader (grouping by op_file with the `.txt -> .parquet` rename); absent
/// manifest = zero entries. Structural
/// violations that would crash the Python loader (a non-mapping group entry)
/// error loudly here too.
fn manifest_entries_for(
    systems_root: &Path,
    op_file_basename: &str,
) -> Result<Vec<ManifestEntry>, AicError> {
    let manifest_path = systems_root.join("perf_data_reuse_manifest.yaml");
    if !manifest_path.exists() {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(&manifest_path).map_err(|e| {
        perf_err(format!(
            "{}: failed to read perf_data_reuse_manifest.yaml: {e}",
            manifest_path.display()
        ))
    })?;
    let value: serde_yaml::Value = serde_yaml::from_str(&text).map_err(|e| {
        perf_err(format!(
            "{}: failed to parse perf_data_reuse_manifest.yaml: {e}",
            manifest_path.display()
        ))
    })?;
    // `yaml.safe_load(f) or {}` — an empty document is an empty manifest.
    let mapping = match value {
        serde_yaml::Value::Null => return Ok(Vec::new()),
        serde_yaml::Value::Mapping(m) => m,
        other => {
            return Err(perf_err(format!(
                "{}: expected a YAML mapping at the top level, got {}",
                manifest_path.display(),
                yaml_type_name(&other)
            )))
        }
    };
    let groups = match mapping.get(serde_yaml::Value::String("groups".to_string())) {
        None | Some(serde_yaml::Value::Null) => return Ok(Vec::new()),
        Some(serde_yaml::Value::Sequence(seq)) => seq,
        Some(other) => {
            return Err(perf_err(format!(
                "{}: 'groups' must be a list, got {}",
                manifest_path.display(),
                yaml_type_name(other)
            )))
        }
    };
    let mut matched = Vec::new();
    for entry in groups {
        let entry_map = match entry {
            serde_yaml::Value::Mapping(m) => m,
            other => {
                return Err(perf_err(format!(
                    "{}: manifest group entries must be mappings, got {}",
                    manifest_path.display(),
                    yaml_type_name(other)
                )))
            }
        };
        let op_file = match entry_map.get(serde_yaml::Value::String("op_file".to_string())) {
            Some(serde_yaml::Value::String(s)) if !s.is_empty() => s.clone(),
            _ => continue, // falsy / non-string op_file: skipped (Python `if not op_file`)
        };
        let op_file = if let Some(stem) = op_file.strip_suffix(".txt") {
            format!("{stem}.parquet")
        } else {
            op_file
        };
        if op_file != op_file_basename {
            continue;
        }
        let kernel_source = match entry_map.get(serde_yaml::Value::String("kernel_source".to_string()))
        {
            Some(serde_yaml::Value::String(s)) if !s.is_empty() => Some(s.clone()),
            _ => None, // falsy kernel_source is skipped at consumption
        };
        let tier = match entry_map.get(serde_yaml::Value::String("tier".to_string())) {
            Some(serde_yaml::Value::String(s)) => Some(s.clone()),
            _ => None,
        };
        let frameworks = match entry_map.get(serde_yaml::Value::String("frameworks".to_string())) {
            None | Some(serde_yaml::Value::Null) => Vec::new(),
            Some(serde_yaml::Value::Sequence(seq)) => {
                let mut fws = Vec::with_capacity(seq.len());
                for fw in seq {
                    match fw {
                        serde_yaml::Value::String(s) => fws.push(s.clone()),
                        other => {
                            return Err(perf_err(format!(
                                "{}: manifest frameworks entries must be strings, got {}",
                                manifest_path.display(),
                                yaml_type_name(other)
                            )))
                        }
                    }
                }
                fws
            }
            Some(other) => {
                return Err(perf_err(format!(
                    "{}: manifest 'frameworks' must be a list, got {}",
                    manifest_path.display(),
                    yaml_type_name(other)
                )))
            }
        };
        matched.push(ManifestEntry {
            kernel_source,
            tier,
            frameworks,
        });
    }
    Ok(matched)
}

// ---------------------------------------------------------------------------
// Version ordering (PEP 440 via pep440_rs; parity with packaging.version)
// ---------------------------------------------------------------------------

fn parse_pep440(version: &str) -> Option<Version> {
    Version::from_str(version).ok()
}

/// Sort key for newest-first ordering: parseable PEP 440 versions form one
/// group and always rank above unparseable strings.
#[derive(PartialEq, Eq)]
enum NewestFirstKey {
    Raw(String),
    Parsed(Version),
}

impl Ord for NewestFirstKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use NewestFirstKey::{Parsed, Raw};
        match (self, other) {
            (Parsed(a), Parsed(b)) => a.cmp(b),
            (Raw(a), Raw(b)) => a.cmp(b),
            (Parsed(_), Raw(_)) => std::cmp::Ordering::Greater,
            (Raw(_), Parsed(_)) => std::cmp::Ordering::Less,
        }
    }
}

impl PartialOrd for NewestFirstKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ---------------------------------------------------------------------------
// The resolver core
// ---------------------------------------------------------------------------

/// Load identity + policy for live source resolution.
#[derive(Clone, Debug)]
pub struct ResolveCtx {
    /// `<systems_root>` (holds `perf_data_reuse_manifest.yaml`).
    pub systems_root: PathBuf,
    /// `<systems_root>/<data_dir>` for the system.
    pub system_data_root: PathBuf,
    pub backend: String,
    pub version: String,
    /// SILICON/HYBRID sibling inheritance on/off (Python
    /// `database.enable_shared_layer`, which callers may explicitly
    /// override away from the mode-derived default).
    pub enable_shared_layer: bool,
    /// Fail-closed provenance mode (`AIC_STRICT_PROVENANCE`): malformed
    /// sidecars error instead of warn-and-continue.
    pub strict: bool,
}

/// Resolve the full source report for ONE op-file basename. `primary_override`
/// preserves the retired Python signature (callers passed the resolved
/// primary path); `None` resolves it via [`resolve_op_data_path`].
pub fn resolve_one(
    ctx: &ResolveCtx,
    op_file_basename: &str,
    primary_override: Option<&Path>,
) -> Result<ResolveReport, AicError> {
    let backend_lower = ctx.backend.to_lowercase();
    let mut warnings: Vec<ResolverWarning> = Vec::new();
    // (version, path, channel, ks_filter) accumulator; `exists` is computed
    // at finish time, mirroring `_finish()`.
    let mut records: Vec<(String, PathBuf, &'static str, Option<BTreeSet<String>>)> = Vec::new();

    let primary_path = match primary_override {
        Some(p) => p.to_path_buf(),
        None => resolve_op_data_path(
            &ctx.system_data_root,
            &ctx.backend,
            &ctx.version,
            op_file_basename,
        ),
    };
    let primary_version_dir = primary_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default();
    let primary_is_file = primary_path.is_file();
    let primary_unusable = primary_is_file
        && version_dir_unusable_for_request(
            &primary_version_dir,
            &ctx.system_data_root,
            ctx.strict,
            &mut warnings,
        )?;
    if primary_is_file
        && !primary_unusable
        && op_file_family_from_path(&primary_path, &ctx.system_data_root).is_some()
    {
        check_strict_provenance_coverage(&primary_version_dir, ctx.strict, None, &mut warnings)?;
    }
    if primary_unusable {
        // Only the unstructured legacy marker is a whole-directory veto.
        warnings.push(warn(
            "primary_veto",
            vec![
                primary_path.display().to_string(),
                op_file_basename.to_string(),
                primary_version_dir.display().to_string(),
            ],
        ));
    } else {
        records.push((ctx.version.clone(), primary_path.clone(), "primary", None));
    }

    let finish = |records: Vec<(String, PathBuf, &'static str, Option<BTreeSet<String>>)>,
                  warnings: Vec<ResolverWarning>| {
        ResolveReport {
            records: records
                .into_iter()
                .map(|(version, path, channel, ks_filter)| ResolvedRecord {
                    version,
                    exists: path.is_file(),
                    path: path.display().to_string(),
                    channel,
                    ks_filter,
                })
                .collect(),
            warnings,
        }
    };

    if !ctx.enable_shared_layer
        || FRAMEWORK_AGNOSTIC_BASENAMES.contains(&op_file_basename)
        || op_file_family_from_path(&primary_path, &ctx.system_data_root).as_deref()
            == Some(COMM_FAMILY_DIR)
    {
        return Ok(finish(records, warnings));
    }

    // Channel 2 (design §6.3): declared donors, in file order, deduped on
    // from_version (first occurrence wins).
    let mut declared_donor_versions: BTreeSet<String> = BTreeSet::new();
    for reuse_entry in requested_version_reuse_entries(
        &ctx.system_data_root,
        &backend_lower,
        &ctx.version,
        op_file_basename,
        ctx.strict,
        &mut warnings,
    )? {
        if declared_donor_versions.contains(&reuse_entry.from_version) {
            warnings.push(warn(
                "duplicate_declared",
                vec![
                    reuse_entry.table.clone(),
                    reuse_entry.from_version.clone(),
                    ctx.system_data_root.display().to_string(),
                ],
            ));
            continue;
        }
        let donor_path = resolve_op_data_path(
            &ctx.system_data_root,
            &backend_lower,
            &reuse_entry.from_version,
            op_file_basename,
        );
        if !donor_path.is_file() {
            continue;
        }
        let donor_dir = donor_path.parent().map(Path::to_path_buf).unwrap_or_default();
        if version_dir_unusable_for_request(&donor_dir, &ctx.system_data_root, ctx.strict, &mut warnings)? {
            continue;
        }
        if op_file_family_from_path(&donor_path, &ctx.system_data_root).is_some() {
            check_strict_provenance_coverage(
                &donor_dir,
                ctx.strict,
                Some(&reuse_entry.table),
                &mut warnings,
            )?;
        }
        records.push((
            reuse_entry.from_version.clone(),
            donor_path,
            "declared_reuse",
            None,
        ));
        declared_donor_versions.insert(reuse_entry.from_version);
    }

    // Channel 3 (design §6.2): nearest-earlier same-backend fallback.
    // Unparseable sibling versions are excluded (warned once, Python-side
    // registry); declared donors are excluded to avoid double-listing.
    if let Some(requested_parsed) = parse_pep440(&ctx.version) {
        let mut sibling_versions: BTreeSet<String> =
            iter_backend_version_dirs(&ctx.system_data_root, &backend_lower, &mut warnings)
                .into_iter()
                .map(|(v, _)| v)
                .collect();
        sibling_versions.remove(&ctx.version);
        for donor in &declared_donor_versions {
            sibling_versions.remove(donor);
        }
        let mut earlier_versions: Vec<(Version, String)> = Vec::new();
        for sibling_version in sibling_versions {
            let Some(parsed) = parse_pep440(&sibling_version) else {
                warnings.push(warn(
                    "unparseable_sibling",
                    vec![
                        ctx.system_data_root.display().to_string(),
                        backend_lower.clone(),
                        sibling_version.clone(),
                    ],
                ));
                continue;
            };
            if parsed >= requested_parsed {
                continue; // Never admit newer-than-requested implicitly.
            }
            earlier_versions.push((parsed, sibling_version));
        }
        earlier_versions.sort_by(|a, b| b.0.cmp(&a.0)); // nearest-earlier first
        for (_, sibling_version) in earlier_versions {
            let sibling_path = resolve_op_data_path(
                &ctx.system_data_root,
                &backend_lower,
                &sibling_version,
                op_file_basename,
            );
            if !sibling_path.is_file() {
                continue;
            }
            let sibling_dir = sibling_path.parent().map(Path::to_path_buf).unwrap_or_default();
            if version_dir_unusable_for_request(&sibling_dir, &ctx.system_data_root, ctx.strict, &mut warnings)? {
                continue;
            }
            records.push((sibling_version, sibling_path, "fallback", None));
        }
    }

    // Channel 4 (design §6.4): cross-backend fill, kernel-identity gated.
    let mut per_framework_filter: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut per_framework_fallback: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for entry in manifest_entries_for(&ctx.systems_root, op_file_basename)? {
        let frameworks_lower: BTreeSet<String> =
            entry.frameworks.iter().map(|fw| fw.to_lowercase()).collect();
        if !frameworks_lower.contains(&backend_lower) {
            continue; // Active backend isn't listed as a consumer of this kernel_source.
        }
        let Some(ks) = entry.kernel_source else {
            continue;
        };
        let tier = entry.tier.as_deref();
        if tier == Some("shared") || tier == Some("shared_fallback") {
            for fw in &frameworks_lower {
                per_framework_filter
                    .entry(fw.clone())
                    .or_default()
                    .insert(ks.clone());
            }
            if tier == Some("shared_fallback") {
                for fw in &frameworks_lower {
                    per_framework_fallback
                        .entry(fw.clone())
                        .or_default()
                        .insert(ks.clone());
                }
            }
        }
    }

    // sorted(set(per_framework_filter) - {backend_lower})
    let ordered_frameworks: Vec<String> = per_framework_filter
        .keys()
        .filter(|fw| **fw != backend_lower)
        .cloned()
        .collect(); // BTreeMap keys iterate sorted

    for framework in ordered_frameworks {
        let ks_filter = per_framework_filter[&framework].clone();
        let fallback_only = per_framework_fallback
            .get(&framework)
            .cloned()
            .unwrap_or_default();
        let mut fw_versions: Vec<String> =
            iter_backend_version_dirs(&ctx.system_data_root, &framework, &mut warnings)
                .into_iter()
                .map(|(v, _)| v)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
        fw_versions.sort_by(|a, b| {
            let key = |v: &String| match parse_pep440(v) {
                Some(parsed) => NewestFirstKey::Parsed(parsed),
                None => NewestFirstKey::Raw(v.clone()),
            };
            key(b).cmp(&key(a)) // newest first
        });
        for sibling_version in fw_versions {
            let sibling_path = resolve_op_data_path(
                &ctx.system_data_root,
                &framework,
                &sibling_version,
                op_file_basename,
            );
            if !sibling_path.is_file() {
                continue;
            }
            let sibling_dir = sibling_path.parent().map(Path::to_path_buf).unwrap_or_default();
            if version_dir_unusable_for_request(&sibling_dir, &ctx.system_data_root, ctx.strict, &mut warnings)? {
                continue;
            }
            if !fallback_only.is_disjoint(&ks_filter) {
                // Low-fidelity framework-implicit rows; deduped per
                // (op file, sibling source) per database on the Python side.
                warnings.push(warn(
                    "low_fidelity",
                    vec![
                        op_file_basename.to_string(),
                        sibling_path.display().to_string(),
                    ],
                ));
            }
            records.push((
                sibling_version,
                sibling_path,
                "cross_backend",
                Some(ks_filter.clone()),
            ));
        }
    }

    Ok(finish(records, warnings))
}

// ---------------------------------------------------------------------------
// SourceResolver: the handle every table resolves through
// ---------------------------------------------------------------------------

enum ResolverKind {
    /// Engine-owned live resolution over the perf-data tree.
    Live(ResolveCtx),
    /// A pre-materialized source map (tests / synthetic injections). Keeps
    /// the retired wire semantics: a PRESENT-but-EMPTY list is a deliberate
    /// veto; an ABSENT basename falls back to the single primary with the
    /// family-first walk.
    Fixed(PerfDbSources),
}

/// Per-load source resolution handle. Tables call [`Self::sources_for`] at
/// construction; the table view resolves ad-hoc basenames through the same
/// handle. Live results are memoized per basename (resolution walks
/// directories and parses sidecar yaml).
pub struct SourceResolver {
    kind: ResolverKind,
    cache: Mutex<BTreeMap<String, Vec<PerfSource>>>,
}

impl SourceResolver {
    pub fn live(ctx: ResolveCtx) -> Self {
        SourceResolver {
            kind: ResolverKind::Live(ctx),
            cache: Mutex::new(BTreeMap::new()),
        }
    }

    pub fn fixed(perf_db_sources: PerfDbSources) -> Self {
        SourceResolver {
            kind: ResolverKind::Fixed(perf_db_sources),
            cache: Mutex::new(BTreeMap::new()),
        }
    }

    /// The ordered source list for one op-file basename. `data_root` is the
    /// legacy `<data>/<backend>/<version>` dir used for the Fixed-map
    /// default-primary fallback (identical to the retired
    /// `resolve_op_sources`).
    pub fn sources_for(&self, basename: &str, data_root: &Path) -> Result<Vec<PerfSource>, AicError> {
        match &self.kind {
            ResolverKind::Fixed(map) => Ok(match map.get(basename) {
                Some(sources) if !sources.is_empty() => sources.clone(),
                // A PRESENT but EMPTY list is a deliberate veto statement:
                // load NO sources. Falling back to the primary here would
                // silently undo the veto.
                Some(_) => Vec::new(),
                None => {
                    let legacy = data_root.join(basename);
                    let path = if legacy.is_file() {
                        legacy
                    } else {
                        find_in_family_dirs(data_root, basename).unwrap_or(legacy)
                    };
                    vec![PerfSource(path, None)]
                }
            }),
            ResolverKind::Live(ctx) => {
                if let Some(cached) = self.cache.lock().unwrap().get(basename) {
                    return Ok(cached.clone());
                }
                let report = resolve_one(ctx, basename, None)?;
                let sources = report.sources();
                self.cache
                    .lock()
                    .unwrap()
                    .insert(basename.to_string(), sources.clone());
                Ok(sources)
            }
        }
    }

    /// Stable identity string for the shared-tables memo. Live resolution is
    /// a pure function of the load identity + policy, so those fields ARE
    /// the identity; a Fixed map renders its full contents (the retired
    /// `shared_tables_key` body).
    pub fn identity_key(&self) -> String {
        match &self.kind {
            ResolverKind::Live(ctx) => format!(
                "live\x1f{}\x1f{}\x1f{}\x1f{}\x1fshared={}\x1fstrict={}",
                ctx.systems_root.display(),
                ctx.system_data_root.display(),
                ctx.backend,
                ctx.version,
                ctx.enable_shared_layer,
                ctx.strict,
            ),
            ResolverKind::Fixed(map) => {
                let mut key = String::from("fixed");
                for (basename, sources) in map {
                    let _ = write!(key, "\x1f{basename}=");
                    for PerfSource(path, kernel_sources) in sources {
                        let _ = write!(key, "{}|{kernel_sources:?};", path.display());
                    }
                }
                key
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, content: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    fn ctx(root: &Path, backend: &str, version: &str) -> ResolveCtx {
        ResolveCtx {
            systems_root: root.to_path_buf(),
            system_data_root: root.join("data"),
            backend: backend.to_string(),
            version: version.to_string(),
            enable_shared_layer: true,
            strict: false,
        }
    }

    fn channels(report: &ResolveReport) -> Vec<&'static str> {
        report.records.iter().map(|r| r.channel).collect()
    }

    fn versions(report: &ResolveReport) -> Vec<&str> {
        report.records.iter().map(|r| r.version.as_str()).collect()
    }

    #[test]
    fn full_channel_order_declared_then_fallback_nearest_then_cross_backend() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let data = root.join("data");
        for v in ["1.0.0", "1.2.0", "0.9.0", "0.5.0"] {
            write(&data.join(format!("gemm/trtllm/{v}/gemm_perf.parquet")), "stub");
        }
        write(&data.join("gemm/vllm/0.5.0/gemm_perf.parquet"), "stub");
        write(
            &data.join("gemm/trtllm/1.0.0/reuse.yaml"),
            "schema_version: 1\nreuse:\n  - table: gemm_perf\n    from_version: '1.2.0'\n    reason: r\n    approved_by: a\n",
        );
        write(
            &root.join("perf_data_reuse_manifest.yaml"),
            "groups:\n  - op_file: gemm_perf.parquet\n    kernel_source: shared_kernel\n    tier: shared\n    frameworks: [trtllm, vllm]\n",
        );
        let report = resolve_one(&ctx(root, "trtllm", "1.0.0"), "gemm_perf.parquet", None).unwrap();
        assert_eq!(
            channels(&report),
            ["primary", "declared_reuse", "fallback", "fallback", "cross_backend"]
        );
        assert_eq!(versions(&report), ["1.0.0", "1.2.0", "0.9.0", "0.5.0", "0.5.0"]);
        let last = report.records.last().unwrap();
        let ks: Vec<&str> = last.ks_filter.as_ref().unwrap().iter().map(String::as_str).collect();
        assert_eq!(ks, ["shared_kernel"]);
        assert!(report.records[..4].iter().all(|r| r.ks_filter.is_none()));
        assert!(report.records.iter().all(|r| r.exists));
    }

    #[test]
    fn legacy_incomplete_primary_vetoed_donors_still_fill() {
        // Mirrors test_reuse_ordering.py::
        // test_legacy_incomplete_primary_not_admitted_donors_still_fill —
        // the vetoed primary sits in the LEGACY (backend-first) layout, so
        // the family walk can't route around it.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let data = root.join("data");
        write(&data.join("trtllm/1.0.0/gemm_perf.parquet"), "stub");
        write(&data.join("trtllm/1.0.0/INCOMPLETE.txt"), "partial collection\n");
        write(&data.join("gemm/trtllm/0.9.0/gemm_perf.parquet"), "stub");
        let report = resolve_one(&ctx(root, "trtllm", "1.0.0"), "gemm_perf.parquet", None).unwrap();
        assert_eq!(channels(&report), ["fallback"]);
        assert_eq!(versions(&report), ["0.9.0"]);
        assert!(report.warnings.iter().any(|w| w.kind == "primary_veto"));
        // structured sidecar supersedes the stale legacy marker
        write(
            &data.join("trtllm/1.0.0/collection_meta.yaml"),
            "schema_version: 1\ntables:\n  gemm_perf: {status: partial}\n",
        );
        let report = resolve_one(&ctx(root, "trtllm", "1.0.0"), "gemm_perf.parquet", None).unwrap();
        assert_eq!(channels(&report), ["primary", "fallback"]);
    }

    #[test]
    fn incomplete_family_dir_is_skipped_by_the_resolver_walk() {
        // Mirrors test_reuse_ordering.py::
        // test_legacy_incomplete_family_dir_is_skipped_by_resolver — a vetoed
        // FAMILY dir is routed around at path-resolution time, so the primary
        // record falls to the (nonexistent) legacy path with exists=false.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let data = root.join("data");
        write(&data.join("gemm/trtllm/1.0.0/gemm_perf.parquet"), "stub");
        write(&data.join("gemm/trtllm/1.0.0/INCOMPLETE.txt"), "");
        write(&data.join("gemm/trtllm/0.9.0/gemm_perf.parquet"), "stub");
        let report = resolve_one(&ctx(root, "trtllm", "1.0.0"), "gemm_perf.parquet", None).unwrap();
        assert_eq!(channels(&report), ["primary", "fallback"]);
        assert!(!report.records[0].exists);
        assert!(!report.records[0].path.contains("gemm/trtllm/1.0.0"));
        assert_eq!(versions(&report), ["1.0.0", "0.9.0"]);
    }

    #[test]
    fn unparseable_sibling_excluded_and_newer_never_implicit() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let data = root.join("data");
        for v in ["1.2.0rc5", "1.2.0", "nightly-build", "1.2.0rc4"] {
            write(&data.join(format!("gemm/trtllm/{v}/gemm_perf.parquet")), "stub");
        }
        let report =
            resolve_one(&ctx(root, "trtllm", "1.2.0rc5"), "gemm_perf.parquet", None).unwrap();
        // 1.2.0 > 1.2.0rc5 (never implicit); nightly-build unparseable (warned).
        assert_eq!(versions(&report), ["1.2.0rc5", "1.2.0rc4"]);
        assert!(report.warnings.iter().any(|w| w.kind == "unparseable_sibling"));
    }

    #[test]
    fn strict_malformed_reuse_yaml_raises_with_python_message() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let data = root.join("data");
        write(&data.join("gemm/trtllm/1.0.0/gemm_perf.parquet"), "stub");
        write(
            &data.join("gemm/trtllm/1.0.0/collection_meta.yaml"),
            "schema_version: 1\ntables:\n  gemm_perf: {status: complete}\n",
        );
        write(
            &data.join("gemm/trtllm/1.0.0/reuse.yaml"),
            "reuse:\n  - table: gemm_perf\n",
        );
        let mut c = ctx(root, "trtllm", "1.0.0");
        c.strict = true;
        let err = resolve_one(&c, "gemm_perf.parquet", None).unwrap_err();
        assert!(
            err.to_string().contains("missing required key(s): from_version, reason, approved_by"),
            "{err}"
        );
        // non-strict: warn and keep resolving with zero declared donors.
        c.strict = false;
        let report = resolve_one(&c, "gemm_perf.parquet", None).unwrap();
        assert_eq!(channels(&report), ["primary"]);
        assert!(report
            .warnings
            .iter()
            .any(|w| w.kind == "malformed_sidecar" || w.kind == "malformed_reuse"));
    }

    #[test]
    fn strict_primary_missing_sidecar_fails_closed() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write(&root.join("data/gemm/trtllm/1.0.0/gemm_perf.parquet"), "stub");
        let mut c = ctx(root, "trtllm", "1.0.0");
        c.strict = true;

        let err = resolve_one(&c, "gemm_perf.parquet", None).unwrap_err();
        assert!(err.to_string().contains("no collection_meta.yaml"), "{err}");
    }

    #[test]
    fn strict_primary_rejects_any_uncovered_table_in_version_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let version_dir = root.join("data/gemm/trtllm/1.0.0");
        write(&version_dir.join("gemm_perf.parquet"), "stub");
        write(&version_dir.join("uncovered_perf.parquet"), "stub");
        write(
            &version_dir.join("collection_meta.yaml"),
            "schema_version: 1\ntables:\n  gemm_perf: {status: complete}\n",
        );
        let mut c = ctx(root, "trtllm", "1.0.0");
        c.strict = true;

        let err = resolve_one(&c, "gemm_perf.parquet", None).unwrap_err();
        assert!(err.to_string().contains("uncovered_perf"), "{err}");
    }

    #[test]
    fn strict_declared_donor_requires_coverage_for_admitted_table() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let requested = root.join("data/moe/trtllm/1.0.0");
        let donor = root.join("data/moe/trtllm/0.9.0");
        write(&requested.join("moe_perf.parquet"), "stub");
        write(
            &requested.join("collection_meta.yaml"),
            "schema_version: 1\ntables:\n  moe_perf: {status: complete}\n",
        );
        write(
            &requested.join("reuse.yaml"),
            "schema_version: 1\nreuse:\n  - table: wideep_moe_perf\n    from_version: '0.9.0'\n    reason: r\n    approved_by: a\n",
        );
        write(&donor.join("wideep_moe_perf.parquet"), "stub");
        write(&donor.join("collection_meta.yaml"), "schema_version: 1\ntables: {}\n");
        let mut c = ctx(root, "trtllm", "1.0.0");
        c.strict = true;

        let err = resolve_one(&c, "wideep_moe_perf.parquet", None).unwrap_err();
        assert!(err.to_string().contains("wideep_moe_perf"), "{err}");
    }

    #[test]
    fn strict_legacy_provenance_graces_uncovered_table() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let version_dir = root.join("data/gemm/trtllm/1.0.0");
        write(&version_dir.join("gemm_perf.parquet"), "stub");
        write(
            &version_dir.join("collection_meta.yaml"),
            "schema_version: 1\nprovenance: legacy\ntables: {}\n",
        );
        let mut c = ctx(root, "trtllm", "1.0.0");
        c.strict = true;

        let report = resolve_one(&c, "gemm_perf.parquet", None).unwrap();
        assert_eq!(channels(&report), ["primary"]);
        assert!(report.warnings.iter().any(|warning| {
            warning.kind == "strict_provenance" && warning.args[0] == "legacy-uncovered"
        }));
    }

    #[test]
    fn comm_family_and_framework_agnostic_early_returns() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let data = root.join("data");
        write(&data.join("comm/trtllm/1.0.0/custom_allreduce_perf.parquet"), "stub");
        write(&data.join("comm/trtllm/0.9.0/custom_allreduce_perf.parquet"), "stub");
        let report = resolve_one(
            &ctx(root, "trtllm", "1.0.0"),
            "custom_allreduce_perf.parquet",
            None,
        )
        .unwrap();
        assert_eq!(channels(&report), ["primary"]);
        // op-name early exit, independent of the family path
        let report = resolve_one(&ctx(root, "trtllm", "1.0.0"), "nccl_perf.parquet", None).unwrap();
        assert_eq!(channels(&report), ["primary"]);
    }

    #[test]
    fn newest_first_key_ranks_parseable_above_raw() {
        let mut keys = vec![
            NewestFirstKey::Raw("zzz".to_string()),
            NewestFirstKey::Parsed(Version::from_str("1.2.0").unwrap()),
            NewestFirstKey::Parsed(Version::from_str("1.10.0").unwrap()),
            NewestFirstKey::Raw("aaa".to_string()),
        ];
        keys.sort_by(|a, b| b.cmp(a));
        let rendered: Vec<String> = keys
            .iter()
            .map(|k| match k {
                NewestFirstKey::Parsed(v) => format!("v{v}"),
                NewestFirstKey::Raw(s) => format!("r{s}"),
            })
            .collect();
        assert_eq!(rendered, ["v1.10.0", "v1.2.0", "rzzz", "raaa"]);
    }

    #[test]
    fn fixed_resolver_keeps_retired_map_semantics() {
        let tmp = tempfile::tempdir().unwrap();
        let data_root = tmp.path().join("trtllm").join("1.0.0");
        std::fs::create_dir_all(&data_root).unwrap();
        let mut map = PerfDbSources::default();
        map.insert("vetoed.parquet".to_string(), Vec::new());
        map.insert(
            "explicit.parquet".to_string(),
            vec![PerfSource(PathBuf::from("/x/explicit.parquet"), None)],
        );
        let resolver = SourceResolver::fixed(map);
        // present-but-empty = deliberate veto
        assert!(resolver.sources_for("vetoed.parquet", &data_root).unwrap().is_empty());
        // explicit list passes through
        assert_eq!(
            resolver.sources_for("explicit.parquet", &data_root).unwrap()[0].0,
            PathBuf::from("/x/explicit.parquet")
        );
        // absent = default primary (legacy path when no family dir has it)
        let sources = resolver.sources_for("absent.parquet", &data_root).unwrap();
        assert_eq!(sources[0].0, data_root.join("absent.parquet"));
    }
}
