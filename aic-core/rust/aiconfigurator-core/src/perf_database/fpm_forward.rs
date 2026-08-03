// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Whole-model `fpm_forward` perf tables (Python `forward_model="fpm"`).
//!
//! Rust port of the loader half of
//! `src/aiconfigurator_core/sdk/operations/fpm_forward.py`: the formal
//! collector pair
//!
//! ```text
//! <data_root>/fpm_forward_perf.parquet
//! <data_root>/fpm_forward_perf.metadata.json
//! ```
//!
//! is validated (sidecar schema/sha256/row_count, per-row workload checks,
//! duplicate physical row keys) and grouped into cells keyed by
//! `(model_path, backend_axis, backend_policy, 11 identity columns)`. Each
//! cell holds one nested table per phase — prefill
//! `[batch][total_prefill][total_kv]`, decode `[batch][total_kv]` — plus the
//! per-phase axis-aligned domain bounding box and a prebuilt
//! [`SiteIndex`](super::perf_interp::SiteIndex).
//!
//! Contract notes, all mirrored from Python:
//! - An ABSENT parquet is the soft "not collected" case: `cells()` errors only
//!   when queried, like `LoadedOpData.raise_if_not_loaded`.
//! - Every structural violation of an EXISTING pair is a loud error (Python's
//!   uniform ValueError contract) — a corrupt supported-database entry is a
//!   data bug, not a fallback condition.
//! - Deliberately NO shared-layer (sibling/cross-version) inheritance: FPM
//!   whole-model data is valid only for its exact backend/version, and the
//!   loader enforces per-row `backend_version == <version dir>`.
//!
//! NOT the crate's `src/fpm/` module: that is `ForwardPassPerfModel`, the
//! online-tuning model over Dynamo ForwardPassMetrics telemetry — an
//! unrelated concept that also abbreviates to "FPM".

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use sha2::{Digest, Sha256};

use super::parquet_loader::PerfReader;
use super::perf_interp::{Node, SiteIndex};
use crate::common::error::AicError;

pub const FPM_FORWARD_BASENAME: &str = "fpm_forward_perf.parquet";
pub const FPM_FORWARD_SCHEMA_NAME: &str = "aic_fpm_forward_perf";
pub const FPM_FORWARD_SCHEMA_VERSION: u64 = 5;
pub const FPM_FORWARD_COORDINATE_SYSTEM: &str = "iteration_totals_balanced_v1";
pub const FPM_FORWARD_PARTITION_POLICY: &str = "balanced_v1";
pub const FPM_FORWARD_SUPPORTED_BACKEND_AXIS: &str = "baseline";

/// Identity columns that select a cell, in row-column order (`model_path` is
/// handled separately; `weight_quantization` is deliberately excluded).
pub const FPM_CELL_MATCH_COLUMNS: [&str; 11] = [
    "gemm_quant_mode",
    "moe_quant_mode",
    "fmha_quant_mode",
    "comm_quant_mode",
    "kv_cache_dtype",
    "tp",
    "pp",
    "dp",
    "moe_tp",
    "moe_ep",
    "cp",
];

pub const FPM_PREFILL_AXES: [&str; 3] =
    ["batch_size", "total_prefill_tokens", "total_kv_read_tokens"];
pub const FPM_DECODE_AXES: [&str; 2] = ["batch_size", "total_kv_read_tokens"];

/// One collected cell: the tables and domains for a single
/// `(model_path, backend_axis, backend_policy, identity)` tuple.
#[derive(Debug)]
pub struct FpmForwardCell {
    pub model_path: String,
    pub backend_axis: String,
    pub backend_policy: String,
    pub match_identity: Vec<String>,
    pub cell_ids: Vec<String>,
    pub prefill: Node,
    pub decode: Node,
    /// Per-axis `(min, max)` over ALL collected prefill points; `None` when
    /// the cell has no prefill rows. Axis order = [`FPM_PREFILL_AXES`].
    pub prefill_domain: Option<[(u32, u32); 3]>,
    pub decode_domain: Option<[(u32, u32); 2]>,
    /// Prebuilt scattered-sites indexes (tables are immutable after load).
    pub prefill_index: Option<SiteIndex>,
    pub decode_index: Option<SiteIndex>,
}

pub struct FpmForwardTable {
    parquet_path: PathBuf,
    system: String,
    backend: String,
    version: String,
    /// `Ok(None)` = parquet absent ("not collected"); errors are structural.
    cells: OnceLock<Result<Option<Vec<FpmForwardCell>>, AicError>>,
}

fn structural(msg: String) -> AicError {
    AicError::PerfDatabase(msg)
}

/// `AicError` is not `Clone`; re-surface `OnceLock`-cached errors as
/// `PerfDatabase` (same pattern as every other table in this module).
fn clone_err(err: &AicError) -> AicError {
    AicError::PerfDatabase(err.to_string())
}

impl FpmForwardTable {
    /// No I/O. `system`/`backend`/`version` are the resolved database
    /// identity, enforced against every row (a misplaced pair — e.g. an h200
    /// parquet copied into a b200 tree — must fail loudly, not merge).
    pub fn new(data_root: PathBuf, system: &str, backend: &str, version: &str) -> Self {
        Self {
            parquet_path: data_root.join(FPM_FORWARD_BASENAME),
            system: system.to_string(),
            backend: backend.to_string(),
            version: version.to_string(),
            cells: OnceLock::new(),
        }
    }

    pub fn parquet_path(&self) -> &Path {
        &self.parquet_path
    }

    /// The loaded cells. Errors when the parquet is absent (with the exact
    /// path, mirroring `LoadedOpData.raise_if_not_loaded`) or when the pair
    /// is structurally invalid.
    pub fn cells(&self) -> Result<&[FpmForwardCell], AicError> {
        let loaded = self
            .cells
            .get_or_init(|| load_pair(&self.parquet_path, &self.system, &self.backend, &self.version));
        match loaded {
            Ok(Some(cells)) => Ok(cells),
            Ok(None) => Err(AicError::PerfDatabase(format!(
                "File does not exist at {}. No fpm_forward data collected for this backend/version.",
                self.parquet_path.display()
            ))),
            Err(err) => Err(clone_err(err)),
        }
    }

    /// Cell selection, mirroring Python `FPMForwardOp._select_cell` exactly:
    /// identity + `backend_axis == "baseline"` filter, exact-`model_path`
    /// preference, then the two ambiguity errors in order.
    pub fn select_cell(
        &self,
        match_identity: &[String],
        model_path: &str,
    ) -> Result<&FpmForwardCell, AicError> {
        let cells = self.cells()?;
        let matches: Vec<&FpmForwardCell> = cells
            .iter()
            .filter(|cell| {
                cell.match_identity == match_identity
                    && cell.backend_axis == FPM_FORWARD_SUPPORTED_BACKEND_AXIS
            })
            .collect();
        if matches.is_empty() {
            let mut available: Vec<String> = cells
                .iter()
                .map(|cell| format!("({:?}, {:?})", cell.model_path, cell.match_identity))
                .collect();
            available.sort();
            available.dedup();
            available.truncate(8);
            let identity: Vec<String> = FPM_CELL_MATCH_COLUMNS
                .iter()
                .zip(match_identity)
                .map(|(c, v)| format!("{c}={v:?}"))
                .collect();
            return Err(structural(format!(
                "No FPM cell matches this model identity: {{{}}}. \
                 Collected cell identities (model_path first): [{}]",
                identity.join(", "),
                available.join(", ")
            )));
        }
        let exact: Vec<&FpmForwardCell> = matches
            .iter()
            .copied()
            .filter(|cell| cell.model_path == model_path)
            .collect();
        let matches = if exact.is_empty() { matches } else { exact };
        let mut distinct_paths: Vec<&str> =
            matches.iter().map(|cell| cell.model_path.as_str()).collect();
        distinct_paths.sort_unstable();
        distinct_paths.dedup();
        if distinct_paths.len() > 1 {
            return Err(structural(format!(
                "Ambiguous FPM cell selection: model_path={model_path:?} matched none exactly and \
                 multiple collected model paths share this identity: {distinct_paths:?}. \
                 Collect/promote data under the exact model path, or query with the matching path."
            )));
        }
        if matches.len() > 1 {
            let mut policies: Vec<&str> = matches
                .iter()
                .map(|cell| cell.backend_policy.as_str())
                .collect();
            policies.sort_unstable();
            return Err(structural(format!(
                "Ambiguous FPM cell selection: multiple backend policies for \
                 model_path={:?}: {policies:?}",
                distinct_paths[0]
            )));
        }
        Ok(matches[0])
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

fn sha256_file(path: &Path) -> Result<String, AicError> {
    let mut file = std::fs::File::open(path).map_err(|source| AicError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).map_err(|source| AicError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// Sidecar validation, mirroring `_validate_sidecar` check-for-check. Returns
/// the sidecar's `row_count` for the post-read cross-check.
fn validate_sidecar(metadata_path: &Path, parquet_path: &Path) -> Result<Option<u64>, AicError> {
    if !metadata_path.exists() {
        return Err(structural(format!(
            "FPM database is missing its metadata sidecar: {}. \
             The parquet/metadata pair is atomic; refusing to load an unmatched parquet.",
            metadata_path.display()
        )));
    }
    let text = std::fs::read_to_string(metadata_path).map_err(|source| AicError::Io {
        path: metadata_path.to_path_buf(),
        source,
    })?;
    let metadata: serde_json::Value = serde_json::from_str(&text).map_err(|err| {
        structural(format!(
            "FPM metadata sidecar is not valid JSON: {} ({err})",
            metadata_path.display()
        ))
    })?;
    let Some(metadata) = metadata.as_object() else {
        return Err(structural(format!(
            "FPM metadata sidecar must be a JSON object: {}",
            metadata_path.display()
        )));
    };
    if metadata.get("schema_name").and_then(|v| v.as_str()) != Some(FPM_FORWARD_SCHEMA_NAME) {
        return Err(structural(format!(
            "unsupported FPM schema_name={:?} (expected {FPM_FORWARD_SCHEMA_NAME:?}): {}",
            metadata.get("schema_name"),
            metadata_path.display()
        )));
    }
    if json_uint(metadata.get("schema_version")) != Some(FPM_FORWARD_SCHEMA_VERSION) {
        return Err(structural(format!(
            "unsupported FPM schema_version={:?} (expected {FPM_FORWARD_SCHEMA_VERSION}): {}",
            metadata.get("schema_version"),
            metadata_path.display()
        )));
    }
    if metadata.get("coordinate_system").and_then(|v| v.as_str())
        != Some(FPM_FORWARD_COORDINATE_SYSTEM)
    {
        return Err(structural(format!(
            "unsupported FPM coordinate_system={:?} (expected {FPM_FORWARD_COORDINATE_SYSTEM:?}): {}",
            metadata.get("coordinate_system"),
            metadata_path.display()
        )));
    }
    let actual_sha = sha256_file(parquet_path)?;
    if metadata.get("parquet_sha256").and_then(|v| v.as_str()) != Some(actual_sha.as_str()) {
        return Err(structural(format!(
            "FPM parquet digest mismatch: sidecar={:?} actual={actual_sha:?}. The pair at {} is \
             inconsistent (interrupted writer?).",
            metadata.get("parquet_sha256"),
            parquet_path.parent().unwrap_or(parquet_path).display()
        )));
    }
    Ok(json_uint(metadata.get("row_count")))
}

/// Python `metadata.get(k) != n` compares by VALUE: a JSON `5.0` equals the
/// int 5. Accept integral floats the way Python does; anything else is None.
fn json_uint(value: Option<&serde_json::Value>) -> Option<u64> {
    let value = value?;
    if let Some(v) = value.as_u64() {
        return Some(v);
    }
    let f = value.as_f64()?;
    (f.fract() == 0.0 && f >= 0.0 && f <= u64::MAX as f64).then_some(f as u64)
}

/// One parsed row. String identity fields are pre-normalized (null -> "");
/// enum-name normalization happened on the collector/producer side, so the
/// stored strings are compared verbatim.
struct FpmRow {
    cell_id: String,
    model_path: String,
    backend_axis: String,
    backend_policy: String,
    match_identity: Vec<String>,
    row_key: Vec<String>,
    workload_kind: String,
    batch_size: u32,
    total_prefill_tokens: u32,
    total_kv_read_tokens: u32,
    latency_ms: f64,
}

impl FpmRow {
    /// The logical cell key: which table slot this row lands in. The
    /// coordinate-collision check and the grouping loop MUST use the same
    /// key, or a colliding row could slip past the check and silently
    /// last-win in the grouped cell — hence one shared constructor.
    fn cell_key(&self) -> Vec<String> {
        let mut key = Vec::with_capacity(3 + self.match_identity.len());
        key.push(self.model_path.clone());
        key.push(self.backend_axis.clone());
        key.push(self.backend_policy.clone());
        key.extend(self.match_identity.iter().cloned());
        key
    }
}

fn load_pair(
    parquet_path: &Path,
    system: &str,
    backend: &str,
    version: &str,
) -> Result<Option<Vec<FpmForwardCell>>, AicError> {
    if !parquet_path.exists() {
        return Ok(None);
    }
    let metadata_path = parquet_path.with_extension("metadata.json");
    let sidecar_row_count = validate_sidecar(&metadata_path, parquet_path)?;

    let reader = PerfReader::open(parquet_path)?;
    // Physical row-key columns (collector contract), in order.
    let str_cols = [
        "cell_id",
        "model_path",
        "system",
        "backend",
        "backend_version",
        "weight_quantization",
        "gemm_quant_mode",
        "moe_quant_mode",
        "fmha_quant_mode",
        "comm_quant_mode",
        "kv_cache_dtype",
        "backend_axis",
        "backend_policy",
        "workload_kind",
        "partition_policy",
    ];
    let mut str_idx = BTreeMap::new();
    for name in str_cols {
        str_idx.insert(name, reader.col(name)?);
    }
    let int_cols = [
        "tp",
        "pp",
        "dp",
        "moe_tp",
        "moe_ep",
        "cp",
        "batch_size",
        "total_prefill_tokens",
        "total_kv_read_tokens",
    ];
    let mut int_idx = BTreeMap::new();
    for name in int_cols {
        int_idx.insert(name, reader.col(name)?);
    }
    let latency_col = reader.col("latency_ms")?;

    // Python checks the sidecar row_count against the FULL row list before any
    // per-row validation (`load_fpm_forward_data`: read_table -> row_count ->
    // empty -> per-row loop); mirror that error precedence with a cheap count
    // pass so a wrong-count pair reports the count, not the first bad row.
    let actual_row_count = reader.rows()?.count() as u64;
    match sidecar_row_count {
        Some(expected) if expected == actual_row_count => {}
        other => {
            return Err(structural(format!(
                "FPM row_count mismatch: sidecar={} actual={actual_row_count}: {}",
                other.map_or("None".to_string(), |v| v.to_string()),
                parquet_path.display()
            )));
        }
    }
    if actual_row_count == 0 {
        return Err(structural(format!(
            "FPM database contains no rows: {}",
            parquet_path.display()
        )));
    }

    let mut rows: Vec<FpmRow> = Vec::new();
    for (index, row) in reader.rows()?.enumerate() {
        let row = row?;
        let get_str = |name: &str| -> Result<String, AicError> {
            // Null identity cells normalize to "" (Python _norm_identity).
            Ok(row.str_optional(Some(str_idx[name]))?.unwrap_or("").to_string())
        };
        // Workload coordinates: required, loud on null/overflow.
        let get_int = |name: &str| -> Result<u32, AicError> { row.u32(int_idx[name]) };
        // Identity ints (tp/pp/dp/moe_tp/moe_ep/cp) are only ever COMPARED as
        // strings; Python `_norm_identity(row.get(col))` maps a null cell to
        // "" and the row simply becomes an unmatchable cell — it must not
        // fail the whole load.
        let get_int_identity = |name: &str| -> Result<String, AicError> {
            Ok(row
                .u32_optional(Some(int_idx[name]))?
                .map_or_else(String::new, |v| v.to_string()))
        };

        let workload_kind = get_str("workload_kind")?;
        if workload_kind != "prefill" && workload_kind != "decode" {
            return Err(structural(format!(
                "FPM row {index} has unknown workload_kind={workload_kind:?}"
            )));
        }
        let partition_policy = get_str("partition_policy")?;
        if partition_policy != FPM_FORWARD_PARTITION_POLICY {
            return Err(structural(format!(
                "FPM row {index} has unsupported partition_policy={partition_policy:?} \
                 (expected {FPM_FORWARD_PARTITION_POLICY:?})"
            )));
        }
        let backend_version = get_str("backend_version")?;
        if backend_version != version {
            return Err(structural(format!(
                "FPM row {index} backend_version={backend_version:?} does not match the \
                 database version directory {version:?}"
            )));
        }
        // `system`/`backend` are part of the physical row key but NOT the
        // cell key: a misplaced pair would merge into the same cells and
        // silently serve wrong latencies. Pin them like the version.
        let row_system = get_str("system")?;
        if row_system != system {
            return Err(structural(format!(
                "FPM row {index} system={row_system:?} does not match the database \
                 system {system:?}"
            )));
        }
        let row_backend = get_str("backend")?;
        if row_backend != backend {
            return Err(structural(format!(
                "FPM row {index} backend={row_backend:?} does not match the database \
                 backend {backend:?}"
            )));
        }
        let latency_ms = row.f64(latency_col)?;
        if !latency_ms.is_finite() || latency_ms <= 0.0 {
            return Err(structural(format!(
                "FPM row {index} has non-finite/non-positive latency_ms={latency_ms:?}"
            )));
        }
        // Coordinates: `PerfRow::u32` already rejects negatives loudly, which
        // subsumes Python's `batch >= 1 / totals >= 0` sign checks.
        let batch_size = get_int("batch_size")?;
        let total_prefill_tokens = get_int("total_prefill_tokens")?;
        let total_kv_read_tokens = get_int("total_kv_read_tokens")?;
        if batch_size < 1 {
            return Err(structural(format!(
                "FPM row {index} has invalid workload coordinates: batch_size={batch_size}, \
                 total_prefill_tokens={total_prefill_tokens}, \
                 total_kv_read_tokens={total_kv_read_tokens}"
            )));
        }
        if workload_kind == "prefill" && total_prefill_tokens < 1 {
            return Err(structural(format!(
                "FPM row {index} is a prefill point with no prefill tokens"
            )));
        }
        if workload_kind == "decode" && total_prefill_tokens != 0 {
            return Err(structural(format!(
                "FPM row {index} is a decode point carrying prefill tokens"
            )));
        }

        let match_identity: Vec<String> = FPM_CELL_MATCH_COLUMNS
            .iter()
            .map(|name| {
                if str_idx.contains_key(name) {
                    get_str(name)
                } else {
                    get_int_identity(name)
                }
            })
            .collect::<Result<_, _>>()?;
        // Full physical row key (collector contract) for duplicate detection,
        // in Python's _ROW_KEY_COLUMNS order.
        let row_key: Vec<String> = vec![
            get_str("cell_id")?,
            get_str("model_path")?,
            get_str("system")?,
            get_str("backend")?,
            backend_version,
            get_str("weight_quantization")?,
            match_identity[0].clone(),
            match_identity[1].clone(),
            match_identity[2].clone(),
            match_identity[3].clone(),
            match_identity[4].clone(),
            match_identity[5].clone(),
            match_identity[6].clone(),
            match_identity[7].clone(),
            match_identity[8].clone(),
            match_identity[9].clone(),
            match_identity[10].clone(),
            get_str("backend_axis")?,
            get_str("backend_policy")?,
            workload_kind.clone(),
            batch_size.to_string(),
            total_prefill_tokens.to_string(),
            total_kv_read_tokens.to_string(),
            partition_policy,
        ];

        rows.push(FpmRow {
            cell_id: get_str("cell_id")?,
            model_path: get_str("model_path")?,
            backend_axis: get_str("backend_axis")?,
            backend_policy: get_str("backend_policy")?,
            match_identity,
            row_key,
            workload_kind,
            batch_size,
            total_prefill_tokens,
            total_kv_read_tokens,
            latency_ms,
        });
    }

    // Duplicate physical row keys are collector bugs, not last-wins merges.
    let mut seen: std::collections::BTreeSet<&[String]> = std::collections::BTreeSet::new();
    for row in &rows {
        if !seen.insert(&row.row_key) {
            return Err(structural(format!(
                "FPM database contains a duplicate physical row key: {:?}",
                row.row_key
            )));
        }
    }
    // The physical row key includes cell_id/weight_quantization, which the
    // cell key does not: two rows differing only there pass the duplicate
    // check yet target the SAME table slot. Producing such rows is the
    // collector's bug to prevent (aggregate_cell dedups; the formal writer
    // keeps per-cell identities disjoint) — but a hand-merged pair must fail
    // loudly here instead of silently last-winning.
    let mut seen_coords: std::collections::BTreeSet<(Vec<String>, &str, u32, u32, u32)> =
        std::collections::BTreeSet::new();
    for (index, row) in rows.iter().enumerate() {
        if !seen_coords.insert((
            row.cell_key(),
            row.workload_kind.as_str(),
            row.batch_size,
            row.total_prefill_tokens,
            row.total_kv_read_tokens,
        )) {
            return Err(structural(format!(
                "FPM row {index} collides with an earlier row at the same cell coordinates \
                 (phase={:?}, batch_size={}, total_prefill_tokens={}, total_kv_read_tokens={}); \
                 refusing to silently overwrite latencies.",
                row.workload_kind, row.batch_size, row.total_prefill_tokens, row.total_kv_read_tokens
            )));
        }
    }

    // Group into cells (BTreeMap keeps a deterministic cell order).
    struct Building {
        cell: FpmForwardCell,
    }
    let mut cells: BTreeMap<Vec<String>, Building> = BTreeMap::new();
    for row in &rows {
        let building = cells.entry(row.cell_key()).or_insert_with(|| Building {
            cell: FpmForwardCell {
                model_path: row.model_path.clone(),
                backend_axis: row.backend_axis.clone(),
                backend_policy: row.backend_policy.clone(),
                match_identity: row.match_identity.clone(),
                cell_ids: Vec::new(),
                prefill: Node::branch(),
                decode: Node::branch(),
                prefill_domain: None,
                decode_domain: None,
                prefill_index: None,
                decode_index: None,
            },
        });
        if !building.cell.cell_ids.contains(&row.cell_id) {
            building.cell.cell_ids.push(row.cell_id.clone());
        }
        if row.workload_kind == "prefill" {
            building.cell.prefill.insert(
                &[
                    row.batch_size,
                    row.total_prefill_tokens,
                    row.total_kv_read_tokens,
                ],
                row.latency_ms,
            );
        } else {
            building
                .cell
                .decode
                .insert(&[row.batch_size, row.total_kv_read_tokens], row.latency_ms);
        }
    }

    let cells = cells
        .into_values()
        .map(|b| {
            let mut cell = b.cell;
            cell.prefill_domain = domain::<3>(&cell.prefill);
            cell.decode_domain = domain::<2>(&cell.decode);
            if cell.prefill_domain.is_some() {
                // prefill sites (batch, kv) at axes (0, 2); curve = axis 1
                cell.prefill_index = Some(SiteIndex::build(&[0, 2], 1, &cell.prefill));
            }
            if cell.decode_domain.is_some() {
                // decode sites (batch,) at axis 0; curve = axis 1
                cell.decode_index = Some(SiteIndex::build(&[0], 1, &cell.decode));
            }
            cell
        })
        .collect();
    Ok(Some(cells))
}

/// Per-axis `(min, max)` over all leaf paths — the axis-aligned bounding box
/// Python computes via `_walk_points`. `None` for an empty table.
fn domain<const N: usize>(table: &Node) -> Option<[(u32, u32); N]> {
    let mut out: Option<[(u32, u32); N]> = None;
    let mut path = [0u32; N];
    walk::<N>(table, 0, &mut path, &mut out);
    out
}

fn walk<const N: usize>(
    node: &Node,
    depth: usize,
    path: &mut [u32; N],
    out: &mut Option<[(u32, u32); N]>,
) {
    match node {
        Node::Leaf(_) => {
            debug_assert_eq!(depth, N);
            match out {
                None => *out = Some(path.map(|v| (v, v))),
                Some(domain) => {
                    for (slot, &v) in domain.iter_mut().zip(path.iter()) {
                        slot.0 = slot.0.min(v);
                        slot.1 = slot.1.max(v);
                    }
                }
            }
        }
        Node::Branch(map) => {
            for (&k, child) in map {
                path[depth] = k;
                walk::<N>(child, depth + 1, path, out);
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;

    use super::*;

    /// Default 9-row fixture mirroring `tests/unit/sdk/test_fpm_forward.py`:
    /// prefill P>0 rows at two batches, one P=0-adjacent KV site, and a decode
    /// KV sweep at two batches.
    /// (workload_kind, batch, total_prefill, total_kv, latency_ms)
    pub(crate) const DEFAULT_ROWS: [(&str, u32, u32, u32, f64); 9] = [
        ("prefill", 1, 1024, 0, 10.0),
        ("prefill", 1, 2048, 0, 20.0),
        ("prefill", 1, 4096, 0, 40.0),
        ("prefill", 1, 2048, 2048, 24.0),
        ("prefill", 2, 2048, 0, 21.0),
        ("decode", 8, 0, 8, 6.0),
        ("decode", 8, 0, 4096, 7.0),
        ("decode", 8, 0, 65536, 9.0),
        ("decode", 16, 0, 65536, 12.0),
    ];

    pub(crate) struct RowSpec {
        pub workload_kind: &'static str,
        pub batch_size: u32,
        pub total_prefill_tokens: u32,
        pub total_kv_read_tokens: u32,
        pub latency_ms: f64,
        pub model_path: &'static str,
        pub backend_axis: &'static str,
        pub backend_policy: &'static str,
        pub backend_version: &'static str,
        pub partition_policy: &'static str,
        pub tp: u32,
        /// Overrides the coordinate-derived cell_id (collision tests).
        pub cell_id: Option<&'static str>,
        pub system: &'static str,
        pub backend: &'static str,
    }

    impl Default for RowSpec {
        fn default() -> Self {
            RowSpec {
                workload_kind: "decode",
                batch_size: 8,
                total_prefill_tokens: 0,
                total_kv_read_tokens: 4096,
                latency_ms: 7.0,
                model_path: "org/model-a",
                backend_axis: "baseline",
                backend_policy: "baseline_auto",
                backend_version: "0.25.1",
                partition_policy: FPM_FORWARD_PARTITION_POLICY,
                tp: 4,
                cell_id: None,
                system: "b200_sxm",
                backend: "vllm",
            }
        }
    }

    pub(crate) fn default_rows() -> Vec<RowSpec> {
        DEFAULT_ROWS
            .iter()
            .map(|&(kind, batch, prefill, kv, lat)| RowSpec {
                workload_kind: kind,
                batch_size: batch,
                total_prefill_tokens: if kind == "decode" { 0 } else { prefill },
                total_kv_read_tokens: if kind == "decode" { kv } else { kv },
                latency_ms: lat,
                ..RowSpec::default()
            })
            .collect()
    }

    /// The 11-string identity every default row carries, in
    /// `FPM_CELL_MATCH_COLUMNS` order.
    pub(crate) fn default_identity(tp: u32) -> Vec<String> {
        vec![
            "nvfp4".to_string(),          // gemm_quant_mode
            "nvfp4".to_string(),          // moe_quant_mode
            "bfloat16".to_string(),       // fmha_quant_mode
            "half".to_string(),           // comm_quant_mode
            "fp8".to_string(),            // kv_cache_dtype
            tp.to_string(),               // tp
            "1".to_string(),              // pp
            "1".to_string(),              // dp
            tp.to_string(),               // moe_tp
            "1".to_string(),              // moe_ep
            "1".to_string(),              // cp
        ]
    }

    /// Write the v5-schema parquet + sha256'd sidecar pair into `dir`.
    pub(crate) fn write_pair(dir: &Path, rows: &[RowSpec]) -> PathBuf {
        write_pair_with(dir, rows, |_| {})
    }

    /// Same, but lets a test corrupt the sidecar after the digest is computed.
    pub(crate) fn write_pair_with(
        dir: &Path,
        rows: &[RowSpec],
        mutate_sidecar: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>),
    ) -> PathBuf {
        use parquet::data_type::{ByteArray, ByteArrayType, DoubleType, Int64Type};
        use parquet::file::properties::WriterProperties;
        use parquet::file::writer::SerializedFileWriter;
        use parquet::schema::parser::parse_message_type;

        let parquet_path = dir.join(FPM_FORWARD_BASENAME);
        let schema = "message schema {
            REQUIRED BINARY cell_id (UTF8);
            REQUIRED BINARY model_path (UTF8);
            REQUIRED BINARY system (UTF8);
            REQUIRED BINARY backend (UTF8);
            REQUIRED BINARY backend_version (UTF8);
            REQUIRED BINARY weight_quantization (UTF8);
            REQUIRED BINARY gemm_quant_mode (UTF8);
            REQUIRED BINARY moe_quant_mode (UTF8);
            REQUIRED BINARY fmha_quant_mode (UTF8);
            REQUIRED BINARY comm_quant_mode (UTF8);
            REQUIRED BINARY kv_cache_dtype (UTF8);
            REQUIRED INT64 tp;
            REQUIRED INT64 pp;
            REQUIRED INT64 dp;
            REQUIRED INT64 moe_tp;
            REQUIRED INT64 moe_ep;
            REQUIRED INT64 cp;
            REQUIRED BINARY backend_axis (UTF8);
            REQUIRED BINARY backend_policy (UTF8);
            REQUIRED BINARY workload_kind (UTF8);
            REQUIRED INT64 batch_size;
            REQUIRED INT64 total_prefill_tokens;
            REQUIRED INT64 total_kv_read_tokens;
            REQUIRED BINARY partition_policy (UTF8);
            REQUIRED DOUBLE latency_ms;
        }";
        let schema = Arc::new(parse_message_type(schema).expect("schema must parse"));
        let file = std::fs::File::create(&parquet_path).expect("create parquet");
        let mut writer =
            SerializedFileWriter::new(file, schema, Arc::new(WriterProperties::builder().build()))
                .expect("writer");
        let mut rg = writer.next_row_group().expect("row group");

        let identity = |r: &RowSpec| default_identity(r.tp);
        let str_col = |f: &dyn Fn(&RowSpec) -> String| -> Vec<ByteArray> {
            rows.iter().map(|r| ByteArray::from(f(r).as_str())).collect()
        };
        let str_batches: Vec<Vec<ByteArray>> = vec![
            str_col(&|r| {
                r.cell_id.map(str::to_string).unwrap_or_else(|| {
                    format!(
                        "fpm-{}-{}-{}-{}",
                        r.workload_kind, r.batch_size, r.total_prefill_tokens, r.total_kv_read_tokens
                    )
                })
            }),
            str_col(&|r| r.model_path.to_string()),
            str_col(&|r| r.system.to_string()),
            str_col(&|r| r.backend.to_string()),
            str_col(&|r| r.backend_version.to_string()),
            str_col(&|r| identity(r)[0].clone()),
        ];
        for values in &str_batches {
            let mut col = rg.next_column().expect("next col").expect("str col");
            col.typed::<ByteArrayType>().write_batch(values, None, None).expect("write");
            col.close().expect("close");
        }
        // gemm/moe/fmha/comm/kv identity string columns
        for idx in 0..5usize {
            let values = str_col(&|r| identity(r)[idx].clone());
            let mut col = rg.next_column().expect("next col").expect("str col");
            col.typed::<ByteArrayType>().write_batch(&values, None, None).expect("write");
            col.close().expect("close");
        }
        // tp pp dp moe_tp moe_ep cp
        for idx in 5..11usize {
            let values: Vec<i64> = rows
                .iter()
                .map(|r| identity(r)[idx].parse::<i64>().unwrap())
                .collect();
            let mut col = rg.next_column().expect("next col").expect("int col");
            col.typed::<Int64Type>().write_batch(&values, None, None).expect("write");
            col.close().expect("close");
        }
        for values in [
            str_col(&|r| r.backend_axis.to_string()),
            str_col(&|r| r.backend_policy.to_string()),
            str_col(&|r| r.workload_kind.to_string()),
        ] {
            let mut col = rg.next_column().expect("next col").expect("str col");
            col.typed::<ByteArrayType>().write_batch(&values, None, None).expect("write");
            col.close().expect("close");
        }
        for values in [
            rows.iter().map(|r| r.batch_size as i64).collect::<Vec<i64>>(),
            rows.iter().map(|r| r.total_prefill_tokens as i64).collect(),
            rows.iter().map(|r| r.total_kv_read_tokens as i64).collect(),
        ] {
            let mut col = rg.next_column().expect("next col").expect("int col");
            col.typed::<Int64Type>().write_batch(&values, None, None).expect("write");
            col.close().expect("close");
        }
        {
            let values = str_col(&|r| r.partition_policy.to_string());
            let mut col = rg.next_column().expect("next col").expect("str col");
            col.typed::<ByteArrayType>().write_batch(&values, None, None).expect("write");
            col.close().expect("close");
        }
        {
            let values: Vec<f64> = rows.iter().map(|r| r.latency_ms).collect();
            let mut col = rg.next_column().expect("next col").expect("f64 col");
            col.typed::<DoubleType>().write_batch(&values, None, None).expect("write");
            col.close().expect("close");
        }
        rg.close().expect("close row group");
        writer.close().expect("close writer");

        let mut sidecar = serde_json::Map::new();
        sidecar.insert("schema_name".into(), FPM_FORWARD_SCHEMA_NAME.into());
        sidecar.insert(
            "schema_version".into(),
            serde_json::Value::from(FPM_FORWARD_SCHEMA_VERSION),
        );
        sidecar.insert(
            "coordinate_system".into(),
            FPM_FORWARD_COORDINATE_SYSTEM.into(),
        );
        sidecar.insert(
            "parquet_sha256".into(),
            sha256_file(&parquet_path).expect("sha256").into(),
        );
        sidecar.insert("row_count".into(), serde_json::Value::from(rows.len() as u64));
        mutate_sidecar(&mut sidecar);
        std::fs::write(
            parquet_path.with_extension("metadata.json"),
            serde_json::to_string_pretty(&serde_json::Value::Object(sidecar)).unwrap(),
        )
        .expect("write sidecar");
        parquet_path
    }

    fn loaded_table(dir: &Path) -> FpmForwardTable {
        FpmForwardTable::new(dir.to_path_buf(), "b200_sxm", "vllm", "0.25.1")
    }

    #[test]
    fn absent_parquet_is_soft_not_collected() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let table = loaded_table(tmp.path());
        let err = table.cells().unwrap_err();
        assert!(
            err.to_string().contains("File does not exist at"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn default_pair_loads_one_cell_with_domains() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_pair(tmp.path(), &default_rows());
        let table = loaded_table(tmp.path());
        let cells = table.cells().expect("must load");
        assert_eq!(cells.len(), 1);
        let cell = &cells[0];
        assert_eq!(cell.model_path, "org/model-a");
        assert_eq!(cell.match_identity, default_identity(4));
        assert_eq!(cell.prefill_domain, Some([(1, 2), (1024, 4096), (0, 2048)]));
        assert_eq!(cell.decode_domain, Some([(8, 16), (8, 65536)]));
        assert!(cell.prefill_index.is_some() && cell.decode_index.is_some());
    }

    #[test]
    fn missing_sidecar_is_a_loud_error() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let parquet = write_pair(tmp.path(), &default_rows());
        std::fs::remove_file(parquet.with_extension("metadata.json")).unwrap();
        let err = loaded_table(tmp.path()).cells().unwrap_err();
        assert!(err.to_string().contains("missing its metadata sidecar"), "{err}");
    }

    #[test]
    fn sidecar_gates_fire_in_order() {
        for (mutate, needle) in [
            (
                Box::new(|m: &mut serde_json::Map<String, serde_json::Value>| {
                    m.insert("schema_name".into(), "other".into());
                }) as Box<dyn FnOnce(&mut serde_json::Map<String, serde_json::Value>)>,
                "unsupported FPM schema_name",
            ),
            (
                Box::new(|m: &mut serde_json::Map<String, serde_json::Value>| {
                    m.insert("schema_version".into(), serde_json::Value::from(4));
                }),
                "unsupported FPM schema_version",
            ),
            (
                Box::new(|m: &mut serde_json::Map<String, serde_json::Value>| {
                    m.insert("coordinate_system".into(), "other_v0".into());
                }),
                "unsupported FPM coordinate_system",
            ),
            (
                Box::new(|m: &mut serde_json::Map<String, serde_json::Value>| {
                    m.insert("parquet_sha256".into(), "deadbeef".into());
                }),
                "digest mismatch",
            ),
            (
                Box::new(|m: &mut serde_json::Map<String, serde_json::Value>| {
                    m.insert("row_count".into(), serde_json::Value::from(3));
                }),
                "row_count mismatch",
            ),
        ] {
            let tmp = tempfile::tempdir().expect("tmpdir");
            write_pair_with(tmp.path(), &default_rows(), mutate);
            let err = loaded_table(tmp.path()).cells().unwrap_err();
            assert!(err.to_string().contains(needle), "wanted {needle:?} in {err}");
        }
    }

    #[test]
    fn row_gates_reject_bad_rows() {
        for (mutate, needle) in [
            (
                Box::new(|r: &mut RowSpec| r.workload_kind = "mixed")
                    as Box<dyn FnOnce(&mut RowSpec)>,
                "unknown workload_kind",
            ),
            (
                Box::new(|r: &mut RowSpec| r.partition_policy = "greedy_v0"),
                "unsupported partition_policy",
            ),
            (
                Box::new(|r: &mut RowSpec| r.backend_version = "0.19.0"),
                "does not match the database version directory",
            ),
            (Box::new(|r: &mut RowSpec| r.latency_ms = 0.0), "non-positive latency_ms"),
            (
                Box::new(|r: &mut RowSpec| r.latency_ms = f64::NAN),
                "non-finite/non-positive latency_ms",
            ),
            (
                Box::new(|r: &mut RowSpec| {
                    r.workload_kind = "prefill";
                    r.total_prefill_tokens = 0;
                }),
                "prefill point with no prefill tokens",
            ),
            (
                Box::new(|r: &mut RowSpec| {
                    r.workload_kind = "decode";
                    r.total_prefill_tokens = 64;
                }),
                "decode point carrying prefill tokens",
            ),
        ] {
            let tmp = tempfile::tempdir().expect("tmpdir");
            let mut rows = default_rows();
            mutate(&mut rows[5]);
            write_pair(tmp.path(), &rows);
            let err = loaded_table(tmp.path()).cells().unwrap_err();
            assert!(err.to_string().contains(needle), "wanted {needle:?} in {err}");
        }
    }

    #[test]
    fn duplicate_physical_row_key_is_rejected() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let mut rows = default_rows();
        let dup = RowSpec {
            workload_kind: rows[6].workload_kind,
            batch_size: rows[6].batch_size,
            total_prefill_tokens: rows[6].total_prefill_tokens,
            total_kv_read_tokens: rows[6].total_kv_read_tokens,
            latency_ms: 99.0, // same key, different latency: still a duplicate
            ..RowSpec::default()
        };
        rows.push(dup);
        write_pair(tmp.path(), &rows);
        let err = loaded_table(tmp.path()).cells().unwrap_err();
        assert!(err.to_string().contains("duplicate physical row key"), "{err}");
    }

    /// Python compares sidecar numbers by VALUE (`5.0 == 5`); a writer that
    /// went through float-typed JSON must not be rejected.
    #[test]
    fn sidecar_integral_floats_are_accepted() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_pair_with(tmp.path(), &default_rows(), |m| {
            m.insert("schema_version".into(), serde_json::Value::from(5.0));
            m.insert("row_count".into(), serde_json::Value::from(9.0));
        });
        assert!(loaded_table(tmp.path()).cells().is_ok());
    }

    /// Python checks row_count BEFORE per-row validation; a pair that is both
    /// short-counted and carries a bad row must report the count mismatch.
    #[test]
    fn row_count_mismatch_preempts_row_gates() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let mut rows = default_rows();
        rows[5].latency_ms = -1.0; // bad row AND wrong sidecar count
        write_pair_with(tmp.path(), &rows, |m| {
            m.insert("row_count".into(), serde_json::Value::from(3));
        });
        let err = loaded_table(tmp.path()).cells().unwrap_err();
        assert!(err.to_string().contains("row_count mismatch"), "{err}");
    }

    /// system/backend are in the physical row key but NOT the cell key: a
    /// misplaced pair must fail loudly, not merge into the same cells.
    #[test]
    fn misplaced_system_or_backend_fails_loudly() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let mut rows = default_rows();
        rows[0].system = "gb200_nvl72";
        write_pair(tmp.path(), &rows);
        let err = loaded_table(tmp.path()).cells().unwrap_err();
        assert!(err.to_string().contains("does not match the database system"), "{err}");

        let tmp = tempfile::tempdir().expect("tmpdir");
        let mut rows = default_rows();
        rows[0].backend = "sglang";
        write_pair(tmp.path(), &rows);
        let err = loaded_table(tmp.path()).cells().unwrap_err();
        assert!(err.to_string().contains("does not match the database backend"), "{err}");
    }

    /// Same cell identity + same coordinates but a different cell_id passes
    /// the physical-row-key duplicate check yet targets the SAME table slot;
    /// the loader must refuse instead of silently last-winning.
    #[test]
    fn coordinate_collision_within_cell_fails_loudly() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let mut rows = default_rows();
        let mut clash = RowSpec {
            workload_kind: rows[6].workload_kind,
            batch_size: rows[6].batch_size,
            total_prefill_tokens: rows[6].total_prefill_tokens,
            total_kv_read_tokens: rows[6].total_kv_read_tokens,
            ..RowSpec::default()
        };
        clash.cell_id = Some("fpm-another-attempt");
        clash.latency_ms = 99.0;
        rows.push(clash);
        write_pair(tmp.path(), &rows);
        let err = loaded_table(tmp.path()).cells().unwrap_err();
        assert!(err.to_string().contains("collides with an earlier row"), "{err}");
    }

    #[test]
    fn select_cell_prefers_exact_model_path_and_reports_ambiguity() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let mut rows = default_rows();
        for row in default_rows() {
            rows.push(RowSpec {
                model_path: "org/model-b",
                latency_ms: row.latency_ms * 2.0,
                ..row
            });
        }
        write_pair(tmp.path(), &rows);
        let table = loaded_table(tmp.path());
        let identity = default_identity(4);

        // Exact path wins.
        let cell = table.select_cell(&identity, "org/model-a").expect("select");
        assert_eq!(cell.model_path, "org/model-a");
        // Unknown path with two candidates: the multi-path ambiguity error.
        let err = table.select_cell(&identity, "org/other").unwrap_err();
        assert!(err.to_string().contains("Ambiguous FPM cell selection"), "{err}");
        // Unknown identity: the no-match error listing what was collected.
        let err = table
            .select_cell(&default_identity(8), "org/model-a")
            .unwrap_err();
        assert!(err.to_string().contains("No FPM cell matches"), "{err}");
    }

    #[test]
    fn non_baseline_backend_axis_never_matches() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let rows: Vec<RowSpec> = default_rows()
            .into_iter()
            .map(|r| RowSpec {
                backend_axis: "wideep",
                ..r
            })
            .collect();
        write_pair(tmp.path(), &rows);
        let err = loaded_table(tmp.path())
            .select_cell(&default_identity(4), "org/model-a")
            .unwrap_err();
        assert!(err.to_string().contains("No FPM cell matches"), "{err}");
    }
}
