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
//! `(model_path, 15 identity columns)`. Each
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
pub const FPM_FORWARD_SCHEMA_VERSION: u64 = 6;
pub const FPM_FORWARD_COORDINATE_SYSTEM: &str = "iteration_totals_balanced_v1";
pub const FPM_FORWARD_PARTITION_POLICY: &str = "balanced_v1";
/// The only measurement policy the collector publishes; pinned in the
/// sidecar gate (a pair measured under a different regime is structural).
pub const FPM_FORWARD_MEASUREMENT_POLICY: &str = "dynamo_native_single_sample_v1";

/// Identity columns that select a cell, in row-column order (`model_path` is
/// handled separately; `weight_quantization` is deliberately excluded).
/// The last four are the schema-v6 explicit backend identity: "auto" = the
/// engine decided; the `enable_*` columns are real parquet booleans,
/// normalized to "True"/"False" (Python `str(bool)`) for comparison.
pub const FPM_CELL_MATCH_COLUMNS: [&str; 15] = [
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
    "moe_backend",
    "attention_backend",
    "enable_wideep",
    "enable_eplb",
];

pub const FPM_PREFILL_AXES: [&str; 3] =
    ["batch_size", "total_prefill_tokens", "total_kv_read_tokens"];
pub const FPM_DECODE_AXES: [&str; 2] = ["batch_size", "total_kv_read_tokens"];

/// One collected cell: the tables and domains for a single
/// `(model_path, identity)` tuple (the 15-column identity carries the
/// backend knobs since schema v6).
#[derive(Debug)]
pub struct FpmForwardCell {
    pub model_path: String,
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
    /// Collected prefill batch ceiling, present only when the data
    /// certifies the batch axis flat at matched totals (median |ratio-1|
    /// <= 5% across consecutive collected batches, >= 3 overlap points):
    /// queries above it clamp their batch coordinate to this value with
    /// TRUE totals — same GEMM/MoE work, same side of the capture cliff,
    /// bounded-conservative on attention. `None` = hard gate as before.
    pub prefill_batch_clamp_max: Option<u32>,
    /// Decode batch-axis bracket metadata (mirrors Python): the engine pads
    /// decode batches to capture rungs — every rung is marked in the data by
    /// an (x, x+1) row pair — and off-lattice queries interpolate between
    /// their segment's bracket rows at the op layer instead of the
    /// regime-blind cross-batch k-NN. `decode_curve_bounds` feeds the
    /// op-layer coverage guard.
    pub decode_batches: Vec<u32>,
    pub decode_rungs: Vec<u32>,
    pub decode_curve_bounds: BTreeMap<u32, (u32, u32)>,
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
        let loaded = self.cells.get_or_init(|| {
            load_pair(
                &self.parquet_path,
                &self.system,
                &self.backend,
                &self.version,
            )
        });
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
    /// strict equality on the 15-column identity and `model_path` (no
    /// fallback), then a hard ambiguity guard.
    pub fn select_cell(
        &self,
        match_identity: &[String],
        model_path: &str,
    ) -> Result<&FpmForwardCell, AicError> {
        let cells = self.cells()?;
        // Exact matching on every identity dimension (D1 resolved: the match
        // identity carries no architecture fingerprint, so borrowing the sole
        // collected model_path could silently answer for a different model).
        let matches: Vec<&FpmForwardCell> = cells
            .iter()
            .filter(|cell| cell.match_identity == match_identity && cell.model_path == model_path)
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
                "No FPM cell matches model_path={model_path:?} with identity {{{}}}. \
                 FPM never substitutes another model's curves; collect data \
                 under this exact model path or query with the collected path. Collected cells \
                 (model_path, identity): [{}]",
                identity.join(", "),
                available.join(", ")
            )));
        }
        if matches.len() > 1 {
            // Unreachable through the loader (cells are keyed by exactly
            // these fields) but kept as a hard guard against key widening.
            return Err(structural(format!(
                "Ambiguous FPM cell selection for model_path={model_path:?}: {} cells share \
                 this identity and backend policy.",
                matches.len()
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
fn validate_sidecar(
    metadata_path: &Path,
    parquet_path: &Path,
    system: &str,
    backend: &str,
    version: &str,
) -> Result<Option<u64>, AicError> {
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
    if metadata.get("measurement_policy").and_then(|v| v.as_str())
        != Some(FPM_FORWARD_MEASUREMENT_POLICY)
    {
        return Err(structural(format!(
            "unsupported FPM measurement_policy={:?} (expected {FPM_FORWARD_MEASUREMENT_POLICY:?}): {}",
            metadata.get("measurement_policy"),
            metadata_path.display()
        )));
    }
    // The commit record names the database identity it was published for;
    // contradictory metadata (a pair copied into the wrong tree with its
    // sidecar) is rejected here, before any row is read.
    for (key, expected) in [
        ("system", system),
        ("backend", backend),
        ("backend_version", version),
    ] {
        if metadata.get(key).and_then(|v| v.as_str()) != Some(expected) {
            return Err(structural(format!(
                "FPM sidecar {key}={:?} does not match the database {key} {expected:?}: {}",
                metadata.get(key),
                metadata_path.display()
            )));
        }
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
        let mut key = Vec::with_capacity(1 + self.match_identity.len());
        key.push(self.model_path.clone());
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
    let sidecar_row_count =
        validate_sidecar(&metadata_path, parquet_path, system, backend, version)?;

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
        "moe_backend",
        "attention_backend",
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
    let bool_cols = ["enable_wideep", "enable_eplb"];
    let mut bool_idx = BTreeMap::new();
    for name in bool_cols {
        bool_idx.insert(name, reader.col(name)?);
    }
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
            Ok(row
                .str_optional(Some(str_idx[name]))?
                .unwrap_or("")
                .to_string())
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
        // Schema-v6 enable_* flags: REAL booleans only (no string/int
        // coercion — Python rejects those, and a coerced "true" would
        // compare as "true" vs the request side's "True"). Normalized to
        // Python str(bool) capitalization.
        let get_bool_identity = |name: &str| -> Result<String, AicError> {
            let value = row
                .bool_strict(bool_idx[name])
                .map_err(|_| structural(format!("FPM row {index} {name} must be a boolean")))?;
            Ok(if value {
                "True".to_string()
            } else {
                "False".to_string()
            })
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
                } else if bool_idx.contains_key(name) {
                    get_bool_identity(name)
                } else {
                    get_int_identity(name)
                }
            })
            .collect::<Result<_, _>>()?;
        // The string backend knobs must be present: "auto" or a pinned name.
        for (offset, name) in [(11usize, "moe_backend"), (12usize, "attention_backend")] {
            if match_identity[offset].is_empty() {
                return Err(structural(format!(
                    "FPM row {index} {name} must be a non-empty string (\"auto\" or a pinned name)"
                )));
            }
        }
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
            match_identity[11].clone(),
            match_identity[12].clone(),
            match_identity[13].clone(),
            match_identity[14].clone(),
            workload_kind.clone(),
            batch_size.to_string(),
            total_prefill_tokens.to_string(),
            total_kv_read_tokens.to_string(),
            partition_policy,
        ];

        rows.push(FpmRow {
            cell_id: get_str("cell_id")?,
            model_path: get_str("model_path")?,
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
                row.workload_kind,
                row.batch_size,
                row.total_prefill_tokens,
                row.total_kv_read_tokens
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
                match_identity: row.match_identity.clone(),
                cell_ids: Vec::new(),
                prefill: Node::branch(),
                decode: Node::branch(),
                prefill_domain: None,
                decode_domain: None,
                prefill_index: None,
                decode_index: None,
                prefill_batch_clamp_max: None,
                decode_batches: Vec::new(),
                decode_rungs: Vec::new(),
                decode_curve_bounds: BTreeMap::new(),
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
            if cell.prefill_domain.is_some() && prefill_batch_axis_is_flat(&cell.prefill) {
                cell.prefill_batch_clamp_max = cell.prefill_domain.map(|d| d[0].1);
            }
            if cell.decode_domain.is_some() {
                // decode sites (batch,) at axis 0; curve = axis 1
                cell.decode_index = Some(SiteIndex::build(&[0], 1, &cell.decode));
                // Bracket metadata (mirrors Python's loader).
                let curves = decode_curves(&cell.decode);
                cell.decode_batches = curves.keys().copied().collect();
                cell.decode_rungs = cell
                    .decode_batches
                    .iter()
                    .copied()
                    .filter(|b| b.checked_add(1).is_some_and(|n| curves.contains_key(&n)))
                    .collect();
                cell.decode_curve_bounds = curves
                    .iter()
                    .map(|(&b, curve)| {
                        let low = *curve.keys().next().expect("non-empty curve");
                        let high = *curve.keys().next_back().expect("non-empty curve");
                        (b, (low, high))
                    })
                    .collect();
            }
            Ok(cell)
        })
        .collect::<Result<Vec<_>, AicError>>()?;
    Ok(Some(cells))
}

/// Piecewise-linear evaluation of one decode KV curve (detection-only; the
/// query path uses perf_interp). Mirrors Python `_decode_curve_value`.
fn decode_curve_value(curve: &BTreeMap<u32, f64>, kv: f64) -> f64 {
    let keys: Vec<u32> = curve.keys().copied().collect();
    let first = keys[0];
    let last = keys[keys.len() - 1];
    if kv <= first as f64 {
        return curve[&first];
    }
    if kv >= last as f64 {
        return curve[&last];
    }
    let hi = keys.partition_point(|&k| (k as f64) < kv);
    let (k0, k1) = (keys[hi - 1], keys[hi]);
    if k1 == k0 {
        return curve[&k0];
    }
    let w = (kv - k0 as f64) / (k1 as f64 - k0 as f64);
    curve[&k0] + (curve[&k1] - curve[&k0]) * w
}

/// Flatten a decode `Node` into `batch -> (kv -> latency)` (detection view).
fn decode_curves(decode: &Node) -> BTreeMap<u32, BTreeMap<u32, f64>> {
    let mut out: BTreeMap<u32, BTreeMap<u32, f64>> = BTreeMap::new();
    if let Node::Branch(batches) = decode {
        for (&batch, curve_node) in batches {
            if let Node::Branch(kvs) = curve_node {
                let curve = out.entry(batch).or_default();
                for (&kv, leaf) in kvs {
                    if let Node::Leaf(value) = leaf {
                        curve.insert(kv, value.latency);
                    }
                }
            }
        }
    }
    out
}

/// Data certificate for the prefill batch clamp, mirroring Python
/// `_prefill_batch_axis_is_flat`: consecutive collected batches must agree
/// (median |ratio - 1| <= 5% over >= 3 overlapping total-token points per
/// shared KV slice). Compared at matched totals = within one CUDA-graph
/// regime; the capture cliff lives on the token axis and cannot leak in.
fn prefill_batch_axis_is_flat(prefill: &Node) -> bool {
    // batch -> kv -> (total -> latency)
    let mut slices: BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, f64>>> = BTreeMap::new();
    if let Node::Branch(batches) = prefill {
        for (&batch, totals_node) in batches {
            if let Node::Branch(totals) = totals_node {
                for (&total, kv_node) in totals {
                    if let Node::Branch(kvs) = kv_node {
                        for (&kv, leaf) in kvs {
                            if let Node::Leaf(value) = leaf {
                                slices
                                    .entry(batch)
                                    .or_default()
                                    .entry(kv)
                                    .or_default()
                                    .insert(total, value.latency);
                            }
                        }
                    }
                }
            }
        }
    }
    let batches: Vec<u32> = slices.keys().copied().collect();
    if batches.len() < 2 {
        return false;
    }
    let mut deviations: Vec<f64> = Vec::new();
    for pair in batches.windows(2) {
        let (lo_b, hi_b) = (pair[0], pair[1]);
        for (kv, upper) in &slices[&hi_b] {
            let Some(lower) = slices[&lo_b].get(kv) else {
                continue;
            };
            let lo_first = *lower.keys().next().expect("non-empty") as f64;
            let lo_last = *lower.keys().next_back().expect("non-empty") as f64;
            for (&total, &lat) in upper {
                let t = total as f64;
                if lo_first <= t && t <= lo_last {
                    let base = decode_curve_value(lower, t);
                    if base > 0.0 {
                        deviations.push((lat / base - 1.0).abs());
                    }
                }
            }
        }
    }
    deviations.len() >= 3 && median(&mut deviations) <= 0.05
}

/// Python `statistics.median`: even count averages the two middle values.
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
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
        pub moe_backend: &'static str,
        pub attention_backend: &'static str,
        pub enable_wideep: bool,
        pub enable_eplb: bool,
        pub backend_version: &'static str,
        pub partition_policy: &'static str,
        pub tp: u32,
        pub moe_tp: u32,
        pub moe_ep: u32,
        pub fmha_quant_mode: &'static str,
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
                moe_backend: "auto",
                attention_backend: "auto",
                enable_wideep: false,
                enable_eplb: false,
                backend_version: "0.25.1",
                partition_policy: FPM_FORWARD_PARTITION_POLICY,
                tp: 4,
                moe_tp: 4,
                moe_ep: 1,
                fmha_quant_mode: "bfloat16",
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
            "nvfp4".to_string(),    // gemm_quant_mode
            "nvfp4".to_string(),    // moe_quant_mode
            "bfloat16".to_string(), // fmha_quant_mode
            "half".to_string(),     // comm_quant_mode
            "fp8".to_string(),      // kv_cache_dtype
            tp.to_string(),         // tp
            "1".to_string(),        // pp
            "1".to_string(),        // dp
            tp.to_string(),         // moe_tp
            "1".to_string(),        // moe_ep
            "1".to_string(),        // cp
            "auto".to_string(),     // moe_backend
            "auto".to_string(),     // attention_backend
            "False".to_string(),    // enable_wideep (str(bool))
            "False".to_string(),    // enable_eplb
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
        use parquet::data_type::{BoolType, ByteArray, ByteArrayType, DoubleType, Int64Type};
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
            REQUIRED BINARY moe_backend (UTF8);
            REQUIRED BINARY attention_backend (UTF8);
            REQUIRED BOOLEAN enable_wideep;
            REQUIRED BOOLEAN enable_eplb;
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

        let identity = |r: &RowSpec| {
            let mut identity = default_identity(r.tp);
            identity[2] = r.fmha_quant_mode.to_string();
            identity[8] = r.moe_tp.to_string();
            identity[9] = r.moe_ep.to_string();
            identity
        };
        let str_col = |f: &dyn Fn(&RowSpec) -> String| -> Vec<ByteArray> {
            rows.iter()
                .map(|r| ByteArray::from(f(r).as_str()))
                .collect()
        };
        let str_batches: Vec<Vec<ByteArray>> = vec![
            str_col(&|r| {
                r.cell_id.map(str::to_string).unwrap_or_else(|| {
                    format!(
                        "fpm-{}-{}-{}-{}",
                        r.workload_kind,
                        r.batch_size,
                        r.total_prefill_tokens,
                        r.total_kv_read_tokens
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
            col.typed::<ByteArrayType>()
                .write_batch(values, None, None)
                .expect("write");
            col.close().expect("close");
        }
        // gemm/moe/fmha/comm/kv identity string columns
        for idx in 0..5usize {
            let values = str_col(&|r| identity(r)[idx].clone());
            let mut col = rg.next_column().expect("next col").expect("str col");
            col.typed::<ByteArrayType>()
                .write_batch(&values, None, None)
                .expect("write");
            col.close().expect("close");
        }
        // tp pp dp moe_tp moe_ep cp
        for idx in 5..11usize {
            let values: Vec<i64> = rows
                .iter()
                .map(|r| identity(r)[idx].parse::<i64>().unwrap())
                .collect();
            let mut col = rg.next_column().expect("next col").expect("int col");
            col.typed::<Int64Type>()
                .write_batch(&values, None, None)
                .expect("write");
            col.close().expect("close");
        }
        for values in [
            str_col(&|r| r.moe_backend.to_string()),
            str_col(&|r| r.attention_backend.to_string()),
        ] {
            let mut col = rg.next_column().expect("next col").expect("str col");
            col.typed::<ByteArrayType>()
                .write_batch(&values, None, None)
                .expect("write");
            col.close().expect("close");
        }
        for values in [
            rows.iter().map(|r| r.enable_wideep).collect::<Vec<bool>>(),
            rows.iter().map(|r| r.enable_eplb).collect::<Vec<bool>>(),
        ] {
            let mut col = rg.next_column().expect("next col").expect("bool col");
            col.typed::<BoolType>()
                .write_batch(&values, None, None)
                .expect("write");
            col.close().expect("close");
        }
        {
            let values = str_col(&|r| r.workload_kind.to_string());
            let mut col = rg.next_column().expect("next col").expect("str col");
            col.typed::<ByteArrayType>()
                .write_batch(&values, None, None)
                .expect("write");
            col.close().expect("close");
        }
        for values in [
            rows.iter()
                .map(|r| r.batch_size as i64)
                .collect::<Vec<i64>>(),
            rows.iter().map(|r| r.total_prefill_tokens as i64).collect(),
            rows.iter().map(|r| r.total_kv_read_tokens as i64).collect(),
        ] {
            let mut col = rg.next_column().expect("next col").expect("int col");
            col.typed::<Int64Type>()
                .write_batch(&values, None, None)
                .expect("write");
            col.close().expect("close");
        }
        {
            let values = str_col(&|r| r.partition_policy.to_string());
            let mut col = rg.next_column().expect("next col").expect("str col");
            col.typed::<ByteArrayType>()
                .write_batch(&values, None, None)
                .expect("write");
            col.close().expect("close");
        }
        {
            let values: Vec<f64> = rows.iter().map(|r| r.latency_ms).collect();
            let mut col = rg.next_column().expect("next col").expect("f64 col");
            col.typed::<DoubleType>()
                .write_batch(&values, None, None)
                .expect("write");
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
        sidecar.insert(
            "row_count".into(),
            serde_json::Value::from(rows.len() as u64),
        );
        sidecar.insert(
            "measurement_policy".into(),
            FPM_FORWARD_MEASUREMENT_POLICY.into(),
        );
        sidecar.insert("system".into(), "b200_sxm".into());
        sidecar.insert("backend".into(), "vllm".into());
        sidecar.insert("backend_version".into(), "0.25.1".into());
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
        assert!(
            err.to_string().contains("missing its metadata sidecar"),
            "{err}"
        );
    }

    #[test]
    fn sidecar_gates_fire_in_order() {
        for (mutate, needle) in [
            (
                Box::new(|m: &mut serde_json::Map<String, serde_json::Value>| {
                    m.insert("schema_name".into(), "other".into());
                })
                    as Box<dyn FnOnce(&mut serde_json::Map<String, serde_json::Value>)>,
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
            assert!(
                err.to_string().contains(needle),
                "wanted {needle:?} in {err}"
            );
        }
    }

    #[test]
    fn contradictory_sidecar_identity_fails_loudly() {
        // The commit record names the identity it was published for; a pair
        // whose rows match the tree but whose sidecar names foreign values
        // is inconsistent and must be rejected before any row is read.
        for (key, value) in [
            ("system", "gb200_nvl72"),
            ("backend", "sglang"),
            ("backend_version", "9.9.9"),
            ("measurement_policy", "multi_sample_v2"),
        ] {
            let tmp = tempfile::tempdir().expect("tmpdir");
            write_pair_with(tmp.path(), &default_rows(), |m| {
                m.insert(key.into(), value.into());
            });
            let err = loaded_table(tmp.path()).cells().unwrap_err();
            assert!(err.to_string().contains(key), "{key}: {err}");
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
            (
                Box::new(|r: &mut RowSpec| r.latency_ms = 0.0),
                "non-positive latency_ms",
            ),
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
            assert!(
                err.to_string().contains(needle),
                "wanted {needle:?} in {err}"
            );
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
        assert!(
            err.to_string().contains("duplicate physical row key"),
            "{err}"
        );
    }

    /// Python compares sidecar numbers by VALUE (`5.0 == 5`); a writer that
    /// went through float-typed JSON must not be rejected.
    #[test]
    fn sidecar_integral_floats_are_accepted() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_pair_with(tmp.path(), &default_rows(), |m| {
            m.insert("schema_version".into(), serde_json::Value::from(6.0));
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
        assert!(
            err.to_string()
                .contains("does not match the database system"),
            "{err}"
        );

        let tmp = tempfile::tempdir().expect("tmpdir");
        let mut rows = default_rows();
        rows[0].backend = "sglang";
        write_pair(tmp.path(), &rows);
        let err = loaded_table(tmp.path()).cells().unwrap_err();
        assert!(
            err.to_string()
                .contains("does not match the database backend"),
            "{err}"
        );
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
        assert!(
            err.to_string().contains("collides with an earlier row"),
            "{err}"
        );
    }

    #[test]
    fn select_cell_requires_exact_model_path() {
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

        // Exact path selects its own cell.
        let cell = table.select_cell(&identity, "org/model-a").expect("select");
        assert_eq!(cell.model_path, "org/model-a");
        // Unknown path: never borrows a collected cell (D1 exact-only).
        let err = table.select_cell(&identity, "org/other").unwrap_err();
        assert!(err.to_string().contains("never substitutes"), "{err}");
        // Unknown identity: the no-match error listing what was collected.
        let err = table
            .select_cell(&default_identity(8), "org/model-a")
            .unwrap_err();
        assert!(err.to_string().contains("No FPM cell matches"), "{err}");
    }

    #[test]
    fn select_cell_with_sole_foreign_path_is_a_miss() {
        // A database collected for exactly one model must not answer any
        // other model's request even when the quant/parallel identity matches.
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_pair(tmp.path(), &default_rows());
        let err = loaded_table(tmp.path())
            .select_cell(&default_identity(4), "some/other-model")
            .unwrap_err();
        assert!(err.to_string().contains("never substitutes"), "{err}");
    }

    fn cliff_decode_rows() -> Vec<RowSpec> {
        let mk = |batch: u32, kv: u32, lat: f64| RowSpec {
            workload_kind: "decode",
            batch_size: batch,
            total_prefill_tokens: 0,
            total_kv_read_tokens: kv,
            latency_ms: lat,
            ..RowSpec::default()
        };
        let mut rows = Vec::new();
        for (b, base) in [(496u32, 9.5), (497, 10.0), (512, 10.5)] {
            for (i, kv) in [1024u32, 2048, 4096].into_iter().enumerate() {
                rows.push(mk(b, kv, base + i as f64));
            }
        }
        for (b, base) in [(513u32, 31.0), (1024, 62.0)] {
            for (i, kv) in [1024u32, 2048, 4096].into_iter().enumerate() {
                rows.push(mk(b, kv, base + 3.0 * i as f64));
            }
        }
        rows
    }

    fn flat_prefill_rows() -> Vec<RowSpec> {
        let mk = |batch: u32, total: u32, lat: f64| RowSpec {
            workload_kind: "prefill",
            batch_size: batch,
            total_prefill_tokens: total,
            total_kv_read_tokens: 0,
            latency_ms: lat,
            ..RowSpec::default()
        };
        let mut rows = Vec::new();
        for (b, bump) in [(1u32, 1.0), (2, 1.02), (4, 1.04)] {
            for (total, lat) in [(1024u32, 10.0), (2048, 20.0), (4096, 40.0)] {
                rows.push(mk(b, total, lat * bump));
            }
        }
        rows
    }

    #[test]
    fn certified_prefill_batch_clamp_is_issued_and_denied_correctly() {
        // Flat ladder (2%/batch) -> certificate -> clamp ceiling = 4.
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_pair(tmp.path(), &flat_prefill_rows());
        let cell_max = loaded_table(tmp.path())
            .select_cell(&default_identity(4), "org/model-a")
            .expect("select")
            .prefill_batch_clamp_max;
        assert_eq!(cell_max, Some(4));
        // Default fixture: one sparse batch pair -> no evidence -> no clamp.
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_pair(tmp.path(), &default_rows());
        let cell_max = loaded_table(tmp.path())
            .select_cell(&default_identity(4), "org/model-a")
            .expect("select")
            .prefill_batch_clamp_max;
        assert_eq!(cell_max, None);
        // A real batch effect (>5%) fails the certificate.
        let mut bumpy = flat_prefill_rows();
        for row in bumpy.iter_mut().filter(|r| r.batch_size == 4) {
            row.latency_ms *= 1.2;
        }
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_pair(tmp.path(), &bumpy);
        let cell_max = loaded_table(tmp.path())
            .select_cell(&default_identity(4), "org/model-a")
            .expect("select")
            .prefill_batch_clamp_max;
        assert_eq!(cell_max, None);
    }

    #[test]
    fn decode_bracket_metadata_built() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_pair(tmp.path(), &cliff_decode_rows());
        let table = loaded_table(tmp.path());
        let cell = table
            .select_cell(&default_identity(4), "org/model-a")
            .expect("select");
        // Rungs = batches whose (x, x+1) pair was collected.
        assert_eq!(cell.decode_rungs, vec![496, 512]);
        assert_eq!(cell.decode_batches, vec![496, 497, 512, 513, 1024]);
        assert_eq!(cell.decode_curve_bounds[&513], (1024, 4096));
        // The full-domain gate still sees the whole box.
        assert_eq!(cell.decode_domain.unwrap()[0], (496, 1024));
    }

    #[test]
    fn decode_without_pairs_has_no_rungs() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_pair(tmp.path(), &default_rows());
        let table = loaded_table(tmp.path());
        let cell = table
            .select_cell(&default_identity(4), "org/model-a")
            .expect("select");
        assert!(cell.decode_rungs.is_empty());
        assert_eq!(cell.decode_batches, vec![8, 16]);
    }

    #[test]
    fn pinned_backend_knob_cell_never_answers_auto_request() {
        // Cells collected with a pinned MoE backend must not answer an
        // "auto" request with the same path + quant/parallel identity.
        let tmp = tempfile::tempdir().expect("tmpdir");
        let rows: Vec<RowSpec> = default_rows()
            .into_iter()
            .map(|r| RowSpec {
                moe_backend: "deepep_moe",
                ..r
            })
            .collect();
        write_pair(tmp.path(), &rows);
        let err = loaded_table(tmp.path())
            .select_cell(&default_identity(4), "org/model-a")
            .unwrap_err();
        assert!(err.to_string().contains("No FPM cell matches"), "{err}");
    }

    #[test]
    fn wideep_cells_answer_wideep_requests() {
        // Schema v6 backend knobs are ordinary identity columns: data
        // collected under enable_wideep=true answers a request whose
        // identity says "True" — and only that request.
        let tmp = tempfile::tempdir().expect("tmpdir");
        let rows: Vec<RowSpec> = default_rows()
            .into_iter()
            .map(|r| RowSpec {
                enable_wideep: true,
                ..r
            })
            .collect();
        write_pair(tmp.path(), &rows);
        let table = loaded_table(tmp.path());
        let mut wideep_identity = default_identity(4);
        wideep_identity[13] = "True".to_string();
        let cell = table
            .select_cell(&wideep_identity, "org/model-a")
            .expect("select");
        assert_eq!(cell.match_identity[13], "True");
        let err = table
            .select_cell(&default_identity(4), "org/model-a")
            .unwrap_err();
        assert!(err.to_string().contains("No FPM cell matches"), "{err}");
    }
}
