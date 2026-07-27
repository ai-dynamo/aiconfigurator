// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin parquet reader shared by every `perf_database/*.rs` table loader.
//!
//! Design notes:
//! - Each loader resolves column names to integer positions ONCE per file
//!   load via `PerfReader::col`, then iterates rows with positional accessors.
//!   That avoids the per-row string-keyed lookup the `Row::get_*` API
//!   forces when callers go through `(name, field)` pairs.
//! - Accessors borrow strings out of the in-memory parquet row (no copy);
//!   loaders that need to store a string in a BTreeMap key call
//!   `PerfRow::str_owned`.
//! - Integer columns in the AIC perf tables are stored as INT64 in parquet
//!   but always fit in u32 in practice. `PerfRow::u32` narrows and errors
//!   on overflow so a perf-DB collection bug surfaces loudly instead of
//!   wrapping silently.
//! - All errors carry the source file path so the failure message points at
//!   the exact perf-DB file (mirrors the path-context error contract used
//!   by the perf_database layer).

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{Row, RowAccessor};

use crate::common::error::AicError;

const LFS_POINTER_PREFIX: &[u8] = b"version https://git-lfs";

/// Opens a parquet file and pre-caches column names so each loader resolves
/// `col("name") -> position` once before the row loop.
pub struct PerfReader {
    file_reader: SerializedFileReader<File>,
    path: PathBuf,
    column_names: Vec<String>,
}

impl PerfReader {
    pub fn open(path: &Path) -> Result<Self, AicError> {
        let mut file = File::open(path).map_err(|source| AicError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        // Surface unresolved git-lfs pointers with a clear message before
        // the parquet reader emits its less-specific "Invalid Parquet"
        // error. Parquet starts with the magic bytes "PAR1"; an unresolved
        // LFS pointer starts with "version https://git-lfs".
        let mut head = [0u8; LFS_POINTER_PREFIX.len()];
        let read = file.read(&mut head).map_err(|source| AicError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if read >= LFS_POINTER_PREFIX.len() && &head == LFS_POINTER_PREFIX {
            return Err(AicError::PerfDatabase(format!(
                "perf file is an unresolved git-lfs pointer: {}; run `git lfs pull`",
                path.display()
            )));
        }
        // Reopen since we consumed the head bytes (the parquet reader needs
        // to start from offset 0 to read the footer offset).
        let file = File::open(path).map_err(|source| AicError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let file_reader = SerializedFileReader::new(file).map_err(|source| AicError::Parquet {
            path: path.to_path_buf(),
            source,
        })?;
        let column_names = file_reader
            .metadata()
            .file_metadata()
            .schema_descr()
            .columns()
            .iter()
            .map(|c| c.name().to_string())
            .collect();
        Ok(Self {
            file_reader,
            path: path.to_path_buf(),
            column_names,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the position of `name` in the parquet schema. Errors when the
    /// column is absent, with the surviving column names in the message so
    /// schema drift is visible without opening the file.
    pub fn col(&self, name: &str) -> Result<usize, AicError> {
        self.column_names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| {
                AicError::PerfDatabase(format!(
                    "parquet column {name:?} missing at {} (have: [{}])",
                    self.path.display(),
                    self.column_names.join(", ")
                ))
            })
    }

    /// Same as `col` but returns `None` instead of erroring — for columns
    /// that are present in some perf-DB versions and absent in others.
    pub fn col_optional(&self, name: &str) -> Option<usize> {
        self.column_names.iter().position(|n| n == name)
    }

    /// Iterate over every row. Errors are surfaced per-row so loaders can
    /// distinguish a single corrupt row from a wholesale file failure.
    pub fn rows(&self) -> Result<impl Iterator<Item = Result<PerfRow, AicError>> + '_, AicError> {
        let row_iter = self.file_reader.get_row_iter(None).map_err(|source| AicError::Parquet {
            path: self.path.clone(),
            source,
        })?;
        let path = self.path.clone();
        Ok(row_iter.map(move |r| {
            r.map(|row| PerfRow {
                row,
                path: path.clone(),
            })
            .map_err(|source| AicError::Parquet {
                path: path.clone(),
                source,
            })
        }))
    }
}

/// Owned parquet row with positional, typed accessors. One Row's worth of
/// `(name, field)` tuples lives on the heap while it's in scope; loaders
/// consume one Row per iteration.
pub struct PerfRow {
    row: Row,
    path: PathBuf,
}

impl PerfRow {
    /// Borrowed string view into the row. Use this when the value is parsed
    /// or compared but not stored.
    pub fn str(&self, col: usize) -> Result<&str, AicError> {
        self.row.get_string(col).map(String::as_str).map_err(|source| AicError::Parquet {
            path: self.path.clone(),
            source,
        })
    }

    /// Cloned String for BTreeMap keys.
    pub fn str_owned(&self, col: usize) -> Result<String, AicError> {
        self.str(col).map(|s| s.to_string())
    }

    /// Narrow INT64 to u32; the AIC perf tables always fit. Errors if not.
    pub fn u32(&self, col: usize) -> Result<u32, AicError> {
        let v = self.row.get_long(col).map_err(|source| AicError::Parquet {
            path: self.path.clone(),
            source,
        })?;
        u32::try_from(v).map_err(|_| {
            AicError::PerfDatabase(format!(
                "parquet column[{col}] = {v} does not fit in u32 at {}",
                self.path.display()
            ))
        })
    }

    /// Optional INT64 → u32. Returns None when the cell is missing/null.
    pub fn u32_optional(&self, col: Option<usize>) -> Result<Option<u32>, AicError> {
        let Some(idx) = col else {
            return Ok(None);
        };
        // `get_long` errors when the field is null; treat NullFound as absence.
        match self.row.get_long(idx) {
            Ok(v) => u32::try_from(v).map(Some).map_err(|_| {
                AicError::PerfDatabase(format!(
                    "parquet column[{idx}] = {v} does not fit in u32 at {}",
                    self.path.display()
                ))
            }),
            Err(parquet::errors::ParquetError::General(_)) => Ok(None),
            Err(source) => Err(AicError::Parquet {
                path: self.path.clone(),
                source,
            }),
        }
    }

    /// INT64 → u64. Used by columns that legitimately need the full range
    /// (e.g. NCCL `message_size`).
    pub fn u64(&self, col: usize) -> Result<u64, AicError> {
        let v = self.row.get_long(col).map_err(|source| AicError::Parquet {
            path: self.path.clone(),
            source,
        })?;
        u64::try_from(v).map_err(|_| {
            AicError::PerfDatabase(format!(
                "parquet column[{col}] = {v} is negative; expected u64 at {}",
                self.path.display()
            ))
        })
    }

    pub fn f64(&self, col: usize) -> Result<f64, AicError> {
        self.row.get_double(col).map_err(|source| AicError::Parquet {
            path: self.path.clone(),
            source,
        })
    }

    /// BOOLEAN column, with fallbacks mirroring Python's `_to_bool` string
    /// coercion (`str(value).strip().lower() in {"1","true","yes","y"}`) for
    /// files that store flags as INT64 or strings.
    pub fn bool(&self, col: usize) -> Result<bool, AicError> {
        if let Ok(v) = self.row.get_bool(col) {
            return Ok(v);
        }
        if let Ok(v) = self.row.get_long(col) {
            return Ok(v == 1);
        }
        let s = self.str(col)?;
        Ok(matches!(s.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "y"))
    }

    /// Optional double. Returns None when the column lookup is None OR the
    /// cell is null. Used by dispatch tables whose split-latency columns
    /// can legitimately be absent for one mode of the schema (DeepEP normal
    /// vs LL).
    pub fn f64_optional(&self, col: Option<usize>) -> Result<Option<f64>, AicError> {
        let Some(idx) = col else {
            return Ok(None);
        };
        match self.row.get_double(idx) {
            Ok(v) => Ok(Some(v)),
            Err(parquet::errors::ParquetError::General(_)) => Ok(None),
            Err(source) => Err(AicError::Parquet {
                path: self.path.clone(),
                source,
            }),
        }
    }

    /// Borrowed string accessor for columns that may be absent or null.
    /// Returns None if the column lookup is None OR the cell is null.
    pub fn str_optional(&self, col: Option<usize>) -> Result<Option<&str>, AicError> {
        let Some(idx) = col else {
            return Ok(None);
        };
        match self.row.get_string(idx) {
            Ok(s) => Ok(Some(s.as_str())),
            Err(parquet::errors::ParquetError::General(_)) => Ok(None),
            Err(source) => Err(AicError::Parquet {
                path: self.path.clone(),
                source,
            }),
        }
    }
}
