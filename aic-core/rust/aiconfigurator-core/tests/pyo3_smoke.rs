// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust -> Python embedding smoke test.
//!
//! Proves that `aiconfigurator-core` links libpython and can start a Python
//! interpreter via PyO3's `auto-initialize`. This is the linkage that
//! `build_aic_engine` relies on. It does NOT import the
//! `aiconfigurator_core` extension itself (that `.so` is not on `sys.path`
//! during `cargo test`); it only exercises the embedded interpreter.

#![cfg(feature = "embed-python")]

use pyo3::prelude::*;
use pyo3::types::PyModule;

#[test]
fn embeds_python_interpreter() {
    Python::with_gil(|py| {
        // Import a stdlib module to confirm the interpreter is live.
        let sys = PyModule::import(py, "sys").expect("import sys");
        let version: String = sys
            .getattr("version")
            .expect("sys.version")
            .extract()
            .expect("version as str");
        assert!(!version.is_empty(), "sys.version should be non-empty");

        // Evaluate a trivial expression and assert the result.
        let result: i64 = py
            .eval(c"6 * 7", None, None)
            .expect("eval 6 * 7")
            .extract()
            .expect("result as i64");
        assert_eq!(result, 42);
    });
}
