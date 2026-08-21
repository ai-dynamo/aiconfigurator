// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared test fixtures for power/energy parquet oracles.

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parquet::data_type::{BoolType, ByteArray, ByteArrayType, DoubleType, Int64Type};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::{SerializedFileWriter, SerializedRowGroupWriter};
use parquet::schema::parser::parse_message_type;

use crate::common::system_spec::SystemSpec;

pub enum Col<'a> {
    Str(&'static str, Vec<&'a str>),
    Bool(&'static str, Vec<bool>),
    I64(&'static str, Vec<i64>),
    F64(&'static str, Vec<f64>),
}

impl Col<'_> {
    fn name(&self) -> &'static str {
        match self {
            Col::Str(name, _) | Col::Bool(name, _) | Col::I64(name, _) | Col::F64(name, _) => name,
        }
    }

    fn len(&self) -> usize {
        match self {
            Col::Str(_, values) => values.len(),
            Col::Bool(_, values) => values.len(),
            Col::I64(_, values) => values.len(),
            Col::F64(_, values) => values.len(),
        }
    }

    fn parquet_type(&self) -> String {
        match self {
            Col::Str(name, _) => format!("REQUIRED BYTE_ARRAY {name} (UTF8);"),
            Col::Bool(name, _) => format!("REQUIRED BOOLEAN {name};"),
            Col::I64(name, _) => format!("REQUIRED INT64 {name};"),
            Col::F64(name, _) => format!("REQUIRED DOUBLE {name};"),
        }
    }
}

fn write_column<T: parquet::data_type::DataType>(
    rg: &mut SerializedRowGroupWriter<'_, File>,
    values: &[T::T],
) {
    let mut col = rg.next_column().unwrap().unwrap();
    col.typed::<T>().write_batch(values, None, None).unwrap();
    col.close().unwrap();
}

pub fn write_parquet(path: &Path, cols: &[Col<'_>]) {
    assert!(!cols.is_empty(), "test parquet needs at least one column");
    let len = cols[0].len();
    for col in cols {
        assert_eq!(col.len(), len, "column {} length mismatch", col.name());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }

    let fields = cols
        .iter()
        .map(Col::parquet_type)
        .collect::<Vec<_>>()
        .join("\n");
    let schema =
        Arc::new(parse_message_type(&format!("message fixture {{\n{fields}\n}}")).unwrap());
    let file = File::create(path).unwrap();
    let mut writer =
        SerializedFileWriter::new(file, schema, Arc::new(WriterProperties::builder().build()))
            .unwrap();
    let mut rg = writer.next_row_group().unwrap();
    for col in cols {
        match col {
            Col::Str(_, values) => {
                let values = values
                    .iter()
                    .map(|value| ByteArray::from(*value))
                    .collect::<Vec<_>>();
                write_column::<ByteArrayType>(&mut rg, &values);
            }
            Col::Bool(_, values) => write_column::<BoolType>(&mut rg, values),
            Col::I64(_, values) => write_column::<Int64Type>(&mut rg, values),
            Col::F64(_, values) => write_column::<DoubleType>(&mut rg, values),
        }
    }
    rg.close().unwrap();
    writer.close().unwrap();
}

pub fn write_energy_systems_root(root: &Path) -> PathBuf {
    let data_root = root.join("data/testsys/vllm/1.0");
    fs::create_dir_all(&data_root).unwrap();
    fs::write(root.join("testsys.yaml"), ENERGY_TEST_SYSTEM_YAML).unwrap();
    data_root
}

pub fn energy_test_spec() -> SystemSpec {
    serde_yaml::from_str(ENERGY_TEST_SYSTEM_YAML).unwrap()
}

const ENERGY_TEST_SYSTEM_YAML: &str = r#"
data_dir: data/testsys
gpu:
  mem_bw: 1000000000000
  mem_capacity: 85899345920
  bfloat16_tc_flops: 100000000000000
  int8_tc_flops: 200000000000000
  fp8_tc_flops: 200000000000000
  fp4_tc_flops: 400000000000000
  power: 700
  sm_version: 100
node:
  num_gpus_per_node: 8
  inter_node_bw: 50000000000
  intra_node_bw: 900000000000
  pcie_bw: 64000000000
  p2p_latency: 0.00001
misc:
  other_mem: 0
  nccl_version: test
"#;
