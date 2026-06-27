# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from collector import case_generator, model_cases
from collector.planner import PlanContext, RuleSource, compile_population, legacy_rule, use_case_catalog
from collector.planner.schemas import register_attention_schemas


def _write_base_cases(path: Path) -> None:
    path.mkdir()
    (path / "gemm.yaml").write_text(
        """
schema_version: 1
model_ops: [gemm]
all_frameworks_op_cases:
  gemm:
    cases: all
""",
        encoding="utf-8",
    )


def _write_model(path: Path, *, architecture: str, model_path: str, heads: int, schema_version: int = 1) -> None:
    path.write_text(
        f"""
schema_version: {schema_version}
architecture: {architecture}
model_path: {model_path}
base_ops: [gemm]
model_case_values:
  attention:
    - model_path: {model_path}
      num_attention_heads: {heads}
      num_key_value_heads: 8
      head_dim: 128
      window_sizes: [0]
      tensor_parallel_sizes: [1, 2, 4, 8]
""",
        encoding="utf-8",
    )


def test_catalog_is_parsed_once_and_reused_by_case_generator(tmp_path, monkeypatch):
    base_dir = tmp_path / "base_ops"
    model_dir = tmp_path / "models"
    _write_base_cases(base_dir)
    model_dir.mkdir()
    _write_model(
        model_dir / "Qwen_cases.yaml",
        architecture="QwenArchitecture",
        model_path="test/qwen",
        heads=32,
    )
    _write_model(
        model_dir / "DeepSeek_cases.yaml",
        architecture="DeepSeekArchitecture",
        model_path="test/deepseek",
        heads=64,
    )

    monkeypatch.setattr(model_cases, "BASE_OP_CASES_DIR", base_dir)
    monkeypatch.setattr(model_cases, "MODEL_CASES_DIR", model_dir)
    reads: dict[Path, int] = {}
    original_load = model_cases.load_yaml_file

    def counted_load(path):
        resolved = Path(path).resolve()
        reads[resolved] = reads.get(resolved, 0) + 1
        return original_load(path)

    monkeypatch.setattr(model_cases, "load_yaml_file", counted_load)
    catalog = model_cases.load_case_catalog(backend="sglang", model_path="test/qwen")
    plan = model_cases.build_collection_case_plan(backend="sglang", catalog=catalog)

    with use_case_catalog(plan.catalog):
        values = case_generator._model_case_values("attention")

    assert [value["model_path"] for value in values] == ["test/qwen"]
    assert set(reads.values()) == {1}


def test_active_catalog_does_not_leak_between_targeted_plans(tmp_path, monkeypatch):
    base_dir = tmp_path / "base_ops"
    model_dir = tmp_path / "models"
    _write_base_cases(base_dir)
    model_dir.mkdir()
    _write_model(
        model_dir / "Qwen_cases.yaml",
        architecture="QwenArchitecture",
        model_path="test/qwen",
        heads=32,
    )
    _write_model(
        model_dir / "DeepSeek_cases.yaml",
        architecture="DeepSeekArchitecture",
        model_path="test/deepseek",
        heads=64,
    )
    monkeypatch.setattr(model_cases, "BASE_OP_CASES_DIR", base_dir)
    monkeypatch.setattr(model_cases, "MODEL_CASES_DIR", model_dir)

    observed = []
    for model_path in ("test/qwen", "test/deepseek"):
        plan = model_cases.build_collection_case_plan(backend="sglang", model_path=model_path)
        with use_case_catalog(plan.catalog):
            values = case_generator._model_case_values("attention")
            observed.append((model_path, values[0]["num_attention_heads"]))

    assert observed == [("test/qwen", 32), ("test/deepseek", 64)]


def test_schema_v2_population_rules_are_additive_and_keep_provenance(tmp_path):
    base_dir = tmp_path / "base_ops"
    _write_base_cases(base_dir)
    model_path = tmp_path / "RuleArchitecture_cases.yaml"
    model_path.write_text(
        """
schema_version: 2
architecture: RuleArchitecture
model_path: test/rules
base_ops: [gemm]
population_rules:
  gemm:
    - id: base_delta
      cases:
        - [bf16, 1, 128, 128]
        - [bf16, 2, 128, 128]
    - id: alias_delta
      cases:
        - [bf16, 1, 128, 128]
        - [fp8, 1, 128, 128]
""",
        encoding="utf-8",
    )
    plan = model_cases.build_collection_case_plan(
        backend="sglang",
        base_cases_path=str(base_dir),
        model_cases_path=str(model_path),
    )
    op_plan = plan.op_cases["gemm"]
    context = PlanContext(backend="sglang", op="gemm", perf_file="gemm_perf.txt")
    result = compile_population(
        [legacy_rule([], source=RuleSource("empty_legacy")), *op_plan.population_rules],
        context,
    )

    assert [case.payload for case in result.cases] == [
        ["bf16", 1, 128, 128],
        ["bf16", 2, 128, 128],
        ["fp8", 1, 128, 128],
    ]
    assert result.report.duplicate_invocations == 1
    assert [source.rule_id for source in result.cases[0].provenance] == ["base_delta", "alias_delta"]


def test_schema_v1_rejects_population_rules(tmp_path):
    base_dir = tmp_path / "base_ops"
    _write_base_cases(base_dir)
    model_path = tmp_path / "RuleArchitecture_cases.yaml"
    model_path.write_text(
        """
schema_version: 1
architecture: RuleArchitecture
model_path: test/rules
base_ops: [gemm]
population_rules:
  gemm:
    - id: invalid
      cases: [[bf16, 1, 128, 128]]
""",
        encoding="utf-8",
    )

    try:
        model_cases.build_collection_case_plan(
            backend="sglang",
            base_cases_path=str(base_dir),
            model_cases_path=str(model_path),
        )
    except ValueError as error:
        assert "schema_version: 2" in str(error)
    else:
        raise AssertionError("schema-v1 population rules must be rejected")


def test_schema_v2_semantic_attention_cases_compile_to_runtime_abi_and_dedupe(tmp_path):
    base_dir = tmp_path / "base_ops"
    base_dir.mkdir()
    (base_dir / "attention_generation.yaml").write_text(
        """
schema_version: 1
model_ops: [attention_generation]
all_frameworks_op_cases:
  attention_generation:
    cases: all
""",
        encoding="utf-8",
    )
    model_path = tmp_path / "AttentionArchitecture_cases.yaml"
    model_path.write_text(
        """
schema_version: 2
architecture: AttentionArchitecture
model_path: test/attention
base_ops: [attention_generation]
population_rules:
  attention_generation:
    - id: canonical_names
      cases:
        - batch_size: 2
          sequence_length: 15
          num_heads: 8
          num_kv_heads: 8
          head_dim: 64
          use_fp8_kv_cache: true
          window_size: 0
    - id: semantic_aliases
      cases:
        - batch_size: 2
          sequence_length: 15
          query_heads: 8
          kv_heads: 8
          head_dim: 64
          fp8_kv_cache: true
          window_size: 0
""",
        encoding="utf-8",
    )
    plan = model_cases.build_collection_case_plan(
        backend="sglang",
        base_cases_path=str(base_dir),
        model_cases_path=str(model_path),
    )
    context = PlanContext(
        backend="sglang",
        op="attention_generation",
        perf_file="generation_attention_perf.txt",
    )
    register_attention_schemas()

    result = compile_population(plan.op_cases["attention_generation"].population_rules, context)

    assert [case.payload for case in result.cases] == [(2, 15, 8, 8, 64, True, False, False, 0)]
    assert result.report.duplicate_invocations == 1
    assert [source.rule_id for source in result.cases[0].provenance] == [
        "canonical_names",
        "semantic_aliases",
    ]


def test_full_plan_keeps_each_model_documents_conditioned_population_rules(tmp_path, monkeypatch):
    base_dir = tmp_path / "base_ops"
    model_dir = tmp_path / "models"
    _write_base_cases(base_dir)
    model_dir.mkdir()
    for name, architecture, model_path, marker in (
        ("A", "ArchitectureA", "model/A", 1),
        ("B", "ArchitectureB", "model/B", 2),
    ):
        (model_dir / f"{name}_cases.yaml").write_text(
            f"""
schema_version: 2
architecture: {architecture}
model_path: {model_path}
base_ops: [gemm]
population_rules:
  gemm:
    - id: {name.lower()}_path_rule
      when:
        model_paths: [{model_path}]
        model_architectures: [{architecture}]
      cases: [[bf16, {marker}, 128, 128]]
""",
            encoding="utf-8",
        )
    monkeypatch.setattr(model_cases, "MODEL_CASES_DIR", model_dir)

    plan = model_cases.build_collection_case_plan(
        backend="sglang",
        base_cases_path=str(base_dir),
        full=True,
    )

    assert [rule.source.rule_id for rule in plan.op_cases["gemm"].population_rules] == [
        "a_path_rule",
        "b_path_rule",
    ]
