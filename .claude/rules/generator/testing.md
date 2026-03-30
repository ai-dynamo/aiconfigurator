# Generator Testing Reference

Generator testing has unique challenges: combinatorial output space, no backend
runtime in CI, silent regressions, and test coverage gaps. This reference defines
the testing strategy.

## When to Use This Reference

- After any generator code change (templates, rules, schemas, rendering)
- Adding test coverage for untested generator components
- Setting up golden output snapshots
- Running the generator validator

## Test Strategy Layers

```text
Layer 1: UNIT TESTS (fast, always run)
  - Test individual functions in isolation
  - Mock: schema loading, template rendering
  - Files: tests/unit/generator/test_*.py

Layer 2: INTEGRATION TESTS (medium, run on PR)
  - Test full pipeline: input -> artifacts
  - Compare output against golden snapshots
  - No external dependencies

Layer 3: VALIDATOR TESTS (slow, run in specialized CI)
  - Import actual backend engine modules
  - Compare generated flags against engine API schemas
  - Requires docker images with backends installed
```

## Workflow

### 1. Determine Test Layer

- Pure logic change (expression, calculation): Unit test
- Template/rule/schema change (output-affecting): Integration + golden snapshot
- New parameter or flag: Validator test

### 2. Write Unit Tests

Location: `tests/unit/generator/`

For aggregators/API changes:
```python
def test_new_parameter_default(self):
    result = collect_generator_params(
        service={"model_path": "test/model", "served_model_name": "test"},
        k8s={"k8s_namespace": "test"},
        backend="trtllm"
    )
    assert result["SectionConfig"]["new_param"] == expected_value
```

For rule evaluation:
```python
def test_moe_tp_remapping(self):
    params = {"tensor_parallel_size": 1, "moe_tensor_parallel_size": 2,
              "moe_expert_parallel_size": 4}
    model_config = {"is_moe": True}
    result = apply_rule_plugins("trtllm", params, model_config=model_config)
    assert result["agg"]["tensor_parallel_size"] == 8  # 2 * 4
```

### 3. Write Golden Snapshot Tests

Purpose: Catch any unintended output change.

1. Generate reference outputs for key configurations:
   - Minimal config (just model_path + backend)
   - Full disagg config (prefill + decode workers)
   - MoE model config
   - Each backend x version combination

2. Store as golden files:
   ```
   tests/golden/generator/<backend>/<version>/<mode>/
     cli_args.txt
     k8s_deploy.yaml
     run.sh
   ```

3. Test compares current output against golden:
   ```python
   def test_trtllm_agg_golden(self):
       output = generate_backend_artifacts(params, backend="trtllm", ...)
       golden = read_golden("trtllm/1.3.0rc5/agg/cli_args.txt")
       assert output["cli_args"] == golden
   ```

4. Update goldens intentionally:
   ```bash
   UPDATE_GOLDEN=1 pytest tests/golden/ -k trtllm
   git diff tests/golden/  # review changes
   ```

### 4. Use Generator Validator

When available (specialized CI or local with backend images):

```bash
# Validate generated TRT-LLM config against engine API
python tools/generator_validator/validator.py --backend trtllm --path output/

# Validate vLLM CLI args
python tools/generator_validator/validator.py --backend vllm --path output/
```

### 5. Test Edge Cases

Always test these scenarios:

- [ ] MoE model (`is_moe=True`, `moe_tensor_parallel_size` set)
- [ ] Disagg mode (prefill + decode workers)
- [ ] Agg mode (single worker)
- [ ] Speculative decoding (`ModelConfig.nextn > 0`)
- [ ] Prefix caching (`ModelConfig.prefix > 0`)
- [ ] Minimal config (only required fields)
- [ ] PVC configuration (`k8s_pvc_name` set, `k8s_hf_home` auto-derived)
- [ ] Each backend (trtllm, vllm, sglang)
- [ ] Benchmark mode (`rule=benchmark`)

## Test File Organization

```text
tests/
  unit/
    generator/
      test_aggregators.py       # Parameter collection, defaults
      test_naive.py             # Naive TP calculation, RFC-1123
      test_trtllm_cli_args.py   # TRT-LLM specific rendering
      test_vllm_cli_args.py     # vLLM specific rendering
      test_sglang_cli_args.py   # SGLang specific rendering
      test_rule_engine.py       # Rule DSL evaluation
      test_schema_defaults.py   # Default expression evaluation
      test_config_mapping.py    # Parameter mapping correctness
  golden/
    generator/
      trtllm/<version>/<mode>/  # Golden outputs per combination
      vllm/<version>/<mode>/
      sglang/<version>/<mode>/
```

## Checklist

```text
[ ] Determine test layer (unit / integration / validator)
[ ] Write unit tests for changed logic
[ ] Update or create golden snapshots for output-affecting changes
[ ] Test MoE, disagg, speculative decoding edge cases
[ ] Test all affected backends
[ ] Run generator validator if backend images available
[ ] Verify test passes in CI
```
