# Collector v2 Case Files

Collector v2 plans collection from model/GPU YAML instead of treating the
collector as a flat op list.

- `base_model_cases.yaml`: shared cases every model can include.
- `models/<architecture>_cases.yaml`: architecture-specific op cases and model
  path aliases.
- `gpus/*_exceptions.yaml`: GPU-specific exceptions applied after base/model
  cases are merged.

Model files are keyed by HuggingFace architecture name and list every model path
alias that should resolve to that architecture plan:

```yaml
schema_version: 1
architecture: Qwen3MoeForCausalLM
model_path: Qwen/Qwen3-235B-A22B
model_paths:
  - Qwen/Qwen3-30B-A3B
  - Qwen/Qwen3-235B-A22B
include_base: true
```

They use two case sections:

```yaml
all_frameworks_op_cases:
  moe:
    cases: all

framework_specific_op_cases:
  sglang:
    wideep_moe:
      cases: all
```

GPU exception files use matching exception sections:

```yaml
all_frameworks_op_exceptions:
  attention_generation:
    drop: true

framework_specific_op_exceptions:
  sglang:
    wideep_moe:
      contains:
        - "tp=32"
```

Case selectors can use `cases: all`, exact `case_ids`, string `contains`
matches, `indices`, `ranges`, and `limit`. Exact case IDs are the stable IDs
created by `helper.create_test_case_id(test_case, run_func_name, full_module_name)`.

Add one architecture file for a new architecture, or add a model path alias to
an existing architecture file when the model uses the same case plan. Add a new
op collector only when no existing op can generate the needed points. Add one
GPU exception file for a new GPU so model intent and hardware exclusions stay
separate.
