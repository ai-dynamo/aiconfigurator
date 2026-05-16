# Collector v2 Case Files

Collector v2 plans collection from model/GPU YAML instead of treating the
collector as a flat op list.

- `base_model_cases.yaml`: shared cases every model can include.
- `models/*_cases.yaml`: model-specific op cases.
- `gpus/*_exceptions.yaml`: GPU-specific exceptions applied after base/model
  cases are merged.

Model files use two sections:

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

Add one model file for a new model, and add a new op collector only when no
existing op can generate the needed points. Add one GPU exception file for a new
GPU so model intent and hardware exclusions stay separate.
