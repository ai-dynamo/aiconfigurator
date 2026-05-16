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
model_ops:
  - gemm
  - moe

all_frameworks_op_cases:
  moe:
    cases: all

framework_specific_op_cases:
  sglang:
    wideep_moe:
      cases: all
```

For simple common ops, `cases` can hold exact generator specs instead of opaque
case IDs. Base GEMM and attention use readable shape names:

```yaml
all_frameworks_op_cases:
  gemm:
    cases:
      - id: base_transformer_gemm_shape_sweep
        token_counts: [1, 2, 4, 8, 16]
        feature_sizes: [128, 256, 512]
        skip_shapes:
          - input_features: 65536
            output_features: 65536
  attention_context:
    cases:
      - id: base_attention_context_shape_sweep
        batch_sizes: [1, 2, 4, 8]
        sequence_lengths: [1, 16, 32, 64]
        query_head_counts: [1, 2, 4, 8]
        kv_head_options: [self, 1, 2, 4]
        head_dims: [128, 256]
```

`token_counts` is the GEMM M dimension, `input_features` is K, and
`output_features` is N. `feature_sizes` is shorthand for using the same explicit
size list for both input and output features. For attention, `kv_head_options:
self` means `num_key_value_heads` equals `query_head_count`.

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
