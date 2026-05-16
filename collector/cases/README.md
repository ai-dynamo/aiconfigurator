# Collector v2 Case Files

Collector v2 plans collection from model/SM YAML instead of treating the
collector as a flat op list.

- `base_op_cases.yaml`: shared common case values and op cases every model can include.
- `models/<architecture>_cases.yaml`: architecture-specific op cases and model
  path aliases.
- `sms/sm<version>_exceptions.yaml`: SM-specific exceptions applied after base-op/model
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

Model-specific op dimensions live in `model_case_values`. These values replace
the old Python config lists:

```yaml
model_case_values:
  moe:
    - model_path: Qwen/Qwen3-235B-A22B
      hidden_size: 4096
      inter_size: 1536
      topk: 8
      num_experts: 128
  mla:
    - model_path: deepseek-ai/DeepSeek-V3
      num_heads: 128
      q_lora_rank: 1536
      kv_lora_rank: 512
      qk_nope_head_dim: 128
      qk_rope_head_dim: 64
      v_head_dim: 128
```

The collector loads these model-specific values by op name and honors
`COLLECTOR_MODEL_PATH`, so support-matrix healing can request cases for one
model without editing Python.

Shared sweep recipes live in `base_op_cases.yaml` under
`common_case_values`. For cross-model ops such as MoE, MLA, Mamba2, GDN, and
MHC, the base op file owns the token counts, batch/sequence sweeps, parallelism
sizes, routing distributions, and generator constraints:

```yaml
common_case_values:
  moe:
    token_counts: [1, 2, 4, 8, 16]
    tensor_parallel_sizes: [1, 2, 4, 8]
    expert_parallel_sizes: [1, 2, 4, 8]
    gpu_counts: [1, 2, 4, 8]
    token_expert_distributions:
      - name: balanced
        power_law_alpha: 0.0
      - name: power_law
        power_law_alpha: 1.01
```

The MoE Python generator only combines those shared sweep values with each
model's `hidden_size`, `inter_size`, `topk`, and `num_experts`. The same pattern
applies to MLA, Mamba2, GDN, and MHC: model YAML stores model dimensions, while
base op YAML stores the reusable sweep policy.

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

Framework-specific common op overrides live under `framework_specific_op_cases`
with the same case `id`. For example, TRT-LLM adds attention `head_dims: [64,
128, 256]`, while vLLM narrows head-count and window-size sweeps. The Python
collectors should read these YAML-backed specs and only keep backend runtime
filters in code.

SM exception files use matching exception sections:

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
They can also use structured `rules` for positional collector test cases:

```yaml
framework_specific_op_exceptions:
  vllm:
    gemm:
      rules:
        - reason_type: hardware_unsupported
          reason: Blackwell vLLM block-FP8 GEMM requires token_count to be divisible by 4.
          fields: [gemm_type, token_count, output_features, input_features]
          match:
            gemm_type: fp8_block
            token_count:
              mod_ne: [4, 0]
```

`reason_type` should be `hardware_unsupported` for SM capability/kernel-shape
limits and `framework_version_unsupported` for a known runtime or framework
version gap. Version-scoped rules can add `version_prefixes`, and computed
conditions can use `product`, `ratio`, `floor_div`, or `field` comparisons.
For skips that happen inside a subprocess after the top-level case is selected,
record the extracted skip under `known_exceptions` instead of over-dropping the
whole top-level case.

Add one architecture file for a new architecture, or add a model path alias to
an existing architecture file when the model uses the same case plan. Add a new
op collector only when no existing op can generate the needed points. Add one
SM exception file for a new SM version so model intent and hardware exclusions stay
separate.
