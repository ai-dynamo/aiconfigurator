# Collector v2 Case Files

Collector v2 plans collection from model/SM YAML instead of treating the
collector as a flat op list.

- `base_ops/<op>.yaml`: shared common case values and op cases every model can include.
- `models/<architecture>_cases.yaml`: architecture-specific op cases and model
  path aliases.
- `sm_exceptions/sm<version>_exceptions.yaml`: SM-specific exceptions applied after base-op/model
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
    - model_paths:
        - Qwen/Qwen3-235B-A22B
        - Qwen/Qwen3-235B-A22B-FP8
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
model without editing Python. Use `model_paths` when aliases share the same
dimensions, or `model_path` for a single model-specific entry.

Shared sweep recipes live in per-op files under `base_ops/<op>.yaml`. For
cross-model ops such as MoE, MLA, Mamba2, GDN, and MHC, the base op file owns
the token counts, batch/sequence sweeps, parallelism sizes, routing
distributions, and generator constraints:

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

Framework-specific model dimensions live in `framework_specific_model_case_values`.
Use this when a case should be available to one backend without expanding the
all-framework common generator:

```yaml
framework_specific_model_case_values:
  vllm_xpu:
    moe:
      - model_path: Qwen/Qwen1.5-MoE-A2.7B
        hidden_size: 2048
        inter_size: 1408
        topk: 4
        num_experts: 60
        sweep: small
        activation: silu
```

For simple common ops, `cases` can hold exact generator specs instead of opaque
case IDs. Base GEMM, attention, and MLA BMM keep readable shape names in their
own `base_ops/<op>.yaml` files:

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
size list for both input and output features. Framework-specific GEMM overrides
can narrow those shape sweeps and, when the backend does not derive precision
cases from runtime capability, set `gemm_types`. For attention, `kv_head_options:
self` means `num_key_value_heads` equals `query_head_count`.
For `mla_bmm_gen_pre` and `mla_bmm_gen_post`, `token_counts`, `head_counts`,
`dtypes`, `num_warmups`, and `num_runs` define the auxiliary MLA generation BMM
micro-collector grids shared by SGLang and TRT-LLM; each collector still filters
runtime-unsupported dtypes before benchmarking. Full MLA/DSA module collectors
read `common_case_values.mla_module` for the inner batch/sequence/head sweeps and
reduced top-level ModelRunner subprocess sweep; framework-specific variants such
as `common_case_values.mla_module_trtllm` can override only the sweep metadata
without creating a new collectable op. Their per-model attention type, native
head count, WideEP eligibility, and architecture come from
`model_case_values.mla_module`. Precision combinations can include `min_sm` and
`phases` when a backend has hardware-gated context/generation support.

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
whole top-level case. Normal op exception rules are applied before collection,
so matching top-level cases are never queued. `known_exceptions` is a runtime
safety net: if a queued case still fails and matches one of those records,
`collect.py` logs the SM/framework reason, records the task as
`expected_failed` in the resume checkpoint, and continues without counting it as
a collector error.

Add one architecture file for a new architecture, or add a model path alias to
an existing architecture file when the model uses the same case plan. Add or
edit one `base_ops/<op>.yaml` file when common op sweeps change. Add a new op
collector only when no existing op can generate the needed points. Add one SM
exception file for a new SM version so model intent and hardware exclusions stay
separate.
