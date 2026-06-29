# Collector v2 Case Files

Collector v2 plans collection from model/SM YAML instead of treating the
collector as a flat op list.

## File Layout

- `base_ops/<op>.yaml`: shared common case values and op cases every model can include.
- `models/<architecture>_cases.yaml`: architecture-specific op cases and model path aliases.
- `sm_exceptions/sm<version>_exceptions.yaml`: SM-specific exceptions applied after base-op/model cases are merged.

## Model Case Files

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

`model_path` is the default representative model. `model_paths` is the alias
list used for support-matrix lookup and targeted collection.

### Op Sections

Model files use all-framework and framework-specific op sections:

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

Use `all_frameworks_op_cases` when the op applies to every backend. Use
`framework_specific_op_cases` when only one backend should collect the op.

### Model Dimensions

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

The collector loads these values by op name and honors `COLLECTOR_MODEL_PATH`,
so support-matrix healing can request cases for one model without editing
Python. Use `model_paths` when aliases share the same dimensions, or
`model_path` for a single model-specific entry.

### Framework-Specific Model Dimensions

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

### Model Dimensions for GEMM and Attention

`gemm`, `attention_context`, `attention_generation`, and `attention_encoder`
also accept `model_case_values` blocks. Unlike MoE/MLA, these ops have a large
shared model-agnostic grid in `base_ops/`; a model entry simply *adds* its own
shape points on top of that grid. Entries are filtered by `COLLECTOR_MODEL_PATH`
when set, so a single-model run only adds that model's shapes. The shared generic
grid is still generated.

GEMM entries are GEMM shape-sweep dicts. `token_counts` (the M dimension) is
inherited from the base GEMM sweep when omitted, so a model only needs its
projection feature shapes (`feature_sizes` shorthand, or explicit
`input_feature_sizes` = K and `output_feature_sizes` = N):

```yaml
model_case_values:
  gemm:
    - model_paths: [org/MyModel]
      output_feature_sizes: [4096, 12288]
      input_feature_sizes: [7168]
      # token_counts: [...]   # optional; omit to inherit base
```

Attention entries overlay onto the per-backend merged base sweep, so caps,
`precision_cases` (trtllm/sglang) and `window_sizes` (vllm) are inherited unless
the entry overrides them. A generic `query_head_counts` is mapped to the field
each phase actually reads (uniform across phases in model YAML): context keeps
`query_head_counts`; generation fans it out to both `mha_query_head_counts` and
`xqa_query_head_counts` (explicit `mha_*`/`xqa_*` win); encoder maps it to
`head_counts`.

```yaml
model_case_values:
  attention_context:
    - model_path: org/MyModel
      query_head_counts: [64]
      kv_head_options: [8]        # `self`/`0` means num_kv_heads == query heads
      head_dims: [192]
      window_sizes: [0, 4096]
      sequence_lengths: [1024, 8192]
      precision_cases:            # optional; omit to inherit base
        - {id: bf16, fp8_kv_cache: false, fp8_context_fmha: false}
  attention_generation:
    - model_path: org/MyModel
      query_head_counts: [64]     # -> both mha and xqa head counts
      kv_head_counts: [8]
      head_dims: [192]
```

Caps behavior: a model entry may override token-budget caps in its own block
(`max_tokens_self_attention`, `max_tokens_grouped_query_attention`,
`max_mha_tokens_per_step`, `max_xqa_tokens_per_step`, `max_kv_elements`,
`max_batch_size_self_attention`, `drop_largest_sequence_for_batch_at_least`,
`min_batch_options_per_sequence`). The hardware `m_num_heads_q_per_kv`
CTA-divisibility check and the SM/version `_skip_*` guards live in the backend
collectors and always apply; they cannot be relaxed from YAML because they are
correctness guards, not budget heuristics.

## Base Op Files

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

### Explicit Case Specs

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

The `id` field above names a YAML generator spec, not a generated concrete case
ID. `token_counts` is the GEMM M dimension, `input_features` is K, and
`output_features` is N. `feature_sizes` is shorthand for using the same explicit
size list for both input and output features. For attention, `kv_head_options:
self` means `num_key_value_heads` equals `query_head_count`.

### Framework-Specific Additions

Framework-specific common op additions live under `framework_specific_op_cases`.
These cases are merged on top of the all-framework base cases for that backend.
For example, TRT-LLM can add attention `head_dims: [64, 128, 256]`, while vLLM
can add window-size sweeps. If a framework-specific case spec reuses the same
`id` as an all-framework case spec, its fields are applied as an overlay for
that backend. This lets a backend narrow a shared sweep without duplicating the
whole base spec. Framework-specific GEMM entries can also set `gemm_types` when
the backend does not derive precision cases from runtime capability.

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

## SM Exception Files

SM exception files describe hardware limits and framework-version gaps for a GPU
architecture. They use matching all-framework and framework-specific sections:

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

Use `drop: true` only when the whole op should be removed for the SM/backend.
For narrower exclusions, use a case selector on the op.

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

## Case Selectors

Case selectors are used when an op section needs to include, exclude, or mark a
subset of generated cases. Model case sections use selectors to choose what to
collect. SM exception sections use selectors to skip cases or classify runtime
failures.

Selectors are OR-based: a case matches when it satisfies any selector field. For
include selectors, matched cases are kept. For exception selectors, matched cases
are skipped or classified as expected failures. `limit` is applied after
selection.

Supported generated-case selector fields:

- `cases: all`: selects every generated case for the op.
- `cases: [...]`: in selector contexts, non-mapping entries are treated as exact generated `case_ids`.
- `case_ids`: exact generated case IDs.
- `contains`: string fragments matched against the generated case string or generated case ID.
- `indices`: zero-based generated case indices.
- `ranges`: inclusive zero-based index ranges, such as `"10-20"` or `[10, 20]`.
- `limit`: keeps only the first N cases after selection.
- `rules`: structured matches over positional collector case fields.

There is one important distinction: a dictionary under `cases` is a generation
recipe, not a selector. For example, a GEMM shape sweep dictionary tells the
collector to create many concrete GEMM cases. It should live in an include
section such as `all_frameworks_op_cases` or `framework_specific_op_cases`.

SM exceptions run after those concrete cases are generated. They should match
the generated cases with `rules`, `contains`, `case_ids`, `indices`, or
`ranges`, not by adding shape-sweep dictionaries under `cases`.

### Generated Case IDs

`case_ids` are exact IDs for fully expanded concrete collector cases. They are
created by:

```python
helper.create_test_case_id(test_case, run_func_name, full_module_name)
```

The current format is:

```text
<full_module_name>:<run_func_name>:<test_case>
```

For example:

```text
sglang.moe:run_moe_torch:MoeCommonTestCase(... tp=4, ep=16, model_name='Qwen/Qwen3-30B-A3B' ...)
```

Use `case_ids` sparingly. They are precise, but they depend on the generated
case string staying stable. Prefer `rules` for durable SM exceptions and
`contains` for quick temporary targeting.

### Structured Rules

Rules match named fields inside positional collector cases:

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

Use `fields` to name the generated tuple/list positions, then use `match` for
exact values or operators such as `lt`, `lte`, `gt`, `gte`, `ne`, `in`,
`not_in`, `contains`, `not_contains`, `mod`, or `mod_ne`.

## Adding New Coverage

Add one architecture file for a new architecture, or add a model path alias to
an existing architecture file when the model uses the same case plan. Add or
edit one `base_ops/<op>.yaml` file when common op sweeps change. Add a new op
collector only when no existing op can generate the needed points. Add one SM
exception file for a new SM version so model intent and hardware exclusions stay
separate.
