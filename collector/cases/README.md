# Collector v2 Case Files

Collector v2 plans collection from model/SM YAML instead of treating the
collector as a flat op list.

The proposed common population, physical-key deduplication, compatibility
manifest, and incremental migration contract is documented in
[`docs/perf_database/collector-v2-population-design.md`](../../docs/perf_database/collector-v2-population-design.md).

## File Layout

- `base_ops/<op>.yaml`: shared recipe library. Only recipes named by a model's
  `base_ops` (or declared universal through the base file's `model_ops`) are activated.
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
base_ops:
  - attention_context
  - attention_generation
  - gemm
```

`model_path` is the default representative model. The top-level `model_paths`
list resolves support-matrix names to this architecture plan; it does not by
itself multiply kernel cases.

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
Python. There are two deliberately different multi-name forms:

- `model_aliases` resolves multiple artifact names to one canonical physical
  case. Use it for base/FP8/NVFP4 checkpoints when quantization is already an
  independent collector dimension and the kernel shape is identical.
- `model_paths` expands one physical case per path. Use it only when the model
  name changes runtime behavior, quantization policy, activation, or module
  loading (for example a collector that actually loads each checkpoint).

Keeping these meanings separate prevents checkpoint suffixes from multiplying
the same shape by every independently swept quantization mode.

### Structural Attention Profiles

Attention correlations belong to the model, not to a global Cartesian product.
Store a native topology under `model_case_values.attention`:

```yaml
model_case_values:
  attention:
    - model_path: openai/gpt-oss-120b
      model_aliases: [openai/gpt-oss-20b]
      num_attention_heads: 64
      num_key_value_heads: 8
      head_dim: 64
      window_sizes: [0, 128]
      tensor_parallel_sizes: [1, 2, 4, 8, 16, 32, 64]
```

The generator expands only valid TP shards of that tuple. It never crosses one
model's query/KV heads with another model's head dimension or window. A targeted
run with an explicit profile uses only that model profile; models not yet
migrated fall back to the collector-v1 compatibility profiles. Full/raw runs
combine the compatibility profiles with all model deltas.

Treat that last rule as a physical-coverage invariant: the full/raw key set must
contain every collector-v1 lookup key. Do not infer compatibility from total
case counts alone. Alias canonicalization may remove duplicate scheduler cases,
but it must not remove a distinct database key. Put synthetic v1 interpolation
anchors under the base recipe's `legacy_model_cases`; targeted real-model runs
will not inherit those unrelated anchors.

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

`include_base: true` means "include the small universal base set" declared by
base-file `model_ops` (currently dense attention and GEMM). It does not opt a
model into every recipe that happens to exist under `base_ops/`. Use an explicit
model `base_ops` list for auxiliary recipes such as `encoder_attention`, or a
`framework_specific_base_ops` list for framework-only recipes. When an op is
required by one concrete checkpoint rather than the whole architecture, use an
artifact-keyed `model_specific_base_ops` mapping. For example, the Qwen3 static
FP8 checkpoint opts into `compute_scale` without making base Qwen3 or dynamic
FP8 checkpoints collect it. Adding a new base recipe therefore cannot silently
fan out across every model or artifact.

```yaml
model_specific_base_ops:
  Qwen/Qwen3-32B-FP8-Static-PerTensor:
    sglang: [compute_scale]
    trtllm: [compute_scale]
    vllm: [compute_scale]
```

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

An SM exception only constrains an op already selected by the model plan. It
never activates an unrelated op merely because the SM catalog mentions it.

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
selection. A concrete selector in a model file overrides the inherited
base-recipe `cases: all`, so `case_ids` or `rules` can genuinely narrow a shared
base op.

Supported generated-case selector fields:

- `cases: all`: selects every generated case for the op.
- `cases: [...]`: in selector contexts, non-mapping entries are treated as exact generated `case_ids`.
- `case_ids`: exact generated case IDs.
- `contains`: string fragments matched against the generated case string or generated case ID.
- `indices`: zero-based generated case indices.
- `ranges`: inclusive zero-based index ranges, such as `"10-20"` or `[10, 20]`.
- `limit`: keeps only the first N cases after selection.
- `rules`: structured matches over positional collector case fields.

There is one important distinction: a dictionary under `cases` in a
`base_ops/<op>.yaml` file is a generation recipe, not a selector. For example,
a GEMM shape sweep dictionary tells the collector to create many concrete GEMM
cases. Model files should put physical dimensions in `model_case_values` and use
generated-case selectors (`rules`, `case_ids`, and so on) in op sections; they
should not define a second generator recipe there.

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
