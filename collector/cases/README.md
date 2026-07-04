# Collector v2 Case Files

Collector v2 plans collection from model YAML instead of treating the
collector as a flat op list. The whole declaration surface is:

```text
plan cases = dedup( expand(base_ops sweep grids) ∪ expand(model_case_values shapes) )
runnable   = plan cases ∩ hardware capability floors  −  hang denylist
```

There is intentionally no per-case selector or exception rule engine. Cases
that cannot run on some framework version fail at runtime and are recorded
with a classification and a (model, dtype) group label in the failure log; a
whole group failing is surfaced in the collection summary as a fix-me signal.
See `.claude/rules/collector/failure_handling.md` for the escalation rules.

## File Layout

- `base_ops/<op>.yaml`: shared recipe library. Only recipes named by a model's
  `base_ops` (or declared universal through the base file's `model_ops`) are
  activated.
- `models/<architecture>_cases.yaml`: architecture-specific op activation,
  model path aliases, and model shape values.
- `capabilities.yaml`: positive hardware floors (dtype/op → min SM), applied
  as a generation-time intersection. Hardware facts only — no shapes, no
  framework versions, no per-backend nesting.
- `denylist.yaml`: cases that hang or kill the node (substring match, dated,
  with reason). Ordinary crashes never go here.

## Model Case Files

Model files are keyed by HuggingFace architecture name and list every model
path alias that should resolve to that architecture plan:

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

Op sections activate ops for the plan. `cases: all` is the only meaningful
value — there are no selectors:

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

Model-specific op dimensions live in `model_case_values`:

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
so a targeted run collects one model without editing Python. Two deliberately
different multi-name forms:

- `model_aliases` resolves multiple artifact names to one canonical physical
  case. Use it only for shape-only ops where the artifact does not affect the
  invoked kernel or persisted key.
- `model_paths` expands one physical case per path. Use it only when the model
  name changes runtime behavior, quantization policy, activation, or module
  loading.

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
model's query/KV heads with another model's head dimension or window. A
targeted run with an explicit profile uses only that model profile. Full/raw
runs combine the base operation's `head_profiles` with all model profiles and
deduplicate the resulting physical tuples.

## Base Op Files

Shared sweep recipes live in per-op files under `base_ops/<op>.yaml`. For
cross-model ops such as MoE, MLA, Mamba2, GDN, and MHC, the base op file owns
the token counts, batch/sequence sweeps, parallelism sizes, routing
distributions, and generator constraints. Model YAML stores model dimensions;
base op YAML stores the reusable sweep policy.

MoE quantization is resolved before a case is queued. Base `quantization_modes`
may declare `min_sm` (axis-level hardware floor), runtime features, an
`allowed_model_paths` allowlist, and optional `module_config`. A model row
narrows that list with `framework_quantization.<backend>.allowed_modes`.
Quant-sensitive checkpoint artifacts use separate model rows even when all
geometry fields are identical.

When frameworks invoke different routed-expert geometry for the same artifact,
a MoE model row may positively declare `frameworks: [vllm]` (or another list).
Omitting the field means all frameworks. `get_common_moe_test_cases()` without
a backend retains the legacy union during collector migrations; upgraded
collectors pass their backend so geometry and quantization remain separate facts.

`include_base: true` means "include the small universal base set" declared by
base-file `model_ops` (currently dense attention and GEMM). Use an explicit
model `base_ops` list for auxiliary recipes, `framework_specific_base_ops` for
framework-only recipes, and artifact-keyed `model_specific_base_ops` when one
concrete checkpoint needs an op (e.g. the static-FP8 Qwen3 artifact opting
into `compute_scale`).

## Capability Floors

`capabilities.yaml` holds positive hardware facts:

```yaml
dtype_min_sm:
  fp8: 89
  fp8_block: 90
  nvfp4: 100
op_min_sm:
  dsa_context_module: 90
```

`collector/capabilities.py` filters generated cases by typed field access
(`moe_type`, `gemm_type`, `dtype`, `kv_cache_dtype`, FP8 flags) before
queueing. Unknown dtypes and unknown ops are permissive: the table encodes
known hardware facts, never guesses. Framework-version gaps do NOT belong
here — they are runtime observations.

## Denylist

`denylist.yaml` is for cases that hang or kill the node — the only failure
mode that fail-fast + failure log cannot handle:

```yaml
entries:
  - contains: "tp=32, ep=16"
    reason: "sglang wideep_moe tp=32 deadlocks in NCCL init on 0.5.10"
    added: 2026-07-04
```

Keep entries dated; re-audit on every framework version bump and delete
entries that no longer reproduce.

## Running Subsets (healing)

Subset selection is a runtime concern, never persisted to YAML:

```bash
# one model on one GPU type
python3 collect.py --backend sglang --model-path <model> --gpu b200_sxm

# substring filter over generated cases (repeatable, OR semantics)
python3 collect.py --backend trtllm --ops moe --case-filter "tp=4"

# finish an interrupted run
python3 collect.py --backend trtllm --resume
```

## Adding New Coverage

Add one architecture file for a new architecture, or add a model path alias to
an existing architecture file when the model uses the same case plan. Add or
edit one `base_ops/<op>.yaml` file when common op sweeps change. Add a new op
collector only when no existing op can generate the needed points. See
`.claude/rules/collector/case_authoring.md` for the full rules.
