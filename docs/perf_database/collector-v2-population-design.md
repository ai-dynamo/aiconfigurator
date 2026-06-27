# Collector v2 Population and Physical-Case Deduplication

## Status

Phase 1 is implemented. The common population path is active for every
collector invocation, and attention/encoder are the first operation family
with consumer-aligned physical deduplication and checked-in historical coverage
manifests.

Other operations already receive stable invocation IDs, additive-rule
deduplication, resume isolation, and output conflict validation. They continue
to use the explicit `legacy_passthrough` schema until their positional ABI can
be projected to a physical row before execution.

## Contract

Collector YAML is additive. Python owns execution and database identity.

```text
one parsed YAML catalog
  -> independently expand base and model rules
  -> normalize through an op schema
  -> reject unreachable or unsupported candidates
  -> derive stable invocation identity
  -> deduplicate and merge provenance
  -> apply selectors once to the populated union
  -> enforce scoped historical physical coverage
  -> launch workers
  -> validate emitted physical rows
  -> atomically merge into parquet
```

This separates three questions that were previously mixed together:

1. Why does a model need a case? YAML and `RuleSource` answer this.
2. Is it one benchmark execution? `InvocationKey` answers this.
3. Is it one AIC database lookup point? `PhysicalRowKey` answers this.

Counts alone are not a compatibility proof. A full/raw plan is compatible only
when every protected historical physical key is still present.

## Goals

- Let authors append model/backend cases without deep-merging independent
  dimensions into a global Cartesian product.
- Collapse artifact aliases when quantization is already a separate axis.
- Keep correlated attention heads, KV heads, head dimension, window, and TP
  topology together.
- Preserve every protected historical lookup point in full/raw collection.
- Make targeted model collection model-exact and substantially smaller.
- Prevent resume, subprocess logging, and incremental finalization from losing
  already collected data.
- Migrate operation families incrementally without forcing one large schema
  rewrite.

## One catalog and one population path

`collector.model_cases.load_case_catalog()` parses the selected base, model,
and SM YAML documents once. The same `CaseCatalog` is activated while legacy
case generators run, so op selection and case generation see the same source
documents and model path.

Both model-planned and raw registry runs pass through `compile_population()`.
An unmigrated op uses `legacy_passthrough`; it is not allowed to bypass stable
identity, central ordering, or reporting. Attention runtime getters and
`--plan-only` call the same framework-free builders.

Some module collectors still load inner sweep values inside a spawned
subprocess. Their complete migration requires moving that inner sweep into the
top-level task ABI; this is tracked as a later operation-schema phase.

## YAML semantics implemented now

Existing schema-v1 sections remain supported:

- `all_frameworks_op_cases` and `framework_specific_op_cases` activate ops and
  define generator recipes or selectors;
- `model_case_values` owns model structure and artifact aliasing;
- `sm_exceptions` owns hardware/framework exclusions.

Schema v2 adds exact, additive `population_rules` and
`framework_specific_population_rules`. Each rule expands independently. Values
from two rules are unioned; their dimensions are never implicitly crossed.

```yaml
schema_version: 2
architecture: Qwen3ForCausalLM
model_path: Qwen/Qwen3-32B
base_ops: [attention_generation]

population_rules:
  attention_generation:
    - id: qwen_decode_delta
      when:
        backends: [sglang, trtllm, vllm]
        model_paths: [Qwen/Qwen3-32B, Qwen/Qwen3-32B-FP8]
        min_sm: 90
      cases:
        - batch_size: 2
          sequence_length: 8191
          num_heads: 32
          num_kv_heads: 4
          head_dim: 128
          use_fp8_kv_cache: true
          window_size: 0
```

Attention mappings also accept `query_heads`/`kv_heads` and
`fp8_kv_cache` aliases. A native topology can request TP projection:

```yaml
cases:
  - batch_size: 4
    sequence_length: 2047
    num_attention_heads: 64
    num_key_value_heads: 8
    tensor_parallel_size: 8
    head_dim: 128
    window_size: 0
```

The schema requires query heads to divide TP and uses the same ceil-division
rule as the existing generator for local KV heads. It then emits the exact
backend positional tuple consumed by `run_*(*case)`.

Supported `when` fields are `backends`, `model_paths`,
`model_architectures`, `sm_versions`, `min_sm`, and `max_sm`. In full mode,
model predicates are evaluated against each source model document, so rules
for different models remain an additive union. A rule cannot activate an op
that the model/base plan did not select.

`profiles`/`sweep` inside schema-v2 population rules, rule inheritance,
`extends`, `replace`, and a general predicate language are not implemented.
Existing structural attention profiles under `model_case_values.attention`
remain the supported compact sweep form. Adding another syntax before more op
schemas exist would create two competing expansion systems.

## Identity

### InvocationKey

An invocation key contains the normalized payload plus its execution scope:

- backend variant and operation;
- performance filename and planner schema version;
- framework version, GPU/system ID, and SM;
- model path and architecture;
- a fingerprint of the selected YAML catalog;
- schema-projected payload.

The canonical JSON form is SHA-256 hashed. Dict order, tuple/list spelling,
sets, enums, dataclasses, and paths do not make IDs unstable. Resume
checkpoints use the same execution dimensions and reject a framework, hardware,
model, YAML-content, power-measurement mode, or measurement-duration change.

### PhysicalRowKey

A physical key is versioned and scoped by performance table. Its fields match
the key that AIC's Python consumer uses; representative Rust consumers are
kept in parity tests. Measurement fields and provenance-only artifact names
are excluded.

Attention examples:

- MHA normalizes `num_key_value_heads == num_heads` to the consumer's zero-KV
  sentinel;
- generation uses `isl + step` as total sequence length;
- a generation-only context-FMHA flag is not a distinct database key;
- encoder keys are `(dtype, head_dim, heads, isl, batch)`.

For attention/encoder, the planner's invocation projection is the physical
row key, so redundant consumer points are pruned before workers start. For
grouped module invocations that emit many rows, the top-level invocation stays
distinct and actual rows are validated during finalization.

### Duplicate metadata

Duplicate candidates merge ordered provenance. Non-overlapping metadata is
merged; conflicting values fail planning rather than silently keeping the
first rule's expected-failure or ownership metadata.

## Selectors

All legacy cases and schema-v2 rules are populated and deduplicated first.
`case_ids`, structured rules, indices, ranges, and `include.limit` are then
applied once to the combined ordered plan. A limit of `N` therefore selects at
most `N` cases, not `N` cases from every additive rule.

`--shuffle` and CLI `--limit` remain runner controls and are applied after
population and YAML selectors.

## Historical coverage guard

Canonical gzip JSONL manifests live under
`collector/planner/manifests/collector_v1/`. Each header fixes source ref,
backend variant, an exact framework version or compatibility range,
GPU/system ID, SM, performance table, and physical-key schema version. A hard
subset assertion runs only for a full/raw plan whose scope matches the
manifest. Targeted model plans are intentionally
not v1-wide. Other scopes report `out_of_scope`, never a false compatibility
claim.

Current exact B200/SM100 and XPU baselines are:

| Backend | Operation | Historical | Current full/raw | Added | Removed |
| --- | --- | ---: | ---: | ---: | ---: |
| SGLang | context attention | 33,714 | 50,901 | 17,187 | 0 |
| SGLang | generation attention | 19,484 | 39,556 | 20,072 | 0 |
| TensorRT-LLM | context attention | 63,192 | 75,483 | 12,291 | 0 |
| TensorRT-LLM | generation attention | 40,240 | 54,318 | 14,078 | 0 |
| vLLM | context attention | 40,392 | 50,932 | 10,540 | 0 |
| vLLM | generation attention | 36,288 | 53,638 | 17,350 | 0 |
| vLLM XPU | context attention | 16,188 | 17,838 | 1,650 | 0 |
| vLLM XPU | generation attention | 26,322 | 30,728 | 4,406 | 0 |
| SGLang | encoder attention | 7,008 | 7,679 | 671 | 0 |
| TensorRT-LLM | encoder attention | 7,008 | 7,679 | 671 | 0 |
| vLLM | encoder attention | 7,008 | 7,679 | 671 | 0 |

Context/generation use the pre-collector-v2 source
`a4827ce203e9fbc24fe6c6779a7eaa2a7dc79f1a`. Encoder did not exist at that
ref; its honest initial baseline is the original hardcoded PR #1092 commit
`36808ecced9af9d0d71d944c716ae96d1d4a2a47`. Restoring sequence lengths
13/26/52/104 recovered 924 encoder keys that the later YAML conversion had
dropped, while retaining new sequence length 1 and model profiles.

For a targeted SGLang Qwen3-32B plan, context/generation are only 2,448 and
2,354 cases. The base and `-FP8` artifacts produce the same structural set;
quantization is not multiplied by the model suffix.

There is currently no removal allowlist. Deliberately deleting a historical
low-value point requires a separately reviewed policy and manifest update.

Regenerate or verify manifests with:

```bash
.venv/bin/python tools/collector/generate_v1_attention_manifests.py --check
```

## Output and resume safety

Every worker runs inside a scoped collector invocation. `log_perf()` writes the
invocation ID to a staging-only CSV column. An environment bridge carries that
ID into module collector subprocesses; the field is removed before parquet is
published.

Finalization applies these rules:

- the same known invocation and physical key is a retry; the last successful
  staging row wins;
- different known invocations producing one physical key are a conflict;
- unattributed legacy rows are never silently reclassified as retries;
- validation writes a temporary clean CSV, so a conversion failure leaves the
  original staging bytes and invocation IDs intact;
- an existing parquet is merged with new unique physical keys instead of being
  overwritten by a targeted or resumed run;
- parquet replacement is atomic;
- any remaining collector error keeps CSV staging in place and skips
  finalization.

`log_perf()` lock or write failures raise into the worker. They are not marked
as completed resume tasks.

## Consumer compatibility

The physical-key registry covers the active performance tables without
guessing keys for unknown files. Known tables fail on a missing key field;
unknown tables retain legacy behavior.

This change also closes consumer discrepancies discovered while tracing case
consumption:

- Mamba and GDN generation use their batch-only physical key in Python and
  Rust; duplicate rows retain the same file-order winner.
- DSA context/generation use phase-correct quant fields and preserve the
  physical `trtllm` versus `flashmla_kv` backend axis.
- MHC treats architecture as provenance, while MoE preserves the two
  kernel-source-specific effective quant modes.
- DSV4 context prefix lookup and generation `isl + step` collision handling
  are consistent across Python and Rust.

Adding the DSA backend field changes the positional Rust op wire, so
`EngineSpec` schema version 2 rejects older bincode payloads from the leading
version word before attempting to decode changed op layouts. These are query
and wire semantics, not collector-only details; a case is not useful coverage
unless AIC can retrieve it unambiguously.

## Migration state and roadmap

| Capability | Attention/encoder | Other registered ops |
| --- | --- | --- |
| Central population and stable ID | active | active |
| Additive schema-v2 exact cases | active | positional exact cases via passthrough |
| Pre-worker consumer-key dedupe | active | pending op schema |
| Historical exact manifest guard | active for checked scopes | pending manifests |
| Final output physical conflict guard | active | active for registered tables |

Next operation families should be migrated in this order:

1. MLA, DSA, Mamba, GDN, and MHC, where one invocation may emit many rows.
2. GEMM, ComputeScale, MoE, WideEP, and DSV4.
3. Only after those schemas exist, consider compact schema-v2 sweep syntax and
   explicit reviewed removal manifests.

Local collector `seen` sets should be removed only after the corresponding
schema, manifest, and Python/Rust query parity tests are active.
