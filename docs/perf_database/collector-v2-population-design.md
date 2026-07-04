# Collector V2 Case Pruning

## Goal

Collector V2 should retain useful measurement coverage and intentional model
additions while avoiding cases created only by unrelated Cartesian-product axes
or execution-equivalent artifact aliases.

The attention migration used the following comparison as a one-time check:

```text
V1 physical cases ⊆ cleaned V2 physical cases
```

`removed_v1_cases` was required to be zero for attention. That audit is not
part of the runtime schema: production YAML contains ordinary default and model
profiles, and the generator has no Collector-V1 compatibility mode.

For the quant-sensitive MoE families migrated in this PR, a physical point must
have a verified checkpoint artifact whose quantization can request it.
Collector V1 crossed each geometry with most backend quant modes, so preserving
every V1 MoE key would preserve measurements that no reviewed artifact can
consume. Those historical cross-products are reported separately below instead
of being reintroduced as synthetic or `legacy_*` profiles. Existing model
families without verified artifact policy retain their broad synthetic sweep;
this document does not claim that every MoE family has been migrated.

Targeted structural population is model-exact. It does not inherit unrelated
default topology profiles when the selected model has an explicit structural
profile; shared workload sweeps remain reusable.

## Scope

This work changes Collector-only population and the operation-local execution
plumbing required to keep generated quantization labels truthful:

- `collector/cases/**/*.yaml`
- `collector/case_generator.py`
- `collector/model_cases.py`
- operation-local case getters and the minimal runtime parameters required by
  declared quantization/precision cases
- the model-specific op selection path in `collector/collect.py`
- Collector tests and documentation

The `collector/collect.py` change removes only the DeepSeek V4 hard-coded op
override so that the resolved YAML case plan remains authoritative. Generic
resume, checkpoint, and output-finalization behavior is unchanged. This work
does not change AIC SDK or Rust lookup behavior, EngineSpec, or Dynamo Planner.
Collector output continues to satisfy the existing consumers; consumer changes
are outside this PR.

The final implementation intentionally contains no Collector-V1 runtime mode,
historical snapshot rewrite, generic resume/checkpoint/finalization change, or
defensive deduplication that has no effect on repository-owned YAML. Schema
fields and filters that lost their only producer during the design were removed
instead of being kept as speculative compatibility surface.

## Baselines

- Attention V1: `a4827ce203e9fbc24fe6c6779a7eaa2a7dc79f1a`, immediately before
  Collector V2.
- Encoder attention: the original hardcoded grid from PR #1092, because the
  pre-V2 attention baseline predates this collector.
- Collector V2 before pruning: upstream `66c6e05fef00cbee6546847fa2280116ef4a38cd`.

The comparison uses consumer-visible physical lookup keys, not model aliases,
scheduler task IDs, latency, or power measurements.

Historical data was used only for the read-only comparison recorded below. This
PR does not add, move, regenerate, or update snapshots.

## Three identities

Collector population must keep three different identities separate:

1. **Recipe identity** explains why YAML requested a case. Multiple model
   documents may provide provenance for the same work.
2. **Benchmark invocation identity** contains everything that can change the
   executed kernel or runtime setup, including path-dependent checkpoint
   quantization.
3. **Persisted physical key** is the unchanged key used by the current AIC
   consumer to load a measurement.

Deduplication is safe only when benchmark invocation identity is equivalent and
the persisted physical key is also equivalent. A consumer-key collision alone
does not prove two invocations are interchangeable. In particular, a BF16,
FP8, or NVFP4 checkpoint path may select different runtime behavior before its
native quantization becomes explicit.

## Population flow

```text
additive YAML profiles
    -> select the model/backend operations
    -> expand correlated structural tuples
    -> resolve model/backend quantization policy
    -> reject unreachable/backend-unsupported tuples
    -> derive operation-local invocation identity
    -> stable deduplication of proven-equivalent invocations
    -> apply model and SM selectors
    -> benchmark queue
```

The important distinction is between an axis list and a structural profile.
For attention, `(query heads, KV heads, head dimension, window size, TP)` is a
correlated tuple. Combining global lists for those fields produces shapes that
no model uses. Batch and sequence axes can still be swept after the structural
tuple is selected.

YAML remains additive. Deduplication happens after enough information is known
to identify the actual benchmark point. Stable first-wins is the default. When
a later selector can distinguish two equivalent recipe representations, the
getter must choose a documented canonical representative that remains
selectable; standalone MLA uses the smallest TP for this reason.

## Separate coverage, pruning, quarantine, and observed failure

Collector V2 defaults to attempting a generated case. A failed measurement is
evidence about one exact invocation; it is not, by itself, a reason to remove
the shape from future plans. Keep these four states distinct:

1. **Out of scope** is a release coverage decision, such as collecting only
   TP1/2/4/8. It does not claim that TP16/32 is unsupported. Record the omitted
   axis and count as plan policy rather than as a kernel exception.
2. **Not applicable** means no valid measurement exists by definition. Examples
   include mathematically invalid TP/EP shards, an artifact/quantization
   mismatch, a framework operation absent from the pinned runtime, or distinct
   invocations that the current database key cannot represent safely. Prune
   these cases during population, or fail population when ambiguity would be
   hidden.
3. **Known unsafe** means an exact invocation has repeatedly poisoned the CUDA
   context, aborted its process, or caused enough deterministic fatal churn to
   make a full run unsafe. Preserve the case identity, reason, failing evidence,
   and nearest successful controls even when execution is quarantined.
4. **Attempted** covers every other generated case. Persist either a valid
   measurement or an explicit failed/expected-failed checkpoint result. Do not
   manufacture a row or silently remove the case to make completion appear
   green.

`expected-failed` is an outcome of an attempted case, not a fifth population
state. Its exact failure contract must exist in the decision catalog before the
run and name the permitted error class/signature. A matching failure records
`expected-failed`; a success records `passed` and makes the old expectation a
stale contract to review. Never relabel an unexpected failure retroactively in
the same checkpoint. A framework version, backend, GPU path, benchmark
contract, or decision-catalog change requires a new fingerprint and namespace.

These categories are a decision contract, not a requirement to introduce a
new class hierarchy or generic rule engine. The first implementation may emit
them through one canonical plan manifest while reusing existing structured
selectors and checkpoints.

Ordinary runtime errors, isolated OOMs, low-priority TP sizes, and failures that
may change with a different backend or framework release remain attempted by
default. The parallel runner records the failed task and replaces a worker
after fatal CUDA errors or process exits; artifact validators must reject
unexplained missing rows or partial output.

A compatibility or safety rule must describe the invocation that actually
failed:

```text
framework + exact version + GPU/SM + model/artifact + quantization
+ resolved backend/kernel + phase + TP/EP + shape
```

A bare rule such as `TP >= 16` is not equivalent to that identity. If the
framework selects CUTLASS on one SM and TRT-LLM on another, those are different
invocations even when model and TP match. Stable model mathematics may be
shared across frameworks and platforms; kernel alignment, launch, and resource
limits are scoped to the framework/version/SM/backend path that established
them.

Do not automatically carry a known-unsafe or expected-failure classification
to a new framework version, backend selection, or GPU path. Regenerate the raw
plan, probe the new selector, and attempt representative failing boundaries and
nearby successes before retaining the classification. Keep raw, out-of-scope,
not-applicable, quarantined, attempted, passed, and failed counts separately so
that a filtered point never disappears from review.

## Simplified ownership model

Collector V2 currently consumes the same YAML through both the case planner and
operation generators, then may remove a case again in a framework getter, an SM
exception, or an operation's inner loop. This is a migration state, not a
license to encode the same fact in every layer. Use the following ownership
contract for new work and migrate existing rules toward it without changing
case IDs first:

```text
model facts + release coverage
    -> raw physical case
    -> exact framework invocation resolver
    -> exact safety quarantine
    -> execution and per-point outcome
    -> one central artifact acceptance gate
```

| Owner | May decide | Must not decide |
| --- | --- | --- |
| Base/model YAML | Stable model and artifact facts, correlated dimensions, shared workload axes, and explicit release coverage | A transient kernel failure, one framework's backend workaround, or a capacity observation from one GPU |
| Population engine | Convert stable mathematical, model/artifact, and database-key facts into a raw invocation candidate or an explicit not-applicable decision | Framework runtime capability, backend failure history, or a silent drop with no decision record |
| Framework getter/resolver | Convert each raw candidate into one exact invocation or an explicit pinned-runtime not-applicable decision, including resolved backend/kernel and framework-local structural constraints | Shared model truth, another framework's support, a silent `continue`, or a historical failure table used only to make a run green |
| Exception/quarantine catalog | One evidence-backed, exactly scoped known-unsafe decision, plus a pre-run expected-failure signature over an attempted case | A second owner for not-applicable, generic TP/EP removal, failure-rate acceptance, retroactive relabeling, or an unversioned copy of another platform's exception |
| Runtime collector | Execute the resolved invocation, fail loudly on invariant or persistence errors, and report every inner point | Silently `continue` after an unexpected error or change backend to avoid a failure |
| Checkpoint and plan manifest | Preserve the complete decision/outcome set and bind it to the runtime and plan identity | Infer success from process exit alone or reuse state across an identity mismatch |
| Artifact validator | Decide whether the complete result is publishable | Manufacture missing rows, accept unexplained gaps, or reinterpret rows from an older snapshot |

Do not add another selector language or a generic compatibility framework while
migrating. Use existing structured named-field rules. Treat `indices`,
`ranges`, string `contains`, and exact generated case IDs as diagnostic tools,
not durable release policy: ordering and case-string changes can silently alter
their meaning.

Keep one source of truth for each decision. A plan-time predicate may have a
matching runtime assertion, but both must evaluate the same narrowly named
invariant; do not maintain two independent lists. Extract a small pure
predicate only when the same rule is genuinely consumed in both places. Do not
introduce a broad helper abstraction for a single filter.

Capacity is a runtime property of a concrete GPU product, not just an SM. An SM
exception file can represent an architecture capability, but it cannot prove
that H100 and H200, or B200 and GB200, share a memory boundary. Derive capacity
from the live runtime where possible; otherwise scope and label the observation
to the measured product rather than promoting it to an SM-wide rule.

## Central acceptance invariants

The default-attempt policy is safe only when incomplete output cannot look
complete. Before removing broad pre-execution filters, Collector core must
enforce these invariants:

1. Unexpected failed tasks remain in the final checkpoint even after resume.
2. Any unresolved unexpected failure keeps CSV staging data for diagnosis,
   prevents parquet finalization, and makes the command exit nonzero.
3. Expected-failed tasks are accepted only through an exact reviewed contract;
   the observed error class/signature must match, and a failure percentage is
   never an acceptance rule.
4. A checkpoint fingerprint binds the framework image digest, package version
   and source revision; collector code/config manifest including base/model
   YAML; GPU product and SM; model/full-plan scope; decision-catalog digest;
   benchmark contract; and canonical expanded leaf invocation IDs. A mismatch
   fails closed instead of silently reusing prior work.
5. A persistence failure cannot mark a task done. Task IDs, output keys,
   duplicate checks, and written rows must reconcile before finalization.
6. Grouped collectors such as MoE, DSV4, GDN, mHC, and MLA-module report
   outcomes for every inner point. An outer task cannot be green while hiding
   unexpected inner failures.

Maintain append-only attempt history and one terminal status per leaf under one
fingerprint. An unexpected failure is resolved only by a successful retry under
that same fingerprint, or by fresh execution in a new decision-manifest and
checkpoint namespace after an exact failure contract was approved. In the
second path, the new observed error must match the pre-existing contract; the
old failure is never reused as its own approval evidence. Use the checkpoint as
the outcome source of truth, the plan manifest as the decision source of truth,
error logs as diagnostic detail, and validators as the publication gate. Do not
create another independent failure ledger.

## AI filter-change gate

An AI agent must not add, remove, widen, or relocate a case rule before meeting
the evidence requirements for its classification.

Every change requires:

1. the exact framework/version, GPU product/SM, operation, model/artifact,
   quantization, phase, resolved backend/kernel when one exists, TP/EP, and
   shape in scope;
2. one population classification and the current rule owner with its complete
   framework/platform blast radius;
3. canonical before/after set diffs, counts, and hashes for out-of-scope,
   not-applicable, quarantined, and attempted decisions; benchmark invocation
   IDs; scheduler task IDs; persisted physical keys; and expected-failure
   contract IDs/signatures;
4. reverse checks for every untouched framework/platform consuming a changed
   shared input;
5. checkpoint and existing artifact compatibility, including the new
   fingerprint/namespace requirement; and
6. an alignment-ledger entry naming the introducing/dropping commit or exact
   uncommitted snapshot when the scoped project maintains such a ledger.

Additional evidence depends on the classification:

- **Out of scope:** require an explicit user or release-owner coverage decision
  and the omitted axis/set. No runtime failure is required and the decision
  must not be described as unsupported.
- **Not applicable:** require mathematical, model/artifact, authoritative
  framework-source, or database-schema proof. Probe a selector or runtime leaf
  only when one exists and is relevant.
- **Known unsafe or attempted with an expected-failure contract:** require
  framework selector/source evidence, clean-GPU reproduction, exact error
  class/signature, post-failure process/GPU state, and nearest same-family
  successes. The population classification remains attempted.
- **Attempted:** require no exception rule; retain and report the observed
  outcome.

If the applicable evidence is unavailable, stop at diagnosis and leave
execution policy unchanged. In particular, an AI agent must not:

- change a shared base/model axis to fix one framework or backend;
- encode one failure as a broad `TP >=`, `SM >=`, model-family, dtype, or OOM
  exclusion;
- copy a skip between Hopper, datacenter Blackwell, RTX Blackwell, Ada, or
  between frameworks without independent selector and runtime evidence;
- weaken coverage, repetitions, graph boundaries, or failure accounting to
  make a run green;
- change the backend away from the pinned framework's production selector;
- add a private kernel patch, process-per-shape workaround, or retry loop
  without explicit user approval;
- use a failure percentage as an artifact acceptance threshold under any
  approval; percentages are investigation signals only;
- resume across a plan/runtime fingerprint mismatch, relabel old measurements,
  or call a structurally complete artifact valid before kernel-path validation;
  or
- modify another framework, SDK schema, consumer lookup, or data publication
  as an incidental fix for a scoped collector failure.

## Future migration order

This section describes follow-up Collector-core work; the current
Collector-pruning scope does not yet implement the generic checkpoint or
finalization changes below. Until the corresponding acceptance prerequisite is
implemented for an operation, do not relax its broad filters merely because
this design defaults to attempted execution.

Simplify without a flag-day schema rewrite:

1. Add the central acceptance gate, plan/outcome manifest, and checkpoint
   fingerprint with zero case-plan delta for operations that already have
   complete per-leaf accounting. Mark grouped operations non-certifiable until
   their inner accounting exists.
2. Inventory and classify existing filters mechanically, preserving task IDs.
3. Migrate one-case/one-measurement operations such as GEMM and attention
   before grouped inner-sweep collectors.
4. Add inner-point accounting to grouped collectors before relaxing their
   skips or exception handling. Treat MoE as grouped while a task carries a
   token list or can emit/skip multiple rows; apply the same rule to DSV4, GDN,
   mHC, MLA-module, and similar collectors.
5. Re-evaluate ordinary and version-sensitive exclusions one exact framework,
   version, backend, and platform at a time. For failure-derived compatibility
   policy, retain only proven not-applicable cases, exact known-unsafe
   quarantines, and pre-run expected-failure contracts. Keep separately
   reviewed out-of-scope release coverage unchanged.
6. Keep already accepted artifacts bound to their original source manifest and
   plan. A later broader plan creates a new artifact contract rather than
   retroactively making the old artifact incomplete.

## Safe deduplication rules

1. Never change a downstream consumer to make a Collector case appear useful.
2. Preserve a field if it changes either the benchmark invocation or any
   current Python/Rust consumer key.
3. Collapse artifact aliases only for shape-only collectors where the model
   path is neither loaded by the benchmark nor part of its persisted key.
4. Do not alias BF16, FP8, or NVFP4 checkpoints before a module benchmark has
   resolved path-dependent native quantization. Once quantization is explicit,
   an operation-local key may collapse only truly identical benchmark inputs.
5. For standalone MLA, total heads and TP may produce the same local-head
   kernel. The getter deduplicates on `(dtype, local heads, batch, sequence)`
   and retains the smallest-TP representation so a later TP selector cannot
   hide that physical point.
6. SM exclusions are pre-execution skips. They are not expected failures after
   a case has already run.
7. Unknown or unproved equivalence is retained. It is better to prune less than
   to silently remove a useful physical point.
8. If distinct benchmark invocations map to one persisted consumer key, fail
   population with both owners and the conflicting key. Do not silently pick a
   representative unless the invocations are already proven equivalent, and do
   not widen the consumer schema inside a Collector-only change.

## Pruning decisions

| Situation | Population behavior | Reason |
|---|---|---|
| Head/KV-head/head-dim/window values from different models | Keep correlated model profiles; do not cross them | Cross-model tuples are not deployable shapes |
| Shape-only collector with base/FP8/NVFP4 names and no checkpoint-native behavior | Canonicalize artifact aliases | Artifact name does not change the invocation or persisted key |
| Quant-sensitive MoE family with verified artifact metadata | Keep one row per artifact and allow only the artifact's declared quant mode | A shared geometry does not make INT4, FP8, MXFP4, and NVFP4 checkpoints interchangeable |
| Module collector that reads checkpoint-native quantization | Retain each path until native quantization is explicit | The path can change the executed kernel |
| Different total-head/TP pairs with the same standalone-MLA local-head key | Deduplicate in the getter | Both invocation and current loader key are equivalent |
| Experimental op with no production consumer | Keep the registry entry, omit it from default model plans | Explicit research runs remain possible without default collection cost |
| Equivalence is uncertain | Retain the cases | Conservative pruning avoids silent coverage loss |

## Attention result

The following canonical B200/SM100 counts compare full/raw physical attention
cases. `Removed` is always measured against the V1 baseline.

| Backend | Operation | V1 | V2 before | Cleaned V2 | Added | Removed |
|---|---|---:|---:|---:|---:|---:|
| SGLang | context | 33,714 | 122,676 | 50,901 | 17,187 | 0 |
| SGLang | generation | 19,484 | 53,654 | 40,468 | 20,984 | 0 |
| TRT-LLM | context | 63,192 | 143,739 | 75,483 | 12,291 | 0 |
| TRT-LLM | generation | 40,240 | 155,582 | 55,230 | 14,990 | 0 |
| vLLM | context | 40,392 | 84,296 | 51,408 | 11,016 | 0 |
| vLLM | generation | 36,288 | 68,920 | 54,270 | 17,982 | 0 |
| vLLM XPU | context | 16,188 | 16,188 | 17,838 | 1,650 | 0 |
| vLLM XPU | generation | 26,322 | 26,322 | 30,728 | 4,406 | 0 |
| **Total** | | **275,820** | **671,377** | **376,326** | **100,506** | **0** |

Relative to upstream Collector V2, this removes 295,051 accidental attention
keys while retaining 100,506 intentional additions over V1.

The unpruned V2 vLLM grids also removed 10,098 context and 8,774 generation
V1 cases, all from the historical `(head_dim=128, window=128)` region. The
final default profiles retain those points, while model-native profiles add
valid new window/head combinations without recreating the global Cartesian
product.

Encoder attention retains all 7,008 original hardcoded cases and adds 671
model-native cases, for 7,679 total.

## Other operation results

These are deterministic shared YAML recipe counts. Backend-specific expansion
may multiply a recipe by dtype, TP/EP, or token lists.

| Operation | V1 | V2 before | Cleaned V2 | Notes |
|---|---:|---:|---:|---|
| GEMM | 35,742 | 35,742 | 35,742 | unchanged |
| ComputeScale | 1,628 | 1,628 | 1,628 | shared recipe unchanged; V2 also activates SGLang/vLLM |
| MoE common | 1,797 | 4,548 | 4,209 | quant-sensitive artifacts have separate recipe identity; backend policy removes invalid products before execution |
| MLA context specs | 220 | 550 | 220 | SGLang/TRT-LLM getters each emit 1,760 unique loader keys |
| MLA generation specs | 362 | 885 | 362 | SGLang emits 2,656 keys after its int32 KV guard; TRT-LLM emits 2,896 |
| Mamba | 8 | 8 | 12 | four default synthetic interpolation profiles added |
| GDN | 16 | 16 | 16 | unchanged |
| mHC | 8 | 8 | 8 | artifact recipes remain distinct; backend getters collapse them to four phase/shape groups before the unchanged token sweep |
| MLA BMM pre/post | 400 / 448 | 400 / 448 | 400 / 448 | unchanged |

With `COLLECTOR_MODEL_PATH` unset, the SM100 raw public-getter audit separates
candidate getter tasks from actual invocation identity. These counts are
measured after artifact quantization policy, before model/SM/version plan
selectors, and before expanding each task's token list:

| Backend | Raw getter tasks before dedupe | Raw getter tasks now | Duplicate tasks removed | Unique invocation/key loss |
|---|---:|---:|---:|---:|
| TRT-LLM | 9,414 | 7,944 | 1,470 | 0 |
| vLLM | 2,799 | 2,352 | 447 | 0 |

The vLLM audit enables the `per_block_fp8`, `nvfp4`, and `mxfp4` runtime
features. Full model plans may filter these raw tasks later, so the table is not
a post-selector or token-expanded queue count.

Artifact-exact policy is a separate pruning stage, not part of the dedupe row
above. Relative to the pre-review population, the reviewed DeepSeek V3,
MiniMax M2, and Nemotron 3 families remove 1,110 TRT-LLM getter tasks / 29,970
token-expanded rows and 501 vLLM getter tasks / 13,527 rows. Every removed row
in this subset combines a model geometry with a quant mode that none of that
geometry's verified checkpoints use. DeepSeek V3/R1/V3.2 and MiniMax M2 use
FP8-block artifacts; their NVIDIA variants use NVFP4. Nemotron Nano is BF16,
Super is NVFP4, and Ultra keeps distinct BF16, FP8, and NVFP4 recipe rows.
Pinned SGLang has no plain per-tensor FP8 MoE path, so the Ultra FP8 artifact
intentionally schedules no SGLang MoE case rather than being relabeled as
FP8-block. Pinned vLLM limits its NVFP4 path to top-k <= 10, so Nemotron
Super/Ultra top-k-22 NVFP4 is likewise an explicit gap rather than a mislabeled
fallback case.

`nvidia/nemotron-ultra-rl-050826` remains available to the shape-only Mamba
profile, but it has no MoE profile: the repository has no checkpoint quant
config proving whether its required FP4 format is NVFP4 or MXFP4. It can be
enabled after that artifact contract and an exact-version runtime smoke exist.

The same distinction makes the V1 comparison explicit. The following SM100
counts are token-expanded consumer keys, with vLLM runtime features enabled:

| Backend | V1 keys | Current keys | V1 keys retained | New keys | V1 cross-products removed |
|---|---:|---:|---:|---:|---:|
| TRT-LLM | 157,869 | 214,488 | 110,160 | 104,328 | 47,709 |
| vLLM | 63,342 | 63,504 | 35,964 | 27,540 | 27,378 |

The removed V1 keys are historical geometry-by-quant products, not modes
requested by the migrated artifact families. Attention retains its stricter
zero-V1-key-loss guarantee; MoE deliberately does not manufacture a legacy
artifact to keep an unreachable physical point alive.

These dedupe paths remain because current repository YAML produces real
duplicates. The analogous MLA-module `seen` guards were removed: current model
specs and sweeps are already unique, so those guards changed no scheduled work
and obscured the path-sensitive checkpoint contract.

The MLA spec rows above are recipes, not final getter queues. With two dtypes on
SM90/SM100, the standalone getters compare as follows:

| Backend | Operation | V1 scheduled | V1 unique physical | Current scheduled | Current unique physical | Removed physical |
|---|---|---:|---:|---:|---:|---:|
| SGLang | context | 3,080 | 1,760 | 1,760 | 1,760 | 0 |
| SGLang | generation | 4,648 | 2,656 | 2,656 | 2,656 | 0 |
| TRT-LLM | context | 1,760 | 1,760 | 1,760 | 1,760 | 0 |
| TRT-LLM | generation | 2,896 | 2,896 | 2,896 | 2,896 | 0 |

For a targeted Kimi TP<=8 plan, SGLang emits 1,100 context / 1,660 generation
cases and TRT-LLM emits 1,100 / 1,810. Both retain local heads
`{128, 64, 32, 16, 8}`. The 64-head YAML profile is required for the last
targeted bucket and for `local_heads=1` in full collection; overlap with the
128-head profile is removed only in the backend getter.

For migrated quant-sensitive MoE families, geometry and checkpoint
quantization are separate identities. New model profiles remain additive, but
a backend schedules only the declared artifact mode rather than taking the
Cartesian product of every shape and every backend quant mode. DeepSeek V4
native artifacts schedule
`w4a8_mxfp4_mxfp8` in TRT-LLM. Pinned SGLang 0.5.10 predates the DSV4-specific
MXFP4 method, and pinned vLLM 0.19.0 has no DSV4 MXFP4/MXFP8 implementation,
so neither schedules a native V4 MoE case. The `sgl-project/*-FP8` artifacts
schedule `fp8_block`.
Kimi-K2-Instruct schedules `fp8_block`, native Kimi-K2.5 schedules
`int4_wo` with group size 32, and NVIDIA Kimi-K2.5 schedules `nvfp4`.

GPT-OSS is also backend- and hardware-specific. On SM100, SGLang and TRT-LLM
retain `w4a16_mxfp4` and add `w4a8_mxfp4_mxfp8` because the two labels select
distinct activation precisions. On SM103, TRT-LLM retains both modes while the
pinned SGLang 0.5.10 path is skipped by a version exception. On SM120, pinned
SGLang 0.5.10 is likewise skipped, and pinned TRT-LLM 1.3.0rc10 skips both
GPT-OSS modes because its fused MoE path rejects them. TRT-LLM retains only
`w4a16_mxfp4` on Hopper, while vLLM collects `w4a16_mxfp4`. SGLang explicitly
selects BF16 activation precision for the W4A16 label and its runtime `default`
selects MXFP8 activation for the W4A8 label.
SGLang 0.5.10's generic MXFP4 method is retained for GPT-OSS W4A16/W4A8, where
its high-level `FusedMoE + Mxfp4Config` path owns the FlashInfer API, TP
padding, and EP-local expert layout. It is not reused for DeepSeek V4: the
DSV4-specific method was added later and has a different top-k/clamp contract.
A future SGLang 0.5.14 collector upgrade can enable native DSV4 W4A8 after an
exact-version smoke instead of carrying a newer-version kernel shim here.

DSA module population retains checkpoint paths because all three backends load
model-path-specific config. SGLang resolves native checkpoint quantization to
reject impossible explicit GEMM combinations, but `quantization=None` may still
auto-detect the artifact's config for a BF16-labelled module case. The getters
therefore retain distinct paths even when those rows later project to the same
consumer key. A physical-key collision by itself is not permission to drop a
path-sensitive invocation. Conversely, a BF16 timed-module label does not make
an unsupported full-model setup valid: for exact SGLang 0.5.14 on SM90, an
NVFP4 checkpoint initializes Marlin before the module benchmark. Because
Marlin is an INT4-WO backend in this collector contract, those checkpoint
invocations are `not_applicable` and are filtered before canonicalization;
ordinary GLM DSA uses the remaining BF16 artifact. SM100/103 native NVFP4
module paths remain separate invocations.

SGLang's inner MLA/DSA module sweep reads the same YAML precision specs and SM
gates as the population layer. In particular, Ada/Hopper expand `fp8_block`,
not `nvfp4`; any future NVFP4 module precision must be declared with a
Blackwell `min_sm` gate in YAML.

The following SM100 counts are raw tasks returned by each public getter with
`COLLECTOR_MODEL_PATH` unset, before model/SM/version plan selectors. For
SGLang, each raw task is a subprocess group whose batch, sequence, prefix, and
precision inner sweep is expanded later; these are not token-expanded
invocation counts.

| Backend | Operation | Upstream V2 raw getter tasks | Current raw getter tasks | Current unique task projections | Removed upstream task projections |
|---|---|---:|---:|---:|---:|
| SGLang | context | 792 | 792 | 528 | 0 |
| SGLang | generation | 48 | 48 | 32 | 0 |
| TRT-LLM | context | 46,848 | 46,848 | 23,424 | 0 |
| TRT-LLM | generation | 35,328 | 35,328 | 17,664 | 0 |
| vLLM | context | 70,272 | 70,272 | 35,136 | 0 |
| vLLM | generation | 35,328 | 35,328 | 17,664 | 0 |

The analogous GLM MoE artifacts are deliberately not merged: SGLang selects
native FP8/NVFP4 MoE quantization by artifact path, so those paths represent
real additional measurements rather than scheduler duplicates.

## Current-model completeness and DeepSeek V4 safety

The correlated-profile audit also covers model paths that were advertised by
Collector V2 but previously fell back to the broad default grid or produced no
model-specific cases:

- Qwen3.5 dense 0.8B/2B and 4B/9B share their exact attention topologies.
- Qwen3.5-122B-A10B has exact attention and MoE profiles.
- MiniMax M2/M2.5/M2.7 share one exact attention topology.
- Qwen3-30B-A3B includes its valid TP8 attention point, and the Qwen3
  235B-2507 artifact resolves to the existing 235B MoE shape.

DeepSeek V4 has three additional population constraints:

1. The persisted module and top-k calibration keys do not contain enough
   model geometry to distinguish Flash from Pro. Full/raw collection therefore
   uses one canonical `sgl-project/DeepSeek-V4-Flash-FP8` profile and never
   combines both models in one output. Targeted native and FP8 artifact paths
   remain supported.
2. The model YAML is the source of truth for backend-specific operations.
   SGLang schedules CSA/HCA context and generation modules, top-k calibration,
   mHC, MoE, and WideEP MoE. vLLM's DSV4 collectors remain registry-only until
   the declared runtime compatibility floor and prefix-aware context contract
   are both satisfied. TRT-LLM continues to schedule only the operations its
   registry implements.
3. MoE artifact aliases are not merged across quantization formats. Native
   DeepSeek V4 artifacts retain only `w4a8_mxfp4_mxfp8` for TRT-LLM, while the
   converted `sgl-project/*-FP8` artifacts retain only `fp8_block`. Pinned
   SGLang 0.5.10 and vLLM 0.19.0 native mode lists are explicitly empty instead
   of advertising unverified future-version paths.

The NVIDIA DeepSeek-V4 NVFP4 checkpoints are intentionally not advertised by
this Collector-only change. The current AIC model catalog and the DSV4 module
collector do not yet define an end-to-end NVFP4 artifact contract; adding only
a standalone MoE shape would create a plan that AIC cannot consume correctly.

mHC keeps native and converted artifacts separate in the shared recipes.
SGLang and vLLM collapse only identical phase/hidden-size/hc-mult groups in
their operation-local getters before sweeping token counts; a targeted run
first filters to the requested artifact, so that artifact remains the
representative.

## Review checklist

When adding or changing a model profile:

1. Identify whether the collector loads the model or only uses its dimensions.
2. Keep correlated dimensions in one profile; do not append them to unrelated
   global axes.
3. State whether quantization is selected by the model artifact or expanded by
   the collector.
4. Compare physical key sets, not only totals.
5. Derive benchmark invocation identity before deciding that artifact names or
   quantization variants are duplicates.
6. Verify targeted model plans do not activate unrelated base operations or
   inherit unrelated default profiles.
7. Keep a synthetic default point when an unchanged consumer still queries it,
   even if the current model metadata would choose a different value.
8. Classify every omitted case as out-of-scope, not-applicable, or known-unsafe;
   record exact invocation scope for the latter and do not inherit it across a
   version, backend, or GPU-path change without a fresh boundary probe.

## Validation

```bash
pytest -q tests/unit/collector
ruff check collector tests/unit/collector
ruff format --check collector tests/unit/collector
git diff --check
```

The final Collector suite reports 299 passed and 2 skipped. The unit coverage
checks final profile expansion, per-operation recipe counts, selector narrowing,
alias handling, hardware/precision boundaries, and operation-local physical
deduplication. The historical key-set comparison above was a one-time read-only
audit, not an ongoing Collector behavior or exact-count unit-test dependency.
