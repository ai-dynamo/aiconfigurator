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
    -> apply declared artifact quantization policy
    -> apply universal mathematics and positive hardware floors
    -> derive operation-local invocation identity
    -> stable deduplication of proven-equivalent invocations
    -> apply registry maturity and hang-only denylist policy
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

## Population, maturity, and observed outcomes

Collector V2 follows the repository rule introduced by PR #1302:

```text
plan cases = dedup(expand(base sweeps) + expand(model shapes))
runnable   = plan cases intersect positive hardware floors
             minus registry-unverified operations and the hang denylist
result     = performance rows plus classified failure records
```

Release coverage is declared in base/model YAML. Omitting TP16/32 there means
only that the release does not collect those axes; it is not an unsupported
hardware claim. Artifact `allowed_modes`, model head divisibility, and universal
TP/EP mathematics also belong to declaration/population because they describe
what the requested measurement is.

Ordinary runtime failures do not feed back into population. Framework-version
gaps, backend alignment errors, isolated OOMs, and cases that may work after a
backend or release change remain queued and become failure records. There is no
declarative expected-failure layer and no automatic failure-to-skip loop.

The only operation-local pre-queue filter is memory feasibility: the getter may
compare an operation-specific footprint with live device capacity, and must log
the dropped count and budget. A fixed shape threshold inferred from one GPU,
framework release, or backend is not such a filter.

For a grouped collector, the getter must attach its exact retained inner-shape
manifest to the queued outer case. The worker consumes that manifest instead
of silently rebuilding and filtering a larger grid, so task IDs and resume
checkpoints bind the work that can produce rows. SGLang 0.5.14 DSV4 context is
the first use: it retains a cell only when the mandatory BF16 hidden-state
lower bound `batch_size * sequence_length * hidden_size * 2` is at most 80% of
the smallest visible device's total memory. The model config owns
`hidden_size`; generation and chunked sparse-score collectors have different
footprints and do not inherit this formula.

Registry `unverified=True` parks an operation that has not been debugged on one
backend; `unverified_sms=(...)` parks it only on named SMs. These are operation
maturity markers, not shape selectors. `denylist.yaml` is only for exact cases
that hang or kill the node, with date and reason. A CUDA exception that the
worker can record and recover from remains an ordinary attempted failure.

## Ownership model

| Owner | May decide | Must not decide |
| --- | --- | --- |
| Base-op YAML | Sweep density/ranges and positive precision min-SM floors | Per-model shapes, framework gaps, or negative drop rules |
| Model YAML | Correlated model shapes, artifact/quant policy, op activation, explicit release coverage | Historical failure rules or a second selector language |
| Capabilities | Positive dtype/op hardware min-SM facts | Framework-version, backend, model, or shape conditions |
| Backend registry | Version routing and whole-op maturity markers | Shape-level information |
| Framework collector | How the framework-selected invocation runs; persistence and classified raises | Whether a queued case runs, fallback kernels, or silent `continue` |
| Hang denylist | Exact cases that hang or kill the node | Ordinary crashes, OOMs, or expected failures |
| Failure records | Append-only observed outcomes and group summaries | Inputs to any declaration layer |

Framework truth owns backend selection. Prefer the pinned framework's selector;
a manual mapping needs pinned-source evidence, including SM and dtype branches.
Record the backend/kernel that actually ran. If the selected path cannot be
constructed, raise instead of substituting a familiar backend. An unverified
framework kernel limit is parked as `FIXME(kernel-limit)` at the invocation
site while the affected cases continue to produce visible failures.

## AI filter-change gate

Before adding, removing, widening, or relocating a population rule, record:

1. exact framework/version, GPU product/SM, operation, artifact/quantization,
   resolved backend, TP/EP, phase, and shape scope;
2. the current owner, introducing commit, and complete framework/platform blast
   radius;
3. canonical before/after invocation, scheduler-task, and persisted-key sets,
   counts, and hashes;
4. reverse checks for every untouched consumer of a shared input;
5. checkpoint/artifact identity consequences; and
6. the chronological platform-ledger entry when the project maintains one.

Default action for a failing case is no code change: confirm the classified
record and inspect the group. Add a denylist entry only after a clean-GPU hang
or node-kill reproduction with exact signature, post-state, date/reason, and
nearby successful controls. If evidence is incomplete, stop at diagnosis.

Never change a shared axis for one backend failure, add broad TP/EP/SM/model/
dtype/OOM skips, switch away from framework dispatch, weaken benchmark or
failure accounting, add retries/private kernels/process-per-shape workarounds,
reuse a mismatched checkpoint, or touch another framework/SDK/Rust/data
consumer as an incidental fix.

## Outcome and artifact invariants

Every persistence failure must fail its task. Grouped collectors must expose a
zero-row result, timeout, or inner failure rather than allowing the outer task
to look green. Checkpoints and error summaries preserve attempted failures
across resume and bind to the exact framework image/version, collector/config
snapshot, GPU product/SM, model/full-plan scope, and expanded task identity.

An isolated recorded failure or a few dozen failures in a large plan can be an
acceptable observed gap after review. Investigate near 10% or earlier when
failures cluster; one-third or an entire family is systemic. These percentages
are triage signals, never rules that delete cases. Artifact review reports the
exact failed groups alongside row/key/schema/source validation and does not
manufacture or relabel missing measurements.

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
6. Use positive hardware floors or a whole-op registry maturity marker for SM
   gating. Do not encode an SM-specific shape failure as population policy.
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
| MoE common | 1,797 | 4,548 | 4,326 | quant-sensitive artifacts have separate recipe identity; backend policy removes invalid products before execution |
| MLA context specs | 220 | 550 | 220 | SGLang/TRT-LLM getters each emit 1,760 unique loader keys |
| MLA generation specs | 362 | 885 | 362 | SGLang emits 2,656 keys after its int32 KV guard; TRT-LLM emits 2,896 |
| Mamba | 8 | 8 | 12 | four default synthetic interpolation profiles added |
| GDN | 16 | 16 | 16 | unchanged |
| mHC | 8 | 8 | 8 | artifact recipes remain distinct; backend getters collapse them to four phase/shape groups before the unchanged token sweep |
| MLA BMM pre/post | 400 / 448 | 400 / 448 | 400 / 448 | unchanged |

With `COLLECTOR_MODEL_PATH` unset, the SM100 raw public-getter audit separates
candidate getter tasks from actual invocation identity. These counts are
measured after artifact quantization policy, before registry maturity and
runtime subset selection, and before expanding each task's token list:

| Backend | Raw getter tasks before dedupe | Raw getter tasks now | Duplicate tasks removed | Unique invocation/key loss |
|---|---:|---:|---:|---:|
| TRT-LLM | 9,414 | 7,944 | 1,470 | 0 |
| vLLM | 2,799 | 2,352 | 447 | 0 |

The vLLM audit enables the `per_block_fp8`, `nvfp4`, and `mxfp4` runtime
features. Full model plans may narrow these raw tasks later, so the table is not
a post-plan or token-expanded queue count.

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

For a targeted Kimi TP<=8 standalone-MLA plan, SGLang emits 1,100 context /
1,660 generation cases and TRT-LLM emits 1,100 / 1,810. Both retain local heads
`{128, 64, 32, 16, 8}`. The 64-head YAML profile is required for the last
targeted bucket and for `local_heads=1` in full collection; overlap with the
128-head profile is removed only in the backend getter. Kimi's separate dense
MHA profile is vLLM-only and emits no SGLang attention case.

For migrated quant-sensitive MoE families, geometry and checkpoint
quantization are separate identities. New model profiles remain additive, but
a backend schedules only the declared artifact mode rather than taking the
Cartesian product of every shape and every backend quant mode. DeepSeek V4
native artifacts schedule
`w4a8_mxfp4_mxfp8` in TRT-LLM. Exact SGLang 0.5.14 schedules native Flash/Pro
artifacts as `w4a8_mxfp4_mxfp8` on SM100/103; the native Pro artifact's
`w4a16_mxfp4` SM90 path was removed from its allowed modes (owner decision
2026-07-05 — Hopper users take the `sgl-project/DeepSeek-V4-Pro-FP8` build
instead). These are distinct framework-owned paths, not Marlin fallbacks;
SGLang SM120 remains fail-closed until its native path is validated. vLLM
0.24.0 can dispatch the native checkpoints' packed FP4 experts, but the
selected activation/backend is system-dependent: SM90 uses Marlin with BF16
activation (`w4a16_mxfp4`), while SM100/SM103 and SM120 select W4A8 variants.
The SDK currently infers only `w4a8_mxfp4_mxfp8` from the native artifact, and
the persisted MoE key has no system/backend dimension with which to select a
different mode. vLLM therefore schedules no native V4 MoE case rather than
writing an unrequestable or falsely labelled measurement. Supporting it needs
a system-aware consumer contract, not a collector-only SM heuristic. The
converted `sgl-project/*-FP8` artifacts still schedule `fp8_block` in SGLang
and TRT-LLM, but not vLLM: their full-width FP8 expert tensors conflict with
vLLM's FP4 default when `expert_dtype` is absent. Removing those 72 converted
tasks / 1,944 token rows is one vLLM-only pruning stage. vLLM 0.24 also builds
Nemotron Ultra's routed `FusedMoE` at `moe_latent_size=2048`; the outer
`hidden_size=8192` belongs to its projection layers. The shared 8192-wide
profiles remain available to SGLang and TRT-LLM, while vLLM drops another 84
tasks / 2,268 token rows. The SM90 vLLM MoE getter then retains every declared
shape for runtime observation rather than applying version-specific shape
predictors, leaving 1,806 grouped tasks / 48,762 token rows. The pinned vLLM
0.24.0 SM90 full plan contains 277,297 grouped task IDs, including unsupported
shapes that remain queued so their failures are observed. Earlier 347k
artifact-expanded probes predated getter deduplication and are not
checkpoint-count totals.
Kimi-K2-Instruct schedules `fp8_block`, native Kimi-K2.5 schedules
`int4_wo` with group size 32, and NVIDIA Kimi-K2.5 schedules `nvfp4`.

GPT-OSS is also backend- and hardware-specific. Exact SGLang 0.5.14 retains
`w4a16_mxfp4` on SM90 and retains both `w4a16_mxfp4` and
`w4a8_mxfp4_mxfp8` on SM100/103 because the two labels select distinct
activation precisions. Its high-level `FusedMoE + Mxfp4Config` path owns the
FlashInfer API, TP padding, and EP-local expert layout. On SM120 the pinned
runtime would repack these checkpoints through Marlin; that is not an MXFP4
measurement, so the stock collector fails closed and leaves a hardware TODO.
TRT-LLM 1.3.0rc10 retains both modes on SM100/103, skips them on SM120, and
retains only `w4a16_mxfp4` on Hopper; vLLM collects `w4a16_mxfp4`. SGLang
explicitly selects BF16 activation precision for the W4A16 label and its
runtime `default` selects MXFP8 activation for the W4A8 label. GPT-OSS and
DeepSeek V4 use separate framework methods and routing contracts; neither is
implemented by borrowing the other's path.

DSA recipes retain checkpoint paths until the backend resolves setup-time
quantization, because all three frameworks load model-path-specific config.
Exact SGLang 0.5.14 first rejects a checkpoint whose native quantization has no
valid backend on the target SM. It then canonicalizes candidates with the same
persisted `(architecture, timed-module GEMM type)` identity to the
longest-context remaining checkpoint; a targeted model run keeps its requested
checkpoint. Block-FP8 remains separate because its DSA projections execute and
persist as `fp8_block`. This is an explicit, operation-local equivalence proof,
not a general rule that consumer-key collisions are interchangeable.

A BF16 timed-module label still does not make an invalid full-model setup a
measurement of another precision. On SM90, loading a GLM NVFP4 checkpoint would
initialize Marlin before the module benchmark. Marlin is an INT4-WO backend in
this collector contract, so the positive NVFP4 SM100 artifact floor keeps that
checkpoint out of the SM90 runnable set; ordinary GLM DSA uses the BF16
artifact. On SM100/103 a native NVFP4 checkpoint can be a valid setup candidate,
but it is still canonicalized with any consumer-equivalent BF16 path during a
full/raw run rather than producing an indistinguishable second invocation.

SGLang's inner MLA/DSA module sweep reads the same YAML precision specs and SM
gates as the population layer. In particular, Ada/Hopper expand `fp8_block`,
not `nvfp4`; any future NVFP4 module precision must be declared with a
Blackwell `min_sm` gate in YAML.

The following SM100 counts are raw tasks returned by each public getter with
`COLLECTOR_MODEL_PATH` unset, before registry maturity and runtime subset selection. For
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
   Stock SGLang 0.5.14 schedules CSA/HCA context and generation modules, top-k
   calibration, mHC, and MoE (`wideep_moe` is not part of that stock model
   plan; it remains an explicit, independent SGLang 0.5.10 run; WideEP MLA is
   unregistered). vLLM 0.24.0 schedules its four production CSA/HCA module
   paths; disabling DSV4 MoE does not disable DSV4 attention. Standalone
   sparse kernels and mHC remain registry-only until their consumer contracts
   represent vLLM 0.24.0's fused execution. TRT-LLM continues to schedule only
   the operations its registry implements.
3. MoE artifact aliases are not merged across quantization formats. Native
   DeepSeek V4 artifacts retain only `w4a8_mxfp4_mxfp8` for TRT-LLM. Exact
   SGLang 0.5.14 retains the native SM100/103 Flash/Pro W4A8 paths described
   above. Converted `sgl-project/*-FP8` artifacts retain `fp8_block` only in
   frameworks that can interpret their actual FP8 expert tensors; the vLLM
   0.24.0 native mode list remains explicitly empty instead of advertising an
   unsupported or unconsumable path.

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
8. Route every omitted case to its declared coverage/artifact fact, positive
   capability floor, registry maturity marker, live-memory filter, or dated
   hang denylist. Ordinary failures remain attempted and visible.

## Validation

```bash
pytest -q tests/unit/collector
ruff check collector tests/unit/collector
ruff format --check collector tests/unit/collector
git diff --check
```

The final SGLang 0.5.14 validation reports 371 non-fork Collector tests and 26
fork/parallel tests passing in separate fresh processes. The unit coverage
checks final profile expansion, per-operation recipe counts, registry
maturity, alias handling, hardware/precision boundaries, and operation-local
physical deduplication. The historical key-set comparison above was a one-time
read-only audit, not an ongoing Collector behavior or exact-count unit-test
dependency.
