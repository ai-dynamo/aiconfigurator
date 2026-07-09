# Collector Upgrade Playbook

Use this playbook to move one AIC collector to one exact framework release and
validate it on one GPU platform. Keep framework upgrades separate: finish and
review SGLang, TensorRT-LLM, or vLLM independently instead of combining all
three in one change.

The tracked agent workflow is
[`aic-auto-collect`](../../.agents/skills/aic-auto-collect/SKILL.md). This
document is the project source of truth; the skill turns it into an executable
workflow.

## 1. Define the contract

Record these decisions before changing code:

- base branch or dependent PR;
- one target framework and exact version;
- official runtime image tag and digest;
- target GPU system and SM version;
- stock versus special/WideEP runtime;
- code-only collector upgrade versus performance-data delivery;
- whether compatibility is exact-version-only;
- operations and model families in scope;
- what may be committed, pushed, or opened as a PR.

`collector/framework_manifest.yaml` is the source of truth for the stock
framework pin and image. Module-level `__compat__` metadata may narrow that
contract but must not silently widen it. Treat stock and WideEP/special images
as separate runtimes with separate evidence.

Do not silently broaden the work to another framework, XPU, WideEP, host
upgrades, or performance-data publication. If the work is stacked on another
PR, keep that dependency explicit when creating the branch and eventual PR.

For a fixed-release collector, prefer an exact version contract when the
framework API and kernel selection changed materially. Do not add compatibility
branches for versions that the project no longer intends to collect.

## 2. Pin and prove the runtime

Run framework inspection and GPU collection inside the target container. Do
not infer the runtime from the image tag alone. Record:

- image tag and immutable digest;
- framework package version and source revision, when available;
- CUDA/runtime versions and GPU compute capability;
- driver-visible GPU name, memory, and power limit;
- collection command, output root, checkpoint root, and log root.

Treat the host as immutable. Do not upgrade the host driver, CUDA toolkit,
kernel, container toolkit, or system packages to make a new image work. Try an
official image variant for the same framework release first. If none is
compatible, report the evidence and agree on a framework/image fallback.

Before every GPU run, inspect `docker ps`, `nvidia-smi`, active GPU processes,
shared memory, and disk space. Residual collector workers commonly produce
false OOMs. Clean only containers and processes created by the current task;
never stop or kill another user's container, restart Docker, or run a global
prune.

## 3. Establish framework backend truth

Collector labels and historical comments are not authoritative. For every
per-model path affected by the upgrade:

1. inspect the pinned framework source inside the container;
2. run the framework's selector or a minimal runtime probe;
3. identify the actual backend/kernel for the model, phase, dtype, quant mode,
   head dimensions, and SM;
4. compare that result with the collector, persisted `kernel_source`, SDK
   resolver, and database lookup key;
5. add a focused regression for each corrected mismatch.

Pay special attention to attention backends, asymmetric Q/K and V dimensions,
sliding-window or sink semantics, checkpoint-native quantization, routing
functions, TP-local dimensions, and architecture-specific module paths. A
collector that runs the wrong model class or backend can produce plausible but
invalid data.

Keep platform decisions explicit. The current priority order is:

1. SM90 Hopper execution and SM100 datacenter Blackwell;
2. SM120 RTX Blackwell;
3. SM89 Ada.

Keep SM103 distinct from SM100 whenever the framework selector does. Mark
source-derived conclusions as hardware-unvalidated until that platform has
actually run them. Do not encode `>= 100` when SM100/103 and SM120 take
different paths, and do not use a catch-all branch to advertise unknown future
architectures.

## 4. Audit the case plan before running GPUs

Use Collector V2 model cases as the release plan. In this repository, full
collection means the complete retained plan after intentional pruning:

```bash
python3 collector/collect.py \
  --backend <backend> \
  --model-cases-full \
  --sm <sm> \
  --plan-only
```

It does not mean the raw backend-registry Cartesian run with no model-case
flags. Preserve the pruning invariants documented in
[`collector-v2-population-design.md`](collector-v2-population-design.md).

For each operation, record raw and retained task counts, physical-key counts,
model/artifact coverage, dtype counts, and SM exception counts. Quant-sensitive
artifacts remain separate when the checkpoint changes the executed kernel.
Verify that every required model emits at least one executable case; a
successful empty plan is a failure.

Keep a small work log outside generated data. It should contain:

| Field | Required evidence |
| --- | --- |
| Runtime | image digest, framework version, source SHA |
| Platform | system name, GPU name, SM, driver, GPU count |
| Plan | op, raw/retained/unique counts, expected filters |
| Smoke | command, representative cases, result |
| Full run | checkpoint, output, row count, failures |
| Backend truth | model, phase, dtype, SM, selected kernel, source/probe |
| Decision | accepted limitation, code fix, or follow-up |

## 5. Progress from smoke to full collection

For each operation, use deterministic smoke coverage rather than relying only
on a few random cases. Cover representative and boundary combinations of:

- context and generation phases;
- short and long sequence lengths;
- small, normal, and capacity-bound batches;
- BF16/FP8/FP4/INT4 modes that the artifact really supports;
- TP/EP boundaries and TP-local alignment;
- dense, MoE, multimodal, sliding-window, and model-specific families.

Run focused collector/unit tests and source probes first, then multi-case smoke,
then the complete retained plan with stable checkpoints. Put expensive or
failure-sensitive MoE collection last so case-enumeration fixes do not
invalidate earlier work.

During a slow operation, inspect progress at roughly 10%-20% increments based
on its observed speed. After the operation finishes, verify its checkpoint,
summary, output shape, errors, container exit, and GPU state, then continue to
the next operation without waiting for approval unless a real anomaly appears.

Use one output/checkpoint namespace per framework, version, platform, and
operation. Resume only when the current plan still contains the completed task
IDs. If enumeration changes, prove which completed IDs remain valid and keep a
hash-verified checkpoint backup before migration.

## 6. Classify failures before changing code

Record the exact case, exception, runtime state, and selected framework path.
Classify failures as collector integration, unsupported configuration,
framework/kernel defect, resource/capacity boundary, or transient environment.
Also record the nearest same-family successes across the relevant shape,
dtype/backend, and TP/EP boundaries. A skip or guard inferred only from failed
points is incomplete evidence; it must preserve those positive controls.

Before editing, trace the regression across the current base commit, the last
validated platform branch, and the working-tree diff. Record the introducing
or dropping commit when it can be identified, especially when Hopper and
Blackwell work are stacked. Distinguish restoring a lost contract from adding
new behavior so that a platform-specific fix is not repeatedly removed and
reintroduced by later rebases.

For SGLang 0.5.14 Hopper/Blackwell continuation work, maintain the platform
alignment ledger — a local campaign record kept outside the repo — in
addition to the task-local run log.

Use these escalation heuristics with judgment:

- a few isolated failures or a few dozen among hundreds/thousands may be
  acceptable when explained and unclustered;
- investigate around 10% unexpected failures, or earlier when they cluster by
  op, kernel, dtype, model, shape family, or SM;
- treat roughly one-third or more failures, or an entire family failing, as a
  systemic problem.

An OOM is not automatically an expected failure. First prove the GPU is clean,
then reproduce the case and distinguish a true capacity edge from leaked
weights, descriptor/JIT caches, stale workers, or an oversized dummy setup.

Fix only failures proven to come from the target framework API or collector
behavior. Prefer narrow source-backed predicates and fail-closed output guards.
Do not make a run green by adding broad skips, shrinking intended coverage,
inventing rows, weakening benchmark repetitions, or adding speculative helper
abstractions.

## 7. Validate every completed operation

For each operation, compare:

- requested, done, failed, and expected-failed checkpoint IDs;
- CSV/parquet row count and unique persisted keys;
- finite latency (positive unless the operation contract explicitly permits a
  modeled zero) and valid power fields when applicable;
- expected columns, framework version, model architecture, dtype, and kernel;
- task keys versus persisted output keys;
- duplicates, malformed rows, and partial writes;
- task-owned container exit and final GPU state.

Do not describe a staged collection as one globally green run when an accepted
stage exited nonzero. Preserve the per-stage summaries and disclose accepted
capacity failures precisely.

## 8. Separate collector code from data delivery

A working collector and a consumable perf database are different deliverables.
For data delivery:

1. map the measured GPU to a real AIC system definition; do not relabel one GPU
   product as a nearby product with the same SM;
2. finalize staging files to parquet only after checkpoint failures are
   resolved or explicitly accepted by the delivery contract;
3. place data under
   `src/aiconfigurator/systems/data/<system>/<backend>/<version>/`;
4. regenerate kernel-source metadata when sources change;
5. run parquet diff/review tools and loader tests;
6. test representative Python and Rust consumer queries using the exact
   backend source and dimensions;
7. run a default AIC workflow before claiming model/backend/system support.

When collector changes introduce a new database key, update all producers and
consumers together or version-gate the change. Test untouched frameworks
against their existing packaged data so a single-framework upgrade does not
regress them.

## 9. Validate and hand off

Before handoff, run the relevant collector tests, SDK/database tests, lint and
format checks, native/Rust tests when touched, and a rebuilt-package integration
test. Record environmental exclusions instead of silently omitting tests.
Run fork/parallel collector tests in a fresh process when importing the target
framework before fork can deadlock the combined suite.

Keep unrelated framework upgrades and pre-existing test debt out of the patch.
Use reviewable signed-off commits. Push or create a PR only when authorized;
long collection does not itself grant permission to publish code or data.

The handoff must separate:

- completed collector code and tests;
- hardware-validated platforms;
- source-derived, hardware-unvalidated platform paths;
- collected but unpublished artifacts;
- accepted failures with exact cases;
- future framework, WideEP, and platform work.

## B200 continuation checklist

When continuing a source-derived SM100 plan on a B200 node:

1. start from the same framework-version branch; do not add another framework;
2. record the B200 system identity and fresh runtime provenance;
3. select the registered B200 system when available and assert compute
   capability 10.0 at runtime instead of trusting only `--sm 100`;
4. regenerate the SM100 model plan and compare its expanded case IDs, counts,
   and exceptions with the source audit (`--plan-only` is an overview, not a
   complete expanded-case artifact);
5. runtime-probe per-model backend selection before collecting;
6. smoke SM100-specific dense attention, encoder attention, FP8/FP4 GEMM, MoE,
   DSA/DSV4, and known reduced-head or alignment boundaries;
7. revalidate source-derived SM100 exceptions, especially reduced-head DSA
   generation at long KV lengths; do not claim full support while such a path
   remains skipped only because of collector dummy setup;
8. remove a source-derived label only after the corresponding B200 path passes;
9. run the full retained SM100 plan with MoE last;
10. publish B200 data only under the matching B200 system definition.

Do not reuse Hopper capacity skips or performance values on B200. A shared
framework version and similar case shape do not prove identical backend,
alignment, memory, or performance behavior.
