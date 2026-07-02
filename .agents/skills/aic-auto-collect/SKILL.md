---
name: aic-auto-collect
description: Upgrade or run one AIC/aiconfigurator GPU performance collector for one exact framework version and GPU platform. Use for fixed-version SGLang, TensorRT-LLM, or vLLM collector bring-up; framework backend/kernel audits; SM90/SM100/SM103/SM120/SM89 validation; deterministic smoke tests; resumable full Collector V2 runs; failure triage; artifact validation; and code/data handoff.
---

# AIC Auto Collect

## Goal

Upgrade or run one AIC GPU performance collector for one exact framework
version and GPU platform. Keep implementation, validation, artifact delivery,
and Git authority explicit. Do not mix unrelated framework upgrades or commit
transient run artifacts.

## Lock the scope

Write down the contract before acting:

- repository, base branch or dependent PR, and working branch;
- exactly one backend and one framework version;
- stock or special/WideEP runtime;
- target GPU product, AIC system name, and SM;
- container image tag/digest and framework source revision;
- operations and model families in scope;
- collector-code-only versus performance-data delivery;
- Git permissions: local changes, commit, push, and PR creation are separate
  authorities.

Treat `collector/framework_manifest.yaml` as the stock version/image source of
truth. Module `__compat__` metadata may narrow it but must not silently widen
it. Keep stock and WideEP/special runtimes separate.

Treat requests for multiple frameworks as an ordered queue. Finish one
framework/version change before starting another. Do not update an untouched
framework merely because a common collector file or SDK consumer is nearby.

Prefer an exact version contract when requested. Do not add speculative
compatibility branches for older or future releases.

## Protect the host and other workloads

Run framework inspection and collection inside the pinned container. Treat the
host driver, CUDA toolkit, kernel, container runtime, and system packages as
immutable.

Before every GPU run, inspect:

```bash
docker ps
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
df -h
df -h /dev/shm
```

Use task-unique container names and persistent output/checkpoint/log roots.
Clean only task-owned workers and containers. Never kill or restart another
container, restart Docker, or run a global prune. If another workload blocks
the run, report it.

Treat an OOM as unclassified until the same case fails on a clean GPU. Check
for stale workers, retained weights, descriptor/JIT caches, and oversized dummy
allocations before adding a capacity rule.

## Prove the runtime

Inside the container, record:

- the installed framework version, not only the image tag;
- source SHA when available;
- CUDA/runtime versions;
- GPU name, memory, and compute capability;
- exact command and task-local artifact roots.

If the image cannot run on the host, try an official image variant for the same
framework release. Do not silently modify the host or downgrade the framework.

## Audit framework backend truth

Do not trust existing collector labels, comments, or historical workarounds.
For each affected per-model path:

1. inspect the exact framework source in the container;
2. call the framework selector or instrument a minimal runtime probe;
3. record model/architecture, phase, dtype/quantization, Q/K/V dimensions,
   window/sink semantics, SM, and selected backend/kernel;
4. compare the selection with the collector invocation, persisted
   `kernel_source`, SDK resolver, and Python/Rust database keys;
5. fix only proven mismatches and add focused tests.

Keep `framework default`, `collector simulation`, and `persisted key` distinct.
A benchmark that successfully invokes the wrong model class or backend is
invalid.

Use this platform priority unless the user changes it:

1. SM90 execution and SM100 datacenter Blackwell;
2. SM120 RTX Blackwell;
3. SM89 Ada.

Preserve SM103 as a distinct selector where the framework does. Mark source
inspection as `source-derived / hardware-unvalidated`; change that label only
after a run on the actual platform.

## Generate and audit the plan

Use the model-centric Collector V2 plan:

```bash
python3 collector/collect.py \
  --backend <backend> \
  --model-cases-full \
  --sm <sm> \
  --plan-only
```

Full collection means this complete retained plan after intentional pruning.
It is not the raw registry run with no model-case flags.

Before GPU work, record per-op raw, retained, and unique physical counts;
artifact/model coverage; dtype counts; and SM exception counts. Verify every
required model produces executable cases. Preserve checkpoint-native
quantization identities when they change the invoked kernel.

## Validate progressively

For each operation:

1. run focused unit/contract tests;
2. run source and selector probes;
3. run deterministic multi-case smoke coverage;
4. run representative boundary smoke coverage;
5. launch the complete retained op plan with a stable checkpoint;
6. validate the checkpoint, summary, output, exit status, and GPU state;
7. continue automatically to the next op unless a real anomaly appears.

Cover phases, sequence and batch boundaries, dtypes/quants, TP/EP boundaries,
model families, and SM-specific paths. Do not treat a default four-case random
smoke as sufficient.

Put MoE last unless the user specifies another order. It is expensive and most
sensitive to artifact policy, routing semantics, TP/EP enumeration, and
retained framework caches.

Inspect a slow op at roughly 10%-20% progress increments based on observed
speed. Do not poll every minute and do not pause for approval between healthy
operations.

Use stable resume namespaces:

```bash
python3 collector/collect.py \
  --backend <backend> \
  --model-cases-full \
  --sm <sm> \
  --checkpoint-dir <checkpoint-root> \
  --resume
```

Resume only when the framework pin, code revision, plan, backend/op, and output
target still match. Before migrating a checkpoint after an intentional plan
change, back it up, hash it, and prove which completed task IDs and persisted
keys remain valid.

## Triage failures with evidence

For every failure group, record:

- op, model/artifact, dtype, SM, TP/EP, and shape;
- exact exception and selected framework path;
- GPU/container state;
- classification and evidence;
- action and rerun result.

Classify it as collector integration, unsupported configuration,
framework/kernel defect, resource/capacity boundary, or transient environment.

Use these thresholds as heuristics:

- isolated failures or a few dozen among a large plan can be acceptable when
  explained and unclustered;
- investigate around 10% unexpected failure, or earlier when failures cluster
  by op, backend, dtype, model, shape family, or SM;
- treat roughly one-third or more failures, or an entire family failing, as a
  systemic collector problem.

Do not hide failures with broad skips, retries, generic OOM labels, reduced
coverage, synthetic rows, or weakened benchmarks. Keep predicates narrow and
source-backed. Record a failed result before resetting a worker after a fatal
CUDA error.

## Accept artifacts fail-closed

For every completed op, verify:

- requested/done/failed/expected-failed checkpoint counts;
- output row count, schema, and unique persisted keys;
- task-key versus output-key coverage;
- framework version, architecture, dtype, quant, and kernel provenance;
- finite latency, positive unless the operation contract explicitly permits a
  modeled zero, and valid power when collected;
- no malformed rows, unintended duplicates, or partial writes;
- container exit and final GPU state.

Preserve separate stage summaries. Disclose accepted capacity failures exactly;
do not call a staged run globally green when one stage exited nonzero.

## Verify consumers and untouched frameworks

When a collector adds or changes a database dimension, verify all producers and
consumers that share it. Test the target backend and representative packaged
data for untouched backends so a single-framework upgrade does not regress
them.

Run relevant checks, including:

```bash
ruff check .
ruff format --check .
pytest tests/unit/collector -q
pytest tests/unit/sdk/database -q
```

Run native/Rust tests and a rebuilt-wheel/container integration test when
shared SDK or Rust contracts change. Record environmental exclusions.
Run fork/parallel collector tests in a fresh process when importing the target
framework before fork can deadlock the combined suite.

## Keep code and data delivery explicit

Treat collector code and performance data as separate deliverables unless the
user explicitly combines them.

For a code change, track manifest pins, collector adaptation, case policies,
tests, and durable documentation. Do not commit checkpoints, logs, temporary
probes, or raw generated artifacts.

For data delivery, map measurements to the actual GPU system, finalize parquet,
regenerate kernel-source metadata, review parquet diffs, run Python/Rust lookup
smoke, and test a representative AIC workflow. Never label measurements from
one GPU product as a nearby product merely because both share an SM.

## B200/SM100 validation

When applying a source-derived plan to B200:

1. keep the same framework/version scope;
2. record fresh B200 and container provenance;
3. use the registered B200 system and assert compute capability 10.0 instead of
   trusting only `--sm 100`;
4. regenerate and compare expanded SM100 case IDs/counts/exceptions; do not
   treat `--plan-only` as the complete expanded-case artifact;
5. runtime-probe per-model backend choices;
6. smoke SM100-specific attention, encoder, FP8/FP4 GEMM, MoE, DSA/DSV, and
   alignment/reduced-head boundaries;
7. revalidate source-derived SM100 exceptions, especially reduced-head DSA at
   long KV lengths, before claiming complete B200 support;
8. update comments/tests from source-derived to hardware-validated only for
   paths that passed;
9. run the complete retained SM100 plan with MoE last;
10. publish data only under the matching B200 system definition.

Do not reuse Hopper performance values, capacity skips, or alignment rules
without B200 evidence.

## Hand off truthfully

Report separately:

- code and tests completed;
- hardware-validated platforms;
- source-derived platform paths;
- collected but unpublished artifacts;
- accepted failures with exact cases;
- future frameworks, platforms, and special runtimes;
- validation commands and results.

Commit, push, or create/update a PR only when authorized. Keep dependent PRs
and base branches explicit, use signed-off commits, and do not mix unrelated
framework or baseline-test fixes into the collector upgrade.
