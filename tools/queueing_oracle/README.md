<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Queueing-model oracle — reference discrete-event scheduler simulation

A compact, stdlib-only discrete-event simulation of a vLLM-style
continuous-batching engine. It exists to **validate `sdk/queueing`** (the
pass-calendar model): the oracle executes the actual scheduling loop
event-by-event, so any residual between the analytical model and the oracle
is a scheduling-semantics error in the model, not a timing error.

Scheduling semantics are anchored clause-by-clause to the vLLM v1 scheduler
(conceptual clause list in `vllm_sim.py`'s module docstring; the provenance
table and recorded validation results live in
`docs/design/queueing_model.md` §5). Per-pass timing = prefill(batch) +
decode(rows) from a pluggable perf model, so scheduling fidelity is
separable from timing fidelity. Validation chain: analytical model ↔ this
oracle ↔ vLLM v1 source.

## Files

| File | What |
|---|---|
| `vllm_sim.py` | engine core + KV manager + event-driven drivers (agg `Simulator`, P/D `DisaggSimulator`) |
| `workload.py` | synthetic workloads (fixed/poisson arrivals, shared-prefix groups) + mooncake-style jsonl trace loader |
| `metrics.py` | TTFT/ITL/E2E/queue percentile summaries |
| `run.py` | standalone CLI |
| `validate_formula.py` | **the gate**: `sdk.queueing` vs this oracle, identical timing on both sides |

Stdlib only — no numpy, no torch, no engine install required.

## Running the validation gate

```bash
PYTHONPATH=src:tools/queueing_oracle python3 tools/queueing_oracle/validate_formula.py
```

Nine agg config families (isl 512–8192, osl 16–512, concurrency 1–128,
budget 2048–8192, chunked prefill on/off, prefix). Two tiers per case:
the limit-cycle evaluator is GATED (within 10–15%, mostly 0.0%); the
closed-form screening tier is reported with sanity checks (its role is
within-sweep candidate ranking — see docs/design/queueing_model.md §1).

The gate also runs in CI as a marked test
(`tests/unit/sdk/queueing/test_oracle_gate.py`), so upstream
scheduler-semantics drift in `sdk/queueing` shows up as a failing check,
not a stale doc claim.

## Disaggregated (P/D) serving

`DisaggSimulator` follows the disagg serving flow: the prefill pool
computes the prompt and produces the first token (the TTFT token), the KV
cache is handed off to a decode worker, and the decode pool continues
from token 2 — the handoff appears as the first ITL gap, not in TTFT. On
the decode worker the transferred KV counts as computed tokens (the
KV-connector convention), so decode-side passes never bill prefill
compute. Dispatch is round-robin per pool; router-level policies
(affinity, queue-depth admission) are out of scope.

KV-transfer time is **computed, not configured**: each handoff is a flow
on the `TransferFabric`, and concurrent flows share per-worker NIC
bandwidth max-min fairly — fan-out (a clump of completions leaving one
prefill worker's egress) and fan-in (several prefill workers landing on
one decode worker's ingress) slow each other down by the computed fair
share. Nominal bandwidths come from the AIC system spec
(`node.inter_node_bw` / `node.intra_node_bw`, Byte/s per GPU) via
`sysspec.transfer_spec_from_system(system, kv_bytes_per_token)`, de-rated
by `bw_efficiency` (default 0.8, following the spec's own
`mem_bw_empirical_scaling_factor` convention). Not modeled: shared-fabric
topology beyond per-worker NICs, and transfer/compute interference on the
GPU itself.

## Standalone CLI examples

```bash
# closed-loop synthetic
python3 run.py --request-count 200 --isl 4096 --osl 256 --concurrency 32

# prefix sharing
python3 run.py --request-count 300 --isl 5000 --osl 200 --concurrency 32 \
    --shared-prefix-ratio 0.5 --num-prefix-groups 8

# KV pressure -> preemption
python3 run.py --request-count 200 --isl 4096 --osl 512 --concurrency 128 \
    --num-gpu-blocks 2048

# naive token-count KV accounting ablation
python3 run.py ... --kv-mode token

# mooncake-style trace
python3 run.py --trace trace.jsonl --trace-block-size 512
```

## Known simplifications

- generated full blocks are freed anonymously (no multi-turn reuse of
  generated text; prompt-block reuse via trace/group hashes is modeled)
- dispatch is round-robin; router-level policies are out of scope
- no watermark logic beyond capacity-exhaustion preemption
- speculative decoding, attention-DP prefill cadence, and async scheduling
  are not modeled (the same scope boundaries as the analytical model —
  docs/design/queueing_model.md §6)
- SGLang-style scheduling (dedicated prefill batches, retraction) not yet
  implemented in the oracle (the analytical model's sglang calendar is
  therefore marked unvalidated)

## Extending

The intended growth direction is more gate coverage, not more simulator:
variable-length workloads, more disagg families, and — the highest-value
step — swapping the semantic authority from this hand-written core to the
engine's own scheduler while keeping the same gate harness.

**Real-`Scheduler` driver (assessed, deferred to a follow-up):** vLLM's
own unit tests construct a live `Scheduler` GPU-free
(`tests/v1/core/utils.py::create_scheduler`: `ModelConfig` +
`SchedulerConfig` + `CacheConfig`, no model weights, tokenizer skippable).
The driver is ~300 lines: `add_request` → `schedule()` → read
`num_scheduled_tokens` per request (the exact prefill-chunk/decode-row
split) → price the pass with the same pluggable perf model → advance a
virtual clock → feed a fabricated `ModelRunnerOutput` to
`update_from_output` → repeat. Output shape matches `des_agg_stats`, so
the gate swaps oracle backends without changing tolerances. What it
needs that this repo's unit CI deliberately does not have: an importable
`vllm` (+ torch) pinned to the perf-DB version, and a cached HF config
for the placeholder model — i.e. a separate CI job on a vLLM image, the
`tools/generator_validator` pattern. Until then, this DES core remains
the reference, with its semantics audited clause-by-clause against the
scheduler source (§5 of the design doc).
