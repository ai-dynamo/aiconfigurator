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

Scheduler semantics are anchored to the vLLM v1 scheduler source, with a
clause-by-clause provenance table in `vllm_sim.py`'s module docstring
(unified token budget :334, running-set-first :346, chunked prefill cap
:372, admission cap :534, waiting-admission allocation failure = put back +
break with no preemption :716-723, running-path preemption :437-471, all in
`vllm/v1/core/sched/scheduler.py`). Per-pass timing = prefill(batch) +
decode(ready) from a pluggable perf model, so scheduling fidelity is
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

Stdlib only — no numpy, no torch, no dynamo required to run the oracle.

## Running the validation gate

```bash
PYTHONPATH=src:tools/queueing_oracle python3 tools/queueing_oracle/validate_formula.py
```

Nine agg config families (isl 512–8192, osl 16–512, concurrency 1–128,
budget 2048–8192, chunked prefill on/off, prefix). Two tiers per case:
the limit-cycle evaluator is GATED (within 10–15%, mostly 0.0%); the
closed-form screening tier is reported with sanity checks (its role is
within-sweep candidate ranking — see docs/design/queueing_model.md §1).

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
- SGLang-style scheduling (dedicated prefill batches, retraction) not yet
  implemented in the oracle (the analytical model's sglang calendar is
  therefore marked unvalidated)
