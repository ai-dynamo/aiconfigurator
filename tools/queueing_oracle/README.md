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

Scheduler semantics follow vLLM v1 iteration-level scheduling, transcribed
from the Dynamo mocker's vLLM scheduler (`lib/mocker/src/scheduler/vllm/
core.rs`, v1.3.0): unified token budget over running-then-waiting, chunked
prefill, LIFO/FIFO preemption with recompute, block-level KV accounting with
hash sharing + LRU inactive pool, and per-pass timing = prefill(batch) +
decode(ready) from a pluggable perf model. The oracle itself was verified
against `dynamo.replay` (see below), which anchors the whole validation
chain: analytical model ↔ this oracle ↔ dynamo mocker replay.

## Files

| File | What |
|---|---|
| `vllm_sim.py` | engine core + KV manager + event-driven drivers (agg `Simulator`, P/D `DisaggSimulator`) |
| `workload.py` | synthetic workloads (fixed/poisson arrivals, shared-prefix groups) + mooncake-style jsonl trace loader |
| `metrics.py` | TTFT/ITL/E2E/queue percentile summaries |
| `run.py` | standalone CLI |
| `validate_formula.py` | **the gate**: `sdk.queueing` vs this oracle, identical timing on both sides |
| `compare_mocker.py` | column-for-column check of this oracle vs a `dynamo.replay` report JSON |

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

## Oracle fidelity vs dynamo.replay

With identical timing functions, this oracle reproduces `dynamo.replay`
offline results on uniform closed-loop workloads to 0.0% on every reported
metric (TTFT/TTST/ITL/TPOT/E2E mean and percentiles, throughput; 200 and
1000 requests; polynomial and database-backed timing):

```bash
python -m dynamo.replay --input-tokens 4096 --output-tokens 256 \
    --request-count 1000 --replay-mode offline --replay-concurrency 32 \
    --num-workers 1 --extra-engine-args '{"block_size":64}' \
    --report-json /tmp/replay.json
python3 compare_mocker.py /tmp/replay.json --request-count 1000
```

For disagg, the oracle matches replay on ITL (0.1%), throughput (0.4%) and
TTFT min/p99 once first-token semantics are aligned (the user-visible first
token is emitted by the decode stage; the KV handoff delay defers decode
enqueue). Residual TTFT-mean differences under `kv_router` dispatch stem
from router-side admission queueing, which the oracle intentionally does not
model (round-robin dispatch only).

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
