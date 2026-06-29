# HANDOFF — close the small-prefill micro-causality account (qwen36, B300×8, vLLM 0.20.1)

**You are picking this up on a GPU compute node.** A prior agent did all the
analysis that needs no GPU (parsing existing nsys sqlite traces on the head node).
Your job is the GPU-side runs that the head node could not do, then **close the
arithmetic**. Do NOT write a verdict/mechanism doc until the columns sum to wall ±10%.
If it doesn't close, report the residual + which item wasn't measured.

Goal restated: build one table — golden vs collector — with columns
`wall / compute / real-comm / spin / GPU-idle` on **one de-inflated time base** that
**adds up to wall (±10%)**, and answer A/B/C in one line each. Constraints: `uv run
--active`; **nsys must be 2025.5.2** (the existing golden + collector traces are
2025.5.2.266 — verify `nsys --version` before profiling, else it's not apples-to-apples);
report raw numbers.

Working dir in container: `/workspace/repo/aiconfigurator` (= `/lustre/fsw/coreai_comparch_inferencex/simonec/repo/aiconfigurator`).
All scripts below already exist in `runs/`.

---

## WHAT IS ALREADY ESTABLISHED (closed from existing 2025.5.2 traces — trust these)

### Golden tp4/ep4 ISOLATED ≤512/256-tok step — FULLY CLOSED
Parsed from `runs/serve_nsys_trace.sqlite` (golden serve driven by `serve_probe.py` =
one isolated prefill at a time) with **merged-interval occupancy** (not summed
durations). Reproduce: `python runs/closeacct_golden.py`.

| term | ms | note |
|---|---:|---|
| step span (**wall**) | **42.6** | this trace is the *isolated* branch |
| merged GPU-busy | 33.2 | ← the "34ms GPU-busy" |
|  • real compute (non-AR) | 5.9 | |
|  • all-reduce kernel | 27.2 | = real-comm floor **1.85** + **spin 25.4** |
| GPU-idle | 9.7 | |
| closure | 5.9+27.2+9.7 = 42.9 ≈ 42.6 ✓ | |

### All-reduce is SPIN, not comm (resolves the "fake busy")
`cross_device_reduce_2stage` per-call p25: **22.8µs CAPTURED → 743µs EAGER** (32×)
while message size only 2×; within ONE captured step the 81 same-size calls span
22.8→1088µs (48×). Identical data, wild duration variance ⇒ it tracks launch-mode /
rank-skew (spin-wait), **not** data volume. Real transfer ≈ p25 ≈ floor.

### Collector 1-GPU real-MoE busy backbone (AR=0)
From `runs/partB_realmoe/...sqlite` (1-GPU, tp1/ep4-sharded, **nsys 2025.5.2**):
merged compute-busy = **8.4 / 9.9 / 11.9 ms** at 128/256/512 tok. BUT these traces
were profiled WITH per-layer markers (`enable_layerwise_nvtx_tracing`), so their
*wall* is marker-contaminated (99ms idle = marker sync) — **unusable as the 33ms wall.**

### Reframing already supported by the above
- **(A)** "34ms busy vs 16ms wall" is NOT one run. Where busy=33, wall=42.6 (idle 9.7) —
  consistent, no impossibility. The 34ms is **spin (25.4) + compute (5.9) + comm (1.85)**,
  not nsys inflation. 16ms is a *different regime*.
- **(C)** existing golden nsys = isolated 42.6ms; 16ms must be the saturated/stream regime
  (not in this trace).

### Comm formula sanity (from code)
`collector/network/collect_all_reduce.py` times the all-reduce **kernel** on real GPUs
via CUDA events vs message_size, ranks in lockstep → measures the **floor** (no spin).
So `comm` DOES include the all-reduce kernel time; spin is correctly excluded.
Golden `pass_config`: `fuse_allreduce_rms=false, fuse_gemm_comms=false, enable_sp=false`
→ AR is **serial on the critical path** (not overlapped) → `compute + comm` is additive.

---

## WHAT YOU MUST RUN (open items — need the GPU)

Each vLLM load is ~8–12 min. Real model snapshot (container path):
`/workspace/models/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0`

### RUN 1 — golden tp4 isolated vs stream, UN-PROFILED (resolves C, items 1+5)
```
cd /workspace/repo/aiconfigurator
MNBT=2048 uv run --active python runs/period_golden.py 2>&1 | tee runs/period_golden.out
```
Reports per nt∈{256,512}: ISOLATED per-step wall, STREAM per-req & per-step wall.
**Expectation to test:** ISOLATED ≈ 40ms, STREAM ≈ 16ms → 16ms is serving/occupancy,
not reproducible by isolated 1-GPU collection ⇒ fix = calibration, not collection-mode
swap. If ISOLATED already ≈16 → the 42.6ms trace was probe/profiler-distorted; pivot to
the nsys-inflation branch.
NOTE on packing: at MNBT=2048 a 256-tok stream packs 8 prefills/step, so per-step is an
8×256 step. For a clean "1 prefill/step, only occupancy differs" comparison also run
`MNBT=512` (512-tok = 1/step; 256-tok = 2/step). Use both to separate throughput-packing
from pure pipelining.

### RUN 2 — golden STREAM under nsys 2025.5.2 (decompose the 16ms regime + inflation)
```
cd /workspace/repo/aiconfigurator
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node \
  --force-overwrite=true -o runs/golden_stream_nsys \
  bash -lc 'MNBT=2048 STREAM_ONLY=1 uv run --active python runs/period_golden.py'
nsys export --type sqlite --force-overwrite true \
  --output runs/golden_stream_nsys.sqlite runs/golden_stream_nsys.nsys-rep
python runs/closeacct_stream.py runs/golden_stream_nsys.sqlite 0.3 1500
```
- `closeacct_stream.py` segments steady stream steps by AR gaps and prints
  `span = compute + AR(comm+spin) + idle` with a CLOSURE line. **Hypothesis:** in the
  saturated stream, spin→~0 and idle→~small; span should land near the RUN-1 STREAM
  per-step wall.
- **nsys inflation factor:** `period_golden.py` prints its own STREAM walls *under nsys*
  here; compare to RUN-1 (no nsys). `inflation = wall_under_nsys / wall_unprofiled`.
  De-inflate the nsys-derived compute/comm/idle by this factor before putting them in the
  final table. (This is how A's "inflation vs spin" gets quantified: spin is already
  isolated above; inflation is this ratio.)

### RUN 3 — collector live-step (stream-mode) ctx, the user's proposed fix (item B/fix-test)
Template = `runs/partB_realmoe.sh`. Add the live-step driver + a host-pipelined latency
source so the collector measures a *continuous-batching* per-step wall instead of the
isolated execute_model wall. Read `runs/../collector/layerwise/vllm/collect.py:195-264`
for the exact flags, then run something like:
```
uv run --active python -m collector.layerwise.vllm.collect \
  --models "Qwen/Qwen3.6-35B-A3B" --tp-sizes 4 --ep-sizes 4 \
  --phases ctx --ctx-new-tokens 256,512 --ctx-past-kv 0 \
  --max-num-batched-tokens 2048 --moe-real-router \
  --live-step-driver --latency-source schedule_to_update \
  --ctx-warmup-runs 2 --ctx-measured-runs 6 --run-dir runs/collector_livestep
```
Purpose: does stream-mode 1-GPU collection reproduce golden's 16ms? Remember the 1-GPU
collector has **no all-reduce** — you must ADD comm from `collect_all_reduce.py`
(tp4, small message ≈ hidden×tokens×2B, latency-bound, × per-step AR count ≈ 2×layers).
Compare `collector_livestep busy + comm` to golden 16ms; the residual is the open ~5ms.

### RUN 4 — causal occupancy knob (item 6)
Cheapest version reuses RUN 1 on 1-GPU: `TP=1 WEIGHTS=dummy MNBT=512 uv run --active
python runs/period_golden.py`. If ISOLATED≈33 and STREAM≈16 on the SAME 1-GPU engine
(no real all-reduce involved), that *causally* proves occupancy hides launch idle.
Stronger version: keep ISOLATED but launch a long dummy kernel on a side stream to fill
the GPU and re-measure — wall should drop; remove it → rises. (Optional if RUN 1 tp1
already shows the drop.)

---

## HOW TO CLOSE THE ACCOUNT (the deliverable)

Build this table on ONE de-inflated base (divide nsys-derived terms by the RUN-2
inflation factor). Both rows must sum to wall ±10%:

| scenario | wall | compute | real-comm | spin | GPU-idle |
|---|---|---|---|---|---|
| golden ISOLATED (have) | 42.6 | 5.9 | 1.85 | 25.4 | 9.7 | ✓ closes |
| golden STREAM (RUN 2) | ~16? | ? | ? | ~0? | ~? |  |
| collector ISOLATED (RUN 3 baseline / known ~33) | ~33 | ~8-12 | 0(1GPU) | 0 | ~20-25 |  |
| collector LIVE-STEP+comm (RUN 3) | ? | 8-12 | +AR | 0 | ? |  |

Then answer in one line each:
- **(A)** of the 34ms busy: X ms is nsys inflation (RUN-2 ratio), Y ms is spin (25.4) —
  give both numbers.
- **(B)** the 30ms collector idle vs golden hiding it: attribute to occupancy /
  cheap-launch / batching using RUN-1 (stream vs isolated) + RUN-4 (knob).
- **(C)** 16ms is isolated-reproducible or serving-only: from RUN-1 (iso vs stream).

**Only write a mechanism/verdict doc if both rows close ±10%.** Otherwise report the
residual (likely the ~5ms = dispatch-residual vs barrier-skew vs MoE shard 1.8×, see
`BUSY_METRIC_VERDICT_PARTD_RESOLUTION.md`) and say which run would pin it.

---

## SCRIPTS (all in `runs/`, already written)
- `period_golden.py` — un-profiled isolated/stream wall. env: `TP MNBT NREQ WEIGHTS(real|dummy) STREAM_ONLY ISO_ONLY`.
- `closeacct_golden.py` — parses the existing golden serve trace (hardcoded for serve_nsys_trace.sqlite).
- `closeacct_stream.py <sqlite> [warmup_frac] [gap_us]` — generalized parser for the FRESH stream trace (RUN 2).
- `closeacct_collector.py <sqlite> [gap_us]` — collector trace parser (bench_step segmented).
- `partA_kernel_class.py`, `partA_moe_perrank.py` — prior per-class / per-rank attribution.

## PRIOR DOCS (read skeptically — their conclusions were not closed; raw numbers ok)
- `BUSY_METRIC_VERDICT.md`, `BUSY_METRIC_VERDICT_PARTD_RESOLUTION.md`.

## KEY CODE POINTERS
- `collector/layerwise/vllm/collect.py:195-264` — `--latency-source {auto,span,gpu,gpu_capped,schedule_to_update,worker_wall,execute_model_gpu}`, `--live-step-driver`.
- `collector/layerwise/vllm/worker.py:1340-1480` — isolated ctx path; `:1647-1770` — live-step/stream driver; `:1267` — `ctx_measure_execute_model_gpu_time` (the 33ms wall).
- `collector/network/collect_all_reduce.py:244-257` — comm (all-reduce kernel) timing vs message_size.
- `collector/helper.py:1201-1224` — `_generate_power_law_distribution` (routing-skew / MoE 1.8× issue).
