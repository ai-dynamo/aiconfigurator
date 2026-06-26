# Small-prefill ~30ms GPU-idle gap: micro-causality (qwen36, b300, vLLM 0.20.1)

Closes the open micro-causality left by `SMALL_PREFILL_MECHANISM_DISCRIMINATION.md`
/ `SMALL_PREFILL_ENVELOPE_ROOTCAUSE.md`: those established the collector's 1-GPU
sharded 256-tok prefill runs **eager-piecewise at ~33ms** (golden graphed ~16ms),
launch-bound with ~510 eager + 41 graph launches over 40 hybrid layers. This doc
nails **where the idle actually goes** and **whether it is per-step fixed overhead
(amortizable) or per-layer launch tax (not).** No production decision rides on this.

In-image B300, vLLM 0.20.1, `uv run --active`, nsys 2025.5.2. 256-tok standalone
prefill (bs1, past_kv=0), sharded tp4/ep4 on 1 GPU.

## Verdict (one line)

The ~30ms gap is **per-layer CPU launch/dispatch starvation** — the host runs the
eager-piecewise hybrid forward and the GPU sits idle ~80% of the step waiting for
the next launch. It is **NOT** GPU compute, **NOT** GPU-sync lockstep, and **NOT**
a one-time per-step fixed overhead. The per-step fixed overhead is only **~1.6ms**;
the rest scales linearly with layer count (~0.85ms/layer, of which ~0.65ms is idle
launch-gap). So it **cannot be amortized away by collecting more layers** — each
layer brings its own dispatch bubble.

---

## PART 1 — busy/gap split (nsys, 256-tok standalone prefill)

Per `bench_step` NVTX window, steady-state step. `runs/analyze_launch_gap.py` +
`runs/analyze_idle_cause.py`. **nsys inflates absolute wall** (CUPTI overhead); the
busy/idle *ratio* and the host-time split are the robust signal, not the abs ms.

| trace | config | eager | graph | GPU-busy | span | wall(NVTX) | idle=wall−busy | busy/wall |
|---|---|---|---|---|---|---|---|---|
| **A** | plain-vLLM 1-GPU **real MoE**, no marker | 510 | 41 | **8.71ms** | 45.07ms | 45.53ms | **36.82ms** | 19% |
| **B** | collector 1-GPU **moe-noop**, +per-layer marker | 510 | 41 | **3.49ms** | 25.68ms | 28.06ms | **24.57ms** | 12% |

(A = new capture `runs/nsys_ctxA/`; B = reused `runs/nsys_ctx256/`.)

**Verdict 1 — A is GAP-BOUND, not busy-bound.** busy (8.7ms) ≪ wall (45.5ms);
GPU idle = 81% of the step. The "1-GPU is slow" is **not** lack of tensor-parallel
speedup (that would be busy-bound) — it is a genuine idle pathology. → proceed to
characterize the idle.

**Where the idle goes (host-time split inside the A window, wall=45.53ms):**

| host bucket | time | % wall | reading |
|---|---|---|---|
| `cudaLaunchKernel`/`cuLaunchKernelEx`/`cudaGraphLaunch` | 5.94ms | 13.1% | launch-API cost |
| `cudaStreamSynchronize`/`Device`/`Event` Synchronize | 0.23ms | 0.5% | **sync-lockstep ruled out** |
| `cudaMemcpy*` (incl. D2H sampling) | 0.69ms | 1.5% | **D2H ruled out** |
| other runtime API | 0.16ms | 0.4% | — |
| **Python / host compute (in NO cuda call)** | **38.51ms** | **84.6%** | **dominant — host dispatch between kernels** |

The GPU's 937 inter-kernel bubbles total 37.8ms; the large ones each overlap exactly
**one** launch call (host preparing the next launch) plus, at the step edge, the
sampling memcpy/other-API cluster. So the bottleneck is **host-side dispatch
throughput** (the Python/C++ eager dispatch of ~510 small ops across 40 hybrid
Mamba/GDN layers), i.e. **launch-rate**, not slow individual launches, not sync, not
compute. B shows the identical shape (87.3% python, 0.1% sync).

**Marker / moe-noop is NOT the cause of B's gap.** A and B have **identical** launch
structure (510 eager + 41 graph) and both are gap-bound. The nsys idle subtraction
B.idle−A.idle = −12.3ms is confounded (A adds real-MoE busy +5ms and carries higher
profiling overhead: A wall 45.5 vs B 28.1), so it does **not** isolate marker cost.
The clean **untraced** numbers already settle it: collector+marker 33.1ms ≈
collector real-MoE 33.3ms ≈ plain-vLLM no-marker 34.7ms (marker delta ≈ −1.6ms, i.e.
within noise). B's gap is intrinsic to the eager-piecewise path, not a marker/noop
artifact.

---

## PART 2 — layer-count scan (FIXED-overhead hypothesis test)

Same 256-tok prefill, model physically rebuilt with **k active decoder layers**
(`num_hidden_layers=k`, `layer_types[:k]` — every one of the k layers computes; no
representative-subset / identity-forward). RAW measured `execute_model` cuda-event
wall (NOT ×layer_multiplier). `runs/part2_standalone.py`, `runs/part2_standalone.out`.

| k | wall_k (ms) | per-layer = wall/k | intercept/k |
|---|---|---|---|
| 1 | 3.244 | 3.244 | 1.614 |
| 2 | 2.691 | 1.345 | 0.807 |
| 4 | 3.891 | 0.973 | 0.404 |
| 8 | 9.200 | 1.150 | 0.202 |
| 20 | 19.213 | 0.961 | 0.081 |
| 40 | 35.475 | **0.887** | 0.040 |

**Linear fit `wall_k = intercept + slope·k`** (R²=0.996; robust — k≥4-only fit gives
slope 0.858, intercept 1.51):

- **slope (true per-layer latency) = 0.854 ms/layer**
- **intercept (per-step FIXED overhead: warmup+marker+sched+launch-ramp) = 1.61 ms**
- full-40 via slope only (drop intercept) = 34.15ms; intercept+slope·40 = 35.76ms
  (vs measured 35.48ms).

Per-layer (wall/k) **converges downward** 3.24 → 0.89 ms as k grows — single-layer
attribution *is* inflated, but only by the `intercept/k` term, which is tiny
(intercept=1.6ms) and washed out by k≈8.

**Verdict 2 — wall grows ~linearly with a SMALL intercept → the gap is PER-LAYER
(launch/dispatch), not one-time fixed overhead.** It cannot be amortized away: each
added layer costs ~0.85ms, of which only ~0.22ms is real-MoE GPU busy (8.7ms/40) and
~0.63ms is host-dispatch idle. This **refines Ilya's "need many layers"**: many
layers are needed to make per-layer *attribution* converge (kill the intercept/k
inflation), but the *gap itself* does not amortize — it is intrinsic per-layer tax.

---

## RELATE — is B.idle (Part 1) ≈ intercept (Part 2)?

**No.** B.idle ≈ 24.6ms (nsys) / ~29.5ms (untraced 33−3.5) **≫** intercept = 1.6ms.

→ The 30ms is **per-launch / per-layer GPU-starvation**, NOT per-step fixed overhead.
The idle lives in the **slope** (≈0.63ms host-dispatch idle × 40 layers ≈ 25ms), not
the intercept. Both parts agree: Part 1 shows the idle is host-Python dispatch
between kernels distributed across the whole span; Part 2 shows that distribution is
per-layer with a negligible fixed floor. Collecting more layers does **not** shrink
the per-step gap.

**Verdict 3 (the 3 discriminations):** launch-rate (host dispatch) ✓ dominant;
sync-lockstep ✗ (0.5%); per-step fixed overhead ✗ (intercept 1.6ms). The fix lever
remains capture coverage (full CUDA-graph replay collapses the ~510 per-layer host
launches into one `cudaGraphLaunch`, deleting the per-layer dispatch tax) — exactly
golden's 16ms path — consistent with the prior root-cause docs.

## Artifacts
- `runs/nsys_ctxA/` — NEW trace A (plain-vLLM, real MoE, no per-layer marker, in-proc).
- `runs/captureA_realmoe_nomarker.py` — A capture script (VLLM_ENABLE_V1_MULTIPROCESSING=0).
- `runs/nsys_ctx256/` — reused trace B (collector, moe-noop, +marker).
- `runs/analyze_idle_cause.py` — host-time split (launch/sync/memcpy/python) + bubbles.
- `runs/part2_standalone.py`, `runs/part2_standalone.out` — k∈{1,2,4,8,20,40} scan.
- `runs/analyze_launch_gap.py` — per-step eager/graph/busy/span/wall (reused).
