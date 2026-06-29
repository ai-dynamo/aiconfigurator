# Small-prefill micro-causality — CLOSED (qwen36 Qwen3.6-35B-A3B, B300×8, vLLM 0.20.1)

**Status: account closes.** Both golden rows sum to wall within ±1%, and the collector
"fix" (stream-mode live-step + added comm) reproduces the golden stream per-request wall
to within **0.6%**. Verdict written per the handoff rule (closure achieved).

Date 2026-06-26. nsys **2025.5.2.266** (matches existing traces). All runs `uv run --active`,
real model snapshot `…/Qwen--Qwen3.6-35B-A3B/…/995ad96…`.

---

## THE CLOSING TABLE (one de-inflated time base; nsys inflation factor = 1.162×)

nsys inflation measured directly: matched 256-tok stream **per-req 16.07 ms under nsys
(RUN 2b) vs 13.83 ms un-profiled (RUN 1)** → **1.162×**.

| scenario | wall (ms) | compute | real-comm | spin | GPU-idle | Σ vs wall |
|---|--:|--:|--:|--:|--:|--|
| **golden ISOLATED** (256-tok step, existing serve trace) | **42.6** | 5.9 | 1.85 | 25.4 | 9.7 | 42.85 ✓ (+0.6%) |
| **golden STREAM** (per-req, de-inflated; RUN 2b NVTX window) | **13.83** | 1.18 | 0.06 | 12.08 | 0.51 | 13.83 ✓ |
| **collector ISOLATED** (bs1, 256-tok; RUN 3 + partB gpu-busy) | **33.96** | 5.64 | 0 (1-GPU) | 0 | 28.32 | 33.96 ✓ |
| **collector LIVE-STEP + comm** (bs8 per-req; RUN 3) | **13.74** | 1.21 | 0.06* | 0 | 12.47 | 13.74 ✓ (−0.6% vs golden 13.83) |

\* comm added from the golden-stream real-AR floor (1-GPU collector cannot measure all-reduce).
Golden STREAM under-nsys per-req before de-inflation: 16.07 = 1.37 + 0.07 + 14.04 + 0.59.

---

## ANSWERS

**(A) Of golden's "34 ms busy", how much is nsys-inflation vs spin?**
The 33.2 ms merged GPU-busy in the isolated step is **real GPU-kernel occupancy, not nsys
inflation**: **spin = 25.4 ms (76%)**, compute 5.9, real-comm 1.85. nsys inflation is
**1.162×** and lands in the **wall/host-gap (idle), ≈ 5.9 ms of the 42.6 ms wall** — it does
**not** inflate kernel durations (so it does not inflate the busy). So: *inflation ≈ 5.9 ms
(in idle), spin ≈ 25.4 ms (in busy)* — distinct buckets. The "fake busy" is spin, confirmed
by the established p25→max all-reduce duration blow-up (same data, 32–48× duration variance).

**(B) The ~30 ms collector idle vs golden "hiding" it — mechanism?**
**Occupancy/throughput-packing** amortizes the fixed per-step cost (host launch + all-reduce
barrier) over more tokens. Evidence, all three on the *same* engine:
- RUN 1 golden tp4: isolated 39.6 → stream/req 13.83 (256-tok, pack-8) = **2.86×**.
- RUN 3 collector (tp4-sharded, 1 GPU): isolated **33.96 → live-step/req 13.68** = **2.48×**.
- RUN 4 full-model TP=1 (already GPU-saturated): isolated 96.3 → stream **94.2 = 1.0× (NO drop)**.

RUN 4 is the causal control: when one step already saturates the GPU there is **nothing to
fill**, so streaming does nothing. The collector's 1/4 *shard* under-fills the GPU (idle-bound)
→ streaming fills it; golden tp4's tiny per-rank compute under-fills it between barriers
→ streaming fills it. Packing is the dominant lever: pure pipelining alone (RUN MNBT=768,
512-tok = 1 prefill/step, no packing) only gets 104.4 → 68.1 ms (1.53×); adding packing
(MNBT=2048, pack-4) reaches 27.6 ms (a further 2.5×).

**(C) Is 16 ms isolated-reproducible or serving-only?**
**Serving/stream-only.** Every isolated single-step measurement lands at 34–43 ms
(golden offline 39.6, golden serve-trace 42.6, collector execute_model 33.96). The ~16 ms
only appears as the **throughput-amortized streaming per-request** wall (golden 13.83,
collector live-step 13.68, under-nsys 16.07). An isolated 1-step collection **cannot**
reproduce 16 ms — it is a per-request throughput number, not a step latency.

---

## THE FIX (validated)
The collector's isolated `execute_model_gpu` path reports the 33.96 ms isolated wall, which
is **not** the serving FPM. Running the collector in **`--live-step-driver` (continuous-batch
stepping) with a packed batch** and **adding the all-reduce floor from `collect_all_reduce.py`**
reproduces golden's serving per-request wall to **0.6%** (13.74 vs 13.83). So the calibration
fix is *collection-mode* (stream/live-step + comm), exactly as proposed — not a constant.

## RESIDUAL / CAVEATS (honest)
1. **No ~5 ms residual at the stream per-req level** (collector+comm vs golden = 0.6%). The
   ~5–6 ms gap that *does* exist is in **isolated** mode (collector 33.96 vs golden offline
   39.6) — the collector lacks the tp4 all-reduce; its host-launch idle (28.3 ms) happens to
   sit ~6 ms below golden's spin+idle (35 ms). Both isolated walls are *overhead*-dominated,
   not compute, which is the whole point.
2. **All-reduce kernel differs by regime** (vLLM selects by message size): golden serve
   isolated step uses `cross_device_reduce_2stage` (81/step); the packed offline stream uses
   `two_shot_all_reduce_kernel_inplace` + fused-triton all-reduce (~40/step). The stream
   parser was extended to recognise all variants (`runs/closeacct_stream_nvtx.py`).
3. **Stream decomposition is single-rank** (per-process nsys clock skew left only the
   driver-aligned rank inside the NVTX window). Cross-checked: two other ranks independently
   show 91–92% all-reduce in their own dense windows ⇒ the AR-barrier dominance is real,
   not a one-rank artifact.
4. **Surprise vs handoff hypothesis:** the saturated stream does **not** drive spin→0. It
   crushes *idle* (23%→4%) but the all-reduce **barrier-spin persists and even rises as a
   fraction (≈88% of the packed step)**. This model has ~3B active params; on tp4 the
   per-layer non-overlapped all-reduce (pass_config: `fuse_allreduce_rms=false`,
   `enable_sp=false`) makes the workload **communication/barrier-bound**, not compute-bound.

---

## RUN LEDGER (raw)
- **RUN 1** `runs/period_golden.out` (MNBT=2048) — iso256 39.6 / stream-req256 13.83;
  iso512 105.2 / stream-req512 27.6. `runs/period_golden_mnbt768.out` — 512-tok 1/step:
  iso 104.4 / stream 68.1 (pipelining-only 1.53×).
- **RUN 2/2b** `runs/golden_stream_nsys*` , `runs/golden_stream_nvtx*` — stream under nsys;
  per-req 16.07 (256). NVTX window decomposition via `runs/closeacct_stream_nvtx.py`.
- **RUN 3** `runs/collector_livestep/layerwise.csv` — schedule_to_update: bs1→33.96,
  bs8→109.44 (256). Compute backbone from `runs/partB_realmoe/layerwise.csv` (gpu source).
- **RUN 4** `runs/run4_tp1_occupancy.out` — TP=1 dummy: iso512 96.3 ≈ stream512 94.2 (no drop).
- Golden isolated baseline reproduced: `python runs/closeacct_golden.py runs/serve_nsys_trace.sqlite`.

## FIXES MADE TO SCRIPTS
- `runs/period_golden.py`: added `if __name__=="__main__"` guard (TP=4 spawn was recursing),
  `NT=` env to restrict token sizes, and `torch.cuda.nvtx` range `STREAMPHASE_<nt>` around the
  timed stream generate (ground-truth window for the parser).
- `runs/closeacct_stream_nvtx.py`: new NVTX-bracketed parser recognising all all-reduce variants.
- Known issue logged: golden tp4 + real-weights + **MNBT=512** crashes in the flashinfer
  `trtllm_bf16_moe` autotuner (illegal address on a specific shape). MNBT=768 and TP=1 are fine;
  worked around by using MNBT=768 for the 1-prefill/step comparison.
