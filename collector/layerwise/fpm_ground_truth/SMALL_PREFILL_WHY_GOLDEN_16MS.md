# WHY golden's <=512-tok prefill is ~16ms vs collector ~33-40ms (qwen36, b300, vLLM 0.20.1)

Closes the open question from `SMALL_PREFILL_MECHANISM_DISCRIMINATION.md`. That doc
proved the gap is not topology and called it "capture coverage" but did NOT explain
why golden replays fast and the collector runs slow with the same FULL_AND_PIECEWISE
config. This doc answers it with golden's actual serving nsys trace.

In-image B300 x8, vLLM 0.20.1, `uv run --active`, `HF_HOME=/workspace/models/hf_home`,
nsys 2025.5.2 (`/workspace/software/nsight-2025.5.2`).

## Verdict: OTHER — comm-bound (golden) vs launch-bound (collector). C1/C2/C3 all refuted.

Golden's ~16ms <=512 prefill is **TP-communication-bound**: per-step GPU-busy is
DOMINATED by the tensor-parallel all-reduce kernel `cross_device_reduce_2stage`
(~25 ms of 34 ms nsys-busy; ~75%). The collector models tp4 as a **1-GPU sharded**
config that has **no all-reduce at all** (single GPU) and tiny moe-noop/sharded
kernels, so its forward is **launch-bound**: 3.4 ms GPU-busy + ~30 ms of inter-launch
idle = ~33 ms wall. The launch structure is **identical** (golden 470 vs collector
510 eager launches; 41 piecewise graph pieces each) — so golden does NOT capture more
(C3 refuted). The two numbers differ because they sit in **different bottleneck
regimes**, not because of graph capture, warmup, or the marker.

## Decisive evidence: golden's serving nsys trace (`runs/serve_nsys_trace`)

`vllm serve` (golden's own async engine), tp4 + EP, real weights, real MoE, NO
layerwise marker, isolated warm 256/512-tok prefills. 53 clean piecewise prefill
steps, all identical. Per-step (rank 0), via gap-clustered graph-launch windows:

| quantity | golden real-tp4 (serving) | collector 1-GPU sharded (moe-noop) |
|---|---|---|
| eager launches (`cudaLaunchKernel`/`cuLaunchKernelEx`) | **470** | 510 |
| piecewise graph pieces (`cudaGraphLaunch`) | **41** | 41 |
| GPU busy (nsys) | **34 ms** | **3.4 ms** |
| top kernel by time | **`cross_device_reduce_2stage` (TP all-reduce), ~25 ms / 75%** | GDN/Mamba `chunk_*` (no comm) |
| `cross_device_reduce` present | **yes (~78/step)** | **NO (single GPU)** |
| wall | ~16 ms (busy/comm-bound) | ~33 ms (launch-bound) |

(nsys inflates absolute time ~2x; golden busy 34 ms nsys ≈ 16 ms un-profiled = its
wall, i.e. busy-bound. Launch *counts* are profiling-invariant.)

### Why this inverts to "more comm yet faster"
Golden adds ~78 TP all-reduces per step, yet runs faster (16 vs 33 ms). The all-reduce
kernels occupy the GPU continuously, so the host's ~470 eager launches issue *behind*
them and never starve the GPU — the step is busy-bound at ~16 ms. The collector's
1-GPU model has no comm and tiny moe-noop kernels (3.4 ms), so the GPU goes idle
~60 us between each of ~510 launches → ~30 ms of launch-gap bubbles → a 33 ms
launch-bound floor that is *higher* than golden's real comm-bound step. Removing the
work (moe-noop, single-GPU, sharded /4 dims) paradoxically raises the wall by exposing
the launch tax.

## Candidate mechanisms — explicit dispositions

- **C1 cold-vs-warm: REFUTED.** Warm steps plateau at ~33-40 ms on the collector
  (in-process warm 20-34 ms; 1-GPU async-stream 35.6 ms; golden's own cold->warm is
  only ~22->16 ms). Warming does not bridge 33->16.
- **C2 marker forces eager: REFUTED as the cause of the core gap.** The 33 ms exists in
  plain vLLM with no marker (prior doc #2: 34.7 ms). (The marker + moe-noop likely do
  explain why the *collector's* real-tp4 runs measured ~40 ms instead of golden's 16 ms
  — see Caveat.)
- **C3 golden captures more / fewer eager launches: REFUTED.** Golden 470 eager + 41
  graph ≈ collector 510 + 41. Same piecewise structure; golden is not more-captured.
- **OTHER: CONFIRMED.** Comm-bound real-tp4 (TP all-reduce dominates GPU-busy) vs
  launch-bound 1-GPU-sharded (no comm, tiny kernels, launch-gap idle).

## What did NOT reproduce 16 ms (ruled out, all measured)
Every measurement on the 1-GPU sharded config gives ~34-38 ms: isolated synced
`execute_model_gpu` (33), host-wall no-sync (34), in-process warm (20-34), in-process
stream (34.6), real multiprocess async stream (35.6). Pipelining/continuous-batching
on 1 GPU does not help (no all-reduce to fill the GPU). Real-tp4 async serving gives
ISOLATED 40 ms but STREAM 14 ms — the 14 ms is **batching** (budget-2048 packs ~8
prefills/step), a separate throughput effect, not golden's per-step path (golden steps
are ctx_requests=1, un-batched). Golden's small-ctx warm latency is flat across
tp2/tp4/tp8 (15.6 / 15.8 / 16.5 ms) — not a topology scaling.

## Implied fix
The 1-GPU sharded approximation cannot represent the real-tp4 step because it omits the
TP all-reduce that dominates it and instead measures a launch-bound floor. Options:
1. **Collect the small-prefill regime at real multi-GPU TP with real MoE and WITHOUT the
   layerwise marker** (i.e., golden's serving config) so the step is comm/busy-bound
   like golden (~16 ms). This is the faithful fix; it requires the clean serving path,
   not the marker/moe-noop path.
2. **Calibrate** the <=512-tok launch-bound ctx envelope down to the comm-bound regime
   (cheap, bounded scope) — the 1-GPU launch-bound floor (~33 ms) over-predicts the real
   ~16 ms.
The single-GPU sharded model stays valid for compute-bound regimes (large prefill,
decode); it diverges only in this launch-bound-vs-comm-bound small-prefill regime.

## Caveat / loose end
The collector's earlier *real-tp4* runs measured ~40 ms (not 16) because they carried
the layerwise marker + moe-noop/dummy MoE, which keep the step launch/overhead-bound;
golden's clean serving path (real MoE, no marker) is 16 ms. So "real-tp collection"
(prior doc's option A) CAN reach 16 ms, but only via the clean serving config — the
marker and moe-noop break it. This refines the prior doc's "option A refuted".

## Artifacts
- `runs/serve_nsys_trace.nsys-rep` / `.sqlite` — golden-style tp4 serving trace (the decisive data)
- `runs/serve_nsys.sh`, `runs/serve_probe.py` — vllm-serve-under-nsys orchestration
- `runs/tp4_serving_period.py` — real-tp4 isolated vs stream period (40 vs 14 ms)
- `runs/coldwarm_ctx.py`, `runs/host_vs_gpu_wall.py`, `runs/concurrency_ctx.py`, `runs/mp_throughput_ctx.py` — STEP 1 ruled-out probes
- nsys 2025.5.2 at `/workspace/software/nsight-2025.5.2` (`/usr/local/bin/nsys`)
