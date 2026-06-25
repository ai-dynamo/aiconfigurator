# Small-prefill gap mechanism discrimination (qwen36, b300, vLLM 0.20.1)

Follow-up to `SMALL_PREFILL_ENVELOPE_ROOTCAUSE.md`. That doc found the collector's
1-GPU sharded config runs small (<=512-tok) prefills **eager-piecewise at ~33ms**
while golden's real-tp4 deployment is **~16ms**, and attributed the gap to
"1-GPU modeling vs real deployment". This doc DISCRIMINATES the mechanism to gate:
**option B** (build a launch-overhead model `active + launch x f(tp)`, divide by tp)
vs **option A** (fall back to real-tp collection).

In-image B300 x8, vLLM 0.20.1, `uv run --active`, `HF_HOME=/workspace/models/hf_home`.

## Verdict: MECHANISM 1 REFUTED. Real-tp is NOT faster. Build NEITHER A nor B.

The 33->16ms gap is **NOT** a 1-GPU-vs-real-TP artifact. Reproduced faithfully at
real TP2/4/8 (full model, real weights or moe-noop, real NCCL, golden's exact
cudagraph config), the per-GPU small-prefill step stays at **~40-43ms — flat and
slightly RISING with tp — never approaching golden's 16ms.** The gap is a
**capture-coverage** effect (golden replays a captured graph; the collector runs
eager-piecewise) that is **tp-invariant**, so:

- **Option B is WRONG** — there is no `f(tp)` shrink to model; the curve has the
  wrong sign. Dividing launch overhead by tp would predict ~16ms; reality is ~43ms.
- **Option A does NOT recover 16ms either** — real-tp collection reproduces the
  same ~40ms eager-piecewise number. Collecting at real tp buys nothing here.
- The real lever is **capture mode** (force the <=512 prefill onto a FULL captured
  graph, or calibrate the launch-bound <=512 ctx envelope down — doc option 2),
  independent of topology.

## Evidence: per-GPU step latency vs TP (the f(tp) test)

`bs1`, `past_kv=0`, ctx phase, qwen36. **worker_wall** is the only metric capturable
across the multi-process TP engine, so it is used for ALL rows incl. the tp1 baseline
(identical metric => valid comparison). `execute_model_gpu` (tighter cuda-event) is
only capturable in-process (tp1) and shown where available.

| topology | mode | worker_wall 256 | worker_wall 512 | execute_model_gpu 256 |
|---|---|---|---|---|
| tp1 (sharded /4 on 1 GPU) | moe-noop, dummy | **40.6 ms** | 40.5 ms | **33.1 ms** (baseline) |
| tp2 real (phys 2 GPU) | moe-noop, dummy | **41.4 ms** | 41.0 ms | n/a (multiproc) |
| tp4 real (phys 4 GPU) | moe-noop, dummy | **42.6 ms** | 42.6 ms | n/a (multiproc) |
| tp8 real (phys 8 GPU) | moe-noop, dummy | **43.0 ms** | 42.9 ms | n/a (multiproc) |
| tp4 real, **real weights** | real MoE+EP | 44.7 (sched) / 37.5* | 37.3* | n/a |
| **golden real-tp4 (target)** | deployment | — | — | **~16 ms** |

`*` schedule_to_update for the real-weights run (`runs/realtp4_ctx`, 37.5ms) vs
moe-noop run (`runs/realtp4_nsys`, 44.7ms). Both >= the tp1 33ms; neither near 16ms.

**f(tp) has the WRONG SIGN: 40.6 -> 41.4 -> 42.6 -> 43.0 ms as tp goes 1->2->4->8.**
Per-GPU latency does not fall with tp; adding NCCL makes it marginally slower, as
physics requires.

## Deployment parity confirmed (golden's exact graph config, from the engine log)

The real-tp runs used golden's settings, NOT a degraded path:
`enforce_eager=False`, `cudagraph_mode=FULL_AND_PIECEWISE`,
`cudagraph_capture_sizes=[1,2,...,496,512]`, `max_cudagraph_capture_size=512`, and
`splitting_ops` includes `gdn_attention_core`/`mamba_mixer2`/`linear_attention`/
`unified_attention_with_output` (the hybrid ops that run EAGER between piecewise
graph pieces). Capture log: "mixed prefill-decode, PIECEWISE: 51" + "decode, FULL: 51"
=> prefill is served PIECEWISE (eager hybrid ops) at every tp, decode is FULL-graph.
NCCL/comm present (`disable_custom_all_reduce=False`, `flashinfer_comm_preloaded`,
`--enable-expert-parallel`).

## Why flat — the mechanistic explanation

The small-prefill regime is **launch-bound**, not compute-bound. On the tp1 baseline
trace (`runs/nsys_ctx256`): GPU-busy = 3.4 ms but the step span is ~23 ms and the
execute wall is 33 ms — ~20 ms is inter-kernel launch-gap idle. The step issues
~510 eager `cudaLaunchKernel`/`cuLaunchKernelEx` + 41 `cudaGraphLaunch` over 40
hybrid layers. That **launch count is per-rank and tp-invariant**: every rank runs
all 40 layers' worth of eager launches regardless of tp; TP shrinks tensor *sizes*
(compute), not the *number* of launches. Since latency is set by the launch tax, not
compute, it is invariant to tp. Golden's 16 ms is faster solely because it **replays
a captured graph** — one `cudaGraphLaunch` replaces hundreds of eager launches,
deleting the launch tax. That is capture coverage, attainable at ANY tp by capturing
the prefill into a full graph; it is not a topology effect.

## Resolves the "comm-added-but-faster" paradox

The paradox ("real-tp4 ADDS all-reduce/all-gather yet runs FASTER") dissolves once
real-tp is measured directly: **it does NOT run faster.** Faithfully reproduced,
real-tp4 adds NCCL and is marginally SLOWER (43 vs 40 ms), exactly as expected.
Golden's 16 ms was never a TP speedup; it is the captured-graph replay path, which
the collector's standalone single-prefill step does not hit at any tp.

## Direct nsys trace confirmation (real-tp4 per-rank launch structure)

`nsys` was installed (`/workspace/software/nsight-2025.5.2`, symlinked to
`/usr/local/bin/nsys`) and a real-tp4 trace captured with bench_step markers
(`runs/realtp4_trace`, moe-noop, full-depth, 4 ranks). Per-rank attribution suffers
the known nsys multi-process NVTX<->CUPTI timestamp-alignment issue (2 of 4 ranks'
step windows don't align in absolute time, so they fall back), but the **two
cleanly-attributed ranks give identical, profiling-invariant launch counts**:

| metric | 1-GPU sharded (baseline) | real-tp4 per-rank | mechanism implication |
|---|---|---|---|
| eager `cudaLaunchKernel`/`cuLaunchKernelEx` | 510 | **511** (ranks 1 & 2, both runs) | A HOLDS (equal) -> NOT fewer launches |
| `cudaGraphLaunch` pieces | 41 | **41** (identical) | B: NOT more graph capture |
| NCCL/comm kernels | 0 | **present** (parser `dropped_comm`=1214; `disable_custom_all_reduce=False`) | F confirmed |
| GPU-busy (sum kernel dur) | 3.4 ms | noisy (multi-rank align) | n/a |
| span / wall | 22.9 / 33 ms | nsys-inflated ~50 ms (use worker_wall instead) | n/a |

So real-tp4 runs the **exact same eager-piecewise structure** (511 eager launches +
41 piecewise graph pieces over 40 hybrid layers) as the 1-GPU sharded config, **plus
added NCCL**, at the same per-GPU latency. This DIRECTLY REFUTES mechanism 2 (tp4
does not capture more / issue fewer eager launches) and, with the flat f(tp) curve,
mechanism 1. (Absolute span/wall from the nsys run are inflated by profiling
overhead and not comparable to the non-traced baseline; the latency comparison uses
the non-nsys worker_wall sweep above. Launch *counts* are unaffected by profiling.)

Net: at every topology the collector issues ~510 eager launches per rank (launch
tax intact); golden's 16 ms replaces them with captured-graph replays. Capture
coverage, not topology.

## Artifacts
- `runs/realtp4_trace/` — real-tp4 nsys trace (markers), per-rank 511 eager + 41 graph
- `runs/analyze_tp4_trace.py` — per-rank bs1 launch/graph/nccl counter for this trace
- nsys 2025.5.2 installed at `/workspace/software/nsight-2025.5.2`, `/usr/local/bin/nsys`
- `runs/emgpu_tp48/` — real tp4 + tp8 worker_wall sweep (the f(tp) curve)
- `runs/emgpu_phystp/` — real tp2 worker_wall
- `runs/realtp4_ctx/` — real-tp4 REAL weights (schedule_to_update 37.5ms)
- `runs/realtp4_nsys/` — real-tp4 moe-noop (worker_wall ~41ms; nsys silently skipped)
- `runs/layerwise_qwen36_tp4ep4_cleanctx4/` — tp1 sharded baseline (execute_model_gpu 33ms)
- `runs/analyze_launch_gap.py` — per-rank launch/graph/span/wall analyzer (validated on
  the tp1 baseline trace: eager=510, graph=41, span~23ms, busy~3.4ms); ready for any
  future nsys trace.
