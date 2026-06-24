# Small-prefill ctx-envelope gap: root-cause (qwen36 tp4_ep4)

Follow-up to `CLEAN_CTX_ENVELOPE_RESULTS.md`. That work left a GENUINE ~2.4x
over-prediction of the AIC ctx layerwise envelope at small prefills (<=512 new
tok) that clean GPU timing did not remove: ctx-only MoE-incl, 128tok AIC 38.5ms
vs golden FPM 15.9ms; 512tok 41.2 vs 16.2; but >=1024tok matches (1024: 55.7 vs
53.5). This doc finds the root cause.

In-image B300, vLLM 0.20.1, `uv run --active`, `HF_HOME=/workspace/models/hf_home`.

## Verdict: H1 (collection / 1-GPU-modeling), NOT H2 (composition)

The gap is a **collection-mode / 1-GPU-modeling fidelity limitation**, not a
mixed-composition bug. It is reproducible with **plain vLLM** (golden's own
serving code, no layerwise harness) on the collector's 1-GPU sharded config, and
golden's fast small-prefill latency is a **standalone** context step.

## Evidence

### #1 — Golden cudagraph config + the ~528 boundary  (effective_vllm_config.json)
- `compilation_config.cudagraph_mode = FULL_AND_PIECEWISE`, `enforce_eager=False`,
  `cudagraph_capture_sizes=[1,2,...,496,512]`, `max_cudagraph_capture_size=512`.
- Golden context-phase FPM latency vs new tokens shows a SHARP STEP at the
  capture limit: `<=512 tok ~16ms`, then `528 tok -> 40.6ms`, `1056 -> 53.5`.
  ctx_tokens 496 (kv=528) = 16.2ms; 528 (kv=0) = 40.6ms. **The boundary is the
  cudagraph capture max (512): <=512-tok prefills replay captured graphs (fast),
  >512 fall to eager (slow).** (NB: the collector's coincidental block_size=528 is
  unrelated to this 512 boundary — see #5.)

### #4 — Golden's 16ms is STANDALONE context, not decode-fused  → rules out H2
- All 70 golden **context-phase** rows have `decode_tokens=0`. The fast ~16ms at
  128-496 tok is a pure standalone prefill (kv up to 3696), not a small prefill
  fused into a decode full-graph. So H2 (composition / fused-graph pricing) is
  refuted — the fast number exists for a standalone prefill.

### #3 — nsys: the collector's measured forward is eager-PIECEWISE
- Measured 256-tok ctx forward (`--latency-source span`, nsys): inside the step,
  **41 `cudaGraphLaunch`** (piecewise graph pieces) + **~473 eager kernel
  launches** (`cudaLaunchKernel`/`cuLaunchKernelEx`) — the attention/GDN ops
  (`splitting_ops`: `unified_attention`, `gdn_attention_core`, ...) run EAGER
  between graph pieces.
- span (active kernel) = 22.67ms; execute_model_gpu (wall) = 33ms -> ~10ms of
  inter-kernel launch-gap bubbles (eliminated by full-graph capture).

### #2 + key experiments — it's the 1-GPU sharded CONFIG, not the harness
Measured 256-tok ctx execute_model_gpu under several conditions:

| condition | 256-tok ctx | note |
|---|---|---|
| collector, moe-noop (cleanctx4) | 33.1 ms | baseline clean |
| collector, **real MoE** (`--moe-real-router`) | 33.3 ms | not moe-noop |
| **plain vLLM**, no marker/no layer-patch, sharded config | **34.7 ms** | golden's own code |
| collector, kv_heads kept=2 (block_size 272) | 32.8 ms | block_size irrelevant |
| golden real-tp4 deployment | **~16 ms** | target |

- **Plain vLLM serving the 1-GPU sharded config = 34.7ms ≈ collector 33ms.** So
  the slowdown is NOT the layerwise marker/patch/driver — it is the sharded
  single-GPU engine config itself. Golden's *code* on this config would also be ~35ms.
- **Real MoE = moe-noop** (33.3 vs 33.1) -> not the MoE-noop hook.
- The floor is **flat across 128/256/512 tok** (33.6/33.3/33.5) -> launch/overhead
  bound, not compute.

### #5 — block_size is a red herring
- The collector is forced to `block_size=528` ("attention page size >= mamba page
  size" — Qwen3.6 is a hybrid Mamba/GDN model); golden's config records 16.
- But reducing it (kept kv_heads=2 -> block_size **272**) left latency unchanged
  (32.8ms). So block_size (528 vs 272) has **no latency effect** here; it is not
  the cause. (Its recorded 16 in golden is likely a pre-override snapshot; the
  page-alignment override is driven by the GDN state size.)

## Root cause
Golden's real-tp4 deployment serves small (<=512-tok) prefills via **captured
CUDA graphs** (~16ms); at >512 tok it exceeds the capture limit and runs eager
(~40ms). The layerwise collector models tp4 as a **sharded config on 1 GPU**, and
that engine runs the prefill **eager-piecewise** (attention/GDN eager, ~473
kernel launches over 40 hybrid layers) at a flat **~33ms launch-bound floor** for
all small sizes — i.e. it always measures golden's *uncaptured* speed.

This is why the envelope **matches golden at >=1024 tok** (where golden is ALSO
eager/uncaptured and compute begins to dominate launch overhead: 1024 AIC 55.7 vs
golden 53.5) but **over-predicts ~2.4x at <=512 tok** (golden graphed 16ms vs
collector eager 33ms). For this hybrid Mamba/GDN model the eager penalty is large
because each layer issues many small GDN kernels that a captured graph would fuse.

Confirmed NOT: composition/fusion (H2; #4), the layerwise marker/patch (#2 plain
vLLM), moe-noop (#2), or block_size (#5).

## Fix
This is a 1-GPU-modeling-vs-real-deployment fidelity gap, not a harness bug, so
"time it under the full graph" is not a simple collector flag — the per-layer
attribution and the tp1 sharded engine both keep the forward on the eager/piecewise
path that the sharded config produces. Options, in order of fidelity:

1. **Collect hybrid (Mamba/GDN) models at real multi-GPU TP** (deployment parity)
   for the small-prefill regime, so vLLM captures the same CUDA graphs golden uses.
   Heaviest, but the only faithful reproduction of the <=512-tok graphed path.
2. **Calibrate** the small-prefill (<=512-tok, launch-bound) ctx envelope down to
   the captured-graph regime (a per-model correction). Cheap; bounded scope.
3. **Leave as-is and document**: the over-prediction is confined to <=512-tok
   prefills; large-ctx and the decode-starved golden mixed steps already match,
   so the mixed median is only modestly inflated by it. (No code change shipped
   here — a config-patch / TP change needs validation against a real-tp4 run,
   which is out of scope on this 1-GPU node.)

The single-GPU sharded approximation remains valid for **compute-bound** regimes
(large prefill, decode); it is only the **launch/memory-bound small-prefill**
regime of hybrid models where it diverges.

## Artifacts
- nsys trace + sqlite: `runs/nsys_ctx256/` (256-tok forward; graph-vs-eager counts).
- `runs/ctx_realmoe2/` (real MoE = same), `runs/ctx_kv2/` (block_size 272 = same).
- `runs/standalone_ctx.py` + `runs/standalone_ctx.out` (plain vLLM = 34.7ms).
