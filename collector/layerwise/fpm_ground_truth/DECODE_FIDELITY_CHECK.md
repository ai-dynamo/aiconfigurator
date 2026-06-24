# Does the eager-vs-captured fidelity gap hit DECODE? (qwen36 tp4_ep4)

Follow-up to `SMALL_PREFILL_ENVELOPE_ROOTCAUSE.md`, which found the small-prefill
over-prediction is a 1-GPU-sharded-EAGER vs real-TP-CAPTURED-GRAPH fidelity gap.
Key question: does the SAME gap hit DECODE (the throughput-relevant regime)?

In-image B300, vLLM 0.20.1, `uv run --active`. qwen36 tp4_ep4.

## VERDICT: NO — the fidelity gap does NOT hit the decode FORWARD

The collector's **decode forward latency matches golden** (~1.0x), in stark
contrast to small prefill (~2.1x). The fidelity gap is isolated to the
launch-bound small-prefill regime; the throughput-relevant decode regime is fine.
Real-TP collection is NOT motivated by decode.

(A separate ~1.5x over-prediction exists in the *composed* AIC decode, but it is
the dense-only **MoE-decode overlay/calibration** applied to this hybrid-MoE model
— the known unvalidated term the task explicitly scoped out — NOT the eager
fidelity gap. See "Caveat".)

## #2 — Decode latency: collector ≈ golden (no 2x gap)

Collector decode FORWARD = `execute_model_gpu` non-MoE layerwise (moe-noop) vs
golden decode-phase FPM (full, incl MoE), kv~4096:

| bs | collector non-MoE | golden full | ratio |
|---|---|---|---|
| 1 | 3.562 ms | 3.478 ms | 1.02 |
| 2 | 3.615 ms | 3.551 ms | 1.02 |
| 4 | 3.626 ms | 3.737 ms | 0.97 |
| 8 | 3.678 ms | 3.877 ms | 0.95 |
| (prefill 256-tok, for contrast) | 33 ms | 16 ms | **2.1** |

Decode ratio ~1.0; prefill ratio ~2.1. Collector non-MoE decode ≈ golden FULL
decode because golden's MoE decode is physically tiny at small bs (~0.16ms/step:
8 routed experts x moe_inter 512 x 40 layers ≈ 33MB/layer to read, B300 ~8TB/s),
so golden_full ≈ golden_nonMoE ≈ collector_nonMoE. If decode had the prefill-style
2x eager penalty, collector non-MoE would be ~7ms, not 3.5ms.

## #1 — nsys: decode dispatches to CUDA graphs (one forward, bs=16 past=2048)

`cudaGraphLaunch` is present in decode (graph replay), unlike the small prefill's
pure piecewise-eager attention path:

| forward | cudaGraphLaunch | in-graph kernels | eager kernels | in-graph % |
|---|---|---|---|---|
| decode bs=16 (per step) | ~3 segments | 36 | 155 | ~19% |
| prefill 256-tok (per step) | 41 pieces | 342 | 473 | 42% |

Both run substantial EAGER kernels — this is a **hybrid-model property**: Qwen3.6
is Qwen3-Next style (30 `linear_attention`/GDN + 10 `full_attention` layers, all
MoE), and the GDN/attention ops are `splitting_ops` that run eager between graph
segments in BOTH decode and prefill (and in golden too — same FULL_AND_PIECEWISE).
So graph-fraction alone does not separate decode from prefill.

What separates them is the **latency regime**:
- **Decode is memory-bound** (~3.5ms reading weights/KV per step). The eager-GDN
  launch overhead is a small constant on top, identical in golden and collector,
  and the 1-GPU-sharded engine reads the same per-GPU weights as one real-tp4 GPU
  → collector decode forward reproduces golden's. **No gap.**
- **Small prefill is launch-bound** (~33ms is mostly eager launch bubbles). There
  the 1-GPU-sharded eager path diverges from golden's real-tp4 captured path
  (16ms). **Gap.**

## #3 — Golden decode is served under captured graphs (config-confirmed)
Golden `compilation_config.cudagraph_mode=FULL_AND_PIECEWISE`, `enforce_eager=False`,
`cudagraph_capture_sizes=[1,2,4,8,16,24,...,512]` → decode batch sizes 1-16 are in
the FULL capture set. Golden serves decode via captured graphs (with GDN eager).

## Caveat — composed AIC decode over-predicts ~1.5x (MoE overlay, out of scope)
Composed AIC decode (non-MoE layerwise + MoE overlay + decode calibration) vs
golden full decode, kv~4096:

| bs | composed AIC | golden full | ratio |
|---|---|---|---|
| 1 | 5.47 | 3.48 | 1.57 |
| 2 | 5.60 | 3.55 | 1.58 |
| 4 | 5.81 | 3.74 | 1.56 |
| 8 | 5.72 | 3.88 | 1.47 |

The over-prediction is entirely in the overlay: composed − non-MoE ≈ **1.9ms/step**
of "MoE decode", but physical MoE decode at bs=1 is ~0.16ms (~10x over-estimate).
The non-MoE FORWARD already matches golden (table above), so this is the dense-only
MoE-decode overlay/calibration mis-pricing this hybrid-MoE model — the known
unvalidated hybrid-MoE-decode term, NOT the eager/fidelity gap. Not chased here.

## Conclusion
- Eager-vs-captured **fidelity gap does NOT hit decode** (forward matches golden
  ~1.0x; decode is memory-bound so the eager-GDN overhead is a shared constant).
- **Small-prefill is an isolated, bounded launch-bound curiosity.** Real-TP
  collection for hybrid models is not motivated by the throughput regime.
- Decode accuracy IS limited, but by the separate **MoE-decode overlay**
  (dense-only calibration on a hybrid-MoE model), which over-prices MoE decode
  ~10x — a distinct, already-known follow-up.

## Artifacts
- `runs/nsys_dec16/` (decode bs=16 past=2048 nsys + a1.sqlite): graph/eager counts.
- `runs/dec_compare.py`: composed-AIC-decode vs golden decode ratio.
- collector decode rows: `runs/layerwise_qwen36_tp4ep4_cleanctx4/layerwise.csv`.
