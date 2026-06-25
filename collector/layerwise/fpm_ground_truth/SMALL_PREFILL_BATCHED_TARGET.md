# Small-prefill fix on the BATCHED target — is it warranted? (qwen36 tp4_ep4, b300, vLLM 0.20.1)

Re-establishes ctx + mixed FPM-vs-AIC MAPE **after** the decode fixes landed
(inter=512 + kernel-source-aware router/dispatch deletion, `vllm_backend.py:1083`).
The small-prefill over-prediction was studied on **isolated** single prefills; the
downstream API contract is **batched serving** (golden FPM mixed rows = real batched
steps). This doc measures where we actually stand on that batched target.

Run (current HEAD):
```
compare_aic_layerwise_fpm_summary.py --case qwen36_tp4_ep4 \
  --layerwise runs/layerwise_qwen36_tp4ep4_cleanctx4/layerwise.csv \
  --moe-perf-file collector/layerwise/wip/moe_perf.txt
```
Fixes confirmed: `moe_inter_size`→512 (resolver, no hardcode); `:1083` guard keyed on
`moe_module_level`, fires both phases.

## Overall MAPE

| phase | n | MAPE | median AIC/FPM |
| --- | --- | --- | --- |
| ctx (isolated) | 3 | **189.1%** | 3.14 |
| gen (decode) | 1 | 19.3% | 1.19 |
| **mixed (batched target)** | **71** | **38.8%** | **1.23** |

## Mixed broken down

By new ctx_tokens:
| bucket | n (share) | MAPE | ratio |
| --- | --- | --- | --- |
| **≤512** | 7 (10%) | **171.0%** | **2.97** |
| 512–2048 | 5 (7%) | 32.0% | 0.93 |
| >2048 | 59 (83%) | 23.7% | 1.23 |

By ctx_requests:
| | n (share) | MAPE | ratio |
| --- | --- | --- | --- |
| single prefill (ctx_req=1) | 12 (17%) | 111.2% | 2.04 |
| packed (ctx_req>1) | 59 (83%) | 24.1% | 1.22 |

Small-ctx single-prefill steps (`ctx_tokens≤512 AND ctx_req=1`) = **7/71 = 10%** of mixed
steps, MAPE 171%, ratio ~3×, and **43% of the total mixed error mass**. 6 of the 7 are the
same shape — **400 new + 3696 cached prefix** (terminal chunk of a chunked 4096 prefill)
co-scheduled with decodes (FPM ~17ms, AIC ~52ms); 1 is a near-fresh 496-tok prefill.

## Verdict — WARRANTED, modest + shape-narrow headroom

- Mixed **38.8%** is above the "decent <30%" line, and the excess is the small-ctx bucket:
  10% of steps, 43% of error mass, ~3× over-prediction (same comm-bound-vs-launch-bound
  floor as isolated — AIC's 1-GPU launch-bound floor over-prices a small-new-token step
  that golden runs fast on a busy GPU).
- **Headroom:** a perfect small-prefill fix → mixed **38.8% → ~21.9%** (~17pp). Excluding
  the bucket, the rest is **24.3%** (compute-bound, already decent).
- **"Batching washes it out" is only partly right:** most prefills pack and become
  compute-bound (~24%), but the small regime persists as the **small-new-token terminal
  chunk of a chunked long prefill** (the 2048 budget can't fill it, so it runs alone,
  ctx_req=1) and keeps the ~3× over-prediction.

**Implement the busy-vs-wall (comm-bound floor) backbone fix** — single largest residual on
the batched target (~17pp, 43% of error mass), pushes mixed into the ~22% band. **Caveat:**
benefiting population is small and concentrated in ~1 shape class (terminal long-prefill
chunks); validate the headroom on a broader ISL/chunk mix before over-investing.

Background: `SMALL_PREFILL_WHY_GOLDEN_16MS` (comm-bound golden), `runs/analyze_prefill_floor.py`
(captured-vs-eager / synced-floor reconciliation).
