# Does layerwise composed DECODE still over-predict golden? (qwen36 tp4_ep4)

Settles the ~10x decode-MoE over-prediction after **two** fixes landed:
1. **backbone** — decode layerwise GPU timing (e1cbf718 / 256e86f8).
2. **overlay routing** — `_model_defaults` for qwen36 hardcoded `moe_inter_size=256`;
   true value is **512**. At 256 the MoE query hit the lower-level `vllm_fused_moe`
   (expert-GEMM only); at 512 it hits the **module-level fused** `vllm_fused_moe_module`
   (router/top-k/gather/scatter already inside, `moe_module_level=True`). Fixed here:
   `_model_defaults` now reads dims from `get_model_config_from_model_path`.

In-image B300, vLLM 0.20.1, `uv run --active`. qwen36 = Qwen/Qwen3.6-35B-A3B, tp4_ep4.

## VERDICT: composed STILL over-predicts → decode genuinely DOUBLE-COUNTS (branch 2)

Ilya's prior claim ("overlay is correct; fix the backbone, not the overlay") was
validated only at **tp1/bs1 with inter256**. With **both** fixes in place at tp4_ep4
the overlay is **not** correct on decode: it adds `router`+`dispatch` *on top of* the
module-level fused `routed` term that already contains them. The `is_context` guard at
`vllm_backend.py:1076` suppressed this double-count only on **prefill**, not decode.

**Fix applied:** drop the `is_context and` from that guard. It is now kernel-source-aware
(fires whenever `moe_module_level=True`, both phases); lower-level/main MoE data is
untouched. Decode MAPE vs golden **51.3% → 21.2%**, ratio **~1.5x → ~1.1–1.27x**.

## STEP 0 — inter fix confirmed
- `model._moe_inter_size = 512` (was 256).
- routed query metadata `{'moe_module_level': True}` → hits the fused module data.
- routed per-layer ×40 = **0.157 ms** ≈ golden's physical MoE-decode (~0.16 ms/step).

## STEP 1 — composed DECODE vs golden, tp4_ep4, past_kv≈4096
Backbone = local clean gen collection (`runs/layerwise_qwen36_tp4ep4_cleanctx4/`
`layerwise_native_tagged128.csv`, `latency_source=execute_model_gpu`, single source,
monotonic). The committed systems `layerwise_perf.csv` decode rows are the OLD `span`
representative-module timing the validator rejects, so the clean backbone is the local
run. Overlay (module-level fused MoE + comm) = committed systems root.
Golden = `fpm_upfront_qwen36_moe_full_once_20260613_201336/tp4_ep4_past4096/`
`fpm_metrics_phase.csv` (decode rows).

### Before the guard fix (router+dispatch added on decode)
| bs | backbone | overlay | composed | golden | ratio | routed ×40 |
|----|---------:|--------:|---------:|-------:|------:|-----------:|
| 1  | 3.768 | 1.662 | 5.431 | 3.478 | 1.56 | 0.157 |
| 2  | 3.824 | 1.741 | 5.565 | 3.551 | 1.57 | 0.212 |
| 4  | 3.839 | 2.048 | 5.887 | 3.737 | 1.58 | 0.417 |
| 8  | 3.941 | 1.818 | 5.759 | 3.877 | 1.49 | 0.640 |
| 16 | 4.143 | 2.095 | 6.238 | 4.530 | 1.38 | 0.903 |

**Decode MAPE = 51.3%.** All ms are full-model (×40 layers). bs=16 golden uses the
nearest kv window (its golden grid has no point in 3800–4300).

### After the guard fix (router+dispatch deleted when moe_module_level=True)
| bs | backbone | overlay | composed | golden | ratio | routed ×40 |
|----|---------:|--------:|---------:|-------:|------:|-----------:|
| 1  | 3.768 | 0.551 | 4.320 | 3.478 | 1.24 | 0.157 |
| 2  | 3.824 | 0.612 | 4.436 | 3.551 | 1.25 | 0.212 |
| 4  | 3.839 | 0.893 | 4.732 | 3.737 | 1.27 | 0.417 |
| 8  | 3.941 | 0.650 | 4.591 | 3.877 | 1.18 | 0.640 |
| 16 | 4.143 | 0.916 | 5.059 | 4.530 | 1.12 | 0.903 |

**Decode MAPE = 21.2%.**

## Why: the double-count quantified (overlay components, ms/step ×40)
| bs | routed | router | dispatch | shared | ep_a2a | overlap | r+d (double-count) |
|----|-------:|-------:|---------:|-------:|-------:|--------:|-------------------:|
| 1  | 0.157 | 0.329 | 0.782 | 0.491 | 0.394 | −0.491 | **1.111** |
| 2  | 0.212 | 0.329 | 0.800 | 0.518 | 0.400 | −0.518 | **1.129** |
| 4  | 0.417 | 0.341 | 0.814 | 0.573 | 0.476 | −0.573 | **1.155** |
| 8  | 0.640 | 0.356 | 0.812 | 0.542 | 0.011 | −0.542 | **1.168** |
| 16 | 0.903 | 0.357 | 0.823 | 0.532 | 0.012 | −0.532 | **1.180** |

- **`routed` alone ≈ golden physical MoE** (0.157 ms ≈ 0.16 ms at bs=1). The fused
  module-level term is correct on its own — confirming branch 2's precondition.
- **`backbone` alone already ≈ golden FULL** (3.768 vs 3.478, 1.08x). Decode is
  memory-bound; physical MoE is ~0.16 ms, so golden_full ≈ golden_nonMoE ≈ backbone.
- **`router`+`dispatch` ≈ 1.1 ms/step of pure double-count** — they are *inside* the
  fused `routed` already. This is the dominant excess and exactly what the guard now
  deletes on decode.
- `shared` (~0.5) is cancelled by the overlap adjustment (−0.5) → nets to ~0, so it
  does not inflate. `ep_a2a` is a genuine cross-GPU NCCL term (not in the per-GPU fused
  module) and is small at the bs that matter.

## Residual (~1.1–1.27x after fix)
Not the MoE double-count anymore. It is the backbone running ~1.08x over golden full
(small constant) plus the genuine small MoE terms. This is in the noise band of the
hybrid-MoE decode model and is **not** the 10x / 1.5x overlay defect this task chased.

## Change landed
- `vllm_backend.py:1076` — dropped `is_context and`; guard is now kernel-source-aware.
- `compare_aic_layerwise_fpm.py:_model_defaults` — qwen36 reads dims from model config
  (moe_inter 256 → 512).
- New regression test `test_noop_moe_addback_skips_router_dispatch_for_module_level_moe_on_decode`.

## Artifacts
- `runs/dec_inter512_check.py` — the decomposition harness producing the tables above.
- 4 pre-existing unrelated test failures on this branch (mixed-step / outlier-smoothing /
  max_num_seqs) are unchanged by this fix.
