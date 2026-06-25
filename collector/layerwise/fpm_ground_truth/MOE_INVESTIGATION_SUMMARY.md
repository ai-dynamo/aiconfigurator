# MoE Layerwise Modeling — Investigation Summary

**Goal:** improve AIC's MoE perf model (started from Frontier's skew-feature idea); the
downstream forward-step-latency API needs this accurate.

## Navigation — the two-layer frame (read this first)

Every forward step decomposes as two independent layers:

```
step latency = backbone (non-MoE, per-layer wall)  +  MoE overlay (routed/router/dispatch/comm/shared)
```

We investigated 4 directions; each hits a **different component**. Knowing which layer a
finding lands in is the whole key to not getting lost:

| # | direction | component | status |
| --- | --- | --- | --- |
| 1 | expert imbalance / router dominates | MoE overlay | ❌ rejected — skew self-averages (per-step CV ~1.8%); AIC's power-law label is enough |
| 2 | layerwise data not apple-to-apple vs FPM | measurement | ❌ small portion — ctx timed with `worker_wall` (host overhead); fixed → `execute_model_gpu` |
| 3 | **MoE overlay modeling inaccurate** | **MoE overlay** | ✅ **FIXED** — decode double-count: the fused `routed` kernel already contains router/dispatch, AIC added them again; MAPE 51→21% |
| 4 | **small-prefill gap (latest)** | **backbone (non-MoE)** | 🆕 **NEW, MoE-unrelated** — comm-bound (golden) vs launch-bound (1-GPU collection); not yet fixed |

**The #3 vs #4 boundary (the thing that's easy to confuse):**
- **#3 is the MoE overlay.** Fixed by using the measured fused per-layer latency and
  deleting the router/dispatch AIC double-added (fusion). Done.
- **#4 is the backbone.** MoE-unrelated: **moe-noop (MoE removed) still shows the 33ms-vs-16ms
  gap** → it is not the overlay. It is the non-MoE per-layer wall, mis-measured because the
  collector models tp4 as a **1-GPU sharded** config that has **no TP all-reduce**.

Both share the roofline insight `total = max(compute, memory, comm)`, but on **different
components / regimes**:
- #3 (overlay, decode): comm is **hidden** under the memory-bound weight load → overlap ≈ 0
  (that's why we deleted the double-counted terms).
- #4 (backbone, small-prefill): compute is tiny so **comm dominates** → golden is comm-bound
  ~16ms; the 1-GPU collection has no comm and mis-measures a launch-bound ~33ms floor.

Per-topic docs: `DECODE_MOE_OVERLAY_VERDICT` (#3); `SMALL_PREFILL_WHY_GOLDEN_16MS` +
`SMALL_PREFILL_MECHANISM_DISCRIMINATION` + `SMALL_PREFILL_ENVELOPE_ROOTCAUSE` (#4);
`MIX_STEP_FIX_VALIDATION` + `CLEAN_CTX_ENVELOPE_RESULTS` (#2); `DECODE_FIDELITY_CHECK` (backbone-OK baseline).

## The real root cause: wrong moe_inter in the diagnostics tool

The FPM-vs-AIC diagnostics (`compare_aic_layerwise_fpm.py:_model_defaults`)
**hardcoded** qwen36 `moe_inter_size=256`. The true value is **512**
(`get_model_config_from_model_path` returns 512; all tests use 512).

That one wrong number cascaded:

1. The MoE query used inter=256, which **hits the lower-level `vllm_fused_moe`
   data** (expert-GEMM only, `module_level=0`) instead of the real inter=512
   **`vllm_fused_moe_module` fused data** (`module_level=1`).
2. Lower-level data contains no router/dispatch, so the overlay code legitimately
   adds router + pack/combine + shared + EP comm back on top — but on the wrong
   (half-inter) GEMM, and as separate serial terms.
3. Result: composed decode MoE ≈ 1.8 ms/step vs golden ≈ 0.16 ms (~10×).

**Decisive test:** the module-level fused `routed` op at inter=512 alone is
**0.157 ms ≈ golden 0.16 ms** — because the fused kernel already includes
router/top-k/gather/scatter. So the canonical path is: use the module-level fused
measurement and do NOT re-add the pieces it already contains.

This is *not* the main-line op-wise model: main SDK reads `moe_inter` from the HF
config resolver (→ 512), so it hits the module-level data. The bug was confined to
the diagnostics validation harness, which is exactly why our validation looked
broken for weeks.

## Fix landed + current state (recomputed on the true inter=512)

Fixed `_model_defaults` to route qwen36 through `get_model_config_from_model_path`
(no more hardcode). On inter=512:

| phase | moe (routed) | router | dispatch | shared | deletion fires? |
| --- | --- | --- | --- | --- | --- |
| **context** | 2.189 | **0** | **0** | 0.587 | ✓ (`is_context and module_level`) |
| **decode** | 0.157 ≈ golden | 0.329 | 0.782 | 0.491 | ✗ (`is_context` guard blocks it) |

- **Context: fixed.** Hits module-level data; the existing deletion logic
  (`_layerwise_noop_moe_addback:1083`) zeroes the double-counted router/dispatch.
- **Decode: fixed** (guard `is_context` dropped — see resolved section below).
  `routed` is correct (0.157 ≈ golden) and the same deletion now fires on decode too.

## Decode residual — RESOLVED (GPU-validated, see DECODE_MOE_OVERLAY_VERDICT.md)

GPU re-ran composed decode vs golden at tp4_ep4 with BOTH fixes (inter=512 + fixed
backbone). Verdict: **decode genuinely double-counts** — Ilya's "overlay is correct"
held only at tp1/bs1/inter256. With the corrected inter512, the module-level fused
`routed` term already contains router/dispatch, so the overlay was adding ~1.1 ms/step
of double-count on decode. The `is_context` guard suppressed the delete only on prefill.

**Fix landed (vllm_backend.py:1083):** dropped `is_context and` → the deletion is now
kernel-source-aware (fires whenever `moe_module_level=True`, both phases); lower-level /
main MoE data (no flag) is untouched. **Decode MAPE 51.3% → 21.2%, ratio ~1.5× → 1.1–1.27×.**

Decomposition (GPU, ms/step ×40):

| term | value | disposition |
| --- | --- | --- |
| routed (fused module) | 0.157 ≈ golden physical MoE | keep — correct on its own |
| router + dispatch | ~1.11 (the double-count) | **deleted** (inside fused routed) |
| shared (~0.5) | cancelled by overlap adj (−0.5) → ~0 | nets out |
| ep_a2a | small | kept — genuine cross-GPU NCCL, not in per-GPU fused module |

Residual ~1.1–1.27× is the **backbone running ~1.08× over golden full** (memory-bound;
golden_full ≈ golden_nonMoE ≈ backbone) plus small genuine MoE terms — in the noise band,
NOT the overlay defect.

**One open nuance (multi-node follow-up):** the fix deletes router+dispatch wholesale.
For single-node this is right (dispatch's comm ≈0/overlapped, validated). But `dispatch`
bundles an attention-TP allreduce whose deletion is justified by "comm≈0 single-node",
NOT by "the module kernel includes it" — a distinct claim that would break multi-node.
The genuine cross-GPU EP term (`ep_a2a`) is kept separately, so multi-node EP is covered;
the deleted dispatch allreduce is the only multi-node risk. Re-check when exposed-comm
(multi-node) topologies are modeled.

## Lessons

- **Verify the collected data AND the harness config before blaming the model.**
  3 of 4 "model bugs" were collection artifacts; the 4th was a hardcoded config
  value in the validation tool itself.
- **A hardcoded special-case is a bug class.** The wrong moe_inter=256 silently
  pointed the query at the wrong dataset. Route through the resolver.
- **A measured fused op already contains fusion + intra-kernel overlap.** Don't
  re-add the pieces it includes — but only once you've confirmed you're hitting
  the fused (module-level) data, not the lower-level expert-GEMM data.
- **Verify, don't flip.** Every reversal in this investigation was a real finding
  that looked like it overturned everything; the overturn always waited on one
  decisive measurement.
