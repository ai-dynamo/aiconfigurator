# Busy-metric reconstruction of golden ctx latency — VERDICT

**Question:** across ALL prefill sizes, can AIC's busy-metric reconstruction
(backbone GPU-busy + MoE-overlay + comm) reproduce golden ctx latency, thereby
fixing BOTH the small-prefill (≤512) over-prediction and the >512 mixed-residual
with one change (promote `latency_source=gpu` + add comm)?

**Model/system:** Qwen3.6-35B-A3B (40 layers = 30 GDN/linear-attn + 10 flash-attn;
every layer MoE-64/top-8 + shared expert), tp4/ep4, B300, vLLM 0.20.1.
Method = ATTRIBUTE (no fitting).

## VERDICT: NOT VIABLE as a pure busy reconstruction.

Σ(kernel-duration) + comm **under-predicts golden at every size**, by a
**regime-dependent, token-independent host/launch/sync overhead** that is *not a
kernel-duration term* and *not any kernel class*. It is ~5 ms in the CAPTURED
(≤512) regime and ~20–35 ms in the EAGER (>512) regime, jumping at the 512
cudagraph-capture boundary. Separately, the collector's 1-GPU-sharded MoE-expert
GEMM is **1.8× golden's per-GPU** value, so the measured backbone isn't even
golden's per-GPU compute. Promoting `latency_source=gpu`+comm therefore trades
the current WALL over-prediction for a comparable under-prediction; it does not
fix either open item. **A viable model needs busy + a regime-dependent host-residual
calibration (capture vs eager) + a shard-correct MoE term** → fall back to
calibration (or serving full-step collection).

The unifying root for BOTH open items: golden serving latency sits *between* the
collector's isolated WALL and its GPU-busy:

```
   busy+comm   <   GOLDEN   <   isolated WALL
   (no host)       (partial)     (full host idle exposed)
   ~10 ms          15.5 ms        ~45 ms     (256 tok example)
```

WALL over-counts the full ~30 ms host/launch idle that async serving overlaps;
BUSY counts none of the residual that serving does *not* overlap. Golden =
busy + comm + partially-overlapped-host-residual(regime).

---

## (A) Golden kernel-class attribution  (`runs/partA_kernel_class.py` on `serve_nsys_trace.sqlite`)

GPU-busy bucketed per class; the all-reduce kernel is split into its synced FLOOR
(real comm = p25×count) and SPIN-WAIT (idle artifact of the isolated probe — the
`cross_device_reduce_2stage` kernel sits on the GPU busy-waiting for cross-rank
sync; its duration tracks launch mode, **not** data volume).

| class | CAPTURED ≤512 (~256 tok) | EAGER >512 (~512 tok) |
|---|---:|---:|
| attn/GDN/Mamba | 1.63 ms (18%) | 1.86 ms |
| **MoE-expert GEMM** (`bmm_Bfloat16`) | 1.61 ms (18%) | 2.13 ms |
| shared-expert | 0.17 ms (2%) | 0.21 ms |
| router (`routing*`) | 0.47 ms (5%) | 0.59 ms |
| dense-proj GEMM (`nvjet`,splitK) | 1.68 ms (19%) | 1.64 ms |
| elementwise/rmsnorm/other | 1.58 ms (18%) | 1.63 ms |
| **all-reduce FLOOR (real comm)** | 1.85 ms (20%) | 60.2 ms* |
| **REAL WORK = busy+comm (prediction)** | **8.98 ms** | (see note) |
| all-reduce SPIN (NOT counted) | 25.4 ms | 23.5 ms |
| **GOLDEN FPM latency** | **15.5 ms** | **39.9 ms** |
| **→ MISSING (golden − real-work)** | **~6.5 ms** | **~30 ms** |

\* In EAGER mode the AR-floor method breaks down: the all-reduce is launch-serialized
and even its p25 (743 µs × 81) is sync-wait, not transfer. The real comm transfer
is latency-bound ~2–3 ms (80 small AR × ~23 µs); the rest is host/launch/sync stall.
Compute backbone barely changes 256→512 (7.14→8.06 ms): these sizes are
overhead-bound, not FLOP-bound.

**Named missing mass:** it is **none of the kernel classes** — all six compute
classes are fully measured and sum to the GPU-busy. The missing term is the
per-step **host-dispatch / kernel-launch / cross-rank-sync idle** that golden
serving genuinely pays (and that Σ-kernel-duration is blind to by construction).

## (B) Collector GPU-busy across sizes  (`runs/partBD_collector_class.py` on the REAL-MoE Part B trace)

1-GPU sharded tp4/ep4, REAL MoE (`--moe-real-router`, full 40-layer depth),
`latency_source=gpu`. all-reduce = 0 (single GPU). NONCOMM = pure compute backbone.

| ntok | attn_gdn | moe_gemm | shared | router | dense | ew/other | **NONCOMM** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1.58 | 2.88 | 0.17 | 0.28 | 1.86 | 1.67 | **8.44** |
| 256 | 1.67 | 2.93 | 0.18 | 0.49 | 1.94 | 1.47 | **8.69** |
| 512 | 1.90 | 3.08 | 0.22 | 0.59 | 1.83 | 1.48 | **9.11** |
| 1024 | 2.41 | 3.60 | 0.33 | 0.76 | 1.99 | 1.72 | **10.80** |
| 2048 | 3.61 | 5.00 | 0.53 | 1.06 | 2.49 | 2.12 | **14.81** |
| 3696 | 5.97 | 7.56 | 0.84 | 1.59 | 3.52 | 2.79 | **22.27** |

(The collector's CSV `gpu` rollup — CUDAGraphWrapper-attributed only — is ~3 ms
lower, e.g. ~5.6 ms @256; using it makes the gap below *larger*, not smaller.)

## (C) Assemble vs golden  (`runs/partC_assemble.py`)

AIC_busy = collector NONCOMM + comm(all-reduce floor, latency-bound ~2–7 ms).

| size | golden | busy | comm | AIC=b+c | gap | gap% | regime | unaccounted class |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 128 | 15.6 | 8.44 | 1.3 | 9.8 | **+5.9** | 38% | CAPTURED | none (host/launch idle) |
| 256 | 15.5 | 8.69 | 1.9 | 10.5 | **+5.0** | 32% | CAPTURED | none (host/launch idle) |
| 512 | 15.5 | 9.11 | 2.6 | 11.7 | **+3.8** | 24% | CAPTURED | none (host/launch idle) |
| 1024 | 53.5 | 10.80 | 3.7 | 14.5 | **+39** | 73% | EAGER | none (eager launch/AR-sync) |
| 2048 | 41.5 | 14.81 | 5.2 | 20.1 | **+21** | 52% | EAGER | none (eager launch/AR-sync) |
| 3696 | 41.7 | 22.27 | 7.0 | 29.3 | **+12** | 30% | EAGER | none (eager launch/AR-sync) |

**Does the missing mass scale with size?** NO — it is a fixed per-step overhead
that flips at the capture→eager boundary:
- CAPTURED ≤512: golden flat ~15.5 ms, busy flat ~8.5–9 ms → gap **flat ~4–6 ms**.
- EAGER >512: golden flat ~40–53 ms while busy grows 11→24 ms → the gap *shrinks*
  only because busy catches up to a **fixed ~30 ms eager step overhead**; golden's
  flatness proves the overhead itself does not scale with tokens (until ~7000+ tok
  where compute finally overtakes and golden rises to ~50 ms).

A token-scaled missing kernel would show a gap growing with size — the opposite is
observed. The missing term is a regime constant, i.e. exactly what a Σ-kernel
metric cannot express without an additive calibration term.

## (D) Shard-equivalence — `moe_gemm` @256

`collector 1-GPU-sharded ep4` / `golden tp4ep4 per-GPU` = **2.93 / 1.61 = 1.82× → NOT equivalent.**

All non-MoE classes match within ~5–15% (attn 1.67 vs 1.63, dense 1.94 vs 1.68,
router/shared equal) — TP-replicated work transfers cleanly. But the MoE-expert
GEMM does not: collapsing ep4 onto one GPU routes ~1.8× the per-expert token load
of a real ep4 rank (no all-to-all reduction of the expert workload). The
collector backbone therefore **over-counts MoE compute**, and the gap above is a
*lower bound* — a shard-corrected (smaller) backbone widens it.

---

## Implication for the two open items

- **≤512 over-prediction:** caused by AIC using isolated WALL (~45 ms) which exposes
  host idle that serving overlaps. Busy+comm (~10.5 ms) under-shoots golden 15.5 ms
  by ~5 ms — swaps over- for under-prediction. Not fixed by busy alone.
- **>512 residual:** golden is dominated by a fixed ~30 ms eager launch/AR-sync
  overhead, not by token-scaled compute. Busy+comm cannot represent it; it
  under-predicts 30–73%.

**Recommendation:** do NOT promote `latency_source=gpu` + comm as the backbone.
Instead model golden = GPU-busy + comm + a **regime-dependent host-residual** (one
constant for cudagraph-replay ≤512, one for eager >512), fitted/calibrated against
golden FPM, AND correct the MoE-expert-GEMM shard term (use ep-correct per-GPU MoE
busy, not 1-GPU-collapsed). Equivalently, collect from the serving full-step
(which already contains the residual) rather than the isolated launch-gap-free step.

### Reproduce
```
uv run --active python runs/partA_kernel_class.py            # (A) golden per-class
bash runs/partB_realmoe.sh                                   # (B) collector real-MoE busy sweep
uv run --active python runs/partBD_collector_class.py runs/partB_realmoe/profiles/nsys/*.sqlite  # (B/D) per-class
uv run --active python runs/partC_assemble.py                # (C) assemble vs golden
```

### Caveat (transparency)
A fresh >512 golden nsys capture at 3696 (a full tp4/ep4 serve under nsys, ~15 min)
was **not** run; Part A's >512 row uses the EAGER ~512-tok steps already present in
`serve_nsys_trace.sqlite`. This is sufficient because (i) golden FPM gives the
authoritative latency at every size and (ii) the collector real-MoE trace gives the
per-class compute growth through 3696 — together they show the >512 gap is
non-kernel and non-scaling. A 3696 golden capture would only add a third confirming
point to an already-monotone trend.
