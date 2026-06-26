# Resolution of verdict §(D): is the MoE-expert-GEMM shard term WRONG, or rank-mis-compared?

**Question settled:** the verdict §(D) reported `collector 2.93 / golden 1.61 = 1.82×
→ NOT equivalent`, comparing collector **rank0** (power_law_1.2 synthetic) vs golden
**rank0** (real). MoE all-to-all/all-reduce is a barrier, so MoE step latency tracks
the **heaviest (bottleneck) rank**, not rank0. If golden's heaviest rank ≈ 2.93, the
collector would be correct and 1.82× a rank-misaligned artifact.

**Reuse:** `runs/serve_nsys_trace.sqlite` (golden tp4/ep4 serve, 4 ranks, real
routing, NO --enable-eplb). Script: `runs/partA_moe_perrank.py`. No new collection.

---

## PART 1 — golden per-rank MoE-GEMM busy (`bmm_Bfloat16`, CAPTURED ≤512 steps, n≈53/rank)

| rank (globalPid) | moe_gemm ms/step |
|---|---:|
| r0  284221356638208 | 1.608 |
| r1  284221373415424 | **1.705** ← MAX (bottleneck) |
| r2  284221390192640 | 1.545 ← MIN |
| r3  284221406969856 | 1.575 |

- **MAX (rank1) = 1.705 ms**, MIN = 1.545, MEAN = 1.608
- **MAX/MEAN = 1.060×**, MAX/MIN = 1.103× → golden's REAL per-rank routing skew
- Independent cross-check (Σ all bmm over whole post-warmup window, captured+eager):
  rank1 heaviest, MAX/MEAN = 1.03×. Same conclusion.

**Golden routing is nearly balanced (≈6% bottleneck-over-mean) despite no EPLB.**

## PART 2 — barrier confirmed (step = max rank + comm, not mean)

Per captured step, per rank: step SPAN (AR-bounded) vs local BUSY (Σ all non-AR kernels):

| rank | moe_gemm | busy(all) ms | span ms | slack=span−busy |
|---|---:|---:|---:|---:|
| r0 | 1.608 | 7.140 | 42.638 | 35.498 |
| r1 | 1.705 | 7.110 | 41.715 | 34.606 |
| r2 | 1.545 | 7.040 | 42.688 | 35.648 |
| r3 | 1.575 | 7.107 | 42.593 | 35.486 |

- **span MAX/MIN = 1.023× (CV 0.9%)** → ~equal across ranks ⇒ collectives are barriers;
  every rank's step is gated to the same wall-time regardless of its local load.
- busy differs by rank, slack (≈35 ms) absorbs the difference. Confirms step latency
  is set by **max-rank busy + comm + host residual**, not the mean.
- (Total busy is also nearly equal here only because golden's MoE skew is tiny and
  MoE is ~23% of busy; the barrier holds independent of that.)

## PART 3 — VERDICT

```
collector single-GPU power_law_1.2 moe_gemm  = 2.93 ms
golden rank0 (verdict comparand)             = 1.608 ms
golden BOTTLENECK rank (MAX)                 = 1.705 ms

collector / golden-rank0   = 1.82×   (the verdict's number)
collector / golden-MAX     = 1.72×   ← still ≫ 1
golden-MAX / golden-rank0  = 1.06×   ← the ONLY part attributable to rank mis-compare

1.82× = 1.06× (rank-misalign) × 1.72× (genuine over-count)
```

**The collector is NOT rank-mis-comparing.** `collector/helper.py:1201-1224`
(`_generate_power_law_distribution`) **deliberately finds the max-load EP rank and
swaps it to rank 0**, so collector rank0 = the bottleneck rank *by construction*. The
collector already collects the heaviest rank — the architecture is right. Only **6%**
of the 1.82× is rank misalignment; the rank-mis-compare hypothesis is **REJECTED**.

**The collector genuinely over-counts MoE-GEMM — verdict's conclusion STANDS — but
the verdict's stated MECHANISM is WRONG.** §(D) attributed the 1.8× to "collapsing
ep4 onto one GPU … no all-to-all reduction of the expert workload." That is not what
the collector does: it shards `experts_per_rank = num_experts // ep` (= 16) and measures
the heaviest rank. The real cause is the **synthetic routing distribution**:

- golden REAL per-rank concentration (MAX/MEAN) = **1.06×**
- power_law_1.2 effective per-GPU concentration (2.93 / golden-MEAN 1.608) = **1.82×**
- power_law_1.2 is ~14× more concentrated *in excess over balanced* (82% vs 6%).

**Which alpha matches golden?** None. Replaying the generator's own sampler offline
(64 experts, ep4, top8, 256 tok), per-rank bottleneck/mean is 1.44× at α=1.2 and floors
at ~1.14× as α→0. Golden's real 1.06× is **below the entire power_law family's floor** —
golden routing at this scale is effectively **uniform**, flatter than any power law.
(The 1.44× token-concentration vs the 1.82× GEMM-time over-count also shows part of
the excess is grouped-GEMM time-vs-token nonlinearity on the heavy rank, not just token
count.)

### Recommendation (supersedes §(D) fix wording)
The MoE term is not fixed by "use ep-correct per-GPU busy" — it already is ep-correct
and bottleneck-correct. The fix is the **routing model**: for this Qwen3.6 / no-EPLB
workload the synthetic `power_law_1.2` over-concentrates vs the real near-uniform
routing. Use a **balanced/near-uniform distribution** (i.e. the EPLB-equivalent path,
or a much weaker α) to match golden — or better, drive the per-rank token histogram
from a real routing capture. With balanced routing the collector's bottleneck-rank
MoE-GEMM would land at ≈golden's 1.6–1.7 ms, and the §(D) "1.8× shard inequivalent"
flag would clear.

**Net:** EP *shard* math is FINE; the *routing distribution* is the error. Verdict's
over-count conclusion holds; its "rank-mis-compare" and "ep-collapse" framings do not.
```
uv run --active python runs/partA_moe_perrank.py runs/serve_nsys_trace.sqlite
```
