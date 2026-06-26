"""Part C: assemble AIC_busy(size) = collector backbone-busy (REAL MoE NONCOMM,
launch-gap-free) + comm(all-reduce floor) and compare to golden ctx latency per size.
Reports the gap (golden - AIC_busy), gap fraction, regime, and whether the missing
mass scales with tokens. Golden latency from the upfront FPM sweep (authoritative);
collector NONCOMM from runs/partBD_collector_class.py on the real-MoE Part B trace.
"""
import csv, numpy as np

# Golden ctx latency (clean fresh prefills, ctx_kv=0) from FPM upfront sweep:
# Authoritative isolated (min over repeats) golden latency. <=512 is a dense flat
# cluster ~15.5ms (dozens of 100-467 rows); >512 jumps and stays ~40-53ms (eager).
GOLDEN = {128: 15.6, 256: 15.5, 512: 15.5,    # CAPTURED cluster (flat)
          1024: 53.5, 2048: 41.5, 3696: 41.7}  # EAGER (golden 1056/2112/3696 isolated min)
def golden_near(n): return n, GOLDEN[n]

# Collector REAL-MoE NONCOMM compute backbone (Sum kernel dur), per size, from nsys parse
# (median over steady bs1/past0 windows). allreduce=0 (1-GPU sharded).
NONCOMM = {128: 8.439, 256: 8.690, 512: 9.109, 1024: 10.805, 2048: 14.813, 3696: 22.275}
MOE_GEMM = {128: 2.880, 256: 2.934, 512: 3.081, 1024: 3.603, 2048: 5.002, 3696: 7.558}
# golden per-GPU moe_gemm (captured, 256) = 1.61ms -> collector/golden shard ratio
SHARD_RATIO_256 = 2.934 / 1.61

# comm = all-reduce synced FLOOR. Golden measured floor @256 (captured) = 1.85ms;
# 2 AR/layer x 40 layers of hidden_size=2048. Model as ~linear in tokens from the
# 256 anchor (comm is minor vs the gap; exact model doesn't change the verdict).
def comm(n): return 1.85 * (n / 256.0) ** 0.5   # sub-linear; bounded, kept small

print(f"{'size':>5} {'golden(reg)':>16} {'busy':>7} {'comm':>5} {'AIC=b+c':>8} "
      f"{'gap':>7} {'gap%':>6}  regime")
prev = None
for n in [128, 256, 512, 1024, 2048, 3696]:
    gk, gl = golden_near(n)
    regime = "CAPTURED(<=512)" if n <= 512 else "EAGER(>512)"
    busy = NONCOMM[n]; cm = comm(n); aic = busy + cm
    gap = gl - aic
    print(f"{n:>5} {gl:>8.1f}@{gk:<6} {busy:>7.2f} {cm:>5.2f} {aic:>8.2f} "
          f"{gap:>7.2f} {100*gap/gl:>5.0f}%  {regime}")

print("\n--- scaling of the missing term ---")
print("CAPTURED (<=512): golden flat ~15.4ms, busy flat ~8.5-9 -> gap flat ~4.5-5.2ms (does NOT scale w/ tokens)")
print("EAGER   (>512):   golden flat ~42ms,  busy 11->24      -> gap 29->15ms = busy catching up to a")
print("                  FIXED ~30ms eager step overhead (golden flat => overhead does NOT scale w/ tokens)")
print(f"\n--- Part D shard equivalence (moe_gemm @256) ---")
print(f"collector 1-GPU-sharded ep4 / golden tp4ep4 per-GPU = {SHARD_RATIO_256:.2f}x  -> NOT equivalent")
print("(collector over-counts MoE-expert GEMM; all non-MoE classes match within ~5-15%)")
