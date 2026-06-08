# Engine-Step Latency Benchmark Snapshots

Reproducible Python SDK vs Rust core hot-path comparison. Updated as Phase 3
progresses; the baseline below anchors the >=3x speedup exit criterion.

## How to reproduce

```bash
AICONFIGURATOR_RUST_CORE_AUTOBUILD=1 \
  uv run python rust/aiconfigurator-core/parity_tests/benchmark_engine_step.py \
  --warmup 5 --iterations 30 --json
```

Each case clears Python database/op/model caches and Rust estimator/library
caches before timing, then runs 5 warmup iterations to repopulate runtime
caches before 30 timed samples per phase. Setup costs are reported separately
and excluded from per-call latency. See `README.md` for `--cache-mode` and
case-specific overrides.

## Baseline (Phase 3 not started)

Captured 2026-05-28 on the pre-Phase 3 branch (commits 04eb3d47 and
14e04574 atop `main`). The current Rust core still uses aggregate-formula
shortcuts in `src/lib.rs`/`src/perf.rs` and is not a faithful port of the
Python op graph; numbers here are the starting point that Phase 3
implementation should improve on (for the slow models) and preserve or
extend (for the fast ones).

Host: Mac15,7 (Apple Silicon, Darwin 25.5.0 / macOS 26.5, arm64).
Toolchain: rustc 1.95.0, Python 3.12.12. CPU-only benchmark; no GPU
participation - both Python and Rust execute the latency model in-process.

Smoke parallelism: `tp=8, pp=1, attention_dp=1, moe_tp=1, moe_ep=8`.

### MiniMaxAI/MiniMax-M2.5 (hybrid MoE) — b200_sxm/vllm/0.19.0, isl=1024 osl=2

| Phase | Python p50/p90/p99 (us) | Rust p50/p90/p99 (us) | Rust speedup (p50) |
| --- | --- | --- | --- |
| context | 23.979 / 24.550 / 26.289 | 15.354 / 18.908 / 20.523 | 1.56x |
| generation | 22.855 / 23.553 / 25.564 | 14.604 / 15.049 / 15.184 | 1.57x |

Setup cost: Python session 33.018 ms, Rust estimator 135.339 ms.

### moonshotai/Kimi-K2.5 (DeepSeek family, MLA) — b200_sxm/vllm/0.19.0, isl=1024 osl=2

| Phase | Python p50/p90/p99 (us) | Rust p50/p90/p99 (us) | Rust speedup (p50) |
| --- | --- | --- | --- |
| context | 25.750 / 26.379 / 27.329 | 121.500 / 122.130 / 126.742 | 0.21x (5x slower) |
| generation | 31.021 / 32.175 / 32.280 | 44.979 / 45.675 / 45.910 | 0.69x (1.5x slower) |

Setup cost: Python session 3.951 ms, Rust estimator 122.381 ms.

## Observations and Phase 3 targets

- MiniMax-style models: Rust is already ~1.55x faster on both context and
  generation hot paths. Need an additional ~2x to hit the 3x exit criterion.
- Kimi/DeepSeek-style MLA models: Rust is currently slower than Python by
  ~5x (context) and ~1.5x (generation). The aggregate-formula path
  overestimates work here. Phase 3's proper MLA op graph (C6 `models/deepseek`
  + C4 `operators/mla`) is the primary remediation.
- Rust estimator setup (~120-135 ms) is one-time and excluded from the hot
  path target. Stage 2's lazy per-op-owner loading should keep it from
  growing as more op tables come online.

## Update policy

- Re-capture the snapshot at the end of every commit that touches the hot
  path (C2, C3, C5, C6, C7) per the Phase 3 plan in
  `../docs/phase1/migration-execution-plan.md`.
- Keep the latest snapshot under "Latest" below the baseline; do not
  overwrite the baseline.
- Final Phase 3 snapshot lands in C12 alongside the docs update (see
  "Phase 3 final snapshot (C12)" below).
- Subsequent Phase 5 hot-path optimization commits append their
  snapshots; the Phase 3 final snapshot stays as the parity-only
  baseline so the Phase 5 wins are bisectable against it.

## After C7 (Phase 3 pipeline wired through FFI)

Captured 2026-05-28 immediately after commit `cb574cc8` (C6/C7 model
graphs + session driver + FFI rewire). FFI now routes through the new
modular `Phase3Estimator` (perf_database/{gemm,attention,mla,moe,...} +
operators/{...} + models/{moe,deepseek,llama}). Same host
(Mac15,7 / Darwin 25.5.0 / arm64 / rustc 1.95.0 / Python 3.12.12) and
same smoke parallelism (`tp=8, pp=1, moe_tp=1, moe_ep=8`).

### MiniMaxAI/MiniMax-M2.5 — b200_sxm/vllm/0.19.0

| Phase | Python p50 (us) | Rust p50 (us) | Rust speedup |
| --- | --- | --- | --- |
| context | 63.94 | 20.83 | **3.07x** ✓ meets 3x exit criterion |
| generation | 22.75 | 21.00 | 1.08x |

### moonshotai/Kimi-K2.5 — b200_sxm/vllm/0.19.0

| Phase | Python p50 (us) | Rust p50 (us) | Rust speedup |
| --- | --- | --- | --- |
| context | 25.67 | 19.96 | 1.29x (was 0.21x / 5x slower at baseline) |
| generation | 30.98 | 23.08 | 1.34x (was 0.69x / 1.5x slower at baseline) |

### Parity drift after Phase 3 pipeline

| Surface | Baseline drift (legacy aggregate) | After C7 (Phase 3 pipeline) |
| --- | --- | --- |
| MiniMax static_ctx | -6.19% | +13.81% |
| MiniMax static_total | -5.22% | +12.86% |
| MiniMax mixed_step | -17.97% | +12.86% |
| MiniMax agg_ttft | -6.24% | +29.53% |
| MiniMax disagg_ttft | -6.19% | +13.81% |
| Kimi static_ctx | -13.03% | **+2.59%** ← close to <1% target |
| Kimi static_total | -14.89% | **+1.75%** |
| Kimi mixed_step | -19.47% | **+1.75%** |
| Kimi agg_ttft | -13.05% | +10.53% |
| Kimi disagg_ttft | -13.03% | **+2.59%** |

## After apple-to-apple refactor (commit `981b40c4`)

Restructured Rust to mirror Python's `model.context_ops`/`generation_ops`
op-list pattern with session-level orchestration. Removed a spurious
per-layer all-reduce in the MoE model that Python doesn't have (Python
composes per-layer comm inside `MoEDispatch`).

### Parity drift

| Surface | After C7 (trait-based) | After refactor (op-list) |
| --- | --- | --- |
| **Kimi static_ctx** | +2.59% | **-0.81%** ✓ (<1% target) |
| **Kimi disagg_ttft** | +2.59% | **-0.81%** ✓ |
| Kimi static_gen | -8.78% | -11.85% |
| Kimi mixed_step | +1.75% | -5.66% |
| Kimi agg_ttft | +5.49% → +10.53% | +2.09% |
| **MiniMax static_gen** | +6.97% | **+0.40%** ✓ |
| MiniMax static_ctx | +13.81% | +3.59% |
| MiniMax mixed_step | +12.86% | -9.65% |
| MiniMax agg_ttft | +18.32% | +3.56% |
| MiniMax disagg_ttft | +13.81% | +3.59% |
| MiniMax-sampled static_ctx | +17.90% | +8.31% |
| MiniMax-sampled agg_ttft | +37.16% | +14.44% |

### Performance (hot-path p50 speedup vs Python)

| Surface | After C7 | After refactor |
| --- | --- | --- |
| MiniMax context | 3.07x | 1.33x |
| MiniMax generation | 1.08x | 1.14x |
| Kimi context | 1.29x | 1.41x |
| Kimi generation | 1.34x | 1.62x |

All paths are still faster than Python. MiniMax context's earlier 3.07x
reflected the under-counting legacy aggregate path (it was computing
fewer-than-real ops, hence "fast"); the refactor brings the op count
back to Python parity, so the speedup converges toward Kimi's 1.3-1.6x
range. The 3x speedup exit criterion is now ambitious given the
faithful op composition — reaching it requires hot-path optimization
(C12 in the plan).

### Key takeaways

- **3 of 12 smoke surfaces now within 1% drift** (Kimi static_ctx,
  Kimi disagg_ttft, MiniMax static_gen) — the first <1% surfaces in
  Phase 3.
- **Apple-to-apple structure decisively beats trait-based composition**
  for parity. Most surfaces moved from 13-29% drift to 0-9% drift.
- **Several surfaces over-correct slightly** (Kimi static_gen -11.85%,
  Kimi mixed_step -5.66%, MiniMax mixed_step -9.65%) — indicates the
  generation phase or mix step is missing a small contributor. Next
  iteration target.

## After Kimi op-graph fixes (commit `c451cdb9`)

Added missing pieces to the Kimi/DeepSeek model graph:

- Shared expert FFN (gate-up GEMM + activation + ffn2 GEMM per layer) —
  Kimi/DeepSeek has `n_shared_experts=1` that every token traverses.
- Context-phase `logits_gemm` (Python's DeepSeek includes this in
  context_ops; MOE puts it only in generation_ops).
- vLLM-specific `tp_allreduce` with `scale_factor = 2 × num_layers`
  (covers attention + FFN AR per layer that vLLM's AllReduceFusionPass
  can't fuse outside pure-decode CUDA-graph steps).
- Removed the spurious `embedding_ar` for DeepSeek (Python uses only
  `tp_allreduce` for vLLM Kimi; embedding_ar exists for MOE but not
  DeepSeek).

Also fixed `CustomAllReduceOp.query` to pass `num_tokens × hidden`
(element count) to the perf table instead of `× hidden × dtype_memory`
(byte count) — matches Python's `size = kwargs["x"] * self._h`.

### Parity drift

| Surface | After C7 | After refactor | After Kimi fixes |
| --- | --- | --- | --- |
| **MiniMax static_gen** | +0.40% | +0.40% | **0.00%** ✓ |
| **Kimi static_gen** | -8.78% | -11.85% | **-0.04%** ✓ |
| **MiniMax-sampled static_gen** | +0.52% | +0.52% | **0.00%** ✓ |
| MiniMax static_ctx | +13.81% | +3.59% | -4.92% |
| MiniMax static_total | +12.86% | +3.30% | -4.24% |
| Kimi static_ctx | +2.59% | -0.81% | +3.39% |
| Kimi static_total | +1.75% | -0.45% | +3.13% |
| MiniMax-sampled static_ctx | +17.90% | +8.31% | -3.30% |
| Kimi agg_ttft | +5.49% | +2.09% | +3.42% |
| MiniMax agg_ttft | +18.32% | +3.56% | -4.86% |
| Kimi disagg_ttft | +2.59% | -0.81% | +3.39% |
| MiniMax disagg_ttft | +13.81% | +3.59% | -4.93% |
| Kimi mixed_step | -5.66% / +1.75% | -7.41% | -4.23% |
| MiniMax mixed_step | +12.86% | -9.65% | -16.99% |

### Performance

| Surface | After refactor | After Kimi fixes |
| --- | --- | --- |
| MiniMax context | 1.33x | 1.35x |
| MiniMax generation | 1.14x | 1.14x |
| Kimi context | 1.41x | 1.38x |
| Kimi generation | 1.62x | 1.51x |

All paths still faster than Python; speedup slightly lower as more ops
are now correctly modeled.

### Takeaway

- **6 of 12 surfaces within 1% drift**: doubled from 3.
- **Most other surfaces within 3-5% drift**: down from 13-29% earlier.
- **Largest remaining real drift**: MiniMax ~5% under (likely missing
  one MoEDispatch component) and `mixed_step` (apples-vs-oranges:
  Python static vs Rust mix-step composition).

Total Phase 3 commits to date: 18. From a -19% / +29% baseline to a
~5% / 0% range across smoke surfaces.

Notes:
- The new pipeline produces qualitatively different numbers than the
  legacy aggregate path — drift direction can flip per surface as the
  modular op graph composes individual table queries instead of
  computing from aggregate formulas.
- Kimi drift dropped substantially (was -14%/-19%, now +1.75%/+10%) —
  the MLA module table path is now active.
- MiniMax over-shoots ~13-30% — likely op double-counting in the MoE
  graph (e.g., extra dispatch latency or AR included beyond Python's
  composition). Identification + fix is the next iteration target.
- All hot paths are now faster than Python. MiniMax context hits the
  3x speedup exit criterion; Kimi context recovered from 5x slower
  (legacy) to 1.29x faster.

## After data-loader + mix-step parity fixes (this commit)

Closed the remaining op-graph and runtime gaps:

- **Norm bytes/token formula**: Python's `ElementWise(num_layers, 2h,
  2h, 0.8)` reads/writes `(2h + 2h) * 2` bytes per token; the legacy
  Rust value used the `0.8` factor (already applied inside
  `mem_op_latency_ms`) and dropped a bfloat16 factor, undercounting by
  2.5x. Fixed in `models/{moe,deepseek,llama}.rs`.

- **DeepSeek MoE workload distribution**: Python's DeepSeek family uses
  `power_law_1.01`; the Rust DeepSeek builder hardcoded
  `power_law_1.2` (the MOE-family value). Switched DeepSeek to
  `power_law_1.01`.

- **Perf-DB first-wins on duplicate rows**: Python's loaders use a
  `try/except KeyError` pattern that keeps the FIRST occurrence on
  duplicate `(shape, query_axis)` rows. The Rust loaders previously
  used `BTreeMap::insert` which kept the LAST. Some perf files contain
  duplicate runs of the same shape — e.g., Kimi `int4_wo` MoE at 1024
  tokens has `2.187 ms` then `2.405 ms` for the same kernel_source.
  Applied `.entry(...).or_insert(...)` across all perf-DB loaders
  (`gemm`, `attention`, `mla`, `moe`, `wideep`, `dsa`, `dsv4`, `mhc`,
  `communication`, `state_space`).

- **Mix-step FPM convention alignment**: Python's
  `estimate_mixed_step_latency_with_rust` was sending
  `sum_prefill_tokens = ctx_tokens` (isl-equivalent units, includes
  prefix accounting), inconsistent with the static-ctx FPM where
  `sum_prefill_tokens = NEW tokens × batch`. Updated the agg→Rust
  converter to subtract cached tokens. Reworked
  `Phase3Estimator::get_mix_step_latency_ms` to compute combined
  effective ISL directly (`ctx_tokens + gen_tokens`, no prefix
  subtraction inside Rust) and added a per-request ISL/scale_factor
  for pass-2 context attention.

### Parity drift

| Surface | After Kimi fixes | After data-loader + FPM fixes |
| --- | --- | --- |
| **Kimi static_ctx** | +3.39% | **-0.78%** ✓ |
| **Kimi static_total** | +3.13% | **-0.71%** ✓ |
| **Kimi agg_ttft** | +3.42% | **<1%** ✓ (within tolerance) |
| **Kimi disagg_ttft** | +3.39% | **<1%** ✓ |
| **MiniMax static_ctx** | -4.92% | **<1%** ✓ |
| **MiniMax agg_ttft** | -4.86% | **<1%** ✓ |
| **MiniMax disagg_ttft** | -4.93% | **<1%** ✓ |
| **MiniMax-sampled static_ctx** | -3.30% | **<1%** ✓ |
| **MiniMax-sampled agg_ttft** | +6.75% | **<1%** ✓ |
| **MiniMax-sampled disagg_ttft** | (n/a) | **<1%** ✓ |
| Kimi mixed_step | -4.23% | -8.11% (structural) |
| MiniMax mixed_step | -16.99% | -12.75% (structural) |
| MiniMax-sampled mixed_step | -7.91% | -6.73% (structural) |

### Performance

| Surface | Latest (post-fix) p50 |
| --- | --- |
| MiniMax context | 1.30x |
| MiniMax generation | 1.08x |
| Kimi context | 1.23x |
| Kimi generation | 1.56x |

### Takeaway

- **12 of 12 surfaces now within 1% drift** (was 6/12 after Kimi
  op-graph fixes, 9/12 after the data-loader + FPM fixes above). All
  static / mixed_step / agg / disagg surfaces pass for both smoke
  models. The remaining `mixed_step` xfail was a test-shape bug
  (Python static_total vs Rust mix-step) — fixed by comparing Python's
  `_get_mix_step_latency` against Rust's `forward_pass_time_ms` for
  the same FPM, which is the actual apples-to-apples surface.

## Phase 3 final snapshot (C12)

Captured after the parity-flip commits (C8-C10) on
b200_sxm / vllm / 0.19.0. Smoke parallelism: `tp=8, pp=1,
attention_dp=1, moe_tp=1, moe_ep=8`. Three independent benchmark
runs (warmup=5, iterations=30 per run):

### Smoke speedup (p50)

| Surface | Python p50 | Rust p50 | Speedup |
| --- | --- | --- | --- |
| MiniMax-M2.5 context | ~24.1 us | ~18.6 us | 1.30x |
| MiniMax-M2.5 generation | ~22.7 us | ~20.2 us | 1.13x |
| Kimi-K2.5 context | ~26.5 us | ~20.2 us | 1.32x |
| Kimi-K2.5 generation | ~32.1 us | ~20.0 us | 1.61x |

### Parity drift (all surfaces within 1% tolerance)

| Surface | Status |
| --- | --- |
| MiniMax-M2.5 static / mixed_step / agg / disagg | passing |
| Kimi-K2.5 static / mixed_step / agg / disagg | passing |
| MiniMax-M2.5 sampled-prefix static / mixed_step / agg / disagg | passing |

The smoke parity assertions have been flipped from `pytest.xfail` to
hard `assert` (C8-C10); 12 of 12 surfaces pass within the 1% drift
tolerance.

### Where the remaining wall-clock is spent

Profiled per-call breakdown for the ~20us Rust hot path on the smoke
shape:

- Pure Rust compute: ~4.7us (per-op table lookups; the MoE op alone is
  ~2.3us).
- Python `_engine_config_json` rebuild per call: ~5.6us — the
  `@cache`d Rust estimator object is keyed by the JSON string, so the
  wrapper regenerates it every invocation.
- `copy.deepcopy` of FPM metrics in `_metrics_by_attention_dp_rank`:
  ~2.3us even for attention_dp_size=1.
- `json.dumps` of the FPM payload before ctypes: ~1.2us.
- ctypes call + Rust JSON parse: ~0.85us.
- Bench harness overhead (`_suppress_output` context manager,
  `time.perf_counter_ns` deltas, closure dispatch): ~2us.

### Why 3x is deferred to Phase 5

The Phase 3 exit criterion called for >=3x p50 speedup. Closing the
gap requires:

1. An identity-keyed (model, db) -> estimator cache on the Python side,
   to skip the per-call JSON rebuild.
2. A packed-primitive FFI fast-path that bypasses JSON encode/decode
   for homogeneous-batch FPMs.
3. A runtime sub-table cache inside the Rust perf-database so each op
   doesn't repeat its `(quant, distribution, shape)` resolution on
   every call.

Each of these introduces non-trivial code structure that drifts from
the Python reference (FFI bifurcation, interior-mutability caches,
identity-based Python dicts). Landing them inside the parity PR would
mix concerns and risk regressions against the parity invariants the
xfail-flip just locked in. They are scheduled as a dedicated Phase 5
(see `../docs/phase1/migration-execution-plan.md`).

## Post-Phase-4 Snapshot (Full Family Coverage)

Captured 2026-05-29 on commit `5c1341e3` (branch `codex/rust-phase-3`).

- Host: Apple M3 Pro (12 cores), Darwin 25.5.0 / macOS 26.5, arm64.
- Toolchain: Python 3.12.12, rustc 1.95.0.
- Workers: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` (BLAS pinned to 1 thread/worker).
- Harness: `benchmark_engine_step.py --warmup 5 --iterations 30` per phase; two runs (one `--cache-mode hot`, one `--cache-mode cold`).

### Smoke parallelism

All cases run at `b200_sxm/vllm/0.19.0`, `isl=1024 osl=2 batch=1`. Parallelism mirrors the entries in `test_engine_step_parity.py::SMOKE_CASES` so the perf-DB tables are known to resolve — the user-requested Qwen3-30B-A3B `tp=8/moe_ep=8` configuration misses perf data for that model's expert count, so this case uses the smoke-suite `tp=4/moe_ep=4` instead; all others use the listed-in-request parallelism.

### Per-case results

#### `minimax-m25` — MoE (MiniMax hybrid)

Model `MiniMaxAI/MiniMax-M2.5`, Phase 3 baseline.

| Phase | Cache | Python p50 (us) | Rust p50 (us) | Speedup (p50) |
| --- | --- | ---: | ---: | ---: |
| context | warm | 24.02 | 18.19 | 1.32x |
| context | cold | 3902.67 | 19.40 | 201.21x |
| generation | warm | 22.65 | 21.42 | 1.06x |
| generation | cold | 18618.15 | 20.54 | 906.35x |

#### `kimi-k25` — DeepSeek family (Kimi MLA)

Model `moonshotai/Kimi-K2.5`, Phase 3 baseline.

| Phase | Cache | Python p50 (us) | Rust p50 (us) | Speedup (p50) |
| --- | --- | ---: | ---: | ---: |
| context | warm | 25.54 | 21.96 | 1.16x |
| context | cold | 19943.92 | 20.46 | 974.87x |
| generation | warm | 31.56 | 20.19 | 1.56x |
| generation | cold | 18999.25 | 20.38 | 932.48x |

#### `qwen3-32b` — Llama/Qwen3 dense

Model `Qwen/Qwen3-32B`, tp=4.

| Phase | Cache | Python p50 (us) | Rust p50 (us) | Speedup (p50) |
| --- | --- | ---: | ---: | ---: |
| context | warm | 26.69 | 19.46 | 1.37x |
| context | cold | 22781.88 | 21.02 | 1083.79x |
| generation | warm | 22.67 | 18.88 | 1.20x |
| generation | cold | 19301.42 | 19.12 | 1009.22x |

#### `qwen3-30b-a3b` — MoE (Qwen3 MoE)

Model `Qwen/Qwen3-30B-A3B`, tp=4, moe_ep=4 (smoke-verified parallelism).

| Phase | Cache | Python p50 (us) | Rust p50 (us) | Speedup (p50) |
| --- | --- | ---: | ---: | ---: |
| context | warm | 24.69 | 17.62 | 1.40x |
| context | cold | 22519.56 | 17.69 | 1273.23x |
| generation | warm | 23.23 | 18.42 | 1.26x |
| generation | cold | 20318.83 | 18.42 | 1103.27x |

#### `deepseek-v3` — DeepSeek family (DSv3 MLA)

Model `deepseek-ai/DeepSeek-V3`, default tp=8/moe_ep=8.

| Phase | Cache | Python p50 (us) | Rust p50 (us) | Speedup (p50) |
| --- | --- | ---: | ---: | ---: |
| context | warm | 25.60 | 18.54 | 1.38x |
| context | cold | 19744.69 | 18.94 | 1042.62x |
| generation | warm | 31.50 | 20.96 | 1.50x |
| generation | cold | 19599.81 | 20.79 | 942.66x |

#### `deepseek-v32` — DeepSeekV32 (DSA attention)

Model `deepseek-ai/DeepSeek-V3.2`, default tp=8/moe_ep=8.

| Phase | Cache | Python p50 (us) | Rust p50 (us) | Speedup (p50) |
| --- | --- | ---: | ---: | ---: |
| context | warm | 20.67 | 36.81 | 0.56x |
| context | cold | 18361.85 | 37.23 | 493.21x |
| generation | warm | 29.27 | 50.35 | 0.58x |
| generation | cold | 19308.00 | 44.35 | 435.31x |

#### `nemotron-nas-49b` — NemotronNas (Puzzle/DeciLM)

Model `nvidia/Llama-3_3-Nemotron-Super-49B-v1`, default tp=8 (per-block config).

| Phase | Cache | Python p50 (us) | Rust p50 (us) | Speedup (p50) |
| --- | --- | ---: | ---: | ---: |
| context | warm | 224.67 | 150.56 | 1.49x |
| context | cold | 24410.40 | 150.73 | 161.95x |
| generation | warm | 204.58 | 144.12 | 1.42x |
| generation | cold | 23127.04 | 153.31 | 150.85x |

#### `nemotron-h-56b` — NemotronH (Mamba2 hybrid)

Model `nvidia/Nemotron-H-56B-Base-8K`, default tp=8.

| Phase | Cache | Python p50 (us) | Rust p50 (us) | Speedup (p50) |
| --- | --- | ---: | ---: | ---: |
| context | warm | 32.50 | 19.12 | 1.70x |
| context | cold | 23220.10 | 19.33 | 1201.06x |
| generation | warm | 34.12 | 18.60 | 1.83x |
| generation | cold | 20208.79 | 17.67 | 1143.87x |

#### `qwen35-397b-a17b` — Qwen3.5 (GDN + MoE hybrid)

Model `Qwen/Qwen3.5-397B-A17B`, default tp=8/moe_ep=8 (substitute for Qwen3-Next, see notes).

| Phase | Cache | Python p50 (us) | Rust p50 (us) | Speedup (p50) |
| --- | --- | ---: | ---: | ---: |
| context | warm | 52.88 | 21.15 | 2.50x |
| context | cold | 22349.40 | 21.40 | 1044.56x |
| generation | warm | 56.15 | 20.71 | 2.71x |
| generation | cold | 20482.19 | 23.62 | 866.97x |

### Summary (p50 speedups, Rust vs Python)

| Family | Model | Warm ctx | Warm gen | Cold ctx | Cold gen |
| --- | --- | ---: | ---: | ---: | ---: |
| MoE (MiniMax hybrid) | `MiniMaxAI/MiniMax-M2.5` | 1.32x | 1.06x | 201.2x | 906.3x |
| DeepSeek family (Kimi MLA) | `moonshotai/Kimi-K2.5` | 1.16x | 1.56x | 974.9x | 932.5x |
| Llama/Qwen3 dense | `Qwen/Qwen3-32B` | 1.37x | 1.20x | 1083.8x | 1009.2x |
| MoE (Qwen3 MoE) | `Qwen/Qwen3-30B-A3B` | 1.40x | 1.26x | 1273.2x | 1103.3x |
| DeepSeek family (DSv3 MLA) | `deepseek-ai/DeepSeek-V3` | 1.38x | 1.50x | 1042.6x | 942.7x |
| DeepSeekV32 (DSA attention) | `deepseek-ai/DeepSeek-V3.2` | 0.56x | 0.58x | 493.2x | 435.3x |
| NemotronNas (Puzzle/DeciLM) | `nvidia/Llama-3_3-Nemotron-Super-49B-v1` | 1.49x | 1.42x | 161.9x | 150.8x |
| NemotronH (Mamba2 hybrid) | `nvidia/Nemotron-H-56B-Base-8K` | 1.70x | 1.83x | 1201.1x | 1143.9x |
| Qwen3.5 (GDN + MoE hybrid) | `Qwen/Qwen3.5-397B-A17B` | 2.50x | 2.71x | 1044.6x | 867.0x |

### Findings

No family hits the >=3x warm-path target. Highest warm-path observation: Qwen3.5-397B-A17B at 2.50x context / 2.71x generation.

Families falling short of >=3x warm-path on both phases:
- `minimax-m25` (MoE (MiniMax hybrid)): warm ctx 1.32x, warm gen 1.06x.
- `kimi-k25` (DeepSeek family (Kimi MLA)): warm ctx 1.16x, warm gen 1.56x.
- `qwen3-32b` (Llama/Qwen3 dense): warm ctx 1.37x, warm gen 1.20x.
- `qwen3-30b-a3b` (MoE (Qwen3 MoE)): warm ctx 1.40x, warm gen 1.26x.
- `deepseek-v3` (DeepSeek family (DSv3 MLA)): warm ctx 1.38x, warm gen 1.50x.
- `deepseek-v32` (DeepSeekV32 (DSA attention)): warm ctx 0.56x, warm gen 0.58x.
- `nemotron-nas-49b` (NemotronNas (Puzzle/DeciLM)): warm ctx 1.49x, warm gen 1.42x.
- `nemotron-h-56b` (NemotronH (Mamba2 hybrid)): warm ctx 1.70x, warm gen 1.83x.
- `qwen35-397b-a17b` (Qwen3.5 (GDN + MoE hybrid)): warm ctx 2.50x, warm gen 2.71x.

Cold-path numbers (~150x to ~1250x) reflect Python re-loading perf-DB tables from disk on every iteration after `_reset_python_runtime_caches`, while the Rust estimator amortises table loading inside its FFI-side cache. Cold-path measures perf-DB load amortisation, not engine-step compute.

Notes:
- The user request listed Qwen3-Next-80B-A3B-Instruct for the Qwen3.5 family. That HuggingFace ID is not in the b200_sxm support matrix; we substituted `Qwen/Qwen3.5-397B-A17B` (the Qwen3.5 MoE hybrid representative used by the smoke suite).
- DeepSeek-V3.2 is the only family where Rust is currently slower than Python on the warm path (0.56x ctx, 0.58x gen) — likely a DSA-attention pipeline regression that is not yet covered by the Phase 3/4 hot-path optimisations.
- Data-gap / multimodal families (Gpt/oss-20b, HybridMoe/Llama-4-Scout, DeepSeekV4/Flash, Gemma4Moe, Qwen3VL) were skipped because their perf-DB tables miss the smoke shape — they would error-symmetrically in both engines, producing no benchable signal.

## After fix campaign C1+C2+C4+C4-residual (2026-06-01)

Captured 2026-06-01 on the `codex/rust-phase-3` branch with the fresh release dylib (`cargo build --release -p aiconfigurator-core`) after C1 (MoE dispatch backend differentiation), C2 (interpolation intersect-then-bracket), C4 (MTP `_mtp_scale_factor` plumbed into DeepSeek-family + Qwen3.5), and C4-residual (`(nextn+1)` batch multiplier in agg-path Rust bridges).

Host: Mac15,7 (Apple Silicon, Darwin 25.5.0 / macOS 26.5, arm64). Same smoke parallelism as prior snapshots (`tp=8, pp=1, attention_dp=1, moe_tp=1, moe_ep=8`), except `qwen3-32b` (tp=4) and `qwen3-30b-a3b` (tp=4/moe_ep=4) to fit perf-DB coverage. `--warmup 3 --iterations 25` on `b200_sxm/vllm/0.19.0`.

### Summary (p50 speedups, Rust vs Python, warm path)

| Family | Model | Context | Generation |
| --- | --- | ---: | ---: |
| MoE (MiniMax hybrid) | `MiniMaxAI/MiniMax-M2.5` | 1.15x | 0.94x |
| DeepSeek family (Kimi MLA) | `moonshotai/Kimi-K2.5` | 1.05x | 1.23x |
| Llama/Qwen3 dense | `Qwen/Qwen3-32B` | 1.21x | 0.99x |
| MoE (Qwen3 MoE) | `Qwen/Qwen3-30B-A3B` | 1.22x | 1.08x |
| DeepSeek family (DSv3 MLA) | `deepseek-ai/DeepSeek-V3` | 1.15x | **1.54x** |
| DeepSeekV32 (DSA attention) | `deepseek-ai/DeepSeek-V3.2` | 0.94x | 1.15x |
| NemotronNas (Puzzle/DeciLM) | `nvidia/Llama-3_3-Nemotron-Super-49B-v1` | 1.00x | 0.87x |
| NemotronH (Mamba2 hybrid) | `nvidia/Nemotron-H-56B-Base-8K` | **1.53x** | **1.71x** |
| Qwen3.5 (GDN + MoE hybrid) | `Qwen/Qwen3.5-397B-A17B` | **2.15x** | **2.23x** |

Setup overhead (one-time, excluded from per-call timings): Rust estimator ~210–240 ms (ctypes + Rust model build + perf-DB load); Python session ~3–4 ms.

### What changed vs the prior snapshot

The C2 interpolation fix and C4 MTP scaling brought DeepSeek-family + Qwen3.5 into structural parity with Python. Side-effect: the warm-path benchmark for those models now reflects the correct (heavier) per-step work, which compresses the speedup relative to the prior snapshot where Rust was systematically under-counting work. The DSv3.2 (0.56x → 0.94x ctx) recovery comes from the same fixes; DSv3.2 was the worst pre-fix outlier and is now within noise of parity.

### FFI overhead caveat — important context for these numbers

Every Rust column above is measured **end-to-end through ctypes**: Python builds a metrics dict → `json.dumps` + UTF-8 encode (`rust_engine_step.py:238`) → ctypes call → Rust JSON deserialize → Rust compute → return marshalling. Steps outside the actual Rust compute cost ~15–25 μs per call regardless of model size, so on small graphs (~25 μs Rust compute) the FFI overhead dominates and the table understates the Rust win.

Bucketing by graph size:

| Bucket | Pattern | Why |
| --- | --- | --- |
| Big graphs (Qwen3.5-397B 2.2x, Nemotron-H 1.7x, DSv3 gen 1.54x) | Rust wins big | Rust compute dominates; FFI tax amortised |
| Mid graphs (Qwen3 / Kimi / DSv3 ctx 1.0–1.2x) | Modest win | Rust compute ≈ FFI tax |
| Small graphs (MiniMax gen, Nemotron-NAS gen, DSv3.2 ctx 0.87–0.94x) | Slight regression | Rust compute so cheap that FFI tax exceeds Python's per-call cost |

The Python baseline pays **none** of the JSON/ctypes overhead — it's pure-Python compute end to end. The "true" Rust compute speedup is materially higher than the table shows; what's reported is *Rust compute speedup discounted by the FFI tax*. A direct measurement of the FFI tax (timing the same workload from a small pure-Rust binary) is open work — see "Future work" below.

### Future work — closing the gap to >=3x warm-path on small graphs

1. Batched FFI entry point (`estimate_batch(Vec<EngineConfig>) → Vec<f64>` with internal rayon) — pays JSON ser/deser **once per sweep** instead of once per point. Expected biggest single win for sweep workloads.
2. Engine handle reuse across sweep points — drop per-call metrics-dict construction to near zero by passing only `(batch, isl, osl)` on the hot path.
3. Pure-Rust caller path (no Python on the hot loop) for Mocker/replay-driven scenarios — design doc §2.3 right column.

See `../docs/design_doc.html` §2.3 for the architectural framing and `../docs/phase1/migration-execution-plan.md` for Phase 5 sequencing.

## E8 perf gate — new compiled-Engine (PyO3) vs old ctypes (2026-06-03)

Phase 1.5 E8 perf gate. Compares the **new compiled-Engine path**
(`EngineHandle.run_static` over the PyO3 `AicEngine`, at HEAD `d3c939fc`)
against the **old ctypes path** (`RustEngineStepEstimator.forward_pass_time_ms`,
the Phase 1 JSON/ctypes FFI). Exit criterion (plan E8 row): **≥3× p50
speedup** on the small-graph families that today regress below 1× warm-path
(MiniMax-M2.5 **gen**, NemotronNas **gen**, DSv3.2 **ctx**); big-graph families
stay at or above their Phase 1 speedup. This gate is best-effort and blocks
nothing downstream.

### Measurement method — worktree at E5 (not E6)

The old ctypes path was deleted at E7 (current HEAD). To measure it on *this*
machine, the OLD baseline was captured from a git worktree at the last
**pre-flip** commit.

> **Commit-selection correction.** The E8 task statement said "at E6,
> `benchmark_engine_step.py` still drives the ctypes `RustEngineStepEstimator`."
> That is **wrong**; the plan's E6 row is right. Reading the code,
> `estimate_static_latency_breakdown_with_rust` (the function that
> `run_static_latency_only(engine_step_backend="rust")` dispatches to) already
> routes through `_cached_engine_handle` / `EngineHandle.run_static` (PyO3) at
> **E6 (`520dcfff`)** — E6 *is* the parity flip; the `RustEngineStepEstimator`
> class is leftover dead code there. The last commit whose `"rust"` path
> genuinely drives ctypes (`_cached_estimator` → `forward_pass_time_ms`) is
> **E5 (`b5cd4742`)**. The OLD baseline below is therefore measured at E5, not
> E6.

- **OLD (ctypes):** worktree `git worktree add /tmp/claude/aic-e5 b5cd4742`,
  `cargo build --release` (the cdylib), then the E5 copy of
  `benchmark_engine_step.py` with `engine_step_backend="rust"`. Import
  resolution was forced to the worktree source via `PYTHONPATH=<worktree>/src`
  and the dylib pinned via `AICONFIGURATOR_RUST_CORE_LIB=<worktree>/target/release/libaiconfigurator_core.dylib`.
  Verified before timing: `aiconfigurator.__file__` resolves under the worktree;
  the rust-static func uses `_cached_estimator` (ctypes), **not**
  `_cached_engine_handle` (PyO3); the loaded dylib is the worktree build.
- **NEW (PyO3):** HEAD checkout, the post-E7 `benchmark_engine_step.py` with
  `engine_step_backend="rust"` (routes through `EngineHandle.run_static`).
- **Validity of OLD = ctypes (no re-run needed):** the OLD arm reproduces the
  sub-1× regression the plan/C1 snapshot describe — old-ctypes-vs-python p50:
  MiniMax gen **0.97×**, NemotronNas gen **0.81×**, DSv3.2 ctx **0.93×**. Only
  the ctypes path produces <1× on those three (HEAD's PyO3 is >1× there), so
  reproducing the regression *is* evidence the OLD arm measured ctypes.

**Methodology:** warm/hot path, `--warmup 50 --iterations 1000 --cache-mode hot`
(N=1000 timed samples per phase; setup/compile/perf-DB-load excluded — built
once, then repeated `run_static` / `forward_pass_time_ms` calls timed).
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. Percentile = p50 (median over N).
All cases `b200_sxm/vllm/0.19.0`, `isl=1024 osl=2 batch=1`, smoke parallelism
`tp=8/pp=1/dp=1/moe_tp=1/moe_ep=8` except `qwen3-32b` (tp=4) and
`qwen3-30b-a3b` (tp=4/moe_ep=4) for perf-DB coverage. New-path one-time setup
(EngineHandle compile + Rust perf-DB load) ≈ 0.2 ms after the process-wide
first compile (~305 ms first time), excluded from per-call timing.

Host: Mac15,7 (Apple Silicon, Darwin 25.5.0 / macOS 26.5, arm64).
Toolchain: rustc 1.95.0, Python 3.12.12.

### Parity spot-check (the benchmark times the right latencies)

Off-loop `run_static_latency_only` on `MiniMaxAI/MiniMax-M2.5`, new-PyO3 vs
pure-Python returned latency (ms), same shape:

| mode | new-PyO3 (ms) | python (ms) | drift |
| --- | ---: | ---: | ---: |
| static_ctx | 39.234388 | 39.234388 | +0.000% |
| static_gen | 6.304839 | 6.302400 | +0.039% |

The new path returns the parity-validated latencies (within the 1% E6 gate
tolerance), so the timing numbers below measure correct work.

### Primary axis — new-PyO3 p50 vs old-ctypes p50 (osl=2)

| Family | Phase | Old ctypes p50 (us) | New PyO3 p50 (us) | Speedup |
| --- | --- | ---: | ---: | ---: |
| MiniMax-M2.5 | context | 21.12 | 15.75 | 1.34x |
| MiniMax-M2.5 | **generation** | 24.25 | 18.71 | **1.30x** |
| Kimi-K2.5 | context | 25.94 | 20.08 | 1.29x |
| Kimi-K2.5 | generation | 27.17 | 21.08 | 1.29x |
| Qwen3-32B | context | 24.12 | 17.08 | 1.41x |
| Qwen3-32B | generation | 23.79 | 17.08 | 1.39x |
| Qwen3-30B-A3B | context | 21.54 | 15.04 | 1.43x |
| Qwen3-30B-A3B | generation | 23.46 | 16.75 | 1.40x |
| DeepSeek-V3 | context | 23.21 | 16.71 | 1.39x |
| DeepSeek-V3 | generation | 24.54 | 18.04 | 1.36x |
| DeepSeek-V3.2 | **context** | 23.17 | 16.42 | **1.41x** |
| DeepSeek-V3.2 | generation | 24.12 | 17.50 | 1.38x |
| NemotronNas-49B | context | 264.08 | 93.38 | 2.83x |
| NemotronNas-49B | **generation** | 253.58 | 90.92 | **2.79x** |
| Nemotron-H-56B | context | 21.12 | 15.54 | 1.36x |
| Nemotron-H-56B | generation | 20.75 | 15.17 | 1.37x |
| Qwen3.5-397B-A17B | context | 27.33 | 20.54 | 1.33x |
| Qwen3.5-397B-A17B | generation | 26.71 | 20.21 | 1.32x |

### Why osl=2 understates the named-gen mechanism — osl=512 re-run

The gate's stated mechanism is "eliminate the per-call Python op-walk over
**decode steps**." HEAD's `run_static` performs the decode-stride quadrature
**inside Rust** (one PyO3 call; Rust loops over osl). The E5 ctypes path loops
**in Python**: `for i in range(osl-1): estimator.forward_pass_time_ms(...)` —
one full FFI roundtrip per decode step. At **osl=2 the decode loop runs once on
both arms**, so the elimination is invisible — the osl=2 generation column above
is essentially a single-step comparison. Re-running the named gen regressors at
a realistic `osl=512` (both arms, `--warmup 10 --iterations 200`):

| Family | Phase | Old ctypes p50 (us) | New PyO3 p50 (us) | Speedup |
| --- | --- | ---: | ---: | ---: |
| MiniMax-M2.5 | generation | 7505.6 | 4156.8 | 1.81x |
| NemotronNas-49B | **generation** | 121467.0 | 40318.2 | **3.01x** ✓ |
| DeepSeek-V3.2 | generation | 6849.9 | 3838.5 | 1.78x |
| MiniMax-M2.5 | context | 22.3 | 15.8 | 1.41x |
| NemotronNas-49B | context | 268.8 | 96.4 | 2.79x |
| DeepSeek-V3.2 | context | 23.3 | 17.9 | 1.30x |

At osl=512 the per-step Rust *compute* dominates and both arms become
compute-bound, so the FFI-loop elimination buys a roughly constant per-step
delta: ~1.8× on the small graphs (MiniMax, DSv3.2) and **3.01× on the
big-graph NemotronNas** where Rust compute is a larger fraction of each step.
DSv3.2 **context** is single-pass (no decode loop) so high osl can't move
its work into Rust — its win stays ~1.3-1.4×.

### Gate verdict per named family

Three independent axes (all on the same N=1000 warm runs):

| Named regressor | (1) new-vs-old ≥3×? | (2) regression fixed (new-vs-py >1×)? | (3) big-graph ≥ Phase 1? |
| --- | --- | --- | --- |
| MiniMax-M2.5 **gen** | ✗ 1.30× (osl=2), 1.81× (osl=512) | ✓ 1.22× (was 0.97×) | small graph |
| NemotronNas **gen** | ~✓ 2.79× (osl=2), **3.01× (osl=512)** | ✓ 2.28× (was 0.81×) | ✓ |
| DSv3.2 **ctx** | ✗ 1.41× | ✓ 1.31× (was 0.93×) | small graph |

**Big-graph families stay at or above Phase 1 speedup (axis 3 — MET
universally).** Comparing new-vs-python against old-vs-python (Phase 1):

| Family | Phase | new-vs-py | old-vs-py (Phase 1) | ≥ Phase 1? |
| --- | --- | ---: | ---: | --- |
| NemotronNas-49B | context | 2.51x | 0.88x | ✓ |
| NemotronNas-49B | generation | 2.28x | 0.81x | ✓ |
| Nemotron-H-56B | context | 2.19x | 1.60x | ✓ |
| Nemotron-H-56B | generation | 2.22x | 1.61x | ✓ |
| Qwen3.5-397B-A17B | context | 2.75x | 2.05x | ✓ |
| Qwen3.5-397B-A17B | generation | 2.93x | 2.18x | ✓ |
| DeepSeek-V3 | generation | 1.93x | 1.32x | ✓ |
| (all other families) | both | 1.2–1.8x | 0.9–1.3x | ✓ |

### Verdict

**E8 gate: PARTIALLY MET (best-effort, blocks nothing).**

- **≥3× new-vs-old (primary criterion):** MET on **NemotronNas gen** (3.01× at
  realistic osl=512; 2.79× even at osl=2). **MISSED on MiniMax-M2.5 gen
  (1.30×/1.81×) and DSv3.2 ctx (1.41×).**
- **Regression eliminated:** ✓ on **all three** named families — every formerly
  sub-1× warm-path surface (MiniMax gen 0.97→1.22×, NemotronNas gen 0.81→2.28×,
  DSv3.2 ctx 0.93→1.31×) is now >1× vs Python. The Phase-1 ctypes regression is
  gone.
- **Big-graph ≥ Phase 1:** ✓ MET universally.

**Why 3× is not met on the two small-graph surfaces (honest cause).** The gate's
premise was that the old ctypes per-call cost was dominated by a ~15–25 µs FFI
tax sitting on top of tiny Rust compute, so removing the tax would yield ≥3×. On
*this* machine at E5 the entire old-ctypes call was only ~21–27 µs for small
graphs — the per-call op-walk + JSON + ctypes overhead was real but not 2× the
compute. The new PyO3 path removes the Python op-walk and per-call JSON
ser/deser, landing at ~15–21 µs — a genuine ~1.3–1.8× win, but the residual
~15 µs is now the shared `run_static_latency_only` Python harness floor
(closure dispatch, runtime-config replace, dict assembly) plus the irreducible
Rust compute, which together exceed old/3 ≈ 7 µs. DSv3.2 **ctx** in particular
is single-pass, so there is no decode loop to move into Rust; its ceiling is the
harness floor regardless of osl. 3× on these surfaces requires shrinking that
shared Python harness floor (sweep-level batching / handle reuse, Phase-5
items below), not more FFI work.

### Notes / follow-ups (NOT implemented here — measurement-only stage)

- **Cheap win candidate (follow-up, do not implement in E8):** the ~15 µs
  small-graph floor is now in the shared Python `run_static_latency_only`
  path, not the Rust core. A batched / handle-reuse entry point that amortises
  the per-call dict assembly across sweep points would lift the small-graph
  surfaces toward 3× — see the Phase-5 "Future work" list above
  (batched FFI / engine-handle reuse / pure-Rust caller).
- The osl=2 generation column is a single-decode-step comparison by
  construction; the osl=512 table is the representative decode-phase number.
