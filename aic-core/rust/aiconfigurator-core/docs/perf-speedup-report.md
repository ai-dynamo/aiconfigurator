<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rust↔Python Engine-Step Speedup Report

**Generated: 2026-07-13 (UTC).**
**Commit: `62633bba3`** (`chore: replace mislabeled vLLM v0.19 FP16 AR data with
real bf16 (#1339)`) **plus the `SiteIndex::resolve` hot-path optimization** in
`src/perf_database/perf_interp.rs` (this change — cache per-site distance +
curve-coverage bounds, `select_nth` instead of a full site sort, single working
allocation).

This report replaces the old `benchmarks.md` (whose numbers were captured on
pre-merge migration branches and no longer reflect `main`). It is the perf
sibling of `parity-scan-report.md`; regenerate both together and re-stamp the
date/commit whenever the Rust hot path changes.

## What is measured

Per-call **engine-step latency**, hot cache, pure compute — the same
`run_static` path the SDK uses. Python and Rust are timed **back-to-back on the
same host**, so the reported **speedup ratio is far more comparable across
machines** than absolute wall-clock: most machine-speed variance divides out (an
x86 CI runner and this ARM host give different µs but broadly similar ratios —
the ratio can still shift somewhat across architectures). Absolute µs below are
for context only.

- Harness: `benchmark_engine_step.py --warmup 10 --iterations 50` (hot cache;
  runtime query caches cleared once per case, then warmed). One-time session /
  estimator setup is excluded from per-call latency.
- Metric: p50 over 50 timed samples, per phase (context / generation).
- All cases: `b200_sxm / vllm / 0.19.0`, `isl=1024 osl=2 batch=1`.
- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` (BLAS pinned; apples-to-apples serial).
- Host: Apple M3 Pro, Darwin 25.5.0 / arm64. Toolchain: rustc 1.97.0,
  Python 3.12.12. CPU-only; both engines run the latency model in-process.

## Results (warm p50, Rust vs Python)

Ordered by speedup. Every surface is faster in Rust.

| Family | Model | Parallelism | Phase | Python p50 (µs) | Rust p50 (µs) | Speedup |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Qwen3.5 (GDN+MoE) | `Qwen/Qwen3.5-397B-A17B` | tp8/ep8 | generation | 59.29 | 23.38 | **2.54×** |
| Qwen3.5 (GDN+MoE) | `Qwen/Qwen3.5-397B-A17B` | tp8/ep8 | context | 56.60 | 23.79 | **2.38×** |
| NemotronNas (Puzzle) | `nvidia/Llama-3_3-Nemotron-Super-49B-v1` | tp8 | context | 208.58 | 91.54 | **2.28×** |
| NemotronH (Mamba2) | `nvidia/Nemotron-H-56B-Base-8K` | tp8 | generation | 34.02 | 16.83 | 2.02× |
| NemotronNas (Puzzle) | `nvidia/Llama-3_3-Nemotron-Super-49B-v1` | tp8 | generation | 175.83 | 94.23 | 1.87× |
| NemotronH (Mamba2) | `nvidia/Nemotron-H-56B-Base-8K` | tp8 | context | 31.58 | 17.92 | 1.76× |
| DeepSeek (DSv3 MLA) | `deepseek-ai/DeepSeek-V3` | tp8/ep8 | generation | 28.25 | 18.81 | 1.50× |
| DeepSeekV32 (DSA) | `deepseek-ai/DeepSeek-V3.2` | tp8/ep8 | generation | 25.81 | 17.21 | 1.50× |
| DeepSeek (Kimi MLA) | `moonshotai/Kimi-K2.5` | tp8/ep8 | generation | 27.88 | 19.29 | 1.44× |
| MoE (MiniMax) | `MiniMaxAI/MiniMax-M2.5` | tp8/ep8 | context | 24.60 | 17.17 | 1.43× |
| Llama/Qwen3 dense | `Qwen/Qwen3-32B` | tp4 | context | 24.96 | 17.79 | 1.40× |
| MoE (Qwen3) | `Qwen/Qwen3-30B-A3B` | tp4/ep4 | context | 24.17 | 17.38 | 1.39× |
| MoE (Qwen3) | `Qwen/Qwen3-30B-A3B` | tp4/ep4 | generation | 23.88 | 19.17 | 1.25× |
| Llama/Qwen3 dense | `Qwen/Qwen3-32B` | tp4 | generation | 22.33 | 18.25 | 1.22× |
| DeepSeek (DSv3 MLA) | `deepseek-ai/DeepSeek-V3` | tp8/ep8 | context | 21.85 | 19.17 | 1.14× |
| MoE (MiniMax) | `MiniMaxAI/MiniMax-M2.5` | tp8/ep8 | generation | 21.60 | 19.38 | 1.12× |
| DeepSeek (Kimi MLA) | `moonshotai/Kimi-K2.5` | tp8/ep8 | context | 21.79 | 19.90 | 1.10× |
| DeepSeekV32 (DSA) | `deepseek-ai/DeepSeek-V3.2` | tp8/ep8 | context | 20.44 | 18.96 | 1.08× |

**Range: 1.08×–2.54×. All 9 families faster in Rust on both phases.**

One-time setup (excluded from the per-call latency above): Python session
~4–5 ms, Rust estimator (ctypes/PyO3 load + model + perf-DB load + construct)
~4–8 ms.

## How to read the range

The speedup scales with **how much interpreter work the graph does per step**,
because each engine-step call pays a roughly fixed ~15–25 µs Python↔Rust FFI tax
(dict build + JSON/PyO3 marshalling) that Rust's compiled op-graph walk cannot
remove:

- **Large graphs** (Qwen3.5-397B, NemotronNas, NemotronH — Python 30–210 µs):
  Rust compute dominates, the fixed FFI tax is amortised → **1.8–2.5×**.
- **Small graphs** (the ~20 µs dense / MoE / MLA models): the fixed tax is a
  large fraction of the tiny compute, so the ratio compresses toward the tax
  floor → **1.1–1.5×**. A regression here is also less impactful in absolute
  terms.

This is a single-thread comparison. A sweep fans out independent points; because
the Rust core releases the GIL (`py.allow_threads`), those points can run truly
in parallel — a further ~N× on N cores that these serial per-call numbers do not
capture.

## Regression guard

The engine-step is protected in CI by three gates in the `rust-engine-step-parity`
job (`.github/workflows/build-test.yml`):

- **Engine-step parity** (`test_engine_step_parity.py`): Rust vs Python numeric
  drift < 1%.
- **Compile-engine parity** (`test_compile_engine_parity.py`): the
  `compile_engine` → `EngineHandle` op-transfer round-trip plus integration
  parity vs the Python `BaseBackend`.
- **Perf** (`test_engine_step_perf.py`): Rust-vs-Python p50 speedup ≥ a per-case
  floor — the same-runner guard that would have caught the `perf_interp` v2
  regression this report's fix resolves (it had pushed the Rust step to
  0.15–0.78× of Python on the large-graph families before the
  `SiteIndex::resolve` fix).

## How to reproduce

```bash
(cd aic-core && ../.venv/bin/maturin develop --release)  # build the current Rust core
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python aic-core/rust/aiconfigurator-core/parity_tests/benchmark_engine_step.py \
  --warmup 10 --iterations 50 --json
```

Omit `--case` (as above) to run all families; pass `--case nemotron-nas-49b`
for a single family. Re-stamp the date and commit at the top of this file when
regenerating.
