# AIC Rust Migration Job Definition

> **Status (2026-06-01):** Phases 0–4 of this plan are complete and
> checkpointed as **Phase 1** of the higher-level migration project.
> See `phase-1-checkpoint.md` for the wrap-up state.
>
> The original Phase 5 (per-call hot-path optimization) is **superseded
> by Phase 1.5** (Python-builds / Rust-executes architecture). See
> `../phase-1.5-execution-plan.md`.
>
> This file is preserved verbatim as the immutable Phase 1 contract.

Here is the project plan. Don’t assume. Don’t hide confusion. Surface tradeoffs. Ask me for anything that needs clarity.

## Goal

Migrate AIC Python SDK engine-step latency logic to Rust, with the ultimate goal of speeding up each engine-step API call.

The top priority is low per-call latency. At the same time, the Rust code should be clean, readable, modular, and extensible as long as that does not sacrifice performance. Prefer the minimum coherent implementation that solves the problem. Do not add speculative abstractions or future features.

## Current Iteration Goal

Create the migration map, parity smoke tests, and benchmark harness needed to safely migrate AIC Python SDK engine-step latency logic into Rust.

Do not implement the full Rust migration until the migration map, parity harness, and benchmark harness are in place.

## Context

There is a POC in `./rust`. It partially works, but it oversimplifies AIC’s architecture. The goal is not to polish the POC as-is. The goal is to build a Rust crate that matches the Python SDK’s observable behavior while improving per-engine-step call latency.

I have coarser-grained tickets here:

https://linear.app/nvidia/project/aic-refactor-reuse-rust-engine-step-latency-api-aa6fc06a9e9d/issues

Feel free to propose sub-issues or new tickets if needed, but do not create them without summarizing why they are needed.

## Relevant Architecture

Current flow:

```
Python frontend -> Python SDK -> perf DB CSV files
```

Target flow:

```
Python frontend -> Rust SDK/core through ABI/bindings -> perf DB CSV files
```

Ownership boundaries:

- AIC core engine-step latency logic should move to Rust.
- AIC CLI, collectors, config generators, Pareto analysis, and other orchestration/UI layers should remain in Python.
- Primary implementation changes should be in `./rust`.
- Python changes are allowed only for tests, thin bindings, or existing Rust integration points such as `src/aiconfigurator/sdk/rust_engine_step.py`.
- Do not refactor unrelated Python SDK internals.

Design guidance:

- Use the Python SDK as the behavior reference, not necessarily the implementation template.
- Preserve the Python SDK’s modular concepts such as `operators/`, `models/`, and `backends/`.
- The Python SDK is still undergoing refactors. Do not blindly translate deprecated or redundant Python code into Rust.
- For example, `perf_database.py` has known redundancy as described in AIC-533:
https://linear.app/nvidia/issue/AIC-533/phase-45-remove-deprecated-perfdatabasequery-methods-update-test
- Also do not copy Python's eager perf DB startup shape. `PerfDatabase.__init__`
  still eagerly calls many op-family loaders as a transition compromise while
  tests migrate to lazy loading. Rust should load only metadata at
  database/session construction, then load each perf-file family lazily on first
  use or explicitly prewarm only the op families required by the current
  model/backend slice.
- Keep one table owner per op family. Do not duplicate deprecated Python
  `PerfDatabase.query_*` compatibility wrappers as Rust `PerfDatabase` methods
  when the behavior belongs in an operator/table module. Preserve parity with
  operator/query-boundary tests before removing or deduplicating Python-era
  paths.
- If Python behavior appears deprecated, redundant, buggy, or unclear, document the issue before deciding whether Rust should match it or intentionally diverge.

Core rule:

```
Match Python SDK observable behavior, not its internal structure.
```

## Tasks

### Phase 0: Migration Map

Status: complete for the current iteration.

Create a migration map before implementing the full Rust migration.

Deliverables:

1. Map relevant Python SDK modules/files to proposed Rust modules/paths.
2. Identify public behaviors the Rust implementation must match.
3. Identify deprecated or redundant Python logic that should not be copied directly.
4. Propose the Rust module design.
5. Create a system diagram showing current flow and target flow.
6. List open questions, architecture tradeoffs, and stop conditions.

### Phase 1: Parity Smoke Harness

Status: complete. The smoke tests live under
`rust/aiconfigurator-core/parity_tests/`. After Phase 3 (C8-C10) all 12
surfaces pass as required assertions; the original xfail markers are
gone.

Add pytest parity smoke tests comparing the existing Python SDK against the current Rust integration.

The tests should use:

```
src/aiconfigurator/sdk/rust_engine_step.py
```

for the Rust implementation path, and compare it against the Python SDK for reported metrics.

Initial smoke coverage:

- MiniMaxAI/MiniMax-M2.5
- Kimi-K2.5
- vLLM 0.19.0
- Sampled forward-pass parameters

During Phase 3 build-up these tests were `pytest.xfail`-able while
specific cases were still drifting; that pattern is now retired. The
Phase 3 (C8-C10) flip turned them into hard `assert`s so a regression
would be loud. Add a fresh `pytest.xfail` for any newly added case that
genuinely isn't ready yet, with the drift print attached, so failures
stay explicit and explained.

Do not let expected current failures hide unrelated regressions.

Example parity case:

```
MiniMaxAI/MiniMax-M2.5
System: b200_sxm
Backend: vLLM 0.19.0
Forward pass:
- ISL = 1024
- OSL = 2
- prefix = 0
```

Python AIC:

```
run_static(mode="static_ctx") = 41.879 ms
run_static(mode="static_gen") = 5.808 ms
total = 47.687 ms
```

Rust AIC FPM:

```
ForwardPassMetrics:
- prefill_reqs = 1
- prefill_tokens = 1024
- prefill_kv = 0
- decode_reqs = 1
- decode_kv = 1024

forward_pass_time_ms = 30.050 ms
```

End-state parity target:

```
Rust and Python metrics differ by <1% on the agreed parity suite.
```

### Phase 2: Benchmark Harness

Status: complete for the current iteration. The benchmark harness lives at
`rust/aiconfigurator-core/parity_tests/benchmark_engine_step.py`.

Add or reuse a script to benchmark forward-step latency for the Python SDK and Rust core.

The benchmark should produce numbers that show actual speed improvement and can be reused to track Rust core performance over time.

Benchmark requirements:

1. Measure Python SDK per-step latency.
2. Measure Rust core per-step latency.
3. Separate cold-start/setup cost from warm hot-path per-call latency where possible.
4. Use reproducible input parameters.
5. Report p50/p90/p99 per-call latency if practical.
6. Report speedup ratio.
7. Include enough detail that another engineer can rerun the benchmark.

End-state performance target:

```
Rust hot-path engine-step calls are at least 3x faster than the Python SDK hot path.
```

### Phase 3: Rust Implementation

Status: pending. This is the next implementation phase.

Only start full implementation after Phase 0-2 are in place.

Implementation guidance:

1. Work module by module according to the migration map.
2. Match Python SDK observable behavior.
3. Avoid copying deprecated or redundant Python internals.
4. Keep Rust code modular and readable without sacrificing hot-path performance.
5. Keep perf DB loading and expensive setup out of the per-step hot path where possible.
6. Run parity tests and benchmark checks as the implementation progresses.

FPM input schema (decision):

AIC models homogeneous batches. At any single step every request shares the
same `(batch_size, context_length)` shape. Python does not iterate per
request; it issues one table query per step with uniform shape and integrates
across decode steps externally. Aggregate FPM v1 fields such as
`sum_decode_kv_tokens` and `num_decode_requests` therefore recover the exact
per-request value losslessly (`mean = max = min`). No FPM schema bump is
expected during Phase 3 parity work. The schema-richness concern in the
migration map applies to a future Dynamo Mocker integration where a live
scheduler produces heterogeneous distributions; that is out of Phase 3 scope.

If the mixed-step parity smoke shows >1% drift attributable to an aggregate
input collapsing distinct per-request shapes, treat it as a stop condition and
escalate before proceeding.

Pre-Phase 3 cleanup (lands before the commit sequence below):

1. Commit the working-tree clarification in
   `rust/aiconfigurator-core/docs/phase1/migration-map.md` that defines what "target
   Rust path" means during the transition.
2. Update `rust/aiconfigurator-core/docs/phase1/migration-map.md`:
   - Note that Python `perf_database` lazy-load + Pattern A cleanup landed on
     main, so Rust matches Python's current shape (not its future shape).
   - Correct the FPM v1 aggregate-input tradeoff entry per the
     homogeneous-batch reasoning above.
   - Document why Qwen3-VL is excluded from Phase 3 smoke (not in any support
     matrix; `cli_estimate` errors out today).
3. Capture a baseline benchmark snapshot to
   `rust/aiconfigurator-core/parity_tests/benchmarks.md` so Phase 3 progress
   is anchored (final Phase 3 snapshot appended in C12; the ≥3x speedup
   target moved to Phase 5).

Qwen3-VL is intentionally out of Phase 3 scope. The model code exists in
`src/aiconfigurator/sdk/models/qwen3vl.py`, but Qwen3-VL has zero entries in
any support matrix CSV (model code landed after the matrix was last
regenerated), and `cli_estimate(mode="static_ctx", ...)` hard-errors with
`ValueError: x is less than the smallest value in the list. x=0` from
`interpolation.py` because vision-encoder ops query the perf DB with a
zero-valued dimension. Multimodal parity waits for the Python path to be
fixed and the support matrix to advertise Qwen3-VL as PASS; Phase 4 picks it
up.

Phase 3 commit sequence (single PR; commits land in this order):

| # | Commit | What lands |
| --- | --- | --- |
| C1 | scaffolding | Carve `{lib,perf,model,ffi}.rs` into target shape: `{config,enums,system_spec,result,perf_database/,operators/,models/,backends/,session}.rs`. Aggregate formula path keeps working through the new modules. Parity unchanged. |
| C2 | foundation types | `enums`, `system_spec` (YAML parse), `model_config_parser` (HF config + hf_quant_config), `result`. |
| C3 | perf_database modular | Lazy per-op-owner shape mirroring Python Pattern A. `interpolation`, then `perf_database/{gemm,attention,mla,moe,communication}`. Unit tests vs Python on sampled grid within float-epsilon. |
| C4 | operator primitives | `operators/{base,gemm,attention,mla,moe,communication,elementwise,embedding,overlap}`. Per-operator unit tests vs Python. |
| C5 | models + backends infra | `models/{base,registry,config_loader}`, `backends/{base,vllm}`. |
| C6 | model implementations | `models/{hybrid_moe,deepseek}` op graphs. Covers both smoke models. |
| C7 | session + FFI rewire | `session.rs` (static + mixed-step + agg + disagg). Remove aggregate `lib.rs` path; FFI now drives session. |
| C8 | flip MiniMax parity | All 5 surfaces (static_ctx, static_gen, mixed_step, agg, disagg) -> required. |
| C9 | flip Kimi parity | All 5 surfaces -> required. |
| C10 | flip prefix-caching parity | `minimax-m25-sampled-prefix` all surfaces -> required. |
| C12 | docs | Update migration-map "Current State" to reflect Phase 3 reality. Capture final benchmark snapshot. Document any intentional Python divergence. |

Note: the original Phase 3 plan included a `C11 hot-path optimization`
commit conditional on the benchmark missing 3x at the end of C10. After
landing C1-C10 the smoke benchmark shows 1.1-1.6x speedup, below the
target. We split that work out into a dedicated **Phase 5: Hot-Path
Optimization** (below) because the remaining wins require a careful
cache design and Python-side FFI restructuring that drifts noticeably
from the current Python reference; landing it inside the parity PR
would mix concerns and risk regressions.

Phase 3 commit dependencies:

- C1-C3 are strictly sequential.
- C4 operator implementations may be split into sub-commits and reviewed in
  any order, but all must land before C5.
- C5 -> C6 -> C7 are sequential.
- C8-C10 may interleave in any order; each requires C7 plus the operators it
  exercises from C4.
- C12 always lands last.

Phase 3 exit criteria:

1. All parity surfaces (3 smoke case variants x 4 modes, minus any documented
   N/A) within 1% drift versus Python SDK; xfails removed.
2. Existing Python tests still pass.

Performance — the ≥3x speedup target is deferred to **Phase 5** (see
below). Phase 3 ships the correctness foundation and a documented
1.1-1.6x speedup baseline.

### Phase 4: Larger Smoke Set

Status: complete (D1-D8 landed). The original Phase 4 framing was a
full comprehensive scan across every entry in the support matrix; once
the inventory step ran, it was clear most of those rows would just be
"unsupported family" skips against the current Rust implementation (3
of 17 supported architectures map to Rust builders today). The goal
was reshaped to "expand the smoke set with more representative cases"
— still increasing real coverage, but without burning the comprehensive
matrix on cases the Rust crate doesn't model yet. By the end of
Phase 4 the Rust crate has a 1↔1 mapping to every supported Python
model family and op-graph primitive.

Phase 4 deliverables (landed):

- **D1: model-graph fixes uncovered during inventory.** The probe of
  the broader matrix surfaced two real Rust bugs:
  - `use_qk_norm` was driven only by an explicit HF field; Python's
    `utils.py` forces it on for `Qwen3ForCausalLM`,
    `Qwen3MoeForCausalLM`, and `MiniMaxM2ForCausalLM` regardless. The
    Rust `config_loader` now mirrors that architecture-driven override.
  - `models/llama.rs` was missing `context_act_gate` (elementwise
    between the gated FFN GEMMs) and `context_logits_gemm` (bf16 vocab
    projection), and was using `dtypes.gemm_quant` instead of bf16 for
    `generation_logits_gemm`. Added all three plus `low_precision_input`
    on ffn2.
- **D2: smoke set expansion.** The smoke suite grew from 3 cases x 4
  modes = 12 surfaces to 41 cases x 4 modes = 164 surfaces, all
  asserted (no xfails) within the 1% drift tolerance (cumulative
  total through D6/D7/D8):
  - 3 original (MiniMax-M2.5, Kimi-K2.5, sampled-prefix).
  - 3 new MoE-family models (MiniMax-M2.7, Qwen3-30B-A3B, Qwen3-235B-A22B).
  - 3 new dense Llama-family models (Qwen3-32B, Llama-3.1-70B,
    Llama-3.1-8B).
  - 2 cross-system MiniMax-M2.5 runs (h200_sxm, h100_sxm).

Phase 4 D4 / D5 closed three gaps that the earlier write-up
mis-classified as needing structural work. Each was a real apple-to-
apple translation that just hadn't been ported yet:

1. **`Op::Overlap` for shared/routed-expert overlap** (D4).
   Python wraps the DeepSeek-V3 / DeepSeek-R1 generation MoE in
   `OverlapOp("generation_moe_overlap", routed, shared)` so the latency
   is `max(sum(routed), sum(shared))`. Rust gained an `Op::Overlap`
   variant in `operators/op.rs` that mirrors the Python op exactly;
   `models/deepseek.rs` now builds the two groups and pushes a single
   `Op::Overlap` between the MLA module and the logits GEMM. Pre-fix
   `static_gen` drifted +15.89% on DeepSeek-V3 / R1; post-fix it
   passes within 0.02%. Kimi-K2.5 — which previously happened to pass
   because its routed-MoE work swamped the shared FFN — still passes.

2. **`Op::Fallback` for missing MLA-module data** (D5).
   Python uses `FallbackOp(primary=MLAModule, fallback=[GEMM, MlaBmm,
   ContextMla/GenerationMla, MlaBmm, GEMM])` so when a perf DB doesn't
   ship `mla_*_module_perf.txt` (true for SGLang / TRT-LLM today), the
   per-kernel chain takes over. Rust gained `Op::Fallback` and the
   DeepSeek model builder now constructs the same fallback chain
   unconditionally. vLLM keeps using the module-level primary; SGLang
   now exercises the fallback and Kimi-K2.5 passes on
   b200_sxm/sglang/0.5.10 (ctx +0.33%, gen -0.02%) and
   h200_sxm/sglang/0.5.10 (ctx +0.35%, gen -0.06%).

   Also fixed the latent `MlaBmm` dispatch: Python's `MLABmm.query`
   keys the BMM table by `batch_size`, not by `num_tokens`. Rust was
   passing `ctx.num_tokens` (which is `batch * effective_isl` for
   context); now uses `ctx.batch_size`. Hidden by vLLM's primary
   succeeding pre-fix; surfaced once the SGLang fallback exercised it.

3. **SGLang dispatch defaults to CustomAllReduce** (D4).
   Python's `MoEDispatch.query` only uses `wideep_deepep_normal_perf`
   when the caller explicitly sets `moe_backend="deepep_moe"`; the
   AIC `cli_estimate` path doesn't, so it falls through to a
   custom_allreduce path identical to vLLM. Rust was hard-coding
   `DispatchFlavor::DeepEpNormal` for SGLang, which then hit a
   missing perf file. Switched both `models/moe.rs` and
   `models/deepseek.rs` to `CustomAllReduce` for the SGLang default.
   The `DeepEpNormal` flavor is preserved in the enum for when we
   thread `moe_backend="deepep_moe"` through the FFI.

4. **MLA effective head count = `128 // tp_size`.** (D4)
   Python `models/deepseek.py` hard-codes `128 // tp_size` for every
   MLA op regardless of the model's `num_attention_heads`. DeepSeek-
   style MLA always profiles against 128 attention heads. Rust was
   using `num_attention_heads / tp` (so Kimi-K2.5 with 64 heads on
   tp=8 was passing 8 instead of 16; DeepSeek-V3 with 128 heads
   happened to be right). The wrong key hit a different slice of the
   MLA module table; drift grew with system speed and ISL on Kimi
   (~-1.06% on h200/ISL=1024, -1.58% at ISL=2048). Fixed.

Phase 4 D6 / D7 / D8 closed the remaining family-coverage and
op-primitive gaps so the Rust crate now matches Python's full model
family list and op-graph composition primitives:

5. **Per-family builders for the remaining ModelFamily variants** (D6).
   `factory.rs` previously errored out for `Gpt`, `HybridMoe`,
   `DeepSeekV4`, and `Gemma4Moe`. Each got a dedicated builder:
   - `models/gpt.rs` — dense GQA with non-gated FFN (ffn1 + act +
     ffn2). Used by `openai/gpt-oss-20b`.
   - `models/hybrid_moe.rs` — Llama-4 / MiMo-V2-Flash 4-bucket layout
     (Scout's interleaved-attention pattern).
   - `models/deepseek_v4.rs` — DSv4 with compress ratios + mHC
     pre/post projection chain.
   - `models/gemma4_moe.rs` — SWA+global attention with shared-MLP +
     MoE FFN.
   All four are exercised by the smoke harness via the error-symmetry
   contract (Python and Rust both raise on missing perf-DB tables for
   today's smoke shapes, which counts as parity pass).

6. **Multimodal Qwen3VL pair** (D7).
   Ported `models/qwen3vl.py` and the vision-encoder op graph from
   `models/vit_ops.py` to Rust as `models/qwen3vl.rs` +
   `operators/vision.rs`. Covers both `Qwen3Vl` (dense) and
   `Qwen3VlMoe` families with the 10-op transformer-block sequence
   plus projector chain. Phase 4 (not 3) territory because Python's
   `cli_estimate` only became reliable for Qwen3-VL after the
   upstream support matrix regenerated.

7. **WideEP MLA and WideEP MoE compute ops** (D8).
   Closed the last op-primitive gap:
   - `WideEpContextMlaOp` / `WideEpGenerationMlaOp`
     (`operators/wideep_mla.rs` + `perf_database/wideep_mla.rs`):
     separate context (kernel/fmha/kv/heads/s/b) and generation
     (kernel/kv/heads/b/s) table nestings; context op applies the
     same `prefix_correction` as the standard MLA path.
   - `WideEpMoeOp` (`operators/wideep_moe.rs` +
     `perf_database/wideep_moe.rs`): 10-level key with
     distribution-string fallback and `attention_dp_size` scaling.
   These primitives unlock the two WideEP DeepSeek variant builders:
   - `models/deepseek_wideep.rs` — SGLang DeepEP path (selected
     when `WideEpMode::SglangDeepEp` and `moe_backend="deepep_moe"`).
   - `models/deepseek_wideep_trtllm.rs` — TRT-LLM WideEP path with
     `PDL_FACTOR=0.9` for the CUDA-graph generation-latency
     adjustment.
   Routing lives in `factory.rs` under
   `ModelFamily::DeepSeek => match config.wideep_mode { ... }`.

8. **DSv3 prefix-heavy parity fix** (D6).
   `get_mix_step_latency_ms` Pass-1 was passing `prefix=0` to ops,
   which bypassed the MLA module's `prefix_correction` for the
   `context_mla_block` operator wrapped under `FallbackOp`. The
   `is_context_attention()` filter doesn't catch fallback-wrapped MLA
   ops, so combined-prefix tokens weren't being credited. Fixed by
   threading `combined_prefix` through Pass-1 in `session.rs`.
   Closed the last -1.21% drift smoke case.

9. **DSv32 sglang DSA extrapolation** (D6).
   Sparse DSA perf tables for low `num_heads` caused
   axis-out-of-range errors. Added
   `interp_2d_1d_grid_extrapolate_inner` to `interpolation.rs` for
   linear extrapolation on y/z axes and routed both
   `query_context` and `query_generation` in
   `perf_database/dsa.rs` through it. Closed the 4 DSv32-sglang
   failures.

10. **lib.rs cleanup** (D8).
    Removed ~500 lines of legacy aggregate paths and duplicated
    re-exports; net file delta -2425 lines. `Phase3Estimator` now
    `Arc`-wrapped (required by `ForwardPassPerfModel: Clone`) with a
    manual `Debug` impl. All 14 `ModelFamily` variants are closed in
    `factory.rs` — no `Err` arm remains.

Re-running the smoke set:

```bash
AICONFIGURATOR_RUST_CORE_AUTOBUILD=1 uv run pytest \
  rust/aiconfigurator-core/parity_tests/test_engine_step_parity.py -p no:xdist
```

A clean run asserts all 164 surfaces (41 cases x 4 modes). Coverage
spans every supported model family:

- Llama / Qwen3 dense: Qwen3-32B, Llama-3.1-70B, Llama-3.1-8B.
- MoE family: MiniMax-M2.5, MiniMax-M2.7, Qwen3-30B-A3B,
  Qwen3-235B-A22B.
- DeepSeek family: Kimi-K2.5, DeepSeek-V3, DeepSeek-R1.
- DeepSeekV32 family: DeepSeek-V3.2, GLM-5 (DSA attention).
- NemotronNas: Llama-3.3-Nemotron-Super-49B-v1 (per-block config).
- NemotronH: Nemotron-H-56B-Base-8K (hybrid Mamba2 + attention + MLP).
- Qwen35: Qwen3.5-27B (hybrid GDN + full-attention),
  Qwen3.5-397B-A17B (hybrid GDN + MoE).
- Shape variations (D2-bis): decode-heavy, prefill-heavy,
  prefix-heavy, large-batch.
- Error-symmetry families (both Python and Rust error on perf-DB
  miss): meta-llama/Llama-4-Scout (HybridMoe), openai/gpt-oss-20b
  (Gpt), deepseek-ai/DeepSeek-V4-Flash (DeepSeekV4).
- Cross-system: MiniMax-M2.5 on b200_sxm / h200_sxm / h100_sxm;
  Kimi-K2.5 on b200_sxm / h200_sxm / h100_sxm.
- Cross-backend: MiniMax-M2.5 on b200_sxm/sglang/0.5.10;
  Kimi-K2.5 on b200_sxm + h200_sxm on sglang/0.5.10;
  NemotronNas / Qwen3.5 / NemotronH on sglang/0.5.10 +
  trtllm/1.3.0rc10.

The error-symmetry contract introduced in D7-A treats "both engines
raise" as parity pass, so families whose perf-DB tables are missing
for the smoke shape still get smoke coverage — the test asserts that
Rust mirrors Python's behavior, not just its successful outputs.

Still deferred:

- **Comprehensive matrix scan.** Now that families on
  vLLM/SGLang/TRT-LLM share a single Rust op graph and every
  Python family maps 1↔1 to a Rust builder, the original Phase 4
  vision (every matrix entry, parallelized, per-row drift CSV) is
  attractive but still gated on Phase 5 hot-path work so a full scan
  finishes in a reasonable time. Re-open as Phase 4-bis once
  Phase 5 lands.

### Phase 5: Hot-Path Optimization

Status: pending. Activated after the Phase 3 parity work merges. Goal:
get smoke-case p50 speedup from the Phase 3 1.1-1.6x baseline to >=3x.

Phase 3 profiling (recorded in `parity_tests/benchmarks.md`) attributes
the remaining cost to several pieces, none of which can be tackled
without measurable code drift from the Python reference:

- **Python wrapper rebuilds the engine-config JSON per call** (~5.6us)
  because the `@cache`d Rust estimator is keyed by that JSON string.
  Fixing it requires an identity-keyed (or otherwise model+db-keyed)
  cache that keeps the Rust handle alive across calls.
- **Per-call `copy.deepcopy` + `json.dumps`** of FPM metrics adds
  ~3.5us. Avoiding it requires a packed-primitive FFI entry-point on
  the Rust side and a fast-path detector on the Python side.
- **Rust op-graph allocations / dispatch** dominates the remaining
  ~4.7us hot-path budget. Reducing it without sacrificing the
  apple-to-apple op-list structure requires a runtime sub-table cache
  (memoized table-resolution per op-shape) plus interned distribution
  strings.

Each of these pulls the implementation away from the Python reference
shape in non-trivial ways: the FFI gains a fast/slow path bifurcation,
the table layout grows a runtime cache (unsafe pointers or interior
mutability), and the Python side grows an identity-keyed estimator
context. Landing those changes safely needs:

1. A documented design for the cache structures (key shape, invalidation
   rules, eviction policy) so future maintainers can reason about them
   without re-deriving correctness.
2. A separate optimization-only PR so any speedup-induced regression is
   bisectable against the parity-only PR.
3. Re-running the Phase 3 smoke parity assert suite after each
   structural change, plus the eventual Phase 4 comprehensive scan.

Exit criterion: benchmark p50 speedup >= 3x for every smoke case while
Phase 3's parity assertions remain green.

## Dependencies / Relationships

- Phase 0 defines the migration/design map.
- Phase 1 provides correctness guardrails.
- Phase 2 provides performance guardrails.
- Phase 3 uses the migration map, parity tests, and benchmark harness to implement the Rust migration safely.
- Phase 4 expands the smoke set with realistic per-family / cross-system
  coverage and closes every Python↔Rust model-family + op-primitive gap
  identified by the inventory step. The remaining work — a full
  comprehensive matrix scan — is gated on Phase 5 hot-path performance
  and re-opens as Phase 4-bis afterwards.
- Phase 5 optimizes the hot path after the parity foundation lands.

Do not treat the phases as isolated checklist items. If work in one phase reveals missing requirements in another, update the plan and surface the change.

## Constraints / Non-goals

- Do not remove the Python SDK in this project. Python SDK deprecation/removal belongs to a separate project.
- Do not move AIC CLI, collectors, config generators, or Pareto analysis into Rust.
- Do not refactor unrelated Python SDK internals.
- Do not blindly port deprecated Python code.
- Do not add speculative Rust abstractions or future features.
- Do not optimize for cleanliness at the cost of hot-path latency.
- Do not optimize for latency in a way that makes the code impossible to understand or maintain.
- Existing Python tests should continue to pass.
- The Rust implementation should become the source of truth for AIC core engine-step latency logic, while Python remains the orchestration and user-facing layer.

## Acceptance Criteria for Current Iteration

The current iteration is complete when:

1. A migration map exists and covers the relevant Python SDK files/modules.
2. A proposed Rust module structure is documented.
3. A system diagram shows current flow and target flow.
4. Parity smoke tests exist for selected model/system/backend/input cases.
5. Known current Rust mismatches are marked as `xfail` or reported explicitly with reasons.
6. A benchmark script exists for Python SDK vs Rust per-step latency.
7. Benchmark output includes reproducible parameters and speedup numbers.
8. Current decisions, remaining open questions, tradeoffs, and stop conditions are documented.
9. Full Rust migration work does not proceed until the map, parity harness, and benchmark harness are ready.

## Final Project Acceptance Criteria

The full project is successful when:

1. Rust and Python SDK metrics differ by <1% on the agreed parity suite.
2. Rust hot-path engine-step calls are at least 3x faster than the Python SDK hot path.
3. Existing Python tests pass.
4. Python CLI, collectors, config generators, and Pareto analysis remain Python-owned.
5. Rust owns the core engine-step latency implementation.
6. A comprehensive parity scan has run at least once across the supported AIC search space.
7. Any intentional divergence from Python behavior is documented and approved.

## Test Strategy

Before implementing Rust logic, derive a test matrix that includes:

1. Happy paths.
2. Boundary cases.
3. Invalid inputs.
4. Unsupported model/system/backend/version tuples.
5. Deprecated or ambiguous Python behavior.
6. Perf DB lookup and missing-data cases.
7. Cold-start vs warm hot-path performance cases.
8. Regression cases for existing Python behavior.
9. A comprehensive, shardable full-scan suite for all AIC-supported entries,
   run after feature completeness rather than as the fast smoke loop.

Prefer tests through public interfaces over tests of private implementation details.

If a test case depends on an ambiguous API or product decision, list it as an open question instead of guessing.

## Stop Conditions

Stop and ask before proceeding if:

1. Python behavior and desired Rust behavior conflict.
2. A required Python integration change violates the `./rust` primary implementation boundary.
3. A parity case depends on deprecated or redundant Python behavior.
4. The benchmark result depends heavily on cold startup or perf DB loading rather than hot-path calls.
5. The existing POC architecture conflicts with the target modular design.
6. A required model/system/backend/version tuple is missing from the perf DB.
7. The test harness cannot compare Python and Rust metrics through stable public interfaces.
8. Achieving <1% parity appears incompatible with the intended Rust architecture.
9. Achieving 3x speedup appears incompatible with parity or readability (the
   Phase 3 outcome: surfaced and resolved by moving the 3x target to a
   dedicated Phase 5).
10. A task requires changing unrelated Python SDK internals.

## Expected Final Summary

At the end, summarize:

1. What changed.
2. What migration map was created.
3. What parity tests were added and which are expected to fail for now.
4. What benchmark script was added or reused.
5. What checks were run.
6. What remains ambiguous.
7. What tradeoffs were surfaced.
8. What the next implementation step should be.
