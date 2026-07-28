# Python ↔ Rust Parity Audit — 2026-07-14

Full-repo audit of the Rust core (`rust/aiconfigurator-core`) against the Python
SDK, in preparation for full migration. Baseline: `origin/main` @ `2feebac53`.

> **STATUS UPDATE (same day, this branch): ALL items below are FIXED** except
> the intentionally-excluded HYBRID/EMPIRICAL port (#1333) and the by-design
> Python-only surfaces (AFD, vision-encoder phase, power/energy, `.txt` legacy
> loading, scheduling/summary orchestration). Every P0/P1 fix landed with a
> pinning test; new end-to-end parity coverage: GLM-5.2 skip-indexer smoke
> case, sglang WideEP/DeepEP suite (h200/0.5.6.post2), trtllm WideEP alltoall
> suite (gb200), chunked-prefill mixed-step cases, imbalance-scale threading
> cases — all bit-identical or within the 1% gate. The mixed/decode-step
> composition was rewritten as a literal mirror of Python's three-pass
> `_get_mix_step_latency` (bit-identical on chunked/prefix/MTP shapes).
> `cargo test --lib` (196 tests, incl. regenerated oracles) is now a CI gate.
> `OpConversionError` now falls back to the Python step instead of crashing;
> the engine-handle cache key carries the un-collapsed quant + ModelConfig
> identity. See the fix-order list at the bottom — items 1-12 all landed.

**Scope exclusion:** HYBRID / EMPIRICAL / `util_empirical` differences are out of
scope (Rust is SILICON-only by design; delegation gate at
`sdk/rust_engine_step.py:224-246`; port tracked in issue #1333 with a separate
PR in flight).

## Verdict

**Not yet answer-identical.** The dense-model core (attention, MLA, GEMM
interpolation, standard MoE, communication, mamba/GDN, DSA full-layer, DSv4
context) is a faithful, oracle-verified port, and the June 2026 full-matrix
parity gate (2,158 entries, 0 regression) still holds for that surface. But the
audit found **6 live high-severity numeric divergences** (concentrated in
WideEP / trtllm-alltoall / DSv4-generation / mixed-step composition), a set of
narrow-trigger divergences, and several whole features with no Rust
counterpart. Deleting the Python query layer today would regress WideEP
DeepSeek, GLM-5.2, DSv4 fp8-decode, GB200-NVL72 comm, and chunked-prefill agg
sweeps.

Items 1, 3, 4, 6 below were independently found by two audit passes (operators
+ DB layers) and hand-verified in source.

---

## P0 — live numeric divergences (fix before any default flip)

### 1. WideEP MLA: bridge writes `tp_size` into the `num_heads` table axis  ✅ hand-verified
- Python queries at `num_heads = 128 // tp_size` (`sdk/operations/mla.py:1195` gen, `:1464` ctx).
- The spec emitter writes raw tp: `"num_heads": op._tp_size` (`sdk/engine.py:463,475` — the comment claims "per-rank head split" but the value is tp).
- Rust passes it straight to the heads axis (`operators/wideep_mla.rs:85,226` → `perf_database/wideep_mla.rs:163-170,214-218`). tp=1 ⇒ Python reads the heads=128 slice, Rust queries heads=1 ⇒ out-of-grid SOL-rescale, order-of-magnitude error.
- Rust unit tests mask it by constructing ops with `num_heads=128` directly (`operators/wideep_mla.rs:121-131`).
- Fix: one line in `engine.py` (`128 // op._tp_size`) + a parity smoke case with tp>1 WideEP.

### 2. trtllm `enable_alltoall` dispatch: loader collapses the table; query errors on shipped data
- Python keys `[kernel_source][op_name][quant][num_nodes][...]` and queries `alltoall_dispatch` + `alltoall_combine` separately (`sdk/operations/moe.py:1128-1172, 2921-2927, 1976-2042`).
- Rust loader drops `kernel_source`/`op_name`/`num_nodes` (`perf_database/wideep.rs:637-693`) — verified **1,556 / 2,096 rows in gb200 `trtllm_alltoall_perf.parquet` collide** with differing latencies (first-wins).
- Rust queries distribution `"uniform"` (`operators/moe_dispatch.rs:370-378`) but shipped data is `"balanced"` ⇒ hard error.
- Missing `fp8_block → fp8` normalization (`moe.py:1963-1973`).

### 3. DSv4 generation: PR #1337 (c4a574a65) ported to `mla.rs` but not `dsv4.rs`
- Python derives generation SOL fmha from KV dtype (`sdk/operations/dsv4.py:1246-1254`); Rust uses the raw label (`operators/dsv4.rs:414-424`, `perf_database/dsv4.rs:368-381`).
- Python's generation table key ignores `mla_dtype` (`dsv4.py:1938-1966`); Rust requires exact fmha+arch match (`perf_database/dsv4.rs:855-880`) ⇒ hard error on fp8-labeled decode where Python resolves.
- The Rust oracle fixtures (`perf_database/dsv4.rs:1237-1284`) were generated pre-#1337 and now certify parity against stale Python. Regenerate after the fix.

### 4. `wideep_moe_perf` dedup: first-wins vs Python's last-wins
- Python direct-assigns (last row wins, `moe.py:2819-2825`); Rust `or_insert` (first wins) with an incorrect "parity" comment (`perf_database/wideep_moe.rs:258-264`).
- Verified **270 duplicate keys with differing latencies** in rtx_pro_6000_server trtllm 1.3.0rc10.
- Same bug class was already fixed for MLA modules (`mla.rs:653-674`); wideep was missed. Fix: `or_insert` → `insert`.

### 5. DSA skip-indexer (GLM-5.2) amortization missing
- Python prices layers as `w*full + (1-w)*skip` (`sdk/operations/dsa.py:774-781, 1412-1420`, CP `:754-759`); Rust drops skip rows at load (`perf_database/dsa.rs:774-779`) and has no `full_frac` field (`operators/dsa.rs:24-44`); the emitter neither sends nor rejects (`engine.py:337-360`).
- Triggered by `model_configs/nvidia--GLM-5.2-NVFP4_config.json` (`index_topk_freq: 4`) on sglang ⇒ **silent** overestimate. DeepSeek-V3.2 / GLM-5 (full_frac=1) unaffected.

### 6. Sequence-imbalance correction scales silently ignored in Rust
- Python threads `seq_imbalance_correction_scale` / `gen_seq_imbalance_correction_scale` into every attention query (`backends/base_backend.py:331,372,950-965`).
- Rust accepts them on the wire but hardcodes `1.0` (`session.rs:45-46,74-75,146-147,178-179,203-204`; admitted at `engine/runtime.rs:33-35`); the facade passes them through implying support (`rust_engine_step.py:275-276`).
- Latent today (no shipped task sets ≠1.0) but ungated: either thread them through `RuntimeContext` or refuse rust routing when ≠1.0.

### 7. Mixed-step composition (agg / chunked prefill) — three formula gaps
Largest first:
- **Pass-2 context attention under chunked prefill** (`ctx_tokens < isl`): Python queries at full per-request isl then divides by `ceil(isl/ctx_tokens)` (`base_backend.py:984-1003`); Rust queries at `ctx_tokens` with scale 1 (`session.rs:153-183`, self-documented limitation at `:158-164`). Attention is superlinear in s.
- **Pass-1 prefix/nextn accounting**: Python `prefix * floor(ctx_tokens/isl)`, raw combined tokens (`base_backend.py:954-964`); Rust `prefix * ceil(...)` (`runtime.rs:348-349`) and inflates pass-1 tokens by `(nextn+1)` (`runtime.rs:362`, `session.rs:129`).
- **±1 kv-token convention**: Python decode queries at `s = isl + osl//2 + 1` (`base_backend.py:371,1075-1085`); Rust at `isl + osl/2` (`runtime.rs:403`, deliberate legacy-FPM convention). Sub-tolerance; either align or codify.

### 8. sglang MoE routing gaps in the bridge
- `deepep_moe` backend: Python routes MoE compute to wideep tables (`moe.py:685-691`); Rust `MoeOp` always reads `moe_perf` — `WideEpTable::query_{context,generation}_moe` exist but are caller-less (`wideep.rs:179-238`).
- EPLB prefill correction `int(num_tokens*0.8)` (`moe.py:684`) dropped by the emitter (`engine.py:241-255`).
- sglang WideEP dispatch flavor: emitter maps everything non-trtllm to CustomAllReduce (`engine.py:258-274`, documented limitation) — DeepEP normal/low-latency never emitted ⇒ **silently wrong** rather than rejected.
- MoE mxfp4 kernel→quant remap unported: Python reroutes `w4a8_mxfp4_mxfp8` / `w4a16_mxfp4` to `_trtllm` / `_cutlass` variants at load (`moe.py:2465-2474`); those quant names don't exist in Rust (`enums.rs:120-129`). Affects shipped sglang 0.5.14 data on 7 systems.

---

## P1 — narrow-trigger / dormant divergences

| # | Area | Python | Rust | Trigger |
|---|---|---|---|---|
| 9 | GEMM fp8_static residual floor | clamp to GEMM SOL (`gemm.py:787-800`) | clamp to 0 (`operators/gemm.rs:92-94`, stale comment) | fp8_static when subtraction dips below SOL |
| 10 | GB200/NVL72 comm reroute | custom-AR → NCCL when `num_gpus_per_node==72 && tp>4` (`communication.py:187-190`; same in `moe.py:1148-1155`) | always custom-AR (`operators/communication.rs:60-87`, `moe_dispatch.rs:396-406`); also missing `sm_version==100` gate (`moe.py:1099`) | GB200/GB300 NVL72 (Rust errors or mis-prices) |
| 11 | ElementWise `scale_num_tokens` | floor tokens then split (`elementwise.py:56-59`) | scale folded into bytes (`engine.py:153-154`, `elementwise.rs:47-48`) | exact only when scale divides tokens; live on DSv3.2 CP add_norm |
| 12 | NCCL fractional message size | float element counts (`models/gemma4.py:374`) | truncates to u64 (`communication.rs:138`) | gemma4 CP KV all-gather |
| 13 | Encoder partial-RoPE extra | `factor * 2 * mem_op(...) * 1.1` (`attention.py:1100-1107`) | field absent (`operators/attention.rs:234-296`) | Qwen3-VL-class (moot while encoder stays Python-side) |
| 14 | WideEP MoE beyond-range hold | num_slots-aware roofline (`moe.py:1782-1797`) | linear token proxy (`wideep_moe.rs:103-109,196`) | out-of-grid tokens only |
| 15 | MLA BMM quant fallback granularity | top-level (`mla.py:564`) | per-(quant,op) (`mla.rs:242-248`) — Rust answers where Python raises | partial-data dirs |
| 16 | Interp exact-distance tie-break | insertion order (`perf_interp/engine.py:278,386`) | sorted keys (`perf_interp.rs:424-432,573-580`) | exact ties only |
| 17 | beam_width | applied to gen batch (`base_backend.py:368-369`) | never applied (`runtime.rs:39-41`), ungated | beam>1 (unexercised) |

Known-and-accepted (from the June gate, still present): GEMM large-n
extrapolation ~30% on the logits op; context-attention ragged-grid ~1.5%.

---

## Unported features (no Rust counterpart)

**Loud (raise `OpConversionError` at spec build — but note this CRASHES the
sweep instead of falling back; `base_backend.py:422` has no try/except):**
- MiniMax-M3: `Context/GenerationMSAModule` (`operations/msa.py:258,302`). In practice usually blocked earlier by the SILICON gate (MSA data is HYBRID).
- DeepSeek-V4 MegaMoE (sglang): `DeepSeekV4MegaMoEModule` (`dsv4.py:1417`); `query_dsv4_megamoe_module` (`perf_database.py:2761`).
- TRT-LLM WideEP DeepSeek / DSv3.2: `TrtLLMWideEPMoEDispatch` (`moe.py:1872`, used by `models/deepseek.py:772-1009`, `models/deepseek_v32.py:573-692`). The paired compute op converts, the dispatch does not — cheap to add, Rust dispatch machinery exists.

**Python-only by design (must stay or be ported later):**
- AFD estimation end-to-end (`inference_session.py:937-1080`, `afd_transfer.py`, `afd_partition.py`, `AFDConfig`).
- Vision encoder phase (`base_backend.py:251-301`; spec excludes encoder ops `engine.py:807-818`; isl-folding compensates for the decoder).
- `query_gemm(database_mode=SOL/SOL_FULL)` — no `DatabaseMode` in the crate (root cause of #9).
- Power/energy everywhere (Rust is latency-only, `perf_interp.rs:23-24`; rust-routed runs return zero energy and a collapsed single-key breakdown — `e2e_power_avg` and per-op detail silently degrade).
- Support-matrix / feasibility / version discovery (`_LazySupportMatrix`, `get_supported_databases`, `get_latest_database_version` incl. `INCOMPLETE.txt` rejection, `get_database_view`) — Rust takes explicit tuples + Python-precomputed `perf_db_sources`.
- `.txt` legacy data loading: Rust is parquet-only (`parquet_loader.rs:42` vs `operations/base.py:62-70`) — txt-only system dirs (e.g. `local_systems/`, new a100 drops) answer in Python, error in Rust.
- Scheduling/summary math around the steps (`run_agg` composition, TTFT/TPOT/throughput, memory/OOM) — by design, "Python builds & orchestrates".

---

## Infra / cross-cutting risks

1. **Engine-handle cache collisions**: `_ENGINE_HANDLE_CACHE` keys on `_engine_config_json` (`rust_engine_step.py:399-432`) but the op payload comes from the actual model. Collapsed dtypes (`sq`→`int8_wo`, `fp8_ootb`→`fp8`, four 4-bit modes→`int4_wo`, two DSv4 MoE modes→`None`) and ~10 identity-omitted `ModelConfig` fields (`comm_quant_mode`, `cp_style`, `moe_backend`, `attention_backend`, `enable_wideep`, `enable_eplb`, `workload_distribution`, `overwrite_num_layers`, `sms`, `wideep_num_slots`) can alias two different models to one cached handle ⇒ silent wrong latencies.
2. **`comm_quant_mode` hardcoded `"half"`** in the emitter (`engine.py:294,310,321`) while Python NCCL queries use the real dtype (`communication.py:385-431`). Rust `CommQuantMode` only has `Half` (`enums.rs:203-205`, stale comment — Python has int8/fp8 at `common.py:1104-1106`).
3. **Silent shared-layer drop**: `_compute_perf_db_sources` swallows all exceptions → `{}` (`engine.py:635-636`); Rust then loads primary-only rows while Python uses the shared layer in the same run. No error surfaced.
4. **Enum drift**: Rust missing MoE `w4a8_mxfp4_mxfp8_trtllm` / `w4a16_mxfp4_cutlass`; `ModelFamily` has `GEMMA4MOE` vs Python `GEMMA4MIX` and lacks `MINIMAXM3`; `PerfDataFilename` still `.txt` (dead code, but its test claims Python parity).
5. **Stale Rust unit tests not in CI**: `parse_b200_sxm` asserts mem_bw 8.0 TB/s but PR #1246 corrected the YAML to 7.7 ⇒ the crate's "match Python" tests currently fail and nothing notices. Add `cargo test` to the gating CI.
6. **system_spec strictness**: Rust serde defaults (`mem_bw_empirical_scaling_factor`=1.0 etc., `system_spec.rs:36-39,77`) where Python KeyErrors — a malformed YAML parses silently with different physics.
7. **`overlap_composition`** (`operators/overlap.rs:20-40`) implements the opposite composition of Python's OverlapOp and has no caller — delete or fix before anyone wires it. (The production `op.rs:291-321` OverlapOp is correct.)
8. **Rust-only leniency**: missing `attn_backend` whitelist in WideEP MLA (`mla.py:1459-1460` raises, Rust succeeds); beam>1 accepted silently.

---

## What is already solid

- Parity CI gates: 40-case engine-step smoke + compile-engine suite + perf gate, every PR (`.github/workflows/build-test.yml:204-258`), 1% rtol, error-symmetry contract.
- Full-matrix scan (`tools/support_matrix/scan_rust_parity.py`, 2,158 entries, probe + pareto layers): gate CLOSED 2026-06-16 with 0 regression, max drift 0.41% ttft / 0.18% tpot (`docs/parity-scan-report.md`).
- Memory: single implementation — Rust `memory.rs` is a pure forwarder into Python (`fetch_python_estimate`); only shared constant is the 0.80 naive KV reservation.
- Interpolation core: step-for-step port, no scipy left; most table layers carry 1e-9 cross-language oracles.
- Static (`run_static`) composition: stride quadrature, `(nextn+1)` scaling, prefix handling formula-identical.

Note the tension: the smoke matrix + June scan passed while P0 items 1–5 exist,
because the affected configs (WideEP tp>1, trtllm alltoall on gb200, GLM-5.2
skip-indexer, post-#1337 DSv4 fp8 decode, deep chunked-prefill shapes) are
outside or under-sampled in the scan matrix, and Rust-side oracles were
constructed to match (heads=128, pre-#1337 fixtures). **Every P0 fix should add
its trigger config to SMOKE_CASES / the scan matrix.**

## Migration gates (from `docs/python-dedup-plan.md`, status unchanged)

- Gate 1 (default flip): blocked by #1333 (SILICON-only) **and now by P0 1–8 above**.
- Gate 2 (golden oracle capture): NOT STARTED — must land before any Python deletion, or the regression detector disappears.
- Gate 3 (delete Python latency code): blocked on 1+2.

## Suggested fix order

1. `engine.py:463,475` → `128 // op._tp_size` (one line) + tp>1 WideEP smoke case.
2. `wideep_moe.rs` `or_insert`→`insert` (one line) + dup-key regression test.
3. Port #1337 into `dsv4.rs` + regenerate DSV4 oracles.
4. trtllm alltoall loader keys (`op_name`/`kernel_source`/`num_nodes`) + distribution + fp8_block normalization.
5. Imbalance scales: thread through `session.rs` or gate in `should_use_rust_engine_step`.
6. Mixed-step pass-1/pass-2 accounting (or codify as accepted tolerance with a test pinning the delta).
7. DSA `full_frac` port or emit-time rejection for GLM-5.2.
8. sglang deepep/EPLB/mxfp4 + WideEP dispatch flavor: port or reject at emit (no silent mis-model).
9. `TrtLLMWideEPMoEDispatch` opspec conversion.
10. Wrap `build_engine_spec_json` call site (`base_backend.py:422`) so `OpConversionError` falls back to Python instead of crashing.
11. Widen `_engine_config_json` identity (add collapsed-dtype + missing ModelConfig fields) to kill cache aliasing.
12. Add `cargo test` to gating CI; fix `parse_b200_sxm`; update stale enum mirrors.
