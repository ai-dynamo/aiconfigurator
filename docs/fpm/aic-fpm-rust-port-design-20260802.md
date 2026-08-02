# FPM Rust port — design (branch `fpm-modeling-rust`, 2026-08-02)

Port of the Python `forward_model="fpm"` whole-model performance model into the
Rust core, per the approved plan (`aic-fpm-modeling-plan.md` §M2/§M3). This
document fixes the change surface and the semantics contract before any code.

## Acceptance criteria (user-set, hard)

1. **Reuse collector output as-is.** The Rust loader reads the exact
   `fpm_forward_perf.parquet` + `fpm_forward_perf.metadata.json` pair the
   collector's `write_formal_database` publishes today (schema
   `aic_fpm_forward_perf` v5, `iteration_totals_balanced_v1`). No format,
   filename, or data-root changes; no re-collection.
2. **Numerical parity with Python.** Same interpolation, clamp, domain-gate,
   and error semantics. Validated three ways: (a) collected grid points on the
   real merged M2.7 database resolve bit-exact (exact-hit path returns the leaf
   verbatim in both languages); (b) off-grid probes within the domain agree
   within the live-parity tolerance `PARITY_RTOL = 0.01` (target and expected:
   ≪ 0.01 — the algorithms are identical; report the observed max); (c) error
   symmetry — where Python raises, Rust errors, and vice versa.
3. **Native Rust interfaces.** Follow the crate's existing structural patterns:
   a `perf_database/` table struct with `OnceLock` lazy load + `PerfReader`,
   an `operators/` op struct + `Op` enum variant, the `EngineSpec` compile
   path, `AicError` → `PyValueError` at the PyO3 boundary. No new frameworks.

## Naming (collision hazard)

The crate already owns `src/fpm/` = `ForwardPassPerfModel` (PR #1152 online
tuning over Dynamo **ForwardPassMetrics** telemetry) — an unrelated concept
that also abbreviates "FPM". The port never touches that namespace:

- module `perf_database/fpm_forward.rs`, type `FpmForwardTable`
- module `operators/fpm_forward.rs`, type `FpmForwardOp`, variant
  `Op::FpmForward`
- mirrors the parquet basename, unambiguous by path; docs must keep the two
  FPMs distinct.

## Change surface

### 1. `perf_database/fpm_forward.rs` — loader (new file)

`FpmForwardTable { data_root, version, cells: OnceLock<Result<Loaded, AicError>> }`

- **Path**: `data_root.join("fpm_forward_perf.parquet")`; sidecar =
  same stem + `.metadata.json`. Deliberately **no** `resolve_op_sources` /
  shared-layer inheritance — FPM data is valid only for its exact
  backend/version (mirrors `fpm_forward.py:455`). Needs `version` to enforce
  the per-row `backend_version == version-dir` check.
- **Absent parquet** = soft "not collected": loads as `NotCollected`, queries
  error `AicError::PerfDatabase("File does not exist at <path> ...")`
  (Python `LoadedOpData.raise_if_not_loaded`).
- **Sidecar validation** (all hard errors, Python's uniform-ValueError
  contract): JSON dict; `schema_name == "aic_fpm_forward_perf"`;
  `schema_version == 5`; `coordinate_system == "iteration_totals_balanced_v1"`;
  `parquet_sha256` == actual file digest (new dependency: `sha2`, RustCrypto,
  MIT/Apache-2.0 — cargo-deny clean); `row_count` == actual rows; > 0 rows.
- **Row validation** (identical to `_validate_row`): `workload_kind ∈
  {prefill, decode}`; `partition_policy == "balanced_v1"`; latency finite and
  > 0; `batch_size ≥ 1`, `total_prefill_tokens ≥ 0`, `total_kv_read_tokens ≥ 0`;
  prefill needs `total_prefill ≥ 1`; decode needs `total_prefill == 0`.
  Duplicate detection over the 24-column normalized physical row key.
- **Cells**: `BTreeMap<CellKey, Cell>`; `CellKey` = normalized strings
  `(model_path, backend_axis, backend_policy, gemm/moe/fmha/comm quant,
  kv_cache_dtype, tp, pp, dp, moe_tp, moe_ep, cp)`. Normalization mirrors
  `_norm_identity`: null → `""`, else string form (Python Enum→`.name` happens
  on the producer side, see §3). `Cell` = prefill `Node`
  (`[batch][total_prefill][total_kv] → latency`), decode `Node`
  (`[batch][total_kv] → latency`), per-phase per-axis `(min, max)` domains,
  `cell_ids`, plus prebuilt `SiteIndex` per phase (crate pattern: build once at
  load, no query-time cache).
- **Wiring**: `pub mod` + `pub use` in `perf_database/mod.rs`, field
  `pub fpm_forward: FpmForwardTable` on `PerfDatabase`, constructed in
  `load_with_sources` (no I/O at construction).

### 2. `perf_interp.rs` — one engine flag (shared-engine change)

Rust `Resolver::ScatteredSites` lacks Python's `own_curve_coverage_fallback`
(`config.py:147`, `engine.py:261-283`): a **collected** site whose own curve
does not cover the query defers to neighbor transfer with the own site
**excluded**, instead of util-holding its own tail. Add the field + the
exclusion logic in `SiteIndex::resolve`, exactly as Python. Default `false`;
the GEMM constructor sets `false` → existing behavior byte-for-byte
(guarded by existing perf_interp tests + a new true/false pair).

### 3. `operators/fpm_forward.rs` + `Op::FpmForward` — the op (new file + enum)

Serde struct (externally tagged on the wire, exactly what `_to_opspec` emits):

```
FpmForwardOp {
  name: String,               // "fpm_forward_prefill" | "fpm_forward_decode"
  phase: Phase,               // Prefill | Decode
  model_path: String,
  match_identity: [String; 11],  // computed by the PYTHON producer via
                                 // _norm_identity — Rust only compares strings,
                                 // no re-normalization drift
  weight_bytes: f64,
  sol_ops: Vec<Op>,           // the ORIGINAL granular op list (recursive,
                              // same precedent as OverlapOp/FallbackOp)
}
```

- **Query semantics** (mirror `FPMForwardOp.query`): read `batch_size`, `s`,
  `prefix` (prefill), `beam_width` from `RuntimeContext`; `batch < 1 || s < 1`
  → error; `beam_width != 1` → error; prefill coords `(B, B·s, B·prefix)`,
  decode coords `(B, B·s)`; cell selection (identity match + `backend_axis ==
  "baseline"`, exact-`model_path` preference, ambiguity errors in Python's
  order); **inclusive bounding-box domain gate BEFORE interpolation** (FPM
  never extrapolates); interp via the shared engine with Python's configs
  (prefill sites `(batch, kv)` / curve `total_prefill`; decode sites `(batch,)`
  / curve `kv`; `own_curve_coverage_fallback = true`, `max_site_distance = 2.0`,
  `nn_sites = 4`, `require_curve_coverage = true`, `k_tail = 3`, RAW); result
  must be finite and > 0. Energy 0.0, source Silicon.
- **`query_pass_baseline(batch)`** (decode only): kv_floor =
  `max(batch, decode-domain KV min)`, resolve `(B, kv_floor)` through the same
  path.
- **SOL roofline**: `sol_fn(coords)` = Σ over `sol_ops` of the op's
  **SOL-mode** latency, with Python's exact coordinate back-mapping — prefill
  `s = max(total_prefill/batch, 1.0)` (float), `prefix = total_kv/batch`
  (unclamped float), `x = batch` for ops whose name contains `"logits_gemm"`
  else `total_prefill`; decode `s = max(total_kv/batch, 1.0)`, `x = batch`.
  This requires a new **SOL query mode** on the op dispatch
  (`Op::query_sol(db, ctx)`), implemented for exactly the op families reachable
  from supported FPM models (Gemm, Embedding, Elementwise, Context/Generation
  Attention, Moe, MoeDispatch, CustomAllReduce, Nccl, P2P, Overlap/Fallback
  pass-through); each mirrors the Python `DatabaseMode.SOL` formula for that
  family, pinned by Python-oracle golden tests at 1e-9 rel. Unsupported
  variants return a hard error (never silently 0). Note fractional `s`/`prefix`
  must be supported by these SOL paths (they are analytic — no table lookups).
- **Enum wiring**: variant in `Op` (`op.rs:92`), arms in `Op::name()`/
  `Op::query()`, entry in the `all_op_variants()` bincode round-trip guard
  (`spec.rs`), plus `Op::query_sol` dispatch.
- **MTP guard**: `Engine::build` rejects an FPM spec with `nextn > 0`
  (`InvalidEngineConfig`) — Python's rewrite already rejects it (ad93e75f);
  enforce, don't half-mirror.

### 4. `engine/runtime.rs` — FPM step branch (mirror of `_get_fpm_mix_step_latency`)

`Engine::is_fpm()` = both op lists are exactly `[Op::FpmForward(...)]` with
phases prefill/decode. Whole-model ops must never reach the name-filtered
3-pass `get_mix_step_ops` (they'd ride pass 1 with the wrong shape), so:

- `mixed_step_latency`: if FPM → marginal-decode composition, Python formulas
  verbatim: prefill component `run_static(static_ctx, batch =
  ceil(ctx_tokens/isl), isl, osl=1, prefix)` divided by `chunk_scale =
  ceil(isl/ctx_tokens)`; decode component `run_static(static_gen, batch =
  gen_tokens, isl' = isl + osl/2, osl = 2)` → samples decode at
  `s = isl + osl/2 + 1` (**the Python `+1` convention, NOT the existing Rust
  step convention `isl + osl/2`** — documented divergence, FPM follows Python);
  `baseline = query_pass_baseline(gen_tokens·(nextn+1))` subtracted, clamped at
  0, **only when `ctx_tokens > 0`**; gen-only keeps full decode; both-zero → 0.
- `decode_step_latency`: if FPM → full decode at the Python convention
  (`batch = gen_tokens·(nextn+1)`, `s = isl + osl/2 + 1`), no baseline.
- `rank_latency_ms` (ForwardPassMetrics dispatch): FPM branch — prefill-only →
  context op at the rank's totals; decode-only → decode op; mixed → the same
  marginal composition. This is what makes `RustForwardPassPerfModel` (the
  other FPM) able to wrap an fpm-mode engine, per plan §6.
- Op-level mode: **zero behavioral change** (guarded by the existing 189-test
  suite + parity CI).

### 5. Python producer — `engine.py`

- `_to_opspec` branch for `FPMForwardOp` emitting `{"FpmForward": {...}}` with
  `match_identity` (the op's existing `_match_identity` 11-tuple), `phase`,
  `model_path`, `weight_bytes`, and `sol_ops` converted recursively through the
  existing granular converters.
- `ENGINE_SPEC_SCHEMA_VERSION` **2 → 3** in lockstep (`config.rs:24`,
  `engine.py`); old specs fail via the existing `UnsupportedSchemaVersion`
  gate. One commit covers both sides — no skew window.

### 6. `base_backend.py` — lift the force-python gates (Python side of M3)

- Drop `model.forward_model != "fpm"` at `:424` and `:1159`; FPM then routes
  through `should_use_rust_engine_step` like op_level.
- `_get_mix_step_latency:935`: FPM branch now honors the Rust route
  (`estimate_mixed_step_latency_with_rust` — the Rust engine carries the
  marginal composition internally); the Python `_get_fpm_mix_step_latency`
  stays as the python-backend path (unchanged, still the reference).
- run_agg's `engine_step_backend_key` (`:1249`) becomes truthful for FPM —
  parity/cache tooling that compared across the transition must clear caches.

### 7. Tests

- **Rust unit** (local `cargo test`; CI runs the crate via pytest parity):
  loader (synthetic parquet + sha256 sidecar via `parquet` + `sha2` into
  tempdirs, mirroring `test_fpm_forward.py::_row/_write_pair`'s 9-row v5
  fixture): sidecar gates, row gates, duplicate key, cells/domains; op:
  exact hit, in-curve lerp, own-curve-fallback transfer, domain gate, baseline
  kv-floor, cell-selection error ordering; engine: FPM mixed/genonly formulas,
  MTP rejection; perf_interp: `own_curve_coverage_fallback` true/false pair;
  SOL families: Python-oracle goldens at 1e-9 (generating command in doc
  comment, crate convention).
- **Live parity** (CI): extend `test_compile_engine_parity.py` (FpmForward
  op-transfer fidelity incl. recursive sol_ops) and
  `test_engine_step_parity.py` (FPM-mode cases over a synthetic dataset in a
  temp systems root — new pattern there; static/mixed/agg/disagg surfaces,
  `PARITY_RTOL = 0.01`, error-symmetry).
- **Real-DB grid parity** (acceptance run, this campaign): script sweeping
  every collected grid point of `database_merged` (20 topologies, 65,461 rows)
  through both engines — expectation bit-exact on exact hits — plus randomized
  in-domain off-grid probes; report max deviation.

## Consciously mirrored invariants

Exact-hit returns the leaf verbatim; bracket weights in plain coordinates;
log2 site distance with the `max(v,1)` site / `max(v,1e-12)` query asymmetry
(the ~40-log2-unit separation isolates P=0 prefill sites — load-bearing);
IDW weight `1/(d²+1e-12)`; `k_tail=3` median (average-of-two on even counts,
`total_cmp` ordering); miss-don't-fabricate on non-positive SOL; first-wins
duplicate handling is N/A here (duplicates are hard errors); integer coord
keys are u32 (collector totals cap ≈ 4.3e9 — loud overflow error via
`PerfRow::u32` is the desired failure mode); `int()` truncation of float
kwargs (numpy `ceil` outputs) mirrored via explicit casts at the runtime
boundary.

## Error mapping

Python distinguishes ValueError (structural data bugs, invalid args) from
PerfDataNotAvailableError (not collected / out of domain / ambiguous cell).
Rust collapses to `AicError::PerfDatabase` → `PyValueError`, which the parity
error-symmetry contract accepts. Two guardrails: (a) FPM ops are only ever
produced by the model rewrite as the sole op per list — never inside a
`FallbackOp` — so the PerfDatabase-swallowing fallback path is unreachable;
assert this in `Engine::build`. (b) Error messages keep Python's wording
(domain-gate message includes axis/value/bounds) for debuggability.

## Out of scope

Collector changes (none — criterion 1), encoder/multimodal FPM, MTP support,
beam search, AFD mode (CLI-rejected), EMPIRICAL/HYBRID database modes for the
Rust step (pre-existing SILICON-only routing), the `src/fpm/`
(ForwardPassPerfModel) module beyond its unchanged consumption of
`Engine::forward_pass_time_ms`.

## Sequencing

1. `perf_interp` flag + tests (smallest shared-engine risk first)
2. loader + unit tests
3. op + SOL mode + enum/spec wiring + schema bump + Python producer (one
   commit — the wire changes atomically)
4. runtime FPM branches + Python gate lift
5. parity suites + real-DB grid run + report
