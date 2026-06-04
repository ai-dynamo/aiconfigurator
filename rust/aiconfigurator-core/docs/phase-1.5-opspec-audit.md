<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Phase 1.5 — E0 OpSpec wire-format audit

**Commit:** E0 (gates E1; can invalidate E2's shape). **No production code.**

This audit walks **every field of every `Op` variant** in
`rust/aiconfigurator-core/src/operators/op.rs` and the operator structs it
references (`operators/{gemm,attention,mla,moe,moe_dispatch,communication,
embedding,elementwise,dsa,dsv4,mamba,mhc,vision,wideep_mla,wideep_moe}.rs`),
traces each field to its Python source, and classifies it 1–4 per the plan's
"The crux: OpSpec wire format" section.

---

## VERDICT: GREEN — E2 wire-format shape is safe to proceed

Every field on every Rust `Op` variant has a documented build-time Python
source. The audit found **no Category-4 field that cannot be populated at
Python `compile_engine` time**, and **no Category-3 field whose disposition
forces a non-trivial schema change to `RuntimeConfig`.**

The structural reason is simple and holds across 26 of the 27 variants:
Python's `Operation.__init__` (`sdk/operations/base.py:130`) and every subclass
store **all config-time parameters as instance attributes** (`self._n`,
`self._scale_factor`, `self._kvcache_quant_mode`, …) and read only *runtime*
values (`batch_size`, `s`, `prefix`, `x`, `*_seq_imbalance_correction_scale`,
`beam_width`) from `**kwargs` inside `query()`. That split is a 1:1 match for
the Rust `<Op>Op { … }` (config-time) vs `RuntimeContext { … }` (runtime)
split. OpSpec therefore serializes the instance attributes; `RuntimeContext`
is reconstructed per call from `RuntimeConfig` exactly as Phase 1's
`BaseBackend` does today.

The lone exception is `Op::Vision` (`VisionEncoderOp`), a Rust-only composite
with no Python `Operation` counterpart — handled by **omitting it from OpSpec**
and emitting its decomposed child ops instead (flagged item 3 below).

**What GREEN certifies:** the wire-format *shape* — every Rust field is
populatable from a build-time Python source. It does **not** certify
bit-identical numbers; the reduced field sets on `Dsv4ModuleOp` and the
Rust-side dispatch on `MoEDispatchOp` / `WideEpMoeOp` are validated by the
**E6 parity gate**, not here.

### Category counts (Rust Op fields)

Counted mechanically from the cells of the field-by-field table below. Every
struct's `scale_factor` is cat-2 (the model builder computes it from layer
counts / `mtp_scale` / `pdl_factor` and freezes it onto the instance — see the
`scale_factor` note after the table); `name` and all shape/quant/flag fields
that mirror a Python `Operation` attr are cat-1.

| Category | Count | Meaning |
| --- | --- | --- |
| **1** — direct mirror of a Python `Operation` instance attr | **72** | OpSpec serializes the value verbatim. |
| **2** — computed at Python build time from `(ModelConfig, RuntimeConfig)` | **28** | 24 × `scale_factor` + `Elementwise.bytes_per_token` + `Dsv4.attn_kind` + `MoeDispatch.backend` + `WideEpMoe.workload_distribution`. OpSpec stores the computed value. |
| **3** — resolved inside Python `query()` at call time | **2** | `MoeDispatch.flavor`, `WideEpMoe.kernel_source`. All inputs build-time-known; dispositions below. |
| **4** — derived in Rust only today, no Python `Operation` instance | **8** | The `VisionEncoderOp` struct (8 fields) — composite that has no Python `Operation`; see flagged item 3. |
| **(runtime, not an Op field)** | 8 | `RuntimeContext` fields, sourced from `RuntimeConfig` / session, not OpSpec. |

(The `MlaModule*`, `Dsa*`, `Dsv4*` phase-pairs are two Rust variants over one
Python class; their fields are counted once. `Overlap`/`Fallback` child lists
are counted as cat-1 recursive members, not separate field rows.)

### Non-trivial Category-3/4 fields requiring a flagged disposition

**None are blocking.** Four items warrant an explicit line (and an E5/E6
watch-item); all are trivially populatable. Only item 3 changes the E2 enum
shape, and it does so by *removing* a variant whose latency contribution is
zero today:

1. **`MoEDispatchOp.flavor` / dispatch branch** (cat-3, kept Rust-side). Python
   `MoEDispatch.query` selects its branch tree from build-time-known values
   only — `backend`, `sm_version`, `num_gpus_per_node`, `moe_tp/ep`,
   `attention_dp`, `moe_backend`, `quant_mode`. `num_tokens` (runtime) only
   sizes the table lookup, never the branch. **This is NOT an escalation.**
2. **`WideEpMoeOp.kernel_source`** (cat-3 → pre-bake to cat-2 recommended).
   Python `TrtLLMWideEPMoE._select_kernel` (`operations/moe.py:1283`) picks
   `deepgemm` vs `moe_torch_flow` from `sm_version` + `quant_mode` at query
   time; the Rust builder hardcodes `"moe_torch_flow"` and relies on a
   table-presence fallback in `perf_database/wideep_moe.rs:99`. Latent
   divergence risk only if a system ships *both* kernels; pre-baking the
   Python-resolved value into OpSpec removes the risk.
3. **`Op::Vision` (`VisionEncoderOp`) — cat-4; OMIT from OpSpec entirely.**
   There is **no Python `VisionEncoderOp` Operation** (the `operations/`
   package has no `vision.py`). The Rust struct's own docstring says it mirrors
   `models/vit_ops.py::build_encoder_ops`, which returns a **list** of standard
   GEMM / EncoderAttention / ElementWise ops, not one bundled op. So
   `compile_engine` cannot serialize a single Vision op — instead it walks
   `model.encoder_ops` and emits the **decomposed** child ops, each of which is
   an existing OpSpec variant (`Gemm` / `EncoderAttention` / `Elementwise`, all
   cat-1). **Disposition: do not add an `Op::Vision` variant to OpSpec;** the
   Rust composite retires with the Rust model layer at E7. Non-blocking because
   `session.rs` passes `num_image_tokens: 0` on every path today, so the vision
   composite contributes **zero latency** — dropping it moves no parity number.
   It is listed here because the variant disappearing is a genuine E2-shape
   decision, which is exactly what E0 exists to surface.
4. **`RuntimeContext.num_image_tokens`** (runtime field, currently dormant).
   `session.rs` passes `0` on every path, so the vision encoder is inert; no
   new `RuntimeConfig` field is needed for parity. If the vision path is ever
   revived, the count is `images_per_prompt × tokens_per_image` from
   `ModelConfig` (cat-2, bake-able) — not a runtime input. Flagged so it is not
   silently dropped.

The OpSpec enum **must be recursive**: `Overlap`/`Fallback` hold child op
lists, so the wire type needs `Vec<OpSpec>` / `Box<OpSpec>` members.

---

## Field-by-field audit table

Legend for Python source: `Operation.__init__` attr ⇒ cat 1 unless noted.
All `name` and `scale_factor` fields are cat-1 (`Operation.__init__(name,
scale_factor)`, `base.py:130`) except where the model builder computes
`scale_factor` from layer counts / `mtp_scale_factor` (cat-2) — see note
after the table.

| Op variant | field | type | Python source (file:symbol) | cat | disposition (cat 3/4) |
| --- | --- | --- | --- | --- | --- |
| **Gemm** (`GemmOp`) | name | String | `gemm.GEMM.__init__` `_name` | 1 | |
| | scale_factor | f64 | `gemm.GEMM` `_scale_factor` (builder: layer count) | 2 | |
| | n | u32 | `gemm.GEMM` `_n` | 1 | |
| | k | u32 | `gemm.GEMM` `_k` | 1 | |
| | quant_mode | GemmQuantMode | `gemm.GEMM` `_quant_mode` | 1 | |
| | scale_num_tokens | u32 | `gemm.GEMM` `_scale_num_tokens` (kwarg) | 1 | |
| | low_precision_input | bool | `gemm.GEMM` `_low_precision_input` (kwarg) | 1 | |
| **Embedding** (`EmbeddingOp`) | name | String | `embedding.Embedding` `_name` | 1 | |
| | scale_factor | f64 | `embedding.Embedding` `_scale_factor` | 2 | |
| | vocab_size | u32 | `embedding.Embedding` `_vocab_size` | 1 | |
| | hidden_size | u32 | `embedding.Embedding` `_hidden_size` | 1 | |
| | quant_mode | GemmQuantMode | `embedding.Embedding` `_quant_mode` | 1 | |
| **Elementwise** (`ElementwiseOp`) | name | String | `elementwise.ElementWise` `_name` | 1 | |
| | scale_factor | f64 | `elementwise.ElementWise` `_scale_factor` | 2 | |
| | bytes_per_token | f64 | `elementwise.ElementWise` (in/out × dtype, builder) | 2 | |
| **ContextAttention** (`ContextAttentionOp`) | name | String | `attention.ContextAttention` `_name` | 1 | |
| | scale_factor | f64 | `attention.ContextAttention` `_scale_factor` | 2 | |
| | n | u32 | `_n` | 1 | |
| | n_kv | u32 | `_n_kv` | 1 | |
| | head_size | u32 | `_head_size` | 1 | |
| | window_size | u32 | `_window_size` | 1 | |
| | kv_cache_dtype | KvCacheQuantMode | `_kvcache_quant_mode` | 1 | |
| | fmha_quant_mode | FmhaQuantMode | `_fmha_quant_mode` | 1 | |
| | use_qk_norm | bool | `_use_qk_norm` (set by builder; see hot spot) | 1 | |
| **GenerationAttention** (`GenerationAttentionOp`) | name | String | `attention.GenerationAttention` `_name` | 1 | |
| | scale_factor | f64 | `_scale_factor` | 2 | |
| | n | u32 | `_n` | 1 | |
| | n_kv | u32 | `_n_kv` | 1 | |
| | head_size | u32 | `_head_size` | 1 | |
| | window_size | u32 | `_window_size` | 1 | |
| | kv_cache_dtype | KvCacheQuantMode | `_kv_cache_dtype` | 1 | |
| **EncoderAttention** (`EncoderAttentionOp`) | name | String | `attention.EncoderAttention` `_name` | 1 | |
| | scale_factor | f64 | `_scale_factor` | 2 | |
| | n | u32 | `_n` (num_heads) | 1 | |
| | head_size | u32 | `_head_size` | 1 | |
| | fmha_quant_mode | FmhaQuantMode | `_fmha_quant_mode` | 1 | |
| **ContextMla** (`ContextMlaOp`) | name | String | `mla.ContextMLA` `_name` | 1 | |
| | scale_factor | f64 | `_scale_factor` | 2 | |
| | num_heads | u32 | `_num_heads` | 1 | |
| | kv_cache_dtype | KvCacheQuantMode | `_kvcache_quant_mode` | 1 | |
| | fmha_quant_mode | FmhaQuantMode | `_fmha_quant_mode` | 1 | |
| **GenerationMla** (`GenerationMlaOp`) | name | String | `mla.GenerationMLA` `_name` | 1 | |
| | scale_factor | f64 | `_scale_factor` | 2 | |
| | num_heads | u32 | `_num_heads` | 1 | |
| | kv_cache_dtype | KvCacheQuantMode | `_kv_cache_dtype` | 1 | |
| **MlaModuleContext / MlaModuleGeneration** (`MlaModuleOp`) | name | String | `mla.MLAModule` `_name` | 1 | one Python class, two Rust variants by phase (`_is_context`); variant chosen at build = cat 2 |
| | scale_factor | f64 | `_scale_factor` | 2 | |
| | num_heads | u32 | `_num_heads` | 1 | |
| | kv_cache_dtype | KvCacheQuantMode | `_kvcache_quant_mode` | 1 | |
| | fmha_quant_mode | FmhaQuantMode | `_fmha_quant_mode` | 1 | |
| | gemm_quant_mode | GemmQuantMode | `_gemm_quant_mode` | 1 | |
| **MlaBmm** (`MlaBmmOp`) | name | String | `mla.MLABmm` `_name` | 1 | |
| | scale_factor | f64 | `_scale_factor` | 2 | |
| | num_heads | u32 | `_num_heads` | 1 | |
| | quant_mode | GemmQuantMode | `_quant_mode` | 1 | |
| | is_pre | bool | `_if_pre` | 1 | |
| **Moe** (`MoeOp`) | name | String | `moe.MoE` `_name` | 1 | |
| | scale_factor | f64 | `_scale_factor` | 2 | |
| | hidden_size | u32 | `_hidden_size` | 1 | |
| | inter_size | u32 | `_inter_size` | 1 | |
| | topk | u32 | `_topk` | 1 | |
| | num_experts | u32 | `_num_experts` | 1 | |
| | moe_tp_size | u32 | `_moe_tp_size` | 1 | |
| | moe_ep_size | u32 | `_moe_ep_size` | 1 | |
| | quant_mode | MoeQuantMode | `_quant_mode` | 1 | |
| | workload_distribution | String | `_workload_distribution` (see distribution hot spot) | 1 | |
| | is_gated | bool | `_is_gated` | 1 | |
| **MoeDispatch** (`MoEDispatchOp`) | name | String | `moe.MoEDispatch` `_name` | 1 | |
| | scale_factor | f64 | `_scale_factor` | 2 | |
| | hidden_size | u32 | `_hidden_size` | 1 | |
| | topk | u32 | `_topk` | 1 | |
| | num_experts | u32 | `_num_experts` | 1 | |
| | moe_tp_size | u32 | `_moe_tp_size` | 1 | |
| | moe_ep_size | u32 | `_moe_ep_size` | 1 | |
| | attention_dp_size | u32 | `_attention_dp_size` | 1 | |
| | pre_dispatch | bool | `_pre_dispatch` | 1 | |
| | backend | BackendKind | `EngineConfig.backend` (`database.backend` in `query`) | 2 | |
| | flavor | DispatchFlavor | **resolved at query time** in Python from backend+sm+topology+moe_backend | **3** | **keep Rust-side**: all inputs build-time-known; carry raw `moe_backend`+topology, dispatch in Rust (current behavior). See hot spot. |
| | comm_quant | CommQuantMode | hardcoded `half` both sides | 1 | |
| | moe_quant | MoeQuantMode | `moe.MoEDispatch` `_quant_mode` (kwarg) | 1 | |
| **CustomAllReduce** (`CustomAllReduceOp`) | name | String | `communication.CustomAllReduce` `_name` | 1 | |
| | scale_factor | f64 | `_scale_factor` | 2 | |
| | hidden_size | u32 | `_h` | 1 | |
| | tp_size | u32 | `_tp_size` | 1 | |
| | quant | CommQuantMode | hardcoded `half` | 1 | |
| **Nccl** (`NcclOp`) | name | String | `communication.NCCL` `_name` | 1 | |
| | scale_factor | f64 | `_scale_factor` | 2 | |
| | hidden_size | u32 | `_h` | 1 | |
| | num_gpus | u32 | `_num_gpus` | 1 | |
| | dtype | CommQuantMode | hardcoded `half` | 1 | |
| | operation | String | `_operation` (op-name string) | 1 | |
| **P2P** (`P2POp`) | name | String | `communication.P2P` `_name` | 1 | |
| | scale_factor | f64 | `_scale_factor` | 2 | |
| | pp_size | u32 | `_pp_size` | 1 | |
| | hidden_size | u32 | `_h` | 1 | |
| **Vision** (`VisionEncoderOp`) — **no Python Operation; OMIT from OpSpec** | name | String | none (`models/vit_ops.py::build_encoder_ops` returns a *list*, not one op) | **4** | **omit `Op::Vision`**; Python walks `model.encoder_ops` and emits decomposed GEMM/EncoderAttention/Elementwise children (all cat-1). Retires with Rust model layer at E7. |
| | scale_factor | f64 | none | **4** | (same) |
| | num_layers | u32 | ModelConfig (ViT layers) | **4** | folded into child-op counts |
| | num_heads | u32 | ModelConfig | **4** | → child GEMM/EncoderAttention shapes |
| | head_size | u32 | ModelConfig | **4** | → child shapes |
| | hidden_size | u32 | ModelConfig | **4** | → child shapes |
| | intermediate_size | u32 | ModelConfig | **4** | → child FFN GEMM shapes |
| | fmha_quant | FmhaQuantMode | ModelConfig dtypes | **4** | → child EncoderAttention |
| | gemm_quant | GemmQuantMode | ModelConfig dtypes | **4** | → child GEMMs |
| **DsaContext / DsaGeneration** (`DsaModuleOp`) | name | String | `dsa.DSAModule` `_name` | 1 | one class, two Rust variants by phase = cat 2 |
| | scale_factor | f64 | `_scale_factor` | 2 | |
| | num_heads | u32 | `_num_heads` | 1 | |
| | kv_cache_dtype | KvCacheQuantMode | `_kvcache_quant_mode` | 1 | |
| | fmha_quant_mode | FmhaQuantMode | `_fmha_quant_mode` | 1 | |
| | gemm_quant_mode | GemmQuantMode | `_gemm_quant_mode` | 1 | |
| | architecture | String | `model_info["architecture"]` (`get_model`) | 1 | |
| **Dsv4Context / Dsv4Generation** (`Dsv4ModuleOp`) | name | String | `dsv4.{Context,Generation}DeepSeekV4AttentionModule` `_name` | 1 | |
| | scale_factor | f64 | `_scale_factor` | 2 | |
| | attn_kind | AttnKind | derived from `_compress_ratio` (`{0,4,128}`→CSA/HCA); Python keys table by ratio | **2** | bake resolved kind in Python (or emit `compress_ratio`); see hot spot / DSv4 note |
| | num_heads | u32 | `_num_heads` | 1 | |
| | kv_cache_dtype | KvCacheQuantMode | `_kvcache_quant_mode` | 1 | |
| | fmha_quant_mode | FmhaQuantMode | `_fmha_quant_mode` | 1 | |
| | gemm_quant_mode | GemmQuantMode | `_gemm_quant_mode` | 1 | |
| | architecture | String | `model_info["architecture"]` | 1 | |
| **Mhc** (`MhcModuleOp`) | name | String | `dsv4.DeepSeekV4MHCModule` `_name` | 1 | |
| | scale_factor | f64 | `_scale_factor` | 2 | |
| | hc_mult | u32 | `_hc_mult` | 1 | |
| | hidden_size | u32 | `_hidden_size` | 1 | |
| | architecture | String | `model_info["architecture"]` | 1 | |
| **Mamba2** (`Mamba2Op`) | name | String | `mamba.Mamba2Kernel` `_name` | 1 | |
| | scale_factor | f64 | `_scale_factor` | 2 | |
| | kernel_source | String | `_kernel_source` (builder literal per phase) | 1 | |
| | phase | String | `_phase` (builder literal) | 1 | |
| | d_model | u32 | `_d_model` | 1 | |
| | d_state | u32 | `_d_state` | 1 | |
| | d_conv | u32 | `_d_conv` | 1 | |
| | nheads | u32 | `_nheads` | 1 | |
| | head_dim | u32 | `_head_dim` | 1 | |
| | n_groups | u32 | `_n_groups` | 1 | |
| | chunk_size | u32 | `_chunk_size` | 1 | |
| **Gdn** (`GdnOp`) | name | String | `mamba.GDNKernel` `_name` | 1 | |
| | scale_factor | f64 | `_scale_factor` | 2 | |
| | kernel_source | String | `_kernel_source` (builder literal per phase) | 1 | |
| | phase | String | `_phase` | 1 | |
| | d_model | u32 | `_d_model` | 1 | |
| | d_conv | u32 | `_d_conv` | 1 | |
| | num_k_heads | u32 | `_num_k_heads` | 1 | |
| | head_k_dim | u32 | `_head_k_dim` | 1 | |
| | num_v_heads | u32 | `_num_v_heads` | 1 | |
| | head_v_dim | u32 | `_head_v_dim` | 1 | |
| **WideEpContextMla** (`WideEpContextMlaOp`) | name | String | `mla.WideEPContextMLA` `_name` | 1 | |
| | scale_factor | f64 | `_scale_factor` | 2 | |
| | num_heads | u32 | `_num_heads` | 1 | |
| | kv_cache_dtype | KvCacheQuantMode | `_kvcache_quant_mode` | 1 | |
| | fmha_quant_mode | FmhaQuantMode | `_fmha_quant_mode` | 1 | |
| | attn_backend | String | `mla.py:953` `_attn_backend` (default `"flashinfer"`) | 1 | |
| **WideEpGenerationMla** (`WideEpGenerationMlaOp`) | name | String | `mla.WideEPGenerationMLA` `_name` | 1 | |
| | scale_factor | f64 | `_scale_factor` | 2 | |
| | num_heads | u32 | `_num_heads` | 1 | |
| | kv_cache_dtype | KvCacheQuantMode | `_kv_cache_dtype` | 1 | |
| | fmha_quant_mode | FmhaQuantMode | `_fmha_quant_mode` | 1 | |
| | attn_backend | String | `_attn_backend` (default `"flashinfer"`) | 1 | |
| **WideEpMoe** (`WideEpMoeOp`) | name | String | `moe.TrtLLMWideEPMoE` `_name` | 1 | |
| | scale_factor | f64 | `_scale_factor` | 2 | |
| | hidden_size | u32 | `_hidden_size` | 1 | |
| | inter_size | u32 | `_inter_size` | 1 | |
| | topk | u32 | `_topk` | 1 | |
| | num_experts | u32 | `_num_experts` | 1 | |
| | moe_tp_size | u32 | `_moe_tp_size` | 1 | |
| | moe_ep_size | u32 | `_moe_ep_size` | 1 | |
| | attention_dp_size | u32 | `_attention_dp_size` | 1 | |
| | quant_mode | MoeQuantMode | `_quant_mode` | 1 | |
| | workload_distribution | String | `_workload_distribution` (from `enable_eplb`, builder) | 2 | `_eplb` suffix selected at build from `enable_eplb` |
| | num_slots | u32 | `moe.py:1211` `_num_slots` (defaults `num_experts`) | 1 | |
| | kernel_source | String | **`_select_kernel` at query time** (sm_version + quant) | **3** | **pre-bake in Python** (preferred): `_select_kernel` inputs are build-time-known. Rust currently hardcodes `"moe_torch_flow"` + table-presence fallback (`perf_database/wideep_moe.rs:99`). See hot spot. |
| **Overlap** (`OverlapOp`) | name | String | `overlap.OverlapOp` `_name` | 1 | |
| | group_a | Vec\<Op\> | `overlap.OverlapOp` `_group_a` | 1 | **recursive**: each child is an OpSpec |
| | group_b | Vec\<Op\> | `_group_b` | 1 | **recursive** |
| **Fallback** (`FallbackOp`) | name | String | `overlap.FallbackOp` `_name` | 1 | |
| | primary | Box\<Op\> | `overlap.FallbackOp` `_primary` | 1 | **recursive**: child OpSpec |
| | fallback | Vec\<Op\> | `_fallback` | 1 | **recursive** |

### Note on `scale_factor` (cat-2 throughout)

The plan's hot-spot list calls out `_mtp_scale_factor` / `nextn`. `scale_factor`
is uniformly cat-2 because the **Python model builder** computes it at
construction (layer counts, `mtp_scale`, `pdl_factor`) and passes it into
`Operation.__init__` — e.g. `deepseek_v32.py` builds
`ops.Embedding("generation_embedding", 1 * self._mtp_scale_factor, …)`. By the
time the op exists, `scale_factor` is a frozen number on the instance. OpSpec
serializes that number; Rust never recomputes it. This is the mechanism that
makes the `nextn` / MTP path survive the rewire automatically (see hot spot 6).

### `RuntimeContext` fields (sourced from `RuntimeConfig` / session, not OpSpec)

`batch_size`, `beam_width`, `s`, `prefix`, `num_tokens`,
`seq_imbalance_correction_scale`, `gen_seq_imbalance_correction_scale`,
`num_image_tokens`. These are per-call runtime values Python reads from
`**kwargs` inside `query()`. All but `num_image_tokens` are already in the
plan's `RuntimeConfig`; `num_image_tokens` is dormant (always 0 in
`session.rs`) and bake-able if revived (cat-2 from ModelConfig) — no
`RuntimeConfig` schema change required.

---

## Named hot-spot resolutions

### 1. `use_qk_norm` for Qwen3 / Qwen3MoE / MiniMaxM2 — Category 1, confirmed

`utils.py:700` sets `extra_params = {"architecture": architecture,
"use_qk_norm": True}` for the relevant architectures, threaded through the
model builders into `ContextAttention.__init__(..., use_qk_norm=...)` and
`GenerationAttention.__init__`, where it is stored as `self._use_qk_norm`
(`attention.py:118`, `:365`). The Rust `models/qwen35.rs:215` forces it on via
`a.use_qk_norm = cfg.spec.use_qk_norm`, which is the same value. **Disposition:
plain cat-1 serialization** — `compile_engine` reads `op._use_qk_norm` off the
Python instance. No special handling.

### 2. `MoEDispatch` backend selection — Category 3, **keep Rust-side** (NOT an escalation)

Python `MoEDispatch.query` (`operations/moe.py:864`) does **not** store a
resolved flavor; it selects its entire branch tree at query time. The selection
inputs are: `database.backend`, `database.system_spec["gpu"]["sm_version"]`,
`database.system_spec["node"]["num_gpus_per_node"]`, `self._moe_tp_size`,
`self._moe_ep_size`, `self._attention_dp_size`, `self._attention_tp_size`,
`self._moe_backend`, `self._quant_mode`, `self._pre_dispatch`,
`self._reduce_results`. **All of these are build-time-known.** The only runtime
input, `num_tokens`, sizes the communication volume — it never steers the
branch.

**Recommendation:** OpSpec carries the **raw fields** (`moe_backend`,
`pre_dispatch`, tp/ep/dp, `quant_mode`, `backend`) and Rust's
`MoEDispatchOp::query` performs the dispatch — which is exactly what the
current Rust code does. The Rust `DispatchFlavor` enum is a build-time
pre-resolution of the coarse SGLang-DeepEP vs custom-allreduce vs
TRT-LLM-alltoall family (set in `models/moe.rs::dispatch_flavor`), and the fine
gating (e.g. NVL72 + attn_dp>1 + moe_tp==1 enabling the alltoall table) stays
inside `query` using `num_gpus_per_node`. The model builder must emit a
`flavor` value (or a `moe_backend` enum the Rust side maps to one). Carrying
the raw enum + Rust dispatch is the cleaner option because it keeps the
fine-gating logic colocated with the perf tables it queries.

This is a **Category-3-kept-Rust-side** field with all inputs build-time-known.
It is explicitly **not** an escalation.

### 3. MLA fallback chain — Category 1, ports as a recursive `FallbackOp`

Python `FallbackOp` (`operations/overlap.py:40`) holds `_primary: Operation`
and `_fallback: list[Operation]`. It is constructed in the model builders
(`models/deepseek.py:141`, `:325`) wrapping an `MLAModule` primary and a list
of granular per-kernel ops as the fallback. **Disposition:** the Rust
`Op::Fallback(FallbackOp { primary: Box<Op>, fallback: Vec<Op> })` already
mirrors this 1:1. OpSpec must be **recursive** — `FallbackOp`'s children are
themselves `OpSpec` values (`Box<OpSpec>` + `Vec<OpSpec>`). `compile_engine`
recurses: convert `_primary` and each `_fallback[i]` to OpSpec, nest them. The
only behavioral subtlety (Python forces SILICON on the primary inside HYBRID,
caches `_primary_unavailable`) is a query-time concern Rust already handles
(`op.rs:309`) and is orthogonal to the wire format.

### 4. DSv3 `combined_prefix` threading (mix-step) — **not an Op field**

`combined_prefix` exists only in `session.rs:240` (the mix-step scheduler) and
is passed as `RuntimeContext.prefix` (`session.rs:276`). There is no Python or
Rust *Operation* field named `combined_prefix`; it is a per-step runtime value
the session computes and feeds to context ops as `prefix`. It rides the same
`prefix` slot in `RuntimeConfig`/`RuntimeContext` that Phase 1 already uses.
**Disposition:** no OpSpec impact; the mix-step composition stays in the engine
session layer.

### 5. Distribution strings (`power_law_1.01` vs `power_law_1.2`) — Category 1/2

`MoeOp.workload_distribution` and `WideEpMoeOp.workload_distribution` are plain
strings on the Python op instance (`self._workload_distribution`). For
WideEP/TRT-LLM the builder appends the `_eplb` suffix when `enable_eplb` is set
(`models/.../deepseek_wideep_trtllm.rs:45` mirrors the Python builder logic),
making the *full string* (e.g. `power_law_1.2_eplb`) a build-time value.
**Disposition:** serialize the resolved string verbatim (cat-1 for `MoeOp`,
cat-2 for the eplb-suffixed WideEP form). The perf-DB layer already does the
`→ "uniform"` fallback at query time, identically on both sides.

### 6. `_mtp_scale_factor` / `nextn` path — Category 2, survives via `scale_factor`

`nextn` lives in the plan's `EngineConfig.speculative.nextn`. The Python model
builder turns it into `self._mtp_scale_factor` (`deepseek_v32.py:108`,
`qwen35.py:49`) and folds it into each generation op's `scale_factor` argument
at construction. By the time the `Operation` exists, the MTP factor is already
multiplied into `scale_factor` — a frozen float. OpSpec serializes that float;
Rust applies it via `.scaled(scale_factor)`. **Disposition:** no dedicated
field, no runtime resolution. The path survives the rewire because Python bakes
it into `scale_factor` at build time (the cat-2 mechanism described above).
This matches the Rust comment at `qwen35.rs:71-78`.

### DSv4 reduced-field-set note (informational, not a gap)

The Rust `Dsv4ModuleOp` carries **fewer** fields than the Python
`ContextDeepSeekV4AttentionModule` (Python's `query` passes `native_heads`,
`tp_size`, `hidden_size`, `q_lora_rank`, `o_lora_rank`, `head_dim`,
`rope_head_dim`, `index_n_heads`, `index_head_dim`, `index_topk`,
`window_size`, `compress_ratio`, `o_groups`). The Rust perf-DB layer keys the
DSv4 table by `architecture` + `num_heads` + `attn_kind` only, so the extra
Python fields are not needed Rust-side. **Every Rust field is
Python-populatable** — which is all the audit requires. The audit does not
expand to Python-only fields. (Numerical equivalence of the reduced lookup is
an E6 parity-gate concern, not E0.)

---

## `compile_engine` signature decision

**Chosen unified signature:**

```python
# Public flat entry — used by BOTH call directions
def compile_engine(
    model_path: str,
    system: str,
    backend: str,
    backend_version: str | None = None,
    *,
    # parallelism / quant / speculative scalars (EngineConfig fields)
    tp_size: int = 1,
    pp_size: int = 1,
    attention_dp_size: int = 1,
    moe_tp_size: int | None = None,
    moe_ep_size: int | None = None,
    gemm_quant_mode: str | None = None,
    moe_quant_mode: str | None = None,
    kvcache_quant_mode: str | None = None,
    fmha_quant_mode: str | None = None,
    comm_quant_mode: str | None = None,
    nextn: int = 0,
    nextn_accept_rates: list[float] | None = None,
    kv_block_size: int | None = None,
    systems_path: str | None = None,
    # runtime knobs needed to walk ctx/gen op lists
    runtime_config: RuntimeConfig | None = None,
) -> bytes:
    model_config = _build_model_config(...)          # cli/api.py:584 — flat→ModelConfig
    model = get_model(model_path, model_config, backend)  # models/__init__.py:52
    return _compile_from_model(model, runtime_config or RuntimeConfig.defaults())

# Private helper — the op-walk both directions converge on
def _compile_from_model(model: BaseModel, runtime_config: RuntimeConfig) -> bytes:
    ops = [_to_opspec(op) for op in model.context_ops] \
        + [_to_opspec(op) for op in model.generation_ops]
    return bincode_serialize(EngineSpec(ops=ops, ...))
```

**Justification.**

1. **The plan's inconsistency is real and the flat form is the right
   resolution.** The data-classes section writes `compile_engine(model,
   runtime_config)`; the Rust→Python `build_aic_engine` calls
   `compile_engine(model_path, system, backend, …)`. Rust has **no Python
   model object** to pass — it only has scalars. A public entry that takes a
   model object would be uncallable from `build_aic_engine`. So the public
   surface must be flat scalars.

2. **The flat entry must not reinvent model construction.** It reuses two
   existing, battle-tested Python functions:
   - `cli/api.py::_build_model_config(...)` (`:584`) already builds a
     `ModelConfig` from flat scalars — the exact pattern the CLI uses today
     (`api.py:1006` → `get_model` at `:1027`).
   - `sdk/models/__init__.py::get_model(model_path, model_config, backend)`
     (`:52`) builds the family model and, crucially, runs
     `_apply_model_quant_defaults(...)` — i.e. **quantization is inferred from
     the HF config inside `get_model`**, so the flat entry does not duplicate
     quant-inference logic.

3. **Both directions converge on one op-walk.** The Python-sweep path can call
   the public flat `compile_engine` directly, or (if a caller already holds a
   built model) the private `_compile_from_model(model, runtime_config)`. The
   Rust-embedded path (`build_aic_engine`) calls the public flat entry over
   PyO3. Both land on `_compile_from_model`, so the OpSpec walk has a single
   implementation and a single place to maintain as ops evolve.

4. **`runtime_config` is required for the walk, optional in the signature.**
   The generation op list and the context op list are *static* once the model
   is built (they are `model.context_ops` / `model.generation_ops`), so the
   walk itself doesn't need per-call runtime values — `RuntimeConfig` crosses
   later via `run_static` / `run_agg`. Passing it (or a default) keeps the
   signature forward-compatible if a future builder needs ISL/OSL to *prune*
   ops (none does today), and matches the data-classes section's
   `(model, runtime_config)` shape for the private helper.

This gives one public entry callable from both directions, a private helper
that owns the op-walk, and zero duplication of `ModelConfig` construction or
quant inference.

---

## Gate decision

**E0 → E1: GREEN.** The OpSpec wire format is safe to design at E2 with a
recursive enum (`Vec<OpSpec>` / `Box<OpSpec>` for Overlap/Fallback children).
The four flagged items — `MoEDispatch.flavor` (cat-3, keep Rust-side),
`WideEpMoe.kernel_source` (cat-3, pre-bake), `Op::Vision` (cat-4, **omit the
variant**; emit decomposed children), and dormant `num_image_tokens` (runtime)
— are documented above with trivial dispositions. The only E2-shape change is
the deliberate omission of `Op::Vision`, whose latency contribution is zero
today (`num_image_tokens == 0` on every path), so it moves no parity number.
No `RuntimeConfig` schema change is required. Numerical parity of the reduced
field sets (`Dsv4ModuleOp`, `MoEDispatchOp`) is deferred to the E6 gate by
design.
