# Head-Axis Keying Convention (#1458)

Extends the DSV4 convention (#1429/#1431, see
[deepseek-v4-sglang-attention-module-design.md](deepseek-v4-sglang-attention-module-design.md))
repo-wide: a second native head geometry landing in any table must never
silently merge with an existing one, and data cleanup (alias dedup, version
pruning) must be derivable from table structure.

## The rule

Every persisted row stores **rank-local** `num_heads`; loaders hard-error on
rows without the column (the retired `128 // tp_size` backfill guess is gone).
On top of that, three keying tiers:

1. **Local-only** — the local key is *computation-complete*: the numeric key
   columns fully determine the physical computation, so a native level would
   add no information and fragment the interpolation grid.
2. **`[native][local]`** — the module bundles model variables outside the
   numeric key (q/kv lora ranks, hidden size). Native = model identity; local
   stays the interpolation axis inside the bucket.
3. **Native-keyed by contract** — sparse-kernel calibration tables, unchanged
   from #1431.

| Table family | Keying | Native source |
|---|---|---|
| GQA attention | local-only | — (key has local q/kv heads, head_dim, window) |
| MLA kernel (`context/generation_mla`, `mla_bmm`) | local-only | — (see below) |
| WideEP MLA | local-only | — (DSV3-only by model dispatch) |
| MLA module | **`[native][local]`** | `model` column via `_MLA_MODULE_NATIVE_HEADS` |
| DSV4 modules | **`[native][local]`** (`#1431`) | `num_heads * tp_size` (genuine tp chains) |
| DSA modules | `[architecture][local]` | guardrail pins one native per arch |
| MiniMax MSA | no table (SOL + DSA xop util borrow) | — |

## Native derivation differs per family

`#1431`'s DSV4 rows carry genuine tp sweeps, so `native = num_heads * tp_size`
holds. The MLA module tables do not: they are single-GPU rank-local head
sweeps (`tp_size` hardcoded 1 by the trtllm/vllm module collectors), where the
product degenerates to the swept value. Their native identity lives only in
the `model` column, resolved through an explicit pin
(`_MLA_MODULE_NATIVE_HEADS` in `operations/mla.py`, byte-equal
`mla_module_native_heads` in Rust). Consequences:

- Landing module data for a new model requires extending the pin in both
  languages — loaders hard-error on unpinned models, and the shipped-parquet
  guardrail fails the repo otherwise. Distinguishability is enforced.
- Provenance aliases collapse: vllm 0.22.0's three names for the one
  128-native DSV3 grid land in one bucket (first source wins) — what makes
  the duplicate rows deletable in the later alias-dedup step.
- Genuine tp-chain rows are cross-checked per row: `tp_size > 1` with
  `num_heads * tp_size != native` is the #1429 stale fingerprint, load error.

Query side: the model builder passes true `native_num_heads`
(`MLAModule` / Rust `MlaModuleOp`, spec field emitted only when set);
resolution uses the #1431 ladder (exact → sole bucket → nearest ≤ →
smallest), so single-native tables behave exactly as before (Rust parity
oracles unchanged).

## Why the MLA kernel tables stay local-only

Kernel work is fully determined by local heads plus dims constant across
every shipped MLA model (`kv_lora_rank=512`, `qk_rope_head_dim=64` — DSV3,
R1, Kimi K2/K2.5/K3 all share them). The data proves it: 128-native and
64-native tp chains coexist in most kernel files and agree to ~0.35% median
at shared locals; the collectors dedupe across models by local key (Kimi case
YAML). Native-keying would actively break things: sglang 0.5.14 ships the
128-native chain at tp=1 only (DSV3 tp>1 is served by 64-chain rows at the
same local), and Kimi-K3 (#1435, 96-native) queries these tables at true
local heads with zero K3 kernel data. If a future MLA model changes
`kv_lora_rank`/`qk_rope_head_dim`, this argument breaks — those dims are part
of the pin.

**Historical 128-head convention (Kimi K2.5):** the `DeepSeekModel` builder
keys sglang/trtllm kernel queries at `128 // tp_size` regardless of true
native (`_MLA_KERNEL_TABLE_NATIVE_HEADS` in `models/deepseek.py`; collection
side pinned in the Kimi case YAML). A shape convention, not physical truth
(Kimi tp=2 runs 32 local heads, modeled with 64). Scoped to that builder —
the module path and Kimi-K3 already use true geometry. Flipping K2.5 to true
geometry is a modeling-semantics change with its own validation burden; keep
it out of keying work.

## DSA: architecture is the identity level

`architecture` already separates V3.2 (128) from the GLM-5 family (64), so no
structural change ships in #1458. The former luck is now a contract:
`test_shipped_dsa_module_tables_keep_one_native_per_architecture` pins one
native per arch per file. The moment a second native ships under one arch
(the DSV4 Flash/Pro scenario), that data PR migrates the DSA loaders to
`[native][local]` with the MLA-module recipe. GLM sweeps contain locals
beyond native (nh=128 under GLM-5) — legal grid points under a model pin.

## Kernel-collection boundaries per framework

Per-(backend, version) tables never collide, but for cross-backend
comparison: trtllm times the attention `forward()` (+ `mla_rope_generation`
in decode); sglang times the `RadixAttention` layer with per-SM kernel
dispatch recorded in `kernel_source`; vLLM has no MLA kernel tables (module
path only). Module tables bundle downscale + q_b/kv_b + attention + o_proj on
all three.

## Loader shapes after #1458

```python
data[fmha][kv][gemm][native][num_heads_local][s][b]        # context module
data[kv][gemm][native][num_heads_local][b][s_total]        # generation module
data[fmha][kv][num_heads_local][s][b]                      # kernel (unchanged)
```

Rust: `ModuleGrids`/`GenModuleGrids` gain a `BTreeMap<u32, Node>` native
level; `resolve_module_native` ↔ `_resolve_mla_module_native_key` byte-equal.
